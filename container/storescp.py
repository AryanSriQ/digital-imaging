#!/usr/bin/env python3

import requests
import io
import os
import logging
import time
import json
import gzip
import shutil
import tempfile
import ffmpeg
import atexit
from PIL import Image
from environs import Env
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import boto3
import botocore
from pynetdicom import AE, evt, AllStoragePresentationContexts, debug_logger
from pynetdicom.sop_class import Verification
from pydicom.filewriter import write_file_meta_info
from pydicom.filereader import dcmread
from logging_config import setup_logging

# Parse the environment first
env = Env()
# mandatory:
region = env("AWS_REGION")
bucket = env("RECEIVE_BUCKET")
folder_name = env("FOLDER_NAME")
backend_url = env("BACKEND_URL")
# optional, with defaults:
create_metadata = env.bool("CREATE_METADATA", False)
gzip_files = env.bool("GZIP_FILES", False)
gzip_level = env.int("GZIP_LEVEL", 5)
add_studyuid_prefix = env.bool("ADD_STUDYUID_PREFIX", False)
s3_upload_workers = env.int("S3_UPLOAD_WORKERS", 10)
scp_port = env.int("SCP_PORT", 11112)   
loglevel = env.log_level("LOG_LEVEL", "INFO")
dicom_prefix = env("DICOM_PREFIX", "")
metadata_prefix = env("METADATA_PREFIX", "")
cstore_delay_ms = env.int("CSTORE_DELAY_MS", 0)
boto_max_pool_connections = env.int("BOTO_MAX_POOL_CONNECTIONS", 10)
dimse_timeout = env.int("DIMSE_TIMEOUT", 30)
maximum_associations = env.int("MAXIMUM_ASSOCIATIONS", 10)
maximum_pdu_size = env.int("MAXIMUM_PDU_SIZE", 0)
network_timeout = env.int("NETWORK_TIMEOUT", 60)

# Set up structured logging FIRST - before any other logging calls
logger = setup_logging(loglevel)

# Format prefixes
if dicom_prefix:
    dicom_prefix = dicom_prefix + '/'
if metadata_prefix:
    metadata_prefix = metadata_prefix + '/'   

logger.info(f'Starting DICOM processor with log level: {loglevel}')

# Create a shared S3 client and use it for all threads (clients are thread-safe)
logger.info(f'Creating S3 client with max_pool_connections = {boto_max_pool_connections}')
try:
    s3client = boto3.client(
        's3', 
        region_name=region, 
        config=botocore.client.Config(max_pool_connections=boto_max_pool_connections)
    )
except Exception as e:
    logger.error(f'Failed to create S3 client: {e}')
    raise

# Initialize thread pool
logger.info(f'Provisioning ThreadPool of {s3_upload_workers} S3 upload workers')
executor = ThreadPoolExecutor(max_workers=int(s3_upload_workers))

# Ensure thread pool is properly shut down
def cleanup_resources():
    """Clean up resources on application exit."""
    logger.info("Shutting down thread pool...")
    executor.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")

atexit.register(cleanup_resources)

class ProcessingTimer:
    def __init__(self):
        self.start_time = time.time()
        self.timings = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def record(self, name):
        self.timings[name] = time.time() - self.start_time
        self.start_time = time.time()

def get_dicom_fps(ds):
    """Extract frame rate from DICOM metadata with improved logic."""
    try:
        fps = None

        # Check RecommendedDisplayFrameRate (0018,0040)
        if hasattr(ds, 'RecommendedDisplayFrameRate'):
            try:
                fps = float(ds.RecommendedDisplayFrameRate)
                if fps > 0:
                    logger.debug(f"Found RecommendedDisplayFrameRate: {fps} FPS")
                    return max(fps, 1)  # Ensure minimum of 1 FPS
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid RecommendedDisplayFrameRate value: {e}")

        # Check FrameTime (0018,1063) - milliseconds per frame
        if hasattr(ds, 'FrameTime'):
            try:
                frame_time = float(ds.FrameTime)  # in ms
                if frame_time > 0:
                    fps = 1000 / frame_time
                    logger.debug(f"Calculated FPS from FrameTime: {fps} FPS")
                    return max(fps, 1)  # Ensure minimum of 1 FPS
            except (ValueError, TypeError, ZeroDivisionError) as e:
                logger.warning(f"Invalid FrameTime value: {e}")

        # Check CineRate (0018,0040) - alternative field
        if hasattr(ds, 'CineRate'):
            try:
                fps = float(ds.CineRate)
                if fps > 0:
                    logger.debug(f"Found CineRate: {fps} FPS")
                    return max(fps, 1)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid CineRate value: {e}")

        # Default fallback
        logger.info("No valid FPS found in DICOM metadata, using default: 25 FPS")
        return 25  # More standard default than 50

    except Exception as e:
        logger.error(f"Error reading DICOM FPS metadata: {e}")
        return 25

def get_dicom_metadata(dicom_file):
    """Extracts relevant metadata from a DICOM file with improved error handling."""
    try:
        if not os.path.exists(dicom_file):
            logger.error(f"DICOM file does not exist: {dicom_file}")
            return None
            
        ds = dcmread(dicom_file)
        
        # Helper function to safely get DICOM attributes
        def safe_get_attr(attr_name, default="N/A"):
            try:
                if hasattr(ds, attr_name):
                    value = getattr(ds, attr_name)
                    return str(value) if value is not None else default
                return default
            except Exception as e:
                logger.warning(f"Error reading DICOM attribute {attr_name}: {e}")
                return default

        def safe_get_int_attr(attr_name, default=1):
            try:
                if hasattr(ds, attr_name):
                    value = getattr(ds, attr_name)
                    return int(value) if value is not None else default
                return default
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting DICOM attribute {attr_name} to int: {e}")
                return default

        # Extract view information safely
        view = "N/A"
        try:
            if hasattr(ds, 'ViewCodeSequence') and len(ds.ViewCodeSequence) > 0:
                view = str(ds.ViewCodeSequence[0].CodeMeaning)
        except (IndexError, AttributeError) as e:
            logger.warning(f"Error reading ViewCodeSequence: {e}")

        metadata = {
            "PatientName": safe_get_attr("PatientName"),
            "PatientID": safe_get_attr("PatientID"),
            "PatientSex": safe_get_attr("PatientSex"),
            "PatientBirthDate": safe_get_attr("PatientBirthDate"),
            "NumberOfFrames": safe_get_int_attr("NumberOfFrames", 1),
            "RecommendedDisplayFrameRate": get_dicom_fps(ds),
            "SOPInstanceUID": safe_get_attr("SOPInstanceUID"),
            "StudyInstanceUID": safe_get_attr("StudyInstanceUID"),
            "SeriesInstanceUID": safe_get_attr("SeriesInstanceUID"),
            "StageName": safe_get_attr("StageName"),
            "View": view,
            "InstanceNumber": safe_get_attr("InstanceNumber"),
            "SeriesNumber": safe_get_attr("SeriesNumber"),
        }
        
        logger.debug(f"Extracted metadata for SOPInstanceUID: {metadata.get('SOPInstanceUID')}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error reading DICOM file {dicom_file}: {e}")
        return None

def convert_dicom_to_jpeg(dicom_path, output_path):
    """Converts a single-frame DICOM to JPEG with improved error handling."""
    try:
        if not os.path.exists(dicom_path):
            logger.error(f"DICOM file does not exist: {dicom_path}")
            return False
            
        ds = dcmread(dicom_path)
        
        if not hasattr(ds, 'pixel_array'):
            logger.error(f"DICOM file {dicom_path} does not contain pixel data")
            return False
            
        # Convert pixel array to image
        pixel_array = ds.pixel_array
        
        # Handle different pixel array formats
        if pixel_array.ndim == 2:
            # Grayscale image
            img = Image.fromarray(pixel_array)
        elif pixel_array.ndim == 3 and pixel_array.shape[0] == 1:
            # Single frame from multi-frame
            img = Image.fromarray(pixel_array[0])
        else:
            logger.error(f"Unexpected pixel array dimensions: {pixel_array.shape}")
            return False
            
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.save(output_path, "JPEG", quality=85)
        logger.info(f"Successfully converted {dicom_path} to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting DICOM to JPEG: {e}")
        return False

def convert_dicom_to_webm(dicom_path, output_path, frame_rate=25):
    """Converts a multi-frame DICOM to WebM with improved error handling."""
    temp_dir = None
    try:
        if not os.path.exists(dicom_path):
            logger.error(f"DICOM file does not exist: {dicom_path}")
            return False
            
        ds = dcmread(dicom_path)
        
        if not hasattr(ds, 'pixel_array'):
            logger.error(f"DICOM file {dicom_path} does not contain pixel data")
            return False
            
        pixel_array = ds.pixel_array
        if pixel_array.ndim < 3:
            logger.error(f"DICOM file {dicom_path} does not contain multi-frame pixel data")
            return False

        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        logger.info(f"Processing {len(pixel_array)} frames for WebM conversion")
        
        for i, frame in enumerate(pixel_array):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            
            # Handle different frame formats
            if frame.ndim == 2:
                img = Image.fromarray(frame)
            else:
                logger.warning(f"Unexpected frame dimensions at index {i}: {frame.shape}")
                continue
                
            # Convert to RGB for better compatibility
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            img.save(frame_path, "PNG")
            frame_paths.append(frame_path)

        if not frame_paths:
            logger.error("No valid frames extracted from DICOM")
            return False

        # Validate frame rate
        frame_rate = max(1, min(frame_rate, 120))  # Clamp between 1-120 FPS
        
        # Use ffmpeg to combine frames into WebM
        try:
            (
                ffmpeg
                .input(os.path.join(temp_dir, "frame_%04d.png"), framerate=frame_rate)
                .output(
                    output_path, 
                    r=frame_rate, 
                    vcodec='libvpx-vp9', 
                    crf=30,  # Slightly lower quality for smaller files
                    preset='medium',  # Better compression than veryfast
                    pix_fmt='yuv420p', 
                    profile='0'
                )
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            return False
            
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error(f"WebM conversion failed - output file is missing or empty")
            return False
            
        logger.info(f"Successfully converted {dicom_path} to {output_path} ({len(frame_paths)} frames at {frame_rate} FPS)")
        return True
        
    except Exception as e:
        logger.error(f"Error converting DICOM to WebM: {e}")
        return False
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")

def process_and_upload(dicom_path, association_log):
    """
    Processes a DICOM file, converts it, and uploads it to S3, while logging statistics.
    """
    file_log = {"file_received_timestamp": time.time()}
    
    try:
        with ProcessingTimer() as timer:
            metadata = get_dicom_metadata(dicom_path)
            if not metadata:
                logger.error(f"Failed to extract metadata from {dicom_path}")
                return None

            timer.record("parse_time")
            
            sop_instance_uid = metadata.get("SOPInstanceUID", "unknown_sop")
            series_instance_uid = metadata.get("SeriesInstanceUID", "unknown_series")
            study_instance_uid = metadata.get("StudyInstanceUID", "unknown_study")

            # Validate required UIDs
            if sop_instance_uid == "unknown_sop" or study_instance_uid == "unknown_study":
                logger.error(f"Missing required DICOM UIDs in file {dicom_path}")
                return None

            if add_studyuid_prefix:
                s3_prefix = f"{dicom_prefix}{study_instance_uid}/"
            else:
                s3_prefix = dicom_prefix

            # Create temporary output file
            with tempfile.NamedTemporaryFile(delete=False) as temp_output:
                output_path = temp_output.name

            try:
                num_frames = metadata.get("NumberOfFrames", 1)
                
                if num_frames > 1:
                    output_path += ".webm"
                    s3_key = f"{folder_name}/{s3_prefix}{study_instance_uid}/{series_instance_uid}/{sop_instance_uid}.webm"
                    content_type = "video/webm"
                    fps = metadata.get("RecommendedDisplayFrameRate", 25)
                    
                    if not convert_dicom_to_webm(dicom_path, output_path, fps):
                        logger.error(f"Failed to convert multi-frame DICOM to WebM: {dicom_path}")
                        return None
                else:
                    output_path += ".jpeg"
                    s3_key = f"{folder_name}/{s3_prefix}{study_instance_uid}/{series_instance_uid}/{sop_instance_uid}.jpeg"
                    content_type = "image/jpeg"
                    
                    if not convert_dicom_to_jpeg(dicom_path, output_path):
                        logger.error(f"Failed to convert DICOM to JPEG: {dicom_path}")
                        return None
                
                timer.record("conversion_time")

                # Prepare S3 metadata (convert all values to strings and lowercase keys)
                s3_metadata = {key.lower(): str(value) for key, value in metadata.items() if value != "N/A"}
                
                # Upload to S3
                upload_log = s3_upload(output_path, bucket, s3_key, content_type, s3_metadata)
                
                if upload_log.get("upload_time", -1) == -1:
                    logger.error(f"S3 upload failed for {s3_key}")
                    return None
                
                timer.record("upload_time")

                file_log.update(timer.timings)
                file_log.update(upload_log)
                association_log.append(file_log)

                # Log patient demographics and S3 URL
                patient_name = metadata.get("PatientName", "N/A")
                patient_id = metadata.get("PatientID", "N/A")
                
                logger.info(f"Successfully processed DICOM - Patient: {patient_name} (ID: {patient_id}), "
                           f"Study: {study_instance_uid}, SOP: {sop_instance_uid}, S3: {s3_key}")

                return {
                    "patientDemographics": {
                        "PatientName": patient_name,
                        "PatientID": patient_id,
                        "PatientSex": metadata.get("PatientSex", "N/A"),
                        "PatientBirthDate": metadata.get("PatientBirthDate", "N/A"),
                        "StudyInstanceUID": study_instance_uid,
                        "SOPInstanceUID": sop_instance_uid,
                        "SeriesInstanceUID": metadata.get("SeriesInstanceUID", "N/A"),
                        "StageName": metadata.get("StageName", "N/A"),
                        "View": metadata.get("View", "N/A"),
                        "InstanceNumber": metadata.get("InstanceNumber", "N/A"),
                        "SeriesNumber": metadata.get("SeriesNumber", "N/A"),
                    },
                    "s3Url": s3_key,
                    "file_statistics": file_log
                }
                
            except Exception as e:
                logger.error(f"Error during file conversion: {e}")
                # Clean up output file if it exists
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except Exception:
                        pass
                return None
                
    except Exception as e:
        logger.error(f"Error in process_and_upload: {e}")
        return None
    finally:
        # Clean up input DICOM file
        if os.path.exists(dicom_path):
            try:
                os.remove(dicom_path)
            except Exception as e:
                logger.warning(f"Failed to clean up DICOM file {dicom_path}: {e}")

def handle_store(event):
    """Handle a C-STORE request event with improved error handling."""
    association_log = []
    temp_dicom_path = None
    
    try:
        # Create temporary DICOM file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_dicom:
            # Write DICOM preamble and prefix
            temp_dicom.write(b'\\x00' * 128)
            temp_dicom.write(b'DICM')
            write_file_meta_info(temp_dicom, event.file_meta)
            temp_dicom.write(event.request.DataSet.getvalue())
            temp_dicom_path = temp_dicom.name

        logger.debug(f"Created temporary DICOM file: {temp_dicom_path}")

        # Process and upload the file
        api_payload = process_and_upload(temp_dicom_path, association_log)
        
        if not api_payload:
            logger.error("Failed to process DICOM file")
            return 0xC211  # DICOM failure status
        
        # Apply C-STORE delay if configured
        if cstore_delay_ms > 0:
            logger.debug(f'Applying C-STORE delay: {cstore_delay_ms} ms')
            time.sleep(cstore_delay_ms / 1000)

        # Create association summary
        if association_log:
            total_processing_time = sum(
                log.get('parse_time', 0) + log.get('conversion_time', 0) + log.get('upload_time', 0) 
                for log in association_log
            )
            total_uploaded_size = sum(log.get('uploaded_file_size', 0) for log in association_log)
            
            association_summary = {
                "total_files_processed": len(association_log),
                "total_processing_time": total_processing_time,
                "total_uploaded_size": total_uploaded_size,
                "files": association_log
            }
            
            api_payload["association_summary"] = association_summary

        # Send data to backend API
        api_url = f"{backend_url}/common/uploadDicomFileFromECS"
        try:
            response = requests.post(
                api_url, 
                json=api_payload, 
                timeout=30,  # Add timeout
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info(f"Successfully sent data to API. Status: {response.status_code}")
            logger.debug(f"API Response: {response.text}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending data to API {api_url}: {e}")
            # Don't return error code for API failures - DICOM transfer was successful

    except Exception as e:
        logger.error(f'Error in C-STORE processing: {e}')
        return 0xC211  # DICOM processing failure
    finally:
        # Clean up temporary DICOM file if it still exists
        if temp_dicom_path and os.path.exists(temp_dicom_path):
            try:
                os.remove(temp_dicom_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary DICOM file {temp_dicom_path}: {e}")
    
    return 0x0000  # Success

def s3_upload(file_path, bucket, key, content_type, metadata):
    """Upload file to S3 with improved error handling and logging."""
    start_time = time.time()
    logger.debug(f'Starting S3 upload of {key}')
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File to upload does not exist: {file_path}")
            return {"upload_time": -1, "uploaded_file_size": -1}
            
        file_size = os.path.getsize(file_path)
        
        if file_size == 0:
            logger.error(f"File to upload is empty: {file_path}")
            return {"upload_time": -1, "uploaded_file_size": -1}
        
        # Validate metadata values (S3 metadata must be strings)
        validated_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, str) and len(v) <= 2048:  # S3 metadata value limit
                validated_metadata[k] = v
            else:
                logger.warning(f"Skipping invalid metadata key {k}: {v}")
        
        with open(file_path, 'rb') as f:
            s3client.upload_fileobj(
                f, 
                bucket, 
                key, 
                ExtraArgs={
                    'ContentType': content_type, 
                    'Metadata': validated_metadata
                }
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f'Successfully uploaded {key} to S3 ({file_size} bytes in {elapsed_time:.2f}s)')
        
        return {"upload_time": elapsed_time, "uploaded_file_size": file_size}
        
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f'S3 ClientError uploading {key}: {error_code} - {e}')
        return {"upload_time": -1, "uploaded_file_size": -1}
    except Exception as e:
        logger.error(f'Unexpected error uploading {key} to S3: {e}')
        return {"upload_time": -1, "uploaded_file_size": -1}
    finally:
        # Always clean up the file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove file {file_path}: {e}")

def json_dumps_compact(data):
    """Create compact JSON representation."""
    return json.dumps(data, separators=(',', ':'), sort_keys=True)

def validate_environment():
    """Validate required environment variables and configuration."""
    required_vars = ["AWS_REGION", "RECEIVE_BUCKET", "FOLDER_NAME", "BACKEND_URL"]
    missing_vars = []
    
    for var in required_vars:
        if not env(var, None):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    # Test S3 connectivity
    try:
        s3client.head_bucket(Bucket=bucket)
        logger.info(f"Successfully connected to S3 bucket: {bucket}")
    except Exception as e:
        logger.error(f"Cannot access S3 bucket {bucket}: {e}")
        return False
    
    # Test backend connectivity
    try:
        test_url = f"{backend_url}/health"  # Assuming a health endpoint exists
        response = requests.get(test_url, timeout=5)
        logger.info(f"Backend connectivity test: {response.status_code}")
    except Exception as e:
        logger.warning(f"Cannot connect to backend {backend_url}: {e}")
        # Don't fail on backend connectivity - it might not have a health endpoint
    
    return True
    
def main():
    """Main application entry point."""
    try:
        logger.warning('Starting DICOM SCP application')
        logger.info(f'Configuration: Region={region}, Bucket={bucket}, Port={scp_port}')
        
        # Validate environment and connectivity
        if not validate_environment():
            logger.error("Environment validation failed. Exiting.")
            return 1
            
        # Set up event handlers
        handlers = [(evt.EVT_C_STORE, handle_store)]

        # Initialize the Application Entity
        ae = AE()
        
        # Configure AE parameters
        if maximum_pdu_size > 0:
            ae.maximum_pdu_size = maximum_pdu_size
        ae.dimse_timeout = dimse_timeout
        ae.maximum_associations = maximum_associations
        ae.network_timeout = network_timeout
        
        ae.supported_contexts = AllStoragePresentationContexts
        ae.add_supported_context(Verification)
        
        logger.warning(f'Starting SCP listener on port {scp_port}')
        logger.info(f'Server configuration: max_associations={maximum_associations}, '
                   f'dimse_timeout={dimse_timeout}s, network_timeout={network_timeout}s')
        
        # Start the SCP server
        scp = ae.start_server(('', scp_port), evt_handlers=handlers)
        
        # Keep the server running
        logger.info("DICOM SCP server is running. Press Ctrl+C to stop.")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        return 0
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        return 1
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)