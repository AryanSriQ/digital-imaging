#!/usr/bin/env python3

import io
import os
import logging
import time
import json
import gzip
import shutil
import tempfile
import ffmpeg
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

# Parse the environment
env=Env()
# mandatory:
region = env("AWS_REGION")
bucket = env("RECEIVE_BUCKET")
# optional, with defaults:
create_metadata = env.bool("CREATE_METADATA", False)
gzip_files = env.bool("GZIP_FILES", False)
gzip_level = env.int("GZIP_LEVEL", 5)
add_studyuid_prefix = env.bool("ADD_STUDYUID_PREFIX", False)
s3_upload_workers = env.int("S3_UPLOAD_WORKERS", 10)
scp_port = env.int("SCP_PORT", 11112)   
loglevel = env.log_level("LOG_LEVEL", "INFO")
dicom_prefix = env("DICOM_PREFIX","")
metadata_prefix = env("METADATA_PREFIX","")
cstore_delay_ms = env.int("CSTORE_DELAY_MS", 0)
boto_max_pool_connections = env.int("BOTO_MAX_POOL_CONNECTIONS", 10)
dimse_timeout = env.int("DIMSE_TIMEOUT", 30)
maximum_associations = env.int("MAXIMUM_ASSOCIATIONS", 10)
maximum_pdu_size = env.int("MAXIMUM_PDU_SIZE", 0)
network_timeout = env.int("NETWORK_TIMEOUT", 60)

if dicom_prefix: 
    dicom_prefix = dicom_prefix + '/'
if metadata_prefix:
    metadata_prefix = metadata_prefix + '/'   

# set default logging configuration
logging.basicConfig(format='%(levelname)s - %(asctime)s.%(msecs)03d %(threadName)s: %(message)s',datefmt='%H:%M:%S', level=loglevel)   
logging.info(f'Setting log level to {loglevel}')

# create a shared S3 client and use it for all threads (clients are thread-safe)
logging.info(f'Creating S3 client with max_pool_connections = {boto_max_pool_connections}.')
s3client = boto3.client('s3', region_name=region, config=botocore.client.Config(max_pool_connections=boto_max_pool_connections) )

# initialize thread pool
logging.info(f'Provisioning ThreadPool of {s3_upload_workers} S3 upload workers.')
executor = ThreadPoolExecutor(max_workers=int(s3_upload_workers))

#debug_logger()

def get_dicom_metadata(dicom_file):
    """Extracts relevant metadata from a DICOM file."""
    try:
        ds = dcmread(dicom_file)
        metadata = {
            "PatientName": str(ds.PatientName) if "PatientName" in ds else "N/A",
            "PatientID": str(ds.PatientID) if "PatientID" in ds else "N/A",
            "PatientSex": str(ds.PatientSex) if "PatientSex" in ds else "N/A",
            "PatientBirthDate": str(ds.PatientBirthDate) if "PatientBirthDate" in ds else "N/A",
            "NumberOfFrames": int(ds.NumberOfFrames) if "NumberOfFrames" in ds else 1,
            "RecommendedDisplayFrameRate": int(ds.RecommendedDisplayFrameRate) if "RecommendedDisplayFrameRate" in ds else 24,
            "SOPInstanceUID": str(ds.SOPInstanceUID) if "SOPInstanceUID" in ds else "N/A",
            "StudyInstanceUID": str(ds.StudyInstanceUID) if "StudyInstanceUID" in ds else "N/A",
        }
        return metadata
    except Exception as e:
        logging.error(f"Error reading DICOM metadata: {e}")
        return None

def convert_dicom_to_jpeg(dicom_path, output_path):
    """Converts a single-frame DICOM to JPEG."""
    try:
        ds = dcmread(dicom_path)
        img = Image.fromarray(ds.pixel_array)
        img.save(output_path, "jpeg")
        logging.info(f"Successfully converted {dicom_path} to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error converting DICOM to JPEG: {e}")
        return False

def convert_dicom_to_webm(dicom_path, output_path, frame_rate=24):
    """Converts a multi-frame DICOM to WebM by extracting frames."""
    try:
        ds = dcmread(dicom_path)
        if not hasattr(ds, 'pixel_array') or ds.pixel_array.ndim < 3:
            logging.error(f"DICOM file {dicom_path} does not contain multi-frame pixel data.")
            return False

        temp_dir = tempfile.mkdtemp()
        frame_paths = []

        for i, frame in enumerate(ds.pixel_array):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            img = Image.fromarray(frame)
            img.save(frame_path, "png")
            frame_paths.append(frame_path)

        # Use ffmpeg to combine frames into WebM
        (
            ffmpeg
            .input(os.path.join(temp_dir, "frame_%04d.png"), framerate=frame_rate)
            .output(output_path, r=frame_rate, vcodec='libvpx-vp9', crf=23, preset='veryfast', pix_fmt='yuv420p', profile='0')\
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        logging.info(f"Successfully converted {dicom_path} to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error converting DICOM to WebM: {e}")
        return False
    finally:
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def process_and_upload(dicom_path):
    """
    Processes a DICOM file, converts it, and uploads it to S3.
    """
    metadata = get_dicom_metadata(dicom_path)
    if not metadata:
        return

    sop_instance_uid = metadata.get("SOPInstanceUID", "unknown_sop")
    study_instance_uid = metadata.get("StudyInstanceUID", "unknown_study")

    if add_studyuid_prefix:
        s3_prefix = f"{dicom_prefix}{study_instance_uid}/"
    else:
        s3_prefix = dicom_prefix

    with tempfile.NamedTemporaryFile(delete=False) as temp_output:
        output_path = temp_output.name

    if metadata.get("NumberOfFrames", 1) > 1:
        output_path += ".webm"
        s3_key = f"{s3_prefix}{sop_instance_uid}.webm"
        content_type = "video/webm"
        if not convert_dicom_to_webm(dicom_path, output_path, metadata.get("RecommendedDisplayFrameRate")):
            return
    else:
        output_path += ".jpeg"
        s3_key = f"{s3_prefix}{sop_instance_uid}.jpeg"
        content_type = "image/jpeg"
        if not convert_dicom_to_jpeg(dicom_path, output_path):
            return

    s3_metadata = {key.lower(): str(value) for key, value in metadata.items()}
    
    executor.submit(s3_upload, output_path, bucket, s3_key, content_type, s3_metadata)

    # Log patient demographics and S3 URL
    patient_name = metadata.get("PatientName", "N/A")
    patient_id = metadata.get("PatientID", "N/A")
    s3_url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
    logging.info(f"Processed DICOM for Patient: {patient_name} (ID: {patient_id}), study Instance UID: {study_instance_uid}, SOP Instance UID: {sop_instance_uid}. Uploaded to S3: {s3_url}")

# Implement a handler for evt.EVT_C_STORE
def handle_store(event):
    """Handle a C-STORE request event."""
 
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as temp_dicom:
            # write raw without decoding
            # https://pydicom.github.io/pynetdicom/stable/examples/storage.html
            # https://github.com/pydicom/pynetdicom/issues/367
            # Write the preamble, prefix, file meta, encoded dataset
            temp_dicom.write(b'\x00' * 128)
            temp_dicom.write(b'DICM')
            write_file_meta_info(temp_dicom, event.file_meta)
            temp_dicom.write(event.request.DataSet.getvalue())
            
            temp_dicom_path = temp_dicom.name

        process_and_upload(temp_dicom_path)
        
        # optional c-store delay, can be used as a front-end throttling mechanism
        if cstore_delay_ms or cstore_delay_ms!=0:
            logging.info(f'Injecting C-STORE delay: {cstore_delay_ms} ms')
            time.sleep(int(cstore_delay_ms) / 1000)
            
    except BaseException as e:
        logging.error(f'Error in C-STORE processing. {e}')
        return 0xC211
    
    # return success after instance is received and written to memory
    return 0x0000
           
def s3_upload(file_path, bucket, key, content_type, metadata):    
    start_time = time.time()
    logging.debug(f'Starting s3 upload of {key}')

    try:
        with open(file_path, 'rb') as f:
            s3client.upload_fileobj(f, bucket, key, ExtraArgs={'ContentType': content_type, 'Metadata': metadata})
    except BaseException as e:
        logging.error(f'Error in S3 upload. {e}')
        return False
    finally:
        os.remove(file_path)

    elapsed_time = time.time() - start_time
    logging.info(f'Finished s3 upload of {key} in {elapsed_time} s')

    return True

def json_dumps_compact(data):
    return json.dumps(data, separators=(',',':'), sort_keys=True)
    
def main():

    logging.warning(f'Starting application.')
    logging.warning(f'Environment: {env}')
          
    # handlers = [    (evt.EVT_C_STORE, handle_store, [os.getcwd()+'/out']), (evt.EVT_CONN_OPEN , handle_open), (evt.EVT_ACCEPTED , handle_accepted), (evt.EVT_RELEASED  , handle_assoc_close ) , (evt.EVT_ABORTED  , handle_assoc_close )]       
    handlers = [(evt.EVT_C_STORE, handle_store)]

    # Initialise the Application Entity
    ae = AE()
    # overwrite AE defaults as per configuration
    ae.maximum_pdu_size = maximum_pdu_size
    ae.dimse_timeout = dimse_timeout
    ae.maximum_associations = maximum_associations
    ae.network_timeout = network_timeout
    
    # Support presentation contexts for all storage SOP Classes
    ae.supported_contexts = AllStoragePresentationContexts
    # enable verification
    ae.add_supported_context(Verification)
    # Start listening for incoming association requests
    logging.warning(f'Starting SCP Listener on port {scp_port}')
    scp = ae.start_server(("", scp_port), evt_handlers=handlers)

if __name__ == "__main__":
    main()
