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
import threading
import asyncio
import aiohttp
from queue import Queue, Full, Empty
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict, Any
from PIL import Image
from environs import Env
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import weakref
import numpy as np
import multiprocessing

import boto3
import botocore
from pynetdicom import AE, evt, AllStoragePresentationContexts, debug_logger, build_context, register_uid
from pynetdicom.service_class import StorageServiceClass

# Register the private SOP Class
# private_uid = '1.3.46.670589.2.5.1.1'
# register_uid(private_uid, 'Private3DPresentationState', StorageServiceClass)

from pynetdicom.sop_class import (
    Verification,
)
from pydicom.filewriter import write_file_meta_info
from pydicom.filereader import dcmread
from pydicom.uid import (
    UID,
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian
)

# Transfer syntaxes
ALL_TRANSFER_SYNTAXES = [
    '1.2.840.10008.1.2',      # Implicit VR Little Endian
    '1.2.840.10008.1.2.1',    # Explicit VR Little Endian
    '1.2.840.10008.1.2.2',    # Explicit VR Big Endian
    '1.2.840.10008.1.2.5',    # RLE Lossless
    '1.2.840.10008.1.2.4.50', # JPEG Baseline (Process 1)
    '1.2.840.10008.1.2.4.57', # JPEG Lossless, Non-Hierarchical
    '1.2.840.10008.1.2.4.70', # JPEG Lossless, SV1
    '1.2.840.10008.1.2.4.80', # JPEG-LS Lossless
    '1.2.840.10008.1.2.4.81', # JPEG-LS Near-Lossless
    '1.2.840.10008.1.2.4.90', # JPEG 2000 Lossless
    '1.2.840.10008.1.2.4.91', # JPEG 2000
    '1.2.840.10008.1.2.4.92', # MPEG2 MP@ML
    '1.2.840.10008.1.2.4.93', # MPEG2 MP@HL
    '1.2.840.10008.1.2.4.102',# MPEG-4 AVC/H.264 BD Compatible
    '1.2.840.10008.1.2.4.103' # MPEG-4 AVC/H.264 High Profile
]

UNCOMPRESSED_TRANSFER_SYNTAXES = [
    ImplicitVRLittleEndian,
    ExplicitVRLittleEndian,
    ExplicitVRBigEndian,
]

# SOP classes you want to support
SUPPORTED_SOP_CLASSES = [
    '1.2.840.10008.1.1',            # Verification
    '1.2.840.10008.5.1.4.1.1.6',    # Ultrasound Image Storage
    '1.2.840.10008.5.1.4.1.1.6.1',  # Ultrasound Image Storage (Retired)
    '1.2.840.10008.5.1.4.1.1.3',    # Ultrasound Multi-frame Image Storage (Retired)
    '1.2.840.10008.5.1.4.1.1.1.3.1', # Digital Intra – oral X-Ray Image Storage – for Processing
    '1.2.840.10008.5.1.4.1.1.3.1',  # Ultrasound Multi-frame Image Storage
    '1.2.840.10008.5.1.4.1.1.7.4',  # Secondary Capture (Multiframe True Color)
    '1.2.840.10008.5.1.4.1.1.66.4', # Segmentation Storage
    '1.2.840.10008.5.1.4.1.1.130',  # Surface Mesh Storage
    '1.2.840.10008.1.20.1',         # Storage Commitment Push Model
]

# --- Configuration ---
FORWARD_DESTINATIONS = {
    # "AE_TITLE": ("hostname", port)
    "DCM4CHEE": ("3.106.12.127", 11112)
}

# Parse the environment
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
conversion_workers = env.int("CONVERSION_WORKERS", 5)
api_workers = env.int("API_WORKERS", 3)
forwarding_workers = env.int("FORWARDING_WORKERS", 2)
scp_port = env.int("SCP_PORT", 11112)
loglevel = env.log_level("LOG_LEVEL", "INFO")
dicom_prefix = env("DICOM_PREFIX", "")
metadata_prefix = env("METADATA_PREFIX", "")
cstore_delay_ms = env.int("CSTORE_DELAY_MS", 0)
boto_max_pool_connections = env.int("BOTO_MAX_POOL_CONNECTIONS", 20)
dimse_timeout = env.int("DIMSE_TIMEOUT", 30)
maximum_associations = env.int("MAXIMUM_ASSOCIATIONS", 10)
maximum_pdu_size = env.int("MAXIMUM_PDU_SIZE", 0)
network_timeout = env.int("NETWORK_TIMEOUT", 60)
max_queue_size = env.int("MAX_QUEUE_SIZE", 1000)
max_concurrent_processing = env.int("MAX_CONCURRENT_PROCESSING", 50)
api_timeout = env.int("API_TIMEOUT", 30)
api_retry_attempts = env.int("API_RETRY_ATTEMPTS", 3)
health_check_interval = env.int("HEALTH_CHECK_INTERVAL", 60)

if dicom_prefix: 
    dicom_prefix = dicom_prefix + '/'
if metadata_prefix:
    metadata_prefix = metadata_prefix + '/'

# Set default logging configuration
logging.basicConfig(
    format='%(levelname)s - %(asctime)s.%(msecs)03d %(threadName)s: %(message)s',
    datefmt='%H:%M:%S', 
    level=loglevel
)

class NoUnknownPDUTypeFilter(logging.Filter):
    def filter(self, record):
        return "Unknown PDU type received" not in record.getMessage()

# Add the filter to the root logger
logging.getLogger().addFilter(NoUnknownPDUTypeFilter())

logging.info(f'Setting log level to {loglevel}')

@dataclass
class ProcessingTask:
    """Data class for processing tasks"""
    dicom_path: str
    task_id: str
    created_at: float
    
@dataclass
class ConversionResult:
    """Data class for conversion results"""
    task_id: str
    success: bool
    output_path: Optional[str] = None
    s3_key: Optional[str] = None
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dicom_size: int = 0

@dataclass
class ForwardingTask:
    """Data class for forwarding tasks"""
    dicom_path: str
    task_id: str

@dataclass
class ProcessingMetrics:
    """Metrics tracking for the processing pipeline"""
    total_received: int = 0
    total_processed: int = 0
    total_failed: int = 0
    total_uploaded: int = 0
    upload_failures: int = 0
    total_forwarded: int = 0
    forwarding_failures: int = 0
    api_successes: int = 0
    api_failures: int = 0
    queue_depth: int = 0
    forwarding_queue_depth: int = 0
    processing_times: list = None
    total_received_size: int = 0
    total_processed_size: int = 0
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = []

@dataclass
class TaskCompletionTracker:
    """Tracks completion of both upload and forwarding for a single DICOM file"""
    task_id: str
    dicom_path: str
    upload_completed: bool = False
    forwarding_completed: bool = False
    
    def is_fully_completed(self) -> bool:
        return self.upload_completed and self.forwarding_completed

class CircuitBreaker:
    """Simple circuit breaker for API calls"""
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def on_success(self):
        with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def on_failure(self):
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

class ResourceManager:
    """Manages all processing resources and queues"""
    
    def __init__(self):
        # Create bounded queues
        self.conversion_queue = Queue(maxsize=max_queue_size)
        self.upload_queue = Queue(maxsize=max_queue_size)
        self.api_queue = Queue(maxsize=max_queue_size)
        self.forwarding_queue = Queue(maxsize=max_queue_size)
        
        # Create separate thread pools
        self.conversion_executor = ThreadPoolExecutor(
            max_workers=conversion_workers,
            thread_name_prefix="conversion"
        )
        self.upload_executor = ThreadPoolExecutor(
            max_workers=s3_upload_workers,
            thread_name_prefix="upload"
        )
        self.api_executor = ThreadPoolExecutor(
            max_workers=api_workers,
            thread_name_prefix="api"
        )
        self.forwarding_executor = ThreadPoolExecutor(
            max_workers=forwarding_workers,
            thread_name_prefix="forwarding"
        )
        
        # Track active futures with weak references to prevent memory leaks
        self.active_futures = weakref.WeakSet()
        self.metrics = ProcessingMetrics()
        self.circuit_breaker = CircuitBreaker()
        self._shutdown_event = threading.Event()
        
        # Track task completions to ensure both workers complete before file cleanup
        self.completion_trackers = {}  # task_id -> TaskCompletionTracker
        self._completion_lock = threading.Lock()
        
        # Create S3 client with connection pooling
        self.s3_client = boto3.client(
            's3', 
            region_name=region,
            config=botocore.client.Config(
                max_pool_connections=boto_max_pool_connections,
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )
        )

        # Create S3 client for the new bucket
        self.s3_client_dicom = boto3.client(
            's3',
            region_name='ap-southeast-2',
            config=botocore.client.Config(
                max_pool_connections=boto_max_pool_connections,
                retries={'max_attempts': 3, 'mode': 'adaptive'}
            )
        )
        
        # Start worker threads
        self._start_workers()
        
        # Start health check thread
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="health-check"
        )
        self.health_check_thread.start()
    
    def _start_workers(self):
        """Start background worker threads"""
        threading.Thread(
            target=self._conversion_worker,
            daemon=True,
            name="conversion-dispatcher"
        ).start()
        
        threading.Thread(
            target=self._upload_worker,
            daemon=True,
            name="upload-dispatcher"
        ).start()
        
        threading.Thread(
            target=self._api_worker,
            daemon=True,
            name="api-dispatcher"
        ).start()

        threading.Thread(
            target=self._forwarding_worker,
            daemon=True,
            name="forwarding-dispatcher"
        ).start()
    
    def _health_check_loop(self):
        """Periodic health check and cleanup"""
        while not self._shutdown_event.wait(health_check_interval):
            try:
                self._cleanup_completed_futures()
                self._log_metrics()
            except Exception as e:
                logging.error(f"Health check error: {e}")
    
    def _cleanup_completed_futures(self):
        """Clean up completed futures to prevent memory leaks"""
        # Note: WeakSet automatically removes unreferenced futures
        active_count = len(self.active_futures)
        logging.debug(f"Active futures: {active_count}")
    
    def _log_metrics(self):
        """Log current processing metrics"""
        m = self.metrics
        total_size_left_to_process = m.total_received_size - m.total_processed_size
        logging.info(
            f"Metrics - Received: {m.total_received}, Processed: {m.total_processed}, "
            f"Failed: {m.total_failed}, Conversion Queue: {m.queue_depth}, "
            f"Forwarding Queue: {m.forwarding_queue_depth}, "
            f"Upload success: {m.total_uploaded}, Upload failures: {m.upload_failures}, "
            f"Forward success: {m.total_forwarded}, Forward failures: {m.forwarding_failures}, "
            f"API success: {m.api_successes}, API failures: {m.api_failures}, "
            f"Total size of processed files: {m.total_processed_size}, "
            f"Total size of received files: {m.total_received_size}, "
            f"Total size of files left to process: {total_size_left_to_process}"
        )
    
    def can_accept_task(self) -> bool:
        """Check if system can accept new processing tasks"""
        return (
            self.conversion_queue.qsize() < max_queue_size * 0.9 and
            self.forwarding_queue.qsize() < max_queue_size * 0.9 and
            self.upload_queue.qsize() < max_queue_size * 0.9 and
            self.api_queue.qsize() < max_queue_size * 0.9
        )
    
    def _register_task_for_completion(self, task_id: str, dicom_path: str):
        """Register a task for completion tracking"""
        with self._completion_lock:
            self.completion_trackers[task_id] = TaskCompletionTracker(
                task_id=task_id,
                dicom_path=dicom_path
            )
    
    def _mark_upload_completed(self, task_id: str) -> bool:
        """Mark upload as completed and return True if both operations are done"""
        with self._completion_lock:
            if task_id in self.completion_trackers:
                tracker = self.completion_trackers[task_id]
                tracker.upload_completed = True
                return tracker.is_fully_completed()
        return False
    
    def _mark_forwarding_completed(self, task_id: str) -> bool:
        """Mark forwarding as completed and return True if both operations are done"""
        with self._completion_lock:
            if task_id in self.completion_trackers:
                tracker = self.completion_trackers[task_id]
                tracker.forwarding_completed = True
                return tracker.is_fully_completed()
        return False
    
    def _cleanup_completed_task(self, task_id: str):
        """Clean up the DICOM file and remove tracker for completed task"""
        with self._completion_lock:
            if task_id in self.completion_trackers:
                tracker = self.completion_trackers[task_id]
                if tracker.is_fully_completed():
                    # Clean up the original DICOM file
                    try:
                        if os.path.exists(tracker.dicom_path):
                            os.remove(tracker.dicom_path)
                            logging.info(f"Cleaned up DICOM file after both workers completed: {tracker.dicom_path}")
                    except OSError as e:
                        logging.warning(f"Failed to cleanup DICOM file {tracker.dicom_path}: {e}")
                    
                    # Remove tracker
                    del self.completion_trackers[task_id]
    
    def submit_task(self, task: ProcessingTask) -> bool:
        """Submit a new processing task"""
        try:
            self.conversion_queue.put_nowait(task)
            self.metrics.total_received += 1
            self.metrics.queue_depth = self.conversion_queue.qsize()
            return True
        except Full:
            logging.warning("Conversion queue full, rejecting task")
            return False

    def submit_forwarding_task(self, task: ForwardingTask) -> bool:
        """Submit a new forwarding task"""
        try:
            self.forwarding_queue.put_nowait(task)
            self.metrics.forwarding_queue_depth = self.forwarding_queue.qsize()
            return True
        except Full:
            logging.warning("Forwarding queue full, rejecting task")
            return False
    
    def _conversion_worker(self):
        """Worker thread for processing conversion queue"""
        while not self._shutdown_event.is_set():
            try:
                task = self.conversion_queue.get(timeout=1.0)
                future = self.conversion_executor.submit(self._process_dicom, task)
                self.active_futures.add(future)
                future.add_done_callback(self._handle_conversion_result)
                self.conversion_queue.task_done()
                self.metrics.queue_depth = self.conversion_queue.qsize()
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Conversion worker error: {e}")
    
    def _upload_worker(self):
        """Worker thread for processing upload queue"""
        while not self._shutdown_event.is_set():
            try:
                result = self.upload_queue.get(timeout=1.0)
                future = self.upload_executor.submit(self._upload_to_s3, result)
                self.active_futures.add(future)
                future.add_done_callback(self._handle_upload_result)
                self.upload_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Upload worker error: {e}")

    def _forwarding_worker(self):
        """Worker thread for processing forwarding queue"""
        while not self._shutdown_event.is_set():
            try:
                task = self.forwarding_queue.get(timeout=1.0)
                future = self.forwarding_executor.submit(self._forward_dicom, task)
                self.active_futures.add(future)
                future.add_done_callback(self._handle_forwarding_result)
                self.forwarding_queue.task_done()
                self.metrics.forwarding_queue_depth = self.forwarding_queue.qsize()
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Forwarding worker error: {e}")

    def _api_worker(self):
        """Worker thread for processing API queue"""
        while not self._shutdown_event.is_set():
            try:
                payload = self.api_queue.get(timeout=1.0)
                future = self.api_executor.submit(self._send_api_notification, payload)
                self.active_futures.add(future)
                future.add_done_callback(self._handle_api_result)
                self.api_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logging.error(f"API worker error: {e}")
    
    def _handle_conversion_result(self, future: Future):
        """Handle conversion completion"""
        try:
            result = future.result()
            self.metrics.total_processed_size += result.dicom_size
            if result.success:
                self.metrics.total_processed += 1
                self.upload_queue.put_nowait(result)
            else:
                self.metrics.total_failed += 1
                logging.error(f"Conversion failed for task {result.task_id}: {result.error}")
                # Mark upload as completed (failed) to allow cleanup
                if self._mark_upload_completed(result.task_id):
                    self._cleanup_completed_task(result.task_id)
        except Exception as e:
            self.metrics.total_failed += 1
            logging.error(f"Conversion result handling error: {e}")
    
    def _handle_upload_result(self, future: Future):
        """Handle upload completion"""
        try:
            success, result = future.result()
            if success:
                self.metrics.total_uploaded += 1
                # Queue for API notification
                if result.metadata:
                    api_payload = self._create_api_payload(result)
                    self.api_queue.put_nowait(api_payload)
            else:
                self.metrics.upload_failures += 1
            
            # Mark upload as completed and cleanup if both workers are done
            if self._mark_upload_completed(result.task_id):
                self._cleanup_completed_task(result.task_id)
        except Exception as e:
            self.metrics.upload_failures += 1
            logging.error(f"Upload result handling error: {e}")

    def _handle_forwarding_result(self, future: Future):
        """Handle forwarding completion"""
        try:
            success, error_message, task_id = future.result()
            if success:
                self.metrics.total_forwarded += 1
            else:
                self.metrics.forwarding_failures += 1
                logging.error(f"Forwarding failed: {error_message}")
            
            # Mark forwarding as completed and cleanup if both workers are done
            if self._mark_forwarding_completed(task_id):
                self._cleanup_completed_task(task_id)
        except Exception as e:
            self.metrics.forwarding_failures += 1
            logging.error(f"Forwarding result handling error: {e}")

    def _handle_api_result(self, future: Future):
        """Handle API notification completion"""
        try:
            success = future.result()
            if success:
                self.metrics.api_successes += 1
                self.circuit_breaker.on_success()
            else:
                self.metrics.api_failures += 1
                self.circuit_breaker.on_failure()
        except Exception as e:
            self.metrics.api_failures += 1
            self.circuit_breaker.on_failure()
            logging.error(f"API result handling error: {e}")
    
    def _process_dicom(self, task: ProcessingTask) -> ConversionResult:
        """Process DICOM file with proper error handling and cleanup"""
        start_time = time.time()
        temp_files_to_cleanup = []  # Only for cleanup on error
        dicom_size = 0
        
        try:
            # Get DICOM file size
            if os.path.exists(task.dicom_path):
                dicom_size = os.path.getsize(task.dicom_path)
            # Extract metadata first
            metadata = get_dicom_metadata(task.dicom_path)
            if not metadata:
                return ConversionResult(
                    task_id=task.task_id,
                    success=False,
                    error="Failed to extract DICOM metadata",
                    dicom_size=dicom_size
                )
            
            # Determine conversion type and paths
            sop_instance_uid = metadata.get("SOPInstanceUID", "unknown_sop")
            series_instance_uid = metadata.get("SeriesInstanceUID", "unknown_series")
            study_instance_uid = metadata.get("StudyInstanceUID", "unknown_study")
            
            if add_studyuid_prefix:
                s3_prefix = f"{dicom_prefix}{study_instance_uid}/"
            else:
                s3_prefix = dicom_prefix
            
            # Create temporary output file
            temp_output = tempfile.NamedTemporaryFile(delete=False)
            temp_files_to_cleanup.append(temp_output.name)
            temp_output.close()
            
            # Convert based on frame count
            if metadata.get("NumberOfFrames", 1) > 1:
                output_path = temp_output.name + ".webm"
                s3_key = f"{folder_name}/{s3_prefix}{study_instance_uid}/{series_instance_uid}/{sop_instance_uid}.webm"
                content_type = "video/webm"
                
                if not convert_dicom_to_webm(
                    task.dicom_path, 
                    output_path, 
                    metadata.get("RecommendedDisplayFrameRate", 50)
                ):
                    temp_files_to_cleanup.append(output_path)
                    return ConversionResult(
                        task_id=task.task_id,
                        success=False,
                        error="WebM conversion failed",
                        dicom_size=dicom_size
                    )
            else:
                output_path = temp_output.name + ".jpeg"
                s3_key = f"{folder_name}/{s3_prefix}{study_instance_uid}/{series_instance_uid}/{sop_instance_uid}.jpeg"
                content_type = "image/jpeg"
                
                if not convert_dicom_to_jpeg(task.dicom_path, output_path):
                    temp_files_to_cleanup.append(output_path)
                    return ConversionResult(
                        task_id=task.task_id,
                        success=False,
                        error="JPEG conversion failed",
                        dicom_size=dicom_size
                    )
            
            processing_time = time.time() - start_time
            self.metrics.processing_times.append(processing_time)
            # Keep only last 1000 times to prevent memory growth
            if len(self.metrics.processing_times) > 1000:
                self.metrics.processing_times = self.metrics.processing_times[-1000:]
            
            logging.info(f"Conversion completed for {task.task_id} in {processing_time:.2f}s")
            
            # Return success with output_path - DON'T clean up the output file yet
            return ConversionResult(
                task_id=task.task_id,
                success=True,
                output_path=output_path,
                s3_key=s3_key,
                content_type=content_type,
                metadata=metadata,
                dicom_size=dicom_size
            )
            
        except Exception as e:
            logging.error(f"DICOM processing error for {task.task_id}: {e}")
            return ConversionResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                dicom_size=dicom_size
            )
        finally:
            # Only clean up temporary files on error, not the final output
            for temp_file in temp_files_to_cleanup:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except OSError:
                    pass

    def _forward_dicom(self, task: ForwardingTask) -> tuple:
        """Forward DICOM file to other SCPs"""
        try:
            ds = dcmread(task.dicom_path)
            orig_ts = UID(ds.file_meta.TransferSyntaxUID)
            # Negotiate original transfer syntax first, then fall back to uncompressed and other common syntaxes
            transfer_syntaxes = list(dict.fromkeys(
                [str(orig_ts)] +
                [str(ts) for ts in UNCOMPRESSED_TRANSFER_SYNTAXES] +
                ALL_TRANSFER_SYNTAXES
            ))

            ae = AE()
            # Tune SCU association in line with server settings
            ae.maximum_pdu_size = maximum_pdu_size if maximum_pdu_size > 0 else 262144
            ae.dimse_timeout = dimse_timeout
            ae.network_timeout = network_timeout
            ae.add_requested_context(ds.SOPClassUID, transfer_syntaxes)

            for ae_title, (host, port) in FORWARD_DESTINATIONS.items():
                logging.info(f"Associating with {ae_title} at {host}:{port}")
                assoc = ae.associate(host, port, ae_title=ae_title)
                if assoc.is_established:
                    logging.info("Association established")
                    # Log accepted contexts
                    # for ctx in assoc.accepted_contexts:
                    #     logging.info(f"Accepted context: abstract {ctx.abstract_syntax}, transfer {ctx.transfer_syntax.name}")

                    # Find accepted ts
                    accepted_ts = None
                    for ctx in assoc.accepted_contexts:
                        if ctx.abstract_syntax == ds.SOPClassUID:
                            accepted_ts = ctx.transfer_syntax
                            break

                    if accepted_ts is None:
                        # logging.error(f"No accepted presentation context for SOP {ds.SOPClassUID.name}")
                        assoc.release()
                        return False, "No accepted presentation context"
                    logging.info(f"Accepted transfer syntax: {accepted_ts}")
                    accepted_ts_uid = UID(accepted_ts[0])

                    # Decompress only if peer requires uncompressed
                    if not accepted_ts_uid.is_compressed and orig_ts.is_compressed:
                        logging.info("Peer accepted uncompressed TS - decompressing for forwarding")
                        ds.decompress()

                    # Set dataset TS to match accepted context
                    ds.file_meta.TransferSyntaxUID = accepted_ts_uid

                    status = assoc.send_c_store(ds)
                    if status:
                        logging.info(f'C-STORE to {ae_title} successful')
                    else:
                        logging.error(f'C-STORE to {ae_title} failed with status: {status}')
                    assoc.release()
                else:
                    logging.error(f'Association with {ae_title} rejected or aborted.')
            # File cleanup will be handled centrally after both workers complete
            return True, None, task.task_id
        except Exception as e:
            logging.error(f"DICOM forwarding error for {task.task_id}: {e}", exc_info=True)
            # File cleanup will be handled centrally after both workers complete
            return False, str(e), task.task_id

    def _upload_to_s3(self, result: ConversionResult) -> tuple:
        """Upload file to S3 with retry logic"""
        if not result.success or not result.output_path:
            return False, result
        
        # Check if file exists before attempting upload
        if not os.path.exists(result.output_path):
            logging.error(f"Converted file not found: {result.output_path}")
            return False, result
        
        start_time = time.time()
        s3_metadata = {key.lower(): str(value) for key, value in result.metadata.items()}
        
        try:
            # Get file size for logging
            file_size = os.path.getsize(result.output_path)
            logging.debug(f"Starting S3 upload of {result.s3_key} ({file_size} bytes)")
            
            with open(result.output_path, 'rb') as f:
                self.s3_client.upload_fileobj(
                    f, 
                    bucket, 
                    result.s3_key,
                    ExtraArgs={
                        'ContentType': result.content_type, 
                        'Metadata': s3_metadata
                    }
                )
            
            elapsed_time = time.time() - start_time
            logging.info(f'S3 upload completed for {result.s3_key} in {elapsed_time:.2f}s')
            
            return True, result
            
        except FileNotFoundError as e:
            logging.error(f'Converted file disappeared during upload: {result.output_path}')
            return False, result
        except Exception as e:
            logging.error(f'S3 upload error for {result.s3_key}: {e}')
            return False, result
        finally:
            # Always clean up converted file after upload attempt
            try:
                if os.path.exists(result.output_path):
                    os.remove(result.output_path)
                    logging.debug(f"Cleaned up converted file: {result.output_path}")
            except OSError as e:
                logging.warning(f"Failed to cleanup converted file {result.output_path}: {e}")
    
    def _create_api_payload(self, result: ConversionResult) -> dict:
        """Create API payload from conversion result"""
        metadata = result.metadata
        return {
            "patientDemographics": {
                "PatientName": metadata.get("PatientName", "N/A"),
                "PatientID": metadata.get("PatientID", "N/A"),
                "PatientSex": metadata.get("PatientSex", "N/A"),
                "PatientBirthDate": metadata.get("PatientBirthDate", "N/A"),
                "StudyInstanceUID": metadata.get("StudyInstanceUID", "N/A"),
                "SOPInstanceUID": metadata.get("SOPInstanceUID", "N/A"),
                "SeriesInstanceUID": metadata.get("SeriesInstanceUID", "N/A"),
                "StageName": metadata.get("StageName", "N/A"),
                "View": metadata.get("View", "N/A"),
                "InstanceNumber": metadata.get("InstanceNumber", "N/A"),
                "SeriesNumber": metadata.get("SeriesNumber", "N/A"),
            },
            "s3Url": result.s3_key
        }
    
    def _send_api_notification(self, payload: dict) -> bool:
        """Send API notification with circuit breaker and retry logic"""
        if not self.circuit_breaker.can_execute():
            logging.warning("Circuit breaker open, skipping API call")
            return False
        
        api_url = f"{backend_url}/common/uploadDicomFileFromECS"
        
        for attempt in range(api_retry_attempts):
            try:
                response = requests.post(
                    api_url, 
                    json=payload, 
                    timeout=api_timeout
                )
                response.raise_for_status()
                
                logging.info(f"API notification sent successfully. Status: {response.status_code}")
                return True
                
            except requests.exceptions.RequestException as e:
                logging.warning(f"API notification attempt {attempt + 1} failed: {e}")
                if attempt < api_retry_attempts - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        logging.error("All API notification attempts failed")
        return False

    def shutdown(self):
        """Graceful shutdown of all resources"""
        logging.info("Initiating graceful shutdown...")
        self._shutdown_event.set()
        
        # Wait for queues to empty
        for queue in [self.conversion_queue, self.upload_queue, self.api_queue, self.forwarding_queue]:
            queue.join()
        
        # Shutdown executors
        for executor in [self.conversion_executor, self.upload_executor, self.api_executor, self.forwarding_executor]:
            executor.shutdown(wait=True)
        
        logging.info("Shutdown complete")

# Global resource manager instance
resource_manager = None

def get_dicom_fps(ds):
    """Extract frame rate from DICOM metadata"""
    try:
        fps = None

        # Check RecommendedDisplayFrameRate (0018,0040)
        if "RecommendedDisplayFrameRate" in ds:
            try:
                fps = int(ds.RecommendedDisplayFrameRate)
                if fps > 0:
                    logging.debug(f"Found RecommendedDisplayFrameRate: {fps} FPS")
                    return fps if fps >= 30 else 50
            except Exception:
                pass

        # Check FrameTime (0018,1063) - milliseconds per frame
        if "FrameTime" in ds:
            try:
                frame_time = float(ds.FrameTime)  # in ms
                if frame_time > 0:
                    fps = round(1000 / frame_time)
                    logging.debug(f"Calculated FPS from FrameTime: {fps} FPS")
                    return fps if fps >= 30 else 50
            except Exception:
                pass

        # Default fallback
        logging.debug("No FPS found in DICOM metadata, using default: 50 FPS")
        return 50

    except Exception as e:
        logging.warning(f"Error reading DICOM FPS: {e}")
        return 50

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
            "RecommendedDisplayFrameRate": get_dicom_fps(ds),
            "SOPInstanceUID": str(ds.SOPInstanceUID) if "SOPInstanceUID" in ds else "N/A",
            "StudyInstanceUID": str(ds.StudyInstanceUID) if "StudyInstanceUID" in ds else "N/A",
            "SeriesInstanceUID": str(ds.SeriesInstanceUID) if "SeriesInstanceUID" in ds else "N/A",
            "StageName": str(ds.StageName) if "StageName" in ds else "N/A",
            "View": str(ds.ViewCodeSequence[0].CodeMeaning) if "ViewCodeSequence" in ds and len(ds.ViewCodeSequence) > 0 else "N/A",
            "InstanceNumber": str(ds.InstanceNumber) if "InstanceNumber" in ds else "N/A",
            "SeriesNumber": str(ds.SeriesNumber) if "SeriesNumber" in ds else "N/A",
        }
        return metadata
    except Exception as e:
        logging.error(f"Error reading DICOM metadata: {e}")
        return None

def convert_dicom_to_jpeg(dicom_path, output_path):
    """Converts a single-frame DICOM to JPEG."""
    try:
        ds = dcmread(dicom_path)
        if not hasattr(ds, 'pixel_array'):
            logging.error(f"DICOM file {dicom_path} has no pixel data")
            return False
        pixel_array = ds.pixel_array
        # Handle 16-bit images by converting to 8-bit
        if pixel_array.dtype == np.uint16 or pixel_array.dtype == np.int16:
            # Normalize to 0-255 range for 8-bit JPEG
            # Method 1: Simple linear scaling
            pixel_min = pixel_array.min()
            pixel_max = pixel_array.max()
            if pixel_max > pixel_min:
                # Scale to 0-255
                pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)
            else:
                # Handle edge case where all pixels have same value
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        # Convert to PIL Image
        img = Image.fromarray(pixel_array)
        # Ensure we're in a JPEG-compatible mode
        if img.mode not in ['L', 'RGB', 'CMYK']:
            if img.mode == 'I' or img.mode == 'I;16':
                # Convert 16-bit integer to 8-bit grayscale
                img = img.convert('L')
            elif img.mode in ['P', 'RGBA']:
                # Convert palette or RGBA to RGB
                img = img.convert('RGB')
        img.save(output_path, "JPEG", quality=95, optimize=True)
        logging.debug(f"Successfully converted {dicom_path} to JPEG")
        return True
    except Exception as e:
        logging.error(f"Error converting DICOM to JPEG: {e}")
        return False

def convert_dicom_to_webm(dicom_path, output_path, frame_rate=24):
    """Converts a multi-frame DICOM to WebM by extracting frames."""
    temp_dir = None
    try:
        ds = dcmread(dicom_path)
        if not hasattr(ds, 'pixel_array') or ds.pixel_array.ndim < 3:
            logging.error(f"DICOM file {dicom_path} does not contain multi-frame pixel data.")
            return False

        temp_dir = tempfile.mkdtemp()
        frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
        
        # Extract frames more efficiently
        for i, frame in enumerate(ds.pixel_array):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            img = Image.fromarray(frame)
            img.save(frame_path, "PNG", optimize=True)

        # Use ffmpeg to combine frames into WebM with optimized settings
        (
            ffmpeg
            .input(frame_pattern, framerate=frame_rate)
            .output(
                output_path,
                r=frame_rate,
                vcodec='libvpx-vp9',
                crf=30,  # Slightly higher CRF for smaller files
                preset='fast',  # Faster encoding
                pix_fmt='yuv420p',
                profile='0',
                threads=2  # Limit threads to avoid resource contention
            )
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        logging.debug(f"Successfully converted {dicom_path} to WebM")
        return True
    except Exception as e:
        logging.error(f"Error converting DICOM to WebM: {e}")
        return False
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

def handle_store(event):
    """Handle a C-STORE request event with improved error handling and flow control."""
    global resource_manager
    
    start_time = time.time()
    
    # Check if the received file is a Structured Report
    sop_class_uid = event.file_meta.MediaStorageSOPClassUID
    if "1.2.840.10008.5.1.4.1.1.88" in sop_class_uid:
        logging.info(f"Received Structured Report (SOP Class UID: {sop_class_uid}). Skipping conversion.")
        return 0x0000
    
    # Check system capacity before accepting
    if not resource_manager.can_accept_task():
        logging.warning("System at capacity, rejecting C-STORE request")
        return 0xA700  # Out of resources
    
    try:
        # Create temporary DICOM file
        temp_dicom = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
        try:
            # Write DICOM data efficiently
            temp_dicom.write(b'\x00' * 128)  # Preamble
            temp_dicom.write(b'DICM')        # Prefix
            write_file_meta_info(temp_dicom, event.file_meta)
            temp_dicom.write(event.request.DataSet.getvalue())
            temp_dicom_path = temp_dicom.name
            # Get file size
            file_size = os.path.getsize(temp_dicom_path)
            resource_manager.metrics.total_received_size += file_size
        finally:
            temp_dicom.close()
        
        task_id=f"{int(start_time * 1000)}_{threading.current_thread().ident}"
        # Create processing task
        task = ProcessingTask(
            dicom_path=temp_dicom_path,
            task_id=task_id,
            created_at=start_time
        )
        
        # Submit task to processing pipeline
        if not resource_manager.submit_task(task):
            # Clean up on failure to submit
            try:
                os.remove(temp_dicom_path)
            except OSError:
                pass
            logging.warning("Failed to submit task to processing queue")
            return 0xA700  # Out of resources

        # Create forwarding task
        forwarding_task = ForwardingTask(
            dicom_path=temp_dicom_path,
            task_id=task_id
        )

        # Submit task to forwarding pipeline
        forwarding_submitted = resource_manager.submit_forwarding_task(forwarding_task)
        if not forwarding_submitted:
            # This part is tricky. If forwarding queue is full, we might want to handle it differently.
            # For now, we'll just log it. The file will still be processed for conversion.
            logging.warning("Failed to submit task to forwarding queue")

        # Register task for completion tracking
        resource_manager._register_task_for_completion(task_id, temp_dicom_path)
        
        # If forwarding was not submitted, mark it as completed immediately
        if not forwarding_submitted:
            resource_manager._mark_forwarding_completed(task_id)

        # Optional C-STORE delay for throttling
        if cstore_delay_ms > 0:
            time.sleep(cstore_delay_ms / 1000)
        
        processing_time = time.time() - start_time
        logging.info(f'DICOM instance received and queued for processing in {processing_time:.3f}s')
        
        return 0x0000  # Success
        
    except Exception as e:
        logging.error(f'Error in C-STORE processing: {e}')
        return 0xC211  # Processing failure

def main():
    global resource_manager
    
    logging.warning('Starting DICOM C-STORE application...')
    logging.warning(f'Configuration: conversion_workers={conversion_workers}, '
                   f's3_upload_workers={s3_upload_workers}, api_workers={api_workers}, '
                   f'forwarding_workers={forwarding_workers}, '
                   f'max_queue_size={max_queue_size}, max_concurrent_processing={max_concurrent_processing}')
    
    # Initialize resource manager
    resource_manager = ResourceManager()
    
    try:
        # Setup event handlers
        handlers = [(evt.EVT_C_STORE, handle_store), (evt.EVT_C_ECHO, lambda event: 0x0000)]

        # Initialize the Application Entity
        ae = AE()
        ae.maximum_pdu_size = maximum_pdu_size
        ae.dimse_timeout = dimse_timeout
        ae.maximum_associations = maximum_associations
        ae.network_timeout = network_timeout

        # Support presentation contexts for all storage SOP Classes
        supported_contexts = AllStoragePresentationContexts
        
        # Add missing presentation contexts
        supported_contexts.append(build_context('1.2.840.10008.5.1.4.1.1.130'))
        supported_contexts.append(build_context('1.2.840.10008.5.1.4.1.1.6'))
        # supported_contexts.append(build_context(ThreeDRenderingAndSegmentationDefaults))
        supported_contexts.append(build_context('1.2.840.10008.5.1.4.1.1.3.1'))
        supported_contexts.append(build_context('1.2.840.10008.5.1.4.1.1.3'))
        supported_contexts.append(build_context('1.2.840.10008.1.20.1'))
        supported_contexts.append(build_context('1.2.840.10008.5.1.4.1.1.7.4'))
        # supported_contexts.append(build_context(Private3DPresentationState))
        # Missing SOP Classes
        supported_contexts.append(build_context('1.2.840.10008.1.1'))            # Verification SOP Class
        supported_contexts.append(build_context('1.2.840.10008.5.1.4.1.1.66.4')) # Segmentation Storage
        supported_contexts.append(build_context('1.2.840.10008.5.1.4.1.1.130'))  # Volume Surface Mesh Storage (closest official, may differ if you meant "Volume Set")
        supported_contexts.append(build_context('1.2.840.10008.5.1.4.1.1.6.1'))  # Ultrasound Image Storage (Retired)
        supported_contexts.append(build_context('1.2.840.10008.5.1.4.1.1.3.1'))  # Ultrasound Multi-frame Image Storage (Retired)
        # Private SOP Classes like "Private 3D Presentation State" require vendor-specific UID

        for context in supported_contexts:
            context.transfer_syntax = [
                '1.2.840.10008.1.2',      # Implicit VR Little Endian
                '1.2.840.10008.1.2.1',    # Explicit VR Little Endian
                '1.2.840.10008.1.2.2',    # Explicit VR Big Endian
                '1.2.840.10008.1.2.5',    # RLE Lossless
                '1.2.840.10008.1.2.4.50', # JPEG Baseline (Process 1)
                '1.2.840.10008.1.2.4.57', # JPEG Lossless, Non-Hierarchical (Process 14)
                '1.2.840.10008.1.2.4.70', # JPEG Lossless, SV1
                '1.2.840.10008.1.2.4.80', # JPEG-LS Lossless
                '1.2.840.10008.1.2.4.81', # JPEG-LS Near-Lossless
                '1.2.840.10008.1.2.4.90', # JPEG 2000 Lossless
                '1.2.840.10008.1.2.4.91', # JPEG 2000
                '1.2.840.10008.1.2.4.92', # MPEG2 MP@ML
                '1.2.840.10008.1.2.4.93', # MPEG2 MP@HL
                '1.2.840.10008.1.2.4.102',# MPEG-4 AVC/H.264 BD Compatible
                '1.2.840.10008.1.2.4.103',# MPEG-4 AVC/H.264 High Profile
                # add HEVC/H.265 if your devices use it:
                '1.2.840.10008.1.2.4.108',# HEVC/H.265 Main Profile
                '1.2.840.10008.1.2.4.109',# HEVC/H.265 Main 10 Profile
            ]
        ae.supported_contexts = supported_contexts
        
        # Enable verification
        ae.add_supported_context(Verification)
        
        # Start listening for incoming association requests
        logging.warning(f'Starting SCP Listener on port {scp_port}')
        scp = ae.start_server(("", scp_port), evt_handlers=handlers)
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(60)  # Sleep and let background threads do the work
        except KeyboardInterrupt:
            logging.info("Received shutdown signal")
        
    except Exception as e:
        logging.error(f"Fatal error in main: {e}")
        raise
    finally:
        # Cleanup
        if resource_manager:
            resource_manager.shutdown()
        logging.warning("Application shutdown complete")

if __name__ == "__main__":
    main()
