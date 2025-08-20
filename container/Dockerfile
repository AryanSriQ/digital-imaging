FROM ubuntu:jammy

# build args
ARG DEBIAN_FRONTEND=noninteractive
ARG SCP_PORT
ARG STUNNEL_PORT

# environment:
ENV TZ=Etc/UTC

# Configure base OS and install packages
RUN apt-get update && apt-get -y install \
    apt-utils \
    tzdata \
    stunnel \
    supervisor \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python packages - added aiohttp for the improved code
RUN pip install --no-cache-dir \
    boto3 \
    pydicom \
    pynetdicom \
    environs \
    ffmpeg-python \
    Pillow \
    numpy \
    requests \
    aiohttp

WORKDIR /app

# configure supervisord
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY --chown=nobody:nogroup prefixlog.sh .
RUN chmod +x prefixlog.sh

# configure storescp
COPY --chown=nobody:nogroup storescp.py .
RUN chmod +x storescp.py

# configure stunnel, note that cert files may be missing.
COPY stunnel.conf cert.pe[m] cert.ke[y] /etc/stunnel/

# if SSL certificate and key are provided - use them. If not - generate a self-signed Demo pair
RUN if [ ! -s /etc/stunnel/cert.key -o ! -s /etc/stunnel/cert.pem ] ; then \
	openssl req -new -x509 -days 365 -nodes -subj '/CN=s3storescpDemo/O=s3StorescpDemo/C=US' -out /etc/stunnel/cert.pem -keyout /etc/stunnel/cert.key ;\
    fi 
   
RUN chmod go-rwx /etc/stunnel/cert.key
RUN sed -i "s/DOCKER_BUILD_SCP_PORT_PLACEHOLDER/${SCP_PORT}/g" /etc/stunnel/stunnel.conf
RUN sed -i "s/DOCKER_BUILD_STUNNEL_PORT_PLACEHOLDER/${STUNNEL_PORT}/g" /etc/stunnel/stunnel.conf

# set ownership of files
RUN chown -R nobody:nogroup /app /etc/stunnel

# Configure ports - storescp, stunnel -> storescp
EXPOSE ${SCP_PORT} ${STUNNEL_PORT}

# Set default environment variables for the new configuration options
ENV CONVERSION_WORKERS=5 \
    S3_UPLOAD_WORKERS=10 \
    API_WORKERS=3 \
    MAX_QUEUE_SIZE=1000 \
    MAX_CONCURRENT_PROCESSING=50 \
    API_TIMEOUT=30 \
    API_RETRY_ATTEMPTS=3 \
    HEALTH_CHECK_INTERVAL=60 \
    BOTO_MAX_POOL_CONNECTIONS=20

# supervisord handles the stunnel and storescp processes
CMD ["/usr/bin/supervisord","-c","/etc/supervisor/supervisord.conf"]
