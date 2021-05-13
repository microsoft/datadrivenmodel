# this is one of the cached base images available for ACI
# FROM python:3.7.4
FROM amd64/python:3.7.7

# Install libraries and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
# Install libraries and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends
# Set up the simulator
WORKDIR /src

# Copy simulator files to /src
COPY . /src

# Install simulator dependencies
RUN pip3 install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html 

# # This will be the command to run the simulator
CMD ["python3", "moab_main.py"]