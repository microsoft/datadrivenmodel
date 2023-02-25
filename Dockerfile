# this is one of the cached base images available for ACI
FROM python:3.8.11
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
# COPY . /src
COPY *.py /src/
COPY requirements.txt /src/
COPY ./models/ /src/models/
COPY ./conf/ /src/conf/
COPY *.csv /src/

# Install simulator dependencies
RUN pip3 install -r requirements.txt

# # This will be the command to run the simulator
CMD ["python3", "ddm_predictor.py", "simulator.workspace_setup=False", "simulator.policy=bonsai"]