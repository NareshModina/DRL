FROM python:3.12

# create and set the directory for next comands execution
WORKDIR /drl/app/models/

# Copy files to drl directory
COPY dependencies.txt /drl/app/
COPY d3qn_agent.py /drl/app/

WORKDIR /drl/app/

# Install app dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /drl/app/dependencies.txt

# Start the application
CMD ["python", "d3qn_agent.py"]