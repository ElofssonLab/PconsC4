FROM python:3.7-slim

# Prevents .pyc file generation in the container
ENV PYTHONDONTWRITEBYTECODE=1  
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1  

# install relevant packages to env (e.g. git)
RUN apt-get -y update \
    && apt-get install --reinstall -y build-essential \
    &&  apt-get install -y gcc

# Install pip requirements for python
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-input pconsc4 \
    && rm requirements.txt

CMD ["bash"]
