# in your Dockerfile
FROM sphinxdoc/sphinx

RUN mkdir /install
WORKDIR /install
ADD ../requirements.txt /install
RUN python3 -m pip install -r /install/requirements.txt
WORKDIR /docs
