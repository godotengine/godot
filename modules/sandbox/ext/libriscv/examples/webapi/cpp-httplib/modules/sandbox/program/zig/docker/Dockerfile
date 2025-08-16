FROM ubuntu:24.04

RUN apt-get -qq update && \
	apt-get install -y -q \
	curl xz-utils lld

# Install Zig
# https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz
RUN curl -L https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz -o zig.tar.xz
# zig-linux-x86_64-0.13.0
RUN tar -xf zig.tar.xz && rm zig.tar.xz && mv zig-linux-x86_64-0.13.0 /usr/zig

# TODO: API in /usr/api

WORKDIR /usr/src
# Copy API files
RUN mkdir -p /usr/api
COPY build.sh api/* /usr/api/

# Make the container a background process
CMD ["/bin/bash", "-c", "tail -f /dev/null"]
