FROM ubuntu:24.04

RUN apt-get -qq update && \
	apt-get install -y -q \
	curl \
	g++-14-riscv64-linux-gnu \
	git cmake mold

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Add RISC-V target
RUN rustup default nightly
RUN rustup target add riscv64gc-unknown-linux-gnu

# Create a project directory
RUN mkdir -p /usr/project
COPY build.sh Cargo.toml /usr/project/
COPY .cargo /usr/project/.cargo
COPY src /usr/project/src

# Create a source directory (which will be mounted)
RUN mkdir -p /usr/src
WORKDIR /usr/src

# Make the container a background process
CMD ["/bin/bash", "-c", "tail -f /dev/null"]
