FROM ubuntu:24.04
#
# To build the image:
# docker build -t riscv64-linux-gnu .
#
# To run the container:
# docker run --name godot-cpp-compiler -dv .:/usr/src riscv64-linux-gnu
#
# Install dependencies
RUN apt-get update && apt-get install -y \
	g++-14-riscv64-linux-gnu \
	mold ccache \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get clean


# Enter the shared directory
WORKDIR /usr/src
# Copy API files
RUN mkdir -p /usr/api
COPY build.sh api/* /usr/api/

# Set the entrypoint to the build script
#ENTRYPOINT "/usr/api/build.sh" "$0" "$@"
CMD ["/bin/bash", "-c", "tail -f /dev/null"]
