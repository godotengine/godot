# install from an image
# download first an appropiate tar.gz image into the current directory 
# from: <https://github.com/alpinelinux/docker-alpine/tree/edge/armv7>
FROM scratch

# Substitute the image name that was downloaded
ADD alpine-minirootfs-20240329-armv7.tar.gz /    

# Install tools
RUN apk add build-base make cmake
RUN apk add git
RUN apk add vim

RUN mkdir -p  /home/dev
WORKDIR /home/dev

# Get mimalloc
RUN git clone https://github.com/microsoft/mimalloc -b dev-slice
RUN mkdir -p mimalloc/out/release
RUN mkdir -p mimalloc/out/debug

# Build mimalloc debug
WORKDIR /home/dev/mimalloc/out/debug
RUN cmake ../.. -DMI_DEBUG_FULL=ON
RUN make -j
RUN make test

CMD ["/bin/sh"]
