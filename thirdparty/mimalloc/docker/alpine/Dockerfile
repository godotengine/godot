# alpine image  
FROM alpine

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