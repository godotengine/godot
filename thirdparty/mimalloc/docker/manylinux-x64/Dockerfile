FROM quay.io/pypa/manylinux2014_x86_64

# Install tools
RUN yum install -y openssl-devel
RUN yum install -y gcc gcc-c++ kernel-devel make
RUN yum install -y git cmake
RUN yum install -y vim

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