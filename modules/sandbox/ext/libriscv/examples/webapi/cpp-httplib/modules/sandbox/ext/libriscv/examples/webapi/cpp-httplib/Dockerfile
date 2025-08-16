FROM yhirose4dockerhub/ubuntu-builder AS builder
WORKDIR /build
COPY httplib.h .
COPY docker/main.cc .
RUN g++ -std=c++23 -static -o server -O2 -I. main.cc && strip server

FROM scratch
COPY --from=builder /build/server /server
COPY docker/html/index.html /html/index.html
EXPOSE 80

ENTRYPOINT ["/server"]
CMD ["0.0.0.0", "80", "/", "/html"]
