FROM alpine:latest

ARG auth="basic"
ARG port="3128"

RUN apk update && apk add --no-cache squid

COPY ./${auth}_squid.conf /etc/squid/squid.conf
COPY ./${auth}_passwd /etc/squid/passwd

EXPOSE ${port}

CMD ["/usr/sbin/squid", "-N"]
