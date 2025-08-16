#!/usr/bin/env bash
if [[ $(openssl version) =~ 3\.[2-9]\.[0-9]+ ]]; then
	OPENSSL_X509_FLAG='-x509v1'
else
	OPENSSL_X509_FLAG='-x509'
fi

openssl genrsa 2048 > key.pem
openssl req -new -batch -config test.conf -key key.pem | openssl x509 -days 3650 -req -signkey key.pem > cert.pem
openssl req -x509 -config test.conf -key key.pem -sha256 -days 3650 -nodes -out cert2.pem -extensions SAN
openssl genrsa 2048 > rootCA.key.pem
openssl req $OPENSSL_X509_FLAG -new -batch -config test.rootCA.conf -key rootCA.key.pem -days 1024 > rootCA.cert.pem
openssl genrsa 2048 > client.key.pem
openssl req -new -batch -config test.conf -key client.key.pem | openssl x509 -days 370 -req -CA rootCA.cert.pem -CAkey rootCA.key.pem -CAcreateserial > client.cert.pem
openssl genrsa -passout pass:test123! 2048 > key_encrypted.pem
openssl req -new -batch -config test.conf -key key_encrypted.pem | openssl x509 -days 3650 -req -signkey key_encrypted.pem > cert_encrypted.pem
openssl genrsa -aes256 -passout pass:test012! 2048 > client_encrypted.key.pem
openssl req -new -batch -config test.conf -key client_encrypted.key.pem -passin pass:test012! | openssl x509 -days 370 -req -CA rootCA.cert.pem -CAkey rootCA.key.pem -CAcreateserial > client_encrypted.cert.pem
