#ifndef STREAM_PEER_SSL_H
#define STREAM_PEER_SSL_H

#ifdef OPENSSL_ENABLED

#include "io/stream_peer.h"
#include <openssl/applink.c> // To prevent crashing (see the OpenSSL FAQ)
#include <openssl/bio.h> // BIO objects for I/O
#include <openssl/ssl.h> // SSL and SSL_CTX for SSL connections
#include <openssl/err.h> // Error reporting

#include <stdio.h> // If you don't know what this is for stop reading now.
class StreamPeerSSL : public StreamPeer {

	OBJ_TYPE(StreamPeerSSL,StreamPeer);
public:

	enum ConnectFlags {

		CONNECT_FLAG_BUG_WORKAROUNDS=1,
		CONNECT_FLAG_NO_SSLV2=2,
		CONNECT_FLAG_NO_SSLV3=4,
		CONNECT_FLAG_NO_TLSV1=8,
		CONNECT_FLAG_NO_COMPRESSION=16,
	};

	SSL_CTX* ctx;
	SSL* ssl;
	BIO* bio;


public:


	Error connect(const String &p_host,int p_port);
	static void initialize_ssl();
	static void finalize_ssl();

	StreamPeerSSL();
};

#endif
#endif // STREAM_PEER_SSL_H
