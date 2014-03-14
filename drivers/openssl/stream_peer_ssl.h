#ifndef STREAM_PEER_SSL_H
#define STREAM_PEER_SSL_H

#include "io/stream_peer.h"

class StreamPeerSSL : public StreamPeer {

	OBJ_TYPE(StreamPeerSSL,StreamPeer);

	Ref<StreamPeer> base;
	bool block;
	static BIO_METHOD bio_methods;

	static int bio_create( BIO *b );
	static int bio_destroy( BIO *b );
	static int bio_read( BIO *b, char *buf, int len );
	static int bio_write( BIO *b, const char *buf, int len );
	static long bio_ctrl( BIO *b, int cmd, long num, void *ptr );
	static int bio_gets( BIO *b, char *buf, int len );
	static int bio_puts( BIO *b, const char *str );

public:
	StreamPeerSSL();
};

#endif // STREAM_PEER_SSL_H
