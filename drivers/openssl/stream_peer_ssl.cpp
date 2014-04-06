#include "stream_peer_ssl.h"


int StreamPeerSSL::bio_create( BIO *b ) {
	b->init = 1;
	b->num = 0;
	b->ptr = NULL;
	b->flags = 0;
	return 1;
}

int StreamPeerSSL::bio_destroy( BIO *b ) {

	if ( b == NULL ) return 0;
	b->ptr = NULL;		/* sb_tls_remove() will free it */
	b->init = 0;
	b->flags = 0;
	return 1;
}

int StreamPeerSSL::bio_read( BIO *b, char *buf, int len ) {

	if ( buf == NULL || len <= 0 ) return 0;

	StreamPeerSSL * sp = (StreamPeerSSL*)b->ptr;

	if (sp->base.is_null())
		return 0;



	BIO_clear_retry_flags( b );

	Error err;
	int ret=0;
	if (sp->block) {
		err = sp->base->get_data((const uint8_t*)buf,len);
		if (err==OK)
			ret=len;
	} else {

		err = sp->base->get_partial_data((const uint8_t*)buf,len,ret);
		if (err==OK && ret!=len) {
			BIO_set_retry_write( b );
		}

	}

	return ret;
}

int StreamPeerSSL::bio_write( BIO *b, const char *buf, int len ) {

	if ( buf == NULL || len <= 0 ) return 0;

	StreamPeerSSL * sp = (StreamPeerSSL*)b->ptr;

	if (sp->base.is_null())
		return 0;

	BIO_clear_retry_flags( b );

	Error err;
	int wrote=0;
	if (sp->block) {
		err = sp->base->put_data((const uint8_t*)buf,len);
		if (err==OK)
			wrote=len;
	} else {

		err = sp->base->put_partial_data((const uint8_t*)buf,len,wrote);
		if (err==OK && wrote!=len) {
			BIO_set_retry_write( b );
		}

	}

	return wrote;
}

long StreamPeerSSL::bio_ctrl( BIO *b, int cmd, long num, void *ptr ) {
	if ( cmd == BIO_CTRL_FLUSH ) {
		/* The OpenSSL library needs this */
		return 1;
	}
	return 0;
}

int StreamPeerSSL::bio_gets( BIO *b, char *buf, int len ) {
	return -1;
}

int StreamPeerSSL::bio_puts( BIO *b, const char *str ) {
	return StreamPeerSSL::bio_write( b, str, strlen( str ) );
}

BIO_METHOD StreamPeerSSL::bio_methods =
{
	( 100 | 0x400 ),		/* it's a source/sink BIO */
	"sockbuf glue",
	StreamPeerSSL::bio_write,
	StreamPeerSSL::bio_read,
	StreamPeerSSL::bio_puts,
	StreamPeerSSL::bio_gets,
	StreamPeerSSL::bio_ctrl,
	StreamPeerSSL::bio_create,
	StreamPeerSSL::bio_destroy
};

StreamPeerSSL::StreamPeerSSL() {
}
