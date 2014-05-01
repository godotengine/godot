#ifdef OPENSSL_ENABLED

#include "register_openssl.h"

#include "stream_peer_openssl.h"

void register_openssl() {

	ObjectTypeDB::register_type<StreamPeerOpenSSL>();
	StreamPeerOpenSSL::initialize_ssl();

}

void unregister_openssl() {

	StreamPeerOpenSSL::finalize_ssl();

}

#endif
