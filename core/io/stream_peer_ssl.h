#ifndef STREAM_PEER_SSL_H
#define STREAM_PEER_SSL_H

#include "io/stream_peer.h"

class StreamPeerSSL : public StreamPeer {
	OBJ_TYPE(StreamPeerSSL,StreamPeer);
protected:
	static StreamPeerSSL* (*_create)();
	static void _bind_methods();
public:

	enum Status {
		STATUS_DISCONNECTED,
		STATUS_CONNECTED,
		STATUS_ERROR_NO_CERTIFICATE,
		STATUS_ERROR_HOSTNAME_MISMATCH
	};

	virtual Error accept(Ref<StreamPeer> p_base)=0;
	virtual Error connect(Ref<StreamPeer> p_base,bool p_validate_certs=false,const String& p_for_hostname=String())=0;
	virtual Status get_status() const=0;

	virtual void disconnect()=0;

	static StreamPeerSSL* create();


	StreamPeerSSL();
};

VARIANT_ENUM_CAST( StreamPeerSSL::Status );

#endif // STREAM_PEER_SSL_H
