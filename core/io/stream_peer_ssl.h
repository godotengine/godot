#ifndef STREAM_PEER_SSL_H
#define STREAM_PEER_SSL_H

#include "io/stream_peer.h"

class StreamPeerSSL : public StreamPeer {
	OBJ_TYPE(StreamPeerSSL,StreamPeer);
public:

	typedef void (*LoadCertsFromMemory)(const ByteArray& p_certs);
protected:
	static StreamPeerSSL* (*_create)();
	static void _bind_methods();

	static LoadCertsFromMemory load_certs_func;
	static bool available;


friend class Main;
	static bool initialize_certs;

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

	static void load_certs_from_memory(const ByteArray& p_memory);
	static bool is_available();

	StreamPeerSSL();
};

VARIANT_ENUM_CAST( StreamPeerSSL::Status );

#endif // STREAM_PEER_SSL_H
