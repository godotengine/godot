#ifndef PACKET_PEER_UDP_H
#define PACKET_PEER_UDP_H


#include "io/packet_peer.h"

class PacketPeerUDP : public PacketPeer {
	OBJ_TYPE(PacketPeerUDP,PacketPeer);

protected:

	static PacketPeerUDP* (*_create)();
	static void _bind_methods();

	int _get_packet_address() const;
	String _get_packet_ip() const;

	virtual Error _set_send_address(const String& p_address,int p_port);

public:

	virtual Error listen(int p_port,int p_recv_buffer_size=65536)=0;
	virtual void close()=0;
	virtual Error wait()=0;
	virtual bool is_listening() const=0;
	virtual IP_Address get_packet_address() const=0;
	virtual int get_packet_port() const=0;
	virtual void set_send_address(const IP_Address& p_address,int p_port)=0;


	static Ref<PacketPeerUDP> create_ref();
	static PacketPeerUDP* create();

	PacketPeerUDP();
};

#endif // PACKET_PEER_UDP_H
