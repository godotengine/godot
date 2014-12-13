#ifndef PACKET_PEER_UDP_POSIX_H
#define PACKET_PEER_UDP_POSIX_H

#ifdef UNIX_ENABLED

#include "io/packet_peer_udp.h"
#include "ring_buffer.h"

class PacketPeerUDPPosix : public PacketPeerUDP {


	enum {
		PACKET_BUFFER_SIZE=65536
	};

	mutable RingBuffer<uint8_t> rb;
	uint8_t recv_buffer[PACKET_BUFFER_SIZE];
	mutable uint8_t packet_buffer[PACKET_BUFFER_SIZE];
	IP_Address packet_ip;
	int packet_port;
	mutable int queue_count;
	int sockfd;

	IP_Address peer_addr;
	int peer_port;

	_FORCE_INLINE_ int _get_socket();

	static PacketPeerUDP* _create();
	virtual Error _poll(bool p_block);

public:

	virtual int get_available_packet_count() const;
	virtual Error get_packet(const uint8_t **r_buffer,int &r_buffer_size) const;
	virtual Error put_packet(const uint8_t *p_buffer,int p_buffer_size);

	virtual int get_max_packet_size() const;

	virtual Error listen(int p_port,int p_recv_buffer_size=65536);
	virtual void close();
	virtual Error wait();
	virtual bool is_listening() const;

	virtual IP_Address get_packet_address() const;
	virtual int get_packet_port() const;

	virtual void set_send_address(const IP_Address& p_address,int p_port);

	static void make_default();

	PacketPeerUDPPosix();
	~PacketPeerUDPPosix();
};

#endif // PACKET_PEER_UDP_POSIX_H
#endif
