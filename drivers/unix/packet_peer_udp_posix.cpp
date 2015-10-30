#include "packet_peer_udp_posix.h"

#ifdef UNIX_ENABLED


#include <errno.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>

#include <netinet/in.h>
#include <stdio.h>

#ifndef NO_FCNTL
	#ifdef __HAIKU__
		#include <fcntl.h>
	#else
		#include <sys/fcntl.h>
	#endif
#else
#include <sys/ioctl.h>
#endif

#ifdef JAVASCRIPT_ENABLED
#include <arpa/inet.h>
#endif


int PacketPeerUDPPosix::get_available_packet_count() const {

	Error err = const_cast<PacketPeerUDPPosix*>(this)->_poll(false);
	if (err!=OK)
		return 0;

	return queue_count;
}

Error PacketPeerUDPPosix::get_packet(const uint8_t **r_buffer,int &r_buffer_size) const{

	Error err = const_cast<PacketPeerUDPPosix*>(this)->_poll(false);
	if (err!=OK)
		return err;
	if (queue_count==0)
		return ERR_UNAVAILABLE;

	uint32_t size;
	rb.read((uint8_t*)&packet_ip.host,4,true);
	rb.read((uint8_t*)&packet_port,4,true);
	rb.read((uint8_t*)&size,4,true);
	rb.read(packet_buffer,size,true);
	--queue_count;
	*r_buffer=packet_buffer;
	r_buffer_size=size;
	return OK;

}
Error PacketPeerUDPPosix::put_packet(const uint8_t *p_buffer,int p_buffer_size){

	int sock = _get_socket();
	ERR_FAIL_COND_V( sock == -1, FAILED );
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(peer_port);
	addr.sin_addr = *((struct in_addr*)&peer_addr.host);

	errno = 0;
	int err;

	while ( (err = sendto(sock, p_buffer, p_buffer_size, 0, (struct sockaddr*)&addr, sizeof(addr))) != p_buffer_size) {

		if (errno != EAGAIN) {
			return FAILED;
		}
	}

	return OK;
}

int PacketPeerUDPPosix::get_max_packet_size() const{

	return 512; // uhm maybe not
}

Error PacketPeerUDPPosix::listen(int p_port, int p_recv_buffer_size){

	close();
	int sock = _get_socket();
	if (sock == -1 )
		return ERR_CANT_CREATE;
	sockaddr_in addr = {0};
	addr.sin_family = AF_INET;
	addr.sin_port = htons(p_port);
	addr.sin_addr.s_addr = INADDR_ANY;
	if (bind(sock, (struct sockaddr*)&addr, sizeof(sockaddr_in)) == -1 ) {
		close();
		return ERR_UNAVAILABLE;
	}
	printf("UDP Connection listening on port %i  bufsize %i \n", p_port,p_recv_buffer_size);
	rb.resize(nearest_shift(p_recv_buffer_size));
	return OK;
}

void PacketPeerUDPPosix::close(){

	if (sockfd != -1)
		::close(sockfd);
	sockfd=-1;
	rb.resize(8);
	queue_count=0;
}


Error PacketPeerUDPPosix::wait() {

	return _poll(true);
}

Error PacketPeerUDPPosix::_poll(bool p_wait) {

	struct sockaddr_in from = {0};
	socklen_t len = sizeof(struct sockaddr_in);
	int ret;
	while ( (ret = recvfrom(sockfd, recv_buffer, MIN(sizeof(recv_buffer),rb.data_left()-12), p_wait?0:MSG_DONTWAIT, (struct sockaddr*)&from, &len)) > 0) {
		rb.write((uint8_t*)&from.sin_addr, 4);
		uint32_t port = ntohs(from.sin_port);
		rb.write((uint8_t*)&port, 4);
		rb.write((uint8_t*)&ret, 4);
		rb.write(recv_buffer, ret);
		len = sizeof(struct sockaddr_in);
		++queue_count;
	};

	if (ret == 0 || (ret == -1 && errno != EAGAIN) ) {
		close();
		return FAILED;
	};

	return OK;
}
bool PacketPeerUDPPosix::is_listening() const{

	return sockfd!=-1;
}

IP_Address PacketPeerUDPPosix::get_packet_address() const {

	return packet_ip;
}

int PacketPeerUDPPosix::get_packet_port() const{

	return packet_port;
}

int PacketPeerUDPPosix::_get_socket() {

	if (sockfd != -1)
		return sockfd;

	sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	ERR_FAIL_COND_V( sockfd == -1, -1 );
	//fcntl(sockfd, F_SETFL, O_NONBLOCK);

	return sockfd;
}


void PacketPeerUDPPosix::set_send_address(const IP_Address& p_address,int p_port) {

	peer_addr=p_address;
	peer_port=p_port;
}

PacketPeerUDP* PacketPeerUDPPosix::_create() {

	return memnew(PacketPeerUDPPosix);
};

void PacketPeerUDPPosix::make_default() {

	PacketPeerUDP::_create = PacketPeerUDPPosix::_create;
};


PacketPeerUDPPosix::PacketPeerUDPPosix() {

	sockfd=-1;
	packet_port=0;
	queue_count=0;
	peer_port=0;
}

PacketPeerUDPPosix::~PacketPeerUDPPosix() {

	close();
}
#endif
