#include "packet_peer_udp_winsock.h"

#include <winsock2.h>

int PacketPeerUDPWinsock::get_available_packet_count() const {

	Error err = const_cast<PacketPeerUDPWinsock*>(this)->_poll(false);
	if (err!=OK)
		return 0;

	return queue_count;
}

Error PacketPeerUDPWinsock::get_packet(const uint8_t **r_buffer,int &r_buffer_size) const{

	Error err = const_cast<PacketPeerUDPWinsock*>(this)->_poll(false);
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
Error PacketPeerUDPWinsock::put_packet(const uint8_t *p_buffer,int p_buffer_size){

	int sock = _get_socket();
	ERR_FAIL_COND_V( sock == -1, FAILED );
	struct sockaddr_in addr;
	addr.sin_family = AF_INET;
	addr.sin_port = htons(peer_port);
	addr.sin_addr = *((struct in_addr*)&peer_addr.host);


	_set_blocking(true);

	errno = 0;
	int err;
	while ( (err = sendto(sock, (const char*)p_buffer, p_buffer_size, 0, (struct sockaddr*)&addr, sizeof(addr))) != p_buffer_size) {

		if (WSAGetLastError() != WSAEWOULDBLOCK) {
			return FAILED;
		};
	}

	return OK;
}

int PacketPeerUDPWinsock::get_max_packet_size() const{

	return 512; // uhm maybe not
}


void PacketPeerUDPWinsock::_set_blocking(bool p_blocking) {
	//am no windows expert
	//hope this is the right thing

	if (blocking==p_blocking)
		return;

	blocking=p_blocking;
	unsigned long par = blocking?0:1;
	if (ioctlsocket(sockfd, FIONBIO, &par)) {
		perror("setting non-block mode");
		//close();
		//return -1;
	};
}

Error PacketPeerUDPWinsock::listen(int p_port, int p_recv_buffer_size){

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

	blocking=true;

	printf("UDP Connection listening on port %i\n", p_port);
	rb.resize(nearest_shift(p_recv_buffer_size));
	return OK;
}

void PacketPeerUDPWinsock::close(){

	if (sockfd != -1)
		::closesocket(sockfd);
	sockfd=-1;
	rb.resize(8);
	queue_count=0;
}


Error PacketPeerUDPWinsock::wait() {

	return _poll(true);
}
Error PacketPeerUDPWinsock::_poll(bool p_wait) {


	_set_blocking(p_wait);


	struct sockaddr_in from = {0};
	int len = sizeof(struct sockaddr_in);
	int ret;
	while ( (ret = recvfrom(sockfd, (char*)recv_buffer, MIN(sizeof(recv_buffer),rb.data_left()-12), 0, (struct sockaddr*)&from, &len)) > 0) {
		rb.write((uint8_t*)&from.sin_addr, 4);
		uint32_t port = ntohs(from.sin_port);
		rb.write((uint8_t*)&port, 4);
		rb.write((uint8_t*)&ret, 4);
		rb.write(recv_buffer, ret);

		len = sizeof(struct sockaddr_in);
		++queue_count;
	};


	if (ret == 0 || (ret == SOCKET_ERROR && WSAGetLastError() != WSAEWOULDBLOCK) ) {
		close();
		return FAILED;
	};


	return OK;
}

bool PacketPeerUDPWinsock::is_listening() const{

	return sockfd!=-1;
}

IP_Address PacketPeerUDPWinsock::get_packet_address() const {

	return packet_ip;
}

int PacketPeerUDPWinsock::get_packet_port() const{

	return packet_port;
}

int PacketPeerUDPWinsock::_get_socket() {

	if (sockfd != -1)
		return sockfd;

	sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	ERR_FAIL_COND_V( sockfd == -1, -1 );
	//fcntl(sockfd, F_SETFL, O_NONBLOCK);

	return sockfd;
}


void PacketPeerUDPWinsock::set_send_address(const IP_Address& p_address,int p_port) {

	peer_addr=p_address;
	peer_port=p_port;
}

void PacketPeerUDPWinsock::make_default() {

	PacketPeerUDP::_create = PacketPeerUDPWinsock::_create;
};


PacketPeerUDP* PacketPeerUDPWinsock::_create() {

	return memnew(PacketPeerUDPWinsock);
};


PacketPeerUDPWinsock::PacketPeerUDPWinsock() {

	sockfd=-1;
	packet_port=0;
	queue_count=0;
	peer_port=0;
}

PacketPeerUDPWinsock::~PacketPeerUDPWinsock() {

	close();
}
