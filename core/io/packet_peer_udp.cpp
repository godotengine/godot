#include "packet_peer_udp.h"
#include "io/ip.h"


PacketPeerUDP* (*PacketPeerUDP::_create)()=NULL;

int PacketPeerUDP::_get_packet_address() const {

	IP_Address ip = get_packet_address();
	return ip.host;
}

String PacketPeerUDP::_get_packet_ip() const {

	return get_packet_address();
}

Error PacketPeerUDP::_set_send_address(const String& p_address,int p_port) {

	IP_Address ip;
	if (p_address.is_valid_ip_address()) {
		ip=p_address;
	} else {
		ip=IP::get_singleton()->resolve_hostname(p_address);
		if (ip==IP_Address())
			return ERR_CANT_RESOLVE;
	}

	set_send_address(ip,p_port);
	return OK;
}

void PacketPeerUDP::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("listen:Error","port","recv_buf_size"),&PacketPeerUDP::listen,DEFVAL(65536));
	ObjectTypeDB::bind_method(_MD("close"),&PacketPeerUDP::close);
	ObjectTypeDB::bind_method(_MD("wait:Error"),&PacketPeerUDP::wait);
	ObjectTypeDB::bind_method(_MD("is_listening"),&PacketPeerUDP::is_listening);
	ObjectTypeDB::bind_method(_MD("get_packet_ip"),&PacketPeerUDP::_get_packet_ip);
	ObjectTypeDB::bind_method(_MD("get_packet_address"),&PacketPeerUDP::_get_packet_address);
	ObjectTypeDB::bind_method(_MD("get_packet_port"),&PacketPeerUDP::get_packet_port);
	ObjectTypeDB::bind_method(_MD("set_send_address","host","port"),&PacketPeerUDP::_set_send_address);


}

Ref<PacketPeerUDP> PacketPeerUDP::create_ref() {

	if (!_create)
		return Ref<PacketPeerUDP>();
	return Ref<PacketPeerUDP>(_create());
}

PacketPeerUDP* PacketPeerUDP::create() {

	if (!_create)
		return NULL;
	return _create();
}

PacketPeerUDP::PacketPeerUDP()
{
}
