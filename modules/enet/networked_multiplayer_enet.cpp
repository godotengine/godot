#include "networked_multiplayer_enet.h"

void NetworkedMultiplayerENet::set_transfer_mode(TransferMode p_mode) {

	transfer_mode=p_mode;
}

void NetworkedMultiplayerENet::set_target_peer(const StringName &p_peer){

	target_peer=p_peer;
}
void NetworkedMultiplayerENet::set_channel(int p_channel){

	send_channel=p_channel;
}


StringName NetworkedMultiplayerENet::get_packet_peer() const{

	ERR_FAIL_COND_V(!active,StringName());
	ERR_FAIL_COND_V(incoming_packets.size()==0,StringName());

	return incoming_packets.front()->get().from;

}
int NetworkedMultiplayerENet::get_packet_channel() const{

	ERR_FAIL_COND_V(!active,0);
	ERR_FAIL_COND_V(incoming_packets.size()==0,0);
	return incoming_packets.front()->get().from_channel;
}


Error NetworkedMultiplayerENet::create_server(int p_port, int p_max_clients, int p_max_channels, int p_in_bandwidth, int p_out_bandwidth){

	ERR_FAIL_COND_V(active,ERR_ALREADY_IN_USE);

	ENetAddress address;
	address.host = ENET_HOST_ANY;
	/* Bind the server to port 1234. */
	address.port = 1234;

	host = enet_host_create (& address /* the address to bind the server host to */,
				     p_max_clients      /* allow up to 32 clients and/or outgoing connections */,
				     p_max_channels      /* allow up to 2 channels to be used, 0 and 1 */,
				     p_in_bandwidth      /* assume any amount of incoming bandwidth */,
				     p_out_bandwidth      /* assume any amount of outgoing bandwidth */);

	ERR_FAIL_COND_V(!host,ERR_CANT_CREATE);

	active=true;
	server=true;
	connection_status=CONNECTION_CONNECTED;
	return OK;
}
Error NetworkedMultiplayerENet::create_client(const IP_Address& p_ip,int p_port, int p_max_channels, int p_in_bandwidth, int p_out_bandwidth){

	ERR_FAIL_COND_V(active,ERR_ALREADY_IN_USE);

	host = enet_host_create (NULL /* create a client host */,
		    1 /* only allow 1 outgoing connection */,
		    p_max_channels /* allow up 2 channels to be used, 0 and 1 */,
		    p_in_bandwidth /* 56K modem with 56 Kbps downstream bandwidth */,
		    p_out_bandwidth /* 56K modem with 14 Kbps upstream bandwidth */);

	ERR_FAIL_COND_V(!host,ERR_CANT_CREATE);


	ENetAddress address;
	address.host=p_ip.host;
	address.port=p_port;

	/* Initiate the connection, allocating the two channels 0 and 1. */
	ENetPeer *peer = enet_host_connect (host, & address, p_max_channels, 0);

	if (peer == NULL) {
		enet_host_destroy(host);
		ERR_FAIL_COND_V(!peer,ERR_CANT_CREATE);
	}

	//technically safe to ignore the peer or anything else.

	connection_status=CONNECTION_CONNECTING;

	return OK;
}

void NetworkedMultiplayerENet::poll(){

	ERR_FAIL_COND(!active);

	_pop_current_packet();

	ENetEvent event;
	/* Wait up to 1000 milliseconds for an event. */
	while (enet_host_service (host, & event, 1000) > 0)
	{
		switch (event.type)
		{
			case ENET_EVENT_TYPE_CONNECT: {
				/* Store any relevant client information here. */

				IP_Address ip;
				ip.host=event.peer -> address.host;

				StringName *new_id = memnew( StringName );
				*new_id = String(ip) +":"+ itos(event.peer -> address.port);

				peer_map[*new_id]=event.peer;

				connection_status=CONNECTION_CONNECTED; //if connecting, this means it connected t something!

				emit_signal("peer_connected",*new_id);

			} break;
			case ENET_EVENT_TYPE_DISCONNECT: {

				/* Reset the peer's client information. */

				StringName *id = (StringName*)event.peer -> data;

				emit_signal("peer_disconnected",*id);

				peer_map.erase(*id);
				memdelete( id );
			} break;
			case ENET_EVENT_TYPE_RECEIVE: {

				Packet packet;
				packet.packet = event.packet;

				StringName *id = (StringName*)event.peer -> data;
				packet.from_channel=event.channelID;
				packet.from=*id;

				incoming_packets.push_back(packet);
				//destroy packet later..

			}break;
			case ENET_EVENT_TYPE_NONE: {
				//do nothing
			} break;
		}
	}
}

void NetworkedMultiplayerENet::disconnect() {

	ERR_FAIL_COND(!active);

	_pop_current_packet();

	enet_host_destroy(host);
	active=false;
	incoming_packets.clear();

	connection_status=CONNECTION_DISCONNECTED;
}

int NetworkedMultiplayerENet::get_available_packet_count() const {

	return incoming_packets.size();
}
Error NetworkedMultiplayerENet::get_packet(const uint8_t **r_buffer,int &r_buffer_size) const{

	ERR_FAIL_COND_V(incoming_packets.size()==0,ERR_UNAVAILABLE);

	_pop_current_packet();

	current_packet = incoming_packets.front()->get();
	incoming_packets.pop_front();

	r_buffer=(const uint8_t**)&current_packet.packet->data;
	r_buffer_size=current_packet.packet->dataLength;

	return OK;
}
Error NetworkedMultiplayerENet::put_packet(const uint8_t *p_buffer,int p_buffer_size){

	ERR_FAIL_COND_V(incoming_packets.size()==0,ERR_UNAVAILABLE);

	Map<StringName,ENetPeer*>::Element *E=NULL;

	if (target_peer!=StringName()) {
		peer_map.find(target_peer);
		if (!E) {
			ERR_EXPLAIN("Invalid Target Peer: "+String(target_peer));
			ERR_FAIL_V(ERR_INVALID_PARAMETER);
		}
	}

	int packet_flags=0;
	switch(transfer_mode) {
		case TRANSFER_MODE_UNRELIABLE: {
			packet_flags=ENET_PACKET_FLAG_UNSEQUENCED;
		} break;
		case TRANSFER_MODE_RELIABLE: {
			packet_flags=ENET_PACKET_FLAG_RELIABLE;
		} break;
		case TRANSFER_MODE_ORDERED: {
			packet_flags=ENET_PACKET_FLAG_RELIABLE;
		} break;
	}

	/* Create a reliable packet of size 7 containing "packet\0" */
	ENetPacket * packet = enet_packet_create (p_buffer,p_buffer_size,packet_flags);

	if (target_peer==StringName()) {
		enet_host_broadcast(host,send_channel,packet);
	} else {
		enet_peer_send (E->get(), send_channel, packet);
	}

	enet_host_flush(host);

	return OK;
}

int NetworkedMultiplayerENet::get_max_packet_size() const {

	return 1<<24; //anything is good
}

void NetworkedMultiplayerENet::_pop_current_packet() const {

	if (current_packet.packet) {
		enet_packet_destroy(current_packet.packet);
		current_packet.packet=NULL;
		current_packet.from=StringName();
	}

}

NetworkedMultiplayerPeer::ConnectionStatus NetworkedMultiplayerENet::get_connection_status() const {

	return connection_status;
}

void NetworkedMultiplayerENet::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("create_server","port","max_clients","max_channels","in_bandwidth","out_bandwidth"),&NetworkedMultiplayerENet::create_server,DEFVAL(32),DEFVAL(1),DEFVAL(0),DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("create_client","ip","port","max_channels","in_bandwidth","out_bandwidth"),&NetworkedMultiplayerENet::create_client,DEFVAL(1),DEFVAL(0),DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("disconnect"),&NetworkedMultiplayerENet::disconnect);

}


NetworkedMultiplayerENet::NetworkedMultiplayerENet(){

	active=false;
	server=false;
	send_channel=0;
	current_packet.packet=NULL;
	transfer_mode=TRANSFER_MODE_ORDERED;
	connection_status=CONNECTION_DISCONNECTED;
}

NetworkedMultiplayerENet::~NetworkedMultiplayerENet(){

	if (active) {
		disconnect();
	}
}
