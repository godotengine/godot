#ifndef NETWORKED_MULTIPLAYER_ENET_H
#define NETWORKED_MULTIPLAYER_ENET_H

#include "io/networked_multiplayer_peer.h"
#include "enet/enet.h"

class NetworkedMultiplayerENet : public NetworkedMultiplayerPeer {

	OBJ_TYPE(NetworkedMultiplayerENet,NetworkedMultiplayerPeer)

	bool active;
	bool server;

	int send_channel;
	StringName target_peer;
	TransferMode transfer_mode;

	ENetEvent event;
	ENetPeer *peer;
	ENetHost *host;

	ConnectionStatus connection_status;

	Map<StringName,ENetPeer*> peer_map;

	struct Packet {

		ENetPacket *packet;
		int from_channel;
		StringName from;
	};

	mutable List<Packet> incoming_packets;

	mutable Packet current_packet;

	void _pop_current_packet() const;

protected:
	static void _bind_methods();
public:

	virtual void set_transfer_mode(TransferMode p_mode);
	virtual void set_target_peer(const StringName& p_peer);
	virtual void set_channel(int p_channel);


	virtual StringName get_packet_peer() const;
	virtual int get_packet_channel() const;


	Error create_server(int p_port, int p_max_clients=32, int p_max_channels=1, int p_in_bandwidth=0, int p_out_bandwidth=0);
	Error create_client(const IP_Address& p_ip,int p_port, int p_max_channels=1, int p_in_bandwidth=0, int p_out_bandwidth=0);

	void disconnect();

	virtual void poll();

	virtual int get_available_packet_count() const;
	virtual Error get_packet(const uint8_t **r_buffer,int &r_buffer_size) const; ///< buffer is GONE after next get_packet
	virtual Error put_packet(const uint8_t *p_buffer,int p_buffer_size);

	virtual int get_max_packet_size() const;

	virtual ConnectionStatus get_connection_status() const;

	NetworkedMultiplayerENet();
	~NetworkedMultiplayerENet();
};


#endif // NETWORKED_MULTIPLAYER_ENET_H
