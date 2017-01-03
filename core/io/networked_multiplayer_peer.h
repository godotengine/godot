#ifndef NETWORKED_MULTIPLAYER_PEER_H
#define NETWORKED_MULTIPLAYER_PEER_H

#include "io/packet_peer.h"

class NetworkedMultiplayerPeer : public PacketPeer {

	GDCLASS(NetworkedMultiplayerPeer,PacketPeer);

protected:
	static void _bind_methods();
public:

	enum {
		TARGET_PEER_BROADCAST=0,
		TARGET_PEER_SERVER=1
	};
	enum TransferMode {
		TRANSFER_MODE_UNRELIABLE,
		TRANSFER_MODE_UNRELIABLE_ORDERED,
		TRANSFER_MODE_RELIABLE,
	};

	enum ConnectionStatus {
		CONNECTION_DISCONNECTED,
		CONNECTION_CONNECTING,
		CONNECTION_CONNECTED,
	};


	virtual void set_transfer_mode(TransferMode p_mode)=0;
	virtual void set_target_peer(int p_peer_id)=0;

	virtual int get_packet_peer() const=0;

	virtual bool is_server() const=0;

	virtual void poll()=0;

	virtual int get_unique_id() const=0;

	virtual void set_refuse_new_connections(bool p_enable)=0;
	virtual bool is_refusing_new_connections() const=0;


	virtual ConnectionStatus get_connection_status() const=0;

	NetworkedMultiplayerPeer();
};

VARIANT_ENUM_CAST( NetworkedMultiplayerPeer::TransferMode )
VARIANT_ENUM_CAST( NetworkedMultiplayerPeer::ConnectionStatus )

#endif // NetworkedMultiplayerPeer_H
