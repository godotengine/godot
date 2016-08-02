#ifndef NETWORKED_MULTIPLAYER_PEER_H
#define NETWORKED_MULTIPLAYER_PEER_H

#include "io/packet_peer.h"

class NetworkedMultiplayerPeer : public PacketPeer {

	OBJ_TYPE(NetworkedMultiplayerPeer,PacketPeer);

protected:
	static void _bind_methods();
public:

	enum {
		TARGET_ALL_PEERS=0xFFFFFF // send to this for all peers
	};

	enum TransferMode {
		TRANSFER_MODE_UNRELIABLE,
		TRANSFER_MODE_RELIABLE,
		TRANSFER_MODE_ORDERED
	};

	virtual void set_transfer_mode(TransferMode p_mode)=0;
	virtual void set_target_peer(int p_peer)=0;
	virtual void set_channel(int p_channel)=0;


	virtual int get_packet_peer() const=0;
	virtual int get_packet_channel() const=0;


	virtual void poll()=0;

	NetworkedMultiplayerPeer();
};

VARIANT_ENUM_CAST( NetworkedMultiplayerPeer::TransferMode )

#endif // NetworkedMultiplayerPeer_H
