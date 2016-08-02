#include "networked_multiplayer_peer.h"


void NetworkedMultiplayerPeer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_transfer_mode","mode"), &NetworkedMultiplayerPeer::set_transfer_mode );
	ObjectTypeDB::bind_method(_MD("set_target_peer","id"), &NetworkedMultiplayerPeer::set_target_peer );
	ObjectTypeDB::bind_method(_MD("set_channel","id"), &NetworkedMultiplayerPeer::set_channel );

	ObjectTypeDB::bind_method(_MD("get_packet_peer"), &NetworkedMultiplayerPeer::get_packet_peer );
	ObjectTypeDB::bind_method(_MD("get_packet_channel"), &NetworkedMultiplayerPeer::get_packet_channel );

	ObjectTypeDB::bind_method(_MD("poll"), &NetworkedMultiplayerPeer::poll );


	BIND_CONSTANT( TARGET_ALL_PEERS );

	BIND_CONSTANT( TRANSFER_MODE_UNRELIABLE );
	BIND_CONSTANT( TRANSFER_MODE_RELIABLE );
	BIND_CONSTANT( TRANSFER_MODE_ORDERED );

	ADD_SIGNAL( MethodInfo("peer_connected",PropertyInfo(Variant::INT,"id")));
	ADD_SIGNAL( MethodInfo("peer_disconnected",PropertyInfo(Variant::INT,"id")));
}

NetworkedMultiplayerPeer::NetworkedMultiplayerPeer() {


}
