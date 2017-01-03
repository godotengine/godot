#include "networked_multiplayer_peer.h"


void NetworkedMultiplayerPeer::_bind_methods() {

	ClassDB::bind_method(_MD("set_transfer_mode","mode"), &NetworkedMultiplayerPeer::set_transfer_mode );
	ClassDB::bind_method(_MD("set_target_peer","id"), &NetworkedMultiplayerPeer::set_target_peer );

	ClassDB::bind_method(_MD("get_packet_peer"), &NetworkedMultiplayerPeer::get_packet_peer );

	ClassDB::bind_method(_MD("poll"), &NetworkedMultiplayerPeer::poll );

	ClassDB::bind_method(_MD("get_connection_status"), &NetworkedMultiplayerPeer::get_connection_status );
	ClassDB::bind_method(_MD("get_unique_id"), &NetworkedMultiplayerPeer::get_unique_id );

	ClassDB::bind_method(_MD("set_refuse_new_connections","enable"), &NetworkedMultiplayerPeer::set_refuse_new_connections );
	ClassDB::bind_method(_MD("is_refusing_new_connections"), &NetworkedMultiplayerPeer::is_refusing_new_connections );

	BIND_CONSTANT( TRANSFER_MODE_UNRELIABLE );
	BIND_CONSTANT( TRANSFER_MODE_UNRELIABLE_ORDERED );
	BIND_CONSTANT( TRANSFER_MODE_RELIABLE );

	BIND_CONSTANT( CONNECTION_DISCONNECTED );
	BIND_CONSTANT( CONNECTION_CONNECTING );
	BIND_CONSTANT( CONNECTION_CONNECTED );

	BIND_CONSTANT( TARGET_PEER_BROADCAST );
	BIND_CONSTANT( TARGET_PEER_SERVER );


	ADD_SIGNAL( MethodInfo("peer_connected",PropertyInfo(Variant::INT,"id")));
	ADD_SIGNAL( MethodInfo("peer_disconnected",PropertyInfo(Variant::INT,"id")));
	ADD_SIGNAL( MethodInfo("server_disconnected"));
	ADD_SIGNAL( MethodInfo("connection_succeeded") );
	ADD_SIGNAL( MethodInfo("connection_failed") );
}

NetworkedMultiplayerPeer::NetworkedMultiplayerPeer() {


}
