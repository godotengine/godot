/**************************************************************************/
/*  scene_multiplayer_metadata.cpp                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "scene_multiplayer.h"
#include "scene_replication_interface.h"

#include "core/debugger/engine_debugger.h"
#include "core/io/marshalls.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"

void SceneMultiplayerMetadata::_send_replicate_lobby_property(String p_prop_name, int p_peer_id) {
	// Construct & Send REPLICATE_LOBBY_PROPERTY Packet to a singular peer
	Vector<uint8_t> packet;
	// 2 Bytes Command Info
	// 4 Bytes String Length
	// x Bytes String
	// 4 Bytes Arg Size
	// x Bytes Variant Value
	Variant value = get_lobby_property(p_prop_name);
	int prop_len = encode_cstring(p_prop_name.utf8().get_data(), nullptr);
	int arg_size;
	multiplayer->encode_and_compress_variant(value, nullptr, arg_size, false);

	packet.resize(1 + 1 + 4 + prop_len + 4 + arg_size);
	packet.write[0] = SceneMultiplayer::NETWORK_COMMAND_DATA;
	packet.write[1] = SceneMultiplayer::REPLICATE_LOBBY_PROPERTY;
	encode_uint32(prop_len, &packet.write[2]);
	encode_cstring(p_prop_name.utf8().get_data(), &packet.write[6]);
	encode_uint32(arg_size, &packet.write[6 + prop_len]);
	multiplayer->encode_and_compress_variant(value, &packet.write[6 + prop_len + 4], arg_size, false);

	Ref<MultiplayerPeer> multiplayer_peer = multiplayer->get_multiplayer_peer();
	ERR_FAIL_COND(multiplayer_peer.is_null());

	multiplayer_peer->set_transfer_channel(0);
	multiplayer_peer->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
	// p_peer_id parameter defaults to 0, but can be passed in as another value to target a single peer.
	multiplayer->send_command(p_peer_id, packet.ptr(), packet.size());
}

void SceneMultiplayerMetadata::_send_replicate_peer_property(String p_prop_name, int p_peer_id) {
	// Construct & Send REPLICATE_PEER_PROPERTY Packet to a singular peer
	Vector<uint8_t> packet;
	// 2 Bytes Command Info
	// 4 Bytes String Length
	// x Bytes String
	// 4 Bytes Arg Size
	// x Bytes Variant Value
	Variant value = get_peer_property(p_prop_name, multiplayer->get_unique_id());
	int prop_len = encode_cstring(p_prop_name.utf8().get_data(), nullptr);
	int arg_size;
	multiplayer->encode_and_compress_variant(value, nullptr, arg_size, false);

	packet.resize(1 + 1 + 4 + prop_len + 4 + arg_size);
	packet.write[0] = SceneMultiplayer::NETWORK_COMMAND_DATA;
	packet.write[1] = SceneMultiplayer::REPLICATE_PEER_PROPERTY;
	encode_uint32(prop_len, &packet.write[2]);
	encode_cstring(p_prop_name.utf8().get_data(), &packet.write[6]);
	encode_uint32(arg_size, &packet.write[6 + prop_len]);
	multiplayer->encode_and_compress_variant(value, &packet.write[6 + prop_len + 4], arg_size, false);

	Ref<MultiplayerPeer> multiplayer_peer = multiplayer->get_multiplayer_peer();
	ERR_FAIL_COND(multiplayer_peer.is_null());

	multiplayer_peer->set_transfer_channel(0);
	multiplayer_peer->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
	// p_peer_id parameter defaults to 0, but can be passed in as another value to target a single peer.
	multiplayer->send_command(p_peer_id, packet.ptr(), packet.size());
}

void SceneMultiplayerMetadata::add_lobby_property(String p_prop_name) {
	if (!lobby_properties.has(p_prop_name)) {
		lobby_properties.insert(p_prop_name, "PH");
		if (multiplayer->get_unique_id() == 1) {
			_send_replicate_lobby_property(p_prop_name);
		}
	}
}

void SceneMultiplayerMetadata::add_peer_property(String p_prop_name) {
	if (!peer_prop_names.has(p_prop_name)) {
		peer_prop_names.insert(p_prop_name);
		if (!my_peer_props.has(multiplayer->get_unique_id())) {
			my_peer_props.insert(multiplayer->get_unique_id(), PeerProps());
		}
		for (KeyValue<int, PeerProps> &E : my_peer_props) {
			E.value.properties.insert(p_prop_name, "PH");
		}
		if (local_peer_props.has(p_prop_name)) {
			my_peer_props[multiplayer->get_unique_id()].properties[p_prop_name] = local_peer_props[p_prop_name];
			_send_replicate_peer_property(p_prop_name);
			local_peer_props.erase(p_prop_name);
		}
		if (multiplayer->get_unique_id() == 1) {
			_send_replicate_peer_property(p_prop_name);
		}
	}
}

bool SceneMultiplayerMetadata::set_lobby_property(String p_prop_name, Variant value) {
	if (!lobby_properties.has(p_prop_name)) {
		return false;
	}
	if (lobby_properties[p_prop_name] == value) {
		return false;
	}
	lobby_properties.insert(p_prop_name, value);
	if (multiplayer->get_unique_id() == 1) {
		_send_replicate_lobby_property(p_prop_name);
	}
	return true;
}

bool SceneMultiplayerMetadata::set_peer_property(String p_prop_name, Variant value, int peer_id) {
	if (!peer_prop_names.has(p_prop_name)) {
		if (peer_id == multiplayer->get_unique_id()) {
			local_peer_props.insert(p_prop_name, value);
			return true;
		}
		return false;
	}
	if (!my_peer_props.has(peer_id)) {
		return false;
	}
	if (my_peer_props[peer_id].properties[p_prop_name] == value) {
		return false;
	}
	my_peer_props[peer_id].properties.insert(p_prop_name, value);
	if (multiplayer->get_unique_id() == peer_id) {
		_send_replicate_peer_property(p_prop_name);
	}
	return true;
}

Variant SceneMultiplayerMetadata::get_lobby_property(String p_prop_name) {
	if (!lobby_properties.has(p_prop_name)) {
		return Variant();
	}
	return lobby_properties[p_prop_name];
}

Variant SceneMultiplayerMetadata::get_peer_property(String p_prop_name, int peer_id) {
	if (local_peer_props.has(p_prop_name) && peer_id == multiplayer->get_unique_id()) {
		return local_peer_props[p_prop_name];
	}
	if (!peer_prop_names.has(p_prop_name)) {
		return Variant();
	}
	if (!my_peer_props.has(peer_id)) {
		return Variant();
	}
	return my_peer_props[peer_id].properties[p_prop_name];
}

void SceneMultiplayerMetadata::clear() {
	lobby_properties.clear();
	peer_prop_names.clear();
	my_peer_props.clear();
}

void SceneMultiplayerMetadata::on_peer_change(int p_id, bool p_connected) {
	if (p_connected) {
		my_peer_props.insert(p_id, PeerProps());
		if (multiplayer->get_unique_id() == 1) {
			for (KeyValue<String, Variant> &E : lobby_properties) {
				_send_replicate_lobby_property(E.key, p_id);
			}
		}
		for (String peer_prop : peer_prop_names) {
			my_peer_props[p_id].properties.insert(peer_prop, "PH");
			_send_replicate_peer_property(peer_prop, p_id);
		}
	}
}
