/*************************************************************************/
/*  rpc_manager.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef MULTIPLAYER_RPC_H
#define MULTIPLAYER_RPC_H

#include "core/multiplayer/multiplayer.h"
#include "core/multiplayer/multiplayer_api.h"
#include "core/object/ref_counted.h"

class RPCManager : public RefCounted {
	GDCLASS(RPCManager, RefCounted);

private:
	enum NetworkNodeIdCompression {
		NETWORK_NODE_ID_COMPRESSION_8 = 0,
		NETWORK_NODE_ID_COMPRESSION_16,
		NETWORK_NODE_ID_COMPRESSION_32,
	};

	enum NetworkNameIdCompression {
		NETWORK_NAME_ID_COMPRESSION_8 = 0,
		NETWORK_NAME_ID_COMPRESSION_16,
	};

	// The RPC meta is composed by a single byte that contains (starting from the least significant bit):
	// - `NetworkCommands` in the first four bits.
	// - `NetworkNodeIdCompression` in the next 2 bits.
	// - `NetworkNameIdCompression` in the next 1 bit.
	// - `byte_only_or_no_args` in the next 1 bit.
	enum {
		NODE_ID_COMPRESSION_SHIFT = MultiplayerAPI::CMD_FLAG_0_SHIFT, // 2 bits for this.
		NAME_ID_COMPRESSION_SHIFT = MultiplayerAPI::CMD_FLAG_2_SHIFT,
		BYTE_ONLY_OR_NO_ARGS_SHIFT = MultiplayerAPI::CMD_FLAG_3_SHIFT,
	};

	enum {
		NODE_ID_COMPRESSION_FLAG = (1 << NODE_ID_COMPRESSION_SHIFT) | (1 << (NODE_ID_COMPRESSION_SHIFT + 1)), // 2 bits for this.
		NAME_ID_COMPRESSION_FLAG = (1 << NAME_ID_COMPRESSION_SHIFT),
		BYTE_ONLY_OR_NO_ARGS_FLAG = (1 << BYTE_ONLY_OR_NO_ARGS_SHIFT),
	};

	MultiplayerAPI *multiplayer = nullptr;
	Vector<uint8_t> packet_cache;

protected:
	_FORCE_INLINE_ void _profile_node_data(const String &p_what, ObjectID p_id);
	void _process_rpc(Node *p_node, const uint16_t p_rpc_method_id, int p_from, const uint8_t *p_packet, int p_packet_len, int p_offset);

	void _send_rpc(Node *p_from, int p_to, uint16_t p_rpc_id, const Multiplayer::RPCConfig &p_config, const StringName &p_name, const Variant **p_arg, int p_argcount);
	Node *_process_get_node(int p_from, const uint8_t *p_packet, uint32_t p_node_target, int p_packet_len);

public:
	// Called by Node.rpc
	void rpcp(Node *p_node, int p_peer_id, const StringName &p_method, const Variant **p_arg, int p_argcount);
	void process_rpc(int p_from, const uint8_t *p_packet, int p_packet_len);

	String get_rpc_md5(const Node *p_node);
	RPCManager(MultiplayerAPI *p_multiplayer) { multiplayer = p_multiplayer; }
};

#endif // MULTIPLAYER_RPC_H
