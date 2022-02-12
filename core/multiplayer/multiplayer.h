/*************************************************************************/
/*  multiplayer.h                                                        */
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

#ifndef MULTIPLAYER_H
#define MULTIPLAYER_H

#include "core/variant/binder_common.h"

#include "core/string/string_name.h"

namespace Multiplayer {

enum TransferMode {
	TRANSFER_MODE_UNRELIABLE,
	TRANSFER_MODE_UNRELIABLE_ORDERED,
	TRANSFER_MODE_RELIABLE
};

enum RPCMode {
	RPC_MODE_DISABLED, // No rpc for this method, calls to this will be blocked (default)
	RPC_MODE_ANY_PEER, // Any peer can call this RPC
	RPC_MODE_AUTHORITY, // / Only the node's multiplayer authority (server by default) can call this RPC
};

struct RPCConfig {
	StringName name;
	RPCMode rpc_mode = RPC_MODE_DISABLED;
	bool call_local = false;
	TransferMode transfer_mode = TRANSFER_MODE_RELIABLE;
	int channel = 0;

	bool operator==(RPCConfig const &p_other) const {
		return name == p_other.name;
	}
};

struct SortRPCConfig {
	StringName::AlphCompare compare;
	bool operator()(const RPCConfig &p_a, const RPCConfig &p_b) const {
		return compare(p_a.name, p_b.name);
	}
};

}; // namespace Multiplayer

// This is needed for proper docs generation (i.e. not "Multiplayer."-prefixed).
typedef Multiplayer::RPCMode RPCMode;
typedef Multiplayer::TransferMode TransferMode;

VARIANT_ENUM_CAST(RPCMode);
VARIANT_ENUM_CAST(TransferMode);

#endif // MULTIPLAYER_H
