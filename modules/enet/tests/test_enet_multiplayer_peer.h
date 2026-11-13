/**************************************************************************/
/*  test_enet_multiplayer_peer.h                                          */
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

#pragma once

#include "core/object/class_db.h"
#include "scene/main/multiplayer_peer.h"
#include "tests/test_macros.h"

namespace TestENetMultiplayerPeer {

// Uses ClassDB so the test doesn't need the ENet headers, which are only on the module's include path.
TEST_CASE("[ENetMultiplayerPeer] create_server sets the host channel limit") {
	Ref<MultiplayerPeer> peer = Object::cast_to<MultiplayerPeer>(ClassDB::instantiate("ENetMultiplayerPeer"));
	REQUIRE(peer.is_valid());

	SUBCASE("Default channels") {
		REQUIRE_EQ(int(peer->call("create_server", 0, 4)), OK);
		// 0 means the ENet maximum of 255.
		Object *host = peer->call("get_host");
		CHECK_EQ(int(host->call("get_max_channels")), 255);
	}

	SUBCASE("Custom channels") {
		REQUIRE_EQ(int(peer->call("create_server", 0, 4, 3)), OK);
		// 3 user channels plus the 2 reserved system channels.
		Object *host = peer->call("get_host");
		CHECK_EQ(int(host->call("get_max_channels")), 5);
	}

	peer->close();
}

} // namespace TestENetMultiplayerPeer
