/**************************************************************************/
/*  scene_multiplayer_metadata.h                                          */
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

#include "core/object/ref_counted.h"
#include "core/templates/rb_set.h"

class SceneMultiplayer;

class SceneMultiplayerMetadata : public RefCounted {
	GDCLASS(SceneMultiplayerMetadata, RefCounted);

private:
	SceneMultiplayer *multiplayer = nullptr;

	HashMap<String, Variant> lobby_properties;
	struct PeerProps {
		HashMap<String, Variant> properties;
	};
	HashSet<String> peer_prop_names;
	HashMap<int, PeerProps> my_peer_props;
	HashMap<String, Variant> local_peer_props;

	void _send_replicate_lobby_property(String p_prop_name, int peer_id = 0);
	void _send_replicate_peer_property(String p_prop_name, int peer_id = 0);

public:
	void add_lobby_property(String p_prop_name);
	void add_peer_property(String p_prop_name);
	bool set_lobby_property(String p_prop_name, Variant value);
	bool set_peer_property(String p_prop_name, Variant value, int peer_id);
	Variant get_lobby_property(String p_prop_name);
	Variant get_peer_property(String p_prop_name, int peer_id);

	void clear();
	void on_peer_change(int p_id, bool p_connected);

	SceneMultiplayerMetadata(SceneMultiplayer *p_multiplayer) {
		multiplayer = p_multiplayer;
	}
};
