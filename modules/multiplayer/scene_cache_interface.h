/**************************************************************************/
/*  scene_cache_interface.h                                               */
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

#ifndef SCENE_CACHE_INTERFACE_H
#define SCENE_CACHE_INTERFACE_H

#include "scene/main/multiplayer_api.h"

class Node;
class SceneMultiplayer;

class SceneCacheInterface : public RefCounted {
	GDCLASS(SceneCacheInterface, RefCounted);

private:
	SceneMultiplayer *multiplayer = nullptr;

	//path sent caches
	struct PathSentCache {
		HashMap<int, bool> confirmed_peers;
		int id;
	};

	//path get caches
	struct PathGetCache {
		struct NodeInfo {
			NodePath path;
			ObjectID instance;
		};

		HashMap<int, NodeInfo> nodes;
	};

	HashMap<NodePath, PathSentCache> path_send_cache;
	HashMap<int, PathGetCache> path_get_cache;
	int last_send_cache_id = 1;

protected:
	Error _send_confirm_path(Node *p_node, NodePath p_path, PathSentCache *psc, const List<int> &p_peers);

public:
	void clear();
	void on_peer_change(int p_id, bool p_connected);
	void process_simplify_path(int p_from, const uint8_t *p_packet, int p_packet_len);
	void process_confirm_path(int p_from, const uint8_t *p_packet, int p_packet_len);

	// Returns true if all peers have cached path.
	bool send_object_cache(Object *p_obj, int p_target, int &p_id);
	int make_object_cache(Object *p_obj);
	Object *get_cached_object(int p_from, uint32_t p_cache_id);
	bool is_cache_confirmed(NodePath p_path, int p_peer);

	SceneCacheInterface(SceneMultiplayer *p_multiplayer) { multiplayer = p_multiplayer; }
};

#endif // SCENE_CACHE_INTERFACE_H
