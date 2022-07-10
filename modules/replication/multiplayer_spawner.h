/*************************************************************************/
/*  multiplayer_spawner.h                                                */
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

#ifndef MULTIPLAYER_SPAWNER_H
#define MULTIPLAYER_SPAWNER_H

#include "scene/main/node.h"

#include "core/templates/local_vector.h"
#include "core/variant/typed_array.h"
#include "scene/resources/packed_scene.h"

class MultiplayerSpawner : public Node {
	GDCLASS(MultiplayerSpawner, Node);

public:
	enum {
		CUSTOM_SCENE_INDEX = 0xFFFFFFFF,
	};

private:
	struct SpawnInfo {
		uint32_t scene_index;
		Variant custom_data;

		SpawnInfo(uint32_t p_scene_index, Variant p_custom_data) :
				scene_index(p_scene_index), custom_data(p_custom_data) {}

		// Empty constructor for HashMap internally.
		SpawnInfo() {}
	};

	NodePath spawn_path = NodePath("..");
	uint32_t spawn_limit = 0;
	Array spawnable_scenes;

	Node *spawn_node = nullptr;
	HashMap<Node *, SpawnInfo> tracked_nodes;
	Vector<Node *> spawned_nodes;

protected:
	static void _bind_methods();
	void _notification(int p_what);

// Editor-only array properties.
#ifdef TOOLS_ENABLED
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
#endif

	void _update_spawn_node();
	void _on_child_added(Node *p_node);

	void _create_spawn_payloads(Node *p_parent, Node *p_node, Array &r_payloads) const;
	void _apply_spawn_payloads(Node *p_parent, Node *p_node, const Array &p_payloads, int &p_index) const;

	void _track(Node *p_node, const uint32_t p_scene_index, const Variant &p_custom_data);
	void _spawn_tracked_node(Node *p_node);
	void _despawn_tracked_node(Node *p_node);

	void _send_spawns(const int p_peer);

	void _on_peer_connected(const int p_peer);
	//void _on_peer_disconnected(const int p_peer);

	void _rpc_spawn(const String &p_name, const uint32_t p_scene_index, const Variant &p_custom_data, const Array &p_payloads);
	void _rpc_despawn(const int p_index);
	void _rpc_request_spawns();

public:
	NodePath get_spawn_path() const;
	void set_spawn_path(const NodePath &p_path);

	void set_spawn_limit(uint32_t p_limit);
	uint32_t get_spawn_limit() const;

	Array get_spawnable_scenes() const;
	void set_spawnable_scenes(const Array &p_scenes);

	Node *spawn_custom(const Variant &p_data = Variant());

	Node *instantiate_scene(const uint32_t p_scene_index);
	Node *instantiate_custom(const Variant &p_data);

	GDVIRTUAL1R(Node *, _spawn_custom, const Variant &);

	MultiplayerSpawner();
};

#endif // MULTIPLAYER_SPAWNER_H
