/**************************************************************************/
/*  shader_graph.h                                                        */
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

#include "core/string/string_builder.h"
#include "core/templates/rb_map.h"
#include "scene/resources/shader.h"

class VisualShaderNodeParameter;
class VisualShaderNode;
class VisualShaderGroup;

class ShaderGraph : public RefCounted {
	GDCLASS(ShaderGraph, RefCounted);

	friend class VisualShaderGroup;

public:
	// Keep in sync with VisualShader::Type.
	enum Type {
		TYPE_VERTEX,
		TYPE_FRAGMENT,
		TYPE_LIGHT,
		TYPE_START,
		TYPE_PROCESS,
		TYPE_COLLIDE,
		TYPE_START_CUSTOM,
		TYPE_PROCESS_CUSTOM,
		TYPE_SKY,
		TYPE_FOG,
		TYPE_TEXTURE_BLIT,
		TYPE_MAX
	};

	struct Node {
		Ref<VisualShaderNode> node;
		Vector2 position;
		LocalVector<int> prev_connected_nodes;
		LocalVector<int> next_connected_nodes;
	};

	struct Connection {
		int from_node = 0;
		int from_port = 0;
		int to_node = 0;
		int to_port = 0;
	};

	union ConnectionKey {
		struct {
			uint64_t node : 32;
			uint64_t port : 32;
		};

		uint64_t key = 0;

		bool operator<(const ConnectionKey &p_key) const {
			return key < p_key.key;
		}
		uint32_t hash() const { return HashMapHasherDefault::hash(key); }
		bool is_same(const ConnectionKey &p_key) const { return HashMapComparatorDefault<uint64_t>::compare(key, p_key.key); }
	};

	struct DefaultTextureParam {
		StringName name;
		List<Ref<Texture>> params;
	};

	static constexpr int NODE_ID_INVALID = -1;
	static constexpr int NODE_ID_OUTPUT = 0;

	int reserved_node_ids = 1;

	RBMap<int, Node> nodes;
	List<Connection> connections;

	void _node_changed();

protected:
	static void _bind_methods();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	void _write_node(
			StringBuilder *p_global_code,
			StringBuilder *p_global_code_per_node,
			HashMap<Type, StringBuilder> *p_global_code_per_func,
			StringBuilder &r_code,
			Vector<ShaderGraph::DefaultTextureParam> &r_def_tex_params,
			const HashMap<ConnectionKey, const List<ShaderGraph::Connection>::Element *> &p_input_connections,
			const HashMap<ConnectionKey, const List<ShaderGraph::Connection>::Element *> &p_output_connections,
			int p_node,
			HashSet<int> &r_processed,
			bool p_for_preview,
			HashSet<StringName> &r_classes,
			Type p_type = TYPE_MAX, // Only used for VisualShader.
			Shader::Mode p_mode = Shader::MODE_MAX // Only used for VisualShader.
	) const;

	bool _check_reroute_subgraph(int p_target_port_type, int p_reroute_node, List<int> *r_visited_reroute_nodes = nullptr) const;

	void add_node(const Ref<VisualShaderNode> &p_node, const Vector2 &p_position, int p_id);
	void set_node_position(int p_id, const Vector2 &p_position);
	Vector2 get_node_position(int p_id) const;
	Ref<VisualShaderNode> get_node(int p_id) const;
	_FORCE_INLINE_ const Ref<VisualShaderNode> &get_node_unchecked(int p_id) const {
		return nodes[p_id].node;
	}
	_FORCE_INLINE_ const LocalVector<int> &get_next_connected_node_ids(int p_id) const {
		return nodes[p_id].next_connected_nodes;
	}
	_FORCE_INLINE_ const LocalVector<int> &get_prev_connected_node_ids(int p_id) const {
		return nodes[p_id].prev_connected_nodes;
	}

	Vector<int> get_node_ids() const;
	int get_valid_node_id() const;
	int find_node_id(const Ref<VisualShaderNode> &p_node) const;
	void remove_node(int p_id);
	void replace_node(int p_id, const StringName &p_new_class);

	bool are_nodes_connected(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const;
	bool is_node_reachable(int p_from, int p_target) const;
	bool can_connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port) const;
	Error connect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	void disconnect_nodes(int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	void connect_nodes_forced(int p_from_node, int p_from_port, int p_to_node, int p_to_port);
	bool is_port_types_compatible(int p_a, int p_b) const;

	void attach_node_to_frame(int p_node, int p_frame);
	void detach_node_from_frame(int p_node);

	String get_reroute_parameter_name(int p_reroute_node) const;

	void get_node_connections(List<ShaderGraph::Connection> *r_connections) const;

	String generate_preview_shader(int p_node, int p_port, Vector<DefaultTextureParam> &r_default_tex_params, const String &p_additional_global_code = String()) const;

	String validate_port_name(const String &p_port_name, VisualShaderNode *p_node, int p_port_id, bool p_output) const;

	ShaderGraph(int reserved_node_ids = 1);
};
