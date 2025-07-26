/**************************************************************************/
/*  graph_connection.h                                                    */
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

#include "core/io/resource.h"
#include "scene/2d/line_2d.h"

class GraphPort;
class GraphNode;

class GraphConnection : public Resource {
	GDCLASS(GraphConnection, Resource);

	friend class GraphEdit;
	friend GraphPort;
	friend GraphNode;

protected:
	struct ConnectionType {
		union {
			uint64_t key = 0;
			struct {
				uint32_t type_a;
				uint32_t type_b;
			};
		};

		static uint32_t hash(const ConnectionType &p_conn) {
			return hash_one_uint64(p_conn.key);
		}
		bool operator==(const ConnectionType &p_type) const {
			return key == p_type.key;
		}

		ConnectionType(uint32_t a = 0, uint32_t b = 0) {
			type_a = a;
			type_b = b;
		}
	};

	float activity = 0.0; // why is this used?

	struct Cache {
		bool dirty = true;
		Vector2 from_pos; // In graph space.
		Vector2 to_pos; // In graph space.
		Color from_color;
		Color to_color;
		Rect2 aabb; // In local screen space.
		Line2D *line = nullptr; // In local screen space.
	} _cache;

	void set_clear_if_invalid(bool p_clear_if_invalid);
	bool get_clear_if_invalid() const;

	void set_line_material(const Ref<ShaderMaterial> p_line_material);
	Ref<ShaderMaterial> get_line_material() const;

	void set_line_width(float width);

	Ref<Gradient> get_line_gradient() const;

	void update_cache();

	static void _bind_methods();

public:
	GraphPort *first_port = nullptr;
	GraphPort *second_port = nullptr;
	bool clear_if_invalid = true;

	void set_first_port(GraphPort *p_port);
	GraphPort *get_first_port() const;

	void set_second_port(GraphPort *p_port);
	GraphPort *get_second_port() const;

	GraphNode *get_first_node() const;
	GraphNode *get_second_node() const;

	GraphPort *get_other_port(const GraphPort *p_port) const;
	GraphPort *get_other_port_by_node(const GraphNode *p_node) const;
	GraphNode *get_other_node(const GraphNode *p_node) const;
	GraphNode *get_other_node_by_port(const GraphPort *p_port) const;
	GraphPort *get_port_by_node(const GraphNode *p_node) const;

	Pair<Pair<String, int>, Pair<String, int>> _to_legacy_data() const;
	bool matches_legacy_data(String p_first_node, int p_first_port, String p_second_node, int p_second_port);

	GraphConnection();
	GraphConnection(GraphPort *p_first_port, GraphPort *p_second_port, bool p_clear_if_invalid);
	~GraphConnection();
};
