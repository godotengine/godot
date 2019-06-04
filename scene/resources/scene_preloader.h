/*************************************************************************/
/*  scene_preloader.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef SCENE_PRELOADER_H
#define SCENE_PRELOADER_H

#include "resource.h"
#include "scene/main/node.h"

class ScenePreloader : public Resource {

	OBJ_TYPE(ScenePreloader, Resource);

	Vector<StringName> names;
	Vector<Variant> variants;

	//missing - instances
	//missing groups
	//missing - owner
	//missing - override names and values

	struct NodeData {

		int parent;
		int type;
		int name;

		struct Property {

			int name;
			int value;
		};

		Vector<Property> properties;
	};

	Vector<NodeData> nodes;

	struct ConnectionData {

		int from;
		int to;
		int signal;
		int method;
		Vector<int> binds;
	};

	Vector<ConnectionData> connections;

	void _parse_node(Node *p_owner, Node *p_node, int p_parent_idx, Map<StringName, int> &name_map, HashMap<Variant, int, VariantHasher> &variant_map, Map<Node *, int> &node_map);
	void _parse_connections(Node *p_node, Map<StringName, int> &name_map, HashMap<Variant, int, VariantHasher> &variant_map, Map<Node *, int> &node_map, bool p_instance);

	String path;

	void _set_bundled_scene(const Dictionary &p_scene);
	Dictionary _get_bundled_scene() const;

protected:
	static void _bind_methods();

public:
	Error load_scene(const String &p_path);
	String get_scene_path() const;
	void clear();

	bool can_instance() const;
	Node *instance() const;

	ScenePreloader();
};

#endif // SCENE_PRELOADER_H
