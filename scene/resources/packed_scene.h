/*************************************************************************/
/*  packed_scene.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef PACKED_SCENE_H
#define PACKED_SCENE_H

#include "resource.h"
#include "scene/main/node.h"

class PackedScene : public Resource {

	OBJ_TYPE( PackedScene, Resource );
	RES_BASE_EXTENSION("scn");
	Vector<StringName> names;
	Vector<Variant> variants;

	//missing - instances
	//missing groups
	//missing - owner
	//missing - override names and values

	struct NodeData {

		int parent;
		int owner;
		int type;
		int name;
		int instance;

		struct Property {

			int name;
			int value;
		};

		Vector<Property> properties;
		Vector<int> groups;
	};


	Vector<NodeData> nodes;

	struct ConnectionData {

		int from;
		int to;
		int signal;
		int method;
		int flags;
		Vector<int> binds;
	};

	Vector<ConnectionData> connections;

	Error _parse_node(Node *p_owner,Node *p_node,int p_parent_idx, Map<StringName,int> &name_map,HashMap<Variant,int,VariantHasher> &variant_map,Map<Node*,int> &node_map);
	Error _parse_connections(Node *p_owner,Node *p_node, Map<StringName,int> &name_map,HashMap<Variant,int,VariantHasher> &variant_map,Map<Node*,int> &node_map);


	void _set_bundled_scene(const Dictionary& p_scene);
	Dictionary _get_bundled_scene() const;

protected:


	static void _bind_methods();
public:


	Error pack(Node *p_scene);

	void clear();

	bool can_instance() const;
	Node *instance(bool p_gen_edit_state=false) const;

	PackedScene();
};

#endif // SCENE_PRELOADER_H
