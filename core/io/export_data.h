/*************************************************************************/
/*  export_data.h                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef EXPORT_DATA_H
#define EXPORT_DATA_H

#include "map.h"
#include "variant.h"
#include "vector.h"
struct ExportData {

	struct Dependency {
		String path;
		String type;
	};

	Map<int, Dependency> dependencies;

	struct PropertyData {
		String name;
		Variant value;
	};

	struct ResourceData {

		String type;
		int index;
		List<PropertyData> properties;
	};

	Vector<ResourceData> resources;

	struct NodeData {

		bool text_data;
		bool instanced;
		String name;
		String type;
		String instance;
		//int info
		int owner_int; //depending type
		int parent_int;
		bool instance_is_placeholder;

		//text info
		NodePath parent;
		NodePath owner;
		String instance_placeholder;

		Vector<String> groups;
		List<PropertyData> properties;

		NodeData() {
			parent_int = 0;
			owner_int = 0;
			text_data = true;
			instanced = false;
		}
	};

	Vector<NodeData> nodes;

	struct Connection {

		bool text_data;

		int from_int;
		int to_int;

		NodePath from;
		NodePath to;
		String signal;
		String method;
		Array binds;
		int flags;

		Connection() { text_data = true; }
	};

	Vector<Connection> connections;
	Vector<NodePath> editables;

	Array node_paths; //for integer packed data
	Variant base_scene;
};

#endif // EXPORT_DATA_H
