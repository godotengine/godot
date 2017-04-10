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
