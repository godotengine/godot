/*************************************************************************/
/*  scene_preloader.cpp                                                  */
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
#include "scene_preloader.h"
#include "globals.h"

bool ScenePreloader::can_instance() const {

	return nodes.size() > 0;
}

Node *ScenePreloader::instance() const {

	int nc = nodes.size();
	ERR_FAIL_COND_V(nc == 0, NULL);

	const StringName *snames = NULL;
	int sname_count = names.size();
	if (sname_count)
		snames = &names[0];

	const Variant *props = NULL;
	int prop_count = variants.size();
	if (prop_count)
		props = &variants[0];

	Vector<Variant> properties;

	const NodeData *nd = &nodes[0];

	Node **ret_nodes = (Node **)alloca(sizeof(Node *) * nc);

	for (int i = 0; i < nc; i++) {

		const NodeData &n = nd[i];

		if (!ObjectTypeDB::is_type_enabled(snames[n.type])) {
			ret_nodes[i] = NULL;
			continue;
		}

		Object *obj = ObjectTypeDB::instance(snames[n.type]);
		ERR_FAIL_COND_V(!obj, NULL);
		Node *node = obj->cast_to<Node>();
		ERR_FAIL_COND_V(!node, NULL);

		int nprop_count = n.properties.size();
		if (nprop_count) {

			const NodeData::Property *nprops = &n.properties[0];

			for (int j = 0; j < nprop_count; j++) {

				bool valid;
				node->set(snames[nprops[j].name], props[nprops[j].value], &valid);
			}
		}

		node->set_name(snames[n.name]);
		ret_nodes[i] = node;
		if (i > 0) {
			ERR_FAIL_INDEX_V(n.parent, i, NULL);
			ERR_FAIL_COND_V(!ret_nodes[n.parent], NULL);
			ret_nodes[n.parent]->add_child(node);
		}
	}

	//do connections

	int cc = connections.size();
	const ConnectionData *cdata = connections.ptr();

	for (int i = 0; i < cc; i++) {

		const ConnectionData &c = cdata[i];
		ERR_FAIL_INDEX_V(c.from, nc, NULL);
		ERR_FAIL_INDEX_V(c.to, nc, NULL);

		Vector<Variant> binds;
		if (c.binds.size()) {
			binds.resize(c.binds.size());
			for (int j = 0; j < c.binds.size(); j++)
				binds[j] = props[c.binds[j]];
		}

		if (!ret_nodes[c.from] || !ret_nodes[c.to])
			continue;
		ret_nodes[c.from]->connect(snames[c.signal], ret_nodes[c.to], snames[c.method], binds, CONNECT_PERSIST);
	}

	return ret_nodes[0];
}

static int _nm_get_string(const String &p_string, Map<StringName, int> &name_map) {

	if (name_map.has(p_string))
		return name_map[p_string];

	int idx = name_map.size();
	name_map[p_string] = idx;
	return idx;
}

static int _vm_get_variant(const Variant &p_variant, HashMap<Variant, int, VariantHasher> &variant_map) {

	if (variant_map.has(p_variant))
		return variant_map[p_variant];

	int idx = variant_map.size();
	variant_map[p_variant] = idx;
	return idx;
}

void ScenePreloader::_parse_node(Node *p_owner, Node *p_node, int p_parent_idx, Map<StringName, int> &name_map, HashMap<Variant, int, VariantHasher> &variant_map, Map<Node *, int> &node_map) {

	if (p_node != p_owner && !p_node->get_owner())
		return;

	NodeData nd;
	nd.name = _nm_get_string(p_node->get_name(), name_map);
	nd.type = _nm_get_string(p_node->get_type(), name_map);
	nd.parent = p_parent_idx;

	List<PropertyInfo> plist;
	p_node->get_property_list(&plist);
	for (List<PropertyInfo>::Element *E = plist.front(); E; E = E->next()) {

		if (!(E->get().usage & PROPERTY_USAGE_STORAGE))
			continue;

		NodeData::Property prop;
		prop.name = _nm_get_string(E->get().name, name_map);
		prop.value = _vm_get_variant(p_node->get(E->get().name), variant_map);
		nd.properties.push_back(prop);
	}

	int idx = nodes.size();
	node_map[p_node] = idx;
	nodes.push_back(nd);

	for (int i = 0; i < p_node->get_child_count(); i++) {

		Node *c = p_node->get_child(i);
		_parse_node(p_owner, c, idx, name_map, variant_map, node_map);
	}
}

void ScenePreloader::_parse_connections(Node *p_node, Map<StringName, int> &name_map, HashMap<Variant, int, VariantHasher> &variant_map, Map<Node *, int> &node_map, bool p_instance) {

	List<MethodInfo> signals;
	p_node->get_signal_list(&signals);

	for (List<MethodInfo>::Element *E = signals.front(); E; E = E->next()) {

		List<Node::Connection> conns;
		p_node->get_signal_connection_list(E->get().name, &conns);
		for (List<Node::Connection>::Element *F = conns.front(); F; F = F->next()) {

			const Node::Connection &c = F->get();
			if (!(c.flags & CONNECT_PERSIST))
				continue;
			Node *n = c.target->cast_to<Node>();
			if (!n)
				continue;

			if (!node_map.has(n))
				continue;

			ConnectionData cd;
			cd.from = node_map[p_node];
			cd.to = node_map[n];
			cd.method = _nm_get_string(c.method, name_map);
			cd.signal = _nm_get_string(c.signal, name_map);
			for (int i = 0; i < c.binds.size(); i++) {

				cd.binds.push_back(_vm_get_variant(c.binds[i], variant_map));
			}
			connections.push_back(cd);
		}
	}
}

Error ScenePreloader::load_scene(const String &p_path) {

	return ERR_CANT_OPEN;
#if 0
	if (path==p_path)
		return OK;

	String p=Globals::get_singleton()->localize_path(p_path);
	clear();

	Node *scene = SceneLoader::load(p);

	ERR_FAIL_COND_V(!scene,ERR_CANT_OPEN);

	path=p;

	Map<StringName,int> name_map;
	HashMap<Variant,int,VariantHasher> variant_map;
	Map<Node*,int> node_map;

	_parse_node(scene,scene,-1,name_map,variant_map,node_map);


	names.resize(name_map.size());

	for(Map<StringName,int>::Element *E=name_map.front();E;E=E->next()) {

		names[E->get()]=E->key();
	}

	variants.resize(variant_map.size());
	const Variant *K=NULL;
	while((K=variant_map.next(K))) {

		int idx = variant_map[*K];
		variants[idx]=*K;
	}


	memdelete(scene); // <- me falto esto :(
	return OK;
#endif
}

String ScenePreloader::get_scene_path() const {

	return path;
}

void ScenePreloader::clear() {

	names.clear();
	variants.clear();
	nodes.clear();
	connections.clear();
}

void ScenePreloader::_set_bundled_scene(const Dictionary &d) {

	ERR_FAIL_COND(!d.has("names"));
	ERR_FAIL_COND(!d.has("variants"));
	ERR_FAIL_COND(!d.has("node_count"));
	ERR_FAIL_COND(!d.has("nodes"));
	ERR_FAIL_COND(!d.has("conn_count"));
	ERR_FAIL_COND(!d.has("conns"));
	ERR_FAIL_COND(!d.has("path"));

	DVector<String> snames = d["names"];
	if (snames.size()) {

		int namecount = snames.size();
		names.resize(namecount);
		DVector<String>::Read r = snames.read();
		for (int i = 0; i < names.size(); i++)
			names[i] = r[i];
	}

	Array svariants = d["variants"];

	if (svariants.size()) {
		int varcount = svariants.size();
		variants.resize(varcount);
		for (int i = 0; i < varcount; i++) {

			variants[i] = svariants[i];
		}

	} else {
		variants.clear();
	}

	nodes.resize(d["node_count"]);
	int nc = nodes.size();
	if (nc) {
		DVector<int> snodes = d["nodes"];
		DVector<int>::Read r = snodes.read();
		int idx = 0;
		for (int i = 0; i < nc; i++) {
			NodeData &nd = nodes[i];
			nd.parent = r[idx++];
			nd.type = r[idx++];
			nd.name = r[idx++];
			nd.properties.resize(r[idx++]);
			for (int j = 0; j < nd.properties.size(); j++) {

				nd.properties[j].name = r[idx++];
				nd.properties[j].value = r[idx++];
			}
		}
	}

	connections.resize(d["conn_count"]);
	int cc = connections.size();

	if (cc) {

		DVector<int> sconns = d["conns"];
		DVector<int>::Read r = sconns.read();
		int idx = 0;
		for (int i = 0; i < nc; i++) {
			ConnectionData &cd = connections[nc];
			cd.from = r[idx++];
			cd.to = r[idx++];
			cd.signal = r[idx++];
			cd.method = r[idx++];
			cd.binds.resize(r[idx++]);
			for (int j = 0; j < cd.binds.size(); j++) {

				cd.binds[j] = r[idx++];
			}
		}
	}

	path = d["path"];
}

Dictionary ScenePreloader::_get_bundled_scene() const {

	DVector<String> rnames;
	rnames.resize(names.size());

	if (names.size()) {

		DVector<String>::Write r = rnames.write();

		for (int i = 0; i < names.size(); i++)
			r[i] = names[i];
	}

	Dictionary d;
	d["names"] = rnames;
	d["variants"] = variants;

	Vector<int> rnodes;
	d["node_count"] = nodes.size();

	for (int i = 0; i < nodes.size(); i++) {

		const NodeData &nd = nodes[i];
		rnodes.push_back(nd.parent);
		rnodes.push_back(nd.type);
		rnodes.push_back(nd.name);
		rnodes.push_back(nd.properties.size());
		for (int j = 0; j < nd.properties.size(); j++) {

			rnodes.push_back(nd.properties[j].name);
			rnodes.push_back(nd.properties[j].value);
		}
	}

	d["nodes"] = rnodes;

	Vector<int> rconns;
	d["conn_count"] = connections.size();

	for (int i = 0; i < connections.size(); i++) {

		const ConnectionData &cd = connections[i];
		rconns.push_back(cd.from);
		rconns.push_back(cd.to);
		rconns.push_back(cd.signal);
		rconns.push_back(cd.method);
		rconns.push_back(cd.binds.size());
		for (int j = 0; j < cd.binds.size(); j++)
			rconns.push_back(cd.binds[j]);
	}

	d["conns"] = rconns;

	d["path"] = path;

	return d;
}

void ScenePreloader::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("load_scene", "path"), &ScenePreloader::load_scene);
	ObjectTypeDB::bind_method(_MD("get_scene_path"), &ScenePreloader::get_scene_path);
	ObjectTypeDB::bind_method(_MD("instance:Node"), &ScenePreloader::instance);
	ObjectTypeDB::bind_method(_MD("can_instance"), &ScenePreloader::can_instance);
	ObjectTypeDB::bind_method(_MD("_set_bundled_scene"), &ScenePreloader::_set_bundled_scene);
	ObjectTypeDB::bind_method(_MD("_get_bundled_scene"), &ScenePreloader::_get_bundled_scene);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "_bundled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_BUNDLE), _SCS("_set_bundled_scene"), _SCS("_get_bundled_scene"));
#if 0
	List<String> extensions;
	SceneLoader::get_recognized_extensions(&extensions);
	String exthint;
	for (List<String>::Element*E=extensions.front();E;E=E->next()) {

		if (exthint!="")
			exthint+=",";
		exthint+="*."+E->get();
	}

	exthint+="; Scenes";

	ADD_PROPERTY( PropertyInfo(Variant::STRING,"scene",PROPERTY_HINT_FILE,exthint),_SCS("load_scene"),_SCS("get_scene_path"));
#endif
}

ScenePreloader::ScenePreloader() {
}
