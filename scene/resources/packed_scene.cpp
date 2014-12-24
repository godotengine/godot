/*************************************************************************/
/*  packed_scene.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "packed_scene.h"
#include "globals.h"
#include "io/resource_loader.h"
#include "scene/3d/spatial.h"
#include "scene/gui/control.h"
#include "scene/2d/node_2d.h"

bool PackedScene::can_instance() const {

	return nodes.size()>0;
}

Node *PackedScene::instance(bool p_gen_edit_state) const {

	int nc = nodes.size();
	ERR_FAIL_COND_V(nc==0,NULL);

	const StringName*snames=NULL;
	int sname_count=names.size();
	if (sname_count)
		snames=&names[0];

	const Variant*props=NULL;
	int prop_count=variants.size();
	if (prop_count)
		props=&variants[0];

	Vector<Variant> properties;

	const NodeData *nd = &nodes[0];

	Node **ret_nodes=(Node**)alloca( sizeof(Node*)*nc );


	for(int i=0;i<nc;i++) {

		const NodeData &n=nd[i];

		if (!ObjectTypeDB::is_type_enabled(snames[n.type])) {
			ret_nodes[i]=NULL;
			continue;
		}

		Node *node=NULL;

		if (n.instance>=0) {
			//instance existing
			Ref<PackedScene> sdata = props[ n.instance ];
			ERR_FAIL_COND_V( !sdata.is_valid(), NULL);
			node = sdata->instance();
			ERR_FAIL_COND_V(!node,NULL);
			if (p_gen_edit_state)
				node->generate_instance_state();

		} else {
			//create anew
			Object * obj = ObjectTypeDB::instance(snames[ n.type ]);
			if (!obj || !obj->cast_to<Node>()) {
				if (obj) {
					memdelete(obj);
					obj=NULL;
				}
				WARN_PRINT(String("Warning node of type "+snames[n.type].operator String()+" does not exist.").ascii().get_data());
				if (n.parent>=0 && n.parent<nc && ret_nodes[n.parent]) {
					if (ret_nodes[n.parent]->cast_to<Spatial>()) {
						obj = memnew( Spatial );
					} else if (ret_nodes[n.parent]->cast_to<Control>()) {
						obj = memnew( Control );
					} else if (ret_nodes[n.parent]->cast_to<Node2D>()) {
						obj = memnew( Node2D );
					}

				}
				if (!obj) {
					obj = memnew( Node );
				}
			}

			node = obj->cast_to<Node>();

		}


		//properties
		int nprop_count=n.properties.size();
		if (nprop_count) {

			const NodeData::Property* nprops=&n.properties[0];

			for(int j=0;j<nprop_count;j++) {

				bool valid;
				ERR_FAIL_INDEX_V( nprops[j].name, sname_count, NULL );
				ERR_FAIL_INDEX_V( nprops[j].value, prop_count, NULL );

				node->set(snames[ nprops[j].name ],props[ nprops[j].value ],&valid);
			}
		}

		//name

		//groups
		for(int j=0;j<n.groups.size();j++) {

			ERR_FAIL_INDEX_V( n.groups[j], sname_count, NULL );
			node->add_to_group( snames[ n.groups[j] ], true );
		}


		ret_nodes[i]=node;

		if (i>0) {			
			ERR_FAIL_INDEX_V(n.parent,i,NULL);
			ERR_FAIL_COND_V(!ret_nodes[n.parent],NULL);
			ret_nodes[n.parent]->_add_child_nocheck(node,snames[n.name]);
		} else {
			node->_set_name_nocheck( snames[ n.name ] );
		}


		if (n.owner>=0) {

			ERR_FAIL_INDEX_V(n.owner,i,NULL);
			node->_set_owner_nocheck(ret_nodes[n.owner]);
		}

	}


	//do connections

	int cc = connections.size();
	const ConnectionData *cdata = connections.ptr();

	for(int i=0;i<cc;i++) {

		const ConnectionData &c=cdata[i];
		ERR_FAIL_INDEX_V( c.from, nc, NULL );
		ERR_FAIL_INDEX_V( c.to, nc, NULL );

		Vector<Variant> binds;
		if (c.binds.size()) {
			binds.resize(c.binds.size());
			for(int j=0;j<c.binds.size();j++)
				binds[j]=props[ c.binds[j] ];
		}

		if (!ret_nodes[c.from] || !ret_nodes[c.to])
			continue;
		ret_nodes[c.from]->connect( snames[ c.signal], ret_nodes[ c.to ], snames[ c.method], binds,CONNECT_PERSIST|c.flags );
	}

	Node *s = ret_nodes[0];

	if (get_path()!="" && get_path().find("::")==-1)
		s->set_filename(get_path());
	return ret_nodes[0];

}


static int _nm_get_string(const String& p_string, Map<StringName,int> &name_map) {

	if (name_map.has(p_string))
		return name_map[p_string];

	int idx = name_map.size();
	name_map[p_string]=idx;
	return idx;
}

static int _vm_get_variant(const Variant& p_variant, HashMap<Variant,int,VariantHasher> &variant_map) {

	if (variant_map.has(p_variant))
		return variant_map[p_variant];

	int idx = variant_map.size();
	variant_map[p_variant]=idx;
	return idx;
}

Error PackedScene::_parse_node(Node *p_owner,Node *p_node,int p_parent_idx, Map<StringName,int> &name_map,HashMap<Variant,int,VariantHasher> &variant_map,Map<Node*,int> &node_map) {

	if (p_node!=p_owner && (p_node->get_owner()!=p_owner))
		return OK; //nothing to do with this node, may either belong to another scene or be onowned

	NodeData nd;

	nd.name=_nm_get_string(p_node->get_name(),name_map);
	nd.type=_nm_get_string(p_node->get_type(),name_map);
	nd.parent=p_parent_idx;	


	Dictionary instance_state;
	Set<StringName> instance_groups;


	if (p_node!=p_owner && p_node->get_filename()!="") {
		//instanced
		Ref<PackedScene> instance = ResourceLoader::load(p_node->get_filename());
		if (!instance.is_valid()) {
			return ERR_CANT_OPEN;
		}

		nd.instance=_vm_get_variant(instance,variant_map);
		instance_state = p_node->get_instance_state();
		Vector<StringName> ig = p_node->get_instance_groups();
		for(int i=0;i<ig.size();i++)
			instance_groups.insert(ig[i]);
	} else {

		nd.instance=-1;
	}


	//instance state makes sure that only changes to instance are saved

	List<PropertyInfo> plist;
	p_node->get_property_list(&plist);
	for (List<PropertyInfo>::Element *E=plist.front();E;E=E->next()) {


		if (!(E->get().usage & PROPERTY_USAGE_STORAGE)) {
			continue;
		}

		String name = E->get().name;
		Variant value = p_node->get( E->get().name );

		if (E->get().usage & PROPERTY_USAGE_STORE_IF_NONZERO && value.is_zero()) {
			continue;
		}


		if (nd.instance>=0) {
			//only save changed properties in instance
			/*
			  // this was commented because it would not save properties created from within script
			  // done with _get_property_list, that are not in the original node.
			  // if some property is not saved, check again

			  if (!instance_state.has(name)) {
				print_line("skip not in instance");
				continue;
			}*/

			if (E->get().usage & PROPERTY_USAGE_NO_INSTANCE_STATE) {
				continue;
			}

			if (instance_state[name]==value) {
				continue;
			}

		}

		NodeData::Property prop;
		prop.name=_nm_get_string( name,name_map);
		prop.value=_vm_get_variant( value, variant_map);
		nd.properties.push_back(prop);

	}


	List<Node::GroupInfo> groups;
	p_node->get_groups(&groups);
	for(List<Node::GroupInfo>::Element *E=groups.front();E;E=E->next()) {
		Node::GroupInfo &gi=E->get();

		if (!gi.persistent)
			continue;
		if (nd.instance>=0 && instance_groups.has(gi.name))
			continue; //group was instanced, don't add here

		nd.groups.push_back(_nm_get_string(gi.name,name_map));
	}

	if (node_map.has(p_node->get_owner()))
		nd.owner=node_map[p_node->get_owner()];
	else
		nd.owner=-1;

	int idx = nodes.size();
	node_map[p_node]=idx;
	nodes.push_back(nd);



	for(int i=0;i<p_node->get_child_count();i++) {

		Node *c=p_node->get_child(i);
		Error err = _parse_node(p_owner,c,idx,name_map,variant_map,node_map);
		if (err)
			return err;
	}

	return OK;

}

Error PackedScene::_parse_connections(Node *p_owner,Node *p_node, Map<StringName,int> &name_map,HashMap<Variant,int,VariantHasher> &variant_map,Map<Node*,int> &node_map) {

	if (p_node!=p_owner && (p_node->get_owner()!=p_owner))
		return OK; //nothing to do with this node, may either belong to another scene or be onowned

	List<MethodInfo> signals;
	p_node->get_signal_list(&signals);

	ERR_FAIL_COND_V( !node_map.has(p_node), ERR_BUG);
	NodeData &nd = nodes[node_map[p_node]];
	Set<Connection> instance_connections;

	if (nd.instance>=0) {

		Vector<Connection> iconns = p_node->get_instance_connections();
		for(int i=0;i<iconns.size();i++) {

			instance_connections.insert(iconns[i]);
		}
	}


	for(List<MethodInfo>::Element *E=signals.front();E;E=E->next()) {

		List<Node::Connection> conns;
		p_node->get_signal_connection_list(E->get().name,&conns);
		for(List<Node::Connection>::Element *F=conns.front();F;F=F->next()) {

			const Node::Connection &c = F->get();
			if (!(c.flags&CONNECT_PERSIST))
				continue;

			if (nd.instance>=0 && instance_connections.has(c))
				continue; //came from instance, don't save!

			Node *n=c.target->cast_to<Node>();
			if (!n)
				continue;

			if (!node_map.has(n)) {
				WARN_PRINT("Connection to node outside scene??")
				continue;
			}

			ConnectionData cd;
			cd.from=node_map[p_node];
			cd.to=node_map[n];
			cd.method=_nm_get_string(c.method,name_map);
			cd.signal=_nm_get_string(c.signal,name_map);
			cd.flags=c.flags;
			for(int i=0;i<c.binds.size();i++) {

				cd.binds.push_back( _vm_get_variant(c.binds[i],variant_map));
			}
			connections.push_back(cd);
		}
	}

	for(int i=0;i<p_node->get_child_count();i++) {

		Node *c=p_node->get_child(i);
		Error err = _parse_connections(p_owner,c,name_map,variant_map,node_map);
		if (err)
			return err;
	}

	return OK;
}


Error PackedScene::pack(Node *p_scene) {
	ERR_FAIL_NULL_V( p_scene, ERR_INVALID_PARAMETER );


	clear();

	Node *scene = p_scene;

	Map<StringName,int> name_map;
	HashMap<Variant,int,VariantHasher> variant_map;
	Map<Node*,int> node_map;

	Error err = _parse_node(scene,scene,-1,name_map,variant_map,node_map);
	if (err) {
		clear();
		ERR_FAIL_V(err);
	}

	err = _parse_connections(scene,scene,name_map,variant_map,node_map);
	if (err) {
		clear();
		ERR_FAIL_V(err);
	}

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

	return OK;
}

void PackedScene::clear() {

	names.clear();
	variants.clear();
	nodes.clear();
	connections.clear();

}

void PackedScene::_set_bundled_scene(const Dictionary& d) {


	ERR_FAIL_COND( !d.has("names"));
	ERR_FAIL_COND( !d.has("variants"));
	ERR_FAIL_COND( !d.has("node_count"));
	ERR_FAIL_COND( !d.has("nodes"));
	ERR_FAIL_COND( !d.has("conn_count"));
	ERR_FAIL_COND( !d.has("conns"));
//	ERR_FAIL_COND( !d.has("path"));

	DVector<String> snames = d["names"];
	if (snames.size()) {

		int namecount = snames.size();
		names.resize(namecount);
		DVector<String>::Read r =snames.read();
		for(int i=0;i<names.size();i++)
			names[i]=r[i];
	}

	Array svariants = d["variants"];

	if (svariants.size()) {
		int varcount=svariants.size();
		variants.resize(varcount);
		for(int i=0;i<varcount;i++) {

			variants[i]=svariants[i];
		}

	} else {
		variants.clear();
	}

	nodes.resize(d["node_count"]);
	int nc=nodes.size();
	if (nc) {
		DVector<int> snodes = d["nodes"];
		DVector<int>::Read r = snodes.read();
		int idx=0;
		for(int i=0;i<nc;i++) {
			NodeData &nd = nodes[i];
			nd.parent=r[idx++];
			nd.owner=r[idx++];
			nd.type=r[idx++];
			nd.name=r[idx++];
			nd.instance=r[idx++];
			nd.properties.resize(r[idx++]);
			for(int j=0;j<nd.properties.size();j++) {

				nd.properties[j].name=r[idx++];
				nd.properties[j].value=r[idx++];
			}
			nd.groups.resize(r[idx++]);
			for(int j=0;j<nd.groups.size();j++) {

				nd.groups[j]=r[idx++];
			}
		}

	}

	connections.resize(d["conn_count"]);
	int cc=connections.size();

	if (cc) {

		DVector<int> sconns = d["conns"];
		DVector<int>::Read r = sconns.read();
		int idx=0;
		for(int i=0;i<cc;i++) {
			ConnectionData &cd = connections[i];
			cd.from=r[idx++];
			cd.to=r[idx++];
			cd.signal=r[idx++];
			cd.method=r[idx++];
			cd.flags=r[idx++];
			cd.binds.resize(r[idx++]);

			for(int j=0;j<cd.binds.size();j++) {

				cd.binds[j]=r[idx++];
			}
		}

	}

//	path=d["path"];

}

Dictionary PackedScene::_get_bundled_scene() const {

	DVector<String> rnames;
	rnames.resize(names.size());

	if (names.size()) {

		DVector<String>::Write r=rnames.write();

		for(int i=0;i<names.size();i++)
			r[i]=names[i];
	}

	Dictionary d;
	d["names"]=rnames;
	d["variants"]=variants;

	Vector<int> rnodes;
	d["node_count"]=nodes.size();

	for(int i=0;i<nodes.size();i++) {

		const NodeData &nd=nodes[i];
		rnodes.push_back(nd.parent);
		rnodes.push_back(nd.owner);
		rnodes.push_back(nd.type);
		rnodes.push_back(nd.name);
		rnodes.push_back(nd.instance);
		rnodes.push_back(nd.properties.size());
		for(int j=0;j<nd.properties.size();j++) {

			rnodes.push_back(nd.properties[j].name);
			rnodes.push_back(nd.properties[j].value);
		}
		rnodes.push_back(nd.groups.size());
		for(int j=0;j<nd.groups.size();j++) {

			rnodes.push_back(nd.groups[j]);
		}
	}

	d["nodes"]=rnodes;

	Vector<int> rconns;
	d["conn_count"]=connections.size();

	for(int i=0;i<connections.size();i++) {

		const ConnectionData &cd=connections[i];
		rconns.push_back(cd.from);
		rconns.push_back(cd.to);
		rconns.push_back(cd.signal);
		rconns.push_back(cd.method);
		rconns.push_back(cd.flags);
		rconns.push_back(cd.binds.size());
		for(int j=0;j<cd.binds.size();j++)
			rconns.push_back(cd.binds[j]);

	}

	d["conns"]=rconns;
	d["version"]=1;

//	d["path"]=path;

	return d;


}

void PackedScene::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("pack","path:Node"),&PackedScene::pack);
	ObjectTypeDB::bind_method(_MD("instance:Node"),&PackedScene::instance,DEFVAL(false));
	ObjectTypeDB::bind_method(_MD("can_instance"),&PackedScene::can_instance);
	ObjectTypeDB::bind_method(_MD("_set_bundled_scene"),&PackedScene::_set_bundled_scene);
	ObjectTypeDB::bind_method(_MD("_get_bundled_scene"),&PackedScene::_get_bundled_scene);

	ADD_PROPERTY( PropertyInfo(Variant::DICTIONARY,"_bundled"),_SCS("_set_bundled_scene"),_SCS("_get_bundled_scene"));

}

PackedScene::PackedScene() {


}
