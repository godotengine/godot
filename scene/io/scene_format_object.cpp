/*************************************************************************/
/*  scene_format_object.cpp                                              */
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
#include "scene_format_object.h"
#include "print_string.h"
#include "globals.h"
#include "scene/resources/packed_scene.h"
#include "io/resource_loader.h"

#ifdef OLD_SCENE_FORMAT_ENABLED

void SceneFormatSaverObject::save_node(const Node* p_root,const Node* p_node,const Node* p_owner,ObjectFormatSaver *p_saver,String p_base_path,uint32_t p_flags,Map<const Node*,uint32_t>& owner_id) const {

	if (p_node!=p_root && p_node->get_owner()==NULL)
		return;


	if (p_flags&SceneSaver::FLAG_BUNDLE_INSTANCED_SCENES || p_node->get_owner() == p_owner || p_node == p_owner ) {
		
		Dictionary d;
		if (p_root!=p_node) {
			d["path"]=p_root->get_path_to(p_node->get_parent());
		}
			
		d["name"]=p_node->get_name();
		
		/* Connections */

		List<MethodInfo> signal_list;

		p_node->get_signal_list(&signal_list);
		
		int conn_count=0;

		Set<Node::Connection> exclude_connections;

		if (!(p_flags&SceneSaver::FLAG_BUNDLE_INSTANCED_SCENES)) {

			Vector<Node::Connection> ex = p_node->get_instance_connections();
			for(int i=0;i<ex.size();i++) {
				exclude_connections.insert(ex[i]);
			}
		}

		for (List<MethodInfo>::Element *S=signal_list.front();S;S=S->next()) {

			List<Node::Connection> connections;
			p_node->get_signal_connection_list(S->get().name,&connections);
			for(List<Node::Connection>::Element *E=connections.front();E;E=E->next()) {

				Node::Connection &c=E->get();
				if (!(c.flags&Object::CONNECT_PERSIST))
					continue;
				if (exclude_connections.has(c))
					continue;

				Node *target = c.target->cast_to<Node>();
				if (!target)
					continue; //connected to something not a node, ignoring

				Dictionary cd;
				cd["signal"]=c.signal;
				cd["target"]=p_node->get_path_to(target);
				cd["method"]=c.method;
				cd["realtime"]=!(c.flags&Object::CONNECT_DEFERRED);
				if (c.binds.size())
					cd["binds"]=c.binds;
				d["connection/"+itos(conn_count+1)]=cd;

				conn_count++;
			}
		}

		d["connection_count"]=conn_count;
		if (owner_id.has(p_node->get_owner())) {

			d["owner"]=owner_id[p_node->get_owner()];
		}
		
		/* Groups */
		
		DVector<String> group_array;
		List<Node::GroupInfo> groups;
		p_node->get_groups(&groups);
		Set<StringName> exclude_groups;

		if (!(p_flags&SceneSaver::FLAG_BUNDLE_INSTANCED_SCENES)) {
			//generate groups to exclude (came from instance)
			Vector<StringName> eg;
			eg=p_node->get_instance_groups();
			for(int i=0;i<eg.size();i++)
				exclude_groups.insert(eg[i]);
		}

		for(List<Node::GroupInfo>::Element*E=groups.front();E;E=E->next()) {
			
			if (E->get().persistent && !exclude_groups.has(E->get().name))
				group_array.push_back(E->get().name);
		}
		
		if (group_array.size())
			d["groups"]=group_array;

		/* Save */
		
		if (p_owner!=p_node && p_node->get_filename()!="") {

			String instance_path;
			if (p_flags&SceneSaver::FLAG_RELATIVE_PATHS)
				instance_path=p_base_path.path_to_file(Globals::get_singleton()->localize_path(p_node->get_filename()));
			else
				instance_path=p_node->get_filename();
			d["instance"]=instance_path;

			if (p_flags&SceneSaver::FLAG_BUNDLE_INSTANCED_SCENES) {

				int id = owner_id.size();
				d["owner_id"]=id;
				owner_id[p_node]=id;

				p_saver->save(p_node,d);

				//owner change!
				for (int i=0;i<p_node->get_child_count();i++) {

					save_node(p_root,p_node->get_child(i),p_node,p_saver,p_base_path,p_flags,owner_id);
				}
				return;

			} else {
				DVector<String> prop_names;
				Array prop_values;

				List<PropertyInfo> properties;
				p_node->get_property_list(&properties);

				//instance state makes sure that only changes to instance are saved
				Dictionary instance_state=p_node->get_instance_state();

				for(List<PropertyInfo>::Element *E=properties.front();E;E=E->next()) {

					if (!(E->get().usage&PROPERTY_USAGE_STORAGE))
						continue;

					String name=E->get().name;
					Variant value=p_node->get(E->get().name);

					if (!instance_state.has(name))
						continue; // did not change since it was loaded, not save
					if (value==instance_state[name])
						continue;
					prop_names.push_back( name );
					prop_values.push_back( value );

				}

				d["override_names"]=prop_names;
				d["override_values"]=prop_values;

				p_saver->save(NULL,d);
			}
		} else {
			
			p_saver->save(p_node,d);	
		}		
	}
	
	for (int i=0;i<p_node->get_child_count();i++) {
		
		save_node(p_root,p_node->get_child(i),p_owner,p_saver,p_base_path,p_flags,owner_id);
	}
}


Error SceneFormatSaverObject::save(const String &p_path,const Node* p_scene,uint32_t p_flags,const Ref<OptimizedSaver> &p_optimizer) {
		
	String extension=p_path.extension();
	if (extension=="scn")
		extension="bin";
	if (extension=="xscn")
		extension="xml";

	String local_path=Globals::get_singleton()->localize_path(p_path);
	uint32_t saver_flags=0;
	if (p_flags&SceneSaver::FLAG_RELATIVE_PATHS)
		saver_flags|=ObjectSaver::FLAG_RELATIVE_PATHS;
	if (p_flags&SceneSaver::FLAG_BUNDLE_RESOURCES)
		saver_flags|=ObjectSaver::FLAG_BUNDLE_RESOURCES;
	if (p_flags&SceneSaver::FLAG_OMIT_EDITOR_PROPERTIES)
		saver_flags|=ObjectSaver::FLAG_OMIT_EDITOR_PROPERTIES;
	if (p_flags&SceneSaver::FLAG_SAVE_BIG_ENDIAN)
		saver_flags|=ObjectSaver::FLAG_SAVE_BIG_ENDIAN;

	ObjectFormatSaver *saver = ObjectSaver::instance_format_saver(local_path,"SCENE",extension,saver_flags,p_optimizer);
	
	ERR_FAIL_COND_V(!saver,ERR_FILE_UNRECOGNIZED);
	
	/* SAVE SCENE */
	
	Map<const Node*,uint32_t> node_id_map;
	save_node(p_scene,p_scene,p_scene,saver,local_path,p_flags,node_id_map);
		
	memdelete(saver);
	
	return OK;
}

void SceneFormatSaverObject::get_recognized_extensions(List<String> *p_extensions) const {
	
	p_extensions->push_back("xml");
	p_extensions->push_back("scn");
	p_extensions->push_back("xscn");

//	ObjectSaver::get_recognized_extensions(p_extensions);
}


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

void SceneFormatLoaderObject::_apply_meta(Node *node,  const Variant&meta, ObjectFormatLoader *p_loader,List<ConnectionItem>& connections,Error& r_err,Map<uint32_t,Node*>& owner_map) {

	r_err = OK;


	Dictionary d=meta;

	if (!d.has("name")) {

		r_err=ERR_WTF;
		memdelete(node);
		ERR_FAIL_COND(!d.has("name"));
	}


	node->set_name(d["name"]);
	int connection_count=d.has("connection_count")?d["connection_count"].operator int():0;


	for (int i=0;i<connection_count;i++) {

		Dictionary cd=d["connection/"+itos(i+1)];

		ERR_CONTINUE(!cd.has("target"));
		ERR_CONTINUE(!cd.has("method"));
		ERR_CONTINUE(!cd.has("realtime"));
		ERR_CONTINUE(!cd.has("signal"));

		ConnectionItem ci;

		ci.node=node;
		ci.target=cd["target"];
		ci.method=cd["method"];
		ci.signal=cd["signal"];
		ci.realtime=cd["realtime"];
		if (cd.has("binds"))
			ci.binds=cd["binds"];

		connections.push_back(ci);

	}

	DVector<String> groups=d.has("groups")?d["groups"].operator DVector<String>():DVector<String>();
	for (int i=0;i<groups.size();i++) {

		node->add_to_group(groups[i],true);
	}

}



Ref<SceneInteractiveLoader> SceneFormatLoaderObject::load_interactive(const String &p_path,bool p_save_root_state) {

	SceneInteractiveLoaderObject *sil = memnew( SceneInteractiveLoaderObject(p_path,p_save_root_state) );

	if (sil->error!=OK) {

		memdelete( sil );
		return Ref<SceneInteractiveLoader>();
	}

	return Ref<SceneInteractiveLoader>( sil );

}


Node* SceneFormatLoaderObject::load_node(Object *obj, const Variant& meta, Node *p_root, ObjectFormatLoader *p_loader,List<ConnectionItem>& connections,Error& r_err,bool p_save_instance_state,Map<uint32_t,Node*>& owner_map) {

	r_err = OK;
		
	Node *node=obj->cast_to<Node>();

	_apply_meta(node,meta,p_loader,connections,r_err,owner_map);
	if (r_err!=OK)
		return NULL;

	Dictionary d=meta;

	if (p_root) {
		NodePath path=d.has("path")?d["path"].operator NodePath():NodePath("."); 

		Node *parent=p_root->get_node(path);
		if (!parent) {
			memdelete(node);
			r_err=ERR_FILE_CORRUPT;
			ERR_FAIL_COND_V(!parent,NULL);
		}

		parent->add_child(node);

		if (d.has("owner_id")) {
			//is owner
			owner_map[d["owner_id"]]=node;
			if (d.has("instance"))
				node->set_filename(d["instance"]);

		}

		if (d.has("owner")) {

			uint32_t owner = d["owner"];
			ERR_FAIL_COND_V(!owner_map.has(owner),NULL);
			node->set_owner(owner_map[owner]);
		} else {

			node->set_owner(p_root);
		}
	}
	
	return node;
}

void SceneFormatLoaderObject::_apply_connections(List<ConnectionItem>& connections) {

	int idx=0;
	for (List<ConnectionItem>::Element *E=connections.front();E;E=E->next()) {

		ConnectionItem &ci=E->get();
		Node *target = ci.node->get_node(ci.target);
		ERR_CONTINUE(!target);
		ci.node->connect(ci.signal,target,ci.method,ci.binds,(ci.realtime?0:Object::CONNECT_DEFERRED)|Object::CONNECT_PERSIST);
		idx++;
	}

}

Node* SceneFormatLoaderObject::load(const String &p_path,bool p_save_instance_state) {

	List<ConnectionItem> connections;

	String extension=p_path.extension();
	if (extension=="scn")
		extension="bin";
	if (extension=="xscn")
		extension="xml";

	String local_path = Globals::get_singleton()->localize_path(p_path);

	ObjectFormatLoader *loader = ObjectLoader::instance_format_loader(local_path,"SCENE",extension);

	ERR_EXPLAIN("Couldn't load scene: "+p_path);
	ERR_FAIL_COND_V(!loader,NULL);
	
	Node *root=NULL;
	Map<uint32_t,Node*> owner_map;
	
	while(true) {
	
		Object *obj=NULL;
		Variant metav;
		Error r_err=loader->load(&obj,metav);

		if (r_err == ERR_SKIP) {
			continue;
		};
		
		if (r_err==ERR_FILE_EOF) {
			memdelete(loader);
			ERR_FAIL_COND_V(!root,NULL);
			_apply_connections(connections);
			return root;
		}

		if (r_err || (!obj && metav.get_type()==Variant::NIL)) {
			memdelete(loader);
			ERR_EXPLAIN("Object Loader Failed for Scene: "+p_path)	;
			ERR_FAIL_COND_V( r_err, NULL);
			ERR_EXPLAIN("Object Loader Failed for Scene: "+p_path)	;
			ERR_FAIL_COND_V( !obj && metav.get_type()==Variant::NIL,NULL);
		}
		
		if (obj) {
			if (obj->cast_to<Node>()) {
			
				Error err;
				Node* node = load_node(obj, metav, root, loader,connections,err,p_save_instance_state,owner_map);
				if (err)
					memdelete(loader);
					
				ERR_FAIL_COND_V( err, NULL );
				if (!root)
					root=node;
			} else {
			
				memdelete(loader);
				ERR_FAIL_V( NULL );
				
			}	
		} else {
		
			// check for instance
			Dictionary meta=metav;
			if (meta.has("instance")) {
				if (!root) {
				
					memdelete(loader);
					ERR_FAIL_COND_V(!root,NULL);
				}
				
				String path = meta["instance"];

				if (path.find("://")==-1 && path.is_rel_path()) {
					// path is relative to file being loaded, so convert to a resource path
					path=Globals::get_singleton()->localize_path(
							local_path.get_base_dir()+"/"+path);
				}				


				Node *scene = SceneLoader::load(path);

				if (!scene) {

					Ref<PackedScene> sd = ResourceLoader::load(path);
					if (sd.is_valid()) {

						scene=sd->instance();
					}
				}

				
				if (!scene) {
				
					memdelete(loader);
					ERR_FAIL_COND_V(!scene,NULL);
				}

				if (p_save_instance_state)
					scene->generate_instance_state();


				Error err;
				_apply_meta(scene,metav,loader,connections,err,owner_map);
				if (err!=OK) {
					memdelete(loader);
					ERR_FAIL_COND_V(err!=OK,NULL);
				}
				
				Node *parent=root;
				
				if (meta.has("path"))
					parent=root->get_node(meta["path"]);
					
				
				if (!parent) {
				
					memdelete(loader);
					ERR_FAIL_COND_V(!parent,NULL);
				}
				

				if (meta.has("override_names") && meta.has("override_values")) {
				
					DVector<String> override_names=meta["override_names"];
					Array override_values=meta["override_values"];
					
					int len = override_names.size();
					if ( len > 0 && len == override_values.size() ) {
					
						DVector<String>::Read names = override_names.read();
						
						for(int i=0;i<len;i++) {
						
							scene->set(names[i],override_values[i]);
						}
					
					}
				
				}
				
				scene->set_filename(path);

				parent->add_child(scene);
				scene->set_owner(root);				
			}
		}
	}
		
	return NULL;	
}

void SceneFormatLoaderObject::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("xml");
	p_extensions->push_back("scn");
	p_extensions->push_back("xscn");

//	ObjectLoader::get_recognized_extensions(p_extensions);
	
}



///////////////////////////////////////////////////


void SceneInteractiveLoaderObject::_apply_meta(Node *node,  const Variant&meta, ObjectFormatLoader *p_loader,List<ConnectionItem>& connections,Error& r_err,Map<uint32_t,Node*>& owner_map) {

	r_err = OK;


	Dictionary d=meta;

	if (!d.has("name")) {

		r_err=ERR_WTF;
		memdelete(node);
		ERR_FAIL_COND(!d.has("name"));
	}


	node->set_name(d["name"]);
	int connection_count=d.has("connection_count")?d["connection_count"].operator int():0;


	for (int i=0;i<connection_count;i++) {

		Dictionary cd=d["connection/"+itos(i+1)];

		ERR_CONTINUE(!cd.has("target"));
		ERR_CONTINUE(!cd.has("method"));
		ERR_CONTINUE(!cd.has("realtime"));
		ERR_CONTINUE(!cd.has("signal"));

		ConnectionItem ci;

		ci.node=node;
		ci.target=cd["target"];
		ci.method=cd["method"];
		ci.signal=cd["signal"];
		ci.realtime=cd["realtime"];
		if (cd.has("binds"))
			ci.binds=cd["binds"];

		connections.push_back(ci);

	}

	DVector<String> groups=d.has("groups")?d["groups"].operator DVector<String>():DVector<String>();
	for (int i=0;i<groups.size();i++) {

		node->add_to_group(groups[i],true);
	}

}



Node* SceneInteractiveLoaderObject::load_node(Object *obj, const Variant& meta, Node *p_root, ObjectFormatLoader *p_loader,List<ConnectionItem>& connections,Error& r_err,bool p_save_instance_state,Map<uint32_t,Node*>& owner_map) {

	r_err = OK;

	Node *node=obj->cast_to<Node>();

	_apply_meta(node,meta,p_loader,connections,r_err,owner_map);
	if (r_err!=OK)
		return NULL;

	Dictionary d=meta;

	if (p_root) {
		NodePath path=d.has("path")?d["path"].operator NodePath():NodePath(".");

		Node *parent=p_root->get_node(path);
		if (!parent) {
			memdelete(node);
			r_err=ERR_FILE_CORRUPT;
			ERR_FAIL_COND_V(!parent,NULL);
		}

		parent->add_child(node);

		if (d.has("owner_id")) {
			//is owner
			owner_map[d["owner_id"]]=node;
			if (d.has("instance"))
				node->set_filename(d["instance"]);

		}

		if (d.has("owner")) {

			uint32_t owner = d["owner"];
			ERR_FAIL_COND_V(!owner_map.has(owner),NULL);
			node->set_owner(owner_map[owner]);
		} else {

			node->set_owner(p_root);
		}
	}

	return node;
}

void SceneInteractiveLoaderObject::_apply_connections(List<ConnectionItem>& connections) {

	int idx=0;
	for (List<ConnectionItem>::Element *E=connections.front();E;E=E->next()) {

		ConnectionItem &ci=E->get();
		Node *target = ci.node->get_node(ci.target);
		ERR_CONTINUE(!target);
		ci.node->connect(ci.signal,target,ci.method,ci.binds,(ci.realtime?0:Object::CONNECT_DEFERRED)|Object::CONNECT_PERSIST);
		idx++;
	}

}

SceneInteractiveLoaderObject::SceneInteractiveLoaderObject(const String &p_path,bool p_save_root_state) {

	error=OK;
	path=p_path;
	save_instance_state=p_save_root_state;
	node_path=p_path;
	root=NULL;
	stage_max=1;
	stage=0;


	String extension=p_path.extension();
	if (extension=="scn")
		extension="bin";
	if (extension=="xscn")
		extension="xml";

	local_path = Globals::get_singleton()->localize_path(p_path);

	loader = ObjectLoader::instance_format_loader(local_path,"SCENE",extension);

	if (!loader) {

		error=ERR_CANT_OPEN;
	}
	ERR_EXPLAIN("Couldn't load scene: "+p_path);
	ERR_FAIL_COND(!loader);

}



void SceneInteractiveLoaderObject::set_local_path(const String& p_local_path) {

	node_path=p_local_path;
}

Node *SceneInteractiveLoaderObject::get_scene() {

	if (error==ERR_FILE_EOF)
		return root;
	return NULL;
}
Error SceneInteractiveLoaderObject::poll() {

	if (error!=OK)
		return error;

	Object *obj=NULL;
	Variant metav;
	Error r_err=loader->load(&obj,metav);


	if (r_err == ERR_SKIP) {
		stage++;
		return OK;
	};

	if (r_err==ERR_FILE_EOF) {
		memdelete(loader);
		error=ERR_FILE_CORRUPT;
		ERR_FAIL_COND_V(!root,ERR_FILE_CORRUPT);
		_apply_connections(connections);
		error=ERR_FILE_EOF;
		if (root)
			root->set_filename(node_path);
		return error;
	}

	if (r_err || (!obj && metav.get_type()==Variant::NIL)) {
		memdelete(loader);
		error=ERR_FILE_CORRUPT;
		ERR_EXPLAIN("Object Loader Failed for Scene: "+path);
		ERR_FAIL_COND_V( r_err, ERR_FILE_CORRUPT);
		ERR_EXPLAIN("Object Loader Failed for Scene: "+path);
		ERR_FAIL_COND_V( !obj && metav.get_type()==Variant::NIL,ERR_FILE_CORRUPT);
	}

	if (obj) {
		if (obj->cast_to<Node>()) {

			Error err;
			Node* node = load_node(obj, metav, root, loader,connections,err,save_instance_state,owner_map);
			if (err) {
				error=ERR_FILE_CORRUPT;
				memdelete(loader);
			}

			ERR_FAIL_COND_V( err, ERR_FILE_CORRUPT );
			if (!root)
				root=node;
		} else {

			error=ERR_FILE_CORRUPT;
			memdelete(loader);
			ERR_EXPLAIN("Loaded something not a node.. (?)");
			ERR_FAIL_V( ERR_FILE_CORRUPT );

		}
	} else {

		// check for instance
		Dictionary meta=metav;
		if (meta.has("instance")) {

			if (!root) {

				error=ERR_FILE_CORRUPT;
				memdelete(loader);
				ERR_FAIL_COND_V(!root,ERR_FILE_CORRUPT);
			}

			String path = meta["instance"];

			if (path.find("://")==-1 && path.is_rel_path()) {
				// path is relative to file being loaded, so convert to a resource path
				path=Globals::get_singleton()->localize_path(
						local_path.get_base_dir()+"/"+path);
			}

			Node *scene = SceneLoader::load(path);

			if (!scene) {

				error=ERR_FILE_CORRUPT;
				memdelete(loader);
				ERR_FAIL_COND_V(!scene,ERR_FILE_CORRUPT);
			}

			if (save_instance_state)
				scene->generate_instance_state();


			Error err;
			_apply_meta(scene,metav,loader,connections,err,owner_map);
			if (err!=OK) {
				error=ERR_FILE_CORRUPT;
				memdelete(loader);
				ERR_FAIL_COND_V(err!=OK,ERR_FILE_CORRUPT);
			}

			Node *parent=root;

			if (meta.has("path"))
				parent=root->get_node(meta["path"]);


			if (!parent) {

				error=ERR_FILE_CORRUPT;
				memdelete(loader);
				ERR_FAIL_COND_V(!parent,ERR_FILE_CORRUPT);
			}


			if (meta.has("override_names") && meta.has("override_values")) {

				DVector<String> override_names=meta["override_names"];
				Array override_values=meta["override_values"];

				int len = override_names.size();
				if ( len > 0 && len == override_values.size() ) {

					DVector<String>::Read names = override_names.read();

					for(int i=0;i<len;i++) {

						scene->set(names[i],override_values[i]);
					}

				}

			}

			scene->set_filename(path);

			parent->add_child(scene);
			scene->set_owner(root);
		}
	}

	stage++;
	error=OK;
	return error;

}
int SceneInteractiveLoaderObject::get_stage() const {

	return stage;
}
int SceneInteractiveLoaderObject::get_stage_count() const {

	return stage_max;
}


#endif
