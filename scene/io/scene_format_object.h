/*************************************************************************/
/*  scene_format_object.h                                                */
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
#ifndef SCENE_FORMAT_OBJECT_H
#define SCENE_FORMAT_OBJECT_H


#include "scene/main/node.h"
#include "scene/io/scene_saver.h"
#include "scene/io/scene_loader.h"
#include "io/object_saver.h"
#include "io/object_loader.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


#ifdef OLD_SCENE_FORMAT_ENABLED

class SceneFormatSaverObject : public SceneFormatSaver {
	
	void save_node(const Node* p_root,const Node* p_node,const Node* p_owner,ObjectFormatSaver *p_saver,String p_base_path,uint32_t p_flags,Map<const Node*,uint32_t>& owner_id) const;
		
public:
	
	virtual Error save(const String &p_path,const Node* p_scenezz,uint32_t p_flags=0,const Ref<OptimizedSaver> &p_optimizer=Ref<OptimizedSaver>());
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual ~SceneFormatSaverObject() {}
};



class SceneFormatLoaderObject : public SceneFormatLoader {
	

	struct ConnectionItem {
		Node *node;
		NodePath target;
		StringName method;
		StringName signal;
		Vector<Variant> binds;
		bool realtime;
	};

	Node* load_node(Object *obj, const Variant& meta, Node *p_root, ObjectFormatLoader *p_loader,List<ConnectionItem>& connections,Error& r_err,bool p_root_scene_hint,Map<uint32_t,Node*>& owner_map);
	void _apply_connections(List<ConnectionItem>& connections);
	void _apply_meta(Node *node, const Variant& meta, ObjectFormatLoader *p_loader,List<ConnectionItem>& connections,Error& r_err,Map<uint32_t,Node*>& owner_map);

public:
	
	virtual Ref<SceneInteractiveLoader> load_interactive(const String &p_path,bool p_root_scene_hint=false);
	virtual Node* load(const String &p_path,bool p_save_root_state=false);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;

};


class SceneInteractiveLoaderObject : public SceneInteractiveLoader {

	OBJ_TYPE(SceneInteractiveLoaderObject,SceneInteractiveLoader);

	struct ConnectionItem {
		Node *node;
		NodePath target;
		StringName method;
		StringName signal;
		Vector<Variant> binds;
		bool realtime;
	};
	ObjectFormatLoader *loader;
	String path;
	String node_path;
	String local_path;
	Error error;
	bool save_instance_state;
	List<ConnectionItem> connections;
	Map<uint32_t,Node*> owner_map;
	Node *root;
	int stage_max;
	int stage;


	Node* load_node(Object *obj, const Variant& meta, Node *p_root, ObjectFormatLoader *p_loader,List<ConnectionItem>& connections,Error& r_err,bool p_root_scene_hint,Map<uint32_t,Node*>& owner_map);
	void _apply_connections(List<ConnectionItem>& connections);
	void _apply_meta(Node *node, const Variant& meta, ObjectFormatLoader *p_loader,List<ConnectionItem>& connections,Error& r_err,Map<uint32_t,Node*>& owner_map);

friend class SceneFormatLoaderObject;
public:

	virtual void set_local_path(const String& p_local_path);
	virtual Node *get_scene();
	virtual Error poll();
	virtual int get_stage() const;
	virtual int get_stage_count() const;


	SceneInteractiveLoaderObject(const String &p_path,bool p_save_root_state=false);
};



#endif
#endif
