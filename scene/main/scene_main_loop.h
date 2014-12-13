/*************************************************************************/
/*  scene_main_loop.h                                                    */
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
#ifndef SCENE_MAIN_LOOP_H
#define SCENE_MAIN_LOOP_H


#include "os/main_loop.h"
#include "scene/resources/world.h"
#include "scene/resources/world_2d.h"
#include "scene/main/scene_singleton.h"
#include "os/thread_safe.h"
#include "self_list.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


class SceneTree;

class Node;
class Viewport;

class SceneTree : public MainLoop {

	_THREAD_SAFE_CLASS_

	OBJ_TYPE( SceneTree, MainLoop );
public:


	enum StretchMode {

		STRETCH_MODE_DISABLED,
		STRETCH_MODE_2D,
		STRETCH_MODE_VIEWPORT,
	};

	enum StretchAspect {

		STRETCH_ASPECT_IGNORE,
		STRETCH_ASPECT_KEEP,
		STRETCH_ASPECT_KEEP_WIDTH,
		STRETCH_ASPECT_KEEP_HEIGHT,
	};
private:


	struct Group {

		Vector<Node*> nodes;
		uint64_t last_tree_version;		
		Group() { last_tree_version=0; };
	};

	Viewport *root;

	uint64_t tree_version;
	float fixed_process_time;
	float idle_process_time;
	bool accept_quit;
	uint32_t last_id;

	bool editor_hint;
	bool pause;
	int root_lock;

	Map<StringName,Group> group_map;
	bool _quit;
	bool initialized;
	bool input_handled;
	Size2 last_screen_size;
	StringName tree_changed_name;
	StringName node_removed_name;


	int64_t current_frame;
	int node_count;

#ifdef TOOLS_ENABLED
	Node *edited_scene_root;
#endif
	struct UGCall {

		StringName group;
		StringName call;

		bool operator<(const UGCall& p_with) const { return group==p_with.group?call<p_with.call:group<p_with.group; }
	};

	//safety for when a node is deleted while a group is being called
	int call_lock;
	Set<Node*> call_skip; //skip erased nodes


	StretchMode stretch_mode;
	StretchAspect stretch_aspect;
	Size2i stretch_min;

	void _update_root_rect();

	List<ObjectID> delete_queue;

	Map<UGCall,Vector<Variant> > unique_group_calls;
	bool ugc_locked;
	void _flush_ugc();
	void _flush_transform_notifications();

	void _update_group_order(Group& g);
	void _update_listener();

	Array _get_nodes_in_group(const StringName& p_group);


	//void _call_group(uint32_t p_call_flags,const StringName& p_group,const StringName& p_function,const Variant& p_arg1,const Variant& p_arg2);

friend class Node;

	void tree_changed();
	void node_removed(Node *p_node);


	void add_to_group(const StringName& p_group, Node *p_node);
	void remove_from_group(const StringName& p_group, Node *p_node);

	void _notify_group_pause(const StringName& p_group,int p_notification);
	void _call_input_pause(const StringName& p_group,const StringName& p_method,const InputEvent& p_input);
	Variant _call_group(const Variant** p_args, int p_argcount, Variant::CallError& r_error);


	static void _debugger_request_tree(void *self);
	void _flush_delete_queue();
//optimization
friend class CanvasItem;
friend class Spatial;
friend class Viewport;

	SelfList<Node>::List xform_change_list;

protected:

	void _notification(int p_notification);
	static void _bind_methods();

public:

	enum {
		NOTIFICATION_TRANSFORM_CHANGED=29
	};

	enum CallGroupFlags {
		GROUP_CALL_DEFAULT=0,
		GROUP_CALL_REVERSE=1,
		GROUP_CALL_REALTIME=2,
		GROUP_CALL_UNIQUE=4,
		GROUP_CALL_MULIILEVEL=8,
	};

	_FORCE_INLINE_ Viewport *get_root() const { return root; }

	uint32_t get_last_event_id() const;

	void call_group(uint32_t p_call_flags,const StringName& p_group,const StringName& p_function,VARIANT_ARG_LIST);
	void notify_group(uint32_t p_call_flags,const StringName& p_group,int p_notification);
	void set_group(uint32_t p_call_flags,const StringName& p_group,const String& p_name,const Variant& p_value);


	virtual void input_text( const String& p_text );
	virtual void input_event( const InputEvent& p_event );
	virtual void init();

	virtual bool iteration(float p_time);
	virtual bool idle(float p_time);

	virtual void finish();

	void set_auto_accept_quit(bool p_enable);

	void quit();

	void set_input_as_handled();
	_FORCE_INLINE_ float get_fixed_process_time() const { return fixed_process_time; }
	_FORCE_INLINE_ float get_idle_process_time() const { return idle_process_time; }

	void set_editor_hint(bool p_enabled);
	bool is_editor_hint() const;

	void set_pause(bool p_enabled);
	bool is_paused() const;

	void set_camera(const RID& p_camera);
	RID get_camera() const;

	int64_t get_frame() const;

	int get_node_count() const;

	void queue_delete(Object *p_object);

	void get_nodes_in_group(const StringName& p_group,List<Node*> *p_list);

	void set_screen_stretch(StretchMode p_mode,StretchAspect p_aspect,const Size2 p_minsize);

	//void change_scene(const String& p_path);
	//Node *get_loaded_scene();

#ifdef TOOLS_ENABLED
	void set_edited_scene_root(Node *p_node);
	Node *get_edited_scene_root() const;
#endif

	SceneTree();
	~SceneTree();

};


VARIANT_ENUM_CAST( SceneTree::StretchMode );
VARIANT_ENUM_CAST( SceneTree::StretchAspect );



#endif
