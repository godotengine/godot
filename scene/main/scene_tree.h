/*************************************************************************/
/*  scene_tree.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef SCENE_TREE_H
#define SCENE_TREE_H

#include "core/multiplayer/multiplayer_api.h"
#include "core/os/main_loop.h"
#include "core/os/thread_safe.h"
#include "core/templates/self_list.h"
#include "scene/resources/mesh.h"
#include "scene/resources/world_2d.h"
#include "scene/resources/world_3d.h"

#undef Window

class PackedScene;
class Node;
class Window;
class Material;
class Mesh;
class SceneDebugger;
class Tween;

class SceneTreeTimer : public RefCounted {
	GDCLASS(SceneTreeTimer, RefCounted);

	double time_left = 0.0;
	bool process_always = true;
	bool ignore_time_scale = false;

protected:
	static void _bind_methods();

public:
	void set_time_left(double p_time);
	double get_time_left() const;

	void set_process_always(bool p_process_always);
	bool is_process_always();

	void set_ignore_time_scale(bool p_ignore);
	bool is_ignore_time_scale();

	void release_connections();

	SceneTreeTimer();
};

class SceneTree : public MainLoop {
	_THREAD_SAFE_CLASS_

	GDCLASS(SceneTree, MainLoop);

public:
	typedef void (*IdleCallback)();

private:
	struct Group {
		Vector<Node *> nodes;
		bool changed = false;
	};

	Window *root = nullptr;

	uint64_t tree_version = 1;
	double physics_process_time = 1.0;
	double process_time = 1.0;
	bool accept_quit = true;
	bool quit_on_go_back = true;

#ifdef DEBUG_ENABLED
	bool debug_collisions_hint = false;
	bool debug_navigation_hint = false;
#endif
	bool paused = false;
	int root_lock = 0;

	Map<StringName, Group> group_map;
	bool _quit = false;
	bool initialized = false;

	StringName tree_changed_name = "tree_changed";
	StringName node_added_name = "node_added";
	StringName node_removed_name = "node_removed";
	StringName node_renamed_name = "node_renamed";

	int64_t current_frame = 0;
	int node_count = 0;

#ifdef TOOLS_ENABLED
	Node *edited_scene_root;
#endif
	struct UGCall {
		StringName group;
		StringName call;

		bool operator<(const UGCall &p_with) const { return group == p_with.group ? call < p_with.call : group < p_with.group; }
	};

	// Safety for when a node is deleted while a group is being called.
	int call_lock = 0;
	Set<Node *> call_skip; // Skip erased nodes.

	List<ObjectID> delete_queue;

	Map<UGCall, Vector<Variant>> unique_group_calls;
	bool ugc_locked = false;
	void _flush_ugc();

	_FORCE_INLINE_ void _update_group_order(Group &g, bool p_use_priority = false);

	Array _get_nodes_in_group(const StringName &p_group);

	Node *current_scene;

	Color debug_collisions_color;
	Color debug_collision_contact_color;
	Color debug_navigation_color;
	Color debug_navigation_disabled_color;
	Ref<ArrayMesh> debug_contact_mesh;
	Ref<Material> navigation_material;
	Ref<Material> navigation_disabled_material;
	Ref<Material> collision_material;
	int collision_debug_contacts;

	void _change_scene(Node *p_to);
	//void _call_group(uint32_t p_call_flags,const StringName& p_group,const StringName& p_function,const Variant& p_arg1,const Variant& p_arg2);

	List<Ref<SceneTreeTimer>> timers;
	List<Ref<Tween>> tweens;

	///network///

	Ref<MultiplayerAPI> multiplayer;
	bool multiplayer_poll = true;

	static SceneTree *singleton;
	friend class Node;

	void tree_changed();
	void node_added(Node *p_node);
	void node_removed(Node *p_node);
	void node_renamed(Node *p_node);
	void process_tweens(float p_delta, bool p_physics_frame);

	Group *add_to_group(const StringName &p_group, Node *p_node);
	void remove_from_group(const StringName &p_group, Node *p_node);
	void make_group_changed(const StringName &p_group);

	void _notify_group_pause(const StringName &p_group, int p_notification);
	Variant _call_group_flags(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Variant _call_group(const Variant **p_args, int p_argcount, Callable::CallError &r_error);

	void _flush_delete_queue();
	// Optimization.
	friend class CanvasItem;
	friend class Node3D;
	friend class Viewport;

	SelfList<Node>::List xform_change_list;

#ifdef DEBUG_ENABLED // No live editor in release build.
	friend class LiveEditor;
#endif

	enum {
		MAX_IDLE_CALLBACKS = 256
	};

	static IdleCallback idle_callbacks[MAX_IDLE_CALLBACKS];
	static int idle_callback_count;
	void _call_idle_callbacks();

	void _main_window_focus_in();
	void _main_window_close();
	void _main_window_go_back();

	enum CallInputType {
		CALL_INPUT_TYPE_INPUT,
		CALL_INPUT_TYPE_UNHANDLED_INPUT,
		CALL_INPUT_TYPE_UNHANDLED_KEY_INPUT,
	};

	//used by viewport
	void _call_input_pause(const StringName &p_group, CallInputType p_call_type, const Ref<InputEvent> &p_input, Viewport *p_viewport);

protected:
	void _notification(int p_notification);
	static void _bind_methods();

public:
	enum {
		NOTIFICATION_TRANSFORM_CHANGED = 2000
	};

	enum GroupCallFlags {
		GROUP_CALL_DEFAULT = 0,
		GROUP_CALL_REVERSE = 1,
		GROUP_CALL_REALTIME = 2,
		GROUP_CALL_UNIQUE = 4,
	};

	_FORCE_INLINE_ Window *get_root() const { return root; }

	void call_group_flags(uint32_t p_call_flags, const StringName &p_group, const StringName &p_function, VARIANT_ARG_LIST);
	void notify_group_flags(uint32_t p_call_flags, const StringName &p_group, int p_notification);
	void set_group_flags(uint32_t p_call_flags, const StringName &p_group, const String &p_name, const Variant &p_value);

	void call_group(const StringName &p_group, const StringName &p_function, VARIANT_ARG_LIST);
	void notify_group(const StringName &p_group, int p_notification);
	void set_group(const StringName &p_group, const String &p_name, const Variant &p_value);

	void flush_transform_notifications();

	virtual void initialize() override;

	virtual bool physics_process(double p_time) override;
	virtual bool process(double p_time) override;

	virtual void finalize() override;

	void set_auto_accept_quit(bool p_enable);
	void set_quit_on_go_back(bool p_enable);

	void quit(int p_exit_code = EXIT_SUCCESS);

	_FORCE_INLINE_ double get_physics_process_time() const { return physics_process_time; }
	_FORCE_INLINE_ double get_process_time() const { return process_time; }

#ifdef TOOLS_ENABLED
	bool is_node_being_edited(const Node *p_node) const;
#else
	bool is_node_being_edited(const Node *p_node) const { return false; }
#endif

	void set_pause(bool p_enabled);
	bool is_paused() const;

#ifdef DEBUG_ENABLED
	void set_debug_collisions_hint(bool p_enabled);
	bool is_debugging_collisions_hint() const;

	void set_debug_navigation_hint(bool p_enabled);
	bool is_debugging_navigation_hint() const;
#else
	void set_debug_collisions_hint(bool p_enabled) {}
	bool is_debugging_collisions_hint() const { return false; }

	void set_debug_navigation_hint(bool p_enabled) {}
	bool is_debugging_navigation_hint() const { return false; }
#endif

	void set_debug_collisions_color(const Color &p_color);
	Color get_debug_collisions_color() const;

	void set_debug_collision_contact_color(const Color &p_color);
	Color get_debug_collision_contact_color() const;

	void set_debug_navigation_color(const Color &p_color);
	Color get_debug_navigation_color() const;

	void set_debug_navigation_disabled_color(const Color &p_color);
	Color get_debug_navigation_disabled_color() const;

	Ref<Material> get_debug_navigation_material();
	Ref<Material> get_debug_navigation_disabled_material();
	Ref<Material> get_debug_collision_material();
	Ref<ArrayMesh> get_debug_contact_mesh();

	int get_collision_debug_contact_count() { return collision_debug_contacts; }

	int64_t get_frame() const;

	int get_node_count() const;

	void queue_delete(Object *p_object);

	void get_nodes_in_group(const StringName &p_group, List<Node *> *p_list);
	Node *get_first_node_in_group(const StringName &p_group);
	bool has_group(const StringName &p_identifier) const;

	//void change_scene(const String& p_path);
	//Node *get_loaded_scene();

	void set_edited_scene_root(Node *p_node);
	Node *get_edited_scene_root() const;

	void set_current_scene(Node *p_scene);
	Node *get_current_scene() const;
	Error change_scene(const String &p_path);
	Error change_scene_to(const Ref<PackedScene> &p_scene);
	Error reload_current_scene();

	Ref<SceneTreeTimer> create_timer(double p_delay_sec, bool p_process_always = true);
	Ref<Tween> create_tween();
	Array get_processed_tweens();

	//used by Main::start, don't use otherwise
	void add_current_scene(Node *p_current);

	static SceneTree *get_singleton() { return singleton; }

	void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;

	//network API

	Ref<MultiplayerAPI> get_multiplayer() const;
	void set_multiplayer_poll_enabled(bool p_enabled);
	bool is_multiplayer_poll_enabled() const;
	void set_multiplayer(Ref<MultiplayerAPI> p_multiplayer);

	static void add_idle_callback(IdleCallback p_callback);

	//default texture settings

	SceneTree();
	~SceneTree();
};

VARIANT_ENUM_CAST(SceneTree::GroupCallFlags);

#endif // SCENE_TREE_H
