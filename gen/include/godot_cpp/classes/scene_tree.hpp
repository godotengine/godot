/**************************************************************************/
/*  scene_tree.hpp                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/main_loop.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class MultiplayerAPI;
class Node;
class Object;
class PackedScene;
class SceneTreeTimer;
class String;
class StringName;
class Tween;
class Variant;
class Window;

class SceneTree : public MainLoop {
	GDEXTENSION_CLASS(SceneTree, MainLoop)

public:
	enum GroupCallFlags {
		GROUP_CALL_DEFAULT = 0,
		GROUP_CALL_REVERSE = 1,
		GROUP_CALL_DEFERRED = 2,
		GROUP_CALL_UNIQUE = 4,
	};

	Window *get_root() const;
	bool has_group(const StringName &p_name) const;
	bool is_accessibility_enabled() const;
	bool is_accessibility_supported() const;
	bool is_auto_accept_quit() const;
	void set_auto_accept_quit(bool p_enabled);
	bool is_quit_on_go_back() const;
	void set_quit_on_go_back(bool p_enabled);
	void set_debug_collisions_hint(bool p_enable);
	bool is_debugging_collisions_hint() const;
	void set_debug_paths_hint(bool p_enable);
	bool is_debugging_paths_hint() const;
	void set_debug_navigation_hint(bool p_enable);
	bool is_debugging_navigation_hint() const;
	void set_edited_scene_root(Node *p_scene);
	Node *get_edited_scene_root() const;
	void set_pause(bool p_enable);
	bool is_paused() const;
	Ref<SceneTreeTimer> create_timer(double p_time_sec, bool p_process_always = true, bool p_process_in_physics = false, bool p_ignore_time_scale = false);
	Ref<Tween> create_tween();
	TypedArray<Ref<Tween>> get_processed_tweens();
	int32_t get_node_count() const;
	int64_t get_frame() const;
	void quit(int32_t p_exit_code = 0);
	void set_physics_interpolation_enabled(bool p_enabled);
	bool is_physics_interpolation_enabled() const;
	void queue_delete(Object *p_obj);

private:
	void call_group_flags_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	void call_group_flags(int64_t p_flags, const StringName &p_group, const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 3 + sizeof...(Args)> variant_args{{ Variant(p_flags), Variant(p_group), Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 3 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		call_group_flags_internal(call_args.data(), variant_args.size());
	}
	void notify_group_flags(uint32_t p_call_flags, const StringName &p_group, int32_t p_notification);
	void set_group_flags(uint32_t p_call_flags, const StringName &p_group, const String &p_property, const Variant &p_value);

private:
	void call_group_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	void call_group(const StringName &p_group, const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 2 + sizeof...(Args)> variant_args{{ Variant(p_group), Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 2 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		call_group_internal(call_args.data(), variant_args.size());
	}
	void notify_group(const StringName &p_group, int32_t p_notification);
	void set_group(const StringName &p_group, const String &p_property, const Variant &p_value);
	TypedArray<Node> get_nodes_in_group(const StringName &p_group);
	Node *get_first_node_in_group(const StringName &p_group);
	int32_t get_node_count_in_group(const StringName &p_group) const;
	void set_current_scene(Node *p_child_node);
	Node *get_current_scene() const;
	Error change_scene_to_file(const String &p_path);
	Error change_scene_to_packed(const Ref<PackedScene> &p_packed_scene);
	Error change_scene_to_node(Node *p_node);
	Error reload_current_scene();
	void unload_current_scene();
	void set_multiplayer(const Ref<MultiplayerAPI> &p_multiplayer, const NodePath &p_root_path = NodePath(""));
	Ref<MultiplayerAPI> get_multiplayer(const NodePath &p_for_path = NodePath("")) const;
	void set_multiplayer_poll_enabled(bool p_enabled);
	bool is_multiplayer_poll_enabled() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		MainLoop::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SceneTree::GroupCallFlags);

