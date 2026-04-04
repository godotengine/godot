/**************************************************************************/
/*  node.hpp                                                              */
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
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class InputEvent;
class MultiplayerAPI;
class SceneTree;
class Tween;
class Viewport;
class Window;

class Node : public Object {
	GDEXTENSION_CLASS(Node, Object)

public:
	enum ProcessMode {
		PROCESS_MODE_INHERIT = 0,
		PROCESS_MODE_PAUSABLE = 1,
		PROCESS_MODE_WHEN_PAUSED = 2,
		PROCESS_MODE_ALWAYS = 3,
		PROCESS_MODE_DISABLED = 4,
	};

	enum ProcessThreadGroup {
		PROCESS_THREAD_GROUP_INHERIT = 0,
		PROCESS_THREAD_GROUP_MAIN_THREAD = 1,
		PROCESS_THREAD_GROUP_SUB_THREAD = 2,
	};

	enum ProcessThreadMessages : uint64_t {
		FLAG_PROCESS_THREAD_MESSAGES = 1,
		FLAG_PROCESS_THREAD_MESSAGES_PHYSICS = 2,
		FLAG_PROCESS_THREAD_MESSAGES_ALL = 3,
	};

	enum PhysicsInterpolationMode {
		PHYSICS_INTERPOLATION_MODE_INHERIT = 0,
		PHYSICS_INTERPOLATION_MODE_ON = 1,
		PHYSICS_INTERPOLATION_MODE_OFF = 2,
	};

	enum DuplicateFlags {
		DUPLICATE_SIGNALS = 1,
		DUPLICATE_GROUPS = 2,
		DUPLICATE_SCRIPTS = 4,
		DUPLICATE_USE_INSTANTIATION = 8,
		DUPLICATE_INTERNAL_STATE = 16,
		DUPLICATE_DEFAULT = 15,
	};

	enum InternalMode {
		INTERNAL_MODE_DISABLED = 0,
		INTERNAL_MODE_FRONT = 1,
		INTERNAL_MODE_BACK = 2,
	};

	enum AutoTranslateMode {
		AUTO_TRANSLATE_MODE_INHERIT = 0,
		AUTO_TRANSLATE_MODE_ALWAYS = 1,
		AUTO_TRANSLATE_MODE_DISABLED = 2,
	};

	static const int NOTIFICATION_ENTER_TREE = 10;
	static const int NOTIFICATION_EXIT_TREE = 11;
	static const int NOTIFICATION_MOVED_IN_PARENT = 12;
	static const int NOTIFICATION_READY = 13;
	static const int NOTIFICATION_PAUSED = 14;
	static const int NOTIFICATION_UNPAUSED = 15;
	static const int NOTIFICATION_PHYSICS_PROCESS = 16;
	static const int NOTIFICATION_PROCESS = 17;
	static const int NOTIFICATION_PARENTED = 18;
	static const int NOTIFICATION_UNPARENTED = 19;
	static const int NOTIFICATION_SCENE_INSTANTIATED = 20;
	static const int NOTIFICATION_DRAG_BEGIN = 21;
	static const int NOTIFICATION_DRAG_END = 22;
	static const int NOTIFICATION_PATH_RENAMED = 23;
	static const int NOTIFICATION_CHILD_ORDER_CHANGED = 24;
	static const int NOTIFICATION_INTERNAL_PROCESS = 25;
	static const int NOTIFICATION_INTERNAL_PHYSICS_PROCESS = 26;
	static const int NOTIFICATION_POST_ENTER_TREE = 27;
	static const int NOTIFICATION_DISABLED = 28;
	static const int NOTIFICATION_ENABLED = 29;
	static const int NOTIFICATION_RESET_PHYSICS_INTERPOLATION = 2001;
	static const int NOTIFICATION_EDITOR_PRE_SAVE = 9001;
	static const int NOTIFICATION_EDITOR_POST_SAVE = 9002;
	static const int NOTIFICATION_WM_MOUSE_ENTER = 1002;
	static const int NOTIFICATION_WM_MOUSE_EXIT = 1003;
	static const int NOTIFICATION_WM_WINDOW_FOCUS_IN = 1004;
	static const int NOTIFICATION_WM_WINDOW_FOCUS_OUT = 1005;
	static const int NOTIFICATION_WM_CLOSE_REQUEST = 1006;
	static const int NOTIFICATION_WM_GO_BACK_REQUEST = 1007;
	static const int NOTIFICATION_WM_SIZE_CHANGED = 1008;
	static const int NOTIFICATION_WM_DPI_CHANGE = 1009;
	static const int NOTIFICATION_VP_MOUSE_ENTER = 1010;
	static const int NOTIFICATION_VP_MOUSE_EXIT = 1011;
	static const int NOTIFICATION_WM_POSITION_CHANGED = 1012;
	static const int NOTIFICATION_OS_MEMORY_WARNING = 2009;
	static const int NOTIFICATION_TRANSLATION_CHANGED = 2010;
	static const int NOTIFICATION_WM_ABOUT = 2011;
	static const int NOTIFICATION_CRASH = 2012;
	static const int NOTIFICATION_OS_IME_UPDATE = 2013;
	static const int NOTIFICATION_APPLICATION_RESUMED = 2014;
	static const int NOTIFICATION_APPLICATION_PAUSED = 2015;
	static const int NOTIFICATION_APPLICATION_FOCUS_IN = 2016;
	static const int NOTIFICATION_APPLICATION_FOCUS_OUT = 2017;
	static const int NOTIFICATION_TEXT_SERVER_CHANGED = 2018;
	static const int NOTIFICATION_ACCESSIBILITY_UPDATE = 3000;
	static const int NOTIFICATION_ACCESSIBILITY_INVALIDATE = 3001;

	static void print_orphan_nodes();
	static TypedArray<int> get_orphan_node_ids();
	void add_sibling(Node *p_sibling, bool p_force_readable_name = false);
	void set_name(const StringName &p_name);
	StringName get_name() const;
	void add_child(Node *p_node, bool p_force_readable_name = false, Node::InternalMode p_internal = (Node::InternalMode)0);
	void remove_child(Node *p_node);
	void reparent(Node *p_new_parent, bool p_keep_global_transform = true);
	int32_t get_child_count(bool p_include_internal = false) const;
	TypedArray<Node> get_children(bool p_include_internal = false) const;
	Node *get_child(int32_t p_idx, bool p_include_internal = false) const;
	bool has_node(const NodePath &p_path) const;
	Node *get_node_internal(const NodePath &p_path) const;
	Node *get_node_or_null(const NodePath &p_path) const;
	Node *get_parent() const;
	Node *find_child(const String &p_pattern, bool p_recursive = true, bool p_owned = true) const;
	TypedArray<Node> find_children(const String &p_pattern, const String &p_type = String(), bool p_recursive = true, bool p_owned = true) const;
	Node *find_parent(const String &p_pattern) const;
	bool has_node_and_resource(const NodePath &p_path) const;
	Array get_node_and_resource(const NodePath &p_path);
	bool is_inside_tree() const;
	bool is_part_of_edited_scene() const;
	bool is_ancestor_of(Node *p_node) const;
	bool is_greater_than(Node *p_node) const;
	NodePath get_path() const;
	NodePath get_path_to(Node *p_node, bool p_use_unique_path = false) const;
	void add_to_group(const StringName &p_group, bool p_persistent = false);
	void remove_from_group(const StringName &p_group);
	bool is_in_group(const StringName &p_group) const;
	void move_child(Node *p_child_node, int32_t p_to_index);
	TypedArray<StringName> get_groups() const;
	void set_owner(Node *p_owner);
	Node *get_owner() const;
	int32_t get_index(bool p_include_internal = false) const;
	void print_tree();
	void print_tree_pretty();
	String get_tree_string();
	String get_tree_string_pretty();
	void set_scene_file_path(const String &p_scene_file_path);
	String get_scene_file_path() const;
	void propagate_notification(int32_t p_what);
	void propagate_call(const StringName &p_method, const Array &p_args = Array(), bool p_parent_first = false);
	void set_physics_process(bool p_enable);
	double get_physics_process_delta_time() const;
	bool is_physics_processing() const;
	double get_process_delta_time() const;
	void set_process(bool p_enable);
	void set_process_priority(int32_t p_priority);
	int32_t get_process_priority() const;
	void set_physics_process_priority(int32_t p_priority);
	int32_t get_physics_process_priority() const;
	bool is_processing() const;
	void set_process_input(bool p_enable);
	bool is_processing_input() const;
	void set_process_shortcut_input(bool p_enable);
	bool is_processing_shortcut_input() const;
	void set_process_unhandled_input(bool p_enable);
	bool is_processing_unhandled_input() const;
	void set_process_unhandled_key_input(bool p_enable);
	bool is_processing_unhandled_key_input() const;
	void set_process_mode(Node::ProcessMode p_mode);
	Node::ProcessMode get_process_mode() const;
	bool can_process() const;
	void set_process_thread_group(Node::ProcessThreadGroup p_mode);
	Node::ProcessThreadGroup get_process_thread_group() const;
	void set_process_thread_messages(BitField<Node::ProcessThreadMessages> p_flags);
	BitField<Node::ProcessThreadMessages> get_process_thread_messages() const;
	void set_process_thread_group_order(int32_t p_order);
	int32_t get_process_thread_group_order() const;
	void queue_accessibility_update();
	RID get_accessibility_element() const;
	void set_display_folded(bool p_fold);
	bool is_displayed_folded() const;
	void set_process_internal(bool p_enable);
	bool is_processing_internal() const;
	void set_physics_process_internal(bool p_enable);
	bool is_physics_processing_internal() const;
	void set_physics_interpolation_mode(Node::PhysicsInterpolationMode p_mode);
	Node::PhysicsInterpolationMode get_physics_interpolation_mode() const;
	bool is_physics_interpolated() const;
	bool is_physics_interpolated_and_enabled() const;
	void reset_physics_interpolation();
	void set_auto_translate_mode(Node::AutoTranslateMode p_mode);
	Node::AutoTranslateMode get_auto_translate_mode() const;
	bool can_auto_translate() const;
	void set_translation_domain_inherited();
	Window *get_window() const;
	Window *get_last_exclusive_window() const;
	SceneTree *get_tree() const;
	Ref<Tween> create_tween();
	Node *duplicate(int32_t p_flags = 15) const;
	void replace_by(Node *p_node, bool p_keep_groups = false);
	void set_scene_instance_load_placeholder(bool p_load_placeholder);
	bool get_scene_instance_load_placeholder() const;
	void set_editable_instance(Node *p_node, bool p_is_editable);
	bool is_editable_instance(Node *p_node) const;
	Viewport *get_viewport() const;
	void queue_free();
	void request_ready();
	bool is_node_ready() const;
	void set_multiplayer_authority(int32_t p_id, bool p_recursive = true);
	int32_t get_multiplayer_authority() const;
	bool is_multiplayer_authority() const;
	Ref<MultiplayerAPI> get_multiplayer() const;
	void rpc_config(const StringName &p_method, const Variant &p_config);
	Variant get_node_rpc_config() const;
	void set_editor_description(const String &p_editor_description);
	String get_editor_description() const;
	void set_unique_name_in_owner(bool p_enable);
	bool is_unique_name_in_owner() const;
	String atr(const String &p_message, const StringName &p_context = String()) const;
	String atr_n(const String &p_message, const StringName &p_plural_message, int32_t p_n, const StringName &p_context = String()) const;

private:
	Error rpc_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	Error rpc(const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return rpc_internal(call_args.data(), variant_args.size());
	}

private:
	Error rpc_id_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	Error rpc_id(int64_t p_peer_id, const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 2 + sizeof...(Args)> variant_args{{ Variant(p_peer_id), Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 2 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return rpc_id_internal(call_args.data(), variant_args.size());
	}
	void update_configuration_warnings();

private:
	Variant call_deferred_thread_group_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	Variant call_deferred_thread_group(const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return call_deferred_thread_group_internal(call_args.data(), variant_args.size());
	}
	void set_deferred_thread_group(const StringName &p_property, const Variant &p_value);
	void notify_deferred_thread_group(int32_t p_what);

private:
	Variant call_thread_safe_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	Variant call_thread_safe(const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 1 + sizeof...(Args)> variant_args{{ Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 1 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		return call_thread_safe_internal(call_args.data(), variant_args.size());
	}
	void set_thread_safe(const StringName &p_property, const Variant &p_value);
	void notify_thread_safe(int32_t p_what);
	virtual void _process(double p_delta);
	virtual void _physics_process(double p_delta);
	virtual void _enter_tree();
	virtual void _exit_tree();
	virtual void _ready();
	virtual PackedStringArray _get_configuration_warnings() const;
	virtual PackedStringArray _get_accessibility_configuration_warnings() const;
	virtual void _input(const Ref<InputEvent> &p_event);
	virtual void _shortcut_input(const Ref<InputEvent> &p_event);
	virtual void _unhandled_input(const Ref<InputEvent> &p_event);
	virtual void _unhandled_key_input(const Ref<InputEvent> &p_event);
	virtual RID _get_focused_accessibility_element() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_process), decltype(&T::_process)>) {
			BIND_VIRTUAL_METHOD(T, _process, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_physics_process), decltype(&T::_physics_process)>) {
			BIND_VIRTUAL_METHOD(T, _physics_process, 373806689);
		}
		if constexpr (!std::is_same_v<decltype(&B::_enter_tree), decltype(&T::_enter_tree)>) {
			BIND_VIRTUAL_METHOD(T, _enter_tree, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_exit_tree), decltype(&T::_exit_tree)>) {
			BIND_VIRTUAL_METHOD(T, _exit_tree, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_ready), decltype(&T::_ready)>) {
			BIND_VIRTUAL_METHOD(T, _ready, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_configuration_warnings), decltype(&T::_get_configuration_warnings)>) {
			BIND_VIRTUAL_METHOD(T, _get_configuration_warnings, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_accessibility_configuration_warnings), decltype(&T::_get_accessibility_configuration_warnings)>) {
			BIND_VIRTUAL_METHOD(T, _get_accessibility_configuration_warnings, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_input), decltype(&T::_input)>) {
			BIND_VIRTUAL_METHOD(T, _input, 3754044979);
		}
		if constexpr (!std::is_same_v<decltype(&B::_shortcut_input), decltype(&T::_shortcut_input)>) {
			BIND_VIRTUAL_METHOD(T, _shortcut_input, 3754044979);
		}
		if constexpr (!std::is_same_v<decltype(&B::_unhandled_input), decltype(&T::_unhandled_input)>) {
			BIND_VIRTUAL_METHOD(T, _unhandled_input, 3754044979);
		}
		if constexpr (!std::is_same_v<decltype(&B::_unhandled_key_input), decltype(&T::_unhandled_key_input)>) {
			BIND_VIRTUAL_METHOD(T, _unhandled_key_input, 3754044979);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_focused_accessibility_element), decltype(&T::_get_focused_accessibility_element)>) {
			BIND_VIRTUAL_METHOD(T, _get_focused_accessibility_element, 2944877500);
		}
	}

	String _to_string() const { return (!get_name().is_empty() ? String(get_name()) + ":" : "") + Object::_to_string(); }

public:
	template <typename T>
	T *get_node(const NodePath &p_path) const { return Object::cast_to<T>(get_node_internal(p_path)); }
};

} // namespace godot

VARIANT_ENUM_CAST(Node::ProcessMode);
VARIANT_ENUM_CAST(Node::ProcessThreadGroup);
VARIANT_BITFIELD_CAST(Node::ProcessThreadMessages);
VARIANT_ENUM_CAST(Node::PhysicsInterpolationMode);
VARIANT_ENUM_CAST(Node::DuplicateFlags);
VARIANT_ENUM_CAST(Node::InternalMode);
VARIANT_ENUM_CAST(Node::AutoTranslateMode);

