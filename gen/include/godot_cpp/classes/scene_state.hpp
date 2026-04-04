/**************************************************************************/
/*  scene_state.hpp                                                       */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PackedScene;

class SceneState : public RefCounted {
	GDEXTENSION_CLASS(SceneState, RefCounted)

public:
	enum GenEditState {
		GEN_EDIT_STATE_DISABLED = 0,
		GEN_EDIT_STATE_INSTANCE = 1,
		GEN_EDIT_STATE_MAIN = 2,
		GEN_EDIT_STATE_MAIN_INHERITED = 3,
	};

	String get_path() const;
	Ref<SceneState> get_base_scene_state() const;
	int32_t get_node_count() const;
	StringName get_node_type(int32_t p_idx) const;
	StringName get_node_name(int32_t p_idx) const;
	NodePath get_node_path(int32_t p_idx, bool p_for_parent = false) const;
	NodePath get_node_owner_path(int32_t p_idx) const;
	bool is_node_instance_placeholder(int32_t p_idx) const;
	String get_node_instance_placeholder(int32_t p_idx) const;
	Ref<PackedScene> get_node_instance(int32_t p_idx) const;
	PackedStringArray get_node_groups(int32_t p_idx) const;
	int32_t get_node_index(int32_t p_idx) const;
	int32_t get_node_property_count(int32_t p_idx) const;
	StringName get_node_property_name(int32_t p_idx, int32_t p_prop_idx) const;
	Variant get_node_property_value(int32_t p_idx, int32_t p_prop_idx) const;
	int32_t get_connection_count() const;
	NodePath get_connection_source(int32_t p_idx) const;
	StringName get_connection_signal(int32_t p_idx) const;
	NodePath get_connection_target(int32_t p_idx) const;
	StringName get_connection_method(int32_t p_idx) const;
	int32_t get_connection_flags(int32_t p_idx) const;
	Array get_connection_binds(int32_t p_idx) const;
	int32_t get_connection_unbinds(int32_t p_idx) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(SceneState::GenEditState);

