/**************************************************************************/
/*  resource.hpp                                                          */
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
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Node;

class Resource : public RefCounted {
	GDEXTENSION_CLASS(Resource, RefCounted)

public:
	enum DeepDuplicateMode {
		DEEP_DUPLICATE_NONE = 0,
		DEEP_DUPLICATE_INTERNAL = 1,
		DEEP_DUPLICATE_ALL = 2,
	};

	void set_path(const String &p_path);
	void take_over_path(const String &p_path);
	String get_path() const;
	void set_path_cache(const String &p_path);
	void set_name(const String &p_name);
	String get_name() const;
	RID get_rid() const;
	void set_local_to_scene(bool p_enable);
	bool is_local_to_scene() const;
	Node *get_local_scene() const;
	void setup_local_to_scene();
	void reset_state();
	void set_id_for_path(const String &p_path, const String &p_id);
	String get_id_for_path(const String &p_path) const;
	bool is_built_in() const;
	static String generate_scene_unique_id();
	void set_scene_unique_id(const String &p_id);
	String get_scene_unique_id() const;
	void emit_changed();
	Ref<Resource> duplicate(bool p_deep = false) const;
	Ref<Resource> duplicate_deep(Resource::DeepDuplicateMode p_deep_subresources_mode = (Resource::DeepDuplicateMode)1) const;
	virtual void _setup_local_to_scene();
	virtual RID _get_rid() const;
	virtual void _reset_state();
	virtual void _set_path_cache(const String &p_path) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_setup_local_to_scene), decltype(&T::_setup_local_to_scene)>) {
			BIND_VIRTUAL_METHOD(T, _setup_local_to_scene, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_rid), decltype(&T::_get_rid)>) {
			BIND_VIRTUAL_METHOD(T, _get_rid, 2944877500);
		}
		if constexpr (!std::is_same_v<decltype(&B::_reset_state), decltype(&T::_reset_state)>) {
			BIND_VIRTUAL_METHOD(T, _reset_state, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_path_cache), decltype(&T::_set_path_cache)>) {
			BIND_VIRTUAL_METHOD(T, _set_path_cache, 3089850668);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Resource::DeepDuplicateMode);

