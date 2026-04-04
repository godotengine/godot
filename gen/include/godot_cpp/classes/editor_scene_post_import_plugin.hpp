/**************************************************************************/
/*  editor_scene_post_import_plugin.hpp                                   */
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
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Node;
class Resource;
class StringName;

class EditorScenePostImportPlugin : public RefCounted {
	GDEXTENSION_CLASS(EditorScenePostImportPlugin, RefCounted)

public:
	enum InternalImportCategory {
		INTERNAL_IMPORT_CATEGORY_NODE = 0,
		INTERNAL_IMPORT_CATEGORY_MESH_3D_NODE = 1,
		INTERNAL_IMPORT_CATEGORY_MESH = 2,
		INTERNAL_IMPORT_CATEGORY_MATERIAL = 3,
		INTERNAL_IMPORT_CATEGORY_ANIMATION = 4,
		INTERNAL_IMPORT_CATEGORY_ANIMATION_NODE = 5,
		INTERNAL_IMPORT_CATEGORY_SKELETON_3D_NODE = 6,
		INTERNAL_IMPORT_CATEGORY_MAX = 7,
	};

	Variant get_option_value(const StringName &p_name) const;
	void add_import_option(const String &p_name, const Variant &p_value);
	void add_import_option_advanced(Variant::Type p_type, const String &p_name, const Variant &p_default_value, PropertyHint p_hint = (PropertyHint)0, const String &p_hint_string = String(), int32_t p_usage_flags = 6);
	virtual void _get_internal_import_options(int32_t p_category);
	virtual Variant _get_internal_option_visibility(int32_t p_category, bool p_for_animation, const String &p_option) const;
	virtual Variant _get_internal_option_update_view_required(int32_t p_category, const String &p_option) const;
	virtual void _internal_process(int32_t p_category, Node *p_base_node, Node *p_node, const Ref<Resource> &p_resource);
	virtual void _get_import_options(const String &p_path);
	virtual Variant _get_option_visibility(const String &p_path, bool p_for_animation, const String &p_option) const;
	virtual void _pre_process(Node *p_scene);
	virtual void _post_process(Node *p_scene);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_internal_import_options), decltype(&T::_get_internal_import_options)>) {
			BIND_VIRTUAL_METHOD(T, _get_internal_import_options, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_internal_option_visibility), decltype(&T::_get_internal_option_visibility)>) {
			BIND_VIRTUAL_METHOD(T, _get_internal_option_visibility, 3811255416);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_internal_option_update_view_required), decltype(&T::_get_internal_option_update_view_required)>) {
			BIND_VIRTUAL_METHOD(T, _get_internal_option_update_view_required, 3957349696);
		}
		if constexpr (!std::is_same_v<decltype(&B::_internal_process), decltype(&T::_internal_process)>) {
			BIND_VIRTUAL_METHOD(T, _internal_process, 3641982463);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_import_options), decltype(&T::_get_import_options)>) {
			BIND_VIRTUAL_METHOD(T, _get_import_options, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_option_visibility), decltype(&T::_get_option_visibility)>) {
			BIND_VIRTUAL_METHOD(T, _get_option_visibility, 298836892);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pre_process), decltype(&T::_pre_process)>) {
			BIND_VIRTUAL_METHOD(T, _pre_process, 1078189570);
		}
		if constexpr (!std::is_same_v<decltype(&B::_post_process), decltype(&T::_post_process)>) {
			BIND_VIRTUAL_METHOD(T, _post_process, 1078189570);
		}
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(EditorScenePostImportPlugin::InternalImportCategory);

