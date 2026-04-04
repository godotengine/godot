/**************************************************************************/
/*  editor_scene_format_importer.hpp                                      */
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
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Dictionary;
class Object;

class EditorSceneFormatImporter : public RefCounted {
	GDEXTENSION_CLASS(EditorSceneFormatImporter, RefCounted)

public:
	static const int IMPORT_SCENE = 1;
	static const int IMPORT_ANIMATION = 2;
	static const int IMPORT_FAIL_ON_MISSING_DEPENDENCIES = 4;
	static const int IMPORT_GENERATE_TANGENT_ARRAYS = 8;
	static const int IMPORT_USE_NAMED_SKIN_BINDS = 16;
	static const int IMPORT_DISCARD_MESHES_AND_MATERIALS = 32;
	static const int IMPORT_FORCE_DISABLE_MESH_COMPRESSION = 64;

	void add_import_option(const String &p_name, const Variant &p_value);
	void add_import_option_advanced(Variant::Type p_type, const String &p_name, const Variant &p_default_value, PropertyHint p_hint = (PropertyHint)0, const String &p_hint_string = String(), int32_t p_usage_flags = 6);
	virtual PackedStringArray _get_extensions() const;
	virtual Object *_import_scene(const String &p_path, uint32_t p_flags, const Dictionary &p_options);
	virtual void _get_import_options(const String &p_path);
	virtual Variant _get_option_visibility(const String &p_path, bool p_for_animation, const String &p_option) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_extensions), decltype(&T::_get_extensions)>) {
			BIND_VIRTUAL_METHOD(T, _get_extensions, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_import_scene), decltype(&T::_import_scene)>) {
			BIND_VIRTUAL_METHOD(T, _import_scene, 3749238728);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_import_options), decltype(&T::_get_import_options)>) {
			BIND_VIRTUAL_METHOD(T, _get_import_options, 83702148);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_option_visibility), decltype(&T::_get_option_visibility)>) {
			BIND_VIRTUAL_METHOD(T, _get_option_visibility, 298836892);
		}
	}

public:
};

} // namespace godot

