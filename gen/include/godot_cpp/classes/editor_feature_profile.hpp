/**************************************************************************/
/*  editor_feature_profile.hpp                                            */
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

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class StringName;

class EditorFeatureProfile : public RefCounted {
	GDEXTENSION_CLASS(EditorFeatureProfile, RefCounted)

public:
	enum Feature {
		FEATURE_3D = 0,
		FEATURE_SCRIPT = 1,
		FEATURE_ASSET_LIB = 2,
		FEATURE_SCENE_TREE = 3,
		FEATURE_NODE_DOCK = 4,
		FEATURE_FILESYSTEM_DOCK = 5,
		FEATURE_IMPORT_DOCK = 6,
		FEATURE_HISTORY_DOCK = 7,
		FEATURE_GAME = 8,
		FEATURE_SIGNALS_DOCK = 9,
		FEATURE_GROUPS_DOCK = 10,
		FEATURE_MAX = 11,
	};

	void set_disable_class(const StringName &p_class_name, bool p_disable);
	bool is_class_disabled(const StringName &p_class_name) const;
	void set_disable_class_editor(const StringName &p_class_name, bool p_disable);
	bool is_class_editor_disabled(const StringName &p_class_name) const;
	void set_disable_class_property(const StringName &p_class_name, const StringName &p_property, bool p_disable);
	bool is_class_property_disabled(const StringName &p_class_name, const StringName &p_property) const;
	void set_disable_feature(EditorFeatureProfile::Feature p_feature, bool p_disable);
	bool is_feature_disabled(EditorFeatureProfile::Feature p_feature) const;
	String get_feature_name(EditorFeatureProfile::Feature p_feature);
	Error save_to_file(const String &p_path);
	Error load_from_file(const String &p_path);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(EditorFeatureProfile::Feature);

