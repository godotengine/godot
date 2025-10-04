/**************************************************************************/
/*  property_info.h                                                       */
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

#pragma once

#include "core/extension/gdextension_interface.h"
#include "core/variant/variant.h"

template <typename T>
class TypedArray;

enum PropertyHint {
	PROPERTY_HINT_NONE, ///< no hint provided.
	PROPERTY_HINT_RANGE, ///< hint_text = "min,max[,step][,or_greater][,or_less][,prefer_slider][,hide_control][,radians_as_degrees][,degrees][,exp][,suffix:<keyword>] range.
	PROPERTY_HINT_ENUM, ///< hint_text= "val1,val2,val3,etc"
	PROPERTY_HINT_ENUM_SUGGESTION, ///< hint_text= "val1,val2,val3,etc"
	PROPERTY_HINT_EXP_EASING, /// exponential easing function (Math::ease) use "attenuation" hint string to revert (flip h), "positive_only" to exclude in-out and out-in. (ie: "attenuation,positive_only")
	PROPERTY_HINT_LINK,
	PROPERTY_HINT_FLAGS, ///< hint_text= "flag1,flag2,etc" (as bit flags)
	PROPERTY_HINT_LAYERS_2D_RENDER,
	PROPERTY_HINT_LAYERS_2D_PHYSICS,
	PROPERTY_HINT_LAYERS_2D_NAVIGATION,
	PROPERTY_HINT_LAYERS_3D_RENDER,
	PROPERTY_HINT_LAYERS_3D_PHYSICS,
	PROPERTY_HINT_LAYERS_3D_NAVIGATION,
	PROPERTY_HINT_FILE, ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,"
	PROPERTY_HINT_DIR, ///< a directory path must be passed
	PROPERTY_HINT_GLOBAL_FILE, ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,"
	PROPERTY_HINT_GLOBAL_DIR, ///< a directory path must be passed
	PROPERTY_HINT_RESOURCE_TYPE, ///< a comma-separated resource object type, e.g. "NoiseTexture,GradientTexture2D". Subclasses can be excluded with a "-" prefix if placed *after* the base class, e.g. "Texture2D,-MeshTexture".
	PROPERTY_HINT_MULTILINE_TEXT, ///< used for string properties that can contain multiple lines
	PROPERTY_HINT_EXPRESSION, ///< used for string properties that can contain multiple lines
	PROPERTY_HINT_PLACEHOLDER_TEXT, ///< used to set a placeholder text for string properties
	PROPERTY_HINT_COLOR_NO_ALPHA, ///< used for ignoring alpha component when editing a color
	PROPERTY_HINT_OBJECT_ID,
	PROPERTY_HINT_TYPE_STRING, ///< a type string, the hint is the base type to choose
	PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE, // Deprecated.
	PROPERTY_HINT_OBJECT_TOO_BIG, ///< object is too big to send
	PROPERTY_HINT_NODE_PATH_VALID_TYPES,
	PROPERTY_HINT_SAVE_FILE, ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,". This opens a save dialog
	PROPERTY_HINT_GLOBAL_SAVE_FILE, ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,". This opens a save dialog
	PROPERTY_HINT_INT_IS_OBJECTID, // Deprecated.
	PROPERTY_HINT_INT_IS_POINTER,
	PROPERTY_HINT_ARRAY_TYPE,
	PROPERTY_HINT_LOCALE_ID,
	PROPERTY_HINT_LOCALIZABLE_STRING,
	PROPERTY_HINT_NODE_TYPE, ///< a node object type
	PROPERTY_HINT_HIDE_QUATERNION_EDIT, /// Only Node3D::transform should hide the quaternion editor.
	PROPERTY_HINT_PASSWORD,
	PROPERTY_HINT_LAYERS_AVOIDANCE,
	PROPERTY_HINT_DICTIONARY_TYPE,
	PROPERTY_HINT_TOOL_BUTTON,
	PROPERTY_HINT_ONESHOT, ///< the property will be changed by self after setting, such as AudioStreamPlayer.playing, Particles.emitting.
	PROPERTY_HINT_NO_NODEPATH, /// < this property will not contain a NodePath, regardless of type (Array, Dictionary, List, etc.). Needed for SceneTreeDock.
	PROPERTY_HINT_GROUP_ENABLE, ///< used to make the property's group checkable. Only use for boolean types.
	PROPERTY_HINT_INPUT_NAME,
	PROPERTY_HINT_FILE_PATH,
	PROPERTY_HINT_MAX,
};

enum PropertyUsageFlags {
	PROPERTY_USAGE_NONE = 0,
	PROPERTY_USAGE_STORAGE = 1 << 1,
	PROPERTY_USAGE_EDITOR = 1 << 2,
	PROPERTY_USAGE_INTERNAL = 1 << 3,
	PROPERTY_USAGE_CHECKABLE = 1 << 4, // Used for editing global variables.
	PROPERTY_USAGE_CHECKED = 1 << 5, // Used for editing global variables.
	PROPERTY_USAGE_GROUP = 1 << 6, // Used for grouping props in the editor.
	PROPERTY_USAGE_CATEGORY = 1 << 7,
	PROPERTY_USAGE_SUBGROUP = 1 << 8,
	PROPERTY_USAGE_CLASS_IS_BITFIELD = 1 << 9,
	PROPERTY_USAGE_NO_INSTANCE_STATE = 1 << 10,
	PROPERTY_USAGE_RESTART_IF_CHANGED = 1 << 11,
	PROPERTY_USAGE_SCRIPT_VARIABLE = 1 << 12,
	PROPERTY_USAGE_STORE_IF_NULL = 1 << 13,
	PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED = 1 << 14,
	PROPERTY_USAGE_SCRIPT_DEFAULT_VALUE = 1 << 15, // Deprecated.
	PROPERTY_USAGE_CLASS_IS_ENUM = 1 << 16,
	PROPERTY_USAGE_NIL_IS_VARIANT = 1 << 17,
	PROPERTY_USAGE_ARRAY = 1 << 18, // Used in the inspector to group properties as elements of an array.
	PROPERTY_USAGE_ALWAYS_DUPLICATE = 1 << 19, // When duplicating a resource, always duplicate, even with subresource duplication disabled.
	PROPERTY_USAGE_NEVER_DUPLICATE = 1 << 20, // When duplicating a resource, never duplicate, even with subresource duplication enabled.
	PROPERTY_USAGE_HIGH_END_GFX = 1 << 21,
	PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT = 1 << 22,
	PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT = 1 << 23,
	PROPERTY_USAGE_KEYING_INCREMENTS = 1 << 24, // Used in inspector to increment property when keyed in animation player.
	PROPERTY_USAGE_DEFERRED_SET_RESOURCE = 1 << 25, // Deprecated.
	PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT = 1 << 26, // For Object properties, instantiate them when creating in editor.
	PROPERTY_USAGE_EDITOR_BASIC_SETTING = 1 << 27, //for project or editor settings, show when basic settings are selected.
	PROPERTY_USAGE_READ_ONLY = 1 << 28, // Mark a property as read-only in the inspector.
	PROPERTY_USAGE_SECRET = 1 << 29, // Export preset credentials that should be stored separately from the rest of the export config.

	PROPERTY_USAGE_DEFAULT = PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_EDITOR,
	PROPERTY_USAGE_NO_EDITOR = PROPERTY_USAGE_STORAGE,
};

struct PropertyInfo {
	Variant::Type type = Variant::NIL;
	String name;
	StringName class_name; // For classes
	PropertyHint hint = PROPERTY_HINT_NONE;
	String hint_string;
	uint32_t usage = PROPERTY_USAGE_DEFAULT;

	// If you are thinking about adding another member to this class, ask the maintainer (Juan) first.

	_FORCE_INLINE_ PropertyInfo added_usage(uint32_t p_fl) const {
		PropertyInfo pi = *this;
		pi.usage |= p_fl;
		return pi;
	}

	operator Dictionary() const;

	static PropertyInfo from_dict(const Dictionary &p_dict);

	PropertyInfo() {}

	PropertyInfo(const Variant::Type p_type, const String &p_name, const PropertyHint p_hint = PROPERTY_HINT_NONE, const String &p_hint_string = "", const uint32_t p_usage = PROPERTY_USAGE_DEFAULT, const StringName &p_class_name = StringName()) :
			type(p_type),
			name(p_name),
			hint(p_hint),
			hint_string(p_hint_string),
			usage(p_usage) {
		if (hint == PROPERTY_HINT_RESOURCE_TYPE) {
			class_name = hint_string;
		} else {
			class_name = p_class_name;
		}
	}

	PropertyInfo(const StringName &p_class_name) :
			type(Variant::OBJECT),
			class_name(p_class_name) {}

	explicit PropertyInfo(const GDExtensionPropertyInfo &pinfo) :
			type((Variant::Type)pinfo.type),
			name(*reinterpret_cast<StringName *>(pinfo.name)),
			class_name(*reinterpret_cast<StringName *>(pinfo.class_name)),
			hint((PropertyHint)pinfo.hint),
			hint_string(*reinterpret_cast<String *>(pinfo.hint_string)),
			usage(pinfo.usage) {}

	bool operator==(const PropertyInfo &p_info) const {
		return ((type == p_info.type) &&
				(name == p_info.name) &&
				(class_name == p_info.class_name) &&
				(hint == p_info.hint) &&
				(hint_string == p_info.hint_string) &&
				(usage == p_info.usage));
	}

	bool operator<(const PropertyInfo &p_info) const {
		return name < p_info.name;
	}
};

TypedArray<Dictionary> convert_property_list(const List<PropertyInfo> *p_list);
TypedArray<Dictionary> convert_property_list(const Vector<PropertyInfo> &p_vector);
