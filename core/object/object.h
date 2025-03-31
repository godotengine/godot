/**************************************************************************/
/*  object.h                                                              */
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
#include "core/object/message_queue.h"
#include "core/object/object_id.h"
#include "core/os/rw_lock.h"
#include "core/os/spin_lock.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"
#include "core/templates/rb_map.h"
#include "core/templates/safe_refcount.h"
#include "core/variant/callable_bind.h"
#include "core/variant/variant.h"

template <typename T>
class TypedArray;

template <typename T>
class Ref;

enum PropertyHint {
	// The property has no hint for the editor.
	PROPERTY_HINT_NONE,

	// Hints that an `int` or `float` property should be within a range specified via the hint string `"min,max"` or `"min,max,step"`.
	// The hint string can optionally include `"or_greater"` and/or `"or_less"`
	// to allow manual input going respectively above the max or below the min values.
	// Example: `"-360,360,1,or_greater,or_less"`
	// Additionally, other keywords can be included: `"exp"` for exponential range editing,
	// `"radians_as_degrees"` for editing radian angles in degrees (the range values are also in degrees),
	// `"degrees"` to hint at an angle, `"suffix:unit"` to display a `unit` suffix,
	// and `"hide_slider"` to hide the slider.
	PROPERTY_HINT_RANGE,

	// Hints that an `int` or `String` property is an enumerated value to pick in a list specified via a hint string.
	// The hint string is a comma separated list of names such as `"Hello,Something,Else"`.
	// Whitespaces are *not* removed from either end of a name.
	// For integer properties, the first name in the list has value 0, the next 1, and so on.
	// Explicit values can also be specified by appending `:integer` to the name, e.g. `"Zero,One,Three:3,Four,Six:6"`.
	PROPERTY_HINT_ENUM,

	// Hints that a `String` property can be an enumerated value to pick in a list
	// specified via a hint string such as `"Hello,Something,Else"`.
	// Unlike `PROPERTY_HINT_ENUM`, a property with this hint still accepts arbitrary values and can be empty.
	// The list of values serves to suggest possible values.
	PROPERTY_HINT_ENUM_SUGGESTION,

	// Hints that a `float` property should be edited via an exponential easing function.
	// The hint string can include `"attenuation"` to flip the curve horizontally and/or
	// `"positive_only"` to exclude in/out easing and limit values to be greater than or equal to zero.
	PROPERTY_HINT_EXP_EASING,

	// Hints that a vector property should allow its components to be linked.
	// For example, this allows `Vector2.x` and `Vector2.y` to be edited together.
	// This is typically used for scale properties to keep proportions.
	PROPERTY_HINT_LINK,

	// Hints that an `int` property is a bitmask with named bit flags.
	// The hint string is a comma separated list of names such as `"Bit0,Bit1,Bit2,Bit3"`.
	// Whitespaces are *not* removed from either end of a name. The first name in the list has value 1, the next 2, then 4, 8, 16 and so on.
	// Explicit values can also be specified by appending `:integer` to the name, e.g. `"A:4,B:8,C:16"`.
	// You can also combine several flags (`"A:4,B:8,AB:12,C:16"`).
	// NOTE: A flag value must be at least `1` and at most `2 ** 32 - 1`.
	// NOTE: Unlike `PROPERTY_HINT_ENUM`, the previous explicit value is not taken into account. For the hint `"A:16,B,C"`, A is 16, B is 2, C is 4.
	PROPERTY_HINT_FLAGS,

	// Hints that an `int` property is a bitmask using the optionally named 2D render layers.
	PROPERTY_HINT_LAYERS_2D_RENDER,

	// Hints that an `int` property is a bitmask using the optionally named 2D physics layers.
	PROPERTY_HINT_LAYERS_2D_PHYSICS,

	// Hints that an `int` property is a bitmask using the optionally named 2D navigation layers.
	PROPERTY_HINT_LAYERS_2D_NAVIGATION,

	// Hints that an `int` property is a bitmask using the optionally named 3D render layers.
	PROPERTY_HINT_LAYERS_3D_RENDER,

	// Hints that an `int` property is a bitmask using the optionally named 3D physics layers.
	PROPERTY_HINT_LAYERS_3D_PHYSICS,

	// Hints that an `int` property is a bitmask using the optionally named 3D navigation layers.
	PROPERTY_HINT_LAYERS_3D_NAVIGATION,

	// Hints that a `String` property is a path to a file.
	// Editing it will show a file dialog for picking the path.
	// The hint string can be a set of filters with wildcards like `"*.png,*.jpg"`. See also `PROPERTY_HINT_SAVE_FILE`.
	PROPERTY_HINT_FILE,

	// Hints that a `String` property is a path to a directory.
	// Editing it will show a file dialog for picking the path.
	PROPERTY_HINT_DIR,

	// Hints that a `String` property is an absolute path to a file outside the project folder.
	// Editing it will show a file dialog for picking the path.
	// The hint string can be a set of filters with wildcards, like `"*.png,*.jpg"`. See also `PROPERTY_HINT_GLOBAL_SAVE_FILE`.
	PROPERTY_HINT_GLOBAL_FILE,

	// Hints that a `String` property is an absolute path to a directory outside the project folder.
	// Editing it will show a file dialog for picking the path.
	PROPERTY_HINT_GLOBAL_DIR,

	// Hints that a property is an instance of a [Resource]-derived type,
	// optionally specified via the hint string (e.g. `"Texture2D"`).
	// Editing it will show a popup menu of valid resource types to instantiate.
	// The hint string can specify multiple classes separated by commas (e.g. `"NoiseTexture,GradientTexture2D"`);
	// all subclasses are included automatically. Specific subclasses can be excluded with a `"-"` prefix
	// if placed *after* the base class, e.g. `"Texture2D,-MeshTexture"`.
	// This is similar to `PROPERTY_HINT_NODE_PATH_VALID_TYPES`, but for resources.
	PROPERTY_HINT_RESOURCE_TYPE,

	// Hints that a `String` property is text with line breaks.
	// Editing it will show a text input field where line breaks can be typed.
	PROPERTY_HINT_MULTILINE_TEXT,

	// Hints that a `String` property is an [Expression].
	PROPERTY_HINT_EXPRESSION,

	// Hints that a `String` property should show a placeholder text on its input field, if empty.
	// The hint string is the placeholder text to use.
	PROPERTY_HINT_PLACEHOLDER_TEXT,

	// Hints that a `Color` property should be edited without affecting its transparency
	// (`Color.a` is not editable).
	PROPERTY_HINT_COLOR_NO_ALPHA,

	// Hints that the property's value is an object encoded as object ID,
	// with its type specified in the hint string. Used by the debugger.
	PROPERTY_HINT_OBJECT_ID,

	// If a property is `String`, hints that the property represents a particular type (class).
	// This allows to select a type from the create dialog. The property will store the selected type as a string.
	// If a property is `Array`, hints the editor how to show elements.
	// The `hint_string` must encode nested types using `":"` and `"/"`.
	PROPERTY_HINT_TYPE_STRING,

	// Deprecated. This hint is not used by the engine.
	PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE,

	// Hints that an object is too big to be sent via the debugger.
	PROPERTY_HINT_OBJECT_TOO_BIG,

	// Hints that the hint string specifies valid node types for property of type `NodePath`.
	// The hint string can specify multiple classes separated by commas (e.g. `"Node2D,Node3D"`);
	// all subclasses are included automatically.
	// Editing the property will show a dialog for picking a node from the scene.
	// This is similar to `PROPERTY_HINT_RESOURCE_TYPE`, but for `NodePath` properties.
	// Use `PROPERTY_HINT_NODE_TYPE` for `Node` properties instead.
	PROPERTY_HINT_NODE_PATH_VALID_TYPES,

	// Hints that a `String` property is a path to a file.
	// Editing it will show a file dialog for picking the path for the file to be saved at.
	// The dialog has access to the project's directory.
	// The hint string can be a set of filters with wildcards like `"*.png,*.jpg"`. See also `PROPERTY_HINT_FILE`.
	PROPERTY_HINT_SAVE_FILE,

	// Hints that a `String` property is a path to a file.
	// Editing it will show a file dialog for picking the path for the file to be saved at.
	// The dialog has access to the entire filesystem.
	// The hint string can be a set of filters with wildcards like `"*.png,*.jpg"`. See also `PROPERTY_HINT_GLOBAL_FILE`.
	PROPERTY_HINT_GLOBAL_SAVE_FILE,

	// Deprecated.
	PROPERTY_HINT_INT_IS_OBJECTID,

	// Hints that an `int` property is a pointer. Used by GDExtension.
	PROPERTY_HINT_INT_IS_POINTER,

	// Hints that a property is an [Array] with the stored type specified in the hint string.
	PROPERTY_HINT_ARRAY_TYPE,

	// Hints that a string property is a locale code.
	// Editing it will show a locale dialog for picking language and country.
	PROPERTY_HINT_LOCALE_ID,

	// Hints that a dictionary property is a string translation map.
	// Dictionary keys are locale codes; values are translated strings.
	PROPERTY_HINT_LOCALIZABLE_STRING,

	// Hints that a property is an instance of a `Node`-derived type, optionally specified via the hint string (e.g. `"Node2D"`).
	// The hint string can specify multiple classes separated by commas (e.g. `"Node2D,Node3D"`);
	// all subclasses are included automatically.
	// Editing the property will show a dialog for picking a node from the scene.
	// This is similar to `PROPERTY_HINT_RESOURCE_TYPE`, but for `Node` properties.
	// Use `PROPERTY_HINT_NODE_PATH_VALID_TYPES` for `NodePath` properties instead.
	PROPERTY_HINT_NODE_TYPE,

	// Hints that a `Quaternion` property should disable the temporary Euler editor.
	PROPERTY_HINT_HIDE_QUATERNION_EDIT,

	// Hints that a string property is a password, and every character is replaced
	// with the secret character in the inspector.
	PROPERTY_HINT_PASSWORD,

	// Hints that an integer property is a bitmask using the optionally named avoidance layers.
	PROPERTY_HINT_LAYERS_AVOIDANCE,

	// Hints that a property is a Dictionary with the stored types specified in the hint string.
	// The hint string contains the key and value types separated by a semicolon (e.g. `"int;String"`).
	PROPERTY_HINT_DICTIONARY_TYPE,

	// Hints that a Callable property should be displayed as a clickable button.
	// When the button is pressed, the callable is called.
	// The hint string specifies the button text and optionally an icon from the `"EditorIcons"` theme type.
	// ```text
	// "Click me!" - A button with the text "Click me!" and the default "Callable" icon.
	// "Click me!,ColorRect" - A button with the text "Click me!" and the "ColorRect" icon.
	// ```
	// NOTE: A Callable cannot be properly serialized and stored in a file,
	// so it is recommended to use `PROPERTY_USAGE_EDITOR` instead of `PROPERTY_USAGE_DEFAULT`.
	PROPERTY_HINT_TOOL_BUTTON,

	// Hints that a property will be changed on its own after setting,
	// such as `AudioStreamPlayer::playing` or `GPUParticles3D::emitting`.
	PROPERTY_HINT_ONESHOT,

	// This property will not contain a `NodePath`, regardless of type (`Array`, `Dictionary`, `List`, etc.).
	// Needed for SceneTreeDock.
	PROPERTY_HINT_NO_NODEPATH,

	// Represents the size of the PropertyHint enum.
	PROPERTY_HINT_MAX,
};

enum PropertyUsageFlags {
	// The property is not stored, and does not display in the editor.
	// NOTE: This is not the default value for properties exposed using `ADD_PROPERTY()`
	// (see `PROPERTY_USAGE_DEFAULT` instead).
	PROPERTY_USAGE_NONE = 0,

	// The property is serialized and saved in the scene file.
	PROPERTY_USAGE_STORAGE = 1 << 1,

	// The property is shown in the EditorInspector.
	PROPERTY_USAGE_EDITOR = 1 << 2,

	// The property is excluded from the class reference.
	PROPERTY_USAGE_INTERNAL = 1 << 3,

	// The property can be checked in the EditorInspector and is currently unchecked.
	// NOTE: Not to be confused with boolean properties' checkbox.
	// `PROPERTY_USAGE_CHECKABLE`'s checkbox is a separate control displayed at the left of the property name.
	PROPERTY_USAGE_CHECKABLE = 1 << 4,

	// The property can be checked in the EditorInspector and is currently checked.
	// NOTE: Not to be confused with boolean properties' checkbox.
	// `PROPERTY_USAGE_CHECKED`'s checkbox is a separate control displayed at the left of the property name.
	PROPERTY_USAGE_CHECKED = 1 << 5,

	// Used to group properties together in the editor.
	PROPERTY_USAGE_GROUP = 1 << 6,

	// Used to categorize properties together in the editor.
	PROPERTY_USAGE_CATEGORY = 1 << 7,

	// Used to group properties together in the editor in a subgroup (under a group).
	PROPERTY_USAGE_SUBGROUP = 1 << 8,

	// The property is a bitfield, i.e. it contains multiple flags represented as bits.
	PROPERTY_USAGE_CLASS_IS_BITFIELD = 1 << 9,

	// The property does not save its state in PackedScene.
	PROPERTY_USAGE_NO_INSTANCE_STATE = 1 << 10,

	// Editing the property prompts the user for restarting the editor.
	PROPERTY_USAGE_RESTART_IF_CHANGED = 1 << 11,

	// The property is a script variable. `PROPERTY_USAGE_SCRIPT_VARIABLE` can be used to distinguish between
	// exported script variables from built-in variables (which don't have this usage flag).
	// By default, `PROPERTY_USAGE_SCRIPT_VARIABLE` is *not* applied to variables
	// that are created by overriding `Object::_get_property_list()` in a script.
	PROPERTY_USAGE_SCRIPT_VARIABLE = 1 << 12,

	// The property value of type [Object] will be stored even if its value is `null`.
	PROPERTY_USAGE_STORE_IF_NULL = 1 << 13,

	// If this property is modified, all inspector fields will be refreshed.
	PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED = 1 << 14,

	// Deprecated. This flag is not used by the engine.
	PROPERTY_USAGE_SCRIPT_DEFAULT_VALUE = 1 << 15,

	// The property is a variable of enum type, i.e. it only takes named integer constants
	// from its associated enumeration.
	PROPERTY_USAGE_CLASS_IS_ENUM = 1 << 16,

	// If property has `nullptr` as default value, its type will be Variant.
	PROPERTY_USAGE_NIL_IS_VARIANT = 1 << 17,

	// The property is an array. This is used in the inspector to group properties as elements of an array.
	PROPERTY_USAGE_ARRAY = 1 << 18,

	// When duplicating a resource with `duplicate()`, and this flag is set on a property of that resource,
	// the property should always be duplicated, regardless of the `subresources` bool parameter.
	PROPERTY_USAGE_ALWAYS_DUPLICATE = 1 << 19,

	// When duplicating a resource with `duplicate()`, and this flag is set on a property of that resource,
	// the property should never be duplicated, regardless of the `subresources` bool parameter.
	PROPERTY_USAGE_NEVER_DUPLICATE = 1 << 20,

	// The property is only shown in the editor if modern renderers are supported
	// (the Compatibility rendering method is excluded).
	PROPERTY_USAGE_HIGH_END_GFX = 1 << 21,

	// The NodePath property will always be relative to the scene's root. Mostly useful for local resources.
	PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT = 1 << 22,

	// Use when a resource is created on the fly, i.e. the getter will always return a different instance.
	// ResourceSaver needs this information to properly save such resources.
	PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT = 1 << 23,

	// Inserting an animation key frame of this property will automatically increment the value,
	// allowing to easily keyframe multiple values in a row.
	PROPERTY_USAGE_KEYING_INCREMENTS = 1 << 24,

	// Deprecated.
	PROPERTY_USAGE_DEFERRED_SET_RESOURCE = 1 << 25,

	// When this property is a `Resource` and base object is a `Node`,
	// a resource instance will be automatically created whenever the node is created in the editor.
	PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT = 1 << 26,

	// The property is considered a basic setting and will appear even when advanced mode is disabled.
	// Used for project settings.
	PROPERTY_USAGE_EDITOR_BASIC_SETTING = 1 << 27,

	// The property is read-only in the [EditorInspector].
	PROPERTY_USAGE_READ_ONLY = 1 << 28,

	// An export preset property with this flag contains confidential information
	// and is stored separately from the rest of the export preset configuration.
	PROPERTY_USAGE_SECRET = 1 << 29,

	// Default usage (storage and editor).
	// NOTE: To hide a property in the inspector, use `PROPERTY_USAGE_NO_EDITOR` instead of `PROPERTY_USAGE_NONE`.
	PROPERTY_USAGE_DEFAULT = PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_EDITOR,

	// Default usage but without showing the property in the editor (storage).
	PROPERTY_USAGE_NO_EDITOR = PROPERTY_USAGE_STORAGE,
};

// Binds a signal with the name `m_signal` to the scripting API.
#define ADD_SIGNAL(m_signal) ::ClassDB::add_signal(get_class_static(), m_signal)

// Binds the PropertyInfo `m_property` to the scripting API.
// `m_setter` and `m_getter` are the setter and getter methods for the property.
// `m_setter` should take exactly 1 parameter (the value), and `m_getter` should take no parameters.
//
// Example:
// ```cpp
// ADD_PROPERTY(PropertyInfo(Variant::INT, "display_mode", PROPERTY_HINT_ENUM, "Thumbnails,List"), "set_display_mode", "get_display_mode");
// ```
#define ADD_PROPERTY(m_property, m_setter, m_getter) ::ClassDB::add_property(get_class_static(), m_property, _scs_create(m_setter), _scs_create(m_getter))

// Binds the indexed PropertyInfo with the name `m_property` to the scripting API.
// This is used for properties that call a setter that takes an index parameter when set, such as Light3D's various properties.
// `m_setter` should take exactly 2 parameters (index then value), and `m_getter` should exactly 1 parameter (the index).
//
// Example:
// ```cpp
// ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "offset_top", PROPERTY_HINT_RANGE, "-4096,4096,1,or_less,or_greater,suffix:px"), "set_offset", "get_offset", SIDE_TOP);
// ```
#define ADD_PROPERTYI(m_property, m_setter, m_getter, m_index) ::ClassDB::add_property(get_class_static(), m_property, _scs_create(m_setter), _scs_create(m_getter), m_index)

// Binds the PropertyInfo with the name `m_property` to the scripting API, with the default value in the documentation set to `m_default`.
// Compared to `ADD_PROPERTY()`, this will not call the method to determine its default value when using `--doctool`.
// This can be used to avoid platform-specific differences in the generated documentation.
//
// Example:
// ```cpp
// ADD_PROPERTY_DEFAULT("input_device", "Default");
// ```
#define ADD_PROPERTY_DEFAULT(m_property, m_default) ::ClassDB::set_property_default_value(get_class_static(), m_property, m_default)

// Begins an inspector property group with the name `m_name` and the prefix `m_prefix`.
// All properties added after this macro will be part of this group until another group is created below it,
// or until `ADD_GROUP("", "")` is called to end the group without creating another group below.
// `m_name` should be in Title Case, while `m_prefix` should be in snake_case and end with an underscore.
// `m_prefix` is purely cosmetic and can be an empty string. If the beginning of a property's name matches `m_prefix`,
// the prefix will be hidden in the inspector. Not all properties have to match the prefix to be added to the group.
//
// Example:
// ```cpp
// ADD_GROUP("Collision", "collision_");
// ```
#define ADD_GROUP(m_name, m_prefix) ::ClassDB::add_property_group(get_class_static(), m_name, m_prefix)

// Begins an inspector property group with the name `m_name`, the prefix `m_prefix` and the depth level `m_depth` (must be 1 or greater).
// All properties added after this macro will be part of this group until another group is created below it,
// or until `ADD_GROUP("", "")` is called to end the group without creating another group below.
// `m_name` should be in Title Case, while `m_prefix` should be in snake_case and end with an underscore.
// `m_prefix` is purely cosmetic and can be an empty string. If the beginning of a property's name matches `m_prefix`,
// the prefix will be hidden in the inspector. Not all properties have to match the prefix to be added to the group.
//
// NOTE: Most of the time, `ADD_SUBGROUP()` should be preferred over `ADD_GROUP_INDENT()`.
//
// Example:
// ```cpp
// ADD_GROUP_INDENT("Grow Direction", "grow_", 1);
// ```
#define ADD_GROUP_INDENT(m_name, m_prefix, m_depth) ::ClassDB::add_property_group(get_class_static(), m_name, m_prefix, m_depth)

// Begins an inspector property subgroup with the name `m_name` and the prefix `m_prefix`.
// All properties added after this macro will be part of this subgroup until another subgroup is created below it,
// or until `ADD_GROUP("", "")` is called to end the subgroup without creating another subgroup below.
// `m_name` should be in Title Case, while `m_prefix` should be in snake_case and end with an underscore.
// `m_prefix` is purely cosmetic and can be an empty string. If the beginning of a property's name matches `m_prefix`,
// the prefix will be hidden in the inspector. Not all properties have to match the prefix to be added to the group.
//
// Example:
// ```cpp
// ADD_SUBGROUP("Tangential Accel", "tangential_");
// ```
#define ADD_SUBGROUP(m_name, m_prefix) ::ClassDB::add_property_subgroup(get_class_static(), m_name, m_prefix)

// Begins an inspector property subgroup with the name `m_name`, the prefix `m_prefix` and the depth level `m_depth` (must be 1 or greater).
// All properties added after this macro will be part of this subgroup until another subgroup is created below it,
// or until `ADD_GROUP("", "")` is called to end the subgroup without creating another group below.
// `m_name` should be in Title Case, while `m_prefix` should be in snake_case and end with an underscore.
// `m_prefix` is purely cosmetic and can be an empty string. If the beginning of a property's name matches `m_prefix`,
// the prefix will be hidden in the inspector. Not all properties have to match the prefix to be added to the group.
//
// Example:
// ```cpp
// ADD_SUBGROUP_INDENT("Grow Direction", "grow_", 1);
// ```
#define ADD_SUBGROUP_INDENT(m_name, m_prefix, m_depth) ::ClassDB::add_property_subgroup(get_class_static(), m_name, m_prefix, m_depth)

// Defines a property to be linked with another (both properties must have been bound using `ADD_PROPERTY` beforehand).
// When `m_property` property is modified through the inspector, the old value of `m_linked_property` is stored before
// the change is made, so that it can be reverted to using the editor's undo/redo system.
// For dynamic properties, the class must also define the `_get_linked_undo_properties()` method for this to work.
// Behavior is one-way by default, but can be made two-way by defining the link in both directions.
//
// Example:
// ```cpp
// ADD_LINKED_PROPERTY("radius", "height");
// ADD_LINKED_PROPERTY("height", "radius");
// ```
#define ADD_LINKED_PROPERTY(m_property, m_linked_property) ::ClassDB::add_linked_property(get_class_static(), m_property, m_linked_property)

// Binds an array to the scripting API, with a dedicated array editor in the inspector.
// `m_label` should be in Title Case, while `m_prefix` should be in snake_case and end with a slash.
// `m_count_property`, `m_count_property_setter` and `m_count_property_getter` are used to set/get
// the array length and are defined like in `ADD_PROPERTY()`.
// To define custom property usage flags, use `ADD_ARRAY_COUNT_WITH_USAGE_FLAGS()` instead.
//
// Example:
// ```cpp
// ADD_ARRAY_COUNT("Items", "item_count", "set_item_count", "get_item_count", "item_");
// ```
#define ADD_ARRAY_COUNT(m_label, m_count_property, m_count_property_setter, m_count_property_getter, m_prefix) ClassDB::add_property_array_count(get_class_static(), m_label, m_count_property, _scs_create(m_count_property_setter), _scs_create(m_count_property_getter), m_prefix)

// Binds an array to the scripting API, with a dedicated array editor in the inspector.
// `m_label` should be in Title Case, while `m_prefix` should be in snake_case and end with a slash.
// `m_count_property`, `m_count_property_setter` and `m_count_property_getter` are used to set/get
// the array length and are defined like in `ADD_PROPERTY()`.
// Property usage flags are defined using `m_property_usage_flags`.
//
// Example:
// ```cpp
// ADD_ARRAY_COUNT_WITH_USAGE_FLAGS("Items", "item_count", "set_item_count", "get_item_count", "item_", PROPERTY_USAGE_NO_EDITOR);
// ```
#define ADD_ARRAY_COUNT_WITH_USAGE_FLAGS(m_label, m_count_property, m_count_property_setter, m_count_property_getter, m_prefix, m_property_usage_flags) ClassDB::add_property_array_count(get_class_static(), m_label, m_count_property, _scs_create(m_count_property_setter), _scs_create(m_count_property_getter), m_prefix, m_property_usage_flags)

// Binds an array to the scripting API, with a dedicated array editor in the inspector.
// `m_array_path` should be in snake_case, while `m_prefix` should be in snake_case and end with an underscore.
// Array properties created this way will have the naming `m_prefix/N`, where `N` is an integer starting from 0
// representing the array index.
//
// For this to be effective, the class must have a `_set()` method implemented that reads the property name
// and acts accordingly if the property name is `m_prefix/N`. Alternatively, you can use PropertyListHelper
// to handle this for you (see the TileMap class for an example).
//
// Example:
// ```cpp
// ADD_ARRAY("physics_layers", "physics_layer_");
// ```
#define ADD_ARRAY(m_array_path, m_prefix) ClassDB::add_property_array(get_class_static(), m_array_path, m_prefix)

// Helper macro to use with `PROPERTY_HINT_ARRAY_TYPE` for arrays of specific resources.
//
// Example:
// ```cpp
// PropertyInfo(Variant::ARRAY, "fallbacks", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("Font"))
// ```
#define MAKE_RESOURCE_TYPE_HINT(m_type) vformat("%s/%s:%s", Variant::OBJECT, PROPERTY_HINT_RESOURCE_TYPE, m_type)

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

enum MethodFlags {
	METHOD_FLAG_NORMAL = 1,
	METHOD_FLAG_EDITOR = 2,
	METHOD_FLAG_CONST = 4,
	METHOD_FLAG_VIRTUAL = 8,
	METHOD_FLAG_VARARG = 16,
	METHOD_FLAG_STATIC = 32,
	METHOD_FLAG_OBJECT_CORE = 64,
	METHOD_FLAG_VIRTUAL_REQUIRED = 128,
	METHOD_FLAGS_DEFAULT = METHOD_FLAG_NORMAL,
};

struct MethodInfo {
	String name;
	PropertyInfo return_val;
	uint32_t flags = METHOD_FLAGS_DEFAULT;
	int id = 0;
	Vector<PropertyInfo> arguments;
	Vector<Variant> default_arguments;
	int return_val_metadata = 0;
	Vector<int> arguments_metadata;

	int get_argument_meta(int p_arg) const {
		ERR_FAIL_COND_V(p_arg < -1 || p_arg > arguments.size(), 0);
		if (p_arg == -1) {
			return return_val_metadata;
		}
		return arguments_metadata.size() > p_arg ? arguments_metadata[p_arg] : 0;
	}

	inline bool operator==(const MethodInfo &p_method) const { return id == p_method.id && name == p_method.name; }
	inline bool operator<(const MethodInfo &p_method) const { return id == p_method.id ? (name < p_method.name) : (id < p_method.id); }

	operator Dictionary() const;

	static MethodInfo from_dict(const Dictionary &p_dict);

	uint32_t get_compatibility_hash() const;

	MethodInfo() {}

	explicit MethodInfo(const GDExtensionMethodInfo &pinfo) :
			name(*reinterpret_cast<StringName *>(pinfo.name)),
			return_val(PropertyInfo(pinfo.return_value)),
			flags(pinfo.flags),
			id(pinfo.id) {
		for (uint32_t i = 0; i < pinfo.argument_count; i++) {
			arguments.push_back(PropertyInfo(pinfo.arguments[i]));
		}
		const Variant *def_values = (const Variant *)pinfo.default_arguments;
		for (uint32_t j = 0; j < pinfo.default_argument_count; j++) {
			default_arguments.push_back(def_values[j]);
		}
	}

	MethodInfo(const String &p_name) { name = p_name; }

	template <typename... VarArgs>
	MethodInfo(const String &p_name, VarArgs... p_params) {
		name = p_name;
		arguments = Vector<PropertyInfo>{ p_params... };
	}

	MethodInfo(Variant::Type ret) { return_val.type = ret; }
	MethodInfo(Variant::Type ret, const String &p_name) {
		return_val.type = ret;
		name = p_name;
	}

	template <typename... VarArgs>
	MethodInfo(Variant::Type ret, const String &p_name, VarArgs... p_params) {
		name = p_name;
		return_val.type = ret;
		arguments = Vector<PropertyInfo>{ p_params... };
	}

	MethodInfo(const PropertyInfo &p_ret, const String &p_name) {
		return_val = p_ret;
		name = p_name;
	}

	template <typename... VarArgs>
	MethodInfo(const PropertyInfo &p_ret, const String &p_name, VarArgs... p_params) {
		return_val = p_ret;
		name = p_name;
		arguments = Vector<PropertyInfo>{ p_params... };
	}
};

// API used to extend in GDExtension and other C compatible compiled languages.
class MethodBind;
class GDExtension;

struct ObjectGDExtension {
	GDExtension *library = nullptr;
	ObjectGDExtension *parent = nullptr;
	List<ObjectGDExtension *> children;
	StringName parent_class_name;
	StringName class_name;
	bool editor_class = false;
	bool reloadable = false;
	bool is_virtual = false;
	bool is_abstract = false;
	bool is_exposed = true;
#ifdef TOOLS_ENABLED
	bool is_runtime = false;
	bool is_placeholder = false;
#endif
	GDExtensionClassSet set;
	GDExtensionClassGet get;
	GDExtensionClassGetPropertyList get_property_list;
	GDExtensionClassFreePropertyList2 free_property_list2;
	GDExtensionClassPropertyCanRevert property_can_revert;
	GDExtensionClassPropertyGetRevert property_get_revert;
	GDExtensionClassValidateProperty validate_property;
#ifndef DISABLE_DEPRECATED
	GDExtensionClassNotification notification;
	GDExtensionClassFreePropertyList free_property_list;
#endif // DISABLE_DEPRECATED
	GDExtensionClassNotification2 notification2;
	GDExtensionClassToString to_string;
	GDExtensionClassReference reference;
	GDExtensionClassReference unreference;
	GDExtensionClassGetRID get_rid;

	_FORCE_INLINE_ bool is_class(const String &p_class) const {
		const ObjectGDExtension *e = this;
		while (e) {
			if (p_class == e->class_name.operator String()) {
				return true;
			}
			e = e->parent;
		}
		return false;
	}
	void *class_userdata = nullptr;

#ifndef DISABLE_DEPRECATED
	GDExtensionClassCreateInstance create_instance;
#endif // DISABLE_DEPRECATED
	GDExtensionClassCreateInstance2 create_instance2;
	GDExtensionClassFreeInstance free_instance;
#ifndef DISABLE_DEPRECATED
	GDExtensionClassGetVirtual get_virtual;
	GDExtensionClassGetVirtualCallData get_virtual_call_data;
#endif // DISABLE_DEPRECATED
	GDExtensionClassGetVirtual2 get_virtual2;
	GDExtensionClassGetVirtualCallData2 get_virtual_call_data2;
	GDExtensionClassCallVirtualWithData call_virtual_with_data;
	GDExtensionClassRecreateInstance recreate_instance;

#ifdef TOOLS_ENABLED
	void *tracking_userdata = nullptr;
	void (*track_instance)(void *p_userdata, void *p_instance) = nullptr;
	void (*untrack_instance)(void *p_userdata, void *p_instance) = nullptr;
#endif
};

#define GDVIRTUAL_CALL(m_name, ...) _gdvirtual_##m_name##_call(__VA_ARGS__)
#define GDVIRTUAL_CALL_PTR(m_obj, m_name, ...) m_obj->_gdvirtual_##m_name##_call(__VA_ARGS__)

#ifdef DEBUG_METHODS_ENABLED
#define GDVIRTUAL_BIND(m_name, ...) ::ClassDB::add_virtual_method(get_class_static(), _gdvirtual_##m_name##_get_method_info(), true, sarray(__VA_ARGS__));
#else
#define GDVIRTUAL_BIND(m_name, ...)
#endif
#define GDVIRTUAL_BIND_COMPAT(m_alias, ...) ::ClassDB::add_virtual_compatibility_method(get_class_static(), _gdvirtual_##m_alias##_get_method_info(), true, sarray(__VA_ARGS__));
#define GDVIRTUAL_IS_OVERRIDDEN(m_name) _gdvirtual_##m_name##_overridden()
#define GDVIRTUAL_IS_OVERRIDDEN_PTR(m_obj, m_name) m_obj->_gdvirtual_##m_name##_overridden()

/*
 * The following is an incomprehensible blob of hacks and workarounds to
 * compensate for many of the fallacies in C++. As a plus, this macro pretty
 * much alone defines the object model.
 */

// Registers a class to be exposed to the scripting API.
// This must be used in the header file within the class definition.
//
// Example:
// ```cpp
// class RenderData : public Object {
//     GDCLASS(RenderData, Object);
//     // ...
// ```
#define GDCLASS(m_class, m_inherits)                                                                                                        \
private:                                                                                                                                    \
	void operator=(const m_class &p_rval) {}                                                                                                \
	friend class ::ClassDB;                                                                                                                 \
                                                                                                                                            \
public:                                                                                                                                     \
	typedef m_class self_type;                                                                                                              \
	static constexpr bool _class_is_enabled = !bool(GD_IS_DEFINED(ClassDB_Disable_##m_class)) && m_inherits::_class_is_enabled;             \
	virtual String get_class() const override {                                                                                             \
		if (_get_extension()) {                                                                                                             \
			return _get_extension()->class_name.operator String();                                                                          \
		}                                                                                                                                   \
		return String(#m_class);                                                                                                            \
	}                                                                                                                                       \
	virtual const StringName *_get_class_namev() const override {                                                                           \
		static StringName _class_name_static;                                                                                               \
		if (unlikely(!_class_name_static)) {                                                                                                \
			StringName::assign_static_unique_class_name(&_class_name_static, #m_class);                                                     \
		}                                                                                                                                   \
		return &_class_name_static;                                                                                                         \
	}                                                                                                                                       \
	static _FORCE_INLINE_ void *get_class_ptr_static() {                                                                                    \
		static int ptr;                                                                                                                     \
		return &ptr;                                                                                                                        \
	}                                                                                                                                       \
	static _FORCE_INLINE_ String get_class_static() {                                                                                       \
		return String(#m_class);                                                                                                            \
	}                                                                                                                                       \
	static _FORCE_INLINE_ String get_parent_class_static() {                                                                                \
		return m_inherits::get_class_static();                                                                                              \
	}                                                                                                                                       \
	static void get_inheritance_list_static(List<String> *p_inheritance_list) {                                                             \
		m_inherits::get_inheritance_list_static(p_inheritance_list);                                                                        \
		p_inheritance_list->push_back(String(#m_class));                                                                                    \
	}                                                                                                                                       \
	virtual bool is_class(const String &p_class) const override {                                                                           \
		if (_get_extension() && _get_extension()->is_class(p_class)) {                                                                      \
			return true;                                                                                                                    \
		}                                                                                                                                   \
		return (p_class == (#m_class)) ? true : m_inherits::is_class(p_class);                                                              \
	}                                                                                                                                       \
	virtual bool is_class_ptr(void *p_ptr) const override {                                                                                 \
		return (p_ptr == get_class_ptr_static()) ? true : m_inherits::is_class_ptr(p_ptr);                                                  \
	}                                                                                                                                       \
                                                                                                                                            \
	static void get_valid_parents_static(List<String> *p_parents) {                                                                         \
		if (m_class::_get_valid_parents_static != m_inherits::_get_valid_parents_static) {                                                  \
			m_class::_get_valid_parents_static(p_parents);                                                                                  \
		}                                                                                                                                   \
                                                                                                                                            \
		m_inherits::get_valid_parents_static(p_parents);                                                                                    \
	}                                                                                                                                       \
                                                                                                                                            \
protected:                                                                                                                                  \
	virtual bool _derives_from(const std::type_info &p_type_info) const override {                                                          \
		return typeid(m_class) == p_type_info || m_inherits::_derives_from(p_type_info);                                                    \
	}                                                                                                                                       \
	_FORCE_INLINE_ static void (*_get_bind_methods())() {                                                                                   \
		return &m_class::_bind_methods;                                                                                                     \
	}                                                                                                                                       \
	_FORCE_INLINE_ static void (*_get_bind_compatibility_methods())() {                                                                     \
		return &m_class::_bind_compatibility_methods;                                                                                       \
	}                                                                                                                                       \
                                                                                                                                            \
public:                                                                                                                                     \
	static void initialize_class() {                                                                                                        \
		static bool initialized = false;                                                                                                    \
		if (initialized) {                                                                                                                  \
			return;                                                                                                                         \
		}                                                                                                                                   \
		m_inherits::initialize_class();                                                                                                     \
		::ClassDB::_add_class<m_class>();                                                                                                   \
		if (m_class::_get_bind_methods() != m_inherits::_get_bind_methods()) {                                                              \
			_bind_methods();                                                                                                                \
		}                                                                                                                                   \
		if (m_class::_get_bind_compatibility_methods() != m_inherits::_get_bind_compatibility_methods()) {                                  \
			_bind_compatibility_methods();                                                                                                  \
		}                                                                                                                                   \
		initialized = true;                                                                                                                 \
	}                                                                                                                                       \
                                                                                                                                            \
protected:                                                                                                                                  \
	virtual void _initialize_classv() override {                                                                                            \
		initialize_class();                                                                                                                 \
	}                                                                                                                                       \
	_FORCE_INLINE_ bool (Object::*_get_get() const)(const StringName &p_name, Variant &) const {                                            \
		return (bool(Object::*)(const StringName &, Variant &) const) & m_class::_get;                                                      \
	}                                                                                                                                       \
	virtual bool _getv(const StringName &p_name, Variant &r_ret) const override {                                                           \
		if (m_class::_get_get() != m_inherits::_get_get()) {                                                                                \
			if (_get(p_name, r_ret)) {                                                                                                      \
				return true;                                                                                                                \
			}                                                                                                                               \
		}                                                                                                                                   \
		return m_inherits::_getv(p_name, r_ret);                                                                                            \
	}                                                                                                                                       \
	_FORCE_INLINE_ bool (Object::*_get_set() const)(const StringName &p_name, const Variant &p_property) {                                  \
		return (bool(Object::*)(const StringName &, const Variant &)) & m_class::_set;                                                      \
	}                                                                                                                                       \
	virtual bool _setv(const StringName &p_name, const Variant &p_property) override {                                                      \
		if (m_inherits::_setv(p_name, p_property)) {                                                                                        \
			return true;                                                                                                                    \
		}                                                                                                                                   \
		if (m_class::_get_set() != m_inherits::_get_set()) {                                                                                \
			return _set(p_name, p_property);                                                                                                \
		}                                                                                                                                   \
		return false;                                                                                                                       \
	}                                                                                                                                       \
	_FORCE_INLINE_ void (Object::*_get_get_property_list() const)(List<PropertyInfo> * p_list) const {                                      \
		return (void(Object::*)(List<PropertyInfo> *) const) & m_class::_get_property_list;                                                 \
	}                                                                                                                                       \
	virtual void _get_property_listv(List<PropertyInfo> *p_list, bool p_reversed) const override {                                          \
		if (!p_reversed) {                                                                                                                  \
			m_inherits::_get_property_listv(p_list, p_reversed);                                                                            \
		}                                                                                                                                   \
		p_list->push_back(PropertyInfo(Variant::NIL, get_class_static(), PROPERTY_HINT_NONE, get_class_static(), PROPERTY_USAGE_CATEGORY)); \
		::ClassDB::get_property_list(#m_class, p_list, true, this);                                                                         \
		if (m_class::_get_get_property_list() != m_inherits::_get_get_property_list()) {                                                    \
			_get_property_list(p_list);                                                                                                     \
		}                                                                                                                                   \
		if (p_reversed) {                                                                                                                   \
			m_inherits::_get_property_listv(p_list, p_reversed);                                                                            \
		}                                                                                                                                   \
	}                                                                                                                                       \
	_FORCE_INLINE_ void (Object::*_get_validate_property() const)(PropertyInfo & p_property) const {                                        \
		return (void(Object::*)(PropertyInfo &) const) & m_class::_validate_property;                                                       \
	}                                                                                                                                       \
	virtual void _validate_propertyv(PropertyInfo &p_property) const override {                                                             \
		m_inherits::_validate_propertyv(p_property);                                                                                        \
		if (m_class::_get_validate_property() != m_inherits::_get_validate_property()) {                                                    \
			_validate_property(p_property);                                                                                                 \
		}                                                                                                                                   \
	}                                                                                                                                       \
	_FORCE_INLINE_ bool (Object::*_get_property_can_revert() const)(const StringName &p_name) const {                                       \
		return (bool(Object::*)(const StringName &) const) & m_class::_property_can_revert;                                                 \
	}                                                                                                                                       \
	virtual bool _property_can_revertv(const StringName &p_name) const override {                                                           \
		if (m_class::_get_property_can_revert() != m_inherits::_get_property_can_revert()) {                                                \
			if (_property_can_revert(p_name)) {                                                                                             \
				return true;                                                                                                                \
			}                                                                                                                               \
		}                                                                                                                                   \
		return m_inherits::_property_can_revertv(p_name);                                                                                   \
	}                                                                                                                                       \
	_FORCE_INLINE_ bool (Object::*_get_property_get_revert() const)(const StringName &p_name, Variant &) const {                            \
		return (bool(Object::*)(const StringName &, Variant &) const) & m_class::_property_get_revert;                                      \
	}                                                                                                                                       \
	virtual bool _property_get_revertv(const StringName &p_name, Variant &r_ret) const override {                                           \
		if (m_class::_get_property_get_revert() != m_inherits::_get_property_get_revert()) {                                                \
			if (_property_get_revert(p_name, r_ret)) {                                                                                      \
				return true;                                                                                                                \
			}                                                                                                                               \
		}                                                                                                                                   \
		return m_inherits::_property_get_revertv(p_name, r_ret);                                                                            \
	}                                                                                                                                       \
	_FORCE_INLINE_ void (Object::*_get_notification() const)(int) {                                                                         \
		return (void(Object::*)(int)) & m_class::_notification;                                                                             \
	}                                                                                                                                       \
	virtual void _notificationv(int p_notification, bool p_reversed) override {                                                             \
		if (!p_reversed) {                                                                                                                  \
			m_inherits::_notificationv(p_notification, p_reversed);                                                                         \
		}                                                                                                                                   \
		if (m_class::_get_notification() != m_inherits::_get_notification()) {                                                              \
			_notification(p_notification);                                                                                                  \
		}                                                                                                                                   \
		if (p_reversed) {                                                                                                                   \
			m_inherits::_notificationv(p_notification, p_reversed);                                                                         \
		}                                                                                                                                   \
	}                                                                                                                                       \
                                                                                                                                            \
private:

#define OBJ_SAVE_TYPE(m_class)                       \
public:                                              \
	virtual String get_save_class() const override { \
		return #m_class;                             \
	}                                                \
                                                     \
private:

class ScriptInstance;

class Object {
public:
	typedef Object self_type;

	enum ConnectFlags {
		CONNECT_DEFERRED = 1,
		CONNECT_PERSIST = 2, // hint for scene to save this connection
		CONNECT_ONE_SHOT = 4,
		CONNECT_REFERENCE_COUNTED = 8,
		CONNECT_INHERITED = 16, // Used in editor builds.
	};

	struct Connection {
		::Signal signal;
		Callable callable;

		uint32_t flags = 0;
		bool operator<(const Connection &p_conn) const;

		operator Variant() const;

		Connection() {}
		Connection(const Variant &p_variant);
	};

private:
#ifdef DEBUG_ENABLED
	friend struct _ObjectDebugLock;
#endif
	friend bool predelete_handler(Object *);
	friend void postinitialize_handler(Object *);

	ObjectGDExtension *_extension = nullptr;
	GDExtensionClassInstancePtr _extension_instance = nullptr;

	struct SignalData {
		struct Slot {
			int reference_count = 0;
			Connection conn;
			List<Connection>::Element *cE = nullptr;
		};

		MethodInfo user;
		HashMap<Callable, Slot, HashableHasher<Callable>> slot_map;
		bool removable = false;
	};

	HashMap<StringName, SignalData> signal_map;
	List<Connection> connections;
#ifdef DEBUG_ENABLED
	SafeRefCount _lock_index;
#endif
	bool _block_signals = false;
	int _predelete_ok = 0;
	ObjectID _instance_id;
	bool _predelete();
	void _initialize();
	void _postinitialize();
	bool _can_translate = true;
	bool _emitting = false;
#ifdef TOOLS_ENABLED
	bool _edited = false;
	uint32_t _edited_version = 0;
	HashSet<String> editor_section_folding;
#endif
	ScriptInstance *script_instance = nullptr;
	Variant script; // Reference does not exist yet, store it in a Variant.
	HashMap<StringName, Variant> metadata;
	HashMap<StringName, Variant *> metadata_properties;
	mutable const StringName *_class_name_ptr = nullptr;

	void _add_user_signal(const String &p_name, const Array &p_args = Array());
	bool _has_user_signal(const StringName &p_name) const;
	void _remove_user_signal(const StringName &p_name);
	Error _emit_signal(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	TypedArray<Dictionary> _get_signal_list() const;
	TypedArray<Dictionary> _get_signal_connection_list(const StringName &p_signal) const;
	TypedArray<Dictionary> _get_incoming_connections() const;
	void _set_bind(const StringName &p_set, const Variant &p_value);
	Variant _get_bind(const StringName &p_name) const;
	void _set_indexed_bind(const NodePath &p_name, const Variant &p_value);
	Variant _get_indexed_bind(const NodePath &p_name) const;
	int _get_method_argument_count_bind(const StringName &p_name) const;

	_FORCE_INLINE_ void _construct_object(bool p_reference);

	friend class RefCounted;
	bool type_is_reference = false;

	BinaryMutex _instance_binding_mutex;
	struct InstanceBinding {
		void *binding = nullptr;
		void *token = nullptr;
		GDExtensionInstanceBindingFreeCallback free_callback = nullptr;
		GDExtensionInstanceBindingReferenceCallback reference_callback = nullptr;
	};
	InstanceBinding *_instance_bindings = nullptr;
	uint32_t _instance_binding_count = 0;

	Object(bool p_reference);

protected:
	StringName _translation_domain;

	_FORCE_INLINE_ bool _instance_binding_reference(bool p_reference) {
		bool can_die = true;
		if (_instance_bindings) {
			MutexLock instance_binding_lock(_instance_binding_mutex);
			for (uint32_t i = 0; i < _instance_binding_count; i++) {
				if (_instance_bindings[i].reference_callback) {
					if (!_instance_bindings[i].reference_callback(_instance_bindings[i].token, _instance_bindings[i].binding, p_reference)) {
						can_die = false;
					}
				}
			}
		}
		return can_die;
	}

	friend class GDExtensionMethodBind;
	_ALWAYS_INLINE_ const ObjectGDExtension *_get_extension() const { return _extension; }
	_ALWAYS_INLINE_ GDExtensionClassInstancePtr _get_extension_instance() const { return _extension_instance; }
	virtual void _initialize_classv() { initialize_class(); }
	virtual bool _setv(const StringName &p_name, const Variant &p_property) { return false; }
	virtual bool _getv(const StringName &p_name, Variant &r_property) const { return false; }
	virtual void _get_property_listv(List<PropertyInfo> *p_list, bool p_reversed) const {}
	virtual void _validate_propertyv(PropertyInfo &p_property) const {}
	virtual bool _property_can_revertv(const StringName &p_name) const { return false; }
	virtual bool _property_get_revertv(const StringName &p_name, Variant &r_property) const { return false; }
	virtual void _notificationv(int p_notification, bool p_reversed) {}

	static void _bind_methods();
	static void _bind_compatibility_methods() {}
	bool _set(const StringName &p_name, const Variant &p_property) { return false; }
	bool _get(const StringName &p_name, Variant &r_property) const { return false; }
	void _get_property_list(List<PropertyInfo> *p_list) const {}
	void _validate_property(PropertyInfo &p_property) const {}
	bool _property_can_revert(const StringName &p_name) const { return false; }
	bool _property_get_revert(const StringName &p_name, Variant &r_property) const { return false; }
	void _notification(int p_notification) {}

	_FORCE_INLINE_ static void (*_get_bind_methods())() {
		return &Object::_bind_methods;
	}
	_FORCE_INLINE_ static void (*_get_bind_compatibility_methods())() {
		return &Object::_bind_compatibility_methods;
	}
	_FORCE_INLINE_ bool (Object::*_get_get() const)(const StringName &p_name, Variant &r_ret) const {
		return &Object::_get;
	}
	_FORCE_INLINE_ bool (Object::*_get_set() const)(const StringName &p_name, const Variant &p_property) {
		return &Object::_set;
	}
	_FORCE_INLINE_ void (Object::*_get_get_property_list() const)(List<PropertyInfo> *p_list) const {
		return &Object::_get_property_list;
	}
	_FORCE_INLINE_ void (Object::*_get_validate_property() const)(PropertyInfo &p_property) const {
		return &Object::_validate_property;
	}
	_FORCE_INLINE_ bool (Object::*_get_property_can_revert() const)(const StringName &p_name) const {
		return &Object::_property_can_revert;
	}
	_FORCE_INLINE_ bool (Object::*_get_property_get_revert() const)(const StringName &p_name, Variant &) const {
		return &Object::_property_get_revert;
	}
	_FORCE_INLINE_ void (Object::*_get_notification() const)(int) {
		return &Object::_notification;
	}
	static void get_valid_parents_static(List<String> *p_parents);
	static void _get_valid_parents_static(List<String> *p_parents);

	Variant _call_bind(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Variant _call_deferred_bind(const Variant **p_args, int p_argcount, Callable::CallError &r_error);

	virtual const StringName *_get_class_namev() const {
		static StringName _class_name_static;
		if (unlikely(!_class_name_static)) {
			StringName::assign_static_unique_class_name(&_class_name_static, "Object");
		}
		return &_class_name_static;
	}

	TypedArray<StringName> _get_meta_list_bind() const;
	TypedArray<Dictionary> _get_property_list_bind() const;
	TypedArray<Dictionary> _get_method_list_bind() const;

	void _clear_internal_resource_paths(const Variant &p_var);

	friend class ClassDB;
	friend class PlaceholderExtensionInstance;

	bool _disconnect(const StringName &p_signal, const Callable &p_callable, bool p_force = false);

#ifdef TOOLS_ENABLED
	struct VirtualMethodTracker {
		void **method;
		bool *initialized;
		VirtualMethodTracker *next;
	};

	mutable VirtualMethodTracker *virtual_method_list = nullptr;
#endif

	virtual bool _derives_from(const std::type_info &p_type_info) const {
		// This could just be false because nobody would reasonably ask if an Object subclass derives from Object,
		// but it would be wrong if somebody actually does ask. It's not too slow to check anyway.
		return typeid(Object) == p_type_info;
	}

public: // Should be protected, but bug in clang++.
	static void initialize_class();
	_FORCE_INLINE_ static void register_custom_data_to_otdb() {}

public:
	static constexpr bool _class_is_enabled = true;

	void notify_property_list_changed();

	static void *get_class_ptr_static() {
		static int ptr;
		return &ptr;
	}

	void detach_from_objectdb();
	_FORCE_INLINE_ ObjectID get_instance_id() const { return _instance_id; }

	template <typename T>
	static T *cast_to(Object *p_object) {
		// This is like dynamic_cast, but faster.
		// The reason is that we can assume no virtual and multiple inheritance.
		static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
		if constexpr (std::is_same_v<std::decay_t<T>, typename T::self_type>) {
			return p_object && p_object->_derives_from(typeid(T)) ? static_cast<T *>(p_object) : nullptr;
		} else {
			// T does not use GDCLASS, must fall back to dynamic_cast.
			return p_object ? dynamic_cast<T *>(p_object) : nullptr;
		}
	}

	template <typename T>
	static const T *cast_to(const Object *p_object) {
		static_assert(std::is_base_of_v<Object, T>, "T must be derived from Object");
		if constexpr (std::is_same_v<std::decay_t<T>, typename T::self_type>) {
			return p_object && p_object->_derives_from(typeid(T)) ? static_cast<const T *>(p_object) : nullptr;
		} else {
			// T does not use GDCLASS, must fall back to dynamic_cast.
			return p_object ? dynamic_cast<const T *>(p_object) : nullptr;
		}
	}

	enum {
		NOTIFICATION_POSTINITIALIZE = 0,
		NOTIFICATION_PREDELETE = 1,
		NOTIFICATION_EXTENSION_RELOADED = 2,
		// Internal notification to send after NOTIFICATION_PREDELETE, not bound to scripting.
		NOTIFICATION_PREDELETE_CLEANUP = 3,
	};

	/* TYPE API */
	static void get_inheritance_list_static(List<String> *p_inheritance_list) { p_inheritance_list->push_back("Object"); }

	static String get_class_static() { return "Object"; }
	static String get_parent_class_static() { return String(); }

	virtual String get_class() const {
		if (_extension) {
			return _extension->class_name.operator String();
		}
		return "Object";
	}
	virtual String get_save_class() const { return get_class(); } //class stored when saving

	virtual bool is_class(const String &p_class) const {
		if (_extension && _extension->is_class(p_class)) {
			return true;
		}
		return (p_class == "Object");
	}
	virtual bool is_class_ptr(void *p_ptr) const { return get_class_ptr_static() == p_ptr; }

	_FORCE_INLINE_ const StringName &get_class_name() const {
		if (_extension) {
			// Can't put inside the unlikely as constructor can run it
			return _extension->class_name;
		}

		if (unlikely(!_class_name_ptr)) {
			// While class is initializing / deinitializing, constructors and destructurs
			// need access to the proper class at the proper stage.
			return *_get_class_namev();
		}
		return *_class_name_ptr;
	}

	StringName get_class_name_for_extension(const GDExtension *p_library) const;

	/* IAPI */

	void set(const StringName &p_name, const Variant &p_value, bool *r_valid = nullptr);
	Variant get(const StringName &p_name, bool *r_valid = nullptr) const;
	void set_indexed(const Vector<StringName> &p_names, const Variant &p_value, bool *r_valid = nullptr);
	Variant get_indexed(const Vector<StringName> &p_names, bool *r_valid = nullptr) const;

	void get_property_list(List<PropertyInfo> *p_list, bool p_reversed = false) const;
	void validate_property(PropertyInfo &p_property) const;
	bool property_can_revert(const StringName &p_name) const;
	Variant property_get_revert(const StringName &p_name) const;

	bool has_method(const StringName &p_method) const;
	int get_method_argument_count(const StringName &p_method, bool *r_is_valid = nullptr) const;
	void get_method_list(List<MethodInfo> *p_list) const;
	Variant callv(const StringName &p_method, const Array &p_args);
	virtual Variant callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	virtual Variant call_const(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error);

	template <typename... VarArgs>
	Variant call(const StringName &p_method, VarArgs... p_args) {
		Variant args[sizeof...(p_args) + 1] = { p_args..., Variant() }; // +1 makes sure zero sized arrays are also supported.
		const Variant *argptrs[sizeof...(p_args) + 1];
		for (uint32_t i = 0; i < sizeof...(p_args); i++) {
			argptrs[i] = &args[i];
		}
		Callable::CallError cerr;
		const Variant ret = callp(p_method, sizeof...(p_args) == 0 ? nullptr : (const Variant **)argptrs, sizeof...(p_args), cerr);
		return (cerr.error == Callable::CallError::CALL_OK) ? ret : Variant();
	}

	void notification(int p_notification, bool p_reversed = false);
	virtual String to_string();

	// Used mainly by script, get and set all INCLUDING string.
	virtual Variant getvar(const Variant &p_key, bool *r_valid = nullptr) const;
	virtual void setvar(const Variant &p_key, const Variant &p_value, bool *r_valid = nullptr);

	/* SCRIPT */

// When in debug, some non-virtual functions can be overridden for multithreaded guards.
#ifdef DEBUG_ENABLED
#define MTVIRTUAL virtual
#else
#define MTVIRTUAL
#endif

	MTVIRTUAL void set_script(const Variant &p_script);
	MTVIRTUAL Variant get_script() const;

	MTVIRTUAL bool has_meta(const StringName &p_name) const;
	MTVIRTUAL void set_meta(const StringName &p_name, const Variant &p_value);
	MTVIRTUAL void remove_meta(const StringName &p_name);
	MTVIRTUAL Variant get_meta(const StringName &p_name, const Variant &p_default = Variant()) const;
	MTVIRTUAL void get_meta_list(List<StringName> *p_list) const;
	MTVIRTUAL void merge_meta_from(const Object *p_src);

#ifdef TOOLS_ENABLED
	void set_edited(bool p_edited);
	bool is_edited() const;
	// This function is used to check when something changed beyond a point, it's used mainly for generating previews.
	uint32_t get_edited_version() const;
#endif

	void set_script_instance(ScriptInstance *p_instance);
	_FORCE_INLINE_ ScriptInstance *get_script_instance() const { return script_instance; }

	// Some script languages can't control instance creation, so this function eases the process.
	void set_script_and_instance(const Variant &p_script, ScriptInstance *p_instance);

	void add_user_signal(const MethodInfo &p_signal);

	template <typename... VarArgs>
	Error emit_signal(const StringName &p_name, VarArgs... p_args) {
		Variant args[sizeof...(p_args) + 1] = { p_args..., Variant() }; // +1 makes sure zero sized arrays are also supported.
		const Variant *argptrs[sizeof...(p_args) + 1];
		for (uint32_t i = 0; i < sizeof...(p_args); i++) {
			argptrs[i] = &args[i];
		}
		return emit_signalp(p_name, sizeof...(p_args) == 0 ? nullptr : (const Variant **)argptrs, sizeof...(p_args));
	}

	MTVIRTUAL Error emit_signalp(const StringName &p_name, const Variant **p_args, int p_argcount);
	MTVIRTUAL bool has_signal(const StringName &p_name) const;
	MTVIRTUAL void get_signal_list(List<MethodInfo> *p_signals) const;
	MTVIRTUAL void get_signal_connection_list(const StringName &p_signal, List<Connection> *p_connections) const;
	MTVIRTUAL void get_all_signal_connections(List<Connection> *p_connections) const;
	MTVIRTUAL int get_persistent_signal_connection_count() const;
	MTVIRTUAL void get_signals_connected_to_this(List<Connection> *p_connections) const;

	MTVIRTUAL Error connect(const StringName &p_signal, const Callable &p_callable, uint32_t p_flags = 0);
	MTVIRTUAL void disconnect(const StringName &p_signal, const Callable &p_callable);
	MTVIRTUAL bool is_connected(const StringName &p_signal, const Callable &p_callable) const;
	MTVIRTUAL bool has_connections(const StringName &p_signal) const;

	template <typename... VarArgs>
	void call_deferred(const StringName &p_name, VarArgs... p_args) {
		MessageQueue::get_singleton()->push_call(this, p_name, p_args...);
	}

	void set_deferred(const StringName &p_property, const Variant &p_value);

	void set_block_signals(bool p_block);
	bool is_blocking_signals() const;

	Variant::Type get_static_property_type(const StringName &p_property, bool *r_valid = nullptr) const;
	Variant::Type get_static_property_type_indexed(const Vector<StringName> &p_path, bool *r_valid = nullptr) const;

	// Translate message (internationalization).
	String tr(const StringName &p_message, const StringName &p_context = "") const;
	// Translate message with plural form (internationalization).
	String tr_n(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;

	bool _is_queued_for_deletion = false; // Set to true by SceneTree::queue_delete().
	bool is_queued_for_deletion() const;

	_FORCE_INLINE_ void set_message_translation(bool p_enable) { _can_translate = p_enable; }
	_FORCE_INLINE_ bool can_translate_messages() const { return _can_translate; }

	virtual StringName get_translation_domain() const;
	virtual void set_translation_domain(const StringName &p_domain);

#ifdef TOOLS_ENABLED
	virtual void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const;
	void editor_set_section_unfold(const String &p_section, bool p_unfolded, bool p_initializing = false);
	bool editor_is_section_unfolded(const String &p_section);
	const HashSet<String> &editor_get_section_folding() const { return editor_section_folding; }
	void editor_clear_section_folding() { editor_section_folding.clear(); }

#endif

	// Used by script languages to store binding data.
	void *get_instance_binding(void *p_token, const GDExtensionInstanceBindingCallbacks *p_callbacks);
	// Used on creation by binding only.
	void set_instance_binding(void *p_token, void *p_binding, const GDExtensionInstanceBindingCallbacks *p_callbacks);
	bool has_instance_binding(void *p_token);
	void free_instance_binding(void *p_token);

#ifdef TOOLS_ENABLED
	void clear_internal_extension();
	void reset_internal_extension(ObjectGDExtension *p_extension);
	bool is_extension_placeholder() const { return _extension && _extension->is_placeholder; }
#endif

	void clear_internal_resource_paths();

	_ALWAYS_INLINE_ bool is_ref_counted() const { return type_is_reference; }

	void cancel_free();

	Object();
	virtual ~Object();
};

bool predelete_handler(Object *p_object);
void postinitialize_handler(Object *p_object);

class ObjectDB {
// This needs to add up to 63, 1 bit is for reference.
#define OBJECTDB_VALIDATOR_BITS 39
#define OBJECTDB_VALIDATOR_MASK ((uint64_t(1) << OBJECTDB_VALIDATOR_BITS) - 1)
#define OBJECTDB_SLOT_MAX_COUNT_BITS 24
#define OBJECTDB_SLOT_MAX_COUNT_MASK ((uint64_t(1) << OBJECTDB_SLOT_MAX_COUNT_BITS) - 1)
#define OBJECTDB_REFERENCE_BIT (uint64_t(1) << (OBJECTDB_SLOT_MAX_COUNT_BITS + OBJECTDB_VALIDATOR_BITS))

	struct ObjectSlot { // 128 bits per slot.
		uint64_t validator : OBJECTDB_VALIDATOR_BITS;
		uint64_t next_free : OBJECTDB_SLOT_MAX_COUNT_BITS;
		uint64_t is_ref_counted : 1;
		Object *object = nullptr;
	};

	static SpinLock spin_lock;
	static uint32_t slot_count;
	static uint32_t slot_max;
	static ObjectSlot *object_slots;
	static uint64_t validator_counter;

	friend class Object;
	friend void unregister_core_types();
	static void cleanup();

	static ObjectID add_instance(Object *p_object);
	static void remove_instance(Object *p_object);

	friend void register_core_types();
	static void setup();

public:
	typedef void (*DebugFunc)(Object *p_obj);

	_ALWAYS_INLINE_ static Object *get_instance(ObjectID p_instance_id) {
		uint64_t id = p_instance_id;
		uint32_t slot = id & OBJECTDB_SLOT_MAX_COUNT_MASK;

		ERR_FAIL_COND_V(slot >= slot_max, nullptr); // This should never happen unless RID is corrupted.

		spin_lock.lock();

		uint64_t validator = (id >> OBJECTDB_SLOT_MAX_COUNT_BITS) & OBJECTDB_VALIDATOR_MASK;

		if (unlikely(object_slots[slot].validator != validator)) {
			spin_lock.unlock();
			return nullptr;
		}

		Object *object = object_slots[slot].object;

		spin_lock.unlock();

		return object;
	}

	template <typename T>
	_ALWAYS_INLINE_ static T *get_instance(ObjectID p_instance_id) {
		return Object::cast_to<T>(get_instance(p_instance_id));
	}

	template <typename T>
	_ALWAYS_INLINE_ static Ref<T> get_ref(ObjectID p_instance_id); // Defined in ref_counted.h

	static void debug_objects(DebugFunc p_func);
	static int get_object_count();
};
