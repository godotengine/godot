/*************************************************************************/
/*  object.h                                                             */
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

#ifndef OBJECT_H
#define OBJECT_H

#include "core/extension/gdnative_interface.h"
#include "core/object/object_id.h"
#include "core/os/rw_lock.h"
#include "core/os/spin_lock.h"
#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#include "core/templates/map.h"
#include "core/templates/safe_refcount.h"
#include "core/templates/set.h"
#include "core/templates/vmap.h"
#include "core/variant/callable_bind.h"
#include "core/variant/variant.h"

#define VARIANT_ARG_LIST const Variant &p_arg1 = Variant(), const Variant &p_arg2 = Variant(), const Variant &p_arg3 = Variant(), const Variant &p_arg4 = Variant(), const Variant &p_arg5 = Variant(), const Variant &p_arg6 = Variant(), const Variant &p_arg7 = Variant(), const Variant &p_arg8 = Variant()
#define VARIANT_ARG_PASS p_arg1, p_arg2, p_arg3, p_arg4, p_arg5, p_arg6, p_arg7, p_arg8
#define VARIANT_ARG_DECLARE const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3, const Variant &p_arg4, const Variant &p_arg5, const Variant &p_arg6, const Variant &p_arg7, const Variant &p_arg8
#define VARIANT_ARG_MAX 8
#define VARIANT_ARGPTRS const Variant *argptr[8] = { &p_arg1, &p_arg2, &p_arg3, &p_arg4, &p_arg5, &p_arg6, &p_arg7, &p_arg8 };
#define VARIANT_ARGPTRS_PASS *argptr[0], *argptr[1], *argptr[2], *argptr[3], *argptr[4], *argptr[5], *argptr[6]], *argptr[7]
#define VARIANT_ARGS_FROM_ARRAY(m_arr) m_arr[0], m_arr[1], m_arr[2], m_arr[3], m_arr[4], m_arr[5], m_arr[6], m_arr[7]

/**
@author Juan Linietsky <reduzio@gmail.com>
*/

enum PropertyHint {
	PROPERTY_HINT_NONE, ///< no hint provided.
	PROPERTY_HINT_RANGE, ///< hint_text = "min,max[,step][,or_greater][,or_lesser][,noslider][,radians][,degrees][,exp][,suffix:<keyword>] range.
	PROPERTY_HINT_ENUM, ///< hint_text= "val1,val2,val3,etc"
	PROPERTY_HINT_ENUM_SUGGESTION, ///< hint_text= "val1,val2,val3,etc"
	PROPERTY_HINT_EXP_EASING, /// exponential easing function (Math::ease) use "attenuation" hint string to revert (flip h), "full" to also include in/out. (ie: "attenuation,inout")
	PROPERTY_HINT_LENGTH, ///< hint_text= "length" (as integer)
	PROPERTY_HINT_KEY_ACCEL, ///< hint_text= "length" (as integer)
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
	PROPERTY_HINT_RESOURCE_TYPE, ///< a resource object type
	PROPERTY_HINT_MULTILINE_TEXT, ///< used for string properties that can contain multiple lines
	PROPERTY_HINT_PLACEHOLDER_TEXT, ///< used to set a placeholder text for string properties
	PROPERTY_HINT_COLOR_NO_ALPHA, ///< used for ignoring alpha component when editing a color
	PROPERTY_HINT_IMAGE_COMPRESS_LOSSY,
	PROPERTY_HINT_IMAGE_COMPRESS_LOSSLESS,
	PROPERTY_HINT_OBJECT_ID,
	PROPERTY_HINT_TYPE_STRING, ///< a type string, the hint is the base type to choose
	PROPERTY_HINT_NODE_PATH_TO_EDITED_NODE, ///< so something else can provide this (used in scripts)
	PROPERTY_HINT_METHOD_OF_VARIANT_TYPE, ///< a method of a type
	PROPERTY_HINT_METHOD_OF_BASE_TYPE, ///< a method of a base type
	PROPERTY_HINT_METHOD_OF_INSTANCE, ///< a method of an instance
	PROPERTY_HINT_METHOD_OF_SCRIPT, ///< a method of a script & base
	PROPERTY_HINT_PROPERTY_OF_VARIANT_TYPE, ///< a property of a type
	PROPERTY_HINT_PROPERTY_OF_BASE_TYPE, ///< a property of a base type
	PROPERTY_HINT_PROPERTY_OF_INSTANCE, ///< a property of an instance
	PROPERTY_HINT_PROPERTY_OF_SCRIPT, ///< a property of a script & base
	PROPERTY_HINT_OBJECT_TOO_BIG, ///< object is too big to send
	PROPERTY_HINT_NODE_PATH_VALID_TYPES,
	PROPERTY_HINT_SAVE_FILE, ///< a file path must be passed, hint_text (optionally) is a filter "*.png,*.wav,*.doc,". This opens a save dialog
	PROPERTY_HINT_INT_IS_OBJECTID,
	PROPERTY_HINT_ARRAY_TYPE,
	PROPERTY_HINT_INT_IS_POINTER,
	PROPERTY_HINT_MAX,
	// When updating PropertyHint, also sync the hardcoded list in VisualScriptEditorVariableEdit
};

enum PropertyUsageFlags {
	PROPERTY_USAGE_NONE = 0,
	PROPERTY_USAGE_STORAGE = 1,
	PROPERTY_USAGE_EDITOR = 2,
	PROPERTY_USAGE_NETWORK = 4,
	PROPERTY_USAGE_EDITOR_HELPER = 8,
	PROPERTY_USAGE_CHECKABLE = 16, //used for editing global variables
	PROPERTY_USAGE_CHECKED = 32, //used for editing global variables
	PROPERTY_USAGE_INTERNATIONALIZED = 64, //hint for internationalized strings
	PROPERTY_USAGE_GROUP = 128, //used for grouping props in the editor
	PROPERTY_USAGE_CATEGORY = 256,
	PROPERTY_USAGE_SUBGROUP = 512,
	PROPERTY_USAGE_NO_INSTANCE_STATE = 2048,
	PROPERTY_USAGE_RESTART_IF_CHANGED = 4096,
	PROPERTY_USAGE_SCRIPT_VARIABLE = 8192,
	PROPERTY_USAGE_STORE_IF_NULL = 16384,
	PROPERTY_USAGE_ANIMATE_AS_TRIGGER = 32768,
	PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED = 65536,
	PROPERTY_USAGE_SCRIPT_DEFAULT_VALUE = 1 << 17,
	PROPERTY_USAGE_CLASS_IS_ENUM = 1 << 18,
	PROPERTY_USAGE_NIL_IS_VARIANT = 1 << 19,
	PROPERTY_USAGE_INTERNAL = 1 << 20,
	PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE = 1 << 21, // If the object is duplicated also this property will be duplicated
	PROPERTY_USAGE_HIGH_END_GFX = 1 << 22,
	PROPERTY_USAGE_NODE_PATH_FROM_SCENE_ROOT = 1 << 23,
	PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT = 1 << 24,
	PROPERTY_USAGE_KEYING_INCREMENTS = 1 << 25, // Used in inspector to increment property when keyed in animation player
	PROPERTY_USAGE_DEFERRED_SET_RESOURCE = 1 << 26, // when loading, the resource for this property can be set at the end of loading
	PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT = 1 << 27, // For Object properties, instantiate them when creating in editor.
	PROPERTY_USAGE_EDITOR_BASIC_SETTING = 1 << 28, //for project or editor settings, show when basic settings are selected
	PROPERTY_USAGE_READ_ONLY = 1 << 29, // Mark a property as read-only in the inspector.
	PROPERTY_USAGE_ARRAY = 1 << 30, // Used in the inspector to group properties as elements of an array.

	PROPERTY_USAGE_DEFAULT = PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_NETWORK,
	PROPERTY_USAGE_DEFAULT_INTL = PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_NETWORK | PROPERTY_USAGE_INTERNATIONALIZED,
	PROPERTY_USAGE_NO_EDITOR = PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_NETWORK,
};

#define ADD_SIGNAL(m_signal) ::ClassDB::add_signal(get_class_static(), m_signal)
#define ADD_PROPERTY(m_property, m_setter, m_getter) ::ClassDB::add_property(get_class_static(), m_property, _scs_create(m_setter), _scs_create(m_getter))
#define ADD_PROPERTYI(m_property, m_setter, m_getter, m_index) ::ClassDB::add_property(get_class_static(), m_property, _scs_create(m_setter), _scs_create(m_getter), m_index)
#define ADD_PROPERTY_DEFAULT(m_property, m_default) ::ClassDB::set_property_default_value(get_class_static(), m_property, m_default)
#define ADD_GROUP(m_name, m_prefix) ::ClassDB::add_property_group(get_class_static(), m_name, m_prefix)
#define ADD_SUBGROUP(m_name, m_prefix) ::ClassDB::add_property_subgroup(get_class_static(), m_name, m_prefix)
#define ADD_LINKED_PROPERTY(m_property, m_linked_property) ::ClassDB::add_linked_property(get_class_static(), m_property, m_linked_property)

#define ADD_ARRAY_COUNT(m_label, m_count_property, m_count_property_setter, m_count_property_getter, m_prefix) ClassDB::add_property_array_count(get_class_static(), m_label, m_count_property, _scs_create(m_count_property_setter), _scs_create(m_count_property_getter), m_prefix)
#define ADD_ARRAY_COUNT_WITH_USAGE_FLAGS(m_label, m_count_property, m_count_property_setter, m_count_property_getter, m_prefix, m_property_usage_flags) ClassDB::add_property_array_count(get_class_static(), m_label, m_count_property, _scs_create(m_count_property_setter), _scs_create(m_count_property_getter), m_prefix, m_property_usage_flags)
#define ADD_ARRAY(m_array_path, m_prefix) ClassDB::add_property_array(get_class_static(), m_array_path, m_prefix)

struct PropertyInfo {
	Variant::Type type = Variant::NIL;
	String name;
	StringName class_name; // For classes
	PropertyHint hint = PROPERTY_HINT_NONE;
	String hint_string;
	uint32_t usage = PROPERTY_USAGE_DEFAULT;

#ifdef TOOLS_ENABLED
	Vector<String> linked_properties;
#endif

	_FORCE_INLINE_ PropertyInfo added_usage(uint32_t p_fl) const {
		PropertyInfo pi = *this;
		pi.usage |= p_fl;
		return pi;
	}

	operator Dictionary() const;

	static PropertyInfo from_dict(const Dictionary &p_dict);

	PropertyInfo() {}

	PropertyInfo(const Variant::Type p_type, const String p_name, const PropertyHint p_hint = PROPERTY_HINT_NONE, const String &p_hint_string = "", const uint32_t p_usage = PROPERTY_USAGE_DEFAULT, const StringName &p_class_name = StringName()) :
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

Array convert_property_list(const List<PropertyInfo> *p_list);

struct MethodInfo {
	String name;
	PropertyInfo return_val;
	uint32_t flags; // NOLINT - prevent clang-tidy to assign method_bind.h constant here, it should stay in .cpp.
	int id = 0;
	List<PropertyInfo> arguments;
	Vector<Variant> default_arguments;

	inline bool operator==(const MethodInfo &p_method) const { return id == p_method.id; }
	inline bool operator<(const MethodInfo &p_method) const { return id == p_method.id ? (name < p_method.name) : (id < p_method.id); }

	operator Dictionary() const;

	static MethodInfo from_dict(const Dictionary &p_dict);

	MethodInfo();
	MethodInfo(const String &p_name);
	MethodInfo(const String &p_name, const PropertyInfo &p_param1);
	MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2);
	MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3);
	MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4);
	MethodInfo(const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4, const PropertyInfo &p_param5);
	MethodInfo(Variant::Type ret);
	MethodInfo(Variant::Type ret, const String &p_name);
	MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1);
	MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2);
	MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3);
	MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4);
	MethodInfo(Variant::Type ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4, const PropertyInfo &p_param5);
	MethodInfo(const PropertyInfo &p_ret, const String &p_name);
	MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1);
	MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2);
	MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3);
	MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4);
	MethodInfo(const PropertyInfo &p_ret, const String &p_name, const PropertyInfo &p_param1, const PropertyInfo &p_param2, const PropertyInfo &p_param3, const PropertyInfo &p_param4, const PropertyInfo &p_param5);
};

// old cast_to
//if ( is_type(T::get_class_static()) )
//return static_cast<T*>(this);
////else
//return nullptr;

// API used to extend in GDNative and other C compatible compiled languages
class MethodBind;

struct ObjectNativeExtension {
	ObjectNativeExtension *parent = nullptr;
	List<ObjectNativeExtension *> children;
	StringName parent_class_name;
	StringName class_name;
	bool editor_class = false;
	GDNativeExtensionClassSet set;
	GDNativeExtensionClassGet get;
	GDNativeExtensionClassGetPropertyList get_property_list;
	GDNativeExtensionClassFreePropertyList free_property_list;
	GDNativeExtensionClassNotification notification;
	GDNativeExtensionClassToString to_string;
	GDNativeExtensionClassReference reference;
	GDNativeExtensionClassReference unreference;

	_FORCE_INLINE_ bool is_class(const String &p_class) const {
		const ObjectNativeExtension *e = this;
		while (e) {
			if (p_class == e->class_name.operator String()) {
				return true;
			}
			e = e->parent;
		}
		return false;
	}
	void *class_userdata = nullptr;

	GDNativeExtensionClassCreateInstance create_instance;
	GDNativeExtensionClassFreeInstance free_instance;
	GDNativeExtensionClassGetVirtual get_virtual;
};

#define GDVIRTUAL_CALL(m_name, ...) _gdvirtual_##m_name##_call(__VA_ARGS__)
#define GDVIRTUAL_CALL_PTR(m_obj, m_name, ...) m_obj->_gdvirtual_##m_name##_call(__VA_ARGS__)
#ifdef DEBUG_METHODS_ENABLED
#define GDVIRTUAL_BIND(m_name, ...) ::ClassDB::add_virtual_method(get_class_static(), _gdvirtual_##m_name##_get_method_info(), true, sarray(__VA_ARGS__));
#else
#define GDVIRTUAL_BIND(m_name, ...)
#endif
#define GDVIRTUAL_IS_OVERRIDDEN(m_name) _gdvirtual_##m_name##_overridden()
#define GDVIRTUAL_IS_OVERRIDDEN_PTR(m_obj, m_name) m_obj->_gdvirtual_##m_name##_overridden()

/*
   the following is an incomprehensible blob of hacks and workarounds to compensate for many of the fallencies in C++. As a plus, this macro pretty much alone defines the object model.
*/

#define REVERSE_GET_PROPERTY_LIST                                  \
public:                                                            \
	_FORCE_INLINE_ bool _is_gpl_reversed() const { return true; }; \
                                                                   \
private:

#define UNREVERSE_GET_PROPERTY_LIST                                 \
public:                                                             \
	_FORCE_INLINE_ bool _is_gpl_reversed() const { return false; }; \
                                                                    \
private:

#define GDCLASS(m_class, m_inherits)                                                                                                             \
private:                                                                                                                                         \
	void operator=(const m_class &p_rval) {}                                                                                                     \
	mutable StringName _class_name;                                                                                                              \
	friend class ::ClassDB;                                                                                                                      \
                                                                                                                                                 \
public:                                                                                                                                          \
	virtual String get_class() const override {                                                                                                  \
		if (_get_extension()) {                                                                                                                  \
			return _get_extension()->class_name.operator String();                                                                               \
		}                                                                                                                                        \
		return String(#m_class);                                                                                                                 \
	}                                                                                                                                            \
	virtual const StringName *_get_class_namev() const override {                                                                                \
		if (_get_extension()) {                                                                                                                  \
			return &_get_extension()->class_name;                                                                                                \
		}                                                                                                                                        \
		if (!_class_name) {                                                                                                                      \
			_class_name = get_class_static();                                                                                                    \
		}                                                                                                                                        \
		return &_class_name;                                                                                                                     \
	}                                                                                                                                            \
	static _FORCE_INLINE_ void *get_class_ptr_static() {                                                                                         \
		static int ptr;                                                                                                                          \
		return &ptr;                                                                                                                             \
	}                                                                                                                                            \
	static _FORCE_INLINE_ String get_class_static() {                                                                                            \
		return String(#m_class);                                                                                                                 \
	}                                                                                                                                            \
	static _FORCE_INLINE_ String get_parent_class_static() {                                                                                     \
		return m_inherits::get_class_static();                                                                                                   \
	}                                                                                                                                            \
	static void get_inheritance_list_static(List<String> *p_inheritance_list) {                                                                  \
		m_inherits::get_inheritance_list_static(p_inheritance_list);                                                                             \
		p_inheritance_list->push_back(String(#m_class));                                                                                         \
	}                                                                                                                                            \
	static String get_category_static() {                                                                                                        \
		String category = m_inherits::get_category_static();                                                                                     \
		if (_get_category != m_inherits::_get_category) {                                                                                        \
			if (!category.is_empty()) {                                                                                                          \
				category += "/";                                                                                                                 \
			}                                                                                                                                    \
			category += _get_category();                                                                                                         \
		}                                                                                                                                        \
		return category;                                                                                                                         \
	}                                                                                                                                            \
	virtual bool is_class(const String &p_class) const override {                                                                                \
		if (_get_extension() && _get_extension()->is_class(p_class)) {                                                                           \
			return true;                                                                                                                         \
		}                                                                                                                                        \
		return (p_class == (#m_class)) ? true : m_inherits::is_class(p_class);                                                                   \
	}                                                                                                                                            \
	virtual bool is_class_ptr(void *p_ptr) const override { return (p_ptr == get_class_ptr_static()) ? true : m_inherits::is_class_ptr(p_ptr); } \
                                                                                                                                                 \
	static void get_valid_parents_static(List<String> *p_parents) {                                                                              \
		if (m_class::_get_valid_parents_static != m_inherits::_get_valid_parents_static) {                                                       \
			m_class::_get_valid_parents_static(p_parents);                                                                                       \
		}                                                                                                                                        \
                                                                                                                                                 \
		m_inherits::get_valid_parents_static(p_parents);                                                                                         \
	}                                                                                                                                            \
                                                                                                                                                 \
protected:                                                                                                                                       \
	_FORCE_INLINE_ static void (*_get_bind_methods())() {                                                                                        \
		return &m_class::_bind_methods;                                                                                                          \
	}                                                                                                                                            \
                                                                                                                                                 \
public:                                                                                                                                          \
	static void initialize_class() {                                                                                                             \
		static bool initialized = false;                                                                                                         \
		if (initialized) {                                                                                                                       \
			return;                                                                                                                              \
		}                                                                                                                                        \
		m_inherits::initialize_class();                                                                                                          \
		::ClassDB::_add_class<m_class>();                                                                                                        \
		if (m_class::_get_bind_methods() != m_inherits::_get_bind_methods()) {                                                                   \
			_bind_methods();                                                                                                                     \
		}                                                                                                                                        \
		initialized = true;                                                                                                                      \
	}                                                                                                                                            \
                                                                                                                                                 \
protected:                                                                                                                                       \
	virtual void _initialize_classv() override {                                                                                                 \
		initialize_class();                                                                                                                      \
	}                                                                                                                                            \
	_FORCE_INLINE_ bool (Object::*_get_get() const)(const StringName &p_name, Variant &) const {                                                 \
		return (bool(Object::*)(const StringName &, Variant &) const) & m_class::_get;                                                           \
	}                                                                                                                                            \
	virtual bool _getv(const StringName &p_name, Variant &r_ret) const override {                                                                \
		if (m_class::_get_get() != m_inherits::_get_get()) {                                                                                     \
			if (_get(p_name, r_ret)) {                                                                                                           \
				return true;                                                                                                                     \
			}                                                                                                                                    \
		}                                                                                                                                        \
		return m_inherits::_getv(p_name, r_ret);                                                                                                 \
	}                                                                                                                                            \
	_FORCE_INLINE_ bool (Object::*_get_set() const)(const StringName &p_name, const Variant &p_property) {                                       \
		return (bool(Object::*)(const StringName &, const Variant &)) & m_class::_set;                                                           \
	}                                                                                                                                            \
	virtual bool _setv(const StringName &p_name, const Variant &p_property) override {                                                           \
		if (m_inherits::_setv(p_name, p_property)) {                                                                                             \
			return true;                                                                                                                         \
		}                                                                                                                                        \
		if (m_class::_get_set() != m_inherits::_get_set()) {                                                                                     \
			return _set(p_name, p_property);                                                                                                     \
		}                                                                                                                                        \
		return false;                                                                                                                            \
	}                                                                                                                                            \
	_FORCE_INLINE_ void (Object::*_get_get_property_list() const)(List<PropertyInfo> * p_list) const {                                           \
		return (void(Object::*)(List<PropertyInfo> *) const) & m_class::_get_property_list;                                                      \
	}                                                                                                                                            \
	virtual void _get_property_listv(List<PropertyInfo> *p_list, bool p_reversed) const override {                                               \
		if (!p_reversed) {                                                                                                                       \
			m_inherits::_get_property_listv(p_list, p_reversed);                                                                                 \
		}                                                                                                                                        \
		p_list->push_back(PropertyInfo(Variant::NIL, get_class_static(), PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_CATEGORY));                \
		if (!_is_gpl_reversed()) {                                                                                                               \
			::ClassDB::get_property_list(#m_class, p_list, true, this);                                                                          \
		}                                                                                                                                        \
		if (m_class::_get_get_property_list() != m_inherits::_get_get_property_list()) {                                                         \
			_get_property_list(p_list);                                                                                                          \
		}                                                                                                                                        \
		if (_is_gpl_reversed()) {                                                                                                                \
			::ClassDB::get_property_list(#m_class, p_list, true, this);                                                                          \
		}                                                                                                                                        \
		if (p_reversed) {                                                                                                                        \
			m_inherits::_get_property_listv(p_list, p_reversed);                                                                                 \
		}                                                                                                                                        \
	}                                                                                                                                            \
	_FORCE_INLINE_ void (Object::*_get_notification() const)(int) {                                                                              \
		return (void(Object::*)(int)) & m_class::_notification;                                                                                  \
	}                                                                                                                                            \
	virtual void _notificationv(int p_notification, bool p_reversed) override {                                                                  \
		if (!p_reversed) {                                                                                                                       \
			m_inherits::_notificationv(p_notification, p_reversed);                                                                              \
		}                                                                                                                                        \
		if (m_class::_get_notification() != m_inherits::_get_notification()) {                                                                   \
			_notification(p_notification);                                                                                                       \
		}                                                                                                                                        \
		if (p_reversed) {                                                                                                                        \
			m_inherits::_notificationv(p_notification, p_reversed);                                                                              \
		}                                                                                                                                        \
	}                                                                                                                                            \
                                                                                                                                                 \
private:

#define OBJ_CATEGORY(m_category)                                        \
protected:                                                              \
	_FORCE_INLINE_ static String _get_category() { return m_category; } \
                                                                        \
private:

#define OBJ_SAVE_TYPE(m_class)                                          \
public:                                                                 \
	virtual String get_save_class() const override { return #m_class; } \
                                                                        \
private:

class ScriptInstance;

class Object {
public:
	enum ConnectFlags {
		CONNECT_DEFERRED = 1,
		CONNECT_PERSIST = 2, // hint for scene to save this connection
		CONNECT_ONESHOT = 4,
		CONNECT_REFERENCE_COUNTED = 8,
	};

	struct Connection {
		::Signal signal;
		Callable callable;

		uint32_t flags = 0;
		Vector<Variant> binds;
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

	ObjectNativeExtension *_extension = nullptr;
	GDExtensionClassInstancePtr _extension_instance = nullptr;

	struct SignalData {
		struct Slot {
			int reference_count = 0;
			Connection conn;
			List<Connection>::Element *cE = nullptr;
		};

		MethodInfo user;
		VMap<Callable, Slot> slot_map;
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
	void _postinitialize();
	bool _can_translate = true;
	bool _emitting = false;
#ifdef TOOLS_ENABLED
	bool _edited = false;
	uint32_t _edited_version = 0;
	Set<String> editor_section_folding;
#endif
	ScriptInstance *script_instance = nullptr;
	Variant script; //reference does not yet exist, store it in a
	Dictionary metadata;
	mutable StringName _class_name;
	mutable const StringName *_class_ptr = nullptr;

	void _add_user_signal(const String &p_name, const Array &p_args = Array());
	bool _has_user_signal(const StringName &p_name) const;
	Variant _emit_signal(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Array _get_signal_list() const;
	Array _get_signal_connection_list(const String &p_signal) const;
	Array _get_incoming_connections() const;
	void _set_bind(const String &p_set, const Variant &p_value);
	Variant _get_bind(const String &p_name) const;
	void _set_indexed_bind(const NodePath &p_name, const Variant &p_value);
	Variant _get_indexed_bind(const NodePath &p_name) const;

	_FORCE_INLINE_ void _construct_object(bool p_reference);

	friend class RefCounted;
	bool type_is_reference = false;

	std::mutex _instance_binding_mutex;
	struct InstanceBinding {
		void *binding;
		void *token;
		GDNativeInstanceBindingFreeCallback free_callback = nullptr;
		GDNativeInstanceBindingReferenceCallback reference_callback = nullptr;
	};
	InstanceBinding *_instance_bindings = nullptr;
	uint32_t _instance_binding_count = 0;

	Object(bool p_reference);

protected:
	_FORCE_INLINE_ bool _instance_binding_reference(bool p_reference) {
		bool can_die = true;
		if (_instance_bindings) {
			_instance_binding_mutex.lock();
			for (uint32_t i = 0; i < _instance_binding_count; i++) {
				if (_instance_bindings[i].reference_callback) {
					if (!_instance_bindings[i].reference_callback(_instance_bindings[i].token, _instance_bindings[i].binding, p_reference)) {
						can_die = false;
					}
				}
			}
			_instance_binding_mutex.unlock();
		}
		return can_die;
	}
	friend class NativeExtensionMethodBind;
	_ALWAYS_INLINE_ const ObjectNativeExtension *_get_extension() const { return _extension; }
	_ALWAYS_INLINE_ GDExtensionClassInstancePtr _get_extension_instance() const { return _extension_instance; }
	virtual void _initialize_classv() { initialize_class(); }
	virtual bool _setv(const StringName &p_name, const Variant &p_property) { return false; };
	virtual bool _getv(const StringName &p_name, Variant &r_property) const { return false; };
	virtual void _get_property_listv(List<PropertyInfo> *p_list, bool p_reversed) const {};
	virtual void _notificationv(int p_notification, bool p_reversed) {}

	static String _get_category() { return ""; }
	static void _bind_methods();
	bool _set(const StringName &p_name, const Variant &p_property) { return false; };
	bool _get(const StringName &p_name, Variant &r_property) const { return false; };
	void _get_property_list(List<PropertyInfo> *p_list) const {};
	void _notification(int p_notification) {}

	_FORCE_INLINE_ static void (*_get_bind_methods())() {
		return &Object::_bind_methods;
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
	_FORCE_INLINE_ void (Object::*_get_notification() const)(int) {
		return &Object::_notification;
	}
	static void get_valid_parents_static(List<String> *p_parents);
	static void _get_valid_parents_static(List<String> *p_parents);

	//Variant _call_bind(const StringName& p_name, const Variant& p_arg1 = Variant(), const Variant& p_arg2 = Variant(), const Variant& p_arg3 = Variant(), const Variant& p_arg4 = Variant());
	//void _call_deferred_bind(const StringName& p_name, const Variant& p_arg1 = Variant(), const Variant& p_arg2 = Variant(), const Variant& p_arg3 = Variant(), const Variant& p_arg4 = Variant());

	Variant _call_bind(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Variant _call_deferred_bind(const Variant **p_args, int p_argcount, Callable::CallError &r_error);

	virtual const StringName *_get_class_namev() const {
		if (!_class_name) {
			_class_name = get_class_static();
		}
		return &_class_name;
	}

	Vector<StringName> _get_meta_list_bind() const;
	Array _get_property_list_bind() const;
	Array _get_method_list_bind() const;

	void _clear_internal_resource_paths(const Variant &p_var);

	friend class ClassDB;
	virtual void _validate_property(PropertyInfo &property) const;

	void _disconnect(const StringName &p_signal, const Callable &p_callable, bool p_force = false);

public: //should be protected, but bug in clang++
	static void initialize_class();
	_FORCE_INLINE_ static void register_custom_data_to_otdb() {}

public:
	void notify_property_list_changed();

	static void *get_class_ptr_static() {
		static int ptr;
		return &ptr;
	}

	bool _is_gpl_reversed() const { return false; }

	_FORCE_INLINE_ ObjectID get_instance_id() const { return _instance_id; }

	template <class T>
	static T *cast_to(Object *p_object) {
#ifndef NO_SAFE_CAST
		return dynamic_cast<T *>(p_object);
#else
		if (!p_object) {
			return nullptr;
		}
		if (p_object->is_class_ptr(T::get_class_ptr_static())) {
			return static_cast<T *>(p_object);
		} else {
			return nullptr;
		}
#endif
	}

	template <class T>
	static const T *cast_to(const Object *p_object) {
#ifndef NO_SAFE_CAST
		return dynamic_cast<const T *>(p_object);
#else
		if (!p_object) {
			return nullptr;
		}
		if (p_object->is_class_ptr(T::get_class_ptr_static())) {
			return static_cast<const T *>(p_object);
		} else {
			return nullptr;
		}
#endif
	}

	enum {
		NOTIFICATION_POSTINITIALIZE = 0,
		NOTIFICATION_PREDELETE = 1
	};

	/* TYPE API */
	static void get_inheritance_list_static(List<String> *p_inheritance_list) { p_inheritance_list->push_back("Object"); }

	static String get_class_static() { return "Object"; }
	static String get_parent_class_static() { return String(); }
	static String get_category_static() { return String(); }

	virtual String get_class() const {
		if (_extension)
			return _extension->class_name.operator String();
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
			return _extension->class_name;
		}
		if (!_class_ptr) {
			return *_get_class_namev();
		} else {
			return *_class_ptr;
		}
	}

	/* IAPI */
	//void set(const String& p_name, const Variant& p_value);
	//Variant get(const String& p_name) const;

	void set(const StringName &p_name, const Variant &p_value, bool *r_valid = nullptr);
	Variant get(const StringName &p_name, bool *r_valid = nullptr) const;
	void set_indexed(const Vector<StringName> &p_names, const Variant &p_value, bool *r_valid = nullptr);
	Variant get_indexed(const Vector<StringName> &p_names, bool *r_valid = nullptr) const;

	void get_property_list(List<PropertyInfo> *p_list, bool p_reversed = false) const;

	bool has_method(const StringName &p_method) const;
	void get_method_list(List<MethodInfo> *p_list) const;
	Variant callv(const StringName &p_method, const Array &p_args);
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Variant call(const StringName &p_name, VARIANT_ARG_LIST); // C++ helper

	void notification(int p_notification, bool p_reversed = false);
	virtual String to_string();

	//used mainly by script, get and set all INCLUDING string
	virtual Variant getvar(const Variant &p_key, bool *r_valid = nullptr) const;
	virtual void setvar(const Variant &p_key, const Variant &p_value, bool *r_valid = nullptr);

	/* SCRIPT */

	void set_script(const Variant &p_script);
	Variant get_script() const;

	/* SCRIPT */

	bool has_meta(const StringName &p_name) const;
	void set_meta(const StringName &p_name, const Variant &p_value);
	void remove_meta(const StringName &p_name);
	Variant get_meta(const StringName &p_name) const;
	void get_meta_list(List<StringName> *p_list) const;

#ifdef TOOLS_ENABLED
	void set_edited(bool p_edited);
	bool is_edited() const;
	uint32_t get_edited_version() const; //this function is used to check when something changed beyond a point, it's used mainly for generating previews
#endif

	void set_script_instance(ScriptInstance *p_instance);
	_FORCE_INLINE_ ScriptInstance *get_script_instance() const { return script_instance; }

	void set_script_and_instance(const Variant &p_script, ScriptInstance *p_instance); //some script languages can't control instance creation, so this function eases the process

	void add_user_signal(const MethodInfo &p_signal);
	Error emit_signal(const StringName &p_name, VARIANT_ARG_LIST);
	Error emit_signal(const StringName &p_name, const Variant **p_args, int p_argcount);
	bool has_signal(const StringName &p_name) const;
	void get_signal_list(List<MethodInfo> *p_signals) const;
	void get_signal_connection_list(const StringName &p_signal, List<Connection> *p_connections) const;
	void get_all_signal_connections(List<Connection> *p_connections) const;
	int get_persistent_signal_connection_count() const;
	void get_signals_connected_to_this(List<Connection> *p_connections) const;

	Error connect(const StringName &p_signal, const Callable &p_callable, const Vector<Variant> &p_binds = Vector<Variant>(), uint32_t p_flags = 0);
	void disconnect(const StringName &p_signal, const Callable &p_callable);
	bool is_connected(const StringName &p_signal, const Callable &p_callable) const;

	void call_deferred(const StringName &p_method, VARIANT_ARG_LIST);
	void set_deferred(const StringName &p_property, const Variant &p_value);

	void set_block_signals(bool p_block);
	bool is_blocking_signals() const;

	Variant::Type get_static_property_type(const StringName &p_property, bool *r_valid = nullptr) const;
	Variant::Type get_static_property_type_indexed(const Vector<StringName> &p_path, bool *r_valid = nullptr) const;

	virtual void get_translatable_strings(List<String> *p_strings) const;

	virtual void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const;

	String tr(const StringName &p_message, const StringName &p_context = "") const; // translate message (internationalization)
	String tr_n(const StringName &p_message, const StringName &p_message_plural, int p_n, const StringName &p_context = "") const;

	bool _is_queued_for_deletion = false; // set to true by SceneTree::queue_delete()
	bool is_queued_for_deletion() const;

	_FORCE_INLINE_ void set_message_translation(bool p_enable) { _can_translate = p_enable; }
	_FORCE_INLINE_ bool can_translate_messages() const { return _can_translate; }

#ifdef TOOLS_ENABLED
	void editor_set_section_unfold(const String &p_section, bool p_unfolded);
	bool editor_is_section_unfolded(const String &p_section);
	const Set<String> &editor_get_section_folding() const { return editor_section_folding; }
	void editor_clear_section_folding() { editor_section_folding.clear(); }

#endif

	// Used by script languages to store binding data.
	void *get_instance_binding(void *p_token, const GDNativeInstanceBindingCallbacks *p_callbacks);
	// Used on creation by binding only.
	void set_instance_binding(void *p_token, void *p_binding, const GDNativeInstanceBindingCallbacks *p_callbacks);
	bool has_instance_binding(void *p_token);

	void clear_internal_resource_paths();

	_ALWAYS_INLINE_ bool is_ref_counted() const { return type_is_reference; }

	Object();
	virtual ~Object();
};

bool predelete_handler(Object *p_object);
void postinitialize_handler(Object *p_object);

class ObjectDB {
//this needs to add up to 63, 1 bit is for reference
#define OBJECTDB_VALIDATOR_BITS 39
#define OBJECTDB_VALIDATOR_MASK ((uint64_t(1) << OBJECTDB_VALIDATOR_BITS) - 1)
#define OBJECTDB_SLOT_MAX_COUNT_BITS 24
#define OBJECTDB_SLOT_MAX_COUNT_MASK ((uint64_t(1) << OBJECTDB_SLOT_MAX_COUNT_BITS) - 1)
#define OBJECTDB_REFERENCE_BIT (uint64_t(1) << (OBJECTDB_SLOT_MAX_COUNT_BITS + OBJECTDB_VALIDATOR_BITS))

	struct ObjectSlot { //128 bits per slot
		uint64_t validator : OBJECTDB_VALIDATOR_BITS;
		uint64_t next_free : OBJECTDB_SLOT_MAX_COUNT_BITS;
		uint64_t is_ref_counted : 1;
		Object *object;
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

		ERR_FAIL_COND_V(slot >= slot_max, nullptr); //this should never happen unless RID is corrupted

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
	static void debug_objects(DebugFunc p_func);
	static int get_object_count();
};

#endif // OBJECT_H
