/**************************************************************************/
/*  node_path.hpp                                                         */
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

#include <godot_cpp/core/defs.hpp>

#include <gdextension_interface.h>

namespace godot {

class Array;
class Dictionary;
class String;
class StringName;
class Variant;

class NodePath {
	static constexpr size_t NODE_PATH_SIZE = 8;
	alignas(8) uint8_t opaque[NODE_PATH_SIZE] = {};

	friend class Variant;

	static struct _MethodBindings {
		GDExtensionTypeFromVariantConstructorFunc from_variant_constructor;
		GDExtensionPtrConstructor constructor_0;
		GDExtensionPtrConstructor constructor_1;
		GDExtensionPtrConstructor constructor_2;
		GDExtensionPtrDestructor destructor;
		GDExtensionPtrBuiltInMethod method_is_absolute;
		GDExtensionPtrBuiltInMethod method_get_name_count;
		GDExtensionPtrBuiltInMethod method_get_name;
		GDExtensionPtrBuiltInMethod method_get_subname_count;
		GDExtensionPtrBuiltInMethod method_hash;
		GDExtensionPtrBuiltInMethod method_get_subname;
		GDExtensionPtrBuiltInMethod method_get_concatenated_names;
		GDExtensionPtrBuiltInMethod method_get_concatenated_subnames;
		GDExtensionPtrBuiltInMethod method_slice;
		GDExtensionPtrBuiltInMethod method_get_as_property_path;
		GDExtensionPtrBuiltInMethod method_is_empty;
		GDExtensionPtrOperatorEvaluator operator_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not;
		GDExtensionPtrOperatorEvaluator operator_equal_NodePath;
		GDExtensionPtrOperatorEvaluator operator_not_equal_NodePath;
		GDExtensionPtrOperatorEvaluator operator_in_Dictionary;
		GDExtensionPtrOperatorEvaluator operator_in_Array;
	} _method_bindings;

	static void init_bindings();
	static void _init_bindings_constructors_destructor();

	NodePath(const Variant *p_variant);

public:
	_FORCE_INLINE_ GDExtensionTypePtr _native_ptr() const { return const_cast<uint8_t(*)[NODE_PATH_SIZE]>(&opaque); }
	NodePath();
	NodePath(const NodePath &p_from);
	NodePath(const String &p_from);
	NodePath(NodePath &&p_other);
	NodePath(const char *p_from);
	NodePath(const wchar_t *p_from);
	NodePath(const char16_t *p_from);
	NodePath(const char32_t *p_from);
	~NodePath();
	bool is_absolute() const;
	int64_t get_name_count() const;
	StringName get_name(int64_t p_idx) const;
	int64_t get_subname_count() const;
	int64_t hash() const;
	StringName get_subname(int64_t p_idx) const;
	StringName get_concatenated_names() const;
	StringName get_concatenated_subnames() const;
	NodePath slice(int64_t p_begin, int64_t p_end = 2147483647) const;
	NodePath get_as_property_path() const;
	bool is_empty() const;
	bool operator==(const Variant &p_other) const;
	bool operator!=(const Variant &p_other) const;
	bool operator!() const;
	bool operator==(const NodePath &p_other) const;
	bool operator!=(const NodePath &p_other) const;
	NodePath &operator=(const NodePath &p_other);
	NodePath &operator=(NodePath &&p_other);
};

} // namespace godot
