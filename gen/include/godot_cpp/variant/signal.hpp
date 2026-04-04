/**************************************************************************/
/*  signal.hpp                                                            */
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
class Callable;
class Dictionary;
class Object;
class StringName;
class Variant;

class Signal {
	static constexpr size_t SIGNAL_SIZE = 16;
	alignas(8) uint8_t opaque[SIGNAL_SIZE] = {};

	friend class Variant;

	static struct _MethodBindings {
		GDExtensionTypeFromVariantConstructorFunc from_variant_constructor;
		GDExtensionPtrConstructor constructor_0;
		GDExtensionPtrConstructor constructor_1;
		GDExtensionPtrConstructor constructor_2;
		GDExtensionPtrDestructor destructor;
		GDExtensionPtrBuiltInMethod method_is_null;
		GDExtensionPtrBuiltInMethod method_get_object;
		GDExtensionPtrBuiltInMethod method_get_object_id;
		GDExtensionPtrBuiltInMethod method_get_name;
		GDExtensionPtrBuiltInMethod method_connect;
		GDExtensionPtrBuiltInMethod method_disconnect;
		GDExtensionPtrBuiltInMethod method_is_connected;
		GDExtensionPtrBuiltInMethod method_get_connections;
		GDExtensionPtrBuiltInMethod method_has_connections;
		GDExtensionPtrBuiltInMethod method_emit;
		GDExtensionPtrOperatorEvaluator operator_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not_equal_Variant;
		GDExtensionPtrOperatorEvaluator operator_not;
		GDExtensionPtrOperatorEvaluator operator_equal_Signal;
		GDExtensionPtrOperatorEvaluator operator_not_equal_Signal;
		GDExtensionPtrOperatorEvaluator operator_in_Dictionary;
		GDExtensionPtrOperatorEvaluator operator_in_Array;
	} _method_bindings;

	static void init_bindings();
	static void _init_bindings_constructors_destructor();

	Signal(const Variant *p_variant);

public:
	_FORCE_INLINE_ GDExtensionTypePtr _native_ptr() const { return const_cast<uint8_t(*)[SIGNAL_SIZE]>(&opaque); }
	Signal();
	Signal(const Signal &p_from);
	Signal(Object *p_object, const StringName &p_signal);
	Signal(Signal &&p_other);
	~Signal();
	bool is_null() const;
	Object *get_object() const;
	int64_t get_object_id() const;
	StringName get_name() const;
	int64_t connect(const Callable &p_callable, int64_t p_flags = 0);
	void disconnect(const Callable &p_callable);
	bool is_connected(const Callable &p_callable) const;
	Array get_connections() const;
	bool has_connections() const;
	template <typename... Args>
	void emit(const Args &...p_args) const;
	bool operator==(const Variant &p_other) const;
	bool operator!=(const Variant &p_other) const;
	bool operator!() const;
	bool operator==(const Signal &p_other) const;
	bool operator!=(const Signal &p_other) const;
	Signal &operator=(const Signal &p_other);
	Signal &operator=(Signal &&p_other);
};

} // namespace godot
