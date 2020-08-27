/*************************************************************************/
/*  callable.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdnative/callable.h"

#include "core/callable.h"
#include "core/resource.h"
#include "core/variant.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(sizeof(godot_callable) == sizeof(Callable), "Callable size mismatch");
static_assert(sizeof(godot_signal) == sizeof(Signal), "Signal size mismatch");

// Callable

void GDAPI godot_callable_new_with_object(godot_callable *r_dest, const godot_object *p_object, const godot_string_name *p_method) {
	Callable *dest = (Callable *)r_dest;
	const Object *object = (const Object *)p_object;
	const StringName *method = (const StringName *)p_method;
	memnew_placement(dest, Callable(object, *method));
}

void GDAPI godot_callable_new_with_object_id(godot_callable *r_dest, uint64_t p_objectid, const godot_string_name *p_method) {
	Callable *dest = (Callable *)r_dest;
	const StringName *method = (const StringName *)p_method;
	memnew_placement(dest, Callable(ObjectID(p_objectid), *method));
}

void GDAPI godot_callable_new_copy(godot_callable *r_dest, const godot_callable *p_src) {
	Callable *dest = (Callable *)r_dest;
	const Callable *src = (const Callable *)p_src;
	memnew_placement(dest, Callable(*src));
}

void GDAPI godot_callable_destroy(godot_callable *p_self) {
	Callable *self = (Callable *)p_self;
	self->~Callable();
}

godot_int GDAPI godot_callable_call(const godot_callable *p_self, const godot_variant **p_arguments, godot_int p_argcount, godot_variant *r_return_value) {
	const Callable *self = (const Callable *)p_self;
	const Variant **arguments = (const Variant **)p_arguments;
	Variant *return_value = (Variant *)r_return_value;
	Variant ret;
	Callable::CallError err;
	self->call(arguments, p_argcount, ret, err);
	if (return_value)
		(*return_value) = ret;
	return (godot_int)err.error;
}

void GDAPI godot_callable_call_deferred(const godot_callable *p_self, const godot_variant **p_arguments, godot_int p_argcount) {
	const Callable *self = (const Callable *)p_self;
	const Variant **arguments = (const Variant **)p_arguments;
	self->call_deferred(arguments, p_argcount);
}

godot_bool GDAPI godot_callable_is_null(const godot_callable *p_self) {
	const Callable *self = (const Callable *)p_self;
	return self->is_null();
}

godot_bool GDAPI godot_callable_is_custom(const godot_callable *p_self) {
	const Callable *self = (const Callable *)p_self;
	return self->is_custom();
}

godot_bool GDAPI godot_callable_is_standard(const godot_callable *p_self) {
	const Callable *self = (const Callable *)p_self;
	return self->is_standard();
}

godot_object GDAPI *godot_callable_get_object(const godot_callable *p_self) {
	const Callable *self = (const Callable *)p_self;
	return (godot_object *)self->get_object();
}

uint64_t GDAPI godot_callable_get_object_id(const godot_callable *p_self) {
	const Callable *self = (const Callable *)p_self;
	return (uint64_t)self->get_object_id();
}

godot_string_name GDAPI godot_callable_get_method(const godot_callable *p_self) {
	godot_string_name raw_dest;
	const Callable *self = (const Callable *)p_self;
	StringName *dest = (StringName *)&raw_dest;
	memnew_placement(dest, StringName(self->get_method()));
	return raw_dest;
}

uint32_t GDAPI godot_callable_hash(const godot_callable *p_self) {
	const Callable *self = (const Callable *)p_self;
	return self->hash();
}

godot_string GDAPI godot_callable_as_string(const godot_callable *p_self) {
	godot_string ret;
	const Callable *self = (const Callable *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

godot_bool GDAPI godot_callable_operator_equal(const godot_callable *p_self, const godot_callable *p_other) {
	const Callable *self = (const Callable *)p_self;
	const Callable *other = (const Callable *)p_other;
	return *self == *other;
}

godot_bool GDAPI godot_callable_operator_less(const godot_callable *p_self, const godot_callable *p_other) {
	const Callable *self = (const Callable *)p_self;
	const Callable *other = (const Callable *)p_other;
	return *self < *other;
}

// Signal

void GDAPI godot_signal_new_with_object(godot_signal *r_dest, const godot_object *p_object, const godot_string_name *p_name) {
	Signal *dest = (Signal *)r_dest;
	const Object *object = (const Object *)p_object;
	const StringName *name = (const StringName *)p_name;
	memnew_placement(dest, Signal(object, *name));
}

void GDAPI godot_signal_new_with_object_id(godot_signal *r_dest, uint64_t p_objectid, const godot_string_name *p_name) {
	Signal *dest = (Signal *)r_dest;
	const StringName *name = (const StringName *)p_name;
	memnew_placement(dest, Signal(ObjectID(p_objectid), *name));
}

void GDAPI godot_signal_new_copy(godot_signal *r_dest, const godot_signal *p_src) {
	Signal *dest = (Signal *)r_dest;
	const Signal *src = (const Signal *)p_src;
	memnew_placement(dest, Signal(*src));
}

void GDAPI godot_signal_destroy(godot_signal *p_self) {
	Signal *self = (Signal *)p_self;
	self->~Signal();
}

godot_int GDAPI godot_signal_emit(const godot_signal *p_self, const godot_variant **p_arguments, godot_int p_argcount) {
	const Signal *self = (const Signal *)p_self;
	const Variant **arguments = (const Variant **)p_arguments;
	return (godot_int)self->emit(arguments, p_argcount);
}

godot_int GDAPI godot_signal_connect(godot_signal *p_self, const godot_callable *p_callable, const godot_array *p_binds, uint32_t p_flags) {
	Signal *self = (Signal *)p_self;
	const Callable *callable = (const Callable *)p_callable;
	const Array *binds_ar = (const Array *)p_binds;
	Vector<Variant> binds;
	for (int i = 0; i < binds_ar->size(); i++) {
		binds.push_back(binds_ar->get(i));
	}
	return (godot_int)self->connect(*callable, binds, p_flags);
}

void GDAPI godot_signal_disconnect(godot_signal *p_self, const godot_callable *p_callable) {
	Signal *self = (Signal *)p_self;
	const Callable *callable = (const Callable *)p_callable;
	self->disconnect(*callable);
}

godot_bool GDAPI godot_signal_is_null(const godot_signal *p_self) {
	const Signal *self = (const Signal *)p_self;
	return self->is_null();
}

godot_bool GDAPI godot_signal_is_connected(const godot_signal *p_self, const godot_callable *p_callable) {
	const Signal *self = (const Signal *)p_self;
	const Callable *callable = (const Callable *)p_callable;
	return self->is_connected(*callable);
}

godot_array GDAPI godot_signal_get_connections(const godot_signal *p_self) {
	godot_array raw_dest;
	const Signal *self = (const Signal *)p_self;
	Array *dest = (Array *)&raw_dest;
	memnew_placement(dest, Array(self->get_connections()));
	return raw_dest;
}

godot_object GDAPI *godot_signal_get_object(const godot_signal *p_self) {
	const Signal *self = (const Signal *)p_self;
	return (godot_object *)self->get_object();
}

uint64_t GDAPI godot_signal_get_object_id(const godot_signal *p_self) {
	const Signal *self = (const Signal *)p_self;
	return (uint64_t)self->get_object_id();
}

godot_string_name GDAPI godot_signal_get_name(const godot_signal *p_self) {
	godot_string_name raw_dest;
	const Signal *self = (const Signal *)p_self;
	StringName *dest = (StringName *)&raw_dest;
	memnew_placement(dest, StringName(self->get_name()));
	return raw_dest;
}

godot_string GDAPI godot_signal_as_string(const godot_signal *p_self) {
	godot_string ret;
	const Signal *self = (const Signal *)p_self;
	memnew_placement(&ret, String(*self));
	return ret;
}

godot_bool GDAPI godot_signal_operator_equal(const godot_signal *p_self, const godot_signal *p_other) {
	const Signal *self = (const Signal *)p_self;
	const Signal *other = (const Signal *)p_other;
	return *self == *other;
}

godot_bool GDAPI godot_signal_operator_less(const godot_signal *p_self, const godot_signal *p_other) {
	const Signal *self = (const Signal *)p_self;
	const Signal *other = (const Signal *)p_other;
	return *self < *other;
}

#ifdef __cplusplus
}
#endif
