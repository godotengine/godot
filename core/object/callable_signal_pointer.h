/**************************************************************************/
/*  callable_signal_pointer.h                                             */
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

#include "callable_method_pointer.h"
#include "core/object/object.h"
#include "core/string/string_name.h"
#include "core/variant/callable.h"

class CallableCustomSignalPointer : public CallableCustomMethodPointerBase {
	struct Data {
		uint64_t object_id;
		Object *instance;
		StringName signal_name;
	} data;

public:
	virtual ObjectID get_object() const {
		if (ObjectDB::get_instance(ObjectID(data.object_id)) == nullptr) {
			return ObjectID();
		}
		return data.instance->get_instance_id(); // Can probably be changed to data.object_id?
	}

	virtual int get_argument_count(bool &r_is_valid) const {
		// Argument count is unknown from just the signal name.
		r_is_valid = false;
		return 0;
	}

	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
		ERR_FAIL_NULL_MSG(ObjectDB::get_instance(ObjectID(data.object_id)), "Invalid Object id '" + uitos(data.object_id) + "', can't call method.");
		r_return_value = data.instance->emit_signalp(data.signal_name, p_arguments, p_argcount);
	}

	CallableCustomSignalPointer(Object *p_instance, const StringName &p_signal_name) {
#ifdef DEBUG_ENABLED
		set_text("Object::emit_signal"); // Include name of signal in this?
#endif // DEBUG_ENABLED
		data.instance = p_instance;
		data.object_id = p_instance->get_instance_id();
		data.signal_name = p_signal_name;
		// Should be tightly packed.
		_setup((uint32_t *)&data, sizeof(uint64_t) + sizeof(Object *) + sizeof(StringName));
	}
};

#define callable_sp(I, S) Callable(memnew(CallableCustomSignalPointer(I, S)))
