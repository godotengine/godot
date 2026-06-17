/**************************************************************************/
/*  EventSignal.cpp                                                       */
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

#include "EventSignal.h"

#include "../Godot-Wasm-Exports/header/caching_ptrs.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"

#include <cstdint>

// JS-side signal forwarder: called from C++ to deliver a signal event to .NET.
// Reads the double args array from WASM heap and forwards to the registered
// _crossRuntimeNotifyCSharp handler in the JS bootstrap layer.
EM_JS(void, notify_csharp_worker, (double obj_id, const char *signal_name, const double *args, int arg_count), {
	if (typeof window._crossRuntimeNotifyCSharp == = 'function') {
		const arr = new Float64Array(HEAPU8.buffer, args, arg_count);
		window._crossRuntimeNotifyCSharp(obj_id, UTF8ToString(signal_name), Array.from(arr));
	}
});

CrossRuntimeEventSignal *CrossRuntimeEventSignal::singleton = nullptr;

// Binds the internal signal receiver and the public connect/disconnect methods
// so they are accessible via ClassDB and callable from the .NET side.
void CrossRuntimeEventSignal::_bind_methods() {
	ClassDB::bind_vararg_method(
			METHOD_FLAGS_DEFAULT,
			"_on_any_signal",
			&CrossRuntimeEventSignal::_on_any_signal);
	ClassDB::bind_method(D_METHOD("connect_signal", "object_id", "signal_name"),
			&CrossRuntimeEventSignal::connect_signal);
	ClassDB::bind_method(D_METHOD("disconnect_signal", "object_id", "signal_name"),
			&CrossRuntimeEventSignal::disconnect_signal);
}

// Constructor: registers this node as the CrossRuntimeEventSignal engine singleton
// so it can be looked up by name from both C++ and .NET.
CrossRuntimeEventSignal::CrossRuntimeEventSignal() {
	singleton = this;
	Engine::get_singleton()->add_singleton(Engine::Singleton("CrossRuntimeEventSignal", this));
}

CrossRuntimeEventSignal::~CrossRuntimeEventSignal() {
	if (singleton == this) {
		singleton = nullptr;
	}
}

// Connects a Godot signal on the given object to the internal _on_any_signal handler.
// The object_id and signal_name are bound into the callable so they are available
// when the signal fires, allowing _on_any_signal to identify the source.
void CrossRuntimeEventSignal::connect_signal(int64_t object_id, const String &signal_name) {
	Object *obj = ObjectDB::get_instance(ObjectID((uint64_t)object_id));
	if (!obj) {
		return;
	}
	Callable cb = Callable(this, "_on_any_signal").bind(object_id, signal_name);
	obj->connect(signal_name, cb);
}

// Disconnects the previously bound signal handler from the object.
// The callable must match exactly (same target, method, and bound args) to disconnect correctly.
void CrossRuntimeEventSignal::disconnect_signal(int64_t object_id, const String &signal_name) {
	Object *obj = ObjectDB::get_instance(ObjectID((uint64_t)object_id));
	if (!obj) {
		return;
	}
	Callable cb = Callable(this, "_on_any_signal").bind(object_id, signal_name);
	obj->disconnect(signal_name, cb);
}

// Vararg signal receiver: called by Godot whenever a connected signal fires.
// The last two arguments are always the bound object_id and signal_name (appended by connect_signal).
// All preceding arguments are the signal's own parameters, converted to doubles for .NET transfer.
void CrossRuntimeEventSignal::_on_any_signal(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	r_error.error = Callable::CallError::CALL_OK;
	if (p_argcount < 2) {
		return;
	}

	int64_t source_id = (int64_t)(*p_args[p_argcount - 2]);
	String signal_name = (String)(*p_args[p_argcount - 1]);

	std::vector<double> args;
	// cache ptrs pattern has not been implemented here yet.
	for (int i = 0; i < p_argcount - 2; i++) {
		const Variant &v = *p_args[i];
		if (v.get_type() == Variant::OBJECT) {
			Object *obj = Object::cast_to<Object>(v);
			args.push_back(obj ? (double)(int64_t)obj : 0.0);
		} else {
			args.push_back((double)v);
		}
	}

	notify_csharp_worker(
			// Resolves the source object's cached raw ptr — .NET identifies objects by ptr,
			// not ObjectID, to avoid negative overflow on 64-bit ids.
			(int64_t)(uintptr_t)extract_ptr(source_id),
			signal_name.utf8().get_data(),
			args.data(),
			(int)args.size());
}
