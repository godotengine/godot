/*************************************************************************/
/*  signal_watcher.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef SIGNAL_WATCHER_H
#define SIGNAL_WATCHER_H

#include "core/map.h"
#include "core/pair.h"
#include "core/reference.h"
#include "core/vector.h"

class SignalWatcher : public Reference {
	GDCLASS(SignalWatcher, Reference);

public:
	struct Params {
		Object *m_object;
		String m_signal;
		Array m_arguments;
	};
	static bool parse_params(const Variant **p_args, int p_argcount, Variant::CallError &r_error, Params &r_params);

	void watch(Object *p_object, const String &p_signal);
	void watch_all(Object *p_object);

	/// The signal was emitted at least once.
	bool called(const Object *p_object, const String &p_signal) const;
	/// The signal was emitted exactly once.
	bool called_once(const Object *p_object, const String &p_signal) const;
	/// The most recent signal was emitted with the specified arguments.
	bool called_with(const Object *p_object, const String &p_signal, const Array &p_arguments) const;
	/// The signal was emitted exactly once and that call was with the specified arguments.
	bool called_once_with(const Object *p_object, const String &p_signal, const Array &p_arguments) const;
	/// The signal has been called with the specified arguments.
	bool any_call(const Object *p_object, const String &p_signal, const Array &p_arguments) const;
	/// Checks if signal was emitted with the specified arguments, returns index of failure
	int has_calls(const Object *p_object, const String &p_signal, const Array &calls, bool any_order = false) const;
	/// Check the signal was never called.
	bool not_called(const Object *p_object, const String &p_signal) const;

	int call_count(const Object *p_object, const String &p_signal) const;
	Array calls(const Object *p_object, const String &p_signal) const;

	void reset();

protected:
	static void _bind_methods();

private:
	typedef Vector<Array> Args;
	typedef Pair<const Object *, String> ObjectSignal;
	typedef PairSort<const Object *, String> ObjectSignalSort;
	typedef Map<ObjectSignal, Args, ObjectSignalSort> ObjectSignalArgs;
	ObjectSignalArgs m_signals;

	Args *write(const Object *p_object, const String &p_signal);
	const Args *read(const Object *p_object, const String &p_signal) const;
	void touch(const Object *p_object, const String &p_signal);

	Variant _handler(const Variant **p_args, int p_argcount, Variant::CallError &r_error);

	Variant _called_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	Variant _called_once_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	Variant _any_call(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
};

#endif // SIGNAL_WATCHER_H
