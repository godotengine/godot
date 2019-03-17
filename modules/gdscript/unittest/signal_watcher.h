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
#include "core/reference.h"
#include "core/vector.h"

class SignalWatcher : public Reference {
	GDCLASS(SignalWatcher , Reference);

public:
	void watch(Object *object, const String &signal);

	/// The signal was emitted at least once.
	bool called(const Object *object, const String &signal) const;
	/// The signal was emitted exactly once.
	bool called_once(const Object *object, const String &signal) const;
	/// The most recent signal was emitted with the specified arguments.
	bool called_with(const Object *object, const String &signal, const Array &arguments) const;
	Variant _called_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	/// The signal was emitted exactly once and that call was with the specified arguments.
	bool called_once_with(const Object *object, const String &signal, const Array &arguments) const;
	Variant _called_once_with(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	/// The signal has been called with the specified arguments.
	bool any_call(const Object *object, const String &signal, const Array &arguments) const;
	Variant _any_call(const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	/// Checks if signal was emitted with the specified arguments.
	bool has_calls(const Object *object, const String &signal, const Array &calls, bool any_order = false) const;
	/// Check the signal was never called.
	bool not_called(const Object *object, const String &signal) const;

	int call_count(const Object *object, const String &signal) const;
	Array calls(const Object *object, const String &signal) const;

	void reset();

protected:
	static void _bind_methods();

private:
	typedef Vector<Variant> Args;
	typedef Map<String, Args> SignalArgs;
	typedef Map<const Object *, SignalArgs> ObjectSignalArgs;
	Map<const Object *, Map<String, Vector<Variant> > > m_signals;

	Args* write(const Object *object, const String &signal);
	const Args *read(const Object *object, const String &signal) const;
	void touch(const Object *object, const String &signal);

	Variant _handler(const Variant **p_args, int p_argcount, Variant::CallError &r_error);

	typedef bool (SignalWatcher::*CheckArg)(const Object *object, const String &signal, const Array &arguments) const;

	Variant _check_arguments(const Variant **p_args, int p_argcount, Variant::CallError &r_error, CheckArg check);
};

#endif // SIGNAL_WATCHER_H
