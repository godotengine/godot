/*************************************************************************/
/*  script_debugger.h                                                    */
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

#ifndef SCRIPT_DEBUGGER_H
#define SCRIPT_DEBUGGER_H

#include "core/object/script_language.h"
#include "core/string/string_name.h"
#include "core/templates/hash_set.h"
#include "core/templates/vector.h"
#include "debugger_marshalls.h"
#include "remote_debugger.h"

class ScriptDebugger {
	using DebugThreadID = ScriptLanguageThreadContext::DebugThreadID;
	using StackInfo = ScriptLanguageThreadContext::StackInfo;

	bool skip_breakpoints = false;

	HashMap<int, HashSet<StringName>> breakpoints;

	// TODO remote configuration?
	bool _hold_other_threads_on_debug_start = true;

	// If true, all non-focused threads hold on step execute also.
	SafeFlag _hold_threads;

	// If true, there is a pending async request for break on any thread (usually from ctrl-c or similar.)
	SafeFlag _break_requested;

	// Ownership of the thread context of a paused thread is temporarily granted to
	// the debugger by storing it into _focused_thread or _held_threads under lock
	// of _mutex_thread_transfer.
	BinaryMutex _mutex_thread_transfer;
	Ref<ScriptLanguageThreadContext> _focused_thread;
	HashMap<DebugThreadID, Ref<ScriptLanguageThreadContext>, VariantHasher, VariantComparator> _held_threads;
	typedef HashMap<DebugThreadID, Ref<ScriptLanguageThreadContext>, VariantHasher, VariantComparator>::ConstIterator ConstHeldThreadsIterator;

	bool _try_claim_debugger(const ScriptLanguageThreadContext &p_any_thread);

public:
	/* BREAKPOINTS */

	String breakpoint_find_source(const String &p_source) const;
	void set_skip_breakpoints(bool p_skip_breakpoints);
	bool is_skipping_breakpoints();
	void insert_breakpoint(int p_line, const StringName &p_source);
	void remove_breakpoint(int p_line, const StringName &p_source);
	bool is_breakpoint(int p_line, const StringName &p_source) const;
	bool is_breakpoint_line(int p_line) const;
	void clear_breakpoints();
	const HashMap<int, HashSet<StringName>> &get_breakpoints() const { return breakpoints; }

	/* DEBUGGING */

	// Start new debugging session, from breakpoint or error, expected to block the caller being debugged.
	void debug(ScriptLanguageThreadContext &p_any_thread);

	// Called from all threads that are not the thread currently being debugged ("focused thread")
	// for every opcode while debugger is active, may block the caller.
	void step(ScriptLanguageThreadContext &p_any_thread);

	// Asynchronously request break by the next script language to execute something.
	void debug_request_break();

	// TODO can LocalDebugger use this or does this need a callback as below?
	void try_get_stack_dump(const DebugThreadID &p_tid, DebuggerMarshalls::ScriptStackDump &p_dump);

	// FIXME make this not depend on RemoteDebugger
	void try_send_stack_frame_variables(const DebugThreadID &p_tid, int p_level, RemoteDebugger &p_remote_debugger);

	ScriptDebugger() = default;
};

#endif // SCRIPT_DEBUGGER_H
