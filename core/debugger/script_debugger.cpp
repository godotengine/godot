/*************************************************************************/
/*  script_debugger.cpp                                                  */
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

#include "script_debugger.h"

#include "core/debugger/engine_debugger.h"

void ScriptDebugger::insert_breakpoint(int p_line, const StringName &p_source) {
	if (!breakpoints.has(p_line)) {
		breakpoints[p_line] = HashSet<StringName>();
	}
	breakpoints[p_line].insert(p_source);
}

void ScriptDebugger::remove_breakpoint(int p_line, const StringName &p_source) {
	if (!breakpoints.has(p_line)) {
		return;
	}

	breakpoints[p_line].erase(p_source);
	if (breakpoints[p_line].size() == 0) {
		breakpoints.erase(p_line);
	}
}

bool ScriptDebugger::is_breakpoint(int p_line, const StringName &p_source) const {
	if (!breakpoints.has(p_line)) {
		return false;
	}
	return breakpoints[p_line].has(p_source);
}

bool ScriptDebugger::is_breakpoint_line(int p_line) const {
	return breakpoints.has(p_line);
}

String ScriptDebugger::breakpoint_find_source(const String &p_source) const {
	return p_source;
}

void ScriptDebugger::clear_breakpoints() {
	breakpoints.clear();
}

void ScriptDebugger::set_skip_breakpoints(bool p_skip_breakpoints) {
	skip_breakpoints = p_skip_breakpoints;
}

bool ScriptDebugger::is_skipping_breakpoints() {
	return skip_breakpoints;
}

bool ScriptDebugger::_try_claim_debugger(const ScriptLanguageThreadContext &p_any_thread) {
	MutexLock<BinaryMutex> lock(_mutex_thread_transfer);
	if (_focused_thread.is_valid() && !_focused_thread->is_dead()) {
		return false;
	}
	_focused_thread = Ref<ScriptLanguageThreadContext>(&p_any_thread);
	return true;
}

void ScriptDebugger::step(ScriptLanguageThreadContext &p_any_thread) {
	if (p_any_thread.is_main_thread()) {
		EngineDebugger::get_singleton()->poll_events(false);
	}

	if (_break_requested.is_set()) {
		// Warning: there is a race condition here: Multiple threads
		// may get this far, but that can be safely ignored. Those threads
		// will simply compete for debug(...) normally, just like multiple threads
		// hitting breakpoints.
		_break_requested.clear();
		debug(p_any_thread);
		// We may have failed to debug, continue below like a secondary thread.
	}

	if (!_hold_threads.is_set()) {
		// Only pause if requested.  This is also where we return after successful debugging.
		return;
	}

	// Check in this context (now conceptually owned by debugger.)
	const DebugThreadID tid = p_any_thread.debug_get_thread_id();
	{
		MutexLock<BinaryMutex> lock(_mutex_thread_transfer);
		// Check again because we could have just been resumed and we
		// were blocked while the previous batch was cleaning out.
		if (!_hold_threads.is_set()) {
			return;
		}
		_held_threads[tid] = Ref<ScriptLanguageThreadContext>(&p_any_thread);
	}

	EngineDebugger::get_singleton()->thread_paused(p_any_thread);
	if (p_any_thread.is_main_thread()) {
		// Keep servicing non-core debug messages while waiting to resume.
		while (!p_any_thread.wait_resume_ms(1)) {
			EngineDebugger::get_singleton()->poll_events(false);
		}
	} else {
		p_any_thread.wait_resume();
	}

	// Reclaim our context to resume running.
	MutexLock<BinaryMutex> lock(_mutex_thread_transfer);
	_held_threads.erase(tid);
}

void ScriptDebugger::debug_request_break() {
	_break_requested.set();
}

void ScriptDebugger::debug(ScriptLanguageThreadContext &p_any_thread) {
	const bool is_first = _try_claim_debugger(p_any_thread);

	if (!is_first) {
		// A thread is already being debugged, just indicate that this thread has also hit a breakpoint or error.
		EngineDebugger::get_singleton()->request_debug(p_any_thread);

		// Suspend on step() like all the other extra threads.
		return;
	}

	if (_hold_other_threads_on_debug_start) {
		_hold_threads.set();
	} else {
		_hold_threads.clear();
	}

	// This thread is also available for interrogation, so check it in,
	// now conceptually owned by debugger.
	// Not using smart lock here because of smart lock on same mutex
	// below (in case of very poor optimizing compiler.)
	const DebugThreadID tid = p_any_thread.debug_get_thread_id();
	_mutex_thread_transfer.lock();
	_held_threads[tid] = Ref<ScriptLanguageThreadContext>(&p_any_thread);
	_mutex_thread_transfer.unlock();

	// Run the "OTHER" channel of debugging protocol on this thread, since it is the only thread that is
	// guaranteed not to block.  If Main gets blocked, then secondary debug captures won't get replies.
	EngineDebugger::get_singleton()->debug(p_any_thread);

	{
		MutexLock<BinaryMutex> lock(_mutex_thread_transfer);

		// We don't need to be resumed.
		_held_threads.erase(tid);

		// Release the debugger, so others can claim it.
		const bool wrong_thread_released = _focused_thread.ptr() != &p_any_thread;
		_focused_thread.unref();

		// Resume everyone.
		_hold_threads.clear();
		for (const KeyValue<DebugThreadID, Ref<ScriptLanguageThreadContext>> &thread : _held_threads) {
			// Thread will remove itself.
			thread.value->resume();
		}

		ERR_FAIL_COND_MSG(wrong_thread_released, "Debugged thread reference was corrupted during debugging; please report broken implementation.");
	}
}

void ScriptDebugger::try_get_stack_dump(const DebugThreadID &p_tid, DebuggerMarshalls::ScriptStackDump &p_dump) {
	MutexLock<BinaryMutex> lock(_mutex_thread_transfer);
	const ConstHeldThreadsIterator it = _held_threads.find(p_tid);
	if (it != _held_threads.end()) {
		p_dump.populate(**(it->value));
	}
}

void ScriptDebugger::try_send_stack_frame_variables(const DebugThreadID &p_tid, int p_level, RemoteDebugger &p_remote_debugger) {
	MutexLock<BinaryMutex> lock(_mutex_thread_transfer);
	const ConstHeldThreadsIterator it = _held_threads.find(p_tid);
	if (it != _held_threads.end()) {
		p_remote_debugger.send_stack_frame_variables(**(it->value), p_level);
	} else {
		p_remote_debugger.send_empty_stack_frame(p_tid);
	}
}
