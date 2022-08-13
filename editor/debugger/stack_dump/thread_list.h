/*************************************************************************/
/*  thread_list.h                                                        */
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

#ifndef THREAD_LIST_H
#define THREAD_LIST_H

#include "core/object/object.h"
#include "thread_info.h"

class Script;

namespace editor::dbg::sd {

class View;

// Manages the thread information collection based on signals from the debugger.
class ThreadList : public Object {
	GDCLASS(ThreadList, Object)
	using Status = ThreadInfo::Status;
	using DebugThreadID = ThreadInfo::DebugThreadID;

	// The GUI view for this data.
	// REVISIT: replace with signals only.
	View *_view = nullptr;
	void _on_view_tree_exiting();

	// Main table of threads.
	HashMap<ThreadInfo::DebugThreadID, Ref<ThreadInfo>, ThreadInfo::Hasher> _threads;

	// A synthetic record shown when main's state is not known.
	Ref<ThreadInfo> _placeholder_main;

	// Running counters to give friendly ID numbers to main/worker thread contexts, respectively.
	// Note there will be more than one numbered context for the same OS thread if multiple languages are used.
	int _next_main_thread_number = 1;
	int _next_thread_number = 100;

	using Field = ThreadInfo::Field;
	Field _current_sort_field = Field::DEBUG_ID;

	Ref<ThreadInfo> _create_thread_info(const DebugThreadID &p_debug_thread_id, bool p_is_main_thread);

	// Custom sorting based on current sort key.
	bool thread_info_less_than(const ThreadInfo *const &p_left, const ThreadInfo *const &p_right) const;

	void _sort_table();
	void _update_placeholder_main(bool p_is_main_thread);
	void _update_or_create_thread(const PackedByteArray &p_debug_thread_id, bool p_is_main_thread, const String &p_reason, Status p_status, int p_severity_code, bool p_can_debug, bool p_has_stack_dump);
	void _update_status_for_threads();
	int _find_least_greater(const Ref<ThreadInfo> &p_thread) const;

	// Warning: The signals that call these are defined as PackedByteArray, even if DebugThreadID ends up being something else, so we don't use that type here.
	void _on_debugger_thread_breaked(const PackedByteArray &p_debug_thread_id, bool p_is_main_thread, const String &p_reason, int p_severity_code, bool p_can_debug);
	void _on_debugger_thread_paused(const PackedByteArray &p_debug_thread_id, bool p_is_main_thread);
	void _on_debugger_thread_alert(const PackedByteArray &p_debug_thread_id, bool p_is_main_thread, const String &p_reason, int p_severity_code, bool p_can_debug, bool p_has_stack_dump);
	void _on_debugger_thread_continued(const PackedByteArray &p_debug_thread_id);
	void _on_debugger_thread_info(const PackedByteArray &p_debug_thread_id, const String &p_language, const PackedByteArray &p_thread_tag, const String &p_thread_name);
	void _on_debugger_thread_stack_dump(const PackedByteArray &p_debug_thread_id, const TypedArray<Dictionary> &p_stack_dump_info);
	void _on_debugger_thread_exited(const PackedByteArray &p_debug_thread_id);
	void _on_debugger_clear_execution(const Ref<Script> &p_script);

public:
	// Connect to all the signals that we need from the debugger.
	Error connect(Object &debugger);
	void disconnect(Object &debugger);

	// Connect to the view. REVISIT: Change to opaque signals only relationship.
	void set_view(View *view);
	void _on_view_item_edited();
	void _on_view_sort_requested(int p_field_index);

	void clear_threads();
};

} // namespace editor::dbg::sd

#endif // THREAD_LIST_H
