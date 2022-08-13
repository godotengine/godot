/*************************************************************************/
/*  thread_list.cpp                                                      */
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

#include "thread_list.h"
#include "thread_info.h"
#include "thread_list_tree.h"

#include "core/object/callable_method_pointer.h"
#include "core/object/script_language.h"

namespace editor::dbg::sd {

//
// Implementation notes common to all these functions:
//
// These functions repeatedly check if _threads.has(key) because code that interacts with UI may erroneously process
// events late because someone inserts a call_deferred() somewhere.  When these late calls can be safely ignored, they
// are.
//
// These functions do not optimize the double lookups into _threads (HashMap::has, HashMap::operator[]) by using HashMap::get_ptr() because
// that would make the code less clean, by introducing more nullable pointers.  Optimization of lookups that happen at GUI
// interaction speed is not a goal of this code.
//

void ThreadList::_update_placeholder_main(bool p_is_main_thread) {
	if (!p_is_main_thread && _placeholder_main.is_null() && _threads.is_empty()) {
		// Show a blocked main thread because we know it exists.
		const DebugThreadID placeholder_thread_id = DebugThreadID();
		Ref<ThreadInfo> thread = _create_thread_info(placeholder_thread_id, true);
		thread->reason = "Main thread is blocked and unavailable for debugging";
		thread->can_debug = false;
		thread->severity_code = 0;
		thread->status = Status::BLOCKED;
		thread->thread_name = "Main (blocked)";
		_placeholder_main = thread;
		_threads[placeholder_thread_id] = thread;
		if (_view != nullptr) {
			_view->add_row(thread);
		}
		return;
	}
	if (p_is_main_thread && _placeholder_main.is_valid()) {
		const DebugThreadID placeholder_thread_id = DebugThreadID();
		_threads.erase(placeholder_thread_id);
		// Self-removes from tree on delete.
		_placeholder_main->free_tree_item();
		_placeholder_main.unref();
	}
}

void ThreadList::set_view(View *view) {
	_view = view;
	Error ret = _view->connect("tree_exiting", callable_mp(this, &ThreadList::_on_view_tree_exiting));
	ERR_FAIL_COND_MSG(ret != OK, "failed to connect required signal 'tree_exiting' from Tree control");
	ret = _view->connect("item_edited", callable_mp(this, &ThreadList::_on_view_item_edited));
	ERR_FAIL_COND_MSG(ret != OK, "failed to connect required signal 'item_edited' from Tree control");
	ret = _view->connect("sort_requested", callable_mp(this, &ThreadList::_on_view_sort_requested));
	if (ret != OK) {
		// This isn't strictly required.
		WARN_PRINT("failed to connect signal 'sort_requested' from thread list view control");
	}
}

void ThreadList::_on_view_item_edited() {
	_sort_table();
}

void ThreadList::_on_view_sort_requested(int p_field_index) {
	if (nullptr == _view) {
		// Deferred late.
		return;
	}
	_current_sort_field = _view->get_field(p_field_index);
	_sort_table();
}

void ThreadList::clear_threads() {
	for (KeyValue<DebugThreadID, Ref<ThreadInfo>> record : _threads) {
		record.value->free_tree_item();
	}
	_threads.clear();
	if (_placeholder_main.is_valid()) {
		_placeholder_main->free_tree_item();
		_placeholder_main.unref();
	}
}

void ThreadList::_on_view_tree_exiting() {
	if (nullptr == _view) {
		// Deferred late.
		return;
	}

	// About to be deallocated by Scene tree.
	_view->disconnect("item_edited", callable_mp(this, &ThreadList::_on_view_item_edited));
	_view->disconnect("tree_exiting", callable_mp(this, &ThreadList::_on_view_tree_exiting));
	_view->disconnect("sort_requested", callable_mp(this, &ThreadList::_on_view_sort_requested));
	_view = nullptr;
}

Ref<ThreadInfo> ThreadList::_create_thread_info(const DebugThreadID &p_debug_thread_id, bool p_is_main_thread) {
	const int thread_number = p_is_main_thread ? _next_main_thread_number++ : _next_thread_number++;
	return Ref<ThreadInfo>(memnew(ThreadInfo(thread_number, p_debug_thread_id, p_is_main_thread)));
}

bool ThreadList::thread_info_less_than(const ThreadInfo *const &p_left, const ThreadInfo *const &p_right) const {
	if (_view != nullptr) {
		if (_view->is_pin_column_visible()) {
			// Pinned threads sort above all other ones.
			const bool left_flag = _view->is_pinned(*p_left);
			const bool right_flag = _view->is_pinned(*p_right);
			if (right_flag && !left_flag) {
				return false;
			}
			if (left_flag && !right_flag) {
				return true;
			}
		}
	}
	switch (_current_sort_field) {
		case Field::ID: {
			if (p_left->thread_number == p_right->thread_number) {
				break;
			}
			return (p_left->thread_number < p_right->thread_number);
		}
		case Field::NAME: {
			// Sort empty last instead of first.
			if (p_left->thread_name.size() < 1) {
				if (p_right->thread_name.size() < 1) {
					break;
				}
				return false;
			}
			if (p_right->thread_name.size() < 1) {
				return true;
			}
			return (p_left->thread_name < p_right->thread_name);
		}
		case Field::LANGUAGE: {
			// Sort empty last instead of first.
			if (p_left->language.size() < 1) {
				if (p_right->language.size() < 1) {
					break;
				}
				return false;
			}
			if (p_right->language.size() < 1) {
				return true;
			}
			return (p_left->language < p_right->language);
		}
		default:
			break;
	}
	// Tie breaker on a unique value to keep view stable.
	return (p_left->get_debug_thread_id_hex() < p_right->get_debug_thread_id_hex());
}

void ThreadList::_update_status_for_threads() {
	if (nullptr == _view) {
		return;
	}
	for (const KeyValue<DebugThreadID, Ref<ThreadInfo>> &record : _threads) {
		_view->update_status(**record.value);
	}
}

int ThreadList::_find_least_greater(const Ref<ThreadInfo> &p_thread) const {
	// REVISIT: We are reaching into the view here, which is gross.  We should maintain the sorted list instead.
	if (nullptr == _view) {
		return -1;
	}
	int before_index = 0;
	for (TreeItem *item = _view->get_root()->get_first_child(); item != nullptr; item = item->get_next()) {
		Ref<ThreadInfo> existing = _view->get_meta_thread(*item);
		if (thread_info_less_than(*p_thread, *existing)) {
			// Found the first item that is greater.
			break;
		}
		++before_index;
	}
	return before_index;
}

void ThreadList::_sort_table() {
	if (nullptr == _view) {
		return;
	}

	// Convert values to Array as shallow uncounted copy.
	Vector<const ThreadInfo *> index;
	index.resize(_threads.size());
	int i = 0;
	for (KeyValue<DebugThreadID, Ref<ThreadInfo>> &record : _threads) {
		index.set(i, *record.value);
		i++;
	}

	// Sort by current sort order.
	index.sort_custom<CallableComparator, true>(callable_mp(this, &ThreadList::thread_info_less_than));

	// Last one never needs to be moved.
	index.resize(index.size() - 1);

	// Shuffle subtree
	bool first = true;
	TreeItem *previous = _view->get_root()->get_first_child();
	for (const ThreadInfo *const &thread : index) {
		TreeItem *item = thread->tree_item;
		if (first) {
			// Special case first item; start the child list with it.
			first = false;
			if (item != previous) {
				item->move_before(previous);
			}
		} else {
			// Append every other item, except last one (see above.)
			item->move_after(previous);
		}
		previous = item;
	}
}

void ThreadList::_on_debugger_thread_breaked(const PackedByteArray &p_debug_thread_id, bool p_is_main_thread, const String &p_reason, int p_severity_code, bool p_can_debug) {
	_update_placeholder_main(p_is_main_thread);
	Ref<ThreadInfo> thread;
	if (_threads.has(p_debug_thread_id)) {
		thread = _threads[p_debug_thread_id];
		thread->reason = p_reason;
		if (thread->tree_item != nullptr) {
			View::set_meta_stack(*thread->tree_item, Dictionary());
		}
	} else {
		thread = _create_thread_info(p_debug_thread_id, p_is_main_thread);
		thread->reason = p_reason;
		_threads[p_debug_thread_id] = thread;
		if (_view != nullptr) {
			_view->add_row(thread, _find_least_greater(thread));
		}
	}
	thread->can_debug = true;
	thread->has_stack_dump = true;
	thread->severity_code = p_severity_code;
	thread->status = p_can_debug ? Status::BREAKPOINT : Status::CRASHED;
	if (_view != nullptr) {
		_view->set_current(**thread);
		_update_status_for_threads();
	}
}

void ThreadList::_update_or_create_thread(const PackedByteArray &p_debug_thread_id, bool p_is_main_thread, const String &p_reason, Status p_status, int p_severity_code, bool p_can_debug, bool p_has_stack_dump) {
	_update_placeholder_main(p_is_main_thread);
	Ref<ThreadInfo> thread;
	if (_threads.has(p_debug_thread_id)) {
		thread = _threads[p_debug_thread_id];
		thread->reason = p_reason;
	} else {
		thread = _create_thread_info(p_debug_thread_id, p_is_main_thread);
		thread->reason = p_reason;
		_threads[p_debug_thread_id] = thread;
		if (_view != nullptr) {
			_view->add_row(thread, _find_least_greater(thread));
		}
	}
	thread->can_debug = p_can_debug;
	thread->has_stack_dump = p_has_stack_dump;
	thread->severity_code = p_severity_code;
	thread->status = p_status;
	if (_view != nullptr) {
		_view->update_status(**thread);
	}
}

void ThreadList::_on_debugger_thread_paused(const PackedByteArray &p_debug_thread_id, bool p_is_main_thread) {
	_update_or_create_thread(p_debug_thread_id, p_is_main_thread, "", Status::PAUSED, 0, true, true);
}

void ThreadList::_on_debugger_thread_alert(const PackedByteArray &p_debug_thread_id, bool p_is_main_thread, const String &p_reason, int p_severity_code, bool p_can_debug, bool p_has_stack_dump) {
	_update_or_create_thread(p_debug_thread_id, p_is_main_thread, p_reason, p_can_debug ? Status::ALERT : Status::CRASHED, p_severity_code, p_can_debug, p_has_stack_dump);
}

void ThreadList::_on_debugger_thread_continued(const PackedByteArray &p_debug_thread_id) {
	if (!_threads.has(p_debug_thread_id)) {
		// Deferred late or out of order.
		return;
	}

	Ref<ThreadInfo> thread = _threads[p_debug_thread_id];
	thread->can_debug = false;
	thread->severity_code = 0;
	thread->status = Status::RUNNING;
	if (_view != nullptr) {
		_view->update_status(**thread);
	}
}

void ThreadList::_on_debugger_thread_info(const PackedByteArray &p_debug_thread_id, const String &p_language, const PackedByteArray &p_thread_tag, const String &p_thread_name) {
	if (!_threads.has(p_debug_thread_id)) {
		// Deferred late or out of order.
		return;
	}

	Ref<ThreadInfo> thread = _threads[p_debug_thread_id];
	thread->language = p_language;
	thread->thread_tag = p_thread_tag;
	thread->thread_name = p_thread_name;
	if (_view != nullptr) {
		_view->update_info(**thread);
	}
	switch (_current_sort_field) {
		case Field::NAME:
		case Field::LANGUAGE:
			_sort_table();
			break;
		default:
			break;
	}
}

void ThreadList::_on_debugger_thread_stack_dump(const PackedByteArray &p_debug_thread_id, const TypedArray<Dictionary> &p_stack_dump_info) {
	if (!_threads.has(p_debug_thread_id)) {
		// Deferred late or out of order.
		return;
	}

	Ref<ThreadInfo> thread = _threads[p_debug_thread_id];
	thread->stack_dump_info = p_stack_dump_info;
	if (_view != nullptr) {
		_view->build_stack_dump(thread);
	}
}

void ThreadList::_on_debugger_thread_exited(const PackedByteArray &p_debug_thread_id) {
	if (!_threads.has(p_debug_thread_id)) {
		// Deferred late or out of order.
		return;
	}

	Ref<ThreadInfo> thread = _threads[p_debug_thread_id];
	thread->free_tree_item();
	_threads.erase(p_debug_thread_id);
}

void ThreadList::_on_debugger_clear_execution(const Ref<Script> &p_script) {
	clear_threads();
}

Error ThreadList::connect(Object &debugger) {
	Error ret = debugger.connect("thread_breaked", callable_mp(this, &ThreadList::_on_debugger_thread_breaked));
	ERR_FAIL_COND_V_MSG(ret != OK, ret, "failed to connect required signal 'thread_breaked' from debugger");

	ret = debugger.connect("thread_paused", callable_mp(this, &ThreadList::_on_debugger_thread_paused));
	ERR_FAIL_COND_V_MSG(ret != OK, ret, "failed to connect required signal 'thread_paused' from debugger");

	ret = debugger.connect("thread_alert", callable_mp(this, &ThreadList::_on_debugger_thread_alert));
	ERR_FAIL_COND_V_MSG(ret != OK, ret, "failed to connect required signal 'thread_alert' from debugger");

	ret = debugger.connect("thread_continued", callable_mp(this, &ThreadList::_on_debugger_thread_continued));
	ERR_FAIL_COND_V_MSG(ret != OK, ret, "failed to connect required signal 'thread_continued' from debugger");

	ret = debugger.connect("thread_exited", callable_mp(this, &ThreadList::_on_debugger_thread_exited));
	ERR_FAIL_COND_V_MSG(ret != OK, ret, "failed to connect required signal 'thread_exited' from debugger");

	ret = debugger.connect("thread_stack_dump", callable_mp(this, &ThreadList::_on_debugger_thread_stack_dump));
	ERR_FAIL_COND_V_MSG(ret != OK, ret, "failed to connect required signal 'thread_stack_dump' from debugger");

	ret = debugger.connect("thread_info", callable_mp(this, &ThreadList::_on_debugger_thread_info));
	ERR_FAIL_COND_V_MSG(ret != OK, ret, "failed to connect required signal 'thread_info' from debugger");

	ret = debugger.connect("clear_execution", callable_mp(this, &ThreadList::_on_debugger_clear_execution));
	ERR_FAIL_COND_V_MSG(ret != OK, ret, "failed to connect required signal 'clear_execution' from debugger");

	return OK;
}

void ThreadList::disconnect(Object &debugger) {
	debugger.disconnect("thread_breaked", callable_mp(this, &ThreadList::_on_debugger_thread_breaked));
	debugger.disconnect("thread_paused", callable_mp(this, &ThreadList::_on_debugger_thread_paused));
	debugger.disconnect("thread_alert", callable_mp(this, &ThreadList::_on_debugger_thread_alert));
	debugger.disconnect("thread_continued", callable_mp(this, &ThreadList::_on_debugger_thread_continued));
	debugger.disconnect("thread_exited", callable_mp(this, &ThreadList::_on_debugger_thread_exited));
	debugger.disconnect("thread_stack_dump", callable_mp(this, &ThreadList::_on_debugger_thread_stack_dump));
	debugger.disconnect("thread_info", callable_mp(this, &ThreadList::_on_debugger_thread_info));
	debugger.disconnect("clear_execution", callable_mp(this, &ThreadList::_on_debugger_clear_execution));
}

} // namespace editor::dbg::sd
