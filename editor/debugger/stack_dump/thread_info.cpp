/*************************************************************************/
/*  thread_info.cpp                                                      */
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

#include "thread_info.h"

#include "scene/gui/tree.h"

namespace editor::dbg::sd {

const ThreadInfo::DebugThreadID &ThreadInfo::get_debug_thread_id() const {
	return debug_thread_id;
}

const String &ThreadInfo::get_debug_thread_id_hex() const {
	// Lazy update the hex string and cache it.
	if (debug_thread_id.size() > 0) {
		if (debug_thread_id_hex.size() < 1) {
			debug_thread_id_hex = String::hex_encode_buffer(debug_thread_id.ptr(), debug_thread_id.size());
		}
	} else {
		if (debug_thread_id_hex.size() > 0) {
			debug_thread_id_hex.clear();
		}
	}
	return debug_thread_id_hex;
}

void ThreadInfo::free_tree_item() {
	if (nullptr == tree_item) {
		return;
	}
	memdelete(tree_item);
	tree_item = nullptr;
}

ThreadInfo::ThreadInfo(int p_thread_number, const DebugThreadID &p_debug_thread_id, bool p_is_main_thread) {
	thread_number = p_thread_number;
	debug_thread_id = p_debug_thread_id;
	is_main_thread = p_is_main_thread;
}

ThreadInfo::~ThreadInfo() = default;

} // namespace editor::dbg::sd
