/*************************************************************************/
/*  thread_info.h                                                        */
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

#ifndef THREAD_INFO_H
#define THREAD_INFO_H

#include "core/object/ref_counted.h"
#include "core/variant/typed_array.h"

class TreeItem;

/**
 * \brief Godot Editor / IDE
 */
namespace editor {
/**
 * \brief Editor / IDE side of Debugger Functionality
 */
namespace dbg {
/**
 * \brief Stack Dump Debugger Functionality
 */
namespace sd {

// Received information about a thread.  Used like a struct with public data, but
// declared as class to get ref counting.
class ThreadInfo : public RefCounted {
	GDCLASS(ThreadInfo, RefCounted)

public:
	using DebugThreadID = PackedByteArray;

private:
	DebugThreadID debug_thread_id;
	mutable String debug_thread_id_hex;

public:
	enum class Status : int {
		BLOCKED = 0,
		RUNNING,
		PAUSED,
		ALERT,
		BREAKPOINT,
		CRASHED,
		DEAD,
		NUM_VALUES
	};

	// Supported column IDs for sorting the data or for display, not the same as currently visible column indices.
	// WARNING: Also used as integer indices for array access, so must be consecutive.
	enum class Field : int {
		STATUS = 0,
		ID,
		NAME,
		STACK,
		LANGUAGE,
		CATEGORY,
		DEBUG_ID,
		NUM_VALUES
	};

	struct Hasher {
		static uint32_t hash(const PackedByteArray &p_buffer) {
			// This is the same hasher selection as Variant uses for PackedByteArray.
			if (likely(p_buffer.size() > 0)) {
				return hash_murmur3_buffer(p_buffer.ptr(), p_buffer.size());
			}
			return hash_murmur3_one_64(0);
		}
	};

	// Link to view, could be separated to appropriate class, but would require another lookup.
	TreeItem *tree_item = nullptr;

	// Aggregated data
	int thread_number;
	const DebugThreadID &get_debug_thread_id() const;
	const String &get_debug_thread_id_hex() const;
	bool is_main_thread = false;
	String reason;
	Status status = Status::RUNNING;
	int severity_code = 0;
	bool can_debug = false;
	bool has_stack_dump = false;
	String language;
	PackedByteArray thread_tag;
	String thread_name;
	TypedArray<Dictionary> stack_dump_info;

	void free_tree_item();

	ThreadInfo(int p_thread_number, const DebugThreadID &p_debug_thread_id, bool p_is_main_thread);
	~ThreadInfo();
};

} // namespace sd
} // namespace dbg
} // namespace editor

#endif // THREAD_INFO_H
