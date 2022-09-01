/*************************************************************************/
/*  memory_tracker.cpp                                                   */
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

#include "memory_tracker.h"

#ifdef ALLOCATION_TRACKING_ENABLED

#include "core/error_macros.h"
#include "core/os/os.h"
#include "core/print_string.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

#define GODOT_ALLOCATION_TRACKING_KEY_TYPE uintptr_t
#define GODOT_ALLOCATION_TRACKING_MAP std::unordered_map<GODOT_ALLOCATION_TRACKING_KEY_TYPE, Alloc>

class MemoryTracker {
	friend class AllocationTracker;

	MemoryTracker() {}

	MemoryTracker(MemoryTracker const &);
	void operator=(MemoryTracker const &);

public:
	static MemoryTracker &get_singleton() {
		static MemoryTracker instance;
		return instance;
	}

	struct Alloc {
		const char *filename;
		void *addr;
		uint32_t line;
		uint32_t size;
	};

	struct Leak {
		// Show the largest leaks first in the list
		bool operator<(const Leak &b) const { return size > b.size; }
		const char *filename;
		int32_t size;
		uint32_t line;
		int32_t count;
	};

	struct LeakTable {
		std::vector<Leak> leaks;
	};

	std::vector<LeakTable *> _leak_tables;
	struct Data {
		void frame_flip() {
			frame_allocs_prev = frame_allocs_curr;
			frame_allocs_curr = 0;
			frame_alloc_size_prev = frame_alloc_size_curr;
			frame_alloc_size_curr = 0;
		}
		void tick_flip() {
			tick_allocs_prev = tick_allocs_curr;
			tick_allocs_curr = 0;
			tick_alloc_size_prev = tick_alloc_size_curr;
			tick_alloc_size_curr = 0;
		}
		uint32_t frame_allocs_curr = 0;
		uint32_t frame_allocs_prev = 0;
		uint32_t frame_alloc_size_curr = 0;
		uint32_t frame_alloc_size_prev = 0;
		uint32_t tick_allocs_curr = 0;
		uint32_t tick_allocs_prev = 0;
		uint32_t tick_alloc_size_curr = 0;
		uint32_t tick_alloc_size_prev = 0;
	} data;

private:
	// Only using STL here because we cannot use Godot containers, because they
	// use godot allocations internally ... chicken egg problem.
	std::mutex _mutex;
	GODOT_ALLOCATION_TRACKING_MAP _allocs;

	const Alloc *_find_alloc(void *p_address) const;
	void _fill_leak_table(LeakTable &r_lt) const;
	void _print_leak_table(LeakTable &p_lt, String p_title) const;

public:
	void add_alloc(const Alloc &p_alloc);
	void remove_alloc(void *p_address);
	void realloc(void *p_address, uint32_t p_new_size);
	void report(String p_title);

	int take_snapshot();
	void delete_snapshot(int p_snapshot_id);
	void compare_snapshots(int p_snapshot_id_a, int p_snapshot_id_b, String p_title);

	// Not called currently, to avoid order of destruction issues,
	// but included for reference
	void shutdown();
};

void AllocationTracking::add_alloc(void *p_address, uint32_t p_size, const char *p_filename, uint32_t p_line_number) {
	MemoryTracker::Alloc a;
	a.addr = p_address;
	a.filename = p_filename;
	a.line = p_line_number;
	a.size = p_size;

	MemoryTracker::get_singleton().add_alloc(a);
}

void AllocationTracking::remove_alloc(void *p_address) {
	MemoryTracker::get_singleton().remove_alloc(p_address);
}

void AllocationTracking::realloc(void *p_address, uint32_t p_new_size) {
	MemoryTracker::get_singleton().realloc(p_address, p_new_size);
}

void AllocationTracking::report(const char *p_title) {
	String title;
	if (p_title) {
		title = p_title;
	}
	MemoryTracker::get_singleton().report(title);
}

int AllocationTracking::take_snapshot() {
	return MemoryTracker::get_singleton().take_snapshot();
}

void AllocationTracking::delete_snapshot(int p_snapshot_id) {
	MemoryTracker::get_singleton().delete_snapshot(p_snapshot_id);
}

void AllocationTracking::compare_snapshots(int p_snapshot_id_a, int p_snapshot_id_b, const char *p_title) {
	String title;
	if (p_title) {
		title = p_title;
	}
	MemoryTracker::get_singleton().compare_snapshots(p_snapshot_id_a, p_snapshot_id_b, title);
}

void AllocationTracking::frame_update() {
	MemoryTracker::get_singleton().data.frame_flip();
}

void AllocationTracking::tick_update() {
	MemoryTracker::get_singleton().data.tick_flip();
}

uint32_t AllocationTracking::get_allocs_per_frame() {
	return MemoryTracker::get_singleton().data.frame_allocs_prev;
}

uint32_t AllocationTracking::get_allocs_per_tick() {
	return MemoryTracker::get_singleton().data.tick_allocs_prev;
}

uint32_t AllocationTracking::get_total_alloc_size_per_frame() {
	return MemoryTracker::get_singleton().data.frame_alloc_size_prev;
}

uint32_t AllocationTracking::get_total_alloc_size_per_tick() {
	return MemoryTracker::get_singleton().data.tick_alloc_size_prev;
}

////////////////////

const MemoryTracker::Alloc *MemoryTracker::_find_alloc(void *p_address) const {
	GODOT_ALLOCATION_TRACKING_MAP::const_iterator got = _allocs.find((GODOT_ALLOCATION_TRACKING_KEY_TYPE)p_address);
	if (got == _allocs.end()) {
		return nullptr;
	}

	return &got->second;
}

void MemoryTracker::add_alloc(const Alloc &p_alloc) {
	std::lock_guard<std::mutex> guard(_mutex);
	_allocs[(GODOT_ALLOCATION_TRACKING_KEY_TYPE)p_alloc.addr] = p_alloc;

	data.frame_allocs_curr += 1;
	data.frame_alloc_size_curr += p_alloc.size;

	data.tick_allocs_curr += 1;
	data.tick_alloc_size_curr += p_alloc.size;
}

void MemoryTracker::remove_alloc(void *p_address) {
	std::lock_guard<std::mutex> guard(_mutex);
	_allocs.erase((GODOT_ALLOCATION_TRACKING_KEY_TYPE)p_address);
}

void MemoryTracker::realloc(void *p_address, uint32_t p_new_size) {
	std::lock_guard<std::mutex> guard(_mutex);
	const Alloc *alloc = _find_alloc(p_address);
	if (alloc) {
		data.frame_allocs_curr += 1;
		data.frame_alloc_size_curr += p_new_size;

		data.tick_allocs_curr += 1;
		data.tick_alloc_size_curr += p_new_size;

		Alloc new_alloc = *alloc;
		new_alloc.size = p_new_size;
		_allocs[(GODOT_ALLOCATION_TRACKING_KEY_TYPE)p_address] = new_alloc;
	}
}

int MemoryTracker::take_snapshot() {
	int snapshot_id = _leak_tables.size();

	LeakTable *lt = new LeakTable();
	_leak_tables.push_back(lt);

	_fill_leak_table(*lt);

	return snapshot_id;
}

void MemoryTracker::delete_snapshot(int p_snapshot_id) {
	ERR_FAIL_INDEX(p_snapshot_id, _leak_tables.size());
	LeakTable *lt = _leak_tables[p_snapshot_id];
	if (lt) {
		delete lt;
		_leak_tables[p_snapshot_id] = nullptr;
	}
}

void MemoryTracker::compare_snapshots(int p_snapshot_id_a, int p_snapshot_id_b, String p_title) {
	ERR_FAIL_INDEX(p_snapshot_id_a, _leak_tables.size());
	ERR_FAIL_INDEX(p_snapshot_id_b, _leak_tables.size());

	const LeakTable *lt_a = _leak_tables[p_snapshot_id_a];
	const LeakTable *lt_b = _leak_tables[p_snapshot_id_b];

	ERR_FAIL_NULL(lt_a);
	ERR_FAIL_NULL(lt_b);

	LeakTable lt;
	for (int b = 0; b < lt_b->leaks.size(); b++) {
		Leak leak = lt_b->leaks[b];

		// find matching leak?
		for (int a = 0; a < lt_a->leaks.size(); a++) {
			const Leak &leak_a = lt_a->leaks[a];
			if ((leak.filename == leak_a.filename) && (leak.line == leak_a.line)) {
				leak.size -= leak_a.size;
				leak.count -= leak_a.count;
				break;
			}
		}

		// is the leak still valid? if so add to the merged leak table
		if ((leak.size != 0) || (leak.count != 0)) {
			lt.leaks.push_back(leak);
		}
	}

	std::sort(lt.leaks.begin(), lt.leaks.end());

	// print the merged result
	_print_leak_table(lt, p_title);
}

void MemoryTracker::_fill_leak_table(LeakTable &r_lt) const {
	for (auto entry : _allocs) {
		const Alloc &alloc = entry.second;

		// find leak?
		int found = -1;
		for (int l = 0; l < r_lt.leaks.size(); l++) {
			const Leak &leak = r_lt.leaks[l];
			if ((leak.filename == alloc.filename) && (leak.line == alloc.line)) {
				found = l;
				break;
			}
		}
		if (found == -1) {
			Leak l;
			l.filename = alloc.filename;
			l.line = alloc.line;
			l.size = alloc.size;
			l.count = 1;
			r_lt.leaks.push_back(l);
		} else {
			Leak &leak = r_lt.leaks[found];
			leak.size += alloc.size;
			leak.count += 1;
		}
	}

	std::sort(r_lt.leaks.begin(), r_lt.leaks.end());
}

void MemoryTracker::_print_leak_table(LeakTable &p_lt, String p_title) const {
	print_line("\n=============================================");
	if (p_title.length()) {
		print_line(p_title);
	}
	print_line(itos(p_lt.leaks.size()) + " allocation sources found.\n");
	int64_t total = 0;
	for (int n = 0; n < p_lt.leaks.size(); n++) {
		const Leak &l = p_lt.leaks[n];

		String st;
		int32_t s = l.size;
		total += s;
		if (s >= 0) {
			st = String::humanize_size(s);
		} else {
			st = "-" + String::humanize_size(-s);
		}

		print_line(st + " in " + itos(l.count) + " allocs in " + String(l.filename) + " : " + itos(l.line));
	}
	print_line("\nTotal " + String::humanize_size(total) + " in tracked allocations.");
	print_line("Static memory usage : " + String::humanize_size(OS::get_singleton()->get_static_memory_usage()));
	print_line("Static memory peak : " + String::humanize_size(OS::get_singleton()->get_static_memory_peak_usage()));
	print_line("Dynamic memory usage : " + String::humanize_size(OS::get_singleton()->get_dynamic_memory_usage()));
	print_line("=============================================");
}

void MemoryTracker::report(String p_title) {
	LeakTable lt;
	_fill_leak_table(lt);
	_print_leak_table(lt, p_title);
}

void MemoryTracker::shutdown() {
	for (int n = 0; n < _leak_tables.size(); n++) {
		LeakTable *lt = _leak_tables[n];
		if (lt) {
			// use delete directly, NOT godot version
			delete lt;
		}
	}
}

#endif
