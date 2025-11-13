/**************************************************************************/
/*  profiling.cpp                                                         */
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

#include "profiling.h"

#if defined(GODOT_USE_TRACY)
// Use the tracy profiler.

#include "core/os/mutex.h"
#include "core/templates/paged_allocator.h"

namespace Internal {

static bool configured = false;
static const char *empty_string = "<empty>";

// UTF-8 version of StringName.
struct TracyInternData {
	String src;
	CharString utf8;

	uint32_t hash = 0;
	TracyInternData *prev = nullptr;
	TracyInternData *next = nullptr;
	TracyInternData() {}
};

struct TracyInternTable {
	constexpr static uint32_t TABLE_BITS = 16;
	constexpr static uint32_t TABLE_LEN = 1 << TABLE_BITS;
	constexpr static uint32_t TABLE_MASK = TABLE_LEN - 1;

	static inline TracyInternData *table[TABLE_LEN];
	static inline BinaryMutex mutex;
	static inline PagedAllocator<TracyInternData> allocator;
};

const char *intern_string(const String &p_name) {
	TracyInternData *_data = nullptr;

	ERR_FAIL_COND_V(!configured, empty_string);

	if (p_name.is_empty()) {
		return empty_string;
	}

	const uint32_t hash = p_name.hash();
	const uint32_t idx = hash & TracyInternTable::TABLE_MASK;

	MutexLock lock(TracyInternTable::mutex);
	_data = TracyInternTable::table[idx];

	while (_data) {
		if (_data->hash == hash && _data->src == p_name) {
			break;
		}
		_data = _data->next;
	}
	if (_data) {
		return _data->utf8.get_data();
	}

	_data = TracyInternTable::allocator.alloc();
	_data->src = p_name;
	_data->utf8 = p_name.utf8(); // Make a new CharString
	_data->hash = hash;
	_data->next = TracyInternTable::table[idx];
	_data->prev = nullptr;

	if (TracyInternTable::table[idx]) {
		TracyInternTable::table[idx]->prev = _data;
	}
	TracyInternTable::table[idx] = _data;

	return _data->utf8.get_data();
}

const char *intern_string(const CharString &p_name) {
	TracyInternData *_data = nullptr;

	ERR_FAIL_COND_V(!configured, empty_string);

	if (p_name.is_empty()) {
		return empty_string;
	}

	const uint32_t hash = p_name.hash();
	const uint32_t idx = hash & TracyInternTable::TABLE_MASK;

	MutexLock lock(TracyInternTable::mutex);
	_data = TracyInternTable::table[idx];

	while (_data) {
		if (_data->hash == hash && _data->utf8 == p_name) {
			break;
		}
		_data = _data->next;
	}
	if (_data) {
		return _data->utf8.get_data();
	}

	_data = TracyInternTable::allocator.alloc();
	_data->utf8 = p_name; // Copy the CharString
	_data->hash = hash;
	_data->next = TracyInternTable::table[idx];
	_data->prev = nullptr;

	if (TracyInternTable::table[idx]) {
		TracyInternTable::table[idx]->prev = _data;
	}
	TracyInternTable::table[idx] = _data;

	return _data->utf8.get_data();
}
} //namespace Internal

void godot_init_profiler() {
	ERR_FAIL_COND(Internal::configured);
	for (uint32_t i = 0; i < Internal::TracyInternTable::TABLE_LEN; i++) {
		Internal::TracyInternTable::table[i] = nullptr;
	}
	Internal::configured = true;

	// Send our first event to tracy; otherwise it doesn't start collecting data.
	// FrameMark is kind of fitting because it communicates "this is where we started tracing".
	FrameMark;
}

void godot_cleanup_profiler() {
	MutexLock lock(Internal::TracyInternTable::mutex);

	for (uint32_t i = 0; i < Internal::TracyInternTable::TABLE_LEN; i++) {
		while (Internal::TracyInternTable::table[i]) {
			Internal::TracyInternData *d = Internal::TracyInternTable::table[i];
			Internal::TracyInternTable::table[i] = Internal::TracyInternTable::table[i]->next;
			Internal::TracyInternTable::allocator.free(d);
		}
	}
	Internal::configured = false;
}

#elif defined(GODOT_USE_PERFETTO)
PERFETTO_TRACK_EVENT_STATIC_STORAGE();

void godot_init_profiler() {
	perfetto::TracingInitArgs args;

	args.backends |= perfetto::kSystemBackend;

	perfetto::Tracing::Initialize(args);
	perfetto::TrackEvent::Register();
}

void godot_cleanup_profiler() {
	// Stub
}

#else
void godot_init_profiler() {
	// Stub
}

void godot_cleanup_profiler() {
	// Stub
}
#endif
