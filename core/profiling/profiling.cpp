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

namespace tracy {
static bool configured = false;

static const char dummy_string[] = "dummy";
static tracy::SourceLocationData dummy_source_location = tracy::SourceLocationData{ dummy_string, dummy_string, dummy_string, 0, 0 };

// Implementation similar to StringName.
struct StringInternData {
	StringName name;
	CharString name_utf8;

	uint32_t hash = 0;
	StringInternData *prev = nullptr;
	StringInternData *next = nullptr;

	StringInternData() {}
};

struct SourceLocationInternData {
	const StringInternData *file;
	const StringInternData *function;
	const StringInternData *name;

	tracy::SourceLocationData source_location_data;

	uint32_t function_ptr_hash = 0;
	SourceLocationInternData *prev = nullptr;
	SourceLocationInternData *next = nullptr;

	SourceLocationInternData() {}
};

struct TracyInternTable {
	constexpr static uint32_t TABLE_BITS = 16;
	constexpr static uint32_t TABLE_LEN = 1 << TABLE_BITS;
	constexpr static uint32_t TABLE_MASK = TABLE_LEN - 1;

	static inline BinaryMutex mutex;

	static inline SourceLocationInternData *source_location_table[TABLE_LEN];
	static inline PagedAllocator<SourceLocationInternData> source_location_allocator;

	static inline StringInternData *string_table[TABLE_LEN];
	static inline PagedAllocator<StringInternData> string_allocator;
};

const StringInternData *_intern_name(const StringName &p_name) {
	CRASH_COND(!configured);

	const uint32_t hash = p_name.hash();
	const uint32_t idx = hash & TracyInternTable::TABLE_MASK;

	StringInternData *_data = TracyInternTable::string_table[idx];

	while (_data) {
		if (_data->hash == hash) {
			return _data;
		}
		_data = _data->next;
	}

	_data = TracyInternTable::string_allocator.alloc();
	_data->name = p_name;
	_data->name_utf8 = p_name.operator String().utf8();

	_data->next = TracyInternTable::string_table[idx];
	_data->prev = nullptr;

	if (TracyInternTable::string_table[idx]) {
		TracyInternTable::string_table[idx]->prev = _data;
	}
	TracyInternTable::string_table[idx] = _data;

	return _data;
}

const tracy::SourceLocationData *intern_source_location(const void *p_function_ptr, const StringName &p_file, const StringName &p_function, const StringName &p_name, uint32_t p_line, bool p_is_script) {
	ERR_FAIL_COND_V(!configured, &dummy_source_location);

	const uint32_t hash = HashMapHasherDefault::hash(p_function_ptr);
	const uint32_t idx = hash & TracyInternTable::TABLE_MASK;

	MutexLock lock(TracyInternTable::mutex);
	SourceLocationInternData *_data = TracyInternTable::source_location_table[idx];

	while (_data) {
		if (_data->function_ptr_hash == hash && _data->source_location_data.line == p_line && _data->file->name == p_file && _data->function->name == p_function && _data->name->name == p_name) {
			return &_data->source_location_data;
		}
		_data = _data->next;
	}

	_data = TracyInternTable::source_location_allocator.alloc();

	_data->function_ptr_hash = hash;
	_data->file = _intern_name(p_file);
	_data->function = _intern_name(p_function);
	_data->name = _intern_name(p_name);

	_data->source_location_data.file = _data->file->name_utf8.get_data();
	_data->source_location_data.function = _data->function->name_utf8.get_data();
	_data->source_location_data.name = _data->name->name_utf8.get_data();

	_data->source_location_data.line = p_line;
	_data->source_location_data.color = p_is_script ? 0x478cbf : 0; // godot_logo_blue

	_data->next = TracyInternTable::source_location_table[idx];
	_data->prev = nullptr;

	if (TracyInternTable::source_location_table[idx]) {
		TracyInternTable::source_location_table[idx]->prev = _data;
	}
	TracyInternTable::source_location_table[idx] = _data;

	return &_data->source_location_data;
}
} // namespace tracy

void godot_init_profiler() {
	MutexLock lock(tracy::TracyInternTable::mutex);
	ERR_FAIL_COND(tracy::configured);

	for (uint32_t i = 0; i < tracy::TracyInternTable::TABLE_LEN; i++) {
		tracy::TracyInternTable::source_location_table[i] = nullptr;
	}
	for (uint32_t i = 0; i < tracy::TracyInternTable::TABLE_LEN; i++) {
		tracy::TracyInternTable::string_table[i] = nullptr;
	}

	tracy::configured = true;

	// Send our first event to tracy; otherwise it doesn't start collecting data.
	// FrameMark is kind of fitting because it communicates "this is where we started tracing".
	FrameMark;
}

void godot_cleanup_profiler() {
	MutexLock lock(tracy::TracyInternTable::mutex);
	ERR_FAIL_COND(!tracy::configured);

	for (uint32_t i = 0; i < tracy::TracyInternTable::TABLE_LEN; i++) {
		while (tracy::TracyInternTable::source_location_table[i]) {
			tracy::SourceLocationInternData *d = tracy::TracyInternTable::source_location_table[i];
			tracy::TracyInternTable::source_location_table[i] = tracy::TracyInternTable::source_location_table[i]->next;
			tracy::TracyInternTable::source_location_allocator.free(d);
		}
	}
	for (uint32_t i = 0; i < tracy::TracyInternTable::TABLE_LEN; i++) {
		while (tracy::TracyInternTable::string_table[i]) {
			tracy::StringInternData *d = tracy::TracyInternTable::string_table[i];
			tracy::TracyInternTable::string_table[i] = tracy::TracyInternTable::string_table[i]->next;
			tracy::TracyInternTable::string_allocator.free(d);
		}
	}

	tracy::configured = false;
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

#elif defined(GODOT_USE_INSTRUMENTS)

namespace apple::instruments {

os_log_t LOG;
os_log_t LOG_TRACING;

} // namespace apple::instruments

void godot_init_profiler() {
	static bool initialized = false;
	if (initialized) {
		return;
	}
	initialized = true;
	apple::instruments::LOG = os_log_create("org.godotengine.godot", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
#ifdef INSTRUMENTS_SAMPLE_CALLSTACKS
	apple::instruments::LOG_TRACING = os_log_create("org.godotengine.godot", OS_LOG_CATEGORY_DYNAMIC_STACK_TRACING);
#else
	apple::instruments::LOG_TRACING = os_log_create("org.godotengine.godot", "tracing");
#endif
}

void godot_cleanup_profiler() {
}

#else
void godot_init_profiler() {
	// Stub
}

void godot_cleanup_profiler() {
	// Stub
}
#endif
