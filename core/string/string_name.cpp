/**************************************************************************/
/*  string_name.cpp                                                       */
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

#include "string_name.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

struct StringName::Table {
	constexpr static uint32_t TABLE_BITS = 16;
	constexpr static uint32_t TABLE_LEN = 1 << TABLE_BITS;
	constexpr static uint32_t TABLE_MASK = TABLE_LEN - 1;

	struct Entry {
		uint32_t hash;
		_Data *data;
	};

	static inline LocalVector<Entry> table[TABLE_LEN];
	static inline Mutex mutex;
	static inline PagedAllocator<_Data> allocator;
};

void StringName::setup() {
	ERR_FAIL_COND(configured);
	configured = true;
}

void StringName::cleanup() {
	MutexLock lock(Table::mutex);

#ifdef DEBUG_ENABLED
	if (unlikely(debug_stringname)) {
		LocalVector<_Data *> data;
		for (int i = 0; i < Table::TABLE_LEN; i++) {
			for (const Table::Entry &entry : Table::table[i]) {
				data.push_back(entry.data);
			}
		}

		print_line("\nStringName reference ranking (from most to least referenced):\n");

		data.sort_custom<DebugSortReferences>();
		int unreferenced_stringnames = 0;
		int rarely_referenced_stringnames = 0;
		for (uint32_t i = 0; i < data.size(); i++) {
			print_line(itos(i + 1) + ": " + data[i]->name + " - " + itos(data[i]->debug_references));
			if (data[i]->debug_references == 0) {
				unreferenced_stringnames += 1;
			} else if (data[i]->debug_references < 5) {
				rarely_referenced_stringnames += 1;
			}
		}

		print_line(vformat("\nOut of %d StringNames, %d StringNames were never referenced during this run (0 times) (%.2f%%).", data.size(), unreferenced_stringnames, unreferenced_stringnames / float(data.size()) * 100));
		print_line(vformat("Out of %d StringNames, %d StringNames were rarely referenced during this run (1-4 times) (%.2f%%).", data.size(), rarely_referenced_stringnames, rarely_referenced_stringnames / float(data.size()) * 100));
	}
#endif
	int lost_strings = 0;
	for (uint32_t i = 0; i < Table::TABLE_LEN; i++) {
		for (const Table::Entry &entry : Table::table[i]) {
			_Data *d = entry.data;
			if (d->static_count.get() != d->refcount.get()) {
				lost_strings++;

				if (OS::get_singleton()->is_stdout_verbose()) {
					print_line(vformat("Orphan StringName: %s (static: %d, total: %d)", d->name, d->static_count.get(), d->refcount.get()));
				}
			}

			Table::allocator.free(d);
		}
		Table::table[i].reset();
	}
	if (lost_strings) {
		print_verbose(vformat("StringName: %d unclaimed string names at exit.", lost_strings));
	}
	configured = false;
}

void StringName::unref() {
	ERR_FAIL_COND(!configured);

	if (_data && _data->refcount.unref()) {
		MutexLock lock(Table::mutex);

		if (CoreGlobals::leak_reporting_enabled && _data->static_count.get() > 0) {
			ERR_PRINT("BUG: Unreferenced static string to 0: " + _data->name);
		}

		const uint32_t bucket_index = _data->hash & Table::TABLE_MASK;
		LocalVector<Table::Entry> &vector = Table::table[bucket_index];
		vector.remove_at_unordered(_data->index_in_bucket);
		if (vector.size() > _data->index_in_bucket) {
			// Wasn't the last entry, we need to fix the index of the entry we just moved.
			vector[_data->index_in_bucket].data->index_in_bucket = _data->index_in_bucket;
		}

		Table::allocator.free(_data);
	}

	_data = nullptr;
}

uint32_t StringName::get_empty_hash() {
	static uint32_t empty_hash = String::hash("");
	return empty_hash;
}

bool StringName::operator==(const String &p_name) const {
	if (_data) {
		return _data->name == p_name;
	}

	return p_name.is_empty();
}

bool StringName::operator==(const char *p_name) const {
	if (_data) {
		return _data->name == p_name;
	}

	return p_name[0] == 0;
}

bool StringName::operator!=(const String &p_name) const {
	return !(operator==(p_name));
}

bool StringName::operator!=(const char *p_name) const {
	return !(operator==(p_name));
}

char32_t StringName::operator[](int p_index) const {
	if (_data) {
		return _data->name[p_index];
	}

	CRASH_BAD_INDEX(p_index, 0);
	return 0;
}

int StringName::length() const {
	if (_data) {
		return _data->name.length();
	}

	return 0;
}

bool StringName::is_empty() const {
	if (_data) {
		return _data->name.is_empty();
	}

	return true;
}

StringName &StringName::operator=(const StringName &p_name) {
	if (this == &p_name) {
		return *this;
	}

	unref();

	if (p_name._data && p_name._data->refcount.ref()) {
		_data = p_name._data;
	}

	return *this;
}

StringName::StringName(const StringName &p_name) {
	_data = nullptr;

	ERR_FAIL_COND(!configured);

	if (p_name._data && p_name._data->refcount.ref()) {
		_data = p_name._data;
	}
}

void StringName::assign_static_unique_class_name(StringName *ptr, const char *p_name) {
	MutexLock lock(Table::mutex);
	if (*ptr == StringName()) {
		*ptr = StringName(p_name, true);
	}
}

StringName::StringName(const char *p_name, bool p_static) {
	_data = nullptr;

	ERR_FAIL_COND(!configured);

	if (!p_name || p_name[0] == 0) {
		return; //empty, ignore
	}

	const uint32_t hash = String::hash(p_name);
	const uint32_t idx = hash & Table::TABLE_MASK;

	LocalVector<Table::Entry> &entries = Table::table[idx];
	MutexLock lock(Table::mutex);

	for (const Table::Entry &entry : entries) {
		if (entry.hash == hash && entry.data->name == p_name) {
			if (!entry.data->refcount.ref()) {
				// Entry was destructed just as we were trying to access it, so we need to make a new entry.
				break;
			}

			_data = entry.data;

			if (p_static) {
				_data->static_count.increment();
			}
#ifdef DEBUG_ENABLED
			if (unlikely(debug_stringname)) {
				_data->debug_references++;
			}
#endif
			return;
		}
	}

	_data = Table::allocator.alloc();
	_data->name = p_name;
	_data->refcount.init();
	_data->static_count.set(p_static ? 1 : 0);
	_data->hash = hash;
	_data->index_in_bucket = entries.size();

	entries.push_back(Table::Entry{ hash, _data });

#ifdef DEBUG_ENABLED
	if (unlikely(debug_stringname)) {
		// Keep in memory, force static.
		_data->refcount.ref();
		_data->static_count.increment();
	}
#endif
}

StringName::StringName(const String &p_name, bool p_static) {
	_data = nullptr;

	ERR_FAIL_COND(!configured);

	if (p_name.is_empty()) {
		return;
	}

	const uint32_t hash = p_name.hash();
	const uint32_t idx = hash & Table::TABLE_MASK;

	LocalVector<Table::Entry> &entries = Table::table[idx];
	MutexLock lock(Table::mutex);

	for (const Table::Entry &entry : entries) {
		if (entry.hash == hash && entry.data->name == p_name) {
			if (!entry.data->refcount.ref()) {
				// Entry was destructed just as we were trying to access it, so we need to make a new entry.
				break;
			}

			_data = entry.data;

			if (p_static) {
				_data->static_count.increment();
			}
#ifdef DEBUG_ENABLED
			if (unlikely(debug_stringname)) {
				_data->debug_references++;
			}
#endif
			return;
		}
	}

	_data = Table::allocator.alloc();
	_data->name = p_name;
	_data->refcount.init();
	_data->static_count.set(p_static ? 1 : 0);
	_data->hash = hash;
	_data->index_in_bucket = entries.size();

	entries.push_back(Table::Entry{ hash, _data });

#ifdef DEBUG_ENABLED
	if (unlikely(debug_stringname)) {
		// Keep in memory, force static.
		_data->refcount.ref();
		_data->static_count.increment();
	}
#endif
}

bool operator==(const String &p_name, const StringName &p_string_name) {
	return p_string_name.operator==(p_name);
}
bool operator!=(const String &p_name, const StringName &p_string_name) {
	return p_string_name.operator!=(p_name);
}

bool operator==(const char *p_name, const StringName &p_string_name) {
	return p_string_name.operator==(p_name);
}
bool operator!=(const char *p_name, const StringName &p_string_name) {
	return p_string_name.operator!=(p_name);
}
