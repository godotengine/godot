/**************************************************************************/
/*  char_intern.cpp                                                       */
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

#include "char_intern.h"

#include "core/os/mutex.h"
#include "core/os/os.h"
#include "core/templates/hashfuncs.h"

static const char *empty = "<empty>";

struct CharIntern::Table {
	constexpr static uint32_t TABLE_BITS = 16;
	constexpr static uint32_t TABLE_LEN = 1 << TABLE_BITS;
	constexpr static uint32_t TABLE_MASK = TABLE_LEN - 1;

	static inline _Data *table[TABLE_LEN];
	static inline BinaryMutex mutex;
	static inline PagedAllocator<_Data> allocator;
};

void CharIntern::setup() {
	ERR_FAIL_COND(configured);
	for (uint32_t i = 0; i < Table::TABLE_LEN; i++) {
		Table::table[i] = nullptr;
	}
	configured = true;
}

const char *CharIntern::intern(const String &p_name) {
	_Data *_data = nullptr;

	ERR_FAIL_COND_V(!configured, empty);

	if (p_name.is_empty()) {
		return empty;
	}

	const uint32_t hash = p_name.hash();
	const uint32_t idx = hash & Table::TABLE_MASK;

	MutexLock lock(Table::mutex);
	_data = Table::table[idx];

	while (_data) {
		if (_data->hash == hash && _data->src == p_name) {
			break;
		}
		_data = _data->next;
	}
	if (_data) {
		return _data->utf8.get_data();
	}

	_data = Table::allocator.alloc();
	_data->src = p_name;
	_data->utf8 = p_name.utf8(); // Make a new CharString
	_data->hash = hash;
	_data->next = Table::table[idx];
	_data->prev = nullptr;

	if (Table::table[idx]) {
		Table::table[idx]->prev = _data;
	}
	Table::table[idx] = _data;

	return _data->utf8.get_data();
}

const char *CharIntern::intern(const CharString &p_name) {
	_Data *_data = nullptr;

	ERR_FAIL_COND_V(!configured, empty);

	if (p_name.is_empty()) {
		return empty;
	}

	const uint32_t hash = p_name.hash();
	const uint32_t idx = hash & Table::TABLE_MASK;

	MutexLock lock(Table::mutex);
	_data = Table::table[idx];

	while (_data) {
		if (_data->hash == hash && _data->utf8 == p_name) {
			break;
		}
		_data = _data->next;
	}
	if (_data) {
		return _data->utf8.get_data();
	}

	_data = Table::allocator.alloc();
	// _data->src = default constructed variant, perhaps too large?;
	// I could make it the same variant, but its a value, ...
	_data->utf8 = p_name; // Make a new CharString
	_data->hash = hash;
	_data->next = Table::table[idx];
	_data->prev = nullptr;

	if (Table::table[idx]) {
		Table::table[idx]->prev = _data;
	}
	Table::table[idx] = _data;

	return _data->utf8.get_data();
}
