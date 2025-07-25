/**************************************************************************/
/*  lru.h                                                                 */
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

#pragma once

#include "hash_map.h"

template <typename TKey, typename TData, typename Hasher = HashMapHasherDefault, typename Comparator = HashMapComparatorDefault<TKey>, void (*BeforeEvict)(const TKey &, TData &) = nullptr>
class LRUCache {
public:
	using Iterator = typename HashMap<TKey, TData, Hasher, Comparator>::Iterator;

private:
	HashMap<TKey, TData, Hasher, Comparator> _map;
	size_t capacity;

public:
	const Iterator insert(const TKey &p_key, const TData &p_value) {
		Iterator old_entry = _map.find(p_key);

		if (old_entry) {
			GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Waddress")
			if constexpr (BeforeEvict != nullptr) {
				BeforeEvict(old_entry->key, old_entry->value);
			}
			GODOT_GCC_WARNING_POP
			_map.erase(p_key);
		}
		Iterator iter = _map.insert(p_key, p_value);

		while (_map.size() > capacity) {
			Iterator first = _map.begin();
			GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Waddress")
			if constexpr (BeforeEvict != nullptr) {
				BeforeEvict(first->key, first->value);
			}
			GODOT_GCC_WARNING_POP
			_map.erase(first->key);
		}

		return iter;
	}

	void clear() { _map.clear(); }
	bool has(const TKey &p_key) const { return _map.has(p_key); }
	bool erase(const TKey &p_key) { return _map.erase(p_key); }

	const TData &get(const TKey &p_key) {
		TData *value = _map.renew_key(p_key);
		CRASH_COND(!value);
		return *value;
	}

	const TData *getptr(const TKey &p_key) { return _map.renew_key(p_key); }

	_FORCE_INLINE_ size_t get_capacity() const { return capacity; }
	_FORCE_INLINE_ size_t get_size() const { return _map.size(); }

	void set_capacity(size_t p_capacity) {
		if (capacity > 0) {
			capacity = p_capacity;
			while (_map.size() > capacity) {
				Iterator first = _map.begin();
				GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Waddress")
				if constexpr (BeforeEvict != nullptr) {
					BeforeEvict(first->key, first->value);
				}
				GODOT_GCC_WARNING_POP
				_map.erase(first->key);
			}
		}
	}

	LRUCache() {
		capacity = 64;
	}

	LRUCache(int p_capacity) {
		capacity = p_capacity;
	}
};
