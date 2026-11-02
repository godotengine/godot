/**************************************************************************/
/*  hash_map.h                                                            */
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

// Minimal shim for Godot's HashMap<K, V> — standalone CLI builds only.
// Iterates in insertion order like Godot's HashMap, so passes emit identical SPIR-V in the CLI and engine.

#include <cstdint>
#include <deque>
#include <unordered_map>

template <typename K, typename V>
struct KeyValue {
	const K key;
	V value;
};

template <typename K, typename V, typename H = std::hash<K>, typename E = std::equal_to<K>>
class HashMap {
	std::deque<KeyValue<K, V>> _elements; // Stable addresses for getptr().
	std::unordered_map<K, V *, H, E> _index;

public:
	V &operator[](const K &p_key) {
		V *value = getptr(p_key);
		return value ? *value : insert(p_key, V());
	}
	const V &operator[](const K &p_key) const { return *_index.at(p_key); }

	bool has(const K &p_key) const { return _index.count(p_key) > 0; }

	V *getptr(const K &p_key) {
		auto it = _index.find(p_key);
		return it != _index.end() ? it->second : nullptr;
	}
	const V *getptr(const K &p_key) const {
		auto it = _index.find(p_key);
		return it != _index.end() ? it->second : nullptr;
	}

	V &insert(const K &p_key, const V &p_value) {
		V *existing = getptr(p_key);
		if (existing) {
			return *existing = p_value;
		}
		_elements.push_back({ p_key, p_value });
		return *(_index[p_key] = &_elements.back().value);
	}
	void clear() {
		_elements.clear();
		_index.clear();
	}
	int64_t size() const { return (int64_t)_elements.size(); }
	bool is_empty() const { return _elements.empty(); }

	typename std::deque<KeyValue<K, V>>::iterator begin() { return _elements.begin(); }
	typename std::deque<KeyValue<K, V>>::iterator end() { return _elements.end(); }
	typename std::deque<KeyValue<K, V>>::const_iterator begin() const { return _elements.begin(); }
	typename std::deque<KeyValue<K, V>>::const_iterator end() const { return _elements.end(); }
};
