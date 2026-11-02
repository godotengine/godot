/**************************************************************************/
/*  hash_set.h                                                            */
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

// Minimal shim for Godot's HashSet<T> — standalone CLI builds only.
// Iterates in insertion order like Godot's HashSet.

#include <cstdint>
#include <unordered_set>
#include <vector>

template <typename T, typename H = std::hash<T>, typename E = std::equal_to<T>>
class HashSet {
	std::vector<T> _elements;
	std::unordered_set<T, H, E> _index;

public:
	void insert(const T &p_value) {
		if (_index.insert(p_value).second) {
			_elements.push_back(p_value);
		}
	}
	bool has(const T &p_value) const { return _index.count(p_value) > 0; }
	void clear() {
		_elements.clear();
		_index.clear();
	}
	int64_t size() const { return (int64_t)_elements.size(); }
	bool is_empty() const { return _elements.empty(); }

	typename std::vector<T>::const_iterator begin() const { return _elements.begin(); }
	typename std::vector<T>::const_iterator end() const { return _elements.end(); }
};
