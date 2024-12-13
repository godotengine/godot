/**************************************************************************/
/*  pair.h                                                                */
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

#ifndef PAIR_H
#define PAIR_H

#include "core/templates/hashfuncs.h"
#include "core/typedefs.h"
template <typename F, typename S>
struct Pair {
	F first;
	S second;

	Pair() :
			first(),
			second() {
	}

	Pair(F p_first, const S &p_second) :
			first(p_first),
			second(p_second) {
	}

	bool operator==(const Pair &other) const {
		return (first == other.first) && (second == other.second);
	}
	INEQUALITY_OPERATOR(const Pair &)
};

template <typename F, typename S>
struct PairSort {
	bool operator()(const Pair<F, S> &A, const Pair<F, S> &B) const {
		if (A.first != B.first) {
			return A.first < B.first;
		}
		return A.second < B.second;
	}
};

template <typename F, typename S>
struct PairHash {
	static uint32_t hash(const Pair<F, S> &P) {
		uint64_t h1 = HashMapHasherDefault::hash(P.first);
		uint64_t h2 = HashMapHasherDefault::hash(P.second);
		return hash_one_uint64((h1 << 32) | h2);
	}
};

template <typename K, typename V>
struct KeyValue {
	const K key;
	V value;

	void operator=(const KeyValue &p_kv) = delete;
	_FORCE_INLINE_ KeyValue(const KeyValue &p_kv) :
			key(p_kv.key),
			value(p_kv.value) {
	}
	_FORCE_INLINE_ KeyValue(const K &p_key, const V &p_value) :
			key(p_key),
			value(p_value) {
	}

	bool operator==(const KeyValue &other) const {
		return (key == other.key) && (value == other.value);
	}
	INEQUALITY_OPERATOR(const KeyValue &)
};

template <typename K, typename V>
struct KeyValueSort {
	bool operator()(const KeyValue<K, V> &A, const KeyValue<K, V> &B) const {
		return A.key < B.key;
	}
};

#endif // PAIR_H
