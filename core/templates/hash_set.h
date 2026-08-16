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

#include "core/templates/a_hash_map.h"

/**
 * Set container using robin hood hashing.
 *
 * Elements are not pointer stable.
 * The element order is arbitrary.
 *
 * Core container guidance:
 * https://docs.godotengine.org/en/latest/engine_details/architecture/core_types.html#containers
 */
template <typename TKey,
		typename Hasher = HashMapHasherDefault,
		typename Comparator = HashMapComparatorDefault<TKey>>
class _WARN_UNUSED_ HashSet {
	using InnerTable = AHashMap<TKey, EmptyValue, Hasher, Comparator>;
	InnerTable _inner;

public:
	_FORCE_INLINE_ uint32_t get_capacity() const { return _inner.get_capacity(); }
	_FORCE_INLINE_ uint32_t size() const { return _inner.size(); }

	/* Standard Godot Container API */

	bool is_empty() const {
		return _inner.is_empty();
	}

	void clear() {
		_inner.clear();
	}

	_FORCE_INLINE_ bool has(const TKey &p_key) const {
		return _inner.has(p_key);
	}

	bool erase(const TKey &p_key) {
		return _inner.erase(p_key);
	}

	// Reserves space for a number of elements, useful to avoid many resizes and rehashes.
	// If adding a known (possibly large) number of elements at once, must be larger than old capacity.
	void reserve(uint32_t p_new_capacity) {
		_inner.reserve(p_new_capacity);
	}

	/** Iterator API **/

	struct Iterator {
		_FORCE_INLINE_ const TKey &operator*() const {
			return _inner.operator*().key;
		}
		_FORCE_INLINE_ const TKey *operator->() const {
			return &_inner.operator->()->key;
		}
		_FORCE_INLINE_ Iterator &operator++() {
			++_inner;
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			--_inner;
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &p_other) const { return _inner == p_other._inner; }
		_FORCE_INLINE_ bool operator!=(const Iterator &p_other) const { return _inner != p_other._inner; }

		_FORCE_INLINE_ explicit operator bool() const {
			return _inner.operator bool();
		}

		_FORCE_INLINE_ void operator=(const Iterator &p_it) {
			_inner = p_it._inner;
		}
		_FORCE_INLINE_ Iterator(typename InnerTable::ConstIterator p_inner) {
			_inner = p_inner;
		}

	private:
		typename InnerTable::ConstIterator _inner;

	};

	_FORCE_INLINE_ Iterator begin() const _LIFETIME_BOUND_ {
		return Iterator(_inner.begin());
	}
	_FORCE_INLINE_ Iterator end() const _LIFETIME_BOUND_ {
		return Iterator(_inner.end());
	}
	_FORCE_INLINE_ Iterator last() const _LIFETIME_BOUND_ {
		return Iterator(_inner.last());
	}

	_FORCE_INLINE_ Iterator find(const TKey &p_key) const _LIFETIME_BOUND_ {
		return Iterator(_inner.find(p_key));
	}

	_FORCE_INLINE_ void remove(const Iterator &p_iter) {
		if (p_iter) {
			erase(*p_iter);
		}
	}

	/* Insert */

	Iterator insert(const TKey &p_key) _LIFETIME_BOUND_ {
		return Iterator(_inner.insert(p_key, EmptyValue {}));
	}

	/* Constructors */

	explicit HashSet(const HashSet &p_other) {
		_inner = p_other._inner;
	}

	HashSet(HashSet &&p_other) {
		_inner = p_other._inner;
	}

	void operator=(const HashSet &p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}

		_inner = p_other._inner;
	}

	void operator=(HashSet &&p_other) {
		if (this == &p_other) {
			return; // Ignore self assignment.
		}

		_inner = p_other._inner;
	}

	bool operator==(const HashSet &p_other) const {
		if (size() != p_other.size()) {
			return false;
		}
		for (const TKey &key : p_other) {
			if (!has(key)) {
				return false;
			}
		}
		return true;
	}
	bool operator!=(const HashSet &p_other) const {
		return !(*this == p_other);
	}

	HashSet(uint32_t p_initial_capacity) {
		// Capacity can't be 0.
		_inner = p_initial_capacity;
	}
	HashSet() {
		_inner = {};
	}
	HashSet(std::initializer_list<TKey> p_init) {
		_inner = p_init.size();
		for (const TKey &E : p_init) {
			insert(E);
		}
	}

	void reset() {
		_inner.reset();
	}
};
