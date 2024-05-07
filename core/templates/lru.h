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

#ifndef LRU_H
#define LRU_H

#include "core/math/math_funcs.h"
#include "core/templates/hash_map.h"
#include "core/templates/list.h"
#include "core/templates/pair.h"

template <typename TKey, typename TValue, typename Hasher = HashMapHasherDefault, typename Comparator = HashMapComparatorDefault<TKey>>
class LRUCache {
private:
	typedef typename List<Pair<TKey, TValue>>::Element *Element;

	List<Pair<TKey, TValue>> _list;
	HashMap<TKey, Element, Hasher, Comparator> _map;
	size_t capacity;

public:
	struct ConstIterator {
		_FORCE_INLINE_ const Pair<TKey, TValue> &operator*() const {
			return E->get();
		}
		_FORCE_INLINE_ const Pair<TKey, TValue> *operator->() const { return &E->get(); }
		_FORCE_INLINE_ ConstIterator &operator++() {
			E = E->next();
			return *this;
		}
		_FORCE_INLINE_ ConstIterator &operator--() {
			E = E->prev();
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const ConstIterator &b) const { return E == b.E; }
		_FORCE_INLINE_ bool operator!=(const ConstIterator &b) const { return E != b.E; }

		_FORCE_INLINE_ explicit operator bool() const {
			return E != nullptr;
		}

		_FORCE_INLINE_ ConstIterator(const Element p_E) { E = p_E; }
		_FORCE_INLINE_ ConstIterator() {}
		_FORCE_INLINE_ ConstIterator(const ConstIterator &p_it) { E = p_it.E; }
		_FORCE_INLINE_ void operator=(const ConstIterator &p_it) {
			E = p_it.E;
		}

	private:
		const Element E = nullptr;
	};

	struct Iterator {
		_FORCE_INLINE_ Pair<TKey, TValue> &operator*() const {
			return E->get();
		}
		_FORCE_INLINE_ Pair<TKey, TValue> *operator->() const { return &E->get(); }
		_FORCE_INLINE_ Iterator &operator++() {
			E = E->next();
			return *this;
		}
		_FORCE_INLINE_ Iterator &operator--() {
			E = E->prev();
			return *this;
		}

		_FORCE_INLINE_ bool operator==(const Iterator &b) const { return E == b.E; }
		_FORCE_INLINE_ bool operator!=(const Iterator &b) const { return E != b.E; }

		_FORCE_INLINE_ explicit operator bool() const {
			return E != nullptr;
		}

		_FORCE_INLINE_ Iterator(Element p_E) { E = p_E; }
		_FORCE_INLINE_ Iterator() {}
		_FORCE_INLINE_ Iterator(const Iterator &p_it) { E = p_it.E; }
		_FORCE_INLINE_ void operator=(const Iterator &p_it) {
			E = p_it.E;
		}

		operator ConstIterator() const {
			return ConstIterator(E);
		}

	private:
		Element E = nullptr;
	};

	_FORCE_INLINE_ Iterator begin() {
		return Iterator(_list.front());
	}

	_FORCE_INLINE_ Iterator end() {
		return Iterator(nullptr);
	}

	_FORCE_INLINE_ ConstIterator begin() const {
		return ConstIterator(_list.front());
	}

	_FORCE_INLINE_ ConstIterator end() const {
		return ConstIterator(nullptr);
	}

	const TValue *insert(const TKey &p_key, const TValue &p_value) {
		Element *e = _map.getptr(p_key);
		Element n = _list.push_front(Pair<TKey, TValue>(p_key, p_value));

		if (e) {
			_list.erase(*e);
			_map.erase(p_key);
		}
		_map[p_key] = _list.front();

		while (_map.size() > capacity) {
			Element d = _list.back();
			_map.erase(d->get().first);
			_list.pop_back();
		}

		return &n->get().second;
	}

	void clear() {
		_map.clear();
		_list.clear();
	}

	bool has(const TKey &p_key) const {
		return _map.getptr(p_key);
	}

	const TValue &get(const TKey &p_key) {
		Element *e = _map.getptr(p_key);
		CRASH_COND(!e);
		_list.move_to_front(*e);
		return (*e)->get().second;
	};

	const TValue *getptr(const TKey &p_key) {
		Element *e = _map.getptr(p_key);
		if (!e) {
			return nullptr;
		} else {
			_list.move_to_front(*e);
			return &(*e)->get().second;
		}
	}

	_FORCE_INLINE_ size_t get_capacity() const { return capacity; }
	_FORCE_INLINE_ size_t get_size() const { return _map.size(); }

	void set_capacity(size_t p_capacity) {
		if (capacity > 0) {
			capacity = p_capacity;
			while (_map.size() > capacity) {
				Element d = _list.back();
				_map.erase(d->get().first);
				_list.pop_back();
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

#endif // LRU_H
