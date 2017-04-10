/*************************************************************************/
/*  vmap.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef VMAP_H
#define VMAP_H

#include "typedefs.h"
#include "vector.h"

template <class T, class V>
class VMap {

	struct _Pair {

		T key;
		V value;

		_FORCE_INLINE_ _Pair() {}

		_FORCE_INLINE_ _Pair(const T &p_key, const V &p_value) {

			key = p_key;
			value = p_value;
		}
	};

	Vector<_Pair> _data;

	_FORCE_INLINE_ int _find(const T &p_val, bool &r_exact) const {

		r_exact = false;
		if (_data.empty())
			return 0;

		int low = 0;
		int high = _data.size() - 1;
		int middle;
		const _Pair *a = &_data[0];

		while (low <= high) {
			middle = (low + high) / 2;

			if (p_val < a[middle].key) {
				high = middle - 1; //search low end of array
			} else if (a[middle].key < p_val) {
				low = middle + 1; //search high end of array
			} else {
				r_exact = true;
				return middle;
			}
		}

		//return the position where this would be inserted
		if (a[middle].key < p_val)
			middle++;
		return middle;
	}

	_FORCE_INLINE_ int _find_exact(const T &p_val) const {

		if (_data.empty())
			return -1;

		int low = 0;
		int high = _data.size() - 1;
		int middle;
		const _Pair *a = &_data[0];

		while (low <= high) {
			middle = (low + high) / 2;

			if (p_val < a[middle].key) {
				high = middle - 1; //search low end of array
			} else if (a[middle].key < p_val) {
				low = middle + 1; //search high end of array
			} else {
				return middle;
			}
		}

		return -1;
	}

public:
	int insert(const T &p_key, const V &p_val) {

		bool exact;
		int pos = _find(p_key, exact);
		if (exact) {
			_data[pos].value = p_val;
			return pos;
		}
		_data.insert(pos, _Pair(p_key, p_val));
		return pos;
	}

	bool has(const T &p_val) const {

		return _find_exact(p_val) != -1;
	}

	void erase(const T &p_val) {

		int pos = _find_exact(p_val);
		if (pos < 0)
			return;
		_data.remove(pos);
	}

	int find(const T &p_val) const {

		return _find_exact(p_val);
	}

	int find_nearest(const T &p_val) const {

		bool exact;
		return _find(p_val, exact);
	}

	_FORCE_INLINE_ int size() const { return _data.size(); }
	_FORCE_INLINE_ bool empty() const { return _data.empty(); }

	const _Pair *get_array() const {

		return _data.ptr();
	}

	_Pair *get_array() {

		return _data.ptr();
	}

	const V &getv(int p_index) const {

		return _data[p_index].value;
	}

	V &getv(int p_index) {

		return _data[p_index].value;
	}

	const T &getk(int p_index) const {

		return _data[p_index].key;
	}

	T &getk(int p_index) {

		return _data[p_index].key;
	}

	inline const V &operator[](const T &p_key) const {

		int pos = _find_exact(p_key);
		if (pos < 0) {
			const T &aux = *((T *)0); //nullreturn
			ERR_FAIL_COND_V(pos < 1, aux);
		}

		return _data[pos].value;
	}

	inline V &operator[](const T &p_key) {

		int pos = _find_exact(p_key);
		if (pos < 0) {
			V val;
			pos = insert(p_key, val);
		}

		return _data[pos].value;
	}
};
#endif // VMAP_H
