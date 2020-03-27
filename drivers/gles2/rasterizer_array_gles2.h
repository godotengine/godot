#pragma once

/*************************************************************************/
/*  rasterizer_array_gles2.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

/**
 * Fast single-threaded growable array for POD types.
 * For use in render drivers, not for general use.
 * TO BE REPLACED by local_vector.
*/

#include "core/os/memory.h"
#include "core/vector.h"

#include <string.h>

template <class T>
class RasterizerArrayGLES2 {
public:
	RasterizerArrayGLES2() {
		_list = 0;
		_size = 0;
		_max_size = 0;
	}
	~RasterizerArrayGLES2() { free(); }

	_FORCE_INLINE_ T &operator[](unsigned int ui) { return _list[ui]; }
	_FORCE_INLINE_ const T &operator[](unsigned int ui) const { return _list[ui]; }

	void free() {
		if (_list) {
			memdelete_arr(_list);
			_list = 0;
		}
		_size = 0;
		_max_size = 0;
	}

	void create(int p_size) {
		free();
		if (p_size) {
			_list = memnew_arr(T, p_size);
		}
		_size = 0;
		_max_size = p_size;
	}

	_FORCE_INLINE_ void reset() { _size = 0; }

	_FORCE_INLINE_ T *request_with_grow() {
		T *p = request();
		if (!p) {
			grow();
			return request_with_grow();
		}
		return p;
	}

	// none of that inefficient pass by value stuff here, thanks
	_FORCE_INLINE_ T *request() {
		if (_size < _max_size) {
			return &_list[_size++];
		}
		return 0;
	}

	// several items at a time
	_FORCE_INLINE_ T *request(int p_num_items) {
		int old_size = _size;
		_size += p_num_items;

		if (_size <= _max_size) {
			return &_list[old_size];
		}

		// revert
		_size = old_size;
		return 0;
	}

	_FORCE_INLINE_ int size() const { return _size; }
	_FORCE_INLINE_ int max_size() const { return _max_size; }
	_FORCE_INLINE_ const T *get_data() const { return _list; }

	bool copy_from(const RasterizerArrayGLES2<T> &o) {
		// no resizing done here, it should be done manually
		if (o.size() > _max_size)
			return false;

		// pod types only please!
		memcpy(_list, o.get_data(), o.size() * sizeof(T));
		_size = o.size();
		return true;
	}

	// if you want this to be cheap, call reset before grow,
	// to ensure there is no data to copy
	void grow() {
		unsigned int new_max_size = _max_size * 2;
		if (!new_max_size)
			new_max_size = 1;

		T *new_list = memnew_arr(T, new_max_size);

		// copy .. pod types only
		if (_list) {
			memcpy(new_list, _list, _size * sizeof(T));
		}

		unsigned int new_size = size();
		free();
		_list = new_list;
		_size = new_size;
		_max_size = new_max_size;
	}

private:
	T *_list;
	int _size;
	int _max_size;
};

template <class T>
class RasterizerArray_non_pod_GLES2 {
public:
	RasterizerArray_non_pod_GLES2() {
		_size = 0;
	}

	const T &operator[](unsigned int ui) const { return _list[ui]; }

	void create(int p_size) {
		_list.resize(p_size);
		_size = 0;
	}
	void reset() { _size = 0; }

	void push_back(const T &val) {
		while (true) {
			if (_size < max_size()) {
				_list.set(_size, val);
				_size++;
				return;
			}

			grow();
		}
	}

	int size() const { return _size; }
	int max_size() const { return _list.size(); }

private:
	void grow() {
		unsigned int new_max_size = _list.size() * 2;
		if (!new_max_size)
			new_max_size = 1;
		_list.resize(new_max_size);
	}

	Vector<T> _list;
	int _size;
};
