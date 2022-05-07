/*************************************************************************/
/*  arguments_vector.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef ARGUMENTS_VECTOR_H
#define ARGUMENTS_VECTOR_H

#include "core/os/memory.h"

template <typename T, int POOL_SIZE = 5>
struct ArgumentsVector {
private:
	T pool[POOL_SIZE];
	T *_ptr;
	int size;

	ArgumentsVector() = delete;
	ArgumentsVector(const ArgumentsVector &) = delete;

public:
	T *ptr() { return _ptr; }
	T &get(int p_idx) { return _ptr[p_idx]; }
	void set(int p_idx, const T &p_value) { _ptr[p_idx] = p_value; }

	explicit ArgumentsVector(int p_size) :
			size(p_size) {
		if (p_size <= POOL_SIZE) {
			_ptr = pool;
		} else {
			_ptr = memnew_arr(T, p_size);
		}
	}

	~ArgumentsVector() {
		if (size > POOL_SIZE) {
			memdelete_arr(_ptr);
		}
	}
};

#endif // ARGUMENTS_VECTOR_H
