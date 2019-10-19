/*************************************************************************/
/*  copyable_atomic.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef COPYABLE_ATOMIC_H
#define COPYABLE_ATOMIC_H

#include <atomic>

// Atomic integral types are trivially copyable, but they have their default copy constructor
// deleted. Therefore, when a class has a member of any of such types, it ends up being not
// copyable. This little helper allows to overcome this limitation, for instance to put a class
// that contains an atomic in a container, but must be used with care.

template <typename T>
class CopyableAtomic : public std::atomic<T> {
public:
	CopyableAtomic() = default;

	CopyableAtomic<T> &operator=(const CopyableAtomic<T> &p_other) {
		this->store(p_other.load());
		return *this;
	}

	CopyableAtomic<T> &operator=(const T &p_value) {
		this->store(p_value);
		return *this;
	}
};

#endif // COPYABLE_ATOMIC_H
