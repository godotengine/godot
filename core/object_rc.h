/*************************************************************************/
/*  object_rc.h                                                          */
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

#ifndef OBJECT_RC_H
#define OBJECT_RC_H

#include "core/os/memory.h"
#include "core/typedefs.h"

#include <atomic>

class Object;

// Used to track Variants pointing to a non-Reference Object
class ObjectRC {
	std::atomic<Object *> _ptr;
	std::atomic<uint32_t> _users;

public:
	// This is for allowing debug builds to check for instance ID validity,
	// so warnings are shown in debug builds when a stray Variant (one pointing
	// to a released Object) would have happened.
	const ObjectID instance_id;

	_FORCE_INLINE_ void increment() {
		_users.fetch_add(1, std::memory_order_relaxed);
	}

	_FORCE_INLINE_ bool decrement() {
		return _users.fetch_sub(1, std::memory_order_relaxed) == 1;
	}

	_FORCE_INLINE_ bool invalidate() {
		_ptr.store(nullptr, std::memory_order_release);
		return decrement();
	}

	_FORCE_INLINE_ Object *get_ptr() {
		return _ptr.load(std::memory_order_acquire);
	}

	_FORCE_INLINE_ ObjectRC(Object *p_object) :
			instance_id(p_object->get_instance_id()) {
		// 1 (the Object) + 1 (the first user)
		_users.store(2, std::memory_order_relaxed);
		_ptr.store(p_object, std::memory_order_release);
	}
};

#endif // OBJECT_RC_H
