/*************************************************************************/
/*  safe_refcount.h                                                      */
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
#ifndef SAFE_REFCOUNT_H
#define SAFE_REFCOUNT_H

#include "os/mutex.h"
/* x86/x86_64 GCC */

#include "platform_config.h"
#include "typedefs.h"

uint32_t atomic_conditional_increment(register uint32_t *counter);
uint32_t atomic_decrement(register uint32_t *pw);
uint32_t atomic_increment(register uint32_t *pw);

struct SafeRefCount {

	uint32_t count;

public:
	// destroy() is called when weak_count_ drops to zero.

	bool ref() { //true on success

		return atomic_conditional_increment(&count) != 0;
	}

	uint32_t refval() { //true on success

		return atomic_conditional_increment(&count);
	}

	bool unref() { // true if must be disposed of

		if (atomic_decrement(&count) == 0) {
			return true;
		}

		return false;
	}

	uint32_t get() const { // nothrow

		return count;
	}

	void init(uint32_t p_value = 1) {

		count = p_value;
	}
};

#endif
