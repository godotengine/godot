/*************************************************************************/
/*  rw_lock_windows.cpp                                                  */
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
#if defined(WINDOWS_ENABLED)

#include "rw_lock_windows.h"

#include "error_macros.h"
#include "os/memory.h"
#include <stdio.h>

void RWLockWindows::read_lock() {

	AcquireSRWLockShared(&lock);
}

void RWLockWindows::read_unlock() {

	ReleaseSRWLockShared(&lock);
}

Error RWLockWindows::read_try_lock() {

	if (TryAcquireSRWLockShared(&lock) == 0) {
		return ERR_BUSY;
	} else {
		return OK;
	}
}

void RWLockWindows::write_lock() {

	AcquireSRWLockExclusive(&lock);
}

void RWLockWindows::write_unlock() {

	ReleaseSRWLockExclusive(&lock);
}

Error RWLockWindows::write_try_lock() {
	if (TryAcquireSRWLockExclusive(&lock) == 0) {
		return ERR_BUSY;
	} else {
		return OK;
	}
}

RWLock *RWLockWindows::create_func_windows() {

	return memnew(RWLockWindows);
}

void RWLockWindows::make_default() {

	create_func = create_func_windows;
}

RWLockWindows::RWLockWindows() {

	InitializeSRWLock(&lock);
}

RWLockWindows::~RWLockWindows() {
}

#endif
