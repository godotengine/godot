/*************************************************************************/
/*  mutex_windows.cpp                                                    */
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
#include "mutex_windows.h"
#include "os/memory.h"

#ifdef WINDOWS_ENABLED

void MutexWindows::lock() {

#ifdef WINDOWS_USE_MUTEX
	WaitForSingleObject(mutex, INFINITE);
#else
	EnterCriticalSection(&mutex);
#endif
}

void MutexWindows::unlock() {

#ifdef WINDOWS_USE_MUTEX
	ReleaseMutex(mutex);
#else
	LeaveCriticalSection(&mutex);
#endif
}

Error MutexWindows::try_lock() {

#ifdef WINDOWS_USE_MUTEX
	return (WaitForSingleObject(mutex, 0) == WAIT_TIMEOUT) ? ERR_BUSY : OK;
#else

	if (TryEnterCriticalSection(&mutex))
		return OK;
	else
		return ERR_BUSY;
#endif
}

Mutex *MutexWindows::create_func_windows(bool p_recursive) {

	return memnew(MutexWindows);
}

void MutexWindows::make_default() {

	create_func = create_func_windows;
}

MutexWindows::MutexWindows() {

#ifdef WINDOWS_USE_MUTEX
	mutex = CreateMutex(NULL, FALSE, NULL);
#else
#ifdef UWP_ENABLED
	InitializeCriticalSectionEx(&mutex, 0, 0);
#else
	InitializeCriticalSection(&mutex);
#endif
#endif
}

MutexWindows::~MutexWindows() {

#ifdef WINDOWS_USE_MUTEX
	CloseHandle(mutex);
#else

	DeleteCriticalSection(&mutex);
#endif
}

#endif
