/**************************************************************************/
/*  thread_posix.cpp                                                      */
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

#if defined(UNIX_ENABLED)

#include "thread_posix.h"

#include "core/os/thread.h"
#include "core/string/ustring.h"

#if defined(PLATFORM_THREAD_OVERRIDE) && defined(__APPLE__)
void init_thread_posix() {
}
#else

#ifdef PTHREAD_BSD_SET_NAME
#include <pthread_np.h>
#endif

static Error set_name(const String &p_name) {
#ifdef PTHREAD_NO_RENAME
	return ERR_UNAVAILABLE;

#else

#ifdef PTHREAD_RENAME_SELF

	// check if thread is the same as caller
	int err = pthread_setname_np(p_name.utf8().get_data());

#else

	pthread_t running_thread = pthread_self();
#ifdef PTHREAD_BSD_SET_NAME
	pthread_set_name_np(running_thread, p_name.utf8().get_data());
	int err = 0; // Open/FreeBSD ignore errors in this function
#elif defined(PTHREAD_NETBSD_SET_NAME)
	int err = pthread_setname_np(running_thread, "%s", const_cast<char *>(p_name.utf8().get_data()));
#else
	int err = pthread_setname_np(running_thread, p_name.utf8().get_data());
#endif // PTHREAD_BSD_SET_NAME

#endif // PTHREAD_RENAME_SELF

	return err == 0 ? OK : ERR_INVALID_PARAMETER;

#endif // PTHREAD_NO_RENAME
}

static bool get_stack_limits(void **r_bottom, void **r_top, void **r_frame) {
	pthread_t pth_self = pthread_self();
	pthread_attr_t pth_attr;
	uint8_t *stack_addr = nullptr;
	uint8_t *frame_addr = (uint8_t *)__builtin_frame_address(0);
	size_t stack_size = 0;
	size_t guard_size = 0;

	if (pthread_getattr_np(pth_self, &pth_attr) == 0) {
		pthread_attr_getstack(&pth_attr, (void **)&stack_addr, &stack_size);
		pthread_attr_getguardsize(&pth_attr, &guard_size);
		pthread_attr_destroy(&pth_attr);
		if (frame_addr > stack_addr) {
			if (r_bottom) {
				*r_bottom = stack_addr + stack_size - MAX(guard_size, (size_t)(4 * 1024));
			}
			if (r_top) {
				*r_top = stack_addr;
			}
		} else {
			if (r_bottom) {
				*r_bottom = stack_addr;
			}
			if (r_top) {
				*r_top = stack_addr - stack_size + MAX(guard_size, (size_t)(4 * 1024));
			}
		}
		if (r_frame) {
			*r_frame = frame_addr;
		}
		return true;
	} else {
		return false;
	}
}

void init_thread_posix() {
	Thread::_set_platform_functions({ .set_name = set_name, .get_stack_limits = get_stack_limits });
}

#endif // PLATFORM_THREAD_OVERRIDE && __APPLE__

#endif // UNIX_ENABLED
