/**************************************************************************/
/*  thread.h                                                              */
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

#include "platform_config.h"
// Define PLATFORM_THREAD_OVERRIDE in your platform's `platform_config.h`
// to use a custom Thread implementation defined in `platform/[your_platform]/platform_thread.h`
// Overriding the platform implementation is required in some proprietary platforms
#ifdef PLATFORM_THREAD_OVERRIDE
#include "platform_thread.h"
#else

#ifndef THREAD_H
#define THREAD_H

#include "core/templates/safe_refcount.h"
#include "core/typedefs.h"

#ifdef MINGW_ENABLED
#define MINGW_STDTHREAD_REDUNDANCY_WARNING
#include "thirdparty/mingw-std-threads/mingw.thread.h"
#define THREADING_NAMESPACE mingw_stdthread
#else
#include <thread>
#define THREADING_NAMESPACE std
#endif

class String;

class Thread {
public:
	typedef void (*Callback)(void *p_userdata);

	typedef uint64_t ID;

	enum : ID {
		UNASSIGNED_ID = 0,
		MAIN_ID = 1
	};

	enum Priority {
		PRIORITY_LOW,
		PRIORITY_NORMAL,
		PRIORITY_HIGH
	};

	struct Settings {
		Priority priority;
		Settings() { priority = PRIORITY_NORMAL; }
	};

	struct PlatformFunctions {
		Error (*set_name)(const String &) = nullptr;
		void (*set_priority)(Thread::Priority) = nullptr;
		void (*init)() = nullptr;
		void (*wrapper)(Thread::Callback, void *) = nullptr;
		void (*term)() = nullptr;
	};

private:
	friend class Main;

	ID id = UNASSIGNED_ID;
	static SafeNumeric<uint64_t> id_counter;
	static thread_local ID caller_id;
	THREADING_NAMESPACE::thread thread;

	static void callback(ID p_caller_id, const Settings &p_settings, Thread::Callback p_callback, void *p_userdata);

	static PlatformFunctions platform_functions;

	static void make_main_thread() { caller_id = MAIN_ID; }
	static void release_main_thread() { caller_id = UNASSIGNED_ID; }

public:
	static void _set_platform_functions(const PlatformFunctions &p_functions);

	_FORCE_INLINE_ ID get_id() const { return id; }
	// get the ID of the caller thread
	_FORCE_INLINE_ static ID get_caller_id() {
		if (unlikely(caller_id == UNASSIGNED_ID)) {
			caller_id = id_counter.increment();
		}
		return caller_id;
	}
	// get the ID of the main thread
	_FORCE_INLINE_ static ID get_main_id() { return MAIN_ID; }

	_FORCE_INLINE_ static bool is_main_thread() { return caller_id == MAIN_ID; } // Gain a tiny bit of perf here because there is no need to validate caller_id here, because only main thread will be set as 1.

	static Error set_name(const String &p_name);

	ID start(Thread::Callback p_callback, void *p_user, const Settings &p_settings = Settings());
	bool is_started() const;
	///< waits until thread is finished, and deallocates it.
	void wait_to_finish();

	Thread();
	~Thread();
};

#endif // THREAD_H

#endif // PLATFORM_THREAD_OVERRIDE
