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

#include "core/typedefs.h"

#if !defined(NO_THREADS)
#include "core/safe_refcount.h"
#include <thread>
#endif

class String;

class Thread {
public:
	typedef void (*Callback)(void *p_userdata);

	typedef uint64_t ID;

	enum Priority {
		PRIORITY_LOW,
		PRIORITY_NORMAL,
		PRIORITY_HIGH
	};

	struct Settings {
		Priority priority;
		Settings() { priority = PRIORITY_NORMAL; }
	};

private:
#if !defined(NO_THREADS)
	friend class Main;

	static ID main_thread_id;

	static uint64_t _thread_id_hash(const std::thread::id &p_t);

	ID id = _thread_id_hash(std::thread::id());
	std::thread thread;

	static void callback(Thread *p_self, const Settings &p_settings, Thread::Callback p_callback, void *p_userdata);

	static Error (*set_name_func)(const String &);
	static void (*set_priority_func)(Thread::Priority);
	static void (*init_func)();
	static void (*term_func)();
#endif

public:
	static void _set_platform_funcs(
			Error (*p_set_name_func)(const String &),
			void (*p_set_priority_func)(Thread::Priority),
			void (*p_init_func)() = nullptr,
			void (*p_term_func)() = nullptr);

#if !defined(NO_THREADS)
	_FORCE_INLINE_ ID get_id() const { return id; }
	// get the ID of the caller thread
	static ID get_caller_id();
	// get the ID of the main thread
	_FORCE_INLINE_ static ID get_main_id() { return main_thread_id; }

	static Error set_name(const String &p_name);

	void start(Thread::Callback p_callback, void *p_user, const Settings &p_settings = Settings());
	bool is_started() const;
	///< waits until thread is finished, and deallocates it.
	void wait_to_finish();

	~Thread();
#else
	_FORCE_INLINE_ ID get_id() const { return 0; }
	// get the ID of the caller thread
	_FORCE_INLINE_ static ID get_caller_id() { return 0; }
	// get the ID of the main thread
	_FORCE_INLINE_ static ID get_main_id() { return 0; }

	static Error set_name(const String &p_name) { return ERR_UNAVAILABLE; }

	void start(Thread::Callback p_callback, void *p_user, const Settings &p_settings = Settings()) {}
	bool is_started() const { return false; }
	void wait_to_finish() {}
#endif
};

#endif // THREAD_H
#endif // PLATFORM_THREAD_OVERRIDE
