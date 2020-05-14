/*************************************************************************/
/*  thread.h                                                             */
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

#ifndef THREAD_H
#define THREAD_H

#include "core/typedefs.h"
#include "core/ustring.h"

typedef void (*ThreadCreateCallback)(void *p_userdata);

class Thread {
public:
	enum Priority {

		PRIORITY_LOW,
		PRIORITY_NORMAL,
		PRIORITY_HIGH
	};

	struct Settings {
		Priority priority;
		Settings() { priority = PRIORITY_NORMAL; }
	};

	typedef uint64_t ID;

protected:
	static Thread *(*create_func)(ThreadCreateCallback p_callback, void *, const Settings &);
	static ID (*get_thread_id_func)();
	static void (*wait_to_finish_func)(Thread *);
	static Error (*set_name_func)(const String &);

	friend class Main;

	static ID _main_thread_id;

	Thread() {}

public:
	virtual ID get_id() const = 0;

	static Error set_name(const String &p_name);
	_FORCE_INLINE_ static ID get_main_id() { return _main_thread_id; } ///< get the ID of the main thread
	static ID get_caller_id(); ///< get the ID of the caller function ID
	static void wait_to_finish(Thread *p_thread); ///< waits until thread is finished, and deallocates it.
	static Thread *create(ThreadCreateCallback p_callback, void *p_user, const Settings &p_settings = Settings()); ///< Static function to create a thread, will call p_callback

	virtual ~Thread() {}
};

#endif // THREAD_H
