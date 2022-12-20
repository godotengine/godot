/*************************************************************************/
/*  process.h                                                            */
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

#ifndef PROCESS_H
#define PROCESS_H

#include "core/object/ref_counted.h"
#include "core/os/mutex.h"

class Process : public RefCounted {
	GDCLASS(Process, RefCounted);

protected:
	static Ref<Process> (*_create)(const String &p_path, const Vector<String> &p_arguments, const String &p_working_dir, bool p_open_stdin);
	static void _bind_methods();

	Mutex mutex;
	Vector<String> stdout_lines;
	Vector<String> stderr_lines;

public:
	static Ref<Process> create(const String &p_path, const Vector<String> &p_arguments = Vector<String>(), const String &p_working_dir = "", bool p_open_stdin = false);
	int get_available_stdout_lines() const;
	int get_available_stderr_lines() const;
	String get_stdout_line();
	String get_stderr_line();
	virtual int get_exit_status() const { return -1; };
	virtual int get_id() const { return -1; };
	virtual void kill(bool m_force = false){};
	virtual bool write(const String &p_input) { return false; };
	virtual void close_stdin(){};

	virtual ~Process(){};
};

#endif // PROCESS_H
