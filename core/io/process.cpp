/*************************************************************************/
/*  process.cpp                                                          */
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

#include "process.h"
#include "core/os/os.h"

Ref<Process> (*Process::_create)(const String &p_path, const Vector<String> &p_arguments, const String &p_working_dir, bool p_open_stdin) = nullptr;

void Process::_bind_methods() {
	ClassDB::bind_static_method("Process", D_METHOD("create", "path", "arguments", "working_directory", "open_stdin"), &Process::create, DEFVAL(Vector<String>()), DEFVAL(""), DEFVAL(false));

	ClassDB::bind_method(D_METHOD("get_available_stdout_lines"), &Process::get_available_stdout_lines);
	ClassDB::bind_method(D_METHOD("get_stdout_line"), &Process::get_stdout_line);
	ClassDB::bind_method(D_METHOD("get_available_stderr_lines"), &Process::get_available_stderr_lines);
	ClassDB::bind_method(D_METHOD("get_stderr_line"), &Process::get_stderr_line);

	ClassDB::bind_method(D_METHOD("get_exit_status"), &Process::get_exit_status);
	ClassDB::bind_method(D_METHOD("kill", "force"), &Process::kill, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_id"), &Process::get_id);
	ClassDB::bind_method(D_METHOD("write", "input"), &Process::write);
	ClassDB::bind_method(D_METHOD("close_stdin"), &Process::close_stdin);
}

Ref<Process> Process::create(const String &p_path, const Vector<String> &p_arguments, const String &p_working_dir, bool p_open_stdin) {
	if (_create) {
		if (!p_working_dir.is_empty()) {
			return _create(p_path, p_arguments, OS::get_singleton()->get_executable_path().get_base_dir(), p_open_stdin);
		} else {
			return _create(p_path, p_arguments, p_working_dir, p_open_stdin);
		}
	}

	ERR_PRINT("Unable to create process, platform not supported");

	return nullptr;
}

int Process::get_available_stdout_lines() const {
	mutex.lock();
	int available_lines = stdout_lines.size();
	mutex.unlock();
	return available_lines;
}

int Process::get_available_stderr_lines() const {
	mutex.lock();
	int available_lines = stderr_lines.size();
	mutex.unlock();
	return available_lines;
}

String Process::get_stdout_line() {
	String line = "";
	mutex.lock();
	if (stdout_lines.size() > 0) {
		line = stdout_lines[0];
		stdout_lines.remove_at(0);
	}
	mutex.unlock();
	return line;
}

String Process::get_stderr_line() {
	String line = "";
	mutex.lock();
	if (stderr_lines.size() > 0) {
		line = stderr_lines[0];
		stderr_lines.remove_at(0);
	}
	mutex.unlock();
	return line;
}
