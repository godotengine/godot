/*************************************************************************/
/*  process_tiny_process_lib.h                                           */
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

#ifndef PROCESS_TINY_PROCESS_LIB_H
#define PROCESS_TINY_PROCESS_LIB_H

#if defined(UNIX_ENABLED) || defined(WINDOWS_ENABLED)
#include "core/io/process.h"
#include "thirdparty/tiny-process-library/process.hpp"

class ProcessTinyProcessLibrary : public Process {
protected:
	TinyProcessLib::Process *process;
	static Ref<Process> create_tpl(const String &m_path, const Vector<String> &p_arguments, const String &p_working_dir, bool p_open_stdin);
	void _on_stdout(const char *m_bytes, size_t m_size);
	void _on_stderr(const char *m_bytes, size_t m_size);

	bool has_open_stdin;

public:
	static void make_default();

	virtual int get_exit_status() const;
	virtual int get_id() const;
	virtual void kill(bool p_force = false);
	virtual bool write(const String &p_input);
	virtual void close_stdin();
	ProcessTinyProcessLibrary(const String &m_path, const Vector<String> &p_arguments, const String &p_working_dir, bool p_open_stdin);
	virtual ~ProcessTinyProcessLibrary();
};

#endif // UNIX_ENABLED || WINDOWS_ENABLED

#endif // PROCESS_TINY_PROCESS_LIB_H
