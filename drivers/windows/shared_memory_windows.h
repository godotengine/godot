/*************************************************************************/
/*  shared_memory_windows.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef SHARED_MEMORY_WINDOWS_H
#define SHARED_MEMORY_WINDOWS_H

#if defined(WINDOWS_ENABLED) && !defined(NO_SHARED_MEMORY)

#include "core/os/shared_memory.h"
#include "core/ustring.h"

#include <windows.h>

class SharedMemoryWindows : public SharedMemory {

	struct ControlData {
		int64_t serial;
		int64_t size;
	};

	CharString name;
	HANDLE control_fm;
	volatile ControlData *control;
	HANDLE data_fm;
	uint8_t *data;
	uint8_t *prev_data;
	int64_t data_serial;

	static SharedMemory *create_func_windows(const String &p_name);

	CharString _build_curr_data_mapping_name();
	void _create_curr_data_mapping();
	void _switch_to_curr_data_mapping();

public:
	static void make_default();

	virtual Error open();
	virtual void close();
	virtual bool is_open();

	virtual uint8_t *begin_access();
	virtual void end_access();

	virtual uint8_t *set_size(int64_t p_size);
	virtual int64_t get_size();

	explicit SharedMemoryWindows(const String &p_name);
	virtual ~SharedMemoryWindows();
};

#endif
#endif
