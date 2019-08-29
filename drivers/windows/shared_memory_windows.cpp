/*************************************************************************/
/*  shared_memory_windows.cpp                                            */
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

#if defined(WINDOWS_ENABLED) && !defined(NO_SHARED_MEMORY)

#include "shared_memory_windows.h"

#include <memoryapi.h>

SharedMemory *SharedMemoryWindows::create_func_windows(const String &p_name) {

	return memnew(SharedMemoryWindows(p_name));
}

void SharedMemoryWindows::make_default() {

	create_func = create_func_windows;
}

Error SharedMemoryWindows::open() {

	ERR_FAIL_COND_V_MSG(control_fm, ERR_ALREADY_IN_USE, ERR_STR_ALREADY_OPEN);

	// Using one shared memory block matching the exact name for control

	control_fm = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, sizeof(ControlData), name.get_data());
	if (!control_fm && GetLastError() == ERR_ALREADY_EXISTS) {
		control_fm = OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name.get_data());
		if (!control_fm) {
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, ERR_STR_CANNOT_CREATE_OR_OPEN);
		}
	}

	control = (ControlData *)MapViewOfFile(control_fm, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(ControlData));
	if (!control) {
		CloseHandle(control_fm);
		control_fm = nullptr;
		ERR_FAIL_V_MSG(ERR_CANT_OPEN, ERR_STR_CANNOT_CREATE_OR_OPEN);
	}

	// Now take care of the actual shared memory block; if the serial is 0 the size still must be set

	if (control->serial > 0) {
		_switch_to_curr_data_mapping();
		if (!data_fm) {
			UnmapViewOfFile((void *)control);
			control = nullptr;
			CloseHandle(control_fm);
			control_fm = nullptr;
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, ERR_STR_CANNOT_CREATE_OR_OPEN);
		}
	}

	return OK;
}

void SharedMemoryWindows::close() {

	if (!is_open()) {
		return;
	}

	if (data) {
		WARN_PRINT(WARN_STR_CLOSING_BEFORE_END_ACCESS);
		end_access();
	}

	data_serial = 0;
	prev_data = nullptr;

	CloseHandle(data_fm);
	data_fm = nullptr;

	UnmapViewOfFile((void *)control);
	control = nullptr;
	CloseHandle(control_fm);
	control_fm = nullptr;
}

_FORCE_INLINE_ bool SharedMemoryWindows::is_open() {

	return control_fm != nullptr;
}

uint8_t *SharedMemoryWindows::begin_access() {

	if (data) {
		WARN_PRINT(WARN_STR_BEGIN_ACCESS_WHILE_ALREADY);
	}

	ERR_FAIL_COND_V_MSG(!is_open(), nullptr, ERR_STR_NOT_OPEN);

	if (control->size == 0) {
		data = static_cast<uint8_t *>(UNSIZED);
	} else {
		if (data_serial != control->serial) {
			_switch_to_curr_data_mapping();
		}

		// Try to remap at the same address as the last time
		if (prev_data) {
			data = (uint8_t *)MapViewOfFileEx(data_fm, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, control->size, prev_data);
			if (!data) {
				prev_data = nullptr;
			}
		}
		if (!prev_data) {
			data = (uint8_t *)MapViewOfFile(data_fm, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, control->size);
		}
	}

	return data;
}

void SharedMemoryWindows::end_access() {

	if (!data) {
		WARN_PRINT(WARN_STR_END_ACCESS_WITHOUT_BEGIN_ACCESS);
		return;
	}

	if (data != UNSIZED) {
		UnmapViewOfFile(data);
		prev_data = data;
	}
	data = nullptr;
}

uint8_t *SharedMemoryWindows::set_size(int64_t p_size) {

	ERR_FAIL_COND_V_MSG(!data, nullptr, ERR_STR_SIZE_NOT_AVAILABLE);

	if (p_size == control->size) {
		return data;
	}

	// Create new generation and transfer data to it

	HANDLE old_data_fm = data_fm;
	uint8_t *old_data = data;
	uint64_t old_size = control->size;

	control->size = p_size;
	++control->serial;
	data_serial = control->serial;
	_create_curr_data_mapping();

	prev_data = nullptr; // Don't bother trying to map at the same address since it's busy for sure
	data = nullptr; // So begin_access() doesn't know old block is already mapped
	begin_access();

	if (old_data != UNSIZED) {
		memcpy(data, old_data, MIN(control->size, old_size));

		// Cleanup old generation

		UnmapViewOfFile(old_data);
		CloseHandle(old_data_fm);
	}

	return data;
}

int64_t SharedMemoryWindows::get_size() {

	ERR_FAIL_COND_V_MSG(!is_open(), -1, ERR_STR_SIZE_NOT_AVAILABLE);

	return control->size;
}

CharString SharedMemoryWindows::_build_curr_data_mapping_name() {

	return ("__" + String(name) + "@" + itos(control->serial) + "__").ascii();
}

void SharedMemoryWindows::_create_curr_data_mapping() {

	CharString curr_name = _build_curr_data_mapping_name();
	data_fm = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, control->size, curr_name.get_data());
}

void SharedMemoryWindows::_switch_to_curr_data_mapping() {

	ERR_FAIL_COND(data_serial == control->serial);

	if (data_fm) {
		CloseHandle(data_fm);
	}

	CharString curr_name = _build_curr_data_mapping_name();
	data_fm = OpenFileMappingA(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, curr_name.get_data());

	data_serial = control->serial;
}

SharedMemoryWindows::SharedMemoryWindows(const String &p_name) :
		name(p_name.ascii()),
		control_fm(nullptr),
		control(nullptr),
		data_fm(nullptr),
		data(nullptr),
		prev_data(nullptr),
		data_serial(0) {
}

SharedMemoryWindows::~SharedMemoryWindows() {

	if (is_open()) {
		close();
	}
}

#endif
