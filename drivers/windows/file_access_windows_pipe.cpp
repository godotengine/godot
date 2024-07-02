/**************************************************************************/
/*  file_access_windows_pipe.cpp                                          */
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

#ifdef WINDOWS_ENABLED

#include "file_access_windows_pipe.h"

#include "core/os/os.h"
#include "core/string/print_string.h"

Error FileAccessWindowsPipe::open_existing(HANDLE p_rfd, HANDLE p_wfd) {
	// Open pipe using handles created by CreatePipe(rfd, wfd, NULL, 4096) call in the OS.execute_with_pipe.
	_close();

	path_src = String();
	ERR_FAIL_COND_V_MSG(fd[0] != 0 || fd[1] != 0, ERR_ALREADY_IN_USE, "Pipe is already in use.");
	fd[0] = p_rfd;
	fd[1] = p_wfd;

	last_error = OK;
	return OK;
}

Error FileAccessWindowsPipe::open_internal(const String &p_path, int p_mode_flags) {
	_close();

	path_src = p_path;
	ERR_FAIL_COND_V_MSG(fd[0] != 0 || fd[1] != 0, ERR_ALREADY_IN_USE, "Pipe is already in use.");

	path = String("\\\\.\\pipe\\LOCAL\\") + p_path.replace("pipe://", "").replace("/", "_");

	HANDLE h = CreateFileW((LPCWSTR)path.utf16().get_data(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (h == INVALID_HANDLE_VALUE) {
		h = CreateNamedPipeW((LPCWSTR)path.utf16().get_data(), PIPE_ACCESS_DUPLEX, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, 4096, 4096, 0, nullptr);
		if (h == INVALID_HANDLE_VALUE) {
			last_error = ERR_FILE_CANT_OPEN;
			return last_error;
		}
		ConnectNamedPipe(h, NULL);
	}
	fd[0] = h;
	fd[1] = h;

	last_error = OK;
	return OK;
}

void FileAccessWindowsPipe::_close() {
	if (fd[0] == 0) {
		return;
	}
	if (fd[1] != fd[0]) {
		CloseHandle(fd[1]);
	}
	CloseHandle(fd[0]);
	fd[0] = 0;
	fd[1] = 0;
}

bool FileAccessWindowsPipe::is_open() const {
	return (fd[0] != 0 || fd[1] != 0);
}

String FileAccessWindowsPipe::get_path() const {
	return path_src;
}

String FileAccessWindowsPipe::get_path_absolute() const {
	return path_src;
}

uint64_t FileAccessWindowsPipe::get_buffer(uint8_t *p_dst, uint64_t p_length) const {
	ERR_FAIL_COND_V_MSG(fd[0] == 0, -1, "Pipe must be opened before use.");
	ERR_FAIL_COND_V(!p_dst && p_length > 0, -1);

	DWORD read = -1;
	if (!ReadFile(fd[0], p_dst, p_length, &read, nullptr) || read != p_length) {
		last_error = ERR_FILE_CANT_READ;
	} else {
		last_error = OK;
	}
	return read;
}

Error FileAccessWindowsPipe::get_error() const {
	return last_error;
}

void FileAccessWindowsPipe::store_buffer(const uint8_t *p_src, uint64_t p_length) {
	ERR_FAIL_COND_MSG(fd[1] == 0, "Pipe must be opened before use.");
	ERR_FAIL_COND(!p_src && p_length > 0);

	DWORD read = -1;
	bool ok = WriteFile(fd[1], p_src, p_length, &read, nullptr);
	if (!ok || read != p_length) {
		last_error = ERR_FILE_CANT_WRITE;
	} else {
		last_error = OK;
	}
}

void FileAccessWindowsPipe::close() {
	_close();
}

FileAccessWindowsPipe::~FileAccessWindowsPipe() {
	_close();
}

#endif // WINDOWS_ENABLED
