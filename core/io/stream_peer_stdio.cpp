/**************************************************************************/
/*  stream_peer_stdio.cpp                                                 */
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

#include "stream_peer_stdio.h"

#include <fcntl.h>

#include <cerrno>
#include <cstdio>

#ifdef WINDOWS_ENABLED
#include <windows.h>

#include <io.h>
#define READ_FUNCTION _read
#define WRITE_FUNCTION _write
#else
#include <sys/ioctl.h>
#include <unistd.h>
#define READ_FUNCTION read
#define WRITE_FUNCTION write
#endif

StreamPeerStdio::StreamPeerStdio() {
	// Set stdin to non-blocking mode and binary mode
#ifdef WINDOWS_ENABLED
	stdin_fileno = _fileno(stdin);
	stdout_fileno = _fileno(stdout);

	invalid_handles = _setmode(stdin_fileno, _O_BINARY) == -1;
	invalid_handles = invalid_handles || (_setmode(stdout_fileno, _O_BINARY) == -1);
#else
	stdin_fileno = STDIN_FILENO;
	stdout_fileno = STDOUT_FILENO;

	int flags = fcntl(stdin_fileno, F_GETFL, 0);
	invalid_handles = fcntl(stdin_fileno, F_SETFL, flags | O_NONBLOCK) == -1;
#endif
}

Error StreamPeerStdio::put_data(const uint8_t *p_data, int p_bytes) {
	int sent;
	return put_partial_data(p_data, p_bytes, sent);
}

Error StreamPeerStdio::put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) {
	if (invalid_handles) {
		ERR_PRINT_ONCE("Can't write to stdout, invalid handle.");
		return FAILED;
	}

	int sent = WRITE_FUNCTION(stdout_fileno, p_data, p_bytes);
	if (sent < 0) {
		r_sent = 0;
		return FAILED;
	}

	r_sent = sent;
	fflush(stdout);

	return OK;
}

Error StreamPeerStdio::get_data(uint8_t *p_buffer, int p_bytes) {
	int received;
	return get_partial_data(p_buffer, p_bytes, received);
}

GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Wlogical-op") // Silence a false positive. See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69602

Error StreamPeerStdio::get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) {
	if (invalid_handles) {
		ERR_PRINT_ONCE("Can't read from stdin, invalid handle.");
		return FAILED;
	}

	int received = READ_FUNCTION(stdin_fileno, p_buffer, p_bytes);
	if (received < 0) {
		if (errno == EAGAIN || errno == EWOULDBLOCK) {
			r_received = 0;
			return ERR_BUSY;
		}
		r_received = 0;
		return FAILED;
	} else if (received == 0) { // EOF
		r_received = 0;
		return FAILED;
	}

	r_received = received;
	return OK;
}

GODOT_GCC_WARNING_POP

int StreamPeerStdio::get_available_bytes() const {
#ifdef WINDOWS_ENABLED
	DWORD buf_rem = 0;
	PeekNamedPipe((HANDLE)_get_osfhandle(stdin_fileno), nullptr, 0, nullptr, &buf_rem, nullptr);
	return (int)buf_rem;
#else
	int buf_rem = 0;
	ioctl(stdin_fileno, FIONREAD, &buf_rem);
	return buf_rem;
#endif
}
