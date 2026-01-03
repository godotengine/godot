/**************************************************************************/
/*  test_stream_peer_stdio.h                                              */
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

#pragma once

#include "core/io/stream_peer_stdio.h"
#include "tests/test_macros.h"

#include <fcntl.h>
#include <cerrno>

#ifdef WINDOWS_ENABLED
#include <io.h>
#include <cstdio>
#define READ_FUNCTION _read
#define WRITE_FUNCTION _write
#define PIPE_FUNCTION _pipe
#define DUP_FUNCTION _dup
#define DUP2_FUNCTION _dup2
#define CLOSE_FUNCTION _close
#else
#include <unistd.h>
#define READ_FUNCTION read
#define WRITE_FUNCTION write
#define PIPE_FUNCTION pipe
#define DUP_FUNCTION dup
#define DUP2_FUNCTION dup2
#define CLOSE_FUNCTION close
#endif

namespace TestStreamPeerStdio {

TEST_CASE("[StreamPeerStdio] Write and read through pipes") {
	int stdin_pipe[2]; // stdin_pipe[0] = read, stdin_pipe[1] = write
	int stdout_pipe[2]; // stdout_pipe[0] = read, stdout_pipe[1] = write

#ifdef WINDOWS_ENABLED
	int stdin_fileno = _fileno(stdin);
	int stdout_fileno = _fileno(stdout);

	CHECK(_pipe(stdin_pipe, 4096, _O_BINARY) == 0);
	CHECK(_pipe(stdout_pipe, 4096, _O_BINARY) == 0);

	// _setmode(stdin_fileno, _O_BINARY);
	// _setmode(stdout_fileno, _O_BINARY);
#else
	int stdin_fileno = STDIN_FILENO;
	int stdout_fileno = STDOUT_FILENO;

	CHECK(pipe(stdin_pipe) == 0);
	CHECK(pipe(stdout_pipe) == 0);

	// int flags = fcntl(stdin_fileno, F_GETFL, 0);
	// fcntl(stdin_fileno, F_SETFL, flags | O_NONBLOCK);
#endif

	int original_stdin = DUP_FUNCTION(stdin_fileno);
	int original_stdout = DUP_FUNCTION(stdout_fileno);

	// This will duplicate our "stdin" read pipe into stdin
	DUP2_FUNCTION(stdin_pipe[0], stdin_fileno);
	CLOSE_FUNCTION(stdin_pipe[0]);

	// This will duplicate our "stdout" write pipe into stdout
	DUP2_FUNCTION(stdout_pipe[1], stdout_fileno);
	CLOSE_FUNCTION(stdout_pipe[1]);

	// Create StreamPeerStdio (it will use the redirected stdin/stdout)
	Ref<StreamPeerStdio> stdio;
	stdio.instantiate();

	// Test 1: Write to stdin pipe, read using StreamPeerStdio
	SUBCASE("Read from stdin using StreamPeerStdio") {
		const char *test_input = "Hello from stdin!";
		int input_len = strlen(test_input);

		// Write directly to the stdin pipe
		size_t written = WRITE_FUNCTION(stdin_pipe[1], test_input, input_len);
		CHECK_EQ(written, input_len);

		// Read using StreamPeerStdio
		uint8_t read_buffer[256];
		memset(read_buffer, 0, sizeof(read_buffer));
		int received = 0;

		Error read_err = stdio->get_partial_data(read_buffer, input_len, received);
		CHECK(read_err == OK);
		CHECK_EQ(received, input_len);
		CHECK_EQ(memcmp(read_buffer, test_input, input_len), 0);
	}

	// Test 2: Write using StreamPeerStdio, read from stdout pipe
	SUBCASE("Write to stdout using StreamPeerStdio") {
		const char *test_output = "Hello to stdout!";
		int output_len = strlen(test_output);
		int sent = 0;

		// Write using StreamPeerStdio
		Error write_err = stdio->put_partial_data((const uint8_t *)test_output, output_len, sent);
		CHECK(write_err == OK);
		CHECK_EQ(sent, output_len);

		// Read directly from the stdout pipe
		uint8_t read_buffer[256];
		memset(read_buffer, 0, sizeof(read_buffer));
		size_t read_bytes = READ_FUNCTION(stdout_pipe[0], read_buffer, output_len);
		CHECK_EQ(read_bytes, output_len);
		CHECK_EQ(memcmp(read_buffer, test_output, output_len), 0);
	}

	// Cleanup
	CLOSE_FUNCTION(stdin_pipe[1]);
	CLOSE_FUNCTION(stdout_pipe[0]);

	// Restore original stdin/stdout
	DUP2_FUNCTION(original_stdin, stdin_fileno);
	DUP2_FUNCTION(original_stdout, stdout_fileno);
	CLOSE_FUNCTION(original_stdin);
	CLOSE_FUNCTION(original_stdout);
}

} // namespace TestStreamPeerStdio
