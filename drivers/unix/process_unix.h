/*************************************************************************/
/*  process_unix.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef PROCESS_UNIX_H
#define PROCESS_UNIX_H

#ifdef UNIX_ENABLED

#include "os/process.h"
#include "ring_buffer.h"

#include <unistd.h>

class ProcessUnix : public Process {
	struct ChannelUnix : public Channel {
		int pipe[2];

		void close();

		ChannelUnix();
	};

	struct ReadChannel : public ChannelUnix {
		RingBuffer<char> ring_buffer;
		Vector<char> sysread_buffer;

		int read(int p_bytes);

		bool open();
		void close();

		ReadChannel();
	};

	struct WriteChannel : public ChannelUnix {
		int write(const char *p_data, int p_max_size);

		bool open();
		void close();

		WriteChannel();
	};

	static Process *_create();

	pid_t pid;
	int forkfd;
	int start_notifier_pipe[2];

	_FORCE_INLINE_ WriteChannel *_stdin_wchan() const { return static_cast<WriteChannel *>(stdin_sama); }
	_FORCE_INLINE_ ReadChannel *_stdout_rchan() const { return static_cast<ReadChannel *>(stdout_chan); }
	_FORCE_INLINE_ ReadChannel *_stderr_rchan() const { return static_cast<ReadChannel *>(stderr_chan); }

	bool _setup_channels();
	void _wait_forkfd();
	void _death_cleanup();
	int _read_bytes_from_channel(ReadChannel *p_channel, int p_max_size);
	void _process_start_notification();
	void _process_forkfd_notification();
	Error _select_fds(int msecs);

protected:
	bool _start();
	void _get_system_env(HashMap<String, String> &r_env);

public:
	static void make_default();

	int64_t get_pid() const;

	bool wait_for_started(int msecs = 20000);
	bool wait_for_finished(int msecs = 20000);

	void terminate();
	void kill();

	Error poll();

	int get_available_bytes() const;
	int next_line_size() const;
	int read_all(char *p_data, int p_max_size);
	int read_line(char *p_data, int p_max_size);

	bool can_write(int p_size) const;
	int write(const char *p_data, int p_max_size);

	ProcessUnix();
	~ProcessUnix();
};

#endif // UNIX_ENABLED

#endif // PROCESS_UNIX_H
