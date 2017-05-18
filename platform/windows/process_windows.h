/*************************************************************************/
/*  process_windows.h                                                    */
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
#ifndef PROCESS_WINDOWS_H
#define PROCESS_WINDOWS_H

#include "os/process.h"
#include "ring_buffer.h"

#include <windows.h>

class ProcessWindows : public Process {
	struct ChannelWindows : public Channel {
		struct IOContext // https://blogs.msdn.microsoft.com/oldnewthing/20101217-00/?p=11983
		{
			OVERLAPPED overlapped;
			ChannelWindows *channel;
		};

		HANDLE pipe[2];
		IOContext io_context;

		void close();

		static void CALLBACK completion_routine(DWORD dwErrorCode, DWORD dwNumberOfBytesTransfered, OVERLAPPED *lpOverlapped);

		virtual void async_callback(DWORD dwErrorCode, DWORD dwNumberOfBytesTransfered) = 0;

		ChannelWindows();
		~ChannelWindows();
	};

	struct ReadChannel : public ChannelWindows {
		RingBuffer<char> ring_buffer;
		Vector<char> readfile_buffer;
		bool reading;
		bool broken_pipe;
		bool read_completed;

		void read_async();
		void async_callback(DWORD dwErrorCode, DWORD dwNumberOfBytesTransfered);

		DWORD peek();

		bool open();
		void close();

		ReadChannel();
	};

	struct WriteChannel : public ChannelWindows {
		RingBuffer<char> ring_buffer;
		Vector<char> writefile_buffer;
		uint64_t bytes_to_write;
		bool writing;

		Error write_async();
		void async_callback(DWORD dwErrorCode, DWORD dwNumberOfBytesTransfered);

		bool open();
		void close();

		WriteChannel();
	};

	static Process *_create();

	LPPROCESS_INFORMATION lpProcessInfo;

	_FORCE_INLINE_ WriteChannel *_stdin_wchan() const { return static_cast<WriteChannel *>(stdin_sama); }
	_FORCE_INLINE_ ReadChannel *_stdout_rchan() const { return static_cast<ReadChannel *>(stdout_chan); }
	_FORCE_INLINE_ ReadChannel *_stderr_rchan() const { return static_cast<ReadChannel *>(stderr_chan); }

	bool _setup_channels();

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

	ProcessWindows();
	~ProcessWindows();
};

#endif // PROCESS_WINDOWS_H
