/*************************************************************************/
/*  process_windows.cpp                                                  */
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
#include "process_windows.h"

#include "global_config.h"
#include "os/dir_access.h"
#include "os/os.h"
#include "pair.h"

#ifndef PIPE_REJECT_REMOTE_CLIENTS
#define PIPE_REJECT_REMOTE_CLIENTS 0x00000008
#endif

#define PIPE_READ 0
#define PIPE_WRITE 1

#define CLEAN_PROCESS_INFO(m_proc_info)     \
	if (m_proc_info) {                      \
		CloseHandle(m_proc_info->hThread);  \
		CloseHandle(m_proc_info->hProcess); \
		memdelete(m_proc_info);             \
		m_proc_info = NULL;                 \
	}

#define CLOSE_HANDLE(m_handle)              \
	if (m_handle != INVALID_HANDLE_VALUE) { \
		CloseHandle(m_handle);              \
		m_handle = INVALID_HANDLE_VALUE;    \
	}

#define CURRENT_READ_CHANNEL (get_read_channel() == CHANNEL_STDERR ? _stderr_rchan() : _stdout_rchan())

enum PipeType {
	PIPE_TYPE_INPUT,
	PIPE_TYPE_OUTPUT
};

// Based on MyCreatePipeEx
static BOOL CreateOverlappedPipe(LPHANDLE lpPipeRead, LPHANDLE lpPipeWrite, PipeType p_type) {
	String pipe_name;
	HANDLE hReadPipe;

	SECURITY_ATTRIBUTES secAttrs;
	secAttrs.nLength = sizeof(secAttrs);
	secAttrs.lpSecurityDescriptor = NULL;
	secAttrs.bInheritHandle = (p_type == PIPE_TYPE_INPUT);

	uint32_t tries_left = 100;
	do {
		pipe_name = "\\\\.\\pipe\\godot-";
		pipe_name += String::num_int64(Math::rand(), 16, true);

		hReadPipe = CreateNamedPipeW(
				pipe_name.c_str(),
				PIPE_ACCESS_INBOUND | FILE_FLAG_OVERLAPPED,
				PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
				1, // nMaxInstances
				0, 1 << 20, // nOutBufferSize, nInBufferSize
				0, // nDefaultTimeOut = 50ms
				&secAttrs);
	} while (hReadPipe == INVALID_HANDLE_VALUE &&
			 GetLastError() == ERROR_PIPE_BUSY && --tries_left > 0);

	ERR_EXPLAIN("CreateNamedPipe failed with error code: " + String::num_int64(GetLastError()));
	ERR_FAIL_COND_V(hReadPipe == INVALID_HANDLE_VALUE, FALSE);

	secAttrs.bInheritHandle = !secAttrs.bInheritHandle;

	HANDLE hWritePipe = CreateFileW(
			pipe_name.c_str(), GENERIC_WRITE,
			0, // dwShareMode
			&secAttrs,
			OPEN_EXISTING, FILE_FLAG_OVERLAPPED,
			NULL // hTemplateFile
			);

	if (hWritePipe == INVALID_HANDLE_VALUE) {
		ERR_PRINTS(String() + "CreateNamedPipe failed with error code: " + String::num_int64(GetLastError()));
		CloseHandle(hReadPipe);
		return FALSE;
	}

	ConnectNamedPipe(hReadPipe, NULL);

	*lpPipeRead = hReadPipe;
	*lpPipeWrite = hWritePipe;

	return TRUE;
}

static String environment_to_string(const HashMap<String, String> &p_env_map) {
	String env_str;

	const String *k = NULL;
	while ((k = p_env_map.next(k))) {
		env_str += *k;
		env_str += L'=';
		env_str += p_env_map.get(*k);
		env_str += L'\0';
	}

	env_str += L'\0';
	env_str += L'\0';

	return env_str;
}

void ProcessWindows::make_default() {
	Process::_create = ProcessWindows::_create;
}

Process *ProcessWindows::_create() {
	return memnew(ProcessWindows);
}

int64_t ProcessWindows::get_pid() const {
	if (!lpProcessInfo)
		return 0;
	return lpProcessInfo->dwProcessId;
}

bool ProcessWindows::_setup_channels() {
	HANDLE hCurrentProcess = GetCurrentProcess();

	if (get_redirect_flags() & REDIRECT_STDIN) {
		if (!_stdin_wchan()->open())
			goto failed_stdin;
	} else {
		if (!DuplicateHandle(hCurrentProcess, GetStdHandle(STD_INPUT_HANDLE),
					hCurrentProcess, &_stdin_wchan()->pipe[PIPE_READ],
					0, TRUE, DUPLICATE_SAME_ACCESS)) {
			ERR_PRINTS("DuplicateHandle failed with error code: " + String::num_int64(GetLastError()));
			goto failed_stdin;
		}
	}

	if (get_redirect_flags() & REDIRECT_STDOUT) {
		if (!_stdout_rchan()->open())
			goto failed_stdout;
	} else {
		if (!DuplicateHandle(hCurrentProcess, GetStdHandle(STD_OUTPUT_HANDLE),
					hCurrentProcess, &_stdout_rchan()->pipe[PIPE_WRITE],
					0, TRUE, DUPLICATE_SAME_ACCESS)) {
			ERR_PRINTS("DuplicateHandle failed with error code: " + String::num_int64(GetLastError()));
			goto failed_stdout;
		}
	}

	if (get_redirect_flags() & REDIRECT_STDERR_TO_STDOUT) {
		HANDLE hCurrentProcess = GetCurrentProcess();
		if (!DuplicateHandle(hCurrentProcess, _stdout_rchan()->pipe[PIPE_WRITE],
					hCurrentProcess, &_stderr_rchan()->pipe[PIPE_WRITE],
					0, TRUE, DUPLICATE_SAME_ACCESS)) {
			ERR_PRINTS("DuplicateHandle failed with error code: " + String::num_int64(GetLastError()));
			return false;
		}
	} else if (get_redirect_flags() & REDIRECT_STDERR) {
		if (!_stderr_rchan()->open())
			goto failed_stderr;
	} else {
		if (!DuplicateHandle(hCurrentProcess, GetStdHandle(STD_ERROR_HANDLE),
					hCurrentProcess, &_stderr_rchan()->pipe[PIPE_WRITE],
					0, TRUE, DUPLICATE_SAME_ACCESS)) {
			ERR_PRINTS("DuplicateHandle failed with error code: " + String::num_int64(GetLastError()));
			goto failed_stderr;
		}
	}

	return true;

failed_stderr:
	_stdout_rchan()->close();
failed_stdout:
	_stdin_wchan()->close();
failed_stdin:
	return false;
}

bool ProcessWindows::_start() {
	CLEAN_PROCESS_INFO(lpProcessInfo);

	lpProcessInfo = memnew(PROCESS_INFORMATION);
	zeromem(lpProcessInfo, sizeof(PROCESS_INFORMATION));

	// Setup channels and redirections

	if (!_setup_channels())
		return false;

	process_state = STATE_STARTING;

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	// Fix program path

	String prog = get_program();
	prog = prog.replace("/", "\\");
	while (true) { // in case of using 2 or more backslash
		String compare = prog.replace("\\\\", "\\");
		if (prog == compare)
			break;
		else
			prog = compare;
	}
	if (prog.begins_with("\\")) {
		int current_drive = da->get_current_drive();
		prog = da->get_drive(current_drive) + prog;
	}

	// Setup arguments

	String argss;
	argss = "\"" + prog + "\"";
	for (int i = 0; i < get_arguments().size(); i++) {
		const String &arg = get_arguments()[i];

		if (arg.begins_with("\"") || arg.ends_with("\"")) {
			argss += String(" ");
			argss += arg;
		} else {
			argss += String(" \"");
			argss += arg;
			argss += "\"";
		}
	}

	// Fix cwd path

	String cwd = get_cwd();
	prog = prog.replace("/", "\\");
	while (true) { // in case of using 2 or more backslash
		String compare = cwd.replace("\\\\", "\\");
		if (cwd == compare)
			break;
		else
			cwd = compare;
	}
	if (cwd.begins_with("\\")) {
		int current_drive = da->get_current_drive();
		cwd = da->get_drive(current_drive) + cwd;
	}

	// Finally execute it

	DWORD dwCreationFlags = (GetConsoleWindow() ? 0 : CREATE_NO_WINDOW) | CREATE_UNICODE_ENVIRONMENT;
	STARTUPINFOW startupInfo = {
		sizeof(STARTUPINFO), // cb
		0, // lpReserved
		0, 0, // lpDesktop, lpTitle
		(DWORD)CW_USEDEFAULT, (DWORD)CW_USEDEFAULT, // dwX, dwY
		(DWORD)CW_USEDEFAULT, (DWORD)CW_USEDEFAULT, // dwXSize, dwYSize
		0, 0, // dwXCountChars, dwYCountChars
		0, // dwFillAttribute
		STARTF_USESTDHANDLES, // dwFlags
		0, // wShowWindow
		0, 0, // cbReserved2, lpReserved2
		_stdin_wchan()->pipe[PIPE_READ], _stdout_rchan()->pipe[PIPE_WRITE], _stderr_rchan()->pipe[PIPE_WRITE]
	};

	String env_str;
	if (get_environment().size()) {
		env_str = environment_to_string(get_environment());
	}

	BOOL success = CreateProcessW(NULL, (LPWSTR)argss.c_str(), NULL, NULL, TRUE, dwCreationFlags,
			env_str.size() ? (LPWSTR)env_str.c_str() : NULL,
			cwd.size() ? (LPWSTR)cwd.c_str() : NULL,
			&startupInfo, lpProcessInfo);

	if (!success) {
		process_state = STATE_NOT_RUNNING;

		stdin_sama->close();
		stdout_chan->close();
		stderr_chan->close();

		return false;
	}

	// Don't need these
	CLOSE_HANDLE(_stdin_wchan()->pipe[PIPE_READ]);
	CLOSE_HANDLE(_stdout_rchan()->pipe[PIPE_WRITE]);
	CLOSE_HANDLE(_stderr_rchan()->pipe[PIPE_WRITE]);

	process_state = STATE_RUNNING;

	return true;
}

void ProcessWindows::_get_system_env(HashMap<String, String> &r_env) {
	WCHAR *env_strw = GetEnvironmentStringsW();

	if (env_strw) {
		const WCHAR *elem = env_strw;
		while (*elem) {
			const int elem_length = int(wcslen(elem));

			if (const WCHAR *sep = wcschr(elem + 1, L'=')) {
				int name_length = sep - elem;

				String name(elem, name_length);
				String value(sep + 1, elem_length - name_length - 1);

				r_env.set(name, value);
			}

			elem += elem_length + 1;
		}

		FreeEnvironmentStringsW(env_strw);
	}
}

Error ProcessWindows::poll() {
	ERR_FAIL_COND_V(get_state() == STATE_NOT_RUNNING, ERR_UNCONFIGURED);

	DWORD dwExitCode = 0;

	if (!GetExitCodeProcess(lpProcessInfo->hProcess, &dwExitCode)) {
		ERR_EXPLAIN(String() + "GetExitCodeProcess failed with error code: " + String::num_int64(GetLastError()));
		ERR_FAIL_V(ERR_BUG);
	} else if (dwExitCode == STILL_ACTIVE) {
		// Still alive // Thanks captain obvious

		// If already reading/writing, give the completion callbacks an oportunity to be called
		// If not reading, you better start doing it

		WriteChannel *stdin_wchan = _stdin_wchan();

		if (!stdin_wchan->closed) {
			if (stdin_wchan->writing)
				SleepEx(0, TRUE);
			else if (stdin_wchan->ring_buffer.data_left() > 0)
				stdin_wchan->write_async();
		}

		ReadChannel *stdout_rchan = _stdout_rchan();

		if (!stdout_rchan->closed && !stdout_rchan->broken_pipe) {
			if (stdout_rchan->reading) {
				SleepEx(0, TRUE);
			} else {
				stdout_rchan->read_async();
			}
		}

		ReadChannel *stderr_rchan = _stderr_rchan();

		if (!stderr_rchan->closed && !stderr_rchan->broken_pipe) {
			if (stderr_rchan->reading) {
				SleepEx(0, TRUE);
			} else {
				stderr_rchan->read_async();
			}
		}
	} else {
		// The process finished

		process_state = STATE_NOT_RUNNING;

		exit_code = dwExitCode;

		if (/* HRESULT */ dwExitCode >= 0x80000000 && dwExitCode < 0xD0000000) { /* SEH exception >= 0xC0000000 */
			exit_status = EXIT_STATUS_CRASH;
		} else {
			exit_status = EXIT_STATUS_NORMAL;
		}

		// Wait for read routines to finish

		ReadChannel *stdout_rchan = _stdout_rchan();
		ReadChannel *stderr_rchan = _stderr_rchan();

		bool reading = false;
		bool routine_completed = false;

		do {
			reading = false;
			routine_completed = false;

			if (stdout_rchan->reading) {
				stdout_rchan->read_completed = false;
				SleepEx(0, TRUE);
				routine_completed |= stdout_rchan->read_completed;
			}

			reading |= stdout_rchan->reading;

			if (stderr_rchan->reading) {
				stderr_rchan->read_completed = false;
				SleepEx(0, TRUE);
				routine_completed |= stderr_rchan->read_completed;
			}

			reading |= stderr_rchan->reading;
		} while (reading && routine_completed);

		// Clean

		stdout_chan->close();
		stderr_chan->close();
		stdin_sama->close();

		CLEAN_PROCESS_INFO(lpProcessInfo);
	}

	return OK;
}

bool ProcessWindows::wait_for_started(int) {
	// no need to wait for anything on windows
	// start returns with either a running or not running state
	return get_state() == STATE_RUNNING;
}

bool ProcessWindows::wait_for_finished(int msecs) {
	uint64_t usecs_left = msecs * 1000;

	do {
		uint64_t last_tick = OS::get_singleton()->get_ticks_usec();

		DWORD retcode = WaitForSingleObject(lpProcessInfo->hProcess, 10);

		poll();

		if (!lpProcessInfo || retcode == WAIT_OBJECT_0)
			return true;

		if (msecs != -1) {
			uint64_t tdiff = OS::get_singleton()->get_ticks_usec() - last_tick;

			if (tdiff > usecs_left) {
				usecs_left = 0;
			} else {
				usecs_left -= tdiff;
			}
		}
	} while (usecs_left || msecs == -1);

	return false;
}

BOOL CALLBACK TerminateAppEnum(HWND hwnd, LPARAM lParam) {
	DWORD dwID;

	GetWindowThreadProcessId(hwnd, &dwID);

	if (dwID == (DWORD)lParam) {
		PostMessage(hwnd, WM_CLOSE, 0, 0);
	}

	return TRUE;
}

void ProcessWindows::terminate() {
	ERR_FAIL_COND(!lpProcessInfo);
	EnumWindows((WNDENUMPROC)TerminateAppEnum, (LPARAM)lpProcessInfo->dwProcessId);
}

void ProcessWindows::kill() {
	ERR_FAIL_COND(!lpProcessInfo);
	TerminateProcess(lpProcessInfo->hProcess, 1);
}

int ProcessWindows::get_available_bytes() const {
	return CURRENT_READ_CHANNEL->ring_buffer.data_left();
}

int ProcessWindows::next_line_size() const {
	ReadChannel *channel = CURRENT_READ_CHANNEL;
	return channel->ring_buffer.find('\n', 0, channel->ring_buffer.data_left());
}

int ProcessWindows::read_all(char *p_data, int p_max_size) {
	ReadChannel *channel = CURRENT_READ_CHANNEL;
	int to_read = MIN(get_available_bytes(), p_max_size);

	if (to_read) {
		int read = channel->ring_buffer.read(p_data, to_read);
		ERR_FAIL_COND_V(read != to_read, read);
		return read;
	} else if (!channel->reading) {
		channel->read_async();
	}

	return 0;
}

int ProcessWindows::read_line(char *p_data, int p_max_size) {
	ReadChannel *channel = CURRENT_READ_CHANNEL;
	int to_read = MIN(next_line_size(), p_max_size);

	if (to_read) {
		int read = channel->ring_buffer.read(p_data, to_read);
		ERR_FAIL_COND_V(read != to_read, read);
		return read;
	} else if (!channel->reading) {
		channel->read_async();
	}

	return 0;
}

bool ProcessWindows::can_write(int p_size) const {
	return !stdin_sama->closed && (p_size == -1 || _stdin_wchan()->ring_buffer.space_left() >= p_size);
}

int ProcessWindows::write(const char *p_data, int p_max_size) {
	if (stdin_sama->closed)
		return -1;

	int to_write = MIN(_stdin_wchan()->ring_buffer.space_left(), p_max_size);

	if (!to_write)
		return 0;

	int wrote = _stdin_wchan()->ring_buffer.write(p_data, to_write);
	ERR_FAIL_COND_V(wrote != to_write, wrote);
	_stdin_wchan()->write_async();

	return wrote;
}

ProcessWindows::ProcessWindows()
	: Process(memnew(WriteChannel), memnew(ReadChannel), memnew(ReadChannel)) {
	lpProcessInfo = NULL;
}

ProcessWindows::~ProcessWindows() {
	stdin_sama->close();
	stdout_chan->close();
	stderr_chan->close();

	CLEAN_PROCESS_INFO(lpProcessInfo);
}

void ProcessWindows::ReadChannel::read_async() {
	const DWORD dwMinReadBufferSize = 4096;

	DWORD dwBytesToRead = peek();

	if (dwBytesToRead < dwMinReadBufferSize)
		dwBytesToRead = dwMinReadBufferSize;

	if (broken_pipe)
		return;

	dwBytesToRead = MIN(dwBytesToRead, ring_buffer.space_left());

	if (dwBytesToRead == 0)
		return;

	closed = false;
	reading = true;
	read_completed = false;

	zeromem(&(io_context.overlapped), sizeof(OVERLAPPED));

	if (!ReadFileEx(pipe[PIPE_READ], readfile_buffer.ptr(), dwBytesToRead, &io_context.overlapped,
				(LPOVERLAPPED_COMPLETION_ROUTINE)&completion_routine)) {
		reading = false;
		read_completed = true;

		DWORD dwError = GetLastError();

		if (dwError == ERROR_BROKEN_PIPE || dwError == ERROR_PIPE_NOT_CONNECTED) {
			broken_pipe = true;
		} else {
			ERR_PRINTS("ReadFileEx failed with error code: " + String::num_int64(dwError));
		}
	}
}

void ProcessWindows::ReadChannel::async_callback(DWORD dwErrorCode, DWORD dwNumberOfBytesTransfered) {
	reading = false;
	read_completed = true;

	switch (dwErrorCode) {
		case ERROR_SUCCESS:
		case ERROR_MORE_DATA:
			break;
		case ERROR_OPERATION_ABORTED: {
			if (!closed) {
				ERR_PRINTS(String() + "ReadFileEx operation aborted");
				broken_pipe = true;
			}
		} break;
		case ERROR_BROKEN_PIPE:
		case ERROR_PIPE_NOT_CONNECTED: {
			broken_pipe = true;
		} break;
		default: {
			ERR_PRINTS(String() + "ReadFileEx completed with error code: " + String::num_int64(dwErrorCode));
			broken_pipe = true;
		} break;
	}

	int stored = ring_buffer.write(readfile_buffer.ptr(), dwNumberOfBytesTransfered);
	ERR_FAIL_COND((DWORD)stored != dwNumberOfBytesTransfered);

	if (closed || broken_pipe)
		return;

	read_async();
}

DWORD ProcessWindows::ReadChannel::peek() {
	DWORD dwTotalBytesAvail = 0;

	if (!PeekNamedPipe(pipe[PIPE_READ], NULL, 0, 0, &dwTotalBytesAvail, 0)) {
		if (!broken_pipe)
			broken_pipe = true;

		return 0;
	}

	return dwTotalBytesAvail;
}

bool ProcessWindows::ReadChannel::open() {
	ring_buffer.clear();

	if (CreateOverlappedPipe(&pipe[PIPE_READ], &pipe[PIPE_WRITE], PIPE_TYPE_OUTPUT)) {
		closed = false;
		broken_pipe = false;
		read_async();
		return true;
	}

	return false;
}

void ProcessWindows::ReadChannel::close() {
	if (!closed) {
		closed = true;

		if (reading) {
			if (!CancelIoEx(pipe[PIPE_READ], &io_context.overlapped)) {
				DWORD dwError = GetLastError();
				if (dwError != ERROR_NOT_FOUND) {
					ERR_PRINTS(String() + "CancelIoEx failed on handle " +
							   String::num_int64((int64_t)pipe[PIPE_READ], 16) + " with error code: " +
							   String::num_int64(dwError));
				}
			}

			while (SleepEx(INFINITE, TRUE) == WAIT_IO_COMPLETION && reading) {
			}
		}
	}

	ChannelWindows::close(); // even if closed, may be a placeholder for pipes
}

ProcessWindows::ReadChannel::ReadChannel() {
	broken_pipe = false;
	reading = false;
	read_completed = true;

	int rbsize = GLOBAL_GET("os/process_max_read_buffer_po2");

	ring_buffer.resize(rbsize);
	readfile_buffer.resize(1 << rbsize);
}

Error ProcessWindows::WriteChannel::write_async() {
	if (writing)
		return ERR_BUSY;

	bytes_to_write = MIN(writefile_buffer.size(), ring_buffer.data_left());
	ring_buffer.read(writefile_buffer.ptr(), bytes_to_write, false);

	writing = true;
	closed = false;

	zeromem(&(io_context.overlapped), sizeof(OVERLAPPED));

	if (!WriteFileEx(pipe[PIPE_WRITE], writefile_buffer.ptr(), bytes_to_write, &io_context.overlapped,
				(LPOVERLAPPED_COMPLETION_ROUTINE)&completion_routine)) {
		writing = false;
		bytes_to_write = 0;
		writefile_buffer.clear();

		ERR_PRINTS("WriteFileEx failed with error code: " + String::num_int64(GetLastError()));

		return FAILED;
	}

	return OK;
}

void ProcessWindows::WriteChannel::async_callback(DWORD dwErrorCode, DWORD dwNumberOfBytesTransfered) {
	DWORD expected_to_write = DWORD(bytes_to_write);
	bytes_to_write = 0;
	writing = false;

	switch (dwErrorCode) {
		case ERROR_SUCCESS: {
			ring_buffer.advance_read(dwNumberOfBytesTransfered);
			ERR_FAIL_COND(dwNumberOfBytesTransfered != expected_to_write);
		} break;
		case ERROR_OPERATION_ABORTED: {
			if (!closed)
				ERR_PRINT("WriteFileEx operation aborted");
		} break;
		default: {
			ERR_PRINTS(String() + "WriteFileEx completed with error code: " + String::num_int64(dwErrorCode));
		} break;
	}

	if (closed)
		return;

	if (ring_buffer.data_left() > 0)
		write_async();
}

bool ProcessWindows::WriteChannel::open() {
	ring_buffer.clear();

	if (CreateOverlappedPipe(&pipe[PIPE_READ], &pipe[PIPE_WRITE], PIPE_TYPE_INPUT)) {
		closed = false;
		return true;
	}

	return false;
}

void ProcessWindows::WriteChannel::close() {
	if (!closed) {
		closed = true;

		if (writing) {
			if (!CancelIoEx(pipe[PIPE_WRITE], &io_context.overlapped)) {
				DWORD dwError = GetLastError();
				if (dwError != ERROR_NOT_FOUND) {
					ERR_PRINTS(String() + "CancelIoEx failed on handle " + String::num_int64((int64_t)pipe[PIPE_WRITE], 16) +
							   " with error code: " + String::num_int64(dwError));
				}
			}

			while (SleepEx(INFINITE, TRUE) == WAIT_IO_COMPLETION && writing) {
			}
		}
	}

	ChannelWindows::close(); // even if closed, may be a placeholder for pipes
}

ProcessWindows::WriteChannel::WriteChannel() {
	writing = false;

	int wbsize = GLOBAL_GET("os/process_max_write_buffer_po2");

	ring_buffer.resize(wbsize);
	writefile_buffer.resize(1 << wbsize);
}

void ProcessWindows::ChannelWindows::close() {
	CLOSE_HANDLE(pipe[PIPE_READ]);
	CLOSE_HANDLE(pipe[PIPE_WRITE]);
}

void ProcessWindows::ChannelWindows::completion_routine(DWORD dwErrorCode, DWORD dwNumberOfBytesTransfered, OVERLAPPED *lpOverlapped) {
	IOContext *c = ((IOContext *)lpOverlapped);
	c->channel->async_callback(dwErrorCode, dwNumberOfBytesTransfered);
}

ProcessWindows::ChannelWindows::ChannelWindows() {
	pipe[PIPE_READ] = INVALID_HANDLE_VALUE;
	pipe[PIPE_WRITE] = INVALID_HANDLE_VALUE;

	io_context.channel = this;

	zeromem(&(io_context.overlapped), sizeof(OVERLAPPED));
}

ProcessWindows::ChannelWindows::~ChannelWindows() {
	io_context.channel = NULL;
}
