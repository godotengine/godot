/*************************************************************************/
/*  process_unix.cpp                                                     */
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
#include "process_unix.h"

#ifdef UNIX_ENABLED

#include "global_config.h"
#include "os/file_access.h"
#include "os/os.h"
#include "thirdparty/forkfd/forkfd.h"

#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>

#ifndef NO_FCNTL
#ifdef __HAIKU__
#include <fcntl.h>
#else
#include <sys/fcntl.h>
#endif
#define SET_NONBLOCK(m_pipe_end)                                             \
	if (m_pipe_end != -1) {                                                  \
		fcntl(m_pipe_end, F_SETFL, fcntl(m_pipe_end, F_GETFL) | O_NONBLOCK); \
	}
#else
#define SET_NONBLOCK(m_pipe_end)           \
	if (m_pipe_end != -1) {                \
		int flag = 1;                      \
		ioctl(m_pipe_end, FIONBIO, &flag); \
	}
#endif

#define PIPE_READ 0
#define PIPE_WRITE 1

#define CLOSE_PIPE_END(m_pipe_end) \
	if (m_pipe_end != -1) {        \
		::close(m_pipe_end);       \
		m_pipe_end = -1;           \
	}

#define CURRENT_READ_CHANNEL (get_read_channel() == CHANNEL_STDERR ? _stderr_rchan() : _stdout_rchan())

static int create_pipe(int r_pipe[2]) {
	CLOSE_PIPE_END(r_pipe[PIPE_READ]);
	CLOSE_PIPE_END(r_pipe[PIPE_WRITE]);

	int ret = pipe(r_pipe);

	if (ret == -1)
		return -1;

#ifndef NO_FCNTL
	fcntl(r_pipe[PIPE_READ], F_SETFD, FD_CLOEXEC);
	fcntl(r_pipe[PIPE_WRITE], F_SETFD, FD_CLOEXEC);
#else
	ioctl(r_pipe[PIPE_READ], FIOCLEX);
	ioctl(r_pipe[PIPE_WRITE], FIOCLEX);
#endif

	return ret;
}

static int get_available_bytes_in(int p_pipe) {
	unsigned long len = 0;
	int ret = ioctl(p_pipe, FIONREAD, &len);
	ERR_FAIL_COND_V(ret == -1, 0);
	return len;
}

void ProcessUnix::make_default() {
	Process::_create = ProcessUnix::_create;
}

Process *ProcessUnix::_create() {
	return memnew(ProcessUnix);
}

int64_t ProcessUnix::get_pid() const {
	return pid;
}

bool ProcessUnix::_setup_channels() {
	if (get_redirect_flags() & REDIRECT_STDIN) {
		if (!_stdin_wchan()->open())
			goto failed_stdin;
	} else {
		dup2(STDIN_FILENO, _stdin_wchan()->pipe[PIPE_READ]);
	}

	if (get_redirect_flags() & REDIRECT_STDOUT) {
		if (!_stdout_rchan()->open())
			goto failed_stdout;
	} else {
		dup2(STDOUT_FILENO, _stdout_rchan()->pipe[PIPE_WRITE]);
	}

	if (get_redirect_flags() & REDIRECT_STDERR_TO_STDOUT) {
		_stderr_rchan()->pipe[PIPE_READ] = -1;
		_stderr_rchan()->pipe[PIPE_WRITE] = -1;
	} else if (get_redirect_flags() & REDIRECT_STDERR) {
		if (!_stderr_rchan()->open())
			goto failed_stderr;
	} else {
		dup2(STDERR_FILENO, _stderr_rchan()->pipe[PIPE_WRITE]);
	}

	if (create_pipe(start_notifier_pipe) == -1)
		goto failed_start_notifier;

	return true;

failed_start_notifier:
	_stderr_rchan()->close();
failed_stderr:
	_stdout_rchan()->close();
failed_stdout:
	_stdin_wchan()->close();
failed_stdin:
	return false;
}

bool ProcessUnix::_start() {
	// Setup channels

	if (!_setup_channels())
		return false;

	process_state = STATE_STARTING;

	// Setup arguments

	String prog = get_program();
	prog = prog.replace("\\", "/"); // No backslash, please

	// If the program is not a path, try to find it in PATH
	if (prog.find("/") == -1) {
		Vector<String> env_path = OS::get_singleton()->get_environment("PATH").split(":", false);

		for (int i = 0; i < env_path.size(); i++) {
			String p = env_path[i].plus_file(prog);

			if (FileAccess::exists(p)) {
				prog = p;
				break;
			}
		}
	} else {
		// Fix program path

		while (true) { // in case of using 2 or more slash
			String compare = prog.replace("//", "/");
			if (prog == compare)
				break;
			else
				prog = compare;
		}
	}

	Vector<CharString> argss;
	Vector<char *> argv;

	argss.push_back(prog.utf8());
	argv.push_back((char *)argss[0].get_data()); // shitty C cast ;)

	for (int i = 1; i <= get_arguments().size(); i++) {
		argss.push_back(get_arguments()[i - 1].utf8());
		argv.push_back((char *)argss[i].get_data()); // shitty C cast ;)
	}

	argv.push_back(NULL);

	// Setup environment

	const HashMap<String, String> &env = get_environment();

	int envc = env.size();

	Vector<CharString> envph;
	Vector<char *> envp;

	int envidx = 0;
	const String *k = NULL;
	while ((k = env.next(k))) {
		String env_str = *k;
		env_str += L'=';
		env_str += env.get(*k);
		env_str += L'\0';

		envph.push_back(env_str.utf8());
		envp.push_back((char *)envph[envidx].get_data());

		++envidx;
	}

	envp.push_back(NULL);

	// Fix cwd path

	String cwd = get_cwd();
	cwd = cwd.replace("\\", "/");
	while (true) { // in case of using 2 or more slash
		String compare = cwd.replace("//", "/");
		if (cwd == compare)
			break;
		else
			cwd = compare;
	}

	// Fork

	pid_t childpid;
	forkfd = ::forkfd(FFD_CLOEXEC, &childpid);

	if (forkfd == -1) {
		process_state = STATE_NOT_RUNNING;
		_death_cleanup();
		return false;
	}

	if (forkfd == FFD_CHILD_PROCESS) {
		::signal(SIGPIPE, SIG_DFL); // if the parent was ignoring SIGPIPE

		// Execute child

		dup2(_stdin_wchan()->pipe[PIPE_READ], STDIN_FILENO);
		dup2(_stdout_rchan()->pipe[PIPE_WRITE], STDOUT_FILENO);

		if (get_redirect_flags() & REDIRECT_STDERR_TO_STDOUT) {
			dup2(STDOUT_FILENO, STDERR_FILENO);
		} else {
			dup2(_stderr_rchan()->pipe[PIPE_WRITE], STDERR_FILENO);
		}

		CLOSE_PIPE_END(start_notifier_pipe[PIPE_READ]);

		CharString cwd_utf8 = cwd.utf8();

		if (cwd_utf8.length() && chdir(cwd_utf8.get_data()) != -1) { // change cwd
			if (envc > 0) {
				execve(argv[0], &argv[0], &envp[0]);
			} else {
				execvp(argv[0], &argv[0]);
			}
		}

		// exec failed, notify parent

		int errnum = errno;
		::write(start_notifier_pipe[PIPE_WRITE], &errnum, sizeof(errnum));
		CLOSE_PIPE_END(start_notifier_pipe[PIPE_WRITE]);

		abort();
	}

	pid = childpid;

	// Don't need these
	CLOSE_PIPE_END(start_notifier_pipe[PIPE_WRITE]);

	CLOSE_PIPE_END(_stdin_wchan()->pipe[PIPE_READ]);
	CLOSE_PIPE_END(_stdout_rchan()->pipe[PIPE_WRITE]);
	CLOSE_PIPE_END(_stderr_rchan()->pipe[PIPE_WRITE]);

	SET_NONBLOCK(_stdin_wchan()->pipe[PIPE_WRITE]);
	SET_NONBLOCK(_stdout_rchan()->pipe[PIPE_READ]);
	SET_NONBLOCK(_stderr_rchan()->pipe[PIPE_READ]);

	return true;
}

extern char **environ;

void ProcessUnix::_get_system_env(HashMap<String, String> &r_env) {
	for (char **current = environ; *current; current++) {
		if (const char *sep = strchr(*current, '=')) {
			String name;
			name.parse_utf8(*current, sep - (*current));
			String value;
			value.parse_utf8(sep + 1);

			r_env.set(name, value);
		}
	}
}

void ProcessUnix::_death_cleanup() {
	pid = 0;

	CLOSE_PIPE_END(start_notifier_pipe[PIPE_READ]);
	CLOSE_PIPE_END(start_notifier_pipe[PIPE_WRITE]);

	CLOSE_PIPE_END(forkfd);

	stdin_sama->close();
	stdout_chan->close();
	stderr_chan->close();
}

void ProcessUnix::_wait_forkfd() {
	if (forkfd != -1) {
		forkfd_info info;
		while (forkfd_wait(forkfd, &info, NULL) == -1 && errno == EINTR) {
		}

		exit_code = info.status;
		exit_status = info.code == CLD_EXITED ? EXIT_STATUS_NORMAL : EXIT_STATUS_CRASH;

		CLOSE_PIPE_END(forkfd);
	}
}

void ProcessUnix::_process_start_notification() {
	int exec_errnum;
	int bytes = ::read(start_notifier_pipe[PIPE_READ], &exec_errnum, sizeof(exec_errnum));

	if (bytes > 0) {
		process_state = STATE_NOT_RUNNING;
		CLOSE_PIPE_END(start_notifier_pipe[PIPE_READ]);
		ERR_PRINTS(String() + "Child exec failed to start with errno: " + String::num_int64(exec_errnum));
		_wait_forkfd();
		_death_cleanup();
	} else {
		process_state = STATE_RUNNING;
		CLOSE_PIPE_END(start_notifier_pipe[PIPE_READ]);
	}
}

void ProcessUnix::_process_forkfd_notification() {
	_wait_forkfd();

	if (process_state == STATE_STARTING) {
		wait_for_started(0);
		if (process_state == STATE_NOT_RUNNING) {
			// exec failed, we are done here
			return;
		}
	}

	_read_bytes_from_channel(_stdout_rchan(), get_available_bytes_in(_stdout_rchan()->pipe[PIPE_READ]));
	_read_bytes_from_channel(_stderr_rchan(), get_available_bytes_in(_stderr_rchan()->pipe[PIPE_READ]));

	process_state = STATE_NOT_RUNNING;

	_death_cleanup();
}

Error ProcessUnix::_select_fds(int msecs) {

#define FD_SET_AND_CHECK_HIGHEST(m_n, m_fd, m_fdset) \
	{                                                \
		FD_SET(m_fd, &m_fdset);                      \
                                                     \
		if (m_fd > m_n) {                            \
			m_n = m_fd;                              \
		}                                            \
	}

	fd_set readfds;
	fd_set writefds;
	int n = -1;

	FD_ZERO(&readfds);
	FD_ZERO(&writefds);

	if (process_state == STATE_STARTING) {
		FD_SET_AND_CHECK_HIGHEST(n, start_notifier_pipe[PIPE_READ], readfds);
	} else if (process_state == STATE_RUNNING) {
		FD_SET_AND_CHECK_HIGHEST(n, forkfd, readfds);
	}

	if (_stdout_rchan()->pipe[PIPE_READ] != -1) {
		FD_SET_AND_CHECK_HIGHEST(n, _stdout_rchan()->pipe[PIPE_READ], readfds);
	}

	if (_stderr_rchan()->pipe[PIPE_READ] != -1) {
		FD_SET_AND_CHECK_HIGHEST(n, _stderr_rchan()->pipe[PIPE_READ], readfds);
	}

	struct timeval tv;

	if (msecs != -1) {
		tv.tv_sec = msecs ? msecs / 1000 : 0;
		tv.tv_usec = msecs ? (msecs % 1000) * 1000 : 0;
	}

	if (::select(n + 1, &readfds, &writefds, NULL, msecs == -1 ? NULL : &tv) <= 0) {
		return FAILED;
	}

	if (process_state == STATE_STARTING) {
		if (FD_ISSET(start_notifier_pipe[PIPE_READ], &readfds)) {
			_process_start_notification();

			if (process_state == STATE_NOT_RUNNING) {
				return OK;
			}
		}
	}

	if (_stdout_rchan()->pipe[PIPE_READ] != -1 && FD_ISSET(_stdout_rchan()->pipe[PIPE_READ], &readfds)) {
		_read_bytes_from_channel(_stdout_rchan(), get_available_bytes_in(_stdout_rchan()->pipe[PIPE_READ]));
	}

	if (_stderr_rchan()->pipe[PIPE_READ] != -1 && FD_ISSET(_stderr_rchan()->pipe[PIPE_READ], &readfds)) {
		_read_bytes_from_channel(_stderr_rchan(), get_available_bytes_in(_stderr_rchan()->pipe[PIPE_READ]));
	}

	if (forkfd != -1 && FD_ISSET(forkfd, &readfds)) {
		_process_forkfd_notification();

		if (process_state == STATE_NOT_RUNNING) {
			return OK;
		}
	}

	return ERR_TIMEOUT;
}

Error ProcessUnix::poll() {
	ERR_FAIL_COND_V(process_state == STATE_NOT_RUNNING, ERR_UNCONFIGURED);
	Error err = _select_fds(0);
	return err == ERR_TIMEOUT ? OK : err;
}

bool ProcessUnix::wait_for_started(int msecs) {
	fd_set readfds;
	FD_ZERO(&readfds);
	FD_SET(start_notifier_pipe[PIPE_READ], &readfds);

	struct timeval tv;

	if (msecs != -1) {
		tv.tv_sec = msecs ? msecs / 1000 : 0;
		tv.tv_usec = msecs ? (msecs % 1000) * 1000 : 0;
	}

	if (::select(start_notifier_pipe[PIPE_READ] + 1, &readfds, NULL, NULL, msecs == -1 ? NULL : &tv)) {
		_process_start_notification();
		return true;
	}

	return false;
}

bool ProcessUnix::wait_for_finished(int msecs) {
	do {
		int last_tick = OS::get_singleton()->get_ticks_msec();

		Error err = _select_fds(msecs);

		if (err == OK)
			return true;
		if (err == FAILED)
			return false;

		// ERR_TIMEOUT

		if (msecs != -1) {
			int tdiff = OS::get_singleton()->get_ticks_msec() - last_tick;

			if (tdiff > msecs) {
				msecs = 0;
			} else {
				msecs -= tdiff;
			}
		}
	} while (msecs || msecs == -1);

	return false;
}

void ProcessUnix::terminate() {
	if (pid) ::kill(pid, SIGTERM);
}

void ProcessUnix::kill() {
	if (pid) ::kill(pid, SIGKILL);
}

int ProcessUnix::get_available_bytes() const {
	return CURRENT_READ_CHANNEL->ring_buffer.data_left();
}

int ProcessUnix::next_line_size() const {
	ReadChannel *channel = CURRENT_READ_CHANNEL;
	return channel->ring_buffer.find('\n', 0, channel->ring_buffer.data_left());
}

int ProcessUnix::_read_bytes_from_channel(ReadChannel *p_channel, int p_max_size) {
	if (!p_max_size)
		return 0;

	int total_read = 0;
	int to_read = p_max_size;

	do {
		int read = p_channel->read(to_read);

		if (read <= 0)
			break;

		total_read += read;
	} while (to_read && total_read < p_max_size);

	return total_read;
}

int ProcessUnix::read_all(char *p_data, int p_max_size) {
	ReadChannel *channel = CURRENT_READ_CHANNEL;
	int diff = p_max_size - channel->ring_buffer.data_left();

	if (diff > 0) {
		int to_read = MIN(get_available_bytes_in(CURRENT_READ_CHANNEL->pipe[PIPE_READ]), diff);
		p_max_size = _read_bytes_from_channel(channel, to_read);
	}

	int read = channel->ring_buffer.read(p_data, p_max_size);
	ERR_FAIL_COND_V(read != p_max_size, read);
	return read;
}

int ProcessUnix::read_line(char *p_data, int p_max_size) {
	ReadChannel *channel = CURRENT_READ_CHANNEL;
	int diff = p_max_size - channel->ring_buffer.data_left();

	if (diff > 0) {
		p_max_size = _read_bytes_from_channel(channel, MIN(next_line_size(), diff));
	}

	int read = channel->ring_buffer.read(p_data, p_max_size);
	ERR_FAIL_COND_V(read != p_max_size, read);
	return read;
}

bool ProcessUnix::can_write(int) const {
	return !stdin_sama->closed;
}

int ProcessUnix::write(const char *p_data, int p_max_size) {
	return _stdin_wchan()->write(p_data, p_max_size);
}

ProcessUnix::ProcessUnix()
	: Process(memnew(WriteChannel), memnew(ReadChannel), memnew(ReadChannel)) {
	pid = 0;
	forkfd = -1;
	start_notifier_pipe[PIPE_READ] = -1;
	start_notifier_pipe[PIPE_WRITE] = -1;
}

ProcessUnix::~ProcessUnix() {
	_death_cleanup();
}

int ProcessUnix::ReadChannel::read(int p_bytes) {
	ERR_FAIL_COND_V(closed, -1);

	int to_read = MIN(p_bytes, ring_buffer.space_left());

	if (to_read) {
		int read = ::read(pipe[PIPE_READ], sysread_buffer.ptr(), to_read);

		if (read > 0) {
			int stored = ring_buffer.write(sysread_buffer.ptr(), read);
			ERR_FAIL_COND_V(stored != read, stored);
		}
		if (read == -1 && errno != EWOULDBLOCK) {
			ERR_PRINTS(String() + "::read failed with error code: " + String::num_int64(errno));
		} else if (read == 0) {
			// eof
			close();
		}

		return read;
	}

	return 0;
}

bool ProcessUnix::ReadChannel::open() {
	ring_buffer.clear();

	if (create_pipe(pipe) != -1) {
		closed = false;
		return true;
	}

	return false;
}

void ProcessUnix::ReadChannel::close() {
	if (!closed)
		closed = true;

	ChannelUnix::close(); // even if closed, may be a placeholder for pipes
}

ProcessUnix::ReadChannel::ReadChannel() {
	int rbsize = GLOBAL_GET("os/process_max_read_buffer_po2");

	ring_buffer.resize(rbsize);
	sysread_buffer.resize(1 << rbsize);
}

int ProcessUnix::WriteChannel::write(const char *p_data, int p_max_size) {
	if (closed)
		return -1;

	int wrote = ::write(pipe[PIPE_WRITE], p_data, p_max_size);

	if (wrote == -1) {
		if (errno == EAGAIN) {
			return 0;
		}

		close();
		return -1;
	}

	return wrote;
}

bool ProcessUnix::WriteChannel::open() {
	if (create_pipe(pipe) != -1) {
		closed = false;
		return true;
	}

	return false;
}

void ProcessUnix::WriteChannel::close() {
	if (!closed)
		closed = true;

	ChannelUnix::close(); // even if closed, may be a placeholder for pipes
}

ProcessUnix::WriteChannel::WriteChannel() {}

void ProcessUnix::ChannelUnix::close() {
	CLOSE_PIPE_END(pipe[PIPE_READ]);
	CLOSE_PIPE_END(pipe[PIPE_WRITE]);
}

ProcessUnix::ChannelUnix::ChannelUnix() {
	pipe[PIPE_READ] = -1;
	pipe[PIPE_WRITE] = -1;
}

#endif // UNIX_ENABLED
