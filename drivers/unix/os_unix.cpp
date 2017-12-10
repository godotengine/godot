/*************************************************************************/
/*  os_unix.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "os_unix.h"

#ifdef UNIX_ENABLED

#include "servers/visual_server.h"

#include "core/os/thread_dummy.h"
#include "mutex_posix.h"
#include "rw_lock_posix.h"
#include "semaphore_posix.h"
#include "thread_posix.h"

//#include "core/io/file_access_buffered_fa.h"
#include "dir_access_unix.h"
#include "file_access_unix.h"
#include "packet_peer_udp_posix.h"
#include "stream_peer_tcp_posix.h"
#include "tcp_server_posix.h"

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#if defined(__FreeBSD__) || defined(__OpenBSD__)
#include <sys/param.h>
#endif
#include "project_settings.h"
#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

void OS_Unix::debug_break() {

	assert(false);
};

int OS_Unix::get_audio_driver_count() const {

	return 1;
}
const char *OS_Unix::get_audio_driver_name(int p_driver) const {

	return "dummy";
}

int OS_Unix::unix_initialize_audio(int p_audio_driver) {

	return 0;
}

// Very simple signal handler to reap processes where ::execute was called with
// !p_blocking
void handle_sigchld(int sig) {
	int saved_errno = errno;
	while (waitpid((pid_t)(-1), 0, WNOHANG) > 0) {
	}
	errno = saved_errno;
}

void OS_Unix::initialize_core() {

#ifdef NO_PTHREADS
	ThreadDummy::make_default();
	SemaphoreDummy::make_default();
	MutexDummy::make_default();
#else
	ThreadPosix::make_default();
	SemaphorePosix::make_default();
	MutexPosix::make_default();
	RWLockPosix::make_default();
#endif
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_RESOURCES);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_FILESYSTEM);
	//FileAccessBufferedFA<FileAccessUnix>::make_default();
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_FILESYSTEM);

#ifndef NO_NETWORK
	TCPServerPosix::make_default();
	StreamPeerTCPPosix::make_default();
	PacketPeerUDPPosix::make_default();
	IP_Unix::make_default();
#endif

	ticks_start = 0;
	ticks_start = get_ticks_usec();

	struct sigaction sa;
	sa.sa_handler = &handle_sigchld;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
	if (sigaction(SIGCHLD, &sa, 0) == -1) {
		perror("ERROR sigaction() failed:");
	}
}

void OS_Unix::finalize_core() {
}

void OS_Unix::alert(const String &p_alert, const String &p_title) {

	fprintf(stderr, "ERROR: %s\n", p_alert.utf8().get_data());
}

String OS_Unix::get_stdin_string(bool p_block) {

	if (p_block) {
		char buff[1024];
		String ret = stdin_buf + fgets(buff, 1024, stdin);
		stdin_buf = "";
		return ret;
	}

	return "";
}

String OS_Unix::get_name() {

	return "Unix";
}

uint64_t OS_Unix::get_unix_time() const {

	return time(NULL);
};

uint64_t OS_Unix::get_system_time_secs() const {
	struct timeval tv_now;
	gettimeofday(&tv_now, NULL);
	return uint64_t(tv_now.tv_sec);
}

OS::Date OS_Unix::get_date(bool utc) const {

	time_t t = time(NULL);
	struct tm *lt;
	if (utc)
		lt = gmtime(&t);
	else
		lt = localtime(&t);
	Date ret;
	ret.year = 1900 + lt->tm_year;
	// Index starting at 1 to match OS_Unix::get_date
	//   and Windows SYSTEMTIME and tm_mon follows the typical structure
	//   of 0-11, noted here: http://www.cplusplus.com/reference/ctime/tm/
	ret.month = (Month)(lt->tm_mon + 1);
	ret.day = lt->tm_mday;
	ret.weekday = (Weekday)lt->tm_wday;
	ret.dst = lt->tm_isdst;

	return ret;
}

OS::Time OS_Unix::get_time(bool utc) const {
	time_t t = time(NULL);
	struct tm *lt;
	if (utc)
		lt = gmtime(&t);
	else
		lt = localtime(&t);
	Time ret;
	ret.hour = lt->tm_hour;
	ret.min = lt->tm_min;
	ret.sec = lt->tm_sec;
	get_time_zone_info();
	return ret;
}

OS::TimeZoneInfo OS_Unix::get_time_zone_info() const {
	time_t t = time(NULL);
	struct tm *lt = localtime(&t);
	char name[16];
	strftime(name, 16, "%Z", lt);
	name[15] = 0;
	TimeZoneInfo ret;
	ret.name = name;

	char bias_buf[16];
	strftime(bias_buf, 16, "%z", lt);
	int bias;
	bias_buf[15] = 0;
	sscanf(bias_buf, "%d", &bias);

	// convert from ISO 8601 (1 minute=1, 1 hour=100) to minutes
	int hour = (int)bias / 100;
	int minutes = bias % 100;
	if (bias < 0)
		ret.bias = hour * 60 - minutes;
	else
		ret.bias = hour * 60 + minutes;

	return ret;
}

void OS_Unix::delay_usec(uint32_t p_usec) const {

	struct timespec rem = { p_usec / 1000000, (p_usec % 1000000) * 1000 };
	while (nanosleep(&rem, &rem) == EINTR) {
	}
}
uint64_t OS_Unix::get_ticks_usec() const {

	struct timeval tv_now;
	gettimeofday(&tv_now, NULL);

	uint64_t longtime = (uint64_t)tv_now.tv_usec + (uint64_t)tv_now.tv_sec * 1000000L;
	longtime -= ticks_start;

	return longtime;
}

Error OS_Unix::execute(const String &p_path, const List<String> &p_arguments, bool p_blocking, ProcessID *r_child_id, String *r_pipe, int *r_exitcode, bool read_stderr) {

	if (p_blocking && r_pipe) {

		String argss;
		argss = "\"" + p_path + "\"";

		for (int i = 0; i < p_arguments.size(); i++) {

			argss += String(" \"") + p_arguments[i] + "\"";
		}

		if (read_stderr) {
			argss += " 2>&1"; // Read stderr too
		} else {
			argss += " 2>/dev/null"; //silence stderr
		}
		FILE *f = popen(argss.utf8().get_data(), "r");

		ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);

		char buf[65535];
		while (fgets(buf, 65535, f)) {

			(*r_pipe) += buf;
		}

		int rv = pclose(f);
		if (r_exitcode)
			*r_exitcode = rv;

		return OK;
	}

	pid_t pid = fork();
	ERR_FAIL_COND_V(pid < 0, ERR_CANT_FORK);

	if (pid == 0) {
		// is child
		Vector<CharString> cs;
		cs.push_back(p_path.utf8());
		for (int i = 0; i < p_arguments.size(); i++)
			cs.push_back(p_arguments[i].utf8());

		Vector<char *> args;
		for (int i = 0; i < cs.size(); i++)
			args.push_back((char *)cs[i].get_data()); // shitty C cast
		args.push_back(0);

#ifdef __FreeBSD__
		if (p_path.find("/")) {
			// exec name contains path so use it
			execv(p_path.utf8().get_data(), &args[0]);
		} else {
			// use program name and search through PATH to find it
			execvp(getprogname(), &args[0]);
		}
#else
		execvp(p_path.utf8().get_data(), &args[0]);
#endif
		// still alive? something failed..
		fprintf(stderr, "**ERROR** OS_Unix::execute - Could not create child process while executing: %s\n", p_path.utf8().get_data());
		abort();
	}

	if (p_blocking) {

		int status;
		waitpid(pid, &status, 0);
		if (r_exitcode)
			*r_exitcode = WEXITSTATUS(status);
	} else {

		if (r_child_id)
			*r_child_id = pid;
	}

	return OK;
}

Error OS_Unix::kill(const ProcessID &p_pid) {

	int ret = ::kill(p_pid, SIGKILL);
	if (!ret) {
		//avoid zombie process
		int st;
		::waitpid(p_pid, &st, 0);
	}
	return ret ? ERR_INVALID_PARAMETER : OK;
}

int OS_Unix::get_process_id() const {

	return getpid();
};

bool OS_Unix::has_environment(const String &p_var) const {

	return getenv(p_var.utf8().get_data()) != NULL;
}

String OS_Unix::get_locale() const {

	if (!has_environment("LANG"))
		return "en";

	String locale = get_environment("LANG");
	int tp = locale.find(".");
	if (tp != -1)
		locale = locale.substr(0, tp);
	return locale;
}

Error OS_Unix::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	p_library_handle = dlopen(p_path.utf8().get_data(), RTLD_NOW);
	if (!p_library_handle) {
		ERR_EXPLAIN("Can't open dynamic library: " + p_path + ". Error: " + dlerror());
		ERR_FAIL_V(ERR_CANT_OPEN);
	}
	return OK;
}

Error OS_Unix::close_dynamic_library(void *p_library_handle) {
	if (dlclose(p_library_handle)) {
		return FAILED;
	}
	return OK;
}

Error OS_Unix::get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional) {
	const char *error;
	dlerror(); // Clear existing errors

	p_symbol_handle = dlsym(p_library_handle, p_name.utf8().get_data());

	error = dlerror();
	if (error != NULL) {
		if (!p_optional) {
			ERR_EXPLAIN("Can't resolve symbol " + p_name + ". Error: " + error);
			ERR_FAIL_V(ERR_CANT_RESOLVE);
		} else {
			return ERR_CANT_RESOLVE;
		}
	}
	return OK;
}

Error OS_Unix::set_cwd(const String &p_cwd) {

	if (chdir(p_cwd.utf8().get_data()) != 0)
		return ERR_CANT_OPEN;

	return OK;
}

String OS_Unix::get_environment(const String &p_var) const {

	if (getenv(p_var.utf8().get_data()))
		return getenv(p_var.utf8().get_data());
	return "";
}

int OS_Unix::get_processor_count() const {

	return sysconf(_SC_NPROCESSORS_CONF);
}

String OS_Unix::get_user_data_dir() const {

	String appname = get_safe_dir_name(ProjectSettings::get_singleton()->get("application/config/name"));
	if (appname != "") {
		bool use_custom_dir = ProjectSettings::get_singleton()->get("application/config/use_custom_user_dir");
		if (use_custom_dir) {
			String custom_dir = get_safe_dir_name(ProjectSettings::get_singleton()->get("application/config/custom_user_dir_name"), true);
			if (custom_dir == "") {
				custom_dir = appname;
			}
			return get_data_path().plus_file(custom_dir);
		} else {
			return get_data_path().plus_file(get_godot_dir_name()).plus_file("app_userdata").plus_file(appname);
		}
	}

	return ProjectSettings::get_singleton()->get_resource_path();
}

String OS_Unix::get_executable_path() const {

#ifdef __linux__
	//fix for running from a symlink
	char buf[256];
	memset(buf, 0, 256);
	readlink("/proc/self/exe", buf, sizeof(buf));
	String b;
	b.parse_utf8(buf);
	if (b == "") {
		WARN_PRINT("Couldn't get executable path from /proc/self/exe, using argv[0]");
		return OS::get_executable_path();
	}
	return b;
#elif defined(__FreeBSD__) || defined(__OpenBSD__)
	char resolved_path[MAXPATHLEN];

	realpath(OS::get_executable_path().utf8().get_data(), resolved_path);

	return String(resolved_path);
#elif defined(__APPLE__)
	char temp_path[1];
	uint32_t buff_size = 1;
	_NSGetExecutablePath(temp_path, &buff_size);

	char *resolved_path = new char[buff_size + 1];

	if (_NSGetExecutablePath(resolved_path, &buff_size) == 1)
		WARN_PRINT("MAXPATHLEN is too small");

	String path(resolved_path);
	delete[] resolved_path;

	return path;
#else
	ERR_PRINT("Warning, don't know how to obtain executable path on this OS! Please override this function properly.");
	return OS::get_executable_path();
#endif
}

void UnixTerminalLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, ErrorType p_type) {
	if (!should_log(true)) {
		return;
	}

	const char *err_details;
	if (p_rationale && p_rationale[0])
		err_details = p_rationale;
	else
		err_details = p_code;

	switch (p_type) {
		case ERR_WARNING:
			logf_error("\E[1;33mWARNING: %s: \E[0m\E[1m%s\n", p_function, err_details);
			logf_error("\E[0;33m   At: %s:%i.\E[0m\n", p_file, p_line);
			break;
		case ERR_SCRIPT:
			logf_error("\E[1;35mSCRIPT ERROR: %s: \E[0m\E[1m%s\n", p_function, err_details);
			logf_error("\E[0;35m   At: %s:%i.\E[0m\n", p_file, p_line);
			break;
		case ERR_SHADER:
			logf_error("\E[1;36mSHADER ERROR: %s: \E[0m\E[1m%s\n", p_function, err_details);
			logf_error("\E[0;36m   At: %s:%i.\E[0m\n", p_file, p_line);
			break;
		case ERR_ERROR:
		default:
			logf_error("\E[1;31mERROR: %s: \E[0m\E[1m%s\n", p_function, err_details);
			logf_error("\E[0;31m   At: %s:%i.\E[0m\n", p_file, p_line);
			break;
	}
}

UnixTerminalLogger::~UnixTerminalLogger() {}

OS_Unix::OS_Unix() {
	Vector<Logger *> loggers;
	loggers.push_back(memnew(UnixTerminalLogger));
	_set_logger(memnew(CompositeLogger(loggers)));
}

#endif
