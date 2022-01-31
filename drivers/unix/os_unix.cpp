/*************************************************************************/
/*  os_unix.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"
#include "drivers/unix/dir_access_unix.h"
#include "drivers/unix/file_access_unix.h"
#include "drivers/unix/net_socket_posix.h"
#include "drivers/unix/thread_posix.h"
#include "servers/rendering_server.h"

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <mach/mach_time.h>
#endif

#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <sys/param.h>
#include <sys/sysctl.h>
#endif

#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/// Clock Setup function (used by get_ticks_usec)
static uint64_t _clock_start = 0;
#if defined(__APPLE__)
static double _clock_scale = 0;
static void _setup_clock() {
	mach_timebase_info_data_t info;
	kern_return_t ret = mach_timebase_info(&info);
	ERR_FAIL_COND_MSG(ret != 0, "OS CLOCK IS NOT WORKING!");
	_clock_scale = ((double)info.numer / (double)info.denom) / 1000.0;
	_clock_start = mach_absolute_time() * _clock_scale;
}
#else
#if defined(CLOCK_MONOTONIC_RAW) && !defined(JAVASCRIPT_ENABLED) // This is a better clock on Linux.
#define GODOT_CLOCK CLOCK_MONOTONIC_RAW
#else
#define GODOT_CLOCK CLOCK_MONOTONIC
#endif
static void _setup_clock() {
	struct timespec tv_now = { 0, 0 };
	ERR_FAIL_COND_MSG(clock_gettime(GODOT_CLOCK, &tv_now) != 0, "OS CLOCK IS NOT WORKING!");
	_clock_start = ((uint64_t)tv_now.tv_nsec / 1000L) + (uint64_t)tv_now.tv_sec * 1000000L;
}
#endif

void OS_Unix::debug_break() {
	assert(false);
};

static void handle_interrupt(int sig) {
	if (!EngineDebugger::is_active()) {
		return;
	}

	EngineDebugger::get_script_debugger()->set_depth(-1);
	EngineDebugger::get_script_debugger()->set_lines_left(1);
}

void OS_Unix::initialize_debugging() {
	if (EngineDebugger::is_active()) {
		struct sigaction action;
		memset(&action, 0, sizeof(action));
		action.sa_handler = handle_interrupt;
		sigaction(SIGINT, &action, nullptr);
	}
}

int OS_Unix::unix_initialize_audio(int p_audio_driver) {
	return 0;
}

void OS_Unix::initialize_core() {
#if !defined(NO_THREADS)
	init_thread_posix();
#endif

	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_RESOURCES);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessUnix>(FileAccess::ACCESS_FILESYSTEM);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessUnix>(DirAccess::ACCESS_FILESYSTEM);

#ifndef NO_NETWORK
	NetSocketPosix::make_default();
	IPUnix::make_default();
#endif

	_setup_clock();
}

void OS_Unix::finalize_core() {
	NetSocketPosix::cleanup();
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

String OS_Unix::get_name() const {
	return "Unix";
}

double OS_Unix::get_unix_time() const {
	struct timeval tv_now;
	gettimeofday(&tv_now, nullptr);
	return (double)tv_now.tv_sec + double(tv_now.tv_usec) / 1000000;
};

OS::Date OS_Unix::get_date(bool p_utc) const {
	time_t t = time(nullptr);
	struct tm lt;
	if (p_utc) {
		gmtime_r(&t, &lt);
	} else {
		localtime_r(&t, &lt);
	}
	Date ret;
	ret.year = 1900 + lt.tm_year;
	// Index starting at 1 to match OS_Unix::get_date
	//   and Windows SYSTEMTIME and tm_mon follows the typical structure
	//   of 0-11, noted here: http://www.cplusplus.com/reference/ctime/tm/
	ret.month = (Month)(lt.tm_mon + 1);
	ret.day = lt.tm_mday;
	ret.weekday = (Weekday)lt.tm_wday;
	ret.dst = lt.tm_isdst;

	return ret;
}

OS::Time OS_Unix::get_time(bool p_utc) const {
	time_t t = time(nullptr);
	struct tm lt;
	if (p_utc) {
		gmtime_r(&t, &lt);
	} else {
		localtime_r(&t, &lt);
	}
	Time ret;
	ret.hour = lt.tm_hour;
	ret.minute = lt.tm_min;
	ret.second = lt.tm_sec;
	get_time_zone_info();
	return ret;
}

OS::TimeZoneInfo OS_Unix::get_time_zone_info() const {
	time_t t = time(nullptr);
	struct tm lt;
	localtime_r(&t, &lt);
	char name[16];
	strftime(name, 16, "%Z", &lt);
	name[15] = 0;
	TimeZoneInfo ret;
	ret.name = name;

	char bias_buf[16];
	strftime(bias_buf, 16, "%z", &lt);
	int bias;
	bias_buf[15] = 0;
	sscanf(bias_buf, "%d", &bias);

	// convert from ISO 8601 (1 minute=1, 1 hour=100) to minutes
	int hour = (int)bias / 100;
	int minutes = bias % 100;
	if (bias < 0) {
		ret.bias = hour * 60 - minutes;
	} else {
		ret.bias = hour * 60 + minutes;
	}

	return ret;
}

void OS_Unix::delay_usec(uint32_t p_usec) const {
	struct timespec requested = { static_cast<time_t>(p_usec / 1000000), (static_cast<long>(p_usec) % 1000000) * 1000 };
	struct timespec remaining;
	while (nanosleep(&requested, &remaining) == -1 && errno == EINTR) {
		requested.tv_sec = remaining.tv_sec;
		requested.tv_nsec = remaining.tv_nsec;
	}
}

uint64_t OS_Unix::get_ticks_usec() const {
#if defined(__APPLE__)
	uint64_t longtime = mach_absolute_time() * _clock_scale;
#else
	// Unchecked return. Static analyzers might complain.
	// If _setup_clock() succeeded, we assume clock_gettime() works.
	struct timespec tv_now = { 0, 0 };
	clock_gettime(GODOT_CLOCK, &tv_now);
	uint64_t longtime = ((uint64_t)tv_now.tv_nsec / 1000L) + (uint64_t)tv_now.tv_sec * 1000000L;
#endif
	longtime -= _clock_start;

	return longtime;
}

Error OS_Unix::execute(const String &p_path, const List<String> &p_arguments, String *r_pipe, int *r_exitcode, bool read_stderr, Mutex *p_pipe_mutex, bool p_open_console) {
#ifdef __EMSCRIPTEN__
	// Don't compile this code at all to avoid undefined references.
	// Actual virtual call goes to OS_JavaScript.
	ERR_FAIL_V(ERR_BUG);
#else
	if (r_pipe) {
		String command = "\"" + p_path + "\"";
		for (int i = 0; i < p_arguments.size(); i++) {
			command += String(" \"") + p_arguments[i] + "\"";
		}
		if (read_stderr) {
			command += " 2>&1"; // Include stderr
		} else {
			command += " 2>/dev/null"; // Silence stderr
		}

		FILE *f = popen(command.utf8().get_data(), "r");
		ERR_FAIL_COND_V_MSG(!f, ERR_CANT_OPEN, "Cannot create pipe from command: " + command);
		char buf[65535];
		while (fgets(buf, 65535, f)) {
			if (p_pipe_mutex) {
				p_pipe_mutex->lock();
			}
			(*r_pipe) += String::utf8(buf);
			if (p_pipe_mutex) {
				p_pipe_mutex->unlock();
			}
		}
		int rv = pclose(f);

		if (r_exitcode) {
			*r_exitcode = WEXITSTATUS(rv);
		}
		return OK;
	}

	pid_t pid = fork();
	ERR_FAIL_COND_V(pid < 0, ERR_CANT_FORK);

	if (pid == 0) {
		// The child process
		Vector<CharString> cs;
		cs.push_back(p_path.utf8());
		for (int i = 0; i < p_arguments.size(); i++) {
			cs.push_back(p_arguments[i].utf8());
		}

		Vector<char *> args;
		for (int i = 0; i < cs.size(); i++) {
			args.push_back((char *)cs[i].get_data());
		}
		args.push_back(0);

		execvp(p_path.utf8().get_data(), &args[0]);
		// The execvp() function only returns if an error occurs.
		ERR_PRINT("Could not create child process: " + p_path);
		raise(SIGKILL);
	}

	int status;
	waitpid(pid, &status, 0);
	if (r_exitcode) {
		*r_exitcode = WIFEXITED(status) ? WEXITSTATUS(status) : status;
	}
	return OK;
#endif
}

Error OS_Unix::create_process(const String &p_path, const List<String> &p_arguments, ProcessID *r_child_id, bool p_open_console) {
#ifdef __EMSCRIPTEN__
	// Don't compile this code at all to avoid undefined references.
	// Actual virtual call goes to OS_JavaScript.
	ERR_FAIL_V(ERR_BUG);
#else
	pid_t pid = fork();
	ERR_FAIL_COND_V(pid < 0, ERR_CANT_FORK);

	if (pid == 0) {
		// The new process
		// Create a new session-ID so parent won't wait for it.
		// This ensures the process won't go zombie at the end.
		setsid();

		Vector<CharString> cs;
		cs.push_back(p_path.utf8());
		for (int i = 0; i < p_arguments.size(); i++) {
			cs.push_back(p_arguments[i].utf8());
		}

		Vector<char *> args;
		for (int i = 0; i < cs.size(); i++) {
			args.push_back((char *)cs[i].get_data());
		}
		args.push_back(0);

		execvp(p_path.utf8().get_data(), &args[0]);
		// The execvp() function only returns if an error occurs.
		ERR_PRINT("Could not create child process: " + p_path);
		raise(SIGKILL);
	}

	if (r_child_id) {
		*r_child_id = pid;
	}
	return OK;
#endif
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
	return getenv(p_var.utf8().get_data()) != nullptr;
}

String OS_Unix::get_locale() const {
	if (!has_environment("LANG")) {
		return "en";
	}

	String locale = get_environment("LANG");
	int tp = locale.find(".");
	if (tp != -1) {
		locale = locale.substr(0, tp);
	}
	return locale;
}

Error OS_Unix::open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path) {
	String path = p_path;

	if (FileAccess::exists(path) && path.is_relative_path()) {
		// dlopen expects a slash, in this case a leading ./ for it to be interpreted as a relative path,
		//  otherwise it will end up searching various system directories for the lib instead and finally failing.
		path = "./" + path;
	}

	if (!FileAccess::exists(path)) {
		//this code exists so gdnative can load .so files from within the executable path
		path = get_executable_path().get_base_dir().plus_file(p_path.get_file());
	}

	if (!FileAccess::exists(path)) {
		//this code exists so gdnative can load .so files from a standard unix location
		path = get_executable_path().get_base_dir().plus_file("../lib").plus_file(p_path.get_file());
	}

	p_library_handle = dlopen(path.utf8().get_data(), RTLD_NOW);
	ERR_FAIL_COND_V_MSG(!p_library_handle, ERR_CANT_OPEN, "Can't open dynamic library: " + p_path + ". Error: " + dlerror());
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
	if (error != nullptr) {
		ERR_FAIL_COND_V_MSG(!p_optional, ERR_CANT_RESOLVE, "Can't resolve symbol " + p_name + ". Error: " + error + ".");

		return ERR_CANT_RESOLVE;
	}
	return OK;
}

Error OS_Unix::set_cwd(const String &p_cwd) {
	if (chdir(p_cwd.utf8().get_data()) != 0) {
		return ERR_CANT_OPEN;
	}

	return OK;
}

String OS_Unix::get_environment(const String &p_var) const {
	if (getenv(p_var.utf8().get_data())) {
		return getenv(p_var.utf8().get_data());
	}
	return "";
}

bool OS_Unix::set_environment(const String &p_var, const String &p_value) const {
	return setenv(p_var.utf8().get_data(), p_value.utf8().get_data(), /* overwrite: */ true) == 0;
}

int OS_Unix::get_processor_count() const {
	return sysconf(_SC_NPROCESSORS_CONF);
}

String OS_Unix::get_user_data_dir() const {
	String appname = get_safe_dir_name(ProjectSettings::get_singleton()->get("application/config/name"));
	if (!appname.is_empty()) {
		bool use_custom_dir = ProjectSettings::get_singleton()->get("application/config/use_custom_user_dir");
		if (use_custom_dir) {
			String custom_dir = get_safe_dir_name(ProjectSettings::get_singleton()->get("application/config/custom_user_dir_name"), true);
			if (custom_dir.is_empty()) {
				custom_dir = appname;
			}
			return get_data_path().plus_file(custom_dir);
		} else {
			return get_data_path().plus_file(get_godot_dir_name()).plus_file("app_userdata").plus_file(appname);
		}
	}

	return get_data_path().plus_file(get_godot_dir_name()).plus_file("app_userdata").plus_file("[unnamed project]");
}

String OS_Unix::get_executable_path() const {
#ifdef __linux__
	//fix for running from a symlink
	char buf[256];
	memset(buf, 0, 256);
	ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf));
	String b;
	if (len > 0) {
		b.parse_utf8(buf, len);
	}
	if (b.is_empty()) {
		WARN_PRINT("Couldn't get executable path from /proc/self/exe, using argv[0]");
		return OS::get_executable_path();
	}
	return b;
#elif defined(__OpenBSD__) || defined(__NetBSD__)
	char resolved_path[MAXPATHLEN];

	realpath(OS::get_executable_path().utf8().get_data(), resolved_path);

	return String(resolved_path);
#elif defined(__FreeBSD__)
	int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
	char buf[MAXPATHLEN];
	size_t len = sizeof(buf);
	if (sysctl(mib, 4, buf, &len, nullptr, 0) != 0) {
		WARN_PRINT("Couldn't get executable path from sysctl");
		return OS::get_executable_path();
	}
	String b;
	b.parse_utf8(buf);
	return b;
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

void UnixTerminalLogger::log_error(const char *p_function, const char *p_file, int p_line, const char *p_code, const char *p_rationale, bool p_editor_notify, ErrorType p_type) {
	if (!should_log(true)) {
		return;
	}

	const char *err_details;
	if (p_rationale && p_rationale[0]) {
		err_details = p_rationale;
	} else {
		err_details = p_code;
	}

	// Disable color codes if stdout is not a TTY.
	// This prevents Godot from writing ANSI escape codes when redirecting
	// stdout and stderr to a file.
	const bool tty = isatty(fileno(stdout));
	const char *gray = tty ? "\E[0;90m" : "";
	const char *red = tty ? "\E[0;91m" : "";
	const char *red_bold = tty ? "\E[1;31m" : "";
	const char *yellow = tty ? "\E[0;93m" : "";
	const char *yellow_bold = tty ? "\E[1;33m" : "";
	const char *magenta = tty ? "\E[0;95m" : "";
	const char *magenta_bold = tty ? "\E[1;35m" : "";
	const char *cyan = tty ? "\E[0;96m" : "";
	const char *cyan_bold = tty ? "\E[1;36m" : "";
	const char *reset = tty ? "\E[0m" : "";

	switch (p_type) {
		case ERR_WARNING:
			logf_error("%sWARNING:%s %s\n", yellow_bold, yellow, err_details);
			logf_error("%s     at: %s (%s:%i)%s\n", gray, p_function, p_file, p_line, reset);
			break;
		case ERR_SCRIPT:
			logf_error("%sSCRIPT ERROR:%s %s\n", magenta_bold, magenta, err_details);
			logf_error("%s          at: %s (%s:%i)%s\n", gray, p_function, p_file, p_line, reset);
			break;
		case ERR_SHADER:
			logf_error("%sSHADER ERROR:%s %s\n", cyan_bold, cyan, err_details);
			logf_error("%s          at: %s (%s:%i)%s\n", gray, p_function, p_file, p_line, reset);
			break;
		case ERR_ERROR:
		default:
			logf_error("%sERROR:%s %s\n", red_bold, red, err_details);
			logf_error("%s   at: %s (%s:%i)%s\n", gray, p_function, p_file, p_line, reset);
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
