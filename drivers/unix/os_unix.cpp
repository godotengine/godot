/*************************************************************************/
/*  os_unix.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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

#include "memory_pool_static_malloc.h"
#include "os/memory_pool_dynamic_static.h"
#include "thread_posix.h"
#include "semaphore_posix.h"
#include "mutex_posix.h"
#include "core/os/thread_dummy.h"

//#include "core/io/file_access_buffered_fa.h"
#include "file_access_unix.h"
#include "dir_access_unix.h"
#include "tcp_server_posix.h"
#include "stream_peer_tcp_posix.h"
#include "packet_peer_udp_posix.h"

#ifdef __FreeBSD__
#include <sys/param.h>
#endif
#include <stdarg.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <poll.h>
#include <errno.h>
#include <assert.h>
#include "globals.h"

extern bool _print_error_enabled;

void OS_Unix::print_error(const char* p_function,const char* p_file,int p_line,const char *p_code,const char*p_rationale,ErrorType p_type) {

	if (!_print_error_enabled)
		return;

	if (p_rationale && p_rationale[0]) {

		print("\E[1;31;40mERROR: %s: \E[1;37;40m%s\n",p_function,p_rationale);
		print("\E[0;31;40m   At: %s:%i.\E[0;0;37m\n",p_file,p_line);

	} else {
		print("\E[1;31;40mERROR: %s: \E[1;37;40m%s\n",p_function,p_code);
		print("\E[0;31;40m   At: %s:%i.\E[0;0;37m\n",p_file,p_line);

	}
}


void OS_Unix::debug_break() {

	assert(false);
};

int OS_Unix::get_audio_driver_count() const {

	return 1;

}
const char * OS_Unix::get_audio_driver_name(int p_driver) const {

	return "dummy";
}
	
int OS_Unix::unix_initialize_audio(int p_audio_driver) {

	return 0;
}
	
static MemoryPoolStaticMalloc *mempool_static=NULL;
static MemoryPoolDynamicStatic *mempool_dynamic=NULL;
	
	
void OS_Unix::initialize_core() {

#ifdef NO_PTHREADS
	ThreadDummy::make_default();
	SemaphoreDummy::make_default();
	MutexDummy::make_default();
#else
	ThreadPosix::make_default();	
	SemaphorePosix::make_default();
	MutexPosix::make_default();	
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
	mempool_static = new MemoryPoolStaticMalloc;
	mempool_dynamic = memnew( MemoryPoolDynamicStatic );

	ticks_start=0;
	ticks_start=get_ticks_usec();
}

void OS_Unix::finalize_core() {


	if (mempool_dynamic)
		memdelete( mempool_dynamic );
	if (mempool_static)
		delete mempool_static;

}


void OS_Unix::vprint(const char* p_format, va_list p_list,bool p_stder) {

	if (p_stder) {

		vfprintf(stderr,p_format,p_list);
		fflush(stderr);
	} else {

		vprintf(p_format,p_list);
		fflush(stdout);
	}
}

void OS_Unix::print(const char *p_format, ... ) {

	va_list argp;
	va_start(argp, p_format);
	vprintf(p_format, argp );
	va_end(argp);

}
void OS_Unix::alert(const String& p_alert,const String& p_title) {

	fprintf(stderr,"ERROR: %s\n",p_alert.utf8().get_data());
}

static int has_data(FILE* p_fd, int timeout_usec = 0) {

	fd_set readset;
	int fd = fileno(p_fd);
	FD_ZERO(&readset);
	FD_SET(fd, &readset);
	timeval time;
	time.tv_sec = 0;
	time.tv_usec = timeout_usec;
	int res = 0;//select(fd + 1, &readset, NULL, NULL, &time);
	return res > 0;
};


String OS_Unix::get_stdin_string(bool p_block) {

	String ret;
	if (p_block) {
		char buff[1024];
		ret = stdin_buf + fgets(buff,1024,stdin);
		stdin_buf = "";
		return ret;
	};

	while (has_data(stdin)) {

		char ch;
		read(fileno(stdin), &ch, 1);
		if (ch == '\n') {
			ret = stdin_buf;
			stdin_buf = "";
			return ret;
		} else {
			char str[2] = { ch, 0 };
			stdin_buf += str;
		};
	};

	return "";
}

String OS_Unix::get_name() {

	return "Unix";
}


uint64_t OS_Unix::get_unix_time() const {

	return time(NULL);
};

uint64_t OS_Unix::get_system_time_msec() const {
	struct timeval tv_now;
	gettimeofday(&tv_now, NULL);
	//localtime(&tv_now.tv_usec);
	//localtime((const long *)&tv_now.tv_usec);
	uint64_t msec = uint64_t(tv_now.tv_sec)*1000+tv_now.tv_usec/1000;
	return msec;
}


OS::Date OS_Unix::get_date(bool utc) const {

	time_t t=time(NULL);
	struct tm *lt;
	if (utc)
		lt=gmtime(&t);
	else
		lt=localtime(&t);
	Date ret;
	ret.year=1900+lt->tm_year;
	ret.month=(Month)(lt->tm_mon + 1);
	ret.day=lt->tm_mday;
	ret.weekday=(Weekday)lt->tm_wday;
	ret.dst=lt->tm_isdst;
	
	return ret;
}
OS::Time OS_Unix::get_time(bool utc) const {
	time_t t=time(NULL);
	struct tm *lt;
	if (utc)
		lt=gmtime(&t);
	else
		lt=localtime(&t);
	Time ret;
	ret.hour=lt->tm_hour;
	ret.min=lt->tm_min;
	ret.sec=lt->tm_sec;
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

	usleep(p_usec);
}
uint64_t OS_Unix::get_ticks_usec() const {

	struct timeval tv_now;
	gettimeofday(&tv_now,NULL);
	
	uint64_t longtime = (uint64_t)tv_now.tv_usec + (uint64_t)tv_now.tv_sec*1000000L;
	longtime-=ticks_start;
	
	return longtime;
}

Error OS_Unix::execute(const String& p_path, const List<String>& p_arguments,bool p_blocking,ProcessID *r_child_id,String* r_pipe,int *r_exitcode) {


	if (p_blocking && r_pipe) {


		String argss;
		argss="\""+p_path+"\"";

		for(int i=0;i<p_arguments.size();i++) {

			argss+=String(" \"")+p_arguments[i]+"\"";
		}

		argss+=" 2>/dev/null"; //silence stderr
		FILE* f=popen(argss.utf8().get_data(),"r");

		ERR_FAIL_COND_V(!f,ERR_CANT_OPEN);

		char buf[65535];
		while(fgets(buf,65535,f)) {

			(*r_pipe)+=buf;
		}

		int rv = pclose(f);
		if (r_exitcode)
			*r_exitcode=rv;

		return OK;
	}


	pid_t pid = fork();
	ERR_FAIL_COND_V(pid<0,ERR_CANT_FORK);
	//print("execute: %s\n",p_path.utf8().get_data());


	if (pid==0) {
		// is child
		Vector<CharString> cs;
		cs.push_back(p_path.utf8());
		for(int i=0;i<p_arguments.size();i++)
			cs.push_back(p_arguments[i].utf8());

		Vector<char*> args;
		for(int i=0;i<cs.size();i++)
			args.push_back((char*)cs[i].get_data());// shitty C cast
		args.push_back(0);

#ifdef __FreeBSD__
		if(p_path.find("/")) {
			// exec name contains path so use it
			execv(p_path.utf8().get_data(),&args[0]);
		}else{
			// use program name and search through PATH to find it
			execvp(getprogname(),&args[0]);
		}
#else
		execv(p_path.utf8().get_data(),&args[0]);
#endif
		// still alive? something failed..
		fprintf(stderr,"**ERROR** OS_Unix::execute - Could not create child process while executing: %s\n",p_path.utf8().get_data());
		abort();
	}

	if (p_blocking) {

		int status;
		pid_t rpid = waitpid(pid,&status,0);
		if (r_exitcode)
			*r_exitcode=WEXITSTATUS(status);

		print("returned: %i, waiting for: %i\n",rpid,pid);
	} else {

		if (r_child_id)
			*r_child_id=pid;
	}

	return OK;

}

Error OS_Unix::kill(const ProcessID& p_pid) {

	int ret = ::kill(p_pid,SIGKILL);
	if (!ret) {
		//avoid zombie process
		int st;
		::waitpid(p_pid,&st,0);

	}
	return ret?ERR_INVALID_PARAMETER:OK;
}

int OS_Unix::get_process_ID() const {

	return getpid();
};


bool OS_Unix::has_environment(const String& p_var) const {

	return getenv(p_var.utf8().get_data())!=NULL;
}

String OS_Unix::get_locale() const {

	if (!has_environment("LANG"))
		return "en";

	String locale = get_environment("LANG");
	int tp = locale.find(".");
	if (tp!=-1)
		locale=locale.substr(0,tp);
	return locale;
}

Error OS_Unix::set_cwd(const String& p_cwd) {

	if (chdir(p_cwd.utf8().get_data())!=0)
		return ERR_CANT_OPEN;

	return OK;
}


String OS_Unix::get_environment(const String& p_var) const {

	if (getenv(p_var.utf8().get_data()))
		return getenv(p_var.utf8().get_data());
	return "";
}

int OS_Unix::get_processor_count() const {

	return sysconf(_SC_NPROCESSORS_CONF);
}

String OS_Unix::get_data_dir() const {

	String an = Globals::get_singleton()->get("application/name");
	if (an!="") {



		if (has_environment("HOME")) {

			bool use_godot = Globals::get_singleton()->get("application/use_shared_user_dir");
			if (use_godot)
				return get_environment("HOME")+"/.godot/app_userdata/"+an;
			else
				return get_environment("HOME")+"/."+an;
		}
	}

	return Globals::get_singleton()->get_resource_path();

}

String OS_Unix::get_executable_path() const {

#ifdef __linux__
	//fix for running from a symlink
	char buf[256];
	memset(buf,0,256);
	readlink("/proc/self/exe", buf, sizeof(buf));
	//print_line("Exec path is:"+String(buf));
	String b;
	b.parse_utf8(buf);
	if (b=="") {
		WARN_PRINT("Couldn't get executable path from /proc/self/exe, using argv[0]");
		return OS::get_executable_path();
	}
	return b;
#elif defined(__FreeBSD__)
	char resolved_path[MAXPATHLEN];

	realpath(OS::get_executable_path().utf8().get_data(), resolved_path);

	return String(resolved_path);
#else
	ERR_PRINT("Warning, don't know how to obtain executable path on this OS! Please override this function properly.");
	return OS::get_executable_path();
#endif
}


#endif
