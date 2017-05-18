/*************************************************************************/
/*  process.h                                                            */
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
#ifndef PROCESS_H
#define PROCESS_H

#include "dvector.h"
#include "reference.h"
#include "ustring.h"

class Process : public Reference {
	GDCLASS(Process, Reference)

public:
	enum ExitStatus {
		EXIT_STATUS_NORMAL,
		EXIT_STATUS_CRASH
	};

	enum State {
		STATE_NOT_RUNNING,
		STATE_STARTING,
		STATE_RUNNING
	};

	enum ProcessChannel {
		CHANNEL_STDOUT,
		CHANNEL_STDERR,
		CHANNEL_STDIN // last, for read_channel with PROPERTY_HINT_ENUM to work
	};

	enum Redirect {
		REDIRECT_STDIN = 1,
		REDIRECT_STDOUT = 2,
		REDIRECT_STDERR = 4,
		REDIRECT_ALL = REDIRECT_STDIN | REDIRECT_STDOUT | REDIRECT_STDERR,
		REDIRECT_STDERR_TO_STDOUT = 8
	};

private:
	String program;
	Vector<String> arguments;
	String cwd;

	HashMap<String, String> environment;
	ProcessChannel read_channel;
	uint32_t redirect_flags;

	void _set_environment_bind(const Dictionary &p_env, bool p_override = false);
	Dictionary _get_environment_bind() const;

	Vector<String> _get_arguments_bind() const { return arguments; }

	PoolVector<uint8_t> _read_all_bind();
	PoolVector<uint8_t> _read_line_bind();
	int _write_bind(const PoolVector<uint8_t> &p_text);

	bool _start_bind(const String &p_program, const Vector<String> &p_arguments);

protected:
	struct Channel {
		bool closed;

		virtual bool open() = 0;
		virtual void close() = 0;

		Channel();
		virtual ~Channel() {}
	};

	Channel *stdin_sama;
	Channel *stdout_chan;
	Channel *stderr_chan;

	State process_state;
	ExitStatus exit_status;
	int exit_code;

	static Process *(*_create)();

	static void _bind_methods();

	virtual bool _start() = 0;
	virtual void _get_system_env(HashMap<String, String> &r_env) = 0;

	Process(Channel *p_stdin, Channel *p_stdout, Channel *p_stderr);

public:
	_FORCE_INLINE_ void set_arguments(const Vector<String> &p_arguments) { arguments = p_arguments; }
	_FORCE_INLINE_ const Vector<String> &get_arguments() const { return arguments; }

	_FORCE_INLINE_ void set_program(const String &p_program) { program = p_program; }
	_FORCE_INLINE_ String get_program() const { return program; }

	_FORCE_INLINE_ void set_cwd(const String &p_dir) { cwd = p_dir; }
	_FORCE_INLINE_ String get_cwd() const { return cwd; }

	_FORCE_INLINE_ State get_state() const { return process_state; }
	_FORCE_INLINE_ ExitStatus get_exit_status() const { return exit_status; }
	_FORCE_INLINE_ int get_exit_code() const { return exit_code; }

	_FORCE_INLINE_ void set_redirect_flags(uint32_t p_redirect_flags) { redirect_flags = p_redirect_flags; }
	_FORCE_INLINE_ uint32_t get_redirect_flags() const { return redirect_flags; }

	void set_environment(const HashMap<String, String> &p_env, bool p_override = false);
	_FORCE_INLINE_ const HashMap<String, String> &get_environment() const { return environment; }

	void set_read_channel(ProcessChannel p_channel);
	_FORCE_INLINE_ ProcessChannel get_read_channel() const { return read_channel; }
	void close_channel(ProcessChannel p_channel);

	bool start(const String &p_program, const Vector<String> &p_arguments);
	bool start();

	bool can_read_line() const;

	virtual int64_t get_pid() const = 0;

	virtual Error poll() = 0;

	virtual bool wait_for_started(int msecs = 20000) = 0;
	virtual bool wait_for_finished(int msecs = 20000) = 0;

	virtual void terminate() = 0;
	virtual void kill() = 0;

	virtual int get_available_bytes() const = 0;
	virtual int next_line_size() const = 0;
	virtual int read_all(char *p_data, int p_max_size) = 0;
	virtual int read_line(char *p_data, int p_max_size) = 0;

	virtual bool can_write(int p_size) const = 0;
	virtual int write(const char *p_text, int p_max_size) = 0;

	static Ref<Process> create_ref();
	static Process *create();

	~Process();
};

VARIANT_ENUM_CAST(Process::ExitStatus)
VARIANT_ENUM_CAST(Process::State)
VARIANT_ENUM_CAST(Process::ProcessChannel)
VARIANT_ENUM_CAST(Process::Redirect)

#endif // PROCESS_H
