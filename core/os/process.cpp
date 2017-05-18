/*************************************************************************/
/*  process.cpp                                                          */
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
#include "process.h"

Process *(*Process::_create)() = NULL;

bool Process::start(const String &p_program, const Vector<String> &p_arguments) {
	ERR_FAIL_COND_V(process_state != STATE_NOT_RUNNING, false);

	set_program(p_program);
	set_arguments(p_arguments);

	return start();
}

bool Process::start() {
	ERR_FAIL_COND_V(process_state != STATE_NOT_RUNNING, false);

	exit_code = 0;
	exit_status = EXIT_STATUS_NORMAL;

	return _start();
}

bool Process::can_read_line() const {
	return next_line_size() >= 0;
}

void Process::_set_environment_bind(const Dictionary &p_env, bool p_override) {
	environment.clear();

	List<Variant> keys;
	p_env.get_key_list(&keys);

	for (List<Variant>::Element *E = keys.front(); E; E = E->next()) {
		environment.set(E->get(), p_env[E->get()]);
	}

	if (!p_override) {
		_get_system_env(environment);
	}
}

Dictionary Process::_get_environment_bind() const {
	Dictionary dict;

	const String *k = NULL;
	while ((k = environment.next(k))) {
		dict[k] = environment.get(*k);
	}

	return dict;
}

PoolVector<uint8_t> Process::_read_all_bind() {
	PoolVector<uint8_t> ret;
	ret.resize(get_available_bytes());
	PoolVector<uint8_t>::Write w = ret.write();
	read_all((char *)w.ptr(), ret.size());
	return ret;
}

PoolVector<uint8_t> Process::_read_line_bind() {
	PoolVector<uint8_t> ret;
	int size = next_line_size();
	if (size == -1)
		return ret;
	ret.resize(size);
	PoolVector<uint8_t>::Write w = ret.write();
	read_line((char *)w.ptr(), ret.size());
	return ret;
}

int Process::_write_bind(const PoolVector<uint8_t> &p_bytes) {
	PoolVector<uint8_t>::Read r = p_bytes.read();
	return write((const char *)r.ptr(), p_bytes.size());
}

bool Process::_start_bind(const String &p_program, const Vector<String> &p_arguments) {
	set_program(p_program);
	set_arguments(p_arguments);
	return start();
}

void Process::close_channel(Process::ProcessChannel p_channel) {
	switch (p_channel) {
		case CHANNEL_STDOUT: {
			stdout_chan->close();
		} break;
		case CHANNEL_STDERR: {
			stderr_chan->close();
		} break;
		case CHANNEL_STDIN: {
			stdin_sama->close();
		} break;
		default: ERR_FAIL();
	}
}

Ref<Process> Process::create_ref() {
	if (!_create)
		return NULL;
	return Ref<Process>(_create());
}

Process *Process::create() {
	if (!_create)
		return NULL;
	return _create();
}

void Process::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_program"), &Process::get_program);
	ClassDB::bind_method(D_METHOD("get_arguments"), &Process::_get_arguments_bind);

	ClassDB::bind_method(D_METHOD("set_cwd", "path"), &Process::set_cwd);
	ClassDB::bind_method(D_METHOD("get_cwd"), &Process::get_cwd);

	ClassDB::bind_method(D_METHOD("get_state"), &Process::get_state);
	ClassDB::bind_method(D_METHOD("get_exit_status"), &Process::get_exit_status);
	ClassDB::bind_method(D_METHOD("get_exit_code"), &Process::get_exit_code);

	ClassDB::bind_method(D_METHOD("set_redirect_flags", "flags"), &Process::set_redirect_flags);
	ClassDB::bind_method(D_METHOD("get_redirect_flags"), &Process::get_redirect_flags);

	ClassDB::bind_method(D_METHOD("set_environment", "env", "override"), &Process::_set_environment_bind, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_environment"), &Process::_get_environment_bind);

	ClassDB::bind_method(D_METHOD("set_read_channel", "channel"), &Process::set_read_channel);
	ClassDB::bind_method(D_METHOD("get_read_channel"), &Process::get_read_channel);
	ClassDB::bind_method(D_METHOD("close_channel", "channel"), &Process::close_channel);

	ClassDB::bind_method(D_METHOD("get_pid"), &Process::get_pid);

	ClassDB::bind_method(D_METHOD("start"), &Process::_start_bind);
	ClassDB::bind_method(D_METHOD("poll"), &Process::poll);

	ClassDB::bind_method(D_METHOD("wait_for_started", "msecs"), &Process::wait_for_started, DEFVAL(20000));
	ClassDB::bind_method(D_METHOD("wait_for_finished", "msecs"), &Process::wait_for_finished, DEFVAL(20000));

	ClassDB::bind_method(D_METHOD("terminate"), &Process::terminate);
	ClassDB::bind_method(D_METHOD("kill"), &Process::kill);

	ClassDB::bind_method(D_METHOD("get_available_bytes"), &Process::get_available_bytes);
	ClassDB::bind_method(D_METHOD("can_read_line"), &Process::can_read_line);
	ClassDB::bind_method(D_METHOD("read_all"), &Process::_read_all_bind);
	ClassDB::bind_method(D_METHOD("read_line"), &Process::_read_line_bind);

	ClassDB::bind_method(D_METHOD("can_write", "bytes"), &Process::can_write);
	ClassDB::bind_method(D_METHOD("write", "data"), &Process::_write_bind);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "cwd", PROPERTY_HINT_DIR), "set_cwd", "get_cwd");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "redirect_flags", PROPERTY_HINT_FLAGS, "RedirectStdin,RedirectStdout,RedirectStderr,RedirectStderrToStdout"), "set_redirect_flags", "get_redirect_flags");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "read_channel", PROPERTY_HINT_ENUM, "ChannelStdout,ChannelStderr"), "set_read_channel", "get_read_channel");

	BIND_CONSTANT(EXIT_STATUS_NORMAL);
	BIND_CONSTANT(EXIT_STATUS_CRASH);

	BIND_CONSTANT(STATE_NOT_RUNNING);
	BIND_CONSTANT(STATE_STARTING);
	BIND_CONSTANT(STATE_RUNNING);

	BIND_CONSTANT(CHANNEL_STDOUT);
	BIND_CONSTANT(CHANNEL_STDERR);
	BIND_CONSTANT(CHANNEL_STDIN);

	BIND_CONSTANT(REDIRECT_STDIN);
	BIND_CONSTANT(REDIRECT_STDOUT);
	BIND_CONSTANT(REDIRECT_STDERR);
	BIND_CONSTANT(REDIRECT_ALL);
	BIND_CONSTANT(REDIRECT_STDERR_TO_STDOUT);
}

Process::Process(Channel *p_stdin, Channel *p_stdout, Channel *p_stderr) {
	exit_code = 0;
	exit_status = EXIT_STATUS_NORMAL;
	process_state = STATE_NOT_RUNNING;
	read_channel = CHANNEL_STDOUT;
	redirect_flags = REDIRECT_ALL;

	stdin_sama = p_stdin;
	stdout_chan = p_stdout;
	stderr_chan = p_stderr;
}

void Process::set_environment(const HashMap<String, String> &p_env, bool p_override) {
	environment = p_env;

	if (!p_override) {
		_get_system_env(environment);
	}
}

void Process::set_read_channel(Process::ProcessChannel p_channel) {
	ERR_FAIL_COND(p_channel == CHANNEL_STDIN);
	read_channel = p_channel;
}

Process::~Process() {
	stdout_chan->close();
	stderr_chan->close();
	stdin_sama->close();

	memdelete(stdout_chan);
	memdelete(stderr_chan);
	memdelete(stdin_sama);
}

Process::Channel::Channel() {
	closed = true;
}
