/*************************************************************************/
/*  test_log.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "test_log.h"

#include "core/os/os.h"

Ref<TestLog::LogMessage> TestLog::LogMessage::log(LogLevel level, const String &script_path, const String &test_func, const String &msg) {
	Ref<LogMessage> message(memnew(LogMessage));
	message->m_time = OS::get_singleton()->get_unix_time();
	message->m_level = level;
	message->m_script_path = script_path;
	message->m_test_func = test_func;
	message->m_message = msg;
	return message;
}

Ref<TestLog::LogMessage> TestLog::LogMessage::trace(const String &script_path, const String &test_func, const String &msg) {
	return log(TRACE, script_path, test_func, msg);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::debug(const String &script_path, const String &test_func, const String &msg) {
	return log(DEBUG, script_path, test_func, msg);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::info(const String &script_path, const String &test_func, const String &msg) {
	return log(INFO, script_path, test_func, msg);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::warn(const String &script_path, const String &test_func, const String &msg) {
	return log(WARN, script_path, test_func, msg);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::error(const String &script_path, const String &test_func, const String &msg) {
	return log(ERROR, script_path, test_func, msg);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::fatal(const String &script_path, const String &test_func, const String &msg) {
	return log(FATAL, script_path, test_func, msg);
}

Color TestLog::LogMessage::level_to_color(LogLevel level) {
	switch (level) {
		case TRACE:
			return Color::named("gray");
		case DEBUG:
			return Color::named("lightgray");
		case INFO:
			return Color::named("white");
		case WARN:
			return Color::named("yellow");
		case ERROR:
			return Color::named("orange");
		case FATAL:
			return Color::named("red");
	}
	return Color();
}

uint64_t TestLog::LogMessage::time() const {
	return m_time;
}

TestLog::LogLevel TestLog::LogMessage::level() const {
	return m_level;
}

const String &TestLog::LogMessage::script_path() const {
	return m_script_path;
}

const String &TestLog::LogMessage::test_func() const {
	return m_test_func;
}

const String &TestLog::LogMessage::message() const {
	return m_message;
}

Dictionary TestLog::LogMessage::to_dict() const {
	Dictionary result;
	result["time"] = m_time;
	result["level"] = m_level;
	result["script_path"] = m_script_path;
	result["test_func"] = m_test_func;
	result["message"] = m_message;
	return result;
}

void TestLog::LogMessage::_bind_methods() {
	ClassDB::bind_method(D_METHOD("time"), &TestLog::LogMessage::time);
	ClassDB::bind_method(D_METHOD("level"), &TestLog::LogMessage::level);
	ClassDB::bind_method(D_METHOD("script_path"), &TestLog::LogMessage::script_path);
	ClassDB::bind_method(D_METHOD("test_func"), &TestLog::LogMessage::test_func);
	ClassDB::bind_method(D_METHOD("message"), &TestLog::LogMessage::message);
}

TestLog::TestLog() {
	m_filter = LogLevel::INFO;
	m_max_level = LogLevel::TRACE;
}

void TestLog::set_filter(LogLevel filter) {
	m_filter = filter;
}

TestLog::LogLevel TestLog::get_filter() const {
	return m_filter;
}

TestLog::LogLevel TestLog::get_max_level() const {
	return m_max_level;
}

void TestLog::append(const Ref<TestLog> &test_log) {
	int size = test_log->m_messages.size();
	for (int i = 0; i < size; i++) {
		add_message(test_log->m_messages[i]);
	}
}

void TestLog::add_message(Ref<LogMessage> message) {
	if (message->level() >= m_filter) {
		if (m_max_level > message->level()) {
			m_max_level = message->level();
		}
		m_messages.push_back(message);
	}
}

void TestLog::log(LogLevel level, const String &script_path, const String &test_func, const String &msg) {
	add_message(LogMessage::log(level, script_path, test_func, msg));
}

void TestLog::trace(const String &script_path, const String &test_func, const String &msg) {
	log(TRACE, script_path, test_func, msg);
}

void TestLog::debug(const String &script_path, const String &test_func, const String &msg) {
	log(DEBUG, script_path, test_func, msg);
}

void TestLog::info(const String &script_path, const String &test_func, const String &msg) {
	log(INFO, script_path, test_func, msg);
}

void TestLog::warn(const String &script_path, const String &test_func, const String &msg) {
	log(WARN, script_path, test_func, msg);
}

void TestLog::error(const String &script_path, const String &test_func, const String &msg) {
	log(ERROR, script_path, test_func, msg);
}

void TestLog::fatal(const String &script_path, const String &test_func, const String &msg) {
	log(FATAL, script_path, test_func, msg);
}

void TestLog::clear() {
	m_max_level = LogLevel::TRACE;
	m_messages.clear();
}

Array TestLog::to_array() const {
	Array messages;
	int size = m_messages.size();
	messages.resize(size);
	for (int i = 0; i < size; i++) {
		messages[i] = m_messages[i]->to_dict();
	}
	return messages;
}

void TestLog::_bind_methods() {
	BIND_ENUM_CONSTANT(TRACE);
	BIND_ENUM_CONSTANT(DEBUG);
	BIND_ENUM_CONSTANT(INFO);
	BIND_ENUM_CONSTANT(WARN);
	BIND_ENUM_CONSTANT(ERROR);
	BIND_ENUM_CONSTANT(FATAL);

	ClassDB::bind_method(D_METHOD("log", "level", "msg"), &TestLog::log);
	ClassDB::bind_method(D_METHOD("trace", "msg"), &TestLog::trace);
	ClassDB::bind_method(D_METHOD("debug", "msg"), &TestLog::debug);
	ClassDB::bind_method(D_METHOD("info", "msg"), &TestLog::info);
	ClassDB::bind_method(D_METHOD("warn", "msg"), &TestLog::warn);
	ClassDB::bind_method(D_METHOD("error", "msg"), &TestLog::error);
	ClassDB::bind_method(D_METHOD("fatal", "msg"), &TestLog::fatal);
	ClassDB::bind_method(D_METHOD("clear"), &TestLog::clear);
}
