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

Ref<TestLog::LogMessage> TestLog::LogMessage::log(LogLevel p_level, uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message) {
	Ref<LogMessage> message(memnew(LogMessage));
	message->m_time = time;
	message->m_level = p_level;
	message->m_script_path = p_script_path;
	message->m_test_func = p_test_func;
	message->m_message = p_message;
	return message;
}

Ref<TestLog::LogMessage> TestLog::LogMessage::trace(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message) {
	return log(TRACE, time, p_script_path, p_test_func, p_message);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::debug(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message) {
	return log(DEBUG, time, p_script_path, p_test_func, p_message);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::info(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message) {
	return log(INFO, time, p_script_path, p_test_func, p_message);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::warn(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message) {
	return log(WARN, time, p_script_path, p_test_func, p_message);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::error(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message) {
	return log(ERROR, time, p_script_path, p_test_func, p_message);
}

Ref<TestLog::LogMessage> TestLog::LogMessage::fatal(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message) {
	return log(FATAL, time, p_script_path, p_test_func, p_message);
}

Color TestLog::LogMessage::level_to_color(LogLevel p_level) {
	switch (p_level) {
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

void TestLog::set_filter(LogLevel p_filter) {
	m_filter = p_filter;
}

TestLog::LogLevel TestLog::get_filter() const {
	return m_filter;
}

TestLog::LogLevel TestLog::get_max_level() const {
	return m_max_level;
}

void TestLog::get_messages(List<const LogMessage *> *messages) const {
	for (int i = 0; i < m_messages.size(); i++) {
		messages->push_back(*m_messages.get(i));
	}
}

void TestLog::append(const Ref<TestLog> &p_test_log) {
	int size = p_test_log->m_messages.size();
	for (int i = 0; i < size; i++) {
		add_message(p_test_log->m_messages[i]);
	}
}

void TestLog::add_message(Ref<LogMessage> p_message) {
	if (p_message->level() >= m_filter) {
		if (m_max_level > p_message->level()) {
			m_max_level = p_message->level();
		}
		m_messages.push_back(p_message);
	}
}

void TestLog::log(LogLevel p_level, const String &p_script_path, const String &p_test_func, const String &p_message) {
	add_message(LogMessage::log(p_level, OS::get_singleton()->get_unix_time(), p_script_path, p_test_func, p_message));
}

void TestLog::trace(const String &p_script_path, const String &p_test_func, const String &p_message) {
	log(TRACE, p_script_path, p_test_func, p_message);
}

void TestLog::debug(const String &p_script_path, const String &p_test_func, const String &p_message) {
	log(DEBUG, p_script_path, p_test_func, p_message);
}

void TestLog::info(const String &p_script_path, const String &p_test_func, const String &p_message) {
	log(INFO, p_script_path, p_test_func, p_message);
}

void TestLog::warn(const String &p_script_path, const String &p_test_func, const String &p_message) {
	log(WARN, p_script_path, p_test_func, p_message);
}

void TestLog::error(const String &p_script_path, const String &p_test_func, const String &p_message) {
	log(ERROR, p_script_path, p_test_func, p_message);
}

void TestLog::fatal(const String &p_script_path, const String &p_test_func, const String &p_message) {
	log(FATAL, p_script_path, p_test_func, p_message);
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

	ClassDB::bind_method(D_METHOD("log", "level", "message"), &TestLog::log);
	ClassDB::bind_method(D_METHOD("trace", "message"), &TestLog::trace);
	ClassDB::bind_method(D_METHOD("debug", "message"), &TestLog::debug);
	ClassDB::bind_method(D_METHOD("info", "message"), &TestLog::info);
	ClassDB::bind_method(D_METHOD("warn", "message"), &TestLog::warn);
	ClassDB::bind_method(D_METHOD("error", "message"), &TestLog::error);
	ClassDB::bind_method(D_METHOD("fatal", "message"), &TestLog::fatal);
	ClassDB::bind_method(D_METHOD("clear"), &TestLog::clear);
}
