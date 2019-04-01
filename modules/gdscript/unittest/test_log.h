/*************************************************************************/
/*  test_log.h                                                           */
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

#ifndef TEST_LOG_H
#define TEST_LOG_H

#include "core/reference.h"

class TestLog : public Reference {
	GDCLASS(TestLog, Reference);

public:
	enum LogLevel {
		TRACE,
		DEBUG,
		INFO,
		WARN,
		ERROR,
		FATAL
	};

	class LogMessage : public Reference {
		GDCLASS(LogMessage, Reference);

	public:
		static Ref<LogMessage> log(LogLevel p_level, uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message);
		static Ref<LogMessage> trace(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message);
		static Ref<LogMessage> debug(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message);
		static Ref<LogMessage> info(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message);
		static Ref<LogMessage> warn(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message);
		static Ref<LogMessage> error(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message);
		static Ref<LogMessage> fatal(uint64_t time, const String &p_script_path, const String &p_test_func, const String &p_message);

		static Color level_to_color(LogLevel level);
		uint64_t time() const;
		const String &script_path() const;
		const String &test_func() const;
		LogLevel level() const;
		const String &message() const;

		Dictionary to_dict() const;

	protected:
		static void _bind_methods();

		uint64_t m_time;
		LogLevel m_level;
		String m_script_path;
		String m_test_func;
		String m_message;
	};

	TestLog();

	void set_filter(LogLevel p_filter);
	LogLevel get_filter() const;

	LogLevel get_max_level() const;

	void get_messages(List<const LogMessage *> *messages) const;

	void append(const Ref<TestLog> &p_test_log);
	void add_message(Ref<LogMessage> p_message);
	void log(LogLevel p_level, const String &p_script_path, const String &p_test_func, const String &p_message);
	void trace(const String &p_script_path, const String &p_test_func, const String &p_message);
	void debug(const String &p_script_path, const String &p_test_func, const String &p_message);
	void info(const String &p_script_path, const String &p_test_func, const String &p_message);
	void warn(const String &p_script_path, const String &p_test_func, const String &p_message);
	void error(const String &p_script_path, const String &p_test_func, const String &p_message);
	void fatal(const String &p_script_path, const String &p_test_func, const String &p_message);
	void clear();

	Array to_array() const;

protected:
	static void _bind_methods();

private:
	Vector<Ref<LogMessage> > m_messages;
	LogLevel m_filter;
	LogLevel m_max_level;
};

VARIANT_ENUM_CAST(TestLog::LogLevel);

#endif // TEST_LOG_H
