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

	protected:
		static void _bind_methods();

	private:
		LogLevel m_level;
		String m_message;
		uint64_t m_time;
	};

	void log(LogLevel lvl, const String &msg);
	void trace(const String &msg);
	void debug(const String &msg);
	void info(const String &msg);
	void warn(const String &msg);
	void error(const String &msg);
	void fatal(const String &msg);
	void clear();

protected:
	static void _bind_methods();

private:
	Vector<Ref<LogMessage> > m_messages;
};

VARIANT_ENUM_CAST(TestLog::LogLevel);

#endif // TEST_LOG_H
