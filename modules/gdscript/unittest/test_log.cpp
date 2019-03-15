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

void TestLog::LogMessage::_bind_methods() {
}

void TestLog::log(LogLevel lvl, const String &msg) {
	OS::get_singleton()->get_unix_time();
}

void TestLog::trace(const String &msg) {
	log(TRACE, msg);
}

void TestLog::debug(const String &msg) {
	log(DEBUG, msg);
}

void TestLog::info(const String &msg) {
	log(INFO, msg);
}

void TestLog::warn(const String &msg) {
	log(WARN, msg);
}

void TestLog::error(const String &msg) {
	log(ERROR, msg);
}

void TestLog::fatal(const String &msg) {
	log(FATAL, msg);
}

void TestLog::clear() {
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
