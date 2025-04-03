/**************************************************************************/
/*  core_string_names.h                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/string/string_name.h"

class CoreStringNames {
	inline static CoreStringNames *singleton = nullptr;

public:
	static void create() { singleton = memnew(CoreStringNames); }
	static void free() {
		memdelete(singleton);
		singleton = nullptr;
	}

	_FORCE_INLINE_ static CoreStringNames *get_singleton() { return singleton; }

	const StringName free_ = StringName("free"); // free would conflict with C++ keyword.
	const StringName changed = StringName("changed");
	const StringName script = StringName("script");
	const StringName script_changed = StringName("script_changed");
	const StringName _iter_init = StringName("_iter_init");
	const StringName _iter_next = StringName("_iter_next");
	const StringName _iter_get = StringName("_iter_get");
	const StringName get_rid = StringName("get_rid");
	const StringName _to_string = StringName("_to_string");
	const StringName _custom_features = StringName("_custom_features");

	const StringName x = StringName("x");
	const StringName y = StringName("y");
	const StringName z = StringName("z");
	const StringName w = StringName("w");
	const StringName r = StringName("r");
	const StringName g = StringName("g");
	const StringName b = StringName("b");
	const StringName a = StringName("a");
	const StringName position = StringName("position");
	const StringName size = StringName("size");
	const StringName end = StringName("end");
	const StringName basis = StringName("basis");
	const StringName origin = StringName("origin");
	const StringName normal = StringName("normal");
	const StringName d = StringName("d");
	const StringName h = StringName("h");
	const StringName s = StringName("s");
	const StringName v = StringName("v");
	const StringName r8 = StringName("r8");
	const StringName g8 = StringName("g8");
	const StringName b8 = StringName("b8");
	const StringName a8 = StringName("a8");

	const StringName call = StringName("call");
	const StringName call_deferred = StringName("call_deferred");
	const StringName bind = StringName("bind");
	const StringName notification = StringName("notification");
	const StringName property_list_changed = StringName("property_list_changed");
};

#define CoreStringName(m_name) CoreStringNames::get_singleton()->m_name
