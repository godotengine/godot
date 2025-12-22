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

	const StringName free_ = "free"; // free would conflict with C++ keyword.
	const StringName changed = "changed";
	const StringName script = "script";
	const StringName script_changed = "script_changed";
	const StringName _iter_init = "_iter_init";
	const StringName _iter_next = "_iter_next";
	const StringName _iter_get = "_iter_get";
	const StringName get_rid = "get_rid";
	const StringName _to_string = "_to_string";
	const StringName _custom_features = "_custom_features";

	const StringName x = "x";
	const StringName y = "y";
	const StringName z = "z";
	const StringName w = "w";
	const StringName r = "r";
	const StringName g = "g";
	const StringName b = "b";
	const StringName a = "a";
	const StringName position = "position";
	const StringName size = "size";
	const StringName end = "end";
	const StringName basis = "basis";
	const StringName origin = "origin";
	const StringName normal = "normal";
	const StringName d = "d";
	const StringName h = "h";
	const StringName s = "s";
	const StringName v = "v";
	const StringName r8 = "r8";
	const StringName g8 = "g8";
	const StringName b8 = "b8";
	const StringName a8 = "a8";

	const StringName call = "call";
	const StringName call_deferred = "call_deferred";
	const StringName bind = "bind";
	const StringName notification = "notification";
	const StringName property_list_changed = "property_list_changed";
	const StringName _property_value_changed = "_property_value_changed";
};

#define CoreStringName(m_name) CoreStringNames::get_singleton()->m_name
