/**************************************************************************/
/*  test_typed_dictionary.h                                               */
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

#include "core/variant/typed_dictionary.h"
#include "tests/test_macros.h"

namespace TestTypedDictionary {

TEST_CASE("[TypedDictionary] Object value init") {
	Object *a = memnew(Object);
	Object *b = memnew(Object);
	TypedDictionary<double, Object *> tdict = {
		{ 0.0, a },
		{ 5.0, b },
	};
	CHECK_EQ(tdict[0.0], Variant(a));
	CHECK_EQ(tdict[5.0], Variant(b));
	memdelete(a);
	memdelete(b);
}

TEST_CASE("[TypedDictionary] RefCounted value init") {
	Ref<RefCounted> a = memnew(RefCounted);
	Ref<RefCounted> b = memnew(RefCounted);
	TypedDictionary<double, Ref<RefCounted>> tdict = {
		{ 0.0, a },
		{ 5.0, b },
	};
	CHECK_EQ(tdict[0.0], Variant(a));
	CHECK_EQ(tdict[5.0], Variant(b));
}

} // namespace TestTypedDictionary
