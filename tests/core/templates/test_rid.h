/**************************************************************************/
/*  test_rid.h                                                            */
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

#ifndef TEST_RID_H
#define TEST_RID_H

#include "core/templates/rid.h"

#include "tests/test_macros.h"

namespace TestRID {
TEST_CASE("[RID] Default Constructor") {
	RID rid;

	CHECK(rid.get_id() == 0);
}

TEST_CASE("[RID] Factory method") {
	RID rid = RID::from_uint64(1);

	CHECK(rid.get_id() == 1);
}

TEST_CASE("[RID] Operators") {
	RID rid = RID::from_uint64(1);

	RID rid_zero = RID::from_uint64(0);
	RID rid_one = RID::from_uint64(1);
	RID rid_two = RID::from_uint64(2);

	CHECK_FALSE(rid == rid_zero);
	CHECK(rid == rid_one);
	CHECK_FALSE(rid == rid_two);

	CHECK_FALSE(rid < rid_zero);
	CHECK_FALSE(rid < rid_one);
	CHECK(rid < rid_two);

	CHECK_FALSE(rid <= rid_zero);
	CHECK(rid <= rid_one);
	CHECK(rid <= rid_two);

	CHECK(rid > rid_zero);
	CHECK_FALSE(rid > rid_one);
	CHECK_FALSE(rid > rid_two);

	CHECK(rid >= rid_zero);
	CHECK(rid >= rid_one);
	CHECK_FALSE(rid >= rid_two);

	CHECK(rid != rid_zero);
	CHECK_FALSE(rid != rid_one);
	CHECK(rid != rid_two);
}

TEST_CASE("[RID] 'is_valid' & 'is_null'") {
	RID rid_zero = RID::from_uint64(0);
	RID rid_one = RID::from_uint64(1);

	CHECK_FALSE(rid_zero.is_valid());
	CHECK(rid_zero.is_null());

	CHECK(rid_one.is_valid());
	CHECK_FALSE(rid_one.is_null());
}

TEST_CASE("[RID] 'get_local_index'") {
	CHECK(RID::from_uint64(1).get_local_index() == 1);
	CHECK(RID::from_uint64(4'294'967'295).get_local_index() == 4'294'967'295);
	CHECK(RID::from_uint64(4'294'967'297).get_local_index() == 1);
}
} // namespace TestRID

#endif // TEST_RID_H
