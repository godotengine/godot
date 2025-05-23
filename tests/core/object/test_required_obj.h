/**************************************************************************/
/*  test_required_obj.h                                                   */
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

#include "core/object/ref_counted.h"
#include "core/object/required_obj.h"
#include "tests/test_macros.h"

namespace TestRequiredObj {

TEST_CASE("[RequiredObj] Constructor assertions") {
	static_assert(!std::is_constructible_v<RequiredObj<Object>>, "RequiredObj shouldn't allow empty construction.");
	static_assert(!std::is_constructible_v<RequiredObj<Object>, std::nullptr_t>, "RequiredObj shouldn't allow nullptr construction.");
	static_assert(!std::is_constructible_v<RequiredObj<Object>, Variant>, "RequiredObj shouldn't allow Variant construction.");

	static_assert(std::is_constructible_v<RequiredObj<Object>, RequiredObj<Object>>, "RequiredObj should allow same-type RequiredObj construction.");
	static_assert(std::is_constructible_v<RequiredObj<Object>, RequiredObj<RefCounted>>, "RequiredObj should allow derived-type RequiredObj construction.");
	static_assert(!std::is_constructible_v<RequiredObj<RefCounted>, RequiredObj<Object>>, "RequiredObj shouldn't allow upcast RequiredObj construction.");

	static_assert(std::is_constructible_v<RequiredObj<Object>, Object *>, "RequiredObj should allow same-type pointer construction.");
	static_assert(std::is_constructible_v<RequiredObj<Object>, RefCounted *>, "RequiredObj should allow derived-type pointer construction.");
	static_assert(!std::is_constructible_v<RequiredObj<RefCounted>, Object *>, "RequiredObj shouldn't allow upcast pointer construction.");

	static_assert(std::is_constructible_v<RequiredObj<RefCounted>, Ref<RefCounted>>, "RequiredObj should allow same-type Ref construction.");
	static_assert(std::is_constructible_v<RequiredObj<RefCounted>, Ref<WeakRef>>, "RequiredObj should allow derived-type Ref construction.");
	static_assert(!std::is_constructible_v<RequiredObj<WeakRef>, Ref<RefCounted>>, "RequiredObj shouldn't allow upcast Ref construction.");
}

TEST_CASE("[RequiredObj] Static construction") {
	RequiredObj<Object> required = RequiredObj<Object>::construct();
	CHECK(required.is_valid());
	memdelete(required.ptr());

	// FIXME: const object pointers don't play nice with `memdelete`
	// const RequiredObj<Object> required_const = RequiredObj<Object>::construct();
	// CHECK(required_const.is_valid());
	// memdelete(required_const.ptr());
}

TEST_CASE("[RequiredObj] Conversion operations") {
	Object *object = memnew(Object);
	RequiredObj<Object> required = RequiredObj(object);
	CHECK(required.is_valid());
	CHECK_EQ(object, required.operator->());
	CHECK_EQ(object, required.ptr());
	memdelete(object);
}

TEST_CASE("[RequiredObj] Const-awareness") {
	static_assert(!std::is_same_v<Object *, const Object *>, "Sanity check.");

	GODOT_DEPRECATED_BEGIN
	RequiredObj<Object> dummy = RequiredObj<Object>::silent_null();
	const RequiredObj<Object> const_dummy = RequiredObj<Object>::silent_null();
	constexpr RequiredObj<Object> constexpr_dummy = RequiredObj<Object>::silent_null();
	GODOT_DEPRECATED_END

	static_assert(std::is_same_v<decltype(dummy.ptr()), Object *>, "A non-const RequiredObj should return non-const pointers.");
	static_assert(std::is_same_v<decltype(const_dummy.ptr()), const Object *>, "A const RequiredObj should return const pointers.");
	static_assert(std::is_same_v<decltype(constexpr_dummy.ptr()), const Object *>, "A constexpr RequiredObj should return const pointers.");
}

template <typename T>
RequiredObj<T> failed_function() {
	ERR_FAIL_V_MSG(RequiredObj<T>::silent_null(), "Should not trigger deprecated warning.");
}

TEST_CASE("[RequiredObj] Allow null as failed return type") {
	ERR_PRINT_OFF
	RequiredObj<Object> required_object = failed_function<Object>();
	ERR_PRINT_ON
	CHECK_MESSAGE(required_object.is_null(), "RequiredObj should allow for null from a failed function.");
}

TEST_CASE("[RequiredObj] Ill-formed behavior") {
	constexpr Object *empty_object = nullptr;
	ERR_PRINT_OFF
	RequiredObj<Object> required_object = RequiredObj(empty_object);
	ERR_PRINT_ON
	CHECK(required_object.is_null());

	Ref<WeakRef> empty_reference = Ref<WeakRef>();
	ERR_PRINT_OFF
	RequiredObj<RefCounted> required_reference = RequiredObj(empty_reference);
	ERR_PRINT_ON
	CHECK(required_reference.is_null());
}

} // namespace TestRequiredObj
