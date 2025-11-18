/**************************************************************************/
/*  call_error_info.h                                                     */
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
#include "core/variant/callable.h"

class CallErrorInfo : public RefCounted {
	GDCLASS(CallErrorInfo, RefCounted)
public:
	enum CallError {
		CALL_OK = Callable::CallError::CALL_OK,
		CALL_ERROR_INVALID_METHOD = Callable::CallError::CALL_ERROR_INVALID_METHOD,
		CALL_ERROR_INVALID_ARGUMENT = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT,
		CALL_ERROR_TOO_MANY_ARGUMENTS = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS,
		CALL_ERROR_TOO_FEW_ARGUMENTS = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS,
		CALL_ERROR_INSTANCE_IS_NULL = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL,
		CALL_ERROR_METHOD_NOT_CONST = Callable::CallError::CALL_ERROR_METHOD_NOT_CONST,
	};

private:
	CallError call_error = CALL_OK;
	CallError call_inner_error = CALL_OK;
	int argument = 0;
	int expected = 0;

protected:
	static void _bind_methods();

public:
	void set_call_error(CallError p_error, int p_argument, int p_expected);
	CallError get_call_error() const { return call_error; }

	void set_call_inner_error(CallError p_error);
	CallError get_call_inner_error() const { return call_inner_error; }

	int get_expected_arguments() const;
	int get_invalid_argument_index() const;
	Variant::Type get_invalid_argument_type() const;
};

VARIANT_ENUM_CAST(CallErrorInfo::CallError);
