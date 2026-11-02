/**************************************************************************/
/*  libgodot_errors.cpp                                                   */
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

#ifdef LIBGODOT_ENABLED

#include "core/error/error_macros.h"
#include "core/extension/libgodot.h"

static libgodot_error_callback error_callback = nullptr;
static void *error_callback_userdata = nullptr;

static void forward_error(void *, const char *p_function, const char *p_file, int p_line, const char *p_error, const char *p_message, bool p_editor_notify, ErrorHandlerType p_type) {
	error_callback(error_callback_userdata, p_function, p_file, p_line, p_error, p_message, p_editor_notify, (int)p_type);
}

static ErrorHandlerList error_handler = { forward_error, nullptr, nullptr };

void libgodot_set_error_callback(libgodot_error_callback p_callback, void *p_userdata) {
	// 2dog: handlers run under the same global lock, so none is in flight once this returns.
	remove_error_handler(&error_handler);
	error_callback = p_callback;
	error_callback_userdata = p_userdata;
	if (p_callback) {
		add_error_handler(&error_handler);
	}
}

#endif // LIBGODOT_ENABLED
