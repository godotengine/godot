/**************************************************************************/
/*  gd_mono_internals.h                                                   */
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

#ifndef GD_MONO_INTERNALS_H
#define GD_MONO_INTERNALS_H

#include <mono/jit/jit.h>

#include "../utils/macros.h"

#include "core/object.h"

namespace GDMonoInternals {
void tie_managed_to_unmanaged(MonoObject *managed, Object *unmanaged);

/**
 * Do not call this function directly.
 * Use GDMonoUtils::debug_unhandled_exception(MonoException *) instead.
 */
void unhandled_exception(MonoException *p_exc);

void gd_unhandled_exception_event(MonoException *p_exc);

void check_call_error(const String &p_method, const Variant *p_instance, const Variant **p_args, int p_arg_count, const Variant::CallError &p_error);
} // namespace GDMonoInternals

#endif // GD_MONO_INTERNALS_H
