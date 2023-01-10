/**************************************************************************/
/*  string_glue.h                                                         */
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

#ifndef STRING_GLUE_H
#define STRING_GLUE_H

#ifdef MONO_GLUE_ENABLED

#include "../mono_gd/gd_mono_marshal.h"

MonoArray *godot_icall_String_md5_buffer(MonoString *p_str);

MonoString *godot_icall_String_md5_text(MonoString *p_str);

int godot_icall_String_rfind(MonoString *p_str, MonoString *p_what, int p_from);

int godot_icall_String_rfindn(MonoString *p_str, MonoString *p_what, int p_from);

MonoArray *godot_icall_String_sha256_buffer(MonoString *p_str);

MonoString *godot_icall_String_sha256_text(MonoString *p_str);

// Register internal calls

void godot_register_string_icalls();

#endif // MONO_GLUE_ENABLED

#endif // STRING_GLUE_H
