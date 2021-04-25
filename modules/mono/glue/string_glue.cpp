/*************************************************************************/
/*  string_glue.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef MONO_GLUE_ENABLED

#include "core/string/ustring.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

#include "../mono_gd/gd_mono_marshal.h"

MonoArray *godot_icall_String_md5_buffer(MonoString *p_str) {
	Vector<uint8_t> ret = GDMonoMarshal::mono_string_to_godot(p_str).md5_buffer();
	// TODO Check possible Array/Vector<uint8_t> problem?
	return GDMonoMarshal::Array_to_mono_array(Variant(ret));
}

MonoString *godot_icall_String_md5_text(MonoString *p_str) {
	String ret = GDMonoMarshal::mono_string_to_godot(p_str).md5_text();
	return GDMonoMarshal::mono_string_from_godot(ret);
}

int godot_icall_String_rfind(MonoString *p_str, MonoString *p_what, int p_from) {
	String what = GDMonoMarshal::mono_string_to_godot(p_what);
	return GDMonoMarshal::mono_string_to_godot(p_str).rfind(what, p_from);
}

int godot_icall_String_rfindn(MonoString *p_str, MonoString *p_what, int p_from) {
	String what = GDMonoMarshal::mono_string_to_godot(p_what);
	return GDMonoMarshal::mono_string_to_godot(p_str).rfindn(what, p_from);
}

MonoArray *godot_icall_String_sha256_buffer(MonoString *p_str) {
	Vector<uint8_t> ret = GDMonoMarshal::mono_string_to_godot(p_str).sha256_buffer();
	return GDMonoMarshal::Array_to_mono_array(Variant(ret));
}

MonoString *godot_icall_String_sha256_text(MonoString *p_str) {
	String ret = GDMonoMarshal::mono_string_to_godot(p_str).sha256_text();
	return GDMonoMarshal::mono_string_from_godot(ret);
}

void godot_register_string_icalls() {
	GDMonoUtils::add_internal_call("Godot.StringExtensions::godot_icall_String_md5_buffer", godot_icall_String_md5_buffer);
	GDMonoUtils::add_internal_call("Godot.StringExtensions::godot_icall_String_md5_text", godot_icall_String_md5_text);
	GDMonoUtils::add_internal_call("Godot.StringExtensions::godot_icall_String_rfind", godot_icall_String_rfind);
	GDMonoUtils::add_internal_call("Godot.StringExtensions::godot_icall_String_rfindn", godot_icall_String_rfindn);
	GDMonoUtils::add_internal_call("Godot.StringExtensions::godot_icall_String_sha256_buffer", godot_icall_String_sha256_buffer);
	GDMonoUtils::add_internal_call("Godot.StringExtensions::godot_icall_String_sha256_text", godot_icall_String_sha256_text);
}

#endif // MONO_GLUE_ENABLED
