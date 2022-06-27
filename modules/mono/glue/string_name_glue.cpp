/*************************************************************************/
/*  string_name_glue.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/string/string_name.h"
#include "core/string/ustring.h"

#include "../mono_gd/gd_mono_marshal.h"

StringName *godot_icall_StringName_Ctor(MonoString *p_path) {
	return memnew(StringName(GDMonoMarshal::mono_string_to_godot(p_path)));
}

void godot_icall_StringName_Dtor(StringName *p_ptr) {
	ERR_FAIL_NULL(p_ptr);
	memdelete(p_ptr);
}

MonoString *godot_icall_StringName_operator_String(StringName *p_np) {
	return GDMonoMarshal::mono_string_from_godot(p_np->operator String());
}

MonoBoolean godot_icall_StringName_is_empty(StringName *p_ptr) {
	return (MonoBoolean)(*p_ptr == StringName());
}

void godot_register_string_name_icalls() {
	GDMonoUtils::add_internal_call("Godot.StringName::godot_icall_StringName_Ctor", godot_icall_StringName_Ctor);
	GDMonoUtils::add_internal_call("Godot.StringName::godot_icall_StringName_Dtor", godot_icall_StringName_Dtor);
	GDMonoUtils::add_internal_call("Godot.StringName::godot_icall_StringName_operator_String", godot_icall_StringName_operator_String);
	GDMonoUtils::add_internal_call("Godot.StringName::godot_icall_StringName_is_empty", godot_icall_StringName_is_empty);
}

#endif // MONO_GLUE_ENABLED
