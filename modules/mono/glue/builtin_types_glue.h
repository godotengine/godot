/*************************************************************************/
/*  builtin_types_glue.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef BUILTIN_TYPES_GLUE_H
#define BUILTIN_TYPES_GLUE_H

#include "core/node_path.h"
#include "core/rid.h"

#include <mono/metadata/object.h>

#include "../mono_gd/gd_mono_marshal.h"

MonoBoolean godot_icall_NodePath_is_absolute(NodePath *p_ptr) {
	return (MonoBoolean)p_ptr->is_absolute();
}

uint32_t godot_icall_NodePath_get_name_count(NodePath *p_ptr) {
	return p_ptr->get_name_count();
}

MonoString *godot_icall_NodePath_get_name(NodePath *p_ptr, uint32_t p_idx) {
	return GDMonoMarshal::mono_string_from_godot(p_ptr->get_name(p_idx));
}

uint32_t godot_icall_NodePath_get_subname_count(NodePath *p_ptr) {
	return p_ptr->get_subname_count();
}

MonoString *godot_icall_NodePath_get_subname(NodePath *p_ptr, uint32_t p_idx) {
	return GDMonoMarshal::mono_string_from_godot(p_ptr->get_subname(p_idx));
}

MonoString *godot_icall_NodePath_get_concatenated_subnames(NodePath *p_ptr) {
	return GDMonoMarshal::mono_string_from_godot(p_ptr->get_concatenated_subnames());
}

NodePath *godot_icall_NodePath_get_as_property_path(NodePath *p_ptr) {
	return memnew(NodePath(p_ptr->get_as_property_path()));
}

MonoBoolean godot_icall_NodePath_is_empty(NodePath *p_ptr) {
	return (MonoBoolean)p_ptr->is_empty();
}

uint32_t godot_icall_RID_get_id(RID *p_ptr) {
	return p_ptr->get_id();
}

void godot_register_builtin_type_icalls() {
	mono_add_internal_call("Godot.NativeCalls::godot_icall_NodePath_get_as_property_path", (void *)godot_icall_NodePath_get_as_property_path);
	mono_add_internal_call("Godot.NativeCalls::godot_icall_NodePath_get_concatenated_subnames", (void *)godot_icall_NodePath_get_concatenated_subnames);
	mono_add_internal_call("Godot.NativeCalls::godot_icall_NodePath_get_name", (void *)godot_icall_NodePath_get_name);
	mono_add_internal_call("Godot.NativeCalls::godot_icall_NodePath_get_name_count", (void *)godot_icall_NodePath_get_name_count);
	mono_add_internal_call("Godot.NativeCalls::godot_icall_NodePath_get_subname", (void *)godot_icall_NodePath_get_subname);
	mono_add_internal_call("Godot.NativeCalls::godot_icall_NodePath_get_subname_count", (void *)godot_icall_NodePath_get_subname_count);
	mono_add_internal_call("Godot.NativeCalls::godot_icall_NodePath_is_absolute", (void *)godot_icall_NodePath_is_absolute);
	mono_add_internal_call("Godot.NativeCalls::godot_icall_NodePath_is_empty", (void *)godot_icall_NodePath_is_empty);
	mono_add_internal_call("Godot.NativeCalls::godot_icall_RID_get_id", (void *)godot_icall_RID_get_id);
}

#endif // BUILTIN_TYPES_GLUE_H
