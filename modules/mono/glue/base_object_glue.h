/**************************************************************************/
/*  base_object_glue.h                                                    */
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

#ifndef BASE_OBJECT_GLUE_H
#define BASE_OBJECT_GLUE_H

#ifdef MONO_GLUE_ENABLED

#include "core/class_db.h"
#include "core/object.h"

#include "../mono_gd/gd_mono_marshal.h"

Object *godot_icall_Object_Ctor(MonoObject *p_obj);

void godot_icall_Object_Disposed(MonoObject *p_obj, Object *p_ptr);

void godot_icall_Reference_Disposed(MonoObject *p_obj, Object *p_ptr, MonoBoolean p_is_finalizer);

MethodBind *godot_icall_Object_ClassDB_get_method(MonoString *p_type, MonoString *p_method);

MonoObject *godot_icall_Object_weakref(Object *p_obj);

int32_t godot_icall_SignalAwaiter_connect(Object *p_source, MonoString *p_signal, Object *p_target, MonoObject *p_awaiter);

// DynamicGodotObject

MonoArray *godot_icall_DynamicGodotObject_SetMemberList(Object *p_ptr);

MonoBoolean godot_icall_DynamicGodotObject_InvokeMember(Object *p_ptr, MonoString *p_name, MonoArray *p_args, MonoObject **r_result);

MonoBoolean godot_icall_DynamicGodotObject_GetMember(Object *p_ptr, MonoString *p_name, MonoObject **r_result);

MonoBoolean godot_icall_DynamicGodotObject_SetMember(Object *p_ptr, MonoString *p_name, MonoObject *p_value);

MonoString *godot_icall_Object_ToString(Object *p_ptr);

// Register internal calls

void godot_register_object_icalls();

#endif // MONO_GLUE_ENABLED

#endif // BASE_OBJECT_GLUE_H
