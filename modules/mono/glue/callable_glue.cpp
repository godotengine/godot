/*************************************************************************/
/*  callable_glue.cpp                                                    */
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

#include "../mono_gd/gd_mono_marshal.h"
#include "arguments_vector.h"

MonoObject *godot_icall_Callable_Call(GDMonoMarshal::M_Callable *p_callable, MonoArray *p_args) {
	Callable callable = GDMonoMarshal::managed_to_callable(*p_callable);

	int argc = mono_array_length(p_args);

	ArgumentsVector<Variant> arg_store(argc);
	ArgumentsVector<const Variant *> args(argc);

	for (int i = 0; i < argc; i++) {
		MonoObject *elem = mono_array_get(p_args, MonoObject *, i);
		arg_store.set(i, GDMonoMarshal::mono_object_to_variant(elem));
		args.set(i, &arg_store.get(i));
	}

	Variant result;
	Callable::CallError error;
	callable.call(args.ptr(), argc, result, error);

	return GDMonoMarshal::variant_to_mono_object(result);
}

void godot_icall_Callable_CallDeferred(GDMonoMarshal::M_Callable *p_callable, MonoArray *p_args) {
	Callable callable = GDMonoMarshal::managed_to_callable(*p_callable);

	int argc = mono_array_length(p_args);

	ArgumentsVector<Variant> arg_store(argc);
	ArgumentsVector<const Variant *> args(argc);

	for (int i = 0; i < argc; i++) {
		MonoObject *elem = mono_array_get(p_args, MonoObject *, i);
		arg_store.set(i, GDMonoMarshal::mono_object_to_variant(elem));
		args.set(i, &arg_store.get(i));
	}

	callable.call_deferred(args.ptr(), argc);
}

void godot_register_callable_icalls() {
	GDMonoUtils::add_internal_call("Godot.Callable::godot_icall_Callable_Call", godot_icall_Callable_Call);
	GDMonoUtils::add_internal_call("Godot.Callable::godot_icall_Callable_CallDeferred", godot_icall_Callable_CallDeferred);
}

#endif // MONO_GLUE_ENABLED
