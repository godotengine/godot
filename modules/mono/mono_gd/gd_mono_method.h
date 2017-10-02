/*************************************************************************/
/*  gd_mono_method.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef GD_MONO_METHOD_H
#define GD_MONO_METHOD_H

#include "gd_mono.h"
#include "gd_mono_header.h"

class GDMonoMethod {

	StringName name;

	bool is_instance;
	int params_count;
	ManagedType return_type;
	Vector<ManagedType> param_types;

	bool attrs_fetched;
	MonoCustomAttrInfo *attributes;

	void _update_signature();
	void _update_signature(MonoMethodSignature *p_method_sig);

	friend class GDMonoClass;

	MonoMethod *mono_method;

public:
	_FORCE_INLINE_ StringName get_name() { return name; }

	_FORCE_INLINE_ bool is_static() { return !is_instance; }
	_FORCE_INLINE_ int get_parameters_count() { return params_count; }
	_FORCE_INLINE_ ManagedType get_return_type() { return return_type; }

	void *get_thunk();

	MonoObject *invoke(MonoObject *p_object, const Variant **p_params, MonoObject **r_exc = NULL);
	MonoObject *invoke(MonoObject *p_object, MonoObject **r_exc = NULL);
	MonoObject *invoke_raw(MonoObject *p_object, void **p_params, MonoObject **r_exc = NULL);

	bool has_attribute(GDMonoClass *p_attr_class);
	MonoObject *get_attribute(GDMonoClass *p_attr_class);
	void fetch_attributes();

	String get_full_name(bool p_signature = false) const;
	String get_full_name_no_class() const;
	String get_ret_type_full_name() const;
	String get_signature_desc(bool p_namespaces = false) const;

	GDMonoMethod(StringName p_name, MonoMethod *p_method);
	~GDMonoMethod();
};

#endif // GD_MONO_METHOD_H
