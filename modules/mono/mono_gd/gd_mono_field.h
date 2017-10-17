/*************************************************************************/
/*  gd_mono_field.h                                                      */
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
#ifndef GDMONOFIELD_H
#define GDMONOFIELD_H

#include "gd_mono.h"
#include "gd_mono_header.h"

class GDMonoField {
	GDMonoClass *owner;
	MonoClassField *mono_field;

	String name;
	ManagedType type;

	bool attrs_fetched;
	MonoCustomAttrInfo *attributes;

public:
	_FORCE_INLINE_ String get_name() const { return name; }
	_FORCE_INLINE_ ManagedType get_type() const { return type; }

	_FORCE_INLINE_ MonoClassField *get_raw() const { return mono_field; }

	void set_value_raw(MonoObject *p_object, void *p_ptr);
	void set_value(MonoObject *p_object, const Variant &p_value);

	_FORCE_INLINE_ MonoObject *get_value(MonoObject *p_object) {
		return mono_field_get_value_object(mono_domain_get(), mono_field, p_object);
	}

	bool get_bool_value(MonoObject *p_object);
	int get_int_value(MonoObject *p_object);
	String get_string_value(MonoObject *p_object);

	bool has_attribute(GDMonoClass *p_attr_class);
	MonoObject *get_attribute(GDMonoClass *p_attr_class);
	void fetch_attributes();

	bool is_static();
	GDMono::MemberVisibility get_visibility();

	GDMonoField(MonoClassField *p_raw_field, GDMonoClass *p_owner);
	~GDMonoField();
};

#endif // GDMONOFIELD_H
