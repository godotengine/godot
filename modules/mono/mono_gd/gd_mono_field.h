/*************************************************************************/
/*  gd_mono_field.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "i_mono_class_member.h"

class GDMonoField : public IMonoClassMember {
	GDMonoClass *owner;
	MonoClassField *mono_field;

	StringName name;
	ManagedType type;

	bool attrs_fetched;
	MonoCustomAttrInfo *attributes;

public:
	virtual GDMonoClass *get_enclosing_class() const final { return owner; }

	virtual MemberType get_member_type() const final { return MEMBER_TYPE_FIELD; }

	virtual StringName get_name() const final { return name; }

	virtual bool is_static() final;
	virtual Visibility get_visibility() final;

	virtual bool has_attribute(GDMonoClass *p_attr_class) final;
	virtual MonoObject *get_attribute(GDMonoClass *p_attr_class) final;
	void fetch_attributes();

	_FORCE_INLINE_ ManagedType get_type() const { return type; }

	void set_value(MonoObject *p_object, MonoObject *p_value);
	void set_value_raw(MonoObject *p_object, void *p_ptr);
	void set_value_from_variant(MonoObject *p_object, const Variant &p_value);

	MonoObject *get_value(MonoObject *p_object);

	bool get_bool_value(MonoObject *p_object);
	int get_int_value(MonoObject *p_object);
	String get_string_value(MonoObject *p_object);

	GDMonoField(MonoClassField *p_mono_field, GDMonoClass *p_owner);
	~GDMonoField();
};

#endif // GDMONOFIELD_H
