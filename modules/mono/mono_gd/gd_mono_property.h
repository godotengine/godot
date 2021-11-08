/*************************************************************************/
/*  gd_mono_property.h                                                   */
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

#ifndef GD_MONO_PROPERTY_H
#define GD_MONO_PROPERTY_H

#include "gd_mono.h"
#include "gd_mono_header.h"
#include "i_mono_class_member.h"

class GDMonoProperty : public IMonoClassMember {
	GDMonoClass *owner;
	MonoProperty *mono_property;

	StringName name;
	ManagedType type;

	bool attrs_fetched;
	MonoCustomAttrInfo *attributes;

	unsigned int param_buffer_size;

public:
	virtual GDMonoClass *get_enclosing_class() const GD_FINAL { return owner; }

	virtual MemberType get_member_type() const GD_FINAL { return MEMBER_TYPE_PROPERTY; }

	virtual StringName get_name() const GD_FINAL { return name; }

	virtual bool is_static() GD_FINAL;
	virtual Visibility get_visibility() GD_FINAL;

	virtual bool has_attribute(GDMonoClass *p_attr_class) GD_FINAL;
	virtual MonoObject *get_attribute(GDMonoClass *p_attr_class) GD_FINAL;
	void fetch_attributes();

	bool has_getter();
	bool has_setter();

	_FORCE_INLINE_ ManagedType get_type() const { return type; }

	void set_value_from_variant(MonoObject *p_object, const Variant &p_value, MonoException **r_exc = NULL);
	MonoObject *get_value(MonoObject *p_object, MonoException **r_exc = NULL);

	bool get_bool_value(MonoObject *p_object);
	int get_int_value(MonoObject *p_object);
	String get_string_value(MonoObject *p_object);

	GDMonoProperty(MonoProperty *p_mono_property, GDMonoClass *p_owner);
	~GDMonoProperty();
};

#endif // GD_MONO_PROPERTY_H
