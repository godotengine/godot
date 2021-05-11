/*************************************************************************/
/*  gd_mono_field.cpp                                                    */
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

#include "gd_mono_field.h"

#include <mono/metadata/attrdefs.h>

#include "gd_mono_cache.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_utils.h"

void GDMonoField::set_value(MonoObject *p_object, MonoObject *p_value) {
	mono_field_set_value(p_object, mono_field, p_value);
}

void GDMonoField::set_value_raw(MonoObject *p_object, void *p_ptr) {
	mono_field_set_value(p_object, mono_field, &p_ptr);
}

void GDMonoField::set_value_from_variant(MonoObject *p_object, const Variant &p_value) {
	MonoReflectionField *reflfield = mono_field_get_object(mono_domain_get(), owner->get_mono_ptr(), mono_field);

	MonoException *exc = nullptr;
	CACHED_METHOD_THUNK(Marshaling, SetFieldValue)
			.invoke(reflfield, p_object, &p_value, &exc);

	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
	}
}

MonoObject *GDMonoField::get_value(MonoObject *p_object) {
	return mono_field_get_value_object(mono_domain_get(), mono_field, p_object);
}

bool GDMonoField::get_bool_value(MonoObject *p_object) {
	return (bool)GDMonoMarshal::unbox<MonoBoolean>(get_value(p_object));
}

int GDMonoField::get_int_value(MonoObject *p_object) {
	return GDMonoMarshal::unbox<int32_t>(get_value(p_object));
}

String GDMonoField::get_string_value(MonoObject *p_object) {
	MonoObject *val = get_value(p_object);
	return GDMonoMarshal::mono_string_to_godot((MonoString *)val);
}

bool GDMonoField::has_attribute(GDMonoClass *p_attr_class) {
	ERR_FAIL_NULL_V(p_attr_class, false);

	if (!attrs_fetched) {
		fetch_attributes();
	}

	if (!attributes) {
		return false;
	}

	return mono_custom_attrs_has_attr(attributes, p_attr_class->get_mono_ptr());
}

MonoObject *GDMonoField::get_attribute(GDMonoClass *p_attr_class) {
	ERR_FAIL_NULL_V(p_attr_class, nullptr);

	if (!attrs_fetched) {
		fetch_attributes();
	}

	if (!attributes) {
		return nullptr;
	}

	return mono_custom_attrs_get_attr(attributes, p_attr_class->get_mono_ptr());
}

void GDMonoField::fetch_attributes() {
	ERR_FAIL_COND(attributes != nullptr);
	attributes = mono_custom_attrs_from_field(owner->get_mono_ptr(), mono_field);
	attrs_fetched = true;
}

bool GDMonoField::is_static() {
	return mono_field_get_flags(mono_field) & MONO_FIELD_ATTR_STATIC;
}

IMonoClassMember::Visibility GDMonoField::get_visibility() {
	switch (mono_field_get_flags(mono_field) & MONO_FIELD_ATTR_FIELD_ACCESS_MASK) {
		case MONO_FIELD_ATTR_PRIVATE:
			return IMonoClassMember::PRIVATE;
		case MONO_FIELD_ATTR_FAM_AND_ASSEM:
			return IMonoClassMember::PROTECTED_AND_INTERNAL;
		case MONO_FIELD_ATTR_ASSEMBLY:
			return IMonoClassMember::INTERNAL;
		case MONO_FIELD_ATTR_FAMILY:
			return IMonoClassMember::PROTECTED;
		case MONO_FIELD_ATTR_PUBLIC:
			return IMonoClassMember::PUBLIC;
		default:
			ERR_FAIL_V(IMonoClassMember::PRIVATE);
	}
}

GDMonoField::GDMonoField(MonoClassField *p_mono_field, GDMonoClass *p_owner) {
	owner = p_owner;
	mono_field = p_mono_field;
	name = String::utf8(mono_field_get_name(mono_field));
	MonoType *field_type = mono_field_get_type(mono_field);
	type.type_encoding = mono_type_get_type(field_type);
	MonoClass *field_type_class = mono_class_from_mono_type(field_type);
	type.type_class = GDMono::get_singleton()->get_class(field_type_class);

	attrs_fetched = false;
	attributes = nullptr;
}

GDMonoField::~GDMonoField() {
	if (attributes) {
		mono_custom_attrs_free(attributes);
	}
}
