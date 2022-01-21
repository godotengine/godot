/*************************************************************************/
/*  gd_mono_property.cpp                                                 */
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

#include "gd_mono_property.h"

#include "gd_mono_cache.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_utils.h"

#include <mono/metadata/attrdefs.h>

GDMonoProperty::GDMonoProperty(MonoProperty *p_mono_property, GDMonoClass *p_owner) {
	owner = p_owner;
	mono_property = p_mono_property;
	name = String::utf8(mono_property_get_name(mono_property));

	MonoMethod *prop_method = mono_property_get_get_method(mono_property);

	if (prop_method) {
		MonoMethodSignature *getter_sig = mono_method_signature(prop_method);

		MonoType *ret_type = mono_signature_get_return_type(getter_sig);

		type.type_encoding = mono_type_get_type(ret_type);
		MonoClass *ret_type_class = mono_class_from_mono_type(ret_type);
		type.type_class = GDMono::get_singleton()->get_class(ret_type_class);
	} else {
		prop_method = mono_property_get_set_method(mono_property);

		MonoMethodSignature *setter_sig = mono_method_signature(prop_method);

		void *iter = nullptr;
		MonoType *param_raw_type = mono_signature_get_params(setter_sig, &iter);

		type.type_encoding = mono_type_get_type(param_raw_type);
		MonoClass *param_type_class = mono_class_from_mono_type(param_raw_type);
		type.type_class = GDMono::get_singleton()->get_class(param_type_class);
	}

	param_buffer_size = GDMonoMarshal::variant_get_managed_unboxed_size(type);

	attrs_fetched = false;
	attributes = nullptr;
}

GDMonoProperty::~GDMonoProperty() {
	if (attributes) {
		mono_custom_attrs_free(attributes);
	}
}

bool GDMonoProperty::is_static() {
	MonoMethod *prop_method = mono_property_get_get_method(mono_property);
	if (prop_method == nullptr) {
		prop_method = mono_property_get_set_method(mono_property);
	}
	return mono_method_get_flags(prop_method, nullptr) & MONO_METHOD_ATTR_STATIC;
}

IMonoClassMember::Visibility GDMonoProperty::get_visibility() {
	MonoMethod *prop_method = mono_property_get_get_method(mono_property);
	if (prop_method == nullptr) {
		prop_method = mono_property_get_set_method(mono_property);
	}

	switch (mono_method_get_flags(prop_method, nullptr) & MONO_METHOD_ATTR_ACCESS_MASK) {
		case MONO_METHOD_ATTR_PRIVATE:
			return IMonoClassMember::PRIVATE;
		case MONO_METHOD_ATTR_FAM_AND_ASSEM:
			return IMonoClassMember::PROTECTED_AND_INTERNAL;
		case MONO_METHOD_ATTR_ASSEM:
			return IMonoClassMember::INTERNAL;
		case MONO_METHOD_ATTR_FAMILY:
			return IMonoClassMember::PROTECTED;
		case MONO_METHOD_ATTR_PUBLIC:
			return IMonoClassMember::PUBLIC;
		default:
			ERR_FAIL_V(IMonoClassMember::PRIVATE);
	}
}

bool GDMonoProperty::has_attribute(GDMonoClass *p_attr_class) {
	ERR_FAIL_NULL_V(p_attr_class, false);

	if (!attrs_fetched) {
		fetch_attributes();
	}

	if (!attributes) {
		return false;
	}

	return mono_custom_attrs_has_attr(attributes, p_attr_class->get_mono_ptr());
}

MonoObject *GDMonoProperty::get_attribute(GDMonoClass *p_attr_class) {
	ERR_FAIL_NULL_V(p_attr_class, nullptr);

	if (!attrs_fetched) {
		fetch_attributes();
	}

	if (!attributes) {
		return nullptr;
	}

	return mono_custom_attrs_get_attr(attributes, p_attr_class->get_mono_ptr());
}

void GDMonoProperty::fetch_attributes() {
	ERR_FAIL_COND(attributes != nullptr);
	attributes = mono_custom_attrs_from_property(owner->get_mono_ptr(), mono_property);
	attrs_fetched = true;
}

bool GDMonoProperty::has_getter() {
	return mono_property_get_get_method(mono_property) != nullptr;
}

bool GDMonoProperty::has_setter() {
	return mono_property_get_set_method(mono_property) != nullptr;
}

void GDMonoProperty::set_value_from_variant(MonoObject *p_object, const Variant &p_value, MonoException **r_exc) {
	uint8_t *buffer = (uint8_t *)alloca(param_buffer_size);
	unsigned int offset = 0;

	void *params[1] = {
		GDMonoMarshal::variant_to_managed_unboxed(p_value, type, buffer, offset)
	};

#ifdef DEBUG_ENABLED
	CRASH_COND(offset != param_buffer_size);
#endif

	MonoException *exc = nullptr;
	GDMonoUtils::property_set_value(mono_property, p_object, params, &exc);
	if (exc) {
		if (r_exc) {
			*r_exc = exc;
		} else {
			GDMonoUtils::set_pending_exception(exc);
		}
	}
}

MonoObject *GDMonoProperty::get_value(MonoObject *p_object, MonoException **r_exc) {
	MonoException *exc = nullptr;
	MonoObject *ret = GDMonoUtils::property_get_value(mono_property, p_object, nullptr, &exc);

	if (exc) {
		ret = nullptr;
		if (r_exc) {
			*r_exc = exc;
		} else {
			GDMonoUtils::set_pending_exception(exc);
		}
	}

	return ret;
}

bool GDMonoProperty::get_bool_value(MonoObject *p_object) {
	return (bool)GDMonoMarshal::unbox<MonoBoolean>(get_value(p_object));
}

int GDMonoProperty::get_int_value(MonoObject *p_object) {
	return GDMonoMarshal::unbox<int32_t>(get_value(p_object));
}

String GDMonoProperty::get_string_value(MonoObject *p_object) {
	MonoObject *val = get_value(p_object);
	return GDMonoMarshal::mono_string_to_godot((MonoString *)val);
}
