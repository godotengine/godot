/*************************************************************************/
/*  gd_mono_method.cpp                                                   */
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
#include "gd_mono_method.h"

#include "gd_mono_class.h"
#include "gd_mono_marshal.h"

void GDMonoMethod::_update_signature() {
	// Apparently MonoMethodSignature needs not to be freed.
	// mono_method_signature caches the result, we don't need to cache it ourselves.

	MonoMethodSignature *method_sig = mono_method_signature(mono_method);
	_update_signature(method_sig);
}

void GDMonoMethod::_update_signature(MonoMethodSignature *p_method_sig) {
	is_instance = mono_signature_is_instance(p_method_sig);
	params_count = mono_signature_get_param_count(p_method_sig);

	MonoType *ret_type = mono_signature_get_return_type(p_method_sig);
	if (ret_type) {
		return_type.type_encoding = mono_type_get_type(ret_type);

		if (return_type.type_encoding != MONO_TYPE_VOID) {
			MonoClass *ret_type_class = mono_class_from_mono_type(ret_type);
			return_type.type_class = GDMono::get_singleton()->get_class(ret_type_class);
		}
	}

	void *iter = NULL;
	MonoType *param_raw_type;
	while ((param_raw_type = mono_signature_get_params(p_method_sig, &iter)) != NULL) {
		ManagedType param_type;

		param_type.type_encoding = mono_type_get_type(param_raw_type);

		if (param_type.type_encoding != MONO_TYPE_VOID) {
			MonoClass *param_type_class = mono_class_from_mono_type(param_raw_type);
			param_type.type_class = GDMono::get_singleton()->get_class(param_type_class);
		}

		param_types.push_back(param_type);
	}
}

void *GDMonoMethod::get_thunk() {
	return mono_method_get_unmanaged_thunk(mono_method);
}

MonoObject *GDMonoMethod::invoke(MonoObject *p_object, const Variant **p_params, MonoObject **r_exc) {
	if (get_return_type().type_encoding != MONO_TYPE_VOID || get_parameters_count() > 0) {
		MonoArray *params = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(MonoObject), get_parameters_count());

		for (int i = 0; i < params_count; i++) {
			MonoObject *boxed_param = GDMonoMarshal::variant_to_mono_object(p_params[i], param_types[i]);
			mono_array_set(params, MonoObject *, i, boxed_param);
		}

		MonoObject *exc = NULL;
		MonoObject *ret = mono_runtime_invoke_array(mono_method, p_object, params, &exc);

		if (exc) {
			if (r_exc) {
				*r_exc = exc;
			} else {
				ERR_PRINT(GDMonoUtils::get_exception_name_and_message(exc).utf8());
				mono_print_unhandled_exception(exc);
			}
		}

		return ret;
	} else {
		MonoObject *exc = NULL;
		mono_runtime_invoke(mono_method, p_object, NULL, &exc);

		if (exc) {
			if (r_exc) {
				*r_exc = exc;
			} else {
				ERR_PRINT(GDMonoUtils::get_exception_name_and_message(exc).utf8());
				mono_print_unhandled_exception(exc);
			}
		}

		return NULL;
	}
}

MonoObject *GDMonoMethod::invoke(MonoObject *p_object, MonoObject **r_exc) {
	ERR_FAIL_COND_V(get_parameters_count() > 0, NULL);
	return invoke_raw(p_object, NULL, r_exc);
}

MonoObject *GDMonoMethod::invoke_raw(MonoObject *p_object, void **p_params, MonoObject **r_exc) {
	MonoObject *exc = NULL;
	MonoObject *ret = mono_runtime_invoke(mono_method, p_object, p_params, &exc);

	if (exc) {
		if (r_exc) {
			*r_exc = exc;
		} else {
			ERR_PRINT(GDMonoUtils::get_exception_name_and_message(exc).utf8());
			mono_print_unhandled_exception(exc);
		}
	}

	return ret;
}

bool GDMonoMethod::has_attribute(GDMonoClass *p_attr_class) {
	ERR_FAIL_NULL_V(p_attr_class, false);

	if (!attrs_fetched)
		fetch_attributes();

	if (!attributes)
		return false;

	return mono_custom_attrs_has_attr(attributes, p_attr_class->get_raw());
}

MonoObject *GDMonoMethod::get_attribute(GDMonoClass *p_attr_class) {
	ERR_FAIL_NULL_V(p_attr_class, NULL);

	if (!attrs_fetched)
		fetch_attributes();

	if (!attributes)
		return NULL;

	return mono_custom_attrs_get_attr(attributes, p_attr_class->get_raw());
}

void GDMonoMethod::fetch_attributes() {
	ERR_FAIL_COND(attributes != NULL);
	attributes = mono_custom_attrs_from_method(mono_method);
	attrs_fetched = true;
}

String GDMonoMethod::get_full_name(bool p_signature) const {
	char *res = mono_method_full_name(mono_method, p_signature);
	String full_name(res);
	mono_free(res);
	return full_name;
}

String GDMonoMethod::get_full_name_no_class() const {
	String res;

	MonoMethodSignature *method_sig = mono_method_signature(mono_method);

	char *ret_str = mono_type_full_name(mono_signature_get_return_type(method_sig));
	res += ret_str;
	mono_free(ret_str);

	res += " ";
	res += name;
	res += "(";

	char *sig_desc = mono_signature_get_desc(method_sig, true);
	res += sig_desc;
	mono_free(sig_desc);

	res += ")";

	return res;
}

String GDMonoMethod::get_ret_type_full_name() const {
	MonoMethodSignature *method_sig = mono_method_signature(mono_method);
	char *ret_str = mono_type_full_name(mono_signature_get_return_type(method_sig));
	String res = ret_str;
	mono_free(ret_str);
	return res;
}

String GDMonoMethod::get_signature_desc(bool p_namespaces) const {
	MonoMethodSignature *method_sig = mono_method_signature(mono_method);
	char *sig_desc = mono_signature_get_desc(method_sig, p_namespaces);
	String res = sig_desc;
	mono_free(sig_desc);
	return res;
}

GDMonoMethod::GDMonoMethod(StringName p_name, MonoMethod *p_method) {
	name = p_name;

	mono_method = p_method;

	attrs_fetched = false;
	attributes = NULL;

	_update_signature();
}

GDMonoMethod::~GDMonoMethod() {
	if (attributes) {
		mono_custom_attrs_free(attributes);
	}
}
