/*************************************************************************/
/*  gd_mono_class.cpp                                                    */
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

#include "gd_mono_class.h"

#include <mono/metadata/attrdefs.h>
#include <mono/metadata/debug-helpers.h>

#include "gd_mono_assembly.h"
#include "gd_mono_cache.h"
#include "gd_mono_marshal.h"

String GDMonoClass::get_full_name(MonoClass *p_mono_class) {
	// mono_type_get_full_name is not exposed to embedders, but this seems to do the job
	MonoReflectionType *type_obj = mono_type_get_object(mono_domain_get(), get_mono_type(p_mono_class));

	MonoException *exc = nullptr;
	MonoString *str = GDMonoUtils::object_to_string((MonoObject *)type_obj, &exc);
	UNHANDLED_EXCEPTION(exc);

	return GDMonoMarshal::mono_string_to_godot(str);
}

MonoType *GDMonoClass::get_mono_type(MonoClass *p_mono_class) {
	return mono_class_get_type(p_mono_class);
}

String GDMonoClass::get_full_name() const {
	return get_full_name(mono_class);
}

String GDMonoClass::get_type_desc() const {
	return GDMonoUtils::get_type_desc(get_mono_type());
}

MonoType *GDMonoClass::get_mono_type() const {
	// Careful, you cannot compare two MonoType*.
	// There is mono_metadata_type_equal, how is this different from comparing two MonoClass*?
	return get_mono_type(mono_class);
}

uint32_t GDMonoClass::get_flags() const {
	return mono_class_get_flags(mono_class);
}

bool GDMonoClass::is_static() const {
	uint32_t static_class_flags = MONO_TYPE_ATTR_ABSTRACT | MONO_TYPE_ATTR_SEALED;
	return (get_flags() & static_class_flags) == static_class_flags;
}

bool GDMonoClass::is_assignable_from(GDMonoClass *p_from) const {
	return mono_class_is_assignable_from(mono_class, p_from->mono_class);
}

StringName GDMonoClass::get_namespace() const {
	GDMonoClass *nesting_class = get_nesting_class();
	if (!nesting_class) {
		return namespace_name;
	}
	return nesting_class->get_namespace();
}

String GDMonoClass::get_name_for_lookup() const {
	GDMonoClass *nesting_class = get_nesting_class();
	if (!nesting_class) {
		return class_name;
	}
	return nesting_class->get_name_for_lookup() + "/" + class_name;
}

GDMonoClass *GDMonoClass::get_parent_class() const {
	MonoClass *parent_mono_class = mono_class_get_parent(mono_class);
	return parent_mono_class ? GDMono::get_singleton()->get_class(parent_mono_class) : nullptr;
}

GDMonoClass *GDMonoClass::get_nesting_class() const {
	MonoClass *nesting_type = mono_class_get_nesting_type(mono_class);
	return nesting_type ? GDMono::get_singleton()->get_class(nesting_type) : nullptr;
}

#ifdef TOOLS_ENABLED
Vector<MonoClassField *> GDMonoClass::get_enum_fields() {
	bool class_is_enum = mono_class_is_enum(mono_class);
	ERR_FAIL_COND_V(!class_is_enum, Vector<MonoClassField *>());

	Vector<MonoClassField *> enum_fields;

	void *iter = nullptr;
	MonoClassField *raw_field = nullptr;
	while ((raw_field = mono_class_get_fields(get_mono_ptr(), &iter)) != nullptr) {
		uint32_t field_flags = mono_field_get_flags(raw_field);

		// Enums have an instance field named value__ which holds the value of the enum.
		// Enum constants are static, so we will use this to ignore the value__ field.
		if (field_flags & MONO_FIELD_ATTR_PUBLIC && field_flags & MONO_FIELD_ATTR_STATIC) {
			enum_fields.push_back(raw_field);
		}
	}

	return enum_fields;
}
#endif

bool GDMonoClass::has_attribute(GDMonoClass *p_attr_class) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_NULL_V(p_attr_class, false);
#endif

	if (!attrs_fetched) {
		fetch_attributes();
	}

	if (!attributes) {
		return false;
	}

	return mono_custom_attrs_has_attr(attributes, p_attr_class->get_mono_ptr());
}

MonoObject *GDMonoClass::get_attribute(GDMonoClass *p_attr_class) {
#ifdef DEBUG_ENABLED
	ERR_FAIL_NULL_V(p_attr_class, nullptr);
#endif

	if (!attrs_fetched) {
		fetch_attributes();
	}

	if (!attributes) {
		return nullptr;
	}

	return mono_custom_attrs_get_attr(attributes, p_attr_class->get_mono_ptr());
}

void GDMonoClass::fetch_attributes() {
	ERR_FAIL_COND(attributes != nullptr);

	attributes = mono_custom_attrs_from_class(get_mono_ptr());
	attrs_fetched = true;
}

void GDMonoClass::fetch_methods_with_godot_api_checks(GDMonoClass *p_native_base) {
	CRASH_COND(!CACHED_CLASS(GodotObject)->is_assignable_from(this));

	if (methods_fetched) {
		return;
	}

	void *iter = nullptr;
	MonoMethod *raw_method = nullptr;
	while ((raw_method = mono_class_get_methods(get_mono_ptr(), &iter)) != nullptr) {
		StringName name = String::utf8(mono_method_get_name(raw_method));

		// get_method implicitly fetches methods and adds them to this->methods
		GDMonoMethod *method = get_method(raw_method, name);
		ERR_CONTINUE(!method);

		if (method->get_name() != name) {
#ifdef DEBUG_ENABLED
			String fullname = method->get_ret_type_full_name() + " " + name + "(" + method->get_signature_desc(true) + ")";
			WARN_PRINT("Method '" + fullname + "' is hidden by Godot API method. Should be '" +
					method->get_full_name_no_class() + "'. In class '" + namespace_name + "." + class_name + "'.");
#endif
			continue;
		}

#ifdef DEBUG_ENABLED
		// For debug builds, we also fetched from native base classes as well before if this is not a native base class.
		// This allows us to warn the user here if he is using snake_case by mistake.

		if (p_native_base != this) {
			GDMonoClass *native_top = p_native_base;
			while (native_top) {
				GDMonoMethod *m = native_top->get_method(name, method->get_parameters_count());

				if (m && m->get_name() != name) {
					// found
					String fullname = m->get_ret_type_full_name() + " " + name + "(" + m->get_signature_desc(true) + ")";
					WARN_PRINT("Method '" + fullname + "' should be '" + m->get_full_name_no_class() +
							"'. In class '" + namespace_name + "." + class_name + "'.");
					break;
				}

				if (native_top == CACHED_CLASS(GodotObject)) {
					break;
				}

				native_top = native_top->get_parent_class();
			}
		}
#endif

		uint32_t flags = mono_method_get_flags(method->mono_method, nullptr);

		if (!(flags & MONO_METHOD_ATTR_VIRTUAL)) {
			continue;
		}

		// Virtual method of Godot Object derived type, let's try to find GodotMethod attribute

		GDMonoClass *top = p_native_base;

		while (top) {
			GDMonoMethod *base_method = top->get_method(name, method->get_parameters_count());

			if (base_method && base_method->has_attribute(CACHED_CLASS(GodotMethodAttribute))) {
				// Found base method with GodotMethod attribute.
				// We get the original API method name from this attribute.
				// This name must point to the virtual method.

				MonoObject *attr = base_method->get_attribute(CACHED_CLASS(GodotMethodAttribute));

				StringName godot_method_name = CACHED_FIELD(GodotMethodAttribute, methodName)->get_string_value(attr);
#ifdef DEBUG_ENABLED
				CRASH_COND(godot_method_name == StringName());
#endif
				MethodKey key = MethodKey(godot_method_name, method->get_parameters_count());
				GDMonoMethod **existing_method = methods.getptr(key);
				if (existing_method) {
					memdelete(*existing_method); // Must delete old one
				}
				methods.set(key, method);

				break;
			}

			if (top == CACHED_CLASS(GodotObject)) {
				break;
			}

			top = top->get_parent_class();
		}
	}

	methods_fetched = true;
}

GDMonoMethod *GDMonoClass::get_fetched_method_unknown_params(const StringName &p_name) {
	ERR_FAIL_COND_V(!methods_fetched, nullptr);

	const MethodKey *k = nullptr;

	while ((k = methods.next(k))) {
		if (k->name == p_name) {
			return methods.get(*k);
		}
	}

	return nullptr;
}

bool GDMonoClass::has_fetched_method_unknown_params(const StringName &p_name) {
	return get_fetched_method_unknown_params(p_name) != nullptr;
}

bool GDMonoClass::implements_interface(GDMonoClass *p_interface) {
	return mono_class_implements_interface(mono_class, p_interface->get_mono_ptr());
}

bool GDMonoClass::has_public_parameterless_ctor() {
	GDMonoMethod *ctor = get_method(".ctor", 0);
	return ctor && ctor->get_visibility() == IMonoClassMember::PUBLIC;
}

GDMonoMethod *GDMonoClass::get_method(const StringName &p_name, uint16_t p_params_count) {
	MethodKey key = MethodKey(p_name, p_params_count);

	GDMonoMethod **match = methods.getptr(key);

	if (match) {
		return *match;
	}

	if (methods_fetched) {
		return nullptr;
	}

	MonoMethod *raw_method = mono_class_get_method_from_name(mono_class, String(p_name).utf8().get_data(), p_params_count);

	if (raw_method) {
		GDMonoMethod *method = memnew(GDMonoMethod(p_name, raw_method));
		methods.set(key, method);

		return method;
	}

	return nullptr;
}

GDMonoMethod *GDMonoClass::get_method(MonoMethod *p_raw_method) {
	MonoMethodSignature *sig = mono_method_signature(p_raw_method);

	int params_count = mono_signature_get_param_count(sig);
	StringName method_name = String::utf8(mono_method_get_name(p_raw_method));

	return get_method(p_raw_method, method_name, params_count);
}

GDMonoMethod *GDMonoClass::get_method(MonoMethod *p_raw_method, const StringName &p_name) {
	MonoMethodSignature *sig = mono_method_signature(p_raw_method);
	int params_count = mono_signature_get_param_count(sig);
	return get_method(p_raw_method, p_name, params_count);
}

GDMonoMethod *GDMonoClass::get_method(MonoMethod *p_raw_method, const StringName &p_name, uint16_t p_params_count) {
	ERR_FAIL_NULL_V(p_raw_method, nullptr);

	MethodKey key = MethodKey(p_name, p_params_count);

	GDMonoMethod **match = methods.getptr(key);

	if (match) {
		return *match;
	}

	GDMonoMethod *method = memnew(GDMonoMethod(p_name, p_raw_method));
	methods.set(key, method);

	return method;
}

GDMonoMethod *GDMonoClass::get_method_with_desc(const String &p_description, bool p_include_namespace) {
	MonoMethodDesc *desc = mono_method_desc_new(p_description.utf8().get_data(), p_include_namespace);
	MonoMethod *method = mono_method_desc_search_in_class(desc, mono_class);
	mono_method_desc_free(desc);

	if (!method) {
		return nullptr;
	}

	ERR_FAIL_COND_V(mono_method_get_class(method) != mono_class, nullptr);

	return get_method(method);
}

GDMonoField *GDMonoClass::get_field(const StringName &p_name) {
	Map<StringName, GDMonoField *>::Element *result = fields.find(p_name);

	if (result) {
		return result->value();
	}

	if (fields_fetched) {
		return nullptr;
	}

	MonoClassField *raw_field = mono_class_get_field_from_name(mono_class, String(p_name).utf8().get_data());

	if (raw_field) {
		GDMonoField *field = memnew(GDMonoField(raw_field, this));
		fields.insert(p_name, field);

		return field;
	}

	return nullptr;
}

const Vector<GDMonoField *> &GDMonoClass::get_all_fields() {
	if (fields_fetched) {
		return fields_list;
	}

	void *iter = nullptr;
	MonoClassField *raw_field = nullptr;
	while ((raw_field = mono_class_get_fields(mono_class, &iter)) != nullptr) {
		StringName name = String::utf8(mono_field_get_name(raw_field));

		Map<StringName, GDMonoField *>::Element *match = fields.find(name);

		if (match) {
			fields_list.push_back(match->get());
		} else {
			GDMonoField *field = memnew(GDMonoField(raw_field, this));
			fields.insert(name, field);
			fields_list.push_back(field);
		}
	}

	fields_fetched = true;

	return fields_list;
}

GDMonoProperty *GDMonoClass::get_property(const StringName &p_name) {
	Map<StringName, GDMonoProperty *>::Element *result = properties.find(p_name);

	if (result) {
		return result->value();
	}

	if (properties_fetched) {
		return nullptr;
	}

	MonoProperty *raw_property = mono_class_get_property_from_name(mono_class, String(p_name).utf8().get_data());

	if (raw_property) {
		GDMonoProperty *property = memnew(GDMonoProperty(raw_property, this));
		properties.insert(p_name, property);

		return property;
	}

	return nullptr;
}

const Vector<GDMonoProperty *> &GDMonoClass::get_all_properties() {
	if (properties_fetched) {
		return properties_list;
	}

	void *iter = nullptr;
	MonoProperty *raw_property = nullptr;
	while ((raw_property = mono_class_get_properties(mono_class, &iter)) != nullptr) {
		StringName name = String::utf8(mono_property_get_name(raw_property));

		Map<StringName, GDMonoProperty *>::Element *match = properties.find(name);

		if (match) {
			properties_list.push_back(match->get());
		} else {
			GDMonoProperty *property = memnew(GDMonoProperty(raw_property, this));
			properties.insert(name, property);
			properties_list.push_back(property);
		}
	}

	properties_fetched = true;

	return properties_list;
}

const Vector<GDMonoClass *> &GDMonoClass::get_all_delegates() {
	if (delegates_fetched) {
		return delegates_list;
	}

	void *iter = nullptr;
	MonoClass *raw_class = nullptr;
	while ((raw_class = mono_class_get_nested_types(mono_class, &iter)) != nullptr) {
		if (mono_class_is_delegate(raw_class)) {
			StringName name = String::utf8(mono_class_get_name(raw_class));

			Map<StringName, GDMonoClass *>::Element *match = delegates.find(name);

			if (match) {
				delegates_list.push_back(match->get());
			} else {
				GDMonoClass *delegate = memnew(GDMonoClass(String::utf8(mono_class_get_namespace(raw_class)), String::utf8(mono_class_get_name(raw_class)), raw_class, assembly));
				delegates.insert(name, delegate);
				delegates_list.push_back(delegate);
			}
		}
	}

	delegates_fetched = true;

	return delegates_list;
}

const Vector<GDMonoMethod *> &GDMonoClass::get_all_methods() {
	if (!method_list_fetched) {
		void *iter = nullptr;
		MonoMethod *raw_method = nullptr;
		while ((raw_method = mono_class_get_methods(get_mono_ptr(), &iter)) != nullptr) {
			method_list.push_back(memnew(GDMonoMethod(String::utf8(mono_method_get_name(raw_method)), raw_method)));
		}

		method_list_fetched = true;
	}

	return method_list;
}

GDMonoClass::GDMonoClass(const StringName &p_namespace, const StringName &p_name, MonoClass *p_class, GDMonoAssembly *p_assembly) {
	namespace_name = p_namespace;
	class_name = p_name;
	mono_class = p_class;
	assembly = p_assembly;

	attrs_fetched = false;
	attributes = nullptr;

	methods_fetched = false;
	method_list_fetched = false;
	fields_fetched = false;
	properties_fetched = false;
	delegates_fetched = false;
}

GDMonoClass::~GDMonoClass() {
	if (attributes) {
		mono_custom_attrs_free(attributes);
	}

	for (const KeyValue<StringName, GDMonoField *> &E : fields) {
		memdelete(E.value);
	}

	for (const KeyValue<StringName, GDMonoProperty *> &E : properties) {
		memdelete(E.value);
	}

	{
		// Ugly workaround...
		// We may have duplicated values, because we redirect snake_case methods to PascalCasel (only Godot API methods).
		// This way, we end with both the snake_case name and the PascalCasel name paired with the same method.
		// Therefore, we must avoid deleting the same pointer twice.

		int offset = 0;
		Vector<GDMonoMethod *> deleted_methods;
		deleted_methods.resize(methods.size());

		const MethodKey *k = nullptr;
		while ((k = methods.next(k))) {
			GDMonoMethod *method = methods.get(*k);

			if (method) {
				for (int i = 0; i < offset; i++) {
					if (deleted_methods[i] == method) {
						// Already deleted
						goto already_deleted;
					}
				}

				deleted_methods.write[offset] = method;
				++offset;

				memdelete(method);
			}

		already_deleted:;
		}

		methods.clear();
	}

	for (int i = 0; i < method_list.size(); ++i) {
		memdelete(method_list[i]);
	}
}
