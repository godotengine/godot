/**************************************************************************/
/*  variant_struct.cpp                                                    */
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

#include "variant_struct.h"

#include "core/config/variant_struct_dev_settings.h" // (dev-note: should remove when squashed)

#include "core/io/resource.h"
#include "core/object/object.h"

#include "core/variant/variant_internal.h" // for some reason, this was causing an error

/////////////////////////////////////

void VariantStruct::_init(const StructDefinition *p_definition) {
	if (instance) {
		_unref();
	}

	instance.allocate_(p_definition->size);
	instance->refcount.init();
	instance->state = Internal::InstanceMetaData::STATE_VALID;
	definition = p_definition;
}

void VariantStruct::_ref(StructPtrType p_instance, const StructDefinition *p_definition) {
	if (instance == p_instance) {
		return;
	}

	if (!p_instance->refcount.ref()) {
		ERR_FAIL_MSG("Could not reference the given struct");
	}

	if (instance) {
		_unref();
	}
	instance = p_instance;
	definition = p_definition;
}

void VariantStruct::_unref() {
	ERR_FAIL_NULL(instance);
	if (instance->refcount.unref()) {
		// Call the destructor for the type, then free the memory
		if (definition->destructor != &StructDefinition::trivial_destructor && instance->state == Internal::InstanceMetaData::STATE_VALID) {
			definition->destructor(definition, instance);
		}
		instance->state = Internal::InstanceMetaData::STATE_CLEARED;
		instance.free(); // sets instance to nullptr
	} else {
		instance = nullptr;
	}
}

void VariantStruct::clear() {
#ifdef VSTRUCT_IS_REFERENCE_TYPE
	// Do nothing if it has no reference
	if (!instance) {
		return;
	}

	// Call the destructor for the type
	if (definition->destructor != &StructDefinition::trivial_destructor && instance->state == Internal::InstanceMetaData::STATE_VALID) {
		definition->destructor(definition, instance);
	}
	instance->state = Internal::InstanceMetaData::STATE_CLEARED;
	_unref();
#else
	// (Cow-type behaviour is to simply unref on clear)
	_unref();
#endif
}

bool VariantStruct::is_empty() const {
	if (!instance) {
		return true;
	}
	return instance->state == Internal::InstanceMetaData::STATE_CLEARED;
}

VariantStruct VariantStruct::duplicate(bool deep) const {
	if (is_empty()) {
		ERR_FAIL_V_MSG(VariantStruct(), "Cannot duplicate an empty struct.");
	}

	return recursive_duplicate(deep, RESOURCE_DEEP_DUPLICATE_NONE, 0);
}

VariantStruct VariantStruct::recursive_duplicate(bool p_deep, ResourceDeepDuplicateMode p_deep_subresources_mode, int recursion_count) const {
	if (is_empty()) {
		return VariantStruct();
	}

	VariantStruct ret;
	ret._init(definition);
	definition->copy_constructor(definition, ret.instance, instance);

	if (recursion_count > MAX_RECURSION) {
		ERR_PRINT("Max recursion reached");
		return ret;
	}

	// if not deep, then the copy constructor should be enough
	if (p_deep) {
		bool is_call_chain_end = recursion_count == 0;
		recursion_count++;

		for (const Internal::StructPropertyInfo &E : definition->properties) {
			switch (E.kind) {
				case Internal::StructPropertyInfo::NATIVE_BUILTIN: {
					E.type_info->write(ret.instance->*E,
							E.type_info->read(instance->*E).recursive_duplicate(p_deep, p_deep_subresources_mode, recursion_count));
				} break;
				case Internal::StructPropertyInfo::NATIVE_OBJECT: {
					Variant v(*(Object **)(instance->*E));
					*(Object **)(instance->*E) = v.recursive_duplicate(p_deep, p_deep_subresources_mode, recursion_count);
				} break;
				default:
					ERR_FAIL_V(ret);
			}
		}

		// Variant::recursive_duplicate() may have created a remap cache by now.
		if (is_call_chain_end) {
			Resource::_teardown_duplicate_from_variant();
		}
	}

	return ret;
}

/////////////////////////////////////

void StructDefinition::generic_constructor(const StructDefinition *p_definition, StructPtrType p_target) {
	for (const Internal::StructPropertyInfo &E : p_definition->properties) {
		switch (E.kind) {
			case Internal::StructPropertyInfo::NATIVE_BUILTIN:
				E.type_info->construct(p_target->*E);
				break;
			case Internal::StructPropertyInfo::NATIVE_OBJECT:
				// RefCount?
				*(Object **)(p_target->*E) = nullptr;
				break;
			default:
				ERR_FAIL();
		}
	}
}

void StructDefinition::generic_copy_constructor(const StructDefinition *p_definition, StructPtrType p_target, const StructPtrType p_other) {
	for (const Internal::StructPropertyInfo &E : p_definition->properties) {
		switch (E.kind) {
			case Internal::StructPropertyInfo::NATIVE_BUILTIN:
				E.type_info->copy_construct(p_target->*E, p_other->*E);
				break;
			case Internal::StructPropertyInfo::NATIVE_OBJECT:
				// RefCount?
				*(Object **)(p_target->*E) = *(Object **)(p_other->*E);
				break;
			default:
				ERR_FAIL();
		}
	}
}

void StructDefinition::generic_destructor(const StructDefinition *p_definition, StructPtrType p_target) {
	for (const Internal::StructPropertyInfo &E : p_definition->properties) {
		switch (E.kind) {
			case Internal::StructPropertyInfo::NATIVE_BUILTIN:
				E.type_info->destruct(p_target->*E);
				break;
			case Internal::StructPropertyInfo::NATIVE_OBJECT:
				// RefCount?
				break;
			default:
				ERR_FAIL();
		}
	}
}

void VariantStruct::set(const StringName &p_name, const Variant &p_value, bool &r_valid) {
	if (is_empty()) {
		r_valid = false;
		return;
	}

	const Internal::StructPropertyInfo *prop = definition->get_property_info(p_name);
	if (prop == nullptr) {
		r_valid = false;
		return;
	}

#ifndef VSTRUCT_IS_REFERENCE_TYPE
	_cow_check();
#endif
	void *my_prop = instance->*(*prop);
	switch (prop->kind) {
		case Internal::StructPropertyInfo::NATIVE_BUILTIN: {
			const Variant::Type prop_type = prop->type_info->get_variant_type();
			const Variant::Type val_type = p_value.get_type();
			if (prop_type == val_type) {
				prop->type_info->ptr_set(my_prop, VariantInternal::get_opaque_pointer(&p_value));
				r_valid = true;
			}
		} break;
		case Internal::StructPropertyInfo::NATIVE_OBJECT:
			// RefCount?
			*(Object **)(my_prop) = (Object *)(p_value);
			break;
		default:
			ERR_FAIL();
	}
}

Variant VariantStruct::get(const StringName &p_name, bool &r_valid) const {
	if (is_empty()) {
		r_valid = false;
		return Variant();
	}

	const Internal::StructPropertyInfo *prop = definition->get_property_info(p_name);
	if (prop == nullptr) {
		r_valid = false;
		return Variant();
	}

	const void *my_ptr = instance->*(*prop);
	switch (prop->kind) {
		case Internal::StructPropertyInfo::NATIVE_BUILTIN:
			r_valid = true;
			return prop->type_info->read(my_ptr);
			break;
		case Internal::StructPropertyInfo::NATIVE_OBJECT:
			// RefCount?
			r_valid = true;
			return Variant(*(Object **)(my_ptr));
		default:
			ERR_FAIL_V(Variant());
	}
	return Variant();
}

/////////////////////////////////////

const StructDefinition::StructPropertyInfo *StructDefinition::get_property_info(const int &p_property_index) const {
	ERR_FAIL_INDEX_V(p_property_index, properties.size(), nullptr);
	return &properties[p_property_index];
}

const StructDefinition::StructPropertyInfo *StructDefinition::get_property_info(const StringName &p_property) const {
	for (const StructDefinition::StructPropertyInfo &E : properties) {
		if (E.get_name() == p_property) {
			return &E;
		}
	}
	return nullptr;
}

const StructDefinition::StructPropertyInfo *StructDefinition::get_property_info(const String &p_property) const {
	uint32_t hash = p_property.hash();
	for (const StructDefinition::StructPropertyInfo &E : properties) {
		if (hash == E.get_name().hash() && E.get_name() == p_property) {
			return &E;
		}
	}
	return nullptr;
}

const StructDefinition::StructPropertyInfo *StructDefinition::get_property_info(const Variant &p_property) const {
	// variant_internal.h was a problem here too
	// switch (p_property.get_type()) {
	// 	case Variant::Type::INT: {
	// 		return get_property_info(VariantInternalAccessor<int>::get(&p_property));
	// 	} break;
	// 	case Variant::Type::STRING: {
	// 		return get_property_info(VariantInternalAccessor<String>::get(&p_property));
	// 	} break;
	// 	case Variant::Type::STRING_NAME: {
	// 		return get_property_info(VariantInternalAccessor<StringName>::get(&p_property));
	// 	} break;
	// }

	// (again, slightly less efficient)
	switch (p_property.get_type()) {
		case Variant::Type::INT: {
			return get_property_info(p_property.operator int());
		} break;
		case Variant::Type::STRING: {
			return get_property_info(p_property.operator String());
		} break;
		case Variant::Type::STRING_NAME: {
			return get_property_info(p_property.operator StringName());
		} break;
	}
	return nullptr;
}

VariantStruct StructDefinition::create_instance() const {
	VariantStruct ret;
	ret._init(this);
	constructor(this, ret.instance);
	return ret;
}

/////////////////////////////////////

HashMap<StringName, StructDefinition *> qualified_struct_definitions;
Vector<StructDefinition *> qualifiedd_definitions_to_clear;

void StructDefinition::_register_struct_definition(StructDefinition *p_definition, bool to_clear) {
	if (to_clear) {
		qualifiedd_definitions_to_clear.push_back(p_definition);
	}
	if (qualified_struct_definitions.has(p_definition->qualified_name)) {
		ERR_FAIL_MSG("Multiple StructDefinitions have identical qualified names");
	}
	qualified_struct_definitions.insert(p_definition->qualified_name, p_definition);
}

void StructDefinition::clean_struct_definitions() {
	for (StructDefinition *E : qualifiedd_definitions_to_clear) {
		memdelete(E);
	}
	qualified_struct_definitions.clear();
	qualifiedd_definitions_to_clear.clear();
}
