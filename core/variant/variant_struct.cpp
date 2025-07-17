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

#include "core/object/object.h"

// #include "core/variant/variant_internal.h" // for some reason, this was causing an error

/////////////////////////////////////

void VariantStruct::_unref() {
	ERR_FAIL_NULL(instance);
	if (instance->store == Internal::InstanceMetaData::IN_HEAP_ROOT && instance->refcount.unref()) {
		// Call the destructor for the type, then free the memory
		if (definition->destructor != &StructDefinition::trivial_destructor && instance->state == Internal::InstanceMetaData::STATE_VALID) {
			definition->destructor({ instance, definition });
			instance->state == Internal::InstanceMetaData::STATE_CLEARED;
		}
		instance.free(); // sets instance to nullptr
	} else {
		instance = nullptr;
	}
}

void VariantStruct::_ref(VarStructRef p_struct) {
	if (instance == p_struct.instance) {
		return;
	}

	if (instance->store == Internal::InstanceMetaData::IN_HEAP_ROOT && !p_struct.instance->refcount.ref()) {
		ERR_FAIL_MSG("Could not reference the given struct");
		return;
	}

	if (instance) {
		_unref();
	}
	instance = p_struct.instance;
	definition = p_struct.definition;
}

#ifndef VSTRUCT_IS_REFERENCE_TYPE
void VariantStruct::_copy_on_write() {
	DEV_ASSERT(instance && definition); // should be checked before getting here
	DEV_ASSERT(instance->state == Internal::InstanceMetaData::STATE_VALID); // should be checked before getting here
	VarStructRef copy = definition->copy_instance({ instance, definition });
	if (instance) {
		_unref();
	}
	instance = copy.instance;
	definition = copy.definition;
	instance->refcount.init();
}
#endif

void VariantStruct::clear() {
#ifdef VSTRUCT_IS_REFERENCE_TYPE
	// Do nothing if it has no reference
	if (!instance) {
		return;
	}

	// Call the destructor for the type
	if (definition->destructor != &StructDefinition::trivial_destructor && instance->state == Internal::InstanceMetaData::STATE_VALID) {
		definition->destructor({ instance, definition });
		instance->state == Internal::InstanceMetaData::STATE_CLEARED;
	}
	_unref();
#else
	// (Cow-type behaviour is to unref for clear)
	_unref();
#endif
}

bool VariantStruct::is_empty() const {
	if (!instance) {
		return true;
	}
	return instance->state == Internal::InstanceMetaData::STATE_CLEARED;
}

/////////////////////////////////////

void StructDefinition::generic_constructor(VarStructRef p_struct) {
	for (const Internal::StructPropertyInfo &E : p_struct.definition->properties) {
		switch (E.kind) {
			case Internal::StructPropertyInfo::NATIVE_BUILTIN:
				E.type_info->construct(p_struct.instance->*E);
				break;
			case Internal::StructPropertyInfo::NATIVE_OBJECT:
				// RefCount?
				*(Object **)(p_struct.instance->*E) = nullptr;
				break;
			default:
				ERR_FAIL();
		}
	}
}

void StructDefinition::generic_copy_constructor(VarStructRef p_struct, const VarStructRef p_other) {
	DEV_ASSERT(p_struct.definition == p_other.definition); // should be checked by whatever is calling this
	for (const Internal::StructPropertyInfo &E : p_struct.definition->properties) {
		switch (E.kind) {
			case Internal::StructPropertyInfo::NATIVE_BUILTIN:
				E.type_info->copy_construct(p_struct.instance->*E, p_other.instance->*E);
				break;
			case Internal::StructPropertyInfo::NATIVE_OBJECT:
				// RefCount?
				*(Object **)(p_struct.instance->*E) = *(Object **)(p_other.instance->*E);
				break;
			default:
				ERR_FAIL();
		}
	}
}

void StructDefinition::generic_destructor(VarStructRef p_struct) {
	for (const Internal::StructPropertyInfo &E : p_struct.definition->properties) {
		switch (E.kind) {
			case Internal::StructPropertyInfo::NATIVE_BUILTIN:
				E.type_info->destruct(p_struct.instance->*E);
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

	void *my_ptr = instance->*(*prop);
	switch (prop->kind) {
		case Internal::StructPropertyInfo::NATIVE_BUILTIN: {
			const Variant::Type prop_type = prop->type_info->get_variant_type();
			const Variant::Type val_type = p_value.get_type();
			if (prop_type == val_type) {
#ifndef VSTRUCT_IS_REFERENCE_TYPE
				if (_p.check_cow()) {
					_copy_on_write();
				}
#endif
				// prop->type_info->ptr_set(my_ptr, VariantInternal::get_opaque_pointer(&p_value)); // couldn't figure out why variant_internal.h was causing errors so commented out for now
				prop->type_info->write(my_ptr, p_value);
				r_valid = true;
			}
		} break;
		case Internal::StructPropertyInfo::NATIVE_OBJECT:
			// RefCount?
			*(Object **)(my_ptr) = (Object *)(p_value);
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
