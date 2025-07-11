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

// #include "core/variant/variant_internal.h" // for some reason, this was causing an error

void VariantStruct::set(const StringName &p_name, const Variant &p_value, bool &r_valid) {
	if (is_empty()) {
		r_valid = false;
		return;
	}

	const StructDefinition::StructPropertyInfo *prop = definition->get_property_info(p_name);
	if (prop == nullptr) {
		r_valid = false;
		return;
	}

	const Variant::Type prop_type = prop->type->get_variant_type();
	const Variant::Type val_type = p_value.get_type();

	if (prop_type == val_type) {
		if (prop_type == Variant::OBJECT) {
			// TODO: Check class inheritance
			r_valid = false;
		} else {
#ifdef VSTRUCT_IS_REFERENCE_TYPE
			// prop->type->ptr_set( member_ptr(prop), VariantInternal::get_opaque_pointer(&p_value) ); // couldn't figure out why variant_internal.h was causing errors so commented out for now
			prop->type->write(member_ptr(prop), p_value); // (note: this is slightly less efficient)
#else
			// prop->type->ptr_set( member_ptrw(prop), VariantInternal::get_opaque_pointer(&p_value) ); // couldn't figure out why variant_internal.h was causing errors so commented out for now
			prop->type->write(member_ptrw(prop), p_value); // (note: this is slightly less efficient)
#endif
			r_valid = true;
		}
	} else if (Variant::can_convert(val_type, prop_type)) {
#ifdef VSTRUCT_IS_REFERENCE_TYPE
		prop->type->write(member_ptr(prop), p_value);
#else
		prop->type->write(member_ptrw(prop), p_value);
#endif
		r_valid = true;
	} else {
		r_valid = false;
	}
}

Variant VariantStruct::get(const StringName &p_name, bool &r_valid) const {
	if (is_empty()) {
		r_valid = false;
		return Variant();
	}

	const StructDefinition::StructPropertyInfo *prop = definition->get_property_info(p_name);
	if (prop == nullptr) {
		r_valid = false;
		return Variant();
	}

	return prop->type->read(member_ptr(prop));
}

void StructDefinition::generic_constructor(void *p_struct, const StructDefinition *p_definition) {
#ifdef SHOULD_PRE_CALC_OFFSET_ADDRESS_BY_REFCOUNTSIZE
	uintptr_t struct_address = *reinterpret_cast<uintptr_t *>(&p_struct) - VariantStruct::STRUCT_OFFSET;
#else
	uintptr_t &struct_address = *reinterpret_cast<uintptr_t *>(&p_struct);
#endif
	for (const StructDefinition::StructPropertyInfo &E : p_definition->properties) {
		uintptr_t ptr = struct_address + E.address; // equivalent to member_ptr
		E.type->construct(reinterpret_cast<void *>(ptr));
	}
}

void StructDefinition::generic_copy_constructor(void *p_struct, const StructDefinition *p_definition, const void *p_other) {
#ifdef SHOULD_PRE_CALC_OFFSET_ADDRESS_BY_REFCOUNTSIZE
	uintptr_t struct_address = *reinterpret_cast<uintptr_t *>(&p_struct) - VariantStruct::STRUCT_OFFSET;
	uintptr_t other_address = *reinterpret_cast<uintptr_t *>(&p_other) - VariantStruct::STRUCT_OFFSET;
#else
	uintptr_t &struct_address = *reinterpret_cast<uintptr_t *>(&p_struct);
	uintptr_t &other_address = *reinterpret_cast<uintptr_t *>(&p_other);
#endif
	for (const StructDefinition::StructPropertyInfo &E : p_definition->properties) {
		uintptr_t to_ptr = struct_address + E.address; // equivalent to member_ptr
		uintptr_t from_ptr = other_address + E.address; // equivalent to member_ptr
		E.type->copy_construct(reinterpret_cast<void *>(to_ptr), reinterpret_cast<const void *>(from_ptr));
	}
}

void StructDefinition::generic_destructor(void *p_struct, const StructDefinition *p_definition) {
#ifdef SHOULD_PRE_CALC_OFFSET_ADDRESS_BY_REFCOUNTSIZE
	uintptr_t struct_address = *reinterpret_cast<uintptr_t *>(&p_struct) - VariantStruct::STRUCT_OFFSET;
#else
	uintptr_t &struct_address = *reinterpret_cast<uintptr_t *>(&p_struct);
#endif
	for (const StructDefinition::StructPropertyInfo &E : p_definition->properties) {
		uintptr_t ptr = struct_address + E.address; // equivalent to member_ptr
		E.type->destruct(reinterpret_cast<void *>(ptr));
	}
}

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

const size_t StructDefinition::get_size() const {
	size_t maxalign = 1;
	size_t cap = 0;
	for (const StructPropertyInfo &E : properties) {
		const size_t a = E.type->align();
		const size_t c = E.type->size() + E.address;
		if (c > cap) {
			cap = c;
		}
		if (a > maxalign) {
			maxalign = a;
		}
	}
	size_t overstep = cap % maxalign;
#ifdef SHOULD_PRE_CALC_OFFSET_ADDRESS_BY_REFCOUNTSIZE
	if (overstep == 0) {
		return cap - VariantStruct::STRUCT_OFFSET;
	}
	return cap + maxalign - overstep - VariantStruct::STRUCT_OFFSET;
#else
	if (overstep == 0) {
		return cap;
	}
	return cap + maxalign - overstep;
#endif
}

const StructDefinition::StructPropertyInfo *StructDefinition::get_property_info(const int &p_property_index) const {
	ERR_FAIL_INDEX_V(p_property_index, properties.size(), nullptr);
	return &properties[p_property_index];
}

const StructDefinition::StructPropertyInfo *StructDefinition::get_property_info(const StringName &p_property) const {
	for (const StructDefinition::StructPropertyInfo &E : properties) {
		if (E.name == p_property) {
			return &E;
		}
	}
	return nullptr;
}

const StructDefinition::StructPropertyInfo *StructDefinition::get_property_info(const String &p_property) const {
	uint32_t hash = p_property.hash();
	for (const StructDefinition::StructPropertyInfo &E : properties) {
		if (E.name.hash() == hash && E.name == p_property) {
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
