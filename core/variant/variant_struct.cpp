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


// VarStructRef VarStructRef::allocate(const StructDefinition *p_definition) {
// 	VarStructRef ret;
// 	ret.instance.allocate_(p_definition->size);
// 	ret.definition = p_definition;
// 	ret.instance->store = InstanceMetaData::IN_HEAP;
// 	return ret;
// }

// VarStructRef VarStructRef::duplicate(VarStructRef copy_from) {
// 	VarStructRef ret = allocate(copy_from.definition);
// 	ret.definition->copy_constructor(ret, copy_from.instance);
// }

// void VarStructRef::duplicate() {
// 	InstancePtrType copy_from = instance;
// 	new_instance(definition);
// 	definition->copy_constructor(*this, copy_from);
// }

// void VarStructRef::new_instance(const StructDefinition *p_definition) {
// 	if (instance) {
// 		_unref();
// 	}
// 	definition = p_definition;
// 	instance.allocate_(definition->size);
// 	instance->refcount.init();
// 	instance->type = Internal::VarStructMetaData::IN_HEAP;
// }

/////////////////////////////////////

void VariantStruct::_copy_on_write() {
	VarStructRef dupe = Internal::duplicate(_p);
	_unref();
	_ref(dupe);
}

// #ifdef VSTRUCT_IS_REFERENCE_TYPE
// VariantStruct VariantStruct::duplicate() {
// 	VarStructPtrType copy_from = instance;
// 	new_instance(definition);
// 	definition->copy_constructor(instance, definition, copy_from);
// }
// #else
// void VariantStruct::_copy_on_write() {
// 	if (instance->type != Internal::VarStructMetaData::IN_HEAP) {
// 		return;
// 	}

// 	if (instance->refcount.get() > 1) {
// 		VarStructPtrType copy_from = instance;
// 		new_instance(definition);
// 		definition->copy_constructor(instance, definition, copy_from);
// 	}
// }
// #endif

// void VariantStruct::new_instance(const StructDefinition *p_definition) {
// 	_unref();
// 	definition = p_definition;
// 	instance.allocate_(definition->size);
// 	instance->refcount.init();
// 	instance->type = Internal::VarStructMetaData::IN_HEAP;
// }

// void VariantStruct::_unref() {
// 	ERR_FAIL_COND_MSG(instance->type != Internal::VarStructMetaData::IN_HEAP, "Only VariantStructs on the heap can be unreferenced");
// 	if (instance->refcount.unref()) {
// 		// Call the destructor for the type, then free the memory
// 		if (definition->destructor != StructDefinition::trivial_destructor) {
// 			definition->destructor(instance, definition);
// 		}
// 		instance.free(); // sets instance to nullptr
// 	} else {
// 		instance = nullptr;
// 	}
// }
// _unref();

void VariantStruct::clear() {
	if (_p.is_refcounted()) {
		// _p.clear();
	} else {
		_unref();
	}
}

bool VariantStruct::is_empty() const {
	return !_p;
}

/////////////////////////////////////

void StructDefinition::generic_constructor(VarStructRef p_struct) {
	for (const Internal::StructPropertyInfo &E : p_struct.get_definition()->properties) {
		switch (E.kind) {
			case Internal::StructPropertyInfo::NATIVE_BUILTIN:
				E.type_info->construct(p_struct->*E);
				break;
			case Internal::StructPropertyInfo::NATIVE_OBJECT:
				// RefCount?
				*(Object **)(p_struct->*E) = nullptr;
				break;
			default:
				ERR_FAIL();
		}
	}
}

void StructDefinition::generic_copy_constructor(VarStructRef p_struct, const VarStructRef p_other) {
	DEV_ASSERT(p_struct.get_definition() == p_other.get_definition());
	for (const StructDefinition::StructPropertyInfo &E : p_struct.get_definition()->properties) {
		switch (E.kind) {
			case Internal::StructPropertyInfo::NATIVE_BUILTIN:
				E.type_info->copy_construct(p_struct->*E, p_other->*E);
				break;
			case Internal::StructPropertyInfo::NATIVE_OBJECT:
				// RefCount?
				*(Object **)(p_struct->*E) = *(Object **)(p_other->*E);
				break;
			default:
				ERR_FAIL();
		}
	}
}

void StructDefinition::generic_destructor(VarStructRef p_struct) {
	for (const StructDefinition::StructPropertyInfo &E : p_struct.get_definition()->properties) {
		switch (E.kind) {
			case Internal::StructPropertyInfo::NATIVE_BUILTIN:
				E.type_info->destruct(p_struct->*E);
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

	const Internal::StructPropertyInfo *prop = _p.get_definition()->get_property_info(p_name);
	if (prop == nullptr) {
		r_valid = false;
		return;
	}

	void *my_ptr = _p->*(*prop);
	switch (prop->kind) {
		case Internal::StructPropertyInfo::NATIVE_BUILTIN: {
			const Variant::Type prop_type = prop->type_info->get_variant_type();
			const Variant::Type val_type = p_value.get_type();
			if (prop_type == val_type) {
				if (_p.check_cow()) {
					_copy_on_write();
				}
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

	const Internal::StructPropertyInfo *prop = _p.get_definition()->get_property_info(p_name);
	// const StructDefinition::StructPropertyInfo *prop = definition->get_property_info(p_name);
	if (prop == nullptr) {
		r_valid = false;
		return Variant();
	}

	const void *my_ptr = _p->*(*prop);
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
	// return prop->read(this);
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

// /////////////////////////////////////

// void StructDefinition::StructPropertyInfo::construct(VarStructPtrType p_struct) const {
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			type_info->construct(p_struct->*(*this));
// 		break;
// 		case NATIVE_OBJECT:
// 			p_struct.get_data<Object *>(*this) = nullptr;
// 		break;
// 		default:
// 			ERR_FAIL();
// 	}
// }
// void StructDefinition::StructPropertyInfo::copy_construct(VarStructPtrType p_struct, VarStructPtrType p_other) const {
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			type_info->copy_construct(p_struct->*(*this), p_other->*(*this));
// 		break;
// 		case NATIVE_OBJECT:
// 			// NOTE: We do not track ref-count for Object pointers declared through native structs
// 			p_struct.get_data<Object *>(*this) = p_other.get_data<Object *>(*this);
// 		break;
// 		default:
// 			ERR_FAIL();
// 	}
// }
// void StructDefinition::StructPropertyInfo::destruct(VarStructPtrType p_struct) const {
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			type_info->destruct(p_struct->*(*this));
// 		break;
// 		case NATIVE_OBJECT:
// 			// Do nothing
// 		break;
// 		default:
// 			ERR_FAIL();
// 	}
// }
// Variant StructDefinition::StructPropertyInfo::read(const VariantStruct *p_struct) const {
// #ifdef VSTRUCT_IS_REFERENCE_TYPE
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			return type_info->read(p_struct->member_ptr(*this));
// 		case NATIVE_OBJECT:
// 			return Variant(p_struct->member_data<Object *>(*this));
// 		default:
// 			ERR_FAIL_V(Variant());
// 	}
// #else
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			return type_info->read(p_struct->member_ptr(*this));
// 		case NATIVE_OBJECT:
// 			return Variant(p_struct->member_r<Object *>(*this));
// 		default:
// 			ERR_FAIL_V(Variant());
// 	}
// #endif
// }
// void StructDefinition::StructPropertyInfo::ptr_get(const VariantStruct *p_struct, void *p_into) const {
// #ifdef VSTRUCT_IS_REFERENCE_TYPE
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			type_info->ptr_get(p_struct->member_ptr(*this), p_into);
// 		break;
// 		case NATIVE_OBJECT:
// 			*(Object **)p_into = p_struct->member_data<Object *>(*this);
// 		break;
// 		default:
// 			ERR_FAIL();
// 	}
// #else
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			type_info->ptr_get(p_struct->member_ptr(*this), p_into);
// 		break;
// 		case NATIVE_OBJECT:
// 			*(Object **)p_into = p_struct->member_r<Object *>(*this);
// 		break;
// 		default:
// 			ERR_FAIL();
// 	}
// #endif
// }
// void StructDefinition::StructPropertyInfo::write(VariantStruct *p_struct, const Variant &p_value) const {
// #ifdef VSTRUCT_IS_REFERENCE_TYPE
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			type_info->write(p_struct->member_ptr(*this), p_value);
// 		break;
// 		case NATIVE_OBJECT:
// 			// NOTE: We do not track ref-count for Object pointers declared through native structs
// 			p_struct->member_data<Object *>(*this) = p_value;
// 		break;
// 		default:
// 			ERR_FAIL();
// 	}
// #else
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			type_info->write(p_struct->member_ptrw(*this), p_value);
// 		break;
// 		case NATIVE_OBJECT:
// 			// NOTE: We do not track ref-count for Object pointers declared through native structs
// 			p_struct->member_rw<Object *>(*this) = p_value;
// 		break;
// 		default:
// 			ERR_FAIL();
// 	}
// #endif
// }
// void StructDefinition::StructPropertyInfo::ptr_set(VariantStruct *p_struct, const void *p_value) const {
// #ifdef VSTRUCT_IS_REFERENCE_TYPE
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			type_info->ptr_set(p_struct->member_ptr(*this), p_value);
// 		break;
// 		case NATIVE_OBJECT:
// 			// NOTE: We do not track ref-count for Object pointers declared through native structs
// 			p_struct->member_data<Object *>(*this) = *(Object **)p_value;
// 		break;
// 		default:
// 			ERR_FAIL();
// 	}
// #else
// 	switch (kind) {
// 		case NATIVE_BUILTIN:
// 			type_info->ptr_set(p_struct->member_ptrw(*this), p_value);
// 		break;
// 		case NATIVE_OBJECT:
// 			// NOTE: We do not track ref-count for Object pointers declared through native structs
// 			p_struct->member_rw<Object *>(*this) = *(Object **)p_value;
// 		break;
// 		default:
// 			ERR_FAIL();
// 	}
// #endif
// }

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
