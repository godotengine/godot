/**************************************************************************/
/*  gdscript_ast.cpp                                                      */
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

#include "gdscript_ast.h"

#include "../gdscript.h"
#include "../gdscript_cache.h"

#include "core/object/class_db.h"

namespace AST = GDScriptAST;

bool AST::PatternNode::has_bind(const StringName &p_name) {
	return binds.has(p_name);
}

AST::IdentifierNode *AST::PatternNode::get_bind(const StringName &p_name) {
	return binds[p_name];
}

bool AST::SuiteNode::has_local(const StringName &p_name) const {
	if (locals_indices.has(p_name)) {
		return true;
	}
	if (parent_block != nullptr) {
		return parent_block->has_local(p_name);
	}
	return false;
}

const AST::SuiteNode::Local &AST::SuiteNode::get_local(const StringName &p_name) const {
	if (locals_indices.has(p_name)) {
		return locals[locals_indices[p_name]];
	}
	if (parent_block != nullptr) {
		return parent_block->get_local(p_name);
	}
	return empty;
}

bool AST::AnnotationNode::applies_to(uint32_t p_target_kinds) const {
	return (info->target_kind & p_target_kinds) > 0;
}

AST::DataType AST::SuiteNode::Local::get_datatype() const {
	switch (type) {
		case CONSTANT:
			return constant->type_constraint;
		case VARIABLE:
			return variable->type_constraint;
		case PARAMETER:
			return parameter->type_constraint;
		case FOR_VARIABLE:
		case PATTERN_BIND:
			return bind->type_constraint;
		case UNDEFINED:
			return DataType();
	}
	return DataType();
}

String AST::SuiteNode::Local::get_name() const {
	switch (type) {
		case SuiteNode::Local::PARAMETER:
			return "parameter";
		case SuiteNode::Local::CONSTANT:
			return "constant";
		case SuiteNode::Local::VARIABLE:
			return "variable";
		case SuiteNode::Local::FOR_VARIABLE:
			return "for loop iterator";
		case SuiteNode::Local::PATTERN_BIND:
			return "pattern bind";
		case SuiteNode::Local::UNDEFINED:
			return "<undefined>";
		default:
			return String();
	}
}

String AST::DataType::to_string() const {
	switch (kind) {
		case VARIANT:
			return "Variant";
		case BUILTIN:
			if (builtin_type == Variant::NIL) {
				return "null";
			}
			if (builtin_type == Variant::ARRAY && has_container_element_type(0)) {
				return vformat("Array[%s]", get_container_element_type(0).to_string());
			}
			if (builtin_type == Variant::DICTIONARY && has_container_element_types()) {
				return vformat("Dictionary[%s, %s]", get_container_element_type_or_variant(0).to_string(), get_container_element_type_or_variant(1).to_string());
			}
			return Variant::get_type_name(builtin_type);
		case NATIVE:
			if (is_meta_type) {
				return GDScriptNativeClass::get_class_static();
			}
			return native_type.string();
		case CLASS:
			if (class_type->identifier != nullptr) {
				return class_type->identifier->name.string();
			}
			return class_type->fqcn;
		case SCRIPT: {
			if (is_meta_type) {
				return script_type.is_valid() ? script_type->get_class_name().string() : "";
			}
			String name = script_type.is_valid() ? script_type->get_name() : "";
			if (!name.is_empty()) {
				return name;
			}
			name = script_path;
			if (!name.is_empty()) {
				return name;
			}
			return native_type.string();
		}
		case ENUM: {
			// native_type contains either the native class defining the enum
			// or the fully qualified class name of the script defining the enum
			return String(native_type).get_file(); // Remove path, keep filename
		}
		case RESOLVING:
		case UNRESOLVED:
			return "<unresolved type>";
	}

	ERR_FAIL_V_MSG("<unresolved type>", "Kind set outside the enum range.");
}

String AST::DataType::to_property_info_hint_string() const {
	switch (kind) {
		case BUILTIN:
			return Variant::get_type_name(builtin_type);
		case NATIVE:
			return native_type;
		case SCRIPT:
			if (script_type.is_valid() && script_type->get_global_name() != StringName()) {
				return script_type->get_global_name();
			} else {
				return native_type;
			}
		case CLASS:
			if (class_type != nullptr && class_type->get_global_name() != StringName()) {
				return class_type->get_global_name();
			} else {
				return native_type;
			}
		case ENUM:
			return String(native_type).replace("::", ".");
		case VARIANT:
			return "Variant";
		case RESOLVING:
		case UNRESOLVED:
			break;
	}
	ERR_FAIL_V_MSG("Variant", "GDScript bug: Unexpected type kind.");
}

PropertyInfo AST::DataType::to_property_info(const String &p_name) const {
	PropertyInfo result;
	result.name = p_name;
	result.usage = PROPERTY_USAGE_NONE;

	if (!is_hard_type()) {
		result.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
		return result;
	}

	switch (kind) {
		case BUILTIN:
			result.type = builtin_type;
			if (builtin_type == Variant::ARRAY && has_container_element_type(0)) {
				const DataType elem_type = get_container_element_type(0);
				if (elem_type.is_variant()) {
					break;
				}

				result.hint = PROPERTY_HINT_ARRAY_TYPE;
				result.hint_string = elem_type.to_property_info_hint_string();
			} else if (builtin_type == Variant::DICTIONARY && has_container_element_types()) {
				const DataType key_type = get_container_element_type_or_variant(0);
				const DataType value_type = get_container_element_type_or_variant(1);
				if (key_type.is_variant() && value_type.is_variant()) {
					break;
				}

				result.hint = PROPERTY_HINT_DICTIONARY_TYPE;
				result.hint_string = key_type.to_property_info_hint_string() + ";" + value_type.to_property_info_hint_string();
			}
			break;
		case NATIVE:
			result.type = Variant::OBJECT;
			if (is_meta_type) {
				result.class_name = GDScriptNativeClass::get_class_static();
			} else {
				result.class_name = native_type;
			}
			break;
		case SCRIPT:
			result.type = Variant::OBJECT;
			if (is_meta_type) {
				result.class_name = script_type.is_valid() ? script_type->get_class_name() : Script::get_class_static();
			} else if (script_type.is_valid() && script_type->get_global_name() != StringName()) {
				result.class_name = script_type->get_global_name();
			} else {
				result.class_name = native_type;
			}
			break;
		case CLASS:
			result.type = Variant::OBJECT;
			if (is_meta_type) {
				result.class_name = GDScript::get_class_static();
			} else if (class_type != nullptr && class_type->get_global_name() != StringName()) {
				result.class_name = class_type->get_global_name();
			} else {
				result.class_name = native_type;
			}
			break;
		case ENUM:
			if (is_meta_type) {
				result.type = Variant::DICTIONARY;
			} else {
				result.type = Variant::INT;
				result.usage |= PROPERTY_USAGE_CLASS_IS_ENUM;
				result.class_name = String(native_type).replace("::", ".");
			}
			break;
		case VARIANT:
		case RESOLVING:
		case UNRESOLVED:
			result.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
			break;
	}

	return result;
}

static Variant::Type _variant_type_to_typed_array_element_type(Variant::Type p_type) {
	switch (p_type) {
		case Variant::PACKED_BYTE_ARRAY:
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
			return Variant::INT;
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
			return Variant::FLOAT;
		case Variant::PACKED_STRING_ARRAY:
			return Variant::STRING;
		case Variant::PACKED_VECTOR2_ARRAY:
			return Variant::VECTOR2;
		case Variant::PACKED_VECTOR3_ARRAY:
			return Variant::VECTOR3;
		case Variant::PACKED_COLOR_ARRAY:
			return Variant::COLOR;
		case Variant::PACKED_VECTOR4_ARRAY:
			return Variant::VECTOR4;
		default:
			return Variant::NIL;
	}
}

bool AST::DataType::is_typed_container_type() const {
	return kind == AST::DataType::BUILTIN && _variant_type_to_typed_array_element_type(builtin_type) != Variant::NIL;
}

AST::DataType AST::DataType::get_typed_container_type() const {
	AST::DataType type;
	type.kind = AST::DataType::BUILTIN;
	type.builtin_type = _variant_type_to_typed_array_element_type(builtin_type);
	return type;
}

bool AST::DataType::can_reference(const AST::DataType &p_other) const {
	if (p_other.is_meta_type) {
		return false;
	} else if (builtin_type != p_other.builtin_type) {
		return false;
	} else if (builtin_type != Variant::OBJECT) {
		return true;
	}

	if (native_type == StringName()) {
		return true;
	} else if (p_other.native_type == StringName()) {
		return false;
	} else if (native_type != p_other.native_type && !ClassDB::is_parent_class(p_other.native_type, native_type)) {
		return false;
	}

	Ref<Script> script = script_type;
	if (kind == AST::DataType::CLASS && script.is_null()) {
		Error err = OK;
		Ref<GDScript> scr = GDScriptCache::get_shallow_script(script_path, err);
		ERR_FAIL_COND_V_MSG(err, false, vformat(R"(Error while getting cache for script "%s".)", script_path));
		script.reference_ptr(scr->find_class(class_type->fqcn));
	}

	Ref<Script> script_other = p_other.script_type;
	if (p_other.kind == AST::DataType::CLASS && script_other.is_null()) {
		Error err = OK;
		Ref<GDScript> scr = GDScriptCache::get_shallow_script(p_other.script_path, err);
		ERR_FAIL_COND_V_MSG(err, false, vformat(R"(Error while getting cache for script "%s".)", p_other.script_path));
		script_other.reference_ptr(scr->find_class(p_other.class_type->fqcn));
	}

	if (script.is_null()) {
		return true;
	} else if (script_other.is_null()) {
		return false;
	} else if (script != script_other && !script_other->inherits_script(script)) {
		return false;
	}

	return true;
}

bool AST::DataType::operator==(const DataType &p_other) const {
	if (type_source == UNDETECTED || p_other.type_source == UNDETECTED) {
		return true; // Can be considered equal for parsing purposes.
	}

	if (type_source == INFERRED || p_other.type_source == INFERRED) {
		return true; // Can be considered equal for parsing purposes.
	}

	if (kind != p_other.kind) {
		return false;
	}

	switch (kind) {
		case VARIANT:
			return true; // All variants are the same.
		case BUILTIN:
			return builtin_type == p_other.builtin_type;
		case NATIVE:
		case ENUM: // Enums use native_type to identify the enum and its base class.
			return native_type == p_other.native_type;
		case SCRIPT:
			return script_type == p_other.script_type;
		case CLASS:
			return class_type == p_other.class_type || class_type->fqcn == p_other.class_type->fqcn;
		case RESOLVING:
		case UNRESOLVED:
			break;
	}

	return false;
}
