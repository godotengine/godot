/*************************************************************************/
/*  test_class_db.h                                                      */
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

#ifndef TEST_CLASS_DB_H
#define TEST_CLASS_DB_H

#include "core/register_core_types.h"

#include "core/core_constants.h"
#include "core/os/os.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/ordered_hash_map.h"
#include "core/variant/variant.h"

#include "tests/test_macros.h"

namespace TestClassDB {

struct TypeReference {
	StringName name;
	bool is_enum = false;
};

struct ConstantData {
	String name;
	int value = 0;
};

struct EnumData {
	StringName name;
	List<ConstantData> constants;

	_FORCE_INLINE_ bool operator==(const EnumData &p_enum) const {
		return p_enum.name == name;
	}
};

struct PropertyData {
	StringName name;
	int index = 0;

	StringName getter;
	StringName setter;
};

struct ArgumentData {
	TypeReference type;
	String name;
	bool has_defval = false;
	Variant defval;
};

struct MethodData {
	StringName name;
	TypeReference return_type;
	List<ArgumentData> arguments;
	bool is_virtual = false;
	bool is_vararg = false;
};

struct SignalData {
	StringName name;
	List<ArgumentData> arguments;
};

struct ExposedClass {
	StringName name;
	StringName base;

	bool is_singleton = false;
	bool is_instantiable = false;
	bool is_reference = false;

	ClassDB::APIType api_type;

	List<ConstantData> constants;
	List<EnumData> enums;
	List<PropertyData> properties;
	List<MethodData> methods;
	List<SignalData> signals_;

	const PropertyData *find_property_by_name(const StringName &p_name) const {
		for (const List<PropertyData>::Element *E = properties.front(); E; E = E->next()) {
			if (E->get().name == p_name) {
				return &E->get();
			}
		}

		return nullptr;
	}

	const MethodData *find_method_by_name(const StringName &p_name) const {
		for (const List<MethodData>::Element *E = methods.front(); E; E = E->next()) {
			if (E->get().name == p_name) {
				return &E->get();
			}
		}

		return nullptr;
	}
};

struct NamesCache {
	StringName variant_type = StaticCString::create("Variant");
	StringName object_class = StaticCString::create("Object");
	StringName reference_class = StaticCString::create("Reference");
	StringName string_type = StaticCString::create("String");
	StringName string_name_type = StaticCString::create("StringName");
	StringName node_path_type = StaticCString::create("NodePath");
	StringName bool_type = StaticCString::create("bool");
	StringName int_type = StaticCString::create("int");
	StringName float_type = StaticCString::create("float");
	StringName void_type = StaticCString::create("void");
	StringName vararg_stub_type = StaticCString::create("@VarArg@");
	StringName vector2_type = StaticCString::create("Vector2");
	StringName rect2_type = StaticCString::create("Rect2");
	StringName vector3_type = StaticCString::create("Vector3");

	// Object not included as it must be checked for all derived classes
	static constexpr int nullable_types_count = 17;
	StringName nullable_types[nullable_types_count] = {
		string_type,
		string_name_type,
		node_path_type,

		StaticCString::create(_STR(Array)),
		StaticCString::create(_STR(Dictionary)),
		StaticCString::create(_STR(Callable)),
		StaticCString::create(_STR(Signal)),

		StaticCString::create(_STR(PackedByteArray)),
		StaticCString::create(_STR(PackedInt32Array)),
		StaticCString::create(_STR(PackedInt64rray)),
		StaticCString::create(_STR(PackedFloat32Array)),
		StaticCString::create(_STR(PackedFloat64Array)),
		StaticCString::create(_STR(PackedStringArray)),
		StaticCString::create(_STR(PackedVector2Array)),
		StaticCString::create(_STR(PackedVector3Array)),
		StaticCString::create(_STR(PackedColorArray)),
	};

	bool is_nullable_type(const StringName &p_type) const {
		for (int i = 0; i < nullable_types_count; i++) {
			if (p_type == nullable_types[i]) {
				return true;
			}
		}

		return false;
	}
};

typedef OrderedHashMap<StringName, ExposedClass> ExposedClasses;

struct Context {
	Vector<StringName> enum_types;
	Vector<StringName> builtin_types;
	ExposedClasses exposed_classes;
	List<EnumData> global_enums;
	NamesCache names_cache;

	const ExposedClass *find_exposed_class(const StringName &p_name) const {
		ExposedClasses::ConstElement elem = exposed_classes.find(p_name);
		return elem ? &elem.value() : nullptr;
	}

	const ExposedClass *find_exposed_class(const TypeReference &p_type_ref) const {
		ExposedClasses::ConstElement elem = exposed_classes.find(p_type_ref.name);
		return elem ? &elem.value() : nullptr;
	}

	bool has_type(const TypeReference &p_type_ref) const {
		if (builtin_types.find(p_type_ref.name) >= 0) {
			return true;
		}

		if (p_type_ref.is_enum) {
			if (enum_types.find(p_type_ref.name) >= 0) {
				return true;
			}

			// Enum not found. Most likely because none of its constants were bound, so it's empty. That's fine. Use int instead.
			return builtin_types.find(names_cache.int_type);
		}

		return false;
	}
};

bool arg_default_value_is_assignable_to_type(const Context &p_context, const Variant &p_val, const TypeReference &p_arg_type, String *r_err_msg = nullptr) {
	if (p_arg_type.name == p_context.names_cache.variant_type) {
		// Variant can take anything
		return true;
	}

	switch (p_val.get_type()) {
		case Variant::NIL:
			return p_context.find_exposed_class(p_arg_type) ||
				   p_context.names_cache.is_nullable_type(p_arg_type.name);
		case Variant::BOOL:
			return p_arg_type.name == p_context.names_cache.bool_type;
		case Variant::INT:
			return p_arg_type.name == p_context.names_cache.int_type ||
				   p_arg_type.name == p_context.names_cache.float_type ||
				   p_arg_type.is_enum;
		case Variant::FLOAT:
			return p_arg_type.name == p_context.names_cache.float_type;
		case Variant::STRING:
		case Variant::STRING_NAME:
			return p_arg_type.name == p_context.names_cache.string_type ||
				   p_arg_type.name == p_context.names_cache.string_name_type ||
				   p_arg_type.name == p_context.names_cache.node_path_type;
		case Variant::NODE_PATH:
			return p_arg_type.name == p_context.names_cache.node_path_type;
		case Variant::TRANSFORM:
		case Variant::TRANSFORM2D:
		case Variant::BASIS:
		case Variant::QUAT:
		case Variant::PLANE:
		case Variant::AABB:
		case Variant::COLOR:
		case Variant::VECTOR2:
		case Variant::RECT2:
		case Variant::VECTOR3:
		case Variant::RID:
		case Variant::ARRAY:
		case Variant::DICTIONARY:
		case Variant::PACKED_BYTE_ARRAY:
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
		case Variant::PACKED_STRING_ARRAY:
		case Variant::PACKED_VECTOR2_ARRAY:
		case Variant::PACKED_VECTOR3_ARRAY:
		case Variant::PACKED_COLOR_ARRAY:
		case Variant::CALLABLE:
		case Variant::SIGNAL:
			return p_arg_type.name == Variant::get_type_name(p_val.get_type());
		case Variant::OBJECT:
			return p_context.find_exposed_class(p_arg_type);
		case Variant::VECTOR2I:
			return p_arg_type.name == p_context.names_cache.vector2_type ||
				   p_arg_type.name == Variant::get_type_name(p_val.get_type());
		case Variant::RECT2I:
			return p_arg_type.name == p_context.names_cache.rect2_type ||
				   p_arg_type.name == Variant::get_type_name(p_val.get_type());
		case Variant::VECTOR3I:
			return p_arg_type.name == p_context.names_cache.vector3_type ||
				   p_arg_type.name == Variant::get_type_name(p_val.get_type());
		default:
			if (r_err_msg) {
				*r_err_msg = "Unexpected Variant type: " + itos(p_val.get_type());
			}
			break;
	}

	return false;
}

void validate_property(const Context &p_context, const ExposedClass &p_class, const PropertyData &p_prop) {
	const MethodData *setter = p_class.find_method_by_name(p_prop.setter);

	// Search it in base classes too
	const ExposedClass *top = &p_class;
	while (!setter && top->base != StringName()) {
		top = p_context.find_exposed_class(top->base);
		TEST_FAIL_COND(!top, "Class not found '", top->base, "'. Inherited by '", top->name, "'.");
		setter = top->find_method_by_name(p_prop.setter);
	}

	const MethodData *getter = p_class.find_method_by_name(p_prop.getter);

	// Search it in base classes too
	top = &p_class;
	while (!getter && top->base != StringName()) {
		top = p_context.find_exposed_class(top->base);
		TEST_FAIL_COND(!top, "Class not found '", top->base, "'. Inherited by '", top->name, "'.");
		getter = top->find_method_by_name(p_prop.getter);
	}

	TEST_FAIL_COND((!setter && !getter),
			"Couldn't find neither the setter nor the getter for property: '", p_class.name, ".", String(p_prop.name), "'.");

	if (setter) {
		int setter_argc = p_prop.index != -1 ? 2 : 1;
		TEST_FAIL_COND(setter->arguments.size() != setter_argc,
				"Invalid property setter argument count: '", p_class.name, ".", String(p_prop.name), "'.");
	}

	if (getter) {
		int getter_argc = p_prop.index != -1 ? 1 : 0;
		TEST_FAIL_COND(getter->arguments.size() != getter_argc,
				"Invalid property setter argument count: '", p_class.name, ".", String(p_prop.name), "'.");
	}

	if (getter && setter) {
		const ArgumentData &setter_first_arg = setter->arguments.back()->get();
		if (getter->return_type.name != setter_first_arg.type.name) {
			// Special case for Node::set_name
			bool whitelisted = getter->return_type.name == p_context.names_cache.string_name_type &&
							   setter_first_arg.type.name == p_context.names_cache.string_type;

			TEST_FAIL_COND(!whitelisted,
					"Return type from getter doesn't match first argument of setter, for property: '", p_class.name, ".", String(p_prop.name), "'.");
		}
	}

	const TypeReference &prop_type_ref = getter ? getter->return_type : setter->arguments.back()->get().type;

	const ExposedClass *prop_class = p_context.find_exposed_class(prop_type_ref);
	if (prop_class) {
		TEST_COND(prop_class->is_singleton,
				"Property type is a singleton: '", p_class.name, ".", String(p_prop.name), "'.");

		if (p_class.api_type == ClassDB::API_CORE) {
			TEST_COND(prop_class->api_type == ClassDB::API_EDITOR,
					"Property '", p_class.name, ".", p_prop.name, "' has type '", prop_class->name,
					"' from the editor API. Core API cannot have dependencies on the editor API.");
		}
	} else {
		// Look for types that don't inherit Object
		TEST_FAIL_COND(!p_context.has_type(prop_type_ref),
				"Property type '", prop_type_ref.name, "' not found: '", p_class.name, ".", String(p_prop.name), "'.");
	}

	if (getter) {
		if (p_prop.index != -1) {
			const ArgumentData &idx_arg = getter->arguments.front()->get();
			if (idx_arg.type.name != p_context.names_cache.int_type) {
				// If not an int, it can be an enum
				TEST_COND(p_context.enum_types.find(idx_arg.type.name) < 0,
						"Invalid type '", idx_arg.type.name, "' for index argument of property getter: '", p_class.name, ".", String(p_prop.name), "'.");
			}
		}
	}

	if (setter) {
		if (p_prop.index != -1) {
			const ArgumentData &idx_arg = setter->arguments.front()->get();
			if (idx_arg.type.name != p_context.names_cache.int_type) {
				// Assume the index parameter is an enum
				// If not an int, it can be an enum
				TEST_COND(p_context.enum_types.find(idx_arg.type.name) < 0,
						"Invalid type '", idx_arg.type.name, "' for index argument of property setter: '", p_class.name, ".", String(p_prop.name), "'.");
			}
		}
	}
}

void validate_method(const Context &p_context, const ExposedClass &p_class, const MethodData &p_method) {
	if (p_method.return_type.name != StringName()) {
		const ExposedClass *return_class = p_context.find_exposed_class(p_method.return_type);
		if (return_class) {
			TEST_COND(return_class->is_singleton,
					"Method return type is a singleton: '", p_class.name, ".", p_method.name, "'.");

			if (p_class.api_type == ClassDB::API_CORE) {
				TEST_COND(return_class->api_type == ClassDB::API_EDITOR,
						"Method '", p_class.name, ".", p_method.name, "' has return type '", return_class->name,
						"' from the editor API. Core API cannot have dependencies on the editor API.");
			}
		} else {
			// Look for types that don't inherit Object
			TEST_FAIL_COND(!p_context.has_type(p_method.return_type),
					"Method return type '", p_method.return_type.name, "' not found: '", p_class.name, ".", p_method.name, "'.");
		}
	}

	for (const List<ArgumentData>::Element *F = p_method.arguments.front(); F; F = F->next()) {
		const ArgumentData &arg = F->get();

		const ExposedClass *arg_class = p_context.find_exposed_class(arg.type);
		if (arg_class) {
			TEST_COND(arg_class->is_singleton,
					"Argument type is a singleton: '", arg.name, "' of method '", p_class.name, ".", p_method.name, "'.");

			if (p_class.api_type == ClassDB::API_CORE) {
				TEST_COND(arg_class->api_type == ClassDB::API_EDITOR,
						"Argument '", arg.name, "' of method '", p_class.name, ".", p_method.name, "' has type '",
						arg_class->name, "' from the editor API. Core API cannot have dependencies on the editor API.");
			}
		} else {
			// Look for types that don't inherit Object
			TEST_FAIL_COND(!p_context.has_type(arg.type),
					"Argument type '", arg.type.name, "' not found: '", arg.name, "' of method", p_class.name, ".", p_method.name, "'.");
		}

		if (arg.has_defval) {
			String type_error_msg;
			bool arg_defval_assignable_to_type = arg_default_value_is_assignable_to_type(p_context, arg.defval, arg.type, &type_error_msg);
			String err_msg = vformat("Invalid default value for parameter '%s' of method '%s.%s'.", arg.name, p_class.name, p_method.name);
			if (!type_error_msg.is_empty()) {
				err_msg += " " + type_error_msg;
			}
			TEST_COND(!arg_defval_assignable_to_type, err_msg.utf8().get_data());
		}
	}
}

void validate_signal(const Context &p_context, const ExposedClass &p_class, const SignalData &p_signal) {
	for (const List<ArgumentData>::Element *F = p_signal.arguments.front(); F; F = F->next()) {
		const ArgumentData &arg = F->get();

		const ExposedClass *arg_class = p_context.find_exposed_class(arg.type);
		if (arg_class) {
			TEST_COND(arg_class->is_singleton,
					"Argument class is a singleton: '", arg.name, "' of signal '", p_class.name, ".", p_signal.name, "'.");

			if (p_class.api_type == ClassDB::API_CORE) {
				TEST_COND(arg_class->api_type == ClassDB::API_EDITOR,
						"Argument '", arg.name, "' of signal '", p_class.name, ".", p_signal.name, "' has type '",
						arg_class->name, "' from the editor API. Core API cannot have dependencies on the editor API.");
			}
		} else {
			// Look for types that don't inherit Object
			TEST_FAIL_COND(!p_context.has_type(arg.type),
					"Argument type '", arg.type.name, "' not found: '", arg.name, "' of signal", p_class.name, ".", p_signal.name, "'.");
		}
	}
}

void validate_class(const Context &p_context, const ExposedClass &p_exposed_class) {
	bool is_derived_type = p_exposed_class.base != StringName();

	if (!is_derived_type) {
		// Asserts about the base Object class
		TEST_FAIL_COND(p_exposed_class.name != p_context.names_cache.object_class,
				"Class '", p_exposed_class.name, "' has no base class.");
		TEST_FAIL_COND(!p_exposed_class.is_instantiable,
				"Object class is not instantiable.");
		TEST_FAIL_COND(p_exposed_class.api_type != ClassDB::API_CORE,
				"Object class is API is not API_CORE.");
		TEST_FAIL_COND(p_exposed_class.is_singleton,
				"Object class is registered as a singleton.");
	}

	TEST_FAIL_COND((p_exposed_class.is_singleton && p_exposed_class.base != p_context.names_cache.object_class),
			"Singleton base class '", String(p_exposed_class.base), "' is not Object, for class '", p_exposed_class.name, "'.");

	TEST_FAIL_COND((is_derived_type && !p_context.exposed_classes.has(p_exposed_class.base)),
			"Base type '", p_exposed_class.base.operator String(), "' does not exist, for class '", p_exposed_class.name, "'.");

	for (const List<PropertyData>::Element *F = p_exposed_class.properties.front(); F; F = F->next()) {
		validate_property(p_context, p_exposed_class, F->get());
	}

	for (const List<MethodData>::Element *F = p_exposed_class.methods.front(); F; F = F->next()) {
		validate_method(p_context, p_exposed_class, F->get());
	}

	for (const List<SignalData>::Element *F = p_exposed_class.signals_.front(); F; F = F->next()) {
		validate_signal(p_context, p_exposed_class, F->get());
	}
}

void add_exposed_classes(Context &r_context) {
	List<StringName> class_list;
	ClassDB::get_class_list(&class_list);
	class_list.sort_custom<StringName::AlphCompare>();

	while (class_list.size()) {
		StringName class_name = class_list.front()->get();

		ClassDB::APIType api_type = ClassDB::get_api_type(class_name);

		if (api_type == ClassDB::API_NONE) {
			class_list.pop_front();
			continue;
		}

		if (!ClassDB::is_class_exposed(class_name)) {
			MESSAGE(vformat("Ignoring class '%s' because it's not exposed.", class_name).utf8().get_data());
			class_list.pop_front();
			continue;
		}

		if (!ClassDB::is_class_enabled(class_name)) {
			MESSAGE(vformat("Ignoring class '%s' because it's not enabled.", class_name).utf8().get_data());
			class_list.pop_front();
			continue;
		}

		ClassDB::ClassInfo *class_info = ClassDB::classes.getptr(class_name);

		ExposedClass exposed_class;
		exposed_class.name = class_name;
		exposed_class.api_type = api_type;
		exposed_class.is_singleton = Engine::get_singleton()->has_singleton(class_name);
		exposed_class.is_instantiable = class_info->creation_func && !exposed_class.is_singleton;
		exposed_class.is_reference = ClassDB::is_parent_class(class_name, "Reference");
		exposed_class.base = ClassDB::get_parent_class(class_name);

		// Add properties

		List<PropertyInfo> property_list;
		ClassDB::get_property_list(class_name, &property_list, true);

		Map<StringName, StringName> accessor_methods;

		for (const List<PropertyInfo>::Element *E = property_list.front(); E; E = E->next()) {
			const PropertyInfo &property = E->get();

			if (property.usage & PROPERTY_USAGE_GROUP || property.usage & PROPERTY_USAGE_SUBGROUP || property.usage & PROPERTY_USAGE_CATEGORY) {
				continue;
			}

			PropertyData prop;
			prop.name = property.name;
			prop.setter = ClassDB::get_property_setter(class_name, prop.name);
			prop.getter = ClassDB::get_property_getter(class_name, prop.name);

			if (prop.setter != StringName()) {
				accessor_methods[prop.setter] = prop.name;
			}
			if (prop.getter != StringName()) {
				accessor_methods[prop.getter] = prop.name;
			}

			bool valid = false;
			prop.index = ClassDB::get_property_index(class_name, prop.name, &valid);
			TEST_FAIL_COND(!valid, "Invalid property: '", exposed_class.name, ".", String(prop.name), "'.");

			exposed_class.properties.push_back(prop);
		}

		// Add methods

		List<MethodInfo> virtual_method_list;
		ClassDB::get_virtual_methods(class_name, &virtual_method_list, true);

		List<MethodInfo> method_list;
		ClassDB::get_method_list(class_name, &method_list, true);
		method_list.sort();

		for (List<MethodInfo>::Element *E = method_list.front(); E; E = E->next()) {
			const MethodInfo &method_info = E->get();

			int argc = method_info.arguments.size();

			if (method_info.name.is_empty()) {
				continue;
			}

			MethodData method;
			method.name = method_info.name;

			if (method_info.flags & METHOD_FLAG_VIRTUAL) {
				method.is_virtual = true;
			}

			PropertyInfo return_info = method_info.return_val;

			MethodBind *m = method.is_virtual ? nullptr : ClassDB::get_method(class_name, method_info.name);

			method.is_vararg = m && m->is_vararg();

			if (!m && !method.is_virtual) {
				TEST_FAIL_COND(!virtual_method_list.find(method_info),
						"Missing MethodBind for non-virtual method: '", exposed_class.name, ".", method.name, "'.");

				// A virtual method without the virtual flag. This is a special case.

				// The method Object.free is registered as a virtual method, but without the virtual flag.
				// This is because this method is not supposed to be overridden, but called.
				// We assume the return type is void.
				method.return_type.name = r_context.names_cache.void_type;

				// Actually, more methods like this may be added in the future, which could return
				// something different. Let's put this check to notify us if that ever happens.
				String warn_msg = vformat(
						"Notification: New unexpected virtual non-overridable method found. "
						"We only expected Object.free, but found '%s.%s'.",
						exposed_class.name, method.name);
				TEST_FAIL_COND_WARN(
						(exposed_class.name != r_context.names_cache.object_class || String(method.name) != "free"),
						warn_msg.utf8().get_data());

			} else if (return_info.type == Variant::INT && return_info.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
				method.return_type.name = return_info.class_name;
				method.return_type.is_enum = true;
			} else if (return_info.class_name != StringName()) {
				method.return_type.name = return_info.class_name;

				bool bad_reference_hint = !method.is_virtual && return_info.hint != PROPERTY_HINT_RESOURCE_TYPE &&
										  ClassDB::is_parent_class(return_info.class_name, r_context.names_cache.reference_class);
				TEST_COND(bad_reference_hint, "Return type is reference but hint is not '" _STR(PROPERTY_HINT_RESOURCE_TYPE) "'.", " Are you returning a reference type by pointer? Method: '",
						exposed_class.name, ".", method.name, "'.");
			} else if (return_info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
				method.return_type.name = return_info.hint_string;
			} else if (return_info.type == Variant::NIL && return_info.usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
				method.return_type.name = r_context.names_cache.variant_type;
			} else if (return_info.type == Variant::NIL) {
				method.return_type.name = r_context.names_cache.void_type;
			} else {
				// NOTE: We don't care about the size and sign of int and float in these tests
				method.return_type.name = Variant::get_type_name(return_info.type);
			}

			for (int i = 0; i < argc; i++) {
				PropertyInfo arg_info = method_info.arguments[i];

				String orig_arg_name = arg_info.name;

				ArgumentData arg;
				arg.name = orig_arg_name;

				if (arg_info.type == Variant::INT && arg_info.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
					arg.type.name = arg_info.class_name;
					arg.type.is_enum = true;
				} else if (arg_info.class_name != StringName()) {
					arg.type.name = arg_info.class_name;
				} else if (arg_info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
					arg.type.name = arg_info.hint_string;
				} else if (arg_info.type == Variant::NIL) {
					arg.type.name = r_context.names_cache.variant_type;
				} else {
					// NOTE: We don't care about the size and sign of int and float in these tests
					arg.type.name = Variant::get_type_name(arg_info.type);
				}

				if (m && m->has_default_argument(i)) {
					arg.has_defval = true;
					arg.defval = m->get_default_argument(i);
				}

				method.arguments.push_back(arg);
			}

			if (method.is_vararg) {
				ArgumentData vararg;
				vararg.type.name = r_context.names_cache.vararg_stub_type;
				vararg.name = "@varargs@";
				method.arguments.push_back(vararg);
			}

			TEST_COND(exposed_class.find_property_by_name(method.name),
					"Method name conflicts with property: '", String(class_name), ".", String(method.name), "'.");

			// Classes starting with an underscore are ignored unless they're used as a property setter or getter
			if (!method.is_virtual && String(method.name)[0] == '_') {
				for (const List<PropertyData>::Element *F = exposed_class.properties.front(); F; F = F->next()) {
					const PropertyData &prop = F->get();

					if (prop.setter == method.name || prop.getter == method.name) {
						exposed_class.methods.push_back(method);
						break;
					}
				}
			} else {
				exposed_class.methods.push_back(method);
			}
		}

		// Add signals

		const HashMap<StringName, MethodInfo> &signal_map = class_info->signal_map;
		const StringName *k = nullptr;

		while ((k = signal_map.next(k))) {
			SignalData signal;

			const MethodInfo &method_info = signal_map.get(*k);

			signal.name = method_info.name;

			int argc = method_info.arguments.size();

			for (int i = 0; i < argc; i++) {
				PropertyInfo arg_info = method_info.arguments[i];

				String orig_arg_name = arg_info.name;

				ArgumentData arg;
				arg.name = orig_arg_name;

				if (arg_info.type == Variant::INT && arg_info.usage & PROPERTY_USAGE_CLASS_IS_ENUM) {
					arg.type.name = arg_info.class_name;
					arg.type.is_enum = true;
				} else if (arg_info.class_name != StringName()) {
					arg.type.name = arg_info.class_name;
				} else if (arg_info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
					arg.type.name = arg_info.hint_string;
				} else if (arg_info.type == Variant::NIL) {
					arg.type.name = r_context.names_cache.variant_type;
				} else {
					// NOTE: We don't care about the size and sign of int and float in these tests
					arg.type.name = Variant::get_type_name(arg_info.type);
				}

				signal.arguments.push_back(arg);
			}

			bool method_conflict = exposed_class.find_property_by_name(signal.name);

			// TODO:
			// ClassDB allows signal names that conflict with method or property names.
			// However registering a signal with a conflicting name is still considered wrong.
			// Unfortunately there are some existing cases that are yet to be fixed.
			// Until those are fixed we will print a warning instead of failing the test.
			String warn_msg = vformat(
					"Signal name conflicts with %s: '%s.%s.",
					method_conflict ? "method" : "property", class_name, signal.name);
			TEST_FAIL_COND_WARN((method_conflict || exposed_class.find_method_by_name(signal.name)),
					warn_msg.utf8().get_data());

			exposed_class.signals_.push_back(signal);
		}

		// Add enums and constants

		List<String> constants;
		ClassDB::get_integer_constant_list(class_name, &constants, true);

		const HashMap<StringName, List<StringName>> &enum_map = class_info->enum_map;
		k = nullptr;

		while ((k = enum_map.next(k))) {
			EnumData enum_;
			enum_.name = *k;

			const List<StringName> &enum_constants = enum_map.get(*k);
			for (const List<StringName>::Element *E = enum_constants.front(); E; E = E->next()) {
				const StringName &constant_name = E->get();
				int *value = class_info->constant_map.getptr(constant_name);
				TEST_FAIL_COND(!value, "Missing enum constant value: '",
						String(class_name), ".", String(enum_.name), ".", String(constant_name), "'.");
				constants.erase(constant_name);

				ConstantData constant;
				constant.name = constant_name;
				constant.value = *value;

				enum_.constants.push_back(constant);
			}

			exposed_class.enums.push_back(enum_);

			r_context.enum_types.push_back(String(class_name) + "." + String(*k));
		}

		for (const List<String>::Element *E = constants.front(); E; E = E->next()) {
			const String &constant_name = E->get();
			int *value = class_info->constant_map.getptr(StringName(E->get()));
			TEST_FAIL_COND(!value, "Missing enum constant value: '", String(class_name), ".", String(constant_name), "'.");

			ConstantData constant;
			constant.name = constant_name;
			constant.value = *value;

			exposed_class.constants.push_back(constant);
		}

		r_context.exposed_classes.insert(class_name, exposed_class);
		class_list.pop_front();
	}
}

void add_builtin_types(Context &r_context) {
	// NOTE: We don't care about the size and sign of int and float in these tests
	for (int i = 0; i < Variant::VARIANT_MAX; i++) {
		r_context.builtin_types.push_back(Variant::get_type_name(Variant::Type(i)));
	}

	r_context.builtin_types.push_back(_STR(Variant));
	r_context.builtin_types.push_back(r_context.names_cache.vararg_stub_type);
	r_context.builtin_types.push_back("void");
}

void add_global_enums(Context &r_context) {
	int global_constants_count = CoreConstants::get_global_constant_count();

	if (global_constants_count > 0) {
		for (int i = 0; i < global_constants_count; i++) {
			StringName enum_name = CoreConstants::get_global_constant_enum(i);

			if (enum_name != StringName()) {
				ConstantData constant;
				constant.name = CoreConstants::get_global_constant_name(i);
				constant.value = CoreConstants::get_global_constant_value(i);

				EnumData enum_;
				enum_.name = enum_name;
				List<EnumData>::Element *enum_match = r_context.global_enums.find(enum_);
				if (enum_match) {
					enum_match->get().constants.push_back(constant);
				} else {
					enum_.constants.push_back(constant);
					r_context.global_enums.push_back(enum_);
				}
			}
		}

		for (List<EnumData>::Element *E = r_context.global_enums.front(); E; E = E->next()) {
			r_context.enum_types.push_back(E->get().name);
		}
	}

	// HARDCODED
	List<StringName> hardcoded_enums;
	hardcoded_enums.push_back("Vector2.Axis");
	hardcoded_enums.push_back("Vector2i.Axis");
	hardcoded_enums.push_back("Vector3.Axis");
	hardcoded_enums.push_back("Vector3i.Axis");
	for (List<StringName>::Element *E = hardcoded_enums.front(); E; E = E->next()) {
		// These enums are not generated and must be written manually (e.g.: Vector3.Axis)
		// Here, we assume core types do not begin with underscore
		r_context.enum_types.push_back(E->get());
	}
}

TEST_SUITE("[ClassDB]") {
	TEST_CASE("[ClassDB] Add exposed classes, builtin types, and global enums") {
		Context context;

		add_exposed_classes(context);
		add_builtin_types(context);
		add_global_enums(context);

		SUBCASE("[ClassDB] Find exposed class") {
			const ExposedClass *object_class = context.find_exposed_class(context.names_cache.object_class);
			TEST_FAIL_COND(!object_class, "Object class not found.");
			TEST_FAIL_COND(object_class->base != StringName(),
					"Object class derives from another class: '", object_class->base, "'.");

			for (ExposedClasses::Element E = context.exposed_classes.front(); E; E = E.next()) {
				validate_class(context, E.value());
			}
		}
	}
}
} // namespace TestClassDB

#endif // TEST_CLASS_DB_H
