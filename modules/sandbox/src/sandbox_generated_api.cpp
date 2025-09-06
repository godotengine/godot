/**************************************************************************/
/*  sandbox_generated_api.cpp                                             */
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

#include "sandbox.h"

#include "core/config/engine.h"
#include "core/core_bind.h"
#include "core/object/class_db.h"
#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/variant/variant_utility.h"
#include "guest_datatypes.h"
#include "sandbox_project_settings.h"
#include <functional>
#include <unordered_map>
namespace riscv {
extern std::unordered_map<std::string, std::function<uint64_t()>> allowed_globals;
}

static constexpr bool VERBOSE = false;
static String *current_generated_api = nullptr;

static const char *cpp_compatible_variant_type(int type) {
	switch (type) {
		case Variant::NIL:
			return "Variant";
		case Variant::BOOL:
			return "bool";
		case Variant::INT:
			return "int64_t";
		case Variant::FLOAT:
			return "double";
		case Variant::STRING:
		case Variant::NODE_PATH:
		case Variant::STRING_NAME:
			return "String";

		case Variant::VECTOR2:
			return "Vector2";
		case Variant::VECTOR2I:
			return "Vector2i";
		case Variant::RECT2:
			return "Rect2";
		case Variant::RECT2I:
			return "Rect2i";
		case Variant::VECTOR3:
			return "Vector3";
		case Variant::VECTOR3I:
			return "Vector3i";
		case Variant::VECTOR4:
			return "Vector4";
		case Variant::VECTOR4I:
			return "Vector4i";
		case Variant::COLOR:
			return "Color";

		case Variant::PLANE:
			return "Plane";
		case Variant::QUATERNION:
			return "Quaternion";
		case Variant::AABB:
			return "Variant";
		case Variant::TRANSFORM2D:
			return "Transform2D";
		case Variant::TRANSFORM3D:
			return "Transform3D";
		case Variant::BASIS:
			return "Basis";
		case Variant::PROJECTION:
			return "Variant";
		case Variant::RID:
			return "::RID";

		case Variant::OBJECT:
			return "Object";
		case Variant::DICTIONARY:
			return "Dictionary";
		case Variant::ARRAY:
			return "Array";
		case Variant::CALLABLE:
			return "Callable";
		case Variant::SIGNAL:
			return "Variant";

		case Variant::PACKED_BYTE_ARRAY:
			return "PackedArray<uint8_t>";
		case Variant::PACKED_INT32_ARRAY:
			return "PackedArray<int32_t>";
		case Variant::PACKED_INT64_ARRAY:
			return "PackedArray<int64_t>";
		case Variant::PACKED_FLOAT32_ARRAY:
			return "PackedArray<float>";
		case Variant::PACKED_FLOAT64_ARRAY:
			return "PackedArray<double>";
		case Variant::PACKED_STRING_ARRAY:
			return "PackedArray<std::string>";
		case Variant::PACKED_VECTOR2_ARRAY:
			return "PackedArray<Vector2>";
		case Variant::PACKED_VECTOR3_ARRAY:
			return "PackedArray<Vector3>";
		case Variant::PACKED_COLOR_ARRAY:
			return "PackedArray<Color>";
		default:
			throw std::runtime_error("Unknown variant type.");
	}
}

// TODO: Invalidate the API if any classes change or new classes are registered.
String Sandbox::generate_api(String language, String header, bool use_argument_names) {
	if (current_generated_api == nullptr) {
		Sandbox::generate_runtime_cpp_api(use_argument_names);
	}
	if (current_generated_api == nullptr) {
		return String();
	}
	return header + *current_generated_api;
}

static String emit_class(CoreBind::Special::ClassDB *class_db, const HashSet<String> &cpp_keywords, const HashSet<String> &singletons, const String &class_name, bool use_argument_names) {
	// Generate a simple API for each class using METHOD() and PROPERTY() macros to a string.
	if constexpr (VERBOSE) {
		print_line("* Currently generating: " + class_name);
	}
	String parent_name = class_db->get_parent_class(class_name);

	String api = "struct " + class_name + " : public " + parent_name + " {\n";

	// Here we need to use an existing constructor, to inherit from the correct class.
	// eg. if it's a Node, we need to inherit from Node.
	// If it's a Node2D, we need to inherit from Node2D. etc.
	api += "    using " + parent_name + "::" + parent_name + ";\n";

	// We just need the names of the properties and methods.
	Array properties = class_db->class_get_property_list(class_name, true);
	for (int j = 0; j < properties.size(); j++) {
		Dictionary property = properties[j];
		String property_name = property["name"];
		const int type = int(property["type"]);
		// Properties that are likely just groups or categories.
		if (type == Variant::NIL) {
			continue;
		}
		// Skip properties with spaces in the name. Yes, this is a thing.
		if (property_name.is_empty() || property_name.contains(" ") || property_name.contains("/") || property_name.contains("-")) {
			continue;
		}
		// Yes, this is a thing too.
		if (property_name == class_name) {
			continue;
		}
		// If matching C++ keywords, capitalize the first letter.
		if (cpp_keywords.has(property_name.to_lower())) {
			property_name = property_name.capitalize();
		}

		String property_type = cpp_compatible_variant_type(type);
		if (property_type == "Variant") {
			api += String("    PROPERTY(") + property_name + ", Variant);\n";
		} else {
			api += String("    PROPERTY(") + property_name + ", " + property_type + ");\n";
		}
	}

	Array methods = class_db->class_get_method_list(class_name, true);
	for (int j = 0; j < methods.size(); j++) {
		Dictionary method = methods[j];
		String method_name = method["name"];
		Dictionary return_value = method["return"];
		const int type = int(return_value["type"]);
		// Skip methods that are empty, and methods with '/' and '-' in the name.
		if (method_name.is_empty() || method_name.contains("/") || method_name.contains("-")) {
			continue;
		}
		// If matching C++ keywords, capitalize the first letter.
		if (cpp_keywords.has(method_name.to_lower())) {
			method_name = method_name.capitalize();
		}

		//if (flags & METHOD_FLAG_STATIC) ...
		// Variant::NIL is a special case, as it's can be a void or Variant return type.
		const int return_flags = int(return_value["usage"]);
		const bool is_void = type == 0 && (return_flags & PROPERTY_USAGE_NIL_IS_VARIANT) == 0;

		if (use_argument_names) {
			// Example: { "name": "play_stream",
			// "args": [{
			//    "name": "stream", "class_name": &"AudioStream", "type": 24, "hint": 17, "hint_string": "AudioStream", "usage": 6
			// }, {
			//    "name": "from_offset", "class_name": &"", "type": 3, "hint": 0, "hint_string": "", "usage": 6
			// }, {
			//    "name": "volume_db", "class_name": &"", "type": 3, "hint": 0, "hint_string": "", "usage": 6
			// }, {
			//    "name": "pitch_scale", "class_name": &"", "type": 3, "hint": 0, "hint_string": "", "usage": 6
			// }, {
			//    "name": "playback_type", "class_name": &"AudioServer.PlaybackType", "type": 2, "hint": 0, "hint_string": "", "usage": 65542
			// }, {
			//    "name": "bus", "class_name": &"", "type": 21, "hint": 0, "hint_string": "", "usage": 6
			// }],
			//  "default_args": [0, 0, 1, 0, &"Master"],
			//  "flags": 1,
			//  "id": 11865,
			//  "return": { "name": "", "class_name": &"", "type": 2, "hint": 0, "hint_string": "", "usage": 6 } }
			api += "    ";
			// Return value:
			if (type != 0) {
				api += String(cpp_compatible_variant_type(type)) + " ";
			} else if (is_void) {
				api += "void ";
			} else {
				api += "Variant ";
			}
			// Method name:
			api += method_name + "(";
			// Arguments:
			const Array arguments = method["args"];
			Vector<String> argument_names;
			// Default arguments:
			const Array default_args = method["default_args"];
			const int default_args_size = default_args.size();
			const int first_default_arg = arguments.size() - default_args_size;

			for (int k = 0; k < arguments.size(); k++) {
				const Dictionary &argument = arguments[k];

				String arg_name = argument["name"];
				if (cpp_keywords.has(arg_name.to_lower())) {
					arg_name += "_";
				}
				const int arg_type = int(argument["type"]);

				api += String(cpp_compatible_variant_type(arg_type)) + " " + arg_name;

				// Default arguments.
				if (k >= first_default_arg) {
					api += " = ";
					const Variant &default_arg = default_args[k - first_default_arg];
					switch (arg_type) {
						case Variant::BOOL:
							api += bool(default_arg) ? "true" : "false";
							break;
						case Variant::INT:
							api += itos(int64_t(default_arg));
							break;
						case Variant::FLOAT:
							api += String::num(double(default_arg));
							break;
						case Variant::STRING:
							api += "\"" + (String)default_arg + "\"";
							break;
						case Variant::VECTOR2: {
							const Vector2 vec = default_arg.operator Vector2();
							api += "Vector2(" + String::num(vec.x) + ", " + String::num(vec.y) + ")";
							break;
						}
						case Variant::VECTOR2I: {
							const Vector2i vec = default_arg.operator Vector2i();
							api += "Vector2i(" + itos(vec.x) + ", " + itos(vec.y) + ")";
							break;
						}
						case Variant::RECT2: {
							const Rect2 rect = default_arg.operator Rect2();
							api += "Rect2(" + String::num(rect.position.x) + ", " + String::num(rect.position.y) + ", " + String::num(rect.size.x) + ", " + String::num(rect.size.y) + ")";
							break;
						}
						case Variant::RECT2I: {
							const Rect2i rect = default_arg.operator Rect2i();
							api += "Rect2i(" + itos(rect.position.x) + ", " + itos(rect.position.y) + ", " + itos(rect.size.x) + ", " + itos(rect.size.y) + ")";
							break;
						}
						case Variant::VECTOR3: {
							const Vector3 vec = default_arg.operator Vector3();
							api += "Vector3(" + String::num(vec.x) + ", " + String::num(vec.y) + ", " + String::num(vec.z) + ")";
							break;
						}
						case Variant::VECTOR3I: {
							const Vector3i vec = default_arg.operator Vector3i();
							api += "Vector3i(" + itos(vec.x) + ", " + itos(vec.y) + ", " + itos(vec.z) + ")";
							break;
						}
						case Variant::VECTOR4: {
							const Vector4 vec = default_arg.operator Vector4();
							api += "Vector4(" + String::num(vec.x) + ", " + String::num(vec.y) + ", " + String::num(vec.z) + ", " + String::num(vec.w) + ")";
							break;
						}
						case Variant::VECTOR4I: {
							const Vector4i vec = default_arg.operator Vector4i();
							api += "Vector4i(" + itos(vec.x) + ", " + itos(vec.y) + ", " + itos(vec.z) + ", " + itos(vec.w) + ")";
							break;
						}
						case Variant::COLOR: {
							const Color color = default_arg.operator Color();
							api += "Color(" + String::num(color.r) + ", " + String::num(color.g) + ", " + String::num(color.b) + ", " + String::num(color.a) + ")";
							break;
						}
						case Variant::PLANE: {
							const Plane plane = default_arg.operator Plane();
							api += "Plane(" + String::num(plane.normal.x) + ", " + String::num(plane.normal.y) + ", " + String::num(plane.normal.z) + ", " + String::num(plane.d) + ")";
							break;
						}
						case Variant::QUATERNION: {
							const Quaternion quat = default_arg.operator Quaternion();
							api += "Quaternion(" + String::num(quat.x) + ", " + String::num(quat.y) + ", " + String::num(quat.z) + ", " + String::num(quat.w) + ")";
							break;
						}
						case Variant::OBJECT: {
							api += "{0}";
							break;
						}
						default:
							api += "{}";
							break;
					}
				}

				if (k != arguments.size() - 1) {
					api += ", ";
				}

				argument_names.push_back(arg_name);
			}
			// TODO: Append const if the method is const.
			// Sadly, it breaks the call operator, so hold off on this for now.
			api += ") {\n";
			// Method body: return operator() (\"" + method_name + "\"", " + argument_list + ");\n";
			if (is_void) {
				// Void return type.
				api += "      voidcall(\"" + method_name + "\"";
			} else {
				// Typed return type.
				api += "      return operator() (\"" + method_name + "\"";
			}
			if (!arguments.is_empty()) {
				api += ", ";
			}
			for (int k = 0; k < argument_names.size(); k++) {
				const String &arg_name = argument_names[k];
				api += arg_name;
				if (k != arguments.size() - 1) {
					api += ", ";
				}
			}
			api += ");\n";
			api += "    }\n";
			// End of method.
			continue;
		}

		// Typed return type.
		if (is_void) {
			api += String("    METHOD(void, ") + method_name + ");\n";
		} else {
			api += String("    METHOD(") + cpp_compatible_variant_type(type) + ", " + method_name + ");\n";
		}
	}

	// Add integer constants.
	PackedStringArray integer_constants = class_db->class_get_integer_constant_list(class_name, true);
	for (int j = 0; j < integer_constants.size(); j++) {
		String constant_name = integer_constants[j];
		// If matching C++ keywords, capitalize the first letter.
		if (cpp_keywords.has(constant_name.to_lower())) {
			constant_name = constant_name.capitalize();
		}
		api += String("    static constexpr int64_t ") + constant_name + " = " + itos(class_db->class_get_integer_constant(class_name, constant_name)) + ";\n";
	}

	// Add singleton getter if the class is a singleton.
	if (singletons.has(class_name)) {
		api += "    static " + class_name + " get_singleton() { return " + class_name + "(Object(\"" + class_name + "\").address()); }\n";
	}

	api += "};\n";
	return api;
}

void Sandbox::generate_runtime_cpp_api(bool use_argument_names) {
	// 1. Get all classes currently registered with the engine.
	// 2. Get all methods and properties for each class.
	// 3. Generate a simple API for each class using METHOD() and PROPERTY() macros to a string.
	// 4. Print the generated API to the console.
	if constexpr (VERBOSE) {
		print_line("* Generating C++ run-time API");
	}
	if (current_generated_api != nullptr) {
		delete current_generated_api;
	}
	current_generated_api = new String("#pragma once\n\n#include <api.hpp>\n#define GENERATED_API 1\n\n");

	HashSet<String> cpp_keywords;
	cpp_keywords.insert("class");
	cpp_keywords.insert("operator");
	cpp_keywords.insert("new");
	cpp_keywords.insert("delete");
	cpp_keywords.insert("this");
	cpp_keywords.insert("virtual");
	cpp_keywords.insert("override");
	cpp_keywords.insert("final");
	cpp_keywords.insert("public");
	cpp_keywords.insert("protected");
	cpp_keywords.insert("private");
	cpp_keywords.insert("static");
	cpp_keywords.insert("const");
	cpp_keywords.insert("constexpr");
	cpp_keywords.insert("inline");
	cpp_keywords.insert("friend");
	cpp_keywords.insert("template");
	cpp_keywords.insert("typename");
	cpp_keywords.insert("typedef");
	cpp_keywords.insert("using");
	cpp_keywords.insert("namespace");
	cpp_keywords.insert("return");
	cpp_keywords.insert("if");
	cpp_keywords.insert("else");
	cpp_keywords.insert("for");
	cpp_keywords.insert("while");
	cpp_keywords.insert("do");
	cpp_keywords.insert("switch");
	cpp_keywords.insert("case");
	cpp_keywords.insert("default");
	cpp_keywords.insert("break");
	cpp_keywords.insert("continue");
	cpp_keywords.insert("goto");
	cpp_keywords.insert("try");
	cpp_keywords.insert("catch");
	cpp_keywords.insert("throw");
	cpp_keywords.insert("static_assert");
	cpp_keywords.insert("sizeof");
	cpp_keywords.insert("alignof");
	cpp_keywords.insert("decltype");
	cpp_keywords.insert("noexcept");
	cpp_keywords.insert("nullptr");
	cpp_keywords.insert("true");
	cpp_keywords.insert("false");
	cpp_keywords.insert("and");
	cpp_keywords.insert("or");
	cpp_keywords.insert("not");
	cpp_keywords.insert("xor");
	cpp_keywords.insert("bitand");
	cpp_keywords.insert("bitor");
	cpp_keywords.insert("compl");
	cpp_keywords.insert("and_eq");
	cpp_keywords.insert("or_eq");
	cpp_keywords.insert("xor_eq");
	cpp_keywords.insert("not_eq");
	cpp_keywords.insert("bool");
	cpp_keywords.insert("char");
	cpp_keywords.insert("short");
	cpp_keywords.insert("int");
	cpp_keywords.insert("long");
	cpp_keywords.insert("float");
	cpp_keywords.insert("double");

	// 1. Get all classes currently registered with the engine.
	// ClassDB in CoreBind::Special namespace doesn't have a get_singleton, so we need to access it differently
	// Let's try getting the global ClassDB instance through Engine
	CoreBind::Special::ClassDB *class_db = Object::cast_to<CoreBind::Special::ClassDB>(Engine::get_singleton()->get_singleton_object("ClassDB"));
	if (!class_db) {
		ERR_PRINT("Failed to get ClassDB singleton");
		return;
	}
	PackedStringArray class_list = class_db->get_class_list();
	Array classes;
	for (int i = 0; i < class_list.size(); i++) {
		classes.push_back(class_list[i]);
	}

	HashSet<String> emitted_classes;
	HashMap<String, TypedArray<String>> waiting_classes;
	Array skipped_class_words = SandboxProjectSettings::generated_api_skipped_classes();
	int total_skipped_classes = 0;

	// 2. Insert all pre-existing classes into the emitted_classes set.
	emitted_classes.insert("Object");
	emitted_classes.insert("Node");
	emitted_classes.insert("CanvasItem");
	emitted_classes.insert("Node2D");
	emitted_classes.insert("Node3D");
	// Also skip some classes we simply don't want to expose.
	emitted_classes.insert("ClassDB");

	// Finally, add singleton getters to certain classes.
	HashSet<String> singletons;
	Vector<String> singleton_list = CoreBind::Engine::get_singleton()->get_singleton_list();
	for (int i = 0; i < singleton_list.size(); i++) {
		singletons.insert(singleton_list[i]);
	}

	// 3. Get all methods and properties for each class.
	for (int i = 0; i < classes.size(); i++) {
		String class_name = classes[i];

		// 4. Emit classes
		// Check if the class is already emitted.
		if (emitted_classes.has(class_name)) {
			continue;
		}

		bool is_skipped = false;
		for (int j = 0; j < skipped_class_words.size(); j++) {
			if (class_name.contains(skipped_class_words[j])) {
				if constexpr (VERBOSE) {
					print_line("* Skipping class: " + class_name);
				}
				total_skipped_classes++;
				is_skipped = true;
				break;
			}
		}
		if (is_skipped) {
			continue;
		}

		// Check if the parent class has been emitted, and if not, add it to the waiting_classes map.
		String parent_name = class_db->get_parent_class(class_name);
		if (!emitted_classes.find(parent_name)) { // Not emitted yet.
			TypedArray<String> *waiting = waiting_classes.getptr(parent_name);
			if (waiting == nullptr) {
				waiting_classes.insert(parent_name, TypedArray<String>());
				waiting = waiting_classes.getptr(parent_name);
			}
			waiting->push_back(class_name);
			continue;
		}
		// Emit the class.
		*current_generated_api += emit_class(class_db, cpp_keywords, singletons, class_name, use_argument_names);
		emitted_classes.insert(class_name);
	}

	// 5. Emit waiting classes.
	while (!waiting_classes.is_empty()) {
		const int initial_waiting_classes = waiting_classes.size();

		for (auto it = waiting_classes.begin(); it != waiting_classes.end(); ++it) {
			const String &parent_name = it->key;
			if (emitted_classes.has(parent_name)) {
				const TypedArray<String> &waiting = it->value;
				for (int i = 0; i < waiting.size(); i++) {
					String class_name = waiting[i];
					*current_generated_api += emit_class(class_db, cpp_keywords, singletons, class_name, use_argument_names);
					emitted_classes.insert(class_name);
				}
				waiting_classes.erase(parent_name);
				break;
			}
		}

		const int remaining_waiting_classes = waiting_classes.size();
		if (remaining_waiting_classes == initial_waiting_classes) {
			if (skipped_class_words.is_empty()) {
				// We have a circular dependency.
				// This is a bug in the engine, and should be reported.
				// We can't continue, so we'll just break out of the loop.
				ERR_PRINT("Circular dependency detected in class inheritance");
				for (auto it = waiting_classes.begin(); it != waiting_classes.end(); ++it) {
					const String &parent_name = it->key;
					const TypedArray<String> &waiting = it->value;
					for (int i = 0; i < waiting.size(); i++) {
						ERR_PRINT("* Waiting class " + String(waiting[i]) + " with parent " + parent_name);
					}
				}
			} else {
				// When we have skipped classes, we can't emit them, so we'll just skip them.
				total_skipped_classes += remaining_waiting_classes;
				WARN_PRINT("Skipped classes left in class inheritance: " + itos(remaining_waiting_classes) + ", total skipped classes: " + itos(total_skipped_classes));
			}
			break;
		}
	}

	if constexpr (VERBOSE) {
		print_line("* Finished generating " + itos(classes.size()) + " classes");
	}
}
