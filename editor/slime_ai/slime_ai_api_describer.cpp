/**************************************************************************/
/*  slime_ai_api_describer.cpp                                            */
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

#include "slime_ai_api_describer.h"

#include "core/object/class_db.h"

namespace SlimeAI {

Dictionary ApiDescriber::describe(const StringName &p_class_name, const StringName &p_member, bool p_method) {
	Dictionary result;
	if (String(p_class_name).is_empty() || String(p_member).is_empty() || String(p_class_name).length() > 128 || String(p_member).length() > 128) {
		result["error"] = "INVALID_ARGUMENT";
		return result;
	}
	if (!ClassDB::class_exists(p_class_name)) {
		result["error"] = "UNSUPPORTED_OPERATION";
		result["recovery"] = "Choose a native class in this build.";
		return result;
	}
	result["class_name"] = String(p_class_name);
	result["member"] = String(p_member);
	result["source"] = "current_native_ClassDB";
	result["kind"] = p_method ? "method" : "property";
	result["enabled_action"] = false;
	if (p_method) {
		List<MethodInfo> methods;
		ClassDB::get_method_list(p_class_name, &methods);
		for (const MethodInfo &method : methods) {
			if (method.name != p_member) {
				continue;
			}
			result["return_type"] = Variant::get_type_name(method.return_val.type);
			Array arguments;
			for (int i = 0; i < MIN(method.arguments.size(), 16); i++) {
				Dictionary argument;
				argument["name"] = method.arguments[i].name;
				argument["type"] = Variant::get_type_name(method.arguments[i].type);
				arguments.push_back(argument);
			}
			result["arguments"] = arguments;
			result["arguments_truncated"] = method.arguments.size() > 16;
			return result;
		}
	} else {
		List<PropertyInfo> properties;
		ClassDB::get_property_list(p_class_name, &properties);
		for (const PropertyInfo &property : properties) {
			if (property.name != p_member) {
				continue;
			}
			result["type"] = Variant::get_type_name(property.type);
			result["usage"] = int(property.usage);
			result["hint"] = int(property.hint);
			const bool supported_class = p_class_name == SNAME("Node2D") || p_class_name == SNAME("Node3D");
			result["enabled_action"] = supported_class && (p_member == SNAME("position") || p_member == SNAME("visible"));
			return result;
		}
	}
	result["error"] = "UNSUPPORTED_OPERATION";
	result["recovery"] = "Select a member present in native ClassDB metadata.";
	return result;
}

} // namespace SlimeAI
