/*************************************************************************/
/*  gdscript_analyzer.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdscript_analyzer.h"

#include "core/class_db.h"
#include "core/hash_map.h"
#include "core/io/resource_loader.h"
#include "core/script_language.h"

Error GDScriptAnalyzer::resolve_inheritance(GDScriptParser::ClassNode *p_class, bool p_recursive) {
	GDScriptParser::DataType result;

	if (p_class->base_type.is_set()) {
		// Already resolved
		return OK;
	}

	if (!p_class->extends_used) {
		result.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
		result.kind = GDScriptParser::DataType::NATIVE;
		result.native_type = "Reference";
	} else {
		result.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;

		GDScriptParser::DataType base;

		if (!p_class->extends_path.empty()) {
			base.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
			base.kind = GDScriptParser::DataType::CLASS;
			// TODO: Don't load the script here to avoid the issue with cycles.
			base.script_type = ResourceLoader::load(p_class->extends_path);
			if (base.script_type.is_null() || !base.script_type->is_valid()) {
				parser->push_error(vformat(R"(Could not load the parent script at "%s".)", p_class->extends_path));
			}
			// TODO: Get this from cache singleton.
			base.gdscript_type = nullptr;
		} else {
			const StringName &name = p_class->extends[0];
			base.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;

			if (ScriptServer::is_global_class(name)) {
				base.kind = GDScriptParser::DataType::CLASS;
				// TODO: Get this from cache singleton.
				base.gdscript_type = nullptr;
				// TODO: Try singletons (create main unified source for those).
			} else if (p_class->members_indices.has(name)) {
				GDScriptParser::ClassNode::Member member = p_class->get_member(name);

				if (member.type == member.CLASS) {
					base.kind = GDScriptParser::DataType::CLASS;
					base.gdscript_type = member.m_class;
				} else if (member.type == member.CONSTANT) {
					// FIXME: This could also be a native type or preloaded GDScript.
					base.kind = GDScriptParser::DataType::CLASS;
					base.gdscript_type = nullptr;
				}
			} else {
				if (ClassDB::class_exists(name)) {
					base.kind = GDScriptParser::DataType::NATIVE;
					base.native_type = name;
				}
			}
		}

		// TODO: Extends with attributes (A.B.C).
		result = base;
	}

	if (!result.is_set()) {
		// TODO: More specific error messages.
		parser->push_error(vformat(R"(Could not resolve inheritance for class "%s".)", p_class->identifier == nullptr ? "<main>" : p_class->identifier->name), p_class);
		return ERR_PARSE_ERROR;
	}

	p_class->set_datatype(result);

	if (p_recursive) {
		for (int i = 0; i < p_class->members.size(); i++) {
			if (p_class->members[i].type == GDScriptParser::ClassNode::Member::CLASS) {
				resolve_inheritance(p_class->members[i].m_class, true);
			}
		}
	}

	return OK;
}

Error GDScriptAnalyzer::resolve_inheritance() {
	return resolve_inheritance(parser->head);
}

// TODO: Move this to a central location (maybe core?).
static HashMap<StringName, StringName> underscore_map;
static const char *underscore_classes[] = {
	"ClassDB",
	"Directory",
	"Engine",
	"File",
	"Geometry",
	"GodotSharp",
	"JSON",
	"Marshalls",
	"Mutex",
	"OS",
	"ResourceLoader",
	"ResourceSaver",
	"Semaphore",
	"Thread",
	"VisualScriptEditor",
	nullptr,
};
static StringName get_real_class_name(const StringName &p_source) {
	if (underscore_map.empty()) {
		const char **class_name = underscore_classes;
		while (*class_name != nullptr) {
			underscore_map[*class_name] = String("_") + *class_name;
			class_name++;
		}
	}
	if (underscore_map.has(p_source)) {
		return underscore_map[p_source];
	}
	return p_source;
}

GDScriptParser::DataType GDScriptAnalyzer::resolve_datatype(const GDScriptParser::TypeNode *p_type) {
	GDScriptParser::DataType result;

	if (p_type == nullptr) {
		return result;
	}

	result.type_source = result.ANNOTATED_EXPLICIT;
	if (p_type->type_base == nullptr) {
		// void.
		result.kind = GDScriptParser::DataType::BUILTIN;
		result.builtin_type = Variant::NIL;
		return result;
	}

	StringName first = p_type->type_base->name;

	if (GDScriptParser::get_builtin_type(first) != Variant::NIL) {
		// Built-in types.
		// FIXME: I'm probably using this wrong here (well, I'm not really using it). Specifier *includes* the base the type.
		if (p_type->type_specifier != nullptr) {
			parser->push_error(R"(Built-in types don't contain subtypes.)", p_type->type_specifier);
			return GDScriptParser::DataType();
		}
		result.kind = GDScriptParser::DataType::BUILTIN;
		result.builtin_type = GDScriptParser::get_builtin_type(first);
	} else if (ClassDB::class_exists(get_real_class_name(first))) {
		// Native engine classes.
		if (p_type->type_specifier != nullptr) {
			parser->push_error(R"(Engine classes don't contain subtypes.)", p_type->type_specifier);
			return GDScriptParser::DataType();
		}
		result.kind = GDScriptParser::DataType::NATIVE;
		result.native_type = first;
	} else if (ScriptServer::is_global_class(first)) {
		// Global class_named classes.
		// TODO: Global classes and singletons.
		parser->push_error("GDScript analyzer: global class type not implemented.", p_type);
		ERR_FAIL_V_MSG(GDScriptParser::DataType(), "GDScript analyzer: global class type not implemented.");
	} else {
		// Classes in current scope.
		GDScriptParser::ClassNode *script_class = parser->current_class;
		bool found = false;
		while (!found && script_class != nullptr) {
			if (script_class->members_indices.has(first)) {
				GDScriptParser::ClassNode::Member member = script_class->members[script_class->members_indices[first]];
				switch (member.type) {
					case GDScriptParser::ClassNode::Member::CLASS:
						result.kind = GDScriptParser::DataType::CLASS;
						result.gdscript_type = member.m_class;
						found = true;
						break;
					default:
						// TODO: Get constants as types, disallow others explicitly.
						parser->push_error(vformat(R"("%s" is a %s but does not contain a type.)", first, member.get_type_name()), p_type);
						return GDScriptParser::DataType();
				}
			}
			script_class = script_class->outer;
		}

		parser->push_error(vformat(R"("%s" is not a valid type.)", first), p_type);
		return GDScriptParser::DataType();
	}

	// TODO: Allow subtypes.
	if (p_type->type_specifier != nullptr) {
		parser->push_error(R"(Subtypes are not yet supported.)", p_type->type_specifier);
		return GDScriptParser::DataType();
	}

	return result;
}

Error GDScriptAnalyzer::resolve_datatypes(GDScriptParser::ClassNode *p_class) {
	GDScriptParser::ClassNode *previous_class = parser->current_class;
	parser->current_class = p_class;

	for (int i = 0; i < p_class->members.size(); i++) {
		GDScriptParser::ClassNode::Member member = p_class->members[i];

		switch (member.type) {
			case GDScriptParser::ClassNode::Member::VARIABLE: {
				GDScriptParser::DataType datatype = resolve_datatype(member.variable->datatype_specifier);
				if (datatype.is_set()) {
					member.variable->set_datatype(datatype);
					if (member.variable->export_info.hint == PROPERTY_HINT_TYPE_STRING) {
						// @export annotation.
						switch (datatype.kind) {
							case GDScriptParser::DataType::BUILTIN:
								member.variable->export_info.hint_string = Variant::get_type_name(datatype.builtin_type);
								break;
							case GDScriptParser::DataType::NATIVE:
								if (ClassDB::is_parent_class(get_real_class_name(datatype.native_type), "Resource")) {
									member.variable->export_info.hint_string = get_real_class_name(datatype.native_type);
								} else {
									parser->push_error(R"(Export type can only be built-in or a resource.)", member.variable);
								}
								break;
							default:
								// TODO: Allow custom user resources.
								parser->push_error(R"(Export type can only be built-in or a resource.)", member.variable);
								break;
						}
					}
				}
				break;
			}
			default:
				// TODO
				break;
		}
	}
	parser->current_class = previous_class;

	return parser->errors.size() > 0 ? ERR_PARSE_ERROR : OK;
}

Error GDScriptAnalyzer::analyze() {
	parser->errors.clear();
	Error err = resolve_inheritance(parser->head);
	if (err) {
		return err;
	}
	return resolve_datatypes(parser->head);
}

GDScriptAnalyzer::GDScriptAnalyzer(GDScriptParser *p_parser) {
	parser = p_parser;
}
