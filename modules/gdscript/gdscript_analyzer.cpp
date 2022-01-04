/*************************************************************************/
/*  gdscript_analyzer.cpp                                                */
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

#include "gdscript_analyzer.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/object/class_db.h"
#include "core/object/script_language.h"
#include "core/templates/hash_map.h"
#include "gdscript.h"
#include "gdscript_utility_functions.h"

static MethodInfo info_from_utility_func(const StringName &p_function) {
	ERR_FAIL_COND_V(!Variant::has_utility_function(p_function), MethodInfo());

	MethodInfo info(p_function);

	if (Variant::has_utility_function_return_value(p_function)) {
		info.return_val.type = Variant::get_utility_function_return_type(p_function);
		if (info.return_val.type == Variant::NIL) {
			info.return_val.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
		}
	}

	if (Variant::is_utility_function_vararg(p_function)) {
		info.flags |= METHOD_FLAG_VARARG;
	} else {
		for (int i = 0; i < Variant::get_utility_function_argument_count(p_function); i++) {
			PropertyInfo pi;
#ifdef DEBUG_METHODS_ENABLED
			pi.name = Variant::get_utility_function_argument_name(p_function, i);
#else
			pi.name = "arg" + itos(i + 1);
#endif
			pi.type = Variant::get_utility_function_argument_type(p_function, i);
			if (pi.type == Variant::NIL) {
				pi.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
			}
			info.arguments.push_back(pi);
		}
	}

	return info;
}

static GDScriptParser::DataType make_callable_type(const MethodInfo &p_info) {
	GDScriptParser::DataType type;
	type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	type.kind = GDScriptParser::DataType::BUILTIN;
	type.builtin_type = Variant::CALLABLE;
	type.is_constant = true;
	type.method_info = p_info;
	return type;
}

static GDScriptParser::DataType make_signal_type(const MethodInfo &p_info) {
	GDScriptParser::DataType type;
	type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	type.kind = GDScriptParser::DataType::BUILTIN;
	type.builtin_type = Variant::SIGNAL;
	type.is_constant = true;
	type.method_info = p_info;
	return type;
}

static GDScriptParser::DataType make_native_meta_type(const StringName &p_class_name) {
	GDScriptParser::DataType type;
	type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	type.kind = GDScriptParser::DataType::NATIVE;
	type.builtin_type = Variant::OBJECT;
	type.is_constant = true;
	type.native_type = p_class_name;
	type.is_meta_type = true;
	return type;
}

static GDScriptParser::DataType make_native_enum_type(const StringName &p_native_class, const StringName &p_enum_name) {
	GDScriptParser::DataType type;
	type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	type.kind = GDScriptParser::DataType::ENUM;
	type.builtin_type = Variant::OBJECT;
	type.is_constant = true;
	type.is_meta_type = true;

	List<StringName> enum_values;
	ClassDB::get_enum_constants(p_native_class, p_enum_name, &enum_values);

	for (const StringName &E : enum_values) {
		type.enum_values[E] = ClassDB::get_integer_constant(p_native_class, E);
	}

	return type;
}

static GDScriptParser::DataType make_builtin_meta_type(Variant::Type p_type) {
	GDScriptParser::DataType type;
	type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	type.kind = GDScriptParser::DataType::BUILTIN;
	type.builtin_type = p_type;
	type.is_constant = true;
	type.is_meta_type = true;
	return type;
}

bool GDScriptAnalyzer::has_member_name_conflict_in_script_class(const StringName &p_member_name, const GDScriptParser::ClassNode *p_class) {
	if (p_class->members_indices.has(p_member_name)) {
		int index = p_class->members_indices[p_member_name];
		const GDScriptParser::ClassNode::Member *member = &p_class->members[index];

		if (member->type == GDScriptParser::ClassNode::Member::VARIABLE ||
				member->type == GDScriptParser::ClassNode::Member::CONSTANT ||
				member->type == GDScriptParser::ClassNode::Member::ENUM ||
				member->type == GDScriptParser::ClassNode::Member::ENUM_VALUE ||
				member->type == GDScriptParser::ClassNode::Member::CLASS ||
				member->type == GDScriptParser::ClassNode::Member::SIGNAL) {
			return true;
		}
	}

	return false;
}

bool GDScriptAnalyzer::has_member_name_conflict_in_native_type(const StringName &p_member_name, const StringName &p_native_type_string) {
	if (ClassDB::has_signal(p_native_type_string, p_member_name)) {
		return true;
	}
	if (ClassDB::has_property(p_native_type_string, p_member_name)) {
		return true;
	}
	if (ClassDB::has_integer_constant(p_native_type_string, p_member_name)) {
		return true;
	}

	return false;
}

Error GDScriptAnalyzer::check_native_member_name_conflict(const StringName &p_member_name, const GDScriptParser::Node *p_member_node, const StringName &p_native_type_string) {
	if (has_member_name_conflict_in_native_type(p_member_name, p_native_type_string)) {
		push_error(vformat(R"(Member "%s" redefined (original in native class '%s'))", p_member_name, p_native_type_string), p_member_node);
		return ERR_PARSE_ERROR;
	}

	if (class_exists(p_member_name)) {
		push_error(vformat(R"(The member "%s" shadows a native class.)", p_member_name), p_member_node);
		return ERR_PARSE_ERROR;
	}

	if (GDScriptParser::get_builtin_type(p_member_name) != Variant::VARIANT_MAX) {
		push_error(vformat(R"(The member "%s" cannot have the same name as a builtin type.)", p_member_name), p_member_node);
		return ERR_PARSE_ERROR;
	}

	return OK;
}

Error GDScriptAnalyzer::check_class_member_name_conflict(const GDScriptParser::ClassNode *p_class_node, const StringName &p_member_name, const GDScriptParser::Node *p_member_node) {
	const GDScriptParser::DataType *current_data_type = &p_class_node->base_type;
	while (current_data_type && current_data_type->kind == GDScriptParser::DataType::Kind::CLASS) {
		GDScriptParser::ClassNode *current_class_node = current_data_type->class_type;
		if (has_member_name_conflict_in_script_class(p_member_name, current_class_node)) {
			push_error(vformat(R"(The member "%s" already exists in a parent class.)", p_member_name),
					p_member_node);
			return ERR_PARSE_ERROR;
		}
		current_data_type = &current_class_node->base_type;
	}

	if (current_data_type && current_data_type->kind == GDScriptParser::DataType::Kind::NATIVE) {
		if (current_data_type->native_type != StringName("")) {
			return check_native_member_name_conflict(
					p_member_name,
					p_member_node,
					current_data_type->native_type);
		}
	}

	return OK;
}

Error GDScriptAnalyzer::resolve_inheritance(GDScriptParser::ClassNode *p_class, bool p_recursive) {
	if (p_class->base_type.is_set()) {
		// Already resolved
		return OK;
	}

	if (p_class == parser->head) {
		if (p_class->identifier) {
			p_class->fqcn = p_class->identifier->name;
		} else {
			p_class->fqcn = parser->script_path;
		}
	} else {
		p_class->fqcn = p_class->outer->fqcn + "::" + String(p_class->identifier->name);
	}

	if (p_class->identifier) {
		StringName class_name = p_class->identifier->name;
		if (class_exists(class_name)) {
			push_error(vformat(R"(Class "%s" hides a native class.)", class_name), p_class->identifier);
		} else if (ScriptServer::is_global_class(class_name) && (ScriptServer::get_global_class_path(class_name) != parser->script_path || p_class != parser->head)) {
			push_error(vformat(R"(Class "%s" hides a global script class.)", class_name), p_class->identifier);
		} else if (ProjectSettings::get_singleton()->has_autoload(class_name) && ProjectSettings::get_singleton()->get_autoload(class_name).is_singleton) {
			push_error(vformat(R"(Class "%s" hides an autoload singleton.)", class_name), p_class->identifier);
		}
	}

	GDScriptParser::DataType result;

	// Set datatype for class.
	GDScriptParser::DataType class_type;
	class_type.is_constant = true;
	class_type.is_meta_type = true;
	class_type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	class_type.kind = GDScriptParser::DataType::CLASS;
	class_type.class_type = p_class;
	class_type.script_path = parser->script_path;
	class_type.builtin_type = Variant::OBJECT;
	p_class->set_datatype(class_type);

	if (!p_class->extends_used) {
		result.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
		result.kind = GDScriptParser::DataType::NATIVE;
		result.native_type = "RefCounted";
	} else {
		result.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;

		GDScriptParser::DataType base;

		int extends_index = 0;

		if (!p_class->extends_path.is_empty()) {
			if (p_class->extends_path.is_relative_path()) {
				p_class->extends_path = class_type.script_path.get_base_dir().plus_file(p_class->extends_path).simplify_path();
			}
			Ref<GDScriptParserRef> parser = get_parser_for(p_class->extends_path);
			if (parser.is_null()) {
				push_error(vformat(R"(Could not resolve super class path "%s".)", p_class->extends_path), p_class);
				return ERR_PARSE_ERROR;
			}

			Error err = parser->raise_status(GDScriptParserRef::INTERFACE_SOLVED);
			if (err != OK) {
				push_error(vformat(R"(Could not resolve super class inheritance from "%s".)", p_class->extends_path), p_class);
				return err;
			}

			base = parser->get_parser()->head->get_datatype();
		} else {
			if (p_class->extends.is_empty()) {
				return ERR_PARSE_ERROR;
			}
			const StringName &name = p_class->extends[extends_index++];
			base.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;

			if (ScriptServer::is_global_class(name)) {
				String base_path = ScriptServer::get_global_class_path(name);

				if (base_path == parser->script_path) {
					base = parser->head->get_datatype();
				} else {
					Ref<GDScriptParserRef> parser = get_parser_for(base_path);
					if (parser.is_null()) {
						push_error(vformat(R"(Could not resolve super class "%s".)", name), p_class);
						return ERR_PARSE_ERROR;
					}

					Error err = parser->raise_status(GDScriptParserRef::INTERFACE_SOLVED);
					if (err != OK) {
						push_error(vformat(R"(Could not resolve super class inheritance from "%s".)", name), p_class);
						return err;
					}
					base = parser->get_parser()->head->get_datatype();
				}
			} else if (ProjectSettings::get_singleton()->has_autoload(name) && ProjectSettings::get_singleton()->get_autoload(name).is_singleton) {
				const ProjectSettings::AutoloadInfo &info = ProjectSettings::get_singleton()->get_autoload(name);
				if (info.path.get_extension().to_lower() != ".gd") {
					push_error(vformat(R"(Singleton %s is not a GDScript.)", info.name), p_class);
					return ERR_PARSE_ERROR;
				}

				Ref<GDScriptParserRef> parser = get_parser_for(info.path);
				if (parser.is_null()) {
					push_error(vformat(R"(Could not parse singleton from "%s".)", info.path), p_class);
					return ERR_PARSE_ERROR;
				}

				Error err = parser->raise_status(GDScriptParserRef::INTERFACE_SOLVED);
				if (err != OK) {
					push_error(vformat(R"(Could not resolve super class inheritance from "%s".)", name), p_class);
					return err;
				}
			} else if (class_exists(name) && ClassDB::can_instantiate(name)) {
				base.kind = GDScriptParser::DataType::NATIVE;
				base.native_type = name;
			} else {
				// Look for other classes in script.
				GDScriptParser::ClassNode *look_class = p_class;
				bool found = false;
				while (look_class != nullptr) {
					if (look_class->identifier && look_class->identifier->name == name) {
						if (!look_class->get_datatype().is_set()) {
							Error err = resolve_inheritance(look_class, false);
							if (err) {
								return err;
							}
						}
						base = look_class->get_datatype();
						found = true;
						break;
					}
					if (look_class->members_indices.has(name) && look_class->get_member(name).type == GDScriptParser::ClassNode::Member::CLASS) {
						GDScriptParser::ClassNode::Member member = look_class->get_member(name);
						if (!member.m_class->get_datatype().is_set()) {
							Error err = resolve_inheritance(member.m_class, false);
							if (err) {
								return err;
							}
						}
						base = member.m_class->get_datatype();
						found = true;
						break;
					}
					look_class = look_class->outer;
				}

				if (!found) {
					push_error(vformat(R"(Could not find base class "%s".)", name), p_class);
					return ERR_PARSE_ERROR;
				}
			}
		}

		for (int index = extends_index; index < p_class->extends.size(); index++) {
			if (base.kind != GDScriptParser::DataType::CLASS) {
				push_error(R"(Super type "%s" is not a GDScript. Cannot get nested types.)", p_class);
				return ERR_PARSE_ERROR;
			}

			// TODO: Extends could use identifier nodes. That way errors can be pointed out properly and it can be used here.
			GDScriptParser::IdentifierNode *id = parser->alloc_node<GDScriptParser::IdentifierNode>();
			id->name = p_class->extends[index];

			reduce_identifier_from_base(id, &base);

			GDScriptParser::DataType id_type = id->get_datatype();
			if (!id_type.is_set()) {
				push_error(vformat(R"(Could not find type "%s" under base "%s".)", id->name, base.to_string()), p_class);
			}

			base = id_type;
		}

		result = base;
	}

	if (!result.is_set()) {
		// TODO: More specific error messages.
		push_error(vformat(R"(Could not resolve inheritance for class "%s".)", p_class->identifier == nullptr ? "<main>" : p_class->identifier->name), p_class);
		return ERR_PARSE_ERROR;
	}

	// Check for cyclic inheritance.
	const GDScriptParser::ClassNode *base_class = result.class_type;
	while (base_class) {
		if (base_class->fqcn == p_class->fqcn) {
			push_error("Cyclic inheritance.", p_class);
			return ERR_PARSE_ERROR;
		}
		base_class = base_class->base_type.class_type;
	}

	p_class->base_type = result;
	class_type.native_type = result.native_type;
	p_class->set_datatype(class_type);

	if (p_recursive) {
		for (int i = 0; i < p_class->members.size(); i++) {
			if (p_class->members[i].type == GDScriptParser::ClassNode::Member::CLASS) {
				Error err = resolve_inheritance(p_class->members[i].m_class, true);
				if (err) {
					return err;
				}
			}
		}
	}

	return OK;
}

GDScriptParser::DataType GDScriptAnalyzer::resolve_datatype(GDScriptParser::TypeNode *p_type) {
	GDScriptParser::DataType result;

	if (p_type == nullptr) {
		result.kind = GDScriptParser::DataType::VARIANT;
		return result;
	}

	result.type_source = result.ANNOTATED_EXPLICIT;
	result.builtin_type = Variant::OBJECT;

	if (p_type->type_chain.is_empty()) {
		// void.
		result.kind = GDScriptParser::DataType::BUILTIN;
		result.builtin_type = Variant::NIL;
		p_type->set_datatype(result);
		return result;
	}

	StringName first = p_type->type_chain[0]->name;

	if (first == "Variant") {
		result.kind = GDScriptParser::DataType::VARIANT;
		if (p_type->type_chain.size() > 1) {
			push_error(R"("Variant" type don't contain nested types.)", p_type->type_chain[1]);
			return GDScriptParser::DataType();
		}
		return result;
	}

	if (first == "Object") {
		result.kind = GDScriptParser::DataType::NATIVE;
		result.native_type = "Object";
		if (p_type->type_chain.size() > 1) {
			push_error(R"("Object" type don't contain nested types.)", p_type->type_chain[1]);
			return GDScriptParser::DataType();
		}
		return result;
	}

	if (GDScriptParser::get_builtin_type(first) < Variant::VARIANT_MAX) {
		// Built-in types.
		if (p_type->type_chain.size() > 1) {
			push_error(R"(Built-in types don't contain nested types.)", p_type->type_chain[1]);
			return GDScriptParser::DataType();
		}
		result.kind = GDScriptParser::DataType::BUILTIN;
		result.builtin_type = GDScriptParser::get_builtin_type(first);

		if (result.builtin_type == Variant::ARRAY) {
			GDScriptParser::DataType container_type = resolve_datatype(p_type->container_type);

			if (container_type.kind != GDScriptParser::DataType::VARIANT) {
				container_type.is_meta_type = false;
				container_type.is_constant = false;
				result.set_container_element_type(container_type);
			}
		}
	} else if (class_exists(first)) {
		// Native engine classes.
		result.kind = GDScriptParser::DataType::NATIVE;
		result.native_type = first;
	} else if (ScriptServer::is_global_class(first)) {
		if (parser->script_path == ScriptServer::get_global_class_path(first)) {
			result = parser->head->get_datatype();
		} else {
			Ref<GDScriptParserRef> ref = get_parser_for(ScriptServer::get_global_class_path(first));
			if (!ref.is_valid() || ref->raise_status(GDScriptParserRef::INTERFACE_SOLVED) != OK) {
				push_error(vformat(R"(Could not parse global class "%s" from "%s".)", first, ScriptServer::get_global_class_path(first)), p_type);
				return GDScriptParser::DataType();
			}
			result = ref->get_parser()->head->get_datatype();
		}
	} else if (ProjectSettings::get_singleton()->has_autoload(first) && ProjectSettings::get_singleton()->get_autoload(first).is_singleton) {
		const ProjectSettings::AutoloadInfo &autoload = ProjectSettings::get_singleton()->get_autoload(first);
		Ref<GDScriptParserRef> ref = get_parser_for(autoload.path);
		if (ref->raise_status(GDScriptParserRef::INTERFACE_SOLVED) != OK) {
			push_error(vformat(R"(Could not parse singleton "%s" from "%s".)", first, autoload.path), p_type);
			return GDScriptParser::DataType();
		}
		result = ref->get_parser()->head->get_datatype();
	} else if (ClassDB::has_enum(parser->current_class->base_type.native_type, first)) {
		// Native enum in current class.
		result = make_native_enum_type(parser->current_class->base_type.native_type, first);
	} else {
		// Classes in current scope.
		GDScriptParser::ClassNode *script_class = parser->current_class;
		bool found = false;
		while (!found && script_class != nullptr) {
			if (script_class->identifier && script_class->identifier->name == first) {
				result = script_class->get_datatype();
				found = true;
				break;
			}
			if (script_class->members_indices.has(first)) {
				GDScriptParser::ClassNode::Member member = script_class->members[script_class->members_indices[first]];
				switch (member.type) {
					case GDScriptParser::ClassNode::Member::CLASS:
						result = member.m_class->get_datatype();
						found = true;
						break;
					case GDScriptParser::ClassNode::Member::ENUM:
						result = member.m_enum->get_datatype();
						found = true;
						break;
					case GDScriptParser::ClassNode::Member::CONSTANT:
						if (member.constant->get_datatype().is_meta_type) {
							result = member.constant->get_datatype();
							result.is_meta_type = false;
							found = true;
							break;
						} else if (Ref<Script>(member.constant->initializer->reduced_value).is_valid()) {
							Ref<GDScript> gdscript = member.constant->initializer->reduced_value;
							if (gdscript.is_valid()) {
								Ref<GDScriptParserRef> ref = get_parser_for(gdscript->get_path());
								if (ref->raise_status(GDScriptParserRef::INTERFACE_SOLVED) != OK) {
									push_error(vformat(R"(Could not parse script from "%s".)", gdscript->get_path()), p_type);
									return GDScriptParser::DataType();
								}
								result = ref->get_parser()->head->get_datatype();
								result.is_meta_type = false;
							} else {
								Ref<GDScript> script = member.constant->initializer->reduced_value;
								result.kind = GDScriptParser::DataType::SCRIPT;
								result.builtin_type = Variant::OBJECT;
								result.script_type = script;
								result.script_path = script->get_path();
								result.native_type = script->get_instance_base_type();
							}
							break;
						}
						[[fallthrough]];
					default:
						push_error(vformat(R"("%s" is a %s but does not contain a type.)", first, member.get_type_name()), p_type);
						return GDScriptParser::DataType();
				}
			}
			script_class = script_class->outer;
		}
	}
	if (!result.is_set()) {
		push_error(vformat(R"("%s" was not found in the current scope.)", first), p_type);
		result.kind = GDScriptParser::DataType::VARIANT; // Leave Variant anyway so future type check don't use an unresolved type.
		return result;
	}

	if (p_type->type_chain.size() > 1) {
		if (result.kind == GDScriptParser::DataType::CLASS) {
			for (int i = 1; i < p_type->type_chain.size(); i++) {
				GDScriptParser::DataType base = result;
				reduce_identifier_from_base(p_type->type_chain[i], &base);
				result = p_type->type_chain[i]->get_datatype();
				if (!result.is_set()) {
					push_error(vformat(R"(Could not find type "%s" under base "%s".)", p_type->type_chain[i]->name, base.to_string()), p_type->type_chain[1]);
					result.kind = GDScriptParser::DataType::VARIANT; // Leave Variant anyway so future type check don't use an unresolved type.
					return result;
				} else if (!result.is_meta_type) {
					push_error(vformat(R"(Member "%s" under base "%s" is not a valid type.)", p_type->type_chain[i]->name, base.to_string()), p_type->type_chain[1]);
					result.kind = GDScriptParser::DataType::VARIANT; // Leave Variant anyway so future type check don't use an unresolved type.
					return result;
				}
			}
		} else if (result.kind == GDScriptParser::DataType::NATIVE) {
			// Only enums allowed for native.
			if (ClassDB::has_enum(result.native_type, p_type->type_chain[1]->name)) {
				if (p_type->type_chain.size() > 2) {
					push_error(R"(Enums cannot contain nested types.)", p_type->type_chain[2]);
				} else {
					result = make_native_enum_type(result.native_type, p_type->type_chain[1]->name);
				}
			}
		} else {
			push_error(vformat(R"(Could not find nested type "%s" under base "%s".)", p_type->type_chain[1]->name, result.to_string()), p_type->type_chain[1]);
			result.kind = GDScriptParser::DataType::VARIANT; // Leave Variant anyway so future type check don't use an unresolved type.
			return result;
		}
	}

	if (result.builtin_type != Variant::ARRAY && p_type->container_type != nullptr) {
		push_error("Only arrays can specify the collection element type.", p_type);
	}

	p_type->set_datatype(result);
	return result;
}

void GDScriptAnalyzer::resolve_class_interface(GDScriptParser::ClassNode *p_class) {
	if (p_class->resolved_interface) {
		return;
	}
	p_class->resolved_interface = true;

	GDScriptParser::ClassNode *previous_class = parser->current_class;
	parser->current_class = p_class;

	for (int i = 0; i < p_class->members.size(); i++) {
		GDScriptParser::ClassNode::Member member = p_class->members[i];

		switch (member.type) {
			case GDScriptParser::ClassNode::Member::VARIABLE: {
				check_class_member_name_conflict(p_class, member.variable->identifier->name, member.variable);

				GDScriptParser::DataType datatype;
				datatype.kind = GDScriptParser::DataType::VARIANT;
				datatype.type_source = GDScriptParser::DataType::UNDETECTED;

				GDScriptParser::DataType specified_type;
				if (member.variable->datatype_specifier != nullptr) {
					specified_type = resolve_datatype(member.variable->datatype_specifier);
					specified_type.is_meta_type = false;
				}

				if (member.variable->initializer != nullptr) {
					member.variable->set_datatype(datatype); // Allow recursive usage.
					reduce_expression(member.variable->initializer);
					if ((member.variable->infer_datatype || (member.variable->datatype_specifier != nullptr && specified_type.has_container_element_type())) && member.variable->initializer->type == GDScriptParser::Node::ARRAY) {
						// Typed array.
						GDScriptParser::ArrayNode *array = static_cast<GDScriptParser::ArrayNode *>(member.variable->initializer);
						// Can only infer typed array if it has elements.
						if ((member.variable->infer_datatype && array->elements.size() > 0) || member.variable->datatype_specifier != nullptr) {
							update_array_literal_element_type(specified_type, array);
						}
					}
					datatype = member.variable->initializer->get_datatype();
					if (datatype.type_source != GDScriptParser::DataType::UNDETECTED) {
						datatype.type_source = GDScriptParser::DataType::INFERRED;
					}
				}

				if (member.variable->datatype_specifier != nullptr) {
					datatype = specified_type;

					if (member.variable->initializer != nullptr) {
						if (!is_type_compatible(datatype, member.variable->initializer->get_datatype(), true)) {
							// Try reverse test since it can be a masked subtype.
							if (!is_type_compatible(member.variable->initializer->get_datatype(), datatype, true)) {
								push_error(vformat(R"(Value of type "%s" cannot be assigned to a variable of type "%s".)", member.variable->initializer->get_datatype().to_string(), datatype.to_string()), member.variable->initializer);
							} else {
								// TODO: Add warning.
								mark_node_unsafe(member.variable->initializer);
								member.variable->use_conversion_assign = true;
							}
						} else if (datatype.builtin_type == Variant::INT && member.variable->initializer->get_datatype().builtin_type == Variant::FLOAT) {
#ifdef DEBUG_ENABLED
							parser->push_warning(member.variable->initializer, GDScriptWarning::NARROWING_CONVERSION);
#endif
						}
						if (member.variable->initializer->get_datatype().is_variant()) {
							// TODO: Warn unsafe assign.
							mark_node_unsafe(member.variable->initializer);
							member.variable->use_conversion_assign = true;
						}
					}
				} else if (member.variable->infer_datatype) {
					if (member.variable->initializer == nullptr) {
						push_error(vformat(R"(Cannot infer the type of "%s" variable because there's no default value.)", member.variable->identifier->name), member.variable->identifier);
					} else if (!datatype.is_set() || datatype.has_no_type()) {
						push_error(vformat(R"(Cannot infer the type of "%s" variable because the initial value doesn't have a set type.)", member.variable->identifier->name), member.variable->initializer);
					} else if (datatype.is_variant()) {
						push_error(vformat(R"(Cannot infer the type of "%s" variable because the initial value is Variant. Use explicit "Variant" type if this is intended.)", member.variable->identifier->name), member.variable->initializer);
					} else if (datatype.builtin_type == Variant::NIL) {
						push_error(vformat(R"(Cannot infer the type of "%s" variable because the initial value is "null".)", member.variable->identifier->name), member.variable->initializer);
					}
					datatype.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
				}

				datatype.is_constant = false;
				member.variable->set_datatype(datatype);

				// Apply annotations.
				for (GDScriptParser::AnnotationNode *&E : member.variable->annotations) {
					E->apply(parser, member.variable);
				}
			} break;
			case GDScriptParser::ClassNode::Member::CONSTANT: {
				check_class_member_name_conflict(p_class, member.constant->identifier->name, member.constant);

				reduce_expression(member.constant->initializer);

				GDScriptParser::DataType specified_type;

				if (member.constant->datatype_specifier != nullptr) {
					specified_type = resolve_datatype(member.constant->datatype_specifier);
					specified_type.is_meta_type = false;
				}

				GDScriptParser::DataType datatype;
				if (member.constant->initializer) {
					datatype = member.constant->initializer->get_datatype();
					if (member.constant->initializer->type == GDScriptParser::Node::ARRAY) {
						GDScriptParser::ArrayNode *array = static_cast<GDScriptParser::ArrayNode *>(member.constant->initializer);
						const_fold_array(array);

						// Can only infer typed array if it has elements.
						if (array->elements.size() > 0 || (member.constant->datatype_specifier != nullptr && specified_type.has_container_element_type())) {
							update_array_literal_element_type(specified_type, array);
						}
					} else if (member.constant->initializer->type == GDScriptParser::Node::DICTIONARY) {
						const_fold_dictionary(static_cast<GDScriptParser::DictionaryNode *>(member.constant->initializer));
					}

					if (!member.constant->initializer->is_constant) {
						push_error(R"(Initializer for a constant must be a constant expression.)", member.constant->initializer);
					}

					if (member.constant->datatype_specifier != nullptr) {
						datatype = specified_type;

						if (!is_type_compatible(datatype, member.constant->initializer->get_datatype(), true)) {
							push_error(vformat(R"(Value of type "%s" cannot be initialized to constant of type "%s".)", member.constant->initializer->get_datatype().to_string(), datatype.to_string()), member.constant->initializer);
						} else if (datatype.builtin_type == Variant::INT && member.constant->initializer->get_datatype().builtin_type == Variant::FLOAT) {
#ifdef DEBUG_ENABLED
							parser->push_warning(member.constant->initializer, GDScriptWarning::NARROWING_CONVERSION);
#endif
						}
					}
				}
				datatype.is_constant = true;

				member.constant->set_datatype(datatype);

				// Apply annotations.
				for (GDScriptParser::AnnotationNode *&E : member.constant->annotations) {
					E->apply(parser, member.constant);
				}
			} break;
			case GDScriptParser::ClassNode::Member::SIGNAL: {
				check_class_member_name_conflict(p_class, member.signal->identifier->name, member.signal);

				for (int j = 0; j < member.signal->parameters.size(); j++) {
					GDScriptParser::DataType signal_type = resolve_datatype(member.signal->parameters[j]->datatype_specifier);
					signal_type.is_meta_type = false;
					member.signal->parameters[j]->set_datatype(signal_type);
				}
				// TODO: Make MethodInfo from signal.
				GDScriptParser::DataType signal_type;
				signal_type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
				signal_type.kind = GDScriptParser::DataType::BUILTIN;
				signal_type.builtin_type = Variant::SIGNAL;

				member.signal->set_datatype(signal_type);

				// Apply annotations.
				for (GDScriptParser::AnnotationNode *&E : member.signal->annotations) {
					E->apply(parser, member.signal);
				}
			} break;
			case GDScriptParser::ClassNode::Member::ENUM: {
				check_class_member_name_conflict(p_class, member.m_enum->identifier->name, member.m_enum);

				GDScriptParser::DataType enum_type;
				enum_type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
				enum_type.kind = GDScriptParser::DataType::ENUM;
				enum_type.builtin_type = Variant::DICTIONARY;
				enum_type.enum_type = member.m_enum->identifier->name;
				enum_type.native_type = p_class->fqcn + "." + member.m_enum->identifier->name;
				enum_type.is_meta_type = true;
				enum_type.is_constant = true;

				// Enums can't be nested, so we can safely override this.
				current_enum = member.m_enum;

				for (int j = 0; j < member.m_enum->values.size(); j++) {
					GDScriptParser::EnumNode::Value &element = member.m_enum->values.write[j];

					if (element.custom_value) {
						reduce_expression(element.custom_value);
						if (!element.custom_value->is_constant) {
							push_error(R"(Enum values must be constant.)", element.custom_value);
						} else if (element.custom_value->reduced_value.get_type() != Variant::INT) {
							push_error(R"(Enum values must be integers.)", element.custom_value);
						} else {
							element.value = element.custom_value->reduced_value;
							element.resolved = true;
						}
					} else {
						if (element.index > 0) {
							element.value = element.parent_enum->values[element.index - 1].value + 1;
						} else {
							element.value = 0;
						}
						element.resolved = true;
					}

					enum_type.enum_values[element.identifier->name] = element.value;
				}

				current_enum = nullptr;

				member.m_enum->set_datatype(enum_type);

				// Apply annotations.
				for (GDScriptParser::AnnotationNode *&E : member.m_enum->annotations) {
					E->apply(parser, member.m_enum);
				}
			} break;
			case GDScriptParser::ClassNode::Member::FUNCTION:
				resolve_function_signature(member.function);
				break;
			case GDScriptParser::ClassNode::Member::ENUM_VALUE: {
				if (member.enum_value.custom_value) {
					check_class_member_name_conflict(p_class, member.enum_value.identifier->name, member.enum_value.custom_value);

					current_enum = member.enum_value.parent_enum;
					reduce_expression(member.enum_value.custom_value);
					current_enum = nullptr;

					if (!member.enum_value.custom_value->is_constant) {
						push_error(R"(Enum values must be constant.)", member.enum_value.custom_value);
					} else if (member.enum_value.custom_value->reduced_value.get_type() != Variant::INT) {
						push_error(R"(Enum values must be integers.)", member.enum_value.custom_value);
					} else {
						member.enum_value.value = member.enum_value.custom_value->reduced_value;
						member.enum_value.resolved = true;
					}
				} else {
					check_class_member_name_conflict(p_class, member.enum_value.identifier->name, member.enum_value.parent_enum);

					if (member.enum_value.index > 0) {
						member.enum_value.value = member.enum_value.parent_enum->values[member.enum_value.index - 1].value + 1;
					} else {
						member.enum_value.value = 0;
					}
					member.enum_value.resolved = true;
				}
				// Also update the original references.
				member.enum_value.parent_enum->values.write[member.enum_value.index] = member.enum_value;
				p_class->members.write[i].enum_value = member.enum_value;
			} break;
			case GDScriptParser::ClassNode::Member::CLASS:
				check_class_member_name_conflict(p_class, member.m_class->identifier->name, member.m_class);
				break;
			case GDScriptParser::ClassNode::Member::UNDEFINED:
				ERR_PRINT("Trying to resolve undefined member.");
				break;
		}
	}

	// Recurse nested classes.
	for (int i = 0; i < p_class->members.size(); i++) {
		GDScriptParser::ClassNode::Member member = p_class->members[i];
		if (member.type != GDScriptParser::ClassNode::Member::CLASS) {
			continue;
		}

		resolve_class_interface(member.m_class);
	}

	parser->current_class = previous_class;
}

void GDScriptAnalyzer::resolve_class_body(GDScriptParser::ClassNode *p_class) {
	if (p_class->resolved_body) {
		return;
	}
	p_class->resolved_body = true;

	GDScriptParser::ClassNode *previous_class = parser->current_class;
	parser->current_class = p_class;

	// Do functions and properties now.
	for (int i = 0; i < p_class->members.size(); i++) {
		GDScriptParser::ClassNode::Member member = p_class->members[i];
		if (member.type == GDScriptParser::ClassNode::Member::FUNCTION) {
			// Apply annotations.
			for (GDScriptParser::AnnotationNode *&E : member.function->annotations) {
				E->apply(parser, member.function);
			}

#ifdef DEBUG_ENABLED
			Set<uint32_t> previously_ignored = parser->ignored_warning_codes;
			for (uint32_t ignored_warning : member.function->ignored_warnings) {
				parser->ignored_warning_codes.insert(ignored_warning);
			}
#endif // DEBUG_ENABLED

			resolve_function_body(member.function);

#ifdef DEBUG_ENABLED
			parser->ignored_warning_codes = previously_ignored;
#endif // DEBUG_ENABLED
		} else if (member.type == GDScriptParser::ClassNode::Member::VARIABLE && member.variable->property != GDScriptParser::VariableNode::PROP_NONE) {
			if (member.variable->property == GDScriptParser::VariableNode::PROP_INLINE) {
				if (member.variable->getter != nullptr) {
					member.variable->getter->set_datatype(member.variable->datatype);

					resolve_function_body(member.variable->getter);
				}
				if (member.variable->setter != nullptr) {
					resolve_function_signature(member.variable->setter);

					if (member.variable->setter->parameters.size() > 0) {
						member.variable->setter->parameters[0]->datatype_specifier = member.variable->datatype_specifier;
						member.variable->setter->parameters[0]->set_datatype(member.get_datatype());
					}

					resolve_function_body(member.variable->setter);
				}
			}
		}
	}

	parser->current_class = previous_class;

	// Recurse nested classes.
	for (int i = 0; i < p_class->members.size(); i++) {
		GDScriptParser::ClassNode::Member member = p_class->members[i];
		if (member.type != GDScriptParser::ClassNode::Member::CLASS) {
			continue;
		}

		resolve_class_body(member.m_class);
	}

	// Check unused variables and datatypes of property getters and setters.
	for (int i = 0; i < p_class->members.size(); i++) {
		GDScriptParser::ClassNode::Member member = p_class->members[i];
		if (member.type == GDScriptParser::ClassNode::Member::VARIABLE) {
#ifdef DEBUG_ENABLED
			Set<uint32_t> previously_ignored = parser->ignored_warning_codes;
			for (uint32_t ignored_warning : member.function->ignored_warnings) {
				parser->ignored_warning_codes.insert(ignored_warning);
			}
			if (member.variable->usages == 0 && String(member.variable->identifier->name).begins_with("_")) {
				parser->push_warning(member.variable->identifier, GDScriptWarning::UNUSED_PRIVATE_CLASS_VARIABLE, member.variable->identifier->name);
			}
#endif

			if (member.variable->property == GDScriptParser::VariableNode::PROP_SETGET) {
				GDScriptParser::FunctionNode *getter_function = nullptr;
				GDScriptParser::FunctionNode *setter_function = nullptr;

				bool has_valid_getter = false;
				bool has_valid_setter = false;

				if (member.variable->getter_pointer != nullptr) {
					if (p_class->has_function(member.variable->getter_pointer->name)) {
						getter_function = p_class->get_member(member.variable->getter_pointer->name).function;
					}

					if (getter_function == nullptr) {
						push_error(vformat(R"(Getter "%s" not found.)", member.variable->getter_pointer->name), member.variable);

					} else if (getter_function->parameters.size() != 0 || getter_function->datatype.has_no_type()) {
						push_error(vformat(R"(Function "%s" cannot be used as getter because of its signature.)", getter_function->identifier->name), member.variable);

					} else if (!is_type_compatible(member.variable->datatype, getter_function->datatype, true)) {
						push_error(vformat(R"(Function with return type "%s" cannot be used as getter for a property of type "%s".)", getter_function->datatype.to_string(), member.variable->datatype.to_string()), member.variable);

					} else {
						has_valid_getter = true;

#ifdef DEBUG_ENABLED
						if (member.variable->datatype.builtin_type == Variant::INT && getter_function->datatype.builtin_type == Variant::FLOAT) {
							parser->push_warning(member.variable, GDScriptWarning::NARROWING_CONVERSION);
						}
#endif
					}
				}

				if (member.variable->setter_pointer != nullptr) {
					if (p_class->has_function(member.variable->setter_pointer->name)) {
						setter_function = p_class->get_member(member.variable->setter_pointer->name).function;
					}

					if (setter_function == nullptr) {
						push_error(vformat(R"(Setter "%s" not found.)", member.variable->setter_pointer->name), member.variable);

					} else if (setter_function->parameters.size() != 1) {
						push_error(vformat(R"(Function "%s" cannot be used as setter because of its signature.)", setter_function->identifier->name), member.variable);

					} else if (!is_type_compatible(member.variable->datatype, setter_function->parameters[0]->datatype, true)) {
						push_error(vformat(R"(Function with argument type "%s" cannot be used as setter for a property of type "%s".)", setter_function->parameters[0]->datatype.to_string(), member.variable->datatype.to_string()), member.variable);

					} else {
						has_valid_setter = true;

#ifdef DEBUG_ENABLED
						if (member.variable->datatype.builtin_type == Variant::FLOAT && setter_function->parameters[0]->datatype.builtin_type == Variant::INT) {
							parser->push_warning(member.variable, GDScriptWarning::NARROWING_CONVERSION);
						}
#endif
					}
				}

				if (member.variable->datatype.is_variant() && has_valid_getter && has_valid_setter) {
					if (!is_type_compatible(getter_function->datatype, setter_function->parameters[0]->datatype, true)) {
						push_error(vformat(R"(Getter with type "%s" cannot be used along with setter of type "%s".)", getter_function->datatype.to_string(), setter_function->parameters[0]->datatype.to_string()), member.variable);
					}
				}
#ifdef DEBUG_ENABLED
				parser->ignored_warning_codes = previously_ignored;
#endif // DEBUG_ENABLED
			}
		}
	}
}

void GDScriptAnalyzer::resolve_node(GDScriptParser::Node *p_node) {
	ERR_FAIL_COND_MSG(p_node == nullptr, "Trying to resolve type of a null node.");

	switch (p_node->type) {
		case GDScriptParser::Node::NONE:
			break; // Unreachable.
		case GDScriptParser::Node::CLASS:
			resolve_class_interface(static_cast<GDScriptParser::ClassNode *>(p_node));
			resolve_class_body(static_cast<GDScriptParser::ClassNode *>(p_node));
			break;
		case GDScriptParser::Node::CONSTANT:
			resolve_constant(static_cast<GDScriptParser::ConstantNode *>(p_node));
			break;
		case GDScriptParser::Node::FOR:
			resolve_for(static_cast<GDScriptParser::ForNode *>(p_node));
			break;
		case GDScriptParser::Node::FUNCTION:
			resolve_function_signature(static_cast<GDScriptParser::FunctionNode *>(p_node));
			resolve_function_body(static_cast<GDScriptParser::FunctionNode *>(p_node));
			break;
		case GDScriptParser::Node::IF:
			resolve_if(static_cast<GDScriptParser::IfNode *>(p_node));
			break;
		case GDScriptParser::Node::SUITE:
			resolve_suite(static_cast<GDScriptParser::SuiteNode *>(p_node));
			break;
		case GDScriptParser::Node::VARIABLE:
			resolve_variable(static_cast<GDScriptParser::VariableNode *>(p_node));
			break;
		case GDScriptParser::Node::WHILE:
			resolve_while(static_cast<GDScriptParser::WhileNode *>(p_node));
			break;
		case GDScriptParser::Node::ANNOTATION:
			resolve_annotation(static_cast<GDScriptParser::AnnotationNode *>(p_node));
			break;
		case GDScriptParser::Node::ASSERT:
			resolve_assert(static_cast<GDScriptParser::AssertNode *>(p_node));
			break;
		case GDScriptParser::Node::MATCH:
			resolve_match(static_cast<GDScriptParser::MatchNode *>(p_node));
			break;
		case GDScriptParser::Node::MATCH_BRANCH:
			resolve_match_branch(static_cast<GDScriptParser::MatchBranchNode *>(p_node), nullptr);
			break;
		case GDScriptParser::Node::PARAMETER:
			resolve_parameter(static_cast<GDScriptParser::ParameterNode *>(p_node));
			break;
		case GDScriptParser::Node::PATTERN:
			resolve_match_pattern(static_cast<GDScriptParser::PatternNode *>(p_node), nullptr);
			break;
		case GDScriptParser::Node::RETURN:
			resolve_return(static_cast<GDScriptParser::ReturnNode *>(p_node));
			break;
		case GDScriptParser::Node::TYPE:
			resolve_datatype(static_cast<GDScriptParser::TypeNode *>(p_node));
			break;
		// Resolving expression is the same as reducing them.
		case GDScriptParser::Node::ARRAY:
		case GDScriptParser::Node::ASSIGNMENT:
		case GDScriptParser::Node::AWAIT:
		case GDScriptParser::Node::BINARY_OPERATOR:
		case GDScriptParser::Node::CALL:
		case GDScriptParser::Node::CAST:
		case GDScriptParser::Node::DICTIONARY:
		case GDScriptParser::Node::GET_NODE:
		case GDScriptParser::Node::IDENTIFIER:
		case GDScriptParser::Node::LAMBDA:
		case GDScriptParser::Node::LITERAL:
		case GDScriptParser::Node::PRELOAD:
		case GDScriptParser::Node::SELF:
		case GDScriptParser::Node::SUBSCRIPT:
		case GDScriptParser::Node::TERNARY_OPERATOR:
		case GDScriptParser::Node::UNARY_OPERATOR:
			reduce_expression(static_cast<GDScriptParser::ExpressionNode *>(p_node), true);
			break;
		case GDScriptParser::Node::BREAK:
		case GDScriptParser::Node::BREAKPOINT:
		case GDScriptParser::Node::CONTINUE:
		case GDScriptParser::Node::ENUM:
		case GDScriptParser::Node::PASS:
		case GDScriptParser::Node::SIGNAL:
			// Nothing to do.
			break;
	}
}

void GDScriptAnalyzer::resolve_annotation(GDScriptParser::AnnotationNode *p_annotation) {
	// TODO: Add second validation function for annotations, so they can use checked types.
}

void GDScriptAnalyzer::resolve_function_signature(GDScriptParser::FunctionNode *p_function) {
	if (p_function->resolved_signature) {
		return;
	}
	p_function->resolved_signature = true;

	GDScriptParser::FunctionNode *previous_function = parser->current_function;
	parser->current_function = p_function;

	for (int i = 0; i < p_function->parameters.size(); i++) {
		resolve_parameter(p_function->parameters[i]);
#ifdef DEBUG_ENABLED
		if (p_function->parameters[i]->usages == 0 && !String(p_function->parameters[i]->identifier->name).begins_with("_")) {
			parser->push_warning(p_function->parameters[i]->identifier, GDScriptWarning::UNUSED_PARAMETER, p_function->identifier->name, p_function->parameters[i]->identifier->name);
		}
		is_shadowing(p_function->parameters[i]->identifier, "function parameter");
#endif // DEBUG_ENABLED
#ifdef TOOLS_ENABLED
		if (p_function->parameters[i]->default_value && p_function->parameters[i]->default_value->is_constant) {
			p_function->default_arg_values.push_back(p_function->parameters[i]->default_value->reduced_value);
		}
#endif // TOOLS_ENABLED
	}

	if (p_function->identifier->name == "_init") {
		// Constructor.
		GDScriptParser::DataType return_type = parser->current_class->get_datatype();
		return_type.is_meta_type = false;
		p_function->set_datatype(return_type);
		if (p_function->return_type) {
			GDScriptParser::DataType declared_return = resolve_datatype(p_function->return_type);
			if (declared_return.kind != GDScriptParser::DataType::BUILTIN || declared_return.builtin_type != Variant::NIL) {
				push_error("Constructor cannot have an explicit return type.", p_function->return_type);
			}
		}
	} else {
		GDScriptParser::DataType return_type = resolve_datatype(p_function->return_type);
		p_function->set_datatype(return_type);
	}

	parser->current_function = previous_function;
}

void GDScriptAnalyzer::resolve_function_body(GDScriptParser::FunctionNode *p_function) {
	if (p_function->resolved_body) {
		return;
	}
	p_function->resolved_body = true;

	GDScriptParser::FunctionNode *previous_function = parser->current_function;
	parser->current_function = p_function;

	resolve_suite(p_function->body);

	GDScriptParser::DataType return_type = p_function->body->get_datatype();

	if (p_function->get_datatype().has_no_type() && return_type.is_set()) {
		// Use the suite inferred type if return isn't explicitly set.
		return_type.type_source = GDScriptParser::DataType::INFERRED;
		p_function->set_datatype(p_function->body->get_datatype());
	} else if (p_function->get_datatype().is_hard_type() && (p_function->get_datatype().kind != GDScriptParser::DataType::BUILTIN || p_function->get_datatype().builtin_type != Variant::NIL)) {
		if (!p_function->body->has_return && p_function->identifier->name != GDScriptLanguage::get_singleton()->strings._init) {
			push_error(R"(Not all code paths return a value.)", p_function);
		}
	}

	parser->current_function = previous_function;
}

void GDScriptAnalyzer::decide_suite_type(GDScriptParser::Node *p_suite, GDScriptParser::Node *p_statement) {
	if (p_statement == nullptr) {
		return;
	}
	switch (p_statement->type) {
		case GDScriptParser::Node::IF:
		case GDScriptParser::Node::FOR:
		case GDScriptParser::Node::MATCH:
		case GDScriptParser::Node::PATTERN:
		case GDScriptParser::Node::RETURN:
		case GDScriptParser::Node::WHILE:
			// Use return or nested suite type as this suite type.
			if (p_suite->get_datatype().is_set() && (p_suite->get_datatype() != p_statement->get_datatype())) {
				// Mixed types.
				// TODO: This could use the common supertype instead.
				p_suite->datatype.kind = GDScriptParser::DataType::VARIANT;
				p_suite->datatype.type_source = GDScriptParser::DataType::UNDETECTED;
			} else {
				p_suite->set_datatype(p_statement->get_datatype());
				p_suite->datatype.type_source = GDScriptParser::DataType::INFERRED;
			}
			break;
		default:
			break;
	}
}

void GDScriptAnalyzer::resolve_suite(GDScriptParser::SuiteNode *p_suite) {
	for (int i = 0; i < p_suite->statements.size(); i++) {
		GDScriptParser::Node *stmt = p_suite->statements[i];
		for (GDScriptParser::AnnotationNode *&annotation : stmt->annotations) {
			annotation->apply(parser, stmt);
		}

#ifdef DEBUG_ENABLED
		Set<uint32_t> previously_ignored = parser->ignored_warning_codes;
		for (uint32_t ignored_warning : stmt->ignored_warnings) {
			parser->ignored_warning_codes.insert(ignored_warning);
		}
#endif // DEBUG_ENABLED

		resolve_node(stmt);

#ifdef DEBUG_ENABLED
		parser->ignored_warning_codes = previously_ignored;
#endif // DEBUG_ENABLED

		decide_suite_type(p_suite, stmt);
	}
}

void GDScriptAnalyzer::resolve_if(GDScriptParser::IfNode *p_if) {
	reduce_expression(p_if->condition);

	resolve_suite(p_if->true_block);
	p_if->set_datatype(p_if->true_block->get_datatype());

	if (p_if->false_block != nullptr) {
		resolve_suite(p_if->false_block);
		decide_suite_type(p_if, p_if->false_block);
	}
}

void GDScriptAnalyzer::resolve_for(GDScriptParser::ForNode *p_for) {
	bool list_resolved = false;

	// Optimize constant range() call to not allocate an array.
	// Use int, Vector2, Vector3 instead, which also can be used as range iterators.
	if (p_for->list && p_for->list->type == GDScriptParser::Node::CALL) {
		GDScriptParser::CallNode *call = static_cast<GDScriptParser::CallNode *>(p_for->list);
		GDScriptParser::Node::Type callee_type = call->get_callee_type();
		if (callee_type == GDScriptParser::Node::IDENTIFIER) {
			GDScriptParser::IdentifierNode *callee = static_cast<GDScriptParser::IdentifierNode *>(call->callee);
			if (callee->name == "range") {
				list_resolved = true;
				if (call->arguments.size() < 1) {
					push_error(R"*(Invalid call for "range()" function. Expected at least 1 argument, none given.)*", call->callee);
				} else if (call->arguments.size() > 3) {
					push_error(vformat(R"*(Invalid call for "range()" function. Expected at most 3 arguments, %d given.)*", call->arguments.size()), call->callee);
				} else {
					// Now we can optimize it.
					bool all_is_constant = true;
					Vector<Variant> args;
					args.resize(call->arguments.size());
					for (int i = 0; i < call->arguments.size(); i++) {
						reduce_expression(call->arguments[i]);

						if (!call->arguments[i]->is_constant) {
							all_is_constant = false;
						} else if (all_is_constant) {
							args.write[i] = call->arguments[i]->reduced_value;
						}

						GDScriptParser::DataType arg_type = call->arguments[i]->get_datatype();
						if (!arg_type.is_variant()) {
							if (arg_type.kind != GDScriptParser::DataType::BUILTIN) {
								all_is_constant = false;
								push_error(vformat(R"*(Invalid argument for "range()" call. Argument %d should be int or float but "%s" was given.)*", i + 1, arg_type.to_string()), call->arguments[i]);
							} else if (arg_type.builtin_type != Variant::INT && arg_type.builtin_type != Variant::FLOAT) {
								all_is_constant = false;
								push_error(vformat(R"*(Invalid argument for "range()" call. Argument %d should be int or float but "%s" was given.)*", i + 1, arg_type.to_string()), call->arguments[i]);
							}
						}
					}

					Variant reduced;

					if (all_is_constant) {
						switch (args.size()) {
							case 1:
								reduced = args[0];
								break;
							case 2:
								reduced = Vector2i(args[0], args[1]);
								break;
							case 3:
								reduced = Vector3i(args[0], args[1], args[2]);
								break;
						}
						p_for->list->is_constant = true;
						p_for->list->reduced_value = reduced;
					}
				}

				if (p_for->list->is_constant) {
					p_for->list->set_datatype(type_from_variant(p_for->list->reduced_value, p_for->list));
				} else {
					GDScriptParser::DataType list_type;
					list_type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
					list_type.kind = GDScriptParser::DataType::BUILTIN;
					list_type.builtin_type = Variant::ARRAY;
					p_for->list->set_datatype(list_type);
				}
			}
		}
	}

	GDScriptParser::DataType variable_type;
	if (list_resolved) {
		variable_type.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
		variable_type.kind = GDScriptParser::DataType::BUILTIN;
		variable_type.builtin_type = Variant::INT; // Can this ever be a float or something else?
		p_for->variable->set_datatype(variable_type);
	} else if (p_for->list) {
		resolve_node(p_for->list);
		if (p_for->list->datatype.has_container_element_type()) {
			variable_type = p_for->list->datatype.get_container_element_type();
			variable_type.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
		} else if (p_for->list->datatype.is_typed_container_type()) {
			variable_type = p_for->list->datatype.get_typed_container_type();
			variable_type.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
		} else {
			// Last resort
			// TODO: Must other cases be handled? Must we mark as unsafe?
			variable_type.type_source = GDScriptParser::DataType::UNDETECTED;
			variable_type.kind = GDScriptParser::DataType::VARIANT;
		}
	}
	if (p_for->variable) {
		p_for->variable->set_datatype(variable_type);
	}

	resolve_suite(p_for->loop);
	p_for->set_datatype(p_for->loop->get_datatype());
#ifdef DEBUG_ENABLED
	if (p_for->variable) {
		is_shadowing(p_for->variable, R"("for" iterator variable)");
	}
#endif
}

void GDScriptAnalyzer::resolve_while(GDScriptParser::WhileNode *p_while) {
	resolve_node(p_while->condition);

	resolve_suite(p_while->loop);
	p_while->set_datatype(p_while->loop->get_datatype());
}

void GDScriptAnalyzer::resolve_variable(GDScriptParser::VariableNode *p_variable) {
	GDScriptParser::DataType type;
	type.kind = GDScriptParser::DataType::VARIANT; // By default.

	GDScriptParser::DataType specified_type;
	if (p_variable->datatype_specifier != nullptr) {
		specified_type = resolve_datatype(p_variable->datatype_specifier);
		specified_type.is_meta_type = false;
	}

	if (p_variable->initializer != nullptr) {
		reduce_expression(p_variable->initializer);
		if ((p_variable->infer_datatype || (p_variable->datatype_specifier != nullptr && specified_type.has_container_element_type())) && p_variable->initializer->type == GDScriptParser::Node::ARRAY) {
			// Typed array.
			GDScriptParser::ArrayNode *array = static_cast<GDScriptParser::ArrayNode *>(p_variable->initializer);
			// Can only infer typed array if it has elements.
			if ((p_variable->infer_datatype && array->elements.size() > 0) || p_variable->datatype_specifier != nullptr) {
				update_array_literal_element_type(specified_type, array);
			}
		}

		type = p_variable->initializer->get_datatype();

		if (p_variable->infer_datatype) {
			type.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;

			if (type.has_no_type()) {
				push_error(vformat(R"(Could not infer the type of the variable "%s" because the initial value does not have a set type.)", p_variable->identifier->name), p_variable->initializer);
			} else if (type.is_variant()) {
				push_error(vformat(R"(Could not infer the type of the variable "%s" because the initial value is a variant. Use explicit "Variant" type if this is intended.)", p_variable->identifier->name), p_variable->initializer);
			} else if (type.kind == GDScriptParser::DataType::BUILTIN && type.builtin_type == Variant::NIL) {
				push_error(vformat(R"(Could not infer the type of the variable "%s" because the initial value is "null".)", p_variable->identifier->name), p_variable->initializer);
			}
		} else {
			type.type_source = GDScriptParser::DataType::INFERRED;
		}
#ifdef DEBUG_ENABLED
		if (p_variable->initializer->type == GDScriptParser::Node::CALL && type.kind == GDScriptParser::DataType::BUILTIN && type.builtin_type == Variant::NIL) {
			parser->push_warning(p_variable->initializer, GDScriptWarning::VOID_ASSIGNMENT, static_cast<GDScriptParser::CallNode *>(p_variable->initializer)->function_name);
		}
#endif
	}

	if (p_variable->datatype_specifier != nullptr) {
		type = specified_type;
		type.is_meta_type = false;

		if (p_variable->initializer != nullptr) {
			if (!is_type_compatible(type, p_variable->initializer->get_datatype(), true)) {
				// Try reverse test since it can be a masked subtype.
				if (!is_type_compatible(p_variable->initializer->get_datatype(), type, true)) {
					push_error(vformat(R"(Value of type "%s" cannot be assigned to a variable of type "%s".)", p_variable->initializer->get_datatype().to_string(), type.to_string()), p_variable->initializer);
				} else {
					// TODO: Add warning.
					mark_node_unsafe(p_variable->initializer);
					p_variable->use_conversion_assign = true;
				}
#ifdef DEBUG_ENABLED
			} else if (type.builtin_type == Variant::INT && p_variable->initializer->get_datatype().builtin_type == Variant::FLOAT) {
				parser->push_warning(p_variable->initializer, GDScriptWarning::NARROWING_CONVERSION);
#endif
			}
			if (p_variable->initializer->get_datatype().is_variant()) {
				// TODO: Warn unsafe assign.
				mark_node_unsafe(p_variable->initializer);
				p_variable->use_conversion_assign = true;
			}
		}
	} else if (p_variable->infer_datatype) {
		if (type.has_no_type()) {
			push_error(vformat(R"(Cannot infer the type of variable "%s" because the initial value doesn't have a set type.)", p_variable->identifier->name), p_variable->identifier);
		}
		type.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
	}

	type.is_constant = false;
	p_variable->set_datatype(type);

#ifdef DEBUG_ENABLED
	if (p_variable->usages == 0 && !String(p_variable->identifier->name).begins_with("_")) {
		parser->push_warning(p_variable, GDScriptWarning::UNUSED_VARIABLE, p_variable->identifier->name);
	} else if (p_variable->assignments == 0) {
		parser->push_warning(p_variable, GDScriptWarning::UNASSIGNED_VARIABLE, p_variable->identifier->name);
	}

	is_shadowing(p_variable->identifier, "variable");
#endif
}

void GDScriptAnalyzer::resolve_constant(GDScriptParser::ConstantNode *p_constant) {
	GDScriptParser::DataType type;

	if (p_constant->initializer != nullptr) {
		reduce_expression(p_constant->initializer);
		if (p_constant->initializer->type == GDScriptParser::Node::ARRAY) {
			const_fold_array(static_cast<GDScriptParser::ArrayNode *>(p_constant->initializer));
		} else if (p_constant->initializer->type == GDScriptParser::Node::DICTIONARY) {
			const_fold_dictionary(static_cast<GDScriptParser::DictionaryNode *>(p_constant->initializer));
		}

		if (!p_constant->initializer->is_constant) {
			push_error(vformat(R"(Assigned value for constant "%s" isn't a constant expression.)", p_constant->identifier->name), p_constant->initializer);
		}

		type = p_constant->initializer->get_datatype();

#ifdef DEBUG_ENABLED
		if (p_constant->initializer->type == GDScriptParser::Node::CALL && type.kind == GDScriptParser::DataType::BUILTIN && type.builtin_type == Variant::NIL) {
			parser->push_warning(p_constant->initializer, GDScriptWarning::VOID_ASSIGNMENT, static_cast<GDScriptParser::CallNode *>(p_constant->initializer)->function_name);
		}
#endif
	}

	if (p_constant->datatype_specifier != nullptr) {
		GDScriptParser::DataType explicit_type = resolve_datatype(p_constant->datatype_specifier);
		explicit_type.is_meta_type = false;
		if (!is_type_compatible(explicit_type, type)) {
			push_error(vformat(R"(Assigned value for constant "%s" has type %s which is not compatible with defined type %s.)", p_constant->identifier->name, type.to_string(), explicit_type.to_string()), p_constant->initializer);
#ifdef DEBUG_ENABLED
		} else if (explicit_type.builtin_type == Variant::INT && type.builtin_type == Variant::FLOAT) {
			parser->push_warning(p_constant->initializer, GDScriptWarning::NARROWING_CONVERSION);
#endif
		}
		type = explicit_type;
	} else if (p_constant->infer_datatype) {
		if (type.has_no_type()) {
			push_error(vformat(R"(Cannot infer the type of constant "%s" because the initial value doesn't have a set type.)", p_constant->identifier->name), p_constant->identifier);
		}
		type.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
	}

	type.is_constant = true;
	p_constant->set_datatype(type);

#ifdef DEBUG_ENABLED
	if (p_constant->usages == 0) {
		parser->push_warning(p_constant, GDScriptWarning::UNUSED_LOCAL_CONSTANT, p_constant->identifier->name);
	}

	is_shadowing(p_constant->identifier, "constant");
#endif
}

void GDScriptAnalyzer::resolve_assert(GDScriptParser::AssertNode *p_assert) {
	reduce_expression(p_assert->condition);
	if (p_assert->message != nullptr) {
		reduce_expression(p_assert->message);
		if (!p_assert->message->is_constant || p_assert->message->reduced_value.get_type() != Variant::STRING) {
			push_error(R"(Expected constant string for assert error message.)", p_assert->message);
		}
	}

	p_assert->set_datatype(p_assert->condition->get_datatype());

#ifdef DEBUG_ENABLED
	if (p_assert->condition->is_constant) {
		if (p_assert->condition->reduced_value.booleanize()) {
			parser->push_warning(p_assert->condition, GDScriptWarning::ASSERT_ALWAYS_TRUE);
		} else {
			parser->push_warning(p_assert->condition, GDScriptWarning::ASSERT_ALWAYS_FALSE);
		}
	}
#endif
}

void GDScriptAnalyzer::resolve_match(GDScriptParser::MatchNode *p_match) {
	reduce_expression(p_match->test);

	for (int i = 0; i < p_match->branches.size(); i++) {
		resolve_match_branch(p_match->branches[i], p_match->test);

		decide_suite_type(p_match, p_match->branches[i]);
	}
}

void GDScriptAnalyzer::resolve_match_branch(GDScriptParser::MatchBranchNode *p_match_branch, GDScriptParser::ExpressionNode *p_match_test) {
	for (int i = 0; i < p_match_branch->patterns.size(); i++) {
		resolve_match_pattern(p_match_branch->patterns[i], p_match_test);
	}

	resolve_suite(p_match_branch->block);

	decide_suite_type(p_match_branch, p_match_branch->block);
}

void GDScriptAnalyzer::resolve_match_pattern(GDScriptParser::PatternNode *p_match_pattern, GDScriptParser::ExpressionNode *p_match_test) {
	if (p_match_pattern == nullptr) {
		return;
	}

	GDScriptParser::DataType result;

	switch (p_match_pattern->pattern_type) {
		case GDScriptParser::PatternNode::PT_LITERAL:
			if (p_match_pattern->literal) {
				reduce_literal(p_match_pattern->literal);
				result = p_match_pattern->literal->get_datatype();
			}
			break;
		case GDScriptParser::PatternNode::PT_EXPRESSION:
			if (p_match_pattern->expression) {
				reduce_expression(p_match_pattern->expression);
				if (!p_match_pattern->expression->is_constant) {
					push_error(R"(Expression in match pattern must be a constant.)", p_match_pattern->expression);
				}
				result = p_match_pattern->expression->get_datatype();
			}
			break;
		case GDScriptParser::PatternNode::PT_BIND:
			if (p_match_test != nullptr) {
				result = p_match_test->get_datatype();
			} else {
				result.kind = GDScriptParser::DataType::VARIANT;
			}
			p_match_pattern->bind->set_datatype(result);
#ifdef DEBUG_ENABLED
			is_shadowing(p_match_pattern->bind, "pattern bind");
			if (p_match_pattern->bind->usages == 0) {
				parser->push_warning(p_match_pattern->bind, GDScriptWarning::UNASSIGNED_VARIABLE, p_match_pattern->bind->name);
			}
#endif
			break;
		case GDScriptParser::PatternNode::PT_ARRAY:
			for (int i = 0; i < p_match_pattern->array.size(); i++) {
				resolve_match_pattern(p_match_pattern->array[i], nullptr);
				decide_suite_type(p_match_pattern, p_match_pattern->array[i]);
			}
			result = p_match_pattern->get_datatype();
			break;
		case GDScriptParser::PatternNode::PT_DICTIONARY:
			for (int i = 0; i < p_match_pattern->dictionary.size(); i++) {
				if (p_match_pattern->dictionary[i].key) {
					reduce_expression(p_match_pattern->dictionary[i].key);
					if (!p_match_pattern->dictionary[i].key->is_constant) {
						push_error(R"(Expression in dictionary pattern key must be a constant.)", p_match_pattern->expression);
					}
				}

				if (p_match_pattern->dictionary[i].value_pattern) {
					resolve_match_pattern(p_match_pattern->dictionary[i].value_pattern, nullptr);
					decide_suite_type(p_match_pattern, p_match_pattern->dictionary[i].value_pattern);
				}
			}
			result = p_match_pattern->get_datatype();
			break;
		case GDScriptParser::PatternNode::PT_WILDCARD:
		case GDScriptParser::PatternNode::PT_REST:
			result.kind = GDScriptParser::DataType::VARIANT;
			break;
	}

	p_match_pattern->set_datatype(result);
}

void GDScriptAnalyzer::resolve_parameter(GDScriptParser::ParameterNode *p_parameter) {
	GDScriptParser::DataType result;
	result.kind = GDScriptParser::DataType::VARIANT;

	if (p_parameter->default_value != nullptr) {
		reduce_expression(p_parameter->default_value);
		result = p_parameter->default_value->get_datatype();
		if (p_parameter->infer_datatype) {
			result.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
		} else {
			result.type_source = GDScriptParser::DataType::INFERRED;
		}
		result.is_constant = false;
	}

	if (p_parameter->datatype_specifier != nullptr) {
		result = resolve_datatype(p_parameter->datatype_specifier);
		result.is_meta_type = false;

		if (p_parameter->default_value != nullptr) {
			if (!is_type_compatible(result, p_parameter->default_value->get_datatype())) {
				push_error(vformat(R"(Type of default value for parameter "%s" (%s) is not compatible with parameter type (%s).)", p_parameter->identifier->name, p_parameter->default_value->get_datatype().to_string(), p_parameter->datatype_specifier->get_datatype().to_string()), p_parameter->default_value);
			} else if (p_parameter->default_value->get_datatype().is_variant()) {
				mark_node_unsafe(p_parameter);
			}
		}
	}

	if (result.builtin_type == Variant::Type::NIL && result.type_source == GDScriptParser::DataType::ANNOTATED_INFERRED && p_parameter->datatype_specifier == nullptr) {
		push_error(vformat(R"(Could not infer the type of the variable "%s" because the initial value is "null".)", p_parameter->identifier->name), p_parameter->default_value);
	}

	p_parameter->set_datatype(result);
}

void GDScriptAnalyzer::resolve_return(GDScriptParser::ReturnNode *p_return) {
	GDScriptParser::DataType result;

	GDScriptParser::DataType expected_type;
	bool has_expected_type = false;

	if (parser->current_function != nullptr) {
		expected_type = parser->current_function->get_datatype();
		has_expected_type = true;
	}

	if (p_return->return_value != nullptr) {
		reduce_expression(p_return->return_value);
		if (p_return->return_value->type == GDScriptParser::Node::ARRAY) {
			// Check if assigned value is an array literal, so we can make it a typed array too if appropriate.
			if (has_expected_type && expected_type.has_container_element_type() && p_return->return_value->type == GDScriptParser::Node::ARRAY) {
				update_array_literal_element_type(expected_type, static_cast<GDScriptParser::ArrayNode *>(p_return->return_value));
			}
		}
		result = p_return->return_value->get_datatype();
	} else {
		// Return type is null by default.
		result.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
		result.kind = GDScriptParser::DataType::BUILTIN;
		result.builtin_type = Variant::NIL;
		result.is_constant = true;
	}

	if (has_expected_type) {
		expected_type.is_meta_type = false;
		if (expected_type.is_hard_type()) {
			if (!is_type_compatible(expected_type, result)) {
				// Try other way. Okay but not safe.
				if (!is_type_compatible(result, expected_type)) {
					push_error(vformat(R"(Cannot return value of type "%s" because the function return type is "%s".)", result.to_string(), expected_type.to_string()), p_return);
				} else {
					// TODO: Add warning.
					mark_node_unsafe(p_return);
				}
#ifdef DEBUG_ENABLED
			} else if (expected_type.builtin_type == Variant::INT && result.builtin_type == Variant::FLOAT) {
				parser->push_warning(p_return, GDScriptWarning::NARROWING_CONVERSION);
			} else if (result.is_variant()) {
				mark_node_unsafe(p_return);
#endif
			}
		}
	}

	p_return->set_datatype(result);
}

void GDScriptAnalyzer::reduce_expression(GDScriptParser::ExpressionNode *p_expression, bool p_is_root) {
	// This one makes some magic happen.

	if (p_expression == nullptr) {
		return;
	}

	if (p_expression->reduced) {
		// Don't do this more than once.
		return;
	}

	p_expression->reduced = true;

	switch (p_expression->type) {
		case GDScriptParser::Node::ARRAY:
			reduce_array(static_cast<GDScriptParser::ArrayNode *>(p_expression));
			break;
		case GDScriptParser::Node::ASSIGNMENT:
			reduce_assignment(static_cast<GDScriptParser::AssignmentNode *>(p_expression));
			break;
		case GDScriptParser::Node::AWAIT:
			reduce_await(static_cast<GDScriptParser::AwaitNode *>(p_expression));
			break;
		case GDScriptParser::Node::BINARY_OPERATOR:
			reduce_binary_op(static_cast<GDScriptParser::BinaryOpNode *>(p_expression));
			break;
		case GDScriptParser::Node::CALL:
			reduce_call(static_cast<GDScriptParser::CallNode *>(p_expression), p_is_root);
			break;
		case GDScriptParser::Node::CAST:
			reduce_cast(static_cast<GDScriptParser::CastNode *>(p_expression));
			break;
		case GDScriptParser::Node::DICTIONARY:
			reduce_dictionary(static_cast<GDScriptParser::DictionaryNode *>(p_expression));
			break;
		case GDScriptParser::Node::GET_NODE:
			reduce_get_node(static_cast<GDScriptParser::GetNodeNode *>(p_expression));
			break;
		case GDScriptParser::Node::IDENTIFIER:
			reduce_identifier(static_cast<GDScriptParser::IdentifierNode *>(p_expression));
			break;
		case GDScriptParser::Node::LAMBDA:
			reduce_lambda(static_cast<GDScriptParser::LambdaNode *>(p_expression));
			break;
		case GDScriptParser::Node::LITERAL:
			reduce_literal(static_cast<GDScriptParser::LiteralNode *>(p_expression));
			break;
		case GDScriptParser::Node::PRELOAD:
			reduce_preload(static_cast<GDScriptParser::PreloadNode *>(p_expression));
			break;
		case GDScriptParser::Node::SELF:
			reduce_self(static_cast<GDScriptParser::SelfNode *>(p_expression));
			break;
		case GDScriptParser::Node::SUBSCRIPT:
			reduce_subscript(static_cast<GDScriptParser::SubscriptNode *>(p_expression));
			break;
		case GDScriptParser::Node::TERNARY_OPERATOR:
			reduce_ternary_op(static_cast<GDScriptParser::TernaryOpNode *>(p_expression));
			break;
		case GDScriptParser::Node::UNARY_OPERATOR:
			reduce_unary_op(static_cast<GDScriptParser::UnaryOpNode *>(p_expression));
			break;
		// Non-expressions. Here only to make sure new nodes aren't forgotten.
		case GDScriptParser::Node::NONE:
		case GDScriptParser::Node::ANNOTATION:
		case GDScriptParser::Node::ASSERT:
		case GDScriptParser::Node::BREAK:
		case GDScriptParser::Node::BREAKPOINT:
		case GDScriptParser::Node::CLASS:
		case GDScriptParser::Node::CONSTANT:
		case GDScriptParser::Node::CONTINUE:
		case GDScriptParser::Node::ENUM:
		case GDScriptParser::Node::FOR:
		case GDScriptParser::Node::FUNCTION:
		case GDScriptParser::Node::IF:
		case GDScriptParser::Node::MATCH:
		case GDScriptParser::Node::MATCH_BRANCH:
		case GDScriptParser::Node::PARAMETER:
		case GDScriptParser::Node::PASS:
		case GDScriptParser::Node::PATTERN:
		case GDScriptParser::Node::RETURN:
		case GDScriptParser::Node::SIGNAL:
		case GDScriptParser::Node::SUITE:
		case GDScriptParser::Node::TYPE:
		case GDScriptParser::Node::VARIABLE:
		case GDScriptParser::Node::WHILE:
			ERR_FAIL_MSG("Reaching unreachable case");
	}
}

void GDScriptAnalyzer::reduce_array(GDScriptParser::ArrayNode *p_array) {
	for (int i = 0; i < p_array->elements.size(); i++) {
		GDScriptParser::ExpressionNode *element = p_array->elements[i];
		reduce_expression(element);
	}

	// It's array in any case.
	GDScriptParser::DataType arr_type;
	arr_type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	arr_type.kind = GDScriptParser::DataType::BUILTIN;
	arr_type.builtin_type = Variant::ARRAY;
	arr_type.is_constant = true;

	p_array->set_datatype(arr_type);
}

// When an array literal is stored (or passed as function argument) to a typed context, we then assume the array is typed.
// This function determines which type is that (if any).
void GDScriptAnalyzer::update_array_literal_element_type(const GDScriptParser::DataType &p_base_type, GDScriptParser::ArrayNode *p_array_literal) {
	GDScriptParser::DataType array_type = p_array_literal->get_datatype();
	if (p_array_literal->elements.size() == 0) {
		// Empty array literal, just make the same type as the storage.
		array_type.set_container_element_type(p_base_type.get_container_element_type());
	} else {
		// Check if elements match.
		bool all_same_type = true;
		bool all_have_type = true;

		GDScriptParser::DataType element_type;
		for (int i = 0; i < p_array_literal->elements.size(); i++) {
			if (i == 0) {
				element_type = p_array_literal->elements[0]->get_datatype();
			} else {
				GDScriptParser::DataType this_element_type = p_array_literal->elements[i]->get_datatype();
				if (this_element_type.has_no_type()) {
					all_same_type = false;
					all_have_type = false;
					break;
				} else if (element_type != this_element_type) {
					if (!is_type_compatible(element_type, this_element_type, false)) {
						if (is_type_compatible(this_element_type, element_type, false)) {
							// This element is a super-type to the previous type, so we use the super-type.
							element_type = this_element_type;
						} else {
							// It's incompatible.
							all_same_type = false;
							break;
						}
					}
				}
			}
		}
		if (all_same_type) {
			element_type.is_constant = false;
			array_type.set_container_element_type(element_type);
		} else if (all_have_type) {
			push_error(vformat(R"(Variant array is not compatible with an array of type "%s".)", p_base_type.get_container_element_type().to_string()), p_array_literal);
		}
	}
	// Update the type on the value itself.
	p_array_literal->set_datatype(array_type);
}

void GDScriptAnalyzer::reduce_assignment(GDScriptParser::AssignmentNode *p_assignment) {
	reduce_expression(p_assignment->assignee);
	reduce_expression(p_assignment->assigned_value);

	if (p_assignment->assigned_value == nullptr || p_assignment->assignee == nullptr) {
		return;
	}

	GDScriptParser::DataType assignee_type = p_assignment->assignee->get_datatype();

	// Check if assigned value is an array literal, so we can make it a typed array too if appropriate.
	if (assignee_type.has_container_element_type() && p_assignment->assigned_value->type == GDScriptParser::Node::ARRAY) {
		update_array_literal_element_type(assignee_type, static_cast<GDScriptParser::ArrayNode *>(p_assignment->assigned_value));
	}

	GDScriptParser::DataType assigned_value_type = p_assignment->assigned_value->get_datatype();

	if (assignee_type.is_constant) {
		push_error("Cannot assign a new value to a constant.", p_assignment->assignee);
	}

	if (!assignee_type.is_variant() && assigned_value_type.is_hard_type()) {
		bool compatible = true;
		GDScriptParser::DataType op_type = assigned_value_type;
		if (p_assignment->operation != GDScriptParser::AssignmentNode::OP_NONE) {
			op_type = get_operation_type(p_assignment->variant_op, assignee_type, assigned_value_type, compatible, p_assignment->assigned_value);
		}

		if (compatible) {
			compatible = is_type_compatible(assignee_type, op_type, true);
			if (!compatible) {
				if (assignee_type.is_hard_type()) {
					// Try reverse test since it can be a masked subtype.
					if (!is_type_compatible(op_type, assignee_type, true)) {
						push_error(vformat(R"(Cannot assign a value of type "%s" to a target of type "%s".)", assigned_value_type.to_string(), assignee_type.to_string()), p_assignment->assigned_value);
					} else {
						// TODO: Add warning.
						mark_node_unsafe(p_assignment);
						p_assignment->use_conversion_assign = true;
					}
				} else {
					// TODO: Warning in this case.
					mark_node_unsafe(p_assignment);
				}
			}
		} else {
			push_error(vformat(R"(Invalid operands "%s" and "%s" for assignment operator.)", assignee_type.to_string(), assigned_value_type.to_string()), p_assignment);
		}
	}

	if (assignee_type.has_no_type() || assigned_value_type.is_variant()) {
		mark_node_unsafe(p_assignment);
		if (assignee_type.is_hard_type()) {
			p_assignment->use_conversion_assign = true;
		}
	}

	if (p_assignment->assignee->type == GDScriptParser::Node::IDENTIFIER) {
		// Change source type so it's not wrongly detected later.
		GDScriptParser::IdentifierNode *identifier = static_cast<GDScriptParser::IdentifierNode *>(p_assignment->assignee);

		switch (identifier->source) {
			case GDScriptParser::IdentifierNode::MEMBER_VARIABLE: {
				GDScriptParser::DataType id_type = identifier->variable_source->get_datatype();
				if (!id_type.is_hard_type()) {
					id_type.kind = GDScriptParser::DataType::VARIANT;
					id_type.type_source = GDScriptParser::DataType::UNDETECTED;
					identifier->variable_source->set_datatype(id_type);
				}
			} break;
			case GDScriptParser::IdentifierNode::FUNCTION_PARAMETER: {
				GDScriptParser::DataType id_type = identifier->parameter_source->get_datatype();
				if (!id_type.is_hard_type()) {
					id_type.kind = GDScriptParser::DataType::VARIANT;
					id_type.type_source = GDScriptParser::DataType::UNDETECTED;
					identifier->parameter_source->set_datatype(id_type);
				}
			} break;
			case GDScriptParser::IdentifierNode::LOCAL_VARIABLE: {
				GDScriptParser::DataType id_type = identifier->variable_source->get_datatype();
				if (!id_type.is_hard_type()) {
					id_type.kind = GDScriptParser::DataType::VARIANT;
					id_type.type_source = GDScriptParser::DataType::UNDETECTED;
					identifier->variable_source->set_datatype(id_type);
				}
			} break;
			case GDScriptParser::IdentifierNode::LOCAL_ITERATOR: {
				GDScriptParser::DataType id_type = identifier->bind_source->get_datatype();
				if (!id_type.is_hard_type()) {
					id_type.kind = GDScriptParser::DataType::VARIANT;
					id_type.type_source = GDScriptParser::DataType::UNDETECTED;
					identifier->variable_source->set_datatype(id_type);
				}
			} break;
			default:
				// Nothing to do.
				break;
		}
	}

#ifdef DEBUG_ENABLED
	if (p_assignment->assigned_value->type == GDScriptParser::Node::CALL && assigned_value_type.kind == GDScriptParser::DataType::BUILTIN && assigned_value_type.builtin_type == Variant::NIL) {
		parser->push_warning(p_assignment->assigned_value, GDScriptWarning::VOID_ASSIGNMENT, static_cast<GDScriptParser::CallNode *>(p_assignment->assigned_value)->function_name);
	} else if (assignee_type.is_hard_type() && assignee_type.builtin_type == Variant::INT && assigned_value_type.builtin_type == Variant::FLOAT) {
		parser->push_warning(p_assignment->assigned_value, GDScriptWarning::NARROWING_CONVERSION);
	}
#endif
}

void GDScriptAnalyzer::reduce_await(GDScriptParser::AwaitNode *p_await) {
	if (p_await->to_await == nullptr) {
		GDScriptParser::DataType await_type;
		await_type.kind = GDScriptParser::DataType::VARIANT;
		p_await->set_datatype(await_type);
		return;
	}

	GDScriptParser::DataType awaiting_type;

	if (p_await->to_await->type == GDScriptParser::Node::CALL) {
		reduce_call(static_cast<GDScriptParser::CallNode *>(p_await->to_await), true);
		awaiting_type = p_await->to_await->get_datatype();
	} else {
		reduce_expression(p_await->to_await);
	}

	if (p_await->to_await->is_constant) {
		p_await->is_constant = p_await->to_await->is_constant;
		p_await->reduced_value = p_await->to_await->reduced_value;

		awaiting_type = p_await->to_await->get_datatype();
	} else {
		awaiting_type.kind = GDScriptParser::DataType::VARIANT;
		awaiting_type.type_source = GDScriptParser::DataType::UNDETECTED;
	}

	p_await->set_datatype(awaiting_type);

#ifdef DEBUG_ENABLED
	if (!awaiting_type.is_coroutine && awaiting_type.builtin_type != Variant::SIGNAL) {
		parser->push_warning(p_await, GDScriptWarning::REDUNDANT_AWAIT);
	}
#endif
}

void GDScriptAnalyzer::reduce_binary_op(GDScriptParser::BinaryOpNode *p_binary_op) {
	reduce_expression(p_binary_op->left_operand);

	if (p_binary_op->operation == GDScriptParser::BinaryOpNode::OP_TYPE_TEST && p_binary_op->right_operand && p_binary_op->right_operand->type == GDScriptParser::Node::IDENTIFIER) {
		reduce_identifier(static_cast<GDScriptParser::IdentifierNode *>(p_binary_op->right_operand), true);
	} else {
		reduce_expression(p_binary_op->right_operand);
	}
	// TODO: Right operand must be a valid type with the `is` operator. Need to check here.

	GDScriptParser::DataType left_type;
	if (p_binary_op->left_operand) {
		left_type = p_binary_op->left_operand->get_datatype();
	}
	GDScriptParser::DataType right_type;
	if (p_binary_op->right_operand) {
		right_type = p_binary_op->right_operand->get_datatype();
	}

	if (!left_type.is_set() || !right_type.is_set()) {
		return;
	}

#ifdef DEBUG_ENABLED
	if (p_binary_op->variant_op == Variant::OP_DIVIDE && left_type.builtin_type == Variant::INT && right_type.builtin_type == Variant::INT) {
		parser->push_warning(p_binary_op, GDScriptWarning::INTEGER_DIVISION);
	}
#endif

	if (p_binary_op->left_operand->is_constant && p_binary_op->right_operand->is_constant) {
		p_binary_op->is_constant = true;
		if (p_binary_op->variant_op < Variant::OP_MAX) {
			bool valid = false;
			Variant::evaluate(p_binary_op->variant_op, p_binary_op->left_operand->reduced_value, p_binary_op->right_operand->reduced_value, p_binary_op->reduced_value, valid);
			if (!valid) {
				if (p_binary_op->reduced_value.get_type() == Variant::STRING) {
					push_error(vformat(R"(%s in operator %s.)", p_binary_op->reduced_value, Variant::get_operator_name(p_binary_op->variant_op)), p_binary_op);
				} else {
					push_error(vformat(R"(Invalid operands to operator %s, %s and %s.)",
									   Variant::get_operator_name(p_binary_op->variant_op),
									   Variant::get_type_name(p_binary_op->left_operand->reduced_value.get_type()),
									   Variant::get_type_name(p_binary_op->right_operand->reduced_value.get_type())),
							p_binary_op);
				}
			}
		} else {
			if (p_binary_op->operation == GDScriptParser::BinaryOpNode::OP_TYPE_TEST) {
				GDScriptParser::DataType test_type = right_type;
				test_type.is_meta_type = false;

				if (!is_type_compatible(test_type, p_binary_op->left_operand->get_datatype(), false)) {
					push_error(vformat(R"(Expression is of type "%s" so it can't be of type "%s".)"), p_binary_op->left_operand);
					p_binary_op->reduced_value = false;
				} else {
					p_binary_op->reduced_value = true;
				}
			} else {
				ERR_PRINT("Parser bug: unknown binary operation.");
			}
		}
		p_binary_op->set_datatype(type_from_variant(p_binary_op->reduced_value, p_binary_op));

		return;
	}

	GDScriptParser::DataType result;

	if (left_type.is_variant() || right_type.is_variant()) {
		// Cannot infer type because one operand can be anything.
		result.kind = GDScriptParser::DataType::VARIANT;
		mark_node_unsafe(p_binary_op);
	} else {
		if (p_binary_op->variant_op < Variant::OP_MAX) {
			bool valid = false;
			result = get_operation_type(p_binary_op->variant_op, left_type, right_type, valid, p_binary_op);

			if (!valid) {
				push_error(vformat(R"(Invalid operands "%s" and "%s" for "%s" operator.)", left_type.to_string(), right_type.to_string(), Variant::get_operator_name(p_binary_op->variant_op)), p_binary_op);
			}
		} else {
			if (p_binary_op->operation == GDScriptParser::BinaryOpNode::OP_TYPE_TEST) {
				GDScriptParser::DataType test_type = right_type;
				test_type.is_meta_type = false;

				if (!is_type_compatible(test_type, p_binary_op->left_operand->get_datatype(), false)) {
					// Test reverse as well to consider for subtypes.
					if (!is_type_compatible(p_binary_op->left_operand->get_datatype(), test_type, false)) {
						if (p_binary_op->left_operand->get_datatype().is_hard_type()) {
							push_error(vformat(R"(Expression is of type "%s" so it can't be of type "%s".)", p_binary_op->left_operand->get_datatype().to_string(), test_type.to_string()), p_binary_op->left_operand);
						} else {
							// TODO: Warning.
							mark_node_unsafe(p_binary_op);
						}
					}
				}

				// "is" operator is always a boolean anyway.
				result.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
				result.kind = GDScriptParser::DataType::BUILTIN;
				result.builtin_type = Variant::BOOL;
			} else {
				ERR_PRINT("Parser bug: unknown binary operation.");
			}
		}
	}

	p_binary_op->set_datatype(result);
}

void GDScriptAnalyzer::reduce_call(GDScriptParser::CallNode *p_call, bool p_is_await, bool p_is_root) {
	bool all_is_constant = true;
	Map<int, GDScriptParser::ArrayNode *> arrays; // For array literal to potentially type when passing.
	for (int i = 0; i < p_call->arguments.size(); i++) {
		reduce_expression(p_call->arguments[i]);
		if (p_call->arguments[i]->type == GDScriptParser::Node::ARRAY) {
			arrays[i] = static_cast<GDScriptParser::ArrayNode *>(p_call->arguments[i]);
		}
		all_is_constant = all_is_constant && p_call->arguments[i]->is_constant;
	}

	GDScriptParser::Node::Type callee_type = p_call->get_callee_type();
	GDScriptParser::DataType call_type;

	if (!p_call->is_super && callee_type == GDScriptParser::Node::IDENTIFIER) {
		// Call to name directly.
		StringName function_name = p_call->function_name;
		Variant::Type builtin_type = GDScriptParser::get_builtin_type(function_name);

		if (builtin_type < Variant::VARIANT_MAX) {
			// Is a builtin constructor.
			call_type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
			call_type.kind = GDScriptParser::DataType::BUILTIN;
			call_type.builtin_type = builtin_type;

			if (builtin_type == Variant::OBJECT) {
				call_type.kind = GDScriptParser::DataType::NATIVE;
				call_type.native_type = function_name; // "Object".
			}

			bool safe_to_fold = true;
			switch (builtin_type) {
				// Those are stored by reference so not suited for compile-time construction.
				// Because in this case they would be the same reference in all constructed values.
				case Variant::OBJECT:
				case Variant::DICTIONARY:
				case Variant::ARRAY:
				case Variant::PACKED_BYTE_ARRAY:
				case Variant::PACKED_INT32_ARRAY:
				case Variant::PACKED_INT64_ARRAY:
				case Variant::PACKED_FLOAT32_ARRAY:
				case Variant::PACKED_FLOAT64_ARRAY:
				case Variant::PACKED_STRING_ARRAY:
				case Variant::PACKED_VECTOR2_ARRAY:
				case Variant::PACKED_VECTOR3_ARRAY:
				case Variant::PACKED_COLOR_ARRAY:
					safe_to_fold = false;
					break;
				default:
					break;
			}

			if (all_is_constant && safe_to_fold) {
				// Construct here.
				Vector<const Variant *> args;
				for (int i = 0; i < p_call->arguments.size(); i++) {
					args.push_back(&(p_call->arguments[i]->reduced_value));
				}

				Callable::CallError err;
				Variant value;
				Variant::construct(builtin_type, value, (const Variant **)args.ptr(), args.size(), err);

				switch (err.error) {
					case Callable::CallError::CALL_ERROR_INVALID_ARGUMENT:
						push_error(vformat(R"(Invalid argument for %s constructor: argument %d should be %s but is %s.)", Variant::get_type_name(builtin_type), err.argument + 1,
										   Variant::get_type_name(Variant::Type(err.expected)), p_call->arguments[err.argument]->get_datatype().to_string()),
								p_call->arguments[err.argument]);
						break;
					case Callable::CallError::CALL_ERROR_INVALID_METHOD: {
						String signature = Variant::get_type_name(builtin_type) + "(";
						for (int i = 0; i < p_call->arguments.size(); i++) {
							if (i > 0) {
								signature += ", ";
							}
							signature += p_call->arguments[i]->get_datatype().to_string();
						}
						signature += ")";
						push_error(vformat(R"(No constructor of "%s" matches the signature "%s".)", Variant::get_type_name(builtin_type), signature), p_call->callee);
					} break;
					case Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS:
						push_error(vformat(R"(Too many arguments for %s constructor. Received %d but expected %d.)", Variant::get_type_name(builtin_type), p_call->arguments.size(), err.expected), p_call);
						break;
					case Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS:
						push_error(vformat(R"(Too few arguments for %s constructor. Received %d but expected %d.)", Variant::get_type_name(builtin_type), p_call->arguments.size(), err.expected), p_call);
						break;
					case Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL:
						break; // Can't happen in a builtin constructor.
					case Callable::CallError::CALL_OK:
						p_call->is_constant = true;
						p_call->reduced_value = value;
						break;
				}
			} else {
				// TODO: Check constructors without constants.

				// If there's one argument, try to use copy constructor (those aren't explicitly defined).
				if (p_call->arguments.size() == 1) {
					GDScriptParser::DataType arg_type = p_call->arguments[0]->get_datatype();
					if (arg_type.is_variant()) {
						mark_node_unsafe(p_call->arguments[0]);
					} else {
						if (arg_type.kind == GDScriptParser::DataType::BUILTIN && arg_type.builtin_type == builtin_type) {
							// Okay.
							p_call->set_datatype(call_type);
							return;
						}
					}
				}
				List<MethodInfo> constructors;
				Variant::get_constructor_list(builtin_type, &constructors);
				bool match = false;

				for (const MethodInfo &info : constructors) {
					if (p_call->arguments.size() < info.arguments.size() - info.default_arguments.size()) {
						continue;
					}
					if (p_call->arguments.size() > info.arguments.size()) {
						continue;
					}

					bool types_match = true;

					for (int i = 0; i < p_call->arguments.size(); i++) {
						GDScriptParser::DataType par_type = type_from_property(info.arguments[i]);

						if (!is_type_compatible(par_type, p_call->arguments[i]->get_datatype(), true)) {
							types_match = false;
							break;
#ifdef DEBUG_ENABLED
						} else {
							if (par_type.builtin_type == Variant::INT && p_call->arguments[i]->get_datatype().builtin_type == Variant::FLOAT) {
								parser->push_warning(p_call, GDScriptWarning::NARROWING_CONVERSION, p_call->function_name);
							}
#endif
						}
					}

					if (types_match) {
						match = true;
						call_type = type_from_property(info.return_val);
						break;
					}
				}

				if (!match) {
					String signature = Variant::get_type_name(builtin_type) + "(";
					for (int i = 0; i < p_call->arguments.size(); i++) {
						if (i > 0) {
							signature += ", ";
						}
						signature += p_call->arguments[i]->get_datatype().to_string();
					}
					signature += ")";
					push_error(vformat(R"(No constructor of "%s" matches the signature "%s".)", Variant::get_type_name(builtin_type), signature), p_call);
				}
			}
			p_call->set_datatype(call_type);
			return;
		} else if (GDScriptUtilityFunctions::function_exists(function_name)) {
			MethodInfo function_info = GDScriptUtilityFunctions::get_function_info(function_name);

			if (all_is_constant && GDScriptUtilityFunctions::is_function_constant(function_name)) {
				// Can call on compilation.
				Vector<const Variant *> args;
				for (int i = 0; i < p_call->arguments.size(); i++) {
					args.push_back(&(p_call->arguments[i]->reduced_value));
				}

				Variant value;
				Callable::CallError err;
				GDScriptUtilityFunctions::get_function(function_name)(&value, (const Variant **)args.ptr(), args.size(), err);

				switch (err.error) {
					case Callable::CallError::CALL_ERROR_INVALID_ARGUMENT: {
						PropertyInfo wrong_arg = function_info.arguments[err.argument];
						push_error(vformat(R"*(Invalid argument for "%s()" function: argument %d should be %s but is %s.)*", function_name, err.argument + 1,
										   type_from_property(wrong_arg).to_string(), p_call->arguments[err.argument]->get_datatype().to_string()),
								p_call->arguments[err.argument]);
					} break;
					case Callable::CallError::CALL_ERROR_INVALID_METHOD:
						push_error(vformat(R"(Invalid call for function "%s".)", function_name), p_call);
						break;
					case Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS:
						push_error(vformat(R"*(Too many arguments for "%s()" call. Expected at most %d but received %d.)*", function_name, err.expected, p_call->arguments.size()), p_call);
						break;
					case Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS:
						push_error(vformat(R"*(Too few arguments for "%s()" call. Expected at least %d but received %d.)*", function_name, err.expected, p_call->arguments.size()), p_call);
						break;
					case Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL:
						break; // Can't happen in a builtin constructor.
					case Callable::CallError::CALL_OK:
						p_call->is_constant = true;
						p_call->reduced_value = value;
						break;
				}
			} else {
				validate_call_arg(function_info, p_call);
			}
			p_call->set_datatype(type_from_property(function_info.return_val));
			return;
		} else if (Variant::has_utility_function(function_name)) {
			MethodInfo function_info = info_from_utility_func(function_name);

			if (all_is_constant && Variant::get_utility_function_type(function_name) == Variant::UTILITY_FUNC_TYPE_MATH) {
				// Can call on compilation.
				Vector<const Variant *> args;
				for (int i = 0; i < p_call->arguments.size(); i++) {
					args.push_back(&(p_call->arguments[i]->reduced_value));
				}

				Variant value;
				Callable::CallError err;
				Variant::call_utility_function(function_name, &value, (const Variant **)args.ptr(), args.size(), err);

				switch (err.error) {
					case Callable::CallError::CALL_ERROR_INVALID_ARGUMENT: {
						PropertyInfo wrong_arg = function_info.arguments[err.argument];
						push_error(vformat(R"*(Invalid argument for "%s()" function: argument %d should be %s but is %s.)*", function_name, err.argument + 1,
										   type_from_property(wrong_arg).to_string(), p_call->arguments[err.argument]->get_datatype().to_string()),
								p_call->arguments[err.argument]);
					} break;
					case Callable::CallError::CALL_ERROR_INVALID_METHOD:
						push_error(vformat(R"(Invalid call for function "%s".)", function_name), p_call);
						break;
					case Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS:
						push_error(vformat(R"*(Too many arguments for "%s()" call. Expected at most %d but received %d.)*", function_name, err.expected, p_call->arguments.size()), p_call);
						break;
					case Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS:
						push_error(vformat(R"*(Too few arguments for "%s()" call. Expected at least %d but received %d.)*", function_name, err.expected, p_call->arguments.size()), p_call);
						break;
					case Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL:
						break; // Can't happen in a builtin constructor.
					case Callable::CallError::CALL_OK:
						p_call->is_constant = true;
						p_call->reduced_value = value;
						break;
				}
			} else {
				validate_call_arg(function_info, p_call);
			}
			p_call->set_datatype(type_from_property(function_info.return_val));
			return;
		}
	}

	GDScriptParser::DataType base_type;
	call_type.kind = GDScriptParser::DataType::VARIANT;
	bool is_self = false;

	if (p_call->is_super) {
		base_type = parser->current_class->base_type;
		base_type.is_meta_type = false;
		is_self = true;
	} else if (callee_type == GDScriptParser::Node::IDENTIFIER) {
		base_type = parser->current_class->get_datatype();
		base_type.is_meta_type = false;
		is_self = true;
	} else if (callee_type == GDScriptParser::Node::SUBSCRIPT) {
		GDScriptParser::SubscriptNode *subscript = static_cast<GDScriptParser::SubscriptNode *>(p_call->callee);
		if (subscript->base == nullptr) {
			// Invalid syntax, error already set on parser.
			p_call->set_datatype(call_type);
			mark_node_unsafe(p_call);
			return;
		}
		if (!subscript->is_attribute) {
			// Invalid call. Error already sent in parser.
			// TODO: Could check if Callable here.
			p_call->set_datatype(call_type);
			mark_node_unsafe(p_call);
			return;
		}
		if (subscript->attribute == nullptr) {
			// Invalid call. Error already sent in parser.
			p_call->set_datatype(call_type);
			mark_node_unsafe(p_call);
			return;
		}

		GDScriptParser::IdentifierNode *base_id = nullptr;
		if (subscript->base->type == GDScriptParser::Node::IDENTIFIER) {
			base_id = static_cast<GDScriptParser::IdentifierNode *>(subscript->base);
		}
		if (base_id && GDScriptParser::get_builtin_type(base_id->name) < Variant::VARIANT_MAX) {
			base_type = make_builtin_meta_type(GDScriptParser::get_builtin_type(base_id->name));
		} else {
			reduce_expression(subscript->base);
			base_type = subscript->base->get_datatype();
		}
	} else {
		// Invalid call. Error already sent in parser.
		// TODO: Could check if Callable here too.
		p_call->set_datatype(call_type);
		mark_node_unsafe(p_call);
		return;
	}

	bool is_static = false;
	bool is_vararg = false;
	int default_arg_count = 0;
	GDScriptParser::DataType return_type;
	List<GDScriptParser::DataType> par_types;

	if (get_function_signature(p_call, base_type, p_call->function_name, return_type, par_types, default_arg_count, is_static, is_vararg)) {
		// If the function require typed arrays we must make literals be typed.
		for (const KeyValue<int, GDScriptParser::ArrayNode *> &E : arrays) {
			int index = E.key;
			if (index < par_types.size() && par_types[index].has_container_element_type()) {
				update_array_literal_element_type(par_types[index], E.value);
			}
		}
		validate_call_arg(par_types, default_arg_count, is_vararg, p_call);

		if (is_self && parser->current_function != nullptr && parser->current_function->is_static && !is_static) {
			push_error(vformat(R"*(Cannot call non-static function "%s()" from static function "%s()".)*", p_call->function_name, parser->current_function->identifier->name), p_call->callee);
		} else if (!is_self && base_type.is_meta_type && !is_static) {
			base_type.is_meta_type = false; // For `to_string()`.
			push_error(vformat(R"*(Cannot call non-static function "%s()" on the class "%s" directly. Make an instance instead.)*", p_call->function_name, base_type.to_string()), p_call->callee);
		} else if (is_self && !is_static && !lambda_stack.is_empty()) {
			push_error(vformat(R"*(Cannot call non-static function "%s()" from a lambda function.)*", p_call->function_name), p_call->callee);
		}

		call_type = return_type;
	} else {
		// Check if the name exists as something else.
		bool found = false;
		if (!p_call->is_super && callee_type != GDScriptParser::Node::NONE) {
			GDScriptParser::IdentifierNode *callee_id;
			if (callee_type == GDScriptParser::Node::IDENTIFIER) {
				callee_id = static_cast<GDScriptParser::IdentifierNode *>(p_call->callee);
			} else {
				// Can only be attribute.
				callee_id = static_cast<GDScriptParser::SubscriptNode *>(p_call->callee)->attribute;
			}
			if (callee_id) {
				reduce_identifier_from_base(callee_id, &base_type);
				GDScriptParser::DataType callee_datatype = callee_id->get_datatype();
				if (callee_datatype.is_set() && !callee_datatype.is_variant()) {
					found = true;
					if (callee_datatype.builtin_type == Variant::CALLABLE) {
						push_error(vformat(R"*(Name "%s" is a Callable. You can call it with "%s.call()" instead.)*", p_call->function_name, p_call->function_name), p_call->callee);
					} else {
						push_error(vformat(R"*(Name "%s" called as a function but is a "%s".)*", p_call->function_name, callee_datatype.to_string()), p_call->callee);
					}
#ifdef DEBUG_ENABLED
				} else if (!is_self && !(base_type.is_hard_type() && base_type.kind == GDScriptParser::DataType::BUILTIN)) {
					parser->push_warning(p_call, GDScriptWarning::UNSAFE_METHOD_ACCESS, p_call->function_name, base_type.to_string());
					mark_node_unsafe(p_call);
#endif
				}
			}
		}
		if (!found && (is_self || (base_type.is_hard_type() && base_type.kind == GDScriptParser::DataType::BUILTIN))) {
			String base_name = is_self && !p_call->is_super ? "self" : base_type.to_string();
			push_error(vformat(R"*(Function "%s()" not found in base %s.)*", p_call->function_name, base_name), p_call->is_super ? p_call : p_call->callee);
		}
	}

	if (call_type.is_coroutine && !p_is_await && !p_is_root) {
		push_error(vformat(R"*(Function "%s()" is a coroutine, so it must be called with "await".)*", p_call->function_name), p_call->callee);
	}

	p_call->set_datatype(call_type);
}

void GDScriptAnalyzer::reduce_cast(GDScriptParser::CastNode *p_cast) {
	reduce_expression(p_cast->operand);

	GDScriptParser::DataType cast_type = resolve_datatype(p_cast->cast_type);

	if (!cast_type.is_set()) {
		return;
	}

	cast_type.is_meta_type = false; // The casted value won't be a type name.
	p_cast->set_datatype(cast_type);

	if (!cast_type.is_variant()) {
		GDScriptParser::DataType op_type = p_cast->operand->get_datatype();
		if (!op_type.is_variant()) {
			bool valid = false;
			if (op_type.kind == GDScriptParser::DataType::BUILTIN && cast_type.kind == GDScriptParser::DataType::BUILTIN) {
				valid = Variant::can_convert(op_type.builtin_type, cast_type.builtin_type);
			} else if (op_type.kind != GDScriptParser::DataType::BUILTIN && cast_type.kind != GDScriptParser::DataType::BUILTIN) {
				valid = is_type_compatible(cast_type, op_type) || is_type_compatible(op_type, cast_type);
			}

			if (!valid) {
				push_error(vformat(R"(Invalid cast. Cannot convert from "%s" to "%s".)", op_type.to_string(), cast_type.to_string()), p_cast->cast_type);
			}
		}
	} else {
		mark_node_unsafe(p_cast);
	}
#ifdef DEBUG_ENABLED
	if (p_cast->operand->get_datatype().is_variant()) {
		parser->push_warning(p_cast, GDScriptWarning::UNSAFE_CAST, cast_type.to_string());
		mark_node_unsafe(p_cast);
	}
#endif

	// TODO: Perform cast on constants.
}

void GDScriptAnalyzer::reduce_dictionary(GDScriptParser::DictionaryNode *p_dictionary) {
	HashMap<Variant, GDScriptParser::ExpressionNode *, VariantHasher, VariantComparator> elements;

	for (int i = 0; i < p_dictionary->elements.size(); i++) {
		const GDScriptParser::DictionaryNode::Pair &element = p_dictionary->elements[i];
		if (p_dictionary->style == GDScriptParser::DictionaryNode::PYTHON_DICT) {
			reduce_expression(element.key);
		}
		reduce_expression(element.value);

		if (element.key->is_constant) {
			if (elements.has(element.key->reduced_value)) {
				push_error(vformat(R"(Key "%s" was already used in this dictionary (at line %d).)", element.key->reduced_value, elements[element.key->reduced_value]->start_line), element.key);
			} else {
				elements[element.key->reduced_value] = element.value;
			}
		}
	}

	// It's dictionary in any case.
	GDScriptParser::DataType dict_type;
	dict_type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	dict_type.kind = GDScriptParser::DataType::BUILTIN;
	dict_type.builtin_type = Variant::DICTIONARY;
	dict_type.is_constant = true;

	p_dictionary->set_datatype(dict_type);
}

void GDScriptAnalyzer::reduce_get_node(GDScriptParser::GetNodeNode *p_get_node) {
	GDScriptParser::DataType result;
	result.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	result.kind = GDScriptParser::DataType::NATIVE;
	result.native_type = "Node";
	result.builtin_type = Variant::OBJECT;

	if (!ClassDB::is_parent_class(parser->current_class->base_type.native_type, result.native_type)) {
		push_error(R"*(Cannot use shorthand "get_node()" notation ("$") on a class that isn't a node.)*", p_get_node);
	} else if (!lambda_stack.is_empty()) {
		push_error(R"*(Cannot use shorthand "get_node()" notation ("$") inside a lambda. Use a captured variable instead.)*", p_get_node);
	}

	p_get_node->set_datatype(result);
}

GDScriptParser::DataType GDScriptAnalyzer::make_global_class_meta_type(const StringName &p_class_name, const GDScriptParser::Node *p_source) {
	Ref<GDScriptParserRef> ref = get_parser_for(ScriptServer::get_global_class_path(p_class_name));
	Error err = ref->raise_status(GDScriptParserRef::INTERFACE_SOLVED);

	if (err) {
		push_error(vformat(R"(Could not resolve class "%s", because of a parser error.)", p_class_name), p_source);
		GDScriptParser::DataType type;
		type.type_source = GDScriptParser::DataType::UNDETECTED;
		type.kind = GDScriptParser::DataType::VARIANT;
		return type;
	}

	GDScriptParser::DataType type;
	type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	type.kind = GDScriptParser::DataType::CLASS;
	type.builtin_type = Variant::OBJECT;
	type.native_type = ScriptServer::get_global_class_native_base(p_class_name);
	type.class_type = ref->get_parser()->head;
	type.script_path = ref->get_parser()->script_path;
	type.is_constant = true;
	type.is_meta_type = true;
	return type;
}

void GDScriptAnalyzer::reduce_identifier_from_base(GDScriptParser::IdentifierNode *p_identifier, GDScriptParser::DataType *p_base) {
	GDScriptParser::DataType base;
	if (p_base == nullptr) {
		base = type_from_metatype(parser->current_class->get_datatype());
	} else {
		base = *p_base;
	}

	const StringName &name = p_identifier->name;

	if (base.kind == GDScriptParser::DataType::BUILTIN) {
		if (base.is_meta_type) {
			bool valid = true;
			Variant result = Variant::get_constant_value(base.builtin_type, name, &valid);
			if (valid) {
				p_identifier->is_constant = true;
				p_identifier->reduced_value = result;
				p_identifier->set_datatype(type_from_variant(result, p_identifier));
			} else if (base.is_hard_type()) {
				push_error(vformat(R"(Cannot find constant "%s" on type "%s".)", name, base.to_string()), p_identifier);
			}
		} else {
			switch (base.builtin_type) {
				case Variant::NIL: {
					if (base.is_hard_type()) {
						push_error(vformat(R"(Invalid get index "%s" on base Nil)", name), p_identifier);
					}
					return;
				}
				case Variant::DICTIONARY: {
					GDScriptParser::DataType dummy;
					dummy.kind = GDScriptParser::DataType::VARIANT;
					p_identifier->set_datatype(dummy);
					return;
				}
				default: {
					Callable::CallError temp;
					Variant dummy;
					Variant::construct(base.builtin_type, dummy, nullptr, 0, temp);
					List<PropertyInfo> properties;
					dummy.get_property_list(&properties);
					for (const PropertyInfo &prop : properties) {
						if (prop.name == name) {
							p_identifier->set_datatype(type_from_property(prop));
							return;
						}
					}
					if (base.is_hard_type()) {
						push_error(vformat(R"(Cannot find property "%s" on base "%s".)", name, base.to_string()), p_identifier);
					}
				}
			}
		}
		return;
	}

	if (base.kind == GDScriptParser::DataType::ENUM) {
		if (base.is_meta_type) {
			if (base.enum_values.has(name)) {
				p_identifier->is_constant = true;
				p_identifier->reduced_value = base.enum_values[name];

				GDScriptParser::DataType result;
				result.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
				result.kind = GDScriptParser::DataType::ENUM_VALUE;
				result.builtin_type = base.builtin_type;
				result.native_type = base.native_type;
				result.enum_type = name;
				p_identifier->set_datatype(result);
			} else {
				// Consider as a Dictionary
				GDScriptParser::DataType dummy;
				dummy.kind = GDScriptParser::DataType::VARIANT;
				p_identifier->set_datatype(dummy);
			}
		} else {
			push_error(R"(Cannot get property from enum value.)", p_identifier);
		}
		return;
	}

	GDScriptParser::ClassNode *base_class = base.class_type;

	// TODO: Switch current class/function/suite here to avoid misrepresenting identifiers (in recursive reduce calls).
	while (base_class != nullptr) {
		if (base_class->identifier && base_class->identifier->name == name) {
			p_identifier->set_datatype(base_class->get_datatype());
			return;
		}
		if (base_class->has_member(name)) {
			const GDScriptParser::ClassNode::Member &member = base_class->get_member(name);
			p_identifier->set_datatype(member.get_datatype());
			switch (member.type) {
				case GDScriptParser::ClassNode::Member::CONSTANT:
					// For out-of-order resolution:
					reduce_expression(member.constant->initializer);
					p_identifier->is_constant = true;
					p_identifier->reduced_value = member.constant->initializer->reduced_value;
					p_identifier->set_datatype(member.constant->initializer->get_datatype());
					p_identifier->source = GDScriptParser::IdentifierNode::MEMBER_CONSTANT;
					p_identifier->constant_source = member.constant;
					break;
				case GDScriptParser::ClassNode::Member::ENUM_VALUE:
					p_identifier->is_constant = true;
					p_identifier->reduced_value = member.enum_value.value;
					p_identifier->source = GDScriptParser::IdentifierNode::MEMBER_CONSTANT;
					break;
				case GDScriptParser::ClassNode::Member::VARIABLE:
					p_identifier->source = GDScriptParser::IdentifierNode::MEMBER_VARIABLE;
					p_identifier->variable_source = member.variable;
					member.variable->usages += 1;
					break;
				case GDScriptParser::ClassNode::Member::FUNCTION:
					resolve_function_signature(member.function);
					p_identifier->set_datatype(make_callable_type(member.function->info));
					break;
				case GDScriptParser::ClassNode::Member::CLASS:
					// For out-of-order resolution:
					resolve_class_interface(member.m_class);
					p_identifier->set_datatype(member.m_class->get_datatype());
					break;
				default:
					break; // Type already set.
			}
			return;
		}
		// Check outer constants.
		// TODO: Allow outer static functions.
		GDScriptParser::ClassNode *outer = base_class->outer;
		while (outer != nullptr) {
			if (outer->has_member(name)) {
				const GDScriptParser::ClassNode::Member &member = outer->get_member(name);
				switch (member.type) {
					case GDScriptParser::ClassNode::Member::CONSTANT: {
						// TODO: Make sure loops won't cause problem. And make special error message for those.
						// For out-of-order resolution:
						reduce_expression(member.constant->initializer);
						p_identifier->set_datatype(member.get_datatype());
						p_identifier->is_constant = true;
						p_identifier->reduced_value = member.constant->initializer->reduced_value;
						return;
					} break;
					case GDScriptParser::ClassNode::Member::ENUM_VALUE: {
						p_identifier->set_datatype(member.get_datatype());
						p_identifier->is_constant = true;
						p_identifier->reduced_value = member.enum_value.value;
						return;
					} break;
					case GDScriptParser::ClassNode::Member::ENUM: {
						p_identifier->set_datatype(member.get_datatype());
						p_identifier->is_constant = false;
						return;
					} break;
					case GDScriptParser::ClassNode::Member::CLASS: {
						resolve_class_interface(member.m_class);
						p_identifier->set_datatype(member.m_class->get_datatype());
						return;
					} break;
					default:
						break;
				}
			}
			outer = outer->outer;
		}

		base_class = base_class->base_type.class_type;
	}

	// Check native members.
	const StringName &native = base.native_type;

	if (class_exists(native)) {
		MethodInfo method_info;
		if (ClassDB::has_property(native, name)) {
			StringName getter_name = ClassDB::get_property_getter(native, name);
			MethodBind *getter = ClassDB::get_method(native, getter_name);
			if (getter != nullptr) {
				p_identifier->set_datatype(type_from_property(getter->get_return_info()));
			}
			return;
		}
		if (ClassDB::get_method_info(native, name, &method_info)) {
			// Method is callable.
			p_identifier->set_datatype(make_callable_type(method_info));
			return;
		}
		if (ClassDB::get_signal(native, name, &method_info)) {
			// Signal is a type too.
			p_identifier->set_datatype(make_signal_type(method_info));
			return;
		}
		if (ClassDB::has_enum(native, name)) {
			p_identifier->set_datatype(make_native_enum_type(native, name));
			return;
		}
		bool valid = false;
		int int_constant = ClassDB::get_integer_constant(native, name, &valid);
		if (valid) {
			p_identifier->is_constant = true;
			p_identifier->reduced_value = int_constant;
			p_identifier->set_datatype(type_from_variant(int_constant, p_identifier));
			return;
		}
	}
}

void GDScriptAnalyzer::reduce_identifier(GDScriptParser::IdentifierNode *p_identifier, bool can_be_builtin) {
	// TODO: This is opportunity to further infer types.

	// Check if we are inside and enum. This allows enum values to access other elements of the same enum.
	if (current_enum) {
		for (int i = 0; i < current_enum->values.size(); i++) {
			const GDScriptParser::EnumNode::Value &element = current_enum->values[i];
			if (element.identifier->name == p_identifier->name) {
				GDScriptParser::DataType type;
				type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
				type.kind = element.parent_enum->identifier ? GDScriptParser::DataType::ENUM_VALUE : GDScriptParser::DataType::BUILTIN;
				type.builtin_type = Variant::INT;
				type.is_constant = true;
				if (element.parent_enum->identifier) {
					type.enum_type = element.parent_enum->identifier->name;
				}
				p_identifier->set_datatype(type);

				if (element.resolved) {
					p_identifier->is_constant = true;
					p_identifier->reduced_value = element.value;
				} else {
					push_error(R"(Cannot use another enum element before it was declared.)", p_identifier);
				}
				return; // Found anyway.
			}
		}
	}

	bool found_source = false;
	// Check if identifier is local.
	// If that's the case, the declaration already was solved before.
	switch (p_identifier->source) {
		case GDScriptParser::IdentifierNode::FUNCTION_PARAMETER:
			p_identifier->set_datatype(p_identifier->parameter_source->get_datatype());
			found_source = true;
			break;
		case GDScriptParser::IdentifierNode::LOCAL_CONSTANT:
		case GDScriptParser::IdentifierNode::MEMBER_CONSTANT:
			p_identifier->set_datatype(p_identifier->constant_source->get_datatype());
			p_identifier->is_constant = true;
			// TODO: Constant should have a value on the node itself.
			p_identifier->reduced_value = p_identifier->constant_source->initializer->reduced_value;
			found_source = true;
			break;
		case GDScriptParser::IdentifierNode::MEMBER_VARIABLE:
			p_identifier->variable_source->usages++;
			[[fallthrough]];
		case GDScriptParser::IdentifierNode::LOCAL_VARIABLE:
			p_identifier->set_datatype(p_identifier->variable_source->get_datatype());
			found_source = true;
			break;
		case GDScriptParser::IdentifierNode::LOCAL_ITERATOR:
			p_identifier->set_datatype(p_identifier->bind_source->get_datatype());
			found_source = true;
			break;
		case GDScriptParser::IdentifierNode::LOCAL_BIND: {
			GDScriptParser::DataType result = p_identifier->bind_source->get_datatype();
			result.is_constant = true;
			p_identifier->set_datatype(result);
			found_source = true;
		} break;
		case GDScriptParser::IdentifierNode::UNDEFINED_SOURCE:
			break;
	}

	// Not a local, so check members.
	if (!found_source) {
		reduce_identifier_from_base(p_identifier);
		if (p_identifier->source != GDScriptParser::IdentifierNode::UNDEFINED_SOURCE || p_identifier->get_datatype().is_set()) {
			// Found.
			found_source = true;
		}
	}

	if (found_source) {
		// If the identifier is local, check if it's any kind of capture by comparing their source function.
		// Only capture locals and members and enum values. Constants are still accessible from the lambda using the script reference.
		if (p_identifier->source == GDScriptParser::IdentifierNode::UNDEFINED_SOURCE || p_identifier->source == GDScriptParser::IdentifierNode::MEMBER_CONSTANT || lambda_stack.is_empty()) {
			return;
		}

		GDScriptParser::FunctionNode *function_test = lambda_stack.back()->get()->function;
		while (function_test != nullptr && function_test != p_identifier->source_function && function_test->source_lambda != nullptr && !function_test->source_lambda->captures_indices.has(p_identifier->name)) {
			function_test->source_lambda->captures_indices[p_identifier->name] = function_test->source_lambda->captures.size();
			function_test->source_lambda->captures.push_back(p_identifier);
			function_test = function_test->source_lambda->parent_function;
		}
		return;
	}

	StringName name = p_identifier->name;
	p_identifier->source = GDScriptParser::IdentifierNode::UNDEFINED_SOURCE;

	// Check globals. We make an exception for Variant::OBJECT because it's the base class for
	// non-builtin types so we allow doing e.g. Object.new()
	Variant::Type builtin_type = GDScriptParser::get_builtin_type(name);
	if (builtin_type != Variant::OBJECT && builtin_type < Variant::VARIANT_MAX) {
		if (can_be_builtin) {
			p_identifier->set_datatype(make_builtin_meta_type(builtin_type));
			return;
		} else {
			push_error(R"(Builtin type cannot be used as a name on its own.)", p_identifier);
		}
	}

	if (class_exists(name)) {
		p_identifier->set_datatype(make_native_meta_type(name));
		return;
	}

	if (ScriptServer::is_global_class(name)) {
		p_identifier->set_datatype(make_global_class_meta_type(name, p_identifier));
		return;
	}

	// Try singletons.
	// Do this before globals because this might be a singleton loading another one before it's compiled.
	if (ProjectSettings::get_singleton()->has_autoload(name)) {
		const ProjectSettings::AutoloadInfo &autoload = ProjectSettings::get_singleton()->get_autoload(name);
		if (autoload.is_singleton) {
			// Singleton exists, so it's at least a Node.
			GDScriptParser::DataType result;
			result.kind = GDScriptParser::DataType::NATIVE;
			result.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
			if (autoload.path.to_lower().ends_with(GDScriptLanguage::get_singleton()->get_extension())) {
				Ref<GDScriptParserRef> parser = get_parser_for(autoload.path);
				if (parser.is_valid()) {
					Error err = parser->raise_status(GDScriptParserRef::INTERFACE_SOLVED);
					if (err == OK) {
						result = type_from_metatype(parser->get_parser()->head->get_datatype());
					}
				}
			}
			result.is_constant = true;
			p_identifier->set_datatype(result);
			return;
		}
	}

	if (GDScriptLanguage::get_singleton()->get_global_map().has(name)) {
		int idx = GDScriptLanguage::get_singleton()->get_global_map()[name];
		Variant constant = GDScriptLanguage::get_singleton()->get_global_array()[idx];
		p_identifier->set_datatype(type_from_variant(constant, p_identifier));
		p_identifier->is_constant = true;
		p_identifier->reduced_value = constant;
		return;
	}

	if (GDScriptLanguage::get_singleton()->get_named_globals_map().has(name)) {
		Variant constant = GDScriptLanguage::get_singleton()->get_named_globals_map()[name];
		p_identifier->set_datatype(type_from_variant(constant, p_identifier));
		p_identifier->is_constant = true;
		p_identifier->reduced_value = constant;
		return;
	}

	// Not found.
	// Check if it's a builtin function.
	if (GDScriptUtilityFunctions::function_exists(name)) {
		push_error(vformat(R"(Built-in function "%s" cannot be used as an identifier.)", name), p_identifier);
	} else {
		push_error(vformat(R"(Identifier "%s" not declared in the current scope.)", name), p_identifier);
	}
	GDScriptParser::DataType dummy;
	dummy.kind = GDScriptParser::DataType::VARIANT;
	p_identifier->set_datatype(dummy); // Just so type is set to something.
}

void GDScriptAnalyzer::reduce_lambda(GDScriptParser::LambdaNode *p_lambda) {
	// Lambda is always a Callable.
	GDScriptParser::DataType lambda_type;
	lambda_type.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;
	lambda_type.kind = GDScriptParser::DataType::BUILTIN;
	lambda_type.builtin_type = Variant::CALLABLE;
	p_lambda->set_datatype(lambda_type);

	if (p_lambda->function == nullptr) {
		return;
	}

	GDScriptParser::FunctionNode *previous_function = parser->current_function;
	parser->current_function = p_lambda->function;

	lambda_stack.push_back(p_lambda);

	for (int i = 0; i < p_lambda->function->parameters.size(); i++) {
		resolve_parameter(p_lambda->function->parameters[i]);
	}

	resolve_suite(p_lambda->function->body);

	int captures_amount = p_lambda->captures.size();
	if (captures_amount > 0) {
		// Create space for lambda parameters.
		// At the beginning to not mess with optional parameters.
		int param_count = p_lambda->function->parameters.size();
		p_lambda->function->parameters.resize(param_count + captures_amount);
		for (int i = param_count - 1; i >= 0; i--) {
			p_lambda->function->parameters.write[i + captures_amount] = p_lambda->function->parameters[i];
			p_lambda->function->parameters_indices[p_lambda->function->parameters[i]->identifier->name] = i + captures_amount;
		}

		// Add captures as extra parameters at the beginning.
		for (int i = 0; i < p_lambda->captures.size(); i++) {
			GDScriptParser::IdentifierNode *capture = p_lambda->captures[i];
			GDScriptParser::ParameterNode *capture_param = parser->alloc_node<GDScriptParser::ParameterNode>();
			capture_param->identifier = capture;
			capture_param->usages = capture->usages;
			capture_param->set_datatype(capture->get_datatype());

			p_lambda->function->parameters.write[i] = capture_param;
			p_lambda->function->parameters_indices[capture->name] = i;
		}
	}

	lambda_stack.pop_back();
	parser->current_function = previous_function;
}

void GDScriptAnalyzer::reduce_literal(GDScriptParser::LiteralNode *p_literal) {
	p_literal->reduced_value = p_literal->value;
	p_literal->is_constant = true;

	p_literal->set_datatype(type_from_variant(p_literal->reduced_value, p_literal));
}

void GDScriptAnalyzer::reduce_preload(GDScriptParser::PreloadNode *p_preload) {
	if (!p_preload->path) {
		return;
	}

	reduce_expression(p_preload->path);

	if (!p_preload->path->is_constant) {
		push_error("Preloaded path must be a constant string.", p_preload->path);
		return;
	}

	if (p_preload->path->reduced_value.get_type() != Variant::STRING) {
		push_error("Preloaded path must be a constant string.", p_preload->path);
	} else {
		p_preload->resolved_path = p_preload->path->reduced_value;
		// TODO: Save this as script dependency.
		if (p_preload->resolved_path.is_relative_path()) {
			p_preload->resolved_path = parser->script_path.get_base_dir().plus_file(p_preload->resolved_path);
		}
		p_preload->resolved_path = p_preload->resolved_path.simplify_path();
		if (!FileAccess::exists(p_preload->resolved_path)) {
			push_error(vformat(R"(Preload file "%s" does not exist.)", p_preload->resolved_path), p_preload->path);
		} else {
			// TODO: Don't load if validating: use completion cache.
			p_preload->resource = ResourceLoader::load(p_preload->resolved_path);
			if (p_preload->resource.is_null()) {
				push_error(vformat(R"(Could not preload resource file "%s".)", p_preload->resolved_path), p_preload->path);
			}
		}
	}

	p_preload->is_constant = true;
	p_preload->reduced_value = p_preload->resource;
	p_preload->set_datatype(type_from_variant(p_preload->reduced_value, p_preload));
}

void GDScriptAnalyzer::reduce_self(GDScriptParser::SelfNode *p_self) {
	p_self->is_constant = false;
	p_self->set_datatype(type_from_metatype(parser->current_class->get_datatype()));
}

void GDScriptAnalyzer::reduce_subscript(GDScriptParser::SubscriptNode *p_subscript) {
	if (p_subscript->base == nullptr) {
		return;
	}
	if (p_subscript->base->type == GDScriptParser::Node::IDENTIFIER) {
		reduce_identifier(static_cast<GDScriptParser::IdentifierNode *>(p_subscript->base), true);
	} else {
		reduce_expression(p_subscript->base);
	}

	GDScriptParser::DataType result_type;

	if (p_subscript->is_attribute) {
		if (p_subscript->attribute == nullptr) {
			return;
		}
		if (p_subscript->base->is_constant) {
			// Just try to get it.
			bool valid = false;
			Variant value = p_subscript->base->reduced_value.get_named(p_subscript->attribute->name, valid);
			if (!valid) {
				push_error(vformat(R"(Cannot get member "%s" from "%s".)", p_subscript->attribute->name, p_subscript->base->reduced_value), p_subscript->index);
			} else {
				p_subscript->is_constant = true;
				p_subscript->reduced_value = value;
				result_type = type_from_variant(value, p_subscript);
			}
			result_type.kind = GDScriptParser::DataType::VARIANT;
		} else {
			GDScriptParser::DataType base_type = p_subscript->base->get_datatype();

			if (base_type.is_variant() || !base_type.is_hard_type()) {
				result_type.kind = GDScriptParser::DataType::VARIANT;
				mark_node_unsafe(p_subscript);
			} else {
				reduce_identifier_from_base(p_subscript->attribute, &base_type);
				GDScriptParser::DataType attr_type = p_subscript->attribute->get_datatype();
				if (attr_type.is_set()) {
					result_type = attr_type;
					p_subscript->is_constant = p_subscript->attribute->is_constant;
					p_subscript->reduced_value = p_subscript->attribute->reduced_value;
				} else {
					if (base_type.kind == GDScriptParser::DataType::BUILTIN) {
						push_error(vformat(R"(Cannot find member "%s" in base "%s".)", p_subscript->attribute->name, base_type.to_string()), p_subscript->attribute);
#ifdef DEBUG_ENABLED
					} else {
						parser->push_warning(p_subscript, GDScriptWarning::UNSAFE_PROPERTY_ACCESS, p_subscript->attribute->name, base_type.to_string());
#endif
					}
					result_type.kind = GDScriptParser::DataType::VARIANT;
				}
			}
		}
	} else {
		if (p_subscript->index == nullptr) {
			return;
		}
		reduce_expression(p_subscript->index);

		if (p_subscript->base->is_constant && p_subscript->index->is_constant) {
			// Just try to get it.
			bool valid = false;
			Variant value = p_subscript->base->reduced_value.get(p_subscript->index->reduced_value, &valid);
			if (!valid) {
				push_error(vformat(R"(Cannot get index "%s" from "%s".)", p_subscript->index->reduced_value, p_subscript->base->reduced_value), p_subscript->index);
			} else {
				p_subscript->is_constant = true;
				p_subscript->reduced_value = value;
				result_type = type_from_variant(value, p_subscript);
			}
			result_type.kind = GDScriptParser::DataType::VARIANT;
		} else {
			GDScriptParser::DataType base_type = p_subscript->base->get_datatype();
			GDScriptParser::DataType index_type = p_subscript->index->get_datatype();

			if (base_type.is_variant()) {
				result_type.kind = GDScriptParser::DataType::VARIANT;
				mark_node_unsafe(p_subscript);
			} else {
				if (base_type.kind == GDScriptParser::DataType::BUILTIN && !index_type.is_variant()) {
					// Check if indexing is valid.
					bool error = index_type.kind != GDScriptParser::DataType::BUILTIN && base_type.builtin_type != Variant::DICTIONARY;
					if (!error) {
						switch (base_type.builtin_type) {
							// Expect int or real as index.
							case Variant::PACKED_BYTE_ARRAY:
							case Variant::PACKED_COLOR_ARRAY:
							case Variant::PACKED_FLOAT32_ARRAY:
							case Variant::PACKED_FLOAT64_ARRAY:
							case Variant::PACKED_INT32_ARRAY:
							case Variant::PACKED_INT64_ARRAY:
							case Variant::PACKED_STRING_ARRAY:
							case Variant::PACKED_VECTOR2_ARRAY:
							case Variant::PACKED_VECTOR3_ARRAY:
							case Variant::ARRAY:
							case Variant::STRING:
								error = index_type.builtin_type != Variant::INT && index_type.builtin_type != Variant::FLOAT;
								break;
							// Expect String only.
							case Variant::RECT2:
							case Variant::RECT2I:
							case Variant::PLANE:
							case Variant::QUATERNION:
							case Variant::AABB:
							case Variant::OBJECT:
								error = index_type.builtin_type != Variant::STRING;
								break;
							// Expect String or number.
							case Variant::BASIS:
							case Variant::VECTOR2:
							case Variant::VECTOR2I:
							case Variant::VECTOR3:
							case Variant::VECTOR3I:
							case Variant::TRANSFORM2D:
							case Variant::TRANSFORM3D:
								error = index_type.builtin_type != Variant::INT && index_type.builtin_type != Variant::FLOAT &&
										index_type.builtin_type != Variant::STRING;
								break;
							// Expect String or int.
							case Variant::COLOR:
								error = index_type.builtin_type != Variant::INT && index_type.builtin_type != Variant::STRING;
								break;
							// Don't support indexing, but we will check it later.
							case Variant::RID:
							case Variant::BOOL:
							case Variant::CALLABLE:
							case Variant::FLOAT:
							case Variant::INT:
							case Variant::NIL:
							case Variant::NODE_PATH:
							case Variant::SIGNAL:
							case Variant::STRING_NAME:
								break;
							// Here for completeness.
							case Variant::DICTIONARY:
							case Variant::VARIANT_MAX:
								break;
						}

						if (error) {
							push_error(vformat(R"(Invalid index type "%s" for a base of type "%s".)", index_type.to_string(), base_type.to_string()), p_subscript->index);
						}
					}
				} else if (base_type.kind != GDScriptParser::DataType::BUILTIN && !index_type.is_variant()) {
					if (index_type.builtin_type != Variant::STRING && index_type.builtin_type != Variant::STRING_NAME) {
						push_error(vformat(R"(Only String or StringName can be used as index for type "%s", but received a "%s".)", base_type.to_string(), index_type.to_string()), p_subscript->index);
					}
				}

				// Check resulting type if possible.
				result_type.builtin_type = Variant::NIL;
				result_type.kind = GDScriptParser::DataType::BUILTIN;
				result_type.type_source = base_type.is_hard_type() ? GDScriptParser::DataType::ANNOTATED_INFERRED : GDScriptParser::DataType::INFERRED;

				if (base_type.kind != GDScriptParser::DataType::BUILTIN) {
					base_type.builtin_type = Variant::OBJECT;
				}
				switch (base_type.builtin_type) {
					// Can't index at all.
					case Variant::RID:
					case Variant::BOOL:
					case Variant::CALLABLE:
					case Variant::FLOAT:
					case Variant::INT:
					case Variant::NIL:
					case Variant::NODE_PATH:
					case Variant::SIGNAL:
					case Variant::STRING_NAME:
						result_type.kind = GDScriptParser::DataType::VARIANT;
						push_error(vformat(R"(Cannot use subscript operator on a base of type "%s".)", base_type.to_string()), p_subscript->base);
						break;
					// Return int.
					case Variant::PACKED_BYTE_ARRAY:
					case Variant::PACKED_INT32_ARRAY:
					case Variant::PACKED_INT64_ARRAY:
					case Variant::VECTOR2I:
					case Variant::VECTOR3I:
						result_type.builtin_type = Variant::INT;
						break;
					// Return float.
					case Variant::PACKED_FLOAT32_ARRAY:
					case Variant::PACKED_FLOAT64_ARRAY:
					case Variant::VECTOR2:
					case Variant::VECTOR3:
					case Variant::QUATERNION:
						result_type.builtin_type = Variant::FLOAT;
						break;
					// Return Color.
					case Variant::PACKED_COLOR_ARRAY:
						result_type.builtin_type = Variant::COLOR;
						break;
					// Return String.
					case Variant::PACKED_STRING_ARRAY:
					case Variant::STRING:
						result_type.builtin_type = Variant::STRING;
						break;
					// Return Vector2.
					case Variant::PACKED_VECTOR2_ARRAY:
					case Variant::TRANSFORM2D:
					case Variant::RECT2:
						result_type.builtin_type = Variant::VECTOR2;
						break;
					// Return Vector2I.
					case Variant::RECT2I:
						result_type.builtin_type = Variant::VECTOR2I;
						break;
					// Return Vector3.
					case Variant::PACKED_VECTOR3_ARRAY:
					case Variant::AABB:
					case Variant::BASIS:
						result_type.builtin_type = Variant::VECTOR3;
						break;
					// Depends on the index.
					case Variant::TRANSFORM3D:
					case Variant::PLANE:
					case Variant::COLOR:
					case Variant::DICTIONARY:
					case Variant::OBJECT:
						result_type.kind = GDScriptParser::DataType::VARIANT;
						result_type.type_source = GDScriptParser::DataType::UNDETECTED;
						break;
					// Can have an element type.
					case Variant::ARRAY:
						if (base_type.has_container_element_type()) {
							result_type = base_type.get_container_element_type();
							result_type.type_source = base_type.type_source;
						} else {
							result_type.kind = GDScriptParser::DataType::VARIANT;
							result_type.type_source = GDScriptParser::DataType::UNDETECTED;
						}
						break;
					// Here for completeness.
					case Variant::VARIANT_MAX:
						break;
				}
			}
		}
	}

	p_subscript->set_datatype(result_type);
}

void GDScriptAnalyzer::reduce_ternary_op(GDScriptParser::TernaryOpNode *p_ternary_op) {
	reduce_expression(p_ternary_op->condition);
	reduce_expression(p_ternary_op->true_expr);
	reduce_expression(p_ternary_op->false_expr);

	GDScriptParser::DataType result;

	if (p_ternary_op->condition && p_ternary_op->condition->is_constant && p_ternary_op->true_expr->is_constant && p_ternary_op->false_expr && p_ternary_op->false_expr->is_constant) {
		p_ternary_op->is_constant = true;
		if (p_ternary_op->condition->reduced_value.booleanize()) {
			p_ternary_op->reduced_value = p_ternary_op->true_expr->reduced_value;
		} else {
			p_ternary_op->reduced_value = p_ternary_op->false_expr->reduced_value;
		}
	}

	GDScriptParser::DataType true_type;
	if (p_ternary_op->true_expr) {
		true_type = p_ternary_op->true_expr->get_datatype();
	} else {
		true_type.kind = GDScriptParser::DataType::VARIANT;
	}
	GDScriptParser::DataType false_type;
	if (p_ternary_op->false_expr) {
		false_type = p_ternary_op->false_expr->get_datatype();
	} else {
		false_type.kind = GDScriptParser::DataType::VARIANT;
	}

	if (true_type.is_variant() || false_type.is_variant()) {
		result.kind = GDScriptParser::DataType::VARIANT;
	} else {
		result = true_type;
		if (!is_type_compatible(true_type, false_type)) {
			result = false_type;
			if (!is_type_compatible(false_type, true_type)) {
				result.type_source = GDScriptParser::DataType::UNDETECTED;
				result.kind = GDScriptParser::DataType::VARIANT;
#ifdef DEBUG_ENABLED
				parser->push_warning(p_ternary_op, GDScriptWarning::INCOMPATIBLE_TERNARY);
#endif
			}
		}
	}

	p_ternary_op->set_datatype(result);
}

void GDScriptAnalyzer::reduce_unary_op(GDScriptParser::UnaryOpNode *p_unary_op) {
	reduce_expression(p_unary_op->operand);

	GDScriptParser::DataType result;

	if (p_unary_op->operand == nullptr) {
		result.kind = GDScriptParser::DataType::VARIANT;
		p_unary_op->set_datatype(result);
		return;
	}

	if (p_unary_op->operand->is_constant) {
		p_unary_op->is_constant = true;
		p_unary_op->reduced_value = Variant::evaluate(p_unary_op->variant_op, p_unary_op->operand->reduced_value, Variant());
		result = type_from_variant(p_unary_op->reduced_value, p_unary_op);
	} else if (p_unary_op->operand->get_datatype().is_variant()) {
		result.kind = GDScriptParser::DataType::VARIANT;
		mark_node_unsafe(p_unary_op);
	} else {
		bool valid = false;
		result = get_operation_type(p_unary_op->variant_op, p_unary_op->operand->get_datatype(), valid, p_unary_op);

		if (!valid) {
			push_error(vformat(R"(Invalid operand of type "%s" for unary operator "%s".)", p_unary_op->operand->get_datatype().to_string(), Variant::get_operator_name(p_unary_op->variant_op)), p_unary_op->operand);
		}
	}

	p_unary_op->set_datatype(result);
}

void GDScriptAnalyzer::const_fold_array(GDScriptParser::ArrayNode *p_array) {
	bool all_is_constant = true;

	for (int i = 0; i < p_array->elements.size(); i++) {
		GDScriptParser::ExpressionNode *element = p_array->elements[i];
		all_is_constant = all_is_constant && element->is_constant;
		if (!all_is_constant) {
			return;
		}
	}

	Array array;
	array.resize(p_array->elements.size());
	for (int i = 0; i < p_array->elements.size(); i++) {
		array[i] = p_array->elements[i]->reduced_value;
	}
	p_array->is_constant = true;
	p_array->reduced_value = array;
}

void GDScriptAnalyzer::const_fold_dictionary(GDScriptParser::DictionaryNode *p_dictionary) {
	bool all_is_constant = true;

	for (int i = 0; i < p_dictionary->elements.size(); i++) {
		const GDScriptParser::DictionaryNode::Pair &element = p_dictionary->elements[i];
		all_is_constant = all_is_constant && element.key->is_constant && element.value->is_constant;
		if (!all_is_constant) {
			return;
		}
	}

	Dictionary dict;
	for (int i = 0; i < p_dictionary->elements.size(); i++) {
		const GDScriptParser::DictionaryNode::Pair &element = p_dictionary->elements[i];
		dict[element.key->reduced_value] = element.value->reduced_value;
	}
	p_dictionary->is_constant = true;
	p_dictionary->reduced_value = dict;
}

GDScriptParser::DataType GDScriptAnalyzer::type_from_variant(const Variant &p_value, const GDScriptParser::Node *p_source) {
	GDScriptParser::DataType result;
	result.is_constant = true;
	result.kind = GDScriptParser::DataType::BUILTIN;
	result.builtin_type = p_value.get_type();
	result.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT; // Constant has explicit type.

	if (p_value.get_type() == Variant::OBJECT) {
		Object *obj = p_value;
		if (!obj) {
			return GDScriptParser::DataType();
		}
		result.native_type = obj->get_class_name();

		Ref<Script> scr = p_value; // Check if value is a script itself.
		if (scr.is_valid()) {
			result.is_meta_type = true;
		} else {
			result.is_meta_type = false;
			scr = obj->get_script();
		}
		if (scr.is_valid()) {
			if (scr->is_valid()) {
				result.script_type = scr;
				result.script_path = scr->get_path();
				Ref<GDScript> gds = scr;
				if (gds.is_valid()) {
					result.kind = GDScriptParser::DataType::CLASS;
					// This might be an inner class, so we want to get the parser for the root.
					// But still get the inner class from that tree.
					GDScript *current = gds.ptr();
					List<StringName> class_chain;
					while (current->_owner) {
						// Push to front so it's in reverse.
						class_chain.push_front(current->name);
						current = current->_owner;
					}

					Ref<GDScriptParserRef> ref = get_parser_for(current->get_path());
					if (ref.is_null()) {
						push_error("Could not find script in path.", p_source);
						GDScriptParser::DataType error_type;
						error_type.kind = GDScriptParser::DataType::VARIANT;
						return error_type;
					}
					ref->raise_status(GDScriptParserRef::INTERFACE_SOLVED);

					GDScriptParser::ClassNode *found = ref->get_parser()->head;

					// It should be okay to assume this exists, since we have a complete script already.
					for (const StringName &E : class_chain) {
						found = found->get_member(E).m_class;
					}

					result.class_type = found;
					result.script_path = ref->get_parser()->script_path;
				} else {
					result.kind = GDScriptParser::DataType::SCRIPT;
				}
				result.native_type = scr->get_instance_base_type();
			} else {
				push_error(vformat(R"(Constant value uses script from "%s" which is loaded but not compiled.)", scr->get_path()), p_source);
				result.kind = GDScriptParser::DataType::VARIANT;
				result.type_source = GDScriptParser::DataType::UNDETECTED;
				result.is_meta_type = false;
			}
		} else {
			result.kind = GDScriptParser::DataType::NATIVE;
			if (result.native_type == GDScriptNativeClass::get_class_static()) {
				result.is_meta_type = true;
			}
		}
	}

	return result;
}

GDScriptParser::DataType GDScriptAnalyzer::type_from_metatype(const GDScriptParser::DataType &p_meta_type) const {
	GDScriptParser::DataType result = p_meta_type;
	result.is_meta_type = false;
	result.is_constant = false;
	return result;
}

GDScriptParser::DataType GDScriptAnalyzer::type_from_property(const PropertyInfo &p_property) const {
	GDScriptParser::DataType result;
	result.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
	if (p_property.type == Variant::NIL && (p_property.usage & PROPERTY_USAGE_NIL_IS_VARIANT)) {
		// Variant
		result.kind = GDScriptParser::DataType::VARIANT;
		return result;
	}
	result.builtin_type = p_property.type;
	if (p_property.type == Variant::OBJECT) {
		result.kind = GDScriptParser::DataType::NATIVE;
		result.native_type = p_property.class_name == StringName() ? "Object" : p_property.class_name;
	} else {
		result.kind = GDScriptParser::DataType::BUILTIN;
		result.builtin_type = p_property.type;
		if (p_property.type == Variant::ARRAY && p_property.hint == PROPERTY_HINT_ARRAY_TYPE) {
			// Check element type.
			StringName elem_type_name = p_property.hint_string;
			GDScriptParser::DataType elem_type;
			elem_type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;

			Variant::Type elem_builtin_type = GDScriptParser::get_builtin_type(elem_type_name);
			if (elem_builtin_type < Variant::VARIANT_MAX) {
				// Builtin type.
				elem_type.kind = GDScriptParser::DataType::BUILTIN;
				elem_type.builtin_type = elem_builtin_type;
			} else if (class_exists(elem_type_name)) {
				elem_type.kind = GDScriptParser::DataType::NATIVE;
				elem_type.builtin_type = Variant::OBJECT;
				elem_type.native_type = p_property.hint_string;
			} else if (ScriptServer::is_global_class(elem_type_name)) {
				// Just load this as it shouldn't be a GDScript.
				Ref<Script> script = ResourceLoader::load(ScriptServer::get_global_class_path(elem_type_name));
				elem_type.kind = GDScriptParser::DataType::SCRIPT;
				elem_type.builtin_type = Variant::OBJECT;
				elem_type.native_type = script->get_instance_base_type();
				elem_type.script_type = script;
			} else {
				ERR_FAIL_V_MSG(result, "Could not find element type from property hint of a typed array.");
			}
			elem_type.is_constant = false;
			result.set_container_element_type(elem_type);
		}
	}
	return result;
}

bool GDScriptAnalyzer::get_function_signature(GDScriptParser::CallNode *p_source, GDScriptParser::DataType p_base_type, const StringName &p_function, GDScriptParser::DataType &r_return_type, List<GDScriptParser::DataType> &r_par_types, int &r_default_arg_count, bool &r_static, bool &r_vararg) {
	r_static = false;
	r_vararg = false;
	r_default_arg_count = 0;
	StringName function_name = p_function;

	if (p_base_type.kind == GDScriptParser::DataType::BUILTIN) {
		// Construct a base type to get methods.
		Callable::CallError err;
		Variant dummy;
		Variant::construct(p_base_type.builtin_type, dummy, nullptr, 0, err);
		if (err.error != Callable::CallError::CALL_OK) {
			ERR_FAIL_V_MSG(false, "Could not construct base Variant type.");
		}
		List<MethodInfo> methods;
		dummy.get_method_list(&methods);

		for (const MethodInfo &E : methods) {
			if (E.name == p_function) {
				function_signature_from_info(E, r_return_type, r_par_types, r_default_arg_count, r_static, r_vararg);
				r_static = Variant::is_builtin_method_static(p_base_type.builtin_type, function_name);
				return true;
			}
		}

		return false;
	}

	bool is_constructor = (p_base_type.is_meta_type || (p_source->callee && p_source->callee->type == GDScriptParser::Node::IDENTIFIER)) && p_function == StaticCString::create("new");
	if (is_constructor) {
		function_name = "_init";
		r_static = true;
	}

	GDScriptParser::ClassNode *base_class = p_base_type.class_type;
	GDScriptParser::FunctionNode *found_function = nullptr;

	while (found_function == nullptr && base_class != nullptr) {
		if (base_class->has_member(function_name)) {
			if (base_class->get_member(function_name).type != GDScriptParser::ClassNode::Member::FUNCTION) {
				// TODO: If this is Callable it can have a better error message.
				push_error(vformat(R"(Member "%s" is not a function.)", function_name), p_source);
				return false;
			}
			found_function = base_class->get_member(function_name).function;
		}
		base_class = base_class->base_type.class_type;
	}

	if (found_function != nullptr) {
		r_static = is_constructor || found_function->is_static;
		for (int i = 0; i < found_function->parameters.size(); i++) {
			r_par_types.push_back(found_function->parameters[i]->get_datatype());
			if (found_function->parameters[i]->default_value != nullptr) {
				r_default_arg_count++;
			}
		}
		r_return_type = found_function->get_datatype();
		r_return_type.is_meta_type = false;
		r_return_type.is_coroutine = found_function->is_coroutine;

		return true;
	}

	Ref<Script> base_script = p_base_type.script_type;

	while (base_script.is_valid() && base_script->is_valid()) {
		MethodInfo info = base_script->get_method_info(function_name);

		if (!(info == MethodInfo())) {
			return function_signature_from_info(info, r_return_type, r_par_types, r_default_arg_count, r_static, r_vararg);
		}
		base_script = base_script->get_base_script();
	}

	// If the base is a script, it might be trying to access members of the Script class itself.
	if (p_base_type.is_meta_type && !is_constructor && (p_base_type.kind == GDScriptParser::DataType::SCRIPT || p_base_type.kind == GDScriptParser::DataType::CLASS)) {
		MethodInfo info;
		StringName script_class = p_base_type.kind == GDScriptParser::DataType::SCRIPT ? p_base_type.script_type->get_class_name() : StringName(GDScript::get_class_static());

		if (ClassDB::get_method_info(script_class, function_name, &info)) {
			return function_signature_from_info(info, r_return_type, r_par_types, r_default_arg_count, r_static, r_vararg);
		}
	}

	StringName base_native = p_base_type.native_type;
#ifdef DEBUG_ENABLED
	if (base_native != StringName()) {
		// Empty native class might happen in some Script implementations.
		// Just ignore it.
		if (!class_exists(base_native)) {
			ERR_FAIL_V_MSG(false, vformat("Native class %s used in script doesn't exist or isn't exposed.", base_native));
		}
	}
#endif

	if (is_constructor) {
		// Native types always have a default constructor.
		r_return_type = p_base_type;
		r_return_type.type_source = GDScriptParser::DataType::ANNOTATED_EXPLICIT;
		r_return_type.is_meta_type = false;
		return true;
	}

	MethodInfo info;
	if (ClassDB::get_method_info(base_native, function_name, &info)) {
		bool valid = function_signature_from_info(info, r_return_type, r_par_types, r_default_arg_count, r_static, r_vararg);
		if (valid && Engine::get_singleton()->has_singleton(base_native)) {
			r_static = true;
		}
		return valid;
	}

	return false;
}

bool GDScriptAnalyzer::function_signature_from_info(const MethodInfo &p_info, GDScriptParser::DataType &r_return_type, List<GDScriptParser::DataType> &r_par_types, int &r_default_arg_count, bool &r_static, bool &r_vararg) {
	r_return_type = type_from_property(p_info.return_val);
	r_default_arg_count = p_info.default_arguments.size();
	r_vararg = (p_info.flags & METHOD_FLAG_VARARG) != 0;

	for (const PropertyInfo &E : p_info.arguments) {
		r_par_types.push_back(type_from_property(E));
	}
	return true;
}

bool GDScriptAnalyzer::validate_call_arg(const MethodInfo &p_method, const GDScriptParser::CallNode *p_call) {
	List<GDScriptParser::DataType> arg_types;

	for (const PropertyInfo &E : p_method.arguments) {
		arg_types.push_back(type_from_property(E));
	}

	return validate_call_arg(arg_types, p_method.default_arguments.size(), (p_method.flags & METHOD_FLAG_VARARG) != 0, p_call);
}

bool GDScriptAnalyzer::validate_call_arg(const List<GDScriptParser::DataType> &p_par_types, int p_default_args_count, bool p_is_vararg, const GDScriptParser::CallNode *p_call) {
	bool valid = true;

	if (p_call->arguments.size() < p_par_types.size() - p_default_args_count) {
		push_error(vformat(R"*(Too few arguments for "%s()" call. Expected at least %d but received %d.)*", p_call->function_name, p_par_types.size() - p_default_args_count, p_call->arguments.size()), p_call);
		valid = false;
	}
	if (!p_is_vararg && p_call->arguments.size() > p_par_types.size()) {
		push_error(vformat(R"*(Too many arguments for "%s()" call. Expected at most %d but received %d.)*", p_call->function_name, p_par_types.size(), p_call->arguments.size()), p_call->arguments[p_par_types.size()]);
		valid = false;
	}

	for (int i = 0; i < p_call->arguments.size(); i++) {
		if (i >= p_par_types.size()) {
			// Already on vararg place.
			break;
		}
		GDScriptParser::DataType par_type = p_par_types[i];
		GDScriptParser::DataType arg_type = p_call->arguments[i]->get_datatype();

		if (arg_type.is_variant()) {
			// Argument can be anything, so this is unsafe.
			mark_node_unsafe(p_call->arguments[i]);
		} else if (par_type.is_hard_type() && !is_type_compatible(par_type, arg_type, true)) {
			// Supertypes are acceptable for dynamic compliance, but it's unsafe.
			mark_node_unsafe(p_call);
			if (!is_type_compatible(arg_type, par_type)) {
				push_error(vformat(R"*(Invalid argument for "%s()" function: argument %d should be %s but is %s.)*",
								   p_call->function_name, i + 1, par_type.to_string(), arg_type.to_string()),
						p_call->arguments[i]);
				valid = false;
			}
#ifdef DEBUG_ENABLED
		} else {
			if (par_type.kind == GDScriptParser::DataType::BUILTIN && par_type.builtin_type == Variant::INT && arg_type.kind == GDScriptParser::DataType::BUILTIN && arg_type.builtin_type == Variant::FLOAT) {
				parser->push_warning(p_call, GDScriptWarning::NARROWING_CONVERSION, p_call->function_name);
			}
#endif
		}
	}
	return valid;
}

#ifdef DEBUG_ENABLED
bool GDScriptAnalyzer::is_shadowing(GDScriptParser::IdentifierNode *p_local, const String &p_context) {
	const StringName &name = p_local->name;
	GDScriptParser::DataType base = parser->current_class->get_datatype();
	GDScriptParser::ClassNode *base_class = base.class_type;

	{
		List<MethodInfo> gdscript_funcs;
		GDScriptLanguage::get_singleton()->get_public_functions(&gdscript_funcs);

		for (MethodInfo &info : gdscript_funcs) {
			if (info.name == name) {
				parser->push_warning(p_local, GDScriptWarning::SHADOWED_GLOBAL_IDENTIFIER, p_context, name, "built-in function");
				return true;
			}
		}
		if (Variant::has_utility_function(name)) {
			parser->push_warning(p_local, GDScriptWarning::SHADOWED_GLOBAL_IDENTIFIER, p_context, name, "built-in function");
			return true;
		} else if (ClassDB::class_exists(name)) {
			parser->push_warning(p_local, GDScriptWarning::SHADOWED_GLOBAL_IDENTIFIER, p_context, name, "global class");
			return true;
		}
	}

	while (base_class != nullptr) {
		if (base_class->has_member(name)) {
			parser->push_warning(p_local, GDScriptWarning::SHADOWED_VARIABLE, p_context, p_local->name, base_class->get_member(name).get_type_name(), itos(base_class->get_member(name).get_line()));
			return true;
		}
		base_class = base_class->base_type.class_type;
	}

	StringName base_native = base.native_type;

	ERR_FAIL_COND_V_MSG(!class_exists(base_native), false, "Non-existent native base class.");

	StringName parent = base_native;
	while (parent != StringName()) {
		if (ClassDB::has_method(parent, name, true)) {
			parser->push_warning(p_local, GDScriptWarning::SHADOWED_VARIABLE_BASE_CLASS, p_context, p_local->name, "method", parent);
			return true;
		} else if (ClassDB::has_signal(parent, name, true)) {
			parser->push_warning(p_local, GDScriptWarning::SHADOWED_VARIABLE_BASE_CLASS, p_context, p_local->name, "signal", parent);
			return true;
		} else if (ClassDB::has_property(parent, name, true)) {
			parser->push_warning(p_local, GDScriptWarning::SHADOWED_VARIABLE_BASE_CLASS, p_context, p_local->name, "property", parent);
			return true;
		} else if (ClassDB::has_integer_constant(parent, name, true)) {
			parser->push_warning(p_local, GDScriptWarning::SHADOWED_VARIABLE_BASE_CLASS, p_context, p_local->name, "constant", parent);
			return true;
		} else if (ClassDB::has_enum(parent, name, true)) {
			parser->push_warning(p_local, GDScriptWarning::SHADOWED_VARIABLE_BASE_CLASS, p_context, p_local->name, "enum", parent);
			return true;
		}
		parent = ClassDB::get_parent_class(parent);
	}

	return false;
}
#endif

GDScriptParser::DataType GDScriptAnalyzer::get_operation_type(Variant::Operator p_operation, const GDScriptParser::DataType &p_a, bool &r_valid, const GDScriptParser::Node *p_source) {
	// Unary version.
	GDScriptParser::DataType nil_type;
	nil_type.builtin_type = Variant::NIL;
	return get_operation_type(p_operation, p_a, nil_type, r_valid, p_source);
}

GDScriptParser::DataType GDScriptAnalyzer::get_operation_type(Variant::Operator p_operation, const GDScriptParser::DataType &p_a, const GDScriptParser::DataType &p_b, bool &r_valid, const GDScriptParser::Node *p_source) {
	GDScriptParser::DataType result;
	result.kind = GDScriptParser::DataType::VARIANT;

	Variant::Type a_type = p_a.builtin_type;
	Variant::Type b_type = p_b.builtin_type;

	Variant::ValidatedOperatorEvaluator op_eval = Variant::get_validated_operator_evaluator(p_operation, a_type, b_type);

	if (op_eval == nullptr) {
		r_valid = false;
		return result;
	}

	r_valid = true;
	result.type_source = GDScriptParser::DataType::ANNOTATED_INFERRED;

	result.kind = GDScriptParser::DataType::BUILTIN;
	result.builtin_type = Variant::get_operator_return_type(p_operation, a_type, b_type);

	return result;
}

// TODO: Add safe/unsafe return variable (for variant cases)
bool GDScriptAnalyzer::is_type_compatible(const GDScriptParser::DataType &p_target, const GDScriptParser::DataType &p_source, bool p_allow_implicit_conversion) const {
	// These return "true" so it doesn't affect users negatively.
	ERR_FAIL_COND_V_MSG(!p_target.is_set(), true, "Parser bug (please report): Trying to check compatibility of unset target type");
	ERR_FAIL_COND_V_MSG(!p_source.is_set(), true, "Parser bug (please report): Trying to check compatibility of unset value type");

	if (p_target.kind == GDScriptParser::DataType::VARIANT) {
		// Variant can receive anything.
		return true;
	}

	if (p_source.kind == GDScriptParser::DataType::VARIANT) {
		// TODO: This is acceptable but unsafe. Make sure unsafe line is set.
		return true;
	}

	if (p_target.kind == GDScriptParser::DataType::BUILTIN) {
		bool valid = p_source.kind == GDScriptParser::DataType::BUILTIN && p_target.builtin_type == p_source.builtin_type;
		if (!valid && p_allow_implicit_conversion) {
			valid = Variant::can_convert_strict(p_source.builtin_type, p_target.builtin_type);
		}
		if (!valid && p_target.builtin_type == Variant::INT && p_source.kind == GDScriptParser::DataType::ENUM_VALUE) {
			// Enum value is also integer.
			valid = true;
		}
		if (valid && p_target.builtin_type == Variant::ARRAY && p_source.builtin_type == Variant::ARRAY) {
			// Check the element type.
			if (p_target.has_container_element_type()) {
				if (!p_source.has_container_element_type()) {
					// TODO: Maybe this is valid but unsafe?
					// Variant array can't be appended to typed array.
					valid = false;
				} else {
					valid = is_type_compatible(p_target.get_container_element_type(), p_source.get_container_element_type(), false);
				}
			}
		}
		return valid;
	}

	if (p_target.kind == GDScriptParser::DataType::ENUM) {
		if (p_source.kind == GDScriptParser::DataType::BUILTIN && p_source.builtin_type == Variant::INT) {
			return true;
		}
		if (p_source.kind == GDScriptParser::DataType::ENUM) {
			if (p_source.native_type == p_target.native_type) {
				return true;
			}
		}
		if (p_source.kind == GDScriptParser::DataType::ENUM_VALUE) {
			if (p_source.native_type == p_target.native_type && p_target.enum_values.has(p_source.enum_type)) {
				return true;
			}
		}
		return false;
	}

	// From here on the target type is an object, so we have to test polymorphism.

	if (p_source.kind == GDScriptParser::DataType::BUILTIN && p_source.builtin_type == Variant::NIL) {
		// null is acceptable in object.
		return true;
	}

	StringName src_native;
	Ref<Script> src_script;
	const GDScriptParser::ClassNode *src_class = nullptr;

	switch (p_source.kind) {
		case GDScriptParser::DataType::NATIVE:
			if (p_target.kind != GDScriptParser::DataType::NATIVE) {
				// Non-native class cannot be supertype of native.
				return false;
			}
			if (p_source.is_meta_type) {
				src_native = GDScriptNativeClass::get_class_static();
			} else {
				src_native = p_source.native_type;
			}
			break;
		case GDScriptParser::DataType::SCRIPT:
			if (p_target.kind == GDScriptParser::DataType::CLASS) {
				// A script type cannot be a subtype of a GDScript class.
				return false;
			}
			if (p_source.is_meta_type) {
				src_native = p_source.script_type->get_class_name();
			} else {
				src_script = p_source.script_type;
				src_native = src_script->get_instance_base_type();
			}
			break;
		case GDScriptParser::DataType::CLASS:
			if (p_source.is_meta_type) {
				src_native = GDScript::get_class_static();
			} else {
				src_class = p_source.class_type;
				const GDScriptParser::ClassNode *base = src_class;
				while (base->base_type.kind == GDScriptParser::DataType::CLASS) {
					base = base->base_type.class_type;
				}
				src_native = base->base_type.native_type;
				src_script = base->base_type.script_type;
			}
			break;
		case GDScriptParser::DataType::VARIANT:
		case GDScriptParser::DataType::BUILTIN:
		case GDScriptParser::DataType::ENUM:
		case GDScriptParser::DataType::ENUM_VALUE:
		case GDScriptParser::DataType::UNRESOLVED:
			break; // Already solved before.
	}

	switch (p_target.kind) {
		case GDScriptParser::DataType::NATIVE: {
			if (p_target.is_meta_type) {
				return ClassDB::is_parent_class(src_native, GDScriptNativeClass::get_class_static());
			}
			return ClassDB::is_parent_class(src_native, p_target.native_type);
		}
		case GDScriptParser::DataType::SCRIPT:
			if (p_target.is_meta_type) {
				return ClassDB::is_parent_class(src_native, p_target.script_type->get_class_name());
			}
			while (src_script.is_valid()) {
				if (src_script == p_target.script_type) {
					return true;
				}
				src_script = src_script->get_base_script();
			}
			return false;
		case GDScriptParser::DataType::CLASS:
			if (p_target.is_meta_type) {
				return ClassDB::is_parent_class(src_native, GDScript::get_class_static());
			}
			while (src_class != nullptr) {
				if (src_class->fqcn == p_target.class_type->fqcn) {
					return true;
				}
				src_class = src_class->base_type.class_type;
			}
			return false;
		case GDScriptParser::DataType::VARIANT:
		case GDScriptParser::DataType::BUILTIN:
		case GDScriptParser::DataType::ENUM:
		case GDScriptParser::DataType::ENUM_VALUE:
		case GDScriptParser::DataType::UNRESOLVED:
			break; // Already solved before.
	}

	return false;
}

void GDScriptAnalyzer::push_error(const String &p_message, const GDScriptParser::Node *p_origin) {
	mark_node_unsafe(p_origin);
	parser->push_error(p_message, p_origin);
}

void GDScriptAnalyzer::mark_node_unsafe(const GDScriptParser::Node *p_node) {
#ifdef DEBUG_ENABLED
	for (int i = p_node->start_line; i <= p_node->end_line; i++) {
		parser->unsafe_lines.insert(i);
	}
#endif
}

bool GDScriptAnalyzer::class_exists(const StringName &p_class) const {
	return ClassDB::class_exists(p_class) && ClassDB::is_class_exposed(p_class);
}

Ref<GDScriptParserRef> GDScriptAnalyzer::get_parser_for(const String &p_path) {
	Ref<GDScriptParserRef> ref;
	if (depended_parsers.has(p_path)) {
		ref = depended_parsers[p_path];
	} else {
		Error err = OK;
		ref = GDScriptCache::get_parser(p_path, GDScriptParserRef::EMPTY, err, parser->script_path);
		if (ref.is_valid()) {
			depended_parsers[p_path] = ref;
		}
	}

	return ref;
}

Error GDScriptAnalyzer::resolve_inheritance() {
	return resolve_inheritance(parser->head);
}

Error GDScriptAnalyzer::resolve_interface() {
	resolve_class_interface(parser->head);
	return parser->errors.is_empty() ? OK : ERR_PARSE_ERROR;
}

Error GDScriptAnalyzer::resolve_body() {
	resolve_class_body(parser->head);
	return parser->errors.is_empty() ? OK : ERR_PARSE_ERROR;
}

Error GDScriptAnalyzer::resolve_program() {
	resolve_class_interface(parser->head);
	resolve_class_body(parser->head);

	List<String> parser_keys;
	depended_parsers.get_key_list(&parser_keys);
	for (const String &E : parser_keys) {
		if (depended_parsers[E].is_null()) {
			return ERR_PARSE_ERROR;
		}
		depended_parsers[E]->raise_status(GDScriptParserRef::FULLY_SOLVED);
	}
	return parser->errors.is_empty() ? OK : ERR_PARSE_ERROR;
}

Error GDScriptAnalyzer::analyze() {
	parser->errors.clear();
	Error err = resolve_inheritance(parser->head);
	if (err) {
		return err;
	}
	return resolve_program();
}

GDScriptAnalyzer::GDScriptAnalyzer(GDScriptParser *p_parser) {
	parser = p_parser;
}
