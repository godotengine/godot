/**************************************************************************/
/*  gdscript_docgen.cpp                                                   */
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

#include "gdscript_docgen.h"
#include "../gdscript.h"

using GDP = GDScriptParser;
using GDType = GDP::DataType;

static String _get_script_path(const String &p_path) {
	return vformat(R"("%s")", p_path.get_slice("://", 1));
}

static String _get_class_name(const GDP::ClassNode &p_class) {
	const GDP::ClassNode *curr_class = &p_class;
	if (!curr_class->identifier) { // All inner classes have a identifier, so this is the outer class
		return _get_script_path(curr_class->fqcn);
	}

	String full_name = curr_class->identifier->name;
	while (curr_class->outer) {
		curr_class = curr_class->outer;
		if (!curr_class->identifier) { // All inner classes have a identifier, so this is the outer class
			return vformat("%s.%s", _get_script_path(curr_class->fqcn), full_name);
		}
		full_name = vformat("%s.%s", curr_class->identifier->name, full_name);
	}
	return full_name;
}

static PropertyInfo _property_info_from_datatype(const GDType &p_type) {
	PropertyInfo pi;
	pi.type = p_type.builtin_type;
	if (p_type.kind == GDType::CLASS) {
		pi.class_name = _get_class_name(*p_type.class_type);
	} else if (p_type.kind == GDType::ENUM && p_type.enum_type != StringName()) {
		pi.type = Variant::INT; // Only int types are recognized as enums by the EditorHelp
		pi.usage |= PROPERTY_USAGE_CLASS_IS_ENUM;
		// Replace :: from enum's use of fully qualified class names with regular .
		pi.class_name = String(p_type.native_type).replace("::", ".");
	} else if (p_type.kind == GDType::NATIVE) {
		pi.class_name = p_type.native_type;
	}
	return pi;
}

void GDScriptDocGen::generate_docs(GDScript *p_script, const GDP::ClassNode *p_class) {
	p_script->_clear_doc();

	DocData::ClassDoc &doc = p_script->doc;

	doc.script_path = _get_script_path(p_script->get_script_path());
	if (p_script->name.is_empty()) {
		doc.name = doc.script_path;
	} else {
		doc.name = p_script->name;
	}

	if (p_script->_owner) {
		doc.name = p_script->_owner->doc.name + "." + doc.name;
		doc.script_path = doc.script_path + "." + doc.name;
	}

	doc.is_script_doc = true;

	if (p_script->base.is_valid() && p_script->base->is_valid()) {
		if (!p_script->base->doc.name.is_empty()) {
			doc.inherits = p_script->base->doc.name;
		} else {
			doc.inherits = p_script->base->get_instance_base_type();
		}
	} else if (p_script->native.is_valid()) {
		doc.inherits = p_script->native->get_name();
	}

	doc.brief_description = p_class->doc_brief_description;
	doc.description = p_class->doc_description;
	for (const Pair<String, String> &p : p_class->doc_tutorials) {
		DocData::TutorialDoc td;
		td.title = p.first;
		td.link = p.second;
		doc.tutorials.append(td);
	}

	for (const GDP::ClassNode::Member &member : p_class->members) {
		switch (member.type) {
			case GDP::ClassNode::Member::CLASS: {
				const GDP::ClassNode *inner_class = member.m_class;
				const StringName &class_name = inner_class->identifier->name;

				p_script->member_lines[class_name] = inner_class->start_line;

				// Recursively generate inner class docs
				// Needs inner GDScripts to exist: previously generated in GDScriptCompiler::make_scripts()
				GDScriptDocGen::generate_docs(*p_script->subclasses[class_name], inner_class);
			} break;

			case GDP::ClassNode::Member::CONSTANT: {
				const GDP::ConstantNode *m_const = member.constant;
				const StringName &const_name = member.constant->identifier->name;

				p_script->member_lines[const_name] = m_const->start_line;

				DocData::ConstantDoc const_doc;
				DocData::constant_doc_from_variant(const_doc, const_name, m_const->initializer->reduced_value, m_const->doc_description);
				doc.constants.push_back(const_doc);
			} break;

			case GDP::ClassNode::Member::FUNCTION: {
				const GDP::FunctionNode *m_func = member.function;
				const StringName &func_name = m_func->identifier->name;

				p_script->member_lines[func_name] = m_func->start_line;

				MethodInfo mi;
				mi.name = func_name;

				if (m_func->return_type) {
					mi.return_val = _property_info_from_datatype(m_func->return_type->get_datatype());
				}
				for (const GDScriptParser::ParameterNode *p : m_func->parameters) {
					PropertyInfo pi = _property_info_from_datatype(p->get_datatype());
					pi.name = p->identifier->name;
					mi.arguments.push_back(pi);
				}

				DocData::MethodDoc method_doc;
				DocData::method_doc_from_methodinfo(method_doc, mi, m_func->doc_description);
				doc.methods.push_back(method_doc);
			} break;

			case GDP::ClassNode::Member::SIGNAL: {
				const GDP::SignalNode *m_signal = member.signal;
				const StringName &signal_name = m_signal->identifier->name;

				p_script->member_lines[signal_name] = m_signal->start_line;

				MethodInfo mi;
				mi.name = signal_name;
				for (const GDScriptParser::ParameterNode *p : m_signal->parameters) {
					PropertyInfo pi = _property_info_from_datatype(p->get_datatype());
					pi.name = p->identifier->name;
					mi.arguments.push_back(pi);
				}

				DocData::MethodDoc signal_doc;
				DocData::signal_doc_from_methodinfo(signal_doc, mi, m_signal->doc_description);
				doc.signals.push_back(signal_doc);
			} break;

			case GDP::ClassNode::Member::VARIABLE: {
				const GDP::VariableNode *m_var = member.variable;
				const StringName &var_name = m_var->identifier->name;

				p_script->member_lines[var_name] = m_var->start_line;

				DocData::PropertyDoc prop_doc;

				prop_doc.name = var_name;
				prop_doc.description = m_var->doc_description;

				GDType dt = m_var->get_datatype();
				switch (dt.kind) {
					case GDType::CLASS:
						prop_doc.type = _get_class_name(*dt.class_type);
						break;
					case GDType::VARIANT:
						prop_doc.type = "Variant";
						break;
					case GDType::ENUM:
						prop_doc.type = Variant::get_type_name(dt.builtin_type);
						// Replace :: from enum's use of fully qualified class names with regular .
						prop_doc.enumeration = String(dt.native_type).replace("::", ".");
						break;
					case GDType::NATIVE:;
						prop_doc.type = dt.native_type;
						break;
					case GDType::BUILTIN:
						prop_doc.type = Variant::get_type_name(dt.builtin_type);
						break;
					default:
						// SCRIPT: can be preload()'d and perhaps used as types directly?
						// RESOLVING & UNRESOLVED should never happen since docgen requires analyzing w/o errors
						break;
				}

				if (m_var->property == GDP::VariableNode::PROP_SETGET) {
					if (m_var->setter_pointer != nullptr) {
						prop_doc.setter = m_var->setter_pointer->name;
					}
					if (m_var->getter_pointer != nullptr) {
						prop_doc.getter = m_var->getter_pointer->name;
					}
				}

				if (m_var->initializer && m_var->initializer->is_constant) {
					prop_doc.default_value = m_var->initializer->reduced_value.get_construct_string().replace("\n", "");
				}

				prop_doc.overridden = false;

				doc.properties.push_back(prop_doc);
			} break;

			case GDP::ClassNode::Member::ENUM: {
				const GDP::EnumNode *m_enum = member.m_enum;
				StringName name = m_enum->identifier->name;

				p_script->member_lines[name] = m_enum->start_line;

				for (const GDP::EnumNode::Value &val : m_enum->values) {
					DocData::ConstantDoc const_doc;
					const_doc.name = val.identifier->name;
					const_doc.value = String(Variant(val.value));
					const_doc.is_value_valid = true;
					const_doc.description = val.doc_description;
					const_doc.enumeration = name;

					doc.enums[const_doc.name] = const_doc.description;
					doc.constants.push_back(const_doc);
				}

			} break;

			case GDP::ClassNode::Member::ENUM_VALUE: {
				const GDP::EnumNode::Value &m_enum_val = member.enum_value;
				const StringName &name = m_enum_val.identifier->name;

				p_script->member_lines[name] = m_enum_val.identifier->start_line;

				DocData::ConstantDoc constant_doc;
				constant_doc.enumeration = "@unnamed_enums";
				DocData::constant_doc_from_variant(constant_doc, name, m_enum_val.value, m_enum_val.doc_description);
				doc.constants.push_back(constant_doc);
			} break;
			case GDP::ClassNode::Member::GROUP:
			case GDP::ClassNode::Member::UNDEFINED:
			default:
				break;
		}
	}

	// Add doc to the outer-most class.
	p_script->_add_doc(doc);
}
