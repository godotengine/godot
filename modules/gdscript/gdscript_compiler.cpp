/*************************************************************************/
/*  gdscript_compiler.cpp                                                */
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

#include "gdscript_compiler.h"

#include "gdscript.h"
#include "gdscript_byte_codegen.h"
#include "gdscript_cache.h"
#include "gdscript_utility_functions.h"

#include "core/config/project_settings.h"

bool GDScriptCompiler::_is_class_member_property(CodeGen &codegen, const StringName &p_name) {
	if (codegen.function_node && codegen.function_node->is_static) {
		return false;
	}

	if (codegen.locals.has(p_name)) {
		return false; //shadowed
	}

	return _is_class_member_property(codegen.script, p_name);
}

bool GDScriptCompiler::_is_class_member_property(GDScript *owner, const StringName &p_name) {
	GDScript *scr = owner;
	GDScriptNativeClass *nc = nullptr;
	while (scr) {
		if (scr->native.is_valid()) {
			nc = scr->native.ptr();
		}
		scr = scr->_base;
	}

	ERR_FAIL_COND_V(!nc, false);

	return ClassDB::has_property(nc->get_name(), p_name);
}

void GDScriptCompiler::_set_error(const String &p_error, const GDScriptParser::Node *p_node) {
	if (!error.is_empty()) {
		return;
	}

	error = p_error;
	if (p_node) {
		err_line = p_node->start_line;
		err_column = p_node->leftmost_column;
	} else {
		err_line = 0;
		err_column = 0;
	}
}

GDScriptDataType GDScriptCompiler::_gdtype_from_datatype(const GDScriptParser::DataType &p_datatype, GDScript *p_owner) const {
	if (!p_datatype.is_set() || !p_datatype.is_hard_type()) {
		return GDScriptDataType();
	}

	GDScriptDataType result;
	result.has_type = true;

	switch (p_datatype.kind) {
		case GDScriptParser::DataType::VARIANT: {
			result.has_type = false;
		} break;
		case GDScriptParser::DataType::BUILTIN: {
			result.kind = GDScriptDataType::BUILTIN;
			result.builtin_type = p_datatype.builtin_type;
		} break;
		case GDScriptParser::DataType::NATIVE: {
			result.kind = GDScriptDataType::NATIVE;
			result.native_type = p_datatype.native_type;
		} break;
		case GDScriptParser::DataType::SCRIPT: {
			result.kind = GDScriptDataType::SCRIPT;
			result.script_type_ref = Ref<Script>(p_datatype.script_type);
			result.script_type = result.script_type_ref.ptr();
			result.native_type = result.script_type->get_instance_base_type();
		} break;
		case GDScriptParser::DataType::CLASS: {
			// Locate class by constructing the path to it and following that path
			GDScriptParser::ClassNode *class_type = p_datatype.class_type;
			if (class_type) {
				const bool is_inner_by_path = (!main_script->path.is_empty()) && (class_type->fqcn.split("::")[0] == main_script->path);
				const bool is_inner_by_name = (!main_script->name.is_empty()) && (class_type->fqcn.split("::")[0] == main_script->name);
				if (is_inner_by_path || is_inner_by_name) {
					// Local class.
					List<StringName> names;
					while (class_type->outer) {
						names.push_back(class_type->identifier->name);
						class_type = class_type->outer;
					}

					Ref<GDScript> script = Ref<GDScript>(main_script);
					while (names.back()) {
						if (!script->subclasses.has(names.back()->get())) {
							ERR_PRINT("Parser bug: Cannot locate datatype class.");
							result.has_type = false;
							return GDScriptDataType();
						}
						script = script->subclasses[names.back()->get()];
						names.pop_back();
					}
					result.kind = GDScriptDataType::GDSCRIPT;
					result.script_type = script.ptr();
					result.native_type = script->get_instance_base_type();
				} else {
					result.kind = GDScriptDataType::GDSCRIPT;
					result.script_type_ref = GDScriptCache::get_shallow_script(p_datatype.script_path, main_script->path);
					result.script_type = result.script_type_ref.ptr();
					result.native_type = p_datatype.native_type;
				}
			}
		} break;
		case GDScriptParser::DataType::ENUM:
		case GDScriptParser::DataType::ENUM_VALUE:
			result.has_type = true;
			result.kind = GDScriptDataType::BUILTIN;
			result.builtin_type = Variant::INT;
			break;
		case GDScriptParser::DataType::UNRESOLVED: {
			ERR_PRINT("Parser bug: converting unresolved type.");
			return GDScriptDataType();
		}
	}

	if (p_datatype.has_container_element_type()) {
		result.set_container_element_type(_gdtype_from_datatype(p_datatype.get_container_element_type()));
	}

	// Only hold strong reference to the script if it's not the owner of the
	// element qualified with this type, to avoid cyclic references (leaks).
	if (result.script_type && result.script_type == p_owner) {
		result.script_type_ref = Ref<Script>();
	}

	return result;
}

static bool _is_exact_type(const PropertyInfo &p_par_type, const GDScriptDataType &p_arg_type) {
	if (!p_arg_type.has_type) {
		return false;
	}
	if (p_par_type.type == Variant::NIL) {
		return false;
	}
	if (p_par_type.type == Variant::OBJECT) {
		if (p_arg_type.kind == GDScriptDataType::BUILTIN) {
			return false;
		}
		StringName class_name;
		if (p_arg_type.kind == GDScriptDataType::NATIVE) {
			class_name = p_arg_type.native_type;
		} else {
			class_name = p_arg_type.native_type == StringName() ? p_arg_type.script_type->get_instance_base_type() : p_arg_type.native_type;
		}
		return p_par_type.class_name == class_name || ClassDB::is_parent_class(class_name, p_par_type.class_name);
	} else {
		if (p_arg_type.kind != GDScriptDataType::BUILTIN) {
			return false;
		}
		return p_par_type.type == p_arg_type.builtin_type;
	}
}

static bool _have_exact_arguments(const MethodBind *p_method, const Vector<GDScriptCodeGenerator::Address> &p_arguments) {
	if (p_method->get_argument_count() != p_arguments.size()) {
		// ptrcall won't work with default arguments.
		return false;
	}
	MethodInfo info;
	ClassDB::get_method_info(p_method->get_instance_class(), p_method->get_name(), &info);
	for (int i = 0; i < p_arguments.size(); i++) {
		const PropertyInfo &prop = info.arguments[i];
		if (!_is_exact_type(prop, p_arguments[i].type)) {
			return false;
		}
	}
	return true;
}

GDScriptCodeGenerator::Address GDScriptCompiler::_parse_expression(CodeGen &codegen, Error &r_error, const GDScriptParser::ExpressionNode *p_expression, bool p_root, bool p_initializer, const GDScriptCodeGenerator::Address &p_index_addr) {
	if (p_expression->is_constant) {
		return codegen.add_constant(p_expression->reduced_value);
	}

	GDScriptCodeGenerator *gen = codegen.generator;

	switch (p_expression->type) {
		case GDScriptParser::Node::IDENTIFIER: {
			// Look for identifiers in current scope.
			const GDScriptParser::IdentifierNode *in = static_cast<const GDScriptParser::IdentifierNode *>(p_expression);

			StringName identifier = in->name;

			// Try function parameters.
			if (codegen.parameters.has(identifier)) {
				return codegen.parameters[identifier];
			}

			// Try local variables and constants.
			if (!p_initializer && codegen.locals.has(identifier)) {
				return codegen.locals[identifier];
			}

			// Try class members.
			if (_is_class_member_property(codegen, identifier)) {
				// Get property.
				GDScriptCodeGenerator::Address temp = codegen.add_temporary(); // TODO: Could get the type of the class member here.
				gen->write_get_member(temp, identifier);
				return temp;
			}

			// Try members.
			if (!codegen.function_node || !codegen.function_node->is_static) {
				// Try member variables.
				if (codegen.script->member_indices.has(identifier)) {
					if (codegen.script->member_indices[identifier].getter != StringName() && codegen.script->member_indices[identifier].getter != codegen.function_name) {
						// Perform getter.
						GDScriptCodeGenerator::Address temp = codegen.add_temporary();
						Vector<GDScriptCodeGenerator::Address> args; // No argument needed.
						gen->write_call_self(temp, codegen.script->member_indices[identifier].getter, args);
						return temp;
					} else {
						// No getter or inside getter: direct member access.,
						int idx = codegen.script->member_indices[identifier].index;
						return GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::MEMBER, idx, codegen.script->get_member_type(identifier));
					}
				}
			}

			// Try class constants.
			{
				GDScript *owner = codegen.script;
				while (owner) {
					GDScript *scr = owner;
					GDScriptNativeClass *nc = nullptr;
					while (scr) {
						if (scr->constants.has(identifier)) {
							return codegen.add_constant(scr->constants[identifier]); // TODO: Get type here.
						}
						if (scr->native.is_valid()) {
							nc = scr->native.ptr();
						}
						scr = scr->_base;
					}

					// Class C++ integer constant.
					if (nc) {
						bool success = false;
						int constant = ClassDB::get_integer_constant(nc->get_name(), identifier, &success);
						if (success) {
							return codegen.add_constant(constant);
						}
					}

					owner = owner->_owner;
				}
			}

			// Try signals and methods (can be made callables).
			{
				if (codegen.class_node->members_indices.has(identifier)) {
					const GDScriptParser::ClassNode::Member &member = codegen.class_node->members[codegen.class_node->members_indices[identifier]];
					if (member.type == GDScriptParser::ClassNode::Member::FUNCTION || member.type == GDScriptParser::ClassNode::Member::SIGNAL) {
						// Get like it was a property.
						GDScriptCodeGenerator::Address temp = codegen.add_temporary(); // TODO: Get type here.
						GDScriptCodeGenerator::Address self(GDScriptCodeGenerator::Address::SELF);

						gen->write_get_named(temp, identifier, self);
						return temp;
					}
				}

				// Try in native base.
				GDScript *scr = codegen.script;
				GDScriptNativeClass *nc = nullptr;
				while (scr) {
					if (scr->native.is_valid()) {
						nc = scr->native.ptr();
					}
					scr = scr->_base;
				}

				if (nc && (ClassDB::has_signal(nc->get_name(), identifier) || ClassDB::has_method(nc->get_name(), identifier))) {
					// Get like it was a property.
					GDScriptCodeGenerator::Address temp = codegen.add_temporary(); // TODO: Get type here.
					GDScriptCodeGenerator::Address self(GDScriptCodeGenerator::Address::SELF);

					gen->write_get_named(temp, identifier, self);
					return temp;
				}
			}

			// Try globals.
			if (GDScriptLanguage::get_singleton()->get_global_map().has(identifier)) {
				// If it's an autoload singleton, we postpone to load it at runtime.
				// This is so one autoload doesn't try to load another before it's compiled.
				OrderedHashMap<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();
				if (autoloads.has(identifier) && autoloads[identifier].is_singleton) {
					GDScriptCodeGenerator::Address global = codegen.add_temporary(_gdtype_from_datatype(in->get_datatype()));
					int idx = GDScriptLanguage::get_singleton()->get_global_map()[identifier];
					gen->write_store_global(global, idx);
					return global;
				} else {
					int idx = GDScriptLanguage::get_singleton()->get_global_map()[identifier];
					Variant global = GDScriptLanguage::get_singleton()->get_global_array()[idx];
					return codegen.add_constant(global);
				}
			}

			// Try global classes.
			if (ScriptServer::is_global_class(identifier)) {
				const GDScriptParser::ClassNode *class_node = codegen.class_node;
				while (class_node->outer) {
					class_node = class_node->outer;
				}

				RES res;

				if (class_node->identifier && class_node->identifier->name == identifier) {
					res = Ref<GDScript>(main_script);
				} else {
					res = ResourceLoader::load(ScriptServer::get_global_class_path(identifier));
					if (res.is_null()) {
						_set_error("Can't load global class " + String(identifier) + ", cyclic reference?", p_expression);
						r_error = ERR_COMPILATION_FAILED;
						return GDScriptCodeGenerator::Address();
					}
				}

				return codegen.add_constant(res);
			}

#ifdef TOOLS_ENABLED
			if (GDScriptLanguage::get_singleton()->get_named_globals_map().has(identifier)) {
				GDScriptCodeGenerator::Address global = codegen.add_temporary(); // TODO: Get type.
				gen->write_store_named_global(global, identifier);
				return global;
			}
#endif

			// Not found, error.
			_set_error("Identifier not found: " + String(identifier), p_expression);
			r_error = ERR_COMPILATION_FAILED;
			return GDScriptCodeGenerator::Address();
		} break;
		case GDScriptParser::Node::LITERAL: {
			// Return constant.
			const GDScriptParser::LiteralNode *cn = static_cast<const GDScriptParser::LiteralNode *>(p_expression);

			return codegen.add_constant(cn->value);
		} break;
		case GDScriptParser::Node::SELF: {
			//return constant
			if (codegen.function_node && codegen.function_node->is_static) {
				_set_error("'self' not present in static function!", p_expression);
				r_error = ERR_COMPILATION_FAILED;
				return GDScriptCodeGenerator::Address();
			}
			return GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::SELF);
		} break;
		case GDScriptParser::Node::ARRAY: {
			const GDScriptParser::ArrayNode *an = static_cast<const GDScriptParser::ArrayNode *>(p_expression);
			Vector<GDScriptCodeGenerator::Address> values;

			// Create the result temporary first since it's the last to be killed.
			GDScriptDataType array_type = _gdtype_from_datatype(an->get_datatype());
			GDScriptCodeGenerator::Address result = codegen.add_temporary(array_type);

			for (int i = 0; i < an->elements.size(); i++) {
				GDScriptCodeGenerator::Address val = _parse_expression(codegen, r_error, an->elements[i]);
				if (r_error) {
					return GDScriptCodeGenerator::Address();
				}
				values.push_back(val);
			}

			if (array_type.has_container_element_type()) {
				gen->write_construct_typed_array(result, array_type.get_container_element_type(), values);
			} else {
				gen->write_construct_array(result, values);
			}

			for (int i = 0; i < values.size(); i++) {
				if (values[i].mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary();
				}
			}

			return result;
		} break;
		case GDScriptParser::Node::DICTIONARY: {
			const GDScriptParser::DictionaryNode *dn = static_cast<const GDScriptParser::DictionaryNode *>(p_expression);
			Vector<GDScriptCodeGenerator::Address> elements;

			// Create the result temporary first since it's the last to be killed.
			GDScriptDataType dict_type;
			dict_type.has_type = true;
			dict_type.kind = GDScriptDataType::BUILTIN;
			dict_type.builtin_type = Variant::DICTIONARY;
			GDScriptCodeGenerator::Address result = codegen.add_temporary(dict_type);

			for (int i = 0; i < dn->elements.size(); i++) {
				// Key.
				GDScriptCodeGenerator::Address element;
				switch (dn->style) {
					case GDScriptParser::DictionaryNode::PYTHON_DICT:
						// Python-style: key is any expression.
						element = _parse_expression(codegen, r_error, dn->elements[i].key);
						if (r_error) {
							return GDScriptCodeGenerator::Address();
						}
						break;
					case GDScriptParser::DictionaryNode::LUA_TABLE:
						// Lua-style: key is an identifier interpreted as StringName.
						StringName key = dn->elements[i].key->reduced_value.operator StringName();
						element = codegen.add_constant(key);
						break;
				}

				elements.push_back(element);

				element = _parse_expression(codegen, r_error, dn->elements[i].value);
				if (r_error) {
					return GDScriptCodeGenerator::Address();
				}

				elements.push_back(element);
			}

			gen->write_construct_dictionary(result, elements);

			for (int i = 0; i < elements.size(); i++) {
				if (elements[i].mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary();
				}
			}

			return result;
		} break;
		case GDScriptParser::Node::CAST: {
			const GDScriptParser::CastNode *cn = static_cast<const GDScriptParser::CastNode *>(p_expression);
			GDScriptDataType cast_type = _gdtype_from_datatype(cn->cast_type->get_datatype());

			// Create temporary for result first since it will be deleted last.
			GDScriptCodeGenerator::Address result = codegen.add_temporary(cast_type);

			GDScriptCodeGenerator::Address source = _parse_expression(codegen, r_error, cn->operand);

			gen->write_cast(result, source, cast_type);

			if (source.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				gen->pop_temporary();
			}

			return result;
		} break;
		case GDScriptParser::Node::CALL: {
			const GDScriptParser::CallNode *call = static_cast<const GDScriptParser::CallNode *>(p_expression);
			GDScriptDataType type = _gdtype_from_datatype(call->get_datatype());
			GDScriptCodeGenerator::Address result = codegen.add_temporary(type);
			GDScriptCodeGenerator::Address nil = GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::NIL);

			GDScriptCodeGenerator::Address return_addr = p_root ? nil : result;

			Vector<GDScriptCodeGenerator::Address> arguments;
			for (int i = 0; i < call->arguments.size(); i++) {
				GDScriptCodeGenerator::Address arg = _parse_expression(codegen, r_error, call->arguments[i]);
				if (r_error) {
					return GDScriptCodeGenerator::Address();
				}
				arguments.push_back(arg);
			}

			if (!call->is_super && call->callee->type == GDScriptParser::Node::IDENTIFIER && GDScriptParser::get_builtin_type(call->function_name) != Variant::VARIANT_MAX) {
				// Construct a built-in type.
				Variant::Type vtype = GDScriptParser::get_builtin_type(static_cast<GDScriptParser::IdentifierNode *>(call->callee)->name);

				gen->write_construct(result, vtype, arguments);
			} else if (!call->is_super && call->callee->type == GDScriptParser::Node::IDENTIFIER && Variant::has_utility_function(call->function_name)) {
				// Variant utility function.
				gen->write_call_utility(result, call->function_name, arguments);
			} else if (!call->is_super && call->callee->type == GDScriptParser::Node::IDENTIFIER && GDScriptUtilityFunctions::function_exists(call->function_name)) {
				// GDScript utility function.
				gen->write_call_gdscript_utility(result, GDScriptUtilityFunctions::get_function(call->function_name), arguments);
			} else {
				// Regular function.
				const GDScriptParser::ExpressionNode *callee = call->callee;

				if (call->is_super) {
					// Super call.
					gen->write_super_call(result, call->function_name, arguments);
				} else {
					if (callee->type == GDScriptParser::Node::IDENTIFIER) {
						// Self function call.
						if (ClassDB::has_method(codegen.script->native->get_name(), call->function_name)) {
							// Native method, use faster path.
							GDScriptCodeGenerator::Address self;
							self.mode = GDScriptCodeGenerator::Address::SELF;
							MethodBind *method = ClassDB::get_method(codegen.script->native->get_name(), call->function_name);

							if (_have_exact_arguments(method, arguments)) {
								// Exact arguments, use ptrcall.
								gen->write_call_ptrcall(result, self, method, arguments);
							} else {
								// Not exact arguments, but still can use method bind call.
								gen->write_call_method_bind(result, self, method, arguments);
							}
						} else if ((codegen.function_node && codegen.function_node->is_static) || call->function_name == "new") {
							GDScriptCodeGenerator::Address self;
							self.mode = GDScriptCodeGenerator::Address::CLASS;
							if (within_await) {
								gen->write_call_async(result, self, call->function_name, arguments);
							} else {
								gen->write_call(return_addr, self, call->function_name, arguments);
							}
						} else {
							if (within_await) {
								gen->write_call_self_async(result, call->function_name, arguments);
							} else {
								gen->write_call_self(return_addr, call->function_name, arguments);
							}
						}
					} else if (callee->type == GDScriptParser::Node::SUBSCRIPT) {
						const GDScriptParser::SubscriptNode *subscript = static_cast<const GDScriptParser::SubscriptNode *>(call->callee);

						if (subscript->is_attribute) {
							// May be static built-in method call.
							if (!call->is_super && subscript->base->type == GDScriptParser::Node::IDENTIFIER && GDScriptParser::get_builtin_type(static_cast<GDScriptParser::IdentifierNode *>(subscript->base)->name) < Variant::VARIANT_MAX) {
								gen->write_call_builtin_type_static(result, GDScriptParser::get_builtin_type(static_cast<GDScriptParser::IdentifierNode *>(subscript->base)->name), subscript->attribute->name, arguments);
							} else {
								GDScriptCodeGenerator::Address base = _parse_expression(codegen, r_error, subscript->base);
								if (r_error) {
									return GDScriptCodeGenerator::Address();
								}
								if (within_await) {
									gen->write_call_async(result, base, call->function_name, arguments);
								} else if (base.type.has_type && base.type.kind != GDScriptDataType::BUILTIN) {
									// Native method, use faster path.
									StringName class_name;
									if (base.type.kind == GDScriptDataType::NATIVE) {
										class_name = base.type.native_type;
									} else {
										class_name = base.type.native_type == StringName() ? base.type.script_type->get_instance_base_type() : base.type.native_type;
									}
									if (ClassDB::class_exists(class_name) && ClassDB::has_method(class_name, call->function_name)) {
										MethodBind *method = ClassDB::get_method(class_name, call->function_name);
										if (_have_exact_arguments(method, arguments)) {
											// Exact arguments, use ptrcall.
											gen->write_call_ptrcall(result, base, method, arguments);
										} else {
											// Not exact arguments, but still can use method bind call.
											gen->write_call_method_bind(result, base, method, arguments);
										}
									} else {
										gen->write_call(return_addr, base, call->function_name, arguments);
									}
								} else if (base.type.has_type && base.type.kind == GDScriptDataType::BUILTIN) {
									gen->write_call_builtin_type(result, base, base.type.builtin_type, call->function_name, arguments);
								} else {
									gen->write_call(return_addr, base, call->function_name, arguments);
								}
								if (base.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
									gen->pop_temporary();
								}
							}
						} else {
							_set_error("Cannot call something that isn't a function.", call->callee);
							r_error = ERR_COMPILATION_FAILED;
							return GDScriptCodeGenerator::Address();
						}
					} else {
						r_error = ERR_COMPILATION_FAILED;
						return GDScriptCodeGenerator::Address();
					}
				}
			}

			for (int i = 0; i < arguments.size(); i++) {
				if (arguments[i].mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary();
				}
			}
			return result;
		} break;
		case GDScriptParser::Node::GET_NODE: {
			const GDScriptParser::GetNodeNode *get_node = static_cast<const GDScriptParser::GetNodeNode *>(p_expression);

			String node_name;
			if (get_node->string != nullptr) {
				node_name += String(get_node->string->value);
			} else {
				for (int i = 0; i < get_node->chain.size(); i++) {
					if (i > 0) {
						node_name += "/";
					}
					node_name += get_node->chain[i]->name;
				}
			}

			Vector<GDScriptCodeGenerator::Address> args;
			args.push_back(codegen.add_constant(NodePath(node_name)));

			GDScriptCodeGenerator::Address result = codegen.add_temporary(_gdtype_from_datatype(get_node->get_datatype()));

			MethodBind *get_node_method = ClassDB::get_method("Node", "get_node");
			gen->write_call_ptrcall(result, GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::SELF), get_node_method, args);

			return result;
		} break;
		case GDScriptParser::Node::PRELOAD: {
			const GDScriptParser::PreloadNode *preload = static_cast<const GDScriptParser::PreloadNode *>(p_expression);

			// Add resource as constant.
			return codegen.add_constant(preload->resource);
		} break;
		case GDScriptParser::Node::AWAIT: {
			const GDScriptParser::AwaitNode *await = static_cast<const GDScriptParser::AwaitNode *>(p_expression);

			GDScriptCodeGenerator::Address result = codegen.add_temporary(_gdtype_from_datatype(p_expression->get_datatype()));
			within_await = true;
			GDScriptCodeGenerator::Address argument = _parse_expression(codegen, r_error, await->to_await);
			within_await = false;
			if (r_error) {
				return GDScriptCodeGenerator::Address();
			}

			gen->write_await(result, argument);

			if (argument.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				gen->pop_temporary();
			}

			return result;
		} break;
		// Indexing operator.
		case GDScriptParser::Node::SUBSCRIPT: {
			const GDScriptParser::SubscriptNode *subscript = static_cast<const GDScriptParser::SubscriptNode *>(p_expression);
			GDScriptCodeGenerator::Address result = codegen.add_temporary(_gdtype_from_datatype(subscript->get_datatype()));

			GDScriptCodeGenerator::Address base = _parse_expression(codegen, r_error, subscript->base);
			if (r_error) {
				return GDScriptCodeGenerator::Address();
			}

			bool named = subscript->is_attribute;
			StringName name;
			GDScriptCodeGenerator::Address index;
			if (p_index_addr.mode != GDScriptCodeGenerator::Address::NIL) {
				index = p_index_addr;
			} else if (subscript->is_attribute) {
				if (subscript->base->type == GDScriptParser::Node::SELF && codegen.script) {
					GDScriptParser::IdentifierNode *identifier = subscript->attribute;
					const Map<StringName, GDScript::MemberInfo>::Element *MI = codegen.script->member_indices.find(identifier->name);

#ifdef DEBUG_ENABLED
					if (MI && MI->get().getter == codegen.function_name) {
						String n = identifier->name;
						_set_error("Must use '" + n + "' instead of 'self." + n + "' in getter.", identifier);
						r_error = ERR_COMPILATION_FAILED;
						return GDScriptCodeGenerator::Address();
					}
#endif

					if (MI && MI->get().getter == "") {
						// Remove result temp as we don't need it.
						gen->pop_temporary();
						// Faster than indexing self (as if no self. had been used).
						return GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::MEMBER, MI->get().index, _gdtype_from_datatype(subscript->get_datatype()));
					}
				}

				name = subscript->attribute->name;
				named = true;
			} else {
				if (subscript->index->is_constant && subscript->index->reduced_value.get_type() == Variant::STRING_NAME) {
					// Also, somehow, named (speed up anyway).
					name = subscript->index->reduced_value;
					named = true;
				} else {
					// Regular indexing.
					index = _parse_expression(codegen, r_error, subscript->index);
					if (r_error) {
						return GDScriptCodeGenerator::Address();
					}
				}
			}

			if (named) {
				gen->write_get_named(result, name, base);
			} else {
				gen->write_get(result, index, base);
			}

			if (index.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				gen->pop_temporary();
			}
			if (base.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				gen->pop_temporary();
			}

			return result;
		} break;
		case GDScriptParser::Node::UNARY_OPERATOR: {
			const GDScriptParser::UnaryOpNode *unary = static_cast<const GDScriptParser::UnaryOpNode *>(p_expression);

			GDScriptCodeGenerator::Address result = codegen.add_temporary(_gdtype_from_datatype(unary->get_datatype()));

			GDScriptCodeGenerator::Address operand = _parse_expression(codegen, r_error, unary->operand);
			if (r_error) {
				return GDScriptCodeGenerator::Address();
			}

			gen->write_unary_operator(result, unary->variant_op, operand);

			if (operand.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				gen->pop_temporary();
			}

			return result;
		}
		case GDScriptParser::Node::BINARY_OPERATOR: {
			const GDScriptParser::BinaryOpNode *binary = static_cast<const GDScriptParser::BinaryOpNode *>(p_expression);

			GDScriptCodeGenerator::Address result = codegen.add_temporary(_gdtype_from_datatype(binary->get_datatype()));

			switch (binary->operation) {
				case GDScriptParser::BinaryOpNode::OP_LOGIC_AND: {
					// AND operator with early out on failure.
					GDScriptCodeGenerator::Address left_operand = _parse_expression(codegen, r_error, binary->left_operand);
					gen->write_and_left_operand(left_operand);
					GDScriptCodeGenerator::Address right_operand = _parse_expression(codegen, r_error, binary->right_operand);
					gen->write_and_right_operand(right_operand);

					gen->write_end_and(result);

					if (right_operand.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
					if (left_operand.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_LOGIC_OR: {
					// OR operator with early out on success.
					GDScriptCodeGenerator::Address left_operand = _parse_expression(codegen, r_error, binary->left_operand);
					gen->write_or_left_operand(left_operand);
					GDScriptCodeGenerator::Address right_operand = _parse_expression(codegen, r_error, binary->right_operand);
					gen->write_or_right_operand(right_operand);

					gen->write_end_or(result);

					if (right_operand.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
					if (left_operand.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_TYPE_TEST: {
					GDScriptCodeGenerator::Address operand = _parse_expression(codegen, r_error, binary->left_operand);

					if (binary->right_operand->type == GDScriptParser::Node::IDENTIFIER && GDScriptParser::get_builtin_type(static_cast<const GDScriptParser::IdentifierNode *>(binary->right_operand)->name) != Variant::VARIANT_MAX) {
						// `is` with builtin type)
						Variant::Type type = GDScriptParser::get_builtin_type(static_cast<const GDScriptParser::IdentifierNode *>(binary->right_operand)->name);
						gen->write_type_test_builtin(result, operand, type);
					} else {
						GDScriptCodeGenerator::Address type = _parse_expression(codegen, r_error, binary->right_operand);
						if (r_error) {
							return GDScriptCodeGenerator::Address();
						}
						gen->write_type_test(result, operand, type);
						if (type.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
							gen->pop_temporary();
						}
					}

					if (operand.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
				} break;
				default: {
					GDScriptCodeGenerator::Address left_operand = _parse_expression(codegen, r_error, binary->left_operand);
					GDScriptCodeGenerator::Address right_operand = _parse_expression(codegen, r_error, binary->right_operand);

					gen->write_binary_operator(result, binary->variant_op, left_operand, right_operand);

					if (right_operand.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
					if (left_operand.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
				}
			}
			return result;
		} break;
		case GDScriptParser::Node::TERNARY_OPERATOR: {
			// x IF a ELSE y operator with early out on failure.
			const GDScriptParser::TernaryOpNode *ternary = static_cast<const GDScriptParser::TernaryOpNode *>(p_expression);
			GDScriptCodeGenerator::Address result = codegen.add_temporary(_gdtype_from_datatype(ternary->get_datatype()));

			gen->write_start_ternary(result);

			GDScriptCodeGenerator::Address condition = _parse_expression(codegen, r_error, ternary->condition);
			if (r_error) {
				return GDScriptCodeGenerator::Address();
			}
			gen->write_ternary_condition(condition);

			if (condition.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				gen->pop_temporary();
			}

			GDScriptCodeGenerator::Address true_expr = _parse_expression(codegen, r_error, ternary->true_expr);
			if (r_error) {
				return GDScriptCodeGenerator::Address();
			}
			gen->write_ternary_true_expr(true_expr);
			if (true_expr.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				gen->pop_temporary();
			}

			GDScriptCodeGenerator::Address false_expr = _parse_expression(codegen, r_error, ternary->false_expr);
			if (r_error) {
				return GDScriptCodeGenerator::Address();
			}
			gen->write_ternary_false_expr(false_expr);
			if (false_expr.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				gen->pop_temporary();
			}

			gen->write_end_ternary();

			return result;
		} break;
		case GDScriptParser::Node::ASSIGNMENT: {
			const GDScriptParser::AssignmentNode *assignment = static_cast<const GDScriptParser::AssignmentNode *>(p_expression);

			if (assignment->assignee->type == GDScriptParser::Node::SUBSCRIPT) {
				// SET (chained) MODE!
				const GDScriptParser::SubscriptNode *subscript = static_cast<GDScriptParser::SubscriptNode *>(assignment->assignee);
#ifdef DEBUG_ENABLED
				if (subscript->is_attribute && subscript->base->type == GDScriptParser::Node::SELF && codegen.script) {
					const Map<StringName, GDScript::MemberInfo>::Element *MI = codegen.script->member_indices.find(subscript->attribute->name);
					if (MI && MI->get().setter == codegen.function_name) {
						String n = subscript->attribute->name;
						_set_error("Must use '" + n + "' instead of 'self." + n + "' in setter.", subscript);
						r_error = ERR_COMPILATION_FAILED;
						return GDScriptCodeGenerator::Address();
					}
				}
#endif
				/* Find chain of sets */

				StringName assign_property;

				List<const GDScriptParser::SubscriptNode *> chain;

				{
					// Create get/set chain.
					const GDScriptParser::SubscriptNode *n = subscript;
					while (true) {
						chain.push_back(n);
						if (n->base->type != GDScriptParser::Node::SUBSCRIPT) {
							// Check for a built-in property.
							if (n->base->type == GDScriptParser::Node::IDENTIFIER) {
								GDScriptParser::IdentifierNode *identifier = static_cast<GDScriptParser::IdentifierNode *>(n->base);
								if (_is_class_member_property(codegen, identifier->name)) {
									assign_property = identifier->name;
								}
							}
							break;
						}
						n = static_cast<const GDScriptParser::SubscriptNode *>(n->base);
					}
				}

				/* Chain of gets */

				// Get at (potential) root stack pos, so it can be returned.
				GDScriptCodeGenerator::Address base = _parse_expression(codegen, r_error, chain.back()->get()->base);
				if (r_error) {
					return GDScriptCodeGenerator::Address();
				}

				GDScriptCodeGenerator::Address prev_base = base;

				struct ChainInfo {
					bool is_named = false;
					GDScriptCodeGenerator::Address base;
					GDScriptCodeGenerator::Address key;
					StringName name;
				};

				List<ChainInfo> set_chain;

				for (List<const GDScriptParser::SubscriptNode *>::Element *E = chain.back(); E; E = E->prev()) {
					if (E == chain.front()) {
						// Skip the main subscript, since we'll assign to that.
						break;
					}
					const GDScriptParser::SubscriptNode *subscript_elem = E->get();
					GDScriptCodeGenerator::Address value = codegen.add_temporary(_gdtype_from_datatype(subscript_elem->get_datatype()));
					GDScriptCodeGenerator::Address key;
					StringName name;

					if (subscript_elem->is_attribute) {
						name = subscript_elem->attribute->name;
						gen->write_get_named(value, name, prev_base);
					} else {
						key = _parse_expression(codegen, r_error, subscript_elem->index);
						if (r_error) {
							return GDScriptCodeGenerator::Address();
						}
						gen->write_get(value, key, prev_base);
					}

					// Store base and key for setting it back later.
					set_chain.push_front({ subscript_elem->is_attribute, prev_base, key, name }); // Push to front to invert the list.
					prev_base = value;
				}

				// Get value to assign.
				GDScriptCodeGenerator::Address assigned = _parse_expression(codegen, r_error, assignment->assigned_value);
				if (r_error) {
					return GDScriptCodeGenerator::Address();
				}
				// Get the key if needed.
				GDScriptCodeGenerator::Address key;
				StringName name;
				if (subscript->is_attribute) {
					name = subscript->attribute->name;
				} else {
					key = _parse_expression(codegen, r_error, subscript->index);
					if (r_error) {
						return GDScriptCodeGenerator::Address();
					}
				}

				// Perform operator if any.
				if (assignment->operation != GDScriptParser::AssignmentNode::OP_NONE) {
					GDScriptCodeGenerator::Address op_result = codegen.add_temporary(_gdtype_from_datatype(assignment->get_datatype()));
					GDScriptCodeGenerator::Address value = codegen.add_temporary(_gdtype_from_datatype(subscript->get_datatype()));
					if (subscript->is_attribute) {
						gen->write_get_named(value, name, prev_base);
					} else {
						gen->write_get(value, key, prev_base);
					}
					gen->write_binary_operator(op_result, assignment->variant_op, value, assigned);
					gen->pop_temporary();
					if (assigned.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
					assigned = op_result;
				}

				// Perform assignment.
				if (subscript->is_attribute) {
					gen->write_set_named(prev_base, name, assigned);
				} else {
					gen->write_set(prev_base, key, assigned);
				}
				if (key.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary();
				}
				if (assigned.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary();
				}

				assigned = prev_base;

				// Set back the values into their bases.
				for (const ChainInfo &info : set_chain) {
					if (!info.is_named) {
						gen->write_set(info.base, info.key, assigned);
						if (info.key.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
							gen->pop_temporary();
						}
					} else {
						gen->write_set_named(info.base, info.name, assigned);
					}
					if (assigned.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
					assigned = info.base;
				}

				// If this is a local member, also assign to it.
				// This allow things like: position.x += 2.0
				if (assign_property != StringName()) {
					gen->write_set_member(assigned, assign_property);
				}

				if (assigned.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary();
				}
			} else if (assignment->assignee->type == GDScriptParser::Node::IDENTIFIER && _is_class_member_property(codegen, static_cast<GDScriptParser::IdentifierNode *>(assignment->assignee)->name)) {
				// Assignment to member property.
				GDScriptCodeGenerator::Address assigned_value = _parse_expression(codegen, r_error, assignment->assigned_value);
				if (r_error) {
					return GDScriptCodeGenerator::Address();
				}

				GDScriptCodeGenerator::Address to_assign = assigned_value;
				bool has_operation = assignment->operation != GDScriptParser::AssignmentNode::OP_NONE;

				StringName name = static_cast<GDScriptParser::IdentifierNode *>(assignment->assignee)->name;

				if (has_operation) {
					GDScriptCodeGenerator::Address op_result = codegen.add_temporary(_gdtype_from_datatype(assignment->get_datatype()));
					GDScriptCodeGenerator::Address member = codegen.add_temporary(_gdtype_from_datatype(assignment->assignee->get_datatype()));
					gen->write_get_member(member, name);
					gen->write_binary_operator(op_result, assignment->variant_op, member, assigned_value);
					gen->pop_temporary(); // Pop member temp.
					to_assign = op_result;
				}

				gen->write_set_member(to_assign, name);

				if (to_assign.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary(); // Pop the assigned expression or the temp result if it has operation.
				}
				if (has_operation && assigned_value.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary(); // Pop the assigned expression if not done before.
				}
			} else {
				// Regular assignment.
				GDScriptCodeGenerator::Address target;

				bool has_setter = false;
				bool is_in_setter = false;
				StringName setter_function;
				if (assignment->assignee->type == GDScriptParser::Node::IDENTIFIER) {
					StringName var_name = static_cast<const GDScriptParser::IdentifierNode *>(assignment->assignee)->name;
					if (!codegen.locals.has(var_name) && codegen.script->member_indices.has(var_name)) {
						setter_function = codegen.script->member_indices[var_name].setter;
						if (setter_function != StringName()) {
							has_setter = true;
							is_in_setter = setter_function == codegen.function_name;
							target.mode = GDScriptCodeGenerator::Address::MEMBER;
							target.address = codegen.script->member_indices[var_name].index;
						}
					}
				}

				if (has_setter) {
					if (!is_in_setter) {
						// Store stack slot for the temp value.
						target = codegen.add_temporary(_gdtype_from_datatype(assignment->assignee->get_datatype()));
					}
				} else {
					target = _parse_expression(codegen, r_error, assignment->assignee);
					if (r_error) {
						return GDScriptCodeGenerator::Address();
					}
				}

				GDScriptCodeGenerator::Address assigned_value = _parse_expression(codegen, r_error, assignment->assigned_value);
				if (r_error) {
					return GDScriptCodeGenerator::Address();
				}

				GDScriptCodeGenerator::Address to_assign;
				bool has_operation = assignment->operation != GDScriptParser::AssignmentNode::OP_NONE;
				if (has_operation) {
					// Perform operation.
					GDScriptCodeGenerator::Address op_result = codegen.add_temporary(_gdtype_from_datatype(assignment->get_datatype()));
					GDScriptCodeGenerator::Address og_value = _parse_expression(codegen, r_error, assignment->assignee);
					gen->write_binary_operator(op_result, assignment->variant_op, og_value, assigned_value);
					to_assign = op_result;

					if (og_value.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						gen->pop_temporary();
					}
				} else {
					to_assign = assigned_value;
				}

				GDScriptDataType assign_type = _gdtype_from_datatype(assignment->assignee->get_datatype());

				if (has_setter && !is_in_setter) {
					// Call setter.
					Vector<GDScriptCodeGenerator::Address> args;
					args.push_back(to_assign);
					gen->write_call(GDScriptCodeGenerator::Address(), GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::SELF), setter_function, args);
				} else {
					// Just assign.
					if (assignment->use_conversion_assign) {
						gen->write_assign_with_conversion(target, to_assign);
					} else {
						gen->write_assign(target, to_assign);
					}
				}

				if (to_assign.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary(); // Pop assigned value or temp operation result.
				}
				if (has_operation && assigned_value.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary(); // Pop assigned value if not done before.
				}
				if (target.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary(); // Pop the target to assignment.
				}
			}
			return GDScriptCodeGenerator::Address(); // Assignment does not return a value.
		} break;
		case GDScriptParser::Node::LAMBDA: {
			const GDScriptParser::LambdaNode *lambda = static_cast<const GDScriptParser::LambdaNode *>(p_expression);
			GDScriptCodeGenerator::Address result = codegen.add_temporary(_gdtype_from_datatype(lambda->get_datatype()));

			Vector<GDScriptCodeGenerator::Address> captures;
			captures.resize(lambda->captures.size());
			for (int i = 0; i < lambda->captures.size(); i++) {
				captures.write[i] = _parse_expression(codegen, r_error, lambda->captures[i]);
				if (r_error) {
					return GDScriptCodeGenerator::Address();
				}
			}

			GDScriptFunction *function = _parse_function(r_error, codegen.script, codegen.class_node, lambda->function, false, true);
			if (r_error) {
				return GDScriptCodeGenerator::Address();
			}

			gen->write_lambda(result, function, captures);

			for (int i = 0; i < captures.size(); i++) {
				if (captures[i].mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					gen->pop_temporary();
				}
			}

			return result;
		} break;
		default: {
			ERR_FAIL_V_MSG(GDScriptCodeGenerator::Address(), "Bug in bytecode compiler, unexpected node in parse tree while parsing expression."); // Unreachable code.
		} break;
	}
}

GDScriptCodeGenerator::Address GDScriptCompiler::_parse_match_pattern(CodeGen &codegen, Error &r_error, const GDScriptParser::PatternNode *p_pattern, const GDScriptCodeGenerator::Address &p_value_addr, const GDScriptCodeGenerator::Address &p_type_addr, const GDScriptCodeGenerator::Address &p_previous_test, bool p_is_first, bool p_is_nested) {
	switch (p_pattern->pattern_type) {
		case GDScriptParser::PatternNode::PT_LITERAL: {
			if (p_is_nested) {
				codegen.generator->write_and_left_operand(p_previous_test);
			} else if (!p_is_first) {
				codegen.generator->write_or_left_operand(p_previous_test);
			}

			// Get literal type into constant map.
			GDScriptCodeGenerator::Address literal_type_addr = codegen.add_constant((int)p_pattern->literal->value.get_type());

			// Equality is always a boolean.
			GDScriptDataType equality_type;
			equality_type.has_type = true;
			equality_type.kind = GDScriptDataType::BUILTIN;
			equality_type.builtin_type = Variant::BOOL;

			// Check type equality.
			GDScriptCodeGenerator::Address type_equality_addr = codegen.add_temporary(equality_type);
			codegen.generator->write_binary_operator(type_equality_addr, Variant::OP_EQUAL, p_type_addr, literal_type_addr);
			codegen.generator->write_and_left_operand(type_equality_addr);

			// Get literal.
			GDScriptCodeGenerator::Address literal_addr = _parse_expression(codegen, r_error, p_pattern->literal);
			if (r_error) {
				return GDScriptCodeGenerator::Address();
			}

			// Check value equality.
			GDScriptCodeGenerator::Address equality_addr = codegen.add_temporary(equality_type);
			codegen.generator->write_binary_operator(equality_addr, Variant::OP_EQUAL, p_value_addr, literal_addr);
			codegen.generator->write_and_right_operand(equality_addr);

			// AND both together (reuse temporary location).
			codegen.generator->write_end_and(type_equality_addr);

			codegen.generator->pop_temporary(); // Remove equality_addr from stack.

			if (literal_addr.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				codegen.generator->pop_temporary();
			}

			// If this isn't the first, we need to OR with the previous pattern. If it's nested, we use AND instead.
			if (p_is_nested) {
				// Use the previous value as target, since we only need one temporary variable.
				codegen.generator->write_and_right_operand(type_equality_addr);
				codegen.generator->write_end_and(p_previous_test);
			} else if (!p_is_first) {
				// Use the previous value as target, since we only need one temporary variable.
				codegen.generator->write_or_right_operand(type_equality_addr);
				codegen.generator->write_end_or(p_previous_test);
			} else {
				// Just assign this value to the accumulator temporary.
				codegen.generator->write_assign(p_previous_test, type_equality_addr);
			}
			codegen.generator->pop_temporary(); // Remove type_equality_addr.

			return p_previous_test;
		} break;
		case GDScriptParser::PatternNode::PT_EXPRESSION: {
			if (p_is_nested) {
				codegen.generator->write_and_left_operand(p_previous_test);
			} else if (!p_is_first) {
				codegen.generator->write_or_left_operand(p_previous_test);
			}
			// Create the result temps first since it's the last to go away.
			GDScriptCodeGenerator::Address result_addr = codegen.add_temporary();
			GDScriptCodeGenerator::Address equality_test_addr = codegen.add_temporary();

			// Evaluate expression.
			GDScriptCodeGenerator::Address expr_addr;
			expr_addr = _parse_expression(codegen, r_error, p_pattern->expression);
			if (r_error) {
				return GDScriptCodeGenerator::Address();
			}

			// Evaluate expression type.
			Vector<GDScriptCodeGenerator::Address> typeof_args;
			typeof_args.push_back(expr_addr);
			codegen.generator->write_call_utility(result_addr, "typeof", typeof_args);

			// Check type equality.
			codegen.generator->write_binary_operator(result_addr, Variant::OP_EQUAL, p_type_addr, result_addr);
			codegen.generator->write_and_left_operand(result_addr);

			// Check value equality.
			codegen.generator->write_binary_operator(equality_test_addr, Variant::OP_EQUAL, p_value_addr, expr_addr);
			codegen.generator->write_and_right_operand(equality_test_addr);

			// AND both type and value equality.
			codegen.generator->write_end_and(result_addr);

			// We don't need the expression temporary anymore.
			if (expr_addr.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
				codegen.generator->pop_temporary();
			}
			codegen.generator->pop_temporary(); // Remove type equality temporary.

			// If this isn't the first, we need to OR with the previous pattern. If it's nested, we use AND instead.
			if (p_is_nested) {
				// Use the previous value as target, since we only need one temporary variable.
				codegen.generator->write_and_right_operand(result_addr);
				codegen.generator->write_end_and(p_previous_test);
			} else if (!p_is_first) {
				// Use the previous value as target, since we only need one temporary variable.
				codegen.generator->write_or_right_operand(result_addr);
				codegen.generator->write_end_or(p_previous_test);
			} else {
				// Just assign this value to the accumulator temporary.
				codegen.generator->write_assign(p_previous_test, result_addr);
			}
			codegen.generator->pop_temporary(); // Remove temp result addr.

			return p_previous_test;
		} break;
		case GDScriptParser::PatternNode::PT_ARRAY: {
			if (p_is_nested) {
				codegen.generator->write_and_left_operand(p_previous_test);
			} else if (!p_is_first) {
				codegen.generator->write_or_left_operand(p_previous_test);
			}
			// Get array type into constant map.
			GDScriptCodeGenerator::Address array_type_addr = codegen.add_constant((int)Variant::ARRAY);

			// Equality is always a boolean.
			GDScriptDataType temp_type;
			temp_type.has_type = true;
			temp_type.kind = GDScriptDataType::BUILTIN;
			temp_type.builtin_type = Variant::BOOL;

			// Check type equality.
			GDScriptCodeGenerator::Address result_addr = codegen.add_temporary(temp_type);
			codegen.generator->write_binary_operator(result_addr, Variant::OP_EQUAL, p_type_addr, array_type_addr);
			codegen.generator->write_and_left_operand(result_addr);

			// Store pattern length in constant map.
			GDScriptCodeGenerator::Address array_length_addr = codegen.add_constant(p_pattern->rest_used ? p_pattern->array.size() - 1 : p_pattern->array.size());

			// Get value length.
			temp_type.builtin_type = Variant::INT;
			GDScriptCodeGenerator::Address value_length_addr = codegen.add_temporary(temp_type);
			Vector<GDScriptCodeGenerator::Address> len_args;
			len_args.push_back(p_value_addr);
			codegen.generator->write_call_gdscript_utility(value_length_addr, GDScriptUtilityFunctions::get_function("len"), len_args);

			// Test length compatibility.
			temp_type.builtin_type = Variant::BOOL;
			GDScriptCodeGenerator::Address length_compat_addr = codegen.add_temporary(temp_type);
			codegen.generator->write_binary_operator(length_compat_addr, p_pattern->rest_used ? Variant::OP_GREATER_EQUAL : Variant::OP_EQUAL, value_length_addr, array_length_addr);
			codegen.generator->write_and_right_operand(length_compat_addr);

			// AND type and length check.
			codegen.generator->write_end_and(result_addr);

			// Remove length temporaries.
			codegen.generator->pop_temporary();
			codegen.generator->pop_temporary();

			// If this isn't the first, we need to OR with the previous pattern. If it's nested, we use AND instead.
			if (p_is_nested) {
				// Use the previous value as target, since we only need one temporary variable.
				codegen.generator->write_and_right_operand(result_addr);
				codegen.generator->write_end_and(p_previous_test);
			} else if (!p_is_first) {
				// Use the previous value as target, since we only need one temporary variable.
				codegen.generator->write_or_right_operand(result_addr);
				codegen.generator->write_end_or(p_previous_test);
			} else {
				// Just assign this value to the accumulator temporary.
				codegen.generator->write_assign(p_previous_test, result_addr);
			}
			codegen.generator->pop_temporary(); // Remove temp result addr.

			// Create temporaries outside the loop so they can be reused.
			GDScriptCodeGenerator::Address element_addr = codegen.add_temporary();
			GDScriptCodeGenerator::Address element_type_addr = codegen.add_temporary();
			GDScriptCodeGenerator::Address test_addr = p_previous_test;

			// Evaluate element by element.
			for (int i = 0; i < p_pattern->array.size(); i++) {
				if (p_pattern->array[i]->pattern_type == GDScriptParser::PatternNode::PT_REST) {
					// Don't want to access an extra element of the user array.
					break;
				}

				// Use AND here too, as we don't want to be checking elements if previous test failed (which means this might be an invalid get).
				codegen.generator->write_and_left_operand(test_addr);

				// Add index to constant map.
				GDScriptCodeGenerator::Address index_addr = codegen.add_constant(i);

				// Get the actual element from the user-sent array.
				codegen.generator->write_get(element_addr, index_addr, p_value_addr);

				// Also get type of element.
				Vector<GDScriptCodeGenerator::Address> typeof_args;
				typeof_args.push_back(element_addr);
				codegen.generator->write_call_utility(element_type_addr, "typeof", typeof_args);

				// Try the pattern inside the element.
				test_addr = _parse_match_pattern(codegen, r_error, p_pattern->array[i], element_addr, element_type_addr, p_previous_test, false, true);
				if (r_error != OK) {
					return GDScriptCodeGenerator::Address();
				}

				codegen.generator->write_and_right_operand(test_addr);
				codegen.generator->write_end_and(test_addr);
			}
			// Remove element temporaries.
			codegen.generator->pop_temporary();
			codegen.generator->pop_temporary();

			return test_addr;
		} break;
		case GDScriptParser::PatternNode::PT_DICTIONARY: {
			if (p_is_nested) {
				codegen.generator->write_and_left_operand(p_previous_test);
			} else if (!p_is_first) {
				codegen.generator->write_or_left_operand(p_previous_test);
			}
			// Get dictionary type into constant map.
			GDScriptCodeGenerator::Address dict_type_addr = codegen.add_constant((int)Variant::DICTIONARY);

			// Equality is always a boolean.
			GDScriptDataType temp_type;
			temp_type.has_type = true;
			temp_type.kind = GDScriptDataType::BUILTIN;
			temp_type.builtin_type = Variant::BOOL;

			// Check type equality.
			GDScriptCodeGenerator::Address result_addr = codegen.add_temporary(temp_type);
			codegen.generator->write_binary_operator(result_addr, Variant::OP_EQUAL, p_type_addr, dict_type_addr);
			codegen.generator->write_and_left_operand(result_addr);

			// Store pattern length in constant map.
			GDScriptCodeGenerator::Address dict_length_addr = codegen.add_constant(p_pattern->rest_used ? p_pattern->dictionary.size() - 1 : p_pattern->dictionary.size());

			// Get user's dictionary length.
			temp_type.builtin_type = Variant::INT;
			GDScriptCodeGenerator::Address value_length_addr = codegen.add_temporary(temp_type);
			Vector<GDScriptCodeGenerator::Address> func_args;
			func_args.push_back(p_value_addr);
			codegen.generator->write_call_gdscript_utility(value_length_addr, GDScriptUtilityFunctions::get_function("len"), func_args);

			// Test length compatibility.
			temp_type.builtin_type = Variant::BOOL;
			GDScriptCodeGenerator::Address length_compat_addr = codegen.add_temporary(temp_type);
			codegen.generator->write_binary_operator(length_compat_addr, p_pattern->rest_used ? Variant::OP_GREATER_EQUAL : Variant::OP_EQUAL, value_length_addr, dict_length_addr);
			codegen.generator->write_and_right_operand(length_compat_addr);

			// AND type and length check.
			codegen.generator->write_end_and(result_addr);

			// Remove length temporaries.
			codegen.generator->pop_temporary();
			codegen.generator->pop_temporary();

			// If this isn't the first, we need to OR with the previous pattern. If it's nested, we use AND instead.
			if (p_is_nested) {
				// Use the previous value as target, since we only need one temporary variable.
				codegen.generator->write_and_right_operand(result_addr);
				codegen.generator->write_end_and(p_previous_test);
			} else if (!p_is_first) {
				// Use the previous value as target, since we only need one temporary variable.
				codegen.generator->write_or_right_operand(result_addr);
				codegen.generator->write_end_or(p_previous_test);
			} else {
				// Just assign this value to the accumulator temporary.
				codegen.generator->write_assign(p_previous_test, result_addr);
			}
			codegen.generator->pop_temporary(); // Remove temp result addr.

			// Create temporaries outside the loop so they can be reused.
			temp_type.builtin_type = Variant::BOOL;
			GDScriptCodeGenerator::Address test_result = codegen.add_temporary(temp_type);
			GDScriptCodeGenerator::Address element_addr = codegen.add_temporary();
			GDScriptCodeGenerator::Address element_type_addr = codegen.add_temporary();
			GDScriptCodeGenerator::Address test_addr = p_previous_test;

			// Evaluate element by element.
			for (int i = 0; i < p_pattern->dictionary.size(); i++) {
				const GDScriptParser::PatternNode::Pair &element = p_pattern->dictionary[i];
				if (element.value_pattern && element.value_pattern->pattern_type == GDScriptParser::PatternNode::PT_REST) {
					// Ignore rest pattern.
					break;
				}

				// Use AND here too, as we don't want to be checking elements if previous test failed (which means this might be an invalid get).
				codegen.generator->write_and_left_operand(test_addr);

				// Get the pattern key.
				GDScriptCodeGenerator::Address pattern_key_addr = _parse_expression(codegen, r_error, element.key);
				if (r_error) {
					return GDScriptCodeGenerator::Address();
				}

				// Check if pattern key exists in user's dictionary. This will be AND-ed with next result.
				func_args.clear();
				func_args.push_back(pattern_key_addr);
				codegen.generator->write_call(test_result, p_value_addr, "has", func_args);

				if (element.value_pattern != nullptr) {
					// Use AND here too, as we don't want to be checking elements if previous test failed (which means this might be an invalid get).
					codegen.generator->write_and_left_operand(test_result);

					// Get actual value from user dictionary.
					codegen.generator->write_get(element_addr, pattern_key_addr, p_value_addr);

					// Also get type of value.
					func_args.clear();
					func_args.push_back(element_addr);
					codegen.generator->write_call_utility(element_type_addr, "typeof", func_args);

					// Try the pattern inside the value.
					test_addr = _parse_match_pattern(codegen, r_error, element.value_pattern, element_addr, element_type_addr, test_addr, false, true);
					if (r_error != OK) {
						return GDScriptCodeGenerator::Address();
					}
					codegen.generator->write_and_right_operand(test_addr);
					codegen.generator->write_end_and(test_addr);
				}

				codegen.generator->write_and_right_operand(test_addr);
				codegen.generator->write_end_and(test_addr);

				// Remove pattern key temporary.
				if (pattern_key_addr.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}
			}

			// Remove element temporaries.
			codegen.generator->pop_temporary();
			codegen.generator->pop_temporary();
			codegen.generator->pop_temporary();

			return test_addr;
		} break;
		case GDScriptParser::PatternNode::PT_REST:
			// Do nothing.
			return p_previous_test;
			break;
		case GDScriptParser::PatternNode::PT_BIND: {
			if (p_is_nested) {
				codegen.generator->write_and_left_operand(p_previous_test);
			} else if (!p_is_first) {
				codegen.generator->write_or_left_operand(p_previous_test);
			}
			// Get the bind address.
			GDScriptCodeGenerator::Address bind = codegen.locals[p_pattern->bind->name];

			// Assign value to bound variable.
			codegen.generator->write_assign(bind, p_value_addr);
		}
			[[fallthrough]]; // Act like matching anything too.
		case GDScriptParser::PatternNode::PT_WILDCARD:
			// If this is a fall through we don't want to do this again.
			if (p_pattern->pattern_type != GDScriptParser::PatternNode::PT_BIND) {
				if (p_is_nested) {
					codegen.generator->write_and_left_operand(p_previous_test);
				} else if (!p_is_first) {
					codegen.generator->write_or_left_operand(p_previous_test);
				}
			}
			// This matches anything so just do the same as `if(true)`.
			// If this isn't the first, we need to OR with the previous pattern. If it's nested, we use AND instead.
			if (p_is_nested) {
				// Use the operator with the `true` constant so it works as always matching.
				GDScriptCodeGenerator::Address constant = codegen.add_constant(true);
				codegen.generator->write_and_right_operand(constant);
				codegen.generator->write_end_and(p_previous_test);
			} else if (!p_is_first) {
				// Use the operator with the `true` constant so it works as always matching.
				GDScriptCodeGenerator::Address constant = codegen.add_constant(true);
				codegen.generator->write_or_right_operand(constant);
				codegen.generator->write_end_or(p_previous_test);
			} else {
				// Just assign this value to the accumulator temporary.
				codegen.generator->write_assign_true(p_previous_test);
			}
			return p_previous_test;
	}
	ERR_FAIL_V_MSG(p_previous_test, "Reaching the end of pattern compilation without matching a pattern.");
}

void GDScriptCompiler::_add_locals_in_block(CodeGen &codegen, const GDScriptParser::SuiteNode *p_block) {
	for (int i = 0; i < p_block->locals.size(); i++) {
		if (p_block->locals[i].type == GDScriptParser::SuiteNode::Local::PARAMETER || p_block->locals[i].type == GDScriptParser::SuiteNode::Local::FOR_VARIABLE) {
			// Parameters are added directly from function and loop variables are declared explicitly.
			continue;
		}
		codegen.add_local(p_block->locals[i].name, _gdtype_from_datatype(p_block->locals[i].get_datatype()));
	}
}

Error GDScriptCompiler::_parse_block(CodeGen &codegen, const GDScriptParser::SuiteNode *p_block, bool p_add_locals) {
	Error error = OK;
	GDScriptCodeGenerator *gen = codegen.generator;

	codegen.start_block();

	if (p_add_locals) {
		_add_locals_in_block(codegen, p_block);
	}

	for (int i = 0; i < p_block->statements.size(); i++) {
		const GDScriptParser::Node *s = p_block->statements[i];

#ifdef DEBUG_ENABLED
		// Add a newline before each statement, since the debugger needs those.
		gen->write_newline(s->start_line);
#endif

		switch (s->type) {
			case GDScriptParser::Node::MATCH: {
				const GDScriptParser::MatchNode *match = static_cast<const GDScriptParser::MatchNode *>(s);

				gen->start_match();
				codegen.start_block();

				// Evaluate the match expression.
				GDScriptCodeGenerator::Address value = codegen.add_local("@match_value", _gdtype_from_datatype(match->test->get_datatype()));
				GDScriptCodeGenerator::Address value_expr = _parse_expression(codegen, error, match->test);
				if (error) {
					return error;
				}

				// Assign to local.
				// TODO: This can be improved by passing the target to parse_expression().
				gen->write_assign(value, value_expr);

				if (value_expr.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}

				// Then, let's save the type of the value in the stack too, so we can reuse for later comparisons.
				GDScriptDataType typeof_type;
				typeof_type.has_type = true;
				typeof_type.kind = GDScriptDataType::BUILTIN;
				typeof_type.builtin_type = Variant::INT;
				GDScriptCodeGenerator::Address type = codegen.add_local("@match_type", typeof_type);

				Vector<GDScriptCodeGenerator::Address> typeof_args;
				typeof_args.push_back(value);
				gen->write_call_utility(type, "typeof", typeof_args);

				// Now we can actually start testing.
				// For each branch.
				for (int j = 0; j < match->branches.size(); j++) {
					if (j > 0) {
						// Use `else` to not check the next branch after matching.
						gen->write_else();
					}

					const GDScriptParser::MatchBranchNode *branch = match->branches[j];

					gen->start_match_branch(); // Need so lower level code can patch 'continue' jumps.
					codegen.start_block(); // Create an extra block around for binds.

					// Add locals in block before patterns, so temporaries don't use the stack address for binds.
					_add_locals_in_block(codegen, branch->block);

#ifdef DEBUG_ENABLED
					// Add a newline before each branch, since the debugger needs those.
					gen->write_newline(branch->start_line);
#endif
					// For each pattern in branch.
					GDScriptCodeGenerator::Address pattern_result = codegen.add_temporary();
					for (int k = 0; k < branch->patterns.size(); k++) {
						pattern_result = _parse_match_pattern(codegen, error, branch->patterns[k], value, type, pattern_result, k == 0, false);
						if (error != OK) {
							return error;
						}
					}

					// Check if pattern did match.
					gen->write_if(pattern_result);

					// Remove the result from stack.
					gen->pop_temporary();

					// Parse the branch block.
					error = _parse_block(codegen, branch->block, false); // Don't add locals again.
					if (error) {
						return error;
					}

					codegen.end_block(); // Get out of extra block.
				}

				// End all nested `if`s.
				for (int j = 0; j < match->branches.size(); j++) {
					gen->write_endif();
				}

				gen->end_match();
			} break;
			case GDScriptParser::Node::IF: {
				const GDScriptParser::IfNode *if_n = static_cast<const GDScriptParser::IfNode *>(s);
				GDScriptCodeGenerator::Address condition = _parse_expression(codegen, error, if_n->condition);
				if (error) {
					return error;
				}

				gen->write_if(condition);

				if (condition.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}

				error = _parse_block(codegen, if_n->true_block);
				if (error) {
					return error;
				}

				if (if_n->false_block) {
					gen->write_else();

					error = _parse_block(codegen, if_n->false_block);
					if (error) {
						return error;
					}
				}

				gen->write_endif();
			} break;
			case GDScriptParser::Node::FOR: {
				const GDScriptParser::ForNode *for_n = static_cast<const GDScriptParser::ForNode *>(s);

				codegen.start_block();
				GDScriptCodeGenerator::Address iterator = codegen.add_local(for_n->variable->name, _gdtype_from_datatype(for_n->variable->get_datatype()));

				gen->start_for(iterator.type, _gdtype_from_datatype(for_n->list->get_datatype()));

				GDScriptCodeGenerator::Address list = _parse_expression(codegen, error, for_n->list);
				if (error) {
					return error;
				}

				gen->write_for_assignment(iterator, list);

				if (list.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}

				gen->write_for();

				error = _parse_block(codegen, for_n->loop);
				if (error) {
					return error;
				}

				gen->write_endfor();

				codegen.end_block();
			} break;
			case GDScriptParser::Node::WHILE: {
				const GDScriptParser::WhileNode *while_n = static_cast<const GDScriptParser::WhileNode *>(s);

				gen->start_while_condition();

				GDScriptCodeGenerator::Address condition = _parse_expression(codegen, error, while_n->condition);
				if (error) {
					return error;
				}

				gen->write_while(condition);

				if (condition.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}

				error = _parse_block(codegen, while_n->loop);
				if (error) {
					return error;
				}

				gen->write_endwhile();
			} break;
			case GDScriptParser::Node::BREAK: {
				gen->write_break();
			} break;
			case GDScriptParser::Node::CONTINUE: {
				const GDScriptParser::ContinueNode *cont = static_cast<const GDScriptParser::ContinueNode *>(s);
				if (cont->is_for_match) {
					gen->write_continue_match();
				} else {
					gen->write_continue();
				}
			} break;
			case GDScriptParser::Node::RETURN: {
				const GDScriptParser::ReturnNode *return_n = static_cast<const GDScriptParser::ReturnNode *>(s);

				GDScriptCodeGenerator::Address return_value;

				if (return_n->return_value != nullptr) {
					return_value = _parse_expression(codegen, error, return_n->return_value);
					if (error) {
						return error;
					}
				}

				gen->write_return(return_value);
				if (return_value.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}
			} break;
			case GDScriptParser::Node::ASSERT: {
#ifdef DEBUG_ENABLED
				const GDScriptParser::AssertNode *as = static_cast<const GDScriptParser::AssertNode *>(s);

				GDScriptCodeGenerator::Address condition = _parse_expression(codegen, error, as->condition);
				if (error) {
					return error;
				}

				GDScriptCodeGenerator::Address message;

				if (as->message) {
					message = _parse_expression(codegen, error, as->message);
					if (error) {
						return error;
					}
				}
				gen->write_assert(condition, message);

				if (condition.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}
				if (message.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}
#endif
			} break;
			case GDScriptParser::Node::BREAKPOINT: {
#ifdef DEBUG_ENABLED
				gen->write_breakpoint();
#endif
			} break;
			case GDScriptParser::Node::VARIABLE: {
				const GDScriptParser::VariableNode *lv = static_cast<const GDScriptParser::VariableNode *>(s);
				// Should be already in stack when the block began.
				GDScriptCodeGenerator::Address local = codegen.locals[lv->identifier->name];
				GDScriptParser::DataType local_type = lv->get_datatype();

				if (lv->initializer != nullptr) {
					// For typed arrays we need to make sure this is already initialized correctly so typed assignment work.
					if (local_type.is_hard_type() && local_type.builtin_type == Variant::ARRAY) {
						if (local_type.has_container_element_type()) {
							codegen.generator->write_construct_typed_array(local, _gdtype_from_datatype(local_type.get_container_element_type(), codegen.script), Vector<GDScriptCodeGenerator::Address>());
						} else {
							codegen.generator->write_construct_array(local, Vector<GDScriptCodeGenerator::Address>());
						}
					}
					GDScriptCodeGenerator::Address src_address = _parse_expression(codegen, error, lv->initializer);
					if (error) {
						return error;
					}
					if (lv->use_conversion_assign) {
						gen->write_assign_with_conversion(local, src_address);
					} else {
						gen->write_assign(local, src_address);
					}
					if (src_address.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						codegen.generator->pop_temporary();
					}
				} else if (lv->get_datatype().is_hard_type()) {
					// Initialize with default for type.
					if (local_type.has_container_element_type()) {
						codegen.generator->write_construct_typed_array(local, _gdtype_from_datatype(local_type.get_container_element_type(), codegen.script), Vector<GDScriptCodeGenerator::Address>());
					} else if (local_type.kind == GDScriptParser::DataType::BUILTIN) {
						codegen.generator->write_construct(local, local_type.builtin_type, Vector<GDScriptCodeGenerator::Address>());
					}
					// The `else` branch is for objects, in such case we leave it as `null`.
				}
			} break;
			case GDScriptParser::Node::CONSTANT: {
				// Local constants.
				const GDScriptParser::ConstantNode *lc = static_cast<const GDScriptParser::ConstantNode *>(s);
				if (!lc->initializer->is_constant) {
					_set_error("Local constant must have a constant value as initializer.", lc->initializer);
					return ERR_PARSE_ERROR;
				}

				codegen.add_local_constant(lc->identifier->name, lc->initializer->reduced_value);
			} break;
			case GDScriptParser::Node::PASS:
				// Nothing to do.
				break;
			default: {
				// Expression.
				if (s->is_expression()) {
					GDScriptCodeGenerator::Address expr = _parse_expression(codegen, error, static_cast<const GDScriptParser::ExpressionNode *>(s), true);
					if (error) {
						return error;
					}
					if (expr.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
						codegen.generator->pop_temporary();
					}
				} else {
					ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Bug in bytecode compiler, unexpected node in parse tree while parsing statement."); // Unreachable code.
				}
			} break;
		}
	}

	codegen.end_block();
	return OK;
}

GDScriptFunction *GDScriptCompiler::_parse_function(Error &r_error, GDScript *p_script, const GDScriptParser::ClassNode *p_class, const GDScriptParser::FunctionNode *p_func, bool p_for_ready, bool p_for_lambda) {
	r_error = OK;
	CodeGen codegen;
	codegen.generator = memnew(GDScriptByteCodeGenerator);

	codegen.class_node = p_class;
	codegen.script = p_script;
	codegen.function_node = p_func;

	StringName func_name;
	bool is_static = false;
	Multiplayer::RPCConfig rpc_config;
	GDScriptDataType return_type;
	return_type.has_type = true;
	return_type.kind = GDScriptDataType::BUILTIN;
	return_type.builtin_type = Variant::NIL;

	if (p_func) {
		if (p_func->identifier) {
			func_name = p_func->identifier->name;
		} else {
			func_name = "<anonymous lambda>";
		}
		is_static = p_func->is_static;
		rpc_config = p_func->rpc_config;
		return_type = _gdtype_from_datatype(p_func->get_datatype(), p_script);
	} else {
		if (p_for_ready) {
			func_name = "_ready";
		} else {
			func_name = "@implicit_new";
		}
	}

	codegen.function_name = func_name;
	codegen.generator->write_start(p_script, func_name, is_static, rpc_config, return_type);

	int optional_parameters = 0;

	if (p_func) {
		for (int i = 0; i < p_func->parameters.size(); i++) {
			const GDScriptParser::ParameterNode *parameter = p_func->parameters[i];
			GDScriptDataType par_type = _gdtype_from_datatype(parameter->get_datatype(), p_script);
			uint32_t par_addr = codegen.generator->add_parameter(parameter->identifier->name, parameter->default_value != nullptr, par_type);
			codegen.parameters[parameter->identifier->name] = GDScriptCodeGenerator::Address(GDScriptCodeGenerator::Address::FUNCTION_PARAMETER, par_addr, par_type);

			if (p_func->parameters[i]->default_value != nullptr) {
				optional_parameters++;
			}
		}
	}

	// Parse initializer if applies.
	bool is_implicit_initializer = !p_for_ready && !p_func && !p_for_lambda;
	bool is_initializer = p_func && !p_for_lambda && String(p_func->identifier->name) == GDScriptLanguage::get_singleton()->strings._init;
	bool is_for_ready = p_for_ready || (p_func && !p_for_lambda && String(p_func->identifier->name) == "_ready");

	if (!p_for_lambda && (is_implicit_initializer || is_for_ready)) {
		// Initialize class fields.
		for (int i = 0; i < p_class->members.size(); i++) {
			if (p_class->members[i].type != GDScriptParser::ClassNode::Member::VARIABLE) {
				continue;
			}
			const GDScriptParser::VariableNode *field = p_class->members[i].variable;
			if (field->onready != is_for_ready) {
				// Only initialize in _ready.
				continue;
			}

			GDScriptParser::DataType field_type = field->get_datatype();

			GDScriptCodeGenerator::Address dst_address(GDScriptCodeGenerator::Address::MEMBER, codegen.script->member_indices[field->identifier->name].index, _gdtype_from_datatype(field->get_datatype()));
			if (field->initializer) {
				// Emit proper line change.
				codegen.generator->write_newline(field->initializer->start_line);

				// For typed arrays we need to make sure this is already initialized correctly so typed assignment work.
				if (field_type.is_hard_type() && field_type.builtin_type == Variant::ARRAY && field_type.has_container_element_type()) {
					if (field_type.has_container_element_type()) {
						codegen.generator->write_construct_typed_array(dst_address, _gdtype_from_datatype(field_type.get_container_element_type(), codegen.script), Vector<GDScriptCodeGenerator::Address>());
					} else {
						codegen.generator->write_construct_array(dst_address, Vector<GDScriptCodeGenerator::Address>());
					}
				}
				GDScriptCodeGenerator::Address src_address = _parse_expression(codegen, r_error, field->initializer, false, true);
				if (r_error) {
					memdelete(codegen.generator);
					return nullptr;
				}

				if (field->use_conversion_assign) {
					codegen.generator->write_assign_with_conversion(dst_address, src_address);
				} else {
					codegen.generator->write_assign(dst_address, src_address);
				}
				if (src_address.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}
			} else if (field->get_datatype().is_hard_type()) {
				codegen.generator->write_newline(field->start_line);

				// Initialize with default for type.
				if (field_type.has_container_element_type()) {
					codegen.generator->write_construct_typed_array(dst_address, _gdtype_from_datatype(field_type.get_container_element_type(), codegen.script), Vector<GDScriptCodeGenerator::Address>());
				} else if (field_type.kind == GDScriptParser::DataType::BUILTIN) {
					codegen.generator->write_construct(dst_address, field_type.builtin_type, Vector<GDScriptCodeGenerator::Address>());
				}
				// The `else` branch is for objects, in such case we leave it as `null`.
			}
		}
	}

	// Parse default argument code if applies.
	if (p_func) {
		if (optional_parameters > 0) {
			codegen.generator->start_parameters();
			for (int i = p_func->parameters.size() - optional_parameters; i < p_func->parameters.size(); i++) {
				const GDScriptParser::ParameterNode *parameter = p_func->parameters[i];
				GDScriptCodeGenerator::Address src_addr = _parse_expression(codegen, r_error, parameter->default_value, true);
				if (r_error) {
					memdelete(codegen.generator);
					return nullptr;
				}
				GDScriptCodeGenerator::Address dst_addr = codegen.parameters[parameter->identifier->name];
				codegen.generator->write_assign_default_parameter(dst_addr, src_addr);
				if (src_addr.mode == GDScriptCodeGenerator::Address::TEMPORARY) {
					codegen.generator->pop_temporary();
				}
			}
			codegen.generator->end_parameters();
		}

		r_error = _parse_block(codegen, p_func->body);
		if (r_error) {
			memdelete(codegen.generator);
			return nullptr;
		}
	}

#ifdef DEBUG_ENABLED
	if (EngineDebugger::is_active()) {
		String signature;
		// Path.
		if (!p_script->get_path().is_empty()) {
			signature += p_script->get_path();
		}
		// Location.
		if (p_func) {
			signature += "::" + itos(p_func->body->start_line);
		} else {
			signature += "::0";
		}

		// Function and class.

		if (p_class->identifier) {
			signature += "::" + String(p_class->identifier->name) + "." + String(func_name);
		} else {
			signature += "::" + String(func_name);
		}

		if (p_for_lambda) {
			signature += "(lambda)";
		}

		codegen.generator->set_signature(signature);
	}
#endif

	if (p_func) {
		codegen.generator->set_initial_line(p_func->start_line);
#ifdef TOOLS_ENABLED
		if (!p_for_lambda) {
			p_script->member_lines[func_name] = p_func->start_line;
			p_script->doc_functions[func_name] = p_func->doc_description;
		}
#endif
	} else {
		codegen.generator->set_initial_line(0);
	}

	GDScriptFunction *gd_function = codegen.generator->write_end();

	if (is_initializer) {
		p_script->initializer = gd_function;
	} else if (is_implicit_initializer) {
		p_script->implicit_initializer = gd_function;
	}

	if (p_func) {
		// if no return statement -> return type is void not unresolved Variant
		if (p_func->body->has_return) {
			gd_function->return_type = _gdtype_from_datatype(p_func->get_datatype());
		} else {
			gd_function->return_type = GDScriptDataType();
			gd_function->return_type.has_type = true;
			gd_function->return_type.kind = GDScriptDataType::BUILTIN;
			gd_function->return_type.builtin_type = Variant::NIL;
		}
#ifdef TOOLS_ENABLED
		gd_function->default_arg_values = p_func->default_arg_values;
#endif
	}

	if (!p_for_lambda) {
		p_script->member_functions[func_name] = gd_function;
	}

	memdelete(codegen.generator);

	return gd_function;
}

Error GDScriptCompiler::_parse_setter_getter(GDScript *p_script, const GDScriptParser::ClassNode *p_class, const GDScriptParser::VariableNode *p_variable, bool p_is_setter) {
	Error error = OK;

	GDScriptParser::FunctionNode *function;

	if (p_is_setter) {
		function = p_variable->setter;
	} else {
		function = p_variable->getter;
	}

	_parse_function(error, p_script, p_class, function);

	return error;
}

Error GDScriptCompiler::_parse_class_level(GDScript *p_script, const GDScriptParser::ClassNode *p_class, bool p_keep_state) {
	parsing_classes.insert(p_script);

	if (p_class->outer && p_class->outer->outer) {
		// Owner is not root
		if (!parsed_classes.has(p_script->_owner)) {
			if (parsing_classes.has(p_script->_owner)) {
				_set_error("Cyclic class reference for '" + String(p_class->identifier->name) + "'.", p_class);
				return ERR_PARSE_ERROR;
			}
			Error err = _parse_class_level(p_script->_owner, p_class->outer, p_keep_state);
			if (err) {
				return err;
			}
		}
	}

#ifdef TOOLS_ENABLED
	p_script->doc_functions.clear();
	p_script->doc_variables.clear();
	p_script->doc_constants.clear();
	p_script->doc_enums.clear();
	p_script->doc_signals.clear();
	p_script->doc_tutorials.clear();

	p_script->doc_brief_description = p_class->doc_brief_description;
	p_script->doc_description = p_class->doc_description;
	for (int i = 0; i < p_class->doc_tutorials.size(); i++) {
		DocData::TutorialDoc td;
		td.title = p_class->doc_tutorials[i].first;
		td.link = p_class->doc_tutorials[i].second;
		p_script->doc_tutorials.append(td);
	}
#endif

	p_script->native = Ref<GDScriptNativeClass>();
	p_script->base = Ref<GDScript>();
	p_script->_base = nullptr;
	p_script->members.clear();
	p_script->constants.clear();
	for (const KeyValue<StringName, GDScriptFunction *> &E : p_script->member_functions) {
		memdelete(E.value);
	}
	p_script->member_functions.clear();
	p_script->member_indices.clear();
	p_script->member_info.clear();
	p_script->_signals.clear();
	p_script->initializer = nullptr;

	p_script->tool = parser->is_tool();
	p_script->name = p_class->identifier ? p_class->identifier->name : "";

	if (!p_script->name.is_empty()) {
		if (ClassDB::class_exists(p_script->name) && ClassDB::is_class_exposed(p_script->name)) {
			_set_error("The class '" + p_script->name + "' shadows a native class", p_class);
			return ERR_ALREADY_EXISTS;
		}
	}

	Ref<GDScriptNativeClass> native;

	GDScriptDataType base_type = _gdtype_from_datatype(p_class->base_type);
	// Inheritance
	switch (base_type.kind) {
		case GDScriptDataType::NATIVE: {
			int native_idx = GDScriptLanguage::get_singleton()->get_global_map()[base_type.native_type];
			native = GDScriptLanguage::get_singleton()->get_global_array()[native_idx];
			ERR_FAIL_COND_V(native.is_null(), ERR_BUG);
			p_script->native = native;
		} break;
		case GDScriptDataType::GDSCRIPT: {
			Ref<GDScript> base = Ref<GDScript>(base_type.script_type);
			p_script->base = base;
			p_script->_base = base.ptr();

			if (p_class->base_type.kind == GDScriptParser::DataType::CLASS && p_class->base_type.class_type != nullptr) {
				if (p_class->base_type.script_path == main_script->path) {
					if (!parsed_classes.has(p_script->_base)) {
						if (parsing_classes.has(p_script->_base)) {
							String class_name = p_class->identifier ? p_class->identifier->name : "<main>";
							_set_error("Cyclic class reference for '" + class_name + "'.", p_class);
							return ERR_PARSE_ERROR;
						}
						Error err = _parse_class_level(p_script->_base, p_class->base_type.class_type, p_keep_state);
						if (err) {
							return err;
						}
					}
				} else {
					Error err = OK;
					base = GDScriptCache::get_full_script(p_class->base_type.script_path, err, main_script->path);
					if (err) {
						return err;
					}
					if (base.is_null() || !base->is_valid()) {
						return ERR_COMPILATION_FAILED;
					}
				}
			}

			p_script->member_indices = base->member_indices;
			native = base->native;
			p_script->native = native;
		} break;
		default: {
			_set_error("Parser bug: invalid inheritance.", p_class);
			return ERR_BUG;
		} break;
	}

	for (int i = 0; i < p_class->members.size(); i++) {
		const GDScriptParser::ClassNode::Member &member = p_class->members[i];
		switch (member.type) {
			case GDScriptParser::ClassNode::Member::VARIABLE: {
				const GDScriptParser::VariableNode *variable = member.variable;
				StringName name = variable->identifier->name;

				GDScript::MemberInfo minfo;
				minfo.index = p_script->member_indices.size();
				switch (variable->property) {
					case GDScriptParser::VariableNode::PROP_NONE:
						break; // Nothing to do.
					case GDScriptParser::VariableNode::PROP_SETGET:
						if (variable->setter_pointer != nullptr) {
							minfo.setter = variable->setter_pointer->name;
						}
						if (variable->getter_pointer != nullptr) {
							minfo.getter = variable->getter_pointer->name;
						}
						break;
					case GDScriptParser::VariableNode::PROP_INLINE:
						if (variable->setter != nullptr) {
							minfo.setter = "@" + variable->identifier->name + "_setter";
						}
						if (variable->getter != nullptr) {
							minfo.getter = "@" + variable->identifier->name + "_getter";
						}
						break;
				}
				minfo.data_type = _gdtype_from_datatype(variable->get_datatype(), p_script);

				PropertyInfo prop_info = minfo.data_type;
				prop_info.name = name;
				PropertyInfo export_info = variable->export_info;

				if (variable->exported) {
					if (!minfo.data_type.has_type) {
						prop_info.type = export_info.type;
						prop_info.class_name = export_info.class_name;
					}
					prop_info.hint = export_info.hint;
					prop_info.hint_string = export_info.hint_string;
					prop_info.usage = export_info.usage | PROPERTY_USAGE_SCRIPT_VARIABLE;
				} else {
					prop_info.usage = PROPERTY_USAGE_SCRIPT_VARIABLE;
				}
#ifdef TOOLS_ENABLED
				p_script->doc_variables[name] = variable->doc_description;
#endif

				p_script->member_info[name] = prop_info;
				p_script->member_indices[name] = minfo;
				p_script->members.insert(name);

#ifdef TOOLS_ENABLED
				if (variable->initializer != nullptr && variable->initializer->is_constant) {
					p_script->member_default_values[name] = variable->initializer->reduced_value;
				} else {
					p_script->member_default_values.erase(name);
				}
				p_script->member_lines[name] = variable->start_line;
#endif
			} break;

			case GDScriptParser::ClassNode::Member::CONSTANT: {
				const GDScriptParser::ConstantNode *constant = member.constant;
				StringName name = constant->identifier->name;

				p_script->constants.insert(name, constant->initializer->reduced_value);
#ifdef TOOLS_ENABLED
				p_script->member_lines[name] = constant->start_line;
				if (!constant->doc_description.is_empty()) {
					p_script->doc_constants[name] = constant->doc_description;
				}
#endif
			} break;

			case GDScriptParser::ClassNode::Member::ENUM_VALUE: {
				const GDScriptParser::EnumNode::Value &enum_value = member.enum_value;
				StringName name = enum_value.identifier->name;

				p_script->constants.insert(name, enum_value.value);
#ifdef TOOLS_ENABLED
				p_script->member_lines[name] = enum_value.identifier->start_line;
				if (!p_script->doc_enums.has("@unnamed_enums")) {
					p_script->doc_enums["@unnamed_enums"] = DocData::EnumDoc();
					p_script->doc_enums["@unnamed_enums"].name = "@unnamed_enums";
				}
				DocData::ConstantDoc const_doc;
				const_doc.name = enum_value.identifier->name;
				const_doc.value = Variant(enum_value.value).operator String(); // TODO-DOC: enum value currently is int.
				const_doc.description = enum_value.doc_description;
				p_script->doc_enums["@unnamed_enums"].values.push_back(const_doc);
#endif
			} break;

			case GDScriptParser::ClassNode::Member::SIGNAL: {
				const GDScriptParser::SignalNode *signal = member.signal;
				StringName name = signal->identifier->name;

				Vector<StringName> parameters_names;
				parameters_names.resize(signal->parameters.size());
				for (int j = 0; j < signal->parameters.size(); j++) {
					parameters_names.write[j] = signal->parameters[j]->identifier->name;
				}
				p_script->_signals[name] = parameters_names;
#ifdef TOOLS_ENABLED
				if (!signal->doc_description.is_empty()) {
					p_script->doc_signals[name] = signal->doc_description;
				}
#endif
			} break;

			case GDScriptParser::ClassNode::Member::ENUM: {
				const GDScriptParser::EnumNode *enum_n = member.m_enum;

				// TODO: Make enums not be just a dictionary?
				Dictionary new_enum;
				for (int j = 0; j < enum_n->values.size(); j++) {
					int value = enum_n->values[j].value;
					// Needs to be string because Variant::get will convert to String.
					new_enum[String(enum_n->values[j].identifier->name)] = value;
				}

				p_script->constants.insert(enum_n->identifier->name, new_enum);
#ifdef TOOLS_ENABLED
				p_script->member_lines[enum_n->identifier->name] = enum_n->start_line;
				p_script->doc_enums[enum_n->identifier->name] = DocData::EnumDoc();
				p_script->doc_enums[enum_n->identifier->name].name = enum_n->identifier->name;
				p_script->doc_enums[enum_n->identifier->name].description = enum_n->doc_description;
				for (int j = 0; j < enum_n->values.size(); j++) {
					DocData::ConstantDoc const_doc;
					const_doc.name = enum_n->values[j].identifier->name;
					const_doc.value = Variant(enum_n->values[j].value).operator String();
					const_doc.description = enum_n->values[j].doc_description;
					p_script->doc_enums[enum_n->identifier->name].values.push_back(const_doc);
				}
#endif
			} break;
			default:
				break; // Nothing to do here.
		}
	}

	parsed_classes.insert(p_script);
	parsing_classes.erase(p_script);

	//parse sub-classes

	for (int i = 0; i < p_class->members.size(); i++) {
		const GDScriptParser::ClassNode::Member &member = p_class->members[i];
		if (member.type != member.CLASS) {
			continue;
		}
		const GDScriptParser::ClassNode *inner_class = member.m_class;
		StringName name = inner_class->identifier->name;
		Ref<GDScript> &subclass = p_script->subclasses[name];
		GDScript *subclass_ptr = subclass.ptr();

		// Subclass might still be parsing, just skip it
		if (!parsed_classes.has(subclass_ptr) && !parsing_classes.has(subclass_ptr)) {
			Error err = _parse_class_level(subclass_ptr, inner_class, p_keep_state);
			if (err) {
				return err;
			}
		}

#ifdef TOOLS_ENABLED

		p_script->member_lines[name] = inner_class->start_line;
#endif

		p_script->constants.insert(name, subclass); //once parsed, goes to the list of constants
	}

	return OK;
}

Error GDScriptCompiler::_parse_class_blocks(GDScript *p_script, const GDScriptParser::ClassNode *p_class, bool p_keep_state) {
	//parse methods

	bool has_ready = false;

	for (int i = 0; i < p_class->members.size(); i++) {
		const GDScriptParser::ClassNode::Member &member = p_class->members[i];
		if (member.type == member.FUNCTION) {
			const GDScriptParser::FunctionNode *function = member.function;
			if (!has_ready && function->identifier->name == "_ready") {
				has_ready = true;
			}
			Error err = OK;
			_parse_function(err, p_script, p_class, function);
			if (err) {
				return err;
			}
		} else if (member.type == member.VARIABLE) {
			const GDScriptParser::VariableNode *variable = member.variable;
			if (variable->property == GDScriptParser::VariableNode::PROP_INLINE) {
				if (variable->setter != nullptr) {
					Error err = _parse_setter_getter(p_script, p_class, variable, true);
					if (err) {
						return err;
					}
				}
				if (variable->getter != nullptr) {
					Error err = _parse_setter_getter(p_script, p_class, variable, false);
					if (err) {
						return err;
					}
				}
			}
		}
	}

	{
		// Create an implicit constructor in any case.
		Error err = OK;
		_parse_function(err, p_script, p_class, nullptr);
		if (err) {
			return err;
		}
	}

	if (!has_ready && p_class->onready_used) {
		//create a _ready constructor
		Error err = OK;
		_parse_function(err, p_script, p_class, nullptr, true);
		if (err) {
			return err;
		}
	}

#ifdef DEBUG_ENABLED

	//validate instances if keeping state

	if (p_keep_state) {
		for (Set<Object *>::Element *E = p_script->instances.front(); E;) {
			Set<Object *>::Element *N = E->next();

			ScriptInstance *si = E->get()->get_script_instance();
			if (si->is_placeholder()) {
#ifdef TOOLS_ENABLED
				PlaceHolderScriptInstance *psi = static_cast<PlaceHolderScriptInstance *>(si);

				if (p_script->is_tool()) {
					//re-create as an instance
					p_script->placeholders.erase(psi); //remove placeholder

					GDScriptInstance *instance = memnew(GDScriptInstance);
					instance->base_ref_counted = Object::cast_to<RefCounted>(E->get());
					instance->members.resize(p_script->member_indices.size());
					instance->script = Ref<GDScript>(p_script);
					instance->owner = E->get();

					//needed for hot reloading
					for (const KeyValue<StringName, GDScript::MemberInfo> &F : p_script->member_indices) {
						instance->member_indices_cache[F.key] = F.value.index;
					}
					instance->owner->set_script_instance(instance);

					/* STEP 2, INITIALIZE AND CONSTRUCT */

					Callable::CallError ce;
					p_script->initializer->call(instance, nullptr, 0, ce);

					if (ce.error != Callable::CallError::CALL_OK) {
						//well, tough luck, not gonna do anything here
					}
				}
#endif
			} else {
				GDScriptInstance *gi = static_cast<GDScriptInstance *>(si);
				gi->reload_members();
			}

			E = N;
		}
	}
#endif

	for (int i = 0; i < p_class->members.size(); i++) {
		if (p_class->members[i].type != GDScriptParser::ClassNode::Member::CLASS) {
			continue;
		}
		const GDScriptParser::ClassNode *inner_class = p_class->members[i].m_class;
		StringName name = inner_class->identifier->name;
		GDScript *subclass = p_script->subclasses[name].ptr();

		Error err = _parse_class_blocks(subclass, inner_class, p_keep_state);
		if (err) {
			return err;
		}
	}

	p_script->valid = true;
	return OK;
}

void GDScriptCompiler::_make_scripts(GDScript *p_script, const GDScriptParser::ClassNode *p_class, bool p_keep_state) {
	Map<StringName, Ref<GDScript>> old_subclasses;

	if (p_keep_state) {
		old_subclasses = p_script->subclasses;
	}

	p_script->subclasses.clear();

	for (int i = 0; i < p_class->members.size(); i++) {
		if (p_class->members[i].type != GDScriptParser::ClassNode::Member::CLASS) {
			continue;
		}
		const GDScriptParser::ClassNode *inner_class = p_class->members[i].m_class;
		StringName name = inner_class->identifier->name;

		Ref<GDScript> subclass;
		String fully_qualified_name = p_script->fully_qualified_name + "::" + name;

		if (old_subclasses.has(name)) {
			subclass = old_subclasses[name];
		} else {
			Ref<GDScript> orphan_subclass = GDScriptLanguage::get_singleton()->get_orphan_subclass(fully_qualified_name);
			if (orphan_subclass.is_valid()) {
				subclass = orphan_subclass;
			} else {
				subclass.instantiate();
			}
		}

		subclass->_owner = p_script;
		subclass->fully_qualified_name = fully_qualified_name;
		p_script->subclasses.insert(name, subclass);

		_make_scripts(subclass.ptr(), inner_class, false);
	}
}

Error GDScriptCompiler::compile(const GDScriptParser *p_parser, GDScript *p_script, bool p_keep_state) {
	err_line = -1;
	err_column = -1;
	error = "";
	parser = p_parser;
	main_script = p_script;
	const GDScriptParser::ClassNode *root = parser->get_tree();

	source = p_script->get_path();

	// The best fully qualified name for a base level script is its file path
	p_script->fully_qualified_name = p_script->path;

	// Create scripts for subclasses beforehand so they can be referenced
	_make_scripts(p_script, root, p_keep_state);

	p_script->_owner = nullptr;
	Error err = _parse_class_level(p_script, root, p_keep_state);

	if (err) {
		return err;
	}

	err = _parse_class_blocks(p_script, root, p_keep_state);

	if (err) {
		return err;
	}

	return GDScriptCache::finish_compiling(p_script->get_path());
}

String GDScriptCompiler::get_error() const {
	return error;
}

int GDScriptCompiler::get_error_line() const {
	return err_line;
}

int GDScriptCompiler::get_error_column() const {
	return err_column;
}

GDScriptCompiler::GDScriptCompiler() {
}
