/*************************************************************************/
/*  gdscript_compiler.cpp                                                */
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

#include "gdscript_compiler.h"

#include "gdscript.h"
#include "gdscript_cache.h"

bool GDScriptCompiler::_is_class_member_property(CodeGen &codegen, const StringName &p_name) {
	if (codegen.function_node && codegen.function_node->is_static) {
		return false;
	}

	if (codegen.stack_identifiers.has(p_name)) {
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
	if (error != "") {
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

bool GDScriptCompiler::_create_unary_operator(CodeGen &codegen, const GDScriptParser::UnaryOpNode *on, Variant::Operator op, int p_stack_level) {
	int src_address_a = _parse_expression(codegen, on->operand, p_stack_level);
	if (src_address_a < 0) {
		return false;
	}

	codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR); // perform operator
	codegen.opcodes.push_back(op); //which operator
	codegen.opcodes.push_back(src_address_a); // argument 1
	codegen.opcodes.push_back(src_address_a); // argument 2 (repeated)
	//codegen.opcodes.push_back(GDScriptFunction::ADDR_TYPE_NIL); // argument 2 (unary only takes one parameter)
	return true;
}

bool GDScriptCompiler::_create_binary_operator(CodeGen &codegen, const GDScriptParser::ExpressionNode *p_left_operand, const GDScriptParser::ExpressionNode *p_right_operand, Variant::Operator op, int p_stack_level, bool p_initializer, int p_index_addr) {
	int src_address_a = _parse_expression(codegen, p_left_operand, p_stack_level, false, p_initializer, p_index_addr);
	if (src_address_a < 0) {
		return false;
	}
	if (src_address_a & GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS) {
		p_stack_level++; //uses stack for return, increase stack
	}

	int src_address_b = _parse_expression(codegen, p_right_operand, p_stack_level, false, p_initializer);
	if (src_address_b < 0) {
		return false;
	}

	codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR); // perform operator
	codegen.opcodes.push_back(op); //which operator
	codegen.opcodes.push_back(src_address_a); // argument 1
	codegen.opcodes.push_back(src_address_b); // argument 2 (unary only takes one parameter)
	return true;
}

bool GDScriptCompiler::_create_binary_operator(CodeGen &codegen, const GDScriptParser::BinaryOpNode *on, Variant::Operator op, int p_stack_level, bool p_initializer, int p_index_addr) {
	return _create_binary_operator(codegen, on->left_operand, on->right_operand, op, p_stack_level, p_initializer, p_index_addr);
}

GDScriptDataType GDScriptCompiler::_gdtype_from_datatype(const GDScriptParser::DataType &p_datatype) const {
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
			result.script_type = p_datatype.script_type;
			result.native_type = result.script_type->get_instance_base_type();
		} break;
		case GDScriptParser::DataType::CLASS: {
			// Locate class by constructing the path to it and following that path
			GDScriptParser::ClassNode *class_type = p_datatype.class_type;
			if (class_type) {
				if (class_type->fqcn.begins_with(main_script->path) || (!main_script->name.empty() && class_type->fqcn.begins_with(main_script->name))) {
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
					result.script_type = script;
					result.native_type = script->get_instance_base_type();
				} else {
					result.kind = GDScriptDataType::GDSCRIPT;
					result.script_type = GDScriptCache::get_shallow_script(p_datatype.script_path, main_script->path);
					result.native_type = p_datatype.native_type;
				}
			}
		} break;
		case GDScriptParser::DataType::ENUM_VALUE:
			result.has_type = true;
			result.kind = GDScriptDataType::BUILTIN;
			result.builtin_type = Variant::INT;
			break;
		case GDScriptParser::DataType::ENUM:
			result.has_type = true;
			result.kind = GDScriptDataType::BUILTIN;
			result.builtin_type = Variant::DICTIONARY;
			break;
		case GDScriptParser::DataType::UNRESOLVED: {
			ERR_PRINT("Parser bug: converting unresolved type.");
			return GDScriptDataType();
		}
	}

	return result;
}

int GDScriptCompiler::_parse_assign_right_expression(CodeGen &codegen, const GDScriptParser::AssignmentNode *p_assignment, int p_stack_level, int p_index_addr) {
	Variant::Operator var_op = Variant::OP_MAX;

	switch (p_assignment->operation) {
		case GDScriptParser::AssignmentNode::OP_ADDITION:
			var_op = Variant::OP_ADD;
			break;
		case GDScriptParser::AssignmentNode::OP_SUBTRACTION:
			var_op = Variant::OP_SUBTRACT;
			break;
		case GDScriptParser::AssignmentNode::OP_MULTIPLICATION:
			var_op = Variant::OP_MULTIPLY;
			break;
		case GDScriptParser::AssignmentNode::OP_DIVISION:
			var_op = Variant::OP_DIVIDE;
			break;
		case GDScriptParser::AssignmentNode::OP_MODULO:
			var_op = Variant::OP_MODULE;
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_SHIFT_LEFT:
			var_op = Variant::OP_SHIFT_LEFT;
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_SHIFT_RIGHT:
			var_op = Variant::OP_SHIFT_RIGHT;
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_AND:
			var_op = Variant::OP_BIT_AND;
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_OR:
			var_op = Variant::OP_BIT_OR;
			break;
		case GDScriptParser::AssignmentNode::OP_BIT_XOR:
			var_op = Variant::OP_BIT_XOR;
			break;
		case GDScriptParser::AssignmentNode::OP_NONE: {
			//none
		} break;
		default: {
			ERR_FAIL_V(-1);
		}
	}

	// bool initializer = p_expression->op == GDScriptParser::OperatorNode::OP_INIT_ASSIGN;

	if (var_op == Variant::OP_MAX) {
		return _parse_expression(codegen, p_assignment->assigned_value, p_stack_level, false, false);
	}

	if (!_create_binary_operator(codegen, p_assignment->assignee, p_assignment->assigned_value, var_op, p_stack_level, false, p_index_addr)) {
		return -1;
	}

	int dst_addr = (p_stack_level) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
	codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
	codegen.alloc_stack(p_stack_level);
	return dst_addr;
}

bool GDScriptCompiler::_generate_typed_assign(CodeGen &codegen, int p_src_address, int p_dst_address, const GDScriptDataType &p_datatype, const GDScriptParser::DataType &p_value_type) {
	if (p_datatype.has_type && p_value_type.is_variant()) {
		// Typed assignment
		switch (p_datatype.kind) {
			case GDScriptDataType::BUILTIN: {
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN); // perform operator
				codegen.opcodes.push_back(p_datatype.builtin_type); // variable type
				codegen.opcodes.push_back(p_dst_address); // argument 1
				codegen.opcodes.push_back(p_src_address); // argument 2
			} break;
			case GDScriptDataType::NATIVE: {
				int class_idx;
				if (GDScriptLanguage::get_singleton()->get_global_map().has(p_datatype.native_type)) {
					class_idx = GDScriptLanguage::get_singleton()->get_global_map()[p_datatype.native_type];
					class_idx |= (GDScriptFunction::ADDR_TYPE_GLOBAL << GDScriptFunction::ADDR_BITS); //argument (stack root)
				} else {
					// _set_error("Invalid native class type '" + String(p_datatype.native_type) + "'.", on->arguments[0]);
					return false;
				}
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN_TYPED_NATIVE); // perform operator
				codegen.opcodes.push_back(class_idx); // variable type
				codegen.opcodes.push_back(p_dst_address); // argument 1
				codegen.opcodes.push_back(p_src_address); // argument 2
			} break;
			case GDScriptDataType::SCRIPT:
			case GDScriptDataType::GDSCRIPT: {
				Variant script = p_datatype.script_type;
				int idx = codegen.get_constant_pos(script); //make it a local constant (faster access)

				codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN_TYPED_SCRIPT); // perform operator
				codegen.opcodes.push_back(idx); // variable type
				codegen.opcodes.push_back(p_dst_address); // argument 1
				codegen.opcodes.push_back(p_src_address); // argument 2
			} break;
			default: {
				ERR_PRINT("Compiler bug: unresolved assign.");

				// Shouldn't get here, but fail-safe to a regular assignment
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN); // perform operator
				codegen.opcodes.push_back(p_dst_address); // argument 1
				codegen.opcodes.push_back(p_src_address); // argument 2 (unary only takes one parameter)
			}
		}
	} else {
		if (p_datatype.kind == GDScriptDataType::BUILTIN && p_value_type.kind == GDScriptParser::DataType::BUILTIN && p_datatype.builtin_type != p_value_type.builtin_type) {
			// Need conversion.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN_TYPED_BUILTIN); // perform operator
			codegen.opcodes.push_back(p_datatype.builtin_type); // variable type
			codegen.opcodes.push_back(p_dst_address); // argument 1
			codegen.opcodes.push_back(p_src_address); // argument 2
		} else {
			// Either untyped assignment or already type-checked by the parser
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN); // perform operator
			codegen.opcodes.push_back(p_dst_address); // argument 1
			codegen.opcodes.push_back(p_src_address); // argument 2 (unary only takes one parameter)
		}
	}
	return true;
}

int GDScriptCompiler::_parse_expression(CodeGen &codegen, const GDScriptParser::ExpressionNode *p_expression, int p_stack_level, bool p_root, bool p_initializer, int p_index_addr) {
	if (p_expression->is_constant) {
		return codegen.get_constant_pos(p_expression->reduced_value);
	}

	switch (p_expression->type) {
		//should parse variable declaration and adjust stack accordingly...
		case GDScriptParser::Node::IDENTIFIER: {
			//return identifier
			//wait, identifier could be a local variable or something else... careful here, must reference properly
			//as stack may be more interesting to work with

			//This could be made much simpler by just indexing "self", but done this way (with custom self-addressing modes) increases performance a lot.

			const GDScriptParser::IdentifierNode *in = static_cast<const GDScriptParser::IdentifierNode *>(p_expression);

			StringName identifier = in->name;

			// TRY STACK!
			if (!p_initializer && codegen.stack_identifiers.has(identifier)) {
				int pos = codegen.stack_identifiers[identifier];
				return pos | (GDScriptFunction::ADDR_TYPE_STACK_VARIABLE << GDScriptFunction::ADDR_BITS);
			}

			// TRY LOCAL CONSTANTS!
			if (codegen.local_named_constants.has(identifier)) {
				return codegen.local_named_constants[identifier] | (GDScriptFunction::ADDR_TYPE_LOCAL_CONSTANT << GDScriptFunction::ADDR_BITS);
			}

			// TRY CLASS MEMBER
			if (_is_class_member_property(codegen, identifier)) {
				//get property
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_GET_MEMBER); // perform operator
				codegen.opcodes.push_back(codegen.get_name_map_pos(identifier)); // argument 2 (unary only takes one parameter)
				int dst_addr = (p_stack_level) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
				codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
				codegen.alloc_stack(p_stack_level);
				return dst_addr;
			}

			//TRY MEMBERS!
			if (!codegen.function_node || !codegen.function_node->is_static) {
				// TRY MEMBER VARIABLES!
				//static function
				if (codegen.script->member_indices.has(identifier)) {
					if (codegen.script->member_indices[identifier].getter != StringName() && codegen.script->member_indices[identifier].getter != codegen.function_name) {
						// Perform getter.
						codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL_RETURN);
						codegen.opcodes.push_back(0); // Argument count.
						codegen.opcodes.push_back(GDScriptFunction::ADDR_TYPE_SELF << GDScriptFunction::ADDR_BITS); // Base (self).
						codegen.opcodes.push_back(codegen.get_name_map_pos(codegen.script->member_indices[identifier].getter)); // Method name.
						// Destination.
						int dst_addr = (p_stack_level) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
						codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
						codegen.alloc_stack(p_stack_level);
						return dst_addr;
					} else {
						// No getter or inside getter: direct member access.
						int idx = codegen.script->member_indices[identifier].index;
						return idx | (GDScriptFunction::ADDR_TYPE_MEMBER << GDScriptFunction::ADDR_BITS); //argument (stack root)
					}
				}
			}

			//TRY CLASS CONSTANTS

			GDScript *owner = codegen.script;
			while (owner) {
				GDScript *scr = owner;
				GDScriptNativeClass *nc = nullptr;
				while (scr) {
					if (scr->constants.has(identifier)) {
						//int idx=scr->constants[identifier];
						int idx = codegen.get_name_map_pos(identifier);
						return idx | (GDScriptFunction::ADDR_TYPE_CLASS_CONSTANT << GDScriptFunction::ADDR_BITS); //argument (stack root)
					}
					if (scr->native.is_valid()) {
						nc = scr->native.ptr();
					}
					scr = scr->_base;
				}

				// CLASS C++ Integer Constant

				if (nc) {
					bool success = false;
					int constant = ClassDB::get_integer_constant(nc->get_name(), identifier, &success);
					if (success) {
						Variant key = constant;
						int idx;

						if (!codegen.constant_map.has(key)) {
							idx = codegen.constant_map.size();
							codegen.constant_map[key] = idx;

						} else {
							idx = codegen.constant_map[key];
						}

						return idx | (GDScriptFunction::ADDR_TYPE_LOCAL_CONSTANT << GDScriptFunction::ADDR_BITS); //make it a local constant (faster access)
					}
				}

				owner = owner->_owner;
			}

			// TRY SIGNALS AND METHODS (can be made callables)
			if (codegen.class_node->members_indices.has(identifier)) {
				const GDScriptParser::ClassNode::Member &member = codegen.class_node->members[codegen.class_node->members_indices[identifier]];
				if (member.type == GDScriptParser::ClassNode::Member::FUNCTION || member.type == GDScriptParser::ClassNode::Member::SIGNAL) {
					// Get like it was a property.
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_GET_NAMED); // perform operator
					codegen.opcodes.push_back(GDScriptFunction::ADDR_TYPE_SELF << GDScriptFunction::ADDR_BITS); // Self.
					codegen.opcodes.push_back(codegen.get_name_map_pos(identifier)); // argument 2 (unary only takes one parameter)
					int dst_addr = (p_stack_level) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
					codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
					codegen.alloc_stack(p_stack_level);
					return dst_addr;
				}
			}

			if (GDScriptLanguage::get_singleton()->get_global_map().has(identifier)) {
				int idx = GDScriptLanguage::get_singleton()->get_global_map()[identifier];
				return idx | (GDScriptFunction::ADDR_TYPE_GLOBAL << GDScriptFunction::ADDR_BITS); //argument (stack root)
			}

			/* TRY GLOBAL CLASSES */

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
						return -1;
					}
				}

				Variant key = res;
				int idx;

				if (!codegen.constant_map.has(key)) {
					idx = codegen.constant_map.size();
					codegen.constant_map[key] = idx;

				} else {
					idx = codegen.constant_map[key];
				}

				return idx | (GDScriptFunction::ADDR_TYPE_LOCAL_CONSTANT << GDScriptFunction::ADDR_BITS); //make it a local constant (faster access)
			}

#ifdef TOOLS_ENABLED
			if (GDScriptLanguage::get_singleton()->get_named_globals_map().has(identifier)) {
				int idx = codegen.named_globals.find(identifier);
				if (idx == -1) {
					idx = codegen.named_globals.size();
					codegen.named_globals.push_back(identifier);
				}
				return idx | (GDScriptFunction::ADDR_TYPE_NAMED_GLOBAL << GDScriptFunction::ADDR_BITS);
			}
#endif

			//not found, error

			_set_error("Identifier not found: " + String(identifier), p_expression);

			return -1;

		} break;
		case GDScriptParser::Node::LITERAL: {
			//return constant
			const GDScriptParser::LiteralNode *cn = static_cast<const GDScriptParser::LiteralNode *>(p_expression);

			int idx;

			if (!codegen.constant_map.has(cn->value)) {
				idx = codegen.constant_map.size();
				codegen.constant_map[cn->value] = idx;

			} else {
				idx = codegen.constant_map[cn->value];
			}

			return idx | (GDScriptFunction::ADDR_TYPE_LOCAL_CONSTANT << GDScriptFunction::ADDR_BITS); //argument (stack root)

		} break;
		case GDScriptParser::Node::SELF: {
			//return constant
			if (codegen.function_node && codegen.function_node->is_static) {
				_set_error("'self' not present in static function!", p_expression);
				return -1;
			}
			return (GDScriptFunction::ADDR_TYPE_SELF << GDScriptFunction::ADDR_BITS);
		} break;
		case GDScriptParser::Node::ARRAY: {
			const GDScriptParser::ArrayNode *an = static_cast<const GDScriptParser::ArrayNode *>(p_expression);
			Vector<int> values;

			int slevel = p_stack_level;

			for (int i = 0; i < an->elements.size(); i++) {
				int ret = _parse_expression(codegen, an->elements[i], slevel);
				if (ret < 0) {
					return ret;
				}
				if ((ret >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
					slevel++;
					codegen.alloc_stack(slevel);
				}

				values.push_back(ret);
			}

			codegen.opcodes.push_back(GDScriptFunction::OPCODE_CONSTRUCT_ARRAY);
			codegen.opcodes.push_back(values.size());
			for (int i = 0; i < values.size(); i++) {
				codegen.opcodes.push_back(values[i]);
			}

			int dst_addr = (p_stack_level) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
			codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
			codegen.alloc_stack(p_stack_level);
			return dst_addr;

		} break;
		case GDScriptParser::Node::DICTIONARY: {
			const GDScriptParser::DictionaryNode *dn = static_cast<const GDScriptParser::DictionaryNode *>(p_expression);
			Vector<int> elements;

			int slevel = p_stack_level;

			for (int i = 0; i < dn->elements.size(); i++) {
				// Key.
				int ret = -1;
				switch (dn->style) {
					case GDScriptParser::DictionaryNode::PYTHON_DICT:
						// Python-style: key is any expression.
						ret = _parse_expression(codegen, dn->elements[i].key, slevel);
						if (ret < 0) {
							return ret;
						}
						if ((ret >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
							slevel++;
							codegen.alloc_stack(slevel);
						}
						break;
					case GDScriptParser::DictionaryNode::LUA_TABLE:
						// Lua-style: key is an identifier interpreted as string.
						String key = static_cast<const GDScriptParser::IdentifierNode *>(dn->elements[i].key)->name;
						ret = codegen.get_constant_pos(key);
						break;
				}

				elements.push_back(ret);

				ret = _parse_expression(codegen, dn->elements[i].value, slevel);
				if (ret < 0) {
					return ret;
				}
				if ((ret >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
					slevel++;
					codegen.alloc_stack(slevel);
				}

				elements.push_back(ret);
			}

			codegen.opcodes.push_back(GDScriptFunction::OPCODE_CONSTRUCT_DICTIONARY);
			codegen.opcodes.push_back(dn->elements.size());
			for (int i = 0; i < elements.size(); i++) {
				codegen.opcodes.push_back(elements[i]);
			}

			int dst_addr = (p_stack_level) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
			codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
			codegen.alloc_stack(p_stack_level);
			return dst_addr;

		} break;
		case GDScriptParser::Node::CAST: {
			const GDScriptParser::CastNode *cn = static_cast<const GDScriptParser::CastNode *>(p_expression);

			int slevel = p_stack_level;
			int src_addr = _parse_expression(codegen, cn->operand, slevel);
			if (src_addr < 0) {
				return src_addr;
			}
			if (src_addr & GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS) {
				slevel++;
				codegen.alloc_stack(slevel);
			}

			GDScriptDataType cast_type = _gdtype_from_datatype(cn->cast_type->get_datatype());

			switch (cast_type.kind) {
				case GDScriptDataType::BUILTIN: {
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_CAST_TO_BUILTIN);
					codegen.opcodes.push_back(cast_type.builtin_type);
				} break;
				case GDScriptDataType::NATIVE: {
					int class_idx;
					if (GDScriptLanguage::get_singleton()->get_global_map().has(cast_type.native_type)) {
						class_idx = GDScriptLanguage::get_singleton()->get_global_map()[cast_type.native_type];
						class_idx |= (GDScriptFunction::ADDR_TYPE_GLOBAL << GDScriptFunction::ADDR_BITS); //argument (stack root)
					} else {
						_set_error("Invalid native class type '" + String(cast_type.native_type) + "'.", cn);
						return -1;
					}
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_CAST_TO_NATIVE); // perform operator
					codegen.opcodes.push_back(class_idx); // variable type
				} break;
				case GDScriptDataType::SCRIPT:
				case GDScriptDataType::GDSCRIPT: {
					Variant script = cast_type.script_type;
					int idx = codegen.get_constant_pos(script); //make it a local constant (faster access)

					codegen.opcodes.push_back(GDScriptFunction::OPCODE_CAST_TO_SCRIPT); // perform operator
					codegen.opcodes.push_back(idx); // variable type
				} break;
				default: {
					_set_error("Parser bug: unresolved data type.", cn);
					return -1;
				}
			}

			codegen.opcodes.push_back(src_addr); // source address
			int dst_addr = (p_stack_level) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
			codegen.opcodes.push_back(dst_addr); // append the stack level as destination address of the opcode
			codegen.alloc_stack(p_stack_level);
			return dst_addr;

		} break;
		//hell breaks loose

#define OPERATOR_RETURN                                                                                  \
	int dst_addr = (p_stack_level) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS); \
	codegen.opcodes.push_back(dst_addr);                                                                 \
	codegen.alloc_stack(p_stack_level);                                                                  \
	return dst_addr

		case GDScriptParser::Node::CALL: {
			const GDScriptParser::CallNode *call = static_cast<const GDScriptParser::CallNode *>(p_expression);
			if (!call->is_super && call->callee->type == GDScriptParser::Node::IDENTIFIER && GDScriptParser::get_builtin_type(static_cast<GDScriptParser::IdentifierNode *>(call->callee)->name) != Variant::VARIANT_MAX) {
				//construct a basic type

				Variant::Type vtype = GDScriptParser::get_builtin_type(static_cast<GDScriptParser::IdentifierNode *>(call->callee)->name);

				Vector<int> arguments;
				int slevel = p_stack_level;
				for (int i = 0; i < call->arguments.size(); i++) {
					int ret = _parse_expression(codegen, call->arguments[i], slevel);
					if (ret < 0) {
						return ret;
					}
					if ((ret >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
						slevel++;
						codegen.alloc_stack(slevel);
					}
					arguments.push_back(ret);
				}

				//push call bytecode
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_CONSTRUCT); // basic type constructor
				codegen.opcodes.push_back(vtype); //instance
				codegen.opcodes.push_back(arguments.size()); //argument count
				codegen.alloc_call(arguments.size());
				for (int i = 0; i < arguments.size(); i++) {
					codegen.opcodes.push_back(arguments[i]); //arguments
				}

			} else if (!call->is_super && call->callee->type == GDScriptParser::Node::IDENTIFIER && GDScriptParser::get_builtin_function(static_cast<GDScriptParser::IdentifierNode *>(call->callee)->name) != GDScriptFunctions::FUNC_MAX) {
				//built in function

				Vector<int> arguments;
				int slevel = p_stack_level;
				for (int i = 0; i < call->arguments.size(); i++) {
					int ret = _parse_expression(codegen, call->arguments[i], slevel);
					if (ret < 0) {
						return ret;
					}

					if ((ret >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
						slevel++;
						codegen.alloc_stack(slevel);
					}

					arguments.push_back(ret);
				}

				codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL_BUILT_IN);
				codegen.opcodes.push_back(GDScriptParser::get_builtin_function(static_cast<GDScriptParser::IdentifierNode *>(call->callee)->name));
				codegen.opcodes.push_back(arguments.size());
				codegen.alloc_call(arguments.size());
				for (int i = 0; i < arguments.size(); i++) {
					codegen.opcodes.push_back(arguments[i]);
				}

			} else {
				//regular function

				const GDScriptParser::ExpressionNode *callee = call->callee;

				Vector<int> arguments;
				int slevel = p_stack_level;

				// TODO: Use callables when possible if needed.
				int ret = -1;
				int super_address = -1;
				if (call->is_super) {
					// Super call.
					if (call->callee == nullptr) {
						// Implicit super function call.
						super_address = codegen.get_name_map_pos(codegen.function_node->identifier->name);
					} else {
						super_address = codegen.get_name_map_pos(static_cast<GDScriptParser::IdentifierNode *>(call->callee)->name);
					}
				} else {
					if (callee->type == GDScriptParser::Node::IDENTIFIER) {
						// Self function call.
						if ((codegen.function_node && codegen.function_node->is_static) || call->function_name == "new") {
							ret = (GDScriptFunction::ADDR_TYPE_CLASS << GDScriptFunction::ADDR_BITS);
						} else {
							ret = (GDScriptFunction::ADDR_TYPE_SELF << GDScriptFunction::ADDR_BITS);
						}
						arguments.push_back(ret);
						ret = codegen.get_name_map_pos(static_cast<GDScriptParser::IdentifierNode *>(call->callee)->name);
						arguments.push_back(ret);
					} else if (callee->type == GDScriptParser::Node::SUBSCRIPT) {
						const GDScriptParser::SubscriptNode *subscript = static_cast<const GDScriptParser::SubscriptNode *>(call->callee);

						if (subscript->is_attribute) {
							ret = _parse_expression(codegen, subscript->base, slevel);
							if (ret < 0) {
								return ret;
							}
							if ((ret >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
								slevel++;
								codegen.alloc_stack(slevel);
							}
							arguments.push_back(ret);
							arguments.push_back(codegen.get_name_map_pos(subscript->attribute->name));
						} else {
							_set_error("Cannot call something that isn't a function.", call->callee);
							return -1;
						}
					} else {
						_set_error("Cannot call something that isn't a function.", call->callee);
						return -1;
					}
				}

				for (int i = 0; i < call->arguments.size(); i++) {
					ret = _parse_expression(codegen, call->arguments[i], slevel);
					if (ret < 0) {
						return ret;
					}
					if ((ret >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
						slevel++;
						codegen.alloc_stack(slevel);
					}
					arguments.push_back(ret);
				}

				int opcode = GDScriptFunction::OPCODE_CALL_RETURN;
				if (call->is_super) {
					opcode = GDScriptFunction::OPCODE_CALL_SELF_BASE;
				} else if (within_await) {
					opcode = GDScriptFunction::OPCODE_CALL_ASYNC;
				} else if (p_root) {
					opcode = GDScriptFunction::OPCODE_CALL;
				}

				codegen.opcodes.push_back(opcode); // perform operator
				if (call->is_super) {
					codegen.opcodes.push_back(super_address);
				}
				codegen.opcodes.push_back(call->arguments.size());
				codegen.alloc_call(call->arguments.size());
				for (int i = 0; i < arguments.size(); i++) {
					codegen.opcodes.push_back(arguments[i]);
				}
			}
			OPERATOR_RETURN;
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

			int arg_address = codegen.get_constant_pos(NodePath(node_name));

			codegen.opcodes.push_back(p_root ? GDScriptFunction::OPCODE_CALL : GDScriptFunction::OPCODE_CALL_RETURN);
			codegen.opcodes.push_back(1); // number of arguments.
			codegen.alloc_call(1);
			codegen.opcodes.push_back(GDScriptFunction::ADDR_TYPE_SELF << GDScriptFunction::ADDR_BITS); // self.
			codegen.opcodes.push_back(codegen.get_name_map_pos("get_node")); // function.
			codegen.opcodes.push_back(arg_address); // argument (NodePath).
			OPERATOR_RETURN;
		} break;
		case GDScriptParser::Node::PRELOAD: {
			const GDScriptParser::PreloadNode *preload = static_cast<const GDScriptParser::PreloadNode *>(p_expression);

			// Add resource as constant.
			return codegen.get_constant_pos(preload->resource);
		} break;
		case GDScriptParser::Node::AWAIT: {
			const GDScriptParser::AwaitNode *await = static_cast<const GDScriptParser::AwaitNode *>(p_expression);

			int slevel = p_stack_level;
			within_await = true;
			int argument = _parse_expression(codegen, await->to_await, slevel);
			within_await = false;
			if (argument < 0) {
				return argument;
			}
			if ((argument >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
				slevel++;
				codegen.alloc_stack(slevel);
			}
			//push call bytecode
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_AWAIT);
			codegen.opcodes.push_back(argument);
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_AWAIT_RESUME);
			//next will be where to place the result :)

			OPERATOR_RETURN;
		} break;

		//indexing operator
		case GDScriptParser::Node::SUBSCRIPT: {
			int slevel = p_stack_level;

			const GDScriptParser::SubscriptNode *subscript = static_cast<const GDScriptParser::SubscriptNode *>(p_expression);

			int from = _parse_expression(codegen, subscript->base, slevel);
			if (from < 0) {
				return from;
			}

			bool named = subscript->is_attribute;
			int index;
			if (p_index_addr != 0) {
				index = p_index_addr;
			} else if (subscript->is_attribute) {
				if (subscript->base->type == GDScriptParser::Node::SELF && codegen.script) {
					GDScriptParser::IdentifierNode *identifier = subscript->attribute;
					const Map<StringName, GDScript::MemberInfo>::Element *MI = codegen.script->member_indices.find(identifier->name);

#ifdef DEBUG_ENABLED
					if (MI && MI->get().getter == codegen.function_name) {
						String n = identifier->name;
						_set_error("Must use '" + n + "' instead of 'self." + n + "' in getter.", identifier);
						return -1;
					}
#endif

					if (MI && MI->get().getter == "") {
						// Faster than indexing self (as if no self. had been used)
						return (MI->get().index) | (GDScriptFunction::ADDR_TYPE_MEMBER << GDScriptFunction::ADDR_BITS);
					}
				}

				index = codegen.get_name_map_pos(subscript->attribute->name);

			} else {
				if (subscript->index->type == GDScriptParser::Node::LITERAL && static_cast<const GDScriptParser::LiteralNode *>(subscript->index)->value.get_type() == Variant::STRING) {
					//also, somehow, named (speed up anyway)
					StringName name = static_cast<const GDScriptParser::LiteralNode *>(subscript->index)->value;
					index = codegen.get_name_map_pos(name);
					named = true;

				} else {
					//regular indexing
					if (from & GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS) {
						slevel++;
						codegen.alloc_stack(slevel);
					}

					index = _parse_expression(codegen, subscript->index, slevel);
					if (index < 0) {
						return index;
					}
				}
			}

			codegen.opcodes.push_back(named ? GDScriptFunction::OPCODE_GET_NAMED : GDScriptFunction::OPCODE_GET); // perform operator
			codegen.opcodes.push_back(from); // argument 1
			codegen.opcodes.push_back(index); // argument 2 (unary only takes one parameter)
			OPERATOR_RETURN;
		} break;
		case GDScriptParser::Node::UNARY_OPERATOR: {
			//unary operators
			const GDScriptParser::UnaryOpNode *unary = static_cast<const GDScriptParser::UnaryOpNode *>(p_expression);
			switch (unary->operation) {
				case GDScriptParser::UnaryOpNode::OP_NEGATIVE: {
					if (!_create_unary_operator(codegen, unary, Variant::OP_NEGATE, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::UnaryOpNode::OP_POSITIVE: {
					if (!_create_unary_operator(codegen, unary, Variant::OP_POSITIVE, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::UnaryOpNode::OP_LOGIC_NOT: {
					if (!_create_unary_operator(codegen, unary, Variant::OP_NOT, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::UnaryOpNode::OP_COMPLEMENT: {
					if (!_create_unary_operator(codegen, unary, Variant::OP_BIT_NEGATE, p_stack_level)) {
						return -1;
					}
				} break;
			}
			OPERATOR_RETURN;
		}
		case GDScriptParser::Node::BINARY_OPERATOR: {
			//binary operators (in precedence order)
			const GDScriptParser::BinaryOpNode *binary = static_cast<const GDScriptParser::BinaryOpNode *>(p_expression);

			switch (binary->operation) {
				case GDScriptParser::BinaryOpNode::OP_LOGIC_AND: {
					// AND operator with early out on failure

					int res = _parse_expression(codegen, binary->left_operand, p_stack_level);
					if (res < 0) {
						return res;
					}
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
					codegen.opcodes.push_back(res);
					int jump_fail_pos = codegen.opcodes.size();
					codegen.opcodes.push_back(0);

					res = _parse_expression(codegen, binary->right_operand, p_stack_level);
					if (res < 0) {
						return res;
					}

					codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
					codegen.opcodes.push_back(res);
					int jump_fail_pos2 = codegen.opcodes.size();
					codegen.opcodes.push_back(0);

					codegen.alloc_stack(p_stack_level); //it will be used..
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN_TRUE);
					codegen.opcodes.push_back(p_stack_level | GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
					codegen.opcodes.push_back(codegen.opcodes.size() + 3);
					codegen.opcodes.write[jump_fail_pos] = codegen.opcodes.size();
					codegen.opcodes.write[jump_fail_pos2] = codegen.opcodes.size();
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN_FALSE);
					codegen.opcodes.push_back(p_stack_level | GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
					return p_stack_level | GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;

				} break;
				case GDScriptParser::BinaryOpNode::OP_LOGIC_OR: {
					// OR operator with early out on success

					int res = _parse_expression(codegen, binary->left_operand, p_stack_level);
					if (res < 0) {
						return res;
					}
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF);
					codegen.opcodes.push_back(res);
					int jump_success_pos = codegen.opcodes.size();
					codegen.opcodes.push_back(0);

					res = _parse_expression(codegen, binary->right_operand, p_stack_level);
					if (res < 0) {
						return res;
					}

					codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF);
					codegen.opcodes.push_back(res);
					int jump_success_pos2 = codegen.opcodes.size();
					codegen.opcodes.push_back(0);

					codegen.alloc_stack(p_stack_level); //it will be used..
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN_FALSE);
					codegen.opcodes.push_back(p_stack_level | GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
					codegen.opcodes.push_back(codegen.opcodes.size() + 3);
					codegen.opcodes.write[jump_success_pos] = codegen.opcodes.size();
					codegen.opcodes.write[jump_success_pos2] = codegen.opcodes.size();
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN_TRUE);
					codegen.opcodes.push_back(p_stack_level | GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
					return p_stack_level | GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;

				} break;
				case GDScriptParser::BinaryOpNode::OP_TYPE_TEST: {
					int slevel = p_stack_level;

					int src_address_a = _parse_expression(codegen, binary->left_operand, slevel);
					if (src_address_a < 0) {
						return -1;
					}

					if (src_address_a & GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS) {
						slevel++; //uses stack for return, increase stack
					}

					int src_address_b = -1;
					bool builtin = false;
					if (binary->right_operand->type == GDScriptParser::Node::IDENTIFIER && GDScriptParser::get_builtin_type(static_cast<const GDScriptParser::IdentifierNode *>(binary->right_operand)->name) != Variant::VARIANT_MAX) {
						// `is` with builtin type
						builtin = true;
						src_address_b = (int)GDScriptParser::get_builtin_type(static_cast<const GDScriptParser::IdentifierNode *>(binary->right_operand)->name);
					} else {
						src_address_b = _parse_expression(codegen, binary->right_operand, slevel);
						if (src_address_b < 0) {
							return -1;
						}
					}

					codegen.opcodes.push_back(builtin ? GDScriptFunction::OPCODE_IS_BUILTIN : GDScriptFunction::OPCODE_EXTENDS_TEST); // perform operator
					codegen.opcodes.push_back(src_address_a); // argument 1
					codegen.opcodes.push_back(src_address_b); // argument 2 (unary only takes one parameter)

				} break;
				case GDScriptParser::BinaryOpNode::OP_CONTENT_TEST: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_IN, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_COMP_EQUAL: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_EQUAL, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_COMP_NOT_EQUAL: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_NOT_EQUAL, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_COMP_LESS: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_LESS, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_COMP_LESS_EQUAL: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_LESS_EQUAL, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_COMP_GREATER: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_GREATER, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_COMP_GREATER_EQUAL: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_GREATER_EQUAL, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_ADDITION: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_ADD, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_SUBTRACTION: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_SUBTRACT, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_MULTIPLICATION: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_MULTIPLY, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_DIVISION: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_DIVIDE, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_MODULO: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_MODULE, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_BIT_AND: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_BIT_AND, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_BIT_OR: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_BIT_OR, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_BIT_XOR: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_BIT_XOR, p_stack_level)) {
						return -1;
					}
				} break;
				//shift
				case GDScriptParser::BinaryOpNode::OP_BIT_LEFT_SHIFT: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_SHIFT_LEFT, p_stack_level)) {
						return -1;
					}
				} break;
				case GDScriptParser::BinaryOpNode::OP_BIT_RIGHT_SHIFT: {
					if (!_create_binary_operator(codegen, binary, Variant::OP_SHIFT_RIGHT, p_stack_level)) {
						return -1;
					}
				} break;
			}
			OPERATOR_RETURN;
		} break;
		// ternary operators
		case GDScriptParser::Node::TERNARY_OPERATOR: {
			// x IF a ELSE y operator with early out on failure

			const GDScriptParser::TernaryOpNode *ternary = static_cast<const GDScriptParser::TernaryOpNode *>(p_expression);
			int res = _parse_expression(codegen, ternary->condition, p_stack_level);
			if (res < 0) {
				return res;
			}
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
			codegen.opcodes.push_back(res);
			int jump_fail_pos = codegen.opcodes.size();
			codegen.opcodes.push_back(0);

			res = _parse_expression(codegen, ternary->true_expr, p_stack_level);
			if (res < 0) {
				return res;
			}

			codegen.alloc_stack(p_stack_level); //it will be used..
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN);
			codegen.opcodes.push_back(p_stack_level | GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
			codegen.opcodes.push_back(res);
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
			int jump_past_pos = codegen.opcodes.size();
			codegen.opcodes.push_back(0);

			codegen.opcodes.write[jump_fail_pos] = codegen.opcodes.size();
			res = _parse_expression(codegen, ternary->false_expr, p_stack_level);
			if (res < 0) {
				return res;
			}

			codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN);
			codegen.opcodes.push_back(p_stack_level | GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
			codegen.opcodes.push_back(res);

			codegen.opcodes.write[jump_past_pos] = codegen.opcodes.size();

			return p_stack_level | GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;

		} break;
		//assignment operators
		case GDScriptParser::Node::ASSIGNMENT: {
			const GDScriptParser::AssignmentNode *assignment = static_cast<const GDScriptParser::AssignmentNode *>(p_expression);

			if (assignment->assignee->type == GDScriptParser::Node::SUBSCRIPT) {
				// SET (chained) MODE!
				const GDScriptParser::SubscriptNode *subscript = static_cast<GDScriptParser::SubscriptNode *>(assignment->assignee);
#ifdef DEBUG_ENABLED
				if (subscript->is_attribute) {
					if (subscript->base->type == GDScriptParser::Node::SELF && codegen.script) {
						const Map<StringName, GDScript::MemberInfo>::Element *MI = codegen.script->member_indices.find(subscript->attribute->name);
						if (MI && MI->get().setter == codegen.function_name) {
							String n = subscript->attribute->name;
							_set_error("Must use '" + n + "' instead of 'self." + n + "' in setter.", subscript);
							return -1;
						}
					}
				}
#endif

				int slevel = p_stack_level;

				/* Find chain of sets */

				StringName assign_property;

				List<const GDScriptParser::SubscriptNode *> chain;

				{
					//create get/set chain
					const GDScriptParser::SubscriptNode *n = subscript;
					while (true) {
						chain.push_back(n);

						if (n->base->type != GDScriptParser::Node::SUBSCRIPT) {
							//check for a built-in property
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

				//get at (potential) root stack pos, so it can be returned
				int prev_pos = _parse_expression(codegen, chain.back()->get()->base, slevel);
				if (prev_pos < 0) {
					return prev_pos;
				}
				int retval = prev_pos;

				if (retval & GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS) {
					slevel++;
					codegen.alloc_stack(slevel);
				}

				Vector<int> setchain;

				if (assign_property != StringName()) {
					// recover and assign at the end, this allows stuff like
					// position.x+=2.0
					// in Node2D
					setchain.push_back(prev_pos);
					setchain.push_back(codegen.get_name_map_pos(assign_property));
					setchain.push_back(GDScriptFunction::OPCODE_SET_MEMBER);
				}

				for (List<const GDScriptParser::SubscriptNode *>::Element *E = chain.back(); E; E = E->prev()) {
					if (E == chain.front()) { //ignore first
						break;
					}

					const GDScriptParser::SubscriptNode *subscript_elem = E->get();
					int key_idx;

					if (subscript_elem->is_attribute) {
						key_idx = codegen.get_name_map_pos(subscript_elem->attribute->name);
						//printf("named key %x\n",key_idx);

					} else {
						if (prev_pos & (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS)) {
							slevel++;
							codegen.alloc_stack(slevel);
						}

						GDScriptParser::ExpressionNode *key = subscript_elem->index;
						key_idx = _parse_expression(codegen, key, slevel);
						//printf("expr key %x\n",key_idx);

						//stack was raised here if retval was stack but..
					}

					if (key_idx < 0) { //error
						return key_idx;
					}

					codegen.opcodes.push_back(subscript_elem->is_attribute ? GDScriptFunction::OPCODE_GET_NAMED : GDScriptFunction::OPCODE_GET);
					codegen.opcodes.push_back(prev_pos);
					codegen.opcodes.push_back(key_idx);
					slevel++;
					codegen.alloc_stack(slevel);
					int dst_pos = (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS) | slevel;

					codegen.opcodes.push_back(dst_pos);

					//add in reverse order, since it will be reverted

					setchain.push_back(dst_pos);
					setchain.push_back(key_idx);
					setchain.push_back(prev_pos);
					setchain.push_back(subscript_elem->is_attribute ? GDScriptFunction::OPCODE_SET_NAMED : GDScriptFunction::OPCODE_SET);

					prev_pos = dst_pos;
				}

				setchain.invert();

				int set_index;

				if (subscript->is_attribute) {
					set_index = codegen.get_name_map_pos(subscript->attribute->name);
				} else {
					set_index = _parse_expression(codegen, subscript->index, slevel + 1);
				}

				if (set_index < 0) { //error
					return set_index;
				}

				if (set_index & GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS) {
					slevel++;
					codegen.alloc_stack(slevel);
				}

				int set_value = _parse_assign_right_expression(codegen, assignment, slevel + 1, subscript->is_attribute ? 0 : set_index);
				if (set_value < 0) { //error
					return set_value;
				}

				codegen.opcodes.push_back(subscript->is_attribute ? GDScriptFunction::OPCODE_SET_NAMED : GDScriptFunction::OPCODE_SET);
				codegen.opcodes.push_back(prev_pos);
				codegen.opcodes.push_back(set_index);
				codegen.opcodes.push_back(set_value);

				for (int i = 0; i < setchain.size(); i++) {
					codegen.opcodes.push_back(setchain[i]);
				}

				return retval;

			} else if (assignment->assignee->type == GDScriptParser::Node::IDENTIFIER && _is_class_member_property(codegen, static_cast<GDScriptParser::IdentifierNode *>(assignment->assignee)->name)) {
				//assignment to member property

				int slevel = p_stack_level;

				int src_address = _parse_assign_right_expression(codegen, assignment, slevel);
				if (src_address < 0) {
					return -1;
				}

				StringName name = static_cast<GDScriptParser::IdentifierNode *>(assignment->assignee)->name;

				codegen.opcodes.push_back(GDScriptFunction::OPCODE_SET_MEMBER);
				codegen.opcodes.push_back(codegen.get_name_map_pos(name));
				codegen.opcodes.push_back(src_address);

				return GDScriptFunction::ADDR_TYPE_NIL << GDScriptFunction::ADDR_BITS;
			} else {
				//REGULAR ASSIGNMENT MODE!!

				int slevel = p_stack_level;
				int dst_address_a = -1;

				bool has_setter = false;
				bool is_in_setter = false;
				StringName setter_function;
				if (assignment->assignee->type == GDScriptParser::Node::IDENTIFIER) {
					StringName var_name = static_cast<const GDScriptParser::IdentifierNode *>(assignment->assignee)->name;
					if (!codegen.stack_identifiers.has(var_name) && codegen.script->member_indices.has(var_name)) {
						setter_function = codegen.script->member_indices[var_name].setter;
						if (setter_function != StringName()) {
							has_setter = true;
							is_in_setter = setter_function == codegen.function_name;
							dst_address_a = codegen.script->member_indices[var_name].index;
						}
					}
				}

				if (has_setter) {
					if (is_in_setter) {
						// Use direct member access.
						dst_address_a |= GDScriptFunction::ADDR_TYPE_MEMBER << GDScriptFunction::ADDR_BITS;
					} else {
						// Store stack slot for the temp value.
						dst_address_a = slevel++;
						codegen.alloc_stack(slevel);
						dst_address_a |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
					}
				} else {
					dst_address_a = _parse_expression(codegen, assignment->assignee, slevel);
					if (dst_address_a < 0) {
						return -1;
					}

					if (dst_address_a & GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS) {
						slevel++;
						codegen.alloc_stack(slevel);
					}
				}

				int src_address_b = _parse_assign_right_expression(codegen, assignment, slevel);
				if (src_address_b < 0) {
					return -1;
				}

				GDScriptDataType assign_type = _gdtype_from_datatype(assignment->assignee->get_datatype());

				if (has_setter && !is_in_setter) {
					// Call setter.
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL);
					codegen.opcodes.push_back(1); // Argument count.
					codegen.opcodes.push_back(GDScriptFunction::ADDR_TYPE_SELF << GDScriptFunction::ADDR_BITS); // Base (self).
					codegen.opcodes.push_back(codegen.get_name_map_pos(setter_function)); // Method name.
					codegen.opcodes.push_back(dst_address_a); // Argument.
					codegen.opcodes.push_back(dst_address_a); // Result address (won't be used here).
					codegen.alloc_call(1);
				} else if (!_generate_typed_assign(codegen, src_address_b, dst_address_a, assign_type, assignment->assigned_value->get_datatype())) {
					return -1;
				}

				return dst_address_a; //if anything, returns wathever was assigned or correct stack position
			}
		} break;
#undef OPERATOR_RETURN
		//TYPE_TYPE,
		default: {
			ERR_FAIL_V_MSG(-1, "Bug in bytecode compiler, unexpected node in parse tree while parsing expression."); //unreachable code
		} break;
	}
}

Error GDScriptCompiler::_parse_match_pattern(CodeGen &codegen, const GDScriptParser::PatternNode *p_pattern, int p_stack_level, int p_value_addr, int p_type_addr, int &r_bound_variables, Vector<int> &r_patch_addresses, Vector<int> &r_block_patch_address) {
	// TODO: Many "repeated" code here that could be abstracted. This compiler is going away when new VM arrives though, so...
	switch (p_pattern->pattern_type) {
		case GDScriptParser::PatternNode::PT_LITERAL: {
			// Get literal type into constant map.
			int literal_type_addr = -1;
			if (!codegen.constant_map.has((int)p_pattern->literal->value.get_type())) {
				literal_type_addr = codegen.constant_map.size();
				codegen.constant_map[(int)p_pattern->literal->value.get_type()] = literal_type_addr;

			} else {
				literal_type_addr = codegen.constant_map[(int)p_pattern->literal->value.get_type()];
			}
			literal_type_addr |= GDScriptFunction::ADDR_TYPE_LOCAL_CONSTANT << GDScriptFunction::ADDR_BITS;

			// Check type equality.
			int equality_addr = p_stack_level++;
			equality_addr |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
			codegen.alloc_stack(p_stack_level);
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR);
			codegen.opcodes.push_back(Variant::OP_EQUAL);
			codegen.opcodes.push_back(p_type_addr);
			codegen.opcodes.push_back(literal_type_addr);
			codegen.opcodes.push_back(equality_addr); // Address to result.

			// Jump away if not the same type.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
			codegen.opcodes.push_back(equality_addr);
			r_patch_addresses.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.

			// Get literal.
			int literal_addr = _parse_expression(codegen, p_pattern->literal, p_stack_level);

			// Check value equality.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR);
			codegen.opcodes.push_back(Variant::OP_EQUAL);
			codegen.opcodes.push_back(p_value_addr);
			codegen.opcodes.push_back(literal_addr);
			codegen.opcodes.push_back(equality_addr); // Address to result.

			// Jump away if doesn't match.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
			codegen.opcodes.push_back(equality_addr);
			r_patch_addresses.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.

			// Jump to the actual block since it matches. This is needed to take multi-pattern into account.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
			r_block_patch_address.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.
		} break;
		case GDScriptParser::PatternNode::PT_EXPRESSION: {
			// Evaluate expression.
			int expr_addr = _parse_expression(codegen, p_pattern->expression, p_stack_level);
			if ((expr_addr >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
				p_stack_level++;
				codegen.alloc_stack(p_stack_level);
			}

			// Evaluate expression type.
			int expr_type_addr = p_stack_level++;
			expr_type_addr |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
			codegen.alloc_stack(p_stack_level);
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL_BUILT_IN);
			codegen.opcodes.push_back(GDScriptFunctions::TYPE_OF);
			codegen.opcodes.push_back(1); // One argument.
			codegen.opcodes.push_back(expr_addr); // Argument is the value we want to test.
			codegen.opcodes.push_back(expr_type_addr); // Address to result.

			// Check type equality.
			int equality_addr = p_stack_level++;
			equality_addr |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
			codegen.alloc_stack(p_stack_level);
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR);
			codegen.opcodes.push_back(Variant::OP_EQUAL);
			codegen.opcodes.push_back(p_type_addr);
			codegen.opcodes.push_back(expr_type_addr);
			codegen.opcodes.push_back(equality_addr); // Address to result.

			// Jump away if not the same type.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
			codegen.opcodes.push_back(equality_addr);
			r_patch_addresses.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.

			// Check value equality.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR);
			codegen.opcodes.push_back(Variant::OP_EQUAL);
			codegen.opcodes.push_back(p_value_addr);
			codegen.opcodes.push_back(expr_addr);
			codegen.opcodes.push_back(equality_addr); // Address to result.

			// Jump away if doesn't match.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
			codegen.opcodes.push_back(equality_addr);
			r_patch_addresses.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.

			// Jump to the actual block since it matches. This is needed to take multi-pattern into account.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
			r_block_patch_address.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.
		} break;
		case GDScriptParser::PatternNode::PT_BIND: {
			// Create new stack variable.
			int bind_addr = p_stack_level | (GDScriptFunction::ADDR_TYPE_STACK_VARIABLE << GDScriptFunction::ADDR_BITS);
			codegen.add_stack_identifier(p_pattern->bind->name, p_stack_level++);
			codegen.alloc_stack(p_stack_level);
			r_bound_variables++;

			// Assign value to bound variable.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN);
			codegen.opcodes.push_back(bind_addr); // Destination.
			codegen.opcodes.push_back(p_value_addr); // Source.
			// Not need to block jump because bind happens only once.
		} break;
		case GDScriptParser::PatternNode::PT_ARRAY: {
			int slevel = p_stack_level;

			// Get array type into constant map.
			int array_type_addr = codegen.get_constant_pos(Variant::ARRAY);

			// Check type equality.
			int equality_addr = slevel++;
			equality_addr |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
			codegen.alloc_stack(slevel);
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR);
			codegen.opcodes.push_back(Variant::OP_EQUAL);
			codegen.opcodes.push_back(p_type_addr);
			codegen.opcodes.push_back(array_type_addr);
			codegen.opcodes.push_back(equality_addr); // Address to result.

			// Jump away if not the same type.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
			codegen.opcodes.push_back(equality_addr);
			r_patch_addresses.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.

			// Store pattern length in constant map.
			int array_length_addr = codegen.get_constant_pos(p_pattern->rest_used ? p_pattern->array.size() - 1 : p_pattern->array.size());

			// Get value length.
			int value_length_addr = slevel++;
			codegen.alloc_stack(slevel);
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL_BUILT_IN);
			codegen.opcodes.push_back(GDScriptFunctions::LEN);
			codegen.opcodes.push_back(1); // One argument.
			codegen.opcodes.push_back(p_value_addr); // Argument is the value we want to test.
			codegen.opcodes.push_back(value_length_addr); // Address to result.

			// Test length compatibility.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR);
			codegen.opcodes.push_back(p_pattern->rest_used ? Variant::OP_GREATER_EQUAL : Variant::OP_EQUAL);
			codegen.opcodes.push_back(value_length_addr);
			codegen.opcodes.push_back(array_length_addr);
			codegen.opcodes.push_back(equality_addr); // Address to result.

			// Jump away if length is not compatible.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
			codegen.opcodes.push_back(equality_addr);
			r_patch_addresses.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.

			// Evaluate element by element.
			for (int i = 0; i < p_pattern->array.size(); i++) {
				if (p_pattern->array[i]->pattern_type == GDScriptParser::PatternNode::PT_REST) {
					// Don't want to access an extra element of the user array.
					break;
				}

				int stlevel = p_stack_level;
				Vector<int> element_block_patches; // I want to internal patterns try the next element instead of going to the block.
				// Add index to constant map.
				int index_addr = codegen.get_constant_pos(i);

				// Get the actual element from the user-sent array.
				int element_addr = stlevel++;
				codegen.alloc_stack(stlevel);
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_GET);
				codegen.opcodes.push_back(p_value_addr); // Source.
				codegen.opcodes.push_back(index_addr); // Index.
				codegen.opcodes.push_back(element_addr); // Destination.

				// Also get type of element.
				int element_type_addr = stlevel++;
				element_type_addr |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
				codegen.alloc_stack(stlevel);
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL_BUILT_IN);
				codegen.opcodes.push_back(GDScriptFunctions::TYPE_OF);
				codegen.opcodes.push_back(1); // One argument.
				codegen.opcodes.push_back(element_addr); // Argument is the value we want to test.
				codegen.opcodes.push_back(element_type_addr); // Address to result.

				// Try the pattern inside the element.
				Error err = _parse_match_pattern(codegen, p_pattern->array[i], stlevel, element_addr, element_type_addr, r_bound_variables, r_patch_addresses, element_block_patches);
				if (err != OK) {
					return err;
				}

				// Patch jumps to block to try the next element.
				for (int j = 0; j < element_block_patches.size(); j++) {
					codegen.opcodes.write[element_block_patches[j]] = codegen.opcodes.size();
				}
			}

			// Jump to the actual block since it matches. This is needed to take multi-pattern into account.
			// Also here for the case of empty arrays.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
			r_block_patch_address.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.
		} break;
		case GDScriptParser::PatternNode::PT_DICTIONARY: {
			int slevel = p_stack_level;

			// Get dictionary type into constant map.
			int dict_type_addr = codegen.get_constant_pos(Variant::DICTIONARY);

			// Check type equality.
			int equality_addr = slevel++;
			equality_addr |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
			codegen.alloc_stack(slevel);
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR);
			codegen.opcodes.push_back(Variant::OP_EQUAL);
			codegen.opcodes.push_back(p_type_addr);
			codegen.opcodes.push_back(dict_type_addr);
			codegen.opcodes.push_back(equality_addr); // Address to result.

			// Jump away if not the same type.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
			codegen.opcodes.push_back(equality_addr);
			r_patch_addresses.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.

			// Store pattern length in constant map.
			int dict_length_addr = codegen.get_constant_pos(p_pattern->rest_used ? p_pattern->dictionary.size() - 1 : p_pattern->dictionary.size());

			// Get user's dictionary length.
			int value_length_addr = slevel++;
			codegen.alloc_stack(slevel);
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL_BUILT_IN);
			codegen.opcodes.push_back(GDScriptFunctions::LEN);
			codegen.opcodes.push_back(1); // One argument.
			codegen.opcodes.push_back(p_value_addr); // Argument is the value we want to test.
			codegen.opcodes.push_back(value_length_addr); // Address to result.

			// Test length compatibility.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_OPERATOR);
			codegen.opcodes.push_back(p_pattern->rest_used ? Variant::OP_GREATER_EQUAL : Variant::OP_EQUAL);
			codegen.opcodes.push_back(value_length_addr);
			codegen.opcodes.push_back(dict_length_addr);
			codegen.opcodes.push_back(equality_addr); // Address to result.

			// Jump away if length is not compatible.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
			codegen.opcodes.push_back(equality_addr);
			r_patch_addresses.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.

			// Evaluate element by element.
			for (int i = 0; i < p_pattern->dictionary.size(); i++) {
				const GDScriptParser::PatternNode::Pair &element = p_pattern->dictionary[i];
				if (element.value_pattern && element.value_pattern->pattern_type == GDScriptParser::PatternNode::PT_REST) {
					// Ignore rest pattern.
					continue;
				}
				int stlevel = p_stack_level;
				Vector<int> element_block_patches; // I want to internal patterns try the next element instead of going to the block.

				// Get the pattern key.
				int pattern_key_addr = _parse_expression(codegen, element.key, stlevel);
				if (pattern_key_addr < 0) {
					return ERR_PARSE_ERROR;
				}
				if ((pattern_key_addr >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
					stlevel++;
					codegen.alloc_stack(stlevel);
				}

				// Create stack slot for test result.
				int test_result = stlevel++;
				test_result |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
				codegen.alloc_stack(stlevel);

				// Check if pattern key exists in user's dictionary.
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL_RETURN);
				codegen.opcodes.push_back(1); // Argument count.
				codegen.opcodes.push_back(p_value_addr); // Base (user dictionary).
				codegen.opcodes.push_back(codegen.get_name_map_pos("has")); // Function name.
				codegen.opcodes.push_back(pattern_key_addr); // Argument (pattern key).
				codegen.opcodes.push_back(test_result); // Return address.

				// Jump away if key doesn't exist.
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
				codegen.opcodes.push_back(test_result);
				r_patch_addresses.push_back(codegen.opcodes.size());
				codegen.opcodes.push_back(0); // Will be replaced.

				if (element.value_pattern != nullptr) {
					// Get actual value from user dictionary.
					int value_addr = stlevel++;
					codegen.alloc_stack(stlevel);
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_GET);
					codegen.opcodes.push_back(p_value_addr); // Source.
					codegen.opcodes.push_back(pattern_key_addr); // Index.
					codegen.opcodes.push_back(value_addr); // Destination.

					// Also get type of value.
					int value_type_addr = stlevel++;
					value_type_addr |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
					codegen.alloc_stack(stlevel);
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL_BUILT_IN);
					codegen.opcodes.push_back(GDScriptFunctions::TYPE_OF);
					codegen.opcodes.push_back(1); // One argument.
					codegen.opcodes.push_back(value_addr); // Argument is the value we want to test.
					codegen.opcodes.push_back(value_type_addr); // Address to result.

					// Try the pattern inside the value.
					Error err = _parse_match_pattern(codegen, element.value_pattern, stlevel, value_addr, value_type_addr, r_bound_variables, r_patch_addresses, element_block_patches);
					if (err != OK) {
						return err;
					}
				}

				// Patch jumps to block to try the next element.
				for (int j = 0; j < element_block_patches.size(); j++) {
					codegen.opcodes.write[element_block_patches[j]] = codegen.opcodes.size();
				}
			}

			// Jump to the actual block since it matches. This is needed to take multi-pattern into account.
			// Also here for the case of empty dictionaries.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
			r_block_patch_address.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.

		} break;
		case GDScriptParser::PatternNode::PT_REST:
			// Do nothing.
			break;
		case GDScriptParser::PatternNode::PT_WILDCARD:
			// This matches anything so just do the jump.
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
			r_block_patch_address.push_back(codegen.opcodes.size());
			codegen.opcodes.push_back(0); // Will be replaced.
	}
	return OK;
}

Error GDScriptCompiler::_parse_block(CodeGen &codegen, const GDScriptParser::SuiteNode *p_block, int p_stack_level, int p_break_addr, int p_continue_addr) {
	codegen.push_stack_identifiers();
	int new_identifiers = 0;
	codegen.current_line = p_block->start_line;

	for (int i = 0; i < p_block->statements.size(); i++) {
		const GDScriptParser::Node *s = p_block->statements[i];

#ifdef DEBUG_ENABLED
		// Add a newline before each statement, since the debugger needs those.
		codegen.opcodes.push_back(GDScriptFunction::OPCODE_LINE);
		codegen.opcodes.push_back(s->start_line);
		codegen.current_line = s->start_line;
#endif

		switch (s->type) {
			case GDScriptParser::Node::MATCH: {
				const GDScriptParser::MatchNode *match = static_cast<const GDScriptParser::MatchNode *>(s);

				int slevel = p_stack_level;

				// First, let's save the addres of the value match.
				int temp_addr = _parse_expression(codegen, match->test, slevel);
				if (temp_addr < 0) {
					return ERR_PARSE_ERROR;
				}
				if ((temp_addr >> GDScriptFunction::ADDR_BITS & GDScriptFunction::ADDR_TYPE_STACK) == GDScriptFunction::ADDR_TYPE_STACK) {
					slevel++;
					codegen.alloc_stack(slevel);
				}

				// Then, let's save the type of the value in the stack too, so we can reuse for later comparisons.
				int type_addr = slevel++;
				type_addr |= GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS;
				codegen.alloc_stack(slevel);
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_CALL_BUILT_IN);
				codegen.opcodes.push_back(GDScriptFunctions::TYPE_OF);
				codegen.opcodes.push_back(1); // One argument.
				codegen.opcodes.push_back(temp_addr); // Argument is the value we want to test.
				codegen.opcodes.push_back(type_addr); // Address to result.

				Vector<int> patch_match_end; // Will patch the jump to the end of match.

				// Now we can actually start testing.
				// For each branch.
				for (int j = 0; j < match->branches.size(); j++) {
					const GDScriptParser::MatchBranchNode *branch = match->branches[j];

					int bound_variables = 0;
					codegen.push_stack_identifiers(); // Create an extra block around for binds.

#ifdef DEBUG_ENABLED
					// Add a newline before each branch, since the debugger needs those.
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_LINE);
					codegen.opcodes.push_back(s->start_line);
					codegen.current_line = s->start_line;
#endif
					Vector<int> patch_addrs; // Will patch with end of pattern to jump.
					Vector<int> block_patch_addrs; // Will patch with start of block to jump.

					// For each pattern in branch.
					for (int k = 0; k < branch->patterns.size(); k++) {
						if (k > 0) {
							// Patch jumps per pattern to allow for multipattern. If a pattern fails it just tries the next.
							for (int l = 0; l < patch_addrs.size(); l++) {
								codegen.opcodes.write[patch_addrs[l]] = codegen.opcodes.size();
							}
							patch_addrs.clear();
						}
						Error err = _parse_match_pattern(codegen, branch->patterns[k], slevel, temp_addr, type_addr, bound_variables, patch_addrs, block_patch_addrs);
						if (err != OK) {
							return err;
						}
					}
					// Patch jumps to the block.
					for (int k = 0; k < block_patch_addrs.size(); k++) {
						codegen.opcodes.write[block_patch_addrs[k]] = codegen.opcodes.size();
					}

					// Leave space for bound variables.
					slevel += bound_variables;
					codegen.alloc_stack(slevel);

					// Parse the branch block.
					_parse_block(codegen, branch->block, slevel, p_break_addr, p_continue_addr);

					// Jump to end of match.
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
					patch_match_end.push_back(codegen.opcodes.size());
					codegen.opcodes.push_back(0); // Will be patched.

					// Patch the addresses of last pattern to jump to the end of the branch, into the next one.
					for (int k = 0; k < patch_addrs.size(); k++) {
						codegen.opcodes.write[patch_addrs[k]] = codegen.opcodes.size();
					}

					codegen.pop_stack_identifiers(); // Get out of extra block.
				}
				// Patch the addresses to jump to the end of the match statement.
				for (int j = 0; j < patch_match_end.size(); j++) {
					codegen.opcodes.write[patch_match_end[j]] = codegen.opcodes.size();
				}
			} break;

			case GDScriptParser::Node::IF: {
				const GDScriptParser::IfNode *if_n = static_cast<const GDScriptParser::IfNode *>(s);
				int ret2 = _parse_expression(codegen, if_n->condition, p_stack_level, false);
				if (ret2 < 0) {
					return ERR_PARSE_ERROR;
				}

				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
				codegen.opcodes.push_back(ret2);
				int else_addr = codegen.opcodes.size();
				codegen.opcodes.push_back(0); //temporary

				Error err = _parse_block(codegen, if_n->true_block, p_stack_level, p_break_addr, p_continue_addr);
				if (err) {
					return err;
				}

				if (if_n->false_block) {
					codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
					int end_addr = codegen.opcodes.size();
					codegen.opcodes.push_back(0);
					codegen.opcodes.write[else_addr] = codegen.opcodes.size();

					Error err2 = _parse_block(codegen, if_n->false_block, p_stack_level, p_break_addr, p_continue_addr);
					if (err2) {
						return err2;
					}

					codegen.opcodes.write[end_addr] = codegen.opcodes.size();
				} else {
					//end without else
					codegen.opcodes.write[else_addr] = codegen.opcodes.size();
				}

			} break;
			case GDScriptParser::Node::FOR: {
				const GDScriptParser::ForNode *for_n = static_cast<const GDScriptParser::ForNode *>(s);
				int slevel = p_stack_level;
				int iter_stack_pos = slevel;
				int iterator_pos = (slevel++) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
				int counter_pos = (slevel++) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
				int container_pos = (slevel++) | (GDScriptFunction::ADDR_TYPE_STACK << GDScriptFunction::ADDR_BITS);
				codegen.alloc_stack(slevel);

				codegen.push_stack_identifiers();
				codegen.add_stack_identifier(for_n->variable->name, iter_stack_pos);

				int ret2 = _parse_expression(codegen, for_n->list, slevel, false);
				if (ret2 < 0) {
					return ERR_COMPILATION_FAILED;
				}

				//assign container
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSIGN);
				codegen.opcodes.push_back(container_pos);
				codegen.opcodes.push_back(ret2);

				//begin loop
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_ITERATE_BEGIN);
				codegen.opcodes.push_back(counter_pos);
				codegen.opcodes.push_back(container_pos);
				codegen.opcodes.push_back(codegen.opcodes.size() + 4);
				codegen.opcodes.push_back(iterator_pos);
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP); //skip code for next
				codegen.opcodes.push_back(codegen.opcodes.size() + 8);
				//break loop
				int break_pos = codegen.opcodes.size();
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP); //skip code for next
				codegen.opcodes.push_back(0); //skip code for next
				//next loop
				int continue_pos = codegen.opcodes.size();
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_ITERATE);
				codegen.opcodes.push_back(counter_pos);
				codegen.opcodes.push_back(container_pos);
				codegen.opcodes.push_back(break_pos);
				codegen.opcodes.push_back(iterator_pos);

				Error err = _parse_block(codegen, for_n->loop, slevel, break_pos, continue_pos);
				if (err) {
					return err;
				}

				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
				codegen.opcodes.push_back(continue_pos);
				codegen.opcodes.write[break_pos + 1] = codegen.opcodes.size();

				codegen.pop_stack_identifiers();

			} break;
			case GDScriptParser::Node::WHILE: {
				const GDScriptParser::WhileNode *while_n = static_cast<const GDScriptParser::WhileNode *>(s);
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
				codegen.opcodes.push_back(codegen.opcodes.size() + 3);
				int break_addr = codegen.opcodes.size();
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
				codegen.opcodes.push_back(0);
				int continue_addr = codegen.opcodes.size();

				int ret2 = _parse_expression(codegen, while_n->condition, p_stack_level, false);
				if (ret2 < 0) {
					return ERR_PARSE_ERROR;
				}
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_IF_NOT);
				codegen.opcodes.push_back(ret2);
				codegen.opcodes.push_back(break_addr);
				Error err = _parse_block(codegen, while_n->loop, p_stack_level, break_addr, continue_addr);
				if (err) {
					return err;
				}
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
				codegen.opcodes.push_back(continue_addr);

				codegen.opcodes.write[break_addr + 1] = codegen.opcodes.size();

			} break;
			case GDScriptParser::Node::BREAK: {
				if (p_break_addr < 0) {
					_set_error("'break'' not within loop", s);
					return ERR_COMPILATION_FAILED;
				}
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
				codegen.opcodes.push_back(p_break_addr);

			} break;
			case GDScriptParser::Node::CONTINUE: {
				if (p_continue_addr < 0) {
					_set_error("'continue' not within loop", s);
					return ERR_COMPILATION_FAILED;
				}

				codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP);
				codegen.opcodes.push_back(p_continue_addr);

			} break;
			case GDScriptParser::Node::RETURN: {
				const GDScriptParser::ReturnNode *return_n = static_cast<const GDScriptParser::ReturnNode *>(s);
				int ret2;

				if (return_n->return_value != nullptr) {
					ret2 = _parse_expression(codegen, return_n->return_value, p_stack_level, false);
					if (ret2 < 0) {
						return ERR_PARSE_ERROR;
					}

				} else {
					ret2 = GDScriptFunction::ADDR_TYPE_NIL << GDScriptFunction::ADDR_BITS;
				}

				codegen.opcodes.push_back(GDScriptFunction::OPCODE_RETURN);
				codegen.opcodes.push_back(ret2);

			} break;
			case GDScriptParser::Node::ASSERT: {
#ifdef DEBUG_ENABLED
				// try subblocks

				const GDScriptParser::AssertNode *as = static_cast<const GDScriptParser::AssertNode *>(s);

				int ret2 = _parse_expression(codegen, as->condition, p_stack_level, false);
				if (ret2 < 0) {
					return ERR_PARSE_ERROR;
				}

				int message_ret = 0;
				if (as->message) {
					message_ret = _parse_expression(codegen, as->message, p_stack_level + 1, false);
					if (message_ret < 0) {
						return ERR_PARSE_ERROR;
					}
				}

				codegen.opcodes.push_back(GDScriptFunction::OPCODE_ASSERT);
				codegen.opcodes.push_back(ret2);
				codegen.opcodes.push_back(message_ret);
#endif
			} break;
			case GDScriptParser::Node::BREAKPOINT: {
#ifdef DEBUG_ENABLED
				// try subblocks
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_BREAKPOINT);
#endif
			} break;
			case GDScriptParser::Node::VARIABLE: {
				const GDScriptParser::VariableNode *lv = static_cast<const GDScriptParser::VariableNode *>(s);

				// since we are using properties now for most class access, allow shadowing of class members to make user's life easier.
				//
				//if (_is_class_member_property(codegen, lv->name)) {
				//	_set_error("Name for local variable '" + String(lv->name) + "' can't shadow class property of the same name.", lv);
				//	return ERR_ALREADY_EXISTS;
				//}

				codegen.add_stack_identifier(lv->identifier->name, p_stack_level++);
				codegen.alloc_stack(p_stack_level);
				new_identifiers++;

				if (lv->initializer != nullptr) {
					int dst_address = codegen.stack_identifiers[lv->identifier->name];
					dst_address |= GDScriptFunction::ADDR_TYPE_STACK_VARIABLE << GDScriptFunction::ADDR_BITS;

					int src_address = _parse_expression(codegen, lv->initializer, p_stack_level);
					if (src_address < 0) {
						return ERR_PARSE_ERROR;
					}
					if (!_generate_typed_assign(codegen, src_address, dst_address, _gdtype_from_datatype(lv->get_datatype()), lv->initializer->get_datatype())) {
						return ERR_PARSE_ERROR;
					}
				}
			} break;
			case GDScriptParser::Node::CONSTANT: {
				// Local constants.
				const GDScriptParser::ConstantNode *lc = static_cast<const GDScriptParser::ConstantNode *>(s);
				if (!lc->initializer->is_constant) {
					_set_error("Local constant must have a constant value as initializer.", lc->initializer);
					return ERR_PARSE_ERROR;
				}
				codegen.local_named_constants[lc->identifier->name] = codegen.get_constant_pos(lc->initializer->reduced_value);
			} break;
			case GDScriptParser::Node::PASS:
				// Nothing to do.
				break;
			default: {
				//expression
				if (s->is_expression()) {
					int ret2 = _parse_expression(codegen, static_cast<const GDScriptParser::ExpressionNode *>(s), p_stack_level, true);
					if (ret2 < 0) {
						return ERR_PARSE_ERROR;
					}
				} else {
					ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Bug in bytecode compiler, unexpected node in parse tree while parsing statement."); //unreachable code
				}
			} break;
		}
	}

	codegen.pop_stack_identifiers();
	return OK;
}

Error GDScriptCompiler::_parse_function(GDScript *p_script, const GDScriptParser::ClassNode *p_class, const GDScriptParser::FunctionNode *p_func, bool p_for_ready) {
	Vector<int> bytecode;
	CodeGen codegen;

	codegen.class_node = p_class;
	codegen.script = p_script;
	codegen.function_node = p_func;
	codegen.stack_max = 0;
	codegen.current_line = 0;
	codegen.call_max = 0;
	codegen.debug_stack = EngineDebugger::is_active();
	Vector<StringName> argnames;

	int stack_level = 0;
	int optional_parameters = 0;

	if (p_func) {
		for (int i = 0; i < p_func->parameters.size(); i++) {
			// since we are using properties now for most class access, allow shadowing of class members to make user's life easier.
			//
			//if (_is_class_member_property(p_script, p_func->arguments[i])) {
			//	_set_error("Name for argument '" + String(p_func->arguments[i]) + "' can't shadow class property of the same name.", p_func);
			//	return ERR_ALREADY_EXISTS;
			//}

			codegen.add_stack_identifier(p_func->parameters[i]->identifier->name, i);
#ifdef TOOLS_ENABLED
			argnames.push_back(p_func->parameters[i]->identifier->name);
#endif
			if (p_func->parameters[i]->default_value != nullptr) {
				optional_parameters++;
			}
		}
		stack_level = p_func->parameters.size();
	}

	codegen.alloc_stack(stack_level);

	/* Parse initializer -if applies- */

	bool is_implicit_initializer = !p_for_ready && !p_func;
	bool is_initializer = p_func && String(p_func->identifier->name) == GDScriptLanguage::get_singleton()->strings._init;

	if (is_implicit_initializer) {
		// Initialize class fields.
		for (int i = 0; i < p_class->members.size(); i++) {
			if (p_class->members[i].type != GDScriptParser::ClassNode::Member::VARIABLE) {
				continue;
			}
			const GDScriptParser::VariableNode *field = p_class->members[i].variable;
			if (field->onready) {
				// Only initialize in _ready.
				continue;
			}

			if (field->initializer) {
				// Emit proper line change.
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_LINE);
				codegen.opcodes.push_back(field->initializer->start_line);

				int src_address = _parse_expression(codegen, field->initializer, stack_level, false, true);
				if (src_address < 0) {
					return ERR_PARSE_ERROR;
				}
				int dst_address = codegen.script->member_indices[field->identifier->name].index;
				dst_address |= GDScriptFunction::ADDR_TYPE_MEMBER << GDScriptFunction::ADDR_BITS;

				if (!_generate_typed_assign(codegen, src_address, dst_address, _gdtype_from_datatype(field->get_datatype()), field->initializer->get_datatype())) {
					return ERR_PARSE_ERROR;
				}
			}
		}
	}

	if (p_for_ready || (p_func && String(p_func->identifier->name) == "_ready")) {
		// Initialize class fields on ready.
		for (int i = 0; i < p_class->members.size(); i++) {
			if (p_class->members[i].type != GDScriptParser::ClassNode::Member::VARIABLE) {
				continue;
			}
			const GDScriptParser::VariableNode *field = p_class->members[i].variable;
			if (!field->onready) {
				continue;
			}

			if (field->initializer) {
				// Emit proper line change.
				codegen.opcodes.push_back(GDScriptFunction::OPCODE_LINE);
				codegen.opcodes.push_back(field->initializer->start_line);

				int src_address = _parse_expression(codegen, field->initializer, stack_level, false, true);
				if (src_address < 0) {
					return ERR_PARSE_ERROR;
				}
				int dst_address = codegen.script->member_indices[field->identifier->name].index;
				dst_address |= GDScriptFunction::ADDR_TYPE_MEMBER << GDScriptFunction::ADDR_BITS;

				if (!_generate_typed_assign(codegen, src_address, dst_address, _gdtype_from_datatype(field->get_datatype()), field->initializer->get_datatype())) {
					return ERR_PARSE_ERROR;
				}
			}
		}
	}

	/* Parse default argument code -if applies- */

	Vector<int> defarg_addr;
	StringName func_name;

	if (p_func) {
		if (optional_parameters > 0) {
			codegen.opcodes.push_back(GDScriptFunction::OPCODE_JUMP_TO_DEF_ARGUMENT);
			defarg_addr.push_back(codegen.opcodes.size());
			for (int i = p_func->parameters.size() - optional_parameters; i < p_func->parameters.size(); i++) {
				int src_addr = _parse_expression(codegen, p_func->parameters[i]->default_value, stack_level, true);
				if (src_addr < 0) {
					return ERR_PARSE_ERROR;
				}
				int dst_addr = codegen.stack_identifiers[p_func->parameters[i]->identifier->name] | (GDScriptFunction::ADDR_TYPE_STACK_VARIABLE << GDScriptFunction::ADDR_BITS);
				if (!_generate_typed_assign(codegen, src_addr, dst_addr, _gdtype_from_datatype(p_func->parameters[i]->get_datatype()), p_func->parameters[i]->default_value->get_datatype())) {
					return ERR_PARSE_ERROR;
				}
				defarg_addr.push_back(codegen.opcodes.size());
			}
			defarg_addr.invert();
		}
		func_name = p_func->identifier->name;
		codegen.function_name = func_name;

		Error err = _parse_block(codegen, p_func->body, stack_level);
		if (err) {
			return err;
		}

	} else {
		if (p_for_ready) {
			func_name = "_ready";
		} else {
			func_name = "@implicit_new";
		}
	}

	codegen.function_name = func_name;
	codegen.opcodes.push_back(GDScriptFunction::OPCODE_END);

	/*
	if (String(p_func->name)=="") { //initializer func
		gdfunc = &p_script->initializer;
	*/
	//} else { //regular func
	p_script->member_functions[func_name] = memnew(GDScriptFunction);
	GDScriptFunction *gdfunc = p_script->member_functions[func_name];
	//}

	if (p_func) {
		gdfunc->_static = p_func->is_static;
		gdfunc->rpc_mode = p_func->rpc_mode;
		gdfunc->argument_types.resize(p_func->parameters.size());
		for (int i = 0; i < p_func->parameters.size(); i++) {
			gdfunc->argument_types.write[i] = _gdtype_from_datatype(p_func->parameters[i]->get_datatype());
		}
		gdfunc->return_type = _gdtype_from_datatype(p_func->get_datatype());
	} else {
		gdfunc->_static = false;
		gdfunc->rpc_mode = MultiplayerAPI::RPC_MODE_DISABLED;
		gdfunc->return_type = GDScriptDataType();
		gdfunc->return_type.has_type = true;
		gdfunc->return_type.kind = GDScriptDataType::BUILTIN;
		gdfunc->return_type.builtin_type = Variant::NIL;
	}

#ifdef TOOLS_ENABLED
	gdfunc->arg_names = argnames;
#endif
	//constants
	if (codegen.constant_map.size()) {
		gdfunc->_constant_count = codegen.constant_map.size();
		gdfunc->constants.resize(codegen.constant_map.size());
		gdfunc->_constants_ptr = gdfunc->constants.ptrw();
		const Variant *K = nullptr;
		while ((K = codegen.constant_map.next(K))) {
			int idx = codegen.constant_map[*K];
			gdfunc->constants.write[idx] = *K;
		}
	} else {
		gdfunc->_constants_ptr = nullptr;
		gdfunc->_constant_count = 0;
	}
	//global names
	if (codegen.name_map.size()) {
		gdfunc->global_names.resize(codegen.name_map.size());
		gdfunc->_global_names_ptr = &gdfunc->global_names[0];
		for (Map<StringName, int>::Element *E = codegen.name_map.front(); E; E = E->next()) {
			gdfunc->global_names.write[E->get()] = E->key();
		}
		gdfunc->_global_names_count = gdfunc->global_names.size();

	} else {
		gdfunc->_global_names_ptr = nullptr;
		gdfunc->_global_names_count = 0;
	}

#ifdef TOOLS_ENABLED
	// Named globals
	if (codegen.named_globals.size()) {
		gdfunc->named_globals.resize(codegen.named_globals.size());
		gdfunc->_named_globals_ptr = gdfunc->named_globals.ptr();
		for (int i = 0; i < codegen.named_globals.size(); i++) {
			gdfunc->named_globals.write[i] = codegen.named_globals[i];
		}
		gdfunc->_named_globals_count = gdfunc->named_globals.size();
	}
#endif

	if (codegen.opcodes.size()) {
		gdfunc->code = codegen.opcodes;
		gdfunc->_code_ptr = &gdfunc->code[0];
		gdfunc->_code_size = codegen.opcodes.size();

	} else {
		gdfunc->_code_ptr = nullptr;
		gdfunc->_code_size = 0;
	}

	if (defarg_addr.size()) {
		gdfunc->default_arguments = defarg_addr;
		gdfunc->_default_arg_count = defarg_addr.size() - 1;
		gdfunc->_default_arg_ptr = &gdfunc->default_arguments[0];
	} else {
		gdfunc->_default_arg_count = 0;
		gdfunc->_default_arg_ptr = nullptr;
	}

	gdfunc->_argument_count = p_func ? p_func->parameters.size() : 0;
	gdfunc->_stack_size = codegen.stack_max;
	gdfunc->_call_size = codegen.call_max;
	gdfunc->name = func_name;
#ifdef DEBUG_ENABLED
	if (EngineDebugger::is_active()) {
		String signature;
		//path
		if (p_script->get_path() != String()) {
			signature += p_script->get_path();
		}
		//loc
		if (p_func) {
			signature += "::" + itos(p_func->body->start_line);
		} else {
			signature += "::0";
		}

		//function and class

		if (p_class->identifier) {
			signature += "::" + String(p_class->identifier->name) + "." + String(func_name);
		} else {
			signature += "::" + String(func_name);
		}

		gdfunc->profile.signature = signature;
	}
#endif
	gdfunc->_script = p_script;
	gdfunc->source = source;

#ifdef DEBUG_ENABLED

	{
		gdfunc->func_cname = (String(source) + " - " + String(func_name)).utf8();
		gdfunc->_func_cname = gdfunc->func_cname.get_data();
	}

#endif
	if (p_func) {
		gdfunc->_initial_line = p_func->start_line;
#ifdef TOOLS_ENABLED

		p_script->member_lines[func_name] = p_func->start_line;
#endif
	} else {
		gdfunc->_initial_line = 0;
	}

	if (codegen.debug_stack) {
		gdfunc->stack_debug = codegen.stack_debug;
	}

	if (is_initializer) {
		p_script->initializer = gdfunc;
	}
	if (is_implicit_initializer) {
		p_script->implicit_initializer = gdfunc;
	}

	return OK;
}

Error GDScriptCompiler::_parse_setter_getter(GDScript *p_script, const GDScriptParser::ClassNode *p_class, const GDScriptParser::VariableNode *p_variable, bool p_is_setter) {
	Vector<int> bytecode;
	CodeGen codegen;

	codegen.class_node = p_class;
	codegen.script = p_script;
	codegen.function_node = nullptr;
	codegen.stack_max = 0;
	codegen.current_line = 0;
	codegen.call_max = 0;
	codegen.debug_stack = EngineDebugger::is_active();
	Vector<StringName> argnames;

	int stack_level = 0;

	if (p_is_setter) {
		codegen.add_stack_identifier(p_variable->setter_parameter->name, stack_level++);
		argnames.push_back(p_variable->setter_parameter->name);
	}

	codegen.alloc_stack(stack_level);

	StringName func_name;

	if (p_is_setter) {
		func_name = "@" + p_variable->identifier->name + "_setter";
	} else {
		func_name = "@" + p_variable->identifier->name + "_getter";
	}
	codegen.function_name = func_name;

	Error err = _parse_block(codegen, p_is_setter ? p_variable->setter : p_variable->getter, stack_level);
	if (err != OK) {
		return err;
	}

	codegen.opcodes.push_back(GDScriptFunction::OPCODE_END);

	p_script->member_functions[func_name] = memnew(GDScriptFunction);
	GDScriptFunction *gdfunc = p_script->member_functions[func_name];

	gdfunc->_static = false;
	gdfunc->rpc_mode = p_variable->rpc_mode;
	gdfunc->argument_types.resize(p_is_setter ? 1 : 0);
	gdfunc->return_type = _gdtype_from_datatype(p_variable->get_datatype());
#ifdef TOOLS_ENABLED
	gdfunc->arg_names = argnames;
#endif

	// TODO: Unify this with function compiler.
	//constants
	if (codegen.constant_map.size()) {
		gdfunc->_constant_count = codegen.constant_map.size();
		gdfunc->constants.resize(codegen.constant_map.size());
		gdfunc->_constants_ptr = gdfunc->constants.ptrw();
		const Variant *K = nullptr;
		while ((K = codegen.constant_map.next(K))) {
			int idx = codegen.constant_map[*K];
			gdfunc->constants.write[idx] = *K;
		}
	} else {
		gdfunc->_constants_ptr = nullptr;
		gdfunc->_constant_count = 0;
	}
	//global names
	if (codegen.name_map.size()) {
		gdfunc->global_names.resize(codegen.name_map.size());
		gdfunc->_global_names_ptr = &gdfunc->global_names[0];
		for (Map<StringName, int>::Element *E = codegen.name_map.front(); E; E = E->next()) {
			gdfunc->global_names.write[E->get()] = E->key();
		}
		gdfunc->_global_names_count = gdfunc->global_names.size();

	} else {
		gdfunc->_global_names_ptr = nullptr;
		gdfunc->_global_names_count = 0;
	}

#ifdef TOOLS_ENABLED
	// Named globals
	if (codegen.named_globals.size()) {
		gdfunc->named_globals.resize(codegen.named_globals.size());
		gdfunc->_named_globals_ptr = gdfunc->named_globals.ptr();
		for (int i = 0; i < codegen.named_globals.size(); i++) {
			gdfunc->named_globals.write[i] = codegen.named_globals[i];
		}
		gdfunc->_named_globals_count = gdfunc->named_globals.size();
	}
#endif

	gdfunc->code = codegen.opcodes;
	gdfunc->_code_ptr = &gdfunc->code[0];
	gdfunc->_code_size = codegen.opcodes.size();
	gdfunc->_default_arg_count = 0;
	gdfunc->_default_arg_ptr = nullptr;
	gdfunc->_argument_count = argnames.size();
	gdfunc->_stack_size = codegen.stack_max;
	gdfunc->_call_size = codegen.call_max;
	gdfunc->name = func_name;
#ifdef DEBUG_ENABLED
	if (EngineDebugger::is_active()) {
		String signature;
		//path
		if (p_script->get_path() != String()) {
			signature += p_script->get_path();
		}
		//loc
		signature += "::" + itos(p_is_setter ? p_variable->setter->start_line : p_variable->getter->start_line);

		//function and class

		if (p_class->identifier) {
			signature += "::" + String(p_class->identifier->name) + "." + String(func_name);
		} else {
			signature += "::" + String(func_name);
		}

		gdfunc->profile.signature = signature;
	}
#endif
	gdfunc->_script = p_script;
	gdfunc->source = source;

#ifdef DEBUG_ENABLED

	{
		gdfunc->func_cname = (String(source) + " - " + String(func_name)).utf8();
		gdfunc->_func_cname = gdfunc->func_cname.get_data();
	}

#endif
	gdfunc->_initial_line = p_is_setter ? p_variable->setter->start_line : p_variable->getter->start_line;
#ifdef TOOLS_ENABLED

	p_script->member_lines[func_name] = gdfunc->_initial_line;
#endif

	if (codegen.debug_stack) {
		gdfunc->stack_debug = codegen.stack_debug;
	}

	return OK;
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

	p_script->native = Ref<GDScriptNativeClass>();
	p_script->base = Ref<GDScript>();
	p_script->_base = nullptr;
	p_script->members.clear();
	p_script->constants.clear();
	for (Map<StringName, GDScriptFunction *>::Element *E = p_script->member_functions.front(); E; E = E->next()) {
		memdelete(E->get());
	}
	p_script->member_functions.clear();
	p_script->member_indices.clear();
	p_script->member_info.clear();
	p_script->_signals.clear();
	p_script->initializer = nullptr;

	p_script->tool = parser->is_tool();
	p_script->name = p_class->identifier ? p_class->identifier->name : "";

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
			Ref<GDScript> base = base_type.script_type;
			p_script->base = base;
			p_script->_base = base.ptr();

			if (p_class->base_type.kind == GDScriptParser::DataType::CLASS && p_class->base_type.class_type != nullptr) {
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
			}

			p_script->member_indices = base->member_indices;
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
				minfo.rpc_mode = variable->rpc_mode;
				minfo.data_type = _gdtype_from_datatype(variable->get_datatype());

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
					prop_info.usage = export_info.usage;
#ifdef TOOLS_ENABLED
					if (variable->initializer != nullptr && variable->initializer->type == GDScriptParser::Node::LITERAL) {
						p_script->member_default_values[name] = static_cast<const GDScriptParser::LiteralNode *>(variable->initializer)->value;
					}
#endif
				} else {
					prop_info.usage = PROPERTY_USAGE_SCRIPT_VARIABLE;
				}

				p_script->member_info[name] = prop_info;
				p_script->member_indices[name] = minfo;
				p_script->members.insert(name);

#ifdef TOOLS_ENABLED
				p_script->member_lines[name] = variable->start_line;
#endif
			} break;

			case GDScriptParser::ClassNode::Member::CONSTANT: {
				const GDScriptParser::ConstantNode *constant = member.constant;
				StringName name = constant->identifier->name;

				ERR_CONTINUE(constant->initializer->type != GDScriptParser::Node::LITERAL);

				const GDScriptParser::LiteralNode *literal = static_cast<const GDScriptParser::LiteralNode *>(constant->initializer);

				p_script->constants.insert(name, literal->value);
#ifdef TOOLS_ENABLED

				p_script->member_lines[name] = constant->start_line;
#endif
			} break;

			case GDScriptParser::ClassNode::Member::ENUM_VALUE: {
				const GDScriptParser::EnumNode::Value &enum_value = member.enum_value;
				StringName name = enum_value.identifier->name;

				p_script->constants.insert(name, enum_value.value);
#ifdef TOOLS_ENABLED
				p_script->member_lines[name] = enum_value.identifier->start_line;
#endif
			} break;

			case GDScriptParser::ClassNode::Member::SIGNAL: {
				const GDScriptParser::SignalNode *signal = member.signal;
				StringName name = signal->identifier->name;

				GDScript *c = p_script;

				while (c) {
					if (c->_signals.has(name)) {
						_set_error("Signal '" + name + "' redefined (in current or parent class)", p_class);
						return ERR_ALREADY_EXISTS;
					}

					if (c->base.is_valid()) {
						c = c->base.ptr();
					} else {
						c = nullptr;
					}
				}

				if (native.is_valid()) {
					if (ClassDB::has_signal(native->get_name(), name)) {
						_set_error("Signal '" + name + "' redefined (original in native class '" + String(native->get_name()) + "')", p_class);
						return ERR_ALREADY_EXISTS;
					}
				}

				Vector<StringName> parameters_names;
				parameters_names.resize(signal->parameters.size());
				for (int j = 0; j < signal->parameters.size(); j++) {
					parameters_names.write[j] = signal->parameters[j]->identifier->name;
				}
				p_script->_signals[name] = parameters_names;
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
			Error err = _parse_function(p_script, p_class, function);
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
		Error err = _parse_function(p_script, p_class, nullptr);
		if (err) {
			return err;
		}
	}

	if (!has_ready && p_class->onready_used) {
		//create a _ready constructor
		Error err = _parse_function(p_script, p_class, nullptr, true);
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
					instance->base_ref = Object::cast_to<Reference>(E->get());
					instance->members.resize(p_script->member_indices.size());
					instance->script = Ref<GDScript>(p_script);
					instance->owner = E->get();

					//needed for hot reloading
					for (Map<StringName, GDScript::MemberInfo>::Element *F = p_script->member_indices.front(); F; F = F->next()) {
						instance->member_indices_cache[F->key()] = F->get().index;
					}
					instance->owner->set_script_instance(instance);

					/* STEP 2, INITIALIZE AND CONSTRUCT */

					Callable::CallError ce;
					p_script->initializer->call(instance, nullptr, 0, ce);

					if (ce.error != Callable::CallError::CALL_OK) {
						//well, tough luck, not goinna do anything here
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
				subclass.instance();
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
