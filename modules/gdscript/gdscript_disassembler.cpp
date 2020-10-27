/*************************************************************************/
/*  gdscript_disassembler.cpp                                            */
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

#ifdef DEBUG_ENABLED

#include "gdscript_function.h"

#include "core/string_builder.h"
#include "gdscript.h"
#include "gdscript_functions.h"

static String _get_variant_string(const Variant &p_variant) {
	String txt;
	if (p_variant.get_type() == Variant::STRING) {
		txt = "\"" + String(p_variant) + "\"";
	} else if (p_variant.get_type() == Variant::STRING_NAME) {
		txt = "&\"" + String(p_variant) + "\"";
	} else if (p_variant.get_type() == Variant::NODE_PATH) {
		txt = "^\"" + String(p_variant) + "\"";
	} else if (p_variant.get_type() == Variant::OBJECT) {
		Object *obj = p_variant;
		if (!obj) {
			txt = "null";
		} else {
			GDScriptNativeClass *cls = Object::cast_to<GDScriptNativeClass>(obj);
			if (cls) {
				txt += cls->get_name();
				txt += " (class)";
			} else {
				txt = obj->get_class();
				if (obj->get_script_instance()) {
					txt += "(" + obj->get_script_instance()->get_script()->get_path() + ")";
				}
			}
		}
	} else {
		txt = p_variant;
	}
	return txt;
}

static String _disassemble_address(const GDScript *p_script, const GDScriptFunction &p_function, int p_address) {
	int addr = p_address & GDScriptFunction::ADDR_MASK;

	switch (p_address >> GDScriptFunction::ADDR_BITS) {
		case GDScriptFunction::ADDR_TYPE_SELF: {
			return "self";
		} break;
		case GDScriptFunction::ADDR_TYPE_CLASS: {
			return "class";
		} break;
		case GDScriptFunction::ADDR_TYPE_MEMBER: {
			return "member(" + p_script->debug_get_member_by_index(addr) + ")";
		} break;
		case GDScriptFunction::ADDR_TYPE_CLASS_CONSTANT: {
			return "class_const(" + p_function.get_global_name(addr) + ")";
		} break;
		case GDScriptFunction::ADDR_TYPE_LOCAL_CONSTANT: {
			return "const(" + _get_variant_string(p_function.get_constant(addr)) + ")";
		} break;
		case GDScriptFunction::ADDR_TYPE_STACK: {
			return "stack(" + itos(addr) + ")";
		} break;
		case GDScriptFunction::ADDR_TYPE_STACK_VARIABLE: {
			return "var_stack(" + itos(addr) + ")";
		} break;
		case GDScriptFunction::ADDR_TYPE_GLOBAL: {
			return "global(" + _get_variant_string(GDScriptLanguage::get_singleton()->get_global_array()[addr]) + ")";
		} break;
		case GDScriptFunction::ADDR_TYPE_NAMED_GLOBAL: {
			return "named_global(" + p_function.get_global_name(addr) + ")";
		} break;
		case GDScriptFunction::ADDR_TYPE_NIL: {
			return "nil";
		} break;
	}

	return "<err>";
}

void GDScriptFunction::disassemble(const Vector<String> &p_code_lines) const {
#define DADDR(m_ip) (_disassemble_address(_script, *this, _code_ptr[ip + m_ip]))

#define DISASSEMBLE_BINARY_OP(m_op_name, m_type_a, m_type_b) \
	DISASSEMBLE_BINARY_OP_VAR(m_op_name, m_op_name, m_type_a, m_type_b)

#define DISASSEMBLE_BINARY_OP_VAR(m_op_name, m_var_op, m_type_a, m_type_b) \
	case OPCODE_OP_##m_op_name##_##m_type_a##_##m_type_b: {                \
		Variant::Operator op = Variant::OP_##m_var_op;                     \
		text += "operator typed (";                                        \
		text += #m_type_a;                                                 \
		text += "_";                                                       \
		text += #m_type_b;                                                 \
		text += ") ";                                                      \
		text += DADDR(3);                                                  \
		text += " = ";                                                     \
		text += DADDR(1);                                                  \
		text += " ";                                                       \
		text += Variant::get_operator_name(Variant::Operator(op));         \
		text += " ";                                                       \
		text += DADDR(2);                                                  \
		incr += 4;                                                         \
	} break

#define DISASSEMBLE_BINARY_OP_NO_TYPE(m_op_name)                   \
	case OPCODE_OP_##m_op_name: {                                  \
		Variant::Operator op = Variant::OP_##m_op_name;            \
		text += "operator typed ";                                 \
		text += DADDR(3);                                          \
		text += " = ";                                             \
		text += DADDR(1);                                          \
		text += " ";                                               \
		text += Variant::get_operator_name(Variant::Operator(op)); \
		text += " ";                                               \
		text += DADDR(2);                                          \
		incr += 4;                                                 \
	} break

#define DISASSEMBLE_BINARY_SINGLE_TYPE(m_op_name, m_type)          \
	case OPCODE_OP_##m_op_name##_##m_type: {                       \
		Variant::Operator op = Variant::OP_##m_op_name;            \
		text += "operator typed (";                                \
		text += #m_type;                                           \
		text += ") ";                                              \
		text += DADDR(3);                                          \
		text += " = ";                                             \
		text += DADDR(1);                                          \
		text += " ";                                               \
		text += Variant::get_operator_name(Variant::Operator(op)); \
		text += " ";                                               \
		text += DADDR(2);                                          \
		incr += 4;                                                 \
	} break

#define DISASSEMBLE_BINARY_OP_NUM(m_op_name)      \
	DISASSEMBLE_BINARY_OP(m_op_name, INT, INT);   \
	DISASSEMBLE_BINARY_OP(m_op_name, INT, FLOAT); \
	DISASSEMBLE_BINARY_OP(m_op_name, FLOAT, INT); \
	DISASSEMBLE_BINARY_OP(m_op_name, FLOAT, FLOAT)

#define DISASSEMBLE_BINARY_OP_VEC(m_op_name)              \
	DISASSEMBLE_BINARY_OP(m_op_name, VECTOR2, VECTOR2);   \
	DISASSEMBLE_BINARY_OP(m_op_name, VECTOR2I, VECTOR2I); \
	DISASSEMBLE_BINARY_OP(m_op_name, VECTOR3, VECTOR3);   \
	DISASSEMBLE_BINARY_OP(m_op_name, VECTOR3I, VECTOR3I)

#define DISASSEMBLE_BINARY_OP_TYPE_NUM_FORWARD(m_op_name, m_type) \
	DISASSEMBLE_BINARY_OP(m_op_name, m_type, m_type);             \
	DISASSEMBLE_BINARY_OP(m_op_name, m_type, INT);                \
	DISASSEMBLE_BINARY_OP(m_op_name, m_type, FLOAT)

#define DISASSEMBLE_BINARY_OP_TYPE_NUM_REVERSE(m_op_name, m_type) \
	DISASSEMBLE_BINARY_OP(m_op_name, INT, m_type);                \
	DISASSEMBLE_BINARY_OP(m_op_name, FLOAT, m_type)

#define DISASSEMBLE_BINARY_OP_TYPE_NUM(m_op_name, m_type)      \
	DISASSEMBLE_BINARY_OP_TYPE_NUM_FORWARD(m_op_name, m_type); \
	DISASSEMBLE_BINARY_OP_TYPE_NUM_REVERSE(m_op_name, m_type)

#define DISASSEMBLE_UNARY_OP(m_op_name, m_type)                    \
	case OPCODE_OP_##m_op_name##_##m_type: {                       \
		Variant::Operator op = Variant::OP_##m_op_name;            \
		text += "operator typed (";                                \
		text += #m_type;                                           \
		text += ") ";                                              \
		text += DADDR(2);                                          \
		text += " = ";                                             \
		text += Variant::get_operator_name(Variant::Operator(op)); \
		text += DADDR(1);                                          \
		incr += 3;                                                 \
	} break

#define DISASSEMBLE_UNARY_OP_NO_TYPE(m_op_name)                    \
	case OPCODE_OP_##m_op_name: {                                  \
		Variant::Operator op = Variant::OP_##m_op_name;            \
		text += "operator typed ";                                 \
		text += DADDR(2);                                          \
		text += " = ";                                             \
		text += Variant::get_operator_name(Variant::Operator(op)); \
		text += DADDR(1);                                          \
		incr += 3;                                                 \
	} break

#define DISASSEMBLE_OPCODE_BINARY_ALL_TYPES(m_op) \
	DISASSEMBLE_OPCODE_ALL_TYPES(DISASSEMBLE_BINARY_SINGLE_TYPE, m_op)

#define DISASSEMBLE_OPCODE_ALL_TYPES(m_macro, m_op) \
	m_macro(m_op, BOOL);                            \
	m_macro(m_op, INT);                             \
	m_macro(m_op, FLOAT);                           \
	m_macro(m_op, STRING);                          \
	m_macro(m_op, VECTOR2);                         \
	m_macro(m_op, VECTOR2I);                        \
	m_macro(m_op, RECT2);                           \
	m_macro(m_op, RECT2I);                          \
	m_macro(m_op, VECTOR3);                         \
	m_macro(m_op, VECTOR3I);                        \
	m_macro(m_op, TRANSFORM2D);                     \
	m_macro(m_op, PLANE);                           \
	m_macro(m_op, QUAT);                            \
	m_macro(m_op, AABB);                            \
	m_macro(m_op, BASIS);                           \
	m_macro(m_op, TRANSFORM);                       \
	m_macro(m_op, COLOR);                           \
	m_macro(m_op, STRING_NAME);                     \
	m_macro(m_op, NODE_PATH);                       \
	m_macro(m_op, RID);                             \
	m_macro(m_op, OBJECT);                          \
	m_macro(m_op, CALLABLE);                        \
	m_macro(m_op, SIGNAL);                          \
	m_macro(m_op, DICTIONARY);                      \
	m_macro(m_op, ARRAY);                           \
	m_macro(m_op, PACKED_BYTE_ARRAY);               \
	m_macro(m_op, PACKED_INT32_ARRAY);              \
	m_macro(m_op, PACKED_INT64_ARRAY);              \
	m_macro(m_op, PACKED_FLOAT32_ARRAY);            \
	m_macro(m_op, PACKED_FLOAT64_ARRAY);            \
	m_macro(m_op, PACKED_STRING_ARRAY);             \
	m_macro(m_op, PACKED_VECTOR2_ARRAY);            \
	m_macro(m_op, PACKED_VECTOR3_ARRAY);            \
	m_macro(m_op, PACKED_COLOR_ARRAY)

#define DISASSEMBLE_GET_TYPED(m_type_base, m_type_index) \
	case OPCODE_GET_##m_type_base##_##m_type_index: {    \
		text += "get (typed ";                           \
		text += #m_type_index;                           \
		text += "index from ";                           \
		text += #m_type_base;                            \
		text += ") ";                                    \
		text += DADDR(3);                                \
		text += " = ";                                   \
		text += DADDR(1);                                \
		text += "[";                                     \
		text += DADDR(2);                                \
		text += "]";                                     \
		incr += 4;                                       \
	} break

#define DISASSEMBLE_GET_NAMED_TYPED(m_type_base)      \
	case OPCODE_GET_NAMED_##m_type_base: {            \
		text += "get_named (typed from ";             \
		text += #m_type_base;                         \
		text += ") ";                                 \
		text += DADDR(3);                             \
		text += " = ";                                \
		text += DADDR(1);                             \
		text += "[\"";                                \
		text += _global_names_ptr[_code_ptr[ip + 2]]; \
		text += "\"]";                                \
		incr += 4;                                    \
	} break
#define DISASSEMBLE_CALL_PTRCALL(m_op, m_type)                     \
	case OPCODE_CALL_##m_op##_##m_type: {                          \
		bool ret = _code_ptr[ip] != OPCODE_CALL_PTRCALL_NO_RETURN; \
		int argc = _code_ptr[ip + 1];                              \
		MethodBind *method = _methods_ptr[_code_ptr[ip + 3]];      \
		text += "call-ptrcall ";                                   \
		if (ret) {                                                 \
			text += "ret ";                                        \
		}                                                          \
		text += " (base ";                                         \
		text += method->get_instance_class();                      \
		if (ret) {                                                 \
			text += " returning ";                                 \
			text += #m_type;                                       \
		}                                                          \
		text += ") ";                                              \
		if (ret) {                                                 \
			text += DADDR(4 + argc) + " = ";                       \
		}                                                          \
		text += DADDR(2) + ".";                                    \
		text += method->get_name();                                \
		text += "(";                                               \
		for (int i = 0; i < argc; i++) {                           \
			if (i > 0)                                             \
				text += ", ";                                      \
			text += DADDR(4 + i);                                  \
		}                                                          \
		text += ")";                                               \
		incr = 5 + argc;                                           \
	} break

#define DISASSEMBLE_BUILTIN_CALL(m_op, m_func, m_argc, m_arg_type) \
	case OPCODE_CALL_##m_op##_##m_arg_type: {                      \
		text += "call-built-in (";                                 \
		text += #m_arg_type;                                       \
		text += " arguments) ";                                    \
		text += #m_func;                                           \
		text += "(";                                               \
		for (int i = 0; i < m_argc; i++) {                         \
			if (i > 0)                                             \
				text += ", ";                                      \
			text += DADDR(1 + m_argc);                             \
		}                                                          \
		text += ")";                                               \
		incr = 2 + m_argc;                                         \
	} break

#define DISASSEMBLE_ITERATE(m_type)      \
	case OPCODE_ITERATE_##m_type: {      \
		text += "for-loop (typed ";      \
		text += #m_type;                 \
		text += ") ";                    \
		text += DADDR(4);                \
		text += " in ";                  \
		text += DADDR(2);                \
		text += " counter ";             \
		text += DADDR(1);                \
		text += " end ";                 \
		text += itos(_code_ptr[ip + 3]); \
		incr += 5;                       \
	} break

#define DISASSEMBLE_ITERATE_BEGIN(m_type) \
	case OPCODE_ITERATE_BEGIN_##m_type: { \
		text += "for-init (typed ";       \
		text += #m_type;                  \
		text += ") ";                     \
		text += DADDR(4);                 \
		text += " in ";                   \
		text += DADDR(2);                 \
		text += " counter ";              \
		text += DADDR(1);                 \
		text += " end ";                  \
		text += itos(_code_ptr[ip + 3]);  \
		incr += 5;                        \
	} break

#define DISASSEMBLE_ITERATE_TYPES(m_macro) \
	m_macro(INT);                          \
	m_macro(FLOAT);                        \
	m_macro(VECTOR2);                      \
	m_macro(VECTOR2I);                     \
	m_macro(VECTOR3);                      \
	m_macro(VECTOR3I);                     \
	m_macro(STRING);                       \
	m_macro(DICTIONARY);                   \
	m_macro(ARRAY);                        \
	m_macro(PACKED_BYTE_ARRAY);            \
	m_macro(PACKED_INT32_ARRAY);           \
	m_macro(PACKED_INT64_ARRAY);           \
	m_macro(PACKED_FLOAT32_ARRAY);         \
	m_macro(PACKED_FLOAT64_ARRAY);         \
	m_macro(PACKED_STRING_ARRAY);          \
	m_macro(PACKED_VECTOR2_ARRAY);         \
	m_macro(PACKED_VECTOR3_ARRAY);         \
	m_macro(PACKED_COLOR_ARRAY);           \
	m_macro(OBJECT)

	for (int ip = 0; ip < _code_size;) {
		StringBuilder text;
		int incr = 0;

		text += " ";
		text += itos(ip);
		text += ": ";

		// This makes the compiler complain if some opcode is unchecked in the switch.
		Opcode code = Opcode(_code_ptr[ip]);
		switch (code) {
			case OPCODE_OPERATOR: {
				int operation = _code_ptr[ip + 1];

				text += "operator ";

				text += DADDR(4);
				text += " = ";
				text += DADDR(2);
				text += " ";
				text += Variant::get_operator_name(Variant::Operator(operation));
				text += " ";
				text += DADDR(3);

				incr += 5;
			} break;
				// Typed operations.
				// Addition.
				DISASSEMBLE_BINARY_OP_NUM(ADD);
				DISASSEMBLE_BINARY_OP_VEC(ADD);
				DISASSEMBLE_BINARY_OP(ADD, QUAT, QUAT);
				DISASSEMBLE_BINARY_OP(ADD, COLOR, COLOR);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, STRING, STRING);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, ARRAY, ARRAY);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, PACKED_BYTE_ARRAY, PACKED_BYTE_ARRAY);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, PACKED_INT32_ARRAY, PACKED_INT32_ARRAY);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, PACKED_INT64_ARRAY, PACKED_INT64_ARRAY);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, PACKED_FLOAT32_ARRAY, PACKED_FLOAT32_ARRAY);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, PACKED_FLOAT64_ARRAY, PACKED_FLOAT64_ARRAY);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, PACKED_STRING_ARRAY, PACKED_STRING_ARRAY);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, PACKED_VECTOR2_ARRAY, PACKED_VECTOR2_ARRAY);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, PACKED_VECTOR3_ARRAY, PACKED_VECTOR3_ARRAY);
				DISASSEMBLE_BINARY_OP_VAR(CONCAT, ADD, PACKED_COLOR_ARRAY, PACKED_COLOR_ARRAY);
				// Subtraction.
				DISASSEMBLE_BINARY_OP_NUM(SUBTRACT);
				DISASSEMBLE_BINARY_OP_VEC(SUBTRACT);
				DISASSEMBLE_BINARY_OP(SUBTRACT, QUAT, QUAT);
				DISASSEMBLE_BINARY_OP(SUBTRACT, COLOR, COLOR);
				// Multiplication.
				DISASSEMBLE_BINARY_OP_NUM(MULTIPLY);
				DISASSEMBLE_BINARY_OP_TYPE_NUM(MULTIPLY, VECTOR2);
				DISASSEMBLE_BINARY_OP_TYPE_NUM(MULTIPLY, VECTOR2I);
				DISASSEMBLE_BINARY_OP_TYPE_NUM(MULTIPLY, VECTOR3);
				DISASSEMBLE_BINARY_OP_TYPE_NUM(MULTIPLY, VECTOR3I);
				DISASSEMBLE_BINARY_OP_TYPE_NUM(MULTIPLY, QUAT);
				DISASSEMBLE_BINARY_OP(MULTIPLY, QUAT, VECTOR3);
				DISASSEMBLE_BINARY_OP_TYPE_NUM(MULTIPLY, COLOR);
				DISASSEMBLE_BINARY_OP(MULTIPLY, TRANSFORM2D, VECTOR2);
				DISASSEMBLE_BINARY_OP(MULTIPLY, TRANSFORM2D, TRANSFORM2D);
				// Division.
				DISASSEMBLE_BINARY_OP_NUM(DIVIDE);
				DISASSEMBLE_BINARY_OP_TYPE_NUM_FORWARD(DIVIDE, VECTOR2);
				DISASSEMBLE_BINARY_OP_TYPE_NUM_FORWARD(DIVIDE, VECTOR2I);
				DISASSEMBLE_BINARY_OP_TYPE_NUM_FORWARD(DIVIDE, VECTOR3);
				DISASSEMBLE_BINARY_OP_TYPE_NUM_FORWARD(DIVIDE, VECTOR3I);
				DISASSEMBLE_BINARY_OP_TYPE_NUM_FORWARD(DIVIDE, COLOR);
				DISASSEMBLE_BINARY_OP(DIVIDE, QUAT, FLOAT);
			// Modulo.
			case OPCODE_OP_MODULO_INT_INT: {
				Variant::Operator op = Variant::OP_MODULE;
				text += "operator typed (INT_INT) ";
				text += DADDR(3);
				text += " = ";
				text += DADDR(1);
				text += " ";
				text += Variant::get_operator_name(Variant::Operator(op));
				text += " ";
				text += DADDR(2);
				incr += 4;
			} break;
				// Unary operators.
				// Negate.
				DISASSEMBLE_UNARY_OP(NEGATE, INT);
				DISASSEMBLE_UNARY_OP(NEGATE, FLOAT);
				DISASSEMBLE_UNARY_OP(NEGATE, VECTOR2);
				DISASSEMBLE_UNARY_OP(NEGATE, VECTOR2I);
				DISASSEMBLE_UNARY_OP(NEGATE, VECTOR3);
				DISASSEMBLE_UNARY_OP(NEGATE, VECTOR3I);
				DISASSEMBLE_UNARY_OP(NEGATE, QUAT);
				DISASSEMBLE_UNARY_OP(NEGATE, COLOR);
				// Bitwise operators.
				DISASSEMBLE_UNARY_OP(BIT_NEGATE, INT);
				DISASSEMBLE_BINARY_OP(BIT_AND, INT, INT);
				DISASSEMBLE_BINARY_OP(BIT_OR, INT, INT);
				DISASSEMBLE_BINARY_OP(BIT_XOR, INT, INT);
				DISASSEMBLE_BINARY_OP_NO_TYPE(SHIFT_LEFT);
				DISASSEMBLE_BINARY_OP_NO_TYPE(SHIFT_RIGHT);
				// Logic operators.
				DISASSEMBLE_UNARY_OP_NO_TYPE(NOT);
				DISASSEMBLE_BINARY_OP_NO_TYPE(AND);
				DISASSEMBLE_BINARY_OP_NO_TYPE(OR);
				// Comparison operators.
				DISASSEMBLE_OPCODE_BINARY_ALL_TYPES(EQUAL);
				DISASSEMBLE_BINARY_OP(EQUAL, INT, FLOAT);
				DISASSEMBLE_BINARY_OP(EQUAL, FLOAT, INT);
				DISASSEMBLE_BINARY_OP(EQUAL, STRING, STRING_NAME);
				DISASSEMBLE_BINARY_OP(EQUAL, STRING_NAME, STRING);
				DISASSEMBLE_BINARY_OP(EQUAL, STRING, NODE_PATH);
				DISASSEMBLE_BINARY_OP(EQUAL, NODE_PATH, STRING);
				DISASSEMBLE_OPCODE_BINARY_ALL_TYPES(NOT_EQUAL);
				DISASSEMBLE_BINARY_OP(NOT_EQUAL, INT, FLOAT);
				DISASSEMBLE_BINARY_OP(NOT_EQUAL, FLOAT, INT);
				DISASSEMBLE_BINARY_OP(NOT_EQUAL, STRING, STRING_NAME);
				DISASSEMBLE_BINARY_OP(NOT_EQUAL, STRING_NAME, STRING);
				DISASSEMBLE_BINARY_OP(NOT_EQUAL, STRING, NODE_PATH);
				DISASSEMBLE_BINARY_OP(NOT_EQUAL, NODE_PATH, STRING);
				DISASSEMBLE_BINARY_OP(LESS, BOOL, BOOL);
				DISASSEMBLE_BINARY_OP_NUM(LESS);
				DISASSEMBLE_BINARY_OP_VEC(LESS);
				DISASSEMBLE_BINARY_OP_NUM(LESS_EQUAL);
				DISASSEMBLE_BINARY_OP_VEC(LESS_EQUAL);
				DISASSEMBLE_BINARY_OP(GREATER, BOOL, BOOL);
				DISASSEMBLE_BINARY_OP_NUM(GREATER);
				DISASSEMBLE_BINARY_OP_VEC(GREATER);
				DISASSEMBLE_BINARY_OP_NUM(GREATER_EQUAL);
				DISASSEMBLE_BINARY_OP_VEC(GREATER_EQUAL);

			case OPCODE_EXTENDS_TEST: {
				text += "is object ";
				text += DADDR(3);
				text += " = ";
				text += DADDR(1);
				text += " is ";
				text += DADDR(2);

				incr += 4;
			} break;
			case OPCODE_IS_BUILTIN: {
				text += "is builtin ";
				text += DADDR(3);
				text += " = ";
				text += DADDR(1);
				text += " is ";
				text += Variant::get_type_name(Variant::Type(_code_ptr[ip + 2]));

				incr += 4;
			} break;
			case OPCODE_SET: {
				text += "set ";
				text += DADDR(1);
				text += "[";
				text += DADDR(2);
				text += "] = ";
				text += DADDR(3);

				incr += 4;
			} break;
			case OPCODE_GET: {
				text += "get ";
				text += DADDR(3);
				text += " = ";
				text += DADDR(1);
				text += "[";
				text += DADDR(2);
				text += "]";

				incr += 4;
			} break;
				DISASSEMBLE_GET_TYPED(STRING, INT);
				DISASSEMBLE_GET_TYPED(STRING, FLOAT);
				DISASSEMBLE_GET_TYPED(VECTOR2, INT);
				DISASSEMBLE_GET_TYPED(VECTOR2, FLOAT);
				DISASSEMBLE_GET_TYPED(VECTOR2, STRING);
				DISASSEMBLE_GET_TYPED(VECTOR2I, INT);
				DISASSEMBLE_GET_TYPED(VECTOR2I, FLOAT);
				DISASSEMBLE_GET_TYPED(VECTOR2I, STRING);
				DISASSEMBLE_GET_TYPED(VECTOR3, INT);
				DISASSEMBLE_GET_TYPED(VECTOR3, FLOAT);
				DISASSEMBLE_GET_TYPED(VECTOR3, STRING);
				DISASSEMBLE_GET_TYPED(VECTOR3I, INT);
				DISASSEMBLE_GET_TYPED(VECTOR3I, FLOAT);
				DISASSEMBLE_GET_TYPED(VECTOR3I, STRING);
				DISASSEMBLE_GET_TYPED(RECT2, STRING);
				DISASSEMBLE_GET_TYPED(RECT2I, STRING);
				DISASSEMBLE_GET_TYPED(TRANSFORM, INT);
				DISASSEMBLE_GET_TYPED(TRANSFORM, FLOAT);
				DISASSEMBLE_GET_TYPED(TRANSFORM, STRING);
				DISASSEMBLE_GET_TYPED(TRANSFORM2D, INT);
				DISASSEMBLE_GET_TYPED(TRANSFORM2D, FLOAT);
				DISASSEMBLE_GET_TYPED(TRANSFORM2D, STRING);
				DISASSEMBLE_GET_TYPED(PLANE, STRING);
				DISASSEMBLE_GET_TYPED(QUAT, STRING);
				DISASSEMBLE_GET_TYPED(AABB, STRING);
				DISASSEMBLE_GET_TYPED(BASIS, INT);
				DISASSEMBLE_GET_TYPED(BASIS, FLOAT);
				DISASSEMBLE_GET_TYPED(BASIS, STRING);
				DISASSEMBLE_GET_TYPED(COLOR, INT);
				DISASSEMBLE_GET_TYPED(COLOR, FLOAT);
				DISASSEMBLE_GET_TYPED(COLOR, STRING);
				DISASSEMBLE_GET_TYPED(OBJECT, STRING);
			case OPCODE_SET_NAMED: {
				text += "set_named ";
				text += DADDR(1);
				text += "[\"";
				text += _global_names_ptr[_code_ptr[ip + 2]];
				text += "\"] = ";
				text += DADDR(3);

				incr += 4;
			} break;
			case OPCODE_GET_NAMED: {
				text += "get_named ";
				text += DADDR(3);
				text += " = ";
				text += DADDR(1);
				text += "[\"";
				text += _global_names_ptr[_code_ptr[ip + 2]];
				text += "\"]";

				incr += 4;
			} break;
				DISASSEMBLE_GET_NAMED_TYPED(VECTOR2);
				DISASSEMBLE_GET_NAMED_TYPED(VECTOR2I);
				DISASSEMBLE_GET_NAMED_TYPED(VECTOR3);
				DISASSEMBLE_GET_NAMED_TYPED(VECTOR3I);
				DISASSEMBLE_GET_NAMED_TYPED(RECT2);
				DISASSEMBLE_GET_NAMED_TYPED(RECT2I);
				DISASSEMBLE_GET_NAMED_TYPED(TRANSFORM);
				DISASSEMBLE_GET_NAMED_TYPED(TRANSFORM2D);
				DISASSEMBLE_GET_NAMED_TYPED(PLANE);
				DISASSEMBLE_GET_NAMED_TYPED(QUAT);
				DISASSEMBLE_GET_NAMED_TYPED(BASIS);
				DISASSEMBLE_GET_NAMED_TYPED(AABB);
				DISASSEMBLE_GET_NAMED_TYPED(COLOR);
				DISASSEMBLE_GET_NAMED_TYPED(OBJECT);
			case OPCODE_SET_MEMBER: {
				text += "set_member ";
				text += "[\"";
				text += _global_names_ptr[_code_ptr[ip + 1]];
				text += "\"] = ";
				text += DADDR(2);

				incr += 3;
			} break;
			case OPCODE_GET_MEMBER: {
				text += "get_member ";
				text += DADDR(2);
				text += " = ";
				text += "[\"";
				text += _global_names_ptr[_code_ptr[ip + 1]];
				text += "\"]";

				incr += 3;
			} break;
			case OPCODE_ASSIGN: {
				text += "assign ";
				text += DADDR(1);
				text += " = ";
				text += DADDR(2);

				incr += 3;
			} break;
			case OPCODE_ASSIGN_TRUE: {
				text += "assign ";
				text += DADDR(1);
				text += " = true";

				incr += 2;
			} break;
			case OPCODE_ASSIGN_FALSE: {
				text += "assign ";
				text += DADDR(1);
				text += " = false";

				incr += 2;
			} break;
			case OPCODE_ASSIGN_TYPED_BUILTIN: {
				text += "assign-typed-builtin (";
				text += Variant::get_type_name((Variant::Type)_code_ptr[ip + 1]);
				text += ") ";
				text += DADDR(2);
				text += " = ";
				text += DADDR(3);

				incr += 4;
			} break;
			case OPCODE_ASSIGN_TYPED_NATIVE: {
				Variant class_name = _constants_ptr[_code_ptr[ip + 1]];
				GDScriptNativeClass *nc = Object::cast_to<GDScriptNativeClass>(class_name.operator Object *());

				text += "assign-typed-native (";
				text += nc->get_name().operator String();
				text += ") ";
				text += DADDR(2);
				text += " = ";
				text += DADDR(3);

				incr += 4;
			} break;
			case OPCODE_ASSIGN_TYPED_SCRIPT: {
				Variant script = _constants_ptr[_code_ptr[ip + 1]];
				Script *sc = Object::cast_to<Script>(script.operator Object *());

				text += "assign-typed-script (";
				text += sc->get_path();
				text += ") ";
				text += DADDR(2);
				text += " = ";
				text += DADDR(3);

				incr += 4;
			} break;
			case OPCODE_CAST_TO_BUILTIN: {
				text += "cast-built-in ";
				text += DADDR(3);
				text += " = ";
				text += DADDR(2);
				text += " as ";
				text += Variant::get_type_name(Variant::Type(_code_ptr[ip + 1]));

				incr += 4;
			} break;
			case OPCODE_CAST_TO_NATIVE: {
				Variant class_name = _constants_ptr[_code_ptr[ip + 1]];
				GDScriptNativeClass *nc = Object::cast_to<GDScriptNativeClass>(class_name.operator Object *());

				text += "cast-native ";
				text += DADDR(3);
				text += " = ";
				text += DADDR(2);
				text += " as ";
				text += nc->get_name();

				incr += 4;
			} break;
			case OPCODE_CAST_TO_SCRIPT: {
				text += "cast ";
				text += DADDR(3);
				text += " = ";
				text += DADDR(2);
				text += " as ";
				text += DADDR(1);

				incr += 4;
			} break;
			case OPCODE_CONSTRUCT: {
				Variant::Type t = Variant::Type(_code_ptr[ip + 1]);
				int argc = _code_ptr[ip + 2];

				text += "construct ";
				text += DADDR(3 + argc);
				text += " = ";

				text += Variant::get_type_name(t) + "(";
				for (int i = 0; i < argc; i++) {
					if (i > 0)
						text += ", ";
					text += DADDR(i + 3);
				}
				text += ")";

				incr = 4 + argc;
			} break;
			case OPCODE_CONSTRUCT_ARRAY: {
				int argc = _code_ptr[ip + 1];
				text += "make_array ";
				text += DADDR(2 + argc);
				text += " = [";

				for (int i = 0; i < argc; i++) {
					if (i > 0)
						text += ", ";
					text += DADDR(2 + i);
				}

				text += "]";

				incr += 3 + argc;
			} break;
			case OPCODE_CONSTRUCT_DICTIONARY: {
				int argc = _code_ptr[ip + 1];
				text += "make_dict ";
				text += DADDR(2 + argc * 2);
				text += " = {";

				for (int i = 0; i < argc; i++) {
					if (i > 0)
						text += ", ";
					text += DADDR(2 + i * 2 + 0);
					text += ": ";
					text += DADDR(2 + i * 2 + 1);
				}

				text += "}";

				incr += 3 + argc * 2;
			} break;
			case OPCODE_CALL:
			case OPCODE_CALL_RETURN:
			case OPCODE_CALL_ASYNC: {
				bool ret = _code_ptr[ip] == OPCODE_CALL_RETURN;
				bool async = _code_ptr[ip] == OPCODE_CALL_ASYNC;

				if (ret) {
					text += "call-ret ";
				} else if (async) {
					text += "call-async ";
				} else {
					text += "call ";
				}

				int argc = _code_ptr[ip + 1];
				if (ret || async) {
					text += DADDR(4 + argc) + " = ";
				}

				text += DADDR(2) + ".";
				text += String(_global_names_ptr[_code_ptr[ip + 3]]);
				text += "(";

				for (int i = 0; i < argc; i++) {
					if (i > 0)
						text += ", ";
					text += DADDR(4 + i);
				}
				text += ")";

				incr = 5 + argc;
			} break;
			case OPCODE_CALL_BUILT_IN: {
				text += "call-built-in ";

				int argc = _code_ptr[ip + 2];
				text += DADDR(3 + argc) + " = ";

				text += GDScriptFunctions::get_func_name(GDScriptFunctions::Function(_code_ptr[ip + 1]));
				text += "(";

				for (int i = 0; i < argc; i++) {
					if (i > 0)
						text += ", ";
					text += DADDR(3 + i);
				}
				text += ")";

				incr = 4 + argc;
			} break;
			case OPCODE_CALL_SELF_BASE: {
				text += "call-self-base ";

				int argc = _code_ptr[ip + 2];
				text += DADDR(3 + argc) + " = ";

				text += _global_names_ptr[_code_ptr[ip + 1]];
				text += "(";

				for (int i = 0; i < argc; i++) {
					if (i > 0)
						text += ", ";
					text += DADDR(3 + i);
				}
				text += ")";

				incr = 4 + argc;
			} break;
			case OPCODE_CALL_METHOD_BIND:
			case OPCODE_CALL_METHOD_BIND_RET: {
				bool ret = _code_ptr[ip] == OPCODE_CALL_METHOD_BIND_RET;
				int argc = _code_ptr[ip + 1];
				MethodBind *method = _methods_ptr[_code_ptr[ip + 3]];

				text += "call-method_bind ";
				if (ret) {
					text += "ret ";
				}

				text += " (base ";
				text += method->get_instance_class();
				text += ") ";

				if (ret) {
					text += DADDR(4 + argc) + " = ";
				}

				text += DADDR(2) + ".";
				text += method->get_name();

				text += "(";

				for (int i = 0; i < argc; i++) {
					if (i > 0)
						text += ", ";
					text += DADDR(4 + i);
				}
				text += ")";

				incr = 5 + argc;
			} break;

			case OPCODE_CALL_BUILTIN_TYPE_FUNC:
			case OPCODE_CALL_BUILTIN_TYPE_FUNC_VALIDATED_NO_RET:
			case OPCODE_CALL_BUILTIN_TYPE_FUNC_VALIDATED_RET:
			case OPCODE_CALL_BUILTIN_TYPE_FUNC_VALIDATED_RET_NULL: {
				int argc = _code_ptr[ip + 1];
				Variant::InternalMethod *method = _variant_methods_ptr[_code_ptr[ip + 3]];

				text += "call-builtin-method ";
				if (_code_ptr[ip] != OPCODE_CALL_BUILTIN_TYPE_FUNC) {
					text += " (validated) ";
				}

				text += " (base ";
				text += Variant::get_type_name(method->get_base_type());
				text += ") ";

				text += DADDR(4 + argc) + " = ";

				text += DADDR(2) + ".";
				text += method->get_name();

				text += "(";

				for (int i = 0; i < argc; i++) {
					if (i > 0)
						text += ", ";
					text += DADDR(4 + i);
				}
				text += ")";

				incr = 5 + argc;
			} break;

				DISASSEMBLE_CALL_PTRCALL(PTRCALL, NO_RETURN);
				DISASSEMBLE_OPCODE_ALL_TYPES(DISASSEMBLE_CALL_PTRCALL, PTRCALL);
				DISASSEMBLE_BUILTIN_CALL(SIN, sin, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(SIN, sin, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(COS, cos, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(COS, cos, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(TAN, tan, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(TAN, tan, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(SINH, sinh, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(SINH, sinh, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(COSH, cosh, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(COSH, cosh, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(TANH, tanh, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(TANH, tanh, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(ASIN, asin, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(ASIN, asin, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(ACOS, acos, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(ACOS, acos, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(ATAN, atan, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(ATAN, atan, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(ATAN2, atan, 2, INT);
				DISASSEMBLE_BUILTIN_CALL(ATAN2, atan, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(SQRT, sqrt, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(SQRT, sqrt, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(FMOD, fmod, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(FPOSMOD, fposmod, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(POSMOD, posmod, 2, INT);
				DISASSEMBLE_BUILTIN_CALL(FLOOR, floor, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(FLOOR, floor, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(CEIL, ceil, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(CEIL, ceil, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(POW, pow, 2, INT);
				DISASSEMBLE_BUILTIN_CALL(POW, pow, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(LOG, log, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(LOG, log, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(EXP, exp, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(EXP, exp, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(DEG2RAD, deg2rad, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(DEG2RAD, deg2rad, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(RAD2DEG, rad2deg, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(RAD2DEG, rad2deg, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(LINEAR2DB, linear2db, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(LINEAR2DB, linear2db, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(DB2LINEAR, db2linear, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(DB2LINEAR, db2linear, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(ROUND, round, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(IS_INF, is_inf, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(IS_NAN, is_nan, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(IS_ZERO_APPROX, is_zero_approx, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(IS_EQUAL_APPROX, is_equal_approx, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(EASE, ease, 2, INT);
				DISASSEMBLE_BUILTIN_CALL(EASE, ease, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(RANDOM, rand_range, 2, INT);
				DISASSEMBLE_BUILTIN_CALL(RANDOM, rand_range, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(RAND, randi, 0, INT);
				DISASSEMBLE_BUILTIN_CALL(RANDF, randf, 0, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(STEP_DECIMALS, step_decimals, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(STEP_DECIMALS, step_decimals, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(STEPIFY, stepify, 2, INT);
				DISASSEMBLE_BUILTIN_CALL(STEPIFY, stepify, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(LERP, lerp, 3, INT);
				DISASSEMBLE_BUILTIN_CALL(LERP, lerp, 3, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(LERP, lerp, 3, VECTOR2);
				DISASSEMBLE_BUILTIN_CALL(LERP, lerp, 3, VECTOR3);
				DISASSEMBLE_BUILTIN_CALL(LERP, lerp, 3, COLOR);
				DISASSEMBLE_BUILTIN_CALL(LERP_ANGLE, lerp_angle, 3, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(INVERSE_LERP, inverse_lerp, 3, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(SMOOTHSTEP, smoothstep, 3, INT);
				DISASSEMBLE_BUILTIN_CALL(SMOOTHSTEP, smoothstep, 3, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(MOVE_TOWARD, move_toward, 3, INT);
				DISASSEMBLE_BUILTIN_CALL(MOVE_TOWARD, move_toward, 3, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(DECTIME, dectime, 3, INT);
				DISASSEMBLE_BUILTIN_CALL(DECTIME, dectime, 3, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(NEAREST_PO2, nearest_po2, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(NEAREST_PO2, nearest_po2, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(ABS, abs, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(ABS, abs, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(WRAP, wrapi, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(WRAP, wrapf, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(MAX, max, 2, INT);
				DISASSEMBLE_BUILTIN_CALL(MAX, max, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(MIN, min, 2, INT);
				DISASSEMBLE_BUILTIN_CALL(MIN, min, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(CLAMP, clamp, 3, INT);
				DISASSEMBLE_BUILTIN_CALL(CLAMP, clamp, 3, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(SIGN, sign, 1, INT);
				DISASSEMBLE_BUILTIN_CALL(SIGN, sign, 1, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(CARTESIAN2POLAR, cartesian2polar, 2, FLOAT);
				DISASSEMBLE_BUILTIN_CALL(POLAR2CARTESIAN, polar2cartesian, 2, FLOAT);

			case OPCODE_AWAIT: {
				text += "await ";
				text += DADDR(1);

				incr += 2;
			} break;
			case OPCODE_AWAIT_RESUME: {
				text += "await resume ";
				text += DADDR(1);

				incr = 2;
			} break;
			case OPCODE_JUMP: {
				text += "jump ";
				text += itos(_code_ptr[ip + 1]);

				incr = 2;
			} break;
			case OPCODE_JUMP_IF: {
				text += "jump-if ";
				text += DADDR(1);
				text += " to ";
				text += itos(_code_ptr[ip + 2]);

				incr = 3;
			} break;
			case OPCODE_JUMP_IF_NOT: {
				text += "jump-if-not ";
				text += DADDR(1);
				text += " to ";
				text += itos(_code_ptr[ip + 2]);

				incr = 3;
			} break;
			case OPCODE_JUMP_TO_DEF_ARGUMENT: {
				text += "jump-to-default-argument ";

				incr = 1;
			} break;
			case OPCODE_RETURN: {
				text += "return ";
				text += DADDR(1);

				incr = 2;
			} break;
			case OPCODE_ITERATE_BEGIN: {
				text += "for-init ";
				text += DADDR(4);
				text += " in ";
				text += DADDR(2);
				text += " counter ";
				text += DADDR(1);
				text += " end ";
				text += itos(_code_ptr[ip + 3]);

				incr += 5;
			} break;
				DISASSEMBLE_ITERATE_TYPES(DISASSEMBLE_ITERATE_BEGIN);
			case OPCODE_ITERATE: {
				text += "for-loop ";
				text += DADDR(4);
				text += " in ";
				text += DADDR(2);
				text += " counter ";
				text += DADDR(1);
				text += " end ";
				text += itos(_code_ptr[ip + 3]);

				incr += 5;
			} break;
				DISASSEMBLE_ITERATE_TYPES(DISASSEMBLE_ITERATE);
			case OPCODE_LINE: {
				int line = _code_ptr[ip + 1] - 1;
				if (line >= 0 && line < p_code_lines.size()) {
					text += "line ";
					text += itos(line + 1);
					text += ": ";
					text += p_code_lines[line];
				} else {
					text += "";
				}

				incr += 2;
			} break;
			case OPCODE_ASSERT: {
				text += "assert (";
				text += DADDR(1);
				text += ", ";
				text += DADDR(2);
				text += ")";

				incr += 3;
			} break;
			case OPCODE_BREAKPOINT: {
				text += "breakpoint";

				incr += 1;
			} break;
			case OPCODE_END: {
				text += "== END ==";

				incr += 1;
			} break;
		}

		ip += incr;
		if (text.get_string_length() > 0) {
			print_line(text.as_string());
		}
	}
}
#endif
