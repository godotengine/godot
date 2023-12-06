/**************************************************************************/
/*  variant_op.cpp                                                        */
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

#include "variant_op.h"

typedef void (*VariantEvaluatorFunction)(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid);

static VariantType operator_return_type_table[(int)VariantOperator::MAX][(int)VariantType::MAX][(int)VariantType::MAX];
static VariantEvaluatorFunction operator_evaluator_table[(int)VariantOperator::MAX][(int)VariantType::MAX][(int)VariantType::MAX];
static Variant::ValidatedOperatorEvaluator validated_operator_evaluator_table[(int)VariantOperator::MAX][(int)VariantType::MAX][(int)VariantType::MAX];
static Variant::PTROperatorEvaluator ptr_operator_evaluator_table[(int)VariantOperator::MAX][(int)VariantType::MAX][(int)VariantType::MAX];

template <class T>
void register_op(VariantOperator p_op, VariantType p_type_a, VariantType p_type_b) {
	operator_return_type_table[(int)p_op][(int)p_type_a][(int)p_type_b] = T::get_return_type();
	operator_evaluator_table[(int)p_op][(int)p_type_a][(int)p_type_b] = T::evaluate;
	validated_operator_evaluator_table[(int)p_op][(int)p_type_a][(int)p_type_b] = T::validated_evaluate;
	ptr_operator_evaluator_table[(int)p_op][(int)p_type_a][(int)p_type_b] = T::ptr_evaluate;
}

// Special cases that can't be done otherwise because of the forced casting to float.

template <>
class OperatorEvaluatorMul<Vector2, Vector2i, double> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector2i &a = *VariantGetInternalPtr<Vector2i>::get_ptr(&p_left);
		const double &b = *VariantGetInternalPtr<double>::get_ptr(&p_right);
		*r_ret = Vector2(a.x, a.y) * b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector2>::get_ptr(r_ret) = Vector2(VariantGetInternalPtr<Vector2i>::get_ptr(left)->x, VariantGetInternalPtr<Vector2i>::get_ptr(left)->y) * *VariantGetInternalPtr<double>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector2>::encode(Vector2(PtrToArg<Vector2i>::convert(left).x, PtrToArg<Vector2i>::convert(left).y) * PtrToArg<double>::convert(right), r_ret);
	}
	static VariantType get_return_type() { return GetTypeInfo<Vector2>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorMul<Vector2, double, Vector2i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector2i &a = *VariantGetInternalPtr<Vector2i>::get_ptr(&p_right);
		const double &b = *VariantGetInternalPtr<double>::get_ptr(&p_left);
		*r_ret = Vector2(a.x, a.y) * b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector2>::get_ptr(r_ret) = Vector2(VariantGetInternalPtr<Vector2i>::get_ptr(right)->x, VariantGetInternalPtr<Vector2i>::get_ptr(right)->y) * *VariantGetInternalPtr<double>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector2>::encode(Vector2(PtrToArg<Vector2i>::convert(right).x, PtrToArg<Vector2i>::convert(right).y) * PtrToArg<double>::convert(left), r_ret);
	}
	static VariantType get_return_type() { return GetTypeInfo<Vector2>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorDivNZ<Vector2, Vector2i, double> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector2i &a = *VariantGetInternalPtr<Vector2i>::get_ptr(&p_left);
		const double &b = *VariantGetInternalPtr<double>::get_ptr(&p_right);
		if (unlikely(b == 0)) {
			r_valid = false;
			*r_ret = "Division by zero error";
			return;
		}
		*r_ret = Vector2(a.x, a.y) / b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector2>::get_ptr(r_ret) = Vector2(VariantGetInternalPtr<Vector2i>::get_ptr(left)->x, VariantGetInternalPtr<Vector2i>::get_ptr(left)->y) / *VariantGetInternalPtr<double>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector2>::encode(Vector2(PtrToArg<Vector2i>::convert(left).x, PtrToArg<Vector2i>::convert(left).y) / PtrToArg<double>::convert(right), r_ret);
	}
	static VariantType get_return_type() { return GetTypeInfo<Vector2>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorMul<Vector3, Vector3i, double> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector3i &a = *VariantGetInternalPtr<Vector3i>::get_ptr(&p_left);
		const double &b = *VariantGetInternalPtr<double>::get_ptr(&p_right);
		*r_ret = Vector3(a.x, a.y, a.z) * b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector3>::get_ptr(r_ret) = Vector3(VariantGetInternalPtr<Vector3i>::get_ptr(left)->x, VariantGetInternalPtr<Vector3i>::get_ptr(left)->y, VariantGetInternalPtr<Vector3i>::get_ptr(left)->z) * *VariantGetInternalPtr<double>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector3>::encode(Vector3(PtrToArg<Vector3i>::convert(left).x, PtrToArg<Vector3i>::convert(left).y, PtrToArg<Vector3i>::convert(left).z) * PtrToArg<double>::convert(right), r_ret);
	}
	static VariantType get_return_type() { return GetTypeInfo<Vector3>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorMul<Vector3, double, Vector3i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector3i &a = *VariantGetInternalPtr<Vector3i>::get_ptr(&p_right);
		const double &b = *VariantGetInternalPtr<double>::get_ptr(&p_left);
		*r_ret = Vector3(a.x, a.y, a.z) * b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector3>::get_ptr(r_ret) = Vector3(VariantGetInternalPtr<Vector3i>::get_ptr(right)->x, VariantGetInternalPtr<Vector3i>::get_ptr(right)->y, VariantGetInternalPtr<Vector3i>::get_ptr(right)->z) * *VariantGetInternalPtr<double>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector3>::encode(Vector3(PtrToArg<Vector3i>::convert(right).x, PtrToArg<Vector3i>::convert(right).y, PtrToArg<Vector3i>::convert(right).z) * PtrToArg<double>::convert(left), r_ret);
	}
	static VariantType get_return_type() { return GetTypeInfo<Vector3>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorDivNZ<Vector3, Vector3i, double> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector3i &a = *VariantGetInternalPtr<Vector3i>::get_ptr(&p_left);
		const double &b = *VariantGetInternalPtr<double>::get_ptr(&p_right);
		if (unlikely(b == 0)) {
			r_valid = false;
			*r_ret = "Division by zero error";
			return;
		}
		*r_ret = Vector3(a.x, a.y, a.z) / b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector3>::get_ptr(r_ret) = Vector3(VariantGetInternalPtr<Vector3i>::get_ptr(left)->x, VariantGetInternalPtr<Vector3i>::get_ptr(left)->y, VariantGetInternalPtr<Vector3i>::get_ptr(left)->z) / *VariantGetInternalPtr<double>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector3>::encode(Vector3(PtrToArg<Vector3i>::convert(left).x, PtrToArg<Vector3i>::convert(left).y, PtrToArg<Vector3i>::convert(left).z) / PtrToArg<double>::convert(right), r_ret);
	}
	static VariantType get_return_type() { return GetTypeInfo<Vector3>::VARIANT_TYPE; }
};

//

template <>
class OperatorEvaluatorMul<Vector4, Vector4i, double> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector4i &a = *VariantGetInternalPtr<Vector4i>::get_ptr(&p_left);
		const double &b = *VariantGetInternalPtr<double>::get_ptr(&p_right);
		*r_ret = Vector4(a.x, a.y, a.z, a.w) * b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector4>::get_ptr(r_ret) = Vector4(VariantGetInternalPtr<Vector4i>::get_ptr(left)->x, VariantGetInternalPtr<Vector4i>::get_ptr(left)->y, VariantGetInternalPtr<Vector4i>::get_ptr(left)->z, VariantGetInternalPtr<Vector4i>::get_ptr(left)->w) * *VariantGetInternalPtr<double>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector4>::encode(Vector4(PtrToArg<Vector4i>::convert(left).x, PtrToArg<Vector4i>::convert(left).y, PtrToArg<Vector4i>::convert(left).z, PtrToArg<Vector4i>::convert(left).w) * PtrToArg<double>::convert(right), r_ret);
	}
	static VariantType get_return_type() { return GetTypeInfo<Vector4>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorMul<Vector4, double, Vector4i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector4i &a = *VariantGetInternalPtr<Vector4i>::get_ptr(&p_right);
		const double &b = *VariantGetInternalPtr<double>::get_ptr(&p_left);
		*r_ret = Vector4(a.x, a.y, a.z, a.w) * b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector4>::get_ptr(r_ret) = Vector4(VariantGetInternalPtr<Vector4i>::get_ptr(right)->x, VariantGetInternalPtr<Vector4i>::get_ptr(right)->y, VariantGetInternalPtr<Vector4i>::get_ptr(right)->z, VariantGetInternalPtr<Vector4i>::get_ptr(right)->w) * *VariantGetInternalPtr<double>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector4>::encode(Vector4(PtrToArg<Vector4i>::convert(right).x, PtrToArg<Vector4i>::convert(right).y, PtrToArg<Vector4i>::convert(right).z, PtrToArg<Vector4i>::convert(right).w) * PtrToArg<double>::convert(left), r_ret);
	}
	static VariantType get_return_type() { return GetTypeInfo<Vector4>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorDivNZ<Vector4, Vector4i, double> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector4i &a = *VariantGetInternalPtr<Vector4i>::get_ptr(&p_left);
		const double &b = *VariantGetInternalPtr<double>::get_ptr(&p_right);
		if (unlikely(b == 0)) {
			r_valid = false;
			*r_ret = "Division by zero error";
			return;
		}
		*r_ret = Vector4(a.x, a.y, a.z, a.w) / b;
		r_valid = true;
	}

	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector4>::get_ptr(r_ret) = Vector4(VariantGetInternalPtr<Vector4i>::get_ptr(left)->x, VariantGetInternalPtr<Vector4i>::get_ptr(left)->y, VariantGetInternalPtr<Vector4i>::get_ptr(left)->z, VariantGetInternalPtr<Vector4i>::get_ptr(left)->w) / *VariantGetInternalPtr<double>::get_ptr(right);
	}

	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector4>::encode(Vector4(PtrToArg<Vector4i>::convert(left).x, PtrToArg<Vector4i>::convert(left).y, PtrToArg<Vector4i>::convert(left).z, PtrToArg<Vector4i>::convert(left).w) / PtrToArg<double>::convert(right), r_ret);
	}

	static VariantType get_return_type() { return GetTypeInfo<Vector4>::VARIANT_TYPE; }
};

#define register_string_op(m_op_type, m_op_code)                                                                       \
	do {                                                                                                               \
		register_op<m_op_type<String, String>>(m_op_code, VariantType::STRING, VariantType::STRING);                   \
		register_op<m_op_type<String, StringName>>(m_op_code, VariantType::STRING, VariantType::STRING_NAME);          \
		register_op<m_op_type<StringName, String>>(m_op_code, VariantType::STRING_NAME, VariantType::STRING);          \
		register_op<m_op_type<StringName, StringName>>(m_op_code, VariantType::STRING_NAME, VariantType::STRING_NAME); \
	} while (false)

#define register_string_modulo_op(m_class, m_type)                                                                                  \
	do {                                                                                                                            \
		register_op<OperatorEvaluatorStringFormat<String, m_class>>(VariantOperator::MODULE, VariantType::STRING, m_type);          \
		register_op<OperatorEvaluatorStringFormat<StringName, m_class>>(VariantOperator::MODULE, VariantType::STRING_NAME, m_type); \
	} while (false)

void Variant::_register_variant_operators() {
	memset(operator_return_type_table, 0, sizeof(operator_return_type_table));
	memset(operator_evaluator_table, 0, sizeof(operator_evaluator_table));
	memset(validated_operator_evaluator_table, 0, sizeof(validated_operator_evaluator_table));
	memset(ptr_operator_evaluator_table, 0, sizeof(ptr_operator_evaluator_table));

	register_op<OperatorEvaluatorAdd<int64_t, int64_t, int64_t>>(VariantOperator::ADD, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorAdd<double, int64_t, double>>(VariantOperator::ADD, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorAdd<double, double, int64_t>>(VariantOperator::ADD, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorAdd<double, double, double>>(VariantOperator::ADD, VariantType::FLOAT, VariantType::FLOAT);
	register_string_op(OperatorEvaluatorStringConcat, VariantOperator::ADD);
	register_op<OperatorEvaluatorAdd<Vector2, Vector2, Vector2>>(VariantOperator::ADD, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorAdd<Vector2i, Vector2i, Vector2i>>(VariantOperator::ADD, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorAdd<Vector3, Vector3, Vector3>>(VariantOperator::ADD, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorAdd<Vector3i, Vector3i, Vector3i>>(VariantOperator::ADD, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorAdd<Vector4, Vector4, Vector4>>(VariantOperator::ADD, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorAdd<Vector4i, Vector4i, Vector4i>>(VariantOperator::ADD, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorAdd<Quaternion, Quaternion, Quaternion>>(VariantOperator::ADD, VariantType::QUATERNION, VariantType::QUATERNION);
	register_op<OperatorEvaluatorAdd<Color, Color, Color>>(VariantOperator::ADD, VariantType::COLOR, VariantType::COLOR);
	register_op<OperatorEvaluatorAddArray>(VariantOperator::ADD, VariantType::ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorAppendArray<uint8_t>>(VariantOperator::ADD, VariantType::PACKED_BYTE_ARRAY, VariantType::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorAppendArray<int32_t>>(VariantOperator::ADD, VariantType::PACKED_INT32_ARRAY, VariantType::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorAppendArray<int64_t>>(VariantOperator::ADD, VariantType::PACKED_INT64_ARRAY, VariantType::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorAppendArray<float>>(VariantOperator::ADD, VariantType::PACKED_FLOAT32_ARRAY, VariantType::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorAppendArray<double>>(VariantOperator::ADD, VariantType::PACKED_FLOAT64_ARRAY, VariantType::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorAppendArray<String>>(VariantOperator::ADD, VariantType::PACKED_STRING_ARRAY, VariantType::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorAppendArray<Vector2>>(VariantOperator::ADD, VariantType::PACKED_VECTOR2_ARRAY, VariantType::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorAppendArray<Vector3>>(VariantOperator::ADD, VariantType::PACKED_VECTOR3_ARRAY, VariantType::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorAppendArray<Color>>(VariantOperator::ADD, VariantType::PACKED_COLOR_ARRAY, VariantType::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorSub<int64_t, int64_t, int64_t>>(VariantOperator::SUBTRACT, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorSub<double, int64_t, double>>(VariantOperator::SUBTRACT, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorSub<double, double, int64_t>>(VariantOperator::SUBTRACT, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorSub<double, double, double>>(VariantOperator::SUBTRACT, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorSub<Vector2, Vector2, Vector2>>(VariantOperator::SUBTRACT, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorSub<Vector2i, Vector2i, Vector2i>>(VariantOperator::SUBTRACT, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorSub<Vector3, Vector3, Vector3>>(VariantOperator::SUBTRACT, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorSub<Vector3i, Vector3i, Vector3i>>(VariantOperator::SUBTRACT, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorSub<Vector4, Vector4, Vector4>>(VariantOperator::SUBTRACT, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorSub<Vector4i, Vector4i, Vector4i>>(VariantOperator::SUBTRACT, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorSub<Quaternion, Quaternion, Quaternion>>(VariantOperator::SUBTRACT, VariantType::QUATERNION, VariantType::QUATERNION);
	register_op<OperatorEvaluatorSub<Color, Color, Color>>(VariantOperator::SUBTRACT, VariantType::COLOR, VariantType::COLOR);

	register_op<OperatorEvaluatorMul<int64_t, int64_t, int64_t>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorMul<double, int64_t, double>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorMul<Vector2, int64_t, Vector2>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::VECTOR2);
	register_op<OperatorEvaluatorMul<Vector2i, int64_t, Vector2i>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorMul<Vector3, int64_t, Vector3>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::VECTOR3);
	register_op<OperatorEvaluatorMul<Vector3i, int64_t, Vector3i>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorMul<Vector4, int64_t, Vector4>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::VECTOR4);
	register_op<OperatorEvaluatorMul<Vector4i, int64_t, Vector4i>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::VECTOR4I);

	register_op<OperatorEvaluatorMul<double, double, double>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorMul<double, double, int64_t>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorMul<Vector2, double, Vector2>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::VECTOR2);
	register_op<OperatorEvaluatorMul<Vector2, double, Vector2i>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorMul<Vector3, double, Vector3>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::VECTOR3);
	register_op<OperatorEvaluatorMul<Vector3, double, Vector3i>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorMul<Vector4, double, Vector4>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::VECTOR4);
	register_op<OperatorEvaluatorMul<Vector4, double, Vector4i>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::VECTOR4I);

	register_op<OperatorEvaluatorMul<Vector2, Vector2, Vector2>>(VariantOperator::MULTIPLY, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorMul<Vector2, Vector2, int64_t>>(VariantOperator::MULTIPLY, VariantType::VECTOR2, VariantType::INT);
	register_op<OperatorEvaluatorMul<Vector2, Vector2, double>>(VariantOperator::MULTIPLY, VariantType::VECTOR2, VariantType::FLOAT);

	register_op<OperatorEvaluatorMul<Vector2i, Vector2i, Vector2i>>(VariantOperator::MULTIPLY, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorMul<Vector2i, Vector2i, int64_t>>(VariantOperator::MULTIPLY, VariantType::VECTOR2I, VariantType::INT);
	register_op<OperatorEvaluatorMul<Vector2, Vector2i, double>>(VariantOperator::MULTIPLY, VariantType::VECTOR2I, VariantType::FLOAT);

	register_op<OperatorEvaluatorMul<Vector3, Vector3, Vector3>>(VariantOperator::MULTIPLY, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorMul<Vector3, Vector3, int64_t>>(VariantOperator::MULTIPLY, VariantType::VECTOR3, VariantType::INT);
	register_op<OperatorEvaluatorMul<Vector3, Vector3, double>>(VariantOperator::MULTIPLY, VariantType::VECTOR3, VariantType::FLOAT);

	register_op<OperatorEvaluatorMul<Vector3i, Vector3i, Vector3i>>(VariantOperator::MULTIPLY, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorMul<Vector3i, Vector3i, int64_t>>(VariantOperator::MULTIPLY, VariantType::VECTOR3I, VariantType::INT);
	register_op<OperatorEvaluatorMul<Vector3, Vector3i, double>>(VariantOperator::MULTIPLY, VariantType::VECTOR3I, VariantType::FLOAT);

	register_op<OperatorEvaluatorMul<Vector4, Vector4, Vector4>>(VariantOperator::MULTIPLY, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorMul<Vector4, Vector4, int64_t>>(VariantOperator::MULTIPLY, VariantType::VECTOR4, VariantType::INT);
	register_op<OperatorEvaluatorMul<Vector4, Vector4, double>>(VariantOperator::MULTIPLY, VariantType::VECTOR4, VariantType::FLOAT);

	register_op<OperatorEvaluatorMul<Vector4i, Vector4i, Vector4i>>(VariantOperator::MULTIPLY, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorMul<Vector4i, Vector4i, int64_t>>(VariantOperator::MULTIPLY, VariantType::VECTOR4I, VariantType::INT);
	register_op<OperatorEvaluatorMul<Vector4, Vector4i, double>>(VariantOperator::MULTIPLY, VariantType::VECTOR4I, VariantType::FLOAT);

	register_op<OperatorEvaluatorMul<Quaternion, Quaternion, Quaternion>>(VariantOperator::MULTIPLY, VariantType::QUATERNION, VariantType::QUATERNION);
	register_op<OperatorEvaluatorMul<Quaternion, Quaternion, int64_t>>(VariantOperator::MULTIPLY, VariantType::QUATERNION, VariantType::INT);
	register_op<OperatorEvaluatorMul<Quaternion, Quaternion, double>>(VariantOperator::MULTIPLY, VariantType::QUATERNION, VariantType::FLOAT);

	register_op<OperatorEvaluatorMul<Color, Color, Color>>(VariantOperator::MULTIPLY, VariantType::COLOR, VariantType::COLOR);
	register_op<OperatorEvaluatorMul<Color, Color, int64_t>>(VariantOperator::MULTIPLY, VariantType::COLOR, VariantType::INT);
	register_op<OperatorEvaluatorMul<Color, Color, double>>(VariantOperator::MULTIPLY, VariantType::COLOR, VariantType::FLOAT);

	register_op<OperatorEvaluatorMul<Transform2D, Transform2D, Transform2D>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM2D, VariantType::TRANSFORM2D);
	register_op<OperatorEvaluatorMul<Transform2D, Transform2D, int64_t>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM2D, VariantType::INT);
	register_op<OperatorEvaluatorMul<Transform2D, Transform2D, double>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM2D, VariantType::FLOAT);
	register_op<OperatorEvaluatorXForm<Vector2, Transform2D, Vector2>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM2D, VariantType::VECTOR2);
	register_op<OperatorEvaluatorXFormInv<Vector2, Vector2, Transform2D>>(VariantOperator::MULTIPLY, VariantType::VECTOR2, VariantType::TRANSFORM2D);
	register_op<OperatorEvaluatorXForm<Rect2, Transform2D, Rect2>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM2D, VariantType::RECT2);
	register_op<OperatorEvaluatorXFormInv<Rect2, Rect2, Transform2D>>(VariantOperator::MULTIPLY, VariantType::RECT2, VariantType::TRANSFORM2D);
	register_op<OperatorEvaluatorXForm<Vector<Vector2>, Transform2D, Vector<Vector2>>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM2D, VariantType::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorXFormInv<Vector<Vector2>, Vector<Vector2>, Transform2D>>(VariantOperator::MULTIPLY, VariantType::PACKED_VECTOR2_ARRAY, VariantType::TRANSFORM2D);

	register_op<OperatorEvaluatorMul<Transform3D, Transform3D, Transform3D>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM3D, VariantType::TRANSFORM3D);
	register_op<OperatorEvaluatorMul<Transform3D, Transform3D, int64_t>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM3D, VariantType::INT);
	register_op<OperatorEvaluatorMul<Transform3D, Transform3D, double>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM3D, VariantType::FLOAT);
	register_op<OperatorEvaluatorXForm<Vector3, Transform3D, Vector3>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM3D, VariantType::VECTOR3);
	register_op<OperatorEvaluatorXFormInv<Vector3, Vector3, Transform3D>>(VariantOperator::MULTIPLY, VariantType::VECTOR3, VariantType::TRANSFORM3D);
	register_op<OperatorEvaluatorXForm<AABB, Transform3D, AABB>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM3D, VariantType::AABB);
	register_op<OperatorEvaluatorXFormInv<AABB, AABB, Transform3D>>(VariantOperator::MULTIPLY, VariantType::AABB, VariantType::TRANSFORM3D);
	register_op<OperatorEvaluatorXForm<Plane, Transform3D, Plane>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM3D, VariantType::PLANE);
	register_op<OperatorEvaluatorXFormInv<Plane, Plane, Transform3D>>(VariantOperator::MULTIPLY, VariantType::PLANE, VariantType::TRANSFORM3D);
	register_op<OperatorEvaluatorXForm<Vector<Vector3>, Transform3D, Vector<Vector3>>>(VariantOperator::MULTIPLY, VariantType::TRANSFORM3D, VariantType::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorXFormInv<Vector<Vector3>, Vector<Vector3>, Transform3D>>(VariantOperator::MULTIPLY, VariantType::PACKED_VECTOR3_ARRAY, VariantType::TRANSFORM3D);

	register_op<OperatorEvaluatorXForm<Vector4, Projection, Vector4>>(VariantOperator::MULTIPLY, VariantType::PROJECTION, VariantType::VECTOR4);
	register_op<OperatorEvaluatorXFormInv<Vector4, Vector4, Projection>>(VariantOperator::MULTIPLY, VariantType::VECTOR4, VariantType::PROJECTION);

	register_op<OperatorEvaluatorMul<Projection, Projection, Projection>>(VariantOperator::MULTIPLY, VariantType::PROJECTION, VariantType::PROJECTION);

	register_op<OperatorEvaluatorMul<Basis, Basis, Basis>>(VariantOperator::MULTIPLY, VariantType::BASIS, VariantType::BASIS);
	register_op<OperatorEvaluatorMul<Basis, Basis, int64_t>>(VariantOperator::MULTIPLY, VariantType::BASIS, VariantType::INT);
	register_op<OperatorEvaluatorMul<Basis, Basis, double>>(VariantOperator::MULTIPLY, VariantType::BASIS, VariantType::FLOAT);
	register_op<OperatorEvaluatorXForm<Vector3, Basis, Vector3>>(VariantOperator::MULTIPLY, VariantType::BASIS, VariantType::VECTOR3);
	register_op<OperatorEvaluatorXFormInv<Vector3, Vector3, Basis>>(VariantOperator::MULTIPLY, VariantType::VECTOR3, VariantType::BASIS);

	register_op<OperatorEvaluatorMul<Quaternion, Quaternion, Quaternion>>(VariantOperator::MULTIPLY, VariantType::QUATERNION, VariantType::QUATERNION);
	register_op<OperatorEvaluatorMul<Quaternion, Quaternion, int64_t>>(VariantOperator::MULTIPLY, VariantType::QUATERNION, VariantType::INT);
	register_op<OperatorEvaluatorMul<Quaternion, int64_t, Quaternion>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::QUATERNION);
	register_op<OperatorEvaluatorMul<Quaternion, Quaternion, double>>(VariantOperator::MULTIPLY, VariantType::QUATERNION, VariantType::FLOAT);
	register_op<OperatorEvaluatorMul<Quaternion, double, Quaternion>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::QUATERNION);
	register_op<OperatorEvaluatorXForm<Vector3, Quaternion, Vector3>>(VariantOperator::MULTIPLY, VariantType::QUATERNION, VariantType::VECTOR3);
	register_op<OperatorEvaluatorXFormInv<Vector3, Vector3, Quaternion>>(VariantOperator::MULTIPLY, VariantType::VECTOR3, VariantType::QUATERNION);

	register_op<OperatorEvaluatorMul<Color, Color, Color>>(VariantOperator::MULTIPLY, VariantType::COLOR, VariantType::COLOR);
	register_op<OperatorEvaluatorMul<Color, Color, int64_t>>(VariantOperator::MULTIPLY, VariantType::COLOR, VariantType::INT);
	register_op<OperatorEvaluatorMul<Color, int64_t, Color>>(VariantOperator::MULTIPLY, VariantType::INT, VariantType::COLOR);
	register_op<OperatorEvaluatorMul<Color, Color, double>>(VariantOperator::MULTIPLY, VariantType::COLOR, VariantType::FLOAT);
	register_op<OperatorEvaluatorMul<Color, double, Color>>(VariantOperator::MULTIPLY, VariantType::FLOAT, VariantType::COLOR);

	register_op<OperatorEvaluatorDivNZ<int64_t, int64_t, int64_t>>(VariantOperator::DIVIDE, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorDiv<double, double, int64_t>>(VariantOperator::DIVIDE, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorDiv<double, int64_t, double>>(VariantOperator::DIVIDE, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorDiv<double, double, double>>(VariantOperator::DIVIDE, VariantType::FLOAT, VariantType::FLOAT);

	register_op<OperatorEvaluatorDiv<Vector2, Vector2, Vector2>>(VariantOperator::DIVIDE, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorDiv<Vector2, Vector2, double>>(VariantOperator::DIVIDE, VariantType::VECTOR2, VariantType::FLOAT);
	register_op<OperatorEvaluatorDiv<Vector2, Vector2, int64_t>>(VariantOperator::DIVIDE, VariantType::VECTOR2, VariantType::INT);

	register_op<OperatorEvaluatorDivNZ<Vector2i, Vector2i, Vector2i>>(VariantOperator::DIVIDE, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorDivNZ<Vector2, Vector2i, double>>(VariantOperator::DIVIDE, VariantType::VECTOR2I, VariantType::FLOAT);
	register_op<OperatorEvaluatorDivNZ<Vector2i, Vector2i, int64_t>>(VariantOperator::DIVIDE, VariantType::VECTOR2I, VariantType::INT);

	register_op<OperatorEvaluatorDiv<Vector3, Vector3, Vector3>>(VariantOperator::DIVIDE, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorDiv<Vector3, Vector3, double>>(VariantOperator::DIVIDE, VariantType::VECTOR3, VariantType::FLOAT);
	register_op<OperatorEvaluatorDiv<Vector3, Vector3, int64_t>>(VariantOperator::DIVIDE, VariantType::VECTOR3, VariantType::INT);

	register_op<OperatorEvaluatorDivNZ<Vector3i, Vector3i, Vector3i>>(VariantOperator::DIVIDE, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorDivNZ<Vector3, Vector3i, double>>(VariantOperator::DIVIDE, VariantType::VECTOR3I, VariantType::FLOAT);
	register_op<OperatorEvaluatorDivNZ<Vector3i, Vector3i, int64_t>>(VariantOperator::DIVIDE, VariantType::VECTOR3I, VariantType::INT);

	register_op<OperatorEvaluatorDiv<Vector4, Vector4, Vector4>>(VariantOperator::DIVIDE, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorDiv<Vector4, Vector4, double>>(VariantOperator::DIVIDE, VariantType::VECTOR4, VariantType::FLOAT);
	register_op<OperatorEvaluatorDiv<Vector4, Vector4, int64_t>>(VariantOperator::DIVIDE, VariantType::VECTOR4, VariantType::INT);

	register_op<OperatorEvaluatorDivNZ<Vector4i, Vector4i, Vector4i>>(VariantOperator::DIVIDE, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorDivNZ<Vector4, Vector4i, double>>(VariantOperator::DIVIDE, VariantType::VECTOR4I, VariantType::FLOAT);
	register_op<OperatorEvaluatorDivNZ<Vector4i, Vector4i, int64_t>>(VariantOperator::DIVIDE, VariantType::VECTOR4I, VariantType::INT);

	register_op<OperatorEvaluatorDiv<Quaternion, Quaternion, double>>(VariantOperator::DIVIDE, VariantType::QUATERNION, VariantType::FLOAT);
	register_op<OperatorEvaluatorDiv<Quaternion, Quaternion, int64_t>>(VariantOperator::DIVIDE, VariantType::QUATERNION, VariantType::INT);

	register_op<OperatorEvaluatorDiv<Color, Color, Color>>(VariantOperator::DIVIDE, VariantType::COLOR, VariantType::COLOR);
	register_op<OperatorEvaluatorDiv<Color, Color, double>>(VariantOperator::DIVIDE, VariantType::COLOR, VariantType::FLOAT);
	register_op<OperatorEvaluatorDiv<Color, Color, int64_t>>(VariantOperator::DIVIDE, VariantType::COLOR, VariantType::INT);

	register_op<OperatorEvaluatorModNZ<int64_t, int64_t, int64_t>>(VariantOperator::MODULE, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorModNZ<Vector2i, Vector2i, Vector2i>>(VariantOperator::MODULE, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorModNZ<Vector2i, Vector2i, int64_t>>(VariantOperator::MODULE, VariantType::VECTOR2I, VariantType::INT);

	register_op<OperatorEvaluatorModNZ<Vector3i, Vector3i, Vector3i>>(VariantOperator::MODULE, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorModNZ<Vector3i, Vector3i, int64_t>>(VariantOperator::MODULE, VariantType::VECTOR3I, VariantType::INT);

	register_op<OperatorEvaluatorModNZ<Vector4i, Vector4i, Vector4i>>(VariantOperator::MODULE, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorModNZ<Vector4i, Vector4i, int64_t>>(VariantOperator::MODULE, VariantType::VECTOR4I, VariantType::INT);

	register_string_modulo_op(void, VariantType::NIL);

	register_string_modulo_op(bool, VariantType::BOOL);
	register_string_modulo_op(int64_t, VariantType::INT);
	register_string_modulo_op(double, VariantType::FLOAT);
	register_string_modulo_op(String, VariantType::STRING);
	register_string_modulo_op(Vector2, VariantType::VECTOR2);
	register_string_modulo_op(Vector2i, VariantType::VECTOR2I);
	register_string_modulo_op(Rect2, VariantType::RECT2);
	register_string_modulo_op(Rect2i, VariantType::RECT2I);
	register_string_modulo_op(Vector3, VariantType::VECTOR3);
	register_string_modulo_op(Vector3i, VariantType::VECTOR3I);
	register_string_modulo_op(Vector4, VariantType::VECTOR4);
	register_string_modulo_op(Vector4i, VariantType::VECTOR4I);
	register_string_modulo_op(Transform2D, VariantType::TRANSFORM2D);
	register_string_modulo_op(Plane, VariantType::PLANE);
	register_string_modulo_op(Quaternion, VariantType::QUATERNION);
	register_string_modulo_op(AABB, VariantType::AABB);
	register_string_modulo_op(Basis, VariantType::BASIS);
	register_string_modulo_op(Transform3D, VariantType::TRANSFORM3D);
	register_string_modulo_op(Projection, VariantType::PROJECTION);

	register_string_modulo_op(Color, VariantType::COLOR);
	register_string_modulo_op(StringName, VariantType::STRING_NAME);
	register_string_modulo_op(NodePath, VariantType::NODE_PATH);
	register_string_modulo_op(Object, VariantType::OBJECT);
	register_string_modulo_op(Callable, VariantType::CALLABLE);
	register_string_modulo_op(Signal, VariantType::SIGNAL);
	register_string_modulo_op(Dictionary, VariantType::DICTIONARY);
	register_string_modulo_op(Array, VariantType::ARRAY);

	register_string_modulo_op(PackedByteArray, VariantType::PACKED_BYTE_ARRAY);
	register_string_modulo_op(PackedInt32Array, VariantType::PACKED_INT32_ARRAY);
	register_string_modulo_op(PackedInt64Array, VariantType::PACKED_INT64_ARRAY);
	register_string_modulo_op(PackedFloat32Array, VariantType::PACKED_FLOAT32_ARRAY);
	register_string_modulo_op(PackedFloat64Array, VariantType::PACKED_FLOAT64_ARRAY);
	register_string_modulo_op(PackedStringArray, VariantType::PACKED_STRING_ARRAY);
	register_string_modulo_op(PackedVector2Array, VariantType::PACKED_VECTOR2_ARRAY);
	register_string_modulo_op(PackedVector3Array, VariantType::PACKED_VECTOR3_ARRAY);
	register_string_modulo_op(PackedColorArray, VariantType::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorPow<int64_t, int64_t, int64_t>>(VariantOperator::POWER, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorPow<double, int64_t, double>>(VariantOperator::POWER, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorPow<double, double, double>>(VariantOperator::POWER, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorPow<double, double, int64_t>>(VariantOperator::POWER, VariantType::FLOAT, VariantType::INT);

	register_op<OperatorEvaluatorNeg<int64_t, int64_t>>(VariantOperator::NEGATE, VariantType::INT, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<double, double>>(VariantOperator::NEGATE, VariantType::FLOAT, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<Vector2, Vector2>>(VariantOperator::NEGATE, VariantType::VECTOR2, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<Vector2i, Vector2i>>(VariantOperator::NEGATE, VariantType::VECTOR2I, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<Vector3, Vector3>>(VariantOperator::NEGATE, VariantType::VECTOR3, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<Vector3i, Vector3i>>(VariantOperator::NEGATE, VariantType::VECTOR3I, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<Vector4, Vector4>>(VariantOperator::NEGATE, VariantType::VECTOR4, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<Vector4i, Vector4i>>(VariantOperator::NEGATE, VariantType::VECTOR4I, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<Quaternion, Quaternion>>(VariantOperator::NEGATE, VariantType::QUATERNION, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<Plane, Plane>>(VariantOperator::NEGATE, VariantType::PLANE, VariantType::NIL);
	register_op<OperatorEvaluatorNeg<Color, Color>>(VariantOperator::NEGATE, VariantType::COLOR, VariantType::NIL);

	register_op<OperatorEvaluatorPos<int64_t, int64_t>>(VariantOperator::POSITIVE, VariantType::INT, VariantType::NIL);
	register_op<OperatorEvaluatorPos<double, double>>(VariantOperator::POSITIVE, VariantType::FLOAT, VariantType::NIL);
	register_op<OperatorEvaluatorPos<Vector2, Vector2>>(VariantOperator::POSITIVE, VariantType::VECTOR2, VariantType::NIL);
	register_op<OperatorEvaluatorPos<Vector2i, Vector2i>>(VariantOperator::POSITIVE, VariantType::VECTOR2I, VariantType::NIL);
	register_op<OperatorEvaluatorPos<Vector3, Vector3>>(VariantOperator::POSITIVE, VariantType::VECTOR3, VariantType::NIL);
	register_op<OperatorEvaluatorPos<Vector3i, Vector3i>>(VariantOperator::POSITIVE, VariantType::VECTOR3I, VariantType::NIL);
	register_op<OperatorEvaluatorPos<Vector4, Vector4>>(VariantOperator::POSITIVE, VariantType::VECTOR4, VariantType::NIL);
	register_op<OperatorEvaluatorPos<Vector4i, Vector4i>>(VariantOperator::POSITIVE, VariantType::VECTOR4I, VariantType::NIL);
	register_op<OperatorEvaluatorPos<Quaternion, Quaternion>>(VariantOperator::POSITIVE, VariantType::QUATERNION, VariantType::NIL);
	register_op<OperatorEvaluatorPos<Plane, Plane>>(VariantOperator::POSITIVE, VariantType::PLANE, VariantType::NIL);
	register_op<OperatorEvaluatorPos<Color, Color>>(VariantOperator::POSITIVE, VariantType::COLOR, VariantType::NIL);

	register_op<OperatorEvaluatorShiftLeft<int64_t, int64_t, int64_t>>(VariantOperator::SHIFT_LEFT, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorShiftRight<int64_t, int64_t, int64_t>>(VariantOperator::SHIFT_RIGHT, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorBitOr<int64_t, int64_t, int64_t>>(VariantOperator::BIT_OR, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorBitAnd<int64_t, int64_t, int64_t>>(VariantOperator::BIT_AND, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorBitXor<int64_t, int64_t, int64_t>>(VariantOperator::BIT_XOR, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorBitNeg<int64_t, int64_t>>(VariantOperator::BIT_NEGATE, VariantType::INT, VariantType::NIL);

	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::EQUAL, VariantType::NIL, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::NIL);
	register_op<OperatorEvaluatorEqual<bool, bool>>(VariantOperator::EQUAL, VariantType::BOOL, VariantType::BOOL);
	register_op<OperatorEvaluatorEqual<int64_t, int64_t>>(VariantOperator::EQUAL, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorEqual<int64_t, double>>(VariantOperator::EQUAL, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorEqual<double, int64_t>>(VariantOperator::EQUAL, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorEqual<double, double>>(VariantOperator::EQUAL, VariantType::FLOAT, VariantType::FLOAT);
	register_string_op(OperatorEvaluatorEqual, VariantOperator::EQUAL);
	register_op<OperatorEvaluatorEqual<Vector2, Vector2>>(VariantOperator::EQUAL, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorEqual<Vector2i, Vector2i>>(VariantOperator::EQUAL, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorEqual<Rect2, Rect2>>(VariantOperator::EQUAL, VariantType::RECT2, VariantType::RECT2);
	register_op<OperatorEvaluatorEqual<Rect2i, Rect2i>>(VariantOperator::EQUAL, VariantType::RECT2I, VariantType::RECT2I);
	register_op<OperatorEvaluatorEqual<Vector3, Vector3>>(VariantOperator::EQUAL, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorEqual<Vector3i, Vector3i>>(VariantOperator::EQUAL, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorEqual<Transform2D, Transform2D>>(VariantOperator::EQUAL, VariantType::TRANSFORM2D, VariantType::TRANSFORM2D);
	register_op<OperatorEvaluatorEqual<Vector4, Vector4>>(VariantOperator::EQUAL, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorEqual<Vector4i, Vector4i>>(VariantOperator::EQUAL, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorEqual<Plane, Plane>>(VariantOperator::EQUAL, VariantType::PLANE, VariantType::PLANE);
	register_op<OperatorEvaluatorEqual<Quaternion, Quaternion>>(VariantOperator::EQUAL, VariantType::QUATERNION, VariantType::QUATERNION);
	register_op<OperatorEvaluatorEqual<AABB, AABB>>(VariantOperator::EQUAL, VariantType::AABB, VariantType::AABB);
	register_op<OperatorEvaluatorEqual<Basis, Basis>>(VariantOperator::EQUAL, VariantType::BASIS, VariantType::BASIS);
	register_op<OperatorEvaluatorEqual<Transform3D, Transform3D>>(VariantOperator::EQUAL, VariantType::TRANSFORM3D, VariantType::TRANSFORM3D);
	register_op<OperatorEvaluatorEqual<Projection, Projection>>(VariantOperator::EQUAL, VariantType::PROJECTION, VariantType::PROJECTION);
	register_op<OperatorEvaluatorEqual<Color, Color>>(VariantOperator::EQUAL, VariantType::COLOR, VariantType::COLOR);

	register_op<OperatorEvaluatorEqual<NodePath, NodePath>>(VariantOperator::EQUAL, VariantType::NODE_PATH, VariantType::NODE_PATH);
	register_op<OperatorEvaluatorEqual<RID, RID>>(VariantOperator::EQUAL, VariantType::RID, VariantType::RID);

	register_op<OperatorEvaluatorEqualObject>(VariantOperator::EQUAL, VariantType::OBJECT, VariantType::OBJECT);
	register_op<OperatorEvaluatorEqualObjectNil>(VariantOperator::EQUAL, VariantType::OBJECT, VariantType::NIL);
	register_op<OperatorEvaluatorEqualNilObject>(VariantOperator::EQUAL, VariantType::NIL, VariantType::OBJECT);

	register_op<OperatorEvaluatorEqual<Callable, Callable>>(VariantOperator::EQUAL, VariantType::CALLABLE, VariantType::CALLABLE);
	register_op<OperatorEvaluatorEqual<Signal, Signal>>(VariantOperator::EQUAL, VariantType::SIGNAL, VariantType::SIGNAL);
	register_op<OperatorEvaluatorEqual<Dictionary, Dictionary>>(VariantOperator::EQUAL, VariantType::DICTIONARY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorEqual<Array, Array>>(VariantOperator::EQUAL, VariantType::ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorEqual<PackedByteArray, PackedByteArray>>(VariantOperator::EQUAL, VariantType::PACKED_BYTE_ARRAY, VariantType::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedInt32Array, PackedInt32Array>>(VariantOperator::EQUAL, VariantType::PACKED_INT32_ARRAY, VariantType::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedInt64Array, PackedInt64Array>>(VariantOperator::EQUAL, VariantType::PACKED_INT64_ARRAY, VariantType::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedFloat32Array, PackedFloat32Array>>(VariantOperator::EQUAL, VariantType::PACKED_FLOAT32_ARRAY, VariantType::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedFloat64Array, PackedFloat64Array>>(VariantOperator::EQUAL, VariantType::PACKED_FLOAT64_ARRAY, VariantType::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedStringArray, PackedStringArray>>(VariantOperator::EQUAL, VariantType::PACKED_STRING_ARRAY, VariantType::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedVector2Array, PackedVector2Array>>(VariantOperator::EQUAL, VariantType::PACKED_VECTOR2_ARRAY, VariantType::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedVector3Array, PackedVector3Array>>(VariantOperator::EQUAL, VariantType::PACKED_VECTOR3_ARRAY, VariantType::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedColorArray, PackedColorArray>>(VariantOperator::EQUAL, VariantType::PACKED_COLOR_ARRAY, VariantType::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::BOOL, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::BOOL, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::INT, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::INT, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::FLOAT, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::FLOAT, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::STRING, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::STRING, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::VECTOR2, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::VECTOR2, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::VECTOR2I, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::VECTOR2I, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::RECT2, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::RECT2, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::RECT2I, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::RECT2I, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::VECTOR3, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::VECTOR3, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::VECTOR3I, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::VECTOR3I, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::VECTOR4, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::VECTOR4, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::VECTOR4I, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::VECTOR4I, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::TRANSFORM2D, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::TRANSFORM2D, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PLANE, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PLANE, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::QUATERNION, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::QUATERNION, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::AABB, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::AABB, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::BASIS, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::BASIS, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::TRANSFORM3D, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::TRANSFORM3D, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PROJECTION, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PROJECTION, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::COLOR, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::COLOR, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::STRING_NAME, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::STRING_NAME, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NODE_PATH, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::NODE_PATH, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::RID, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::RID, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::CALLABLE, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::CALLABLE, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::SIGNAL, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::SIGNAL, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::DICTIONARY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::DICTIONARY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PACKED_BYTE_ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PACKED_BYTE_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PACKED_INT32_ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PACKED_INT32_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PACKED_INT64_ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PACKED_INT64_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PACKED_FLOAT32_ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PACKED_FLOAT32_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PACKED_FLOAT64_ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PACKED_FLOAT64_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PACKED_STRING_ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PACKED_STRING_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PACKED_VECTOR2_ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PACKED_VECTOR2_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PACKED_VECTOR3_ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PACKED_VECTOR3_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::PACKED_COLOR_ARRAY, VariantType::NIL>>(VariantOperator::EQUAL, VariantType::PACKED_COLOR_ARRAY, VariantType::NIL);

	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::BOOL>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::BOOL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::INT>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::INT);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::FLOAT>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::FLOAT);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::STRING>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::STRING);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR2>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR2);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR2I>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::RECT2>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::RECT2);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::RECT2I>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::RECT2I);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR3>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR3);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR3I>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR4>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR4);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR4I>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::TRANSFORM2D>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::TRANSFORM2D);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PLANE>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PLANE);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::QUATERNION>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::QUATERNION);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::AABB>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::AABB);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::BASIS>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::BASIS);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::TRANSFORM3D>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::TRANSFORM3D);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PROJECTION>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PROJECTION);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::COLOR>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::COLOR);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::STRING_NAME>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::STRING_NAME);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::NODE_PATH>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::NODE_PATH);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::RID>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::RID);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::CALLABLE>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::CALLABLE);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::SIGNAL>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::SIGNAL);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::DICTIONARY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::ARRAY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_BYTE_ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_INT32_ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_INT64_ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_FLOAT32_ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_FLOAT64_ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_STRING_ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_VECTOR2_ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_VECTOR3_ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_COLOR_ARRAY>>(VariantOperator::EQUAL, VariantType::NIL, VariantType::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::NIL);
	register_op<OperatorEvaluatorNotEqual<bool, bool>>(VariantOperator::NOT_EQUAL, VariantType::BOOL, VariantType::BOOL);
	register_op<OperatorEvaluatorNotEqual<int64_t, int64_t>>(VariantOperator::NOT_EQUAL, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorNotEqual<int64_t, double>>(VariantOperator::NOT_EQUAL, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorNotEqual<double, int64_t>>(VariantOperator::NOT_EQUAL, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorNotEqual<double, double>>(VariantOperator::NOT_EQUAL, VariantType::FLOAT, VariantType::FLOAT);
	register_string_op(OperatorEvaluatorNotEqual, VariantOperator::NOT_EQUAL);
	register_op<OperatorEvaluatorNotEqual<Vector2, Vector2>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorNotEqual<Vector2i, Vector2i>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorNotEqual<Rect2, Rect2>>(VariantOperator::NOT_EQUAL, VariantType::RECT2, VariantType::RECT2);
	register_op<OperatorEvaluatorNotEqual<Rect2i, Rect2i>>(VariantOperator::NOT_EQUAL, VariantType::RECT2I, VariantType::RECT2I);
	register_op<OperatorEvaluatorNotEqual<Vector3, Vector3>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorNotEqual<Vector3i, Vector3i>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorNotEqual<Vector4, Vector4>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorNotEqual<Vector4i, Vector4i>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorNotEqual<Transform2D, Transform2D>>(VariantOperator::NOT_EQUAL, VariantType::TRANSFORM2D, VariantType::TRANSFORM2D);
	register_op<OperatorEvaluatorNotEqual<Plane, Plane>>(VariantOperator::NOT_EQUAL, VariantType::PLANE, VariantType::PLANE);
	register_op<OperatorEvaluatorNotEqual<Quaternion, Quaternion>>(VariantOperator::NOT_EQUAL, VariantType::QUATERNION, VariantType::QUATERNION);
	register_op<OperatorEvaluatorNotEqual<AABB, AABB>>(VariantOperator::NOT_EQUAL, VariantType::AABB, VariantType::AABB);
	register_op<OperatorEvaluatorNotEqual<Basis, Basis>>(VariantOperator::NOT_EQUAL, VariantType::BASIS, VariantType::BASIS);
	register_op<OperatorEvaluatorNotEqual<Transform3D, Transform3D>>(VariantOperator::NOT_EQUAL, VariantType::TRANSFORM3D, VariantType::TRANSFORM3D);
	register_op<OperatorEvaluatorNotEqual<Projection, Projection>>(VariantOperator::NOT_EQUAL, VariantType::PROJECTION, VariantType::PROJECTION);
	register_op<OperatorEvaluatorNotEqual<Color, Color>>(VariantOperator::NOT_EQUAL, VariantType::COLOR, VariantType::COLOR);

	register_op<OperatorEvaluatorNotEqual<NodePath, NodePath>>(VariantOperator::NOT_EQUAL, VariantType::NODE_PATH, VariantType::NODE_PATH);
	register_op<OperatorEvaluatorNotEqual<RID, RID>>(VariantOperator::NOT_EQUAL, VariantType::RID, VariantType::RID);

	register_op<OperatorEvaluatorNotEqualObject>(VariantOperator::NOT_EQUAL, VariantType::OBJECT, VariantType::OBJECT);
	register_op<OperatorEvaluatorNotEqualObjectNil>(VariantOperator::NOT_EQUAL, VariantType::OBJECT, VariantType::NIL);
	register_op<OperatorEvaluatorNotEqualNilObject>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::OBJECT);

	register_op<OperatorEvaluatorNotEqual<Callable, Callable>>(VariantOperator::NOT_EQUAL, VariantType::CALLABLE, VariantType::CALLABLE);
	register_op<OperatorEvaluatorNotEqual<Signal, Signal>>(VariantOperator::NOT_EQUAL, VariantType::SIGNAL, VariantType::SIGNAL);
	register_op<OperatorEvaluatorNotEqual<Dictionary, Dictionary>>(VariantOperator::NOT_EQUAL, VariantType::DICTIONARY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorNotEqual<Array, Array>>(VariantOperator::NOT_EQUAL, VariantType::ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedByteArray, PackedByteArray>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_BYTE_ARRAY, VariantType::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedInt32Array, PackedInt32Array>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_INT32_ARRAY, VariantType::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedInt64Array, PackedInt64Array>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_INT64_ARRAY, VariantType::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedFloat32Array, PackedFloat32Array>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_FLOAT32_ARRAY, VariantType::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedFloat64Array, PackedFloat64Array>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_FLOAT64_ARRAY, VariantType::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedStringArray, PackedStringArray>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_STRING_ARRAY, VariantType::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedVector2Array, PackedVector2Array>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_VECTOR2_ARRAY, VariantType::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedVector3Array, PackedVector3Array>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_VECTOR3_ARRAY, VariantType::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedColorArray, PackedColorArray>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_COLOR_ARRAY, VariantType::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::BOOL, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::BOOL, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::INT, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::INT, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::FLOAT, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::FLOAT, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::STRING, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::STRING, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::VECTOR2, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR2, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::VECTOR2I, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR2I, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::RECT2, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::RECT2, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::RECT2I, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::RECT2I, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::VECTOR3, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR3, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::VECTOR3I, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR3I, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::TRANSFORM2D, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::TRANSFORM2D, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::VECTOR4, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR4, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::VECTOR4I, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::VECTOR4I, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PLANE, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PLANE, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::QUATERNION, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::QUATERNION, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::AABB, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::AABB, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::BASIS, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::BASIS, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::TRANSFORM3D, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::TRANSFORM3D, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PROJECTION, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PROJECTION, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::COLOR, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::COLOR, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::STRING_NAME, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::STRING_NAME, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NODE_PATH, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::NODE_PATH, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::RID, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::RID, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::CALLABLE, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::CALLABLE, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::SIGNAL, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::SIGNAL, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::DICTIONARY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::DICTIONARY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PACKED_BYTE_ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_BYTE_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PACKED_INT32_ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_INT32_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PACKED_INT64_ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_INT64_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PACKED_FLOAT32_ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_FLOAT32_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PACKED_FLOAT64_ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_FLOAT64_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PACKED_STRING_ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_STRING_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PACKED_VECTOR2_ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_VECTOR2_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PACKED_VECTOR3_ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_VECTOR3_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::PACKED_COLOR_ARRAY, VariantType::NIL>>(VariantOperator::NOT_EQUAL, VariantType::PACKED_COLOR_ARRAY, VariantType::NIL);

	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::BOOL>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::BOOL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::INT>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::INT);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::FLOAT>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::FLOAT);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::STRING>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::STRING);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR2>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR2);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR2I>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::RECT2>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::RECT2);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::RECT2I>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::RECT2I);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR3>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR3);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR3I>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR4>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR4);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR4I>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::TRANSFORM2D>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::TRANSFORM2D);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PLANE>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PLANE);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::QUATERNION>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::QUATERNION);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::AABB>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::AABB);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::BASIS>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::BASIS);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::TRANSFORM3D>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::TRANSFORM3D);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PROJECTION>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PROJECTION);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::COLOR>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::COLOR);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::STRING_NAME>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::STRING_NAME);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::NODE_PATH>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::NODE_PATH);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::RID>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::RID);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::CALLABLE>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::CALLABLE);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::SIGNAL>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::SIGNAL);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::DICTIONARY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::ARRAY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_BYTE_ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_INT32_ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_INT64_ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_FLOAT32_ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_FLOAT64_ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_STRING_ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_VECTOR2_ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_VECTOR3_ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_COLOR_ARRAY>>(VariantOperator::NOT_EQUAL, VariantType::NIL, VariantType::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorLess<bool, bool>>(VariantOperator::LESS, VariantType::BOOL, VariantType::BOOL);
	register_op<OperatorEvaluatorLess<int64_t, int64_t>>(VariantOperator::LESS, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorLess<int64_t, double>>(VariantOperator::LESS, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorLess<double, int64_t>>(VariantOperator::LESS, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorLess<double, double>>(VariantOperator::LESS, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorLess<String, String>>(VariantOperator::LESS, VariantType::STRING, VariantType::STRING);
	register_op<OperatorEvaluatorLess<StringName, StringName>>(VariantOperator::LESS, VariantType::STRING_NAME, VariantType::STRING_NAME);
	register_op<OperatorEvaluatorLess<Vector2, Vector2>>(VariantOperator::LESS, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorLess<Vector2i, Vector2i>>(VariantOperator::LESS, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorLess<Vector3, Vector3>>(VariantOperator::LESS, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorLess<Vector3i, Vector3i>>(VariantOperator::LESS, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorLess<Vector4, Vector4>>(VariantOperator::LESS, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorLess<Vector4i, Vector4i>>(VariantOperator::LESS, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorLess<RID, RID>>(VariantOperator::LESS, VariantType::RID, VariantType::RID);
	register_op<OperatorEvaluatorLess<Array, Array>>(VariantOperator::LESS, VariantType::ARRAY, VariantType::ARRAY);

	register_op<OperatorEvaluatorLessEqual<int64_t, int64_t>>(VariantOperator::LESS_EQUAL, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorLessEqual<int64_t, double>>(VariantOperator::LESS_EQUAL, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorLessEqual<double, int64_t>>(VariantOperator::LESS_EQUAL, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorLessEqual<double, double>>(VariantOperator::LESS_EQUAL, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorLessEqual<String, String>>(VariantOperator::LESS_EQUAL, VariantType::STRING, VariantType::STRING);
	register_op<OperatorEvaluatorLessEqual<StringName, StringName>>(VariantOperator::LESS_EQUAL, VariantType::STRING_NAME, VariantType::STRING_NAME);
	register_op<OperatorEvaluatorLessEqual<Vector2, Vector2>>(VariantOperator::LESS_EQUAL, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorLessEqual<Vector2i, Vector2i>>(VariantOperator::LESS_EQUAL, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorLessEqual<Vector3, Vector3>>(VariantOperator::LESS_EQUAL, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorLessEqual<Vector3i, Vector3i>>(VariantOperator::LESS_EQUAL, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorLessEqual<Vector4, Vector4>>(VariantOperator::LESS_EQUAL, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorLessEqual<Vector4i, Vector4i>>(VariantOperator::LESS_EQUAL, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorLessEqual<RID, RID>>(VariantOperator::LESS_EQUAL, VariantType::RID, VariantType::RID);
	register_op<OperatorEvaluatorLessEqual<Array, Array>>(VariantOperator::LESS_EQUAL, VariantType::ARRAY, VariantType::ARRAY);

	register_op<OperatorEvaluatorGreater<bool, bool>>(VariantOperator::GREATER, VariantType::BOOL, VariantType::BOOL);
	register_op<OperatorEvaluatorGreater<int64_t, int64_t>>(VariantOperator::GREATER, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorGreater<int64_t, double>>(VariantOperator::GREATER, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorGreater<double, int64_t>>(VariantOperator::GREATER, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorGreater<double, double>>(VariantOperator::GREATER, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorGreater<String, String>>(VariantOperator::GREATER, VariantType::STRING, VariantType::STRING);
	register_op<OperatorEvaluatorGreater<StringName, StringName>>(VariantOperator::GREATER, VariantType::STRING_NAME, VariantType::STRING_NAME);
	register_op<OperatorEvaluatorGreater<Vector2, Vector2>>(VariantOperator::GREATER, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorGreater<Vector2i, Vector2i>>(VariantOperator::GREATER, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorGreater<Vector3, Vector3>>(VariantOperator::GREATER, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorGreater<Vector3i, Vector3i>>(VariantOperator::GREATER, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorGreater<Vector4, Vector4>>(VariantOperator::GREATER, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorGreater<Vector4i, Vector4i>>(VariantOperator::GREATER, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorGreater<RID, RID>>(VariantOperator::GREATER, VariantType::RID, VariantType::RID);
	register_op<OperatorEvaluatorGreater<Array, Array>>(VariantOperator::GREATER, VariantType::ARRAY, VariantType::ARRAY);

	register_op<OperatorEvaluatorGreaterEqual<int64_t, int64_t>>(VariantOperator::GREATER_EQUAL, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorGreaterEqual<int64_t, double>>(VariantOperator::GREATER_EQUAL, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorGreaterEqual<double, int64_t>>(VariantOperator::GREATER_EQUAL, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorGreaterEqual<double, double>>(VariantOperator::GREATER_EQUAL, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorGreaterEqual<String, String>>(VariantOperator::GREATER_EQUAL, VariantType::STRING, VariantType::STRING);
	register_op<OperatorEvaluatorGreaterEqual<StringName, StringName>>(VariantOperator::GREATER_EQUAL, VariantType::STRING_NAME, VariantType::STRING_NAME);
	register_op<OperatorEvaluatorGreaterEqual<Vector2, Vector2>>(VariantOperator::GREATER_EQUAL, VariantType::VECTOR2, VariantType::VECTOR2);
	register_op<OperatorEvaluatorGreaterEqual<Vector2i, Vector2i>>(VariantOperator::GREATER_EQUAL, VariantType::VECTOR2I, VariantType::VECTOR2I);
	register_op<OperatorEvaluatorGreaterEqual<Vector3, Vector3>>(VariantOperator::GREATER_EQUAL, VariantType::VECTOR3, VariantType::VECTOR3);
	register_op<OperatorEvaluatorGreaterEqual<Vector3i, Vector3i>>(VariantOperator::GREATER_EQUAL, VariantType::VECTOR3I, VariantType::VECTOR3I);
	register_op<OperatorEvaluatorGreaterEqual<Vector4, Vector4>>(VariantOperator::GREATER_EQUAL, VariantType::VECTOR4, VariantType::VECTOR4);
	register_op<OperatorEvaluatorGreaterEqual<Vector4i, Vector4i>>(VariantOperator::GREATER_EQUAL, VariantType::VECTOR4I, VariantType::VECTOR4I);
	register_op<OperatorEvaluatorGreaterEqual<RID, RID>>(VariantOperator::GREATER_EQUAL, VariantType::RID, VariantType::RID);
	register_op<OperatorEvaluatorGreaterEqual<Array, Array>>(VariantOperator::GREATER_EQUAL, VariantType::ARRAY, VariantType::ARRAY);

	register_op<OperatorEvaluatorAlwaysFalse<VariantOperator::OR, VariantType::NIL, VariantType::NIL>>(VariantOperator::OR, VariantType::NIL, VariantType::NIL);

	// OR
	register_op<OperatorEvaluatorNilXBoolOr>(VariantOperator::OR, VariantType::NIL, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXNilOr>(VariantOperator::OR, VariantType::BOOL, VariantType::NIL);
	register_op<OperatorEvaluatorNilXIntOr>(VariantOperator::OR, VariantType::NIL, VariantType::INT);
	register_op<OperatorEvaluatorIntXNilOr>(VariantOperator::OR, VariantType::INT, VariantType::NIL);
	register_op<OperatorEvaluatorNilXFloatOr>(VariantOperator::OR, VariantType::NIL, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXNilOr>(VariantOperator::OR, VariantType::FLOAT, VariantType::NIL);
	register_op<OperatorEvaluatorNilXObjectOr>(VariantOperator::OR, VariantType::NIL, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXNilOr>(VariantOperator::OR, VariantType::OBJECT, VariantType::NIL);

	register_op<OperatorEvaluatorBoolXBoolOr>(VariantOperator::OR, VariantType::BOOL, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXIntOr>(VariantOperator::OR, VariantType::BOOL, VariantType::INT);
	register_op<OperatorEvaluatorIntXBoolOr>(VariantOperator::OR, VariantType::INT, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXFloatOr>(VariantOperator::OR, VariantType::BOOL, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXBoolOr>(VariantOperator::OR, VariantType::FLOAT, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXObjectOr>(VariantOperator::OR, VariantType::BOOL, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXBoolOr>(VariantOperator::OR, VariantType::OBJECT, VariantType::BOOL);

	register_op<OperatorEvaluatorIntXIntOr>(VariantOperator::OR, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorIntXFloatOr>(VariantOperator::OR, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXIntOr>(VariantOperator::OR, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorIntXObjectOr>(VariantOperator::OR, VariantType::INT, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXIntOr>(VariantOperator::OR, VariantType::OBJECT, VariantType::INT);

	register_op<OperatorEvaluatorFloatXFloatOr>(VariantOperator::OR, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXObjectOr>(VariantOperator::OR, VariantType::FLOAT, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXFloatOr>(VariantOperator::OR, VariantType::OBJECT, VariantType::FLOAT);
	register_op<OperatorEvaluatorObjectXObjectOr>(VariantOperator::OR, VariantType::OBJECT, VariantType::OBJECT);

	// AND
	register_op<OperatorEvaluatorNilXBoolAnd>(VariantOperator::AND, VariantType::NIL, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXNilAnd>(VariantOperator::AND, VariantType::BOOL, VariantType::NIL);
	register_op<OperatorEvaluatorNilXIntAnd>(VariantOperator::AND, VariantType::NIL, VariantType::INT);
	register_op<OperatorEvaluatorIntXNilAnd>(VariantOperator::AND, VariantType::INT, VariantType::NIL);
	register_op<OperatorEvaluatorNilXFloatAnd>(VariantOperator::AND, VariantType::NIL, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXNilAnd>(VariantOperator::AND, VariantType::FLOAT, VariantType::NIL);
	register_op<OperatorEvaluatorNilXObjectAnd>(VariantOperator::AND, VariantType::NIL, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXNilAnd>(VariantOperator::AND, VariantType::OBJECT, VariantType::NIL);

	register_op<OperatorEvaluatorBoolXBoolAnd>(VariantOperator::AND, VariantType::BOOL, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXIntAnd>(VariantOperator::AND, VariantType::BOOL, VariantType::INT);
	register_op<OperatorEvaluatorIntXBoolAnd>(VariantOperator::AND, VariantType::INT, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXFloatAnd>(VariantOperator::AND, VariantType::BOOL, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXBoolAnd>(VariantOperator::AND, VariantType::FLOAT, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXObjectAnd>(VariantOperator::AND, VariantType::BOOL, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXBoolAnd>(VariantOperator::AND, VariantType::OBJECT, VariantType::BOOL);

	register_op<OperatorEvaluatorIntXIntAnd>(VariantOperator::AND, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorIntXFloatAnd>(VariantOperator::AND, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXIntAnd>(VariantOperator::AND, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorIntXObjectAnd>(VariantOperator::AND, VariantType::INT, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXIntAnd>(VariantOperator::AND, VariantType::OBJECT, VariantType::INT);

	register_op<OperatorEvaluatorFloatXFloatAnd>(VariantOperator::AND, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXObjectAnd>(VariantOperator::AND, VariantType::FLOAT, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXFloatAnd>(VariantOperator::AND, VariantType::OBJECT, VariantType::FLOAT);
	register_op<OperatorEvaluatorObjectXObjectAnd>(VariantOperator::AND, VariantType::OBJECT, VariantType::OBJECT);

	// XOR
	register_op<OperatorEvaluatorNilXBoolXor>(VariantOperator::XOR, VariantType::NIL, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXNilXor>(VariantOperator::XOR, VariantType::BOOL, VariantType::NIL);
	register_op<OperatorEvaluatorNilXIntXor>(VariantOperator::XOR, VariantType::NIL, VariantType::INT);
	register_op<OperatorEvaluatorIntXNilXor>(VariantOperator::XOR, VariantType::INT, VariantType::NIL);
	register_op<OperatorEvaluatorNilXFloatXor>(VariantOperator::XOR, VariantType::NIL, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXNilXor>(VariantOperator::XOR, VariantType::FLOAT, VariantType::NIL);
	register_op<OperatorEvaluatorNilXObjectXor>(VariantOperator::XOR, VariantType::NIL, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXNilXor>(VariantOperator::XOR, VariantType::OBJECT, VariantType::NIL);

	register_op<OperatorEvaluatorBoolXBoolXor>(VariantOperator::XOR, VariantType::BOOL, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXIntXor>(VariantOperator::XOR, VariantType::BOOL, VariantType::INT);
	register_op<OperatorEvaluatorIntXBoolXor>(VariantOperator::XOR, VariantType::INT, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXFloatXor>(VariantOperator::XOR, VariantType::BOOL, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXBoolXor>(VariantOperator::XOR, VariantType::FLOAT, VariantType::BOOL);
	register_op<OperatorEvaluatorBoolXObjectXor>(VariantOperator::XOR, VariantType::BOOL, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXBoolXor>(VariantOperator::XOR, VariantType::OBJECT, VariantType::BOOL);

	register_op<OperatorEvaluatorIntXIntXor>(VariantOperator::XOR, VariantType::INT, VariantType::INT);
	register_op<OperatorEvaluatorIntXFloatXor>(VariantOperator::XOR, VariantType::INT, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXIntXor>(VariantOperator::XOR, VariantType::FLOAT, VariantType::INT);
	register_op<OperatorEvaluatorIntXObjectXor>(VariantOperator::XOR, VariantType::INT, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXIntXor>(VariantOperator::XOR, VariantType::OBJECT, VariantType::INT);

	register_op<OperatorEvaluatorFloatXFloatXor>(VariantOperator::XOR, VariantType::FLOAT, VariantType::FLOAT);
	register_op<OperatorEvaluatorFloatXObjectXor>(VariantOperator::XOR, VariantType::FLOAT, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectXFloatXor>(VariantOperator::XOR, VariantType::OBJECT, VariantType::FLOAT);
	register_op<OperatorEvaluatorObjectXObjectXor>(VariantOperator::XOR, VariantType::OBJECT, VariantType::OBJECT);

	register_op<OperatorEvaluatorAlwaysTrue<VariantOperator::NOT, VariantType::NIL, VariantType::NIL>>(VariantOperator::NOT, VariantType::NIL, VariantType::NIL);
	register_op<OperatorEvaluatorNotBool>(VariantOperator::NOT, VariantType::BOOL, VariantType::NIL);
	register_op<OperatorEvaluatorNotInt>(VariantOperator::NOT, VariantType::INT, VariantType::NIL);
	register_op<OperatorEvaluatorNotFloat>(VariantOperator::NOT, VariantType::FLOAT, VariantType::NIL);
	register_op<OperatorEvaluatorNotObject>(VariantOperator::NOT, VariantType::OBJECT, VariantType::NIL);
	register_op<OperatorEvaluatorNot<String>>(VariantOperator::NOT, VariantType::STRING, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Vector2>>(VariantOperator::NOT, VariantType::VECTOR2, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Vector2i>>(VariantOperator::NOT, VariantType::VECTOR2I, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Rect2>>(VariantOperator::NOT, VariantType::RECT2, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Rect2i>>(VariantOperator::NOT, VariantType::RECT2I, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Vector3>>(VariantOperator::NOT, VariantType::VECTOR3, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Vector3i>>(VariantOperator::NOT, VariantType::VECTOR3I, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Transform2D>>(VariantOperator::NOT, VariantType::TRANSFORM2D, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Vector4>>(VariantOperator::NOT, VariantType::VECTOR4, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Vector4i>>(VariantOperator::NOT, VariantType::VECTOR4I, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Plane>>(VariantOperator::NOT, VariantType::PLANE, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Quaternion>>(VariantOperator::NOT, VariantType::QUATERNION, VariantType::NIL);
	register_op<OperatorEvaluatorNot<AABB>>(VariantOperator::NOT, VariantType::AABB, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Basis>>(VariantOperator::NOT, VariantType::BASIS, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Transform3D>>(VariantOperator::NOT, VariantType::TRANSFORM3D, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Projection>>(VariantOperator::NOT, VariantType::PROJECTION, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Color>>(VariantOperator::NOT, VariantType::COLOR, VariantType::NIL);
	register_op<OperatorEvaluatorNot<StringName>>(VariantOperator::NOT, VariantType::STRING_NAME, VariantType::NIL);
	register_op<OperatorEvaluatorNot<NodePath>>(VariantOperator::NOT, VariantType::NODE_PATH, VariantType::NIL);
	register_op<OperatorEvaluatorNot<RID>>(VariantOperator::NOT, VariantType::RID, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Callable>>(VariantOperator::NOT, VariantType::CALLABLE, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Signal>>(VariantOperator::NOT, VariantType::SIGNAL, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Dictionary>>(VariantOperator::NOT, VariantType::DICTIONARY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<Array>>(VariantOperator::NOT, VariantType::ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<PackedByteArray>>(VariantOperator::NOT, VariantType::PACKED_BYTE_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<PackedInt32Array>>(VariantOperator::NOT, VariantType::PACKED_INT32_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<PackedInt64Array>>(VariantOperator::NOT, VariantType::PACKED_INT64_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<PackedFloat32Array>>(VariantOperator::NOT, VariantType::PACKED_FLOAT32_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<PackedFloat64Array>>(VariantOperator::NOT, VariantType::PACKED_FLOAT64_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<PackedStringArray>>(VariantOperator::NOT, VariantType::PACKED_STRING_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<PackedVector2Array>>(VariantOperator::NOT, VariantType::PACKED_VECTOR2_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<PackedVector3Array>>(VariantOperator::NOT, VariantType::PACKED_VECTOR3_ARRAY, VariantType::NIL);
	register_op<OperatorEvaluatorNot<PackedColorArray>>(VariantOperator::NOT, VariantType::PACKED_COLOR_ARRAY, VariantType::NIL);

	register_string_op(OperatorEvaluatorInStringFind, VariantOperator::OP_IN);

	register_op<OperatorEvaluatorInDictionaryHasNil>(VariantOperator::OP_IN, VariantType::NIL, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<bool>>(VariantOperator::OP_IN, VariantType::BOOL, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<int64_t>>(VariantOperator::OP_IN, VariantType::INT, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<double>>(VariantOperator::OP_IN, VariantType::FLOAT, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<String>>(VariantOperator::OP_IN, VariantType::STRING, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector2>>(VariantOperator::OP_IN, VariantType::VECTOR2, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector2i>>(VariantOperator::OP_IN, VariantType::VECTOR2I, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Rect2>>(VariantOperator::OP_IN, VariantType::RECT2, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Rect2i>>(VariantOperator::OP_IN, VariantType::RECT2I, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector3>>(VariantOperator::OP_IN, VariantType::VECTOR3, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector3i>>(VariantOperator::OP_IN, VariantType::VECTOR3I, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector4>>(VariantOperator::OP_IN, VariantType::VECTOR4, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector4i>>(VariantOperator::OP_IN, VariantType::VECTOR4I, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Transform2D>>(VariantOperator::OP_IN, VariantType::TRANSFORM2D, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Plane>>(VariantOperator::OP_IN, VariantType::PLANE, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Quaternion>>(VariantOperator::OP_IN, VariantType::QUATERNION, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<AABB>>(VariantOperator::OP_IN, VariantType::AABB, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Basis>>(VariantOperator::OP_IN, VariantType::BASIS, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Transform3D>>(VariantOperator::OP_IN, VariantType::TRANSFORM3D, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Projection>>(VariantOperator::OP_IN, VariantType::PROJECTION, VariantType::DICTIONARY);

	register_op<OperatorEvaluatorInDictionaryHas<Color>>(VariantOperator::OP_IN, VariantType::COLOR, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<StringName>>(VariantOperator::OP_IN, VariantType::STRING_NAME, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<NodePath>>(VariantOperator::OP_IN, VariantType::NODE_PATH, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHasObject>(VariantOperator::OP_IN, VariantType::OBJECT, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Callable>>(VariantOperator::OP_IN, VariantType::CALLABLE, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Signal>>(VariantOperator::OP_IN, VariantType::SIGNAL, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Dictionary>>(VariantOperator::OP_IN, VariantType::DICTIONARY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Array>>(VariantOperator::OP_IN, VariantType::ARRAY, VariantType::DICTIONARY);

	register_op<OperatorEvaluatorInDictionaryHas<PackedByteArray>>(VariantOperator::OP_IN, VariantType::PACKED_BYTE_ARRAY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedInt32Array>>(VariantOperator::OP_IN, VariantType::PACKED_INT32_ARRAY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedInt64Array>>(VariantOperator::OP_IN, VariantType::PACKED_INT64_ARRAY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedFloat32Array>>(VariantOperator::OP_IN, VariantType::PACKED_FLOAT32_ARRAY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedFloat64Array>>(VariantOperator::OP_IN, VariantType::PACKED_FLOAT64_ARRAY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedStringArray>>(VariantOperator::OP_IN, VariantType::PACKED_STRING_ARRAY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedVector2Array>>(VariantOperator::OP_IN, VariantType::PACKED_VECTOR2_ARRAY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedVector3Array>>(VariantOperator::OP_IN, VariantType::PACKED_VECTOR3_ARRAY, VariantType::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedColorArray>>(VariantOperator::OP_IN, VariantType::PACKED_COLOR_ARRAY, VariantType::DICTIONARY);

	register_op<OperatorEvaluatorInArrayFindNil>(VariantOperator::OP_IN, VariantType::NIL, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<bool, Array>>(VariantOperator::OP_IN, VariantType::BOOL, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<int64_t, Array>>(VariantOperator::OP_IN, VariantType::INT, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<double, Array>>(VariantOperator::OP_IN, VariantType::FLOAT, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<String, Array>>(VariantOperator::OP_IN, VariantType::STRING, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector2, Array>>(VariantOperator::OP_IN, VariantType::VECTOR2, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector2i, Array>>(VariantOperator::OP_IN, VariantType::VECTOR2I, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Rect2, Array>>(VariantOperator::OP_IN, VariantType::RECT2, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Rect2i, Array>>(VariantOperator::OP_IN, VariantType::RECT2I, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector3, Array>>(VariantOperator::OP_IN, VariantType::VECTOR3, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector3i, Array>>(VariantOperator::OP_IN, VariantType::VECTOR3I, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector4, Array>>(VariantOperator::OP_IN, VariantType::VECTOR4, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector4i, Array>>(VariantOperator::OP_IN, VariantType::VECTOR4I, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Transform2D, Array>>(VariantOperator::OP_IN, VariantType::TRANSFORM2D, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Plane, Array>>(VariantOperator::OP_IN, VariantType::PLANE, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Quaternion, Array>>(VariantOperator::OP_IN, VariantType::QUATERNION, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<AABB, Array>>(VariantOperator::OP_IN, VariantType::AABB, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Basis, Array>>(VariantOperator::OP_IN, VariantType::BASIS, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Transform3D, Array>>(VariantOperator::OP_IN, VariantType::TRANSFORM3D, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Projection, Array>>(VariantOperator::OP_IN, VariantType::PROJECTION, VariantType::ARRAY);

	register_op<OperatorEvaluatorInArrayFind<Color, Array>>(VariantOperator::OP_IN, VariantType::COLOR, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<StringName, Array>>(VariantOperator::OP_IN, VariantType::STRING_NAME, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<NodePath, Array>>(VariantOperator::OP_IN, VariantType::NODE_PATH, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFindObject>(VariantOperator::OP_IN, VariantType::OBJECT, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Callable, Array>>(VariantOperator::OP_IN, VariantType::CALLABLE, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Signal, Array>>(VariantOperator::OP_IN, VariantType::SIGNAL, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Dictionary, Array>>(VariantOperator::OP_IN, VariantType::DICTIONARY, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Array, Array>>(VariantOperator::OP_IN, VariantType::ARRAY, VariantType::ARRAY);

	register_op<OperatorEvaluatorInArrayFind<PackedByteArray, Array>>(VariantOperator::OP_IN, VariantType::PACKED_BYTE_ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedInt32Array, Array>>(VariantOperator::OP_IN, VariantType::PACKED_INT32_ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedInt64Array, Array>>(VariantOperator::OP_IN, VariantType::PACKED_INT64_ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedFloat32Array, Array>>(VariantOperator::OP_IN, VariantType::PACKED_FLOAT32_ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedFloat64Array, Array>>(VariantOperator::OP_IN, VariantType::PACKED_FLOAT64_ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedStringArray, Array>>(VariantOperator::OP_IN, VariantType::PACKED_STRING_ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedVector2Array, Array>>(VariantOperator::OP_IN, VariantType::PACKED_VECTOR2_ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedVector3Array, Array>>(VariantOperator::OP_IN, VariantType::PACKED_VECTOR3_ARRAY, VariantType::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedColorArray, Array>>(VariantOperator::OP_IN, VariantType::PACKED_COLOR_ARRAY, VariantType::ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedByteArray>>(VariantOperator::OP_IN, VariantType::INT, VariantType::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedByteArray>>(VariantOperator::OP_IN, VariantType::FLOAT, VariantType::PACKED_BYTE_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedInt32Array>>(VariantOperator::OP_IN, VariantType::INT, VariantType::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedInt32Array>>(VariantOperator::OP_IN, VariantType::FLOAT, VariantType::PACKED_INT32_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedInt64Array>>(VariantOperator::OP_IN, VariantType::INT, VariantType::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedInt64Array>>(VariantOperator::OP_IN, VariantType::FLOAT, VariantType::PACKED_INT64_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedFloat32Array>>(VariantOperator::OP_IN, VariantType::INT, VariantType::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedFloat32Array>>(VariantOperator::OP_IN, VariantType::FLOAT, VariantType::PACKED_FLOAT32_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedFloat64Array>>(VariantOperator::OP_IN, VariantType::INT, VariantType::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedFloat64Array>>(VariantOperator::OP_IN, VariantType::FLOAT, VariantType::PACKED_FLOAT64_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<String, PackedStringArray>>(VariantOperator::OP_IN, VariantType::STRING, VariantType::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<StringName, PackedStringArray>>(VariantOperator::OP_IN, VariantType::STRING_NAME, VariantType::PACKED_STRING_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<Vector2, PackedVector2Array>>(VariantOperator::OP_IN, VariantType::VECTOR2, VariantType::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector3, PackedVector3Array>>(VariantOperator::OP_IN, VariantType::VECTOR3, VariantType::PACKED_VECTOR3_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<Color, PackedColorArray>>(VariantOperator::OP_IN, VariantType::COLOR, VariantType::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorObjectHasPropertyString>(VariantOperator::OP_IN, VariantType::STRING, VariantType::OBJECT);
	register_op<OperatorEvaluatorObjectHasPropertyStringName>(VariantOperator::OP_IN, VariantType::STRING_NAME, VariantType::OBJECT);
}

#undef register_string_op
#undef register_string_modulo_op

void Variant::_unregister_variant_operators() {
}

void Variant::evaluate(const VariantOperator &p_op, const Variant &p_a,
		const Variant &p_b, Variant &r_ret, bool &r_valid) {
	ERR_FAIL_INDEX((int)p_op, (int)VariantOperator::MAX);
	VariantType type_a = p_a.get_type();
	VariantType type_b = p_b.get_type();
	ERR_FAIL_INDEX((int)type_a, (int)VariantType::MAX);
	ERR_FAIL_INDEX((int)type_b, (int)VariantType::MAX);

	VariantEvaluatorFunction ev = operator_evaluator_table[(int)p_op][(int)type_a][(int)type_b];
	if (unlikely(!ev)) {
		r_valid = false;
		r_ret = Variant();
		return;
	}

	ev(p_a, p_b, &r_ret, r_valid);
}

VariantType Variant::get_operator_return_type(VariantOperator p_operator, VariantType p_type_a, VariantType p_type_b) {
	ERR_FAIL_INDEX_V((int)p_operator, (int)VariantOperator::MAX, VariantType::NIL);
	ERR_FAIL_INDEX_V((int)p_type_a, (int)VariantType::MAX, VariantType::NIL);
	ERR_FAIL_INDEX_V((int)p_type_b, (int)VariantType::MAX, VariantType::NIL);

	return operator_return_type_table[(int)p_operator][(int)p_type_a][(int)p_type_b];
}

Variant::ValidatedOperatorEvaluator Variant::get_validated_operator_evaluator(VariantOperator p_operator, VariantType p_type_a, VariantType p_type_b) {
	ERR_FAIL_INDEX_V((int)p_operator, (int)VariantOperator::MAX, nullptr);
	ERR_FAIL_INDEX_V((int)p_type_a, (int)VariantType::MAX, nullptr);
	ERR_FAIL_INDEX_V((int)p_type_b, (int)VariantType::MAX, nullptr);
	return validated_operator_evaluator_table[(int)p_operator][(int)p_type_a][(int)p_type_b];
}

Variant::PTROperatorEvaluator Variant::get_ptr_operator_evaluator(VariantOperator p_operator, VariantType p_type_a, VariantType p_type_b) {
	ERR_FAIL_INDEX_V((int)p_operator, (int)VariantOperator::MAX, nullptr);
	ERR_FAIL_INDEX_V((int)p_type_a, (int)VariantType::MAX, nullptr);
	ERR_FAIL_INDEX_V((int)p_type_b, (int)VariantType::MAX, nullptr);
	return ptr_operator_evaluator_table[(int)p_operator][(int)p_type_a][(int)p_type_b];
}

static const char *_op_names[(int)VariantOperator::MAX] = {
	"==",
	"!=",
	"<",
	"<=",
	">",
	">=",
	"+",
	"-",
	"*",
	"/",
	"unary-",
	"unary+",
	"%",
	"**",
	"<<",
	">>",
	"&",
	"|",
	"^",
	"~",
	"and",
	"or",
	"xor",
	"not",
	"in"
};

String Variant::get_operator_name(VariantOperator p_op) {
	ERR_FAIL_INDEX_V((int)p_op, (int)VariantOperator::MAX, "");
	return _op_names[(int)p_op];
}

Variant::operator bool() const {
	return booleanize();
}

// We consider all uninitialized or empty types to be false based on the type's
// zeroiness.
bool Variant::booleanize() const {
	return !is_zero();
}

bool Variant::in(const Variant &p_index, bool *r_valid) const {
	bool valid;
	Variant ret;
	evaluate(VariantOperator::OP_IN, p_index, *this, ret, valid);
	if (r_valid) {
		*r_valid = valid;
	}
	if (valid) {
		ERR_FAIL_COND_V(ret.type != VariantType::BOOL, false);
		return *VariantGetInternalPtr<bool>::get_ptr(&ret);
	} else {
		return false;
	}
}
