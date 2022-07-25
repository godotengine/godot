/*************************************************************************/
/*  variant_op.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef VARIANT_OP_H
#define VARIANT_OP_H

#include "variant.h"

#include "core/core_string_names.h"
#include "core/debugger/engine_debugger.h"
#include "core/object/class_db.h"

template <class R, class A, class B>
class OperatorEvaluatorAdd {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a + b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) + *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) + PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorSub {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a - b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) - *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) - PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorMul {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a * b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) * *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) * PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorPow {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = R(Math::pow((double)a, (double)b));
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = R(Math::pow((double)*VariantGetInternalPtr<A>::get_ptr(left), (double)*VariantGetInternalPtr<B>::get_ptr(right)));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(R(Math::pow((double)PtrToArg<A>::convert(left), (double)PtrToArg<B>::convert(right))), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorXForm {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a.xform(b);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = VariantGetInternalPtr<A>::get_ptr(left)->xform(*VariantGetInternalPtr<B>::get_ptr(right));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left).xform(PtrToArg<B>::convert(right)), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorXFormInv {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = b.xform_inv(a);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = VariantGetInternalPtr<B>::get_ptr(right)->xform_inv(*VariantGetInternalPtr<A>::get_ptr(left));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<B>::convert(right).xform_inv(PtrToArg<A>::convert(left)), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorDiv {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a / b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) / *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) / PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorDivNZ {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		if (b == 0) {
			r_valid = false;
			*r_ret = "Division by zero error";
			return;
		}
		*r_ret = a / b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) / *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) / PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorDivNZ<Vector2i, Vector2i, Vector2i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector2i &a = *VariantGetInternalPtr<Vector2i>::get_ptr(&p_left);
		const Vector2i &b = *VariantGetInternalPtr<Vector2i>::get_ptr(&p_right);
		if (unlikely(b.x == 0 || b.y == 0)) {
			r_valid = false;
			*r_ret = "Division by zero error";
			return;
		}
		*r_ret = a / b;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Vector2i>::change(r_ret);
		*VariantGetInternalPtr<Vector2i>::get_ptr(r_ret) = *VariantGetInternalPtr<Vector2i>::get_ptr(left) / *VariantGetInternalPtr<Vector2i>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector2i>::encode(PtrToArg<Vector2i>::convert(left) / PtrToArg<Vector2i>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector2i>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorDivNZ<Vector3i, Vector3i, Vector3i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector3i &a = *VariantGetInternalPtr<Vector3i>::get_ptr(&p_left);
		const Vector3i &b = *VariantGetInternalPtr<Vector3i>::get_ptr(&p_right);
		if (unlikely(b.x == 0 || b.y == 0 || b.z == 0)) {
			r_valid = false;
			*r_ret = "Division by zero error";
			return;
		}
		*r_ret = a / b;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Vector3i>::change(r_ret);
		*VariantGetInternalPtr<Vector3i>::get_ptr(r_ret) = *VariantGetInternalPtr<Vector3i>::get_ptr(left) / *VariantGetInternalPtr<Vector3i>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector3i>::encode(PtrToArg<Vector3i>::convert(left) / PtrToArg<Vector3i>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector3i>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorMod {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a % b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) % *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) % PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorModNZ {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		if (b == 0) {
			r_valid = false;
			*r_ret = "Module by zero error";
			return;
		}
		*r_ret = a % b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) % *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) % PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorModNZ<Vector2i, Vector2i, Vector2i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector2i &a = *VariantGetInternalPtr<Vector2i>::get_ptr(&p_left);
		const Vector2i &b = *VariantGetInternalPtr<Vector2i>::get_ptr(&p_right);
		if (unlikely(b.x == 0 || b.y == 0)) {
			r_valid = false;
			*r_ret = "Module by zero error";
			return;
		}
		*r_ret = a % b;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Vector2i>::change(r_ret);
		*VariantGetInternalPtr<Vector2i>::get_ptr(r_ret) = *VariantGetInternalPtr<Vector2i>::get_ptr(left) % *VariantGetInternalPtr<Vector2i>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector2i>::encode(PtrToArg<Vector2i>::convert(left) / PtrToArg<Vector2i>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector2i>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorModNZ<Vector3i, Vector3i, Vector3i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector3i &a = *VariantGetInternalPtr<Vector3i>::get_ptr(&p_left);
		const Vector3i &b = *VariantGetInternalPtr<Vector3i>::get_ptr(&p_right);
		if (unlikely(b.x == 0 || b.y == 0 || b.z == 0)) {
			r_valid = false;
			*r_ret = "Module by zero error";
			return;
		}
		*r_ret = a % b;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Vector3i>::change(r_ret);
		*VariantGetInternalPtr<Vector3i>::get_ptr(r_ret) = *VariantGetInternalPtr<Vector3i>::get_ptr(left) % *VariantGetInternalPtr<Vector3i>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector3i>::encode(PtrToArg<Vector3i>::convert(left) % PtrToArg<Vector3i>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector3i>::VARIANT_TYPE; }
};

template <class R, class A>
class OperatorEvaluatorNeg {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		*r_ret = -a;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = -*VariantGetInternalPtr<A>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(-PtrToArg<A>::convert(left), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A>
class OperatorEvaluatorPos {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		*r_ret = a;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorShiftLeft {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);

#if defined(DEBUG_ENABLED)
		if (b < 0 || a < 0) {
			*r_ret = "Invalid operands for bit shifting. Only positive operands are supported.";
			r_valid = false;
			return;
		}
#endif
		*r_ret = a << b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) << *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) << PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorShiftRight {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);

#if defined(DEBUG_ENABLED)
		if (b < 0 || a < 0) {
			*r_ret = "Invalid operands for bit shifting. Only positive operands are supported.";
			r_valid = false;
			return;
		}
#endif
		*r_ret = a >> b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) >> *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) >> PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorBitOr {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a | b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) | *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) | PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorBitAnd {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a & b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) & *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) & PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A, class B>
class OperatorEvaluatorBitXor {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a ^ b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) ^ *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) ^ PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A>
class OperatorEvaluatorBitNeg {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		*r_ret = ~a;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = ~*VariantGetInternalPtr<A>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(~PtrToArg<A>::convert(left), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class A, class B>
class OperatorEvaluatorEqual {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a == b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) == *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) == PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorEqualObject {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Object *a = p_left.get_validated_object();
		const Object *b = p_right.get_validated_object();
		*r_ret = a == b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Object *a = left->get_validated_object();
		const Object *b = right->get_validated_object();
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = a == b;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) == PtrToArg<Object *>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorEqualObjectNil {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Object *a = p_left.get_validated_object();
		*r_ret = a == nullptr;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Object *a = left->get_validated_object();
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = a == nullptr;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) == nullptr, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorEqualNilObject {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Object *b = p_right.get_validated_object();
		*r_ret = nullptr == b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Object *b = right->get_validated_object();
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = nullptr == b;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(nullptr == PtrToArg<Object *>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <class A, class B>
class OperatorEvaluatorNotEqual {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a != b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) != *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) != PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorNotEqualObject {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		Object *a = p_left.get_validated_object();
		Object *b = p_right.get_validated_object();
		*r_ret = a != b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *a = left->get_validated_object();
		Object *b = right->get_validated_object();
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = a != b;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) != PtrToArg<Object *>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorNotEqualObjectNil {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		Object *a = p_left.get_validated_object();
		*r_ret = a != nullptr;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *a = left->get_validated_object();
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = a != nullptr;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) != nullptr, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorNotEqualNilObject {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		Object *b = p_right.get_validated_object();
		*r_ret = nullptr != b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *b = right->get_validated_object();
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = nullptr != b;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(nullptr != PtrToArg<Object *>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <class A, class B>
class OperatorEvaluatorLess {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a < b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) < *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) < PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <class A, class B>
class OperatorEvaluatorLessEqual {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a <= b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) <= *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) <= PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <class A, class B>
class OperatorEvaluatorGreater {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a > b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) > *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) > PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <class A, class B>
class OperatorEvaluatorGreaterEqual {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a >= b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) >= *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) >= PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

// Should this be kept?
#define XOR_OP(m_a, m_b) (((m_a) || (m_b)) && !((m_a) && (m_b)))
template <class A, class B>
class OperatorEvaluatorXor {
public:
	_FORCE_INLINE_ static bool xor_op(const A &a, const B &b) {
		return ((a) || (b)) && !((a) && (b));
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = xor_op(a, b);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = xor_op(*VariantGetInternalPtr<A>::get_ptr(left), *VariantGetInternalPtr<B>::get_ptr(right));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(xor_op(PtrToArg<A>::convert(left), PtrToArg<B>::convert(right)), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

//// CUSTOM ////

class OperatorEvaluatorAddArray {
public:
	_FORCE_INLINE_ static void _add_arrays(Array &sum, const Array &array_a, const Array &array_b) {
		int asize = array_a.size();
		int bsize = array_b.size();
		sum.resize(asize + bsize);
		for (int i = 0; i < asize; i++) {
			sum[i] = array_a[i];
		}
		for (int i = 0; i < bsize; i++) {
			sum[i + asize] = array_b[i];
		}
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Array &array_a = *VariantGetInternalPtr<Array>::get_ptr(&p_left);
		const Array &array_b = *VariantGetInternalPtr<Array>::get_ptr(&p_right);
		Array sum;
		_add_arrays(sum, array_a, array_b);
		*r_ret = sum;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*r_ret = Array();
		_add_arrays(*VariantGetInternalPtr<Array>::get_ptr(r_ret), *VariantGetInternalPtr<Array>::get_ptr(left), *VariantGetInternalPtr<Array>::get_ptr(right));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		Array ret;
		_add_arrays(ret, PtrToArg<Array>::convert(left), PtrToArg<Array>::convert(right));
		PtrToArg<Array>::encode(ret, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::ARRAY; }
};

template <class T>
class OperatorEvaluatorAppendArray {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector<T> &array_a = *VariantGetInternalPtr<Vector<T>>::get_ptr(&p_left);
		const Vector<T> &array_b = *VariantGetInternalPtr<Vector<T>>::get_ptr(&p_right);
		Vector<T> sum = array_a;
		sum.append_array(array_b);
		*r_ret = sum;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<Vector<T>>::get_ptr(r_ret) = *VariantGetInternalPtr<Vector<T>>::get_ptr(left);
		VariantGetInternalPtr<Vector<T>>::get_ptr(r_ret)->append_array(*VariantGetInternalPtr<Vector<T>>::get_ptr(right));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		Vector<T> sum = PtrToArg<Vector<T>>::convert(left);
		sum.append_array(PtrToArg<Vector<T>>::convert(right));
		PtrToArg<Vector<T>>::encode(sum, r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector<T>>::VARIANT_TYPE; }
};

class OperatorEvaluatorStringModNil {
public:
	_FORCE_INLINE_ static String do_mod(const String &s, bool *r_valid) {
		Array values;
		values.push_back(Variant());

		String a = s.sprintf(values, r_valid);
		if (r_valid) {
			*r_valid = !*r_valid;
		}
		return a;
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const String &a = *VariantGetInternalPtr<String>::get_ptr(&p_left);
		*r_ret = do_mod(a, &r_valid);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<String>::get_ptr(r_ret) = do_mod(*VariantGetInternalPtr<String>::get_ptr(left), nullptr);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<String>::encode(do_mod(PtrToArg<String>::convert(left), nullptr), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::STRING; }
};

class OperatorEvaluatorStringModArray {
public:
	_FORCE_INLINE_ static String do_mod(const String &s, const Array &p_values, bool *r_valid) {
		String a = s.sprintf(p_values, r_valid);
		if (r_valid) {
			*r_valid = !*r_valid;
		}
		return a;
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const String &a = *VariantGetInternalPtr<String>::get_ptr(&p_left);
		*r_ret = do_mod(a, *VariantGetInternalPtr<Array>::get_ptr(&p_right), &r_valid);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<String>::get_ptr(r_ret) = do_mod(*VariantGetInternalPtr<String>::get_ptr(left), *VariantGetInternalPtr<Array>::get_ptr(right), nullptr);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<String>::encode(do_mod(PtrToArg<String>::convert(left), PtrToArg<Array>::convert(right), nullptr), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::STRING; }
};

class OperatorEvaluatorStringModObject {
public:
	_FORCE_INLINE_ static String do_mod(const String &s, const Object *p_object, bool *r_valid) {
		Array values;
		values.push_back(p_object);
		String a = s.sprintf(values, r_valid);
		if (r_valid) {
			*r_valid = !*r_valid;
		}

		return a;
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const String &a = *VariantGetInternalPtr<String>::get_ptr(&p_left);
		*r_ret = do_mod(a, p_right.get_validated_object(), &r_valid);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<String>::get_ptr(r_ret) = do_mod(*VariantGetInternalPtr<String>::get_ptr(left), right->get_validated_object(), nullptr);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<String>::encode(do_mod(PtrToArg<String>::convert(left), PtrToArg<Object *>::convert(right), nullptr), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::STRING; }
};

template <class T>
class OperatorEvaluatorStringModT {
public:
	_FORCE_INLINE_ static String do_mod(const String &s, const T &p_value, bool *r_valid) {
		Array values;
		values.push_back(p_value);
		String a = s.sprintf(values, r_valid);
		if (r_valid) {
			*r_valid = !*r_valid;
		}
		return a;
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const String &a = *VariantGetInternalPtr<String>::get_ptr(&p_left);
		*r_ret = do_mod(a, *VariantGetInternalPtr<T>::get_ptr(&p_right), &r_valid);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<String>::get_ptr(r_ret) = do_mod(*VariantGetInternalPtr<String>::get_ptr(left), *VariantGetInternalPtr<T>::get_ptr(right), nullptr);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<String>::encode(do_mod(PtrToArg<String>::convert(left), PtrToArg<T>::convert(right), nullptr), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::STRING; }
};

template <Variant::Operator op, Variant::Type type_left, Variant::Type type_right>
class OperatorEvaluatorAlwaysTrue {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = true;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = true;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(true, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <Variant::Operator op, Variant::Type type_left, Variant::Type type_right>
class OperatorEvaluatorAlwaysFalse {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = false;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = false;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(false, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

///// LOGICAL OPS ///////

template <Variant::Operator Op>
_FORCE_INLINE_ bool _logical_op(const bool &p_left, const bool &p_right) {
	ERR_FAIL_V_MSG(false, vformat("Unsupported logical operation %s.", Variant::get_operator_name(Op)));
}
template <>
_FORCE_INLINE_ bool _logical_op<Variant::Operator::OP_AND>(const bool &p_left, const bool &p_right) {
	return p_left && p_right;
}
template <>
_FORCE_INLINE_ bool _logical_op<Variant::Operator::OP_OR>(const bool &p_left, const bool &p_right) {
	return p_left || p_right;
}

template <Variant::Type T>
_FORCE_INLINE_ bool _ptr_type_to_bool(const void *p_ptr) {
	ERR_FAIL_V_MSG(false, vformat("Type %s cannot be converted to bool.", Variant::get_type_name(T)));
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::NIL>(const void *p_ptr) {
	return false;
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::BOOL>(const void *p_ptr) {
	return PtrToArg<bool>::convert(p_ptr);
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::INT>(const void *p_ptr) {
	return PtrToArg<int64_t>::convert(p_ptr) != 0;
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::FLOAT>(const void *p_ptr) {
	return PtrToArg<double>::convert(p_ptr) != 0.0;
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::STRING>(const void *p_ptr) {
	return !PtrToArg<String>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::VECTOR2>(const void *p_ptr) {
	return PtrToArg<Vector2>::convert(p_ptr) != Vector2();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::VECTOR2I>(const void *p_ptr) {
	return PtrToArg<Vector2i>::convert(p_ptr) != Vector2i();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::RECT2>(const void *p_ptr) {
	return PtrToArg<Rect2>::convert(p_ptr) != Rect2();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::RECT2I>(const void *p_ptr) {
	return PtrToArg<Rect2i>::convert(p_ptr) != Rect2i();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::VECTOR3>(const void *p_ptr) {
	return PtrToArg<Vector3>::convert(p_ptr) != Vector3();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::VECTOR3I>(const void *p_ptr) {
	return PtrToArg<Vector3i>::convert(p_ptr) != Vector3i();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::TRANSFORM2D>(const void *p_ptr) {
	return PtrToArg<Transform2D>::convert(p_ptr) != Transform2D();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PLANE>(const void *p_ptr) {
	return PtrToArg<Plane>::convert(p_ptr) != Plane();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::QUATERNION>(const void *p_ptr) {
	return PtrToArg<Quaternion>::convert(p_ptr) != Quaternion();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::AABB>(const void *p_ptr) {
	return PtrToArg<AABB>::convert(p_ptr) != AABB();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::BASIS>(const void *p_ptr) {
	return PtrToArg<Basis>::convert(p_ptr) != Basis();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::TRANSFORM3D>(const void *p_ptr) {
	return PtrToArg<Transform3D>::convert(p_ptr) != Transform3D();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::COLOR>(const void *p_ptr) {
	return PtrToArg<Color>::convert(p_ptr) != Color();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::STRING_NAME>(const void *p_ptr) {
	return PtrToArg<StringName>::convert(p_ptr) != StringName();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::NODE_PATH>(const void *p_ptr) {
	return PtrToArg<NodePath>::convert(p_ptr) != NodePath();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::RID>(const void *p_ptr) {
	return PtrToArg<RID>::convert(p_ptr) != RID();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::OBJECT>(const void *p_ptr) {
	return PtrToArg<Object *>::convert(p_ptr) != nullptr;
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::CALLABLE>(const void *p_ptr) {
	return !PtrToArg<Callable>::convert(p_ptr).is_null();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::SIGNAL>(const void *p_ptr) {
	return !PtrToArg<Signal>::convert(p_ptr).is_null();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::DICTIONARY>(const void *p_ptr) {
	return !PtrToArg<Dictionary>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::ARRAY>(const void *p_ptr) {
	return !PtrToArg<Array>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PACKED_BYTE_ARRAY>(const void *p_ptr) {
	return !PtrToArg<PackedByteArray>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PACKED_INT32_ARRAY>(const void *p_ptr) {
	return !PtrToArg<PackedInt32Array>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PACKED_INT64_ARRAY>(const void *p_ptr) {
	return !PtrToArg<PackedInt64Array>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PACKED_FLOAT32_ARRAY>(const void *p_ptr) {
	return !PtrToArg<PackedFloat32Array>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PACKED_FLOAT64_ARRAY>(const void *p_ptr) {
	return !PtrToArg<PackedFloat64Array>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PACKED_STRING_ARRAY>(const void *p_ptr) {
	return !PtrToArg<PackedStringArray>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PACKED_VECTOR2_ARRAY>(const void *p_ptr) {
	return !PtrToArg<PackedVector2Array>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PACKED_VECTOR3_ARRAY>(const void *p_ptr) {
	return !PtrToArg<PackedVector3Array>::convert(p_ptr).is_empty();
}
template <>
_FORCE_INLINE_ bool _ptr_type_to_bool<Variant::PACKED_COLOR_ARRAY>(const void *p_ptr) {
	return !PtrToArg<PackedColorArray>::convert(p_ptr).is_empty();
}

template <Variant::Operator Op, Variant::Type T_left, Variant::Type T_right>
_FORCE_INLINE_ bool _ptr_logical_op(const void *p_left, const void *p_right) {
	return _logical_op<Op>(_ptr_type_to_bool<T_left>(p_left), _ptr_type_to_bool<T_right>(p_right));
}

template <Variant::Operator Op, Variant::Type T_left, Variant::Type T_right>
class OperatorEvaluatorLogicalOperation {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = _logical_op<Op>(p_left.booleanize(), p_right.booleanize());
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = _logical_op<Op>(left->booleanize(), right->booleanize());
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(_ptr_logical_op<Op, T_left, T_right>(left, right), r_ret);
	}
	static Variant::Type get_return_type() {
		return Variant::BOOL;
	}
};

template <Variant::Type T_left>
class OperatorEvaluatorNot {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = !p_left.booleanize();
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = !left->booleanize();
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(_ptr_type_to_bool<T_left>(left), r_ret);
	}
	static Variant::Type get_return_type() {
		return Variant::BOOL;
	}
};

////

template <class Left>
class OperatorEvaluatorInStringFind {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Left &str_a = *VariantGetInternalPtr<Left>::get_ptr(&p_left);
		const String &str_b = *VariantGetInternalPtr<String>::get_ptr(&p_right);

		*r_ret = str_b.find(str_a) != -1;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Left &str_a = *VariantGetInternalPtr<Left>::get_ptr(left);
		const String &str_b = *VariantGetInternalPtr<String>::get_ptr(right);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = str_b.find(str_a) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<String>::convert(right).find(PtrToArg<Left>::convert(left)) != -1, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <class Left>
class OperatorEvaluatorInStringNameFind {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Left &str_a = *VariantGetInternalPtr<Left>::get_ptr(&p_left);
		const String str_b = VariantGetInternalPtr<StringName>::get_ptr(&p_right)->operator String();

		*r_ret = str_b.find(str_a) != -1;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Left &str_a = *VariantGetInternalPtr<Left>::get_ptr(left);
		const String str_b = VariantGetInternalPtr<StringName>::get_ptr(right)->operator String();
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = str_b.find(str_a) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<StringName>::convert(right).operator String().find(PtrToArg<Left>::convert(left)) != -1, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <class A, class B>
class OperatorEvaluatorInArrayFind {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);

		*r_ret = b.find(a) != -1;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(right);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = b.find(a) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<B>::convert(right).find(PtrToArg<A>::convert(left)) != -1, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorInArrayFindNil {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Array &b = *VariantGetInternalPtr<Array>::get_ptr(&p_right);
		*r_ret = b.find(Variant()) != -1;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Array &b = *VariantGetInternalPtr<Array>::get_ptr(right);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = b.find(Variant()) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Array>::convert(right).find(Variant()) != -1, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorInArrayFindObject {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Array &b = *VariantGetInternalPtr<Array>::get_ptr(&p_right);
		*r_ret = b.find(p_left) != -1;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Array &b = *VariantGetInternalPtr<Array>::get_ptr(right);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = b.find(*left) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Array>::convert(right).find(PtrToArg<Object *>::convert(left)) != -1, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <class A>
class OperatorEvaluatorInDictionaryHas {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Dictionary &b = *VariantGetInternalPtr<Dictionary>::get_ptr(&p_right);
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);

		*r_ret = b.has(a);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Dictionary &b = *VariantGetInternalPtr<Dictionary>::get_ptr(right);
		const A &a = *VariantGetInternalPtr<A>::get_ptr(left);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = b.has(a);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Dictionary>::convert(right).has(PtrToArg<A>::convert(left)), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorInDictionaryHasNil {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Dictionary &b = *VariantGetInternalPtr<Dictionary>::get_ptr(&p_right);

		*r_ret = b.has(Variant());
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Dictionary &b = *VariantGetInternalPtr<Dictionary>::get_ptr(right);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = b.has(Variant());
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Dictionary>::convert(right).has(Variant()), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorInDictionaryHasObject {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Dictionary &b = *VariantGetInternalPtr<Dictionary>::get_ptr(&p_right);

		*r_ret = b.has(p_left);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Dictionary &b = *VariantGetInternalPtr<Dictionary>::get_ptr(right);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = b.has(*left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Dictionary>::convert(right).has(PtrToArg<Object *>::convert(left)), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorObjectHasPropertyString {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		Object *b = p_right.get_validated_object();
		if (!b) {
			*r_ret = "Invalid base object for 'in'";
			r_valid = false;
			return;
		}

		const String &a = *VariantGetInternalPtr<String>::get_ptr(&p_left);

		bool exist;
		b->get(a, &exist);
		*r_ret = exist;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *l = right->get_validated_object();
		ERR_FAIL_COND(l == nullptr);
		const String &a = *VariantGetInternalPtr<String>::get_ptr(left);

		bool valid;
		l->get(a, &valid);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = valid;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		bool valid;
		PtrToArg<Object *>::convert(right)->get(PtrToArg<String>::convert(left), &valid);
		PtrToArg<bool>::encode(valid, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorObjectHasPropertyStringName {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		Object *b = p_right.get_validated_object();
		if (!b) {
			*r_ret = "Invalid base object for 'in'";
			r_valid = false;
			return;
		}

		const StringName &a = *VariantGetInternalPtr<StringName>::get_ptr(&p_left);

		bool exist;
		b->get(a, &exist);
		*r_ret = exist;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *l = right->get_validated_object();
		ERR_FAIL_COND(l == nullptr);
		const StringName &a = *VariantGetInternalPtr<StringName>::get_ptr(left);

		bool valid;
		l->get(a, &valid);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = valid;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		bool valid;
		PtrToArg<Object *>::convert(right)->get(PtrToArg<StringName>::convert(left), &valid);
		PtrToArg<bool>::encode(valid, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

#endif // VARIANT_OP_H
