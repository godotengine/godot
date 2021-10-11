/*************************************************************************/
/*  variant_op.h                                                         */
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

#ifndef VARIANT_OP_H
#define VARIANT_OP_H

#include "variant.h"
#include "variant_op4.h"

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

template <class A, class B>
class OperatorEvaluatorAnd {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a && b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) && *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) && PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

template <class A, class B>
class OperatorEvaluatorOr {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(&p_right);
		*r_ret = a || b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) || *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) || PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

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

template <class A>
class OperatorEvaluatorNot {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		*r_ret = !a;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = !*VariantGetInternalPtr<A>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(!PtrToArg<A>::convert(left));
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

///// OR ///////

_FORCE_INLINE_ static bool _operate_or(bool p_left, bool p_right) {
	return p_left || p_right;
}

_FORCE_INLINE_ static bool _operate_and(bool p_left, bool p_right) {
	return p_left && p_right;
}

_FORCE_INLINE_ static bool _operate_xor(bool p_left, bool p_right) {
	return (p_left || p_right) && !(p_left && p_right);
}

_FORCE_INLINE_ static bool _operate_get_nil(const Variant *p_ptr) {
	return p_ptr->get_validated_object() != nullptr;
}

_FORCE_INLINE_ static bool _operate_get_bool(const Variant *p_ptr) {
	return *VariantGetInternalPtr<bool>::get_ptr(p_ptr);
}

_FORCE_INLINE_ static bool _operate_get_int(const Variant *p_ptr) {
	return *VariantGetInternalPtr<int64_t>::get_ptr(p_ptr) != 0;
}

_FORCE_INLINE_ static bool _operate_get_float(const Variant *p_ptr) {
	return *VariantGetInternalPtr<double>::get_ptr(p_ptr) != 0.0;
}

_FORCE_INLINE_ static bool _operate_get_string(const Variant *p_ptr) {
	return !VariantGetInternalPtr<String>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_vector2(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Vector2>::get_ptr(p_ptr) != Vector2();
}

_FORCE_INLINE_ static bool _operate_get_vector2i(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Vector2i>::get_ptr(p_ptr) != Vector2i();
}

_FORCE_INLINE_ static bool _operate_get_rect2(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Rect2>::get_ptr(p_ptr) != Rect2();
}

_FORCE_INLINE_ static bool _operate_get_rect2i(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Rect2i>::get_ptr(p_ptr) != Rect2i();
}

_FORCE_INLINE_ static bool _operate_get_vector3(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Vector3>::get_ptr(p_ptr) != Vector3();
}

_FORCE_INLINE_ static bool _operate_get_vector3i(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Vector3i>::get_ptr(p_ptr) != Vector3i();
}

_FORCE_INLINE_ static bool _operate_get_transform2d(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Transform2D>::get_ptr(p_ptr) != Transform2D();
}

_FORCE_INLINE_ static bool _operate_get_plane(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Plane>::get_ptr(p_ptr) != Plane();
}

_FORCE_INLINE_ static bool _operate_get_quaternion(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Quaternion>::get_ptr(p_ptr) != Quaternion();
}

_FORCE_INLINE_ static bool _operate_get_aabb(const Variant *p_ptr) {
	return *VariantGetInternalPtr<AABB>::get_ptr(p_ptr) != AABB();
}

_FORCE_INLINE_ static bool _operate_get_basis(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Basis>::get_ptr(p_ptr) != Basis();
}

_FORCE_INLINE_ static bool _operate_get_transform3d(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Transform3D>::get_ptr(p_ptr) != Transform3D();
}

_FORCE_INLINE_ static bool _operate_get_color(const Variant *p_ptr) {
	return *VariantGetInternalPtr<Color>::get_ptr(p_ptr) != Color();
}

_FORCE_INLINE_ static bool _operate_get_string_name(const Variant *p_ptr) {
	return *VariantGetInternalPtr<StringName>::get_ptr(p_ptr) != StringName();
}

_FORCE_INLINE_ static bool _operate_get_node_path(const Variant *p_ptr) {
	return *VariantGetInternalPtr<NodePath>::get_ptr(p_ptr) != NodePath();
}

_FORCE_INLINE_ static bool _operate_get_rid(const Variant *p_ptr) {
	return *VariantGetInternalPtr<RID>::get_ptr(p_ptr) != RID();
}

_FORCE_INLINE_ static bool _operate_get_object(const Variant *p_ptr) {
	return p_ptr->get_validated_object() != nullptr;
}

_FORCE_INLINE_ static bool _operate_get_callable(const Variant *p_ptr) {
	return !VariantGetInternalPtr<Callable>::get_ptr(p_ptr)->is_null();
}

_FORCE_INLINE_ static bool _operate_get_signal(const Variant *p_ptr) {
	return !VariantGetInternalPtr<Signal>::get_ptr(p_ptr)->is_null();
}

_FORCE_INLINE_ static bool _operate_get_dictionary(const Variant *p_ptr) {
	return !VariantGetInternalPtr<Dictionary>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<Array>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_packed_byte_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<PackedByteArray>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_packed_int32_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<PackedInt32Array>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_packed_int64_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<PackedInt64Array>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_packed_float32_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<PackedFloat32Array>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_packed_float64_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<PackedFloat64Array>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_packed_string_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<PackedStringArray>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_packed_vector2_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<PackedVector2Array>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_packed_vector3_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<PackedVector3Array>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_packed_color_array(const Variant *p_ptr) {
	return !VariantGetInternalPtr<PackedColorArray>::get_ptr(p_ptr)->is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_nil(const void *p_ptr) {
	return false;
}

_FORCE_INLINE_ static bool _operate_get_ptr_bool(const void *p_ptr) {
	return PtrToArg<bool>::convert(p_ptr);
}

_FORCE_INLINE_ static bool _operate_get_ptr_int(const void *p_ptr) {
	return PtrToArg<int64_t>::convert(p_ptr) != 0;
}

_FORCE_INLINE_ static bool _operate_get_ptr_float(const void *p_ptr) {
	return PtrToArg<double>::convert(p_ptr) != 0.0;
}

_FORCE_INLINE_ static bool _operate_get_ptr_string(const void *p_ptr) {
	return !PtrToArg<String>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_vector2(const void *p_ptr) {
	return PtrToArg<Vector2>::convert(p_ptr) != Vector2();
}

_FORCE_INLINE_ static bool _operate_get_ptr_vector2i(const void *p_ptr) {
	return PtrToArg<Vector2i>::convert(p_ptr) != Vector2i();
}

_FORCE_INLINE_ static bool _operate_get_ptr_rect2(const void *p_ptr) {
	return PtrToArg<Rect2>::convert(p_ptr) != Rect2();
}

_FORCE_INLINE_ static bool _operate_get_ptr_rect2i(const void *p_ptr) {
	return PtrToArg<Rect2i>::convert(p_ptr) != Rect2i();
}

_FORCE_INLINE_ static bool _operate_get_ptr_vector3(const void *p_ptr) {
	return PtrToArg<Vector3>::convert(p_ptr) != Vector3();
}

_FORCE_INLINE_ static bool _operate_get_ptr_vector3i(const void *p_ptr) {
	return PtrToArg<Vector3i>::convert(p_ptr) != Vector3i();
}

_FORCE_INLINE_ static bool _operate_get_ptr_transform2d(const void *p_ptr) {
	return PtrToArg<Transform2D>::convert(p_ptr) != Transform2D();
}

_FORCE_INLINE_ static bool _operate_get_ptr_plane(const void *p_ptr) {
	return PtrToArg<Plane>::convert(p_ptr) != Plane();
}

_FORCE_INLINE_ static bool _operate_get_ptr_quaternion(const void *p_ptr) {
	return PtrToArg<Quaternion>::convert(p_ptr) != Quaternion();
}

_FORCE_INLINE_ static bool _operate_get_ptr_aabb(const void *p_ptr) {
	return PtrToArg<AABB>::convert(p_ptr) != AABB();
}

_FORCE_INLINE_ static bool _operate_get_ptr_basis(const void *p_ptr) {
	return PtrToArg<Basis>::convert(p_ptr) != Basis();
}

_FORCE_INLINE_ static bool _operate_get_ptr_transform3d(const void *p_ptr) {
	return PtrToArg<Transform3D>::convert(p_ptr) != Transform3D();
}

_FORCE_INLINE_ static bool _operate_get_ptr_color(const void *p_ptr) {
	return PtrToArg<Color>::convert(p_ptr) != Color();
}

_FORCE_INLINE_ static bool _operate_get_ptr_string_name(const void *p_ptr) {
	return PtrToArg<StringName>::convert(p_ptr) != StringName();
}

_FORCE_INLINE_ static bool _operate_get_ptr_node_path(const void *p_ptr) {
	return PtrToArg<NodePath>::convert(p_ptr) != NodePath();
}

_FORCE_INLINE_ static bool _operate_get_ptr_rid(const void *p_ptr) {
	return PtrToArg<RID>::convert(p_ptr) != RID();
}

_FORCE_INLINE_ static bool _operate_get_ptr_object(const void *p_ptr) {
	return PtrToArg<Object *>::convert(p_ptr) != nullptr;
}

_FORCE_INLINE_ static bool _operate_get_ptr_callable(const void *p_ptr) {
	return !PtrToArg<Callable>::convert(p_ptr).is_null();
}

_FORCE_INLINE_ static bool _operate_get_ptr_signal(const void *p_ptr) {
	return !PtrToArg<Signal>::convert(p_ptr).is_null();
}

_FORCE_INLINE_ static bool _operate_get_ptr_dictionary(const void *p_ptr) {
	return !PtrToArg<Dictionary>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_array(const void *p_ptr) {
	return !PtrToArg<Array>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_packed_byte_array(const void *p_ptr) {
	return !PtrToArg<PackedByteArray>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_packed_int32_array(const void *p_ptr) {
	return !PtrToArg<PackedInt32Array>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_packed_int64_array(const void *p_ptr) {
	return !PtrToArg<PackedInt64Array>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_packed_float32_array(const void *p_ptr) {
	return !PtrToArg<PackedFloat32Array>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_packed_float64_array(const void *p_ptr) {
	return !PtrToArg<PackedFloat64Array>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_packed_string_array(const void *p_ptr) {
	return !PtrToArg<PackedStringArray>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_packed_vector2_array(const void *p_ptr) {
	return !PtrToArg<PackedVector2Array>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_packed_vector3_array(const void *p_ptr) {
	return !PtrToArg<PackedVector3Array>::convert(p_ptr).is_empty();
}

_FORCE_INLINE_ static bool _operate_get_ptr_packed_color_array(const void *p_ptr) {
	return !PtrToArg<PackedColorArray>::convert(p_ptr).is_empty();
}

#define OP_EVALUATOR(m_class_name, m_left, m_right, m_op)                                                                    \
	class m_class_name {                                                                                                     \
	public:                                                                                                                  \
		static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {                 \
			*r_ret = m_op(_operate_get_##m_left(&p_left), _operate_get_##m_right(&p_right));                                 \
			r_valid = true;                                                                                                  \
		}                                                                                                                    \
                                                                                                                             \
		static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {                   \
			*VariantGetInternalPtr<bool>::get_ptr(r_ret) = m_op(_operate_get_##m_left(left), _operate_get_##m_right(right)); \
		}                                                                                                                    \
                                                                                                                             \
		static void ptr_evaluate(const void *left, const void *right, void *r_ret) {                                         \
			PtrToArg<bool>::encode(m_op(_operate_get_ptr_##m_left(left), _operate_get_ptr_##m_right(right)), r_ret);         \
		}                                                                                                                    \
                                                                                                                             \
		static Variant::Type get_return_type() {                                                                             \
			return Variant::BOOL;                                                                                            \
		}                                                                                                                    \
	};

// OR

// nil
OP_EVALUATOR(OperatorEvaluatorNilXNilOr, nil, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXBoolOr, nil, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXIntOr, nil, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXFloatOr, nil, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXStringOr, nil, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXVector2Or, nil, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXVector2iOr, nil, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXRect2Or, nil, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXRect2iOr, nil, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXVector3Or, nil, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXVector3iOr, nil, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXTransform2DOr, nil, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPlaneOr, nil, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXQuaternionOr, nil, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXAABBOr, nil, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXBasisOr, nil, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXTransform3DOr, nil, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXColorOr, nil, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXStringNameOr, nil, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXNodePathOr, nil, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXRIDOr, nil, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXObjectOr, nil, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXCallableOr, nil, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXSignalOr, nil, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXDictionaryOr, nil, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXArrayOr, nil, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPackedByteArrayOr, nil, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPackedInt32ArrayOr, nil, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPackedInt64ArrayOr, nil, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPackedFloat32ArrayOr, nil, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPackedFloat64ArrayOr, nil, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPackedStringArrayOr, nil, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPackedVector2ArrayOr, nil, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPackedVector3ArrayOr, nil, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXPackedColorArrayOr, nil, packed_color_array, _operate_or)

// bool
OP_EVALUATOR(OperatorEvaluatorBoolXNilOr, bool, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXBoolOr, bool, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXIntOr, bool, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXFloatOr, bool, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXStringOr, bool, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXVector2Or, bool, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXVector2iOr, bool, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXRect2Or, bool, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXRect2iOr, bool, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXVector3Or, bool, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXVector3iOr, bool, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXTransform2DOr, bool, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPlaneOr, bool, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXQuaternionOr, bool, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXAABBOr, bool, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXBasisOr, bool, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXTransform3DOr, bool, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXColorOr, bool, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXStringNameOr, bool, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXNodePathOr, bool, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXRIDOr, bool, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXObjectOr, bool, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXCallableOr, bool, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXSignalOr, bool, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXDictionaryOr, bool, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXArrayOr, bool, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedByteArrayOr, bool, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedInt32ArrayOr, bool, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedInt64ArrayOr, bool, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedFloat32ArrayOr, bool, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedFloat64ArrayOr, bool, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedStringArrayOr, bool, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedVector2ArrayOr, bool, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedVector3ArrayOr, bool, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedColorArrayOr, bool, packed_color_array, _operate_or)

// int
OP_EVALUATOR(OperatorEvaluatorIntXNilOr, int, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXBoolOr, int, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXIntOr, int, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXFloatOr, int, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXStringOr, int, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXVector2Or, int, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXVector2iOr, int, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXRect2Or, int, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXRect2iOr, int, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXVector3Or, int, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXVector3iOr, int, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXTransform2DOr, int, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPlaneOr, int, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXQuaternionOr, int, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXAABBOr, int, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXBasisOr, int, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXTransform3DOr, int, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXColorOr, int, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXStringNameOr, int, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXNodePathOr, int, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXRIDOr, int, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXObjectOr, int, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXCallableOr, int, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXSignalOr, int, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXDictionaryOr, int, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXArrayOr, int, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPackedByteArrayOr, int, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPackedInt32ArrayOr, int, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPackedInt64ArrayOr, int, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPackedFloat32ArrayOr, int, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPackedFloat64ArrayOr, int, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPackedStringArrayOr, int, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPackedVector2ArrayOr, int, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPackedVector3ArrayOr, int, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXPackedColorArrayOr, int, packed_color_array, _operate_or)

// float
OP_EVALUATOR(OperatorEvaluatorFloatXNilOr, float, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXBoolOr, float, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXIntOr, float, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXFloatOr, float, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXStringOr, float, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXVector2Or, float, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXVector2iOr, float, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXRect2Or, float, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXRect2iOr, float, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXVector3Or, float, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXVector3iOr, float, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXTransform2DOr, float, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPlaneOr, float, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXQuaternionOr, float, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXAABBOr, float, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXBasisOr, float, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXTransform3DOr, float, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXColorOr, float, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXStringNameOr, float, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXNodePathOr, float, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXRIDOr, float, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXObjectOr, float, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXCallableOr, float, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXSignalOr, float, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXDictionaryOr, float, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXArrayOr, float, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedByteArrayOr, float, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedInt32ArrayOr, float, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedInt64ArrayOr, float, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedFloat32ArrayOr, float, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedFloat64ArrayOr, float, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedStringArrayOr, float, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedVector2ArrayOr, float, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedVector3ArrayOr, float, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedColorArrayOr, float, packed_color_array, _operate_or)

// string
OP_EVALUATOR(OperatorEvaluatorStringXNilOr, string, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXBoolOr, string, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXIntOr, string, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXFloatOr, string, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXStringOr, string, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXVector2Or, string, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXVector2iOr, string, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXRect2Or, string, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXRect2iOr, string, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXVector3Or, string, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXVector3iOr, string, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXTransform2DOr, string, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPlaneOr, string, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXQuaternionOr, string, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXAABBOr, string, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXBasisOr, string, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXTransform3DOr, string, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXColorOr, string, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXStringNameOr, string, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXNodePathOr, string, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXRIDOr, string, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXObjectOr, string, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXCallableOr, string, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXSignalOr, string, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXDictionaryOr, string, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXArrayOr, string, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPackedByteArrayOr, string, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPackedInt32ArrayOr, string, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPackedInt64ArrayOr, string, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPackedFloat32ArrayOr, string, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPackedFloat64ArrayOr, string, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPackedStringArrayOr, string, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPackedVector2ArrayOr, string, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPackedVector3ArrayOr, string, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringXPackedColorArrayOr, string, packed_color_array, _operate_or)

// vector2
OP_EVALUATOR(OperatorEvaluatorVector2XNilOr, vector2, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XBoolOr, vector2, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XIntOr, vector2, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XFloatOr, vector2, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XStringOr, vector2, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XVector2Or, vector2, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XVector2iOr, vector2, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XRect2Or, vector2, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XRect2iOr, vector2, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XVector3Or, vector2, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XVector3iOr, vector2, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XTransform2DOr, vector2, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPlaneOr, vector2, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XQuaternionOr, vector2, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XAABBOr, vector2, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XBasisOr, vector2, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XTransform3DOr, vector2, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XColorOr, vector2, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XStringNameOr, vector2, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XNodePathOr, vector2, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XRIDOr, vector2, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XObjectOr, vector2, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XCallableOr, vector2, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XSignalOr, vector2, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XDictionaryOr, vector2, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XArrayOr, vector2, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedByteArrayOr, vector2, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedInt32ArrayOr, vector2, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedInt64ArrayOr, vector2, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedFloat32ArrayOr, vector2, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedFloat64ArrayOr, vector2, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedStringArrayOr, vector2, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedVector2ArrayOr, vector2, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedVector3ArrayOr, vector2, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedColorArrayOr, vector2, packed_color_array, _operate_or)

// vector2i
OP_EVALUATOR(OperatorEvaluatorVector2iXNilOr, vector2i, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXBoolOr, vector2i, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXIntOr, vector2i, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXFloatOr, vector2i, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXStringOr, vector2i, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXVector2Or, vector2i, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXVector2iOr, vector2i, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXRect2Or, vector2i, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXRect2iOr, vector2i, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXVector3Or, vector2i, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXVector3iOr, vector2i, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXTransform2DOr, vector2i, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPlaneOr, vector2i, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXQuaternionOr, vector2i, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXAABBOr, vector2i, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXBasisOr, vector2i, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXTransform3DOr, vector2i, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXColorOr, vector2i, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXStringNameOr, vector2i, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXNodePathOr, vector2i, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXRIDOr, vector2i, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXObjectOr, vector2i, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXCallableOr, vector2i, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXSignalOr, vector2i, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXDictionaryOr, vector2i, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXArrayOr, vector2i, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedByteArrayOr, vector2i, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedInt32ArrayOr, vector2i, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedInt64ArrayOr, vector2i, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedFloat32ArrayOr, vector2i, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedFloat64ArrayOr, vector2i, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedStringArrayOr, vector2i, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedVector2ArrayOr, vector2i, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedVector3ArrayOr, vector2i, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedColorArrayOr, vector2i, packed_color_array, _operate_or)

// rect2
OP_EVALUATOR(OperatorEvaluatorRect2XNilOr, rect2, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XBoolOr, rect2, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XIntOr, rect2, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XFloatOr, rect2, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XStringOr, rect2, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XVector2Or, rect2, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XVector2iOr, rect2, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XRect2Or, rect2, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XRect2iOr, rect2, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XVector3Or, rect2, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XVector3iOr, rect2, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XTransform2DOr, rect2, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPlaneOr, rect2, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XQuaternionOr, rect2, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XAABBOr, rect2, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XBasisOr, rect2, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XTransform3DOr, rect2, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XColorOr, rect2, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XStringNameOr, rect2, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XNodePathOr, rect2, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XRIDOr, rect2, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XObjectOr, rect2, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XCallableOr, rect2, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XSignalOr, rect2, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XDictionaryOr, rect2, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XArrayOr, rect2, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedByteArrayOr, rect2, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedInt32ArrayOr, rect2, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedInt64ArrayOr, rect2, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedFloat32ArrayOr, rect2, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedFloat64ArrayOr, rect2, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedStringArrayOr, rect2, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedVector2ArrayOr, rect2, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedVector3ArrayOr, rect2, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedColorArrayOr, rect2, packed_color_array, _operate_or)

// rect2i
OP_EVALUATOR(OperatorEvaluatorRect2iXNilOr, rect2i, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXBoolOr, rect2i, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXIntOr, rect2i, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXFloatOr, rect2i, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXStringOr, rect2i, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXVector2Or, rect2i, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXVector2iOr, rect2i, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXRect2Or, rect2i, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXRect2iOr, rect2i, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXVector3Or, rect2i, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXVector3iOr, rect2i, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXTransform2DOr, rect2i, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPlaneOr, rect2i, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXQuaternionOr, rect2i, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXAABBOr, rect2i, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXBasisOr, rect2i, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXTransform3DOr, rect2i, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXColorOr, rect2i, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXStringNameOr, rect2i, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXNodePathOr, rect2i, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXRIDOr, rect2i, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXObjectOr, rect2i, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXCallableOr, rect2i, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXSignalOr, rect2i, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXDictionaryOr, rect2i, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXArrayOr, rect2i, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedByteArrayOr, rect2i, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedInt32ArrayOr, rect2i, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedInt64ArrayOr, rect2i, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedFloat32ArrayOr, rect2i, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedFloat64ArrayOr, rect2i, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedStringArrayOr, rect2i, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedVector2ArrayOr, rect2i, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedVector3ArrayOr, rect2i, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedColorArrayOr, rect2i, packed_color_array, _operate_or)

// vector3
OP_EVALUATOR(OperatorEvaluatorVector3XNilOr, vector3, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XBoolOr, vector3, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XIntOr, vector3, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XFloatOr, vector3, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XStringOr, vector3, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XVector2Or, vector3, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XVector2iOr, vector3, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XRect2Or, vector3, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XRect2iOr, vector3, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XVector3Or, vector3, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XVector3iOr, vector3, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XTransform2DOr, vector3, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPlaneOr, vector3, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XQuaternionOr, vector3, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XAABBOr, vector3, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XBasisOr, vector3, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XTransform3DOr, vector3, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XColorOr, vector3, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XStringNameOr, vector3, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XNodePathOr, vector3, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XRIDOr, vector3, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XObjectOr, vector3, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XCallableOr, vector3, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XSignalOr, vector3, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XDictionaryOr, vector3, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XArrayOr, vector3, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedByteArrayOr, vector3, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedInt32ArrayOr, vector3, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedInt64ArrayOr, vector3, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedFloat32ArrayOr, vector3, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedFloat64ArrayOr, vector3, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedStringArrayOr, vector3, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedVector2ArrayOr, vector3, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedVector3ArrayOr, vector3, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedColorArrayOr, vector3, packed_color_array, _operate_or)

// vector3i
OP_EVALUATOR(OperatorEvaluatorVector3iXNilOr, vector3i, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXBoolOr, vector3i, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXIntOr, vector3i, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXFloatOr, vector3i, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXStringOr, vector3i, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXVector2Or, vector3i, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXVector2iOr, vector3i, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXRect2Or, vector3i, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXRect2iOr, vector3i, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXVector3Or, vector3i, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXVector3iOr, vector3i, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXTransform2DOr, vector3i, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPlaneOr, vector3i, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXQuaternionOr, vector3i, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXAABBOr, vector3i, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXBasisOr, vector3i, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXTransform3DOr, vector3i, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXColorOr, vector3i, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXStringNameOr, vector3i, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXNodePathOr, vector3i, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXRIDOr, vector3i, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXObjectOr, vector3i, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXCallableOr, vector3i, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXSignalOr, vector3i, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXDictionaryOr, vector3i, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXArrayOr, vector3i, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedByteArrayOr, vector3i, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedInt32ArrayOr, vector3i, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedInt64ArrayOr, vector3i, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedFloat32ArrayOr, vector3i, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedFloat64ArrayOr, vector3i, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedStringArrayOr, vector3i, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedVector2ArrayOr, vector3i, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedVector3ArrayOr, vector3i, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedColorArrayOr, vector3i, packed_color_array, _operate_or)

// transform2d
OP_EVALUATOR(OperatorEvaluatorTransform2DXNilOr, transform2d, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXBoolOr, transform2d, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXIntOr, transform2d, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXFloatOr, transform2d, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXStringOr, transform2d, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXVector2Or, transform2d, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXVector2iOr, transform2d, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXRect2Or, transform2d, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXRect2iOr, transform2d, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXVector3Or, transform2d, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXVector3iOr, transform2d, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXTransform2DOr, transform2d, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPlaneOr, transform2d, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXQuaternionOr, transform2d, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXAABBOr, transform2d, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXBasisOr, transform2d, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXTransform3DOr, transform2d, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXColorOr, transform2d, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXStringNameOr, transform2d, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXNodePathOr, transform2d, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXRIDOr, transform2d, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXObjectOr, transform2d, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXCallableOr, transform2d, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXSignalOr, transform2d, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXDictionaryOr, transform2d, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXArrayOr, transform2d, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedByteArrayOr, transform2d, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedInt32ArrayOr, transform2d, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedInt64ArrayOr, transform2d, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedFloat32ArrayOr, transform2d, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedFloat64ArrayOr, transform2d, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedStringArrayOr, transform2d, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedVector2ArrayOr, transform2d, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedVector3ArrayOr, transform2d, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedColorArrayOr, transform2d, packed_color_array, _operate_or)

// plane
OP_EVALUATOR(OperatorEvaluatorPlaneXNilOr, plane, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXBoolOr, plane, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXIntOr, plane, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXFloatOr, plane, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXStringOr, plane, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXVector2Or, plane, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXVector2iOr, plane, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXRect2Or, plane, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXRect2iOr, plane, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXVector3Or, plane, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXVector3iOr, plane, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXTransform2DOr, plane, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPlaneOr, plane, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXQuaternionOr, plane, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXAABBOr, plane, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXBasisOr, plane, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXTransform3DOr, plane, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXColorOr, plane, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXStringNameOr, plane, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXNodePathOr, plane, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXRIDOr, plane, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXObjectOr, plane, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXCallableOr, plane, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXSignalOr, plane, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXDictionaryOr, plane, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXArrayOr, plane, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedByteArrayOr, plane, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedInt32ArrayOr, plane, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedInt64ArrayOr, plane, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedFloat32ArrayOr, plane, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedFloat64ArrayOr, plane, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedStringArrayOr, plane, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedVector2ArrayOr, plane, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedVector3ArrayOr, plane, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedColorArrayOr, plane, packed_color_array, _operate_or)

// quaternion
OP_EVALUATOR(OperatorEvaluatorQuaternionXNilOr, quaternion, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXBoolOr, quaternion, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXIntOr, quaternion, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXFloatOr, quaternion, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXStringOr, quaternion, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXVector2Or, quaternion, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXVector2iOr, quaternion, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXRect2Or, quaternion, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXRect2iOr, quaternion, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXVector3Or, quaternion, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXVector3iOr, quaternion, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXTransform2DOr, quaternion, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPlaneOr, quaternion, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXQuaternionOr, quaternion, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXAABBOr, quaternion, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXBasisOr, quaternion, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXTransform3DOr, quaternion, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXColorOr, quaternion, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXStringNameOr, quaternion, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXNodePathOr, quaternion, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXRIDOr, quaternion, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXObjectOr, quaternion, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXCallableOr, quaternion, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXSignalOr, quaternion, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXDictionaryOr, quaternion, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXArrayOr, quaternion, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedByteArrayOr, quaternion, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedInt32ArrayOr, quaternion, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedInt64ArrayOr, quaternion, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedFloat32ArrayOr, quaternion, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedFloat64ArrayOr, quaternion, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedStringArrayOr, quaternion, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedVector2ArrayOr, quaternion, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedVector3ArrayOr, quaternion, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedColorArrayOr, quaternion, packed_color_array, _operate_or)

// aabb
OP_EVALUATOR(OperatorEvaluatorAABBXNilOr, aabb, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXBoolOr, aabb, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXIntOr, aabb, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXFloatOr, aabb, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXStringOr, aabb, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXVector2Or, aabb, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXVector2iOr, aabb, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXRect2Or, aabb, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXRect2iOr, aabb, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXVector3Or, aabb, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXVector3iOr, aabb, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXTransform2DOr, aabb, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPlaneOr, aabb, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXQuaternionOr, aabb, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXAABBOr, aabb, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXBasisOr, aabb, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXTransform3DOr, aabb, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXColorOr, aabb, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXStringNameOr, aabb, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXNodePathOr, aabb, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXRIDOr, aabb, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXObjectOr, aabb, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXCallableOr, aabb, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXSignalOr, aabb, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXDictionaryOr, aabb, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXArrayOr, aabb, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedByteArrayOr, aabb, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedInt32ArrayOr, aabb, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedInt64ArrayOr, aabb, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedFloat32ArrayOr, aabb, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedFloat64ArrayOr, aabb, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedStringArrayOr, aabb, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedVector2ArrayOr, aabb, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedVector3ArrayOr, aabb, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedColorArrayOr, aabb, packed_color_array, _operate_or)

// basis
OP_EVALUATOR(OperatorEvaluatorBasisXNilOr, basis, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXBoolOr, basis, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXIntOr, basis, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXFloatOr, basis, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXStringOr, basis, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXVector2Or, basis, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXVector2iOr, basis, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXRect2Or, basis, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXRect2iOr, basis, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXVector3Or, basis, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXVector3iOr, basis, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXTransform2DOr, basis, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPlaneOr, basis, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXQuaternionOr, basis, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXAABBOr, basis, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXBasisOr, basis, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXTransform3DOr, basis, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXColorOr, basis, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXStringNameOr, basis, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXNodePathOr, basis, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXRIDOr, basis, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXObjectOr, basis, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXCallableOr, basis, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXSignalOr, basis, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXDictionaryOr, basis, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXArrayOr, basis, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedByteArrayOr, basis, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedInt32ArrayOr, basis, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedInt64ArrayOr, basis, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedFloat32ArrayOr, basis, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedFloat64ArrayOr, basis, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedStringArrayOr, basis, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedVector2ArrayOr, basis, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedVector3ArrayOr, basis, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedColorArrayOr, basis, packed_color_array, _operate_or)

// transform3d
OP_EVALUATOR(OperatorEvaluatorTransform3DXNilOr, transform3d, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXBoolOr, transform3d, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXIntOr, transform3d, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXFloatOr, transform3d, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXStringOr, transform3d, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXVector2Or, transform3d, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXVector2iOr, transform3d, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXRect2Or, transform3d, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXRect2iOr, transform3d, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXVector3Or, transform3d, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXVector3iOr, transform3d, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXTransform2DOr, transform3d, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPlaneOr, transform3d, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXQuaternionOr, transform3d, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXAABBOr, transform3d, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXBasisOr, transform3d, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXTransform3DOr, transform3d, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXColorOr, transform3d, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXStringNameOr, transform3d, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXNodePathOr, transform3d, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXRIDOr, transform3d, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXObjectOr, transform3d, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXCallableOr, transform3d, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXSignalOr, transform3d, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXDictionaryOr, transform3d, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXArrayOr, transform3d, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedByteArrayOr, transform3d, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedInt32ArrayOr, transform3d, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedInt64ArrayOr, transform3d, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedFloat32ArrayOr, transform3d, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedFloat64ArrayOr, transform3d, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedStringArrayOr, transform3d, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedVector2ArrayOr, transform3d, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedVector3ArrayOr, transform3d, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedColorArrayOr, transform3d, packed_color_array, _operate_or)

// color
OP_EVALUATOR(OperatorEvaluatorColorXNilOr, color, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXBoolOr, color, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXIntOr, color, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXFloatOr, color, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXStringOr, color, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXVector2Or, color, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXVector2iOr, color, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXRect2Or, color, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXRect2iOr, color, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXVector3Or, color, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXVector3iOr, color, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXTransform2DOr, color, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPlaneOr, color, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXQuaternionOr, color, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXAABBOr, color, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXBasisOr, color, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXTransform3DOr, color, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXColorOr, color, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXStringNameOr, color, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXNodePathOr, color, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXRIDOr, color, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXObjectOr, color, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXCallableOr, color, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXSignalOr, color, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXDictionaryOr, color, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXArrayOr, color, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPackedByteArrayOr, color, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPackedInt32ArrayOr, color, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPackedInt64ArrayOr, color, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPackedFloat32ArrayOr, color, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPackedFloat64ArrayOr, color, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPackedStringArrayOr, color, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPackedVector2ArrayOr, color, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPackedVector3ArrayOr, color, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorColorXPackedColorArrayOr, color, packed_color_array, _operate_or)

// string_name
OP_EVALUATOR(OperatorEvaluatorStringNameXNilOr, string_name, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXBoolOr, string_name, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXIntOr, string_name, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXFloatOr, string_name, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXStringOr, string_name, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXVector2Or, string_name, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXVector2iOr, string_name, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXRect2Or, string_name, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXRect2iOr, string_name, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXVector3Or, string_name, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXVector3iOr, string_name, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXTransform2DOr, string_name, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPlaneOr, string_name, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXQuaternionOr, string_name, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXAABBOr, string_name, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXBasisOr, string_name, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXTransform3DOr, string_name, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXColorOr, string_name, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXStringNameOr, string_name, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXNodePathOr, string_name, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXRIDOr, string_name, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXObjectOr, string_name, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXCallableOr, string_name, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXSignalOr, string_name, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXDictionaryOr, string_name, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXArrayOr, string_name, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedByteArrayOr, string_name, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedInt32ArrayOr, string_name, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedInt64ArrayOr, string_name, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedFloat32ArrayOr, string_name, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedFloat64ArrayOr, string_name, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedStringArrayOr, string_name, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedVector2ArrayOr, string_name, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedVector3ArrayOr, string_name, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedColorArrayOr, string_name, packed_color_array, _operate_or)

// node_path
OP_EVALUATOR(OperatorEvaluatorNodePathXNilOr, node_path, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXBoolOr, node_path, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXIntOr, node_path, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXFloatOr, node_path, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXStringOr, node_path, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXVector2Or, node_path, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXVector2iOr, node_path, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXRect2Or, node_path, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXRect2iOr, node_path, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXVector3Or, node_path, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXVector3iOr, node_path, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXTransform2DOr, node_path, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPlaneOr, node_path, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXQuaternionOr, node_path, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXAABBOr, node_path, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXBasisOr, node_path, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXTransform3DOr, node_path, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXColorOr, node_path, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXStringNameOr, node_path, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXNodePathOr, node_path, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXRIDOr, node_path, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXObjectOr, node_path, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXCallableOr, node_path, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXSignalOr, node_path, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXDictionaryOr, node_path, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXArrayOr, node_path, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedByteArrayOr, node_path, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedInt32ArrayOr, node_path, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedInt64ArrayOr, node_path, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedFloat32ArrayOr, node_path, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedFloat64ArrayOr, node_path, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedStringArrayOr, node_path, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedVector2ArrayOr, node_path, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedVector3ArrayOr, node_path, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedColorArrayOr, node_path, packed_color_array, _operate_or)

// rid
OP_EVALUATOR(OperatorEvaluatorRIDXNilOr, rid, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXBoolOr, rid, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXIntOr, rid, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXFloatOr, rid, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXStringOr, rid, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXVector2Or, rid, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXVector2iOr, rid, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXRect2Or, rid, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXRect2iOr, rid, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXVector3Or, rid, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXVector3iOr, rid, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXTransform2DOr, rid, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPlaneOr, rid, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXQuaternionOr, rid, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXAABBOr, rid, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXBasisOr, rid, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXTransform3DOr, rid, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXColorOr, rid, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXStringNameOr, rid, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXNodePathOr, rid, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXRIDOr, rid, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXObjectOr, rid, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXCallableOr, rid, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXSignalOr, rid, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXDictionaryOr, rid, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXArrayOr, rid, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedByteArrayOr, rid, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedInt32ArrayOr, rid, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedInt64ArrayOr, rid, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedFloat32ArrayOr, rid, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedFloat64ArrayOr, rid, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedStringArrayOr, rid, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedVector2ArrayOr, rid, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedVector3ArrayOr, rid, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedColorArrayOr, rid, packed_color_array, _operate_or)

// object
OP_EVALUATOR(OperatorEvaluatorObjectXNilOr, object, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXBoolOr, object, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXIntOr, object, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXFloatOr, object, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXStringOr, object, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXVector2Or, object, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXVector2iOr, object, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXRect2Or, object, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXRect2iOr, object, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXVector3Or, object, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXVector3iOr, object, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXTransform2DOr, object, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPlaneOr, object, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXQuaternionOr, object, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXAABBOr, object, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXBasisOr, object, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXTransform3DOr, object, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXColorOr, object, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXStringNameOr, object, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXNodePathOr, object, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXRIDOr, object, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXObjectOr, object, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXCallableOr, object, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXSignalOr, object, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXDictionaryOr, object, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXArrayOr, object, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedByteArrayOr, object, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedInt32ArrayOr, object, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedInt64ArrayOr, object, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedFloat32ArrayOr, object, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedFloat64ArrayOr, object, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedStringArrayOr, object, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedVector2ArrayOr, object, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedVector3ArrayOr, object, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedColorArrayOr, object, packed_color_array, _operate_or)

// callable
OP_EVALUATOR(OperatorEvaluatorCallableXNilOr, callable, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXBoolOr, callable, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXIntOr, callable, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXFloatOr, callable, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXStringOr, callable, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXVector2Or, callable, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXVector2iOr, callable, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXRect2Or, callable, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXRect2iOr, callable, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXVector3Or, callable, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXVector3iOr, callable, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXTransform2DOr, callable, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPlaneOr, callable, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXQuaternionOr, callable, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXAABBOr, callable, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXBasisOr, callable, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXTransform3DOr, callable, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXColorOr, callable, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXStringNameOr, callable, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXNodePathOr, callable, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXRIDOr, callable, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXObjectOr, callable, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXCallableOr, callable, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXSignalOr, callable, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXDictionaryOr, callable, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXArrayOr, callable, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedByteArrayOr, callable, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedInt32ArrayOr, callable, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedInt64ArrayOr, callable, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedFloat32ArrayOr, callable, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedFloat64ArrayOr, callable, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedStringArrayOr, callable, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedVector2ArrayOr, callable, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedVector3ArrayOr, callable, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedColorArrayOr, callable, packed_color_array, _operate_or)

// signal
OP_EVALUATOR(OperatorEvaluatorSignalXNilOr, signal, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXBoolOr, signal, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXIntOr, signal, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXFloatOr, signal, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXStringOr, signal, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXVector2Or, signal, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXVector2iOr, signal, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXRect2Or, signal, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXRect2iOr, signal, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXVector3Or, signal, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXVector3iOr, signal, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXTransform2DOr, signal, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPlaneOr, signal, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXQuaternionOr, signal, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXAABBOr, signal, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXBasisOr, signal, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXTransform3DOr, signal, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXColorOr, signal, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXStringNameOr, signal, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXNodePathOr, signal, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXRIDOr, signal, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXObjectOr, signal, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXCallableOr, signal, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXSignalOr, signal, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXDictionaryOr, signal, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXArrayOr, signal, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedByteArrayOr, signal, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedInt32ArrayOr, signal, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedInt64ArrayOr, signal, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedFloat32ArrayOr, signal, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedFloat64ArrayOr, signal, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedStringArrayOr, signal, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedVector2ArrayOr, signal, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedVector3ArrayOr, signal, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedColorArrayOr, signal, packed_color_array, _operate_or)

// dictionary
OP_EVALUATOR(OperatorEvaluatorDictionaryXNilOr, dictionary, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXBoolOr, dictionary, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXIntOr, dictionary, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXFloatOr, dictionary, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXStringOr, dictionary, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXVector2Or, dictionary, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXVector2iOr, dictionary, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXRect2Or, dictionary, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXRect2iOr, dictionary, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXVector3Or, dictionary, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXVector3iOr, dictionary, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXTransform2DOr, dictionary, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPlaneOr, dictionary, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXQuaternionOr, dictionary, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXAABBOr, dictionary, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXBasisOr, dictionary, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXTransform3DOr, dictionary, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXColorOr, dictionary, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXStringNameOr, dictionary, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXNodePathOr, dictionary, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXRIDOr, dictionary, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXObjectOr, dictionary, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXCallableOr, dictionary, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXSignalOr, dictionary, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXDictionaryOr, dictionary, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXArrayOr, dictionary, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedByteArrayOr, dictionary, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedInt32ArrayOr, dictionary, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedInt64ArrayOr, dictionary, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedFloat32ArrayOr, dictionary, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedFloat64ArrayOr, dictionary, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedStringArrayOr, dictionary, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedVector2ArrayOr, dictionary, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedVector3ArrayOr, dictionary, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedColorArrayOr, dictionary, packed_color_array, _operate_or)

// array
OP_EVALUATOR(OperatorEvaluatorArrayXNilOr, array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXBoolOr, array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXIntOr, array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXFloatOr, array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXStringOr, array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXVector2Or, array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXVector2iOr, array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXRect2Or, array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXRect2iOr, array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXVector3Or, array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXVector3iOr, array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXTransform2DOr, array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPlaneOr, array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXQuaternionOr, array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXAABBOr, array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXBasisOr, array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXTransform3DOr, array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXColorOr, array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXStringNameOr, array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXNodePathOr, array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXRIDOr, array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXObjectOr, array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXCallableOr, array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXSignalOr, array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXDictionaryOr, array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXArrayOr, array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedByteArrayOr, array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedInt32ArrayOr, array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedInt64ArrayOr, array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedFloat32ArrayOr, array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedFloat64ArrayOr, array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedStringArrayOr, array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedVector2ArrayOr, array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedVector3ArrayOr, array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedColorArrayOr, array, packed_color_array, _operate_or)

// packed_byte_array
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXNilOr, packed_byte_array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXBoolOr, packed_byte_array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXIntOr, packed_byte_array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXFloatOr, packed_byte_array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXStringOr, packed_byte_array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXVector2Or, packed_byte_array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXVector2iOr, packed_byte_array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXRect2Or, packed_byte_array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXRect2iOr, packed_byte_array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXVector3Or, packed_byte_array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXVector3iOr, packed_byte_array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXTransform2DOr, packed_byte_array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPlaneOr, packed_byte_array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXQuaternionOr, packed_byte_array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXAABBOr, packed_byte_array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXBasisOr, packed_byte_array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXTransform3DOr, packed_byte_array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXColorOr, packed_byte_array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXStringNameOr, packed_byte_array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXNodePathOr, packed_byte_array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXRIDOr, packed_byte_array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXObjectOr, packed_byte_array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXCallableOr, packed_byte_array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXSignalOr, packed_byte_array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXDictionaryOr, packed_byte_array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXArrayOr, packed_byte_array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedByteArrayOr, packed_byte_array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedInt32ArrayOr, packed_byte_array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedInt64ArrayOr, packed_byte_array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedFloat32ArrayOr, packed_byte_array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedFloat64ArrayOr, packed_byte_array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedStringArrayOr, packed_byte_array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedVector2ArrayOr, packed_byte_array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedVector3ArrayOr, packed_byte_array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedColorArrayOr, packed_byte_array, packed_color_array, _operate_or)

// packed_int32_array
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXNilOr, packed_int32_array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXBoolOr, packed_int32_array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXIntOr, packed_int32_array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXFloatOr, packed_int32_array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXStringOr, packed_int32_array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXVector2Or, packed_int32_array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXVector2iOr, packed_int32_array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXRect2Or, packed_int32_array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXRect2iOr, packed_int32_array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXVector3Or, packed_int32_array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXVector3iOr, packed_int32_array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXTransform2DOr, packed_int32_array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPlaneOr, packed_int32_array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXQuaternionOr, packed_int32_array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXAABBOr, packed_int32_array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXBasisOr, packed_int32_array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXTransform3DOr, packed_int32_array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXColorOr, packed_int32_array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXStringNameOr, packed_int32_array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXNodePathOr, packed_int32_array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXRIDOr, packed_int32_array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXObjectOr, packed_int32_array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXCallableOr, packed_int32_array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXSignalOr, packed_int32_array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXDictionaryOr, packed_int32_array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXArrayOr, packed_int32_array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedByteArrayOr, packed_int32_array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedInt32ArrayOr, packed_int32_array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedInt64ArrayOr, packed_int32_array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedFloat32ArrayOr, packed_int32_array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedFloat64ArrayOr, packed_int32_array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedStringArrayOr, packed_int32_array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedVector2ArrayOr, packed_int32_array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedVector3ArrayOr, packed_int32_array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedColorArrayOr, packed_int32_array, packed_color_array, _operate_or)

// packed_int64_array
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXNilOr, packed_int64_array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXBoolOr, packed_int64_array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXIntOr, packed_int64_array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXFloatOr, packed_int64_array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXStringOr, packed_int64_array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXVector2Or, packed_int64_array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXVector2iOr, packed_int64_array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXRect2Or, packed_int64_array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXRect2iOr, packed_int64_array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXVector3Or, packed_int64_array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXVector3iOr, packed_int64_array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXTransform2DOr, packed_int64_array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPlaneOr, packed_int64_array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXQuaternionOr, packed_int64_array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXAABBOr, packed_int64_array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXBasisOr, packed_int64_array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXTransform3DOr, packed_int64_array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXColorOr, packed_int64_array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXStringNameOr, packed_int64_array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXNodePathOr, packed_int64_array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXRIDOr, packed_int64_array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXObjectOr, packed_int64_array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXCallableOr, packed_int64_array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXSignalOr, packed_int64_array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXDictionaryOr, packed_int64_array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXArrayOr, packed_int64_array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedByteArrayOr, packed_int64_array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedInt32ArrayOr, packed_int64_array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedInt64ArrayOr, packed_int64_array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedFloat32ArrayOr, packed_int64_array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedFloat64ArrayOr, packed_int64_array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedStringArrayOr, packed_int64_array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedVector2ArrayOr, packed_int64_array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedVector3ArrayOr, packed_int64_array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedColorArrayOr, packed_int64_array, packed_color_array, _operate_or)

// packed_float32_array
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXNilOr, packed_float32_array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXBoolOr, packed_float32_array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXIntOr, packed_float32_array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXFloatOr, packed_float32_array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXStringOr, packed_float32_array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXVector2Or, packed_float32_array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXVector2iOr, packed_float32_array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXRect2Or, packed_float32_array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXRect2iOr, packed_float32_array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXVector3Or, packed_float32_array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXVector3iOr, packed_float32_array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXTransform2DOr, packed_float32_array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPlaneOr, packed_float32_array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXQuaternionOr, packed_float32_array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXAABBOr, packed_float32_array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXBasisOr, packed_float32_array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXTransform3DOr, packed_float32_array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXColorOr, packed_float32_array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXStringNameOr, packed_float32_array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXNodePathOr, packed_float32_array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXRIDOr, packed_float32_array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXObjectOr, packed_float32_array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXCallableOr, packed_float32_array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXSignalOr, packed_float32_array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXDictionaryOr, packed_float32_array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXArrayOr, packed_float32_array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedByteArrayOr, packed_float32_array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedInt32ArrayOr, packed_float32_array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedInt64ArrayOr, packed_float32_array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedFloat32ArrayOr, packed_float32_array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedFloat64ArrayOr, packed_float32_array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedStringArrayOr, packed_float32_array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedVector2ArrayOr, packed_float32_array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedVector3ArrayOr, packed_float32_array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedColorArrayOr, packed_float32_array, packed_color_array, _operate_or)

// packed_float64_array
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXNilOr, packed_float64_array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXBoolOr, packed_float64_array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXIntOr, packed_float64_array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXFloatOr, packed_float64_array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXStringOr, packed_float64_array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXVector2Or, packed_float64_array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXVector2iOr, packed_float64_array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXRect2Or, packed_float64_array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXRect2iOr, packed_float64_array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXVector3Or, packed_float64_array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXVector3iOr, packed_float64_array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXTransform2DOr, packed_float64_array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPlaneOr, packed_float64_array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXQuaternionOr, packed_float64_array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXAABBOr, packed_float64_array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXBasisOr, packed_float64_array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXTransform3DOr, packed_float64_array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXColorOr, packed_float64_array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXStringNameOr, packed_float64_array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXNodePathOr, packed_float64_array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXRIDOr, packed_float64_array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXObjectOr, packed_float64_array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXCallableOr, packed_float64_array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXSignalOr, packed_float64_array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXDictionaryOr, packed_float64_array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXArrayOr, packed_float64_array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedByteArrayOr, packed_float64_array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedInt32ArrayOr, packed_float64_array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedInt64ArrayOr, packed_float64_array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedFloat32ArrayOr, packed_float64_array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedFloat64ArrayOr, packed_float64_array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedStringArrayOr, packed_float64_array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedVector2ArrayOr, packed_float64_array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedVector3ArrayOr, packed_float64_array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedColorArrayOr, packed_float64_array, packed_color_array, _operate_or)

// packed_string_array
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXNilOr, packed_string_array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXBoolOr, packed_string_array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXIntOr, packed_string_array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXFloatOr, packed_string_array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXStringOr, packed_string_array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXVector2Or, packed_string_array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXVector2iOr, packed_string_array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXRect2Or, packed_string_array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXRect2iOr, packed_string_array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXVector3Or, packed_string_array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXVector3iOr, packed_string_array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXTransform2DOr, packed_string_array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPlaneOr, packed_string_array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXQuaternionOr, packed_string_array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXAABBOr, packed_string_array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXBasisOr, packed_string_array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXTransform3DOr, packed_string_array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXColorOr, packed_string_array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXStringNameOr, packed_string_array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXNodePathOr, packed_string_array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXRIDOr, packed_string_array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXObjectOr, packed_string_array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXCallableOr, packed_string_array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXSignalOr, packed_string_array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXDictionaryOr, packed_string_array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXArrayOr, packed_string_array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedByteArrayOr, packed_string_array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedInt32ArrayOr, packed_string_array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedInt64ArrayOr, packed_string_array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedFloat32ArrayOr, packed_string_array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedFloat64ArrayOr, packed_string_array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedStringArrayOr, packed_string_array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedVector2ArrayOr, packed_string_array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedVector3ArrayOr, packed_string_array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedColorArrayOr, packed_string_array, packed_color_array, _operate_or)

// packed_vector2_array
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXNilOr, packed_vector2_array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXBoolOr, packed_vector2_array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXIntOr, packed_vector2_array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXFloatOr, packed_vector2_array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXStringOr, packed_vector2_array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXVector2Or, packed_vector2_array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXVector2iOr, packed_vector2_array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXRect2Or, packed_vector2_array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXRect2iOr, packed_vector2_array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXVector3Or, packed_vector2_array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXVector3iOr, packed_vector2_array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXTransform2DOr, packed_vector2_array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPlaneOr, packed_vector2_array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXQuaternionOr, packed_vector2_array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXAABBOr, packed_vector2_array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXBasisOr, packed_vector2_array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXTransform3DOr, packed_vector2_array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXColorOr, packed_vector2_array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXStringNameOr, packed_vector2_array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXNodePathOr, packed_vector2_array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXRIDOr, packed_vector2_array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXObjectOr, packed_vector2_array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXCallableOr, packed_vector2_array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXSignalOr, packed_vector2_array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXDictionaryOr, packed_vector2_array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXArrayOr, packed_vector2_array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedByteArrayOr, packed_vector2_array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedInt32ArrayOr, packed_vector2_array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedInt64ArrayOr, packed_vector2_array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedFloat32ArrayOr, packed_vector2_array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedFloat64ArrayOr, packed_vector2_array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedStringArrayOr, packed_vector2_array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedVector2ArrayOr, packed_vector2_array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedVector3ArrayOr, packed_vector2_array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedColorArrayOr, packed_vector2_array, packed_color_array, _operate_or)

// packed_vector3_array
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXNilOr, packed_vector3_array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXBoolOr, packed_vector3_array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXIntOr, packed_vector3_array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXFloatOr, packed_vector3_array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXStringOr, packed_vector3_array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXVector2Or, packed_vector3_array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXVector2iOr, packed_vector3_array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXRect2Or, packed_vector3_array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXRect2iOr, packed_vector3_array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXVector3Or, packed_vector3_array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXVector3iOr, packed_vector3_array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXTransform2DOr, packed_vector3_array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPlaneOr, packed_vector3_array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXQuaternionOr, packed_vector3_array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXAABBOr, packed_vector3_array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXBasisOr, packed_vector3_array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXTransform3DOr, packed_vector3_array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXColorOr, packed_vector3_array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXStringNameOr, packed_vector3_array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXNodePathOr, packed_vector3_array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXRIDOr, packed_vector3_array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXObjectOr, packed_vector3_array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXCallableOr, packed_vector3_array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXSignalOr, packed_vector3_array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXDictionaryOr, packed_vector3_array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXArrayOr, packed_vector3_array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedByteArrayOr, packed_vector3_array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedInt32ArrayOr, packed_vector3_array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedInt64ArrayOr, packed_vector3_array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedFloat32ArrayOr, packed_vector3_array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedFloat64ArrayOr, packed_vector3_array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedStringArrayOr, packed_vector3_array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedVector2ArrayOr, packed_vector3_array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedVector3ArrayOr, packed_vector3_array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedColorArrayOr, packed_vector3_array, packed_color_array, _operate_or)

// packed_color_array
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXNilOr, packed_color_array, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXBoolOr, packed_color_array, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXIntOr, packed_color_array, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXFloatOr, packed_color_array, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXStringOr, packed_color_array, string, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXVector2Or, packed_color_array, vector2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXVector2iOr, packed_color_array, vector2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXRect2Or, packed_color_array, rect2, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXRect2iOr, packed_color_array, rect2i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXVector3Or, packed_color_array, vector3, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXVector3iOr, packed_color_array, vector3i, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXTransform2DOr, packed_color_array, transform2d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPlaneOr, packed_color_array, plane, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXQuaternionOr, packed_color_array, quaternion, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXAABBOr, packed_color_array, aabb, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXBasisOr, packed_color_array, basis, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXTransform3DOr, packed_color_array, transform3d, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXColorOr, packed_color_array, color, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXStringNameOr, packed_color_array, string_name, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXNodePathOr, packed_color_array, node_path, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXRIDOr, packed_color_array, rid, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXObjectOr, packed_color_array, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXCallableOr, packed_color_array, callable, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXSignalOr, packed_color_array, signal, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXDictionaryOr, packed_color_array, dictionary, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXArrayOr, packed_color_array, array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedByteArrayOr, packed_color_array, packed_byte_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedInt32ArrayOr, packed_color_array, packed_int32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedInt64ArrayOr, packed_color_array, packed_int64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedFloat32ArrayOr, packed_color_array, packed_float32_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedFloat64ArrayOr, packed_color_array, packed_float64_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedStringArrayOr, packed_color_array, packed_string_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedVector2ArrayOr, packed_color_array, packed_vector2_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedVector3ArrayOr, packed_color_array, packed_vector3_array, _operate_or)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedColorArrayOr, packed_color_array, packed_color_array, _operate_or)

// AND

// nil
OP_EVALUATOR(OperatorEvaluatorNilXNilAnd, nil, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXBoolAnd, nil, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXIntAnd, nil, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXFloatAnd, nil, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXStringAnd, nil, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXVector2And, nil, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXVector2iAnd, nil, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXRect2And, nil, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXRect2iAnd, nil, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXVector3And, nil, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXVector3iAnd, nil, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXTransform2DAnd, nil, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPlaneAnd, nil, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXQuaternionAnd, nil, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXAABBAnd, nil, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXBasisAnd, nil, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXTransform3DAnd, nil, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXColorAnd, nil, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXStringNameAnd, nil, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXNodePathAnd, nil, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXRIDAnd, nil, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXObjectAnd, nil, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXCallableAnd, nil, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXSignalAnd, nil, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXDictionaryAnd, nil, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXArrayAnd, nil, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPackedByteArrayAnd, nil, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPackedInt32ArrayAnd, nil, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPackedInt64ArrayAnd, nil, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPackedFloat32ArrayAnd, nil, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPackedFloat64ArrayAnd, nil, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPackedStringArrayAnd, nil, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPackedVector2ArrayAnd, nil, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPackedVector3ArrayAnd, nil, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXPackedColorArrayAnd, nil, packed_color_array, _operate_and)

// bool
OP_EVALUATOR(OperatorEvaluatorBoolXNilAnd, bool, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXBoolAnd, bool, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXIntAnd, bool, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXFloatAnd, bool, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXStringAnd, bool, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXVector2And, bool, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXVector2iAnd, bool, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXRect2And, bool, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXRect2iAnd, bool, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXVector3And, bool, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXVector3iAnd, bool, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXTransform2DAnd, bool, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPlaneAnd, bool, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXQuaternionAnd, bool, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXAABBAnd, bool, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXBasisAnd, bool, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXTransform3DAnd, bool, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXColorAnd, bool, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXStringNameAnd, bool, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXNodePathAnd, bool, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXRIDAnd, bool, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXObjectAnd, bool, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXCallableAnd, bool, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXSignalAnd, bool, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXDictionaryAnd, bool, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXArrayAnd, bool, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedByteArrayAnd, bool, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedInt32ArrayAnd, bool, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedInt64ArrayAnd, bool, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedFloat32ArrayAnd, bool, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedFloat64ArrayAnd, bool, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedStringArrayAnd, bool, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedVector2ArrayAnd, bool, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedVector3ArrayAnd, bool, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXPackedColorArrayAnd, bool, packed_color_array, _operate_and)

// int
OP_EVALUATOR(OperatorEvaluatorIntXNilAnd, int, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXBoolAnd, int, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXIntAnd, int, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXFloatAnd, int, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXStringAnd, int, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXVector2And, int, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXVector2iAnd, int, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXRect2And, int, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXRect2iAnd, int, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXVector3And, int, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXVector3iAnd, int, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXTransform2DAnd, int, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPlaneAnd, int, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXQuaternionAnd, int, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXAABBAnd, int, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXBasisAnd, int, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXTransform3DAnd, int, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXColorAnd, int, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXStringNameAnd, int, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXNodePathAnd, int, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXRIDAnd, int, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXObjectAnd, int, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXCallableAnd, int, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXSignalAnd, int, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXDictionaryAnd, int, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXArrayAnd, int, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPackedByteArrayAnd, int, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPackedInt32ArrayAnd, int, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPackedInt64ArrayAnd, int, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPackedFloat32ArrayAnd, int, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPackedFloat64ArrayAnd, int, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPackedStringArrayAnd, int, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPackedVector2ArrayAnd, int, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPackedVector3ArrayAnd, int, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXPackedColorArrayAnd, int, packed_color_array, _operate_and)

// float
OP_EVALUATOR(OperatorEvaluatorFloatXNilAnd, float, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXBoolAnd, float, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXIntAnd, float, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXFloatAnd, float, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXStringAnd, float, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXVector2And, float, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXVector2iAnd, float, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXRect2And, float, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXRect2iAnd, float, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXVector3And, float, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXVector3iAnd, float, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXTransform2DAnd, float, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPlaneAnd, float, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXQuaternionAnd, float, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXAABBAnd, float, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXBasisAnd, float, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXTransform3DAnd, float, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXColorAnd, float, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXStringNameAnd, float, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXNodePathAnd, float, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXRIDAnd, float, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXObjectAnd, float, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXCallableAnd, float, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXSignalAnd, float, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXDictionaryAnd, float, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXArrayAnd, float, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedByteArrayAnd, float, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedInt32ArrayAnd, float, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedInt64ArrayAnd, float, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedFloat32ArrayAnd, float, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedFloat64ArrayAnd, float, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedStringArrayAnd, float, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedVector2ArrayAnd, float, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedVector3ArrayAnd, float, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXPackedColorArrayAnd, float, packed_color_array, _operate_and)

// string
OP_EVALUATOR(OperatorEvaluatorStringXNilAnd, string, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXBoolAnd, string, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXIntAnd, string, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXFloatAnd, string, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXStringAnd, string, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXVector2And, string, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXVector2iAnd, string, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXRect2And, string, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXRect2iAnd, string, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXVector3And, string, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXVector3iAnd, string, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXTransform2DAnd, string, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPlaneAnd, string, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXQuaternionAnd, string, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXAABBAnd, string, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXBasisAnd, string, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXTransform3DAnd, string, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXColorAnd, string, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXStringNameAnd, string, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXNodePathAnd, string, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXRIDAnd, string, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXObjectAnd, string, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXCallableAnd, string, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXSignalAnd, string, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXDictionaryAnd, string, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXArrayAnd, string, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPackedByteArrayAnd, string, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPackedInt32ArrayAnd, string, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPackedInt64ArrayAnd, string, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPackedFloat32ArrayAnd, string, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPackedFloat64ArrayAnd, string, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPackedStringArrayAnd, string, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPackedVector2ArrayAnd, string, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPackedVector3ArrayAnd, string, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringXPackedColorArrayAnd, string, packed_color_array, _operate_and)

// vector2
OP_EVALUATOR(OperatorEvaluatorVector2XNilAnd, vector2, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XBoolAnd, vector2, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XIntAnd, vector2, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XFloatAnd, vector2, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XStringAnd, vector2, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XVector2And, vector2, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XVector2iAnd, vector2, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XRect2And, vector2, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XRect2iAnd, vector2, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XVector3And, vector2, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XVector3iAnd, vector2, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XTransform2DAnd, vector2, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPlaneAnd, vector2, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XQuaternionAnd, vector2, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XAABBAnd, vector2, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XBasisAnd, vector2, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XTransform3DAnd, vector2, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XColorAnd, vector2, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XStringNameAnd, vector2, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XNodePathAnd, vector2, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XRIDAnd, vector2, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XObjectAnd, vector2, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XCallableAnd, vector2, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XSignalAnd, vector2, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XDictionaryAnd, vector2, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XArrayAnd, vector2, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedByteArrayAnd, vector2, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedInt32ArrayAnd, vector2, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedInt64ArrayAnd, vector2, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedFloat32ArrayAnd, vector2, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedFloat64ArrayAnd, vector2, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedStringArrayAnd, vector2, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedVector2ArrayAnd, vector2, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedVector3ArrayAnd, vector2, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2XPackedColorArrayAnd, vector2, packed_color_array, _operate_and)

// vector2i
OP_EVALUATOR(OperatorEvaluatorVector2iXNilAnd, vector2i, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXBoolAnd, vector2i, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXIntAnd, vector2i, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXFloatAnd, vector2i, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXStringAnd, vector2i, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXVector2And, vector2i, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXVector2iAnd, vector2i, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXRect2And, vector2i, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXRect2iAnd, vector2i, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXVector3And, vector2i, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXVector3iAnd, vector2i, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXTransform2DAnd, vector2i, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPlaneAnd, vector2i, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXQuaternionAnd, vector2i, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXAABBAnd, vector2i, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXBasisAnd, vector2i, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXTransform3DAnd, vector2i, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXColorAnd, vector2i, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXStringNameAnd, vector2i, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXNodePathAnd, vector2i, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXRIDAnd, vector2i, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXObjectAnd, vector2i, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXCallableAnd, vector2i, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXSignalAnd, vector2i, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXDictionaryAnd, vector2i, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXArrayAnd, vector2i, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedByteArrayAnd, vector2i, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedInt32ArrayAnd, vector2i, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedInt64ArrayAnd, vector2i, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedFloat32ArrayAnd, vector2i, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedFloat64ArrayAnd, vector2i, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedStringArrayAnd, vector2i, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedVector2ArrayAnd, vector2i, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedVector3ArrayAnd, vector2i, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector2iXPackedColorArrayAnd, vector2i, packed_color_array, _operate_and)

// rect2
OP_EVALUATOR(OperatorEvaluatorRect2XNilAnd, rect2, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XBoolAnd, rect2, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XIntAnd, rect2, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XFloatAnd, rect2, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XStringAnd, rect2, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XVector2And, rect2, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XVector2iAnd, rect2, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XRect2And, rect2, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XRect2iAnd, rect2, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XVector3And, rect2, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XVector3iAnd, rect2, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XTransform2DAnd, rect2, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPlaneAnd, rect2, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XQuaternionAnd, rect2, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XAABBAnd, rect2, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XBasisAnd, rect2, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XTransform3DAnd, rect2, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XColorAnd, rect2, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XStringNameAnd, rect2, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XNodePathAnd, rect2, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XRIDAnd, rect2, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XObjectAnd, rect2, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XCallableAnd, rect2, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XSignalAnd, rect2, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XDictionaryAnd, rect2, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XArrayAnd, rect2, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedByteArrayAnd, rect2, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedInt32ArrayAnd, rect2, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedInt64ArrayAnd, rect2, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedFloat32ArrayAnd, rect2, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedFloat64ArrayAnd, rect2, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedStringArrayAnd, rect2, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedVector2ArrayAnd, rect2, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedVector3ArrayAnd, rect2, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2XPackedColorArrayAnd, rect2, packed_color_array, _operate_and)

// rect2i
OP_EVALUATOR(OperatorEvaluatorRect2iXNilAnd, rect2i, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXBoolAnd, rect2i, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXIntAnd, rect2i, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXFloatAnd, rect2i, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXStringAnd, rect2i, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXVector2And, rect2i, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXVector2iAnd, rect2i, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXRect2And, rect2i, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXRect2iAnd, rect2i, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXVector3And, rect2i, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXVector3iAnd, rect2i, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXTransform2DAnd, rect2i, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPlaneAnd, rect2i, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXQuaternionAnd, rect2i, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXAABBAnd, rect2i, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXBasisAnd, rect2i, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXTransform3DAnd, rect2i, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXColorAnd, rect2i, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXStringNameAnd, rect2i, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXNodePathAnd, rect2i, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXRIDAnd, rect2i, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXObjectAnd, rect2i, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXCallableAnd, rect2i, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXSignalAnd, rect2i, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXDictionaryAnd, rect2i, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXArrayAnd, rect2i, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedByteArrayAnd, rect2i, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedInt32ArrayAnd, rect2i, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedInt64ArrayAnd, rect2i, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedFloat32ArrayAnd, rect2i, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedFloat64ArrayAnd, rect2i, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedStringArrayAnd, rect2i, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedVector2ArrayAnd, rect2i, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedVector3ArrayAnd, rect2i, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRect2iXPackedColorArrayAnd, rect2i, packed_color_array, _operate_and)

// vector3
OP_EVALUATOR(OperatorEvaluatorVector3XNilAnd, vector3, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XBoolAnd, vector3, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XIntAnd, vector3, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XFloatAnd, vector3, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XStringAnd, vector3, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XVector2And, vector3, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XVector2iAnd, vector3, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XRect2And, vector3, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XRect2iAnd, vector3, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XVector3And, vector3, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XVector3iAnd, vector3, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XTransform2DAnd, vector3, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPlaneAnd, vector3, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XQuaternionAnd, vector3, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XAABBAnd, vector3, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XBasisAnd, vector3, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XTransform3DAnd, vector3, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XColorAnd, vector3, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XStringNameAnd, vector3, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XNodePathAnd, vector3, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XRIDAnd, vector3, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XObjectAnd, vector3, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XCallableAnd, vector3, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XSignalAnd, vector3, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XDictionaryAnd, vector3, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XArrayAnd, vector3, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedByteArrayAnd, vector3, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedInt32ArrayAnd, vector3, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedInt64ArrayAnd, vector3, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedFloat32ArrayAnd, vector3, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedFloat64ArrayAnd, vector3, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedStringArrayAnd, vector3, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedVector2ArrayAnd, vector3, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedVector3ArrayAnd, vector3, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3XPackedColorArrayAnd, vector3, packed_color_array, _operate_and)

// vector3i
OP_EVALUATOR(OperatorEvaluatorVector3iXNilAnd, vector3i, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXBoolAnd, vector3i, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXIntAnd, vector3i, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXFloatAnd, vector3i, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXStringAnd, vector3i, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXVector2And, vector3i, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXVector2iAnd, vector3i, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXRect2And, vector3i, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXRect2iAnd, vector3i, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXVector3And, vector3i, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXVector3iAnd, vector3i, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXTransform2DAnd, vector3i, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPlaneAnd, vector3i, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXQuaternionAnd, vector3i, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXAABBAnd, vector3i, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXBasisAnd, vector3i, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXTransform3DAnd, vector3i, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXColorAnd, vector3i, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXStringNameAnd, vector3i, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXNodePathAnd, vector3i, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXRIDAnd, vector3i, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXObjectAnd, vector3i, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXCallableAnd, vector3i, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXSignalAnd, vector3i, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXDictionaryAnd, vector3i, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXArrayAnd, vector3i, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedByteArrayAnd, vector3i, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedInt32ArrayAnd, vector3i, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedInt64ArrayAnd, vector3i, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedFloat32ArrayAnd, vector3i, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedFloat64ArrayAnd, vector3i, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedStringArrayAnd, vector3i, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedVector2ArrayAnd, vector3i, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedVector3ArrayAnd, vector3i, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorVector3iXPackedColorArrayAnd, vector3i, packed_color_array, _operate_and)

// transform2d
OP_EVALUATOR(OperatorEvaluatorTransform2DXNilAnd, transform2d, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXBoolAnd, transform2d, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXIntAnd, transform2d, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXFloatAnd, transform2d, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXStringAnd, transform2d, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXVector2And, transform2d, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXVector2iAnd, transform2d, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXRect2And, transform2d, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXRect2iAnd, transform2d, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXVector3And, transform2d, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXVector3iAnd, transform2d, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXTransform2DAnd, transform2d, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPlaneAnd, transform2d, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXQuaternionAnd, transform2d, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXAABBAnd, transform2d, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXBasisAnd, transform2d, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXTransform3DAnd, transform2d, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXColorAnd, transform2d, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXStringNameAnd, transform2d, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXNodePathAnd, transform2d, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXRIDAnd, transform2d, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXObjectAnd, transform2d, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXCallableAnd, transform2d, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXSignalAnd, transform2d, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXDictionaryAnd, transform2d, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXArrayAnd, transform2d, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedByteArrayAnd, transform2d, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedInt32ArrayAnd, transform2d, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedInt64ArrayAnd, transform2d, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedFloat32ArrayAnd, transform2d, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedFloat64ArrayAnd, transform2d, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedStringArrayAnd, transform2d, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedVector2ArrayAnd, transform2d, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedVector3ArrayAnd, transform2d, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform2DXPackedColorArrayAnd, transform2d, packed_color_array, _operate_and)

// plane
OP_EVALUATOR(OperatorEvaluatorPlaneXNilAnd, plane, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXBoolAnd, plane, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXIntAnd, plane, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXFloatAnd, plane, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXStringAnd, plane, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXVector2And, plane, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXVector2iAnd, plane, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXRect2And, plane, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXRect2iAnd, plane, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXVector3And, plane, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXVector3iAnd, plane, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXTransform2DAnd, plane, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPlaneAnd, plane, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXQuaternionAnd, plane, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXAABBAnd, plane, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXBasisAnd, plane, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXTransform3DAnd, plane, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXColorAnd, plane, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXStringNameAnd, plane, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXNodePathAnd, plane, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXRIDAnd, plane, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXObjectAnd, plane, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXCallableAnd, plane, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXSignalAnd, plane, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXDictionaryAnd, plane, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXArrayAnd, plane, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedByteArrayAnd, plane, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedInt32ArrayAnd, plane, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedInt64ArrayAnd, plane, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedFloat32ArrayAnd, plane, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedFloat64ArrayAnd, plane, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedStringArrayAnd, plane, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedVector2ArrayAnd, plane, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedVector3ArrayAnd, plane, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPlaneXPackedColorArrayAnd, plane, packed_color_array, _operate_and)

// quaternion
OP_EVALUATOR(OperatorEvaluatorQuaternionXNilAnd, quaternion, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXBoolAnd, quaternion, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXIntAnd, quaternion, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXFloatAnd, quaternion, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXStringAnd, quaternion, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXVector2And, quaternion, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXVector2iAnd, quaternion, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXRect2And, quaternion, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXRect2iAnd, quaternion, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXVector3And, quaternion, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXVector3iAnd, quaternion, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXTransform2DAnd, quaternion, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPlaneAnd, quaternion, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXQuaternionAnd, quaternion, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXAABBAnd, quaternion, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXBasisAnd, quaternion, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXTransform3DAnd, quaternion, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXColorAnd, quaternion, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXStringNameAnd, quaternion, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXNodePathAnd, quaternion, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXRIDAnd, quaternion, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXObjectAnd, quaternion, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXCallableAnd, quaternion, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXSignalAnd, quaternion, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXDictionaryAnd, quaternion, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXArrayAnd, quaternion, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedByteArrayAnd, quaternion, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedInt32ArrayAnd, quaternion, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedInt64ArrayAnd, quaternion, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedFloat32ArrayAnd, quaternion, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedFloat64ArrayAnd, quaternion, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedStringArrayAnd, quaternion, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedVector2ArrayAnd, quaternion, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedVector3ArrayAnd, quaternion, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorQuaternionXPackedColorArrayAnd, quaternion, packed_color_array, _operate_and)

// aabb
OP_EVALUATOR(OperatorEvaluatorAABBXNilAnd, aabb, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXBoolAnd, aabb, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXIntAnd, aabb, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXFloatAnd, aabb, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXStringAnd, aabb, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXVector2And, aabb, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXVector2iAnd, aabb, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXRect2And, aabb, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXRect2iAnd, aabb, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXVector3And, aabb, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXVector3iAnd, aabb, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXTransform2DAnd, aabb, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPlaneAnd, aabb, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXQuaternionAnd, aabb, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXAABBAnd, aabb, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXBasisAnd, aabb, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXTransform3DAnd, aabb, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXColorAnd, aabb, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXStringNameAnd, aabb, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXNodePathAnd, aabb, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXRIDAnd, aabb, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXObjectAnd, aabb, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXCallableAnd, aabb, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXSignalAnd, aabb, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXDictionaryAnd, aabb, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXArrayAnd, aabb, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedByteArrayAnd, aabb, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedInt32ArrayAnd, aabb, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedInt64ArrayAnd, aabb, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedFloat32ArrayAnd, aabb, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedFloat64ArrayAnd, aabb, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedStringArrayAnd, aabb, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedVector2ArrayAnd, aabb, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedVector3ArrayAnd, aabb, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorAABBXPackedColorArrayAnd, aabb, packed_color_array, _operate_and)

// basis
OP_EVALUATOR(OperatorEvaluatorBasisXNilAnd, basis, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXBoolAnd, basis, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXIntAnd, basis, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXFloatAnd, basis, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXStringAnd, basis, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXVector2And, basis, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXVector2iAnd, basis, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXRect2And, basis, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXRect2iAnd, basis, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXVector3And, basis, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXVector3iAnd, basis, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXTransform2DAnd, basis, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPlaneAnd, basis, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXQuaternionAnd, basis, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXAABBAnd, basis, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXBasisAnd, basis, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXTransform3DAnd, basis, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXColorAnd, basis, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXStringNameAnd, basis, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXNodePathAnd, basis, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXRIDAnd, basis, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXObjectAnd, basis, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXCallableAnd, basis, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXSignalAnd, basis, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXDictionaryAnd, basis, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXArrayAnd, basis, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedByteArrayAnd, basis, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedInt32ArrayAnd, basis, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedInt64ArrayAnd, basis, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedFloat32ArrayAnd, basis, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedFloat64ArrayAnd, basis, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedStringArrayAnd, basis, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedVector2ArrayAnd, basis, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedVector3ArrayAnd, basis, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBasisXPackedColorArrayAnd, basis, packed_color_array, _operate_and)

// transform3d
OP_EVALUATOR(OperatorEvaluatorTransform3DXNilAnd, transform3d, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXBoolAnd, transform3d, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXIntAnd, transform3d, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXFloatAnd, transform3d, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXStringAnd, transform3d, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXVector2And, transform3d, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXVector2iAnd, transform3d, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXRect2And, transform3d, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXRect2iAnd, transform3d, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXVector3And, transform3d, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXVector3iAnd, transform3d, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXTransform2DAnd, transform3d, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPlaneAnd, transform3d, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXQuaternionAnd, transform3d, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXAABBAnd, transform3d, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXBasisAnd, transform3d, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXTransform3DAnd, transform3d, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXColorAnd, transform3d, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXStringNameAnd, transform3d, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXNodePathAnd, transform3d, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXRIDAnd, transform3d, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXObjectAnd, transform3d, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXCallableAnd, transform3d, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXSignalAnd, transform3d, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXDictionaryAnd, transform3d, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXArrayAnd, transform3d, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedByteArrayAnd, transform3d, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedInt32ArrayAnd, transform3d, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedInt64ArrayAnd, transform3d, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedFloat32ArrayAnd, transform3d, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedFloat64ArrayAnd, transform3d, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedStringArrayAnd, transform3d, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedVector2ArrayAnd, transform3d, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedVector3ArrayAnd, transform3d, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorTransform3DXPackedColorArrayAnd, transform3d, packed_color_array, _operate_and)

// color
OP_EVALUATOR(OperatorEvaluatorColorXNilAnd, color, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXBoolAnd, color, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXIntAnd, color, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXFloatAnd, color, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXStringAnd, color, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXVector2And, color, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXVector2iAnd, color, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXRect2And, color, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXRect2iAnd, color, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXVector3And, color, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXVector3iAnd, color, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXTransform2DAnd, color, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPlaneAnd, color, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXQuaternionAnd, color, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXAABBAnd, color, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXBasisAnd, color, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXTransform3DAnd, color, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXColorAnd, color, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXStringNameAnd, color, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXNodePathAnd, color, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXRIDAnd, color, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXObjectAnd, color, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXCallableAnd, color, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXSignalAnd, color, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXDictionaryAnd, color, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXArrayAnd, color, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPackedByteArrayAnd, color, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPackedInt32ArrayAnd, color, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPackedInt64ArrayAnd, color, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPackedFloat32ArrayAnd, color, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPackedFloat64ArrayAnd, color, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPackedStringArrayAnd, color, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPackedVector2ArrayAnd, color, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPackedVector3ArrayAnd, color, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorColorXPackedColorArrayAnd, color, packed_color_array, _operate_and)

// string_name
OP_EVALUATOR(OperatorEvaluatorStringNameXNilAnd, string_name, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXBoolAnd, string_name, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXIntAnd, string_name, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXFloatAnd, string_name, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXStringAnd, string_name, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXVector2And, string_name, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXVector2iAnd, string_name, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXRect2And, string_name, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXRect2iAnd, string_name, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXVector3And, string_name, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXVector3iAnd, string_name, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXTransform2DAnd, string_name, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPlaneAnd, string_name, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXQuaternionAnd, string_name, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXAABBAnd, string_name, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXBasisAnd, string_name, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXTransform3DAnd, string_name, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXColorAnd, string_name, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXStringNameAnd, string_name, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXNodePathAnd, string_name, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXRIDAnd, string_name, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXObjectAnd, string_name, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXCallableAnd, string_name, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXSignalAnd, string_name, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXDictionaryAnd, string_name, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXArrayAnd, string_name, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedByteArrayAnd, string_name, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedInt32ArrayAnd, string_name, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedInt64ArrayAnd, string_name, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedFloat32ArrayAnd, string_name, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedFloat64ArrayAnd, string_name, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedStringArrayAnd, string_name, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedVector2ArrayAnd, string_name, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedVector3ArrayAnd, string_name, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorStringNameXPackedColorArrayAnd, string_name, packed_color_array, _operate_and)

// node_path
OP_EVALUATOR(OperatorEvaluatorNodePathXNilAnd, node_path, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXBoolAnd, node_path, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXIntAnd, node_path, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXFloatAnd, node_path, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXStringAnd, node_path, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXVector2And, node_path, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXVector2iAnd, node_path, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXRect2And, node_path, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXRect2iAnd, node_path, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXVector3And, node_path, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXVector3iAnd, node_path, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXTransform2DAnd, node_path, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPlaneAnd, node_path, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXQuaternionAnd, node_path, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXAABBAnd, node_path, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXBasisAnd, node_path, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXTransform3DAnd, node_path, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXColorAnd, node_path, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXStringNameAnd, node_path, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXNodePathAnd, node_path, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXRIDAnd, node_path, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXObjectAnd, node_path, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXCallableAnd, node_path, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXSignalAnd, node_path, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXDictionaryAnd, node_path, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXArrayAnd, node_path, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedByteArrayAnd, node_path, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedInt32ArrayAnd, node_path, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedInt64ArrayAnd, node_path, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedFloat32ArrayAnd, node_path, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedFloat64ArrayAnd, node_path, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedStringArrayAnd, node_path, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedVector2ArrayAnd, node_path, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedVector3ArrayAnd, node_path, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNodePathXPackedColorArrayAnd, node_path, packed_color_array, _operate_and)

// rid
OP_EVALUATOR(OperatorEvaluatorRIDXNilAnd, rid, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXBoolAnd, rid, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXIntAnd, rid, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXFloatAnd, rid, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXStringAnd, rid, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXVector2And, rid, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXVector2iAnd, rid, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXRect2And, rid, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXRect2iAnd, rid, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXVector3And, rid, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXVector3iAnd, rid, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXTransform2DAnd, rid, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPlaneAnd, rid, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXQuaternionAnd, rid, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXAABBAnd, rid, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXBasisAnd, rid, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXTransform3DAnd, rid, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXColorAnd, rid, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXStringNameAnd, rid, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXNodePathAnd, rid, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXRIDAnd, rid, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXObjectAnd, rid, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXCallableAnd, rid, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXSignalAnd, rid, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXDictionaryAnd, rid, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXArrayAnd, rid, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedByteArrayAnd, rid, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedInt32ArrayAnd, rid, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedInt64ArrayAnd, rid, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedFloat32ArrayAnd, rid, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedFloat64ArrayAnd, rid, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedStringArrayAnd, rid, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedVector2ArrayAnd, rid, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedVector3ArrayAnd, rid, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorRIDXPackedColorArrayAnd, rid, packed_color_array, _operate_and)

// object
OP_EVALUATOR(OperatorEvaluatorObjectXNilAnd, object, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXBoolAnd, object, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXIntAnd, object, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXFloatAnd, object, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXStringAnd, object, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXVector2And, object, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXVector2iAnd, object, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXRect2And, object, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXRect2iAnd, object, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXVector3And, object, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXVector3iAnd, object, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXTransform2DAnd, object, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPlaneAnd, object, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXQuaternionAnd, object, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXAABBAnd, object, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXBasisAnd, object, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXTransform3DAnd, object, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXColorAnd, object, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXStringNameAnd, object, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXNodePathAnd, object, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXRIDAnd, object, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXObjectAnd, object, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXCallableAnd, object, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXSignalAnd, object, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXDictionaryAnd, object, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXArrayAnd, object, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedByteArrayAnd, object, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedInt32ArrayAnd, object, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedInt64ArrayAnd, object, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedFloat32ArrayAnd, object, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedFloat64ArrayAnd, object, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedStringArrayAnd, object, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedVector2ArrayAnd, object, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedVector3ArrayAnd, object, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXPackedColorArrayAnd, object, packed_color_array, _operate_and)

// callable
OP_EVALUATOR(OperatorEvaluatorCallableXNilAnd, callable, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXBoolAnd, callable, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXIntAnd, callable, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXFloatAnd, callable, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXStringAnd, callable, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXVector2And, callable, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXVector2iAnd, callable, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXRect2And, callable, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXRect2iAnd, callable, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXVector3And, callable, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXVector3iAnd, callable, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXTransform2DAnd, callable, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPlaneAnd, callable, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXQuaternionAnd, callable, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXAABBAnd, callable, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXBasisAnd, callable, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXTransform3DAnd, callable, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXColorAnd, callable, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXStringNameAnd, callable, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXNodePathAnd, callable, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXRIDAnd, callable, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXObjectAnd, callable, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXCallableAnd, callable, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXSignalAnd, callable, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXDictionaryAnd, callable, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXArrayAnd, callable, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedByteArrayAnd, callable, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedInt32ArrayAnd, callable, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedInt64ArrayAnd, callable, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedFloat32ArrayAnd, callable, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedFloat64ArrayAnd, callable, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedStringArrayAnd, callable, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedVector2ArrayAnd, callable, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedVector3ArrayAnd, callable, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorCallableXPackedColorArrayAnd, callable, packed_color_array, _operate_and)

// signal
OP_EVALUATOR(OperatorEvaluatorSignalXNilAnd, signal, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXBoolAnd, signal, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXIntAnd, signal, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXFloatAnd, signal, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXStringAnd, signal, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXVector2And, signal, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXVector2iAnd, signal, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXRect2And, signal, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXRect2iAnd, signal, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXVector3And, signal, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXVector3iAnd, signal, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXTransform2DAnd, signal, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPlaneAnd, signal, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXQuaternionAnd, signal, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXAABBAnd, signal, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXBasisAnd, signal, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXTransform3DAnd, signal, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXColorAnd, signal, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXStringNameAnd, signal, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXNodePathAnd, signal, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXRIDAnd, signal, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXObjectAnd, signal, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXCallableAnd, signal, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXSignalAnd, signal, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXDictionaryAnd, signal, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXArrayAnd, signal, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedByteArrayAnd, signal, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedInt32ArrayAnd, signal, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedInt64ArrayAnd, signal, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedFloat32ArrayAnd, signal, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedFloat64ArrayAnd, signal, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedStringArrayAnd, signal, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedVector2ArrayAnd, signal, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedVector3ArrayAnd, signal, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorSignalXPackedColorArrayAnd, signal, packed_color_array, _operate_and)

// dictionary
OP_EVALUATOR(OperatorEvaluatorDictionaryXNilAnd, dictionary, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXBoolAnd, dictionary, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXIntAnd, dictionary, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXFloatAnd, dictionary, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXStringAnd, dictionary, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXVector2And, dictionary, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXVector2iAnd, dictionary, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXRect2And, dictionary, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXRect2iAnd, dictionary, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXVector3And, dictionary, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXVector3iAnd, dictionary, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXTransform2DAnd, dictionary, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPlaneAnd, dictionary, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXQuaternionAnd, dictionary, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXAABBAnd, dictionary, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXBasisAnd, dictionary, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXTransform3DAnd, dictionary, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXColorAnd, dictionary, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXStringNameAnd, dictionary, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXNodePathAnd, dictionary, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXRIDAnd, dictionary, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXObjectAnd, dictionary, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXCallableAnd, dictionary, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXSignalAnd, dictionary, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXDictionaryAnd, dictionary, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXArrayAnd, dictionary, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedByteArrayAnd, dictionary, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedInt32ArrayAnd, dictionary, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedInt64ArrayAnd, dictionary, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedFloat32ArrayAnd, dictionary, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedFloat64ArrayAnd, dictionary, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedStringArrayAnd, dictionary, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedVector2ArrayAnd, dictionary, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedVector3ArrayAnd, dictionary, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorDictionaryXPackedColorArrayAnd, dictionary, packed_color_array, _operate_and)

// array
OP_EVALUATOR(OperatorEvaluatorArrayXNilAnd, array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXBoolAnd, array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXIntAnd, array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXFloatAnd, array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXStringAnd, array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXVector2And, array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXVector2iAnd, array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXRect2And, array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXRect2iAnd, array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXVector3And, array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXVector3iAnd, array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXTransform2DAnd, array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPlaneAnd, array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXQuaternionAnd, array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXAABBAnd, array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXBasisAnd, array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXTransform3DAnd, array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXColorAnd, array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXStringNameAnd, array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXNodePathAnd, array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXRIDAnd, array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXObjectAnd, array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXCallableAnd, array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXSignalAnd, array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXDictionaryAnd, array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXArrayAnd, array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedByteArrayAnd, array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedInt32ArrayAnd, array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedInt64ArrayAnd, array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedFloat32ArrayAnd, array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedFloat64ArrayAnd, array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedStringArrayAnd, array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedVector2ArrayAnd, array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedVector3ArrayAnd, array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorArrayXPackedColorArrayAnd, array, packed_color_array, _operate_and)

// packed_byte_array
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXNilAnd, packed_byte_array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXBoolAnd, packed_byte_array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXIntAnd, packed_byte_array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXFloatAnd, packed_byte_array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXStringAnd, packed_byte_array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXVector2And, packed_byte_array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXVector2iAnd, packed_byte_array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXRect2And, packed_byte_array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXRect2iAnd, packed_byte_array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXVector3And, packed_byte_array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXVector3iAnd, packed_byte_array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXTransform2DAnd, packed_byte_array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPlaneAnd, packed_byte_array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXQuaternionAnd, packed_byte_array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXAABBAnd, packed_byte_array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXBasisAnd, packed_byte_array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXTransform3DAnd, packed_byte_array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXColorAnd, packed_byte_array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXStringNameAnd, packed_byte_array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXNodePathAnd, packed_byte_array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXRIDAnd, packed_byte_array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXObjectAnd, packed_byte_array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXCallableAnd, packed_byte_array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXSignalAnd, packed_byte_array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXDictionaryAnd, packed_byte_array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXArrayAnd, packed_byte_array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedByteArrayAnd, packed_byte_array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedInt32ArrayAnd, packed_byte_array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedInt64ArrayAnd, packed_byte_array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedFloat32ArrayAnd, packed_byte_array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedFloat64ArrayAnd, packed_byte_array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedStringArrayAnd, packed_byte_array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedVector2ArrayAnd, packed_byte_array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedVector3ArrayAnd, packed_byte_array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedByteArrayXPackedColorArrayAnd, packed_byte_array, packed_color_array, _operate_and)

// packed_int32_array
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXNilAnd, packed_int32_array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXBoolAnd, packed_int32_array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXIntAnd, packed_int32_array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXFloatAnd, packed_int32_array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXStringAnd, packed_int32_array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXVector2And, packed_int32_array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXVector2iAnd, packed_int32_array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXRect2And, packed_int32_array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXRect2iAnd, packed_int32_array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXVector3And, packed_int32_array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXVector3iAnd, packed_int32_array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXTransform2DAnd, packed_int32_array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPlaneAnd, packed_int32_array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXQuaternionAnd, packed_int32_array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXAABBAnd, packed_int32_array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXBasisAnd, packed_int32_array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXTransform3DAnd, packed_int32_array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXColorAnd, packed_int32_array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXStringNameAnd, packed_int32_array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXNodePathAnd, packed_int32_array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXRIDAnd, packed_int32_array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXObjectAnd, packed_int32_array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXCallableAnd, packed_int32_array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXSignalAnd, packed_int32_array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXDictionaryAnd, packed_int32_array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXArrayAnd, packed_int32_array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedByteArrayAnd, packed_int32_array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedInt32ArrayAnd, packed_int32_array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedInt64ArrayAnd, packed_int32_array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedFloat32ArrayAnd, packed_int32_array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedFloat64ArrayAnd, packed_int32_array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedStringArrayAnd, packed_int32_array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedVector2ArrayAnd, packed_int32_array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedVector3ArrayAnd, packed_int32_array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt32ArrayXPackedColorArrayAnd, packed_int32_array, packed_color_array, _operate_and)

// packed_int64_array
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXNilAnd, packed_int64_array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXBoolAnd, packed_int64_array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXIntAnd, packed_int64_array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXFloatAnd, packed_int64_array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXStringAnd, packed_int64_array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXVector2And, packed_int64_array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXVector2iAnd, packed_int64_array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXRect2And, packed_int64_array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXRect2iAnd, packed_int64_array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXVector3And, packed_int64_array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXVector3iAnd, packed_int64_array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXTransform2DAnd, packed_int64_array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPlaneAnd, packed_int64_array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXQuaternionAnd, packed_int64_array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXAABBAnd, packed_int64_array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXBasisAnd, packed_int64_array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXTransform3DAnd, packed_int64_array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXColorAnd, packed_int64_array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXStringNameAnd, packed_int64_array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXNodePathAnd, packed_int64_array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXRIDAnd, packed_int64_array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXObjectAnd, packed_int64_array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXCallableAnd, packed_int64_array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXSignalAnd, packed_int64_array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXDictionaryAnd, packed_int64_array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXArrayAnd, packed_int64_array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedByteArrayAnd, packed_int64_array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedInt32ArrayAnd, packed_int64_array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedInt64ArrayAnd, packed_int64_array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedFloat32ArrayAnd, packed_int64_array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedFloat64ArrayAnd, packed_int64_array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedStringArrayAnd, packed_int64_array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedVector2ArrayAnd, packed_int64_array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedVector3ArrayAnd, packed_int64_array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedInt64ArrayXPackedColorArrayAnd, packed_int64_array, packed_color_array, _operate_and)

// packed_float32_array
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXNilAnd, packed_float32_array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXBoolAnd, packed_float32_array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXIntAnd, packed_float32_array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXFloatAnd, packed_float32_array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXStringAnd, packed_float32_array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXVector2And, packed_float32_array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXVector2iAnd, packed_float32_array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXRect2And, packed_float32_array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXRect2iAnd, packed_float32_array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXVector3And, packed_float32_array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXVector3iAnd, packed_float32_array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXTransform2DAnd, packed_float32_array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPlaneAnd, packed_float32_array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXQuaternionAnd, packed_float32_array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXAABBAnd, packed_float32_array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXBasisAnd, packed_float32_array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXTransform3DAnd, packed_float32_array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXColorAnd, packed_float32_array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXStringNameAnd, packed_float32_array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXNodePathAnd, packed_float32_array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXRIDAnd, packed_float32_array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXObjectAnd, packed_float32_array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXCallableAnd, packed_float32_array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXSignalAnd, packed_float32_array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXDictionaryAnd, packed_float32_array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXArrayAnd, packed_float32_array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedByteArrayAnd, packed_float32_array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedInt32ArrayAnd, packed_float32_array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedInt64ArrayAnd, packed_float32_array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedFloat32ArrayAnd, packed_float32_array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedFloat64ArrayAnd, packed_float32_array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedStringArrayAnd, packed_float32_array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedVector2ArrayAnd, packed_float32_array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedVector3ArrayAnd, packed_float32_array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat32ArrayXPackedColorArrayAnd, packed_float32_array, packed_color_array, _operate_and)

// packed_float64_array
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXNilAnd, packed_float64_array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXBoolAnd, packed_float64_array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXIntAnd, packed_float64_array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXFloatAnd, packed_float64_array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXStringAnd, packed_float64_array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXVector2And, packed_float64_array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXVector2iAnd, packed_float64_array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXRect2And, packed_float64_array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXRect2iAnd, packed_float64_array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXVector3And, packed_float64_array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXVector3iAnd, packed_float64_array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXTransform2DAnd, packed_float64_array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPlaneAnd, packed_float64_array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXQuaternionAnd, packed_float64_array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXAABBAnd, packed_float64_array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXBasisAnd, packed_float64_array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXTransform3DAnd, packed_float64_array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXColorAnd, packed_float64_array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXStringNameAnd, packed_float64_array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXNodePathAnd, packed_float64_array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXRIDAnd, packed_float64_array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXObjectAnd, packed_float64_array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXCallableAnd, packed_float64_array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXSignalAnd, packed_float64_array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXDictionaryAnd, packed_float64_array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXArrayAnd, packed_float64_array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedByteArrayAnd, packed_float64_array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedInt32ArrayAnd, packed_float64_array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedInt64ArrayAnd, packed_float64_array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedFloat32ArrayAnd, packed_float64_array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedFloat64ArrayAnd, packed_float64_array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedStringArrayAnd, packed_float64_array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedVector2ArrayAnd, packed_float64_array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedVector3ArrayAnd, packed_float64_array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedFloat64ArrayXPackedColorArrayAnd, packed_float64_array, packed_color_array, _operate_and)

// packed_string_array
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXNilAnd, packed_string_array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXBoolAnd, packed_string_array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXIntAnd, packed_string_array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXFloatAnd, packed_string_array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXStringAnd, packed_string_array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXVector2And, packed_string_array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXVector2iAnd, packed_string_array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXRect2And, packed_string_array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXRect2iAnd, packed_string_array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXVector3And, packed_string_array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXVector3iAnd, packed_string_array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXTransform2DAnd, packed_string_array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPlaneAnd, packed_string_array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXQuaternionAnd, packed_string_array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXAABBAnd, packed_string_array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXBasisAnd, packed_string_array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXTransform3DAnd, packed_string_array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXColorAnd, packed_string_array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXStringNameAnd, packed_string_array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXNodePathAnd, packed_string_array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXRIDAnd, packed_string_array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXObjectAnd, packed_string_array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXCallableAnd, packed_string_array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXSignalAnd, packed_string_array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXDictionaryAnd, packed_string_array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXArrayAnd, packed_string_array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedByteArrayAnd, packed_string_array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedInt32ArrayAnd, packed_string_array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedInt64ArrayAnd, packed_string_array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedFloat32ArrayAnd, packed_string_array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedFloat64ArrayAnd, packed_string_array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedStringArrayAnd, packed_string_array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedVector2ArrayAnd, packed_string_array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedVector3ArrayAnd, packed_string_array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedStringArrayXPackedColorArrayAnd, packed_string_array, packed_color_array, _operate_and)

// packed_vector2_array
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXNilAnd, packed_vector2_array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXBoolAnd, packed_vector2_array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXIntAnd, packed_vector2_array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXFloatAnd, packed_vector2_array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXStringAnd, packed_vector2_array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXVector2And, packed_vector2_array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXVector2iAnd, packed_vector2_array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXRect2And, packed_vector2_array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXRect2iAnd, packed_vector2_array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXVector3And, packed_vector2_array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXVector3iAnd, packed_vector2_array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXTransform2DAnd, packed_vector2_array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPlaneAnd, packed_vector2_array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXQuaternionAnd, packed_vector2_array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXAABBAnd, packed_vector2_array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXBasisAnd, packed_vector2_array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXTransform3DAnd, packed_vector2_array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXColorAnd, packed_vector2_array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXStringNameAnd, packed_vector2_array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXNodePathAnd, packed_vector2_array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXRIDAnd, packed_vector2_array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXObjectAnd, packed_vector2_array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXCallableAnd, packed_vector2_array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXSignalAnd, packed_vector2_array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXDictionaryAnd, packed_vector2_array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXArrayAnd, packed_vector2_array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedByteArrayAnd, packed_vector2_array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedInt32ArrayAnd, packed_vector2_array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedInt64ArrayAnd, packed_vector2_array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedFloat32ArrayAnd, packed_vector2_array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedFloat64ArrayAnd, packed_vector2_array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedStringArrayAnd, packed_vector2_array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedVector2ArrayAnd, packed_vector2_array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedVector3ArrayAnd, packed_vector2_array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector2ArrayXPackedColorArrayAnd, packed_vector2_array, packed_color_array, _operate_and)

// packed_vector3_array
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXNilAnd, packed_vector3_array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXBoolAnd, packed_vector3_array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXIntAnd, packed_vector3_array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXFloatAnd, packed_vector3_array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXStringAnd, packed_vector3_array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXVector2And, packed_vector3_array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXVector2iAnd, packed_vector3_array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXRect2And, packed_vector3_array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXRect2iAnd, packed_vector3_array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXVector3And, packed_vector3_array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXVector3iAnd, packed_vector3_array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXTransform2DAnd, packed_vector3_array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPlaneAnd, packed_vector3_array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXQuaternionAnd, packed_vector3_array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXAABBAnd, packed_vector3_array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXBasisAnd, packed_vector3_array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXTransform3DAnd, packed_vector3_array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXColorAnd, packed_vector3_array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXStringNameAnd, packed_vector3_array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXNodePathAnd, packed_vector3_array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXRIDAnd, packed_vector3_array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXObjectAnd, packed_vector3_array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXCallableAnd, packed_vector3_array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXSignalAnd, packed_vector3_array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXDictionaryAnd, packed_vector3_array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXArrayAnd, packed_vector3_array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedByteArrayAnd, packed_vector3_array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedInt32ArrayAnd, packed_vector3_array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedInt64ArrayAnd, packed_vector3_array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedFloat32ArrayAnd, packed_vector3_array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedFloat64ArrayAnd, packed_vector3_array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedStringArrayAnd, packed_vector3_array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedVector2ArrayAnd, packed_vector3_array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedVector3ArrayAnd, packed_vector3_array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedVector3ArrayXPackedColorArrayAnd, packed_vector3_array, packed_color_array, _operate_and)

// packed_color_array
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXNilAnd, packed_color_array, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXBoolAnd, packed_color_array, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXIntAnd, packed_color_array, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXFloatAnd, packed_color_array, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXStringAnd, packed_color_array, string, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXVector2And, packed_color_array, vector2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXVector2iAnd, packed_color_array, vector2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXRect2And, packed_color_array, rect2, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXRect2iAnd, packed_color_array, rect2i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXVector3And, packed_color_array, vector3, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXVector3iAnd, packed_color_array, vector3i, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXTransform2DAnd, packed_color_array, transform2d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPlaneAnd, packed_color_array, plane, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXQuaternionAnd, packed_color_array, quaternion, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXAABBAnd, packed_color_array, aabb, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXBasisAnd, packed_color_array, basis, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXTransform3DAnd, packed_color_array, transform3d, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXColorAnd, packed_color_array, color, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXStringNameAnd, packed_color_array, string_name, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXNodePathAnd, packed_color_array, node_path, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXRIDAnd, packed_color_array, rid, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXObjectAnd, packed_color_array, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXCallableAnd, packed_color_array, callable, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXSignalAnd, packed_color_array, signal, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXDictionaryAnd, packed_color_array, dictionary, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXArrayAnd, packed_color_array, array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedByteArrayAnd, packed_color_array, packed_byte_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedInt32ArrayAnd, packed_color_array, packed_int32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedInt64ArrayAnd, packed_color_array, packed_int64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedFloat32ArrayAnd, packed_color_array, packed_float32_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedFloat64ArrayAnd, packed_color_array, packed_float64_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedStringArrayAnd, packed_color_array, packed_string_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedVector2ArrayAnd, packed_color_array, packed_vector2_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedVector3ArrayAnd, packed_color_array, packed_vector3_array, _operate_and)
OP_EVALUATOR(OperatorEvaluatorPackedColorArrayXPackedColorArrayAnd, packed_color_array, packed_color_array, _operate_and)

// XOR

// nil
OP_EVALUATOR(OperatorEvaluatorNilXBoolXor, nil, bool, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorBoolXNilXor, bool, nil, _operate_xor)

OP_EVALUATOR(OperatorEvaluatorNilXIntXor, nil, int, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorIntXNilXor, int, nil, _operate_xor)

OP_EVALUATOR(OperatorEvaluatorNilXFloatXor, nil, float, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorFloatXNilXor, float, nil, _operate_xor)

OP_EVALUATOR(OperatorEvaluatorObjectXNilXor, object, nil, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorNilXObjectXor, nil, object, _operate_xor)

// bool
OP_EVALUATOR(OperatorEvaluatorBoolXBoolXor, bool, bool, _operate_xor)

OP_EVALUATOR(OperatorEvaluatorBoolXIntXor, bool, int, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorIntXBoolXor, int, bool, _operate_xor)

OP_EVALUATOR(OperatorEvaluatorBoolXFloatXor, bool, float, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorFloatXBoolXor, float, bool, _operate_xor)

OP_EVALUATOR(OperatorEvaluatorBoolXObjectXor, bool, object, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorObjectXBoolXor, object, bool, _operate_xor)

// int
OP_EVALUATOR(OperatorEvaluatorIntXIntXor, int, int, _operate_xor)

OP_EVALUATOR(OperatorEvaluatorIntXFloatXor, int, float, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorFloatXIntXor, float, int, _operate_xor)

OP_EVALUATOR(OperatorEvaluatorIntXObjectXor, int, object, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorObjectXIntXor, object, int, _operate_xor)

// float
OP_EVALUATOR(OperatorEvaluatorFloatXFloatXor, float, float, _operate_xor)

OP_EVALUATOR(OperatorEvaluatorFloatXObjectXor, float, object, _operate_xor)
OP_EVALUATOR(OperatorEvaluatorObjectXFloatXor, object, float, _operate_xor)

// object
OP_EVALUATOR(OperatorEvaluatorObjectXObjectXor, object, object, _operate_xor)

class OperatorEvaluatorNotBool {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = !*VariantGetInternalPtr<bool>::get_ptr(&p_left);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = !*VariantGetInternalPtr<bool>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(!PtrToArg<bool>::convert(left), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorNotInt {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = !*VariantGetInternalPtr<int64_t>::get_ptr(&p_left);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = !*VariantGetInternalPtr<int64_t>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(!PtrToArg<int64_t>::convert(left), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorNotFloat {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = !*VariantGetInternalPtr<double>::get_ptr(&p_left);
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = !*VariantGetInternalPtr<double>::get_ptr(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(!PtrToArg<double>::convert(left), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

class OperatorEvaluatorNotObject {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = p_left.get_validated_object() == nullptr;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = left->get_validated_object() == nullptr;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) == nullptr, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
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
