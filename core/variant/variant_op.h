/**************************************************************************/
/*  variant_op.h                                                          */
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

#pragma once

#include "variant.h"

#include "core/debugger/engine_debugger.h"
#include "core/object/class_db.h"

template <typename Evaluator>
class CommonEvaluate {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		VariantTypeChanger<typename Evaluator::ReturnType>::change(r_ret);
		Evaluator::validated_evaluate(&p_left, &p_right, r_ret);
		r_valid = true;
	}
	static Variant::Type get_return_type() { return GetTypeInfo<typename Evaluator::ReturnType>::VARIANT_TYPE; }
};

template <typename R, typename A, typename B>
class OperatorEvaluatorAdd : public CommonEvaluate<OperatorEvaluatorAdd<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) + VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) + PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorSub : public CommonEvaluate<OperatorEvaluatorSub<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) - VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) - PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorMul : public CommonEvaluate<OperatorEvaluatorMul<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) * VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) * PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorPow : public CommonEvaluate<OperatorEvaluatorPow<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = R(Math::pow((double)VariantInternalAccessor<A>::get(left), (double)VariantInternalAccessor<B>::get(right)));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(R(Math::pow((double)PtrToArg<A>::convert(left), (double)PtrToArg<B>::convert(right))), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorXForm : public CommonEvaluate<OperatorEvaluatorXForm<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left).xform(VariantInternalAccessor<B>::get(right));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left).xform(PtrToArg<B>::convert(right)), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorXFormInv : public CommonEvaluate<OperatorEvaluatorXFormInv<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<B>::get(right).xform_inv(VariantInternalAccessor<A>::get(left));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<B>::convert(right).xform_inv(PtrToArg<A>::convert(left)), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorDiv : public CommonEvaluate<OperatorEvaluatorDiv<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) / VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) / PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorDivNZ {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = VariantInternalAccessor<A>::get(&p_left);
		const B &b = VariantInternalAccessor<B>::get(&p_right);
		if (b == 0) {
			r_valid = false;
			*r_ret = "Division by zero error";
			return;
		}
		*r_ret = a / b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) / VariantInternalAccessor<B>::get(right);
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
		const Vector2i &a = VariantInternalAccessor<Vector2i>::get(&p_left);
		const Vector2i &b = VariantInternalAccessor<Vector2i>::get(&p_right);
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
		VariantInternalAccessor<Vector2i>::get(r_ret) = VariantInternalAccessor<Vector2i>::get(left) / VariantInternalAccessor<Vector2i>::get(right);
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
		const Vector3i &a = VariantInternalAccessor<Vector3i>::get(&p_left);
		const Vector3i &b = VariantInternalAccessor<Vector3i>::get(&p_right);
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
		VariantInternalAccessor<Vector3i>::get(r_ret) = VariantInternalAccessor<Vector3i>::get(left) / VariantInternalAccessor<Vector3i>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector3i>::encode(PtrToArg<Vector3i>::convert(left) / PtrToArg<Vector3i>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector3i>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorDivNZ<Vector4i, Vector4i, Vector4i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector4i &a = VariantInternalAccessor<Vector4i>::get(&p_left);
		const Vector4i &b = VariantInternalAccessor<Vector4i>::get(&p_right);
		if (unlikely(b.x == 0 || b.y == 0 || b.z == 0 || b.w == 0)) {
			r_valid = false;
			*r_ret = "Division by zero error";
			return;
		}
		*r_ret = a / b;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Vector4i>::change(r_ret);
		VariantInternalAccessor<Vector4i>::get(r_ret) = VariantInternalAccessor<Vector4i>::get(left) / VariantInternalAccessor<Vector4i>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector4i>::encode(PtrToArg<Vector4i>::convert(left) / PtrToArg<Vector4i>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector4i>::VARIANT_TYPE; }
};

template <typename R, typename A, typename B>
class OperatorEvaluatorMod : public CommonEvaluate<OperatorEvaluatorMod<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) % VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) % PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorModNZ {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = VariantInternalAccessor<A>::get(&p_left);
		const B &b = VariantInternalAccessor<B>::get(&p_right);
		if (b == 0) {
			r_valid = false;
			*r_ret = "Modulo by zero error";
			return;
		}
		*r_ret = a % b;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) % VariantInternalAccessor<B>::get(right);
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
		const Vector2i &a = VariantInternalAccessor<Vector2i>::get(&p_left);
		const Vector2i &b = VariantInternalAccessor<Vector2i>::get(&p_right);
		if (unlikely(b.x == 0 || b.y == 0)) {
			r_valid = false;
			*r_ret = "Modulo by zero error";
			return;
		}
		*r_ret = a % b;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Vector2i>::change(r_ret);
		VariantInternalAccessor<Vector2i>::get(r_ret) = VariantInternalAccessor<Vector2i>::get(left) % VariantInternalAccessor<Vector2i>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector2i>::encode(PtrToArg<Vector2i>::convert(left) % PtrToArg<Vector2i>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector2i>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorModNZ<Vector3i, Vector3i, Vector3i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector3i &a = VariantInternalAccessor<Vector3i>::get(&p_left);
		const Vector3i &b = VariantInternalAccessor<Vector3i>::get(&p_right);
		if (unlikely(b.x == 0 || b.y == 0 || b.z == 0)) {
			r_valid = false;
			*r_ret = "Modulo by zero error";
			return;
		}
		*r_ret = a % b;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Vector3i>::change(r_ret);
		VariantInternalAccessor<Vector3i>::get(r_ret) = VariantInternalAccessor<Vector3i>::get(left) % VariantInternalAccessor<Vector3i>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector3i>::encode(PtrToArg<Vector3i>::convert(left) % PtrToArg<Vector3i>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector3i>::VARIANT_TYPE; }
};

template <>
class OperatorEvaluatorModNZ<Vector4i, Vector4i, Vector4i> {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const Vector4i &a = VariantInternalAccessor<Vector4i>::get(&p_left);
		const Vector4i &b = VariantInternalAccessor<Vector4i>::get(&p_right);
		if (unlikely(b.x == 0 || b.y == 0 || b.z == 0 || b.w == 0)) {
			r_valid = false;
			*r_ret = "Modulo by zero error";
			return;
		}
		*r_ret = a % b;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Vector4i>::change(r_ret);
		VariantInternalAccessor<Vector4i>::get(r_ret) = VariantInternalAccessor<Vector4i>::get(left) % VariantInternalAccessor<Vector4i>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<Vector4i>::encode(PtrToArg<Vector4i>::convert(left) % PtrToArg<Vector4i>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<Vector4i>::VARIANT_TYPE; }
};

template <typename R, typename A>
class OperatorEvaluatorNeg : public CommonEvaluate<OperatorEvaluatorNeg<R, A>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = -VariantInternalAccessor<A>::get(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(-PtrToArg<A>::convert(left), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A>
class OperatorEvaluatorPos : public CommonEvaluate<OperatorEvaluatorPos<R, A>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorShiftLeft {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = VariantInternalAccessor<A>::get(&p_left);
		const B &b = VariantInternalAccessor<B>::get(&p_right);

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
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) << VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) << PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <typename R, typename A, typename B>
class OperatorEvaluatorShiftRight {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = VariantInternalAccessor<A>::get(&p_left);
		const B &b = VariantInternalAccessor<B>::get(&p_right);

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
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) >> VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) >> PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <typename R, typename A, typename B>
class OperatorEvaluatorBitOr : public CommonEvaluate<OperatorEvaluatorBitOr<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) | VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) | PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorBitAnd : public CommonEvaluate<OperatorEvaluatorBitAnd<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) & VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) & PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A, typename B>
class OperatorEvaluatorBitXor : public CommonEvaluate<OperatorEvaluatorBitXor<R, A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = VariantInternalAccessor<A>::get(left) ^ VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) ^ PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = R;
};

template <typename R, typename A>
class OperatorEvaluatorBitNeg : public CommonEvaluate<OperatorEvaluatorBitNeg<R, A>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<R>::get(r_ret) = ~VariantInternalAccessor<A>::get(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(~PtrToArg<A>::convert(left), r_ret);
	}
	using ReturnType = R;
};

template <typename A, typename B>
class OperatorEvaluatorEqual : public CommonEvaluate<OperatorEvaluatorEqual<A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = VariantInternalAccessor<A>::get(left) == VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) == PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorEqualObject : public CommonEvaluate<OperatorEvaluatorEqualObject> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Object *a = left->get_validated_object();
		const Object *b = right->get_validated_object();
		VariantInternalAccessor<bool>::get(r_ret) = a == b;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) == PtrToArg<Object *>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorEqualObjectNil : public CommonEvaluate<OperatorEvaluatorEqualObjectNil> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Object *a = left->get_validated_object();
		VariantInternalAccessor<bool>::get(r_ret) = a == nullptr;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) == nullptr, r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorEqualNilObject : public CommonEvaluate<OperatorEvaluatorEqualNilObject> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Object *b = right->get_validated_object();
		VariantInternalAccessor<bool>::get(r_ret) = nullptr == b;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(nullptr == PtrToArg<Object *>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

template <typename A, typename B>
class OperatorEvaluatorNotEqual : public CommonEvaluate<OperatorEvaluatorNotEqual<A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = VariantInternalAccessor<A>::get(left) != VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) != PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorNotEqualObject : public CommonEvaluate<OperatorEvaluatorNotEqualObject> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *a = left->get_validated_object();
		Object *b = right->get_validated_object();
		VariantInternalAccessor<bool>::get(r_ret) = a != b;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) != PtrToArg<Object *>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorNotEqualObjectNil : public CommonEvaluate<OperatorEvaluatorNotEqualObjectNil> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *a = left->get_validated_object();
		VariantInternalAccessor<bool>::get(r_ret) = a != nullptr;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) != nullptr, r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorNotEqualNilObject : public CommonEvaluate<OperatorEvaluatorNotEqualNilObject> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *b = right->get_validated_object();
		VariantInternalAccessor<bool>::get(r_ret) = nullptr != b;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(nullptr != PtrToArg<Object *>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

template <typename A, typename B>
class OperatorEvaluatorLess : public CommonEvaluate<OperatorEvaluatorLess<A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = VariantInternalAccessor<A>::get(left) < VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) < PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

template <typename A, typename B>
class OperatorEvaluatorLessEqual : public CommonEvaluate<OperatorEvaluatorLessEqual<A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = VariantInternalAccessor<A>::get(left) <= VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) <= PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

template <typename A, typename B>
class OperatorEvaluatorGreater : public CommonEvaluate<OperatorEvaluatorGreater<A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = VariantInternalAccessor<A>::get(left) > VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) > PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

template <typename A, typename B>
class OperatorEvaluatorGreaterEqual : public CommonEvaluate<OperatorEvaluatorGreaterEqual<A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = VariantInternalAccessor<A>::get(left) >= VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) >= PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

template <typename A, typename B>
class OperatorEvaluatorAnd : public CommonEvaluate<OperatorEvaluatorAnd<A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = VariantInternalAccessor<A>::get(left) && VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) && PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

template <typename A, typename B>
class OperatorEvaluatorOr : public CommonEvaluate<OperatorEvaluatorOr<A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = VariantInternalAccessor<A>::get(left) || VariantInternalAccessor<B>::get(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) || PtrToArg<B>::convert(right), r_ret);
	}
	using ReturnType = bool;
};

#define XOR_OP(m_a, m_b) (((m_a) || (m_b)) && !((m_a) && (m_b)))
template <typename A, typename B>
class OperatorEvaluatorXor : public CommonEvaluate<OperatorEvaluatorXor<A, B>> {
public:
	_FORCE_INLINE_ static bool xor_op(const A &a, const B &b) {
		return ((a) || (b)) && !((a) && (b));
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = xor_op(VariantInternalAccessor<A>::get(left), VariantInternalAccessor<B>::get(right));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(xor_op(PtrToArg<A>::convert(left), PtrToArg<B>::convert(right)), r_ret);
	}
	using ReturnType = bool;
};

template <typename A>
class OperatorEvaluatorNot : public CommonEvaluate<OperatorEvaluatorNot<A>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = VariantInternalAccessor<A>::get(left) == A();
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<A>::convert(left) == A(), r_ret);
	}
	using ReturnType = bool;
};

//// CUSTOM ////

class OperatorEvaluatorAddArray : public CommonEvaluate<OperatorEvaluatorAddArray> {
public:
	_FORCE_INLINE_ static void _add_arrays(Array &sum, const Array &array_a, const Array &array_b) {
		int asize = array_a.size();
		int bsize = array_b.size();

		if (array_a.is_typed() && array_a.is_same_typed(array_b)) {
			sum.set_typed(array_a.get_typed_builtin(), array_a.get_typed_class_name(), array_a.get_typed_script());
		}

		sum.resize(asize + bsize);
		for (int i = 0; i < asize; i++) {
			sum[i] = array_a[i];
		}
		for (int i = 0; i < bsize; i++) {
			sum[i + asize] = array_b[i];
		}
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		*r_ret = Array();
		_add_arrays(VariantInternalAccessor<Array>::get(r_ret), VariantInternalAccessor<Array>::get(left), VariantInternalAccessor<Array>::get(right));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		Array ret;
		_add_arrays(ret, PtrToArg<Array>::convert(left), PtrToArg<Array>::convert(right));
		PtrToArg<Array>::encode(ret, r_ret);
	}
	using ReturnType = Array;
};

template <typename T>
class OperatorEvaluatorAppendArray : public CommonEvaluate<OperatorEvaluatorAppendArray<T>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<Vector<T>>::get(r_ret) = VariantInternalAccessor<Vector<T>>::get(left);
		VariantInternalAccessor<Vector<T>>::get(r_ret).append_array(VariantInternalAccessor<Vector<T>>::get(right));
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		Vector<T> sum = PtrToArg<Vector<T>>::convert(left);
		sum.append_array(PtrToArg<Vector<T>>::convert(right));
		PtrToArg<Vector<T>>::encode(sum, r_ret);
	}
	using ReturnType = Vector<T>;
};

template <typename Left, typename Right>
class OperatorEvaluatorStringConcat : public CommonEvaluate<OperatorEvaluatorStringConcat<Left, Right>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const String a(VariantInternalAccessor<Left>::get(left));
		const String b(VariantInternalAccessor<Right>::get(right));
		VariantInternalAccessor<String>::get(r_ret) = a + b;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		const String a(PtrToArg<Left>::convert(left));
		const String b(PtrToArg<Right>::convert(right));
		PtrToArg<String>::encode(a + b, r_ret);
	}
	using ReturnType = String;
};

template <typename S, typename T>
class OperatorEvaluatorStringFormat;

template <typename S>
class OperatorEvaluatorStringFormat<S, void> {
public:
	_FORCE_INLINE_ static String do_mod(const String &s, bool *r_valid) {
		Array values = { Variant() };
		String a = s.sprintf(values, r_valid);
		if (r_valid) {
			*r_valid = !*r_valid;
		}
		return a;
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = do_mod(VariantInternalAccessor<S>::get(&p_left), &r_valid);
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		bool valid = true;
		String result = do_mod(VariantInternalAccessor<S>::get(left), &valid);
		if (unlikely(!valid)) {
			VariantInternalAccessor<String>::get(r_ret) = VariantInternalAccessor<S>::get(left);
			ERR_FAIL_MSG(vformat("String formatting error: %s.", result));
		}
		VariantInternalAccessor<String>::get(r_ret) = result;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<String>::encode(do_mod(PtrToArg<S>::convert(left), nullptr), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::STRING; }
};

template <typename S>
class OperatorEvaluatorStringFormat<S, Array> {
public:
	_FORCE_INLINE_ static String do_mod(const String &s, const Array &p_values, bool *r_valid) {
		String a = s.sprintf(p_values, r_valid);
		if (r_valid) {
			*r_valid = !*r_valid;
		}
		return a;
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = do_mod(VariantInternalAccessor<S>::get(&p_left), VariantInternalAccessor<Array>::get(&p_right), &r_valid);
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		bool valid = true;
		String result = do_mod(VariantInternalAccessor<S>::get(left), VariantInternalAccessor<Array>::get(right), &valid);
		if (unlikely(!valid)) {
			VariantInternalAccessor<String>::get(r_ret) = VariantInternalAccessor<S>::get(left);
			ERR_FAIL_MSG(vformat("String formatting error: %s.", result));
		}
		VariantInternalAccessor<String>::get(r_ret) = result;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<String>::encode(do_mod(PtrToArg<S>::convert(left), PtrToArg<Array>::convert(right), nullptr), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::STRING; }
};

template <typename S>
class OperatorEvaluatorStringFormat<S, Object> {
public:
	_FORCE_INLINE_ static String do_mod(const String &s, const Object *p_object, bool *r_valid) {
		Array values = { p_object };
		String a = s.sprintf(values, r_valid);
		if (r_valid) {
			*r_valid = !*r_valid;
		}

		return a;
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = do_mod(VariantInternalAccessor<S>::get(&p_left), p_right.get_validated_object(), &r_valid);
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		bool valid = true;
		String result = do_mod(VariantInternalAccessor<S>::get(left), right->get_validated_object(), &valid);
		if (unlikely(!valid)) {
			VariantInternalAccessor<String>::get(r_ret) = VariantInternalAccessor<S>::get(left);
			ERR_FAIL_MSG(vformat("String formatting error: %s.", result));
		}
		VariantInternalAccessor<String>::get(r_ret) = result;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<String>::encode(do_mod(PtrToArg<S>::convert(left), PtrToArg<Object *>::convert(right), nullptr), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::STRING; }
};

template <typename S, typename T>
class OperatorEvaluatorStringFormat {
public:
	_FORCE_INLINE_ static String do_mod(const String &s, const T &p_value, bool *r_valid) {
		Array values = { p_value };
		String a = s.sprintf(values, r_valid);
		if (r_valid) {
			*r_valid = !*r_valid;
		}
		return a;
	}
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = do_mod(VariantInternalAccessor<S>::get(&p_left), VariantInternalAccessor<T>::get(&p_right), &r_valid);
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		bool valid = true;
		String result = do_mod(VariantInternalAccessor<S>::get(left), VariantInternalAccessor<T>::get(right), &valid);
		if (unlikely(!valid)) {
			VariantInternalAccessor<String>::get(r_ret) = VariantInternalAccessor<S>::get(left);
			ERR_FAIL_MSG(vformat("String formatting error: %s.", result));
		}
		VariantInternalAccessor<String>::get(r_ret) = result;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<String>::encode(do_mod(PtrToArg<S>::convert(left), PtrToArg<T>::convert(right), nullptr), r_ret);
	}
	static Variant::Type get_return_type() { return Variant::STRING; }
};

class OperatorEvaluatorAlwaysTrue : public CommonEvaluate<OperatorEvaluatorAlwaysTrue> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = true;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(true, r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorAlwaysFalse : public CommonEvaluate<OperatorEvaluatorAlwaysFalse> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = false;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(false, r_ret);
	}
	using ReturnType = bool;
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
	return VariantInternalAccessor<bool>::get(p_ptr);
}

_FORCE_INLINE_ static bool _operate_get_int(const Variant *p_ptr) {
	return VariantInternalAccessor<int64_t>::get(p_ptr) != 0;
}

_FORCE_INLINE_ static bool _operate_get_float(const Variant *p_ptr) {
	return VariantInternalAccessor<double>::get(p_ptr) != 0.0;
}

_FORCE_INLINE_ static bool _operate_get_object(const Variant *p_ptr) {
	return p_ptr->get_validated_object() != nullptr;
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

_FORCE_INLINE_ static bool _operate_get_ptr_object(const void *p_ptr) {
	return PtrToArg<Object *>::convert(p_ptr) != nullptr;
}

#define OP_EVALUATOR(m_class_name, m_left, m_right, m_op)                                                                 \
	class m_class_name : public CommonEvaluate<m_class_name> {                                                            \
	public:                                                                                                               \
		static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {                \
			VariantInternalAccessor<bool>::get(r_ret) = m_op(_operate_get_##m_left(left), _operate_get_##m_right(right)); \
		}                                                                                                                 \
                                                                                                                          \
		static void ptr_evaluate(const void *left, const void *right, void *r_ret) {                                      \
			PtrToArg<bool>::encode(m_op(_operate_get_ptr_##m_left(left), _operate_get_ptr_##m_right(right)), r_ret);      \
		}                                                                                                                 \
                                                                                                                          \
		using ReturnType = bool;                                                                                          \
	};

// OR

// nil
OP_EVALUATOR(OperatorEvaluatorNilXBoolOr, nil, bool, _operate_or)
OP_EVALUATOR(OperatorEvaluatorBoolXNilOr, bool, nil, _operate_or)

OP_EVALUATOR(OperatorEvaluatorNilXIntOr, nil, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXNilOr, int, nil, _operate_or)

OP_EVALUATOR(OperatorEvaluatorNilXFloatOr, nil, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXNilOr, float, nil, _operate_or)

OP_EVALUATOR(OperatorEvaluatorObjectXNilOr, object, nil, _operate_or)
OP_EVALUATOR(OperatorEvaluatorNilXObjectOr, nil, object, _operate_or)

// bool
OP_EVALUATOR(OperatorEvaluatorBoolXBoolOr, bool, bool, _operate_or)

OP_EVALUATOR(OperatorEvaluatorBoolXIntOr, bool, int, _operate_or)
OP_EVALUATOR(OperatorEvaluatorIntXBoolOr, int, bool, _operate_or)

OP_EVALUATOR(OperatorEvaluatorBoolXFloatOr, bool, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXBoolOr, float, bool, _operate_or)

OP_EVALUATOR(OperatorEvaluatorBoolXObjectOr, bool, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXBoolOr, object, bool, _operate_or)

// int
OP_EVALUATOR(OperatorEvaluatorIntXIntOr, int, int, _operate_or)

OP_EVALUATOR(OperatorEvaluatorIntXFloatOr, int, float, _operate_or)
OP_EVALUATOR(OperatorEvaluatorFloatXIntOr, float, int, _operate_or)

OP_EVALUATOR(OperatorEvaluatorIntXObjectOr, int, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXIntOr, object, int, _operate_or)

// float
OP_EVALUATOR(OperatorEvaluatorFloatXFloatOr, float, float, _operate_or)

OP_EVALUATOR(OperatorEvaluatorFloatXObjectOr, float, object, _operate_or)
OP_EVALUATOR(OperatorEvaluatorObjectXFloatOr, object, float, _operate_or)

// object
OP_EVALUATOR(OperatorEvaluatorObjectXObjectOr, object, object, _operate_or)

// AND

// nil
OP_EVALUATOR(OperatorEvaluatorNilXBoolAnd, nil, bool, _operate_and)
OP_EVALUATOR(OperatorEvaluatorBoolXNilAnd, bool, nil, _operate_and)

OP_EVALUATOR(OperatorEvaluatorNilXIntAnd, nil, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXNilAnd, int, nil, _operate_and)

OP_EVALUATOR(OperatorEvaluatorNilXFloatAnd, nil, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXNilAnd, float, nil, _operate_and)

OP_EVALUATOR(OperatorEvaluatorObjectXNilAnd, object, nil, _operate_and)
OP_EVALUATOR(OperatorEvaluatorNilXObjectAnd, nil, object, _operate_and)

// bool
OP_EVALUATOR(OperatorEvaluatorBoolXBoolAnd, bool, bool, _operate_and)

OP_EVALUATOR(OperatorEvaluatorBoolXIntAnd, bool, int, _operate_and)
OP_EVALUATOR(OperatorEvaluatorIntXBoolAnd, int, bool, _operate_and)

OP_EVALUATOR(OperatorEvaluatorBoolXFloatAnd, bool, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXBoolAnd, float, bool, _operate_and)

OP_EVALUATOR(OperatorEvaluatorBoolXObjectAnd, bool, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXBoolAnd, object, bool, _operate_and)

// int
OP_EVALUATOR(OperatorEvaluatorIntXIntAnd, int, int, _operate_and)

OP_EVALUATOR(OperatorEvaluatorIntXFloatAnd, int, float, _operate_and)
OP_EVALUATOR(OperatorEvaluatorFloatXIntAnd, float, int, _operate_and)

OP_EVALUATOR(OperatorEvaluatorIntXObjectAnd, int, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXIntAnd, object, int, _operate_and)

// float
OP_EVALUATOR(OperatorEvaluatorFloatXFloatAnd, float, float, _operate_and)

OP_EVALUATOR(OperatorEvaluatorFloatXObjectAnd, float, object, _operate_and)
OP_EVALUATOR(OperatorEvaluatorObjectXFloatAnd, object, float, _operate_and)

// object
OP_EVALUATOR(OperatorEvaluatorObjectXObjectAnd, object, object, _operate_and)

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

class OperatorEvaluatorNotBool : public CommonEvaluate<OperatorEvaluatorNotBool> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = !VariantInternalAccessor<bool>::get(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(!PtrToArg<bool>::convert(left), r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorNotInt : public CommonEvaluate<OperatorEvaluatorNotInt> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = !VariantInternalAccessor<int64_t>::get(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(!PtrToArg<int64_t>::convert(left), r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorNotFloat : public CommonEvaluate<OperatorEvaluatorNotFloat> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = !VariantInternalAccessor<double>::get(left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(!PtrToArg<double>::convert(left), r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorNotObject : public CommonEvaluate<OperatorEvaluatorNotObject> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantInternalAccessor<bool>::get(r_ret) = left->get_validated_object() == nullptr;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) == nullptr, r_ret);
	}
	using ReturnType = bool;
};

////

template <typename Left, typename Right>
class OperatorEvaluatorInStringFind;

template <typename Left>
class OperatorEvaluatorInStringFind<Left, String> : public CommonEvaluate<OperatorEvaluatorInStringFind<Left, String>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Left &str_a = VariantInternalAccessor<Left>::get(left);
		const String &str_b = VariantInternalAccessor<String>::get(right);
		VariantInternalAccessor<bool>::get(r_ret) = str_b.find(str_a) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<String>::convert(right).find(PtrToArg<Left>::convert(left)) != -1, r_ret);
	}
	using ReturnType = bool;
};

template <typename Left>
class OperatorEvaluatorInStringFind<Left, StringName> : public CommonEvaluate<OperatorEvaluatorInStringFind<Left, StringName>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Left &str_a = VariantInternalAccessor<Left>::get(left);
		const String str_b = VariantInternalAccessor<StringName>::get(right).operator String();
		VariantInternalAccessor<bool>::get(r_ret) = str_b.find(str_a) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<StringName>::convert(right).operator String().find(PtrToArg<Left>::convert(left)) != -1, r_ret);
	}
	using ReturnType = bool;
};

template <typename A, typename B>
class OperatorEvaluatorInArrayFind : public CommonEvaluate<OperatorEvaluatorInArrayFind<A, B>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const A &a = VariantInternalAccessor<A>::get(left);
		const B &b = VariantInternalAccessor<B>::get(right);
		VariantInternalAccessor<bool>::get(r_ret) = b.find(a) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<B>::convert(right).find(PtrToArg<A>::convert(left)) != -1, r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorInArrayFindNil : public CommonEvaluate<OperatorEvaluatorInArrayFindNil> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Array &b = VariantInternalAccessor<Array>::get(right);
		VariantInternalAccessor<bool>::get(r_ret) = b.find(Variant()) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Array>::convert(right).find(Variant()) != -1, r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorInArrayFindObject : public CommonEvaluate<OperatorEvaluatorInArrayFindObject> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Array &b = VariantInternalAccessor<Array>::get(right);
		VariantInternalAccessor<bool>::get(r_ret) = b.find(*left) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Array>::convert(right).find(PtrToArg<Object *>::convert(left)) != -1, r_ret);
	}
	using ReturnType = bool;
};

template <typename A>
class OperatorEvaluatorInDictionaryHas : public CommonEvaluate<OperatorEvaluatorInDictionaryHas<A>> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Dictionary &b = VariantInternalAccessor<Dictionary>::get(right);
		const A &a = VariantInternalAccessor<A>::get(left);
		VariantInternalAccessor<bool>::get(r_ret) = b.has(a);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Dictionary>::convert(right).has(PtrToArg<A>::convert(left)), r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorInDictionaryHasNil : public CommonEvaluate<OperatorEvaluatorInDictionaryHasNil> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Dictionary &b = VariantInternalAccessor<Dictionary>::get(right);
		VariantInternalAccessor<bool>::get(r_ret) = b.has(Variant());
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Dictionary>::convert(right).has(Variant()), r_ret);
	}
	using ReturnType = bool;
};

class OperatorEvaluatorInDictionaryHasObject : public CommonEvaluate<OperatorEvaluatorInDictionaryHasObject> {
public:
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Dictionary &b = VariantInternalAccessor<Dictionary>::get(right);
		VariantInternalAccessor<bool>::get(r_ret) = b.has(*left);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Dictionary>::convert(right).has(PtrToArg<Object *>::convert(left)), r_ret);
	}
	using ReturnType = bool;
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

		const String &a = VariantInternalAccessor<String>::get(&p_left);

		bool exist;
		b->get(a, &exist);
		*r_ret = exist;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *l = right->get_validated_object();
		if (unlikely(!l)) {
			VariantInternalAccessor<bool>::get(r_ret) = false;
			ERR_FAIL_MSG("Invalid base object for 'in'.");
		}
		const String &a = VariantInternalAccessor<String>::get(left);

		bool valid;
		l->get(a, &valid);
		VariantInternalAccessor<bool>::get(r_ret) = valid;
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

		const StringName &a = VariantInternalAccessor<StringName>::get(&p_left);

		bool exist;
		b->get(a, &exist);
		*r_ret = exist;
		r_valid = true;
	}
	static inline void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *l = right->get_validated_object();
		if (unlikely(!l)) {
			VariantInternalAccessor<bool>::get(r_ret) = false;
			ERR_FAIL_MSG("Invalid base object for 'in'.");
		}
		const StringName &a = VariantInternalAccessor<StringName>::get(left);

		bool valid;
		l->get(a, &valid);
		VariantInternalAccessor<bool>::get(r_ret) = valid;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		bool valid;
		PtrToArg<Object *>::convert(right)->get(PtrToArg<StringName>::convert(left), &valid);
		PtrToArg<bool>::encode(valid, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};
