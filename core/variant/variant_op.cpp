/*************************************************************************/
/*  variant_op.cpp                                                       */
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) / *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) / PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
		*VariantGetInternalPtr<R>::get_ptr(r_ret) = *VariantGetInternalPtr<A>::get_ptr(left) % *VariantGetInternalPtr<B>::get_ptr(right);
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<R>::encode(PtrToArg<A>::convert(left) % PtrToArg<B>::convert(right), r_ret);
	}
	static Variant::Type get_return_type() { return GetTypeInfo<R>::VARIANT_TYPE; }
};

template <class R, class A>
class OperatorEvaluatorNeg {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(&p_left);
		*r_ret = -a;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<R>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Object *a = left->get_validated_object();
		const Object *b = right->get_validated_object();
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Object *a = left->get_validated_object();
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Object *b = right->get_validated_object();
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *a = left->get_validated_object();
		Object *b = right->get_validated_object();
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *a = left->get_validated_object();
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *b = right->get_validated_object();
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Array>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<Vector<T>>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<String>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<String>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<String>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<String>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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

#define OP_EVALUATOR(m_class_name, m_left, m_right, m_op)                                                                    \
	class m_class_name {                                                                                                     \
	public:                                                                                                                  \
		static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {                 \
			*r_ret = m_op(_operate_get_##m_left(&p_left), _operate_get_##m_right(&p_right));                                 \
			r_valid = true;                                                                                                  \
		}                                                                                                                    \
                                                                                                                             \
		static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {                          \
			VariantTypeChanger<bool>::change(r_ret);                                                                         \
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

class OperatorEvaluatorNotBool {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		*r_ret = !*VariantGetInternalPtr<bool>::get_ptr(&p_left);
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		VariantTypeChanger<bool>::change(r_ret);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = left->get_validated_object() == nullptr;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<Object *>::convert(left) == nullptr, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

////

class OperatorEvaluatorInStringFind {
public:
	static void evaluate(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid) {
		const String &str_a = *VariantGetInternalPtr<String>::get_ptr(&p_left);
		const String &str_b = *VariantGetInternalPtr<String>::get_ptr(&p_right);

		*r_ret = str_b.find(str_a) != -1;
		r_valid = true;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const String &str_a = *VariantGetInternalPtr<String>::get_ptr(left);
		const String &str_b = *VariantGetInternalPtr<String>::get_ptr(right);
		VariantTypeChanger<bool>::change(r_ret);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = str_b.find(str_a) != -1;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		PtrToArg<bool>::encode(PtrToArg<String>::convert(right).find(PtrToArg<String>::convert(left)) != -1, r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const A &a = *VariantGetInternalPtr<A>::get_ptr(left);
		const B &b = *VariantGetInternalPtr<B>::get_ptr(right);
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Array &b = *VariantGetInternalPtr<Array>::get_ptr(right);
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Array &b = *VariantGetInternalPtr<Array>::get_ptr(right);
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Dictionary &b = *VariantGetInternalPtr<Dictionary>::get_ptr(right);
		const A &a = *VariantGetInternalPtr<A>::get_ptr(left);
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Dictionary &b = *VariantGetInternalPtr<Dictionary>::get_ptr(right);
		VariantTypeChanger<bool>::change(r_ret);
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
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		const Dictionary &b = *VariantGetInternalPtr<Dictionary>::get_ptr(right);
		VariantTypeChanger<bool>::change(r_ret);
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

		b->get(a, &r_valid);
		*r_ret = r_valid;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *l = right->get_validated_object();
		ERR_FAIL_COND(l == nullptr);
		const String &a = *VariantGetInternalPtr<String>::get_ptr(left);

		bool valid;
		l->get(a, &valid);
		VariantTypeChanger<bool>::change(r_ret);
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

		b->get(a, &r_valid);
		*r_ret = r_valid;
	}
	static void validated_evaluate(const Variant *left, const Variant *right, Variant *r_ret) {
		Object *l = right->get_validated_object();
		ERR_FAIL_COND(l == nullptr);
		const StringName &a = *VariantGetInternalPtr<StringName>::get_ptr(left);

		bool valid;
		l->get(a, &valid);
		VariantTypeChanger<bool>::change(r_ret);
		*VariantGetInternalPtr<bool>::get_ptr(r_ret) = valid;
	}
	static void ptr_evaluate(const void *left, const void *right, void *r_ret) {
		bool valid;
		PtrToArg<Object *>::convert(right)->get(PtrToArg<StringName>::convert(left), &valid);
		PtrToArg<bool>::encode(valid, r_ret);
	}
	static Variant::Type get_return_type() { return Variant::BOOL; }
};

typedef void (*VariantEvaluatorFunction)(const Variant &p_left, const Variant &p_right, Variant *r_ret, bool &r_valid);

static Variant::Type operator_return_type_table[Variant::OP_MAX][Variant::VARIANT_MAX][Variant::VARIANT_MAX];
static VariantEvaluatorFunction operator_evaluator_table[Variant::OP_MAX][Variant::VARIANT_MAX][Variant::VARIANT_MAX];
static Variant::ValidatedOperatorEvaluator validated_operator_evaluator_table[Variant::OP_MAX][Variant::VARIANT_MAX][Variant::VARIANT_MAX];
static Variant::PTROperatorEvaluator ptr_operator_evaluator_table[Variant::OP_MAX][Variant::VARIANT_MAX][Variant::VARIANT_MAX];

template <class T>
void register_op(Variant::Operator p_op, Variant::Type p_type_a, Variant::Type p_type_b) {
	operator_return_type_table[p_op][p_type_a][p_type_b] = T::get_return_type();
	operator_evaluator_table[p_op][p_type_a][p_type_b] = T::evaluate;
	validated_operator_evaluator_table[p_op][p_type_a][p_type_b] = T::validated_evaluate;
	ptr_operator_evaluator_table[p_op][p_type_a][p_type_b] = T::ptr_evaluate;
}

void Variant::_register_variant_operators() {
	memset(operator_return_type_table, 0, sizeof(operator_return_type_table));
	memset(operator_evaluator_table, 0, sizeof(operator_evaluator_table));
	memset(validated_operator_evaluator_table, 0, sizeof(validated_operator_evaluator_table));
	memset(ptr_operator_evaluator_table, 0, sizeof(ptr_operator_evaluator_table));

	register_op<OperatorEvaluatorAdd<int64_t, int64_t, int64_t>>(Variant::OP_ADD, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorAdd<double, int64_t, double>>(Variant::OP_ADD, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorAdd<double, double, int64_t>>(Variant::OP_ADD, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorAdd<double, double, double>>(Variant::OP_ADD, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorAdd<String, String, String>>(Variant::OP_ADD, Variant::STRING, Variant::STRING);
	register_op<OperatorEvaluatorAdd<Vector2, Vector2, Vector2>>(Variant::OP_ADD, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorAdd<Vector2i, Vector2i, Vector2i>>(Variant::OP_ADD, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorAdd<Vector3, Vector3, Vector3>>(Variant::OP_ADD, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorAdd<Vector3i, Vector3i, Vector3i>>(Variant::OP_ADD, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorAdd<Quat, Quat, Quat>>(Variant::OP_ADD, Variant::QUAT, Variant::QUAT);
	register_op<OperatorEvaluatorAdd<Color, Color, Color>>(Variant::OP_ADD, Variant::COLOR, Variant::COLOR);
	register_op<OperatorEvaluatorAddArray>(Variant::OP_ADD, Variant::ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorAppendArray<uint8_t>>(Variant::OP_ADD, Variant::PACKED_BYTE_ARRAY, Variant::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorAppendArray<int32_t>>(Variant::OP_ADD, Variant::PACKED_INT32_ARRAY, Variant::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorAppendArray<int64_t>>(Variant::OP_ADD, Variant::PACKED_INT64_ARRAY, Variant::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorAppendArray<float>>(Variant::OP_ADD, Variant::PACKED_FLOAT32_ARRAY, Variant::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorAppendArray<double>>(Variant::OP_ADD, Variant::PACKED_FLOAT64_ARRAY, Variant::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorAppendArray<String>>(Variant::OP_ADD, Variant::PACKED_STRING_ARRAY, Variant::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorAppendArray<Vector2>>(Variant::OP_ADD, Variant::PACKED_VECTOR2_ARRAY, Variant::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorAppendArray<Vector3>>(Variant::OP_ADD, Variant::PACKED_VECTOR3_ARRAY, Variant::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorAppendArray<Color>>(Variant::OP_ADD, Variant::PACKED_COLOR_ARRAY, Variant::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorSub<int64_t, int64_t, int64_t>>(Variant::OP_SUBTRACT, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorSub<double, int64_t, double>>(Variant::OP_SUBTRACT, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorSub<double, double, int64_t>>(Variant::OP_SUBTRACT, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorSub<double, double, double>>(Variant::OP_SUBTRACT, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorSub<Vector2, Vector2, Vector2>>(Variant::OP_SUBTRACT, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorSub<Vector2i, Vector2i, Vector2i>>(Variant::OP_SUBTRACT, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorSub<Vector3, Vector3, Vector3>>(Variant::OP_SUBTRACT, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorSub<Vector3i, Vector3i, Vector3i>>(Variant::OP_SUBTRACT, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorSub<Quat, Quat, Quat>>(Variant::OP_SUBTRACT, Variant::QUAT, Variant::QUAT);
	register_op<OperatorEvaluatorSub<Color, Color, Color>>(Variant::OP_SUBTRACT, Variant::COLOR, Variant::COLOR);

	register_op<OperatorEvaluatorMul<int64_t, int64_t, int64_t>>(Variant::OP_MULTIPLY, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorMul<double, int64_t, double>>(Variant::OP_MULTIPLY, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorMul<Vector2, int64_t, Vector2>>(Variant::OP_MULTIPLY, Variant::INT, Variant::VECTOR2);
	register_op<OperatorEvaluatorMul<Vector2i, int64_t, Vector2i>>(Variant::OP_MULTIPLY, Variant::INT, Variant::VECTOR2I);
	register_op<OperatorEvaluatorMul<Vector3, int64_t, Vector3>>(Variant::OP_MULTIPLY, Variant::INT, Variant::VECTOR3);
	register_op<OperatorEvaluatorMul<Vector3i, int64_t, Vector3i>>(Variant::OP_MULTIPLY, Variant::INT, Variant::VECTOR3I);

	register_op<OperatorEvaluatorMul<double, double, double>>(Variant::OP_MULTIPLY, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorMul<double, double, int64_t>>(Variant::OP_MULTIPLY, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorMul<Vector2, double, Vector2>>(Variant::OP_MULTIPLY, Variant::FLOAT, Variant::VECTOR2);
	register_op<OperatorEvaluatorMul<Vector2i, double, Vector2i>>(Variant::OP_MULTIPLY, Variant::FLOAT, Variant::VECTOR2I);
	register_op<OperatorEvaluatorMul<Vector3, double, Vector3>>(Variant::OP_MULTIPLY, Variant::FLOAT, Variant::VECTOR3);
	register_op<OperatorEvaluatorMul<Vector3i, double, Vector3i>>(Variant::OP_MULTIPLY, Variant::FLOAT, Variant::VECTOR3I);

	register_op<OperatorEvaluatorMul<Vector2, Vector2, Vector2>>(Variant::OP_MULTIPLY, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorMul<Vector2, Vector2, int64_t>>(Variant::OP_MULTIPLY, Variant::VECTOR2, Variant::INT);
	register_op<OperatorEvaluatorMul<Vector2, Vector2, double>>(Variant::OP_MULTIPLY, Variant::VECTOR2, Variant::FLOAT);

	register_op<OperatorEvaluatorMul<Vector2i, Vector2i, Vector2i>>(Variant::OP_MULTIPLY, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorMul<Vector2i, Vector2i, int64_t>>(Variant::OP_MULTIPLY, Variant::VECTOR2I, Variant::INT);
	register_op<OperatorEvaluatorMul<Vector2i, Vector2i, double>>(Variant::OP_MULTIPLY, Variant::VECTOR2I, Variant::FLOAT);

	register_op<OperatorEvaluatorMul<Vector3, Vector3, Vector3>>(Variant::OP_MULTIPLY, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorMul<Vector3, Vector3, int64_t>>(Variant::OP_MULTIPLY, Variant::VECTOR3, Variant::INT);
	register_op<OperatorEvaluatorMul<Vector3, Vector3, double>>(Variant::OP_MULTIPLY, Variant::VECTOR3, Variant::FLOAT);

	register_op<OperatorEvaluatorMul<Vector3i, Vector3i, Vector3i>>(Variant::OP_MULTIPLY, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorMul<Vector3i, Vector3i, int64_t>>(Variant::OP_MULTIPLY, Variant::VECTOR3I, Variant::INT);
	register_op<OperatorEvaluatorMul<Vector3i, Vector3i, double>>(Variant::OP_MULTIPLY, Variant::VECTOR3I, Variant::FLOAT);

	register_op<OperatorEvaluatorMul<Quat, Quat, Quat>>(Variant::OP_MULTIPLY, Variant::QUAT, Variant::QUAT);
	register_op<OperatorEvaluatorMul<Quat, Quat, int64_t>>(Variant::OP_MULTIPLY, Variant::QUAT, Variant::INT);
	register_op<OperatorEvaluatorMul<Quat, Quat, double>>(Variant::OP_MULTIPLY, Variant::QUAT, Variant::FLOAT);

	register_op<OperatorEvaluatorMul<Color, Color, Color>>(Variant::OP_MULTIPLY, Variant::COLOR, Variant::COLOR);
	register_op<OperatorEvaluatorMul<Color, Color, int64_t>>(Variant::OP_MULTIPLY, Variant::COLOR, Variant::INT);
	register_op<OperatorEvaluatorMul<Color, Color, double>>(Variant::OP_MULTIPLY, Variant::COLOR, Variant::FLOAT);

	register_op<OperatorEvaluatorMul<Transform2D, Transform2D, Transform2D>>(Variant::OP_MULTIPLY, Variant::TRANSFORM2D, Variant::TRANSFORM2D);
	register_op<OperatorEvaluatorXForm<Vector2, Transform2D, Vector2>>(Variant::OP_MULTIPLY, Variant::TRANSFORM2D, Variant::VECTOR2);
	register_op<OperatorEvaluatorXFormInv<Vector2, Vector2, Transform2D>>(Variant::OP_MULTIPLY, Variant::VECTOR2, Variant::TRANSFORM2D);
	register_op<OperatorEvaluatorXForm<Rect2, Transform2D, Rect2>>(Variant::OP_MULTIPLY, Variant::TRANSFORM2D, Variant::RECT2);
	register_op<OperatorEvaluatorXFormInv<Rect2, Rect2, Transform2D>>(Variant::OP_MULTIPLY, Variant::RECT2, Variant::TRANSFORM2D);
	register_op<OperatorEvaluatorXForm<Vector<Vector2>, Transform2D, Vector<Vector2>>>(Variant::OP_MULTIPLY, Variant::TRANSFORM2D, Variant::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorXFormInv<Vector<Vector2>, Vector<Vector2>, Transform2D>>(Variant::OP_MULTIPLY, Variant::PACKED_VECTOR2_ARRAY, Variant::TRANSFORM2D);

	register_op<OperatorEvaluatorMul<Transform, Transform, Transform>>(Variant::OP_MULTIPLY, Variant::TRANSFORM, Variant::TRANSFORM);
	register_op<OperatorEvaluatorXForm<Vector3, Transform, Vector3>>(Variant::OP_MULTIPLY, Variant::TRANSFORM, Variant::VECTOR3);
	register_op<OperatorEvaluatorXFormInv<Vector3, Vector3, Transform>>(Variant::OP_MULTIPLY, Variant::VECTOR3, Variant::TRANSFORM);
	register_op<OperatorEvaluatorXForm<::AABB, Transform, ::AABB>>(Variant::OP_MULTIPLY, Variant::TRANSFORM, Variant::AABB);
	register_op<OperatorEvaluatorXFormInv<::AABB, ::AABB, Transform>>(Variant::OP_MULTIPLY, Variant::AABB, Variant::TRANSFORM);
	register_op<OperatorEvaluatorXForm<Vector<Vector3>, Transform, Vector<Vector3>>>(Variant::OP_MULTIPLY, Variant::TRANSFORM, Variant::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorXFormInv<Vector<Vector3>, Vector<Vector3>, Transform>>(Variant::OP_MULTIPLY, Variant::PACKED_VECTOR3_ARRAY, Variant::TRANSFORM);

	register_op<OperatorEvaluatorMul<Basis, Basis, Basis>>(Variant::OP_MULTIPLY, Variant::BASIS, Variant::BASIS);
	register_op<OperatorEvaluatorXForm<Vector3, Basis, Vector3>>(Variant::OP_MULTIPLY, Variant::BASIS, Variant::VECTOR3);
	register_op<OperatorEvaluatorXFormInv<Vector3, Vector3, Basis>>(Variant::OP_MULTIPLY, Variant::VECTOR3, Variant::BASIS);

	register_op<OperatorEvaluatorMul<Quat, Quat, Quat>>(Variant::OP_MULTIPLY, Variant::QUAT, Variant::QUAT);
	register_op<OperatorEvaluatorMul<Quat, Quat, int64_t>>(Variant::OP_MULTIPLY, Variant::QUAT, Variant::INT);
	register_op<OperatorEvaluatorMul<Quat, int64_t, Quat>>(Variant::OP_MULTIPLY, Variant::INT, Variant::QUAT);
	register_op<OperatorEvaluatorMul<Quat, Quat, double>>(Variant::OP_MULTIPLY, Variant::QUAT, Variant::FLOAT);
	register_op<OperatorEvaluatorMul<Quat, double, Quat>>(Variant::OP_MULTIPLY, Variant::FLOAT, Variant::QUAT);
	register_op<OperatorEvaluatorXForm<Vector3, Quat, Vector3>>(Variant::OP_MULTIPLY, Variant::QUAT, Variant::VECTOR3);
	register_op<OperatorEvaluatorXFormInv<Vector3, Vector3, Quat>>(Variant::OP_MULTIPLY, Variant::VECTOR3, Variant::QUAT);

	register_op<OperatorEvaluatorMul<Color, Color, Color>>(Variant::OP_MULTIPLY, Variant::COLOR, Variant::COLOR);
	register_op<OperatorEvaluatorMul<Color, Color, int64_t>>(Variant::OP_MULTIPLY, Variant::COLOR, Variant::INT);
	register_op<OperatorEvaluatorMul<Color, int64_t, Color>>(Variant::OP_MULTIPLY, Variant::INT, Variant::COLOR);
	register_op<OperatorEvaluatorMul<Color, Color, double>>(Variant::OP_MULTIPLY, Variant::COLOR, Variant::FLOAT);
	register_op<OperatorEvaluatorMul<Color, double, Color>>(Variant::OP_MULTIPLY, Variant::FLOAT, Variant::COLOR);

	register_op<OperatorEvaluatorDivNZ<int64_t, int64_t, int64_t>>(Variant::OP_DIVIDE, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorDiv<double, double, int64_t>>(Variant::OP_DIVIDE, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorDiv<double, int64_t, double>>(Variant::OP_DIVIDE, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorDiv<double, double, double>>(Variant::OP_DIVIDE, Variant::FLOAT, Variant::FLOAT);

	register_op<OperatorEvaluatorDiv<Vector2, Vector2, Vector2>>(Variant::OP_DIVIDE, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorDiv<Vector2, Vector2, double>>(Variant::OP_DIVIDE, Variant::VECTOR2, Variant::FLOAT);
	register_op<OperatorEvaluatorDiv<Vector2, Vector2, int64_t>>(Variant::OP_DIVIDE, Variant::VECTOR2, Variant::INT);

	register_op<OperatorEvaluatorDiv<Vector2i, Vector2i, Vector2i>>(Variant::OP_DIVIDE, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorDivNZ<Vector2i, Vector2i, double>>(Variant::OP_DIVIDE, Variant::VECTOR2I, Variant::FLOAT);
	register_op<OperatorEvaluatorDivNZ<Vector2i, Vector2i, int64_t>>(Variant::OP_DIVIDE, Variant::VECTOR2I, Variant::INT);

	register_op<OperatorEvaluatorDiv<Vector2, Vector2, Vector2>>(Variant::OP_DIVIDE, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorDiv<Vector2, Vector2, double>>(Variant::OP_DIVIDE, Variant::VECTOR2, Variant::FLOAT);
	register_op<OperatorEvaluatorDiv<Vector2, Vector2, int64_t>>(Variant::OP_DIVIDE, Variant::VECTOR2, Variant::INT);

	register_op<OperatorEvaluatorDiv<Vector3, Vector3, Vector3>>(Variant::OP_DIVIDE, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorDiv<Vector3, Vector3, double>>(Variant::OP_DIVIDE, Variant::VECTOR3, Variant::FLOAT);
	register_op<OperatorEvaluatorDiv<Vector3, Vector3, int64_t>>(Variant::OP_DIVIDE, Variant::VECTOR3, Variant::INT);

	register_op<OperatorEvaluatorDiv<Vector3i, Vector3i, Vector3i>>(Variant::OP_DIVIDE, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorDivNZ<Vector3i, Vector3i, double>>(Variant::OP_DIVIDE, Variant::VECTOR3I, Variant::FLOAT);
	register_op<OperatorEvaluatorDivNZ<Vector3i, Vector3i, int64_t>>(Variant::OP_DIVIDE, Variant::VECTOR3I, Variant::INT);

	register_op<OperatorEvaluatorDiv<Quat, Quat, double>>(Variant::OP_DIVIDE, Variant::QUAT, Variant::FLOAT);
	register_op<OperatorEvaluatorDiv<Quat, Quat, int64_t>>(Variant::OP_DIVIDE, Variant::QUAT, Variant::INT);

	register_op<OperatorEvaluatorDiv<Color, Color, Color>>(Variant::OP_DIVIDE, Variant::COLOR, Variant::COLOR);
	register_op<OperatorEvaluatorDiv<Color, Color, double>>(Variant::OP_DIVIDE, Variant::COLOR, Variant::FLOAT);
	register_op<OperatorEvaluatorDiv<Color, Color, int64_t>>(Variant::OP_DIVIDE, Variant::COLOR, Variant::INT);

	register_op<OperatorEvaluatorModNZ<int64_t, int64_t, int64_t>>(Variant::OP_MODULE, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorMod<Vector2i, Vector2i, Vector2i>>(Variant::OP_MODULE, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorModNZ<Vector2i, Vector2i, int64_t>>(Variant::OP_MODULE, Variant::VECTOR2I, Variant::INT);

	register_op<OperatorEvaluatorMod<Vector3i, Vector3i, Vector3i>>(Variant::OP_MODULE, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorModNZ<Vector3i, Vector3i, int64_t>>(Variant::OP_MODULE, Variant::VECTOR3I, Variant::INT);

	register_op<OperatorEvaluatorStringModNil>(Variant::OP_MODULE, Variant::STRING, Variant::NIL);

	register_op<OperatorEvaluatorStringModT<bool>>(Variant::OP_MODULE, Variant::STRING, Variant::BOOL);
	register_op<OperatorEvaluatorStringModT<int64_t>>(Variant::OP_MODULE, Variant::STRING, Variant::INT);
	register_op<OperatorEvaluatorStringModT<double>>(Variant::OP_MODULE, Variant::STRING, Variant::FLOAT);
	register_op<OperatorEvaluatorStringModT<String>>(Variant::OP_MODULE, Variant::STRING, Variant::STRING);
	register_op<OperatorEvaluatorStringModT<Vector2>>(Variant::OP_MODULE, Variant::STRING, Variant::VECTOR2);
	register_op<OperatorEvaluatorStringModT<Vector2i>>(Variant::OP_MODULE, Variant::STRING, Variant::VECTOR2I);
	register_op<OperatorEvaluatorStringModT<Rect2>>(Variant::OP_MODULE, Variant::STRING, Variant::RECT2);
	register_op<OperatorEvaluatorStringModT<Rect2i>>(Variant::OP_MODULE, Variant::STRING, Variant::RECT2I);
	register_op<OperatorEvaluatorStringModT<Vector3>>(Variant::OP_MODULE, Variant::STRING, Variant::VECTOR3);
	register_op<OperatorEvaluatorStringModT<Vector3i>>(Variant::OP_MODULE, Variant::STRING, Variant::VECTOR3I);
	register_op<OperatorEvaluatorStringModT<Transform2D>>(Variant::OP_MODULE, Variant::STRING, Variant::TRANSFORM2D);
	register_op<OperatorEvaluatorStringModT<Plane>>(Variant::OP_MODULE, Variant::STRING, Variant::PLANE);
	register_op<OperatorEvaluatorStringModT<Quat>>(Variant::OP_MODULE, Variant::STRING, Variant::QUAT);
	register_op<OperatorEvaluatorStringModT<::AABB>>(Variant::OP_MODULE, Variant::STRING, Variant::AABB);
	register_op<OperatorEvaluatorStringModT<Basis>>(Variant::OP_MODULE, Variant::STRING, Variant::BASIS);
	register_op<OperatorEvaluatorStringModT<Transform>>(Variant::OP_MODULE, Variant::STRING, Variant::TRANSFORM);

	register_op<OperatorEvaluatorStringModT<Color>>(Variant::OP_MODULE, Variant::STRING, Variant::COLOR);
	register_op<OperatorEvaluatorStringModT<StringName>>(Variant::OP_MODULE, Variant::STRING, Variant::STRING_NAME);
	register_op<OperatorEvaluatorStringModT<NodePath>>(Variant::OP_MODULE, Variant::STRING, Variant::NODE_PATH);
	register_op<OperatorEvaluatorStringModObject>(Variant::OP_MODULE, Variant::STRING, Variant::OBJECT);
	register_op<OperatorEvaluatorStringModT<Callable>>(Variant::OP_MODULE, Variant::STRING, Variant::CALLABLE);
	register_op<OperatorEvaluatorStringModT<Signal>>(Variant::OP_MODULE, Variant::STRING, Variant::SIGNAL);
	register_op<OperatorEvaluatorStringModT<Dictionary>>(Variant::OP_MODULE, Variant::STRING, Variant::DICTIONARY);
	register_op<OperatorEvaluatorStringModArray>(Variant::OP_MODULE, Variant::STRING, Variant::ARRAY);

	register_op<OperatorEvaluatorStringModT<PackedByteArray>>(Variant::OP_MODULE, Variant::STRING, Variant::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorStringModT<PackedInt32Array>>(Variant::OP_MODULE, Variant::STRING, Variant::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorStringModT<PackedInt64Array>>(Variant::OP_MODULE, Variant::STRING, Variant::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorStringModT<PackedFloat32Array>>(Variant::OP_MODULE, Variant::STRING, Variant::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorStringModT<PackedFloat64Array>>(Variant::OP_MODULE, Variant::STRING, Variant::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorStringModT<PackedStringArray>>(Variant::OP_MODULE, Variant::STRING, Variant::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorStringModT<PackedVector2Array>>(Variant::OP_MODULE, Variant::STRING, Variant::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorStringModT<PackedVector3Array>>(Variant::OP_MODULE, Variant::STRING, Variant::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorStringModT<PackedColorArray>>(Variant::OP_MODULE, Variant::STRING, Variant::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorNeg<int64_t, int64_t>>(Variant::OP_NEGATE, Variant::INT, Variant::NIL);
	register_op<OperatorEvaluatorNeg<double, double>>(Variant::OP_NEGATE, Variant::FLOAT, Variant::NIL);
	register_op<OperatorEvaluatorNeg<Vector2, Vector2>>(Variant::OP_NEGATE, Variant::VECTOR2, Variant::NIL);
	register_op<OperatorEvaluatorNeg<Vector2i, Vector2i>>(Variant::OP_NEGATE, Variant::VECTOR2I, Variant::NIL);
	register_op<OperatorEvaluatorNeg<Vector3, Vector3>>(Variant::OP_NEGATE, Variant::VECTOR3, Variant::NIL);
	register_op<OperatorEvaluatorNeg<Vector3i, Vector3i>>(Variant::OP_NEGATE, Variant::VECTOR3I, Variant::NIL);
	register_op<OperatorEvaluatorNeg<Quat, Quat>>(Variant::OP_NEGATE, Variant::QUAT, Variant::NIL);
	register_op<OperatorEvaluatorNeg<Plane, Plane>>(Variant::OP_NEGATE, Variant::PLANE, Variant::NIL);
	register_op<OperatorEvaluatorNeg<Color, Color>>(Variant::OP_NEGATE, Variant::COLOR, Variant::NIL);

	register_op<OperatorEvaluatorPos<int64_t, int64_t>>(Variant::OP_POSITIVE, Variant::INT, Variant::NIL);
	register_op<OperatorEvaluatorPos<double, double>>(Variant::OP_POSITIVE, Variant::FLOAT, Variant::NIL);
	register_op<OperatorEvaluatorPos<Vector2, Vector2>>(Variant::OP_POSITIVE, Variant::VECTOR2, Variant::NIL);
	register_op<OperatorEvaluatorPos<Vector2i, Vector2i>>(Variant::OP_POSITIVE, Variant::VECTOR2I, Variant::NIL);
	register_op<OperatorEvaluatorPos<Vector3, Vector3>>(Variant::OP_POSITIVE, Variant::VECTOR3, Variant::NIL);
	register_op<OperatorEvaluatorPos<Vector3i, Vector3i>>(Variant::OP_POSITIVE, Variant::VECTOR3I, Variant::NIL);
	register_op<OperatorEvaluatorPos<Quat, Quat>>(Variant::OP_POSITIVE, Variant::QUAT, Variant::NIL);
	register_op<OperatorEvaluatorPos<Plane, Plane>>(Variant::OP_POSITIVE, Variant::PLANE, Variant::NIL);
	register_op<OperatorEvaluatorPos<Color, Color>>(Variant::OP_POSITIVE, Variant::COLOR, Variant::NIL);

	register_op<OperatorEvaluatorShiftLeft<int64_t, int64_t, int64_t>>(Variant::OP_SHIFT_LEFT, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorShiftRight<int64_t, int64_t, int64_t>>(Variant::OP_SHIFT_RIGHT, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorBitOr<int64_t, int64_t, int64_t>>(Variant::OP_BIT_OR, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorBitAnd<int64_t, int64_t, int64_t>>(Variant::OP_BIT_AND, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorBitXor<int64_t, int64_t, int64_t>>(Variant::OP_BIT_XOR, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorBitNeg<int64_t, int64_t>>(Variant::OP_BIT_NEGATE, Variant::INT, Variant::NIL);

	register_op<OperatorEvaluatorBitNeg<int64_t, int64_t>>(Variant::OP_BIT_NEGATE, Variant::INT, Variant::NIL);

	register_op<OperatorEvaluatorAlwaysTrue<Variant::OP_EQUAL, Variant::NIL, Variant::NIL>>(Variant::OP_EQUAL, Variant::NIL, Variant::NIL);
	register_op<OperatorEvaluatorEqual<bool, bool>>(Variant::OP_EQUAL, Variant::BOOL, Variant::BOOL);
	register_op<OperatorEvaluatorEqual<int64_t, int64_t>>(Variant::OP_EQUAL, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorEqual<int64_t, double>>(Variant::OP_EQUAL, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorEqual<double, int64_t>>(Variant::OP_EQUAL, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorEqual<double, double>>(Variant::OP_EQUAL, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorEqual<String, String>>(Variant::OP_EQUAL, Variant::STRING, Variant::STRING);
	register_op<OperatorEvaluatorEqual<Vector2, Vector2>>(Variant::OP_EQUAL, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorEqual<Vector2i, Vector2i>>(Variant::OP_EQUAL, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorEqual<Rect2, Rect2>>(Variant::OP_EQUAL, Variant::RECT2, Variant::RECT2);
	register_op<OperatorEvaluatorEqual<Rect2i, Rect2i>>(Variant::OP_EQUAL, Variant::RECT2I, Variant::RECT2I);
	register_op<OperatorEvaluatorEqual<Vector3, Vector3>>(Variant::OP_EQUAL, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorEqual<Vector3i, Vector3i>>(Variant::OP_EQUAL, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorEqual<Transform2D, Transform2D>>(Variant::OP_EQUAL, Variant::TRANSFORM2D, Variant::TRANSFORM2D);
	register_op<OperatorEvaluatorEqual<Plane, Plane>>(Variant::OP_EQUAL, Variant::PLANE, Variant::PLANE);
	register_op<OperatorEvaluatorEqual<Quat, Quat>>(Variant::OP_EQUAL, Variant::QUAT, Variant::QUAT);
	register_op<OperatorEvaluatorEqual<::AABB, ::AABB>>(Variant::OP_EQUAL, Variant::AABB, Variant::AABB);
	register_op<OperatorEvaluatorEqual<Basis, Basis>>(Variant::OP_EQUAL, Variant::BASIS, Variant::BASIS);
	register_op<OperatorEvaluatorEqual<Transform, Transform>>(Variant::OP_EQUAL, Variant::TRANSFORM, Variant::TRANSFORM);
	register_op<OperatorEvaluatorEqual<Color, Color>>(Variant::OP_EQUAL, Variant::COLOR, Variant::COLOR);

	register_op<OperatorEvaluatorEqual<StringName, String>>(Variant::OP_EQUAL, Variant::STRING_NAME, Variant::STRING);
	register_op<OperatorEvaluatorEqual<String, StringName>>(Variant::OP_EQUAL, Variant::STRING, Variant::STRING_NAME);
	register_op<OperatorEvaluatorEqual<StringName, StringName>>(Variant::OP_EQUAL, Variant::STRING_NAME, Variant::STRING_NAME);

	register_op<OperatorEvaluatorEqual<NodePath, NodePath>>(Variant::OP_EQUAL, Variant::NODE_PATH, Variant::NODE_PATH);
	register_op<OperatorEvaluatorEqual<::RID, ::RID>>(Variant::OP_EQUAL, Variant::RID, Variant::RID);

	register_op<OperatorEvaluatorEqualObject>(Variant::OP_EQUAL, Variant::OBJECT, Variant::OBJECT);
	register_op<OperatorEvaluatorEqualObjectNil>(Variant::OP_EQUAL, Variant::OBJECT, Variant::NIL);
	register_op<OperatorEvaluatorEqualNilObject>(Variant::OP_EQUAL, Variant::NIL, Variant::OBJECT);

	register_op<OperatorEvaluatorEqual<Callable, Callable>>(Variant::OP_EQUAL, Variant::CALLABLE, Variant::CALLABLE);
	register_op<OperatorEvaluatorEqual<Signal, Signal>>(Variant::OP_EQUAL, Variant::SIGNAL, Variant::SIGNAL);
	register_op<OperatorEvaluatorEqual<Dictionary, Dictionary>>(Variant::OP_EQUAL, Variant::DICTIONARY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorEqual<Array, Array>>(Variant::OP_EQUAL, Variant::ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorEqual<PackedByteArray, PackedByteArray>>(Variant::OP_EQUAL, Variant::PACKED_BYTE_ARRAY, Variant::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedInt32Array, PackedInt32Array>>(Variant::OP_EQUAL, Variant::PACKED_INT32_ARRAY, Variant::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedInt64Array, PackedInt64Array>>(Variant::OP_EQUAL, Variant::PACKED_INT64_ARRAY, Variant::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedFloat32Array, PackedFloat32Array>>(Variant::OP_EQUAL, Variant::PACKED_FLOAT32_ARRAY, Variant::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedFloat64Array, PackedFloat64Array>>(Variant::OP_EQUAL, Variant::PACKED_FLOAT64_ARRAY, Variant::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedStringArray, PackedStringArray>>(Variant::OP_EQUAL, Variant::PACKED_STRING_ARRAY, Variant::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedVector2Array, PackedVector2Array>>(Variant::OP_EQUAL, Variant::PACKED_VECTOR2_ARRAY, Variant::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedVector3Array, PackedVector3Array>>(Variant::OP_EQUAL, Variant::PACKED_VECTOR3_ARRAY, Variant::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorEqual<PackedColorArray, PackedColorArray>>(Variant::OP_EQUAL, Variant::PACKED_COLOR_ARRAY, Variant::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorAlwaysFalse<Variant::OP_NOT_EQUAL, Variant::NIL, Variant::NIL>>(Variant::OP_NOT_EQUAL, Variant::NIL, Variant::NIL);
	register_op<OperatorEvaluatorNotEqual<bool, bool>>(Variant::OP_NOT_EQUAL, Variant::BOOL, Variant::BOOL);
	register_op<OperatorEvaluatorNotEqual<int64_t, int64_t>>(Variant::OP_NOT_EQUAL, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorNotEqual<int64_t, double>>(Variant::OP_NOT_EQUAL, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorNotEqual<double, int64_t>>(Variant::OP_NOT_EQUAL, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorNotEqual<double, double>>(Variant::OP_NOT_EQUAL, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorNotEqual<String, String>>(Variant::OP_NOT_EQUAL, Variant::STRING, Variant::STRING);
	register_op<OperatorEvaluatorNotEqual<Vector2, Vector2>>(Variant::OP_NOT_EQUAL, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorNotEqual<Vector2i, Vector2i>>(Variant::OP_NOT_EQUAL, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorNotEqual<Rect2, Rect2>>(Variant::OP_NOT_EQUAL, Variant::RECT2, Variant::RECT2);
	register_op<OperatorEvaluatorNotEqual<Rect2i, Rect2i>>(Variant::OP_NOT_EQUAL, Variant::RECT2I, Variant::RECT2I);
	register_op<OperatorEvaluatorNotEqual<Vector3, Vector3>>(Variant::OP_NOT_EQUAL, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorNotEqual<Vector3i, Vector3i>>(Variant::OP_NOT_EQUAL, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorNotEqual<Transform2D, Transform2D>>(Variant::OP_NOT_EQUAL, Variant::TRANSFORM2D, Variant::TRANSFORM2D);
	register_op<OperatorEvaluatorNotEqual<Plane, Plane>>(Variant::OP_NOT_EQUAL, Variant::PLANE, Variant::PLANE);
	register_op<OperatorEvaluatorNotEqual<Quat, Quat>>(Variant::OP_NOT_EQUAL, Variant::QUAT, Variant::QUAT);
	register_op<OperatorEvaluatorNotEqual<::AABB, ::AABB>>(Variant::OP_NOT_EQUAL, Variant::AABB, Variant::AABB);
	register_op<OperatorEvaluatorNotEqual<Basis, Basis>>(Variant::OP_NOT_EQUAL, Variant::BASIS, Variant::BASIS);
	register_op<OperatorEvaluatorNotEqual<Transform, Transform>>(Variant::OP_NOT_EQUAL, Variant::TRANSFORM, Variant::TRANSFORM);
	register_op<OperatorEvaluatorNotEqual<Color, Color>>(Variant::OP_NOT_EQUAL, Variant::COLOR, Variant::COLOR);

	register_op<OperatorEvaluatorNotEqual<StringName, String>>(Variant::OP_NOT_EQUAL, Variant::STRING_NAME, Variant::STRING);
	register_op<OperatorEvaluatorNotEqual<String, StringName>>(Variant::OP_NOT_EQUAL, Variant::STRING, Variant::STRING_NAME);
	register_op<OperatorEvaluatorNotEqual<StringName, StringName>>(Variant::OP_NOT_EQUAL, Variant::STRING_NAME, Variant::STRING_NAME);

	register_op<OperatorEvaluatorNotEqual<NodePath, NodePath>>(Variant::OP_NOT_EQUAL, Variant::NODE_PATH, Variant::NODE_PATH);
	register_op<OperatorEvaluatorNotEqual<::RID, ::RID>>(Variant::OP_NOT_EQUAL, Variant::RID, Variant::RID);

	register_op<OperatorEvaluatorNotEqualObject>(Variant::OP_NOT_EQUAL, Variant::OBJECT, Variant::OBJECT);
	register_op<OperatorEvaluatorNotEqualObjectNil>(Variant::OP_NOT_EQUAL, Variant::OBJECT, Variant::NIL);
	register_op<OperatorEvaluatorNotEqualNilObject>(Variant::OP_NOT_EQUAL, Variant::NIL, Variant::OBJECT);

	register_op<OperatorEvaluatorNotEqual<Callable, Callable>>(Variant::OP_NOT_EQUAL, Variant::CALLABLE, Variant::CALLABLE);
	register_op<OperatorEvaluatorNotEqual<Signal, Signal>>(Variant::OP_NOT_EQUAL, Variant::SIGNAL, Variant::SIGNAL);
	register_op<OperatorEvaluatorNotEqual<Dictionary, Dictionary>>(Variant::OP_NOT_EQUAL, Variant::DICTIONARY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorNotEqual<Array, Array>>(Variant::OP_NOT_EQUAL, Variant::ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedByteArray, PackedByteArray>>(Variant::OP_NOT_EQUAL, Variant::PACKED_BYTE_ARRAY, Variant::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedInt32Array, PackedInt32Array>>(Variant::OP_NOT_EQUAL, Variant::PACKED_INT32_ARRAY, Variant::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedInt64Array, PackedInt64Array>>(Variant::OP_NOT_EQUAL, Variant::PACKED_INT64_ARRAY, Variant::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedFloat32Array, PackedFloat32Array>>(Variant::OP_NOT_EQUAL, Variant::PACKED_FLOAT32_ARRAY, Variant::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedFloat64Array, PackedFloat64Array>>(Variant::OP_NOT_EQUAL, Variant::PACKED_FLOAT64_ARRAY, Variant::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedStringArray, PackedStringArray>>(Variant::OP_NOT_EQUAL, Variant::PACKED_STRING_ARRAY, Variant::PACKED_STRING_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedVector2Array, PackedVector2Array>>(Variant::OP_NOT_EQUAL, Variant::PACKED_VECTOR2_ARRAY, Variant::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedVector3Array, PackedVector3Array>>(Variant::OP_NOT_EQUAL, Variant::PACKED_VECTOR3_ARRAY, Variant::PACKED_VECTOR3_ARRAY);
	register_op<OperatorEvaluatorNotEqual<PackedColorArray, PackedColorArray>>(Variant::OP_NOT_EQUAL, Variant::PACKED_COLOR_ARRAY, Variant::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorLess<bool, bool>>(Variant::OP_LESS, Variant::BOOL, Variant::BOOL);
	register_op<OperatorEvaluatorLess<int64_t, int64_t>>(Variant::OP_LESS, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorLess<int64_t, double>>(Variant::OP_LESS, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorLess<double, int64_t>>(Variant::OP_LESS, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorLess<double, double>>(Variant::OP_LESS, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorLess<String, String>>(Variant::OP_LESS, Variant::STRING, Variant::STRING);
	register_op<OperatorEvaluatorLess<Vector2, Vector2>>(Variant::OP_LESS, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorLess<Vector2i, Vector2i>>(Variant::OP_LESS, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorLess<Vector3, Vector3>>(Variant::OP_LESS, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorLess<Vector3i, Vector3i>>(Variant::OP_LESS, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorLess<::RID, ::RID>>(Variant::OP_LESS, Variant::RID, Variant::RID);
	register_op<OperatorEvaluatorLess<Array, Array>>(Variant::OP_LESS, Variant::ARRAY, Variant::ARRAY);

	register_op<OperatorEvaluatorLessEqual<int64_t, int64_t>>(Variant::OP_LESS_EQUAL, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorLessEqual<int64_t, double>>(Variant::OP_LESS_EQUAL, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorLessEqual<double, int64_t>>(Variant::OP_LESS_EQUAL, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorLessEqual<double, double>>(Variant::OP_LESS_EQUAL, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorLessEqual<String, String>>(Variant::OP_LESS_EQUAL, Variant::STRING, Variant::STRING);
	register_op<OperatorEvaluatorLessEqual<Vector2, Vector2>>(Variant::OP_LESS_EQUAL, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorLessEqual<Vector2i, Vector2i>>(Variant::OP_LESS_EQUAL, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorLessEqual<Vector3, Vector3>>(Variant::OP_LESS_EQUAL, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorLessEqual<Vector3i, Vector3i>>(Variant::OP_LESS_EQUAL, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorLessEqual<::RID, ::RID>>(Variant::OP_LESS_EQUAL, Variant::RID, Variant::RID);
	register_op<OperatorEvaluatorLessEqual<Array, Array>>(Variant::OP_LESS_EQUAL, Variant::ARRAY, Variant::ARRAY);

	register_op<OperatorEvaluatorGreater<bool, bool>>(Variant::OP_GREATER, Variant::BOOL, Variant::BOOL);
	register_op<OperatorEvaluatorGreater<int64_t, int64_t>>(Variant::OP_GREATER, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorGreater<int64_t, double>>(Variant::OP_GREATER, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorGreater<double, int64_t>>(Variant::OP_GREATER, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorGreater<double, double>>(Variant::OP_GREATER, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorGreater<String, String>>(Variant::OP_GREATER, Variant::STRING, Variant::STRING);
	register_op<OperatorEvaluatorGreater<Vector2, Vector2>>(Variant::OP_GREATER, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorGreater<Vector2i, Vector2i>>(Variant::OP_GREATER, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorGreater<Vector3, Vector3>>(Variant::OP_GREATER, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorGreater<Vector3i, Vector3i>>(Variant::OP_GREATER, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorGreater<::RID, ::RID>>(Variant::OP_GREATER, Variant::RID, Variant::RID);
	register_op<OperatorEvaluatorGreater<Array, Array>>(Variant::OP_GREATER, Variant::ARRAY, Variant::ARRAY);

	register_op<OperatorEvaluatorGreaterEqual<int64_t, int64_t>>(Variant::OP_GREATER_EQUAL, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorGreaterEqual<int64_t, double>>(Variant::OP_GREATER_EQUAL, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorGreaterEqual<double, int64_t>>(Variant::OP_GREATER_EQUAL, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorGreaterEqual<double, double>>(Variant::OP_GREATER_EQUAL, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorGreaterEqual<String, String>>(Variant::OP_GREATER_EQUAL, Variant::STRING, Variant::STRING);
	register_op<OperatorEvaluatorGreaterEqual<Vector2, Vector2>>(Variant::OP_GREATER_EQUAL, Variant::VECTOR2, Variant::VECTOR2);
	register_op<OperatorEvaluatorGreaterEqual<Vector2i, Vector2i>>(Variant::OP_GREATER_EQUAL, Variant::VECTOR2I, Variant::VECTOR2I);
	register_op<OperatorEvaluatorGreaterEqual<Vector3, Vector3>>(Variant::OP_GREATER_EQUAL, Variant::VECTOR3, Variant::VECTOR3);
	register_op<OperatorEvaluatorGreaterEqual<Vector3i, Vector3i>>(Variant::OP_GREATER_EQUAL, Variant::VECTOR3I, Variant::VECTOR3I);
	register_op<OperatorEvaluatorGreaterEqual<::RID, ::RID>>(Variant::OP_GREATER_EQUAL, Variant::RID, Variant::RID);
	register_op<OperatorEvaluatorGreaterEqual<Array, Array>>(Variant::OP_GREATER_EQUAL, Variant::ARRAY, Variant::ARRAY);

	register_op<OperatorEvaluatorAlwaysFalse<Variant::OP_OR, Variant::NIL, Variant::NIL>>(Variant::OP_OR, Variant::NIL, Variant::NIL);

	// OR
	register_op<OperatorEvaluatorNilXBoolOr>(Variant::OP_OR, Variant::NIL, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXNilOr>(Variant::OP_OR, Variant::BOOL, Variant::NIL);
	register_op<OperatorEvaluatorNilXIntOr>(Variant::OP_OR, Variant::NIL, Variant::INT);
	register_op<OperatorEvaluatorIntXNilOr>(Variant::OP_OR, Variant::INT, Variant::NIL);
	register_op<OperatorEvaluatorNilXFloatOr>(Variant::OP_OR, Variant::NIL, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXNilOr>(Variant::OP_OR, Variant::FLOAT, Variant::NIL);
	register_op<OperatorEvaluatorNilXObjectOr>(Variant::OP_OR, Variant::NIL, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXNilOr>(Variant::OP_OR, Variant::OBJECT, Variant::NIL);

	register_op<OperatorEvaluatorBoolXBoolOr>(Variant::OP_OR, Variant::BOOL, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXIntOr>(Variant::OP_OR, Variant::BOOL, Variant::INT);
	register_op<OperatorEvaluatorIntXBoolOr>(Variant::OP_OR, Variant::INT, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXFloatOr>(Variant::OP_OR, Variant::BOOL, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXBoolOr>(Variant::OP_OR, Variant::FLOAT, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXObjectOr>(Variant::OP_OR, Variant::BOOL, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXBoolOr>(Variant::OP_OR, Variant::OBJECT, Variant::BOOL);

	register_op<OperatorEvaluatorIntXIntOr>(Variant::OP_OR, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorIntXFloatOr>(Variant::OP_OR, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXIntOr>(Variant::OP_OR, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorIntXObjectOr>(Variant::OP_OR, Variant::INT, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXIntOr>(Variant::OP_OR, Variant::OBJECT, Variant::INT);

	register_op<OperatorEvaluatorFloatXFloatOr>(Variant::OP_OR, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXObjectOr>(Variant::OP_OR, Variant::FLOAT, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXFloatOr>(Variant::OP_OR, Variant::OBJECT, Variant::FLOAT);
	register_op<OperatorEvaluatorObjectXObjectOr>(Variant::OP_OR, Variant::OBJECT, Variant::OBJECT);

	// AND
	register_op<OperatorEvaluatorNilXBoolAnd>(Variant::OP_AND, Variant::NIL, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXNilAnd>(Variant::OP_AND, Variant::BOOL, Variant::NIL);
	register_op<OperatorEvaluatorNilXIntAnd>(Variant::OP_AND, Variant::NIL, Variant::INT);
	register_op<OperatorEvaluatorIntXNilAnd>(Variant::OP_AND, Variant::INT, Variant::NIL);
	register_op<OperatorEvaluatorNilXFloatAnd>(Variant::OP_AND, Variant::NIL, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXNilAnd>(Variant::OP_AND, Variant::FLOAT, Variant::NIL);
	register_op<OperatorEvaluatorNilXObjectAnd>(Variant::OP_AND, Variant::NIL, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXNilAnd>(Variant::OP_AND, Variant::OBJECT, Variant::NIL);

	register_op<OperatorEvaluatorBoolXBoolAnd>(Variant::OP_AND, Variant::BOOL, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXIntAnd>(Variant::OP_AND, Variant::BOOL, Variant::INT);
	register_op<OperatorEvaluatorIntXBoolAnd>(Variant::OP_AND, Variant::INT, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXFloatAnd>(Variant::OP_AND, Variant::BOOL, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXBoolAnd>(Variant::OP_AND, Variant::FLOAT, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXObjectAnd>(Variant::OP_AND, Variant::BOOL, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXBoolAnd>(Variant::OP_AND, Variant::OBJECT, Variant::BOOL);

	register_op<OperatorEvaluatorIntXIntAnd>(Variant::OP_AND, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorIntXFloatAnd>(Variant::OP_AND, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXIntAnd>(Variant::OP_AND, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorIntXObjectAnd>(Variant::OP_AND, Variant::INT, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXIntAnd>(Variant::OP_AND, Variant::OBJECT, Variant::INT);

	register_op<OperatorEvaluatorFloatXFloatAnd>(Variant::OP_AND, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXObjectAnd>(Variant::OP_AND, Variant::FLOAT, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXFloatAnd>(Variant::OP_AND, Variant::OBJECT, Variant::FLOAT);
	register_op<OperatorEvaluatorObjectXObjectAnd>(Variant::OP_AND, Variant::OBJECT, Variant::OBJECT);

	// XOR
	register_op<OperatorEvaluatorNilXBoolXor>(Variant::OP_XOR, Variant::NIL, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXNilXor>(Variant::OP_XOR, Variant::BOOL, Variant::NIL);
	register_op<OperatorEvaluatorNilXIntXor>(Variant::OP_XOR, Variant::NIL, Variant::INT);
	register_op<OperatorEvaluatorIntXNilXor>(Variant::OP_XOR, Variant::INT, Variant::NIL);
	register_op<OperatorEvaluatorNilXFloatXor>(Variant::OP_XOR, Variant::NIL, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXNilXor>(Variant::OP_XOR, Variant::FLOAT, Variant::NIL);
	register_op<OperatorEvaluatorNilXObjectXor>(Variant::OP_XOR, Variant::NIL, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXNilXor>(Variant::OP_XOR, Variant::OBJECT, Variant::NIL);

	register_op<OperatorEvaluatorBoolXBoolXor>(Variant::OP_XOR, Variant::BOOL, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXIntXor>(Variant::OP_XOR, Variant::BOOL, Variant::INT);
	register_op<OperatorEvaluatorIntXBoolXor>(Variant::OP_XOR, Variant::INT, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXFloatXor>(Variant::OP_XOR, Variant::BOOL, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXBoolXor>(Variant::OP_XOR, Variant::FLOAT, Variant::BOOL);
	register_op<OperatorEvaluatorBoolXObjectXor>(Variant::OP_XOR, Variant::BOOL, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXBoolXor>(Variant::OP_XOR, Variant::OBJECT, Variant::BOOL);

	register_op<OperatorEvaluatorIntXIntXor>(Variant::OP_XOR, Variant::INT, Variant::INT);
	register_op<OperatorEvaluatorIntXFloatXor>(Variant::OP_XOR, Variant::INT, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXIntXor>(Variant::OP_XOR, Variant::FLOAT, Variant::INT);
	register_op<OperatorEvaluatorIntXObjectXor>(Variant::OP_XOR, Variant::INT, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXIntXor>(Variant::OP_XOR, Variant::OBJECT, Variant::INT);

	register_op<OperatorEvaluatorFloatXFloatXor>(Variant::OP_XOR, Variant::FLOAT, Variant::FLOAT);
	register_op<OperatorEvaluatorFloatXObjectXor>(Variant::OP_XOR, Variant::FLOAT, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectXFloatXor>(Variant::OP_XOR, Variant::OBJECT, Variant::FLOAT);
	register_op<OperatorEvaluatorObjectXObjectXor>(Variant::OP_XOR, Variant::OBJECT, Variant::OBJECT);

	register_op<OperatorEvaluatorAlwaysTrue<Variant::OP_NOT, Variant::NIL, Variant::NIL>>(Variant::OP_NOT, Variant::NIL, Variant::NIL);
	register_op<OperatorEvaluatorNotBool>(Variant::OP_NOT, Variant::BOOL, Variant::NIL);
	register_op<OperatorEvaluatorNotInt>(Variant::OP_NOT, Variant::INT, Variant::NIL);
	register_op<OperatorEvaluatorNotFloat>(Variant::OP_NOT, Variant::FLOAT, Variant::NIL);
	register_op<OperatorEvaluatorNotObject>(Variant::OP_NOT, Variant::OBJECT, Variant::NIL);

	register_op<OperatorEvaluatorInStringFind>(Variant::OP_IN, Variant::STRING, Variant::STRING);

	register_op<OperatorEvaluatorInDictionaryHasNil>(Variant::OP_IN, Variant::NIL, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<bool>>(Variant::OP_IN, Variant::BOOL, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<int64_t>>(Variant::OP_IN, Variant::INT, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<double>>(Variant::OP_IN, Variant::FLOAT, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<String>>(Variant::OP_IN, Variant::STRING, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector2>>(Variant::OP_IN, Variant::VECTOR2, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector2i>>(Variant::OP_IN, Variant::VECTOR2I, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Rect2>>(Variant::OP_IN, Variant::RECT2, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Rect2i>>(Variant::OP_IN, Variant::RECT2I, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector3>>(Variant::OP_IN, Variant::VECTOR3, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Vector3i>>(Variant::OP_IN, Variant::VECTOR3I, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Transform2D>>(Variant::OP_IN, Variant::TRANSFORM2D, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Plane>>(Variant::OP_IN, Variant::PLANE, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Quat>>(Variant::OP_IN, Variant::QUAT, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<::AABB>>(Variant::OP_IN, Variant::AABB, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Basis>>(Variant::OP_IN, Variant::BASIS, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Transform>>(Variant::OP_IN, Variant::TRANSFORM, Variant::DICTIONARY);

	register_op<OperatorEvaluatorInDictionaryHas<Color>>(Variant::OP_IN, Variant::COLOR, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<StringName>>(Variant::OP_IN, Variant::STRING_NAME, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<NodePath>>(Variant::OP_IN, Variant::NODE_PATH, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHasObject>(Variant::OP_IN, Variant::OBJECT, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Callable>>(Variant::OP_IN, Variant::CALLABLE, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Signal>>(Variant::OP_IN, Variant::SIGNAL, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Dictionary>>(Variant::OP_IN, Variant::DICTIONARY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<Array>>(Variant::OP_IN, Variant::ARRAY, Variant::DICTIONARY);

	register_op<OperatorEvaluatorInDictionaryHas<PackedByteArray>>(Variant::OP_IN, Variant::PACKED_BYTE_ARRAY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedInt32Array>>(Variant::OP_IN, Variant::PACKED_INT32_ARRAY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedInt64Array>>(Variant::OP_IN, Variant::PACKED_INT64_ARRAY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedFloat32Array>>(Variant::OP_IN, Variant::PACKED_FLOAT32_ARRAY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedFloat64Array>>(Variant::OP_IN, Variant::PACKED_FLOAT64_ARRAY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedStringArray>>(Variant::OP_IN, Variant::PACKED_STRING_ARRAY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedVector2Array>>(Variant::OP_IN, Variant::PACKED_VECTOR2_ARRAY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedVector3Array>>(Variant::OP_IN, Variant::PACKED_VECTOR3_ARRAY, Variant::DICTIONARY);
	register_op<OperatorEvaluatorInDictionaryHas<PackedColorArray>>(Variant::OP_IN, Variant::PACKED_COLOR_ARRAY, Variant::DICTIONARY);

	register_op<OperatorEvaluatorInArrayFindNil>(Variant::OP_IN, Variant::NIL, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<bool, Array>>(Variant::OP_IN, Variant::BOOL, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<int64_t, Array>>(Variant::OP_IN, Variant::INT, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<double, Array>>(Variant::OP_IN, Variant::FLOAT, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<String, Array>>(Variant::OP_IN, Variant::STRING, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector2, Array>>(Variant::OP_IN, Variant::VECTOR2, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector2i, Array>>(Variant::OP_IN, Variant::VECTOR2I, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Rect2, Array>>(Variant::OP_IN, Variant::RECT2, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Rect2i, Array>>(Variant::OP_IN, Variant::RECT2I, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector3, Array>>(Variant::OP_IN, Variant::VECTOR3, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector3i, Array>>(Variant::OP_IN, Variant::VECTOR3I, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Transform2D, Array>>(Variant::OP_IN, Variant::TRANSFORM2D, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Plane, Array>>(Variant::OP_IN, Variant::PLANE, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Quat, Array>>(Variant::OP_IN, Variant::QUAT, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<::AABB, Array>>(Variant::OP_IN, Variant::AABB, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Basis, Array>>(Variant::OP_IN, Variant::BASIS, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Transform, Array>>(Variant::OP_IN, Variant::TRANSFORM, Variant::ARRAY);

	register_op<OperatorEvaluatorInArrayFind<Color, Array>>(Variant::OP_IN, Variant::COLOR, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<StringName, Array>>(Variant::OP_IN, Variant::STRING_NAME, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<NodePath, Array>>(Variant::OP_IN, Variant::NODE_PATH, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFindObject>(Variant::OP_IN, Variant::OBJECT, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Callable, Array>>(Variant::OP_IN, Variant::CALLABLE, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Signal, Array>>(Variant::OP_IN, Variant::SIGNAL, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Dictionary, Array>>(Variant::OP_IN, Variant::DICTIONARY, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Array, Array>>(Variant::OP_IN, Variant::ARRAY, Variant::ARRAY);

	register_op<OperatorEvaluatorInArrayFind<PackedByteArray, Array>>(Variant::OP_IN, Variant::PACKED_BYTE_ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedInt32Array, Array>>(Variant::OP_IN, Variant::PACKED_INT32_ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedInt64Array, Array>>(Variant::OP_IN, Variant::PACKED_INT64_ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedFloat32Array, Array>>(Variant::OP_IN, Variant::PACKED_FLOAT32_ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedFloat64Array, Array>>(Variant::OP_IN, Variant::PACKED_FLOAT64_ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedStringArray, Array>>(Variant::OP_IN, Variant::PACKED_STRING_ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedVector2Array, Array>>(Variant::OP_IN, Variant::PACKED_VECTOR2_ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedVector3Array, Array>>(Variant::OP_IN, Variant::PACKED_VECTOR3_ARRAY, Variant::ARRAY);
	register_op<OperatorEvaluatorInArrayFind<PackedColorArray, Array>>(Variant::OP_IN, Variant::PACKED_COLOR_ARRAY, Variant::ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedByteArray>>(Variant::OP_IN, Variant::INT, Variant::PACKED_BYTE_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedByteArray>>(Variant::OP_IN, Variant::FLOAT, Variant::PACKED_BYTE_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedInt32Array>>(Variant::OP_IN, Variant::INT, Variant::PACKED_INT32_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedInt32Array>>(Variant::OP_IN, Variant::FLOAT, Variant::PACKED_INT32_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedInt64Array>>(Variant::OP_IN, Variant::INT, Variant::PACKED_INT64_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedInt64Array>>(Variant::OP_IN, Variant::FLOAT, Variant::PACKED_INT64_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedFloat32Array>>(Variant::OP_IN, Variant::INT, Variant::PACKED_FLOAT32_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedFloat32Array>>(Variant::OP_IN, Variant::FLOAT, Variant::PACKED_FLOAT32_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<int, PackedFloat64Array>>(Variant::OP_IN, Variant::INT, Variant::PACKED_FLOAT64_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<float, PackedFloat64Array>>(Variant::OP_IN, Variant::FLOAT, Variant::PACKED_FLOAT64_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<String, PackedStringArray>>(Variant::OP_IN, Variant::STRING, Variant::PACKED_STRING_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<Vector2, PackedVector2Array>>(Variant::OP_IN, Variant::VECTOR2, Variant::PACKED_VECTOR2_ARRAY);
	register_op<OperatorEvaluatorInArrayFind<Vector3, PackedVector3Array>>(Variant::OP_IN, Variant::VECTOR3, Variant::PACKED_VECTOR3_ARRAY);

	register_op<OperatorEvaluatorInArrayFind<Color, PackedColorArray>>(Variant::OP_IN, Variant::COLOR, Variant::PACKED_COLOR_ARRAY);

	register_op<OperatorEvaluatorObjectHasPropertyString>(Variant::OP_IN, Variant::STRING, Variant::OBJECT);
	register_op<OperatorEvaluatorObjectHasPropertyStringName>(Variant::OP_IN, Variant::STRING_NAME, Variant::OBJECT);
}

void Variant::_unregister_variant_operators() {
}

void Variant::evaluate(const Operator &p_op, const Variant &p_a,
		const Variant &p_b, Variant &r_ret, bool &r_valid) {
	ERR_FAIL_INDEX(p_op, Variant::OP_MAX);
	Variant::Type type_a = p_a.get_type();
	Variant::Type type_b = p_b.get_type();
	ERR_FAIL_INDEX(type_a, Variant::VARIANT_MAX);
	ERR_FAIL_INDEX(type_b, Variant::VARIANT_MAX);

	VariantEvaluatorFunction ev = operator_evaluator_table[p_op][type_a][type_b];
	if (unlikely(!ev)) {
		r_valid = false;
		r_ret = Variant();
		return;
	}

	ev(p_a, p_b, &r_ret, r_valid);
}

Variant::Type Variant::get_operator_return_type(Operator p_operator, Type p_type_a, Type p_type_b) {
	ERR_FAIL_INDEX_V(p_operator, Variant::OP_MAX, Variant::NIL);
	ERR_FAIL_INDEX_V(p_type_a, Variant::VARIANT_MAX, Variant::NIL);
	ERR_FAIL_INDEX_V(p_type_b, Variant::VARIANT_MAX, Variant::NIL);

	return operator_return_type_table[p_operator][p_type_a][p_type_b];
}

Variant::ValidatedOperatorEvaluator Variant::get_validated_operator_evaluator(Operator p_operator, Type p_type_a, Type p_type_b) {
	ERR_FAIL_INDEX_V(p_operator, Variant::OP_MAX, nullptr);
	ERR_FAIL_INDEX_V(p_type_a, Variant::VARIANT_MAX, nullptr);
	ERR_FAIL_INDEX_V(p_type_b, Variant::VARIANT_MAX, nullptr);
	return validated_operator_evaluator_table[p_operator][p_type_a][p_type_b];
}

Variant::PTROperatorEvaluator Variant::get_ptr_operator_evaluator(Operator p_operator, Type p_type_a, Type p_type_b) {
	ERR_FAIL_INDEX_V(p_operator, Variant::OP_MAX, nullptr);
	ERR_FAIL_INDEX_V(p_type_a, Variant::VARIANT_MAX, nullptr);
	ERR_FAIL_INDEX_V(p_type_b, Variant::VARIANT_MAX, nullptr);
	return ptr_operator_evaluator_table[p_operator][p_type_a][p_type_b];
}

static const char *_op_names[Variant::OP_MAX] = {
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
	"-",
	"+",
	"%",
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

String Variant::get_operator_name(Operator p_op) {
	ERR_FAIL_INDEX_V(p_op, OP_MAX, "");
	return _op_names[p_op];
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
	evaluate(OP_IN, p_index, *this, ret, valid);
	if (r_valid) {
		*r_valid = valid;
		return false;
	}
	ERR_FAIL_COND_V(ret.type != BOOL, false);
	return *VariantGetInternalPtr<bool>::get_ptr(&ret);
}
