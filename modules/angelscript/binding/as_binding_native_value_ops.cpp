/**************************************************************************/
/*  as_binding_native_value_ops.cpp                                       */
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

#include "as_binding_native_value_ops.h"

#include "as_binding_value_types.h"

#include "core/error/error_macros.h"
#include "core/math/aabb.h"
#include "core/math/basis.h"
#include "core/math/color.h"
#include "core/math/plane.h"
#include "core/math/projection.h"
#include "core/math/quaternion.h"
#include "core/math/rect2.h"
#include "core/math/rect2i.h"
#include "core/math/transform_2d.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2.h"
#include "core/math/vector2i.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/math/vector4.h"
#include "core/math/vector4i.h"
#include "core/variant/variant_internal.h"

#include <angelscript.h>

#include <new>
#include <type_traits>
#include <utility>

namespace {

template <typename T>
struct OpAdd {
	static T apply(const T &p_a, const T &p_b) { return p_a + p_b; }
};

template <typename T>
struct OpSub {
	static T apply(const T &p_a, const T &p_b) { return p_a - p_b; }
};

template <typename T>
struct OpMul {
	static T apply(const T &p_a, const T &p_b) { return p_a * p_b; }
};

template <typename T>
struct OpDiv {
	static T apply(const T &p_a, const T &p_b) { return p_a / p_b; }
};

// 二元运算：返回新对象。原生存储下 T 为非 POD（含构造/析构/赋值/拷贝构造子标志），
// AS 以“内存返回”约定传入隐藏返回缓冲 ret（见 spec §4.3 / §7.1）。
template <typename T, typename Op>
void thunk_binop(T *ret, const T *self, const T *other) {
	ASNativeValueOps::note_thunk_call();
	new (ret) T(Op::apply(*self, *other));
}

// opAssign：原地改写 self 的原生存储。AS 侧声明为返回引用，
// 故原生函数返回 self 的地址以匹配引用返回的 ABI（见 spec §4.4）。
template <typename T>
T *thunk_assign(T *self, const T *other) {
	ASNativeValueOps::note_thunk_call();
	*self = *other;
	return self;
}

// opEquals：比较两个存储的原生值。AS 侧声明为返回 bool，按值返回即可（见 spec §4.5）。
template <typename T>
bool thunk_equals(const T *self, const T *other) {
	ASNativeValueOps::note_thunk_call();
	return *self == *other;
}

// 运算符可用性探测：值类型不一定支持全部六个运算符（如 Projection 无 `/`、Rect2 无四则）。
// 只有真正存在的运算符才实例化对应 thunk；其余返回 nullptr，由调用方回退 generic。
template <typename T, typename = void>
struct has_add : std::false_type {};
template <typename T>
struct has_add<T, std::void_t<decltype(std::declval<T>() + std::declval<T>())>> : std::true_type {};

template <typename T, typename = void>
struct has_sub : std::false_type {};
template <typename T>
struct has_sub<T, std::void_t<decltype(std::declval<T>() - std::declval<T>())>> : std::true_type {};

template <typename T, typename = void>
struct has_mul : std::false_type {};
template <typename T>
struct has_mul<T, std::void_t<decltype(std::declval<T>() * std::declval<T>())>> : std::true_type {};

template <typename T, typename = void>
struct has_div : std::false_type {};
template <typename T>
struct has_div<T, std::void_t<decltype(std::declval<T>() / std::declval<T>())>> : std::true_type {};

template <typename T, typename = void>
struct has_eq : std::false_type {};
template <typename T>
struct has_eq<T, std::void_t<decltype(std::declval<T>() == std::declval<T>())>> : std::true_type {};

// 按运算符名取该类型的原生 thunk 函数指针；该类型不支持或未覆盖的运算符返回 nullptr。
template <typename T>
void *thunk_for_op(const String &p_op) {
	if constexpr (has_add<T>::value) {
		if (p_op == "opAdd") {
			return (void *)&thunk_binop<T, OpAdd<T>>;
		}
	}
	if constexpr (has_sub<T>::value) {
		if (p_op == "opSub") {
			return (void *)&thunk_binop<T, OpSub<T>>;
		}
	}
	if constexpr (has_mul<T>::value) {
		if (p_op == "opMul") {
			return (void *)&thunk_binop<T, OpMul<T>>;
		}
	}
	if constexpr (has_div<T>::value) {
		if (p_op == "opDiv") {
			return (void *)&thunk_binop<T, OpDiv<T>>;
		}
	}
	if (p_op == "opAssign") {
		return (void *)&thunk_assign<T>;
	}
	if constexpr (has_eq<T>::value) {
		if (p_op == "opEquals") {
			return (void *)&thunk_equals<T>;
		}
	}
	return nullptr;
}

// 无参标量方法：直接返回原生标量（见 spec §5）。
template <typename T, typename R, auto M>
R thunk_method_scalar(const T *self) {
	ASNativeValueOps::note_thunk_call();
	return (R)((self->*M)());
}

// 无参对象方法：返回新值类型对象，走内存返回（见 spec §5）。
template <typename T, typename R, auto M>
void thunk_method_obj(T *ret, const T *self) {
	ASNativeValueOps::note_thunk_call();
	new (ret) T((R)((self->*M)()));
}

// 单参标量方法：形参为同类型值对象（按 const 引用传入，见 spec §5）。
template <typename T, typename R, auto M>
R thunk_method1_scalar(const T *self, const T *arg) {
	ASNativeValueOps::note_thunk_call();
	return (R)((self->*M)(*arg));
}

// 单参对象方法：形参为同类型值对象，返回新值类型对象，走内存返回（见 spec §5）。
template <typename T, typename R, auto M>
void thunk_method1_obj(T *ret, const T *self, const T *arg) {
	ASNativeValueOps::note_thunk_call();
	new (ret) T((R)((self->*M)(*arg)));
}

// 单参对象+标量方法：形参为同类型值对象与实数权重，返回新值类型对象（见 spec §5）。
template <typename T, typename R, auto M>
void thunk_method1_obj_f(T *ret, const T *self, const T *arg, double weight) {
	ASNativeValueOps::note_thunk_call();
	new (ret) T((R)((self->*M)(*arg, (real_t)weight)));
}

// 双参对象方法：两个同类型值对象形参，返回新值类型对象（见 spec §5）。
template <typename T, typename R, auto M>
void thunk_method2_obj(T *ret, const T *self, const T *a, const T *b) {
	ASNativeValueOps::note_thunk_call();
	new (ret) T((R)((self->*M)(*a, *b)));
}

// 单标量参对象方法：形参为实数，返回新值类型对象（见 spec §5）。
template <typename T, typename R, auto M>
void thunk_method_f_obj(T *ret, const T *self, double arg) {
	ASNativeValueOps::note_thunk_call();
	new (ret) T((R)((self->*M)((real_t)arg)));
}

// Boxed lifecycle: keep Variant storage; do not change VM flags or ownership.
template <typename T>
void thunk_boxed_default(Variant *self) {
	ASNativeValueOps::note_thunk_call();
	new (self) Variant(T());
}

void thunk_boxed_copy(Variant *self, const Variant *other) {
	ASNativeValueOps::note_thunk_call();
	new (self) Variant(*other);
}

void thunk_boxed_destroy(Variant *self) {
	ASNativeValueOps::note_thunk_call();
	self->~Variant();
}

template <typename T>
void thunk_variant_construct(Variant *self, T value) {
	ASNativeValueOps::note_thunk_call();
	new (self) Variant(value);
}

template <typename T>
T thunk_variant_convert(const Variant *self) {
	ASNativeValueOps::note_thunk_call();
	return (T)*self;
}

void thunk_variant_to_string(Variant *ret, const Variant *self) {
	ASNativeValueOps::note_thunk_call();
	new (ret) Variant((String)*self);
}

// Packed scalar arrays: access the same Vector inside PackedArrayRef as Godot's
// validated methods. Do NOT copy it locally: that adds a ref and forces COW.
// Variant copies intentionally share PackedArrayRef; duplicate() separates it.
template <typename T>
int64_t thunk_packed_size(const Variant *self) {
	ASNativeValueOps::note_thunk_call();
	return VariantInternalAccessor<Vector<T>>::get(self).size();
}

template <typename T, typename R>
R thunk_packed_get(const Variant *self, int64_t index) {
	ASNativeValueOps::note_thunk_call();
	const Vector<T> &values = VariantInternalAccessor<Vector<T>>::get(self);
	if (index < 0) {
		index += values.size();
	}
	if (index < 0 || index >= values.size()) {
		return R(0);
	}
	return (R)values[index];
}

template <typename T, typename R>
void thunk_packed_set(Variant *self, int64_t index, R value) {
	ASNativeValueOps::note_thunk_call();
	Vector<T> &values = VariantInternalAccessor<Vector<T>>::get(self);
	if (index < 0) {
		index += values.size();
	}
	if (index < 0 || index >= values.size()) {
		return;
	}
	values.write[index] = (T)value;
}

template <typename T, typename R>
void thunk_packed_push(Variant *self, R value) {
	ASNativeValueOps::note_thunk_call();
	VariantInternalAccessor<Vector<T>>::get(self).push_back((T)value);
}

// ---------- String（存储为 Variant）的原生运算符 ----------
// Godot String 已带 asOBJ_APP_CLASS，可原生按值返回；其存储是 sizeof(Variant)，
// 故 thunk 直接读写 Variant 存储。绕过 generic 跳板完成拼接/赋值/相等。
void thunk_string_add(Variant *ret, const Variant *self, const Variant *other) {
	ASNativeValueOps::note_thunk_call();
	new (ret) Variant(VariantInternalAccessor<String>::get(self) + VariantInternalAccessor<String>::get(other));
}

Variant *thunk_string_assign(Variant *self, const Variant *other) {
	ASNativeValueOps::note_thunk_call();
	*self = *other;
	return self;
}

bool thunk_string_equals(const Variant *self, const Variant *other) {
	ASNativeValueOps::note_thunk_call();
	return VariantInternalAccessor<String>::get(self) == VariantInternalAccessor<String>::get(other);
}

void *thunk_for_string_op(const String &p_op) {
	if (p_op == "opAdd") {
		return (void *)&thunk_string_add;
	}
	if (p_op == "opAssign") {
		return (void *)&thunk_string_assign;
	}
	if (p_op == "opEquals") {
		return (void *)&thunk_string_equals;
	}
	return nullptr;
}

// Direct lowercase string arguments still accept both literal factory pointers
// and pointer-sized variable slots. The shared decoder owns that distinction.
void thunk_literal_construct(Variant *self, const void *arg) {
	ASNativeValueOps::note_thunk_call();
	new (self) Variant(ASBindingValueTypes::string_from_argument(arg));
}

void thunk_literal_add(Variant *ret, const Variant *self, const void *arg) {
	ASNativeValueOps::note_thunk_call();
	new (ret) Variant(VariantInternalAccessor<String>::get(self) + ASBindingValueTypes::string_from_argument(arg));
}

// Direct Array core methods retain Godot's typed/read-only checks. Do not
// access its backing Vector or construct a temporary Variant return for scalars.
template <auto M, typename R>
R thunk_array0(const Variant *self) {
	ASNativeValueOps::note_thunk_call();
	return (R)(VariantInternalAccessor<Array>::get(self).*M)();
}

template <auto M>
void thunk_array_mutate0(Variant *self) {
	ASNativeValueOps::note_thunk_call();
	(VariantInternalAccessor<Array>::get(self).*M)();
}

template <auto M>
void thunk_array_mutate1(Variant *self, const Variant *arg) {
	ASNativeValueOps::note_thunk_call();
	(VariantInternalAccessor<Array>::get(self).*M)(*arg);
}

template <auto M, typename R>
R thunk_array1(const Variant *self, const Variant *arg) {
	ASNativeValueOps::note_thunk_call();
	return (R)(VariantInternalAccessor<Array>::get(self).*M)(*arg);
}

template <auto M>
void thunk_array_return(Variant *ret, Variant *self) {
	ASNativeValueOps::note_thunk_call();
	new (ret) Variant((VariantInternalAccessor<Array>::get(self).*M)());
}

// ---------- Array 的 Variant 元素索引 ----------
// 需 Variant 注册为 asOBJ_APP_CLASS，按值原生返回才被 AS 配置校验接受。
// 元素为 Variant（引用计数负载），thunk 就地构造返回槽。
void thunk_variant_index_get(Variant *ret, const Variant *self, int64_t index) {
	ASNativeValueOps::note_thunk_call();
	const Array &values = VariantInternalAccessor<Array>::get(self);
	if (index < 0) {
		index += values.size();
	}
	if (index < 0 || index >= values.size()) {
		new (ret) Variant();
		return;
	}
	new (ret) Variant(values[index]);
}

void thunk_variant_index_set(Variant *self, int64_t index, const Variant *value) {
	ASNativeValueOps::note_thunk_call();
	bool valid = false;
	bool oob = false;
	self->set_indexed(index, *value, valid, oob);
}

// ---------- Dictionary 的 Variant 键值索引 ----------
// Dictionary 的 key/value 都是 Variant，不能复用整数索引 thunk；这里保留 Godot
// get_keyed/set_keyed 的 valid 语义，并在返回槽中正确构造引用计数负载。
void thunk_variant_keyed_get(Variant *ret, const Variant *self, const Variant *key) {
	ASNativeValueOps::note_thunk_call();
	// Const getptr retains typed-key validation without inserting a missing key.
	const Dictionary &values = VariantInternalAccessor<Dictionary>::get(self);
	const Variant *value = values.getptr(*key);
	if (value) {
		new (ret) Variant(*value);
	} else {
		new (ret) Variant();
	}
}

void thunk_variant_keyed_set(Variant *self, const Variant *key, const Variant *value) {
	ASNativeValueOps::note_thunk_call();
	bool valid = false;
	self->set_keyed(*key, *value, valid);
}

// ---------- A2：Array/Dictionary 安全内建方法白名单 ----------
// 函数指针无法携带方法名字符串，故用 traits 结构承载名字以参数化 thunk 模板。
struct BmSize { static const char *get() { return "size"; } };
struct BmIsEmpty { static const char *get() { return "is_empty"; } };
struct BmHash { static const char *get() { return "hash"; } };
struct BmClear { static const char *get() { return "clear"; } };
struct BmPushBack { static const char *get() { return "push_back"; } };
struct BmPushFront { static const char *get() { return "push_front"; } };
struct BmAppend { static const char *get() { return "append"; } };
struct BmHas { static const char *get() { return "has"; } };
struct BmCount { static const char *get() { return "count"; } };
struct BmFront { static const char *get() { return "front"; } };
struct BmBack { static const char *get() { return "back"; } };
struct BmPopBack { static const char *get() { return "pop_back"; } };
struct BmPopFront { static const char *get() { return "pop_front"; } };
struct BmErase { static const char *get() { return "erase"; } };

// 标量/布尔返回槽必须预先初始化为对应 Variant 类型：VariantInternalAccessor<T>::set
// 只写 Variant 的 _data 存储、不设置 type 字段（generic_value_call 因此在调用前
// 会 VariantInternal::initialize）。这里用等价的公开赋值预置，否则返回值会被当作 NIL 0。
template <typename R>
static void init_bm_scalar_slot(Variant &r) {
	if constexpr (std::is_same_v<R, bool>) {
		r = false;
	} else {
		r = (int64_t)0;
	}
}

// 无参、标量/布尔返回：AS 侧走寄存器返回，self 是首参。
template <Variant::Type T, typename BM, typename R>
R thunk_bm0_scalar(const Variant *self) {
	ASNativeValueOps::note_thunk_call();
	static Variant::ValidatedBuiltInMethod m = Variant::get_validated_builtin_method(T, BM::get());
	Variant r;
	init_bm_scalar_slot<R>(r);
	if (m) {
		m(const_cast<Variant *>(self), nullptr, 0, &r);
	}
	return (R)r;
}

// 无参、Variant 按值返回：隐式返回指针在最前，就地构造返回槽。先写入本地 Variant，
// 再 placement-new 拷贝到返回槽，避免对未构造的槽做赋值（与 generic marshal 一致）。
template <Variant::Type T, typename BM>
void thunk_bm0_variant(Variant *ret, Variant *self) {
	ASNativeValueOps::note_thunk_call();
	static Variant::ValidatedBuiltInMethod m = Variant::get_validated_builtin_method(T, BM::get());
	Variant r;
	if (m) {
		m(self, nullptr, 0, &r);
	}
	new (ret) Variant(r);
}

// 无参、void 返回。
template <Variant::Type T, typename BM>
void thunk_bm0_void(Variant *self) {
	ASNativeValueOps::note_thunk_call();
	static Variant::ValidatedBuiltInMethod m = Variant::get_validated_builtin_method(T, BM::get());
	Variant r;
	if (m) {
		m(self, nullptr, 0, &r);
	}
}

// 单个 Variant 参、标量/布尔返回。
template <Variant::Type T, typename BM, typename R>
R thunk_bm1_scalar(const Variant *self, const Variant *arg) {
	ASNativeValueOps::note_thunk_call();
	static Variant::ValidatedBuiltInMethod m = Variant::get_validated_builtin_method(T, BM::get());
	Variant r;
	init_bm_scalar_slot<R>(r);
	if (m) {
		const Variant *args[1] = { arg };
		m(const_cast<Variant *>(self), args, 1, &r);
	}
	return (R)r;
}

// 单个 Variant 参、void 返回。
template <Variant::Type T, typename BM>
void thunk_bm1_void(Variant *self, const Variant *arg) {
	ASNativeValueOps::note_thunk_call();
	static Variant::ValidatedBuiltInMethod m = Variant::get_validated_builtin_method(T, BM::get());
	Variant r;
	if (m) {
		const Variant *args[1] = { arg };
		m(self, args, 1, &r);
	}
}

} // namespace

namespace ASNativeValueOps {

static int g_native_registration_count = 0;
static int g_native_thunk_calls = 0;

bool try_add_boxed_lifecycle(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type) {
#if !AS_NATIVE_VALUE_OPS_ENABLED
	return false;
#else
	void *ctor = nullptr;
	switch (p_type) {
		case Variant::NIL:
			ctor = (void *)&thunk_boxed_default<Variant>;
			break;
		case Variant::STRING:
			ctor = (void *)&thunk_boxed_default<String>;
			break;
		case Variant::STRING_NAME:
			ctor = (void *)&thunk_boxed_default<StringName>;
			break;
		case Variant::ARRAY:
			ctor = (void *)&thunk_boxed_default<Array>;
			break;
		case Variant::DICTIONARY:
			ctor = (void *)&thunk_boxed_default<Dictionary>;
			break;
		case Variant::PACKED_BYTE_ARRAY:
			ctor = (void *)&thunk_boxed_default<Vector<uint8_t>>;
			break;
		case Variant::PACKED_INT32_ARRAY:
			ctor = (void *)&thunk_boxed_default<Vector<int32_t>>;
			break;
		case Variant::PACKED_INT64_ARRAY:
			ctor = (void *)&thunk_boxed_default<Vector<int64_t>>;
			break;
		case Variant::PACKED_FLOAT32_ARRAY:
			ctor = (void *)&thunk_boxed_default<Vector<float>>;
			break;
		case Variant::PACKED_FLOAT64_ARRAY:
			ctor = (void *)&thunk_boxed_default<Vector<double>>;
			break;
		default:
			return false;
	}
	CharString name = p_type_name.utf8();
	int a = p_engine->RegisterObjectBehaviour(name.get_data(), asBEHAVE_CONSTRUCT, "void f()", asFUNCTION(ctor), asCALL_CDECL_OBJFIRST);
	int b = p_engine->RegisterObjectBehaviour(name.get_data(), asBEHAVE_CONSTRUCT, ("void f(const " + p_type_name + " &in)").utf8().get_data(), asFUNCTION(thunk_boxed_copy), asCALL_CDECL_OBJFIRST);
	int c = p_engine->RegisterObjectBehaviour(name.get_data(), asBEHAVE_DESTRUCT, "void f()", asFUNCTION(thunk_boxed_destroy), asCALL_CDECL_OBJFIRST);
	ERR_FAIL_COND_V(a < 0 || b < 0 || c < 0, false);
	g_native_registration_count += 3;
	return true;
#endif
}

bool try_add_variant_conversions(asIScriptEngine *p_engine) {
#if !AS_NATIVE_VALUE_OPS_ENABLED
	return false;
#else
	struct Entry {
		const char *decl;
		void *fn;
	} ctors[] = {
		{ "void f(int64)", (void *)&thunk_variant_construct<int64_t> },
		{ "void f(double)", (void *)&thunk_variant_construct<double> },
		{ "void f(bool)", (void *)&thunk_variant_construct<bool> },
		{ "void f(const String &in)", (void *)&thunk_boxed_copy },
	},
	  conversions[] = {
		  { "int64 opImplConv() const", (void *)&thunk_variant_convert<int64_t> },
		  { "double opImplConv() const", (void *)&thunk_variant_convert<double> },
		  { "bool opImplConv() const", (void *)&thunk_variant_convert<bool> },
		  { "String opImplConv() const", (void *)&thunk_variant_to_string },
	  };
	for (const Entry &e : ctors) {
		int id = p_engine->RegisterObjectBehaviour("Variant", asBEHAVE_CONSTRUCT, e.decl, asFUNCTION(e.fn), asCALL_CDECL_OBJFIRST);
		ERR_FAIL_COND_V(id < 0, false);
		g_native_registration_count++;
	}
	for (const Entry &e : conversions) {
		int id = p_engine->RegisterObjectMethod("Variant", e.decl, asFUNCTION(e.fn), asCALL_CDECL_OBJFIRST);
		ERR_FAIL_COND_V(id < 0, false);
		g_native_registration_count++;
	}
	return true;
#endif
}

bool try_add_literal_support(asIScriptEngine *p_engine) {
#if !AS_NATIVE_VALUE_OPS_ENABLED
	return false;
#else
	for (const char *name : { "String", "Variant" }) {
		int id = p_engine->RegisterObjectBehaviour(name, asBEHAVE_CONSTRUCT, "void f(const string &in)", asFUNCTION(thunk_literal_construct), asCALL_CDECL_OBJFIRST);
		ERR_FAIL_COND_V(id < 0, false);
		g_native_registration_count++;
	}
	int id = p_engine->RegisterObjectMethod("String", "String opAdd(const string &in) const", asFUNCTION(thunk_literal_add), asCALL_CDECL_OBJFIRST);
	ERR_FAIL_COND_V(id < 0, false);
	g_native_registration_count++;
	return true;
#endif
}

// 命中即原生注册并返回 true（调用方跳过 generic 跳板）；未命中返回 false。
bool try_add_op(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const String &p_op, const String &p_decl) {
#if !AS_NATIVE_VALUE_OPS_ENABLED
	// 未验证平台：不使用原生调用约定，整体回退 generic（spec §6）。
	(void)p_engine;
	(void)p_type_name;
	(void)p_type;
	(void)p_op;
	(void)p_decl;
	return false;
#else
	void *fn = nullptr;
	switch (p_type) {
		case Variant::VECTOR2:
			fn = thunk_for_op<Vector2>(p_op);
			break;
		case Variant::VECTOR3:
			fn = thunk_for_op<Vector3>(p_op);
			break;
		case Variant::VECTOR2I:
			fn = thunk_for_op<Vector2i>(p_op);
			break;
		case Variant::VECTOR3I:
			fn = thunk_for_op<Vector3i>(p_op);
			break;
		// P3b：与原生存储类型表一一对应。仅当 register_type_members 已确认该类型存在
		// (T, T) -> T 形式的运算符时才会走到这里，故未覆盖的运算符（如 Color*float）
		// 不会命中，安全回退 generic。
		case Variant::VECTOR4:
			fn = thunk_for_op<Vector4>(p_op);
			break;
		case Variant::VECTOR4I:
			fn = thunk_for_op<Vector4i>(p_op);
			break;
		case Variant::RECT2:
			fn = thunk_for_op<Rect2>(p_op);
			break;
		case Variant::RECT2I:
			fn = thunk_for_op<Rect2i>(p_op);
			break;
		case Variant::TRANSFORM2D:
			fn = thunk_for_op<Transform2D>(p_op);
			break;
		case Variant::PLANE:
			fn = thunk_for_op<Plane>(p_op);
			break;
		case Variant::QUATERNION:
			fn = thunk_for_op<Quaternion>(p_op);
			break;
		case Variant::AABB:
			fn = thunk_for_op<AABB>(p_op);
			break;
		case Variant::BASIS:
			fn = thunk_for_op<Basis>(p_op);
			break;
		case Variant::TRANSFORM3D:
			fn = thunk_for_op<Transform3D>(p_op);
			break;
		case Variant::PROJECTION:
			fn = thunk_for_op<Projection>(p_op);
			break;
		case Variant::COLOR:
			fn = thunk_for_op<Color>(p_op);
			break;
		case Variant::STRING:
			fn = thunk_for_string_op(p_op);
			break;
		case Variant::NIL:
		case Variant::STRING_NAME:
		case Variant::ARRAY:
		case Variant::DICTIONARY:
		case Variant::PACKED_BYTE_ARRAY:
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
			if (p_op == "opAssign") {
				fn = (void *)&thunk_assign<Variant>;
			}
			break;
		default:
			return false; // 其余类型暂走 generic 回退。
	}
	if (!fn) {
		return false;
	}
	if (p_engine->RegisterObjectMethod(p_type_name.utf8().get_data(), p_decl.utf8().get_data(), asFUNCTION(fn), asCALL_CDECL_OBJFIRST) >= 0) {
		g_native_registration_count++;
		return true;
	}
	// 注册失败时 asCScriptEngine 已进入 configFailed，无法再回退 generic：必须响亮报错。
	ERR_PRINT(vformat("AngelScript: failed to register native value operator '%s' for '%s' (decl: %s).", p_op, p_type_name, p_decl));
	return false;
#endif
}

// 命中即原生注册并返回 true，否则返回 false（调用方继续 generic 注册）。
#if AS_NATIVE_VALUE_OPS_ENABLED
static bool reg_method(asIScriptEngine *p_engine, const String &p_type_name, const String &p_decl, void *p_fn) {
	if (p_engine->RegisterObjectMethod(p_type_name.utf8().get_data(), p_decl.utf8().get_data(), asFUNCTION(p_fn), asCALL_CDECL_OBJFIRST) >= 0) {
		g_native_registration_count++;
		return true;
	}
	ERR_PRINT(vformat("AngelScript: failed to register native value method for '%s' (decl: %s).", p_type_name, p_decl));
	return false;
}
#endif

bool try_add_method(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const StringName &p_method, const String &p_decl) {
#if !AS_NATIVE_VALUE_OPS_ENABLED
	// 未验证平台：不使用原生调用约定，整体回退 generic（spec §6）。
	(void)p_engine;
	(void)p_type_name;
	(void)p_type;
	(void)p_method;
	(void)p_decl;
	return false;
#else
	const String m = String(p_method);
	switch (p_type) {
		case Variant::VECTOR2:
			if (m == "length") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_scalar<Vector2, double, &Vector2::length>);
			}
			if (m == "length_squared") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_scalar<Vector2, double, &Vector2::length_squared>);
			}
			if (m == "normalized") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_obj<Vector2, Vector2, &Vector2::normalized>);
			}
			if (m == "dot") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector2, double, &Vector2::dot>);
			}
			if (m == "distance_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector2, double, &Vector2::distance_to>);
			}
			if (m == "distance_squared_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector2, double, &Vector2::distance_squared_to>);
			}
			if (m == "cross") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector2, double, &Vector2::cross>);
			}
			if (m == "angle") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_scalar<Vector2, double, &Vector2::angle>);
			}
			if (m == "angle_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector2, double, &Vector2::angle_to>);
			}
			if (m == "angle_to_point") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector2, double, &Vector2::angle_to_point>);
			}
			if (m == "direction_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector2, Vector2, &Vector2::direction_to>);
			}
			if (m == "project") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector2, Vector2, &Vector2::project>);
			}
			if (m == "slide") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector2, Vector2, &Vector2::slide>);
			}
			if (m == "bounce") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector2, Vector2, &Vector2::bounce>);
			}
			if (m == "reflect") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector2, Vector2, &Vector2::reflect>);
			}
			if (m == "lerp") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj_f<Vector2, Vector2, &Vector2::lerp>);
			}
			if (m == "slerp") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj_f<Vector2, Vector2, &Vector2::slerp>);
			}
			if (m == "move_toward") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj_f<Vector2, Vector2, &Vector2::move_toward>);
			}
			if (m == "clamp") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method2_obj<Vector2, Vector2, &Vector2::clamp>);
			}
			if (m == "rotated") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_f_obj<Vector2, Vector2, &Vector2::rotated>);
			}
			if (m == "posmod") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_f_obj<Vector2, Vector2, &Vector2::posmod>);
			}
			break;
		case Variant::VECTOR3:
			if (m == "length") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_scalar<Vector3, double, &Vector3::length>);
			}
			if (m == "length_squared") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_scalar<Vector3, double, &Vector3::length_squared>);
			}
			if (m == "normalized") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_obj<Vector3, Vector3, &Vector3::normalized>);
			}
			if (m == "dot") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector3, double, &Vector3::dot>);
			}
			if (m == "distance_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector3, double, &Vector3::distance_to>);
			}
			if (m == "distance_squared_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector3, double, &Vector3::distance_squared_to>);
			}
			if (m == "cross") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector3, Vector3, &Vector3::cross>);
			}
			if (m == "angle_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector3, double, &Vector3::angle_to>);
			}
			if (m == "direction_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector3, Vector3, &Vector3::direction_to>);
			}
			if (m == "project") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector3, Vector3, &Vector3::project>);
			}
			if (m == "slide") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector3, Vector3, &Vector3::slide>);
			}
			if (m == "bounce") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector3, Vector3, &Vector3::bounce>);
			}
			if (m == "reflect") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj<Vector3, Vector3, &Vector3::reflect>);
			}
			if (m == "lerp") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj_f<Vector3, Vector3, &Vector3::lerp>);
			}
			if (m == "slerp") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj_f<Vector3, Vector3, &Vector3::slerp>);
			}
			if (m == "move_toward") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_obj_f<Vector3, Vector3, &Vector3::move_toward>);
			}
			if (m == "clamp") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method2_obj<Vector3, Vector3, &Vector3::clamp>);
			}
			break;
		case Variant::VECTOR2I:
			if (m == "length") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_scalar<Vector2i, double, &Vector2i::length>);
			}
			if (m == "length_squared") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_scalar<Vector2i, int64_t, &Vector2i::length_squared>);
			}
			if (m == "distance_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector2i, double, &Vector2i::distance_to>);
			}
			if (m == "distance_squared_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector2i, int64_t, &Vector2i::distance_squared_to>);
			}
			break;
		case Variant::VECTOR3I:
			if (m == "length") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_scalar<Vector3i, double, &Vector3i::length>);
			}
			if (m == "length_squared") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method_scalar<Vector3i, int64_t, &Vector3i::length_squared>);
			}
			if (m == "distance_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector3i, double, &Vector3i::distance_to>);
			}
			if (m == "distance_squared_to") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_method1_scalar<Vector3i, int64_t, &Vector3i::distance_squared_to>);
			}
			break;
		case Variant::PACKED_BYTE_ARRAY:
			if (m == "size") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_size<uint8_t>);
			}
			if (m == "push_back") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_push<uint8_t, int64_t>);
			}
			break;
		case Variant::PACKED_INT32_ARRAY:
			if (m == "size") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_size<int32_t>);
			}
			if (m == "push_back") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_push<int32_t, int64_t>);
			}
			break;
		case Variant::PACKED_INT64_ARRAY:
			if (m == "size") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_size<int64_t>);
			}
			if (m == "push_back") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_push<int64_t, int64_t>);
			}
			break;
		case Variant::PACKED_FLOAT32_ARRAY:
			if (m == "size") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_size<float>);
			}
			if (m == "push_back") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_push<float, double>);
			}
			break;
		case Variant::PACKED_FLOAT64_ARRAY:
			if (m == "size") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_size<double>);
			}
			if (m == "push_back") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_push<double, double>);
			}
			break;
		// A2：Array/Dictionary 安全内建方法白名单。仅覆盖无参标量/Variant 返回、单个
		// Variant 参、无默认值、无 Object handle 的方法；其余方法保持 generic 回退。
		case Variant::ARRAY:
			if (m == "size") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array0<&Array::size, int64_t>);
			}
			if (m == "is_empty") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array0<&Array::is_empty, bool>);
			}
			if (m == "hash") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array0<&Array::hash, int64_t>);
			}
			if (m == "clear") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array_mutate0<&Array::clear>);
			}
			if (m == "push_back") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array_mutate1<&Array::push_back>);
			}
			if (m == "push_front") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array_mutate1<&Array::push_front>);
			}
			if (m == "append") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array_mutate1<&Array::append>);
			}
			if (m == "has") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array1<&Array::has, bool>);
			}
			if (m == "count") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array1<&Array::count, int64_t>);
			}
			if (m == "front") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array_return<&Array::front>);
			}
			if (m == "back") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array_return<&Array::back>);
			}
			if (m == "pop_back") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array_return<&Array::pop_back>);
			}
			if (m == "pop_front") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_array_return<&Array::pop_front>);
			}
			break;
		case Variant::DICTIONARY:
			if (m == "size") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_bm0_scalar<Variant::DICTIONARY, BmSize, int64_t>);
			}
			if (m == "is_empty") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_bm0_scalar<Variant::DICTIONARY, BmIsEmpty, bool>);
			}
			if (m == "hash") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_bm0_scalar<Variant::DICTIONARY, BmHash, int64_t>);
			}
			if (m == "clear") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_bm0_void<Variant::DICTIONARY, BmClear>);
			}
			if (m == "has") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_bm1_scalar<Variant::DICTIONARY, BmHas, bool>);
			}
			if (m == "erase") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_bm1_scalar<Variant::DICTIONARY, BmErase, bool>);
			}
			break;
		default:
			break;
	}
	return false; // 未覆盖的方法走 generic 回退。
#endif
}

bool try_add_index_accessor(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const String &p_get_decl, const String &p_set_decl) {
#if !AS_NATIVE_VALUE_OPS_ENABLED
	// 未验证平台：整体回退 generic（spec §6）。
	(void)p_engine;
	(void)p_type_name;
	(void)p_type;
	(void)p_get_decl;
	(void)p_set_decl;
	return false;
#else
	void *get_fn = nullptr;
	void *set_fn = nullptr;
	switch (p_type) {
		case Variant::PACKED_BYTE_ARRAY:
			get_fn = (void *)&thunk_packed_get<uint8_t, int64_t>;
			set_fn = (void *)&thunk_packed_set<uint8_t, int64_t>;
			break;
		case Variant::PACKED_INT32_ARRAY:
			get_fn = (void *)&thunk_packed_get<int32_t, int64_t>;
			set_fn = (void *)&thunk_packed_set<int32_t, int64_t>;
			break;
		case Variant::PACKED_INT64_ARRAY:
			get_fn = (void *)&thunk_packed_get<int64_t, int64_t>;
			set_fn = (void *)&thunk_packed_set<int64_t, int64_t>;
			break;
		case Variant::PACKED_FLOAT32_ARRAY:
			get_fn = (void *)&thunk_packed_get<float, double>;
			set_fn = (void *)&thunk_packed_set<float, double>;
			break;
		case Variant::PACKED_FLOAT64_ARRAY:
			get_fn = (void *)&thunk_packed_get<double, double>;
			set_fn = (void *)&thunk_packed_set<double, double>;
			break;
		case Variant::ARRAY:
			// 元素的 AS 类型是 Variant；需 Variant 带 asOBJ_APP_CLASS 才能原生返回。
			get_fn = (void *)&thunk_variant_index_get;
			set_fn = (void *)&thunk_variant_index_set;
			break;
		default:
			return false; // 元素为 Variant/对象/字符串等的容器仍走 generic。
	}
	if (p_engine->RegisterObjectMethod(p_type_name.utf8().get_data(), p_get_decl.utf8().get_data(), asFUNCTION(get_fn), asCALL_CDECL_OBJFIRST) < 0) {
		ERR_PRINT(vformat("AngelScript: failed to register native packed index getter for '%s' (decl: %s).", p_type_name, p_get_decl));
		return false;
	}
	if (p_engine->RegisterObjectMethod(p_type_name.utf8().get_data(), p_set_decl.utf8().get_data(), asFUNCTION(set_fn), asCALL_CDECL_OBJFIRST) < 0) {
		ERR_PRINT(vformat("AngelScript: failed to register native packed index setter for '%s' (decl: %s).", p_type_name, p_set_decl));
		return false;
	}
	g_native_registration_count += 2;
	return true;
#endif
}

bool try_add_keyed_accessor(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const String &p_get_decl, const String &p_set_decl) {
#if !AS_NATIVE_VALUE_OPS_ENABLED
	(void)p_engine;
	(void)p_type_name;
	(void)p_type;
	(void)p_get_decl;
	(void)p_set_decl;
	return false;
#else
	if (p_type != Variant::DICTIONARY) {
		return false;
	}
	if (p_engine->RegisterObjectMethod(p_type_name.utf8().get_data(), p_get_decl.utf8().get_data(), asFUNCTION(thunk_variant_keyed_get), asCALL_CDECL_OBJFIRST) < 0) {
		ERR_PRINT(vformat("AngelScript: failed to register native keyed getter for '%s' (decl: %s).", p_type_name, p_get_decl));
		return false;
	}
	if (p_engine->RegisterObjectMethod(p_type_name.utf8().get_data(), p_set_decl.utf8().get_data(), asFUNCTION(thunk_variant_keyed_set), asCALL_CDECL_OBJFIRST) < 0) {
		ERR_PRINT(vformat("AngelScript: failed to register native keyed setter for '%s' (decl: %s).", p_type_name, p_set_decl));
		return false;
	}
	g_native_registration_count += 2;
	return true;
#endif
}

int native_registration_count() { return g_native_registration_count; }
int native_thunk_calls() { return g_native_thunk_calls; }
void reset_native_thunk_calls() { g_native_thunk_calls = 0; }
void note_thunk_call() { g_native_thunk_calls++; }

bool native_ops_enabled() { return AS_NATIVE_VALUE_OPS_ENABLED != 0; }

const char *native_ops_gate_name() {
#if AS_NATIVE_VALUE_OPS_GATE == AS_NATIVE_VALUE_OPS_GATE_VERIFIED
	return "verified-msvc-x64";
#elif AS_NATIVE_VALUE_OPS_GATE == AS_NATIVE_VALUE_OPS_GATE_OVERRIDE
	return "unverified-override";
#elif AS_NATIVE_VALUE_OPS_GATE == AS_NATIVE_VALUE_OPS_GATE_FORCED_GENERIC
	return "forced-generic";
#else
	return "platform-fallback";
#endif
}

} // namespace ASNativeValueOps
