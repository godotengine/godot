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

// ---------- P2：Packed 标量数组 ----------
// 这些操作的存储仍是 Variant（容器不是原生存储类型），thunk 直接调用 Variant 的索引
// 访问器，跳过 asCALL_GENERIC 跳板。元素为标量（int64/double），可按值返回。

// size()：返回元素个数。对所有带索引的容器均等价。
int64_t thunk_packed_size(const Variant *self) {
	ASNativeValueOps::note_thunk_call();
	return (int64_t)self->get_indexed_size();
}

// 索引读：int64 元素（PACKED_BYTE/INT32/INT64）。
int64_t thunk_index_get_i64(const Variant *self, int64_t index) {
	ASNativeValueOps::note_thunk_call();
	bool valid = false;
	bool oob = false;
	Variant v = self->get_indexed(index, valid, oob); // 越界返回默认值 0，与 generic 一致。
	return (int64_t)v;
}

// 索引读：double 元素（PACKED_FLOAT32/FLOAT64）。
double thunk_index_get_f64(const Variant *self, int64_t index) {
	ASNativeValueOps::note_thunk_call();
	bool valid = false;
	bool oob = false;
	Variant v = self->get_indexed(index, valid, oob);
	return (double)v;
}

// 索引写：越界静默忽略（与 generic 的 set_indexed 语义一致）。
void thunk_index_set_i64(Variant *self, int64_t index, int64_t value) {
	ASNativeValueOps::note_thunk_call();
	bool valid = false;
	bool oob = false;
	self->set_indexed(index, Variant(value), valid, oob);
}

void thunk_index_set_f64(Variant *self, int64_t index, double value) {
	ASNativeValueOps::note_thunk_call();
	bool valid = false;
	bool oob = false;
	self->set_indexed(index, Variant(value), valid, oob);
}

} // namespace

namespace ASNativeValueOps {

static int g_native_registration_count = 0;
static int g_native_thunk_calls = 0;

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
		// P2：Packed 标量数组的 size()（元素个数），无参标量返回，直接走原生 thunk。
		case Variant::PACKED_BYTE_ARRAY:
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
			if (m == "size") {
				return reg_method(p_engine, p_type_name, p_decl, (void *)&thunk_packed_size);
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
		case Variant::PACKED_INT32_ARRAY:
		case Variant::PACKED_INT64_ARRAY:
			get_fn = (void *)&thunk_index_get_i64;
			set_fn = (void *)&thunk_index_set_i64;
			break;
		case Variant::PACKED_FLOAT32_ARRAY:
		case Variant::PACKED_FLOAT64_ARRAY:
			get_fn = (void *)&thunk_index_get_f64;
			set_fn = (void *)&thunk_index_set_f64;
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

int native_registration_count() { return g_native_registration_count; }
int native_thunk_calls() { return g_native_thunk_calls; }
void reset_native_thunk_calls() { g_native_thunk_calls = 0; }
void note_thunk_call() { g_native_thunk_calls++; }

} // namespace ASNativeValueOps
