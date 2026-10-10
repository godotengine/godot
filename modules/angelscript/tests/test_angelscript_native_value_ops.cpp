/**************************************************************************/
/*  test_angelscript_native_value_ops.cpp                                 */
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

#include "../as_engine.h"
#include "../binding/as_binding_native_storage.h"
#include "../binding/as_binding_native_value_ops.h"
#include "../binding/as_binding_value_types.h"

#define ANGELSCRIPT_NATIVE_VALUE_OPS_TESTS_IMPL
#include "test_angelscript_native_value_ops.h"

#include "core/math/color.h"
#include "core/math/transform_2d.h"
#include "core/math/vector2.h"
#include "core/math/vector2i.h"
#include "core/math/vector3.h"
#include "core/math/vector3i.h"
#include "core/string/print_string.h"

#include <angelscript.h>

static bool nearly(double p_a, double p_b) {
	return p_a > p_b - 0.001 && p_a < p_b + 0.001;
}

// 编译并执行 `double main()`，取回其 double 返回值。
static bool run_double(const String &p_source, double *r_out) {
	ASEngine *as = ASEngine::get_singleton();
	if (!as->ensure_initialized() || !as->is_initialized()) {
		REQUIRE(false);
		return false;
	}
	asIScriptEngine *engine = as->get_engine();
	if (ASBindingValueTypes::register_all(engine) != OK) {
		REQUIRE(false);
		return false;
	}
	String err;
	if (!as->compile_module("as_native_vo_run", p_source, &err)) {
		print_line("compile error: " + err);
		REQUIRE(false);
		return false;
	}
	asIScriptModule *mod = engine->GetModule("as_native_vo_run");
	asIScriptFunction *func = mod ? mod->GetFunctionByName("main") : nullptr;
	if (!func) {
		REQUIRE(false);
		return false;
	}
	asIScriptContext *ctx = engine->CreateContext();
	if (!ctx || ctx->Prepare(func) < 0) {
		if (ctx) {
			ctx->Release();
		}
		REQUIRE(false);
		return false;
	}
	int rc = ctx->Execute();
	double out = ctx->GetReturnDouble();
	ctx->Release();
	if (rc != asEXECUTION_FINISHED) {
		REQUIRE(false);
		return false;
	}
	if (r_out) {
		*r_out = out;
	}
	return true;
}

void as_native_vo_scaffold_fallback_unchanged() {
	double out = 0.0;
	if (!run_double("double main() { Vector2 a(1, 2); Vector2 b(3, 4); Vector2 c = a + b; return c.x + c.y; }", &out)) {
		return;
	}
	CHECK(nearly(out, 10.0));
}

void as_native_vo_vector2_opadd() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double("double main() { Vector2 a(1, 2); Vector2 b(3, 4); Vector2 c = a + b; return c.x * 10.0 + c.y; }", &out)) {
		return;
	}
	CHECK(nearly(out, 46.0)); // (1+3)*10 + (2+4)
	CHECK(ASNativeValueOps::native_thunk_calls() > 0); // 证明走了原生 opAdd
}

void as_native_vo_vector2_arith_assign() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double("double main() { Vector2 a(10, 8); Vector2 b(2, 3); Vector2 s = a - b; Vector2 m = a * b; Vector2 d = a / b; Vector2 r; r = s + m + d; return r.x * 100.0 + r.y; }", &out)) {
		return;
	}
	CHECK(out > 3331.0); // s=(8,5) m=(20,24) d=(5,2.666..) r=(33,31.666..)
	CHECK(out < 3332.0);
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);

	// 隔离验证：仅减法也必须走原生（排除 opAdd 对计数的干扰）。
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double("double main() { Vector2 a(10, 8); Vector2 b(2, 3); Vector2 s = a - b; return s.x * 100.0 + s.y; }", &out)) {
		return;
	}
	CHECK(nearly(out, 805.0)); // (8,5)
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

void as_native_vo_vector3_arith_assign() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double("double main() { Vector3 a(1, 2, 3); Vector3 b(4, 5, 6); Vector3 c = a + b; Vector3 d = b - a; Vector3 m = a * b; Vector3 q = b / a; Vector3 f; f = c; return f.x + d.y + m.z + q.x; }", &out)) {
		return;
	}
	CHECK(nearly(out, 30.0)); // f.x=5 (a+b) + d.y=3 (b-a) + m.z=18 (a*b) + q.x=4 (b/a)
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

void as_native_vo_integer_vectors() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(
				"double main() {"
				"  Vector2i a(7, 8);"
				"  Vector2i b(3, 2);"
				"  Vector2i d = a / b;"
				"  Vector3i p(2, 3, 4);"
				"  Vector3i q(5, 6, 7);"
				"  Vector3i s = p + q;"
				"  Vector3i r; r = s;"
				"  int64 v = d.x + d.y + r.y;"
				"  return double(v);"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 2.0 + 4.0 + 9.0)); // 整数除法 (7/3,8/2)=(2,4) + s.y=9
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

void as_native_vo_op_equals() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(
				"double main() {"
				"  Vector2 a(1, 2); Vector2 b(1, 2); Vector2 c(1, 3);"
				"  Vector3 p(1, 2, 3); Vector3 q(1, 2, 3); Vector3 s(1, 2, 4);"
				"  Vector2i u(1, 2); Vector2i v(1, 2); Vector2i w(2, 2);"
				"  Vector3i x(1, 2, 3); Vector3i y(1, 2, 3); Vector3i z(1, 2, 4);"
				"  return (a == b) && !(a == c) && (p == q) && !(p == s) && (u == v) && !(u == w) && (x == y) && !(x == z) ? 1.0 : 0.0;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.0)); // 四类型 opEquals：相等为 true、不相等为 false
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

void as_native_vo_method_core() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(
				"double main() {"
				"  Vector2 a(3, 4);"
				"  Vector2 b(0, 0);"
				"  double L = a.length();"
				"  double D = a.distance_to(b);"
				"  double dot = a.dot(b);"
				"  return a.normalized().x + a.normalized().y + L + D + dot + b.normalized().x;"
				"}",
				&out)) {
		return;
	}
	// a=(3,4): normalized=(0.6,0.8), length=5, distance_to((0,0))=5, dot=0；
	// b 为零向量，normalized 应为 (0,0)（与 generic 语义一致）。
	CHECK(nearly(out, 11.4));
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

void as_native_vo_cross() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	// 用 `p.cross(q).z` 直接取临时对象的成员，避免 `Vector3 c3 = ...` 触发已原生的 opAssign 污染计数（同 T9 裁定）。
	if (!run_double(
				"double main() {"
				"  Vector2 a(1, 0); Vector2 b(0, 1);"
				"  double c2 = a.cross(b);"
				"  Vector3 p(1, 0, 0); Vector3 q(0, 1, 0);"
				"  return c2 + p.cross(q).z;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 2.0));
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

void as_native_vo_geometry() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	// 用临时对象成员访问避免 `Vector2 r = ...` 触发已原生 opAssign 污染计数（同 T9 裁定）。
	if (!run_double(
				"double main() {"
				"  Vector2 a(1, 0); Vector2 b(0, 1);"
				"  double ang = a.angle();"
				"  double a2 = a.angle_to(b);"
				"  double rx = a.reflect(Vector2(0, 1)).x;"
				"  double lx = a.lerp(b, 0.5).x;"
				"  double ly = a.lerp(b, 0.5).y;"
				"  return ang + a2 + rx + lx + ly;"
				"}",
				&out)) {
		return;
	}
	CHECK(nearly(out, 1.5707963267948966));
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

// 全方法覆盖 + C++ 参照值 parity：调用每个已原生化的方法，与相同表达式用
// Godot 原生 Vector2 计算的结果对比（这正是 generic 路径原先的期望值）。
void as_native_vo_vector2_methods_all() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(
				"double main() {"
				"  Vector2 a(3, 4); Vector2 b(1, 0); Vector2 c(0, 1);"
				"  Vector2 lo(-1, -1); Vector2 hi(1, 1);"
				"  double r = 0.0;"
				"  r += a.length();"
				"  r += a.length_squared();"
				"  r += a.dot(b);"
				"  r += a.distance_to(b);"
				"  r += a.distance_squared_to(b);"
				"  r += a.cross(b);"
				"  r += a.angle();"
				"  r += a.angle_to(b);"
				"  r += a.angle_to_point(b);"
				"  r += a.normalized().x + a.normalized().y;"
				"  r += a.direction_to(b).x + a.direction_to(b).y;"
				"  r += a.project(b).x + a.project(b).y;"
				"  r += a.slide(b).x + a.slide(b).y;"
				"  r += a.bounce(b).x + a.bounce(b).y;"
				"  r += a.reflect(b).x + a.reflect(b).y;"
				"  r += a.lerp(c, 0.0).x + a.lerp(c, 1.0).y;"
				"  r += a.slerp(c, 0.5).x + a.slerp(c, 0.5).y;"
				"  r += a.move_toward(c, 100.0).x + a.move_toward(c, 100.0).y;"
				"  r += a.clamp(lo, hi).x + a.clamp(lo, hi).y;"
				"  r += a.rotated(0.5).x + a.rotated(0.5).y;"
				"  r += a.posmod(2.0).x + a.posmod(2.0).y;"
				"  return r;"
				"}",
				&out)) {
		return;
	}
	Vector2 a(3, 4), b(1, 0), c(0, 1), lo(-1, -1), hi(1, 1);
	double e = 0.0;
	e += a.length();
	e += a.length_squared();
	e += a.dot(b);
	e += a.distance_to(b);
	e += a.distance_squared_to(b);
	e += a.cross(b);
	e += a.angle();
	e += a.angle_to(b);
	e += a.angle_to_point(b);
	e += a.normalized().x + a.normalized().y;
	e += a.direction_to(b).x + a.direction_to(b).y;
	e += a.project(b).x + a.project(b).y;
	e += a.slide(b).x + a.slide(b).y;
	e += a.bounce(b).x + a.bounce(b).y;
	e += a.reflect(b).x + a.reflect(b).y;
	e += a.lerp(c, 0.0).x + a.lerp(c, 1.0).y;
	e += a.slerp(c, 0.5).x + a.slerp(c, 0.5).y;
	e += a.move_toward(c, 100.0).x + a.move_toward(c, 100.0).y;
	e += a.clamp(lo, hi).x + a.clamp(lo, hi).y;
	e += a.rotated(0.5).x + a.rotated(0.5).y;
	e += a.posmod(2.0).x + a.posmod(2.0).y;
	CHECK(nearly(out, e));
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

void as_native_vo_vector3_methods_all() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(
				"double main() {"
				"  Vector3 a(1, 2, 2); Vector3 b(1, 0, 0); Vector3 c(0, 1, 0);"
				"  Vector3 lo(-1, -1, -1); Vector3 hi(1, 1, 1);"
				"  double r = 0.0;"
				"  r += a.length();"
				"  r += a.length_squared();"
				"  r += a.dot(b);"
				"  r += a.distance_to(b);"
				"  r += a.distance_squared_to(b);"
				"  r += a.cross(b).z;"
				"  r += a.angle_to(b);"
				"  r += a.direction_to(b).x + a.direction_to(b).y + a.direction_to(b).z;"
				"  r += a.project(b).x + a.project(b).y + a.project(b).z;"
				"  r += a.slide(b).x + a.slide(b).y + a.slide(b).z;"
				"  r += a.bounce(b).x + a.bounce(b).y + a.bounce(b).z;"
				"  r += a.reflect(b).x + a.reflect(b).y + a.reflect(b).z;"
				"  r += a.normalized().x + a.normalized().y + a.normalized().z;"
				"  r += a.lerp(c, 0.0).x + a.lerp(c, 1.0).y + a.lerp(c, 1.0).z;"
				"  r += a.slerp(c, 0.5).x + a.slerp(c, 0.5).y + a.slerp(c, 0.5).z;"
				"  r += a.move_toward(c, 100.0).x + a.move_toward(c, 100.0).y + a.move_toward(c, 100.0).z;"
				"  r += a.clamp(lo, hi).x + a.clamp(lo, hi).y + a.clamp(lo, hi).z;"
				"  return r;"
				"}",
				&out)) {
		return;
	}
	Vector3 a(1, 2, 2), b(1, 0, 0), c(0, 1, 0), lo(-1, -1, -1), hi(1, 1, 1);
	double e = 0.0;
	e += a.length();
	e += a.length_squared();
	e += a.dot(b);
	e += a.distance_to(b);
	e += a.distance_squared_to(b);
	e += a.cross(b).z;
	e += a.angle_to(b);
	e += a.direction_to(b).x + a.direction_to(b).y + a.direction_to(b).z;
	e += a.project(b).x + a.project(b).y + a.project(b).z;
	e += a.slide(b).x + a.slide(b).y + a.slide(b).z;
	e += a.bounce(b).x + a.bounce(b).y + a.bounce(b).z;
	e += a.reflect(b).x + a.reflect(b).y + a.reflect(b).z;
	e += a.normalized().x + a.normalized().y + a.normalized().z;
	e += a.lerp(c, 0.0).x + a.lerp(c, 1.0).y + a.lerp(c, 1.0).z;
	e += a.slerp(c, 0.5).x + a.slerp(c, 0.5).y + a.slerp(c, 0.5).z;
	e += a.move_toward(c, 100.0).x + a.move_toward(c, 100.0).y + a.move_toward(c, 100.0).z;
	e += a.clamp(lo, hi).x + a.clamp(lo, hi).y + a.clamp(lo, hi).z;
	CHECK(nearly(out, e));
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

void as_native_vo_integer_methods_all() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(
				"double main() {"
				"  Vector2i a(3, 4); Vector2i b(1, 0);"
				"  Vector3i p(1, 2, 2); Vector3i q(4, 5, 6);"
				"  double r = 0.0;"
				"  r += a.length();"
				"  r += a.length_squared();"
				"  r += a.distance_to(b);"
				"  r += a.distance_squared_to(b);"
				"  r += p.length();"
				"  r += p.length_squared();"
				"  r += p.distance_to(q);"
				"  r += p.distance_squared_to(q);"
				"  return r;"
				"}",
				&out)) {
		return;
	}
	Vector2i a(3, 4), b(1, 0);
	Vector3i p(1, 2, 2), q(4, 5, 6);
	double e = 0.0;
	e += a.length();
	e += a.length_squared();
	e += a.distance_to(b);
	e += a.distance_squared_to(b);
	e += p.length();
	e += p.length_squared();
	e += p.distance_to(q);
	e += p.distance_squared_to(q);
	CHECK(nearly(out, e));
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

void as_native_vo_storage_slot_helpers() {
	// 仅验证门控与 4 类型判定；未启用平台应为 false。
	if (!ASNativeValueStorage::is_native_storage_type(Variant::VECTOR2)) {
		CHECK_FALSE(ASNativeValueStorage::is_native_storage_type(Variant::VECTOR2));
		return; // 未启用平台：短路
	}
	Vector2 v(1.5, -2.5);
	Variant back = ASNativeValueStorage::native_to_variant(Variant::VECTOR2, &v);
	CHECK_EQ((Vector2)back, v);
	Vector2 slot;
	ASNativeValueStorage::variant_to_native(Variant::VECTOR2, Variant(Vector2(3, 4)), &slot);
	CHECK_EQ(slot, Vector2(3, 4));
}

void as_native_vo_native_storage_round_trip() {
	if (!ASNativeValueStorage::is_native_storage_type(Variant::VECTOR2)) {
		return; // 未启用平台跳过（与今日 generic 行为一致）。
	}
	double out = 0.0;
	CHECK(run_double(R"(
		double main() {
			Vector2 a(3.0, 4.0);
			Vector2 b = a;
			b.x = 6.0;
			Vector2 c = a + b;          // (9, 8)
			c.y = c.y * 2.0;            // (9, 16)
			return c.x + c.y + a.length(); // 9 + 16 + 5 = 30
		}
	)", &out));
	CHECK(nearly(out, 30.0));
}

void as_native_vo_native_fast_path_active() {
	if (!ASNativeValueStorage::is_native_storage_type(Variant::VECTOR2)) {
		return; // 未启用平台跳过：无原生快路径。
	}
	ASNativeValueOps::reset_native_thunk_calls();
	double out = 0.0;
	CHECK(run_double("double main() { Vector2 a(1, 2); Vector2 b(3, 4); Vector2 c = a + b; return c.length(); }", &out));
	CHECK(ASNativeValueOps::native_thunk_calls() > 0); // opAdd 与 length 均命中原生 thunk
}

// 边界编组：原生存储类型在"原生槽 ↔ Variant"的每一个出入口都必须双向正确。
// 这里用 PackedVector2Array 同时覆盖两类边界（同一编组单点 as_binding_decl.cpp）：
//   - `push_back(Vector2)` 形参：引擎方法形参边界，原生 Vector2 槽 → Variant；
//   - `arr[i]` 返回：引擎方法返回边界，Variant → 原生 Vector2 槽。
// 断言取精确分量（而非仅归一化长度），否则编组失真仍可能因"任意非零向量归一到单位长度"而假绿。
void as_native_vo_boundary_conversions() {
	if (!ASNativeValueStorage::is_native_storage_type(Variant::VECTOR2)) {
		return; // 未启用平台跳过：无原生存储边界。
	}
	double out = 0.0;
	CHECK(run_double(R"(
		double main() {
			PackedVector2Array arr;
			arr.push_back(Vector2(1, 2));   // 形参边界：native -> Variant
			arr.push_back(Vector2(3, 4));
			Vector2 a = arr[0];             // 返回边界：Variant -> native
			Vector2 b = arr[1];
			Vector2 s = a + b;              // (4, 6)
			Vector2 n = s.normalized();
			return 1000.0 * (a.x + b.y) + n.length(); // 1000*(1+4) + 1 = 5001
		}
	)", &out));
	CHECK(nearly(out, 5001.0));
}

// P2：Packed 标量数组索引访问器（get/set_opIndex）原生 thunk。读写命中即计数增加。
void as_native_vo_packed_index_accessors() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(R"(
		double main() {
			PackedInt64Array a;
			a.push_back(10);
			a.push_back(20);
			a.push_back(30);
			a[1] = 25;                        // set_opIndex（原生）
			PackedFloat64Array f;
			f.push_back(1.5);
			f.push_back(2.5);
			f[0] = 3.5;                       // set_opIndex（原生）
			int64 ai = a[0] + a[1] + a[2];    // get_opIndex（原生）
			double fd = f[0] + f[1];
			return double(ai) + fd;
		}
	)", &out)) {
		return;
	}
	CHECK(nearly(out, 71.0)); // (10+25+30) + (3.5+2.5) = 65 + 6
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

// P2：Packed 标量数组 size() 原生 thunk。
void as_native_vo_packed_size() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(R"(
		double main() {
			PackedInt64Array a;
			a.push_back(1);
			a.push_back(2);
			PackedFloat64Array f;
			f.push_back(1.0);
			return double(a.size()) * 10.0 + double(f.size());
		}
	)", &out)) {
		return;
	}
	CHECK(nearly(out, 21.0));
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

// P2：越界语义与 generic 一致（读返回默认 0，写静默忽略），不得崩溃。
void as_native_vo_packed_index_oob() {
	double out = 0.0;
	if (!run_double(R"(
		double main() {
			PackedInt64Array a;
			a.push_back(5);
			int64 oob = a[7];   // 越界读 -> 0
			a[7] = 9;           // 越界写 -> 静默忽略
			return double(a[0]) + double(oob);
		}
	)", &out)) {
		return;
	}
	CHECK(nearly(out, 5.0));
}

// P2+：Packed 标量数组 push_back 原生 thunk。5 次 push_back 应各命中一次原生 thunk，
// 叠加随后的 5 次索引读，总数应为 10；若 push_back 仍走 generic，则只有索引的 5 次。
void as_native_vo_packed_push_back() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(R"(
		double main() {
			PackedInt64Array a;
			a.push_back(10);
			a.push_back(20);
			a.push_back(30);
			PackedFloat64Array f;
			f.push_back(1.5);
			f.push_back(2.5);
			return double(a[0] + a[1] + a[2]) + f[0] + f[1];
		}
	)", &out)) {
		return;
	}
	CHECK(nearly(out, 64.0)); // (10+20+30) + (1.5+2.5) = 60 + 4
	CHECK(ASNativeValueOps::native_thunk_calls() >= 10);
}

// String（存储为 Variant）的原生运算符：拼接/赋值/相等。命中即计数增加；
// 未原生化时三步均走 generic，计数保持 0。
void as_native_vo_string_ops() {
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	if (!run_double(R"(
		double main() {
			String a = "foo";
			String b = a + "bar";   // opAdd（原生）
			String c = b;           // 拷贝构造（泛型）
			c = a;                  // opAssign（原生）
			bool eq = (a == "foo"); // opEquals（原生）
			return double(b.length()) * 100.0 + double(c.length()) * 10.0 + (eq ? 1.0 : 0.0);
		}
	)", &out)) {
		return;
	}
	CHECK(nearly(out, 631.0)); // b="foobar"(6)*100 + c="foo"(3)*10 + 1
	CHECK(ASNativeValueOps::native_thunk_calls() > 0);
}

// P3b：更多几何值类型纳入原生存储。原生存储只改变脚本侧内存布局与生命周期，
// 不改变脚本可见语义，故主要验证：判定表覆盖到位、槽 ↔ Variant 双向转换精确、
// 且这些类型在脚本中的构造/拷贝/运算仍与既有 generic 行为一致。
void as_native_vo_extended_native_storage() {
	if (!ASNativeValueStorage::is_native_storage_type(Variant::VECTOR2)) {
		return; // 未启用平台跳过（判定表整体返回 false）。
	}
	const Variant::Type extended[] = {
		Variant::VECTOR4, Variant::VECTOR4I, Variant::RECT2, Variant::RECT2I,
		Variant::TRANSFORM2D, Variant::PLANE, Variant::QUATERNION, Variant::AABB,
		Variant::BASIS, Variant::TRANSFORM3D, Variant::PROJECTION, Variant::COLOR,
	};
	for (const Variant::Type t : extended) {
		CHECK(ASNativeValueStorage::is_native_storage_type(t));
	}
	// Color 槽双向转换。
	Color c(0.1f, 0.2f, 0.3f, 0.4f);
	Variant cback = ASNativeValueStorage::native_to_variant(Variant::COLOR, &c);
	CHECK(nearly((double)((Color)cback).r, 0.1));
	Color cslot;
	ASNativeValueStorage::variant_to_native(Variant::COLOR, Variant(Color(0.5, 0.6, 0.7, 1.0)), &cslot);
	CHECK(nearly((double)cslot.g, 0.6));
	// Transform3D 槽双向转换（较大的结构体，验证非 8/12 字节布局也正确）。
	Transform3D xf(Basis(), Vector3(1.5, -2.5, 3.5));
	Variant xback = ASNativeValueStorage::native_to_variant(Variant::TRANSFORM3D, &xf);
	CHECK_EQ((Transform3D)xback, xf);
}

// P3b：脚本侧新原生存储类型语义等价（Color 算术 + 成员读写、Transform2D 拷贝/构造/取原点）。
void as_native_vo_extended_types_script() {
	if (!ASNativeValueStorage::is_native_storage_type(Variant::COLOR)) {
		return;
	}
	double out = 0.0;
	ASNativeValueOps::reset_native_thunk_calls();
	CHECK(run_double(R"(
		double main() {
			Color a(1.0, 0.5, 0.25, 1.0);
			Color b = a;                 // 拷贝构造
			b.g = 1.0;                   // 成员写
			Color s = a + b;             // (2, 1.5, 0.5, 2)
			Transform2D t(Vector2(1, 0), Vector2(0, 1), Vector2(3, 4));
			Transform2D u = t;           // 拷贝
			Vector2 o = u.get_origin();  // (3, 4)
			return 100.0 * s.b + s.r + s.a + o.x + o.y; // 50 + 2 + 2 + 3 + 4 = 61
		}
	)", &out));
	CHECK(nearly(out, 61.0));
	CHECK(ASNativeValueOps::native_thunk_calls() > 0); // Color 的 opAdd/opAssign 已命中原生 thunk
}

// P3b：新原生运算符的表值对齐（Color 四则与相等、Vector4 分量乘、Transform2D 组合乘）。
// 值均取二进制精确数（0.5/0.25/0.125 等），保证期望值是确定有理数，便于检测 thunk 语义偏差。
void as_native_vo_extended_types_ops_parity() {
	if (!ASNativeValueStorage::is_native_storage_type(Variant::COLOR)) {
		return;
	}
	double out = 0.0;
	CHECK(run_double(R"(
		double main() {
			Color a(0.5, 0.25, 1.0, 1.0);
			Color b(0.25, 0.5, 0.5, 0.5);
			Color s = a + b;   // (0.75, 0.75, 1.5, 1.5)
			Color d = a - b;   // (0.25, -0.25, 0.5, 0.5)
			Color m = a * b;   // (0.125, 0.125, 0.5, 0.5)
			Color q = a / b;   // (2.0, 0.5, 2.0, 2.0)
			Vector4 v(1.0, 2.0, 3.0, 4.0);
			Vector4 w = v * v; // 分量乘 (1, 4, 9, 16)
			Transform2D t(Vector2(1, 0), Vector2(0, 1), Vector2(3, 4));
			Transform2D c = t * t;          // 组合：origin 翻倍
			Vector2 o = c.get_origin();     // (6, 8)
			bool eq = (a == a);             // opEquals
			return 100.0 * (s.r + s.g + s.b + s.a)  // 450
					+ 10.0 * (d.r + d.b + d.a)       // 12.5
					+ (m.r + m.g + m.b)              // 0.75
					+ q.r                            // 2.0
					+ (w.x + w.y + w.z + w.w)        // 30.0
					+ (eq ? 1.0 : 0.0)               // 1.0
					+ (o.x + o.y);                   // 14.0
		}
	)", &out));
	CHECK(nearly(out, 510.25));
}

