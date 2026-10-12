/**************************************************************************/
/*  as_binding_decl.h                                                     */
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

#include "core/object/object.h"
#include "core/string/string_name.h"
#include "core/variant/variant.h"

// AngelScript 前向声明：本头会被不含 AS 头的测试翻译单元包含，不能直接 include <angelscript.h>。
class asIScriptGeneric;

// 跳板在运行期按形参/返回值的“类别”决定如何编组：标量走 AS 原生类型，
// 内建值类型走 Variant 存储（本模块的统一内存布局），对象走 Object*。
enum ASBindingKind {
	AS_KIND_VOID,
	AS_KIND_BOOL,
	AS_KIND_INT64,
	AS_KIND_DOUBLE,
	AS_KIND_VALUE,
	// 对象句柄按**静态类型**分成两类（spec §0 R3）：
	//   OWNING    —— 静态类型派生自 RefCounted：槽里是裸 Object*，引用计数由 AS 的 addref/release 维护。
	//   NONOWNING —— 其余（含 Object 自身）：槽里是 ObjectID，每次解引用都必须经 ObjectDB 校验。
	AS_KIND_OBJECT_OWNING,
	AS_KIND_OBJECT_NONOWNING,
};

struct ASBindingType {
	bool valid = false;
	String as_name; // "int64" / "String" / "Node@" / "Vector2"
	ASBindingKind kind = AS_KIND_VOID;
	String unbound_reason; // 仅在 valid == false 时非空。
};

class ASBindingDecl {
public:
	// Variant 标量/值类型 -> AS 类型名（OBJECT 不在此处理，因为它需要 class_name）。
	static String variant_type_to_as(Variant::Type p_type);
	// AS 类型名 -> Variant 类型（variant_type_to_as 的反查）；无对应类型返回 Variant::NIL。
	static Variant::Type as_name_to_variant_type(const String &p_as_name);
	// PropertyInfo -> AS 类型（含 OBJECT 的 class_name；枚举型属性退化为 int64）。
	static ASBindingType resolve(const PropertyInfo &p_info);
	// MethodInfo -> AS 声明串（不含所属类名）。失败时返回 false 并写 r_reason。
	static bool method_to_decl(const MethodInfo &p_info, String *r_decl, String *r_reason);
	// 静态方法 -> AS **全局函数** 声明串。AS 2.38 解析器不识别 `static`，静态类方法必须按
	// 全局函数注册；名字沿用 `<作用域>_<成员名>` 约定（与常量、枚举一致），无 self、无 const。
	// 非静态方法返回 false 且 r_reason == "not-static-method"。
	static bool static_method_to_decl(const MethodInfo &p_info, const StringName &p_scope, String *r_decl, String *r_reason);
};

// 通用跳板共用的编组辅助：值类型/对象/全局函数三处复用（Task 3/4）。
// 从 asIScriptGeneric 的第 p_index 个实参按 kind 取出 Variant。
Variant as_binding_marshal_arg(asIScriptGeneric *p_gen, int p_index, ASBindingKind p_kind, Variant::Type p_stored_type = Variant::NIL);
// 把 Variant 按 kind 写回 asIScriptGeneric 的返回位置。
void as_binding_marshal_return(asIScriptGeneric *p_gen, ASBindingKind p_kind, const Variant &p_value, Variant::Type p_return_type = Variant::NIL);
// 形参传递形式：bool/int64/double 与 `T@` 句柄按值；其余内建值类型按 const 引用。
String as_binding_render_param(const String &p_as_name);
