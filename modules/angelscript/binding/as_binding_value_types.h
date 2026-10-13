/**************************************************************************/
/*  as_binding_value_types.h                                              */
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

#include "core/error/error_list.h"
#include "core/variant/variant.h"

#include <angelscript.h>

// 内建 Variant 值类型注册器（spec §3.3 / §3.8 M2 规格）。
//
// 设计要点：34 个内建值类型 + Variant 自身在 AS 侧都是独立的 asOBJ_VALUE 类型，
// 但底层存储统一为 Godot Variant（sizeof(Variant)）：AS 的对象内存就是一块 Variant，
// 构造/析构/赋值分别是 placement-new / ~Variant / operator=。
// 内建方法、成员、运算符、索引全部由 Variant 的内省表驱动注册，跳板统一走 asCALL_GENERIC，
// 因此新增值类型或 Godot 内省项无需改动本文件。
class ASBindingValueTypes {
public:
	// 注册全部内建值类型与 string 工厂。幂等：已注册的类型通过 GetTypeInfoByName 跳过。
	static Error register_all(asIScriptEngine *p_engine);

	// 注册依赖对象类型的 Variant 转换构造器（`Variant(Object @)`）。
	// 必须在全部对象骨架注册完成之后调用：值类型阶段解析不到 `Object` 类型名。
	static Error register_object_conversions(asIScriptEngine *p_engine);

	// asCALL_GENERIC 跳板：内建方法/成员/运算符/索引共用；具体行为由 GetUserData 的绑定记录决定。
	static void generic_value_call(asIScriptGeneric *p_gen);

	// 从 AS `string` 的对象槽里取出字符串内容。
	// `string` 是唯一不以 Variant 为存储的绑定值类型：它按 sizeof(void *) 注册，
	// 槽里装的是驻留工厂对象的指针。槽内容对 .cpp 之外不透明，调试器要读它就只能走这里。
	// p_slot 是对象槽地址（即 AS 的 this / GetAddressOfVar 结果），不是工厂指针本身。
	static String string_from_slot(const void *p_slot);

	// const string &in call argument: literal factory pointer OR variable slot.
	// Returned reference belongs to the immutable process-lifetime intern pool.
	static const String &string_from_argument(const void *p_arg);
};
