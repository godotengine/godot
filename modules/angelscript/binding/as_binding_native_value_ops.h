/**************************************************************************/
/*  as_binding_native_value_ops.h                                         */
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

#include "core/variant/variant.h"

#include <angelscript.h>

// 原生调用约定（asCALL_CDECL_OBJFIRST）的 thunk 签名与具体平台 ABI 绑定，必须逐平台验证。
// 这里只在已锁定的平台上启用；其余平台（含 AS_MAX_PORTABILITY：Emscripten/PPC/RISC-V 等）
// 整体回退既有 generic 路径。注意：在 AS_MAX_PORTABILITY 平台对非 asCALL_GENERIC 的调用会让
// asCScriptEngine 永久进入 configFailed，所以必须在注册前就跳过，而不是靠注册失败回退。
// 扩展平台时，在下方条件里追加已验证的组合即可（见 spec §6）。
#if !defined(AS_MAX_PORTABILITY) && defined(_MSC_VER) && defined(_M_X64)
#define AS_NATIVE_VALUE_OPS_ENABLED 1
#else
#define AS_NATIVE_VALUE_OPS_ENABLED 0
#endif

// 值类型原生运算符/方法注册入口（P1）。
//
// 命中表项的运算符/方法走原生 C++ thunk，绕过 asCALL_GENERIC 跳板；未命中返回 false，
// 调用方回退既有 generic 路径。存储始终是 Variant（sizeof(Variant)），thunk 只做
// Variant 存储 ↔ 原生值 的转换。
namespace ASNativeValueOps {

bool try_add_op(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const String &p_op, const String &p_decl);
bool try_add_method(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const StringName &p_method, const String &p_decl);

// Packed 标量数组索引访问器（get/set_opIndex）原生注册（P2）。
// 仅覆盖元素为 int64/double 的 Packed*Array；成功注册 get 与 set 各一次并返回 true。
bool try_add_index_accessor(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const String &p_get_decl, const String &p_set_decl);

int native_registration_count();
int native_thunk_calls();
void reset_native_thunk_calls();

// 每个原生 thunk 开头调用，累计 thunk 调用次数（测试用）。
void note_thunk_call();

} // namespace ASNativeValueOps
