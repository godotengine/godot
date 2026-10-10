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
//
// 已验证组合：MSVC x64（Windows）。其它平台的原生 thunk ABI 尚未校验，故默认关闭。
// 在新平台启用前必须依次完成（见 spec §6 与本优化计划 P4）：
//   1) 用 asIScriptEngine / as_callfunc 文档核对 asCALL_CDECL_OBJFIRST 的传参与返回约定
//      与本文件 thunk 形状（`void op(T* ret, const T* self, ...)`）一致；
//   2) 在本机跑 tests/bench_project 与 `--headless --test` 全量，确认引擎未进入 configFailed；
//   3) 把该平台加入下方 allowlist（而不是全局打开）。
// 仅用于上述校验性试验的逃生舱：显式定义 AS_NATIVE_VALUE_OPS_ENABLE_UNVERIFIED 才会在
// 未知平台打开；禁止直接合入发布分支。
#if !defined(AS_MAX_PORTABILITY) && defined(_MSC_VER) && defined(_M_X64)
#define AS_NATIVE_VALUE_OPS_ENABLED 1
#elif defined(AS_NATIVE_VALUE_OPS_ENABLE_UNVERIFIED) && !defined(AS_MAX_PORTABILITY)
#define AS_NATIVE_VALUE_OPS_ENABLED 1
#else
#define AS_NATIVE_VALUE_OPS_ENABLED 0
#endif

// AS_MAX_PORTABILITY 下任何非 generic 调用都会永久污染引擎；此组合必须编译期拦截。
#if defined(AS_MAX_PORTABILITY) && defined(AS_NATIVE_VALUE_OPS_ENABLE_UNVERIFIED)
#error "AS_MAX_PORTABILITY 平台禁止启用原生 thunk：非 generic 调用会使 asCScriptEngine 永久进入 configFailed。"
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
