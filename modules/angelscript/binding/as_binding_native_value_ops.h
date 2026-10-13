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
//
// 门控宏在任何重头文件之前定义：误组合可第一时间 #error，而不是先拉进 Variant 全量依赖。
// 门控取值（诊断用，见 native_ops_gate_name）：
#define AS_NATIVE_VALUE_OPS_GATE_VERIFIED 0       // 已验证平台（MSVC x64），默认开启原生 thunk。
#define AS_NATIVE_VALUE_OPS_GATE_OVERRIDE 1       // 未验证平台经 AS_NATIVE_VALUE_OPS_ENABLE_UNVERIFIED 逃生舱开启。
#define AS_NATIVE_VALUE_OPS_GATE_FORCED_GENERIC 2 // 显式定义 AS_NATIVE_VALUE_OPS_FORCE_GENERIC 强制 generic。
#define AS_NATIVE_VALUE_OPS_GATE_PLATFORM 3       // 未验证平台且未开逃生舱，整体回退 generic。
// AS_NATIVE_VALUE_OPS_ENABLE_UNVERIFIED 仅用于平台 ABI 校验性试验；禁止直接合入发布分支。
// AS_NATIVE_VALUE_OPS_FORCE_GENERIC 用于在已启用平台复现完整 generic 回退路径（测试/对比用）。
#if defined(AS_NATIVE_VALUE_OPS_FORCE_GENERIC)
#define AS_NATIVE_VALUE_OPS_ENABLED 0
#define AS_NATIVE_VALUE_OPS_GATE AS_NATIVE_VALUE_OPS_GATE_FORCED_GENERIC
#elif !defined(AS_MAX_PORTABILITY) && defined(_MSC_VER) && defined(_M_X64)
#define AS_NATIVE_VALUE_OPS_ENABLED 1
#define AS_NATIVE_VALUE_OPS_GATE AS_NATIVE_VALUE_OPS_GATE_VERIFIED
#elif defined(AS_NATIVE_VALUE_OPS_ENABLE_UNVERIFIED) && !defined(AS_MAX_PORTABILITY)
#define AS_NATIVE_VALUE_OPS_ENABLED 1
#define AS_NATIVE_VALUE_OPS_GATE AS_NATIVE_VALUE_OPS_GATE_OVERRIDE
#else
#define AS_NATIVE_VALUE_OPS_ENABLED 0
#define AS_NATIVE_VALUE_OPS_GATE AS_NATIVE_VALUE_OPS_GATE_PLATFORM
#endif

// AS_MAX_PORTABILITY 下任何非 generic 调用都会永久污染引擎；此组合必须编译期拦截。
#if defined(AS_MAX_PORTABILITY) && defined(AS_NATIVE_VALUE_OPS_ENABLE_UNVERIFIED)
#error "AS_MAX_PORTABILITY 平台禁止启用原生 thunk：非 generic 调用会使 asCScriptEngine 永久进入 configFailed。"
#endif

#include "core/variant/variant.h"

#include <angelscript.h>

// 原生 thunk 的 ABI 只在下述组合下逐平台验证过：64 位指针宽度 + AngelScript 2.38.0 的 callfunc
// 实现。任何一项变化都必须重新走上面的平台启用流程，否则调用约定可能静默错位。
#if AS_NATIVE_VALUE_OPS_ENABLED
static_assert(sizeof(void *) == 8, "原生值 thunk 假定 64 位指针宽度；32 位平台必须回退 generic。");
static_assert(ANGELSCRIPT_VERSION == 23800, "原生值 thunk ABI 仅针对 AngelScript 2.38.0 校验；升级 vendored 版本后必须逐平台重新验证再启用。");
#endif

// 值类型原生运算符/方法注册入口（P1）。
//
// 命中表项的运算符/方法走原生 C++ thunk，绕过 asCALL_GENERIC 跳板；未命中返回 false，
// 调用方回退既有 generic 路径。存储始终是 Variant（sizeof(Variant)），thunk 只做
// Variant 存储 ↔ 原生值 的转换。
namespace ASNativeValueOps {

// Boxed types retain Variant storage and their existing copy/alias semantics.
bool try_add_boxed_lifecycle(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type);
bool try_add_variant_conversions(asIScriptEngine *p_engine);
bool try_add_literal_support(asIScriptEngine *p_engine);

bool try_add_op(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const String &p_op, const String &p_decl);
bool try_add_method(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const StringName &p_method, const String &p_decl);

// Packed 标量数组索引访问器（get/set_opIndex）原生注册（P2）。
// 仅覆盖元素为 int64/double 的 Packed*Array；成功注册 get 与 set 各一次并返回 true。
bool try_add_index_accessor(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const String &p_get_decl, const String &p_set_decl);

// Dictionary 键值访问器（get/set_opIndex）原生注册；不与普通索引访问器合并。
bool try_add_keyed_accessor(asIScriptEngine *p_engine, const String &p_type_name, Variant::Type p_type, const String &p_get_decl, const String &p_set_decl);

int native_registration_count();
int native_thunk_calls();
void reset_native_thunk_calls();

// 门控诊断：native_ops_enabled() 与编译期 AS_NATIVE_VALUE_OPS_ENABLED 一致；
// native_ops_gate_name() 返回稳定的门控原因字符串（verified-msvc-x64 / unverified-override /
// forced-generic / platform-fallback），供测试与日志区分「已验证平台」与「未验证回退」。
bool native_ops_enabled();
const char *native_ops_gate_name();

// 每个原生 thunk 开头调用，累计 thunk 调用次数（测试用）。
void note_thunk_call();

} // namespace ASNativeValueOps
