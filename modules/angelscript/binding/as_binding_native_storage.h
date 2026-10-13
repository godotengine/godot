/**************************************************************************/
/*  as_binding_native_storage.h                                           */
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

class asIScriptEngine;
class asIScriptGeneric;

// P3 阶段 1：原生值存储（Vector2/Vector3/Vector2i/Vector3i）。
//
// 目标：把这 4 个向量类型在 AS 侧的存储由 sizeof(Variant) 改为原生的 sizeof(T)，
// 并整体绕过 asCALL_GENERIC 跳板。仅在已验证平台（MSVC x64）启用；未验证平台
// 全部函数返回 false，让调用方走原 generic 路径（见 spec §6）。
//
// 返回 true = 原生方式已处理，调用方必须跳过 generic 骨架/生命周期注册。
// 返回 false = 未处理（平台门控关闭或注册失败），调用方继续 generic。
namespace ASNativeValueStorage {

// 门控 + 类型判定：仅 MSVC x64 且为 4 个向量类型时为真。
bool is_native_storage_type(Variant::Type p_type);
// Explicit experimental opt-in, never changes the native hidden-return-slot ABI.
bool is_trivial_probe_type(Variant::Type p_type);
// 生命周期阶段：为原生类型注册原生（非 generic）默认构造/拷贝构造/析构。
// 命中→true（调用方跳过 generic 行为注册）；非原生类型或未启用平台→false。
bool register_native_lifecycle(asIScriptEngine *p_engine, const String &p_name, Variant::Type p_type);
// 骨架阶段：以 sizeof(T) + 非 POD 子标志注册；命中→true（调用方跳过 Variant 骨架）。
bool register_skeleton(asIScriptEngine *p_engine, Variant::Type p_type);
// 原生槽 ↔ Variant。
Variant native_to_variant(Variant::Type p_type, const void *p_slot);
void variant_to_native(Variant::Type p_type, const Variant &p_value, void *p_slot);
// 以原生 T 写入 generic 返回槽（AS 经原生拷贝构造复制）。
bool write_native_return(asIScriptGeneric *p_gen, Variant::Type p_type, const Variant &p_value);

} // namespace ASNativeValueStorage
