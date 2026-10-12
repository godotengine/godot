/**************************************************************************/
/*  as_binding_dumper.h                                                   */
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

#include "as_binding_plan.h"

#include "core/error/error_list.h"
#include "core/string/ustring.h"

// API 声明转储（spec §3.8.7）：把绑定计划渲染成 `.d.as` 声明文件 + 未绑定清单。
// 与运行期注册器消费同一份 plan，因此产物里的类型名/枚举名/方法签名与引擎实际注册一致，
// 但有一处已知偏差：
//  1. plan 的类成员集含整条继承链（AS 不支持注册类型之间的继承，只能靠 opImplCast），
//     所以继承成员会在每个派生类下重复声明（`get_class()` 同时出现在 Object 与 Node 下）。
// 运行期「属性访问器名已被同类方法占用」的让位不再算偏差：plan 用同一判定在编译期把这类
// 属性归入 unbound（reason accessor-name-taken），dump 不再声明它们（评审 I1 已解决）。
// 静态方法运行期是全局函数（RegisterGlobalFunction，名 <类名>_<成员名>），dump 同样放在
// 类体外的「global functions」区，与类方法区分。
class ASBindingDumper {
	static String _render_declaration(const ASBindingPlan &p_plan);
	static String _render_unbound(const ASBindingPlan &p_plan);

public:
	// 写入 <dir>/angelscript_api.d.as 与 <dir>/angelscript_unbound.txt；目录不存在时递归创建。
	static Error write(const ASBindingPlan &p_plan, const String &p_dir);

	// 内容 hash（sha256）：热更侧用它判断脚本是否与当前引擎 API 匹配。
	static String compute_api_version(const ASBindingPlan &p_plan);
};
