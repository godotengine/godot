/**************************************************************************/
/*  test_angelscript_binding_dump.cpp                                     */
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

#include "../binding/as_binding_dumper.h"
#include "../binding/as_binding_plan.h"

#define ANGELSCRIPT_BINDING_DUMP_TESTS_IMPL
#include "test_angelscript_binding_dump.h"

#include "core/io/file_access.h"
#include "core/string/print_string.h"

static String _read_all_text(const String &p_path) {
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	if (f.is_null()) {
		return String();
	}
	return f->get_as_text();
}

void as_binding_dump_files_and_determinism() {
	// 空 scope = 默认全量（白/黑名单为空表示全部放行）。
	ASBindingPlan plan;
	plan.build(ASBindingScope());

	CHECK(ASBindingDumper::write(plan, "user://as_dump_a") == OK);
	CHECK(ASBindingDumper::write(plan, "user://as_dump_b") == OK);

	const String sa = _read_all_text("user://as_dump_a/angelscript_api.d.as");
	const String sb = _read_all_text("user://as_dump_b/angelscript_api.d.as");
	REQUIRE(!sa.is_empty());
	REQUIRE(!sb.is_empty());
	if (sa.is_empty() || sb.is_empty()) {
		return; // 无异常模式：REQUIRE 失败后必须显式返回，否则后续调用会崩溃。
	}

	// 确定性：同一份 plan 两次 dump 必须逐字节一致（HashMap 遍历顺序不得泄进产物）。
	CHECK(sa == sb);
	CHECK(sa.contains("// angelscript-api-version: "));
	CHECK(sa.contains("class Node"));
	CHECK(sa.contains("class Vector2")); // 值类型也进声明文件。
	CHECK(sa.contains("enum Node_ProcessMode")); // 运行期真实枚举名（<声明类>_<枚举名>）。
	CHECK(sa.contains("Object_NOTIFICATION_PREDELETE")); // BIND_CONSTANT 全局常量。

	// 头部版本号是 64 位十六进制 sha256（内容 hash，便于热更侧比对）。
	const String version = sa.get_slice("\n", 0).trim_prefix("// angelscript-api-version: ").strip_edges();
	CHECK(version.length() == 64);
	CHECK(version.is_valid_hex_number(false));

	const String ua = _read_all_text("user://as_dump_a/angelscript_unbound.txt");
	REQUIRE(!ua.is_empty());
	if (ua.is_empty()) {
		return;
	}
	CHECK(ua.ends_with("\n"));
}

void as_binding_dump_content_matches_runtime() {
	// 只看一条窄链路，便于断言“dump 与运行期注册完全同名”。
	ASBindingScope scope;
	for (const char *n : { "Object", "RefCounted", "Resource", "Node", "CanvasItem", "Node2D", "Sprite2D" }) {
		scope.whitelist.push_back(String(n));
	}
	ASBindingPlan plan;
	plan.build(scope);

	CHECK(ASBindingDumper::write(plan, "user://as_dump_c") == OK);

	const String ds = _read_all_text("user://as_dump_c/angelscript_api.d.as");
	REQUIRE(!ds.is_empty());
	if (ds.is_empty()) {
		return;
	}
	CHECK(ds.contains("class Node2D"));
	CHECK(ds.contains("Node@ opImplCast() const")); // 继承边：每个可见祖先一条。
	CHECK(ds.contains("get_position"));
	CHECK(ds.contains("set_position"));
	CHECK(ds.contains("add_child"));
	// 白名单外的类不得出现（可见性过滤必须同样作用于 dump）。
	CHECK_FALSE(ds.contains("class Sprite3D"));

	const String ua = _read_all_text("user://as_dump_c/angelscript_unbound.txt");
	REQUIRE(!ua.is_empty());
	if (ua.is_empty()) {
		return;
	}
	// 每行三列：owner<TAB>member<TAB>reason。
	const PackedStringArray lines = ua.split("\n");
	for (const String &line : lines) {
		if (line.is_empty()) {
			continue;
		}
		CHECK(line.split("\t").size() == 3);
	}
}

void as_binding_dump_static_methods_are_global() {
	// 静态方法运行期以 RegisterGlobalFunction 注册（全局名 <类名>_<成员名>），dump 必须
	// 同样放在类体外，不能在 class 体内声明成实例方法。
	ASBindingScope scope;
	scope.whitelist.push_back("Object");
	scope.whitelist.push_back("Node");
	ASBindingPlan plan;
	plan.build(scope);

	const String dir = "user://as_dump_static";
	REQUIRE(ASBindingDumper::write(plan, dir) == OK);
	const String decl = _read_all_text(dir.path_join("angelscript_api.d.as"));
	REQUIRE(!decl.is_empty());
	if (decl.is_empty()) {
		return;
	}

	const int start = decl.find("class Node {");
	REQUIRE(start >= 0);
	const int open = decl.find("{", start);
	REQUIRE(open >= 0);
	const int close = decl.find("}", open);
	REQUIRE(close > open);
	if (start < 0 || open < 0 || close <= open) {
		return;
	}
	const String block = decl.substr(open, close - open);
	CHECK(block.find("Node_print_orphan_nodes") < 0); // 不得作为类方法声明。
	CHECK(decl.contains("// ---------------- global functions ----------------"));
	CHECK(decl.find("void Node_print_orphan_nodes();") >= 0); // 作为全局函数声明。
}

#include "core/templates/hash_map.h"

void as_binding_dump_accessor_name_taken() {
	ASBindingPlan plan;
	plan.build(ASBindingScope());

	const String dir = "user://as_dump_accessor";
	REQUIRE(ASBindingDumper::write(plan, dir) == OK);
	const String decl = _read_all_text(dir.path_join("angelscript_api.d.as"));

	// Image.data 在运行期让位（get_data 是同类真实方法）：dump 只声明方法，不再声明属性。
	const int start = decl.find("class Image {");
	REQUIRE(start >= 0);
	const int open = decl.find("{", start);
	const int close = decl.find("}", open);
	REQUIRE(open >= 0);
	REQUIRE(close > open);
	const String block = decl.substr(open, close - open);
	CHECK(block.find("get_data() const;") >= 0);
	CHECK(block.find("get_data() const property;") < 0);

	// 反向保护：dump 声明的 getter 行数必须与 plan 的属性总数一致（一致性修复不得误删）。
	int expected_getters = 0;
	for (const ASBindingClass &c : plan.get_classes()) {
		expected_getters += c.properties.size();
	}
	int actual_getters = 0;
	int pos = 0;
	while (true) {
		const int idx = decl.find("() const property;", pos);
		if (idx < 0) {
			break;
		}
		actual_getters++;
		pos = idx + 1;
	}
	CHECK(actual_getters == expected_getters);
}
