/**************************************************************************/
/*  as_binding_plan.cpp                                                   */
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

#include "as_binding_plan.h"

#include "core/config/project_settings.h"
#include "core/templates/hash_set.h"
#include "core/templates/local_vector.h"

// 34 个内建值类型（标量 BOOL/INT/FLOAT 与 OBJECT 不在此列）。顺序固定，
// 它同时决定 .d.as 的声明顺序与稳定 hash 的输入顺序。
static const Variant::Type VALUE_TYPES[] = {
	Variant::STRING, Variant::STRING_NAME, Variant::NODE_PATH, Variant::RID,
	Variant::VECTOR2, Variant::VECTOR2I, Variant::VECTOR3, Variant::VECTOR3I,
	Variant::VECTOR4, Variant::VECTOR4I, Variant::RECT2, Variant::RECT2I,
	Variant::TRANSFORM2D, Variant::PLANE, Variant::QUATERNION, Variant::AABB,
	Variant::BASIS, Variant::TRANSFORM3D, Variant::PROJECTION, Variant::COLOR,
	Variant::PACKED_BYTE_ARRAY, Variant::PACKED_INT32_ARRAY, Variant::PACKED_INT64_ARRAY,
	Variant::PACKED_FLOAT32_ARRAY, Variant::PACKED_FLOAT64_ARRAY, Variant::PACKED_STRING_ARRAY,
	Variant::PACKED_VECTOR2_ARRAY, Variant::PACKED_VECTOR3_ARRAY, Variant::PACKED_VECTOR4_ARRAY,
	Variant::PACKED_COLOR_ARRAY, Variant::ARRAY, Variant::DICTIONARY, Variant::CALLABLE, Variant::SIGNAL,
};
static const int VALUE_TYPE_COUNT = 34;

ASBindingScope ASBindingScope::from_project_settings() {
	ASBindingScope s;
	if (ProjectSettings::get_singleton()->has_setting("angel_script/class_whitelist")) {
		s.whitelist = GLOBAL_GET("angel_script/class_whitelist");
	}
	if (ProjectSettings::get_singleton()->has_setting("angel_script/class_blacklist")) {
		s.blacklist = GLOBAL_GET("angel_script/class_blacklist");
	}
	return s;
}

void ASBindingPlan::build(const ASBindingScope &p_scope) {
	// GLOBAL_DEF 幂等：已存在则不改动，仅保证键可被 ProjectSettings 导出/查询。
	GLOBAL_DEF("angel_script/class_whitelist", PackedStringArray());
	GLOBAL_DEF("angel_script/class_blacklist", PackedStringArray());

	classes.clear();
	value_types.clear();
	enums.clear();
	unbound.clear();

	// 值类型表：34 个内建 + Variant 自身 = 35。
	for (int i = 0; i < VALUE_TYPE_COUNT; i++) {
		ASBindingValueType vt;
		vt.type = VALUE_TYPES[i];
		vt.as_name = ASBindingDecl::variant_type_to_as(vt.type);
		value_types.push_back(vt);
	}
	{
		ASBindingValueType vt;
		vt.type = Variant::NIL;
		vt.as_name = "Variant";
		value_types.push_back(vt);
	}

	// 可见集 V：空白名单放行全部；黑名单优先；Object 无条件保留。
	LocalVector<StringName> all;
	ClassDB::get_class_list(all);
	HashSet<StringName> visible;
	for (const StringName &c : all) {
		if (c == StringName("Object")) {
			visible.insert(c);
			continue;
		}
		bool in_wl = p_scope.whitelist.is_empty() || p_scope.whitelist.has(String(c));
		bool in_bl = p_scope.blacklist.has(String(c));
		if (in_wl && !in_bl) {
			visible.insert(c);
		}
	}
	// 白名单非空时自动补祖先链：否则子类方法里引用父类的签名会集体变成 unbound。
	if (!p_scope.whitelist.is_empty()) {
		for (const StringName &c : all) {
			if (!visible.has(c)) {
				continue;
			}
			for (StringName p = ClassDB::get_parent_class_nocheck(c); p != StringName(); p = ClassDB::get_parent_class_nocheck(p)) {
				if (p_scope.blacklist.has(String(p))) {
					break; // 被拉黑的祖先不强制回填。
				}
				visible.insert(p);
			}
		}
	}

	// 拓扑排序：父先于子。Object 无父，必然最先。
	Vector<StringName> pending;
	for (const StringName &c : all) {
		if (visible.has(c)) {
			pending.push_back(c);
		}
	}
	HashSet<StringName> emitted;
	Vector<StringName> ordered;
	while (!pending.is_empty()) {
		Vector<StringName> next;
		bool progress = false;
		for (const StringName &c : pending) {
			StringName p = ClassDB::get_parent_class_nocheck(c);
			if (p == StringName() || !visible.has(p) || emitted.has(p)) {
				ordered.push_back(c);
				emitted.insert(c);
				progress = true;
			} else {
				next.push_back(c);
			}
		}
		if (!progress) {
			// 继承图不应有环；兜底避免死循环。
			ordered.append_array(next);
			break;
		}
		pending = next;
	}

	for (const StringName &c : ordered) {
		_build_class(c, visible);
	}
}

void ASBindingPlan::_build_class(const StringName &p_class, const HashSet<StringName> &p_visible) {
	ASBindingClass c;
	c.name = p_class;
	c.parent = ClassDB::get_parent_class_nocheck(p_class);
	c.instantiable = ClassDB::can_instantiate(p_class);

	auto refs_invisible = [&p_visible](const PropertyInfo &p_info, String *r_bad) -> bool {
		if (p_info.type == Variant::OBJECT && !p_info.class_name.is_empty() && !p_visible.has(p_info.class_name)) {
			if (r_bad) {
				*r_bad = String(p_info.class_name);
			}
			return true;
		}
		return false;
	};

	// 上游把属性访问器重新并入了方法表（get_method_list 去掉了 p_exclude_from_properties），
	// 这里按属性访问器名过滤，避免 get_*/set_* 作为方法注册、挤掉同名属性。
	HashSet<StringName> accessor_names;
	List<PropertyInfo> accessor_props;
	ClassDB::get_property_list(p_class, &accessor_props, false);
	for (const PropertyInfo &pi : accessor_props) {
		const StringName getter = ClassDB::get_property_getter(p_class, pi.name);
		const StringName setter = ClassDB::get_property_setter(p_class, pi.name);
		if (getter != StringName()) {
			accessor_names.insert(getter);
		}
		if (setter != StringName()) {
			accessor_names.insert(setter);
		}
	}

	// 方法：no_inheritance=false（每个 AS 类型自包含）。
	List<MethodInfo> methods;
	ClassDB::get_method_list(p_class, &methods, false);
	for (const MethodInfo &mi : methods) {
		if (accessor_names.has(mi.name)) {
			continue;
		}
		String decl, reason;
		const bool is_static = (mi.flags & METHOD_FLAG_STATIC) != 0;
		// 静态方法本质是全局函数：走 static_method_to_decl（声明名 <类名>_<成员名>），
		// 运行期由 register_class 以 RegisterGlobalFunction 注册，不做 RegisterObjectMethod。
		const bool decl_ok = is_static
				? ASBindingDecl::static_method_to_decl(mi, p_class, &decl, &reason)
				: ASBindingDecl::method_to_decl(mi, &decl, &reason);
		if (!decl_ok) {
			ASUnboundEntry e;
			e.owner = p_class;
			e.member = String(mi.name);
			e.reason = reason;
			unbound.push_back(e);
			continue;
		}
		String bad;
		bool invisible = refs_invisible(mi.return_val, &bad);
		for (const PropertyInfo &pi : mi.arguments) {
			if (refs_invisible(pi, &bad)) {
				invisible = true;
			}
		}
		if (invisible) {
			ASUnboundEntry e;
			e.owner = p_class;
			e.member = String(mi.name);
			e.reason = "references-invisible-type: " + bad;
			unbound.push_back(e);
			continue;
		}

		// 虚方法（GDVIRTUAL / Object 核心虚方法）在 ClassDB 里只有 MethodInfo，没有可由 AS
		// 直接调用的实现，必须归入 unbound。注意不能用 get_method()==nullptr 判断：同名实
		// 方法存在时它会返回实方法的指针，从而把虚方法误当成可绑定项，注册出「同名同参、仅
		// 返回类型不同」的重载（B4 解禁 Variant 形参/返回后撞出 asALREADY_REGISTERED 并永久
		// 污染引擎）。必须按 METHOD_FLAG_VIRTUAL 判定。
		if (mi.flags & METHOD_FLAG_VIRTUAL) {
			ASUnboundEntry e;
			e.owner = p_class;
			e.member = String(mi.name);
			e.reason = "virtual";
			unbound.push_back(e);
			continue;
		}

		ASBindingMethod m;
		m.name = String(mi.name);
		m.as_decl = decl;
		m.bind = ClassDB::get_method(p_class, mi.name);
		// 虚拟方法（GDVIRTUAL）只有 MethodInfo 没有 MethodBind：AS 侧注册出来也调不到实现，
		// 归入 unbound，保证 plan 与运行期注册、dump 三者一致。
		if (m.bind == nullptr) {
			ASUnboundEntry e;
			e.owner = p_class;
			e.member = String(mi.name);
			e.reason = "virtual";
			unbound.push_back(e);
			continue;
		}
		m.is_static = (mi.flags & METHOD_FLAG_STATIC);
		// 真 void（NIL 且无 NIL_IS_VARIANT）不编组返回值；其余（含真 Variant）按 resolve 的 kind。
		const PropertyInfo &rv = mi.return_val;
		m.return_kind = (rv.type == Variant::NIL && !(rv.usage & PROPERTY_USAGE_NIL_IS_VARIANT))
				? AS_KIND_VOID
				: ASBindingDecl::resolve(rv).kind;
		m.return_type = rv.type;
		for (const PropertyInfo &pi : mi.arguments) {
			m.param_kinds.push_back(ASBindingDecl::resolve(pi).kind);
			m.param_types.push_back(pi.type);
		}
		c.methods.push_back(m);
	}

	// 属性：含继承（每个 AS 类型自包含），跳过分类/分组标记。
	//
	// 运行期先注册方法、再注册属性；属性循环遇到「访问器名已是同类方法」就让位
	// （GetMethodByName 命中即跳过），成员仍可通过同名方法访问。上面 accessor_names
	// 只能过滤 ClassDB 明确报告了 getter/setter 的属性；Image.data 这类属性 ClassDB
	// 不报告访问器名，其 get_data 会作为方法存活，运行期于是跳过该属性。这里用同一
	// 判定在编译期确定性地让属性让位并记入 unbound，避免 dump 声明一个运行期并不存在
	// 的属性（评审 I1）。
	HashSet<String> method_names;
	for (const ASBindingMethod &m : c.methods) {
		if (!m.is_static) {
			method_names.insert(m.name);
		}
	}
	List<PropertyInfo> props;
	ClassDB::get_property_list(p_class, &props, false);
	for (const PropertyInfo &pi : props) {
		if (pi.usage & (PROPERTY_USAGE_CATEGORY | PROPERTY_USAGE_GROUP | PROPERTY_USAGE_SUBGROUP)) {
			continue;
		}
		// 属性名不一定是合法标识符：ClassDB 里存在 "frame_0/texture" 这类仅供检查器
		// 分组的索引型属性，用它拼出的访问器名（get_frame_0/texture）会被 AS 以
		// asINVALID_DECLARATION 拒绝，而任何 Register* 失败都会永久污染引擎。
		if (!String(pi.name).is_valid_identifier()) {
			ASUnboundEntry e;
			e.owner = p_class;
			e.member = String(pi.name);
			e.reason = "unsupported-property-name: " + String(pi.name);
			unbound.push_back(e);
			continue;
		}
		const bool read_only = (pi.usage & PROPERTY_USAGE_READ_ONLY);
		if (method_names.has("get_" + String(pi.name)) || (!read_only && method_names.has("set_" + String(pi.name)))) {
			ASUnboundEntry e;
			e.owner = p_class;
			e.member = String(pi.name);
			e.reason = "accessor-name-taken";
			unbound.push_back(e);
			continue;
		}
		ASBindingType t = ASBindingDecl::resolve(pi);
		if (!t.valid) {
			ASUnboundEntry e;
			e.owner = p_class;
			e.member = String(pi.name);
			e.reason = "unsupported-property-type: " + t.unbound_reason;
			unbound.push_back(e);
			continue;
		}
		// 与方法的 refs_invisible 同理：引用不可见类的属性在 AS 侧注册会因未知类型而失败
		// （而任何 Register* 失败都会永久污染引擎），必须一并在 plan 阶段过滤。
		String bad;
		if (refs_invisible(pi, &bad)) {
			ASUnboundEntry e;
			e.owner = p_class;
			e.member = String(pi.name);
			e.reason = "references-invisible-type: " + bad;
			unbound.push_back(e);
			continue;
		}
		ASBindingProperty bp;
		bp.name = String(pi.name);
		bp.as_type = t.as_name;
		bp.kind = t.kind;
		bp.type = pi.type;
		bp.read_only = (pi.usage & PROPERTY_USAGE_READ_ONLY);
		c.properties.push_back(bp);
	}

	// 枚举与常量：只取本类自有成员（no_inheritance=true），
	// 否则会在 namespace/全局里按继承深度成倍重复。
	List<StringName> enum_names;
	ClassDB::get_enum_list(p_class, &enum_names, true);
	List<String> consts;
	ClassDB::get_integer_constant_list(p_class, &consts, true);
	for (const StringName &en : enum_names) {
		ASBindingEnum be;
		be.scope = p_class;
		be.name = String(en);
		for (const String &n : consts) {
			if (ClassDB::get_integer_constant_enum(p_class, n, true) != en) {
				continue;
			}
			bool ok = false;
			int64_t v = ClassDB::get_integer_constant(p_class, n, &ok);
			if (ok) {
				be.values.push_back(Pair<String, int64_t>(n, v));
			}
		}
		enums.push_back(be);
	}
	for (const String &n : consts) {
		if (!ClassDB::get_integer_constant_enum(p_class, n, true).is_empty()) {
			continue; // 已归入某个枚举。
		}
		bool ok = false;
		int64_t v = ClassDB::get_integer_constant(p_class, n, &ok);
		if (!ok) {
			continue;
		}
		ASBindingConstant k;
		k.name = n;
		k.value = v;
		c.constants.push_back(k);
	}

	classes.push_back(c);
}

bool ASBindingPlan::has_class(const StringName &p_name) const {
	for (const ASBindingClass &c : classes) {
		if (c.name == p_name) {
			return true;
		}
	}
	return false;
}

String ASBindingPlan::serialize_stable() const {
	Vector<String> lines;
	for (const ASBindingValueType &v : value_types) {
		lines.push_back("vt:" + v.as_name);
	}
	for (const ASBindingEnum &e : enums) {
		String s = "enum:" + String(e.scope) + "::" + e.name;
		for (const Pair<String, int64_t> &p : e.values) {
			s += ":" + p.first + "=" + itos(p.second);
		}
		lines.push_back(s);
	}
	for (const ASBindingClass &c : classes) {
		lines.push_back("class:" + String(c.name) + "<" + String(c.parent) + (c.instantiable ? ":i" : ""));
		for (const ASBindingMethod &m : c.methods) {
			lines.push_back("m:" + String(c.name) + ":" + m.as_decl);
		}
		for (const ASBindingProperty &p : c.properties) {
			lines.push_back("p:" + String(c.name) + ":" + p.name + ":" + p.as_type + ":" + itos((int)p.kind) + (p.read_only ? ":ro" : ""));
		}
		for (const ASBindingConstant &k : c.constants) {
			lines.push_back("k:" + String(c.name) + ":" + k.name + "=" + itos(k.value));
		}
	}
	// 未绑定清单来自内省顺序，可能是非确定的；排序后再拼接以保证稳定。
	Vector<String> unb;
	for (const ASUnboundEntry &u : unbound) {
		unb.push_back("u:" + String(u.owner) + ":" + u.member + ":" + u.reason);
	}
	unb.sort();
	lines.append_array(unb);
	return String("\n").join(lines);
}
