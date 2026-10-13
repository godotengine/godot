/**************************************************************************/
/*  as_binding_value_types.cpp                                            */
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

#include "as_binding_value_types.h"

#include "as_binding_decl.h"
#include "as_binding_native_storage.h"
#include "as_binding_native_value_ops.h"

#include "core/os/memory.h"
#include "core/string/string_name.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"
#include "core/variant/callable.h"
#include "core/variant/variant_internal.h"

#include <angelscript.h>
#include <cstdint>
#include <cstring>
#include <new>

// 跳板行为分类：决定 generic_value_call 如何解释 self/实参。
enum ASValueBindingKind {
	VT_CTOR_DEFAULT, // self = 待构造内存；无参构造
	VT_CTOR_COPY, // arg0 = 同类型对象
	VT_CTOR_ARGS, // argN = 内省构造器实参
	VT_DTOR,
	VT_ASSIGN, // arg0 = 同类型对象；返回 T&
	VT_METHOD, // 内省内建方法
	VT_MEMBER_GET, // 内省成员读
	VT_MEMBER_SET, // 内省成员写
	VT_INDEX_GET, // opIndex
	VT_INDEX_SET, // opIndexAssign
	VT_KEYED_GET, // opIndex(key)
	VT_KEYED_SET, // opIndexAssign(value, key)
	VT_OP, // 运算符
	VT_CONV, // Variant -> 标量/String 转换
	VT_NOOP, // 无操作（string 析构等）
	VT_CTOR_FROM_STRING, // String/Variant(const string &in)
	VT_ADD_STRING, // String + const string &in
	VT_STRING_DEFAULT, // string() -> 空串
	VT_STRING_COPY, // string(const string &in)
	VT_STRING_ASSIGN, // string &opAssign(const string &in)
	VT_STRING_FROM_GODOT, // string(const String &in)
	VT_STRING_LENGTH, // int64 length() const
	VT_STATIC_METHOD, // 静态内建方法：无对象 this，注册为 <类型>_<方法> 全局函数
};

// 不可变绑定记录通过注册 auxiliary 直接传给跳板；UserData 仅保留供外部检查。
// 注册只发生一次（register_all 幂等），因此这些记录的生命周期与进程一致，不做释放。
struct ASValueBinding {
	ASValueBindingKind kind = VT_METHOD;
	Variant::Type type = Variant::NIL;
	Variant::Type return_type = Variant::NIL;
	ASBindingKind return_kind = AS_KIND_VOID;
	Vector<ASBindingKind> param_kinds;
	Vector<Variant::Type> param_types; // 与 param_kinds 平行：实参的 Variant::Type（标量占位 NIL）。
	int arg_count = 0;
	Variant::ValidatedBuiltInMethod method = nullptr;
	Variant::ValidatedOperatorEvaluator op_eval = nullptr;
	Variant::ValidatedGetter member_get = nullptr;
	Variant::ValidatedSetter member_set = nullptr;
};

static void attach_binding(asIScriptEngine *p_engine, int p_id, ASValueBinding *p_binding) {
	if (p_id < 0 || !p_binding) {
		return;
	}
	asIScriptFunction *f = p_engine->GetFunctionById(p_id);
	if (f) {
		f->SetUserData(p_binding);
	}
}

// 通用调用约定没有 SetException，需通过当前执行上下文抛出。
static void set_exception(const char *p_message) {
	asIScriptContext *ctx = asGetActiveContext();
	if (ctx) {
		ctx->SetException(p_message);
	}
}

static void add_method(asIScriptEngine *p_engine, const CharString &p_type, const String &p_decl, ASValueBinding *p_binding) {
	CharString decl = p_decl.utf8();
	int id = p_engine->RegisterObjectMethod(p_type.get_data(), decl.get_data(), asFUNCTION(ASBindingValueTypes::generic_value_call), asCALL_GENERIC, p_binding);
	attach_binding(p_engine, id, p_binding);
}

static void add_behaviour(asIScriptEngine *p_engine, const CharString &p_type, asEBehaviours p_behaviour, const String &p_decl, ASValueBinding *p_binding) {
	CharString decl = p_decl.utf8();
	int id = p_engine->RegisterObjectBehaviour(p_type.get_data(), p_behaviour, decl.get_data(), asFUNCTION(ASBindingValueTypes::generic_value_call), asCALL_GENERIC, p_binding);
	attach_binding(p_engine, id, p_binding);
}

// 静态内建方法没有对象 this，按 <类型>_<方法> 全局函数注册（与 ClassDB 静态方法
// 注册为全局函数的约定一致）。
static void add_global_function(asIScriptEngine *p_engine, const String &p_decl, ASValueBinding *p_binding) {
	CharString decl = p_decl.utf8();
	int id = p_engine->RegisterGlobalFunction(decl.get_data(), asFUNCTION(ASBindingValueTypes::generic_value_call), asCALL_GENERIC, p_binding);
	attach_binding(p_engine, id, p_binding);
}

// 全局符号（函数/属性）重名会让 Register* 失败并永久置位引擎 configFailed，
// 故注册前必须查重；这同时让 register_all 天然幂等。
static bool global_symbol_taken(asIScriptEngine *p_engine, const String &p_name) {
	for (asUINT i = 0; i < p_engine->GetGlobalFunctionCount(); i++) {
		if (String(p_engine->GetGlobalFunctionByIndex(i)->GetName()) == p_name) {
			return true;
		}
	}
	for (asUINT i = 0; i < p_engine->GetGlobalPropertyCount(); i++) {
		const char *n = nullptr;
		p_engine->GetGlobalPropertyByIndex(i, &n);
		if (n && String(n) == p_name) {
			return true;
		}
	}
	return false;
}

// ---------- string 互操作 ----------
// 本仓库的 AngelScript 2.38 不含内建 string 类型，且字符串字面量依赖
// asIScriptEngine::RegisterStringFactory。工厂对象由我们定义：一个持有 Godot String 的
// 驻留对象（intern），因此不会泄漏，也不需要（也不允许）在 AS 侧做引用计数。
struct ASGodotStringObject {
	String str;
};

static HashMap<String, ASGodotStringObject *> g_interned_strings;
// 工厂指针集合：用于区分「字面量实参」（AS 直接压入工厂指针）与「变量实参」
// （AS 压入保存指针的存储地址）。见 string_slot()。
static HashSet<const ASGodotStringObject *> g_interned_string_ptrs;

static const void *intern_string(const String &p_str) {
	HashMap<String, ASGodotStringObject *>::Iterator it = g_interned_strings.find(p_str);
	if (it) {
		return it->value;
	}
	ASGodotStringObject *o = memnew(ASGodotStringObject);
	o->str = p_str;
	g_interned_strings.insert(p_str, o);
	g_interned_string_ptrs.insert(o);
	return o;
}

class ASGodotStringFactory : public asIStringFactory {
public:
	const void *GetStringConstant(const char *data, asUINT length) override {
		return intern_string(String::utf8(data, (int)length));
	}
	int ReleaseStringConstant(const void * /*str*/) override {
		return 0; // 驻留对象与进程同生命周期。
	}
	int GetRawStringData(const void *str, char *data, asUINT *length) const override {
		if (!str) {
			if (length) {
				*length = 0;
			}
			return 0;
		}
		CharString utf8 = ((const ASGodotStringObject *)str)->str.utf8();
		asUINT n = (asUINT)utf8.length();
		if (data && length) {
			asUINT copy = *length < n ? *length : n;
			memcpy(data, utf8.get_data(), copy);
		}
		if (length) {
			*length = n;
		}
		return 0;
	}
};

static ASGodotStringFactory g_string_factory;

// ---------- 渲染辅助 ----------
static String param_type_name(Variant::Type p_type) {
	if (p_type == Variant::NIL) {
		return "Variant";
	}
	return ASBindingDecl::variant_type_to_as(p_type);
}

static ASBindingKind kind_of_type(Variant::Type p_type) {
	switch (p_type) {
		case Variant::BOOL:
			return AS_KIND_BOOL;
		case Variant::INT:
			return AS_KIND_INT64;
		case Variant::FLOAT:
			return AS_KIND_DOUBLE;
		case Variant::OBJECT:
			// 值类型方法没有静态 class_name 可用（且 variant_type_to_as 不处理 OBJECT，
			// 这类签名在 render_args 阶段就会被拒绝），此处仅为防御性填充：按“任意 Object”= 非拥有。
			return AS_KIND_OBJECT_NONOWNING;
		default:
			return AS_KIND_VALUE; // NIL(Variant) 与全部内建值类型统一走 Variant 存储。
	}
}

static bool render_args(const Vector<PropertyInfo> &p_args, String *r_args, Vector<ASBindingKind> *r_kinds, Vector<Variant::Type> *r_types = nullptr) {
	String args;
	for (int i = 0; i < p_args.size(); i++) {
		String name = param_type_name(p_args[i].type);
		if (name.is_empty()) {
			return false;
		}
		if (i > 0) {
			args += ", ";
		}
		args += as_binding_render_param(name);
		if (r_kinds) {
			r_kinds->push_back(kind_of_type(p_args[i].type));
		}
		if (r_types) {
			r_types->push_back(p_args[i].type);
		}
	}
	if (r_args) {
		*r_args = args;
	}
	return true;
}

// ---------- 单类型注册 ----------
// 第一阶段：只注册类型骨架（asOBJ_VALUE，内建值类型另加 asOBJ_APP_CLASS）。必须先让所有值类型都存在，
// 内省出的构造函数/方法签名里引用的其它值类型才能被 AS 解析；
// 否则任一 Register* 失败都会让 asCScriptEngine 永久进入 configFailed。
static void register_type_skeleton(asIScriptEngine *p_engine, const String &p_name, int p_size, bool p_native_app_class) {
	CharString cname = p_name.utf8();
	if (p_engine->GetTypeInfoByName(cname.get_data())) {
		return; // 幂等。
	}
	// asOBJ_APP_CLASS 仅启用“按值返回/传参的原生 C++ 函数”合法；不带子标志 ⇒ 不改变脚本侧
	// generic 的拷贝/析构/赋值语义（见 docs/superpowers/specs/2026-10-10-angelscript-vector-native-ops-design.md §4.1）。
	// 只有需要原生 thunk 的 34 个内建值类型才加该标志；Variant / string 仍走纯 generic，不加。
	asDWORD flags = asOBJ_VALUE;
	if (p_native_app_class) {
		flags |= asOBJ_APP_CLASS;
	}
	p_engine->RegisterObjectType(cname.get_data(), p_size, flags);
}

// 第二阶段：骨架齐备后注册构造/析构/方法/属性/索引/运算符。
static void register_type_members(asIScriptEngine *p_engine, const String &p_name, Variant::Type p_type) {
	CharString cname = p_name.utf8();

	// 原生存储类型的默认构造/拷贝构造/析构注册为原生（非 generic）行为：热路径上每个
	// 临时对象的析构都经 generic 跳板时，其开销会主导循环（实测使 bench_vector ≈280 ns/op；
	// 原生化后 ≈60）。未启用平台或非原生类型返回 false，保持原 generic 路径。
	if (!ASNativeValueStorage::register_native_lifecycle(p_engine, p_name, p_type) &&
			!ASNativeValueOps::try_add_boxed_lifecycle(p_engine, p_name, p_type)) {
		{
			ASValueBinding *b = memnew(ASValueBinding);
			b->kind = VT_CTOR_DEFAULT;
			b->type = p_type;
			add_behaviour(p_engine, cname, asBEHAVE_CONSTRUCT, "void f()", b);
		}
		{
			ASValueBinding *b = memnew(ASValueBinding);
			b->kind = VT_CTOR_COPY;
			b->type = p_type;
			add_behaviour(p_engine, cname, asBEHAVE_CONSTRUCT, "void f(const " + p_name + " &in)", b);
		}
		{
			ASValueBinding *b = memnew(ASValueBinding);
			b->kind = VT_DTOR;
			b->type = p_type;
			add_behaviour(p_engine, cname, asBEHAVE_DESTRUCT, "void f()", b);
		}
	}
	if (!ASNativeValueStorage::is_trivial_probe_type(p_type)) {
		// AS 没有 asBEHAVE_ASSIGNMENT；opAssign 作为运算符方法注册。
		String decl = p_name + " &opAssign(const " + p_name + " &in)";
		if (!ASNativeValueOps::try_add_op(p_engine, p_name, p_type, "opAssign", decl)) {
			ASValueBinding *b = memnew(ASValueBinding);
			b->kind = VT_ASSIGN;
			b->type = p_type;
			add_method(p_engine, cname, decl, b);
		}
	}

	// 内省构造器（无参构造已由上面统一注册）。
	List<MethodInfo> ctors;
	Variant::get_constructor_list(p_type, &ctors);
	for (const MethodInfo &mi : ctors) {
		if (mi.arguments.is_empty()) {
			continue;
		}
		// 单参且参数类型就是自身类型的内省构造器等价于拷贝构造；拷贝构造已在上面
		// 注册（原生生命周期或 generic）。若再注册一次，`Vector2(a)` 会因两个完全
		// 相同的 `Vector2(const Vector2 &in)` 报「Multiple matching signatures」。
		if (mi.arguments.size() == 1 && mi.arguments[0].type == p_type) {
			continue;
		}
		String args;
		Vector<ASBindingKind> kinds;
		Vector<Variant::Type> types;
		if (!render_args(mi.arguments, &args, &kinds, &types)) {
			continue;
		}
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_CTOR_ARGS;
		b->type = p_type;
		b->arg_count = mi.arguments.size();
		b->param_kinds = kinds;
		b->param_types = types;
		add_behaviour(p_engine, cname, asBEHAVE_CONSTRUCT, "void f(" + args + ")", b);
	}

	// 成员访问器名：内建方法清单里也有同名 get_X/set_X（如 Transform2D.get_origin），
	// 若两边都注册会 asALREADY_REGISTERED 并污染引擎。成员统一走虚属性，内建方法遇同名则跳过。
	List<StringName> members;
	Variant::get_member_list(p_type, &members);
	HashSet<String> member_accessors;
	for (const StringName &mn : members) {
		member_accessors.insert("get_" + String(mn));
		member_accessors.insert("set_" + String(mn));
	}

	// 内省内建方法。
	List<StringName> builtin_methods;
	Variant::get_builtin_method_list(p_type, &builtin_methods);
	for (const StringName &mn : builtin_methods) {
		if (Variant::is_builtin_method_vararg(p_type, mn)) {
			continue;
		}
		if (member_accessors.has(String(mn))) {
			continue; // 已由成员虚属性覆盖。
		}
		Variant::ValidatedBuiltInMethod vm = Variant::get_validated_builtin_method(p_type, mn);
		if (!vm) {
			continue;
		}
		MethodInfo mi = Variant::get_builtin_method_info(p_type, mn);
		// NIL 返回值有两种语义：真正的 void，以及返回 Variant。后者由内省带上
		// PROPERTY_USAGE_NIL_IS_VARIANT（见 variant_call.cpp get_method_info）。
		// 若把 Variant 返回值误判为 void，脚本侧 `Variant r = a.pop_back();` 会因
		// 数据源为 void 而报错，且副作用之外的返回值被静默丢弃。
		bool returns_variant = (mi.return_val.type == Variant::NIL) && (mi.return_val.usage & PROPERTY_USAGE_NIL_IS_VARIANT);
		bool returns_void = (mi.return_val.type == Variant::NIL) && !returns_variant;
		String ret = returns_void ? "void" : param_type_name(mi.return_val.type);
		if (ret.is_empty()) {
			continue;
		}
		String args;
		Vector<ASBindingKind> kinds;
		Vector<Variant::Type> types;
		if (!render_args(mi.arguments, &args, &kinds, &types)) {
			continue;
		}
		bool is_static = Variant::is_builtin_method_static(p_type, mn);
		if (is_static) {
			// 静态内建方法的 validated thunk 忽略 base（variant_call.cpp STATIC_METHOD_CLASS），
			// 可安全以 nullptr 调用，故注册为 <类型>_<方法> 全局函数。
			String gname = p_name + "_" + String(mn);
			if (global_symbol_taken(p_engine, gname)) {
				continue;
			}
			String gdecl = ret + " " + gname + "(" + args + ")";
			ASValueBinding *sb = memnew(ASValueBinding);
			sb->kind = VT_STATIC_METHOD;
			sb->type = p_type;
			sb->method = vm;
			sb->return_type = mi.return_val.type;
			sb->return_kind = returns_void ? AS_KIND_VOID : kind_of_type(mi.return_val.type);
			sb->param_kinds = kinds;
			sb->param_types = types;
			sb->arg_count = mi.arguments.size();
			add_global_function(p_engine, gdecl, sb);
			continue;
		}
		bool is_const = Variant::is_builtin_method_const(p_type, mn);
		String decl = ret + " " + String(mn) + "(" + args + ")" + (is_const ? " const" : "");
		if (ASNativeValueOps::try_add_method(p_engine, p_name, p_type, mn, decl)) {
			continue; // 已原生注册，跳过 generic 跳板。
		}
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_METHOD;
		b->type = p_type;
		b->method = vm;
		b->return_type = mi.return_val.type;
		b->return_kind = returns_void ? AS_KIND_VOID : kind_of_type(mi.return_val.type);
		b->param_kinds = kinds;
		b->param_types = types;
		b->arg_count = mi.arguments.size();
		add_method(p_engine, cname, decl, b);
	}

	// 内省成员 -> AS 虚属性（get_X/set_X + property 属性）。
	for (const StringName &mn : members) {
		Variant::Type mt = Variant::get_member_type(p_type, mn);
		String mt_name = param_type_name(mt);
		if (mt_name.is_empty()) {
			continue;
		}
		Variant::ValidatedGetter getter = Variant::get_member_validated_getter(p_type, mn);
		Variant::ValidatedSetter setter = Variant::get_member_validated_setter(p_type, mn);
		if (getter) {
			ASValueBinding *b = memnew(ASValueBinding);
			b->kind = VT_MEMBER_GET;
			b->type = p_type;
			b->member_get = getter;
			b->return_type = mt;
			b->return_kind = kind_of_type(mt);
			add_method(p_engine, cname, mt_name + " get_" + String(mn) + "() const property", b);
		}
		if (setter) {
			ASValueBinding *b = memnew(ASValueBinding);
			b->kind = VT_MEMBER_SET;
			b->type = p_type;
			b->member_set = setter;
			b->param_kinds.push_back(kind_of_type(mt));
			b->param_types.push_back(mt);
			add_method(p_engine, cname, "void set_" + String(mn) + "(" + as_binding_render_param(mt_name) + ") property", b);
		}
	}

	// 索引/键值访问：AS 2.38 没有 opIndexAssign，读用 opIndex 方法会屏蔽掉 property 访问器，
	// 从而使 `a[i] = v` 只作用在 opIndex 的临时返回值上（静默丢弃）。因此索引访问统一注册为
	// get_opIndex/set_opIndex 属性访问器（propertyAccessorMode == 3 要求 property 标记），
	// 读写路径都经 FindPropertyAccessor 解析（as_compiler.cpp:14527-14589）。
	// Dictionary 同时满足 has_indexing 与 is_keyed：两者都注册会造成
	// "Found multiple get accessors for property 'opIndex'"。键值访问优先，
	// 此时只注册 Variant 键版本（Godot 侧 Dictionary 的索引本就走 Variant）。
	const bool keyed = Variant::is_keyed(p_type);
	if (Variant::has_indexing(p_type) && !keyed) {
		Variant::Type et = Variant::get_indexed_element_type(p_type);
		String et_name = param_type_name(et);
		if (!et_name.is_empty()) {
			String get_decl = et_name + " get_opIndex(int64 index) const property";
			String set_decl = "void set_opIndex(int64 index, " + as_binding_render_param(et_name) + ") property";
			// P2：Packed 标量数组的索引读写走原生 thunk，跳过 generic 跳板；
			// 未命中（含 Variant/对象/字符串元素）返回 false，保持既有 generic 注册。
			if (!ASNativeValueOps::try_add_index_accessor(p_engine, p_name, p_type, get_decl, set_decl)) {
				ASValueBinding *g = memnew(ASValueBinding);
				g->kind = VT_INDEX_GET;
				g->type = p_type;
				g->return_type = et;
				g->return_kind = kind_of_type(et);
				add_method(p_engine, cname, get_decl, g);

				ASValueBinding *s = memnew(ASValueBinding);
				s->kind = VT_INDEX_SET;
				s->type = p_type;
				s->param_kinds.push_back(kind_of_type(et));
				s->param_types.push_back(et);
				add_method(p_engine, cname, set_decl, s);
			}
		}
	}
	if (Variant::is_keyed(p_type)) {
		const String keyed_get_decl = "Variant get_opIndex(const Variant &in key) const property";
		const String keyed_set_decl = "void set_opIndex(const Variant &in key, const Variant &in value) property";
		// Dictionary 的 keyed accessor 与普通 Array accessor 分开注册，避免重复 opIndex。
		if (!ASNativeValueOps::try_add_keyed_accessor(p_engine, p_name, p_type, keyed_get_decl, keyed_set_decl)) {
			ASValueBinding *g = memnew(ASValueBinding);
			g->kind = VT_KEYED_GET;
			g->type = p_type;
			g->return_kind = AS_KIND_VALUE;
			add_method(p_engine, cname, keyed_get_decl, g);

			ASValueBinding *s = memnew(ASValueBinding);
			s->kind = VT_KEYED_SET;
			s->type = p_type;
			s->param_kinds.push_back(AS_KIND_VALUE);
			s->param_types.push_back(Variant::NIL);
			add_method(p_engine, cname, keyed_set_decl, s);
		}
	}

	// 运算符：仅注册 (T, T) 形式的二元算术与相等比较。
	static const Variant::Operator BIN_OPS[] = { Variant::OP_ADD, Variant::OP_SUBTRACT, Variant::OP_MULTIPLY, Variant::OP_DIVIDE };
	// AngelScript 的二元运算符方法名：加法 opAdd，减法 opSub，乘法 opMul，除法 opDiv
	// （见 thirdparty/angelscript/source/as_compiler.cpp 的运算符名映射）。
	static const char *BIN_OP_NAMES[] = { "opAdd", "opSub", "opMul", "opDiv" };
	for (int i = 0; i < 4; i++) {
		Variant::Type rt = Variant::get_operator_return_type(BIN_OPS[i], p_type, p_type);
		if (rt == Variant::NIL) {
			continue;
		}
		String rt_name = param_type_name(rt);
		if (rt_name.is_empty()) {
			continue;
		}
		Variant::ValidatedOperatorEvaluator ev = Variant::get_validated_operator_evaluator(BIN_OPS[i], p_type, p_type);
		if (!ev) {
			continue;
		}
		String decl = rt_name + " " + BIN_OP_NAMES[i] + "(const " + p_name + " &in other) const";
		if (ASNativeValueOps::try_add_op(p_engine, p_name, p_type, BIN_OP_NAMES[i], decl)) {
			continue; // 已原生注册，跳过 generic 跳板。
		}
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_OP;
		b->type = p_type;
		b->op_eval = ev;
		b->return_type = rt;
		b->return_kind = kind_of_type(rt);
		b->param_kinds.push_back(kind_of_type(p_type));
		b->param_types.push_back(p_type);
		b->arg_count = 1;
		add_method(p_engine, cname, decl, b);
	}
	{
		Variant::Type rt = Variant::get_operator_return_type(Variant::OP_EQUAL, p_type, p_type);
		Variant::ValidatedOperatorEvaluator ev = Variant::get_validated_operator_evaluator(Variant::OP_EQUAL, p_type, p_type);
		if (rt == Variant::BOOL && ev) {
			String decl = "bool opEquals(const " + p_name + " &in other) const";
			if (!ASNativeValueOps::try_add_op(p_engine, p_name, p_type, "opEquals", decl)) {
				ASValueBinding *b = memnew(ASValueBinding);
				b->kind = VT_OP;
				b->type = p_type;
				b->op_eval = ev;
				b->return_type = Variant::BOOL;
				b->return_kind = AS_KIND_BOOL;
				b->param_kinds.push_back(kind_of_type(p_type));
				b->param_types.push_back(p_type);
				b->arg_count = 1;
				add_method(p_engine, cname, decl, b);
			}
		}
	}
}

// Variant 自身的转换构造与转换运算符：脚本里 Variant 与标量/String 的互转。
static void register_variant_conversions(asIScriptEngine *p_engine) {
	if (ASNativeValueOps::try_add_variant_conversions(p_engine)) {
		return;
	}
	CharString name = String("Variant").utf8();

	struct {
		const char *decl;
		ASBindingKind kind;
	} ctors[] = {
		{ "void f(int64)", AS_KIND_INT64 },
		{ "void f(double)", AS_KIND_DOUBLE },
		{ "void f(bool)", AS_KIND_BOOL },
		{ "void f(const String &in)", AS_KIND_VALUE },
	};
	for (const auto &c : ctors) {
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_CTOR_ARGS;
		b->type = Variant::NIL;
		b->arg_count = 1;
		b->param_kinds.push_back(c.kind);
		add_behaviour(p_engine, name, asBEHAVE_CONSTRUCT, c.decl, b);
	}

	struct {
		const char *decl;
		ASBindingKind kind;
	} convs[] = {
		{ "int64 opImplConv() const", AS_KIND_INT64 },
		{ "double opImplConv() const", AS_KIND_DOUBLE },
		{ "bool opImplConv() const", AS_KIND_BOOL },
		{ "String opImplConv() const", AS_KIND_VALUE },
	};
	for (const auto &c : convs) {
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_CONV;
		b->type = Variant::NIL;
		b->return_kind = c.kind;
		add_method(p_engine, name, c.decl, b);
	}
}

// string 类型：工厂 + 与 Godot String 的双向构造 + length()。
// 类型骨架（asOBJ_VALUE，sizeof(void*)）已在第一阶段注册。
static void register_string_support(asIScriptEngine *p_engine) {
	CharString str_name = String("string").utf8();

	{
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_STRING_DEFAULT;
		add_behaviour(p_engine, str_name, asBEHAVE_CONSTRUCT, "void f()", b);
	}
	{
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_STRING_COPY;
		add_behaviour(p_engine, str_name, asBEHAVE_CONSTRUCT, "void f(const string &in)", b);
	}
	{
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_STRING_FROM_GODOT;
		add_behaviour(p_engine, str_name, asBEHAVE_CONSTRUCT, "void f(const String &in)", b);
	}
	{
		// AS 对值类型的 `string t = s;` 需要 opAssign，仅靠拷贝构造不生成。
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_STRING_ASSIGN;
		add_method(p_engine, str_name, "string &opAssign(const string &in)", b);
	}
	{
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_NOOP;
		add_behaviour(p_engine, str_name, asBEHAVE_DESTRUCT, "void f()", b);
	}
	{
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_STRING_LENGTH;
		b->return_kind = AS_KIND_INT64;
		add_method(p_engine, str_name, "int64 length() const", b);
	}
	// 类型已可解析，绑定工厂：字符串字面量由它创建。
	p_engine->RegisterStringFactory("string", &g_string_factory);

	if (!ASNativeValueOps::try_add_literal_support(p_engine)) {
		for (const char *name : { "String", "Variant" }) {
			ASValueBinding *b = memnew(ASValueBinding);
			b->kind = VT_CTOR_FROM_STRING;
			b->type = Variant::STRING;
			add_behaviour(p_engine, String(name).utf8(), asBEHAVE_CONSTRUCT, "void f(const string &in)", b);
		}
		ASValueBinding *b = memnew(ASValueBinding);
		b->kind = VT_ADD_STRING;
		b->type = Variant::STRING;
		add_method(p_engine, String("String").utf8(), "String opAdd(const string &in) const", b);
	}
}

// 值类型注册是「每引擎一次」，因此注册状态按引擎指针记录。
// 早前的单指针写法在多引擎场景（单元测试、引擎销毁后重建）下会让第二个引擎
// 拿不到值类型骨架，之后任何引用 String/Vector2 等的 Register* 都会失败，
// 而 AS 一旦有 Register* 失败就会永久置 configFailed。
static HashSet<asIScriptEngine *> g_registered_engines;
// 同上：Variant 对象转换构造器每个引擎只需注册一次。
static HashSet<asIScriptEngine *> g_object_conv_engines;

// 注册依赖对象类型的 Variant 转换构造器（`Variant(Object @)`）。
// 必须在全部对象骨架注册完成之后调用：值类型阶段解析不到 `Object` 类型名，
// 一旦 Register* 失败引擎会被永久标记为 configFailed 且不可恢复。
Error ASBindingValueTypes::register_object_conversions(asIScriptEngine *p_engine) {
	if (!p_engine) {
		return ERR_INVALID_PARAMETER;
	}
	if (g_object_conv_engines.has(p_engine)) {
		return OK; // 幂等：同一引擎只注册一次。
	}
	if (p_engine->GetTypeInfoByName("Object") == nullptr) {
		return OK; // 对象骨架尚未注册，交给后续调用点。
	}
	// 对象句柄作为**非拥有**句柄传入：Variant 只持引用，不接管生命周期。
	// 跳板走 VT_CTOR_ARGS 的 Variant 自身分支（`*self = *args[0]`）。
	ASValueBinding *b = memnew(ASValueBinding);
	b->kind = VT_CTOR_ARGS;
	b->type = Variant::NIL;
	b->arg_count = 1;
	b->param_kinds.push_back(AS_KIND_OBJECT_NONOWNING);
	add_behaviour(p_engine, String("Variant").utf8(), asBEHAVE_CONSTRUCT, "void f(Object @)", b);

	g_object_conv_engines.insert(p_engine);
	return OK;
}

Error ASBindingValueTypes::register_all(asIScriptEngine *p_engine) {
	if (!p_engine) {
		return ERR_INVALID_PARAMETER;
	}
	if (g_registered_engines.has(p_engine)) {
		return OK; // 同一引擎只注册一次。
	}
	g_registered_engines.insert(p_engine);

	static const Variant::Type TYPES[] = {
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
	// 第一阶段：先把全部值类型骨架注册好（含 Variant 与 AS 的 string）。
	for (int i = 0; i < 34; i++) {
		if (ASNativeValueStorage::register_skeleton(p_engine, TYPES[i])) {
			continue; // 4 个向量类型已按原生布局注册。
		}
		register_type_skeleton(p_engine, ASBindingDecl::variant_type_to_as(TYPES[i]), sizeof(Variant), true);
	}
	// Variant 开 asOBJ_APP_CLASS：使其可按值原生返回，从而 Array 的 Variant 元素索引可走
	// 原生 thunk（thunk 就地构造返回槽，AS 对脚本侧拷贝仍调用已注册的 generic 拷贝构造，
	// 不做位拷贝）。ABI 前提同几何值类型：仅 AS_NATIVE_VALUE_OPS_ENABLED 平台启用。
	register_type_skeleton(p_engine, "Variant", sizeof(Variant), true);
	register_type_skeleton(p_engine, "string", sizeof(void *), false);

	// 第二阶段：骨架齐备后做内省注册，签名里的类型引用才都能解析。
	for (int i = 0; i < 34; i++) {
		register_type_members(p_engine, ASBindingDecl::variant_type_to_as(TYPES[i]), TYPES[i]);
	}
	register_type_members(p_engine, "Variant", Variant::NIL);
	register_variant_conversions(p_engine);
	register_string_support(p_engine);
	return OK;
}

// `const string &in` 的实参在 AS 侧有两种形态（见 as_compiler.cpp 的 isRefSafe 处理）：
//  - 字面量：asBC_PGA 直接把工厂指针作为实参压入，不能再解引用；
//  - 变量：压入的是保存工厂指针的存储地址，必须解一次引用。
// 用驻留指针集合区分二者，得到统一的工厂指针。
static const ASGodotStringObject *string_argument(const void *arg) {
	if (!arg) {
		return nullptr;
	}
	const ASGodotStringObject *o = (const ASGodotStringObject *)arg;
	if (g_interned_string_ptrs.has(o)) {
		return o; // 字面量。
	}
	return *(const ASGodotStringObject *const *)arg; // 变量：存储地址解一次。
}

static const ASGodotStringObject *string_slot(asIScriptGeneric *p_gen, int p_index) {
	return string_argument(p_gen->GetArgObject(p_index));
}

const String &ASBindingValueTypes::string_from_argument(const void *p_arg) {
	const ASGodotStringObject *o = string_argument(p_arg);
	static const String empty;
	return o ? o->str : empty;
}

String ASBindingValueTypes::string_from_slot(const void *p_slot) {
	if (p_slot == nullptr) {
		return String();
	}
	const ASGodotStringObject *o = *(const ASGodotStringObject *const *)p_slot;
	return o != nullptr ? o->str : String();
}

void ASBindingValueTypes::generic_value_call(asIScriptGeneric *p_gen) {
	ASValueBinding *b = static_cast<ASValueBinding *>(p_gen->GetAuxiliary());
	if (!b) {
		set_exception("angelscript: missing value type binding");
		return;
	}
	void *self_slot = p_gen->GetObject();
	const bool self_native = ASNativeValueStorage::is_native_storage_type(b->type) && self_slot != nullptr;
	Variant self_tmp;
	Variant *self = (Variant *)self_slot;
	if (self_native) {
		self_tmp = ASNativeValueStorage::native_to_variant(b->type, self_slot);
		self = &self_tmp;
	}
	Variant ret;

	switch (b->kind) {
		case VT_CTOR_DEFAULT: {
			if (!self_slot) {
				set_exception("angelscript: null self in value constructor");
				return;
			}
			// 原生 lifecycle 注册成功时，本 generic 构造不会被注册（见 register_type_members），
			// 故 self_native 仅作防御性兜底：若出现「骨架成功但 lifecycle 未接管」的中间态，
			// 仍按原生槽正确构造，绝不退化成按 Variant 写。
			if (self_native) {
				Callable::CallError err;
				const Variant *noargs[1] = { nullptr };
				Variant v;
				Variant::construct(b->type, v, noargs, 0, err);
				ASNativeValueStorage::variant_to_native(b->type, v, self_slot);
				return;
			}
			new (self) Variant();
			if (b->type != Variant::NIL) {
				Callable::CallError err;
				const Variant *noargs[1] = { nullptr };
				Variant::construct(b->type, *self, noargs, 0, err);
			}
			return;
		}
		case VT_CTOR_COPY: {
			if (self_native) { // 防御性兜底，理由同 VT_CTOR_DEFAULT。
				Variant v = ASNativeValueStorage::native_to_variant(b->type, p_gen->GetArgObject(0));
				ASNativeValueStorage::variant_to_native(b->type, v, self_slot);
				return;
			}
			new (self) Variant(*(const Variant *)p_gen->GetArgObject(0));
			return;
		}
		case VT_CTOR_ARGS: {
			Variant vals[8];
			const Variant *args[8];
			int n = b->arg_count < 8 ? b->arg_count : 8;
			for (int i = 0; i < n; i++) {
				Variant::Type pt = i < b->param_types.size() ? b->param_types[i] : Variant::NIL;
				vals[i] = as_binding_marshal_arg(p_gen, i, b->param_kinds[i], pt);
				args[i] = &vals[i];
			}
			if (self_native) {
				Variant constructed;
				Callable::CallError err;
				Variant::construct(b->type, constructed, args, n, err);
				ASNativeValueStorage::variant_to_native(b->type, constructed, self_slot);
				return;
			}
			new (self) Variant();
			if (b->type == Variant::NIL) {
				if (n > 0) {
					*self = *args[0]; // Variant 自身：拷贝构造 / 隐式转换。
				}
			} else {
				Callable::CallError err;
				Variant::construct(b->type, *self, args, n, err);
			}
			return;
		}
		case VT_DTOR: {
			if (self_native) {
				return; // 原生存储为平凡类型，无需析构。
			}
			if (self) {
				self->~Variant();
			}
			return;
		}
		case VT_ASSIGN: {
			if (self_native) { // 防御性兜底，理由同 VT_CTOR_DEFAULT。
				Variant v = ASNativeValueStorage::native_to_variant(b->type, p_gen->GetArgObject(0));
				ASNativeValueStorage::variant_to_native(b->type, v, self_slot);
				p_gen->SetReturnAddress(self_slot);
				return;
			}
			*self = *(const Variant *)p_gen->GetArgObject(0);
			p_gen->SetReturnAddress(self); // 声明为 T& f(const T&)。
			return;
		}
		case VT_METHOD: {
			if (!b->method || !self) {
				set_exception("angelscript: invalid builtin method binding");
				return;
			}
			Variant vals[8];
			const Variant *args[8];
			int n = b->arg_count;
			for (int i = 0; i < n && i < 8; i++) {
				Variant::Type pt = i < b->param_types.size() ? b->param_types[i] : Variant::NIL;
				vals[i] = as_binding_marshal_arg(p_gen, i, b->param_kinds[i], pt);
				args[i] = &vals[i];
			}
			VariantInternal::initialize(&ret, b->return_type);
			b->method(self, args, n, &ret);
			as_binding_marshal_return(p_gen, b->return_kind, ret, b->return_type);
			return;
		}
		case VT_STATIC_METHOD: {
			if (!b->method) {
				set_exception("angelscript: invalid static builtin method binding");
				return;
			}
			Variant vals[8];
			const Variant *args[8];
			int n = b->arg_count;
			for (int i = 0; i < n && i < 8; i++) {
				Variant::Type pt = i < b->param_types.size() ? b->param_types[i] : Variant::NIL;
				vals[i] = as_binding_marshal_arg(p_gen, i, b->param_kinds[i], pt);
				args[i] = &vals[i];
			}
			VariantInternal::initialize(&ret, b->return_type);
			b->method(nullptr, args, n, &ret);
			as_binding_marshal_return(p_gen, b->return_kind, ret, b->return_type);
			return;
		}
		case VT_MEMBER_GET: {
			if (!b->member_get || !self) {
				set_exception("angelscript: invalid member getter binding");
				return;
			}
			VariantInternal::initialize(&ret, b->return_type);
			b->member_get(self, &ret);
			as_binding_marshal_return(p_gen, b->return_kind, ret, b->return_type);
			return;
		}
		case VT_MEMBER_SET: {
			if (!b->member_set || !self) {
				set_exception("angelscript: invalid member setter binding");
				return;
			}
			Variant::Type pt = b->param_types.is_empty() ? Variant::NIL : b->param_types[0];
			Variant v = as_binding_marshal_arg(p_gen, 0, b->param_kinds[0], pt);
			b->member_set(self, &v);
			if (self_native) {
				ASNativeValueStorage::variant_to_native(b->type, *self, self_slot);
			}
			return;
		}
		case VT_INDEX_GET: {
			bool valid = false;
			bool oob = false;
			ret = self->get_indexed((int64_t)p_gen->GetArgQWord(0), valid, oob);
			as_binding_marshal_return(p_gen, b->return_kind, ret, b->return_type);
			return;
		}
		case VT_INDEX_SET: {
			// set_opIndex 的形参顺序为 (index, value)。
			Variant::Type pt = b->param_types.is_empty() ? Variant::NIL : b->param_types[0];
			Variant v = as_binding_marshal_arg(p_gen, 1, b->param_kinds[0], pt);
			bool valid = false;
			bool oob = false;
			self->set_indexed((int64_t)p_gen->GetArgQWord(0), v, valid, oob);
			if (self_native) {
				ASNativeValueStorage::variant_to_native(b->type, *self, self_slot);
			}
			return;
		}
		case VT_STRING_ASSIGN: {
			if (!self) {
				set_exception("angelscript: null self in string assign");
				return;
			}
			*(const void **)self = string_slot(p_gen, 0);
			p_gen->SetReturnAddress(self);
			return;
		}
		case VT_KEYED_GET: {
			Variant key = as_binding_marshal_arg(p_gen, 0, AS_KIND_VALUE);
			bool valid = false;
			ret = self->get_keyed(key, valid);
			as_binding_marshal_return(p_gen, b->return_kind, ret, b->return_type);
			return;
		}
		case VT_KEYED_SET: {
			// set_opIndex 的形参顺序为 (key, value)。
			Variant key = as_binding_marshal_arg(p_gen, 0, AS_KIND_VALUE);
			Variant v = as_binding_marshal_arg(p_gen, 1, AS_KIND_VALUE);
			bool valid = false;
			self->set_keyed(key, v, valid);
			if (self_native) {
				ASNativeValueStorage::variant_to_native(b->type, *self, self_slot);
			}
			return;
		}
		case VT_OP: {
			if (!b->op_eval || !self) {
				set_exception("angelscript: invalid operator binding");
				return;
			}
			Variant right;
			const Variant *right_ptr = nullptr;
			if (b->arg_count > 0) {
				Variant::Type pt = b->param_types.is_empty() ? Variant::NIL : b->param_types[0];
				right = as_binding_marshal_arg(p_gen, 0, b->param_kinds[0], pt);
				right_ptr = &right;
			}
			VariantInternal::initialize(&ret, b->return_type);
			b->op_eval(self, right_ptr, &ret);
			as_binding_marshal_return(p_gen, b->return_kind, ret, b->return_type);
			return;
		}
		case VT_CONV: {
			switch (b->return_kind) {
				case AS_KIND_INT64:
					*(int64_t *)p_gen->GetAddressOfReturnLocation() = (int64_t)*self;
					return;
				case AS_KIND_DOUBLE:
					*(double *)p_gen->GetAddressOfReturnLocation() = (double)*self;
					return;
				case AS_KIND_BOOL:
					*(bool *)p_gen->GetAddressOfReturnLocation() = (bool)*self;
					return;
				case AS_KIND_VALUE: {
					// VT_CONV 只由 register_variant_conversions 生成，返回类型恒为 String（非原生
					// 存储），因此无需原生返回分支；String 的存储本身就是一颗 Variant。
					Variant s = Variant((String)*self);
					p_gen->SetReturnObject(&s);
					return;
				}
				default:
					return;
			}
		}
		case VT_NOOP:
			return;
		case VT_CTOR_FROM_STRING: {
			new (self) Variant(string_from_argument(p_gen->GetArgObject(0)));
			return;
		}
		case VT_ADD_STRING: {
			new (p_gen->GetAddressOfReturnLocation()) Variant(VariantInternalAccessor<String>::get(self) + string_from_argument(p_gen->GetArgObject(0)));
			return;
		}
		case VT_STRING_DEFAULT: {
			*(const void **)self = intern_string(String());
			return;
		}
		case VT_STRING_COPY: {
			*(const void **)self = string_slot(p_gen, 0);
			return;
		}
		case VT_STRING_FROM_GODOT: {
			Variant v = as_binding_marshal_arg(p_gen, 0, AS_KIND_VALUE);
			*(const void **)self = intern_string((String)v);
			return;
		}
		case VT_STRING_LENGTH: {
			const ASGodotStringObject *o = self ? (const ASGodotStringObject *)*(const void **)self : nullptr;
			*(int64_t *)p_gen->GetAddressOfReturnLocation() = o ? (int64_t)o->str.length() : 0;
			return;
		}
	}
}
