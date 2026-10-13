/**************************************************************************/
/*  as_binding_object.cpp                                                 */
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

#include "as_binding_object.h"

#include "as_binding_decl.h"
#include "as_binding_native_value_ops.h"
#include "core/object/class_db.h"
#include "core/object/ref_counted.h"
#include "core/os/memory.h"
#include "core/string/string_name.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/variant/callable.h"
#include "scene/main/node.h"

// 非拥有句柄的槽里放的是 64 位 ObjectID（validator + slot + ref 位），必须完整放进 sizeof(void*) 的槽。
// 32 位平台的 uintptr_t 会截断高位，解码时可能命中另一个存活对象（对象错认），而不是安全降级为 null。
// 本里程碑只承诺 64 位平台（五平台均为 64 位构建）：让 32 位构建在编译期大声失败（spec §3.3）。
static_assert(sizeof(void *) >= 8, "AngelScript 非拥有对象句柄要求 64 位平台：ObjectID 为 64 位，32 位指针槽会截断。");

namespace {

// 跳板运行时要靠“成员是谁”来转发调用，这些记录以泄漏式堆对象挂在
// asIScriptFunction::userdata 上：注册只在启动时发生一次，生命周期与引擎一致，不做回收。
struct ASObjectBinding {
	enum Kind {
		KIND_METHOD,
		KIND_PROPERTY_GET,
		KIND_PROPERTY_SET,
	};
	Kind kind = KIND_METHOD;
	StringName member;
	const MethodBind *bind = nullptr;
	Vector<ASBindingKind> param_kinds;
	ASBindingKind return_kind = AS_KIND_VOID;
	// 与 param_kinds 平行：实参/返回的 Variant::Type（标量为 NIL）。
	// 原生存储类型（Vector2/3/2i/3i）的槽是原生 T，编组必须带类型走布局感知路径。
	Vector<Variant::Type> param_types;
	Variant::Type return_type = Variant::NIL;
	// self 槽的语义：由所属类是否派生自 RefCounted 决定（spec §0 R3）。
	ASBindingKind object_kind = AS_KIND_OBJECT_NONOWNING;
};

struct ASObjectFactoryBinding {
	StringName class_name;
	ASBindingKind object_kind = AS_KIND_OBJECT_NONOWNING;
};

// 继承边的元数据：向上转换必须知道源槽与目标槽各自的语义，才能决定是否重新编码。
struct ASUpcastBinding {
	ASBindingKind source_kind = AS_KIND_OBJECT_NONOWNING;
	ASBindingKind target_kind = AS_KIND_OBJECT_NONOWNING;
};

// RegisterGlobalProperty 要求存储地址一直有效到引擎销毁，所以用 memnew 泄漏，
// 不能把 HashMap 里的值取址（rehash 会让地址失效）。
// 注册守卫必须按引擎区分：同一进程内可能同时存在多个 AS 引擎
// （单元测试、引擎销毁后重建）。用进程级名字集合会让第二个引擎误判「已注册」
// 而静默跳过，导致引用该名字的 Register* 全部失败并置 configFailed。
HashMap<asIScriptEngine *, HashSet<StringName>> g_global_names;
// register_class 幂等：重复挂同一个类的成员会以 asNAME_TAKEN 污染引擎。
HashMap<asIScriptEngine *, HashSet<StringName>> g_registered_classes;

void set_exception(const String &p_message) {
	asIScriptContext *ctx = asGetActiveContext();
	if (ctx) {
		CharString msg = p_message.utf8();
		ctx->SetException(msg.get_data());
	}
}

void attach_binding(asIScriptEngine *p_engine, int p_id, void *p_userdata) {
	if (p_id < 0 || p_userdata == nullptr) {
		return;
	}
	asIScriptFunction *f = p_engine->GetFunctionById(p_id);
	if (f) {
		f->SetUserData(p_userdata);
	}
}

ASObjectBinding *get_binding(asIScriptGeneric *p_gen, ASObjectBinding::Kind p_expected) {
	ASObjectBinding *b = static_cast<ASObjectBinding *>(p_gen->GetAuxiliary());
	if (b == nullptr || b->kind != p_expected) {
		set_exception("AngelScript: binding metadata missing");
		return nullptr;
	}
	return b;
}

// 静态方法以全局函数注册，名字 `<类名>_<成员名>` 可能与工具函数等既有全局符号重名。
// 任何 Register* 失败都会永久污染引擎，故注册前先扫描已存在的全局函数名。
bool global_function_name_taken(asIScriptEngine *p_engine, const String &p_name) {
	const asUINT count = p_engine->GetGlobalFunctionCount();
	for (asUINT i = 0; i < count; i++) {
		const asIScriptFunction *fn = p_engine->GetGlobalFunctionByIndex(i);
		if (fn != nullptr && String(fn->GetName()) == p_name) {
			return true;
		}
	}
	return false;
}

} // namespace

Object *as_handle_decode(void *p_slot, ASBindingKind p_kind) {
	if (p_kind == AS_KIND_OBJECT_OWNING) {
		// 拥有句柄：槽里就是裸指针，且 AS 的 addref/release 保证它在本句柄存活期间有效。
		return (Object *)p_slot;
	}
	// 非拥有句柄：槽里是 ObjectID（0 为 null 哨兵）。
	// 对象可能已被 free()，所以每次都要经 ObjectDB 查表；查不到即为失效句柄。
	const ObjectID id((uint64_t)(uintptr_t)p_slot);
	if (!id.is_valid()) {
		return nullptr;
	}
	return ObjectDB::get_instance(id);
}

void *as_handle_encode(Object *p_obj, ASBindingKind p_kind) {
	if (p_obj == nullptr) {
		return nullptr;
	}
	if (p_kind == AS_KIND_OBJECT_OWNING) {
		return (void *)p_obj;
	}
	// 非拥有：只交出 ID，绝不交出裸指针 —— 否则 AS 侧会绕过 ObjectDB 直接解引用已释放对象。
	return (void *)(uintptr_t)(uint64_t)p_obj->get_instance_id();
}

#if AS_NATIVE_VALUE_OPS_ENABLED
namespace {

// P1：引擎成员热点原生 thunk。全量不划算（~12.8 万成员，且原生约定无 per-call userdata），
// 只对确认的热点按「(类, 方法) + self 槽语义」白名单注册专用 thunk；未命中一律回退 generic。
// 新增热点 = 在此加一个 thunk + 一条白名单分支。
int64_t hot_node_get_child_count(const void *p_self, bool p_include_internal) {
	ASNativeValueOps::note_thunk_call();
	// Node 非 RefCounted → self 槽是非拥有句柄（ObjectID），必须经 ObjectDB 解码。
	Object *obj = as_handle_decode(const_cast<void *>(p_self), AS_KIND_OBJECT_NONOWNING);
	if (obj == nullptr) {
		set_exception("AngelScript: get_child_count: object is null or has been freed");
		return 0;
	}
	return (int64_t)static_cast<Node *>(obj)->get_child_count(p_include_internal);
}

// 返回 true 表示已用原生 thunk 注册（调用方跳过 generic）。
bool try_register_hot_method(asIScriptEngine *p_engine, const char *p_class_cstr, const StringName &p_class, ASBindingKind p_kind, const StringName &p_method, const String &p_decl) {
	if (p_class == StringName("Node") && p_kind == AS_KIND_OBJECT_NONOWNING && p_method == StringName("get_child_count")) {
		const CharString decl = p_decl.utf8();
		const int r = p_engine->RegisterObjectMethod(p_class_cstr, decl.get_data(), asFUNCTION(hot_node_get_child_count), asCALL_CDECL_OBJFIRST);
		if (r < 0) {
			ERR_PRINT(vformat("AngelScript: failed to register native hot method Node::get_child_count (%d)", r));
			return false;
		}
		return true;
	}
	return false;
}

} // namespace
#endif // AS_NATIVE_VALUE_OPS_ENABLED

Error ASBindingObject::register_skeleton(const ASBindingClass &p_class, asIScriptEngine *p_engine) {
	ERR_FAIL_COND_V(p_engine == nullptr, ERR_INVALID_PARAMETER);

	const CharString cname = String(p_class.name).utf8();
	// 值类型（T3 注册的 Vector2/Array/...）优先占名；同名说明 plan 与值类型表冲突，跳过即可，
	// 不能重复注册：RegisterObjectType 失败会给引擎置上 configFailed。
	if (p_engine->GetTypeInfoByName(cname.get_data()) != nullptr) {
		return OK;
	}

	if (p_engine->RegisterObjectType(cname.get_data(), 0, asOBJ_REF) < 0) {
		ERR_PRINT(vformat("AngelScript: RegisterObjectType('%s') failed", p_class.name));
		return ERR_CANT_CREATE;
	}

	// RefCounted 类：句柄槽里是裸指针，直接按 RefCounted 补/放引用。
	// 非 RefCounted 类（Node 等）：addref/release 为空操作——脚本持有的 `Object@`/`Node@`
	// 是弱引用（M3 语义），不保活对象、无跨边界引用环；句柄槽存 ObjectID，对象 free() 后
	// 解引用经 as_handle_decode 反查为 null 并抛异常，绝不触碰已释放内存。
	// 需要保活请声明具体 RefCounted 类型（如 `Resource@`）。
	const bool refcounted = ClassDB::is_parent_class(p_class.name, "RefCounted");
	asSFuncPtr addref_fn = refcounted ? asFUNCTION(generic_addref) : asFUNCTION(generic_addref_noop);
	asSFuncPtr release_fn = refcounted ? asFUNCTION(generic_release) : asFUNCTION(generic_release_noop);
	p_engine->RegisterObjectBehaviour(cname.get_data(), asBEHAVE_ADDREF, "void f()", addref_fn, asCALL_GENERIC);
	p_engine->RegisterObjectBehaviour(cname.get_data(), asBEHAVE_RELEASE, "void f()", release_fn, asCALL_GENERIC);

	if (p_class.instantiable) {
		ASObjectFactoryBinding *fb = memnew(ASObjectFactoryBinding);
		fb->class_name = p_class.name;
		fb->object_kind = refcounted ? AS_KIND_OBJECT_OWNING : AS_KIND_OBJECT_NONOWNING;
		const String decl = String(p_class.name) + "@ f()";
		const CharString cdecl_utf8 = decl.utf8();
		attach_binding(p_engine, p_engine->RegisterObjectBehaviour(cname.get_data(), asBEHAVE_FACTORY, cdecl_utf8.get_data(), asFUNCTION(generic_instantiate), asCALL_GENERIC, fb), fb);
	}

	return OK;
}

Error ASBindingObject::register_class(const ASBindingClass &p_class, asIScriptEngine *p_engine) {
	ERR_FAIL_COND_V(p_engine == nullptr, ERR_INVALID_PARAMETER);

	const CharString cname = String(p_class.name).utf8();
	asITypeInfo *ti = p_engine->GetTypeInfoByName(cname.get_data());
	if (ti == nullptr) {
		// 骨架没建起来（不可见或与值类型同名），成员自然也无从挂载。
		return ERR_DOES_NOT_EXIST;
	}
	HashSet<StringName> &registered_classes = g_registered_classes[p_engine];
	if (registered_classes.has(p_class.name)) {
		return OK;
	}
	registered_classes.insert(p_class.name);

	// self 槽语义与 register_skeleton 的 addref/release 选择必须一致：同一判据、同一处推导。
	const ASBindingKind class_kind = ClassDB::is_parent_class(p_class.name, "RefCounted")
			? AS_KIND_OBJECT_OWNING
			: AS_KIND_OBJECT_NONOWNING;

	for (const ASBindingMethod &m : p_class.methods) {
		if (m.bind == nullptr) {
			continue;
		}
		if (m.is_static) {
			// AS 2.38 无 `static` 关键字：静态方法注册为**全局函数**，命名沿用
			// <类名>_<成员名> 约定（与常量、枚举一致）。名字冲突时跳过，
			// 否则 Register* 失败会永久污染引擎。
			const String gname = String(p_class.name) + "_" + m.name;
			if (g_global_names[p_engine].has(gname) || global_function_name_taken(p_engine, gname)) {
				continue;
			}
			ASObjectBinding *b = memnew(ASObjectBinding);
			b->kind = ASObjectBinding::KIND_METHOD;
			b->member = m.name;
			b->bind = m.bind;
			b->param_kinds = m.param_kinds;
			b->return_kind = m.return_kind;
			b->param_types = m.param_types;
			b->return_type = m.return_type;
			const CharString decl = m.as_decl.utf8();
			const int id = p_engine->RegisterGlobalFunction(decl.get_data(), asFUNCTION(generic_static_method_call), asCALL_GENERIC, b);
			if (id < 0) {
				memdelete(b);
				continue;
			}
			g_global_names[p_engine].insert(gname);
			attach_binding(p_engine, id, b);
			continue;
		}
#if AS_NATIVE_VALUE_OPS_ENABLED
		if (try_register_hot_method(p_engine, cname.get_data(), p_class.name, class_kind, m.name, m.as_decl)) {
			continue;
		}
#endif
		ASObjectBinding *b = memnew(ASObjectBinding);
		b->kind = ASObjectBinding::KIND_METHOD;
		b->member = m.name;
		b->bind = m.bind;
		b->param_kinds = m.param_kinds;
		b->return_kind = m.return_kind;
		b->param_types = m.param_types;
		b->return_type = m.return_type;
		b->object_kind = class_kind;
		const CharString decl = m.as_decl.utf8();
		attach_binding(p_engine, p_engine->RegisterObjectMethod(cname.get_data(), decl.get_data(), asFUNCTION(generic_method_call), asCALL_GENERIC, b), b);
	}

	for (const ASBindingProperty &p : p_class.properties) {
		// 属性名必须是合法标识符（plan 已过滤，这里防御性再查一次）：
		// "frame_0/texture" 这类名字会拼出非法声明并被 AS 以 asINVALID_DECLARATION 拒绝。
		if (!p.name.is_valid_identifier()) {
			continue;
		}
		const String getter = "get_" + p.name;
		const String setter = "set_" + p.name;
		const CharString cgetter = getter.utf8();
		const CharString csetter = setter.utf8();
		// 与方法（或基类访问器）重名时跳过：asNAME_TAKEN 同样会永久污染引擎。
		if (ti->GetMethodByName(cgetter.get_data()) != nullptr) {
			continue;
		}
		if (!p.read_only && ti->GetMethodByName(csetter.get_data()) != nullptr) {
			continue;
		}

		const String gdecl = p.as_type + " " + getter + "() const property";
		const CharString cgdecl = gdecl.utf8();
		ASObjectBinding *gb = memnew(ASObjectBinding);
		gb->kind = ASObjectBinding::KIND_PROPERTY_GET;
		gb->member = p.name;
		gb->return_kind = p.kind;
		gb->return_type = p.type;
		gb->object_kind = class_kind;
		attach_binding(p_engine, p_engine->RegisterObjectMethod(cname.get_data(), cgdecl.get_data(), asFUNCTION(generic_property_get), asCALL_GENERIC, gb), gb);

		if (p.read_only) {
			continue;
		}
		// setter 的形参必须与 getter 返回类型一致（AS 校验 virtual property 的规则）。
		const String sdecl = "void " + setter + "(" + as_binding_render_param(p.as_type) + ") property";
		const CharString csdecl = sdecl.utf8();
		ASObjectBinding *sb = memnew(ASObjectBinding);
		sb->kind = ASObjectBinding::KIND_PROPERTY_SET;
		sb->member = p.name;
		sb->param_kinds.push_back(p.kind);
		sb->param_types.push_back(p.type);
		sb->object_kind = class_kind;
		attach_binding(p_engine, p_engine->RegisterObjectMethod(cname.get_data(), csdecl.get_data(), asFUNCTION(generic_property_set), asCALL_GENERIC, sb), sb);
	}

	for (const ASBindingConstant &c : p_class.constants) {
		const String gname = String(p_class.name) + "_" + c.name;
		if (g_global_names[p_engine].has(gname)) {
			continue;
		}
		const CharString cgname = gname.utf8();
		if (p_engine->GetGlobalPropertyIndexByName(cgname.get_data()) >= 0) {
			continue;
		}
		int64_t *storage = memnew(int64_t);
		*storage = c.value;
		const String decl = "const int64 " + gname;
		const CharString cdecl_utf8 = decl.utf8();
		if (p_engine->RegisterGlobalProperty(cdecl_utf8.get_data(), storage) < 0) {
			memdelete(storage);
			continue;
		}
		g_global_names[p_engine].insert(gname);
	}

	// 继承边：AS 不做跨级隐式转换，因此对每个已注册祖先各注册一条 opImplCast。
	// 槽值必须按**目标**类型重新编码（Resource@ 的裸指针 → Object@ 的 ObjectID），
	// 所以每条边都要带一份源/目标 kind 的元数据。
	StringName ancestor = p_class.parent;
	while (!ancestor.is_empty()) {
		const CharString ancestor_data = String(ancestor).utf8();
		if (p_engine->GetTypeInfoByName(ancestor_data.get_data()) == nullptr) {
			break;
		}
		ASUpcastBinding *ub = memnew(ASUpcastBinding);
		ub->source_kind = class_kind;
		ub->target_kind = ClassDB::is_parent_class(ancestor, "RefCounted")
				? AS_KIND_OBJECT_OWNING
				: AS_KIND_OBJECT_NONOWNING;
		const String decl = String(ancestor) + "@ opImplCast() const";
		const CharString cdecl_utf8 = decl.utf8();
		attach_binding(p_engine, p_engine->RegisterObjectMethod(cname.get_data(), cdecl_utf8.get_data(), asFUNCTION(generic_upcast), asCALL_GENERIC, ub), ub);
		ancestor = ClassDB::get_parent_class_nocheck(ancestor);
	}

	return OK;
}

Error ASBindingObject::register_enums(const Vector<ASBindingEnum> &p_enums, asIScriptEngine *p_engine) {
	ERR_FAIL_COND_V(p_engine == nullptr, ERR_INVALID_PARAMETER);

	for (const ASBindingEnum &e : p_enums) {
		// ClassDB 只给出裸枚举名（如 "ProcessMode"），不同类会重名；AS 侧一律用
		// "<声明类>_<枚举名>" 限定，同时把枚举名里可能出现的 '.' 净化成 '_'。
		const String ename = (String(e.scope) + "_" + String(e.name)).replace(".", "_");
		if (ename.is_empty() || g_global_names[p_engine].has(ename)) {
			continue;
		}
		const CharString cename = ename.utf8();
		if (p_engine->GetTypeInfoByName(cename.get_data()) != nullptr) {
			continue;
		}

		// RegisterEnumValue 只接受 int32：越界值直接丢弃，不留半截枚举。
		Vector<Pair<String, int64_t>> kept;
		for (const Pair<String, int64_t> &v : e.values) {
			if (v.second >= INT32_MIN && v.second <= INT32_MAX) {
				kept.push_back(v);
			}
		}
		if (kept.is_empty()) {
			continue;
		}

		if (p_engine->RegisterEnum(cename.get_data()) < 0) {
			continue;
		}
		g_global_names[p_engine].insert(ename);
		for (const Pair<String, int64_t> &v : kept) {
			const CharString cval = v.first.utf8();
			p_engine->RegisterEnumValue(cename.get_data(), cval.get_data(), (int)v.second);
		}
	}

	return OK;
}

void ASBindingObject::generic_method_call(asIScriptGeneric *p_gen) {
	ASObjectBinding *b = get_binding(p_gen, ASObjectBinding::KIND_METHOD);
	if (b == nullptr) {
		return;
	}
	Object *self = as_handle_decode(p_gen->GetObject(), b->object_kind);
	if (self == nullptr) {
		// 两种原因合并成一条：槽为 null，或非拥有句柄指向的对象已经被 free()。
		// 带上成员名，使用者才能定位是哪个句柄失效了。
		set_exception(vformat("AngelScript: %s: object is null or has been freed", b->member));
		return;
	}

	const int argc = b->param_kinds.size();
	Vector<Variant> args;
	args.resize(argc);
	for (int i = 0; i < argc; i++) {
		const Variant::Type pt = i < b->param_types.size() ? b->param_types[i] : Variant::NIL;
		args.write[i] = as_binding_marshal_arg(p_gen, i, b->param_kinds[i], pt);
	}
	// MethodBind 要的是“Variant 指针数组”，不能把 Vector<Variant> 强转成指针数组。
	Vector<const Variant *> argptrs;
	argptrs.resize(argc);
	for (int i = 0; i < argc; i++) {
		argptrs.write[i] = &args[i];
	}

	Callable::CallError ce;
	const Variant **argv = nullptr;
	if (argc > 0) {
		argv = argptrs.ptrw();
	}
	Variant ret = b->bind->call(self, argv, argc, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		set_exception(vformat("AngelScript: %s() failed (%d)", b->member, (int)ce.error));
		return;
	}
	as_binding_marshal_return(p_gen, b->return_kind, ret, b->return_type);
}

void ASBindingObject::generic_static_method_call(asIScriptGeneric *p_gen) {
	ASObjectBinding *b = get_binding(p_gen, ASObjectBinding::KIND_METHOD);
	if (b == nullptr) {
		return;
	}
	// 静态方法没有 self：MethodBind 的静态实现忽略实例指针，传 nullptr 即可。

	const int argc = b->param_kinds.size();
	Vector<Variant> args;
	args.resize(argc);
	for (int i = 0; i < argc; i++) {
		const Variant::Type pt = i < b->param_types.size() ? b->param_types[i] : Variant::NIL;
		args.write[i] = as_binding_marshal_arg(p_gen, i, b->param_kinds[i], pt);
	}
	Vector<const Variant *> argptrs;
	argptrs.resize(argc);
	for (int i = 0; i < argc; i++) {
		argptrs.write[i] = &args[i];
	}

	Callable::CallError ce;
	const Variant **argv = nullptr;
	if (argc > 0) {
		argv = argptrs.ptrw();
	}
	Variant ret = b->bind->call(nullptr, argv, argc, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		set_exception(vformat("AngelScript: %s() failed (%d)", b->member, (int)ce.error));
		return;
	}
	as_binding_marshal_return(p_gen, b->return_kind, ret, b->return_type);
}

void ASBindingObject::generic_property_get(asIScriptGeneric *p_gen) {
	ASObjectBinding *b = get_binding(p_gen, ASObjectBinding::KIND_PROPERTY_GET);
	if (b == nullptr) {
		return;
	}
	Object *self = as_handle_decode(p_gen->GetObject(), b->object_kind);
	if (self == nullptr) {
		// 两种原因合并成一条：槽为 null，或非拥有句柄指向的对象已经被 free()。
		// 带上成员名，使用者才能定位是哪个句柄失效了。
		set_exception(vformat("AngelScript: %s: object is null or has been freed", b->member));
		return;
	}
	bool valid = false;
	Variant v = self->get(b->member, &valid);
	if (!valid) {
		set_exception(vformat("AngelScript: property '%s' is not readable", b->member));
		return;
	}
	as_binding_marshal_return(p_gen, b->return_kind, v, b->return_type);
}

void ASBindingObject::generic_property_set(asIScriptGeneric *p_gen) {
	ASObjectBinding *b = get_binding(p_gen, ASObjectBinding::KIND_PROPERTY_SET);
	if (b == nullptr) {
		return;
	}
	Object *self = as_handle_decode(p_gen->GetObject(), b->object_kind);
	if (self == nullptr) {
		// 两种原因合并成一条：槽为 null，或非拥有句柄指向的对象已经被 free()。
		// 带上成员名，使用者才能定位是哪个句柄失效了。
		set_exception(vformat("AngelScript: %s: object is null or has been freed", b->member));
		return;
	}
	Variant v = as_binding_marshal_arg(p_gen, 0, b->param_kinds[0], b->param_types.is_empty() ? Variant::NIL : b->param_types[0]);
	bool valid = false;
	self->set(b->member, v, &valid);
	if (!valid) {
		set_exception(vformat("AngelScript: property '%s' is not writable", b->member));
	}
}

void ASBindingObject::generic_instantiate(asIScriptGeneric *p_gen) {
	ASObjectFactoryBinding *fb = static_cast<ASObjectFactoryBinding *>(p_gen->GetAuxiliary());
	if (fb == nullptr) {
		set_exception("AngelScript: factory metadata missing");
		return;
	}
	Object *o = ClassDB::instantiate(fb->class_name);
	if (o == nullptr) {
		set_exception(vformat("AngelScript: cannot instantiate '%s'", fb->class_name));
		return;
	}
	// instantiate 自带一份 RefCounted 的“创建者引用”，SetReturnObject 的 addref 会通过
	// RefCounted::init_ref 接管它，因此这里既不能额外 reference 也不能 deinit_ref。
	// 非 RefCounted 走 NONOWNING：只交出 ObjectID，不交出裸指针，也不改变任何引用计数。
	p_gen->SetReturnObject(as_handle_encode(o, fb->object_kind));
}

void ASBindingObject::generic_addref(asIScriptGeneric *p_gen) {
	RefCounted *rc = static_cast<RefCounted *>((Object *)p_gen->GetObject());
	if (rc) {
		// 只用 init_ref()，不要写成 `if (!init_ref()) { reference(); }`（spec §7 审计结论）。
		// 本 fork 的 init_ref()：refcount 自增后，若这是“第一份引用”（refcount_init 仍为 1）
		// 则回退一次，净变化 0；其后每次调用 refcount_init 已为 0，净变化 +1。
		// 而 reference() 只在 refcount 已归零时返回 false —— 归零意味着对象已被 memdelete，
		// 此时再调 reference() 是读已释放内存。因此 init_ref() 既是正确语义，也是唯一安全写法。
		rc->init_ref();
	}
}

void ASBindingObject::generic_release(asIScriptGeneric *p_gen) {
	RefCounted *rc = static_cast<RefCounted *>((Object *)p_gen->GetObject());
	if (rc) {
		// unreference() 只把计数减一，并用返回值告诉调用方“是否该销毁”，它自己不销毁。
		// 丢弃返回值会让计数归零的对象永远留在 ObjectDB（M2 的既有缺陷，表现为退出期
		// "ObjectDB instances were leaked"）。对齐 Ref<T>::unref() 的写法。
		if (rc->unreference()) {
			memdelete(rc);
		}
	}
}

void ASBindingObject::generic_addref_noop(asIScriptGeneric *p_gen) {
	// 有意留空：非 RefCounted 句柄是弱引用，不参与引用计数；且槽里是 ObjectID，
	// 对象已 free() 时也绝不能解引用。
	(void)p_gen;
}

void ASBindingObject::generic_release_noop(asIScriptGeneric *p_gen) {
	(void)p_gen;
}

void ASBindingObject::generic_upcast(asIScriptGeneric *p_gen) {
	ASUpcastBinding *b = static_cast<ASUpcastBinding *>(p_gen->GetAuxiliary());
	if (b == nullptr) {
		set_exception("AngelScript: upcast metadata missing");
		return;
	}
	void *slot = p_gen->GetObject();
	if (b->source_kind == AS_KIND_OBJECT_NONOWNING) {
		// 非拥有源槽里已经是 ObjectID。目标只可能也是非拥有 —— 拥有类型不可能是
		// 非拥有类型的祖先（Object/Node 都不派生自 RefCounted）。
		if (b->target_kind != AS_KIND_OBJECT_NONOWNING) {
			set_exception("AngelScript: invalid upcast target kind");
			return;
		}
		p_gen->SetReturnObject(slot);
		return;
	}
	// 拥有源槽里是裸指针：目标可能同样拥有（透传），也可能非拥有（折成 ObjectID）。
	// 借 SetReturnObject 自己的 addref 交付一份引用，正好由接收变量接管。
	p_gen->SetReturnObject(as_handle_encode((Object *)slot, b->target_kind));
}
