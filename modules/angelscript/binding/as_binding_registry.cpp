/**************************************************************************/
/*  as_binding_registry.cpp                                               */
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

#include "as_binding_registry.h"

#include "as_binding_decl.h"
#include "as_binding_object.h"
#include "as_binding_value_types.h"
#include "as_typed_containers.h"

#include "core/object/method_info.h"
#include "core/os/memory.h"
#include "core/string/string_name.h"
#include "core/templates/hash_set.h"
#include "core/templates/list.h"
#include "core/variant/callable.h"

namespace {

// 一个已注册的 @GlobalScope 工具函数的运行期元数据。
struct ASUtilityBinding {
	StringName name;
	Vector<ASBindingKind> param_kinds;
	// 形参/返回值的静态 Variant 类型：编组入口据此判定「原生 T 槽」还是「Variant 盒」。
	// 上游 @GlobalScope 元数据可能退化为 NIL（如经 FUNCBINDVR* 注册的 lerp/clamp），
	// 此时按 Variant 盒编组；一旦上游补全类型，原生化类型会在此被正确识别。
	Vector<Variant::Type> param_types;
	ASBindingKind return_kind = AS_KIND_VOID;
	Variant::Type return_type = Variant::NIL;
};

void set_exception(const String &p_message) {
	asIScriptContext *ctx = asGetActiveContext();
	if (ctx != nullptr) {
		ctx->SetException(p_message.utf8().get_data());
	}
}

// @GlobalScope 工具函数的统一跳板：按 kind 编组实参后交给 Variant::call_utility_function，
// 与对象方法跳板共用 as_binding_marshal_arg / as_binding_marshal_return。
void generic_utility_call(asIScriptGeneric *p_gen) {
	const ASUtilityBinding *binding = static_cast<const ASUtilityBinding *>(p_gen->GetAuxiliary());
	if (binding == nullptr) {
		set_exception("AngelScript: utility function binding metadata missing");
		return;
	}

	const int argc = (int)binding->param_kinds.size();
	Vector<Variant> args;
	args.resize(argc);
	for (int i = 0; i < argc; i++) {
		const Variant::Type stored_type = i < binding->param_types.size() ? binding->param_types[i] : Variant::NIL;
		args.write[i] = as_binding_marshal_arg(p_gen, i, binding->param_kinds[i], stored_type);
	}
	Vector<const Variant *> argptrs;
	argptrs.resize(argc);
	for (int i = 0; i < argc; i++) {
		argptrs.write[i] = &args[i];
	}
	const Variant **argv = nullptr;
	if (argc > 0) {
		argv = argptrs.ptrw();
	}

	Variant ret;
	Callable::CallError ce;
	Variant::call_utility_function(binding->name, &ret, argv, argc, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		set_exception(vformat("AngelScript: %s() failed (%d)", binding->name, (int)ce.error));
		return;
	}
	as_binding_marshal_return(p_gen, binding->return_kind, ret, binding->return_type);
}

// 已注册 @GlobalScope 工具函数的引擎集合：RegisterGlobalFunction 重复同名声明会返回
// 负值，而 AS 内部会因此把引擎永久标记为 configFailed（不可恢复）。惰性路径与全量兜底
// 都会调用 register_globals，必须按引擎幂等。
static HashSet<asIScriptEngine *> g_globals_engines;

// 注册所有“签名可表达”的 @GlobalScope 工具函数。vararg（如 print）或签名里出现
// Variant 的工具函数（如 clamp）在 M2 无法表达，统一跳过（见 README 的已知限制）。
// 必须在值类型注册之后调用：签名里会出现 String/Array 等值类型。
void register_global_functions(asIScriptEngine *p_engine) {
	List<StringName> names;
	Variant::get_utility_function_list(&names);

	for (const StringName &name : names) {
		const MethodInfo info = Variant::get_utility_function_info(name);
		String decl;
		String reason;
		if (!ASBindingDecl::method_to_decl(info, &decl, &reason)) {
			continue;
		}

		ASUtilityBinding *binding = memnew(ASUtilityBinding);
		binding->name = name;
		// 与 plan 同一判据：真 void（NIL 且无 NIL_IS_VARIANT）不编组；真 Variant 走 Variant 拷贝构造。
		const PropertyInfo &rv = info.return_val;
		binding->return_kind = (rv.type == Variant::NIL && !(rv.usage & PROPERTY_USAGE_NIL_IS_VARIANT))
				? AS_KIND_VOID
				: ASBindingDecl::resolve(rv).kind;
		binding->return_type = rv.type;
		for (const PropertyInfo &arg : info.arguments) {
			binding->param_kinds.push_back(ASBindingDecl::resolve(arg).kind);
			binding->param_types.push_back(arg.type);
		}

		const CharString decl_utf8 = decl.utf8();
		const int id = p_engine->RegisterGlobalFunction(decl_utf8.get_data(), asFUNCTION(generic_utility_call), asCALL_GENERIC, binding);
		if (id < 0) {
			memdelete(binding);
			continue;
		}
		asIScriptFunction *func = p_engine->GetFunctionById(id);
		if (func != nullptr) {
			func->SetUserData(binding);
		}
	}
}

} // namespace

Error ASBindingRegistry::register_value_types_and_skeletons(const ASBindingPlan &p_plan, asIScriptEngine *p_engine) {
	ERR_FAIL_NULL_V(p_engine, ERR_INVALID_PARAMETER);

	const Error err = ASBindingValueTypes::register_all(p_engine);
	if (err != OK) {
		return err;
	}

	// 两遍注册的第一遍：先给所有类建骨架，再挂成员。方法签名会引用其他类，
	// 一次性逐类注册会让先注册的类因“未知类型”失败，而 AS 里任何一次
	// Register* 失败都会把引擎标记为配置错误且不可恢复。
	for (const ASBindingClass &c : p_plan.get_classes()) {
		const Error e = ASBindingObject::register_skeleton(c, p_engine);
		ERR_FAIL_COND_V_MSG(e != OK, e, vformat("AngelScript: skeleton registration failed for '%s'", c.name));
	}
	// 对象骨架齐备后，才能注册依赖 `Object` 类型名的 Variant 转换构造器及
	// 保存 ObjectID 的强类型容器方法签名。
	const Error object_err = ASBindingValueTypes::register_object_conversions(p_engine);
	ERR_FAIL_COND_V(object_err != OK, object_err);
	return ASTypedContainers::register_all(p_engine);
}

Error ASBindingRegistry::register_types(const ASBindingPlan &p_plan, const Vector<StringName> &p_types, asIScriptEngine *p_engine) {
	ERR_FAIL_NULL_V(p_engine, ERR_INVALID_PARAMETER);

	HashSet<StringName> wanted;
	wanted.reserve(p_types.size());
	for (const StringName &t : p_types) {
		wanted.insert(t);
	}

	// 两遍注册的第二遍：只挂被请求的类的成员。枚举的 scope 即其所属类，随宿主类一起注册。
	for (const ASBindingClass &c : p_plan.get_classes()) {
		if (!wanted.has(c.name)) {
			continue;
		}
		const Error e = ASBindingObject::register_class(c, p_engine);
		ERR_FAIL_COND_V_MSG(e != OK, e, vformat("AngelScript: member registration failed for '%s'", c.name));

		Vector<ASBindingEnum> enums;
		for (const ASBindingEnum &en : p_plan.get_enums()) {
			if (en.scope == c.name) {
				enums.push_back(en);
			}
		}
		const Error ee = ASBindingObject::register_enums(enums, p_engine);
		ERR_FAIL_COND_V_MSG(ee != OK, ee, vformat("AngelScript: enum registration failed for '%s'", c.name));
	}
	return OK;
}

Error ASBindingRegistry::register_globals(asIScriptEngine *p_engine) {
	ERR_FAIL_NULL_V(p_engine, ERR_INVALID_PARAMETER);
	// 幂等：重复注册同名全局函数会污染引擎（configFailed 不可恢复）。
	if (g_globals_engines.has(p_engine)) {
		return OK;
	}
	g_globals_engines.insert(p_engine);
	register_global_functions(p_engine);
	return OK;
}

Error ASBindingRegistry::register_plan(const ASBindingPlan &p_plan, asIScriptEngine *p_engine) {
	Error err = register_value_types_and_skeletons(p_plan, p_engine);
	ERR_FAIL_COND_V(err != OK, err);

	// 全量：一次挂所有类成员。
	Vector<StringName> all;
	all.reserve(p_plan.get_classes().size());
	for (const ASBindingClass &c : p_plan.get_classes()) {
		all.push_back(c.name);
	}
	err = register_types(p_plan, all, p_engine);
	ERR_FAIL_COND_V(err != OK, err);

	return register_globals(p_engine);
}
