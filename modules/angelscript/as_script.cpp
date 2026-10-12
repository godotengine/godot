/**************************************************************************/
/*  as_script.cpp                                                         */
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

#include "as_script.h"

#include "as_bytecode.h"
#include "as_engine.h"
#include "as_script_instance.h"
#include "as_script_language.h"
#include "binding/as_binding_decl.h"
#include "binding/as_binding_lazy.h"
#include "binding/as_binding_plan.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/object/class_db.h"

static const char *AS_BASE_DIRECTIVE = "// godot_base:";

// 扫描前 10 行里的 `// godot_base:` 指令；找到返回 true 并写出实参，否则返回 false。
// 必须忽略块注释里的伪指令：源码注释是作者侧文档，`/* ... */` 里恰好有一行以指令开头
// 时（旧实现按逐行字面量匹配会误采纳）不能左右基类。编辑器高亮词法器早已区分块注释，
// 这里对齐同一语义：跟踪块注释状态，并跳过字符串/字符字面量与行注释，避免把 `"/*"`
// 这类文本当成注释开始。
static bool _scan_base_directive(const Vector<String> &p_lines, String *r_base_type) {
	const int scan_lines = MIN(p_lines.size(), 10);
	bool in_block_comment = false;
	for (int i = 0; i < scan_lines; i++) {
		const String &line = p_lines[i];
		const bool line_starts_in_block = in_block_comment;
		bool in_string = false;
		char32_t string_quote = 0;
		for (int j = 0; j < line.length(); j++) {
			const char32_t c = line[j];
			if (in_block_comment) {
				if (c == U'*' && j + 1 < line.length() && line[j + 1] == U'/') {
					in_block_comment = false;
					j++;
				}
				continue;
			}
			if (in_string) {
				if (c == U'\\') {
					j++;
					continue;
				}
				if (c == string_quote) {
					in_string = false;
				}
				continue;
			}
			if (c == U'/' && j + 1 < line.length()) {
				if (line[j + 1] == U'*') {
					in_block_comment = true;
					j++;
					continue;
				}
				if (line[j + 1] == U'/') {
					break; // 行注释：本行余下内容不再是代码（含嵌入式指令）。
				}
			}
			if (c == U'"' || c == U'\'') {
				in_string = true;
				string_quote = c;
			}
		}
		// 整行位于块注释内时，即便文本看起来像指令也不采纳。
		if (line_starts_in_block) {
			continue;
		}
		const String trimmed = line.strip_edges();
		if (trimmed.begins_with(AS_BASE_DIRECTIVE)) {
			if (r_base_type) {
				*r_base_type = trimmed.substr(String(AS_BASE_DIRECTIVE).length()).strip_edges();
			}
			return true;
		}
	}
	return false;
}

// AngelScript 的模块名直接复用资源路径：热更时同一路径整体替换（spec §7），
// 且路径天然唯一，不需要额外的模块名分配表。
ASScript::ASScript() {
}

ASScript::~ASScript() {
	clear();
}

void ASScript::clear() {
	valid = false;
	compile_error = String();
	class_name = StringName();
	instance_base_type = StringName();
	source_code = String();
	signals.clear();
	if (!module_name.is_empty() && ASEngine::get_singleton()->is_initialized()) {
		asIScriptEngine *engine = ASEngine::get_singleton()->get_engine();
		CharString module_cstr = module_name.utf8();
		if (engine->GetModule(module_cstr.get_data(), asGM_ONLY_IF_EXISTS) != nullptr) {
			engine->DiscardModule(module_cstr.get_data());
		}
	}
	module_name = String();
}

String ASScript::get_class_name_for_path(const String &p_path) {
	return p_path.get_file().get_basename();
}

asITypeInfo *ASScript::_find_script_class(asIScriptModule *p_module, const String &p_class_name) {
	if (p_module == nullptr) {
		return nullptr;
	}
	for (asUINT i = 0; i < p_module->GetObjectTypeCount(); i++) {
		asITypeInfo *type = p_module->GetObjectTypeByIndex(i);
		if (type == nullptr || !(type->GetFlags() & asOBJ_SCRIPT_OBJECT)) {
			continue;
		}
		if (type->GetName() != nullptr && String(type->GetName()) == p_class_name) {
			return type;
		}
	}
	return nullptr;
}

bool ASScript::compile_source(const String &p_source, const String &p_path, String *r_error) {
	// 先取副本再 clear()：reload() 会把成员 source_code 作为 p_source 传进来（别名），
	// clear() 清空 source_code 后 p_source 会失效，拷贝必须发生在此之前。
	String src = p_source;
	clear();

	if (p_path.is_empty()) {
		compile_error = "AngelScript requires a resource path to derive the class name";
		if (r_error) {
			*r_error = compile_error;
		}
		return false;
	}

	// 统一行尾并剥掉 UTF-8 BOM：Godot 内部字符串是 UTF-32，BOM 会变成首字符 U+FEFF，
	// 不剥掉就会让第一行的 `// godot_base:` 指令扫描失效。
	// 不能用 `String::utf8("\xEF\xBB\xBF", 3)` 判定：parse_utf8/append_utf8 遇到开头的 BOM 会
	// 直接跳过（见 core/string/ustring.cpp 中 "just skip it" 注释），该表达式恒为空串，而空串是
	// 任意字符串的前缀（begins_with("") 恒真），会无条件剥掉首字符。这里直接比对码点。
	src = src.replace("\r\n", "\n");
	if (!src.is_empty() && src[0] == char32_t(0xFEFF)) {
		src = src.substr(1);
	}

	const String expected_class = get_class_name_for_path(p_path);
	if (expected_class.is_empty()) {
		compile_error = "cannot derive class name from path: " + p_path;
		if (r_error) {
			*r_error = compile_error;
		}
		return false;
	}

	StringName base_type;
	bool found_directive = false;
	Vector<String> lines = src.split("\n");
	String base_directive_arg;
	if (_scan_base_directive(lines, &base_directive_arg)) {
		base_type = StringName(base_directive_arg);
		found_directive = true;
	}

	if (!found_directive) {
		compile_error = String("missing '") + AS_BASE_DIRECTIVE + " <Type>' directive in the first 10 lines";
		if (r_error) {
			*r_error = compile_error;
		}
		return false;
	}
	if (!ClassDB::class_exists(base_type)) {
		compile_error = "unknown base type: " + String(base_type);
		if (r_error) {
			*r_error = compile_error;
		}
		return false;
	}

	ASEngine *as = ASEngine::get_singleton();
	if (!as->ensure_initialized()) {
		compile_error = "AngelScript engine is not available";
		if (r_error) {
			*r_error = compile_error;
		}
		return false;
	}

	const String new_module_name = p_path;
	String error;
	if (!as->compile_module(new_module_name, src, &error)) {
		compile_error = error;
		if (r_error) {
			*r_error = compile_error;
		}
		return false;
	}

	module_name = new_module_name;
	asITypeInfo *type = _find_script_class(get_module(), expected_class);
	if (type == nullptr) {
		// 先 clear() 丢弃刚编译出来的模块，再写错误信息：clear() 会把 compile_error 清空，
		// 顺序颠倒会让 r_error 变成空串。
		clear();
		compile_error = "script class '" + expected_class + "' (must equal the file name) was not found in " + p_path;
		if (r_error) {
			*r_error = compile_error;
		}
		return false;
	}

	source_code = src;
	class_name = expected_class;
	instance_base_type = base_type;
	valid = true;
	_collect_signals();
	return true;
}

Error ASScript::save_bytecode(const String &p_out_path, const Vector<StringName> &p_required_types, String *r_error) {
	if (!valid) {
		if (r_error) {
			*r_error = "Cannot save bytecode for an invalid script.";
		}
		return ERR_INVALID_DATA;
	}
	asIScriptModule *module = get_module();
	if (module == nullptr) {
		if (r_error) {
			*r_error = "Cannot save bytecode: the script module is not available.";
		}
		return ERR_INVALID_DATA;
	}

	Vector<uint8_t> payload;
	ASVectorWriteStream stream(payload);
	// stripDebugInfo 必须为 false，否则信号参数名会丢失。
	if (module->SaveByteCode(&stream, false) < 0) {
		if (r_error) {
			*r_error = "AngelScript failed to serialize the module bytecode.";
		}
		return ERR_CANT_CREATE;
	}

	Vector<uint8_t> packed = as_bytecode_pack(instance_base_type, p_required_types, payload);

	String dir = p_out_path.get_base_dir();
	if (!dir.is_empty() && !DirAccess::dir_exists_absolute(dir)) {
		DirAccess::make_dir_recursive_absolute(dir);
	}
	Error open_error = OK;
	Ref<FileAccess> file = FileAccess::open(p_out_path, FileAccess::WRITE, &open_error);
	if (file.is_null()) {
		if (r_error) {
			*r_error = vformat("Cannot open '%s' for writing bytecode.", p_out_path);
		}
		return open_error == OK ? ERR_CANT_CREATE : open_error;
	}
	if (!file->store_buffer(packed) || file->get_error() != OK) {
		if (r_error) {
			*r_error = vformat("Cannot write bytecode to '%s'.", p_out_path);
		}
		return ERR_FILE_CANT_WRITE;
	}
	file->flush();
	if (file->get_error() != OK) {
		if (r_error) {
			*r_error = vformat("Cannot flush bytecode to '%s'.", p_out_path);
		}
		return ERR_FILE_CANT_WRITE;
	}
	if (r_error) {
		*r_error = String();
	}
	return OK;
}

bool ASScript::load_bytecode(const Vector<uint8_t> &p_bytes, const String &p_path, String *r_error) {
	clear();
	if (p_path.is_empty()) {
		if (r_error) {
			*r_error = "Cannot load bytecode without a resource path.";
		}
		return false;
	}
	const String expected_class = get_class_name_for_path(p_path);

	ASEngine::get_singleton()->ensure_initialized();
	asIScriptEngine *engine = ASEngine::get_singleton()->get_engine();
	if (engine == nullptr) {
		if (r_error) {
			*r_error = "Cannot load bytecode: the AngelScript engine is not available.";
		}
		return false;
	}

	ASByteCode bytecode;
	String unpack_error;
	if (as_bytecode_unpack(p_bytes, bytecode, &unpack_error) != OK) {
		if (r_error) {
			*r_error = unpack_error;
		}
		return false;
	}

	// 容器携带符号表：加载前先把字节码引用的 ClassDB 类型（含基类）增量注册，避免退回全量注册。
	// 严禁在此处退回 register_all 做兜底：那会让懒启动（惰性注册）秒级启动的收益全部作废。
	// 符号表缺失/类型未注册时的正确行为是让 LoadByteCode 失败并向上报错，由业务层负责
	// 重新下发匹配 ABI 的 `.asb`（ABI/签名/回退策略归业务层 manifest，见 M4 spec §6.4）。
	Vector<StringName> register_types = bytecode.required_types;
	if (!bytecode.base_type.is_empty() && !register_types.has(bytecode.base_type)) {
		register_types.push_back(bytecode.base_type);
	}
	ASBindingLazyRegistry::get_singleton()->ensure_registered(
			ASEngine::get_singleton()->get_binding_plan(), register_types, engine);

	module_name = p_path;
	asIScriptModule *module = engine->GetModule(module_name.utf8().get_data(), asGM_ALWAYS_CREATE);
	if (module == nullptr) {
		clear();
		if (r_error) {
			*r_error = "AngelScript failed to create a module for bytecode.";
		}
		return false;
	}

	ASMemoryReadStream stream(bytecode.payload.ptr(), (uint32_t)bytecode.payload.size());
	bool was_debug_info_stripped = false;
	if (module->LoadByteCode(&stream, &was_debug_info_stripped) < 0) {
		clear();
		if (r_error) {
			*r_error = "Unable to load AngelScript bytecode (missing symbols or corrupted payload).";
		}
		return false;
	}

	asITypeInfo *type = _find_script_class(module, expected_class);
	if (type == nullptr) {
		clear();
		if (r_error) {
			*r_error = vformat("Bytecode file name and script class name must match (expected '%s').", expected_class);
		}
		return false;
	}

	// AS 脚本类不携带 `// godot_base:` 继承信息，基类由容器随字节码携带。
	if (bytecode.base_type.is_empty()) {
		clear();
		if (r_error) {
			*r_error = "Loaded bytecode is missing its base type.";
		}
		return false;
	}
	const String base_name = String(bytecode.base_type);
	if (!ClassDB::class_exists(bytecode.base_type)) {
		clear();
		if (r_error) {
			*r_error = vformat("Loaded bytecode base class '%s' is not a known ClassDB type.", base_name);
		}
		return false;
	}

	class_name = expected_class;
	instance_base_type = bytecode.base_type;
	valid = true;
	_collect_signals();
	if (r_error) {
		*r_error = String();
	}
	return true;
}

asIScriptModule *ASScript::get_module() const {
	if (module_name.is_empty() || !ASEngine::get_singleton()->is_initialized()) {
		return nullptr;
	}
	CharString module_cstr = module_name.utf8();
	return ASEngine::get_singleton()->get_engine()->GetModule(module_cstr.get_data(), asGM_ONLY_IF_EXISTS);
}

asITypeInfo *ASScript::get_type_info() const {
	if (!valid) {
		return nullptr;
	}
	return _find_script_class(get_module(), class_name);
}

Error ASScript::reload(bool p_keep_state) {
	if (source_code.is_empty()) {
		return ERR_INVALID_DATA;
	}
	String error;
	if (!compile_source(source_code, get_path().is_empty() ? module_name : get_path(), &error)) {
		return ERR_PARSE_ERROR;
	}
#ifdef TOOLS_ENABLED
	// 重编译后属性集合可能变化：刷新占位实例，使 Inspector 立即显示新属性（等价 GDScript update_exports）。
	update_placeholders();
#endif
	return OK;
}

#ifdef TOOLS_ENABLED
void ASScript::_add_instance(ASScriptInstance *p_instance) {
	instances.insert(p_instance);
}

void ASScript::_remove_instance(ASScriptInstance *p_instance) {
	instances.erase(p_instance);
}

void ASScript::update_placeholders() {
	if (placeholders.is_empty()) {
		return;
	}
	List<PropertyInfo> props;
	get_script_property_list(&props);
	HashMap<StringName, Variant> values;
	for (PlaceHolderScriptInstance *E : placeholders) {
		E->update(props, values);
	}
}

void ASScript::reload_with_instances(bool p_keep_state) {
	// 保存-重编译-重挂：必须在重编译前摘除所有实例与占位实例。reload() 内部 clear() 会
	// DiscardModule()，若仍有活跃的 asIScriptObject 指向该模块，重挂之前就已悬垂（UAF）。
	HashMap<ObjectID, List<Pair<StringName, Variant>>> saved_states;

	Vector<ASScriptInstance *> live_instances;
	live_instances.reserve(instances.size());
	for (ASScriptInstance *E : instances) {
		live_instances.push_back(E);
	}
	for (ASScriptInstance *E : live_instances) {
		Object *owner = E->get_owner();
		if (owner == nullptr) {
			continue;
		}
		E->get_property_state(saved_states[owner->get_instance_id()]);
		// 摘除会析构实例并经 _remove_instance 回改集合，因此遍历的是上面的快照。
		owner->set_script(Variant());
	}

	Vector<PlaceHolderScriptInstance *> live_placeholders;
	live_placeholders.reserve(placeholders.size());
	for (PlaceHolderScriptInstance *E : placeholders) {
		live_placeholders.push_back(E);
	}
	for (PlaceHolderScriptInstance *E : live_placeholders) {
		Object *owner = E->get_owner();
		if (owner == nullptr || owner->get_script_instance() != E) {
			continue;
		}
		E->get_property_state(saved_states[owner->get_instance_id()]);
		owner->set_script(Variant());
	}

	reload(p_keep_state);

	for (KeyValue<ObjectID, List<Pair<StringName, Variant>>> &E : saved_states) {
		Object *obj = ObjectDB::get_instance(E.key);
		if (obj == nullptr) {
			continue;
		}
		obj->set_script(Ref<Script>(this));
		ScriptInstance *si = obj->get_script_instance();
		if (si == nullptr) {
			continue;
		}
		if (si->is_placeholder()) {
			PlaceHolderScriptInstance *ph = static_cast<PlaceHolderScriptInstance *>(si);
			for (const Pair<StringName, Variant> &G : E.value) {
				ph->property_set_fallback(G.first, G.second);
			}
		} else {
			for (const Pair<StringName, Variant> &G : E.value) {
				si->set(G.first, G.second);
			}
		}
	}
}
#endif

bool ASScript::has_method(const StringName &p_method) const {
	// `signal_<name>` 是信号声明约定，不作为普通方法暴露（与 has_script_signal 分工）。
	if (String(p_method).begins_with("signal_")) {
		return false;
	}
	asITypeInfo *type = get_type_info();
	if (type == nullptr) {
		return false;
	}
	CharString name = String(p_method).utf8();
	return type->GetMethodByName(name.get_data(), true) != nullptr;
}

ScriptLanguage *ASScript::get_language() const {
	return ASScriptLanguage::get_singleton();
}

void ASScript::get_script_method_list(List<MethodInfo> *p_list) const {
	asITypeInfo *type = get_type_info();
	if (type == nullptr) {
		return;
	}
	for (asUINT i = 0; i < type->GetMethodCount(); i++) {
		asIScriptFunction *func = type->GetMethodByIndex(i, false);
		if (func == nullptr || func->GetName() == nullptr) {
			continue;
		}
		// `signal_<name>` 是信号声明约定，不进普通方法表。
		if (String(func->GetName()).begins_with("signal_")) {
			continue;
		}
		MethodInfo mi;
		mi.name = StringName(func->GetName());
		p_list->push_back(mi);
	}
}

void ASScript::_collect_signals() {
	signals.clear();
	asITypeInfo *type = get_type_info();
	if (type == nullptr) {
		return;
	}
	for (asUINT i = 0; i < type->GetMethodCount(); i++) {
		// 必须用 getVirtual=false 取真实实现：AS 的虚函数桩（CreateVirtualFunction）不复制参数名，
		// 用默认的 true 会拿到空的 parameterNames（类型却仍是拷贝过来的，故只有名字会丢）。
		asIScriptFunction *fn = type->GetMethodByIndex(i, false);
		if (fn == nullptr || fn->GetName() == nullptr) {
			continue;
		}
		const String full_name = String(fn->GetName());
		if (!full_name.begins_with("signal_")) {
			continue;
		}
		ASSignalInfo sig;
		sig.name = StringName(full_name.substr(7));
		bool mapped = true;
		for (asUINT p = 0; p < fn->GetParamCount(); p++) {
			int type_id = 0;
			const char *arg_name = nullptr;
			fn->GetParam(p, &type_id, nullptr, &arg_name);
			PropertyInfo pi;
			switch (type_id) {
				case asTYPEID_BOOL:
					pi.type = Variant::BOOL;
					break;
				case asTYPEID_INT32:
				case asTYPEID_INT64:
					pi.type = Variant::INT;
					break;
				case asTYPEID_FLOAT:
				case asTYPEID_DOUBLE:
					pi.type = Variant::FLOAT;
					break;
				default: {
					asITypeInfo *ti = ASEngine::get_singleton()->get_engine()->GetTypeInfoById(type_id & ~asTYPEID_OBJHANDLE);
					if (ti == nullptr) {
						mapped = false;
						break;
					}
					const Variant::Type vt = ASBindingDecl::as_name_to_variant_type(String(ti->GetName()));
					if (vt == Variant::NIL) {
						// 对象类型参数（`Node@` 等）本轮不支持：整个信号不声明，避免 connect/emit 时参数对不上。
						mapped = false;
						break;
					}
					pi.type = vt;
					break;
				}
			}
			if (!mapped) {
				break;
			}
			pi.name = String(arg_name != nullptr ? arg_name : "");
			sig.args.push_back(pi);
		}
		if (mapped) {
			signals.push_back(sig);
		}
	}
}

bool ASScript::has_script_signal(const StringName &p_signal) const {
	for (const ASSignalInfo &sig : signals) {
		if (sig.name == p_signal) {
			return true;
		}
	}
	return false;
}

void ASScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	for (const ASSignalInfo &sig : signals) {
		MethodInfo mi;
		mi.name = sig.name;
		mi.arguments = sig.args;
		r_signals->push_back(mi);
	}
}

void ASScript::get_script_property_list(List<PropertyInfo> *p_list) const {
	asITypeInfo *type = get_type_info();
	if (type == nullptr) {
		return;
	}
	for (asUINT i = 0; i < type->GetPropertyCount(); i++) {
		// asITypeInfo 的枚举式属性查询（asIScriptObject 上的 GetPropertyName/GetPropertyTypeId 是实例接口，此处不可用）。
		const char *prop_name = nullptr;
		int type_id = 0;
		if (type->GetProperty(i, &prop_name, &type_id) < 0 || prop_name == nullptr) {
			continue;
		}
		Variant::Type vt = Variant::NIL;
		StringName prop_class_name;
		if (type_id & asTYPEID_OBJHANDLE) {
			// 对象句柄属性：报 OBJECT 并带上静态类型名，供编辑器/调用方校验。
			asIScriptEngine *engine = ASEngine::get_singleton()->get_engine();
			asITypeInfo *ti = engine != nullptr ? engine->GetTypeInfoById(type_id) : nullptr;
			if (ti == nullptr) {
				continue;
			}
			vt = Variant::OBJECT;
			prop_class_name = StringName(ti->GetName());
		} else if (type_id & asTYPEID_MASK_OBJECT) {
			// 内建值类型（AS 侧槽就是一颗 Variant），经绑定层映射回 Variant::Type。
			asIScriptEngine *engine = ASEngine::get_singleton()->get_engine();
			asITypeInfo *ti = engine != nullptr ? engine->GetTypeInfoById(type_id) : nullptr;
			if (ti == nullptr) {
				continue;
			}
			const String tn = String(ti->GetName());
			if (tn == "string") {
				continue; // 小写内建 string 不做往返（与 ASScriptInstance::set/get 一致）。
			}
			vt = ASBindingDecl::as_name_to_variant_type(tn);
			if (vt == Variant::NIL) {
				continue;
			}
		} else {
			switch (type_id) {
				case asTYPEID_BOOL:
					vt = Variant::BOOL;
					break;
				case asTYPEID_INT32:
				case asTYPEID_INT64:
					vt = Variant::INT;
					break;
				case asTYPEID_FLOAT:
				case asTYPEID_DOUBLE:
					vt = Variant::FLOAT;
					break;
				default:
					continue; // 不支持的类型（既非标量也非已映射的值类型/句柄）不报成有效属性。
			}
		}
		if (vt == Variant::OBJECT) {
			p_list->push_back(PropertyInfo(Variant::OBJECT, String(prop_name), PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT, prop_class_name));
		} else {
			p_list->push_back(PropertyInfo(vt, StringName(prop_name)));
		}
	}
}

ScriptInstance *ASScript::instance_create(Object *p_this) {
	if (!valid) {
		return nullptr;
	}
	ASScriptInstance *instance = memnew(ASScriptInstance(Ref<ASScript>(this), p_this));
	if (!instance->is_instantiated()) {
		// 构造失败（无默认 factory / 构造执行异常）时不能把半成品交出去，
		// 否则 Object 会把一个无法调用任何方法的"哑实例"当成有效脚本实例。
		memdelete(instance);
		return nullptr;
	}
	return instance;
}

PlaceHolderScriptInstance *ASScript::placeholder_instance_create(Object *p_this) {
#ifdef TOOLS_ENABLED
	// 编辑器里脚本尚不可实例化时（例如刚由「新建脚本」创建、还未编译成功），
	// Object::set_script() 会退而请求占位实例。这里若返回 nullptr，节点就拿不到任何
	// ScriptInstance，表现为「挂不上脚本、Inspector 的 Script 栏为空」。
	PlaceHolderScriptInstance *si = memnew(PlaceHolderScriptInstance(ASScriptLanguage::get_singleton(), Ref<Script>(this), p_this));
	placeholders.insert(si);
	// 脚本若已编译，属性列表可用；未编译时为空列表，update 也能安全接受。
	List<PropertyInfo> props;
	get_script_property_list(&props);
	HashMap<StringName, Variant> values;
	si->update(props, values);
	return si;
#else
	return nullptr;
#endif
}

#ifdef TOOLS_ENABLED
void ASScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
	placeholders.erase(p_placeholder);
}
#endif
