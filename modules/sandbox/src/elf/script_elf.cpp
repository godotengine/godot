/**************************************************************************/
/*  script_elf.cpp                                                        */
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

#include "script_elf.h"

#include "../cpp/script_cpp.h"
// Docker support removed
#include "../register_types.h"
#include "../sandbox.h"
#include "../sandbox_project_settings.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/variant/variant.h"
#include "script_instance.h"
#include "script_language_elf.h"

static constexpr bool VERBOSE_ELFSCRIPT = false;

// Provide access to ELF language singleton
ScriptLanguage *get_elf_language_singleton() {
	static ELFScriptLanguage elf_language;
	return &elf_language;
}

void ELFScript::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_sandbox_for", "for_object"), &ELFScript::get_sandbox_for);
	ClassDB::bind_method(D_METHOD("get_sandbox_objects"), &ELFScript::get_sandbox_objects);
	ClassDB::bind_method(D_METHOD("get_content"), &ELFScript::get_content);
}

Sandbox *ELFScript::get_sandbox_for(Object *p_for_object) const {
	for (ELFScriptInstance *instance : this->instances) {
		if (instance->get_owner() == p_for_object) {
			auto [sandbox, auto_created] = instance->get_sandbox();
			return sandbox;
		}
	}
	ERR_PRINT("ELFScript::get_sandbox_for: Sandbox not found for object " + p_for_object->get_class());
	return nullptr;
}

Array ELFScript::get_sandbox_objects() const {
	Array result;
	for (ELFScriptInstance *instance : this->instances) {
		result.push_back(instance->get_owner());
	}
	return result;
}

ELFScriptInstance *ELFScript::get_script_instance(Object *p_for_object) const {
	for (ELFScriptInstance *instance : this->instances) {
		if (instance->get_owner() == p_for_object) {
			return instance;
		}
	}
	ERR_PRINT("ELFScript::get_script_instance: Script instance not found for object " + p_for_object->get_class());
	return nullptr;
}

// Internal Script API methods (no underscore prefix)
bool ELFScript::can_instantiate() const {
	return true;
}
Ref<Script> ELFScript::get_base_script() const {
	return Ref<Script>();
}
StringName ELFScript::get_global_name() const {
	if (SandboxProjectSettings::use_global_sandbox_names()) {
		return global_name;
	}
	return "ELFScript";
}
bool ELFScript::inherits_script(const Ref<Script> &p_script) const {
	return false;
}
StringName ELFScript::get_instance_base_type() const {
	return StringName("Sandbox");
}
ScriptInstance *ELFScript::instance_create(Object *p_for_object) {
	ELFScriptInstance *instance = memnew(ELFScriptInstance(p_for_object, Ref<ELFScript>(this)));
	instances.insert(instance);
	return instance;
}
PlaceHolderScriptInstance *ELFScript::placeholder_instance_create(Object *p_for_object) {
	return nullptr; // TODO: implement if needed
}
bool ELFScript::instance_has(const Object *p_object) const {
	return false;
}
bool ELFScript::has_source_code() const {
	return true;
}
String ELFScript::get_source_code() const {
	if (source_code.is_empty()) {
		return String();
	}
	if (functions.is_empty()) {
		return Variant(function_names).stringify();
	} else {
		return Variant(functions).stringify();
	}
}
void ELFScript::set_source_code(const String &p_code) {
}
Error ELFScript::reload(bool p_keep_state) {
	this->source_version++;
	this->set_file(this->path);
	return Error::OK;
}
bool ELFScript::has_method(const StringName &p_method) const {
	bool result = function_names.find(p_method) != -1;
	if (!result) {
		if (p_method == StringName("_init"))
			result = true;
	}
	if constexpr (VERBOSE_ELFSCRIPT) {
		printf("ELFScript::_has_method: method %s => %s\n",
				String(p_method).utf8().ptr(), result ? "true" : "false");
	}

	return result;
}
bool ELFScript::has_static_method(const StringName &p_method) const {
	return false;
}
MethodInfo ELFScript::get_method_info(const StringName &p_method) const {
	for (const String &function : function_names) {
		if (function == p_method) {
			if constexpr (VERBOSE_ELFSCRIPT) {
				printf("ELFScript::get_method_info: method %s\n", String(p_method).utf8().ptr());
			}
			MethodInfo mi;
			mi.name = function;
			mi.flags = METHOD_FLAG_VARARG;
			return mi;
		}
	}
	return MethodInfo();
}
bool ELFScript::is_tool() const {
	return true;
}
bool ELFScript::is_valid() const {
	return true;
}
bool ELFScript::is_abstract() const {
	return false;
}
ScriptLanguage *ELFScript::get_language() const {
	return get_elf_language_singleton();
}
bool ELFScript::has_script_signal(const StringName &p_signal) const {
	return false;
}
void ELFScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	// No signals to add
}

bool ELFScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	return false;
}
void ELFScript::update_exports() {
}
void ELFScript::get_script_method_list(List<MethodInfo> *p_list) const {
	// Convert our functions to MethodInfo format
	for (const String &function : function_names) {
		MethodInfo mi;
		mi.name = function;
		mi.flags = METHOD_FLAG_VARARG;
		p_list->push_back(mi);
	}
}
void ELFScript::get_script_property_list(List<PropertyInfo> *p_list) const {
	std::vector<PropertyInfo> props = Sandbox::create_sandbox_property_list();
	for (const PropertyInfo &prop : props) {
		p_list->push_back(prop);
	}
}
int ELFScript::get_member_line(const StringName &p_member) const {
	PackedStringArray formatted_functions = get_source_code().split("\n");
	for (int i = 0; i < formatted_functions.size(); i++) {
		if (formatted_functions[i].find(p_member) != -1) {
			return i + 1;
		}
	}
	return 0;
}
void ELFScript::get_constants(HashMap<StringName, Variant> *p_constants) {
	// No constants to add
}
void ELFScript::get_members(HashSet<StringName> *p_members) {
	// Add function names as members
	for (const String &function : function_names) {
		p_members->insert(StringName(function));
	}
}
const Variant ELFScript::get_rpc_config() const {
	return Variant();
}

#ifdef TOOLS_ENABLED
StringName ELFScript::get_doc_class_name() const {
	return get_global_name();
}
void ELFScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
	// Handle placeholder cleanup if needed
}
String ELFScript::get_class_icon_path() const {
	return String("res://addons/godot_sandbox/Sandbox.svg");
}
Vector<DocData::ClassDoc> ELFScript::get_documentation() const {
	return Vector<DocData::ClassDoc>();
}
bool ELFScript::is_placeholder_fallback_enabled() const {
	return false;
}
#endif

const PackedByteArray &ELFScript::get_content() {
	return source_code;
}

String ELFScript::get_elf_programming_language() const {
	return elf_programming_language;
}

void ELFScript::set_file(const String &p_path) {
	if (p_path.is_empty()) {
		if constexpr (VERBOSE_ELFSCRIPT) {
			printf("ELFScript::set_file: Empty path provided, skipping.\n");
		}
		return;
	}
	// res://path/to/file.elf
	this->path = String(p_path);
	// path/to/file.elf as a C++ string
	CharString resless_path = p_path.replace("res://", "").utf8();
	this->std_path = std::string(resless_path.ptr(), resless_path.length());

	PackedByteArray new_source_code = FileAccess::get_file_as_bytes(this->path);
	if (new_source_code == source_code) {
		if constexpr (VERBOSE_ELFSCRIPT) {
			printf("ELFScript::set_file: No changes in %s\n", path.utf8().ptr());
		}
		return;
	} else if (new_source_code.is_empty()) {
		ERR_FAIL_MSG("ELFScript::set_file: Failed to load file '" + this->path + "'. The file is empty or does not exist.");
		return;
	}
	source_code = std::move(new_source_code);

	global_name = "Sandbox_" + path.get_basename().replace("res://", "").replace("/", "_").replace("-", "_").capitalize().replace(" ", "");
	Sandbox::BinaryInfo info = Sandbox::get_program_info_from_binary(source_code);
	this->function_names = std::move(info.functions);
	this->functions.clear();

	this->elf_programming_language = info.language;
	this->elf_api_version = info.version;

	if constexpr (VERBOSE_ELFSCRIPT) {
		printf("ELFScript::set_file: %s Sandbox instances: %u\n", std_path.c_str(), sandbox_map[path].size());
	}
	for (Sandbox *sandbox : sandbox_map[path]) {
		sandbox->set_program(Ref<ELFScript>(this));
	}

	// Update the instance methods only if functions are still empty
	if (functions.is_empty()) {
		for (ELFScriptInstance *instance : this->instances) {
			instance->update_methods();
		}
	}
}

void ELFScript::set_public_api_functions(Array &&p_functions) {
	functions = std::move(p_functions);

	if constexpr (VERBOSE_ELFSCRIPT) {
		printf("ELFScript::set_public_api_functions: %s\n", path.utf8().ptr());
	}
	this->update_public_api_functions();
}

void ELFScript::update_public_api_functions() {
	// Update the function names
	function_names.clear();
	for (int i = 0; i < functions.size(); i++) {
		Dictionary func = functions[i];
		function_names.push_back(func["name"]);
	}

	// Update the instance methods
	for (ELFScriptInstance *instance : this->instances) {
		instance->update_methods();
	}
}

String ELFScript::get_dockerized_program_path() const {
	// Docker support removed - return empty string
	return String();
}
