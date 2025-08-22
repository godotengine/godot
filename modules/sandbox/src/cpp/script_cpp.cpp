/**************************************************************************/
/*  script_cpp.cpp                                                        */
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

#include "script_cpp.h"

#include "../elf/script_elf.h"
#include "../elf/script_instance.h"
#include "../sandbox_project_settings.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/object/class_db.h"
#include "script_cpp_instance.h"
#include "script_language_cpp.h"
static constexpr bool VERBOSE_LOGGING = false;

bool CPPScript::DetectCMakeOrSConsProject() {
	static bool detected = false;
	static bool detected_value = false;
	// Avoid file system checks if the project type has already been detected
	if (detected) {
		return detected_value;
	}
	// If the project root contains a CMakeLists.txt file, or a cmake/CMakeLists.txt,
	// build the project using CMake
	// Get the project root using res://
	String project_root = "res://";
	detected = true;

	// Check for CMakeLists.txt in the project root
	Ref<FileAccess> fa_cmake = FileAccess::open(project_root + "CMakeLists.txt", FileAccess::READ);
	if (fa_cmake.is_valid()) {
		detected_value = true;
		return true;
	}
	Ref<FileAccess> fa_cmake_dir = FileAccess::open(project_root + "cmake/CMakeLists.txt", FileAccess::READ);
	if (fa_cmake_dir.is_valid()) {
		detected_value = true;
		return true;
	}
	Ref<FileAccess> fa_scons = FileAccess::open(project_root + "SConstruct", FileAccess::READ);
	if (fa_scons.is_valid()) {
		detected_value = true;
		return true;
	}
	detected_value = false;
	return false;
}

bool CPPScript::can_instantiate() const {
	return true;
}
Ref<Script> CPPScript::get_base_script() const {
	return Ref<Script>();
}
StringName CPPScript::get_global_name() const {
	return PathToGlobalName(this->path);
}
bool CPPScript::inherits_script(const Ref<Script> &p_script) const {
	return false;
}
StringName CPPScript::get_instance_base_type() const {
	return StringName("Sandbox");
}
ScriptInstance *CPPScript::instance_create(Object *p_for_object) {
	CPPScriptInstance *instance = memnew(CPPScriptInstance(p_for_object, Ref<CPPScript>(this)));
	instances.insert(instance);
	return instance;
}
PlaceHolderScriptInstance *CPPScript::placeholder_instance_create(Object *p_for_object) {
	return nullptr; // TODO: implement if needed
}
bool CPPScript::instance_has(const Object *p_object) const {
	return false;
}
bool CPPScript::has_source_code() const {
	return true;
}
String CPPScript::get_source_code() const {
	return source_code;
}
void CPPScript::set_source_code(const String &p_code) {
	source_code = p_code;
}
Error CPPScript::reload(bool p_keep_state) {
	this->set_file(this->path);
	return Error::OK;
}
#ifdef TOOLS_ENABLED
Vector<DocData::ClassDoc> CPPScript::get_documentation() const {
	return Vector<DocData::ClassDoc>();
}
#endif
#ifdef TOOLS_ENABLED
String CPPScript::get_class_icon_path() const {
	return String("res://addons/godot_sandbox/CPPScript.svg");
}
#endif
bool CPPScript::has_method(const StringName &p_method) const {
	if (p_method == StringName("_init"))
		return true;
	if (instances.is_empty()) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScript::has_method: No instances available.");
		}
		return false;
	}
	CPPScriptInstance *instance = *instances.begin();
	if (instance == nullptr) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScript::has_method: Instance is null.");
		}
		return false;
	}
	ELFScriptInstance *elf = instance->get_script_instance();
	if (elf == nullptr) {
		return false;
	}
	// Get the method information from the ELFScriptInstance
	if (elf->get_elf_script()) {
		return elf->get_elf_script()->has_method(p_method);
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScript::has_method: ELFScriptInstance had no ELFScript!?");
	}
	return false;
}
bool CPPScript::has_static_method(const StringName &p_method) const {
	return false;
}
MethodInfo CPPScript::get_method_info(const StringName &p_method) const {
	if (instances.is_empty()) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScript::_get_method_info: No instances available.");
		}
		return MethodInfo();
	}
	CPPScriptInstance *instance = *instances.begin();
	if (instance == nullptr) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScript::get_method_info: Instance is null.");
		}
		return MethodInfo();
	}
	ELFScriptInstance *elf = instance->get_script_instance();
	if (elf == nullptr) {
		return MethodInfo();
	}
	// Get the method information from the ELFScriptInstance
	MethodInfo mi;
	mi.name = p_method;
	mi.flags = METHOD_FLAG_VARARG;
	return mi;
}
bool CPPScript::is_tool() const {
	return true;
}
bool CPPScript::is_valid() const {
	return true;
}
bool CPPScript::is_abstract() const {
	return false;
}
ScriptLanguage *CPPScript::get_language() const {
	return nullptr; // TODO: implement CPPScriptLanguage singleton
}
bool CPPScript::has_script_signal(const StringName &p_signal) const {
	return false;
}
void CPPScript::update_exports() {}
int CPPScript::get_member_line(const StringName &p_member) const {
	return 0;
}
void CPPScript::get_constants(HashMap<StringName, Variant> *p_constants) {
	// No constants to add
}
void CPPScript::get_members(HashSet<StringName> *p_members) {
	// No members to add
}
const Variant CPPScript::get_rpc_config() const {
	return Variant();
}

#ifdef TOOLS_ENABLED
bool CPPScript::is_placeholder_fallback_enabled() const {
	return false;
}

StringName CPPScript::get_doc_class_name() const {
	return get_global_name();
}
#endif

void CPPScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	// No signals to add
}

bool CPPScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	return false;
}

void CPPScript::get_script_method_list(List<MethodInfo> *p_list) const {
	// Add basic method info for available functions
}

void CPPScript::get_script_property_list(List<PropertyInfo> *p_list) const {
	// Add script properties if needed
}

#ifdef TOOLS_ENABLED
void CPPScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
	// Handle placeholder cleanup if needed
}
#endif

CPPScript::CPPScript() {
	source_code = R"C0D3(#include "api.hpp"

static Variant my_function(Vector4 v) {
	print("Arg: ", v);
	return 123;
}

static Variant _process() {
	static int counter = 0;
	if (++counter % 100 == 0) {
		print("Process called " + std::to_string(counter) + " times");
	}
	return Nil;
}

static Vector4 my_vector4(1.0f, 2.0f, 3.0f, 4.0f);
static String my_string("Hello, World!");
int main() {
	ADD_PROPERTY(my_vector4, Variant::VECTOR4);
	ADD_PROPERTY(my_string, Variant::STRING);

	ADD_API_FUNCTION(my_function, "int", "Vector4 v");
	ADD_API_FUNCTION(_process, "void");
}
)C0D3";
}
CPPScript::~CPPScript() {
}

void CPPScript::set_file(const String &p_path) {
	if (p_path.is_empty()) {
		WARN_PRINT("CPPScript::set_file: Empty resource path.");
		return;
	}
	this->path = p_path;
	this->source_code = FileAccess::get_file_as_string(p_path);
}
bool CPPScript::detect_script_instance() {
	// It's possible to speculate that eg. a fitting ELFScript would be located at
	// "res://this/path.cpp" replacing the extension with ".elf".
	if (this->path.is_empty()) {
		WARN_PRINT("CPPScript::detect_script_instance: Empty resource path.");
		return false;
	}
	const String elf_path = this->path.get_basename() + ".elf";
	Ref<FileAccess> fa_elf = FileAccess::open(elf_path, FileAccess::READ);
	if (fa_elf.is_valid()) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("CPPScript::detect_script_instance: Found ELF script at " + elf_path);
		}
		// Try to get the resource from the path
		Ref<ELFScript> res = ResourceLoader::load(elf_path, "ELFScript");
		if (res.is_valid()) {
			if constexpr (VERBOSE_LOGGING) {
				ERR_PRINT("CPPScript::detect_script_instance: ELF script loaded successfully.");
				this->elf_script = res;
				return true;
			}
		}
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("CPPScript::detect_script_instance: No ELF script found at " + elf_path);
	}
	return false;
}
void CPPScript::remove_instance(CPPScriptInstance *p_instance) {
	instances.erase(p_instance);
	if (instances.is_empty()) {
		this->elf_script.unref();
	}
}
