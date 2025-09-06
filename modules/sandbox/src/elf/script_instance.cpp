/**************************************************************************/
/*  script_instance.cpp                                                   */
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

#include "script_instance.h"

#include "../cpp/script_cpp.h"
// #include "../rust/script_rust.h" // Rust support removed
#include "../sandbox_project_settings.h"
#include "../scoped_tree_base.h"
// #include "../zig/script_zig.h" // Zig support not available
#include "core/object/callable_method_pointer.h"
#include "core/object/object.h"
#include "core/templates/local_vector.h"
#include "script_elf.h"
#include "script_instance_helper.h" // register_types.h
static constexpr bool VERBOSE_LOGGING = false;

#ifdef PLATFORM_HAS_EDITOR
static void handle_language_warnings(Array &warnings, const Ref<ELFScript> &script) {
	if (!SandboxProjectSettings::get_docker_enabled()) {
		return;
	}
	const String language = script->get_elf_programming_language();
	if (language == "C++") {
		// Check if the project is a CMake or SCons project and avoid
		// using the Docker container in that case. It's a big download.
		// This detection is cached and returns fast the second time.
		if (CPPScript::DetectCMakeOrSConsProject()) {
			return;
		}
		// Compare C++ version against Docker version
		const int docker_version = CPPScript::DockerContainerVersion();
		if (docker_version < 0) {
			warnings.push_back("C++ Docker container not found");
		} else {
			const int script_version = script->get_elf_api_version();
			if (script_version < docker_version) {
				String w = "C++ API version is newer (" + String::num_int64(script_version) + " vs " + String::num_int64(docker_version) + "), please rebuild the program";
				warnings.push_back(std::move(w));
			}
		}
	}
	// Rust and Zig support commented out - not available in this build
}
#endif

bool ELFScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("ELFScriptInstance::set " + p_name);
	}

	auto [sandbox, created] = get_sandbox();
	if (sandbox) {
		ScopedTreeBase stb(sandbox, ::Object::cast_to<Node>(this->owner));
		return sandbox->set_property(p_name, p_value);
	}

	return false;
}

bool ELFScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
	static const StringName s_script("script");
	if (p_name == s_script) {
		r_ret = script;
		return true;
	}
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("ELFScriptInstance::get " + p_name);
	}

	auto [sandbox, created] = get_sandbox();
	if (sandbox) {
		ScopedTreeBase stb(sandbox, ::Object::cast_to<Node>(this->owner));
		return sandbox->get_property(p_name, r_ret);
	}

	return false;
}

::String ELFScriptInstance::to_string(bool *r_is_valid) {
	return "<ELFScript>";
}

void ELFScriptInstance::notification(int32_t p_what, bool p_reversed) {
}

Variant ELFScriptInstance::callp(
		const StringName &p_method,
		const Variant **p_args, const int p_argument_count,
		Callable::CallError &r_error) {
	if (script.is_null()) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("callp: script is null");
		}
		r_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return Variant();
	}

retry_callp:
	if (script->function_names.has(p_method)) {
		if (current_sandbox && current_sandbox->has_program_loaded()) {
			// Set the Sandbox instance tree base to the owner node
			ScopedTreeBase stb(current_sandbox, ::Object::cast_to<Node>(this->owner));
			// Perform the vmcall
			return current_sandbox->vmcall_fn(p_method, p_args, p_argument_count, r_error);
		}
	}

#ifdef PLATFORM_HAS_EDITOR
	// Handle internal methods
	if (p_method == StringName("_get_editor_name")) {
		r_error.error = Callable::CallError::CALL_OK;
		return Variant("ELFScriptInstance");
	} else if (p_method == StringName("_hide_script_from_inspector")) {
		r_error.error = Callable::CallError::CALL_OK;
		return false;
	} else if (p_method == StringName("_is_read_only")) {
		r_error.error = Callable::CallError::CALL_OK;
		return true;
	} else if (p_method == StringName("_get_configuration_warnings")) {
		// Returns an array of strings with warnings about the script configuration
		Array warnings;
		if (script->function_names.is_empty()) {
			warnings.push_back("No public functions found");
		}
		if (script->get_elf_programming_language() == "Unknown") {
			warnings.push_back("Unknown programming language");
		}
		handle_language_warnings(warnings, script);
		r_error.error = Callable::CallError::CALL_OK;
		return warnings;
	}
#endif

	// When the script instance must have a sandbox as owner,
	// use _enter_tree to get the sandbox instance.
	// Also, avoid calling internal methods.
	if (!this->auto_created_sandbox) {
		if (p_method == StringName("_enter_tree")) {
			current_sandbox->set_program(script);
		}
	}

	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("method called " + p_method);
	}

	// If the program has been loaded, but the method list has not been updated, update it and retry the vmcall
	if (!this->has_updated_methods) {
		if (current_sandbox) {
			// Only update methods if the program is already loaded
			if (current_sandbox->has_program_loaded()) {
				this->update_methods();
				goto retry_callp;
			}
		}
	}

	struct RecursiveTrap {
		bool &recursive_trap;
		RecursiveTrap(bool &trap) :
				recursive_trap(trap) {
			recursive_trap = true; // Set the trap
		}
		~RecursiveTrap() {
			recursive_trap = false; // Reset the trap
		}
	};

	// Try calling a method on the Sandbox instance, but only if the owner is *NOT* a Sandbox.
	// Otherwise, this will clobber the owner Sandbox instance's own methods.
	const CharString method_name = String(p_method).ascii();
	if (current_sandbox != nullptr && !recursive_trap && sandbox_functions.count(method_name.ptr()) != 0) {
		RecursiveTrap trap(this->recursive_trap);
		Array args;
		for (int i = 0; i < p_argument_count; i++) {
			args.push_back(*p_args[i]);
		}
		r_error.error = Callable::CallError::CALL_OK;
		return current_sandbox->callv(p_method, args);
	}
	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	return Variant();
}

void ELFScriptInstance::update_methods() const {
	if (script.is_null()) {
		return;
	}
	this->has_updated_methods = true;
	this->methods_info.clear();

	if (script->functions.is_empty()) {
		// Fallback: Use the function names from the ELFScript
		for (const String &function : script->function_names) {
			MethodInfo method_info(
					Variant::NIL,
					StringName(function));
			this->methods_info.push_back(method_info);
		}
	} else {
		// Create highly specific MethodInfo based on 'functions' Array
		for (int32_t i = 0; i < script->functions.size(); i++) {
			const Dictionary func = script->functions[i].operator Dictionary();
			this->methods_info.push_back(MethodInfo::from_dict(func));
		}
	}
}

void ELFScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
	if (script.is_null()) {
		return;
	}

	if (!this->has_updated_methods) {
		this->update_methods();
	}

	for (const ::MethodInfo &method_info : methods_info) {
		if constexpr (VERBOSE_LOGGING) {
			ERR_PRINT("ELFScriptInstance::get_method_list: method " + String(method_info.name));
		}
		p_list->push_back(method_info);
	}
}

void ELFScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	auto [sandbox, auto_created] = get_sandbox();
	if (!sandbox) {
		if constexpr (VERBOSE_LOGGING) {
			printf("ELFScriptInstance::get_property_list: no sandbox\n");
		}
		return;
	}

	std::vector<PropertyInfo> prop_list;
	if (auto_created) {
		// This is a shared Sandbox instance that won't be able to show any properties
		// in the editor, unless we expose them here.
		prop_list = sandbox->create_sandbox_property_list();
	}

	// Sandboxed properties
	const std::vector<SandboxProperty> &properties = sandbox->get_properties();

	for (const SandboxProperty &property : properties) {
		if constexpr (VERBOSE_LOGGING) {
			printf("ELFScriptInstance::get_property_list %s\n", String(property.name()).utf8().ptr());
			fflush(stdout);
		}
		PropertyInfo prop_info(property.type(), property.name(), PROPERTY_HINT_NONE, "",
				PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_SCRIPT_VARIABLE | PROPERTY_USAGE_NIL_IS_VARIANT);
		p_properties->push_back(prop_info);
	}
	for (const PropertyInfo &prop : prop_list) {
		if constexpr (VERBOSE_LOGGING) {
			printf("ELFScriptInstance::get_property_list %s\n", String(prop.name).utf8().ptr());
			fflush(stdout);
		}
		p_properties->push_back(prop);
	}
}

Variant::Type ELFScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	auto [sandbox, created] = get_sandbox();
	if (sandbox) {
		if (const SandboxProperty *prop = sandbox->find_property_or_null(p_name)) {
			if (r_is_valid) {
				*r_is_valid = true;
			}
			return prop->type();
		}
	}
	if (r_is_valid) {
		*r_is_valid = false;
	}
	return Variant::NIL;
}

void ELFScriptInstance::validate_property(PropertyInfo &p_property) const {
	auto [sandbox, created] = get_sandbox();
	if (!sandbox) {
		if constexpr (VERBOSE_LOGGING) {
			printf("ELFScriptInstance::validate_property: no sandbox\n");
		}
		return;
	}
	for (const SandboxProperty &property : sandbox->get_properties()) {
		if (p_property.name == property.name()) {
			if constexpr (VERBOSE_LOGGING) {
				printf("ELFScriptInstance::validate_property %s => true\n", String(property.name()).utf8().ptr());
			}
			return;
		}
	}
	if constexpr (VERBOSE_LOGGING) {
		printf("ELFScriptInstance::validate_property %s => false\n", String(p_property.name).utf8().ptr());
	}
}

int ELFScriptInstance::get_method_argument_count(const StringName &p_method, bool *r_is_valid) const {
	if (r_is_valid) {
		*r_is_valid = false;
	}
	return 0;
}

bool ELFScriptInstance::has_method(const StringName &p_name) const {
	if (script.is_null()) {
		return true;
	}
	bool result = script->function_names.has(p_name);
	if (!result) {
		for (const StringName &function : godot_functions) {
			if (p_name == function) {
				result = true;
				break;
			}
		}
	}

	if constexpr (VERBOSE_LOGGING) {
		fprintf(stderr, "ELFScriptInstance::has_method %s => %d\n", String(p_name).utf8().ptr(), result);
	}
	return result;
}

bool ELFScriptInstance::property_can_revert(const StringName &p_name) const {
	auto [sandbox, created] = get_sandbox();
	if (!sandbox) {
		return false;
	}
	if (sandbox->find_property_or_null(p_name)) {
		return true;
	}
	const String name = p_name;
	if (name == "references_max" || name == "memory_max" || name == "execution_timeout" || name == "allocations_max" || name == "unboxed_arguments" || name == "precise_simulation"
#ifdef RISCV_LIBTCC
			|| name == "binary_translation_nbit_as" || name == "binary_translation_register_caching"
#endif // RISCV_LIBTCC
			|| name == "profiling" || name == "restrictions") {
		// These are default properties that can be reverted
		return true;
	}
	return false;
}

bool ELFScriptInstance::property_get_revert(const StringName &p_name, Variant &r_ret) const {
	auto [sandbox, created] = get_sandbox();
	if (!sandbox) {
		return false;
	}
	if (const SandboxProperty *prop = sandbox->find_property_or_null(p_name)) {
		r_ret = prop->default_value();
		return true;
	}
	const String name = p_name;
	if (name == "references_max") {
		r_ret = Sandbox::MAX_REFS;
		return true;
	} else if (name == "memory_max") {
		r_ret = Sandbox::MAX_VMEM;
		return true;
	} else if (name == "execution_timeout") {
		r_ret = Sandbox::MAX_INSTRUCTIONS;
		return true;
	} else if (name == "allocations_max") {
		r_ret = Sandbox::MAX_HEAP_ALLOCS;
		return true;
	} else if (name == "unboxed_arguments") {
		r_ret = true;
		return true;
	} else if (name == "precise_simulation") {
		r_ret = false;
		return true;
#ifdef RISCV_LIBTCC
	} else if (name == "binary_translation_nbit_as") {
		r_ret = false;
		return true;
	} else if (name == "binary_translation_register_caching") {
		r_ret = true;
		return true;
#endif // RISCV_LIBTCC
	} else if (name == "profiling") {
		r_ret = false;
		return true;
	} else if (name == "restrictions") {
		r_ret = false;
		return true;
	}
	return false;
}

void ELFScriptInstance::refcount_incremented() {
}

bool ELFScriptInstance::refcount_decremented() {
	return false;
}

Object *ELFScriptInstance::get_owner() {
	return owner;
}

Ref<Script> ELFScriptInstance::get_script() const {
	return script;
}

bool ELFScriptInstance::is_placeholder() const {
	return false;
}

void ELFScriptInstance::property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid) {
	*r_valid = false;
}

Variant ELFScriptInstance::property_get_fallback(const StringName &p_name, bool *r_valid) {
	*r_valid = false;
	return Variant::NIL;
}

ScriptLanguage *ELFScriptInstance::get_language() {
	return get_elf_language_singleton();
}

ELFScriptInstance::ELFScriptInstance(Object *p_owner, const Ref<ELFScript> p_script) :
		owner(p_owner), script(p_script) {
	if (godot_functions.empty()) {
		godot_functions = {
			"_get_editor_name",
			"_hide_script_from_inspector",
			"_is_read_only",
		};
		sandbox_functions = {
			"FromBuffer",
			"FromProgram",
			"load_buffer",
			"reset",
			"vmcall",
			"vmcallv",
			"vmcallable",
			"vmcallable_address",
			"set_restrictions",
			"get_restrictions",
			"add_allowed_object",
			"remove_allowed_object",
			"clear_allowed_objects",
			"set_class_allowed_callback",
			"set_object_allowed_callback",
			"set_method_allowed_callback",
			"set_property_allowed_callback",
			"set_resource_allowed_callback",
			"is_allowed_class",
			"is_allowed_object",
			"is_allowed_method",
			"is_allowed_property",
			"is_allowed_resource",
			"restrictive_callback_function",
			"set_redirect_stdout",
			"get_general_registers",
			"get_floating_point_registers",
			"set_argument_registers",
			"get_current_instruction",
			"make_resumable",
			"resume",
			"has_function",
			"address_of",
			"lookup_address",
			"generate_api",
			"download_program",
			"get_hotspots",
			"clear_hotspots",
			"emit_binary_translation",
			"load_binary_translation",
			"try_compile_binary_translation",
			"is_binary_translated",
			"set_max_refs",
			"get_max_refs",
			"set_memory_max",
			"get_memory_max",
			"set_instructions_max",
			"get_instructions_max",
			"set_allocations_max",
			"get_allocations_max",
			"set_unboxed_arguments",
			"get_unboxed_arguments",
			"set_precise_simulation",
			"get_precise_simulation",
#ifdef RISCV_LIBTCC
			"set_binary_translation_bg_compilation",
			"get_binary_translation_bg_compilation",
			"set_binary_translation_register_caching",
			"get_binary_translation_register_caching",
			"set_binary_translation_nbit_as",
			"get_binary_translation_nbit_as",
#endif // RISCV_LIBTCC
			"share_byte_array",
			"share_float32_array",
			"share_float64_array",
			"share_int32_array",
			"share_int64_array",
			"share_vec2_array",
			"share_vec3_array",
			"share_vec4_array",
			"unshare_array",
			"set_profiling",
			"get_profiling",
			"set_program",
			"get_program",
			"has_program_loaded",
			"get_heap_usage",
			"get_heap_chunk_count",
			"get_heap_allocation_counter",
			"get_heap_deallocation_counter",
			"get_exceptions",
			"get_timeouts",
			"get_calls_made",
			"get_global_calls_made",
			"get_global_exceptions",
			"get_global_timeouts",
			"get_global_instance_count",
			"get_accumulated_startup_time"
		};
	}

	this->current_sandbox = Object::cast_to<Sandbox>(p_owner);
	this->auto_created_sandbox = (this->current_sandbox == nullptr);
	if (auto_created_sandbox) {
		this->current_sandbox = create_sandbox(p_script);
		//ERR_PRINT("ELFScriptInstance: owner is not a Sandbox");
		//fprintf(stderr, "ELFScriptInstance: owner is instead a '%s'!\n", p_owner->get_class().utf8().get_data());
	}
	this->current_sandbox->set_tree_base(::Object::cast_to<::Node>(owner));

	for (const StringName &godot_function : godot_functions) {
		MethodInfo method_info = MethodInfo(
				Variant::NIL,
				godot_function);
		methods_info.push_back(method_info);
	}
}

ELFScriptInstance::~ELFScriptInstance() {
	if (this->script.is_valid()) {
		script->instances.erase(this);
	}
}

// When a Sandbox needs to be automatically created, we instead share it
// across all instances of the same script. This is done to save an
// enormous amount of memory, as each Node using an ELFScriptInstance would
// otherwise have its own Sandbox instance.
static std::unordered_map<ELFScript *, Sandbox *> sandbox_instances;

std::tuple<Sandbox *, bool> ELFScriptInstance::get_sandbox() const {
	auto it = sandbox_instances.find(this->script.ptr());
	if (it != sandbox_instances.end()) {
		return { it->second, true };
	}

	Sandbox *sandbox_ptr = Object::cast_to<Sandbox>(this->owner);
	if (sandbox_ptr != nullptr) {
		return { sandbox_ptr, false };
	}

	ERR_PRINT("ELFScriptInstance: owner is not a Sandbox");
	if constexpr (VERBOSE_LOGGING) {
		fprintf(stderr, "ELFScriptInstance: owner is instead a '%s'!\n", this->owner->get_class().utf8().get_data());
	}
	return { nullptr, false };
}

Sandbox *ELFScriptInstance::create_sandbox(const Ref<ELFScript> &p_script) {
	auto it = sandbox_instances.find(p_script.ptr());
	if (it != sandbox_instances.end()) {
		return it->second;
	}

	Sandbox *sandbox_ptr = memnew(Sandbox);
	sandbox_ptr->set_program(p_script);
	sandbox_instances.insert_or_assign(p_script.ptr(), sandbox_ptr);
	if constexpr (VERBOSE_LOGGING) {
		ERR_PRINT("ELFScriptInstance: created sandbox for " + Object::cast_to<Node>(owner)->get_name());
	}

	return sandbox_ptr;
}
