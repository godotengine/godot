/**************************************************************************/
/*  sandbox.cpp                                                           */
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

#include "sandbox.h"

#include "core/config/engine.h"
#include "core/object/class_db.h"
#include "core/os/time.h"
#include "core/string/print_string.h"
#include "core/variant/variant_utility.h"
#include "elf/script_elf.h"
#include "guest_datatypes.h"
#include "sandbox_project_settings.h"
#include "vmcallable.h"
#include <mutex>
#include <vector>
#if defined(RISCV_BINARY_TRANSLATION) && defined(RISCV_LIBTCC)
#include <future>
#endif

// GuestVariant method implementations are in guest_variant.cpp

static constexpr bool VERBOSE_PROPERTIES = false;
static const int HEAP_SYSCALLS_BASE = 480;
static const int MEMORY_SYSCALLS_BASE = 485;
static const std::vector<std::string> program_arguments = { "program" };
// Thread-safe dummy machine singleton
static machine_t *get_dummy_machine() {
	static std::once_flag dummy_initialized;
	static machine_t *dummy = nullptr;
	std::call_once(dummy_initialized, []() {
		dummy = new machine_t{};
	});
	return dummy;
}

// Cleanup function for dummy machine
static void cleanup_dummy_machine() {
	static std::once_flag cleanup_flag;
	std::call_once(cleanup_flag, []() {
		machine_t *dummy = get_dummy_machine();
		if (dummy) {
			delete dummy;
		}
	});
}
enum SandboxPropertyNameIndex : int {
	PROP_REFERENCES_MAX,
	PROP_MEMORY_MAX,
	PROP_EXECUTION_TIMEOUT,
	PROP_ALLOCATIONS_MAX,
	PROP_UNBOXED_ARGUMENTS,
	PROP_PRECISE_SIMULATION,
#ifdef RISCV_LIBTCC
	PROP_BINTR_NBIT_AS,
	PROP_BINTR_REG_CACHE,
#endif // RISCV_LIBTCC
	PROP_PROFILING,
	PROP_RESTRICTIONS,
	PROP_PROGRAM,
	PROP_MONITOR_HEAP_USAGE,
	PROP_MONITOR_HEAP_CHUNK_COUNT,
	PROP_MONITOR_HEAP_ALLOCATION_COUNTER,
	PROP_MONITOR_HEAP_DEALLOCATION_COUNTER,
	PROP_MONITOR_EXCEPTIONS,
	PROP_MONITOR_EXECUTION_TIMEOUTS,
	PROP_MONITOR_CALLS_MADE,
	PROP_MONITOR_BINARY_TRANSLATED,
	PROP_GLOBAL_CALLS_MADE,
	PROP_GLOBAL_EXCEPTIONS,
	PROP_GLOBAL_TIMEOUTS,
	PROP_MONITOR_ACCUMULATED_STARTUP_TIME,
	PROP_MONITOR_GLOBAL_INSTANCE_COUNT,
};
static std::vector<StringName> property_names;

String Sandbox::_to_string() const {
	return "[ GDExtension::Sandbox <--> Instance ID:" + uitos(get_instance_id()) + " ]";
}

void Sandbox::_bind_methods() {
	// Constructors.
	ClassDB::bind_static_method("Sandbox", D_METHOD("FromBuffer", "buffer"), &Sandbox::FromBuffer);
	ClassDB::bind_static_method("Sandbox", D_METHOD("FromProgram", "program"), &Sandbox::FromProgram);
	// Methods.
	ClassDB::bind_method(D_METHOD("load_buffer", "buffer"), &Sandbox::load_buffer);
	ClassDB::bind_method(D_METHOD("reset", "unload"), &Sandbox::reset, DEFVAL(false));
	{
		MethodInfo mi;
		//mi.arguments.push_back(PropertyInfo(Variant::STRING, "function"));
		mi.name = "vmcall";
		mi.return_val = PropertyInfo(Variant::OBJECT, "result");
		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "vmcall", &Sandbox::vmcall, mi, DEFVAL(Vector<Variant>{}));
		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "vmcallv", &Sandbox::vmcallv, mi, DEFVAL(Vector<Variant>{}));
	}
	ClassDB::bind_method(D_METHOD("vmcallable", "function", "args"), &Sandbox::vmcallable, DEFVAL(Array{}));
	ClassDB::bind_method(D_METHOD("vmcallable_address", "address", "args"), &Sandbox::vmcallable_address, DEFVAL(Array{}));

	// Sandbox restrictions.
	ClassDB::bind_method(D_METHOD("set_restrictions", "restrictions"), &Sandbox::set_restrictions);
	ClassDB::bind_method(D_METHOD("get_restrictions"), &Sandbox::get_restrictions);
	ClassDB::bind_method(D_METHOD("add_allowed_object", "instance"), &Sandbox::add_allowed_object);
	ClassDB::bind_method(D_METHOD("remove_allowed_object", "instance"), &Sandbox::remove_allowed_object);
	ClassDB::bind_method(D_METHOD("clear_allowed_objects"), &Sandbox::clear_allowed_objects);
	ClassDB::bind_method(D_METHOD("set_class_allowed_callback", "instance"), &Sandbox::set_class_allowed_callback);
	ClassDB::bind_method(D_METHOD("set_object_allowed_callback", "instance"), &Sandbox::set_object_allowed_callback);
	ClassDB::bind_method(D_METHOD("set_method_allowed_callback", "instance"), &Sandbox::set_method_allowed_callback);
	ClassDB::bind_method(D_METHOD("set_property_allowed_callback", "instance"), &Sandbox::set_property_allowed_callback);
	ClassDB::bind_method(D_METHOD("set_resource_allowed_callback", "instance"), &Sandbox::set_resource_allowed_callback);
	ClassDB::bind_method(D_METHOD("is_allowed_class", "name"), &Sandbox::is_allowed_class);
	ClassDB::bind_method(D_METHOD("is_allowed_object", "instance"), &Sandbox::is_allowed_object);
	ClassDB::bind_method(D_METHOD("is_allowed_method", "instance", "method"), &Sandbox::is_allowed_method);
	ClassDB::bind_method(D_METHOD("is_allowed_property", "instance", "property", "is_set"), &Sandbox::is_allowed_property, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("is_allowed_resource", "res"), &Sandbox::is_allowed_resource);
	ClassDB::bind_static_method("Sandbox", D_METHOD("restrictive_callback_function", "arg"), &Sandbox::restrictive_callback_function);

	// Internal testing, debugging and introspection.
	ClassDB::bind_method(D_METHOD("set_redirect_stdout", "callback"), &Sandbox::set_redirect_stdout);
	ClassDB::bind_method(D_METHOD("get_general_registers"), &Sandbox::get_general_registers);
	ClassDB::bind_method(D_METHOD("get_floating_point_registers"), &Sandbox::get_floating_point_registers);
	ClassDB::bind_method(D_METHOD("set_argument_registers", "args"), &Sandbox::set_argument_registers);
	ClassDB::bind_method(D_METHOD("get_current_instruction"), &Sandbox::get_current_instruction);
	ClassDB::bind_method(D_METHOD("make_resumable"), &Sandbox::make_resumable);
	ClassDB::bind_method(D_METHOD("resume", "max_instructions"), &Sandbox::resume);

	ClassDB::bind_method(D_METHOD("has_function", "function"), &Sandbox::has_function);
	ClassDB::bind_method(D_METHOD("address_of", "symbol"), &Sandbox::address_of);
	ClassDB::bind_method(D_METHOD("lookup_address", "address"), &Sandbox::lookup_address);
	ClassDB::bind_static_method("Sandbox", D_METHOD("generate_api", "language", "header_extra", "use_argument_names"), &Sandbox::generate_api, DEFVAL("cpp"), DEFVAL(""), DEFVAL(false));
	ClassDB::bind_static_method("Sandbox", D_METHOD("download_program", "program_name"), &Sandbox::download_program, DEFVAL("hello_world"));

	// Shared memory.
	ClassDB::bind_method(D_METHOD("share_byte_array", "allow_write", "array"), &Sandbox::share_byte_array);
	ClassDB::bind_method(D_METHOD("share_float32_array", "allow_write", "array"), &Sandbox::share_float32_array);
	ClassDB::bind_method(D_METHOD("share_float64_array", "allow_write", "array"), &Sandbox::share_float64_array);
	ClassDB::bind_method(D_METHOD("share_int32_array", "allow_write", "array"), &Sandbox::share_int32_array);
	ClassDB::bind_method(D_METHOD("share_int64_array", "allow_write", "array"), &Sandbox::share_int64_array);
	ClassDB::bind_method(D_METHOD("share_vec2_array", "allow_write", "array"), &Sandbox::share_vec2_array);
	ClassDB::bind_method(D_METHOD("share_vec3_array", "allow_write", "array"), &Sandbox::share_vec3_array);
	ClassDB::bind_method(D_METHOD("share_vec4_array", "allow_write", "array"), &Sandbox::share_vec4_array);
	ClassDB::bind_method(D_METHOD("unshare_array", "address"), &Sandbox::unshare_array);

	// Profiling.
	ClassDB::bind_static_method("Sandbox", D_METHOD("get_hotspots", "total", "callable"), &Sandbox::get_hotspots, DEFVAL(6), DEFVAL(Callable()));
	ClassDB::bind_static_method("Sandbox", D_METHOD("clear_hotspots"), &Sandbox::clear_hotspots);

	// Binary translation.
	ClassDB::bind_method(D_METHOD("emit_binary_translation", "ignore_instruction_limit", "automatic_nbit_address_space"), &Sandbox::emit_binary_translation, DEFVAL(false), DEFVAL(false));
	ClassDB::bind_static_method("Sandbox", D_METHOD("load_binary_translation", "shared_library_path", "allow_insecure"), &Sandbox::load_binary_translation, DEFVAL("res://bintr.so"), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("try_compile_binary_translation", "shared_library_path", "compiler", "extra_cflags", "ignore_instruction_limit", "automatic_nbit_as"), &Sandbox::try_compile_binary_translation, DEFVAL("res://bintr"), DEFVAL("cc"), DEFVAL(""), DEFVAL(false), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("is_binary_translated"), &Sandbox::is_binary_translated);
	ClassDB::bind_method(D_METHOD("is_jit"), &Sandbox::is_jit);
	ClassDB::bind_static_method("Sandbox", D_METHOD("set_jit_enabled", "enable"), &Sandbox::set_jit_enabled);
	ClassDB::bind_static_method("Sandbox", D_METHOD("is_jit_enabled"), &Sandbox::is_jit_enabled);
	ClassDB::bind_static_method("Sandbox", D_METHOD("has_feature_jit"), &Sandbox::has_feature_jit);

	// Properties.
	// Note: set, get, and get_property_list are inherited from Node and don't need explicit binding

	ClassDB::bind_method(D_METHOD("set_max_refs", "max"), &Sandbox::set_max_refs, DEFVAL(MAX_REFS));
	ClassDB::bind_method(D_METHOD("get_max_refs"), &Sandbox::get_max_refs);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "references_max", PROPERTY_HINT_NONE, "Maximum objects and variants referenced by a sandbox call"), "set_max_refs", "get_max_refs");

	ClassDB::bind_method(D_METHOD("set_memory_max", "max"), &Sandbox::set_memory_max, DEFVAL(MAX_VMEM));
	ClassDB::bind_method(D_METHOD("get_memory_max"), &Sandbox::get_memory_max);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "memory_max", PROPERTY_HINT_NONE, "Maximum memory (in MiB) used by the sandboxed program"), "set_memory_max", "get_memory_max");

	ClassDB::bind_method(D_METHOD("set_instructions_max", "max"), &Sandbox::set_instructions_max, DEFVAL(MAX_INSTRUCTIONS));
	ClassDB::bind_method(D_METHOD("get_instructions_max"), &Sandbox::get_instructions_max);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "execution_timeout", PROPERTY_HINT_NONE, "Maximum millions of instructions executed before canceling execution"), "set_instructions_max", "get_instructions_max");

	ClassDB::bind_method(D_METHOD("set_allocations_max", "max"), &Sandbox::set_allocations_max, DEFVAL(MAX_HEAP_ALLOCS));
	ClassDB::bind_method(D_METHOD("get_allocations_max"), &Sandbox::get_allocations_max);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "allocations_max", PROPERTY_HINT_NONE, "Maximum number of allocations allowed"), "set_allocations_max", "get_allocations_max");

	ClassDB::bind_method(D_METHOD("set_unboxed_arguments", "unboxed_arguments"), &Sandbox::set_unboxed_arguments);
	ClassDB::bind_method(D_METHOD("get_unboxed_arguments"), &Sandbox::get_unboxed_arguments);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "unboxed_arguments", PROPERTY_HINT_NONE, "Use unboxed arguments for VM function calls"), "set_unboxed_arguments", "get_unboxed_arguments");

	ClassDB::bind_method(D_METHOD("set_precise_simulation", "precise_simulation"), &Sandbox::set_precise_simulation);
	ClassDB::bind_method(D_METHOD("get_precise_simulation"), &Sandbox::get_precise_simulation);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "precise_simulation", PROPERTY_HINT_NONE, "Use precise simulation for VM execution"), "set_precise_simulation", "get_precise_simulation");

	ClassDB::bind_method(D_METHOD("set_binary_translation_nbit_as", "use_nbit_as"), &Sandbox::set_binary_translation_automatic_nbit_as);
	ClassDB::bind_method(D_METHOD("get_binary_translation_nbit_as"), &Sandbox::get_binary_translation_automatic_nbit_as);
#ifdef RISCV_LIBTCC
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "binary_translation_nbit_as", PROPERTY_HINT_NONE, "Use n-bit address space for binary translation"), "set_binary_translation_nbit_as", "get_binary_translation_nbit_as");
#endif // RISCV_LIBTCC

	ClassDB::bind_method(D_METHOD("set_binary_translation_register_caching", "register_caching"), &Sandbox::set_binary_translation_register_caching);
	ClassDB::bind_method(D_METHOD("get_binary_translation_register_caching"), &Sandbox::get_binary_translation_register_caching);
#ifdef RISCV_LIBTCC
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "binary_translation_register_caching", PROPERTY_HINT_NONE, "Use register caching for binary translation"), "set_binary_translation_register_caching", "get_binary_translation_register_caching");
#endif // RISCV_LIBTCC

	ClassDB::bind_method(D_METHOD("set_binary_translation_bg_compilation", "bg_compilation"), &Sandbox::set_binary_translation_bg_compilation);
	ClassDB::bind_method(D_METHOD("get_binary_translation_bg_compilation"), &Sandbox::get_binary_translation_bg_compilation);

	ClassDB::bind_method(D_METHOD("set_profiling", "enable"), &Sandbox::set_profiling, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_profiling"), &Sandbox::get_profiling);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "profiling", PROPERTY_HINT_NONE, "Enable profiling of VM calls"), "set_profiling", "get_profiling");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "restrictions", PROPERTY_HINT_NONE, "Enable sandbox restrictions"), "set_restrictions", "get_restrictions");

	ClassDB::bind_method(D_METHOD("set_program", "program"), &Sandbox::set_program);
	ClassDB::bind_method(D_METHOD("get_program"), &Sandbox::get_program);
	ClassDB::bind_method(D_METHOD("has_program_loaded"), &Sandbox::has_program_loaded);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "program", PROPERTY_HINT_RESOURCE_TYPE, "ELFScript"), "set_program", "get_program");

	// Group for monitored Sandbox health.
	ADD_GROUP("Sandbox Monitoring", "monitor_");

	ClassDB::bind_method(D_METHOD("get_heap_usage"), &Sandbox::get_heap_usage);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_heap_usage", PROPERTY_HINT_NONE, "Current memory arena usage", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_heap_usage");

	ClassDB::bind_method(D_METHOD("get_heap_chunk_count"), &Sandbox::get_heap_chunk_count);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_heap_chunk_count", PROPERTY_HINT_NONE, "Number of memory chunks allocated", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_heap_chunk_count");

	ClassDB::bind_method(D_METHOD("get_heap_allocation_counter"), &Sandbox::get_heap_allocation_counter);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_heap_allocation_counter", PROPERTY_HINT_NONE, "Number of heap allocations", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_heap_allocation_counter");

	ClassDB::bind_method(D_METHOD("get_heap_deallocation_counter"), &Sandbox::get_heap_deallocation_counter);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_heap_deallocation_counter", PROPERTY_HINT_NONE, "Number of heap deallocations", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_heap_deallocation_counter");

	ClassDB::bind_method(D_METHOD("get_exceptions"), &Sandbox::get_exceptions);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_exceptions", PROPERTY_HINT_NONE, "Number of exceptions thrown", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_exceptions");

	ClassDB::bind_method(D_METHOD("get_timeouts"), &Sandbox::get_timeouts);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_execution_timeouts", PROPERTY_HINT_NONE, "Number of execution timeouts", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_timeouts");

	ClassDB::bind_method(D_METHOD("get_calls_made"), &Sandbox::get_calls_made);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_calls_made", PROPERTY_HINT_NONE, "Number of calls made", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_calls_made");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "monitor_binary_translated", PROPERTY_HINT_NONE, "Number of calls made", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "is_binary_translated");

	ClassDB::bind_static_method("Sandbox", D_METHOD("get_global_calls_made"), &Sandbox::get_global_calls_made);
	ClassDB::bind_static_method("Sandbox", D_METHOD("get_global_exceptions"), &Sandbox::get_global_exceptions);
	ClassDB::bind_static_method("Sandbox", D_METHOD("get_global_timeouts"), &Sandbox::get_global_timeouts);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_global_calls_made", PROPERTY_HINT_NONE, "Number of calls made", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_global_calls_made");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_global_exceptions", PROPERTY_HINT_NONE, "Number of exceptions thrown", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_global_exceptions");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_global_execution_timeouts", PROPERTY_HINT_NONE, "Number of execution timeouts", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_global_timeouts");

	ClassDB::bind_static_method("Sandbox", D_METHOD("get_global_instance_count"), &Sandbox::get_global_instance_count);
	ClassDB::bind_static_method("Sandbox", D_METHOD("get_accumulated_startup_time"), &Sandbox::get_accumulated_startup_time);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "monitor_global_instance_count", PROPERTY_HINT_NONE, "Number of active sandbox instances", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_global_instance_count");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "monitor_accumulated_startup_time", PROPERTY_HINT_NONE, "Accumulated startup time of all sandbox instantiations", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY), "", "get_accumulated_startup_time");

	// Group for sandboxed properties.
	ADD_GROUP("Sandboxed Properties", "custom_");
}

std::vector<PropertyInfo> Sandbox::create_sandbox_property_list() {
	std::vector<PropertyInfo> list;
	// Create a list of properties for the Sandbox class only.
	// This is used to expose the basic properties to the editor.

	// Group for sandbox restrictions.
	list.push_back(PropertyInfo(Variant::INT, "references_max", PROPERTY_HINT_NONE));
	list.push_back(PropertyInfo(Variant::INT, "memory_max", PROPERTY_HINT_NONE));
	list.push_back(PropertyInfo(Variant::INT, "execution_timeout", PROPERTY_HINT_NONE));
	list.push_back(PropertyInfo(Variant::INT, "allocations_max", PROPERTY_HINT_NONE));
	list.push_back(PropertyInfo(Variant::BOOL, "unboxed_arguments", PROPERTY_HINT_NONE));
	list.push_back(PropertyInfo(Variant::BOOL, "precise_simulation", PROPERTY_HINT_NONE));
#ifdef RISCV_LIBTCC
	list.push_back(PropertyInfo(Variant::BOOL, "binary_translation_nbit_as", PROPERTY_HINT_NONE));
	list.push_back(PropertyInfo(Variant::BOOL, "binary_translation_register_caching", PROPERTY_HINT_NONE));
#endif // RISCV_LIBTCC
	list.push_back(PropertyInfo(Variant::BOOL, "profiling", PROPERTY_HINT_NONE));
	list.push_back(PropertyInfo(Variant::BOOL, "restrictions", PROPERTY_HINT_NONE));

	// Group for sandbox properties.
	list.push_back(PropertyInfo(Variant::OBJECT, "program", PROPERTY_HINT_RESOURCE_TYPE, "ELFScript"));

	// Group for monitored Sandbox health.
	// Add the group name to the property name to group them in the editor.
	list.push_back(PropertyInfo(Variant::NIL, "Monitoring", PROPERTY_HINT_NONE, "monitor_", PROPERTY_USAGE_GROUP));
	list.push_back(PropertyInfo(Variant::INT, "monitor_heap_usage", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
	list.push_back(PropertyInfo(Variant::INT, "monitor_heap_chunk_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
	list.push_back(PropertyInfo(Variant::INT, "monitor_heap_allocation_counter", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
	list.push_back(PropertyInfo(Variant::INT, "monitor_heap_deallocation_counter", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
	list.push_back(PropertyInfo(Variant::INT, "monitor_exceptions", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
	list.push_back(PropertyInfo(Variant::INT, "monitor_execution_timeouts", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
	list.push_back(PropertyInfo(Variant::INT, "monitor_calls_made", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));
	list.push_back(PropertyInfo(Variant::BOOL, "monitor_binary_translated", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY));

	return list;
}

void Sandbox::constructor_initialize() {
	this->m_current_state = &this->m_states[0];
	this->m_use_unboxed_arguments = SandboxProjectSettings::use_native_types();
	// For each call state, reset the state
	for (size_t i = 0; i < this->m_states.size(); i++) {
		this->m_states[i].reinitialize(i, this->m_max_refs);
	}
}
void Sandbox::reset_machine() {
	try {
		machine_t *dummy = get_dummy_machine();
		if (this->m_machine != dummy) {
			delete this->m_machine;
			this->m_machine = dummy;
		}
	} catch (const std::exception &e) {
		ERR_PRINT(("Sandbox exception: " + std::string(e.what())).c_str());
	}
}
void Sandbox::full_reset() {
	this->reset_machine();
	const bool saved_unboxed_arguments = this->get_unboxed_arguments();
	this->constructor_initialize();
	this->set_unboxed_arguments(saved_unboxed_arguments);

	this->m_properties.clear();
	this->m_lookup.clear();
	this->m_allowed_objects.clear();
}
Sandbox::Sandbox() {
	// Thread-safe initialization of static property names
	static std::once_flag property_names_initialized;
	std::call_once(property_names_initialized, []() {
		property_names = {
			"references_max",
			"memory_max",
			"execution_timeout",
			"allocations_max",
			"unboxed_arguments",
			"precise_simulation",
#ifdef RISCV_LIBTCC
			"binary_translation_nbit_as",
			"binary_translation_register_caching",
#endif // RISCV_LIBTCC
			"profiling",
			"restrictions",
			"program",
			"monitor_heap_usage",
			"monitor_heap_chunk_count",
			"monitor_heap_allocation_counter",
			"monitor_heap_deallocation_counter",
			"monitor_exceptions",
			"monitor_execution_timeouts",
			"monitor_calls_made",
			"monitor_binary_translated",
			"global_calls_made",
			"global_exceptions",
			"global_timeouts",
			"monitor_accumulated_startup_time",
			"monitor_global_instance_count",
		};
	});

	this->constructor_initialize();
	this->m_tree_base = this;
	this->m_global_instances_current += 1;
	this->m_global_instances_seen += 1;
	// In order to reduce checks we guarantee that this
	// class is well-formed at all times.
	this->reset_machine();
}
Sandbox::Sandbox(const PackedByteArray &buffer) :
		Sandbox() {
	this->load_buffer(buffer);
}
Sandbox::Sandbox(Ref<ELFScript> program) :
		Sandbox() {
	this->set_program(program);
}

Sandbox::~Sandbox() {
	if (this->is_in_vmcall()) {
		ERR_PRINT("Sandbox instance destroyed while a VM call is in progress.");
	}
	this->m_global_instances_current -= 1;
	this->set_program_data_internal(nullptr);
	try {
		machine_t *dummy = get_dummy_machine();
		if (this->m_machine != dummy)
			delete this->m_machine;
	} catch (const std::exception &e) {
		ERR_PRINT(("Sandbox exception: " + std::string(e.what())).c_str());
	}
}

void Sandbox::set_memory_max(uint32_t max) {
	m_memory_max = max;
	if (this->has_program_loaded() && !this->is_in_vmcall()) {
		// Reset the machine if the memory limit is changed
		const gaddr_t current_arena = machine().memory.memory_arena_size();
		const gaddr_t new_arena_size = uint64_t(max) << 20;
		if (new_arena_size < current_arena) {
			this->reset();
		}
	}
}

void Sandbox::set_program(Ref<ELFScript> program) {
	// Check if a call is being made from the VM already,
	// which could spell trouble when we now reset the machine.
	if (this->is_in_vmcall()) {
		ERR_PRINT("Cannot load a new program while a VM call is in progress.");
		return;
	}

	// Avoid reloading the same program
	if (program.is_valid() && this->m_program_data == program) {
		if (this->m_source_version == program->get_source_version()) {
			return;
		}
	} else {
		this->m_source_version = -1;
	}

	// Try to retain Sandboxed properties
	std::vector<Variant> property_values;
	property_values.reserve(this->m_properties.size());
	for (const SandboxProperty &prop : this->m_properties) {
		Variant value;
		if (this->get_property(prop.name(), value)) {
			property_values.push_back(value);
		} else {
			property_values.push_back(Variant());
		}
	}
	// Move the properties to a temporary vector (reset coming up)
	std::vector<SandboxProperty> properties = std::move(this->m_properties);

	this->set_program_data_internal(program);
	this->m_program_bytes = {};

	// Unload program and reset the machine
	this->full_reset();

	if (this->m_program_data.is_null())
		return;

	if (this->load(&m_program_data->get_content())) {
		this->m_source_version = m_program_data->get_source_version();
	}

	// Restore Sandboxed properties by comparing the new program's properties
	// with the old ones, then comparing the type. If the property is found,
	// try to set the property with the old value.
	for (const SandboxProperty &old_prop : properties) {
		const Variant *value = nullptr;
		for (const SandboxProperty &new_prop : this->m_properties) {
			if (new_prop.name() == old_prop.name() && new_prop.type() == old_prop.type()) {
				value = &property_values[&old_prop - &properties[0]];
				break;
			}
		}
		if (value) {
			this->set_property(old_prop.name(), *value);
		}
	}
}
void Sandbox::set_program_data_internal(Ref<ELFScript> program) {
	if (this->m_program_data.is_valid()) {
		//printf("Sandbox %p: Program *unset* from %s\n", this, this->m_program_data->get_path().utf8().ptr());
		this->m_program_data->unregister_instance(this);
	}
	this->m_program_data = program;
	if (this->m_program_data.is_valid()) {
		//printf("Sandbox %p: Program set to %s\n", this, this->m_program_data->get_path().utf8().ptr());
		this->m_program_data->register_instance(this);
	}
}
Ref<ELFScript> Sandbox::get_program() {
	return m_program_data;
}
void Sandbox::load_buffer(const PackedByteArray &buffer) {
	// Check if a call is being made from the VM already,
	// which could spell trouble when we now reset the machine.
	if (this->is_in_vmcall()) {
		ERR_PRINT("Cannot load a new program while a VM call is in progress.");
		return;
	}

	this->set_program_data_internal(nullptr);
	this->m_program_bytes = buffer;

	// Reset the machine
	this->full_reset();

	this->load(&this->m_program_bytes);
}
void Sandbox::reset(bool unload) {
	// Check if a call is being made from the VM already,
	// which could spell trouble when we now reset the machine.
	if (this->is_in_vmcall()) {
		ERR_PRINT("Cannot reset the sandbox while a VM call is in progress.");
		return;
	}

	// Allow the program to be reloaded
	this->m_source_version = -1;
	if (unload) {
		this->set_program_data_internal(nullptr);
		this->m_program_bytes = {};
		this->full_reset();
	} else {
		// Reset the machine
		if (this->m_program_data.is_valid()) {
			this->set_program(this->m_program_data);
		} else if (!this->m_program_bytes.is_empty()) {
			this->load_buffer(this->m_program_bytes);
		}
	}
}
bool Sandbox::has_program_loaded() const {
	return !machine().memory.binary().empty();
}
bool Sandbox::load(const PackedByteArray *buffer, const std::vector<std::string> *argv_ptr) {
	if (buffer == nullptr || buffer->is_empty()) {
		ERR_PRINT("Empty binary, cannot load program.");
		this->reset_machine();
		return false;
	}
	const std::string_view binary_view = std::string_view{ (const char *)buffer->ptr(), static_cast<size_t>(buffer->size()) };

	// Get t0 for the startup time
	const uint64_t startup_t0 = Time::get_singleton()->get_ticks_usec();

	/** We can't handle exceptions until the Machine is fully constructed. Two steps.  */
	try {
		// Reset the machine
		machine_t *dummy = get_dummy_machine();
		if (this->m_machine != dummy)
			delete this->m_machine;

		auto options = std::make_shared<riscv::MachineOptions<RISCV_ARCH>>(riscv::MachineOptions<RISCV_ARCH>{
				.memory_max = uint64_t(get_memory_max()) << 20, // in MiB
		//.verbose_loader = true,
#ifdef RISCV_BINARY_TRANSLATION
				.translate_enabled = riscv::libtcc_enabled && m_bintr_jit,
				.translate_enable_embedded = true,
				.translate_future_segments = false,
				.translate_invoke_compiler = riscv::libtcc_enabled && m_bintr_jit,
		//.translate_trace = true,
		//.translate_timing = true,
#ifdef RISCV_LIBTCC
				.translate_ignore_instruction_limit = get_instructions_max() <= 0,
				.translate_use_register_caching = this->m_bintr_register_caching,
				.translate_use_syscall_clobbering_optimization = true,
				.translate_automatic_nbit_address_space = this->m_bintr_automatic_nbit_as,
				.translate_live_patching = false, // Don't meddle with instruction stream
#endif // RISCV_LIBTCC
#endif
		});
#if defined(RISCV_BINARY_TRANSLATION) && defined(RISCV_LIBTCC)
		// Background compilation, if enabled, will run the compilation in a separate thread
		// and live-patch the results into the decoder cache after the compilation is done.
		if (this->m_bintr_bg_compilation) {
			options->translate_background_callback = [](std::function<void()> &callback) {
				// This is called from inside the binary translator in the main thread,
				// and the goal is to run the callback in a separate thread, to avoid
				// blocking the main thread while the compilation step is running.
				std::thread([callback = std::move(callback)]() {
					// Run the callback in a separate thread
					// This is useful for long-running compilation tasks
					// that should not block the main thread.
					// The callback will be called when the compilation is done.
					// Note: This is a no-op if the callback is empty.
					try {
						if (callback)
							callback();
					} catch (const std::exception &e) {
						String what = e.what();
						ERR_PRINT(("Binary translation background compilation exception: " + what));
					}
				}).detach();
			};
		}
#endif

		this->m_machine = new machine_t{ binary_view, *options };
		this->m_machine->set_options(std::move(options));
	} catch (const std::exception &e) {
		ERR_PRINT(("Sandbox construction exception: " + std::string(e.what())).c_str());
		this->m_machine = get_dummy_machine();
		return false;
	}

	/** Now we can process symbols, backtraces etc. */
	try {
		this->m_is_initialization = true;
		machine_t &m = machine();

		m.set_userdata(this);
		m.set_printer([](const machine_t &machine_ref, const char *str, size_t len) {
			Sandbox *sandbox = machine_ref.get_userdata<Sandbox>();
			sandbox->print(String::utf8(str, len));
		});

		this->initialize_syscalls();

		const gaddr_t heap_size = gaddr_t(machine().memory.memory_arena_size() * 0.8) & ~0xFFFLL;
		const gaddr_t heap_area = machine().memory.mmap_allocate(heap_size);

		// Add native system call interfaces
		machine().setup_native_heap(HEAP_SYSCALLS_BASE, heap_area, heap_size);
		machine().setup_native_memory(MEMORY_SYSCALLS_BASE);
		machine().arena().set_max_chunks(get_allocations_max());

		// Set up a Linux environment for the program
		const std::vector<std::string> *argv = argv_ptr ? argv_ptr : &program_arguments;
		m.setup_linux(*argv, { "LC_CTYPE=C", "LC_ALL=C", "TZ=UTC", "LD_LIBRARY_PATH=" });

		// Run the program through to its main() function
		if (!this->m_resumable_mode) {
			if (!this->get_precise_simulation()) {
				if (get_instructions_max() <= 0) {
					m.cpu.simulate_inaccurate(m.cpu.pc());
				} else {
					m.simulate(get_instructions_max() << 20);
				}
			} else {
				// Precise simulation can help discover bugs in the program,
				// as the exact PC address will be known when an exception occurs.
				uint64_t max_instr = get_instructions_max() << 20;
				m.set_max_instructions(max_instr ? max_instr : ~0ULL);
				m.cpu.simulate_precise();
				if (m.instruction_limit_reached()) {
					throw riscv::MachineTimeoutException(riscv::MAX_INSTRUCTIONS_REACHED,
							"Instruction count limit reached", max_instr);
				}
			}
		}
		this->m_is_initialization = false;
	} catch (const std::exception &e) {
		ERR_PRINT(("Sandbox exception: " + std::string(e.what())).c_str());
		this->m_is_initialization = false;
		this->handle_exception(machine().cpu.pc());
	}

	// Read the program's custom properties, if any
	this->read_program_properties(true);

	// Attempt to read the public API functions when an ELF program is loaded
	if (this->m_program_data.is_valid()) {
		// We can't read them without having loaded the program first
		// If the functions Array in the ELFScript object is empty, we will look for the API functions
		if (!this->m_program_data->functions.is_empty()) {
			// Cache the public API functions from the ELFScript object
			for (int i = 0; i < this->m_program_data->functions.size(); i++) {
				const Dictionary func = this->m_program_data->functions[i];
				String name = func["name"];
				const gaddr_t address = func.get("address", 0x0);
				this->m_lookup.insert_or_assign(name.hash(), LookupEntry{ std::move(name), address });
			}
			this->m_program_data->update_public_api_functions();
		}
	}

	// Accumulate startup time
	const uint64_t startup_t1 = Time::get_singleton()->get_ticks_usec();
	double startup_time = (startup_t1 - startup_t0) / 1e6;
	m_accumulated_startup_time += startup_time;
	//fprintf(stderr, "Sandbox startup time: %.3f seconds\n", startup_time);
	return true;
}

Variant Sandbox::vmcall_address(gaddr_t address, const Variant **args, int arg_count, Callable::CallError &error) {
	error.error = Callable::CallError::CALL_OK;
	return this->vmcall_internal(address, args, arg_count);
}
Variant Sandbox::vmcall(const Variant **args, int arg_count, Callable::CallError &error) {
	if (arg_count < 1) {
		error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		error.argument = -1;
		return Variant();
	}

	const Variant &function = *args[0];
	args += 1;
	arg_count -= 1;
	const String function_name = function.operator String();
	const gaddr_t address = cached_address_of(function_name.hash(), function_name);
	if (address == 0) {
		ERR_PRINT("Function not found: " + function_name + " (Added to the public API?)");
		error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		error.argument = 0;
		return Variant();
	}

	error.error = Callable::CallError::CALL_OK;
	return this->vmcall_internal(address, args, arg_count);
}
Variant Sandbox::vmcallv(const Variant **args, int arg_count, Callable::CallError &error) {
	if (arg_count < 1) {
		error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		error.argument = -1;
		return Variant();
	}

	const Variant &function = *args[0];
	args += 1;
	arg_count -= 1;
	const String function_name = function.operator String();
	const gaddr_t address = cached_address_of(function_name.hash(), function_name);
	if (address == 0) {
		ERR_PRINT("Function not found: " + function_name + " (Added to the public API?)");
		error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		error.argument = 0;
		return Variant();
	}

	// Store unboxed_arguments state and restore it after the call
	Variant result;
	auto old_unboxed_arguments = this->get_unboxed_arguments();
	this->set_unboxed_arguments(false);
	result = this->vmcall_internal(address, args, arg_count);
	this->set_unboxed_arguments(old_unboxed_arguments);

	error.error = Callable::CallError::CALL_OK;
	return result;
}
Variant Sandbox::vmcall_fn(const StringName &function_name, const Variant **args, int arg_count, Callable::CallError &error) {
	if (this->m_throttled > 0) {
		this->m_throttled--;
		return Variant();
	}
	// Sandbox.call() is a special case that allows calling functions by name
	if (function_name == StringName("call")) {
		// Redirect to vmcall() with the first argument as the function name
		return this->vmcall(args, arg_count, error);
	}
	const gaddr_t address = cached_address_of(function_name.hash(), function_name);
	if (address == 0) {
		ERR_PRINT("Function not found: " + function_name + " (Added to the public API?)");
		error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
		return Variant();
	}

	Variant result = this->vmcall_internal(address, args, arg_count);
	error.error = Callable::CallError::CALL_OK;
	return result;
}
void Sandbox::setup_arguments_native(gaddr_t arrayDataPtr, GuestVariant *v, const Variant **args, int64_t argc) {
	// In this mode we will try to use registers when possible
	// The stack is already set up from setup_arguments(), so we just need to set up the registers
	machine_t &machine = this->machine();
	int index = 11;
	int flindex = 10;

	for (int64_t i = 0; i < argc; i++) {
		const Variant &arg = *args[i];
		// Get variant data without using deprecated _native_ptr() method
		const GDNativeVariant inner_data = {
			.type = (uint8_t)arg.get_type(),
			.padding = { 0 },
			.value = 0,
			.i64_padding = 0
		};
		const GDNativeVariant *inner = &inner_data;

		// Extract the actual value based on type
		switch (arg.get_type()) {
			case Variant::BOOL:
				const_cast<GDNativeVariant *>(inner)->value = arg.operator bool() ? 1 : 0;
				break;
			case Variant::INT:
				const_cast<GDNativeVariant *>(inner)->value = arg.operator int64_t();
				break;
			case Variant::FLOAT:
				const_cast<GDNativeVariant *>(inner)->flt = arg.operator double();
				break;
			case Variant::VECTOR2: {
				Vector2 v2 = arg.operator Vector2();
				const_cast<GDNativeVariant *>(inner)->vec2_flt[0] = v2.x;
				const_cast<GDNativeVariant *>(inner)->vec2_flt[1] = v2.y;
				break;
			}
			case Variant::VECTOR2I: {
				Vector2i v2i = arg.operator Vector2i();
				const_cast<GDNativeVariant *>(inner)->value = (uint64_t(v2i.y) << 32) | uint32_t(v2i.x);
				break;
			}
			case Variant::VECTOR3: {
				Vector3 v3 = arg.operator Vector3();
				const_cast<GDNativeVariant *>(inner)->vec3_flt[0] = v3.x;
				const_cast<GDNativeVariant *>(inner)->vec3_flt[1] = v3.y;
				const_cast<GDNativeVariant *>(inner)->vec3_flt[2] = v3.z;
				break;
			}
			case Variant::VECTOR3I: {
				Vector3i v3i = arg.operator Vector3i();
				const_cast<GDNativeVariant *>(inner)->ivec3_int[0] = v3i.x;
				const_cast<GDNativeVariant *>(inner)->ivec3_int[1] = v3i.y;
				const_cast<GDNativeVariant *>(inner)->ivec3_int[2] = v3i.z;
				break;
			}
			case Variant::VECTOR4: {
				Vector4 v4 = arg.operator Vector4();
				const_cast<GDNativeVariant *>(inner)->vec4_flt[0] = v4.x;
				const_cast<GDNativeVariant *>(inner)->vec4_flt[1] = v4.y;
				const_cast<GDNativeVariant *>(inner)->vec4_flt[2] = v4.z;
				const_cast<GDNativeVariant *>(inner)->vec4_flt[3] = v4.w;
				break;
			}
			case Variant::VECTOR4I: {
				Vector4i v4i = arg.operator Vector4i();
				const_cast<GDNativeVariant *>(inner)->ivec4_int[0] = v4i.x;
				const_cast<GDNativeVariant *>(inner)->ivec4_int[1] = v4i.y;
				const_cast<GDNativeVariant *>(inner)->ivec4_int[2] = v4i.z;
				const_cast<GDNativeVariant *>(inner)->ivec4_int[3] = v4i.w;
				break;
			}
			case Variant::COLOR: {
				Color color = arg.operator Color();
				const_cast<GDNativeVariant *>(inner)->color_flt[0] = color.r;
				const_cast<GDNativeVariant *>(inner)->color_flt[1] = color.g;
				const_cast<GDNativeVariant *>(inner)->color_flt[2] = color.b;
				const_cast<GDNativeVariant *>(inner)->color_flt[3] = color.a;
				break;
			}
			case Variant::OBJECT:
				const_cast<GDNativeVariant *>(inner)->object_ptr = arg.operator Object *();
				break;
			default:
				break;
		}

		// Incoming arguments are implicitly trusted, as they are provided by the host
		// They also have have the guaranteed lifetime of the function call
		switch (arg.get_type()) {
			case Variant::Type::BOOL:
				machine.cpu.reg(index++) = inner->value;
				break;
			case Variant::Type::INT:
				//printf("Type: %u Value: %ld\n", inner->type, inner->value);
				machine.cpu.reg(index++) = inner->value;
				break;
			case Variant::Type::FLOAT: // Variant floats are always 64-bit
				//printf("Type: %u Value: %f\n", inner->type, inner->flt);
				machine.cpu.registers().getfl(flindex++).set_double(inner->flt);
				break;
			case Variant::VECTOR2: { // 8- or 16-byte structs can be passed in registers
				machine.cpu.registers().getfl(flindex++).set_float(inner->vec2_flt[0]);
				machine.cpu.registers().getfl(flindex++).set_float(inner->vec2_flt[1]);
				break;
			}
			case Variant::VECTOR2I: { // 8- or 16-byte structs can be passed in registers
				machine.cpu.reg(index++) = inner->value; // 64-bit packed integers
				break;
			}
			case Variant::VECTOR3: {
				gaddr_t reg_val1, reg_val2;
				memcpy(&reg_val1, &inner->vec3_flt[0], sizeof(gaddr_t));
				memcpy(&reg_val2, &inner->vec3_flt[2], sizeof(gaddr_t));
				machine.cpu.reg(index++) = reg_val1;
				machine.cpu.reg(index++) = reg_val2;
				break;
			}
			case Variant::VECTOR3I: {
				gaddr_t reg_val1;
				memcpy(&reg_val1, &inner->ivec3_int[0], sizeof(gaddr_t));
				machine.cpu.reg(index++) = reg_val1;
				machine.cpu.reg(index++) = inner->ivec3_int[2];
				break;
			}
			case Variant::VECTOR4: {
				gaddr_t reg_val1, reg_val2;
				memcpy(&reg_val1, &inner->vec4_flt[0], sizeof(gaddr_t));
				memcpy(&reg_val2, &inner->vec4_flt[2], sizeof(gaddr_t));
				machine.cpu.reg(index++) = reg_val1;
				machine.cpu.reg(index++) = reg_val2;
				break;
			}
			case Variant::VECTOR4I: {
				gaddr_t reg_val1, reg_val2;
				memcpy(&reg_val1, &inner->ivec4_int[0], sizeof(gaddr_t));
				memcpy(&reg_val2, &inner->ivec4_int[2], sizeof(gaddr_t));
				machine.cpu.reg(index++) = reg_val1;
				machine.cpu.reg(index++) = reg_val2;
				break;
			}
			case Variant::COLOR: { // 16-byte struct (must use integer registers)
				// RVG calling convention:
				// Unions and arrays containing floats are passed in integer registers
				gaddr_t reg_val1, reg_val2;
				memcpy(&reg_val1, &inner->color_flt[0], sizeof(gaddr_t));
				memcpy(&reg_val2, &inner->color_flt[2], sizeof(gaddr_t));
				machine.cpu.reg(index++) = reg_val1;
				machine.cpu.reg(index++) = reg_val2;
				break;
			}
			case Variant::PLANE: {
				gaddr_t reg_val1, reg_val2;
				memcpy(&reg_val1, &inner->vec4_flt[0], sizeof(gaddr_t));
				memcpy(&reg_val2, &inner->vec4_flt[2], sizeof(gaddr_t));
				machine.cpu.reg(index++) = reg_val1;
				machine.cpu.reg(index++) = reg_val2;
				break;
			}
			case Variant::OBJECT: { // Objects are represented as uintptr_t
				::Object *obj = inner->to_object();
				this->add_scoped_object(obj);
				machine.cpu.reg(index++) = uintptr_t(obj); // Fits in a single register
				break;
			}
			case Variant::ARRAY:
			case Variant::DICTIONARY:
			case Variant::STRING:
			case Variant::STRING_NAME:
			case Variant::NODE_PATH:
			case Variant::RID:
			case Variant::CALLABLE:
			case Variant::TRANSFORM2D:
			case Variant::BASIS:
			case Variant::TRANSFORM3D:
			case Variant::QUATERNION:
			case Variant::PACKED_BYTE_ARRAY:
			case Variant::PACKED_FLOAT32_ARRAY:
			case Variant::PACKED_FLOAT64_ARRAY:
			case Variant::PACKED_INT32_ARRAY:
			case Variant::PACKED_INT64_ARRAY:
			case Variant::PACKED_VECTOR2_ARRAY:
			case Variant::PACKED_VECTOR3_ARRAY:
			case Variant::PACKED_VECTOR4_ARRAY:
			case Variant::PACKED_COLOR_ARRAY:
			case Variant::PACKED_STRING_ARRAY: { // Uses Variant index to reference the object
				unsigned idx = this->add_scoped_variant(&arg);
				machine.cpu.reg(index++) = idx;
				break;
			}
			default: { // Complex types are passed byref, pushed onto the stack as GuestVariant
				GuestVariant &g_arg = v[i + 1];
				g_arg.set(*this, arg, true);
				machine.cpu.reg(index++) = arrayDataPtr + (i + 1) * sizeof(GuestVariant);
			}
		}
	}

	if (UNLIKELY(index > 18 || flindex > 18)) {
		throw std::runtime_error("Sandbox: Too many arguments for VM function call (register overflow)");
	}
}
GuestVariant *Sandbox::setup_arguments(gaddr_t &sp, const Variant **args, int64_t argc) {
	if (this->get_unboxed_arguments()) {
		sp -= sizeof(GuestVariant) * (argc + 1);
		sp &= ~gaddr_t(0xF); // re-align stack pointer
		const gaddr_t arrayDataPtr = sp;
		const int arrayElements = argc + 1;

		GuestVariant *v = m_machine->memory.memarray<GuestVariant>(arrayDataPtr, arrayElements);

		// Set up first argument (return value, also a Variant)
		m_machine->cpu.reg(10) = arrayDataPtr;

		if (argc > 11)
			throw std::runtime_error("Sandbox: Too many arguments for VM function call");
		setup_arguments_native(arrayDataPtr, v, args, argc);
		// A0 is the return value (Variant) of the function
		return &v[0];
	}

	// We will support up to 16 arguments, with the first argument being the return value
	if (argc > 16)
		throw std::runtime_error("Sandbox: Too many arguments for VM function call");

	// The offset to where the first Variant is stored
	// The first argument is the return value, so we start at 1
	// The rest are overflow arguments, which are pushed onto the stack
	const int overflow_args = argc > 7 ? argc - 7 : 0;

	sp -= sizeof(GuestVariant) * (argc + 1) + sizeof(gaddr_t) * overflow_args;
	sp &= ~gaddr_t(0xF); // re-align stack pointer
	const gaddr_t arrayDataPtr = sp + sizeof(gaddr_t) * overflow_args;
	const int arrayElements = argc + 1;

	GuestVariant *v = m_machine->memory.memarray<GuestVariant>(arrayDataPtr, arrayElements);
	gaddr_t *overflow = nullptr;
	if (overflow_args > 0)
		overflow = m_machine->memory.memarray<gaddr_t>(sp, overflow_args);

	// Set up first argument (return value, also a Variant)
	m_machine->cpu.reg(10) = arrayDataPtr + overflow_args * sizeof(GuestVariant);

	for (int64_t i = 0; i < argc; i++) {
		const Variant &arg = *args[i];
		GuestVariant &g_arg = v[1 + i];
		// Fast-path for simple types
		// Extract variant data directly without using deprecated _native_ptr()
		GDNativeVariant inner_data = {
			.type = (uint8_t)arg.get_type(),
			.padding = { 0 },
			.value = 0,
			.i64_padding = 0
		};
		GDNativeVariant *inner = &inner_data;

		// Set the actual value based on type
		switch (arg.get_type()) {
			case Variant::BOOL:
				inner->value = arg.operator bool() ? 1 : 0;
				break;
			case Variant::INT:
				inner->value = arg.operator int64_t();
				break;
			case Variant::FLOAT:
				inner->flt = arg.operator double();
				break;
			case Variant::OBJECT:
				inner->object_ptr = arg.operator Object *();
				break;
			default:
				break;
		}
		// Incoming arguments are implicitly trusted, as they are provided by the host
		// They also have have the guaranteed lifetime of the function call
		switch (arg.get_type()) {
			case Variant::Type::NIL:
				g_arg.type = Variant::Type::NIL;
				break;
			case Variant::Type::BOOL:
				g_arg.type = Variant::Type::BOOL;
				g_arg.v.b = inner->value;
				break;
			case Variant::Type::INT:
				g_arg.type = Variant::Type::INT;
				g_arg.v.i = inner->value;
				break;
			case Variant::Type::FLOAT:
				g_arg.type = Variant::Type::FLOAT;
				g_arg.v.f = inner->flt;
				break;
			case Variant::OBJECT: {
				::Object *obj = inner->to_object();
				// Objects passed directly as arguments are implicitly trusted/allowed
				g_arg.set_object(*this, obj);
				break;
			}
			default:
				g_arg.set(*this, *args[i], true);
		}
		if (i < 7) {
			m_machine->cpu.reg(11 + i) = arrayDataPtr + (1 + i) * sizeof(GuestVariant);
		} else {
			overflow[i - 7] = arrayDataPtr + (1 + i) * sizeof(GuestVariant);
		}
	}
	// A0 is the return value (Variant) of the function
	return &v[overflow_args];
}
Variant Sandbox::vmcall_internal(gaddr_t address, const Variant **args, int argc) {
	this->m_current_state += 1;
	const auto *beginptr = this->m_states.data();
	const auto *endptr = this->m_states.data() + this->m_states.size();
	if (UNLIKELY(this->m_current_state >= endptr)) {
		ERR_PRINT("Too many VM calls in progress");
		this->m_exceptions++;
		this->m_global_exceptions++;
		this->m_current_state -= 1;
		return Variant();
	}

	CurrentState &state = *this->m_current_state;
	const bool is_reentrant_call = (this->m_current_state - beginptr) > 1;
	state.reset();

	// Call statistics
	this->m_calls_made++;
	Sandbox::m_global_calls_made++;

	try {
		GuestVariant *retvar = nullptr;
		riscv::CPU<RISCV_ARCH> &cpu = m_machine->cpu;
		auto &sp = cpu.reg(riscv::REG_SP);
		// execute guest function
		if (!is_reentrant_call) {
			cpu.reg(riscv::REG_RA) = m_machine->memory.exit_address();
			// reset the stack pointer to its initial location
			sp = m_machine->memory.stack_initial();
			// set up each argument, and return value
			retvar = this->setup_arguments(sp, args, argc);
			// execute!
			if (UNLIKELY(this->m_precise_simulation)) {
				m_machine->set_instruction_counter(0);
				uint64_t max_instr = get_instructions_max() << 20;
				m_machine->set_max_instructions(max_instr ? max_instr : ~0ULL);
				m_machine->cpu.jump(address);
				m_machine->cpu.simulate_precise();
				if (m_machine->instruction_limit_reached()) {
					throw riscv::MachineTimeoutException(riscv::MAX_INSTRUCTIONS_REACHED,
							"Instruction count limit reached", max_instr);
				}
			} else if (UNLIKELY(this->get_profiling())) {
				LocalProfilingData &profdata = *this->m_local_profiling_data;
				m_machine->cpu.jump(address);
				do {
					const int32_t next = std::max(int32_t(1), int32_t(profdata.profiling_interval) - int32_t(profdata.profiler_icounter_accumulator));
					m_machine->simulate<false>(next, 0u);
					if (m_machine->instruction_limit_reached()) {
						profdata.profiler_icounter_accumulator = 0;
						profdata.visited.push_back(m_machine->cpu.pc());
					}
				} while (m_machine->instruction_limit_reached());
				// update the accumulator with the remaining instructions
				profdata.profiler_icounter_accumulator += m_machine->instruction_counter();
				if (profdata.profiler_icounter_accumulator >= profdata.profiling_interval) {
					profdata.profiler_icounter_accumulator = 0;
				}
				if (!profdata.visited.empty()) {
					ProfilingData &gprofdata = *this->m_profiling_data;
					// Determine ELF path
					std::string_view path = "";
					if (this->m_program_data.is_valid()) {
						path = this->m_program_data->get_std_path();
					}
					// Update the global profiler
					{
						std::scoped_lock lock(profiling_mutex);
						ProfilingState &gprofstate = gprofdata.state[path];
						// Add all the local known functions to the global state,
						// to aid lookup in the profiler later on
						if (gprofstate.lookup.size() < this->m_lookup.size()) {
							gprofstate.lookup.clear();
							for (const auto &[hash, entry] : this->m_lookup) {
								gprofstate.lookup.push_back(entry);
							}
						}
						// Update the global visited map
						std::unordered_map<gaddr_t, int> &hotspots = gprofstate.hotspots;
						for (const gaddr_t visited_address : profdata.visited) {
							hotspots[visited_address]++;
						}
					}
					profdata.visited.clear();
				}
			} else if (get_instructions_max() <= 0) {
				m_machine->cpu.simulate_inaccurate(address);
			} else {
				m_machine->simulate_with(get_instructions_max() << 20, 0u, address);
			}
		} else {
			riscv::Registers<RISCV_ARCH> regs;
			regs = cpu.registers();
			// we are in a recursive call, so wait before setting exit address
			cpu.reg(riscv::REG_RA) = m_machine->memory.exit_address();
			// we need to make some stack room
			sp -= 16u;
			// set up each argument, and return value
			retvar = this->setup_arguments(sp, args, argc);
			// execute preemption! (precise simulation not supported)
			uint64_t max_instr = get_instructions_max() << 20;
			cpu.preempt_internal(regs, true, true, address, max_instr ? max_instr : ~0ULL);
		}

		// Treat return value as pointer to Variant
		Variant result = retvar->toVariant(*this);
		// Restore the previous state
		this->m_current_state -= 1;
		return result;

	} catch (const std::exception &e) {
		if (Engine::get_singleton()->is_editor_hint()) {
			// Throttle exceptions in the sandbox when calling from the editor
			this->m_throttled += EDITOR_THROTTLE;
		}
		this->handle_exception(address);
		// TODO: Free the function arguments and return value? Will help keep guest memory clean

		this->m_current_state -= 1;
		return Variant();
	}
}
Variant Sandbox::vmcallable(String function, Array args) {
	const gaddr_t address = cached_address_of(function.hash(), function);
	if (address == 0x0) {
		ERR_PRINT("Function not found in the guest: " + function);
		return Variant();
	}

	RiscvCallable *call = memnew(RiscvCallable);
	call->init(this, address, std::move(args));
	return Callable(call);
}
Variant Sandbox::vmcallable_address(gaddr_t address, Array args) {
	RiscvCallable *call = memnew(RiscvCallable);
	call->init(this, address, std::move(args));
	return Callable(call);
}
void RiscvCallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	if (m_varargs_base_count > 0) {
		// We may be receiving extra arguments, so we will fill at the end of m_varargs_ptrs array
		const int total_args = m_varargs_base_count + p_argcount;
		if (size_t(total_args) > m_varargs_ptrs.size()) {
			ERR_PRINT("Too many arguments for VM function call");
			r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
			r_call_error.argument = p_argcount;
			return;
		}

		for (int i = 0; i < p_argcount; i++) {
			m_varargs_ptrs[m_varargs_base_count + i] = p_arguments[i];
		}
		r_return_value = self->vmcall_internal(address, m_varargs_ptrs.data(), total_args);
	} else {
		r_return_value = self->vmcall_internal(address, p_arguments, p_argcount);
	}
	r_call_error.error = Callable::CallError::CALL_OK;
}

gaddr_t Sandbox::cached_address_of(int64_t hash, const String &function) const {
	gaddr_t address = 0x0;
	auto it = m_lookup.find(hash);
	if (it != m_lookup.end()) {
		return it->second.address;
	} else if (m_machine != get_dummy_machine()) {
		const CharString ascii = function.ascii();
		const std::string_view str{ ascii.get_data(), (size_t)ascii.length() };
		address = machine().address_of(str);
		if (address != 0x0) {
			// We tolerate exceptions here, as we are just trying to improve performance
			// If that fails, the function will still work, just a tiny bit slower
			// Some broken linker scripts put a few late functions outside of .text, so we will
			// just have to catch the exception and move on
			try {
				// Cheating a bit here, as we are in a const function
				// But this does not functionally change the Machine, it only boosts performance a bit
				const_cast<machine_t *>(m_machine)->cpu.create_fast_path_function(address);
			} catch (const riscv::MachineException &me) {
				String error = "Sandbox exception: " + String(me.what());
				char buffer[32];
				snprintf(buffer, sizeof(buffer), " (Address 0x%lX)", long(address));
				error += buffer;
				ERR_PRINT(error);
			}
		}
		// Cache the address and symbol name
		LookupEntry entry{ function, address };
		m_lookup.insert_or_assign(hash, std::move(entry));
	}
	return address;
}

gaddr_t Sandbox::address_of(const String &symbol) const {
	const int64_t hash = symbol.hash();
	return cached_address_of(hash, symbol);
}

String Sandbox::lookup_address(gaddr_t address) const {
	for (const auto &entry : m_lookup) {
		if (entry.second.address == address) {
			return entry.second.name;
		}
	}
	riscv::Memory<RISCV_ARCH>::Callsite callsite = machine().memory.lookup(address);
	return String::utf8(callsite.name.c_str(), callsite.name.size());
}

bool Sandbox::has_function(const StringName &p_function) const {
	const gaddr_t address = cached_address_of(p_function.hash(), p_function);
	return address != 0x0;
}

void Sandbox::add_cached_address(const String &name, gaddr_t address) const {
	m_lookup.insert_or_assign(name.hash(), LookupEntry{ name, address });
}

//-- Scoped objects and variants --//

unsigned Sandbox::add_scoped_variant(const Variant *value) const {
	CurrentState &st = this->state();
	if (st.scoped_variants.size() >= st.variants.capacity()) {
		ERR_PRINT("Maximum number of scoped variants reached.");
		throw std::runtime_error("Maximum number of scoped variants reached.");
	}
	st.scoped_variants.push_back(value);
	if (&st != &this->m_states[0])
		return int32_t(st.scoped_variants.size()) - 1;
	else
		return -int32_t(st.scoped_variants.size());
}
unsigned Sandbox::create_scoped_variant(Variant &&value) const {
	CurrentState &st = this->state();
	if (st.scoped_variants.size() >= st.variants.capacity()) {
		ERR_PRINT("Maximum number of scoped variants reached.");
		throw std::runtime_error("Maximum number of scoped variants reached.");
	}
	st.append(std::move(value));
	if (&st != &this->m_states[0])
		return int32_t(st.scoped_variants.size()) - 1;
	else
		return -int32_t(st.scoped_variants.size());
}
std::optional<const Variant *> Sandbox::get_scoped_variant(int32_t index) const noexcept {
	// Add bounds checking to detect memory corruption
	if (index >= 0) {
		// Positive index is access into current state
		const auto &current_state = state();
		if (index < static_cast<int32_t>(current_state.scoped_variants.size())) {
			return current_state.scoped_variants[index];
		}
		// Check for obviously corrupted indices
		if (index > 100000) {
			ERR_PRINT("Detected corrupted scoped variant index: " + itos(index) + " (possible memory corruption)");
		} else {
			ERR_PRINT("Invalid scoped variant index: " + itos(index) + " (max: " + itos(current_state.scoped_variants.size()) + ")");
		}
		return std::nullopt;
	} else {
		// Negative index is access into initialization state
		int32_t perm_index = -index - 1;
		const auto &init_state = this->m_states[0];
		if (perm_index >= 0 && perm_index < static_cast<int32_t>(init_state.scoped_variants.size())) {
			return init_state.scoped_variants[perm_index];
		}
		// Check for obviously corrupted indices
		if (index < -100000) {
			ERR_PRINT("Detected corrupted permanent variant index: " + itos(index) + " (possible memory corruption)");
		} else {
			ERR_PRINT("Invalid permanent variant index: " + itos(perm_index) + " (max: " + itos(init_state.scoped_variants.size()) + ")");
		}
		return std::nullopt;
	}
}
Variant &Sandbox::get_mutable_scoped_variant(int32_t index) {
	std::optional<const Variant *> var_opt = get_scoped_variant(index);
	if (!var_opt.has_value()) {
		ERR_PRINT("Invalid scoped variant index.");
		throw std::runtime_error("Invalid scoped variant index.");
	}
	const Variant *var = var_opt.value();
	// Find the variant in the variants list
	auto it = std::find_if(state().variants.begin(), state().variants.end(), [var](const Variant &v) {
		return &v == var;
	});
	if (it == state().variants.end()) {
		// Create a new variant in the list using the existing one, and return it
		if (state().variants.size() >= state().variants.capacity()) {
			ERR_PRINT("Maximum number of scoped variants reached.");
			throw std::runtime_error("Maximum number of scoped variants reached.");
		}
		state().append(Variant(*var));
		return state().variants.back();
	}
	return *it;
}
unsigned Sandbox::create_permanent_variant(unsigned idx) {
	if (int32_t(idx) < 0) {
		// It's already a permanent variant
		return idx;
	}
	std::optional<const Variant *> var_opt = get_scoped_variant(idx);
	if (!var_opt.has_value()) {
		ERR_PRINT("create_permanent_variant(): Invalid scoped variant index " + itos(idx));
		throw std::runtime_error("Could not make permanent: Invalid scoped variant index " + std::to_string(idx));
	}
	const Variant *var = var_opt.value();
	// Find the variant in the variants list
	auto it = std::find_if(state().variants.begin(), state().variants.end(), [var](const Variant &v) {
		return &v == var;
	});

	CurrentState &perm_state = this->m_states[0];
	if (perm_state.variants.size() >= perm_state.variants.capacity()) {
		ERR_PRINT("Maximum number of scoped variants in permanent state reached.");
		// Just return the old scoped variant
		return idx;
	}

	if (it == state().variants.end()) {
		// Create a new variant in the permanent list
		perm_state.append(var->duplicate());
	} else {
		// Move the variant to the permanent list, leave the old one in the scoped list
		perm_state.append(std::move(*it));
	}
	unsigned perm_idx = perm_state.variants.size() - 1;
	// Return the index of the new permanent variant converted to negative
	return -int32_t(perm_idx) - 1;
}
void Sandbox::assign_permanent_variant(int32_t idx, Variant &&val) {
	if (idx < 0) {
		// It's a permanent variant, verify the index
		idx = -idx - 1;
		if (static_cast<size_t>(idx) < this->m_states[0].variants.size()) {
			this->m_states[0].variants[idx] = std::move(val);
			return;
		}
	}
	// It's either a scoped (temporary) variant, or invalid
	ERR_PRINT("Invalid permanent variant index.");
	throw std::runtime_error("Invalid permanent variant index: " + std::to_string(idx));
}
unsigned Sandbox::try_reuse_assign_variant(int32_t src_idx, const Variant &src_var, int32_t assign_to_idx, const Variant &new_value) {
	if (this->is_permanent_variant(assign_to_idx)) {
		// The Variant is permanent, so we need to assign it directly.
		// Permanent Variants are scarce and should not be duplicated.
		this->assign_permanent_variant(assign_to_idx, Variant(new_value));
		return assign_to_idx;
	} else if (assign_to_idx == src_idx && this->state().is_mutable_variant(src_var)) {
		// They are the same, and the Variant belongs to the current state, so we can modify it directly.
		const_cast<Variant &>(src_var) = new_value;
		return assign_to_idx;
	} else {
		// The Variant is either temporary or invalid, so we can replace it directly.
		return this->create_scoped_variant(Variant(new_value));
	}
}

void Sandbox::add_scoped_object(const void *ptr) {
	if (state().scoped_objects.size() >= this->m_max_refs) {
		ERR_PRINT("Maximum number of scoped objects reached.");
		throw std::runtime_error("Maximum number of scoped objects reached.");
	}
	state().scoped_objects.push_back(reinterpret_cast<uintptr_t>(ptr));
}

//-- Properties --//

void Sandbox::read_program_properties(bool editor) const {
	gaddr_t prop_addr = 0x0;
	try {
		// Properties is an array named properties, that ends with an invalid property
		prop_addr = machine().address_of("properties");
		if (prop_addr == 0x0)
			return;
	} catch (...) {
		return;
	}
	try {
		struct GuestProperty {
			gaddr_t g_name;
			unsigned size;
			Variant::Type type;
			gaddr_t getter;
			gaddr_t setter;
			GuestVariant def_val;
		};
		auto *props = machine().memory.memarray<GuestProperty>(prop_addr, MAX_PROPERTIES);

		for (unsigned int i = 0; i < MAX_PROPERTIES; i++) {
			const GuestProperty *prop = &props[i];
			// Invalid property: stop reading
			if (prop->g_name == 0)
				break;
			// Check if the property is valid by checking its size
			if (prop->size != sizeof(GuestProperty)) {
				//ERR_PRINT("Sandbox: Invalid property size");
				break;
			}
			const std::string c_name = machine().memory.memstring(prop->g_name);
			Variant def_val = prop->def_val.toVariant(*this);

			this->add_property(String::utf8(c_name.c_str(), c_name.size()), prop->type, prop->setter, prop->getter, def_val);
		}
	} catch (const std::exception &e) {
		ERR_PRINT("Sandbox exception in " + get_name() + " while reading properties: " + String(e.what()));
	}
}

void Sandbox::add_property(const String &name, Variant::Type vtype, uint64_t setter, uint64_t getter, const Variant &def) const {
	if (setter == 0 || getter == 0) {
		ERR_PRINT("Sandbox: Setter and getter not found for property: " + name);
		return;
	} else if (m_properties.size() >= MAX_PROPERTIES) {
		ERR_PRINT("Sandbox: Maximum number of properties reached");
		return;
	}
	for (const SandboxProperty &prop : m_properties) {
		if (prop.name() == name) {
			// TODO: Allow overriding properties?
			//ERR_PRINT("Sandbox: Property already exists: " + name);
			return;
		}
	}
	m_properties.emplace_back(name, vtype, setter, getter, def);

	// Make the property getter/setter functions visible to address_of and profiling
	this->add_cached_address("set_" + name, getter);
	this->add_cached_address("get_" + name, setter);
}

bool Sandbox::set_property(const StringName &name, const Variant &value) {
	for (SandboxProperty &prop : m_properties) {
		if (prop.name() == name) {
			prop.set(*this, value);
			//ERR_PRINT("Sandbox: SetProperty *found*: " + name);
			return true;
		}
	}
	// Not the most efficient way to do this, but it's (currently) a small list
	if (name == property_names[PROP_REFERENCES_MAX]) {
		set_max_refs(value);
		return true;
	} else if (name == property_names[PROP_MEMORY_MAX]) {
		set_memory_max(value);
		return true;
	} else if (name == property_names[PROP_EXECUTION_TIMEOUT]) {
		set_instructions_max(value);
		return true;
	} else if (name == property_names[PROP_ALLOCATIONS_MAX]) {
		set_allocations_max(value);
		return true;
	} else if (name == property_names[PROP_UNBOXED_ARGUMENTS]) {
		set_unboxed_arguments(value);
		return true;
	} else if (name == property_names[PROP_PRECISE_SIMULATION]) {
		set_precise_simulation(value);
		return true;
#ifdef RISCV_LIBTCC
	} else if (name == property_names[PROP_BINTR_NBIT_AS]) {
		set_binary_translation_automatic_nbit_as(value);
		return true;
	} else if (name == property_names[PROP_BINTR_REG_CACHE]) {
		set_binary_translation_register_caching(value);
		return true;
#endif // RISCV_LIBTCC
	} else if (name == property_names[PROP_PROFILING]) {
		set_profiling(value);
		return true;
	} else if (name == property_names[PROP_RESTRICTIONS]) {
		set_restrictions(value);
		return true;
	} else if (name == property_names[PROP_PROGRAM]) {
		set_program(value);
		return true;
	}
	if constexpr (VERBOSE_PROPERTIES) {
		printf("Sandbox: SetProperty *not found*: %s\n", String(name).utf8().get_data());
	}
	return false;
}

bool Sandbox::get_property(const StringName &name, Variant &r_ret) {
	for (const SandboxProperty &prop : m_properties) {
		if (prop.name() == name) {
			r_ret = prop.get(*this);
			//ERR_PRINT("Sandbox: GetProperty *found*: " + name);
			return true;
		}
	}
	// Not the most efficient way to do this, but it's (currently) a small list
	if (name == property_names[PROP_REFERENCES_MAX]) {
		r_ret = get_max_refs();
		return true;
	} else if (name == property_names[PROP_MEMORY_MAX]) {
		r_ret = get_memory_max();
		return true;
	} else if (name == property_names[PROP_EXECUTION_TIMEOUT]) {
		r_ret = get_instructions_max();
		return true;
	} else if (name == property_names[PROP_ALLOCATIONS_MAX]) {
		r_ret = get_allocations_max();
		return true;
	} else if (name == property_names[PROP_UNBOXED_ARGUMENTS]) {
		r_ret = get_unboxed_arguments();
		return true;
	} else if (name == property_names[PROP_PRECISE_SIMULATION]) {
		r_ret = get_precise_simulation();
		return true;
#ifdef RISCV_LIBTCC
	} else if (name == property_names[PROP_BINTR_NBIT_AS]) {
		r_ret = this->m_bintr_automatic_nbit_as;
		return true;
	} else if (name == property_names[PROP_BINTR_REG_CACHE]) {
		r_ret = this->m_bintr_register_caching;
		return true;
#endif // RISCV_LIBTCC
	} else if (name == property_names[PROP_PROFILING]) {
		r_ret = get_profiling();
		return true;
	} else if (name == property_names[PROP_RESTRICTIONS]) {
		r_ret = get_restrictions();
		return true;
	} else if (name == property_names[PROP_PROGRAM]) {
		r_ret = get_program();
		return true;
	} else if (name == property_names[PROP_MONITOR_HEAP_USAGE]) {
		r_ret = get_heap_usage();
		return true;
	} else if (name == property_names[PROP_MONITOR_HEAP_CHUNK_COUNT]) {
		r_ret = get_heap_chunk_count();
		return true;
	} else if (name == property_names[PROP_MONITOR_HEAP_ALLOCATION_COUNTER]) {
		r_ret = get_heap_allocation_counter();
		return true;
	} else if (name == property_names[PROP_MONITOR_HEAP_DEALLOCATION_COUNTER]) {
		r_ret = get_heap_deallocation_counter();
		return true;
	} else if (name == property_names[PROP_MONITOR_EXCEPTIONS]) {
		r_ret = get_exceptions();
		return true;
	} else if (name == property_names[PROP_MONITOR_EXECUTION_TIMEOUTS]) {
		r_ret = get_timeouts();
		return true;
	} else if (name == property_names[PROP_MONITOR_CALLS_MADE]) {
		r_ret = get_calls_made();
		return true;
	} else if (name == property_names[PROP_MONITOR_BINARY_TRANSLATED]) {
		r_ret = is_binary_translated();
		return true;
	} else if (name == property_names[PROP_GLOBAL_CALLS_MADE]) {
		r_ret = get_global_calls_made();
		return true;
	} else if (name == property_names[PROP_GLOBAL_EXCEPTIONS]) {
		r_ret = get_global_exceptions();
		return true;
	} else if (name == property_names[PROP_GLOBAL_TIMEOUTS]) {
		r_ret = get_global_timeouts();
		return true;
	} else if (name == property_names[PROP_MONITOR_ACCUMULATED_STARTUP_TIME]) {
		r_ret = get_accumulated_startup_time();
		return true;
	} else if (name == property_names[PROP_MONITOR_GLOBAL_INSTANCE_COUNT]) {
		r_ret = get_global_instance_count();
		return true;
	}
	if constexpr (VERBOSE_PROPERTIES) {
		printf("Sandbox: GetProperty *not found*: %s\n", String(name).utf8().get_data());
	}
	return false;
}

const SandboxProperty *Sandbox::find_property_or_null(const StringName &name) const {
	for (const SandboxProperty &prop : m_properties) {
		if (prop.name() == name) {
			return &prop;
		}
	}
	return nullptr;
}

Variant Sandbox::get(const StringName &name) {
	Variant result;
	if (get_property(name, result)) {
		return result;
	}
	// Get as if it's on the underlying Node object
	return Node::get(name);
}

void Sandbox::set(const StringName &name, const Variant &value) {
	if (!set_property(name, value)) {
		// Set as if it's on the underlying Node object
		Node::set(name, value);
	}
}

Array Sandbox::get_property_list() const {
	Array arr;
	// Sandboxed properties
	for (const SandboxProperty &prop : m_properties) {
		Dictionary d;
		d["name"] = prop.name();
		d["type"] = prop.type();
		d["usage"] = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_SCRIPT_VARIABLE;
		arr.push_back(d);
	}
	// Our properties
	for (const PropertyInfo &prop : this->create_sandbox_property_list()) {
		Dictionary d;
		d["name"] = prop.name;
		d["type"] = prop.type;
		d["usage"] = PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_SCRIPT_VARIABLE;
		arr.push_back(d);
	}
	// Node properties
	List<PropertyInfo> node_props;
	Node::get_property_list(&node_props);
	for (const PropertyInfo &prop : node_props) {
		Dictionary d;
		d["name"] = prop.name;
		d["type"] = prop.type;
		d["usage"] = prop.usage;
		arr.push_back(d);
	}
	return arr;
}

void SandboxProperty::set(Sandbox &sandbox, const Variant &value) {
	if (m_setter_address == 0) {
		ERR_PRINT("Sandbox: Setter was invalid for property: " + m_name);
		return;
	}
	const Variant *args[] = { &value };
	// Store unboxed_arguments state and restore it after the call
	// It's much more convenient to use Variant arguments for properties
	auto old_unboxed_arguments = sandbox.get_unboxed_arguments();
	sandbox.set_unboxed_arguments(false);
	sandbox.vmcall_internal(m_setter_address, args, 1);
	sandbox.set_unboxed_arguments(old_unboxed_arguments);
}

Variant SandboxProperty::get(const Sandbox &sandbox) const {
	if (m_getter_address == 0) {
		ERR_PRINT("Sandbox: Getter was invalid for property: " + m_name);
		return Variant();
	}
	return const_cast<Sandbox &>(sandbox).vmcall_internal(m_getter_address, nullptr, 0);
}

void Sandbox::CurrentState::initialize(unsigned level, unsigned max_refs) {
	(void)level;
	this->variants.reserve(max_refs);
}
void Sandbox::CurrentState::reinitialize(unsigned level, unsigned max_refs) {
	(void)level;
	this->variants.reserve(max_refs);
	this->variants.clear();
	this->scoped_objects.clear();
	this->scoped_variants.clear();
}
bool Sandbox::CurrentState::is_mutable_variant(const Variant &var) const {
	// Check if the address of the variant is within the range of the current state std::vector
	const Variant *ptr = &var;
	return ptr >= &variants[0] && ptr < &variants[0] + variants.size();
}

void Sandbox::set_max_refs(uint32_t max) {
	this->m_max_refs = max;
	// If we are not in a call, reset the states
	if (!this->is_in_vmcall()) {
		for (size_t i = 0; i < this->m_states.size(); i++) {
			this->m_states[i].initialize(i, max);
		}
	} else {
		ERR_PRINT("Sandbox: Cannot change max references during a Sandbox call.");
	}
}

void Sandbox::set_allocations_max(int64_t max) {
	this->m_allocations_max = max;
	if (machine().has_arena()) {
		machine().arena().set_max_chunks(max);
	}
}

int64_t Sandbox::get_heap_usage() const {
	if (machine().has_arena()) {
		return machine().arena().bytes_used();
	}
	return 0;
}

int64_t Sandbox::get_heap_chunk_count() const {
	if (machine().has_arena()) {
		return machine().arena().chunks_used();
	}
	return 0;
}

int64_t Sandbox::get_heap_allocation_counter() const {
	if (machine().has_arena()) {
		return machine().arena().allocation_counter();
	}
	return 0;
}

int64_t Sandbox::get_heap_deallocation_counter() const {
	if (machine().has_arena()) {
		return machine().arena().deallocation_counter();
	}
	return 0;
}

void Sandbox::print(const Variant &v) {
	static bool already_been_here = false;
	if (already_been_here) {
		ERR_PRINT("Recursive call to Sandbox::print() detected, ignoring.");
		return;
	}
	already_been_here = true;

	if (this->m_redirect_stdout.is_valid()) {
		// Redirect to a GDScript callback function
		this->m_redirect_stdout.call(v);
	} else {
		// Print to the console
		print_line(v);
	}

	already_been_here = false;
}

void Sandbox::cleanup_static_resources() {
	cleanup_dummy_machine();
}
