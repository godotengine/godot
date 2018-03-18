/*************************************************************************/
/*  gd_mono.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "gd_mono.h"

#include <mono/metadata/exception.h>
#include <mono/metadata/mono-config.h>
#include <mono/metadata/mono-debug.h>
#include <mono/metadata/mono-gc.h>

#include "os/dir_access.h"
#include "os/file_access.h"
#include "os/os.h"
#include "os/thread.h"
#include "project_settings.h"

#include "../csharp_script.h"
#include "../godotsharp_dirs.h"
#include "../utils/path_utils.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_utils.h"

#ifdef TOOLS_ENABLED
#include "../editor/godotsharp_editor.h"
#include "main/main.h"
#endif

void gdmono_unhandled_exception_hook(MonoObject *exc, void *user_data) {

	(void)user_data; // UNUSED

	GDMonoUtils::print_unhandled_exception(exc);
	abort();
}

#ifdef MONO_PRINT_HANDLER_ENABLED
void gdmono_MonoPrintCallback(const char *string, mono_bool is_stdout) {

	if (is_stdout) {
		OS::get_singleton()->print(string);
	} else {
		OS::get_singleton()->printerr(string);
	}
}
#endif

GDMono *GDMono::singleton = NULL;

namespace {

void setup_runtime_main_args() {
	CharString execpath = OS::get_singleton()->get_executable_path().utf8();

	List<String> cmdline_args = OS::get_singleton()->get_cmdline_args();

	List<CharString> cmdline_args_utf8;
	Vector<char *> main_args;
	main_args.resize(cmdline_args.size() + 1);

	main_args[0] = execpath.ptrw();

	int i = 1;
	for (List<String>::Element *E = cmdline_args.front(); E; E = E->next()) {
		CharString &stored = cmdline_args_utf8.push_back(E->get().utf8())->get();
		main_args[i] = stored.ptrw();
		i++;
	}

	mono_runtime_set_main_args(main_args.size(), main_args.ptrw());
}

#ifdef DEBUG_ENABLED

static bool _wait_for_debugger_msecs(uint32_t p_msecs) {

	do {
		if (mono_is_debugger_attached())
			return true;

		int last_tick = OS::get_singleton()->get_ticks_msec();

		OS::get_singleton()->delay_usec((p_msecs < 25 ? p_msecs : 25) * 1000);

		int tdiff = OS::get_singleton()->get_ticks_msec() - last_tick;

		if (tdiff > p_msecs) {
			p_msecs = 0;
		} else {
			p_msecs -= tdiff;
		}
	} while (p_msecs > 0);

	return mono_is_debugger_attached();
}

void gdmono_debug_init() {

	mono_debug_init(MONO_DEBUG_FORMAT_MONO);

	int da_port = GLOBAL_DEF("mono/debugger_agent/port", 23685);
	bool da_suspend = GLOBAL_DEF("mono/debugger_agent/wait_for_debugger", false);
	int da_timeout = GLOBAL_DEF("mono/debugger_agent/wait_timeout", 3000);

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() ||
			ProjectSettings::get_singleton()->get_resource_path().empty() ||
			Main::is_project_manager()) {
		return;
	}
#endif

	CharString da_args = String("--debugger-agent=transport=dt_socket,address=127.0.0.1:" + itos(da_port) +
								",embedding=1,server=y,suspend=" + (da_suspend ? "y,timeout=" + itos(da_timeout) : "n"))
								 .utf8();
	// --debugger-agent=help
	const char *options[] = {
		"--soft-breakpoints",
		da_args.get_data()
	};
	mono_jit_parse_options(2, (char **)options);
}

#endif

} // namespace

void GDMono::initialize() {

	ERR_FAIL_NULL(Engine::get_singleton());

	OS::get_singleton()->print("Mono: Initializing module...\n");

#ifdef DEBUG_METHODS_ENABLED
	_initialize_and_check_api_hashes();
#endif

	GDMonoLog::get_singleton()->initialize();

#ifdef MONO_PRINT_HANDLER_ENABLED
	mono_trace_set_print_handler(gdmono_MonoPrintCallback);
	mono_trace_set_printerr_handler(gdmono_MonoPrintCallback);
#endif

#ifdef WINDOWS_ENABLED
	mono_reg_info = MonoRegUtils::find_mono();

	CharString assembly_dir;
	CharString config_dir;

	if (mono_reg_info.assembly_dir.length() && DirAccess::exists(mono_reg_info.assembly_dir)) {
		assembly_dir = mono_reg_info.assembly_dir.utf8();
	}

	if (mono_reg_info.config_dir.length() && DirAccess::exists(mono_reg_info.config_dir)) {
		config_dir = mono_reg_info.config_dir.utf8();
	}

	mono_set_dirs(assembly_dir.length() ? assembly_dir.get_data() : NULL,
			config_dir.length() ? config_dir.get_data() : NULL);
#else
	mono_set_dirs(NULL, NULL);
#endif

	GDMonoAssembly::initialize();

#ifdef DEBUG_ENABLED
	gdmono_debug_init();
#endif

	mono_config_parse(NULL);

	root_domain = mono_jit_init_version("GodotEngine.RootDomain", "v4.0.30319");

	ERR_EXPLAIN("Mono: Failed to initialize runtime");
	ERR_FAIL_NULL(root_domain);

	GDMonoUtils::set_main_thread(GDMonoUtils::get_current_thread());

	setup_runtime_main_args(); // Required for System.Environment.GetCommandLineArgs

	runtime_initialized = true;

	OS::get_singleton()->print("Mono: Runtime initialized\n");

	// mscorlib assembly MUST be present at initialization
	ERR_EXPLAIN("Mono: Failed to load mscorlib assembly");
	ERR_FAIL_COND(!_load_corlib_assembly());

#ifdef TOOLS_ENABLED
	// The tools domain must be loaded here, before the scripts domain.
	// Otherwise domain unload on the scripts domain will hang indefinitely.

	ERR_EXPLAIN("Mono: Failed to load tools domain");
	ERR_FAIL_COND(_load_tools_domain() != OK);

	// TODO move to editor init callback, and do it lazily when required before editor init (e.g.: bindings generation)
	ERR_EXPLAIN("Mono: Failed to load Editor Tools assembly");
	ERR_FAIL_COND(!_load_editor_tools_assembly());
#endif

	ERR_EXPLAIN("Mono: Failed to load scripts domain");
	ERR_FAIL_COND(_load_scripts_domain() != OK);

#ifdef DEBUG_ENABLED
	bool debugger_attached = _wait_for_debugger_msecs(500);
	if (!debugger_attached && OS::get_singleton()->is_stdout_verbose())
		OS::get_singleton()->printerr("Mono: Debugger wait timeout\n");
#endif

	_register_internal_calls();

	// The following assemblies are not required at initialization
#ifndef MONO_GLUE_DISABLED
	if (_load_api_assemblies()) {
		if (!core_api_assembly_out_of_sync && !editor_api_assembly_out_of_sync && GDMonoUtils::mono_cache.godot_api_cache_updated) {
			// Everything is fine with the api assemblies, load the project assembly
			_load_project_assembly();
		} else {
#ifdef TOOLS_ENABLED
			// The assembly was successfuly loaded, but the full api could not be cached.
			// This is most likely an outdated assembly loaded because of an invalid version in the metadata,
			// so we invalidate the version in the metadata and unload the script domain.

			if (core_api_assembly_out_of_sync) {
				ERR_PRINT("The loaded Core API assembly is out of sync");
				metadata_set_api_assembly_invalidated(APIAssembly::API_CORE, true);
			} else if (!GDMonoUtils::mono_cache.godot_api_cache_updated) {
				ERR_PRINT("The loaded Core API assembly is in sync, but the cache update failed");
				metadata_set_api_assembly_invalidated(APIAssembly::API_CORE, true);
			}

			if (editor_api_assembly_out_of_sync) {
				ERR_PRINT("The loaded Editor API assembly is out of sync");
				metadata_set_api_assembly_invalidated(APIAssembly::API_EDITOR, true);
			}

			OS::get_singleton()->print("Mono: Proceeding to unload scripts domain because of invalid API assemblies\n");

			Error err = _unload_scripts_domain();
			if (err != OK) {
				WARN_PRINT("Mono: Failed to unload scripts domain");
			}
#else
			ERR_PRINT("The loaded API assembly is invalid");
			CRASH_NOW();
#endif
		}
	}
#else
	if (OS::get_singleton()->is_stdout_verbose())
		OS::get_singleton()->print("Mono: Glue disabled, ignoring script assemblies\n");
#endif

	mono_install_unhandled_exception_hook(gdmono_unhandled_exception_hook, NULL);

	OS::get_singleton()->print("Mono: INITIALIZED\n");
}

#ifndef MONO_GLUE_DISABLED
namespace GodotSharpBindings {

uint64_t get_core_api_hash();
#ifdef TOOLS_ENABLED
uint64_t get_editor_api_hash();
#endif // TOOLS_ENABLED
uint32_t get_bindings_version();
uint32_t get_cs_glue_version();

void register_generated_icalls();
} // namespace GodotSharpBindings
#endif

void GDMono::_register_internal_calls() {
#ifndef MONO_GLUE_DISABLED
	GodotSharpBindings::register_generated_icalls();
#endif

#ifdef TOOLS_ENABLED
	GodotSharpBuilds::_register_internal_calls();
#endif
}

#ifdef DEBUG_METHODS_ENABLED
void GDMono::_initialize_and_check_api_hashes() {

	api_core_hash = ClassDB::get_api_hash(ClassDB::API_CORE);

#ifndef MONO_GLUE_DISABLED
	if (api_core_hash != GodotSharpBindings::get_core_api_hash()) {
		ERR_PRINT("Mono: Core API hash mismatch!");
	}
#endif

#ifdef TOOLS_ENABLED
	api_editor_hash = ClassDB::get_api_hash(ClassDB::API_EDITOR);

#ifndef MONO_GLUE_DISABLED
	if (api_editor_hash != GodotSharpBindings::get_editor_api_hash()) {
		ERR_PRINT("Mono: Editor API hash mismatch!");
	}
#endif

#endif // TOOLS_ENABLED
}
#endif // DEBUG_METHODS_ENABLED

void GDMono::add_assembly(uint32_t p_domain_id, GDMonoAssembly *p_assembly) {

	assemblies[p_domain_id][p_assembly->get_name()] = p_assembly;
}

GDMonoAssembly **GDMono::get_loaded_assembly(const String &p_name) {

	MonoDomain *domain = mono_domain_get();
	uint32_t domain_id = domain ? mono_domain_get_id(domain) : 0;
	return assemblies[domain_id].getptr(p_name);
}

bool GDMono::load_assembly(const String &p_name, GDMonoAssembly **r_assembly, bool p_refonly) {

	CRASH_COND(!r_assembly);

	MonoAssemblyName *aname = mono_assembly_name_new(p_name.utf8());
	bool result = load_assembly(p_name, aname, r_assembly, p_refonly);
	mono_assembly_name_free(aname);
	mono_free(aname);

	return result;
}

bool GDMono::load_assembly(const String &p_name, MonoAssemblyName *p_aname, GDMonoAssembly **r_assembly, bool p_refonly) {

	CRASH_COND(!r_assembly);

	if (OS::get_singleton()->is_stdout_verbose())
		OS::get_singleton()->print((String() + "Mono: Loading assembly " + p_name + (p_refonly ? " (refonly)" : "") + "...\n").utf8());

	MonoImageOpenStatus status = MONO_IMAGE_OK;
	MonoAssembly *assembly = mono_assembly_load_full(p_aname, NULL, &status, p_refonly);

	if (!assembly)
		return false;

	ERR_FAIL_COND_V(status != MONO_IMAGE_OK, false);

	uint32_t domain_id = mono_domain_get_id(mono_domain_get());

	GDMonoAssembly **stored_assembly = assemblies[domain_id].getptr(p_name);

	ERR_FAIL_COND_V(stored_assembly == NULL, false);
	ERR_FAIL_COND_V((*stored_assembly)->get_assembly() != assembly, false);

	*r_assembly = *stored_assembly;

	if (OS::get_singleton()->is_stdout_verbose())
		OS::get_singleton()->print(String("Mono: Assembly " + p_name + (p_refonly ? " (refonly)" : "") + " loaded from path: " + (*r_assembly)->get_path() + "\n").utf8());

	return true;
}

APIAssembly::Version APIAssembly::Version::get_from_loaded_assembly(GDMonoAssembly *p_api_assembly, APIAssembly::Type p_api_type) {
	APIAssembly::Version api_assembly_version;

	const char *nativecalls_name = p_api_type == APIAssembly::API_CORE ?
										   BINDINGS_CLASS_NATIVECALLS :
										   BINDINGS_CLASS_NATIVECALLS_EDITOR;

	GDMonoClass *nativecalls_klass = p_api_assembly->get_class(BINDINGS_NAMESPACE, nativecalls_name);

	if (nativecalls_klass) {
		GDMonoField *api_hash_field = nativecalls_klass->get_field("godot_api_hash");
		if (api_hash_field)
			api_assembly_version.godot_api_hash = GDMonoMarshal::unbox<uint64_t>(api_hash_field->get_value(NULL));

		GDMonoField *binds_ver_field = nativecalls_klass->get_field("bindings_version");
		if (binds_ver_field)
			api_assembly_version.bindings_version = GDMonoMarshal::unbox<uint32_t>(binds_ver_field->get_value(NULL));

		GDMonoField *cs_glue_ver_field = nativecalls_klass->get_field("cs_glue_version");
		if (cs_glue_ver_field)
			api_assembly_version.cs_glue_version = GDMonoMarshal::unbox<uint32_t>(cs_glue_ver_field->get_value(NULL));
	}

	return api_assembly_version;
}

String APIAssembly::to_string(APIAssembly::Type p_type) {
	return p_type == APIAssembly::API_CORE ? "API_CORE" : "API_EDITOR";
}

bool GDMono::_load_corlib_assembly() {

	if (corlib_assembly)
		return true;

	bool success = load_assembly("mscorlib", &corlib_assembly);

	if (success)
		GDMonoUtils::update_corlib_cache();

	return success;
}

bool GDMono::_load_core_api_assembly() {

	if (core_api_assembly)
		return true;

#ifdef TOOLS_ENABLED
	if (metadata_is_api_assembly_invalidated(APIAssembly::API_CORE))
		return false;
#endif

	bool success = load_assembly(API_ASSEMBLY_NAME, &core_api_assembly);

	if (success) {
#ifndef MONO_GLUE_DISABLED
		APIAssembly::Version api_assembly_ver = APIAssembly::Version::get_from_loaded_assembly(core_api_assembly, APIAssembly::API_CORE);
		core_api_assembly_out_of_sync = GodotSharpBindings::get_core_api_hash() != api_assembly_ver.godot_api_hash ||
										GodotSharpBindings::get_bindings_version() != api_assembly_ver.bindings_version ||
										GodotSharpBindings::get_cs_glue_version() != api_assembly_ver.cs_glue_version;
#endif
		GDMonoUtils::update_godot_api_cache();
	}

	return success;
}

#ifdef TOOLS_ENABLED
bool GDMono::_load_editor_api_assembly() {

	if (editor_api_assembly)
		return true;

#ifdef TOOLS_ENABLED
	if (metadata_is_api_assembly_invalidated(APIAssembly::API_EDITOR))
		return false;
#endif

	bool success = load_assembly(EDITOR_API_ASSEMBLY_NAME, &editor_api_assembly);

	if (success) {
#ifndef MONO_GLUE_DISABLED
		APIAssembly::Version api_assembly_ver = APIAssembly::Version::get_from_loaded_assembly(editor_api_assembly, APIAssembly::API_EDITOR);
		editor_api_assembly_out_of_sync = GodotSharpBindings::get_editor_api_hash() != api_assembly_ver.godot_api_hash ||
										  GodotSharpBindings::get_bindings_version() != api_assembly_ver.bindings_version ||
										  GodotSharpBindings::get_cs_glue_version() != api_assembly_ver.cs_glue_version;
#endif
	}

	return success;
}
#endif

#ifdef TOOLS_ENABLED
bool GDMono::_load_editor_tools_assembly() {

	if (editor_tools_assembly)
		return true;

	_GDMONO_SCOPE_DOMAIN_(tools_domain)

	return load_assembly(EDITOR_TOOLS_ASSEMBLY_NAME, &editor_tools_assembly);
}
#endif

bool GDMono::_load_project_assembly() {

	if (project_assembly)
		return true;

	String name = ProjectSettings::get_singleton()->get("application/config/name");
	if (name.empty()) {
		name = "UnnamedProject";
	}

	bool success = load_assembly(name, &project_assembly);

	if (success) {
		mono_assembly_set_main(project_assembly->get_assembly());
	} else {
		if (OS::get_singleton()->is_stdout_verbose())
			OS::get_singleton()->printerr("Mono: Failed to load project assembly\n");
	}

	return success;
}

bool GDMono::_load_api_assemblies() {

	if (!_load_core_api_assembly()) {
		if (OS::get_singleton()->is_stdout_verbose())
			OS::get_singleton()->printerr("Mono: Failed to load Core API assembly\n");
		return false;
	} else {
#ifdef TOOLS_ENABLED
		if (!_load_editor_api_assembly()) {
			if (OS::get_singleton()->is_stdout_verbose())
				OS::get_singleton()->printerr("Mono: Failed to load Editor API assembly\n");
			return false;
		}
#endif
	}

	return true;
}

#ifdef TOOLS_ENABLED
String GDMono::_get_api_assembly_metadata_path() {

	return GodotSharpDirs::get_res_metadata_dir().plus_file("api_assemblies.cfg");
}

void GDMono::metadata_set_api_assembly_invalidated(APIAssembly::Type p_api_type, bool p_invalidated) {

	String section = APIAssembly::to_string(p_api_type);
	String path = _get_api_assembly_metadata_path();

	Ref<ConfigFile> metadata;
	metadata.instance();
	metadata->load(path);

	metadata->set_value(section, "invalidated", p_invalidated);

	String assembly_path = GodotSharpDirs::get_res_assemblies_dir()
								   .plus_file(p_api_type == APIAssembly::API_CORE ?
													  API_ASSEMBLY_NAME ".dll" :
													  EDITOR_API_ASSEMBLY_NAME ".dll");

	ERR_FAIL_COND(!FileAccess::exists(assembly_path));

	uint64_t modified_time = FileAccess::get_modified_time(assembly_path);

	metadata->set_value(section, "invalidated_asm_modified_time", String::num_uint64(modified_time));

	String dir = path.get_base_dir();
	if (!DirAccess::exists(dir)) {
		DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		ERR_FAIL_COND(!da);
		Error err = da->make_dir_recursive(ProjectSettings::get_singleton()->globalize_path(dir));
		ERR_FAIL_COND(err != OK);
	}

	Error save_err = metadata->save(path);
	ERR_FAIL_COND(save_err != OK);
}

bool GDMono::metadata_is_api_assembly_invalidated(APIAssembly::Type p_api_type) {

	String section = APIAssembly::to_string(p_api_type);

	Ref<ConfigFile> metadata;
	metadata.instance();
	metadata->load(_get_api_assembly_metadata_path());

	String assembly_path = GodotSharpDirs::get_res_assemblies_dir()
								   .plus_file(p_api_type == APIAssembly::API_CORE ?
													  API_ASSEMBLY_NAME ".dll" :
													  EDITOR_API_ASSEMBLY_NAME ".dll");

	if (!FileAccess::exists(assembly_path))
		return false;

	uint64_t modified_time = FileAccess::get_modified_time(assembly_path);

	uint64_t stored_modified_time = metadata->get_value(section, "invalidated_asm_modified_time", 0);

	return metadata->get_value(section, "invalidated", false) && modified_time <= stored_modified_time;
}
#endif

Error GDMono::_load_scripts_domain() {

	ERR_FAIL_COND_V(scripts_domain != NULL, ERR_BUG);

	if (OS::get_singleton()->is_stdout_verbose()) {
		OS::get_singleton()->print("Mono: Loading scripts domain...\n");
	}

	scripts_domain = GDMonoUtils::create_domain("GodotEngine.ScriptsDomain");

	ERR_EXPLAIN("Mono: Could not create scripts app domain");
	ERR_FAIL_NULL_V(scripts_domain, ERR_CANT_CREATE);

	mono_domain_set(scripts_domain, true);

	return OK;
}

Error GDMono::_unload_scripts_domain() {

	ERR_FAIL_NULL_V(scripts_domain, ERR_BUG);

	if (OS::get_singleton()->is_stdout_verbose()) {
		OS::get_singleton()->print("Mono: Unloading scripts domain...\n");
	}

	_GodotSharp::get_singleton()->_dispose_callback();

	if (mono_domain_get() != root_domain)
		mono_domain_set(root_domain, true);

	mono_gc_collect(mono_gc_max_generation());

	finalizing_scripts_domain = true;
	mono_domain_finalize(scripts_domain, 2000);
	finalizing_scripts_domain = false;

	mono_gc_collect(mono_gc_max_generation());

	_domain_assemblies_cleanup(mono_domain_get_id(scripts_domain));

	core_api_assembly = NULL;
	project_assembly = NULL;
#ifdef TOOLS_ENABLED
	editor_api_assembly = NULL;
#endif

	core_api_assembly_out_of_sync = false;
	editor_api_assembly_out_of_sync = false;

	MonoDomain *domain = scripts_domain;
	scripts_domain = NULL;

	_GodotSharp::get_singleton()->_dispose_callback();

	MonoObject *ex = NULL;
	mono_domain_try_unload(domain, &ex);

	if (ex) {
		ERR_PRINT("Exception thrown when unloading scripts domain:");
		mono_print_unhandled_exception(ex);
		return FAILED;
	}

	return OK;
}

#ifdef TOOLS_ENABLED
Error GDMono::_load_tools_domain() {

	ERR_FAIL_COND_V(tools_domain != NULL, ERR_BUG);

	if (OS::get_singleton()->is_stdout_verbose()) {
		OS::get_singleton()->print("Mono: Loading tools domain...\n");
	}

	tools_domain = GDMonoUtils::create_domain("GodotEngine.ToolsDomain");

	ERR_EXPLAIN("Mono: Could not create tools app domain");
	ERR_FAIL_NULL_V(tools_domain, ERR_CANT_CREATE);

	return OK;
}
#endif

#ifdef TOOLS_ENABLED
Error GDMono::reload_scripts_domain() {

	ERR_FAIL_COND_V(!runtime_initialized, ERR_BUG);

	if (scripts_domain) {
		Error err = _unload_scripts_domain();
		if (err != OK) {
			ERR_PRINT("Mono: Failed to unload scripts domain");
			return err;
		}
	}

	Error err = _load_scripts_domain();
	if (err != OK) {
		ERR_PRINT("Mono: Failed to load scripts domain");
		return err;
	}

#ifndef MONO_GLUE_DISABLED
	if (!_load_api_assemblies()) {
		return ERR_CANT_OPEN;
	}

	if (!core_api_assembly_out_of_sync && !editor_api_assembly_out_of_sync && GDMonoUtils::mono_cache.godot_api_cache_updated) {
		// Everything is fine with the api assemblies, load the project assembly
		_load_project_assembly();
	} else {
		// The assembly was successfuly loaded, but the full api could not be cached.
		// This is most likely an outdated assembly loaded because of an invalid version in the metadata,
		// so we invalidate the version in the metadata and unload the script domain.

		if (core_api_assembly_out_of_sync) {
			metadata_set_api_assembly_invalidated(APIAssembly::API_CORE, true);
		} else if (!GDMonoUtils::mono_cache.godot_api_cache_updated) {
			ERR_PRINT("Core API assembly is in sync, but the cache update failed");
			metadata_set_api_assembly_invalidated(APIAssembly::API_CORE, true);
		}

		if (editor_api_assembly_out_of_sync) {
			metadata_set_api_assembly_invalidated(APIAssembly::API_EDITOR, true);
		}

		Error err = _unload_scripts_domain();
		if (err != OK) {
			WARN_PRINT("Mono: Failed to unload scripts domain");
		}

		return ERR_CANT_RESOLVE;
	}

	if (!_load_project_assembly())
		return ERR_CANT_OPEN;
#else
	if (OS::get_singleton()->is_stdout_verbose())
		OS::get_singleton()->print("Mono: Glue disabled, ignoring script assemblies\n");
#endif

	return OK;
}
#endif

Error GDMono::finalize_and_unload_domain(MonoDomain *p_domain) {

	CRASH_COND(p_domain == NULL);

	String domain_name = mono_domain_get_friendly_name(p_domain);

	if (OS::get_singleton()->is_stdout_verbose()) {
		OS::get_singleton()->print(String("Mono: Unloading domain `" + domain_name + "`...\n").utf8());
	}

	if (mono_domain_get() != root_domain)
		mono_domain_set(root_domain, true);

	mono_gc_collect(mono_gc_max_generation());
	mono_domain_finalize(p_domain, 2000);
	mono_gc_collect(mono_gc_max_generation());

	_domain_assemblies_cleanup(mono_domain_get_id(p_domain));

	MonoObject *ex = NULL;
	mono_domain_try_unload(p_domain, &ex);

	if (ex) {
		ERR_PRINTS("Exception thrown when unloading domain `" + domain_name + "`:");
		mono_print_unhandled_exception(ex);
		return FAILED;
	}

	return OK;
}

GDMonoClass *GDMono::get_class(MonoClass *p_raw_class) {

	MonoImage *image = mono_class_get_image(p_raw_class);

	if (image == corlib_assembly->get_image())
		return corlib_assembly->get_class(p_raw_class);

	uint32_t domain_id = mono_domain_get_id(mono_domain_get());
	HashMap<String, GDMonoAssembly *> &domain_assemblies = assemblies[domain_id];

	const String *k = NULL;
	while ((k = domain_assemblies.next(k))) {
		GDMonoAssembly *assembly = domain_assemblies.get(*k);
		if (assembly->get_image() == image) {
			GDMonoClass *klass = assembly->get_class(p_raw_class);

			if (klass)
				return klass;
		}
	}

	return NULL;
}

void GDMono::_domain_assemblies_cleanup(uint32_t p_domain_id) {

	HashMap<String, GDMonoAssembly *> &domain_assemblies = assemblies[p_domain_id];

	const String *k = NULL;
	while ((k = domain_assemblies.next(k))) {
		memdelete(domain_assemblies.get(*k));
	}

	assemblies.erase(p_domain_id);
}

GDMono::GDMono() {

	singleton = this;

	gdmono_log = memnew(GDMonoLog);

	runtime_initialized = false;
	finalizing_scripts_domain = false;

	root_domain = NULL;
	scripts_domain = NULL;
#ifdef TOOLS_ENABLED
	tools_domain = NULL;
#endif

	core_api_assembly_out_of_sync = false;
	editor_api_assembly_out_of_sync = false;

	corlib_assembly = NULL;
	core_api_assembly = NULL;
	project_assembly = NULL;
#ifdef TOOLS_ENABLED
	editor_api_assembly = NULL;
	editor_tools_assembly = NULL;
#endif

#ifdef DEBUG_METHODS_ENABLED
	api_core_hash = 0;
#ifdef TOOLS_ENABLED
	api_editor_hash = 0;
#endif
#endif
}

GDMono::~GDMono() {

	if (runtime_initialized) {

		if (scripts_domain) {

			Error err = _unload_scripts_domain();
			if (err != OK) {
				WARN_PRINT("Mono: Failed to unload scripts domain");
			}
		}

		const uint32_t *k = NULL;
		while ((k = assemblies.next(k))) {
			HashMap<String, GDMonoAssembly *> &domain_assemblies = assemblies.get(*k);

			const String *kk = NULL;
			while ((kk = domain_assemblies.next(kk))) {
				memdelete(domain_assemblies.get(*kk));
			}
		}
		assemblies.clear();

		GDMonoUtils::clear_cache();

		OS::get_singleton()->print("Mono: Runtime cleanup...\n");

		runtime_initialized = false;
		mono_jit_cleanup(root_domain);
	}

	if (gdmono_log)
		memdelete(gdmono_log);

	singleton = NULL;
}

_GodotSharp *_GodotSharp::singleton = NULL;

void _GodotSharp::_dispose_object(Object *p_object) {

	if (p_object->get_script_instance()) {
		CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(p_object->get_script_instance());
		if (cs_instance) {
			cs_instance->mono_object_disposed();
			return;
		}
	}

	// Unsafe refcount decrement. The managed instance also counts as a reference.
	// See: CSharpLanguage::alloc_instance_binding_data(Object *p_object)
	if (Object::cast_to<Reference>(p_object)->unreference()) {
		memdelete(p_object);
	}
}

void _GodotSharp::_dispose_callback() {

#ifndef NO_THREADS
	queue_mutex->lock();
#endif

	for (List<Object *>::Element *E = obj_delete_queue.front(); E; E = E->next()) {
		_dispose_object(E->get());
	}

	for (List<NodePath *>::Element *E = np_delete_queue.front(); E; E = E->next()) {
		memdelete(E->get());
	}

	for (List<RID *>::Element *E = rid_delete_queue.front(); E; E = E->next()) {
		memdelete(E->get());
	}

	obj_delete_queue.clear();
	np_delete_queue.clear();
	rid_delete_queue.clear();
	queue_empty = true;

#ifndef NO_THREADS
	queue_mutex->unlock();
#endif
}

void _GodotSharp::attach_thread() {

	GDMonoUtils::attach_current_thread();
}

void _GodotSharp::detach_thread() {

	GDMonoUtils::detach_current_thread();
}

bool _GodotSharp::is_finalizing_domain() {

	return GDMono::get_singleton()->is_finalizing_scripts_domain();
}

bool _GodotSharp::is_domain_loaded() {

	return GDMono::get_singleton()->get_scripts_domain() != NULL;
}

#define ENQUEUE_FOR_DISPOSAL(m_queue, m_inst)                                   \
	m_queue.push_back(m_inst);                                                  \
	if (queue_empty) {                                                          \
		queue_empty = false;                                                    \
		if (!is_finalizing_domain()) { /* call_deferred may not be safe here */ \
			call_deferred("_dispose_callback");                                 \
		}                                                                       \
	}

void _GodotSharp::queue_dispose(MonoObject *p_mono_object, Object *p_object) {

	if (GDMonoUtils::is_main_thread() && !GDMono::get_singleton()->is_finalizing_scripts_domain()) {
		_dispose_object(p_object);
	} else {
#ifndef NO_THREADS
		queue_mutex->lock();
#endif

		// This is our last chance to invoke notification predelete (this is being called from the finalizer)
		// We must use the MonoObject* passed by the finalizer, because the weak GC handle target returns NULL at this point
		CSharpInstance *si = CAST_CSHARP_INSTANCE(p_object->get_script_instance());
		if (si) {
			si->call_notification_no_check(p_mono_object, Object::NOTIFICATION_PREDELETE);
		}

		ENQUEUE_FOR_DISPOSAL(obj_delete_queue, p_object);

#ifndef NO_THREADS
		queue_mutex->unlock();
#endif
	}
}

void _GodotSharp::queue_dispose(NodePath *p_node_path) {

	if (GDMonoUtils::is_main_thread() && !GDMono::get_singleton()->is_finalizing_scripts_domain()) {
		memdelete(p_node_path);
	} else {
#ifndef NO_THREADS
		queue_mutex->lock();
#endif

		ENQUEUE_FOR_DISPOSAL(np_delete_queue, p_node_path);

#ifndef NO_THREADS
		queue_mutex->unlock();
#endif
	}
}

void _GodotSharp::queue_dispose(RID *p_rid) {

	if (GDMonoUtils::is_main_thread() && !GDMono::get_singleton()->is_finalizing_scripts_domain()) {
		memdelete(p_rid);
	} else {
#ifndef NO_THREADS
		queue_mutex->lock();
#endif

		ENQUEUE_FOR_DISPOSAL(rid_delete_queue, p_rid);

#ifndef NO_THREADS
		queue_mutex->unlock();
#endif
	}
}

void _GodotSharp::_bind_methods() {

	ClassDB::bind_method(D_METHOD("attach_thread"), &_GodotSharp::attach_thread);
	ClassDB::bind_method(D_METHOD("detach_thread"), &_GodotSharp::detach_thread);

	ClassDB::bind_method(D_METHOD("is_finalizing_domain"), &_GodotSharp::is_finalizing_domain);
	ClassDB::bind_method(D_METHOD("is_domain_loaded"), &_GodotSharp::is_domain_loaded);

	ClassDB::bind_method(D_METHOD("_dispose_callback"), &_GodotSharp::_dispose_callback);
}

_GodotSharp::_GodotSharp() {

	singleton = this;
	queue_empty = true;
#ifndef NO_THREADS
	queue_mutex = Mutex::create();
#endif
}

_GodotSharp::~_GodotSharp() {

	singleton = NULL;

	if (queue_mutex) {
		memdelete(queue_mutex);
	}
}
