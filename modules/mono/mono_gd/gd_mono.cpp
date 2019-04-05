/*************************************************************************/
/*  gd_mono.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include <mono/metadata/profiler.h>

#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/project_settings.h"

#include "../csharp_script.h"
#include "../glue/cs_glue_version.gen.h"
#include "../godotsharp_dirs.h"
#include "../utils/path_utils.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_utils.h"

#ifdef TOOLS_ENABLED
#include "../editor/godotsharp_editor.h"
#include "main/main.h"
#endif

#define OUT_OF_SYNC_ERR_MESSAGE(m_assembly_name) "The assembly '" m_assembly_name "' is out of sync. "                    \
												 "This error is expected if you just upgraded to a newer Godot version. " \
												 "Building the project will update the assembly to the correct version."

GDMono *GDMono::singleton = NULL;

namespace {

void setup_runtime_main_args() {
	CharString execpath = OS::get_singleton()->get_executable_path().utf8();

	List<String> cmdline_args = OS::get_singleton()->get_cmdline_args();

	List<CharString> cmdline_args_utf8;
	Vector<char *> main_args;
	main_args.resize(cmdline_args.size() + 1);

	main_args.write[0] = execpath.ptrw();

	int i = 1;
	for (List<String>::Element *E = cmdline_args.front(); E; E = E->next()) {
		CharString &stored = cmdline_args_utf8.push_back(E->get().utf8())->get();
		main_args.write[i] = stored.ptrw();
		i++;
	}

	mono_runtime_set_main_args(main_args.size(), main_args.ptrw());
}

void gdmono_profiler_init() {
	String profiler_args = GLOBAL_DEF("mono/profiler/args", "log:calls,alloc,sample,output=output.mlpd");
	bool profiler_enabled = GLOBAL_DEF("mono/profiler/enabled", false);
	if (profiler_enabled) {
		mono_profiler_load(profiler_args.utf8());
	}
}

#ifdef DEBUG_ENABLED

static bool _wait_for_debugger_msecs(uint32_t p_msecs) {

	do {
		if (mono_is_debugger_attached())
			return true;

		int last_tick = OS::get_singleton()->get_ticks_msec();

		OS::get_singleton()->delay_usec((p_msecs < 25 ? p_msecs : 25) * 1000);

		uint32_t tdiff = OS::get_singleton()->get_ticks_msec() - last_tick;

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

	CharString da_args = OS::get_singleton()->get_environment("GODOT_MONO_DEBUGGER_AGENT").utf8();

	if (da_args.length() == 0) {
		da_args = String("--debugger-agent=transport=dt_socket,address=127.0.0.1:" + itos(da_port) +
						 ",embedding=1,server=y,suspend=" + (da_suspend ? "y,timeout=" + itos(da_timeout) : "n"))
						  .utf8();
	}

	// --debugger-agent=help
	const char *options[] = {
		"--soft-breakpoints",
		da_args.get_data()
	};
	mono_jit_parse_options(2, (char **)options);
}

#endif

} // namespace

void GDMono::add_mono_shared_libs_dir_to_path() {
	// By default Mono seems to search shared libraries in the following directories:
	// Current working directory, @executable_path@ and PATH
	// The parent directory of the image file (assembly where the dllimport method is declared)
	// @executable_path@/../lib
	// @executable_path@/../Libraries (__MACH__ only)

	// This does not work when embedding Mono unless we use the same directory structure.
	// To fix this we append the directory containing our shared libraries to PATH.

#if defined(WINDOWS_ENABLED) || defined(UNIX_ENABLED)
	String path_var("PATH");
	String path_value = OS::get_singleton()->get_environment(path_var);

#ifdef WINDOWS_ENABLED
	path_value += ';';

	String bundled_bin_dir = GodotSharpDirs::get_data_mono_bin_dir();
#ifdef TOOLS_ENABLED
	if (DirAccess::exists(bundled_bin_dir)) {
		path_value += bundled_bin_dir;
	} else {
		path_value += mono_reg_info.bin_dir;
	}
#else
	if (DirAccess::exists(bundled_bin_dir))
		path_value += bundled_bin_dir;
#endif // TOOLS_ENABLED

#else
	path_value += ':';

	String bundled_lib_dir = GodotSharpDirs::get_data_mono_lib_dir();
	if (DirAccess::exists(bundled_lib_dir)) {
		path_value += bundled_lib_dir;
	} else {
		// TODO: Do we need to add the lib dir when using the system installed Mono on Unix platforms?
	}
#endif // WINDOWS_ENABLED

	OS::get_singleton()->set_environment(path_var, path_value);
#endif // WINDOWS_ENABLED || UNIX_ENABLED
}

void GDMono::initialize() {

	ERR_FAIL_NULL(Engine::get_singleton());

	print_verbose("Mono: Initializing module...");

#ifdef DEBUG_METHODS_ENABLED
	_initialize_and_check_api_hashes();
#endif

	GDMonoLog::get_singleton()->initialize();

	String assembly_rootdir;
	String config_dir;

#ifdef TOOLS_ENABLED
#if defined(WINDOWS_ENABLED)
	mono_reg_info = MonoRegUtils::find_mono();

	if (mono_reg_info.assembly_dir.length() && DirAccess::exists(mono_reg_info.assembly_dir)) {
		assembly_rootdir = mono_reg_info.assembly_dir;
	}

	if (mono_reg_info.config_dir.length() && DirAccess::exists(mono_reg_info.config_dir)) {
		config_dir = mono_reg_info.config_dir;
	}
#elif defined(OSX_ENABLED)
	const char *c_assembly_rootdir = mono_assembly_getrootdir();
	const char *c_config_dir = mono_get_config_dir();

	if (!c_assembly_rootdir || !c_config_dir || !DirAccess::exists(c_assembly_rootdir) || !DirAccess::exists(c_config_dir)) {
		Vector<const char *> locations;
		locations.push_back("/Library/Frameworks/Mono.framework/Versions/Current/");
		locations.push_back("/usr/local/var/homebrew/linked/mono/");

		for (int i = 0; i < locations.size(); i++) {
			String hint_assembly_rootdir = path_join(locations[i], "lib");
			String hint_mscorlib_path = path_join(hint_assembly_rootdir, "mono", "4.5", "mscorlib.dll");
			String hint_config_dir = path_join(locations[i], "etc");

			if (FileAccess::exists(hint_mscorlib_path) && DirAccess::exists(hint_config_dir)) {
				assembly_rootdir = hint_assembly_rootdir;
				config_dir = hint_config_dir;
				break;
			}
		}
	}
#endif
#endif // TOOLS_ENABLED

	String bundled_assembly_rootdir = GodotSharpDirs::get_data_mono_lib_dir();
	String bundled_config_dir = GodotSharpDirs::get_data_mono_etc_dir();

#ifdef TOOLS_ENABLED
	if (DirAccess::exists(bundled_assembly_rootdir) && DirAccess::exists(bundled_config_dir)) {
		assembly_rootdir = bundled_assembly_rootdir;
		config_dir = bundled_config_dir;
	}

#ifdef WINDOWS_ENABLED
	if (assembly_rootdir.empty() || config_dir.empty()) {
		// Assertion: if they are not set, then they weren't found in the registry
		CRASH_COND(mono_reg_info.assembly_dir.length() > 0 || mono_reg_info.config_dir.length() > 0);

		ERR_PRINT("Cannot find Mono in the registry");
	}
#endif // WINDOWS_ENABLED

#else
	// These are always the directories in export templates
	assembly_rootdir = bundled_assembly_rootdir;
	config_dir = bundled_config_dir;
#endif // TOOLS_ENABLED

	// Leak if we call mono_set_dirs more than once
	mono_set_dirs(assembly_rootdir.length() ? assembly_rootdir.utf8().get_data() : NULL,
			config_dir.length() ? config_dir.utf8().get_data() : NULL);

	add_mono_shared_libs_dir_to_path();

	GDMonoAssembly::initialize();

	gdmono_profiler_init();

#ifdef DEBUG_ENABLED
	gdmono_debug_init();
#endif

	mono_config_parse(NULL);

	mono_install_unhandled_exception_hook(&unhandled_exception_hook, NULL);

#ifndef TOOLS_ENABLED
	if (!DirAccess::exists("res://.mono")) {
		// 'res://.mono/' is missing so there is nothing to load. We don't need to initialize mono, but
		// we still do so unless mscorlib is missing (which is the case for projects that don't use C#).

		String mscorlib_fname("mscorlib.dll");

		Vector<String> search_dirs;
		GDMonoAssembly::fill_search_dirs(search_dirs);

		bool found = false;
		for (int i = 0; i < search_dirs.size(); i++) {
			if (FileAccess::exists(search_dirs[i].plus_file(mscorlib_fname))) {
				found = true;
				break;
			}
		}

		if (!found)
			return; // mscorlib is missing, do not initialize mono
	}
#endif

	root_domain = mono_jit_init_version("GodotEngine.RootDomain", "v4.0.30319");

	ERR_EXPLAIN("Mono: Failed to initialize runtime");
	ERR_FAIL_NULL(root_domain);

	GDMonoUtils::set_main_thread(GDMonoUtils::get_current_thread());

	setup_runtime_main_args(); // Required for System.Environment.GetCommandLineArgs

	runtime_initialized = true;

	print_verbose("Mono: Runtime initialized");

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
		print_error("Mono: Debugger wait timeout");
#endif

	_register_internal_calls();

	// The following assemblies are not required at initialization
#ifdef MONO_GLUE_ENABLED
	if (_load_api_assemblies()) {
		// Everything is fine with the api assemblies, load the project assembly
		_load_project_assembly();
	} else {
		if ((core_api_assembly && (core_api_assembly_out_of_sync || !GDMonoUtils::mono_cache.godot_api_cache_updated))
#ifdef TOOLS_ENABLED
				|| (editor_api_assembly && editor_api_assembly_out_of_sync)
#endif
		) {
#ifdef TOOLS_ENABLED
			// The assembly was successfully loaded, but the full api could not be cached.
			// This is most likely an outdated assembly loaded because of an invalid version in the
			// metadata, so we invalidate the version in the metadata and unload the script domain.

			if (core_api_assembly_out_of_sync) {
				ERR_PRINT(OUT_OF_SYNC_ERR_MESSAGE(CORE_API_ASSEMBLY_NAME));
				metadata_set_api_assembly_invalidated(APIAssembly::API_CORE, true);
			} else if (!GDMonoUtils::mono_cache.godot_api_cache_updated) {
				ERR_PRINT("The loaded assembly '" CORE_API_ASSEMBLY_NAME "' is in sync, but the cache update failed");
				metadata_set_api_assembly_invalidated(APIAssembly::API_CORE, true);
			}

			if (editor_api_assembly_out_of_sync) {
				ERR_PRINT(OUT_OF_SYNC_ERR_MESSAGE(EDITOR_API_ASSEMBLY_NAME));
				metadata_set_api_assembly_invalidated(APIAssembly::API_EDITOR, true);
			}

			print_line("Mono: Proceeding to unload scripts domain because of invalid API assemblies.");

			Error err = _unload_scripts_domain();
			if (err != OK) {
				WARN_PRINT("Mono: Failed to unload scripts domain");
			}
#else
			ERR_PRINT("The loaded API assembly is invalid");
			CRASH_NOW();
#endif // TOOLS_ENABLED
		}
	}
#else
	print_verbose("Mono: Glue disabled, ignoring script assemblies.");
#endif // MONO_GLUE_ENABLED

	print_verbose("Mono: INITIALIZED");
}

#ifdef MONO_GLUE_ENABLED
namespace GodotSharpBindings {

uint64_t get_core_api_hash();
#ifdef TOOLS_ENABLED
uint64_t get_editor_api_hash();
#endif
uint32_t get_bindings_version();

void register_generated_icalls();
} // namespace GodotSharpBindings
#endif

void GDMono::_register_internal_calls() {
#ifdef MONO_GLUE_ENABLED
	GodotSharpBindings::register_generated_icalls();
#endif

#ifdef TOOLS_ENABLED
	GodotSharpEditor::register_internal_calls();
#endif
}

void GDMono::_initialize_and_check_api_hashes() {

#ifdef MONO_GLUE_ENABLED
	if (get_api_core_hash() != GodotSharpBindings::get_core_api_hash()) {
		ERR_PRINT("Mono: Core API hash mismatch!");
	}

#ifdef TOOLS_ENABLED
	if (get_api_editor_hash() != GodotSharpBindings::get_editor_api_hash()) {
		ERR_PRINT("Mono: Editor API hash mismatch!");
	}
#endif // TOOLS_ENABLED
#endif // MONO_GLUE_ENABLED
}

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

	print_verbose("Mono: Loading assembly " + p_name + (p_refonly ? " (refonly)" : "") + "...");

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

	print_verbose("Mono: Assembly " + p_name + (p_refonly ? " (refonly)" : "") + " loaded from path: " + (*r_assembly)->get_path());

	return true;
}

bool GDMono::load_assembly_from(const String &p_name, const String &p_path, GDMonoAssembly **r_assembly, bool p_refonly) {

	CRASH_COND(!r_assembly);

	print_verbose("Mono: Loading assembly " + p_name + (p_refonly ? " (refonly)" : "") + "...");

	GDMonoAssembly *assembly = GDMonoAssembly::load_from(p_name, p_path, p_refonly);

	if (!assembly)
		return false;

#ifdef DEBUG_ENABLED
	uint32_t domain_id = mono_domain_get_id(mono_domain_get());
	GDMonoAssembly **stored_assembly = assemblies[domain_id].getptr(p_name);

	ERR_FAIL_COND_V(stored_assembly == NULL, false);
	ERR_FAIL_COND_V(*stored_assembly != assembly, false);
#endif

	*r_assembly = assembly;

	print_verbose("Mono: Assembly " + p_name + (p_refonly ? " (refonly)" : "") + " loaded from path: " + (*r_assembly)->get_path());

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
	if (metadata_is_api_assembly_invalidated(APIAssembly::API_CORE)) {
		print_verbose("Mono: Skipping loading of Core API assembly because it was invalidated");
		return false;
	}
#endif

	String assembly_path = GodotSharpDirs::get_res_assemblies_dir().plus_file(CORE_API_ASSEMBLY_NAME ".dll");

	if (!FileAccess::exists(assembly_path))
		return false;

	bool success = load_assembly_from(CORE_API_ASSEMBLY_NAME,
			assembly_path,
			&core_api_assembly);

	if (success) {
#ifdef MONO_GLUE_ENABLED
		APIAssembly::Version api_assembly_ver = APIAssembly::Version::get_from_loaded_assembly(core_api_assembly, APIAssembly::API_CORE);
		core_api_assembly_out_of_sync = GodotSharpBindings::get_core_api_hash() != api_assembly_ver.godot_api_hash ||
										GodotSharpBindings::get_bindings_version() != api_assembly_ver.bindings_version ||
										CS_GLUE_VERSION != api_assembly_ver.cs_glue_version;
		if (!core_api_assembly_out_of_sync) {
			GDMonoUtils::update_godot_api_cache();

			_install_trace_listener();
		}
#else
		GDMonoUtils::update_godot_api_cache();
#endif
	}

	return success;
}

#ifdef TOOLS_ENABLED
bool GDMono::_load_editor_api_assembly() {

	if (editor_api_assembly)
		return true;

	if (metadata_is_api_assembly_invalidated(APIAssembly::API_EDITOR)) {
		print_verbose("Mono: Skipping loading of Editor API assembly because it was invalidated");
		return false;
	}

	String assembly_path = GodotSharpDirs::get_res_assemblies_dir().plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");

	if (!FileAccess::exists(assembly_path))
		return false;

	bool success = load_assembly_from(EDITOR_API_ASSEMBLY_NAME,
			assembly_path,
			&editor_api_assembly);

	if (success) {
#ifdef MONO_GLUE_ENABLED
		APIAssembly::Version api_assembly_ver = APIAssembly::Version::get_from_loaded_assembly(editor_api_assembly, APIAssembly::API_EDITOR);
		editor_api_assembly_out_of_sync = GodotSharpBindings::get_editor_api_hash() != api_assembly_ver.godot_api_hash ||
										  GodotSharpBindings::get_bindings_version() != api_assembly_ver.bindings_version ||
										  CS_GLUE_VERSION != api_assembly_ver.cs_glue_version;
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

		CSharpLanguage::get_singleton()->project_assembly_loaded();
	} else {
		if (OS::get_singleton()->is_stdout_verbose())
			print_error("Mono: Failed to load project assembly");
	}

	return success;
}

bool GDMono::_load_api_assemblies() {

	if (!_load_core_api_assembly()) {
		if (OS::get_singleton()->is_stdout_verbose())
			print_error("Mono: Failed to load Core API assembly");
		return false;
	}

	if (core_api_assembly_out_of_sync || !GDMonoUtils::mono_cache.godot_api_cache_updated)
		return false;

#ifdef TOOLS_ENABLED
	if (!_load_editor_api_assembly()) {
		if (OS::get_singleton()->is_stdout_verbose())
			print_error("Mono: Failed to load Editor API assembly");
		return false;
	}

	if (editor_api_assembly_out_of_sync)
		return false;
#endif

	return true;
}

void GDMono::_install_trace_listener() {

#ifdef DEBUG_ENABLED
	// Install the trace listener now before the project assembly is loaded
	typedef void (*DebuggingUtils_InstallTraceListener)(MonoObject **);
	MonoException *exc = NULL;
	GDMonoClass *debug_utils = core_api_assembly->get_class(BINDINGS_NAMESPACE, "DebuggingUtils");
	DebuggingUtils_InstallTraceListener install_func =
			(DebuggingUtils_InstallTraceListener)debug_utils->get_method_thunk("InstallTraceListener");
	install_func((MonoObject **)&exc);
	if (exc) {
		ERR_PRINT("Failed to install System.Diagnostics.Trace listener");
		GDMonoUtils::debug_print_unhandled_exception(exc);
	}
#endif
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
													  CORE_API_ASSEMBLY_NAME ".dll" :
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
													  CORE_API_ASSEMBLY_NAME ".dll" :
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

	print_verbose("Mono: Loading scripts domain...");

	scripts_domain = GDMonoUtils::create_domain("GodotEngine.ScriptsDomain");

	ERR_EXPLAIN("Mono: Could not create scripts app domain");
	ERR_FAIL_NULL_V(scripts_domain, ERR_CANT_CREATE);

	mono_domain_set(scripts_domain, true);

	return OK;
}

Error GDMono::_unload_scripts_domain() {

	ERR_FAIL_NULL_V(scripts_domain, ERR_BUG);

	print_verbose("Mono: Unloading scripts domain...");

	if (mono_domain_get() != root_domain)
		mono_domain_set(root_domain, true);

	finalizing_scripts_domain = true;

	if (!mono_domain_finalize(scripts_domain, 2000)) {
		ERR_PRINT("Mono: Domain finalization timeout");
	}

	finalizing_scripts_domain = false;

	mono_gc_collect(mono_gc_max_generation());

	_domain_assemblies_cleanup(mono_domain_get_id(scripts_domain));

	core_api_assembly = NULL;
	project_assembly = NULL;
#ifdef TOOLS_ENABLED
	editor_api_assembly = NULL;
#endif

	core_api_assembly_out_of_sync = false;
#ifdef TOOLS_ENABLED
	editor_api_assembly_out_of_sync = false;
#endif

	MonoDomain *domain = scripts_domain;
	scripts_domain = NULL;

	MonoException *exc = NULL;
	mono_domain_try_unload(domain, (MonoObject **)&exc);

	if (exc) {
		ERR_PRINT("Exception thrown when unloading scripts domain");
		GDMonoUtils::debug_unhandled_exception(exc);
		return FAILED;
	}

	return OK;
}

#ifdef TOOLS_ENABLED
Error GDMono::_load_tools_domain() {

	ERR_FAIL_COND_V(tools_domain != NULL, ERR_BUG);

	print_verbose("Mono: Loading tools domain...");

	tools_domain = GDMonoUtils::create_domain("GodotEngine.ToolsDomain");

	ERR_EXPLAIN("Mono: Could not create tools app domain");
	ERR_FAIL_NULL_V(tools_domain, ERR_CANT_CREATE);

	return OK;
}
#endif

#ifdef GD_MONO_HOT_RELOAD
Error GDMono::reload_scripts_domain() {

	ERR_FAIL_COND_V(!runtime_initialized, ERR_BUG);

	if (scripts_domain) {
		Error err = _unload_scripts_domain();
		if (err != OK) {
			ERR_PRINT("Mono: Failed to unload scripts domain");
			return err;
		}
	}

	CSharpLanguage::get_singleton()->_uninitialize_script_bindings();

	Error err = _load_scripts_domain();
	if (err != OK) {
		ERR_PRINT("Mono: Failed to load scripts domain");
		return err;
	}

#ifdef MONO_GLUE_ENABLED
	if (!_load_api_assemblies()) {
		if ((core_api_assembly && (core_api_assembly_out_of_sync || !GDMonoUtils::mono_cache.godot_api_cache_updated))
#ifdef TOOLS_ENABLED
				|| (editor_api_assembly && editor_api_assembly_out_of_sync)
#endif
		) {
#ifdef TOOLS_ENABLED
			// The assembly was successfully loaded, but the full api could not be cached.
			// This is most likely an outdated assembly loaded because of an invalid version in the
			// metadata, so we invalidate the version in the metadata and unload the script domain.

			if (core_api_assembly_out_of_sync) {
				ERR_PRINT(OUT_OF_SYNC_ERR_MESSAGE(CORE_API_ASSEMBLY_NAME));
				metadata_set_api_assembly_invalidated(APIAssembly::API_CORE, true);
			} else if (!GDMonoUtils::mono_cache.godot_api_cache_updated) {
				ERR_PRINT("The loaded Core API assembly is in sync, but the cache update failed");
				metadata_set_api_assembly_invalidated(APIAssembly::API_CORE, true);
			}

			if (editor_api_assembly_out_of_sync) {
				ERR_PRINT(OUT_OF_SYNC_ERR_MESSAGE(EDITOR_API_ASSEMBLY_NAME));
				metadata_set_api_assembly_invalidated(APIAssembly::API_EDITOR, true);
			}

			err = _unload_scripts_domain();
			if (err != OK) {
				WARN_PRINT("Mono: Failed to unload scripts domain");
			}

			return ERR_CANT_RESOLVE;
#else
			ERR_PRINT("The loaded API assembly is invalid");
			CRASH_NOW();
#endif
		} else {
			return ERR_CANT_OPEN;
		}
	}

	if (!_load_project_assembly()) {
		return ERR_CANT_OPEN;
	}
#else
	print_verbose("Mono: Glue disabled, ignoring script assemblies.");
#endif // MONO_GLUE_ENABLED

	return OK;
}
#endif

Error GDMono::finalize_and_unload_domain(MonoDomain *p_domain) {

	CRASH_COND(p_domain == NULL);

	String domain_name = mono_domain_get_friendly_name(p_domain);

	print_verbose("Mono: Unloading domain `" + domain_name + "`...");

	if (mono_domain_get() == p_domain)
		mono_domain_set(root_domain, true);

	if (!mono_domain_finalize(p_domain, 2000)) {
		ERR_PRINT("Mono: Domain finalization timeout");
	}

	mono_gc_collect(mono_gc_max_generation());

	_domain_assemblies_cleanup(mono_domain_get_id(p_domain));

#ifdef TOOLS_ENABLED
	if (p_domain == tools_domain) {
		editor_tools_assembly = NULL;
	}
#endif

	MonoException *exc = NULL;
	mono_domain_try_unload(p_domain, (MonoObject **)&exc);

	if (exc) {
		ERR_PRINTS("Exception thrown when unloading domain `" + domain_name + "`");
		GDMonoUtils::debug_unhandled_exception(exc);
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

void GDMono::unhandled_exception_hook(MonoObject *p_exc, void *) {

	// This method will be called by the runtime when a thrown exception is not handled.
	// It won't be called when we manually treat a thrown exception as unhandled.
	// We assume the exception was already printed before calling this hook.

#ifdef DEBUG_ENABLED
	GDMonoUtils::debug_send_unhandled_exception_error((MonoException *)p_exc);
	if (ScriptDebugger::get_singleton())
		ScriptDebugger::get_singleton()->idle_poll();
#endif
	abort();
	GD_UNREACHABLE();
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
#ifdef TOOLS_ENABLED
	editor_api_assembly_out_of_sync = false;
#endif

	corlib_assembly = NULL;
	core_api_assembly = NULL;
	project_assembly = NULL;
#ifdef TOOLS_ENABLED
	editor_api_assembly = NULL;
	editor_tools_assembly = NULL;
#endif

	api_core_hash = 0;
#ifdef TOOLS_ENABLED
	api_editor_hash = 0;
#endif
}

GDMono::~GDMono() {

	if (is_runtime_initialized()) {

#ifdef TOOLS_ENABLED
		if (tools_domain) {
			Error err = finalize_and_unload_domain(tools_domain);
			if (err != OK) {
				ERR_PRINT("Mono: Failed to unload tools domain");
			}
		}
#endif

		if (scripts_domain) {
			Error err = _unload_scripts_domain();
			if (err != OK) {
				ERR_PRINT("Mono: Failed to unload scripts domain");
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

		print_verbose("Mono: Runtime cleanup...");

		mono_jit_cleanup(root_domain);

		print_verbose("Mono: Finalized");

		runtime_initialized = false;
	}

	if (gdmono_log)
		memdelete(gdmono_log);

	singleton = NULL;
}

_GodotSharp *_GodotSharp::singleton = NULL;

void _GodotSharp::attach_thread() {

	GDMonoUtils::attach_current_thread();
}

void _GodotSharp::detach_thread() {

	GDMonoUtils::detach_current_thread();
}

int32_t _GodotSharp::get_domain_id() {

	MonoDomain *domain = mono_domain_get();
	CRASH_COND(!domain); // User must check if runtime is initialized before calling this method
	return mono_domain_get_id(domain);
}

int32_t _GodotSharp::get_scripts_domain_id() {

	MonoDomain *domain = SCRIPTS_DOMAIN;
	CRASH_COND(!domain); // User must check if scripts domain is loaded before calling this method
	return mono_domain_get_id(domain);
}

bool _GodotSharp::is_scripts_domain_loaded() {

	return GDMono::get_singleton()->is_runtime_initialized() && SCRIPTS_DOMAIN != NULL;
}

bool _GodotSharp::_is_domain_finalizing_for_unload(int32_t p_domain_id) {

	return is_domain_finalizing_for_unload(p_domain_id);
}

bool _GodotSharp::is_domain_finalizing_for_unload() {

	return is_domain_finalizing_for_unload(mono_domain_get());
}

bool _GodotSharp::is_domain_finalizing_for_unload(int32_t p_domain_id) {

	return is_domain_finalizing_for_unload(mono_domain_get_by_id(p_domain_id));
}

bool _GodotSharp::is_domain_finalizing_for_unload(MonoDomain *p_domain) {

	if (!p_domain)
		return true;
	if (p_domain == SCRIPTS_DOMAIN && GDMono::get_singleton()->is_finalizing_scripts_domain())
		return true;
	return mono_domain_is_unloading(p_domain);
}

bool _GodotSharp::is_runtime_shutting_down() {

	return mono_runtime_is_shutting_down();
}

bool _GodotSharp::is_runtime_initialized() {

	return GDMono::get_singleton()->is_runtime_initialized();
}

void _GodotSharp::_bind_methods() {

	ClassDB::bind_method(D_METHOD("attach_thread"), &_GodotSharp::attach_thread);
	ClassDB::bind_method(D_METHOD("detach_thread"), &_GodotSharp::detach_thread);

	ClassDB::bind_method(D_METHOD("get_domain_id"), &_GodotSharp::get_domain_id);
	ClassDB::bind_method(D_METHOD("get_scripts_domain_id"), &_GodotSharp::get_scripts_domain_id);
	ClassDB::bind_method(D_METHOD("is_scripts_domain_loaded"), &_GodotSharp::is_scripts_domain_loaded);
	ClassDB::bind_method(D_METHOD("is_domain_finalizing_for_unload", "domain_id"), &_GodotSharp::_is_domain_finalizing_for_unload);

	ClassDB::bind_method(D_METHOD("is_runtime_shutting_down"), &_GodotSharp::is_runtime_shutting_down);
	ClassDB::bind_method(D_METHOD("is_runtime_initialized"), &_GodotSharp::is_runtime_initialized);
}

_GodotSharp::_GodotSharp() {

	singleton = this;
}

_GodotSharp::~_GodotSharp() {

	singleton = NULL;
}
