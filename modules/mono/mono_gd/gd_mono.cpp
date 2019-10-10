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

#include <mono/metadata/environment.h>
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
#include "../godotsharp_dirs.h"
#include "../utils/path_utils.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_utils.h"

#ifdef TOOLS_ENABLED
#include "main/main.h"
#endif

#ifdef ANDROID_ENABLED
#include "android_mono_config.h"
#endif

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

bool _wait_for_debugger_msecs(uint32_t p_msecs) {

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

	CharString da_args = OS::get_singleton()->get_environment("GODOT_MONO_DEBUGGER_AGENT").utf8();

#ifdef TOOLS_ENABLED
	int da_port = GLOBAL_DEF("mono/debugger_agent/port", 23685);
	bool da_suspend = GLOBAL_DEF("mono/debugger_agent/wait_for_debugger", false);
	int da_timeout = GLOBAL_DEF("mono/debugger_agent/wait_timeout", 3000);

	if (Engine::get_singleton()->is_editor_hint() ||
			ProjectSettings::get_singleton()->get_resource_path().empty() ||
			Main::is_project_manager()) {
		if (da_args.size() == 0)
			return;
	}

	if (da_args.length() == 0) {
		da_args = String("--debugger-agent=transport=dt_socket,address=127.0.0.1:" + itos(da_port) +
						 ",embedding=1,server=y,suspend=" + (da_suspend ? "y,timeout=" + itos(da_timeout) : "n"))
						  .utf8();
	}
#else
	if (da_args.length() == 0)
		return; // Exported games don't use the project settings to setup the debugger agent
#endif

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

	char *runtime_build_info = mono_get_runtime_build_info();
	print_verbose("Mono JIT compiler version " + String(runtime_build_info));
	mono_free(runtime_build_info);

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
			String hint_assembly_rootdir = path::join(locations[i], "lib");
			String hint_mscorlib_path = path::join(hint_assembly_rootdir, "mono", "4.5", "mscorlib.dll");
			String hint_config_dir = path::join(locations[i], "etc");

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
	if (DirAccess::exists(bundled_assembly_rootdir)) {
		assembly_rootdir = bundled_assembly_rootdir;
	}

	if (DirAccess::exists(bundled_config_dir)) {
		config_dir = bundled_config_dir;
	}

#ifdef WINDOWS_ENABLED
	if (assembly_rootdir.empty() || config_dir.empty()) {
		ERR_PRINT("Cannot find Mono in the registry.");
		// Assertion: if they are not set, then they weren't found in the registry
		CRASH_COND(mono_reg_info.assembly_dir.length() > 0 || mono_reg_info.config_dir.length() > 0);
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

	{
		PropertyInfo exc_policy_prop = PropertyInfo(Variant::INT, "mono/unhandled_exception_policy", PROPERTY_HINT_ENUM,
				vformat("Terminate Application:%s,Log Error:%s", (int)POLICY_TERMINATE_APP, (int)POLICY_LOG_ERROR));
		unhandled_exception_policy = (UnhandledExceptionPolicy)(int)GLOBAL_DEF(exc_policy_prop.name, (int)POLICY_TERMINATE_APP);
		ProjectSettings::get_singleton()->set_custom_property_info(exc_policy_prop.name, exc_policy_prop);

		if (Engine::get_singleton()->is_editor_hint()) {
			// Unhandled exceptions should not terminate the editor
			unhandled_exception_policy = POLICY_LOG_ERROR;
		}
	}

	GDMonoAssembly::initialize();

	gdmono_profiler_init();

#ifdef DEBUG_ENABLED
	gdmono_debug_init();
#endif

#ifdef ANDROID_ENABLED
	mono_config_parse_memory(get_godot_android_mono_config().utf8().get_data());
#else
	mono_config_parse(NULL);
#endif

	mono_install_unhandled_exception_hook(&unhandled_exception_hook, NULL);

#ifndef TOOLS_ENABLED
	// Export templates only load the Mono runtime if the project uses it
	if (!DirAccess::exists("res://.mono"))
		return;
#endif

#if !defined(WINDOWS_ENABLED) && !defined(NO_MONO_THREADS_SUSPEND_WORKAROUND)
	// FIXME: Temporary workaround. See: https://github.com/godotengine/godot/issues/29812
	if (!OS::get_singleton()->has_environment("MONO_THREADS_SUSPEND")) {
		OS::get_singleton()->set_environment("MONO_THREADS_SUSPEND", "preemptive");
	}
#endif

	root_domain = mono_jit_init_version("GodotEngine.RootDomain", "v4.0.30319");
	ERR_FAIL_NULL_MSG(root_domain, "Mono: Failed to initialize runtime.");

	GDMonoUtils::set_main_thread(GDMonoUtils::get_current_thread());

	setup_runtime_main_args(); // Required for System.Environment.GetCommandLineArgs

	runtime_initialized = true;

	print_verbose("Mono: Runtime initialized");

	// mscorlib assembly MUST be present at initialization
	bool corlib_loaded = _load_corlib_assembly();
	ERR_FAIL_COND_MSG(!corlib_loaded, "Mono: Failed to load mscorlib assembly.");

	Error domain_load_err = _load_scripts_domain();
	ERR_FAIL_COND_MSG(domain_load_err != OK, "Mono: Failed to load scripts domain.");

#ifdef DEBUG_ENABLED
	bool debugger_attached = _wait_for_debugger_msecs(500);
	if (!debugger_attached && OS::get_singleton()->is_stdout_verbose())
		print_error("Mono: Debugger wait timeout");
#endif

	_register_internal_calls();

	print_verbose("Mono: INITIALIZED");
}

void GDMono::initialize_load_assemblies() {

#ifndef MONO_GLUE_ENABLED
	CRASH_NOW_MSG("Mono: This binary was built with 'mono_glue=no'; cannot load assemblies.");
#endif

	// Load assemblies. The API and tools assemblies are required,
	// the application is aborted if these assemblies cannot be loaded.

	_load_api_assemblies();

#if defined(TOOLS_ENABLED)
	bool tool_assemblies_loaded = _load_tools_assemblies();
	CRASH_COND_MSG(!tool_assemblies_loaded, "Mono: Failed to load '" TOOLS_ASM_NAME "' assemblies.");
#endif

	// Load the project's main assembly. This doesn't necessarily need to succeed.
	// The game may not be using .NET at all, or if the project does use .NET and
	// we're running in the editor, it may just happen to be it wasn't built yet.
	if (!_load_project_assembly()) {
		if (OS::get_singleton()->is_stdout_verbose())
			print_error("Mono: Failed to load project assembly");
	}
}

bool GDMono::_are_api_assemblies_out_of_sync() {
	bool out_of_sync = core_api_assembly && (core_api_assembly_out_of_sync || !GDMonoUtils::mono_cache.godot_api_cache_updated);
#ifdef TOOLS_ENABLED
	if (!out_of_sync)
		out_of_sync = editor_api_assembly && editor_api_assembly_out_of_sync;
#endif
	return out_of_sync;
}

namespace GodotSharpBindings {
#ifdef MONO_GLUE_ENABLED

uint64_t get_core_api_hash();
#ifdef TOOLS_ENABLED
uint64_t get_editor_api_hash();
#endif
uint32_t get_bindings_version();
uint32_t get_cs_glue_version();

void register_generated_icalls();

#else

uint64_t get_core_api_hash() {
	GD_UNREACHABLE();
}
#ifdef TOOLS_ENABLED
uint64_t get_editor_api_hash() {
	GD_UNREACHABLE();
}
#endif
uint32_t get_bindings_version() {
	GD_UNREACHABLE();
}
uint32_t get_cs_glue_version() {
	GD_UNREACHABLE();
}

void register_generated_icalls() {
	/* Fine, just do nothing */
}

#endif // MONO_GLUE_ENABLED
} // namespace GodotSharpBindings

void GDMono::_register_internal_calls() {
	GodotSharpBindings::register_generated_icalls();
}

void GDMono::_initialize_and_check_api_hashes() {
#ifdef MONO_GLUE_ENABLED
#ifdef DEBUG_METHODS_ENABLED
	if (get_api_core_hash() != GodotSharpBindings::get_core_api_hash()) {
		ERR_PRINT("Mono: Core API hash mismatch.");
	}

#ifdef TOOLS_ENABLED
	if (get_api_editor_hash() != GodotSharpBindings::get_editor_api_hash()) {
		ERR_PRINT("Mono: Editor API hash mismatch.");
	}
#endif // TOOLS_ENABLED
#endif // DEBUG_METHODS_ENABLED
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

#ifdef TOOLS_ENABLED
bool GDMono::copy_prebuilt_api_assembly(APIAssembly::Type p_api_type, const String &p_config) {

	bool &api_assembly_out_of_sync = (p_api_type == APIAssembly::API_CORE) ?
											 GDMono::get_singleton()->core_api_assembly_out_of_sync :
											 GDMono::get_singleton()->editor_api_assembly_out_of_sync;

	String src_dir = GodotSharpDirs::get_data_editor_prebuilt_api_dir().plus_file(p_config);
	String dst_dir = GodotSharpDirs::get_res_assemblies_base_dir().plus_file(p_config);

	String assembly_name = p_api_type == APIAssembly::API_CORE ? CORE_API_ASSEMBLY_NAME : EDITOR_API_ASSEMBLY_NAME;

	// Create destination directory if needed
	if (!DirAccess::exists(dst_dir)) {
		DirAccess *da = DirAccess::create_for_path(dst_dir);
		Error err = da->make_dir_recursive(dst_dir);
		memdelete(da);

		if (err != OK) {
			ERR_PRINTS("Failed to create destination directory for the API assemblies. Error: " + itos(err) + ".");
			return false;
		}
	}

	String assembly_file = assembly_name + ".dll";
	String assembly_src = src_dir.plus_file(assembly_file);
	String assembly_dst = dst_dir.plus_file(assembly_file);

	if (!FileAccess::exists(assembly_dst) || api_assembly_out_of_sync) {
		DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

		String xml_file = assembly_name + ".xml";
		if (da->copy(src_dir.plus_file(xml_file), dst_dir.plus_file(xml_file)) != OK)
			WARN_PRINTS("Failed to copy '" + xml_file + "'.");

		String pdb_file = assembly_name + ".pdb";
		if (da->copy(src_dir.plus_file(pdb_file), dst_dir.plus_file(pdb_file)) != OK)
			WARN_PRINTS("Failed to copy '" + pdb_file + "'.");

		Error err = da->copy(assembly_src, assembly_dst);

		if (err != OK) {
			ERR_PRINTS("Failed to copy '" + assembly_file + "'.");
			return false;
		}

		api_assembly_out_of_sync = false;
	}

	return true;
}

String GDMono::update_api_assemblies_from_prebuilt() {

#define FAIL_REASON(m_out_of_sync, m_prebuilt_exists)                            \
	(                                                                            \
			(m_out_of_sync ?                                                     \
							String("The assembly is invalidated ") :             \
							String("The assembly was not found ")) +             \
			(m_prebuilt_exists ?                                                 \
							String("and the prebuilt assemblies are missing.") : \
							String("and we failed to copy the prebuilt assemblies.")))

	bool api_assembly_out_of_sync = core_api_assembly_out_of_sync || editor_api_assembly_out_of_sync;

	String core_assembly_path = GodotSharpDirs::get_res_assemblies_dir().plus_file(CORE_API_ASSEMBLY_NAME ".dll");
	String editor_assembly_path = GodotSharpDirs::get_res_assemblies_dir().plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");

	if (!api_assembly_out_of_sync && FileAccess::exists(core_assembly_path) && FileAccess::exists(editor_assembly_path))
		return String(); // No update needed

	const int CONFIGS_LEN = 2;
	String configs[CONFIGS_LEN] = { String("Debug"), String("Release") };

	for (int i = 0; i < CONFIGS_LEN; i++) {
		String config = configs[i];

		print_verbose("Updating '" + config + "' API assemblies");

		String prebuilt_api_dir = GodotSharpDirs::get_data_editor_prebuilt_api_dir().plus_file(config);
		String prebuilt_core_dll_path = prebuilt_api_dir.plus_file(CORE_API_ASSEMBLY_NAME ".dll");
		String prebuilt_editor_dll_path = prebuilt_api_dir.plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");

		if (!FileAccess::exists(prebuilt_core_dll_path) || !FileAccess::exists(prebuilt_editor_dll_path)) {
			return FAIL_REASON(api_assembly_out_of_sync, /* prebuilt_exists: */ false);
		}

		// Copy the prebuilt Api
		if (!copy_prebuilt_api_assembly(APIAssembly::API_CORE, config) ||
				!copy_prebuilt_api_assembly(APIAssembly::API_EDITOR, config)) {
			return FAIL_REASON(api_assembly_out_of_sync, /* prebuilt_exists: */ true);
		}
	}

	return String(); // Updated successfully

#undef FAIL_REASON
}
#endif

bool GDMono::_load_core_api_assembly() {

	if (core_api_assembly)
		return true;

#ifdef TOOLS_ENABLED
	// For the editor and the editor player we want to load it from a specific path to make sure we can keep it up to date

	// If running the project manager, load it from the prebuilt API directory
	String assembly_dir = !Main::is_project_manager() ?
								  GodotSharpDirs::get_res_assemblies_dir() :
								  GodotSharpDirs::get_data_editor_prebuilt_api_dir().plus_file("Debug");

	String assembly_path = assembly_dir.plus_file(CORE_API_ASSEMBLY_NAME ".dll");

	bool success = FileAccess::exists(assembly_path) &&
				   load_assembly_from(CORE_API_ASSEMBLY_NAME, assembly_path, &core_api_assembly);
#else
	bool success = load_assembly(CORE_API_ASSEMBLY_NAME, &core_api_assembly);
#endif

	if (success) {
		APIAssembly::Version api_assembly_ver = APIAssembly::Version::get_from_loaded_assembly(core_api_assembly, APIAssembly::API_CORE);
		core_api_assembly_out_of_sync = GodotSharpBindings::get_core_api_hash() != api_assembly_ver.godot_api_hash ||
										GodotSharpBindings::get_bindings_version() != api_assembly_ver.bindings_version ||
										GodotSharpBindings::get_cs_glue_version() != api_assembly_ver.cs_glue_version;
		if (!core_api_assembly_out_of_sync) {
			GDMonoUtils::update_godot_api_cache();

			_install_trace_listener();
		}
	} else {
		core_api_assembly_out_of_sync = false;
	}

	return success;
}

#ifdef TOOLS_ENABLED
bool GDMono::_load_editor_api_assembly() {

	if (editor_api_assembly)
		return true;

	// For the editor and the editor player we want to load it from a specific path to make sure we can keep it up to date

	// If running the project manager, load it from the prebuilt API directory
	String assembly_dir = !Main::is_project_manager() ?
								  GodotSharpDirs::get_res_assemblies_dir() :
								  GodotSharpDirs::get_data_editor_prebuilt_api_dir().plus_file("Debug");

	String assembly_path = assembly_dir.plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");

	bool success = FileAccess::exists(assembly_path) &&
				   load_assembly_from(EDITOR_API_ASSEMBLY_NAME, assembly_path, &editor_api_assembly);

	if (success) {
		APIAssembly::Version api_assembly_ver = APIAssembly::Version::get_from_loaded_assembly(editor_api_assembly, APIAssembly::API_EDITOR);
		editor_api_assembly_out_of_sync = GodotSharpBindings::get_editor_api_hash() != api_assembly_ver.godot_api_hash ||
										  GodotSharpBindings::get_bindings_version() != api_assembly_ver.bindings_version ||
										  GodotSharpBindings::get_cs_glue_version() != api_assembly_ver.cs_glue_version;
	} else {
		editor_api_assembly_out_of_sync = false;
	}

	return success;
}
#endif

bool GDMono::_try_load_api_assemblies() {

	if (!_load_core_api_assembly()) {
		if (OS::get_singleton()->is_stdout_verbose())
			print_error("Mono: Failed to load Core API assembly");
		return false;
	}

#ifdef TOOLS_ENABLED
	if (!_load_editor_api_assembly()) {
		if (OS::get_singleton()->is_stdout_verbose())
			print_error("Mono: Failed to load Editor API assembly");
		return false;
	}

	if (editor_api_assembly_out_of_sync)
		return false;
#endif

	// Check if the core API assembly is out of sync only after trying to load the
	// editor API assembly. Otherwise, if both assemblies are out of sync, we would
	// only update the former as we won't know the latter also needs to be updated.
	if (core_api_assembly_out_of_sync || !GDMonoUtils::mono_cache.godot_api_cache_updated)
		return false;

	return true;
}

void GDMono::_load_api_assemblies() {

	bool api_assemblies_loaded = _try_load_api_assemblies();

	if (!api_assemblies_loaded) {
#ifdef TOOLS_ENABLED
		// The API assemblies are out of sync. Fine, try one more time, but this time
		// update them from the prebuilt assemblies directory before trying to load them.

		// Shouldn't happen. The project manager loads the prebuilt API assemblies
		CRASH_COND_MSG(Main::is_project_manager(), "Failed to load one of the prebuilt API assemblies.");

		// 1. Unload the scripts domain
		Error domain_unload_err = _unload_scripts_domain();
		CRASH_COND_MSG(domain_unload_err != OK, "Mono: Failed to unload scripts domain.");

		// 2. Update the API assemblies
		String update_error = update_api_assemblies_from_prebuilt();
		CRASH_COND_MSG(!update_error.empty(), update_error);

		// 3. Load the scripts domain again
		Error domain_load_err = _load_scripts_domain();
		CRASH_COND_MSG(domain_load_err != OK, "Mono: Failed to load scripts domain.");

		// 4. Try loading the updated assemblies
		api_assemblies_loaded = _try_load_api_assemblies();
#endif
	}

	if (!api_assemblies_loaded) {
		// welp... too bad

		if (_are_api_assemblies_out_of_sync()) {
			if (core_api_assembly_out_of_sync) {
				ERR_PRINT("The assembly '" CORE_API_ASSEMBLY_NAME "' is out of sync.");
			} else if (!GDMonoUtils::mono_cache.godot_api_cache_updated) {
				ERR_PRINT("The loaded assembly '" CORE_API_ASSEMBLY_NAME "' is in sync, but the cache update failed.");
			}

#ifdef TOOLS_ENABLED
			if (editor_api_assembly_out_of_sync) {
				ERR_PRINT("The assembly '" EDITOR_API_ASSEMBLY_NAME "' is out of sync.");
			}
#endif

			CRASH_NOW();
		} else {
			CRASH_NOW_MSG("Failed to load one of the API assemblies.");
		}
	}
}

#ifdef TOOLS_ENABLED
bool GDMono::_load_tools_assemblies() {

	if (tools_assembly && tools_project_editor_assembly)
		return true;

	bool success = load_assembly(TOOLS_ASM_NAME, &tools_assembly) &&
				   load_assembly(TOOLS_PROJECT_EDITOR_ASM_NAME, &tools_project_editor_assembly);

	return success;
}
#endif

bool GDMono::_load_project_assembly() {

	if (project_assembly)
		return true;

	String appname = ProjectSettings::get_singleton()->get("application/config/name");
	String appname_safe = OS::get_singleton()->get_safe_dir_name(appname);
	if (appname_safe.empty()) {
		appname_safe = "UnnamedProject";
	}

	bool success = load_assembly(appname_safe, &project_assembly);

	if (success) {
		mono_assembly_set_main(project_assembly->get_assembly());
	}

	return success;
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
		ERR_PRINT("Failed to install 'System.Diagnostics.Trace' listener.");
		GDMonoUtils::debug_print_unhandled_exception(exc);
	}
#endif
}

Error GDMono::_load_scripts_domain() {

	ERR_FAIL_COND_V(scripts_domain != NULL, ERR_BUG);

	print_verbose("Mono: Loading scripts domain...");

	scripts_domain = GDMonoUtils::create_domain("GodotEngine.ScriptsDomain");

	ERR_FAIL_NULL_V_MSG(scripts_domain, ERR_CANT_CREATE, "Mono: Could not create scripts app domain.");

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
		ERR_PRINT("Mono: Domain finalization timeout.");
	}

	finalizing_scripts_domain = false;

	mono_gc_collect(mono_gc_max_generation());

	GDMonoUtils::clear_godot_api_cache();

	_domain_assemblies_cleanup(mono_domain_get_id(scripts_domain));

	core_api_assembly = NULL;
	project_assembly = NULL;
#ifdef TOOLS_ENABLED
	editor_api_assembly = NULL;
	tools_assembly = NULL;
	tools_project_editor_assembly = NULL;
#endif

	MonoDomain *domain = scripts_domain;
	scripts_domain = NULL;

	MonoException *exc = NULL;
	mono_domain_try_unload(domain, (MonoObject **)&exc);

	if (exc) {
		ERR_PRINT("Exception thrown when unloading scripts domain.");
		GDMonoUtils::debug_unhandled_exception(exc);
		return FAILED;
	}

	return OK;
}

#ifdef GD_MONO_HOT_RELOAD
Error GDMono::reload_scripts_domain() {

	ERR_FAIL_COND_V(!runtime_initialized, ERR_BUG);

	if (scripts_domain) {
		Error domain_unload_err = _unload_scripts_domain();
		ERR_FAIL_COND_V_MSG(domain_unload_err != OK, domain_unload_err, "Mono: Failed to unload scripts domain.");
	}

	CSharpLanguage::get_singleton()->_on_scripts_domain_unloaded();

	Error domain_load_err = _load_scripts_domain();
	ERR_FAIL_COND_V_MSG(domain_load_err != OK, domain_load_err, "Mono: Failed to load scripts domain.");

	// Load assemblies. The API and tools assemblies are required,
	// the application is aborted if these assemblies cannot be loaded.

	_load_api_assemblies();

#if defined(TOOLS_ENABLED)
	bool tools_assemblies_loaded = _load_tools_assemblies();
	CRASH_COND_MSG(!tools_assemblies_loaded, "Mono: Failed to load '" TOOLS_ASM_NAME "' assemblies.");
#endif

	// Load the project's main assembly. Here, during hot-reloading, we do
	// consider failing to load the project's main assembly to be an error.
	// However, unlike the API and tools assemblies, the application can continue working.
	if (!_load_project_assembly()) {
		print_error("Mono: Failed to load project assembly");
		return ERR_CANT_OPEN;
	}

	return OK;
}
#endif

Error GDMono::finalize_and_unload_domain(MonoDomain *p_domain) {

	CRASH_COND(p_domain == NULL);
	CRASH_COND(p_domain == GDMono::get_singleton()->get_scripts_domain()); // Should use _unload_scripts_domain() instead

	String domain_name = mono_domain_get_friendly_name(p_domain);

	print_verbose("Mono: Unloading domain '" + domain_name + "'...");

	if (mono_domain_get() == p_domain)
		mono_domain_set(root_domain, true);

	if (!mono_domain_finalize(p_domain, 2000)) {
		ERR_PRINT("Mono: Domain finalization timeout.");
	}

	mono_gc_collect(mono_gc_max_generation());

	_domain_assemblies_cleanup(mono_domain_get_id(p_domain));

	MonoException *exc = NULL;
	mono_domain_try_unload(p_domain, (MonoObject **)&exc);

	if (exc) {
		ERR_PRINTS("Exception thrown when unloading domain '" + domain_name + "'.");
		GDMonoUtils::debug_print_unhandled_exception(exc);
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

GDMonoClass *GDMono::get_class(const StringName &p_namespace, const StringName &p_name) {

	uint32_t domain_id = mono_domain_get_id(mono_domain_get());
	HashMap<String, GDMonoAssembly *> &domain_assemblies = assemblies[domain_id];

	const String *k = NULL;
	while ((k = domain_assemblies.next(k))) {
		GDMonoAssembly *assembly = domain_assemblies.get(*k);
		GDMonoClass *klass = assembly->get_class(p_namespace, p_name);
		if (klass)
			return klass;
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

	exit(mono_environment_exitcode_get());

	GD_UNREACHABLE();
}

GDMono::GDMono() {

	singleton = this;

	gdmono_log = memnew(GDMonoLog);

	runtime_initialized = false;
	finalizing_scripts_domain = false;

	root_domain = NULL;
	scripts_domain = NULL;

	core_api_assembly_out_of_sync = false;
#ifdef TOOLS_ENABLED
	editor_api_assembly_out_of_sync = false;
#endif

	corlib_assembly = NULL;
	core_api_assembly = NULL;
	project_assembly = NULL;
#ifdef TOOLS_ENABLED
	editor_api_assembly = NULL;
	tools_assembly = NULL;
	tools_project_editor_assembly = NULL;
#endif

	api_core_hash = 0;
#ifdef TOOLS_ENABLED
	api_editor_hash = 0;
#endif

	unhandled_exception_policy = POLICY_TERMINATE_APP;
}

GDMono::~GDMono() {

	if (is_runtime_initialized()) {
		if (scripts_domain) {
			Error err = _unload_scripts_domain();
			if (err != OK) {
				ERR_PRINT("Mono: Failed to unload scripts domain.");
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

	MonoDomain *domain = GDMono::get_singleton()->get_scripts_domain();
	CRASH_COND(!domain); // User must check if scripts domain is loaded before calling this method
	return mono_domain_get_id(domain);
}

bool _GodotSharp::is_scripts_domain_loaded() {

	return GDMono::get_singleton()->is_runtime_initialized() && GDMono::get_singleton()->get_scripts_domain() != NULL;
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
	if (p_domain == GDMono::get_singleton()->get_scripts_domain() && GDMono::get_singleton()->is_finalizing_scripts_domain())
		return true;
	return mono_domain_is_unloading(p_domain);
}

bool _GodotSharp::is_runtime_shutting_down() {

	return mono_runtime_is_shutting_down();
}

bool _GodotSharp::is_runtime_initialized() {

	return GDMono::get_singleton()->is_runtime_initialized();
}

void _GodotSharp::_reload_assemblies(bool p_soft_reload) {
#ifdef GD_MONO_HOT_RELOAD
	CSharpLanguage::get_singleton()->reload_assemblies(p_soft_reload);
#endif
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
	ClassDB::bind_method(D_METHOD("_reload_assemblies"), &_GodotSharp::_reload_assemblies);
}

_GodotSharp::_GodotSharp() {

	singleton = this;
}

_GodotSharp::~_GodotSharp() {

	singleton = NULL;
}
