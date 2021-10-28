/*************************************************************************/
/*  gd_mono.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "core/os/thread.h"

#include "../csharp_script.h"
#include "../godotsharp_dirs.h"
#include "../utils/path_utils.h"
#include "gd_mono_cache.h"
#include "gd_mono_class.h"
#include "gd_mono_marshal.h"
#include "gd_mono_utils.h"

#ifdef TOOLS_ENABLED
#include "main/main.h"
#endif

#ifdef ANDROID_ENABLED
#include "android_mono_config.h"
#include "support/android_support.h"
#elif defined(IPHONE_ENABLED)
#include "support/ios_support.h"
#endif

#if defined(TOOL_ENABLED) && defined(GD_MONO_SINGLE_APPDOMAIN)
// This will no longer be the case if we replace appdomains with AssemblyLoadContext
#error "Editor build requires support for multiple appdomains"
#endif

#if defined(GD_MONO_HOT_RELOAD) && defined(GD_MONO_SINGLE_APPDOMAIN)
#error "Hot reloading requires multiple appdomains"
#endif

// TODO:
// This has turned into a gigantic mess. There's too much going on here. Too much #ifdef as well.
// It's just painful to read... It needs to be re-structured. Please, clean this up, future me.

GDMono *GDMono::singleton = nullptr;

namespace {

#if defined(JAVASCRIPT_ENABLED)
extern "C" {
void mono_wasm_load_runtime(const char *managed_path, int enable_debugging);
}
#endif

#if !defined(JAVASCRIPT_ENABLED)

void gd_mono_setup_runtime_main_args() {
	CharString execpath = OS::get_singleton()->get_executable_path().utf8();

	List<String> cmdline_args = OS::get_singleton()->get_cmdline_args();

	List<CharString> cmdline_args_utf8;
	Vector<char *> main_args;
	main_args.resize(cmdline_args.size() + 1);

	main_args.write[0] = execpath.ptrw();

	int i = 1;
	for (const String &E : cmdline_args) {
		CharString &stored = cmdline_args_utf8.push_back(E.utf8())->get();
		main_args.write[i] = stored.ptrw();
		i++;
	}

	mono_runtime_set_main_args(main_args.size(), main_args.ptrw());
}

void gd_mono_profiler_init() {
	String profiler_args = GLOBAL_DEF("mono/profiler/args", "log:calls,alloc,sample,output=output.mlpd");
	bool profiler_enabled = GLOBAL_DEF("mono/profiler/enabled", false);
	if (profiler_enabled) {
		mono_profiler_load(profiler_args.utf8());
		return;
	}

	const String env_var_name = "MONO_ENV_OPTIONS";
	if (OS::get_singleton()->has_environment(env_var_name)) {
		const String mono_env_ops = OS::get_singleton()->get_environment(env_var_name);
		// Usually MONO_ENV_OPTIONS looks like:   --profile=jb:prof=timeline,ctl=remote,host=127.0.0.1:55467
		const String prefix = "--profile=";
		if (mono_env_ops.begins_with(prefix)) {
			const String ops = mono_env_ops.substr(prefix.length(), mono_env_ops.length());
			mono_profiler_load(ops.utf8());
		}
	}
}

void gd_mono_debug_init() {
	CharString da_args = OS::get_singleton()->get_environment("GODOT_MONO_DEBUGGER_AGENT").utf8();

	if (da_args.length()) {
		OS::get_singleton()->set_environment("GODOT_MONO_DEBUGGER_AGENT", String());
	}

#ifdef TOOLS_ENABLED
	int da_port = GLOBAL_DEF("mono/debugger_agent/port", 23685);
	bool da_suspend = GLOBAL_DEF("mono/debugger_agent/wait_for_debugger", false);
	int da_timeout = GLOBAL_DEF("mono/debugger_agent/wait_timeout", 3000);

	if (Engine::get_singleton()->is_editor_hint() ||
			ProjectSettings::get_singleton()->get_resource_path().is_empty() ||
			Main::is_project_manager()) {
		if (da_args.size() == 0) {
			return;
		}
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

	// Debugging enabled

	mono_debug_init(MONO_DEBUG_FORMAT_MONO);

	// --debugger-agent=help
	const char *options[] = {
		"--soft-breakpoints",
		da_args.get_data()
	};
	mono_jit_parse_options(2, (char **)options);
}

#endif // !defined(JAVASCRIPT_ENABLED)

#if defined(JAVASCRIPT_ENABLED)
MonoDomain *gd_initialize_mono_runtime() {
	const char *vfs_prefix = "managed";
	int enable_debugging = 0;

	// TODO: Provide a way to enable debugging on WASM release builds.
#ifdef DEBUG_ENABLED
	enable_debugging = 1;
#endif

	mono_wasm_load_runtime(vfs_prefix, enable_debugging);

	return mono_get_root_domain();
}
#else
MonoDomain *gd_initialize_mono_runtime() {
	gd_mono_debug_init();

#if defined(IPHONE_ENABLED) || defined(ANDROID_ENABLED)
	// I don't know whether this actually matters or not
	const char *runtime_version = "mobile";
#else
	const char *runtime_version = "v4.0.30319";
#endif

	return mono_jit_init_version("GodotEngine.RootDomain", runtime_version);
}
#endif
} // namespace

void GDMono::add_mono_shared_libs_dir_to_path() {
	// TODO: Replace this with a mono_dl_fallback

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

void GDMono::determine_mono_dirs(String &r_assembly_rootdir, String &r_config_dir) {
	String bundled_assembly_rootdir = GodotSharpDirs::get_data_mono_lib_dir();
	String bundled_config_dir = GodotSharpDirs::get_data_mono_etc_dir();

#ifdef TOOLS_ENABLED

#if defined(WINDOWS_ENABLED)
	mono_reg_info = MonoRegUtils::find_mono();

	if (mono_reg_info.assembly_dir.length() && DirAccess::exists(mono_reg_info.assembly_dir)) {
		r_assembly_rootdir = mono_reg_info.assembly_dir;
	}

	if (mono_reg_info.config_dir.length() && DirAccess::exists(mono_reg_info.config_dir)) {
		r_config_dir = mono_reg_info.config_dir;
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
				r_assembly_rootdir = hint_assembly_rootdir;
				r_config_dir = hint_config_dir;
				break;
			}
		}
	}
#endif

	if (DirAccess::exists(bundled_assembly_rootdir)) {
		r_assembly_rootdir = bundled_assembly_rootdir;
	}

	if (DirAccess::exists(bundled_config_dir)) {
		r_config_dir = bundled_config_dir;
	}

#ifdef WINDOWS_ENABLED
	if (r_assembly_rootdir.is_empty() || r_config_dir.is_empty()) {
		ERR_PRINT("Cannot find Mono in the registry.");
		// Assertion: if they are not set, then they weren't found in the registry
		CRASH_COND(mono_reg_info.assembly_dir.length() > 0 || mono_reg_info.config_dir.length() > 0);
	}
#endif // WINDOWS_ENABLED

#else
	// Export templates always use the bundled directories
	r_assembly_rootdir = bundled_assembly_rootdir;
	r_config_dir = bundled_config_dir;
#endif
}

void GDMono::initialize() {
	ERR_FAIL_NULL(Engine::get_singleton());

	print_verbose("Mono: Initializing module...");

	char *runtime_build_info = mono_get_runtime_build_info();
	print_verbose("Mono JIT compiler version " + String(runtime_build_info));
	mono_free(runtime_build_info);

	_init_godot_api_hashes();
	_init_exception_policy();

	GDMonoLog::get_singleton()->initialize();

#if !defined(JAVASCRIPT_ENABLED)
	String assembly_rootdir;
	String config_dir;
	determine_mono_dirs(assembly_rootdir, config_dir);

	// Leak if we call mono_set_dirs more than once
	mono_set_dirs(assembly_rootdir.length() ? assembly_rootdir.utf8().get_data() : nullptr,
			config_dir.length() ? config_dir.utf8().get_data() : nullptr);

	add_mono_shared_libs_dir_to_path();
#endif

#ifdef ANDROID_ENABLED
	mono_config_parse_memory(get_godot_android_mono_config().utf8().get_data());
#else
	mono_config_parse(nullptr);
#endif

#if defined(ANDROID_ENABLED)
	gdmono::android::support::initialize();
#elif defined(IPHONE_ENABLED)
	gdmono::ios::support::initialize();
#endif

	GDMonoAssembly::initialize();

#if !defined(JAVASCRIPT_ENABLED)
	gd_mono_profiler_init();
#endif

	mono_install_unhandled_exception_hook(&unhandled_exception_hook, nullptr);

#ifndef TOOLS_ENABLED
	// Exported games that don't use C# must still work. They likely don't ship with mscorlib.
	// We only initialize the Mono runtime if we can find mscorlib. Otherwise it would crash.
	if (GDMonoAssembly::find_assembly("mscorlib.dll").is_empty()) {
		print_verbose("Mono: Skipping runtime initialization because 'mscorlib.dll' could not be found");
		return;
	}
#endif

#if !defined(NO_MONO_THREADS_SUSPEND_WORKAROUND)
	// FIXME: Temporary workaround. See: https://github.com/godotengine/godot/issues/29812
	if (!OS::get_singleton()->has_environment("MONO_THREADS_SUSPEND")) {
		OS::get_singleton()->set_environment("MONO_THREADS_SUSPEND", "preemptive");
	}
#endif

	// NOTE: Internal calls must be registered after the Mono runtime initialization.
	// Otherwise registration fails with the error: 'assertion 'hash != nullptr' failed'.

	root_domain = gd_initialize_mono_runtime();
	ERR_FAIL_NULL_MSG(root_domain, "Mono: Failed to initialize runtime.");

	GDMonoUtils::set_main_thread(GDMonoUtils::get_current_thread());

#if !defined(JAVASCRIPT_ENABLED)
	gd_mono_setup_runtime_main_args(); // Required for System.Environment.GetCommandLineArgs
#endif

	runtime_initialized = true;

	print_verbose("Mono: Runtime initialized");

#if defined(ANDROID_ENABLED)
	gdmono::android::support::register_internal_calls();
#endif

	// mscorlib assembly MUST be present at initialization
	bool corlib_loaded = _load_corlib_assembly();
	ERR_FAIL_COND_MSG(!corlib_loaded, "Mono: Failed to load mscorlib assembly.");

#ifndef GD_MONO_SINGLE_APPDOMAIN
	Error domain_load_err = _load_scripts_domain();
	ERR_FAIL_COND_MSG(domain_load_err != OK, "Mono: Failed to load scripts domain.");
#else
	scripts_domain = root_domain;
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

	if (Main::is_project_manager()) {
		return;
	}
#endif

	// Load the project's main assembly. This doesn't necessarily need to succeed.
	// The game may not be using .NET at all, or if the project does use .NET and
	// we're running in the editor, it may just happen to be it wasn't built yet.
	if (!_load_project_assembly()) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			print_error("Mono: Failed to load project assembly");
		}
	}
}

bool GDMono::_are_api_assemblies_out_of_sync() {
	bool out_of_sync = core_api_assembly.assembly && (core_api_assembly.out_of_sync || !GDMonoCache::cached_data.godot_api_cache_updated);
#ifdef TOOLS_ENABLED
	if (!out_of_sync) {
		out_of_sync = editor_api_assembly.assembly && editor_api_assembly.out_of_sync;
	}
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

void GDMono::_init_godot_api_hashes() {
#if defined(MONO_GLUE_ENABLED) && defined(DEBUG_METHODS_ENABLED)
	if (get_api_core_hash() != GodotSharpBindings::get_core_api_hash()) {
		ERR_PRINT("Mono: Core API hash mismatch.");
	}

#ifdef TOOLS_ENABLED
	if (get_api_editor_hash() != GodotSharpBindings::get_editor_api_hash()) {
		ERR_PRINT("Mono: Editor API hash mismatch.");
	}
#endif // TOOLS_ENABLED
#endif // MONO_GLUE_ENABLED && DEBUG_METHODS_ENABLED
}

void GDMono::_init_exception_policy() {
	PropertyInfo exc_policy_prop = PropertyInfo(Variant::INT, "mono/unhandled_exception_policy", PROPERTY_HINT_ENUM,
			vformat("Terminate Application:%s,Log Error:%s", (int)POLICY_TERMINATE_APP, (int)POLICY_LOG_ERROR));
	unhandled_exception_policy = (UnhandledExceptionPolicy)(int)GLOBAL_DEF(exc_policy_prop.name, (int)POLICY_TERMINATE_APP);
	ProjectSettings::get_singleton()->set_custom_property_info(exc_policy_prop.name, exc_policy_prop);

	if (Engine::get_singleton()->is_editor_hint()) {
		// Unhandled exceptions should not terminate the editor
		unhandled_exception_policy = POLICY_LOG_ERROR;
	}
}

void GDMono::add_assembly(int32_t p_domain_id, GDMonoAssembly *p_assembly) {
	assemblies[p_domain_id][p_assembly->get_name()] = p_assembly;
}

GDMonoAssembly *GDMono::get_loaded_assembly(const String &p_name) {
	if (p_name == "mscorlib" && corlib_assembly) {
		return corlib_assembly;
	}

	MonoDomain *domain = mono_domain_get();
	int32_t domain_id = domain ? mono_domain_get_id(domain) : 0;
	GDMonoAssembly **result = assemblies[domain_id].getptr(p_name);
	return result ? *result : nullptr;
}

bool GDMono::load_assembly(const String &p_name, GDMonoAssembly **r_assembly, bool p_refonly) {
#ifdef DEBUG_ENABLED
	CRASH_COND(!r_assembly);
#endif

	MonoAssemblyName *aname = mono_assembly_name_new(p_name.utf8());
	bool result = load_assembly(p_name, aname, r_assembly, p_refonly);
	mono_assembly_name_free(aname);
	mono_free(aname);

	return result;
}

bool GDMono::load_assembly(const String &p_name, MonoAssemblyName *p_aname, GDMonoAssembly **r_assembly, bool p_refonly) {
#ifdef DEBUG_ENABLED
	CRASH_COND(!r_assembly);
#endif

	return load_assembly(p_name, p_aname, r_assembly, p_refonly, GDMonoAssembly::get_default_search_dirs());
}

bool GDMono::load_assembly(const String &p_name, MonoAssemblyName *p_aname, GDMonoAssembly **r_assembly, bool p_refonly, const Vector<String> &p_search_dirs) {
#ifdef DEBUG_ENABLED
	CRASH_COND(!r_assembly);
#endif

	print_verbose("Mono: Loading assembly " + p_name + (p_refonly ? " (refonly)" : "") + "...");

	GDMonoAssembly *assembly = GDMonoAssembly::load(p_name, p_aname, p_refonly, p_search_dirs);

	if (!assembly) {
		return false;
	}

	*r_assembly = assembly;

	print_verbose("Mono: Assembly " + p_name + (p_refonly ? " (refonly)" : "") + " loaded from path: " + (*r_assembly)->get_path());

	return true;
}

bool GDMono::load_assembly_from(const String &p_name, const String &p_path, GDMonoAssembly **r_assembly, bool p_refonly) {
	CRASH_COND(!r_assembly);

	print_verbose("Mono: Loading assembly " + p_name + (p_refonly ? " (refonly)" : "") + "...");

	GDMonoAssembly *assembly = GDMonoAssembly::load_from(p_name, p_path, p_refonly);

	if (!assembly) {
		return false;
	}

	*r_assembly = assembly;

	print_verbose("Mono: Assembly " + p_name + (p_refonly ? " (refonly)" : "") + " loaded from path: " + (*r_assembly)->get_path());

	return true;
}

ApiAssemblyInfo::Version ApiAssemblyInfo::Version::get_from_loaded_assembly(GDMonoAssembly *p_api_assembly, ApiAssemblyInfo::Type p_api_type) {
	ApiAssemblyInfo::Version api_assembly_version;

	const char *nativecalls_name = p_api_type == ApiAssemblyInfo::API_CORE ?
			  BINDINGS_CLASS_NATIVECALLS :
			  BINDINGS_CLASS_NATIVECALLS_EDITOR;

	GDMonoClass *nativecalls_klass = p_api_assembly->get_class(BINDINGS_NAMESPACE, nativecalls_name);

	if (nativecalls_klass) {
		GDMonoField *api_hash_field = nativecalls_klass->get_field("godot_api_hash");
		if (api_hash_field) {
			api_assembly_version.godot_api_hash = GDMonoMarshal::unbox<uint64_t>(api_hash_field->get_value(nullptr));
		}

		GDMonoField *binds_ver_field = nativecalls_klass->get_field("bindings_version");
		if (binds_ver_field) {
			api_assembly_version.bindings_version = GDMonoMarshal::unbox<uint32_t>(binds_ver_field->get_value(nullptr));
		}

		GDMonoField *cs_glue_ver_field = nativecalls_klass->get_field("cs_glue_version");
		if (cs_glue_ver_field) {
			api_assembly_version.cs_glue_version = GDMonoMarshal::unbox<uint32_t>(cs_glue_ver_field->get_value(nullptr));
		}
	}

	return api_assembly_version;
}

String ApiAssemblyInfo::to_string(ApiAssemblyInfo::Type p_type) {
	return p_type == ApiAssemblyInfo::API_CORE ? "API_CORE" : "API_EDITOR";
}

bool GDMono::_load_corlib_assembly() {
	if (corlib_assembly) {
		return true;
	}

	bool success = load_assembly("mscorlib", &corlib_assembly);

	if (success) {
		GDMonoCache::update_corlib_cache();
	}

	return success;
}

#ifdef TOOLS_ENABLED
bool GDMono::copy_prebuilt_api_assembly(ApiAssemblyInfo::Type p_api_type, const String &p_config) {
	String src_dir = GodotSharpDirs::get_data_editor_prebuilt_api_dir().plus_file(p_config);
	String dst_dir = GodotSharpDirs::get_res_assemblies_base_dir().plus_file(p_config);

	String assembly_name = p_api_type == ApiAssemblyInfo::API_CORE ? CORE_API_ASSEMBLY_NAME : EDITOR_API_ASSEMBLY_NAME;

	// Create destination directory if needed
	if (!DirAccess::exists(dst_dir)) {
		DirAccess *da = DirAccess::create_for_path(dst_dir);
		Error err = da->make_dir_recursive(dst_dir);
		memdelete(da);

		if (err != OK) {
			ERR_PRINT("Failed to create destination directory for the API assemblies. Error: " + itos(err) + ".");
			return false;
		}
	}

	DirAccessRef da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	String xml_file = assembly_name + ".xml";
	if (da->copy(src_dir.plus_file(xml_file), dst_dir.plus_file(xml_file)) != OK) {
		WARN_PRINT("Failed to copy '" + xml_file + "'.");
	}

	String pdb_file = assembly_name + ".pdb";
	if (da->copy(src_dir.plus_file(pdb_file), dst_dir.plus_file(pdb_file)) != OK) {
		WARN_PRINT("Failed to copy '" + pdb_file + "'.");
	}

	String assembly_file = assembly_name + ".dll";
	if (da->copy(src_dir.plus_file(assembly_file), dst_dir.plus_file(assembly_file)) != OK) {
		ERR_PRINT("Failed to copy '" + assembly_file + "'.");
		return false;
	}

	return true;
}

static bool try_get_cached_api_hash_for(const String &p_api_assemblies_dir, bool &r_out_of_sync) {
	String core_api_assembly_path = p_api_assemblies_dir.plus_file(CORE_API_ASSEMBLY_NAME ".dll");
	String editor_api_assembly_path = p_api_assemblies_dir.plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");

	if (!FileAccess::exists(core_api_assembly_path) || !FileAccess::exists(editor_api_assembly_path)) {
		return false;
	}

	String cached_api_hash_path = p_api_assemblies_dir.plus_file("api_hash_cache.cfg");

	if (!FileAccess::exists(cached_api_hash_path)) {
		return false;
	}

	Ref<ConfigFile> cfg;
	cfg.instantiate();
	Error cfg_err = cfg->load(cached_api_hash_path);
	ERR_FAIL_COND_V(cfg_err != OK, false);

	// Checking the modified time is good enough
	if (FileAccess::get_modified_time(core_api_assembly_path) != (uint64_t)cfg->get_value("core", "modified_time") ||
			FileAccess::get_modified_time(editor_api_assembly_path) != (uint64_t)cfg->get_value("editor", "modified_time")) {
		return false;
	}

	r_out_of_sync = GodotSharpBindings::get_bindings_version() != (uint32_t)cfg->get_value("core", "bindings_version") ||
			GodotSharpBindings::get_cs_glue_version() != (uint32_t)cfg->get_value("core", "cs_glue_version") ||
			GodotSharpBindings::get_bindings_version() != (uint32_t)cfg->get_value("editor", "bindings_version") ||
			GodotSharpBindings::get_cs_glue_version() != (uint32_t)cfg->get_value("editor", "cs_glue_version") ||
			GodotSharpBindings::get_core_api_hash() != (uint64_t)cfg->get_value("core", "api_hash") ||
			GodotSharpBindings::get_editor_api_hash() != (uint64_t)cfg->get_value("editor", "api_hash");

	return true;
}

static void create_cached_api_hash_for(const String &p_api_assemblies_dir) {
	String core_api_assembly_path = p_api_assemblies_dir.plus_file(CORE_API_ASSEMBLY_NAME ".dll");
	String editor_api_assembly_path = p_api_assemblies_dir.plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");
	String cached_api_hash_path = p_api_assemblies_dir.plus_file("api_hash_cache.cfg");

	Ref<ConfigFile> cfg;
	cfg.instantiate();

	cfg->set_value("core", "modified_time", FileAccess::get_modified_time(core_api_assembly_path));
	cfg->set_value("editor", "modified_time", FileAccess::get_modified_time(editor_api_assembly_path));

	cfg->set_value("core", "bindings_version", GodotSharpBindings::get_bindings_version());
	cfg->set_value("core", "cs_glue_version", GodotSharpBindings::get_cs_glue_version());
	cfg->set_value("editor", "bindings_version", GodotSharpBindings::get_bindings_version());
	cfg->set_value("editor", "cs_glue_version", GodotSharpBindings::get_cs_glue_version());

	// This assumes the prebuilt api assemblies we copied to the project are not out of sync
	cfg->set_value("core", "api_hash", GodotSharpBindings::get_core_api_hash());
	cfg->set_value("editor", "api_hash", GodotSharpBindings::get_editor_api_hash());

	Error err = cfg->save(cached_api_hash_path);
	ERR_FAIL_COND(err != OK);
}

bool GDMono::_temp_domain_load_are_assemblies_out_of_sync(const String &p_config) {
	MonoDomain *temp_domain = GDMonoUtils::create_domain("GodotEngine.Domain.CheckApiAssemblies");
	ERR_FAIL_NULL_V(temp_domain, "Failed to create temporary domain to check API assemblies");
	_GDMONO_SCOPE_EXIT_DOMAIN_UNLOAD_(temp_domain);

	_GDMONO_SCOPE_DOMAIN_(temp_domain);

	GDMono::LoadedApiAssembly temp_core_api_assembly;
	GDMono::LoadedApiAssembly temp_editor_api_assembly;

	if (!_try_load_api_assemblies(temp_core_api_assembly, temp_editor_api_assembly,
				p_config, /* refonly: */ true, /* loaded_callback: */ nullptr)) {
		return temp_core_api_assembly.out_of_sync || temp_editor_api_assembly.out_of_sync;
	}

	return true; // Failed to load, assume they're outdated assemblies
}

String GDMono::update_api_assemblies_from_prebuilt(const String &p_config, const bool *p_core_api_out_of_sync, const bool *p_editor_api_out_of_sync) {
#define FAIL_REASON(m_out_of_sync, m_prebuilt_exists)                            \
	(                                                                            \
			(m_out_of_sync ?                                                     \
							  String("The assembly is invalidated ") :             \
							  String("The assembly was not found ")) +             \
			(m_prebuilt_exists ?                                                 \
							  String("and the prebuilt assemblies are missing.") : \
							  String("and we failed to copy the prebuilt assemblies.")))

	String dst_assemblies_dir = GodotSharpDirs::get_res_assemblies_base_dir().plus_file(p_config);

	String core_assembly_path = dst_assemblies_dir.plus_file(CORE_API_ASSEMBLY_NAME ".dll");
	String editor_assembly_path = dst_assemblies_dir.plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");

	bool api_assemblies_out_of_sync = false;

	if (p_core_api_out_of_sync && p_editor_api_out_of_sync) {
		api_assemblies_out_of_sync = p_core_api_out_of_sync || p_editor_api_out_of_sync;
	} else if (FileAccess::exists(core_assembly_path) && FileAccess::exists(editor_assembly_path)) {
		// Determine if they're out of sync
		if (!try_get_cached_api_hash_for(dst_assemblies_dir, api_assemblies_out_of_sync)) {
			api_assemblies_out_of_sync = _temp_domain_load_are_assemblies_out_of_sync(p_config);
		}
	}

	// Note: Even if only one of the assemblies if missing or out of sync, we update both

	if (!api_assemblies_out_of_sync && FileAccess::exists(core_assembly_path) && FileAccess::exists(editor_assembly_path)) {
		return String(); // No update needed
	}

	print_verbose("Updating '" + p_config + "' API assemblies");

	String prebuilt_api_dir = GodotSharpDirs::get_data_editor_prebuilt_api_dir().plus_file(p_config);
	String prebuilt_core_dll_path = prebuilt_api_dir.plus_file(CORE_API_ASSEMBLY_NAME ".dll");
	String prebuilt_editor_dll_path = prebuilt_api_dir.plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");

	if (!FileAccess::exists(prebuilt_core_dll_path) || !FileAccess::exists(prebuilt_editor_dll_path)) {
		return FAIL_REASON(api_assemblies_out_of_sync, /* prebuilt_exists: */ false);
	}

	// Copy the prebuilt Api
	if (!copy_prebuilt_api_assembly(ApiAssemblyInfo::API_CORE, p_config) ||
			!copy_prebuilt_api_assembly(ApiAssemblyInfo::API_EDITOR, p_config)) {
		return FAIL_REASON(api_assemblies_out_of_sync, /* prebuilt_exists: */ true);
	}

	// Cache the api hash of the assemblies we just copied
	create_cached_api_hash_for(dst_assemblies_dir);

	return String(); // Updated successfully

#undef FAIL_REASON
}
#endif

bool GDMono::_load_core_api_assembly(LoadedApiAssembly &r_loaded_api_assembly, const String &p_config, bool p_refonly) {
	if (r_loaded_api_assembly.assembly) {
		return true;
	}

#ifdef TOOLS_ENABLED
	// For the editor and the editor player we want to load it from a specific path to make sure we can keep it up to date

	// If running the project manager, load it from the prebuilt API directory
	String assembly_dir = !Main::is_project_manager() ?
			  GodotSharpDirs::get_res_assemblies_base_dir().plus_file(p_config) :
			  GodotSharpDirs::get_data_editor_prebuilt_api_dir().plus_file(p_config);

	String assembly_path = assembly_dir.plus_file(CORE_API_ASSEMBLY_NAME ".dll");

	bool success = FileAccess::exists(assembly_path) &&
			load_assembly_from(CORE_API_ASSEMBLY_NAME, assembly_path, &r_loaded_api_assembly.assembly, p_refonly);
#else
	bool success = load_assembly(CORE_API_ASSEMBLY_NAME, &r_loaded_api_assembly.assembly, p_refonly);
#endif

	if (success) {
		ApiAssemblyInfo::Version api_assembly_ver = ApiAssemblyInfo::Version::get_from_loaded_assembly(r_loaded_api_assembly.assembly, ApiAssemblyInfo::API_CORE);
		r_loaded_api_assembly.out_of_sync = GodotSharpBindings::get_core_api_hash() != api_assembly_ver.godot_api_hash ||
				GodotSharpBindings::get_bindings_version() != api_assembly_ver.bindings_version ||
				GodotSharpBindings::get_cs_glue_version() != api_assembly_ver.cs_glue_version;
	} else {
		r_loaded_api_assembly.out_of_sync = false;
	}

	return success;
}

#ifdef TOOLS_ENABLED
bool GDMono::_load_editor_api_assembly(LoadedApiAssembly &r_loaded_api_assembly, const String &p_config, bool p_refonly) {
	if (r_loaded_api_assembly.assembly) {
		return true;
	}

	// For the editor and the editor player we want to load it from a specific path to make sure we can keep it up to date

	// If running the project manager, load it from the prebuilt API directory
	String assembly_dir = !Main::is_project_manager() ?
			  GodotSharpDirs::get_res_assemblies_base_dir().plus_file(p_config) :
			  GodotSharpDirs::get_data_editor_prebuilt_api_dir().plus_file(p_config);

	String assembly_path = assembly_dir.plus_file(EDITOR_API_ASSEMBLY_NAME ".dll");

	bool success = FileAccess::exists(assembly_path) &&
			load_assembly_from(EDITOR_API_ASSEMBLY_NAME, assembly_path, &r_loaded_api_assembly.assembly, p_refonly);

	if (success) {
		ApiAssemblyInfo::Version api_assembly_ver = ApiAssemblyInfo::Version::get_from_loaded_assembly(r_loaded_api_assembly.assembly, ApiAssemblyInfo::API_EDITOR);
		r_loaded_api_assembly.out_of_sync = GodotSharpBindings::get_editor_api_hash() != api_assembly_ver.godot_api_hash ||
				GodotSharpBindings::get_bindings_version() != api_assembly_ver.bindings_version ||
				GodotSharpBindings::get_cs_glue_version() != api_assembly_ver.cs_glue_version;
	} else {
		r_loaded_api_assembly.out_of_sync = false;
	}

	return success;
}
#endif

bool GDMono::_try_load_api_assemblies(LoadedApiAssembly &r_core_api_assembly, LoadedApiAssembly &r_editor_api_assembly,
		const String &p_config, bool p_refonly, CoreApiAssemblyLoadedCallback p_callback) {
	if (!_load_core_api_assembly(r_core_api_assembly, p_config, p_refonly)) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			print_error("Mono: Failed to load Core API assembly");
		}
		return false;
	}

#ifdef TOOLS_ENABLED
	if (!_load_editor_api_assembly(r_editor_api_assembly, p_config, p_refonly)) {
		if (OS::get_singleton()->is_stdout_verbose()) {
			print_error("Mono: Failed to load Editor API assembly");
		}
		return false;
	}

	if (r_editor_api_assembly.out_of_sync) {
		return false;
	}
#endif

	// Check if the core API assembly is out of sync only after trying to load the
	// editor API assembly. Otherwise, if both assemblies are out of sync, we would
	// only update the former as we won't know the latter also needs to be updated.
	if (r_core_api_assembly.out_of_sync) {
		return false;
	}

	if (p_callback) {
		return p_callback();
	}

	return true;
}

bool GDMono::_on_core_api_assembly_loaded() {
	GDMonoCache::update_godot_api_cache();

	if (!GDMonoCache::cached_data.godot_api_cache_updated) {
		return false;
	}

	get_singleton()->_install_trace_listener();

	return true;
}

bool GDMono::_try_load_api_assemblies_preset() {
	return _try_load_api_assemblies(core_api_assembly, editor_api_assembly,
			get_expected_api_build_config(), /* refonly: */ false, _on_core_api_assembly_loaded);
}

void GDMono::_load_api_assemblies() {
	bool api_assemblies_loaded = _try_load_api_assemblies_preset();

#if defined(TOOLS_ENABLED) && !defined(GD_MONO_SINGLE_APPDOMAIN)
	if (!api_assemblies_loaded) {
		// The API assemblies are out of sync or some other error happened. Fine, try one more time, but
		// this time update them from the prebuilt assemblies directory before trying to load them again.

		// Shouldn't happen. The project manager loads the prebuilt API assemblies
		CRASH_COND_MSG(Main::is_project_manager(), "Failed to load one of the prebuilt API assemblies.");

		// 1. Unload the scripts domain
		Error domain_unload_err = _unload_scripts_domain();
		CRASH_COND_MSG(domain_unload_err != OK, "Mono: Failed to unload scripts domain.");

		// 2. Update the API assemblies
		String update_error = update_api_assemblies_from_prebuilt("Debug", &core_api_assembly.out_of_sync, &editor_api_assembly.out_of_sync);
		CRASH_COND_MSG(!update_error.is_empty(), update_error);

		// 3. Load the scripts domain again
		Error domain_load_err = _load_scripts_domain();
		CRASH_COND_MSG(domain_load_err != OK, "Mono: Failed to load scripts domain.");

		// 4. Try loading the updated assemblies
		api_assemblies_loaded = _try_load_api_assemblies_preset();
	}
#endif

	if (!api_assemblies_loaded) {
		// welp... too bad

		if (_are_api_assemblies_out_of_sync()) {
			if (core_api_assembly.out_of_sync) {
				ERR_PRINT("The assembly '" CORE_API_ASSEMBLY_NAME "' is out of sync.");
			} else if (!GDMonoCache::cached_data.godot_api_cache_updated) {
				ERR_PRINT("The loaded assembly '" CORE_API_ASSEMBLY_NAME "' is in sync, but the cache update failed.");
			}

#ifdef TOOLS_ENABLED
			if (editor_api_assembly.out_of_sync) {
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
	if (tools_assembly && tools_project_editor_assembly) {
		return true;
	}

	bool success = load_assembly(TOOLS_ASM_NAME, &tools_assembly) &&
			load_assembly(TOOLS_PROJECT_EDITOR_ASM_NAME, &tools_project_editor_assembly);

	return success;
}
#endif

bool GDMono::_load_project_assembly() {
	if (project_assembly) {
		return true;
	}

	String appname = ProjectSettings::get_singleton()->get("application/config/name");
	String appname_safe = OS::get_singleton()->get_safe_dir_name(appname);
	if (appname_safe.is_empty()) {
		appname_safe = "UnnamedProject";
	}

	bool success = load_assembly(appname_safe, &project_assembly);

	if (success) {
		mono_assembly_set_main(project_assembly->get_assembly());
		CSharpLanguage::get_singleton()->lookup_scripts_in_assembly(project_assembly);
	}

	return success;
}

void GDMono::_install_trace_listener() {
#ifdef DEBUG_ENABLED
	// Install the trace listener now before the project assembly is loaded
	GDMonoClass *debug_utils = get_core_api_assembly()->get_class(BINDINGS_NAMESPACE, "DebuggingUtils");
	GDMonoMethod *install_func = debug_utils->get_method("InstallTraceListener");

	MonoException *exc = nullptr;
	install_func->invoke_raw(nullptr, nullptr, &exc);
	if (exc) {
		GDMonoUtils::debug_print_unhandled_exception(exc);
		ERR_PRINT("Failed to install 'System.Diagnostics.Trace' listener.");
	}
#endif
}

#ifndef GD_MONO_SINGLE_APPDOMAIN
Error GDMono::_load_scripts_domain() {
	ERR_FAIL_COND_V(scripts_domain != nullptr, ERR_BUG);

	print_verbose("Mono: Loading scripts domain...");

	scripts_domain = GDMonoUtils::create_domain("GodotEngine.Domain.Scripts");

	ERR_FAIL_NULL_V_MSG(scripts_domain, ERR_CANT_CREATE, "Mono: Could not create scripts app domain.");

	mono_domain_set(scripts_domain, true);

	return OK;
}

Error GDMono::_unload_scripts_domain() {
	ERR_FAIL_NULL_V(scripts_domain, ERR_BUG);

	print_verbose("Mono: Finalizing scripts domain...");

	if (mono_domain_get() != root_domain) {
		mono_domain_set(root_domain, true);
	}

	finalizing_scripts_domain = true;

	if (!mono_domain_finalize(scripts_domain, 2000)) {
		ERR_PRINT("Mono: Domain finalization timeout.");
	}

	finalizing_scripts_domain = false;

	mono_gc_collect(mono_gc_max_generation());

	GDMonoCache::clear_godot_api_cache();

	_domain_assemblies_cleanup(mono_domain_get_id(scripts_domain));

	core_api_assembly.assembly = nullptr;
#ifdef TOOLS_ENABLED
	editor_api_assembly.assembly = nullptr;
#endif

	project_assembly = nullptr;
#ifdef TOOLS_ENABLED
	tools_assembly = nullptr;
	tools_project_editor_assembly = nullptr;
#endif

	MonoDomain *domain = scripts_domain;
	scripts_domain = nullptr;

	print_verbose("Mono: Unloading scripts domain...");

	MonoException *exc = nullptr;
	mono_domain_try_unload(domain, (MonoObject **)&exc);

	if (exc) {
		ERR_PRINT("Exception thrown when unloading scripts domain.");
		GDMonoUtils::debug_unhandled_exception(exc);
		return FAILED;
	}

	return OK;
}
#endif

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

#ifndef GD_MONO_SINGLE_APPDOMAIN
Error GDMono::finalize_and_unload_domain(MonoDomain *p_domain) {
	CRASH_COND(p_domain == nullptr);
	CRASH_COND(p_domain == GDMono::get_singleton()->get_scripts_domain()); // Should use _unload_scripts_domain() instead

	String domain_name = mono_domain_get_friendly_name(p_domain);

	print_verbose("Mono: Unloading domain '" + domain_name + "'...");

	if (mono_domain_get() == p_domain) {
		mono_domain_set(root_domain, true);
	}

	if (!mono_domain_finalize(p_domain, 2000)) {
		ERR_PRINT("Mono: Domain finalization timeout.");
	}

	mono_gc_collect(mono_gc_max_generation());

	_domain_assemblies_cleanup(mono_domain_get_id(p_domain));

	MonoException *exc = nullptr;
	mono_domain_try_unload(p_domain, (MonoObject **)&exc);

	if (exc) {
		ERR_PRINT("Exception thrown when unloading domain '" + domain_name + "'.");
		GDMonoUtils::debug_print_unhandled_exception(exc);
		return FAILED;
	}

	return OK;
}
#endif

GDMonoClass *GDMono::get_class(MonoClass *p_raw_class) {
	MonoImage *image = mono_class_get_image(p_raw_class);

	if (image == corlib_assembly->get_image()) {
		return corlib_assembly->get_class(p_raw_class);
	}

	int32_t domain_id = mono_domain_get_id(mono_domain_get());
	HashMap<String, GDMonoAssembly *> &domain_assemblies = assemblies[domain_id];

	const String *k = nullptr;
	while ((k = domain_assemblies.next(k))) {
		GDMonoAssembly *assembly = domain_assemblies.get(*k);
		if (assembly->get_image() == image) {
			GDMonoClass *klass = assembly->get_class(p_raw_class);
			if (klass) {
				return klass;
			}
		}
	}

	return nullptr;
}

GDMonoClass *GDMono::get_class(const StringName &p_namespace, const StringName &p_name) {
	GDMonoClass *klass = corlib_assembly->get_class(p_namespace, p_name);
	if (klass) {
		return klass;
	}

	int32_t domain_id = mono_domain_get_id(mono_domain_get());
	HashMap<String, GDMonoAssembly *> &domain_assemblies = assemblies[domain_id];

	const String *k = nullptr;
	while ((k = domain_assemblies.next(k))) {
		GDMonoAssembly *assembly = domain_assemblies.get(*k);
		klass = assembly->get_class(p_namespace, p_name);
		if (klass) {
			return klass;
		}
	}

	return nullptr;
}

void GDMono::_domain_assemblies_cleanup(int32_t p_domain_id) {
	HashMap<String, GDMonoAssembly *> &domain_assemblies = assemblies[p_domain_id];

	const String *k = nullptr;
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
	if (EngineDebugger::is_active()) {
		EngineDebugger::get_singleton()->poll_events(false);
	}
#endif

	exit(mono_environment_exitcode_get());

	GD_UNREACHABLE();
}

GDMono::GDMono() {
	singleton = this;

	gdmono_log = memnew(GDMonoLog);

	runtime_initialized = false;
	finalizing_scripts_domain = false;

	root_domain = nullptr;
	scripts_domain = nullptr;

	corlib_assembly = nullptr;
	project_assembly = nullptr;
#ifdef TOOLS_ENABLED
	tools_assembly = nullptr;
	tools_project_editor_assembly = nullptr;
#endif

	api_core_hash = 0;
#ifdef TOOLS_ENABLED
	api_editor_hash = 0;
#endif

	unhandled_exception_policy = POLICY_TERMINATE_APP;
}

GDMono::~GDMono() {
	if (is_runtime_initialized()) {
#ifndef GD_MONO_SINGLE_APPDOMAIN
		if (scripts_domain) {
			Error err = _unload_scripts_domain();
			if (err != OK) {
				ERR_PRINT("Mono: Failed to unload scripts domain.");
			}
		}
#else
		CRASH_COND(scripts_domain != root_domain);

		print_verbose("Mono: Finalizing scripts domain...");

		if (mono_domain_get() != root_domain)
			mono_domain_set(root_domain, true);

		finalizing_scripts_domain = true;

		if (!mono_domain_finalize(root_domain, 2000)) {
			ERR_PRINT("Mono: Domain finalization timeout.");
		}

		finalizing_scripts_domain = false;

		mono_gc_collect(mono_gc_max_generation());

		GDMonoCache::clear_godot_api_cache();

		_domain_assemblies_cleanup(mono_domain_get_id(root_domain));

		core_api_assembly.assembly = nullptr;

		project_assembly = nullptr;

		root_domain = nullptr;
		scripts_domain = nullptr;

		// Leave the rest to 'mono_jit_cleanup'
#endif

		const int32_t *k = nullptr;
		while ((k = assemblies.next(k))) {
			HashMap<String, GDMonoAssembly *> &domain_assemblies = assemblies.get(*k);

			const String *kk = nullptr;
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

#if defined(ANDROID_ENABLED)
	gdmono::android::support::cleanup();
#endif

	if (gdmono_log) {
		memdelete(gdmono_log);
	}

	singleton = nullptr;
}

namespace mono_bind {

GodotSharp *GodotSharp::singleton = nullptr;

void GodotSharp::attach_thread() {
	GDMonoUtils::attach_current_thread();
}

void GodotSharp::detach_thread() {
	GDMonoUtils::detach_current_thread();
}

int32_t GodotSharp::get_domain_id() {
	MonoDomain *domain = mono_domain_get();
	ERR_FAIL_NULL_V(domain, -1);
	return mono_domain_get_id(domain);
}

int32_t GodotSharp::get_scripts_domain_id() {
	ERR_FAIL_NULL_V_MSG(GDMono::get_singleton(),
			-1, "The Mono runtime is not initialized");
	MonoDomain *domain = GDMono::get_singleton()->get_scripts_domain();
	ERR_FAIL_NULL_V(domain, -1);
	return mono_domain_get_id(domain);
}

bool GodotSharp::is_scripts_domain_loaded() {
	return GDMono::get_singleton() != nullptr &&
			GDMono::get_singleton()->is_runtime_initialized() &&
			GDMono::get_singleton()->get_scripts_domain() != nullptr;
}

bool GodotSharp::_is_domain_finalizing_for_unload(int32_t p_domain_id) {
	return is_domain_finalizing_for_unload(p_domain_id);
}

bool GodotSharp::is_domain_finalizing_for_unload(int32_t p_domain_id) {
	return is_domain_finalizing_for_unload(mono_domain_get_by_id(p_domain_id));
}

bool GodotSharp::is_domain_finalizing_for_unload(MonoDomain *p_domain) {
	GDMono *gd_mono = GDMono::get_singleton();

	ERR_FAIL_COND_V_MSG(!gd_mono || !gd_mono->is_runtime_initialized(),
			false, "The Mono runtime is not initialized");

	ERR_FAIL_NULL_V(p_domain, true);

	if (p_domain == gd_mono->get_scripts_domain() && gd_mono->is_finalizing_scripts_domain()) {
		return true;
	}

	return mono_domain_is_unloading(p_domain);
}

bool GodotSharp::is_runtime_shutting_down() {
	return mono_runtime_is_shutting_down();
}

bool GodotSharp::is_runtime_initialized() {
	return GDMono::get_singleton() != nullptr && GDMono::get_singleton()->is_runtime_initialized();
}

void GodotSharp::_reload_assemblies(bool p_soft_reload) {
#ifdef GD_MONO_HOT_RELOAD
	CRASH_COND(CSharpLanguage::get_singleton() == nullptr);
	// This method may be called more than once with `call_deferred`, so we need to check
	// again if reloading is needed to avoid reloading multiple times unnecessarily.
	if (CSharpLanguage::get_singleton()->is_assembly_reloading_needed()) {
		CSharpLanguage::get_singleton()->reload_assemblies(p_soft_reload);
	}
#endif
}

void GodotSharp::_bind_methods() {
	ClassDB::bind_method(D_METHOD("attach_thread"), &GodotSharp::attach_thread);
	ClassDB::bind_method(D_METHOD("detach_thread"), &GodotSharp::detach_thread);

	ClassDB::bind_method(D_METHOD("get_domain_id"), &GodotSharp::get_domain_id);
	ClassDB::bind_method(D_METHOD("get_scripts_domain_id"), &GodotSharp::get_scripts_domain_id);
	ClassDB::bind_method(D_METHOD("is_scripts_domain_loaded"), &GodotSharp::is_scripts_domain_loaded);
	ClassDB::bind_method(D_METHOD("is_domain_finalizing_for_unload", "domain_id"), &GodotSharp::_is_domain_finalizing_for_unload);

	ClassDB::bind_method(D_METHOD("is_runtime_shutting_down"), &GodotSharp::is_runtime_shutting_down);
	ClassDB::bind_method(D_METHOD("is_runtime_initialized"), &GodotSharp::is_runtime_initialized);
	ClassDB::bind_method(D_METHOD("_reload_assemblies"), &GodotSharp::_reload_assemblies);
}

GodotSharp::GodotSharp() {
	singleton = this;
}

GodotSharp::~GodotSharp() {
	singleton = nullptr;
}

} // namespace mono_bind
