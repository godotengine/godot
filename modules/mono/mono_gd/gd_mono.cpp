/**************************************************************************/
/*  gd_mono.cpp                                                           */
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

#include "gd_mono.h"

#include "../csharp_script.h"
#include "../glue/runtime_interop.h"
#include "../godotsharp_dirs.h"
#include "../thirdparty/coreclr_delegates.h"
#include "../thirdparty/hostfxr.h"
#include "../utils/path_utils.h"
#include "gd_mono_cache.h"

#ifdef TOOLS_ENABLED
#include "../editor/hostfxr_resolver.h"
#include "../editor/semver.h"
#endif

#include "core/config/project_settings.h"
#include "core/debugger/engine_debugger.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/os.h"
#include "core/os/thread.h"

#ifdef UNIX_ENABLED
#include <dlfcn.h>
#endif

#ifndef TOOLS_ENABLED
#ifdef ANDROID_ENABLED
#include "../thirdparty/mono_delegates.h"
#endif
#endif

GDMono *GDMono::singleton = nullptr;

namespace {
hostfxr_initialize_for_dotnet_command_line_fn hostfxr_initialize_for_dotnet_command_line = nullptr;
hostfxr_initialize_for_runtime_config_fn hostfxr_initialize_for_runtime_config = nullptr;
hostfxr_get_runtime_delegate_fn hostfxr_get_runtime_delegate = nullptr;
hostfxr_close_fn hostfxr_close = nullptr;

#ifndef TOOLS_ENABLED
typedef int(CORECLR_DELEGATE_CALLTYPE *coreclr_create_delegate_fn)(void *hostHandle, unsigned int domainId, const char *entryPointAssemblyName, const char *entryPointTypeName, const char *entryPointMethodName, void **delegate);
typedef int(CORECLR_DELEGATE_CALLTYPE *coreclr_initialize_fn)(const char *exePath, const char *appDomainFriendlyName, int propertyCount, const char **propertyKeys, const char **propertyValues, void **hostHandle, unsigned int *domainId);

coreclr_create_delegate_fn coreclr_create_delegate = nullptr;
coreclr_initialize_fn coreclr_initialize = nullptr;

#ifdef ANDROID_ENABLED
mono_install_assembly_preload_hook_fn mono_install_assembly_preload_hook = nullptr;
mono_assembly_name_get_name_fn mono_assembly_name_get_name = nullptr;
mono_assembly_name_get_culture_fn mono_assembly_name_get_culture = nullptr;
mono_image_open_from_data_with_name_fn mono_image_open_from_data_with_name = nullptr;
mono_assembly_load_from_full_fn mono_assembly_load_from_full = nullptr;
#endif
#endif

#ifdef _WIN32
static_assert(sizeof(char_t) == sizeof(char16_t));
using HostFxrCharString = Char16String;
#define HOSTFXR_STR(m_str) L##m_str
#else
static_assert(sizeof(char_t) == sizeof(char));
using HostFxrCharString = CharString;
#define HOSTFXR_STR(m_str) m_str
#endif

HostFxrCharString str_to_hostfxr(const String &p_str) {
#ifdef _WIN32
	return p_str.utf16();
#else
	return p_str.utf8();
#endif
}

const char_t *get_data(const HostFxrCharString &p_char_str) {
	return (const char_t *)p_char_str.get_data();
}

#ifdef TOOLS_ENABLED
bool try_get_dotnet_root_from_command_line(String &r_dotnet_root) {
	String pipe;
	List<String> args;
	args.push_back("--list-sdks");

	int exitcode;
	Error err = OS::get_singleton()->execute("dotnet", args, &pipe, &exitcode, true);

	ERR_FAIL_COND_V_MSG(err != OK, false, String(".NET failed to get list of installed SDKs. Error: ") + error_names[err]);
	ERR_FAIL_COND_V_MSG(exitcode != 0, false, pipe);

	Vector<String> sdks = pipe.strip_edges().replace("\r\n", "\n").split("\n", false);

	godotsharp::SemVerParser sem_ver_parser;

	godotsharp::SemVer latest_sdk_version;
	String latest_sdk_path;

	for (const String &sdk : sdks) {
		// The format of the SDK lines is:
		// 8.0.401 [/usr/share/dotnet/sdk]
		String version_string = sdk.get_slice(" ", 0);
		String path = sdk.get_slice(" ", 1);
		path = path.substr(1, path.length() - 2);

		godotsharp::SemVer version;
		if (!sem_ver_parser.parse(version_string, version)) {
			WARN_PRINT("Unable to parse .NET SDK version '" + version_string + "'.");
			continue;
		}

		if (!DirAccess::exists(path)) {
			WARN_PRINT("Found .NET SDK version '" + version_string + "' with invalid path '" + path + "'.");
			continue;
		}

		if (version > latest_sdk_version) {
			latest_sdk_version = version;
			latest_sdk_path = path;
		}
	}

	if (!latest_sdk_path.is_empty()) {
		print_verbose("Found .NET SDK at " + latest_sdk_path);
		// The `dotnet_root` is the parent directory.
		r_dotnet_root = latest_sdk_path.path_join("..").simplify_path();
		return true;
	}

	return false;
}
#endif

String find_hostfxr() {
#ifdef TOOLS_ENABLED
	String dotnet_root;
	String fxr_path;
	if (godotsharp::hostfxr_resolver::try_get_path(dotnet_root, fxr_path)) {
		return fxr_path;
	}

	// hostfxr_resolver doesn't look for dotnet in `PATH`. If it fails, we try to use the dotnet
	// executable in `PATH` to find the `dotnet_root` and get the `hostfxr_path` from there.
	if (try_get_dotnet_root_from_command_line(dotnet_root)) {
		if (godotsharp::hostfxr_resolver::try_get_path_from_dotnet_root(dotnet_root, fxr_path)) {
			return fxr_path;
		}
	}

	ERR_PRINT(String() + ".NET: One of the dependent libraries is missing. " +
			"Typically when the `hostfxr`, `hostpolicy` or `coreclr` dynamic " +
			"libraries are not present in the expected locations.");

	return String();
#else

#if defined(WINDOWS_ENABLED)
	String probe_path = GodotSharpDirs::get_api_assemblies_dir()
								.path_join("hostfxr.dll");
#elif defined(MACOS_ENABLED)
	String probe_path = GodotSharpDirs::get_api_assemblies_dir()
								.path_join("libhostfxr.dylib");
#elif defined(UNIX_ENABLED)
	String probe_path = GodotSharpDirs::get_api_assemblies_dir()
								.path_join("libhostfxr.so");
#else
#error "Platform not supported (yet?)"
#endif

	if (FileAccess::exists(probe_path)) {
		return probe_path;
	}

	return String();

#endif
}

#ifndef TOOLS_ENABLED
String find_monosgen() {
#if defined(ANDROID_ENABLED)
	// Android includes all native libraries in the libs directory of the APK
	// so we assume it exists and use only the name to dlopen it.
	return "libmonosgen-2.0.so";
#else
#if defined(WINDOWS_ENABLED)
	String probe_path = GodotSharpDirs::get_api_assemblies_dir()
								.path_join("monosgen-2.0.dll");
#elif defined(MACOS_ENABLED)
	String probe_path = GodotSharpDirs::get_api_assemblies_dir()
								.path_join("libmonosgen-2.0.dylib");
#elif defined(UNIX_ENABLED)
	String probe_path = GodotSharpDirs::get_api_assemblies_dir()
								.path_join("libmonosgen-2.0.so");
#else
#error "Platform not supported (yet?)"
#endif

	if (FileAccess::exists(probe_path)) {
		return probe_path;
	}

	return String();
#endif
}

String find_coreclr() {
#if defined(WINDOWS_ENABLED)
	String probe_path = GodotSharpDirs::get_api_assemblies_dir()
								.path_join("coreclr.dll");
#elif defined(MACOS_ENABLED)
	String probe_path = GodotSharpDirs::get_api_assemblies_dir()
								.path_join("libcoreclr.dylib");
#elif defined(UNIX_ENABLED)
	String probe_path = GodotSharpDirs::get_api_assemblies_dir()
								.path_join("libcoreclr.so");
#else
#error "Platform not supported (yet?)"
#endif

	if (FileAccess::exists(probe_path)) {
		return probe_path;
	}

	return String();
}
#endif

bool load_hostfxr(void *&r_hostfxr_dll_handle) {
	String hostfxr_path = find_hostfxr();

	if (hostfxr_path.is_empty()) {
		return false;
	}

	print_verbose("Found hostfxr: " + hostfxr_path);

	Error err = OS::get_singleton()->open_dynamic_library(hostfxr_path, r_hostfxr_dll_handle);

	if (err != OK) {
		return false;
	}

	void *lib = r_hostfxr_dll_handle;

	void *symbol = nullptr;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "hostfxr_initialize_for_dotnet_command_line", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	hostfxr_initialize_for_dotnet_command_line = (hostfxr_initialize_for_dotnet_command_line_fn)symbol;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "hostfxr_initialize_for_runtime_config", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	hostfxr_initialize_for_runtime_config = (hostfxr_initialize_for_runtime_config_fn)symbol;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "hostfxr_get_runtime_delegate", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	hostfxr_get_runtime_delegate = (hostfxr_get_runtime_delegate_fn)symbol;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "hostfxr_close", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	hostfxr_close = (hostfxr_close_fn)symbol;

	return (hostfxr_initialize_for_runtime_config &&
			hostfxr_get_runtime_delegate &&
			hostfxr_close);
}

#ifndef TOOLS_ENABLED
bool load_coreclr(void *&r_coreclr_dll_handle) {
	String coreclr_path = find_coreclr();

	bool is_monovm = false;
	if (coreclr_path.is_empty()) {
		// Fallback to MonoVM (should have the same API as CoreCLR).
		coreclr_path = find_monosgen();
		is_monovm = true;
	}

	if (coreclr_path.is_empty()) {
		return false;
	}

	const String coreclr_name = is_monovm ? "monosgen" : "coreclr";
	print_verbose("Found " + coreclr_name + ": " + coreclr_path);

	Error err = OS::get_singleton()->open_dynamic_library(coreclr_path, r_coreclr_dll_handle);

	if (err != OK) {
		return false;
	}

	void *lib = r_coreclr_dll_handle;

	void *symbol = nullptr;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "coreclr_initialize", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	coreclr_initialize = (coreclr_initialize_fn)symbol;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "coreclr_create_delegate", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	coreclr_create_delegate = (coreclr_create_delegate_fn)symbol;

#ifdef ANDROID_ENABLED
	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "mono_install_assembly_preload_hook", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	mono_install_assembly_preload_hook = (mono_install_assembly_preload_hook_fn)symbol;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "mono_assembly_name_get_name", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	mono_assembly_name_get_name = (mono_assembly_name_get_name_fn)symbol;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "mono_assembly_name_get_culture", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	mono_assembly_name_get_culture = (mono_assembly_name_get_culture_fn)symbol;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "mono_image_open_from_data_with_name", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	mono_image_open_from_data_with_name = (mono_image_open_from_data_with_name_fn)symbol;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(lib, "mono_assembly_load_from_full", symbol);
	ERR_FAIL_COND_V(err != OK, false);
	mono_assembly_load_from_full = (mono_assembly_load_from_full_fn)symbol;
#endif

	return (coreclr_initialize &&
			coreclr_create_delegate);
}
#endif

#ifdef TOOLS_ENABLED
load_assembly_and_get_function_pointer_fn initialize_hostfxr_for_config(const char_t *p_config_path) {
	hostfxr_handle cxt = nullptr;
	int rc = hostfxr_initialize_for_runtime_config(p_config_path, nullptr, &cxt);
	if (rc != 0 || cxt == nullptr) {
		hostfxr_close(cxt);
		ERR_FAIL_V_MSG(nullptr, "hostfxr_initialize_for_runtime_config failed with code: " + itos(rc));
	}

	void *load_assembly_and_get_function_pointer = nullptr;

	rc = hostfxr_get_runtime_delegate(cxt,
			hdt_load_assembly_and_get_function_pointer, &load_assembly_and_get_function_pointer);
	if (rc != 0 || load_assembly_and_get_function_pointer == nullptr) {
		ERR_FAIL_V_MSG(nullptr, "hostfxr_get_runtime_delegate failed with code: " + itos(rc));
	}

	hostfxr_close(cxt);

	return (load_assembly_and_get_function_pointer_fn)load_assembly_and_get_function_pointer;
}
#else
load_assembly_and_get_function_pointer_fn initialize_hostfxr_self_contained(
		const char_t *p_main_assembly_path) {
	hostfxr_handle cxt = nullptr;

	List<String> cmdline_args = OS::get_singleton()->get_cmdline_args();

	List<HostFxrCharString> argv_store;
	Vector<const char_t *> argv;
	argv.resize(cmdline_args.size() + 1);

	argv.write[0] = p_main_assembly_path;

	int i = 1;
	for (const String &E : cmdline_args) {
		HostFxrCharString &stored = argv_store.push_back(str_to_hostfxr(E))->get();
		argv.write[i] = get_data(stored);
		i++;
	}

	int rc = hostfxr_initialize_for_dotnet_command_line(argv.size(), argv.ptrw(), nullptr, &cxt);
	if (rc != 0 || cxt == nullptr) {
		hostfxr_close(cxt);
		ERR_FAIL_V_MSG(nullptr, "hostfxr_initialize_for_dotnet_command_line failed with code: " + itos(rc));
	}

	void *load_assembly_and_get_function_pointer = nullptr;

	rc = hostfxr_get_runtime_delegate(cxt,
			hdt_load_assembly_and_get_function_pointer, &load_assembly_and_get_function_pointer);
	if (rc != 0 || load_assembly_and_get_function_pointer == nullptr) {
		ERR_FAIL_V_MSG(nullptr, "hostfxr_get_runtime_delegate failed with code: " + itos(rc));
	}

	hostfxr_close(cxt);

	return (load_assembly_and_get_function_pointer_fn)load_assembly_and_get_function_pointer;
}
#endif

#ifdef TOOLS_ENABLED
using godot_plugins_initialize_fn = bool (*)(void *, bool, gdmono::PluginCallbacks *, GDMonoCache::ManagedCallbacks *, const void **, int32_t);
#else
using godot_plugins_initialize_fn = bool (*)(void *, GDMonoCache::ManagedCallbacks *, const void **, int32_t);
#endif

#ifdef TOOLS_ENABLED
godot_plugins_initialize_fn initialize_hostfxr_and_godot_plugins(bool &r_runtime_initialized) {
	godot_plugins_initialize_fn godot_plugins_initialize = nullptr;

	HostFxrCharString godot_plugins_path = str_to_hostfxr(
			GodotSharpDirs::get_api_assemblies_dir().path_join("GodotPlugins.dll"));

	HostFxrCharString config_path = str_to_hostfxr(
			GodotSharpDirs::get_api_assemblies_dir().path_join("GodotPlugins.runtimeconfig.json"));

	load_assembly_and_get_function_pointer_fn load_assembly_and_get_function_pointer =
			initialize_hostfxr_for_config(get_data(config_path));

	if (load_assembly_and_get_function_pointer == nullptr) {
		// Show a message box to the user to make the problem explicit (and explain a potential crash).
		OS::get_singleton()->alert(TTR("Unable to load .NET runtime, no compatible version was found.\nAttempting to create/edit a project will lead to a crash.\n\nPlease install the .NET SDK 8.0 or later from https://get.dot.net and restart Godot."), TTR("Failed to load .NET runtime"));
		ERR_FAIL_V_MSG(nullptr, ".NET: Failed to load compatible .NET runtime");
	}

	r_runtime_initialized = true;

	print_verbose(".NET: hostfxr initialized");

	int rc = load_assembly_and_get_function_pointer(get_data(godot_plugins_path),
			HOSTFXR_STR("GodotPlugins.Main, GodotPlugins"),
			HOSTFXR_STR("InitializeFromEngine"),
			UNMANAGEDCALLERSONLY_METHOD,
			nullptr,
			(void **)&godot_plugins_initialize);
	ERR_FAIL_COND_V_MSG(rc != 0, nullptr, ".NET: Failed to get GodotPlugins initialization function pointer");

	return godot_plugins_initialize;
}
#else
godot_plugins_initialize_fn initialize_hostfxr_and_godot_plugins(bool &r_runtime_initialized) {
	godot_plugins_initialize_fn godot_plugins_initialize = nullptr;

	String assembly_name = Path::get_csharp_project_name();

	HostFxrCharString assembly_path = str_to_hostfxr(GodotSharpDirs::get_api_assemblies_dir()
					.path_join(assembly_name + ".dll"));

	load_assembly_and_get_function_pointer_fn load_assembly_and_get_function_pointer =
			initialize_hostfxr_self_contained(get_data(assembly_path));
	ERR_FAIL_NULL_V(load_assembly_and_get_function_pointer, nullptr);
}
#endif

void GDMono::initialize(bool p_tools, const String &p_resource_dir_path) {
	print_verbose("Mono: Initializing module...");

	_tools = p_tools;

	assembly_loaded = nullptr;
	set_up_diagnostics_url_builder = nullptr;
	install_diagnostics_trace_listener = nullptr;

	log_exceptions_enabled = Engine::get_singleton()->is_generate_objc_bindings_enabled();

#ifdef WEB_ENABLED
	// For web platform, we need to set up the Mono runtime differently
	// The runtime is statically linked in this case
	print_verbose("Mono: Static linking for web platform.");
	runtime_initialized = true;
	_domain = nullptr;

	mono_jit_init("godot");
	
	_load_assemblies(p_resource_dir_path);

	return;
#endif

	// Rest of the initialization code for non-web platforms goes here

	initialized = true;
}

GDMono::GDMono() {
	singleton = this;
}

GDMono::~GDMono() {
	singleton = nullptr;
}
