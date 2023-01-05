/**************************************************************************/
/*  godotsharp_dirs.cpp                                                   */
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

#include "godotsharp_dirs.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/os/os.h"

#ifdef TOOLS_ENABLED
#include "core/version.h"
#include "editor/editor_paths.h"
#endif

#ifdef ANDROID_ENABLED
#include "mono_gd/support/android_support.h"
#endif

#include "mono_gd/gd_mono.h"

namespace GodotSharpDirs {

String _get_expected_build_config() {
#ifdef TOOLS_ENABLED
	return "Debug";
#else

#ifdef DEBUG_ENABLED
	return "ExportDebug";
#else
	return "ExportRelease";
#endif

#endif
}

String _get_mono_user_dir() {
#ifdef TOOLS_ENABLED
	if (EditorPaths::get_singleton()) {
		return EditorPaths::get_singleton()->get_data_dir().path_join("mono");
	} else {
		String settings_path;

		// Self-contained mode if a `._sc_` or `_sc_` file is present in executable dir.
		String exe_dir = OS::get_singleton()->get_executable_path().get_base_dir();

		// On macOS, look outside .app bundle, since .app bundle is read-only.
		if (OS::get_singleton()->has_feature("macos") && exe_dir.ends_with("MacOS") && exe_dir.path_join("..").simplify_path().ends_with("Contents")) {
			exe_dir = exe_dir.path_join("../../..").simplify_path();
		}

		Ref<DirAccess> d = DirAccess::create_for_path(exe_dir);

		if (d->file_exists("._sc_") || d->file_exists("_sc_")) {
			// contain yourself
			settings_path = exe_dir.path_join("editor_data");
		} else {
			settings_path = OS::get_singleton()->get_data_path().path_join(OS::get_singleton()->get_godot_dir_name());
		}

		return settings_path.path_join("mono");
	}
#else
	return OS::get_singleton()->get_user_data_dir().path_join("mono");
#endif
}

class _GodotSharpDirs {
public:
	String res_metadata_dir;
	String res_temp_assemblies_dir;
	String mono_user_dir;
	String api_assemblies_dir;

#ifdef TOOLS_ENABLED
	String build_logs_dir;
	String data_editor_tools_dir;
#endif

private:
	_GodotSharpDirs() {
		String res_data_dir = ProjectSettings::get_singleton()->get_project_data_path().path_join("mono");
		res_metadata_dir = res_data_dir.path_join("metadata");

		// TODO use paths from csproj
		res_temp_assemblies_dir = res_data_dir.path_join("temp").path_join("bin").path_join(_get_expected_build_config());

#ifdef WEB_ENABLED
		mono_user_dir = "user://";
#else
		mono_user_dir = _get_mono_user_dir();
#endif

		String exe_dir = OS::get_singleton()->get_executable_path().get_base_dir();
		String res_dir = OS::get_singleton()->get_bundle_resource_dir();

#ifdef TOOLS_ENABLED
		String data_dir_root = exe_dir.path_join("GodotSharp");
		data_editor_tools_dir = data_dir_root.path_join("Tools");
		String api_assemblies_base_dir = data_dir_root.path_join("Api");
		build_logs_dir = mono_user_dir.path_join("build_logs");
#ifdef MACOS_ENABLED
		if (!DirAccess::exists(data_editor_tools_dir)) {
			data_editor_tools_dir = res_dir.path_join("GodotSharp").path_join("Tools");
		}
		if (!DirAccess::exists(api_assemblies_base_dir)) {
			api_assemblies_base_dir = res_dir.path_join("GodotSharp").path_join("Api");
		}
#endif
		api_assemblies_dir = api_assemblies_base_dir.path_join(GDMono::get_expected_api_build_config());
#else // TOOLS_ENABLED
		String arch = Engine::get_singleton()->get_architecture_name();
		String appname = GLOBAL_GET("application/config/name");
		String appname_safe = OS::get_singleton()->get_safe_dir_name(appname);
		String data_dir_root = exe_dir.path_join("data_" + appname_safe + "_" + arch);
		if (!DirAccess::exists(data_dir_root)) {
			data_dir_root = exe_dir.path_join("data_Godot_" + arch);
		}
#ifdef MACOS_ENABLED
		if (!DirAccess::exists(data_dir_root)) {
			data_dir_root = res_dir.path_join("data_" + appname_safe + "_" + arch);
		}
		if (!DirAccess::exists(data_dir_root)) {
			data_dir_root = res_dir.path_join("data_Godot_" + arch);
		}
#endif
		api_assemblies_dir = data_dir_root;
#endif
	}

public:
	static _GodotSharpDirs &get_singleton() {
		static _GodotSharpDirs singleton;
		return singleton;
	}
};

String get_res_metadata_dir() {
	return _GodotSharpDirs::get_singleton().res_metadata_dir;
}

String get_res_temp_assemblies_dir() {
	return _GodotSharpDirs::get_singleton().res_temp_assemblies_dir;
}

String get_api_assemblies_dir() {
	return _GodotSharpDirs::get_singleton().api_assemblies_dir;
}

String get_mono_user_dir() {
	return _GodotSharpDirs::get_singleton().mono_user_dir;
}

#ifdef TOOLS_ENABLED
String get_build_logs_dir() {
	return _GodotSharpDirs::get_singleton().build_logs_dir;
}

String get_data_editor_tools_dir() {
	return _GodotSharpDirs::get_singleton().data_editor_tools_dir;
}
#endif

} // namespace GodotSharpDirs
