/*************************************************************************/
/*  godotsharp_dirs.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "godotsharp_dirs.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/os/os.h"

#ifdef TOOLS_ENABLED
#include "core/version.h"
#include "editor/editor_settings.h"
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
		return EditorPaths::get_singleton()->get_data_dir().plus_file("mono");
	} else {
		String settings_path;

		// Self-contained mode if a `._sc_` or `_sc_` file is present in executable dir.
		String exe_dir = OS::get_singleton()->get_executable_path().get_base_dir();

		// On macOS, look outside .app bundle, since .app bundle is read-only.
		if (OS::get_singleton()->has_feature("macos") && exe_dir.ends_with("MacOS") && exe_dir.plus_file("..").simplify_path().ends_with("Contents")) {
			exe_dir = exe_dir.plus_file("../../..").simplify_path();
		}

		DirAccessRef d = DirAccess::create_for_path(exe_dir);

		if (d->file_exists("._sc_") || d->file_exists("_sc_")) {
			// contain yourself
			settings_path = exe_dir.plus_file("editor_data");
		} else {
			settings_path = OS::get_singleton()->get_data_path().plus_file(OS::get_singleton()->get_godot_dir_name());
		}

		return settings_path.plus_file("mono");
	}
#else
	return OS::get_singleton()->get_user_data_dir().plus_file("mono");
#endif
}

class _GodotSharpDirs {
public:
	String res_data_dir;
	String res_metadata_dir;
	String res_assemblies_base_dir;
	String res_assemblies_dir;
	String res_config_dir;
	String res_temp_dir;
	String res_temp_assemblies_base_dir;
	String res_temp_assemblies_dir;
	String mono_user_dir;
	String mono_logs_dir;

#ifdef TOOLS_ENABLED
	String mono_solutions_dir;
	String build_logs_dir;

	String sln_filepath;
	String csproj_filepath;

	String data_editor_tools_dir;
	String data_editor_prebuilt_api_dir;
#else
	// Equivalent of res_assemblies_dir, but in the data directory rather than in 'res://'.
	// Only defined on export templates. Used when exporting assemblies outside of PCKs.
	String data_game_assemblies_dir;
#endif

	String data_mono_etc_dir;
	String data_mono_lib_dir;

#ifdef WINDOWS_ENABLED
	String data_mono_bin_dir;
#endif

private:
	_GodotSharpDirs() {
		res_data_dir = ProjectSettings::get_singleton()->get_project_data_path().plus_file("mono");
		res_metadata_dir = res_data_dir.plus_file("metadata");
		res_assemblies_base_dir = res_data_dir.plus_file("assemblies");
		res_assemblies_dir = res_assemblies_base_dir.plus_file(GDMono::get_expected_api_build_config());
		res_config_dir = res_data_dir.plus_file("etc").plus_file("mono");

		// TODO use paths from csproj
		res_temp_dir = res_data_dir.plus_file("temp");
		res_temp_assemblies_base_dir = res_temp_dir.plus_file("bin");
		res_temp_assemblies_dir = res_temp_assemblies_base_dir.plus_file(_get_expected_build_config());

#ifdef JAVASCRIPT_ENABLED
		mono_user_dir = "user://";
#else
		mono_user_dir = _get_mono_user_dir();
#endif
		mono_logs_dir = mono_user_dir.plus_file("mono_logs");

#ifdef TOOLS_ENABLED
		mono_solutions_dir = mono_user_dir.plus_file("solutions");
		build_logs_dir = mono_user_dir.plus_file("build_logs");

		String appname = ProjectSettings::get_singleton()->get("application/config/name");
		String appname_safe = OS::get_singleton()->get_safe_dir_name(appname);
		if (appname_safe.is_empty()) {
			appname_safe = "UnnamedProject";
		}

		String base_path = ProjectSettings::get_singleton()->globalize_path("res://");

		sln_filepath = base_path.plus_file(appname_safe + ".sln");
		csproj_filepath = base_path.plus_file(appname_safe + ".csproj");
#endif

		String exe_dir = OS::get_singleton()->get_executable_path().get_base_dir();

#ifdef TOOLS_ENABLED

		String data_dir_root = exe_dir.plus_file("GodotSharp");
		data_editor_tools_dir = data_dir_root.plus_file("Tools");
		data_editor_prebuilt_api_dir = data_dir_root.plus_file("Api");

		String data_mono_root_dir = data_dir_root.plus_file("Mono");
		data_mono_etc_dir = data_mono_root_dir.plus_file("etc");

#ifdef ANDROID_ENABLED
		data_mono_lib_dir = gdmono::android::support::get_app_native_lib_dir();
#else
		data_mono_lib_dir = data_mono_root_dir.plus_file("lib");
#endif

#ifdef WINDOWS_ENABLED
		data_mono_bin_dir = data_mono_root_dir.plus_file("bin");
#endif

#ifdef OSX_ENABLED
		if (!DirAccess::exists(data_editor_tools_dir)) {
			data_editor_tools_dir = exe_dir.plus_file("../Resources/GodotSharp/Tools");
		}

		if (!DirAccess::exists(data_editor_prebuilt_api_dir)) {
			data_editor_prebuilt_api_dir = exe_dir.plus_file("../Resources/GodotSharp/Api");
		}

		if (!DirAccess::exists(data_mono_root_dir)) {
			data_mono_etc_dir = exe_dir.plus_file("../Resources/GodotSharp/Mono/etc");
			data_mono_lib_dir = exe_dir.plus_file("../Resources/GodotSharp/Mono/lib");
		}
#endif

#else

		String appname = ProjectSettings::get_singleton()->get("application/config/name");
		String appname_safe = OS::get_singleton()->get_safe_dir_name(appname);
		String data_dir_root = exe_dir.plus_file("data_" + appname_safe);
		if (!DirAccess::exists(data_dir_root)) {
			data_dir_root = exe_dir.plus_file("data_Godot");
		}

		String data_mono_root_dir = data_dir_root.plus_file("Mono");
		data_mono_etc_dir = data_mono_root_dir.plus_file("etc");

#ifdef ANDROID_ENABLED
		data_mono_lib_dir = gdmono::android::support::get_app_native_lib_dir();
#else
		data_mono_lib_dir = data_mono_root_dir.plus_file("lib");
		data_game_assemblies_dir = data_dir_root.plus_file("Assemblies");
#endif

#ifdef WINDOWS_ENABLED
		data_mono_bin_dir = data_mono_root_dir.plus_file("bin");
#endif

#ifdef OSX_ENABLED
		if (!DirAccess::exists(data_mono_root_dir)) {
			data_mono_etc_dir = exe_dir.plus_file("../Resources/GodotSharp/Mono/etc");
			data_mono_lib_dir = exe_dir.plus_file("../Resources/GodotSharp/Mono/lib");
		}

		if (!DirAccess::exists(data_game_assemblies_dir)) {
			data_game_assemblies_dir = exe_dir.plus_file("../Resources/GodotSharp/Assemblies");
		}
#endif

#endif
	}

public:
	static _GodotSharpDirs &get_singleton() {
		static _GodotSharpDirs singleton;
		return singleton;
	}
};

String get_res_data_dir() {
	return _GodotSharpDirs::get_singleton().res_data_dir;
}

String get_res_metadata_dir() {
	return _GodotSharpDirs::get_singleton().res_metadata_dir;
}

String get_res_assemblies_base_dir() {
	return _GodotSharpDirs::get_singleton().res_assemblies_base_dir;
}

String get_res_assemblies_dir() {
	return _GodotSharpDirs::get_singleton().res_assemblies_dir;
}

String get_res_config_dir() {
	return _GodotSharpDirs::get_singleton().res_config_dir;
}

String get_res_temp_dir() {
	return _GodotSharpDirs::get_singleton().res_temp_dir;
}

String get_res_temp_assemblies_base_dir() {
	return _GodotSharpDirs::get_singleton().res_temp_assemblies_base_dir;
}

String get_res_temp_assemblies_dir() {
	return _GodotSharpDirs::get_singleton().res_temp_assemblies_dir;
}

String get_mono_user_dir() {
	return _GodotSharpDirs::get_singleton().mono_user_dir;
}

String get_mono_logs_dir() {
	return _GodotSharpDirs::get_singleton().mono_logs_dir;
}

#ifdef TOOLS_ENABLED
String get_mono_solutions_dir() {
	return _GodotSharpDirs::get_singleton().mono_solutions_dir;
}

String get_build_logs_dir() {
	return _GodotSharpDirs::get_singleton().build_logs_dir;
}

String get_project_sln_path() {
	return _GodotSharpDirs::get_singleton().sln_filepath;
}

String get_project_csproj_path() {
	return _GodotSharpDirs::get_singleton().csproj_filepath;
}

String get_data_editor_tools_dir() {
	return _GodotSharpDirs::get_singleton().data_editor_tools_dir;
}

String get_data_editor_prebuilt_api_dir() {
	return _GodotSharpDirs::get_singleton().data_editor_prebuilt_api_dir;
}
#else
String get_data_game_assemblies_dir() {
	return _GodotSharpDirs::get_singleton().data_game_assemblies_dir;
}
#endif

String get_data_mono_etc_dir() {
	return _GodotSharpDirs::get_singleton().data_mono_etc_dir;
}

String get_data_mono_lib_dir() {
	return _GodotSharpDirs::get_singleton().data_mono_lib_dir;
}

#ifdef WINDOWS_ENABLED
String get_data_mono_bin_dir() {
	return _GodotSharpDirs::get_singleton().data_mono_bin_dir;
}
#endif
} // namespace GodotSharpDirs
