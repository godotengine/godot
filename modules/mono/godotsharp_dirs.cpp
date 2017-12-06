/*************************************************************************/
/*  godotsharp_dirs.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "os/os.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_settings.h"
#include "os/dir_access.h"
#include "project_settings.h"
#include "version.h"
#endif

namespace GodotSharpDirs {

String _get_expected_build_config() {
#ifdef TOOLS_ENABLED
	return "Tools";
#else

#ifdef DEBUG_ENABLED
	return "Debug";
#else
	return "Release";
#endif

#endif
}

String _get_mono_user_dir() {
#ifdef TOOLS_ENABLED
	if (EditorSettings::get_singleton()) {
		return EditorSettings::get_singleton()->get_data_dir().plus_file("mono");
	} else {
		String settings_path;

		String exe_dir = OS::get_singleton()->get_executable_path().get_base_dir();
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
#endif

private:
	_GodotSharpDirs() {
		res_data_dir = "res://.mono";
		res_metadata_dir = res_data_dir.plus_file("metadata");
		res_assemblies_dir = res_data_dir.plus_file("assemblies");
		res_config_dir = res_data_dir.plus_file("etc").plus_file("mono");

		// TODO use paths from csproj
		res_temp_dir = res_data_dir.plus_file("temp");
		res_temp_assemblies_base_dir = res_temp_dir.plus_file("bin");
		res_temp_assemblies_dir = res_temp_assemblies_base_dir.plus_file(_get_expected_build_config());

		mono_user_dir = _get_mono_user_dir();
		mono_logs_dir = mono_user_dir.plus_file("mono_logs");

#ifdef TOOLS_ENABLED
		mono_solutions_dir = mono_user_dir.plus_file("solutions");
		build_logs_dir = mono_user_dir.plus_file("build_logs");

		String name = ProjectSettings::get_singleton()->get("application/config/name");
		if (name.empty()) {
			name = "UnnamedProject";
		}

		String base_path = String("res://") + name;

		sln_filepath = ProjectSettings::get_singleton()->globalize_path(base_path + ".sln");
		csproj_filepath = ProjectSettings::get_singleton()->globalize_path(base_path + ".csproj");
#endif
	}

	_GodotSharpDirs(const _GodotSharpDirs &);
	_GodotSharpDirs &operator=(const _GodotSharpDirs &);

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
#endif
} // namespace GodotSharpDirs
