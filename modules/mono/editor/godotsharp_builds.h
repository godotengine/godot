/*************************************************************************/
/*  godotsharp_builds.h                                                  */
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

#ifndef GODOTSHARP_BUILDS_H
#define GODOTSHARP_BUILDS_H

#include "../mono_gd/gd_mono.h"
#include "mono_bottom_panel.h"
#include "mono_build_info.h"

typedef void (*GodotSharpBuild_ExitCallback)(int);

class GodotSharpBuilds {

private:
	struct BuildProcess {
		Ref<MonoGCHandle> build_instance;
		MonoBuildInfo build_info;
		MonoBuildTab *build_tab;
		GodotSharpBuild_ExitCallback exit_callback;
		bool exited;
		int exit_code;

		void on_exit(int p_exit_code);
		void start(bool p_blocking = false);

		BuildProcess() {}
		BuildProcess(const MonoBuildInfo &p_build_info, GodotSharpBuild_ExitCallback p_callback = NULL);
	};

	HashMap<MonoBuildInfo, BuildProcess, MonoBuildInfo::Hasher> builds;

	static String _api_folder_name(APIAssembly::Type p_api_type);

	static GodotSharpBuilds *singleton;

public:
	enum BuildTool {
		MSBUILD_MONO,
#ifdef WINDOWS_ENABLED
		MSBUILD_VS,
#endif
		XBUILD // Deprecated
	};

	_FORCE_INLINE_ static GodotSharpBuilds *get_singleton() { return singleton; }

	static void register_internal_calls();

	static void show_build_error_dialog(const String &p_message);

	void build_exit_callback(const MonoBuildInfo &p_build_info, int p_exit_code);

	void restart_build(MonoBuildTab *p_build_tab);
	void stop_build(MonoBuildTab *p_build_tab);

	bool build(const MonoBuildInfo &p_build_info);
	bool build_async(const MonoBuildInfo &p_build_info, GodotSharpBuild_ExitCallback p_callback = NULL);

	static bool build_api_sln(const String &p_api_sln_dir, const String &p_config);
	static bool copy_api_assembly(const String &p_src_dir, const String &p_dst_dir, const String &p_assembly_name, APIAssembly::Type p_api_type);

	static bool make_api_assembly(APIAssembly::Type p_api_type);

	static bool build_project_blocking(const String &p_config);

	static bool editor_build_callback();

	GodotSharpBuilds();
	~GodotSharpBuilds();
};

#endif // GODOTSHARP_BUILDS_H
