/*************************************************************************/
/*  application_configuration.h                                          */
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

#ifndef APPLICATION_CONFIGURATION_H
#define APPLICATION_CONFIGURATION_H

#include "../core/error/error_list.h"
#include "servers/display_server.h"

enum class ApplicationType {
	PROJECT,
	SCRIPT,
	EDITOR,
	PROJECT_MANAGER,
};

enum class StandaloneTool {
	DOC_TOOL,
	EXPORT,
	VALIDATE_SCRIPT
};

enum class OutputVerbosity {
	NORMAL,
	QUIET,
	VERBOSE
};

typedef struct ExportConfiguration {
	String preset = "";
	String path = "";
	bool debug_build = false;
	bool pack_only = false;
} ExportConfiguration;

typedef struct ApplicationConfiguration {
	ApplicationType application_type = ApplicationType::PROJECT;
	String scene_path = "";

	String project_path = ".";
	String main_pack = "";
	bool scan_folders_upwards = false;

	String script_path;

	// Tools
	bool run_tool = false;
	StandaloneTool selected_tool;

#ifdef TOOLS_ENABLED
	String doc_tool_path = "";
	bool doc_base_types = true;

	ExportConfiguration export_config;

	bool auto_build_solutions;
#endif

	bool auto_quit = false;

	// Drivers
	int display_driver_index = -1;
	int audio_driver_index = -1;
	int text_driver_index = -1;
	String tablet_driver_name;

	// Display
	bool single_window = false;
	bool init_always_on_top;
	bool allow_hidpi = true;
	bool no_window_mode = false;

	bool use_vsync = false;
	bool window_vsync_via_compositor = false;
	bool saw_vsync_via_compositor_override = false;
	bool override_vsync_via_compositor_value = false;

	DisplayServer::WindowMode window_mode = DisplayServer::WindowMode::WINDOW_MODE_WINDOWED;
	bool forced_window_mode = false;
	uint32_t window_flags = 0;
	Point2 window_position;
	Size2i window_size = Size2i(1024, 600);
	bool forced_window_size = false;

	// Debug
	bool debug_output = false;
	String debug_uri = "";
	PackedStringArray breakpoints;
	bool skip_breakpoints = false;
	bool enable_debug_collisions = false;
	bool enable_debug_navigation = false;
	bool enable_debug_profiler = false;
	bool enable_gpu_profiler = false;
	bool disable_crash_handler = false;
	bool disable_render_loop = false;
	bool use_validation_layers = false;
	bool abort_on_gpu_errors = false;

	bool print_fps = false;
	int fixed_fps = -1;
	OutputVerbosity output_verbosity = OutputVerbosity::NORMAL;

	String remote_filesystem_address;
	String remote_filesystem_password;
	int remote_filesystem_port;

	OS::RenderThreadMode render_thread_mode;
	double time_scale = 1;
	int frame_delay = 0;
	String locale;

	OS::ProcessID allow_focus_steal_pid = -1; // Not exposed to user.

	String exec_path;
	List<String> main_args;
	List<String> user_args;
} ApplicationConfiguration;

void print_audio_drivers();
void print_display_drivers();

Error parse_configuration(const char *exec_path, int argc, char *argv[], ApplicationConfiguration &r_configuration);
Error finalize_configuration(ApplicationConfiguration &r_configuration);

String get_program_string();
#endif // APPLICATION_CONFIGURATION_H
