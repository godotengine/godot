/*************************************************************************/
/*  application_configuration.cpp                                        */
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

#include "application_configuration.h"

#include "core/config/project_settings.h"
#include "core/os/dir_access.h"
#include "core/templates/list.h"
#include "core/version.h"
#include "core/version_hash.gen.h"
#include "servers/audio_server.h"
#include "servers/display_server.h"
#include "servers/text_server.h"

void print_help(const char *p_exec_path);

// Moves the iterator if the next argument is a parameter.
#define EAT_PARAMETER                                                                                                                     \
	if (!I || !is_parameter(I->get())) {                                                                                                  \
		OS::get_singleton()->printerr("Missing parameter for argument '%s'.\nUse --help for more information.\n", arg.utf8().get_data()); \
		return ERR_INVALID_PARAMETER;                                                                                                     \
	}                                                                                                                                     \
	param = I->get();                                                                                                                     \
	I = I->next();

// Prints a predefined message.
#define EXIT_COND(m_cond, m_msg)                                                                                                                                                  \
	if (m_cond) {                                                                                                                                                                 \
		OS::get_singleton()->printerr("Invalid parameter '%s' for argument '%s'. %s\nUse --help for more information.\n", param.utf8().get_data(), arg.utf8().get_data(), m_msg); \
		return ERR_INVALID_PARAMETER;                                                                                                                                             \
	}

static String clean_argument(const char *arg) {
	return String::utf8(arg).strip_edges().replace("%20", " ");
}

bool is_parameter(const String &arg) {
	return arg.length() > 0 && arg[0] != '-';
}

String get_full_version_string() {
	String hash = String(VERSION_HASH);
	if (hash.length() != 0) {
		hash = "." + hash.left(9);
	}
	return String(VERSION_FULL_BUILD) + hash;
}

String get_program_string() {
	return String(VERSION_NAME) + " v" + get_full_version_string() + " - " + String(VERSION_WEBSITE);
}

Error parse_configuration(const char *exec_path, int argc, char *argv[], ApplicationConfiguration &r_config) {
	r_config.exec_path = exec_path;

	List<String> args;
	for (int i = 0; i < argc; i++) {
		args.push_back(clean_argument(argv[i]));
	}

	bool file_specified = false;

	// Step through the list of arguments, filling in the configuration
	// as requested. An argument can have multiple required or optional
	// parameters. All parameters for an argument are handled within a
	// single loop iteration, so that at the end of iteration the list
	// iterator points to the next argument.
	List<String>::Element *I = args.front();
	while (I) {
		const String arg = I->get();
		I = I->next();
		String param;

#ifdef OSX_ENABLED
		// Ignore the process serial number argument passed by macOS Gatekeeper.
		// Otherwise, Godot would try to open a non-existent project on the first start and abort.
		if (arg.begins_with("-psn_")) {
			continue;
		}
#endif

		if (arg == "--") {
			// Pass the remaining arguments to the project.
			while (I) {
				r_config.user_args.push_back(I->get());
				I = I->next();
			}
		} else if (is_parameter(arg) && !file_specified) {
			file_specified = true;

			if (arg.ends_with("project.godot")) {
#ifdef TOOLS_ENABLED
				if (r_config.application_type == ApplicationType::PROJECT) {
					r_config.application_type = ApplicationType::EDITOR;
				}
#endif
				String directory = arg.get_base_dir();

				if (OS::get_singleton()->set_cwd(directory) != OK) {
					r_config.project_path = directory;
				}
			} else if (arg.ends_with(".scn") ||
					   arg.ends_with(".tscn") ||
					   arg.ends_with(".escn") ||
					   arg.ends_with(".res") ||
					   arg.ends_with(".tres")) {
				// Only consider the argument to be a scene path if it ends with
				// a file extension associated with Godot scenes. This makes it possible
				// for projects to parse command-line arguments for custom CLI arguments
				// or other file extensions without trouble. This can be used to implement
				// "drag-and-drop onto executable" logic, which can prove helpful
				// for non-game applications.
				r_config.scene_path = arg;
			}
		} else if (arg == "-s" || arg == "--script") {
			EAT_PARAMETER;
			r_config.script_path = param;
			r_config.application_type = ApplicationType::SCRIPT;

			if (I && I->get() == "--check-only") {
				r_config.selected_tool = StandaloneTool::VALIDATE_SCRIPT;
				r_config.run_tool = true;
				I = I->next();
			}
		} else if (arg == "--path") {
			EAT_PARAMETER;

			if (OS::get_singleton()->set_cwd(param) != OK) {
				r_config.project_path = param; //use project_path instead
			}
		} else if (arg == "--main-pack") {
			EAT_PARAMETER;
			r_config.main_pack = param;
		} else if (arg == "-h" || arg == "--help" || arg == "/?") {
			print_help(exec_path);
			return ERR_INVALID_PARAMETER;
		} else if (arg == "--version") {
			print_line(get_full_version_string());
			return ERR_INVALID_PARAMETER;
		} else if (arg == "-v" || arg == "--verbose") {
			r_config.output_verbosity = OutputVerbosity::VERBOSE;
		} else if (arg == "--quiet") {
			r_config.output_verbosity = OutputVerbosity::QUIET;
		} else if (arg == "--audio-driver") {
			EAT_PARAMETER;

			const int audio_driver_count = AudioDriverManager::get_driver_count();

			for (int i = 0; i < audio_driver_count; i++) {
				if (param == AudioDriverManager::get_driver(i)->get_name()) {
					r_config.audio_driver_index = i;
					break;
				}
			}

			if (r_config.audio_driver_index < 0) {
				OS::get_singleton()->printerr("Unknown audio driver '%s'. Choose one of: ", param.utf8().get_data());
				print_audio_drivers();
				return ERR_INVALID_PARAMETER;
			}
		} else if (arg == "--text-driver") {
			EAT_PARAMETER;

			const int text_interface_count = TextServerManager::get_interface_count();
			for (int i = 0; i < text_interface_count; i++) {
				if (param == TextServerManager::get_interface_name(i)) {
					r_config.text_driver_index = i;
					break;
				}
			}

			if (r_config.text_driver_index < 0) {
				OS::get_singleton()->printerr("Unknown text interface '%s'.", param.utf8().get_data());
				return ERR_INVALID_PARAMETER;
			}
		} else if (arg == "--display-driver") {
			EAT_PARAMETER;

			const int display_driver_count = DisplayServer::get_create_function_count();

			for (int i = 0; i < display_driver_count; i++) {
				if (param == DisplayServer::get_create_function_name(i)) {
					r_config.display_driver_index = i;
					break;
				}
			}

			if (r_config.display_driver_index < 0) {
				OS::get_singleton()->print("Unknown display driver '%s'. Choose one of: ", param.utf8().get_data());
				print_display_drivers();
				return ERR_INVALID_PARAMETER;
			}
#ifndef SERVER_ENABLED // Display options
		} else if (arg == "--tablet-driver") {
			EAT_PARAMETER;
			r_config.tablet_driver_name = param;
		} else if (arg == "-f" || arg == "--fullscreen") {
			r_config.window_mode = DisplayServer::WindowMode::WINDOW_MODE_FULLSCREEN;
			r_config.forced_window_mode = true;
		} else if (arg == "-m" || arg == "--maximized") {
			r_config.window_mode = DisplayServer::WindowMode::WINDOW_MODE_MAXIMIZED;
			r_config.forced_window_mode = true;
		} else if (arg == "-w" || arg == "--windowed") {
			r_config.window_mode = DisplayServer::WindowMode::WINDOW_MODE_WINDOWED;
			r_config.forced_window_mode = true;
		} else if (arg == "-t" || arg == "--always-on-top") {
			r_config.window_flags |= DisplayServer::WINDOW_FLAG_ALWAYS_ON_TOP;
		} else if (arg == "--no-window") {
			r_config.no_window_mode = true;
		} else if (arg == "--single-window") {
			r_config.single_window = true;
		} else if (arg == "--low-dpi") {
			r_config.allow_hidpi = false;
		} else if (arg == "--enable-vsync-via-compositor") {
			r_config.override_vsync_via_compositor_value = true;
			r_config.window_vsync_via_compositor = true;
		} else if (arg == "--disable-vsync-via-compositor") {
			r_config.override_vsync_via_compositor_value = true;
			r_config.window_vsync_via_compositor = false;
#ifdef DEBUG_ENABLED
		} else if (arg == "--gpu-abort") {
			r_config.abort_on_gpu_errors = true;
#endif // DEBUG_ENABLED
		} else if (arg == "--resolution") {
			EAT_PARAMETER;
			EXIT_COND(param.find("x") == -1, "Resolution should be e.g. 600x400.");

			int w = param.get_slice("x", 0).to_int();
			int h = param.get_slice("x", 1).to_int();
			EXIT_COND(w <= 0 || h <= 0, "Width and height should be greater than 0.");

			r_config.window_size = Size2i(w, h);
			r_config.forced_window_size = true;
		} else if (arg == "--position") {
			EAT_PARAMETER;
			EXIT_COND(param.find(",") == -1, "Position should be e.g. 100,200.");

			int x = param.get_slice(",", 0).to_int();
			int y = param.get_slice(",", 1).to_int();

			r_config.window_position = Point2(x, y);
#endif // !SERVER_ENABLED
		} else if (arg == "-l" || arg == "--language") {
			EAT_PARAMETER;
			r_config.locale = param;
		} else if (arg == "--remote-fs") {
			EAT_PARAMETER;
			if (param.find(":") != -1) {
				r_config.remote_filesystem_address = param.get_slicec(':', 0);
				r_config.remote_filesystem_port = param.get_slicec(':', 1).to_int();
			} else {
				r_config.remote_filesystem_address = param;
				r_config.remote_filesystem_port = 6010;
			}
		} else if (arg == "--remote-fs-password") {
			EAT_PARAMETER;
			r_config.remote_filesystem_password = param;
		} else if (arg == "--render-thread") {
			EAT_PARAMETER;
			if (param == "safe") {
				r_config.render_thread_mode = OS::RENDER_THREAD_SAFE;
			} else if (param == "unsafe") {
				r_config.render_thread_mode = OS::RENDER_THREAD_UNSAFE;
			} else if (param == "separate") {
				r_config.render_thread_mode = OS::RENDER_SEPARATE_THREAD;
			} else {
				EXIT_COND(true, "Invalid parameter for argument '%s'. It should be 'safe', 'unsafe' or 'separate'");
			}
		} else if (arg == "-u" || arg == "--upwards") {
			r_config.scan_folders_upwards = true;
		} else if (arg == "-q" || arg == "--quit") {
			r_config.auto_quit = true;
		} else if (arg == "-d" || arg == "--debug") {
			r_config.debug_uri = "local://";
			r_config.debug_output = true;
		} else if (arg == "-b" || arg == "--breakpoints") {
			EAT_PARAMETER;
			r_config.breakpoints = param.split(",");
		} else if (arg == "--skip-breakpoints") {
			r_config.skip_breakpoints = true;
		} else if (arg == "--profiling") {
			r_config.enable_debug_profiler = true;
		} else if (arg == "--profile-gpu") {
			r_config.enable_gpu_profiler = true;
		} else if (arg == "--vk-layers") {
			r_config.use_validation_layers = true;
		} else if (arg == "--disable-crash-handler") {
			r_config.disable_crash_handler = true;
		} else if (arg == "--allow_focus_steal_pid") {
			EAT_PARAMETER;
			r_config.allow_focus_steal_pid = param.to_int();
		} else if (arg == "--disable-render-loop") {
			r_config.disable_render_loop = true;
#if defined(DEBUG_ENABLED) && !defined(SERVER_ENABLED)
		} else if (arg == "--debug-collisions") {
			r_config.enable_debug_collisions = true;
		} else if (arg == "--debug-navigation") {
			r_config.enable_debug_navigation = true;
#endif
		} else if (arg == "--frame-delay") {
			EAT_PARAMETER;
			r_config.frame_delay = param.to_int();
		} else if (arg == "--time-scale") {
			EAT_PARAMETER;
			r_config.time_scale = param.to_float();
		} else if (arg == "--fixed-fps") {
			EAT_PARAMETER;
			r_config.fixed_fps = param.to_int();
		} else if (arg == "--print-fps") {
			r_config.print_fps = true;
		} else if (arg == "--remote-debug") {
			EAT_PARAMETER;
			ERR_FAIL_COND_V_MSG((param.find("://") == -1), ERR_INVALID_PARAMETER, "Invalid debug host address, it should be of the form <protocol>://<host/IP>:<port>.");

			r_config.debug_uri = param;
#ifdef TOOLS_ENABLED
		} else if (arg == "-e" || arg == "--editor") {
			r_config.application_type = ApplicationType::EDITOR;
			r_config.main_args.push_back(arg);
		} else if (arg == "-p" || arg == "--project-manager") {
			r_config.application_type = ApplicationType::PROJECT_MANAGER;
		} else if (arg == "--build-solutions") {
			r_config.auto_build_solutions = true;
			r_config.application_type = ApplicationType::EDITOR;
		} else if (arg == "--export") {
			EAT_PARAMETER;
			r_config.export_config.preset = param;

			EAT_PARAMETER;
			r_config.export_config.path = param;

			r_config.main_args.push_back(arg);
			r_config.selected_tool = StandaloneTool::EXPORT;

			// Export has multiple optional flags, and they can appear in any order.
			Vector<String> flag_candidates;
			if (I) {
				flag_candidates.push_back(I->get());
				if (I->next()) {
					flag_candidates.push_back(I->next()->get());
				}
			}

			for (int i = 0; i < flag_candidates.size(); i++) {
				bool valid_flag = true;
				if (flag_candidates[i] == "--export-debug") {
					r_config.export_config.debug_build = true;
				} else if (flag_candidates[i] == "--export-pack") {
					r_config.export_config.pack_only = true;
				} else {
					valid_flag = false;
				}

				if (valid_flag) {
					I = I->next();
				}
			}
		} else if (arg == "--doctool") {
			r_config.selected_tool = StandaloneTool::DOC_TOOL;
			r_config.run_tool = true;

			r_config.doc_tool_path = "."; // Assume current directory if no path is given.
			r_config.doc_base_types = true;

			// DocTool has multiple optional parameters, but they are ordered.
			if (I) {
				param = I->get();

				if (is_parameter(param)) {
					r_config.doc_tool_path = param;

					if (!I->next()) {
						continue;
					}

					param = I->next()->get();
				}

				if (param == "--no-docbase") {
					r_config.doc_base_types = false;
				}
			}
#ifdef DEBUG_METHODS_ENABLED
		} else if (arg == "--gdnative-generate-json-api" || arg == "--gdnative-generate-json-builtin-api") {
			// Register as an editor instance to use low-end fallback if relevant.
			r_config.application_type = ApplicationType::EDITOR;

			// We still pass it to the main arguments since the argument handling itself is not done in this function
			r_config.main_args.push_back(arg);
#endif // DEBUG_METHODS_ENABLED
#endif // TOOLS_ENABLED
		} else {
			r_config.main_args.push_back(arg);
		}
	}

	return OK;
}

Error finalize_configuration(ApplicationConfiguration &r_config) {
	// Window configuration
	{
		if (r_config.application_type != ApplicationType::PROJECT_MANAGER && r_config.application_type != ApplicationType::EDITOR) {
			if (!r_config.forced_window_size) {
#ifdef TOOLS_ENABLED
				r_config.window_size.width = GLOBAL_GET("display/window/size/test_width");
				r_config.window_size.height = GLOBAL_GET("display/window/size/test_height");

				if (r_config.window_size == Size2i()) {
					r_config.window_size.width = GLOBAL_GET("display/window/size/width");
					r_config.window_size.height = GLOBAL_GET("display/window/size/height");
				}
#else
				r_config.window_size.width = GLOBAL_GET("display/window/size/width");
				r_config.window_size.height = GLOBAL_GET("display/window/size/height");
#endif // TOOLS_ENABLED
			}

			if (!bool(GLOBAL_GET("display/window/size/resizable"))) {
				r_config.window_flags |= DisplayServer::WINDOW_FLAG_RESIZE_DISABLED_BIT;
			}
			if (bool(GLOBAL_GET("display/window/size/borderless"))) {
				r_config.window_flags |= DisplayServer::WINDOW_FLAG_BORDERLESS_BIT;
			}
			if (bool(GLOBAL_GET("display/window/size/always_on_top"))) {
				r_config.window_flags |= DisplayServer::WINDOW_FLAG_ALWAYS_ON_TOP_BIT;
			}
			if (bool(GLOBAL_GET("display/window/size/fullscreen"))) {
				r_config.window_mode = DisplayServer::WINDOW_MODE_FULLSCREEN;
			}
		}

#ifdef TOOLS_ENABLED
		if (r_config.run_tool) {
			r_config.no_window_mode = true;
		}

		if (r_config.application_type == ApplicationType::EDITOR && !r_config.forced_window_mode) {
			r_config.window_mode = DisplayServer::WINDOW_MODE_MAXIMIZED;
		}
#endif // TOOLS_ENABLED
	}

	if ((r_config.application_type == ApplicationType::EDITOR || r_config.application_type == ApplicationType::PROJECT) && r_config.scene_path == "") {
		r_config.scene_path = String(GLOBAL_GET("application/run/main_scene"));

		// Only fail if running the projecct.
		bool should_fail = (r_config.application_type == ApplicationType::PROJECT && r_config.scene_path == "");
		ERR_FAIL_COND_V_MSG(should_fail, ERR_INVALID_PARAMETER, vformat("Can't run project '%s': no main scene defined.\n", r_config.project_path.utf8().get_data()));
	}

#ifdef TOOLS_ENABLED
	if (r_config.export_config.preset != "") {
		r_config.scene_path = ""; // Do not load anything.
	}

	if (r_config.doc_tool_path != "") {
		DirAccessRef da = DirAccess::open(r_config.doc_tool_path);
		ERR_FAIL_COND_V_MSG(!da, ERR_INVALID_PARAMETER, "Argument supplied to --doctool must be a valid directory path.");
	}
#endif // TOOLS_ENABLED

	if (!r_config.override_vsync_via_compositor_value) {
		// If one of the command line options to enable/disable vsync via the
		// window compositor ("--enable-vsync-via-compositor" or
		// "--disable-vsync-via-compositor") was present then it overrides the
		// project setting.
		r_config.window_vsync_via_compositor = GLOBAL_GET("display/window/vsync/vsync_via_compositor");
	}

#ifdef NO_THREADS
	r_config.render_thread_mode = OS::RENDER_THREAD_UNSAFE; // No threads available on this platform.
#else
#ifdef TOOLS_ENABLED
	if (r_config.application_type == ApplicationType::EDITOR) {
		r_config.render_thread_mode = OS::RENDER_THREAD_SAFE;
	}
#endif // TOOLS_ENABLED
#endif // NO_THREADS

	return OK;
}

void print_audio_drivers() {
	const int audio_driver_count = AudioDriverManager::get_driver_count();
	if (audio_driver_count == 0) {
		OS::get_singleton()->print("[].\n");
		return;
	}

	OS::get_singleton()->print("['%s'", AudioDriverManager::get_driver(0)->get_name());
	if (audio_driver_count > 1) {
		for (int i = 1; i < audio_driver_count - 1; i++) {
			OS::get_singleton()->print(", '%s'", AudioDriverManager::get_driver(i)->get_name());
		}

		OS::get_singleton()->print(" and '%s'", AudioDriverManager::get_driver(audio_driver_count - 1)->get_name());
	}
	OS::get_singleton()->print("].\n");
}

void print_display_drivers() {
	const int display_driver_count = DisplayServer::get_create_function_count();
	if (display_driver_count == 0) {
		OS::get_singleton()->print("[].\n");
		return;
	}

	OS::get_singleton()->print("['%s'", DisplayServer::get_create_function_name(0));
	if (display_driver_count > 1) {
		for (int i = 1; i < display_driver_count; i++) {
			OS::get_singleton()->print(", '%s'", DisplayServer::get_create_function_name(i));
		}
	}
	OS::get_singleton()->print("].\n");
}

void print_display_and_rendering_drivers() {
	const int display_driver_count = DisplayServer::get_create_function_count();
	if (display_driver_count == 0) {
		OS::get_singleton()->print("[].\n");
		return;
	}

	for (int i = 0; i < display_driver_count; i++) {
		if (i > 0) {
			OS::get_singleton()->print(", ");
		}
		OS::get_singleton()->print("['%s' (", DisplayServer::get_create_function_name(i));
		Vector<String> rd = DisplayServer::get_create_function_rendering_drivers(i);
		for (int j = 0; j < rd.size(); j++) {
			if (j > 0) {
				OS::get_singleton()->print(", ");
			}
			OS::get_singleton()->print("'%s'", rd[j].utf8().get_data());
		}
		OS::get_singleton()->print(")");
	}
	OS::get_singleton()->print("].\n");
}

void print_help(const char *p_exec_path) {
	print_line(get_program_string());
	OS::get_singleton()->print("Free and open source software under the terms of the MIT license.\n");
	OS::get_singleton()->print("(c) 2007-2021 Juan Linietsky, Ariel Manzur.\n");
	OS::get_singleton()->print("(c) 2014-2021 Godot Engine contributors.\n");
	OS::get_singleton()->print("\n");
	OS::get_singleton()->print("Usage: %s [options] [path to scene or 'project.godot' file]\n", p_exec_path);
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("General options:\n");
	OS::get_singleton()->print("  -h, --help                                   Display this help message and exit.\n");
	OS::get_singleton()->print("  --version                                    Display the version string and exit.\n");
	OS::get_singleton()->print("  -v, --verbose                                Use verbose stdout mode.\n");
	OS::get_singleton()->print("  --quiet                                      Quiet mode, silences stdout messages. Errors are still displayed.\n");
	OS::get_singleton()->print("  --                                           Pass every following argument to the project. \n");
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("Run options:\n");
#ifdef TOOLS_ENABLED
	OS::get_singleton()->print("  -e, --editor                                 Start the editor instead of running the scene.\n");
	OS::get_singleton()->print("  -p, --project-manager                        Start the project manager, even if a project is auto-detected.\n");
#endif
	OS::get_singleton()->print("  -q, --quit                                   Quit after the first main loop iteration.\n");
	OS::get_singleton()->print("  -l, --language <locale>                      Use a specific locale (<locale> being a two-letter code).\n");
	OS::get_singleton()->print("  --path <directory>                           Path to a project (<directory> must contain a 'project.godot' file).\n");
	OS::get_singleton()->print("  -u, --upwards                                Scan folders upwards for project.godot file.\n");
	OS::get_singleton()->print("  --main-pack <file>                           Path to a pack (.pck) file to load.\n");
	OS::get_singleton()->print("  --render-thread <mode>                       Render thread mode ('unsafe', 'safe', 'separate').\n");
	OS::get_singleton()->print("  --remote-fs <address>                        Remote filesystem (<host/IP>[:<port>] address).\n");
	OS::get_singleton()->print("  --remote-fs-password <password>              Password for remote filesystem.\n");

	OS::get_singleton()->print("  --audio-driver <driver>                      Choose audio driver: ");
	print_audio_drivers();

	OS::get_singleton()->print("  --display-driver <driver>                    Choose display driver: ");
	print_display_and_rendering_drivers();

	OS::get_singleton()->print("  --rendering-driver <driver>                  Rendering driver (depends on display driver).\n");
	OS::get_singleton()->print("  --text-driver <driver>                       Text driver (Fonts, BiDi, shaping)\n");
	OS::get_singleton()->print("\n");

#ifndef SERVER_ENABLED
	OS::get_singleton()->print("Display options:\n");
	OS::get_singleton()->print("  -f, --fullscreen                             Request fullscreen mode.\n");
	OS::get_singleton()->print("  -m, --maximized                              Request a maximized window.\n");
	OS::get_singleton()->print("  -w, --windowed                               Request windowed mode.\n");
	OS::get_singleton()->print("  -t, --always-on-top                          Request an always-on-top window.\n");
	OS::get_singleton()->print("  --resolution <W>x<H>                         Request window resolution. Format: WxH (e.g. 600x400).\n");
	OS::get_singleton()->print("  --position <X>,<Y>                           Request window position. Format: X,Y (e.g. 100,200).\n");
	OS::get_singleton()->print("  --low-dpi                                    Force low-DPI mode (macOS and Windows only).\n");
	OS::get_singleton()->print("  --no-window                                  Disable window creation (Currently not implemented).\n");
	OS::get_singleton()->print("  --enable-vsync-via-compositor                When vsync is enabled, vsync via the OS' window compositor (Windows only).\n");
	OS::get_singleton()->print("  --disable-vsync-via-compositor               Disable vsync via the OS' window compositor (Windows only).\n");
	OS::get_singleton()->print("  --single-window                              Use a single window (no separate subwindows).\n");
	OS::get_singleton()->print("  --tablet-driver                              Pen tablet input driver.\n");
	OS::get_singleton()->print("\n");
#endif

	OS::get_singleton()->print("Debug options:\n"); //set implementation-specific option. The following options are available:
	OS::get_singleton()->print("  -d, --debug                                  Debug (local stdout debugger).\n");
	OS::get_singleton()->print("  -b, --breakpoints                            Breakpoint list as source::line comma-separated pairs, no spaces (use %%20 instead).\n");
	OS::get_singleton()->print("  --profiling                                  Enable profiling in the script debugger.\n");
	OS::get_singleton()->print("  --vk-layers                                  Enable Vulkan Validation layers for debugging.\n");
#if DEBUG_ENABLED
	OS::get_singleton()->print("  --gpu-abort                                  Abort on GPU errors (usually validation layer errors), may help see the problem if your system freezes.\n");
#endif
	OS::get_singleton()->print("  --remote-debug <uri>                         Remote debug (<protocol>://<host/IP>[:<port>], e.g. tcp://127.0.0.1:6007).\n");
#if defined(DEBUG_ENABLED) && !defined(SERVER_ENABLED)
	OS::get_singleton()->print("  --debug-collisions                           Show collision shapes when running the scene.\n");
	OS::get_singleton()->print("  --debug-navigation                           Show navigation polygons when running the scene.\n");
#endif
	OS::get_singleton()->print("  --frame-delay <ms>                           Simulate high CPU load (delay each frame by <ms> milliseconds).\n");
	OS::get_singleton()->print("  --time-scale <scale>                         Force time scale (higher values are faster, 1.0 is normal speed).\n");
	OS::get_singleton()->print("  --disable-render-loop                        Disable render loop so rendering only occurs when called explicitly from script.\n");
	OS::get_singleton()->print("  --disable-crash-handler                      Disable crash handler when supported by the platform code.\n");
	OS::get_singleton()->print("  --fixed-fps <fps>                            Force a fixed number of frames per second. This setting disables real-time synchronization.\n");
	OS::get_singleton()->print("  --print-fps                                  Print the frames per second to the stdout.\n");
	OS::get_singleton()->print("  --profile-gpu                                Show a simple profile of the tasks that took more time during frame rendering.\n");
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("Standalone tools:\n");
	OS::get_singleton()->print("  -s, --script <path> [--check-only]           Run a script (must inherit from MainLoop). Append --check-only to only parse the script for errors.\n");
#ifdef TOOLS_ENABLED
	OS::get_singleton()->print("  --export <preset> <path> [parameter]         Export the project using the given preset and matching release template. The preset name\n");
	OS::get_singleton()->print("                                               should match one defined in export_presets.cfg. <path> should be absolute or relative to\n");
	OS::get_singleton()->print("                                               the project directory, and include the filename for the binary (e.g. 'builds/game.exe').\n");
	OS::get_singleton()->print("                                               The target directory should exist.\n");
	OS::get_singleton()->print("                                               Optional parameters: --export-debug   Use debug template.\n");
	OS::get_singleton()->print("                                                                    --export-pack    Only export the game pack. The <path> extension\n");
	OS::get_singleton()->print("											   determines whether it will be in PCK or ZIP format.\n");
	OS::get_singleton()->print("  --doctool [<path>] [--no-docbase]            Dump the engine API reference to the given <path> (defaults to current dir), merging if\n");
	OS::get_singleton()->print("                                               existing files are found. Append --no-docbase to disallow dumping the base types.\n");
	OS::get_singleton()->print("  --build-solutions                            Build the scripting solutions (e.g. for C# projects). Implies --editor and requires a valid project to edit.\n");
#ifdef DEBUG_METHODS_ENABLED
	OS::get_singleton()->print("  --gdnative-generate-json-api <path>          Generate JSON dump of the Godot API for GDNative bindings and save it on the file specified in <path>.\n");
	OS::get_singleton()->print("  --gdnative-generate-json-builtin-api <path>  Generate JSON dump of the Godot API of the builtin Variant types and utility functions\n");
	OS::get_singleton()->print("                                               for GDNative bindings and save it on the file specified in <path>.\n");
#endif
#ifdef TESTS_ENABLED
	OS::get_singleton()->print("  --test [--help]                              Run unit tests. Use --test --help for more information.\n");
#endif
#endif
	OS::get_singleton()->print("\n");
}
