/*************************************************************************/
/*  test_application_configuration.h                                     */
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

#ifndef TEST_APPLICATION_CONFIGURATION_H
#define TEST_APPLICATION_CONFIGURATION_H

#include "core/config/project_settings.h"
#include "core/error/error_list.h"
#include "main/application_configuration.h"

#include "tests/test_macros.h"
#include "thirdparty/doctest/doctest.h"

namespace TestApplicationConfiguration {

// Converts arguments to char ** and parses the arguments.
#define PARSE_CONFIG()                                             \
	char **argv = new char *[args.size()];                         \
	const int n = args.size();                                     \
                                                                   \
	for (int i = 0; i < n; i++) {                                  \
		const char *str = args[i].utf8().get_data();               \
		argv[i] = new char[strlen(str) + 1];                       \
		memcpy(argv[i], str, strlen(str) + 1);                     \
	}                                                              \
                                                                   \
	ApplicationConfiguration configuration;                        \
	Error error = parse_configuration("", n, argv, configuration); \
                                                                   \
	for (int i = 0; i < n; i++) {                                  \
		delete[] argv[i];                                          \
	}                                                              \
	delete[] argv;

TEST_SUITE("[ApplicationConfiguration] Parsing") {
	TEST_CASE("[ApplicationConfiguration] project.godot or scene file." * doctest::description("Test desc.")) {
		SUBCASE("Empty argv should return OK.") {
			ApplicationConfiguration configuration;
			Error error = parse_configuration("", 0, {}, configuration);

			CHECK(error == OK);
			CHECK(configuration.project_path == ".");
			CHECK(configuration.application_type == ApplicationType::PROJECT);
		}

		SUBCASE("Project file should open editor.") {
			Vector<String> args;
			args.push_back("./project.godot");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.application_type == ApplicationType::EDITOR);
		}

		SUBCASE(".scn should be recognized as a scene file.") {
			Vector<String> args;
			args.push_back("path/to/scene.scn");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.scene_path == "path/to/scene.scn");
			CHECK(configuration.application_type == ApplicationType::PROJECT);
		}

		SUBCASE(".tscn should be recognized as a scene file.") {
			Vector<String> args;
			args.push_back("path/to/scene.tscn");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.scene_path == "path/to/scene.tscn");
			CHECK(configuration.application_type == ApplicationType::PROJECT);
		}

		SUBCASE(".escn should be recognized as a scene file.") {
			Vector<String> args;
			args.push_back("path/to/scene.escn");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.scene_path == "path/to/scene.escn");
			CHECK(configuration.application_type == ApplicationType::PROJECT);
		}

		SUBCASE(".res should be recognized as a scene file.") {
			Vector<String> args;
			args.push_back("path/to/scene.res");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.scene_path == "path/to/scene.res");
			CHECK(configuration.application_type == ApplicationType::PROJECT);
		}

		SUBCASE(".tres should be recognized as a scene file.") {
			Vector<String> args;
			args.push_back("path/to/scene.tres");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.scene_path == "path/to/scene.tres");
			CHECK(configuration.application_type == ApplicationType::PROJECT);
		}
	}

	TEST_CASE("[ApplicationConfiguration] Export") {
		ERR_PRINT_OFF;

		SUBCASE("Regular export") {
			Vector<String> args;
			args.push_back("--export");
			args.push_back("W");
			args.push_back("builds/project.exe");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.export_config.preset == args[1]);
			CHECK(configuration.export_config.path == args[2]);
			CHECK(configuration.export_config.debug_build == false);
			CHECK(configuration.export_config.pack_only == false);
		}

		SUBCASE("Debug export set correct config.") {
			Vector<String> args;
			args.push_back("--export");
			args.push_back("W");
			args.push_back("builds/project.exe");
			args.push_back("--export-debug");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.export_config.preset == args[1]);
			CHECK(configuration.export_config.path == args[2]);
			CHECK(configuration.export_config.debug_build == true);
			CHECK(configuration.export_config.pack_only == false);
		}

		SUBCASE("Pack export should set correct config.") {
			Vector<String> args;
			args.push_back("--export");
			args.push_back("W");
			args.push_back("builds/project.pck");
			args.push_back("--export-pack");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.export_config.preset == args[1]);
			CHECK(configuration.export_config.path == args[2]);
			CHECK(configuration.export_config.debug_build == false);
			CHECK(configuration.export_config.pack_only == true);
		}

		SUBCASE("Combined export options should set correct config.") {
			Vector<String> args;
			args.push_back("--export");
			args.push_back("W");
			args.push_back("builds/project.exe");
			args.push_back("--export-debug");
			args.push_back("--export-pack");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.export_config.preset == args[1]);
			CHECK(configuration.export_config.path == args[2]);
			CHECK(configuration.export_config.debug_build == true);
			CHECK(configuration.export_config.pack_only == true);
		}

		SUBCASE("Combined export options should set correct config, independent of order.") {
			Vector<String> args;
			args.push_back("--export");
			args.push_back("W");
			args.push_back("builds/project.exe");
			args.push_back("--export-pack");
			args.push_back("--export-debug");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.export_config.preset == args[1]);
			CHECK(configuration.export_config.path == args[2]);
			CHECK(configuration.export_config.debug_build == true);
			CHECK(configuration.export_config.pack_only == true);
		}

		SUBCASE("Export without path should fail.") {
			Vector<String> args;
			args.push_back("--export");
			args.push_back("W");

			PARSE_CONFIG();

			CHECK(error == ERR_INVALID_PARAMETER);
		}

		SUBCASE("Export without preset or path should fail.") {
			Vector<String> args;
			args.push_back("--export");

			PARSE_CONFIG();

			CHECK(error == ERR_INVALID_PARAMETER);
		}

		SUBCASE("Export should not treat next argument as parameters.") {
			Vector<String> args;
			args.push_back("--export");
			args.push_back("--verbose");
			args.push_back("-q");

			PARSE_CONFIG();

			CHECK(error == ERR_INVALID_PARAMETER);
		}

		ERR_PRINT_ON;
	}

	TEST_CASE("[ApplicationConfiguration] Script") {
		ERR_PRINT_OFF;

		SUBCASE("Script with regular argument.") {
			Vector<String> args;
			args.push_back("--script");
			args.push_back("path/to/script.gd");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.application_type == ApplicationType::SCRIPT);
			CHECK(configuration.script_path == args[1]);
			CHECK(configuration.run_tool == false);
		}

		SUBCASE("Script shortcut argument.") {
			Vector<String> args;
			args.push_back("-s");
			args.push_back("path/to/script.gd");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.application_type == ApplicationType::SCRIPT);
			CHECK(configuration.script_path == args[1]);
			CHECK(configuration.run_tool == false);
		}

		SUBCASE("Script with check-only.") {
			Vector<String> args;
			args.push_back("--script");
			args.push_back("path/to/script.gd");
			args.push_back("--check-only");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.run_tool == true);
			CHECK(configuration.selected_tool == StandaloneTool::VALIDATE_SCRIPT);
			CHECK(configuration.script_path == args[1]);
		}

		ERR_PRINT_ON;
	}

	TEST_CASE("[ApplicationConfiguration] DocTool.") {
		SUBCASE("Doctool without parameters.") {
			Vector<String> args;
			args.push_back("--doctool");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.doc_tool_path == ".");
			CHECK(configuration.doc_base_types == true);
		}

		SUBCASE("Doctool with regular parameters.") {
			Vector<String> args;
			args.push_back("--doctool");
			args.push_back("path/to/somewhere");
			args.push_back("--no-docbase");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.doc_tool_path == args[1]);
			CHECK(configuration.doc_base_types == false);
		}

		SUBCASE("Doctool without path.") {
			Vector<String> args;
			args.push_back("--doctool");
			args.push_back("--no-docbase");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.doc_tool_path == ".");
			CHECK(configuration.doc_base_types == false);
		}

		SUBCASE("Doctool without options.") {
			Vector<String> args;
			args.push_back("--doctool");
			args.push_back(".");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.doc_tool_path == args[1]);
			CHECK(configuration.doc_base_types == true);
		}
	}

	TEST_CASE("[ApplicationConfiguration] Remote file system.") {
		SUBCASE("Remote file sytem with port should extract address and port.") {
			Vector<String> args;
			args.push_back("--remote-fs");
			args.push_back("localhost:8000");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.remote_filesystem_address == "localhost");
			CHECK(configuration.remote_filesystem_port == 8000);
		}

		SUBCASE("Remote file sytem without port should assign default port.") {
			Vector<String> args;
			args.push_back("--remote-fs");
			args.push_back("localhost");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.remote_filesystem_address == "localhost");
			CHECK(configuration.remote_filesystem_port == 6010);
		}

		SUBCASE("Remote file sytem password.") {
			Vector<String> args;
			args.push_back("--remote-fs-password");
			args.push_back("secret");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.remote_filesystem_password == args[1]);
		}
	}

	TEST_CASE("[ApplicationConfiguration] Position") {
		SUBCASE("Should set valid window position.") {
			Vector<String> args;
			args.push_back("--position");
			args.push_back("199,188");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.window_position == Point2i(199, 188));
		}

		SUBCASE("Should fail with invalid window position format.") {
			Vector<String> args;
			args.push_back("--position");
			args.push_back("199x188");

			ERR_PRINT_OFF;
			PARSE_CONFIG();
			ERR_PRINT_ON;

			// Assert
			CHECK(error == ERR_INVALID_PARAMETER);
		}

		SUBCASE("Should fail with invalid window position format.") {
			Vector<String> args;
			args.push_back("--position");
			args.push_back("199");

			ERR_PRINT_OFF;
			PARSE_CONFIG();
			ERR_PRINT_ON;

			// Assert
			CHECK(error == ERR_INVALID_PARAMETER);
		}
	}

	TEST_CASE("[ApplicationConfiguration] Resolution") {
		SUBCASE("Should set valid window size.") {
			Vector<String> args;
			args.push_back("--resolution");
			args.push_back("188x188");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.window_size == Size2i(188, 188));
		}

		SUBCASE("Should fail with invalid window position format.") {
			Vector<String> args;
			args.push_back("--resolution");
			args.push_back("188,188");

			ERR_PRINT_OFF;
			PARSE_CONFIG();
			ERR_PRINT_ON;

			// Assert
			CHECK(error == ERR_INVALID_PARAMETER);
		}

		SUBCASE("Should fail with invalid window height.") {
			Vector<String> args;
			args.push_back("--resolution");
			args.push_back("188,-188");

			ERR_PRINT_OFF;
			PARSE_CONFIG();
			ERR_PRINT_ON;

			// Assert
			CHECK(error == ERR_INVALID_PARAMETER);
		}

		SUBCASE("Should fail with invalid window width.") {
			Vector<String> args;
			args.push_back("--resolution");
			args.push_back("0,188");

			ERR_PRINT_OFF;
			PARSE_CONFIG();
			ERR_PRINT_ON;

			// Assert
			CHECK(error == ERR_INVALID_PARAMETER);
		}
	}

	TEST_CASE("[ApplicationConfiguration] Render thread.") {
		SUBCASE("No parameter") {
			Vector<String> args;
			args.push_back("--render-thread");

			ERR_PRINT_OFF;
			PARSE_CONFIG();
			ERR_PRINT_ON;

			// Assert
			CHECK(error == ERR_INVALID_PARAMETER);
		}

		SUBCASE("Safe") {
			Vector<String> args;
			args.push_back("--render-thread");
			args.push_back("safe");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.render_thread_mode == OS::RENDER_THREAD_SAFE);
		}

		SUBCASE("Unsafe") {
			Vector<String> args;
			args.push_back("--render-thread");
			args.push_back("unsafe");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.render_thread_mode == OS::RENDER_THREAD_UNSAFE);
		}

		SUBCASE("Separate") {
			Vector<String> args;
			args.push_back("--render-thread");
			args.push_back("separate");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.render_thread_mode == OS::RENDER_SEPARATE_THREAD);
		}

		SUBCASE("Incorrect parameter") {
			Vector<String> args;
			args.push_back("--render-thread");
			args.push_back("somethingelse");

			ERR_PRINT_OFF;
			PARSE_CONFIG();
			ERR_PRINT_ON;

			CHECK(error == ERR_INVALID_PARAMETER);
		}
	}

	TEST_CASE("[ApplicationConfiguration] Individual flags.") {
		SUBCASE("Start project manager.") {
			Vector<String> args;
			args.push_back("--project-manager");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.application_type == ApplicationType::PROJECT_MANAGER);
		}

		SUBCASE("Start project manager. (shorthand)") {
			Vector<String> args;
			args.push_back("-p");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.application_type == ApplicationType::PROJECT_MANAGER);
		}

		SUBCASE("Start editor.") {
			Vector<String> args;
			args.push_back("--editor");

			PARSE_CONFIG();

			// Assert
			CHECK(error == OK);
			CHECK(configuration.application_type == ApplicationType::EDITOR);
		}

		SUBCASE("Start editor. (shorthand)") {
			Vector<String> args;
			args.push_back("-e");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.application_type == ApplicationType::EDITOR);
		}

		SUBCASE("Enable verbose printing.") {
			Vector<String> args;
			args.push_back("--verbose");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.output_verbosity == OutputVerbosity::VERBOSE);
		}

		SUBCASE("Enable verbose printing. (shorthand)") {
			Vector<String> args;
			args.push_back("-v");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.output_verbosity == OutputVerbosity::VERBOSE);
		}

		SUBCASE("Disable printing.") {
			Vector<String> args;
			args.push_back("--quiet");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.output_verbosity == OutputVerbosity::QUIET);
		}

		SUBCASE("Enable scanning folders upwards.") {
			Vector<String> args;
			args.push_back("--upwards");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.scan_folders_upwards);
		}

		SUBCASE("Enable profiler.") {
			Vector<String> args;
			args.push_back("--profiling");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.enable_debug_profiler);
		}

		SUBCASE("Enable GPU profiler.") {
			Vector<String> args;
			args.push_back("--profile-gpu");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.enable_gpu_profiler);
		}

		SUBCASE("Enable Vulkan validation layers.") {
			Vector<String> args;
			args.push_back("--vk-layers");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.use_validation_layers);
		}

		SUBCASE("Disable crash handler.") {
			Vector<String> args;
			args.push_back("--disable-crash-handler");

			PARSE_CONFIG();

			CHECK(error == OK);
			CHECK(configuration.disable_crash_handler);
		}
	}
}

TEST_SUITE("[ApplicationConfiguration] Finalize") {
	TEST_CASE("[ApplicationConfiguration] Finalize Project") {
		ERR_PRINT_OFF;

		ApplicationConfiguration configuration;
		configuration.application_type = ApplicationType::PROJECT;

		SUBCASE("Window size") {
			ProjectSettings::get_singleton()->set("display/window/size/test_width", 700);
			ProjectSettings::get_singleton()->set("display/window/size/test_height", 700);
			ProjectSettings::get_singleton()->set("display/window/size/width", 900);
			ProjectSettings::get_singleton()->set("display/window/size/height", 900);
			ProjectSettings::get_singleton()->set("application/run/main_scene", "prevent_fail.scn");

			SUBCASE("Requested window size should not be overridden.") {
				// Arrange
				const Size2i requested_size = Size2i(100, 100);
				configuration.window_size = requested_size;
				configuration.forced_window_size = true;

				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.window_size == requested_size);
			}

			SUBCASE("Window size should be set to test size if possible.") {
				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.window_size == Size2i(700, 700));
			}

			SUBCASE("Window size should be set to regular size if test size is unavailable.") {
				// Arrange
				ProjectSettings::get_singleton()->set("display/window/size/test_width", 0);
				ProjectSettings::get_singleton()->set("display/window/size/test_height", 0);

				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.window_size == Size2i(900, 900));
			}
		}

		SUBCASE("Window flags") {
			ProjectSettings::get_singleton()->set("display/window/size/resizable", true);
			ProjectSettings::get_singleton()->set("display/window/size/borderless", false);
			ProjectSettings::get_singleton()->set("display/window/size/always_on_top", false);
			ProjectSettings::get_singleton()->set("display/window/size/fullscreen", false);

			SUBCASE("Should not set any window flags if not configured.") {
				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.window_flags == 0);
				CHECK(configuration.window_mode == DisplayServer::WINDOW_MODE_WINDOWED);
			}

			SUBCASE("Should set resizable window flag.") {
				// Arrange
				ProjectSettings::get_singleton()->set("display/window/size/resizable", false);

				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.window_flags == DisplayServer::WINDOW_FLAG_RESIZE_DISABLED_BIT);
			}

			SUBCASE("Should set borderless window flag.") {
				// Arrange
				ProjectSettings::get_singleton()->set("display/window/size/borderless", true);

				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.window_flags == DisplayServer::WINDOW_FLAG_BORDERLESS_BIT);
			}

			SUBCASE("Should set always_on_top window flag.") {
				// Arrange
				ProjectSettings::get_singleton()->set("display/window/size/always_on_top", true);

				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.window_flags == DisplayServer::WINDOW_FLAG_ALWAYS_ON_TOP_BIT);
			}

			SUBCASE("Should set fullscreen window flag.") {
				// Arrange
				ProjectSettings::get_singleton()->set("display/window/size/fullscreen", true);

				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.window_mode == DisplayServer::WINDOW_MODE_FULLSCREEN);
			}
		}

		SUBCASE("Scene") {
			ProjectSettings::get_singleton()->set("application/run/main_scene", "the_scene.scn");

			SUBCASE("Should load requested scene.") {
				// Arrange
				const String the_other_scene = "the_other_scene.tscn";
				configuration.scene_path = the_other_scene;

				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.scene_path == the_other_scene);
			}

			SUBCASE("Should load project main scene if no scene requested.") {
				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == OK);
				CHECK(configuration.scene_path == "the_scene.scn");
			}

			SUBCASE("Should load project main scene if no scene requested.") {
				// Arrange
				ProjectSettings::get_singleton()->set("application/run/main_scene", "");

				// Act
				Error error = finalize_configuration(configuration);

				// Assert
				CHECK(error == ERR_INVALID_PARAMETER);
			}
		}

		ERR_PRINT_ON;
	}

	TEST_CASE("[ApplicationConfiguration] Finalize Project Manager") {
		ERR_PRINT_OFF;

		ApplicationConfiguration configuration;
		configuration.application_type = ApplicationType::PROJECT_MANAGER;

		SUBCASE("Window size should be set to requested size.") {
			// Arrange
			const Size2i requested_size = Size2i(1000, 1000);
			configuration.window_size = requested_size;

			// Act
			Error error = finalize_configuration(configuration);

			// Assert
			CHECK(error == OK);
			CHECK(configuration.window_size == requested_size);
		}

		SUBCASE("Window size should be set to default if not requested.") {
			// Act
			Error error = finalize_configuration(configuration);

			// Assert
			CHECK(error == OK);
			CHECK(configuration.window_size == Size2i(1024, 600));
		}

		ERR_PRINT_ON;
	}

	TEST_CASE("[ApplicationConfiguration] Finalize Editor") {
		ERR_PRINT_OFF;

		ApplicationConfiguration configuration;
		configuration.application_type = ApplicationType::EDITOR;

		SUBCASE("Should not set window mode to maximized if requested.") {
			// Arrange
			configuration.window_mode = DisplayServer::WINDOW_MODE_FULLSCREEN;
			configuration.forced_window_mode = true;

			// Act
			Error error = finalize_configuration(configuration);

			// Assert
			CHECK(error == OK);
			CHECK(configuration.window_mode == DisplayServer::WINDOW_MODE_FULLSCREEN);
		}

		SUBCASE("Should set window mode to maximized by default.") {
			// Act
			Error error = finalize_configuration(configuration);

			// Assert
			CHECK(error == OK);
			CHECK(configuration.window_mode == DisplayServer::WINDOW_MODE_MAXIMIZED);
		}

		ERR_PRINT_ON;
	}
}

} // namespace TestApplicationConfiguration

#endif // TEST_APPLICATION_CONFIGURATION_H
