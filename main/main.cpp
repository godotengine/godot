/*************************************************************************/
/*  main.cpp                                                             */
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

#include "main.h"

#include "application_configuration.h"
#include "core/config/project_settings.h"
#include "core/core_string_names.h"
#include "core/crypto/crypto.h"
#include "core/debugger/engine_debugger.h"
#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/io/file_access_network.h"
#include "core/io/file_access_pack.h"
#include "core/io/file_access_zip.h"
#include "core/io/image_loader.h"
#include "core/io/ip.h"
#include "core/io/resource_loader.h"
#include "core/object/message_queue.h"
#include "core/os/dir_access.h"
#include "core/os/os.h"
#include "core/register_core_types.h"
#include "core/string/translation.h"
#include "core/version.h"
#include "core/version_hash.gen.h"
#include "drivers/register_driver_types.h"
#include "main/app_icon.gen.h"
#include "main/main_timer_sync.h"
#include "main/performance.h"
#include "main/splash.gen.h"
#include "main/splash_editor.gen.h"
#include "modules/modules_enabled.gen.h"
#include "modules/register_module_types.h"
#include "platform/register_platform_apis.h"
#include "scene/main/window.h"
#include "scene/register_scene_types.h"
#include "scene/resources/packed_scene.h"
#include "servers/audio_server.h"
#include "servers/camera_server.h"
#include "servers/display_server.h"
#include "servers/navigation_server_2d.h"
#include "servers/navigation_server_3d.h"
#include "servers/physics_server_2d.h"
#include "servers/physics_server_3d.h"
#include "servers/register_server_types.h"
#include "servers/rendering/rendering_server_default.h"
#include "servers/text_server.h"
#include "servers/xr_server.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/project_manager.h"
#include "standalone_tools.h"
#endif

#ifdef TESTS_ENABLED
#include "tests/test_main.h"
#endif

/* Singletons */
// Initialized in setup()
static Engine *engine = nullptr;
static ProjectSettings *project_settings = nullptr;
static Input *input = nullptr;
static InputMap *input_map = nullptr;
static TranslationServer *translation_server = nullptr;
static Performance *performance = nullptr;
static PackedData *packed_data = nullptr;
static FileAccessNetworkClient *file_access_network_client = nullptr;
static MessageQueue *message_queue = nullptr;
#ifdef MINIZIP_ENABLED
static ZipArchive *zip_packed_data = nullptr;
#endif

// Initialized in finish_setup()
static AudioServer *audio_server = nullptr;
static DisplayServer *display_server = nullptr;
static RenderingServer *rendering_server = nullptr;
static CameraServer *camera_server = nullptr;
static XRServer *xr_server = nullptr;
static TextServerManager *text_server_manager = nullptr;
static PhysicsServer3D *physics_server_3d = nullptr;
static PhysicsServer2D *physics_server_2d = nullptr;
static NavigationServer3D *navigation_server = nullptr;
static NavigationServer2D *navigation_2d_server = nullptr;

static ApplicationConfiguration configuration;
static bool is_tool_application;

// We error out if finish_setup() doesn't turn this true
static bool _setup_completed = false;

// Everything the main loop needs to know about frame timings
static MainTimerSync main_timer_sync;
static uint64_t last_ticks = 0;
static uint32_t frames = 0;
static uint32_t frame = 0;
static bool force_redraw_requested = false;
static int iterating = 0;

// For performance metrics.
static uint64_t physics_process_max = 0;
static uint64_t process_max = 0;

// TODO: these methods can't be defined in main.h, because including scene_tree.h
// conflicts with Window on X11.
SceneTree *create_scene_tree(MainLoop *p_main_loop);
MainLoop *create_main_loop(const String &p_main_loop_type);

//#define DEBUG_INIT
#ifdef DEBUG_INIT
#define MAIN_PRINT(m_txt) print_line(m_txt)
#else
#define MAIN_PRINT(m_txt)
#endif

int Main::test_entrypoint(int argc, char *argv[], bool &finished) {
#ifdef TESTS_ENABLED
	for (int x = 0; x < argc; x++) {
		if ((strncmp(argv[x], "--test", 6) == 0) && (strlen(argv[x]) == 6)) {
			finished = true;
			test_setup();
			int status = test_main(argc, argv);
			test_cleanup();
			return status;
		}
	}
#endif // TESTS_ENABLED
	finished = false;
	return 0;
}

/* Engine initialization
 *
 * Consists of several methods that are called by each platform's specific main(argc, argv).
 * To fully understand engine init, one should therefore start from the platform's main and
 * see how it calls into the Main class' methods.
 *
 * The initialization is typically done in 3 steps (with the finish_setup step triggered either
 * automatically by setup, or manually in the platform's main).
 *
 * - setup(exec_path, argc, argv, p_finish_setup) is the main entry point for all platforms,
 *   responsible for the initialization of all low level singletons and core types according
 *   to command line arguments.
 *   If <p_finish_setup> is true, it will chain into finish_setup() (default behaviour). This is
 *   disabled on some platforms (Android, iOS, UWP) which trigger the second step in their
 *   own time.
 *
 * - p_finish_setup(p_main_tid_override) registers high level servers and singletons, displays the
 *   boot splash, then registers higher level types (scene, editor, etc.).
 *
 * - start() is the last step and either creates a main loop, or runs a standalone tool.
 */
Error Main::setup(const char *exec_path, int argc, char *argv[], bool p_finish_setup) {
	OS::get_singleton()->initialize();

	MAIN_PRINT("Main: Parse CMDLine");
	Error error = parse_configuration(exec_path, argc, argv, configuration);
	if (error != OK) {
		deinitialize_early(false);
		return error;
	}

	MAIN_PRINT("Main: Setup Core");
	// Create necessary core singletons for the following steps.
	{
		engine = memnew(Engine);
		register_core();
		input_map = memnew(InputMap);
		project_settings = memnew(ProjectSettings);
		packed_data = memnew(PackedData);
	}

	error = load_project_settings();
	if (error != OK) {
		deinitialize_early();
		return error;
	}

	print_line(get_program_string());

	// Validate and manually adjust the configuration.
	finalize_configuration(configuration);
	if (error != OK) {
		deinitialize_early();
		return error;
	}

	MAIN_PRINT("Main: Initialize Core");
	initialize_core();

	if (p_finish_setup) {
		return finish_setup();
	}

	return OK;
}

bool Main::start() {
	ERR_FAIL_COND_V(!_setup_completed, false);

	// If configured, run a tool and exit the application.
	if (configuration.run_tool) {
		switch (configuration.selected_tool) {
			case StandaloneTool::VALIDATE_SCRIPT: {
				Ref<Script> script_res = ResourceLoader::load(configuration.script_path);
				ERR_FAIL_COND_V_MSG(script_res.is_null(), false, "Can't load script: " + configuration.script_path);

				if (!script_res->is_valid()) {
					OS::get_singleton()->set_exit_code(EXIT_FAILURE);
				}
			} break;
#ifdef TOOLS_ENABLED
			case StandaloneTool::DOC_TOOL: {
				run_doc_tool(configuration.doc_tool_path, configuration.doc_base_types);
			} break;
#endif
			default:
				break;
		}

		return false;
	}

	// Create a MainLoop
	// Usually main loop will be a a SceneTree. However, scripts and project settings
	// can overwrite this. If nothing is configured we default to SceneTree.
	String main_loop_type = GLOBAL_GET("application/run/main_loop_type");

	const bool is_scene_tree = configuration.application_type != ApplicationType::SCRIPT && (main_loop_type == "SceneTree" || main_loop_type == "");
	MainLoop *main_loop = is_scene_tree ? memnew(SceneTree) : create_main_loop(main_loop_type);
	ERR_FAIL_COND_V_MSG(!main_loop, false, "Main loop could not be created.");

	if (is_scene_tree || main_loop->is_class("SceneTree")) {
		SceneTree *scene_tree = create_scene_tree(main_loop);

		// Set the root node.
		switch (configuration.application_type) {
#ifdef TOOLS_ENABLED
			case ApplicationType::EDITOR: {
				DisplayServer::get_singleton()->set_context(DisplayServer::CONTEXT_EDITOR);

				EditorNode *editor_node = memnew(EditorNode);
				scene_tree->get_root()->add_child(editor_node);

				if (configuration.selected_tool == StandaloneTool::EXPORT) {
					editor_node->export_preset(configuration.export_config.preset, configuration.export_config.path, configuration.export_config.debug_build, configuration.export_config.pack_only);
				}

				if (configuration.scene_path != String(GLOBAL_GET("application/run/main_scene")) || !editor_node->has_scenes_in_session()) {
					Error serr = editor_node->load_scene(localize_scene_path(configuration.scene_path));
					if (serr != OK) {
						ERR_PRINT("Failed to load scene.");
					}
				}
			} break;
			case ApplicationType::PROJECT_MANAGER: {
				DisplayServer::get_singleton()->set_context(DisplayServer::CONTEXT_PROJECTMAN);
				scene_tree->get_root()->add_child(memnew(ProjectManager));
			} break;
#endif
			case ApplicationType::PROJECT: {
				DisplayServer::get_singleton()->set_context(DisplayServer::CONTEXT_ENGINE);

				String localized_scene_path = localize_scene_path(configuration.scene_path);
				Node *scene = nullptr;
				Ref<PackedScene> packed_scene = ResourceLoader::load(localized_scene_path);
				if (packed_scene.is_valid()) {
					scene = packed_scene->instance();
				}
				ERR_FAIL_COND_V_MSG(!scene, false, "Failed loading scene: " + localized_scene_path);

				scene_tree->add_current_scene(scene);
			} break;
			case ApplicationType::SCRIPT: {
				DisplayServer::get_singleton()->set_context(DisplayServer::CONTEXT_ENGINE);
			} break;
			default:
				break;
		}
	}

	OS::get_singleton()->set_main_loop(main_loop);
	main_timer_sync.init(OS::get_singleton()->get_ticks_usec());
	return true;
}

/* Main iteration
 *
 * Advances the state of physics, rendering and audio. It's called directly
 * by the platform's OS::run method, where the loop is created and monitored.
 *
 * The OS implementation can impact its draw step with the Main::force_redraw() method.
 */
bool Main::iteration() {
	//for now do not error on this
	//ERR_FAIL_COND_V(iterating, false);

	iterating++;

	uint64_t ticks = OS::get_singleton()->get_ticks_usec();
	Engine::get_singleton()->_frame_ticks = ticks;
	main_timer_sync.set_cpu_ticks_usec(ticks);
	main_timer_sync.set_fixed_fps(configuration.fixed_fps);

	uint64_t ticks_elapsed = ticks - last_ticks;

	int physics_fps = Engine::get_singleton()->get_iterations_per_second();
	float physics_step = 1.0 / physics_fps;

	float time_scale = Engine::get_singleton()->get_time_scale();

	MainFrameTime advance = main_timer_sync.advance(physics_step, physics_fps);
	double process_step = advance.process_step;
	double scaled_step = process_step * time_scale;

	Engine::get_singleton()->_process_step = process_step;
	Engine::get_singleton()->_physics_interpolation_fraction = advance.interpolation_fraction;

	uint64_t physics_process_ticks = 0;
	uint64_t process_ticks = 0;

	frame += ticks_elapsed;

	last_ticks = ticks;

	static const int max_physics_steps = 8;
	if (configuration.fixed_fps == -1 && advance.physics_steps > max_physics_steps) {
		process_step -= (advance.physics_steps - max_physics_steps) * physics_step;
		advance.physics_steps = max_physics_steps;
	}

	bool exit = false;

	Engine::get_singleton()->_in_physics = true;

	for (int iters = 0; iters < advance.physics_steps; ++iters) {
		uint64_t physics_begin = OS::get_singleton()->get_ticks_usec();

		PhysicsServer3D::get_singleton()->sync();
		PhysicsServer3D::get_singleton()->flush_queries();

		PhysicsServer2D::get_singleton()->sync();
		PhysicsServer2D::get_singleton()->flush_queries();

		if (OS::get_singleton()->get_main_loop()->physics_process(physics_step * time_scale)) {
			exit = true;
			break;
		}

		NavigationServer3D::get_singleton_mut()->process(physics_step * time_scale);

		message_queue->flush();

		PhysicsServer3D::get_singleton()->end_sync();
		PhysicsServer3D::get_singleton()->step(physics_step * time_scale);

		PhysicsServer2D::get_singleton()->end_sync();
		PhysicsServer2D::get_singleton()->step(physics_step * time_scale);

		message_queue->flush();

		physics_process_ticks = MAX(physics_process_ticks, OS::get_singleton()->get_ticks_usec() - physics_begin); // keep the largest one for reference
		physics_process_max = MAX(OS::get_singleton()->get_ticks_usec() - physics_begin, physics_process_max);
		Engine::get_singleton()->_physics_frames++;
	}

	Engine::get_singleton()->_in_physics = false;

	uint64_t process_begin = OS::get_singleton()->get_ticks_usec();

	if (OS::get_singleton()->get_main_loop()->process(process_step * time_scale)) {
		exit = true;
	}
	message_queue->flush();

	RenderingServer::get_singleton()->sync(); //sync if still drawing from previous frames.

	if (DisplayServer::get_singleton()->can_any_window_draw() &&
			RenderingServer::get_singleton()->is_render_loop_enabled()) {
		if (!force_redraw_requested && OS::get_singleton()->is_in_low_processor_usage_mode()) {
			if (RenderingServer::get_singleton()->has_changed()) {
				RenderingServer::get_singleton()->draw(true, scaled_step); // flush visual commands
				Engine::get_singleton()->frames_drawn++;
			}
		} else {
			RenderingServer::get_singleton()->draw(true, scaled_step); // flush visual commands
			Engine::get_singleton()->frames_drawn++;
			force_redraw_requested = false;
		}
	}

	process_ticks = OS::get_singleton()->get_ticks_usec() - process_begin;
	process_max = MAX(process_ticks, process_max);
	uint64_t frame_time = OS::get_singleton()->get_ticks_usec() - ticks;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->frame();
	}

	AudioServer::get_singleton()->update();

	if (EngineDebugger::is_active()) {
		EngineDebugger::get_singleton()->iteration(frame_time, process_ticks, physics_process_ticks, physics_step);
	}

	frames++;
	Engine::get_singleton()->_process_frames++;

	if (frame > 1000000) {
		if (is_tool_application) {
#ifdef TOOLS_ENABLED
			if (configuration.print_fps) {
				print_line(vformat("Editor FPS: %d (%s mspf)", frames, rtos(1000.0 / frames).pad_decimals(1)));
			}
#endif
		} else if (GLOBAL_GET("debug/settings/stdout/print_fps") || configuration.print_fps) {
			print_line(vformat("Project FPS: %d (%s mspf)", frames, rtos(1000.0 / frames).pad_decimals(1)));
		}

		Engine::get_singleton()->_fps = frames;
		Performance::get_singleton()->set_process_time(USEC_TO_SEC(process_max));
		Performance::get_singleton()->set_physics_process_time(USEC_TO_SEC(physics_process_max));
		process_max = 0;
		physics_process_max = 0;

		frame %= 1000000;
		frames = 0;
	}

	iterating--;

	if (configuration.fixed_fps != -1) {
		return exit;
	}

	OS::get_singleton()->add_frame_delay(DisplayServer::get_singleton()->window_can_draw());

#ifdef TOOLS_ENABLED
	if (configuration.auto_build_solutions) {
		configuration.auto_build_solutions = false;
		// Only relevant when running the editor.
		ERR_FAIL_V_MSG((configuration.application_type != ApplicationType::EDITOR), "Command line option --build-solutions was passed, but no project is being edited. Aborting.");
		ERR_FAIL_V_MSG(!EditorNode::get_singleton()->call_build(), "Command line option --build-solutions was passed, but the build callback failed. Aborting.");
	}
#endif

	return exit || configuration.auto_quit;
}

void Main::deinitialize_early(bool p_unregister_core) {
	if (p_unregister_core) {
		unregister_core();
	}

	if (file_access_network_client) {
		memdelete(file_access_network_client);
	}
	if (packed_data) {
		memdelete(packed_data);
	}
	if (project_settings) {
		memdelete(project_settings);
	}
	if (engine) {
		memdelete(engine);
	}

	configuration.user_args.clear();
	configuration.main_args.clear();
	OS::get_singleton()->_cmdline.clear();

	OS::get_singleton()->finalize_core();
}

/* Engine deinitialization
 *
 * Responsible for freeing all the memory allocated by previous setup steps,
 * so that the engine closes cleanly without leaking memory or crashing.
 * The order matters as some of those steps are linked with each other.
 */
void Main::cleanup(bool p_force) {
	if (!p_force) {
		ERR_FAIL_COND(!_setup_completed);
	}

	// The order of deallocating matters as some of these steps are linked with each other.

	EngineDebugger::deinitialize();

	ResourceLoader::remove_custom_loaders();
	ResourceSaver::remove_custom_savers();

	// Flush before uninitializing the scene, but delete the MessageQueue as late as possible.
	message_queue->flush();

	OS::get_singleton()->delete_main_loop();

	OS::get_singleton()->_cmdline.clear();
	OS::get_singleton()->_execpath = "";
	OS::get_singleton()->_local_clipboard = "";

	ResourceLoader::clear_translation_remaps();
	ResourceLoader::clear_path_remaps();

	ScriptServer::finish_languages();

	// Sync pending commands that may have been queued from a different thread during ScriptServer finalization
	RenderingServer::get_singleton()->sync();

	//clear global shader variables before scene and other graphics stuff are deinitialized.
	rendering_server->global_variables_clear();

#ifdef TOOLS_ENABLED
	EditorNode::unregister_editor_types();
#endif

	if (xr_server) {
		// cleanup now before we pull the rug from underneath...
		memdelete(xr_server);
	}

	ImageLoader::cleanup();

	unregister_module_types();
	unregister_platform_apis();
	unregister_scene_types();
	unregister_server_types();

	if (audio_server) {
		audio_server->finish();
		memdelete(audio_server);
	}

	if (camera_server) {
		memdelete(camera_server);
	}

	OS::get_singleton()->finalize();

	{
		physics_server_3d->finish();
		memdelete(physics_server_3d);

		physics_server_2d->finish();
		memdelete(physics_server_2d);
	}
	{
		memdelete(navigation_server);
		navigation_server = nullptr;

		memdelete(navigation_2d_server);
		navigation_2d_server = nullptr;
	}
	{
		rendering_server->finish();
		memdelete(rendering_server);

		memdelete(display_server);
	}

	if (text_server_manager) {
		memdelete(text_server_manager);
	}

	if (input) {
		memdelete(input);
	}

	if (packed_data) {
		memdelete(packed_data);
	}
	if (file_access_network_client) {
		memdelete(file_access_network_client);
	}
	if (performance) {
		memdelete(performance);
	}
	if (input_map) {
		memdelete(input_map);
	}
	if (translation_server) {
		memdelete(translation_server);
	}
	if (project_settings) {
		memdelete(project_settings);
	}
	if (engine) {
		memdelete(engine);
	}

	if (OS::get_singleton()->is_restart_on_exit_set()) {
		//attempt to restart with arguments
		String exec = OS::get_singleton()->get_executable_path();
		List<String> args = OS::get_singleton()->get_restart_on_exit_arguments();
		OS::get_singleton()->create_process(exec, args);
		OS::get_singleton()->set_restart_on_exit(false, List<String>()); //clear list (uses memory)
	}

	// Now should be safe to delete MessageQueue (famous last words).
	message_queue->flush();
	memdelete(message_queue);

	unregister_core_driver_types();
	unregister_core_types();
	unregister_core();

	OS::get_singleton()->finalize_core();
}

bool Main::is_iterating() {
	return iterating > 0;
}

void Main::force_redraw() {
	force_redraw_requested = true;
}

// Used by Mono module, should likely be registered in Engine singleton instead
bool Main::is_project_manager() {
	return configuration.application_type == ApplicationType::PROJECT_MANAGER;
};

Error Main::load_project_settings() {
	// Try to load project settings.
	if (configuration.application_type == ApplicationType::PROJECT || configuration.application_type == ApplicationType::EDITOR) {
		bool failed = false;

		// Network file system needs to be configured before project settings, since project settings are based on the
		// 'project.godot' file which will only be available through the network if this is enabled
		FileAccessNetwork::configure();
		if (configuration.remote_filesystem_address != "") {
			file_access_network_client = memnew(FileAccessNetworkClient);

			Error err = file_access_network_client->connect(configuration.remote_filesystem_address, configuration.remote_filesystem_port, configuration.remote_filesystem_password);
			if (err) {
				OS::get_singleton()->printerr("Could not connect to remote filesystem: %s:%i.\n", configuration.remote_filesystem_address.utf8().get_data(), configuration.remote_filesystem_port);
				failed = true;
			}

			FileAccess::make_default<FileAccessNetwork>(FileAccess::ACCESS_RESOURCES);
		}

		if (project_settings->setup(configuration.project_path, configuration.main_pack, configuration.scan_folders_upwards) != OK) {
			failed = true;
#ifndef TOOLS_ENABLED
			const String error_message = vformat("Error: Couldn't load project data at '%s'. Is the .pck file missing?\nIf you've renamed the executable, the associated .pck file should also be renamed to match the executable's name (without the extension).\n", configuration.project_path.utf8().get_data());
			DisplayServer::get_singleton()->alert(error_message);
			OS::get_singleton()->print("%s", error_message.utf8().get_data());
#endif
		}

		if (failed) {
#ifdef TOOLS_ENABLED
			configuration.application_type = ApplicationType::PROJECT_MANAGER;
#else
			ERR_FAIL_COND_V(failed, ERR_INVALID_PARAMETER);
#endif
		}
	}

	// Now that program type won't change anymore, load default project settings for the remaining settings.
	is_tool_application = configuration.application_type == ApplicationType::PROJECT_MANAGER || configuration.application_type == ApplicationType::EDITOR;
	load_default_project_settings();

	// Needs to be done before initializing anything.
	if (configuration.output_verbosity == OutputVerbosity::QUIET || bool(GLOBAL_GET("application/run/disable_stdout"))) {
		_print_line_enabled = false;
	}
	if (bool(GLOBAL_GET("application/run/disable_stderr"))) {
		_print_error_enabled = false;
	};

	return OK;
}

void Main::load_default_project_settings() {
	GLOBAL_DEF_BASIC("display/window/size/width", 1024);
	GLOBAL_DEF_BASIC("display/window/size/height", 600);

	if (!is_tool_application) {
		GLOBAL_DEF_BASIC("display/window/stretch/mode", "disabled");
		GLOBAL_DEF_BASIC("display/window/stretch/aspect", "ignore");

		GLOBAL_DEF("application/config/auto_accept_quit", true);
		GLOBAL_DEF("application/config/quit_on_go_back", true);

		GLOBAL_DEF("gui/common/snap_controls_to_pixels", true);
		GLOBAL_DEF("gui/fonts/dynamic_fonts/use_oversampling", true);

		GLOBAL_DEF("rendering/textures/canvas_textures/default_texture_filter", 1);
		GLOBAL_DEF("rendering/textures/canvas_textures/default_texture_repeat", 0);
	} else {
#ifdef TOOLS_ENABLED
		GLOBAL_DEF_BASIC("display/window/stretch/mode", "disabled");
		GLOBAL_DEF_BASIC("display/window/stretch/aspect", "ignore");
		GLOBAL_DEF_BASIC("display/window/stretch/shrink", 1.0);
		GLOBAL_DEF("application/config/auto_accept_quit", true);
		GLOBAL_DEF("application/config/quit_on_go_back", true);
		GLOBAL_DEF_BASIC("gui/common/snap_controls_to_pixels", true);
		GLOBAL_DEF_BASIC("gui/fonts/dynamic_fonts/use_oversampling", true);
		GLOBAL_DEF_BASIC("rendering/textures/canvas_textures/default_texture_filter", 1);
		GLOBAL_DEF_BASIC("rendering/textures/canvas_textures/default_texture_repeat", 0);

		ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/mode",
				PropertyInfo(Variant::STRING, "display/window/stretch/mode", PROPERTY_HINT_ENUM, "disabled,canvas_items,viewport"));

		ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/aspect",
				PropertyInfo(Variant::STRING, "display/window/stretch/aspect", PROPERTY_HINT_ENUM, "ignore,keep,keep_width,keep_height,expand"));

		ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/shrink",
				PropertyInfo(Variant::FLOAT, "display/window/stretch/shrink", PROPERTY_HINT_RANGE, "1.0,8.0,0.1"));

		ProjectSettings::get_singleton()->set_custom_property_info("rendering/textures/canvas_textures/default_texture_filter",
				PropertyInfo(Variant::INT, "rendering/textures/canvas_textures/default_texture_filter", PROPERTY_HINT_ENUM, "Nearest,Linear,Linear Mipmap,Nearest Mipmap"));

		ProjectSettings::get_singleton()->set_custom_property_info("rendering/textures/canvas_textures/default_texture_repeat",
				PropertyInfo(Variant::INT, "rendering/textures/canvas_textures/default_texture_repeat", PROPERTY_HINT_ENUM, "Disable,Enable,Mirror"));
#endif // TOOLS_ENABLED
	}

// Boot splash
#if defined(TOOLS_ENABLED) && !defined(NO_EDITOR_SPLASH)
	GLOBAL_DEF("application/boot_splash/bg_color", is_tool_application ? boot_splash_editor_bg_color : boot_splash_bg_color);
#else
	GLOBAL_DEF("application/boot_splash/bg_color", boot_splash_bg_color);
#endif

	// Display
	GLOBAL_DEF_BASIC("display/window/size/resizable", true);
	GLOBAL_DEF_BASIC("display/window/size/borderless", false);
	GLOBAL_DEF_BASIC("display/window/size/fullscreen", false);
	GLOBAL_DEF("display/window/size/always_on_top", false);
	GLOBAL_DEF("display/window/size/test_width", 0);
	GLOBAL_DEF("display/window/size/test_height", 0);

	GLOBAL_DEF_BASIC("display/window/handheld/orientation", DisplayServer::ScreenOrientation::SCREEN_LANDSCAPE);
	GLOBAL_DEF("display/window/ios/hide_home_indicator", true);
	GLOBAL_DEF("display/window/dpi/allow_hidpi", false);
	GLOBAL_DEF_RST("display/window/vsync/use_vsync", true);
	GLOBAL_DEF("display/window/energy_saving/keep_screen_on", true);
	GLOBAL_DEF("display/window/vsync/vsync_via_compositor", false);
	GLOBAL_DEF("display/window/per_pixel_transparency/allowed", false);
	GLOBAL_DEF("display/window/per_pixel_transparency/enabled", false);
	GLOBAL_DEF("display/window/subwindows/embed_subwindows", false);

	GLOBAL_DEF("display/mouse_cursor/custom_image", String());
	GLOBAL_DEF("display/mouse_cursor/custom_image_hotspot", Vector2());
	GLOBAL_DEF("display/mouse_cursor/tooltip_position_offset", Point2(10, 10));

	// Application
	GLOBAL_DEF("application/run/frame_delay_msec", 0);
	GLOBAL_DEF("application/run/low_processor_mode", false);
	GLOBAL_DEF("application/run/low_processor_mode_sleep_usec", 6900); // Roughly 144 FPS
	// Only flush stdout in debug builds by default, as spamming `print()` will
	// decrease performance if this is enabled.
	GLOBAL_DEF_RST("application/run/flush_stdout_on_print", false);
	GLOBAL_DEF_RST("application/run/flush_stdout_on_print.debug", true);
	GLOBAL_DEF("application/config/icon", String());
	GLOBAL_DEF("application/config/windows_native_icon", String());
	GLOBAL_DEF("application/config/macos_native_icon", String());
	GLOBAL_DEF("application/run/main_loop_type", "SceneTree");
	GLOBAL_DEF("application/boot_splash/image", String());
	GLOBAL_DEF("application/boot_splash/fullsize", true);
	GLOBAL_DEF("application/boot_splash/use_filter", true);

	// Debug
	GLOBAL_DEF("debug/settings/stdout/print_fps", false);
	GLOBAL_DEF("debug/settings/stdout/verbose_stdout", false);
	GLOBAL_DEF("debug/settings/fps/force_fps", 0);
	GLOBAL_DEF("debug/settings/crash_handler/message",
			String("Please include this when reporting the bug on https://github.com/godotengine/godot/issues"));
	GLOBAL_DEF("debug/file_logging/enable_file_logging", false);
	GLOBAL_DEF("debug/file_logging/log_path", "user://logs/godot.log");
	GLOBAL_DEF("debug/file_logging/max_log_files", 5);

	// Only enable file logging by default on desktop platforms as logs can't be
	// accessed easily on mobile/Web platforms (if at all).
	// This also prevents logs from being created for the editor instance, as feature tags
	// are disabled while in the editor (even if they should logically apply).
	GLOBAL_DEF("debug/file_logging/enable_file_logging.pc", true);

	// Input
	GLOBAL_DEF("input_devices/pointing/ios/touch_delay", 0.150);
	GLOBAL_DEF("input_devices/pointing/emulate_mouse_from_touch", true);
	GLOBAL_DEF("input_devices/pointing/emulate_touch_from_mouse", false);
	GLOBAL_DEF_RST_NOVAL("input_devices/pen_tablet/driver", "");
	GLOBAL_DEF_RST_NOVAL("input_devices/pen_tablet/driver.Windows", "");

	// Network
	GLOBAL_DEF("network/ssl/certificate_bundle_override", "");
	GLOBAL_DEF("network/limits/debugger/max_chars_per_second", 32768);
	GLOBAL_DEF("network/limits/debugger/max_queued_messages", 2048);
	GLOBAL_DEF("network/limits/debugger/max_errors_per_second", 400);
	GLOBAL_DEF("network/limits/debugger/max_warnings_per_second", 400);
	GLOBAL_DEF("network/limits/tcp/connect_timeout_seconds", (30));
	GLOBAL_DEF_RST("network/limits/packet_peer_stream/max_buffer_po2", (16));
	GLOBAL_DEF("network/ssl/certificate_bundle_override", "");

	// Other
	GLOBAL_DEF_BASIC("physics/common/physics_fps", 60);
	GLOBAL_DEF("physics/common/physics_jitter_fix", 0.5);

	GLOBAL_DEF("memory/limits/multithreaded_server/rid_pool_prealloc", 60);

	GLOBAL_DEF("internationalization/rendering/force_right_to_left_layout_direction", false);
	GLOBAL_DEF("internationalization/locale/include_text_server_data", false);

	GLOBAL_DEF("rendering/driver/driver_name", "Vulkan");
	GLOBAL_DEF("rendering/driver/threads/thread_model", OS::RENDER_THREAD_SAFE);
	GLOBAL_DEF("rendering/environment/defaults/default_clear_color", Color(0.3, 0.3, 0.3));

#ifdef TOOLS_ENABLED
	// Application
	ProjectSettings::get_singleton()->set_custom_property_info("application/boot_splash/image",
			PropertyInfo(Variant::STRING, "application/boot_splash/image", PROPERTY_HINT_FILE, "*.png"));

	ProjectSettings::get_singleton()->set_custom_property_info("application/config/icon",
			PropertyInfo(Variant::STRING, "application/config/icon", PROPERTY_HINT_FILE, "*.png,*.webp,*.svg,*.svgz"));

	ProjectSettings::get_singleton()->set_custom_property_info("application/config/windows_native_icon",
			PropertyInfo(Variant::STRING, "application/config/windows_native_icon", PROPERTY_HINT_FILE, "*.ico"));

	ProjectSettings::get_singleton()->set_custom_property_info("application/config/macos_native_icon",
			PropertyInfo(Variant::STRING, "application/config/macos_native_icon", PROPERTY_HINT_FILE, "*.icns"));

	// Display
	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/width",
			PropertyInfo(Variant::INT, "display/window/size/width", PROPERTY_HINT_RANGE, "0,7680,or_greater")); // 8K resolution

	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/height",
			PropertyInfo(Variant::INT, "display/window/size/height", PROPERTY_HINT_RANGE, "0,4320,or_greater")); // 8K resolution

	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/test_width",
			PropertyInfo(Variant::INT, "display/window/size/test_width", PROPERTY_HINT_RANGE, "0,7680,or_greater")); // 8K resolution

	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/test_height",
			PropertyInfo(Variant::INT, "display/window/size/test_height", PROPERTY_HINT_RANGE, "0,4320,or_greater")); // 8K resolution

	ProjectSettings::get_singleton()->set_custom_property_info("display/mouse_cursor/custom_image",
			PropertyInfo(Variant::STRING, "display/mouse_cursor/custom_image", PROPERTY_HINT_FILE, "*.png,*.webp"));

	// Debug
	ProjectSettings::get_singleton()->set_custom_property_info("debug/settings/fps/force_fps",
			PropertyInfo(Variant::INT, "debug/settings/fps/force_fps", PROPERTY_HINT_RANGE, "0,120,1,or_greater"));

	ProjectSettings::get_singleton()->set_custom_property_info("debug/file_logging/max_log_files",
			PropertyInfo(Variant::INT, "debug/file_logging/max_log_files", PROPERTY_HINT_RANGE, "0,20,1,or_greater")); //no negative numbers

	ProjectSettings::get_singleton()->set_custom_property_info("application/run/frame_delay_msec",
			PropertyInfo(Variant::INT, "application/run/frame_delay_msec", PROPERTY_HINT_RANGE, "0,100,1,or_greater")); // No negative numbers

	ProjectSettings::get_singleton()->set_custom_property_info("application/run/low_processor_mode_sleep_usec",
			PropertyInfo(Variant::INT, "application/run/low_processor_mode_sleep_usec", PROPERTY_HINT_RANGE, "0,33200,1,or_greater")); // No negative numbers

	ProjectSettings::get_singleton()->set_custom_property_info("memory/limits/multithreaded_server/rid_pool_prealloc",
			PropertyInfo(Variant::INT, "memory/limits/multithreaded_server/rid_pool_prealloc", PROPERTY_HINT_RANGE, "0,500,1")); // No negative and limit to 500 due to crashes

	// Network
	ProjectSettings::get_singleton()->set_custom_property_info("network/ssl/certificate_bundle_override",
			PropertyInfo(Variant::STRING, "network/ssl/certificate_bundle_override", PROPERTY_HINT_FILE, "*.crt"));

	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/tcp/connect_timeout_seconds",
			PropertyInfo(Variant::INT, "network/limits/tcp/connect_timeout_seconds", PROPERTY_HINT_RANGE, "1,1800,1"));

	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/packet_peer_stream/max_buffer_po2",
			PropertyInfo(Variant::INT, "network/limits/packet_peer_stream/max_buffer_po2", PROPERTY_HINT_RANGE, "0,64,1,or_greater"));

	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger/max_chars_per_second",
			PropertyInfo(Variant::INT, "network/limits/debugger/max_chars_per_second", PROPERTY_HINT_RANGE, "0, 4096, 1, or_greater"));

	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger/max_queued_messages",
			PropertyInfo(Variant::INT, "network/limits/debugger/max_queued_messages", PROPERTY_HINT_RANGE, "0, 8192, 1, or_greater"));

	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger/max_errors_per_second",
			PropertyInfo(Variant::INT, "network/limits/debugger/max_errors_per_second", PROPERTY_HINT_RANGE, "0, 200, 1, or_greater"));

	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger/max_warnings_per_second",
			PropertyInfo(Variant::INT, "network/limits/debugger/max_warnings_per_second", PROPERTY_HINT_RANGE, "0, 200, 1, or_greater"));

	// Other
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/driver/driver_name",
			PropertyInfo(Variant::STRING, "rendering/driver/driver_name", PROPERTY_HINT_ENUM, "Vulkan"));

	ProjectSettings::get_singleton()->set_custom_property_info("input_devices/pen_tablet/driver.Windows",
			PropertyInfo(Variant::STRING, "input_devices/pen_tablet/driver.Windows", PROPERTY_HINT_ENUM, "wintab,winink"));

	ProjectSettings::get_singleton()->set_custom_property_info("physics/common/physics_fps",
			PropertyInfo(Variant::INT, "physics/common/physics_fps", PROPERTY_HINT_RANGE, "1,120,1,or_greater"));
#endif // TOOLS_ENABLED
}

void Main::initialize_core() {
	// Initialize OS
	{
		OS::get_singleton()->set_cmdline(configuration.exec_path, configuration.main_args);
		OS::get_singleton()->_use_vsync = GLOBAL_GET("display/window/vsync/use_vsync");
		OS::get_singleton()->_vsync_via_compositor = configuration.window_vsync_via_compositor;
		OS::get_singleton()->_render_thread_mode = OS::RenderThreadMode(configuration.render_thread_mode);
		OS::get_singleton()->_keep_screen_on = GLOBAL_GET("display/window/energy_saving/keep_screen_on");
		OS::get_singleton()->set_low_processor_usage_mode(GLOBAL_GET("application/run/low_processor_mode"));
		OS::get_singleton()->set_low_processor_usage_mode_sleep_usec(GLOBAL_GET("application/run/low_processor_mode_sleep_usec"));

		/* todo restore
		OS::get_singleton()->_allow_layered = GLOBAL_GET("display/window/per_pixel_transparency/allowed");
		video_mode.layered = GLOBAL_GET("display/window/per_pixel_transparency/enabled");
		*/

		if (configuration.disable_crash_handler) {
			OS::get_singleton()->disable_crash_handler();
		}
		if (configuration.debug_output) {
			OS::get_singleton()->_debug_stdout = true;
		}
		if (configuration.no_window_mode) {
			// Currently not used.
			OS::get_singleton()->set_no_window_mode(true);
		}
		if (configuration.output_verbosity == OutputVerbosity::VERBOSE) {
			OS::get_singleton()->_verbose_stdout = true;
		}
		if (configuration.allow_hidpi) {
			OS::get_singleton()->_allow_hidpi = GLOBAL_GET("display/window/dpi/allow_hidpi");
		}
		if (!OS::get_singleton()->_verbose_stdout) { // Not manually overridden.
			OS::get_singleton()->_verbose_stdout = GLOBAL_GET("debug/settings/stdout/verbose_stdout");
		}
#ifdef TOOLS_ENABLED
		if (is_tool_application) {
			// The editor and project manager always detect and use hiDPI if needed
			OS::get_singleton()->_allow_hidpi = true;
			OS::get_singleton()->_allow_layered = false;
		}
#endif // TOOLS_ENABLED

		OS::get_singleton()->ensure_user_data_dir();

		if (!is_tool_application && GLOBAL_GET("debug/file_logging/enable_file_logging") && FileAccess::get_create_func(FileAccess::ACCESS_USERDATA)) {
			String base_path = GLOBAL_GET("debug/file_logging/log_path");
			int max_files = GLOBAL_GET("debug/file_logging/max_log_files");
			OS::get_singleton()->add_logger(memnew(RotatedFileLogger(base_path, max_files)));
		}
	}

	// Initalize Engine
	{
		Engine::get_singleton()->set_iterations_per_second(GLOBAL_GET("physics/common/physics_fps"));
		Engine::get_singleton()->set_physics_jitter_fix(GLOBAL_GET("physics/common/physics_jitter_fix"));
		Engine::get_singleton()->set_target_fps(GLOBAL_GET("debug/settings/fps/force_fps"));
		Engine::get_singleton()->set_frame_delay(configuration.frame_delay);
		Engine::get_singleton()->abort_on_gpu_errors = configuration.abort_on_gpu_errors;
		Engine::get_singleton()->use_validation_layers = configuration.use_validation_layers;
		Engine::get_singleton()->set_time_scale(configuration.time_scale);
#ifdef TOOLS_ENABLED
		Engine::get_singleton()->set_editor_hint(is_tool_application);
#endif // TOOLS_ENABLED
	}

	if (is_tool_application || configuration.run_tool) {
		input_map->load_default();
	} else {
		input_map->load_from_project_settings();
	}

	translation_server = memnew(TranslationServer);
	performance = memnew(Performance);
	engine->add_singleton(Engine::Singleton("Performance", performance));

#ifdef MINIZIP_ENABLED
	//XXX: always get_singleton() == 0x0
	zip_packed_data = ZipArchive::get_singleton();
	//TODO: remove this temporary fix
	if (!zip_packed_data) {
		zip_packed_data = memnew(ZipArchive);
	}

	packed_data->add_pack_source(zip_packed_data);
#endif // MINIZIP_ENABLED
#ifdef TOOLS_ENABLED
	if (configuration.application_type == ApplicationType::EDITOR) {
		packed_data->set_disabled(true);
		project_settings->set_disable_feature_overrides(true);
	}
#endif // TOOLS_ENABLED

	EngineDebugger::initialize(configuration.debug_uri, configuration.skip_breakpoints, configuration.breakpoints);
	Logger::set_flush_stdout_on_print(GLOBAL_GET("application/run/flush_stdout_on_print"));

	message_queue = memnew(MessageQueue);
}

Error Main::finish_setup(Thread::ID p_main_tid_override) {
	preregister_module_types();
	preregister_server_types();

#if !defined(NO_THREADS)
	if (p_main_tid_override) {
		Thread::main_thread_id = p_main_tid_override;
	}
#endif

#ifdef TOOLS_ENABLED
	if (is_tool_application) {
		EditorNode::register_editor_paths(configuration.application_type == ApplicationType::PROJECT_MANAGER);
	}
#endif

	input = memnew(Input);
	OS::get_singleton()->initialize_joypads();

	Error error = initialize_servers();
	ERR_FAIL_COND_V(error != OK, error);

	register_core_types();
	register_core_singletons();
	register_core_driver_types();
	register_server_types();

	MAIN_PRINT("Main: Load Boot Image");
	load_boot_graphics();

	const String img_path = ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image");
	if (img_path != String()) {
		Ref<Texture2D> cursor = ResourceLoader::load(img_path);
		if (cursor.is_valid()) {
			Vector2 hotspot = ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image_hotspot");
			Input::get_singleton()->set_custom_mouse_cursor(cursor, Input::CURSOR_ARROW, hotspot);
		}
	}

	// Input
	{
		if (!is_tool_application && bool(GLOBAL_GET("input_devices/pointing/emulate_touch_from_mouse"))) {
			bool found_touchscreen = false;
			for (int i = 0; i < DisplayServer::get_singleton()->get_screen_count(); i++) {
				if (DisplayServer::get_singleton()->screen_is_touchscreen(i)) {
					found_touchscreen = true;
				}
			}
			if (!found_touchscreen) {
				//only if no touchscreen ui hint, set emulation
				Input::get_singleton()->set_emulate_touch_from_mouse(true);
			}
		}

		Input::get_singleton()->set_emulate_mouse_from_touch(bool(GLOBAL_GET("input_devices/pointing/emulate_mouse_from_touch")));
	}

	MAIN_PRINT("Main: Load Translations and Remaps");
	{
		translation_server->setup();
		if (configuration.locale != "" && translation_server->is_locale_valid(configuration.locale)) {
			translation_server->set_locale(configuration.locale);
		}
		translation_server->load_translations();

		ResourceLoader::load_translation_remaps(); //load remaps for resources
		ResourceLoader::load_path_remaps();
	}

	// Register types
	{
		MAIN_PRINT("Main: Load Scene Types");
		register_scene_types();

#ifdef TOOLS_ENABLED
		ClassDB::set_current_api(ClassDB::API_EDITOR);
		EditorNode::register_editor_types();
		ClassDB::set_current_api(ClassDB::API_CORE);
#endif

		MAIN_PRINT("Main: Load Modules, Physics, Drivers, Scripts");
		register_platform_apis();
		register_module_types();
	}

	if (is_tool_application) {
#ifdef TOOLS_ENABLED
		EditorSettings::create();

		// Load SSL Certificates from Editor Settings (or builtin)
		Crypto::load_default_certificates(EditorSettings::get_singleton()->get_setting("network/ssl/editor_ssl_certificates").operator String());

		if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CONSOLE_WINDOW)) {
			// Hide console window if requested (Windows-only).
			bool hide_console = EditorSettings::get_singleton()->get_setting("interface/editor/hide_console_window");
			DisplayServer::get_singleton()->console_set_visible(!hide_console);
		}
#endif
	} else {
		// Load SSL Certificates from Project Settings (or builtin).
		Crypto::load_default_certificates(GLOBAL_GET("network/ssl/certificate_bundle_override"));

		String app_name = TranslationServer::get_singleton()->translate(GLOBAL_GET("application/config/name"));
#ifdef DEBUG_ENABLED
		// Append a suffix to the window title to denote that the project is running
		// from a debug build (including the editor). Since this results in lower performance,
		// this should be clearly presented to the user.
		DisplayServer::get_singleton()->window_set_title(vformat("%s (DEBUG)", app_name));
#else
		DisplayServer::get_singleton()->window_set_title(app_name);
#endif
	}

	camera_server = CameraServer::create();

	// FIXME: Could maybe be moved to PhysicsServer3DManager and PhysicsServer2DManager directly
	// to have less code in main.cpp.
	{
		physics_server_3d = PhysicsServer3DManager::new_server(GLOBAL_GET("physics/3d/physics_engine"));
		if (!physics_server_3d) {
			physics_server_3d = PhysicsServer3DManager::new_default_server();
		}
		if (physics_server_3d) {
			physics_server_3d->init();
		}

		physics_server_2d = PhysicsServer2DManager::new_server(GLOBAL_GET("physics/3d/physics_engine"));
		if (!physics_server_2d) {
			physics_server_2d = PhysicsServer2DManager::new_default_server();
		}
		if (physics_server_2d) {
			physics_server_2d->init();
		}

		navigation_server = NavigationServer3DManager::new_default_server();
		navigation_2d_server = memnew(NavigationServer2D);
	}

	register_server_singletons();

	// This loads global classes, so it must happen before custom loaders and savers are registered
	ScriptServer::init_languages();

	ResourceLoader::add_custom_loaders();
	ResourceSaver::add_custom_savers();

	audio_server->load_default_bus_layout();

	if (configuration.enable_debug_profiler && EngineDebugger::is_active()) {
		// Start the "scripts" profiler, used in local debugging.
		// We could add more, and make the CLI arg require a comma-separated list of profilers.
		EngineDebugger::get_singleton()->profiler_enable("scripts", true);
	}

	if (configuration.application_type != ApplicationType::PROJECT_MANAGER) {
		// If not running the project manager, and now that the engine is
		// able to load resources, load the global shader variables.
		// If running on editor, don't load the textures because the editor
		// may want to import them first. Editor will reload those later.
		rendering_server->global_variables_load_settings(configuration.application_type != ApplicationType::EDITOR);
	}

	ClassDB::set_current_api(ClassDB::API_NONE); //no more APIs are registered at this point
	print_verbose("CORE API HASH: " + uitos(ClassDB::get_api_hash(ClassDB::API_CORE)));
	print_verbose("EDITOR API HASH: " + uitos(ClassDB::get_api_hash(ClassDB::API_EDITOR)));

	MAIN_PRINT("Main: Done");
	_setup_completed = true;

	return OK;
}

Error Main::initialize_servers() {
	// Audio
	{
		if (configuration.audio_driver_index < 0) {
			configuration.audio_driver_index = 0; // Default to first audio driver if not specified.
		}

		GLOBAL_DEF_RST_NOVAL("audio/driver/driver", AudioDriverManager::get_driver(configuration.audio_driver_index)->get_name());
		AudioDriverManager::initialize(configuration.audio_driver_index);
		audio_server = memnew(AudioServer);
		audio_server->init();
	}

	// Display
	{
		if (configuration.display_driver_index < 0) {
			String driver_name = GLOBAL_GET("rendering/driver/driver_name");
			const int display_driver_count = DisplayServer::get_create_function_count();

			for (int i = 0; i < display_driver_count; i++) {
				if (driver_name == DisplayServer::get_create_function_name(i)) {
					configuration.display_driver_index = i;
					break;
				}
			}
		}

		if (configuration.display_driver_index < 0) {
			configuration.display_driver_index = 0;
		}

		Error err;
		String rendering_driver; // temp broken
		display_server = DisplayServer::create(configuration.display_driver_index, rendering_driver, configuration.window_mode, configuration.window_flags, configuration.window_size, err);
		if (err != OK || display_server == nullptr) {
			//ok i guess we can't use this display server, try other ones
			const int display_driver_count = DisplayServer::get_create_function_count();

			for (int i = 0; i < display_driver_count; i++) {
				if (i == configuration.display_driver_index) {
					continue; //don't try the same twice
				}

				display_server = DisplayServer::create(i, rendering_driver, configuration.window_mode, configuration.window_flags, configuration.window_size, err);
				if (err == OK && display_server != nullptr) {
					break;
				}
			}
		}
		ERR_FAIL_COND_V_MSG((err != OK || display_server == nullptr), ERR_INVALID_PARAMETER, "Unable to create DisplayServer, all display drivers failed.");

		if (configuration.allow_focus_steal_pid > 0) {
			DisplayServer::get_singleton()->enable_for_stealing_focus(configuration.allow_focus_steal_pid);
		}
		if (display_server->has_feature(DisplayServer::FEATURE_ORIENTATION)) {
			display_server->screen_set_orientation(DisplayServer::ScreenOrientation(int(GLOBAL_GET("display/window/handheld/orientation"))));
		}
		if (configuration.window_position != Point2()) {
			display_server->window_set_position(configuration.window_position);
		}

		// TODO: Already in DisplayServer::create()?
		if (configuration.init_always_on_top) {
			DisplayServer::get_singleton()->window_set_flag(DisplayServer::WINDOW_FLAG_ALWAYS_ON_TOP, true);
		}

		// TODO: Already in DisplayServer::create()?
		if (configuration.window_mode != DisplayServer::WINDOW_MODE_WINDOWED) {
			DisplayServer::get_singleton()->window_set_mode(configuration.window_mode);
		}

		// Tablet driver
		{
			bool driver_found = false;
			if (configuration.tablet_driver_name != "") {
				const int tablet_driver_count = DisplayServer::get_singleton()->tablet_get_driver_count();
				for (int i = 0; i < tablet_driver_count; i++) {
					if (configuration.tablet_driver_name == DisplayServer::get_singleton()->tablet_get_driver_name(i)) {
						driver_found = true;
						break;
					}
				}
			}

			if (!driver_found) {
				configuration.tablet_driver_name = GLOBAL_GET("input_devices/pen_tablet/driver");
				if (configuration.tablet_driver_name == "") {
					configuration.tablet_driver_name = DisplayServer::get_singleton()->tablet_get_driver_name(0);
				}
			}

			print_verbose("Using \"" + configuration.tablet_driver_name + "\" pen tablet driver...");
			DisplayServer::get_singleton()->tablet_set_current_driver(configuration.tablet_driver_name);
		}
	}

	// Text
	{
		// Determine text driver.
		{
			if (configuration.text_driver_index < 0) {
				String text_driver_name = GLOBAL_GET("internationalization/rendering/text_driver");
				for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
					if (text_driver_name == TextServerManager::get_interface_name(i)) {
						configuration.text_driver_index = i;
						break;
					}
				}

				if (configuration.text_driver_index < 0) {
					/* If not selected, use one with the most features available. */
					int max_features = 0;
					for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
						uint32_t ftrs = TextServerManager::get_interface_features(i);
						int features = 0;
						while (ftrs) {
							features += ftrs & 1;
							ftrs >>= 1;
						}
						if (features >= max_features) {
							max_features = features;
							configuration.text_driver_index = i;
						}
					}
				}
			}
			print_verbose("Using \"" + TextServerManager::get_interface_name(configuration.text_driver_index) + "\" text server...");
		}

		text_server_manager = memnew(TextServerManager);
		Error err;
		TextServer *text_server = TextServerManager::initialize(configuration.text_driver_index, err);
		if (err != OK || text_server == nullptr) {
			for (int i = 0; i < TextServerManager::get_interface_count(); i++) {
				if (i == configuration.text_driver_index) {
					continue; //don't try the same twice
				}
				text_server = TextServerManager::initialize(i, err);
				if (err == OK && text_server != nullptr) {
					break;
				}
			}
		}

		if (err != OK || text_server == nullptr) {
			ERR_PRINT("Unable to create TextServer, all text drivers failed.");
			return err;
		}
	}

	// Rendering
	{
		rendering_server = memnew(RenderingServerDefault(OS::get_singleton()->get_render_thread_mode() == OS::RENDER_SEPARATE_THREAD));
		rendering_server->init();
		rendering_server->set_render_loop_enabled(!configuration.disable_render_loop);
		if (configuration.enable_gpu_profiler) {
			rendering_server->set_print_gpu_profile(true);
		}
	}

	xr_server = memnew(XRServer);

	return OK;
}

void Main::load_boot_graphics() {
#if !defined(JAVASCRIPT_ENABLED) && !defined(ANDROID_ENABLED)
	if (configuration.application_type != ApplicationType::PROJECT_MANAGER) {
		String boot_logo_path = GLOBAL_GET("application/boot_splash/image");
		bool boot_logo_scale = GLOBAL_GET("application/boot_splash/fullsize");
		bool boot_logo_filter = GLOBAL_GET("application/boot_splash/use_filter");
		const Color boot_bg_color = GLOBAL_GET("application/boot_splash/bg_color");

		boot_logo_path = boot_logo_path.strip_edges();
		Ref<Image> boot_logo;
		if (boot_logo_path != String()) {
			boot_logo.instance();
			Error load_err = ImageLoader::load_image(boot_logo_path, boot_logo);
			if (load_err) {
				ERR_PRINT("Non-existing or invalid boot splash at '" + boot_logo_path + "'. Loading default splash.");
			}
		}

		if (boot_logo.is_valid()) {
			RenderingServer::get_singleton()->set_boot_image(boot_logo, boot_bg_color, boot_logo_scale, boot_logo_filter);
		} else {
#ifndef NO_DEFAULT_BOOT_LOGO
			MAIN_PRINT("Main: Create bootsplash");
			MAIN_PRINT("Main: ClearColor");
			RenderingServer::get_singleton()->set_default_clear_color(boot_bg_color);

			MAIN_PRINT("Main: Image");
#if defined(TOOLS_ENABLED) && !defined(NO_EDITOR_SPLASH)
			const Ref<Image> splash = is_tool_application ? memnew(Image(boot_splash_editor_png)) : memnew(Image(boot_splash_png));
#else
			const Ref<Image> splash = memnew(Image(boot_splash_png));
#endif
			RenderingServer::get_singleton()->set_boot_image(splash, boot_bg_color, false);
#endif // NO_DEFAULT_BOOT_LOGO
		}
	}
#endif

	//Configure icon
	bool icon_configured = false;
	if (configuration.application_type == ApplicationType::PROJECT) {
		String icon_path;
#ifdef WINDOWS_ENABLED
		icon_path = GLOBAL_GET("application/config/windows_native_icon");
#elif OSX_ENABLED
		icon_path = GLOBAL_GET("application/config/macos_native_icon");
#endif
		if (icon_path != "") {
			DisplayServer::get_singleton()->set_native_icon(icon_path);
			icon_configured = true;
		} else {
			icon_path = GLOBAL_GET("application/config/icon");
			if (icon_path != "") {
				Ref<Image> icon;
				icon.instance();
				if (ImageLoader::load_image(icon_path, icon) == OK) {
					DisplayServer::get_singleton()->set_icon(icon);
					icon_configured = true;
				}
			}
		}
	}

	if (!icon_configured) {
		DisplayServer::get_singleton()->set_icon(memnew(Image(app_icon_png)));
	}
}

MainLoop *create_main_loop(const String &p_main_loop_type) {
	MainLoop *main_loop = nullptr;

	if (configuration.script_path != "") {
		// Load from script.
		Ref<Script> script_res = ResourceLoader::load(configuration.script_path);
		ERR_FAIL_COND_V_MSG(script_res.is_null(), main_loop, "Can't load script: " + configuration.script_path);
		ERR_FAIL_COND_V_MSG(!script_res->can_instance(), main_loop, "Can't instance script: " + configuration.script_path);

		Object *obj = ClassDB::instance(script_res->get_instance_base_type());
		main_loop = Object::cast_to<MainLoop>(obj);
		if (!main_loop) {
			if (obj) {
				memdelete(obj);
			}
			ERR_FAIL_V_MSG(main_loop, vformat("Can't load the script \"%s\" as it doesn't inherit from SceneTree or MainLoop.",
											  configuration.script_path));
		}

		main_loop->set_initialize_script(script_res);
	} else if (!ClassDB::class_exists(p_main_loop_type)) {
		if (!ScriptServer::is_global_class(p_main_loop_type)) {
			DisplayServer::get_singleton()->alert("Error: MainLoop type doesn't exist: " + p_main_loop_type);
			return main_loop;
		}

		// Load global class
		String script_path = ScriptServer::get_global_class_path(p_main_loop_type);
		StringName script_base = ScriptServer::get_global_class_native_base(p_main_loop_type);

		Ref<Script> script_res = ResourceLoader::load(script_path);
		Object *obj = ClassDB::instance(script_base);
		main_loop = Object::cast_to<MainLoop>(obj);
		if (!main_loop) {
			if (obj) {
				memdelete(obj);
			}
			ERR_FAIL_V_MSG(main_loop, vformat("The class %s does not inherit from SceneTree or MainLoop.", p_main_loop_type));
		}
		main_loop->set_initialize_script(script_res);
	}

	return main_loop;
}

SceneTree *create_scene_tree(MainLoop *p_main_loop) {
	SceneTree *scene_tree = Object::cast_to<SceneTree>(p_main_loop);

	if (is_tool_application) {
#ifdef TOOLS_ENABLED
		scene_tree->set_auto_accept_quit(GLOBAL_GET("application/config/auto_accept_quit"));
		scene_tree->set_quit_on_go_back(GLOBAL_GET("application/config/quit_on_go_back"));
#endif // TOOLS_ENABLED
	} else {
		String stretch_mode = GLOBAL_GET("display/window/stretch/mode");
		Window::ContentScaleMode scale_mode = Window::CONTENT_SCALE_MODE_DISABLED;
		if (stretch_mode == "canvas_items") {
			scale_mode = Window::CONTENT_SCALE_MODE_CANVAS_ITEMS;
		} else if (stretch_mode == "viewport") {
			scale_mode = Window::CONTENT_SCALE_MODE_VIEWPORT;
		}
		scene_tree->get_root()->set_content_scale_mode(scale_mode);

		String stretch_aspect = GLOBAL_GET("display/window/stretch/aspect");
		Window::ContentScaleAspect scale_aspect = Window::CONTENT_SCALE_ASPECT_IGNORE;
		if (stretch_aspect == "keep") {
			scale_aspect = Window::CONTENT_SCALE_ASPECT_KEEP;
		} else if (stretch_aspect == "keep_width") {
			scale_aspect = Window::CONTENT_SCALE_ASPECT_KEEP_WIDTH;
		} else if (stretch_aspect == "keep_height") {
			scale_aspect = Window::CONTENT_SCALE_ASPECT_KEEP_HEIGHT;
		} else if (stretch_aspect == "expand") {
			scale_aspect = Window::CONTENT_SCALE_ASPECT_EXPAND;
		}
		scene_tree->get_root()->set_content_scale_aspect(scale_aspect);

		const Size2i scale_size = Size2i(GLOBAL_GET("display/window/size/width"), GLOBAL_GET("display/window/size/height"));
		scene_tree->get_root()->set_content_scale_size(scale_size);

		scene_tree->set_auto_accept_quit(GLOBAL_GET("application/config/auto_accept_quit"));
		scene_tree->set_quit_on_go_back(GLOBAL_GET("application/config/quit_on_go_back"));

		scene_tree->get_root()->set_snap_controls_to_pixels(GLOBAL_GET("gui/common/snap_controls_to_pixels"));
		scene_tree->get_root()->set_use_font_oversampling(GLOBAL_GET("gui/fonts/dynamic_fonts/use_oversampling"));

		int texture_filter = GLOBAL_GET("rendering/textures/canvas_textures/default_texture_filter");
		int texture_repeat = GLOBAL_GET("rendering/textures/canvas_textures/default_texture_repeat");
		scene_tree->get_root()->set_default_canvas_item_texture_filter(
				Viewport::DefaultCanvasItemTextureFilter(texture_filter));
		scene_tree->get_root()->set_default_canvas_item_texture_repeat(
				Viewport::DefaultCanvasItemTextureRepeat(texture_repeat));

		// Add autoload nodes.
		{
			Map<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();

			//first pass, add the constants so they exist before any script is loaded
			for (Map<StringName, ProjectSettings::AutoloadInfo>::Element *E = autoloads.front(); E; E = E->next()) {
				const ProjectSettings::AutoloadInfo &info = E->get();

				if (info.is_singleton) {
					for (int i = 0; i < ScriptServer::get_language_count(); i++) {
						ScriptServer::get_language(i)->add_global_constant(info.name, Variant());
					}
				}
			}

			//second pass, load into global constants
			List<Node *> to_add;
			for (Map<StringName, ProjectSettings::AutoloadInfo>::Element *E = autoloads.front(); E; E = E->next()) {
				const ProjectSettings::AutoloadInfo &info = E->get();

				RES res = ResourceLoader::load(info.path);
				ERR_CONTINUE_MSG(res.is_null(), "Can't autoload: " + info.path);
				Node *n = nullptr;
				if (res->is_class("PackedScene")) {
					Ref<PackedScene> ps = res;
					n = ps->instance();
				} else if (res->is_class("Script")) {
					Ref<Script> script_res = res;
					StringName ibt = script_res->get_instance_base_type();
					bool valid_type = ClassDB::is_parent_class(ibt, "Node");
					ERR_CONTINUE_MSG(!valid_type, "Script does not inherit a Node: " + info.path);

					Object *obj = ClassDB::instance(ibt);

					ERR_CONTINUE_MSG(obj == nullptr,
							"Cannot instance script for autoload, expected 'Node' inheritance, got: " +
									String(ibt));

					n = Object::cast_to<Node>(obj);
					n->set_script(script_res);
				}

				ERR_CONTINUE_MSG(!n, "Path in autoload not a node or script: " + info.path);
				n->set_name(info.name);

				//defer so references are all valid on _ready()
				to_add.push_back(n);

				if (info.is_singleton) {
					for (int i = 0; i < ScriptServer::get_language_count(); i++) {
						ScriptServer::get_language(i)->add_global_constant(info.name, n);
					}
				}
			}

			for (List<Node *>::Element *E = to_add.front(); E; E = E->next()) {
				scene_tree->get_root()->add_child(E->get());
			}
		}
	}

#ifdef DEBUG_ENABLED
	scene_tree->set_debug_collisions_hint(configuration.enable_debug_collisions);
	scene_tree->set_debug_navigation_hint(configuration.enable_debug_navigation);
#endif

	bool embed_subwindows = false;
	if (configuration.application_type == ApplicationType::EDITOR) {
#ifdef TOOLS_ENABLED
		embed_subwindows = EditorSettings::get_singleton()->get_setting("interface/editor/single_window_mode");
#endif
	} else {
		embed_subwindows = configuration.single_window || GLOBAL_GET("display/window/subwindows/embed_subwindows");
	}

	if (embed_subwindows) {
		scene_tree->get_root()->set_embed_subwindows_hint(true);
	}

	return scene_tree;
}

String Main::localize_scene_path(const String &p_scene_path) {
	String path = p_scene_path.replace("\\", "/");

	if (!path.begins_with("res://")) {
		bool absolute =
				(path.size() > 1) && (path[0] == '/' || path[1] == ':');

		if (!absolute) {
			if (ProjectSettings::get_singleton()->is_using_datapack()) {
				path = "res://" + path;
			} else {
				int sep = path.rfind("/");

				if (sep == -1) {
					DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
					path = da->get_current_dir().plus_file(path);
					memdelete(da);
				} else {
					DirAccess *da = DirAccess::open(path.substr(0, sep));
					if (da) {
						path = da->get_current_dir().plus_file(
								path.substr(sep + 1, path.length()));
						memdelete(da);
					}
				}
			}
		}
	}

	return ProjectSettings::get_singleton()->localize_path(path);
}

#ifdef TESTS_ENABLED
// The order is the same as in `Main::setup()`, only core and some editor types
// are initialized here. This also combines `Main::complete_setup()` initialization.
Error Main::test_setup() {
	OS::get_singleton()->initialize();

	engine = memnew(Engine);

	register_core();
	register_core_types();
	register_core_driver_types();

	packed_data = memnew(PackedData);

	project_settings = memnew(ProjectSettings);

	GLOBAL_DEF("debug/settings/crash_handler/message",
			String("Please include this when reporting the bug on https://github.com/godotengine/godot/issues"));

	translation_server = memnew(TranslationServer);

	// From `Main::complete_setup()`.
	preregister_module_types();
	preregister_server_types();

	register_core_singletons();

	register_server_types();

	translation_server->setup(); //register translations, load them, etc.
	translation_server->load_translations();
	ResourceLoader::load_translation_remaps(); //load remaps for resources

	ResourceLoader::load_path_remaps();

	register_scene_types();

#ifdef TOOLS_ENABLED
	ClassDB::set_current_api(ClassDB::API_EDITOR);
	EditorNode::register_editor_types();

	ClassDB::set_current_api(ClassDB::API_CORE);
#endif
	register_platform_apis();

	register_module_types();

	ClassDB::set_current_api(ClassDB::API_NONE);

	return OK;
}

// The order is the same as in `Main::cleanup()`.
void Main::test_cleanup() {
	EngineDebugger::deinitialize();

	ResourceLoader::remove_custom_loaders();
	ResourceSaver::remove_custom_savers();

#ifdef TOOLS_ENABLED
	EditorNode::unregister_editor_types();
#endif
	unregister_module_types();
	unregister_platform_apis();
	unregister_scene_types();
	unregister_server_types();

	OS::get_singleton()->finalize();

	if (translation_server) {
		memdelete(translation_server);
	}
	if (project_settings) {
		memdelete(project_settings);
	}
	if (packed_data) {
		memdelete(packed_data);
	}
	if (engine) {
		memdelete(engine);
	}

	unregister_core_driver_types();
	unregister_core_types();
	unregister_core();

	OS::get_singleton()->finalize_core();
}
#endif // TESTS_ENABLED
