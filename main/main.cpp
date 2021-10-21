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

#include "core/config/project_settings.h"
#include "core/core_string_names.h"
#include "core/crypto/crypto.h"
#include "core/debugger/engine_debugger.h"
#include "core/extension/extension_api_dump.h"
#include "core/input/input.h"
#include "core/input/input_map.h"
#include "core/io/dir_access.h"
#include "core/io/file_access_network.h"
#include "core/io/file_access_pack.h"
#include "core/io/file_access_zip.h"
#include "core/io/image_loader.h"
#include "core/io/ip.h"
#include "core/io/resource_loader.h"
#include "core/object/message_queue.h"
#include "core/os/os.h"
#include "core/os/time.h"
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
#include "scene/main/scene_tree.h"
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

#ifdef TESTS_ENABLED
#include "tests/test_main.h"
#endif

#ifdef TOOLS_ENABLED

#include "editor/doc_data_class_path.gen.h"
#include "editor/doc_tools.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/progress_dialog.h"
#include "editor/project_manager.h"

#endif

/* Static members */

// Singletons

// Initialized in setup()
static Engine *engine = nullptr;
static ProjectSettings *globals = nullptr;
static Input *input = nullptr;
static InputMap *input_map = nullptr;
static TranslationServer *translation_server = nullptr;
static Performance *performance = nullptr;
static PackedData *packed_data = nullptr;
static Time *time_singleton = nullptr;
#ifdef MINIZIP_ENABLED
static ZipArchive *zip_packed_data = nullptr;
#endif
static FileAccessNetworkClient *file_access_network_client = nullptr;
static MessageQueue *message_queue = nullptr;

// Initialized in setup2()
static AudioServer *audio_server = nullptr;
static DisplayServer *display_server = nullptr;
static RenderingServer *rendering_server = nullptr;
static CameraServer *camera_server = nullptr;
static XRServer *xr_server = nullptr;
static TextServerManager *tsman = nullptr;
static PhysicsServer3D *physics_server = nullptr;
static PhysicsServer2D *physics_2d_server = nullptr;
static NavigationServer3D *navigation_server = nullptr;
static NavigationServer2D *navigation_2d_server = nullptr;
// We error out if setup2() doesn't turn this true
static bool _start_success = false;

// Drivers

String tablet_driver = "";
String text_driver = "";

static int text_driver_idx = -1;
static int display_driver_idx = -1;
static int audio_driver_idx = -1;

// Engine config/tools

static bool editor = false;
static bool project_manager = false;
static bool cmdline_tool = false;
static String locale;
static bool show_help = false;
static bool auto_quit = false;
static OS::ProcessID allow_focus_steal_pid = 0;
#ifdef TOOLS_ENABLED
static bool auto_build_solutions = false;
static String debug_server_uri;
#endif

// Display

static DisplayServer::WindowMode window_mode = DisplayServer::WINDOW_MODE_WINDOWED;
static DisplayServer::ScreenOrientation window_orientation = DisplayServer::SCREEN_LANDSCAPE;
static DisplayServer::VSyncMode window_vsync_mode = DisplayServer::VSYNC_ENABLED;
static uint32_t window_flags = 0;
static Size2i window_size = Size2i(1024, 600);

static int init_screen = -1;
static bool init_fullscreen = false;
static bool init_maximized = false;
static bool init_windowed = false;
static bool init_always_on_top = false;
static bool init_use_custom_pos = false;
static Vector2 init_custom_pos;
static bool force_lowdpi = false;

// Debug

static bool use_debug_profiler = false;
#ifdef DEBUG_ENABLED
static bool debug_collisions = false;
static bool debug_navigation = false;
#endif
static int frame_delay = 0;
static bool disable_render_loop = false;
static int fixed_fps = -1;
static bool print_fps = false;
#ifdef TOOLS_ENABLED
static bool dump_extension_api = false;
#endif
bool profile_gpu = false;

/* Helper methods */

// Used by Mono module, should likely be registered in Engine singleton instead
// FIXME: This is also not 100% accurate, `project_manager` is only true when it was requested,
// but not if e.g. we fail to load and project and fallback to the manager.
bool Main::is_project_manager() {
	return project_manager;
}

bool Main::is_cmdline_tool() {
	return cmdline_tool;
}

static String unescape_cmdline(const String &p_str) {
	return p_str.replace("%20", " ");
}

static String get_full_version_string() {
	String hash = String(VERSION_HASH);
	if (hash.length() != 0) {
		hash = "." + hash.left(9);
	}
	return String(VERSION_FULL_BUILD) + hash;
}

// FIXME: Could maybe be moved to PhysicsServer3DManager and PhysicsServer2DManager directly
// to have less code in main.cpp.
void initialize_physics() {
	/// 3D Physics Server
	physics_server = PhysicsServer3DManager::new_server(
			ProjectSettings::get_singleton()->get(PhysicsServer3DManager::setting_property_name));
	if (!physics_server) {
		// Physics server not found, Use the default physics
		physics_server = PhysicsServer3DManager::new_default_server();
	}
	ERR_FAIL_COND(!physics_server);
	physics_server->init();

	/// 2D Physics server
	physics_2d_server = PhysicsServer2DManager::new_server(
			ProjectSettings::get_singleton()->get(PhysicsServer2DManager::setting_property_name));
	if (!physics_2d_server) {
		// Physics server not found, Use the default physics
		physics_2d_server = PhysicsServer2DManager::new_default_server();
	}
	ERR_FAIL_COND(!physics_2d_server);
	physics_2d_server->init();
}

void finalize_physics() {
	physics_server->finish();
	memdelete(physics_server);

	physics_2d_server->finish();
	memdelete(physics_2d_server);
}

void finalize_display() {
	rendering_server->finish();
	memdelete(rendering_server);

	memdelete(display_server);
}

void initialize_navigation_server() {
	ERR_FAIL_COND(navigation_server != nullptr);

	navigation_server = NavigationServer3DManager::new_default_server();
	navigation_2d_server = memnew(NavigationServer2D);
}

void finalize_navigation_server() {
	memdelete(navigation_server);
	navigation_server = nullptr;

	memdelete(navigation_2d_server);
	navigation_2d_server = nullptr;
}

//#define DEBUG_INIT
#ifdef DEBUG_INIT
#define MAIN_PRINT(m_txt) print_line(m_txt)
#else
#define MAIN_PRINT(m_txt)
#endif

void Main::print_help(const char *p_binary) {
	print_line(String(VERSION_NAME) + " v" + get_full_version_string() + " - " + String(VERSION_WEBSITE));
	OS::get_singleton()->print("Free and open source software under the terms of the MIT license.\n");
	OS::get_singleton()->print("(c) 2007-2021 Juan Linietsky, Ariel Manzur.\n");
	OS::get_singleton()->print("(c) 2014-2021 Godot Engine contributors.\n");
	OS::get_singleton()->print("\n");
	OS::get_singleton()->print("Usage: %s [options] [path to scene or 'project.godot' file]\n", p_binary);
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("General options:\n");
	OS::get_singleton()->print("  -h, --help                                   Display this help message.\n");
	OS::get_singleton()->print("  --version                                    Display the version string.\n");
	OS::get_singleton()->print("  -v, --verbose                                Use verbose stdout mode.\n");
	OS::get_singleton()->print("  --quiet                                      Quiet mode, silences stdout messages. Errors are still displayed.\n");
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("Run options:\n");
#ifdef TOOLS_ENABLED
	OS::get_singleton()->print("  -e, --editor                                 Start the editor instead of running the scene.\n");
	OS::get_singleton()->print("  -p, --project-manager                        Start the project manager, even if a project is auto-detected.\n");
	OS::get_singleton()->print("  --debug-server <uri>                         Start the editor debug server (<protocol>://<host/IP>[:<port>], e.g. tcp://127.0.0.1:6007)\n");
#endif
	OS::get_singleton()->print("  -q, --quit                                   Quit after the first iteration.\n");
	OS::get_singleton()->print("  -l, --language <locale>                      Use a specific locale (<locale> being a two-letter code).\n");
	OS::get_singleton()->print("  --path <directory>                           Path to a project (<directory> must contain a 'project.godot' file).\n");
	OS::get_singleton()->print("  -u, --upwards                                Scan folders upwards for project.godot file.\n");
	OS::get_singleton()->print("  --main-pack <file>                           Path to a pack (.pck) file to load.\n");
	OS::get_singleton()->print("  --render-thread <mode>                       Render thread mode ('unsafe', 'safe', 'separate').\n");
	OS::get_singleton()->print("  --remote-fs <address>                        Remote filesystem (<host/IP>[:<port>] address).\n");
	OS::get_singleton()->print("  --remote-fs-password <password>              Password for remote filesystem.\n");

	OS::get_singleton()->print("  --audio-driver <driver>                      Audio driver [");
	for (int i = 0; i < AudioDriverManager::get_driver_count(); i++) {
		if (i > 0) {
			OS::get_singleton()->print(", ");
		}
		OS::get_singleton()->print("'%s'", AudioDriverManager::get_driver(i)->get_name());
	}
	OS::get_singleton()->print("].\n");

	OS::get_singleton()->print("  --display-driver <driver>                    Display driver (and rendering driver) [");
	for (int i = 0; i < DisplayServer::get_create_function_count(); i++) {
		if (i > 0) {
			OS::get_singleton()->print(", ");
		}
		OS::get_singleton()->print("'%s' (", DisplayServer::get_create_function_name(i));
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

	OS::get_singleton()->print("  --rendering-driver <driver>                  Rendering driver (depends on display driver).\n");

	OS::get_singleton()->print("  --text-driver <driver>                       Text driver (Fonts, BiDi, shaping)\n");

	OS::get_singleton()->print("  --headless                                   Enable headless mode (--display-driver headless --audio-driver Dummy). Useful for servers and with --script.\n");

	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("Display options:\n");
	OS::get_singleton()->print("  -f, --fullscreen                             Request fullscreen mode.\n");
	OS::get_singleton()->print("  -m, --maximized                              Request a maximized window.\n");
	OS::get_singleton()->print("  -w, --windowed                               Request windowed mode.\n");
	OS::get_singleton()->print("  -t, --always-on-top                          Request an always-on-top window.\n");
	OS::get_singleton()->print("  --resolution <W>x<H>                         Request window resolution.\n");
	OS::get_singleton()->print("  --position <X>,<Y>                           Request window position.\n");
	OS::get_singleton()->print("  --low-dpi                                    Force low-DPI mode (macOS and Windows only).\n");
	OS::get_singleton()->print("  --single-window                              Use a single window (no separate subwindows).\n");
	OS::get_singleton()->print("  --tablet-driver                              Pen tablet input driver.\n");
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("Debug options:\n");
	OS::get_singleton()->print("  -d, --debug                                  Debug (local stdout debugger).\n");
	OS::get_singleton()->print("  -b, --breakpoints                            Breakpoint list as source::line comma-separated pairs, no spaces (use %%20 instead).\n");
	OS::get_singleton()->print("  --profiling                                  Enable profiling in the script debugger.\n");
	OS::get_singleton()->print("  --vk-layers                                  Enable Vulkan Validation layers for debugging.\n");
#if DEBUG_ENABLED
	OS::get_singleton()->print("  --gpu-abort                                  Abort on GPU errors (usually validation layer errors), may help see the problem if your system freezes.\n");
#endif
	OS::get_singleton()->print("  --remote-debug <uri>                         Remote debug (<protocol>://<host/IP>[:<port>], e.g. tcp://127.0.0.1:6007).\n");
#if defined(DEBUG_ENABLED)
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
	OS::get_singleton()->print("  -s, --script <script>                        Run a script.\n");
	OS::get_singleton()->print("  --check-only                                 Only parse for errors and quit (use with --script).\n");
#ifdef TOOLS_ENABLED
	OS::get_singleton()->print("  --export <preset> <path>                     Export the project using the given preset and matching release template. The preset name should match one defined in export_presets.cfg.\n");
	OS::get_singleton()->print("                                               <path> should be absolute or relative to the project directory, and include the filename for the binary (e.g. 'builds/game.exe'). The target directory should exist.\n");
	OS::get_singleton()->print("  --export-debug <preset> <path>               Same as --export, but using the debug template.\n");
	OS::get_singleton()->print("  --export-pack <preset> <path>                Same as --export, but only export the game pack for the given preset. The <path> extension determines whether it will be in PCK or ZIP format.\n");
	OS::get_singleton()->print("  --doctool [<path>]                           Dump the engine API reference to the given <path> (defaults to current dir) in XML format, merging if existing files are found.\n");
	OS::get_singleton()->print("  --no-docbase                                 Disallow dumping the base types (used with --doctool).\n");
	OS::get_singleton()->print("  --build-solutions                            Build the scripting solutions (e.g. for C# projects). Implies --editor and requires a valid project to edit.\n");
	OS::get_singleton()->print("  --dump-extension-api                         Generate JSON dump of the Godot API for GDExtension bindings named 'extension_api.json' in the current folder.\n");
#ifdef DEBUG_METHODS_ENABLED
	// TODO: Should be removed together with nativescript eventually.
	OS::get_singleton()->print("  --gdnative-generate-json-api <path>          Generate JSON dump of the Godot API for GDNative bindings and save it on the file specified in <path>.\n");
	OS::get_singleton()->print("  --gdnative-generate-json-builtin-api <path>  Generate JSON dump of the Godot API of the builtin Variant types and utility functions for GDNative bindings and save it on the file specified in <path>.\n");
#endif
#ifdef TESTS_ENABLED
	OS::get_singleton()->print("  --test [--help]                              Run unit tests. Use --test --help for more information.\n");
#endif
#endif
	OS::get_singleton()->print("\n");
}

#ifdef TESTS_ENABLED
// The order is the same as in `Main::setup()`, only core and some editor types
// are initialized here. This also combines `Main::setup2()` initialization.
Error Main::test_setup() {
	OS::get_singleton()->initialize();

	engine = memnew(Engine);

	register_core_types();
	register_core_driver_types();

	packed_data = memnew(PackedData);

	globals = memnew(ProjectSettings);

	GLOBAL_DEF("debug/settings/crash_handler/message",
			String("Please include this when reporting the bug on https://github.com/godotengine/godot/issues"));
	GLOBAL_DEF_RST("rendering/occlusion_culling/bvh_build_quality", 2);

	translation_server = memnew(TranslationServer);
	tsman = memnew(TextServerManager);

	register_core_extensions();

	// From `Main::setup2()`.
	preregister_module_types();
	preregister_server_types();

	register_core_singletons();

	register_server_types();

	translation_server->setup(); //register translations, load them, etc.
	if (locale != "") {
		translation_server->set_locale(locale);
	}
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
	register_driver_types();

	ERR_FAIL_COND_V(TextServerManager::get_singleton()->get_interface_count() == 0, ERR_CANT_CREATE);
	TextServerManager::get_singleton()->set_primary_interface(TextServerManager::get_singleton()->get_interface(0));

	ClassDB::set_current_api(ClassDB::API_NONE);

	_start_success = true;

	return OK;
}
// The order is the same as in `Main::cleanup()`.
void Main::test_cleanup() {
	ERR_FAIL_COND(!_start_success);

	EngineDebugger::deinitialize();

	ResourceLoader::remove_custom_loaders();
	ResourceSaver::remove_custom_savers();

	unregister_driver_types();
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
	if (tsman) {
		memdelete(tsman);
	}
	if (globals) {
		memdelete(globals);
	}
	if (packed_data) {
		memdelete(packed_data);
	}
	if (engine) {
		memdelete(engine);
	}

	unregister_core_driver_types();
	unregister_core_types();

	OS::get_singleton()->finalize_core();
}
#endif

int Main::test_entrypoint(int argc, char *argv[], bool &tests_need_run) {
#ifdef TESTS_ENABLED
	for (int x = 0; x < argc; x++) {
		if ((strncmp(argv[x], "--test", 6) == 0) && (strlen(argv[x]) == 6)) {
			tests_need_run = true;
			// TODO: need to come up with different test contexts.
			// Not every test requires high-level functionality like `ClassDB`.
			test_setup();
			int status = test_main(argc, argv);
			test_cleanup();
			return status;
		}
	}
#endif
	tests_need_run = false;
	return 0;
}

/* Engine initialization
 *
 * Consists of several methods that are called by each platform's specific main(argc, argv).
 * To fully understand engine init, one should therefore start from the platform's main and
 * see how it calls into the Main class' methods.
 *
 * The initialization is typically done in 3 steps (with the setup2 step triggered either
 * automatically by setup, or manually in the platform's main).
 *
 * - setup(execpath, argc, argv, p_second_phase) is the main entry point for all platforms,
 *   responsible for the initialization of all low level singletons and core types, and parsing
 *   command line arguments to configure things accordingly.
 *   If p_second_phase is true, it will chain into setup2() (default behaviour). This is
 *   disabled on some platforms (Android, iOS, UWP) which trigger the second step in their
 *   own time.
 *
 * - setup2(p_main_tid_override) registers high level servers and singletons, displays the
 *   boot splash, then registers higher level types (scene, editor, etc.).
 *
 * - start() is the last step and that's where command line tools can run, or the main loop
 *   can be created eventually and the project settings put into action. That's also where
 *   the editor node is created, if relevant.
 *   start() does it own argument parsing for a subset of the command line arguments described
 *   in help, it's a bit messy and should be globalized with the setup() parsing somehow.
 */

Error Main::setup(const char *execpath, int argc, char *argv[], bool p_second_phase) {
	OS::get_singleton()->initialize();

	engine = memnew(Engine);

	MAIN_PRINT("Main: Initialize CORE");

	register_core_types();
	register_core_driver_types();

	MAIN_PRINT("Main: Initialize Globals");

	input_map = memnew(InputMap);
	time_singleton = memnew(Time);
	globals = memnew(ProjectSettings);

	register_core_settings(); //here globals are present

	translation_server = memnew(TranslationServer);
	performance = memnew(Performance);
	GDREGISTER_CLASS(Performance);
	engine->add_singleton(Engine::Singleton("Performance", performance));

	// Only flush stdout in debug builds by default, as spamming `print()` will
	// decrease performance if this is enabled.
	GLOBAL_DEF_RST("application/run/flush_stdout_on_print", false);
	GLOBAL_DEF_RST("application/run/flush_stdout_on_print.debug", true);

	GLOBAL_DEF("debug/settings/crash_handler/message",
			String("Please include this when reporting the bug on https://github.com/godotengine/godot/issues"));

	MAIN_PRINT("Main: Parse CMDLine");

	/* argument parsing and main creation */
	List<String> args;
	List<String> main_args;

	for (int i = 0; i < argc; i++) {
		args.push_back(String::utf8(argv[i]));
	}

	List<String>::Element *I = args.front();

	while (I) {
		I->get() = unescape_cmdline(I->get().strip_edges());
		I = I->next();
	}

	I = args.front();

	String display_driver = "";
	String audio_driver = "";
	String project_path = ".";
	bool upwards = false;
	String debug_uri = "";
	bool skip_breakpoints = false;
	String main_pack;
	bool quiet_stdout = false;
	int rtm = -1;

	String remotefs;
	String remotefs_pass;

	Vector<String> breakpoints;
	bool use_custom_res = true;
	bool force_res = false;
#ifdef TOOLS_ENABLED
	bool found_project = false;
#endif

	packed_data = PackedData::get_singleton();
	if (!packed_data) {
		packed_data = memnew(PackedData);
	}

#ifdef MINIZIP_ENABLED

	//XXX: always get_singleton() == 0x0
	zip_packed_data = ZipArchive::get_singleton();
	//TODO: remove this temporary fix
	if (!zip_packed_data) {
		zip_packed_data = memnew(ZipArchive);
	}

	packed_data->add_pack_source(zip_packed_data);
#endif

	I = args.front();
	while (I) {
#ifdef OSX_ENABLED
		// Ignore the process serial number argument passed by macOS Gatekeeper.
		// Otherwise, Godot would try to open a non-existent project on the first start and abort.
		if (I->get().begins_with("-psn_")) {
			I = I->next();
			continue;
		}
#endif

		List<String>::Element *N = I->next();

		if (I->get() == "-h" || I->get() == "--help" || I->get() == "/?") { // display help

			show_help = true;
			goto error;

		} else if (I->get() == "--version") {
			print_line(get_full_version_string());
			goto error;

		} else if (I->get() == "-v" || I->get() == "--verbose") { // verbose output

			OS::get_singleton()->_verbose_stdout = true;
		} else if (I->get() == "--quiet") { // quieter output

			quiet_stdout = true;

		} else if (I->get() == "--audio-driver") { // audio driver

			if (I->next()) {
				audio_driver = I->next()->get();

				bool found = false;
				for (int i = 0; i < AudioDriverManager::get_driver_count(); i++) {
					if (audio_driver == AudioDriverManager::get_driver(i)->get_name()) {
						found = true;
					}
				}

				if (!found) {
					OS::get_singleton()->print("Unknown audio driver '%s', aborting.\nValid options are ",
							audio_driver.utf8().get_data());

					for (int i = 0; i < AudioDriverManager::get_driver_count(); i++) {
						if (i == AudioDriverManager::get_driver_count() - 1) {
							OS::get_singleton()->print(" and ");
						} else if (i != 0) {
							OS::get_singleton()->print(", ");
						}

						OS::get_singleton()->print("'%s'", AudioDriverManager::get_driver(i)->get_name());
					}

					OS::get_singleton()->print(".\n");

					goto error;
				}

				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing audio driver argument, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--text-driver") {
			if (I->next()) {
				text_driver = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing text driver argument, aborting.\n");
				goto error;
			}

		} else if (I->get() == "--display-driver") { // force video driver

			if (I->next()) {
				display_driver = I->next()->get();

				bool found = false;
				for (int i = 0; i < DisplayServer::get_create_function_count(); i++) {
					if (display_driver == DisplayServer::get_create_function_name(i)) {
						found = true;
					}
				}

				if (!found) {
					OS::get_singleton()->print("Unknown display driver '%s', aborting.\nValid options are ",
							display_driver.utf8().get_data());

					for (int i = 0; i < DisplayServer::get_create_function_count(); i++) {
						if (i == DisplayServer::get_create_function_count() - 1) {
							OS::get_singleton()->print(" and ");
						} else if (i != 0) {
							OS::get_singleton()->print(", ");
						}

						OS::get_singleton()->print("'%s'", DisplayServer::get_create_function_name(i));
					}

					OS::get_singleton()->print(".\n");

					goto error;
				}

				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing video driver argument, aborting.\n");
				goto error;
			}
		} else if (I->get() == "-f" || I->get() == "--fullscreen") { // force fullscreen

			init_fullscreen = true;
		} else if (I->get() == "-m" || I->get() == "--maximized") { // force maximized window

			init_maximized = true;
			window_mode = DisplayServer::WINDOW_MODE_MAXIMIZED;

		} else if (I->get() == "-w" || I->get() == "--windowed") { // force windowed window

			init_windowed = true;
		} else if (I->get() == "--vk-layers") {
			Engine::singleton->use_validation_layers = true;
#ifdef DEBUG_ENABLED
		} else if (I->get() == "--gpu-abort") {
			Engine::singleton->abort_on_gpu_errors = true;
#endif
		} else if (I->get() == "--tablet-driver") {
			if (I->next()) {
				tablet_driver = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing tablet driver argument, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--single-window") { // force single window

			OS::get_singleton()->_single_window = true;
		} else if (I->get() == "-t" || I->get() == "--always-on-top") { // force always-on-top window

			init_always_on_top = true;
		} else if (I->get() == "--resolution") { // force resolution

			if (I->next()) {
				String vm = I->next()->get();

				if (vm.find("x") == -1) { // invalid parameter format

					OS::get_singleton()->print("Invalid resolution '%s', it should be e.g. '1280x720'.\n",
							vm.utf8().get_data());
					goto error;
				}

				int w = vm.get_slice("x", 0).to_int();
				int h = vm.get_slice("x", 1).to_int();

				if (w <= 0 || h <= 0) {
					OS::get_singleton()->print("Invalid resolution '%s', width and height must be above 0.\n",
							vm.utf8().get_data());
					goto error;
				}

				window_size.width = w;
				window_size.height = h;
				force_res = true;

				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing resolution argument, aborting.\n");
				goto error;
			}

		} else if (I->get() == "--position") { // set window position

			if (I->next()) {
				String vm = I->next()->get();

				if (vm.find(",") == -1) { // invalid parameter format

					OS::get_singleton()->print("Invalid position '%s', it should be e.g. '80,128'.\n",
							vm.utf8().get_data());
					goto error;
				}

				int x = vm.get_slice(",", 0).to_int();
				int y = vm.get_slice(",", 1).to_int();

				init_custom_pos = Point2(x, y);
				init_use_custom_pos = true;

				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing position argument, aborting.\n");
				goto error;
			}

		} else if (I->get() == "--low-dpi") { // force low DPI (macOS only)

			force_lowdpi = true;
		} else if (I->get() == "--headless") { // enable headless mode (no audio, no rendering).

			audio_driver = "Dummy";
			display_driver = "headless";

		} else if (I->get() == "--profiling") { // enable profiling

			use_debug_profiler = true;

		} else if (I->get() == "-l" || I->get() == "--language") { // language

			if (I->next()) {
				locale = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing language argument, aborting.\n");
				goto error;
			}

		} else if (I->get() == "--remote-fs") { // remote filesystem

			if (I->next()) {
				remotefs = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing remote filesystem address, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--remote-fs-password") { // remote filesystem password

			if (I->next()) {
				remotefs_pass = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing remote filesystem password, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--render-thread") { // render thread mode

			if (I->next()) {
				if (I->next()->get() == "safe") {
					rtm = OS::RENDER_THREAD_SAFE;
				} else if (I->next()->get() == "unsafe") {
					rtm = OS::RENDER_THREAD_UNSAFE;
				} else if (I->next()->get() == "separate") {
					rtm = OS::RENDER_SEPARATE_THREAD;
				}

				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing render thread mode argument, aborting.\n");
				goto error;
			}
#ifdef TOOLS_ENABLED
		} else if (I->get() == "-e" || I->get() == "--editor") { // starts editor

			editor = true;
		} else if (I->get() == "-p" || I->get() == "--project-manager") { // starts project manager

			project_manager = true;
		} else if (I->get() == "--debug-server") {
			if (I->next()) {
				debug_server_uri = I->next()->get();
				if (debug_server_uri.find("://") == -1) { // wrong address
					OS::get_singleton()->print("Invalid debug server uri. It should be of the form <protocol>://<bind_address>:<port>.\n");
					goto error;
				}
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing remote debug server uri, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--build-solutions") { // Build the scripting solution such C#

			auto_build_solutions = true;
			editor = true;
			cmdline_tool = true;
#ifdef DEBUG_METHODS_ENABLED
		} else if (I->get() == "--gdnative-generate-json-api" || I->get() == "--gdnative-generate-json-builtin-api") {
			// Register as an editor instance to use low-end fallback if relevant.
			editor = true;
			cmdline_tool = true;
			// We still pass it to the main arguments since the argument handling itself is not done in this function,
			// it's done in nativescript init code.
			main_args.push_back(I->get());
#endif
		} else if (I->get() == "--dump-extension-api") {
			// Register as an editor instance to use low-end fallback if relevant.
			editor = true;
			cmdline_tool = true;
			dump_extension_api = true;
			print_line("Dumping Extension API");
			// Hack. Not needed but otherwise we end up detecting that this should
			// run the project instead of a cmdline tool.
			// Needs full refactoring to fix properly.
			main_args.push_back(I->get());
		} else if (I->get() == "--export" || I->get() == "--export-debug" ||
				   I->get() == "--export-pack") { // Export project
			// Actually handling is done in start().
			editor = true;
			cmdline_tool = true;
			main_args.push_back(I->get());
		} else if (I->get() == "--doctool") {
			// Actually handling is done in start().
			cmdline_tool = true;
			main_args.push_back(I->get());
#endif
		} else if (I->get() == "--path") { // set path of project to start or edit

			if (I->next()) {
				String p = I->next()->get();
				if (OS::get_singleton()->set_cwd(p) == OK) {
					//nothing
				} else {
					project_path = I->next()->get(); //use project_path instead
				}
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing relative or absolute path, aborting.\n");
				goto error;
			}
		} else if (I->get() == "-u" || I->get() == "--upwards") { // scan folders upwards
			upwards = true;
		} else if (I->get() == "-q" || I->get() == "--quit") { // Auto quit at the end of the first main loop iteration
			auto_quit = true;
		} else if (I->get().ends_with("project.godot")) {
			String path;
			String file = I->get();
			int sep = MAX(file.rfind("/"), file.rfind("\\"));
			if (sep == -1) {
				path = ".";
			} else {
				path = file.substr(0, sep);
			}
			if (OS::get_singleton()->set_cwd(path) == OK) {
				// path already specified, don't override
			} else {
				project_path = path;
			}
#ifdef TOOLS_ENABLED
			editor = true;
#endif
		} else if (I->get() == "-b" || I->get() == "--breakpoints") { // add breakpoints

			if (I->next()) {
				String bplist = I->next()->get();
				breakpoints = bplist.split(",");
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing list of breakpoints, aborting.\n");
				goto error;
			}

		} else if (I->get() == "--frame-delay") { // force frame delay

			if (I->next()) {
				frame_delay = I->next()->get().to_int();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing frame delay argument, aborting.\n");
				goto error;
			}

		} else if (I->get() == "--time-scale") { // force time scale

			if (I->next()) {
				Engine::get_singleton()->set_time_scale(I->next()->get().to_float());
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing time scale argument, aborting.\n");
				goto error;
			}

		} else if (I->get() == "--main-pack") {
			if (I->next()) {
				main_pack = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing path to main pack file, aborting.\n");
				goto error;
			};

		} else if (I->get() == "-d" || I->get() == "--debug") {
			debug_uri = "local://";
			OS::get_singleton()->_debug_stdout = true;
#if defined(DEBUG_ENABLED)
		} else if (I->get() == "--debug-collisions") {
			debug_collisions = true;
		} else if (I->get() == "--debug-navigation") {
			debug_navigation = true;
		} else if (I->get() == "--debug-stringnames") {
			StringName::set_debug_stringnames(true);
#endif
		} else if (I->get() == "--remote-debug") {
			if (I->next()) {
				debug_uri = I->next()->get();
				if (debug_uri.find("://") == -1) { // wrong address
					OS::get_singleton()->print(
							"Invalid debug host address, it should be of the form <protocol>://<host/IP>:<port>.\n");
					goto error;
				}
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing remote debug host address, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--allow_focus_steal_pid") { // not exposed to user
			if (I->next()) {
				allow_focus_steal_pid = I->next()->get().to_int();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing editor PID argument, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--disable-render-loop") {
			disable_render_loop = true;
		} else if (I->get() == "--fixed-fps") {
			if (I->next()) {
				fixed_fps = I->next()->get().to_int();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing fixed-fps argument, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--print-fps") {
			print_fps = true;
		} else if (I->get() == "--profile-gpu") {
			profile_gpu = true;
		} else if (I->get() == "--disable-crash-handler") {
			OS::get_singleton()->disable_crash_handler();
		} else if (I->get() == "--skip-breakpoints") {
			skip_breakpoints = true;
		} else {
			main_args.push_back(I->get());
		}

		I = N;
	}

#ifdef TOOLS_ENABLED
	if (editor && project_manager) {
		OS::get_singleton()->print(
				"Error: Command line arguments implied opening both editor and project manager, which is not possible. Aborting.\n");
		goto error;
	}
#endif

	// Network file system needs to be configured before globals, since globals are based on the
	// 'project.godot' file which will only be available through the network if this is enabled
	FileAccessNetwork::configure();
	if (remotefs != "") {
		file_access_network_client = memnew(FileAccessNetworkClient);
		int port;
		if (remotefs.find(":") != -1) {
			port = remotefs.get_slicec(':', 1).to_int();
			remotefs = remotefs.get_slicec(':', 0);
		} else {
			port = 6010;
		}

		Error err = file_access_network_client->connect(remotefs, port, remotefs_pass);
		if (err) {
			OS::get_singleton()->printerr("Could not connect to remotefs: %s:%i.\n", remotefs.utf8().get_data(), port);
			goto error;
		}

		FileAccess::make_default<FileAccessNetwork>(FileAccess::ACCESS_RESOURCES);
	}

	if (globals->setup(project_path, main_pack, upwards) == OK) {
#ifdef TOOLS_ENABLED
		found_project = true;
#endif
	} else {
#ifdef TOOLS_ENABLED
		editor = false;
#else
		const String error_msg = "Error: Couldn't load project data at path \"" + project_path + "\". Is the .pck file missing?\nIf you've renamed the executable, the associated .pck file should also be renamed to match the executable's name (without the extension).\n";
		OS::get_singleton()->print("%s", error_msg.utf8().get_data());
		OS::get_singleton()->alert(error_msg);

		goto error;
#endif
	}

	// Initialize user data dir.
	OS::get_singleton()->ensure_user_data_dir();

	ResourceUID::get_singleton()->load_from_cache(); // load UUIDs from cache.

	GLOBAL_DEF("memory/limits/multithreaded_server/rid_pool_prealloc", 60);
	ProjectSettings::get_singleton()->set_custom_property_info("memory/limits/multithreaded_server/rid_pool_prealloc",
			PropertyInfo(Variant::INT,
					"memory/limits/multithreaded_server/rid_pool_prealloc",
					PROPERTY_HINT_RANGE,
					"0,500,1")); // No negative and limit to 500 due to crashes
	GLOBAL_DEF("network/limits/debugger/max_chars_per_second", 32768);
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger/max_chars_per_second",
			PropertyInfo(Variant::INT,
					"network/limits/debugger/max_chars_per_second",
					PROPERTY_HINT_RANGE,
					"0, 4096, 1, or_greater"));
	GLOBAL_DEF("network/limits/debugger/max_queued_messages", 2048);
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger/max_queued_messages",
			PropertyInfo(Variant::INT,
					"network/limits/debugger/max_queued_messages",
					PROPERTY_HINT_RANGE,
					"0, 8192, 1, or_greater"));
	GLOBAL_DEF("network/limits/debugger/max_errors_per_second", 400);
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger/max_errors_per_second",
			PropertyInfo(Variant::INT,
					"network/limits/debugger/max_errors_per_second",
					PROPERTY_HINT_RANGE,
					"0, 200, 1, or_greater"));
	GLOBAL_DEF("network/limits/debugger/max_warnings_per_second", 400);
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger/max_warnings_per_second",
			PropertyInfo(Variant::INT,
					"network/limits/debugger/max_warnings_per_second",
					PROPERTY_HINT_RANGE,
					"0, 200, 1, or_greater"));

	EngineDebugger::initialize(debug_uri, skip_breakpoints, breakpoints);

#ifdef TOOLS_ENABLED
	if (editor) {
		packed_data->set_disabled(true);
		globals->set_disable_feature_overrides(true);
	}
#endif

#ifdef TOOLS_ENABLED
	if (editor) {
		Engine::get_singleton()->set_editor_hint(true);
		main_args.push_back("--editor");
		if (!init_windowed) {
			init_maximized = true;
			window_mode = DisplayServer::WINDOW_MODE_MAXIMIZED;
		}
	}

	if (!project_manager && !editor) {
		// If we didn't find a project, we fall back to the project manager.
		project_manager = !found_project && !cmdline_tool;
	}
#endif

	GLOBAL_DEF("debug/file_logging/enable_file_logging", false);
	// Only file logging by default on desktop platforms as logs can't be
	// accessed easily on mobile/Web platforms (if at all).
	// This also prevents logs from being created for the editor instance, as feature tags
	// are disabled while in the editor (even if they should logically apply).
	GLOBAL_DEF("debug/file_logging/enable_file_logging.pc", true);
	GLOBAL_DEF("debug/file_logging/log_path", "user://logs/godot.log");
	GLOBAL_DEF("debug/file_logging/max_log_files", 5);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/file_logging/max_log_files",
			PropertyInfo(Variant::INT,
					"debug/file_logging/max_log_files",
					PROPERTY_HINT_RANGE,
					"0,20,1,or_greater")); //no negative numbers
	if (!project_manager && !editor && FileAccess::get_create_func(FileAccess::ACCESS_USERDATA) &&
			GLOBAL_GET("debug/file_logging/enable_file_logging")) {
		// Don't create logs for the project manager as they would be written to
		// the current working directory, which is inconvenient.
		String base_path = GLOBAL_GET("debug/file_logging/log_path");
		int max_files = GLOBAL_GET("debug/file_logging/max_log_files");
		OS::get_singleton()->add_logger(memnew(RotatedFileLogger(base_path, max_files)));
	}

	if (main_args.size() == 0 && String(GLOBAL_GET("application/run/main_scene")) == "") {
#ifdef TOOLS_ENABLED
		if (!editor && !project_manager) {
#endif
			const String error_msg = "Error: Can't run project: no main scene defined in the project.\n";
			OS::get_singleton()->print("%s", error_msg.utf8().get_data());
			OS::get_singleton()->alert(error_msg);
			goto error;
#ifdef TOOLS_ENABLED
		}
#endif
	}

	if (editor || project_manager) {
		Engine::get_singleton()->set_editor_hint(true);
		use_custom_res = false;
		input_map->load_default(); //keys for editor
	} else {
		input_map->load_from_project_settings(); //keys for game
	}

	if (bool(ProjectSettings::get_singleton()->get("application/run/disable_stdout"))) {
		quiet_stdout = true;
	}
	if (bool(ProjectSettings::get_singleton()->get("application/run/disable_stderr"))) {
		_print_error_enabled = false;
	};

	if (quiet_stdout) {
		_print_line_enabled = false;
	}

	Logger::set_flush_stdout_on_print(ProjectSettings::get_singleton()->get("application/run/flush_stdout_on_print"));

	OS::get_singleton()->set_cmdline(execpath, main_args);

	register_core_extensions(); //before display

	GLOBAL_DEF("rendering/driver/driver_name", "Vulkan");
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/driver/driver_name",
			PropertyInfo(Variant::STRING,
					"rendering/driver/driver_name",
					PROPERTY_HINT_ENUM, "Vulkan"));
	if (display_driver == "") {
		display_driver = GLOBAL_GET("rendering/driver/driver_name");
	}

	GLOBAL_DEF_BASIC("display/window/size/width", 1024);
	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/width",
			PropertyInfo(Variant::INT, "display/window/size/width",
					PROPERTY_HINT_RANGE,
					"0,7680,or_greater")); // 8K resolution
	GLOBAL_DEF_BASIC("display/window/size/height", 600);
	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/height",
			PropertyInfo(Variant::INT, "display/window/size/height",
					PROPERTY_HINT_RANGE,
					"0,4320,or_greater")); // 8K resolution
	GLOBAL_DEF_BASIC("display/window/size/resizable", true);
	GLOBAL_DEF_BASIC("display/window/size/borderless", false);
	GLOBAL_DEF_BASIC("display/window/size/fullscreen", false);
	GLOBAL_DEF("display/window/size/always_on_top", false);
	GLOBAL_DEF("display/window/size/test_width", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/test_width",
			PropertyInfo(Variant::INT,
					"display/window/size/test_width",
					PROPERTY_HINT_RANGE,
					"0,7680,or_greater")); // 8K resolution
	GLOBAL_DEF("display/window/size/test_height", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/test_height",
			PropertyInfo(Variant::INT,
					"display/window/size/test_height",
					PROPERTY_HINT_RANGE,
					"0,4320,or_greater")); // 8K resolution

	if (use_custom_res) {
		if (!force_res) {
			window_size.width = GLOBAL_GET("display/window/size/width");
			window_size.height = GLOBAL_GET("display/window/size/height");

			if (globals->has_setting("display/window/size/test_width") &&
					globals->has_setting("display/window/size/test_height")) {
				int tw = globals->get("display/window/size/test_width");
				if (tw > 0) {
					window_size.width = tw;
				}
				int th = globals->get("display/window/size/test_height");
				if (th > 0) {
					window_size.height = th;
				}
			}
		}

		if (!bool(GLOBAL_GET("display/window/size/resizable"))) {
			window_flags |= DisplayServer::WINDOW_FLAG_RESIZE_DISABLED_BIT;
		}
		if (bool(GLOBAL_GET("display/window/size/borderless"))) {
			window_flags |= DisplayServer::WINDOW_FLAG_BORDERLESS_BIT;
		}
		if (bool(GLOBAL_GET("display/window/size/fullscreen"))) {
			window_mode = DisplayServer::WINDOW_MODE_FULLSCREEN;
		}

		if (bool(GLOBAL_GET("display/window/size/always_on_top"))) {
			window_flags |= DisplayServer::WINDOW_FLAG_ALWAYS_ON_TOP_BIT;
		}
	}

	GLOBAL_DEF("internationalization/rendering/force_right_to_left_layout_direction", false);
	GLOBAL_DEF("internationalization/locale/include_text_server_data", false);

	if (!force_lowdpi) {
		OS::get_singleton()->_allow_hidpi = GLOBAL_DEF("display/window/dpi/allow_hidpi", false);
	}

	/* todo restore
    OS::get_singleton()->_allow_layered = GLOBAL_DEF("display/window/per_pixel_transparency/allowed", false);
    video_mode.layered = GLOBAL_DEF("display/window/per_pixel_transparency/enabled", false);
*/
	if (editor || project_manager) {
		// The editor and project manager always detect and use hiDPI if needed
		OS::get_singleton()->_allow_hidpi = true;
		OS::get_singleton()->_allow_layered = false;
	}

	OS::get_singleton()->_keep_screen_on = GLOBAL_DEF("display/window/energy_saving/keep_screen_on", true);
	if (rtm == -1) {
		rtm = GLOBAL_DEF("rendering/driver/threads/thread_model", OS::RENDER_THREAD_SAFE);
	}

	if (rtm >= 0 && rtm < 3) {
#ifdef NO_THREADS
		rtm = OS::RENDER_THREAD_UNSAFE; // No threads available on this platform.
#else
		if (editor) {
			rtm = OS::RENDER_THREAD_SAFE;
		}
#endif
		OS::get_singleton()->_render_thread_mode = OS::RenderThreadMode(rtm);
	}

	/* Determine audio and video drivers */

	for (int i = 0; i < DisplayServer::get_create_function_count(); i++) {
		if (display_driver == DisplayServer::get_create_function_name(i)) {
			display_driver_idx = i;
			break;
		}
	}

	if (display_driver_idx < 0) {
		display_driver_idx = 0;
	}

	GLOBAL_DEF_RST_NOVAL("audio/driver/driver", AudioDriverManager::get_driver(0)->get_name());
	if (audio_driver == "") { // Specified in project.godot.
		audio_driver = GLOBAL_GET("audio/driver/driver");
	}

	for (int i = 0; i < AudioDriverManager::get_driver_count(); i++) {
		if (audio_driver == AudioDriverManager::get_driver(i)->get_name()) {
			audio_driver_idx = i;
			break;
		}
	}

	if (audio_driver_idx < 0) {
		audio_driver_idx = 0;
	}

	{
		window_orientation = DisplayServer::ScreenOrientation(int(GLOBAL_DEF_BASIC("display/window/handheld/orientation", DisplayServer::ScreenOrientation::SCREEN_LANDSCAPE)));
	}
	{
		window_vsync_mode = DisplayServer::VSyncMode(int(GLOBAL_DEF("display/window/vsync/vsync_mode", DisplayServer::VSyncMode::VSYNC_ENABLED)));
	}
	Engine::get_singleton()->set_physics_ticks_per_second(GLOBAL_DEF_BASIC("physics/common/physics_ticks_per_second", 60));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/common/physics_ticks_per_second",
			PropertyInfo(Variant::INT, "physics/common/physics_ticks_per_second",
					PROPERTY_HINT_RANGE, "1,1000,1"));
	Engine::get_singleton()->set_physics_jitter_fix(GLOBAL_DEF("physics/common/physics_jitter_fix", 0.5));
	Engine::get_singleton()->set_target_fps(GLOBAL_DEF("debug/settings/fps/force_fps", 0));
	ProjectSettings::get_singleton()->set_custom_property_info("debug/settings/fps/force_fps",
			PropertyInfo(Variant::INT,
					"debug/settings/fps/force_fps",
					PROPERTY_HINT_RANGE, "0,1000,1"));

	GLOBAL_DEF("debug/settings/stdout/print_fps", false);
	GLOBAL_DEF("debug/settings/stdout/print_gpu_profile", false);
	GLOBAL_DEF("debug/settings/stdout/verbose_stdout", false);

	if (!OS::get_singleton()->_verbose_stdout) { // Not manually overridden.
		OS::get_singleton()->_verbose_stdout = GLOBAL_GET("debug/settings/stdout/verbose_stdout");
	}

	if (frame_delay == 0) {
		frame_delay = GLOBAL_DEF("application/run/frame_delay_msec", 0);
		ProjectSettings::get_singleton()->set_custom_property_info("application/run/frame_delay_msec",
				PropertyInfo(Variant::INT,
						"application/run/frame_delay_msec",
						PROPERTY_HINT_RANGE,
						"0,100,1,or_greater")); // No negative numbers
	}

	OS::get_singleton()->set_low_processor_usage_mode(GLOBAL_DEF("application/run/low_processor_mode", false));
	OS::get_singleton()->set_low_processor_usage_mode_sleep_usec(
			GLOBAL_DEF("application/run/low_processor_mode_sleep_usec", 6900)); // Roughly 144 FPS
	ProjectSettings::get_singleton()->set_custom_property_info("application/run/low_processor_mode_sleep_usec",
			PropertyInfo(Variant::INT,
					"application/run/low_processor_mode_sleep_usec",
					PROPERTY_HINT_RANGE,
					"0,33200,1,or_greater")); // No negative numbers

	GLOBAL_DEF("display/window/ios/hide_home_indicator", true);
	GLOBAL_DEF("input_devices/pointing/ios/touch_delay", 0.150);

	Engine::get_singleton()->set_frame_delay(frame_delay);

	message_queue = memnew(MessageQueue);

	if (p_second_phase) {
		return setup2();
	}

	return OK;

error:

	text_driver = "";
	display_driver = "";
	audio_driver = "";
	tablet_driver = "";
	project_path = "";

	args.clear();
	main_args.clear();

	if (show_help) {
		print_help(execpath);
	}

	EngineDebugger::deinitialize();

	if (performance) {
		memdelete(performance);
	}
	if (input_map) {
		memdelete(input_map);
	}
	if (time_singleton) {
		memdelete(time_singleton);
	}
	if (translation_server) {
		memdelete(translation_server);
	}
	if (globals) {
		memdelete(globals);
	}
	if (engine) {
		memdelete(engine);
	}
	if (packed_data) {
		memdelete(packed_data);
	}
	if (file_access_network_client) {
		memdelete(file_access_network_client);
	}

	unregister_core_driver_types();
	unregister_core_types();

	OS::get_singleton()->_cmdline.clear();

	if (message_queue) {
		memdelete(message_queue);
	}
	OS::get_singleton()->finalize_core();
	locale = String();

	return ERR_INVALID_PARAMETER;
}

Error Main::setup2(Thread::ID p_main_tid_override) {
	tsman = memnew(TextServerManager);

	preregister_module_types();
	preregister_server_types();

	// Print engine name and version
	print_line(String(VERSION_NAME) + " v" + get_full_version_string() + " - " + String(VERSION_WEBSITE));

#if !defined(NO_THREADS)
	if (p_main_tid_override) {
		Thread::main_thread_id = p_main_tid_override;
	}
#endif

#ifdef TOOLS_ENABLED
	if (editor || project_manager || cmdline_tool) {
		EditorPaths::create();
	}
#endif

	/* Initialize Input */

	input = memnew(Input);

	/* Initialize Display Server */

	{
		String rendering_driver; // temp broken

		Error err;
		display_server = DisplayServer::create(display_driver_idx, rendering_driver, window_mode, window_vsync_mode, window_flags, window_size, err);
		if (err != OK || display_server == nullptr) {
			//ok i guess we can't use this display server, try other ones
			for (int i = 0; i < DisplayServer::get_create_function_count(); i++) {
				if (i == display_driver_idx) {
					continue; //don't try the same twice
				}
				display_server = DisplayServer::create(i, rendering_driver, window_mode, window_vsync_mode, window_flags, window_size, err);
				if (err == OK && display_server != nullptr) {
					break;
				}
			}
		}

		if (err != OK || display_server == nullptr) {
			ERR_PRINT("Unable to create DisplayServer, all display drivers failed.");
			return err;
		}
	}

	if (display_server->has_feature(DisplayServer::FEATURE_ORIENTATION)) {
		display_server->screen_set_orientation(window_orientation);
	}

	/* Initialize Pen Tablet Driver */

	{
		GLOBAL_DEF_RST_NOVAL("input_devices/pen_tablet/driver", "");
		GLOBAL_DEF_RST_NOVAL("input_devices/pen_tablet/driver.windows", "");
		ProjectSettings::get_singleton()->set_custom_property_info("input_devices/pen_tablet/driver.windows", PropertyInfo(Variant::STRING, "input_devices/pen_tablet/driver.windows", PROPERTY_HINT_ENUM, "wintab,winink"));
	}

	if (tablet_driver == "") { // specified in project.godot
		tablet_driver = GLOBAL_GET("input_devices/pen_tablet/driver");
		if (tablet_driver == "") {
			tablet_driver = DisplayServer::get_singleton()->tablet_get_driver_name(0);
		}
	}

	for (int i = 0; i < DisplayServer::get_singleton()->tablet_get_driver_count(); i++) {
		if (tablet_driver == DisplayServer::get_singleton()->tablet_get_driver_name(i)) {
			DisplayServer::get_singleton()->tablet_set_current_driver(DisplayServer::get_singleton()->tablet_get_driver_name(i));
			break;
		}
	}

	if (DisplayServer::get_singleton()->tablet_get_current_driver() == "") {
		DisplayServer::get_singleton()->tablet_set_current_driver(DisplayServer::get_singleton()->tablet_get_driver_name(0));
	}

	print_verbose("Using \"" + tablet_driver + "\" pen tablet driver...");

	/* Initialize Rendering Server */

	rendering_server = memnew(RenderingServerDefault(OS::get_singleton()->get_render_thread_mode() == OS::RENDER_SEPARATE_THREAD));

	rendering_server->init();
	rendering_server->set_render_loop_enabled(!disable_render_loop);

	if (profile_gpu || (!editor && bool(GLOBAL_GET("debug/settings/stdout/print_gpu_profile")))) {
		rendering_server->set_print_gpu_profile(true);
	}

#ifdef UNIX_ENABLED
	// Print warning after initializing the renderer but before initializing audio.
	if (OS::get_singleton()->get_environment("USER") == "root" && !OS::get_singleton()->has_environment("GODOT_SILENCE_ROOT_WARNING")) {
		WARN_PRINT("Started the engine as `root`/superuser. This is a security risk, and subsystems like audio may not work correctly.\nSet the environment variable `GODOT_SILENCE_ROOT_WARNING` to 1 to silence this warning.");
	}
#endif

	OS::get_singleton()->initialize_joypads();

	/* Initialize Audio Driver */

	AudioDriverManager::initialize(audio_driver_idx);

	print_line(" "); //add a blank line for readability

	if (init_use_custom_pos) {
		display_server->window_set_position(init_custom_pos);
	}

	// right moment to create and initialize the audio server

	audio_server = memnew(AudioServer);
	audio_server->init();

	// also init our xr_server from here
	xr_server = memnew(XRServer);

	register_core_singletons();

	MAIN_PRINT("Main: Setup Logo");

#if defined(JAVASCRIPT_ENABLED) || defined(ANDROID_ENABLED)
	bool show_logo = false;
#else
	bool show_logo = true;
#endif

	if (init_screen != -1) {
		DisplayServer::get_singleton()->window_set_current_screen(init_screen);
	}
	if (init_windowed) {
		//do none..
	} else if (init_maximized) {
		DisplayServer::get_singleton()->window_set_mode(DisplayServer::WINDOW_MODE_MAXIMIZED);
	} else if (init_fullscreen) {
		DisplayServer::get_singleton()->window_set_mode(DisplayServer::WINDOW_MODE_FULLSCREEN);
	}
	if (init_always_on_top) {
		DisplayServer::get_singleton()->window_set_flag(DisplayServer::WINDOW_FLAG_ALWAYS_ON_TOP, true);
	}

	if (allow_focus_steal_pid) {
		DisplayServer::get_singleton()->enable_for_stealing_focus(allow_focus_steal_pid);
	}

	register_server_types();

	MAIN_PRINT("Main: Load Boot Image");

	Color clear = GLOBAL_DEF("rendering/environment/defaults/default_clear_color", Color(0.3, 0.3, 0.3));
	RenderingServer::get_singleton()->set_default_clear_color(clear);

	if (show_logo) { //boot logo!
		String boot_logo_path = GLOBAL_DEF("application/boot_splash/image", String());
		bool boot_logo_scale = GLOBAL_DEF("application/boot_splash/fullsize", true);
		bool boot_logo_filter = GLOBAL_DEF("application/boot_splash/use_filter", true);
		ProjectSettings::get_singleton()->set_custom_property_info("application/boot_splash/image",
				PropertyInfo(Variant::STRING,
						"application/boot_splash/image",
						PROPERTY_HINT_FILE, "*.png"));

		Ref<Image> boot_logo;

		boot_logo_path = boot_logo_path.strip_edges();

		if (boot_logo_path != String()) {
			boot_logo.instantiate();
			Error load_err = ImageLoader::load_image(boot_logo_path, boot_logo);
			if (load_err) {
				ERR_PRINT("Non-existing or invalid boot splash at '" + boot_logo_path + "'. Loading default splash.");
			}
		}

#if defined(TOOLS_ENABLED) && !defined(NO_EDITOR_SPLASH)
		const Color boot_bg_color =
				GLOBAL_DEF("application/boot_splash/bg_color",
						(editor || project_manager) ? boot_splash_editor_bg_color : boot_splash_bg_color);
#else
		const Color boot_bg_color = GLOBAL_DEF("application/boot_splash/bg_color", boot_splash_bg_color);
#endif
		if (boot_logo.is_valid()) {
			RenderingServer::get_singleton()->set_boot_image(boot_logo, boot_bg_color, boot_logo_scale,
					boot_logo_filter);

		} else {
#ifndef NO_DEFAULT_BOOT_LOGO
			MAIN_PRINT("Main: Create bootsplash");
#if defined(TOOLS_ENABLED) && !defined(NO_EDITOR_SPLASH)
			Ref<Image> splash = (editor || project_manager) ? memnew(Image(boot_splash_editor_png)) : memnew(Image(boot_splash_png));
#else
			Ref<Image> splash = memnew(Image(boot_splash_png));
#endif

			MAIN_PRINT("Main: ClearColor");
			RenderingServer::get_singleton()->set_default_clear_color(boot_bg_color);
			MAIN_PRINT("Main: Image");
			RenderingServer::get_singleton()->set_boot_image(splash, boot_bg_color, false);
#endif
		}

#ifdef TOOLS_ENABLED
		if (OS::get_singleton()->get_bundle_icon_path().is_empty()) {
			Ref<Image> icon = memnew(Image(app_icon_png));
			DisplayServer::get_singleton()->set_icon(icon);
		}
#endif
	}

	MAIN_PRINT("Main: DCC");
	RenderingServer::get_singleton()->set_default_clear_color(
			GLOBAL_DEF("rendering/environment/defaults/default_clear_color", Color(0.3, 0.3, 0.3)));

	GLOBAL_DEF("application/config/icon", String());
	ProjectSettings::get_singleton()->set_custom_property_info("application/config/icon",
			PropertyInfo(Variant::STRING, "application/config/icon",
					PROPERTY_HINT_FILE, "*.png,*.webp,*.svg,*.svgz"));

	GLOBAL_DEF("application/config/macos_native_icon", String());
	ProjectSettings::get_singleton()->set_custom_property_info("application/config/macos_native_icon",
			PropertyInfo(Variant::STRING,
					"application/config/macos_native_icon",
					PROPERTY_HINT_FILE, "*.icns"));

	GLOBAL_DEF("application/config/windows_native_icon", String());
	ProjectSettings::get_singleton()->set_custom_property_info("application/config/windows_native_icon",
			PropertyInfo(Variant::STRING,
					"application/config/windows_native_icon",
					PROPERTY_HINT_FILE, "*.ico"));

	Input *id = Input::get_singleton();
	if (id) {
		agile_input_event_flushing = GLOBAL_DEF("input_devices/buffering/agile_event_flushing", false);

		if (bool(GLOBAL_DEF("input_devices/pointing/emulate_touch_from_mouse", false)) &&
				!(editor || project_manager)) {
			bool found_touchscreen = false;
			for (int i = 0; i < DisplayServer::get_singleton()->get_screen_count(); i++) {
				if (DisplayServer::get_singleton()->screen_is_touchscreen(i)) {
					found_touchscreen = true;
				}
			}
			if (!found_touchscreen) {
				//only if no touchscreen ui hint, set emulation
				id->set_emulate_touch_from_mouse(true);
			}
		}

		id->set_emulate_mouse_from_touch(bool(GLOBAL_DEF("input_devices/pointing/emulate_mouse_from_touch", true)));
	}

	MAIN_PRINT("Main: Load Translations and Remaps");

	translation_server->setup(); //register translations, load them, etc.
	if (locale != "") {
		translation_server->set_locale(locale);
	}
	translation_server->load_translations();
	ResourceLoader::load_translation_remaps(); //load remaps for resources

	ResourceLoader::load_path_remaps();

	MAIN_PRINT("Main: Load TextServer");

	/* Enum text drivers */
	GLOBAL_DEF("internationalization/rendering/text_driver", "");
	String text_driver_options;
	for (int i = 0; i < TextServerManager::get_singleton()->get_interface_count(); i++) {
		if (i > 0) {
			text_driver_options += ",";
		}
		text_driver_options += TextServerManager::get_singleton()->get_interface(i)->get_name();
	}
	ProjectSettings::get_singleton()->set_custom_property_info("internationalization/rendering/text_driver", PropertyInfo(Variant::STRING, "internationalization/rendering/text_driver", PROPERTY_HINT_ENUM, text_driver_options));

	/* Determine text driver */
	if (text_driver == "") {
		text_driver = GLOBAL_GET("internationalization/rendering/text_driver");
	}

	if (text_driver != "") {
		/* Load user selected text server. */
		for (int i = 0; i < TextServerManager::get_singleton()->get_interface_count(); i++) {
			if (TextServerManager::get_singleton()->get_interface(i)->get_name() == text_driver) {
				text_driver_idx = i;
				break;
			}
		}
	}

	if (text_driver_idx < 0) {
		/* If not selected, use one with the most features available. */
		int max_features = 0;
		for (int i = 0; i < TextServerManager::get_singleton()->get_interface_count(); i++) {
			uint32_t features = TextServerManager::get_singleton()->get_interface(i)->get_features();
			int feature_number = 0;
			while (features) {
				feature_number += features & 1;
				features >>= 1;
			}
			if (feature_number >= max_features) {
				max_features = feature_number;
				text_driver_idx = i;
			}
		}
	}
	if (text_driver_idx >= 0) {
		TextServerManager::get_singleton()->set_primary_interface(TextServerManager::get_singleton()->get_interface(text_driver_idx));
	} else {
		ERR_PRINT("TextServer: Unable to create TextServer interface.");
		return ERR_CANT_CREATE;
	}

	MAIN_PRINT("Main: Load Scene Types");

	register_scene_types();

#ifdef TOOLS_ENABLED
	ClassDB::set_current_api(ClassDB::API_EDITOR);
	EditorNode::register_editor_types();

	ClassDB::set_current_api(ClassDB::API_CORE);

#endif

	MAIN_PRINT("Main: Load Modules");

	register_platform_apis();
	register_module_types();

	GLOBAL_DEF("display/mouse_cursor/custom_image", String());
	GLOBAL_DEF("display/mouse_cursor/custom_image_hotspot", Vector2());
	GLOBAL_DEF("display/mouse_cursor/tooltip_position_offset", Point2(10, 10));
	ProjectSettings::get_singleton()->set_custom_property_info("display/mouse_cursor/custom_image",
			PropertyInfo(Variant::STRING,
					"display/mouse_cursor/custom_image",
					PROPERTY_HINT_FILE, "*.png,*.webp"));

	if (String(ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image")) != String()) {
		Ref<Texture2D> cursor = ResourceLoader::load(
				ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image"));
		if (cursor.is_valid()) {
			Vector2 hotspot = ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image_hotspot");
			Input::get_singleton()->set_custom_mouse_cursor(cursor, Input::CURSOR_ARROW, hotspot);
		}
	}

	camera_server = CameraServer::create();

	MAIN_PRINT("Main: Load Physics, Drivers, Scripts");

	initialize_physics();
	initialize_navigation_server();
	register_server_singletons();

	register_driver_types();

	// This loads global classes, so it must happen before custom loaders and savers are registered
	ScriptServer::init_languages();

	audio_server->load_default_bus_layout();

	if (use_debug_profiler && EngineDebugger::is_active()) {
		// Start the "scripts" profiler, used in local debugging.
		// We could add more, and make the CLI arg require a comma-separated list of profilers.
		EngineDebugger::get_singleton()->profiler_enable("scripts", true);
	}

	if (!project_manager) {
		// If not running the project manager, and now that the engine is
		// able to load resources, load the global shader variables.
		// If running on editor, don't load the textures because the editor
		// may want to import them first. Editor will reload those later.
		rendering_server->global_variables_load_settings(!editor);
	}

	_start_success = true;
	locale = String();

	ClassDB::set_current_api(ClassDB::API_NONE); //no more APIs are registered at this point

	print_verbose("CORE API HASH: " + uitos(ClassDB::get_api_hash(ClassDB::API_CORE)));
	print_verbose("EDITOR API HASH: " + uitos(ClassDB::get_api_hash(ClassDB::API_EDITOR)));
	MAIN_PRINT("Main: Done");

	return OK;
}

// everything the main loop needs to know about frame timings
static MainTimerSync main_timer_sync;

bool Main::start() {
	ERR_FAIL_COND_V(!_start_success, false);

	bool hasicon = false;
	String positional_arg;
	String game_path;
	String script;
	bool check_only = false;

#ifdef TOOLS_ENABLED
	String doc_tool_path;
	bool doc_base = true;
	String _export_preset;
	bool export_debug = false;
	bool export_pack_only = false;
#endif

	main_timer_sync.init(OS::get_singleton()->get_ticks_usec());
	List<String> args = OS::get_singleton()->get_cmdline_args();

	for (int i = 0; i < args.size(); i++) {
		// First check parameters that do not have an argument to the right.

		// Doctest Unit Testing Handler
		// Designed to override and pass arguments to the unit test handler.
		if (args[i] == "--check-only") {
			check_only = true;
#ifdef TOOLS_ENABLED
		} else if (args[i] == "--no-docbase") {
			doc_base = false;
		} else if (args[i] == "-e" || args[i] == "--editor") {
			editor = true;
		} else if (args[i] == "-p" || args[i] == "--project-manager") {
			project_manager = true;
#endif
		} else if (args[i].length() && args[i][0] != '-' && positional_arg == "") {
			positional_arg = args[i];

			if (args[i].ends_with(".scn") ||
					args[i].ends_with(".tscn") ||
					args[i].ends_with(".escn") ||
					args[i].ends_with(".res") ||
					args[i].ends_with(".tres")) {
				// Only consider the positional argument to be a scene path if it ends with
				// a file extension associated with Godot scenes. This makes it possible
				// for projects to parse command-line arguments for custom CLI arguments
				// or other file extensions without trouble. This can be used to implement
				// "drag-and-drop onto executable" logic, which can prove helpful
				// for non-game applications.
				game_path = args[i];
			}
		}
		// Then parameters that have an argument to the right.
		else if (i < (args.size() - 1)) {
			bool parsed_pair = true;
			if (args[i] == "-s" || args[i] == "--script") {
				script = args[i + 1];
#ifdef TOOLS_ENABLED
			} else if (args[i] == "--doctool") {
				doc_tool_path = args[i + 1];
				if (doc_tool_path.begins_with("-")) {
					// Assuming other command line arg, so default to cwd.
					doc_tool_path = ".";
					parsed_pair = false;
				}
			} else if (args[i] == "--export") {
				editor = true; //needs editor
				_export_preset = args[i + 1];
			} else if (args[i] == "--export-debug") {
				editor = true; //needs editor
				_export_preset = args[i + 1];
				export_debug = true;
			} else if (args[i] == "--export-pack") {
				editor = true;
				_export_preset = args[i + 1];
				export_pack_only = true;
#endif
			} else {
				// The parameter does not match anything known, don't skip the next argument
				parsed_pair = false;
			}
			if (parsed_pair) {
				i++;
			}
		}
#ifdef TOOLS_ENABLED
		// Handle case where no path is given to --doctool.
		else if (args[i] == "--doctool") {
			doc_tool_path = ".";
		}
#endif
	}

#ifdef TOOLS_ENABLED
	if (doc_tool_path != "") {
		// Needed to instance editor-only classes for their default values
		Engine::get_singleton()->set_editor_hint(true);

		{
			DirAccessRef da = DirAccess::open(doc_tool_path);
			ERR_FAIL_COND_V_MSG(!da, false, "Argument supplied to --doctool must be a valid directory path.");
		}

#ifndef MODULE_MONO_ENABLED
		// Hack to define Mono-specific project settings even on non-Mono builds,
		// so that we don't lose their descriptions and default values in DocData.
		// Default values should be synced with mono_gd/gd_mono.cpp.
		GLOBAL_DEF("mono/debugger_agent/port", 23685);
		GLOBAL_DEF("mono/debugger_agent/wait_for_debugger", false);
		GLOBAL_DEF("mono/debugger_agent/wait_timeout", 3000);
		GLOBAL_DEF("mono/profiler/args", "log:calls,alloc,sample,output=output.mlpd");
		GLOBAL_DEF("mono/profiler/enabled", false);
		GLOBAL_DEF("mono/unhandled_exception_policy", 0);
		// From editor/csharp_project.cpp.
		GLOBAL_DEF("mono/project/auto_update_project", true);
#endif

		DocTools doc;
		doc.generate(doc_base);

		DocTools docsrc;
		Map<String, String> doc_data_classes;
		Set<String> checked_paths;
		print_line("Loading docs...");

		for (int i = 0; i < _doc_data_class_path_count; i++) {
			// Custom modules are always located by absolute path.
			String path = _doc_data_class_paths[i].path;
			if (path.is_relative_path()) {
				path = doc_tool_path.plus_file(path);
			}
			String name = _doc_data_class_paths[i].name;
			doc_data_classes[name] = path;
			if (!checked_paths.has(path)) {
				checked_paths.insert(path);

				// Create the module documentation directory if it doesn't exist
				DirAccess *da = DirAccess::create_for_path(path);
				da->make_dir_recursive(path);
				memdelete(da);

				docsrc.load_classes(path);
				print_line("Loading docs from: " + path);
			}
		}

		String index_path = doc_tool_path.plus_file("doc/classes");
		// Create the main documentation directory if it doesn't exist
		DirAccess *da = DirAccess::create_for_path(index_path);
		da->make_dir_recursive(index_path);
		memdelete(da);

		docsrc.load_classes(index_path);
		checked_paths.insert(index_path);
		print_line("Loading docs from: " + index_path);

		print_line("Merging docs...");
		doc.merge_from(docsrc);
		for (Set<String>::Element *E = checked_paths.front(); E; E = E->next()) {
			print_line("Erasing old docs at: " + E->get());
			DocTools::erase_classes(E->get());
		}

		print_line("Generating new docs...");
		doc.save_classes(index_path, doc_data_classes);

		return false;
	}

	if (dump_extension_api) {
		NativeExtensionAPIDump::generate_extension_json_file("extension_api.json");
		return false;
	}
#endif

	if (script == "" && game_path == "" && String(GLOBAL_GET("application/run/main_scene")) != "") {
		game_path = GLOBAL_GET("application/run/main_scene");
	}

#ifdef TOOLS_ENABLED
	if (!editor && !project_manager && !cmdline_tool && script == "" && game_path == "") {
		// If we end up here, it means we didn't manage to detect what we want to run.
		// Let's throw an error gently. The code leading to this is pretty brittle so
		// this might end up triggered by valid usage, in which case we'll have to
		// fine-tune further.
		OS::get_singleton()->alert("Couldn't detect whether to run the editor, the project manager or a specific project. Aborting.");
		ERR_FAIL_V_MSG(false, "Couldn't detect whether to run the editor, the project manager or a specific project. Aborting.");
	}
#endif

	MainLoop *main_loop = nullptr;
	if (editor) {
		main_loop = memnew(SceneTree);
	}
	String main_loop_type = GLOBAL_DEF("application/run/main_loop_type", "SceneTree");

	if (script != "") {
		Ref<Script> script_res = ResourceLoader::load(script);
		ERR_FAIL_COND_V_MSG(script_res.is_null(), false, "Can't load script: " + script);

		if (check_only) {
			if (!script_res->is_valid()) {
				OS::get_singleton()->set_exit_code(EXIT_FAILURE);
			}
			return false;
		}

		if (script_res->can_instantiate()) {
			StringName instance_type = script_res->get_instance_base_type();
			Object *obj = ClassDB::instantiate(instance_type);
			MainLoop *script_loop = Object::cast_to<MainLoop>(obj);
			if (!script_loop) {
				if (obj) {
					memdelete(obj);
				}
				OS::get_singleton()->alert(vformat("Can't load the script \"%s\" as it doesn't inherit from SceneTree or MainLoop.", script));
				ERR_FAIL_V_MSG(false, vformat("Can't load the script \"%s\" as it doesn't inherit from SceneTree or MainLoop.", script));
			}

			script_loop->set_initialize_script(script_res);
			main_loop = script_loop;
		} else {
			return false;
		}
	} else { // Not based on script path.
		if (!editor && !ClassDB::class_exists(main_loop_type) && ScriptServer::is_global_class(main_loop_type)) {
			String script_path = ScriptServer::get_global_class_path(main_loop_type);
			Ref<Script> script_res = ResourceLoader::load(script_path);
			StringName script_base = ScriptServer::get_global_class_native_base(main_loop_type);
			Object *obj = ClassDB::instantiate(script_base);
			MainLoop *script_loop = Object::cast_to<MainLoop>(obj);
			if (!script_loop) {
				if (obj) {
					memdelete(obj);
				}
				OS::get_singleton()->alert("Error: Invalid MainLoop script base type: " + script_base);
				ERR_FAIL_V_MSG(false, vformat("The global class %s does not inherit from SceneTree or MainLoop.", main_loop_type));
			}
			script_loop->set_initialize_script(script_res);
			main_loop = script_loop;
		}
	}

	if (!main_loop && main_loop_type == "") {
		main_loop_type = "SceneTree";
	}

	if (!main_loop) {
		if (!ClassDB::class_exists(main_loop_type)) {
			OS::get_singleton()->alert("Error: MainLoop type doesn't exist: " + main_loop_type);
			return false;
		} else {
			Object *ml = ClassDB::instantiate(main_loop_type);
			ERR_FAIL_COND_V_MSG(!ml, false, "Can't instance MainLoop type.");

			main_loop = Object::cast_to<MainLoop>(ml);
			if (!main_loop) {
				memdelete(ml);
				ERR_FAIL_V_MSG(false, "Invalid MainLoop type.");
			}
		}
	}

	if (main_loop->is_class("SceneTree")) {
		SceneTree *sml = Object::cast_to<SceneTree>(main_loop);

#ifdef DEBUG_ENABLED
		if (debug_collisions) {
			sml->set_debug_collisions_hint(true);
		}
		if (debug_navigation) {
			sml->set_debug_navigation_hint(true);
		}
#endif

		bool embed_subwindows = GLOBAL_DEF("display/window/subwindows/embed_subwindows", true);

		if (OS::get_singleton()->is_single_window() || (!project_manager && !editor && embed_subwindows)) {
			sml->get_root()->set_embed_subwindows_hint(true);
		}
		ResourceLoader::add_custom_loaders();
		ResourceSaver::add_custom_savers();

		if (!project_manager && !editor) { // game
			if (game_path != "" || script != "") {
				//autoload
				OrderedHashMap<StringName, ProjectSettings::AutoloadInfo> autoloads = ProjectSettings::get_singleton()->get_autoload_list();

				//first pass, add the constants so they exist before any script is loaded
				for (OrderedHashMap<StringName, ProjectSettings::AutoloadInfo>::Element E = autoloads.front(); E; E = E.next()) {
					const ProjectSettings::AutoloadInfo &info = E.get();

					if (info.is_singleton) {
						for (int i = 0; i < ScriptServer::get_language_count(); i++) {
							ScriptServer::get_language(i)->add_global_constant(info.name, Variant());
						}
					}
				}

				//second pass, load into global constants
				List<Node *> to_add;
				for (OrderedHashMap<StringName, ProjectSettings::AutoloadInfo>::Element E = autoloads.front(); E; E = E.next()) {
					const ProjectSettings::AutoloadInfo &info = E.get();

					RES res = ResourceLoader::load(info.path);
					ERR_CONTINUE_MSG(res.is_null(), "Can't autoload: " + info.path);
					Node *n = nullptr;
					if (res->is_class("PackedScene")) {
						Ref<PackedScene> ps = res;
						n = ps->instantiate();
					} else if (res->is_class("Script")) {
						Ref<Script> script_res = res;
						StringName ibt = script_res->get_instance_base_type();
						bool valid_type = ClassDB::is_parent_class(ibt, "Node");
						ERR_CONTINUE_MSG(!valid_type, "Script does not inherit a Node: " + info.path);

						Object *obj = ClassDB::instantiate(ibt);

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

				for (Node *E : to_add) {
					sml->get_root()->add_child(E);
				}
			}
		}

#ifdef TOOLS_ENABLED
		EditorNode *editor_node = nullptr;
		if (editor) {
			editor_node = memnew(EditorNode);
			sml->get_root()->add_child(editor_node);

			if (_export_preset != "") {
				editor_node->export_preset(_export_preset, positional_arg, export_debug, export_pack_only);
				game_path = ""; // Do not load anything.
			}
		}
#endif

		if (!editor && !project_manager) {
			//standard helpers that can be changed from main config

			String stretch_mode = GLOBAL_DEF_BASIC("display/window/stretch/mode", "disabled");
			String stretch_aspect = GLOBAL_DEF_BASIC("display/window/stretch/aspect", "keep");
			Size2i stretch_size = Size2i(GLOBAL_DEF_BASIC("display/window/size/width", 0),
					GLOBAL_DEF_BASIC("display/window/size/height", 0));

			Window::ContentScaleMode cs_sm = Window::CONTENT_SCALE_MODE_DISABLED;
			if (stretch_mode == "canvas_items") {
				cs_sm = Window::CONTENT_SCALE_MODE_CANVAS_ITEMS;
			} else if (stretch_mode == "viewport") {
				cs_sm = Window::CONTENT_SCALE_MODE_VIEWPORT;
			}

			Window::ContentScaleAspect cs_aspect = Window::CONTENT_SCALE_ASPECT_IGNORE;
			if (stretch_aspect == "keep") {
				cs_aspect = Window::CONTENT_SCALE_ASPECT_KEEP;
			} else if (stretch_aspect == "keep_width") {
				cs_aspect = Window::CONTENT_SCALE_ASPECT_KEEP_WIDTH;
			} else if (stretch_aspect == "keep_height") {
				cs_aspect = Window::CONTENT_SCALE_ASPECT_KEEP_HEIGHT;
			} else if (stretch_aspect == "expand") {
				cs_aspect = Window::CONTENT_SCALE_ASPECT_EXPAND;
			}

			sml->get_root()->set_content_scale_mode(cs_sm);
			sml->get_root()->set_content_scale_aspect(cs_aspect);
			sml->get_root()->set_content_scale_size(stretch_size);

			sml->set_auto_accept_quit(GLOBAL_DEF("application/config/auto_accept_quit", true));
			sml->set_quit_on_go_back(GLOBAL_DEF("application/config/quit_on_go_back", true));
			String appname = ProjectSettings::get_singleton()->get("application/config/name");
			appname = TranslationServer::get_singleton()->translate(appname);
#ifdef DEBUG_ENABLED
			// Append a suffix to the window title to denote that the project is running
			// from a debug build (including the editor). Since this results in lower performance,
			// this should be clearly presented to the user.
			DisplayServer::get_singleton()->window_set_title(vformat("%s (DEBUG)", appname));
#else
			DisplayServer::get_singleton()->window_set_title(appname);
#endif

			// Define a very small minimum window size to prevent bugs such as GH-37242.
			// It can still be overridden by the user in a script.
			DisplayServer::get_singleton()->window_set_min_size(Size2i(64, 64));

			bool snap_controls = GLOBAL_DEF("gui/common/snap_controls_to_pixels", true);
			sml->get_root()->set_snap_controls_to_pixels(snap_controls);

			bool font_oversampling = GLOBAL_DEF("gui/fonts/dynamic_fonts/use_oversampling", true);
			sml->get_root()->set_use_font_oversampling(font_oversampling);

			int texture_filter = GLOBAL_DEF("rendering/textures/canvas_textures/default_texture_filter", 1);
			int texture_repeat = GLOBAL_DEF("rendering/textures/canvas_textures/default_texture_repeat", 0);
			sml->get_root()->set_default_canvas_item_texture_filter(
					Viewport::DefaultCanvasItemTextureFilter(texture_filter));
			sml->get_root()->set_default_canvas_item_texture_repeat(
					Viewport::DefaultCanvasItemTextureRepeat(texture_repeat));

		} else {
			GLOBAL_DEF_BASIC("display/window/stretch/mode", "disabled");
			ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/mode",
					PropertyInfo(Variant::STRING,
							"display/window/stretch/mode",
							PROPERTY_HINT_ENUM,
							"disabled,canvas_items,viewport"));
			GLOBAL_DEF_BASIC("display/window/stretch/aspect", "keep");
			ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/aspect",
					PropertyInfo(Variant::STRING,
							"display/window/stretch/aspect",
							PROPERTY_HINT_ENUM,
							"ignore,keep,keep_width,keep_height,expand"));
			GLOBAL_DEF_BASIC("display/window/stretch/shrink", 1.0);
			ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/shrink",
					PropertyInfo(Variant::FLOAT,
							"display/window/stretch/shrink",
							PROPERTY_HINT_RANGE,
							"1.0,8.0,0.1"));
			sml->set_auto_accept_quit(GLOBAL_DEF("application/config/auto_accept_quit", true));
			sml->set_quit_on_go_back(GLOBAL_DEF("application/config/quit_on_go_back", true));
			GLOBAL_DEF_BASIC("gui/common/snap_controls_to_pixels", true);
			GLOBAL_DEF_BASIC("gui/fonts/dynamic_fonts/use_oversampling", true);

			GLOBAL_DEF_BASIC("rendering/textures/canvas_textures/default_texture_filter", 1);
			ProjectSettings::get_singleton()->set_custom_property_info(
					"rendering/textures/canvas_textures/default_texture_filter",
					PropertyInfo(Variant::INT, "rendering/textures/canvas_textures/default_texture_filter", PROPERTY_HINT_ENUM,
							"Nearest,Linear,Linear Mipmap,Nearest Mipmap"));
			GLOBAL_DEF_BASIC("rendering/textures/canvas_textures/default_texture_repeat", 0);
			ProjectSettings::get_singleton()->set_custom_property_info(
					"rendering/textures/canvas_textures/default_texture_repeat",
					PropertyInfo(Variant::INT, "rendering/textures/canvas_textures/default_texture_repeat", PROPERTY_HINT_ENUM,
							"Disable,Enable,Mirror"));
		}

#ifdef TOOLS_ENABLED
		if (editor) {
			bool editor_embed_subwindows = EditorSettings::get_singleton()->get_setting(
					"interface/editor/single_window_mode");

			if (editor_embed_subwindows) {
				sml->get_root()->set_embed_subwindows_hint(true);
			}
		}
#endif

		String local_game_path;
		if (game_path != "" && !project_manager) {
			local_game_path = game_path.replace("\\", "/");

			if (!local_game_path.begins_with("res://")) {
				bool absolute =
						(local_game_path.size() > 1) && (local_game_path[0] == '/' || local_game_path[1] == ':');

				if (!absolute) {
					if (ProjectSettings::get_singleton()->is_using_datapack()) {
						local_game_path = "res://" + local_game_path;

					} else {
						int sep = local_game_path.rfind("/");

						if (sep == -1) {
							DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
							local_game_path = da->get_current_dir().plus_file(local_game_path);
							memdelete(da);
						} else {
							DirAccess *da = DirAccess::open(local_game_path.substr(0, sep));
							if (da) {
								local_game_path = da->get_current_dir().plus_file(
										local_game_path.substr(sep + 1, local_game_path.length()));
								memdelete(da);
							}
						}
					}
				}
			}

			local_game_path = ProjectSettings::get_singleton()->localize_path(local_game_path);

#ifdef TOOLS_ENABLED
			if (editor) {
				if (game_path != String(GLOBAL_GET("application/run/main_scene")) || !editor_node->has_scenes_in_session()) {
					Error serr = editor_node->load_scene(local_game_path);
					if (serr != OK) {
						ERR_PRINT("Failed to load scene");
					}
				}
				DisplayServer::get_singleton()->set_context(DisplayServer::CONTEXT_EDITOR);
				if (!debug_server_uri.is_empty()) {
					EditorDebuggerNode::get_singleton()->start(debug_server_uri);
				}
			}
#endif
			if (!editor) {
				DisplayServer::get_singleton()->set_context(DisplayServer::CONTEXT_ENGINE);
			}
		}

		if (!project_manager && !editor) { // game

			// Load SSL Certificates from Project Settings (or builtin).
			Crypto::load_default_certificates(GLOBAL_DEF("network/ssl/certificate_bundle_override", ""));

			if (game_path != "") {
				Node *scene = nullptr;
				Ref<PackedScene> scenedata = ResourceLoader::load(local_game_path);
				if (scenedata.is_valid()) {
					scene = scenedata->instantiate();
				}

				ERR_FAIL_COND_V_MSG(!scene, false, "Failed loading scene: " + local_game_path);
				sml->add_current_scene(scene);

#ifdef OSX_ENABLED
				String mac_iconpath = GLOBAL_DEF("application/config/macos_native_icon", "Variant()");
				if (mac_iconpath != "") {
					DisplayServer::get_singleton()->set_native_icon(mac_iconpath);
					hasicon = true;
				}
#endif

#ifdef WINDOWS_ENABLED
				String win_iconpath = GLOBAL_DEF("application/config/windows_native_icon", "Variant()");
				if (win_iconpath != "") {
					DisplayServer::get_singleton()->set_native_icon(win_iconpath);
					hasicon = true;
				}
#endif

				String iconpath = GLOBAL_DEF("application/config/icon", "Variant()");
				if ((iconpath != "") && (!hasicon)) {
					Ref<Image> icon;
					icon.instantiate();
					if (ImageLoader::load_image(iconpath, icon) == OK) {
						DisplayServer::get_singleton()->set_icon(icon);
						hasicon = true;
					}
				}
			}
		}

#ifdef TOOLS_ENABLED
		if (project_manager) {
			Engine::get_singleton()->set_editor_hint(true);
			ProjectManager *pmanager = memnew(ProjectManager);
			ProgressDialog *progress_dialog = memnew(ProgressDialog);
			pmanager->add_child(progress_dialog);
			sml->get_root()->add_child(pmanager);
			DisplayServer::get_singleton()->set_context(DisplayServer::CONTEXT_PROJECTMAN);
		}

		if (project_manager || editor) {
			if (DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_CONSOLE_WINDOW)) {
				// Hide console window if requested (Windows-only).
				bool hide_console = EditorSettings::get_singleton()->get_setting(
						"interface/editor/hide_console_window");
				DisplayServer::get_singleton()->console_set_visible(!hide_console);
			}

			// Load SSL Certificates from Editor Settings (or builtin)
			Crypto::load_default_certificates(
					EditorSettings::get_singleton()->get_setting("network/ssl/editor_ssl_certificates").operator String());
		}
#endif
	}

	if (!hasicon && OS::get_singleton()->get_bundle_icon_path().is_empty()) {
		Ref<Image> icon = memnew(Image(app_icon_png));
		DisplayServer::get_singleton()->set_icon(icon);
	}

	OS::get_singleton()->set_main_loop(main_loop);

	return true;
}

/* Main iteration
 *
 * This is the iteration of the engine's game loop, advancing the state of physics,
 * rendering and audio.
 * It's called directly by the platform's OS::run method, where the loop is created
 * and monitored.
 *
 * The OS implementation can impact its draw step with the Main::force_redraw() method.
 */

uint64_t Main::last_ticks = 0;
uint32_t Main::frames = 0;
uint32_t Main::frame = 0;
bool Main::force_redraw_requested = false;
int Main::iterating = 0;
bool Main::agile_input_event_flushing = false;

bool Main::is_iterating() {
	return iterating > 0;
}

// For performance metrics.
static uint64_t physics_process_max = 0;
static uint64_t process_max = 0;

bool Main::iteration() {
	//for now do not error on this
	//ERR_FAIL_COND_V(iterating, false);

	iterating++;

	const uint64_t ticks = OS::get_singleton()->get_ticks_usec();
	Engine::get_singleton()->_frame_ticks = ticks;
	main_timer_sync.set_cpu_ticks_usec(ticks);
	main_timer_sync.set_fixed_fps(fixed_fps);

	const uint64_t ticks_elapsed = ticks - last_ticks;

	const int physics_ticks_per_second = Engine::get_singleton()->get_physics_ticks_per_second();
	const double physics_step = 1.0 / physics_ticks_per_second;

	const double time_scale = Engine::get_singleton()->get_time_scale();

	MainFrameTime advance = main_timer_sync.advance(physics_step, physics_ticks_per_second);
	double process_step = advance.process_step;
	double scaled_step = process_step * time_scale;

	Engine::get_singleton()->_process_step = process_step;
	Engine::get_singleton()->_physics_interpolation_fraction = advance.interpolation_fraction;

	uint64_t physics_process_ticks = 0;
	uint64_t process_ticks = 0;

	frame += ticks_elapsed;

	last_ticks = ticks;

	static const int max_physics_steps = 8;
	if (fixed_fps == -1 && advance.physics_steps > max_physics_steps) {
		process_step -= (advance.physics_steps - max_physics_steps) * physics_step;
		advance.physics_steps = max_physics_steps;
	}

	bool exit = false;

	// process all our active interfaces
	XRServer::get_singleton()->_process();

	for (int iters = 0; iters < advance.physics_steps; ++iters) {
		if (Input::get_singleton()->is_using_input_buffering() && agile_input_event_flushing) {
			Input::get_singleton()->flush_buffered_events();
		}

		Engine::get_singleton()->_in_physics = true;

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

		Engine::get_singleton()->_in_physics = false;
	}

	if (Input::get_singleton()->is_using_input_buffering() && agile_input_event_flushing) {
		Input::get_singleton()->flush_buffered_events();
	}

	uint64_t process_begin = OS::get_singleton()->get_ticks_usec();

	if (OS::get_singleton()->get_main_loop()->process(process_step * time_scale)) {
		exit = true;
	}
	message_queue->flush();

	RenderingServer::get_singleton()->sync(); //sync if still drawing from previous frames.

	if (DisplayServer::get_singleton()->can_any_window_draw() &&
			RenderingServer::get_singleton()->is_render_loop_enabled()) {
		if ((!force_redraw_requested) && OS::get_singleton()->is_in_low_processor_usage_mode()) {
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
		if (editor || project_manager) {
			if (print_fps) {
				print_line(vformat("Editor FPS: %d (%s mspf)", frames, rtos(1000.0 / frames).pad_decimals(1)));
			}
		} else if (GLOBAL_GET("debug/settings/stdout/print_fps") || print_fps) {
			print_line(vformat("Project FPS: %d (%s mspf)", frames, rtos(1000.0 / frames).pad_decimals(1)));
		}

		Engine::get_singleton()->_fps = frames;
		performance->set_process_time(USEC_TO_SEC(process_max));
		performance->set_physics_process_time(USEC_TO_SEC(physics_process_max));
		process_max = 0;
		physics_process_max = 0;

		frame %= 1000000;
		frames = 0;
	}

	iterating--;

	// Needed for OSs using input buffering regardless accumulation (like Android)
	if (Input::get_singleton()->is_using_input_buffering() && !agile_input_event_flushing) {
		Input::get_singleton()->flush_buffered_events();
	}

	if (fixed_fps != -1) {
		return exit;
	}

	OS::get_singleton()->add_frame_delay(DisplayServer::get_singleton()->window_can_draw());

#ifdef TOOLS_ENABLED
	if (auto_build_solutions) {
		auto_build_solutions = false;
		// Only relevant when running the editor.
		if (!editor) {
			ERR_FAIL_V_MSG(true,
					"Command line option --build-solutions was passed, but no project is being edited. Aborting.");
		}
		if (!EditorNode::get_singleton()->call_build()) {
			ERR_FAIL_V_MSG(true,
					"Command line option --build-solutions was passed, but the build callback failed. Aborting.");
		}
	}
#endif

	return exit || auto_quit;
}

void Main::force_redraw() {
	force_redraw_requested = true;
}

/* Engine deinitialization
 *
 * Responsible for freeing all the memory allocated by previous setup steps,
 * so that the engine closes cleanly without leaking memory or crashing.
 * The order matters as some of those steps are linked with each other.
 */
void Main::cleanup(bool p_force) {
	if (!p_force) {
		ERR_FAIL_COND(!_start_success);
	}

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

	if (xr_server) {
		// Now that we're unregistering properly in plugins we need to keep access to xr_server for a little longer
		// We do however unset our primary interface
		xr_server->set_primary_interface(Ref<XRInterface>());
	}

	unregister_driver_types();

#ifdef TOOLS_ENABLED
	EditorNode::unregister_editor_types();
#endif

	ImageLoader::cleanup();

	unregister_module_types();
	unregister_platform_apis();
	unregister_scene_types();
	unregister_server_types();

	if (xr_server) {
		memdelete(xr_server);
	}

	if (audio_server) {
		audio_server->finish();
		memdelete(audio_server);
	}

	if (camera_server) {
		memdelete(camera_server);
	}

	OS::get_singleton()->finalize();

	finalize_physics();
	finalize_navigation_server();
	finalize_display();

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
	if (time_singleton) {
		memdelete(time_singleton);
	}
	if (translation_server) {
		memdelete(translation_server);
	}
	if (tsman) {
		memdelete(tsman);
	}
	if (globals) {
		memdelete(globals);
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

	OS::get_singleton()->finalize_core();
}
