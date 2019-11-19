/*************************************************************************/
/*  main.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "core/crypto/crypto.h"
#include "core/input_map.h"
#include "core/io/file_access_network.h"
#include "core/io/file_access_pack.h"
#include "core/io/file_access_zip.h"
#include "core/io/image_loader.h"
#include "core/io/ip.h"
#include "core/io/resource_loader.h"
#include "core/message_queue.h"
#include "core/os/dir_access.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/register_core_types.h"
#include "core/script_debugger_local.h"
#include "core/script_language.h"
#include "core/translation.h"
#include "core/version.h"
#include "core/version_hash.gen.h"
#include "drivers/register_driver_types.h"
#include "main/app_icon.gen.h"
#include "main/input_default.h"
#include "main/main_timer_sync.h"
#include "main/performance.h"
#include "main/splash.gen.h"
#include "main/splash_editor.gen.h"
#include "main/tests/test_main.h"
#include "modules/register_module_types.h"
#include "platform/register_platform_apis.h"
#include "scene/debugger/script_debugger_remote.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/register_scene_types.h"
#include "scene/resources/packed_scene.h"
#include "servers/arvr_server.h"
#include "servers/audio_server.h"
#include "servers/camera_server.h"
#include "servers/physics_2d_server.h"
#include "servers/physics_server.h"
#include "servers/register_server_types.h"

#ifdef TOOLS_ENABLED
#include "editor/doc/doc_data.h"
#include "editor/doc/doc_data_class_path.gen.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/project_manager.h"
#endif

/* Static members */

// Singletons

// Initialized in setup()
static Engine *engine = NULL;
static ProjectSettings *globals = NULL;
static InputMap *input_map = NULL;
static TranslationServer *translation_server = NULL;
static Performance *performance = NULL;
static PackedData *packed_data = NULL;
#ifdef MINIZIP_ENABLED
static ZipArchive *zip_packed_data = NULL;
#endif
static FileAccessNetworkClient *file_access_network_client = NULL;
static ScriptDebugger *script_debugger = NULL;
static MessageQueue *message_queue = NULL;

// Initialized in setup2()
static AudioServer *audio_server = NULL;
static CameraServer *camera_server = NULL;
static ARVRServer *arvr_server = NULL;
static PhysicsServer *physics_server = NULL;
static Physics2DServer *physics_2d_server = NULL;

// Data only needed to initialize after the argument parsing

// Drivers
static int video_driver_idx = -1;
static int audio_driver_idx = -1;

// Engine config/tools
static String locale;
static OS::ProcessID allow_focus_steal_pid = 0;
#ifdef TOOLS_ENABLED
static bool auto_build_solutions = false;
#endif

// Display
static OS::VideoMode video_mode;
static Vector2 init_custom_pos;
static int init_screen = -1;
static bool init_fullscreen = false;
static bool init_maximized = false;
static bool init_windowed = false;
static bool init_always_on_top = false;
static bool init_use_custom_pos = false;
static bool force_lowdpi = false;

// Debug
#ifdef DEBUG_ENABLED
static bool debug_collisions = false;
static bool debug_navigation = false;
#endif
static int frame_delay = 0;

/* Helper methods */

// Used by Mono module, should likely be registered in Engine singleton instead
// FIXME: This is also not 100% accurate, `project_manager` is only true when it was requested,
// but not if e.g. we fail to load and project and fallback to the manager.
bool Main::is_project_manager() {
	return project_manager;
}

// FIXME: Could maybe be moved to PhysicsServerManager and Physics2DServerManager directly
// to have less code in main.cpp.
void initialize_physics() {
	/// 3D Physics Server
	physics_server = PhysicsServerManager::new_server(ProjectSettings::get_singleton()->get(PhysicsServerManager::setting_property_name));
	if (!physics_server) {
		// Physics server not found, Use the default physics
		physics_server = PhysicsServerManager::new_default_server();
	}
	ERR_FAIL_COND(!physics_server);
	physics_server->init();

	/// 2D Physics server
	physics_2d_server = Physics2DServerManager::new_server(ProjectSettings::get_singleton()->get(Physics2DServerManager::setting_property_name));
	if (!physics_2d_server) {
		// Physics server not found, Use the default physics
		physics_2d_server = Physics2DServerManager::new_default_server();
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

//#define DEBUG_INIT
#ifdef DEBUG_INIT
#define MAIN_PRINT(m_txt) print_line(m_txt)
#else
#define MAIN_PRINT(m_txt)
#endif

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
 */

Error Main::setup(const char *execpath, int argc, char *argv[], bool p_second_phase) {
	RID_OwnerBase::init_rid();

	OS::get_singleton()->initialize_core();

	engine = memnew(Engine);

	ClassDB::init();

	MAIN_PRINT("Main: Initialize CORE");

	register_core_types();
	register_core_driver_types();

	MAIN_PRINT("Main: Initialize Globals");

	Thread::_main_thread_id = Thread::get_caller_id();

	globals = memnew(ProjectSettings);
	input_map = memnew(InputMap);

	register_core_settings(); //here globals is present

	translation_server = memnew(TranslationServer);
	performance = memnew(Performance);
	ClassDB::register_class<Performance>();
	engine->add_singleton(Engine::Singleton("Performance", performance));

	GLOBAL_DEF("debug/settings/crash_handler/message", String("Please include this when reporting the bug on https://github.com/godotengine/godot/issues"));

	MAIN_PRINT("Main: Parse CMDLine");

	OS::get_singleton()->_command_parser = memnew(CommandParser);
	ClassDB::register_class<CommandParser>();
	ClassDB::register_class<CommandFlag>();
	/* argument parsing and main creation */
	List<String> args;
	// Save cmd args
	for (int i = 0; i < argc; i++) {
		args.push_back(String::utf8(argv[i]));
	}
	OS::get_singleton()->set_cmdline(execpath, args);

	String video_driver;
	String audio_driver;
	String main_pack;
	bool quiet_stdout = false;
	bool upwards = false;
	bool skip_breakpoints = false;

	String project_path = ".";
	String debug_mode;
	String debug_host;
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
	if (!packed_data)
		packed_data = memnew(PackedData);

#ifdef MINIZIP_ENABLED

	//XXX: always get_singleton() == 0x0
	zip_packed_data = ZipArchive::get_singleton();
	//TODO: remove this temporary fix
	if (!zip_packed_data) {
		zip_packed_data = memnew(ZipArchive);
	}

	packed_data->add_pack_source(zip_packed_data);
#endif

	CommandParser *cli_parser = OS::get_singleton()->get_command_parser();
	cli_parser->init_engine_defaults();
	Error parse_err = cli_parser->parse();

	if (parse_err) {
		goto error;
	}

	if (cli_parser->is_argument_set("--help")) { // display help

		cli_parser->print_help();
		goto error;
	}
	if (cli_parser->is_argument_set("--version")) {

		cli_parser->print_version();
		goto error;
	}

	if (cli_parser->is_argument_set("--print-fps")) {

		print_fps = true;
	}
	if (cli_parser->is_argument_set("--windowed")) { // force windowed window

		init_windowed = true;
	}
	if (cli_parser->is_argument_set("--always-on-top")) { // force always-on-top window

		init_always_on_top = true;
	}
	if (cli_parser->is_argument_set("--profiling")) { // enable profiling

		use_debug_profiler = true;
	}
	if (cli_parser->is_argument_set("--low-dpi")) { // force low DPI (macOS only)

		force_lowdpi = true;
	}
	if (cli_parser->is_argument_set("--upwards")) { // scan folders upwards

		upwards = true;
	}
	if (cli_parser->is_argument_set("--video-driver")) { // force video driver

		video_driver = cli_parser->get_argument("--video-driver");
	}
	if (cli_parser->is_argument_set("--language")) { // language

		locale = cli_parser->get_argument("--language");
	}
	if (cli_parser->is_argument_set("--script")) {

		script = cli_parser->get_argument("--script");
	}
	if (cli_parser->is_argument_set("--check-only")) {

		check_only = true;
	}
	if (cli_parser->is_argument_set("--disable-render-loop")) {

		disable_render_loop = true;
	}
	if (cli_parser->is_argument_set("--test")) {

		test = cli_parser->get_argument("--test");
	}
	if (cli_parser->is_argument_set("--no-docbase")) {

		doc_base = !cli_parser->is_argument_set("--no-docbase");
	}
	if (cli_parser->is_argument_set("--video-driver")) {

		video_driver = cli_parser->get_argument("--video-driver");
	}
	if (cli_parser->is_argument_set("--audio-driver")) {

		audio_driver = cli_parser->get_argument("--audio-driver");
	}
	if (cli_parser->is_argument_set("--main-pack")) {

		main_pack = cli_parser->get_argument("--main-pack");
	}
	if (cli_parser->is_argument_set("--quiet")) {

		quiet_stdout = true;
	}
	if (cli_parser->is_argument_set("--upwards")) {

		upwards = true;
	}
	if (cli_parser->is_argument_set("--remote-fs")) {

		remotefs = cli_parser->get_argument("--remote-fs");
	}
	if (cli_parser->is_argument_set("--remote-fs-password")) {

		remotefs_pass = cli_parser->get_argument("--remote-fs-password");
	}
	if (cli_parser->is_argument_set("--resolution")) { // force resolution

		String vm = cli_parser->get_argument("--resolution");

		int w = vm.get_slice("x", 0).to_int();
		int h = vm.get_slice("x", 1).to_int();

		video_mode.width = w;
		video_mode.height = h;
		force_res = true;
	}
	if (cli_parser->is_argument_set("--position")) { // set window position

		String vm = cli_parser->get_argument("--position");

		int x = vm.get_slice(",", 0).to_int();
		int y = vm.get_slice(",", 1).to_int();

		init_custom_pos = Point2(x, y);
		init_use_custom_pos = true;
	}
	if (cli_parser->is_argument_set("--maximized")) { // force maximized window

		init_maximized = true;
		video_mode.maximized = true;
	}
	if (cli_parser->is_argument_set("--render-thread")) { // render thread mode

		String rtm_str = cli_parser->get_argument("--render-thread");
		if (rtm_str == "safe")
			rtm = OS::RENDER_THREAD_SAFE;
		else if (rtm_str == "unsafe")
			rtm = OS::RENDER_THREAD_UNSAFE;
		else if (rtm_str == "separate")
			rtm = OS::RENDER_SEPARATE_THREAD;
	}
	if (cli_parser->is_argument_set("--fullscreen")) { // force fullscreen

		//video_mode.fullscreen=false;
		init_fullscreen = true;
	}
#ifdef TOOLS_ENABLED
	if (cli_parser->is_argument_set("--editor")) { // starts editor
		editor = true;
	}
	if (cli_parser->is_argument_set("--project-manager")) { // starts project manager
		project_manager = true;
	}
#ifdef DEBUG_METHODS_ENABLED
	if (cli_parser->is_argument_set("--gdnative-generate-json-api")) {
		// Register as an editor instance to use the GLES2 fallback automatically on hardware that doesn't support the GLES3 backend
		editor = true;
	}
#endif

	if (cli_parser->is_argument_set("--build-solutions")) { // Build the scripting solution such C#

		auto_build_solutions = true;
		editor = true;
	}
#endif
	if (cli_parser->is_argument_set("--no-window")) { // disable window creation, Windows only

		OS::get_singleton()->set_no_window_mode(true);
	}
	if (cli_parser->is_argument_set("--verbose")) { // verbose output

		OS::get_singleton()->_verbose_stdout = true;
	}
	if (cli_parser->is_argument_set("--path")) { // set path of project to start or edit

		String p = cli_parser->get_argument("--path");
		if (OS::get_singleton()->set_cwd(p) != OK) {
			project_path = p; //use project_path instead
		}
	}
	if (cli_parser->is_argument_set("--quit")) { // Auto quit at the end of the first main loop iteration
		auto_quit = true;
	}
	if (cli_parser->has_project_defined()) {
		String path;
		String file = cli_parser->get_defined_project_file();

		int sep = MAX(file.find_last("/"), file.find_last("\\"));
		if (sep == -1)
			path = ".";
		else {
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
	}
	if (cli_parser->has_scene_defined()) {

		game_path = cli_parser->get_defined_project_file();
	}
	if (cli_parser->is_argument_set("--breakpoints")) { // add breakpoints

		String bplist = cli_parser->get_argument("--breakpoints");
		breakpoints = bplist.split(",");
	}
	if (cli_parser->is_argument_set("--frame-delay")) { // force frame delay

		frame_delay = cli_parser->get_argument("--frame-delay").to_int();
	}
	if (cli_parser->is_argument_set("--time-scale")) { // force time scale

		Engine::get_singleton()->set_time_scale(cli_parser->get_argument("--time-scale").to_double());
	}
	if (cli_parser->is_argument_set("--debug")) {
		debug_mode = "local";
	}
#ifdef DEBUG_ENABLED
	if (cli_parser->is_argument_set("--debug-collisions")) {
		debug_collisions = true;
	}
	if (cli_parser->is_argument_set("--debug-navigation")) {
		debug_navigation = true;
	}
#endif
	if (cli_parser->is_argument_set("--remote-debug")) {

		debug_mode = "remote";
		debug_host = cli_parser->get_argument("--remote-debug");
	}
	if (cli_parser->is_argument_set("--allow_focus_steal_pid")) {

		allow_focus_steal_pid = cli_parser->get_argument("--allow_focus_steal_pid").to_int64();
	}
	if (cli_parser->is_argument_set("--fixed-fps")) {

		fixed_fps = cli_parser->get_argument("--fixed-fps").to_int();
	}
	if (cli_parser->is_argument_set("--disable-crash-handler")) {

		OS::get_singleton()->disable_crash_handler();
	}
	if (cli_parser->is_argument_set("--skip-breakpoints")) {
		skip_breakpoints = true;
	}
#ifdef TOOLS_ENABLED
	if (cli_parser->is_argument_set("--doctool")) {
		doc_tool = cli_parser->get_argument("--doctool");
	}
	if (cli_parser->is_argument_set("--export")) {
		editor = true; //needs editor
		_export_preset = cli_parser->get_argument("--export");
		game_path = cli_parser->get_defined_project_file();
	}
	if (cli_parser->is_argument_set("--export-debug")) {
		editor = true; //needs editor
		_export_preset = cli_parser->get_argument("--export-debug");
		export_debug = true;
		game_path = cli_parser->get_defined_project_file();
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
		String error_msg = "Error: Could not load game data at path '" + project_path + "'. Is the .pck file missing?\n";
		OS::get_singleton()->print("%s", error_msg.ascii().get_data());
		OS::get_singleton()->alert(error_msg);

		goto error;
#endif
	}

	GLOBAL_DEF("memory/limits/multithreaded_server/rid_pool_prealloc", 60);
	ProjectSettings::get_singleton()->set_custom_property_info("memory/limits/multithreaded_server/rid_pool_prealloc", PropertyInfo(Variant::INT, "memory/limits/multithreaded_server/rid_pool_prealloc", PROPERTY_HINT_RANGE, "0,500,1")); // No negative and limit to 500 due to crashes
	GLOBAL_DEF("network/limits/debugger_stdout/max_chars_per_second", 2048);
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger_stdout/max_chars_per_second", PropertyInfo(Variant::INT, "network/limits/debugger_stdout/max_chars_per_second", PROPERTY_HINT_RANGE, "0, 4096, 1, or_greater"));
	GLOBAL_DEF("network/limits/debugger_stdout/max_messages_per_frame", 10);
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger_stdout/max_messages_per_frame", PropertyInfo(Variant::INT, "network/limits/debugger_stdout/max_messages_per_frame", PROPERTY_HINT_RANGE, "0, 20, 1, or_greater"));
	GLOBAL_DEF("network/limits/debugger_stdout/max_errors_per_second", 100);
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger_stdout/max_errors_per_second", PropertyInfo(Variant::INT, "network/limits/debugger_stdout/max_errors_per_second", PROPERTY_HINT_RANGE, "0, 200, 1, or_greater"));
	GLOBAL_DEF("network/limits/debugger_stdout/max_warnings_per_second", 100);
	ProjectSettings::get_singleton()->set_custom_property_info("network/limits/debugger_stdout/max_warnings_per_second", PropertyInfo(Variant::INT, "network/limits/debugger_stdout/max_warnings_per_second", PROPERTY_HINT_RANGE, "0, 200, 1, or_greater"));

	if (debug_mode == "remote") {

		ScriptDebuggerRemote *sdr = memnew(ScriptDebuggerRemote);
		uint16_t debug_port = 6007;
		if (debug_host.find(":") != -1) {
			int sep_pos = debug_host.find_last(":");
			debug_port = debug_host.substr(sep_pos + 1, debug_host.length()).to_int();
			debug_host = debug_host.substr(0, sep_pos);
		}
		Error derr = sdr->connect_to_host(debug_host, debug_port);

		sdr->set_skip_breakpoints(skip_breakpoints);

		if (derr != OK) {
			memdelete(sdr);
		} else {
			script_debugger = sdr;
		}
	} else if (debug_mode == "local") {

		script_debugger = memnew(ScriptDebuggerLocal);
		OS::get_singleton()->initialize_debugging();
	}
	if (script_debugger) {
		//there is a debugger, parse breakpoints

		for (int i = 0; i < breakpoints.size(); i++) {

			String bp = breakpoints[i];
			int sp = bp.find_last(":");
			ERR_CONTINUE_MSG(sp == -1, "Invalid breakpoint: '" + bp + "', expected file:line format.");

			script_debugger->insert_breakpoint(bp.substr(sp + 1, bp.length()).to_int(), bp.substr(0, sp));
		}
	}

#ifdef TOOLS_ENABLED
	if (editor) {
		packed_data->set_disabled(true);
		globals->set_disable_feature_overrides(true);
	}

#endif

	GLOBAL_DEF("logging/file_logging/enable_file_logging", false);
	GLOBAL_DEF("logging/file_logging/log_path", "user://logs/log.txt");
	GLOBAL_DEF("logging/file_logging/max_log_files", 10);
	ProjectSettings::get_singleton()->set_custom_property_info("logging/file_logging/max_log_files", PropertyInfo(Variant::INT, "logging/file_logging/max_log_files", PROPERTY_HINT_RANGE, "0,20,1,or_greater")); //no negative numbers
	if (FileAccess::get_create_func(FileAccess::ACCESS_USERDATA) && GLOBAL_GET("logging/file_logging/enable_file_logging")) {
		String base_path = GLOBAL_GET("logging/file_logging/log_path");
		int max_files = GLOBAL_GET("logging/file_logging/max_log_files");
		OS::get_singleton()->add_logger(memnew(RotatedFileLogger(base_path, max_files)));
	}

#ifdef TOOLS_ENABLED
	if (editor) {
		Engine::get_singleton()->set_editor_hint(true);
		if (!init_windowed) {
			init_maximized = true;
			video_mode.maximized = true;
		}
	}

	if (!project_manager) {
		// Determine if the project manager should be requested
		project_manager = !found_project && game_path.empty();
	}
#endif

	if (String(GLOBAL_DEF("application/run/main_scene", "")) == "" && game_path == "") {
#ifdef TOOLS_ENABLED
		if (!editor && !project_manager) {
#endif
			OS::get_singleton()->print("Error: Can't run project: no main scene defined.\n");
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
		input_map->load_from_globals(); //keys for game
	}

	if (bool(ProjectSettings::get_singleton()->get("application/run/disable_stdout"))) {
		quiet_stdout = true;
	}
	if (bool(ProjectSettings::get_singleton()->get("application/run/disable_stderr"))) {
		_print_error_enabled = false;
	};

	if (quiet_stdout)
		_print_line_enabled = false;

	GLOBAL_DEF("rendering/quality/driver/driver_name", "GLES3");
	ProjectSettings::get_singleton()->set_custom_property_info("rendering/quality/driver/driver_name", PropertyInfo(Variant::STRING, "rendering/quality/driver/driver_name", PROPERTY_HINT_ENUM, "GLES2,GLES3"));
	if (video_driver == "") {
		video_driver = GLOBAL_GET("rendering/quality/driver/driver_name");
	}

	GLOBAL_DEF("rendering/quality/driver/fallback_to_gles2", false);

	// Assigning here even though it's GLES2-specific, to be sure that it appears in docs
	GLOBAL_DEF("rendering/quality/2d/gles2_use_nvidia_rect_flicker_workaround", false);

	GLOBAL_DEF("display/window/size/width", 1024);
	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/width", PropertyInfo(Variant::INT, "display/window/size/width", PROPERTY_HINT_RANGE, "0,7680,or_greater")); // 8K resolution
	GLOBAL_DEF("display/window/size/height", 600);
	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/height", PropertyInfo(Variant::INT, "display/window/size/height", PROPERTY_HINT_RANGE, "0,4320,or_greater")); // 8K resolution
	GLOBAL_DEF("display/window/size/resizable", true);
	GLOBAL_DEF("display/window/size/borderless", false);
	GLOBAL_DEF("display/window/size/fullscreen", false);
	GLOBAL_DEF("display/window/size/always_on_top", false);
	GLOBAL_DEF("display/window/size/test_width", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/test_width", PropertyInfo(Variant::INT, "display/window/size/test_width", PROPERTY_HINT_RANGE, "0,7680,or_greater")); // 8K resolution
	GLOBAL_DEF("display/window/size/test_height", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("display/window/size/test_height", PropertyInfo(Variant::INT, "display/window/size/test_height", PROPERTY_HINT_RANGE, "0,4320,or_greater")); // 8K resolution

	if (use_custom_res) {

		if (!force_res) {
			video_mode.width = GLOBAL_GET("display/window/size/width");
			video_mode.height = GLOBAL_GET("display/window/size/height");

			if (globals->has_setting("display/window/size/test_width") && globals->has_setting("display/window/size/test_height")) {

				int tw = globals->get("display/window/size/test_width");
				if (tw > 0) {
					video_mode.width = tw;
				}
				int th = globals->get("display/window/size/test_height");
				if (th > 0) {
					video_mode.height = th;
				}
			}
		}

		video_mode.resizable = GLOBAL_GET("display/window/size/resizable");
		video_mode.borderless_window = GLOBAL_GET("display/window/size/borderless");
		video_mode.fullscreen = GLOBAL_GET("display/window/size/fullscreen");
		video_mode.always_on_top = GLOBAL_GET("display/window/size/always_on_top");
	}

	if (!force_lowdpi) {
		OS::get_singleton()->_allow_hidpi = GLOBAL_DEF("display/window/dpi/allow_hidpi", false);
	}

	video_mode.use_vsync = GLOBAL_DEF_RST("display/window/vsync/use_vsync", true);
	OS::get_singleton()->_use_vsync = video_mode.use_vsync;

	if (cli_parser->is_argument_set("--enable-vsync-via-compositor")) {
		video_mode.vsync_via_compositor = true;
	} else if (cli_parser->is_argument_set("--disable-vsync-via-compositor")) {
		video_mode.vsync_via_compositor = false;
	} else {
		// If one of the command line options to enable/disable vsync via the
		// window compositor ("--enable-vsync-via-compositor" or
		// "--disable-vsync-via-compositor") was present then it overrides the
		// project setting.
		video_mode.vsync_via_compositor = GLOBAL_DEF("display/window/vsync/vsync_via_compositor", false);
	}
	OS::get_singleton()->_vsync_via_compositor = video_mode.vsync_via_compositor;

	OS::get_singleton()->_allow_layered = GLOBAL_DEF("display/window/per_pixel_transparency/allowed", false);
	video_mode.layered = GLOBAL_DEF("display/window/per_pixel_transparency/enabled", false);

	GLOBAL_DEF("rendering/quality/intended_usage/framebuffer_allocation", 2);
	GLOBAL_DEF("rendering/quality/intended_usage/framebuffer_allocation.mobile", 3);

	if (editor || project_manager) {
		// The editor and project manager always detect and use hiDPI if needed
		OS::get_singleton()->_allow_hidpi = true;
		OS::get_singleton()->_allow_layered = false;
	}

	Engine::get_singleton()->_pixel_snap = GLOBAL_DEF("rendering/quality/2d/use_pixel_snap", false);
	OS::get_singleton()->_keep_screen_on = GLOBAL_DEF("display/window/energy_saving/keep_screen_on", true);
	if (rtm == -1) {
		rtm = GLOBAL_DEF("rendering/threads/thread_model", OS::RENDER_THREAD_SAFE);
	}

	if (rtm >= 0 && rtm < 3) {
		if (editor) {
			rtm = OS::RENDER_THREAD_SAFE;
		}
		OS::get_singleton()->_render_thread_mode = OS::RenderThreadMode(rtm);
	}

	/* Determine audio and video drivers */

	for (int i = 0; i < OS::get_singleton()->get_video_driver_count(); i++) {

		if (video_driver == OS::get_singleton()->get_video_driver_name(i)) {

			video_driver_idx = i;
			break;
		}
	}

	if (video_driver_idx < 0) {

		//OS::get_singleton()->alert("Invalid Video Driver: " + video_driver);
		video_driver_idx = 0;
		//goto error;
	}

	if (audio_driver == "") { // specified in project.godot
		audio_driver = GLOBAL_DEF_RST("audio/driver", OS::get_singleton()->get_audio_driver_name(0));
	}

	for (int i = 0; i < OS::get_singleton()->get_audio_driver_count(); i++) {

		if (audio_driver == OS::get_singleton()->get_audio_driver_name(i)) {

			audio_driver_idx = i;
			break;
		}
	}

	if (audio_driver_idx < 0) {

		OS::get_singleton()->alert("Invalid Audio Driver: " + audio_driver);
		audio_driver_idx = 0;
		//goto error;
	}

	{
		String orientation = GLOBAL_DEF("display/window/handheld/orientation", "landscape");

		if (orientation == "portrait")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_PORTRAIT);
		else if (orientation == "reverse_landscape")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_REVERSE_LANDSCAPE);
		else if (orientation == "reverse_portrait")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_REVERSE_PORTRAIT);
		else if (orientation == "sensor_landscape")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_SENSOR_LANDSCAPE);
		else if (orientation == "sensor_portrait")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_SENSOR_PORTRAIT);
		else if (orientation == "sensor")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_SENSOR);
		else
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_LANDSCAPE);
	}

	Engine::get_singleton()->set_iterations_per_second(GLOBAL_DEF("physics/common/physics_fps", 60));
	ProjectSettings::get_singleton()->set_custom_property_info("physics/common/physics_fps", PropertyInfo(Variant::INT, "physics/common/physics_fps", PROPERTY_HINT_RANGE, "1,120,1,or_greater"));
	Engine::get_singleton()->set_physics_jitter_fix(GLOBAL_DEF("physics/common/physics_jitter_fix", 0.5));
	Engine::get_singleton()->set_target_fps(GLOBAL_DEF("debug/settings/fps/force_fps", 0));
	ProjectSettings::get_singleton()->set_custom_property_info("debug/settings/fps/force_fps", PropertyInfo(Variant::INT, "debug/settings/fps/force_fps", PROPERTY_HINT_RANGE, "0,120,1,or_greater"));

	GLOBAL_DEF("debug/settings/stdout/print_fps", false);

	if (!OS::get_singleton()->_verbose_stdout) //overridden
		OS::get_singleton()->_verbose_stdout = GLOBAL_DEF("debug/settings/stdout/verbose_stdout", false);

	if (frame_delay == 0) {
		frame_delay = GLOBAL_DEF("application/run/frame_delay_msec", 0);
		ProjectSettings::get_singleton()->set_custom_property_info("application/run/frame_delay_msec", PropertyInfo(Variant::INT, "application/run/frame_delay_msec", PROPERTY_HINT_RANGE, "0,100,1,or_greater")); // No negative numbers
	}

	OS::get_singleton()->set_low_processor_usage_mode(GLOBAL_DEF("application/run/low_processor_mode", false));
	OS::get_singleton()->set_low_processor_usage_mode_sleep_usec(GLOBAL_DEF("application/run/low_processor_mode_sleep_usec", 6900)); // Roughly 144 FPS
	ProjectSettings::get_singleton()->set_custom_property_info("application/run/low_processor_mode_sleep_usec", PropertyInfo(Variant::INT, "application/run/low_processor_mode_sleep_usec", PROPERTY_HINT_RANGE, "0,33200,1,or_greater")); // No negative numbers

	Engine::get_singleton()->set_frame_delay(frame_delay);

	message_queue = memnew(MessageQueue);

	if (p_second_phase)
		return setup2();

	return OK;

error:

	video_driver = "";
	audio_driver = "";
	project_path = "";

	if (performance)
		memdelete(performance);
	if (input_map)
		memdelete(input_map);
	if (translation_server)
		memdelete(translation_server);
	if (globals)
		memdelete(globals);
	if (engine)
		memdelete(engine);
	if (script_debugger)
		memdelete(script_debugger);
	if (packed_data)
		memdelete(packed_data);
	if (file_access_network_client)
		memdelete(file_access_network_client);

	OS::get_singleton()->delete_command_parser();

	unregister_core_driver_types();
	unregister_core_types();

	if (message_queue)
		memdelete(message_queue);
	OS::get_singleton()->finalize_core();
	locale = String();

	return ERR_INVALID_PARAMETER;
}

Error Main::setup2(Thread::ID p_main_tid_override) {

	// Print engine name and version
	print_line(String(VERSION_NAME) + " v" + OS::get_singleton()->get_command_parser()->get_version() + " - " + String(VERSION_WEBSITE));

	if (p_main_tid_override) {
		Thread::_main_thread_id = p_main_tid_override;
	}

	Error err = OS::get_singleton()->initialize(video_mode, video_driver_idx, audio_driver_idx);
	if (err != OK) {
		return err;
	}

	print_line(" "); //add a blank line for readability

	if (init_use_custom_pos) {
		OS::get_singleton()->set_window_position(init_custom_pos);
	}

	// right moment to create and initialize the audio server

	audio_server = memnew(AudioServer);
	audio_server->init();

	// also init our arvr_server from here
	arvr_server = memnew(ARVRServer);

	register_core_singletons();

	MAIN_PRINT("Main: Setup Logo");

#ifdef JAVASCRIPT_ENABLED
	bool show_logo = false;
#else
	bool show_logo = true;
#endif

	if (init_screen != -1) {
		OS::get_singleton()->set_current_screen(init_screen);
	}
	if (init_windowed) {
		//do none..
	} else if (init_maximized) {
		OS::get_singleton()->set_window_maximized(true);
	} else if (init_fullscreen) {
		OS::get_singleton()->set_window_fullscreen(true);
	}
	if (init_always_on_top) {
		OS::get_singleton()->set_window_always_on_top(true);
	}

	if (allow_focus_steal_pid) {
		OS::get_singleton()->enable_for_stealing_focus(allow_focus_steal_pid);
	}

	register_server_types();

	MAIN_PRINT("Main: Load Remaps");

	Color clear = GLOBAL_DEF("rendering/environment/default_clear_color", Color(0.3, 0.3, 0.3));
	VisualServer::get_singleton()->set_default_clear_color(clear);

	if (show_logo) { //boot logo!
		String boot_logo_path = GLOBAL_DEF("application/boot_splash/image", String());
		bool boot_logo_scale = GLOBAL_DEF("application/boot_splash/fullsize", true);
		bool boot_logo_filter = GLOBAL_DEF("application/boot_splash/use_filter", true);
		ProjectSettings::get_singleton()->set_custom_property_info("application/boot_splash/image", PropertyInfo(Variant::STRING, "application/boot_splash/image", PROPERTY_HINT_FILE, "*.png"));

		Ref<Image> boot_logo;

		boot_logo_path = boot_logo_path.strip_edges();

		if (boot_logo_path != String()) {
			boot_logo.instance();
			Error load_err = ImageLoader::load_image(boot_logo_path, boot_logo);
			if (load_err)
				ERR_PRINTS("Non-existing or invalid boot splash at '" + boot_logo_path + "'. Loading default splash.");
		}

		Color boot_bg_color = GLOBAL_DEF("application/boot_splash/bg_color", boot_splash_bg_color);
		if (boot_logo.is_valid()) {
			OS::get_singleton()->_msec_splash = OS::get_singleton()->get_ticks_msec();
			VisualServer::get_singleton()->set_boot_image(boot_logo, boot_bg_color, boot_logo_scale, boot_logo_filter);

		} else {
#ifndef NO_DEFAULT_BOOT_LOGO
			MAIN_PRINT("Main: Create bootsplash");
#if defined(TOOLS_ENABLED) && !defined(NO_EDITOR_SPLASH)
			Ref<Image> splash = (editor || project_manager) ? memnew(Image(boot_splash_editor_png)) : memnew(Image(boot_splash_png));
#else
			Ref<Image> splash = memnew(Image(boot_splash_png));
#endif

			MAIN_PRINT("Main: ClearColor");
			VisualServer::get_singleton()->set_default_clear_color(boot_bg_color);
			MAIN_PRINT("Main: Image");
			VisualServer::get_singleton()->set_boot_image(splash, boot_bg_color, false);
#endif
		}

#ifdef TOOLS_ENABLED
		Ref<Image> icon = memnew(Image(app_icon_png));
		OS::get_singleton()->set_icon(icon);
#endif
	}

	MAIN_PRINT("Main: DCC");
	VisualServer::get_singleton()->set_default_clear_color(GLOBAL_DEF("rendering/environment/default_clear_color", Color(0.3, 0.3, 0.3)));
	MAIN_PRINT("Main: END");

	GLOBAL_DEF("application/config/icon", String());
	ProjectSettings::get_singleton()->set_custom_property_info("application/config/icon", PropertyInfo(Variant::STRING, "application/config/icon", PROPERTY_HINT_FILE, "*.png,*.webp"));

	GLOBAL_DEF("application/config/macos_native_icon", String());
	ProjectSettings::get_singleton()->set_custom_property_info("application/config/macos_native_icon", PropertyInfo(Variant::STRING, "application/config/macos_native_icon", PROPERTY_HINT_FILE, "*.icns"));

	GLOBAL_DEF("application/config/windows_native_icon", String());
	ProjectSettings::get_singleton()->set_custom_property_info("application/config/windows_native_icon", PropertyInfo(Variant::STRING, "application/config/windows_native_icon", PROPERTY_HINT_FILE, "*.ico"));

	InputDefault *id = Object::cast_to<InputDefault>(Input::get_singleton());
	if (id) {
		if (bool(GLOBAL_DEF("input_devices/pointing/emulate_touch_from_mouse", false)) && !(editor || project_manager)) {
			if (!OS::get_singleton()->has_touchscreen_ui_hint()) {
				//only if no touchscreen ui hint, set emulation
				id->set_emulate_touch_from_mouse(true);
			}
		}

		id->set_emulate_mouse_from_touch(bool(GLOBAL_DEF("input_devices/pointing/emulate_mouse_from_touch", true)));
	}

	MAIN_PRINT("Main: Load Remaps");

	MAIN_PRINT("Main: Load Scene Types");

	register_scene_types();

	GLOBAL_DEF("display/mouse_cursor/custom_image", String());
	GLOBAL_DEF("display/mouse_cursor/custom_image_hotspot", Vector2());
	GLOBAL_DEF("display/mouse_cursor/tooltip_position_offset", Point2(10, 10));
	ProjectSettings::get_singleton()->set_custom_property_info("display/mouse_cursor/custom_image", PropertyInfo(Variant::STRING, "display/mouse_cursor/custom_image", PROPERTY_HINT_FILE, "*.png,*.webp"));

	if (String(ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image")) != String()) {

		Ref<Texture> cursor = ResourceLoader::load(ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image"));
		if (cursor.is_valid()) {
			Vector2 hotspot = ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image_hotspot");
			Input::get_singleton()->set_custom_mouse_cursor(cursor, Input::CURSOR_ARROW, hotspot);
		}
	}
#ifdef TOOLS_ENABLED
	ClassDB::set_current_api(ClassDB::API_EDITOR);
	EditorNode::register_editor_types();

	ClassDB::set_current_api(ClassDB::API_CORE);

#endif

	MAIN_PRINT("Main: Load Modules, Physics, Drivers, Scripts");

	register_platform_apis();
	register_module_types();

	camera_server = CameraServer::create();

	initialize_physics();
	register_server_singletons();

	register_driver_types();

	// This loads global classes, so it must happen before custom loaders and savers are registered
	ScriptServer::init_languages();

	MAIN_PRINT("Main: Load Translations");

	translation_server->setup(); //register translations, load them, etc.
	if (locale != "") {

		translation_server->set_locale(locale);
	}
	translation_server->load_translations();
	ResourceLoader::load_translation_remaps(); //load remaps for resources

	ResourceLoader::load_path_remaps();

	audio_server->load_default_bus_layout();

	if (use_debug_profiler && script_debugger) {
		script_debugger->profiling_start();
	}
	_start_success = true;
	locale = String();

	ClassDB::set_current_api(ClassDB::API_NONE); //no more api is registered at this point

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

	main_timer_sync.init(OS::get_singleton()->get_ticks_usec());

	String main_loop_type;
#ifdef TOOLS_ENABLED
	if (doc_tool != "") {

		Engine::get_singleton()->set_editor_hint(true); // Needed to instance editor-only classes for their default values

		{
			DirAccessRef da = DirAccess::open(doc_tool);
			ERR_FAIL_COND_V_MSG(!da, false, "Argument supplied to --doctool must be a base Godot build directory.");
		}
		DocData doc;
		doc.generate(doc_base);

		DocData docsrc;
		Map<String, String> doc_data_classes;
		Set<String> checked_paths;
		print_line("Loading docs...");

		for (int i = 0; i < _doc_data_class_path_count; i++) {
			String path = doc_tool.plus_file(_doc_data_class_paths[i].path);
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

		String index_path = doc_tool.plus_file("doc/classes");
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
			DocData::erase_classes(E->get());
		}

		print_line("Generating new docs...");
		doc.save_classes(index_path, doc_data_classes);

		return false;
	}

#endif

	if (_export_preset != "") {
		if (game_path == "") {
			String err = "Command line param ";
			err += export_debug ? "--export-debug" : "--export";
			err += " passed but no destination path given.\n";
			err += "Please specify the binary's file path to export to. Aborting export.";
			ERR_PRINT(err.utf8().get_data());
			return false;
		}
	}

	if (script == "" && game_path == "" && String(GLOBAL_DEF("application/run/main_scene", "")) != "") {
		game_path = GLOBAL_DEF("application/run/main_scene", "");
	}

	MainLoop *main_loop = NULL;
	if (editor) {
		main_loop = memnew(SceneTree);
	};

	if (test != "") {
#ifdef TOOLS_ENABLED
		main_loop = test_main(test, OS::get_singleton()->get_cmdline_args());

		if (!main_loop)
			return false;

#endif

	} else if (script != "") {

		Ref<Script> script_res = ResourceLoader::load(script);
		ERR_FAIL_COND_V_MSG(script_res.is_null(), false, "Can't load script: " + script);

		if (check_only) {
			if (!script_res->is_valid()) {
				OS::get_singleton()->set_exit_code(1);
			}
			return false;
		}

		if (script_res->can_instance() /*&& script_res->inherits_from("SceneTreeScripted")*/) {

			StringName instance_type = script_res->get_instance_base_type();
			Object *obj = ClassDB::instance(instance_type);
			MainLoop *script_loop = Object::cast_to<MainLoop>(obj);
			if (!script_loop) {
				if (obj)
					memdelete(obj);
				ERR_FAIL_V_MSG(false, "Can't load script '" + script + "', it does not inherit from a MainLoop type.");
			}

			script_loop->set_init_script(script_res);
			main_loop = script_loop;
		} else {

			return false;
		}

	} else {
		main_loop_type = GLOBAL_DEF("application/run/main_loop_type", "");
	}

	if (!main_loop && main_loop_type == "")
		main_loop_type = "SceneTree";

	if (!main_loop) {
		if (!ClassDB::class_exists(main_loop_type)) {
			OS::get_singleton()->alert("Error: MainLoop type doesn't exist: " + main_loop_type);
			return false;
		} else {

			Object *ml = ClassDB::instance(main_loop_type);
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

		ResourceLoader::add_custom_loaders();
		ResourceSaver::add_custom_savers();

		if (!project_manager && !editor) { // game
			if (game_path != "" || script != "") {
				if (script_debugger && script_debugger->is_remote()) {
					ScriptDebuggerRemote *remote_debugger = static_cast<ScriptDebuggerRemote *>(script_debugger);

					remote_debugger->set_scene_tree(sml);
				}

				//autoload
				List<PropertyInfo> props;
				ProjectSettings::get_singleton()->get_property_list(&props);

				//first pass, add the constants so they exist before any script is loaded
				for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {

					String s = E->get().name;
					if (!s.begins_with("autoload/"))
						continue;
					String name = s.get_slicec('/', 1);
					String path = ProjectSettings::get_singleton()->get(s);
					bool global_var = false;
					if (path.begins_with("*")) {
						global_var = true;
					}

					if (global_var) {
						for (int i = 0; i < ScriptServer::get_language_count(); i++) {
							ScriptServer::get_language(i)->add_global_constant(name, Variant());
						}
					}
				}

				//second pass, load into global constants
				List<Node *> to_add;
				for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {

					String s = E->get().name;
					if (!s.begins_with("autoload/"))
						continue;
					String name = s.get_slicec('/', 1);
					String path = ProjectSettings::get_singleton()->get(s);
					bool global_var = false;
					if (path.begins_with("*")) {
						global_var = true;
						path = path.substr(1, path.length() - 1);
					}

					RES res = ResourceLoader::load(path);
					ERR_CONTINUE_MSG(res.is_null(), "Can't autoload: " + path);
					Node *n = NULL;
					if (res->is_class("PackedScene")) {
						Ref<PackedScene> ps = res;
						n = ps->instance();
					} else if (res->is_class("Script")) {
						Ref<Script> script_res = res;
						StringName ibt = script_res->get_instance_base_type();
						bool valid_type = ClassDB::is_parent_class(ibt, "Node");
						ERR_CONTINUE_MSG(!valid_type, "Script does not inherit a Node: " + path);

						Object *obj = ClassDB::instance(ibt);

						ERR_CONTINUE_MSG(obj == NULL, "Cannot instance script for autoload, expected 'Node' inheritance, got: " + String(ibt));

						n = Object::cast_to<Node>(obj);
						n->set_script(script_res.get_ref_ptr());
					}

					ERR_CONTINUE_MSG(!n, "Path in autoload not a node or script: " + path);
					n->set_name(name);

					//defer so references are all valid on _ready()
					to_add.push_back(n);

					if (global_var) {
						for (int i = 0; i < ScriptServer::get_language_count(); i++) {
							ScriptServer::get_language(i)->add_global_constant(name, n);
						}
					}
				}

				for (List<Node *>::Element *E = to_add.front(); E; E = E->next()) {

					sml->get_root()->add_child(E->get());
				}
			}
		}

#ifdef TOOLS_ENABLED

		EditorNode *editor_node = NULL;
		if (editor) {

			editor_node = memnew(EditorNode);
			sml->get_root()->add_child(editor_node);

			//root_node->set_editor(editor);
			//startup editor

			if (_export_preset != "") {

				editor_node->export_preset(_export_preset, game_path, export_debug, "", true);
				game_path = ""; //no load anything
			}
		}
#endif

		if (!editor && !project_manager) {
			//standard helpers that can be changed from main config

			String stretch_mode = GLOBAL_DEF("display/window/stretch/mode", "disabled");
			String stretch_aspect = GLOBAL_DEF("display/window/stretch/aspect", "ignore");
			Size2i stretch_size = Size2(GLOBAL_DEF("display/window/size/width", 0), GLOBAL_DEF("display/window/size/height", 0));
			real_t stretch_shrink = GLOBAL_DEF("display/window/stretch/shrink", 1.0);

			SceneTree::StretchMode sml_sm = SceneTree::STRETCH_MODE_DISABLED;
			if (stretch_mode == "2d")
				sml_sm = SceneTree::STRETCH_MODE_2D;
			else if (stretch_mode == "viewport")
				sml_sm = SceneTree::STRETCH_MODE_VIEWPORT;

			SceneTree::StretchAspect sml_aspect = SceneTree::STRETCH_ASPECT_IGNORE;
			if (stretch_aspect == "keep")
				sml_aspect = SceneTree::STRETCH_ASPECT_KEEP;
			else if (stretch_aspect == "keep_width")
				sml_aspect = SceneTree::STRETCH_ASPECT_KEEP_WIDTH;
			else if (stretch_aspect == "keep_height")
				sml_aspect = SceneTree::STRETCH_ASPECT_KEEP_HEIGHT;
			else if (stretch_aspect == "expand")
				sml_aspect = SceneTree::STRETCH_ASPECT_EXPAND;

			sml->set_screen_stretch(sml_sm, sml_aspect, stretch_size, stretch_shrink);

			sml->set_auto_accept_quit(GLOBAL_DEF("application/config/auto_accept_quit", true));
			sml->set_quit_on_go_back(GLOBAL_DEF("application/config/quit_on_go_back", true));
			String appname = ProjectSettings::get_singleton()->get("application/config/name");
			appname = TranslationServer::get_singleton()->translate(appname);
			OS::get_singleton()->set_window_title(appname);

			int shadow_atlas_size = GLOBAL_GET("rendering/quality/shadow_atlas/size");
			int shadow_atlas_q0_subdiv = GLOBAL_GET("rendering/quality/shadow_atlas/quadrant_0_subdiv");
			int shadow_atlas_q1_subdiv = GLOBAL_GET("rendering/quality/shadow_atlas/quadrant_1_subdiv");
			int shadow_atlas_q2_subdiv = GLOBAL_GET("rendering/quality/shadow_atlas/quadrant_2_subdiv");
			int shadow_atlas_q3_subdiv = GLOBAL_GET("rendering/quality/shadow_atlas/quadrant_3_subdiv");

			sml->get_root()->set_shadow_atlas_size(shadow_atlas_size);
			sml->get_root()->set_shadow_atlas_quadrant_subdiv(0, Viewport::ShadowAtlasQuadrantSubdiv(shadow_atlas_q0_subdiv));
			sml->get_root()->set_shadow_atlas_quadrant_subdiv(1, Viewport::ShadowAtlasQuadrantSubdiv(shadow_atlas_q1_subdiv));
			sml->get_root()->set_shadow_atlas_quadrant_subdiv(2, Viewport::ShadowAtlasQuadrantSubdiv(shadow_atlas_q2_subdiv));
			sml->get_root()->set_shadow_atlas_quadrant_subdiv(3, Viewport::ShadowAtlasQuadrantSubdiv(shadow_atlas_q3_subdiv));
			Viewport::Usage usage = Viewport::Usage(int(GLOBAL_GET("rendering/quality/intended_usage/framebuffer_allocation")));
			sml->get_root()->set_usage(usage);

			bool snap_controls = GLOBAL_DEF("gui/common/snap_controls_to_pixels", true);
			sml->get_root()->set_snap_controls_to_pixels(snap_controls);

			bool font_oversampling = GLOBAL_DEF("rendering/quality/dynamic_fonts/use_oversampling", true);
			sml->set_use_font_oversampling(font_oversampling);

		} else {

			GLOBAL_DEF("display/window/stretch/mode", "disabled");
			ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/mode", PropertyInfo(Variant::STRING, "display/window/stretch/mode", PROPERTY_HINT_ENUM, "disabled,2d,viewport"));
			GLOBAL_DEF("display/window/stretch/aspect", "ignore");
			ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/aspect", PropertyInfo(Variant::STRING, "display/window/stretch/aspect", PROPERTY_HINT_ENUM, "ignore,keep,keep_width,keep_height,expand"));
			GLOBAL_DEF("display/window/stretch/shrink", 1.0);
			ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/shrink", PropertyInfo(Variant::REAL, "display/window/stretch/shrink", PROPERTY_HINT_RANGE, "1.0,8.0,0.1"));
			sml->set_auto_accept_quit(GLOBAL_DEF("application/config/auto_accept_quit", true));
			sml->set_quit_on_go_back(GLOBAL_DEF("application/config/quit_on_go_back", true));
			GLOBAL_DEF("gui/common/snap_controls_to_pixels", true);
			GLOBAL_DEF("rendering/quality/dynamic_fonts/use_oversampling", true);
		}

		String local_game_path;
		if (game_path != "" && !project_manager) {

			local_game_path = game_path.replace("\\", "/");

			if (!local_game_path.begins_with("res://")) {
				bool absolute = (local_game_path.size() > 1) && (local_game_path[0] == '/' || local_game_path[1] == ':');

				if (!absolute) {

					if (ProjectSettings::get_singleton()->is_using_datapack()) {

						local_game_path = "res://" + local_game_path;

					} else {
						int sep = local_game_path.find_last("/");

						if (sep == -1) {
							DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
							local_game_path = da->get_current_dir().plus_file(local_game_path);
							memdelete(da);
						} else {

							DirAccess *da = DirAccess::open(local_game_path.substr(0, sep));
							if (da) {
								local_game_path = da->get_current_dir().plus_file(local_game_path.substr(sep + 1, local_game_path.length()));
								memdelete(da);
							}
						}
					}
				}
			}

			local_game_path = ProjectSettings::get_singleton()->localize_path(local_game_path);

#ifdef TOOLS_ENABLED
			if (editor) {

				if (game_path != GLOBAL_GET("application/run/main_scene") || !editor_node->has_scenes_in_session()) {
					Error serr = editor_node->load_scene(local_game_path);
					if (serr != OK)
						ERR_PRINT("Failed to load scene");
				}
				OS::get_singleton()->set_context(OS::CONTEXT_EDITOR);
			}
#endif
			if (!editor) {
				OS::get_singleton()->set_context(OS::CONTEXT_ENGINE);
			}
		}

		if (!project_manager && !editor) { // game

			// Load SSL Certificates from Project Settings (or builtin).
			Crypto::load_default_certificates(GLOBAL_DEF("network/ssl/certificates", ""));

			if (game_path != "") {
				Node *scene = NULL;
				Ref<PackedScene> scenedata = ResourceLoader::load(local_game_path);
				if (scenedata.is_valid())
					scene = scenedata->instance();

				ERR_FAIL_COND_V_MSG(!scene, false, "Failed loading scene: " + local_game_path);
				sml->add_current_scene(scene);

#ifdef OSX_ENABLED
				String mac_iconpath = GLOBAL_DEF("application/config/macos_native_icon", "Variant()");
				if (mac_iconpath != "") {
					OS::get_singleton()->set_native_icon(mac_iconpath);
					hasicon = true;
				}
#endif

#ifdef WINDOWS_ENABLED
				String win_iconpath = GLOBAL_DEF("application/config/windows_native_icon", "Variant()");
				if (win_iconpath != "") {
					OS::get_singleton()->set_native_icon(win_iconpath);
					hasicon = true;
				}
#endif

				String iconpath = GLOBAL_DEF("application/config/icon", "Variant()");
				if ((iconpath != "") && (!hasicon)) {
					Ref<Image> icon;
					icon.instance();
					if (ImageLoader::load_image(iconpath, icon) == OK) {
						OS::get_singleton()->set_icon(icon);
						hasicon = true;
					}
				}
			}
		}

#ifdef TOOLS_ENABLED
		if (project_manager || (script == "" && test == "" && game_path == "" && !editor)) {

			Engine::get_singleton()->set_editor_hint(true);
			ProjectManager *pmanager = memnew(ProjectManager);
			ProgressDialog *progress_dialog = memnew(ProgressDialog);
			pmanager->add_child(progress_dialog);
			sml->get_root()->add_child(pmanager);
			OS::get_singleton()->set_context(OS::CONTEXT_PROJECTMAN);
			project_manager = true;
		}

		if (project_manager || editor) {
			// Hide console window if requested (Windows-only).
			bool hide_console = EditorSettings::get_singleton()->get_setting("interface/editor/hide_console_window");
			OS::get_singleton()->set_console_visible(!hide_console);

			// Load SSL Certificates from Editor Settings (or builtin).
			Crypto::load_default_certificates(EditorSettings::get_singleton()->get_setting("network/ssl/editor_ssl_certificates").operator String());
		}
#endif
	}

	if (!hasicon) {
		Ref<Image> icon = memnew(Image(app_icon_png));
		OS::get_singleton()->set_icon(icon);
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
uint64_t Main::target_ticks = 0;
uint32_t Main::frames = 0;
uint32_t Main::frame = 0;
bool Main::force_redraw_requested = false;
int Main::iterating = 0;

bool Main::_start_success = false;
bool Main::project_manager = false;
bool Main::editor = false;
bool Main::use_debug_profiler = false;
bool Main::disable_render_loop = false;
bool Main::print_fps = false;
bool Main::auto_quit = false;
int Main::fixed_fps = -1;

bool Main::doc_base = true;
bool Main::export_debug = false;
bool Main::check_only = false;
String Main::doc_tool = "";
String Main::game_path = "";
String Main::script = "";
String Main::test = "";
String Main::_export_preset = "";

bool Main::is_iterating() {
	return iterating > 0;
}

// For performance metrics.
static uint64_t physics_process_max = 0;
static uint64_t idle_process_max = 0;

bool Main::iteration() {

	//for now do not error on this
	//ERR_FAIL_COND_V(iterating, false);

	iterating++;

	uint64_t ticks = OS::get_singleton()->get_ticks_usec();
	Engine::get_singleton()->_frame_ticks = ticks;
	main_timer_sync.set_cpu_ticks_usec(ticks);
	main_timer_sync.set_fixed_fps(fixed_fps);

	uint64_t ticks_elapsed = ticks - last_ticks;

	int physics_fps = Engine::get_singleton()->get_iterations_per_second();
	float frame_slice = 1.0 / physics_fps;

	float time_scale = Engine::get_singleton()->get_time_scale();

	MainFrameTime advance = main_timer_sync.advance(frame_slice, physics_fps);
	double step = advance.idle_step;
	double scaled_step = step * time_scale;

	Engine::get_singleton()->_frame_step = step;
	Engine::get_singleton()->_physics_interpolation_fraction = advance.interpolation_fraction;

	uint64_t physics_process_ticks = 0;
	uint64_t idle_process_ticks = 0;

	frame += ticks_elapsed;

	last_ticks = ticks;

	static const int max_physics_steps = 8;
	if (fixed_fps == -1 && advance.physics_steps > max_physics_steps) {
		step -= (advance.physics_steps - max_physics_steps) * frame_slice;
		advance.physics_steps = max_physics_steps;
	}

	bool exit = false;

	Engine::get_singleton()->_in_physics = true;

	for (int iters = 0; iters < advance.physics_steps; ++iters) {

		uint64_t physics_begin = OS::get_singleton()->get_ticks_usec();

		PhysicsServer::get_singleton()->sync();
		PhysicsServer::get_singleton()->flush_queries();

		Physics2DServer::get_singleton()->sync();
		Physics2DServer::get_singleton()->flush_queries();

		if (OS::get_singleton()->get_main_loop()->iteration(frame_slice * time_scale)) {
			exit = true;
			break;
		}

		message_queue->flush();

		PhysicsServer::get_singleton()->step(frame_slice * time_scale);

		Physics2DServer::get_singleton()->end_sync();
		Physics2DServer::get_singleton()->step(frame_slice * time_scale);

		message_queue->flush();

		physics_process_ticks = MAX(physics_process_ticks, OS::get_singleton()->get_ticks_usec() - physics_begin); // keep the largest one for reference
		physics_process_max = MAX(OS::get_singleton()->get_ticks_usec() - physics_begin, physics_process_max);
		Engine::get_singleton()->_physics_frames++;
	}

	Engine::get_singleton()->_in_physics = false;

	uint64_t idle_begin = OS::get_singleton()->get_ticks_usec();

	if (OS::get_singleton()->get_main_loop()->idle(step * time_scale)) {
		exit = true;
	}
	message_queue->flush();

	VisualServer::get_singleton()->sync(); //sync if still drawing from previous frames.

	if (OS::get_singleton()->can_draw() && !disable_render_loop) {

		if ((!force_redraw_requested) && OS::get_singleton()->is_in_low_processor_usage_mode()) {
			if (VisualServer::get_singleton()->has_changed()) {
				VisualServer::get_singleton()->draw(true, scaled_step); // flush visual commands
				Engine::get_singleton()->frames_drawn++;
			}
		} else {
			VisualServer::get_singleton()->draw(true, scaled_step); // flush visual commands
			Engine::get_singleton()->frames_drawn++;
			force_redraw_requested = false;
		}
	}

	idle_process_ticks = OS::get_singleton()->get_ticks_usec() - idle_begin;
	idle_process_max = MAX(idle_process_ticks, idle_process_max);
	uint64_t frame_time = OS::get_singleton()->get_ticks_usec() - ticks;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->frame();
	}

	AudioServer::get_singleton()->update();

	if (script_debugger) {
		if (script_debugger->is_profiling()) {
			script_debugger->profiling_set_frame_times(USEC_TO_SEC(frame_time), USEC_TO_SEC(idle_process_ticks), USEC_TO_SEC(physics_process_ticks), frame_slice);
		}
		script_debugger->idle_poll();
	}

	frames++;
	Engine::get_singleton()->_idle_frames++;

	if (frame > 1000000) {

		if (editor || project_manager) {
			if (print_fps) {
				print_line("Editor FPS: " + itos(frames));
			}
		} else if (GLOBAL_GET("debug/settings/stdout/print_fps") || print_fps) {
			print_line("Game FPS: " + itos(frames));
		}

		Engine::get_singleton()->_fps = frames;
		performance->set_process_time(USEC_TO_SEC(idle_process_max));
		performance->set_physics_process_time(USEC_TO_SEC(physics_process_max));
		idle_process_max = 0;
		physics_process_max = 0;

		frame %= 1000000;
		frames = 0;
	}

	iterating--;

	if (fixed_fps != -1)
		return exit;

	if (OS::get_singleton()->is_in_low_processor_usage_mode() || !OS::get_singleton()->can_draw())
		OS::get_singleton()->delay_usec(OS::get_singleton()->get_low_processor_usage_mode_sleep_usec()); //apply some delay to force idle time
	else {
		uint32_t frame_delay = Engine::get_singleton()->get_frame_delay();
		if (frame_delay)
			OS::get_singleton()->delay_usec(Engine::get_singleton()->get_frame_delay() * 1000);
	}

	int target_fps = Engine::get_singleton()->get_target_fps();
	if (target_fps > 0 && !Engine::get_singleton()->is_editor_hint()) {
		uint64_t time_step = 1000000L / target_fps;
		target_ticks += time_step;
		uint64_t current_ticks = OS::get_singleton()->get_ticks_usec();
		if (current_ticks < target_ticks) OS::get_singleton()->delay_usec(target_ticks - current_ticks);
		current_ticks = OS::get_singleton()->get_ticks_usec();
		target_ticks = MIN(MAX(target_ticks, current_ticks - time_step), current_ticks + time_step);
	}

#ifdef TOOLS_ENABLED
	if (auto_build_solutions) {
		auto_build_solutions = false;
		if (!EditorNode::get_singleton()->call_build()) {
			ERR_FAIL_V(true);
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
void Main::cleanup() {

	ERR_FAIL_COND(!_start_success);

	if (script_debugger) {
		// Flush any remaining messages
		script_debugger->idle_poll();
	}

	ResourceLoader::remove_custom_loaders();
	ResourceSaver::remove_custom_savers();

	message_queue->flush();
	memdelete(message_queue);

	if (script_debugger) {
		if (use_debug_profiler) {
			script_debugger->profiling_end();
		}

		memdelete(script_debugger);
	}

	OS::get_singleton()->delete_main_loop();

	OS::get_singleton()->delete_command_parser();
	OS::get_singleton()->_execpath = "";
	OS::get_singleton()->_local_clipboard = "";

	ResourceLoader::clear_translation_remaps();
	ResourceLoader::clear_path_remaps();

	ScriptServer::finish_languages();

#ifdef TOOLS_ENABLED
	EditorNode::unregister_editor_types();
#endif

	if (arvr_server) {
		// cleanup now before we pull the rug from underneath...
		memdelete(arvr_server);
	}

	ImageLoader::cleanup();

	unregister_driver_types();
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
	finalize_physics();

	if (packed_data)
		memdelete(packed_data);
	if (file_access_network_client)
		memdelete(file_access_network_client);
	if (performance)
		memdelete(performance);
	if (input_map)
		memdelete(input_map);
	if (translation_server)
		memdelete(translation_server);
	if (globals)
		memdelete(globals);
	if (engine)
		memdelete(engine);

	if (OS::get_singleton()->is_restart_on_exit_set()) {
		//attempt to restart with arguments
		String exec = OS::get_singleton()->get_executable_path();
		List<String> args = OS::get_singleton()->get_restart_on_exit_arguments();
		OS::ProcessID pid = 0;
		OS::get_singleton()->execute(exec, args, false, &pid);
		OS::get_singleton()->set_restart_on_exit(false, List<String>()); //clear list (uses memory)
	}

	unregister_core_driver_types();
	unregister_core_types();

	OS::get_singleton()->finalize_core();
}
