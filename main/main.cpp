/*************************************************************************/
/*  main.cpp                                                             */
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
#include "main.h"

#include "app_icon.gen.h"
#include "core/register_core_types.h"
#include "drivers/register_driver_types.h"
#include "message_queue.h"
#include "modules/register_module_types.h"
#include "os/os.h"
#include "platform/register_platform_apis.h"
#include "project_settings.h"
#include "scene/register_scene_types.h"
#include "script_debugger_local.h"
#include "script_debugger_remote.h"
#include "servers/register_server_types.h"
#include "splash.gen.h"
#include "splash_editor.gen.h"

#include "input_map.h"
#include "io/resource_loader.h"
#include "scene/main/scene_tree.h"
#include "servers/arvr_server.h"
#include "servers/audio_server.h"
#include "servers/physics_2d_server.h"
#include "servers/physics_server.h"

#include "io/resource_loader.h"
#include "script_language.h"

#include "core/io/ip.h"
#include "main/tests/test_main.h"
#include "os/dir_access.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

#ifdef TOOLS_ENABLED
#include "editor/doc/doc_data.h"
#include "editor/doc/doc_data_class_path.gen.h"
#include "editor/editor_node.h"
#include "editor/project_manager.h"
#endif

#include "io/file_access_network.h"
#include "servers/physics_2d_server.h"

#include "core/io/file_access_pack.h"
#include "core/io/file_access_zip.h"
#include "core/io/stream_peer_ssl.h"
#include "core/io/stream_peer_tcp.h"
#include "main/input_default.h"
#include "performance.h"
#include "translation.h"
#include "version.h"
#include "version_hash.gen.h"

static ProjectSettings *globals = NULL;
static Engine *engine = NULL;
static InputMap *input_map = NULL;
static bool _start_success = false;
static ScriptDebugger *script_debugger = NULL;
AudioServer *audio_server = NULL;
ARVRServer *arvr_server = NULL;
PhysicsServer *physics_server = NULL;
Physics2DServer *physics_2d_server = NULL;

static MessageQueue *message_queue = NULL;
static Performance *performance = NULL;

static PackedData *packed_data = NULL;
#ifdef MINIZIP_ENABLED
static ZipArchive *zip_packed_data = NULL;
#endif
static FileAccessNetworkClient *file_access_network_client = NULL;
static TranslationServer *translation_server = NULL;

static OS::VideoMode video_mode;
static bool init_maximized = false;
static bool init_windowed = false;
static bool init_fullscreen = false;
static bool init_use_custom_pos = false;
#ifdef DEBUG_ENABLED
static bool debug_collisions = false;
static bool debug_navigation = false;
#endif
static int frame_delay = 0;
static Vector2 init_custom_pos;
static int video_driver_idx = -1;
static int audio_driver_idx = -1;
static String locale;
static bool use_debug_profiler = false;
static bool force_lowdpi = false;
static int init_screen = -1;
static bool use_vsync = true;
static bool editor = false;
static bool show_help = false;
static bool disable_render_loop = false;
static int fixed_fps = -1;

static OS::ProcessID allow_focus_steal_pid = 0;

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

static String unescape_cmdline(const String &p_str) {

	return p_str.replace("%20", " ");
}

static String get_full_version_string() {

	String hash = String(VERSION_HASH);
	if (hash.length() != 0)
		hash = "." + hash.left(7);
	return String(VERSION_MKSTRING) + hash;
}

//#define DEBUG_INIT
#ifdef DEBUG_INIT
#define MAIN_PRINT(m_txt) print_line(m_txt)
#else
#define MAIN_PRINT(m_txt)
#endif

void Main::print_help(const char *p_binary) {

	print_line(String(VERSION_NAME) + " v" + get_full_version_string() + " - https://godotengine.org");
	OS::get_singleton()->print("(c) 2007-2017 Juan Linietsky, Ariel Manzur.\n");
	OS::get_singleton()->print("(c) 2014-2017 Godot Engine contributors.\n");
	OS::get_singleton()->print("\n");
	OS::get_singleton()->print("Usage: %s [options] [path to scene or 'project.godot' file]\n", p_binary);
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("General options:\n");
	OS::get_singleton()->print("  -h, --help                       Display this help message.\n");
	OS::get_singleton()->print("  --version                        Display the version string.\n");
	OS::get_singleton()->print("  -v, --verbose                    Use verbose stdout mode.\n");
	OS::get_singleton()->print("  --quiet                          Quiet mode, silences stdout messages. Errors are still displayed.\n");
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("Run options:\n");
#ifdef TOOLS_ENABLED
	OS::get_singleton()->print("  -e, --editor                     Start the editor instead of running the scene.\n");
	OS::get_singleton()->print("  -p, --project-manager            Start the project manager, even if a project is auto-detected.\n");
#endif
	OS::get_singleton()->print("  -l, --language <locale>          Use a specific locale (<locale> being a two-letter code).\n");
	OS::get_singleton()->print("  --path <directory>               Path to a project (<directory> must contain a 'project.godot' file).\n");
	OS::get_singleton()->print("  -u, --upwards                    Scan folders upwards for project.godot file.\n");
	OS::get_singleton()->print("  --main-pack <file>               Path to a pack (.pck) file to load.\n");
	OS::get_singleton()->print("  --render-thread <mode>           Render thread mode ('unsafe', 'safe', 'separate').\n");
	OS::get_singleton()->print("  --remote-fs <address>            Remote filesystem (<host/IP>[:<port>] address).\n");
	OS::get_singleton()->print("  --remote-fs-password <password>  Password for remote filesystem.\n");
	OS::get_singleton()->print("  --audio-driver <driver>          Audio driver (");
	for (int i = 0; i < OS::get_singleton()->get_audio_driver_count(); i++) {
		if (i != 0)
			OS::get_singleton()->print(", ");
		OS::get_singleton()->print("'%s'", OS::get_singleton()->get_audio_driver_name(i));
	}
	OS::get_singleton()->print(").\n");
	OS::get_singleton()->print("  --video-driver <driver>          Video driver (");
	for (int i = 0; i < OS::get_singleton()->get_video_driver_count(); i++) {
		if (i != 0)
			OS::get_singleton()->print(", ");
		OS::get_singleton()->print("'%s'", OS::get_singleton()->get_video_driver_name(i));
	}
	OS::get_singleton()->print(").\n");
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("Display options:\n");
	OS::get_singleton()->print("  -f, --fullscreen                 Request fullscreen mode.\n");
	OS::get_singleton()->print("  -m, --maximized                  Request a maximized window.\n");
	OS::get_singleton()->print("  -w, --windowed                   Request windowed mode.\n");
	OS::get_singleton()->print("  --resolution <W>x<H>             Request window resolution.\n");
	OS::get_singleton()->print("  --position <X>,<Y>               Request window position.\n");
	OS::get_singleton()->print("  --low-dpi                        Force low-DPI mode (macOS and Windows only).\n");
	OS::get_singleton()->print("  --no-window                      Disable window creation (Windows only). Useful together with --script.\n");
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("Debug options:\n");
	OS::get_singleton()->print("  -d, --debug                      Debug (local stdout debugger).\n");
	OS::get_singleton()->print("  -b, --breakpoints                Breakpoint list as source::line comma-separated pairs, no spaces (use %%20 instead).\n");
	OS::get_singleton()->print("  --profiling                      Enable profiling in the script debugger.\n");
	OS::get_singleton()->print("  --remote-debug <address>         Remote debug (<host/IP>:<port> address).\n");
#ifdef DEBUG_ENABLED
	OS::get_singleton()->print("  --debug-collisions               Show collisions shapes when running the scene.\n");
	OS::get_singleton()->print("  --debug-navigation               Show navigation polygons when running the scene.\n");
#endif
	OS::get_singleton()->print("  --frame-delay <ms>               Simulate high CPU load (delay each frame by <ms> milliseconds).\n");
	OS::get_singleton()->print("  --time-scale <scale>             Force time scale (higher values are faster, 1.0 is normal speed).\n");
	OS::get_singleton()->print("  --disable-render-loop            Disable render loop so rendering only occurs when called explicitly from script.\n");
	OS::get_singleton()->print("  --disable-crash-handler          Disable crash handler when supported by the platform code.\n");
	OS::get_singleton()->print("  --fixed-fps <fps>                Force a fixed number of frames per second. This setting disables real-time synchronization.\n");
	OS::get_singleton()->print("\n");

	OS::get_singleton()->print("Standalone tools:\n");
	OS::get_singleton()->print("  -s, --script <script>            Run a script.\n");
#ifdef TOOLS_ENABLED
	OS::get_singleton()->print("  --export <target>                Export the project using the given export target.\n");
	OS::get_singleton()->print("  --export-debug                   Use together with --export, enables debug mode for the template.\n");
	OS::get_singleton()->print("  --doctool <path>                 Dump the engine API reference to the given <path> in XML format, merging if existing files are found.\n");
	OS::get_singleton()->print("  --no-docbase                     Disallow dumping the base types (used with --doctool).\n");
#ifdef DEBUG_METHODS_ENABLED
	OS::get_singleton()->print("  --gdnative-generate-json-api     Generate JSON dump of the Godot API for GDNative bindings.\n");
#endif
	OS::get_singleton()->print("  --test <test>                    Run a unit test (");
	const char **test_names = tests_get_names();
	const char *comma = "";
	while (*test_names) {
		OS::get_singleton()->print("%s'%s'", comma, *test_names);
		test_names++;
		comma = ", ";
	}
	OS::get_singleton()->print(").\n");
#endif
}

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

	/* argument parsing and main creation */
	List<String> args;
	List<String> main_args;

	for (int i = 0; i < argc; i++) {

		args.push_back(String::utf8(argv[i]));
	}

	List<String>::Element *I = args.front();

	I = args.front();

	while (I) {

		I->get() = unescape_cmdline(I->get().strip_escapes());
		I = I->next();
	}

	I = args.front();

	String video_driver = "";
	String audio_driver = "";
	String game_path = ".";
	bool upwards = false;
	String debug_mode;
	String debug_host;
	String main_pack;
	bool quiet_stdout = false;
	int rtm = -1;

	String remotefs;
	String remotefs_pass;

	Vector<String> breakpoints;
	bool use_custom_res = true;
	bool force_res = false;

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

	I = args.front();
	while (I) {

		List<String>::Element *N = I->next();

		if (I->get() == "-h" || I->get() == "--help" || I->get() == "/?") { // display help

			show_help = true;
			goto error;

		} else if (I->get() == "--version") {

			print_line(get_full_version_string());
			goto error;

		} else if (I->get() == "--resolution") { // force resolution

			if (I->next()) {

				String vm = I->next()->get();

				if (vm.find("x") == -1) { // invalid parameter format

					OS::get_singleton()->print("Invalid resolution '%s', it should be e.g. '1280x720'.\n", vm.utf8().get_data());
					goto error;
				}

				int w = vm.get_slice("x", 0).to_int();
				int h = vm.get_slice("x", 1).to_int();

				if (w <= 0 || h <= 0) {

					OS::get_singleton()->print("Invalid resolution '%s', width and height must be above 0.\n", vm.utf8().get_data());
					goto error;
				}

				video_mode.width = w;
				video_mode.height = h;
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

					OS::get_singleton()->print("Invalid position '%s', it should be e.g. '80,128'.\n", vm.utf8().get_data());
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

		} else if (I->get() == "-m" || I->get() == "--maximized") { // force maximized window

			init_maximized = true;
			video_mode.maximized = true;
		} else if (I->get() == "-w" || I->get() == "--windowed") { // force windowed window

			init_windowed = true;
		} else if (I->get() == "--profiling") { // enable profiling

			use_debug_profiler = true;
		} else if (I->get() == "--video-driver") { // force video driver

			if (I->next()) {

				video_driver = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing video driver argument, aborting.\n");
				goto error;
			}
		} else if (I->get() == "-l" || I->get() == "--language") { // language

			if (I->next()) {

				locale = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing language argument, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--low-dpi") { // force low DPI (macOS only)

			force_lowdpi = true;
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

				if (I->next()->get() == "safe")
					rtm = OS::RENDER_THREAD_SAFE;
				else if (I->next()->get() == "unsafe")
					rtm = OS::RENDER_THREAD_UNSAFE;
				else if (I->next()->get() == "separate")
					rtm = OS::RENDER_SEPARATE_THREAD;

				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing render thread mode argument, aborting.\n");
				goto error;
			}

		} else if (I->get() == "--audio-driver") { // audio driver

			if (I->next()) {

				audio_driver = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing audio driver argument, aborting.\n");
				goto error;
			}

		} else if (I->get() == "-f" || I->get() == "--fullscreen") { // force fullscreen

			//video_mode.fullscreen=false;
			init_fullscreen = true;
		} else if (I->get() == "-e" || I->get() == "--editor") { // starts editor

			editor = true;
		} else if (I->get() == "--no-window") { // disable window creation, Windows only

			OS::get_singleton()->set_no_window_mode(true);
		} else if (I->get() == "--quiet") { // quieter output

			quiet_stdout = true;
		} else if (I->get() == "-v" || I->get() == "--verbose") { // verbose output
			OS::get_singleton()->_verbose_stdout = true;
		} else if (I->get() == "--path") { // set path of project to start or edit

			if (I->next()) {

				String p = I->next()->get();
				if (OS::get_singleton()->set_cwd(p) == OK) {
					//nothing
				} else {
					game_path = I->next()->get(); //use game_path instead
				}
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing relative or absolute path, aborting.\n");
				goto error;
			}
		} else if (I->get() == "-u" || I->get() == "--upwards") { // scan folders upwards
			upwards = true;
		} else if (I->get().ends_with("project.godot")) {
			String path;
			String file = I->get();
			int sep = MAX(file.find_last("/"), file.find_last("\\"));
			if (sep == -1)
				path = ".";
			else {
				path = file.substr(0, sep);
			}
			if (OS::get_singleton()->set_cwd(path) == OK) {
				// path already specified, don't override
			} else {
				game_path = path;
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

				Engine::get_singleton()->set_time_scale(I->next()->get().to_double());
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
			debug_mode = "local";
#ifdef DEBUG_ENABLED
		} else if (I->get() == "--debug-collisions") {
			debug_collisions = true;
		} else if (I->get() == "--debug-navigation") {
			debug_navigation = true;
#endif
		} else if (I->get() == "--remote-debug") {
			if (I->next()) {

				debug_mode = "remote";
				debug_host = I->next()->get();
				if (debug_host.find(":") == -1) { // wrong address
					OS::get_singleton()->print("Invalid debug host address, it should be of the form <host/IP>:<port>.\n");
					goto error;
				}
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Missing remote debug host address, aborting.\n");
				goto error;
			}
		} else if (I->get() == "--allow_focus_steal_pid") { // not exposed to user
			if (I->next()) {

				allow_focus_steal_pid = I->next()->get().to_int64();
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
		} else if (I->get() == "--disable-crash-handler") {
			OS::get_singleton()->disable_crash_handler();
		} else {

			//test for game path
			bool gpfound = false;

			if (!I->get().begins_with("-") && game_path == "") {
				DirAccess *da = DirAccess::open(I->get());
				if (da != NULL) {
					game_path = I->get();
					gpfound = true;
					memdelete(da);
				}
			}

			if (!gpfound) {
				main_args.push_back(I->get());
			}
		}

		I = N;
	}

	GLOBAL_DEF("memory/limits/multithreaded_server/rid_pool_prealloc", 60);
	GLOBAL_DEF("network/limits/debugger_stdout/max_chars_per_second", 2048);

	if (debug_mode == "remote") {

		ScriptDebuggerRemote *sdr = memnew(ScriptDebuggerRemote);
		uint16_t debug_port = 6007;
		if (debug_host.find(":") != -1) {
			int sep_pos = debug_host.find_last(":");
			debug_port = debug_host.substr(sep_pos + 1, debug_host.length()).to_int();
			debug_host = debug_host.substr(0, sep_pos);
		}
		Error derr = sdr->connect_to_host(debug_host, debug_port);

		if (derr != OK) {
			memdelete(sdr);
		} else {
			script_debugger = sdr;
		}
	} else if (debug_mode == "local") {

		script_debugger = memnew(ScriptDebuggerLocal);
	}

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
	if (script_debugger) {
		//there is a debugger, parse breakpoints

		for (int i = 0; i < breakpoints.size(); i++) {

			String bp = breakpoints[i];
			int sp = bp.find_last(":");
			if (sp == -1) {
				ERR_EXPLAIN("Invalid breakpoint: '" + bp + "', expected file:line format.");
				ERR_CONTINUE(sp == -1);
			}

			script_debugger->insert_breakpoint(bp.substr(sp + 1, bp.length()).to_int(), bp.substr(0, sp));
		}
	}

#ifdef TOOLS_ENABLED
	if (editor) {
		packed_data->set_disabled(true);
		globals->set_disable_feature_overrides(true);
		StreamPeerSSL::initialize_certs = false; //will be initialized by editor
	}

#endif

	if (globals->setup(game_path, main_pack, upwards) != OK) {

#ifdef TOOLS_ENABLED
		editor = false;
#else
		OS::get_singleton()->print("Error: Could not load game path '%s'.\n", game_path.ascii().get_data());

		goto error;
#endif
	}

	GLOBAL_DEF("logging/file_logging/enable_file_logging", false);
	GLOBAL_DEF("logging/file_logging/log_path", "user://logs/log.txt");
	GLOBAL_DEF("logging/file_logging/max_log_files", 10);
	if (FileAccess::get_create_func(FileAccess::ACCESS_USERDATA) && GLOBAL_GET("logging/file_logging/enable_file_logging")) {
		String base_path = GLOBAL_GET("logging/file_logging/log_path");
		int max_files = GLOBAL_GET("logging/file_logging/max_log_files");
		OS::get_singleton()->add_logger(memnew(RotatedFileLogger(base_path, max_files)));
	}

	if (editor) {
		Engine::get_singleton()->set_editor_hint(true);
		main_args.push_back("--editor");
		init_maximized = true;
		video_mode.maximized = true;
		use_custom_res = false;
	}

	if (bool(ProjectSettings::get_singleton()->get("application/run/disable_stdout"))) {
		quiet_stdout = true;
	}
	if (bool(ProjectSettings::get_singleton()->get("application/run/disable_stderr"))) {
		_print_error_enabled = false;
	};

	if (quiet_stdout)
		_print_line_enabled = false;

	OS::get_singleton()->set_cmdline(execpath, main_args);

#ifdef TOOLS_ENABLED

	if (main_args.size() == 0 && (!ProjectSettings::get_singleton()->has_setting("application/run/main_loop_type")) && (!ProjectSettings::get_singleton()->has_setting("application/run/main_scene") || String(ProjectSettings::get_singleton()->get("application/run/main_scene")) == ""))
		use_custom_res = false; //project manager (run without arguments)

#endif

	if (editor)
		input_map->load_default(); //keys for editor
	else
		input_map->load_from_globals(); //keys for game

	//if (video_driver == "") // useless for now, so removing
	//	video_driver = GLOBAL_DEF("display/driver/name", Variant((const char *)OS::get_singleton()->get_video_driver_name(0)));

	GLOBAL_DEF("display/window/size/width", 1024);
	GLOBAL_DEF("display/window/size/height", 600);
	GLOBAL_DEF("display/window/size/resizable", true);
	GLOBAL_DEF("display/window/size/borderless", false);
	GLOBAL_DEF("display/window/size/fullscreen", false);
	GLOBAL_DEF("display/window/size/test_width", 0);
	GLOBAL_DEF("display/window/size/test_height", 0);

	if (use_custom_res) {

		if (!force_res) {
			video_mode.width = GLOBAL_GET("display/window/size/width");
			video_mode.height = GLOBAL_GET("display/window/size/height");

			if (globals->has_setting("display/window/size/test_width") && globals->has_setting("display/window/size/test_height")) {
				int tw = globals->get("display/window/size/test_width");
				int th = globals->get("display/window/size/test_height");
				if (tw > 0 && th > 0) {
					video_mode.width = tw;
					video_mode.height = th;
				}
			}
		}

		video_mode.resizable = GLOBAL_GET("display/window/size/resizable");
		video_mode.borderless_window = GLOBAL_GET("display/window/size/borderless");
		video_mode.fullscreen = GLOBAL_GET("display/window/size/fullscreen");
	}

	if (!force_lowdpi) {
		OS::get_singleton()->_allow_hidpi = GLOBAL_DEF("display/window/dpi/allow_hidpi", false);
	}

	video_mode.use_vsync = GLOBAL_DEF("display/window/vsync/use_vsync", true);

	GLOBAL_DEF("rendering/quality/intended_usage/framebuffer_allocation", 2);
	GLOBAL_DEF("rendering/quality/intended_usage/framebuffer_allocation.mobile", 3);

	if (editor) {
		OS::get_singleton()->_allow_hidpi = true; //editors always in hidpi
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
		audio_driver = GLOBAL_DEF("audio/driver", OS::get_singleton()->get_audio_driver_name(0));
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

	Engine::get_singleton()->set_iterations_per_second(GLOBAL_DEF("physics/common/fixed_fps", 60));
	Engine::get_singleton()->set_target_fps(GLOBAL_DEF("debug/settings/fps/force_fps", 0));

	GLOBAL_DEF("debug/settings/stdout/print_fps", OS::get_singleton()->is_stdout_verbose());

	if (!OS::get_singleton()->_verbose_stdout) //overrided
		OS::get_singleton()->_verbose_stdout = GLOBAL_DEF("debug/settings/stdout/verbose_stdout", false);

	if (frame_delay == 0) {
		frame_delay = GLOBAL_DEF("application/run/frame_delay_msec", 0);
	}

	OS::get_singleton()->set_low_processor_usage_mode(GLOBAL_DEF("application/run/low_processor_mode", false));
	OS::get_singleton()->set_low_processor_usage_mode_sleep_usec(GLOBAL_DEF("application/run/low_processor_mode_sleep_usec", 8000));

	Engine::get_singleton()->set_frame_delay(frame_delay);

	message_queue = memnew(MessageQueue);

	ProjectSettings::get_singleton()->register_global_defaults();

	if (p_second_phase)
		return setup2();

	return OK;

error:

	video_driver = "";
	audio_driver = "";
	game_path = "";

	args.clear();
	main_args.clear();

	if (show_help)
		print_help(execpath);

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

	// Note 1: *zip_packed_data live into *packed_data
	// Note 2: PackedData::~PackedData destroy this.
	/*
#ifdef MINIZIP_ENABLED
	if (zip_packed_data)
		memdelete( zip_packed_data );
#endif
*/

	unregister_core_driver_types();
	unregister_core_types();

	OS::get_singleton()->_cmdline.clear();

	if (message_queue)
		memdelete(message_queue);
	OS::get_singleton()->finalize_core();
	locale = String();

	return ERR_INVALID_PARAMETER;
}

Error Main::setup2(Thread::ID p_main_tid_override) {

	if (p_main_tid_override) {
		Thread::_main_thread_id = p_main_tid_override;
	}

	OS::get_singleton()->initialize(video_mode, video_driver_idx, audio_driver_idx);
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

	register_server_types();

	MAIN_PRINT("Main: Load Remaps");

	Color clear = GLOBAL_DEF("rendering/environment/default_clear_color", Color(0.3, 0.3, 0.3));
	VisualServer::get_singleton()->set_default_clear_color(clear);

	if (show_logo) { //boot logo!
		String boot_logo_path = GLOBAL_DEF("application/boot_splash/image", String());
		bool boot_logo_scale = GLOBAL_DEF("application/boot_splash/fullsize", true);
		ProjectSettings::get_singleton()->set_custom_property_info("application/boot_splash/image", PropertyInfo(Variant::STRING, "application/boot_splash/image", PROPERTY_HINT_FILE, "*.png"));

		Ref<Image> boot_logo;

		boot_logo_path = boot_logo_path.strip_edges();

		if (boot_logo_path != String() /*&& FileAccess::exists(boot_logo_path)*/) {
			print_line("Boot splash path: " + boot_logo_path);
			boot_logo.instance();
			Error err = boot_logo->load(boot_logo_path);
			if (err)
				ERR_PRINTS("Non-existing or invalid boot splash at: " + boot_logo_path + ". Loading default splash.");
		}

		if (boot_logo.is_valid()) {
			OS::get_singleton()->_msec_splash = OS::get_singleton()->get_ticks_msec();
			Color boot_bg = GLOBAL_DEF("application/boot_splash/bg_color", clear);
			VisualServer::get_singleton()->set_boot_image(boot_logo, boot_bg, boot_logo_scale);
#ifndef TOOLS_ENABLED
//no tools, so free the boot logo (no longer needed)
//ProjectSettings::get_singleton()->set("application/boot_logo",Image());
#endif

		} else {
#ifndef NO_DEFAULT_BOOT_LOGO

			MAIN_PRINT("Main: Create bootsplash");
#if defined(TOOLS_ENABLED) && !defined(NO_EDITOR_SPLASH)

			Ref<Image> splash = editor ? memnew(Image(boot_splash_editor_png)) : memnew(Image(boot_splash_png));
#else
			Ref<Image> splash = memnew(Image(boot_splash_png));
#endif

			MAIN_PRINT("Main: ClearColor");
			VisualServer::get_singleton()->set_default_clear_color(boot_splash_bg_color);
			MAIN_PRINT("Main: Image");
			VisualServer::get_singleton()->set_boot_image(splash, boot_splash_bg_color, false);
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

	if (bool(GLOBAL_DEF("display/window/handheld/emulate_touchscreen", false))) {
		if (!OS::get_singleton()->has_touchscreen_ui_hint() && Input::get_singleton() && !editor) {
			//only if no touchscreen ui hint, set emulation
			InputDefault *id = Object::cast_to<InputDefault>(Input::get_singleton());
			if (id)
				id->set_emulate_touch(true);
		}
	}

	MAIN_PRINT("Main: Load Remaps");

	MAIN_PRINT("Main: Load Scene Types");

	register_scene_types();

	GLOBAL_DEF("display/mouse_cursor/custom_image", String());
	GLOBAL_DEF("display/mouse_cursor/custom_image_hotspot", Vector2());
	ProjectSettings::get_singleton()->set_custom_property_info("display/mouse_cursor/custom_image", PropertyInfo(Variant::STRING, "display/mouse_cursor/custom_image", PROPERTY_HINT_FILE, "*.png,*.webp"));

	if (String(ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image")) != String()) {

		//print_line("use custom cursor");
		Ref<Texture> cursor = ResourceLoader::load(ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image"));
		if (cursor.is_valid()) {
			//print_line("loaded ok");
			Vector2 hotspot = ProjectSettings::get_singleton()->get("display/mouse_cursor/custom_image_hotspot");
			Input::get_singleton()->set_custom_mouse_cursor(cursor, hotspot);
		}
	}
#ifdef TOOLS_ENABLED
	ClassDB::set_current_api(ClassDB::API_EDITOR);
	EditorNode::register_editor_types();

	ClassDB::set_current_api(ClassDB::API_CORE);

#endif

	if (allow_focus_steal_pid) {
		OS::get_singleton()->enable_for_stealing_focus(allow_focus_steal_pid);
	}

	MAIN_PRINT("Main: Load Modules, Physics, Drivers, Scripts");

	register_platform_apis();
	register_module_types();

	initialize_physics();
	register_server_singletons();

	register_driver_types();

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

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("CORE API HASH: " + itos(ClassDB::get_api_hash(ClassDB::API_CORE)));
		print_line("EDITOR API HASH: " + itos(ClassDB::get_api_hash(ClassDB::API_EDITOR)));
	}
	MAIN_PRINT("Main: Done");

	return OK;
}

bool Main::start() {

	ERR_FAIL_COND_V(!_start_success, false);

	bool hasicon = false;
	bool editor = false;
	String doc_tool;
	List<String> removal_docs;
	bool doc_base = true;
	String game_path;
	String script;
	String test;
	String _export_preset;
	bool export_debug = false;
	bool project_manager_request = false;

	List<String> args = OS::get_singleton()->get_cmdline_args();
	for (int i = 0; i < args.size(); i++) {
		//parameters that do not have an argument to the right
		if (args[i] == "--no-docbase") {
			doc_base = false;
		} else if (args[i] == "-e" || args[i] == "--editor") {
			editor = true;
		} else if (args[i] == "-p" || args[i] == "--project-manager") {
			project_manager_request = true;
		} else if (args[i].length() && args[i][0] != '-' && game_path == "") {
			game_path = args[i];
		}
		//parameters that have an argument to the right
		else if (i < (args.size() - 1)) {
			bool parsed_pair = true;
			if (args[i] == "--doctool") {
				doc_tool = args[i + 1];
				for (int j = i + 2; j < args.size(); j++)
					removal_docs.push_back(args[j]);
			} else if (args[i] == "-s" || args[i] == "--script") {
				script = args[i + 1];
			} else if (args[i] == "--test") {
				test = args[i + 1];
			} else if (args[i] == "--export") {
				editor = true; //needs editor
				if (i + 1 < args.size()) {
					_export_preset = args[i + 1];
				} else {
					ERR_PRINT("Export preset name not specified");
					return false;
				}
			} else if (args[i] == "--export-debug") {
				editor = true; //needs editor
				if (i + 1 < args.size()) {
					_export_preset = args[i + 1];
				} else {
					ERR_PRINT("Export preset name not specified");
					return false;
				}
				export_debug = true;
			} else {
				// The parameter does not match anything known, don't skip the next argument
				parsed_pair = false;
			}
			if (parsed_pair) {
				i++;
			}
		}
	}

	GLOBAL_DEF("editor/active", editor);

	String main_loop_type;
#ifdef TOOLS_ENABLED
	if (doc_tool != "") {

		{
			DirAccessRef da = DirAccess::open(doc_tool);
			if (!da) {
				ERR_EXPLAIN("Argument supplied to --doctool must be a base godot build directory");
				ERR_FAIL_V(false);
			}
		}
		DocData doc;
		doc.generate(doc_base);

		DocData docsrc;
		Map<String, String> doc_data_classes;
		Set<String> checked_paths;
		print_line("Loading docs..");

		for (int i = 0; i < _doc_data_class_path_count; i++) {
			String path = doc_tool.plus_file(_doc_data_class_paths[i].path);
			String name = _doc_data_class_paths[i].name;
			doc_data_classes[name] = path;
			if (!checked_paths.has(path)) {
				checked_paths.insert(path);
				docsrc.load_classes(path);
				print_line("Loading docs from: " + path);
			}
		}

		String index_path = doc_tool.plus_file("doc/classes");
		docsrc.load_classes(index_path);
		checked_paths.insert(index_path);
		print_line("Loading docs from: " + index_path);

		print_line("Merging docs..");
		doc.merge_from(docsrc);
		for (Set<String>::Element *E = checked_paths.front(); E; E = E->next()) {
			print_line("Erasing old docs at: " + E->get());
			DocData::erase_classes(E->get());
		}

		print_line("Generating new docs..");
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
#ifdef DEBUG_ENABLED
		main_loop = test_main(test, args);

		if (!main_loop)
			return false;

#endif

	} else if (script != "") {

		Ref<Script> script_res = ResourceLoader::load(script);
		ERR_EXPLAIN("Can't load script: " + script);
		ERR_FAIL_COND_V(script_res.is_null(), false);

		if (script_res->can_instance() /*&& script_res->inherits_from("SceneTreeScripted")*/) {

			StringName instance_type = script_res->get_instance_base_type();
			Object *obj = ClassDB::instance(instance_type);
			MainLoop *script_loop = Object::cast_to<MainLoop>(obj);
			if (!script_loop) {
				if (obj)
					memdelete(obj);
				ERR_EXPLAIN("Can't load script '" + script + "', it does not inherit from a MainLoop type");
				ERR_FAIL_COND_V(!script_loop, false);
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
			if (!ml) {
				ERR_EXPLAIN("Can't instance MainLoop type");
				ERR_FAIL_V(false);
			}

			main_loop = Object::cast_to<MainLoop>(ml);
			if (!main_loop) {

				memdelete(ml);
				ERR_EXPLAIN("Invalid MainLoop type");
				ERR_FAIL_V(false);
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

		{
		}

		if (!editor) {
			//standard helpers that can be changed from main config

			String stretch_mode = GLOBAL_DEF("display/window/stretch/mode", "disabled");
			String stretch_aspect = GLOBAL_DEF("display/window/stretch/aspect", "ignore");
			Size2i stretch_size = Size2(GLOBAL_DEF("display/window/size/width", 0), GLOBAL_DEF("display/window/size/height", 0));
			real_t stretch_shrink = GLOBAL_DEF("display/window/stretch/shrink", 1.0f);

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

			bool font_oversampling = GLOBAL_DEF("rendering/quality/dynamic_fonts/use_oversampling", false);
			sml->set_use_font_oversampling(font_oversampling);

		} else {
			GLOBAL_DEF("display/window/stretch/mode", "disabled");
			ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/mode", PropertyInfo(Variant::STRING, "display/window/stretch/mode", PROPERTY_HINT_ENUM, "disabled,2d,viewport"));
			GLOBAL_DEF("display/window/stretch/aspect", "ignore");
			ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/aspect", PropertyInfo(Variant::STRING, "display/window/stretch/aspect", PROPERTY_HINT_ENUM, "ignore,keep,keep_width,keep_height,expand"));
			GLOBAL_DEF("display/window/stretch/shrink", 1);
			ProjectSettings::get_singleton()->set_custom_property_info("display/window/stretch/shrink", PropertyInfo(Variant::STRING, "display/window/stretch/shrink", PROPERTY_HINT_RANGE, "1,8,1"));
			sml->set_auto_accept_quit(GLOBAL_DEF("application/config/auto_accept_quit", true));
			sml->set_quit_on_go_back(GLOBAL_DEF("application/config/quit_on_go_back", true));
			GLOBAL_DEF("gui/common/snap_controls_to_pixels", true);
			GLOBAL_DEF("rendering/quality/dynamic_fonts/use_oversampling", false);
		}

		String local_game_path;
		if (game_path != "" && !project_manager_request) {

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
							local_game_path = da->get_current_dir() + "/" + local_game_path;
							memdelete(da);
						} else {

							DirAccess *da = DirAccess::open(local_game_path.substr(0, sep));
							if (da) {
								local_game_path = da->get_current_dir() + "/" + local_game_path.substr(sep + 1, local_game_path.length());
								memdelete(da);
							}
						}
					}
				}
			}

			local_game_path = ProjectSettings::get_singleton()->localize_path(local_game_path);

#ifdef TOOLS_ENABLED
			if (editor) {

				Error serr = editor_node->load_scene(local_game_path);
				if (serr != OK)
					ERR_PRINT("Failed to load scene");
				OS::get_singleton()->set_context(OS::CONTEXT_EDITOR);
			}
#endif
		}

		if (!project_manager_request && !editor) {
			if (game_path != "" || script != "") {
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
					ERR_EXPLAIN("Can't autoload: " + path);
					ERR_CONTINUE(res.is_null());
					Node *n = NULL;
					if (res->is_class("PackedScene")) {
						Ref<PackedScene> ps = res;
						n = ps->instance();
					} else if (res->is_class("Script")) {
						Ref<Script> s = res;
						StringName ibt = s->get_instance_base_type();
						bool valid_type = ClassDB::is_parent_class(ibt, "Node");
						ERR_EXPLAIN("Script does not inherit a Node: " + path);
						ERR_CONTINUE(!valid_type);

						Object *obj = ClassDB::instance(ibt);

						ERR_EXPLAIN("Cannot instance script for autoload, expected 'Node' inheritance, got: " + String(ibt));
						ERR_CONTINUE(obj == NULL);

						n = Object::cast_to<Node>(obj);
						n->set_script(s.get_ref_ptr());
					}

					ERR_EXPLAIN("Path in autoload not a node or script: " + path);
					ERR_CONTINUE(!n);
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
				//singletons
			}

			if (game_path != "") {
				Node *scene = NULL;
				Ref<PackedScene> scenedata = ResourceLoader::load(local_game_path);
				if (scenedata.is_valid())
					scene = scenedata->instance();

				ERR_EXPLAIN("Failed loading scene: " + local_game_path);
				ERR_FAIL_COND_V(!scene, false)
				sml->add_current_scene(scene);

				String iconpath = GLOBAL_DEF("application/config/icon", "Variant()");
				if (iconpath != "") {
					Ref<Image> icon;
					icon.instance();
					if (icon->load(iconpath) == OK) {
						OS::get_singleton()->set_icon(icon);
						hasicon = true;
					}
				}
			}
		}

#ifdef TOOLS_ENABLED
		if (project_manager_request || (script == "" && test == "" && game_path == "" && !editor)) {

			ProjectManager *pmanager = memnew(ProjectManager);
			ProgressDialog *progress_dialog = memnew(ProgressDialog);
			pmanager->add_child(progress_dialog);
			sml->get_root()->add_child(pmanager);
			OS::get_singleton()->set_context(OS::CONTEXT_PROJECTMAN);
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

uint64_t Main::last_ticks = 0;
uint64_t Main::target_ticks = 0;
float Main::time_accum = 0;
uint32_t Main::frames = 0;
uint32_t Main::frame = 0;
bool Main::force_redraw_requested = false;

//for performance metrics
static uint64_t physics_process_max = 0;
static uint64_t idle_process_max = 0;

bool Main::iteration() {

	uint64_t ticks = OS::get_singleton()->get_ticks_usec();
	Engine::get_singleton()->_frame_ticks = ticks;

	uint64_t ticks_elapsed = ticks - last_ticks;

	double step = (double)ticks_elapsed / 1000000.0;
	if (fixed_fps != -1)
		step = 1.0 / fixed_fps;

	float frame_slice = 1.0 / Engine::get_singleton()->get_iterations_per_second();

	Engine::get_singleton()->_frame_step = step;

	/*
	if (time_accum+step < frame_slice)
		return false;
	*/

	uint64_t physics_process_ticks = 0;
	uint64_t idle_process_ticks = 0;

	frame += ticks_elapsed;

	last_ticks = ticks;

	if (fixed_fps == -1 && step > frame_slice * 8)
		step = frame_slice * 8;

	time_accum += step;

	float time_scale = Engine::get_singleton()->get_time_scale();

	bool exit = false;

	int iters = 0;

	Engine::get_singleton()->_in_physics = true;

	while (time_accum > frame_slice) {

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

		time_accum -= frame_slice;
		message_queue->flush();

		physics_process_ticks = MAX(physics_process_ticks, OS::get_singleton()->get_ticks_usec() - physics_begin); // keep the largest one for reference
		physics_process_max = MAX(OS::get_singleton()->get_ticks_usec() - physics_begin, physics_process_max);
		iters++;
		Engine::get_singleton()->_physics_frames++;
	}

	Engine::get_singleton()->_in_physics = false;

	uint64_t idle_begin = OS::get_singleton()->get_ticks_usec();

	OS::get_singleton()->get_main_loop()->idle(step * time_scale);
	message_queue->flush();

	VisualServer::get_singleton()->sync(); //sync if still drawing from previous frames.

	if (OS::get_singleton()->can_draw() && !disable_render_loop) {

		if ((!force_redraw_requested) && OS::get_singleton()->is_in_low_processor_usage_mode()) {
			if (VisualServer::get_singleton()->has_changed()) {
				VisualServer::get_singleton()->draw(); // flush visual commands
				Engine::get_singleton()->frames_drawn++;
			}
		} else {
			VisualServer::get_singleton()->draw(); // flush visual commands
			Engine::get_singleton()->frames_drawn++;
			force_redraw_requested = false;
		}
	}

	if (AudioServer::get_singleton())
		AudioServer::get_singleton()->update();

	idle_process_ticks = OS::get_singleton()->get_ticks_usec() - idle_begin;
	idle_process_max = MAX(idle_process_ticks, idle_process_max);
	uint64_t frame_time = OS::get_singleton()->get_ticks_usec() - ticks;

	for (int i = 0; i < ScriptServer::get_language_count(); i++) {
		ScriptServer::get_language(i)->frame();
	}

	if (script_debugger) {
		if (script_debugger->is_profiling()) {
			script_debugger->profiling_set_frame_times(USEC_TO_SEC(frame_time), USEC_TO_SEC(idle_process_ticks), USEC_TO_SEC(physics_process_ticks), frame_slice);
		}
		script_debugger->idle_poll();
	}

	frames++;
	Engine::get_singleton()->_idle_frames++;

	if (frame > 1000000) {

		if (GLOBAL_DEF("debug/settings/stdout/print_fps", OS::get_singleton()->is_stdout_verbose())) {
			print_line("FPS: " + itos(frames));
		};

		Engine::get_singleton()->_fps = frames;
		performance->set_process_time(USEC_TO_SEC(idle_process_max));
		performance->set_physics_process_time(USEC_TO_SEC(physics_process_max));
		idle_process_max = 0;
		physics_process_max = 0;

		frame %= 1000000;
		frames = 0;
	}

	if (fixed_fps != -1)
		return exit;

	if (OS::get_singleton()->is_in_low_processor_usage_mode() || !OS::get_singleton()->can_draw())
		OS::get_singleton()->delay_usec(OS::get_singleton()->get_low_processor_usage_mode_sleep_usec()); //apply some delay to force idle time (results in about 60 FPS max)
	else {
		uint32_t frame_delay = Engine::get_singleton()->get_frame_delay();
		if (frame_delay)
			OS::get_singleton()->delay_usec(Engine::get_singleton()->get_frame_delay() * 1000);
	}

	int target_fps = Engine::get_singleton()->get_target_fps();
	if (target_fps > 0) {
		uint64_t time_step = 1000000L / target_fps;
		target_ticks += time_step;
		uint64_t current_ticks = OS::get_singleton()->get_ticks_usec();
		if (current_ticks < target_ticks) OS::get_singleton()->delay_usec(target_ticks - current_ticks);
		current_ticks = OS::get_singleton()->get_ticks_usec();
		target_ticks = MIN(MAX(target_ticks, current_ticks - time_step), current_ticks + time_step);
	}

	return exit;
}

void Main::force_redraw() {

	force_redraw_requested = true;
};

void Main::cleanup() {

	ERR_FAIL_COND(!_start_success);

	if (script_debugger) {
		if (use_debug_profiler) {
			script_debugger->profiling_end();
		}

		memdelete(script_debugger);
	}

	OS::get_singleton()->delete_main_loop();

	OS::get_singleton()->_cmdline.clear();
	OS::get_singleton()->_execpath = "";
	OS::get_singleton()->_local_clipboard = "";

	ResourceLoader::clear_translation_remaps();
	ResourceLoader::clear_path_remaps();

	ScriptServer::finish_languages();

#ifdef TOOLS_ENABLED
	EditorNode::unregister_editor_types();
#endif

	if (audio_server) {
		audio_server->finish();
		memdelete(audio_server);
	}

	if (arvr_server) {
		// cleanup now before we pull the rug from underneath...
		memdelete(arvr_server);
	}

	unregister_driver_types();
	unregister_module_types();
	unregister_platform_apis();
	unregister_scene_types();
	unregister_server_types();

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

	message_queue->flush();
	memdelete(message_queue);

	unregister_core_driver_types();
	unregister_core_types();

	OS::get_singleton()->clear_last_error();
	OS::get_singleton()->finalize_core();
}
