/*************************************************************************/
/*  main.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "core/register_core_types.h"
#include "drivers/register_driver_types.h"
#include "global_config.h"
#include "message_queue.h"
#include "modules/register_module_types.h"
#include "os/os.h"
#include "scene/register_scene_types.h"
#include "script_debugger_local.h"
#include "script_debugger_remote.h"
#include "servers/register_server_types.h"
#include "splash.h"

#include "input_map.h"
#include "io/resource_loader.h"
#include "scene/main/scene_main_loop.h"
#include "servers/audio_server.h"

#include "io/resource_loader.h"
#include "script_language.h"

#include "core/io/ip.h"
#include "main/tests/test_main.h"
#include "os/dir_access.h"
#include "scene/main/viewport.h"
#include "scene/resources/packed_scene.h"

#ifdef TOOLS_ENABLED
#include "editor/doc/doc_data.h"
#include "editor/editor_node.h"
#include "editor/project_manager.h"
#endif

#include "io/file_access_network.h"
#include "servers/physics_2d_server.h"

#include "core/io/file_access_pack.h"
#include "core/io/file_access_zip.h"
#include "core/io/stream_peer_ssl.h"
#include "core/io/stream_peer_tcp.h"
#include "core/os/thread.h"
#include "main/input_default.h"
#include "performance.h"
#include "translation.h"
#include "version.h"

static GlobalConfig *globals = NULL;
static Engine *engine = NULL;
static InputMap *input_map = NULL;
static bool _start_success = false;
static ScriptDebugger *script_debugger = NULL;
AudioServer *audio_server = NULL;

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

static String unescape_cmdline(const String &p_str) {

	return p_str.replace("%20", " ");
}

//#define DEBUG_INIT

#ifdef DEBUG_INIT
#define MAIN_PRINT(m_txt) print_line(m_txt)
#else
#define MAIN_PRINT(m_txt)
#endif

void Main::print_help(const char *p_binary) {

	OS::get_singleton()->print(VERSION_FULL_NAME " (c) 2008-2017 Juan Linietsky, Ariel Manzur.\n");
	OS::get_singleton()->print("Usage: %s [options] [scene]\n", p_binary);
	OS::get_singleton()->print("Options:\n");
	OS::get_singleton()->print("\t-path [dir] : Path to a game, containing godot.cfg\n");
#ifdef TOOLS_ENABLED
	OS::get_singleton()->print("\t-e,-editor : Bring up the editor instead of running the scene.\n");
#endif
	OS::get_singleton()->print("\t-test [test] : Run a test.\n");
	OS::get_singleton()->print("\t\t(");
	const char **test_names = tests_get_names();
	const char *coma = "";
	while (*test_names) {

		OS::get_singleton()->print("%s%s", coma, *test_names);
		test_names++;
		coma = ", ";
	}
	OS::get_singleton()->print(")\n");

	OS::get_singleton()->print("\t-r WIDTHxHEIGHT\t : Request Window Resolution\n");
	OS::get_singleton()->print("\t-p XxY\t : Request Window Position\n");
	OS::get_singleton()->print("\t-f\t\t : Request Fullscreen\n");
	OS::get_singleton()->print("\t-mx\t\t Request Maximized\n");
	OS::get_singleton()->print("\t-w\t\t Request Windowed\n");
	OS::get_singleton()->print("\t-vd DRIVER\t : Video Driver (");
	for (int i = 0; i < OS::get_singleton()->get_video_driver_count(); i++) {

		if (i != 0)
			OS::get_singleton()->print(", ");
		OS::get_singleton()->print("%s", OS::get_singleton()->get_video_driver_name(i));
	}
	OS::get_singleton()->print(")\n");
	OS::get_singleton()->print("\t-ldpi\t : Force low-dpi mode (OSX Only)\n");

	OS::get_singleton()->print("\t-ad DRIVER\t : Audio Driver (");
	for (int i = 0; i < OS::get_singleton()->get_audio_driver_count(); i++) {

		if (i != 0)
			OS::get_singleton()->print(", ");
		OS::get_singleton()->print("%s", OS::get_singleton()->get_audio_driver_name(i));
	}
	OS::get_singleton()->print(")\n");
	OS::get_singleton()->print("\t-rthread <mode>\t : Render Thread Mode ('unsafe', 'safe', 'separate').\n");
	OS::get_singleton()->print("\t-s,-script [script] : Run a script.\n");
	OS::get_singleton()->print("\t-d,-debug : Debug (local stdout debugger).\n");
	OS::get_singleton()->print("\t-rdebug ADDRESS : Remote debug (<ip>:<port> host address).\n");
	OS::get_singleton()->print("\t-fdelay [msec]: Simulate high CPU load (delay each frame by [msec]).\n");
	OS::get_singleton()->print("\t-timescale [msec]: Simulate high CPU load (delay each frame by [msec]).\n");
	OS::get_singleton()->print("\t-bp : breakpoint list as source::line comma separated pairs, no spaces (%%20,%%2C,etc instead).\n");
	OS::get_singleton()->print("\t-v : Verbose stdout mode\n");
	OS::get_singleton()->print("\t-lang [locale]: Use a specific locale\n");
	OS::get_singleton()->print("\t-rfs <host/ip>[:<port>] : Remote FileSystem.\n");
	OS::get_singleton()->print("\t-rfs_pass <password> : Password for Remote FileSystem.\n");
#ifdef TOOLS_ENABLED
	OS::get_singleton()->print("\t-doctool FILE: Dump the whole engine api to FILE in XML format. If FILE exists, it will be merged.\n");
	OS::get_singleton()->print("\t-nodocbase: Disallow dump the base types (used with -doctool).\n");
	OS::get_singleton()->print("\t-export [target] Export the project using given export target.\n");
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

	Thread::_main_thread_id = Thread::get_caller_ID();

	globals = memnew(GlobalConfig);
	input_map = memnew(InputMap);

	register_core_settings(); //here globals is present

	translation_server = memnew(TranslationServer);
	performance = memnew(Performance);
	globals->add_singleton(GlobalConfig::Singleton("Performance", performance));

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
		//print_line("CMD: "+I->get());
		I = I->next();
	}

	I = args.front();

	video_mode = OS::get_singleton()->get_default_video_mode();

	String video_driver = "";
	String audio_driver = "";
	String game_path = ".";
	String debug_mode;
	String debug_host;
	String main_pack;
	bool quiet_stdout = false;
	int rtm = -1;

	String remotefs;
	String remotefs_pass;

	String screen = "";

	List<String> pack_list;
	Vector<String> breakpoints;
	bool use_custom_res = true;
	bool force_res = false;

	I = args.front();

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

	while (I) {

		List<String>::Element *N = I->next();

		if (I->get() == "-noop") {

			// no op
		} else if (I->get() == "-h" || I->get() == "--help" || I->get() == "/?") { // resolution

			goto error;

		} else if (I->get() == "-r") { // resolution

			if (I->next()) {

				String vm = I->next()->get();

				if (vm.find("x") == -1) { // invalid parameter format

					OS::get_singleton()->print("Invalid -r argument: %s\n", vm.utf8().get_data());
					goto error;
				}

				int w = vm.get_slice("x", 0).to_int();
				int h = vm.get_slice("x", 1).to_int();

				if (w == 0 || h == 0) {

					OS::get_singleton()->print("Invalid -r resolution, x and y must be >0\n");
					goto error;
				}

				video_mode.width = w;
				video_mode.height = h;
				force_res = true;

				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Invalid -p argument, needs resolution\n");
				goto error;
			}
		} else if (I->get() == "-p") { // position

			if (I->next()) {

				String vm = I->next()->get();

				if (vm.find("x") == -1) { // invalid parameter format

					OS::get_singleton()->print("Invalid -p argument: %s\n", vm.utf8().get_data());
					goto error;
				}

				int x = vm.get_slice("x", 0).to_int();
				int y = vm.get_slice("x", 1).to_int();

				init_custom_pos = Point2(x, y);
				init_use_custom_pos = true;

				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Invalid -r argument, needs position\n");
				goto error;
			}

		} else if (I->get() == "-mx") { // video driver

			init_maximized = true;
		} else if (I->get() == "-w") { // video driver

			init_windowed = true;
		} else if (I->get() == "-profile") { // video driver

			use_debug_profiler = true;
		} else if (I->get() == "-vd") { // video driver

			if (I->next()) {

				video_driver = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Invalid -cd argument, needs driver name\n");
				goto error;
			}
		} else if (I->get() == "-lang") { // language

			if (I->next()) {

				locale = I->next()->get();
				N = I->next()->next();
			} else {
				OS::get_singleton()->print("Invalid -lang argument, needs language code\n");
				goto error;
			}
		} else if (I->get() == "-ldpi") { // language

			force_lowdpi = true;
		} else if (I->get() == "-rfs") { // language

			if (I->next()) {

				remotefs = I->next()->get();
				N = I->next()->next();
			} else {
				goto error;
			}
		} else if (I->get() == "-rfs_pass") { // language

			if (I->next()) {

				remotefs_pass = I->next()->get();
				N = I->next()->next();
			} else {
				goto error;
			}
		} else if (I->get() == "-rthread") { // language

			if (I->next()) {

				if (I->next()->get() == "safe")
					rtm = OS::RENDER_THREAD_SAFE;
				else if (I->next()->get() == "unsafe")
					rtm = OS::RENDER_THREAD_UNSAFE;
				else if (I->next()->get() == "separate")
					rtm = OS::RENDER_SEPARATE_THREAD;

				N = I->next()->next();
			} else {
				goto error;
			}

		} else if (I->get() == "-ad") { // video driver

			if (I->next()) {

				audio_driver = I->next()->get();
				N = I->next()->next();
			} else {
				goto error;
			}

		} else if (I->get() == "-f") { // fullscreen

			//video_mode.fullscreen=false;
			init_fullscreen = true;
		} else if (I->get() == "-e" || I->get() == "-editor") { // fonud editor

			editor = true;
		} else if (I->get() == "-nowindow") { // fullscreen

			OS::get_singleton()->set_no_window_mode(true);
		} else if (I->get() == "-quiet") { // fullscreen

			quiet_stdout = true;
		} else if (I->get() == "-v") { // fullscreen
			OS::get_singleton()->_verbose_stdout = true;
		} else if (I->get() == "-path") { // resolution

			if (I->next()) {

				String p = I->next()->get();
				if (OS::get_singleton()->set_cwd(p) == OK) {
					//nothing
				} else {
					game_path = I->next()->get(); //use game_path instead
				}
				N = I->next()->next();
			} else {
				goto error;
			}
		} else if (I->get() == "-bp") { // /breakpoints

			if (I->next()) {

				String bplist = I->next()->get();
				breakpoints = bplist.split(",");
				N = I->next()->next();
			} else {
				goto error;
			}

		} else if (I->get() == "-fdelay") { // resolution

			if (I->next()) {

				frame_delay = I->next()->get().to_int();
				N = I->next()->next();
			} else {
				goto error;
			}

		} else if (I->get() == "-timescale") { // resolution

			if (I->next()) {

				Engine::get_singleton()->set_time_scale(I->next()->get().to_double());
				N = I->next()->next();
			} else {
				goto error;
			}

		} else if (I->get() == "-pack") {

			if (I->next()) {

				pack_list.push_back(I->next()->get());
				N = I->next()->next();
			} else {

				goto error;
			};

		} else if (I->get() == "-main_pack") {

			if (I->next()) {

				main_pack = I->next()->get();
				N = I->next()->next();
			} else {

				goto error;
			};

		} else if (I->get() == "-debug" || I->get() == "-d") {
			debug_mode = "local";
#ifdef DEBUG_ENABLED
		} else if (I->get() == "-debugcol" || I->get() == "-dc") {
			debug_collisions = true;
		} else if (I->get() == "-debugnav" || I->get() == "-dn") {
			debug_navigation = true;
#endif
		} else if (I->get() == "-editor_scene") {

			if (I->next()) {

				GlobalConfig::get_singleton()->set("editor_scene", game_path = I->next()->get());
			} else {
				goto error;
			}

		} else if (I->get() == "-rdebug") {
			if (I->next()) {

				debug_mode = "remote";
				debug_host = I->next()->get();
				if (debug_host.find(":") == -1) { //wrong host
					OS::get_singleton()->print("Invalid debug host string\n");
					goto error;
				}
				N = I->next()->next();
			} else {
				goto error;
			}
		} else if (I->get() == "-epid") {
			if (I->next()) {

				int editor_pid = I->next()->get().to_int();
				GlobalConfig::get_singleton()->set("editor_pid", editor_pid);
				N = I->next()->next();
			} else {
				goto error;
			}
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

	GLOBAL_DEF("memory/multithread/thread_rid_pool_prealloc", 60);

	GLOBAL_DEF("network/debug/max_remote_stdout_chars_per_second", 2048);
	GLOBAL_DEF("network/debug/remote_port", 6007);

	if (debug_mode == "remote") {

		ScriptDebuggerRemote *sdr = memnew(ScriptDebuggerRemote);
		uint16_t debug_port = GLOBAL_GET("network/debug/remote_port");
		if (debug_host.find(":") != -1) {
			debug_port = debug_host.get_slicec(':', 1).to_int();
			debug_host = debug_host.get_slicec(':', 0);
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
			OS::get_singleton()->printerr("Could not connect to remotefs: %s:%i\n", remotefs.utf8().get_data(), port);
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
		globals->set_disable_platform_override(true);
		StreamPeerSSL::initialize_certs = false; //will be initialized by editor
	}

#endif

	if (globals->setup(game_path, main_pack) != OK) {

#ifdef TOOLS_ENABLED
		editor = false;
#else
		OS::get_singleton()->print("error: Couldn't load game path '%s'\n", game_path.ascii().get_data());

		goto error;
#endif
	}

	if (editor) {
		main_args.push_back("-editor");
		init_maximized = true;
		use_custom_res = false;
	}

	if (bool(GlobalConfig::get_singleton()->get("application/disable_stdout"))) {
		quiet_stdout = true;
	}
	if (bool(GlobalConfig::get_singleton()->get("application/disable_stderr"))) {
		_print_error_enabled = false;
	};

	if (quiet_stdout)
		_print_line_enabled = false;

	OS::get_singleton()->set_cmdline(execpath, main_args);

#ifdef TOOLS_ENABLED

	if (main_args.size() == 0 && (!GlobalConfig::get_singleton()->has("application/main_loop_type")) && (!GlobalConfig::get_singleton()->has("application/main_scene") || String(GlobalConfig::get_singleton()->get("application/main_scene")) == ""))
		use_custom_res = false; //project manager (run without arguments)

#endif

	if (editor)
		input_map->load_default(); //keys for editor
	else
		input_map->load_from_globals(); //keys for game

	if (video_driver == "") // specified in godot.cfg
		video_driver = GLOBAL_DEF("display/driver/name", Variant((const char *)OS::get_singleton()->get_video_driver_name(0)));

	if (!force_res && use_custom_res && globals->has("display/window/width"))
		video_mode.width = globals->get("display/window/width");
	if (!force_res && use_custom_res && globals->has("display/window/height"))
		video_mode.height = globals->get("display/window/height");
	if (!editor && (!bool(globals->get("display/window/allow_hidpi")) || force_lowdpi)) {
		OS::get_singleton()->_allow_hidpi = false;
	}
	if (use_custom_res && globals->has("display/window/fullscreen"))
		video_mode.fullscreen = globals->get("display/window/fullscreen");
	if (use_custom_res && globals->has("display/window/resizable"))
		video_mode.resizable = globals->get("display/window/resizable");
	if (use_custom_res && globals->has("display/window/borderless"))
		video_mode.borderless_window = globals->get("display/window/borderless");

	if (!force_res && use_custom_res && globals->has("display/window/test_width") && globals->has("display/window/test_height")) {
		int tw = globals->get("display/window/test_width");
		int th = globals->get("display/window/test_height");
		if (tw > 0 && th > 0) {
			video_mode.width = tw;
			video_mode.height = th;
		}
	}

	GLOBAL_DEF("display/window/width", video_mode.width);
	GLOBAL_DEF("display/window/height", video_mode.height);
	GLOBAL_DEF("display/window/allow_hidpi", false);
	GLOBAL_DEF("display/window/fullscreen", video_mode.fullscreen);
	GLOBAL_DEF("display/window/resizable", video_mode.resizable);
	GLOBAL_DEF("display/window/borderless", video_mode.borderless_window);
	use_vsync = GLOBAL_DEF("display/window/use_vsync", use_vsync);
	GLOBAL_DEF("display/window/test_width", 0);
	GLOBAL_DEF("display/window/test_height", 0);
	Engine::get_singleton()->_pixel_snap = GLOBAL_DEF("rendering/2d/use_pixel_snap", false);
	OS::get_singleton()->_keep_screen_on = GLOBAL_DEF("display/energy_saving/keep_screen_on", true);
	if (rtm == -1) {
		rtm = GLOBAL_DEF("rendering/threads/thread_model", OS::RENDER_THREAD_SAFE);
		if (rtm >= 1) //hack for now
			rtm = 1;
	}

	if (rtm >= 0 && rtm < 3) {
		if (editor) {
			rtm = OS::RENDER_THREAD_SAFE;
		}
		OS::get_singleton()->_render_thread_mode = OS::RenderThreadMode(rtm);
	}

	/* Determine Video Driver */

	if (audio_driver == "") { // specified in godot.cfg
		audio_driver = GLOBAL_DEF("audio/driver", OS::get_singleton()->get_audio_driver_name(0));
	}

	for (int i = 0; i < OS::get_singleton()->get_video_driver_count(); i++) {

		if (video_driver == OS::get_singleton()->get_video_driver_name(i)) {

			video_driver_idx = i;
			break;
		}
	}

	if (video_driver_idx < 0) {

		OS::get_singleton()->alert("Invalid Video Driver: " + video_driver);
		video_driver_idx = 0;
		//goto error;
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
		String orientation = GLOBAL_DEF("display/handheld/orientation", "landscape");

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
	Engine::get_singleton()->set_target_fps(GLOBAL_DEF("debug/fps/force_fps", 0));

	GLOBAL_DEF("debug/stdout/print_fps", OS::get_singleton()->is_stdout_verbose());

	if (!OS::get_singleton()->_verbose_stdout) //overrided
		OS::get_singleton()->_verbose_stdout = GLOBAL_DEF("debug/stdout/verbose_stdout", false);

	if (frame_delay == 0) {
		frame_delay = GLOBAL_DEF("application/frame_delay_msec", 0);
	}

	Engine::get_singleton()->set_frame_delay(frame_delay);

	message_queue = memnew(MessageQueue);

	GlobalConfig::get_singleton()->register_global_defaults();

	if (p_second_phase)
		return setup2();

	return OK;

error:

	video_driver = "";
	audio_driver = "";
	game_path = "";

	args.clear();
	main_args.clear();

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

Error Main::setup2() {

	OS::get_singleton()->initialize(video_mode, video_driver_idx, audio_driver_idx);
	if (init_use_custom_pos) {
		OS::get_singleton()->set_window_position(init_custom_pos);
	}

	//right moment to create and initialize the audio server

	audio_server = memnew(AudioServer);
	audio_server->init();

	OS::get_singleton()->set_use_vsync(use_vsync);

	register_core_singletons();

	MAIN_PRINT("Main: Setup Logo");

	bool show_logo = true;
#ifdef JAVASCRIPT_ENABLED
	show_logo = false;
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
	MAIN_PRINT("Main: Load Remaps");

	Color clear = GLOBAL_DEF("rendering/viewport/default_clear_color", Color(0.3, 0.3, 0.3));
	VisualServer::get_singleton()->set_default_clear_color(clear);

	if (show_logo) { //boot logo!
		String boot_logo_path = GLOBAL_DEF("application/boot_splash", String());
		bool boot_logo_scale = GLOBAL_DEF("application/boot_splash_fullsize", true);
		GlobalConfig::get_singleton()->set_custom_property_info("application/boot_splash", PropertyInfo(Variant::STRING, "application/boot_splash", PROPERTY_HINT_FILE, "*.png"));

		Image boot_logo;

		boot_logo_path = boot_logo_path.strip_edges();

		if (boot_logo_path != String() /*&& FileAccess::exists(boot_logo_path)*/) {
			print_line("Boot splash path: " + boot_logo_path);
			Error err = boot_logo.load(boot_logo_path);
			if (err)
				ERR_PRINTS("Non-existing or invalid boot splash at: " + boot_logo_path + ". Loading default splash.");
		}

		if (!boot_logo.empty()) {
			OS::get_singleton()->_msec_splash = OS::get_singleton()->get_ticks_msec();
			Color boot_bg = GLOBAL_DEF("application/boot_bg_color", clear);
			VisualServer::get_singleton()->set_boot_image(boot_logo, boot_bg, boot_logo_scale);
#ifndef TOOLS_ENABLED
//no tools, so free the boot logo (no longer needed)
//GlobalConfig::get_singleton()->set("application/boot_logo",Image());
#endif

		} else {
#ifndef NO_DEFAULT_BOOT_LOGO

			MAIN_PRINT("Main: Create bootsplash");
			Image splash(boot_splash_png);

			MAIN_PRINT("Main: ClearColor");
			VisualServer::get_singleton()->set_default_clear_color(boot_splash_bg_color);
			MAIN_PRINT("Main: Image");
			VisualServer::get_singleton()->set_boot_image(splash, boot_splash_bg_color, false);
#endif
		}

		Image icon(app_icon_png);
		OS::get_singleton()->set_icon(icon);
	}

	MAIN_PRINT("Main: DCC");
	VisualServer::get_singleton()->set_default_clear_color(GLOBAL_DEF("rendering/viewport/default_clear_color", Color(0.3, 0.3, 0.3)));
	MAIN_PRINT("Main: END");

	GLOBAL_DEF("application/icon", String());
	GlobalConfig::get_singleton()->set_custom_property_info("application/icon", PropertyInfo(Variant::STRING, "application/icon", PROPERTY_HINT_FILE, "*.png,*.webp"));

	if (bool(GLOBAL_DEF("display/handheld/emulate_touchscreen", false))) {
		if (!OS::get_singleton()->has_touchscreen_ui_hint() && Input::get_singleton() && !editor) {
			//only if no touchscreen ui hint, set emulation
			InputDefault *id = Input::get_singleton()->cast_to<InputDefault>();
			if (id)
				id->set_emulate_touch(true);
		}
	}

	MAIN_PRINT("Main: Load Remaps");

	MAIN_PRINT("Main: Load Scene Types");

	register_scene_types();
	register_server_types();

	GLOBAL_DEF("display/mouse_cursor/custom_image", String());
	GLOBAL_DEF("display/mouse_cursor/custom_image_hotspot", Vector2());
	GlobalConfig::get_singleton()->set_custom_property_info("display/mouse_cursor/custom_image", PropertyInfo(Variant::STRING, "display/mouse_cursor/custom_image", PROPERTY_HINT_FILE, "*.png,*.webp"));

	if (String(GlobalConfig::get_singleton()->get("display/mouse_cursor/custom_image")) != String()) {

		//print_line("use custom cursor");
		Ref<Texture> cursor = ResourceLoader::load(GlobalConfig::get_singleton()->get("display/mouse_cursor/custom_image"));
		if (cursor.is_valid()) {
			//print_line("loaded ok");
			Vector2 hotspot = GlobalConfig::get_singleton()->get("display/mouse_cursor/custom_image_hotspot");
			Input::get_singleton()->set_custom_mouse_cursor(cursor, hotspot);
		}
	}
#ifdef TOOLS_ENABLED
	ClassDB::set_current_api(ClassDB::API_EDITOR);
	EditorNode::register_editor_types();

	ClassDB::set_current_api(ClassDB::API_CORE);

#endif

	MAIN_PRINT("Main: Load Scripts, Modules, Drivers");

	register_module_types();
	register_driver_types();

	ScriptServer::init_languages();

	MAIN_PRINT("Main: Load Translations");

	translation_server->setup(); //register translations, load them, etc.
	if (locale != "") {

		translation_server->set_locale(locale);
	}
	translation_server->load_translations();

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

	bool editor = false;
	String doc_tool;
	List<String> removal_docs;
	bool doc_base = true;
	String game_path;
	String script;
	String test;
	String screen;
	String _export_platform;
	String _import;
	String _import_script;
	bool noquit = false;
	bool export_debug = false;
	bool project_manager_request = false;
	List<String> args = OS::get_singleton()->get_cmdline_args();
	for (int i = 0; i < args.size(); i++) {
		//parameters that do not have an argument to the right
		if (args[i] == "-nodocbase") {
			doc_base = false;
		} else if (args[i] == "-noquit") {
			noquit = true;
		} else if (args[i] == "-editor" || args[i] == "-e") {
			editor = true;
		} else if (args[i] == "-pm" || args[i] == "-project_manager") {
			project_manager_request = true;
		} else if (args[i].length() && args[i][0] != '-' && game_path == "") {
			game_path = args[i];
		}
		//parameters that have an argument to the right
		else if (i < (args.size() - 1)) {
			bool parsed_pair = true;
			if (args[i] == "-doctool") {
				doc_tool = args[i + 1];
				for (int j = i + 2; j < args.size(); j++)
					removal_docs.push_back(args[j]);
			} else if (args[i] == "-script" || args[i] == "-s") {
				script = args[i + 1];
			} else if (args[i] == "-level" || args[i] == "-l") {
				Engine::get_singleton()->_custom_level = args[i + 1];
			} else if (args[i] == "-test") {
				test = args[i + 1];
			} else if (args[i] == "-export") {
				editor = true; //needs editor
				_export_platform = args[i + 1];
			} else if (args[i] == "-export_debug") {
				editor = true; //needs editor
				_export_platform = args[i + 1];
				export_debug = true;
			} else if (args[i] == "-import") {
				editor = true; //needs editor
				_import = args[i + 1];
			} else if (args[i] == "-import_script") {
				editor = true; //needs editor
				_import_script = args[i + 1];
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

		DocData doc;
		doc.generate(doc_base);

		DocData docsrc;
		if (docsrc.load(doc_tool) == OK) {
			print_line("Doc exists. Merging..");
			doc.merge_from(docsrc);
		} else {
			print_line("No Doc exists. Generating empty.");
		}

		for (List<String>::Element *E = removal_docs.front(); E; E = E->next()) {
			DocData rmdoc;
			if (rmdoc.load(E->get()) == OK) {
				print_line(String("Removing classes in ") + E->get());
				doc.remove_from(rmdoc);
			}
		}

		doc.save(doc_tool);

		return false;
	}

#endif

	if (_export_platform != "") {
		if (game_path == "") {
			String err = "Command line param ";
			err += export_debug ? "-export_debug" : "-export";
			err += " passed but no destination path given.\n";
			err += "Please specify the binary's file path to export to. Aborting export.";
			ERR_PRINT(err.utf8().get_data());
			return false;
		}
	}

	if (script == "" && game_path == "" && String(GLOBAL_DEF("application/main_scene", "")) != "") {
		game_path = GLOBAL_DEF("application/main_scene", "");
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
			MainLoop *script_loop = obj ? obj->cast_to<MainLoop>() : NULL;
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
		main_loop_type = GLOBAL_DEF("application/main_loop_type", "");
	}

	if (!main_loop && main_loop_type == "")
		main_loop_type = "SceneTree";

	if (!main_loop) {
		if (!ClassDB::class_exists(main_loop_type)) {
			OS::get_singleton()->alert("godot: error: MainLoop type doesn't exist: " + main_loop_type);
			return false;
		} else {

			Object *ml = ClassDB::instance(main_loop_type);
			if (!ml) {
				ERR_EXPLAIN("Can't instance MainLoop type");
				ERR_FAIL_V(false);
			}

			main_loop = ml->cast_to<MainLoop>();
			if (!main_loop) {

				memdelete(ml);
				ERR_EXPLAIN("Invalid MainLoop type");
				ERR_FAIL_V(false);
			}
		}
	}

	if (main_loop->is_class("SceneTree")) {

		SceneTree *sml = main_loop->cast_to<SceneTree>();

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

			if (_export_platform != "") {

				editor_node->export_platform(_export_platform, game_path, export_debug, "", true);
				game_path = ""; //no load anything
			}
		}
#endif

		if (!editor) {
			//standard helpers that can be changed from main config

			String stretch_mode = GLOBAL_DEF("display/stretch/mode", "disabled");
			String stretch_aspect = GLOBAL_DEF("display/stretch/aspect", "ignore");
			Size2i stretch_size = Size2(GLOBAL_DEF("display/screen/width", 0), GLOBAL_DEF("display/screen/height", 0));

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

			sml->set_screen_stretch(sml_sm, sml_aspect, stretch_size);

			sml->set_auto_accept_quit(GLOBAL_DEF("application/auto_accept_quit", true));
			sml->set_quit_on_go_back(GLOBAL_DEF("application/quit_on_go_back", true));
			String appname = GlobalConfig::get_singleton()->get("application/name");
			appname = TranslationServer::get_singleton()->translate(appname);
			OS::get_singleton()->set_window_title(appname);

			int shadow_atlas_size = GLOBAL_DEF("rendering/shadow_atlas/size", 2048);
			int shadow_atlas_q0_subdiv = GLOBAL_DEF("rendering/shadow_atlas/quadrant_0_subdiv", 2);
			int shadow_atlas_q1_subdiv = GLOBAL_DEF("rendering/shadow_atlas/quadrant_1_subdiv", 2);
			int shadow_atlas_q2_subdiv = GLOBAL_DEF("rendering/shadow_atlas/quadrant_2_subdiv", 3);
			int shadow_atlas_q3_subdiv = GLOBAL_DEF("rendering/shadow_atlas/quadrant_3_subdiv", 4);

			sml->get_root()->set_shadow_atlas_size(shadow_atlas_size);
			sml->get_root()->set_shadow_atlas_quadrant_subdiv(0, Viewport::ShadowAtlasQuadrantSubdiv(shadow_atlas_q0_subdiv));
			sml->get_root()->set_shadow_atlas_quadrant_subdiv(1, Viewport::ShadowAtlasQuadrantSubdiv(shadow_atlas_q1_subdiv));
			sml->get_root()->set_shadow_atlas_quadrant_subdiv(2, Viewport::ShadowAtlasQuadrantSubdiv(shadow_atlas_q2_subdiv));
			sml->get_root()->set_shadow_atlas_quadrant_subdiv(3, Viewport::ShadowAtlasQuadrantSubdiv(shadow_atlas_q3_subdiv));

		} else {
			GLOBAL_DEF("display/stretch/mode", "disabled");
			GlobalConfig::get_singleton()->set_custom_property_info("display/stretch/mode", PropertyInfo(Variant::STRING, "display/stretch/mode", PROPERTY_HINT_ENUM, "disabled,2d,viewport"));
			GLOBAL_DEF("display/stretch/aspect", "ignore");
			GlobalConfig::get_singleton()->set_custom_property_info("display/stretch/aspect", PropertyInfo(Variant::STRING, "display/stretch/aspect", PROPERTY_HINT_ENUM, "ignore,keep,keep_width,keep_height"));
			sml->set_auto_accept_quit(GLOBAL_DEF("application/auto_accept_quit", true));
			sml->set_quit_on_go_back(GLOBAL_DEF("application/quit_on_go_back", true));

			GLOBAL_DEF("rendering/shadow_atlas/size", 2048);
			GlobalConfig::get_singleton()->set_custom_property_info("rendering/shadow_atlas/size", PropertyInfo(Variant::INT, "rendering/shadow_atlas/size", PROPERTY_HINT_RANGE, "256,16384"));

			GLOBAL_DEF("rendering/shadow_atlas/quadrant_0_subdiv", 2);
			GLOBAL_DEF("rendering/shadow_atlas/quadrant_1_subdiv", 2);
			GLOBAL_DEF("rendering/shadow_atlas/quadrant_2_subdiv", 3);
			GLOBAL_DEF("rendering/shadow_atlas/quadrant_3_subdiv", 4);
			GlobalConfig::get_singleton()->set_custom_property_info("rendering/shadow_atlas/quadrant_0_subdiv", PropertyInfo(Variant::INT, "rendering/shadow_atlas/quadrant_0_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
			GlobalConfig::get_singleton()->set_custom_property_info("rendering/shadow_atlas/quadrant_1_subdiv", PropertyInfo(Variant::INT, "rendering/shadow_atlas/quadrant_1_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
			GlobalConfig::get_singleton()->set_custom_property_info("rendering/shadow_atlas/quadrant_2_subdiv", PropertyInfo(Variant::INT, "rendering/shadow_atlas/quadrant_2_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
			GlobalConfig::get_singleton()->set_custom_property_info("rendering/shadow_atlas/quadrant_3_subdiv", PropertyInfo(Variant::INT, "rendering/shadow_atlas/quadrant_3_subdiv", PROPERTY_HINT_ENUM, "Disabled,1 Shadow,4 Shadows,16 Shadows,64 Shadows,256 Shadows,1024 Shadows"));
		}

		String local_game_path;
		if (game_path != "" && !project_manager_request) {

			local_game_path = game_path.replace("\\", "/");

			if (!local_game_path.begins_with("res://")) {
				bool absolute = (local_game_path.size() > 1) && (local_game_path[0] == '/' || local_game_path[1] == ':');

				if (!absolute) {

					if (GlobalConfig::get_singleton()->is_using_datapack()) {

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

			local_game_path = GlobalConfig::get_singleton()->localize_path(local_game_path);

#ifdef TOOLS_ENABLED
			if (editor) {

				if (_import != "") {

					//editor_node->import_scene(_import,local_game_path,_import_script);
					if (!noquit)
						sml->quit();
					game_path = ""; //no load anything
				} else {

					Error serr = editor_node->load_scene(local_game_path);
				}
				OS::get_singleton()->set_context(OS::CONTEXT_EDITOR);

				//editor_node->set_edited_scene(game);
			}
#endif
		}

		if (!project_manager_request && !editor) {
			if (game_path != "" || script != "") {
				//autoload
				List<PropertyInfo> props;
				GlobalConfig::get_singleton()->get_property_list(&props);

				//first pass, add the constants so they exist before any script is loaded
				for (List<PropertyInfo>::Element *E = props.front(); E; E = E->next()) {

					String s = E->get().name;
					if (!s.begins_with("autoload/"))
						continue;
					String name = s.get_slicec('/', 1);
					String path = GlobalConfig::get_singleton()->get(s);
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
					String path = GlobalConfig::get_singleton()->get(s);
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

						n = obj->cast_to<Node>();
						n->set_script(s.get_ref_ptr());
					}

					ERR_EXPLAIN("Path in autoload not a node or script: " + path);
					ERR_CONTINUE(!n);
					n->set_name(name);

					//defer so references are all valid on _ready()
					//sml->get_root()->add_child(n);
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
				//sml->get_root()->add_child(scene);
				sml->add_current_scene(scene);

				String iconpath = GLOBAL_DEF("application/icon", "Variant()");
				if (iconpath != "") {
					Image icon;
					if (icon.load(iconpath) == OK)
						OS::get_singleton()->set_icon(icon);
				}
			}
		}

#ifdef TOOLS_ENABLED

		/*if (_export_platform!="") {

			sml->quit();
		}*/

		/*
		if (sml->get_root_node()) {

			Console *console = memnew( Console );

			sml->get_root_node()->cast_to<RootNode>()->set_console(console);
			if (GLOBAL_DEF("console/visible_default",false).operator bool()) {

				console->show();
			} else {P

				console->hide();
			};
		}
*/
		if (project_manager_request || (script == "" && test == "" && game_path == "" && !editor)) {

			ProjectManager *pmanager = memnew(ProjectManager);
			ProgressDialog *progress_dialog = memnew(ProgressDialog);
			pmanager->add_child(progress_dialog);
			sml->get_root()->add_child(pmanager);
			OS::get_singleton()->set_context(OS::CONTEXT_PROJECTMAN);
		}

#endif
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
static uint64_t fixed_process_max = 0;
static uint64_t idle_process_max = 0;

bool Main::iteration() {

	uint64_t ticks = OS::get_singleton()->get_ticks_usec();
	uint64_t ticks_elapsed = ticks - last_ticks;

	double step = (double)ticks_elapsed / 1000000.0;
	float frame_slice = 1.0 / Engine::get_singleton()->get_iterations_per_second();

	/*
	if (time_accum+step < frame_slice)
		return false;
	*/

	uint64_t fixed_process_ticks = 0;
	uint64_t idle_process_ticks = 0;

	frame += ticks_elapsed;

	last_ticks = ticks;

	if (step > frame_slice * 8)
		step = frame_slice * 8;

	time_accum += step;

	float time_scale = Engine::get_singleton()->get_time_scale();

	bool exit = false;

	int iters = 0;

	Engine::get_singleton()->_in_fixed = true;

	while (time_accum > frame_slice) {

		uint64_t fixed_begin = OS::get_singleton()->get_ticks_usec();

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
		/*
		if (AudioServer::get_singleton())
			AudioServer::get_singleton()->update();
		*/

		fixed_process_ticks = MAX(fixed_process_ticks, OS::get_singleton()->get_ticks_usec() - fixed_begin); // keep the largest one for reference
		fixed_process_max = MAX(OS::get_singleton()->get_ticks_usec() - fixed_begin, fixed_process_max);
		iters++;
		Engine::get_singleton()->_fixed_frames++;
	}

	Engine::get_singleton()->_in_fixed = false;

	uint64_t idle_begin = OS::get_singleton()->get_ticks_usec();

	OS::get_singleton()->get_main_loop()->idle(step * time_scale);
	message_queue->flush();

	VisualServer::get_singleton()->sync(); //sync if still drawing from previous frames.

	if (OS::get_singleton()->can_draw()) {

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
			script_debugger->profiling_set_frame_times(USEC_TO_SEC(frame_time), USEC_TO_SEC(idle_process_ticks), USEC_TO_SEC(fixed_process_ticks), frame_slice);
		}
		script_debugger->idle_poll();
	}

	//x11_delay_usec(10000);
	frames++;
	Engine::get_singleton()->_idle_frames++;

	if (frame > 1000000) {

		if (GLOBAL_DEF("debug/stdout/print_fps", OS::get_singleton()->is_stdout_verbose())) {
			print_line("FPS: " + itos(frames));
		};

		Engine::get_singleton()->_fps = frames;
		performance->set_process_time(USEC_TO_SEC(idle_process_max));
		performance->set_fixed_process_time(USEC_TO_SEC(fixed_process_max));
		idle_process_max = 0;
		fixed_process_max = 0;

		frame %= 1000000;
		frames = 0;
	}

	if (OS::get_singleton()->is_in_low_processor_usage_mode() || !OS::get_singleton()->can_draw())
		OS::get_singleton()->delay_usec(16600); //apply some delay to force idle time (results in about 60 FPS max)
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

	if (audio_server) {
		memdelete(audio_server);
	}

#ifdef TOOLS_ENABLED
	EditorNode::unregister_editor_types();
#endif

	unregister_driver_types();
	unregister_module_types();
	unregister_scene_types();
	unregister_server_types();

	OS::get_singleton()->finalize();

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

	memdelete(message_queue);

	unregister_core_driver_types();
	unregister_core_types();

	//PerformanceMetrics::finish();
	OS::get_singleton()->clear_last_error();
	OS::get_singleton()->finalize_core();
}
