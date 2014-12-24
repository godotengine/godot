/*************************************************************************/
/*  main.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "os/os.h"
#include "globals.h"
#include "splash.h"
#include "core/register_core_types.h"
#include "scene/register_scene_types.h"
#include "drivers/register_driver_types.h"
#include "servers/register_server_types.h"
#include "modules/register_module_types.h"
#include "script_debugger_local.h"
#include "script_debugger_remote.h"
#include "message_queue.h"
#include "path_remap.h"
#include "input_map.h"
#include "io/resource_loader.h"
#include "scene/main/scene_main_loop.h"


#include "script_language.h"
#include "io/resource_loader.h"

#include "bin/tests/test_main.h"
#include "os/dir_access.h"
#include "core/io/ip.h"
#include "scene/resources/packed_scene.h"
#include "scene/main/viewport.h"

#ifdef TOOLS_ENABLED
#include "tools/editor/editor_node.h"
#include "tools/editor/project_manager.h"
#include "tools/editor/console.h"
#include "tools/pck/pck_packer.h"
#endif

#include "io/file_access_network.h"
#include "tools/doc/doc_data.h"


#include "servers/spatial_sound_server.h"
#include "servers/spatial_sound_2d_server.h"
#include "servers/physics_2d_server.h"


#include "core/io/stream_peer_tcp.h"
#include "core/os/thread.h"
#include "core/io/file_access_pack.h"
#include "core/io/file_access_zip.h"
#include "translation.h"
#include "version.h"

#include "performance.h"

static Globals *globals=NULL;
static InputMap *input_map=NULL;
static bool _start_success=false;
static ScriptDebugger *script_debugger=NULL;

static MessageQueue *message_queue=NULL;
static Performance *performance = NULL;
static PathRemap *path_remap;
static PackedData *packed_data=NULL;
static FileAccessNetworkClient *file_access_network_client=NULL;
static TranslationServer *translation_server = NULL;

static OS::VideoMode video_mode;
static int video_driver_idx=-1;
static int audio_driver_idx=-1;
static String locale;

static String unescape_cmdline(const String& p_str) {

	return p_str.replace("%20"," ");
}


//#define DEBUG_INIT

#ifdef DEBUG_INIT
#define MAIN_PRINT(m_txt) print_line(m_txt)
#else
#define MAIN_PRINT(m_txt)
#endif

void Main::print_help(const char* p_binary) {

	OS::get_singleton()->print(VERSION_FULL_NAME" (c) 2008-2015 Juan Linietsky, Ariel Manzur.\n");
	OS::get_singleton()->print("Usage: %s [options] [scene]\n",p_binary);
	OS::get_singleton()->print("Options:\n");
	OS::get_singleton()->print("\t-path [dir] : Path to a game, containing engine.cfg\n");
#ifdef TOOLS_ENABLED
	OS::get_singleton()->print("\t-e,-editor : Bring up the editor instead of running the scene.\n");
#endif
	OS::get_singleton()->print("\t-test [test] : Run a test.\n");
	OS::get_singleton()->print("\t\t(");
	const char **test_names=tests_get_names();
	const char* coma = "";
	while(*test_names) {
	
		OS::get_singleton()->print("%s%s", coma, *test_names);
		test_names++;
		coma = ", ";
	}
	OS::get_singleton()->print(")\n");
	
	OS::get_singleton()->print("\t-r WIDTHxHEIGHT\t : Request Screen Resolution\n");
	OS::get_singleton()->print("\t-f\t\t : Request Fullscreen\n");
	OS::get_singleton()->print("\t-vd DRIVER\t : Video Driver (");
	for (int i=0;i<OS::get_singleton()->get_video_driver_count();i++) {
		
		if (i!=0)
			OS::get_singleton()->print(", ");
		OS::get_singleton()->print("%s",OS::get_singleton()->get_video_driver_name(i));
	}
	OS::get_singleton()->print(")\n");
	OS::get_singleton()->print("\t-ad DRIVER\t : Audio Driver (");
	for (int i=0;i<OS::get_singleton()->get_audio_driver_count();i++) {
		
		if (i!=0)
			OS::get_singleton()->print(", ");
		OS::get_singleton()->print("%s",OS::get_singleton()->get_audio_driver_name(i));
	}
    OS::get_singleton()->print(")\n");
	OS::get_singleton()->print("\t-rthread <mode>\t : Render Thread Mode ('unsafe', 'safe', 'separate).");
	OS::get_singleton()->print(")\n");
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
	OS::get_singleton()->print("\t-optimize FILE Save an optimized copy of scene to FILE.\n");
	OS::get_singleton()->print("\t-optimize_preset [preset] Use a given preset for optimization.\n");
	OS::get_singleton()->print("\t-export [target] Export the project using given export target.\n");
#endif
}


Error Main::setup(const char *execpath,int argc, char *argv[],bool p_second_phase) {

	RID_OwnerBase::init_rid();

	OS::get_singleton()->initialize_core();
	ObjectTypeDB::init();

	MAIN_PRINT("Main: Initialize CORE");

	register_core_types();
	register_core_driver_types();

	MAIN_PRINT("Main: Initialize Globals");


	Thread::_main_thread_id = Thread::get_caller_ID();

	globals = memnew( Globals );
	input_map = memnew( InputMap );


	path_remap = memnew( PathRemap );
	translation_server = memnew( TranslationServer );
	performance = memnew( Performance );
	globals->add_singleton(Globals::Singleton("Performance",performance));

	MAIN_PRINT("Main: Parse CMDLine");

	/* argument parsing and main creation */
	List<String> args;
	List<String> main_args;
	
	for(int i=0;i<argc;i++) {

		args.push_back(String::utf8(argv[i]));
	}

	List<String>::Element *I=args.front();

	I=args.front();

	while (I) {

		I->get()=unescape_cmdline(I->get().strip_escapes());
//		print_line("CMD: "+I->get());
		I=I->next();
	}

	I=args.front();
	
	video_mode = OS::get_singleton()->get_default_video_mode();

	String video_driver="";
	String audio_driver="";
	String game_path=".";
	String debug_mode;
	String debug_host;
	String main_pack;
	bool quiet_stdout=false;
	int rtm=-1;

	String remotefs;
	String remotefs_pass;

	String screen = "";

	List<String> pack_list;
	Vector<String> breakpoints;
	bool use_custom_res=true;
	bool force_res=false;

	I=args.front();

	packed_data = PackedData::get_singleton();
	if (!packed_data)
		packed_data = memnew(PackedData);

#ifdef MINIZIP_ENABLED
	packed_data->add_pack_source(ZipArchive::get_singleton());
#endif

	bool editor=false;

	while(I) {

		List<String>::Element *N=I->next();

		if (I->get() == "-noop") {

			// no op
		} else if (I->get()=="-h" || I->get()=="--help" || I->get()=="/?") { // resolution
			
			goto error;
			
			
		} else if (I->get()=="-r") { // resolution
		
			if (I->next()) {
			
				String vm=I->next()->get();
				
				if (vm.find("x")==-1) { // invalid parameter format
				
					goto error;
					
				
				}
				
				int w=vm.get_slice("x",0).to_int();
				int h=vm.get_slice("x",1).to_int();
				
				if (w==0 || h==0) {
				
					goto error;
					
				}
				
				video_mode.width=w;
				video_mode.height=h;
				force_res=true;
				
				N=I->next()->next();
			} else {
				goto error;
				
			
			}
			
		} else if (I->get()=="-vd") { // video driver
		
			if (I->next()) {
			
				video_driver=I->next()->get();
				N=I->next()->next();
			} else {
				goto error;
				
			}
		} else if (I->get()=="-lang") { // language

			if (I->next()) {

				locale=I->next()->get();
				N=I->next()->next();
			} else {
				goto error;

			}
		} else if (I->get()=="-rfs") { // language

			if (I->next()) {

				remotefs=I->next()->get();
				N=I->next()->next();
			} else {
				goto error;

			}
		} else if (I->get()=="-rfs_pass") { // language

			if (I->next()) {

				remotefs_pass=I->next()->get();
				N=I->next()->next();
			} else {
				goto error;

			}
		} else if (I->get()=="-rthread") { // language

			if (I->next()) {

				if (I->next()->get()=="safe")
					rtm=OS::RENDER_THREAD_SAFE;
				else if (I->next()->get()=="unsafe")
					rtm=OS::RENDER_THREAD_UNSAFE;
				else if (I->next()->get()=="separate")
					rtm=OS::RENDER_SEPARATE_THREAD;


				N=I->next()->next();
			} else {
				goto error;

			}

		} else if (I->get()=="-ad") { // video driver
		
			if (I->next()) {
			
				audio_driver=I->next()->get();
				N=I->next()->next();
			} else {
				goto error;
				
			}
			
		} else if (I->get()=="-f") { // fullscreen
		
			video_mode.fullscreen=true;
		} else if (I->get()=="-e" || I->get()=="-editor") { // fonud editor

			editor=true;

		} else if (I->get()=="-nowindow") { // fullscreen

			OS::get_singleton()->set_no_window_mode(true);
		} else if (I->get()=="-quiet") { // fullscreen

			quiet_stdout=true;
		} else if (I->get()=="-v") { // fullscreen
			OS::get_singleton()->_verbose_stdout=true;
		} else if (I->get()=="-path") { // resolution
		
			if (I->next()) {
			
				String p = I->next()->get();
				if (OS::get_singleton()->set_cwd(p)==OK) {
					//nothing
				} else {
					game_path=I->next()->get(); //use game_path instead
				}

				N=I->next()->next();
			} else {
				goto error;
				
			}
		} else if (I->get()=="-bp") { // /breakpoints

			if (I->next()) {

				String bplist = I->next()->get();
				breakpoints= bplist.split(",");
				N=I->next()->next();
			} else {
				goto error;

			}


		} else if (I->get()=="-fdelay") { // resolution

			if (I->next()) {

				OS::get_singleton()->set_frame_delay(I->next()->get().to_int());
				N=I->next()->next();
			} else {
				goto error;

			}

		} else if (I->get()=="-timescale") { // resolution

			if (I->next()) {

				OS::get_singleton()->set_time_scale(I->next()->get().to_double());
				N=I->next()->next();
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

				main_pack=I->next()->get();
				N = I->next()->next();
			} else {

				goto error;
			};

		} else if (I->get()=="-debug" || I->get()=="-d") {
			debug_mode="local";
		} else if (I->get()=="-editor_scene") {

			if (I->next()) {

				Globals::get_singleton()->set("editor_scene",game_path=I->next()->get());
			} else {
				goto error;

			}

		} else if (I->get()=="-rdebug") {
			if (I->next()) {

				debug_mode="remote";
				debug_host=I->next()->get();
				if (debug_host.find(":")==-1) //wrong host
					goto error;
				N=I->next()->next();
			} else {
				goto error;

			}
		} else {

			//test for game path
			bool gpfound=false;

			if (!I->get().begins_with("-") && game_path=="") {
				DirAccess* da = DirAccess::open(I->get());
				if (da!=NULL) {
					game_path=I->get();
					gpfound=true;
					memdelete(da);
				}

			}

			if (!gpfound) {
				main_args.push_back(I->get());
			}
		}
			
		I=N;
	}



	if (debug_mode == "remote") {

		ScriptDebuggerRemote *sdr = memnew( ScriptDebuggerRemote );
		uint16_t debug_port = GLOBAL_DEF("debug/remote_port",6007);
		if (debug_host.find(":")!=-1) {
		    debug_port=debug_host.get_slice(":",1).to_int();
		    debug_host=debug_host.get_slice(":",0);
		}
		Error derr = sdr->connect_to_host(debug_host,debug_port);

		if (derr!=OK) {
			memdelete(sdr);
		} else {
			script_debugger=sdr;

		}
	} else if (debug_mode=="local") {

		script_debugger = memnew( ScriptDebuggerLocal );
	}


	if (remotefs!="") {

		file_access_network_client=memnew(FileAccessNetworkClient);
		int port;
		if (remotefs.find(":")!=-1) {
			port=remotefs.get_slice(":",1).to_int();
			remotefs=remotefs.get_slice(":",0);
		} else {
			port=6010;
		}

		Error err = file_access_network_client->connect(remotefs,port,remotefs_pass);
		if (err) {
			OS::get_singleton()->printerr("Could not connect to remotefs: %s:%i\n",remotefs.utf8().get_data(),port);
			goto error;
		}

		FileAccess::make_default<FileAccessNetwork>(FileAccess::ACCESS_RESOURCES);
	}
	if (script_debugger) {
		//there is a debugger, parse breakpoints

		for(int i=0;i<breakpoints.size();i++) {

			String bp = breakpoints[i];
			int sp=bp.find_last(":");
			if (sp==-1) {
				ERR_EXPLAIN("Invalid breakpoint: '"+bp+"', expected file:line format.");
				ERR_CONTINUE(sp==-1);
			}

			script_debugger->insert_breakpoint(bp.substr(sp+1,bp.length()).to_int(),bp.substr(0,sp));
		}
	}


#ifdef TOOLS_ENABLED
	if (editor) {
		packed_data->set_disabled(true);
		globals->set_disable_platform_override(true);
	}

#endif


	if (globals->setup(game_path,main_pack)!=OK) {
		
#ifdef TOOLS_ENABLED
		editor=false;
#else
		OS::get_singleton()->print("error: Couldn't load game path '%s'\n",game_path.ascii().get_data());

		goto error;
#endif
	}

	if (editor) {
		main_args.push_back("-editor");
		use_custom_res=false;
	}

	if (bool(Globals::get_singleton()->get("application/disable_stdout"))) {
		quiet_stdout=true;
	}

	if (quiet_stdout)
		_print_line_enabled=false;

	OS::get_singleton()->set_cmdline(execpath, main_args);

#ifdef TOOLS_ENABLED

	if (main_args.size()==0 && (!Globals::get_singleton()->has("application/main_loop_type")) && (!Globals::get_singleton()->has("application/main_scene") || String(Globals::get_singleton()->get("application/main_scene"))==""))
		use_custom_res=false; //project manager (run without arguments)

#endif

	input_map->load_from_globals();

	if (video_driver=="") // specified in engine.cfg
		video_driver=_GLOBAL_DEF("display/driver",Variant((const char*)OS::get_singleton()->get_video_driver_name(0)));

	if (!force_res && use_custom_res && globals->has("display/width"))
		video_mode.width=globals->get("display/width");
	if (!force_res &&use_custom_res && globals->has("display/height"))
		video_mode.height=globals->get("display/height");
	if (use_custom_res && globals->has("display/fullscreen"))
		video_mode.fullscreen=globals->get("display/fullscreen");
	if (use_custom_res && globals->has("display/resizable"))
		video_mode.resizable=globals->get("display/resizable");

	if (!force_res && use_custom_res && globals->has("display/test_width") && globals->has("display/test_height")) {
		int tw = globals->get("display/test_width");
		int th = globals->get("display/test_height");
		if (tw>0 && th>0) {
			video_mode.width=tw;
			video_mode.height=th;
		}
	}


	GLOBAL_DEF("display/width",video_mode.width);
	GLOBAL_DEF("display/height",video_mode.height);
	GLOBAL_DEF("display/fullscreen",video_mode.fullscreen);
	GLOBAL_DEF("display/resizable",video_mode.resizable);
	GLOBAL_DEF("display/test_width",0);
	GLOBAL_DEF("display/test_height",0);
	if (rtm==-1) {
		rtm=GLOBAL_DEF("render/thread_model",OS::RENDER_THREAD_SAFE);
	}

	if (rtm>=0 && rtm<3)
		OS::get_singleton()->_render_thread_mode=OS::RenderThreadMode(rtm);



	/* Determine Video Driver */

	if (audio_driver=="") // specified in engine.cfg
		audio_driver=GLOBAL_DEF("audio/driver",OS::get_singleton()->get_audio_driver_name(0));
		
	
	for (int i=0;i<OS::get_singleton()->get_video_driver_count();i++) {

		if (video_driver==OS::get_singleton()->get_video_driver_name(i)) {
		
			video_driver_idx=i;
			break;
		}
	}

	if (video_driver_idx<0) {
	
		OS::get_singleton()->alert( "Invalid Video Driver: "+video_driver );
		video_driver_idx = 0;
		//goto error;
	}

	for (int i=0;i<OS::get_singleton()->get_audio_driver_count();i++) {
	
		if (audio_driver==OS::get_singleton()->get_audio_driver_name(i)) {
		
			audio_driver_idx=i;
			break;
		}
	}

	if (audio_driver_idx<0) {
	
		OS::get_singleton()->alert( "Invalid Audio Driver: "+audio_driver );
		goto error;
	}

	{
		String orientation = GLOBAL_DEF("display/orientation","landscape");

		if (orientation=="portrait")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_PORTRAIT);
		else if (orientation=="reverse_landscape")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_REVERSE_LANDSCAPE);
		else if (orientation=="reverse_portrait")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_REVERSE_PORTRAIT);
		else if (orientation=="sensor_landscape")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_SENSOR_LANDSCAPE);
		else if (orientation=="sensor_portrait")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_SENSOR_PORTRAIT);
		else if (orientation=="sensor")
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_SENSOR);
		else
			OS::get_singleton()->set_screen_orientation(OS::SCREEN_LANDSCAPE);
	}

	OS::get_singleton()->set_iterations_per_second(GLOBAL_DEF("physics/fixed_fps",60));
	OS::get_singleton()->set_target_fps(GLOBAL_DEF("application/target_fps",0));

	if (!OS::get_singleton()->_verbose_stdout) //overrided
		OS::get_singleton()->_verbose_stdout=GLOBAL_DEF("debug/verbose_stdout",false);

	message_queue = memnew( MessageQueue );

	Globals::get_singleton()->register_global_defaults();

	if (p_second_phase)
		return setup2();

	return OK;

	error:
	
	video_driver="";	
	audio_driver="";
	game_path="";
		
	args.clear(); 
	main_args.clear();
	
	print_help(execpath);
	

	if (performance)
		memdelete(performance);
	if (input_map)
		memdelete(input_map);
	if (translation_server)
		memdelete( translation_server );
	if (globals)
		memdelete(globals);
	if (script_debugger)
		memdelete(script_debugger);
	if (packed_data)
		memdelete(packed_data);
	if (file_access_network_client)
		memdelete(file_access_network_client);
	unregister_core_types();
	
	OS::get_singleton()->_cmdline.clear();

	if (message_queue)
		memdelete( message_queue);
	OS::get_singleton()->finalize_core();
	locale=String();
	
	return ERR_INVALID_PARAMETER;
}

Error Main::setup2() {


	OS::get_singleton()->initialize(video_mode,video_driver_idx,audio_driver_idx);

	register_core_singletons();

	MAIN_PRINT("Main: Setup Logo");

	bool show_logo=true;
#ifdef JAVASCRIPT_ENABLED
	show_logo=false;
#endif

	if (show_logo) { //boot logo!
		Image boot_logo=GLOBAL_DEF("application/boot_logo",Image());

		if (!boot_logo.empty()) {
			Color clear = GLOBAL_DEF("render/default_clear_color",Color(0.3,0.3,0.3));
			VisualServer::get_singleton()->set_default_clear_color(clear);
			Color boot_bg = GLOBAL_DEF("application/boot_bg_color", clear);
			VisualServer::get_singleton()->set_boot_image(boot_logo, boot_bg);
#ifndef TOOLS_ENABLED
			//no tools, so free the boot logo (no longer needed)
			Globals::get_singleton()->set("application/boot_logo",Image());
#endif

		} else {
#ifndef NO_DEFAULT_BOOT_LOGO

			MAIN_PRINT("Main: Create botsplash");
			Image splash(boot_splash_png);

			MAIN_PRINT("Main: ClearColor");
			VisualServer::get_singleton()->set_default_clear_color(boot_splash_bg_color);
			MAIN_PRINT("Main: Image");
			VisualServer::get_singleton()->set_boot_image(splash, boot_splash_bg_color);
#endif
			MAIN_PRINT("Main: DCC");
			VisualServer::get_singleton()->set_default_clear_color(GLOBAL_DEF("render/default_clear_color",Color(0.3,0.3,0.3)));
			MAIN_PRINT("Main: END");
		}

		Image icon(app_icon_png);
		OS::get_singleton()->set_icon(icon);
	}
	GLOBAL_DEF("application/icon",String());
	Globals::get_singleton()->set_custom_property_info("application/icon",PropertyInfo(Variant::STRING,"application/icon",PROPERTY_HINT_FILE,"*.png,*.webp"));

	MAIN_PRINT("Main: Load Remaps");

	path_remap->load_remaps();

	MAIN_PRINT("Main: Load Scene Types");

	register_scene_types();
	register_server_types();

#ifdef TOOLS_ENABLED
	EditorNode::register_editor_types();
	ObjectTypeDB::register_type<PCKPacker>(); // todo: move somewhere else
#endif

	MAIN_PRINT("Main: Load Scripts, Modules, Drivers");

	register_module_types();
	register_driver_types();

	ScriptServer::init_languages();

	MAIN_PRINT("Main: Load Translations");

	translation_server->setup(); //register translations, load them, etc.
	if (locale!="") {

		translation_server->set_locale(locale);
	}
	translation_server->load_translations();



	_start_success=true;
	locale=String();

	MAIN_PRINT("Main: Done");

	return OK;

}



bool Main::start() {

	ERR_FAIL_COND_V(!_start_success,false);

	bool editor=false;
	String doc_tool;
	bool doc_base=true;
	String game_path;
	String script;
	String test;
	String screen;
	String optimize;
	String optimize_preset;
	String _export_platform;
	String _import;
	String _import_script;
	String dumpstrings;
	bool noquit=false;
	bool convert_old=false;
	bool export_debug=false;
	List<String> args = OS::get_singleton()->get_cmdline_args();
	for (int i=0;i<args.size();i++) {
		

		if (args[i]=="-doctool" && i <(args.size()-1)) {

			doc_tool=args[i+1];
			i++;
		}else if (args[i]=="-nodocbase") {

			doc_base=false;
		} else if ((args[i]=="-script" || args[i]=="-s") && i <(args.size()-1)) {
		
			script=args[i+1];
			i++;
		} else if ((args[i]=="-level" || args[i]=="-l") && i <(args.size()-1)) {

			OS::get_singleton()->_custom_level=args[i+1];
			i++;
		} else if (args[i]=="-test" && i <(args.size()-1)) {
			test=args[i+1];
			i++;
		} else if (args[i]=="-optimize" && i <(args.size()-1)) {
			optimize=args[i+1];
			i++;
		} else if (args[i]=="-optimize_preset" && i <(args.size()-1)) {
			optimize_preset=args[i+1];
			i++;
		} else if (args[i]=="-export" && i <(args.size()-1)) {
			editor=true; //needs editor
			_export_platform=args[i+1];
			i++;
		} else if (args[i]=="-export_debug" && i <(args.size()-1)) {
			editor=true; //needs editor
			_export_platform=args[i+1];
			export_debug=true;
			i++;
		} else if (args[i]=="-import" && i <(args.size()-1)) {
			editor=true; //needs editor
			_import=args[i+1];
			i++;
		} else if (args[i]=="-import_script" && i <(args.size()-1)) {
			editor=true; //needs editor
			_import_script=args[i+1];
			i++;
		} else if (args[i]=="-noquit" ) {
			noquit=true;
		} else if (args[i]=="-dumpstrings" && i <(args.size()-1)) {
			editor=true; //needs editor
			dumpstrings=args[i+1];
			i++;
		} else if (args[i]=="-editor" || args[i]=="-e") {
			editor=true;
		} else if (args[i]=="-convert_old") {
			convert_old=true;
		} else if (args[i].length() && args[i][0] != '-' && game_path == "") {

			game_path=args[i];
		}
	}

	if (editor)
		Globals::get_singleton()->set("editor_active",true);


	String main_loop_type;
#ifdef TOOLS_ENABLED
	if(doc_tool!="") {

		DocData doc;
		doc.generate(doc_base);

		DocData docsrc;
		if (docsrc.load(doc_tool)==OK) {
			print_line("Doc exists. Merging..");
			doc.merge_from(docsrc);
		} else {
			print_line("No Doc exists. Generating empty.");

		}

		doc.save(doc_tool);

		return false;
	}

	if (optimize!="")
		editor=true; //need editor



#endif

	if(script=="" && game_path=="" && !editor && String(GLOBAL_DEF("application/main_scene",""))!="") {
		game_path=GLOBAL_DEF("application/main_scene","");
	}


	MainLoop *main_loop=NULL;
	if (editor) {
		main_loop = memnew(SceneTree);
	};

	if (test!="") {
#ifdef DEBUG_ENABLED
		main_loop = test_main(test,args);

		if (!main_loop)
			return false;

#endif

	} else if (script!="") {

		Ref<Script> script_res = ResourceLoader::load(script);
		ERR_EXPLAIN("Can't load script: "+script);
		ERR_FAIL_COND_V(script_res.is_null(),false);
		
		if( script_res->can_instance() /*&& script_res->inherits_from("SceneTreeScripted")*/) {
		

			StringName instance_type=script_res->get_instance_base_type();
			Object *obj = ObjectTypeDB::instance(instance_type);
			MainLoop *script_loop = obj?obj->cast_to<MainLoop>():NULL;
			if (!script_loop) {
				if (obj)
					memdelete(obj);
				ERR_EXPLAIN("Can't load script '"+script+"', it does not inherit from a MainLoop type");
				ERR_FAIL_COND_V(!script_loop,false);
			}


			script_loop->set_init_script(script_res);
			main_loop=script_loop;
		} else {

			return false;
		}

	} else {
		main_loop_type=GLOBAL_DEF("application/main_loop_type","");
	}
	
	if (!main_loop && main_loop_type=="")
		main_loop_type="SceneTree";
	
	if (!main_loop) {
		if (!ObjectTypeDB::type_exists(main_loop_type)) {
			OS::get_singleton()->alert("godot: error: MainLoop type doesn't exist: "+main_loop_type);
			return false;
		} else {

			Object *ml = ObjectTypeDB::instance(main_loop_type);
			if (!ml) {
				ERR_EXPLAIN("Can't instance MainLoop type");
				ERR_FAIL_V(false);
			}

			main_loop=ml->cast_to<MainLoop>();
			if (!main_loop) {
				
				memdelete(ml);
				ERR_EXPLAIN("Invalid MainLoop type");
				ERR_FAIL_V(false);
				
			}
		}
	}

	if (main_loop->is_type("SceneTree")) {
		
		SceneTree *sml = main_loop->cast_to<SceneTree>();

#ifdef TOOLS_ENABLED

		EditorNode *editor_node=NULL;
		if (editor) {

			editor_node = memnew( EditorNode );			
			sml->get_root()->add_child(editor_node);

			//root_node->set_editor(editor);
			//startup editor

			if (_export_platform!="") {

				editor_node->export_platform(_export_platform,game_path,export_debug,"",true);
				game_path=""; //no load anything
			}
		}
#endif

		if (!editor) {
			//standard helpers that can be changed from main config

			String stretch_mode = GLOBAL_DEF("display/stretch_mode","disabled");
			String stretch_aspect = GLOBAL_DEF("display/stretch_aspect","ignore");
			Size2i stretch_size = Size2(GLOBAL_DEF("display/width",0),GLOBAL_DEF("display/height",0));

			SceneTree::StretchMode sml_sm=SceneTree::STRETCH_MODE_DISABLED;
			if (stretch_mode=="2d")
				sml_sm=SceneTree::STRETCH_MODE_2D;
			else if (stretch_mode=="viewport")
				sml_sm=SceneTree::STRETCH_MODE_VIEWPORT;

			SceneTree::StretchAspect sml_aspect=SceneTree::STRETCH_ASPECT_IGNORE;
			if (stretch_aspect=="keep")
				sml_aspect=SceneTree::STRETCH_ASPECT_KEEP;
			else if (stretch_aspect=="keep_width")
				sml_aspect=SceneTree::STRETCH_ASPECT_KEEP_WIDTH;
			else if (stretch_aspect=="keep_height")
				sml_aspect=SceneTree::STRETCH_ASPECT_KEEP_HEIGHT;

			sml->set_screen_stretch(sml_sm,sml_aspect,stretch_size);

			sml->set_auto_accept_quit(GLOBAL_DEF("application/auto_accept_quit",true));
			String appname = Globals::get_singleton()->get("application/name");
			appname = TranslationServer::get_singleton()->translate(appname);
			OS::get_singleton()->set_window_title(appname);


		} else {
			GLOBAL_DEF("display/stretch_mode","disabled");
			Globals::get_singleton()->set_custom_property_info("display/stretch_mode",PropertyInfo(Variant::STRING,"display/stretch_mode",PROPERTY_HINT_ENUM,"disabled,2d,viewport"));
			GLOBAL_DEF("display/stretch_aspect","ignore");
			Globals::get_singleton()->set_custom_property_info("display/stretch_aspect",PropertyInfo(Variant::STRING,"display/stretch_aspect",PROPERTY_HINT_ENUM,"ignore,keep,keep_width,keep_height"));
			sml->set_auto_accept_quit(GLOBAL_DEF("application/auto_accept_quit",true));


		}


		if (game_path!="") {

			String local_game_path=game_path.replace("\\","/");

			if (!local_game_path.begins_with("res://")) {
				bool absolute=(local_game_path.size()>1) && (local_game_path[0]=='/' || local_game_path[1]==':');

				if (!absolute) {

					if (Globals::get_singleton()->is_using_datapack()) {

						local_game_path="res://"+local_game_path;

					} else {
						int sep=local_game_path.find_last("/");

						if (sep==-1) {
							DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
							local_game_path=da->get_current_dir()+"/"+local_game_path;
							memdelete(da)						;
						} else {

							DirAccess *da = DirAccess::open(local_game_path.substr(0,sep));
							if (da) {
								local_game_path=da->get_current_dir()+"/"+local_game_path.substr(sep+1,local_game_path.length());;
								memdelete(da);
							}
						}
					}

				}
			}

			local_game_path=Globals::get_singleton()->localize_path(local_game_path);

#ifdef TOOLS_ENABLED
			if (editor) {


				if (_import!="") {

					//editor_node->import_scene(_import,local_game_path,_import_script);
					if (!noquit)
						sml->quit();
					game_path=""; //no load anything
				} else {

					Error serr = editor_node->load_scene(local_game_path);

					if (serr==OK) {

						if (optimize!="") {

							editor_node->save_optimized_copy(optimize,optimize_preset);
							if (!noquit)
								sml->quit();
						}

						if (dumpstrings!="") {

							editor_node->save_translatable_strings(dumpstrings);
							if (!noquit)
								sml->quit();
						}
					}
				}

				//editor_node->set_edited_scene(game);
			} else {
#endif

				{
					//autoload
					List<PropertyInfo> props;
					Globals::get_singleton()->get_property_list(&props);
					for(List<PropertyInfo>::Element *E=props.front();E;E=E->next()) {

						String s = E->get().name;
						if (!s.begins_with("autoload/"))
							continue;
						String name = s.get_slice("/",1);
						String path = Globals::get_singleton()->get(s);
						RES res = ResourceLoader::load(path);
						ERR_EXPLAIN("Can't autoload: "+path);
						ERR_CONTINUE(res.is_null());
						Node *n=NULL;
						if (res->is_type("PackedScene")) {
							Ref<PackedScene> ps = res;
							n=ps->instance();
						} else if (res->is_type("Script")) {
							Ref<Script> s = res;
							StringName ibt = s->get_instance_base_type();
							ERR_EXPLAIN("Script does not inherit a Node: "+path);
							ERR_CONTINUE( !ObjectTypeDB::is_type(ibt,"Node") );

							Object *obj = ObjectTypeDB::instance(ibt);

							ERR_EXPLAIN("Cannot instance node for autoload type: "+String(ibt));
							ERR_CONTINUE( obj==NULL );

							n = obj->cast_to<Node>();
							n->set_script(s.get_ref_ptr());
						}

						ERR_EXPLAIN("Path in autoload not a node or script: "+path);
						ERR_CONTINUE(!n);
						n->set_name(name);
						sml->get_root()->add_child(n);
					}

				}

				Node *scene=NULL;
				Ref<PackedScene> scenedata = ResourceLoader::load(local_game_path);
				if (scenedata.is_valid())
					scene=scenedata->instance();

				ERR_EXPLAIN("Failed loading scene: "+local_game_path);
				ERR_FAIL_COND_V(!scene,false)
				sml->get_root()->add_child(scene);

				String iconpath = GLOBAL_DEF("application/icon","Variant()""");
				if (iconpath!="") {
					Image icon;
					if (icon.load(iconpath)==OK)
						OS::get_singleton()->set_icon(icon);
				}


				//singletons
#ifdef TOOLS_ENABLED
			}
#endif
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
		if (script=="" && test=="" && game_path=="" && !editor) {

			ProjectManager *pmanager = memnew( ProjectManager );
			sml->get_root()->add_child(pmanager);
		}

#endif
	}

	OS::get_singleton()->set_main_loop( main_loop );

	return true;
}

uint64_t Main::last_ticks=0;
uint64_t Main::target_ticks=0;
float Main::time_accum=0;
uint32_t Main::frames=0;
uint32_t Main::frame=0;
bool Main::force_redraw_requested = false;

static uint64_t fixed_process_max=0;
static uint64_t idle_process_max=0;


bool Main::iteration() {

	uint64_t ticks=OS::get_singleton()->get_ticks_usec();
	uint64_t ticks_elapsed=ticks-last_ticks;

	frame+=ticks_elapsed;

	last_ticks=ticks;
	double step=(double)ticks_elapsed / 1000000.0;

	float frame_slice=1.0/OS::get_singleton()->get_iterations_per_second();

	if (step>frame_slice*8)
		step=frame_slice*8;

	time_accum+=step;

	float time_scale = OS::get_singleton()->get_time_scale();

	bool exit=false;


	int iters = 0;

	while(time_accum>frame_slice) {

		uint64_t fixed_begin = OS::get_singleton()->get_ticks_usec();

		PhysicsServer::get_singleton()->sync();
		PhysicsServer::get_singleton()->flush_queries();

		Physics2DServer::get_singleton()->sync();
		Physics2DServer::get_singleton()->flush_queries();

		if (OS::get_singleton()->get_main_loop()->iteration( frame_slice*time_scale )) {
			exit=true;
			break;
		}

		message_queue->flush();

		PhysicsServer::get_singleton()->step(frame_slice*time_scale);
		Physics2DServer::get_singleton()->step(frame_slice*time_scale);

		time_accum-=frame_slice;
		message_queue->flush();
		//if (AudioServer::get_singleton())
		//	AudioServer::get_singleton()->update();

		fixed_process_max=MAX(OS::get_singleton()->get_ticks_usec()-fixed_begin,fixed_process_max);
		iters++;
	}

	uint64_t idle_begin = OS::get_singleton()->get_ticks_usec();

	OS::get_singleton()->get_main_loop()->idle( step*time_scale );
	message_queue->flush();

	if (SpatialSoundServer::get_singleton())
		SpatialSoundServer::get_singleton()->update( step*time_scale );
	if (SpatialSound2DServer::get_singleton())
		SpatialSound2DServer::get_singleton()->update( step*time_scale );


	if (OS::get_singleton()->can_draw()) {

		if ((!force_redraw_requested) && OS::get_singleton()->is_in_low_processor_usage_mode()) {
			if (VisualServer::get_singleton()->has_changed()) {
				VisualServer::get_singleton()->draw(); // flush visual commands
				OS::get_singleton()->frames_drawn++;
			}
		} else {
			VisualServer::get_singleton()->draw(); // flush visual commands
			OS::get_singleton()->frames_drawn++;
			force_redraw_requested = false;
		}
	} else {
		VisualServer::get_singleton()->flush(); // flush visual commands
	}

	if (AudioServer::get_singleton())
		AudioServer::get_singleton()->update();

	for(int i=0;i<ScriptServer::get_language_count();i++) {
		ScriptServer::get_language(i)->frame();
	}

	idle_process_max=MAX(OS::get_singleton()->get_ticks_usec()-idle_begin,idle_process_max);

	if (script_debugger)
		script_debugger->idle_poll();


	//	x11_delay_usec(10000);
	frames++;

	if (frame>1000000) {

		if (GLOBAL_DEF("debug/print_fps", OS::get_singleton()->is_stdout_verbose())) {
			print_line("FPS: "+itos(frames));
		};

		OS::get_singleton()->_fps=frames;
		performance->set_process_time(idle_process_max/1000000.0);
		performance->set_fixed_process_time(fixed_process_max/1000000.0);
		idle_process_max=0;
		fixed_process_max=0;

		if (GLOBAL_DEF("debug/print_metrics", false)) {

			//PerformanceMetrics::print();
		};

		frame%=1000000;
		frames=0;
	}

	if (OS::get_singleton()->is_in_low_processor_usage_mode() || !OS::get_singleton()->can_draw())
		OS::get_singleton()->delay_usec(25000); //apply some delay to force idle time
	else {
		uint32_t frame_delay = OS::get_singleton()->get_frame_delay();
		if (frame_delay)
			OS::get_singleton()->delay_usec( OS::get_singleton()->get_frame_delay()*1000 );
	}

	int taret_fps = OS::get_singleton()->get_target_fps();
	if (taret_fps>0) {
		uint64_t time_step = 1000000L/taret_fps;
		target_ticks += time_step;
		uint64_t current_ticks = OS::get_singleton()->get_ticks_usec();
		if (current_ticks<target_ticks) OS::get_singleton()->delay_usec(target_ticks-current_ticks);
		current_ticks = OS::get_singleton()->get_ticks_usec();
		target_ticks = MIN(MAX(target_ticks,current_ticks-time_step),current_ticks+time_step);
	}

	return exit;
}

void Main::force_redraw() {

	force_redraw_requested = true;
};


void Main::cleanup() {

	ERR_FAIL_COND(!_start_success);

	if (script_debugger)
		memdelete(script_debugger);

	OS::get_singleton()->delete_main_loop();

	OS::get_singleton()->_cmdline.clear();
	OS::get_singleton()->_execpath="";
	OS::get_singleton()->_local_clipboard="";

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
		memdelete( translation_server );
	if (path_remap)
		memdelete(path_remap);
	if (globals)
		memdelete(globals);




	memdelete( message_queue );

	unregister_core_driver_types();
	unregister_core_types();

	//PerformanceMetrics::finish();
	OS::get_singleton()->clear_last_error();
	OS::get_singleton()->finalize_core();


}

