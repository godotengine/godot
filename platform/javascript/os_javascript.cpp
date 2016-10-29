/*************************************************************************/
/*  os_javascript.cpp                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "os_javascript.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "core/io/file_access_buffered_fa.h"
#include "drivers/unix/file_access_unix.h"
#include "drivers/unix/dir_access_unix.h"

#include "servers/visual/visual_server_raster.h"

#include "main/main.h"

#include "core/globals.h"
#include "emscripten.h"
#include "dom_keys.h"

int OS_JavaScript::get_video_driver_count() const {

	return 1;
}
const char * OS_JavaScript::get_video_driver_name(int p_driver) const {

	return "GLES2";
}

OS::VideoMode OS_JavaScript::get_default_video_mode() const {

	return OS::VideoMode();
}

int OS_JavaScript::get_audio_driver_count() const {

	return 1;
}

const char * OS_JavaScript::get_audio_driver_name(int p_driver) const {

	return "JavaScript";
}

void OS_JavaScript::initialize_core() {

	OS_Unix::initialize_core();
	FileAccess::make_default<FileAccessBufferedFA<FileAccessUnix> >(FileAccess::ACCESS_RESOURCES);

}

void OS_JavaScript::set_opengl_extensions(const char* p_gl_extensions) {

	ERR_FAIL_COND(!p_gl_extensions);
	gl_extensions=p_gl_extensions;
}

static InputEvent _setup_key_event(const EmscriptenKeyboardEvent *emscripten_event) {

	InputEvent ev;
	ev.type = InputEvent::KEY;
	ev.key.echo = emscripten_event->repeat;
	ev.key.mod.alt = emscripten_event->altKey;
	ev.key.mod.shift = emscripten_event->shiftKey;
	ev.key.mod.control = emscripten_event->ctrlKey;
	ev.key.mod.meta = emscripten_event->metaKey;
	ev.key.scancode = dom2godot_scancode(emscripten_event->keyCode);

	String unicode = String::utf8(emscripten_event->key);
	if (unicode.length()!=1) {
		unicode = String::utf8(emscripten_event->charValue);
	}
	if (unicode.length()==1) {
		ev.key.unicode=unicode[0];
	}

	return ev;
}

static InputEvent deferred_key_event;

static EM_BOOL _keydown_callback(int event_type, const EmscriptenKeyboardEvent *key_event, void *user_data) {

	ERR_FAIL_COND_V(event_type!=EMSCRIPTEN_EVENT_KEYDOWN, false);

	InputEvent ev = _setup_key_event(key_event);
	ev.key.pressed = true;
	if (ev.key.unicode==0 && keycode_has_unicode(ev.key.scancode)) {
		// defer to keypress event for legacy unicode retrieval
		deferred_key_event = ev;
		return false; // do not suppress keypress event
	}
	static_cast<OS_JavaScript*>(user_data)->push_input(ev);
	return true;
}

static EM_BOOL _keypress_callback(int event_type, const EmscriptenKeyboardEvent *key_event, void *user_data) {

	ERR_FAIL_COND_V(event_type!=EMSCRIPTEN_EVENT_KEYPRESS, false);

	deferred_key_event.key.unicode = key_event->charCode;
	static_cast<OS_JavaScript*>(user_data)->push_input(deferred_key_event);
	return true;
}

static EM_BOOL _keyup_callback(int event_type, const EmscriptenKeyboardEvent *key_event, void *user_data) {

	ERR_FAIL_COND_V(event_type!=EMSCRIPTEN_EVENT_KEYUP, false);

	InputEvent ev = _setup_key_event(key_event);
	ev.key.pressed = false;
	static_cast<OS_JavaScript*>(user_data)->push_input(ev);
	return ev.key.scancode!=KEY_UNKNOWN && ev.key.scancode!=0;

}

static EM_BOOL joy_callback_func(int p_type, const EmscriptenGamepadEvent *p_event, void *p_user) {
	OS_JavaScript *os = (OS_JavaScript*) OS::get_singleton();
	if (os) {
		return os->joy_connection_changed(p_type, p_event);
	}
	return false;
}

void OS_JavaScript::initialize(const VideoMode& p_desired,int p_video_driver,int p_audio_driver) {

	print_line("Init OS");

	if (gfx_init_func)
		gfx_init_func(gfx_init_ud,use_gl2,p_desired.width,p_desired.height,p_desired.fullscreen);

	default_videomode=p_desired;

	print_line("Init Audio");

	AudioDriverManagerSW::add_driver(&audio_driver_javascript);

	if (true) {
		RasterizerGLES2 *rasterizer_gles22=memnew( RasterizerGLES2(false,false,false,false) );;
		rasterizer_gles22->set_use_framebuffers(false); //not supported by emscripten
		if (gl_extensions)
			rasterizer_gles22->set_extensions(gl_extensions);
		rasterizer = rasterizer_gles22;
	} else {
//		rasterizer = memnew( RasterizerGLES1(true, false) );
	}

	print_line("Init VS");

	visual_server = memnew( VisualServerRaster(rasterizer) );
	visual_server->init();
	visual_server->cursor_set_visible(false, 0);

	/*AudioDriverManagerSW::get_driver(p_audio_driver)->set_singleton();

	if (AudioDriverManagerSW::get_driver(p_audio_driver)->init()!=OK) {

		ERR_PRINT("Initializing audio failed.");
	}*/

	print_line("Init SM");

	//sample_manager = memnew( SampleManagerMallocSW );
	audio_server = memnew( AudioServerJavascript );

	print_line("Init Mixer");

	//audio_server->set_mixer_params(AudioMixerSW::INTERPOLATION_LINEAR,false);
	audio_server->init();

	print_line("Init SoundServer");

	spatial_sound_server = memnew( SpatialSoundServerSW );
	spatial_sound_server->init();

	print_line("Init SpatialSoundServer");

	spatial_sound_2d_server = memnew( SpatialSound2DServerSW );
	spatial_sound_2d_server->init();

	//
	print_line("Init Physicsserver");

	physics_server = memnew( PhysicsServerSW );
	physics_server->init();
	physics_2d_server = memnew( Physics2DServerSW );
	physics_2d_server->init();

	input = memnew( InputDefault );

	EMSCRIPTEN_RESULT result = emscripten_set_keydown_callback(NULL, this , true, &_keydown_callback);
	if (result!=EMSCRIPTEN_RESULT_SUCCESS) {
		ERR_PRINTS( "Error while setting Emscripten keydown callback: Code " + itos(result) );
	}
	result = emscripten_set_keypress_callback(NULL, this, true, &_keypress_callback);
	if (result!=EMSCRIPTEN_RESULT_SUCCESS) {
		ERR_PRINTS( "Error while setting Emscripten keypress callback: Code " + itos(result) );
	}
	result = emscripten_set_keyup_callback(NULL, this, true, &_keyup_callback);
	if (result!=EMSCRIPTEN_RESULT_SUCCESS) {
		ERR_PRINTS( "Error while setting Emscripten keyup callback: Code " + itos(result) );
	}
	result = emscripten_set_gamepadconnected_callback(NULL, true, &joy_callback_func);
	if (result!=EMSCRIPTEN_RESULT_SUCCESS) {
		ERR_PRINTS( "Error while setting Emscripten gamepadconnected callback: Code " + itos(result) );
	}
	result = emscripten_set_gamepaddisconnected_callback(NULL, true, &joy_callback_func);
	if (result!=EMSCRIPTEN_RESULT_SUCCESS) {
		ERR_PRINTS( "Error while setting Emscripten gamepaddisconnected callback: Code " + itos(result) );
	}

#ifdef JAVASCRIPT_EVAL_ENABLED
	javascript_eval = memnew(JavaScript);
	Globals::get_singleton()->add_singleton(Globals::Singleton("JavaScript", javascript_eval));
#endif
}

void OS_JavaScript::set_main_loop( MainLoop * p_main_loop ) {

	main_loop=p_main_loop;
	input->set_main_loop(p_main_loop);

}

void OS_JavaScript::delete_main_loop() {

	memdelete( main_loop );
}

void OS_JavaScript::finalize() {

	memdelete(input);
}


void OS_JavaScript::vprint(const char* p_format, va_list p_list, bool p_stderr) {

	if (p_stderr) {

		vfprintf(stderr,p_format,p_list);
		fflush(stderr);
	} else {

		vprintf(p_format,p_list);
		fflush(stdout);
	}
}

void OS_JavaScript::print(const char *p_format, ... ) {

	va_list argp;
	va_start(argp, p_format);
	vprintf(p_format, argp );
	va_end(argp);

}

void OS_JavaScript::alert(const String& p_alert) {

	print("ALERT: %s\n",p_alert.utf8().get_data());
}


void OS_JavaScript::set_mouse_show(bool p_show) {

	//javascript has no mouse...
}

void OS_JavaScript::set_mouse_grab(bool p_grab) {

	//it really has no mouse...!
}

bool OS_JavaScript::is_mouse_grab_enabled() const {

	//*sigh* technology has evolved so much since i was a kid..
	return false;
}
Point2 OS_JavaScript::get_mouse_pos() const {

	return Point2();
}
int OS_JavaScript::get_mouse_button_state() const {

	return 0;
}
void OS_JavaScript::set_window_title(const String& p_title) {


}

//interesting byt not yet
//void set_clipboard(const String& p_text);
//String get_clipboard() const;

void OS_JavaScript::set_video_mode(const VideoMode& p_video_mode,int p_screen) {


}

OS::VideoMode OS_JavaScript::get_video_mode(int p_screen) const {

	return default_videomode;
}

Size2 OS_JavaScript::get_window_size() const {

	return Vector2(default_videomode.width,default_videomode.height);
}

void OS_JavaScript::get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen) const {

	p_list->push_back(default_videomode);
}

String OS_JavaScript::get_name() {

	return "HTML5";
}

MainLoop *OS_JavaScript::get_main_loop() const {

	return main_loop;
}

bool OS_JavaScript::can_draw() const {

	return true; //always?
}

void OS_JavaScript::set_cursor_shape(CursorShape p_shape) {

	//javascript really really really has no mouse.. how amazing..
}

void OS_JavaScript::main_loop_begin() {

	if (main_loop)
		main_loop->init();
}
bool OS_JavaScript::main_loop_iterate() {

	if (!main_loop)
		return false;

	if (time_to_save_sync>=0) {
		int64_t newtime = get_ticks_msec();
		int64_t elapsed = newtime - last_sync_time;
		last_sync_time=newtime;

		time_to_save_sync-=elapsed;

		print_line("elapsed "+itos(elapsed)+" tts "+itos(time_to_save_sync));

		if (time_to_save_sync<0) {
			//time to sync, for real
			// run 'success'
			print_line("DOING SYNCH!");
			EM_ASM(
			  FS.syncfs(function (err) {
			    assert(!err);
				console.log("Synched!");
			    //ccall('success', 'v');
			  });
			);
		}


	}
	process_joysticks();
	return Main::iteration();
}

void OS_JavaScript::main_loop_end() {

	if (main_loop)
		main_loop->finish();

}

void OS_JavaScript::main_loop_focusout() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
	//audio_driver_javascript.set_pause(true);

}

void OS_JavaScript::main_loop_focusin(){

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
	//audio_driver_javascript.set_pause(false);

}

void OS_JavaScript::push_input(const InputEvent& p_ev) {

	InputEvent ev = p_ev;
	ev.ID=last_id++;
	if (ev.type==InputEvent::MOUSE_MOTION) {
		input->set_mouse_pos(Point2(ev.mouse_motion.x, ev.mouse_motion.y));
	}
	input->parse_input_event(p_ev);
}

void OS_JavaScript::process_touch(int p_what,int p_pointer, const Vector<TouchPos>& p_points) {

//	print_line("ev: "+itos(p_what)+" pnt: "+itos(p_pointer)+" pointc: "+itos(p_points.size()));

	switch(p_what) {
		case 0: { //gesture begin

			if (touch.size()) {
				//end all if exist
				InputEvent ev;
				ev.type=InputEvent::MOUSE_BUTTON;
				ev.ID=last_id++;
				ev.mouse_button.button_index=BUTTON_LEFT;
				ev.mouse_button.button_mask=BUTTON_MASK_LEFT;
				ev.mouse_button.pressed=false;
				ev.mouse_button.x=touch[0].pos.x;
				ev.mouse_button.y=touch[0].pos.y;
				ev.mouse_button.global_x=touch[0].pos.x;
				ev.mouse_button.global_y=touch[0].pos.y;
				input->parse_input_event(ev);


				for(int i=0;i<touch.size();i++) {

					InputEvent ev;
					ev.type=InputEvent::SCREEN_TOUCH;
					ev.ID=last_id++;
					ev.screen_touch.index=touch[i].id;
					ev.screen_touch.pressed=false;
					ev.screen_touch.x=touch[i].pos.x;
					ev.screen_touch.y=touch[i].pos.y;
					input->parse_input_event(ev);

				}
			}

			touch.resize(p_points.size());
			for(int i=0;i<p_points.size();i++) {
				touch[i].id=p_points[i].id;
				touch[i].pos=p_points[i].pos;
			}

			{
				//send mouse
				InputEvent ev;
				ev.type=InputEvent::MOUSE_BUTTON;
				ev.ID=last_id++;
				ev.mouse_button.button_index=BUTTON_LEFT;
				ev.mouse_button.button_mask=BUTTON_MASK_LEFT;
				ev.mouse_button.pressed=true;
				ev.mouse_button.x=touch[0].pos.x;
				ev.mouse_button.y=touch[0].pos.y;
				ev.mouse_button.global_x=touch[0].pos.x;
				ev.mouse_button.global_y=touch[0].pos.y;
				last_mouse=touch[0].pos;
				input->parse_input_event(ev);
			}


			//send touch
			for(int i=0;i<touch.size();i++) {

				InputEvent ev;
				ev.type=InputEvent::SCREEN_TOUCH;
				ev.ID=last_id++;
				ev.screen_touch.index=touch[i].id;
				ev.screen_touch.pressed=true;
				ev.screen_touch.x=touch[i].pos.x;
				ev.screen_touch.y=touch[i].pos.y;
				input->parse_input_event(ev);
			}

		} break;
		case 1: { //motion


			if (p_points.size()) {
				//send mouse, should look for point 0?
				InputEvent ev;
				ev.type=InputEvent::MOUSE_MOTION;
				ev.ID=last_id++;
				ev.mouse_motion.button_mask=BUTTON_MASK_LEFT;
				ev.mouse_motion.x=p_points[0].pos.x;
				ev.mouse_motion.y=p_points[0].pos.y;
				input->set_mouse_pos(Point2(ev.mouse_motion.x,ev.mouse_motion.y));
				ev.mouse_motion.speed_x=input->get_mouse_speed().x;
				ev.mouse_motion.speed_y=input->get_mouse_speed().y;
				ev.mouse_motion.relative_x=p_points[0].pos.x-last_mouse.x;
				ev.mouse_motion.relative_y=p_points[0].pos.y-last_mouse.y;
				last_mouse=p_points[0].pos;
				input->parse_input_event(ev);
			}

			ERR_FAIL_COND(touch.size()!=p_points.size());

			for(int i=0;i<touch.size();i++) {

				int idx=-1;
				for(int j=0;j<p_points.size();j++) {

					if (touch[i].id==p_points[j].id) {
						idx=j;
						break;
					}

				}

				ERR_CONTINUE(idx==-1);

				if (touch[i].pos==p_points[idx].pos)
					continue; //no move unncesearily

				InputEvent ev;
				ev.type=InputEvent::SCREEN_DRAG;
				ev.ID=last_id++;
				ev.screen_drag.index=touch[i].id;
				ev.screen_drag.x=p_points[idx].pos.x;
				ev.screen_drag.y=p_points[idx].pos.y;
				ev.screen_drag.relative_x=p_points[idx].pos.x - touch[i].pos.x;
				ev.screen_drag.relative_y=p_points[idx].pos.y - touch[i].pos.y;
				input->parse_input_event(ev);
				touch[i].pos=p_points[idx].pos;
			}


		} break;
		case 2: { //release



			if (touch.size()) {
				//end all if exist
				InputEvent ev;
				ev.type=InputEvent::MOUSE_BUTTON;
				ev.ID=last_id++;
				ev.mouse_button.button_index=BUTTON_LEFT;
				ev.mouse_button.button_mask=BUTTON_MASK_LEFT;
				ev.mouse_button.pressed=false;
				ev.mouse_button.x=touch[0].pos.x;
				ev.mouse_button.y=touch[0].pos.y;
				ev.mouse_button.global_x=touch[0].pos.x;
				ev.mouse_button.global_y=touch[0].pos.y;
				input->parse_input_event(ev);


				for(int i=0;i<touch.size();i++) {

					InputEvent ev;
					ev.type=InputEvent::SCREEN_TOUCH;
					ev.ID=last_id++;
					ev.screen_touch.index=touch[i].id;
					ev.screen_touch.pressed=false;
					ev.screen_touch.x=touch[i].pos.x;
					ev.screen_touch.y=touch[i].pos.y;
					input->parse_input_event(ev);

				}
				touch.clear();
			}

		} break;
		case 3: { // add tuchi





			ERR_FAIL_INDEX(p_pointer,p_points.size());

			TouchPos tp=p_points[p_pointer];
			touch.push_back(tp);

			InputEvent ev;
			ev.type=InputEvent::SCREEN_TOUCH;
			ev.ID=last_id++;
			ev.screen_touch.index=tp.id;
			ev.screen_touch.pressed=true;
			ev.screen_touch.x=tp.pos.x;
			ev.screen_touch.y=tp.pos.y;
			input->parse_input_event(ev);

		} break;
		case 4: {


			for(int i=0;i<touch.size();i++) {
				if (touch[i].id==p_pointer) {

					InputEvent ev;
					ev.type=InputEvent::SCREEN_TOUCH;
					ev.ID=last_id++;
					ev.screen_touch.index=touch[i].id;
					ev.screen_touch.pressed=false;
					ev.screen_touch.x=touch[i].pos.x;
					ev.screen_touch.y=touch[i].pos.y;
					input->parse_input_event(ev);
					touch.remove(i);
					i--;
				}
			}

		} break;

	}

}

void OS_JavaScript::process_accelerometer(const Vector3& p_accelerometer) {

	input->set_accelerometer(p_accelerometer);
}

bool OS_JavaScript::has_touchscreen_ui_hint() const {

	return false; //???
}

void OS_JavaScript::main_loop_request_quit() {

	if (main_loop)
		main_loop->notification(MainLoop::NOTIFICATION_WM_QUIT_REQUEST);
}

void OS_JavaScript::set_display_size(Size2 p_size) {

	default_videomode.width=p_size.x;
	default_videomode.height=p_size.y;
}

void OS_JavaScript::reload_gfx() {

	if (gfx_init_func)
		gfx_init_func(gfx_init_ud,use_gl2,default_videomode.width,default_videomode.height,default_videomode.fullscreen);
	if (rasterizer)
		rasterizer->reload_vram();
}

Error OS_JavaScript::shell_open(String p_uri) {

	if (open_uri_func)
		return open_uri_func(p_uri)?ERR_CANT_OPEN:OK;
	return ERR_UNAVAILABLE;
};

String OS_JavaScript::get_resource_dir() const {

	return "/"; //javascript has it's own filesystem for resources inside the APK
}

String OS_JavaScript::get_locale() const {

	if (get_locale_func)
		return get_locale_func();
	return OS_Unix::get_locale();
}


String OS_JavaScript::get_data_dir() const {

	//if (get_data_dir_func)
	//	return get_data_dir_func();
	return "/userfs";
	//return Globals::get_singleton()->get_singleton_object("GodotOS")->call("get_data_dir");
};


void OS_JavaScript::_close_notification_funcs(const String& p_file,int p_flags) {

	print_line("close "+p_file+" flags "+itos(p_flags));
	if (p_file.begins_with("/userfs") && p_flags&FileAccess::WRITE) {
		static_cast<OS_JavaScript*>(get_singleton())->last_sync_time=OS::get_singleton()->get_ticks_msec();
		static_cast<OS_JavaScript*>(get_singleton())->time_to_save_sync=5000; //five seconds since last save
	}
}

void OS_JavaScript::process_joysticks() {

	int joy_count = emscripten_get_num_gamepads();
	for (int i = 0; i < joy_count; i++) {
		EmscriptenGamepadEvent state;
		emscripten_get_gamepad_status(i, &state);
		if (state.connected) {

			int num_buttons = MIN(state.numButtons, 18);
			int num_axes = MIN(state.numAxes, 8);
			for (int j = 0; j < num_buttons; j++) {

				float value = state.analogButton[j];
				if (String(state.mapping) == "standard" && (j == 6 || j == 7)) {
					InputDefault::JoyAxis jx;
					jx.min = 0;
					jx.value = value;
					last_id = input->joy_axis(last_id, i, j, jx);
				}
				else {
					last_id = input->joy_button(last_id, i, j, value);
				}
			}
			for (int j = 0; j < num_axes; j++) {

				InputDefault::JoyAxis jx;
				jx.min = -1;
				jx.value = state.axis[j];
				last_id = input->joy_axis(last_id, i, j, jx);
			}
		}
	}
}

bool OS_JavaScript::joy_connection_changed(int p_type, const EmscriptenGamepadEvent *p_event) {
	if (p_type == EMSCRIPTEN_EVENT_GAMEPADCONNECTED) {

		String guid = "";
		if (String(p_event->mapping) == "standard")
			guid = "Default HTML5 Gamepad";
		input->joy_connection_changed(p_event->index, true, String(p_event->id), guid);
	}
	else {
		input->joy_connection_changed(p_event->index, false, "");
	}
	return true;
}

bool OS_JavaScript::is_joy_known(int p_device) {
	return input->is_joy_mapped(p_device);
}

String OS_JavaScript::get_joy_guid(int p_device) const {
	return input->get_joy_guid_remapped(p_device);
}

OS_JavaScript::OS_JavaScript(GFXInitFunc p_gfx_init_func,void*p_gfx_init_ud, OpenURIFunc p_open_uri_func, GetDataDirFunc p_get_data_dir_func,GetLocaleFunc p_get_locale_func) {


	default_videomode.width=800;
	default_videomode.height=600;
	default_videomode.fullscreen=true;
	default_videomode.resizable=false;

	gfx_init_func=p_gfx_init_func;
	gfx_init_ud=p_gfx_init_ud;
	main_loop=NULL;
	last_id=1;
	gl_extensions=NULL;
	rasterizer=NULL;

	open_uri_func=p_open_uri_func;
	get_data_dir_func=p_get_data_dir_func;
	get_locale_func=p_get_locale_func;
	FileAccessUnix::close_notification_func=_close_notification_funcs;

	time_to_save_sync=-1;
}

OS_JavaScript::~OS_JavaScript() {


}
