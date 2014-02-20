
#include "os_android.h"
#include "java_glue.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "servers/visual/visual_server_raster.h"

#include "file_access_jandroid.h"
#include "dir_access_jandroid.h"
#include "core/io/file_access_buffered_fa.h"
#include "main/main.h"
int OS_Android::get_video_driver_count() const {

	return 1;
}
const char * OS_Android::get_video_driver_name(int p_driver) const {

	return "GLES2";
}

OS::VideoMode OS_Android::get_default_video_mode() const {

	return OS::VideoMode();
}

int OS_Android::get_audio_driver_count() const {

	return 1;
}
const char * OS_Android::get_audio_driver_name(int p_driver) const {

	return "Android";
}

void OS_Android::initialize_core() {

	OS_Unix::initialize_core();
	//FileAccessJAndroid::make_default();
	DirAccessJAndroid::make_default();
	FileAccessBufferedFA<FileAccessJAndroid>::make_default();


}

void OS_Android::initialize(const VideoMode& p_desired,int p_video_driver,int p_audio_driver) {

	AudioDriverManagerSW::add_driver(&audio_driver_android);

	rasterizer = memnew( RasterizerGLES2 );
	visual_server = memnew( VisualServerRaster(rasterizer) );
	visual_server->init();
	visual_server->cursor_set_visible(false, 0);

	AudioDriverManagerSW::get_driver(p_audio_driver)->set_singleton();

	if (AudioDriverManagerSW::get_driver(p_audio_driver)->init()!=OK) {

		ERR_PRINT("Initializing audio failed.");
	}

	sample_manager = memnew( SampleManagerMallocSW );
	audio_server = memnew( AudioServerSW(sample_manager) );

	audio_server->set_mixer_params(AudioMixerSW::INTERPOLATION_LINEAR,false);
	audio_server->init();

	spatial_sound_server = memnew( SpatialSoundServerSW );
	spatial_sound_server->init();

	spatial_sound_2d_server = memnew( SpatialSound2DServerSW );
	spatial_sound_2d_server->init();

	//
	physics_server = memnew( PhysicsServerSW );
	physics_server->init();
	physics_2d_server = memnew( Physics2DServerSW );
	physics_2d_server->init();

	input = memnew( InputDefault );


}

void OS_Android::set_main_loop( MainLoop * p_main_loop ) {



	main_loop=p_main_loop;
}

void OS_Android::delete_main_loop() {

	memdelete( main_loop );
}

void OS_Android::finalize() {

	memdelete(input);

}


void OS_Android::vprint(const char* p_format, va_list p_list, bool p_stderr) {

	__android_log_vprint(p_stderr?ANDROID_LOG_ERROR:ANDROID_LOG_INFO,"godot",p_format,p_list);
}

void OS_Android::print(const char *p_format, ... ) {

	va_list argp;
	va_start(argp, p_format);
	__android_log_vprint(ANDROID_LOG_INFO,"godot",p_format,argp);
	va_end(argp);

}

void OS_Android::alert(const String& p_alert) {

	print("ALERT: %s\n",p_alert.utf8().get_data());
}


void OS_Android::set_mouse_show(bool p_show) {

	//android has no mouse...
}

void OS_Android::set_mouse_grab(bool p_grab) {

	//it really has no mouse...!
}

bool OS_Android::is_mouse_grab_enabled() const {

	//*sigh* technology has evolved so much since i was a kid..
	return false;
}
Point2 OS_Android::get_mouse_pos() const {

	return Point2();
}
int OS_Android::get_mouse_button_state() const {

	return 0;
}
void OS_Android::set_window_title(const String& p_title) {


}

//interesting byt not yet
//void set_clipboard(const String& p_text);
//String get_clipboard() const;

void OS_Android::set_video_mode(const VideoMode& p_video_mode,int p_screen) {


}

OS::VideoMode OS_Android::get_video_mode(int p_screen) const {

	return default_videomode;
}
void OS_Android::get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen) const {

	p_list->push_back(default_videomode);
}

String OS_Android::get_name() {

	return "Android";
}

MainLoop *OS_Android::get_main_loop() const {

	return main_loop;
}

bool OS_Android::can_draw() const {

	return true; //always?
}

void OS_Android::set_cursor_shape(CursorShape p_shape) {

	//android really really really has no mouse.. how amazing..
}

void OS_Android::main_loop_begin() {

	if (main_loop)
		main_loop->init();
}
bool OS_Android::main_loop_iterate() {

	if (!main_loop)
		return false;
	return Main::iteration();
}

void OS_Android::main_loop_end() {

	if (main_loop)
		main_loop->finish();

}

void OS_Android::process_touch(int p_what,int p_pointer, const Vector<TouchPos>& p_points) {



	switch(p_what) {
		case 0: { //gesture begin

			if (touch.size()) {
				//end all if exist
				InputEvent ev;
				ev.type=InputEvent::MOUSE_BUTTON;
				ev.ID=++last_id;
				ev.mouse_button.button_index=BUTTON_LEFT;
				ev.mouse_button.button_mask=BUTTON_MASK_LEFT;
				ev.mouse_button.pressed=false;
				ev.mouse_button.x=touch[0].pos.x;
				ev.mouse_button.y=touch[0].pos.y;
				ev.mouse_button.global_x=touch[0].pos.x;
				ev.mouse_button.global_y=touch[0].pos.y;
				input->set_mouse_pos(Point2(ev.mouse_button.x,ev.mouse_button.y));
				main_loop->input_event(ev);


				for(int i=0;i<touch.size();i++) {

					InputEvent ev;
					ev.type=InputEvent::SCREEN_TOUCH;
					ev.ID=++last_id;
					ev.screen_touch.index=touch[i].id;
					ev.screen_touch.pressed=false;
					ev.screen_touch.x=touch[i].pos.x;
					ev.screen_touch.y=touch[i].pos.y;
					main_loop->input_event(ev);

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
				ev.ID=++last_id;
				ev.mouse_button.button_index=BUTTON_LEFT;
				ev.mouse_button.button_mask=BUTTON_MASK_LEFT;
				ev.mouse_button.pressed=true;
				ev.mouse_button.x=touch[0].pos.x;
				ev.mouse_button.y=touch[0].pos.y;
				ev.mouse_button.global_x=touch[0].pos.x;
				ev.mouse_button.global_y=touch[0].pos.y;
				last_mouse=touch[0].pos;
				input->set_mouse_pos(Point2(ev.mouse_button.x,ev.mouse_button.y));
				main_loop->input_event(ev);
			}


			//send touch
			for(int i=0;i<touch.size();i++) {

				InputEvent ev;
				ev.type=InputEvent::SCREEN_TOUCH;
				ev.ID=++last_id;
				ev.screen_touch.index=touch[i].id;
				ev.screen_touch.pressed=true;
				ev.screen_touch.x=touch[i].pos.x;
				ev.screen_touch.y=touch[i].pos.y;
				main_loop->input_event(ev);
			}

		} break;
		case 1: { //motion


			if (p_points.size()) {
				//send mouse, should look for point 0?
				InputEvent ev;
				ev.type=InputEvent::MOUSE_MOTION;
				ev.ID=++last_id;
				ev.mouse_motion.button_mask=BUTTON_MASK_LEFT;
				ev.mouse_motion.x=p_points[0].pos.x;
				ev.mouse_motion.y=p_points[0].pos.y;
				input->set_mouse_pos(Point2(ev.mouse_motion.x,ev.mouse_motion.y));
				ev.mouse_motion.speed_x=input->get_mouse_speed().x;
				ev.mouse_motion.speed_y=input->get_mouse_speed().y;
				ev.mouse_motion.relative_x=p_points[0].pos.x-last_mouse.x;
				ev.mouse_motion.relative_y=p_points[0].pos.y-last_mouse.y;
				last_mouse=p_points[0].pos;
				main_loop->input_event(ev);
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
				ev.ID=++last_id;
				ev.screen_drag.index=touch[i].id;
				ev.screen_drag.x=p_points[idx].pos.x;
				ev.screen_drag.y=p_points[idx].pos.y;
				ev.screen_drag.x=p_points[idx].pos.x - touch[i].pos.x;
				ev.screen_drag.y=p_points[idx].pos.y - touch[i].pos.y;
				main_loop->input_event(ev);
				touch[i].pos=p_points[idx].pos;
			}


		} break;
		case 2: { //release



			if (touch.size()) {
				//end all if exist
				InputEvent ev;
				ev.type=InputEvent::MOUSE_BUTTON;
				ev.ID=++last_id;
				ev.mouse_button.button_index=BUTTON_LEFT;
				ev.mouse_button.button_mask=BUTTON_MASK_LEFT;
				ev.mouse_button.pressed=false;
				ev.mouse_button.x=touch[0].pos.x;
				ev.mouse_button.y=touch[0].pos.y;
				ev.mouse_button.global_x=touch[0].pos.x;
				ev.mouse_button.global_y=touch[0].pos.y;
				main_loop->input_event(ev);


				for(int i=0;i<touch.size();i++) {

					InputEvent ev;
					ev.type=InputEvent::SCREEN_TOUCH;
					ev.ID=++last_id;
					ev.screen_touch.index=touch[i].id;
					ev.screen_touch.pressed=false;
					ev.screen_touch.x=touch[i].pos.x;
					ev.screen_touch.y=touch[i].pos.y;
					main_loop->input_event(ev);

				}
			}

		} break;
		case 3: { // add tuchi





			ERR_FAIL_INDEX(p_pointer,p_points.size());

			TouchPos tp=p_points[p_pointer];
			touch.push_back(tp);

			InputEvent ev;
			ev.type=InputEvent::SCREEN_TOUCH;
			ev.ID=++last_id;
			ev.screen_touch.index=tp.id;
			ev.screen_touch.pressed=true;
			ev.screen_touch.x=tp.pos.x;
			ev.screen_touch.y=tp.pos.y;
			main_loop->input_event(ev);

		} break;
		case 4: {


			for(int i=0;i<touch.size();i++) {
				if (touch[i].id==p_pointer) {

					InputEvent ev;
					ev.type=InputEvent::SCREEN_TOUCH;
					ev.ID=++last_id;
					ev.screen_touch.index=touch[i].id;
					ev.screen_touch.pressed=false;
					ev.screen_touch.x=touch[i].pos.x;
					ev.screen_touch.y=touch[i].pos.y;
					main_loop->input_event(ev);
					touch.remove(i);
					i--;
				}
			}

		} break;

	}

}

OS_Android::OS_Android(int p_video_width,int p_video_height) {

	default_videomode.width=p_video_width;
	default_videomode.height=p_video_height;
	default_videomode.fullscreen=true;
	default_videomode.resizable=false;
	main_loop=NULL;
	last_id=1;
}

OS_Android::~OS_Android() {


}
