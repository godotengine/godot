#include "os_flash.h"

#include "main/main.h"

#include "rasterizer_flash.h"
#include "drivers/gles1/rasterizer_gles1.h"
#include "servers/visual/visual_server_raster.h"
#include "dir_access_flash.h"

//#include "AS3.h"

int OSFlash::get_video_driver_count() const {
	return 1;
};

const char * OSFlash::get_video_driver_name(int p_driver) const {
	return "Flash";
};

OS::VideoMode OSFlash::get_default_video_mode() const {
	return OS::VideoMode();
};

int OSFlash::get_audio_driver_count() const {
	return 1;
};

const char * OSFlash::get_audio_driver_name(int p_driver) const {
	return "Flash";
};

void OSFlash::initialize_core() {

	OS_Unix::initialize_core();

	//DirAccessFlash::make_default();
};

void OSFlash::initialize(const OS::VideoMode& p_desired,int p_video_driver,int p_audio_driver) {

	input = memnew( InputDefault );

	rasterizer = memnew( RasterizerGLES1(false) );
	//rasterizer = memnew( RasterizerFlash(false) );
	visual_server = memnew( VisualServerRaster(rasterizer) );
	visual_server->init();
	visual_server->cursor_set_visible(false, 0);

	audio_driver = memnew(AudioDriverDummy);
	audio_driver->set_singleton();
	audio_driver->init();

	sample_manager = memnew( SampleManagerMallocSW );
	audio_server = memnew( AudioServerSW(sample_manager) );
	audio_server->set_mixer_params(AudioMixerSW::INTERPOLATION_LINEAR,false);
	audio_server->init();

	spatial_sound_server = memnew( SpatialSoundServerSW );
	spatial_sound_server->init();

	spatial_sound_2d_server = memnew( SpatialSound2DServerSW );
	spatial_sound_2d_server->init();


	physics_server = memnew( PhysicsServerSW );
	physics_server->init();
	physics_2d_server = memnew( Physics2DServerSW );
	physics_2d_server->init();
};

void OSFlash::set_main_loop( MainLoop * p_main_loop ) {

	input->set_main_loop(p_main_loop);
	main_loop=p_main_loop;
};

void OSFlash::delete_main_loop() {

	memdelete( main_loop );
	main_loop = NULL;
};

void OSFlash::finalize() {

	memdelete(input);
};

void OSFlash::set_mouse_show(bool p_show) {

};

void OSFlash::set_mouse_grab(bool p_grab) {

};

bool OSFlash::is_mouse_grab_enabled() const {
	return false;
};

Point2 OSFlash::get_mouse_pos() const {

	return Point2();
};

int OSFlash::get_mouse_button_state() const {
	return 0;
};

void OSFlash::set_window_title(const String& p_title) {

};

bool OSFlash::has_virtual_keyboard() const {
	return false;
};

void OSFlash::show_virtual_keyboard(const String& p_existing_text,const Rect2& p_screen_rect) {

};

void OSFlash::hide_virtual_keyboard() {

};

void OSFlash::set_video_mode(const OS::VideoMode& p_video_mode,int p_screen) {
	default_videomode = p_video_mode;
};

OS::VideoMode OSFlash::get_video_mode(int p_screen) const {
	return default_videomode;
};

void OSFlash::get_fullscreen_mode_list(List<OS::VideoMode> *p_list,int p_screen) const {
	p_list->push_back(default_videomode);
};

String OSFlash::get_name() {

	return "Flash";
};

MainLoop *OSFlash::get_main_loop() const {

	return main_loop;
};

bool OSFlash::can_draw() const {
	return true;
}

void OSFlash::set_cursor_shape(CursorShape p_shape) {

};

bool OSFlash::has_touchscreen_ui_hint() const {
	return false;
};

Error OSFlash::shell_open(String p_uri) {
	return ERR_UNAVAILABLE;
};

void OSFlash::yield() {

	//flyield();
	//inline_as3(
	//	"flyield();\n"
	//);
};

bool OSFlash::iterate() {

	if (!main_loop)
		return true;

	return Main::iteration();
};

