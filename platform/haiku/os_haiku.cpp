#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "os_haiku.h"

OS_Haiku::OS_Haiku() {
	
};

void OS_Haiku::run() {

}

String OS_Haiku::get_name() {
	return "Haiku";
}

int OS_Haiku::get_video_driver_count() const {
	
}

const char* OS_Haiku::get_video_driver_name(int p_driver) const {
	
}

OS::VideoMode OS_Haiku::get_default_video_mode() const {
	
}

void OS_Haiku::initialize(const VideoMode& p_desired, int p_video_driver, int p_audio_driver) {
	main_loop = NULL;

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)
	//context_gl = memnew( ContextGL_X11( x11_display, x11_window,current_videomode, false ) );
	//context_gl->initialize();

	rasterizer = memnew(RasterizerGLES2);
#endif

	visual_server = memnew(VisualServerRaster(rasterizer));

	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server, get_render_thread_mode() == RENDER_SEPARATE_THREAD));
	}
}

void OS_Haiku::finalize() {
	if (main_loop) {
		memdelete(main_loop);
	}

	main_loop = NULL;

	visual_server->finish();
	memdelete(visual_server);
	memdelete(rasterizer);
}

void OS_Haiku::set_main_loop(MainLoop* p_main_loop) {
	main_loop = p_main_loop;
	
	// TODO: enable
	//input->set_main_loop(p_main_loop);
}

MainLoop* OS_Haiku::get_main_loop() const {
	return main_loop;
}

void OS_Haiku::delete_main_loop() {
	if (main_loop) {
		memdelete(main_loop);
	}

	main_loop = NULL;
}

bool OS_Haiku::can_draw() const {
	
}

Point2 OS_Haiku::get_mouse_pos() const {
	
}

int OS_Haiku::get_mouse_button_state() const {
	
}

void OS_Haiku::set_cursor_shape(CursorShape p_shape) {
	
}

void OS_Haiku::set_window_title(const String& p_title) {
	
}

Size2 OS_Haiku::get_window_size() const {
	
}

void OS_Haiku::set_video_mode(const VideoMode& p_video_mode, int p_screen) {
	
}

OS::VideoMode OS_Haiku::get_video_mode(int p_screen) const {
	
}

void OS_Haiku::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
	
}

String OS_Haiku::get_executable_path() const {
	return OS::get_executable_path();
}
