#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"
#include "drivers/gles2/rasterizer_gles2.h"
#include "os_haiku.h"

OS_Haiku::OS_Haiku() {
	
};

void OS_Haiku::run() {
	ERR_PRINT("run() NOT IMPLEMENTED");
}

String OS_Haiku::get_name() {
	return "Haiku";
}

int OS_Haiku::get_video_driver_count() const {
	return 1;
}

const char* OS_Haiku::get_video_driver_name(int p_driver) const {
	return "GLES2";
}

OS::VideoMode OS_Haiku::get_default_video_mode() const {
	return OS::VideoMode(800, 600, false);
}

void OS_Haiku::initialize(const VideoMode& p_desired, int p_video_driver, int p_audio_driver) {
	main_loop = NULL;
	current_video_mode = p_desired;

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)
	//context_gl = memnew( ContextGL_X11( x11_display, x11_window,current_videomode, false ) );
	//context_gl->initialize();

	rasterizer = memnew(RasterizerGLES2);
#endif

	visual_server = memnew(VisualServerRaster(rasterizer));

	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server, get_render_thread_mode() == RENDER_SEPARATE_THREAD));
	}

	visual_server->init();
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
	ERR_PRINT("can_draw() NOT IMPLEMENTED");
}

Point2 OS_Haiku::get_mouse_pos() const {
	ERR_PRINT("get_mouse_pos() NOT IMPLEMENTED");
}

int OS_Haiku::get_mouse_button_state() const {
	ERR_PRINT("get_mouse_button_state() NOT IMPLEMENTED");
}

void OS_Haiku::set_cursor_shape(CursorShape p_shape) {
	ERR_PRINT("set_cursor_shape() NOT IMPLEMENTED");
}

void OS_Haiku::set_window_title(const String& p_title) {
	ERR_PRINT("set_window_title() NOT IMPLEMENTED");
}

Size2 OS_Haiku::get_window_size() const {
	ERR_PRINT("get_window_size() NOT IMPLEMENTED");
}

void OS_Haiku::set_video_mode(const VideoMode& p_video_mode, int p_screen) {
	ERR_PRINT("set_video_mode() NOT IMPLEMENTED");
}

OS::VideoMode OS_Haiku::get_video_mode(int p_screen) const {
	return current_video_mode;
}

void OS_Haiku::get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen) const {
	ERR_PRINT("get_fullscreen_mode_list() NOT IMPLEMENTED");
}

String OS_Haiku::get_executable_path() const {
	return OS::get_executable_path();
}
