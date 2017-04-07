/*************************************************************************/
/*  os_haiku.cpp                                                         */
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
#include <Screen.h>

#include "drivers/gles2/rasterizer_gles2.h"
#include "servers/physics/physics_server_sw.h"
#include "servers/visual/visual_server_raster.h"
#include "servers/visual/visual_server_wrap_mt.h"
//#include "servers/physics_2d/physics_2d_server_wrap_mt.h"
#include "main/main.h"

#include "os_haiku.h"

OS_Haiku::OS_Haiku() {
#ifdef MEDIA_KIT_ENABLED
	AudioDriverManager::add_driver(&driver_media_kit);
#endif
};

void OS_Haiku::run() {
	if (!main_loop) {
		return;
	}

	main_loop->init();
	context_gl->release_current();

	// TODO: clean up
	BMessenger *bms = new BMessenger(window);
	BMessage *msg = new BMessage();
	bms->SendMessage(LOCKGL_MSG, msg);

	window->StartMessageRunner();
	app->Run();
	window->StopMessageRunner();

	delete app;

	delete bms;
	delete msg;
	main_loop->finish();
}

String OS_Haiku::get_name() {
	return "Haiku";
}

int OS_Haiku::get_video_driver_count() const {
	return 1;
}

const char *OS_Haiku::get_video_driver_name(int p_driver) const {
	return "GLES2";
}

OS::VideoMode OS_Haiku::get_default_video_mode() const {
	return OS::VideoMode(800, 600, false);
}

void OS_Haiku::initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver) {
	main_loop = NULL;
	current_video_mode = p_desired;

	app = new HaikuApplication();

	BRect frame;
	frame.Set(50, 50, 50 + current_video_mode.width - 1, 50 + current_video_mode.height - 1);

	window = new HaikuDirectWindow(frame);
	window->SetVideoMode(&current_video_mode);

	if (current_video_mode.fullscreen) {
		window->SetFullScreen(true);
	}

	if (!current_video_mode.resizable) {
		uint32 flags = window->Flags();
		flags |= B_NOT_RESIZABLE;
		window->SetFlags(flags);
	}

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)
	context_gl = memnew(ContextGL_Haiku(window));
	context_gl->initialize();
	context_gl->make_current();

	rasterizer = memnew(RasterizerGLES2);
#endif

	visual_server = memnew(VisualServerRaster(rasterizer));

	ERR_FAIL_COND(!visual_server);

	// TODO: enable multithreaded VS
	/*
	if (get_render_thread_mode() != RENDER_THREAD_UNSAFE) {
		visual_server = memnew(VisualServerWrapMT(visual_server, get_render_thread_mode() == RENDER_SEPARATE_THREAD));
	}
	*/

	input = memnew(InputDefault);
	window->SetInput(input);

	window->Show();
	visual_server->init();

	physics_server = memnew(PhysicsServerSW);
	physics_server->init();
	physics_2d_server = memnew(Physics2DServerSW);
	// TODO: enable multithreaded PS
	//physics_2d_server = Physics2DServerWrapMT::init_server<Physics2DServerSW>();
	physics_2d_server->init();

	AudioDriverManager::get_driver(p_audio_driver)->set_singleton();

	if (AudioDriverManager::get_driver(p_audio_driver)->init() != OK) {
		ERR_PRINT("Initializing audio failed.");
	}

	power_manager = memnew(PowerHaiku);
}

void OS_Haiku::finalize() {
	if (main_loop) {
		memdelete(main_loop);
	}

	main_loop = NULL;

	visual_server->finish();
	memdelete(visual_server);
	memdelete(rasterizer);

	physics_server->finish();
	memdelete(physics_server);

	physics_2d_server->finish();
	memdelete(physics_2d_server);

	memdelete(input);

#if defined(OPENGL_ENABLED) || defined(LEGACYGL_ENABLED)
	memdelete(context_gl);
#endif
}

void OS_Haiku::set_main_loop(MainLoop *p_main_loop) {
	main_loop = p_main_loop;
	input->set_main_loop(p_main_loop);
	window->SetMainLoop(p_main_loop);
}

MainLoop *OS_Haiku::get_main_loop() const {
	return main_loop;
}

void OS_Haiku::delete_main_loop() {
	if (main_loop) {
		memdelete(main_loop);
	}

	main_loop = NULL;
	window->SetMainLoop(NULL);
}

void OS_Haiku::release_rendering_thread() {
	context_gl->release_current();
}

void OS_Haiku::make_rendering_thread() {
	context_gl->make_current();
}

bool OS_Haiku::can_draw() const {
	// TODO: implement
	return true;
}

void OS_Haiku::swap_buffers() {
	context_gl->swap_buffers();
}

Point2 OS_Haiku::get_mouse_pos() const {
	return window->GetLastMousePosition();
}

int OS_Haiku::get_mouse_button_state() const {
	return window->GetLastButtonMask();
}

void OS_Haiku::set_cursor_shape(CursorShape p_shape) {
	//ERR_PRINT("set_cursor_shape() NOT IMPLEMENTED");
}

int OS_Haiku::get_screen_count() const {
	// TODO: implement get_screen_count()
	return 1;
}

int OS_Haiku::get_current_screen() const {
	// TODO: implement get_current_screen()
	return 0;
}

void OS_Haiku::set_current_screen(int p_screen) {
	// TODO: implement set_current_screen()
}

Point2 OS_Haiku::get_screen_position(int p_screen) const {
	// TODO: make this work with the p_screen parameter
	BScreen *screen = new BScreen(window);
	BRect frame = screen->Frame();
	delete screen;
	return Point2i(frame.left, frame.top);
}

Size2 OS_Haiku::get_screen_size(int p_screen) const {
	// TODO: make this work with the p_screen parameter
	BScreen *screen = new BScreen(window);
	BRect frame = screen->Frame();
	delete screen;
	return Size2i(frame.IntegerWidth() + 1, frame.IntegerHeight() + 1);
}

void OS_Haiku::set_window_title(const String &p_title) {
	window->SetTitle(p_title.utf8().get_data());
}

Size2 OS_Haiku::get_window_size() const {
	BSize size = window->Size();
	return Size2i(size.IntegerWidth() + 1, size.IntegerHeight() + 1);
}

void OS_Haiku::set_window_size(const Size2 p_size) {
	// TODO: why does it stop redrawing after this is called?
	window->ResizeTo(p_size.x, p_size.y);
}

Point2 OS_Haiku::get_window_position() const {
	BPoint point(0, 0);
	window->ConvertToScreen(&point);
	return Point2i(point.x, point.y);
}

void OS_Haiku::set_window_position(const Point2 &p_position) {
	window->MoveTo(p_position.x, p_position.y);
}

void OS_Haiku::set_window_fullscreen(bool p_enabled) {
	window->SetFullScreen(p_enabled);
	current_video_mode.fullscreen = p_enabled;
	visual_server->init();
}

bool OS_Haiku::is_window_fullscreen() const {
	return current_video_mode.fullscreen;
}

void OS_Haiku::set_window_resizable(bool p_enabled) {
	uint32 flags = window->Flags();

	if (p_enabled) {
		flags &= ~(B_NOT_RESIZABLE);
	} else {
		flags |= B_NOT_RESIZABLE;
	}

	window->SetFlags(flags);
	current_video_mode.resizable = p_enabled;
}

bool OS_Haiku::is_window_resizable() const {
	return current_video_mode.resizable;
}

void OS_Haiku::set_window_minimized(bool p_enabled) {
	window->Minimize(p_enabled);
}

bool OS_Haiku::is_window_minimized() const {
	return window->IsMinimized();
}

void OS_Haiku::set_window_maximized(bool p_enabled) {
	window->Minimize(!p_enabled);
}

bool OS_Haiku::is_window_maximized() const {
	return !window->IsMinimized();
}

void OS_Haiku::set_video_mode(const VideoMode &p_video_mode, int p_screen) {
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
