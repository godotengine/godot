/*************************************************************************/
/*  os_nacl.cpp                                                          */
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
#include "os_nacl.h"

#include "drivers/unix/memory_pool_static_malloc.h"
#include "os/memory_pool_dynamic_static.h"
#include "main/main.h"
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include "io/file_access_memory.h"
#include "core/io/file_access_pack.h"
#include "scene/io/scene_loader.h"
#include "scene/main/scene_main_loop.h"

#include "servers/visual/visual_server_raster.h"

#include "drivers/gles2/rasterizer_gles2.h"
#include "nacl_keycodes.h"

#include "core/globals.h"
#include "core/input_map.h"

#include <ppapi/cpp/point.h>
#include <ppapi/cpp/var.h>

#define UNIX_ENABLED
#include "drivers/unix/thread_posix.h"
#include "drivers/unix/semaphore_posix.h"
#include "drivers/unix/mutex_posix.h"


int OSNacl::get_video_driver_count() const {

	return 1;
};
const char * OSNacl::get_video_driver_name(int p_driver) const {

	return "GLES2";
};

OS::VideoMode OSNacl::get_default_video_mode() const {

	return OS::VideoMode(800,600,false);
};

int OSNacl::get_audio_driver_count() const {

	return 1;
};

const char * OSNacl::get_audio_driver_name(int p_driver) const {

	return "nacl_audio";
};

static MemoryPoolStaticMalloc *mempool_static=NULL;
static MemoryPoolDynamicStatic *mempool_dynamic=NULL;

void OSNacl::initialize_core() {

	ticks_start=0;
	ticks_start=get_ticks_usec();
};

void OSNacl::initialize(const VideoMode& p_desired,int p_video_driver,int p_audio_driver) {

	rasterizer = memnew( RasterizerGLES2 );
	visual_server = memnew( VisualServerRaster(rasterizer) );
	visual_server->init();
	visual_server->cursor_set_visible(false, 0);

	audio_driver = memnew(AudioDriverNacl);
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

	//
	physics_server = memnew( PhysicsServerSW );
	physics_server->init();

	physics_2d_server = memnew( Physics2DServerSW );
	physics_2d_server->init();

    input = memnew(InputDefault);
};

void OSNacl::set_main_loop( MainLoop * p_main_loop ) {

	main_loop = p_main_loop;
    input->set_main_loop(p_main_loop);
    main_loop->init();
};

void OSNacl::delete_main_loop() {

	if (main_loop)
		memdelete(main_loop);
};

void OSNacl::finalize() {


};
void OSNacl::finalize_core() {

	if (mempool_dynamic)
		memdelete( mempool_dynamic );
	if (mempool_static)
		delete mempool_static;

};

void OSNacl::alert(const String& p_alert,const String& p_title) {

	fprintf(stderr,"ERROR: %s\n",p_alert.utf8().get_data());
};

void OSNacl::vprint(const char* p_format, va_list p_list, bool p_strerr) {

	vprintf(p_format,p_list);
	fflush(stdout);
}


String OSNacl::get_stdin_string(bool p_block) {

	char buff[1024];
	return fgets(buff,1024,stdin);
};

void OSNacl::set_mouse_show(bool p_show) {

};

void OSNacl::set_mouse_grab(bool p_grab) {

};

bool OSNacl::is_mouse_grab_enabled() const {

	return false;
};

int OSNacl::get_mouse_button_state() const {

	return mouse_mask;
};


Point2 OSNacl::get_mouse_pos() const {

	return Point2();
};

void OSNacl::set_window_title(const String& p_title) {
};

void OSNacl::set_video_mode(const VideoMode& p_video_mode, int p_screen) {

	video_mode = p_video_mode;
};

OS::VideoMode OSNacl::get_video_mode(int p_screen) const {

	return video_mode;
};

void OSNacl::get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen) const {

};

Error OSNacl::execute(const String& p_path, const List<String>& p_arguments,bool p_blocking, OS::ProcessID *r_child_id, String* r_pipe, int *r_exitcode) {

	return ERR_UNAVAILABLE;
};

Error OSNacl::kill(const ProcessID& p_pid) {

	return ERR_UNAVAILABLE;
};

bool OSNacl::has_environment(const String& p_var) const {

	return getenv(p_var.utf8().get_data())!=NULL;
};

String OSNacl::get_environment(const String& p_var) const {

	if (getenv(p_var.utf8().get_data()))
		return getenv(p_var.utf8().get_data());
	return "";
};

String OSNacl::get_name() {

	return "NaCl";
};

MainLoop *OSNacl::get_main_loop() const {

	return main_loop;
};

OS::Date OSNacl::get_date() const {

	time_t t=time(NULL);
	struct tm *lt=localtime(&t);
	Date ret;
	ret.year=lt->tm_year;
	ret.month=(Month)lt->tm_mon;
	ret.day=lt->tm_mday;
	ret.weekday=(Weekday)lt->tm_wday;
	ret.dst=lt->tm_isdst;

	return ret;
};

OS::Time OSNacl::get_time() const {

	time_t t=time(NULL);
	struct tm *lt=localtime(&t);
	Time ret;
	ret.hour=lt->tm_hour;
	ret.min=lt->tm_min;
	ret.sec=lt->tm_sec;
	return ret;
};

void OSNacl::delay_usec(uint32_t p_usec) const {

	//usleep(p_usec);
};

uint64_t OSNacl::get_ticks_usec() const {

	struct timeval tv_now;
	gettimeofday(&tv_now,NULL);

	uint64_t longtime = (uint64_t)tv_now.tv_usec + (uint64_t)tv_now.tv_sec*1000000L;
	longtime-=ticks_start;

	return longtime;
};

bool OSNacl::can_draw() const {

	return minimized != true;
};

void OSNacl::queue_event(const InputEvent& p_event) {

	ERR_FAIL_INDEX( event_count, MAX_EVENTS );

	event_queue[event_count++] = p_event;
};

void OSNacl::add_package(String p_name, Vector<uint8_t> p_data) {

	FileAccessMemory::register_file(p_name, p_data);
    FileAccess::make_default<FileAccessMemory>(FileAccess::ACCESS_RESOURCES);
    FileAccess::make_default<FileAccessMemory>(FileAccess::ACCESS_USERDATA);
    FileAccess::make_default<FileAccessMemory>(FileAccess::ACCESS_FILESYSTEM);

    if (!PackedData::get_singleton())
        memnew(PackedData);

    printf("adding package %ls, %x\n", p_name.c_str(), PackedData::get_singleton());
    PackedData::get_singleton()->set_disabled(true);
    PackedData::get_singleton()->add_pack(p_name);
    PackedData::get_singleton()->set_disabled(false);
    printf("added\n");
};

void OSNacl::set_cursor_shape(CursorShape p_shape) {


};

String OSNacl::get_resource_dir() const {

    return ".";
};

static int mouse_button(int p_nacl_but) {

	switch (p_nacl_but) {

	case PP_INPUTEVENT_MOUSEBUTTON_LEFT:
		return BUTTON_LEFT;
	case PP_INPUTEVENT_MOUSEBUTTON_MIDDLE:
		return BUTTON_MIDDLE;
	case PP_INPUTEVENT_MOUSEBUTTON_RIGHT:
		return BUTTON_RIGHT;
	};

	return 0;
};

static InputModifierState modifier(uint32_t p_mod) {

	InputModifierState mod_mask;

	mod_mask.shift = p_mod & PP_INPUTEVENT_MODIFIER_SHIFTKEY;
	mod_mask.alt = p_mod & PP_INPUTEVENT_MODIFIER_ALTKEY;
	mod_mask.control = p_mod & PP_INPUTEVENT_MODIFIER_CONTROLKEY;
	mod_mask.meta = p_mod & PP_INPUTEVENT_MODIFIER_METAKEY;

	return mod_mask;
};



void OSNacl::handle_event(const pp::InputEvent& p_event) {

	int type = p_event.GetType();
	switch (type) {

	case PP_INPUTEVENT_TYPE_MOUSEDOWN:
	case PP_INPUTEVENT_TYPE_MOUSEUP:
	case PP_INPUTEVENT_TYPE_WHEEL: {

		InputEvent event;
		event.ID=++event_id;
		event.type = InputEvent::MOUSE_BUTTON;
		event.device=0;

		pp::MouseInputEvent mevent(p_event);
		if (type == PP_INPUTEVENT_TYPE_WHEEL) {

			pp::WheelInputEvent wevent(p_event);;
			float ticks = wevent.GetTicks().y();
			if (ticks == 0)
				break; // whut?

			event.mouse_button.pressed = true;
			event.mouse_button.button_index = ticks > 0 ? BUTTON_WHEEL_UP : BUTTON_WHEEL_DOWN;
			event.mouse_button.doubleclick = false;

		} else {

			event.mouse_button.pressed = (type == PP_INPUTEVENT_TYPE_MOUSEDOWN);
			event.mouse_button.button_index = mouse_button(mevent.GetButton());
			event.mouse_button.doubleclick = (mevent.GetClickCount() % 2) == 0;

			mouse_mask &= ~(1<< (event.mouse_button.button_index - 1));
			mouse_mask |= (event.mouse_button.pressed << (event.mouse_button.button_index - 1));
		};
		pp::Point pos = mevent.GetPosition();
		event.mouse_button.button_mask = mouse_mask;
		event.mouse_button.global_x = pos.x();
		event.mouse_button.x = pos.x();
		event.mouse_button.global_y = pos.y();
		event.mouse_button.y = pos.y();
		event.mouse_button.pointer_index = 0;
		event.mouse_button.mod = modifier(p_event.GetModifiers());
		queue_event(event);

	} break;

	case PP_INPUTEVENT_TYPE_MOUSEMOVE: {

		pp::MouseInputEvent mevent(p_event);
		pp::Point pos = mevent.GetPosition();

		InputEvent event;
		event.ID=++event_id;
		event.type = InputEvent::MOUSE_MOTION;
		event.mouse_motion.pointer_index = 0;
		event.mouse_motion.global_x = pos.x();
		event.mouse_motion.global_y = pos.y();
		event.mouse_motion.x = pos.x();
		event.mouse_motion.y = pos.y();
		event.mouse_motion.button_mask = mouse_mask;
		event.mouse_motion.mod = modifier(p_event.GetModifiers());

		event.mouse_motion.relative_x = pos.x() - mouse_last_x;
		event.mouse_motion.relative_y = pos.y() - mouse_last_y;
		mouse_last_x = pos.x();
		mouse_last_y = pos.y();

		queue_event(event);

	} break;

	case PP_INPUTEVENT_TYPE_RAWKEYDOWN:
	case PP_INPUTEVENT_TYPE_KEYDOWN:
	case PP_INPUTEVENT_TYPE_KEYUP: {

		pp::KeyboardInputEvent kevent(p_event);
		bool is_char;
		uint32_t key = godot_key(kevent.GetKeyCode(), is_char);
		if (type != PP_INPUTEVENT_TYPE_KEYUP && is_char) {

			last_scancode = key;
			break;
		};

		InputEvent event;
		event.ID=++event_id;
		event.type = InputEvent::KEY;
		event.key.pressed = (type != PP_INPUTEVENT_TYPE_KEYUP);
		event.key.scancode = key;
		event.key.unicode = key;

		event.key.echo = p_event.GetModifiers() & PP_INPUTEVENT_MODIFIER_ISAUTOREPEAT;
		event.key.mod = modifier(p_event.GetModifiers());
		queue_event(event);
	} break;

	case PP_INPUTEVENT_TYPE_CHAR: {

		pp::KeyboardInputEvent kevent(p_event);
		InputEvent event;
		event.ID = ++event_id;
		event.type = InputEvent::KEY;
		event.key.pressed = true;
		event.key.scancode = last_scancode;
		event.key.unicode = kevent.GetCharacterText().AsString().c_str()[0];
		event.key.mod = modifier(p_event.GetModifiers());
		event.key.echo = p_event.GetModifiers() & PP_INPUTEVENT_MODIFIER_ISAUTOREPEAT;
		queue_event(event);

	} break;

	/*
	case NPEventType_Minimize: {

		minimized = p_event->u.minimize.value == 1;

	} break;


	case NPEventType_Focus: {

		if (p_event->u.focus.value == 1) {
			main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_IN);
		} else {
			main_loop->notification(MainLoop::NOTIFICATION_WM_FOCUS_OUT);
		};
	} break;

	*/

	default:
		;
	};
};

bool OSNacl::iterate() {

	if (!main_loop) {
		event_count = 0;
		return true;
	};

	for (int i=0; i<event_count; i++) {

        input->parse_input_event(event_queue[i]);
	};

	event_count = 0;

	return Main::iteration();
};


OSNacl::OSNacl() {

	main_loop=NULL;
	mempool_dynamic = NULL;
	mempool_static = NULL;
	mouse_last_x = 0;
	mouse_last_y = 0;
	event_count = 0;
	event_id = 0;
	mouse_mask = 0;
	video_mode = get_default_video_mode();
	last_scancode = 0;
	minimized = false;

	ThreadPosix::make_default();
	SemaphorePosix::make_default();
	MutexPosix::make_default();
	mempool_static = new MemoryPoolStaticMalloc;
	mempool_dynamic = memnew( MemoryPoolDynamicStatic );
};

OSNacl::~OSNacl() {

};
