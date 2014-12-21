/*************************************************************************/
/*  os_winrt.cpp                                                       */
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
#include "drivers/gles2/rasterizer_gles2.h"
#include "drivers/gles1/rasterizer_gles1.h"
#include "os_winrt.h"
#include "drivers/nedmalloc/memory_pool_static_nedmalloc.h"
#include "drivers/unix/memory_pool_static_malloc.h"
#include "os/memory_pool_dynamic_static.h"
#include "thread_winrt.h"
//#include "drivers/windows/semaphore_windows.h"
#include "drivers/windows/mutex_windows.h"
#include "main/main.h"
#include "drivers/windows/file_access_windows.h"
#include "drivers/windows/dir_access_windows.h"


#include "servers/visual/visual_server_raster.h"
#include "servers/audio/audio_server_sw.h"
#include "servers/visual/visual_server_wrap_mt.h"

#include "os/pc_joystick_map.h"
#include "os/memory_pool_dynamic_prealloc.h"
#include "globals.h"
#include "io/marshalls.h"

#include <wrl.h>

using namespace Windows::ApplicationModel::Core;
using namespace Windows::ApplicationModel::Activation;
using namespace Windows::UI::Core;
using namespace Windows::UI::Input;
using namespace Windows::Foundation;
using namespace Windows::Graphics::Display;
using namespace Microsoft::WRL;


int OSWinrt::get_video_driver_count() const {

	return 2;
}
const char * OSWinrt::get_video_driver_name(int p_driver) const {

	return p_driver==0?"GLES2":"GLES1";
}

OS::VideoMode OSWinrt::get_default_video_mode() const {

	return video_mode;
}

int OSWinrt::get_audio_driver_count() const {

	return AudioDriverManagerSW::get_driver_count();
}
const char * OSWinrt::get_audio_driver_name(int p_driver) const {

	AudioDriverSW* driver = AudioDriverManagerSW::get_driver(p_driver);
	ERR_FAIL_COND_V( !driver, "" );
	return AudioDriverManagerSW::get_driver(p_driver)->get_name();
}

static MemoryPoolStatic *mempool_static=NULL;
static MemoryPoolDynamic *mempool_dynamic=NULL;

void OSWinrt::initialize_core() {


	last_button_state=0;

	//RedirectIOToConsole();

	ThreadWinrt::make_default();
	//SemaphoreWindows::make_default();
	MutexWindows::make_default();	

	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_RESOURCES);
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_USERDATA);
	FileAccess::make_default<FileAccessWindows>(FileAccess::ACCESS_FILESYSTEM);
	//FileAccessBufferedFA<FileAccessWindows>::make_default();
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_RESOURCES);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_USERDATA);
	DirAccess::make_default<DirAccessWindows>(DirAccess::ACCESS_FILESYSTEM);

	//TCPServerWinsock::make_default();
	//StreamPeerWinsock::make_default();
	
	mempool_static = new MemoryPoolStaticMalloc;
#if 1
	mempool_dynamic = memnew( MemoryPoolDynamicStatic );
#else
#define DYNPOOL_SIZE 4*1024*1024
	void * buffer = malloc( DYNPOOL_SIZE );
	mempool_dynamic = memnew( MemoryPoolDynamicPrealloc(buffer,DYNPOOL_SIZE) );

#endif
	
	   // We need to know how often the clock is updated
	if( !QueryPerformanceFrequency((LARGE_INTEGER *)&ticks_per_second) )
		ticks_per_second = 1000;
	// If timeAtGameStart is 0 then we get the time since
	// the start of the computer when we call GetGameTime()
	ticks_start = 0;
	ticks_start = get_ticks_usec();

	cursor_shape=CURSOR_ARROW;
}

bool OSWinrt::can_draw() const {

	return !minimized;
};


void OSWinrt::set_gl_context(ContextEGL* p_context) {

	gl_context = p_context;
};

void OSWinrt::screen_size_changed() {

	gl_context->reset();
};

void OSWinrt::initialize(const VideoMode& p_desired,int p_video_driver,int p_audio_driver) {

    main_loop=NULL;
    outside=true;

	gl_context->initialize();
	VideoMode vm;
	vm.width = gl_context->get_window_width();
	vm.height = gl_context->get_window_height();
	vm.fullscreen = true;
	vm.resizable = false;

	set_video_mode(vm);

	gl_context->make_current();
	rasterizer = memnew( RasterizerGLES2 );

	visual_server = memnew( VisualServerRaster(rasterizer) );
	if (get_render_thread_mode()!=RENDER_THREAD_UNSAFE) {

		visual_server =memnew(VisualServerWrapMT(visual_server,get_render_thread_mode()==RENDER_SEPARATE_THREAD));
	}

	//
	physics_server = memnew( PhysicsServerSW );
	physics_server->init();

	physics_2d_server = memnew( Physics2DServerSW );
	physics_2d_server->init();

	visual_server->init();	

	input = memnew( InputDefault );

	AudioDriverManagerSW::get_driver(p_audio_driver)->set_singleton();

	if (AudioDriverManagerSW::get_driver(p_audio_driver)->init()!=OK) {

		ERR_PRINT("Initializing audio failed.");
	}

	sample_manager = memnew( SampleManagerMallocSW );
	audio_server = memnew( AudioServerSW(sample_manager) );

	audio_server->init();

	spatial_sound_server = memnew( SpatialSoundServerSW );
	spatial_sound_server->init();
	spatial_sound_2d_server = memnew( SpatialSound2DServerSW );
	spatial_sound_2d_server->init();


	_ensure_data_dir();
}

void OSWinrt::set_clipboard(const String& p_text) {

	/*
	if (!OpenClipboard(hWnd)) {
		ERR_EXPLAIN("Unable to open clipboard.");
		ERR_FAIL();
	};
	EmptyClipboard();

	HGLOBAL mem = GlobalAlloc(GMEM_MOVEABLE, (p_text.length() + 1) * sizeof(CharType));
	if (mem == NULL) {
		ERR_EXPLAIN("Unable to allocate memory for clipboard contents.");
		ERR_FAIL();
	};
	LPWSTR lptstrCopy = (LPWSTR)GlobalLock(mem);
	memcpy(lptstrCopy, p_text.c_str(), (p_text.length() + 1) * sizeof(CharType));
	//memset((lptstrCopy + p_text.length()), 0, sizeof(CharType));
	GlobalUnlock(mem);

	SetClipboardData(CF_UNICODETEXT, mem);

	// set the CF_TEXT version (not needed?)
	CharString utf8 = p_text.utf8();
	mem = GlobalAlloc(GMEM_MOVEABLE, utf8.length() + 1);
	if (mem == NULL) {
		ERR_EXPLAIN("Unable to allocate memory for clipboard contents.");
		ERR_FAIL();
	};
	LPTSTR ptr = (LPTSTR)GlobalLock(mem);
	memcpy(ptr, utf8.get_data(), utf8.length());
	ptr[utf8.length()] = 0;
	GlobalUnlock(mem);

	SetClipboardData(CF_TEXT, mem);

	CloseClipboard();
	*/
};

String OSWinrt::get_clipboard() const {

	/*
	String ret;
	if (!OpenClipboard(hWnd)) {
		ERR_EXPLAIN("Unable to open clipboard.");
		ERR_FAIL_V("");
	};

	if (IsClipboardFormatAvailable(CF_UNICODETEXT)) {

		HGLOBAL mem = GetClipboardData(CF_UNICODETEXT);
		if (mem != NULL) {

			LPWSTR ptr = (LPWSTR)GlobalLock(mem);
			if (ptr != NULL) {

				ret = String((CharType*)ptr);
				GlobalUnlock(mem);
			};
		};

	} else if (IsClipboardFormatAvailable(CF_TEXT)) {

		HGLOBAL mem = GetClipboardData(CF_UNICODETEXT);
		if (mem != NULL) {

			LPTSTR ptr = (LPTSTR)GlobalLock(mem);
			if (ptr != NULL) {

				ret.parse_utf8((const char*)ptr);
				GlobalUnlock(mem);
			};
		};
	};

	CloseClipboard();

	return ret;
	*/
	return "";
};


void OSWinrt::input_event(InputEvent &p_event) {
	p_event.ID = ++last_id;
	input->parse_input_event(p_event);
};

void OSWinrt::delete_main_loop() {

	if (main_loop)
		memdelete(main_loop);
	main_loop=NULL;
}

void OSWinrt::set_main_loop( MainLoop * p_main_loop ) {

	input->set_main_loop(p_main_loop);
	main_loop=p_main_loop;
}

void OSWinrt::finalize() {

	if(main_loop)
		memdelete(main_loop);

	main_loop=NULL;
	
	visual_server->finish();
	memdelete(visual_server);
#ifdef OPENGL_ENABLED
	if (gl_context)
		memdelete(gl_context);
#endif
	if (rasterizer)
		memdelete(rasterizer);

	spatial_sound_server->finish();
	memdelete(spatial_sound_server);
	spatial_sound_2d_server->finish();
	memdelete(spatial_sound_2d_server);

	//if (debugger_connection_console) {
//		memdelete(debugger_connection_console);
//}

	audio_server->finish();
	memdelete(audio_server);
	memdelete(sample_manager);

	memdelete(input);

	physics_server->finish();
	memdelete(physics_server);

	physics_2d_server->finish();
	memdelete(physics_2d_server);

}
void OSWinrt::finalize_core() {

	if (mempool_dynamic)
		memdelete( mempool_dynamic );
	if (mempool_static)
		delete mempool_static;

}

void OSWinrt::vprint(const char* p_format, va_list p_list, bool p_stderr) {

	char buf[16384+1];
	int len = vsnprintf(buf,16384,p_format,p_list);
	if (len<=0)
		return;
	buf[len]=0;


	int wlen = MultiByteToWideChar(CP_UTF8,0,buf,len,NULL,0);
	if (wlen<0)
		return;

	wchar_t *wbuf = (wchar_t*)malloc((len+1)*sizeof(wchar_t));
	MultiByteToWideChar(CP_UTF8,0,buf,len,wbuf,wlen);
	wbuf[wlen]=0;

	if (p_stderr)
		fwprintf(stderr,L"%s",wbuf);
	else
		wprintf(L"%s",wbuf);

#ifdef STDOUT_FILE
	//vwfprintf(stdo,p_format,p_list);
#endif
	free(wbuf);

	fflush(stdout);
};

void OSWinrt::alert(const String& p_alert,const String& p_title) {

	print_line("ALERT: "+p_alert);
}

void OSWinrt::set_mouse_mode(MouseMode p_mode) {

}

OSWinrt::MouseMode OSWinrt::get_mouse_mode() const{

	return mouse_mode;
}



Point2 OSWinrt::get_mouse_pos() const {

	return Point2(old_x, old_y);
}

int OSWinrt::get_mouse_button_state() const {

	return last_button_state;
}

void OSWinrt::set_window_title(const String& p_title) {

}

void OSWinrt::set_video_mode(const VideoMode& p_video_mode,int p_screen) {

	video_mode = p_video_mode;
}
OS::VideoMode OSWinrt::get_video_mode(int p_screen) const {

	return video_mode;
}
void OSWinrt::get_fullscreen_mode_list(List<VideoMode> *p_list,int p_screen) const {

	
}

void OSWinrt::print_error(const char* p_function,const char* p_file,int p_line,const char *p_code,const char*p_rationale,ErrorType p_type) {

	if (p_rationale && p_rationale[0]) {

		print("\E[1;31;40mERROR: %s: \E[1;37;40m%s\n",p_function,p_rationale);
		print("\E[0;31;40m   At: %s:%i.\E[0;0;37m\n",p_file,p_line);

	} else {
		print("\E[1;31;40mERROR: %s: \E[1;37;40m%s\n",p_function,p_code);
		print("\E[0;31;40m   At: %s:%i.\E[0;0;37m\n",p_file,p_line);

	}
}


String OSWinrt::get_name() {

	return "WinRT";
}

OS::Date OSWinrt::get_date() const {

	SYSTEMTIME systemtime;
	GetSystemTime(&systemtime);
	Date date;
	date.day=systemtime.wDay;
	date.month=Month(systemtime.wMonth);
	date.weekday=Weekday(systemtime.wDayOfWeek);
	date.year=systemtime.wYear;
	date.dst=false;
	return date;
}
OS::Time OSWinrt::get_time() const {

	SYSTEMTIME systemtime;
	GetSystemTime(&systemtime);

	Time time;
	time.hour=systemtime.wHour;
	time.min=systemtime.wMinute;
	time.sec=systemtime.wSecond;
	return time;
}

uint64_t OSWinrt::get_unix_time() const {

	FILETIME ft;
	SYSTEMTIME st;
	GetSystemTime(&st);
	SystemTimeToFileTime(&st, &ft);

	SYSTEMTIME ep;
	ep.wYear = 1970;
	ep.wMonth = 1;
	ep.wDayOfWeek = 4;
	ep.wDay = 1;
	ep.wHour = 0;
	ep.wMinute = 0;
	ep.wSecond = 0;
	ep.wMilliseconds = 0;
	FILETIME fep;
	SystemTimeToFileTime(&ep, &fep);

	return (*(uint64_t*)&ft - *(uint64_t*)&fep) / 10000000;
};

void OSWinrt::delay_usec(uint32_t p_usec) const {

	int msec = p_usec < 1000 ? 1 : p_usec / 1000;

	// no Sleep()
	WaitForSingleObjectEx(GetCurrentThread(), msec, false);
	
}
uint64_t OSWinrt::get_ticks_usec() const {

	uint64_t ticks;
	uint64_t time;
	// This is the number of clock ticks since start
	QueryPerformanceCounter((LARGE_INTEGER *)&ticks);
	// Divide by frequency to get the time in seconds
	time = ticks * 1000000L / ticks_per_second;
	// Subtract the time at game start to get
	// the time since the game started
	time -= ticks_start;
	return time;
}


void OSWinrt::process_events() {

}

void OSWinrt::set_cursor_shape(CursorShape p_shape) {

}

Error OSWinrt::execute(const String& p_path, const List<String>& p_arguments,bool p_blocking,ProcessID *r_child_id,String* r_pipe,int *r_exitcode) {

	return FAILED;
};

Error OSWinrt::kill(const ProcessID& p_pid) {

	return FAILED;
};

Error OSWinrt::set_cwd(const String& p_cwd) {

	return FAILED;
}

String OSWinrt::get_executable_path() const {

	return "";
}

void OSWinrt::set_icon(const Image& p_icon) {

}


bool OSWinrt::has_environment(const String& p_var) const {

	return false;
};

String OSWinrt::get_environment(const String& p_var) const {

	return "";
};

String OSWinrt::get_stdin_string(bool p_block) {

	return String();
}


void OSWinrt::move_window_to_foreground() {

}

Error OSWinrt::shell_open(String p_uri) {

	return FAILED;
}


String OSWinrt::get_locale() const {

#if WINAPI_FAMILY == WINAPI_FAMILY_PHONE_APP // this should work on phone 8.1, but it doesn't
	return "en";
#else
	Platform::String ^language = Windows::Globalization::Language::CurrentInputMethodLanguageTag;
	return language->Data();
#endif
}

void OSWinrt::release_rendering_thread() {

	gl_context->release_current();
}

void OSWinrt::make_rendering_thread() {

	gl_context->make_current();
}

void OSWinrt::swap_buffers() {

	gl_context->swap_buffers();
}


void OSWinrt::run() {

	if (!main_loop)
		return;
		
	main_loop->init();
		
	uint64_t last_ticks=get_ticks_usec();
	
	int frames=0;
	uint64_t frame=0;
	
	while (!force_quit) {
	
		CoreWindow::GetForCurrentThread()->Dispatcher->ProcessEvents(CoreProcessEventsOption::ProcessAllIfPresent);
		process_events(); // get rid of pending events
		if (Main::iteration()==true)
			break;
	};
	
	main_loop->finish();

}



MainLoop *OSWinrt::get_main_loop() const {

	return main_loop;
}


String OSWinrt::get_data_dir() const {

	Windows::Storage::StorageFolder ^data_folder = Windows::Storage::ApplicationData::Current->LocalFolder;

	return data_folder->Path->Data();
}


OSWinrt::OSWinrt() {

	key_event_pos=0;
	force_quit=false;
	alt_mem=false;
	gr_mem=false;
	shift_mem=false;
	control_mem=false;
	meta_mem=false;
	minimized = false;

	pressrc=0;
	old_invalid=true;
	last_id=0;
	mouse_mode=MOUSE_MODE_VISIBLE;
#ifdef STDOUT_FILE
	stdo=fopen("stdout.txt","wb");
#endif

	gl_context = NULL;

	AudioDriverManagerSW::add_driver(&audio_driver);
}


OSWinrt::~OSWinrt()
{
#ifdef STDOUT_FILE
	fclose(stdo);
#endif
}


