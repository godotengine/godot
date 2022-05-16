/*************************************************************************/
/*  os_windows.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef OS_WINDOWS_H
#define OS_WINDOWS_H

#include "context_gl_windows.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "crash_handler_windows.h"
#include "drivers/unix/ip_unix.h"
#include "drivers/wasapi/audio_driver_wasapi.h"
#include "drivers/winmidi/midi_driver_winmidi.h"
#include "key_mapping_windows.h"
#include "main/input_default.h"
#include "power_windows.h"
#include "servers/audio_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"
#ifdef XAUDIO2_ENABLED
#include "drivers/xaudio2/audio_driver_xaudio2.h"
#endif

#include <dwmapi.h>
#include <fcntl.h>
#include <io.h>
#include <shellapi.h>
#include <stdio.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
// WinTab API
#define WT_PACKET 0x7FF0
#define WT_PROXIMITY 0x7FF5
#define WT_INFOCHANGE 0x7FF6
#define WT_CSRCHANGE 0x7FF7

#define WTI_DEFSYSCTX 4
#define WTI_DEVICES 100
#define DVC_NPRESSURE 15
#define DVC_TPRESSURE 16
#define DVC_ORIENTATION 17
#define DVC_ROTATION 18

#define CXO_MESSAGES 0x0004
#define PK_NORMAL_PRESSURE 0x0400
#define PK_TANGENT_PRESSURE 0x0800
#define PK_ORIENTATION 0x1000

typedef struct tagLOGCONTEXTW {
	WCHAR lcName[40];
	UINT lcOptions;
	UINT lcStatus;
	UINT lcLocks;
	UINT lcMsgBase;
	UINT lcDevice;
	UINT lcPktRate;
	DWORD lcPktData;
	DWORD lcPktMode;
	DWORD lcMoveMask;
	DWORD lcBtnDnMask;
	DWORD lcBtnUpMask;
	LONG lcInOrgX;
	LONG lcInOrgY;
	LONG lcInOrgZ;
	LONG lcInExtX;
	LONG lcInExtY;
	LONG lcInExtZ;
	LONG lcOutOrgX;
	LONG lcOutOrgY;
	LONG lcOutOrgZ;
	LONG lcOutExtX;
	LONG lcOutExtY;
	LONG lcOutExtZ;
	DWORD lcSensX;
	DWORD lcSensY;
	DWORD lcSensZ;
	BOOL lcSysMode;
	int lcSysOrgX;
	int lcSysOrgY;
	int lcSysExtX;
	int lcSysExtY;
	DWORD lcSysSensX;
	DWORD lcSysSensY;
} LOGCONTEXTW;

typedef struct tagAXIS {
	LONG axMin;
	LONG axMax;
	UINT axUnits;
	DWORD axResolution;
} AXIS;

typedef struct tagORIENTATION {
	int orAzimuth;
	int orAltitude;
	int orTwist;
} ORIENTATION;

typedef struct tagPACKET {
	int pkNormalPressure;
	int pkTangentPressure;
	ORIENTATION pkOrientation;
} PACKET;

typedef HANDLE(WINAPI *WTOpenPtr)(HWND p_window, LOGCONTEXTW *p_ctx, BOOL p_enable);
typedef BOOL(WINAPI *WTClosePtr)(HANDLE p_ctx);
typedef UINT(WINAPI *WTInfoPtr)(UINT p_category, UINT p_index, LPVOID p_output);
typedef BOOL(WINAPI *WTPacketPtr)(HANDLE p_ctx, UINT p_param, LPVOID p_packets);
typedef BOOL(WINAPI *WTEnablePtr)(HANDLE p_ctx, BOOL p_enable);

// Windows Ink API
#ifndef POINTER_STRUCTURES

#define POINTER_STRUCTURES

typedef DWORD POINTER_INPUT_TYPE;
typedef UINT32 POINTER_FLAGS;
typedef UINT32 PEN_FLAGS;
typedef UINT32 PEN_MASK;

#ifndef PEN_MASK_PRESSURE
#define PEN_MASK_PRESSURE 0x00000001
#endif

#ifndef PEN_MASK_TILT_X
#define PEN_MASK_TILT_X 0x00000004
#endif

#ifndef PEN_MASK_TILT_Y
#define PEN_MASK_TILT_Y 0x00000008
#endif

#ifndef POINTER_MESSAGE_FLAG_FIRSTBUTTON
#define POINTER_MESSAGE_FLAG_FIRSTBUTTON 0x00000010
#endif

enum tagPOINTER_INPUT_TYPE {
	PT_POINTER = 0x00000001,
	PT_TOUCH = 0x00000002,
	PT_PEN = 0x00000003,
	PT_MOUSE = 0x00000004,
	PT_TOUCHPAD = 0x00000005
};

typedef enum tagPOINTER_BUTTON_CHANGE_TYPE {
	POINTER_CHANGE_NONE,
	POINTER_CHANGE_FIRSTBUTTON_DOWN,
	POINTER_CHANGE_FIRSTBUTTON_UP,
	POINTER_CHANGE_SECONDBUTTON_DOWN,
	POINTER_CHANGE_SECONDBUTTON_UP,
	POINTER_CHANGE_THIRDBUTTON_DOWN,
	POINTER_CHANGE_THIRDBUTTON_UP,
	POINTER_CHANGE_FOURTHBUTTON_DOWN,
	POINTER_CHANGE_FOURTHBUTTON_UP,
	POINTER_CHANGE_FIFTHBUTTON_DOWN,
	POINTER_CHANGE_FIFTHBUTTON_UP,
} POINTER_BUTTON_CHANGE_TYPE;

typedef struct tagPOINTER_INFO {
	POINTER_INPUT_TYPE pointerType;
	UINT32 pointerId;
	UINT32 frameId;
	POINTER_FLAGS pointerFlags;
	HANDLE sourceDevice;
	HWND hwndTarget;
	POINT ptPixelLocation;
	POINT ptHimetricLocation;
	POINT ptPixelLocationRaw;
	POINT ptHimetricLocationRaw;
	DWORD dwTime;
	UINT32 historyCount;
	INT32 InputData;
	DWORD dwKeyStates;
	UINT64 PerformanceCount;
	POINTER_BUTTON_CHANGE_TYPE ButtonChangeType;
} POINTER_INFO;

typedef struct tagPOINTER_PEN_INFO {
	POINTER_INFO pointerInfo;
	PEN_FLAGS penFlags;
	PEN_MASK penMask;
	UINT32 pressure;
	UINT32 rotation;
	INT32 tiltX;
	INT32 tiltY;
} POINTER_PEN_INFO;

#endif

#ifndef WM_POINTERUPDATE
#define WM_POINTERUPDATE 0x0245
#endif

#ifndef WM_POINTERENTER
#define WM_POINTERENTER 0x0249
#endif

#ifndef WM_POINTERLEAVE
#define WM_POINTERLEAVE 0x024A
#endif

typedef BOOL(WINAPI *GetPointerTypePtr)(uint32_t p_id, POINTER_INPUT_TYPE *p_type);
typedef BOOL(WINAPI *GetPointerPenInfoPtr)(uint32_t p_id, POINTER_PEN_INFO *p_pen_info);

typedef struct {
	BYTE bWidth; // Width, in pixels, of the image
	BYTE bHeight; // Height, in pixels, of the image
	BYTE bColorCount; // Number of colors in image (0 if >=8bpp)
	BYTE bReserved; // Reserved ( must be 0)
	WORD wPlanes; // Color Planes
	WORD wBitCount; // Bits per pixel
	DWORD dwBytesInRes; // How many bytes in this resource?
	DWORD dwImageOffset; // Where in the file is this image?
} ICONDIRENTRY, *LPICONDIRENTRY;

typedef struct {
	WORD idReserved; // Reserved (must be 0)
	WORD idType; // Resource Type (1 for icons)
	WORD idCount; // How many images?
	ICONDIRENTRY idEntries[1]; // An entry for each image (idCount of 'em)
} ICONDIR, *LPICONDIR;

class JoypadWindows;
class OS_Windows : public OS {
	String tablet_driver;
	Vector<String> tablet_drivers;

	// WinTab API
	static bool wintab_available;
	static WTOpenPtr wintab_WTOpen;
	static WTClosePtr wintab_WTClose;
	static WTInfoPtr wintab_WTInfo;
	static WTPacketPtr wintab_WTPacket;
	static WTEnablePtr wintab_WTEnable;

	// Windows Ink API
	static bool winink_available;
	static GetPointerTypePtr win8p_GetPointerType;
	static GetPointerPenInfoPtr win8p_GetPointerPenInfo;

	HANDLE wtctx;
	LOGCONTEXTW wtlc;
	int min_pressure;
	int max_pressure;
	bool tilt_supported;
	bool block_mm = false;

	int last_pressure_update;
	float last_pressure;
	Vector2 last_tilt;

	enum {
		KEY_EVENT_BUFFER_SIZE = 512
	};

#ifdef STDOUT_FILE
	FILE *stdo;
#endif

	struct KeyEvent {
		bool alt, shift, control, meta;
		UINT uMsg;
		WPARAM wParam;
		LPARAM lParam;
	};

	KeyEvent key_event_buffer[KEY_EVENT_BUFFER_SIZE];
	int key_event_pos;

	uint64_t ticks_start;
	uint64_t ticks_per_second;

	bool old_invalid;
	bool outside;
	int old_x, old_y;
	Point2i center;
#if defined(OPENGL_ENABLED)
	ContextGL_Windows *gl_context;
#endif
	VisualServer *visual_server;
	int pressrc;
	HINSTANCE hInstance; // Holds The Instance Of The Application
	HWND hWnd;

	Vector<Vector2> mpath;
	Point2 last_pos;

	bool layered_window;

	uint32_t move_timer_id;

	Ref<Image> icon;

	HCURSOR hCursor;

	Size2 min_size;
	Size2 max_size;

	Size2 window_rect;
	VideoMode video_mode;
	bool preserve_window_size = false;

	MainLoop *main_loop;

	WNDPROC user_proc;

	// IME
	HIMC im_himc;
	Vector2 im_position;

	MouseMode mouse_mode;
	int restore_mouse_trails;
	bool alt_mem;
	bool gr_mem;
	bool shift_mem;
	bool control_mem;
	bool meta_mem;
	bool force_quit;
	bool window_has_focus;
	uint32_t last_button_state;
	bool use_raw_input;
	bool drop_events;

	HCURSOR cursors[CURSOR_MAX] = { NULL };
	CursorShape cursor_shape;
	Map<CursorShape, Vector<Variant>> cursors_cache;

	InputDefault *input;
	JoypadWindows *joypad;
	Map<int, Vector2> touch_state;

	PowerWindows *power_manager;

	int video_driver_index;
#ifdef WASAPI_ENABLED
	AudioDriverWASAPI driver_wasapi;
#endif
#ifdef XAUDIO2_ENABLED
	AudioDriverXAudio2 driver_xaudio2;
#endif
#ifdef WINMIDI_ENABLED
	MIDIDriverWinMidi driver_midi;
#endif

	CrashHandler crash_handler;

	void _drag_event(float p_x, float p_y, int idx);
	void _touch_event(bool p_pressed, float p_x, float p_y, int idx);

	void _update_window_style(bool p_repaint = true, bool p_maximized = false);
	void _update_window_mouse_passthrough();

	void _set_mouse_mode_impl(MouseMode p_mode);

	// functions used by main to initialize/deinitialize the OS
protected:
	virtual int get_current_video_driver() const;

	virtual void initialize_core();
	virtual Error initialize(const VideoMode &p_desired, int p_video_driver, int p_audio_driver);

	virtual bool is_offscreen_gl_available() const;
	virtual void set_offscreen_gl_current(bool p_current);

	virtual void set_main_loop(MainLoop *p_main_loop);
	virtual void delete_main_loop();

	virtual void finalize();
	virtual void finalize_core();

	void process_events();
	void process_key_events();

	String _quote_command_line_argument(const String &p_text) const;

	struct ProcessInfo {
		STARTUPINFO si;
		PROCESS_INFORMATION pi;
	};
	Map<ProcessID, ProcessInfo> *process_map;

	bool pre_fs_valid;
	RECT pre_fs_rect;
	bool maximized;
	bool minimized;
	bool borderless;
	bool window_focused;
	bool was_maximized;

public:
	LRESULT WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");
	String get_stdin_string(bool p_block);

	void set_mouse_mode(MouseMode p_mode);
	MouseMode get_mouse_mode() const;

	virtual void warp_mouse_position(const Point2 &p_to);
	virtual Point2 get_mouse_position() const;
	void update_real_mouse_position();
	virtual int get_mouse_button_state() const;
	virtual void set_window_title(const String &p_title);
	virtual void set_window_mouse_passthrough(const PoolVector2Array &p_region);

	virtual void set_video_mode(const VideoMode &p_video_mode, int p_screen = 0);
	virtual VideoMode get_video_mode(int p_screen = 0) const;
	virtual void get_fullscreen_mode_list(List<VideoMode> *p_list, int p_screen = 0) const;

	virtual int get_tablet_driver_count() const;
	virtual String get_tablet_driver_name(int p_driver) const;
	virtual String get_current_tablet_driver() const;
	virtual void set_current_tablet_driver(const String &p_driver);

	virtual int get_screen_count() const;
	virtual int get_current_screen() const;
	virtual void set_current_screen(int p_screen);
	virtual Point2 get_screen_position(int p_screen = -1) const;
	virtual Size2 get_screen_size(int p_screen = -1) const;
	virtual int get_screen_dpi(int p_screen = -1) const;
	virtual float get_screen_refresh_rate(int p_screen = -1) const;

	virtual Point2 get_window_position() const;
	virtual void set_window_position(const Point2 &p_position);
	virtual Size2 get_window_size() const;
	virtual Size2 get_real_window_size() const;
	virtual Size2 get_max_window_size() const;
	virtual Size2 get_min_window_size() const;
	virtual void set_min_window_size(const Size2 p_size);
	virtual void set_max_window_size(const Size2 p_size);
	virtual void set_window_size(const Size2 p_size);
	virtual void set_window_fullscreen(bool p_enabled);
	virtual bool is_window_fullscreen() const;
	virtual void set_window_resizable(bool p_enabled);
	virtual bool is_window_resizable() const;
	virtual void set_window_minimized(bool p_enabled);
	virtual bool is_window_minimized() const;
	virtual void set_window_maximized(bool p_enabled);
	virtual bool is_window_maximized() const;
	virtual void set_window_always_on_top(bool p_enabled);
	virtual bool is_window_always_on_top() const;
	virtual bool is_window_focused() const;
	virtual void request_attention();
	virtual void *get_native_handle(int p_handle_type);

	virtual void set_borderless_window(bool p_borderless);
	virtual bool get_borderless_window();

	virtual bool get_window_per_pixel_transparency_enabled() const;
	virtual void set_window_per_pixel_transparency_enabled(bool p_enabled);

	virtual Error open_dynamic_library(const String p_path, void *&p_library_handle, bool p_also_set_library_path = false);
	virtual Error close_dynamic_library(void *p_library_handle);
	virtual Error get_dynamic_library_symbol_handle(void *p_library_handle, const String p_name, void *&p_symbol_handle, bool p_optional = false);

	virtual MainLoop *get_main_loop() const;

	virtual uint64_t get_embedded_pck_offset() const;

	virtual String get_name() const;

	virtual Date get_date(bool utc) const;
	virtual Time get_time(bool utc) const;
	virtual TimeZoneInfo get_time_zone_info() const;
	virtual uint64_t get_unix_time() const;
	virtual uint64_t get_system_time_secs() const;
	virtual uint64_t get_system_time_msecs() const;

	virtual bool can_draw() const;
	virtual Error set_cwd(const String &p_cwd);

	virtual void delay_usec(uint32_t p_usec) const;
	virtual uint64_t get_ticks_usec() const;

	virtual Error execute(const String &p_path, const List<String> &p_arguments, bool p_blocking = true, ProcessID *r_child_id = NULL, String *r_pipe = NULL, int *r_exitcode = NULL, bool read_stderr = false, Mutex *p_pipe_mutex = NULL, bool p_open_console = false);
	virtual Error kill(const ProcessID &p_pid);
	virtual int get_process_id() const;
	virtual bool is_process_running(const ProcessID &p_pid) const;

	virtual bool has_environment(const String &p_var) const;
	virtual String get_environment(const String &p_var) const;
	virtual bool set_environment(const String &p_var, const String &p_value) const;

	virtual void set_clipboard(const String &p_text);
	virtual String get_clipboard() const;

	void set_cursor_shape(CursorShape p_shape);
	CursorShape get_cursor_shape() const;
	virtual void set_custom_mouse_cursor(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot);
	void GetMaskBitmaps(HBITMAP hSourceBitmap, COLORREF clrTransparent, OUT HBITMAP &hAndMaskBitmap, OUT HBITMAP &hXorMaskBitmap);

	void set_native_icon(const String &p_filename);
	void set_icon(const Ref<Image> &p_icon);

	virtual String get_executable_path() const;

	virtual String get_locale() const;

	virtual int get_processor_count() const;
	virtual String get_processor_name() const;

	virtual LatinKeyboardVariant get_latin_keyboard_variant() const;
	virtual int keyboard_get_layout_count() const;
	virtual int keyboard_get_current_layout() const;
	virtual void keyboard_set_current_layout(int p_index);
	virtual String keyboard_get_layout_language(int p_index) const;
	virtual String keyboard_get_layout_name(int p_index) const;
	virtual uint32_t keyboard_get_scancode_from_physical(uint32_t p_scancode) const;

	virtual void enable_for_stealing_focus(ProcessID pid);
	virtual void move_window_to_foreground();

	virtual String get_config_path() const;
	virtual String get_data_path() const;
	virtual String get_cache_path() const;
	virtual String get_godot_dir_name() const;

	virtual String get_system_dir(SystemDir p_dir, bool p_shared_storage = true) const;
	virtual String get_user_data_dir() const;

	virtual String get_unique_id() const;

	virtual void set_ime_active(const bool p_active);
	virtual void set_ime_position(const Point2 &p_pos);

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual Error shell_open(String p_uri);

	void run();

	virtual bool get_swap_ok_cancel() { return true; }

	virtual bool is_joy_known(int p_device);
	virtual String get_joy_guid(int p_device) const;

	virtual void _set_use_vsync(bool p_enable);
	//virtual bool is_vsync_enabled() const;

	virtual OS::PowerState get_power_state();
	virtual int get_power_seconds_left();
	virtual int get_power_percent_left();

	virtual bool _check_internal_feature_support(const String &p_feature);

	void disable_crash_handler();
	bool is_disable_crash_handler() const;
	virtual void initialize_debugging();

	void force_process_input();

	virtual Error move_to_trash(const String &p_path);

	virtual void process_and_drop_events();

	OS_Windows(HINSTANCE _hInstance);
	~OS_Windows();
};

#endif
