/*************************************************************************/
/*  display_server_windows.h                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef DISPLAY_SERVER_WINDOWS_H
#define DISPLAY_SERVER_WINDOWS_H

#include "servers/display_server.h"

#include "core/input/input_filter.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "crash_handler_windows.h"
#include "drivers/unix/ip_unix.h"
#include "drivers/wasapi/audio_driver_wasapi.h"
#include "drivers/winmidi/midi_driver_winmidi.h"
#include "joypad_windows.h"
#include "key_mapping_windows.h"
#include "servers/audio_server.h"
#include "servers/rendering/rasterizer.h"
#include "servers/rendering/rasterizer_rd/rasterizer_rd.h"
#include "servers/rendering_server.h"

#ifdef XAUDIO2_ENABLED
#include "drivers/xaudio2/audio_driver_xaudio2.h"
#endif

#if defined(OPENGL_ENABLED)
#include "context_gl_windows.h"
#endif

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_device_vulkan.h"
#include "platform/windows/vulkan_context_win.h"
#endif

#include <fcntl.h>
#include <io.h>
#include <stdio.h>
#include <windows.h>
#include <windowsx.h>

#ifndef POINTER_STRUCTURES

#define POINTER_STRUCTURES

typedef DWORD POINTER_INPUT_TYPE;
typedef UINT32 POINTER_FLAGS;
typedef UINT32 PEN_FLAGS;
typedef UINT32 PEN_MASK;

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

#endif //POINTER_STRUCTURES

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

class DisplayServerWindows : public DisplayServer {
	//No need to register, it's platform-specific and nothing is added
	//GDCLASS(DisplayServerWindows, DisplayServer)

	_THREAD_SAFE_CLASS_

	static GetPointerTypePtr win8p_GetPointerType;
	static GetPointerPenInfoPtr win8p_GetPointerPenInfo;

	void GetMaskBitmaps(HBITMAP hSourceBitmap, COLORREF clrTransparent, OUT HBITMAP &hAndMaskBitmap, OUT HBITMAP &hXorMaskBitmap);

	enum {
		KEY_EVENT_BUFFER_SIZE = 512
	};

	struct KeyEvent {

		WindowID window_id;
		bool alt, shift, control, meta;
		UINT uMsg;
		WPARAM wParam;
		LPARAM lParam;
	};

	KeyEvent key_event_buffer[KEY_EVENT_BUFFER_SIZE];
	int key_event_pos;

	bool old_invalid;
	bool outside;
	int old_x, old_y;
	Point2i center;

#if defined(OPENGL_ENABLED)
	ContextGL_Windows *context_gles2;
#endif

#if defined(VULKAN_ENABLED)
	VulkanContextWindows *context_vulkan;
	RenderingDeviceVulkan *rendering_device_vulkan;
#endif

	Map<int, Vector2> touch_state;

	int pressrc;
	HINSTANCE hInstance; // Holds The Instance Of The Application
	String rendering_driver;

	struct WindowData {
		HWND hWnd;
		//layered window

		bool preserve_window_size = false;
		bool pre_fs_valid = false;
		RECT pre_fs_rect;
		bool maximized = false;
		bool minimized = false;
		bool fullscreen = false;
		bool borderless = false;
		bool resizable = true;
		bool window_focused = false;
		bool was_maximized = false;
		bool always_on_top = false;
		bool no_focus = false;
		bool window_has_focus = false;

		HBITMAP hBitmap; //DIB section for layered window
		uint8_t *dib_data = nullptr;
		Size2 dib_size;
		HDC hDC_dib;
		Size2 min_size;
		Size2 max_size;
		int width = 0, height = 0;

		Size2 window_rect;
		Point2 last_pos;

		ObjectID instance_id;

		// IME
		HIMC im_himc;
		Vector2 im_position;

		bool layered_window = false;

		Callable rect_changed_callback;
		Callable event_callback;
		Callable input_event_callback;
		Callable input_text_callback;
		Callable drop_files_callback;

		WindowID transient_parent = INVALID_WINDOW_ID;
		Set<WindowID> transient_children;
	};

	JoypadWindows *joypad;

	WindowID _create_window(WindowMode p_mode, uint32_t p_flags, const Rect2i &p_rect);
	WindowID window_id_counter = MAIN_WINDOW_ID;
	Map<WindowID, WindowData> windows;

	WindowID last_focused_window = INVALID_WINDOW_ID;

	uint32_t move_timer_id;

	HCURSOR hCursor;

	WNDPROC user_proc = nullptr;

	void _send_window_event(const WindowData &wd, WindowEvent p_event);
	void _get_window_style(bool p_main_window, bool p_fullscreen, bool p_borderless, bool p_resizable, bool p_maximized, bool p_no_activate_focus, DWORD &r_style, DWORD &r_style_ex);

	MouseMode mouse_mode;
	bool alt_mem = false;
	bool gr_mem = false;
	bool shift_mem = false;
	bool control_mem = false;
	bool meta_mem = false;
	uint32_t last_button_state = 0;
	bool use_raw_input = false;
	bool drop_events = false;
	bool console_visible = false;

	WNDCLASSEXW wc;

	HCURSOR cursors[CURSOR_MAX] = { nullptr };
	CursorShape cursor_shape;
	Map<CursorShape, Vector<Variant>> cursors_cache;

	void _drag_event(WindowID p_window, float p_x, float p_y, int idx);
	void _touch_event(WindowID p_window, bool p_pressed, float p_x, float p_y, int idx);

	void _update_window_style(WindowID p_window, bool p_repaint = true, bool p_maximized = false);

	void _update_real_mouse_position(WindowID p_window);

	void _set_mouse_mode_impl(MouseMode p_mode);

	void _process_key_events();

	static void _dispatch_input_events(const Ref<InputEvent> &p_event);
	void _dispatch_input_event(const Ref<InputEvent> &p_event);

public:
	LRESULT WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

	virtual bool has_feature(Feature p_feature) const;
	virtual String get_name() const;

	virtual void alert(const String &p_alert, const String &p_title = "ALERT!");

	virtual void mouse_set_mode(MouseMode p_mode);
	virtual MouseMode mouse_get_mode() const;

	virtual void mouse_warp_to_position(const Point2i &p_to);
	virtual Point2i mouse_get_position() const;
	virtual int mouse_get_button_state() const;

	virtual void clipboard_set(const String &p_text);
	virtual String clipboard_get() const;

	virtual int get_screen_count() const;
	virtual Point2i screen_get_position(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual Size2i screen_get_size(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual Rect2i screen_get_usable_rect(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual int screen_get_dpi(int p_screen = SCREEN_OF_MAIN_WINDOW) const;
	virtual bool screen_is_touchscreen(int p_screen = SCREEN_OF_MAIN_WINDOW) const;

	virtual void screen_set_orientation(ScreenOrientation p_orientation, int p_screen = SCREEN_OF_MAIN_WINDOW);
	ScreenOrientation screen_get_orientation(int p_screen = SCREEN_OF_MAIN_WINDOW) const;

	virtual void screen_set_keep_on(bool p_enable); //disable screensaver
	virtual bool screen_is_kept_on() const;

	virtual Vector<DisplayServer::WindowID> get_window_list() const;

	virtual WindowID create_sub_window(WindowMode p_mode, uint32_t p_flags, const Rect2i &p_rect = Rect2i());
	virtual void delete_sub_window(WindowID p_window);

	virtual WindowID get_window_at_screen_position(const Point2i &p_position) const;

	virtual void window_attach_instance_id(ObjectID p_instance, WindowID p_window = MAIN_WINDOW_ID);
	virtual ObjectID window_get_attached_instance_id(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);

	virtual void window_set_window_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_input_event_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_input_text_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);

	virtual void window_set_drop_files_callback(const Callable &p_callable, WindowID p_window = MAIN_WINDOW_ID);

	virtual void window_set_title(const String &p_title, WindowID p_window = MAIN_WINDOW_ID);

	virtual int window_get_current_screen(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_current_screen(int p_screen, WindowID p_window = MAIN_WINDOW_ID);

	virtual Point2i window_get_position(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual void window_set_position(const Point2i &p_position, WindowID p_window = MAIN_WINDOW_ID);

	virtual void window_set_transient(WindowID p_window, WindowID p_parent);

	virtual void window_set_max_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID);
	virtual Size2i window_get_max_size(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_set_min_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID);
	virtual Size2i window_get_min_size(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_set_size(const Size2i p_size, WindowID p_window = MAIN_WINDOW_ID);
	virtual Size2i window_get_size(WindowID p_window = MAIN_WINDOW_ID) const;
	virtual Size2i window_get_real_size(WindowID p_window = MAIN_WINDOW_ID) const; //wtf is this? should probable use proper name

	virtual void window_set_mode(WindowMode p_mode, WindowID p_window = MAIN_WINDOW_ID);
	virtual WindowMode window_get_mode(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual bool window_is_maximize_allowed(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window = MAIN_WINDOW_ID);
	virtual bool window_get_flag(WindowFlags p_flag, WindowID p_window = MAIN_WINDOW_ID) const;

	virtual void window_request_attention(WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_move_to_foreground(WindowID p_window = MAIN_WINDOW_ID);

	virtual bool window_can_draw(WindowID p_window = MAIN_WINDOW_ID) const;

	virtual bool can_any_window_draw() const;

	virtual void window_set_ime_active(const bool p_active, WindowID p_window = MAIN_WINDOW_ID);
	virtual void window_set_ime_position(const Point2i &p_pos, WindowID p_window = MAIN_WINDOW_ID);

	virtual void console_set_visible(bool p_enabled);
	virtual bool is_console_visible() const;

	virtual void cursor_set_shape(CursorShape p_shape);
	virtual CursorShape cursor_get_shape() const;
	virtual void cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape = CURSOR_ARROW, const Vector2 &p_hotspot = Vector2());

	virtual bool get_swap_ok_cancel();

	virtual void enable_for_stealing_focus(OS::ProcessID pid);

	virtual LatinKeyboardVariant get_latin_keyboard_variant() const;

	virtual void process_events();

	virtual void force_process_and_drop_events();

	virtual void release_rendering_thread();
	virtual void make_rendering_thread();
	virtual void swap_buffers();

	virtual void set_native_icon(const String &p_filename);
	virtual void set_icon(const Ref<Image> &p_icon);

	virtual void vsync_set_use_via_compositor(bool p_enable);
	virtual bool vsync_is_using_via_compositor() const;

	virtual void set_context(Context p_context);

	static DisplayServer *create_func(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	static Vector<String> get_rendering_drivers_func();
	static void register_windows_driver();

	DisplayServerWindows(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
	~DisplayServerWindows();
};

#endif // DISPLAY_SERVER_WINDOWS_H
