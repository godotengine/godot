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
#include "key_mapping_windows.h"
#include "servers/audio_server.h"
#include "servers/visual/rasterizer.h"
#include "servers/visual_server.h"
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

class DisplayServerWindows : public DisplayServer {
	//No need to register, it's platform-specific and nothing is added
	//GDCLASS(DisplayServerWindows, DisplayServer)

	_THREAD_SAFE_CLASS_

	static GetPointerTypePtr win8p_GetPointerType;
	static GetPointerPenInfoPtr win8p_GetPointerPenInfo;

	enum {
		KEY_EVENT_BUFFER_SIZE = 512
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

	struct WindowData {
		HWND hWnd;
		//layered window

		bool preserve_window_size = false;
		bool pre_fs_valid = false;
		RECT pre_fs_rect;
		bool maximized = false;
		bool minimized = false;
		bool borderless = false;
		bool window_focused = false;
		bool was_maximized = false;
		bool window_has_focus = false;

		HBITMAP hBitmap; //DIB section for layered window
		uint8_t *dib_data = nullptr;
		Size2 dib_size;
		HDC hDC_dib;
		Size2 min_size;
		Size2 max_size;

		Size2 window_rect;

		// IME
		HIMC im_himc;
		Vector2 im_position;

		bool layered_window = false;
	};

	Map<WindowID, WindowData> windows;

	Point2 last_pos;

	uint32_t move_timer_id;

	HCURSOR hCursor;

	WNDPROC user_proc;

	MouseMode mouse_mode;
	bool alt_mem = false;
	bool gr_mem = false;
	bool shift_mem = false;
	bool control_mem = false;
	bool meta_mem = false;
	bool force_quit = false;
	uint32_t last_button_state = 0;
	bool use_raw_input = false;
	bool drop_events = false;
	bool console_visible = false;

	HCURSOR cursors[CURSOR_MAX] = { NULL };
	CursorShape cursor_shape;
	Map<CursorShape, Vector<Variant> > cursors_cache;

	void _drag_event(float p_x, float p_y, int idx);
	void _touch_event(bool p_pressed, float p_x, float p_y, int idx);

	void _update_window_style(bool p_repaint = true, bool p_maximized = false);

	void _set_mouse_mode_impl(MouseMode p_mode);

	struct ProcessInfo {

		STARTUPINFO si;
		PROCESS_INFORMATION pi;
	};
	Map<ProcessID, ProcessInfo> *process_map;

public:
	DisplayServerWindows(const String &p_rendering_driver, WindowMode p_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error);
};

#endif // DISPLAY_SERVER_WINDOWS_H
