/*************************************************************************/
/*  display_server_windows.cpp                                           */
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

#include "display_server_windows.h"

#include "core/io/marshalls.h"
#include "core/math/geometry_2d.h"
#include "main/main.h"
#include "os_windows.h"
#include "scene/resources/texture.h"

#include <avrt.h>

#if defined(GLES3_ENABLED)
#include "drivers/gles3/rasterizer_gles3.h"
#endif

#if defined(__GNUC__)
// Workaround GCC warning from -Wcast-function-type.
#define GetProcAddress (void *)GetProcAddress
#endif

static String format_error_message(DWORD id) {
	LPWSTR messageBuffer = nullptr;
	size_t size = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			nullptr, id, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPWSTR)&messageBuffer, 0, nullptr);

	String msg = "Error " + itos(id) + ": " + String::utf16((const char16_t *)messageBuffer, size);

	LocalFree(messageBuffer);

	return msg;
}

static void track_mouse_leave_event(HWND hWnd) {
	TRACKMOUSEEVENT tme;
	tme.cbSize = sizeof(TRACKMOUSEEVENT);
	tme.dwFlags = TME_LEAVE;
	tme.hwndTrack = hWnd;
	tme.dwHoverTime = HOVER_DEFAULT;
	TrackMouseEvent(&tme);
}

bool DisplayServerWindows::has_feature(Feature p_feature) const {
	switch (p_feature) {
		case FEATURE_SUBWINDOWS:
		case FEATURE_TOUCHSCREEN:
		case FEATURE_MOUSE:
		case FEATURE_MOUSE_WARP:
		case FEATURE_CLIPBOARD:
		case FEATURE_CURSOR_SHAPE:
		case FEATURE_CUSTOM_CURSOR_SHAPE:
		case FEATURE_IME:
		case FEATURE_WINDOW_TRANSPARENCY:
		case FEATURE_HIDPI:
		case FEATURE_ICON:
		case FEATURE_NATIVE_ICON:
		case FEATURE_SWAP_BUFFERS:
		case FEATURE_KEEP_SCREEN_ON:
			return true;
		default:
			return false;
	}
}

String DisplayServerWindows::get_name() const {
	return "Windows";
}

void DisplayServerWindows::_set_mouse_mode_impl(MouseMode p_mode) {
	if (windows.has(MAIN_WINDOW_ID) && (p_mode == MOUSE_MODE_CAPTURED || p_mode == MOUSE_MODE_CONFINED || p_mode == MOUSE_MODE_CONFINED_HIDDEN)) {
		// Mouse is grabbed (captured or confined).
		WindowData &wd = windows[MAIN_WINDOW_ID];

		RECT clipRect;
		GetClientRect(wd.hWnd, &clipRect);
		ClientToScreen(wd.hWnd, (POINT *)&clipRect.left);
		ClientToScreen(wd.hWnd, (POINT *)&clipRect.right);
		ClipCursor(&clipRect);
		if (p_mode == MOUSE_MODE_CAPTURED) {
			center = window_get_size() / 2;
			POINT pos = { (int)center.x, (int)center.y };
			ClientToScreen(wd.hWnd, &pos);
			SetCursorPos(pos.x, pos.y);
			SetCapture(wd.hWnd);
		}
	} else {
		// Mouse is free to move around (not captured or confined).
		ReleaseCapture();
		ClipCursor(nullptr);
	}

	if (p_mode == MOUSE_MODE_HIDDEN || p_mode == MOUSE_MODE_CAPTURED || p_mode == MOUSE_MODE_CONFINED_HIDDEN) {
		if (hCursor == nullptr) {
			hCursor = SetCursor(nullptr);
		} else {
			SetCursor(nullptr);
		}
	} else {
		CursorShape c = cursor_shape;
		cursor_shape = CURSOR_MAX;
		cursor_set_shape(c);
	}
}

void DisplayServerWindows::mouse_set_mode(MouseMode p_mode) {
	_THREAD_SAFE_METHOD_

	if (mouse_mode == p_mode) {
		// Already in the same mode; do nothing.
		return;
	}

	mouse_mode = p_mode;

	_set_mouse_mode_impl(p_mode);
}

DisplayServer::MouseMode DisplayServerWindows::mouse_get_mode() const {
	return mouse_mode;
}

void DisplayServerWindows::mouse_warp_to_position(const Point2i &p_to) {
	_THREAD_SAFE_METHOD_

	if (!windows.has(last_focused_window)) {
		return; // No focused window?
	}

	if (mouse_mode == MOUSE_MODE_CAPTURED) {
		old_x = p_to.x;
		old_y = p_to.y;
	} else {
		POINT p;
		p.x = p_to.x;
		p.y = p_to.y;
		ClientToScreen(windows[last_focused_window].hWnd, &p);

		SetCursorPos(p.x, p.y);
	}
}

Point2i DisplayServerWindows::mouse_get_position() const {
	POINT p;
	GetCursorPos(&p);
	return Point2i(p.x, p.y);
}

MouseButton DisplayServerWindows::mouse_get_button_state() const {
	return last_button_state;
}

void DisplayServerWindows::clipboard_set(const String &p_text) {
	_THREAD_SAFE_METHOD_

	if (!windows.has(last_focused_window)) {
		return; // No focused window?
	}

	// Convert LF line endings to CRLF in clipboard content.
	// Otherwise, line endings won't be visible when pasted in other software.
	String text = p_text.replace("\r\n", "\n").replace("\n", "\r\n"); // Avoid \r\r\n.

	if (!OpenClipboard(windows[last_focused_window].hWnd)) {
		ERR_FAIL_MSG("Unable to open clipboard.");
	}
	EmptyClipboard();

	Char16String utf16 = text.utf16();
	HGLOBAL mem = GlobalAlloc(GMEM_MOVEABLE, (utf16.length() + 1) * sizeof(WCHAR));
	ERR_FAIL_COND_MSG(mem == nullptr, "Unable to allocate memory for clipboard contents.");

	LPWSTR lptstrCopy = (LPWSTR)GlobalLock(mem);
	memcpy(lptstrCopy, utf16.get_data(), (utf16.length() + 1) * sizeof(WCHAR));
	GlobalUnlock(mem);

	SetClipboardData(CF_UNICODETEXT, mem);

	// Set the CF_TEXT version (not needed?).
	CharString utf8 = text.utf8();
	mem = GlobalAlloc(GMEM_MOVEABLE, utf8.length() + 1);
	ERR_FAIL_COND_MSG(mem == nullptr, "Unable to allocate memory for clipboard contents.");

	LPTSTR ptr = (LPTSTR)GlobalLock(mem);
	memcpy(ptr, utf8.get_data(), utf8.length());
	ptr[utf8.length()] = 0;
	GlobalUnlock(mem);

	SetClipboardData(CF_TEXT, mem);

	CloseClipboard();
}

String DisplayServerWindows::clipboard_get() const {
	_THREAD_SAFE_METHOD_

	if (!windows.has(last_focused_window)) {
		return String(); // No focused window?
	}

	String ret;
	if (!OpenClipboard(windows[last_focused_window].hWnd)) {
		ERR_FAIL_V_MSG("", "Unable to open clipboard.");
	};

	if (IsClipboardFormatAvailable(CF_UNICODETEXT)) {
		HGLOBAL mem = GetClipboardData(CF_UNICODETEXT);
		if (mem != nullptr) {
			LPWSTR ptr = (LPWSTR)GlobalLock(mem);
			if (ptr != nullptr) {
				ret = String::utf16((const char16_t *)ptr);
				GlobalUnlock(mem);
			};
		};

	} else if (IsClipboardFormatAvailable(CF_TEXT)) {
		HGLOBAL mem = GetClipboardData(CF_UNICODETEXT);
		if (mem != nullptr) {
			LPTSTR ptr = (LPTSTR)GlobalLock(mem);
			if (ptr != nullptr) {
				ret.parse_utf8((const char *)ptr);
				GlobalUnlock(mem);
			};
		};
	};

	CloseClipboard();

	return ret;
}

typedef struct {
	int count;
	int screen;
	HMONITOR monitor;
} EnumScreenData;

static BOOL CALLBACK _MonitorEnumProcScreen(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
	EnumScreenData *data = (EnumScreenData *)dwData;
	if (data->monitor == hMonitor) {
		data->screen = data->count;
	}

	data->count++;
	return TRUE;
}

static BOOL CALLBACK _MonitorEnumProcCount(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
	int *data = (int *)dwData;
	(*data)++;
	return TRUE;
}

int DisplayServerWindows::get_screen_count() const {
	_THREAD_SAFE_METHOD_

	int data = 0;
	EnumDisplayMonitors(nullptr, nullptr, _MonitorEnumProcCount, (LPARAM)&data);
	return data;
}

typedef struct {
	int count;
	int screen;
	Point2 pos;
} EnumPosData;

static BOOL CALLBACK _MonitorEnumProcPos(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
	EnumPosData *data = (EnumPosData *)dwData;
	if (data->count == data->screen) {
		data->pos.x = lprcMonitor->left;
		data->pos.y = lprcMonitor->top;
	}

	data->count++;
	return TRUE;
}

Point2i DisplayServerWindows::screen_get_position(int p_screen) const {
	_THREAD_SAFE_METHOD_

	EnumPosData data = { 0, p_screen == SCREEN_OF_MAIN_WINDOW ? window_get_current_screen() : p_screen, Point2() };
	EnumDisplayMonitors(nullptr, nullptr, _MonitorEnumProcPos, (LPARAM)&data);
	return data.pos;
}

typedef struct {
	int count;
	int screen;
	Size2 size;
} EnumSizeData;

typedef struct {
	int count;
	int screen;
	Rect2i rect;
} EnumRectData;

static BOOL CALLBACK _MonitorEnumProcSize(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
	EnumSizeData *data = (EnumSizeData *)dwData;
	if (data->count == data->screen) {
		data->size.x = lprcMonitor->right - lprcMonitor->left;
		data->size.y = lprcMonitor->bottom - lprcMonitor->top;
	}

	data->count++;
	return TRUE;
}

Size2i DisplayServerWindows::screen_get_size(int p_screen) const {
	_THREAD_SAFE_METHOD_

	EnumSizeData data = { 0, p_screen == SCREEN_OF_MAIN_WINDOW ? window_get_current_screen() : p_screen, Size2() };
	EnumDisplayMonitors(nullptr, nullptr, _MonitorEnumProcSize, (LPARAM)&data);
	return data.size;
}

static BOOL CALLBACK _MonitorEnumProcUsableSize(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
	EnumRectData *data = (EnumRectData *)dwData;
	if (data->count == data->screen) {
		MONITORINFO minfo;
		memset(&minfo, 0, sizeof(MONITORINFO));
		minfo.cbSize = sizeof(MONITORINFO);
		GetMonitorInfoA(hMonitor, &minfo);

		data->rect.position.x = minfo.rcWork.left;
		data->rect.position.y = minfo.rcWork.top;
		data->rect.size.x = minfo.rcWork.right - minfo.rcWork.left;
		data->rect.size.y = minfo.rcWork.bottom - minfo.rcWork.top;
	}

	data->count++;
	return TRUE;
}

Rect2i DisplayServerWindows::screen_get_usable_rect(int p_screen) const {
	_THREAD_SAFE_METHOD_

	EnumRectData data = { 0, p_screen == SCREEN_OF_MAIN_WINDOW ? window_get_current_screen() : p_screen, Rect2i() };
	EnumDisplayMonitors(nullptr, nullptr, _MonitorEnumProcUsableSize, (LPARAM)&data);
	return data.rect;
}

typedef struct {
	int count;
	int screen;
	int dpi;
} EnumDpiData;

enum _MonitorDpiType {
	MDT_Effective_DPI = 0,
	MDT_Angular_DPI = 1,
	MDT_Raw_DPI = 2,
	MDT_Default = MDT_Effective_DPI
};

static int QueryDpiForMonitor(HMONITOR hmon, _MonitorDpiType dpiType = MDT_Default) {
	int dpiX = 96, dpiY = 96;

	static HMODULE Shcore = nullptr;
	typedef HRESULT(WINAPI * GetDPIForMonitor_t)(HMONITOR hmonitor, _MonitorDpiType dpiType, UINT * dpiX, UINT * dpiY);
	static GetDPIForMonitor_t getDPIForMonitor = nullptr;

	if (Shcore == nullptr) {
		Shcore = LoadLibraryW(L"Shcore.dll");
		getDPIForMonitor = Shcore ? (GetDPIForMonitor_t)GetProcAddress(Shcore, "GetDpiForMonitor") : nullptr;

		if ((Shcore == nullptr) || (getDPIForMonitor == nullptr)) {
			if (Shcore)
				FreeLibrary(Shcore);
			Shcore = (HMODULE)INVALID_HANDLE_VALUE;
		}
	}

	UINT x = 0, y = 0;
	HRESULT hr = E_FAIL;
	if (hmon && (Shcore != (HMODULE)INVALID_HANDLE_VALUE)) {
		hr = getDPIForMonitor(hmon, dpiType /*MDT_Effective_DPI*/, &x, &y);
		if (SUCCEEDED(hr) && (x > 0) && (y > 0)) {
			dpiX = (int)x;
			dpiY = (int)y;
		}
	} else {
		static int overallX = 0, overallY = 0;
		if (overallX <= 0 || overallY <= 0) {
			HDC hdc = GetDC(nullptr);
			if (hdc) {
				overallX = GetDeviceCaps(hdc, LOGPIXELSX);
				overallY = GetDeviceCaps(hdc, LOGPIXELSY);
				ReleaseDC(nullptr, hdc);
			}
		}
		if (overallX > 0 && overallY > 0) {
			dpiX = overallX;
			dpiY = overallY;
		}
	}

	return (dpiX + dpiY) / 2;
}

static BOOL CALLBACK _MonitorEnumProcDpi(HMONITOR hMonitor, HDC hdcMonitor, LPRECT lprcMonitor, LPARAM dwData) {
	EnumDpiData *data = (EnumDpiData *)dwData;
	if (data->count == data->screen) {
		data->dpi = QueryDpiForMonitor(hMonitor);
	}

	data->count++;
	return TRUE;
}

int DisplayServerWindows::screen_get_dpi(int p_screen) const {
	_THREAD_SAFE_METHOD_

	EnumDpiData data = { 0, p_screen == SCREEN_OF_MAIN_WINDOW ? window_get_current_screen() : p_screen, 72 };
	EnumDisplayMonitors(nullptr, nullptr, _MonitorEnumProcDpi, (LPARAM)&data);
	return data.dpi;
}

bool DisplayServerWindows::screen_is_touchscreen(int p_screen) const {
#ifndef _MSC_VER
#warning touchscreen not working
#endif
	return false;
}

void DisplayServerWindows::screen_set_orientation(ScreenOrientation p_orientation, int p_screen) {
}

DisplayServer::ScreenOrientation DisplayServerWindows::screen_get_orientation(int p_screen) const {
	return SCREEN_LANDSCAPE;
}

void DisplayServerWindows::screen_set_keep_on(bool p_enable) {
}

bool DisplayServerWindows::screen_is_kept_on() const {
	return false;
}

Vector<DisplayServer::WindowID> DisplayServerWindows::get_window_list() const {
	_THREAD_SAFE_METHOD_

	Vector<DisplayServer::WindowID> ret;
	for (const KeyValue<WindowID, WindowData> &E : windows) {
		ret.push_back(E.key);
	}
	return ret;
}

DisplayServer::WindowID DisplayServerWindows::get_window_at_screen_position(const Point2i &p_position) const {
	POINT p;
	p.x = p_position.x;
	p.y = p_position.y;
	HWND hwnd = WindowFromPoint(p);
	for (const KeyValue<WindowID, WindowData> &E : windows) {
		if (E.value.hWnd == hwnd) {
			return E.key;
		}
	}

	return INVALID_WINDOW_ID;
}

DisplayServer::WindowID DisplayServerWindows::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	_THREAD_SAFE_METHOD_

	WindowID window_id = _create_window(p_mode, p_vsync_mode, p_flags, p_rect);
	ERR_FAIL_COND_V_MSG(window_id == INVALID_WINDOW_ID, INVALID_WINDOW_ID, "Failed to create sub window.");

	WindowData &wd = windows[window_id];

	if (p_flags & WINDOW_FLAG_RESIZE_DISABLED_BIT) {
		wd.resizable = false;
	}
	if (p_flags & WINDOW_FLAG_BORDERLESS_BIT) {
		wd.borderless = true;
	}
	if (p_flags & WINDOW_FLAG_ALWAYS_ON_TOP_BIT && p_mode != WINDOW_MODE_FULLSCREEN) {
		wd.always_on_top = true;
	}
	if (p_flags & WINDOW_FLAG_NO_FOCUS_BIT) {
		wd.no_focus = true;
	}

	return window_id;
}

void DisplayServerWindows::show_window(WindowID p_id) {
	ERR_FAIL_COND(!windows.has(p_id));

	WindowData &wd = windows[p_id];

	if (p_id != MAIN_WINDOW_ID) {
		_update_window_style(p_id);
	}

	ShowWindow(wd.hWnd, wd.no_focus ? SW_SHOWNOACTIVATE : SW_SHOW); // Show the window.
	if (!wd.no_focus) {
		SetForegroundWindow(wd.hWnd); // Slightly higher priority.
		SetFocus(wd.hWnd); // Set keyboard focus.
	}
}

void DisplayServerWindows::delete_sub_window(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	ERR_FAIL_COND_MSG(p_window == MAIN_WINDOW_ID, "Main window cannot be deleted.");

	WindowData &wd = windows[p_window];

	while (wd.transient_children.size()) {
		window_set_transient(wd.transient_children.front()->get(), INVALID_WINDOW_ID);
	}

	if (wd.transient_parent != INVALID_WINDOW_ID) {
		window_set_transient(p_window, INVALID_WINDOW_ID);
	}

#ifdef VULKAN_ENABLED
	if (context_vulkan) {
		context_vulkan->window_destroy(p_window);
	}
#endif
#ifdef GLES3_ENABLED
	if (gl_manager) {
		gl_manager->window_destroy(p_window);
	}
#endif

	if ((tablet_get_current_driver() == "wintab") && wintab_available && windows[p_window].wtctx) {
		wintab_WTClose(windows[p_window].wtctx);
		windows[p_window].wtctx = 0;
	}
	DestroyWindow(windows[p_window].hWnd);
	windows.erase(p_window);
}

void DisplayServerWindows::gl_window_make_current(DisplayServer::WindowID p_window_id) {
#if defined(GLES3_ENABLED)
	gl_manager->window_make_current(p_window_id);
#endif
}

void DisplayServerWindows::window_attach_instance_id(ObjectID p_instance, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].instance_id = p_instance;
}

ObjectID DisplayServerWindows::window_get_attached_instance_id(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), ObjectID());
	return windows[p_window].instance_id;
}

void DisplayServerWindows::window_set_rect_changed_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].rect_changed_callback = p_callable;
}

void DisplayServerWindows::window_set_window_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].event_callback = p_callable;
}

void DisplayServerWindows::window_set_input_event_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].input_event_callback = p_callable;
}

void DisplayServerWindows::window_set_input_text_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].input_text_callback = p_callable;
}

void DisplayServerWindows::window_set_drop_files_callback(const Callable &p_callable, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].drop_files_callback = p_callable;
}

void DisplayServerWindows::window_set_title(const String &p_title, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	SetWindowTextW(windows[p_window].hWnd, (LPCWSTR)(p_title.utf16().get_data()));
}

void DisplayServerWindows::window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	windows[p_window].mpath = p_region;
	_update_window_mouse_passthrough(p_window);
}

void DisplayServerWindows::_update_window_mouse_passthrough(WindowID p_window) {
	ERR_FAIL_COND(!windows.has(p_window));

	if (windows[p_window].mpath.size() == 0) {
		SetWindowRgn(windows[p_window].hWnd, nullptr, TRUE);
	} else {
		POINT *points = (POINT *)memalloc(sizeof(POINT) * windows[p_window].mpath.size());
		for (int i = 0; i < windows[p_window].mpath.size(); i++) {
			if (windows[p_window].borderless) {
				points[i].x = windows[p_window].mpath[i].x;
				points[i].y = windows[p_window].mpath[i].y;
			} else {
				points[i].x = windows[p_window].mpath[i].x + GetSystemMetrics(SM_CXSIZEFRAME);
				points[i].y = windows[p_window].mpath[i].y + GetSystemMetrics(SM_CYSIZEFRAME) + GetSystemMetrics(SM_CYCAPTION);
			}
		}

		HRGN region = CreatePolygonRgn(points, windows[p_window].mpath.size(), ALTERNATE);
		SetWindowRgn(windows[p_window].hWnd, region, TRUE);
		DeleteObject(region);
		memfree(points);
	}
}

int DisplayServerWindows::window_get_current_screen(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), -1);

	EnumScreenData data = { 0, 0, MonitorFromWindow(windows[p_window].hWnd, MONITOR_DEFAULTTONEAREST) };
	EnumDisplayMonitors(nullptr, nullptr, _MonitorEnumProcScreen, (LPARAM)&data);
	return data.screen;
}

void DisplayServerWindows::window_set_current_screen(int p_screen, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	ERR_FAIL_INDEX(p_screen, get_screen_count());

	const WindowData &wd = windows[p_window];
	if (wd.fullscreen) {
		int cs = window_get_current_screen(p_window);
		if (cs == p_screen) {
			return;
		}
		Point2 pos = screen_get_position(p_screen);
		Size2 size = screen_get_size(p_screen);

		MoveWindow(wd.hWnd, pos.x, pos.y, size.width, size.height, TRUE);
	} else {
		Vector2 ofs = window_get_position(p_window) - screen_get_position(window_get_current_screen(p_window));
		window_set_position(ofs + screen_get_position(p_screen), p_window);
	}
}

Point2i DisplayServerWindows::window_get_position(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Point2i());
	const WindowData &wd = windows[p_window];

	if (wd.minimized) {
		return wd.last_pos;
	}

	POINT point;
	point.x = 0;
	point.y = 0;

	ClientToScreen(wd.hWnd, &point);

	return Point2i(point.x, point.y);
}

void DisplayServerWindows::_update_real_mouse_position(WindowID p_window) {
	ERR_FAIL_COND(!windows.has(p_window));

	POINT mouse_pos;
	if (GetCursorPos(&mouse_pos) && ScreenToClient(windows[p_window].hWnd, &mouse_pos)) {
		if (mouse_pos.x > 0 && mouse_pos.y > 0 && mouse_pos.x <= windows[p_window].width && mouse_pos.y <= windows[p_window].height) {
			old_x = mouse_pos.x;
			old_y = mouse_pos.y;
			old_invalid = false;
			Input::get_singleton()->set_mouse_position(Point2i(mouse_pos.x, mouse_pos.y));
		}
	}
}

void DisplayServerWindows::window_set_position(const Point2i &p_position, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (wd.fullscreen) {
		return;
	}

	RECT rc;
	rc.left = p_position.x;
	rc.right = p_position.x + wd.width;
	rc.bottom = p_position.y + wd.height;
	rc.top = p_position.y;

	const DWORD style = GetWindowLongPtr(wd.hWnd, GWL_STYLE);
	const DWORD exStyle = GetWindowLongPtr(wd.hWnd, GWL_EXSTYLE);

	AdjustWindowRectEx(&rc, style, false, exStyle);
	MoveWindow(wd.hWnd, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top, TRUE);

	// Don't let the mouse leave the window when moved.
	if (mouse_mode == MOUSE_MODE_CONFINED || mouse_mode == MOUSE_MODE_CONFINED_HIDDEN) {
		RECT rect;
		GetClientRect(wd.hWnd, &rect);
		ClientToScreen(wd.hWnd, (POINT *)&rect.left);
		ClientToScreen(wd.hWnd, (POINT *)&rect.right);
		ClipCursor(&rect);
	}

	wd.last_pos = p_position;
	_update_real_mouse_position(p_window);
}

void DisplayServerWindows::window_set_transient(WindowID p_window, WindowID p_parent) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(p_window == p_parent);
	ERR_FAIL_COND(!windows.has(p_window));

	WindowData &wd_window = windows[p_window];

	ERR_FAIL_COND(wd_window.transient_parent == p_parent);
	ERR_FAIL_COND_MSG(wd_window.always_on_top, "Windows with the 'on top' can't become transient.");

	if (p_parent == INVALID_WINDOW_ID) {
		// Remove transient.

		ERR_FAIL_COND(wd_window.transient_parent == INVALID_WINDOW_ID);
		ERR_FAIL_COND(!windows.has(wd_window.transient_parent));

		WindowData &wd_parent = windows[wd_window.transient_parent];

		wd_window.transient_parent = INVALID_WINDOW_ID;
		wd_parent.transient_children.erase(p_window);

		SetWindowLongPtr(wd_window.hWnd, GWLP_HWNDPARENT, (LONG_PTR) nullptr);
	} else {
		ERR_FAIL_COND(!windows.has(p_parent));
		ERR_FAIL_COND_MSG(wd_window.transient_parent != INVALID_WINDOW_ID, "Window already has a transient parent");
		WindowData &wd_parent = windows[p_parent];

		wd_window.transient_parent = p_parent;
		wd_parent.transient_children.insert(p_window);

		SetWindowLongPtr(wd_window.hWnd, GWLP_HWNDPARENT, (LONG_PTR)wd_parent.hWnd);
	}
}

void DisplayServerWindows::window_set_max_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if ((p_size != Size2()) && ((p_size.x < wd.min_size.x) || (p_size.y < wd.min_size.y))) {
		ERR_PRINT("Maximum window size can't be smaller than minimum window size!");
		return;
	}
	wd.max_size = p_size;
}

Size2i DisplayServerWindows::window_get_max_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	return wd.max_size;
}

void DisplayServerWindows::window_set_min_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if ((p_size != Size2()) && (wd.max_size != Size2()) && ((p_size.x > wd.max_size.x) || (p_size.y > wd.max_size.y))) {
		ERR_PRINT("Minimum window size can't be larger than maximum window size!");
		return;
	}
	wd.min_size = p_size;
}

Size2i DisplayServerWindows::window_get_min_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];
	return wd.min_size;
}

void DisplayServerWindows::window_set_size(const Size2i p_size, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	int w = p_size.width;
	int h = p_size.height;

	wd.width = w;
	wd.height = h;

#if defined(VULKAN_ENABLED)
	if (context_vulkan) {
		context_vulkan->window_resize(p_window, w, h);
	}
#endif
#if defined(GLES3_ENABLED)
	if (gl_manager) {
		gl_manager->window_resize(p_window, w, h);
	}
#endif

	if (wd.fullscreen) {
		return;
	}

	RECT rect;
	GetWindowRect(wd.hWnd, &rect);

	if (!wd.borderless) {
		RECT crect;
		GetClientRect(wd.hWnd, &crect);

		w += (rect.right - rect.left) - (crect.right - crect.left);
		h += (rect.bottom - rect.top) - (crect.bottom - crect.top);
	}

	MoveWindow(wd.hWnd, rect.left, rect.top, w, h, TRUE);

	// Don't let the mouse leave the window when resizing to a smaller resolution.
	if (mouse_mode == MOUSE_MODE_CONFINED || mouse_mode == MOUSE_MODE_CONFINED_HIDDEN) {
		RECT crect;
		GetClientRect(wd.hWnd, &crect);
		ClientToScreen(wd.hWnd, (POINT *)&crect.left);
		ClientToScreen(wd.hWnd, (POINT *)&crect.right);
		ClipCursor(&crect);
	}
}

Size2i DisplayServerWindows::window_get_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];

	// GetClientRect() returns a zero rect for a minimized window, so we need to get the size in another way.
	if (wd.minimized) {
		return Size2(wd.width, wd.height);
	}

	RECT r;
	if (GetClientRect(wd.hWnd, &r)) { // Retrieves area inside of window border, including decoration.
		return Size2(r.right - r.left, r.bottom - r.top);
	}
	return Size2();
}

Size2i DisplayServerWindows::window_get_real_size(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), Size2i());
	const WindowData &wd = windows[p_window];

	RECT r;
	if (GetWindowRect(wd.hWnd, &r)) { // Retrieves area inside of window border, including decoration.
		return Size2(r.right - r.left, r.bottom - r.top);
	}
	return Size2();
}

void DisplayServerWindows::_get_window_style(bool p_main_window, bool p_fullscreen, bool p_borderless, bool p_resizable, bool p_maximized, bool p_no_activate_focus, DWORD &r_style, DWORD &r_style_ex) {
	// Windows docs for window styles:
	// https://docs.microsoft.com/en-us/windows/win32/winmsg/window-styles
	// https://docs.microsoft.com/en-us/windows/win32/winmsg/extended-window-styles

	r_style = 0;
	r_style_ex = WS_EX_WINDOWEDGE;
	if (p_main_window) {
		r_style_ex |= WS_EX_APPWINDOW;
	}

	if (p_fullscreen || p_borderless) {
		r_style |= WS_POPUP; // p_borderless was WS_EX_TOOLWINDOW in the past.
	} else {
		if (p_resizable) {
			if (p_maximized) {
				r_style = WS_OVERLAPPEDWINDOW | WS_MAXIMIZE;
			} else {
				r_style = WS_OVERLAPPEDWINDOW;
			}
		} else {
			r_style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
		}
	}
	if (!p_borderless) {
		r_style |= WS_VISIBLE;
	}

	if (p_no_activate_focus) {
		r_style_ex |= WS_EX_TOPMOST | WS_EX_NOACTIVATE;
	}
	r_style |= WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
}

void DisplayServerWindows::_update_window_style(WindowID p_window, bool p_repaint) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	DWORD style = 0;
	DWORD style_ex = 0;

	_get_window_style(p_window == MAIN_WINDOW_ID, wd.fullscreen, wd.borderless, wd.resizable, wd.maximized, wd.no_focus, style, style_ex);

	SetWindowLongPtr(wd.hWnd, GWL_STYLE, style);
	SetWindowLongPtr(wd.hWnd, GWL_EXSTYLE, style_ex);

	SetWindowPos(wd.hWnd, wd.always_on_top ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | (wd.no_focus ? SWP_NOACTIVATE : 0));

	if (p_repaint) {
		RECT rect;
		GetWindowRect(wd.hWnd, &rect);
		MoveWindow(wd.hWnd, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, TRUE);
	}
}

void DisplayServerWindows::window_set_mode(WindowMode p_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (wd.fullscreen && p_mode != WINDOW_MODE_FULLSCREEN) {
		RECT rect;

		wd.fullscreen = false;
		wd.maximized = wd.was_maximized;

		if (wd.pre_fs_valid) {
			rect = wd.pre_fs_rect;
		} else {
			rect.left = 0;
			rect.right = wd.width;
			rect.top = 0;
			rect.bottom = wd.height;
			wd.pre_fs_valid = true;
		}

		_update_window_style(p_window, false);

		MoveWindow(wd.hWnd, rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top, TRUE);

		if (restore_mouse_trails > 1) {
			SystemParametersInfoA(SPI_SETMOUSETRAILS, restore_mouse_trails, 0, 0);
			restore_mouse_trails = 0;
		}
	} else if (p_mode == WINDOW_MODE_WINDOWED) {
		ShowWindow(wd.hWnd, SW_RESTORE);
		wd.maximized = false;
		wd.minimized = false;
	}

	if (p_mode == WINDOW_MODE_MAXIMIZED) {
		ShowWindow(wd.hWnd, SW_MAXIMIZE);
		wd.maximized = true;
		wd.minimized = false;
	}

	if (p_mode == WINDOW_MODE_MINIMIZED) {
		ShowWindow(wd.hWnd, SW_MINIMIZE);
		wd.maximized = false;
		wd.minimized = true;
	}

	if (p_mode == WINDOW_MODE_FULLSCREEN && !wd.fullscreen) {
		if (wd.minimized) {
			ShowWindow(wd.hWnd, SW_RESTORE);
		}
		wd.was_maximized = wd.maximized;

		if (wd.pre_fs_valid) {
			GetWindowRect(wd.hWnd, &wd.pre_fs_rect);
		}

		int cs = window_get_current_screen(p_window);
		Point2 pos = screen_get_position(cs);
		Size2 size = screen_get_size(cs);

		wd.fullscreen = true;
		wd.maximized = false;
		wd.minimized = false;

		_update_window_style(false);

		MoveWindow(wd.hWnd, pos.x, pos.y, size.width, size.height, TRUE);

		// If the user has mouse trails enabled in windows, then sometimes the cursor disappears in fullscreen mode.
		// Save number of trails so we can restore when exiting, then turn off mouse trails
		SystemParametersInfoA(SPI_GETMOUSETRAILS, 0, &restore_mouse_trails, 0);
		if (restore_mouse_trails > 1) {
			SystemParametersInfoA(SPI_SETMOUSETRAILS, 0, 0, 0);
		}
	}
}

DisplayServer::WindowMode DisplayServerWindows::window_get_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), WINDOW_MODE_WINDOWED);
	const WindowData &wd = windows[p_window];

	if (wd.fullscreen) {
		return WINDOW_MODE_FULLSCREEN;
	} else if (wd.minimized) {
		return WINDOW_MODE_MINIMIZED;
	} else if (wd.maximized) {
		return WINDOW_MODE_MAXIMIZED;
	} else {
		return WINDOW_MODE_WINDOWED;
	}
}

bool DisplayServerWindows::window_is_maximize_allowed(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);

	// FIXME: Implement this, or confirm that it should always be true.

	return true;
}

void DisplayServerWindows::window_set_flag(WindowFlags p_flag, bool p_enabled, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];
	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			wd.resizable = !p_enabled;
			_update_window_style(p_window);
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			wd.borderless = p_enabled;
			_update_window_style(p_window);
			_update_window_mouse_passthrough(p_window);
		} break;
		case WINDOW_FLAG_ALWAYS_ON_TOP: {
			ERR_FAIL_COND_MSG(wd.transient_parent != INVALID_WINDOW_ID && p_enabled, "Transient windows can't become on top");
			wd.always_on_top = p_enabled;
			_update_window_style(p_window);
		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			// FIXME: Implement.
		} break;
		case WINDOW_FLAG_NO_FOCUS: {
			wd.no_focus = p_enabled;
			_update_window_style(p_window);
		} break;
		case WINDOW_FLAG_MAX:
			break;
	}
}

bool DisplayServerWindows::window_get_flag(WindowFlags p_flag, WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);
	const WindowData &wd = windows[p_window];
	switch (p_flag) {
		case WINDOW_FLAG_RESIZE_DISABLED: {
			return !wd.resizable;
		} break;
		case WINDOW_FLAG_BORDERLESS: {
			return wd.borderless;
		} break;
		case WINDOW_FLAG_ALWAYS_ON_TOP: {
			return wd.always_on_top;
		} break;
		case WINDOW_FLAG_TRANSPARENT: {
			// FIXME: Implement.
		} break;
		case WINDOW_FLAG_NO_FOCUS: {
			return wd.no_focus;
		} break;
		case WINDOW_FLAG_MAX:
			break;
	}

	return false;
}

void DisplayServerWindows::window_request_attention(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	FLASHWINFO info;
	info.cbSize = sizeof(FLASHWINFO);
	info.hwnd = wd.hWnd;
	info.dwFlags = FLASHW_TRAY;
	info.dwTimeout = 0;
	info.uCount = 2;
	FlashWindowEx(&info);
}

void DisplayServerWindows::window_move_to_foreground(WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	SetForegroundWindow(wd.hWnd);
}

bool DisplayServerWindows::window_can_draw(WindowID p_window) const {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND_V(!windows.has(p_window), false);
	const WindowData &wd = windows[p_window];
	return !wd.minimized;
}

bool DisplayServerWindows::can_any_window_draw() const {
	_THREAD_SAFE_METHOD_

	for (const KeyValue<WindowID, WindowData> &E : windows) {
		if (!E.value.minimized) {
			return true;
		}
	}

	return false;
}

void DisplayServerWindows::window_set_ime_active(const bool p_active, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	if (p_active) {
		ImmAssociateContext(wd.hWnd, wd.im_himc);

		window_set_ime_position(wd.im_position, p_window);
	} else {
		ImmAssociateContext(wd.hWnd, (HIMC)0);
	}
}

void DisplayServerWindows::window_set_ime_position(const Point2i &p_pos, WindowID p_window) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!windows.has(p_window));
	WindowData &wd = windows[p_window];

	wd.im_position = p_pos;

	HIMC himc = ImmGetContext(wd.hWnd);
	if (himc == (HIMC)0)
		return;

	COMPOSITIONFORM cps;
	cps.dwStyle = CFS_FORCE_POSITION;
	cps.ptCurrentPos.x = wd.im_position.x;
	cps.ptCurrentPos.y = wd.im_position.y;
	ImmSetCompositionWindow(himc, &cps);
	ImmReleaseContext(wd.hWnd, himc);
}

void DisplayServerWindows::cursor_set_shape(CursorShape p_shape) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX(p_shape, CURSOR_MAX);

	if (cursor_shape == p_shape) {
		return;
	}

	if (mouse_mode != MOUSE_MODE_VISIBLE && mouse_mode != MOUSE_MODE_CONFINED) {
		cursor_shape = p_shape;
		return;
	}

	static const LPCTSTR win_cursors[CURSOR_MAX] = {
		IDC_ARROW,
		IDC_IBEAM,
		IDC_HAND, // Finger.
		IDC_CROSS,
		IDC_WAIT,
		IDC_APPSTARTING,
		IDC_SIZEALL,
		IDC_ARROW,
		IDC_NO,
		IDC_SIZENS,
		IDC_SIZEWE,
		IDC_SIZENESW,
		IDC_SIZENWSE,
		IDC_SIZEALL,
		IDC_SIZENS,
		IDC_SIZEWE,
		IDC_HELP
	};

	if (cursors[p_shape] != nullptr) {
		SetCursor(cursors[p_shape]);
	} else {
		SetCursor(LoadCursor(hInstance, win_cursors[p_shape]));
	}

	cursor_shape = p_shape;
}

DisplayServer::CursorShape DisplayServerWindows::cursor_get_shape() const {
	return cursor_shape;
}

void DisplayServerWindows::GetMaskBitmaps(HBITMAP hSourceBitmap, COLORREF clrTransparent, OUT HBITMAP &hAndMaskBitmap, OUT HBITMAP &hXorMaskBitmap) {
	// Get the system display DC.
	HDC hDC = GetDC(nullptr);

	// Create helper DC.
	HDC hMainDC = CreateCompatibleDC(hDC);
	HDC hAndMaskDC = CreateCompatibleDC(hDC);
	HDC hXorMaskDC = CreateCompatibleDC(hDC);

	// Get the dimensions of the source bitmap.
	BITMAP bm;
	GetObject(hSourceBitmap, sizeof(BITMAP), &bm);

	// Create the mask bitmaps.
	hAndMaskBitmap = CreateCompatibleBitmap(hDC, bm.bmWidth, bm.bmHeight); // Color.
	hXorMaskBitmap = CreateCompatibleBitmap(hDC, bm.bmWidth, bm.bmHeight); // Color.

	// Release the system display DC.
	ReleaseDC(nullptr, hDC);

	// Select the bitmaps to helper DC.
	HBITMAP hOldMainBitmap = (HBITMAP)SelectObject(hMainDC, hSourceBitmap);
	HBITMAP hOldAndMaskBitmap = (HBITMAP)SelectObject(hAndMaskDC, hAndMaskBitmap);
	HBITMAP hOldXorMaskBitmap = (HBITMAP)SelectObject(hXorMaskDC, hXorMaskBitmap);

	// Assign the monochrome AND mask bitmap pixels so that the pixels of the source bitmap
	// with 'clrTransparent' will be white pixels of the monochrome bitmap.
	SetBkColor(hMainDC, clrTransparent);
	BitBlt(hAndMaskDC, 0, 0, bm.bmWidth, bm.bmHeight, hMainDC, 0, 0, SRCCOPY);

	// Assign the color XOR mask bitmap pixels so that the pixels of the source bitmap
	// with 'clrTransparent' will be black and rest the pixels same as corresponding
	// pixels of the source bitmap.
	SetBkColor(hXorMaskDC, RGB(0, 0, 0));
	SetTextColor(hXorMaskDC, RGB(255, 255, 255));
	BitBlt(hXorMaskDC, 0, 0, bm.bmWidth, bm.bmHeight, hAndMaskDC, 0, 0, SRCCOPY);
	BitBlt(hXorMaskDC, 0, 0, bm.bmWidth, bm.bmHeight, hMainDC, 0, 0, SRCAND);

	// Deselect bitmaps from the helper DC.
	SelectObject(hMainDC, hOldMainBitmap);
	SelectObject(hAndMaskDC, hOldAndMaskBitmap);
	SelectObject(hXorMaskDC, hOldXorMaskBitmap);

	// Delete the helper DC.
	DeleteDC(hXorMaskDC);
	DeleteDC(hAndMaskDC);
	DeleteDC(hMainDC);
}

void DisplayServerWindows::cursor_set_custom_image(const RES &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	_THREAD_SAFE_METHOD_

	if (p_cursor.is_valid()) {
		Map<CursorShape, Vector<Variant>>::Element *cursor_c = cursors_cache.find(p_shape);

		if (cursor_c) {
			if (cursor_c->get()[0] == p_cursor && cursor_c->get()[1] == p_hotspot) {
				cursor_set_shape(p_shape);
				return;
			}

			cursors_cache.erase(p_shape);
		}

		Ref<Texture2D> texture = p_cursor;
		Ref<AtlasTexture> atlas_texture = p_cursor;
		Ref<Image> image;
		Size2 texture_size;
		Rect2 atlas_rect;

		if (texture.is_valid()) {
			image = texture->get_image();
		}

		if (!image.is_valid() && atlas_texture.is_valid()) {
			texture = atlas_texture->get_atlas();

			atlas_rect.size.width = texture->get_width();
			atlas_rect.size.height = texture->get_height();
			atlas_rect.position.x = atlas_texture->get_region().position.x;
			atlas_rect.position.y = atlas_texture->get_region().position.y;

			texture_size.width = atlas_texture->get_region().size.x;
			texture_size.height = atlas_texture->get_region().size.y;
		} else if (image.is_valid()) {
			texture_size.width = texture->get_width();
			texture_size.height = texture->get_height();
		}

		ERR_FAIL_COND(!texture.is_valid());
		ERR_FAIL_COND(p_hotspot.x < 0 || p_hotspot.y < 0);
		ERR_FAIL_COND(texture_size.width > 256 || texture_size.height > 256);
		ERR_FAIL_COND(p_hotspot.x > texture_size.width || p_hotspot.y > texture_size.height);

		image = texture->get_image();

		ERR_FAIL_COND(!image.is_valid());

		UINT image_size = texture_size.width * texture_size.height;

		// Create the BITMAP with alpha channel.
		COLORREF *buffer = (COLORREF *)memalloc(sizeof(COLORREF) * image_size);

		for (UINT index = 0; index < image_size; index++) {
			int row_index = floor(index / texture_size.width) + atlas_rect.position.y;
			int column_index = (index % int(texture_size.width)) + atlas_rect.position.x;

			if (atlas_texture.is_valid()) {
				column_index = MIN(column_index, atlas_rect.size.width - 1);
				row_index = MIN(row_index, atlas_rect.size.height - 1);
			}

			*(buffer + index) = image->get_pixel(column_index, row_index).to_argb32();
		}

		// Using 4 channels, so 4 * 8 bits.
		HBITMAP bitmap = CreateBitmap(texture_size.width, texture_size.height, 1, 4 * 8, buffer);
		COLORREF clrTransparent = -1;

		// Create the AND and XOR masks for the bitmap.
		HBITMAP hAndMask = nullptr;
		HBITMAP hXorMask = nullptr;

		GetMaskBitmaps(bitmap, clrTransparent, hAndMask, hXorMask);

		if (nullptr == hAndMask || nullptr == hXorMask) {
			memfree(buffer);
			DeleteObject(bitmap);
			return;
		}

		// Finally, create the icon.
		ICONINFO iconinfo;
		iconinfo.fIcon = FALSE;
		iconinfo.xHotspot = p_hotspot.x;
		iconinfo.yHotspot = p_hotspot.y;
		iconinfo.hbmMask = hAndMask;
		iconinfo.hbmColor = hXorMask;

		if (cursors[p_shape]) {
			DestroyIcon(cursors[p_shape]);
		}

		cursors[p_shape] = CreateIconIndirect(&iconinfo);

		Vector<Variant> params;
		params.push_back(p_cursor);
		params.push_back(p_hotspot);
		cursors_cache.insert(p_shape, params);

		if (p_shape == cursor_shape) {
			if (mouse_mode == MOUSE_MODE_VISIBLE || mouse_mode == MOUSE_MODE_CONFINED) {
				SetCursor(cursors[p_shape]);
			}
		}

		if (hAndMask != nullptr) {
			DeleteObject(hAndMask);
		}

		if (hXorMask != nullptr) {
			DeleteObject(hXorMask);
		}

		memfree(buffer);
		DeleteObject(bitmap);
	} else {
		// Reset to default system cursor.
		if (cursors[p_shape]) {
			DestroyIcon(cursors[p_shape]);
			cursors[p_shape] = nullptr;
		}

		CursorShape c = cursor_shape;
		cursor_shape = CURSOR_MAX;
		cursor_set_shape(c);

		cursors_cache.erase(p_shape);
	}
}

bool DisplayServerWindows::get_swap_cancel_ok() {
	return true;
}

void DisplayServerWindows::enable_for_stealing_focus(OS::ProcessID pid) {
	_THREAD_SAFE_METHOD_

	AllowSetForegroundWindow(pid);
}

int DisplayServerWindows::keyboard_get_layout_count() const {
	return GetKeyboardLayoutList(0, nullptr);
}

int DisplayServerWindows::keyboard_get_current_layout() const {
	HKL cur_layout = GetKeyboardLayout(0);

	int layout_count = GetKeyboardLayoutList(0, nullptr);
	HKL *layouts = (HKL *)memalloc(layout_count * sizeof(HKL));
	GetKeyboardLayoutList(layout_count, layouts);

	for (int i = 0; i < layout_count; i++) {
		if (cur_layout == layouts[i]) {
			memfree(layouts);
			return i;
		}
	}
	memfree(layouts);
	return -1;
}

void DisplayServerWindows::keyboard_set_current_layout(int p_index) {
	int layout_count = GetKeyboardLayoutList(0, nullptr);

	ERR_FAIL_INDEX(p_index, layout_count);

	HKL *layouts = (HKL *)memalloc(layout_count * sizeof(HKL));
	GetKeyboardLayoutList(layout_count, layouts);
	ActivateKeyboardLayout(layouts[p_index], KLF_SETFORPROCESS);
	memfree(layouts);
}

String DisplayServerWindows::keyboard_get_layout_language(int p_index) const {
	int layout_count = GetKeyboardLayoutList(0, nullptr);

	ERR_FAIL_INDEX_V(p_index, layout_count, "");

	HKL *layouts = (HKL *)memalloc(layout_count * sizeof(HKL));
	GetKeyboardLayoutList(layout_count, layouts);

	WCHAR buf[LOCALE_NAME_MAX_LENGTH];
	memset(buf, 0, LOCALE_NAME_MAX_LENGTH * sizeof(WCHAR));
	LCIDToLocaleName(MAKELCID(LOWORD(layouts[p_index]), SORT_DEFAULT), buf, LOCALE_NAME_MAX_LENGTH, 0);

	memfree(layouts);

	return String::utf16((const char16_t *)buf).substr(0, 2);
}

Key DisplayServerWindows::keyboard_get_keycode_from_physical(Key p_keycode) const {
	Key modifiers = p_keycode & KeyModifierMask::MODIFIER_MASK;
	Key keycode_no_mod = (Key)(p_keycode & KeyModifierMask::CODE_MASK);

	if (keycode_no_mod == Key::PRINT ||
			keycode_no_mod == Key::KP_ADD ||
			keycode_no_mod == Key::KP_5 ||
			(keycode_no_mod >= Key::KEY_0 && keycode_no_mod <= Key::KEY_9)) {
		return p_keycode;
	}

	unsigned int scancode = KeyMappingWindows::get_scancode(keycode_no_mod);
	if (scancode == 0) {
		return p_keycode;
	}

	HKL current_layout = GetKeyboardLayout(0);
	UINT vk = MapVirtualKeyEx(scancode, MAPVK_VSC_TO_VK, current_layout);
	if (vk == 0) {
		return p_keycode;
	}

	UINT char_code = MapVirtualKeyEx(vk, MAPVK_VK_TO_CHAR, current_layout) & 0x7FFF;
	// Unlike a similar Linux/BSD check which matches full Latin-1 range,
	// we limit these to ASCII to fix some layouts, including Arabic ones
	if (char_code >= 32 && char_code <= 127) {
		// Godot uses 'braces' instead of 'brackets'
		if (char_code == (unsigned int)Key::BRACKETLEFT || char_code == (unsigned int)Key::BRACKETRIGHT) {
			char_code += 32;
		}
		return (Key)(char_code | (unsigned int)modifiers);
	}

	return (Key)(KeyMappingWindows::get_keysym(vk) | modifiers);
}

String _get_full_layout_name_from_registry(HKL p_layout) {
	String id = "SYSTEM\\CurrentControlSet\\Control\\Keyboard Layouts\\" + String::num_int64((int64_t)p_layout, 16, false).lpad(8, "0");
	String ret;

	HKEY hkey;
	WCHAR layout_text[1024];
	memset(layout_text, 0, 1024 * sizeof(WCHAR));

	if (RegOpenKeyExW(HKEY_LOCAL_MACHINE, (LPCWSTR)(id.utf16().get_data()), 0, KEY_QUERY_VALUE, &hkey) != ERROR_SUCCESS) {
		return ret;
	}

	DWORD buffer = 1024;
	DWORD vtype = REG_SZ;
	if (RegQueryValueExW(hkey, L"Layout Text", nullptr, &vtype, (LPBYTE)layout_text, &buffer) == ERROR_SUCCESS) {
		ret = String::utf16((const char16_t *)layout_text);
	}
	RegCloseKey(hkey);
	return ret;
}

String DisplayServerWindows::keyboard_get_layout_name(int p_index) const {
	int layout_count = GetKeyboardLayoutList(0, nullptr);

	ERR_FAIL_INDEX_V(p_index, layout_count, "");

	HKL *layouts = (HKL *)memalloc(layout_count * sizeof(HKL));
	GetKeyboardLayoutList(layout_count, layouts);

	String ret = _get_full_layout_name_from_registry(layouts[p_index]); // Try reading full name from Windows registry, fallback to locale name if failed (e.g. on Wine).
	if (ret.is_empty()) {
		WCHAR buf[LOCALE_NAME_MAX_LENGTH];
		memset(buf, 0, LOCALE_NAME_MAX_LENGTH * sizeof(WCHAR));
		LCIDToLocaleName(MAKELCID(LOWORD(layouts[p_index]), SORT_DEFAULT), buf, LOCALE_NAME_MAX_LENGTH, 0);

		WCHAR name[1024];
		memset(name, 0, 1024 * sizeof(WCHAR));
		GetLocaleInfoEx(buf, LOCALE_SLOCALIZEDDISPLAYNAME, (LPWSTR)&name, 1024);

		ret = String::utf16((const char16_t *)name);
	}
	memfree(layouts);

	return ret;
}

void DisplayServerWindows::process_events() {
	_THREAD_SAFE_METHOD_

	MSG msg;

	if (!drop_events) {
		joypad->process_joypads();
	}

	while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
		TranslateMessage(&msg);
		DispatchMessageW(&msg);
	}

	if (!drop_events) {
		_process_key_events();
		Input::get_singleton()->flush_buffered_events();
	}
}

void DisplayServerWindows::force_process_and_drop_events() {
	_THREAD_SAFE_METHOD_

	drop_events = true;
	process_events();
	drop_events = false;
}

void DisplayServerWindows::release_rendering_thread() {
}

void DisplayServerWindows::make_rendering_thread() {
}

void DisplayServerWindows::swap_buffers() {
#if defined(GLES3_ENABLED)
	gl_manager->swap_buffers();
#endif
}

void DisplayServerWindows::set_native_icon(const String &p_filename) {
	_THREAD_SAFE_METHOD_

	FileAccess *f = FileAccess::open(p_filename, FileAccess::READ);
	ERR_FAIL_COND_MSG(!f, "Cannot open file with icon '" + p_filename + "'.");

	ICONDIR *icon_dir = (ICONDIR *)memalloc(sizeof(ICONDIR));
	int pos = 0;

	icon_dir->idReserved = f->get_32();
	pos += sizeof(WORD);
	f->seek(pos);

	icon_dir->idType = f->get_32();
	pos += sizeof(WORD);
	f->seek(pos);

	ERR_FAIL_COND_MSG(icon_dir->idType != 1, "Invalid icon file format!");

	icon_dir->idCount = f->get_32();
	pos += sizeof(WORD);
	f->seek(pos);

	icon_dir = (ICONDIR *)memrealloc(icon_dir, 3 * sizeof(WORD) + icon_dir->idCount * sizeof(ICONDIRENTRY));
	f->get_buffer((uint8_t *)&icon_dir->idEntries[0], icon_dir->idCount * sizeof(ICONDIRENTRY));

	int small_icon_index = -1; // Select 16x16 with largest color count.
	int small_icon_cc = 0;
	int big_icon_index = -1; // Select largest.
	int big_icon_width = 16;
	int big_icon_cc = 0;

	for (int i = 0; i < icon_dir->idCount; i++) {
		int colors = (icon_dir->idEntries[i].bColorCount == 0) ? 32768 : icon_dir->idEntries[i].bColorCount;
		int width = (icon_dir->idEntries[i].bWidth == 0) ? 256 : icon_dir->idEntries[i].bWidth;
		if (width == 16) {
			if (colors >= small_icon_cc) {
				small_icon_index = i;
				small_icon_cc = colors;
			}
		}
		if (width >= big_icon_width) {
			if (colors >= big_icon_cc) {
				big_icon_index = i;
				big_icon_width = width;
				big_icon_cc = colors;
			}
		}
	}

	ERR_FAIL_COND_MSG(big_icon_index == -1, "No valid icons found!");

	if (small_icon_index == -1) {
		WARN_PRINT("No small icon found, reusing " + itos(big_icon_width) + "x" + itos(big_icon_width) + " @" + itos(big_icon_cc) + " icon!");
		small_icon_index = big_icon_index;
		small_icon_cc = big_icon_cc;
	}

	// Read the big icon.
	DWORD bytecount_big = icon_dir->idEntries[big_icon_index].dwBytesInRes;
	Vector<uint8_t> data_big;
	data_big.resize(bytecount_big);
	pos = icon_dir->idEntries[big_icon_index].dwImageOffset;
	f->seek(pos);
	f->get_buffer((uint8_t *)&data_big.write[0], bytecount_big);
	HICON icon_big = CreateIconFromResource((PBYTE)&data_big.write[0], bytecount_big, TRUE, 0x00030000);
	ERR_FAIL_COND_MSG(!icon_big, "Could not create " + itos(big_icon_width) + "x" + itos(big_icon_width) + " @" + itos(big_icon_cc) + " icon, error: " + format_error_message(GetLastError()) + ".");

	// Read the small icon.
	DWORD bytecount_small = icon_dir->idEntries[small_icon_index].dwBytesInRes;
	Vector<uint8_t> data_small;
	data_small.resize(bytecount_small);
	pos = icon_dir->idEntries[small_icon_index].dwImageOffset;
	f->seek(pos);
	f->get_buffer((uint8_t *)&data_small.write[0], bytecount_small);
	HICON icon_small = CreateIconFromResource((PBYTE)&data_small.write[0], bytecount_small, TRUE, 0x00030000);
	ERR_FAIL_COND_MSG(!icon_small, "Could not create 16x16 @" + itos(small_icon_cc) + " icon, error: " + format_error_message(GetLastError()) + ".");

	// Online tradition says to be sure last error is cleared and set the small icon first.
	int err = 0;
	SetLastError(err);

	SendMessage(windows[MAIN_WINDOW_ID].hWnd, WM_SETICON, ICON_SMALL, (LPARAM)icon_small);
	err = GetLastError();
	ERR_FAIL_COND_MSG(err, "Error setting ICON_SMALL: " + format_error_message(err) + ".");

	SendMessage(windows[MAIN_WINDOW_ID].hWnd, WM_SETICON, ICON_BIG, (LPARAM)icon_big);
	err = GetLastError();
	ERR_FAIL_COND_MSG(err, "Error setting ICON_BIG: " + format_error_message(err) + ".");

	memdelete(f);
	memdelete(icon_dir);
}

void DisplayServerWindows::set_icon(const Ref<Image> &p_icon) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!p_icon.is_valid());
	Ref<Image> icon = p_icon->duplicate();
	if (icon->get_format() != Image::FORMAT_RGBA8) {
		icon->convert(Image::FORMAT_RGBA8);
	}
	int w = icon->get_width();
	int h = icon->get_height();

	// Create temporary bitmap buffer.
	int icon_len = 40 + h * w * 4;
	Vector<BYTE> v;
	v.resize(icon_len);
	BYTE *icon_bmp = v.ptrw();

	encode_uint32(40, &icon_bmp[0]);
	encode_uint32(w, &icon_bmp[4]);
	encode_uint32(h * 2, &icon_bmp[8]);
	encode_uint16(1, &icon_bmp[12]);
	encode_uint16(32, &icon_bmp[14]);
	encode_uint32(BI_RGB, &icon_bmp[16]);
	encode_uint32(w * h * 4, &icon_bmp[20]);
	encode_uint32(0, &icon_bmp[24]);
	encode_uint32(0, &icon_bmp[28]);
	encode_uint32(0, &icon_bmp[32]);
	encode_uint32(0, &icon_bmp[36]);

	uint8_t *wr = &icon_bmp[40];
	const uint8_t *r = icon->get_data().ptr();

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			const uint8_t *rpx = &r[((h - i - 1) * w + j) * 4];
			uint8_t *wpx = &wr[(i * w + j) * 4];
			wpx[0] = rpx[2];
			wpx[1] = rpx[1];
			wpx[2] = rpx[0];
			wpx[3] = rpx[3];
		}
	}

	HICON hicon = CreateIconFromResource(icon_bmp, icon_len, TRUE, 0x00030000);

	// Set the icon for the window.
	SendMessage(windows[MAIN_WINDOW_ID].hWnd, WM_SETICON, ICON_SMALL, (LPARAM)hicon);

	// Set the icon in the task manager (should we do this?).
	SendMessage(windows[MAIN_WINDOW_ID].hWnd, WM_SETICON, ICON_BIG, (LPARAM)hicon);
}

void DisplayServerWindows::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	_THREAD_SAFE_METHOD_
#if defined(VULKAN_ENABLED)
	// TODO disabling for now
	//context_vulkan->set_vsync_mode(p_window, p_vsync_mode);
#endif
}

DisplayServer::VSyncMode DisplayServerWindows::window_get_vsync_mode(WindowID p_window) const {
	_THREAD_SAFE_METHOD_
#if defined(VULKAN_ENABLED)
	//TODO disabling for now
	//return context_vulkan->get_vsync_mode(p_window);
#endif
	return DisplayServer::VSYNC_ENABLED;
}

void DisplayServerWindows::set_context(Context p_context) {
}

#define MI_WP_SIGNATURE 0xFF515700
#define SIGNATURE_MASK 0xFFFFFF00
// Keeping the name suggested by Microsoft, but this macro really answers:
// Is this mouse event emulated from touch or pen input?
#define IsPenEvent(dw) (((dw)&SIGNATURE_MASK) == MI_WP_SIGNATURE)
// This one tells whether the event comes from touchscreen (and not from pen).
#define IsTouchEvent(dw) (IsPenEvent(dw) && ((dw)&0x80))

void DisplayServerWindows::_touch_event(WindowID p_window, bool p_pressed, float p_x, float p_y, int idx) {
	if (touch_state.has(idx) == p_pressed) {
		return;
	}

	if (p_pressed) {
		touch_state.insert(idx, Vector2(p_x, p_y));
	} else {
		touch_state.erase(idx);
	}

	Ref<InputEventScreenTouch> event;
	event.instantiate();
	event->set_index(idx);
	event->set_window_id(p_window);
	event->set_pressed(p_pressed);
	event->set_position(Vector2(p_x, p_y));

	Input::get_singleton()->parse_input_event(event);
}

void DisplayServerWindows::_drag_event(WindowID p_window, float p_x, float p_y, int idx) {
	Map<int, Vector2>::Element *curr = touch_state.find(idx);
	if (!curr) {
		return;
	}

	if (curr->get() == Vector2(p_x, p_y)) {
		return;
	}

	Ref<InputEventScreenDrag> event;
	event.instantiate();
	event->set_window_id(p_window);
	event->set_index(idx);
	event->set_position(Vector2(p_x, p_y));
	event->set_relative(Vector2(p_x, p_y) - curr->get());

	Input::get_singleton()->parse_input_event(event);

	curr->get() = Vector2(p_x, p_y);
}

void DisplayServerWindows::_send_window_event(const WindowData &wd, WindowEvent p_event) {
	if (!wd.event_callback.is_null()) {
		Variant event = int(p_event);
		Variant *eventp = &event;
		Variant ret;
		Callable::CallError ce;
		wd.event_callback.call((const Variant **)&eventp, 1, ret, ce);
	}
}

void DisplayServerWindows::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	((DisplayServerWindows *)(get_singleton()))->_dispatch_input_event(p_event);
}

void DisplayServerWindows::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	_THREAD_SAFE_METHOD_
	if (in_dispatch_input_event) {
		return;
	}

	in_dispatch_input_event = true;
	Variant ev = p_event;
	Variant *evp = &ev;
	Variant ret;
	Callable::CallError ce;

	Ref<InputEventFromWindow> event_from_window = p_event;
	if (event_from_window.is_valid() && event_from_window->get_window_id() != INVALID_WINDOW_ID) {
		// Send to a single window.
		if (!windows.has(event_from_window->get_window_id())) {
			in_dispatch_input_event = false;
			ERR_FAIL_MSG("DisplayServerWindows: Invalid window id in input event.");
		}
		Callable callable = windows[event_from_window->get_window_id()].input_event_callback;
		if (callable.is_null()) {
			in_dispatch_input_event = false;
			return;
		}
		callable.call((const Variant **)&evp, 1, ret, ce);
	} else {
		// Send to all windows.
		for (const KeyValue<WindowID, WindowData> &E : windows) {
			const Callable callable = E.value.input_event_callback;
			if (callable.is_null()) {
				continue;
			}
			callable.call((const Variant **)&evp, 1, ret, ce);
		}
	}

	in_dispatch_input_event = false;
}

// Our default window procedure to handle processing of window-related system messages/events.
// Also known as DefProc or DefWindowProc.
// See: https://docs.microsoft.com/en-us/windows/win32/winmsg/window-procedures
LRESULT DisplayServerWindows::WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	if (drop_events) {
		if (user_proc) {
			return CallWindowProcW(user_proc, hWnd, uMsg, wParam, lParam);
		} else {
			return DefWindowProcW(hWnd, uMsg, wParam, lParam);
		}
	};

	WindowID window_id = INVALID_WINDOW_ID;
	bool window_created = false;

	// Check whether window exists.
	for (const KeyValue<WindowID, WindowData> &E : windows) {
		if (E.value.hWnd == hWnd) {
			window_id = E.key;
			window_created = true;
			break;
		}
	}

	// Window doesn't exist or creation in progress, don't handle messages yet.
	if (!window_created) {
		window_id = window_id_counter;
		ERR_FAIL_COND_V(!windows.has(window_id), 0);
	}

	// Process window messages.
	switch (uMsg) {
		case WM_SETFOCUS: {
			windows[window_id].window_has_focus = true;
			last_focused_window = window_id;

			// Restore mouse mode.
			_set_mouse_mode_impl(mouse_mode);

			if (!app_focused) {
				if (OS::get_singleton()->get_main_loop()) {
					OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_IN);
				}
				app_focused = true;
			}
		} break;
		case WM_KILLFOCUS: {
			windows[window_id].window_has_focus = false;
			last_focused_window = window_id;

			// Release capture unconditionally because it can be set due to dragging, in addition to captured mode.
			ReleaseCapture();

			// Release every touch to avoid sticky points.
			for (const KeyValue<int, Vector2> &E : touch_state) {
				_touch_event(window_id, false, E.value.x, E.value.y, E.key);
			}
			touch_state.clear();

			bool self_steal = false;
			HWND new_hwnd = (HWND)wParam;
			if (IsWindow(new_hwnd)) {
				self_steal = true;
			}

			if (!self_steal) {
				if (OS::get_singleton()->get_main_loop()) {
					OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_APPLICATION_FOCUS_OUT);
				}
				app_focused = false;
			}
		} break;
		case WM_ACTIVATE: { // Watch for window activate message.
			if (!windows[window_id].window_focused) {
				_process_activate_event(window_id, wParam, lParam);
			} else {
				windows[window_id].saved_wparam = wParam;
				windows[window_id].saved_lparam = lParam;

				// Run a timer to prevent event catching warning if the focused window is closing.
				windows[window_id].focus_timer_id = SetTimer(windows[window_id].hWnd, 2, USER_TIMER_MINIMUM, (TIMERPROC) nullptr);
			}
			if (wParam != WA_INACTIVE) {
				track_mouse_leave_event(hWnd);
			}
			return 0; // Return to the message loop.
		} break;
		case WM_GETMINMAXINFO: {
			if (windows[window_id].resizable && !windows[window_id].fullscreen) {
				// Size of window decorations.
				Size2 decor = window_get_real_size(window_id) - window_get_size(window_id);

				MINMAXINFO *min_max_info = (MINMAXINFO *)lParam;
				if (windows[window_id].min_size != Size2()) {
					min_max_info->ptMinTrackSize.x = windows[window_id].min_size.x + decor.x;
					min_max_info->ptMinTrackSize.y = windows[window_id].min_size.y + decor.y;
				}
				if (windows[window_id].max_size != Size2()) {
					min_max_info->ptMaxTrackSize.x = windows[window_id].max_size.x + decor.x;
					min_max_info->ptMaxTrackSize.y = windows[window_id].max_size.y + decor.y;
				}
				return 0;
			}
		} break;
		case WM_PAINT: {
			Main::force_redraw();
		} break;
		case WM_SYSCOMMAND: // Intercept system commands.
		{
			switch (wParam) // Check system calls.
			{
				case SC_SCREENSAVE: // Screensaver trying to start?
				case SC_MONITORPOWER: // Monitor trying to enter powersave?
					return 0; // Prevent from happening.
				case SC_KEYMENU:
					if ((lParam >> 16) <= 0)
						return 0;
			}
		} break;
		case WM_CLOSE: // Did we receive a close message?
		{
			if (windows[window_id].focus_timer_id != 0U) {
				KillTimer(windows[window_id].hWnd, windows[window_id].focus_timer_id);
			}
			_send_window_event(windows[window_id], WINDOW_EVENT_CLOSE_REQUEST);

			return 0; // Jump back.
		}
		case WM_MOUSELEAVE: {
			old_invalid = true;
			outside = true;

			_send_window_event(windows[window_id], WINDOW_EVENT_MOUSE_EXIT);

		} break;
		case WM_INPUT: {
			if (mouse_mode != MOUSE_MODE_CAPTURED || !use_raw_input) {
				break;
			}

			UINT dwSize;

			GetRawInputData((HRAWINPUT)lParam, RID_INPUT, nullptr, &dwSize, sizeof(RAWINPUTHEADER));
			LPBYTE lpb = new BYTE[dwSize];
			if (lpb == nullptr) {
				return 0;
			}

			if (GetRawInputData((HRAWINPUT)lParam, RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER)) != dwSize)
				OutputDebugString(TEXT("GetRawInputData does not return correct size !\n"));

			RAWINPUT *raw = (RAWINPUT *)lpb;

			if (raw->header.dwType == RIM_TYPEMOUSE) {
				Ref<InputEventMouseMotion> mm;
				mm.instantiate();

				mm->set_window_id(window_id);
				mm->set_ctrl_pressed(control_mem);
				mm->set_shift_pressed(shift_mem);
				mm->set_alt_pressed(alt_mem);

				mm->set_pressure((raw->data.mouse.ulButtons & RI_MOUSE_LEFT_BUTTON_DOWN) ? 1.0f : 0.0f);

				mm->set_button_mask(last_button_state);

				Point2i c(windows[window_id].width / 2, windows[window_id].height / 2);

				// Centering just so it works as before.
				POINT pos = { (int)c.x, (int)c.y };
				ClientToScreen(windows[window_id].hWnd, &pos);
				SetCursorPos(pos.x, pos.y);

				mm->set_position(c);
				mm->set_global_position(c);
				Input::get_singleton()->set_mouse_position(c);
				mm->set_velocity(Vector2(0, 0));

				if (raw->data.mouse.usFlags == MOUSE_MOVE_RELATIVE) {
					mm->set_relative(Vector2(raw->data.mouse.lLastX, raw->data.mouse.lLastY));

				} else if (raw->data.mouse.usFlags == MOUSE_MOVE_ABSOLUTE) {
					int nScreenWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
					int nScreenHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
					int nScreenLeft = GetSystemMetrics(SM_XVIRTUALSCREEN);
					int nScreenTop = GetSystemMetrics(SM_YVIRTUALSCREEN);

					Vector2 abs_pos(
							(double(raw->data.mouse.lLastX) - 65536.0 / (nScreenWidth)) * nScreenWidth / 65536.0 + nScreenLeft,
							(double(raw->data.mouse.lLastY) - 65536.0 / (nScreenHeight)) * nScreenHeight / 65536.0 + nScreenTop);

					POINT coords; // Client coords.
					coords.x = abs_pos.x;
					coords.y = abs_pos.y;

					ScreenToClient(hWnd, &coords);

					mm->set_relative(Vector2(coords.x - old_x, coords.y - old_y));
					old_x = coords.x;
					old_y = coords.y;
				}

				if (windows[window_id].window_has_focus && mm->get_relative() != Vector2()) {
					Input::get_singleton()->parse_input_event(mm);
				}
			}
			delete[] lpb;
		} break;
		case WT_CSRCHANGE:
		case WT_PROXIMITY: {
			if ((tablet_get_current_driver() == "wintab") && wintab_available && windows[window_id].wtctx) {
				AXIS pressure;
				if (wintab_WTInfo(WTI_DEVICES + windows[window_id].wtlc.lcDevice, DVC_NPRESSURE, &pressure)) {
					windows[window_id].min_pressure = int(pressure.axMin);
					windows[window_id].max_pressure = int(pressure.axMax);
				}
				AXIS orientation[3];
				if (wintab_WTInfo(WTI_DEVICES + windows[window_id].wtlc.lcDevice, DVC_ORIENTATION, &orientation)) {
					windows[window_id].tilt_supported = orientation[0].axResolution && orientation[1].axResolution;
				}
				return 0;
			}
		} break;
		case WT_PACKET: {
			if ((tablet_get_current_driver() == "wintab") && wintab_available && windows[window_id].wtctx) {
				PACKET packet;
				if (wintab_WTPacket(windows[window_id].wtctx, wParam, &packet)) {
					float pressure = float(packet.pkNormalPressure - windows[window_id].min_pressure) / float(windows[window_id].max_pressure - windows[window_id].min_pressure);
					windows[window_id].last_pressure = pressure;
					windows[window_id].last_pressure_update = 0;

					double azim = (packet.pkOrientation.orAzimuth / 10.0f) * (Math_PI / 180);
					double alt = Math::tan((Math::abs(packet.pkOrientation.orAltitude / 10.0f)) * (Math_PI / 180));

					if (windows[window_id].tilt_supported) {
						windows[window_id].last_tilt = Vector2(Math::atan(Math::sin(azim) / alt), Math::atan(Math::cos(azim) / alt));
					} else {
						windows[window_id].last_tilt = Vector2();
					}

					POINT coords;
					GetCursorPos(&coords);
					ScreenToClient(windows[window_id].hWnd, &coords);

					// Don't calculate relative mouse movement if we don't have focus in CAPTURED mode.
					if (!windows[window_id].window_has_focus && mouse_mode == MOUSE_MODE_CAPTURED)
						break;

					Ref<InputEventMouseMotion> mm;
					mm.instantiate();
					mm->set_window_id(window_id);
					mm->set_ctrl_pressed(GetKeyState(VK_CONTROL) < 0);
					mm->set_shift_pressed(GetKeyState(VK_SHIFT) < 0);
					mm->set_alt_pressed(alt_mem);

					mm->set_pressure(windows[window_id].last_pressure);
					mm->set_tilt(windows[window_id].last_tilt);

					mm->set_button_mask(last_button_state);

					mm->set_position(Vector2(coords.x, coords.y));
					mm->set_global_position(Vector2(coords.x, coords.y));

					if (mouse_mode == MOUSE_MODE_CAPTURED) {
						Point2i c(windows[window_id].width / 2, windows[window_id].height / 2);
						old_x = c.x;
						old_y = c.y;

						if (mm->get_position() == c) {
							center = c;
							return 0;
						}

						Point2i ncenter = mm->get_position();
						center = ncenter;
						POINT pos = { (int)c.x, (int)c.y };
						ClientToScreen(windows[window_id].hWnd, &pos);
						SetCursorPos(pos.x, pos.y);
					}

					Input::get_singleton()->set_mouse_position(mm->get_position());
					mm->set_velocity(Input::get_singleton()->get_last_mouse_velocity());

					if (old_invalid) {
						old_x = mm->get_position().x;
						old_y = mm->get_position().y;
						old_invalid = false;
					}

					mm->set_relative(Vector2(mm->get_position() - Vector2(old_x, old_y)));
					old_x = mm->get_position().x;
					old_y = mm->get_position().y;
					if (windows[window_id].window_has_focus)
						Input::get_singleton()->parse_input_event(mm);
				}
				return 0;
			}
		} break;
		case WM_POINTERENTER: {
			if (mouse_mode == MOUSE_MODE_CAPTURED && use_raw_input) {
				break;
			}

			if ((tablet_get_current_driver() != "winink") || !winink_available) {
				break;
			}

			uint32_t pointer_id = LOWORD(wParam);
			POINTER_INPUT_TYPE pointer_type = PT_POINTER;
			if (!win8p_GetPointerType(pointer_id, &pointer_type)) {
				break;
			}

			if (pointer_type != PT_PEN) {
				break;
			}

			windows[window_id].block_mm = true;
			return 0;
		} break;
		case WM_POINTERLEAVE: {
			windows[window_id].block_mm = false;
			return 0;
		} break;
		case WM_POINTERUPDATE: {
			if (mouse_mode == MOUSE_MODE_CAPTURED && use_raw_input) {
				break;
			}

			if ((tablet_get_current_driver() != "winink") || !winink_available) {
				break;
			}

			uint32_t pointer_id = LOWORD(wParam);
			POINTER_INPUT_TYPE pointer_type = PT_POINTER;
			if (!win8p_GetPointerType(pointer_id, &pointer_type)) {
				break;
			}

			if (pointer_type != PT_PEN) {
				break;
			}

			POINTER_PEN_INFO pen_info;
			if (!win8p_GetPointerPenInfo(pointer_id, &pen_info)) {
				break;
			}

			if (Input::get_singleton()->is_emulating_mouse_from_touch()) {
				// Universal translation enabled; ignore OS translation.
				LPARAM extra = GetMessageExtraInfo();
				if (IsTouchEvent(extra)) {
					break;
				}
			}

			if (outside) {
				// Mouse enter.

				if (mouse_mode != MOUSE_MODE_CAPTURED) {
					_send_window_event(windows[window_id], WINDOW_EVENT_MOUSE_ENTER);
				}

				CursorShape c = cursor_shape;
				cursor_shape = CURSOR_MAX;
				cursor_set_shape(c);
				outside = false;

				// Once-off notification, must call again.
				track_mouse_leave_event(hWnd);
			}

			// Don't calculate relative mouse movement if we don't have focus in CAPTURED mode.
			if (!windows[window_id].window_has_focus && mouse_mode == MOUSE_MODE_CAPTURED) {
				break;
			}

			Ref<InputEventMouseMotion> mm;
			mm.instantiate();

			mm->set_window_id(window_id);
			if (pen_info.penMask & PEN_MASK_PRESSURE) {
				mm->set_pressure((float)pen_info.pressure / 1024);
			} else {
				mm->set_pressure((HIWORD(wParam) & POINTER_MESSAGE_FLAG_FIRSTBUTTON) ? 1.0f : 0.0f);
			}
			if ((pen_info.penMask & PEN_MASK_TILT_X) && (pen_info.penMask & PEN_MASK_TILT_Y)) {
				mm->set_tilt(Vector2((float)pen_info.tiltX / 90, (float)pen_info.tiltY / 90));
			}

			mm->set_ctrl_pressed(GetKeyState(VK_CONTROL) < 0);
			mm->set_shift_pressed(GetKeyState(VK_SHIFT) < 0);
			mm->set_alt_pressed(alt_mem);

			mm->set_button_mask(last_button_state);

			POINT coords; // Client coords.
			coords.x = GET_X_LPARAM(lParam);
			coords.y = GET_Y_LPARAM(lParam);

			ScreenToClient(windows[window_id].hWnd, &coords);

			mm->set_position(Vector2(coords.x, coords.y));
			mm->set_global_position(Vector2(coords.x, coords.y));

			if (mouse_mode == MOUSE_MODE_CAPTURED) {
				Point2i c(windows[window_id].width / 2, windows[window_id].height / 2);
				old_x = c.x;
				old_y = c.y;

				if (mm->get_position() == c) {
					center = c;
					return 0;
				}

				Point2i ncenter = mm->get_position();
				center = ncenter;
				POINT pos = { (int)c.x, (int)c.y };
				ClientToScreen(hWnd, &pos);
				SetCursorPos(pos.x, pos.y);
			}

			Input::get_singleton()->set_mouse_position(mm->get_position());
			mm->set_velocity(Input::get_singleton()->get_last_mouse_velocity());

			if (old_invalid) {
				old_x = mm->get_position().x;
				old_y = mm->get_position().y;
				old_invalid = false;
			}

			mm->set_relative(Vector2(mm->get_position() - Vector2(old_x, old_y)));
			old_x = mm->get_position().x;
			old_y = mm->get_position().y;
			if (windows[window_id].window_has_focus) {
				Input::get_singleton()->parse_input_event(mm);
			}

			return 0; // Pointer event handled return 0 to avoid duplicate WM_MOUSEMOVE event.
		} break;
		case WM_MOUSEMOVE: {
			if (windows[window_id].block_mm) {
				break;
			}

			if (mouse_mode == MOUSE_MODE_CAPTURED && use_raw_input) {
				break;
			}

			if (Input::get_singleton()->is_emulating_mouse_from_touch()) {
				// Universal translation enabled; ignore OS translation.
				LPARAM extra = GetMessageExtraInfo();
				if (IsTouchEvent(extra)) {
					break;
				}
			}

			if (outside) {
				// Mouse enter.

				if (mouse_mode != MOUSE_MODE_CAPTURED) {
					_send_window_event(windows[window_id], WINDOW_EVENT_MOUSE_ENTER);
				}

				CursorShape c = cursor_shape;
				cursor_shape = CURSOR_MAX;
				cursor_set_shape(c);
				outside = false;

				// Once-off notification, must call again.
				track_mouse_leave_event(hWnd);
			}

			// Don't calculate relative mouse movement if we don't have focus in CAPTURED mode.
			if (!windows[window_id].window_has_focus && mouse_mode == MOUSE_MODE_CAPTURED) {
				break;
			}

			Ref<InputEventMouseMotion> mm;
			mm.instantiate();
			mm->set_window_id(window_id);
			mm->set_ctrl_pressed((wParam & MK_CONTROL) != 0);
			mm->set_shift_pressed((wParam & MK_SHIFT) != 0);
			mm->set_alt_pressed(alt_mem);

			if ((tablet_get_current_driver() == "wintab") && wintab_available && windows[window_id].wtctx) {
				// Note: WinTab sends both WT_PACKET and WM_xBUTTONDOWN/UP/MOUSEMOVE events, use mouse 1/0 pressure only when last_pressure was not updated recently.
				if (windows[window_id].last_pressure_update < 10) {
					windows[window_id].last_pressure_update++;
				} else {
					windows[window_id].last_tilt = Vector2();
					windows[window_id].last_pressure = (wParam & MK_LBUTTON) ? 1.0f : 0.0f;
				}
			} else {
				windows[window_id].last_tilt = Vector2();
				windows[window_id].last_pressure = (wParam & MK_LBUTTON) ? 1.0f : 0.0f;
			}

			mm->set_pressure(windows[window_id].last_pressure);
			mm->set_tilt(windows[window_id].last_tilt);

			mm->set_button_mask(last_button_state);

			mm->set_position(Vector2(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)));
			mm->set_global_position(Vector2(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)));

			if (mouse_mode == MOUSE_MODE_CAPTURED) {
				Point2i c(windows[window_id].width / 2, windows[window_id].height / 2);
				old_x = c.x;
				old_y = c.y;

				if (mm->get_position() == c) {
					center = c;
					return 0;
				}

				Point2i ncenter = mm->get_position();
				center = ncenter;
				POINT pos = { (int)c.x, (int)c.y };
				ClientToScreen(windows[window_id].hWnd, &pos);
				SetCursorPos(pos.x, pos.y);
			}

			Input::get_singleton()->set_mouse_position(mm->get_position());
			mm->set_velocity(Input::get_singleton()->get_last_mouse_velocity());

			if (old_invalid) {
				old_x = mm->get_position().x;
				old_y = mm->get_position().y;
				old_invalid = false;
			}

			mm->set_relative(Vector2(mm->get_position() - Vector2(old_x, old_y)));
			old_x = mm->get_position().x;
			old_y = mm->get_position().y;
			if (windows[window_id].window_has_focus)
				Input::get_singleton()->parse_input_event(mm);

		} break;
		case WM_LBUTTONDOWN:
		case WM_LBUTTONUP:
			if (Input::get_singleton()->is_emulating_mouse_from_touch()) {
				// Universal translation enabled; ignore OS translations for left button.
				LPARAM extra = GetMessageExtraInfo();
				if (IsTouchEvent(extra)) {
					break;
				}
			}
			[[fallthrough]];
		case WM_MBUTTONDOWN:
		case WM_MBUTTONUP:
		case WM_RBUTTONDOWN:
		case WM_RBUTTONUP:
		case WM_MOUSEWHEEL:
		case WM_MOUSEHWHEEL:
		case WM_LBUTTONDBLCLK:
		case WM_MBUTTONDBLCLK:
		case WM_RBUTTONDBLCLK:
		case WM_XBUTTONDBLCLK:
		case WM_XBUTTONDOWN:
		case WM_XBUTTONUP: {
			Ref<InputEventMouseButton> mb;
			mb.instantiate();
			mb->set_window_id(window_id);

			switch (uMsg) {
				case WM_LBUTTONDOWN: {
					mb->set_pressed(true);
					mb->set_button_index(MouseButton::LEFT);
				} break;
				case WM_LBUTTONUP: {
					mb->set_pressed(false);
					mb->set_button_index(MouseButton::LEFT);
				} break;
				case WM_MBUTTONDOWN: {
					mb->set_pressed(true);
					mb->set_button_index(MouseButton::MIDDLE);
				} break;
				case WM_MBUTTONUP: {
					mb->set_pressed(false);
					mb->set_button_index(MouseButton::MIDDLE);
				} break;
				case WM_RBUTTONDOWN: {
					mb->set_pressed(true);
					mb->set_button_index(MouseButton::RIGHT);
				} break;
				case WM_RBUTTONUP: {
					mb->set_pressed(false);
					mb->set_button_index(MouseButton::RIGHT);
				} break;
				case WM_LBUTTONDBLCLK: {
					mb->set_pressed(true);
					mb->set_button_index(MouseButton::LEFT);
					mb->set_double_click(true);
				} break;
				case WM_RBUTTONDBLCLK: {
					mb->set_pressed(true);
					mb->set_button_index(MouseButton::RIGHT);
					mb->set_double_click(true);
				} break;
				case WM_MBUTTONDBLCLK: {
					mb->set_pressed(true);
					mb->set_button_index(MouseButton::MIDDLE);
					mb->set_double_click(true);
				} break;
				case WM_MOUSEWHEEL: {
					mb->set_pressed(true);
					int motion = (short)HIWORD(wParam);
					if (!motion) {
						return 0;
					}

					if (motion > 0) {
						mb->set_button_index(MouseButton::WHEEL_UP);
					} else {
						mb->set_button_index(MouseButton::WHEEL_DOWN);
					}
					mb->set_factor(fabs((double)motion / (double)WHEEL_DELTA));
				} break;
				case WM_MOUSEHWHEEL: {
					mb->set_pressed(true);
					int motion = (short)HIWORD(wParam);
					if (!motion) {
						return 0;
					}

					if (motion < 0) {
						mb->set_button_index(MouseButton::WHEEL_LEFT);
					} else {
						mb->set_button_index(MouseButton::WHEEL_RIGHT);
					}
					mb->set_factor(fabs((double)motion / (double)WHEEL_DELTA));
				} break;
				case WM_XBUTTONDOWN: {
					mb->set_pressed(true);
					if (HIWORD(wParam) == XBUTTON1) {
						mb->set_button_index(MouseButton::MB_XBUTTON1);
					} else {
						mb->set_button_index(MouseButton::MB_XBUTTON2);
					}
				} break;
				case WM_XBUTTONUP: {
					mb->set_pressed(false);
					if (HIWORD(wParam) == XBUTTON1) {
						mb->set_button_index(MouseButton::MB_XBUTTON1);
					} else {
						mb->set_button_index(MouseButton::MB_XBUTTON2);
					}
				} break;
				case WM_XBUTTONDBLCLK: {
					mb->set_pressed(true);
					if (HIWORD(wParam) == XBUTTON1) {
						mb->set_button_index(MouseButton::MB_XBUTTON1);
					} else {
						mb->set_button_index(MouseButton::MB_XBUTTON2);
					}
					mb->set_double_click(true);
				} break;
				default: {
					return 0;
				}
			}

			mb->set_ctrl_pressed((wParam & MK_CONTROL) != 0);
			mb->set_shift_pressed((wParam & MK_SHIFT) != 0);
			mb->set_alt_pressed(alt_mem);
			// mb->is_alt_pressed()=(wParam&MK_MENU)!=0;
			if (mb->is_pressed()) {
				last_button_state |= mouse_button_to_mask(mb->get_button_index());
			} else {
				last_button_state &= ~mouse_button_to_mask(mb->get_button_index());
			}
			mb->set_button_mask(last_button_state);

			mb->set_position(Vector2(GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)));

			if (mouse_mode == MOUSE_MODE_CAPTURED && !use_raw_input) {
				mb->set_position(Vector2(old_x, old_y));
			}

			if (uMsg != WM_MOUSEWHEEL && uMsg != WM_MOUSEHWHEEL) {
				if (mb->is_pressed()) {
					if (++pressrc > 0 && mouse_mode != MOUSE_MODE_CAPTURED)
						SetCapture(hWnd);
				} else {
					if (--pressrc <= 0) {
						if (mouse_mode != MOUSE_MODE_CAPTURED) {
							ReleaseCapture();
						}
						pressrc = 0;
					}
				}
			} else {
				// For reasons unknown to mankind, wheel comes in screen coordinates.
				POINT coords;
				coords.x = mb->get_position().x;
				coords.y = mb->get_position().y;

				ScreenToClient(hWnd, &coords);

				mb->set_position(Vector2(coords.x, coords.y));
			}

			mb->set_global_position(mb->get_position());

			Input::get_singleton()->parse_input_event(mb);
			if (mb->is_pressed() && mb->get_button_index() >= MouseButton::WHEEL_UP && mb->get_button_index() <= MouseButton::WHEEL_RIGHT) {
				// Send release for mouse wheel.
				Ref<InputEventMouseButton> mbd = mb->duplicate();
				mbd->set_window_id(window_id);
				last_button_state &= ~mouse_button_to_mask(mbd->get_button_index());
				mbd->set_button_mask(last_button_state);
				mbd->set_pressed(false);
				Input::get_singleton()->parse_input_event(mbd);
			}

		} break;
		case WM_MOVE: {
			if (!IsIconic(windows[window_id].hWnd)) {
				int x = int16_t(LOWORD(lParam));
				int y = int16_t(HIWORD(lParam));
				windows[window_id].last_pos = Point2(x, y);

				if (!windows[window_id].rect_changed_callback.is_null()) {
					Variant size = Rect2i(windows[window_id].last_pos.x, windows[window_id].last_pos.y, windows[window_id].width, windows[window_id].height);
					Variant *sizep = &size;
					Variant ret;
					Callable::CallError ce;
					windows[window_id].rect_changed_callback.call((const Variant **)&sizep, 1, ret, ce);
				}
			}
		} break;
		case WM_SIZE: {
			// Ignore window size change when a SIZE_MINIMIZED event is triggered.
			if (wParam != SIZE_MINIMIZED) {
				// The new width and height of the client area.
				int window_w = LOWORD(lParam);
				int window_h = HIWORD(lParam);

				// Set new value to the size if it isn't preserved.
				if (window_w > 0 && window_h > 0 && !windows[window_id].preserve_window_size) {
					windows[window_id].width = window_w;
					windows[window_id].height = window_h;

#if defined(VULKAN_ENABLED)
					if (context_vulkan && window_created) {
						context_vulkan->window_resize(window_id, windows[window_id].width, windows[window_id].height);
					}
#endif

				} else { // If the size is preserved.
					windows[window_id].preserve_window_size = false;

					// Restore the old size.
					window_set_size(Size2(windows[window_id].width, windows[window_id].height), window_id);
				}
			} else { // When the window has been minimized, preserve its size.
				windows[window_id].preserve_window_size = true;
			}

			// Call windows rect change callback.
			if (!windows[window_id].rect_changed_callback.is_null()) {
				Variant size = Rect2i(windows[window_id].last_pos.x, windows[window_id].last_pos.y, windows[window_id].width, windows[window_id].height);
				Variant *size_ptr = &size;
				Variant ret;
				Callable::CallError ce;
				windows[window_id].rect_changed_callback.call((const Variant **)&size_ptr, 1, ret, ce);
			}

			// The window has been maximized.
			if (wParam == SIZE_MAXIMIZED) {
				windows[window_id].maximized = true;
				windows[window_id].minimized = false;
			}
			// The window has been minimized.
			else if (wParam == SIZE_MINIMIZED) {
				windows[window_id].maximized = false;
				windows[window_id].minimized = true;
				windows[window_id].preserve_window_size = false;
			}
			// The window has been resized, but neither the SIZE_MINIMIZED nor SIZE_MAXIMIZED value applies.
			else if (wParam == SIZE_RESTORED) {
				windows[window_id].maximized = false;
				windows[window_id].minimized = false;
			}
#if 0
			if (is_layered_allowed() && layered_window) {
				DeleteObject(hBitmap);

				RECT r;
				GetWindowRect(hWnd, &r);
				dib_size = Size2i(r.right - r.left, r.bottom - r.top);

				BITMAPINFO bmi;
				ZeroMemory(&bmi, sizeof(BITMAPINFO));
				bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
				bmi.bmiHeader.biWidth = dib_size.x;
				bmi.bmiHeader.biHeight = dib_size.y;
				bmi.bmiHeader.biPlanes = 1;
				bmi.bmiHeader.biBitCount = 32;
				bmi.bmiHeader.biCompression = BI_RGB;
				bmi.bmiHeader.biSizeImage = dib_size.x * dib_size.y * 4;
				hBitmap = CreateDIBSection(hDC_dib, &bmi, DIB_RGB_COLORS, (void **)&dib_data, nullptr, 0x0);
				SelectObject(hDC_dib, hBitmap);

				ZeroMemory(dib_data, dib_size.x * dib_size.y * 4);
			}
#endif
		} break;
		case WM_ENTERSIZEMOVE: {
			Input::get_singleton()->release_pressed_events();
			windows[window_id].move_timer_id = SetTimer(windows[window_id].hWnd, 1, USER_TIMER_MINIMUM, (TIMERPROC) nullptr);
		} break;
		case WM_EXITSIZEMOVE: {
			KillTimer(windows[window_id].hWnd, windows[window_id].move_timer_id);
		} break;
		case WM_TIMER: {
			if (wParam == windows[window_id].move_timer_id) {
				_process_key_events();
				if (!Main::is_iterating()) {
					Main::iteration();
				}
			} else if (wParam == windows[window_id].focus_timer_id) {
				_process_activate_event(window_id, windows[window_id].saved_wparam, windows[window_id].saved_lparam);
				KillTimer(windows[window_id].hWnd, wParam);
				windows[window_id].focus_timer_id = 0U;
			}
		} break;
		case WM_SYSKEYDOWN:
		case WM_SYSKEYUP:
		case WM_KEYUP:
		case WM_KEYDOWN: {
			if (wParam == VK_SHIFT)
				shift_mem = (uMsg == WM_KEYDOWN || uMsg == WM_SYSKEYDOWN);
			if (wParam == VK_CONTROL)
				control_mem = (uMsg == WM_KEYDOWN || uMsg == WM_SYSKEYDOWN);
			if (wParam == VK_MENU) {
				alt_mem = (uMsg == WM_KEYDOWN || uMsg == WM_SYSKEYDOWN);
				if (lParam & (1 << 24))
					gr_mem = alt_mem;
			}

			if (mouse_mode == MOUSE_MODE_CAPTURED) {
				// When SetCapture is used, ALT+F4 hotkey is ignored by Windows, so handle it ourselves
				if (wParam == VK_F4 && alt_mem && (uMsg == WM_KEYDOWN || uMsg == WM_SYSKEYDOWN)) {
					_send_window_event(windows[window_id], WINDOW_EVENT_CLOSE_REQUEST);
				}
			}
			/*
			if (wParam==VK_WIN) TODO wtf is this?
				meta_mem=uMsg==WM_KEYDOWN;
			*/
			[[fallthrough]];
		}
		case WM_CHAR: {
			ERR_BREAK(key_event_pos >= KEY_EVENT_BUFFER_SIZE);

			// Make sure we don't include modifiers for the modifier key itself.
			KeyEvent ke;
			ke.shift = (wParam != VK_SHIFT) ? shift_mem : false;
			ke.alt = (!(wParam == VK_MENU && (uMsg == WM_KEYDOWN || uMsg == WM_SYSKEYDOWN))) ? alt_mem : false;
			ke.control = (wParam != VK_CONTROL) ? control_mem : false;
			ke.meta = meta_mem;
			ke.uMsg = uMsg;
			ke.window_id = window_id;

			if (ke.uMsg == WM_SYSKEYDOWN)
				ke.uMsg = WM_KEYDOWN;
			if (ke.uMsg == WM_SYSKEYUP)
				ke.uMsg = WM_KEYUP;

			ke.wParam = wParam;
			ke.lParam = lParam;
			key_event_buffer[key_event_pos++] = ke;

		} break;
		case WM_INPUTLANGCHANGEREQUEST: {
			// FIXME: Do something?
		} break;
		case WM_TOUCH: {
			BOOL bHandled = FALSE;
			UINT cInputs = LOWORD(wParam);
			PTOUCHINPUT pInputs = memnew_arr(TOUCHINPUT, cInputs);
			if (pInputs) {
				if (GetTouchInputInfo((HTOUCHINPUT)lParam, cInputs, pInputs, sizeof(TOUCHINPUT))) {
					for (UINT i = 0; i < cInputs; i++) {
						TOUCHINPUT ti = pInputs[i];
						POINT touch_pos = {
							TOUCH_COORD_TO_PIXEL(ti.x),
							TOUCH_COORD_TO_PIXEL(ti.y),
						};
						ScreenToClient(hWnd, &touch_pos);
						// Do something with each touch input entry.
						if (ti.dwFlags & TOUCHEVENTF_MOVE) {
							_drag_event(window_id, touch_pos.x, touch_pos.y, ti.dwID);
						} else if (ti.dwFlags & (TOUCHEVENTF_UP | TOUCHEVENTF_DOWN)) {
							_touch_event(window_id, ti.dwFlags & TOUCHEVENTF_DOWN, touch_pos.x, touch_pos.y, ti.dwID);
						};
					}
					bHandled = TRUE;
				} else {
					// TODO: Handle the error here.
				}
				memdelete_arr(pInputs);
			} else {
				// TODO: Handle the error here, probably out of memory.
			}
			if (bHandled) {
				CloseTouchInputHandle((HTOUCHINPUT)lParam);
				return 0;
			};

		} break;
		case WM_DEVICECHANGE: {
			joypad->probe_joypads();
		} break;
		case WM_DESTROY: {
			Input::get_singleton()->flush_buffered_events();
		} break;
		case WM_SETCURSOR: {
			if (LOWORD(lParam) == HTCLIENT) {
				if (windows[window_id].window_has_focus && (mouse_mode == MOUSE_MODE_HIDDEN || mouse_mode == MOUSE_MODE_CAPTURED || mouse_mode == MOUSE_MODE_CONFINED_HIDDEN)) {
					// Hide the cursor.
					if (hCursor == nullptr) {
						hCursor = SetCursor(nullptr);
					} else {
						SetCursor(nullptr);
					}
				} else {
					if (hCursor != nullptr) {
						CursorShape c = cursor_shape;
						cursor_shape = CURSOR_MAX;
						cursor_set_shape(c);
						hCursor = nullptr;
					}
				}
			}
		} break;
		case WM_DROPFILES: {
			HDROP hDropInfo = (HDROP)wParam;
			const int buffsize = 4096;
			WCHAR buf[buffsize];

			int fcount = DragQueryFileW(hDropInfo, 0xFFFFFFFF, nullptr, 0);

			Vector<String> files;

			for (int i = 0; i < fcount; i++) {
				DragQueryFileW(hDropInfo, i, buf, buffsize);
				String file = String::utf16((const char16_t *)buf);
				files.push_back(file);
			}

			if (files.size() && !windows[window_id].drop_files_callback.is_null()) {
				Variant v = files;
				Variant *vp = &v;
				Variant ret;
				Callable::CallError ce;
				windows[window_id].drop_files_callback.call((const Variant **)&vp, 1, ret, ce);
			}
		} break;
		default: {
			if (user_proc) {
				return CallWindowProcW(user_proc, hWnd, uMsg, wParam, lParam);
			};
		};
	}

	return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
	DisplayServerWindows *ds_win = static_cast<DisplayServerWindows *>(DisplayServer::get_singleton());
	if (ds_win)
		return ds_win->WndProc(hWnd, uMsg, wParam, lParam);
	else
		return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

void DisplayServerWindows::_process_activate_event(WindowID p_window_id, WPARAM wParam, LPARAM lParam) {
	if (LOWORD(wParam) == WA_ACTIVE || LOWORD(wParam) == WA_CLICKACTIVE) {
		_send_window_event(windows[p_window_id], WINDOW_EVENT_FOCUS_IN);
		windows[p_window_id].window_focused = true;
		alt_mem = false;
		control_mem = false;
		shift_mem = false;
	} else { // WM_INACTIVE.
		Input::get_singleton()->release_pressed_events();
		_send_window_event(windows[p_window_id], WINDOW_EVENT_FOCUS_OUT);
		windows[p_window_id].window_focused = false;
		alt_mem = false;
	}

	if ((tablet_get_current_driver() == "wintab") && wintab_available && windows[p_window_id].wtctx) {
		wintab_WTEnable(windows[p_window_id].wtctx, GET_WM_ACTIVATE_STATE(wParam, lParam));
	}
}

void DisplayServerWindows::_process_key_events() {
	for (int i = 0; i < key_event_pos; i++) {
		KeyEvent &ke = key_event_buffer[i];
		switch (ke.uMsg) {
			case WM_CHAR: {
				// Extended keys should only be processed as WM_KEYDOWN message.
				if (!KeyMappingWindows::is_extended_key(ke.wParam) && ((i == 0 && ke.uMsg == WM_CHAR) || (i > 0 && key_event_buffer[i - 1].uMsg == WM_CHAR))) {
					static char32_t prev_wc = 0;
					char32_t unicode = ke.wParam;
					if ((unicode & 0xfffffc00) == 0xd800) {
						if (prev_wc != 0) {
							ERR_PRINT("invalid utf16 surrogate input");
						}
						prev_wc = unicode;
						break; // Skip surrogate.
					} else if ((unicode & 0xfffffc00) == 0xdc00) {
						if (prev_wc == 0) {
							ERR_PRINT("invalid utf16 surrogate input");
							break; // Skip invalid surrogate.
						}
						unicode = (prev_wc << 10UL) + unicode - ((0xd800 << 10UL) + 0xdc00 - 0x10000);
						prev_wc = 0;
					} else {
						prev_wc = 0;
					}
					Ref<InputEventKey> k;
					k.instantiate();

					k->set_window_id(ke.window_id);
					k->set_shift_pressed(ke.shift);
					k->set_alt_pressed(ke.alt);
					k->set_ctrl_pressed(ke.control);
					k->set_meta_pressed(ke.meta);
					k->set_pressed(true);
					k->set_keycode((Key)KeyMappingWindows::get_keysym(ke.wParam));
					k->set_physical_keycode((Key)(KeyMappingWindows::get_scansym((ke.lParam >> 16) & 0xFF, ke.lParam & (1 << 24))));
					k->set_unicode(unicode);
					if (k->get_unicode() && gr_mem) {
						k->set_alt_pressed(false);
						k->set_ctrl_pressed(false);
					}

					if (k->get_unicode() < 32)
						k->set_unicode(0);

					Input::get_singleton()->parse_input_event(k);
				} else {
					// Do nothing.
				}
			} break;
			case WM_KEYUP:
			case WM_KEYDOWN: {
				Ref<InputEventKey> k;
				k.instantiate();

				k->set_window_id(ke.window_id);
				k->set_shift_pressed(ke.shift);
				k->set_alt_pressed(ke.alt);
				k->set_ctrl_pressed(ke.control);
				k->set_meta_pressed(ke.meta);

				k->set_pressed(ke.uMsg == WM_KEYDOWN);

				if ((ke.lParam & (1 << 24)) && (ke.wParam == VK_RETURN)) {
					// Special case for Numpad Enter key.
					k->set_keycode(Key::KP_ENTER);
				} else {
					k->set_keycode((Key)KeyMappingWindows::get_keysym(ke.wParam));
				}

				k->set_physical_keycode((Key)(KeyMappingWindows::get_scansym((ke.lParam >> 16) & 0xFF, ke.lParam & (1 << 24))));

				if (i + 1 < key_event_pos && key_event_buffer[i + 1].uMsg == WM_CHAR) {
					char32_t unicode = key_event_buffer[i + 1].wParam;
					static char32_t prev_wck = 0;
					if ((unicode & 0xfffffc00) == 0xd800) {
						if (prev_wck != 0) {
							ERR_PRINT("invalid utf16 surrogate input");
						}
						prev_wck = unicode;
						break; // Skip surrogate.
					} else if ((unicode & 0xfffffc00) == 0xdc00) {
						if (prev_wck == 0) {
							ERR_PRINT("invalid utf16 surrogate input");
							break; // Skip invalid surrogate.
						}
						unicode = (prev_wck << 10UL) + unicode - ((0xd800 << 10UL) + 0xdc00 - 0x10000);
						prev_wck = 0;
					} else {
						prev_wck = 0;
					}
					k->set_unicode(unicode);
				}
				if (k->get_unicode() && gr_mem) {
					k->set_alt_pressed(false);
					k->set_ctrl_pressed(false);
				}

				if (k->get_unicode() < 32)
					k->set_unicode(0);

				k->set_echo((ke.uMsg == WM_KEYDOWN && (ke.lParam & (1 << 30))));

				Input::get_singleton()->parse_input_event(k);

			} break;
		}
	}

	key_event_pos = 0;
}

void DisplayServerWindows::_update_tablet_ctx(const String &p_old_driver, const String &p_new_driver) {
	for (KeyValue<WindowID, WindowData> &E : windows) {
		WindowData &wd = E.value;
		wd.block_mm = false;
		if ((p_old_driver == "wintab") && wintab_available && wd.wtctx) {
			wintab_WTEnable(wd.wtctx, false);
			wintab_WTClose(wd.wtctx);
			wd.wtctx = 0;
		}
		if ((p_new_driver == "wintab") && wintab_available) {
			wintab_WTInfo(WTI_DEFSYSCTX, 0, &wd.wtlc);
			wd.wtlc.lcOptions |= CXO_MESSAGES;
			wd.wtlc.lcPktData = PK_NORMAL_PRESSURE | PK_TANGENT_PRESSURE | PK_ORIENTATION;
			wd.wtlc.lcMoveMask = PK_NORMAL_PRESSURE | PK_TANGENT_PRESSURE;
			wd.wtlc.lcPktMode = 0;
			wd.wtlc.lcOutOrgX = 0;
			wd.wtlc.lcOutExtX = wd.wtlc.lcInExtX;
			wd.wtlc.lcOutOrgY = 0;
			wd.wtlc.lcOutExtY = -wd.wtlc.lcInExtY;
			wd.wtctx = wintab_WTOpen(wd.hWnd, &wd.wtlc, false);
			if (wd.wtctx) {
				wintab_WTEnable(wd.wtctx, true);
				AXIS pressure;
				if (wintab_WTInfo(WTI_DEVICES + wd.wtlc.lcDevice, DVC_NPRESSURE, &pressure)) {
					wd.min_pressure = int(pressure.axMin);
					wd.max_pressure = int(pressure.axMax);
				}
				AXIS orientation[3];
				if (wintab_WTInfo(WTI_DEVICES + wd.wtlc.lcDevice, DVC_ORIENTATION, &orientation)) {
					wd.tilt_supported = orientation[0].axResolution && orientation[1].axResolution;
				}
				wintab_WTEnable(wd.wtctx, true);
			} else {
				print_verbose("WinTab context creation failed.");
			}
		}
	}
}

DisplayServer::WindowID DisplayServerWindows::_create_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect) {
	DWORD dwExStyle;
	DWORD dwStyle;

	_get_window_style(window_id_counter == MAIN_WINDOW_ID, p_mode == WINDOW_MODE_FULLSCREEN, p_flags & WINDOW_FLAG_BORDERLESS_BIT, !(p_flags & WINDOW_FLAG_RESIZE_DISABLED_BIT), p_mode == WINDOW_MODE_MAXIMIZED, (p_flags & WINDOW_FLAG_NO_FOCUS_BIT), dwStyle, dwExStyle);

	RECT WindowRect;

	WindowRect.left = p_rect.position.x;
	WindowRect.right = p_rect.position.x + p_rect.size.x;
	WindowRect.top = p_rect.position.y;
	WindowRect.bottom = p_rect.position.y + p_rect.size.y;

	if (p_mode == WINDOW_MODE_FULLSCREEN) {
		int nearest_area = 0;
		Rect2i screen_rect;
		for (int i = 0; i < get_screen_count(); i++) {
			Rect2i r;
			r.position = screen_get_position(i);
			r.size = screen_get_size(i);
			Rect2 inters = r.intersection(p_rect);
			int area = inters.size.width * inters.size.height;
			if (area >= nearest_area) {
				screen_rect = r;
				nearest_area = area;
			}
		}

		WindowRect.left = screen_rect.position.x;
		WindowRect.right = screen_rect.position.x + screen_rect.size.x;
		WindowRect.top = screen_rect.position.y;
		WindowRect.bottom = screen_rect.position.y + screen_rect.size.y;
	}

	AdjustWindowRectEx(&WindowRect, dwStyle, FALSE, dwExStyle);

	WindowID id = window_id_counter;
	{
		WindowData &wd = windows[id];

		wd.hWnd = CreateWindowExW(
				dwExStyle,
				L"Engine", L"",
				dwStyle,
				//				(GetSystemMetrics(SM_CXSCREEN) - WindowRect.right) / 2,
				//				(GetSystemMetrics(SM_CYSCREEN) - WindowRect.bottom) / 2,
				WindowRect.left,
				WindowRect.top,
				WindowRect.right - WindowRect.left,
				WindowRect.bottom - WindowRect.top,
				nullptr, nullptr, hInstance, nullptr);
		if (!wd.hWnd) {
			MessageBoxW(nullptr, L"Window Creation Error.", L"ERROR", MB_OK | MB_ICONEXCLAMATION);
			windows.erase(id);
			return INVALID_WINDOW_ID;
		}
		if (p_mode != WINDOW_MODE_FULLSCREEN) {
			wd.pre_fs_valid = true;
		}

#ifdef VULKAN_ENABLED
		if (context_vulkan) {
			if (context_vulkan->window_create(id, p_vsync_mode, wd.hWnd, hInstance, WindowRect.right - WindowRect.left, WindowRect.bottom - WindowRect.top) == -1) {
				memdelete(context_vulkan);
				context_vulkan = nullptr;
				windows.erase(id);
				ERR_FAIL_V_MSG(INVALID_WINDOW_ID, "Failed to create Vulkan Window.");
			}
		}
#endif

#ifdef GLES3_ENABLED
		if (gl_manager) {
			Error err = gl_manager->window_create(id, wd.hWnd, hInstance, WindowRect.right - WindowRect.left, WindowRect.bottom - WindowRect.top);
			ERR_FAIL_COND_V_MSG(err != OK, INVALID_WINDOW_ID, "Failed to create an OpenGL window.");
		}
#endif

		RegisterTouchWindow(wd.hWnd, 0);
		DragAcceptFiles(wd.hWnd, true);

		if ((tablet_get_current_driver() == "wintab") && wintab_available) {
			wintab_WTInfo(WTI_DEFSYSCTX, 0, &wd.wtlc);
			wd.wtlc.lcOptions |= CXO_MESSAGES;
			wd.wtlc.lcPktData = PK_NORMAL_PRESSURE | PK_TANGENT_PRESSURE | PK_ORIENTATION;
			wd.wtlc.lcMoveMask = PK_NORMAL_PRESSURE | PK_TANGENT_PRESSURE;
			wd.wtlc.lcPktMode = 0;
			wd.wtlc.lcOutOrgX = 0;
			wd.wtlc.lcOutExtX = wd.wtlc.lcInExtX;
			wd.wtlc.lcOutOrgY = 0;
			wd.wtlc.lcOutExtY = -wd.wtlc.lcInExtY;
			wd.wtctx = wintab_WTOpen(wd.hWnd, &wd.wtlc, false);
			if (wd.wtctx) {
				wintab_WTEnable(wd.wtctx, true);
				AXIS pressure;
				if (wintab_WTInfo(WTI_DEVICES + wd.wtlc.lcDevice, DVC_NPRESSURE, &pressure)) {
					wd.min_pressure = int(pressure.axMin);
					wd.max_pressure = int(pressure.axMax);
				}
				AXIS orientation[3];
				if (wintab_WTInfo(WTI_DEVICES + wd.wtlc.lcDevice, DVC_ORIENTATION, &orientation)) {
					wd.tilt_supported = orientation[0].axResolution && orientation[1].axResolution;
				}
			} else {
				print_verbose("WinTab context creation failed.");
			}
		} else {
			wd.wtctx = 0;
		}

		wd.last_pressure = 0;
		wd.last_pressure_update = 0;
		wd.last_tilt = Vector2();

		// IME.
		wd.im_himc = ImmGetContext(wd.hWnd);
		ImmReleaseContext(wd.hWnd, wd.im_himc);

		wd.im_position = Vector2();
		wd.last_pos = p_rect.position;
		wd.width = p_rect.size.width;
		wd.height = p_rect.size.height;

		window_id_counter++;
	}

	return id;
}

// WinTab API.
bool DisplayServerWindows::wintab_available = false;
WTOpenPtr DisplayServerWindows::wintab_WTOpen = nullptr;
WTClosePtr DisplayServerWindows::wintab_WTClose = nullptr;
WTInfoPtr DisplayServerWindows::wintab_WTInfo = nullptr;
WTPacketPtr DisplayServerWindows::wintab_WTPacket = nullptr;
WTEnablePtr DisplayServerWindows::wintab_WTEnable = nullptr;

// Windows Ink API.
bool DisplayServerWindows::winink_available = false;
GetPointerTypePtr DisplayServerWindows::win8p_GetPointerType = nullptr;
GetPointerPenInfoPtr DisplayServerWindows::win8p_GetPointerPenInfo = nullptr;

typedef enum _SHC_PROCESS_DPI_AWARENESS {
	SHC_PROCESS_DPI_UNAWARE = 0,
	SHC_PROCESS_SYSTEM_DPI_AWARE = 1,
	SHC_PROCESS_PER_MONITOR_DPI_AWARE = 2
} SHC_PROCESS_DPI_AWARENESS;

int DisplayServerWindows::tablet_get_driver_count() const {
	return tablet_drivers.size();
}

String DisplayServerWindows::tablet_get_driver_name(int p_driver) const {
	if (p_driver < 0 || p_driver >= tablet_drivers.size()) {
		return "";
	} else {
		return tablet_drivers[p_driver];
	}
}

String DisplayServerWindows::tablet_get_current_driver() const {
	return tablet_driver;
}

void DisplayServerWindows::tablet_set_current_driver(const String &p_driver) {
	if (tablet_get_driver_count() == 0) {
		return;
	}
	bool found = false;
	for (int i = 0; i < tablet_get_driver_count(); i++) {
		if (p_driver == tablet_get_driver_name(i)) {
			found = true;
		}
	}
	if (found) {
		_update_tablet_ctx(tablet_driver, p_driver);
		tablet_driver = p_driver;
	} else {
		ERR_PRINT("Unknown tablet driver " + p_driver + ".");
	}
}

DisplayServerWindows::DisplayServerWindows(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	drop_events = false;
	key_event_pos = 0;

	alt_mem = false;
	gr_mem = false;
	shift_mem = false;
	control_mem = false;
	meta_mem = false;
	hInstance = ((OS_Windows *)OS::get_singleton())->get_hinstance();

	pressrc = 0;
	old_invalid = true;
	mouse_mode = MOUSE_MODE_VISIBLE;

	outside = true;

	rendering_driver = p_rendering_driver;

	// Note: Wacom WinTab driver API for pen input, for devices incompatible with Windows Ink.
	HMODULE wintab_lib = LoadLibraryW(L"wintab32.dll");
	if (wintab_lib) {
		wintab_WTOpen = (WTOpenPtr)GetProcAddress(wintab_lib, "WTOpenW");
		wintab_WTClose = (WTClosePtr)GetProcAddress(wintab_lib, "WTClose");
		wintab_WTInfo = (WTInfoPtr)GetProcAddress(wintab_lib, "WTInfoW");
		wintab_WTPacket = (WTPacketPtr)GetProcAddress(wintab_lib, "WTPacket");
		wintab_WTEnable = (WTEnablePtr)GetProcAddress(wintab_lib, "WTEnable");

		wintab_available = wintab_WTOpen && wintab_WTClose && wintab_WTInfo && wintab_WTPacket && wintab_WTEnable;
	}

	if (wintab_available) {
		tablet_drivers.push_back("wintab");
	}

	// Note: Windows Ink API for pen input, available on Windows 8+ only.
	HMODULE user32_lib = LoadLibraryW(L"user32.dll");
	if (user32_lib) {
		win8p_GetPointerType = (GetPointerTypePtr)GetProcAddress(user32_lib, "GetPointerType");
		win8p_GetPointerPenInfo = (GetPointerPenInfoPtr)GetProcAddress(user32_lib, "GetPointerPenInfo");

		winink_available = win8p_GetPointerType && win8p_GetPointerPenInfo;
	}

	if (winink_available) {
		tablet_drivers.push_back("winink");
	}

	if (OS::get_singleton()->is_hidpi_allowed()) {
		HMODULE Shcore = LoadLibraryW(L"Shcore.dll");

		if (Shcore != nullptr) {
			typedef HRESULT(WINAPI * SetProcessDpiAwareness_t)(SHC_PROCESS_DPI_AWARENESS);

			SetProcessDpiAwareness_t SetProcessDpiAwareness = (SetProcessDpiAwareness_t)GetProcAddress(Shcore, "SetProcessDpiAwareness");

			if (SetProcessDpiAwareness) {
				SetProcessDpiAwareness(SHC_PROCESS_SYSTEM_DPI_AWARE);
			}
		}
	}

	memset(&wc, 0, sizeof(WNDCLASSEXW));
	wc.cbSize = sizeof(WNDCLASSEXW);
	wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS;
	wc.lpfnWndProc = (WNDPROC)::WndProc;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInstance ? hInstance : GetModuleHandle(nullptr);
	wc.hIcon = LoadIcon(nullptr, IDI_WINLOGO);
	wc.hCursor = nullptr; //LoadCursor(nullptr, IDC_ARROW);
	wc.hbrBackground = nullptr;
	wc.lpszMenuName = nullptr;
	wc.lpszClassName = L"Engine";

	if (!RegisterClassExW(&wc)) {
		MessageBox(nullptr, "Failed To Register The Window Class.", "ERROR", MB_OK | MB_ICONEXCLAMATION);
		r_error = ERR_UNAVAILABLE;
		return;
	}

	use_raw_input = true;

	RAWINPUTDEVICE Rid[1];

	Rid[0].usUsagePage = 0x01;
	Rid[0].usUsage = 0x02;
	Rid[0].dwFlags = 0;
	Rid[0].hwndTarget = 0;

	if (RegisterRawInputDevices(Rid, 1, sizeof(Rid[0])) == FALSE) {
		// Registration failed.
		use_raw_input = false;
	}

#if defined(VULKAN_ENABLED)
	if (rendering_driver == "vulkan") {
		context_vulkan = memnew(VulkanContextWindows);
		if (context_vulkan->initialize() != OK) {
			memdelete(context_vulkan);
			context_vulkan = nullptr;
			r_error = ERR_UNAVAILABLE;
			return;
		}
	}
#endif
	// Init context and rendering device
#if defined(GLES3_ENABLED)

	if (rendering_driver == "opengl3") {
		GLManager_Windows::ContextType opengl_api_type = GLManager_Windows::GLES_3_0_COMPATIBLE;

		gl_manager = memnew(GLManager_Windows(opengl_api_type));

		if (gl_manager->initialize() != OK) {
			memdelete(gl_manager);
			gl_manager = nullptr;
			r_error = ERR_UNAVAILABLE;
			return;
		}

		//		gl_manager->set_use_vsync(current_videomode.use_vsync);
		RasterizerGLES3::make_current();
	}
#endif

	Point2i window_position(
			(screen_get_size(0).width - p_resolution.width) / 2,
			(screen_get_size(0).height - p_resolution.height) / 2);

	WindowID main_window = _create_window(p_mode, p_vsync_mode, 0, Rect2i(window_position, p_resolution));
	ERR_FAIL_COND_MSG(main_window == INVALID_WINDOW_ID, "Failed to create main window.");

	for (int i = 0; i < WINDOW_FLAG_MAX; i++) {
		if (p_flags & (1 << i)) {
			window_set_flag(WindowFlags(i), true, main_window);
		}
	}

	show_window(MAIN_WINDOW_ID);

#if defined(VULKAN_ENABLED)

	if (rendering_driver == "vulkan") {
		rendering_device_vulkan = memnew(RenderingDeviceVulkan);
		rendering_device_vulkan->initialize(context_vulkan);

		RendererCompositorRD::make_current();
	}
#endif

	//set_ime_active(false);

	if (!OS::get_singleton()->is_in_low_processor_usage_mode()) {
		SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
		DWORD index = 0;
		HANDLE handle = AvSetMmThreadCharacteristics("Games", &index);
		if (handle)
			AvSetMmThreadPriority(handle, AVRT_PRIORITY_CRITICAL);

		// This is needed to make sure that background work does not starve the main thread.
		// This is only setting the priority of this thread, not the whole process.
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
	}

	cursor_shape = CURSOR_ARROW;

	_update_real_mouse_position(MAIN_WINDOW_ID);

	joypad = new JoypadWindows(&windows[MAIN_WINDOW_ID].hWnd);

	r_error = OK;

	((OS_Windows *)OS::get_singleton())->set_main_window(windows[MAIN_WINDOW_ID].hWnd);
	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);
}

Vector<String> DisplayServerWindows::get_rendering_drivers_func() {
	Vector<String> drivers;

#ifdef VULKAN_ENABLED
	drivers.push_back("vulkan");
#endif
#ifdef GLES3_ENABLED
	drivers.push_back("opengl3");
#endif

	return drivers;
}

DisplayServer *DisplayServerWindows::create_func(const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i &p_resolution, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerWindows(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_resolution, r_error));
	if (r_error != OK) {
		OS::get_singleton()->alert("Your video card driver does not support any of the supported Vulkan or OpenGL versions.\n"
								   "Please update your drivers or if you have a very old or integrated GPU upgrade it.",
				"Unable to initialize Video driver");
	}
	return ds;
}

void DisplayServerWindows::register_windows_driver() {
	register_create_function("windows", create_func, get_rendering_drivers_func);
}

DisplayServerWindows::~DisplayServerWindows() {
	delete joypad;
	touch_state.clear();

	cursors_cache.clear();

	if (user_proc) {
		SetWindowLongPtr(windows[MAIN_WINDOW_ID].hWnd, GWLP_WNDPROC, (LONG_PTR)user_proc);
	};

#ifdef GLES3_ENABLED
		// destroy windows .. NYI?
#endif

	if (windows.has(MAIN_WINDOW_ID)) {
#ifdef VULKAN_ENABLED
		if (context_vulkan) {
			context_vulkan->window_destroy(MAIN_WINDOW_ID);
		}
#endif
		if (wintab_available && windows[MAIN_WINDOW_ID].wtctx) {
			wintab_WTClose(windows[MAIN_WINDOW_ID].wtctx);
			windows[MAIN_WINDOW_ID].wtctx = 0;
		}
		DestroyWindow(windows[MAIN_WINDOW_ID].hWnd);
	}

#if defined(VULKAN_ENABLED)
	if (rendering_device_vulkan) {
		rendering_device_vulkan->finalize();
		memdelete(rendering_device_vulkan);
		rendering_device_vulkan = nullptr;
	}

	if (context_vulkan) {
		memdelete(context_vulkan);
		context_vulkan = nullptr;
	}
#endif

	if (restore_mouse_trails > 1) {
		SystemParametersInfoA(SPI_SETMOUSETRAILS, restore_mouse_trails, 0, 0);
	}
#ifdef GLES3_ENABLED
	if (gl_manager) {
		memdelete(gl_manager);
		gl_manager = nullptr;
	}
#endif
}
