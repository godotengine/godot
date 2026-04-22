/**************************************************************************/
/*  winrt_utils.cpp                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "winrt_utils.h"

#include "core/typedefs.h"

#ifdef WINRT_ENABLED

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wnon-virtual-dtor")
GODOT_GCC_WARNING_IGNORE("-Wctor-dtor-privacy")
GODOT_GCC_WARNING_IGNORE("-Wshadow")
GODOT_GCC_WARNING_IGNORE("-Wstrict-aliasing")
GODOT_CLANG_WARNING_PUSH
GODOT_CLANG_WARNING_IGNORE("-Wnon-virtual-dtor")

#include <inspectable.h>
#include <winrt/Windows.Foundation.Metadata.h>
#include <winrt/Windows.Graphics.Display.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.UI.Input.h>
#include <winrt/Windows.UI.ViewManagement.Core.h>

enum DISPATCHERQUEUE_THREAD_APARTMENTTYPE {
	DQTAT_COM_NONE = 0,
	DQTAT_COM_ASTA = 1,
	DQTAT_COM_STA = 2
};

enum DISPATCHERQUEUE_THREAD_TYPE {
	DQTYPE_THREAD_DEDICATED = 1,
	DQTYPE_THREAD_CURRENT = 2,
};

struct DispatcherQueueOptions {
	DWORD dwSize;
	DISPATCHERQUEUE_THREAD_TYPE threadType;
	DISPATCHERQUEUE_THREAD_APARTMENTTYPE apartmentType;
};

extern "C" HRESULT __declspec(dllexport) WINAPI CreateDispatcherQueueController(DispatcherQueueOptions options, void *dispatcherQueueController);

#ifndef E_BOUNDS
#define E_BOUNDS _HRESULT_TYPEDEF_(0x8000000B)
#endif // E_BOUNDS

#if defined __MINGW32__ || defined __MINGW64__

#ifndef __IDisplayInformationStaticsInterop_INTERFACE_DEFINED__
#define __IDisplayInformationStaticsInterop_INTERFACE_DEFINED__

// clang-format off
MIDL_INTERFACE("7449121c-382b-4705-8da7-a795ba482013")
IDisplayInformationStaticsInterop : public IInspectable {
public:
	virtual HRESULT STDMETHODCALLTYPE GetForWindow(
			HWND window,
			REFIID riid,
			void **displayInfo) = 0;

	virtual HRESULT STDMETHODCALLTYPE GetForMonitor(
			HMONITOR monitor,
			REFIID riid,
			void **displayInfo) = 0;
};
// clang-format on
#ifdef __CRT_UUID_DECL
__CRT_UUID_DECL(IDisplayInformationStaticsInterop, 0x7449121c, 0x382b, 0x4705, 0x8d, 0xa7, 0xa7, 0x95, 0xba, 0x48, 0x20, 0x13)
#endif // __CRT_UUID_DECL

#endif // __IDisplayInformationStaticsInterop_INTERFACE_DEFINED__

#else // defined __MINGW32__ || defined __MINGW64__

#include <windows.graphics.display.interop.h>

#endif // defined __MINGW32__ || defined __MINGW64__

GODOT_GCC_WARNING_POP
GODOT_CLANG_WARNING_POP

using namespace winrt::Windows::Graphics::Display;
using namespace winrt::Windows::System;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Foundation::Metadata;
using namespace winrt::Windows::UI::ViewManagement::Core;

DispatcherQueueController controller{ nullptr };

class WinRTWindowData {
	friend class WinRTUtils;

	int64_t id = 0;
	bool has_disp_info = false;
	DisplayInformation disp_info{ nullptr };
	Callable cb;
	winrt::event_token token{};
};

bool WinRTUtils::create_queue() {
	DispatcherQueueOptions options{ sizeof(options), DQTYPE_THREAD_CURRENT, DQTAT_COM_NONE };
	HRESULT res = CreateDispatcherQueueController(options, winrt::put_abi(controller));
	return SUCCEEDED(res);
}

void WinRTUtils::destroy_queue() {
	IAsyncAction action = controller.ShutdownQueueAsync();
	while (action.Status() == AsyncStatus::Started) {
		MSG msg = {};
		while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessageW(&msg);
		}
	}
	ERR_FAIL_COND_MSG(action.Status() == AsyncStatus::Error, "DispatcherQueueController shutdown failed.");
}

bool WinRTUtils::try_show_onecore_emoji_picker() {
	if (ApiInformation::IsApiContractPresent(L"Windows.Foundation.UniversalApiContract", 7)) { // Windows 10, 1809+
		return CoreInputView::GetForCurrentView().TryShow(CoreInputViewKind::Emoji);
	}
	return false;
}

bool WinRTUtils::window_has_display_info(const WinRTWindowData *p_data) {
	if (p_data) {
		return p_data->has_disp_info;
	} else {
		return false;
	}
}

void WinRTUtils::window_get_advanced_color_info(const WinRTWindowData *p_data, bool &r_hdr_supported, float &r_min_luminance, float &r_max_luminance, float &r_max_average_luminance, float &r_sdr_white_level) {
	if (p_data && p_data->has_disp_info) {
		Dictionary info;

		AdvancedColorInfo adv_info = p_data->disp_info.GetAdvancedColorInfo();

		r_hdr_supported = (adv_info.CurrentAdvancedColorKind() == AdvancedColorKind::HighDynamicRange);
		r_min_luminance = adv_info.MinLuminanceInNits();
		r_max_luminance = adv_info.MaxLuminanceInNits();
		r_max_average_luminance = adv_info.MaxAverageFullFrameLuminanceInNits();
		r_sdr_white_level = adv_info.SdrWhiteLevelInNits();
	}
}

WinRTWindowData *WinRTUtils::create_wd(HWND p_window, const Callable &p_color_cb, int64_t p_window_id) {
	WinRTWindowData *wd = memnew(WinRTWindowData);

	wd->id = p_window_id;
	wd->cb = p_color_cb;
	if (ApiInformation::IsApiContractPresent(L"Windows.Foundation.UniversalApiContract", 6)) {
		try {
			HRESULT res = winrt::get_activation_factory<DisplayInformation, IDisplayInformationStaticsInterop>()->GetForWindow(p_window, winrt::guid_of<DisplayInformation>(), winrt::put_abi(wd->disp_info));
			if (res == S_OK && wd->disp_info) {
				wd->has_disp_info = true;
				wd->token = wd->disp_info.AdvancedColorInfoChanged([wd](const DisplayInformation &p_sender, auto &&) {
					wd->cb.call_deferred(wd->id);
				});
			}
		} catch (...) {
			memdelete(wd);
			return nullptr;
		}
	}

	return wd;
}

void WinRTUtils::destroy_wd(WinRTWindowData *p_data) {
	if (p_data) {
		if (p_data->token) {
			p_data->disp_info.AdvancedColorInfoChanged(p_data->token);
		}
		p_data->disp_info = nullptr;

		memdelete(p_data);
	}
}

#else

bool WinRTUtils::try_show_onecore_emoji_picker() {
	return false;
}

bool WinRTUtils::create_queue() {
	return false;
}

void WinRTUtils::destroy_queue() {}

bool WinRTUtils::window_has_display_info(const WinRTWindowData *p_data) {
	return false;
}

void WinRTUtils::window_get_advanced_color_info(const WinRTWindowData *p_data, bool &r_hdr_supported, float &r_min_luminance, float &r_max_luminance, float &r_max_average_luminance, float &r_sdr_white_level) {}

WinRTWindowData *WinRTUtils::create_wd(HWND p_window, const Callable &p_color_cb, int64_t p_window_id) {
	return nullptr;
}

void WinRTUtils::destroy_wd(WinRTWindowData *p_data) {}

#endif
