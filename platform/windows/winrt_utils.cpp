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

typedef HRESULT(WINAPI *CreateDispatcherQueueControllerPtr)(DispatcherQueueOptions options, void *dispatcherQueueController);
CreateDispatcherQueueControllerPtr GD_CreateDispatcherQueueController;

typedef HRESULT(WINAPI *RoGetActivationFactoryPtr)(void *classId, REFIID iid, void **factory);
RoGetActivationFactoryPtr GD_RoGetActivationFactory;

typedef HRESULT(WINAPI *CoIncrementMTAUsagePtr)(void **pCookie);
CoIncrementMTAUsagePtr GD_CoIncrementMTAUsage;

typedef HRESULT(WINAPI *CoDecrementMTAUsagePtr)(void *pCookie);
CoDecrementMTAUsagePtr GD_CoDecrementMTAUsage;

static const WCHAR *RODisplayInformationName = L"Windows.Graphics.Display.DisplayInformation";
static const WCHAR *ROGlobalizationPreferencesStaticsName = L"Windows.System.UserProfile.GlobalizationPreferences";
static const WCHAR *ROCoreInputViewName = L"Windows.UI.ViewManagement.Core.CoreInputView";
static const WCHAR *ROApiInformationName = L"Windows.Foundation.Metadata.ApiInformation";

GODOT_GCC_WARNING_PUSH
GODOT_GCC_WARNING_IGNORE("-Wnon-virtual-dtor")
GODOT_GCC_WARNING_IGNORE("-Wctor-dtor-privacy")
GODOT_GCC_WARNING_IGNORE("-Wshadow")
GODOT_GCC_WARNING_IGNORE("-Wstrict-aliasing")
GODOT_CLANG_WARNING_PUSH
GODOT_CLANG_WARNING_IGNORE("-Wnon-virtual-dtor")

/**************************************************************************/
/* DisplayInformation5 events                                             */
/**************************************************************************/

class GodotAdvancedColorInfoChangedEventHandler : public ROTypedEventHandler {
	TYPED_EVENT_HANDLER_CLASS(GodotAdvancedColorInfoChangedEventHandler, ROTypedEventHandler)

private:
	int64_t id = 0;
	Callable cb;

public:
	HRESULT STDMETHODCALLTYPE Invoke(void *p_sender, IInspectable *p_args) {
		cb.call_deferred(id);
		return S_OK;
	}

	GodotAdvancedColorInfoChangedEventHandler(int64_t p_id, const Callable &p_cb) {
		id = p_id;
		cb = p_cb;
	}
};

GODOT_GCC_WARNING_POP
GODOT_CLANG_WARNING_POP

/**************************************************************************/
/* WinRTUtils                                                             */
/**************************************************************************/

ComPtr<RODispatcherQueueController> queue_ctrl;
void *mta_cookie = nullptr;
bool api_initialized = false;

bool WinRTUtils::is_api_contract_present(const String &p_contract, uint16_t p_version) {
	ComPtr<ROApiInformationStatics> api_info;
	HRESULT res = activation_factory(ROApiInformationName, IID_PPV_ARGS(&api_info));
	if (FAILED(res)) {
		return false;
	}
	Ref<HStringWrapper> name;
	name.instantiate();
	name->set_string(p_contract);

	bool ret = false;
	ERR_FAIL_COND_V(FAILED(api_info->IsApiContractPresentByMajor(name->get_ptr(), p_version, &ret)), false);
	return ret;
}

bool WinRTUtils::is_type_present(const String &p_type) {
	ComPtr<ROApiInformationStatics> api_info;
	HRESULT res = activation_factory(ROApiInformationName, IID_PPV_ARGS(&api_info));
	if (FAILED(res)) {
		return false;
	}
	Ref<HStringWrapper> name;
	name.instantiate();
	name->set_string(p_type);

	bool ret = false;
	ERR_FAIL_COND_V(FAILED(api_info->IsTypePresent(name->get_ptr(), &ret)), false);
	return ret;
}

HRESULT WinRTUtils::activation_factory(const String &p_class_name, REFIID p_iid, void **p_factory) {
	Ref<HStringWrapper> name;
	name.instantiate();
	name->set_string(p_class_name);

	HRESULT res = GD_RoGetActivationFactory(name->get_ptr(), p_iid, p_factory);
	if (res == (HRESULT)0x800401F0 && !mta_cookie) {
		ERR_FAIL_COND_V(FAILED(GD_CoIncrementMTAUsage(&mta_cookie)), E_ABORT);
		res = GD_RoGetActivationFactory(name->get_ptr(), p_iid, p_factory);
	}
	if (res == (HRESULT)0x80004001 || res == (HRESULT)0x80004002) {
		print_verbose(vformat("RoGetActivationFactory(%s) not supported.", p_class_name));
		return res;
	}
	ERR_FAIL_COND_V_MSG(FAILED(res), res, vformat("RoGetActivationFactory(%s) failed with error 0x%08ux.", p_class_name, (uint64_t)res));

	return res;
}

class WinRTWindowData {
	friend class WinRTUtils;

	bool has_disp_info = false;
	ComPtr<RODisplayInformation5> disp_info;
	ComPtr<ROTypedEventHandler> handler;
	ROEventToken token;
};

bool WinRTUtils::is_initialized() {
	return api_initialized;
}

void WinRTUtils::init() {
	HMODULE combase = LoadLibraryW(L"combase.dll");
	if (!combase) {
		return;
	}
	GD_RoGetActivationFactory = (RoGetActivationFactoryPtr)(void *)GetProcAddress(combase, "RoGetActivationFactory");
	if (!GD_RoGetActivationFactory) {
		return;
	}
	GD_CoIncrementMTAUsage = (CoIncrementMTAUsagePtr)(void *)GetProcAddress(combase, "CoIncrementMTAUsage");
	if (!GD_CoIncrementMTAUsage) {
		return;
	}
	GD_CoDecrementMTAUsage = (CoDecrementMTAUsagePtr)(void *)GetProcAddress(combase, "CoDecrementMTAUsage");
	if (!GD_CoDecrementMTAUsage) {
		return;
	}
	api_initialized = true;
}

bool WinRTUtils::create_queue() {
	ERR_FAIL_COND_V(!api_initialized, false);

	HMODULE coremessaging = LoadLibraryW(L"coremessaging.dll");
	if (!coremessaging) {
		return false;
	}
	GD_CreateDispatcherQueueController = (CreateDispatcherQueueControllerPtr)(void *)GetProcAddress(coremessaging, "CreateDispatcherQueueController");
	if (!GD_CreateDispatcherQueueController) {
		return false;
	}

	DispatcherQueueOptions options{ sizeof(options), DQTYPE_THREAD_CURRENT, DQTAT_COM_NONE };
	HRESULT res = GD_CreateDispatcherQueueController(options, queue_ctrl.GetAddressOf());
	ERR_FAIL_COND_V_MSG(FAILED(res), false, "CreateDispatcherQueueController failed with error " + vformat("0x%08ux", (uint64_t)res) + ".");

	return true;
}

void WinRTUtils::destroy_queue() {
	ERR_FAIL_COND(!api_initialized);

	ComPtr<ROAsyncAction> action;
	ERR_FAIL_COND(FAILED(queue_ctrl->ShutdownQueueAsync((void **)action.GetAddressOf())));

	ComPtr<ROAsyncInfo> info;
	ERR_FAIL_COND(FAILED(action.As(&info)));

	ROAsyncStatus status = ROAsyncStatus::Started;
	while (status == ROAsyncStatus::Started) {
		info->get_Status((int32_t *)&status);
		MSG msg = {};
		while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessageW(&msg);
		}
	}
	ERR_FAIL_COND_MSG(status == ROAsyncStatus::Error, "DispatcherQueueController shutdown failed.");
}

void WinRTUtils::cleanup() {
	if (mta_cookie) {
		GD_CoDecrementMTAUsage(mta_cookie);
		mta_cookie = nullptr;
	}
}

Vector<String> WinRTUtils::get_preferred_locales() {
	Vector<String> out;
	ERR_FAIL_COND_V(!api_initialized, out);

	ComPtr<ROGlobalizationPreferencesStatics> glob_prefs;
	if (FAILED(activation_factory(ROGlobalizationPreferencesStaticsName, IID_PPV_ARGS(&glob_prefs)))) {
		return out;
	}

	ComPtr<ROVectorView_HSTRING> languages;
	HRESULT res = glob_prefs->get_Languages((void **)languages.GetAddressOf());
	ERR_FAIL_COND_V_MSG(FAILED(res), out, "GlobalizationPreferencesStatics::get_Languages failed with error " + vformat("0x%08ux", (uint64_t)res) + ".");

	uint32_t size = 0;
	res = languages->get_Size(&size);
	ERR_FAIL_COND_V_MSG(FAILED(res), out, "VectorView<HSTRING>::get_Size failed with error " + vformat("0x%08ux", (uint64_t)res) + ".");

	for (uint32_t i = 0; i < size; i++) {
		Ref<HStringWrapper> lang;
		lang.instantiate();
		ERR_CONTINUE(FAILED(languages->GetAt(i, lang->get_ptrw())));

		String lang_str = lang->get_string();
		if (!lang_str.is_empty()) {
			out.push_back(lang_str.replace_char('-', '_'));
		}
	}

	return out;
}

bool WinRTUtils::try_show_onecore_emoji_picker() {
	ERR_FAIL_COND_V(!api_initialized, false);

	if (!WinRTUtils::is_api_contract_present(L"Windows.Foundation.UniversalApiContract", 7)) {
		return false;
	}

	ComPtr<ROCoreInputViewStatics> core_view;
	if (FAILED(activation_factory(ROCoreInputViewName, IID_PPV_ARGS(&core_view)))) {
		return false;
	}

	ComPtr<IInspectable> core_input_view_ii;
	ComPtr<ROCoreInputView3> core_input_view;
	ERR_FAIL_COND_V(FAILED(core_view->GetForCurrentView((void **)core_input_view_ii.GetAddressOf())), false);
	ERR_FAIL_COND_V(FAILED(core_input_view_ii.As(&core_input_view)), false);

	bool status = false;
	ERR_FAIL_COND_V(FAILED(core_input_view->TryShowWithKind((int32_t)ROCoreInputViewKind::Emoji, &status)), false);
	return status;
}

bool WinRTUtils::window_has_display_info(const WinRTWindowData *p_data) {
	ERR_FAIL_COND_V(!api_initialized, false);

	if (p_data) {
		return p_data->has_disp_info;
	} else {
		return false;
	}
}

void WinRTUtils::window_get_advanced_color_info(const WinRTWindowData *p_data, bool &r_hdr_supported, float &r_min_luminance, float &r_max_luminance, float &r_max_average_luminance, float &r_sdr_white_level) {
	ERR_FAIL_COND(!api_initialized);

	if (p_data && p_data->has_disp_info) {
		ComPtr<ROAdvancedColorInfo> adv_info;

		ERR_FAIL_COND(FAILED(p_data->disp_info->GetAdvancedColorInfo((void **)adv_info.GetAddressOf())));

		ROAdvancedColorKind color_type = ROAdvancedColorKind::StandardDynamicRange;
		ERR_FAIL_COND(FAILED(adv_info->get_CurrentAdvancedColorKind((int32_t *)&color_type)));
		r_hdr_supported = (color_type == ROAdvancedColorKind::HighDynamicRange);

		ERR_FAIL_COND(FAILED(adv_info->get_MinLuminanceInNits(&r_min_luminance)));
		ERR_FAIL_COND(FAILED(adv_info->get_MaxLuminanceInNits(&r_max_luminance)));
		ERR_FAIL_COND(FAILED(adv_info->get_MaxAverageFullFrameLuminanceInNits(&r_max_average_luminance)));
		ERR_FAIL_COND(FAILED(adv_info->get_SdrWhiteLevelInNits(&r_sdr_white_level)));
	}
}

WinRTWindowData *WinRTUtils::create_wd(HWND p_window, const Callable &p_color_cb, int64_t p_window_id) {
	WinRTWindowData *wd = memnew(WinRTWindowData);

	ERR_FAIL_COND_V(!api_initialized, wd);
	if (WinRTUtils::is_api_contract_present(L"Windows.Foundation.UniversalApiContract", 6)) {
		ComPtr<RODisplayInformationStaticsInterop> interop;
		if (FAILED(activation_factory(RODisplayInformationName, IID_PPV_ARGS(&interop)))) {
			return wd;
		}

		ERR_FAIL_COND_V(FAILED(interop->GetForWindow(p_window, IID_PPV_ARGS(&wd->disp_info))), wd);

		GodotAdvancedColorInfoChangedEventHandler *handler = new GodotAdvancedColorInfoChangedEventHandler(p_window_id, p_color_cb);
		ERR_FAIL_COND_V(FAILED(handler->QueryInterface(IID_PPV_ARGS(&wd->handler))), wd);
		ERR_FAIL_COND_V(FAILED(wd->disp_info->add_AdvancedColorInfoChanged(wd->handler.Get(), &wd->token)), wd);

		wd->has_disp_info = true;
	}
	return wd;
}

void WinRTUtils::destroy_wd(WinRTWindowData *p_data) {
	ERR_FAIL_COND(!api_initialized);
	if (p_data) {
		if (p_data->token) {
			p_data->disp_info->remove_AdvancedColorInfoChanged(p_data->token);
		}
		memdelete(p_data);
	}
}
