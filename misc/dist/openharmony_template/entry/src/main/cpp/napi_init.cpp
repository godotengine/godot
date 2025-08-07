/**************************************************************************/
/*  napi_init.cpp                                                         */
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

#include "bridge_openharmony.h"
#include <hilog/log.h>
#include <napi/native_api.h>
#include <native_window/external_window.h>
#include <rawfile/raw_file_manager.h>
#include <cstdint>
#include <vector>

#undef LOG_DOMAIN
#undef LOG_TAG
#define LOG_DOMAIN 0x3200
#define LOG_TAG "LIB_ENTRY"

static NativeResourceManager *resourceManager = nullptr;
static OHNativeWindow *nativeWindow = nullptr;
static int32_t windowId = -1;
static uint32_t windowWidth = 0;
static uint32_t windowHeight = 0;
static bool initialized = false;

static napi_value NAPI_Global_setResourceManager(napi_env env, napi_callback_info info) {
	size_t argc = 1;
	napi_value args[1] = { nullptr };
	if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "GetContext napi_get_cb_info failed");
		return nullptr;
	}
	resourceManager = OH_ResourceManager_InitNativeResourceManager(env, args[0]);
	return nullptr;
}

static napi_value NAPI_Global_setWindowId(napi_env env, napi_callback_info info) {
	if (windowId != -1) {
		OH_LOG_ERROR(LOG_APP, "Window id already exists");
		return nullptr;
	}
	size_t argc = 1;
	napi_value args[1] = { nullptr };
	if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "GetContext napi_get_cb_info failed");
		return nullptr;
	}
	if (napi_ok != napi_get_value_int32(env, args[0], &windowId)) {
		OH_LOG_ERROR(LOG_APP, "Get window id failed");
		return nullptr;
	}
	return nullptr;
}

static napi_value NAPI_Global_setSurfaceId(napi_env env, napi_callback_info info) {
	if (nativeWindow != nullptr) {
		OH_LOG_ERROR(LOG_APP, "Native window already exists");
		return nullptr;
	}
	size_t argc = 1;
	napi_value args[1] = { nullptr };
	if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "GetContext napi_get_cb_info failed");
		return nullptr;
	}
	int64_t surfaceId = 0;
	bool lossless = true;
	if (napi_ok != napi_get_value_bigint_int64(env, args[0], &surfaceId, &lossless)) {
		OH_LOG_ERROR(LOG_APP, "Get surface id failed");
		return nullptr;
	}
	OH_NativeWindow_CreateNativeWindowFromSurfaceId(surfaceId, &nativeWindow);
	return nullptr;
}

static napi_value NAPI_Global_changeSurface(napi_env env, napi_callback_info info) {
	if (nativeWindow == nullptr) {
		OH_LOG_ERROR(LOG_APP, "Native window does not exists");
		return nullptr;
	}
	size_t argc = 3;
	napi_value args[3] = { nullptr };

	if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "GetContext napi_get_cb_info failed");
		return nullptr;
	}
	int64_t surfaceId = 0;
	bool lossless = true;
	if (napi_ok != napi_get_value_bigint_int64(env, args[0], &surfaceId, &lossless)) {
		OH_LOG_ERROR(LOG_APP, "Get surface id failed");
		return nullptr;
	}
	if (napi_ok != napi_get_value_uint32(env, args[1], &windowWidth)) {
		OH_LOG_ERROR(LOG_APP, "Get width failed");
		return nullptr;
	}
	if (napi_ok != napi_get_value_uint32(env, args[2], &windowHeight)) {
		OH_LOG_ERROR(LOG_APP, "Get height failed");
		return nullptr;
	}
	if (initialized) {
		godot_resize(windowWidth, windowHeight);
	}
	return nullptr;
}

static napi_value NAPI_Global_destroySurface(napi_env env, napi_callback_info info) {
	if (nativeWindow == nullptr) {
		return nullptr;
	}
	OH_NativeWindow_DestroyNativeWindow(nativeWindow);
	nativeWindow = nullptr;
	return nullptr;
}

static napi_value NAPI_Global_setup(napi_env env, napi_callback_info info) {
	size_t argc = 1;
	napi_value args[1] = { nullptr };
	if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "GetContext napi_get_cb_info failed");
		return nullptr;
	}
	char allowed_permissions[2048];
	if (napi_ok != napi_get_value_string_utf8(env, args[0], &allowed_permissions[0], 2048, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "Get array length failed");
		return nullptr;
	}
	godot_init(resourceManager, nativeWindow, windowId, windowWidth, windowHeight, &allowed_permissions[0]);
	initialized = true;
	return nullptr;
}

static napi_value NAPI_Global_inputTouch(napi_env env, napi_callback_info info) {
	if (!initialized) {
		return nullptr;
	}
	size_t argc = 1;
	napi_value args[1] = { nullptr };
	if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "GetContext napi_get_cb_info failed");
		return nullptr;
	}
	uint32_t array_length = 0;
	if (napi_ok != napi_get_array_length(env, args[0], &array_length)) {
		OH_LOG_ERROR(LOG_APP, "Get array length failed");
		return nullptr;
	}

	std::vector<GodotTouchEvent> events;

	for (int i = 0; i < array_length; i++) {
		napi_value element;
		if (napi_ok != napi_get_element(env, args[0], i, &element)) {
			OH_LOG_ERROR(LOG_APP, "Get array element failed");
			return nullptr;
		}

		napi_value event_type;
		if (napi_ok != napi_get_named_property(env, element, "type", &event_type)) {
			OH_LOG_ERROR(LOG_APP, "Get event type failed");
			return nullptr;
		}

		int32_t event_type_int;
		if (napi_ok != napi_get_value_int32(env, event_type, &event_type_int)) {
			OH_LOG_ERROR(LOG_APP, "Get event type int failed");
			return nullptr;
		}

		napi_value event_id;
		if (napi_ok != napi_get_named_property(env, element, "id", &event_id)) {
			OH_LOG_ERROR(LOG_APP, "Get event id failed");
			return nullptr;
		}

		int32_t event_id_int;
		if (napi_ok != napi_get_value_int32(env, event_id, &event_id_int)) {
			OH_LOG_ERROR(LOG_APP, "Get event id int failed");
			return nullptr;
		}

		napi_value event_x;
		if (napi_ok != napi_get_named_property(env, element, "x", &event_x)) {
			OH_LOG_ERROR(LOG_APP, "Get event x failed");
			return nullptr;
		}

		double event_x_double;
		if (napi_ok != napi_get_value_double(env, event_x, &event_x_double)) {
			OH_LOG_ERROR(LOG_APP, "Get event x double failed");
			return nullptr;
		}

		napi_value event_y;
		if (napi_ok != napi_get_named_property(env, element, "y", &event_y)) {
			OH_LOG_ERROR(LOG_APP, "Get event y failed");
			return nullptr;
		}

		double event_y_double;
		if (napi_ok != napi_get_value_double(env, event_y, &event_y_double)) {
			OH_LOG_ERROR(LOG_APP, "Get event y double failed");
			return nullptr;
		}

		GodotTouchEvent event;
		event.type = event_type_int;
		event.id = event_id_int;
		event.x = event_x_double;
		event.y = event_y_double;
		events.push_back(event);
	}
	godot_touch(&events[0], events.size());
	return nullptr;
}

static napi_value NAPI_Global_sendWindowEvent(napi_env env, napi_callback_info info) {
	if (!initialized) {
		return nullptr;
	}

	size_t argc = 1;
	napi_value args[1] = { nullptr };

	if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "GetContext napi_get_cb_info failed");
		return nullptr;
	}
	int32_t event = 0;
	if (napi_ok != napi_get_value_int32(env, args[0], &event)) {
		OH_LOG_ERROR(LOG_APP, "Get event id failed");
		return nullptr;
	}
	godot_window_event(event);
	return nullptr;
}

static napi_value NAPI_Global_inputKey(napi_env env, napi_callback_info info) {
	if (!initialized) {
		return nullptr;
	}
	size_t argc = 1;
	napi_value args[1] = { nullptr };
	if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "GetContext napi_get_cb_info failed");
		return nullptr;
	}

	napi_value element = args[0];

	napi_value code;
	if (napi_ok != napi_get_named_property(env, element, "code", &code)) {
		OH_LOG_ERROR(LOG_APP, "Get code failed");
		return nullptr;
	}

	uint32_t code_uint;
	if (napi_ok != napi_get_value_uint32(env, code, &code_uint)) {
		OH_LOG_ERROR(LOG_APP, "Get code int failed");
		return nullptr;
	}

	napi_value unicode;
	if (napi_ok != napi_get_named_property(env, element, "unicode", &unicode)) {
		OH_LOG_ERROR(LOG_APP, "Get unicode failed");
		return nullptr;
	}

	uint32_t unicode_uint;
	if (napi_ok != napi_get_value_uint32(env, unicode, &unicode_uint)) {
		OH_LOG_ERROR(LOG_APP, "Get unicode int failed");
		return nullptr;
	}

	napi_value pressed;
	if (napi_ok != napi_get_named_property(env, element, "pressed", &pressed)) {
		OH_LOG_ERROR(LOG_APP, "Get pressed failed");
		return nullptr;
	}

	bool pressed_bool;
	if (napi_ok != napi_get_value_bool(env, pressed, &pressed_bool)) {
		OH_LOG_ERROR(LOG_APP, "Get pressed bool failed");
		return nullptr;
	}

	napi_value alt;
	if (napi_ok != napi_get_named_property(env, element, "alt", &alt)) {
		OH_LOG_ERROR(LOG_APP, "Get alt failed");
		return nullptr;
	}

	bool alt_bool;
	if (napi_ok != napi_get_value_bool(env, alt, &alt_bool)) {
		OH_LOG_ERROR(LOG_APP, "Get alt bool failed");
		return nullptr;
	}

	napi_value ctrl;
	if (napi_ok != napi_get_named_property(env, element, "ctrl", &ctrl)) {
		OH_LOG_ERROR(LOG_APP, "Get ctrl failed");
		return nullptr;
	}

	bool ctrl_bool;
	if (napi_ok != napi_get_value_bool(env, ctrl, &ctrl_bool)) {
		OH_LOG_ERROR(LOG_APP, "Get ctrl bool failed");
		return nullptr;
	}

	napi_value shift;
	if (napi_ok != napi_get_named_property(env, element, "shift", &shift)) {
		OH_LOG_ERROR(LOG_APP, "Get shift failed");
		return nullptr;
	}

	bool shift_bool;
	if (napi_ok != napi_get_value_bool(env, shift, &shift_bool)) {
		OH_LOG_ERROR(LOG_APP, "Get shift bool failed");
		return nullptr;
	}

	napi_value meta;
	if (napi_ok != napi_get_named_property(env, element, "meta", &meta)) {
		OH_LOG_ERROR(LOG_APP, "Get meta failed");
		return nullptr;
	}

	bool meta_bool;
	if (napi_ok != napi_get_value_bool(env, meta, &meta_bool)) {
		OH_LOG_ERROR(LOG_APP, "Get meta bool failed");
		return nullptr;
	}

	GodotKeyEvent event;
	event.code = code_uint;
	event.unicode = unicode_uint;
	event.pressed = pressed_bool;
	event.alt = alt_bool;
	event.ctrl = ctrl_bool;
	event.shift = shift_bool;
	event.meta = meta_bool;
	godot_key(&event);
	return nullptr;
}

static napi_value NAPI_Global_inputMouse(napi_env env, napi_callback_info info) {
	if (!initialized) {
		return nullptr;
	}
	size_t argc = 1;
	napi_value args[1] = { nullptr };
	if (napi_ok != napi_get_cb_info(env, info, &argc, args, nullptr, nullptr)) {
		OH_LOG_ERROR(LOG_APP, "GetContext napi_get_cb_info failed");
		return nullptr;
	}

	napi_value element = args[0];

	napi_value type;
	if (napi_ok != napi_get_named_property(env, element, "type", &type)) {
		OH_LOG_ERROR(LOG_APP, "Get type failed");
		return nullptr;
	}

	uint32_t type_uint;
	if (napi_ok != napi_get_value_uint32(env, type, &type_uint)) {
		OH_LOG_ERROR(LOG_APP, "Get type int failed");
		return nullptr;
	}

	napi_value button;
	if (napi_ok != napi_get_named_property(env, element, "button", &button)) {
		OH_LOG_ERROR(LOG_APP, "Get button failed");
		return nullptr;
	}

	uint32_t button_uint;
	if (napi_ok != napi_get_value_uint32(env, button, &button_uint)) {
		OH_LOG_ERROR(LOG_APP, "Get button int failed");
		return nullptr;
	}

	napi_value mask;
	if (napi_ok != napi_get_named_property(env, element, "mask", &mask)) {
		OH_LOG_ERROR(LOG_APP, "Get mask failed");
		return nullptr;
	}

	uint32_t mask_uint;
	if (napi_ok != napi_get_value_uint32(env, mask, &mask_uint)) {
		OH_LOG_ERROR(LOG_APP, "Get mask int failed");
		return nullptr;
	}

	napi_value x;
	if (napi_ok != napi_get_named_property(env, element, "x", &x)) {
		OH_LOG_ERROR(LOG_APP, "Get x failed");
		return nullptr;
	}

	double x_double;
	if (napi_ok != napi_get_value_double(env, x, &x_double)) {
		OH_LOG_ERROR(LOG_APP, "Get x double failed");
		return nullptr;
	}

	napi_value y;
	if (napi_ok != napi_get_named_property(env, element, "y", &y)) {
		OH_LOG_ERROR(LOG_APP, "Get y failed");
		return nullptr;
	}

	double y_double;
	if (napi_ok != napi_get_value_double(env, y, &y_double)) {
		OH_LOG_ERROR(LOG_APP, "Get y double failed");
		return nullptr;
	}

	GodotMouseEvent event;
	event.type = type_uint;
	event.button = button_uint;
	event.mask = mask_uint;
	event.x = x_double;
	event.y = y_double;
	godot_mouse(&event);
	return nullptr;
}

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
	napi_property_descriptor desc[] = {
		{ "setResourceManager", nullptr, NAPI_Global_setResourceManager, nullptr, nullptr, nullptr, napi_default,
				nullptr },
		{ "setSurfaceId", nullptr, NAPI_Global_setSurfaceId, nullptr, nullptr, nullptr, napi_default, nullptr },
		{ "changeSurface", nullptr, NAPI_Global_changeSurface, nullptr, nullptr, nullptr, napi_default, nullptr },
		{ "destroySurface", nullptr, NAPI_Global_destroySurface, nullptr, nullptr, nullptr, napi_default, nullptr },
		{ "setup", nullptr, NAPI_Global_setup, nullptr, nullptr, nullptr, napi_default, nullptr },
		{ "inputTouch", nullptr, NAPI_Global_inputTouch, nullptr, nullptr, nullptr, napi_default, nullptr },
		{ "setWindowId", nullptr, NAPI_Global_setWindowId, nullptr, nullptr, nullptr, napi_default, nullptr },
		{ "sendWindowEvent", nullptr, NAPI_Global_sendWindowEvent, nullptr, nullptr, nullptr, napi_default, nullptr },
		{ "inputKey", nullptr, NAPI_Global_inputKey, nullptr, nullptr, nullptr, napi_default, nullptr },
		{ "inputMouse", nullptr, NAPI_Global_inputMouse, nullptr, nullptr, nullptr, napi_default, nullptr }
	};
	napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
	return exports;
}
EXTERN_C_END

static napi_module demoModule = {
	.nm_version = 1,
	.nm_flags = 0,
	.nm_filename = nullptr,
	.nm_register_func = Init,
	.nm_modname = "entry",
	.nm_priv = ((void *)0),
	.reserved = { 0 },
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void) {
	napi_module_register(&demoModule);
}
