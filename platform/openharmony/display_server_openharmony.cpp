/**************************************************************************/
/*  display_server_openharmony.cpp                                        */
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

#include "display_server_openharmony.h"

#include "os_openharmony.h"
#include "rendering_context_driver_vulkan_openharmony.h"
#include "wrapper_openharmony.h"

#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/rendering_device.h"

#include <database/pasteboard/oh_pasteboard.h>
#include <database/udmf/udmf.h>
#include <database/udmf/uds.h>

void DisplayServerOpenHarmony::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	get_singleton()->send_input_event(p_event);
}

DisplayServerOpenHarmony *DisplayServerOpenHarmony::get_singleton() {
	return static_cast<DisplayServerOpenHarmony *>(DisplayServer::get_singleton());
}

Vector<String> DisplayServerOpenHarmony::get_rendering_drivers_func() {
	Vector<String> drivers;
	drivers.push_back("vulkan");
	return drivers;
}

DisplayServer *DisplayServerOpenHarmony::create_func(const String &p_rendering_driver, DisplayServer::WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error) {
	DisplayServer *ds = memnew(DisplayServerOpenHarmony(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, p_parent_window, r_error));
	if (r_error != OK) {
		OS::get_singleton()->alert(
				"Your device seems not to support the required Vulkan version.\n\n"
				"Unable to initialize Vulkan video driver.");
	}
	return ds;
}

void DisplayServerOpenHarmony::register_openharmony_driver() {
	register_create_function("openharmony", create_func, get_rendering_drivers_func);
}

DisplayServerOpenHarmony::DisplayServerOpenHarmony(const String &p_rendering_driver, WindowMode p_mode, DisplayServer::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error) {
	rendering_driver = p_rendering_driver;

	rendering_context = nullptr;
	rendering_device = nullptr;

	if (rendering_driver != "vulkan") {
		ERR_PRINT(vformat("Failed to create %s context.", rendering_driver));
		r_error = ERR_UNAVAILABLE;
	}

	rendering_context = memnew(RenderingContextDriverVulkanOpenHarmony);

	if (rendering_context->initialize() != OK) {
		memdelete(rendering_context);
		rendering_context = nullptr;
		ERR_PRINT(vformat("Failed to initialize %s context.", rendering_driver));
		r_error = ERR_UNAVAILABLE;
		return;
	}
	RenderingContextDriverVulkanOpenHarmony::WindowPlatformData vulkan;
	OHNativeWindow *native_window = OS_OpenHarmony::get_singleton()->get_native_window();
	ERR_FAIL_NULL(native_window);
	vulkan.window = native_window;

	if (rendering_context->window_create(MAIN_WINDOW_ID, &vulkan) != OK) {
		ERR_PRINT(vformat("Failed to create %s window.", rendering_driver));
		memdelete(rendering_context);
		rendering_context = nullptr;
		r_error = ERR_UNAVAILABLE;
		return;
	}

	Size2i display_size = OS_OpenHarmony::get_singleton()->get_display_size();
	rendering_context->window_set_size(MAIN_WINDOW_ID, display_size.width, display_size.height);
	rendering_context->window_set_vsync_mode(MAIN_WINDOW_ID, p_vsync_mode);

	rendering_device = memnew(RenderingDevice);
	if (rendering_device->initialize(rendering_context, MAIN_WINDOW_ID) != OK) {
		rendering_device = nullptr;
		memdelete(rendering_context);
		rendering_context = nullptr;
		r_error = ERR_UNAVAILABLE;
		return;
	}
	rendering_device->screen_create(MAIN_WINDOW_ID);

	RendererCompositorRD::make_current();

	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);

	r_error = OK;
}

DisplayServerOpenHarmony::~DisplayServerOpenHarmony() {
}

void DisplayServerOpenHarmony::_window_callback(const Callable &p_callable, const Variant &p_arg, bool p_deferred) const {
	if (p_callable.is_valid()) {
		if (p_deferred) {
			p_callable.call_deferred(p_arg);
		} else {
			p_callable.call(p_arg);
		}
	}
}

void DisplayServerOpenHarmony::send_input_event(const Ref<InputEvent> &p_event) const {
	_window_callback(input_event_callback, p_event);
}

void DisplayServerOpenHarmony::resize_window(uint32_t p_width, uint32_t p_height) {
	Size2i size = Size2i(p_width, p_height);

#if defined(RD_ENABLED)
	if (rendering_context) {
		rendering_context->window_set_size(MAIN_WINDOW_ID, size.x, size.y);
	}
#endif

	Variant resize_rect = Rect2i(Point2i(), size);
	_window_callback(window_resize_callback, resize_rect);
}

void DisplayServerOpenHarmony::send_window_event(DisplayServer::WindowEvent p_event) const {
	_window_callback(window_event_callback, int(p_event));
}

bool DisplayServerOpenHarmony::has_feature(Feature p_feature) const {
	switch (p_feature) {
		case FEATURE_TOUCHSCREEN:
		case FEATURE_CLIPBOARD:
		case FEATURE_VIRTUAL_KEYBOARD:
		case FEATURE_IME:
		case FEATURE_KEEP_SCREEN_ON:
			return true;
		default:
			return false;
	}
}

String DisplayServerOpenHarmony::get_name() const {
	return "OpenHarmony";
}

int DisplayServerOpenHarmony::get_screen_count() const {
	return 1;
}

int DisplayServerOpenHarmony::get_primary_screen() const {
	return 0;
}

Point2i DisplayServerOpenHarmony::screen_get_position(int p_screen) const {
	return Point2i(0, 0);
}

Size2i DisplayServerOpenHarmony::screen_get_size(int p_screen) const {
	return OS_OpenHarmony::get_singleton()->get_display_size();
}

Rect2i DisplayServerOpenHarmony::screen_get_usable_rect(int p_screen) const {
	Size2i display_size = OS_OpenHarmony::get_singleton()->get_display_size();
	return Rect2i(0, 0, display_size.width, display_size.height);
}

int DisplayServerOpenHarmony::screen_get_dpi(int p_screen) const {
	return ohos_wrapper_get_display_dpi();
}

float DisplayServerOpenHarmony::screen_get_scale(int p_screen) const {
	return ohos_wrapper_get_display_scaled_density();
}

float DisplayServerOpenHarmony::screen_get_refresh_rate(int p_screen) const {
	return ohos_wrapper_get_display_refresh_rate();
}

bool DisplayServerOpenHarmony::is_touchscreen_available() const {
	return true;
}

void DisplayServerOpenHarmony::screen_set_orientation(DisplayServer::ScreenOrientation p_orientation, int p_screen) {
	// Not supported on OpenHarmony.
}

DisplayServer::ScreenOrientation DisplayServerOpenHarmony::screen_get_orientation(int p_screen) const {
	switch (ohos_wrapper_get_display_orientation()) {
		case WrapperScreenOrientation::WRAPPER_SCREEN_LANDSCAPE:
			return SCREEN_LANDSCAPE;
		case WrapperScreenOrientation::WRAPPER_SCREEN_PORTRAIT:
			return SCREEN_PORTRAIT;
		case WrapperScreenOrientation::WRAPPER_SCREEN_REVERSE_LANDSCAPE:
			return SCREEN_REVERSE_LANDSCAPE;
		case WrapperScreenOrientation::WRAPPER_SCREEN_REVERSE_PORTRAIT:
			return SCREEN_REVERSE_PORTRAIT;
		default:
			return SCREEN_PORTRAIT;
	}
}

void DisplayServerOpenHarmony::clipboard_set(const String &p_text) {
	OH_Pasteboard *pasteboard = OH_Pasteboard_Create();
	OH_UdsPlainText *plainText = OH_UdsPlainText_Create();
	OH_UdsPlainText_SetContent(plainText, p_text.utf8().get_data());
	OH_UdmfRecord *record = OH_UdmfRecord_Create();
	OH_UdmfRecord_AddPlainText(record, plainText);
	OH_UdmfData *data = OH_UdmfData_Create();
	OH_UdmfData_AddRecord(data, record);
	int status = OH_Pasteboard_SetData(pasteboard, data);
	if (status != 0) {
		ERR_PRINT("Failed to set clipboard data with PASTEBOARD_ErrCode: " + itos(status));
	}
	OH_UdsPlainText_Destroy(plainText);
	OH_UdmfRecord_Destroy(record);
	OH_UdmfData_Destroy(data);
	OH_Pasteboard_Destroy(pasteboard);
}

String DisplayServerOpenHarmony::clipboard_get() const {
	String content;
	OH_Pasteboard *pasteboard = OH_Pasteboard_Create();
	bool hasPlainTextData = OH_Pasteboard_HasType(pasteboard, "text/plain");
	if (hasPlainTextData) {
		int status = 0;
		OH_UdmfData *udmfData = OH_Pasteboard_GetData(pasteboard, &status);
		if (status == 0) {
			OH_UdmfRecord *record = OH_UdmfData_GetRecord(udmfData, 0);
			OH_UdsPlainText *plainText = OH_UdsPlainText_Create();
			OH_UdmfRecord_GetPlainText(record, plainText);
			content = String::utf8(OH_UdsPlainText_GetContent(plainText));
			OH_UdsPlainText_Destroy(plainText);
		} else {
			ERR_PRINT("Failed to get clipboard data with PASTEBOARD_ErrCode: " + itos(status));
		}
		OH_UdmfData_Destroy(udmfData);
	}
	OH_Pasteboard_Destroy(pasteboard);
	return content;
}

void DisplayServerOpenHarmony::screen_set_keep_on(bool p_enable) {
	ohos_wrapper_screen_set_keep_on(OS_OpenHarmony::get_singleton()->get_window_id(), p_enable);
}

bool DisplayServerOpenHarmony::screen_is_kept_on() const {
	return ohos_wrapper_screen_is_kept_on(OS_OpenHarmony::get_singleton()->get_window_id());
}

void DisplayServerOpenHarmony::_get_text_config(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_TextConfig *p_text_config) {
	InputMethod_TextInputType input_type = IME_TEXT_INPUT_TYPE_TEXT;
	InputMethod_EnterKeyType enter_key_type = IME_ENTER_KEY_DONE;
	switch (get_singleton()->keyboard_type) {
		case KEYBOARD_TYPE_DEFAULT:
			input_type = IME_TEXT_INPUT_TYPE_TEXT;
			break;
		case KEYBOARD_TYPE_MULTILINE:
			input_type = IME_TEXT_INPUT_TYPE_MULTILINE;
			enter_key_type = IME_ENTER_KEY_NEWLINE;
			break;
		case KEYBOARD_TYPE_NUMBER:
			input_type = IME_TEXT_INPUT_TYPE_NUMBER;
			break;
		case KEYBOARD_TYPE_NUMBER_DECIMAL:
			input_type = IME_TEXT_INPUT_TYPE_NUMBER_DECIMAL;
			break;
		case KEYBOARD_TYPE_PHONE:
			input_type = IME_TEXT_INPUT_TYPE_PHONE;
			break;
		case KEYBOARD_TYPE_EMAIL_ADDRESS:
			input_type = IME_TEXT_INPUT_TYPE_EMAIL_ADDRESS;
			break;
		case KEYBOARD_TYPE_PASSWORD:
			input_type = IME_TEXT_INPUT_TYPE_VISIBLE_PASSWORD;
			break;
		case KEYBOARD_TYPE_URL:
			input_type = IME_TEXT_INPUT_TYPE_URL;
			break;
		default:
			break;
	}
	OH_TextConfig_SetInputType(p_text_config, input_type);
	OH_TextConfig_SetPreviewTextSupport(p_text_config, false);
	OH_TextConfig_SetEnterKeyType(p_text_config, enter_key_type);
}

void DisplayServerOpenHarmony::_insert_text(InputMethod_TextEditorProxy *p_text_editor_proxy, const char16_t *p_text, size_t length) {
	String characters = String::utf16(p_text, length);

	for (int i = 0; i < characters.size(); i++) {
		int character = characters[i];
		Key key = Key::NONE;

		if (character == '\t') { // 0x09
			key = Key::TAB;
		} else if (character == '\n') { // 0x0A
			key = Key::ENTER;
		} else if (character == 0x2006) {
			key = Key::SPACE;
		}

		_input_text_key(key, character, key, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
		_input_text_key(key, character, key, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
	}
}

void DisplayServerOpenHarmony::_delete_forward(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t length) {
	for (int i = 0; i < length; i++) {
		_input_text_key(Key::KEY_DELETE, 0, Key::KEY_DELETE, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
		_input_text_key(Key::KEY_DELETE, 0, Key::KEY_DELETE, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
	}
}

void DisplayServerOpenHarmony::_delete_backward(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t length) {
	for (int i = 0; i < length; i++) {
		_input_text_key(Key::BACKSPACE, 0, Key::BACKSPACE, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
		_input_text_key(Key::BACKSPACE, 0, Key::BACKSPACE, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
	}
}

void DisplayServerOpenHarmony::_send_keyboard_status(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_KeyboardStatus status) {
	get_singleton()->keyboard_status = status;
}

void DisplayServerOpenHarmony::_send_enter_key(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_EnterKeyType enter_key_type) {
	_input_text_key(Key::ENTER, 0, Key::ENTER, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
	_input_text_key(Key::ENTER, 0, Key::ENTER, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
}

void DisplayServerOpenHarmony::_move_cursor(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_Direction direction) {
	switch (direction) {
		case IME_DIRECTION_LEFT:
			_input_text_key(Key::LEFT, 0, Key::LEFT, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
			_input_text_key(Key::LEFT, 0, Key::LEFT, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
			break;
		case IME_DIRECTION_RIGHT:
			_input_text_key(Key::RIGHT, 0, Key::RIGHT, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
			_input_text_key(Key::RIGHT, 0, Key::RIGHT, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
			break;
		case IME_DIRECTION_UP:
			_input_text_key(Key::UP, 0, Key::UP, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
			_input_text_key(Key::UP, 0, Key::UP, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
			break;
		case IME_DIRECTION_DOWN:
			_input_text_key(Key::DOWN, 0, Key::DOWN, Key::NONE, 0, true, KeyLocation::UNSPECIFIED);
			_input_text_key(Key::DOWN, 0, Key::DOWN, Key::NONE, 0, false, KeyLocation::UNSPECIFIED);
			break;
		default:
			break;
	}
}

void DisplayServerOpenHarmony::_handle_set_selection(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t start, int32_t end) {
	// Not supported by Godot.
}

void DisplayServerOpenHarmony::_handle_extend_action(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_ExtendAction action) {
	// Not supported by Godot.
}

void DisplayServerOpenHarmony::_get_left_text_of_cursor(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t number, char16_t *p_text, size_t *p_length) {
	// Not supported by Godot.
}

void DisplayServerOpenHarmony::_get_right_text_of_cursor(InputMethod_TextEditorProxy *p_text_editor_proxy, int32_t number, char16_t *p_text, size_t *p_length) {
	// Not supported by Godot.
}

int32_t DisplayServerOpenHarmony::_get_text_index_at_cursor(InputMethod_TextEditorProxy *p_text_editor_proxy) {
	// Not supported by Godot.
	return 0;
}

int32_t DisplayServerOpenHarmony::_receive_private_command(InputMethod_TextEditorProxy *p_text_editor_proxy, InputMethod_PrivateCommand *p_command[], size_t length) {
	// Not supported by Godot.
	return 0;
}

int32_t DisplayServerOpenHarmony::_set_preview_text(InputMethod_TextEditorProxy *p_text_editor_proxy, const char16_t *p_text, size_t length, int32_t start, int32_t end) {
	// Not supported by Godot.
	return 0;
}

void DisplayServerOpenHarmony::_finish_text_preview(InputMethod_TextEditorProxy *p_text_editor_proxy) {
	// Not supported by Godot.
}

void DisplayServerOpenHarmony::_input_text_key(Key p_key, char32_t p_char, Key p_unshifted, Key p_physical, int p_modifier, bool p_pressed, KeyLocation p_location) {
	Ref<InputEventKey> ev;
	ev.instantiate();
	ev->set_echo(false);
	ev->set_pressed(p_pressed);
	ev->set_keycode(fix_keycode(p_char, p_key));
	ev->set_key_label(p_unshifted);
	ev->set_physical_keycode(p_physical);
	ev->set_unicode(fix_unicode(p_char));
	ev->set_location(p_location);
	Input::get_singleton()->parse_input_event(ev);
}

void DisplayServerOpenHarmony::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, VirtualKeyboardType p_type, int p_max_length, int p_cursor_start, int p_cursor_end) {
	if (keyboard_status == IME_KEYBOARD_STATUS_SHOW && keyboard_type == p_type) {
		return;
	}
	if (keyboard_status != IME_KEYBOARD_STATUS_NONE) {
		virtual_keyboard_hide();
	}

	keyboard_type = p_type;
	text_editor_proxy = OH_TextEditorProxy_Create();
	attach_options = OH_AttachOptions_Create(true);

	OH_TextEditorProxy_SetGetTextConfigFunc(text_editor_proxy, _get_text_config);
	OH_TextEditorProxy_SetInsertTextFunc(text_editor_proxy, _insert_text);
	OH_TextEditorProxy_SetDeleteForwardFunc(text_editor_proxy, _delete_forward);
	OH_TextEditorProxy_SetDeleteBackwardFunc(text_editor_proxy, _delete_backward);
	OH_TextEditorProxy_SetSendKeyboardStatusFunc(text_editor_proxy, _send_keyboard_status);
	OH_TextEditorProxy_SetSendEnterKeyFunc(text_editor_proxy, _send_enter_key);
	OH_TextEditorProxy_SetMoveCursorFunc(text_editor_proxy, _move_cursor);
	OH_TextEditorProxy_SetHandleSetSelectionFunc(text_editor_proxy, _handle_set_selection);
	OH_TextEditorProxy_SetHandleExtendActionFunc(text_editor_proxy, _handle_extend_action);
	OH_TextEditorProxy_SetGetLeftTextOfCursorFunc(text_editor_proxy, _get_left_text_of_cursor);
	OH_TextEditorProxy_SetGetRightTextOfCursorFunc(text_editor_proxy, _get_right_text_of_cursor);
	OH_TextEditorProxy_SetGetTextIndexAtCursorFunc(text_editor_proxy, _get_text_index_at_cursor);
	OH_TextEditorProxy_SetReceivePrivateCommandFunc(text_editor_proxy, _receive_private_command);
	OH_TextEditorProxy_SetSetPreviewTextFunc(text_editor_proxy, _set_preview_text);
	OH_TextEditorProxy_SetFinishTextPreviewFunc(text_editor_proxy, _finish_text_preview);

	auto retult = OH_InputMethodController_Attach(text_editor_proxy, attach_options, &input_method_proxy);
	ERR_FAIL_COND_MSG(retult != IME_ERR_OK, vformat("Failed to attach input method controller: %d.", retult));
}

void DisplayServerOpenHarmony::virtual_keyboard_hide() {
	if (keyboard_status == IME_KEYBOARD_STATUS_SHOW) {
		if (OH_InputMethodProxy_HideKeyboard(input_method_proxy) != IME_ERR_OK) {
			ERR_PRINT("Failed to hide keyboard.");
		}
	}
	if (input_method_proxy) {
		if (OH_InputMethodController_Detach(input_method_proxy) != IME_ERR_OK) {
			ERR_PRINT("Failed to detach input method controller.");
		}
		input_method_proxy = nullptr;
	}
	if (attach_options) {
		OH_AttachOptions_Destroy(attach_options);
		attach_options = nullptr;
	}
	if (text_editor_proxy) {
		OH_TextEditorProxy_Destroy(text_editor_proxy);
		text_editor_proxy = nullptr;
	}
	keyboard_status = IME_KEYBOARD_STATUS_NONE;
}

int DisplayServerOpenHarmony::virtual_keyboard_get_height() const {
	if (keyboard_status == IME_KEYBOARD_STATUS_SHOW) {
		int height = ohos_wrapper_get_keyboard_avoid_area(OS_OpenHarmony::get_singleton()->get_window_id());
		return height;
	}
	return 0;
}

void DisplayServerOpenHarmony::window_set_ime_active(const bool p_active, DisplayServer::WindowID p_window) {
	ime_active = p_active;
}

void DisplayServerOpenHarmony::window_set_ime_position(const Point2i &p_pos, DisplayServer::WindowID p_window) {
	if (ime_active) {
		InputMethod_CursorInfo *info = OH_CursorInfo_Create(p_pos.x, p_pos.y, 0, 30);
		OH_InputMethodProxy_NotifyCursorUpdate(input_method_proxy, info);
	}
}

Vector<DisplayServer::WindowID> DisplayServerOpenHarmony::get_window_list() const {
	Vector<WindowID> ret;
	ret.push_back(MAIN_WINDOW_ID);
	return ret;
}

DisplayServer::WindowID DisplayServerOpenHarmony::get_window_at_screen_position(const Point2i &p_position) const {
	return MAIN_WINDOW_ID;
}

void DisplayServerOpenHarmony::window_attach_instance_id(ObjectID p_instance, DisplayServer::WindowID p_window) {
	window_attached_instance_id = p_instance;
}

ObjectID DisplayServerOpenHarmony::window_get_attached_instance_id(DisplayServer::WindowID p_window) const {
	return window_attached_instance_id;
}

void DisplayServerOpenHarmony::window_set_window_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	window_event_callback = p_callable;
}

void DisplayServerOpenHarmony::window_set_input_event_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	input_event_callback = p_callable;
}

void DisplayServerOpenHarmony::window_set_input_text_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	input_text_callback = p_callable;
}

void DisplayServerOpenHarmony::window_set_rect_changed_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	window_resize_callback = p_callable;
}

void DisplayServerOpenHarmony::window_set_drop_files_callback(const Callable &p_callable, DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

void DisplayServerOpenHarmony::window_set_title(const String &p_title, DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

int DisplayServerOpenHarmony::window_get_current_screen(DisplayServer::WindowID p_window) const {
	return SCREEN_OF_MAIN_WINDOW;
}

void DisplayServerOpenHarmony::window_set_current_screen(int p_screen, DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

Point2i DisplayServerOpenHarmony::window_get_position(DisplayServer::WindowID p_window) const {
	return Point2i();
}

Point2i DisplayServerOpenHarmony::window_get_position_with_decorations(DisplayServer::WindowID p_window) const {
	return Point2i();
}

void DisplayServerOpenHarmony::window_set_position(const Point2i &p_position, DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

void DisplayServerOpenHarmony::window_set_transient(DisplayServer::WindowID p_window, DisplayServer::WindowID p_parent) {
	// Not supported on OpenHarmony.
}

void DisplayServerOpenHarmony::window_set_max_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

Size2i DisplayServerOpenHarmony::window_get_max_size(DisplayServer::WindowID p_window) const {
	return Size2i();
}

void DisplayServerOpenHarmony::window_set_min_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

Size2i DisplayServerOpenHarmony::window_get_min_size(DisplayServer::WindowID p_window) const {
	return Size2i();
}

void DisplayServerOpenHarmony::window_set_size(const Size2i p_size, DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

Size2i DisplayServerOpenHarmony::window_get_size(DisplayServer::WindowID p_window) const {
	return OS_OpenHarmony::get_singleton()->get_display_size();
}

Size2i DisplayServerOpenHarmony::window_get_size_with_decorations(DisplayServer::WindowID p_window) const {
	return OS_OpenHarmony::get_singleton()->get_display_size();
}

void DisplayServerOpenHarmony::window_set_mode(DisplayServer::WindowMode p_mode, DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

DisplayServer::WindowMode DisplayServerOpenHarmony::window_get_mode(DisplayServer::WindowID p_window) const {
	return WINDOW_MODE_FULLSCREEN;
}

void DisplayServerOpenHarmony::window_set_vsync_mode(VSyncMode p_vsync_mode, WindowID p_window) {
	// Not supported on OpenHarmony.
}

DisplayServer::VSyncMode DisplayServerOpenHarmony::window_get_vsync_mode(WindowID p_window) const {
	return VSyncMode::VSYNC_ADAPTIVE;
}

bool DisplayServerOpenHarmony::window_is_maximize_allowed(DisplayServer::WindowID p_window) const {
	return false;
}

void DisplayServerOpenHarmony::window_set_flag(DisplayServer::WindowFlags p_flag, bool p_enabled, DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

bool DisplayServerOpenHarmony::window_get_flag(DisplayServer::WindowFlags p_flag, DisplayServer::WindowID p_window) const {
	return false;
}

void DisplayServerOpenHarmony::window_request_attention(DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

void DisplayServerOpenHarmony::window_move_to_foreground(DisplayServer::WindowID p_window) {
	// Not supported on OpenHarmony.
}

bool DisplayServerOpenHarmony::window_is_focused(WindowID p_window) const {
	return true;
}

bool DisplayServerOpenHarmony::window_can_draw(DisplayServer::WindowID p_window) const {
	return true;
}

bool DisplayServerOpenHarmony::can_any_window_draw() const {
	return true;
}

void DisplayServerOpenHarmony::process_events() {
	Input::get_singleton()->flush_buffered_events();
}
