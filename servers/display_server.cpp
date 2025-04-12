/**************************************************************************/
/*  display_server.cpp                                                    */
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

#include "display_server.h"
#include "display_server.compat.inc"

#include "core/input/input.h"
#include "scene/resources/texture.h"
#include "servers/display_server_headless.h"

#if defined(VULKAN_ENABLED)
#include "drivers/vulkan/rendering_context_driver_vulkan.h"
#undef CursorShape
#endif
#if defined(D3D12_ENABLED)
#include "drivers/d3d12/rendering_context_driver_d3d12.h"
#endif
#if defined(METAL_ENABLED)
#include "drivers/metal/rendering_context_driver_metal.h"
#endif

DisplayServer *DisplayServer::singleton = nullptr;

DisplayServer::AccessibilityMode DisplayServer::accessibility_mode = DisplayServer::AccessibilityMode::ACCESSIBILITY_AUTO;

bool DisplayServer::hidpi_allowed = false;

bool DisplayServer::window_early_clear_override_enabled = false;
Color DisplayServer::window_early_clear_override_color = Color(0, 0, 0, 0);

DisplayServer::DisplayServerCreate DisplayServer::server_create_functions[DisplayServer::MAX_SERVERS] = {
	{ "headless", &DisplayServerHeadless::create_func, &DisplayServerHeadless::get_rendering_drivers_func }
};

int DisplayServer::server_create_count = 1;

void DisplayServer::help_set_search_callbacks(const Callable &p_search_callback, const Callable &p_action_callback) {
	WARN_PRINT("Native help is not supported by this display server.");
}

#ifndef DISABLE_DEPRECATED

RID DisplayServer::_get_rid_from_name(NativeMenu *p_nmenu, const String &p_menu_root) const {
	if (p_menu_root == "_main") {
		return p_nmenu->get_system_menu(NativeMenu::MAIN_MENU_ID);
	} else if (p_menu_root == "_apple") {
		return p_nmenu->get_system_menu(NativeMenu::APPLICATION_MENU_ID);
	} else if (p_menu_root == "_dock") {
		return p_nmenu->get_system_menu(NativeMenu::DOCK_MENU_ID);
	} else if (p_menu_root == "_help") {
		return p_nmenu->get_system_menu(NativeMenu::HELP_MENU_ID);
	} else if (p_menu_root == "_window") {
		return p_nmenu->get_system_menu(NativeMenu::WINDOW_MENU_ID);
	} else if (menu_names.has(p_menu_root)) {
		return menu_names[p_menu_root];
	}

	RID rid = p_nmenu->create_menu();
	menu_names[p_menu_root] = rid;
	return rid;
}

int DisplayServer::global_menu_add_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->add_item(_get_rid_from_name(nmenu, p_menu_root), p_label, p_callback, p_key_callback, p_tag, p_accel, p_index);
}

int DisplayServer::global_menu_add_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->add_check_item(_get_rid_from_name(nmenu, p_menu_root), p_label, p_callback, p_key_callback, p_tag, p_accel, p_index);
}

int DisplayServer::global_menu_add_icon_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->add_icon_item(_get_rid_from_name(nmenu, p_menu_root), p_icon, p_label, p_callback, p_key_callback, p_tag, p_accel, p_index);
}

int DisplayServer::global_menu_add_icon_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->add_icon_check_item(_get_rid_from_name(nmenu, p_menu_root), p_icon, p_label, p_callback, p_key_callback, p_tag, p_accel, p_index);
}

int DisplayServer::global_menu_add_radio_check_item(const String &p_menu_root, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->add_radio_check_item(_get_rid_from_name(nmenu, p_menu_root), p_label, p_callback, p_key_callback, p_tag, p_accel, p_index);
}

int DisplayServer::global_menu_add_icon_radio_check_item(const String &p_menu_root, const Ref<Texture2D> &p_icon, const String &p_label, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->add_icon_radio_check_item(_get_rid_from_name(nmenu, p_menu_root), p_icon, p_label, p_callback, p_key_callback, p_tag, p_accel, p_index);
}

int DisplayServer::global_menu_add_multistate_item(const String &p_menu_root, const String &p_label, int p_max_states, int p_default_state, const Callable &p_callback, const Callable &p_key_callback, const Variant &p_tag, Key p_accel, int p_index) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->add_multistate_item(_get_rid_from_name(nmenu, p_menu_root), p_label, p_max_states, p_default_state, p_callback, p_key_callback, p_tag, p_accel, p_index);
}

void DisplayServer::global_menu_set_popup_callbacks(const String &p_menu_root, const Callable &p_open_callback, const Callable &p_close_callback) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_popup_open_callback(_get_rid_from_name(nmenu, p_menu_root), p_open_callback);
	nmenu->set_popup_open_callback(_get_rid_from_name(nmenu, p_menu_root), p_close_callback);
}

int DisplayServer::global_menu_add_submenu_item(const String &p_menu_root, const String &p_label, const String &p_submenu, int p_index) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->add_submenu_item(_get_rid_from_name(nmenu, p_menu_root), p_label, _get_rid_from_name(nmenu, p_submenu), Variant(), p_index);
}

int DisplayServer::global_menu_add_separator(const String &p_menu_root, int p_index) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->add_separator(_get_rid_from_name(nmenu, p_menu_root), p_index);
}

int DisplayServer::global_menu_get_item_index_from_text(const String &p_menu_root, const String &p_text) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->find_item_index_with_text(_get_rid_from_name(nmenu, p_menu_root), p_text);
}

int DisplayServer::global_menu_get_item_index_from_tag(const String &p_menu_root, const Variant &p_tag) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->find_item_index_with_tag(_get_rid_from_name(nmenu, p_menu_root), p_tag);
}

void DisplayServer::global_menu_set_item_callback(const String &p_menu_root, int p_idx, const Callable &p_callback) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_callback(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_callback);
}

void DisplayServer::global_menu_set_item_hover_callbacks(const String &p_menu_root, int p_idx, const Callable &p_callback) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_hover_callbacks(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_callback);
}

void DisplayServer::global_menu_set_item_key_callback(const String &p_menu_root, int p_idx, const Callable &p_key_callback) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_key_callback(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_key_callback);
}

bool DisplayServer::global_menu_is_item_checked(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, false);
	return nmenu->is_item_checked(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

bool DisplayServer::global_menu_is_item_checkable(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, false);
	return nmenu->is_item_checkable(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

bool DisplayServer::global_menu_is_item_radio_checkable(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, false);
	return nmenu->is_item_radio_checkable(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

Callable DisplayServer::global_menu_get_item_callback(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, Callable());
	return nmenu->get_item_callback(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

Callable DisplayServer::global_menu_get_item_key_callback(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, Callable());
	return nmenu->get_item_key_callback(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

Variant DisplayServer::global_menu_get_item_tag(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, Variant());
	return nmenu->get_item_tag(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

String DisplayServer::global_menu_get_item_text(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, String());
	return nmenu->get_item_text(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

String DisplayServer::global_menu_get_item_submenu(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, String());
	RID rid = nmenu->get_item_submenu(_get_rid_from_name(nmenu, p_menu_root), p_idx);
	if (!nmenu->is_system_menu(rid)) {
		for (HashMap<String, RID>::Iterator E = menu_names.begin(); E; ++E) {
			if (E->value == rid) {
				return E->key;
			}
		}
	}
	return String();
}

Key DisplayServer::global_menu_get_item_accelerator(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, Key::NONE);
	return nmenu->get_item_accelerator(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

bool DisplayServer::global_menu_is_item_disabled(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, false);
	return nmenu->is_item_disabled(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

bool DisplayServer::global_menu_is_item_hidden(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, false);
	return nmenu->is_item_hidden(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

String DisplayServer::global_menu_get_item_tooltip(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, String());
	return nmenu->get_item_tooltip(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

int DisplayServer::global_menu_get_item_state(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->get_item_state(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

int DisplayServer::global_menu_get_item_max_states(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, -1);
	return nmenu->get_item_max_states(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

Ref<Texture2D> DisplayServer::global_menu_get_item_icon(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, Ref<Texture2D>());
	return nmenu->get_item_icon(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

int DisplayServer::global_menu_get_item_indentation_level(const String &p_menu_root, int p_idx) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, 0);
	return nmenu->get_item_indentation_level(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

void DisplayServer::global_menu_set_item_checked(const String &p_menu_root, int p_idx, bool p_checked) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_checked(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_checked);
}

void DisplayServer::global_menu_set_item_checkable(const String &p_menu_root, int p_idx, bool p_checkable) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_checkable(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_checkable);
}

void DisplayServer::global_menu_set_item_radio_checkable(const String &p_menu_root, int p_idx, bool p_checkable) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_radio_checkable(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_checkable);
}

void DisplayServer::global_menu_set_item_tag(const String &p_menu_root, int p_idx, const Variant &p_tag) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_tag(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_tag);
}

void DisplayServer::global_menu_set_item_text(const String &p_menu_root, int p_idx, const String &p_text) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_text(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_text);
}

void DisplayServer::global_menu_set_item_submenu(const String &p_menu_root, int p_idx, const String &p_submenu) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_submenu(_get_rid_from_name(nmenu, p_menu_root), p_idx, _get_rid_from_name(nmenu, p_submenu));
}

void DisplayServer::global_menu_set_item_accelerator(const String &p_menu_root, int p_idx, Key p_keycode) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_accelerator(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_keycode);
}

void DisplayServer::global_menu_set_item_disabled(const String &p_menu_root, int p_idx, bool p_disabled) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_disabled(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_disabled);
}

void DisplayServer::global_menu_set_item_hidden(const String &p_menu_root, int p_idx, bool p_hidden) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_hidden(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_hidden);
}

void DisplayServer::global_menu_set_item_tooltip(const String &p_menu_root, int p_idx, const String &p_tooltip) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_tooltip(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_tooltip);
}

void DisplayServer::global_menu_set_item_state(const String &p_menu_root, int p_idx, int p_state) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_state(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_state);
}

void DisplayServer::global_menu_set_item_max_states(const String &p_menu_root, int p_idx, int p_max_states) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_max_states(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_max_states);
}

void DisplayServer::global_menu_set_item_icon(const String &p_menu_root, int p_idx, const Ref<Texture2D> &p_icon) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_icon(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_icon);
}

void DisplayServer::global_menu_set_item_indentation_level(const String &p_menu_root, int p_idx, int p_level) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->set_item_indentation_level(_get_rid_from_name(nmenu, p_menu_root), p_idx, p_level);
}

int DisplayServer::global_menu_get_item_count(const String &p_menu_root) const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, 0);
	return nmenu->get_item_count(_get_rid_from_name(nmenu, p_menu_root));
}

void DisplayServer::global_menu_remove_item(const String &p_menu_root, int p_idx) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	nmenu->remove_item(_get_rid_from_name(nmenu, p_menu_root), p_idx);
}

void DisplayServer::global_menu_clear(const String &p_menu_root) {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL(nmenu);
	RID rid = _get_rid_from_name(nmenu, p_menu_root);
	nmenu->clear(rid);
	if (!nmenu->is_system_menu(rid)) {
		nmenu->free_menu(rid);
		menu_names.erase(p_menu_root);
	}
}

Dictionary DisplayServer::global_menu_get_system_menu_roots() const {
	NativeMenu *nmenu = NativeMenu::get_singleton();
	ERR_FAIL_NULL_V(nmenu, Dictionary());

	Dictionary out;
	if (nmenu->has_system_menu(NativeMenu::DOCK_MENU_ID)) {
		out["_dock"] = "@Dock";
	}
	if (nmenu->has_system_menu(NativeMenu::APPLICATION_MENU_ID)) {
		out["_apple"] = "@Apple";
	}
	if (nmenu->has_system_menu(NativeMenu::WINDOW_MENU_ID)) {
		out["_window"] = "Window";
	}
	if (nmenu->has_system_menu(NativeMenu::HELP_MENU_ID)) {
		out["_help"] = "Help";
	}
	return out;
}

#endif

bool DisplayServer::tts_is_speaking() const {
	WARN_PRINT("TTS is not supported by this display server.");
	return false;
}

bool DisplayServer::tts_is_paused() const {
	WARN_PRINT("TTS is not supported by this display server.");
	return false;
}

void DisplayServer::tts_pause() {
	WARN_PRINT("TTS is not supported by this display server.");
}

void DisplayServer::tts_resume() {
	WARN_PRINT("TTS is not supported by this display server.");
}

TypedArray<Dictionary> DisplayServer::tts_get_voices() const {
	WARN_PRINT("TTS is not supported by this display server.");
	return TypedArray<Dictionary>();
}

PackedStringArray DisplayServer::tts_get_voices_for_language(const String &p_language) const {
	PackedStringArray ret;
	TypedArray<Dictionary> voices = tts_get_voices();
	for (int i = 0; i < voices.size(); i++) {
		const Dictionary &voice = voices[i];
		if (voice.has("id") && voice.has("language") && voice["language"].operator String().begins_with(p_language)) {
			ret.push_back(voice["id"]);
		}
	}
	return ret;
}

void DisplayServer::tts_speak(const String &p_text, const String &p_voice, int p_volume, float p_pitch, float p_rate, int p_utterance_id, bool p_interrupt) {
	WARN_PRINT("TTS is not supported by this display server.");
}

void DisplayServer::tts_stop() {
	WARN_PRINT("TTS is not supported by this display server.");
}

void DisplayServer::tts_set_utterance_callback(TTSUtteranceEvent p_event, const Callable &p_callable) {
	ERR_FAIL_INDEX(p_event, DisplayServer::TTS_UTTERANCE_MAX);
	utterance_callback[p_event] = p_callable;
}

void DisplayServer::tts_post_utterance_event(TTSUtteranceEvent p_event, int p_id, int p_pos) {
	ERR_FAIL_INDEX(p_event, DisplayServer::TTS_UTTERANCE_MAX);
	switch (p_event) {
		case DisplayServer::TTS_UTTERANCE_STARTED:
		case DisplayServer::TTS_UTTERANCE_ENDED:
		case DisplayServer::TTS_UTTERANCE_CANCELED: {
			if (utterance_callback[p_event].is_valid()) {
				utterance_callback[p_event].call_deferred(p_id); // Should be deferred, on some platforms utterance events can be called from different threads in a rapid succession.
			}
		} break;
		case DisplayServer::TTS_UTTERANCE_BOUNDARY: {
			if (utterance_callback[p_event].is_valid()) {
				utterance_callback[p_event].call_deferred(p_pos, p_id); // Should be deferred, on some platforms utterance events can be called from different threads in a rapid succession.
			}
		} break;
		default:
			break;
	}
}

bool DisplayServer::_get_window_early_clear_override(Color &r_color) {
	if (window_early_clear_override_enabled) {
		r_color = window_early_clear_override_color;
		return true;
	} else if (RenderingServer::get_singleton()) {
		r_color = RenderingServer::get_singleton()->get_default_clear_color();
		return true;
	} else {
		return false;
	}
}

void DisplayServer::set_early_window_clear_color_override(bool p_enabled, Color p_color) {
	window_early_clear_override_enabled = p_enabled;
	window_early_clear_override_color = p_color;
}

void DisplayServer::mouse_set_mode(MouseMode p_mode) {
	WARN_PRINT("Mouse is not supported by this display server.");
}

DisplayServer::MouseMode DisplayServer::mouse_get_mode() const {
	return MOUSE_MODE_VISIBLE;
}

void DisplayServer::mouse_set_mode_override(MouseMode p_mode) {
	WARN_PRINT("Mouse is not supported by this display server.");
}

DisplayServer::MouseMode DisplayServer::mouse_get_mode_override() const {
	return MOUSE_MODE_VISIBLE;
}

void DisplayServer::mouse_set_mode_override_enabled(bool p_override_enabled) {
	WARN_PRINT("Mouse is not supported by this display server.");
}

bool DisplayServer::mouse_is_mode_override_enabled() const {
	return false;
}

void DisplayServer::warp_mouse(const Point2i &p_position) {
}

Point2i DisplayServer::mouse_get_position() const {
	ERR_FAIL_V_MSG(Point2i(), "Mouse is not supported by this display server.");
}

BitField<MouseButtonMask> DisplayServer::mouse_get_button_state() const {
	ERR_FAIL_V_MSG(MouseButtonMask::NONE, "Mouse is not supported by this display server.");
}

void DisplayServer::clipboard_set(const String &p_text) {
	WARN_PRINT("Clipboard is not supported by this display server.");
}

String DisplayServer::clipboard_get() const {
	ERR_FAIL_V_MSG(String(), "Clipboard is not supported by this display server.");
}

Ref<Image> DisplayServer::clipboard_get_image() const {
	ERR_FAIL_V_MSG(Ref<Image>(), "Clipboard is not supported by this display server.");
}

bool DisplayServer::clipboard_has() const {
	return !clipboard_get().is_empty();
}

bool DisplayServer::clipboard_has_image() const {
	return clipboard_get_image().is_valid();
}

void DisplayServer::clipboard_set_primary(const String &p_text) {
	WARN_PRINT("Primary clipboard is not supported by this display server.");
}

String DisplayServer::clipboard_get_primary() const {
	ERR_FAIL_V_MSG(String(), "Primary clipboard is not supported by this display server.");
}

void DisplayServer::screen_set_orientation(ScreenOrientation p_orientation, int p_screen) {
	WARN_PRINT("Orientation not supported by this display server.");
}

DisplayServer::ScreenOrientation DisplayServer::screen_get_orientation(int p_screen) const {
	return SCREEN_LANDSCAPE;
}

float DisplayServer::screen_get_scale(int p_screen) const {
	return 1.0f;
}

bool DisplayServer::is_touchscreen_available() const {
	return Input::get_singleton() && Input::get_singleton()->is_emulating_touch_from_mouse();
}

void DisplayServer::screen_set_keep_on(bool p_enable) {
	WARN_PRINT("Keeping screen on not supported by this display server.");
}

bool DisplayServer::screen_is_kept_on() const {
	return false;
}

int DisplayServer::get_screen_from_rect(const Rect2 &p_rect) const {
	int nearest_area = 0;
	int pos_screen = -1;
	for (int i = 0; i < get_screen_count(); i++) {
		Rect2i r;
		r.position = screen_get_position(i);
		r.size = screen_get_size(i);
		Rect2 inters = r.intersection(p_rect);
		int area = inters.size.width * inters.size.height;
		if (area > nearest_area) {
			pos_screen = i;
			nearest_area = area;
		}
	}
	return pos_screen;
}

DisplayServer::WindowID DisplayServer::create_sub_window(WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Rect2i &p_rect, bool p_exclusive, WindowID p_transient_parent) {
	ERR_FAIL_V_MSG(INVALID_WINDOW_ID, "Sub-windows not supported by this display server.");
}

void DisplayServer::show_window(WindowID p_id) {
	ERR_FAIL_MSG("Sub-windows not supported by this display server.");
}

void DisplayServer::delete_sub_window(WindowID p_id) {
	ERR_FAIL_MSG("Sub-windows not supported by this display server.");
}

void DisplayServer::window_set_exclusive(WindowID p_window, bool p_exclusive) {
	// Do nothing, if not supported.
}

void DisplayServer::window_set_mouse_passthrough(const Vector<Vector2> &p_region, WindowID p_window) {
	ERR_FAIL_MSG("Mouse passthrough not supported by this display server.");
}

void DisplayServer::gl_window_make_current(DisplayServer::WindowID p_window_id) {
	// noop except in gles
}

void DisplayServer::window_set_ime_active(const bool p_active, WindowID p_window) {
	WARN_PRINT("IME not supported by this display server.");
}

void DisplayServer::window_set_ime_position(const Point2i &p_pos, WindowID p_window) {
	WARN_PRINT("IME not supported by this display server.");
}

RID DisplayServer::accessibility_create_element(WindowID p_window, DisplayServer::AccessibilityRole p_role) {
	if (accessibility_driver) {
		return accessibility_driver->accessibility_create_element(p_window, p_role);
	} else {
		return RID();
	}
}

RID DisplayServer::accessibility_create_sub_element(const RID &p_parent_rid, DisplayServer::AccessibilityRole p_role, int p_insert_pos) {
	if (accessibility_driver) {
		return accessibility_driver->accessibility_create_sub_element(p_parent_rid, p_role, p_insert_pos);
	} else {
		return RID();
	}
}

RID DisplayServer::accessibility_create_sub_text_edit_elements(const RID &p_parent_rid, const RID &p_shaped_text, float p_min_height, int p_insert_pos) {
	if (accessibility_driver) {
		return accessibility_driver->accessibility_create_sub_text_edit_elements(p_parent_rid, p_shaped_text, p_min_height, p_insert_pos);
	} else {
		return RID();
	}
}

bool DisplayServer::accessibility_has_element(const RID &p_id) const {
	if (accessibility_driver) {
		return accessibility_driver->accessibility_has_element(p_id);
	} else {
		return false;
	}
}

void DisplayServer::accessibility_free_element(const RID &p_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_free_element(p_id);
	}
}

void DisplayServer::accessibility_element_set_meta(const RID &p_id, const Variant &p_meta) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_element_set_meta(p_id, p_meta);
	}
}

Variant DisplayServer::accessibility_element_get_meta(const RID &p_id) const {
	if (accessibility_driver) {
		return accessibility_driver->accessibility_element_get_meta(p_id);
	} else {
		return Variant();
	}
}

void DisplayServer::accessibility_update_if_active(const Callable &p_callable) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_if_active(p_callable);
	}
}

void DisplayServer::accessibility_update_set_focus(const RID &p_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_focus(p_id);
	}
}

RID DisplayServer::accessibility_get_window_root(DisplayServer::WindowID p_window_id) const {
	if (accessibility_driver) {
		return accessibility_driver->accessibility_get_window_root(p_window_id);
	} else {
		return RID();
	}
}

void DisplayServer::accessibility_set_window_rect(DisplayServer::WindowID p_window_id, const Rect2 &p_rect_out, const Rect2 &p_rect_in) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_set_window_rect(p_window_id, p_rect_out, p_rect_in);
	}
}

void DisplayServer::accessibility_set_window_focused(DisplayServer::WindowID p_window_id, bool p_focused) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_set_window_focused(p_window_id, p_focused);
	}
}

void DisplayServer::accessibility_update_set_role(const RID &p_id, DisplayServer::AccessibilityRole p_role) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_role(p_id, p_role);
	}
}

void DisplayServer::accessibility_update_set_name(const RID &p_id, const String &p_name) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_name(p_id, p_name);
	}
}

void DisplayServer::accessibility_update_set_description(const RID &p_id, const String &p_description) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_description(p_id, p_description);
	}
}

void DisplayServer::accessibility_update_set_extra_info(const RID &p_id, const String &p_name_extra_info) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_extra_info(p_id, p_name_extra_info);
	}
}

void DisplayServer::accessibility_update_set_value(const RID &p_id, const String &p_value) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_value(p_id, p_value);
	}
}

void DisplayServer::accessibility_update_set_tooltip(const RID &p_id, const String &p_tooltip) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_tooltip(p_id, p_tooltip);
	}
}

void DisplayServer::accessibility_update_set_bounds(const RID &p_id, const Rect2 &p_rect) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_bounds(p_id, p_rect);
	}
}

void DisplayServer::accessibility_update_set_transform(const RID &p_id, const Transform2D &p_transform) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_transform(p_id, p_transform);
	}
}

void DisplayServer::accessibility_update_add_child(const RID &p_id, const RID &p_child_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_add_child(p_id, p_child_id);
	}
}

void DisplayServer::accessibility_update_add_related_controls(const RID &p_id, const RID &p_related_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_add_related_controls(p_id, p_related_id);
	}
}

void DisplayServer::accessibility_update_add_related_details(const RID &p_id, const RID &p_related_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_add_related_details(p_id, p_related_id);
	}
}

void DisplayServer::accessibility_update_add_related_described_by(const RID &p_id, const RID &p_related_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_add_related_described_by(p_id, p_related_id);
	}
}

void DisplayServer::accessibility_update_add_related_flow_to(const RID &p_id, const RID &p_related_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_add_related_flow_to(p_id, p_related_id);
	}
}

void DisplayServer::accessibility_update_add_related_labeled_by(const RID &p_id, const RID &p_related_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_add_related_labeled_by(p_id, p_related_id);
	}
}

void DisplayServer::accessibility_update_add_related_radio_group(const RID &p_id, const RID &p_related_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_add_related_radio_group(p_id, p_related_id);
	}
}

void DisplayServer::accessibility_update_set_active_descendant(const RID &p_id, const RID &p_other_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_active_descendant(p_id, p_other_id);
	}
}

void DisplayServer::accessibility_update_set_next_on_line(const RID &p_id, const RID &p_other_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_next_on_line(p_id, p_other_id);
	}
}

void DisplayServer::accessibility_update_set_previous_on_line(const RID &p_id, const RID &p_other_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_previous_on_line(p_id, p_other_id);
	}
}

void DisplayServer::accessibility_update_set_member_of(const RID &p_id, const RID &p_group_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_member_of(p_id, p_group_id);
	}
}

void DisplayServer::accessibility_update_set_in_page_link_target(const RID &p_id, const RID &p_other_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_in_page_link_target(p_id, p_other_id);
	}
}

void DisplayServer::accessibility_update_set_error_message(const RID &p_id, const RID &p_other_id) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_error_message(p_id, p_other_id);
	}
}

void DisplayServer::accessibility_update_set_live(const RID &p_id, DisplayServer::AccessibilityLiveMode p_live) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_live(p_id, p_live);
	}
}

void DisplayServer::accessibility_update_add_action(const RID &p_id, DisplayServer::AccessibilityAction p_action, const Callable &p_callable) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_add_action(p_id, p_action, p_callable);
	}
}

void DisplayServer::accessibility_update_add_custom_action(const RID &p_id, int p_action_id, const String &p_action_description) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_add_custom_action(p_id, p_action_id, p_action_description);
	}
}

void DisplayServer::accessibility_update_set_table_row_count(const RID &p_id, int p_count) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_table_row_count(p_id, p_count);
	}
}

void DisplayServer::accessibility_update_set_table_column_count(const RID &p_id, int p_count) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_table_column_count(p_id, p_count);
	}
}

void DisplayServer::accessibility_update_set_table_row_index(const RID &p_id, int p_index) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_table_row_index(p_id, p_index);
	}
}

void DisplayServer::accessibility_update_set_table_column_index(const RID &p_id, int p_index) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_table_column_index(p_id, p_index);
	}
}

void DisplayServer::accessibility_update_set_table_cell_position(const RID &p_id, int p_row_index, int p_column_index) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_table_cell_position(p_id, p_row_index, p_column_index);
	}
}

void DisplayServer::accessibility_update_set_table_cell_span(const RID &p_id, int p_row_span, int p_column_span) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_table_cell_span(p_id, p_row_span, p_column_span);
	}
}

void DisplayServer::accessibility_update_set_list_item_count(const RID &p_id, int p_size) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_list_item_count(p_id, p_size);
	}
}

void DisplayServer::accessibility_update_set_list_item_index(const RID &p_id, int p_index) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_list_item_index(p_id, p_index);
	}
}

void DisplayServer::accessibility_update_set_list_item_level(const RID &p_id, int p_level) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_list_item_level(p_id, p_level);
	}
}

void DisplayServer::accessibility_update_set_list_item_selected(const RID &p_id, bool p_selected) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_list_item_selected(p_id, p_selected);
	}
}

void DisplayServer::accessibility_update_set_list_item_expanded(const RID &p_id, bool p_expanded) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_list_item_expanded(p_id, p_expanded);
	}
}

void DisplayServer::accessibility_update_set_popup_type(const RID &p_id, DisplayServer::AccessibilityPopupType p_popup) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_popup_type(p_id, p_popup);
	}
}

void DisplayServer::accessibility_update_set_checked(const RID &p_id, bool p_checekd) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_checked(p_id, p_checekd);
	}
}

void DisplayServer::accessibility_update_set_num_value(const RID &p_id, double p_position) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_num_value(p_id, p_position);
	}
}

void DisplayServer::accessibility_update_set_num_range(const RID &p_id, double p_min, double p_max) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_num_range(p_id, p_min, p_max);
	}
}

void DisplayServer::accessibility_update_set_num_step(const RID &p_id, double p_step) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_num_step(p_id, p_step);
	}
}

void DisplayServer::accessibility_update_set_num_jump(const RID &p_id, double p_jump) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_num_jump(p_id, p_jump);
	}
}

void DisplayServer::accessibility_update_set_scroll_x(const RID &p_id, double p_position) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_scroll_x(p_id, p_position);
	}
}

void DisplayServer::accessibility_update_set_scroll_x_range(const RID &p_id, double p_min, double p_max) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_scroll_x_range(p_id, p_min, p_max);
	}
}

void DisplayServer::accessibility_update_set_scroll_y(const RID &p_id, double p_position) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_scroll_y(p_id, p_position);
	}
}

void DisplayServer::accessibility_update_set_scroll_y_range(const RID &p_id, double p_min, double p_max) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_scroll_y_range(p_id, p_min, p_max);
	}
}

void DisplayServer::accessibility_update_set_text_decorations(const RID &p_id, bool p_underline, bool p_strikethrough, bool p_overline) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_text_decorations(p_id, p_underline, p_strikethrough, p_overline);
	}
}

void DisplayServer::accessibility_update_set_text_align(const RID &p_id, HorizontalAlignment p_align) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_text_align(p_id, p_align);
	}
}

void DisplayServer::accessibility_update_set_text_selection(const RID &p_id, const RID &p_text_start_id, int p_start_char, const RID &p_text_end_id, int p_end_char) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_text_selection(p_id, p_text_start_id, p_start_char, p_text_end_id, p_end_char);
	}
}

void DisplayServer::accessibility_update_set_flag(const RID &p_id, DisplayServer::AccessibilityFlags p_flag, bool p_value) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_flag(p_id, p_flag, p_value);
	}
}

void DisplayServer::accessibility_update_set_classname(const RID &p_id, const String &p_classname) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_classname(p_id, p_classname);
	}
}

void DisplayServer::accessibility_update_set_placeholder(const RID &p_id, const String &p_placeholder) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_placeholder(p_id, p_placeholder);
	}
}

void DisplayServer::accessibility_update_set_language(const RID &p_id, const String &p_language) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_language(p_id, p_language);
	}
}

void DisplayServer::accessibility_update_set_text_orientation(const RID &p_id, bool p_vertical) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_text_orientation(p_id, p_vertical);
	}
}

void DisplayServer::accessibility_update_set_list_orientation(const RID &p_id, bool p_vertical) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_list_orientation(p_id, p_vertical);
	}
}

void DisplayServer::accessibility_update_set_shortcut(const RID &p_id, const String &p_shortcut) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_shortcut(p_id, p_shortcut);
	}
}

void DisplayServer::accessibility_update_set_url(const RID &p_id, const String &p_url) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_url(p_id, p_url);
	}
}

void DisplayServer::accessibility_update_set_role_description(const RID &p_id, const String &p_description) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_role_description(p_id, p_description);
	}
}

void DisplayServer::accessibility_update_set_state_description(const RID &p_id, const String &p_description) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_state_description(p_id, p_description);
	}
}

void DisplayServer::accessibility_update_set_color_value(const RID &p_id, const Color &p_color) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_color_value(p_id, p_color);
	}
}

void DisplayServer::accessibility_update_set_background_color(const RID &p_id, const Color &p_color) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_background_color(p_id, p_color);
	}
}

void DisplayServer::accessibility_update_set_foreground_color(const RID &p_id, const Color &p_color) {
	if (accessibility_driver) {
		accessibility_driver->accessibility_update_set_foreground_color(p_id, p_color);
	}
}

Point2i DisplayServer::ime_get_selection() const {
	ERR_FAIL_V_MSG(Point2i(), "IME or NOTIFICATION_WM_IME_UPDATE not supported by this display server.");
}

String DisplayServer::ime_get_text() const {
	ERR_FAIL_V_MSG(String(), "IME or NOTIFICATION_WM_IME_UPDATE not supported by this display server.");
}

void DisplayServer::virtual_keyboard_show(const String &p_existing_text, const Rect2 &p_screen_rect, VirtualKeyboardType p_type, int p_max_length, int p_cursor_start, int p_cursor_end) {
	WARN_PRINT("Virtual keyboard not supported by this display server.");
}

void DisplayServer::virtual_keyboard_hide() {
	WARN_PRINT("Virtual keyboard not supported by this display server.");
}

// returns height of the currently shown keyboard (0 if keyboard is hidden)
int DisplayServer::virtual_keyboard_get_height() const {
	WARN_PRINT("Virtual keyboard not supported by this display server.");
	return 0;
}

bool DisplayServer::has_hardware_keyboard() const {
	return true;
}

void DisplayServer::cursor_set_shape(CursorShape p_shape) {
	WARN_PRINT("Cursor shape not supported by this display server.");
}

DisplayServer::CursorShape DisplayServer::cursor_get_shape() const {
	return CURSOR_ARROW;
}

void DisplayServer::cursor_set_custom_image(const Ref<Resource> &p_cursor, CursorShape p_shape, const Vector2 &p_hotspot) {
	WARN_PRINT("Custom cursor shape not supported by this display server.");
}

bool DisplayServer::get_swap_cancel_ok() {
	return false;
}

void DisplayServer::enable_for_stealing_focus(OS::ProcessID pid) {
}

Error DisplayServer::embed_process(WindowID p_window, OS::ProcessID p_pid, const Rect2i &p_rect, bool p_visible, bool p_grab_focus) {
	WARN_PRINT("Embedded process not supported by this display server.");
	return ERR_UNAVAILABLE;
}

Error DisplayServer::request_close_embedded_process(OS::ProcessID p_pid) {
	WARN_PRINT("Embedded process not supported by this display server.");
	return ERR_UNAVAILABLE;
}

Error DisplayServer::remove_embedded_process(OS::ProcessID p_pid) {
	WARN_PRINT("Embedded process not supported by this display server.");
	return ERR_UNAVAILABLE;
}

OS::ProcessID DisplayServer::get_focused_process_id() {
	WARN_PRINT("Embedded process not supported by this display server.");
	return 0;
}

Error DisplayServer::dialog_show(String p_title, String p_description, Vector<String> p_buttons, const Callable &p_callback) {
	WARN_PRINT("Native dialogs not supported by this display server.");
	return ERR_UNAVAILABLE;
}

Error DisplayServer::dialog_input_text(String p_title, String p_description, String p_partial, const Callable &p_callback) {
	WARN_PRINT("Native dialogs not supported by this display server.");
	return ERR_UNAVAILABLE;
}

Error DisplayServer::file_dialog_show(const String &p_title, const String &p_current_directory, const String &p_filename, bool p_show_hidden, FileDialogMode p_mode, const Vector<String> &p_filters, const Callable &p_callback, WindowID p_window_id) {
	WARN_PRINT("Native dialogs not supported by this display server.");
	return ERR_UNAVAILABLE;
}

Error DisplayServer::file_dialog_with_options_show(const String &p_title, const String &p_current_directory, const String &p_root, const String &p_filename, bool p_show_hidden, FileDialogMode p_mode, const Vector<String> &p_filters, const TypedArray<Dictionary> &p_options, const Callable &p_callback, WindowID p_window_id) {
	WARN_PRINT("Native dialogs not supported by this display server.");
	return ERR_UNAVAILABLE;
}

void DisplayServer::beep() const {
}

int DisplayServer::keyboard_get_layout_count() const {
	return 0;
}

int DisplayServer::keyboard_get_current_layout() const {
	return -1;
}

void DisplayServer::keyboard_set_current_layout(int p_index) {
}

String DisplayServer::keyboard_get_layout_language(int p_index) const {
	return "";
}

String DisplayServer::keyboard_get_layout_name(int p_index) const {
	return "Not supported";
}

Key DisplayServer::keyboard_get_keycode_from_physical(Key p_keycode) const {
	ERR_FAIL_V_MSG(p_keycode, "Not supported by this display server.");
}

Key DisplayServer::keyboard_get_label_from_physical(Key p_keycode) const {
	ERR_FAIL_V_MSG(p_keycode, "Not supported by this display server.");
}

void DisplayServer::show_emoji_and_symbol_picker() const {
}

bool DisplayServer::color_picker(const Callable &p_callback) {
	return false;
}

void DisplayServer::force_process_and_drop_events() {
}

void DisplayServer::release_rendering_thread() {
	WARN_PRINT("Rendering thread not supported by this display server.");
}

void DisplayServer::swap_buffers() {
	WARN_PRINT("Swap buffers not supported by this display server.");
}

void DisplayServer::set_native_icon(const String &p_filename) {
	WARN_PRINT("Native icon not supported by this display server.");
}

void DisplayServer::set_icon(const Ref<Image> &p_icon) {
	WARN_PRINT("Icon not supported by this display server.");
}

DisplayServer::IndicatorID DisplayServer::create_status_indicator(const Ref<Texture2D> &p_icon, const String &p_tooltip, const Callable &p_callback) {
	WARN_PRINT("Status indicator not supported by this display server.");
	return INVALID_INDICATOR_ID;
}

void DisplayServer::status_indicator_set_icon(IndicatorID p_id, const Ref<Texture2D> &p_icon) {
	WARN_PRINT("Status indicator not supported by this display server.");
}

void DisplayServer::status_indicator_set_tooltip(IndicatorID p_id, const String &p_tooltip) {
	WARN_PRINT("Status indicator not supported by this display server.");
}

void DisplayServer::status_indicator_set_menu(IndicatorID p_id, const RID &p_menu_rid) {
	WARN_PRINT("Status indicator not supported by this display server.");
}

void DisplayServer::status_indicator_set_callback(IndicatorID p_id, const Callable &p_callback) {
	WARN_PRINT("Status indicator not supported by this display server.");
}

Rect2 DisplayServer::status_indicator_get_rect(IndicatorID p_id) const {
	WARN_PRINT("Status indicator not supported by this display server.");
	return Rect2();
}

void DisplayServer::delete_status_indicator(IndicatorID p_id) {
	WARN_PRINT("Status indicator not supported by this display server.");
}

int64_t DisplayServer::window_get_native_handle(HandleType p_handle_type, WindowID p_window) const {
	WARN_PRINT("Native handle not supported by this display server.");
	return 0;
}

void DisplayServer::window_set_vsync_mode(DisplayServer::VSyncMode p_vsync_mode, WindowID p_window) {
	WARN_PRINT("Changing the V-Sync mode is not supported by this display server.");
}

DisplayServer::VSyncMode DisplayServer::window_get_vsync_mode(WindowID p_window) const {
	WARN_PRINT("Changing the V-Sync mode is not supported by this display server.");
	return VSyncMode::VSYNC_ENABLED;
}

DisplayServer::WindowID DisplayServer::get_focused_window() const {
	return MAIN_WINDOW_ID; // Proper value for single windows.
}

void DisplayServer::set_context(Context p_context) {
}

void DisplayServer::register_additional_output(Object *p_object) {
	ObjectID id = p_object->get_instance_id();
	if (!additional_outputs.has(id)) {
		additional_outputs.push_back(id);
	}
}

void DisplayServer::unregister_additional_output(Object *p_object) {
	additional_outputs.erase(p_object->get_instance_id());
}

void DisplayServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_feature", "feature"), &DisplayServer::has_feature);
	ClassDB::bind_method(D_METHOD("get_name"), &DisplayServer::get_name);

	ClassDB::bind_method(D_METHOD("help_set_search_callbacks", "search_callback", "action_callback"), &DisplayServer::help_set_search_callbacks);

#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("global_menu_set_popup_callbacks", "menu_root", "open_callback", "close_callback"), &DisplayServer::global_menu_set_popup_callbacks);
	ClassDB::bind_method(D_METHOD("global_menu_add_submenu_item", "menu_root", "label", "submenu", "index"), &DisplayServer::global_menu_add_submenu_item, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("global_menu_add_item", "menu_root", "label", "callback", "key_callback", "tag", "accelerator", "index"), &DisplayServer::global_menu_add_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("global_menu_add_check_item", "menu_root", "label", "callback", "key_callback", "tag", "accelerator", "index"), &DisplayServer::global_menu_add_check_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("global_menu_add_icon_item", "menu_root", "icon", "label", "callback", "key_callback", "tag", "accelerator", "index"), &DisplayServer::global_menu_add_icon_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("global_menu_add_icon_check_item", "menu_root", "icon", "label", "callback", "key_callback", "tag", "accelerator", "index"), &DisplayServer::global_menu_add_icon_check_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("global_menu_add_radio_check_item", "menu_root", "label", "callback", "key_callback", "tag", "accelerator", "index"), &DisplayServer::global_menu_add_radio_check_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("global_menu_add_icon_radio_check_item", "menu_root", "icon", "label", "callback", "key_callback", "tag", "accelerator", "index"), &DisplayServer::global_menu_add_icon_radio_check_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("global_menu_add_multistate_item", "menu_root", "label", "max_states", "default_state", "callback", "key_callback", "tag", "accelerator", "index"), &DisplayServer::global_menu_add_multistate_item, DEFVAL(Callable()), DEFVAL(Callable()), DEFVAL(Variant()), DEFVAL(Key::NONE), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("global_menu_add_separator", "menu_root", "index"), &DisplayServer::global_menu_add_separator, DEFVAL(-1));

	ClassDB::bind_method(D_METHOD("global_menu_get_item_index_from_text", "menu_root", "text"), &DisplayServer::global_menu_get_item_index_from_text);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_index_from_tag", "menu_root", "tag"), &DisplayServer::global_menu_get_item_index_from_tag);

	ClassDB::bind_method(D_METHOD("global_menu_is_item_checked", "menu_root", "idx"), &DisplayServer::global_menu_is_item_checked);
	ClassDB::bind_method(D_METHOD("global_menu_is_item_checkable", "menu_root", "idx"), &DisplayServer::global_menu_is_item_checkable);
	ClassDB::bind_method(D_METHOD("global_menu_is_item_radio_checkable", "menu_root", "idx"), &DisplayServer::global_menu_is_item_radio_checkable);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_callback", "menu_root", "idx"), &DisplayServer::global_menu_get_item_callback);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_key_callback", "menu_root", "idx"), &DisplayServer::global_menu_get_item_key_callback);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_tag", "menu_root", "idx"), &DisplayServer::global_menu_get_item_tag);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_text", "menu_root", "idx"), &DisplayServer::global_menu_get_item_text);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_submenu", "menu_root", "idx"), &DisplayServer::global_menu_get_item_submenu);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_accelerator", "menu_root", "idx"), &DisplayServer::global_menu_get_item_accelerator);
	ClassDB::bind_method(D_METHOD("global_menu_is_item_disabled", "menu_root", "idx"), &DisplayServer::global_menu_is_item_disabled);
	ClassDB::bind_method(D_METHOD("global_menu_is_item_hidden", "menu_root", "idx"), &DisplayServer::global_menu_is_item_hidden);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_tooltip", "menu_root", "idx"), &DisplayServer::global_menu_get_item_tooltip);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_state", "menu_root", "idx"), &DisplayServer::global_menu_get_item_state);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_max_states", "menu_root", "idx"), &DisplayServer::global_menu_get_item_max_states);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_icon", "menu_root", "idx"), &DisplayServer::global_menu_get_item_icon);
	ClassDB::bind_method(D_METHOD("global_menu_get_item_indentation_level", "menu_root", "idx"), &DisplayServer::global_menu_get_item_indentation_level);

	ClassDB::bind_method(D_METHOD("global_menu_set_item_checked", "menu_root", "idx", "checked"), &DisplayServer::global_menu_set_item_checked);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_checkable", "menu_root", "idx", "checkable"), &DisplayServer::global_menu_set_item_checkable);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_radio_checkable", "menu_root", "idx", "checkable"), &DisplayServer::global_menu_set_item_radio_checkable);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_callback", "menu_root", "idx", "callback"), &DisplayServer::global_menu_set_item_callback);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_hover_callbacks", "menu_root", "idx", "callback"), &DisplayServer::global_menu_set_item_hover_callbacks);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_key_callback", "menu_root", "idx", "key_callback"), &DisplayServer::global_menu_set_item_key_callback);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_tag", "menu_root", "idx", "tag"), &DisplayServer::global_menu_set_item_tag);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_text", "menu_root", "idx", "text"), &DisplayServer::global_menu_set_item_text);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_submenu", "menu_root", "idx", "submenu"), &DisplayServer::global_menu_set_item_submenu);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_accelerator", "menu_root", "idx", "keycode"), &DisplayServer::global_menu_set_item_accelerator);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_disabled", "menu_root", "idx", "disabled"), &DisplayServer::global_menu_set_item_disabled);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_hidden", "menu_root", "idx", "hidden"), &DisplayServer::global_menu_set_item_hidden);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_tooltip", "menu_root", "idx", "tooltip"), &DisplayServer::global_menu_set_item_tooltip);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_state", "menu_root", "idx", "state"), &DisplayServer::global_menu_set_item_state);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_max_states", "menu_root", "idx", "max_states"), &DisplayServer::global_menu_set_item_max_states);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_icon", "menu_root", "idx", "icon"), &DisplayServer::global_menu_set_item_icon);
	ClassDB::bind_method(D_METHOD("global_menu_set_item_indentation_level", "menu_root", "idx", "level"), &DisplayServer::global_menu_set_item_indentation_level);

	ClassDB::bind_method(D_METHOD("global_menu_get_item_count", "menu_root"), &DisplayServer::global_menu_get_item_count);

	ClassDB::bind_method(D_METHOD("global_menu_remove_item", "menu_root", "idx"), &DisplayServer::global_menu_remove_item);
	ClassDB::bind_method(D_METHOD("global_menu_clear", "menu_root"), &DisplayServer::global_menu_clear);

	ClassDB::bind_method(D_METHOD("global_menu_get_system_menu_roots"), &DisplayServer::global_menu_get_system_menu_roots);
#endif

	ClassDB::bind_method(D_METHOD("tts_is_speaking"), &DisplayServer::tts_is_speaking);
	ClassDB::bind_method(D_METHOD("tts_is_paused"), &DisplayServer::tts_is_paused);
	ClassDB::bind_method(D_METHOD("tts_get_voices"), &DisplayServer::tts_get_voices);
	ClassDB::bind_method(D_METHOD("tts_get_voices_for_language", "language"), &DisplayServer::tts_get_voices_for_language);

	ClassDB::bind_method(D_METHOD("tts_speak", "text", "voice", "volume", "pitch", "rate", "utterance_id", "interrupt"), &DisplayServer::tts_speak, DEFVAL(50), DEFVAL(1.f), DEFVAL(1.f), DEFVAL(0), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("tts_pause"), &DisplayServer::tts_pause);
	ClassDB::bind_method(D_METHOD("tts_resume"), &DisplayServer::tts_resume);
	ClassDB::bind_method(D_METHOD("tts_stop"), &DisplayServer::tts_stop);

	ClassDB::bind_method(D_METHOD("tts_set_utterance_callback", "event", "callable"), &DisplayServer::tts_set_utterance_callback);
	ClassDB::bind_method(D_METHOD("_tts_post_utterance_event", "event", "id", "char_pos"), &DisplayServer::tts_post_utterance_event);

	ClassDB::bind_method(D_METHOD("is_dark_mode_supported"), &DisplayServer::is_dark_mode_supported);
	ClassDB::bind_method(D_METHOD("is_dark_mode"), &DisplayServer::is_dark_mode);
	ClassDB::bind_method(D_METHOD("get_accent_color"), &DisplayServer::get_accent_color);
	ClassDB::bind_method(D_METHOD("get_base_color"), &DisplayServer::get_base_color);
	ClassDB::bind_method(D_METHOD("set_system_theme_change_callback", "callable"), &DisplayServer::set_system_theme_change_callback);

	ClassDB::bind_method(D_METHOD("mouse_set_mode", "mouse_mode"), &DisplayServer::mouse_set_mode);
	ClassDB::bind_method(D_METHOD("mouse_get_mode"), &DisplayServer::mouse_get_mode);

	ClassDB::bind_method(D_METHOD("warp_mouse", "position"), &DisplayServer::warp_mouse);
	ClassDB::bind_method(D_METHOD("mouse_get_position"), &DisplayServer::mouse_get_position);
	ClassDB::bind_method(D_METHOD("mouse_get_button_state"), &DisplayServer::mouse_get_button_state);

	ClassDB::bind_method(D_METHOD("clipboard_set", "clipboard"), &DisplayServer::clipboard_set);
	ClassDB::bind_method(D_METHOD("clipboard_get"), &DisplayServer::clipboard_get);
	ClassDB::bind_method(D_METHOD("clipboard_get_image"), &DisplayServer::clipboard_get_image);
	ClassDB::bind_method(D_METHOD("clipboard_has"), &DisplayServer::clipboard_has);
	ClassDB::bind_method(D_METHOD("clipboard_has_image"), &DisplayServer::clipboard_has_image);
	ClassDB::bind_method(D_METHOD("clipboard_set_primary", "clipboard_primary"), &DisplayServer::clipboard_set_primary);
	ClassDB::bind_method(D_METHOD("clipboard_get_primary"), &DisplayServer::clipboard_get_primary);

	ClassDB::bind_method(D_METHOD("get_display_cutouts"), &DisplayServer::get_display_cutouts);
	ClassDB::bind_method(D_METHOD("get_display_safe_area"), &DisplayServer::get_display_safe_area);

	ClassDB::bind_method(D_METHOD("get_screen_count"), &DisplayServer::get_screen_count);
	ClassDB::bind_method(D_METHOD("get_primary_screen"), &DisplayServer::get_primary_screen);
	ClassDB::bind_method(D_METHOD("get_keyboard_focus_screen"), &DisplayServer::get_keyboard_focus_screen);
	ClassDB::bind_method(D_METHOD("get_screen_from_rect", "rect"), &DisplayServer::get_screen_from_rect);
	ClassDB::bind_method(D_METHOD("screen_get_position", "screen"), &DisplayServer::screen_get_position, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_size", "screen"), &DisplayServer::screen_get_size, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_usable_rect", "screen"), &DisplayServer::screen_get_usable_rect, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_dpi", "screen"), &DisplayServer::screen_get_dpi, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_scale", "screen"), &DisplayServer::screen_get_scale, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("is_touchscreen_available"), &DisplayServer::is_touchscreen_available);
	ClassDB::bind_method(D_METHOD("screen_get_max_scale"), &DisplayServer::screen_get_max_scale);
	ClassDB::bind_method(D_METHOD("screen_get_refresh_rate", "screen"), &DisplayServer::screen_get_refresh_rate, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_pixel", "position"), &DisplayServer::screen_get_pixel);
	ClassDB::bind_method(D_METHOD("screen_get_image", "screen"), &DisplayServer::screen_get_image, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_image_rect", "rect"), &DisplayServer::screen_get_image_rect);

	ClassDB::bind_method(D_METHOD("screen_set_orientation", "orientation", "screen"), &DisplayServer::screen_set_orientation, DEFVAL(SCREEN_OF_MAIN_WINDOW));
	ClassDB::bind_method(D_METHOD("screen_get_orientation", "screen"), &DisplayServer::screen_get_orientation, DEFVAL(SCREEN_OF_MAIN_WINDOW));

	ClassDB::bind_method(D_METHOD("screen_set_keep_on", "enable"), &DisplayServer::screen_set_keep_on);
	ClassDB::bind_method(D_METHOD("screen_is_kept_on"), &DisplayServer::screen_is_kept_on);

	ClassDB::bind_method(D_METHOD("get_window_list"), &DisplayServer::get_window_list);
	ClassDB::bind_method(D_METHOD("get_window_at_screen_position", "position"), &DisplayServer::get_window_at_screen_position);

	ClassDB::bind_method(D_METHOD("window_get_native_handle", "handle_type", "window_id"), &DisplayServer::window_get_native_handle, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_get_active_popup"), &DisplayServer::window_get_active_popup);
	ClassDB::bind_method(D_METHOD("window_set_popup_safe_rect", "window", "rect"), &DisplayServer::window_set_popup_safe_rect);
	ClassDB::bind_method(D_METHOD("window_get_popup_safe_rect", "window"), &DisplayServer::window_get_popup_safe_rect);

	ClassDB::bind_method(D_METHOD("window_set_title", "title", "window_id"), &DisplayServer::window_set_title, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_get_title_size", "title", "window_id"), &DisplayServer::window_get_title_size, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_mouse_passthrough", "region", "window_id"), &DisplayServer::window_set_mouse_passthrough, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_current_screen", "window_id"), &DisplayServer::window_get_current_screen, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_current_screen", "screen", "window_id"), &DisplayServer::window_set_current_screen, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_position", "window_id"), &DisplayServer::window_get_position, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_get_position_with_decorations", "window_id"), &DisplayServer::window_get_position_with_decorations, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_position", "position", "window_id"), &DisplayServer::window_set_position, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_size", "window_id"), &DisplayServer::window_get_size, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_size", "size", "window_id"), &DisplayServer::window_set_size, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_rect_changed_callback", "callback", "window_id"), &DisplayServer::window_set_rect_changed_callback, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_window_event_callback", "callback", "window_id"), &DisplayServer::window_set_window_event_callback, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_input_event_callback", "callback", "window_id"), &DisplayServer::window_set_input_event_callback, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_input_text_callback", "callback", "window_id"), &DisplayServer::window_set_input_text_callback, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_drop_files_callback", "callback", "window_id"), &DisplayServer::window_set_drop_files_callback, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_attached_instance_id", "window_id"), &DisplayServer::window_get_attached_instance_id, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_max_size", "window_id"), &DisplayServer::window_get_max_size, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_max_size", "max_size", "window_id"), &DisplayServer::window_set_max_size, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_min_size", "window_id"), &DisplayServer::window_get_min_size, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_min_size", "min_size", "window_id"), &DisplayServer::window_set_min_size, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_size_with_decorations", "window_id"), &DisplayServer::window_get_size_with_decorations, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_get_mode", "window_id"), &DisplayServer::window_get_mode, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_mode", "mode", "window_id"), &DisplayServer::window_set_mode, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_set_flag", "flag", "enabled", "window_id"), &DisplayServer::window_set_flag, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_get_flag", "flag", "window_id"), &DisplayServer::window_get_flag, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_set_window_buttons_offset", "offset", "window_id"), &DisplayServer::window_set_window_buttons_offset, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_get_safe_title_margins", "window_id"), &DisplayServer::window_get_safe_title_margins, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_request_attention", "window_id"), &DisplayServer::window_request_attention, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_move_to_foreground", "window_id"), &DisplayServer::window_move_to_foreground, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_is_focused", "window_id"), &DisplayServer::window_is_focused, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_can_draw", "window_id"), &DisplayServer::window_can_draw, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_set_transient", "window_id", "parent_window_id"), &DisplayServer::window_set_transient);
	ClassDB::bind_method(D_METHOD("window_set_exclusive", "window_id", "exclusive"), &DisplayServer::window_set_exclusive);

	ClassDB::bind_method(D_METHOD("window_set_ime_active", "active", "window_id"), &DisplayServer::window_set_ime_active, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_set_ime_position", "position", "window_id"), &DisplayServer::window_set_ime_position, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_set_vsync_mode", "vsync_mode", "window_id"), &DisplayServer::window_set_vsync_mode, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_get_vsync_mode", "window_id"), &DisplayServer::window_get_vsync_mode, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("window_is_maximize_allowed", "window_id"), &DisplayServer::window_is_maximize_allowed, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_maximize_on_title_dbl_click"), &DisplayServer::window_maximize_on_title_dbl_click);
	ClassDB::bind_method(D_METHOD("window_minimize_on_title_dbl_click"), &DisplayServer::window_minimize_on_title_dbl_click);

	ClassDB::bind_method(D_METHOD("window_start_drag", "window_id"), &DisplayServer::window_start_drag, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("window_start_resize", "edge", "window_id"), &DisplayServer::window_start_resize, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("accessibility_should_increase_contrast"), &DisplayServer::accessibility_should_increase_contrast);
	ClassDB::bind_method(D_METHOD("accessibility_should_reduce_animation"), &DisplayServer::accessibility_should_reduce_animation);
	ClassDB::bind_method(D_METHOD("accessibility_should_reduce_transparency"), &DisplayServer::accessibility_should_reduce_transparency);
	ClassDB::bind_method(D_METHOD("accessibility_screen_reader_active"), &DisplayServer::accessibility_screen_reader_active);

	ClassDB::bind_method(D_METHOD("accessibility_create_element", "window_id", "role"), &DisplayServer::accessibility_create_element);
	ClassDB::bind_method(D_METHOD("accessibility_create_sub_element", "parent_rid", "role", "insert_pos"), &DisplayServer::accessibility_create_sub_element, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("accessibility_create_sub_text_edit_elements", "parent_rid", "shaped_text", "min_height", "insert_pos"), &DisplayServer::accessibility_create_sub_text_edit_elements, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("accessibility_has_element", "id"), &DisplayServer::accessibility_has_element);
	ClassDB::bind_method(D_METHOD("accessibility_free_element", "id"), &DisplayServer::accessibility_free_element);
	ClassDB::bind_method(D_METHOD("accessibility_element_set_meta", "id", "meta"), &DisplayServer::accessibility_element_set_meta);
	ClassDB::bind_method(D_METHOD("accessibility_element_get_meta", "id"), &DisplayServer::accessibility_element_get_meta);

	ClassDB::bind_method(D_METHOD("_accessibility_update_if_active", "callback"), &DisplayServer::accessibility_update_if_active);

	ClassDB::bind_method(D_METHOD("accessibility_set_window_rect", "window_id", "rect_out", "rect_in"), &DisplayServer::accessibility_set_window_rect);
	ClassDB::bind_method(D_METHOD("accessibility_set_window_focused", "window_id", "focused"), &DisplayServer::accessibility_set_window_focused);

	ClassDB::bind_method(D_METHOD("accessibility_update_set_focus", "id"), &DisplayServer::accessibility_update_set_focus);
	ClassDB::bind_method(D_METHOD("accessibility_get_window_root", "window_id"), &DisplayServer::accessibility_get_window_root);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_role", "id", "role"), &DisplayServer::accessibility_update_set_role);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_name", "id", "name"), &DisplayServer::accessibility_update_set_name);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_extra_info", "id", "name"), &DisplayServer::accessibility_update_set_extra_info);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_description", "id", "description"), &DisplayServer::accessibility_update_set_description);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_value", "id", "value"), &DisplayServer::accessibility_update_set_value);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_tooltip", "id", "tooltip"), &DisplayServer::accessibility_update_set_tooltip);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_bounds", "id", "p_rect"), &DisplayServer::accessibility_update_set_bounds);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_transform", "id", "transform"), &DisplayServer::accessibility_update_set_transform);
	ClassDB::bind_method(D_METHOD("accessibility_update_add_child", "id", "child_id"), &DisplayServer::accessibility_update_add_child);
	ClassDB::bind_method(D_METHOD("accessibility_update_add_related_controls", "id", "related_id"), &DisplayServer::accessibility_update_add_related_controls);
	ClassDB::bind_method(D_METHOD("accessibility_update_add_related_details", "id", "related_id"), &DisplayServer::accessibility_update_add_related_details);
	ClassDB::bind_method(D_METHOD("accessibility_update_add_related_described_by", "id", "related_id"), &DisplayServer::accessibility_update_add_related_described_by);
	ClassDB::bind_method(D_METHOD("accessibility_update_add_related_flow_to", "id", "related_id"), &DisplayServer::accessibility_update_add_related_flow_to);
	ClassDB::bind_method(D_METHOD("accessibility_update_add_related_labeled_by", "id", "related_id"), &DisplayServer::accessibility_update_add_related_labeled_by);
	ClassDB::bind_method(D_METHOD("accessibility_update_add_related_radio_group", "id", "related_id"), &DisplayServer::accessibility_update_add_related_radio_group);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_active_descendant", "id", "other_id"), &DisplayServer::accessibility_update_set_active_descendant);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_next_on_line", "id", "other_id"), &DisplayServer::accessibility_update_set_next_on_line);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_previous_on_line", "id", "other_id"), &DisplayServer::accessibility_update_set_previous_on_line);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_member_of", "id", "group_id"), &DisplayServer::accessibility_update_set_member_of);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_in_page_link_target", "id", "other_id"), &DisplayServer::accessibility_update_set_in_page_link_target);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_error_message", "id", "other_id"), &DisplayServer::accessibility_update_set_error_message);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_live", "id", "live"), &DisplayServer::accessibility_update_set_live);
	ClassDB::bind_method(D_METHOD("accessibility_update_add_action", "id", "action", "callable"), &DisplayServer::accessibility_update_add_action);
	ClassDB::bind_method(D_METHOD("accessibility_update_add_custom_action", "id", "action_id", "action_description"), &DisplayServer::accessibility_update_add_custom_action);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_table_row_count", "id", "count"), &DisplayServer::accessibility_update_set_table_row_count);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_table_column_count", "id", "count"), &DisplayServer::accessibility_update_set_table_column_count);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_table_row_index", "id", "index"), &DisplayServer::accessibility_update_set_table_row_index);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_table_column_index", "id", "index"), &DisplayServer::accessibility_update_set_table_column_index);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_table_cell_position", "id", "row_index", "column_index"), &DisplayServer::accessibility_update_set_table_cell_position);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_table_cell_span", "id", "row_span", "column_span"), &DisplayServer::accessibility_update_set_table_cell_span);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_list_item_count", "id", "size"), &DisplayServer::accessibility_update_set_list_item_count);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_list_item_index", "id", "index"), &DisplayServer::accessibility_update_set_list_item_index);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_list_item_level", "id", "level"), &DisplayServer::accessibility_update_set_list_item_level);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_list_item_selected", "id", "selected"), &DisplayServer::accessibility_update_set_list_item_selected);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_list_item_expanded", "id", "expanded"), &DisplayServer::accessibility_update_set_list_item_expanded);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_popup_type", "id", "popup"), &DisplayServer::accessibility_update_set_popup_type);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_checked", "id", "checekd"), &DisplayServer::accessibility_update_set_checked);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_num_value", "id", "position"), &DisplayServer::accessibility_update_set_num_value);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_num_range", "id", "min", "max"), &DisplayServer::accessibility_update_set_num_range);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_num_step", "id", "step"), &DisplayServer::accessibility_update_set_num_step);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_num_jump", "id", "jump"), &DisplayServer::accessibility_update_set_num_jump);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_scroll_x", "id", "position"), &DisplayServer::accessibility_update_set_scroll_x);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_scroll_x_range", "id", "min", "max"), &DisplayServer::accessibility_update_set_scroll_x_range);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_scroll_y", "id", "position"), &DisplayServer::accessibility_update_set_scroll_y);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_scroll_y_range", "id", "min", "max"), &DisplayServer::accessibility_update_set_scroll_y_range);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_text_decorations", "id", "underline", "strikethrough", "overline"), &DisplayServer::accessibility_update_set_text_decorations);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_text_align", "id", "align"), &DisplayServer::accessibility_update_set_text_align);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_text_selection", "id", "text_start_id", "start_char", "text_end_id", "end_char"), &DisplayServer::accessibility_update_set_text_selection);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_flag", "id", "flag", "value"), &DisplayServer::accessibility_update_set_flag);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_classname", "id", "classname"), &DisplayServer::accessibility_update_set_classname);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_placeholder", "id", "placeholder"), &DisplayServer::accessibility_update_set_placeholder);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_language", "id", "language"), &DisplayServer::accessibility_update_set_language);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_text_orientation", "id", "vertical"), &DisplayServer::accessibility_update_set_text_orientation);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_list_orientation", "id", "vertical"), &DisplayServer::accessibility_update_set_list_orientation);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_shortcut", "id", "shortcut"), &DisplayServer::accessibility_update_set_shortcut);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_url", "id", "url"), &DisplayServer::accessibility_update_set_url);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_role_description", "id", "description"), &DisplayServer::accessibility_update_set_role_description);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_state_description", "id", "description"), &DisplayServer::accessibility_update_set_state_description);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_color_value", "id", "color"), &DisplayServer::accessibility_update_set_color_value);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_background_color", "id", "color"), &DisplayServer::accessibility_update_set_background_color);
	ClassDB::bind_method(D_METHOD("accessibility_update_set_foreground_color", "id", "color"), &DisplayServer::accessibility_update_set_foreground_color);

	ClassDB::bind_method(D_METHOD("ime_get_selection"), &DisplayServer::ime_get_selection);
	ClassDB::bind_method(D_METHOD("ime_get_text"), &DisplayServer::ime_get_text);

	ClassDB::bind_method(D_METHOD("virtual_keyboard_show", "existing_text", "position", "type", "max_length", "cursor_start", "cursor_end"), &DisplayServer::virtual_keyboard_show, DEFVAL(Rect2()), DEFVAL(KEYBOARD_TYPE_DEFAULT), DEFVAL(-1), DEFVAL(-1), DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("virtual_keyboard_hide"), &DisplayServer::virtual_keyboard_hide);

	ClassDB::bind_method(D_METHOD("virtual_keyboard_get_height"), &DisplayServer::virtual_keyboard_get_height);

	ClassDB::bind_method(D_METHOD("has_hardware_keyboard"), &DisplayServer::has_hardware_keyboard);
	ClassDB::bind_method(D_METHOD("set_hardware_keyboard_connection_change_callback", "callable"), &DisplayServer::set_hardware_keyboard_connection_change_callback);

	ClassDB::bind_method(D_METHOD("cursor_set_shape", "shape"), &DisplayServer::cursor_set_shape);
	ClassDB::bind_method(D_METHOD("cursor_get_shape"), &DisplayServer::cursor_get_shape);
	ClassDB::bind_method(D_METHOD("cursor_set_custom_image", "cursor", "shape", "hotspot"), &DisplayServer::cursor_set_custom_image, DEFVAL(CURSOR_ARROW), DEFVAL(Vector2()));

	ClassDB::bind_method(D_METHOD("get_swap_cancel_ok"), &DisplayServer::get_swap_cancel_ok);

	ClassDB::bind_method(D_METHOD("enable_for_stealing_focus", "process_id"), &DisplayServer::enable_for_stealing_focus);

	ClassDB::bind_method(D_METHOD("dialog_show", "title", "description", "buttons", "callback"), &DisplayServer::dialog_show);
	ClassDB::bind_method(D_METHOD("dialog_input_text", "title", "description", "existing_text", "callback"), &DisplayServer::dialog_input_text);

	ClassDB::bind_method(D_METHOD("file_dialog_show", "title", "current_directory", "filename", "show_hidden", "mode", "filters", "callback", "parent_window_id"), &DisplayServer::file_dialog_show, DEFVAL(MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("file_dialog_with_options_show", "title", "current_directory", "root", "filename", "show_hidden", "mode", "filters", "options", "callback", "parent_window_id"), &DisplayServer::file_dialog_with_options_show, DEFVAL(MAIN_WINDOW_ID));

	ClassDB::bind_method(D_METHOD("beep"), &DisplayServer::beep);

	ClassDB::bind_method(D_METHOD("keyboard_get_layout_count"), &DisplayServer::keyboard_get_layout_count);
	ClassDB::bind_method(D_METHOD("keyboard_get_current_layout"), &DisplayServer::keyboard_get_current_layout);
	ClassDB::bind_method(D_METHOD("keyboard_set_current_layout", "index"), &DisplayServer::keyboard_set_current_layout);
	ClassDB::bind_method(D_METHOD("keyboard_get_layout_language", "index"), &DisplayServer::keyboard_get_layout_language);
	ClassDB::bind_method(D_METHOD("keyboard_get_layout_name", "index"), &DisplayServer::keyboard_get_layout_name);
	ClassDB::bind_method(D_METHOD("keyboard_get_keycode_from_physical", "keycode"), &DisplayServer::keyboard_get_keycode_from_physical);
	ClassDB::bind_method(D_METHOD("keyboard_get_label_from_physical", "keycode"), &DisplayServer::keyboard_get_label_from_physical);

	ClassDB::bind_method(D_METHOD("show_emoji_and_symbol_picker"), &DisplayServer::show_emoji_and_symbol_picker);
	ClassDB::bind_method(D_METHOD("color_picker", "callback"), &DisplayServer::color_picker);

	ClassDB::bind_method(D_METHOD("process_events"), &DisplayServer::process_events);
	ClassDB::bind_method(D_METHOD("force_process_and_drop_events"), &DisplayServer::force_process_and_drop_events);

	ClassDB::bind_method(D_METHOD("set_native_icon", "filename"), &DisplayServer::set_native_icon);
	ClassDB::bind_method(D_METHOD("set_icon", "image"), &DisplayServer::set_icon);

	ClassDB::bind_method(D_METHOD("create_status_indicator", "icon", "tooltip", "callback"), &DisplayServer::create_status_indicator);
	ClassDB::bind_method(D_METHOD("status_indicator_set_icon", "id", "icon"), &DisplayServer::status_indicator_set_icon);
	ClassDB::bind_method(D_METHOD("status_indicator_set_tooltip", "id", "tooltip"), &DisplayServer::status_indicator_set_tooltip);
	ClassDB::bind_method(D_METHOD("status_indicator_set_menu", "id", "menu_rid"), &DisplayServer::status_indicator_set_menu);
	ClassDB::bind_method(D_METHOD("status_indicator_set_callback", "id", "callback"), &DisplayServer::status_indicator_set_callback);
	ClassDB::bind_method(D_METHOD("status_indicator_get_rect", "id"), &DisplayServer::status_indicator_get_rect);
	ClassDB::bind_method(D_METHOD("delete_status_indicator", "id"), &DisplayServer::delete_status_indicator);

	ClassDB::bind_method(D_METHOD("tablet_get_driver_count"), &DisplayServer::tablet_get_driver_count);
	ClassDB::bind_method(D_METHOD("tablet_get_driver_name", "idx"), &DisplayServer::tablet_get_driver_name);
	ClassDB::bind_method(D_METHOD("tablet_get_current_driver"), &DisplayServer::tablet_get_current_driver);
	ClassDB::bind_method(D_METHOD("tablet_set_current_driver", "name"), &DisplayServer::tablet_set_current_driver);

	ClassDB::bind_method(D_METHOD("is_window_transparency_available"), &DisplayServer::is_window_transparency_available);

	ClassDB::bind_method(D_METHOD("register_additional_output", "object"), &DisplayServer::register_additional_output);
	ClassDB::bind_method(D_METHOD("unregister_additional_output", "object"), &DisplayServer::unregister_additional_output);
	ClassDB::bind_method(D_METHOD("has_additional_outputs"), &DisplayServer::has_additional_outputs);

#ifndef DISABLE_DEPRECATED
	BIND_ENUM_CONSTANT(FEATURE_GLOBAL_MENU);
#endif
	BIND_ENUM_CONSTANT(FEATURE_SUBWINDOWS);
	BIND_ENUM_CONSTANT(FEATURE_TOUCHSCREEN);
	BIND_ENUM_CONSTANT(FEATURE_MOUSE);
	BIND_ENUM_CONSTANT(FEATURE_MOUSE_WARP);
	BIND_ENUM_CONSTANT(FEATURE_CLIPBOARD);
	BIND_ENUM_CONSTANT(FEATURE_VIRTUAL_KEYBOARD);
	BIND_ENUM_CONSTANT(FEATURE_CURSOR_SHAPE);
	BIND_ENUM_CONSTANT(FEATURE_CUSTOM_CURSOR_SHAPE);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_DIALOG);
	BIND_ENUM_CONSTANT(FEATURE_IME);
	BIND_ENUM_CONSTANT(FEATURE_WINDOW_TRANSPARENCY);
	BIND_ENUM_CONSTANT(FEATURE_HIDPI);
	BIND_ENUM_CONSTANT(FEATURE_ICON);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_ICON);
	BIND_ENUM_CONSTANT(FEATURE_ORIENTATION);
	BIND_ENUM_CONSTANT(FEATURE_SWAP_BUFFERS);
	BIND_ENUM_CONSTANT(FEATURE_CLIPBOARD_PRIMARY);
	BIND_ENUM_CONSTANT(FEATURE_TEXT_TO_SPEECH);
	BIND_ENUM_CONSTANT(FEATURE_EXTEND_TO_TITLE);
	BIND_ENUM_CONSTANT(FEATURE_SCREEN_CAPTURE);
	BIND_ENUM_CONSTANT(FEATURE_STATUS_INDICATOR);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_HELP);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_DIALOG_INPUT);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_DIALOG_FILE);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_DIALOG_FILE_EXTRA);
	BIND_ENUM_CONSTANT(FEATURE_WINDOW_DRAG);
	BIND_ENUM_CONSTANT(FEATURE_SCREEN_EXCLUDE_FROM_CAPTURE);
	BIND_ENUM_CONSTANT(FEATURE_WINDOW_EMBEDDING);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_DIALOG_FILE_MIME);
	BIND_ENUM_CONSTANT(FEATURE_EMOJI_AND_SYMBOL_PICKER);
	BIND_ENUM_CONSTANT(FEATURE_NATIVE_COLOR_PICKER);
	BIND_ENUM_CONSTANT(FEATURE_SELF_FITTING_WINDOWS);
	BIND_ENUM_CONSTANT(FEATURE_ACCESSIBILITY_SCREEN_READER);

	BIND_ENUM_CONSTANT(ROLE_UNKNOWN);
	BIND_ENUM_CONSTANT(ROLE_DEFAULT_BUTTON);
	BIND_ENUM_CONSTANT(ROLE_AUDIO);
	BIND_ENUM_CONSTANT(ROLE_VIDEO);
	BIND_ENUM_CONSTANT(ROLE_STATIC_TEXT);
	BIND_ENUM_CONSTANT(ROLE_CONTAINER);
	BIND_ENUM_CONSTANT(ROLE_PANEL);
	BIND_ENUM_CONSTANT(ROLE_BUTTON);
	BIND_ENUM_CONSTANT(ROLE_LINK);
	BIND_ENUM_CONSTANT(ROLE_CHECK_BOX);
	BIND_ENUM_CONSTANT(ROLE_RADIO_BUTTON);
	BIND_ENUM_CONSTANT(ROLE_CHECK_BUTTON);
	BIND_ENUM_CONSTANT(ROLE_SCROLL_BAR);
	BIND_ENUM_CONSTANT(ROLE_SCROLL_VIEW);
	BIND_ENUM_CONSTANT(ROLE_SPLITTER);
	BIND_ENUM_CONSTANT(ROLE_SLIDER);
	BIND_ENUM_CONSTANT(ROLE_SPIN_BUTTON);
	BIND_ENUM_CONSTANT(ROLE_PROGRESS_INDICATOR);
	BIND_ENUM_CONSTANT(ROLE_TEXT_FIELD);
	BIND_ENUM_CONSTANT(ROLE_MULTILINE_TEXT_FIELD);
	BIND_ENUM_CONSTANT(ROLE_COLOR_PICKER);
	BIND_ENUM_CONSTANT(ROLE_TABLE);
	BIND_ENUM_CONSTANT(ROLE_CELL);
	BIND_ENUM_CONSTANT(ROLE_ROW);
	BIND_ENUM_CONSTANT(ROLE_ROW_GROUP);
	BIND_ENUM_CONSTANT(ROLE_ROW_HEADER);
	BIND_ENUM_CONSTANT(ROLE_COLUMN_HEADER);
	BIND_ENUM_CONSTANT(ROLE_TREE);
	BIND_ENUM_CONSTANT(ROLE_TREE_ITEM);
	BIND_ENUM_CONSTANT(ROLE_LIST);
	BIND_ENUM_CONSTANT(ROLE_LIST_ITEM);
	BIND_ENUM_CONSTANT(ROLE_LIST_BOX);
	BIND_ENUM_CONSTANT(ROLE_LIST_BOX_OPTION);
	BIND_ENUM_CONSTANT(ROLE_TAB_BAR);
	BIND_ENUM_CONSTANT(ROLE_TAB);
	BIND_ENUM_CONSTANT(ROLE_TAB_PANEL);
	BIND_ENUM_CONSTANT(ROLE_MENU_BAR);
	BIND_ENUM_CONSTANT(ROLE_MENU);
	BIND_ENUM_CONSTANT(ROLE_MENU_ITEM);
	BIND_ENUM_CONSTANT(ROLE_MENU_ITEM_CHECK_BOX);
	BIND_ENUM_CONSTANT(ROLE_MENU_ITEM_RADIO);
	BIND_ENUM_CONSTANT(ROLE_IMAGE);
	BIND_ENUM_CONSTANT(ROLE_WINDOW);
	BIND_ENUM_CONSTANT(ROLE_TITLE_BAR);
	BIND_ENUM_CONSTANT(ROLE_DIALOG);
	BIND_ENUM_CONSTANT(ROLE_TOOLTIP);

	BIND_ENUM_CONSTANT(POPUP_UNKNOWN);
	BIND_ENUM_CONSTANT(POPUP_MENU);
	BIND_ENUM_CONSTANT(POPUP_LIST);
	BIND_ENUM_CONSTANT(POPUP_TREE);
	BIND_ENUM_CONSTANT(POPUP_DIALOG);

	BIND_ENUM_CONSTANT(FLAG_HIDDEN);
	BIND_ENUM_CONSTANT(FLAG_LINKED);
	BIND_ENUM_CONSTANT(FLAG_MULTISELECTABLE);
	BIND_ENUM_CONSTANT(FLAG_REQUIRED);
	BIND_ENUM_CONSTANT(FLAG_VISITED);
	BIND_ENUM_CONSTANT(FLAG_BUSY);
	BIND_ENUM_CONSTANT(FLAG_MODAL);
	BIND_ENUM_CONSTANT(FLAG_TOUCH_PASSTHROUGH);
	BIND_ENUM_CONSTANT(FLAG_READONLY);
	BIND_ENUM_CONSTANT(FLAG_DISABLED);
	BIND_ENUM_CONSTANT(FLAG_CLIPS_CHILDREN);

	BIND_ENUM_CONSTANT(ACTION_CLICK);
	BIND_ENUM_CONSTANT(ACTION_FOCUS);
	BIND_ENUM_CONSTANT(ACTION_BLUR);
	BIND_ENUM_CONSTANT(ACTION_COLLAPSE);
	BIND_ENUM_CONSTANT(ACTION_EXPAND);
	BIND_ENUM_CONSTANT(ACTION_DECREMENT);
	BIND_ENUM_CONSTANT(ACTION_INCREMENT);
	BIND_ENUM_CONSTANT(ACTION_HIDE_TOOLTIP);
	BIND_ENUM_CONSTANT(ACTION_SHOW_TOOLTIP);
	BIND_ENUM_CONSTANT(ACTION_SET_TEXT_SELECTION);
	BIND_ENUM_CONSTANT(ACTION_REPLACE_SELECTED_TEXT);
	BIND_ENUM_CONSTANT(ACTION_SCROLL_BACKWARD);
	BIND_ENUM_CONSTANT(ACTION_SCROLL_DOWN);
	BIND_ENUM_CONSTANT(ACTION_SCROLL_FORWARD);
	BIND_ENUM_CONSTANT(ACTION_SCROLL_LEFT);
	BIND_ENUM_CONSTANT(ACTION_SCROLL_RIGHT);
	BIND_ENUM_CONSTANT(ACTION_SCROLL_UP);
	BIND_ENUM_CONSTANT(ACTION_SCROLL_INTO_VIEW);
	BIND_ENUM_CONSTANT(ACTION_SCROLL_TO_POINT);
	BIND_ENUM_CONSTANT(ACTION_SET_SCROLL_OFFSET);
	BIND_ENUM_CONSTANT(ACTION_SET_VALUE);
	BIND_ENUM_CONSTANT(ACTION_SHOW_CONTEXT_MENU);
	BIND_ENUM_CONSTANT(ACTION_CUSTOM);

	BIND_ENUM_CONSTANT(LIVE_OFF);
	BIND_ENUM_CONSTANT(LIVE_POLITE);
	BIND_ENUM_CONSTANT(LIVE_ASSERTIVE);

	BIND_ENUM_CONSTANT(MOUSE_MODE_VISIBLE);
	BIND_ENUM_CONSTANT(MOUSE_MODE_HIDDEN);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CAPTURED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED);
	BIND_ENUM_CONSTANT(MOUSE_MODE_CONFINED_HIDDEN);
	BIND_ENUM_CONSTANT(MOUSE_MODE_MAX);

	BIND_CONSTANT(SCREEN_WITH_MOUSE_FOCUS);
	BIND_CONSTANT(SCREEN_WITH_KEYBOARD_FOCUS);
	BIND_CONSTANT(SCREEN_PRIMARY);
	BIND_CONSTANT(SCREEN_OF_MAIN_WINDOW);

	BIND_CONSTANT(MAIN_WINDOW_ID);
	BIND_CONSTANT(INVALID_WINDOW_ID);
	BIND_CONSTANT(INVALID_INDICATOR_ID);

	BIND_ENUM_CONSTANT(SCREEN_LANDSCAPE);
	BIND_ENUM_CONSTANT(SCREEN_PORTRAIT);
	BIND_ENUM_CONSTANT(SCREEN_REVERSE_LANDSCAPE);
	BIND_ENUM_CONSTANT(SCREEN_REVERSE_PORTRAIT);
	BIND_ENUM_CONSTANT(SCREEN_SENSOR_LANDSCAPE);
	BIND_ENUM_CONSTANT(SCREEN_SENSOR_PORTRAIT);
	BIND_ENUM_CONSTANT(SCREEN_SENSOR);

	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_DEFAULT);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_MULTILINE);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_NUMBER);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_NUMBER_DECIMAL);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_PHONE);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_EMAIL_ADDRESS);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_PASSWORD);
	BIND_ENUM_CONSTANT(KEYBOARD_TYPE_URL);

	BIND_ENUM_CONSTANT(CURSOR_ARROW);
	BIND_ENUM_CONSTANT(CURSOR_IBEAM);
	BIND_ENUM_CONSTANT(CURSOR_POINTING_HAND);
	BIND_ENUM_CONSTANT(CURSOR_CROSS);
	BIND_ENUM_CONSTANT(CURSOR_WAIT);
	BIND_ENUM_CONSTANT(CURSOR_BUSY);
	BIND_ENUM_CONSTANT(CURSOR_DRAG);
	BIND_ENUM_CONSTANT(CURSOR_CAN_DROP);
	BIND_ENUM_CONSTANT(CURSOR_FORBIDDEN);
	BIND_ENUM_CONSTANT(CURSOR_VSIZE);
	BIND_ENUM_CONSTANT(CURSOR_HSIZE);
	BIND_ENUM_CONSTANT(CURSOR_BDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_FDIAGSIZE);
	BIND_ENUM_CONSTANT(CURSOR_MOVE);
	BIND_ENUM_CONSTANT(CURSOR_VSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HSPLIT);
	BIND_ENUM_CONSTANT(CURSOR_HELP);
	BIND_ENUM_CONSTANT(CURSOR_MAX);

	BIND_ENUM_CONSTANT(FILE_DIALOG_MODE_OPEN_FILE);
	BIND_ENUM_CONSTANT(FILE_DIALOG_MODE_OPEN_FILES);
	BIND_ENUM_CONSTANT(FILE_DIALOG_MODE_OPEN_DIR);
	BIND_ENUM_CONSTANT(FILE_DIALOG_MODE_OPEN_ANY);
	BIND_ENUM_CONSTANT(FILE_DIALOG_MODE_SAVE_FILE);

	BIND_ENUM_CONSTANT(WINDOW_MODE_WINDOWED);
	BIND_ENUM_CONSTANT(WINDOW_MODE_MINIMIZED);
	BIND_ENUM_CONSTANT(WINDOW_MODE_MAXIMIZED);
	BIND_ENUM_CONSTANT(WINDOW_MODE_FULLSCREEN);
	BIND_ENUM_CONSTANT(WINDOW_MODE_EXCLUSIVE_FULLSCREEN);

	BIND_ENUM_CONSTANT(WINDOW_FLAG_RESIZE_DISABLED);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_BORDERLESS);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_ALWAYS_ON_TOP);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_TRANSPARENT);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_NO_FOCUS);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_POPUP);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_EXTEND_TO_TITLE);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_MOUSE_PASSTHROUGH);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_SHARP_CORNERS);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_EXCLUDE_FROM_CAPTURE);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_POPUP_WM_HINT);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_MINIMIZE_DISABLED);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_MAXIMIZE_DISABLED);
	BIND_ENUM_CONSTANT(WINDOW_FLAG_MAX);

	BIND_ENUM_CONSTANT(WINDOW_EVENT_MOUSE_ENTER);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_MOUSE_EXIT);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_FOCUS_IN);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_FOCUS_OUT);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_CLOSE_REQUEST);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_GO_BACK_REQUEST);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_DPI_CHANGE);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_TITLEBAR_CHANGE);
	BIND_ENUM_CONSTANT(WINDOW_EVENT_FORCE_CLOSE);

	BIND_ENUM_CONSTANT(WINDOW_EDGE_TOP_LEFT);
	BIND_ENUM_CONSTANT(WINDOW_EDGE_TOP);
	BIND_ENUM_CONSTANT(WINDOW_EDGE_TOP_RIGHT);
	BIND_ENUM_CONSTANT(WINDOW_EDGE_LEFT);
	BIND_ENUM_CONSTANT(WINDOW_EDGE_RIGHT);
	BIND_ENUM_CONSTANT(WINDOW_EDGE_BOTTOM_LEFT);
	BIND_ENUM_CONSTANT(WINDOW_EDGE_BOTTOM);
	BIND_ENUM_CONSTANT(WINDOW_EDGE_BOTTOM_RIGHT);
	BIND_ENUM_CONSTANT(WINDOW_EDGE_MAX);

	BIND_ENUM_CONSTANT(VSYNC_DISABLED);
	BIND_ENUM_CONSTANT(VSYNC_ENABLED);
	BIND_ENUM_CONSTANT(VSYNC_ADAPTIVE);
	BIND_ENUM_CONSTANT(VSYNC_MAILBOX);

	BIND_ENUM_CONSTANT(DISPLAY_HANDLE);
	BIND_ENUM_CONSTANT(WINDOW_HANDLE);
	BIND_ENUM_CONSTANT(WINDOW_VIEW);
	BIND_ENUM_CONSTANT(OPENGL_CONTEXT);
	BIND_ENUM_CONSTANT(EGL_DISPLAY);
	BIND_ENUM_CONSTANT(EGL_CONFIG);

	BIND_ENUM_CONSTANT(TTS_UTTERANCE_STARTED);
	BIND_ENUM_CONSTANT(TTS_UTTERANCE_ENDED);
	BIND_ENUM_CONSTANT(TTS_UTTERANCE_CANCELED);
	BIND_ENUM_CONSTANT(TTS_UTTERANCE_BOUNDARY);
}

Ref<Image> DisplayServer::_get_cursor_image_from_resource(const Ref<Resource> &p_cursor, const Vector2 &p_hotspot) {
	Ref<Image> image;
	ERR_FAIL_COND_V_MSG(p_hotspot.x < 0 || p_hotspot.y < 0, image, "Hotspot outside cursor image.");

	Ref<Texture2D> texture = p_cursor;
	if (texture.is_valid()) {
		image = texture->get_image();
	} else {
		image = p_cursor;
	}
	ERR_FAIL_COND_V(image.is_null(), image);

	Size2 image_size = image->get_size();
	ERR_FAIL_COND_V_MSG(p_hotspot.x > image_size.width || p_hotspot.y > image_size.height, image, "Hotspot outside cursor image.");
	ERR_FAIL_COND_V_MSG(image_size.width > 256 || image_size.height > 256, image, "Cursor image too big. Max supported size is 256x256.");

	if (image->is_compressed()) {
		image = image->duplicate(true);
		Error err = image->decompress();
		ERR_FAIL_COND_V_MSG(err != OK, Ref<Image>(), "Couldn't decompress VRAM-compressed custom mouse cursor image. Switch to a lossless compression mode in the Import dock.");
	}
	return image;
}

void DisplayServer::register_create_function(const char *p_name, CreateFunction p_function, GetRenderingDriversFunction p_get_drivers) {
	ERR_FAIL_COND(server_create_count == MAX_SERVERS);
	// Headless display server is always last
	server_create_functions[server_create_count] = server_create_functions[server_create_count - 1];
	server_create_functions[server_create_count - 1].name = p_name;
	server_create_functions[server_create_count - 1].create_function = p_function;
	server_create_functions[server_create_count - 1].get_rendering_drivers_function = p_get_drivers;
	server_create_count++;
}

int DisplayServer::get_create_function_count() {
	return server_create_count;
}

const char *DisplayServer::get_create_function_name(int p_index) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, nullptr);
	return server_create_functions[p_index].name;
}

Vector<String> DisplayServer::get_create_function_rendering_drivers(int p_index) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, Vector<String>());
	return server_create_functions[p_index].get_rendering_drivers_function();
}

DisplayServer *DisplayServer::create(int p_index, const String &p_rendering_driver, WindowMode p_mode, VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, Context p_context, int64_t p_parent_window, Error &r_error) {
	ERR_FAIL_INDEX_V(p_index, server_create_count, nullptr);
	return server_create_functions[p_index].create_function(p_rendering_driver, p_mode, p_vsync_mode, p_flags, p_position, p_resolution, p_screen, p_context, p_parent_window, r_error);
}

void DisplayServer::_input_set_mouse_mode(Input::MouseMode p_mode) {
	singleton->mouse_set_mode(MouseMode(p_mode));
}

Input::MouseMode DisplayServer::_input_get_mouse_mode() {
	return Input::MouseMode(singleton->mouse_get_mode());
}

void DisplayServer::_input_set_mouse_mode_override(Input::MouseMode p_mode) {
	singleton->mouse_set_mode_override(MouseMode(p_mode));
}

Input::MouseMode DisplayServer::_input_get_mouse_mode_override() {
	return Input::MouseMode(singleton->mouse_get_mode_override());
}

void DisplayServer::_input_set_mouse_mode_override_enabled(bool p_enabled) {
	singleton->mouse_set_mode_override_enabled(p_enabled);
}

bool DisplayServer::_input_is_mouse_mode_override_enabled() {
	return singleton->mouse_is_mode_override_enabled();
}

void DisplayServer::_input_warp(const Vector2 &p_to_pos) {
	singleton->warp_mouse(p_to_pos);
}

Input::CursorShape DisplayServer::_input_get_current_cursor_shape() {
	return (Input::CursorShape)singleton->cursor_get_shape();
}

void DisplayServer::_input_set_custom_mouse_cursor_func(const Ref<Resource> &p_image, Input::CursorShape p_shape, const Vector2 &p_hotspot) {
	singleton->cursor_set_custom_image(p_image, (CursorShape)p_shape, p_hotspot);
}

bool DisplayServer::is_rendering_device_supported() {
#if defined(RD_ENABLED)
	RenderingDevice *device = RenderingDevice::get_singleton();
	if (device) {
		return true;
	}

	if (supported_rendering_device == RenderingDeviceCreationStatus::SUCCESS) {
		return true;
	} else if (supported_rendering_device == RenderingDeviceCreationStatus::FAILURE) {
		return false;
	}

	Error err;

#if defined(WINDOWS_ENABLED) || defined(LINUXBSD_ENABLED)
	// On some drivers combining OpenGL and RenderingDevice can result in crash, offload the check to the subprocess.
	List<String> arguments;
	arguments.push_back("--test-rd-support");
	if (get_singleton()) {
		arguments.push_back("--display-driver");
		arguments.push_back(get_singleton()->get_name().to_lower());
	}

	String pipe;
	int exitcode = 0;
	err = OS::get_singleton()->execute(OS::get_singleton()->get_executable_path(), arguments, &pipe, &exitcode);
	if (err == OK && exitcode == 0) {
		supported_rendering_device = RenderingDeviceCreationStatus::SUCCESS;
		return true;
	} else {
		supported_rendering_device = RenderingDeviceCreationStatus::FAILURE;
	}
#else // WINDOWS_ENABLED

	RenderingContextDriver *rcd = nullptr;

#if defined(VULKAN_ENABLED)
	rcd = memnew(RenderingContextDriverVulkan);
#endif
#ifdef D3D12_ENABLED
	if (rcd == nullptr) {
		rcd = memnew(RenderingContextDriverD3D12);
	}
#endif
#ifdef METAL_ENABLED
	if (rcd == nullptr) {
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")
		// Eliminate "RenderingContextDriverMetal is only available on iOS 14.0 or newer".
		rcd = memnew(RenderingContextDriverMetal);
		GODOT_CLANG_WARNING_POP
	}
#endif

	if (rcd != nullptr) {
		err = rcd->initialize();
		if (err == OK) {
			RenderingDevice *rd = memnew(RenderingDevice);
			err = rd->initialize(rcd);
			memdelete(rd);
			rd = nullptr;
			if (err == OK) {
				// Creating a RenderingDevice is quite slow.
				// Cache the result for future usage, so that it's much faster on subsequent calls.
				supported_rendering_device = RenderingDeviceCreationStatus::SUCCESS;
				memdelete(rcd);
				rcd = nullptr;
				return true;
			} else {
				supported_rendering_device = RenderingDeviceCreationStatus::FAILURE;
			}
		}

		memdelete(rcd);
		rcd = nullptr;
	}

#endif // WINDOWS_ENABLED
#endif // RD_ENABLED
	return false;
}

bool DisplayServer::can_create_rendering_device() {
	if (get_singleton() && get_singleton()->get_name() == "headless") {
		return false;
	}

#if defined(RD_ENABLED)
	RenderingDevice *device = RenderingDevice::get_singleton();
	if (device) {
		return true;
	}

	if (created_rendering_device == RenderingDeviceCreationStatus::SUCCESS) {
		return true;
	} else if (created_rendering_device == RenderingDeviceCreationStatus::FAILURE) {
		return false;
	}

	Error err;

#ifdef WINDOWS_ENABLED
	// On some NVIDIA drivers combining OpenGL and RenderingDevice can result in crash, offload the check to the subprocess.
	List<String> arguments;
	arguments.push_back("--test-rd-creation");

	String pipe;
	int exitcode = 0;
	err = OS::get_singleton()->execute(OS::get_singleton()->get_executable_path(), arguments, &pipe, &exitcode);
	if (err == OK && exitcode == 0) {
		created_rendering_device = RenderingDeviceCreationStatus::SUCCESS;
		return true;
	} else {
		created_rendering_device = RenderingDeviceCreationStatus::FAILURE;
	}
#else // WINDOWS_ENABLED

	RenderingContextDriver *rcd = nullptr;

#if defined(VULKAN_ENABLED)
	rcd = memnew(RenderingContextDriverVulkan);
#endif
#ifdef D3D12_ENABLED
	if (rcd == nullptr) {
		rcd = memnew(RenderingContextDriverD3D12);
	}
#endif
#ifdef METAL_ENABLED
	if (rcd == nullptr) {
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")
		// Eliminate "RenderingContextDriverMetal is only available on iOS 14.0 or newer".
		rcd = memnew(RenderingContextDriverMetal);
		GODOT_CLANG_WARNING_POP
	}
#endif

	if (rcd != nullptr) {
		err = rcd->initialize();
		if (err == OK) {
			RenderingDevice *rd = memnew(RenderingDevice);
			err = rd->initialize(rcd);
			memdelete(rd);
			rd = nullptr;
			if (err == OK) {
				// Creating a RenderingDevice is quite slow.
				// Cache the result for future usage, so that it's much faster on subsequent calls.
				created_rendering_device = RenderingDeviceCreationStatus::SUCCESS;
				memdelete(rcd);
				rcd = nullptr;
				return true;
			} else {
				created_rendering_device = RenderingDeviceCreationStatus::FAILURE;
			}
		}

		memdelete(rcd);
		rcd = nullptr;
	}

#endif // WINDOWS_ENABLED
#endif // RD_ENABLED
	return false;
}

DisplayServer::DisplayServer() {
	singleton = this;
	Input::set_mouse_mode_func = _input_set_mouse_mode;
	Input::get_mouse_mode_func = _input_get_mouse_mode;
	Input::set_mouse_mode_override_func = _input_set_mouse_mode_override;
	Input::get_mouse_mode_override_func = _input_get_mouse_mode_override;
	Input::set_mouse_mode_override_enabled_func = _input_set_mouse_mode_override_enabled;
	Input::is_mouse_mode_override_enabled_func = _input_is_mouse_mode_override_enabled;
	Input::warp_mouse_func = _input_warp;
	Input::get_current_cursor_shape_func = _input_get_current_cursor_shape;
	Input::set_custom_mouse_cursor_func = _input_set_custom_mouse_cursor_func;
}

DisplayServer::~DisplayServer() {
	singleton = nullptr;
}
