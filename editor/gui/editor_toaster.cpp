/**************************************************************************/
/*  editor_toaster.cpp                                                    */
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

#include "editor_toaster.h"
#include "editor_toaster.compat.inc"

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/object/message_queue.h"
#include "editor/editor_string_names.h"
#include "editor/inspector/editor_inspector.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/style_box_flat.h"
#include "servers/display/display_server.h"

EditorToaster *EditorToaster::singleton = nullptr;

void EditorToaster::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			const double delta = get_process_delta_time();

			// Check if one element is hovered, if so, don't elapse time.
			bool hovered = false;
			for (const Toast *toast : toasts) {
				if (Rect2(Vector2(), toast->get_size()).has_point(toast->get_local_mouse_position())) {
					hovered = true;
					break;
				}
			}

			// Check if notification panel is hovered, if so, don't elapse time.
			if (Rect2(Vector2(), disable_notifications_panel->get_size()).has_point(disable_notifications_panel->get_local_mouse_position()) ||
					Rect2(Vector2(), main_button->get_size()).has_point(main_button->get_local_mouse_position())) {
				hovered = true;
			}

			// Elapses the time and remove toasts if needed.
			if (!hovered) {
				for (Toast *toast : toasts) {
					if (!toast->popped || toast->duration <= 0 || toast->requires_action) {
						continue;
					}
					toast->remaining_time -= delta;
					if (toast->remaining_time < 0) {
						toast->close();
					}
					toast->queue_redraw();
				}
			} else {
				// Reset the timers when hovered.
				for (Toast *toast : toasts) {
					if (!toast->popped || toast->duration <= 0) {
						continue;
					}
					toast->remaining_time = toast->duration;
					toast->queue_redraw();
				}
			}

			// Change alpha over time.
			bool needs_update = false;
			for (Toast *toast : toasts) {
				Color modulate_fade = toast->get_modulate();

				// Change alpha over time.
				if (toast->popped && modulate_fade.a < 1.0) {
					modulate_fade.a += delta * 3;
					toast->set_modulate(modulate_fade);
					disable_notifications_panel->set_modulate(modulate_fade);
				} else if (!toast->popped && modulate_fade.a > 0.0) {
					modulate_fade.a -= delta * 2;
					toast->set_modulate(modulate_fade);
				}

				// Hide element if it is not visible anymore.
				if (modulate_fade.a <= 0.0 && toast->is_visible()) {
					toast->hide();
					needs_update = true;
				} else if (modulate_fade.a > 0.0 && !toast->is_visible()) {
					toast->show();
					needs_update = true;
				}
			}

			if (needs_update) {
				_update_vbox_position();
				_update_disable_notifications_button();
				main_button->queue_redraw();
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			if (vbox_container->is_visible()) {
				main_button->set_button_icon(get_editor_theme_icon(SNAME("Notification")));
			} else {
				main_button->set_button_icon(get_editor_theme_icon(SNAME("NotificationDisabled")));
			}
			disable_notifications_panel->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("info"), SNAME("Toast")));
			clear_notifications_button->set_button_icon(get_editor_theme_icon(SNAME("Clear")));
			disable_notifications_button->set_button_icon(get_editor_theme_icon(SNAME("NotificationDisabled")));
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
			_update_vbox_position();
			_update_disable_notifications_button();
		} break;
	}
}

void EditorToaster::_error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type) {
	// This may be called from a thread. Since we will deal with non-thread-safe elements,
	// we have to put it in the queue for safety.
	callable_mp_static(&EditorToaster::_error_handler_impl).call_deferred(String::utf8(p_file), p_line, String::utf8(p_error), String::utf8(p_errorexp), p_editor_notify, p_type);
}

void EditorToaster::_error_handler_impl(const String &p_file, int p_line, const String &p_error, const String &p_errorexp, bool p_editor_notify, int p_type) {
	if (!EditorToaster::get_singleton() || !EditorToaster::get_singleton()->is_inside_tree()) {
		return;
	}

#ifdef DEV_ENABLED
	bool in_dev = true;
#else
	bool in_dev = false;
#endif

	const int show_all_setting = EDITOR_GET("interface/editor/behavior/show_internal_errors_in_toast_notifications");

	if (p_editor_notify || (show_all_setting == 0 && in_dev) || show_all_setting == 1) {
		String err_str = !p_errorexp.is_empty() ? p_errorexp : p_error;
		const String tooltip_str = p_file + ":" + itos(p_line);

		if (!p_editor_notify) {
			if (p_type == ERR_HANDLER_WARNING) {
				err_str = "INTERNAL WARNING: " + err_str;
			} else {
				err_str = "INTERNAL ERROR: " + err_str;
			}
		}

		const Severity severity = ((ErrorHandlerType)p_type == ERR_HANDLER_WARNING) ? SEVERITY_WARNING : SEVERITY_ERROR;
		EditorToaster::get_singleton()->popup_str(err_str, severity, tooltip_str);
	}
}

// This is kind of a workaround because it's hard to keep the VBox anchored to the bottom.
void EditorToaster::_update_vbox_position() const {
	vbox_container->set_size(Vector2());

	Point2 pos = get_global_position();
	const Size2 vbox_size = vbox_container->get_size();
	pos.y -= vbox_size.y + 15 * EDSCALE;
	if (!is_layout_rtl()) {
		pos.x = pos.x - vbox_size.x + get_size().x;
	}

	vbox_container->set_position(pos);
}

void EditorToaster::_update_disable_notifications_button() const {
	bool any_visible = false;
	for (const Toast *toast : toasts) {
		if (toast->is_visible()) {
			any_visible = true;
			break;
		}
	}

	if (!any_visible || !vbox_container->is_visible()) {
		disable_notifications_panel->hide();
	} else {
		disable_notifications_panel->show();

		Point2 pos = get_global_position();
		const int sep = 15 * EDSCALE;
		const Size2 disable_panel_size = disable_notifications_panel->get_minimum_size();
		pos.y -= disable_panel_size.y + sep;
		if (is_layout_rtl()) {
			pos.x = pos.x - disable_panel_size.x - sep;
		} else {
			pos.x += get_size().x + sep;
		}

		disable_notifications_panel->set_position(pos);
	}
}

void EditorToaster::_auto_hide_or_free_toasts(bool p_clear) {
	// Hide or free old temporary items.
	int visible_temporary = 0;
	int temporary = 0;
	LocalVector<Toast *> to_delete;
	for (int i = vbox_container->get_child_count() - 1; i >= 0; i--) {
		Toast *toast = Object::cast_to<Toast>(vbox_container->get_child(i));

		// If we are clearing all, just delete toast.
		if (p_clear) {
			to_delete.push_back(toast);
			disable_notifications_panel->hide();
			continue;
		}

		if (toast->duration <= 0) {
			continue; // Ignore non-temporary toasts.
		}

		temporary++;
		if (toast->is_visible()) {
			visible_temporary++;
		}

		// Hide
		if (visible_temporary > max_temporary_count) {
			toast->close();
		}

		// Free
		if (temporary > max_temporary_count * 2) {
			to_delete.push_back(toast);
		}
	}

	// Delete the control right away (removed as child) as it might cause issues otherwise when iterating over the vbox_container children.
	for (Toast *t : to_delete) {
		vbox_container->remove_child(t);
		t->queue_free();
		toasts.erase(t);
	}

	if (toasts.is_empty()) {
		main_button->set_tooltip_text(TTRC("No notifications."));
		main_button->set_modulate(Color(0.5, 0.5, 0.5));
		main_button->set_disabled(true);
		set_process_internal(false);
	} else {
		main_button->set_tooltip_text(TTRC("Show notifications."));
		main_button->set_modulate(Color(1, 1, 1));
		main_button->set_disabled(false);
	}
}

void EditorToaster::_draw_button() const {
	bool has_one = false;
	Severity highest_severity = SEVERITY_INFO;
	for (const Toast *toast : toasts) {
		if (!toast->is_visible()) {
			continue;
		}
		has_one = true;
		if (toast->severity > highest_severity) {
			highest_severity = toast->severity;
		}
	}

	if (!has_one) {
		return;
	}

	Color color;
	const real_t button_radius = main_button->get_size().x / 8;
	switch (highest_severity) {
		case SEVERITY_INFO:
			color = get_theme_color(SNAME("accent_color"), EditorStringName(Editor));
			break;
		case SEVERITY_WARNING:
			color = get_theme_color(SNAME("warning_color"), EditorStringName(Editor));
			break;
		case SEVERITY_ERROR:
			color = get_theme_color(SNAME("error_color"), EditorStringName(Editor));
			break;
		default:
			break;
	}
	main_button->draw_circle(Vector2(button_radius * 2, button_radius * 2), button_radius, color);
}

void EditorToaster::_draw_progress(Toast *p_toast) const {
	if (toasts.has(p_toast) && p_toast->remaining_time > 0 && p_toast->duration > 0) {
		Ref<StyleBoxFlat> stylebox;
		switch (p_toast->severity) {
			case SEVERITY_INFO:
				stylebox = get_theme_stylebox(SNAME("info_progress"), SNAME("Toast"));
				break;
			case SEVERITY_WARNING:
				stylebox = get_theme_stylebox(SNAME("warning_progress"), SNAME("Toast"));
				break;
			case SEVERITY_ERROR:
				stylebox = get_theme_stylebox(SNAME("error_progress"), SNAME("Toast"));
				break;
			default:
				break;
		}

		const Size2 size = p_toast->get_size();
		Size2 progress = size;
		progress.width *= MIN(1, Math::remap(p_toast->remaining_time, 0, p_toast->duration, 0, 2));
		if (is_layout_rtl()) {
			p_toast->draw_style_box(stylebox, Rect2(size - progress, progress));
		} else {
			p_toast->draw_style_box(stylebox, Rect2(Vector2(), progress));
		}
	}
}

void EditorToaster::_set_notifications_enabled(bool p_enabled) const {
	vbox_container->set_visible(p_enabled);
	if (p_enabled) {
		main_button->set_button_icon(get_editor_theme_icon(SNAME("Notification")));
	} else {
		main_button->set_button_icon(get_editor_theme_icon(SNAME("NotificationDisabled")));
	}
	_update_disable_notifications_button();
}

void EditorToaster::_repop_old() const {
	// Repop olds, up to max_temporary_count
	bool needs_update = false;
	int visible_count = 0;
	for (int i = vbox_container->get_child_count() - 1; i >= 0; i--) {
		Toast *toast = Object::cast_to<Toast>(vbox_container->get_child(i));
		if (!toast->is_visible()) {
			toast->show();
			toast->remaining_time = toast->duration;
			toast->popped = true;
			needs_update = true;
		}
		visible_count++;
		if (visible_count >= max_temporary_count) {
			break;
		}
	}
	if (needs_update) {
		_update_vbox_position();
		_update_disable_notifications_button();
		main_button->queue_redraw();
	}
}

Toast *EditorToaster::popup(Control *p_control, Severity p_severity, double p_time, const String &p_tooltip) {
	// Create the panel according to the severity.
	Toast *toast = memnew(Toast);
	toast->set_tooltip_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	toast->set_tooltip_text(p_tooltip);
	toast->set_modulate(Color(1, 1, 1, 0));
	toast->connect(SceneStringName(draw), callable_mp(this, &EditorToaster::_draw_progress).bind(toast));
	toast->connect(SceneStringName(theme_changed), callable_mp(this, &EditorToaster::_toast_theme_changed).bind(toast));

	toasts.insert(toast);

	// Vertical container.
	VBoxContainer *vbox = memnew(VBoxContainer);

	// Horizontal container.
	HBoxContainer *hbox = memnew(HBoxContainer);
	hbox->set_h_size_flags(SIZE_EXPAND_FILL);

	vbox->add_child(hbox);
	toast->add_child(vbox);

	// Add severity icon.
	const int icon_size = get_theme_constant(SNAME("class_icon_size"), EditorStringName(Editor));

	TextureRect *icon = memnew(TextureRect);
	icon->set_custom_minimum_size(Size2(icon_size, icon_size));
	icon->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	icon->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
	icon->set_v_size_flags(SIZE_SHRINK_CENTER);
	switch (p_severity) {
		case SEVERITY_INFO:
			toast->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("info"), SNAME("Toast")));
			icon->set_texture(get_editor_theme_icon(SNAME("Popup")));
			break;
		case SEVERITY_WARNING:
			toast->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("warning"), SNAME("Toast")));
			icon->set_texture(get_editor_theme_icon(SNAME("StatusWarning")));
			break;
		case SEVERITY_ERROR:
			toast->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("error"), SNAME("Toast")));
			icon->set_texture(get_editor_theme_icon(SNAME("StatusError")));
			break;
		default:
			break;
	}
	hbox->add_child(icon);

	// Content control.
	p_control->set_h_size_flags(SIZE_EXPAND_FILL);
	hbox->add_child(p_control);

	// Add buttons.
	if (p_time > 0.0) {
		Button *copy_button = memnew(Button);
		copy_button->set_accessibility_name(TTRC("Copy"));
		copy_button->set_flat(true);
		copy_button->set_tooltip_text(TTRC("Copy toast message."));
		copy_button->connect(SceneStringName(pressed), callable_mp(toast, &Toast::copy));
		hbox->add_child(copy_button);

		Button *close_button = memnew(Button);
		close_button->set_accessibility_name(TTRC("Close"));
		close_button->set_flat(true);
		close_button->connect(SceneStringName(pressed), callable_mp(toast, &Toast::instant_close));
		hbox->add_child(close_button);

		HBoxContainer *action_container = memnew(HBoxContainer);
		action_container->set_h_size_flags(SIZE_EXPAND_FILL);
		action_container->set_alignment(ALIGNMENT_END);
		action_container->add_theme_constant_override("separation", 16);
		vbox->add_child(action_container);

		toast->copy_button = copy_button;
		toast->close_button = close_button;
		toast->action_container = action_container;
	}

	toast->severity = p_severity;
	if (p_time > 0.0) {
		toast->duration = p_time;
		toast->remaining_time = p_time;
	} else {
		toast->duration = -1.0;
	}
	toast->popped = true;
	vbox_container->add_child(toast);
	_auto_hide_or_free_toasts();
	_update_vbox_position();
	_update_disable_notifications_button();
	main_button->queue_redraw();

	return toast;
}

Toast *EditorToaster::_popup_str(const String &p_message, Severity p_severity, const String &p_tooltip) {
	// Check if we already have a popup with the given message.
	Toast *toast = nullptr;
	for (Toast *t : toasts) {
		if (t->message == p_message && t->severity == p_severity && t->tooltip == p_tooltip) {
			toast = t;
			break;
		}
	}

	// Create a new message if needed.
	if (toast == nullptr) {
		HBoxContainer *hb = memnew(HBoxContainer);
		hb->add_theme_constant_override("separation", 0);

		Label *label = memnew(Label);
		label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
		label->set_focus_mode(FOCUS_ACCESSIBILITY);
		hb->add_child(label);

		Label *count_label = memnew(Label);
		count_label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
		hb->add_child(count_label);

		toast = popup(hb, p_severity, default_message_duration, p_tooltip);

		toast->message = p_message;
		toast->tooltip = p_tooltip;
		toast->count = 1;
		toast->message_label = label;
		toast->message_count_label = count_label;
	} else {
		if (toast->popped) {
			toast->count += 1;
		} else {
			toast->count = 1;
		}
		toast->remaining_time = toast->duration;
		toast->popped = true;
		toast->show();
		vbox_container->move_child(toast, vbox_container->get_child_count());
		_auto_hide_or_free_toasts();
		_update_vbox_position();
		_update_disable_notifications_button();
		main_button->queue_redraw();
	}

	// Retrieve the label back, then update the text.
	Label *message_label = toast->message_label;
	ERR_FAIL_NULL_V(message_label, toast);
	message_label->set_text(p_message);
	message_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	message_label->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	message_label->set_v_size_flags(SIZE_EXPAND_FILL);
	message_label->set_custom_minimum_size(Size2(get_viewport_rect().size.x / 4, 0));
	message_label->set_max_lines_visible(4);

	// Retrieve the count label back, then update the text.
	Label *message_count_label = toast->message_count_label;
	if (toast->count == 1) {
		message_count_label->hide();
	} else {
		message_count_label->set_text(vformat("(%d)", toast->count));
		message_count_label->show();
	}

	vbox_container->reset_size();
	set_process_internal(true);

	return toast;
}

void EditorToaster::_toast_theme_changed(Toast *p_toast) const {
	ERR_FAIL_COND(!toasts.has(p_toast));

	if (p_toast->close_button) {
		p_toast->close_button->set_button_icon(get_editor_theme_icon(SNAME("Close")));
	}
	if (p_toast->copy_button) {
		p_toast->copy_button->set_button_icon(get_editor_theme_icon(SNAME("ActionCopy")));
	}
}

void EditorToaster::_bind_methods() {
	ClassDB::bind_method(D_METHOD("push_toast", "message", "severity", "tooltip"), &EditorToaster::_popup_str, DEFVAL(EditorToaster::SEVERITY_INFO), DEFVAL(String()));

	BIND_ENUM_CONSTANT(SEVERITY_INFO);
	BIND_ENUM_CONSTANT(SEVERITY_WARNING);
	BIND_ENUM_CONSTANT(SEVERITY_ERROR);
}

EditorToaster *EditorToaster::get_singleton() {
	return singleton;
}

Toast *EditorToaster::popup_str(const String &p_message, Severity p_severity, const String &p_tooltip) {
	if (Thread::is_main_thread()) {
		return _popup_str(p_message, p_severity, p_tooltip);
	}

	// Can't return the created object from deferred execution.
	MessageQueue::get_main_singleton()->push_callable(
			callable_mp(this, &EditorToaster::_popup_str), p_message, p_severity, p_tooltip);
	return nullptr;
}

EditorToaster::EditorToaster() {
	set_notify_transform(true);

	// VBox.
	vbox_container = memnew(VBoxContainer);
	vbox_container->set_as_top_level(true);
	vbox_container->connect(SceneStringName(resized), callable_mp(this, &EditorToaster::_update_vbox_position));
	vbox_container->add_theme_constant_override(SNAME("separation"), 8 * EDSCALE);
	vbox_container->set_alignment(ALIGNMENT_END);
	add_child(vbox_container);

	// Main button.
	main_button = memnew(Button);
	main_button->set_accessibility_name(TTRC("Notifications:"));
	main_button->set_tooltip_text(TTRC("No notifications."));
	main_button->set_modulate(Color(0.5, 0.5, 0.5));
	main_button->set_disabled(true);
	main_button->set_theme_type_variation("FlatMenuButton");
	main_button->connect(SceneStringName(pressed), callable_mp(this, &EditorToaster::_set_notifications_enabled).bind(true));
	main_button->connect(SceneStringName(pressed), callable_mp(this, &EditorToaster::_repop_old));
	main_button->connect(SceneStringName(draw), callable_mp(this, &EditorToaster::_draw_button));
	add_child(main_button);

	// Disable notification panel.
	disable_notifications_panel = memnew(PanelContainer);
	disable_notifications_panel->set_as_top_level(true);
	disable_notifications_panel->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SNAME("info"), SNAME("Toast")));
	add_child(disable_notifications_panel);

	VBoxContainer *notification_options_hbox = memnew(VBoxContainer);
	disable_notifications_panel->add_child(notification_options_hbox);

	// Clear notification button.
	clear_notifications_button = memnew(Button);
	clear_notifications_button->set_tooltip_text(TTRC("Clear all notifications."));
	clear_notifications_button->set_flat(true);
	clear_notifications_button->set_button_icon(get_editor_theme_icon(SNAME("Clear")));
	clear_notifications_button->connect(SceneStringName(pressed), callable_mp(this, &EditorToaster::_auto_hide_or_free_toasts).bind(true));
	notification_options_hbox->add_child(clear_notifications_button);

	// Disable notification button.
	disable_notifications_button = memnew(Button);
	disable_notifications_button->set_tooltip_text(TTRC("Silence the notifications."));
	disable_notifications_button->set_flat(true);
	disable_notifications_button->connect(SceneStringName(pressed), callable_mp(this, &EditorToaster::_set_notifications_enabled).bind(false));
	notification_options_hbox->add_child(disable_notifications_button);

	// Other
	singleton = this;

	eh.errfunc = _error_handler;
	add_error_handler(&eh);
}

EditorToaster::~EditorToaster() {
	singleton = nullptr;
	remove_error_handler(&eh);
}

void Toast::_bind_methods() {
	ClassDB::bind_method(D_METHOD("close"), &Toast::close);
	ClassDB::bind_method(D_METHOD("copy"), &Toast::copy);
	ClassDB::bind_method(D_METHOD("get_message"), &Toast::get_message);
	ClassDB::bind_method(D_METHOD("get_severity"), &Toast::get_severity);
	ClassDB::bind_method(D_METHOD("instant_close"), &Toast::instant_close);
	ClassDB::bind_method(D_METHOD("set_action", "action", "callback", "icon"), &Toast::set_action, DEFVAL(StringName()));
}

String Toast::get_message() const {
	return message;
}

EditorToaster::Severity Toast::get_severity() const {
	return severity;
}

void Toast::close() {
	remaining_time = -1.0;
	popped = false;

	if (!actions.is_empty()) {
		for (int i = action_container->get_child_count() - 1; i >= 0; i--) {
			Node *child = action_container->get_child(i);
			action_container->remove_child(child);
			child->queue_free();
		}
		actions.clear();
		requires_action = false;
	}
}

void Toast::instant_close() {
	close();
	set_modulate(Color(1, 1, 1, 0));
}

void Toast::copy() const {
	DisplayServer::get_singleton()->clipboard_set(message);
}

Toast *Toast::set_action(const String &p_label, const Callable &p_callback, const StringName &p_icon_name) {
	if (actions.has(p_label)) {
		return this;
	}

	actions.insert(p_label);
	requires_action = true;

	EditorInspectorActionButton *action_button = memnew(EditorInspectorActionButton(p_label, p_icon_name));
	action_button->connect(SceneStringName(pressed), p_callback);
	action_container->add_child(action_button);

	return this;
}
