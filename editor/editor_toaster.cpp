/*************************************************************************/
/*  editor_toaster.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"

#include "editor_toaster.h"

EditorToaster *EditorToaster::singleton = nullptr;

void EditorToaster::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_INTERNAL_PROCESS: {
			double delta = get_process_delta_time();

			// Check if one element is hovered, if so, don't elapse time.
			bool hovered = false;
			for (const KeyValue<Control *, Toast> &element : toasts) {
				if (Rect2(Vector2(), element.key->get_size()).has_point(element.key->get_local_mouse_position())) {
					hovered = true;
					break;
				}
			}

			// Elapses the time and remove toasts if needed.
			if (!hovered) {
				for (const KeyValue<Control *, Toast> &element : toasts) {
					if (!element.value.popped || element.value.duration <= 0) {
						continue;
					}
					toasts[element.key].remaining_time -= delta;
					if (toasts[element.key].remaining_time < 0) {
						close(element.key);
					}
					element.key->update();
				}
			} else {
				// Reset the timers when hovered.
				for (const KeyValue<Control *, Toast> &element : toasts) {
					if (!element.value.popped || element.value.duration <= 0) {
						continue;
					}
					toasts[element.key].remaining_time = element.value.duration;
					element.key->update();
				}
			}

			// Change alpha over time.
			bool needs_update = false;
			for (const KeyValue<Control *, Toast> &element : toasts) {
				Color modulate = element.key->get_modulate();

				// Change alpha over time.
				if (element.value.popped && modulate.a < 1.0) {
					modulate.a += delta * 3;
					element.key->set_modulate(modulate);
				} else if (!element.value.popped && modulate.a > 0.0) {
					modulate.a -= delta * 2;
					element.key->set_modulate(modulate);
				}

				// Hide element if it is not visible anymore.
				if (modulate.a <= 0) {
					if (element.key->is_visible()) {
						element.key->hide();
						needs_update = true;
					}
				}
			}

			if (needs_update) {
				_update_vbox_position();
				_update_disable_notifications_button();
				main_button->update();
			}
		} break;
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			if (vbox_container->is_visible()) {
				main_button->set_icon(get_theme_icon(SNAME("Notification"), SNAME("EditorIcons")));
			} else {
				main_button->set_icon(get_theme_icon(SNAME("NotificationDisabled"), SNAME("EditorIcons")));
			}
			disable_notifications_button->set_icon(get_theme_icon(SNAME("NotificationDisabled"), SNAME("EditorIcons")));

			// Styleboxes background.
			info_panel_style_background->set_bg_color(get_theme_color("base_color", "Editor"));

			warning_panel_style_background->set_bg_color(get_theme_color("base_color", "Editor"));
			warning_panel_style_background->set_border_color(get_theme_color("warning_color", "Editor"));

			error_panel_style_background->set_bg_color(get_theme_color("base_color", "Editor"));
			error_panel_style_background->set_border_color(get_theme_color("error_color", "Editor"));

			// Styleboxes progress.
			info_panel_style_progress->set_bg_color(get_theme_color("base_color", "Editor").lightened(0.03));

			warning_panel_style_progress->set_bg_color(get_theme_color("base_color", "Editor").lightened(0.03));
			warning_panel_style_progress->set_border_color(get_theme_color("warning_color", "Editor"));

			error_panel_style_progress->set_bg_color(get_theme_color("base_color", "Editor").lightened(0.03));
			error_panel_style_progress->set_border_color(get_theme_color("error_color", "Editor"));

			main_button->update();
			disable_notifications_button->update();
		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			_update_vbox_position();
			_update_disable_notifications_button();
		} break;
		default:
			break;
	}
}

void EditorToaster::_error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type) {
	if (!EditorToaster::get_singleton() || !EditorToaster::get_singleton()->is_inside_tree()) {
		return;
	}

#ifdef DEV_ENABLED
	bool in_dev = true;
#else
	bool in_dev = false;
#endif

	int show_all_setting = EDITOR_GET("interface/editor/show_internal_errors_in_toast_notifications");

	if (p_editor_notify || (show_all_setting == 0 && in_dev) || show_all_setting == 1) {
		String err_str;
		if (p_errorexp && p_errorexp[0]) {
			err_str = p_errorexp;
		} else {
			err_str = String(p_error);
		}
		String tooltip_str = String(p_file) + ":" + itos(p_line);

		if (!p_editor_notify) {
			if (p_type == ERR_HANDLER_WARNING) {
				err_str = "INTERNAL WARNING: " + err_str;
			} else {
				err_str = "INTERNAL ERROR: " + err_str;
			}
		}

		if (p_type == ERR_HANDLER_WARNING) {
			EditorToaster::get_singleton()->popup_str(err_str, SEVERITY_WARNING, tooltip_str);
		} else {
			EditorToaster::get_singleton()->popup_str(err_str, SEVERITY_ERROR, tooltip_str);
		}
	}
}

void EditorToaster::_update_vbox_position() {
	// This is kind of a workaround because it's hard to keep the VBox anchroed to the bottom.
	vbox_container->set_size(Vector2());
	vbox_container->set_position(get_global_position() - vbox_container->get_size() + Vector2(get_size().x, -5 * EDSCALE));
}

void EditorToaster::_update_disable_notifications_button() {
	bool any_visible = false;
	for (KeyValue<Control *, Toast> element : toasts) {
		if (element.key->is_visible()) {
			any_visible = true;
			break;
		}
	}

	if (!any_visible || !vbox_container->is_visible()) {
		disable_notifications_panel->hide();
	} else {
		disable_notifications_panel->show();
		disable_notifications_panel->set_position(get_global_position() + Vector2(5 * EDSCALE, -disable_notifications_panel->get_minimum_size().y) + Vector2(get_size().x, -5 * EDSCALE));
	}
}

void EditorToaster::_auto_hide_or_free_toasts() {
	// Hide or free old temporary items.
	int visible_temporary = 0;
	int temporary = 0;
	LocalVector<Control *> to_delete;
	for (int i = vbox_container->get_child_count() - 1; i >= 0; i--) {
		Control *control = Object::cast_to<Control>(vbox_container->get_child(i));
		if (toasts[control].duration <= 0) {
			continue; // Ignore non-temporary toasts.
		}

		temporary++;
		if (control->is_visible()) {
			visible_temporary++;
		}

		// Hide
		if (visible_temporary > max_temporary_count) {
			close(control);
		}

		// Free
		if (temporary > max_temporary_count * 2) {
			to_delete.push_back(control);
		}
	}

	// Delete the control right away (removed as child) as it might cause issues otherwise when iterative over the vbox_container children.
	for (unsigned int i = 0; i < to_delete.size(); i++) {
		vbox_container->remove_child(to_delete[i]);
		to_delete[i]->queue_delete();
		toasts.erase(to_delete[i]);
	}
}

void EditorToaster::_draw_button() {
	bool has_one = false;
	Severity highest_severity = SEVERITY_INFO;
	for (const KeyValue<Control *, Toast> &element : toasts) {
		if (!element.key->is_visible()) {
			continue;
		}
		has_one = true;
		if (element.value.severity > highest_severity) {
			highest_severity = element.value.severity;
		}
	}

	if (!has_one) {
		return;
	}

	Color color;
	real_t button_radius = main_button->get_size().x / 8;
	switch (highest_severity) {
		case SEVERITY_INFO:
			color = get_theme_color("accent_color", "Editor");
			break;
		case SEVERITY_WARNING:
			color = get_theme_color("warning_color", "Editor");
			break;
		case SEVERITY_ERROR:
			color = get_theme_color("error_color", "Editor");
			break;
		default:
			break;
	}
	main_button->draw_circle(Vector2(button_radius * 2, button_radius * 2), button_radius, color);
}

void EditorToaster::_draw_progress(Control *panel) {
	if (toasts.has(panel) && toasts[panel].remaining_time > 0 && toasts[panel].duration > 0) {
		Size2 size = panel->get_size();
		size.x *= MIN(1, Math::range_lerp(toasts[panel].remaining_time, 0, toasts[panel].duration, 0, 2));

		Ref<StyleBoxFlat> stylebox;
		switch (toasts[panel].severity) {
			case SEVERITY_INFO:
				stylebox = info_panel_style_progress;
				break;
			case SEVERITY_WARNING:
				stylebox = warning_panel_style_progress;
				break;
			case SEVERITY_ERROR:
				stylebox = error_panel_style_progress;
				break;
			default:
				break;
		}
		panel->draw_style_box(stylebox, Rect2(Vector2(), size));
	}
}

void EditorToaster::_set_notifications_enabled(bool p_enabled) {
	vbox_container->set_visible(p_enabled);
	if (p_enabled) {
		main_button->set_icon(get_theme_icon(SNAME("Notification"), SNAME("EditorIcons")));
	} else {
		main_button->set_icon(get_theme_icon(SNAME("NotificationDisabled"), SNAME("EditorIcons")));
	}
	_update_disable_notifications_button();
}

void EditorToaster::_repop_old() {
	// Repop olds, up to max_temporary_count
	bool needs_update = false;
	int visible = 0;
	for (int i = vbox_container->get_child_count() - 1; i >= 0; i--) {
		Control *control = Object::cast_to<Control>(vbox_container->get_child(i));
		if (!control->is_visible()) {
			control->show();
			toasts[control].remaining_time = toasts[control].duration;
			toasts[control].popped = true;
			needs_update = true;
		}
		visible++;
		if (visible >= max_temporary_count) {
			break;
		}
	}
	if (needs_update) {
		_update_vbox_position();
		_update_disable_notifications_button();
		main_button->update();
	}
}

Control *EditorToaster::popup(Control *p_control, Severity p_severity, double p_time, String p_tooltip) {
	// Create the panel according to the severity.
	PanelContainer *panel = memnew(PanelContainer);
	panel->set_tooltip(p_tooltip);
	switch (p_severity) {
		case SEVERITY_INFO:
			panel->add_theme_style_override("panel", info_panel_style_background);
			break;
		case SEVERITY_WARNING:
			panel->add_theme_style_override("panel", warning_panel_style_background);
			break;
		case SEVERITY_ERROR:
			panel->add_theme_style_override("panel", error_panel_style_background);
			break;
		default:
			break;
	}
	panel->set_modulate(Color(1, 1, 1, 0));
	panel->connect("draw", callable_bind(callable_mp(this, &EditorToaster::_draw_progress), panel));

	// Horizontal container.
	HBoxContainer *hbox_container = memnew(HBoxContainer);
	hbox_container->set_h_size_flags(SIZE_EXPAND_FILL);
	panel->add_child(hbox_container);

	// Content control.
	p_control->set_h_size_flags(SIZE_EXPAND_FILL);
	hbox_container->add_child(p_control);

	// Close button.
	if (p_time > 0.0) {
		Button *close_button = memnew(Button);
		close_button->set_flat(true);
		close_button->set_icon(get_theme_icon("Close", "EditorIcons"));
		close_button->connect("pressed", callable_bind(callable_mp(this, &EditorToaster::close), panel));
		hbox_container->add_child(close_button);
	}

	toasts[panel].severity = p_severity;
	if (p_time > 0.0) {
		toasts[panel].duration = p_time;
		toasts[panel].remaining_time = p_time;
	} else {
		toasts[panel].duration = -1.0;
	}
	toasts[panel].popped = true;
	vbox_container->add_child(panel);
	_auto_hide_or_free_toasts();
	_update_vbox_position();
	_update_disable_notifications_button();
	main_button->update();

	return panel;
}

void EditorToaster::popup_str(String p_message, Severity p_severity, String p_tooltip) {
	// Check if we already have a popup with the given message.
	Control *control = nullptr;
	for (KeyValue<Control *, Toast> element : toasts) {
		if (element.value.message == p_message && element.value.severity == p_severity && element.value.tooltip == p_tooltip) {
			control = element.key;
			break;
		}
	}

	// Create a new message if needed.
	if (control == nullptr) {
		Label *label = memnew(Label);

		control = popup(label, p_severity, default_message_duration, p_tooltip);
		toasts[control].message = p_message;
		toasts[control].tooltip = p_tooltip;
		toasts[control].count = 1;
	} else {
		if (toasts[control].popped) {
			toasts[control].count += 1;
		} else {
			toasts[control].count = 1;
		}
		toasts[control].remaining_time = toasts[control].duration;
		toasts[control].popped = true;
		control->show();
		vbox_container->move_child(control, vbox_container->get_child_count());
		_auto_hide_or_free_toasts();
		_update_vbox_position();
		_update_disable_notifications_button();
		main_button->update();
	}

	// Retrieve the label back then update the text.
	Label *label = Object::cast_to<Label>(control->get_child(0)->get_child(0));
	ERR_FAIL_COND(!label);
	if (toasts[control].count == 1) {
		label->set_text(p_message);
	} else {
		label->set_text(vformat("%s (%d)", p_message, toasts[control].count));
	}
}

void EditorToaster::close(Control *p_control) {
	ERR_FAIL_COND(!toasts.has(p_control));
	toasts[p_control].remaining_time = -1.0;
	toasts[p_control].popped = false;
}

EditorToaster *EditorToaster::get_singleton() {
	return singleton;
}

EditorToaster::EditorToaster() {
	set_notify_transform(true);
	set_process_internal(true);

	// VBox.
	vbox_container = memnew(VBoxContainer);
	vbox_container->set_as_top_level(true);
	vbox_container->connect("resized", callable_mp(this, &EditorToaster::_update_vbox_position));
	add_child(vbox_container);

	// Theming (background).
	info_panel_style_background.instantiate();
	info_panel_style_background->set_corner_radius_all(stylebox_radius * EDSCALE);

	warning_panel_style_background.instantiate();
	warning_panel_style_background->set_border_width(SIDE_LEFT, stylebox_radius * EDSCALE);
	warning_panel_style_background->set_corner_radius_all(stylebox_radius * EDSCALE);

	error_panel_style_background.instantiate();
	error_panel_style_background->set_border_width(SIDE_LEFT, stylebox_radius * EDSCALE);
	error_panel_style_background->set_corner_radius_all(stylebox_radius * EDSCALE);

	Ref<StyleBoxFlat> boxes[] = { info_panel_style_background, warning_panel_style_background, error_panel_style_background };
	for (int i = 0; i < 3; i++) {
		boxes[i]->set_default_margin(SIDE_LEFT, int(stylebox_radius * 2.5));
		boxes[i]->set_default_margin(SIDE_RIGHT, int(stylebox_radius * 2.5));
		boxes[i]->set_default_margin(SIDE_TOP, 3);
		boxes[i]->set_default_margin(SIDE_BOTTOM, 3);
	}

	// Theming (progress).
	info_panel_style_progress.instantiate();
	info_panel_style_progress->set_corner_radius_all(stylebox_radius * EDSCALE);

	warning_panel_style_progress.instantiate();
	warning_panel_style_progress->set_border_width(SIDE_LEFT, stylebox_radius * EDSCALE);
	warning_panel_style_progress->set_corner_radius_all(stylebox_radius * EDSCALE);

	error_panel_style_progress.instantiate();
	error_panel_style_progress->set_border_width(SIDE_LEFT, stylebox_radius * EDSCALE);
	error_panel_style_progress->set_corner_radius_all(stylebox_radius * EDSCALE);

	// Main button.
	main_button = memnew(Button);
	main_button->set_flat(true);
	main_button->connect("pressed", callable_mp(this, &EditorToaster::_set_notifications_enabled), varray(true));
	main_button->connect("pressed", callable_mp(this, &EditorToaster::_repop_old));
	main_button->connect("draw", callable_mp(this, &EditorToaster::_draw_button));
	add_child(main_button);

	// Disable notification button.
	disable_notifications_panel = memnew(PanelContainer);
	disable_notifications_panel->set_as_top_level(true);
	disable_notifications_panel->add_theme_style_override("panel", info_panel_style_background);
	add_child(disable_notifications_panel);

	disable_notifications_button = memnew(Button);
	disable_notifications_button->set_flat(true);
	disable_notifications_button->connect("pressed", callable_mp(this, &EditorToaster::_set_notifications_enabled), varray(false));
	disable_notifications_panel->add_child(disable_notifications_button);

	// Other
	singleton = this;

	eh.errfunc = _error_handler;
	add_error_handler(&eh);
};

EditorToaster::~EditorToaster() {
	singleton = nullptr;
	remove_error_handler(&eh);
}
