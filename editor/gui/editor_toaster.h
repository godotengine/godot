/**************************************************************************/
/*  editor_toaster.h                                                      */
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

#pragma once

#include "scene/gui/box_container.h"
#include "scene/gui/panel_container.h"

class Toast;
class Button;
class PanelContainer;
class StyleBoxFlat;

class EditorToaster : public HBoxContainer {
	GDCLASS(EditorToaster, HBoxContainer);

public:
	enum Severity {
		SEVERITY_INFO = 0,
		SEVERITY_WARNING,
		SEVERITY_ERROR,
	};

private:
	ErrorHandlerList eh;

	const double default_message_duration = 5.0;
	const int max_temporary_count = 5;

	Button *main_button = nullptr;
	PanelContainer *disable_notifications_panel = nullptr;
	Button *clear_notifications_button = nullptr;
	Button *disable_notifications_button = nullptr;

	VBoxContainer *vbox_container = nullptr;
	HashSet<Toast *> toasts;

	static void _error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type);
	static void _error_handler_impl(const String &p_file, int p_line, const String &p_error, const String &p_errorexp, bool p_editor_notify, int p_type);
	void _update_vbox_position() const;
	void _update_disable_notifications_button() const;
	void _auto_hide_or_free_toasts(bool p_clear = false);

	void _draw_button() const;
	void _draw_progress(Toast *p_toast) const;

	void _set_notifications_enabled(bool p_enabled) const;
	void _repop_old() const;
	Toast *_popup_str(const String &p_message, Severity p_severity, const String &p_tooltip);
	void _toast_theme_changed(Toast *p_toast) const;

protected:
	static void _bind_methods();

#ifndef DISABLE_DEPRECATED
	void _popup_str_bind_compat_117700(const String &p_message, Severity p_severity, const String &p_tooltip);
	static void _bind_compatibility_methods();
#endif

	static EditorToaster *singleton;

	void _notification(int p_what);

public:
	static EditorToaster *get_singleton();

	Toast *popup(Control *p_control, Severity p_severity = SEVERITY_INFO, double p_time = 0.0, const String &p_tooltip = String());
	Toast *popup_str(const String &p_message, Severity p_severity = SEVERITY_INFO, const String &p_tooltip = String());

	EditorToaster();
	~EditorToaster();
};

VARIANT_ENUM_CAST(EditorToaster::Severity);

class Toast : public PanelContainer {
	GDCLASS(Toast, PanelContainer);

protected:
	static void _bind_methods();

public:
	EditorToaster::Severity severity = EditorToaster::SEVERITY_INFO;

	// Timing.
	real_t duration = -1.0;
	real_t remaining_time = 0.0;
	bool popped = false;
	bool requires_action = false;

	// Buttons
	Button *copy_button = nullptr;
	Button *close_button = nullptr;

	// Actions
	HBoxContainer *action_container = nullptr;
	HashSet<String> actions;

	// Messages
	String message;
	String tooltip;
	int count = 0;
	Label *message_label = nullptr;
	Label *message_count_label = nullptr;

	String get_message() const;
	EditorToaster::Severity get_severity() const;

	void close();
	void instant_close();
	void copy() const;

	Toast *set_action(const String &p_label, const Callable &p_callback, const StringName &p_icon_name = StringName());
};
