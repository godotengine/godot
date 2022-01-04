/*************************************************************************/
/*  editor_toaster.h                                                     */
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

#ifndef EDITOR_TOASTER_H
#define EDITOR_TOASTER_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/popup.h"

#include "core/string/ustring.h"
#include "core/templates/local_vector.h"

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

	const int stylebox_radius = 3;

	Ref<StyleBoxFlat> info_panel_style_background;
	Ref<StyleBoxFlat> warning_panel_style_background;
	Ref<StyleBoxFlat> error_panel_style_background;

	Ref<StyleBoxFlat> info_panel_style_progress;
	Ref<StyleBoxFlat> warning_panel_style_progress;
	Ref<StyleBoxFlat> error_panel_style_progress;

	Button *main_button;
	PanelContainer *disable_notifications_panel;
	Button *disable_notifications_button;

	VBoxContainer *vbox_container;
	const int max_temporary_count = 5;
	struct Toast {
		Severity severity = SEVERITY_INFO;

		// Timing.
		real_t duration = -1.0;
		real_t remaining_time = 0.0;
		bool popped = false;

		// Messages
		String message;
		String tooltip;
		int count = 0;
	};
	Map<Control *, Toast> toasts;

	const double default_message_duration = 5.0;

	static void _error_handler(void *p_self, const char *p_func, const char *p_file, int p_line, const char *p_error, const char *p_errorexp, bool p_editor_notify, ErrorHandlerType p_type);
	void _update_vbox_position();
	void _update_disable_notifications_button();
	void _auto_hide_or_free_toasts();

	void _draw_button();
	void _draw_progress(Control *panel);

	void _set_notifications_enabled(bool p_enabled);
	void _repop_old();
	void _popup_str(String p_message, Severity p_severity, String p_tooltip);

protected:
	static EditorToaster *singleton;
	static void _bind_methods();

	void _notification(int p_what);

public:
	static EditorToaster *get_singleton();

	Control *popup(Control *p_control, Severity p_severity = SEVERITY_INFO, double p_time = 0.0, String p_tooltip = String());
	void popup_str(String p_message, Severity p_severity = SEVERITY_INFO, String p_tooltip = String());
	void close(Control *p_control);

	EditorToaster();
	~EditorToaster();
};

VARIANT_ENUM_CAST(EditorToaster::Severity);

#endif // EDITOR_TOASTER_H
