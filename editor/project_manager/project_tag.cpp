/**************************************************************************/
/*  project_tag.cpp                                                       */
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

#include "project_tag.h"

#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/color_rect.h"

void ProjectTag::_notification(int p_what) {
	// Needed to get the correct icons in ProjectList and Manage Project Tags dialog on Project Manager Startup.
	if (is_add_button && p_what == NOTIFICATION_THEME_CHANGED) {
		aux_button->set_button_icon(get_editor_theme_icon(SNAME("Add")));
	}
	if (!is_add_button && is_aux_button_visible && p_what == NOTIFICATION_THEME_CHANGED) {
		aux_button->set_button_icon(get_theme_icon(SNAME("close"), SNAME("TabBar")));
	}
	// HACK: Can't be set in constructor because `get_size()` would return empty.
	// This logic should be migrated once `Button` utilizes internal labels.
	if (p_what == NOTIFICATION_READY) {
		button->set_custom_minimum_size(button->get_size());
		button->set_text_overrun_behavior(TextServer::OverrunBehavior::OVERRUN_TRIM_ELLIPSIS);
	}
}

void ProjectTag::update_aux_button_visibility(bool p_show) {
	if (p_show) {
		aux_button->set_button_icon(get_theme_icon(SNAME("close"), SNAME("TabBar")));
	} else {
		Ref<Texture2D> icon;
		aux_button->set_button_icon(icon);
	}
}

void ProjectTag::connect_button_to(const Callable &p_callable) {
	button->connect(SceneStringName(pressed), p_callable, CONNECT_DEFERRED);
}

void ProjectTag::connect_aux_button_to(const Callable &p_callable) {
	aux_button->connect(SceneStringName(pressed), p_callable, CONNECT_DEFERRED);
}

Size2 ProjectTag::get_aux_button_size() const {
	return aux_button->get_custom_maximum_size();
}

const String ProjectTag::get_tag() const {
	return tag_string;
}

ProjectTag::ProjectTag(const String &p_text, bool p_aux_is_add_button, bool p_aux_button_visible) {
	add_theme_constant_override(SNAME("separation"), 0);
	set_v_size_flags(SIZE_SHRINK_CENTER);
	set_custom_maximum_size(Vector2(356, 24) * EDSCALE);

	tag_string = p_text;
	is_add_button = p_aux_is_add_button;
	is_aux_button_visible = p_aux_button_visible;

	Color tag_color = Color(1, 0, 0);
	tag_color.set_ok_hsl_s(0.8);
	tag_color.set_ok_hsl_h(float(p_text.hash() * 10001 % UINT32_MAX) / float(UINT32_MAX));
	set_self_modulate(tag_color);

	ColorRect *cr = memnew(ColorRect);
	add_child(cr);
	cr->set_custom_minimum_size(Vector2(4, 0) * EDSCALE);
	cr->set_color(tag_color);
	cr->set_mouse_filter(MOUSE_FILTER_PASS);

	button = memnew(Button);
	add_child(button);
	;
	button->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	button->set_text(p_text.capitalize());
	button->set_tooltip_text(p_text.capitalize());
	button->set_focus_mode(FOCUS_ACCESSIBILITY);
	button->set_accessibility_name(vformat(TTR("Project Tag: %s"), p_text));
	button->set_theme_type_variation(SNAME("ProjectTagButton"));
	button->set_custom_maximum_size(Vector2(332, 24) * EDSCALE);
	button->set_mouse_filter(MOUSE_FILTER_PASS);

	aux_button = memnew(Button);
	add_child(aux_button);
	aux_button->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	aux_button->set_focus_mode(FOCUS_ACCESSIBILITY);
	aux_button->set_accessibility_name(vformat(TTR("Remove Tag: %s"), p_text));
	aux_button->set_icon_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	aux_button->set_theme_type_variation(SNAME("ProjectTagAuxButton"));
	aux_button->set_custom_minimum_size(Vector2(20, 24) * EDSCALE);
	aux_button->set_custom_maximum_size(Vector2(20, 24) * EDSCALE);
	aux_button->set_mouse_filter(MOUSE_FILTER_PASS);
}
