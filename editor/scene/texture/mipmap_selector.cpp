/**************************************************************************/
/*  mipmap_selector.cpp                                                   */
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

#include "mipmap_selector.h"

#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/style_box_flat.h"

MipmapSelector::MipmapSelector() {
	toggle_button = memnew(Button);
	toggle_button->set_flat(true);
	toggle_button->set_toggle_mode(true);
	toggle_button->connect(SceneStringName(toggled), callable_mp(this, &MipmapSelector::on_toggled));
	toggle_button->add_theme_style_override("focus", memnew(StyleBoxEmpty));
	add_child(toggle_button);

	panel = memnew(PanelContainer);
	panel->hide();

	container = memnew(HBoxContainer);
	container->add_theme_constant_override("separation", 0);

	mipmap_buttons.instantiate();
	mipmap_buttons->connect(SceneStringName(pressed), callable_mp(this, &MipmapSelector::on_mipmap_button_toggled));

	// Use a bit of transparency to be less distracting.
	set_modulate(Color(1, 1, 1, 0.7));

	panel->add_child(container);

	add_child(panel);
}

void MipmapSelector::_notification(int p_what) {
	if (p_what == NOTIFICATION_THEME_CHANGED) {
		// PanelContainer's background is invisible in the editor. We need a background.
		// And we need this in turn because buttons don't look good without background (for example, hover is transparent).
		Ref<StyleBox> bg_style = get_theme_stylebox(SceneStringName(panel), "TabContainer");
		ERR_FAIL_COND(bg_style.is_null());
		bg_style = bg_style->duplicate();
		// The default content margin makes the widget become a bit too large. It should be like mini-toolbar.
		const float editor_scale = EditorScale::get_scale();
		bg_style->set_content_margin(SIDE_LEFT, 1.0f * editor_scale);
		bg_style->set_content_margin(SIDE_RIGHT, 1.0f * editor_scale);
		bg_style->set_content_margin(SIDE_TOP, 1.0f * editor_scale);
		bg_style->set_content_margin(SIDE_BOTTOM, 1.0f * editor_scale);
		panel->add_theme_style_override(SceneStringName(panel), bg_style);

		Ref<Texture2D> icon = get_editor_theme_icon(SNAME("ImageTexture"));
		toggle_button->set_button_icon(icon);
	}
}

void MipmapSelector::set_mipmap_count(int count) {
	for (int i = 0; i <= count; ++i) {
		create_button(itos(i), container);
	}
}

int MipmapSelector::get_selected_mipmap() const {
	return mipmap_buttons->get_pressed_button()->get_index();
}

void MipmapSelector::on_mipmap_button_toggled(BaseButton *button) {
	emit_signal("selected_mipmap_changed");
}

void MipmapSelector::create_button(const String &p_text, Control *p_parent) {
	Button *button = memnew(Button);
	button->set_text(p_text);
	button->set_toggle_mode(true);
	// button->set_pressed(false);

	// Don't show focus, it stands out too much and remains visible which can be confusing.
	button->add_theme_style_override("focus", memnew(StyleBoxEmpty));

	// Make it look similar to toolbar buttons.
	button->set_theme_type_variation(SceneStringName(FlatButton));

	button->set_button_group(mipmap_buttons);
	p_parent->add_child(button);
}

void MipmapSelector::on_toggled(bool p_pressed) {
	panel->set_visible(p_pressed);
}

void MipmapSelector::_bind_methods() {
	ADD_SIGNAL(MethodInfo("selected_mipmap_changed"));
}
