/**************************************************************************/
/*  credits_roll.cpp                                                      */
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

#include "credits_roll.h"

#include "core/authors.gen.h"
#include "core/donors.gen.h"
#include "core/input/input.h"
#include "core/license.gen.h"
#include "core/string/string_builder.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/color_rect.h"
#include "scene/gui/label.h"
#include "scene/gui/texture_rect.h"

Label *CreditsRoll::_create_label(const String &p_with_text, LabelSize p_size) {
	Label *label = memnew(Label);
	label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	label->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	label->set_text(p_with_text);

	switch (p_size) {
		case LabelSize::NORMAL: {
			label->add_theme_font_size_override(SceneStringName(font_size), font_size_normal);
			label->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
		} break;

		case LabelSize::HEADER: {
			label->add_theme_font_size_override(SceneStringName(font_size), font_size_header);
			label->add_theme_font_override(SceneStringName(font), bold_font);
		} break;

		case LabelSize::BIG_HEADER: {
			label->add_theme_font_size_override(SceneStringName(font_size), font_size_big_header);
			label->add_theme_font_override(SceneStringName(font), bold_font);
		} break;
	}
	content->add_child(label);
	return label;
}

void CreditsRoll::_create_nothing(int p_size) {
	if (p_size == -1) {
		p_size = 30 * EDSCALE;
	}
	Control *c = memnew(Control);
	c->set_custom_minimum_size(Vector2(0, p_size));
	content->add_child(c);
}

String CreditsRoll::_build_string(const char *const *p_from) const {
	StringBuilder sb;

	while (*p_from) {
		sb.append(String::utf8(*p_from));
		sb.append("\n");
		p_from++;
	}
	return sb.as_string();
}

void CreditsRoll::_visibility_changed() {
	if (!is_visible()) {
		mouse_enabled = false;
		set_process_internal(false);
		set_process_input(false);
	}
}

void CreditsRoll::input(const Ref<InputEvent> &p_event) {
	// Block inputs from going elsewhere while the credits roll.
	get_tree()->get_root()->set_input_as_handled();
}

void CreditsRoll::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			connect("visibility_changed", callable_mp(this, &CreditsRoll::_visibility_changed));
		} break;

		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (project_manager) {
				project_manager->set_text(TTR("Project Manager", "Job Title"));
			}
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			const Vector2 pos = content->get_position();
			if (pos.y < -content->get_size().y - 30) {
				hide(); // No more credits left, show's over.
				break;
			}

			if (Input::get_singleton()->is_mouse_button_pressed(MouseButton::RIGHT) || Input::get_singleton()->is_action_pressed(SNAME("ui_cancel"))) {
				hide();
				break;
			}

			bool lmb = Input::get_singleton()->is_mouse_button_pressed(MouseButton::LEFT);
			if (!mouse_enabled && !lmb) {
				// Makes sure that the initial double click does not speed up text.
				mouse_enabled = true;
			}

			if ((mouse_enabled && lmb) || Input::get_singleton()->is_action_pressed(SNAME("ui_accept"))) {
				content->set_position(Vector2(pos.x, pos.y - 2000 * get_process_delta_time()));
			} else {
				content->set_position(Vector2(pos.x, pos.y - 100 * get_process_delta_time()));
			}
		} break;
	}
}

void CreditsRoll::roll_credits() {
	if (!project_manager) {
		font_size_normal = EditorNode::get_singleton()->get_editor_theme()->get_font_size("main_size", EditorStringName(EditorFonts)) * 2;
		font_size_header = font_size_normal + 10 * EDSCALE;
		font_size_big_header = font_size_header + 20 * EDSCALE;
		bold_font = EditorNode::get_singleton()->get_editor_theme()->get_font("bold", EditorStringName(EditorFonts));

		{
			const Ref<Texture2D> logo_texture = EditorNode::get_singleton()->get_editor_theme()->get_icon("Logo", EditorStringName(EditorIcons));

			TextureRect *logo = memnew(TextureRect);
			logo->set_custom_minimum_size(Vector2(0, logo_texture->get_height() * 3));
			logo->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
			logo->set_texture(logo_texture);
			content->add_child(logo);
		}

		_create_label(TTRC("Credits"), LabelSize::BIG_HEADER);

		_create_nothing();

		_create_label(TTRC("Project Founders"), LabelSize::HEADER);
		_create_label(_build_string(AUTHORS_FOUNDERS));

		_create_nothing();

		_create_label(TTRC("Lead Developer"), LabelSize::HEADER);
		_create_label(_build_string(AUTHORS_LEAD_DEVELOPERS));

		_create_nothing();

		project_manager = _create_label(TTR("Project Manager", "Job Title"), LabelSize::HEADER);
		project_manager->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
		_create_label(_build_string(AUTHORS_PROJECT_MANAGERS));

		_create_nothing();

		_create_label(TTRC("Developers"), LabelSize::HEADER);
		_create_label(_build_string(AUTHORS_DEVELOPERS));

		_create_nothing();

		_create_label(TTRC("Patrons"), LabelSize::HEADER);
		_create_label(_build_string(DONORS_PATRONS));

		_create_nothing();

		_create_label(TTRC("Platinum Sponsors"), LabelSize::HEADER);
		_create_label(_build_string(DONORS_SPONSORS_PLATINUM));

		_create_nothing();

		_create_label(TTRC("Gold Sponsors"), LabelSize::HEADER);
		_create_label(_build_string(DONORS_SPONSORS_GOLD));

		_create_nothing();

		_create_label(TTRC("Silver Sponsors"), LabelSize::HEADER);
		_create_label(_build_string(DONORS_SPONSORS_SILVER));

		_create_nothing();

		_create_label(TTRC("Diamond Members"), LabelSize::HEADER);
		_create_label(_build_string(DONORS_MEMBERS_DIAMOND));

		_create_nothing();

		_create_label(TTRC("Titanium Members"), LabelSize::HEADER);
		_create_label(_build_string(DONORS_MEMBERS_TITANIUM));

		_create_nothing();

		_create_label(TTRC("Platinum Members"), LabelSize::HEADER);
		_create_label(_build_string(DONORS_MEMBERS_PLATINUM));

		_create_nothing();

		_create_label(TTRC("Gold Members"), LabelSize::HEADER);
		_create_label(_build_string(DONORS_MEMBERS_GOLD));

		_create_nothing();
		_create_label(String::utf8(GODOT_LICENSE_TEXT));

		_create_nothing(400 * EDSCALE);
		_create_label(TTRC("Thank you for choosing Godot Engine!"), LabelSize::BIG_HEADER);
	}
	// Needs to be set here, so it stays centered even if the window is resized.
	content->set_anchors_and_offsets_preset(Control::PRESET_VCENTER_WIDE);

	Window *root = get_tree()->get_root();
	content->set_position(Vector2(content->get_position().x, root->get_size().y + 30));

	set_process_internal(true);
	set_process_input(true);
}

CreditsRoll::CreditsRoll() {
	ColorRect *background = memnew(ColorRect);
	background->set_color(Color(0, 0, 0, 1));
	background->set_anchors_and_offsets_preset(Control::PRESET_FULL_RECT);
	add_child(background);

	content = memnew(VBoxContainer);
	add_child(content);
}
