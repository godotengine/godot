/**************************************************************************/
/*  licenses_dialog.cpp                                                   */
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

#include "licenses_dialog.h"

#include "core/license.gen.h"
#include "core/string/string_buffer.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"
#ifdef ADVANCED_GUI_DISABLED
#include "scene/gui/scroll_container.h"
#else
#include "scene/gui/advanced/rich_text_label.h"
#endif // ADVANCED_GUI_DISABLED
#include "scene/main/viewport.h"
#include "scene/resources/style_box_flat.h"

void LicensesDialog::_close_button_pressed() {
	SceneTree::get_singleton()->set_licenses_dialog_visible(false);
}

void LicensesDialog::unhandled_key_input(const Ref<InputEvent> &p_event) {
	if (p_event->is_action_pressed(SNAME("ui_cancel"), false, true)) {
		SceneTree::get_singleton()->set_licenses_dialog_visible(false);
		Node::get_viewport()->set_input_as_handled();
	}
}

LicensesDialog::LicensesDialog() {
	// Set on the highest layer, so that nothing else can draw on top.
	set_layer(128);

	// Keep UI interactions functional even if the game is paused.
	set_process_mode(Node::PROCESS_MODE_ALWAYS);

	set_process_unhandled_key_input(true);

	MarginContainer *margin_container = memnew(MarginContainer);
	margin_container->set_anchors_preset(Control::PRESET_FULL_RECT);
	const float default_base_scale = margin_container->get_theme_default_base_scale();
	const float default_font_size = margin_container->get_theme_default_font_size();
	margin_container->add_theme_constant_override("margin_top", Math::round(20 * default_base_scale));
	margin_container->add_theme_constant_override("margin_right", Math::round(20 * default_base_scale));
	margin_container->add_theme_constant_override("margin_bottom", Math::round(20 * default_base_scale));
	margin_container->add_theme_constant_override("margin_left", Math::round(20 * default_base_scale));
	add_child(margin_container);

	PanelContainer *panel_container = memnew(PanelContainer);
	margin_container->add_child(panel_container);

	MarginContainer *inner_margin_container = memnew(MarginContainer);
	inner_margin_container->add_theme_constant_override("margin_top", Math::round(10 * default_base_scale));
	inner_margin_container->add_theme_constant_override("margin_right", Math::round(10 * default_base_scale));
	inner_margin_container->add_theme_constant_override("margin_bottom", Math::round(10 * default_base_scale));
	inner_margin_container->add_theme_constant_override("margin_left", Math::round(10 * default_base_scale));
	panel_container->add_child(inner_margin_container);

	VBoxContainer *vbox_container = memnew(VBoxContainer);
	vbox_container->add_theme_constant_override("separation", Math::round(10 * default_base_scale));
	inner_margin_container->add_child(vbox_container);

	Label *title_label = memnew(Label);
	title_label->set_text(RTR("Third-party notices"));
	title_label->add_theme_font_size_override(SceneStringName(font_size), Math::round(1.333 * default_font_size));
	title_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	vbox_container->add_child(title_label);

	// Based on `editor_about.cpp` with references to TreeItem removed,
	// as we only have the "All Components" view here. A preamble is also added.
	StringBuffer<> long_text;
	long_text += RTR("This project is powered by Godot Engine, which relies on a number of third-party free and open source libraries, all compatible with the terms of its MIT license. The following is an exhaustive list of all such third-party components with their respective copyright statements and license terms.") + "\n\n";

	long_text += RTR("Components:") + "\n\n";

	for (int component_index = 0; component_index < COPYRIGHT_INFO_COUNT; component_index++) {
		const ComponentCopyright &component = COPYRIGHT_INFO[component_index];
		const String component_name = String::utf8(component.name);
		long_text += "- " + component_name;
		for (int part_index = 0; part_index < component.part_count; part_index++) {
			const ComponentCopyrightPart &part = component.parts[part_index];
			String copyright;
			for (int copyright_index = 0; copyright_index < part.copyright_count; copyright_index++) {
				copyright += String::utf8("\n    \xc2\xa9 ") + String::utf8(part.copyright_statements[copyright_index]);
			}
			long_text += copyright;
			String license = "\n    License: " + String::utf8(part.license) + "\n";
			long_text += license + "\n\n";
		}
	}

	long_text += RTR("Licenses:") + "\n\n";

	for (int i = 0; i < LICENSE_COUNT; i++) {
		const String licensename = String::utf8(LICENSE_NAMES[i]);
		long_text += "- " + licensename + "\n";
		const String licensebody = String::utf8(LICENSE_BODIES[i]);
		long_text += "    " + licensebody.replace("\n", "\n    ") + "\n\n";
	}

	// Create a background for the scrollable area with the license text.
	Ref<StyleBoxFlat> background;
	background.instantiate();
	background->set_bg_color(Color(0, 0, 0, 0.5));
	background->set_content_margin_all(Math::round(10 * default_base_scale));

	Button *close_button = memnew(Button);
	close_button->set_text(RTR("Close"));
	close_button->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	close_button->set_custom_minimum_size(Vector2(100, 40) * default_base_scale);
	close_button->connect(SceneStringName(pressed), callable_mp(this, &LicensesDialog::_close_button_pressed));

	const float license_font_size = Math::round(0.75 * default_font_size);
#ifdef ADVANCED_GUI_DISABLED
	Label *text_control = memnew(Label);
	text_control->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	text_control->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	text_control->add_theme_font_size_override("font_size", license_font_size);
	ScrollContainer *scroll = memnew(ScrollContainer);
	scroll->add_child(text_control);
	scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	scroll->add_theme_style_override("panel", background);
	vbox_container->add_child(scroll);

	// Since `ScrollContainer` doesn't allow for keyboard navigation, give focus to the close button.
	callable_mp((Control *)close_button, &Control::grab_focus).call_deferred();
#else
	RichTextLabel *text_control = memnew(RichTextLabel);
	text_control->set_threaded(true);
	text_control->set_focus_mode(Control::FOCUS_ALL);
	text_control->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	text_control->add_theme_style_override(CoreStringName(normal), background);
	text_control->add_theme_font_size_override("normal_font_size", license_font_size);
	vbox_container->add_child(text_control);

	// Allow for keyboard navigation by grabbing focus immediately on the scrollable control.
	callable_mp((Control *)text_control, &Control::grab_focus).call_deferred();
#endif // ADVANCED_GUI_DISABLED
	text_control->set_text(long_text);

	vbox_container->add_child(close_button);
}
