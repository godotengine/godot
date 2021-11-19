/*************************************************************************/
/*  editor_tool_drawer.cpp                                               */
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

#include "editor_tool_drawer.h"

#include "editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/texture_rect.h"

void EditorToolDrawer::set_title(const String p_title) {
	title_label->set_text(p_title);
}

void EditorToolDrawer::set_title_icon(const Ref<Texture2D> &p_icon) {
	title_icon->set_texture(p_icon);
}

void EditorToolDrawer::add_content(Control *p_content) {
	ERR_FAIL_NULL_MSG(p_content, "EditorToolDrawer content must be a valid Control node.");

	main_vb->add_child(p_content);
}

void EditorToolDrawer::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			bg_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("drawer_bg"), SNAME("EditorToolDrawer")));
			title_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("drawer_title"), SNAME("EditorToolDrawer")));
		} break;
	}
}

EditorToolDrawer::EditorToolDrawer() {
	bg_panel = memnew(PanelContainer);
	add_child(bg_panel);

	main_vb = memnew(VBoxContainer);
	bg_panel->add_child(main_vb);

	title_panel = memnew(PanelContainer);
	main_vb->add_child(title_panel);

	HBoxContainer *title_hb = memnew(HBoxContainer);
	title_hb->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	title_panel->add_child(title_hb);

	title_icon = memnew(TextureRect);
	title_icon->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
	title_hb->add_child(title_icon);

	title_label = memnew(Label);
	title_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	title_hb->add_child(title_label);
}

void EditorToolDrawerItemGroup::set_title(const String p_title) {
	title_label->set_text(p_title);
}

void EditorToolDrawerItemGroup::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			title_label->add_theme_color_override("font_color", get_theme_color(SNAME("group_title_color"), SNAME("EditorToolDrawer")));
			title_label->add_theme_style_override("normal", get_theme_stylebox(SNAME("group_title_bg"), SNAME("EditorToolDrawer")));
		} break;
	}
}

EditorToolDrawerItemGroup::EditorToolDrawerItemGroup() {
	title_label = memnew(Label);
	title_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	add_child(title_label);
}

void EditorToolDrawerContainer::add_drawer(const String p_name, const Ref<Texture2D> &p_icon, Control *p_control) {
	ERR_FAIL_COND(p_name.is_empty());
	ERR_FAIL_COND(p_icon.is_null());
	ERR_FAIL_NULL(p_control);

	DrawerItem di;

	// Create the activator button for the drawer control.
	di.button = memnew(Button);
	di.button->set_custom_minimum_size(Size2(24, 24) * EDSCALE);
	di.button->set_toggle_mode(true);
	di.button->set_tooltip(p_name);
	di.button->set_icon(p_icon);
	tool_drawer_bar->add_child(di.button);

	di.label = memnew(Label);
	di.label->set_anchors_and_offsets_preset(LayoutPreset::PRESET_CENTER_LEFT);
	di.label->set_h_grow_direction(GROW_DIRECTION_BEGIN);
	di.label->set_v_grow_direction(GROW_DIRECTION_BOTH);
	di.label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	di.label->set_text(p_name);
	di.button->add_child(di.label);

	// Create the drawer wrapper for the drawer control.
	di.drawer = memnew(EditorToolDrawer);
	di.drawer->set_title(p_name);
	di.drawer->set_title_icon(p_icon);
	di.drawer->hide();
	tool_drawer_vb->add_child(di.drawer);
	di.drawer->add_content(p_control);

	// Connect input signals.
	di.button->connect(SNAME("pressed"), callable_mp(this, &EditorToolDrawerContainer::_toggle_drawer), varray(p_control));

	bool force_display_drawers = EDITOR_GET("editors/common/always_display_drawers");
	if (force_display_drawers) {
		di.button->set_pressed(true);
		di.drawer->set_visible(true);
		di.expanded = true;
	}

	// Map the drawer control to the related nodes and data.
	tool_drawer_map[p_control] = di;

	_update_tool_drawer();
	_update_mass_toggle_visibility();
	_update_mass_toggle_icon();
}

void EditorToolDrawerContainer::remove_drawer(Control *p_control) {
	ERR_FAIL_NULL(p_control);

	if (tool_drawer_map.has(p_control)) {
		DrawerItem &di = tool_drawer_map[p_control];
		tool_drawer_map.erase(p_control);

		// Remove the activator button.
		if (di.button) {
			tool_drawer_bar->remove_child(di.button);
			memdelete(di.button);
		}

		// Remove the drawer wrapper.
		if (di.drawer) {
			tool_drawer_vb->remove_child(di.drawer);
			memdelete(di.drawer);
		}
	}

	_update_mass_toggle_visibility();
	_update_mass_toggle_icon();
}

void EditorToolDrawerContainer::set_drawer_visible(Control *p_control, bool p_visible) {
	ERR_FAIL_NULL(p_control);

	bool expanded = false;
	bool force_display_drawers = EDITOR_GET("editors/common/always_display_drawers");
	if (force_display_drawers) {
		expanded = true;
	} else if (tool_drawer_map.has(p_control)) {
		expanded = tool_drawer_map[p_control].expanded;
	}

	if (tool_drawer_map.has(p_control)) {
		DrawerItem &di = tool_drawer_map[p_control];

		if (di.button) {
			di.button->set_visible(p_visible);
			if (p_visible) {
				di.button->set_pressed(expanded);
			} else {
				di.button->set_pressed(false);
			}
		}

		if (di.drawer) {
			if (p_visible) {
				di.drawer->set_visible(expanded);
			} else {
				di.drawer->set_visible(false);
			}
		}
	}

	_update_tool_drawer();
	_update_mass_toggle_visibility();
	_update_mass_toggle_icon();
}

void EditorToolDrawerContainer::_toggle_drawer(Control *p_control) {
	bool force_display_drawers = EDITOR_GET("editors/common/always_display_drawers");
	if (force_display_drawers) {
		// Can't toggle drawers per editor setting.
		return;
	}

	ERR_FAIL_COND(!tool_drawer_map.has(p_control));
	DrawerItem &di = tool_drawer_map[p_control];
	ERR_FAIL_NULL(di.button);
	ERR_FAIL_NULL(di.drawer);

	if (di.button->is_pressed()) {
		di.drawer->set_visible(true);
		di.expanded = true;
	} else {
		di.drawer->set_visible(false);
		di.expanded = false;
	}

	_update_tool_drawer();
	_update_mass_toggle_icon();
}

void EditorToolDrawerContainer::_toggle_all_drawers() {
	bool force_display_drawers = EDITOR_GET("editors/common/always_display_drawers");
	if (force_display_drawers) {
		// Can't toggle drawers per editor setting.
		return;
	}

	bool has_collapsed = false;

	for (const KeyValue<Control *, DrawerItem> &E : tool_drawer_map) {
		if (!E.value.button || !E.value.button->is_visible()) {
			continue;
		}

		if (E.value.button->is_visible() && !E.value.button->is_pressed()) {
			has_collapsed = true;
		}
	}

	// If there are collapsed drawers, expand all. If not, collapse all.
	_set_drawers_toggled_nocheck(has_collapsed);

	_update_tool_drawer();
	_update_mass_toggle_icon();
}

void EditorToolDrawerContainer::_set_drawers_toggled_nocheck(bool p_expanded) {
	for (KeyValue<Control *, DrawerItem> &E : tool_drawer_map) {
		if (!E.value.button || !E.value.button->is_visible() || !E.value.drawer) {
			continue;
		}

		E.value.button->set_pressed(p_expanded);
		E.value.drawer->set_visible(p_expanded);
		E.value.expanded = p_expanded;
	}
}

void EditorToolDrawerContainer::_update_tool_drawer() {
	bool has_expanded = false;

	for (int i = 0; i < tool_drawer_vb->get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(tool_drawer_vb->get_child(i));
		if (!c) {
			continue;
		}

		if (c->is_visible()) {
			has_expanded = true;
			break;
		}
	}

	tool_drawer_scroll->set_visible(has_expanded);
	tool_drawer_vb->set_visible(has_expanded);

	bool display_labels = EDITOR_GET("editors/common/display_drawer_labels");
	for (const KeyValue<Control *, DrawerItem> &E : tool_drawer_map) {
		if (!E.value.label) {
			continue;
		}

		E.value.label->set_visible(display_labels && !has_expanded);
	}
}

void EditorToolDrawerContainer::_update_mass_toggle_visibility() {
	bool force_display_drawers = EDITOR_GET("editors/common/always_display_drawers");
	if (force_display_drawers) {
		tool_drawer_mass_toggle->set_visible(false);
		return;
	}

	bool has_visible = false;

	for (const KeyValue<Control *, DrawerItem> &E : tool_drawer_map) {
		if (E.value.button && E.value.button->is_visible()) {
			has_visible = true;
			break;
		}
	}

	tool_drawer_mass_toggle->set_visible(has_visible);
}

void EditorToolDrawerContainer::_update_mass_toggle_icon() {
	bool has_collapsed = false;

	for (const KeyValue<Control *, DrawerItem> &E : tool_drawer_map) {
		if (!E.value.button || !E.value.button->is_visible()) {
			continue;
		}

		if (E.value.button->is_visible() && !E.value.button->is_pressed()) {
			has_collapsed = true;
		}
	}

	if (has_collapsed) {
		tool_drawer_mass_toggle->set_tooltip(TTR("Make all tool drawers visible."));
		tool_drawer_mass_toggle->set_icon(get_theme_icon(SNAME("ToolDrawerExpand"), SNAME("EditorIcons")));
	} else {
		tool_drawer_mass_toggle->set_tooltip(TTR("Hide all tool drawers."));
		tool_drawer_mass_toggle->set_icon(get_theme_icon(SNAME("ToolDrawerCollapse"), SNAME("EditorIcons")));
	}
}

void EditorToolDrawerContainer::_update_scroll_area() {
	// This keeps the scroll container's size as small as possible to fit the drawers,
	// but at the same time keeps it from overflowing the viewport area.
	// We can't anchor SC normally, because we want the area below the visible drawers to not
	// be obstructed by the invisible part of the scroll container.

	Control *parent = Object::cast_to<Control>(get_parent());
	if (!parent) {
		return;
	}

	int top_offset = get_theme_constant(SNAME("base_top_offset"), SNAME("EditorToolDrawer"));
	int right_offset = get_theme_constant(SNAME("base_right_offset"), SNAME("EditorToolDrawer"));
	int panel_separation = get_theme_constant(SNAME("panel_separation"), SNAME("EditorToolDrawer"));
	bool force_display_drawers = EDITOR_GET("editors/common/always_display_drawers");

	float parent_height = parent->get_size().y - 2 * top_offset;
	float drawer_height = tool_drawer_vb->get_size().y;
	float max_height = MIN(drawer_height, parent_height);

	// When the scroll is somewhere in the middle and we change to a state where scrollbar is not
	// needed anymore, a weird sizing glitch happens. This prevents it.
	if (drawer_height < parent_height) {
		tool_drawer_scroll->set_v_scroll(0);
	}

	tool_drawer_scroll->set_custom_minimum_size(Size2(0, max_height));
	tool_drawer_scroll->set_anchors_and_offsets_preset(PRESET_TOP_RIGHT);
	tool_drawer_scroll->reset_size();

	float panel_offset = right_offset;
	if (!force_display_drawers) {
		panel_offset -= panel_separation + tool_drawer_bar->get_size().x;
	}
	tool_drawer_scroll->set_offset(SIDE_RIGHT, panel_offset);
	tool_drawer_scroll->set_offset(SIDE_TOP, top_offset);
}

void EditorToolDrawerContainer::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			bool force_display_drawers = EDITOR_GET("editors/common/always_display_drawers");

			tool_drawer_mass_toggle->set_visible(!force_display_drawers);
			tool_drawer_bar->set_visible(!force_display_drawers);
			if (force_display_drawers) {
				_set_drawers_toggled_nocheck(true);
			}

			int max_button_size = 0;
			for (const KeyValue<Control *, DrawerItem> &E : tool_drawer_map) {
				if (E.value.button) {
					E.value.button->add_theme_style_override("normal", get_theme_stylebox(SNAME("button_normal"), SNAME("EditorToolDrawer")));
					E.value.button->add_theme_style_override("hover", get_theme_stylebox(SNAME("button_hover"), SNAME("EditorToolDrawer")));
					E.value.button->add_theme_style_override("pressed", get_theme_stylebox(SNAME("button_pressed"), SNAME("EditorToolDrawer")));
					E.value.button->add_theme_style_override("disabled", get_theme_stylebox(SNAME("button_disabled"), SNAME("EditorToolDrawer")));

					Ref<Texture2D> button_icon = E.value.button->get_icon();
					if (button_icon.is_valid() && max_button_size < button_icon->get_size().x) {
						max_button_size = button_icon->get_size().x;
					}
				}

				if (E.value.label) {
					// This fixes some blurriness when rendering the label's text from the central anchor.
					E.value.label->set_offset(SIDE_TOP, -1);
					E.value.label->set_offset(SIDE_BOTTOM, 0);
					E.value.label->set_offset(SIDE_RIGHT, get_theme_constant(SNAME("button_label_offset"), SNAME("EditorToolDrawer")));
					E.value.label->add_theme_style_override("normal", get_theme_stylebox(SNAME("button_label"), SNAME("EditorToolDrawer")));
					E.value.label->add_theme_color_override("font_shadow_color", get_theme_color(SNAME("button_label_shadow_color"), SNAME("EditorToolDrawer")));
				}
			}

			Ref<StyleBox> button_style = get_theme_stylebox(SNAME("button_normal"), SNAME("EditorToolDrawer"));
			if (button_style.is_valid()) {
				max_button_size += button_style->get_margin(SIDE_LEFT) + button_style->get_margin(SIDE_RIGHT);
			}

			int top_offset = get_theme_constant(SNAME("base_top_offset"), SNAME("EditorToolDrawer"));
			int buttons_top_offset = get_theme_constant(SNAME("buttons_top_offset"), SNAME("EditorToolDrawer"));
			int right_offset = get_theme_constant(SNAME("base_right_offset"), SNAME("EditorToolDrawer"));
			int panel_separation = get_theme_constant(SNAME("panel_separation"), SNAME("EditorToolDrawer"));

			tool_drawer_bar->set_offset(SIDE_RIGHT, right_offset);
			tool_drawer_bar->set_offset(SIDE_TOP, buttons_top_offset);
			tool_drawer_bar->add_theme_constant_override("separation", panel_separation);

			_update_scroll_area();

			float panel_offset = right_offset;
			if (!force_display_drawers) {
				panel_offset -= panel_separation + max_button_size;
			}
			tool_drawer_scroll->set_offset(SIDE_RIGHT, panel_offset);
			tool_drawer_scroll->set_offset(SIDE_TOP, top_offset);

			tool_drawer_vb->add_theme_constant_override("separation", get_theme_constant(SNAME("drawer_separation"), SNAME("EditorToolDrawer")));

			tool_drawer_mass_toggle->set_offset(SIDE_TOP, top_offset);
			tool_drawer_mass_toggle->set_offset(SIDE_RIGHT, right_offset);
			tool_drawer_mass_toggle->add_theme_style_override("normal", get_theme_stylebox(SNAME("toggle_button_normal"), SNAME("EditorToolDrawer")));
			tool_drawer_mass_toggle->add_theme_style_override("hover", get_theme_stylebox(SNAME("toggle_button_hover"), SNAME("EditorToolDrawer")));
			tool_drawer_mass_toggle->add_theme_style_override("pressed", get_theme_stylebox(SNAME("toggle_button_pressed"), SNAME("EditorToolDrawer")));
			tool_drawer_mass_toggle->add_theme_style_override("disabled", get_theme_stylebox(SNAME("toggle_button_disabled"), SNAME("EditorToolDrawer")));

			_update_tool_drawer();
			_update_mass_toggle_visibility();
			_update_mass_toggle_icon();
		} break;

		case NOTIFICATION_READY: {
			Control *parent = Object::cast_to<Control>(get_parent());
			if (parent) {
				parent->connect("resized", callable_mp(this, &EditorToolDrawerContainer::_update_scroll_area), varray(), CONNECT_DEFERRED);
			}
			tool_drawer_vb->connect("resized", callable_mp(this, &EditorToolDrawerContainer::_update_scroll_area), varray(), CONNECT_DEFERRED);
			tool_drawer_bar->connect("resized", callable_mp(this, &EditorToolDrawerContainer::_update_scroll_area), varray(), CONNECT_DEFERRED);
		} break;
	}
}

EditorToolDrawerContainer::EditorToolDrawerContainer() {
	ED_SHORTCUT("editor/toggle_all_tool_drawers", TTR("Toggle All Tool Drawers"), Key::M);

	tool_drawer_mass_toggle = memnew(Button);
	tool_drawer_mass_toggle->set_anchors_and_offsets_preset(PRESET_TOP_RIGHT);
	tool_drawer_mass_toggle->set_h_grow_direction(GROW_DIRECTION_BEGIN);
	tool_drawer_mass_toggle->set_custom_minimum_size(Size2(24, 8) * EDSCALE);
	tool_drawer_mass_toggle->set_shortcut(ED_GET_SHORTCUT("editor/toggle_all_tool_drawers"));
	tool_drawer_mass_toggle->hide();
	add_child(tool_drawer_mass_toggle);
	tool_drawer_mass_toggle->connect("pressed", callable_mp(this, &EditorToolDrawerContainer::_toggle_all_drawers));

	tool_drawer_bar = memnew(VBoxContainer);
	tool_drawer_bar->set_anchors_and_offsets_preset(PRESET_TOP_RIGHT);
	tool_drawer_bar->set_h_grow_direction(GROW_DIRECTION_BEGIN);
	add_child(tool_drawer_bar);

	tool_drawer_scroll = memnew(ScrollContainer);
	tool_drawer_scroll->set_anchors_and_offsets_preset(PRESET_TOP_RIGHT);
	tool_drawer_scroll->set_h_grow_direction(GROW_DIRECTION_BEGIN);
	tool_drawer_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	tool_drawer_scroll->hide();
	add_child(tool_drawer_scroll);

	tool_drawer_vb = memnew(VBoxContainer);
	tool_drawer_vb->set_h_size_flags(SIZE_EXPAND_FILL);
	tool_drawer_vb->hide();
	tool_drawer_scroll->add_child(tool_drawer_vb);
}
