/**************************************************************************/
/*  editor_dock_manager.cpp                                               */
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

#include "editor_dock_manager.h"

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/popup.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"
#include "scene/main/window.h"

#include "editor/editor_command_palette.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/filesystem_dock.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/themes/editor_scale.h"
#include "editor/window_wrapper.h"

EditorDockManager *EditorDockManager::singleton = nullptr;

static const char *META_TOGGLE_SHORTCUT = "_toggle_shortcut";

void DockSplitContainer::_update_visibility() {
	if (is_updating) {
		return;
	}
	is_updating = true;
	bool any_visible = false;
	for (int i = 0; i < 2; i++) {
		Control *split = get_containable_child(i);
		if (split && split->is_visible()) {
			any_visible = true;
			break;
		}
	}
	set_visible(any_visible);
	is_updating = false;
}

void DockSplitContainer::add_child_notify(Node *p_child) {
	SplitContainer::add_child_notify(p_child);

	Control *child_control = nullptr;
	for (int i = 0; i < 2; i++) {
		Control *split = get_containable_child(i);
		if (p_child == split) {
			child_control = split;
			break;
		}
	}
	if (!child_control) {
		return;
	}

	child_control->connect("visibility_changed", callable_mp(this, &DockSplitContainer::_update_visibility));
	_update_visibility();
}

void DockSplitContainer::remove_child_notify(Node *p_child) {
	SplitContainer::remove_child_notify(p_child);

	Control *child_control = nullptr;
	for (int i = 0; i < 2; i++) {
		Control *split = get_containable_child(i);
		if (p_child == split) {
			child_control = split;
			break;
		}
	}
	if (!child_control) {
		return;
	}

	child_control->disconnect("visibility_changed", callable_mp(this, &DockSplitContainer::_update_visibility));
	_update_visibility();
}

void EditorDockManager::_dock_select_popup_theme_changed() {
	if (dock_float) {
		dock_float->set_icon(dock_select_popup->get_editor_theme_icon(SNAME("MakeFloating")));
	}
	if (dock_select_popup->is_layout_rtl()) {
		dock_tab_move_left->set_icon(dock_select_popup->get_editor_theme_icon(SNAME("Forward")));
		dock_tab_move_right->set_icon(dock_select_popup->get_editor_theme_icon(SNAME("Back")));
	} else {
		dock_tab_move_left->set_icon(dock_select_popup->get_editor_theme_icon(SNAME("Back")));
		dock_tab_move_right->set_icon(dock_select_popup->get_editor_theme_icon(SNAME("Forward")));
	}

	dock_to_bottom->set_icon(dock_select_popup->get_editor_theme_icon(SNAME("ControlAlignBottomWide")));
}

void EditorDockManager::_dock_popup_exit() {
	dock_select_rect_over_idx = -1;
	dock_select->queue_redraw();
}

void EditorDockManager::_dock_pre_popup(int p_dock_slot) {
	dock_popup_selected_idx = p_dock_slot;
	dock_bottom_selected_idx = -1;

	if (bool(dock_slot[p_dock_slot]->get_current_tab_control()->call("_can_dock_horizontal"))) {
		dock_to_bottom->show();
	} else {
		dock_to_bottom->hide();
	}

	if (dock_float) {
		dock_float->show();
	}
	dock_tab_move_right->show();
	dock_tab_move_left->show();
}

void EditorDockManager::_dock_move_left() {
	if (dock_popup_selected_idx < 0 || dock_popup_selected_idx >= DOCK_SLOT_MAX) {
		return;
	}
	Control *current_ctl = dock_slot[dock_popup_selected_idx]->get_tab_control(dock_slot[dock_popup_selected_idx]->get_current_tab());
	Control *prev_ctl = dock_slot[dock_popup_selected_idx]->get_tab_control(dock_slot[dock_popup_selected_idx]->get_current_tab() - 1);
	if (!current_ctl || !prev_ctl) {
		return;
	}
	dock_slot[dock_popup_selected_idx]->move_child(current_ctl, prev_ctl->get_index(false));
	dock_select->queue_redraw();
	_edit_current();
	emit_signal(SNAME("layout_changed"));
}

void EditorDockManager::_dock_move_right() {
	if (dock_popup_selected_idx < 0 || dock_popup_selected_idx >= DOCK_SLOT_MAX) {
		return;
	}
	Control *current_ctl = dock_slot[dock_popup_selected_idx]->get_tab_control(dock_slot[dock_popup_selected_idx]->get_current_tab());
	Control *next_ctl = dock_slot[dock_popup_selected_idx]->get_tab_control(dock_slot[dock_popup_selected_idx]->get_current_tab() + 1);
	if (!current_ctl || !next_ctl) {
		return;
	}
	dock_slot[dock_popup_selected_idx]->move_child(next_ctl, current_ctl->get_index(false));
	dock_select->queue_redraw();
	_edit_current();
	emit_signal(SNAME("layout_changed"));
}

void EditorDockManager::_dock_select_input(const Ref<InputEvent> &p_input) {
	Ref<InputEventMouse> me = p_input;

	if (me.is_valid()) {
		Vector2 point = me->get_position();

		int nrect = -1;
		for (int i = 0; i < DOCK_SLOT_MAX; i++) {
			if (dock_select_rect[i].has_point(point)) {
				nrect = i;
				break;
			}
		}

		if (nrect != dock_select_rect_over_idx) {
			dock_select->queue_redraw();
			dock_select_rect_over_idx = nrect;
		}

		if (nrect == -1) {
			return;
		}

		Ref<InputEventMouseButton> mb = me;

		if (mb.is_valid() && mb->get_button_index() == MouseButton::LEFT && mb->is_pressed()) {
			if (dock_bottom_selected_idx != -1) {
				EditorNode::get_bottom_panel()->remove_item(bottom_docks[dock_bottom_selected_idx]);

				bottom_docks[dock_bottom_selected_idx]->call("_set_dock_horizontal", false);

				dock_slot[nrect]->add_child(bottom_docks[dock_bottom_selected_idx]);
				dock_slot[nrect]->show();
				bottom_docks.remove_at(dock_bottom_selected_idx);
				dock_bottom_selected_idx = -1;
				dock_popup_selected_idx = nrect; // Move to dock popup selected.
				dock_select->queue_redraw();

				update_dock_slots_visibility(true);

				_edit_current();
				emit_signal(SNAME("layout_changed"));
			}

			if (dock_popup_selected_idx != nrect) {
				dock_slot[nrect]->move_tab_from_tab_container(dock_slot[dock_popup_selected_idx], dock_slot[dock_popup_selected_idx]->get_current_tab(), dock_slot[nrect]->get_tab_count());

				if (dock_slot[dock_popup_selected_idx]->get_tab_count() == 0) {
					dock_slot[dock_popup_selected_idx]->hide();
				} else {
					dock_slot[dock_popup_selected_idx]->set_current_tab(0);
				}

				dock_popup_selected_idx = nrect;
				dock_slot[nrect]->show();
				dock_select->queue_redraw();

				update_dock_slots_visibility(true);

				_edit_current();
				emit_signal(SNAME("layout_changed"));
			}
		}
	}
}

void EditorDockManager::_dock_select_draw() {
	Size2 s = dock_select->get_size();
	s.y /= 2.0;
	s.x /= 6.0;

	Color used = Color(0.6, 0.6, 0.6, 0.8);
	Color used_selected = Color(0.8, 0.8, 0.8, 0.8);
	Color tab_selected = dock_select->get_theme_color(SNAME("mono_color"), EditorStringName(Editor));
	Color unused = used;
	unused.a = 0.4;
	Color unusable = unused;
	unusable.a = 0.1;

	Rect2 unr(s.x * 2, 0, s.x * 2, s.y * 2);
	unr.position += Vector2(2, 5);
	unr.size -= Vector2(4, 7);

	dock_select->draw_rect(unr, unusable);

	dock_tab_move_left->set_disabled(true);
	dock_tab_move_right->set_disabled(true);

	if (dock_popup_selected_idx != -1 && dock_slot[dock_popup_selected_idx]->get_tab_count()) {
		dock_tab_move_left->set_disabled(dock_slot[dock_popup_selected_idx]->get_current_tab() == 0);
		dock_tab_move_right->set_disabled(dock_slot[dock_popup_selected_idx]->get_current_tab() >= dock_slot[dock_popup_selected_idx]->get_tab_count() - 1);
	}

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		Vector2 ofs;

		switch (i) {
			case DOCK_SLOT_LEFT_UL: {
			} break;
			case DOCK_SLOT_LEFT_BL: {
				ofs.y += s.y;
			} break;
			case DOCK_SLOT_LEFT_UR: {
				ofs.x += s.x;
			} break;
			case DOCK_SLOT_LEFT_BR: {
				ofs += s;
			} break;
			case DOCK_SLOT_RIGHT_UL: {
				ofs.x += s.x * 4;
			} break;
			case DOCK_SLOT_RIGHT_BL: {
				ofs.x += s.x * 4;
				ofs.y += s.y;

			} break;
			case DOCK_SLOT_RIGHT_UR: {
				ofs.x += s.x * 4;
				ofs.x += s.x;

			} break;
			case DOCK_SLOT_RIGHT_BR: {
				ofs.x += s.x * 4;
				ofs += s;

			} break;
		}

		Rect2 r(ofs, s);
		dock_select_rect[i] = r;
		r.position += Vector2(2, 5);
		r.size -= Vector2(4, 7);

		if (i == dock_select_rect_over_idx) {
			dock_select->draw_rect(r, used_selected);
		} else if (dock_slot[i]->get_tab_count() == 0) {
			dock_select->draw_rect(r, unused);
		} else {
			dock_select->draw_rect(r, used);
		}

		for (int j = 0; j < MIN(3, dock_slot[i]->get_tab_count()); j++) {
			int xofs = (r.size.width / 3) * j;
			Color c = used;
			if (i == dock_popup_selected_idx && (dock_slot[i]->get_current_tab() > 3 || dock_slot[i]->get_current_tab() == j)) {
				c = tab_selected;
			}
			dock_select->draw_rect(Rect2(2 + ofs.x + xofs, ofs.y, r.size.width / 3 - 1, 3), c);
		}
	}
}

void EditorDockManager::_dock_split_dragged(int p_offset) {
	EditorNode::get_singleton()->save_editor_layout_delayed();
}

void EditorDockManager::_dock_tab_changed(int p_tab) {
	// Update visibility but don't set current tab.
	update_dock_slots_visibility(true);
}

void EditorDockManager::_edit_current() {
	EditorNode::get_singleton()->edit_current();
}

void EditorDockManager::_dock_floating_close_request(WindowWrapper *p_wrapper) {
	int dock_slot_num = p_wrapper->get_meta("dock_slot");
	int dock_slot_index = p_wrapper->get_meta("dock_index");

	// Give back the dock to the original owner.
	Control *dock = p_wrapper->release_wrapped_control();

	int target_index = MIN(dock_slot_index, dock_slot[dock_slot_num]->get_tab_count());
	dock_slot[dock_slot_num]->add_child(dock);
	dock_slot[dock_slot_num]->move_child(dock, target_index);
	dock_slot[dock_slot_num]->set_current_tab(target_index);

	floating_docks.erase(p_wrapper);
	p_wrapper->queue_free();

	update_dock_slots_visibility(true);

	_edit_current();
}

void EditorDockManager::_dock_make_selected_float() {
	Control *dock = dock_slot[dock_popup_selected_idx]->get_current_tab_control();
	_dock_make_float(dock, dock_popup_selected_idx);

	dock_select_popup->hide();
	_edit_current();
}

void EditorDockManager::bottom_dock_show_placement_popup(const Rect2i &p_position, Control *p_dock) {
	dock_bottom_selected_idx = bottom_docks.find(p_dock);
	ERR_FAIL_COND(dock_bottom_selected_idx == -1);
	dock_popup_selected_idx = -1;
	dock_to_bottom->hide();

	Vector2 popup_pos = p_position.position;
	popup_pos.y += p_position.size.height;

	if (!EditorNode::get_singleton()->get_gui_base()->is_layout_rtl()) {
		popup_pos.x -= dock_select_popup->get_size().width;
		popup_pos.x += p_position.size.width;
	}
	dock_select_popup->set_position(popup_pos);
	dock_select_popup->popup();
	if (dock_float) {
		dock_float->hide();
	}
	dock_tab_move_right->hide();
	dock_tab_move_left->hide();
}

void EditorDockManager::_dock_move_selected_to_bottom() {
	Control *dock = dock_slot[dock_popup_selected_idx]->get_current_tab_control();
	dock_slot[dock_popup_selected_idx]->remove_child(dock);

	dock->call("_set_dock_horizontal", true);

	bottom_docks.push_back(dock);

	// Force docks moved to the bottom to appear first in the list, and give them their associated shortcut to toggle their bottom panel.
	EditorNode::get_bottom_panel()->add_item(dock->get_name(), dock, dock->get_meta(META_TOGGLE_SHORTCUT), true);

	dock_select_popup->hide();
	update_dock_slots_visibility(true);
	_edit_current();
	emit_signal(SNAME("layout_changed"));

	EditorNode::get_bottom_panel()->make_item_visible(dock);
}

void EditorDockManager::_dock_make_float(Control *p_dock, int p_slot_index, bool p_show_window) {
	ERR_FAIL_NULL(p_dock);

	Size2 borders = Size2(4, 4) * EDSCALE;
	// Remember size and position before removing it from the main window.
	Size2 dock_size = p_dock->get_size() + borders * 2;
	Point2 dock_screen_pos = p_dock->get_screen_position();

	int dock_index = p_dock->get_index() - 1;
	dock_slot[p_slot_index]->remove_child(p_dock);

	WindowWrapper *wrapper = memnew(WindowWrapper);
	wrapper->set_window_title(vformat(TTR("%s - Godot Engine"), p_dock->get_name()));
	wrapper->set_margins_enabled(true);

	EditorNode::get_singleton()->get_gui_base()->add_child(wrapper);

	wrapper->set_wrapped_control(p_dock);
	wrapper->set_meta("dock_slot", p_slot_index);
	wrapper->set_meta("dock_index", dock_index);
	wrapper->set_meta("dock_name", p_dock->get_name().operator String());
	p_dock->show();

	wrapper->connect("window_close_requested", callable_mp(this, &EditorDockManager::_dock_floating_close_request).bind(wrapper));

	dock_select_popup->hide();

	if (p_show_window) {
		wrapper->restore_window(Rect2i(dock_screen_pos, dock_size), EditorNode::get_singleton()->get_gui_base()->get_window()->get_current_screen());
	}

	update_dock_slots_visibility(true);

	floating_docks.push_back(wrapper);

	_edit_current();
}

void EditorDockManager::_restore_floating_dock(const Dictionary &p_dock_dump, Control *p_dock, int p_slot_index) {
	WindowWrapper *wrapper = Object::cast_to<WindowWrapper>(p_dock);
	if (!wrapper) {
		_dock_make_float(p_dock, p_slot_index, false);
		wrapper = floating_docks[floating_docks.size() - 1];
	}

	wrapper->restore_window_from_saved_position(
			p_dock_dump.get("window_rect", Rect2i()),
			p_dock_dump.get("window_screen", -1),
			p_dock_dump.get("window_screen_rect", Rect2i()));
}

void EditorDockManager::save_docks_to_config(Ref<ConfigFile> p_layout, const String &p_section) const {
	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		String names;
		for (int j = 0; j < dock_slot[i]->get_tab_count(); j++) {
			String name = dock_slot[i]->get_tab_control(j)->get_name();
			if (!names.is_empty()) {
				names += ",";
			}
			names += name;
		}

		String config_key = "dock_" + itos(i + 1);

		if (p_layout->has_section_key(p_section, config_key)) {
			p_layout->erase_section_key(p_section, config_key);
		}

		if (!names.is_empty()) {
			p_layout->set_value(p_section, config_key, names);
		}

		int selected_tab_idx = dock_slot[i]->get_current_tab();
		if (selected_tab_idx >= 0) {
			p_layout->set_value(p_section, "dock_" + itos(i + 1) + "_selected_tab_idx", selected_tab_idx);
		}
	}

	Dictionary floating_docks_dump;

	for (WindowWrapper *wrapper : floating_docks) {
		Control *dock = wrapper->get_wrapped_control();

		Dictionary dock_dump;
		dock_dump["window_rect"] = wrapper->get_window_rect();

		int screen = wrapper->get_window_screen();
		dock_dump["window_screen"] = wrapper->get_window_screen();
		dock_dump["window_screen_rect"] = DisplayServer::get_singleton()->screen_get_usable_rect(screen);

		String name = dock->get_name();
		floating_docks_dump[name] = dock_dump;

		int dock_slot_id = wrapper->get_meta("dock_slot");
		String config_key = "dock_" + itos(dock_slot_id + 1);

		String names = p_layout->get_value(p_section, config_key, "");
		if (names.is_empty()) {
			names = name;
		} else {
			names += "," + name;
		}
		p_layout->set_value(p_section, config_key, names);
	}

	p_layout->set_value(p_section, "dock_floating", floating_docks_dump);

	Array bottom_docks_dump;

	for (Control *bdock : bottom_docks) {
		bottom_docks_dump.push_back(bdock->get_name());
	}

	p_layout->set_value(p_section, "dock_bottom", bottom_docks_dump);

	for (int i = 0; i < vsplits.size(); i++) {
		if (vsplits[i]->is_visible_in_tree()) {
			p_layout->set_value(p_section, "dock_split_" + itos(i + 1), vsplits[i]->get_split_offset());
		}
	}

	for (int i = 0; i < hsplits.size(); i++) {
		p_layout->set_value(p_section, "dock_hsplit_" + itos(i + 1), hsplits[i]->get_split_offset());
	}

	FileSystemDock::get_singleton()->save_layout_to_config(p_layout, p_section);
}

void EditorDockManager::load_docks_from_config(Ref<ConfigFile> p_layout, const String &p_section) {
	Dictionary floating_docks_dump = p_layout->get_value(p_section, "dock_floating", Dictionary());

	bool restore_window_on_load = EDITOR_GET("interface/multi_window/restore_windows_on_load");

	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		if (!p_layout->has_section_key(p_section, "dock_" + itos(i + 1))) {
			continue;
		}

		Vector<String> names = String(p_layout->get_value(p_section, "dock_" + itos(i + 1))).split(",");

		for (int j = names.size() - 1; j >= 0; j--) {
			String name = names[j];

			// FIXME: Find it, in a horribly inefficient way.
			int atidx = -1;
			int bottom_idx = -1;
			Control *node = nullptr;
			for (int k = 0; k < DOCK_SLOT_MAX; k++) {
				if (!dock_slot[k]->has_node(name)) {
					continue;
				}
				node = Object::cast_to<Control>(dock_slot[k]->get_node(name));
				if (!node) {
					continue;
				}
				atidx = k;
				break;
			}

			if (atidx == -1) {
				// Try floating docks.
				for (WindowWrapper *wrapper : floating_docks) {
					if (wrapper->get_meta("dock_name") == name) {
						if (restore_window_on_load && floating_docks_dump.has(name)) {
							_restore_floating_dock(floating_docks_dump[name], wrapper, i);
						} else {
							atidx = wrapper->get_meta("dock_slot");
							node = wrapper->get_wrapped_control();
							wrapper->set_window_enabled(false);
						}
						break;
					}
				}
			}

			if (atidx == -1) {
				// Try bottom docks.
				for (Control *bdock : bottom_docks) {
					if (bdock->get_name() == name) {
						node = bdock;
						bottom_idx = bottom_docks.find(node);
						break;
					}
				}
			}

			if (!node) {
				// Well, it's not anywhere.
				continue;
			}

			if (atidx == i) {
				dock_slot[i]->move_child(node, 0);
			} else if (atidx != -1) {
				dock_slot[i]->set_block_signals(true);
				dock_slot[atidx]->set_block_signals(true);
				dock_slot[i]->move_tab_from_tab_container(dock_slot[atidx], dock_slot[atidx]->get_tab_idx_from_control(node), 0);
				dock_slot[i]->set_block_signals(false);
				dock_slot[atidx]->set_block_signals(false);
			} else if (bottom_idx != -1) {
				bottom_docks.erase(node);
				EditorNode::get_bottom_panel()->remove_item(node);
				dock_slot[i]->add_child(node);
				node->call("_set_dock_horizontal", false);
			}

			WindowWrapper *wrapper = Object::cast_to<WindowWrapper>(node);
			if (restore_window_on_load && floating_docks_dump.has(name)) {
				if (!dock_slot[i]->is_tab_hidden(dock_slot[i]->get_tab_idx_from_control(node))) {
					_restore_floating_dock(floating_docks_dump[name], node, i);
				}
			} else if (wrapper) {
				wrapper->set_window_enabled(false);
			}
		}

		if (!p_layout->has_section_key(p_section, "dock_" + itos(i + 1) + "_selected_tab_idx")) {
			continue;
		}

		int selected_tab_idx = p_layout->get_value(p_section, "dock_" + itos(i + 1) + "_selected_tab_idx");
		if (selected_tab_idx >= 0 && selected_tab_idx < dock_slot[i]->get_tab_count()) {
			callable_mp(dock_slot[i], &TabContainer::set_current_tab).call_deferred(selected_tab_idx);
		}
	}

	Array dock_bottom = p_layout->get_value(p_section, "dock_bottom", Array());
	for (int i = 0; i < dock_bottom.size(); i++) {
		const String &name = dock_bottom[i];
		// FIXME: Find it, in a horribly inefficient way.
		int atidx = -1;
		Control *node = nullptr;
		for (int k = 0; k < DOCK_SLOT_MAX; k++) {
			if (!dock_slot[k]->has_node(name)) {
				continue;
			}
			node = Object::cast_to<Control>(dock_slot[k]->get_node(name));
			if (!node) {
				continue;
			}
			atidx = k;
			break;
		}

		if (atidx == -1) {
			// Try floating docks.
			for (WindowWrapper *wrapper : floating_docks) {
				if (wrapper->get_meta("dock_name") == name) {
					atidx = wrapper->get_meta("dock_slot");
					node = wrapper->get_wrapped_control();
					wrapper->set_window_enabled(false);
					break;
				}
			}
		}

		if (node) {
			dock_slot[atidx]->remove_child(node);

			node->call("_set_dock_horizontal", true);

			bottom_docks.push_back(node);
			// Force docks moved to the bottom to appear first in the list, and give them their associated shortcut to toggle their bottom panel.
			EditorNode::get_bottom_panel()->add_item(node->get_name(), node, node->get_meta(META_TOGGLE_SHORTCUT), true);
		}
	}

	for (int i = 0; i < vsplits.size(); i++) {
		if (!p_layout->has_section_key(p_section, "dock_split_" + itos(i + 1))) {
			continue;
		}

		int ofs = p_layout->get_value(p_section, "dock_split_" + itos(i + 1));
		vsplits[i]->set_split_offset(ofs);
	}

	for (int i = 0; i < hsplits.size(); i++) {
		if (!p_layout->has_section_key(p_section, "dock_hsplit_" + itos(i + 1))) {
			continue;
		}
		int ofs = p_layout->get_value(p_section, "dock_hsplit_" + itos(i + 1));
		hsplits[i]->set_split_offset(ofs);
	}

	update_dock_slots_visibility(false);

	FileSystemDock::get_singleton()->load_layout_from_config(p_layout, p_section);
}

void EditorDockManager::update_dock_slots_visibility(bool p_keep_selected_tabs) {
	if (!docks_visible) {
		for (int i = 0; i < DOCK_SLOT_MAX; i++) {
			dock_slot[i]->hide();
		}
	} else {
		for (int i = 0; i < DOCK_SLOT_MAX; i++) {
			int first_tab_visible = -1;
			for (int j = 0; j < dock_slot[i]->get_tab_count(); j++) {
				if (!dock_slot[i]->is_tab_hidden(j)) {
					first_tab_visible = j;
					break;
				}
			}
			if (first_tab_visible >= 0) {
				dock_slot[i]->show();
				if (p_keep_selected_tabs) {
					int current_tab = dock_slot[i]->get_current_tab();
					if (dock_slot[i]->is_tab_hidden(current_tab)) {
						dock_slot[i]->set_block_signals(true);
						dock_slot[i]->select_next_available();
						dock_slot[i]->set_block_signals(false);
					}
				} else {
					dock_slot[i]->set_block_signals(true);
					dock_slot[i]->set_current_tab(first_tab_visible);
					dock_slot[i]->set_block_signals(false);
				}
			} else {
				dock_slot[i]->hide();
			}
		}
	}
}

void EditorDockManager::close_all_floating_docks() {
	for (WindowWrapper *wrapper : floating_docks) {
		wrapper->set_window_enabled(false);
	}
}

void EditorDockManager::add_control_to_dock(DockSlot p_slot, Control *p_control, const String &p_name, const Ref<Shortcut> &p_shortcut) {
	ERR_FAIL_INDEX(p_slot, DOCK_SLOT_MAX);
	p_control->set_meta(META_TOGGLE_SHORTCUT, p_shortcut);
	dock_slot[p_slot]->add_child(p_control);
	if (!p_name.is_empty()) {
		dock_slot[p_slot]->set_tab_title(dock_slot[p_slot]->get_tab_idx_from_control(p_control), p_name);
	}
}

void EditorDockManager::remove_control_from_dock(Control *p_control) {
	// If the dock is floating, close it first.
	for (WindowWrapper *wrapper : floating_docks) {
		if (p_control == wrapper->get_wrapped_control()) {
			wrapper->set_window_enabled(false);
			break;
		}
	}

	Control *dock = nullptr;
	for (int i = 0; i < DOCK_SLOT_MAX; i++) {
		if (p_control->get_parent() == dock_slot[i]) {
			dock = dock_slot[i];
			break;
		}
	}

	ERR_FAIL_NULL_MSG(dock, "Control is not in a dock.");

	dock->remove_child(p_control);
	update_dock_slots_visibility();
}

void EditorDockManager::set_docks_visible(bool p_show) {
	docks_visible = p_show;
	update_dock_slots_visibility(true);
}

bool EditorDockManager::are_docks_visible() const {
	return docks_visible;
}

void EditorDockManager::add_vsplit(DockSplitContainer *p_split) {
	vsplits.push_back(p_split);
	p_split->connect("dragged", callable_mp(this, &EditorDockManager::_dock_split_dragged));
}

void EditorDockManager::add_hsplit(DockSplitContainer *p_split) {
	hsplits.push_back(p_split);
	p_split->connect("dragged", callable_mp(this, &EditorDockManager::_dock_split_dragged));
}

void EditorDockManager::register_dock_slot(DockSlot p_dock_slot, TabContainer *p_tab_container) {
	ERR_FAIL_NULL(p_tab_container);
	ERR_FAIL_INDEX(p_dock_slot, DOCK_SLOT_MAX);

	dock_slot[p_dock_slot] = p_tab_container;

	p_tab_container->set_custom_minimum_size(Size2(170, 0) * EDSCALE);
	p_tab_container->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	p_tab_container->set_popup(dock_select_popup);
	p_tab_container->connect("pre_popup_pressed", callable_mp(this, &EditorDockManager::_dock_pre_popup).bind(p_dock_slot));
	p_tab_container->set_drag_to_rearrange_enabled(true);
	p_tab_container->set_tabs_rearrange_group(1);
	p_tab_container->connect("tab_changed", callable_mp(this, &EditorDockManager::_dock_tab_changed));
	p_tab_container->set_use_hidden_tabs_for_min_size(true);
}

int EditorDockManager::get_vsplit_count() const {
	return vsplits.size();
}

void EditorDockManager::_bind_methods() {
	ADD_SIGNAL(MethodInfo("layout_changed"));
}

EditorDockManager::EditorDockManager() {
	singleton = this;

	dock_select_popup = memnew(PopupPanel);
	EditorNode::get_singleton()->get_gui_base()->add_child(dock_select_popup);
	VBoxContainer *dock_vb = memnew(VBoxContainer);
	dock_select_popup->add_child(dock_vb);
	dock_select_popup->connect("theme_changed", callable_mp(this, &EditorDockManager::_dock_select_popup_theme_changed));

	HBoxContainer *dock_hb = memnew(HBoxContainer);
	dock_tab_move_left = memnew(Button);
	dock_tab_move_left->set_flat(true);
	dock_tab_move_left->set_focus_mode(Control::FOCUS_NONE);
	dock_tab_move_left->connect("pressed", callable_mp(this, &EditorDockManager::_dock_move_left));
	dock_hb->add_child(dock_tab_move_left);

	Label *dock_label = memnew(Label);
	dock_label->set_text(TTR("Dock Position"));
	dock_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dock_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	dock_hb->add_child(dock_label);

	dock_tab_move_right = memnew(Button);
	dock_tab_move_right->set_flat(true);
	dock_tab_move_right->set_focus_mode(Control::FOCUS_NONE);
	dock_tab_move_right->connect("pressed", callable_mp(this, &EditorDockManager::_dock_move_right));

	dock_hb->add_child(dock_tab_move_right);
	dock_vb->add_child(dock_hb);

	dock_select = memnew(Control);
	dock_select->set_custom_minimum_size(Size2(128, 64) * EDSCALE);
	dock_select->connect("gui_input", callable_mp(this, &EditorDockManager::_dock_select_input));
	dock_select->connect("draw", callable_mp(this, &EditorDockManager::_dock_select_draw));
	dock_select->connect("mouse_exited", callable_mp(this, &EditorDockManager::_dock_popup_exit));
	dock_select->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	dock_vb->add_child(dock_select);

	dock_float = memnew(Button);
	dock_float->set_text(TTR("Make Floating"));
	if (!EditorNode::get_singleton()->is_multi_window_enabled()) {
		dock_float->set_disabled(true);
		dock_float->set_tooltip_text(EditorNode::get_singleton()->get_multiwindow_support_tooltip_text());
	} else {
		dock_float->set_tooltip_text(TTR("Make this dock floating."));
	}
	dock_float->set_focus_mode(Control::FOCUS_NONE);
	dock_float->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dock_float->connect("pressed", callable_mp(this, &EditorDockManager::_dock_make_selected_float));
	dock_vb->add_child(dock_float);

	dock_to_bottom = memnew(Button);
	dock_to_bottom->set_text(TTR("Move to Bottom"));
	dock_to_bottom->set_focus_mode(Control::FOCUS_NONE);
	dock_to_bottom->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	dock_to_bottom->connect("pressed", callable_mp(this, &EditorDockManager::_dock_move_selected_to_bottom));
	dock_to_bottom->hide();
	dock_vb->add_child(dock_to_bottom);

	dock_select_popup->reset_size();
}
