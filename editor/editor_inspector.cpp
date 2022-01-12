/*************************************************************************/
/*  editor_inspector.cpp                                                 */
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

#include "editor_inspector.h"

#include "array_property_edit.h"
#include "core/os/keyboard.h"
#include "dictionary_property_edit.h"
#include "editor/doc_tools.h"
#include "editor_feature_profile.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "multi_node_edit.h"
#include "scene/property_utils.h"
#include "scene/resources/packed_scene.h"

Size2 EditorProperty::get_minimum_size() const {
	Size2 ms;
	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Tree"));
	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Tree"));
	ms.height = font->get_height(font_size);

	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}
		if (!c->is_visible()) {
			continue;
		}
		if (c == bottom_editor) {
			continue;
		}

		Size2 minsize = c->get_combined_minimum_size();
		ms.width = MAX(ms.width, minsize.width);
		ms.height = MAX(ms.height, minsize.height);
	}

	if (keying) {
		Ref<Texture2D> key = get_theme_icon(SNAME("Key"), SNAME("EditorIcons"));
		ms.width += key->get_width() + get_theme_constant(SNAME("hseparator"), SNAME("Tree"));
	}

	if (deletable) {
		Ref<Texture2D> key = get_theme_icon(SNAME("Close"), SNAME("EditorIcons"));
		ms.width += key->get_width() + get_theme_constant(SNAME("hseparator"), SNAME("Tree"));
	}

	if (checkable) {
		Ref<Texture2D> check = get_theme_icon(SNAME("checked"), SNAME("CheckBox"));
		ms.width += check->get_width() + get_theme_constant(SNAME("hseparation"), SNAME("CheckBox")) + get_theme_constant(SNAME("hseparator"), SNAME("Tree"));
	}

	if (bottom_editor != nullptr && bottom_editor->is_visible()) {
		ms.height += get_theme_constant(SNAME("vseparation"));
		Size2 bems = bottom_editor->get_combined_minimum_size();
		//bems.width += get_constant("item_margin", "Tree");
		ms.height += bems.height;
		ms.width = MAX(ms.width, bems.width);
	}

	return ms;
}

void EditorProperty::emit_changed(const StringName &p_property, const Variant &p_value, const StringName &p_field, bool p_changing) {
	Variant args[4] = { p_property, p_value, p_field, p_changing };
	const Variant *argptrs[4] = { &args[0], &args[1], &args[2], &args[3] };

	cache[p_property] = p_value;
	emit_signal(SNAME("property_changed"), (const Variant **)argptrs, 4);
}

void EditorProperty::_notification(int p_what) {
	if (p_what == NOTIFICATION_SORT_CHILDREN) {
		Size2 size = get_size();
		Rect2 rect;
		Rect2 bottom_rect;

		right_child_rect = Rect2();
		bottom_child_rect = Rect2();

		{
			int child_room = size.width * (1.0 - split_ratio);
			Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Tree"));
			int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Tree"));
			int height = font->get_height(font_size);
			bool no_children = true;

			//compute room needed
			for (int i = 0; i < get_child_count(); i++) {
				Control *c = Object::cast_to<Control>(get_child(i));
				if (!c) {
					continue;
				}
				if (c->is_set_as_top_level()) {
					continue;
				}
				if (c == bottom_editor) {
					continue;
				}

				Size2 minsize = c->get_combined_minimum_size();
				child_room = MAX(child_room, minsize.width);
				height = MAX(height, minsize.height);
				no_children = false;
			}

			if (no_children) {
				text_size = size.width;
				rect = Rect2(size.width - 1, 0, 1, height);
			} else {
				text_size = MAX(0, size.width - (child_room + 4 * EDSCALE));
				if (is_layout_rtl()) {
					rect = Rect2(1, 0, child_room, height);
				} else {
					rect = Rect2(size.width - child_room, 0, child_room, height);
				}
			}

			if (bottom_editor) {
				int m = 0; //get_constant("item_margin", "Tree");

				bottom_rect = Rect2(m, rect.size.height + get_theme_constant(SNAME("vseparation")), size.width - m, bottom_editor->get_combined_minimum_size().height);
			}

			if (keying) {
				Ref<Texture2D> key;

				if (use_keying_next()) {
					key = get_theme_icon(SNAME("KeyNext"), SNAME("EditorIcons"));
				} else {
					key = get_theme_icon(SNAME("Key"), SNAME("EditorIcons"));
				}

				rect.size.x -= key->get_width() + get_theme_constant(SNAME("hseparator"), SNAME("Tree"));
				if (is_layout_rtl()) {
					rect.position.x += key->get_width() + get_theme_constant(SNAME("hseparator"), SNAME("Tree"));
				}

				if (no_children) {
					text_size -= key->get_width() + 4 * EDSCALE;
				}
			}

			if (deletable) {
				Ref<Texture2D> close;

				close = get_theme_icon(SNAME("Close"), SNAME("EditorIcons"));

				rect.size.x -= close->get_width() + get_theme_constant(SNAME("hseparator"), SNAME("Tree"));

				if (is_layout_rtl()) {
					rect.position.x += close->get_width() + get_theme_constant(SNAME("hseparator"), SNAME("Tree"));
				}

				if (no_children) {
					text_size -= close->get_width() + 4 * EDSCALE;
				}
			}
		}

		//set children
		for (int i = 0; i < get_child_count(); i++) {
			Control *c = Object::cast_to<Control>(get_child(i));
			if (!c) {
				continue;
			}
			if (c->is_set_as_top_level()) {
				continue;
			}
			if (c == bottom_editor) {
				continue;
			}

			fit_child_in_rect(c, rect);
			right_child_rect = rect;
		}

		if (bottom_editor) {
			fit_child_in_rect(bottom_editor, bottom_rect);
			bottom_child_rect = bottom_rect;
		}

		update(); //need to redraw text
	}

	if (p_what == NOTIFICATION_DRAW) {
		Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Tree"));
		int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Tree"));
		Color dark_color = get_theme_color(SNAME("dark_color_2"), SNAME("Editor"));
		bool rtl = is_layout_rtl();

		Size2 size = get_size();
		if (bottom_editor) {
			size.height = bottom_editor->get_offset(SIDE_TOP);
		} else if (label_reference) {
			size.height = label_reference->get_size().height;
		}

		Ref<StyleBox> sb;
		if (selected) {
			sb = get_theme_stylebox(SNAME("bg_selected"));
		} else {
			sb = get_theme_stylebox(SNAME("bg"));
		}

		draw_style_box(sb, Rect2(Vector2(), size));

		if (draw_top_bg && right_child_rect != Rect2()) {
			draw_rect(right_child_rect, dark_color);
		}
		if (bottom_child_rect != Rect2()) {
			draw_rect(bottom_child_rect, dark_color);
		}

		Color color;
		if (draw_warning) {
			color = get_theme_color(is_read_only() ? SNAME("readonly_warning_color") : SNAME("warning_color"));
		} else {
			color = get_theme_color(is_read_only() ? SNAME("readonly_color") : SNAME("property_color"));
		}
		if (label.find(".") != -1) {
			// FIXME: Move this to the project settings editor, as this is only used
			// for project settings feature tag overrides.
			color.a = 0.5;
		}

		int ofs = get_theme_constant(SNAME("font_offset"));
		int text_limit = text_size;

		if (checkable) {
			Ref<Texture2D> checkbox;
			if (checked) {
				checkbox = get_theme_icon(SNAME("GuiChecked"), SNAME("EditorIcons"));
			} else {
				checkbox = get_theme_icon(SNAME("GuiUnchecked"), SNAME("EditorIcons"));
			}

			Color color2(1, 1, 1);
			if (check_hover) {
				color2.r *= 1.2;
				color2.g *= 1.2;
				color2.b *= 1.2;
			}
			check_rect = Rect2(ofs, ((size.height - checkbox->get_height()) / 2), checkbox->get_width(), checkbox->get_height());
			if (rtl) {
				draw_texture(checkbox, Vector2(size.width - check_rect.position.x - checkbox->get_width(), check_rect.position.y), color2);
			} else {
				draw_texture(checkbox, check_rect.position, color2);
			}
			ofs += get_theme_constant(SNAME("hseparator"), SNAME("Tree")) + checkbox->get_width() + get_theme_constant(SNAME("hseparation"), SNAME("CheckBox"));
			text_limit -= ofs;
		} else {
			check_rect = Rect2();
		}

		if (can_revert && !is_read_only()) {
			Ref<Texture2D> reload_icon = get_theme_icon(SNAME("ReloadSmall"), SNAME("EditorIcons"));
			text_limit -= reload_icon->get_width() + get_theme_constant(SNAME("hseparator"), SNAME("Tree")) * 2;
			revert_rect = Rect2(text_limit + get_theme_constant(SNAME("hseparator"), SNAME("Tree")), (size.height - reload_icon->get_height()) / 2, reload_icon->get_width(), reload_icon->get_height());

			Color color2(1, 1, 1);
			if (revert_hover) {
				color2.r *= 1.2;
				color2.g *= 1.2;
				color2.b *= 1.2;
			}
			if (rtl) {
				draw_texture(reload_icon, Vector2(size.width - revert_rect.position.x - reload_icon->get_width(), revert_rect.position.y), color2);
			} else {
				draw_texture(reload_icon, revert_rect.position, color2);
			}
		} else {
			revert_rect = Rect2();
		}

		if (!pin_hidden && pinned) {
			Ref<Texture2D> pinned_icon = get_theme_icon(SNAME("Pin"), SNAME("EditorIcons"));
			int margin_w = get_theme_constant(SNAME("hseparator"), SNAME("Tree")) * 2;
			int total_icon_w = margin_w + pinned_icon->get_width();
			int text_w = font->get_string_size(label, font_size, rtl ? HORIZONTAL_ALIGNMENT_RIGHT : HORIZONTAL_ALIGNMENT_LEFT, text_limit - total_icon_w).x;
			int y = (size.height - pinned_icon->get_height()) / 2;
			if (rtl) {
				draw_texture(pinned_icon, Vector2(size.width - ofs - text_w - total_icon_w, y), color);
			} else {
				draw_texture(pinned_icon, Vector2(ofs + text_w + margin_w, y), color);
			}
			text_limit -= total_icon_w;
		}

		int v_ofs = (size.height - font->get_height(font_size)) / 2;
		if (rtl) {
			draw_string(font, Point2(size.width - ofs - text_limit, v_ofs + font->get_ascent(font_size)), label, HORIZONTAL_ALIGNMENT_RIGHT, text_limit, font_size, color);
		} else {
			draw_string(font, Point2(ofs, v_ofs + font->get_ascent(font_size)), label, HORIZONTAL_ALIGNMENT_LEFT, text_limit, font_size, color);
		}

		if (keying) {
			Ref<Texture2D> key;

			if (use_keying_next()) {
				key = get_theme_icon(SNAME("KeyNext"), SNAME("EditorIcons"));
			} else {
				key = get_theme_icon(SNAME("Key"), SNAME("EditorIcons"));
			}

			ofs = size.width - key->get_width() - get_theme_constant(SNAME("hseparator"), SNAME("Tree"));

			Color color2(1, 1, 1);
			if (keying_hover) {
				color2.r *= 1.2;
				color2.g *= 1.2;
				color2.b *= 1.2;
			}
			keying_rect = Rect2(ofs, ((size.height - key->get_height()) / 2), key->get_width(), key->get_height());
			if (rtl) {
				draw_texture(key, Vector2(size.width - keying_rect.position.x - key->get_width(), keying_rect.position.y), color2);
			} else {
				draw_texture(key, keying_rect.position, color2);
			}

		} else {
			keying_rect = Rect2();
		}

		if (deletable) {
			Ref<Texture2D> close;

			close = get_theme_icon(SNAME("Close"), SNAME("EditorIcons"));

			ofs = size.width - close->get_width() - get_theme_constant(SNAME("hseparator"), SNAME("Tree"));

			Color color2(1, 1, 1);
			if (delete_hover) {
				color2.r *= 1.2;
				color2.g *= 1.2;
				color2.b *= 1.2;
			}
			delete_rect = Rect2(ofs, ((size.height - close->get_height()) / 2), close->get_width(), close->get_height());
			if (rtl) {
				draw_texture(close, Vector2(size.width - delete_rect.position.x - close->get_width(), delete_rect.position.y), color2);
			} else {
				draw_texture(close, delete_rect.position, color2);
			}
		} else {
			delete_rect = Rect2();
		}
	}
}

void EditorProperty::set_label(const String &p_label) {
	label = p_label;
	update();
}

String EditorProperty::get_label() const {
	return label;
}

Object *EditorProperty::get_edited_object() {
	return object;
}

StringName EditorProperty::get_edited_property() {
	return property;
}

void EditorProperty::update_property() {
	GDVIRTUAL_CALL(_update_property);
}

void EditorProperty::_set_read_only(bool p_read_only) {
}

void EditorProperty::set_read_only(bool p_read_only) {
	read_only = p_read_only;
	_set_read_only(p_read_only);
}

bool EditorProperty::is_read_only() const {
	return read_only;
}

Variant EditorPropertyRevert::get_property_revert_value(Object *p_object, const StringName &p_property, bool *r_is_valid) {
	if (p_object->has_method("property_can_revert") && p_object->call("property_can_revert", p_property)) {
		if (r_is_valid) {
			*r_is_valid = true;
		}
		return p_object->call("property_get_revert", p_property);
	}

	return PropertyUtils::get_property_default_value(p_object, p_property, r_is_valid);
}

bool EditorPropertyRevert::can_property_revert(Object *p_object, const StringName &p_property) {
	bool is_valid_revert = false;
	Variant revert_value = EditorPropertyRevert::get_property_revert_value(p_object, p_property, &is_valid_revert);
	if (!is_valid_revert) {
		return false;
	}
	Variant current_value = p_object->get(p_property);
	return PropertyUtils::is_property_value_different(current_value, revert_value);
}

void EditorProperty::update_revert_and_pin_status() {
	if (property == StringName()) {
		return; //no property, so nothing to do
	}

	bool new_pinned = false;
	if (can_pin) {
		Node *node = Object::cast_to<Node>(object);
		CRASH_COND(!node);
		new_pinned = node->is_property_pinned(property);
	}
	bool new_can_revert = EditorPropertyRevert::can_property_revert(object, property) && !is_read_only();

	if (new_can_revert != can_revert || new_pinned != pinned) {
		can_revert = new_can_revert;
		pinned = new_pinned;
		update();
	}
}

bool EditorProperty::use_keying_next() const {
	List<PropertyInfo> plist;
	object->get_property_list(&plist, true);

	for (List<PropertyInfo>::Element *I = plist.front(); I; I = I->next()) {
		PropertyInfo &p = I->get();

		if (p.name == property) {
			return (p.usage & PROPERTY_USAGE_KEYING_INCREMENTS);
		}
	}

	return false;
}

void EditorProperty::set_checkable(bool p_checkable) {
	checkable = p_checkable;
	update();
	queue_sort();
}

bool EditorProperty::is_checkable() const {
	return checkable;
}

void EditorProperty::set_checked(bool p_checked) {
	checked = p_checked;
	update();
}

bool EditorProperty::is_checked() const {
	return checked;
}

void EditorProperty::set_draw_warning(bool p_draw_warning) {
	draw_warning = p_draw_warning;
	update();
}

void EditorProperty::set_keying(bool p_keying) {
	keying = p_keying;
	update();
	queue_sort();
}

void EditorProperty::set_deletable(bool p_deletable) {
	deletable = p_deletable;
	update();
	queue_sort();
}

bool EditorProperty::is_deletable() const {
	return deletable;
}

bool EditorProperty::is_keying() const {
	return keying;
}

bool EditorProperty::is_draw_warning() const {
	return draw_warning;
}

void EditorProperty::_focusable_focused(int p_index) {
	if (!selectable) {
		return;
	}
	bool already_selected = selected;
	selected = true;
	selected_focusable = p_index;
	update();
	if (!already_selected && selected) {
		emit_signal(SNAME("selected"), property, selected_focusable);
	}
}

void EditorProperty::add_focusable(Control *p_control) {
	p_control->connect("focus_entered", callable_mp(this, &EditorProperty::_focusable_focused), varray(focusables.size()));
	focusables.push_back(p_control);
}

void EditorProperty::select(int p_focusable) {
	bool already_selected = selected;

	if (p_focusable >= 0) {
		ERR_FAIL_INDEX(p_focusable, focusables.size());
		focusables[p_focusable]->grab_focus();
	} else {
		selected = true;
		update();
	}

	if (!already_selected && selected) {
		emit_signal(SNAME("selected"), property, selected_focusable);
	}
}

void EditorProperty::deselect() {
	selected = false;
	selected_focusable = -1;
	update();
}

bool EditorProperty::is_selected() const {
	return selected;
}

void EditorProperty::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (property == StringName()) {
		return;
	}

	Ref<InputEventMouse> me = p_event;

	if (me.is_valid()) {
		Vector2 mpos = me->get_position();
		if (is_layout_rtl()) {
			mpos.x = get_size().x - mpos.x;
		}
		bool button_left = (me->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE;

		bool new_keying_hover = keying_rect.has_point(mpos) && !button_left;
		if (new_keying_hover != keying_hover) {
			keying_hover = new_keying_hover;
			update();
		}

		bool new_delete_hover = delete_rect.has_point(mpos) && !button_left;
		if (new_delete_hover != delete_hover) {
			delete_hover = new_delete_hover;
			update();
		}

		bool new_revert_hover = revert_rect.has_point(mpos) && !button_left;
		if (new_revert_hover != revert_hover) {
			revert_hover = new_revert_hover;
			update();
		}

		bool new_check_hover = check_rect.has_point(mpos) && !button_left;
		if (new_check_hover != check_hover) {
			check_hover = new_check_hover;
			update();
		}
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		Vector2 mpos = mb->get_position();
		if (is_layout_rtl()) {
			mpos.x = get_size().x - mpos.x;
		}

		if (!selected && selectable) {
			selected = true;
			emit_signal(SNAME("selected"), property, -1);
			update();
		}

		if (keying_rect.has_point(mpos)) {
			emit_signal(SNAME("property_keyed"), property, use_keying_next());

			if (use_keying_next()) {
				if (property == "frame_coords" && (object->is_class("Sprite2D") || object->is_class("Sprite3D"))) {
					Vector2i new_coords = object->get(property);
					new_coords.x++;
					if (new_coords.x >= object->get("hframes").operator int64_t()) {
						new_coords.x = 0;
						new_coords.y++;
					}

					call_deferred(SNAME("emit_changed"), property, new_coords, "", false);
				} else {
					call_deferred(SNAME("emit_changed"), property, object->get(property).operator int64_t() + 1, "", false);
				}

				call_deferred(SNAME("update_property"));
			}
		}
		if (delete_rect.has_point(mpos)) {
			emit_signal(SNAME("property_deleted"), property);
		}

		if (revert_rect.has_point(mpos)) {
			bool is_valid_revert = false;
			Variant revert_value = EditorPropertyRevert::get_property_revert_value(object, property, &is_valid_revert);
			ERR_FAIL_COND(!is_valid_revert);
			emit_changed(property, revert_value);
			update_property();
		}

		if (check_rect.has_point(mpos)) {
			checked = !checked;
			update();
			emit_signal(SNAME("property_checked"), property, checked);
		}
	} else if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::RIGHT) {
		_update_popup();
		menu->set_position(get_screen_position() + get_local_mouse_position());
		menu->reset_size();
		menu->popup();
		select();
		return;
	}
}

void EditorProperty::unhandled_key_input(const Ref<InputEvent> &p_event) {
	if (!selected || !p_event->is_pressed()) {
		return;
	}

	const Ref<InputEventKey> k = p_event;

	if (k.is_valid() && k->is_pressed()) {
		if (ED_IS_SHORTCUT("property_editor/copy_property", p_event)) {
			menu_option(MENU_COPY_PROPERTY);
			accept_event();
		} else if (ED_IS_SHORTCUT("property_editor/paste_property", p_event) && !is_read_only()) {
			menu_option(MENU_PASTE_PROPERTY);
			accept_event();
		} else if (ED_IS_SHORTCUT("property_editor/copy_property_path", p_event)) {
			menu_option(MENU_COPY_PROPERTY_PATH);
			accept_event();
		}
	}
}

const Color *EditorProperty::_get_property_colors() {
	const Color base = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
	const float saturation = base.get_s() * 0.75;
	const float value = base.get_v();

	static Color c[4];
	c[0].set_hsv(0.0 / 3.0 + 0.05, saturation, value);
	c[1].set_hsv(1.0 / 3.0 + 0.05, saturation, value);
	c[2].set_hsv(2.0 / 3.0 + 0.05, saturation, value);
	c[3].set_hsv(1.5 / 3.0 + 0.05, saturation, value);
	return c;
}

void EditorProperty::set_label_reference(Control *p_control) {
	label_reference = p_control;
}

void EditorProperty::set_bottom_editor(Control *p_control) {
	bottom_editor = p_control;
}

bool EditorProperty::is_cache_valid() const {
	if (object) {
		for (const KeyValue<StringName, Variant> &E : cache) {
			bool valid;
			Variant value = object->get(E.key, &valid);
			if (!valid || value != E.value) {
				return false;
			}
		}
	}
	return true;
}
void EditorProperty::update_cache() {
	cache.clear();
	if (object && property != StringName()) {
		bool valid;
		Variant value = object->get(property, &valid);
		if (valid) {
			cache[property] = value;
		}
	}
}
Variant EditorProperty::get_drag_data(const Point2 &p_point) {
	if (property == StringName()) {
		return Variant();
	}

	Dictionary dp;
	dp["type"] = "obj_property";
	dp["object"] = object;
	dp["property"] = property;
	dp["value"] = object->get(property);

	Label *label = memnew(Label);
	label->set_text(property);
	set_drag_preview(label);
	return dp;
}

void EditorProperty::set_use_folding(bool p_use_folding) {
	use_folding = p_use_folding;
}

bool EditorProperty::is_using_folding() const {
	return use_folding;
}

void EditorProperty::expand_all_folding() {
}

void EditorProperty::collapse_all_folding() {
}

void EditorProperty::set_selectable(bool p_selectable) {
	selectable = p_selectable;
}

bool EditorProperty::is_selectable() const {
	return selectable;
}

void EditorProperty::set_name_split_ratio(float p_ratio) {
	split_ratio = p_ratio;
}

float EditorProperty::get_name_split_ratio() const {
	return split_ratio;
}

void EditorProperty::set_object_and_property(Object *p_object, const StringName &p_property) {
	object = p_object;
	property = p_property;
	_update_pin_flags();
}

static bool _is_value_potential_override(Node *p_node, const String &p_property) {
	// Consider a value is potentially overriding another if either of the following is true:
	// a) The node is foreign (inheriting or an instance), so the original value may come from another scene.
	// b) The node belongs to the scene, but the original value comes from somewhere but the builtin class (i.e., a script).
	Node *edited_scene = EditorNode::get_singleton()->get_edited_scene();
	Vector<SceneState::PackState> states_stack = PropertyUtils::get_node_states_stack(p_node, edited_scene);
	if (states_stack.size()) {
		return true;
	} else {
		bool is_valid_default = false;
		bool is_class_default = false;
		PropertyUtils::get_property_default_value(p_node, p_property, &is_valid_default, &states_stack, false, nullptr, &is_class_default);
		return !is_class_default;
	}
}

void EditorProperty::_update_pin_flags() {
	can_pin = false;
	pin_hidden = true;
	if (read_only) {
		return;
	}
	if (Node *node = Object::cast_to<Node>(object)) {
		// Avoid errors down the road by ignoring nodes which are not part of a scene
		if (!node->get_owner()) {
			bool is_scene_root = false;
			for (int i = 0; i < EditorNode::get_singleton()->get_editor_data().get_edited_scene_count(); ++i) {
				if (EditorNode::get_singleton()->get_editor_data().get_edited_scene_root(i) == node) {
					is_scene_root = true;
					break;
				}
			}
			if (!is_scene_root) {
				return;
			}
		}
		if (!_is_value_potential_override(node, property)) {
			return;
		}
		pin_hidden = false;
		{
			Set<StringName> storable_properties;
			node->get_storable_properties(storable_properties);
			if (storable_properties.has(node->get_property_store_alias(property))) {
				can_pin = true;
			}
		}
	}
}

Control *EditorProperty::make_custom_tooltip(const String &p_text) const {
	tooltip_text = p_text;
	EditorHelpBit *help_bit = memnew(EditorHelpBit);
	//help_bit->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("TooltipPanel")));
	help_bit->get_rich_text()->set_fixed_size_to_width(360 * EDSCALE);

	String text;
	PackedStringArray slices = p_text.split("::", false);
	if (!slices.is_empty()) {
		String property_name = slices[0].strip_edges();
		text = TTR("Property:") + " [u][b]" + property_name + "[/b][/u]";

		if (slices.size() > 1) {
			String property_doc = slices[1].strip_edges();
			if (property_name != property_doc) {
				text += "\n" + property_doc;
			}
		}
		help_bit->call_deferred(SNAME("set_text"), text); //hack so it uses proper theme once inside scene
	}

	return help_bit;
}

String EditorProperty::get_tooltip_text() const {
	return tooltip_text;
}

void EditorProperty::menu_option(int p_option) {
	switch (p_option) {
		case MENU_COPY_PROPERTY: {
			EditorNode::get_singleton()->get_inspector()->set_property_clipboard(object->get(property));
		} break;
		case MENU_PASTE_PROPERTY: {
			emit_changed(property, EditorNode::get_singleton()->get_inspector()->get_property_clipboard());
		} break;
		case MENU_COPY_PROPERTY_PATH: {
			DisplayServer::get_singleton()->clipboard_set(property);
		} break;
		case MENU_PIN_VALUE: {
			emit_signal(SNAME("property_pinned"), property, !pinned);
			update();
		} break;
	}
}

void EditorProperty::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_label", "text"), &EditorProperty::set_label);
	ClassDB::bind_method(D_METHOD("get_label"), &EditorProperty::get_label);

	ClassDB::bind_method(D_METHOD("set_read_only", "read_only"), &EditorProperty::set_read_only);
	ClassDB::bind_method(D_METHOD("is_read_only"), &EditorProperty::is_read_only);

	ClassDB::bind_method(D_METHOD("set_checkable", "checkable"), &EditorProperty::set_checkable);
	ClassDB::bind_method(D_METHOD("is_checkable"), &EditorProperty::is_checkable);

	ClassDB::bind_method(D_METHOD("set_checked", "checked"), &EditorProperty::set_checked);
	ClassDB::bind_method(D_METHOD("is_checked"), &EditorProperty::is_checked);

	ClassDB::bind_method(D_METHOD("set_draw_warning", "draw_warning"), &EditorProperty::set_draw_warning);
	ClassDB::bind_method(D_METHOD("is_draw_warning"), &EditorProperty::is_draw_warning);

	ClassDB::bind_method(D_METHOD("set_keying", "keying"), &EditorProperty::set_keying);
	ClassDB::bind_method(D_METHOD("is_keying"), &EditorProperty::is_keying);

	ClassDB::bind_method(D_METHOD("set_deletable", "deletable"), &EditorProperty::set_deletable);
	ClassDB::bind_method(D_METHOD("is_deletable"), &EditorProperty::is_deletable);

	ClassDB::bind_method(D_METHOD("get_edited_property"), &EditorProperty::get_edited_property);
	ClassDB::bind_method(D_METHOD("get_edited_object"), &EditorProperty::get_edited_object);

	ClassDB::bind_method(D_METHOD("get_tooltip_text"), &EditorProperty::get_tooltip_text);
	ClassDB::bind_method(D_METHOD("update_property"), &EditorProperty::update_property);

	ClassDB::bind_method(D_METHOD("add_focusable", "control"), &EditorProperty::add_focusable);
	ClassDB::bind_method(D_METHOD("set_bottom_editor", "editor"), &EditorProperty::set_bottom_editor);

	ClassDB::bind_method(D_METHOD("emit_changed", "property", "value", "field", "changing"), &EditorProperty::emit_changed, DEFVAL(StringName()), DEFVAL(false));

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "label"), "set_label", "get_label");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "read_only"), "set_read_only", "is_read_only");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "checkable"), "set_checkable", "is_checkable");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "checked"), "set_checked", "is_checked");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "draw_warning"), "set_draw_warning", "is_draw_warning");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "keying"), "set_keying", "is_keying");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "deletable"), "set_deletable", "is_deletable");
	ADD_SIGNAL(MethodInfo("property_changed", PropertyInfo(Variant::STRING_NAME, "property"), PropertyInfo(Variant::NIL, "value", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("multiple_properties_changed", PropertyInfo(Variant::PACKED_STRING_ARRAY, "properties"), PropertyInfo(Variant::ARRAY, "value")));
	ADD_SIGNAL(MethodInfo("property_keyed", PropertyInfo(Variant::STRING_NAME, "property")));
	ADD_SIGNAL(MethodInfo("property_deleted", PropertyInfo(Variant::STRING_NAME, "property")));
	ADD_SIGNAL(MethodInfo("property_keyed_with_value", PropertyInfo(Variant::STRING_NAME, "property"), PropertyInfo(Variant::NIL, "value", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
	ADD_SIGNAL(MethodInfo("property_checked", PropertyInfo(Variant::STRING_NAME, "property"), PropertyInfo(Variant::BOOL, "checked")));
	ADD_SIGNAL(MethodInfo("property_pinned", PropertyInfo(Variant::STRING_NAME, "property"), PropertyInfo(Variant::BOOL, "pinned")));
	ADD_SIGNAL(MethodInfo("resource_selected", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::OBJECT, "resource", PROPERTY_HINT_RESOURCE_TYPE, "Resource")));
	ADD_SIGNAL(MethodInfo("object_id_selected", PropertyInfo(Variant::STRING_NAME, "property"), PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("selected", PropertyInfo(Variant::STRING, "path"), PropertyInfo(Variant::INT, "focusable_idx")));

	GDVIRTUAL_BIND(_update_property)
	ClassDB::bind_method(D_METHOD("_update_revert_and_pin_status"), &EditorProperty::update_revert_and_pin_status);
}

EditorProperty::EditorProperty() {
	draw_top_bg = true;
	object = nullptr;
	split_ratio = 0.5;
	selectable = true;
	text_size = 0;
	read_only = false;
	checkable = false;
	checked = false;
	draw_warning = false;
	keying = false;
	deletable = false;
	keying_hover = false;
	revert_hover = false;
	check_hover = false;
	can_revert = false;
	can_pin = false;
	pin_hidden = false;
	pinned = false;
	use_folding = false;
	property_usage = 0;
	selected = false;
	selected_focusable = -1;
	label_reference = nullptr;
	bottom_editor = nullptr;
	delete_hover = false;
	menu = nullptr;
	set_process_unhandled_key_input(true);
}

void EditorProperty::_update_popup() {
	if (menu) {
		menu->clear();
	} else {
		menu = memnew(PopupMenu);
		add_child(menu);
		menu->connect("id_pressed", callable_mp(this, &EditorProperty::menu_option));
	}
	menu->add_shortcut(ED_GET_SHORTCUT("property_editor/copy_property"), MENU_COPY_PROPERTY);
	menu->add_shortcut(ED_GET_SHORTCUT("property_editor/paste_property"), MENU_PASTE_PROPERTY);
	menu->add_shortcut(ED_GET_SHORTCUT("property_editor/copy_property_path"), MENU_COPY_PROPERTY_PATH);
	menu->set_item_disabled(MENU_PASTE_PROPERTY, is_read_only());
	if (!pin_hidden) {
		menu->add_separator();
		if (can_pin) {
			menu->add_check_item(TTR("Pin value"), MENU_PIN_VALUE);
			menu->set_item_checked(menu->get_item_index(MENU_PIN_VALUE), pinned);
			menu->set_item_tooltip(menu->get_item_index(MENU_PIN_VALUE), TTR("Pinning a value forces it to be saved even if it's equal to the default."));
		} else {
			menu->add_check_item(vformat(TTR("Pin value [Disabled because '%s' is editor-only]"), property), MENU_PIN_VALUE);
			menu->set_item_disabled(menu->get_item_index(MENU_PIN_VALUE), true);
		}
	}
}

////////////////////////////////////////////////
////////////////////////////////////////////////

void EditorInspectorPlugin::add_custom_control(Control *control) {
	AddedEditor ae;
	ae.property_editor = control;
	added_editors.push_back(ae);
}

void EditorInspectorPlugin::add_property_editor(const String &p_for_property, Control *p_prop) {
	ERR_FAIL_COND(Object::cast_to<EditorProperty>(p_prop) == nullptr);

	AddedEditor ae;
	ae.properties.push_back(p_for_property);
	ae.property_editor = p_prop;
	added_editors.push_back(ae);
}

void EditorInspectorPlugin::add_property_editor_for_multiple_properties(const String &p_label, const Vector<String> &p_properties, Control *p_prop) {
	AddedEditor ae;
	ae.properties = p_properties;
	ae.property_editor = p_prop;
	ae.label = p_label;
	added_editors.push_back(ae);
}

bool EditorInspectorPlugin::can_handle(Object *p_object) {
	bool success;
	if (GDVIRTUAL_CALL(_can_handle, p_object, success)) {
		return success;
	}
	return false;
}

void EditorInspectorPlugin::parse_begin(Object *p_object) {
	GDVIRTUAL_CALL(_parse_begin, p_object);
}

void EditorInspectorPlugin::parse_category(Object *p_object, const String &p_category) {
	GDVIRTUAL_CALL(_parse_category, p_object, p_category);
}

void EditorInspectorPlugin::parse_group(Object *p_object, const String &p_group) {
	GDVIRTUAL_CALL(_parse_group, p_object, p_group);
}

bool EditorInspectorPlugin::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide) {
	bool ret;
	if (GDVIRTUAL_CALL(_parse_property, p_object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide, ret)) {
		return ret;
	}
	return false;
}

void EditorInspectorPlugin::parse_end(Object *p_object) {
	GDVIRTUAL_CALL(_parse_end, p_object);
}

void EditorInspectorPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_custom_control", "control"), &EditorInspectorPlugin::add_custom_control);
	ClassDB::bind_method(D_METHOD("add_property_editor", "property", "editor"), &EditorInspectorPlugin::add_property_editor);
	ClassDB::bind_method(D_METHOD("add_property_editor_for_multiple_properties", "label", "properties", "editor"), &EditorInspectorPlugin::add_property_editor_for_multiple_properties);

	GDVIRTUAL_BIND(_can_handle, "object")
	GDVIRTUAL_BIND(_parse_begin, "object")
	GDVIRTUAL_BIND(_parse_category, "object", "category")
	GDVIRTUAL_BIND(_parse_group, "object", "group")
	GDVIRTUAL_BIND(_parse_property, "object", "type", "name", "hint_type", "hint_string", "usage_flags", "wide");
	GDVIRTUAL_BIND(_parse_end, "object")
}

////////////////////////////////////////////////
////////////////////////////////////////////////

void EditorInspectorCategory::_notification(int p_what) {
	if (p_what == NOTIFICATION_DRAW) {
		Ref<StyleBox> sb = get_theme_stylebox(SNAME("prop_category_style"), SNAME("Editor"));

		draw_style_box(sb, Rect2(Vector2(), get_size()));

		Ref<Font> font = get_theme_font(SNAME("bold"), SNAME("EditorFonts"));
		int font_size = get_theme_font_size(SNAME("bold_size"), SNAME("EditorFonts"));

		int hs = get_theme_constant(SNAME("hseparation"), SNAME("Tree"));

		int w = font->get_string_size(label, font_size).width;
		if (icon.is_valid()) {
			w += hs + icon->get_width();
		}

		int ofs = (get_size().width - w) / 2;

		if (icon.is_valid()) {
			draw_texture(icon, Point2(ofs, (get_size().height - icon->get_height()) / 2).floor());
			ofs += hs + icon->get_width();
		}

		Color color = get_theme_color(SNAME("font_color"), SNAME("Tree"));
		draw_string(font, Point2(ofs, font->get_ascent(font_size) + (get_size().height - font->get_height(font_size)) / 2).floor(), label, HORIZONTAL_ALIGNMENT_LEFT, get_size().width, font_size, color);
	}
}

Control *EditorInspectorCategory::make_custom_tooltip(const String &p_text) const {
	tooltip_text = p_text;
	EditorHelpBit *help_bit = memnew(EditorHelpBit);
	help_bit->add_theme_style_override("panel", get_theme_stylebox(SNAME("panel"), SNAME("TooltipPanel")));
	help_bit->get_rich_text()->set_fixed_size_to_width(360 * EDSCALE);

	PackedStringArray slices = p_text.split("::", false);
	if (!slices.is_empty()) {
		String property_name = slices[0].strip_edges();
		String text = "[u][b]" + property_name + "[/b][/u]";

		if (slices.size() > 1) {
			String property_doc = slices[1].strip_edges();
			if (property_name != property_doc) {
				text += "\n" + property_doc;
			}
		}
		help_bit->call_deferred(SNAME("set_text"), text); //hack so it uses proper theme once inside scene
	}

	return help_bit;
}

Size2 EditorInspectorCategory::get_minimum_size() const {
	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Tree"));
	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Tree"));

	Size2 ms;
	ms.width = 1;
	ms.height = font->get_height(font_size);
	if (icon.is_valid()) {
		ms.height = MAX(icon->get_height(), ms.height);
	}
	ms.height += get_theme_constant(SNAME("vseparation"), SNAME("Tree"));

	return ms;
}

void EditorInspectorCategory::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_tooltip_text"), &EditorInspectorCategory::get_tooltip_text);
}

String EditorInspectorCategory::get_tooltip_text() const {
	return tooltip_text;
}

EditorInspectorCategory::EditorInspectorCategory() {
}

////////////////////////////////////////////////
////////////////////////////////////////////////

void EditorInspectorSection::_test_unfold() {
	if (!vbox_added) {
		add_child(vbox);
		move_child(vbox, 0);
		vbox_added = true;
	}
}

void EditorInspectorSection::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			update_minimum_size();
		} break;
		case NOTIFICATION_SORT_CHILDREN: {
			if (!vbox_added) {
				return;
			}
			// Get the section header font.
			Ref<Font> font = get_theme_font(SNAME("bold"), SNAME("EditorFonts"));
			int font_size = get_theme_font_size(SNAME("bold_size"), SNAME("EditorFonts"));

			// Get the right direction arrow texture, if the section is foldable.
			Ref<Texture2D> arrow;
			if (foldable) {
				if (object->editor_is_section_unfolded(section)) {
					arrow = get_theme_icon(SNAME("arrow"), SNAME("Tree"));
				} else {
					if (is_layout_rtl()) {
						arrow = get_theme_icon(SNAME("arrow_collapsed_mirrored"), SNAME("Tree"));
					} else {
						arrow = get_theme_icon(SNAME("arrow_collapsed"), SNAME("Tree"));
					}
				}
			}

			// Compute the height of the section header.
			int header_height = font->get_height(font_size);
			if (arrow.is_valid()) {
				header_height = MAX(header_height, arrow->get_height());
			}
			header_height += get_theme_constant(SNAME("vseparation"), SNAME("Tree"));

			int inspector_margin = get_theme_constant(SNAME("inspector_margin"), SNAME("Editor"));
			Size2 size = get_size() - Vector2(inspector_margin, 0);
			Vector2 offset = Vector2(is_layout_rtl() ? 0 : inspector_margin, header_height);
			for (int i = 0; i < get_child_count(); i++) {
				Control *c = Object::cast_to<Control>(get_child(i));
				if (!c) {
					continue;
				}
				if (c->is_set_as_top_level()) {
					continue;
				}

				fit_child_in_rect(c, Rect2(offset, size));
			}
		} break;
		case NOTIFICATION_DRAW: {
			// Get the section header font.
			Ref<Font> font = get_theme_font(SNAME("bold"), SNAME("EditorFonts"));
			int font_size = get_theme_font_size(SNAME("bold_size"), SNAME("EditorFonts"));

			// Get the right direction arrow texture, if the section is foldable.
			Ref<Texture2D> arrow;
			if (foldable) {
				if (object->editor_is_section_unfolded(section)) {
					arrow = get_theme_icon(SNAME("arrow"), SNAME("Tree"));
				} else {
					if (is_layout_rtl()) {
						arrow = get_theme_icon(SNAME("arrow_collapsed_mirrored"), SNAME("Tree"));
					} else {
						arrow = get_theme_icon(SNAME("arrow_collapsed"), SNAME("Tree"));
					}
				}
			}

			bool rtl = is_layout_rtl();

			// Compute the height of the section header.
			int header_height = font->get_height(font_size);
			if (arrow.is_valid()) {
				header_height = MAX(header_height, arrow->get_height());
			}
			header_height += get_theme_constant(SNAME("vseparation"), SNAME("Tree"));

			Rect2 header_rect = Rect2(Vector2(), Vector2(get_size().width, header_height));
			Color c = bg_color;
			c.a *= 0.4;
			if (foldable && header_rect.has_point(get_local_mouse_position())) {
				c = c.lightened(Input::get_singleton()->is_mouse_button_pressed(MouseButton::LEFT) ? -0.05 : 0.2);
			}
			draw_rect(header_rect, c);

			const int arrow_margin = 2;
			const int arrow_width = arrow.is_valid() ? arrow->get_width() : 0;
			Color color = get_theme_color(SNAME("font_color"));
			float text_width = get_size().width - Math::round(arrow_width + arrow_margin * EDSCALE);
			draw_string(font, Point2(rtl ? 0 : Math::round(arrow_width + arrow_margin * EDSCALE), font->get_ascent(font_size) + (header_height - font->get_height(font_size)) / 2).floor(), label, rtl ? HORIZONTAL_ALIGNMENT_RIGHT : HORIZONTAL_ALIGNMENT_LEFT, text_width, font_size, color);

			if (arrow.is_valid()) {
				if (rtl) {
					draw_texture(arrow, Point2(get_size().width - arrow->get_width() - Math::round(arrow_margin * EDSCALE), (header_height - arrow->get_height()) / 2).floor());
				} else {
					draw_texture(arrow, Point2(Math::round(arrow_margin * EDSCALE), (header_height - arrow->get_height()) / 2).floor());
				}
			}

			if (dropping && !vbox->is_visible_in_tree()) {
				Color accent_color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
				draw_rect(Rect2(Point2(), get_size()), accent_color, false);
			}
		} break;
		case NOTIFICATION_DRAG_BEGIN: {
			Dictionary dd = get_viewport()->gui_get_drag_data();

			// Only allow dropping if the section contains properties which can take the dragged data.
			bool children_can_drop = false;
			for (int child_idx = 0; child_idx < vbox->get_child_count(); child_idx++) {
				Control *editor_property = Object::cast_to<Control>(vbox->get_child(child_idx));

				// Test can_drop_data and can_drop_data_fw, since can_drop_data only works if set up with forwarding or if script attached.
				if (editor_property && (editor_property->can_drop_data(Point2(), dd) || editor_property->call("_can_drop_data_fw", Point2(), dd, this))) {
					children_can_drop = true;
					break;
				}
			}

			dropping = children_can_drop;
			update();
		} break;
		case NOTIFICATION_DRAG_END: {
			dropping = false;
			update();
		} break;
		case NOTIFICATION_MOUSE_ENTER: {
			if (dropping) {
				dropping_unfold_timer->start();
			}
			update();
		} break;

		case NOTIFICATION_MOUSE_EXIT: {
			if (dropping) {
				dropping_unfold_timer->stop();
			}
			update();
		} break;
	}
}

Size2 EditorInspectorSection::get_minimum_size() const {
	Size2 ms;
	for (int i = 0; i < get_child_count(); i++) {
		Control *c = Object::cast_to<Control>(get_child(i));
		if (!c) {
			continue;
		}
		if (c->is_set_as_top_level()) {
			continue;
		}
		if (!c->is_visible()) {
			continue;
		}
		Size2 minsize = c->get_combined_minimum_size();
		ms.width = MAX(ms.width, minsize.width);
		ms.height = MAX(ms.height, minsize.height);
	}

	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Tree"));
	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Tree"));
	ms.height += font->get_height(font_size) + get_theme_constant(SNAME("vseparation"), SNAME("Tree"));
	ms.width += get_theme_constant(SNAME("inspector_margin"), SNAME("Editor"));

	return ms;
}

void EditorInspectorSection::setup(const String &p_section, const String &p_label, Object *p_object, const Color &p_bg_color, bool p_foldable) {
	section = p_section;
	label = p_label;
	object = p_object;
	bg_color = p_bg_color;
	foldable = p_foldable;

	if (!foldable && !vbox_added) {
		add_child(vbox);
		move_child(vbox, 0);
		vbox_added = true;
	}

	if (foldable) {
		_test_unfold();
		if (object->editor_is_section_unfolded(section)) {
			vbox->show();
		} else {
			vbox->hide();
		}
	}
}

void EditorInspectorSection::gui_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	if (!foldable) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Tree"));
		int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Tree"));
		if (mb->get_position().y > font->get_height(font_size)) { //clicked outside
			return;
		}

		bool should_unfold = !object->editor_is_section_unfolded(section);
		if (should_unfold) {
			unfold();
		} else {
			fold();
		}
	} else if (mb.is_valid() && !mb->is_pressed()) {
		update();
	}
}

VBoxContainer *EditorInspectorSection::get_vbox() {
	return vbox;
}

void EditorInspectorSection::unfold() {
	if (!foldable) {
		return;
	}

	_test_unfold();

	object->editor_set_section_unfold(section, true);
	vbox->show();
	update();
}

void EditorInspectorSection::fold() {
	if (!foldable) {
		return;
	}

	if (!vbox_added) {
		return;
	}

	object->editor_set_section_unfold(section, false);
	vbox->hide();
	update();
}

void EditorInspectorSection::_bind_methods() {
	ClassDB::bind_method(D_METHOD("setup", "section", "label", "object", "bg_color", "foldable"), &EditorInspectorSection::setup);
	ClassDB::bind_method(D_METHOD("get_vbox"), &EditorInspectorSection::get_vbox);
	ClassDB::bind_method(D_METHOD("unfold"), &EditorInspectorSection::unfold);
	ClassDB::bind_method(D_METHOD("fold"), &EditorInspectorSection::fold);
}

EditorInspectorSection::EditorInspectorSection() {
	object = nullptr;
	foldable = false;
	vbox = memnew(VBoxContainer);
	vbox_added = false;

	dropping = false;
	dropping_unfold_timer = memnew(Timer);
	dropping_unfold_timer->set_wait_time(0.6);
	dropping_unfold_timer->set_one_shot(true);
	add_child(dropping_unfold_timer);
	dropping_unfold_timer->connect("timeout", callable_mp(this, &EditorInspectorSection::unfold));
}

EditorInspectorSection::~EditorInspectorSection() {
	if (!vbox_added) {
		memdelete(vbox);
	}
}

////////////////////////////////////////////////
////////////////////////////////////////////////
int EditorInspectorArray::_get_array_count() {
	if (mode == MODE_USE_MOVE_ARRAY_ELEMENT_FUNCTION) {
		List<PropertyInfo> object_property_list;
		object->get_property_list(&object_property_list);
		return _extract_properties_as_array(object_property_list).size();
	} else if (mode == MODE_USE_COUNT_PROPERTY) {
		bool valid;
		int count = object->get(count_property, &valid);
		ERR_FAIL_COND_V_MSG(!valid, 0, vformat("%s is not a valid property to be used as array count.", count_property));
		return count;
	}
	return 0;
}

void EditorInspectorArray::_add_button_pressed() {
	_move_element(-1, -1);
}

void EditorInspectorArray::_first_page_button_pressed() {
	emit_signal("page_change_request", 0);
}

void EditorInspectorArray::_prev_page_button_pressed() {
	emit_signal("page_change_request", MAX(0, page - 1));
}

void EditorInspectorArray::_page_line_edit_text_submitted(String p_text) {
	if (p_text.is_valid_int()) {
		int new_page = p_text.to_int() - 1;
		new_page = MIN(MAX(0, new_page), max_page);
		page_line_edit->set_text(Variant(new_page));
		emit_signal("page_change_request", new_page);
	} else {
		page_line_edit->set_text(Variant(page));
	}
}

void EditorInspectorArray::_next_page_button_pressed() {
	emit_signal("page_change_request", MIN(max_page, page + 1));
}

void EditorInspectorArray::_last_page_button_pressed() {
	emit_signal("page_change_request", max_page);
}

void EditorInspectorArray::_rmb_popup_id_pressed(int p_id) {
	switch (p_id) {
		case OPTION_MOVE_UP:
			if (popup_array_index_pressed > 0) {
				_move_element(popup_array_index_pressed, popup_array_index_pressed - 1);
			}
			break;
		case OPTION_MOVE_DOWN:
			if (popup_array_index_pressed < count - 1) {
				_move_element(popup_array_index_pressed, popup_array_index_pressed + 2);
			}
			break;
		case OPTION_NEW_BEFORE:
			_move_element(-1, popup_array_index_pressed);
			break;
		case OPTION_NEW_AFTER:
			_move_element(-1, popup_array_index_pressed + 1);
			break;
		case OPTION_REMOVE:
			_move_element(popup_array_index_pressed, -1);
			break;
		case OPTION_CLEAR_ARRAY:
			_clear_array();
			break;
		case OPTION_RESIZE_ARRAY:
			new_size = count;
			new_size_line_edit->set_text(Variant(new_size));
			resize_dialog->get_ok_button()->set_disabled(true);
			resize_dialog->popup_centered();
			new_size_line_edit->grab_focus();
			new_size_line_edit->select_all();
			break;
		default:
			break;
	}
}

void EditorInspectorArray::_control_dropping_draw() {
	int drop_position = _drop_position();

	if (dropping && drop_position >= 0) {
		Vector2 from;
		Vector2 to;
		if (drop_position < elements_vbox->get_child_count()) {
			Transform2D xform = Object::cast_to<Control>(elements_vbox->get_child(drop_position))->get_transform();
			from = xform.xform(Vector2());
			to = xform.xform(Vector2(elements_vbox->get_size().x, 0));
		} else {
			Control *child = Object::cast_to<Control>(elements_vbox->get_child(drop_position - 1));
			Transform2D xform = child->get_transform();
			from = xform.xform(Vector2(0, child->get_size().y));
			to = xform.xform(Vector2(elements_vbox->get_size().x, child->get_size().y));
		}
		Color color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
		control_dropping->draw_line(from, to, color, 2);
	}
}

void EditorInspectorArray::_vbox_visibility_changed() {
	control_dropping->set_visible(vbox->is_visible_in_tree());
}

void EditorInspectorArray::_panel_draw(int p_index) {
	ERR_FAIL_INDEX(p_index, (int)array_elements.size());

	Ref<StyleBox> style = get_theme_stylebox("Focus", "EditorStyles");
	if (!style.is_valid()) {
		return;
	}
	if (array_elements[p_index].panel->has_focus()) {
		array_elements[p_index].panel->draw_style_box(style, Rect2(Vector2(), array_elements[p_index].panel->get_size()));
	}
}

void EditorInspectorArray::_panel_gui_input(Ref<InputEvent> p_event, int p_index) {
	ERR_FAIL_INDEX(p_index, (int)array_elements.size());

	Ref<InputEventKey> key_ref = p_event;
	if (key_ref.is_valid()) {
		const InputEventKey &key = **key_ref;

		if (array_elements[p_index].panel->has_focus() && key.is_pressed() && key.get_keycode() == Key::KEY_DELETE) {
			_move_element(begin_array_index + p_index, -1);
			array_elements[p_index].panel->accept_event();
		}
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::RIGHT) {
			popup_array_index_pressed = begin_array_index + p_index;
			rmb_popup->set_item_disabled(OPTION_MOVE_UP, popup_array_index_pressed == 0);
			rmb_popup->set_item_disabled(OPTION_MOVE_DOWN, popup_array_index_pressed == count - 1);
			rmb_popup->set_position(get_screen_position() + mb->get_position());
			rmb_popup->reset_size();
			rmb_popup->popup();
		}
	}
}

void EditorInspectorArray::_move_element(int p_element_index, int p_to_pos) {
	String action_name;
	if (p_element_index < 0) {
		action_name = vformat("Add element to property array with prefix %s.", array_element_prefix);
	} else if (p_to_pos < 0) {
		action_name = vformat("Remove element %d from property array with prefix %s.", p_element_index, array_element_prefix);
	} else {
		action_name = vformat("Move element %d to position %d in property array with prefix %s.", p_element_index, p_to_pos, array_element_prefix);
	}
	undo_redo->create_action(action_name);
	if (mode == MODE_USE_MOVE_ARRAY_ELEMENT_FUNCTION) {
		// Call the function.
		Callable move_function = EditorNode::get_singleton()->get_editor_data().get_move_array_element_function(object->get_class_name());
		if (move_function.is_valid()) {
			Variant args[] = { (Object *)undo_redo, object, array_element_prefix, p_element_index, p_to_pos };
			const Variant *args_p[] = { &args[0], &args[1], &args[2], &args[3], &args[4] };
			Variant return_value;
			Callable::CallError call_error;
			move_function.call(args_p, 5, return_value, call_error);
		} else {
			WARN_PRINT(vformat("Could not find a function to move arrays elements for class %s. Register a move element function using EditorData::add_move_array_element_function", object->get_class_name()));
		}
	} else if (mode == MODE_USE_COUNT_PROPERTY) {
		ERR_FAIL_COND(p_to_pos < -1 || p_to_pos > count);
		List<PropertyInfo> object_property_list;
		object->get_property_list(&object_property_list);

		Array properties_as_array = _extract_properties_as_array(object_property_list);
		properties_as_array.resize(count);

		// For undoing things
		undo_redo->add_undo_property(object, count_property, properties_as_array.size());
		for (int i = 0; i < (int)properties_as_array.size(); i++) {
			Dictionary d = Dictionary(properties_as_array[i]);
			Array keys = d.keys();
			for (int j = 0; j < keys.size(); j++) {
				String key = keys[j];
				undo_redo->add_undo_property(object, vformat(key, i), d[key]);
			}
		}

		if (p_element_index < 0) {
			// Add an element.
			properties_as_array.insert(p_to_pos < 0 ? properties_as_array.size() : p_to_pos, Dictionary());
		} else if (p_to_pos < 0) {
			// Delete the element.
			properties_as_array.remove_at(p_element_index);
		} else {
			// Move the element.
			properties_as_array.insert(p_to_pos, properties_as_array[p_element_index].duplicate());
			properties_as_array.remove_at(p_to_pos < p_element_index ? p_element_index + 1 : p_element_index);
		}

		// Change the array size then set the properties.
		undo_redo->add_do_property(object, count_property, properties_as_array.size());
		for (int i = 0; i < (int)properties_as_array.size(); i++) {
			Dictionary d = properties_as_array[i];
			Array keys = d.keys();
			for (int j = 0; j < keys.size(); j++) {
				String key = keys[j];
				undo_redo->add_do_property(object, vformat(key, i), d[key]);
			}
		}
	}
	undo_redo->commit_action();

	// Handle page change and update counts.
	if (p_element_index < 0) {
		int added_index = p_to_pos < 0 ? count : p_to_pos;
		emit_signal("page_change_request", added_index / page_lenght);
		count += 1;
	} else if (p_to_pos < 0) {
		count -= 1;
		if (page == max_page && (MAX(0, count - 1) / page_lenght != max_page)) {
			emit_signal("page_change_request", max_page - 1);
		}
	}
	begin_array_index = page * page_lenght;
	end_array_index = MIN(count, (page + 1) * page_lenght);
	max_page = MAX(0, count - 1) / page_lenght;
}

void EditorInspectorArray::_clear_array() {
	undo_redo->create_action(vformat("Clear property array with prefix %s.", array_element_prefix));
	if (mode == MODE_USE_MOVE_ARRAY_ELEMENT_FUNCTION) {
		for (int i = count - 1; i >= 0; i--) {
			// Call the function.
			Callable move_function = EditorNode::get_singleton()->get_editor_data().get_move_array_element_function(object->get_class_name());
			if (move_function.is_valid()) {
				Variant args[] = { (Object *)undo_redo, object, array_element_prefix, i, -1 };
				const Variant *args_p[] = { &args[0], &args[1], &args[2], &args[3], &args[4] };
				Variant return_value;
				Callable::CallError call_error;
				move_function.call(args_p, 5, return_value, call_error);
			} else {
				WARN_PRINT(vformat("Could not find a function to move arrays elements for class %s. Register a move element function using EditorData::add_move_array_element_function", object->get_class_name()));
			}
		}
	} else if (mode == MODE_USE_COUNT_PROPERTY) {
		List<PropertyInfo> object_property_list;
		object->get_property_list(&object_property_list);

		Array properties_as_array = _extract_properties_as_array(object_property_list);
		properties_as_array.resize(count);

		// For undoing things
		undo_redo->add_undo_property(object, count_property, count);
		for (int i = 0; i < (int)properties_as_array.size(); i++) {
			Dictionary d = Dictionary(properties_as_array[i]);
			Array keys = d.keys();
			for (int j = 0; j < keys.size(); j++) {
				String key = keys[j];
				undo_redo->add_undo_property(object, vformat(key, i), d[key]);
			}
		}

		// Change the array size then set the properties.
		undo_redo->add_do_property(object, count_property, 0);
	}
	undo_redo->commit_action();

	// Handle page change and update counts.
	emit_signal("page_change_request", 0);
	count = 0;
	begin_array_index = 0;
	end_array_index = 0;
	max_page = 0;
}

void EditorInspectorArray::_resize_array(int p_size) {
	ERR_FAIL_COND(p_size < 0);
	if (p_size == count) {
		return;
	}

	undo_redo->create_action(vformat("Resize property array with prefix %s.", array_element_prefix));
	if (p_size > count) {
		if (mode == MODE_USE_MOVE_ARRAY_ELEMENT_FUNCTION) {
			for (int i = count; i < p_size; i++) {
				// Call the function.
				Callable move_function = EditorNode::get_singleton()->get_editor_data().get_move_array_element_function(object->get_class_name());
				if (move_function.is_valid()) {
					Variant args[] = { (Object *)undo_redo, object, array_element_prefix, -1, -1 };
					const Variant *args_p[] = { &args[0], &args[1], &args[2], &args[3], &args[4] };
					Variant return_value;
					Callable::CallError call_error;
					move_function.call(args_p, 5, return_value, call_error);
				} else {
					WARN_PRINT(vformat("Could not find a function to move arrays elements for class %s. Register a move element function using EditorData::add_move_array_element_function", object->get_class_name()));
				}
			}
		} else if (mode == MODE_USE_COUNT_PROPERTY) {
			undo_redo->add_undo_property(object, count_property, count);
			undo_redo->add_do_property(object, count_property, p_size);
		}
	} else {
		if (mode == MODE_USE_MOVE_ARRAY_ELEMENT_FUNCTION) {
			for (int i = count - 1; i > p_size - 1; i--) {
				// Call the function.
				Callable move_function = EditorNode::get_singleton()->get_editor_data().get_move_array_element_function(object->get_class_name());
				if (move_function.is_valid()) {
					Variant args[] = { (Object *)undo_redo, object, array_element_prefix, i, -1 };
					const Variant *args_p[] = { &args[0], &args[1], &args[2], &args[3], &args[4] };
					Variant return_value;
					Callable::CallError call_error;
					move_function.call(args_p, 5, return_value, call_error);
				} else {
					WARN_PRINT(vformat("Could not find a function to move arrays elements for class %s. Register a move element function using EditorData::add_move_array_element_function", object->get_class_name()));
				}
			}
		} else if (mode == MODE_USE_COUNT_PROPERTY) {
			List<PropertyInfo> object_property_list;
			object->get_property_list(&object_property_list);

			Array properties_as_array = _extract_properties_as_array(object_property_list);
			properties_as_array.resize(count);

			// For undoing things
			undo_redo->add_undo_property(object, count_property, count);
			for (int i = count - 1; i > p_size - 1; i--) {
				Dictionary d = Dictionary(properties_as_array[i]);
				Array keys = d.keys();
				for (int j = 0; j < keys.size(); j++) {
					String key = keys[j];
					undo_redo->add_undo_property(object, vformat(key, i), d[key]);
				}
			}

			// Change the array size then set the properties.
			undo_redo->add_do_property(object, count_property, p_size);
		}
	}
	undo_redo->commit_action();

	// Handle page change and update counts.
	emit_signal("page_change_request", 0);
	/*
	count = 0;
	begin_array_index = 0;
	end_array_index = 0;
	max_page = 0;
	*/
}

Array EditorInspectorArray::_extract_properties_as_array(const List<PropertyInfo> &p_list) {
	Array output;

	for (const PropertyInfo &pi : p_list) {
		if (pi.name.begins_with(array_element_prefix)) {
			String str = pi.name.trim_prefix(array_element_prefix);

			int to_char_index = 0;
			while (to_char_index < str.length()) {
				if (str[to_char_index] < '0' || str[to_char_index] > '9') {
					break;
				}
				to_char_index++;
			}
			if (to_char_index > 0) {
				int array_index = str.left(to_char_index).to_int();
				Error error = OK;
				if (array_index >= output.size()) {
					error = output.resize(array_index + 1);
				}
				if (error == OK) {
					String format_string = String(array_element_prefix) + "%d" + str.substr(to_char_index);
					Dictionary dict = output[array_index];
					dict[format_string] = object->get(pi.name);
					output[array_index] = dict;
				} else {
					WARN_PRINT(vformat("Array element %s has an index too high. Array allocation failed.", pi.name));
				}
			}
		}
	}
	return output;
}

int EditorInspectorArray::_drop_position() const {
	for (int i = 0; i < (int)array_elements.size(); i++) {
		const ArrayElement &ae = array_elements[i];

		Size2 size = ae.panel->get_size();
		Vector2 mp = ae.panel->get_local_mouse_position();

		if (Rect2(Vector2(), size).has_point(mp)) {
			if (mp.y < size.y / 2) {
				return i;
			} else {
				return i + 1;
			}
		}
	}
	return -1;
}

void EditorInspectorArray::_new_size_line_edit_text_changed(String p_text) {
	bool valid = false;
	;
	if (p_text.is_valid_int()) {
		int val = p_text.to_int();
		if (val > 0 && val != count) {
			valid = true;
		}
	}
	resize_dialog->get_ok_button()->set_disabled(!valid);
}

void EditorInspectorArray::_new_size_line_edit_text_submitted(String p_text) {
	bool valid = false;
	;
	if (p_text.is_valid_int()) {
		int val = p_text.to_int();
		if (val > 0 && val != count) {
			new_size = val;
			valid = true;
		}
	}
	if (valid) {
		resize_dialog->hide();
		_resize_array(new_size);
	} else {
		new_size_line_edit->set_text(Variant(new_size));
	}
}

void EditorInspectorArray::_resize_dialog_confirmed() {
	_new_size_line_edit_text_submitted(new_size_line_edit->get_text());
}

void EditorInspectorArray::_setup() {
	// Setup counts.
	count = _get_array_count();
	begin_array_index = page * page_lenght;
	end_array_index = MIN(count, (page + 1) * page_lenght);
	max_page = MAX(0, count - 1) / page_lenght;
	array_elements.resize(MAX(0, end_array_index - begin_array_index));
	if (page < 0 || page > max_page) {
		WARN_PRINT(vformat("Invalid page number %d", page));
		page = CLAMP(page, 0, max_page);
	}

	for (int i = 0; i < (int)array_elements.size(); i++) {
		ArrayElement &ae = array_elements[i];

		// Panel and its hbox.
		ae.panel = memnew(PanelContainer);
		ae.panel->set_focus_mode(FOCUS_ALL);
		ae.panel->set_mouse_filter(MOUSE_FILTER_PASS);
		ae.panel->set_drag_forwarding(this);
		ae.panel->set_meta("index", begin_array_index + i);
		ae.panel->set_tooltip(vformat(TTR("Element %d: %s%d*"), i, array_element_prefix, i));
		ae.panel->connect("focus_entered", callable_mp((CanvasItem *)ae.panel, &PanelContainer::update));
		ae.panel->connect("focus_exited", callable_mp((CanvasItem *)ae.panel, &PanelContainer::update));
		ae.panel->connect("draw", callable_bind(callable_mp(this, &EditorInspectorArray::_panel_draw), i));
		ae.panel->connect("gui_input", callable_bind(callable_mp(this, &EditorInspectorArray::_panel_gui_input), i));
		ae.panel->add_theme_style_override(SNAME("panel"), i % 2 ? odd_style : even_style);
		elements_vbox->add_child(ae.panel);

		ae.margin = memnew(MarginContainer);
		ae.margin->set_mouse_filter(MOUSE_FILTER_PASS);
		if (is_inside_tree()) {
			Size2 min_size = get_theme_stylebox("Focus", "EditorStyles")->get_minimum_size();
			ae.margin->add_theme_constant_override("margin_left", min_size.x / 2);
			ae.margin->add_theme_constant_override("margin_top", min_size.y / 2);
			ae.margin->add_theme_constant_override("margin_right", min_size.x / 2);
			ae.margin->add_theme_constant_override("margin_bottom", min_size.y / 2);
		}
		ae.panel->add_child(ae.margin);

		ae.hbox = memnew(HBoxContainer);
		ae.hbox->set_h_size_flags(SIZE_EXPAND_FILL);
		ae.hbox->set_v_size_flags(SIZE_EXPAND_FILL);
		ae.margin->add_child(ae.hbox);

		// Move button.
		ae.move_texture_rect = memnew(TextureRect);
		ae.move_texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_CENTERED);
		if (is_inside_tree()) {
			ae.move_texture_rect->set_texture(get_theme_icon(SNAME("TripleBar"), SNAME("EditorIcons")));
		}
		ae.hbox->add_child(ae.move_texture_rect);

		// Right vbox.
		ae.vbox = memnew(VBoxContainer);
		ae.vbox->set_h_size_flags(SIZE_EXPAND_FILL);
		ae.vbox->set_v_size_flags(SIZE_EXPAND_FILL);
		ae.hbox->add_child(ae.vbox);
	}

	// Hide/show the add button.
	add_button->set_visible(page == max_page);

	if (max_page == 0) {
		hbox_pagination->hide();
	} else {
		// Update buttons.
		first_page_button->set_disabled(page == 0);
		prev_page_button->set_disabled(page == 0);
		next_page_button->set_disabled(page == max_page);
		last_page_button->set_disabled(page == max_page);

		// Update page number and page count.
		page_line_edit->set_text(vformat("%d", page + 1));
		page_count_label->set_text(vformat("/ %d", max_page + 1));
	}
}

Variant EditorInspectorArray::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	int index = p_from->get_meta("index");
	Dictionary dict;
	dict["type"] = "property_array_element";
	dict["property_array_prefix"] = array_element_prefix;
	dict["index"] = index;

	return dict;
}

void EditorInspectorArray::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	Dictionary dict = p_data;

	int to_drop = dict["index"];
	int drop_position = _drop_position();
	if (drop_position < 0) {
		return;
	}
	_move_element(to_drop, begin_array_index + drop_position);
}

bool EditorInspectorArray::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	// First, update drawing.
	control_dropping->update();

	if (p_data.get_type() != Variant::DICTIONARY) {
		return false;
	}
	Dictionary dict = p_data;
	int drop_position = _drop_position();
	if (!dict.has("type") || dict["type"] != "property_array_element" || String(dict["property_array_prefix"]) != array_element_prefix || drop_position < 0) {
		return false;
	}

	// Check in dropping at the given index does indeed move the item.
	int moved_array_index = (int)dict["index"];
	int drop_array_index = begin_array_index + drop_position;

	return drop_array_index != moved_array_index && drop_array_index - 1 != moved_array_index;
}

void EditorInspectorArray::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			Color color = get_theme_color(SNAME("dark_color_1"), SNAME("Editor"));
			odd_style->set_bg_color(color.lightened(0.15));
			even_style->set_bg_color(color.darkened(0.15));

			for (int i = 0; i < (int)array_elements.size(); i++) {
				ArrayElement &ae = array_elements[i];
				ae.move_texture_rect->set_texture(get_theme_icon(SNAME("TripleBar"), SNAME("EditorIcons")));

				Size2 min_size = get_theme_stylebox("Focus", "EditorStyles")->get_minimum_size();
				ae.margin->add_theme_constant_override("margin_left", min_size.x / 2);
				ae.margin->add_theme_constant_override("margin_top", min_size.y / 2);
				ae.margin->add_theme_constant_override("margin_right", min_size.x / 2);
				ae.margin->add_theme_constant_override("margin_bottom", min_size.y / 2);
			}

			add_button->set_icon(get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
			first_page_button->set_icon(get_theme_icon(SNAME("PageFirst"), SNAME("EditorIcons")));
			prev_page_button->set_icon(get_theme_icon(SNAME("PagePrevious"), SNAME("EditorIcons")));
			next_page_button->set_icon(get_theme_icon(SNAME("PageNext"), SNAME("EditorIcons")));
			last_page_button->set_icon(get_theme_icon(SNAME("PageLast"), SNAME("EditorIcons")));
			update_minimum_size();
		} break;
		case NOTIFICATION_DRAG_BEGIN: {
			Dictionary dict = get_viewport()->gui_get_drag_data();
			if (dict.has("type") && dict["type"] == "property_array_element" && String(dict["property_array_prefix"]) == array_element_prefix) {
				dropping = true;
				control_dropping->update();
			}
		} break;
		case NOTIFICATION_DRAG_END: {
			if (dropping) {
				dropping = false;
				control_dropping->update();
			}
		} break;
	}
}

void EditorInspectorArray::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_get_drag_data_fw"), &EditorInspectorArray::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("_can_drop_data_fw"), &EditorInspectorArray::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("_drop_data_fw"), &EditorInspectorArray::drop_data_fw);

	ADD_SIGNAL(MethodInfo("page_change_request"));
}

void EditorInspectorArray::set_undo_redo(UndoRedo *p_undo_redo) {
	undo_redo = p_undo_redo;
}

void EditorInspectorArray::setup_with_move_element_function(Object *p_object, String p_label, const StringName &p_array_element_prefix, int p_page, const Color &p_bg_color, bool p_foldable) {
	count_property = "";
	mode = MODE_USE_MOVE_ARRAY_ELEMENT_FUNCTION;
	array_element_prefix = p_array_element_prefix;
	page = p_page;

	EditorInspectorSection::setup(String(p_array_element_prefix) + "_array", p_label, p_object, p_bg_color, p_foldable);

	_setup();
}

void EditorInspectorArray::setup_with_count_property(Object *p_object, String p_label, const StringName &p_count_property, const StringName &p_array_element_prefix, int p_page, const Color &p_bg_color, bool p_foldable) {
	count_property = p_count_property;
	mode = MODE_USE_COUNT_PROPERTY;
	array_element_prefix = p_array_element_prefix;
	page = p_page;

	EditorInspectorSection::setup(String(count_property) + "_array", p_label, p_object, p_bg_color, p_foldable);

	_setup();
}

VBoxContainer *EditorInspectorArray::get_vbox(int p_index) {
	if (p_index >= begin_array_index && p_index < end_array_index) {
		return array_elements[p_index - begin_array_index].vbox;
	} else if (p_index < 0) {
		return vbox;
	} else {
		return nullptr;
	}
}

EditorInspectorArray::EditorInspectorArray() {
	set_mouse_filter(Control::MOUSE_FILTER_STOP);

	odd_style.instantiate();
	even_style.instantiate();

	rmb_popup = memnew(PopupMenu);
	rmb_popup->add_item(TTR("Move Up"), OPTION_MOVE_UP);
	rmb_popup->add_item(TTR("Move Down"), OPTION_MOVE_DOWN);
	rmb_popup->add_separator();
	rmb_popup->add_item(TTR("Insert New Before"), OPTION_NEW_BEFORE);
	rmb_popup->add_item(TTR("Insert New After"), OPTION_NEW_AFTER);
	rmb_popup->add_separator();
	rmb_popup->add_item(TTR("Remove"), OPTION_REMOVE);
	rmb_popup->add_separator();
	rmb_popup->add_item(TTR("Clear Array"), OPTION_CLEAR_ARRAY);
	rmb_popup->add_item(TTR("Resize Array..."), OPTION_RESIZE_ARRAY);
	rmb_popup->connect("id_pressed", callable_mp(this, &EditorInspectorArray::_rmb_popup_id_pressed));
	add_child(rmb_popup);

	elements_vbox = memnew(VBoxContainer);
	elements_vbox->add_theme_constant_override("separation", 0);
	vbox->add_child(elements_vbox);

	add_button = memnew(Button);
	add_button->set_text(TTR("Add Element"));
	add_button->set_text_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	add_button->connect("pressed", callable_mp(this, &EditorInspectorArray::_add_button_pressed));
	vbox->add_child(add_button);

	hbox_pagination = memnew(HBoxContainer);
	hbox_pagination->set_h_size_flags(SIZE_EXPAND_FILL);
	hbox_pagination->set_alignment(BoxContainer::ALIGNMENT_CENTER);
	vbox->add_child(hbox_pagination);

	first_page_button = memnew(Button);
	first_page_button->set_flat(true);
	first_page_button->connect("pressed", callable_mp(this, &EditorInspectorArray::_first_page_button_pressed));
	hbox_pagination->add_child(first_page_button);

	prev_page_button = memnew(Button);
	prev_page_button->set_flat(true);
	prev_page_button->connect("pressed", callable_mp(this, &EditorInspectorArray::_prev_page_button_pressed));
	hbox_pagination->add_child(prev_page_button);

	page_line_edit = memnew(LineEdit);
	page_line_edit->connect("text_submitted", callable_mp(this, &EditorInspectorArray::_page_line_edit_text_submitted));
	page_line_edit->add_theme_constant_override("minimum_character_width", 2);
	hbox_pagination->add_child(page_line_edit);

	page_count_label = memnew(Label);
	hbox_pagination->add_child(page_count_label);
	next_page_button = memnew(Button);
	next_page_button->set_flat(true);
	next_page_button->connect("pressed", callable_mp(this, &EditorInspectorArray::_next_page_button_pressed));
	hbox_pagination->add_child(next_page_button);

	last_page_button = memnew(Button);
	last_page_button->set_flat(true);
	last_page_button->connect("pressed", callable_mp(this, &EditorInspectorArray::_last_page_button_pressed));
	hbox_pagination->add_child(last_page_button);

	control_dropping = memnew(Control);
	control_dropping->connect("draw", callable_mp(this, &EditorInspectorArray::_control_dropping_draw));
	control_dropping->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	add_child(control_dropping);

	resize_dialog = memnew(AcceptDialog);
	resize_dialog->set_title(TTRC("Resize Array"));
	resize_dialog->add_cancel_button();
	resize_dialog->connect("confirmed", callable_mp(this, &EditorInspectorArray::_resize_dialog_confirmed));
	add_child(resize_dialog);

	VBoxContainer *resize_dialog_vbox = memnew(VBoxContainer);
	resize_dialog->add_child(resize_dialog_vbox);

	new_size_line_edit = memnew(LineEdit);
	new_size_line_edit->connect("text_changed", callable_mp(this, &EditorInspectorArray::_new_size_line_edit_text_changed));
	new_size_line_edit->connect("text_submitted", callable_mp(this, &EditorInspectorArray::_new_size_line_edit_text_submitted));
	resize_dialog_vbox->add_margin_child(TTRC("New Size:"), new_size_line_edit);

	vbox->connect("visibility_changed", callable_mp(this, &EditorInspectorArray::_vbox_visibility_changed));
}

////////////////////////////////////////////////
////////////////////////////////////////////////

Ref<EditorInspectorPlugin> EditorInspector::inspector_plugins[MAX_PLUGINS];
int EditorInspector::inspector_plugin_count = 0;

EditorProperty *EditorInspector::instantiate_property_editor(Object *p_object, const Variant::Type p_type, const String &p_path, PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide) {
	for (int i = inspector_plugin_count - 1; i >= 0; i--) {
		inspector_plugins[i]->parse_property(p_object, p_type, p_path, p_hint, p_hint_text, p_usage, p_wide);
		if (inspector_plugins[i]->added_editors.size()) {
			for (int j = 1; j < inspector_plugins[i]->added_editors.size(); j++) { //only keep first one
				memdelete(inspector_plugins[i]->added_editors[j].property_editor);
			}

			EditorProperty *prop = Object::cast_to<EditorProperty>(inspector_plugins[i]->added_editors[0].property_editor);
			if (prop) {
				inspector_plugins[i]->added_editors.clear();
				return prop;
			} else {
				memdelete(inspector_plugins[i]->added_editors[0].property_editor);
				inspector_plugins[i]->added_editors.clear();
			}
		}
	}
	return nullptr;
}

void EditorInspector::add_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin) {
	ERR_FAIL_COND(inspector_plugin_count == MAX_PLUGINS);

	for (int i = 0; i < inspector_plugin_count; i++) {
		if (inspector_plugins[i] == p_plugin) {
			return; //already exists
		}
	}
	inspector_plugins[inspector_plugin_count++] = p_plugin;
}

void EditorInspector::remove_inspector_plugin(const Ref<EditorInspectorPlugin> &p_plugin) {
	ERR_FAIL_COND(inspector_plugin_count == MAX_PLUGINS);

	int idx = -1;
	for (int i = 0; i < inspector_plugin_count; i++) {
		if (inspector_plugins[i] == p_plugin) {
			idx = i;
			break;
		}
	}

	ERR_FAIL_COND_MSG(idx == -1, "Trying to remove nonexistent inspector plugin.");
	for (int i = idx; i < inspector_plugin_count - 1; i++) {
		inspector_plugins[i] = inspector_plugins[i + 1];
	}
	inspector_plugins[inspector_plugin_count - 1] = Ref<EditorInspectorPlugin>();

	inspector_plugin_count--;
}

void EditorInspector::cleanup_plugins() {
	for (int i = 0; i < inspector_plugin_count; i++) {
		inspector_plugins[i].unref();
	}
	inspector_plugin_count = 0;
}

void EditorInspector::set_undo_redo(UndoRedo *p_undo_redo) {
	undo_redo = p_undo_redo;
}

String EditorInspector::get_selected_path() const {
	return property_selected;
}

void EditorInspector::_parse_added_editors(VBoxContainer *current_vbox, Ref<EditorInspectorPlugin> ped) {
	for (const EditorInspectorPlugin::AddedEditor &F : ped->added_editors) {
		EditorProperty *ep = Object::cast_to<EditorProperty>(F.property_editor);
		current_vbox->add_child(F.property_editor);

		if (ep) {
			ep->object = object;
			ep->connect("property_changed", callable_mp(this, &EditorInspector::_property_changed));
			ep->connect("property_keyed", callable_mp(this, &EditorInspector::_property_keyed));
			ep->connect("property_deleted", callable_mp(this, &EditorInspector::_property_deleted), varray(), CONNECT_DEFERRED);
			ep->connect("property_keyed_with_value", callable_mp(this, &EditorInspector::_property_keyed_with_value));
			ep->connect("property_checked", callable_mp(this, &EditorInspector::_property_checked));
			ep->connect("property_pinned", callable_mp(this, &EditorInspector::_property_pinned));
			ep->connect("selected", callable_mp(this, &EditorInspector::_property_selected));
			ep->connect("multiple_properties_changed", callable_mp(this, &EditorInspector::_multiple_properties_changed));
			ep->connect("resource_selected", callable_mp(this, &EditorInspector::_resource_selected), varray(), CONNECT_DEFERRED);
			ep->connect("object_id_selected", callable_mp(this, &EditorInspector::_object_id_selected), varray(), CONNECT_DEFERRED);

			if (F.properties.size()) {
				if (F.properties.size() == 1) {
					//since it's one, associate:
					ep->property = F.properties[0];
					ep->property_usage = 0;
				}

				if (!F.label.is_empty()) {
					ep->set_label(F.label);
				}

				for (int i = 0; i < F.properties.size(); i++) {
					String prop = F.properties[i];

					if (!editor_property_map.has(prop)) {
						editor_property_map[prop] = List<EditorProperty *>();
					}
					editor_property_map[prop].push_back(ep);
				}
			}

			ep->set_read_only(read_only);
			ep->update_property();
			ep->_update_pin_flags();
			ep->update_revert_and_pin_status();
			ep->set_deletable(deletable_properties);
			ep->update_cache();
		}
	}
	ped->added_editors.clear();
}

bool EditorInspector::_is_property_disabled_by_feature_profile(const StringName &p_property) {
	Ref<EditorFeatureProfile> profile = EditorFeatureProfileManager::get_singleton()->get_current_profile();
	if (profile.is_null()) {
		return false;
	}

	StringName class_name = object->get_class();

	while (class_name != StringName()) {
		if (profile->is_class_property_disabled(class_name, p_property)) {
			return true;
		}
		if (profile->is_class_disabled(class_name)) {
			//won't see properties of a disabled class
			return true;
		}
		class_name = ClassDB::get_parent_class(class_name);
	}

	return false;
}

void EditorInspector::update_tree() {
	//to update properly if all is refreshed
	StringName current_selected = property_selected;
	int current_focusable = -1;

	if (property_focusable != -1) {
		//check focusable is really focusable
		bool restore_focus = false;
		Control *focused = get_focus_owner();
		if (focused) {
			Node *parent = focused->get_parent();
			while (parent) {
				EditorInspector *inspector = Object::cast_to<EditorInspector>(parent);
				if (inspector) {
					restore_focus = inspector == this; //may be owned by another inspector
					break; //exit after the first inspector is found, since there may be nested ones
				}
				parent = parent->get_parent();
			}
		}

		if (restore_focus) {
			current_focusable = property_focusable;
		}
	}

	_clear();

	if (!object) {
		return;
	}

	List<Ref<EditorInspectorPlugin>> valid_plugins;

	for (int i = inspector_plugin_count - 1; i >= 0; i--) { //start by last, so lastly added can override newly added
		if (!inspector_plugins[i]->can_handle(object)) {
			continue;
		}
		valid_plugins.push_back(inspector_plugins[i]);
	}

	// Decide if properties should be drawn with the warning color (yellow).
	bool draw_warning = false;
	if (is_inside_tree()) {
		Node *nod = Object::cast_to<Node>(object);
		Node *es = EditorNode::get_singleton()->get_edited_scene();
		if (nod && es != nod && nod->get_owner() != es) {
			// Draw in warning color edited nodes that are not in the currently edited scene,
			// as changes may be lost in the future.
			draw_warning = true;
		}
	}

	String filter = search_box ? search_box->get_text() : "";
	String group;
	String group_base;
	String subgroup;
	String subgroup_base;
	VBoxContainer *category_vbox = nullptr;

	List<PropertyInfo> plist;
	object->get_property_list(&plist, true);
	_update_script_class_properties(*object, plist);

	Map<VBoxContainer *, HashMap<String, VBoxContainer *>> vbox_per_path;
	Map<String, EditorInspectorArray *> editor_inspector_array_per_prefix;

	Color sscolor = get_theme_color(SNAME("prop_subsection"), SNAME("Editor"));

	// Get the lists of editors to add the beginning.
	for (Ref<EditorInspectorPlugin> &ped : valid_plugins) {
		ped->parse_begin(object);
		_parse_added_editors(main_vbox, ped);
	}

	// Get the lists of editors for properties.
	for (List<PropertyInfo>::Element *E_property = plist.front(); E_property; E_property = E_property->next()) {
		PropertyInfo &p = E_property->get();

		if (p.usage & PROPERTY_USAGE_SUBGROUP) {
			// Setup a property sub-group.
			subgroup = p.name;
			subgroup_base = p.hint_string;

			continue;

		} else if (p.usage & PROPERTY_USAGE_GROUP) {
			// Setup a property group.
			group = p.name;
			group_base = p.hint_string;
			subgroup = "";
			subgroup_base = "";

			continue;

		} else if (p.usage & PROPERTY_USAGE_CATEGORY) {
			// Setup a property category.
			group = "";
			group_base = "";
			subgroup = "";
			subgroup_base = "";

			if (!show_categories) {
				continue;
			}

			// Iterate over remaining properties. If no properties in category, skip the category.
			List<PropertyInfo>::Element *N = E_property->next();
			bool valid = true;
			while (N) {
				if (N->get().usage & PROPERTY_USAGE_EDITOR && (!restrict_to_basic || (N->get().usage & PROPERTY_USAGE_EDITOR_BASIC_SETTING))) {
					break;
				}
				if (N->get().usage & PROPERTY_USAGE_CATEGORY) {
					valid = false;
					break;
				}
				N = N->next();
			}
			if (!valid) {
				continue; // Empty, ignore it.
			}

			// Create an EditorInspectorCategory and add it to the inspector.
			EditorInspectorCategory *category = memnew(EditorInspectorCategory);
			main_vbox->add_child(category);
			category_vbox = nullptr; //reset

			String type = p.name;

			// Set the category icon.
			if (!ClassDB::class_exists(type) && !ScriptServer::is_global_class(type) && p.hint_string.length() && FileAccess::exists(p.hint_string)) {
				// If we have a category inside a script, search for the first script with a valid icon.
				Ref<Script> script = ResourceLoader::load(p.hint_string, "Script");
				String base_type;
				if (script.is_valid()) {
					base_type = script->get_instance_base_type();
				}
				while (script.is_valid()) {
					StringName name = EditorNode::get_editor_data().script_class_get_name(script->get_path());
					String icon_path = EditorNode::get_editor_data().script_class_get_icon_path(name);
					if (name != StringName() && icon_path.length()) {
						category->icon = ResourceLoader::load(icon_path, "Texture");
						break;
					}
					script = script->get_base_script();
				}
				if (category->icon.is_null() && has_theme_icon(base_type, SNAME("EditorIcons"))) {
					category->icon = get_theme_icon(base_type, SNAME("EditorIcons"));
				}
			}
			if (category->icon.is_null()) {
				if (!type.is_empty()) { // Can happen for built-in scripts.
					category->icon = EditorNode::get_singleton()->get_class_icon(type, "Object");
				}
			}

			// Set the category label.
			category->label = type;

			if (use_doc_hints) {
				// Sets the category tooltip to show documentation.
				StringName type2 = p.name;
				if (!class_descr_cache.has(type2)) {
					String descr;
					DocTools *dd = EditorHelp::get_doc_data();
					Map<String, DocData::ClassDoc>::Element *E = dd->class_list.find(type2);
					if (E) {
						descr = DTR(E->get().brief_description);
					}
					class_descr_cache[type2] = descr;
				}

				category->set_tooltip(p.name + "::" + (class_descr_cache[type2].is_empty() ? "" : class_descr_cache[type2]));
			}

			// Add editors at the start of a category.
			for (Ref<EditorInspectorPlugin> &ped : valid_plugins) {
				ped->parse_category(object, p.name);
				_parse_added_editors(main_vbox, ped);
			}

			continue;

		} else if (!(p.usage & PROPERTY_USAGE_EDITOR) || _is_property_disabled_by_feature_profile(p.name) || (restrict_to_basic && !(p.usage & PROPERTY_USAGE_EDITOR_BASIC_SETTING))) {
			// Ignore properties that are not supposed to be in the inspector.
			continue;
		}

		if (p.name == "script") {
			// Script should go into its own category.
			category_vbox = nullptr;
		}

		if (p.usage & PROPERTY_USAGE_HIGH_END_GFX && RS::get_singleton()->is_low_end()) {
			// Do not show this property in low end gfx.
			continue;
		}

		if (p.name == "script" && (hide_script || bool(object->call("_hide_script_from_inspector")))) {
			// Hide script variables from inspector if required.
			continue;
		}

		// Get the path for property.
		String path = p.name;

		// First check if we have an array that fits the prefix.
		String array_prefix = "";
		int array_index = -1;
		for (Map<String, EditorInspectorArray *>::Element *E = editor_inspector_array_per_prefix.front(); E; E = E->next()) {
			if (p.name.begins_with(E->key()) && E->key().length() > array_prefix.length()) {
				array_prefix = E->key();
			}
		}

		if (!array_prefix.is_empty()) {
			// If we have an array element, find the according index in array.
			String str = p.name.trim_prefix(array_prefix);
			int to_char_index = 0;
			while (to_char_index < str.length()) {
				if (str[to_char_index] < '0' || str[to_char_index] > '9') {
					break;
				}
				to_char_index++;
			}
			if (to_char_index > 0) {
				array_index = str.left(to_char_index).to_int();
			} else {
				array_prefix = "";
			}
		}

		if (!array_prefix.is_empty()) {
			path = path.trim_prefix(array_prefix);
			int char_index = path.find("/");
			if (char_index >= 0) {
				path = path.right(-char_index - 1);
			} else {
				path = vformat(TTR("Element %s"), array_index);
			}
		} else {
			// Check if we exit or not a subgroup. If there is a prefix, remove it from the property label string.
			if (!subgroup.is_empty() && !subgroup_base.is_empty()) {
				if (path.begins_with(subgroup_base)) {
					path = path.trim_prefix(subgroup_base);
				} else if (subgroup_base.begins_with(path)) {
					// Keep it, this is used pretty often.
				} else {
					subgroup = ""; // The prefix changed, we are no longer in the subgroup.
				}
			}

			// Check if we exit or not a group. If there is a prefix, remove it from the property label string.
			if (!group.is_empty() && !group_base.is_empty() && subgroup.is_empty()) {
				if (path.begins_with(group_base)) {
					path = path.trim_prefix(group_base);
				} else if (group_base.begins_with(path)) {
					// Keep it, this is used pretty often.
				} else {
					group = ""; // The prefix changed, we are no longer in the group.
					subgroup = "";
				}
			}

			// Add the group and subgroup to the path.
			if (!subgroup.is_empty()) {
				path = subgroup + "/" + path;
			}
			if (!group.is_empty()) {
				path = group + "/" + path;
			}
		}

		// Get the property label's string.
		String property_label_string = (path.find("/") != -1) ? path.substr(path.rfind("/") + 1) : path;
		if (capitalize_paths) {
			// Capitalize paths.
			int dot = property_label_string.find(".");
			if (dot != -1) {
				String ov = property_label_string.substr(dot);
				property_label_string = property_label_string.substr(0, dot);
				property_label_string = property_label_string.capitalize();
				property_label_string += ov;
			} else {
				property_label_string = property_label_string.capitalize();
			}
		}

		// Remove the property from the path.
		int idx = path.rfind("/");
		if (idx > -1) {
			path = path.left(idx);
		} else {
			path = "";
		}

		// Ignore properties that do not fit the filter.
		if (use_filter && !filter.is_empty()) {
			if (!filter.is_subsequence_ofi(path) && !filter.is_subsequence_ofi(property_label_string) && property_prefix.to_lower().find(filter.to_lower()) == -1) {
				continue;
			}
		}

		// Recreate the category vbox if it was reset.
		if (category_vbox == nullptr) {
			category_vbox = memnew(VBoxContainer);
			main_vbox->add_child(category_vbox);
		}

		// Find the correct section/vbox to add the property editor to.
		VBoxContainer *root_vbox = array_prefix.is_empty() ? main_vbox : editor_inspector_array_per_prefix[array_prefix]->get_vbox(array_index);
		if (!root_vbox) {
			continue;
		}

		if (!vbox_per_path.has(root_vbox)) {
			vbox_per_path[root_vbox] = HashMap<String, VBoxContainer *>();
			vbox_per_path[root_vbox][""] = root_vbox;
		}

		VBoxContainer *current_vbox = root_vbox;
		String acc_path = "";
		int level = 1;

		Vector<String> components = path.split("/");
		for (int i = 0; i < components.size(); i++) {
			String component = components[i];
			acc_path += (i > 0) ? "/" + component : component;

			if (!vbox_per_path[root_vbox].has(acc_path)) {
				// If the section does not exists, create it.
				EditorInspectorSection *section = memnew(EditorInspectorSection);
				current_vbox->add_child(section);
				sections.push_back(section);

				if (capitalize_paths) {
					component = component.capitalize();
				}

				Color c = sscolor;
				c.a /= level;
				section->setup(acc_path, component, object, c, use_folding);

				// Add editors at the start of a group.
				for (Ref<EditorInspectorPlugin> &ped : valid_plugins) {
					ped->parse_group(object, path);
					_parse_added_editors(section->get_vbox(), ped);
				}

				vbox_per_path[root_vbox][acc_path] = section->get_vbox();
			}

			current_vbox = vbox_per_path[root_vbox][acc_path];
			level = (MIN(level + 1, 4));
		}

		// If we did not find a section to add the property to, add it to the category vbox instead (the category vbox handles margins correctly).
		if (current_vbox == main_vbox) {
			current_vbox = category_vbox;
		}

		// Check if the property is an array counter, if so create a dedicated array editor for the array.
		if (p.usage & PROPERTY_USAGE_ARRAY) {
			EditorInspectorArray *editor_inspector_array = nullptr;
			StringName array_element_prefix;
			Color c = sscolor;
			c.a /= level;
			if (p.type == Variant::NIL) {
				// Setup the array to use a method to create/move/delete elements.
				array_element_prefix = p.class_name;
				editor_inspector_array = memnew(EditorInspectorArray);

				String array_label = (path.find("/") != -1) ? path.substr(path.rfind("/") + 1) : path;
				array_label = property_label_string.capitalize();
				int page = per_array_page.has(array_element_prefix) ? per_array_page[array_element_prefix] : 0;
				editor_inspector_array->setup_with_move_element_function(object, array_label, array_element_prefix, page, c, use_folding);
				editor_inspector_array->connect("page_change_request", callable_mp(this, &EditorInspector::_page_change_request), varray(array_element_prefix));
				editor_inspector_array->set_undo_redo(undo_redo);
			} else if (p.type == Variant::INT) {
				// Setup the array to use the count property and built-in functions to create/move/delete elements.
				Vector<String> class_name_components = String(p.class_name).split(",");
				if (class_name_components.size() == 2) {
					array_element_prefix = class_name_components[1];
					editor_inspector_array = memnew(EditorInspectorArray);
					int page = per_array_page.has(array_element_prefix) ? per_array_page[array_element_prefix] : 0;
					editor_inspector_array->setup_with_count_property(object, class_name_components[0], p.name, array_element_prefix, page, c, use_folding);
					editor_inspector_array->connect("page_change_request", callable_mp(this, &EditorInspector::_page_change_request), varray(array_element_prefix));
					editor_inspector_array->set_undo_redo(undo_redo);
				}
			}

			if (editor_inspector_array) {
				current_vbox->add_child(editor_inspector_array);
				editor_inspector_array_per_prefix[array_element_prefix] = editor_inspector_array;
			}
			continue;
		}

		// Checkable and checked properties.
		bool checkable = false;
		bool checked = false;
		if (p.usage & PROPERTY_USAGE_CHECKABLE) {
			checkable = true;
			checked = p.usage & PROPERTY_USAGE_CHECKED;
		}

		bool property_read_only = (p.usage & PROPERTY_USAGE_READ_ONLY) || read_only;

		// Mark properties that would require an editor restart (mostly when editing editor settings).
		if (p.usage & PROPERTY_USAGE_RESTART_IF_CHANGED) {
			restart_request_props.insert(p.name);
		}

		String doc_hint;

		if (use_doc_hints) {
			// Build the doc hint, to use as tooltip.

			// Get the class name.
			StringName classname = object->get_class_name();
			if (!object_class.is_empty()) {
				classname = object_class;
			}

			StringName propname = property_prefix + p.name;
			String descr;
			bool found = false;

			// Search for the property description in the cache.
			Map<StringName, Map<StringName, String>>::Element *E = descr_cache.find(classname);
			if (E) {
				Map<StringName, String>::Element *F = E->get().find(propname);
				if (F) {
					found = true;
					descr = F->get();
				}
			}

			if (!found) {
				// Build the property description String and add it to the cache.
				DocTools *dd = EditorHelp::get_doc_data();
				Map<String, DocData::ClassDoc>::Element *F = dd->class_list.find(classname);
				while (F && descr.is_empty()) {
					for (int i = 0; i < F->get().properties.size(); i++) {
						if (F->get().properties[i].name == propname.operator String()) {
							descr = DTR(F->get().properties[i].description);
							break;
						}
					}

					Vector<String> slices = propname.operator String().split("/");
					if (slices.size() == 2 && slices[0].begins_with("theme_override_")) {
						for (int i = 0; i < F->get().theme_properties.size(); i++) {
							if (F->get().theme_properties[i].name == slices[1]) {
								descr = DTR(F->get().theme_properties[i].description);
								break;
							}
						}
					}

					if (!F->get().inherits.is_empty()) {
						F = dd->class_list.find(F->get().inherits);
					} else {
						break;
					}
				}
				descr_cache[classname][propname] = descr;
			}

			doc_hint = descr;
		}

		// Search for the inspector plugin that will handle the properties. Then add the correct property editor to it.
		for (Ref<EditorInspectorPlugin> &ped : valid_plugins) {
			bool exclusive = ped->parse_property(object, p.type, p.name, p.hint, p.hint_string, p.usage, wide_editors);

			List<EditorInspectorPlugin::AddedEditor> editors = ped->added_editors; // Make a copy, since plugins may be used again in a sub-inspector.
			ped->added_editors.clear();

			for (const EditorInspectorPlugin::AddedEditor &F : editors) {
				EditorProperty *ep = Object::cast_to<EditorProperty>(F.property_editor);

				if (ep) {
					// Set all this before the control gets the ENTER_TREE notification.
					ep->object = object;

					if (F.properties.size()) {
						if (F.properties.size() == 1) {
							//since it's one, associate:
							ep->property = F.properties[0];
							ep->property_usage = p.usage;
							//and set label?
						}

						if (!F.label.is_empty()) {
							ep->set_label(F.label);
						} else {
							// Use the existing one.
							ep->set_label(property_label_string);
						}
						for (int i = 0; i < F.properties.size(); i++) {
							String prop = F.properties[i];

							if (!editor_property_map.has(prop)) {
								editor_property_map[prop] = List<EditorProperty *>();
							}
							editor_property_map[prop].push_back(ep);
						}
					}
					ep->set_draw_warning(draw_warning);
					ep->set_use_folding(use_folding);
					ep->set_checkable(checkable);
					ep->set_checked(checked);
					ep->set_keying(keying);
					ep->set_read_only(property_read_only);
					ep->set_deletable(deletable_properties);
				}

				current_vbox->add_child(F.property_editor);

				if (ep) {
					// Eventually, set other properties/signals after the property editor got added to the tree.
					bool update_all = (p.usage & PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED);
					ep->connect("property_changed", callable_mp(this, &EditorInspector::_property_changed), varray(update_all));
					ep->connect("property_keyed", callable_mp(this, &EditorInspector::_property_keyed));
					ep->connect("property_deleted", callable_mp(this, &EditorInspector::_property_deleted), varray(), CONNECT_DEFERRED);
					ep->connect("property_keyed_with_value", callable_mp(this, &EditorInspector::_property_keyed_with_value));
					ep->connect("property_checked", callable_mp(this, &EditorInspector::_property_checked));
					ep->connect("property_pinned", callable_mp(this, &EditorInspector::_property_pinned));
					ep->connect("selected", callable_mp(this, &EditorInspector::_property_selected));
					ep->connect("multiple_properties_changed", callable_mp(this, &EditorInspector::_multiple_properties_changed));
					ep->connect("resource_selected", callable_mp(this, &EditorInspector::_resource_selected), varray(), CONNECT_DEFERRED);
					ep->connect("object_id_selected", callable_mp(this, &EditorInspector::_object_id_selected), varray(), CONNECT_DEFERRED);
					if (!doc_hint.is_empty()) {
						ep->set_tooltip(property_prefix + p.name + "::" + doc_hint);
					} else {
						ep->set_tooltip(property_prefix + p.name);
					}
					ep->update_property();
					ep->_update_pin_flags();
					ep->update_revert_and_pin_status();
					ep->update_cache();

					if (current_selected && ep->property == current_selected) {
						ep->select(current_focusable);
					}
				}
			}

			if (exclusive) {
				// If we know the plugin is exclusive, we don't need to go through other plugins.
				break;
			}
		}
	}

	// Get the lists of to add at the end.
	for (Ref<EditorInspectorPlugin> &ped : valid_plugins) {
		ped->parse_end(object);
		_parse_added_editors(main_vbox, ped);
	}
}

void EditorInspector::update_property(const String &p_prop) {
	if (!editor_property_map.has(p_prop)) {
		return;
	}

	for (EditorProperty *E : editor_property_map[p_prop]) {
		E->update_property();
		E->update_revert_and_pin_status();
		E->update_cache();
	}
}

void EditorInspector::_clear() {
	while (main_vbox->get_child_count()) {
		memdelete(main_vbox->get_child(0));
	}
	property_selected = StringName();
	property_focusable = -1;
	editor_property_map.clear();
	sections.clear();
	pending.clear();
	restart_request_props.clear();
}

Object *EditorInspector::get_edited_object() {
	return object;
}

void EditorInspector::edit(Object *p_object) {
	if (object == p_object) {
		return;
	}
	if (object) {
		_clear();
		object->disconnect("property_list_changed", callable_mp(this, &EditorInspector::_changed_callback));
	}
	per_array_page.clear();

	object = p_object;

	if (object) {
		update_scroll_request = 0; //reset
		if (scroll_cache.has(object->get_instance_id())) { //if exists, set something else
			update_scroll_request = scroll_cache[object->get_instance_id()]; //done this way because wait until full size is accommodated
		}
		object->connect("property_list_changed", callable_mp(this, &EditorInspector::_changed_callback));
		update_tree();
	}
}

void EditorInspector::set_keying(bool p_active) {
	if (keying == p_active) {
		return;
	}
	keying = p_active;
	update_tree();
}

void EditorInspector::set_read_only(bool p_read_only) {
	read_only = p_read_only;
	update_tree();
}

bool EditorInspector::is_capitalize_paths_enabled() const {
	return capitalize_paths;
}

void EditorInspector::set_enable_capitalize_paths(bool p_capitalize) {
	capitalize_paths = p_capitalize;
	update_tree();
}

void EditorInspector::set_autoclear(bool p_enable) {
	autoclear = p_enable;
}

void EditorInspector::set_show_categories(bool p_show) {
	show_categories = p_show;
	update_tree();
}

void EditorInspector::set_use_doc_hints(bool p_enable) {
	use_doc_hints = p_enable;
	update_tree();
}

void EditorInspector::set_hide_script(bool p_hide) {
	hide_script = p_hide;
	update_tree();
}

void EditorInspector::set_use_filter(bool p_use) {
	use_filter = p_use;
	update_tree();
}

void EditorInspector::register_text_enter(Node *p_line_edit) {
	search_box = Object::cast_to<LineEdit>(p_line_edit);
	if (search_box) {
		search_box->connect("text_changed", callable_mp(this, &EditorInspector::_filter_changed));
	}
}

void EditorInspector::_filter_changed(const String &p_text) {
	_clear();
	update_tree();
}

void EditorInspector::set_use_folding(bool p_enable) {
	use_folding = p_enable;
	update_tree();
}

bool EditorInspector::is_using_folding() {
	return use_folding;
}

void EditorInspector::collapse_all_folding() {
	for (EditorInspectorSection *E : sections) {
		E->fold();
	}

	for (const KeyValue<StringName, List<EditorProperty *>> &F : editor_property_map) {
		for (EditorProperty *E : F.value) {
			E->collapse_all_folding();
		}
	}
}

void EditorInspector::expand_all_folding() {
	for (EditorInspectorSection *E : sections) {
		E->unfold();
	}
	for (const KeyValue<StringName, List<EditorProperty *>> &F : editor_property_map) {
		for (EditorProperty *E : F.value) {
			E->expand_all_folding();
		}
	}
}

void EditorInspector::set_scroll_offset(int p_offset) {
	set_v_scroll(p_offset);
}

int EditorInspector::get_scroll_offset() const {
	return get_v_scroll();
}

void EditorInspector::set_use_wide_editors(bool p_enable) {
	wide_editors = p_enable;
}

void EditorInspector::_update_inspector_bg() {
	if (sub_inspector) {
		int count_subinspectors = 0;
		Node *n = get_parent();
		while (n) {
			EditorInspector *ei = Object::cast_to<EditorInspector>(n);
			if (ei && ei->sub_inspector) {
				count_subinspectors++;
			}
			n = n->get_parent();
		}
		count_subinspectors = MIN(15, count_subinspectors);
		add_theme_style_override("bg", get_theme_stylebox("sub_inspector_bg" + itos(count_subinspectors), "Editor"));
	} else {
		add_theme_style_override("bg", get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
	}
}
void EditorInspector::set_sub_inspector(bool p_enable) {
	sub_inspector = p_enable;
	if (!is_inside_tree()) {
		return;
	}

	_update_inspector_bg();
}

void EditorInspector::set_use_deletable_properties(bool p_enabled) {
	deletable_properties = p_enabled;
}

void EditorInspector::_page_change_request(int p_new_page, const StringName &p_array_prefix) {
	int prev_page = per_array_page.has(p_array_prefix) ? per_array_page[p_array_prefix] : 0;
	int new_page = MAX(0, p_new_page);
	if (new_page != prev_page) {
		per_array_page[p_array_prefix] = new_page;
		update_tree_pending = true;
	}
}

void EditorInspector::_edit_request_change(Object *p_object, const String &p_property) {
	if (object != p_object) { //may be undoing/redoing for a non edited object, so ignore
		return;
	}

	if (changing) {
		return;
	}

	if (p_property.is_empty()) {
		update_tree_pending = true;
	} else {
		pending.insert(p_property);
	}
}

void EditorInspector::_edit_set(const String &p_name, const Variant &p_value, bool p_refresh_all, const String &p_changed_field) {
	if (autoclear && editor_property_map.has(p_name)) {
		for (EditorProperty *E : editor_property_map[p_name]) {
			if (E->is_checkable()) {
				E->set_checked(true);
			}
		}
	}

	if (!undo_redo || bool(object->call("_dont_undo_redo"))) {
		object->set(p_name, p_value);
		if (p_refresh_all) {
			_edit_request_change(object, "");
		} else {
			_edit_request_change(object, p_name);
		}

		emit_signal(_prop_edited, p_name);

	} else if (Object::cast_to<MultiNodeEdit>(object)) {
		Object::cast_to<MultiNodeEdit>(object)->set_property_field(p_name, p_value, p_changed_field);
		_edit_request_change(object, p_name);
		emit_signal(_prop_edited, p_name);
	} else {
		undo_redo->create_action(vformat(TTR("Set %s"), p_name), UndoRedo::MERGE_ENDS);
		undo_redo->add_do_property(object, p_name, p_value);
		bool valid = false;
		Variant value = object->get(p_name, &valid);
		if (valid) {
			undo_redo->add_undo_property(object, p_name, value);
		}

		PropertyInfo prop_info;
		if (ClassDB::get_property_info(object->get_class_name(), p_name, &prop_info)) {
			for (const String &linked_prop : prop_info.linked_properties) {
				valid = false;
				value = object->get(linked_prop, &valid);
				if (valid) {
					undo_redo->add_undo_property(object, linked_prop, value);
				}
			}
		}

		Variant v_undo_redo = (Object *)undo_redo;
		Variant v_object = object;
		Variant v_name = p_name;
		for (int i = 0; i < EditorNode::get_singleton()->get_editor_data().get_undo_redo_inspector_hook_callback().size(); i++) {
			const Callable &callback = EditorNode::get_singleton()->get_editor_data().get_undo_redo_inspector_hook_callback()[i];

			const Variant *p_arguments[] = { &v_undo_redo, &v_object, &v_name, &p_value };
			Variant return_value;
			Callable::CallError call_error;

			callback.call(p_arguments, 4, return_value, call_error);
			if (call_error.error != Callable::CallError::CALL_OK) {
				ERR_PRINT("Invalid UndoRedo callback.");
			}
		}

		if (p_refresh_all) {
			undo_redo->add_do_method(this, "_edit_request_change", object, "");
			undo_redo->add_undo_method(this, "_edit_request_change", object, "");
		} else {
			undo_redo->add_do_method(this, "_edit_request_change", object, p_name);
			undo_redo->add_undo_method(this, "_edit_request_change", object, p_name);
		}

		Resource *r = Object::cast_to<Resource>(object);
		if (r) {
			if (String(p_name) == "resource_local_to_scene") {
				bool prev = object->get(p_name);
				bool next = p_value;
				if (next) {
					undo_redo->add_do_method(r, "setup_local_to_scene");
				}
				if (prev) {
					undo_redo->add_undo_method(r, "setup_local_to_scene");
				}
			}
		}
		undo_redo->add_do_method(this, "emit_signal", _prop_edited, p_name);
		undo_redo->add_undo_method(this, "emit_signal", _prop_edited, p_name);
		undo_redo->commit_action();
	}

	if (editor_property_map.has(p_name)) {
		for (EditorProperty *E : editor_property_map[p_name]) {
			E->update_revert_and_pin_status();
		}
	}
}

void EditorInspector::_property_changed(const String &p_path, const Variant &p_value, const String &p_name, bool p_changing, bool p_update_all) {
	// The "changing" variable must be true for properties that trigger events as typing occurs,
	// like "text_changed" signal. E.g. text property of Label, Button, RichTextLabel, etc.
	if (p_changing) {
		this->changing++;
	}

	_edit_set(p_path, p_value, p_update_all, p_name);

	if (p_changing) {
		this->changing--;
	}

	if (restart_request_props.has(p_path)) {
		emit_signal(SNAME("restart_requested"));
	}
}

void EditorInspector::_multiple_properties_changed(Vector<String> p_paths, Array p_values, bool p_changing) {
	ERR_FAIL_COND(p_paths.size() == 0 || p_values.size() == 0);
	ERR_FAIL_COND(p_paths.size() != p_values.size());
	String names;
	for (int i = 0; i < p_paths.size(); i++) {
		if (i > 0) {
			names += ",";
		}
		names += p_paths[i];
	}
	undo_redo->create_action(TTR("Set Multiple:") + " " + names, UndoRedo::MERGE_ENDS);
	for (int i = 0; i < p_paths.size(); i++) {
		_edit_set(p_paths[i], p_values[i], false, "");
		if (restart_request_props.has(p_paths[i])) {
			emit_signal(SNAME("restart_requested"));
		}
	}
	if (p_changing) {
		changing++;
	}
	undo_redo->commit_action();
	if (p_changing) {
		changing--;
	}
}

void EditorInspector::_property_keyed(const String &p_path, bool p_advance) {
	if (!object) {
		return;
	}

	// The second parameter could be null, causing the event to fire with less arguments, so use the pointer call which preserves it.
	const Variant args[3] = { p_path, object->get(p_path), p_advance };
	const Variant *argp[3] = { &args[0], &args[1], &args[2] };
	emit_signal(SNAME("property_keyed"), argp, 3);
}

void EditorInspector::_property_deleted(const String &p_path) {
	if (!object) {
		return;
	}

	emit_signal(SNAME("property_deleted"), p_path);
}

void EditorInspector::_property_keyed_with_value(const String &p_path, const Variant &p_value, bool p_advance) {
	if (!object) {
		return;
	}

	// The second parameter could be null, causing the event to fire with less arguments, so use the pointer call which preserves it.
	const Variant args[3] = { p_path, p_value, p_advance };
	const Variant *argp[3] = { &args[0], &args[1], &args[2] };
	emit_signal(SNAME("property_keyed"), argp, 3);
}

void EditorInspector::_property_checked(const String &p_path, bool p_checked) {
	if (!object) {
		return;
	}

	//property checked
	if (autoclear) {
		if (!p_checked) {
			object->set(p_path, Variant());
		} else {
			Variant to_create;
			List<PropertyInfo> pinfo;
			object->get_property_list(&pinfo);
			for (const PropertyInfo &E : pinfo) {
				if (E.name == p_path) {
					Callable::CallError ce;
					Variant::construct(E.type, to_create, nullptr, 0, ce);
					break;
				}
			}
			object->set(p_path, to_create);
		}

		if (editor_property_map.has(p_path)) {
			for (EditorProperty *E : editor_property_map[p_path]) {
				E->update_property();
				E->update_revert_and_pin_status();
				E->update_cache();
			}
		}

	} else {
		emit_signal(SNAME("property_toggled"), p_path, p_checked);
	}
}

void EditorInspector::_property_pinned(const String &p_path, bool p_pinned) {
	if (!object) {
		return;
	}

	Node *node = Object::cast_to<Node>(object);
	ERR_FAIL_COND(!node);

	if (undo_redo) {
		undo_redo->create_action(vformat(p_pinned ? TTR("Pinned %s") : TTR("Unpinned %s"), p_path));
		undo_redo->add_do_method(node, "_set_property_pinned", p_path, p_pinned);
		undo_redo->add_undo_method(node, "_set_property_pinned", p_path, !p_pinned);
		if (editor_property_map.has(p_path)) {
			for (List<EditorProperty *>::Element *E = editor_property_map[p_path].front(); E; E = E->next()) {
				undo_redo->add_do_method(E->get(), "_update_revert_and_pin_status");
				undo_redo->add_undo_method(E->get(), "_update_revert_and_pin_status");
			}
		}
		undo_redo->commit_action();
	} else {
		node->set_property_pinned(p_path, p_pinned);
		if (editor_property_map.has(p_path)) {
			for (List<EditorProperty *>::Element *E = editor_property_map[p_path].front(); E; E = E->next()) {
				E->get()->update_revert_and_pin_status();
			}
		}
	}
}

void EditorInspector::_property_selected(const String &p_path, int p_focusable) {
	property_selected = p_path;
	property_focusable = p_focusable;
	//deselect the others
	for (const KeyValue<StringName, List<EditorProperty *>> &F : editor_property_map) {
		if (F.key == property_selected) {
			continue;
		}
		for (EditorProperty *E : F.value) {
			if (E->is_selected()) {
				E->deselect();
			}
		}
	}

	emit_signal(SNAME("property_selected"), p_path);
}

void EditorInspector::_object_id_selected(const String &p_path, ObjectID p_id) {
	emit_signal(SNAME("object_id_selected"), p_id);
}

void EditorInspector::_resource_selected(const String &p_path, RES p_resource) {
	emit_signal(SNAME("resource_selected"), p_resource, p_path);
}

void EditorInspector::_node_removed(Node *p_node) {
	if (p_node == object) {
		edit(nullptr);
	}
}

void EditorInspector::_notification(int p_what) {
	if (p_what == NOTIFICATION_READY) {
		EditorFeatureProfileManager::get_singleton()->connect("current_feature_profile_changed", callable_mp(this, &EditorInspector::_feature_profile_changed));
		set_process(is_visible_in_tree());
		_update_inspector_bg();
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {
		if (!sub_inspector) {
			get_tree()->connect("node_removed", callable_mp(this, &EditorInspector::_node_removed));
		}
	}
	if (p_what == NOTIFICATION_PREDELETE) {
		edit(nullptr); //just in case
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {
		if (!sub_inspector) {
			get_tree()->disconnect("node_removed", callable_mp(this, &EditorInspector::_node_removed));
		}
		edit(nullptr);
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		set_process(is_visible_in_tree());
	}

	if (p_what == NOTIFICATION_PROCESS) {
		if (update_scroll_request >= 0) {
			get_v_scroll_bar()->call_deferred(SNAME("set_value"), update_scroll_request);
			update_scroll_request = -1;
		}
		if (refresh_countdown > 0) {
			refresh_countdown -= get_process_delta_time();
			if (refresh_countdown <= 0) {
				for (const KeyValue<StringName, List<EditorProperty *>> &F : editor_property_map) {
					for (EditorProperty *E : F.value) {
						if (!E->is_cache_valid()) {
							E->update_property();
							E->update_revert_and_pin_status();
							E->update_cache();
						}
					}
				}
				refresh_countdown = float(EditorSettings::get_singleton()->get("docks/property_editor/auto_refresh_interval"));
			}
		}

		changing++;

		if (update_tree_pending) {
			update_tree();
			update_tree_pending = false;
			pending.clear();

		} else {
			while (pending.size()) {
				StringName prop = pending.front()->get();
				if (editor_property_map.has(prop)) {
					for (EditorProperty *E : editor_property_map[prop]) {
						E->update_property();
						E->update_revert_and_pin_status();
						E->update_cache();
					}
				}
				pending.erase(pending.front());
			}
		}

		changing--;
	}

	if (p_what == EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED) {
		_update_inspector_bg();

		update_tree();
	}
}

void EditorInspector::_changed_callback() {
	//this is called when property change is notified via notify_property_list_changed()
	if (object != nullptr) {
		_edit_request_change(object, String());
	}
}

void EditorInspector::_vscroll_changed(double p_offset) {
	if (update_scroll_request >= 0) { //waiting, do nothing
		return;
	}

	if (object) {
		scroll_cache[object->get_instance_id()] = p_offset;
	}
}

void EditorInspector::set_property_prefix(const String &p_prefix) {
	property_prefix = p_prefix;
}

String EditorInspector::get_property_prefix() const {
	return property_prefix;
}

void EditorInspector::set_object_class(const String &p_class) {
	object_class = p_class;
}

String EditorInspector::get_object_class() const {
	return object_class;
}

void EditorInspector::_feature_profile_changed() {
	update_tree();
}

void EditorInspector::_update_script_class_properties(const Object &p_object, List<PropertyInfo> &r_list) const {
	Ref<Script> script = p_object.get_script();
	if (script.is_null()) {
		return;
	}

	List<Ref<Script>> classes;

	// NodeC -> NodeB -> NodeA
	while (script.is_valid()) {
		classes.push_front(script);
		script = script->get_base_script();
	}

	if (classes.is_empty()) {
		return;
	}

	// Script Variables -> to insert: NodeC..B..A -> bottom (insert_here)
	List<PropertyInfo>::Element *script_variables = nullptr;
	List<PropertyInfo>::Element *bottom = nullptr;
	List<PropertyInfo>::Element *insert_here = nullptr;
	for (List<PropertyInfo>::Element *E = r_list.front(); E; E = E->next()) {
		PropertyInfo &pi = E->get();
		if (pi.name != "Script Variables") {
			continue;
		}
		script_variables = E;
		bottom = r_list.insert_after(script_variables, PropertyInfo());
		insert_here = bottom;
		break;
	}

	Set<StringName> added;
	for (const Ref<Script> &s : classes) {
		String path = s->get_path();
		String name = EditorNode::get_editor_data().script_class_get_name(path);
		if (name.is_empty()) {
			if (!s->is_built_in()) {
				name = path.get_file();
			} else {
				name = TTR("Built-in script");
			}
		}

		List<PropertyInfo> props;
		s->get_script_property_list(&props);

		// Script Variables -> NodeA -> bottom (insert_here)
		List<PropertyInfo>::Element *category = r_list.insert_before(insert_here, PropertyInfo(Variant::NIL, name, PROPERTY_HINT_NONE, path, PROPERTY_USAGE_CATEGORY));

		// Script Variables -> NodeA -> A props... -> bottom (insert_here)
		for (List<PropertyInfo>::Element *P = props.front(); P; P = P->next()) {
			PropertyInfo &pi = P->get();
			if (added.has(pi.name)) {
				continue;
			}
			added.insert(pi.name);

			r_list.insert_before(insert_here, pi);
		}

		// Script Variables -> NodeA (insert_here) -> A props... -> bottom
		insert_here = category;
	}

	// NodeC -> C props... -> NodeB..C..
	if (script_variables) {
		r_list.erase(script_variables);
		List<PropertyInfo>::Element *to_delete = bottom->next();
		while (to_delete && !(to_delete->get().usage & PROPERTY_USAGE_CATEGORY)) {
			r_list.erase(to_delete);
			to_delete = bottom->next();
		}
		r_list.erase(bottom);
	}
}

void EditorInspector::set_restrict_to_basic_settings(bool p_restrict) {
	restrict_to_basic = p_restrict;
	update_tree();
}

void EditorInspector::set_property_clipboard(const Variant &p_value) {
	property_clipboard = p_value;
}

Variant EditorInspector::get_property_clipboard() const {
	return property_clipboard;
}

void EditorInspector::_bind_methods() {
	ClassDB::bind_method("_edit_request_change", &EditorInspector::_edit_request_change);

	ADD_SIGNAL(MethodInfo("property_selected", PropertyInfo(Variant::STRING, "property")));
	ADD_SIGNAL(MethodInfo("property_keyed", PropertyInfo(Variant::STRING, "property"), PropertyInfo(Variant::NIL, "value", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT), PropertyInfo(Variant::BOOL, "advance")));
	ADD_SIGNAL(MethodInfo("property_deleted", PropertyInfo(Variant::STRING, "property")));
	ADD_SIGNAL(MethodInfo("resource_selected", PropertyInfo(Variant::OBJECT, "res"), PropertyInfo(Variant::STRING, "prop")));
	ADD_SIGNAL(MethodInfo("object_id_selected", PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("property_edited", PropertyInfo(Variant::STRING, "property")));
	ADD_SIGNAL(MethodInfo("property_toggled", PropertyInfo(Variant::STRING, "property"), PropertyInfo(Variant::BOOL, "checked")));
	ADD_SIGNAL(MethodInfo("restart_requested"));
}

EditorInspector::EditorInspector() {
	object = nullptr;
	undo_redo = nullptr;
	main_vbox = memnew(VBoxContainer);
	main_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
	main_vbox->add_theme_constant_override("separation", 0);
	add_child(main_vbox);
	set_horizontal_scroll_mode(SCROLL_MODE_DISABLED);

	wide_editors = false;
	show_categories = false;
	hide_script = true;
	use_doc_hints = false;
	capitalize_paths = true;
	use_filter = false;
	autoclear = false;
	changing = 0;
	use_folding = false;
	update_all_pending = false;
	update_tree_pending = false;
	read_only = false;
	search_box = nullptr;
	keying = false;
	_prop_edited = "property_edited";
	set_process(false);
	property_focusable = -1;
	sub_inspector = false;
	deletable_properties = false;
	property_clipboard = Variant();

	get_v_scroll_bar()->connect("value_changed", callable_mp(this, &EditorInspector::_vscroll_changed));
	update_scroll_request = -1;
	if (EditorSettings::get_singleton()) {
		refresh_countdown = float(EditorSettings::get_singleton()->get("docks/property_editor/auto_refresh_interval"));
	} else {
		//used when class is created by the docgen to dump default values of everything bindable, editorsettings may not be created
		refresh_countdown = 0.33;
	}

	ED_SHORTCUT("property_editor/copy_property", TTR("Copy Property"), KeyModifierMask::CMD | Key::C);
	ED_SHORTCUT("property_editor/paste_property", TTR("Paste Property"), KeyModifierMask::CMD | Key::V);
	ED_SHORTCUT("property_editor/copy_property_path", TTR("Copy Property Path"), KeyModifierMask::CMD | KeyModifierMask::SHIFT | Key::C);
}
