/**************************************************************************/
/*  animation_tree_player_editor_plugin.cpp                               */
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

#include "animation_tree_player_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/os/input.h"
#include "core/os/keyboard.h"
#include "core/project_settings.h"
#include "editor/editor_scale.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/main/viewport.h"

void AnimationTreePlayerEditor::edit(AnimationTreePlayer *p_anim_tree) {
	anim_tree = p_anim_tree;

	if (!anim_tree) {
		hide();
	} else {
		order.clear();
		p_anim_tree->get_node_list(&order);
		/*
		for(List<StringName>::Element* E=order.front();E;E=E->next()) {

			if (E->get() >= (int)last_id)
				last_id=E->get()+1;
		}*/
		play_button->set_pressed(p_anim_tree->is_active());
		//read the orders
	}
}

Size2 AnimationTreePlayerEditor::_get_maximum_size() {
	Size2 max;

	for (List<StringName>::Element *E = order.front(); E; E = E->next()) {
		Point2 pos = anim_tree->node_get_position(E->get());

		if (click_type == CLICK_NODE && click_node == E->get()) {
			pos += click_motion - click_pos;
		}
		pos += get_node_size(E->get());
		if (pos.x > max.x) {
			max.x = pos.x;
		}
		if (pos.y > max.y) {
			max.y = pos.y;
		}
	}

	return max;
}

const char *AnimationTreePlayerEditor::_node_type_names[] = { "Output", "Animation", "OneShot", "Mix", "Blend2", "Blend3", "Blend4", "TimeScale", "TimeSeek", "Transition" };

Size2 AnimationTreePlayerEditor::get_node_size(const StringName &p_node) const {
	AnimationTreePlayer::NodeType type = anim_tree->node_get_type(p_node);

	Ref<StyleBox> style = get_stylebox("panel", "PopupMenu");
	Ref<Font> font = get_font("font", "PopupMenu");

	Size2 size = style->get_minimum_size();

	int count = 2; // title and name
	int inputs = anim_tree->node_get_input_count(p_node);
	count += inputs ? inputs : 1;
	String name = p_node;

	float name_w = font->get_string_size(name).width;
	float type_w = font->get_string_size(String(_node_type_names[type])).width;
	float max_w = MAX(name_w, type_w);

	switch (type) {
		case AnimationTreePlayer::NODE_TIMESEEK:
		case AnimationTreePlayer::NODE_OUTPUT: {
		} break;
		case AnimationTreePlayer::NODE_ANIMATION:
		case AnimationTreePlayer::NODE_ONESHOT:
		case AnimationTreePlayer::NODE_MIX:
		case AnimationTreePlayer::NODE_BLEND2:
		case AnimationTreePlayer::NODE_BLEND3:
		case AnimationTreePlayer::NODE_BLEND4:
		case AnimationTreePlayer::NODE_TIMESCALE:
		case AnimationTreePlayer::NODE_TRANSITION: {
			size.height += font->get_height();
		} break;
		case AnimationTreePlayer::NODE_MAX: {
		}
	}

	size.x += max_w + 20;
	size.y += count * (font->get_height() + get_constant("vseparation", "PopupMenu"));

	return size;
}

void AnimationTreePlayerEditor::_edit_dialog_changede(String) {
	edit_dialog->hide();
}

void AnimationTreePlayerEditor::_edit_dialog_changeds(String s) {
	_edit_dialog_changed();
}

void AnimationTreePlayerEditor::_edit_dialog_changedf(float) {
	_edit_dialog_changed();
}

void AnimationTreePlayerEditor::_edit_dialog_changed() {
	if (updating_edit) {
		return;
	}

	if (renaming_edit) {
		if (anim_tree->node_rename(edited_node, edit_line[0]->get_text()) == OK) {
			for (List<StringName>::Element *E = order.front(); E; E = E->next()) {
				if (E->get() == edited_node) {
					E->get() = edit_line[0]->get_text();
				}
			}
			edited_node = edit_line[0]->get_text();
		}
		update();
		return;
	}

	AnimationTreePlayer::NodeType type = anim_tree->node_get_type(edited_node);

	switch (type) {
		case AnimationTreePlayer::NODE_TIMESCALE:
			anim_tree->timescale_node_set_scale(edited_node, edit_line[0]->get_text().to_double());
			break;
		case AnimationTreePlayer::NODE_ONESHOT:
			anim_tree->oneshot_node_set_fadein_time(edited_node, edit_line[0]->get_text().to_double());
			anim_tree->oneshot_node_set_fadeout_time(edited_node, edit_line[1]->get_text().to_double());
			anim_tree->oneshot_node_set_autorestart_delay(edited_node, edit_line[2]->get_text().to_double());
			anim_tree->oneshot_node_set_autorestart_random_delay(edited_node, edit_line[3]->get_text().to_double());
			anim_tree->oneshot_node_set_autorestart(edited_node, edit_check->is_pressed());
			anim_tree->oneshot_node_set_mix_mode(edited_node, edit_option->get_selected());

			break;

		case AnimationTreePlayer::NODE_MIX:

			anim_tree->mix_node_set_amount(edited_node, edit_scroll[0]->get_value());
			break;
		case AnimationTreePlayer::NODE_BLEND2:
			anim_tree->blend2_node_set_amount(edited_node, edit_scroll[0]->get_value());

			break;

		case AnimationTreePlayer::NODE_BLEND3:
			anim_tree->blend3_node_set_amount(edited_node, edit_scroll[0]->get_value());

			break;
		case AnimationTreePlayer::NODE_BLEND4:

			anim_tree->blend4_node_set_amount(edited_node, Point2(edit_scroll[0]->get_value(), edit_scroll[1]->get_value()));

			break;

		case AnimationTreePlayer::NODE_TRANSITION: {
			anim_tree->transition_node_set_xfade_time(edited_node, edit_line[0]->get_text().to_double());
			if (anim_tree->transition_node_get_current(edited_node) != edit_option->get_selected()) {
				anim_tree->transition_node_set_current(edited_node, edit_option->get_selected());
			}
		} break;
		default: {
		}
	}
}

void AnimationTreePlayerEditor::_edit_dialog_animation_changed() {
	Ref<Animation> anim = property_editor->get_variant().operator RefPtr();
	anim_tree->animation_node_set_animation(edited_node, anim);
	update();
}

void AnimationTreePlayerEditor::_edit_dialog_edit_animation() {
	if (Engine::get_singleton()->is_editor_hint()) {
		get_tree()->get_root()->get_child(0)->call("_resource_selected", property_editor->get_variant().operator RefPtr());
	};
};

void AnimationTreePlayerEditor::_edit_oneshot_start() {
	anim_tree->oneshot_node_start(edited_node);
}

void AnimationTreePlayerEditor::_play_toggled() {
	anim_tree->set_active(play_button->is_pressed());
}

void AnimationTreePlayerEditor::_master_anim_menu_item(int p_item) {
	if (p_item == 0) {
		_edit_filters();
	} else {
		String str = master_anim_popup->get_item_text(p_item);
		anim_tree->animation_node_set_master_animation(edited_node, str);
	}
	update();
}

void AnimationTreePlayerEditor::_popup_edit_dialog() {
	updating_edit = true;

	for (int i = 0; i < 2; i++) {
		edit_scroll[i]->hide();
	}

	for (int i = 0; i < 4; i++) {
		edit_line[i]->hide();
		edit_label[i]->hide();
	}

	edit_option->hide();
	edit_button->hide();
	filter_button->hide();
	edit_check->hide();

	Point2 pos = anim_tree->node_get_position(edited_node) - Point2(h_scroll->get_value(), v_scroll->get_value());
	Ref<StyleBox> style = get_stylebox("panel", "PopupMenu");
	Size2 size = get_node_size(edited_node);
	Point2 popup_pos(pos.x + style->get_margin(MARGIN_LEFT), pos.y + size.y - style->get_margin(MARGIN_BOTTOM));
	popup_pos += get_global_position();

	if (renaming_edit) {
		edit_label[0]->set_text(TTR("New name:"));
		edit_label[0]->show();
		edit_line[0]->set_text(edited_node);
		edit_line[0]->show();
		edit_dialog->set_size(Size2(150, 50));

	} else {
		AnimationTreePlayer::NodeType type = anim_tree->node_get_type(edited_node);

		switch (type) {
			case AnimationTreePlayer::NODE_ANIMATION:

				if (anim_tree->get_master_player() != NodePath() && anim_tree->has_node(anim_tree->get_master_player()) && Object::cast_to<AnimationPlayer>(anim_tree->get_node(anim_tree->get_master_player()))) {
					AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(anim_tree->get_node(anim_tree->get_master_player()));
					master_anim_popup->clear();
					master_anim_popup->add_item(TTR("Edit Filters"));
					master_anim_popup->add_separator();
					List<StringName> sn;
					ap->get_animation_list(&sn);
					sn.sort_custom<StringName::AlphCompare>();
					for (List<StringName>::Element *E = sn.front(); E; E = E->next()) {
						master_anim_popup->add_item(E->get());
					}

					master_anim_popup->set_position(popup_pos);
					master_anim_popup->popup();
				} else {
					property_editor->edit(this, "", Variant::OBJECT, anim_tree->animation_node_get_animation(edited_node), PROPERTY_HINT_RESOURCE_TYPE, "Animation");
					property_editor->set_position(popup_pos);
					property_editor->popup();
					updating_edit = false;
				}
				return;
			case AnimationTreePlayer::NODE_TIMESCALE:
				edit_label[0]->set_text(TTR("Scale:"));
				edit_label[0]->show();
				edit_line[0]->set_text(rtos(anim_tree->timescale_node_get_scale(edited_node)));
				edit_line[0]->show();
				edit_dialog->set_size(Size2(150, 50));
				break;
			case AnimationTreePlayer::NODE_ONESHOT:
				edit_label[0]->set_text(TTR("Fade In (s):"));
				edit_label[0]->show();
				edit_line[0]->set_text(rtos(anim_tree->oneshot_node_get_fadein_time(edited_node)));
				edit_line[0]->show();
				edit_label[1]->set_text(TTR("Fade Out (s):"));
				edit_label[1]->show();
				edit_line[1]->set_text(rtos(anim_tree->oneshot_node_get_fadeout_time(edited_node)));
				edit_line[1]->show();

				edit_option->clear();
				edit_option->add_item(TTR("Blend"), 0);
				edit_option->add_item(TTR("Add"), 1);

				edit_option->select(anim_tree->oneshot_node_get_mix_mode(edited_node));
				edit_option->show();

				edit_check->set_text(TTR("Auto Restart:"));
				edit_check->set_pressed(anim_tree->oneshot_node_has_autorestart(edited_node));
				edit_check->show();

				edit_label[2]->set_text(TTR("Restart (s):"));
				edit_label[2]->show();
				edit_line[2]->set_text(rtos(anim_tree->oneshot_node_get_autorestart_delay(edited_node)));
				edit_line[2]->show();
				edit_label[3]->set_text(TTR("Random Restart (s):"));
				edit_label[3]->show();
				edit_line[3]->set_text(rtos(anim_tree->oneshot_node_get_autorestart_random_delay(edited_node)));
				edit_line[3]->show();

				filter_button->show();

				edit_button->set_text(TTR("Start!"));

				edit_button->show();

				edit_dialog->set_size(Size2(180, 293));

				break;

			case AnimationTreePlayer::NODE_MIX:

				edit_label[0]->set_text(TTR("Amount:"));
				edit_label[0]->show();
				edit_scroll[0]->set_min(0);
				edit_scroll[0]->set_max(1);
				edit_scroll[0]->set_step(0.01);
				edit_scroll[0]->set_value(anim_tree->mix_node_get_amount(edited_node));
				edit_scroll[0]->show();
				edit_dialog->set_size(Size2(150, 50));

				break;
			case AnimationTreePlayer::NODE_BLEND2:
				edit_label[0]->set_text(TTR("Blend:"));
				edit_label[0]->show();
				edit_scroll[0]->set_min(0);
				edit_scroll[0]->set_max(1);
				edit_scroll[0]->set_step(0.01);
				edit_scroll[0]->set_value(anim_tree->blend2_node_get_amount(edited_node));
				edit_scroll[0]->show();
				filter_button->show();
				edit_dialog->set_size(Size2(150, 74));

				break;

			case AnimationTreePlayer::NODE_BLEND3:
				edit_label[0]->set_text(TTR("Blend:"));
				edit_label[0]->show();
				edit_scroll[0]->set_min(-1);
				edit_scroll[0]->set_max(1);
				edit_scroll[0]->set_step(0.01);
				edit_scroll[0]->set_value(anim_tree->blend3_node_get_amount(edited_node));
				edit_scroll[0]->show();
				edit_dialog->set_size(Size2(150, 50));

				break;
			case AnimationTreePlayer::NODE_BLEND4:

				edit_label[0]->set_text(TTR("Blend 0:"));
				edit_label[0]->show();
				edit_scroll[0]->set_min(0);
				edit_scroll[0]->set_max(1);
				edit_scroll[0]->set_step(0.01);
				edit_scroll[0]->set_value(anim_tree->blend4_node_get_amount(edited_node).x);
				edit_scroll[0]->show();
				edit_label[1]->set_text(TTR("Blend 1:"));
				edit_label[1]->show();
				edit_scroll[1]->set_min(0);
				edit_scroll[1]->set_max(1);
				edit_scroll[1]->set_step(0.01);
				edit_scroll[1]->set_value(anim_tree->blend4_node_get_amount(edited_node).y);
				edit_scroll[1]->show();
				edit_dialog->set_size(Size2(150, 100));

				break;

			case AnimationTreePlayer::NODE_TRANSITION: {
				edit_label[0]->set_text(TTR("X-Fade Time (s):"));
				edit_label[0]->show();
				edit_line[0]->set_text(rtos(anim_tree->transition_node_get_xfade_time(edited_node)));
				edit_line[0]->show();

				edit_option->clear();

				for (int i = 0; i < anim_tree->transition_node_get_input_count(edited_node); i++) {
					edit_option->add_item(itos(i), i);
				}

				edit_option->select(anim_tree->transition_node_get_current(edited_node));
				edit_option->show();
				edit_dialog->set_size(Size2(150, 100));

			} break;
			default: {
			}
		}
	}

	edit_dialog->set_position(popup_pos);
	edit_dialog->popup();

	updating_edit = false;
}

void AnimationTreePlayerEditor::_draw_node(const StringName &p_node) {
	RID ci = get_canvas_item();
	AnimationTreePlayer::NodeType type = anim_tree->node_get_type(p_node);

	Ref<StyleBox> style = get_stylebox("panel", "PopupMenu");
	Ref<Font> font = get_font("font", "PopupMenu");
	Color font_color = get_color("font_color", "PopupMenu");
	Color font_color_title = get_color("font_color_hover", "PopupMenu");
	font_color_title.a *= 0.8;
	Ref<Texture> slot_icon = get_icon("VisualShaderPort", "EditorIcons");

	Size2 size = get_node_size(p_node);
	Point2 pos = anim_tree->node_get_position(p_node);
	if (click_type == CLICK_NODE && click_node == p_node) {
		pos += click_motion - click_pos;
		if (pos.x < 5) {
			pos.x = 5;
		}
		if (pos.y < 5) {
			pos.y = 5;
		}
	}

	pos -= Point2(h_scroll->get_value(), v_scroll->get_value());

	style->draw(ci, Rect2(pos, size));

	float w = size.width - style->get_minimum_size().width;
	float h = font->get_height() + get_constant("vseparation", "PopupMenu");

	Point2 ofs = style->get_offset() + pos;
	Point2 ascofs(0, font->get_ascent());

	Color bx = font_color_title;
	bx.a *= 0.1;
	draw_rect(Rect2(ofs, Size2(size.width - style->get_minimum_size().width, font->get_height())), bx);
	font->draw_halign(ci, ofs + ascofs, HALIGN_CENTER, w, String(_node_type_names[type]), font_color_title);

	ofs.y += h;
	font->draw_halign(ci, ofs + ascofs, HALIGN_CENTER, w, p_node, font_color);
	ofs.y += h;

	int inputs = anim_tree->node_get_input_count(p_node);

	float icon_h_ofs = Math::floor((font->get_height() - slot_icon->get_height()) / 2.0) + 1;

	if (type != AnimationTreePlayer::NODE_OUTPUT) {
		slot_icon->draw(ci, ofs + Point2(w, icon_h_ofs)); //output
	}

	if (inputs) {
		for (int i = 0; i < inputs; i++) {
			slot_icon->draw(ci, ofs + Point2(-slot_icon->get_width(), icon_h_ofs));
			String text;
			switch (type) {
				case AnimationTreePlayer::NODE_TIMESCALE:
				case AnimationTreePlayer::NODE_TIMESEEK:
					text = "in";
					break;
				case AnimationTreePlayer::NODE_OUTPUT:
					text = "out";
					break;
				case AnimationTreePlayer::NODE_ANIMATION:
					break;
				case AnimationTreePlayer::NODE_ONESHOT:
					text = (i == 0 ? "in" : "add");
					break;
				case AnimationTreePlayer::NODE_BLEND2:
				case AnimationTreePlayer::NODE_MIX:
					text = (i == 0 ? "a" : "b");
					break;
				case AnimationTreePlayer::NODE_BLEND3:
					switch (i) {
						case 0:
							text = "b-";
							break;
						case 1:
							text = "a";
							break;
						case 2:
							text = "b+";
							break;
					}
					break;

				case AnimationTreePlayer::NODE_BLEND4:
					switch (i) {
						case 0:
							text = "a0";
							break;
						case 1:
							text = "b0";
							break;
						case 2:
							text = "a1";
							break;
						case 3:
							text = "b1";
							break;
					}
					break;

				case AnimationTreePlayer::NODE_TRANSITION:
					text = itos(i);
					if (anim_tree->transition_node_has_input_auto_advance(p_node, i)) {
						text += "->";
					}

					break;
				default: {
				}
			}
			font->draw(ci, ofs + ascofs + Point2(3, 0), text, font_color);

			ofs.y += h;
		}
	} else {
		ofs.y += h;
	}

	Ref<StyleBox> pg_bg = get_stylebox("bg", "ProgressBar");
	Ref<StyleBox> pg_fill = get_stylebox("fill", "ProgressBar");
	Rect2 pg_rect(ofs, Size2(w, h));

	bool editable = true;
	switch (type) {
		case AnimationTreePlayer::NODE_ANIMATION: {
			Ref<Animation> anim = anim_tree->animation_node_get_animation(p_node);
			String text;
			if (anim_tree->animation_node_get_master_animation(p_node) != "") {
				text = anim_tree->animation_node_get_master_animation(p_node);
			} else if (anim.is_null()) {
				text = "load...";
			} else {
				text = anim->get_name();
			}

			font->draw_halign(ci, ofs + ascofs, HALIGN_CENTER, w, text, font_color_title);

		} break;
		case AnimationTreePlayer::NODE_ONESHOT:
		case AnimationTreePlayer::NODE_MIX:
		case AnimationTreePlayer::NODE_BLEND2:
		case AnimationTreePlayer::NODE_BLEND3:
		case AnimationTreePlayer::NODE_BLEND4:
		case AnimationTreePlayer::NODE_TIMESCALE:
		case AnimationTreePlayer::NODE_TRANSITION: {
			font->draw_halign(ci, ofs + ascofs, HALIGN_CENTER, w, "edit...", font_color_title);
		} break;
		default:
			editable = false;
	}

	if (editable) {
		Ref<Texture> arrow = get_icon("GuiDropdown", "EditorIcons");
		Point2 arrow_ofs(w - arrow->get_width(), Math::floor((h - arrow->get_height()) / 2));
		arrow->draw(ci, ofs + arrow_ofs);
	}
}

AnimationTreePlayerEditor::ClickType AnimationTreePlayerEditor::_locate_click(const Point2 &p_click, StringName *p_node_id, int *p_slot_index) const {
	Ref<StyleBox> style = get_stylebox("panel", "PopupMenu");
	Ref<Font> font = get_font("font", "PopupMenu");

	float h = (font->get_height() + get_constant("vseparation", "PopupMenu"));

	for (const List<StringName>::Element *E = order.back(); E; E = E->prev()) {
		const StringName &node = E->get();

		AnimationTreePlayer::NodeType type = anim_tree->node_get_type(node);

		Point2 pos = anim_tree->node_get_position(node);
		Size2 size = get_node_size(node);

		pos -= Point2(h_scroll->get_value(), v_scroll->get_value());

		if (!Rect2(pos, size).has_point(p_click)) {
			continue;
		}

		if (p_node_id) {
			*p_node_id = node;
		}

		pos = p_click - pos;

		float y = pos.y - style->get_offset().height;

		if (y < 2 * h) {
			return CLICK_NODE;
		}
		y -= 2 * h;

		int inputs = anim_tree->node_get_input_count(node);
		int count = MAX(inputs, 1);

		if (inputs == 0 || (pos.x > size.width / 2 && type != AnimationTreePlayer::NODE_OUTPUT)) {
			if (y < count * h) {
				if (p_slot_index) {
					*p_slot_index = 0;
				}
				return CLICK_OUTPUT_SLOT;
			}
		}

		for (int i = 0; i < count; i++) {
			if (y < h) {
				if (p_slot_index) {
					*p_slot_index = i;
				}
				return CLICK_INPUT_SLOT;
			}
			y -= h;
		}

		bool has_parameters = type != AnimationTreePlayer::NODE_OUTPUT && type != AnimationTreePlayer::NODE_TIMESEEK;
		return has_parameters ? CLICK_PARAMETER : CLICK_NODE;
	}

	return CLICK_NONE;
}

Point2 AnimationTreePlayerEditor::_get_slot_pos(const StringName &p_node_id, bool p_input, int p_slot) {
	Ref<StyleBox> style = get_stylebox("panel", "PopupMenu");
	Ref<Font> font = get_font("font", "PopupMenu");
	Ref<Texture> slot_icon = get_icon("VisualShaderPort", "EditorIcons");

	Size2 size = get_node_size(p_node_id);
	Point2 pos = anim_tree->node_get_position(p_node_id);

	if (click_type == CLICK_NODE && click_node == p_node_id) {
		pos += click_motion - click_pos;
		if (pos.x < 5) {
			pos.x = 5;
		}
		if (pos.y < 5) {
			pos.y = 5;
		}
	}

	pos -= Point2(h_scroll->get_value(), v_scroll->get_value());

	float w = size.width - style->get_minimum_size().width;
	float h = font->get_height() + get_constant("vseparation", "PopupMenu");

	pos += style->get_offset();

	pos.y += h * 2;

	pos.y += h * p_slot;

	pos += Point2(-slot_icon->get_width() / 2.0, h / 2.0).floor();

	if (!p_input) {
		pos.x += w + slot_icon->get_width();
	}

	return pos;
}

void AnimationTreePlayerEditor::_gui_input(Ref<InputEvent> p_event) {
	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->is_pressed()) {
			if (mb->get_button_index() == 1) {
				click_pos = Point2(mb->get_position().x, mb->get_position().y);
				click_motion = click_pos;
				click_type = _locate_click(click_pos, &click_node, &click_slot);
				if (click_type != CLICK_NONE) {
					order.erase(click_node);
					order.push_back(click_node);
					update();
				}

				switch (click_type) {
					case CLICK_INPUT_SLOT: {
						click_pos = _get_slot_pos(click_node, true, click_slot);
					} break;
					case CLICK_OUTPUT_SLOT: {
						click_pos = _get_slot_pos(click_node, false, click_slot);
					} break;
					case CLICK_PARAMETER: {
						edited_node = click_node;
						renaming_edit = false;
						_popup_edit_dialog();
						//open editor
						//_node_edit_property(click_node);
					} break;
					default: {
					}
				}
			}
			if (mb->get_button_index() == 2) {
				if (click_type != CLICK_NONE) {
					click_type = CLICK_NONE;
					update();
				} else {
					// try to disconnect/remove

					Point2 rclick_pos = Point2(mb->get_position().x, mb->get_position().y);
					rclick_type = _locate_click(rclick_pos, &rclick_node, &rclick_slot);
					if (rclick_type == CLICK_INPUT_SLOT || rclick_type == CLICK_OUTPUT_SLOT) {
						node_popup->clear();
						node_popup->set_size(Size2(1, 1));
						node_popup->add_item(TTR("Disconnect"), NODE_DISCONNECT);
						if (anim_tree->node_get_type(rclick_node) == AnimationTreePlayer::NODE_TRANSITION) {
							node_popup->add_item(TTR("Add Input"), NODE_ADD_INPUT);
							if (rclick_type == CLICK_INPUT_SLOT) {
								if (anim_tree->transition_node_has_input_auto_advance(rclick_node, rclick_slot)) {
									node_popup->add_item(TTR("Clear Auto-Advance"), NODE_CLEAR_AUTOADVANCE);
								} else {
									node_popup->add_item(TTR("Set Auto-Advance"), NODE_SET_AUTOADVANCE);
								}
								node_popup->add_item(TTR("Delete Input"), NODE_DELETE_INPUT);
							}
						}

						node_popup->set_position(rclick_pos + get_global_position());
						node_popup->popup();
					}

					if (rclick_type == CLICK_NODE) {
						node_popup->clear();
						node_popup->set_size(Size2(1, 1));
						node_popup->add_item(TTR("Rename"), NODE_RENAME);
						node_popup->add_item(TTR("Remove"), NODE_ERASE);
						if (anim_tree->node_get_type(rclick_node) == AnimationTreePlayer::NODE_TRANSITION) {
							node_popup->add_item(TTR("Add Input"), NODE_ADD_INPUT);
						}
						node_popup->set_position(rclick_pos + get_global_position());
						node_popup->popup();
					}
				}
			}
		} else {
			if (mb->get_button_index() == 1 && click_type != CLICK_NONE) {
				switch (click_type) {
					case CLICK_INPUT_SLOT:
					case CLICK_OUTPUT_SLOT: {
						Point2 dst_click_pos = Point2(mb->get_position().x, mb->get_position().y);
						StringName id;
						int slot;
						ClickType dst_click_type = _locate_click(dst_click_pos, &id, &slot);

						if (dst_click_type == CLICK_INPUT_SLOT && click_type == CLICK_OUTPUT_SLOT) {
							anim_tree->connect_nodes(click_node, id, slot);
						}
						if (click_type == CLICK_INPUT_SLOT && dst_click_type == CLICK_OUTPUT_SLOT) {
							anim_tree->connect_nodes(id, click_node, click_slot);
						}

					} break;
					case CLICK_NODE: {
						Point2 new_pos = anim_tree->node_get_position(click_node) + (click_motion - click_pos);
						if (new_pos.x < 5) {
							new_pos.x = 5;
						}
						if (new_pos.y < 5) {
							new_pos.y = 5;
						}
						anim_tree->node_set_position(click_node, new_pos);

					} break;
					default: {
					}
				}

				click_type = CLICK_NONE;
				update();
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (mm->get_button_mask() & 1 && click_type != CLICK_NONE) {
			click_motion = Point2(mm->get_position().x, mm->get_position().y);
			update();
		}
		if (mm->get_button_mask() & 4 || Input::get_singleton()->is_key_pressed(KEY_SPACE)) {
			h_scroll->set_value(h_scroll->get_value() - mm->get_relative().x);
			v_scroll->set_value(v_scroll->get_value() - mm->get_relative().y);
			update();
		}
	}
}

void AnimationTreePlayerEditor::_draw_cos_line(const Vector2 &p_from, const Vector2 &p_to, const Color &p_color) {
	static const int steps = 20;

	Rect2 r;
	r.position = p_from;
	r.expand_to(p_to);
	Vector2 sign = Vector2((p_from.x < p_to.x) ? 1 : -1, (p_from.y < p_to.y) ? 1 : -1);
	bool flip = sign.x * sign.y < 0;

	Vector2 prev;
	for (int i = 0; i <= steps; i++) {
		float d = i / float(steps);
		float c = -Math::cos(d * Math_PI) * 0.5 + 0.5;
		if (flip) {
			c = 1.0 - c;
		}
		Vector2 p = r.position + Vector2(d * r.size.width, c * r.size.height);

		if (i > 0) {
			draw_line(prev, p, p_color, 2);
		}

		prev = p;
	}
}

void AnimationTreePlayerEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			play_button->set_icon(get_icon("Play", "EditorIcons"));
			add_menu->set_icon(get_icon("Add", "EditorIcons"));
		} break;
		case NOTIFICATION_DRAW: {
			_update_scrollbars();
			//VisualServer::get_singleton()->canvas_item_add_rect(get_canvas_item(),Rect2(Point2(),get_size()),Color(0,0,0,1));
			get_stylebox("bg", "Tree")->draw(get_canvas_item(), Rect2(Point2(), get_size()));

			for (List<StringName>::Element *E = order.front(); E; E = E->next()) {
				_draw_node(E->get());
			}

			if (click_type == CLICK_INPUT_SLOT || click_type == CLICK_OUTPUT_SLOT) {
				_draw_cos_line(click_pos, click_motion, Color(0.5, 1, 0.5, 0.8));
			}

			List<AnimationTreePlayer::Connection> connections;
			anim_tree->get_connection_list(&connections);

			for (List<AnimationTreePlayer::Connection>::Element *E = connections.front(); E; E = E->next()) {
				const AnimationTreePlayer::Connection &c = E->get();
				Point2 source = _get_slot_pos(c.src_node, false, 0);
				Point2 dest = _get_slot_pos(c.dst_node, true, c.dst_input);
				Color col = Color(1, 1, 0.5, 0.8);
				/*
				if (click_type==CLICK_NODE && click_node==c.src_node) {

					source+=click_motion-click_pos;
				}

				if (click_type==CLICK_NODE && click_node==c.dst_node) {

					dest+=click_motion-click_pos;
				}*/

				_draw_cos_line(source, dest, col);
			}

			const Ref<Font> f = get_font("font", "Label");
			const Point2 status_offset = Point2(5, 25) * EDSCALE + Point2(0, f->get_ascent());

			switch (anim_tree->get_last_error()) {
				case AnimationTreePlayer::CONNECT_OK: {
					f->draw(get_canvas_item(), status_offset, TTR("Animation tree is valid."), Color(0, 1, 0.6, 0.8));
				} break;
				default: {
					f->draw(get_canvas_item(), status_offset, TTR("Animation tree is invalid."), Color(1, 0.6, 0.0, 0.8));
				} break;
			}

		} break;
	}
}

void AnimationTreePlayerEditor::_update_scrollbars() {
	Size2 size = get_size();
	Size2 hmin = h_scroll->get_combined_minimum_size();
	Size2 vmin = v_scroll->get_combined_minimum_size();

	v_scroll->set_begin(Point2(size.width - vmin.width, 0));
	v_scroll->set_end(Point2(size.width, size.height));

	h_scroll->set_begin(Point2(0, size.height - hmin.height));
	h_scroll->set_end(Point2(size.width - vmin.width, size.height));

	Size2 min = _get_maximum_size();

	if (min.height < size.height - hmin.height) {
		v_scroll->hide();
		offset.y = 0;
	} else {
		v_scroll->show();
		v_scroll->set_max(min.height);
		v_scroll->set_page(size.height - hmin.height);
		offset.y = v_scroll->get_value();
	}

	if (min.width < size.width - vmin.width) {
		h_scroll->hide();
		offset.x = 0;
	} else {
		h_scroll->show();
		h_scroll->set_max(min.width);
		h_scroll->set_page(size.width - vmin.width);
		offset.x = h_scroll->get_value();
	}
}

void AnimationTreePlayerEditor::_scroll_moved(float) {
	offset.x = h_scroll->get_value();
	offset.y = v_scroll->get_value();
	update();
}

void AnimationTreePlayerEditor::_node_menu_item(int p_item) {
	switch (p_item) {
		case NODE_DISCONNECT: {
			if (rclick_type == CLICK_INPUT_SLOT) {
				anim_tree->disconnect_nodes(rclick_node, rclick_slot);
				update();
			}

			if (rclick_type == CLICK_OUTPUT_SLOT) {
				List<AnimationTreePlayer::Connection> connections;
				anim_tree->get_connection_list(&connections);

				for (List<AnimationTreePlayer::Connection>::Element *E = connections.front(); E; E = E->next()) {
					const AnimationTreePlayer::Connection &c = E->get();
					if (c.dst_node == rclick_node) {
						anim_tree->disconnect_nodes(c.dst_node, c.dst_input);
					}
				}
				update();
			}

		} break;
		case NODE_RENAME: {
			renaming_edit = true;
			edited_node = rclick_node;
			_popup_edit_dialog();

		} break;
		case NODE_ADD_INPUT: {
			anim_tree->transition_node_set_input_count(rclick_node, anim_tree->transition_node_get_input_count(rclick_node) + 1);
			update();
		} break;
		case NODE_DELETE_INPUT: {
			anim_tree->transition_node_delete_input(rclick_node, rclick_slot);
			update();
		} break;
		case NODE_SET_AUTOADVANCE: {
			anim_tree->transition_node_set_input_auto_advance(rclick_node, rclick_slot, true);
			update();

		} break;
		case NODE_CLEAR_AUTOADVANCE: {
			anim_tree->transition_node_set_input_auto_advance(rclick_node, rclick_slot, false);
			update();

		} break;

		case NODE_ERASE: {
			if (rclick_node == "out") {
				break;
			}
			order.erase(rclick_node);
			anim_tree->remove_node(rclick_node);
			update();
		} break;
	}
}

StringName AnimationTreePlayerEditor::_add_node(int p_item) {
	static const char *bname[] = {
		"out",
		"anim",
		"oneshot",
		"mix",
		"blend2",
		"blend3",
		"blend4",
		"scale",
		"seek",
		"transition"
	};

	String name;
	int idx = 1;

	while (true) {
		name = bname[p_item];
		if (idx > 1) {
			name += " " + itos(idx);
		}
		if (anim_tree->node_exists(name)) {
			idx++;
		} else {
			break;
		}
	}

	anim_tree->add_node((AnimationTreePlayer::NodeType)p_item, name);
	anim_tree->node_set_position(name, Point2(last_x, last_y));
	order.push_back(name);
	last_x += 10;
	last_y += 10;
	last_x = last_x % (int)get_size().width;
	last_y = last_y % (int)get_size().height;
	update();

	return name;
};

void AnimationTreePlayerEditor::_file_dialog_selected(String p_path) {
	switch (file_op) {
		case MENU_IMPORT_ANIMATIONS: {
			Vector<String> files = file_dialog->get_selected_files();

			for (int i = 0; i < files.size(); i++) {
				StringName node = _add_node(AnimationTreePlayer::NODE_ANIMATION);

				RES anim = ResourceLoader::load(files[i]);
				anim_tree->animation_node_set_animation(node, anim);
				//anim_tree->node_set_name(node, files[i].get_file());
			};
		} break;

		default:
			break;
	};
};

void AnimationTreePlayerEditor::_add_menu_item(int p_item) {
	if (p_item == MENU_GRAPH_CLEAR) {
		//clear
	} else if (p_item == MENU_IMPORT_ANIMATIONS) {
		file_op = MENU_IMPORT_ANIMATIONS;
		file_dialog->set_mode(EditorFileDialog::MODE_OPEN_FILE);
		file_dialog->popup_centered_ratio();

	} else {
		_add_node(p_item);
	}
}

Size2 AnimationTreePlayerEditor::get_minimum_size() const {
	return Size2(10, 200);
}

void AnimationTreePlayerEditor::_find_paths_for_filter(const StringName &p_node, Set<String> &paths) {
	ERR_FAIL_COND(!anim_tree->node_exists(p_node));

	for (int i = 0; i < anim_tree->node_get_input_count(p_node); i++) {
		StringName port = anim_tree->node_get_input_source(p_node, i);
		if (port == StringName()) {
			continue;
		}
		_find_paths_for_filter(port, paths);
	}

	if (anim_tree->node_get_type(p_node) == AnimationTreePlayer::NODE_ANIMATION) {
		Ref<Animation> anim = anim_tree->animation_node_get_animation(p_node);
		if (anim.is_valid()) {
			for (int i = 0; i < anim->get_track_count(); i++) {
				paths.insert(anim->track_get_path(i));
			}
		}
	}
}

void AnimationTreePlayerEditor::_filter_edited() {
	TreeItem *ed = filter->get_edited();
	if (!ed) {
		return;
	}

	if (anim_tree->node_get_type(edited_node) == AnimationTreePlayer::NODE_ONESHOT) {
		anim_tree->oneshot_node_set_filter_path(edited_node, ed->get_metadata(0), ed->is_checked(0));
	} else if (anim_tree->node_get_type(edited_node) == AnimationTreePlayer::NODE_BLEND2) {
		anim_tree->blend2_node_set_filter_path(edited_node, ed->get_metadata(0), ed->is_checked(0));
	} else if (anim_tree->node_get_type(edited_node) == AnimationTreePlayer::NODE_ANIMATION) {
		anim_tree->animation_node_set_filter_path(edited_node, ed->get_metadata(0), ed->is_checked(0));
	}
}

void AnimationTreePlayerEditor::_edit_filters() {
	filter_dialog->popup_centered_ratio();
	filter->clear();

	Set<String> npb;
	_find_paths_for_filter(edited_node, npb);

	TreeItem *root = filter->create_item();
	filter->set_hide_root(true);
	Map<String, TreeItem *> pm;

	Node *base = anim_tree->get_node(anim_tree->get_base_path());

	for (Set<String>::Element *E = npb.front(); E; E = E->next()) {
		TreeItem *parent = root;
		String descr = E->get();
		if (base) {
			NodePath np = E->get();

			if (np.get_subname_count() == 1) {
				Node *n = base->get_node(np);
				Skeleton *s = Object::cast_to<Skeleton>(n);
				if (s) {
					String skelbase = E->get().substr(0, E->get().find(":"));

					int bidx = s->find_bone(np.get_subname(0));

					if (bidx != -1) {
						int bparent = s->get_bone_parent(bidx);
						//
						if (bparent != -1) {
							String bpn = skelbase + ":" + s->get_bone_name(bparent);
							if (pm.has(bpn)) {
								parent = pm[bpn];
								descr = np.get_subname(0);
							}
						} else {
							if (pm.has(skelbase)) {
								parent = pm[skelbase];
							}
						}
					}
				}
			}
		}

		TreeItem *it = filter->create_item(parent);
		it->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		it->set_text(0, descr);
		it->set_metadata(0, NodePath(E->get()));
		it->set_editable(0, true);
		if (anim_tree->node_get_type(edited_node) == AnimationTreePlayer::NODE_ONESHOT) {
			it->set_checked(0, anim_tree->oneshot_node_is_path_filtered(edited_node, E->get()));
		} else if (anim_tree->node_get_type(edited_node) == AnimationTreePlayer::NODE_BLEND2) {
			it->set_checked(0, anim_tree->blend2_node_is_path_filtered(edited_node, E->get()));
		} else if (anim_tree->node_get_type(edited_node) == AnimationTreePlayer::NODE_ANIMATION) {
			it->set_checked(0, anim_tree->animation_node_is_path_filtered(edited_node, E->get()));
		}
		pm[E->get()] = it;
	}
}

void AnimationTreePlayerEditor::_bind_methods() {
	ClassDB::bind_method("_add_menu_item", &AnimationTreePlayerEditor::_add_menu_item);
	ClassDB::bind_method("_node_menu_item", &AnimationTreePlayerEditor::_node_menu_item);
	ClassDB::bind_method("_gui_input", &AnimationTreePlayerEditor::_gui_input);
	//ClassDB::bind_method( "_node_param_changed", &AnimationTreeEditor::_node_param_changed );
	ClassDB::bind_method("_scroll_moved", &AnimationTreePlayerEditor::_scroll_moved);
	ClassDB::bind_method("_edit_dialog_changeds", &AnimationTreePlayerEditor::_edit_dialog_changeds);
	ClassDB::bind_method("_edit_dialog_changede", &AnimationTreePlayerEditor::_edit_dialog_changede);
	ClassDB::bind_method("_edit_dialog_changedf", &AnimationTreePlayerEditor::_edit_dialog_changedf);
	ClassDB::bind_method("_edit_dialog_changed", &AnimationTreePlayerEditor::_edit_dialog_changed);
	ClassDB::bind_method("_edit_dialog_animation_changed", &AnimationTreePlayerEditor::_edit_dialog_animation_changed);
	ClassDB::bind_method("_edit_dialog_edit_animation", &AnimationTreePlayerEditor::_edit_dialog_edit_animation);
	ClassDB::bind_method("_play_toggled", &AnimationTreePlayerEditor::_play_toggled);
	ClassDB::bind_method("_edit_oneshot_start", &AnimationTreePlayerEditor::_edit_oneshot_start);
	ClassDB::bind_method("_file_dialog_selected", &AnimationTreePlayerEditor::_file_dialog_selected);
	ClassDB::bind_method("_master_anim_menu_item", &AnimationTreePlayerEditor::_master_anim_menu_item);
	ClassDB::bind_method("_edit_filters", &AnimationTreePlayerEditor::_edit_filters);
	ClassDB::bind_method("_filter_edited", &AnimationTreePlayerEditor::_filter_edited);
}

AnimationTreePlayerEditor::AnimationTreePlayerEditor() {
	set_focus_mode(FOCUS_ALL);

	PopupMenu *p;
	List<PropertyInfo> defaults;

	add_menu = memnew(MenuButton);
	//add_menu->set_
	add_menu->set_position(Point2(0, 0));
	add_menu->set_size(Point2(25, 15));
	add_child(add_menu);

	p = add_menu->get_popup();
	p->add_item(TTR("Animation Node"), AnimationTreePlayer::NODE_ANIMATION);
	p->add_item(TTR("OneShot Node"), AnimationTreePlayer::NODE_ONESHOT);
	p->add_item(TTR("Mix Node"), AnimationTreePlayer::NODE_MIX);
	p->add_item(TTR("Blend2 Node"), AnimationTreePlayer::NODE_BLEND2);
	p->add_item(TTR("Blend3 Node"), AnimationTreePlayer::NODE_BLEND3);
	p->add_item(TTR("Blend4 Node"), AnimationTreePlayer::NODE_BLEND4);
	p->add_item(TTR("TimeScale Node"), AnimationTreePlayer::NODE_TIMESCALE);
	p->add_item(TTR("TimeSeek Node"), AnimationTreePlayer::NODE_TIMESEEK);
	p->add_item(TTR("Transition Node"), AnimationTreePlayer::NODE_TRANSITION);
	p->add_separator();
	p->add_item(TTR("Import Animations..."), MENU_IMPORT_ANIMATIONS); // wtf
	p->add_separator();
	p->add_item(TTR("Clear"), MENU_GRAPH_CLEAR);

	p->connect("id_pressed", this, "_add_menu_item");

	play_button = memnew(Button);
	play_button->set_position(Point2(25, 0) * EDSCALE);
	play_button->set_size(Point2(25, 15));
	add_child(play_button);
	play_button->set_toggle_mode(true);
	play_button->connect("pressed", this, "_play_toggled");

	last_x = 50;
	last_y = 50;

	property_editor = memnew(CustomPropertyEditor);
	add_child(property_editor);
	property_editor->connect("variant_changed", this, "_edit_dialog_animation_changed");
	property_editor->connect("resource_edit_request", this, "_edit_dialog_edit_animation");

	h_scroll = memnew(HScrollBar);
	v_scroll = memnew(VScrollBar);

	add_child(h_scroll);
	add_child(v_scroll);

	h_scroll->connect("value_changed", this, "_scroll_moved");
	v_scroll->connect("value_changed", this, "_scroll_moved");

	node_popup = memnew(PopupMenu);
	add_child(node_popup);
	node_popup->set_as_toplevel(true);

	master_anim_popup = memnew(PopupMenu);
	add_child(master_anim_popup);
	master_anim_popup->connect("id_pressed", this, "_master_anim_menu_item");

	node_popup->connect("id_pressed", this, "_node_menu_item");

	updating_edit = false;

	edit_dialog = memnew(PopupPanel);
	add_child(edit_dialog);

	VBoxContainer *vb = memnew(VBoxContainer);
	edit_dialog->add_child(vb);
	vb->set_anchors_preset(PRESET_WIDE);

	edit_option = memnew(OptionButton);
	vb->add_child(edit_option);
	edit_option->connect("item_selected", this, "_edit_dialog_changedf");
	edit_option->hide();

	for (int i = 0; i < 4; i++) {
		edit_label[i] = memnew(Label);
		vb->add_child(edit_label[i]);
		edit_label[i]->hide();

		edit_line[i] = memnew(LineEdit);
		vb->add_child(edit_line[i]);
		edit_line[i]->hide();
		edit_line[i]->connect("text_changed", this, "_edit_dialog_changeds");
		edit_line[i]->connect("text_entered", this, "_edit_dialog_changede");

		if (i < 2) {
			edit_scroll[i] = memnew(HSlider);
			vb->add_child(edit_scroll[i]);
			edit_scroll[i]->hide();
			edit_scroll[i]->connect("value_changed", this, "_edit_dialog_changedf");
		}
	}

	edit_button = memnew(Button);
	vb->add_child(edit_button);
	edit_button->hide();
	edit_button->connect("pressed", this, "_edit_oneshot_start");

	edit_check = memnew(CheckButton);
	vb->add_child(edit_check);
	edit_check->hide();
	edit_check->connect("pressed", this, "_edit_dialog_changed");

	file_dialog = memnew(EditorFileDialog);
	file_dialog->set_enable_multiple_selection(true);
	file_dialog->set_current_dir(ProjectSettings::get_singleton()->get_resource_path());
	add_child(file_dialog);
	file_dialog->connect("file_selected", this, "_file_dialog_selected");

	filter_dialog = memnew(AcceptDialog);
	filter_dialog->set_title(TTR("Edit Node Filters"));
	add_child(filter_dialog);

	filter = memnew(Tree);
	filter_dialog->add_child(filter);
	//filter_dialog->set_child_rect(filter);
	filter->connect("item_edited", this, "_filter_edited");

	filter_button = memnew(Button);
	filter_button->set_anchor(MARGIN_RIGHT, ANCHOR_END);
	filter_button->set_margin(MARGIN_RIGHT, -10);
	vb->add_child(filter_button);
	filter_button->hide();
	filter_button->set_text(TTR("Filters..."));
	filter_button->connect("pressed", this, "_edit_filters");

	set_clip_contents(true);
}

void AnimationTreePlayerEditorPlugin::edit(Object *p_object) {
	anim_tree_editor->edit(Object::cast_to<AnimationTreePlayer>(p_object));
}

bool AnimationTreePlayerEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("AnimationTreePlayer");
}

void AnimationTreePlayerEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		//editor->hide_animation_player_editors();
		//editor->animation_panel_make_visible(true);
		button->show();
		editor->make_bottom_panel_item_visible(anim_tree_editor);
		anim_tree_editor->set_physics_process(true);
	} else {
		if (anim_tree_editor->is_visible_in_tree()) {
			editor->hide_bottom_panel();
		}
		button->hide();
		anim_tree_editor->set_physics_process(false);
	}
}

AnimationTreePlayerEditorPlugin::AnimationTreePlayerEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	anim_tree_editor = memnew(AnimationTreePlayerEditor);
	anim_tree_editor->set_custom_minimum_size(Size2(0, 300) * EDSCALE);

	button = editor->add_bottom_panel_item(TTR("AnimationTree"), anim_tree_editor);
	button->hide();
}

AnimationTreePlayerEditorPlugin::~AnimationTreePlayerEditorPlugin() {
}
