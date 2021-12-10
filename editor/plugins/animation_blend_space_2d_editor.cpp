/*************************************************************************/
/*  animation_blend_space_2d_editor.cpp                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "animation_blend_space_2d_editor.h"

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/io/resource_loader.h"
#include "core/math/geometry_2d.h"
#include "core/os/keyboard.h"
#include "editor/editor_scale.h"
#include "scene/animation/animation_blend_tree.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/main/window.h"

bool AnimationNodeBlendSpace2DEditor::can_edit(const Ref<AnimationNode> &p_node) {
	Ref<AnimationNodeBlendSpace2D> bs2d = p_node;
	return bs2d.is_valid();
}

void AnimationNodeBlendSpace2DEditor::_blend_space_changed() {
	blend_space_draw->update();
}

void AnimationNodeBlendSpace2DEditor::edit(const Ref<AnimationNode> &p_node) {
	if (blend_space.is_valid()) {
		blend_space->disconnect("triangles_updated", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_blend_space_changed));
	}
	blend_space = p_node;

	if (!blend_space.is_null()) {
		blend_space->connect("triangles_updated", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_blend_space_changed));
		_update_space();
	}
}

StringName AnimationNodeBlendSpace2DEditor::get_blend_position_path() const {
	StringName path = AnimationTreeEditor::get_singleton()->get_base_path() + "blend_position";
	return path;
}

void AnimationNodeBlendSpace2DEditor::_blend_space_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;
	if (tool_select->is_pressed() && k.is_valid() && k->is_pressed() && k->get_keycode() == Key::KEY_DELETE && !k->is_echo()) {
		if (selected_point != -1 || selected_triangle != -1) {
			_erase_selected();
			accept_event();
		}
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed() && ((tool_select->is_pressed() && mb->get_button_index() == MouseButton::RIGHT) || (mb->get_button_index() == MouseButton::LEFT && tool_create->is_pressed()))) {
		menu->clear();
		animations_menu->clear();
		animations_to_add.clear();
		List<StringName> classes;
		classes.sort_custom<StringName::AlphCompare>();

		ClassDB::get_inheriters_from_class("AnimationRootNode", &classes);
		menu->add_submenu_item(TTR("Add Animation"), "animations");

		AnimationTree *gp = AnimationTreeEditor::get_singleton()->get_tree();
		ERR_FAIL_COND(!gp);
		if (gp && gp->has_node(gp->get_animation_player())) {
			AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(gp->get_node(gp->get_animation_player()));
			if (ap) {
				List<StringName> names;
				ap->get_animation_list(&names);
				for (const StringName &E : names) {
					animations_menu->add_icon_item(get_theme_icon(SNAME("Animation"), SNAME("EditorIcons")), E);
					animations_to_add.push_back(E);
				}
			}
		}

		for (const StringName &E : classes) {
			String name = String(E).replace_first("AnimationNode", "");
			if (name == "Animation") {
				continue; // nope
			}
			int idx = menu->get_item_count();
			menu->add_item(vformat("Add %s", name), idx);
			menu->set_item_metadata(idx, E);
		}

		Ref<AnimationNode> clipb = EditorSettings::get_singleton()->get_resource_clipboard();
		if (clipb.is_valid()) {
			menu->add_separator();
			menu->add_item(TTR("Paste"), MENU_PASTE);
		}
		menu->add_separator();
		menu->add_item(TTR("Load..."), MENU_LOAD_FILE);

		menu->set_position(blend_space_draw->get_screen_position() + mb->get_position());
		menu->reset_size();
		menu->popup();
		add_point_pos = (mb->get_position() / blend_space_draw->get_size());
		add_point_pos.y = 1.0 - add_point_pos.y;
		add_point_pos *= (blend_space->get_max_space() - blend_space->get_min_space());
		add_point_pos += blend_space->get_min_space();

		if (snap->is_pressed()) {
			add_point_pos = add_point_pos.snapped(blend_space->get_snap());
		}
	}

	if (mb.is_valid() && mb->is_pressed() && tool_select->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		blend_space_draw->update(); //update anyway
		//try to see if a point can be selected
		selected_point = -1;
		selected_triangle = -1;
		_update_tool_erase();

		for (int i = 0; i < points.size(); i++) {
			if (points[i].distance_to(mb->get_position()) < 10 * EDSCALE) {
				selected_point = i;
				Ref<AnimationNode> node = blend_space->get_blend_point_node(i);
				EditorNode::get_singleton()->push_item(node.ptr(), "", true);
				dragging_selected_attempt = true;
				drag_from = mb->get_position();
				_update_tool_erase();
				_update_edited_point_pos();
				return;
			}
		}

		//then try to see if a triangle can be selected
		if (!blend_space->get_auto_triangles()) { //if autotriangles use, disable this
			for (int i = 0; i < blend_space->get_triangle_count(); i++) {
				Vector<Vector2> triangle;

				for (int j = 0; j < 3; j++) {
					int idx = blend_space->get_triangle_point(i, j);
					ERR_FAIL_INDEX(idx, points.size());
					triangle.push_back(points[idx]);
				}

				if (Geometry2D::is_point_in_triangle(mb->get_position(), triangle[0], triangle[1], triangle[2])) {
					selected_triangle = i;
					_update_tool_erase();
					return;
				}
			}
		}
	}

	if (mb.is_valid() && mb->is_pressed() && tool_triangle->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		blend_space_draw->update(); //update anyway
		//try to see if a point can be selected
		selected_point = -1;

		for (int i = 0; i < points.size(); i++) {
			if (making_triangle.find(i) != -1) {
				continue;
			}

			if (points[i].distance_to(mb->get_position()) < 10 * EDSCALE) {
				making_triangle.push_back(i);
				if (making_triangle.size() == 3) {
					//add triangle!
					if (blend_space->has_triangle(making_triangle[0], making_triangle[1], making_triangle[2])) {
						making_triangle.clear();
						EditorNode::get_singleton()->show_warning(TTR("Triangle already exists."));
						return;
					}

					updating = true;
					undo_redo->create_action(TTR("Add Triangle"));
					undo_redo->add_do_method(blend_space.ptr(), "add_triangle", making_triangle[0], making_triangle[1], making_triangle[2]);
					undo_redo->add_undo_method(blend_space.ptr(), "remove_triangle", blend_space->get_triangle_count());
					undo_redo->add_do_method(this, "_update_space");
					undo_redo->add_undo_method(this, "_update_space");
					undo_redo->commit_action();
					updating = false;
					making_triangle.clear();
				}
				return;
			}
		}
	}

	if (mb.is_valid() && !mb->is_pressed() && dragging_selected_attempt && mb->get_button_index() == MouseButton::LEFT) {
		if (dragging_selected) {
			//move
			Vector2 point = blend_space->get_blend_point_position(selected_point);
			point += drag_ofs;
			if (snap->is_pressed()) {
				point = point.snapped(blend_space->get_snap());
			}

			updating = true;
			undo_redo->create_action(TTR("Move Node Point"));
			undo_redo->add_do_method(blend_space.ptr(), "set_blend_point_position", selected_point, point);
			undo_redo->add_undo_method(blend_space.ptr(), "set_blend_point_position", selected_point, blend_space->get_blend_point_position(selected_point));
			undo_redo->add_do_method(this, "_update_space");
			undo_redo->add_undo_method(this, "_update_space");
			undo_redo->add_do_method(this, "_update_edited_point_pos");
			undo_redo->add_undo_method(this, "_update_edited_point_pos");
			undo_redo->commit_action();
			updating = false;
			_update_edited_point_pos();
		}
		dragging_selected_attempt = false;
		dragging_selected = false;
		blend_space_draw->update();
	}

	if (mb.is_valid() && mb->is_pressed() && tool_blend->is_pressed() && mb->get_button_index() == MouseButton::LEFT) {
		Vector2 blend_pos = (mb->get_position() / blend_space_draw->get_size());
		blend_pos.y = 1.0 - blend_pos.y;
		blend_pos *= (blend_space->get_max_space() - blend_space->get_min_space());
		blend_pos += blend_space->get_min_space();

		AnimationTreeEditor::get_singleton()->get_tree()->set(get_blend_position_path(), blend_pos);

		blend_space_draw->update();
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid() && !blend_space_draw->has_focus()) {
		blend_space_draw->grab_focus();
		blend_space_draw->update();
	}

	if (mm.is_valid() && dragging_selected_attempt) {
		dragging_selected = true;
		drag_ofs = ((mm->get_position() - drag_from) / blend_space_draw->get_size()) * (blend_space->get_max_space() - blend_space->get_min_space()) * Vector2(1, -1);
		blend_space_draw->update();
		_update_edited_point_pos();
	}

	if (mm.is_valid() && tool_triangle->is_pressed() && making_triangle.size()) {
		blend_space_draw->update();
	}

	if (mm.is_valid() && !tool_triangle->is_pressed() && making_triangle.size()) {
		making_triangle.clear();
		blend_space_draw->update();
	}

	if (mm.is_valid() && tool_blend->is_pressed() && (mm->get_button_mask() & MouseButton::MASK_LEFT) != MouseButton::NONE) {
		Vector2 blend_pos = (mm->get_position() / blend_space_draw->get_size());
		blend_pos.y = 1.0 - blend_pos.y;
		blend_pos *= (blend_space->get_max_space() - blend_space->get_min_space());
		blend_pos += blend_space->get_min_space();

		AnimationTreeEditor::get_singleton()->get_tree()->set(get_blend_position_path(), blend_pos);

		blend_space_draw->update();
	}
}

void AnimationNodeBlendSpace2DEditor::_file_opened(const String &p_file) {
	file_loaded = ResourceLoader::load(p_file);
	if (file_loaded.is_valid()) {
		_add_menu_type(MENU_LOAD_FILE_CONFIRM);
	}
}

void AnimationNodeBlendSpace2DEditor::_add_menu_type(int p_index) {
	Ref<AnimationRootNode> node;
	if (p_index == MENU_LOAD_FILE) {
		open_file->clear_filters();
		List<String> filters;
		ResourceLoader::get_recognized_extensions_for_type("AnimationRootNode", &filters);
		for (const String &E : filters) {
			open_file->add_filter("*." + E);
		}
		open_file->popup_file_dialog();
		return;
	} else if (p_index == MENU_LOAD_FILE_CONFIRM) {
		node = file_loaded;
		file_loaded.unref();
	} else if (p_index == MENU_PASTE) {
		node = EditorSettings::get_singleton()->get_resource_clipboard();
	} else {
		String type = menu->get_item_metadata(p_index);

		Object *obj = ClassDB::instantiate(type);
		ERR_FAIL_COND(!obj);
		AnimationNode *an = Object::cast_to<AnimationNode>(obj);
		ERR_FAIL_COND(!an);

		node = Ref<AnimationNode>(an);
	}

	if (!node.is_valid()) {
		EditorNode::get_singleton()->show_warning(TTR("This type of node can't be used. Only root nodes are allowed."));
		return;
	}

	updating = true;
	undo_redo->create_action(TTR("Add Node Point"));
	undo_redo->add_do_method(blend_space.ptr(), "add_blend_point", node, add_point_pos);
	undo_redo->add_undo_method(blend_space.ptr(), "remove_blend_point", blend_space->get_blend_point_count());
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->commit_action();
	updating = false;

	blend_space_draw->update();
}

void AnimationNodeBlendSpace2DEditor::_add_animation_type(int p_index) {
	Ref<AnimationNodeAnimation> anim;
	anim.instantiate();

	anim->set_animation(animations_to_add[p_index]);

	updating = true;
	undo_redo->create_action(TTR("Add Animation Point"));
	undo_redo->add_do_method(blend_space.ptr(), "add_blend_point", anim, add_point_pos);
	undo_redo->add_undo_method(blend_space.ptr(), "remove_blend_point", blend_space->get_blend_point_count());
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->commit_action();
	updating = false;

	blend_space_draw->update();
}

void AnimationNodeBlendSpace2DEditor::_update_tool_erase() {
	tool_erase->set_disabled(!(selected_point >= 0 && selected_point < blend_space->get_blend_point_count()) && !(selected_triangle >= 0 && selected_triangle < blend_space->get_triangle_count()));
	if (selected_point >= 0 && selected_point < blend_space->get_blend_point_count()) {
		Ref<AnimationNode> an = blend_space->get_blend_point_node(selected_point);
		if (AnimationTreeEditor::get_singleton()->can_edit(an)) {
			open_editor->show();
		} else {
			open_editor->hide();
		}
		edit_hb->show();
	} else {
		edit_hb->hide();
	}
}

void AnimationNodeBlendSpace2DEditor::_tool_switch(int p_tool) {
	making_triangle.clear();

	if (p_tool == 2) {
		Vector<Vector2> points;
		for (int i = 0; i < blend_space->get_blend_point_count(); i++) {
			points.push_back(blend_space->get_blend_point_position(i));
		}
		Vector<Delaunay2D::Triangle> tr = Delaunay2D::triangulate(points);
		for (int i = 0; i < tr.size(); i++) {
			blend_space->add_triangle(tr[i].points[0], tr[i].points[1], tr[i].points[2]);
		}
	}

	if (p_tool == 0) {
		tool_erase->show();
		tool_erase_sep->show();
	} else {
		tool_erase->hide();
		tool_erase_sep->hide();
	}
	_update_tool_erase();
	blend_space_draw->update();
}

void AnimationNodeBlendSpace2DEditor::_blend_space_draw() {
	Color linecolor = get_theme_color(SNAME("font_color"), SNAME("Label"));
	Color linecolor_soft = linecolor;
	linecolor_soft.a *= 0.5;
	Ref<Font> font = get_theme_font(SNAME("font"), SNAME("Label"));
	int font_size = get_theme_font_size(SNAME("font_size"), SNAME("Label"));
	Ref<Texture2D> icon = get_theme_icon(SNAME("KeyValue"), SNAME("EditorIcons"));
	Ref<Texture2D> icon_selected = get_theme_icon(SNAME("KeySelected"), SNAME("EditorIcons"));

	Size2 s = blend_space_draw->get_size();

	if (blend_space_draw->has_focus()) {
		Color color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
		blend_space_draw->draw_rect(Rect2(Point2(), s), color, false);
	}
	blend_space_draw->draw_line(Point2(1, 0), Point2(1, s.height - 1), linecolor);
	blend_space_draw->draw_line(Point2(1, s.height - 1), Point2(s.width - 1, s.height - 1), linecolor);

	blend_space_draw->draw_line(Point2(0, 0), Point2(5 * EDSCALE, 0), linecolor);
	if (blend_space->get_min_space().y < 0) {
		int y = (blend_space->get_max_space().y / (blend_space->get_max_space().y - blend_space->get_min_space().y)) * s.height;
		blend_space_draw->draw_line(Point2(0, y), Point2(5 * EDSCALE, y), linecolor);
		blend_space_draw->draw_string(font, Point2(2 * EDSCALE, y - font->get_height(font_size) + font->get_ascent(font_size)), "0", HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, linecolor);
		blend_space_draw->draw_line(Point2(5 * EDSCALE, y), Point2(s.width, y), linecolor_soft);
	}

	if (blend_space->get_min_space().x < 0) {
		int x = (-blend_space->get_min_space().x / (blend_space->get_max_space().x - blend_space->get_min_space().x)) * s.width;
		blend_space_draw->draw_line(Point2(x, s.height - 1), Point2(x, s.height - 5 * EDSCALE), linecolor);
		blend_space_draw->draw_string(font, Point2(x + 2 * EDSCALE, s.height - 2 * EDSCALE - font->get_height(font_size) + font->get_ascent(font_size)), "0", HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, linecolor);
		blend_space_draw->draw_line(Point2(x, s.height - 5 * EDSCALE), Point2(x, 0), linecolor_soft);
	}

	if (snap->is_pressed()) {
		linecolor_soft.a = linecolor.a * 0.1;

		if (blend_space->get_snap().x > 0) {
			int prev_idx = 0;
			for (int i = 0; i < s.x; i++) {
				float v = blend_space->get_min_space().x + i * (blend_space->get_max_space().x - blend_space->get_min_space().x) / s.x;
				int idx = int(v / blend_space->get_snap().x);

				if (i > 0 && prev_idx != idx) {
					blend_space_draw->draw_line(Point2(i, 0), Point2(i, s.height), linecolor_soft);
				}

				prev_idx = idx;
			}
		}

		if (blend_space->get_snap().y > 0) {
			int prev_idx = 0;
			for (int i = 0; i < s.y; i++) {
				float v = blend_space->get_max_space().y - i * (blend_space->get_max_space().y - blend_space->get_min_space().y) / s.y;
				int idx = int(v / blend_space->get_snap().y);

				if (i > 0 && prev_idx != idx) {
					blend_space_draw->draw_line(Point2(0, i), Point2(s.width, i), linecolor_soft);
				}

				prev_idx = idx;
			}
		}
	}

	//triangles first
	for (int i = 0; i < blend_space->get_triangle_count(); i++) {
		Vector<Vector2> points;
		points.resize(3);

		for (int j = 0; j < 3; j++) {
			int point_idx = blend_space->get_triangle_point(i, j);
			Vector2 point = blend_space->get_blend_point_position(point_idx);
			if (dragging_selected && selected_point == point_idx) {
				point += drag_ofs;
				if (snap->is_pressed()) {
					point = point.snapped(blend_space->get_snap());
				}
			}
			point = (point - blend_space->get_min_space()) / (blend_space->get_max_space() - blend_space->get_min_space());
			point *= s;
			point.y = s.height - point.y;
			points.write[j] = point;
		}

		for (int j = 0; j < 3; j++) {
			blend_space_draw->draw_line(points[j], points[(j + 1) % 3], linecolor, 1);
		}

		Color color;
		if (i == selected_triangle) {
			color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
			color.a *= 0.5;
		} else {
			color = linecolor;
			color.a *= 0.2;
		}

		Vector<Color> colors;
		colors.push_back(color);
		colors.push_back(color);
		colors.push_back(color);
		blend_space_draw->draw_primitive(points, colors, Vector<Vector2>());
	}

	points.clear();
	for (int i = 0; i < blend_space->get_blend_point_count(); i++) {
		Vector2 point = blend_space->get_blend_point_position(i);
		if (dragging_selected && selected_point == i) {
			point += drag_ofs;
			if (snap->is_pressed()) {
				point = point.snapped(blend_space->get_snap());
			}
		}
		point = (point - blend_space->get_min_space()) / (blend_space->get_max_space() - blend_space->get_min_space());
		point *= s;
		point.y = s.height - point.y;

		points.push_back(point);
		point -= (icon->get_size() / 2);
		point = point.floor();

		if (i == selected_point) {
			blend_space_draw->draw_texture(icon_selected, point);
		} else {
			blend_space_draw->draw_texture(icon, point);
		}
	}

	if (making_triangle.size()) {
		Vector<Vector2> points;
		for (int i = 0; i < making_triangle.size(); i++) {
			Vector2 point = blend_space->get_blend_point_position(making_triangle[i]);
			point = (point - blend_space->get_min_space()) / (blend_space->get_max_space() - blend_space->get_min_space());
			point *= s;
			point.y = s.height - point.y;
			points.push_back(point);
		}

		for (int i = 0; i < points.size() - 1; i++) {
			blend_space_draw->draw_line(points[i], points[i + 1], linecolor, 2);
		}
		blend_space_draw->draw_line(points[points.size() - 1], blend_space_draw->get_local_mouse_position(), linecolor, 2);
	}

	///draw cursor position

	{
		Color color;
		if (tool_blend->is_pressed()) {
			color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
		} else {
			color = linecolor;
			color.a *= 0.5;
		}

		Vector2 blend_pos = AnimationTreeEditor::get_singleton()->get_tree()->get(get_blend_position_path());
		Vector2 point = blend_pos;

		point = (point - blend_space->get_min_space()) / (blend_space->get_max_space() - blend_space->get_min_space());
		point *= s;
		point.y = s.height - point.y;

		if (blend_space->get_triangle_count()) {
			Vector2 closest = blend_space->get_closest_point(blend_pos);
			closest = (closest - blend_space->get_min_space()) / (blend_space->get_max_space() - blend_space->get_min_space());
			closest *= s;
			closest.y = s.height - closest.y;

			Color lcol = color;
			lcol.a *= 0.4;
			blend_space_draw->draw_line(point, closest, lcol, 2);
		}

		float mind = 5 * EDSCALE;
		float maxd = 15 * EDSCALE;
		blend_space_draw->draw_line(point + Vector2(mind, 0), point + Vector2(maxd, 0), color, 2);
		blend_space_draw->draw_line(point + Vector2(-mind, 0), point + Vector2(-maxd, 0), color, 2);
		blend_space_draw->draw_line(point + Vector2(0, mind), point + Vector2(0, maxd), color, 2);
		blend_space_draw->draw_line(point + Vector2(0, -mind), point + Vector2(0, -maxd), color, 2);
	}
}

void AnimationNodeBlendSpace2DEditor::_snap_toggled() {
	blend_space_draw->update();
}

void AnimationNodeBlendSpace2DEditor::_update_space() {
	if (updating) {
		return;
	}

	updating = true;

	if (blend_space->get_auto_triangles()) {
		tool_triangle->hide();
	} else {
		tool_triangle->show();
	}

	auto_triangles->set_pressed(blend_space->get_auto_triangles());

	interpolation->select(blend_space->get_blend_mode());

	max_x_value->set_value(blend_space->get_max_space().x);
	max_y_value->set_value(blend_space->get_max_space().y);

	min_x_value->set_value(blend_space->get_min_space().x);
	min_y_value->set_value(blend_space->get_min_space().y);

	label_x->set_text(blend_space->get_x_label());
	label_y->set_text(blend_space->get_y_label());

	snap_x->set_value(blend_space->get_snap().x);
	snap_y->set_value(blend_space->get_snap().y);

	blend_space_draw->update();

	updating = false;
}

void AnimationNodeBlendSpace2DEditor::_config_changed(double) {
	if (updating) {
		return;
	}

	updating = true;
	undo_redo->create_action(TTR("Change BlendSpace2D Limits"));
	undo_redo->add_do_method(blend_space.ptr(), "set_max_space", Vector2(max_x_value->get_value(), max_y_value->get_value()));
	undo_redo->add_undo_method(blend_space.ptr(), "set_max_space", blend_space->get_max_space());
	undo_redo->add_do_method(blend_space.ptr(), "set_min_space", Vector2(min_x_value->get_value(), min_y_value->get_value()));
	undo_redo->add_undo_method(blend_space.ptr(), "set_min_space", blend_space->get_min_space());
	undo_redo->add_do_method(blend_space.ptr(), "set_snap", Vector2(snap_x->get_value(), snap_y->get_value()));
	undo_redo->add_undo_method(blend_space.ptr(), "set_snap", blend_space->get_snap());
	undo_redo->add_do_method(blend_space.ptr(), "set_blend_mode", interpolation->get_selected());
	undo_redo->add_undo_method(blend_space.ptr(), "set_blend_mode", blend_space->get_blend_mode());
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->commit_action();
	updating = false;

	blend_space_draw->update();
}

void AnimationNodeBlendSpace2DEditor::_labels_changed(String) {
	if (updating) {
		return;
	}

	updating = true;
	undo_redo->create_action(TTR("Change BlendSpace2D Labels"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(blend_space.ptr(), "set_x_label", label_x->get_text());
	undo_redo->add_undo_method(blend_space.ptr(), "set_x_label", blend_space->get_x_label());
	undo_redo->add_do_method(blend_space.ptr(), "set_y_label", label_y->get_text());
	undo_redo->add_undo_method(blend_space.ptr(), "set_y_label", blend_space->get_y_label());
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->commit_action();
	updating = false;
}

void AnimationNodeBlendSpace2DEditor::_erase_selected() {
	if (selected_point != -1) {
		updating = true;
		undo_redo->create_action(TTR("Remove BlendSpace2D Point"));
		undo_redo->add_do_method(blend_space.ptr(), "remove_blend_point", selected_point);
		undo_redo->add_undo_method(blend_space.ptr(), "add_blend_point", blend_space->get_blend_point_node(selected_point), blend_space->get_blend_point_position(selected_point), selected_point);

		//restore triangles using this point
		for (int i = 0; i < blend_space->get_triangle_count(); i++) {
			for (int j = 0; j < 3; j++) {
				if (blend_space->get_triangle_point(i, j) == selected_point) {
					undo_redo->add_undo_method(blend_space.ptr(), "add_triangle", blend_space->get_triangle_point(i, 0), blend_space->get_triangle_point(i, 1), blend_space->get_triangle_point(i, 2), i);
					break;
				}
			}
		}

		undo_redo->add_do_method(this, "_update_space");
		undo_redo->add_undo_method(this, "_update_space");
		undo_redo->commit_action();
		updating = false;

		blend_space_draw->update();
	} else if (selected_triangle != -1) {
		updating = true;
		undo_redo->create_action(TTR("Remove BlendSpace2D Triangle"));
		undo_redo->add_do_method(blend_space.ptr(), "remove_triangle", selected_triangle);
		undo_redo->add_undo_method(blend_space.ptr(), "add_triangle", blend_space->get_triangle_point(selected_triangle, 0), blend_space->get_triangle_point(selected_triangle, 1), blend_space->get_triangle_point(selected_triangle, 2), selected_triangle);

		undo_redo->add_do_method(this, "_update_space");
		undo_redo->add_undo_method(this, "_update_space");
		undo_redo->commit_action();
		updating = false;

		blend_space_draw->update();
	}
}

void AnimationNodeBlendSpace2DEditor::_update_edited_point_pos() {
	if (updating) {
		return;
	}

	if (selected_point >= 0 && selected_point < blend_space->get_blend_point_count()) {
		Vector2 pos = blend_space->get_blend_point_position(selected_point);
		if (dragging_selected) {
			pos += drag_ofs;
			if (snap->is_pressed()) {
				pos = pos.snapped(blend_space->get_snap());
			}
		}
		updating = true;
		edit_x->set_value(pos.x);
		edit_y->set_value(pos.y);
		updating = false;
	}
}

void AnimationNodeBlendSpace2DEditor::_edit_point_pos(double) {
	if (updating) {
		return;
	}
	updating = true;
	undo_redo->create_action(TTR("Move Node Point"));
	undo_redo->add_do_method(blend_space.ptr(), "set_blend_point_position", selected_point, Vector2(edit_x->get_value(), edit_y->get_value()));
	undo_redo->add_undo_method(blend_space.ptr(), "set_blend_point_position", selected_point, blend_space->get_blend_point_position(selected_point));
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->add_do_method(this, "_update_edited_point_pos");
	undo_redo->add_undo_method(this, "_update_edited_point_pos");
	undo_redo->commit_action();
	updating = false;

	blend_space_draw->update();
}

void AnimationNodeBlendSpace2DEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		error_panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
		error_label->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
		panel->add_theme_style_override("panel", get_theme_stylebox(SNAME("bg"), SNAME("Tree")));
		tool_blend->set_icon(get_theme_icon(SNAME("EditPivot"), SNAME("EditorIcons")));
		tool_select->set_icon(get_theme_icon(SNAME("ToolSelect"), SNAME("EditorIcons")));
		tool_create->set_icon(get_theme_icon(SNAME("EditKey"), SNAME("EditorIcons")));
		tool_triangle->set_icon(get_theme_icon(SNAME("ToolTriangle"), SNAME("EditorIcons")));
		tool_erase->set_icon(get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
		snap->set_icon(get_theme_icon(SNAME("SnapGrid"), SNAME("EditorIcons")));
		open_editor->set_icon(get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")));
		auto_triangles->set_icon(get_theme_icon(SNAME("AutoTriangle"), SNAME("EditorIcons")));
		interpolation->clear();
		interpolation->add_icon_item(get_theme_icon(SNAME("TrackContinuous"), SNAME("EditorIcons")), "", 0);
		interpolation->add_icon_item(get_theme_icon(SNAME("TrackDiscrete"), SNAME("EditorIcons")), "", 1);
		interpolation->add_icon_item(get_theme_icon(SNAME("TrackCapture"), SNAME("EditorIcons")), "", 2);
	}

	if (p_what == NOTIFICATION_PROCESS) {
		String error;

		if (!AnimationTreeEditor::get_singleton()->get_tree()) {
			error = TTR("BlendSpace2D does not belong to an AnimationTree node.");
		} else if (!AnimationTreeEditor::get_singleton()->get_tree()->is_active()) {
			error = TTR("AnimationTree is inactive.\nActivate to enable playback, check node warnings if activation fails.");
		} else if (AnimationTreeEditor::get_singleton()->get_tree()->is_state_invalid()) {
			error = AnimationTreeEditor::get_singleton()->get_tree()->get_invalid_state_reason();
		} else if (blend_space->get_triangle_count() == 0) {
			error = TTR("No triangles exist, so no blending can take place.");
		}

		if (error != error_label->get_text()) {
			error_label->set_text(error);
			if (!error.is_empty()) {
				error_panel->show();
			} else {
				error_panel->hide();
			}
		}
	}

	if (p_what == NOTIFICATION_VISIBILITY_CHANGED) {
		set_process(is_visible_in_tree());
	}
}

void AnimationNodeBlendSpace2DEditor::_open_editor() {
	if (selected_point >= 0 && selected_point < blend_space->get_blend_point_count()) {
		Ref<AnimationNode> an = blend_space->get_blend_point_node(selected_point);
		ERR_FAIL_COND(an.is_null());
		AnimationTreeEditor::get_singleton()->enter_editor(itos(selected_point));
	}
}

void AnimationNodeBlendSpace2DEditor::_removed_from_graph() {
	EditorNode::get_singleton()->edit_item(nullptr);
}

void AnimationNodeBlendSpace2DEditor::_auto_triangles_toggled() {
	undo_redo->create_action(TTR("Toggle Auto Triangles"));
	undo_redo->add_do_method(blend_space.ptr(), "set_auto_triangles", auto_triangles->is_pressed());
	undo_redo->add_undo_method(blend_space.ptr(), "set_auto_triangles", blend_space->get_auto_triangles());
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->commit_action();
}

void AnimationNodeBlendSpace2DEditor::_bind_methods() {
	ClassDB::bind_method("_update_space", &AnimationNodeBlendSpace2DEditor::_update_space);
	ClassDB::bind_method("_update_tool_erase", &AnimationNodeBlendSpace2DEditor::_update_tool_erase);

	ClassDB::bind_method("_update_edited_point_pos", &AnimationNodeBlendSpace2DEditor::_update_edited_point_pos);

	ClassDB::bind_method("_removed_from_graph", &AnimationNodeBlendSpace2DEditor::_removed_from_graph);
}

AnimationNodeBlendSpace2DEditor *AnimationNodeBlendSpace2DEditor::singleton = nullptr;

AnimationNodeBlendSpace2DEditor::AnimationNodeBlendSpace2DEditor() {
	singleton = this;
	updating = false;

	HBoxContainer *top_hb = memnew(HBoxContainer);
	add_child(top_hb);

	Ref<ButtonGroup> bg;
	bg.instantiate();

	tool_blend = memnew(Button);
	tool_blend->set_flat(true);
	tool_blend->set_toggle_mode(true);
	tool_blend->set_button_group(bg);
	top_hb->add_child(tool_blend);
	tool_blend->set_pressed(true);
	tool_blend->set_tooltip(TTR("Set the blending position within the space"));
	tool_blend->connect("pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_tool_switch), varray(3));

	tool_select = memnew(Button);
	tool_select->set_flat(true);
	tool_select->set_toggle_mode(true);
	tool_select->set_button_group(bg);
	top_hb->add_child(tool_select);
	tool_select->set_tooltip(TTR("Select and move points, create points with RMB."));
	tool_select->connect("pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_tool_switch), varray(0));

	tool_create = memnew(Button);
	tool_create->set_flat(true);
	tool_create->set_toggle_mode(true);
	tool_create->set_button_group(bg);
	top_hb->add_child(tool_create);
	tool_create->set_tooltip(TTR("Create points."));
	tool_create->connect("pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_tool_switch), varray(1));

	tool_triangle = memnew(Button);
	tool_triangle->set_flat(true);
	tool_triangle->set_toggle_mode(true);
	tool_triangle->set_button_group(bg);
	top_hb->add_child(tool_triangle);
	tool_triangle->set_tooltip(TTR("Create triangles by connecting points."));
	tool_triangle->connect("pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_tool_switch), varray(2));

	tool_erase_sep = memnew(VSeparator);
	top_hb->add_child(tool_erase_sep);
	tool_erase = memnew(Button);
	tool_erase->set_flat(true);
	top_hb->add_child(tool_erase);
	tool_erase->set_tooltip(TTR("Erase points and triangles."));
	tool_erase->connect("pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_erase_selected));
	tool_erase->set_disabled(true);

	top_hb->add_child(memnew(VSeparator));

	auto_triangles = memnew(Button);
	auto_triangles->set_flat(true);
	top_hb->add_child(auto_triangles);
	auto_triangles->connect("pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_auto_triangles_toggled));
	auto_triangles->set_toggle_mode(true);
	auto_triangles->set_tooltip(TTR("Generate blend triangles automatically (instead of manually)"));

	top_hb->add_child(memnew(VSeparator));

	snap = memnew(Button);
	snap->set_flat(true);
	snap->set_toggle_mode(true);
	top_hb->add_child(snap);
	snap->set_pressed(true);
	snap->set_tooltip(TTR("Enable snap and show grid."));
	snap->connect("pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_snap_toggled));

	snap_x = memnew(SpinBox);
	top_hb->add_child(snap_x);
	snap_x->set_prefix("x:");
	snap_x->set_min(0.01);
	snap_x->set_step(0.01);
	snap_x->set_max(1000);

	snap_y = memnew(SpinBox);
	top_hb->add_child(snap_y);
	snap_y->set_prefix("y:");
	snap_y->set_min(0.01);
	snap_y->set_step(0.01);
	snap_y->set_max(1000);

	top_hb->add_child(memnew(VSeparator));

	top_hb->add_child(memnew(Label(TTR("Blend:"))));
	interpolation = memnew(OptionButton);
	top_hb->add_child(interpolation);
	interpolation->connect("item_selected", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_config_changed));

	edit_hb = memnew(HBoxContainer);
	top_hb->add_child(edit_hb);
	edit_hb->add_child(memnew(VSeparator));
	edit_hb->add_child(memnew(Label(TTR("Point"))));
	edit_x = memnew(SpinBox);
	edit_hb->add_child(edit_x);
	edit_x->set_min(-1000);
	edit_x->set_step(0.01);
	edit_x->set_max(1000);
	edit_x->connect("value_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_edit_point_pos));
	edit_y = memnew(SpinBox);
	edit_hb->add_child(edit_y);
	edit_y->set_min(-1000);
	edit_y->set_step(0.01);
	edit_y->set_max(1000);
	edit_y->connect("value_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_edit_point_pos));
	open_editor = memnew(Button);
	edit_hb->add_child(open_editor);
	open_editor->set_text(TTR("Open Editor"));
	open_editor->connect("pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_open_editor), varray(), CONNECT_DEFERRED);
	edit_hb->hide();
	open_editor->hide();

	HBoxContainer *main_hb = memnew(HBoxContainer);
	add_child(main_hb);
	main_hb->set_v_size_flags(SIZE_EXPAND_FILL);

	GridContainer *main_grid = memnew(GridContainer);
	main_grid->set_columns(2);
	main_hb->add_child(main_grid);
	main_grid->set_h_size_flags(SIZE_EXPAND_FILL);
	{
		VBoxContainer *left_vbox = memnew(VBoxContainer);
		main_grid->add_child(left_vbox);
		left_vbox->set_v_size_flags(SIZE_EXPAND_FILL);
		max_y_value = memnew(SpinBox);
		left_vbox->add_child(max_y_value);
		left_vbox->add_spacer();
		label_y = memnew(LineEdit);
		left_vbox->add_child(label_y);
		label_y->set_expand_to_text_length_enabled(true);
		left_vbox->add_spacer();
		min_y_value = memnew(SpinBox);
		left_vbox->add_child(min_y_value);

		max_y_value->set_max(10000);
		max_y_value->set_min(0.01);
		max_y_value->set_step(0.01);

		min_y_value->set_min(-10000);
		min_y_value->set_max(0);
		min_y_value->set_step(0.01);
	}

	panel = memnew(PanelContainer);
	panel->set_clip_contents(true);
	main_grid->add_child(panel);
	panel->set_h_size_flags(SIZE_EXPAND_FILL);

	blend_space_draw = memnew(Control);
	blend_space_draw->connect("gui_input", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_blend_space_gui_input));
	blend_space_draw->connect("draw", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_blend_space_draw));
	blend_space_draw->set_focus_mode(FOCUS_ALL);

	panel->add_child(blend_space_draw);
	main_grid->add_child(memnew(Control)); //empty bottom left

	{
		HBoxContainer *bottom_vbox = memnew(HBoxContainer);
		main_grid->add_child(bottom_vbox);
		bottom_vbox->set_h_size_flags(SIZE_EXPAND_FILL);
		min_x_value = memnew(SpinBox);
		bottom_vbox->add_child(min_x_value);
		bottom_vbox->add_spacer();
		label_x = memnew(LineEdit);
		bottom_vbox->add_child(label_x);
		label_x->set_expand_to_text_length_enabled(true);
		bottom_vbox->add_spacer();
		max_x_value = memnew(SpinBox);
		bottom_vbox->add_child(max_x_value);

		max_x_value->set_max(10000);
		max_x_value->set_min(0.01);
		max_x_value->set_step(0.01);

		min_x_value->set_min(-10000);
		min_x_value->set_max(0);
		min_x_value->set_step(0.01);
	}

	snap_x->connect("value_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_config_changed));
	snap_y->connect("value_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_config_changed));
	max_x_value->connect("value_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_config_changed));
	min_x_value->connect("value_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_config_changed));
	max_y_value->connect("value_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_config_changed));
	min_y_value->connect("value_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_config_changed));
	label_x->connect("text_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_labels_changed));
	label_y->connect("text_changed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_labels_changed));

	error_panel = memnew(PanelContainer);
	add_child(error_panel);
	error_label = memnew(Label);
	error_panel->add_child(error_label);
	error_label->set_text("eh");

	undo_redo = EditorNode::get_undo_redo();

	set_custom_minimum_size(Size2(0, 300 * EDSCALE));

	menu = memnew(PopupMenu);
	add_child(menu);
	menu->connect("id_pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_add_menu_type));

	animations_menu = memnew(PopupMenu);
	menu->add_child(animations_menu);
	animations_menu->set_name("animations");
	animations_menu->connect("index_pressed", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_add_animation_type));

	open_file = memnew(EditorFileDialog);
	add_child(open_file);
	open_file->set_title(TTR("Open Animation Node"));
	open_file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	open_file->connect("file_selected", callable_mp(this, &AnimationNodeBlendSpace2DEditor::_file_opened));
	undo_redo = EditorNode::get_undo_redo();

	selected_point = -1;
	selected_triangle = -1;

	dragging_selected = false;
	dragging_selected_attempt = false;
}
