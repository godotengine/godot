#include "animation_blend_space_1d_editor.h"

#include "os/keyboard.h"
#include "scene/animation/animation_blend_tree.h"

void AnimationNodeBlendSpace1DEditorPlugin::edit(Object *p_object) {
	anim_tree_editor->edit(Object::cast_to<AnimationNodeBlendSpace1D>(p_object));
}

bool AnimationNodeBlendSpace1DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("AnimationNodeBlendSpace1D");
}

void AnimationNodeBlendSpace1DEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		button->show();
		editor->make_bottom_panel_item_visible(anim_tree_editor);
		anim_tree_editor->set_process(true);
	} else {
		if (anim_tree_editor->is_visible_in_tree()) {
			editor->hide_bottom_panel();
		}

		button->hide();
		anim_tree_editor->set_process(false);
	}
}

AnimationNodeBlendSpace1DEditorPlugin::AnimationNodeBlendSpace1DEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	anim_tree_editor = memnew(AnimationNodeBlendSpace1DEditor);
	anim_tree_editor->set_custom_minimum_size(Size2(0, 150 * EDSCALE));

	button = editor->add_bottom_panel_item(TTR("BlendSpace1D"), anim_tree_editor);
	button->hide();
}

AnimationNodeBlendSpace1DEditorPlugin::~AnimationNodeBlendSpace1DEditorPlugin() {
}

void AnimationNodeBlendSpace1DEditor::_blend_space_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> k = p_event;

	if (tool_select->is_pressed() && k.is_valid() && k->is_pressed() && k->get_scancode() == KEY_DELETE && !k->is_echo()) {
		if (selected_point != -1) {
			_erase_selected();
			accept_event();
		}
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed() && ((tool_select->is_pressed() && mb->get_button_index() == BUTTON_RIGHT) || (mb->get_button_index() == BUTTON_LEFT && tool_create->is_pressed()))) {
		menu->clear();
		animations_menu->clear();
		animations_to_add.clear();

		List<StringName> classes;
		ClassDB::get_inheriters_from_class("AnimationRootNode", &classes);
		classes.sort_custom<StringName::AlphCompare>();

		menu->add_submenu_item(TTR("Add Animation"), "animations");

		AnimationGraphPlayer *gp = blend_space->get_graph_player();
		ERR_FAIL_COND(!gp);

		if (gp->has_node(gp->get_animation_player())) {
			AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(gp->get_node(gp->get_animation_player()));

			if (ap) {
				List<StringName> names;
				ap->get_animation_list(&names);

				for (List<StringName>::Element *E = names.front(); E; E = E->next()) {
					animations_menu->add_icon_item(get_icon("Animation", "Editoricons"), E->get());
					animations_to_add.push_back(E->get());
				}
			}
		}

		for (List<StringName>::Element *E = classes.front(); E; E = E->next()) {
			String name = String(E->get()).replace_first("AnimationNode", "");
			if (name == "Animation")
				continue;

			int idx = menu->get_item_count();
			menu->add_item(vformat("Add %s", name));
			menu->set_item_metadata(idx, E->get());
		}

		menu->set_global_position(blend_space_draw->get_global_transform().xform(mb->get_position()));
		menu->popup();

		add_point_pos = (mb->get_position() / blend_space_draw->get_size()).x;
		add_point_pos *= (blend_space->get_max_space() - blend_space->get_min_space());
		add_point_pos += blend_space->get_min_space();

		if (snap->is_pressed()) {
			add_point_pos = Math::stepify(add_point_pos, blend_space->get_snap());
		}
	}

	if (mb.is_valid() && mb->is_pressed() && tool_select->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		blend_space_draw->update(); // why not

		// try to see if a point can be selected
		selected_point = -1;
		_update_tool_erase();

		for (int i = 0; i < points.size(); i++) {

			if (Math::abs(float(points[i] - mb->get_position().x)) < 10 * EDSCALE) {
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
	}

	if (mb.is_valid() && !mb->is_pressed() && dragging_selected_attempt && mb->get_button_index() == BUTTON_LEFT) {
		if (dragging_selected) {
			// move
			float point = blend_space->get_blend_point_position(selected_point);
			point += drag_ofs.x;

			if (snap->is_pressed()) {
				point = Math::stepify(point, blend_space->get_snap());
			}

			updating = true;
			undo_redo->create_action("Move Node Point");
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

	// *set* the blend
	if (mb.is_valid() && !mb->is_pressed() && tool_blend->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
		float blend_pos = mb->get_position().x / blend_space_draw->get_size().x;
		blend_pos *= blend_space->get_max_space() - blend_space->get_min_space();
		blend_pos += blend_space->get_min_space();

		blend_space->set_blend_pos(blend_pos);
		blend_space_draw->update();
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid() && !blend_space_draw->has_focus()) {
		blend_space_draw->grab_focus();
		blend_space_draw->update();
	}

	if (mm.is_valid() && dragging_selected_attempt) {
		dragging_selected = true;
		drag_ofs = ((mm->get_position() - drag_from) / blend_space_draw->get_size()) * ((blend_space->get_max_space() - blend_space->get_min_space()) * Vector2(1, 0));
		blend_space_draw->update();
		_update_edited_point_pos();
	}

	if (mm.is_valid() && tool_blend->is_pressed() && mm->get_button_mask() & BUTTON_MASK_LEFT) {
		float blend_pos = mm->get_position().x / blend_space_draw->get_size().x;
		blend_pos *= blend_space->get_max_space() - blend_space->get_min_space();
		blend_pos += blend_space->get_min_space();

		blend_space->set_blend_pos(blend_pos);
		blend_space_draw->update();
	}
}

void AnimationNodeBlendSpace1DEditor::_blend_space_draw() {

	Color linecolor = get_color("font_color", "Label");
	Color linecolor_soft = linecolor;
	linecolor_soft.a *= 0.5;

	Ref<Font> font = get_font("font", "Label");
	Ref<Texture> icon = get_icon("KeyValue", "EditorIcons");
	Ref<Texture> icon_selected = get_icon("KeySelected", "EditorIcons");

	Size2 s = blend_space_draw->get_size();

	if (blend_space_draw->has_focus()) {
		Color color = get_color("accent_color", "Editor");
		blend_space_draw->draw_rect(Rect2(Point2(), s), color, false);
	}

	blend_space_draw->draw_line(Point2(1, s.height - 1), Point2(s.width - 1, s.height - 1), linecolor);

	if (blend_space->get_min_space() < 0) {
		float point = 0.0;
		point = (point - blend_space->get_min_space()) / (blend_space->get_max_space() - blend_space->get_min_space());
		point *= s.width;

		float x = point;

		blend_space_draw->draw_line(Point2(x, s.height - 1), Point2(x, s.height - 5 * EDSCALE), linecolor);
		blend_space_draw->draw_string(font, Point2(x + 2 * EDSCALE, s.height - 2 * EDSCALE - font->get_height() + font->get_ascent()), "0", linecolor);
		blend_space_draw->draw_line(Point2(x, s.height - 5 * EDSCALE), Point2(x, 0), linecolor_soft);
	}

	if (snap->is_pressed()) {

		linecolor_soft.a = linecolor.a * 0.1;

		if (blend_space->get_snap() > 0) {
			int prev_idx = -1;

			for (int i = 0; i < s.x; i++) {
				float v = blend_space->get_min_space() + i * (blend_space->get_max_space() - blend_space->get_min_space()) / s.x;
				int idx = int(v / blend_space->get_snap());

				if (i > 0 && prev_idx != idx) {
					blend_space_draw->draw_line(Point2(i, 0), Point2(i, s.height), linecolor_soft);
				}

				prev_idx = idx;
			}
		}
	}

	points.clear();

	for (int i = 0; i < blend_space->get_blend_point_count(); i++) {
		float point = blend_space->get_blend_point_position(i);

		if (dragging_selected && selected_point == i) {
			point += drag_ofs.x;
			if (snap->is_pressed()) {
				point = Math::stepify(point, blend_space->get_snap());
			}
		}

		point = (point - blend_space->get_min_space()) / (blend_space->get_max_space() - blend_space->get_min_space());
		point *= s.width;

		points.push_back(point);

		Vector2 gui_point = Vector2(point, s.height / 2.0);

		gui_point -= (icon->get_size() / 2.0);

		gui_point = gui_point.floor();

		if (i == selected_point) {
			blend_space_draw->draw_texture(icon_selected, gui_point);
		} else {
			blend_space_draw->draw_texture(icon, gui_point);
		}
	}

	// blend position
	{
		Color color;
		if (tool_blend->is_pressed()) {
			color = get_color("accent_color", "Editor");
		} else {
			color = linecolor;
			color.a *= 0.5;
		}

		float point = blend_space->get_blend_pos();
		point = (point - blend_space->get_min_space()) / (blend_space->get_max_space() - blend_space->get_min_space());
		point *= s.width;

		Vector2 gui_point = Vector2(point, s.height / 2.0);

		float mind = 5 * EDSCALE;
		float maxd = 15 * EDSCALE;
		blend_space_draw->draw_line(gui_point + Vector2(mind, 0), gui_point + Vector2(maxd, 0), color, 2);
		blend_space_draw->draw_line(gui_point + Vector2(-mind, 0), gui_point + Vector2(-maxd, 0), color, 2);
		blend_space_draw->draw_line(gui_point + Vector2(0, mind), gui_point + Vector2(0, maxd), color, 2);
		blend_space_draw->draw_line(gui_point + Vector2(0, -mind), gui_point + Vector2(0, -maxd), color, 2);
	}
}

void AnimationNodeBlendSpace1DEditor::_update_space() {

	if (updating)
		return;

	updating = true;

	if (blend_space->get_parent().is_valid()) {
		goto_parent_hb->show();
	} else {
		goto_parent_hb->hide();
	}

	max_value->set_value(blend_space->get_max_space());
	min_value->set_value(blend_space->get_min_space());

	label_value->set_text(blend_space->get_value_label());

	snap_value->set_value(blend_space->get_snap());

	blend_space_draw->update();

	updating = false;
}

void AnimationNodeBlendSpace1DEditor::_config_changed(double) {
	if (updating)
		return;

	updating = true;
	undo_redo->create_action("Change BlendSpace1D Limits");
	undo_redo->add_do_method(blend_space.ptr(), "set_max_space", max_value->get_value());
	undo_redo->add_undo_method(blend_space.ptr(), "set_max_space", blend_space->get_max_space());
	undo_redo->add_do_method(blend_space.ptr(), "set_min_space", min_value->get_value());
	undo_redo->add_undo_method(blend_space.ptr(), "set_min_space", blend_space->get_min_space());
	undo_redo->add_do_method(blend_space.ptr(), "set_snap", snap_value->get_value());
	undo_redo->add_undo_method(blend_space.ptr(), "set_snap", blend_space->get_snap());
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->commit_action();
	updating = false;

	blend_space_draw->update();
}

void AnimationNodeBlendSpace1DEditor::_labels_changed(String) {
	if (updating)
		return;

	updating = true;
	undo_redo->create_action("Change BlendSpace1D Labels", UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(blend_space.ptr(), "set_value_label", label_value->get_text());
	undo_redo->add_undo_method(blend_space.ptr(), "set_value_label", blend_space->get_value_label());
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->commit_action();
	updating = false;
}

void AnimationNodeBlendSpace1DEditor::_snap_toggled() {
	blend_space_draw->update();
}

void AnimationNodeBlendSpace1DEditor::_add_menu_type(int p_index) {
	String type = menu->get_item_metadata(p_index);

	Object *obj = ClassDB::instance(type);
	ERR_FAIL_COND(!obj);
	AnimationNode *an = Object::cast_to<AnimationNode>(obj);
	ERR_FAIL_COND(!an);

	Ref<AnimationNode> node(an);

	updating = true;
	undo_redo->create_action("Add Node Point");
	undo_redo->add_do_method(blend_space.ptr(), "add_blend_point", node, add_point_pos);
	undo_redo->add_undo_method(blend_space.ptr(), "remove_blend_point", blend_space->get_blend_point_count());
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->commit_action();
	updating = false;

	blend_space_draw->update();
}

void AnimationNodeBlendSpace1DEditor::_add_animation_type(int p_index) {
	Ref<AnimationNodeAnimation> anim;
	anim.instance();

	anim->set_animation(animations_to_add[p_index]);

	updating = true;
	undo_redo->create_action("Add Animation Point");
	undo_redo->add_do_method(blend_space.ptr(), "add_blend_point", anim, add_point_pos);
	undo_redo->add_undo_method(blend_space.ptr(), "remove_blend_point", blend_space->get_blend_point_count());
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->commit_action();
	updating = false;

	blend_space_draw->update();
}

void AnimationNodeBlendSpace1DEditor::_tool_switch(int p_tool) {

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

void AnimationNodeBlendSpace1DEditor::_update_edited_point_pos() {
	if (updating)
		return;

	if (selected_point >= 0 && selected_point < blend_space->get_blend_point_count()) {
		float pos = blend_space->get_blend_point_position(selected_point);

		if (dragging_selected) {
			pos += drag_ofs.x;

			if (snap->is_pressed()) {
				pos = Math::stepify(pos, blend_space->get_snap());
			}
		}

		updating = true;
		edit_value->set_value(pos);
		updating = false;
	}
}

void AnimationNodeBlendSpace1DEditor::_update_tool_erase() {

	bool point_valid = selected_point >= 0 && selected_point < blend_space->get_blend_point_count();
	tool_erase->set_disabled(!point_valid);

	if (point_valid) {
		Ref<AnimationNode> an = blend_space->get_blend_point_node(selected_point);

		if (EditorNode::get_singleton()->item_has_editor(an.ptr())) {
			open_editor->show();
		} else {
			open_editor->hide();
		}

		edit_hb->show();
	} else {
		edit_hb->hide();
	}
}

void AnimationNodeBlendSpace1DEditor::_erase_selected() {
	if (selected_point != -1) {
		updating = true;

		undo_redo->create_action("Remove BlendSpace1D Point");
		undo_redo->add_do_method(blend_space.ptr(), "remove_blend_point", selected_point);
		undo_redo->add_undo_method(blend_space.ptr(), "add_blend_point", blend_space->get_blend_point_node(selected_point), blend_space->get_blend_point_position(selected_point), selected_point);
		undo_redo->add_do_method(this, "_update_space");
		undo_redo->add_undo_method(this, "_update_space");
		undo_redo->commit_action();

		updating = false;

		blend_space_draw->update();
	}
}

void AnimationNodeBlendSpace1DEditor::_edit_point_pos(double) {
	if (updating)
		return;

	updating = true;
	undo_redo->create_action("Move BlendSpace1D Node Point");
	undo_redo->add_do_method(blend_space.ptr(), "set_blend_point_position", selected_point, edit_value->get_value());
	undo_redo->add_undo_method(blend_space.ptr(), "set_blend_point_position", selected_point, blend_space->get_blend_point_position(selected_point));
	undo_redo->add_do_method(this, "_update_space");
	undo_redo->add_undo_method(this, "_update_space");
	undo_redo->add_do_method(this, "_update_edited_point_pos");
	undo_redo->add_undo_method(this, "_update_edited_point_pos");
	undo_redo->commit_action();
	updating = false;

	blend_space_draw->update();
}

void AnimationNodeBlendSpace1DEditor::_open_editor() {

	if (selected_point >= 0 && selected_point < blend_space->get_blend_point_count()) {
		Ref<AnimationNode> an = blend_space->get_blend_point_node(selected_point);
		ERR_FAIL_COND(an.is_null());
		EditorNode::get_singleton()->edit_item(an.ptr());
	}
}

void AnimationNodeBlendSpace1DEditor::_goto_parent() {
	EditorNode::get_singleton()->edit_item(blend_space->get_parent().ptr());
}

void AnimationNodeBlendSpace1DEditor::_removed_from_graph() {
	EditorNode::get_singleton()->edit_item(NULL);
}

void AnimationNodeBlendSpace1DEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_ENTER_TREE || p_what == NOTIFICATION_THEME_CHANGED) {
		error_panel->add_style_override("panel", get_stylebox("bg", "Tree"));
		error_label->add_color_override("font_color", get_color("error_color", "Editor"));
		panel->add_style_override("panel", get_stylebox("bg", "Tree"));
		tool_blend->set_icon(get_icon("EditPivot", "EditorIcons"));
		tool_select->set_icon(get_icon("ToolSelect", "EditorIcons"));
		tool_create->set_icon(get_icon("EditKey", "EditorIcons"));
		tool_erase->set_icon(get_icon("Remove", "EditorIcons"));
		snap->set_icon(get_icon("SnapGrid", "EditorIcons"));
		open_editor->set_icon(get_icon("Edit", "EditorIcons"));
		goto_parent->set_icon(get_icon("MoveUp", "EditorIcons"));
	}

	if (p_what == NOTIFICATION_PROCESS) {
		String error;

		if (!blend_space->get_graph_player()) {
			error = TTR("BlendSpace1D does not belong to an AnimationGraphPlayer node.");
		} else if (!blend_space->get_graph_player()->is_active()) {
			error = TTR("AnimationGraphPlayer is inactive.\nActivate to enable playback, check node warnings if activation fails.");
		} else if (blend_space->get_graph_player()->is_state_invalid()) {
			error = blend_space->get_graph_player()->get_invalid_state_reason();
		}

		if (error != error_label->get_text()) {
			error_label->set_text(error);
			if (error != String()) {
				error_panel->show();
			} else {
				error_panel->hide();
			}
		}
	}
}

void AnimationNodeBlendSpace1DEditor::_bind_methods() {
	ClassDB::bind_method("_blend_space_gui_input", &AnimationNodeBlendSpace1DEditor::_blend_space_gui_input);
	ClassDB::bind_method("_blend_space_draw", &AnimationNodeBlendSpace1DEditor::_blend_space_draw);
	ClassDB::bind_method("_config_changed", &AnimationNodeBlendSpace1DEditor::_config_changed);
	ClassDB::bind_method("_labels_changed", &AnimationNodeBlendSpace1DEditor::_labels_changed);
	ClassDB::bind_method("_update_space", &AnimationNodeBlendSpace1DEditor::_update_space);
	ClassDB::bind_method("_snap_toggled", &AnimationNodeBlendSpace1DEditor::_snap_toggled);
	ClassDB::bind_method("_tool_switch", &AnimationNodeBlendSpace1DEditor::_tool_switch);
	ClassDB::bind_method("_erase_selected", &AnimationNodeBlendSpace1DEditor::_erase_selected);
	ClassDB::bind_method("_update_tool_erase", &AnimationNodeBlendSpace1DEditor::_update_tool_erase);
	ClassDB::bind_method("_edit_point_pos", &AnimationNodeBlendSpace1DEditor::_edit_point_pos);

	ClassDB::bind_method("_add_menu_type", &AnimationNodeBlendSpace1DEditor::_add_menu_type);
	ClassDB::bind_method("_add_animation_type", &AnimationNodeBlendSpace1DEditor::_add_animation_type);

	ClassDB::bind_method("_update_edited_point_pos", &AnimationNodeBlendSpace1DEditor::_update_edited_point_pos);

	ClassDB::bind_method("_open_editor", &AnimationNodeBlendSpace1DEditor::_open_editor);
	ClassDB::bind_method("_goto_parent", &AnimationNodeBlendSpace1DEditor::_goto_parent);

	ClassDB::bind_method("_removed_from_graph", &AnimationNodeBlendSpace1DEditor::_removed_from_graph);
}

void AnimationNodeBlendSpace1DEditor::edit(AnimationNodeBlendSpace1D *p_blend_space) {

	if (blend_space.is_valid()) {
		blend_space->disconnect("removed_from_graph", this, "_removed_from_graph");
	}

	if (p_blend_space) {
		blend_space = Ref<AnimationNodeBlendSpace1D>(p_blend_space);
	} else {
		blend_space.unref();
	}

	if (blend_space.is_null()) {
		hide();
	} else {
		blend_space->connect("removed_from_graph", this, "_removed_from_graph");

		_update_space();
	}
}

AnimationNodeBlendSpace1DEditor *AnimationNodeBlendSpace1DEditor::singleton = NULL;

AnimationNodeBlendSpace1DEditor::AnimationNodeBlendSpace1DEditor() {
	singleton = this;
	updating = false;

	HBoxContainer *top_hb = memnew(HBoxContainer);
	add_child(top_hb);

	Ref<ButtonGroup> bg;
	bg.instance();

	goto_parent_hb = memnew(HBoxContainer);
	top_hb->add_child(goto_parent_hb);

	goto_parent = memnew(ToolButton);
	goto_parent->connect("pressed", this, "_goto_parent", varray(), CONNECT_DEFERRED);
	goto_parent_hb->add_child(goto_parent);
	goto_parent_hb->add_child(memnew(VSeparator));
	goto_parent_hb->hide();

	tool_blend = memnew(ToolButton);
	tool_blend->set_toggle_mode(true);
	tool_blend->set_button_group(bg);
	top_hb->add_child(tool_blend);
	tool_blend->set_pressed(true);
	tool_blend->set_tooltip(TTR("Set the blending position within the space"));
	tool_blend->connect("pressed", this, "_tool_switch", varray(3));

	tool_select = memnew(ToolButton);
	tool_select->set_toggle_mode(true);
	tool_select->set_button_group(bg);
	top_hb->add_child(tool_select);
	tool_select->set_tooltip(TTR("Select and move points, create points with RMB."));
	tool_select->connect("pressed", this, "_tool_switch", varray(0));

	tool_create = memnew(ToolButton);
	tool_create->set_toggle_mode(true);
	tool_create->set_button_group(bg);
	top_hb->add_child(tool_create);
	tool_create->set_tooltip(TTR("Create points."));
	tool_create->connect("pressed", this, "_tool_switch", varray(1));

	tool_erase_sep = memnew(VSeparator);
	top_hb->add_child(tool_erase_sep);
	tool_erase = memnew(ToolButton);
	top_hb->add_child(tool_erase);
	tool_erase->set_tooltip(TTR("Erase points."));
	tool_erase->connect("pressed", this, "_erase_selected");

	top_hb->add_child(memnew(VSeparator));

	snap = memnew(ToolButton);
	snap->set_toggle_mode(true);
	top_hb->add_child(snap);
	snap->set_pressed(true);
	snap->connect("pressed", this, "_snap_toggled");

	snap_value = memnew(SpinBox);
	top_hb->add_child(snap_value);
	snap_value->set_min(0.01);
	snap_value->set_step(0.01);
	snap_value->set_max(1000);

	edit_hb = memnew(HBoxContainer);
	top_hb->add_child(edit_hb);
	edit_hb->add_child(memnew(VSeparator));
	edit_hb->add_child(memnew(Label(TTR("Point"))));

	edit_value = memnew(SpinBox);
	edit_hb->add_child(edit_value);
	edit_value->set_min(-1000);
	edit_value->set_max(1000);
	edit_value->set_step(0.01);
	edit_value->connect("value_changed", this, "_edit_point_pos");

	open_editor = memnew(Button);
	edit_hb->add_child(open_editor);
	open_editor->set_text(TTR("Open Editor"));
	open_editor->connect("pressed", this, "_open_editor", varray(), CONNECT_DEFERRED);

	edit_hb->hide();
	open_editor->hide();

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);
	main_vb->set_v_size_flags(SIZE_EXPAND_FILL);

	panel = memnew(PanelContainer);
	panel->set_clip_contents(true);
	main_vb->add_child(panel);
	panel->set_h_size_flags(SIZE_EXPAND_FILL);
	panel->set_v_size_flags(SIZE_EXPAND_FILL);

	blend_space_draw = memnew(Control);
	blend_space_draw->connect("gui_input", this, "_blend_space_gui_input");
	blend_space_draw->connect("draw", this, "_blend_space_draw");
	blend_space_draw->set_focus_mode(FOCUS_ALL);

	panel->add_child(blend_space_draw);

	{
		HBoxContainer *bottom_hb = memnew(HBoxContainer);
		main_vb->add_child(bottom_hb);
		bottom_hb->set_h_size_flags(SIZE_EXPAND_FILL);

		min_value = memnew(SpinBox);
		min_value->set_max(0);
		min_value->set_min(-10000);
		min_value->set_step(0.01);

		max_value = memnew(SpinBox);
		max_value->set_max(10000);
		max_value->set_min(0.01);
		max_value->set_step(0.01);

		label_value = memnew(LineEdit);
		label_value->set_expand_to_text_length(true);

		// now add

		bottom_hb->add_child(min_value);
		bottom_hb->add_spacer();
		bottom_hb->add_child(label_value);
		bottom_hb->add_spacer();
		bottom_hb->add_child(max_value);
	}

	snap_value->connect("value_changed", this, "_config_changed");
	min_value->connect("value_changed", this, "_config_changed");
	max_value->connect("value_changed", this, "_config_changed");
	label_value->connect("text_changed", this, "_labels_changed");

	error_panel = memnew(PanelContainer);
	add_child(error_panel);

	error_label = memnew(Label);
	error_panel->add_child(error_label);
	error_label->set_text("hmmm");

	undo_redo = EditorNode::get_singleton()->get_undo_redo();

	menu = memnew(PopupMenu);
	add_child(menu);
	menu->connect("index_pressed", this, "_add_menu_type");

	animations_menu = memnew(PopupMenu);
	menu->add_child(animations_menu);
	animations_menu->set_name("animations");
	animations_menu->connect("index_pressed", this, "_add_animation_type");

	selected_point = -1;
	dragging_selected = false;
	dragging_selected_attempt = false;

	set_custom_minimum_size(Size2(0, 150 * EDSCALE));
}
