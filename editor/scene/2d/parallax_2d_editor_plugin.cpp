/**************************************************************************/
/*  parallax_2d_editor_plugin.cpp                                         */
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

#include "parallax_2d_editor_plugin.h"

#include "core/config/project_settings.h"
#include "editor/docks/scene_tree_dock.h"
#include "editor/scene/canvas_item_editor_plugin.h"
#include "editor/settings/editor_settings.h"
#include "scene/2d/parallax_2d.h"

void Parallax2DEditorPlugin::edit(Object *p_object) {
	parallax_2d = Object::cast_to<Parallax2D>(p_object);
}

bool Parallax2DEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<Parallax2D>(p_object) != nullptr;
}

void Parallax2DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		toolbar->show();
	} else {
		toolbar->hide();
	}

	update_overlays();
}

void Parallax2DEditorPlugin::_options_callback(int p_idx) {
	EditorSettings::get_singleton()->set_setting("editors/2d/parallax_preview_mode", p_idx);
	EditorSettings::get_singleton()->save();
	_update_preview_mode();
}

void Parallax2DEditorPlugin::_update_preview_mode() {
	int mode = EDITOR_GET("editors/2d/parallax_preview_mode");

	if (mode == OPTION_DISABLED) {
		set_process_internal(false);
		List<Node *> parallax_list;
		get_tree()->get_nodes_in_group("_parallax_2d", &parallax_list);

		for (Node *E : parallax_list) {
			Parallax2D *parallax = Object::cast_to<Parallax2D>(E);

			if (parallax) {
				parallax->set_screen_offset(Vector2());
			}
		}
	} else {
		set_process_internal(true);
	}
}

void Parallax2DEditorPlugin::_update_preview() {
	List<Node *> parallax_list;
	get_tree()->get_nodes_in_group("_parallax_2d", &parallax_list);

	if (parallax_list.is_empty()) {
		return;
	}

	int mode = EDITOR_GET("editors/2d/parallax_preview_mode");
	Size2 vps = Size2(GLOBAL_GET("display/window/size/viewport_width"), GLOBAL_GET("display/window/size/viewport_height"));
	Vector2 screen_offset = Vector2();

	if (mode == OPTION_CENTERED) {
		Transform2D vp_transform = CanvasItemEditor::get_singleton()->get_canvas_transform();
		Vector2 vp_pos = vp_transform.get_origin() / vp_transform.get_scale();
		Rect2 view_rect = Rect2(-vp_pos, Vector2(CanvasItemEditor::get_singleton()->get_viewport_control()->get_size()) / vp_transform.get_scale());
		Rect2 editor_rect = Rect2(view_rect.position + view_rect.size * 0.5 - vps * 0.5, vps);
		screen_offset = editor_rect.position;
	} else if (mode == OPTION_TOPLEFT) {
		Transform2D vp_transform = CanvasItemEditor::get_singleton()->get_canvas_transform();
		Vector2 vp_viewport_pos = -vp_transform.get_origin() / vp_transform.get_scale();
		screen_offset = vp_viewport_pos;
	}

	for (Node *E : parallax_list) {
		Parallax2D *parallax = Object::cast_to<Parallax2D>(E);

		if (parallax) {
			parallax->set_screen_offset(screen_offset);
		}
	}
}

void Parallax2DEditorPlugin::_node_added(Node *p_node) {
	Parallax2D *parallax = Object::cast_to<Parallax2D>(p_node);

	if (parallax) {
		p_node->add_to_group("_parallax_2d");
	}
}

void Parallax2DEditorPlugin::_node_removed(Node *p_node) {
	Parallax2D *parallax = Object::cast_to<Parallax2D>(p_node);

	if (parallax) {
		p_node->remove_from_group("_parallax_2d");
	}
}

void Parallax2DEditorPlugin::forward_canvas_draw_over_viewport(Control *p_overlay) {
	int mode = EDITOR_GET("editors/2d/parallax_preview_mode");

	if (mode != OPTION_CENTERED) {
		return;
	}

	Size2 vps_cache = Size2(GLOBAL_GET("display/window/size/viewport_width"), GLOBAL_GET("display/window/size/viewport_height"));
	Control *vp_control = CanvasItemEditor::get_singleton()->get_viewport_control();
	Transform2D vp_transform = CanvasItemEditor::get_singleton()->get_canvas_transform();
	Vector2 vp_pos = vp_transform.get_origin() / vp_transform.get_scale();
	Rect2 view_rect = Rect2(-vp_pos, vp_control->get_size() / vp_transform.get_scale());
	Rect2 editor_rect = Rect2(view_rect.position + view_rect.size * 0.5 - vps_cache * 0.5, vps_cache);
	editor_rect = CanvasItemEditor::get_singleton()->get_canvas_transform().xform(editor_rect);
	p_overlay->draw_dashed_line(editor_rect.position, editor_rect.position + Vector2(editor_rect.size.x, 0), Color(1, 1, 0, 0.63), 2);
	p_overlay->draw_dashed_line(editor_rect.position, editor_rect.position + Vector2(0, editor_rect.size.y), Color(1, 1, 0, 0.63), 2);
	p_overlay->draw_dashed_line(editor_rect.position + Vector2(editor_rect.size.x, 0), editor_rect.get_end(), Color(1, 1, 0, 0.63), 2);
	p_overlay->draw_dashed_line(editor_rect.position + Vector2(0, editor_rect.size.y), editor_rect.get_end(), Color(1, 1, 0, 0.63), 2);
}

void Parallax2DEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			options->connect(SceneStringName(item_selected), callable_mp(this, &Parallax2DEditorPlugin::_options_callback));
			get_tree()->connect("node_added", callable_mp(this, &Parallax2DEditorPlugin::_node_added));
			get_tree()->connect("node_removed", callable_mp(this, &Parallax2DEditorPlugin::_node_removed));
			_update_preview_mode();
		} break;
		case NOTIFICATION_EXIT_TREE: {
			get_tree()->disconnect("node_added", callable_mp(this, &Parallax2DEditorPlugin::_node_added));
			get_tree()->disconnect("node_removed", callable_mp(this, &Parallax2DEditorPlugin::_node_removed));
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			_update_preview();
		} break;
	}
}

Parallax2DEditorPlugin::Parallax2DEditorPlugin() {
	toolbar = memnew(HBoxContainer);
	toolbar->hide();
	add_control_to_container(CONTAINER_CANVAS_EDITOR_MENU, toolbar);

	int mode = EDITOR_GET("editors/2d/parallax_preview_mode");
	options = memnew(OptionButton);
	options->add_item(TTR("Disabled"), OPTION_DISABLED);
	options->add_item(TTR("Centered"), OPTION_CENTERED);
	options->add_item(TTR("Top Left"), OPTION_TOPLEFT);
	options->select(mode);
	toolbar->add_child(options);
}
