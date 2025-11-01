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
#include "scene/2d/parallax_2d.h"
#include "scene/main/viewport.h"

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
	ProjectSettings::get_singleton()->set_setting("rendering/2d/parallax/parallax_preview_mode", p_idx);
	ProjectSettings::get_singleton()->save();

	Node *root = get_tree()->get_edited_scene_root();
	_update_parallax2d(root, p_idx);
}

void Parallax2DEditorPlugin::_update_parallax2d(Node *p_node, int &p_mode) {
#ifdef TOOLS_ENABLED
	Parallax2D *parallax = Object::cast_to<Parallax2D>(p_node);

	if (parallax) {
		parallax->set_preview_mode(p_mode);
	}

	for (int i = 0; i < p_node->get_child_count(); ++i) {
		_update_parallax2d(p_node->get_child(i), p_mode);
	}
#endif // TOOLS_ENABLED
}

void Parallax2DEditorPlugin::forward_canvas_draw_over_viewport(Control *p_overlay) {
	int mode = GLOBAL_GET("rendering/2d/parallax/parallax_preview_mode");

	if (mode != OPTION_ACCURATE) {
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
		} break;
	}
}

Parallax2DEditorPlugin::Parallax2DEditorPlugin() {
	toolbar = memnew(HBoxContainer);
	toolbar->hide();
	add_control_to_container(CONTAINER_CANVAS_EDITOR_MENU, toolbar);

	int mode = GLOBAL_GET("rendering/2d/parallax/parallax_preview_mode");
	options = memnew(OptionButton);
	options->add_item(TTR("Disabled"), OPTION_DISABLED);
	options->add_item(TTR("Accurate"), OPTION_ACCURATE);
	options->add_item(TTR("Basic"), OPTION_BASIC);
	options->select(mode);
	toolbar->add_child(options);
}
