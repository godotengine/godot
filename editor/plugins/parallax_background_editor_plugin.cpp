/**************************************************************************/
/*  parallax_background_editor_plugin.cpp                                 */
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

#include "parallax_background_editor_plugin.h"

#include "canvas_item_editor_plugin.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/scene_tree_dock.h"
#include "scene/2d/parallax_2d.h"
#include "scene/2d/parallax_background.h"
#include "scene/2d/parallax_layer.h"
#include "scene/gui/box_container.h"
#include "scene/gui/menu_button.h"

void ParallaxBackgroundEditorPlugin::edit(Object *p_object) {
	parallax_background = Object::cast_to<ParallaxBackground>(p_object);
}

bool ParallaxBackgroundEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<ParallaxBackground>(p_object) != nullptr;
}

void ParallaxBackgroundEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		toolbar->show();
	} else {
		toolbar->hide();
	}
}

void ParallaxBackgroundEditorPlugin::_menu_callback(int p_idx) {
	if (p_idx == MENU_CONVERT_TO_PARALLAX_2D) {
		convert_to_parallax2d();
	}
}

void ParallaxBackgroundEditorPlugin::convert_to_parallax2d() {
	ParallaxBackground *parallax_bg = parallax_background;
	TypedArray<Node> children = parallax_bg->get_children();

	EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
	ur->create_action(TTR("Convert to Parallax2D"), UndoRedo::MERGE_DISABLE, parallax_bg);

	for (int i = 0; i < children.size(); i++) {
		ParallaxLayer *parallax_layer = Object::cast_to<ParallaxLayer>(children[i]);

		if (!parallax_layer) {
			continue;
		}

		Parallax2D *parallax2d = memnew(Parallax2D);

		Point2 offset = parallax_bg->get_scroll_base_offset() * parallax_layer->get_motion_scale();
		offset += parallax_layer->get_motion_offset() + parallax_layer->get_position();
		parallax2d->set_scroll_offset(offset);

		Point2 limit_begin = parallax2d->get_limit_begin();
		Point2 limit_end = parallax2d->get_limit_end();

		if (parallax_bg->get_limit_begin().x != 0 || parallax_bg->get_limit_end().x != 0) {
			limit_begin.x = parallax_bg->get_limit_begin().x;
			limit_end.x = parallax_bg->get_limit_end().x;
		}

		if (parallax_bg->get_limit_begin().y != 0 || parallax_bg->get_limit_end().y != 0) {
			limit_begin.y = parallax_bg->get_limit_begin().y;
			limit_end.y = parallax_bg->get_limit_end().y;
		}

		parallax2d->set_limit_begin(limit_begin);
		parallax2d->set_limit_end(limit_end);
		parallax2d->set_follow_viewport(!parallax_bg->is_ignore_camera_zoom());
		parallax2d->set_repeat_size(parallax_layer->get_mirroring());
		parallax2d->set_scroll_scale(parallax_bg->get_scroll_base_scale() * parallax_layer->get_motion_scale());

		SceneTreeDock::get_singleton()->replace_node(parallax_layer, parallax2d);
	}

	if (parallax_bg->is_ignore_camera_zoom()) {
		CanvasLayer *canvas_layer = memnew(CanvasLayer);
		SceneTreeDock::get_singleton()->replace_node(parallax_bg, canvas_layer);
	} else {
		Node2D *node2d = memnew(Node2D);
		SceneTreeDock::get_singleton()->replace_node(parallax_bg, node2d);
	}

	ur->commit_action(false);
}

void ParallaxBackgroundEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			menu->get_popup()->connect(SceneStringName(id_pressed), callable_mp(this, &ParallaxBackgroundEditorPlugin::_menu_callback));
			menu->set_button_icon(menu->get_editor_theme_icon(SNAME("ParallaxBackground")));
		} break;
	}
}

ParallaxBackgroundEditorPlugin::ParallaxBackgroundEditorPlugin() {
	toolbar = memnew(HBoxContainer);
	toolbar->hide();
	add_control_to_container(CONTAINER_CANVAS_EDITOR_MENU, toolbar);

	menu = memnew(MenuButton);
	menu->get_popup()->add_item(TTR("Convert to Parallax2D"), MENU_CONVERT_TO_PARALLAX_2D);
	menu->set_text(TTR("ParallaxBackground"));
	menu->set_switch_on_hover(true);
	toolbar->add_child(menu);
}
