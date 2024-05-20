/**************************************************************************/
/*  skeleton_ik_3d_editor_plugin.cpp                                      */
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

#include "skeleton_ik_3d_editor_plugin.h"

#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "scene/3d/skeleton_ik_3d.h"
#include "scene/gui/button.h"

void SkeletonIK3DEditorPlugin::_play() {
	if (!skeleton_ik) {
		return;
	}

	if (!skeleton_ik->get_parent_skeleton()) {
		return;
	}

	if (play_btn->is_pressed()) {
		skeleton_ik->start();
	} else {
		skeleton_ik->stop();
	}
}

void SkeletonIK3DEditorPlugin::edit(Object *p_object) {
	SkeletonIK3D *s = Object::cast_to<SkeletonIK3D>(p_object);
	if (!s) {
		return;
	}

	skeleton_ik = s;
	play_btn->set_pressed(skeleton_ik->is_running());
}

bool SkeletonIK3DEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("SkeletonIK3D");
}

void SkeletonIK3DEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		play_btn->show();
	} else {
		play_btn->hide();
	}
}

void SkeletonIK3DEditorPlugin::_bind_methods() {
}

SkeletonIK3DEditorPlugin::SkeletonIK3DEditorPlugin() {
	play_btn = memnew(Button);
	play_btn->set_icon(EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("Play"), EditorStringName(EditorIcons)));
	play_btn->set_text(TTR("Play IK"));
	play_btn->set_toggle_mode(true);
	play_btn->hide();
	play_btn->connect(SceneStringName(pressed), callable_mp(this, &SkeletonIK3DEditorPlugin::_play));
	add_control_to_container(CONTAINER_SPATIAL_EDITOR_MENU, play_btn);
	skeleton_ik = nullptr;
}

SkeletonIK3DEditorPlugin::~SkeletonIK3DEditorPlugin() {}
