/*************************************************************************/
/*  animation_tree_editor_plugin.cpp                                     */
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

#include "animation_tree_editor_plugin.h"

#include "animation_blend_space_1d_editor.h"
#include "animation_blend_space_2d_editor.h"
#include "animation_blend_tree_editor_plugin.h"
#include "animation_state_machine_editor.h"
#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/io/resource_loader.h"
#include "core/math/delaunay_2d.h"
#include "core/os/keyboard.h"
#include "editor/editor_scale.h"
#include "scene/animation/animation_blend_tree.h"
#include "scene/animation/animation_player.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/main/window.h"
#include "scene/scene_string_names.h"

void AnimationTreeEditor::edit(AnimationTree *p_tree) {
	if (tree == p_tree) {
		return;
	}

	tree = p_tree;

	Vector<String> path;
	if (tree && tree->has_meta("_tree_edit_path")) {
		path = tree->get_meta("_tree_edit_path");
		edit_path(path);
	} else {
		current_root = ObjectID();
	}
}

void AnimationTreeEditor::_path_button_pressed(int p_path) {
	edited_path.clear();
	for (int i = 0; i <= p_path; i++) {
		edited_path.push_back(button_path[i]);
	}
}

void AnimationTreeEditor::_update_path() {
	while (path_hb->get_child_count() > 1) {
		memdelete(path_hb->get_child(1));
	}

	Ref<ButtonGroup> group;
	group.instantiate();

	Button *b = memnew(Button);
	b->set_text(TTR("Root"));
	b->set_toggle_mode(true);
	b->set_button_group(group);
	b->set_pressed(true);
	b->set_focus_mode(FOCUS_NONE);
	b->connect("pressed", callable_mp(this, &AnimationTreeEditor::_path_button_pressed), varray(-1));
	path_hb->add_child(b);
	for (int i = 0; i < button_path.size(); i++) {
		b = memnew(Button);
		b->set_text(button_path[i]);
		b->set_toggle_mode(true);
		b->set_button_group(group);
		path_hb->add_child(b);
		b->set_pressed(true);
		b->set_focus_mode(FOCUS_NONE);
		b->connect("pressed", callable_mp(this, &AnimationTreeEditor::_path_button_pressed), varray(i));
	}
}

void AnimationTreeEditor::edit_path(const Vector<String> &p_path) {
	button_path.clear();

	Ref<AnimationNode> node = tree->get_tree_root();

	if (node.is_valid()) {
		current_root = node->get_instance_id();

		for (int i = 0; i < p_path.size(); i++) {
			Ref<AnimationNode> child = node->get_child_by_name(p_path[i]);
			ERR_BREAK(child.is_null());
			node = child;
			button_path.push_back(p_path[i]);
		}

		edited_path = button_path;

		for (int i = 0; i < editors.size(); i++) {
			if (editors[i]->can_edit(node)) {
				editors[i]->edit(node);
				editors[i]->show();
			} else {
				editors[i]->edit(Ref<AnimationNode>());
				editors[i]->hide();
			}
		}
	} else {
		current_root = ObjectID();
		edited_path = button_path;
	}

	_update_path();
}

Vector<String> AnimationTreeEditor::get_edited_path() const {
	return button_path;
}

void AnimationTreeEditor::enter_editor(const String &p_path) {
	Vector<String> path = edited_path;
	path.push_back(p_path);
	edit_path(path);
}

void AnimationTreeEditor::_notification(int p_what) {
	if (p_what == NOTIFICATION_PROCESS) {
		ObjectID root;
		if (tree && tree->get_tree_root().is_valid()) {
			root = tree->get_tree_root()->get_instance_id();
		}

		if (root != current_root) {
			edit_path(Vector<String>());
		}

		if (button_path.size() != edited_path.size()) {
			edit_path(edited_path);
		}
	}
}

void AnimationTreeEditor::_bind_methods() {
}

AnimationTreeEditor *AnimationTreeEditor::singleton = nullptr;

void AnimationTreeEditor::add_plugin(AnimationTreeNodeEditorPlugin *p_editor) {
	ERR_FAIL_COND(p_editor->get_parent());
	editor_base->add_child(p_editor);
	editors.push_back(p_editor);
	p_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	p_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	p_editor->hide();
}

void AnimationTreeEditor::remove_plugin(AnimationTreeNodeEditorPlugin *p_editor) {
	ERR_FAIL_COND(p_editor->get_parent() != editor_base);
	editor_base->remove_child(p_editor);
	editors.erase(p_editor);
}

String AnimationTreeEditor::get_base_path() {
	String path = SceneStringNames::get_singleton()->parameters_base_path;
	for (int i = 0; i < edited_path.size(); i++) {
		path += edited_path[i] + "/";
	}
	return path;
}

bool AnimationTreeEditor::can_edit(const Ref<AnimationNode> &p_node) const {
	for (int i = 0; i < editors.size(); i++) {
		if (editors[i]->can_edit(p_node)) {
			return true;
		}
	}
	return false;
}

Vector<String> AnimationTreeEditor::get_animation_list() {
	if (!singleton->is_visible()) {
		return Vector<String>();
	}

	AnimationTree *tree = singleton->tree;
	if (!tree || !tree->has_node(tree->get_animation_player())) {
		return Vector<String>();
	}

	AnimationPlayer *ap = Object::cast_to<AnimationPlayer>(tree->get_node(tree->get_animation_player()));

	if (!ap) {
		return Vector<String>();
	}

	List<StringName> anims;
	ap->get_animation_list(&anims);
	Vector<String> ret;
	for (const StringName &E : anims) {
		ret.push_back(E);
	}

	return ret;
}

AnimationTreeEditor::AnimationTreeEditor() {
	AnimationNodeAnimation::get_editable_animation_list = get_animation_list;
	path_edit = memnew(ScrollContainer);
	add_child(path_edit);
	path_edit->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	path_hb = memnew(HBoxContainer);
	path_edit->add_child(path_hb);
	path_hb->add_child(memnew(Label(TTR("Path:"))));

	add_child(memnew(HSeparator));

	singleton = this;
	editor_base = memnew(MarginContainer);
	editor_base->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(editor_base);

	add_plugin(memnew(AnimationNodeBlendTreeEditor));
	add_plugin(memnew(AnimationNodeBlendSpace1DEditor));
	add_plugin(memnew(AnimationNodeBlendSpace2DEditor));
	add_plugin(memnew(AnimationNodeStateMachineEditor));
}

void AnimationTreeEditorPlugin::edit(Object *p_object) {
	anim_tree_editor->edit(Object::cast_to<AnimationTree>(p_object));
}

bool AnimationTreeEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("AnimationTree");
}

void AnimationTreeEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		//editor->hide_animation_player_editors();
		//editor->animation_panel_make_visible(true);
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

AnimationTreeEditorPlugin::AnimationTreeEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	anim_tree_editor = memnew(AnimationTreeEditor);
	anim_tree_editor->set_custom_minimum_size(Size2(0, 300) * EDSCALE);

	button = editor->add_bottom_panel_item(TTR("AnimationTree"), anim_tree_editor);
	button->hide();
}

AnimationTreeEditorPlugin::~AnimationTreeEditorPlugin() {
}
