/**************************************************************************/
/*  animation_tree_editor_plugin.cpp                                      */
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

#include "animation_tree_editor_plugin.h"

#include "animation_blend_space_1d_editor.h"
#include "animation_blend_space_2d_editor.h"
#include "animation_blend_tree_editor_plugin.h"
#include "animation_state_machine_editor.h"
#include "core/string/string_buffer.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/themes/editor_scale.h"
#include "scene/animation/animation_blend_tree.h"
#include "scene/gui/button.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/texture_rect.h"

RichTextLabel *AnimationTreeNodeEditorPlugin::create_error_label_node() {
	RichTextLabel *error_label = memnew(RichTextLabel);
	error_label->set_focus_mode(FOCUS_ACCESSIBILITY);
	error_label->set_fit_content(true);
	error_label->set_h_size_flags(SIZE_EXPAND_FILL);
	error_label->set_meta_underline(true);
	error_label->connect("meta_clicked", callable_mp(this, &AnimationTreeNodeEditorPlugin::_meta_clicked));
	return error_label;
}

void AnimationTreeNodeEditorPlugin::_meta_clicked(Variant p_meta) {
	if (p_meta.get_type() != Variant::STRING) {
		return;
	}
	const String raw_info_str = p_meta;
	const String full_path_str = raw_info_str.get_slice("::", 0);
	ERR_FAIL_COND(!full_path_str.ends_with("/"));

	int input_index = -1;
	if (raw_info_str.contains("::")) {
		const String input_index_str = raw_info_str.get_slice("::", 1);
		ERR_FAIL_COND(!input_index_str.is_valid_int());
		input_index = input_index_str.to_int();
	}

	AnimationTree *tree = AnimationTreeEditor::get_singleton()->get_animation_tree();
	ERR_FAIL_NULL(tree);

	// e.g. "parameters/blend_tree/node_name/" -> "parameters/blend_tree/"
	String parent_path = full_path_str.rstrip("/").substr(0, full_path_str.rstrip("/").rfind_char('/') + 1);

	Ref<AnimationRootNode> root_node = tree->get_animation_node_by_path(parent_path);
	ERR_FAIL_COND(root_node.is_null());

	String to_edit_root_path = String(parent_path);
	to_edit_root_path = to_edit_root_path.replace_first(Animation::PARAMETERS_BASE_PATH, "");
	to_edit_root_path = to_edit_root_path.trim_suffix("/");

	Vector<String> navigate_to;
	if (!to_edit_root_path.is_empty()) {
		// empty string still has 1 element when split.
		navigate_to = to_edit_root_path.split("/");
	}
	if (AnimationTreeEditor::get_singleton()->get_edited_path() != navigate_to) {
		AnimationTreeEditor::get_singleton()->edit_path(navigate_to);
	}

	// Special case for AnimationNodeBlendTree.
	if (Ref<AnimationNodeBlendTree> blend_tree = root_node; blend_tree.is_valid()) {
		String child_name = full_path_str;
		child_name = child_name.replace_first(parent_path, "");
		child_name = child_name.trim_suffix("/");
		callable_mp(AnimationNodeBlendTreeEditor::get_singleton(), &AnimationNodeBlendTreeEditor::pan_to_node).call_deferred(child_name, input_index);
	}
}

void AnimationTreeNodeEditorPlugin::update_error_message(const AnimationTree *p_tree, PanelContainer *p_error_panel, RichTextLabel *p_error_label, const String *p_other_errors) {
	const String editor_error_message = p_tree->get_editor_error_message();
	const AHashMap<StringName, AnimationNode::InvalidInstance> &invalid_instances = p_tree->get_invalid_instances();

	if (editor_error_message.is_empty() && invalid_instances.is_empty() && (!p_other_errors || p_other_errors->is_empty())) {
		last_error_key = String();
		p_error_panel->hide();
		return;
	}

	// Cheaper to do this, than rebuild rich text label every frame.
	// Though it would be better, to only call this when the tree changes.
	{
		StringBuffer k;
		k += editor_error_message;
		if (p_other_errors) {
			k += *p_other_errors;
		}
		for (const KeyValue<StringName, AnimationNode::InvalidInstance> &kv : invalid_instances) {
			k += kv.key;
			for (const String &reason : kv.value.errors) {
				k += reason;
			}
			for (const AnimationNode::InvalidInstance::InputError &E : kv.value.input_errors) {
				k += itos(E.index);
				k += E.error;
			}
		}

		String error_key = k.as_string();
		if (error_key == last_error_key) {
			return;
		}
		last_error_key = error_key;
	}

	const String point = String::utf8("•  ");

	p_error_label->clear();
	p_error_label->append_text(editor_error_message);
	if (p_other_errors) {
		p_error_label->append_text(*p_other_errors);
	}

	bool first = true;
	for (const KeyValue<StringName, AnimationNode::InvalidInstance> &kv : invalid_instances) {
		if (!first) {
			p_error_label->add_newline();
		}
		first = false;

		Ref<AnimationNode> node = p_tree->get_animation_node_by_path(kv.key);
		ERR_CONTINUE(node.is_null());

		p_error_label->append_text(vformat(RTR("%s at "), node->get_class()));
		p_error_label->push_meta(String(kv.key));
		{
			p_error_label->append_text(vformat(RTR("'%s'"), kv.key));
		}
		p_error_label->pop();
		p_error_label->append_text(RTR(" has errors.\n"));

		StringBuffer instance_error_builder;
		for (const String &reason : kv.value.errors) {
			instance_error_builder += point;
			instance_error_builder += reason;
			instance_error_builder += "\n";
		}
		p_error_label->append_text(instance_error_builder.as_string());

		// Input errors.
		String input_error_base = String(kv.key) + "::";
		for (const AnimationNode::InvalidInstance::InputError &input_error : kv.value.input_errors) {
			const String input_name = node->get_input_name(input_error.index);

			p_error_label->append_text(point + input_error.error + " ");
			p_error_label->push_meta(input_error_base + itos(input_error.index));
			{
				p_error_label->append_text(vformat(RTR("input %d '%s'."), input_error.index, input_name));
			}
			p_error_label->pop();
			p_error_label->add_newline();
		}
	}

	p_error_panel->show();
}

void AnimationTreeEditor::edit(AnimationTree *p_tree) {
	if (p_tree && !p_tree->is_connected("animation_list_changed", callable_mp(this, &AnimationTreeEditor::_animation_list_changed))) {
		p_tree->connect("animation_list_changed", callable_mp(this, &AnimationTreeEditor::_animation_list_changed), CONNECT_DEFERRED);
	}

	if (tree == p_tree) {
		return;
	}

	if (tree && tree->is_connected("animation_list_changed", callable_mp(this, &AnimationTreeEditor::_animation_list_changed))) {
		tree->disconnect("animation_list_changed", callable_mp(this, &AnimationTreeEditor::_animation_list_changed));
	}

	tree = p_tree;

	Vector<String> path;
	if (tree) {
		edit_path(path);
	}
}

void AnimationTreeEditor::_node_removed(Node *p_node) {
	if (p_node == tree) {
		tree = nullptr;
		_clear_editors();
	}
}

void AnimationTreeEditor::_path_button_pressed(int p_path) {
	edited_path.clear();
	for (int i = 0; i <= p_path; i++) {
		edited_path.push_back(button_path[i]);
	}
}

void AnimationTreeEditor::_animation_list_changed() {
	AnimationNodeBlendTreeEditor *bte = AnimationNodeBlendTreeEditor::get_singleton();
	if (bte) {
		bte->update_graph();
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
	b->set_focus_mode(FOCUS_ACCESSIBILITY);
	b->connect(SceneStringName(pressed), callable_mp(this, &AnimationTreeEditor::_path_button_pressed).bind(-1));
	path_hb->add_child(b);
	for (int i = 0; i < button_path.size(); i++) {
		// bread crumbs.
		TextureRect *texture_rect = memnew(TextureRect);
		texture_rect->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);
		texture_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
		texture_rect->set_custom_minimum_size(Size2(16, 16) * EDSCALE);
		texture_rect->set_texture(get_editor_theme_icon(SNAME("GuiTreeArrowRight")));
		path_hb->add_child(texture_rect);

		b = memnew(Button);
		b->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
		b->set_text(button_path[i]);
		b->set_toggle_mode(true);
		b->set_button_group(group);
		path_hb->add_child(b);
		b->set_pressed(true);
		b->set_focus_mode(FOCUS_ACCESSIBILITY);
		b->connect(SceneStringName(pressed), callable_mp(this, &AnimationTreeEditor::_path_button_pressed).bind(i));
	}
}

void AnimationTreeEditor::edit_path(const Vector<String> &p_path) {
	button_path.clear();

	Ref<AnimationNode> node = tree->get_root_animation_node();

	if (node.is_valid()) {
		current_root = node->get_instance_id();

		for (int i = 0; i < p_path.size(); i++) {
			Ref<AnimationNode> child = node->get_child_by_name(p_path[i]);
			ERR_BREAK_MSG(child.is_null(), vformat("Cannot edit path '%s': node '%s' not found as child of '%s'.", String("/").join(p_path), p_path[i], node->get_class()));
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
		for (int i = 0; i < editors.size(); i++) {
			editors[i]->edit(Ref<AnimationNode>());
			editors[i]->hide();
		}
	}

	_update_path();
}

void AnimationTreeEditor::_clear_editors() {
	button_path.clear();
	current_root = ObjectID();
	edited_path = button_path;
	for (int i = 0; i < editors.size(); i++) {
		editors[i]->edit(Ref<AnimationNode>());
		editors[i]->hide();
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
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			ObjectID root;
			if (tree && tree->get_root_animation_node().is_valid()) {
				root = tree->get_root_animation_node()->get_instance_id();
			}

			if (root != current_root) {
				edit_path(Vector<String>());
			}

			if (button_path.size() != edited_path.size()) {
				edit_path(edited_path);
			}
		} break;

		case NOTIFICATION_ENTER_TREE: {
			get_tree()->connect("node_removed", callable_mp(this, &AnimationTreeEditor::_node_removed));
		} break;

		case NOTIFICATION_EXIT_TREE: {
			get_tree()->disconnect("node_removed", callable_mp(this, &AnimationTreeEditor::_node_removed));
		} break;
	}
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
	String path = Animation::PARAMETERS_BASE_PATH;
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

LocalVector<StringName> AnimationTreeEditor::get_animation_list() {
	// This can be called off the main thread due to resource preview generation. Quit early in that case.
	if (!singleton->tree || !Thread::is_main_thread() || !singleton->is_visible()) {
		// When tree is empty, singleton not in the main thread.
		return LocalVector<StringName>();
	}

	AnimationTree *tree = singleton->tree;
	if (!tree) {
		return LocalVector<StringName>();
	}

	return tree->get_sorted_animation_list();
}

AnimationTreeEditor::AnimationTreeEditor() {
	singleton = this;
	AnimationNodeAnimation::get_editable_animation_list = get_animation_list;

	set_name(TTRC("AnimationTree"));
	set_icon_name("AnimationTreeDock");
	set_dock_shortcut(ED_SHORTCUT_AND_COMMAND("bottom_panels/toggle_animation_tree_bottom_panel", TTRC("Toggle AnimationTree Dock")));
	set_default_slot(DockConstants::DOCK_SLOT_BOTTOM);
	set_available_layouts(EditorDock::DOCK_LAYOUT_HORIZONTAL | EditorDock::DOCK_LAYOUT_FLOATING);
	set_global(false);
	set_transient(true);

	VBoxContainer *main_vbox_container = memnew(VBoxContainer);
	add_child(main_vbox_container);

	path_edit = memnew(ScrollContainer);
	path_edit->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	main_vbox_container->add_child(path_edit);

	path_hb = memnew(HBoxContainer);
	path_hb->add_child(memnew(Label(TTR("Path:"))));
	path_edit->add_child(path_hb);

	main_vbox_container->add_child(memnew(HSeparator));

	editor_base = memnew(MarginContainer);
	editor_base->set_v_size_flags(SIZE_EXPAND_FILL);
	main_vbox_container->add_child(editor_base);

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
		anim_tree_editor->make_visible();
	} else {
		anim_tree_editor->close();
	}

	anim_tree_editor->set_process(p_visible);
}

AnimationTreeEditorPlugin::AnimationTreeEditorPlugin() {
	anim_tree_editor = memnew(AnimationTreeEditor);
	anim_tree_editor->set_custom_minimum_size(Size2(0, 300) * EDSCALE);
	EditorDockManager::get_singleton()->add_dock(anim_tree_editor);
	anim_tree_editor->close();
}
