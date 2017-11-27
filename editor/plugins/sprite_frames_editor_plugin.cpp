/*************************************************************************/
/*  sprite_frames_editor_plugin.cpp                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "sprite_frames_editor_plugin.h"

#include "editor/editor_settings.h"
#include "io/resource_loader.h"
#include "project_settings.h"
#include "scene/3d/sprite_3d.h"

void SpriteFramesEditor::_gui_input(Ref<InputEvent> p_event) {
}

void SpriteFramesEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_PHYSICS_PROCESS) {
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {
		load->set_icon(get_icon("Load", "EditorIcons"));
		_delete->set_icon(get_icon("Remove", "EditorIcons"));
		new_anim->set_icon(get_icon("New", "EditorIcons"));
		remove_anim->set_icon(get_icon("Remove", "EditorIcons"));
	}

	if (p_what == NOTIFICATION_READY) {

		//NodePath("/root")->connect("node_removed", this,"_node_removed",Vector<Variant>(),true);
	}

	if (p_what == NOTIFICATION_DRAW) {
	}
}

void SpriteFramesEditor::_file_load_request(const PoolVector<String> &p_path, int p_at_pos) {

	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	List<Ref<Texture> > resources;

	for (int i = 0; i < p_path.size(); i++) {

		Ref<Texture> resource;
		resource = ResourceLoader::load(p_path[i]);

		if (resource.is_null()) {
			dialog->set_text(TTR("ERROR: Couldn't load frame resource!"));
			dialog->set_title(TTR("Error!"));
			//dialog->get_cancel()->set_text("Close");
			dialog->get_ok()->set_text(TTR("Close"));
			dialog->popup_centered_minsize();
			return; ///beh should show an error i guess
		}

		resources.push_back(resource);
	}

	if (resources.empty()) {
		//print_line("added frames!");
		return;
	}

	undo_redo->create_action(TTR("Add Frame"));
	int fc = frames->get_frame_count(edited_anim);

	int count = 0;

	for (List<Ref<Texture> >::Element *E = resources.front(); E; E = E->next()) {

		undo_redo->add_do_method(frames, "add_frame", edited_anim, E->get(), p_at_pos == -1 ? -1 : p_at_pos + count);
		undo_redo->add_undo_method(frames, "remove_frame", edited_anim, p_at_pos == -1 ? fc : p_at_pos);
		count++;
	}
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");

	undo_redo->commit_action();
	//print_line("added frames!");
}

void SpriteFramesEditor::_load_pressed() {

	ERR_FAIL_COND(!frames->has_animation(edited_anim));
	loading_scene = false;

	file->clear_filters();
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Texture", &extensions);
	for (int i = 0; i < extensions.size(); i++)
		file->add_filter("*." + extensions[i]);

	file->set_mode(EditorFileDialog::MODE_OPEN_FILES);

	file->popup_centered_ratio();
}

void SpriteFramesEditor::_paste_pressed() {

	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	Ref<Texture> r = EditorSettings::get_singleton()->get_resource_clipboard();
	if (!r.is_valid()) {
		dialog->set_text(TTR("Resource clipboard is empty or not a texture!"));
		dialog->set_title(TTR("Error!"));
		//dialog->get_cancel()->set_text("Close");
		dialog->get_ok()->set_text(TTR("Close"));
		dialog->popup_centered_minsize();
		return; ///beh should show an error i guess
	}

	undo_redo->create_action(TTR("Paste Frame"));
	undo_redo->add_do_method(frames, "add_frame", edited_anim, r);
	undo_redo->add_undo_method(frames, "remove_frame", edited_anim, frames->get_frame_count(edited_anim));
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void SpriteFramesEditor::_copy_pressed() {
	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	if (tree->get_current() < 0)
		return;
	Ref<Texture> r = frames->get_frame(edited_anim, tree->get_current());
	if (!r.is_valid()) {
		return;
	}

	EditorSettings::get_singleton()->set_resource_clipboard(r);
}

void SpriteFramesEditor::_empty_pressed() {

	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	int from = -1;

	if (tree->get_current() >= 0) {

		from = tree->get_current();
		sel = from;

	} else {
		from = frames->get_frame_count(edited_anim);
	}

	Ref<Texture> r;

	undo_redo->create_action(TTR("Add Empty"));
	undo_redo->add_do_method(frames, "add_frame", edited_anim, r, from);
	undo_redo->add_undo_method(frames, "remove_frame", edited_anim, from);
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void SpriteFramesEditor::_empty2_pressed() {

	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	int from = -1;

	if (tree->get_current() >= 0) {

		from = tree->get_current();
		sel = from;

	} else {
		from = frames->get_frame_count(edited_anim);
	}

	Ref<Texture> r;

	undo_redo->create_action(TTR("Add Empty"));
	undo_redo->add_do_method(frames, "add_frame", edited_anim, r, from + 1);
	undo_redo->add_undo_method(frames, "remove_frame", edited_anim, from + 1);
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void SpriteFramesEditor::_up_pressed() {

	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	if (tree->get_current() < 0)
		return;

	int to_move = tree->get_current();
	if (to_move < 1)
		return;

	sel = to_move;
	sel -= 1;

	Ref<Texture> r = frames->get_frame(edited_anim, to_move);
	undo_redo->create_action(TTR("Delete Resource"));
	undo_redo->add_do_method(frames, "set_frame", edited_anim, to_move, frames->get_frame(edited_anim, to_move - 1));
	undo_redo->add_do_method(frames, "set_frame", edited_anim, to_move - 1, frames->get_frame(edited_anim, to_move));
	undo_redo->add_undo_method(frames, "set_frame", edited_anim, to_move, frames->get_frame(edited_anim, to_move));
	undo_redo->add_undo_method(frames, "set_frame", edited_anim, to_move - 1, frames->get_frame(edited_anim, to_move - 1));
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void SpriteFramesEditor::_down_pressed() {

	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	if (tree->get_current() < 0)
		return;

	int to_move = tree->get_current();
	if (to_move < 0 || to_move >= frames->get_frame_count(edited_anim) - 1)
		return;

	sel = to_move;
	sel += 1;

	Ref<Texture> r = frames->get_frame(edited_anim, to_move);
	undo_redo->create_action(TTR("Delete Resource"));
	undo_redo->add_do_method(frames, "set_frame", edited_anim, to_move, frames->get_frame(edited_anim, to_move + 1));
	undo_redo->add_do_method(frames, "set_frame", edited_anim, to_move + 1, frames->get_frame(edited_anim, to_move));
	undo_redo->add_undo_method(frames, "set_frame", edited_anim, to_move, frames->get_frame(edited_anim, to_move));
	undo_redo->add_undo_method(frames, "set_frame", edited_anim, to_move + 1, frames->get_frame(edited_anim, to_move + 1));
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void SpriteFramesEditor::_delete_pressed() {

	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	if (tree->get_current() < 0)
		return;

	int to_delete = tree->get_current();
	if (to_delete < 0 || to_delete >= frames->get_frame_count(edited_anim)) {
		return;
	}

	undo_redo->create_action(TTR("Delete Resource"));
	undo_redo->add_do_method(frames, "remove_frame", edited_anim, to_delete);
	undo_redo->add_undo_method(frames, "add_frame", edited_anim, frames->get_frame(edited_anim, to_delete), to_delete);
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void SpriteFramesEditor::_animation_select() {

	if (updating)
		return;

	TreeItem *selected = animations->get_selected();
	ERR_FAIL_COND(!selected);
	edited_anim = selected->get_text(0);
	_update_library(true);
}

static void _find_anim_sprites(Node *p_node, List<Node *> *r_nodes, Ref<SpriteFrames> p_sfames) {

	Node *edited = EditorNode::get_singleton()->get_edited_scene();
	if (!edited)
		return;
	if (p_node != edited && p_node->get_owner() != edited)
		return;

	{
		AnimatedSprite *as = Object::cast_to<AnimatedSprite>(p_node);
		if (as && as->get_sprite_frames() == p_sfames) {
			r_nodes->push_back(p_node);
		}
	}

	{
		AnimatedSprite3D *as = Object::cast_to<AnimatedSprite3D>(p_node);
		if (as && as->get_sprite_frames() == p_sfames) {
			r_nodes->push_back(p_node);
		}
	}

	for (int i = 0; i < p_node->get_child_count(); i++) {
		_find_anim_sprites(p_node->get_child(i), r_nodes, p_sfames);
	}
}

void SpriteFramesEditor::_animation_name_edited() {

	if (updating)
		return;

	if (!frames->has_animation(edited_anim))
		return;

	TreeItem *edited = animations->get_edited();
	if (!edited)
		return;

	String new_name = edited->get_text(0);

	if (new_name == String(edited_anim))
		return;

	new_name = new_name.replace("/", "_").replace(",", " ");

	String name = new_name;
	int counter = 0;
	while (frames->has_animation(name)) {
		counter++;
		name = new_name + " " + itos(counter);
	}

	List<Node *> nodes;
	_find_anim_sprites(EditorNode::get_singleton()->get_edited_scene(), &nodes, Ref<SpriteFrames>(frames));

	undo_redo->create_action(TTR("Rename Animation"));
	undo_redo->add_do_method(frames, "rename_animation", edited_anim, name);
	undo_redo->add_undo_method(frames, "rename_animation", name, edited_anim);

	for (List<Node *>::Element *E = nodes.front(); E; E = E->next()) {

		String current = E->get()->call("get_animation");
		if (current != edited_anim)
			continue;

		undo_redo->add_do_method(E->get(), "set_animation", name);
		undo_redo->add_undo_method(E->get(), "set_animation", edited_anim);
	}

	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");

	edited_anim = new_name;

	undo_redo->commit_action();
}
void SpriteFramesEditor::_animation_add() {

	String name = "New Anim";
	int counter = 0;
	while (frames->has_animation(name)) {
		counter++;
		name = "New Anim " + itos(counter);
	}

	List<Node *> nodes;
	_find_anim_sprites(EditorNode::get_singleton()->get_edited_scene(), &nodes, Ref<SpriteFrames>(frames));

	undo_redo->create_action(TTR("Add Animation"));
	undo_redo->add_do_method(frames, "add_animation", name);
	undo_redo->add_undo_method(frames, "remove_animation", name);
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");

	for (List<Node *>::Element *E = nodes.front(); E; E = E->next()) {

		String current = E->get()->call("get_animation");
		if (frames->has_animation(current))
			continue;

		undo_redo->add_do_method(E->get(), "set_animation", name);
		undo_redo->add_undo_method(E->get(), "set_animation", current);
	}

	edited_anim = name;

	undo_redo->commit_action();
	animations->grab_focus();
}
void SpriteFramesEditor::_animation_remove() {

	//fuck everything
	if (updating)
		return;

	if (!frames->has_animation(edited_anim))
		return;

	undo_redo->create_action(TTR("Remove Animation"));
	undo_redo->add_do_method(frames, "remove_animation", edited_anim);
	undo_redo->add_undo_method(frames, "add_animation", edited_anim);
	undo_redo->add_undo_method(frames, "set_animation_speed", edited_anim, frames->get_animation_speed(edited_anim));
	undo_redo->add_undo_method(frames, "set_animation_loop", edited_anim, frames->get_animation_loop(edited_anim));
	int fc = frames->get_frame_count(edited_anim);
	for (int i = 0; i < fc; i++) {
		Ref<Texture> frame = frames->get_frame(edited_anim, i);
		undo_redo->add_undo_method(frames, "add_frame", edited_anim, frame);
	}
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");

	undo_redo->commit_action();
}

void SpriteFramesEditor::_animation_loop_changed() {

	if (updating)
		return;

	undo_redo->create_action(TTR("Change Animation Loop"));
	undo_redo->add_do_method(frames, "set_animation_loop", edited_anim, anim_loop->is_pressed());
	undo_redo->add_undo_method(frames, "set_animation_loop", edited_anim, frames->get_animation_loop(edited_anim));
	undo_redo->add_do_method(this, "_update_library", true);
	undo_redo->add_undo_method(this, "_update_library", true);
	undo_redo->commit_action();
}

void SpriteFramesEditor::_animation_fps_changed(double p_value) {

	if (updating)
		return;

	undo_redo->create_action(TTR("Change Animation FPS"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(frames, "set_animation_speed", edited_anim, p_value);
	undo_redo->add_undo_method(frames, "set_animation_speed", edited_anim, frames->get_animation_speed(edited_anim));
	undo_redo->add_do_method(this, "_update_library", true);
	undo_redo->add_undo_method(this, "_update_library", true);

	undo_redo->commit_action();
}

void SpriteFramesEditor::_update_library(bool p_skip_selector) {

	updating = true;

	if (!p_skip_selector) {
		animations->clear();

		TreeItem *anim_root = animations->create_item();

		List<StringName> anim_names;

		anim_names.sort_custom<StringName::AlphCompare>();

		frames->get_animation_list(&anim_names);

		anim_names.sort_custom<StringName::AlphCompare>();

		for (List<StringName>::Element *E = anim_names.front(); E; E = E->next()) {

			String name = E->get();

			TreeItem *it = animations->create_item(anim_root);

			it->set_metadata(0, name);

			it->set_text(0, name);
			it->set_editable(0, true);

			if (E->get() == edited_anim) {
				it->select(0);
			}
		}
	}

	tree->clear();

	if (!frames->has_animation(edited_anim)) {
		updating = false;
		return;
	}

	if (sel >= frames->get_frame_count(edited_anim))
		sel = frames->get_frame_count(edited_anim) - 1;
	else if (sel < 0 && frames->get_frame_count(edited_anim))
		sel = 0;

	for (int i = 0; i < frames->get_frame_count(edited_anim); i++) {

		String name;
		Ref<Texture> icon;

		if (frames->get_frame(edited_anim, i).is_null()) {

			name = itos(i) + ": " + TTR("(empty)");

		} else {
			name = itos(i) + ": " + frames->get_frame(edited_anim, i)->get_name();
			icon = frames->get_frame(edited_anim, i);
		}

		tree->add_item(name, icon);
		if (frames->get_frame(edited_anim, i).is_valid())
			tree->set_item_tooltip(tree->get_item_count() - 1, frames->get_frame(edited_anim, i)->get_path());
		if (sel == i)
			tree->select(tree->get_item_count() - 1);
	}

	anim_speed->set_value(frames->get_animation_speed(edited_anim));
	anim_loop->set_pressed(frames->get_animation_loop(edited_anim));

	updating = false;
	//player->add_resource("default",resource);
}

void SpriteFramesEditor::edit(SpriteFrames *p_frames) {

	if (frames == p_frames)
		return;

	frames = p_frames;

	if (p_frames) {

		if (!p_frames->has_animation(edited_anim)) {

			List<StringName> anim_names;
			frames->get_animation_list(&anim_names);
			anim_names.sort_custom<StringName::AlphCompare>();
			if (anim_names.size()) {
				edited_anim = anim_names.front()->get();
			} else {
				edited_anim = StringName();
			}
		}

		_update_library();
	} else {

		hide();
		//set_physics_process(false);
	}
}

Variant SpriteFramesEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {

	if (!frames->has_animation(edited_anim))
		return false;

	int idx = tree->get_item_at_position(p_point, true);

	if (idx < 0 || idx >= frames->get_frame_count(edited_anim))
		return Variant();

	RES frame = frames->get_frame(edited_anim, idx);

	if (frame.is_null())
		return Variant();

	return EditorNode::get_singleton()->drag_resource(frame, p_from);
}

bool SpriteFramesEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {

	Dictionary d = p_data;

	if (!d.has("type"))
		return false;

	if (d.has("from") && (Object *)(d["from"]) == tree)
		return false;

	if (String(d["type"]) == "resource" && d.has("resource")) {
		RES r = d["resource"];

		Ref<Texture> texture = r;

		if (texture.is_valid()) {

			return true;
		}
	}

	if (String(d["type"]) == "files") {

		Vector<String> files = d["files"];

		if (files.size() == 0)
			return false;

		for (int i = 0; i < files.size(); i++) {
			String file = files[0];
			String ftype = EditorFileSystem::get_singleton()->get_file_type(file);

			if (!ClassDB::is_parent_class(ftype, "Texture")) {
				return false;
			}
		}

		return true;
	}
	return false;
}

void SpriteFramesEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {

	if (!can_drop_data_fw(p_point, p_data, p_from))
		return;

	Dictionary d = p_data;

	if (!d.has("type"))
		return;

	int at_pos = tree->get_item_at_position(p_point, true);

	if (String(d["type"]) == "resource" && d.has("resource")) {
		RES r = d["resource"];

		Ref<Texture> texture = r;

		if (texture.is_valid()) {

			undo_redo->create_action(TTR("Add Frame"));
			undo_redo->add_do_method(frames, "add_frame", edited_anim, texture, at_pos == -1 ? -1 : at_pos);
			undo_redo->add_undo_method(frames, "remove_frame", edited_anim, at_pos == -1 ? frames->get_frame_count(edited_anim) : at_pos);
			undo_redo->add_do_method(this, "_update_library");
			undo_redo->add_undo_method(this, "_update_library");
			undo_redo->commit_action();
		}
	}

	if (String(d["type"]) == "files") {

		PoolVector<String> files = d["files"];

		_file_load_request(files, at_pos);
	}
}

void SpriteFramesEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &SpriteFramesEditor::_gui_input);
	ClassDB::bind_method(D_METHOD("_load_pressed"), &SpriteFramesEditor::_load_pressed);
	ClassDB::bind_method(D_METHOD("_empty_pressed"), &SpriteFramesEditor::_empty_pressed);
	ClassDB::bind_method(D_METHOD("_empty2_pressed"), &SpriteFramesEditor::_empty2_pressed);
	ClassDB::bind_method(D_METHOD("_delete_pressed"), &SpriteFramesEditor::_delete_pressed);
	ClassDB::bind_method(D_METHOD("_copy_pressed"), &SpriteFramesEditor::_copy_pressed);
	ClassDB::bind_method(D_METHOD("_paste_pressed"), &SpriteFramesEditor::_paste_pressed);
	ClassDB::bind_method(D_METHOD("_file_load_request", "files", "at_position"), &SpriteFramesEditor::_file_load_request, DEFVAL(-1));
	ClassDB::bind_method(D_METHOD("_update_library", "skipsel"), &SpriteFramesEditor::_update_library, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("_up_pressed"), &SpriteFramesEditor::_up_pressed);
	ClassDB::bind_method(D_METHOD("_down_pressed"), &SpriteFramesEditor::_down_pressed);
	ClassDB::bind_method(D_METHOD("_animation_select"), &SpriteFramesEditor::_animation_select);
	ClassDB::bind_method(D_METHOD("_animation_name_edited"), &SpriteFramesEditor::_animation_name_edited);
	ClassDB::bind_method(D_METHOD("_animation_add"), &SpriteFramesEditor::_animation_add);
	ClassDB::bind_method(D_METHOD("_animation_remove"), &SpriteFramesEditor::_animation_remove);
	ClassDB::bind_method(D_METHOD("_animation_loop_changed"), &SpriteFramesEditor::_animation_loop_changed);
	ClassDB::bind_method(D_METHOD("_animation_fps_changed"), &SpriteFramesEditor::_animation_fps_changed);
	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &SpriteFramesEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &SpriteFramesEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &SpriteFramesEditor::drop_data_fw);
}

SpriteFramesEditor::SpriteFramesEditor() {

	//add_style_override("panel", EditorNode::get_singleton()->get_gui_base()->get_stylebox("panel","Panel"));

	split = memnew(HSplitContainer);
	add_child(split);

	VBoxContainer *vbc_animlist = memnew(VBoxContainer);
	split->add_child(vbc_animlist);
	vbc_animlist->set_custom_minimum_size(Size2(150, 0) * EDSCALE);
	//vbc_animlist->set_v_size_flags(SIZE_EXPAND_FILL);

	VBoxContainer *sub_vb = memnew(VBoxContainer);
	vbc_animlist->add_margin_child(TTR("Animations"), sub_vb, true);
	sub_vb->set_v_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *hbc_animlist = memnew(HBoxContainer);
	sub_vb->add_child(hbc_animlist);

	new_anim = memnew(Button);
	new_anim->set_flat(true);
	hbc_animlist->add_child(new_anim);
	new_anim->set_h_size_flags(SIZE_EXPAND_FILL);
	new_anim->connect("pressed", this, "_animation_add");

	remove_anim = memnew(Button);
	remove_anim->set_flat(true);
	hbc_animlist->add_child(remove_anim);
	remove_anim->connect("pressed", this, "_animation_remove");

	animations = memnew(Tree);
	sub_vb->add_child(animations);
	animations->set_v_size_flags(SIZE_EXPAND_FILL);
	animations->set_hide_root(true);
	animations->connect("cell_selected", this, "_animation_select");
	animations->connect("item_edited", this, "_animation_name_edited");
	animations->set_allow_reselect(true);

	anim_speed = memnew(SpinBox);
	vbc_animlist->add_margin_child(TTR("Speed (FPS):"), anim_speed);
	anim_speed->set_min(0);
	anim_speed->set_max(100);
	anim_speed->set_step(0.01);
	anim_speed->connect("value_changed", this, "_animation_fps_changed");

	anim_loop = memnew(CheckButton);
	anim_loop->set_text(TTR("Loop"));
	vbc_animlist->add_child(anim_loop);
	anim_loop->connect("pressed", this, "_animation_loop_changed");

	VBoxContainer *vbc = memnew(VBoxContainer);
	split->add_child(vbc);
	vbc->set_h_size_flags(SIZE_EXPAND_FILL);

	sub_vb = memnew(VBoxContainer);
	vbc->add_margin_child(TTR("Animation Frames"), sub_vb, true);

	HBoxContainer *hbc = memnew(HBoxContainer);
	sub_vb->add_child(hbc);

	//animations = memnew( ItemList );

	load = memnew(Button);
	load->set_flat(true);
	load->set_tooltip(TTR("Load Resource"));
	hbc->add_child(load);

	copy = memnew(Button);
	copy->set_text(TTR("Copy"));
	hbc->add_child(copy);

	paste = memnew(Button);
	paste->set_text(TTR("Paste"));
	hbc->add_child(paste);

	empty = memnew(Button);
	empty->set_text(TTR("Insert Empty (Before)"));
	hbc->add_child(empty);

	empty2 = memnew(Button);
	empty2->set_text(TTR("Insert Empty (After)"));
	hbc->add_child(empty2);

	move_up = memnew(Button);
	move_up->set_text(TTR("Move (Before)"));
	hbc->add_child(move_up);

	move_down = memnew(Button);
	move_down->set_text(TTR("Move (After)"));
	hbc->add_child(move_down);

	_delete = memnew(Button);
	_delete->set_flat(true);
	hbc->add_child(_delete);

	file = memnew(EditorFileDialog);
	add_child(file);

	tree = memnew(ItemList);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	tree->set_icon_mode(ItemList::ICON_MODE_TOP);

	int thumbnail_size = 96;
	tree->set_max_columns(0);
	tree->set_icon_mode(ItemList::ICON_MODE_TOP);
	tree->set_fixed_column_width(thumbnail_size * 3 / 2);
	tree->set_max_text_lines(2);
	tree->set_fixed_icon_size(Size2(thumbnail_size, thumbnail_size));
	//tree->set_min_icon_size(Size2(thumbnail_size,thumbnail_size));
	tree->set_drag_forwarding(this);

	sub_vb->add_child(tree);

	dialog = memnew(AcceptDialog);
	add_child(dialog);

	load->connect("pressed", this, "_load_pressed");
	_delete->connect("pressed", this, "_delete_pressed");
	copy->connect("pressed", this, "_copy_pressed");
	paste->connect("pressed", this, "_paste_pressed");
	empty->connect("pressed", this, "_empty_pressed");
	empty2->connect("pressed", this, "_empty2_pressed");
	move_up->connect("pressed", this, "_up_pressed");
	move_down->connect("pressed", this, "_down_pressed");
	file->connect("files_selected", this, "_file_load_request");
	loading_scene = false;
	sel = -1;

	updating = false;

	edited_anim = "default";
}

void SpriteFramesEditorPlugin::edit(Object *p_object) {

	frames_editor->set_undo_redo(&get_undo_redo());
	SpriteFrames *s = Object::cast_to<SpriteFrames>(p_object);
	if (!s)
		return;

	frames_editor->edit(s);
}

bool SpriteFramesEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("SpriteFrames");
}

void SpriteFramesEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {
		button->show();
		editor->make_bottom_panel_item_visible(frames_editor);
		//frames_editor->set_process(true);
	} else {

		button->hide();
		if (frames_editor->is_visible_in_tree())
			editor->hide_bottom_panel();

		//frames_editor->set_process(false);
	}
}

SpriteFramesEditorPlugin::SpriteFramesEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	frames_editor = memnew(SpriteFramesEditor);
	frames_editor->set_custom_minimum_size(Size2(0, 300) * EDSCALE);
	button = editor->add_bottom_panel_item("SpriteFrames", frames_editor);
	button->hide();
}

SpriteFramesEditorPlugin::~SpriteFramesEditorPlugin() {
}
