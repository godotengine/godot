/*************************************************************************/
/*  animation_player_editor_plugin.cpp                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "animation_player_editor_plugin.h"

#include "editor/animation_editor.h"
#include "editor/editor_settings.h"
#include "global_config.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/keyboard.h"

void AnimationPlayerEditor::_node_removed(Node *p_node) {

	if (player && player == p_node) {
		player = NULL;

		set_process(false);

		key_editor->set_animation(Ref<Animation>());
		key_editor->set_root(NULL);
		key_editor->show_select_node_warning(true);
		_update_player();
		//editor->animation_editor_make_visible(false);
	}
}

void AnimationPlayerEditor::_gui_input(InputEvent p_event) {
}

void AnimationPlayerEditor::_notification(int p_what) {

	if (p_what == NOTIFICATION_PROCESS) {

		if (!player)
			return;

		updating = true;

		if (player->is_playing()) {

			{
				String animname = player->get_current_animation();

				if (player->has_animation(animname)) {
					Ref<Animation> anim = player->get_animation(animname);
					if (!anim.is_null()) {

						frame->set_max(anim->get_length());
					}
				}
			}
			frame->set_value(player->get_current_animation_pos());
			key_editor->set_anim_pos(player->get_current_animation_pos());
			EditorNode::get_singleton()->get_property_editor()->refresh();

		} else if (last_active) {
			//need the last frame after it stopped

			frame->set_value(player->get_current_animation_pos());
		}

		last_active = player->is_playing();
		//seek->set_val(player->get_pos());
		updating = false;
	}

	if (p_what == NOTIFICATION_ENTER_TREE) {

		//editor->connect("hide_animation_player_editors",this,"_hide_anim_editors");
		add_anim->set_icon(get_icon("New", "EditorIcons"));
		rename_anim->set_icon(get_icon("Rename", "EditorIcons"));
		duplicate_anim->set_icon(get_icon("Duplicate", "EditorIcons"));
		autoplay->set_icon(get_icon("AutoPlay", "EditorIcons"));
		load_anim->set_icon(get_icon("Folder", "EditorIcons"));
		save_anim->set_icon(get_icon("Save", "EditorIcons"));
		save_anim->get_popup()->connect("id_pressed", this, "_animation_save_menu");
		remove_anim->set_icon(get_icon("Remove", "EditorIcons"));

		blend_anim->set_icon(get_icon("Blend", "EditorIcons"));
		play->set_icon(get_icon("PlayStart", "EditorIcons"));
		play_from->set_icon(get_icon("Play", "EditorIcons"));
		play_bw->set_icon(get_icon("PlayStartBackwards", "EditorIcons"));
		play_bw_from->set_icon(get_icon("PlayBackwards", "EditorIcons"));

		autoplay_icon = get_icon("AutoPlay", "EditorIcons");
		stop->set_icon(get_icon("Stop", "EditorIcons"));
		resource_edit_anim->set_icon(get_icon("EditResource", "EditorIcons"));
		pin->set_icon(get_icon("Pin", "EditorIcons"));
		tool_anim->set_icon(get_icon("Tools", "EditorIcons"));
		tool_anim->get_popup()->connect("id_pressed", this, "_animation_tool_menu");

		blend_editor.next->connect("item_selected", this, "_blend_editor_next_changed");

		nodename->set_icon(get_icon("AnimationPlayer", "EditorIcons"));

		/*
		anim_editor_load->set_normal_texture( get_icon("AnimGet","EditorIcons"));
		anim_editor_store->set_normal_texture( get_icon("AnimSet","EditorIcons"));
		anim_editor_load->set_pressed_texture( get_icon("AnimGet","EditorIcons"));
		anim_editor_store->set_pressed_texture( get_icon("AnimSet","EditorIcons"));
		anim_editor_load->set_hover_texture( get_icon("AnimGetHl","EditorIcons"));
		anim_editor_store->set_hover_texture( get_icon("AnimSetHl","EditorIcons"));
*/

		get_tree()->connect("node_removed", this, "_node_removed");
	}
}

void AnimationPlayerEditor::_autoplay_pressed() {

	if (updating)
		return;
	if (animation->get_item_count() == 0) {
		return;
	}

	String current = animation->get_item_text(animation->get_selected());
	if (player->get_autoplay() == current) {
		//unset
		undo_redo->create_action(TTR("Toggle Autoplay"));
		undo_redo->add_do_method(player, "set_autoplay", "");
		undo_redo->add_undo_method(player, "set_autoplay", player->get_autoplay());
		undo_redo->add_do_method(this, "_animation_player_changed", player);
		undo_redo->add_undo_method(this, "_animation_player_changed", player);
		undo_redo->commit_action();

	} else {
		//set
		undo_redo->create_action(TTR("Toggle Autoplay"));
		undo_redo->add_do_method(player, "set_autoplay", current);
		undo_redo->add_undo_method(player, "set_autoplay", player->get_autoplay());
		undo_redo->add_do_method(this, "_animation_player_changed", player);
		undo_redo->add_undo_method(this, "_animation_player_changed", player);
		undo_redo->commit_action();
	}
}

void AnimationPlayerEditor::_play_pressed() {

	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {

		current = animation->get_item_text(animation->get_selected());
	}

	if (current != "") {

		if (current == player->get_current_animation())
			player->stop(); //so it wont blend with itself
		player->play(current);
	}

	//unstop
	stop->set_pressed(false);
	//unpause
	//pause->set_pressed(false);
}

void AnimationPlayerEditor::_play_from_pressed() {

	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {

		current = animation->get_item_text(animation->get_selected());
	}

	if (current != "") {

		float time = player->get_current_animation_pos();

		if (current == player->get_current_animation() && player->is_playing()) {

			player->stop(); //so it wont blend with itself
		}

		player->play(current);
		player->seek(time);
	}

	//unstop
	stop->set_pressed(false);
	//unpause
	//pause->set_pressed(false);
}

void AnimationPlayerEditor::_play_bw_pressed() {

	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {

		current = animation->get_item_text(animation->get_selected());
	}

	if (current != "") {

		if (current == player->get_current_animation())
			player->stop(); //so it wont blend with itself
		player->play(current, -1, -1, true);
	}

	//unstop
	stop->set_pressed(false);
	//unpause
	//pause->set_pressed(false);
}

void AnimationPlayerEditor::_play_bw_from_pressed() {

	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {

		current = animation->get_item_text(animation->get_selected());
	}

	if (current != "") {

		float time = player->get_current_animation_pos();
		if (current == player->get_current_animation())
			player->stop(); //so it wont blend with itself

		player->play(current, -1, -1, true);
		player->seek(time);
	}

	//unstop
	stop->set_pressed(false);
	//unpause
	//pause->set_pressed(false);
}
void AnimationPlayerEditor::_stop_pressed() {

	player->stop(false);
	play->set_pressed(false);
	stop->set_pressed(true);
	//pause->set_pressed(false);
	//player->set_pause(false);
}

void AnimationPlayerEditor::_pause_pressed() {

	//player->set_pause( pause->is_pressed() );
}
void AnimationPlayerEditor::_animation_selected(int p_which) {

	if (updating)
		return;
	// when selecting an animation, the idea is that the only interesting behavior
	// ui-wise is that it should play/blend the next one if currently playing
	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {

		current = animation->get_item_text(animation->get_selected());
	}

	if (current != "") {

		player->set_current_animation(current);

		Ref<Animation> anim = player->get_animation(current);
		{

			key_editor->set_animation(anim);
			Node *root = player->get_node(player->get_root());
			if (root) {
				key_editor->set_root(root);
			}
		}
		frame->set_max(anim->get_length());
		if (anim->get_step())
			frame->set_step(anim->get_step());
		else
			frame->set_step(0.00001);

	} else {
		key_editor->set_animation(Ref<Animation>());
		key_editor->set_root(NULL);
	}

	autoplay->set_pressed(current == player->get_autoplay());
}

void AnimationPlayerEditor::_animation_new() {

	renaming = false;
	name_title->set_text(TTR("New Animation Name:"));

	int count = 1;
	String base = TTR("New Anim");
	while (true) {
		String attempt = base;
		if (count > 1)
			attempt += " (" + itos(count) + ")";
		if (player->has_animation(attempt)) {
			count++;
			continue;
		}
		base = attempt;
		break;
	}

	name->set_text(base);
	name_dialog->popup_centered(Size2(300, 90));
	name->select_all();
	name->grab_focus();
}
void AnimationPlayerEditor::_animation_rename() {

	if (animation->get_item_count() == 0)
		return;
	int selected = animation->get_selected();
	String selected_name = animation->get_item_text(selected);

	name_title->set_text(TTR("Change Animation Name:"));
	name->set_text(selected_name);
	renaming = true;
	name_dialog->popup_centered(Size2(300, 90));
	name->select_all();
	name->grab_focus();
}
void AnimationPlayerEditor::_animation_load() {
	ERR_FAIL_COND(!player);
	file->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	file->clear_filters();
	List<String> extensions;

	ResourceLoader::get_recognized_extensions_for_type("Animation", &extensions);
	for (List<String>::Element *E = extensions.front(); E; E = E->next()) {

		file->add_filter("*." + E->get() + " ; " + E->get().to_upper());
	}

	file->popup_centered_ratio();
	current_option = RESOURCE_LOAD;
}

void AnimationPlayerEditor::_animation_save_in_path(const Ref<Resource> &p_resource, const String &p_path) {

	int flg = 0;
	if (EditorSettings::get_singleton()->get("filesystem/on_save/compress_binary_resources"))
		flg |= ResourceSaver::FLAG_COMPRESS;
	/*
	if (EditorSettings::get_singleton()->get("filesystem/on_save/save_paths_as_relative"))
		flg |= ResourceSaver::FLAG_RELATIVE_PATHS;
	*/

	String path = GlobalConfig::get_singleton()->localize_path(p_path);
	Error err = ResourceSaver::save(path, p_resource, flg | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS);

	if (err != OK) {
		accept->set_text(TTR("Error saving resource!"));
		accept->popup_centered_minsize();
		return;
	}
	//EditorFileSystem::get_singleton()->update_file(path,p_resource->get_type());

	((Resource *)p_resource.ptr())->set_path(path);
	editor->emit_signal("resource_saved", p_resource);
}

void AnimationPlayerEditor::_animation_save(const Ref<Resource> &p_resource) {

	if (p_resource->get_path().is_resource_file()) {
		_animation_save_in_path(p_resource, p_resource->get_path());
	} else {
		_animation_save_as(p_resource);
	}
}

void AnimationPlayerEditor::_animation_save_as(const Ref<Resource> &p_resource) {

	file->set_mode(EditorFileDialog::MODE_SAVE_FILE);

	List<String> extensions;
	ResourceSaver::get_recognized_extensions(p_resource, &extensions);
	file->clear_filters();
	for (int i = 0; i < extensions.size(); i++) {

		file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
	}

	//file->set_current_path(current_path);
	if (p_resource->get_path() != "") {
		file->set_current_path(p_resource->get_path());
		if (extensions.size()) {
			String ext = p_resource->get_path().get_extension().to_lower();
			if (extensions.find(ext) == NULL) {
				file->set_current_path(p_resource->get_path().replacen("." + ext, "." + extensions.front()->get()));
			}
		}
	} else {

		String existing;
		if (extensions.size()) {
			if (p_resource->get_name() != "") {
				existing = p_resource->get_name() + "." + extensions.front()->get().to_lower();
			} else {
				existing = "new_" + p_resource->get_class().to_lower() + "." + extensions.front()->get().to_lower();
			}
		}
		file->set_current_path(existing);
	}
	file->popup_centered_ratio();
	file->set_title(TTR("Save Resource As.."));
	current_option = RESOURCE_SAVE;
}

void AnimationPlayerEditor::_animation_remove() {

	if (animation->get_item_count() == 0)
		return;

	delete_dialog->set_text(TTR("Delete Animation?"));
	delete_dialog->popup_centered_minsize();
}

void AnimationPlayerEditor::_animation_remove_confirmed() {

	String current = animation->get_item_text(animation->get_selected());
	Ref<Animation> anim = player->get_animation(current);

	undo_redo->create_action(TTR("Remove Animation"));
	undo_redo->add_do_method(player, "remove_animation", current);
	undo_redo->add_undo_method(player, "add_animation", current, anim);
	undo_redo->add_do_method(this, "_animation_player_changed", player);
	undo_redo->add_undo_method(this, "_animation_player_changed", player);
	undo_redo->commit_action();
}

void AnimationPlayerEditor::_select_anim_by_name(const String &p_anim) {

	int idx = -1;
	for (int i = 0; i < animation->get_item_count(); i++) {

		if (animation->get_item_text(i) == p_anim) {

			idx = i;
			break;
		}
	}

	ERR_FAIL_COND(idx == -1);

	animation->select(idx);

	_animation_selected(idx);
}

void AnimationPlayerEditor::_animation_name_edited() {

	player->stop();

	String new_name = name->get_text();
	if (new_name == "" || new_name.find(":") != -1 || new_name.find("/") != -1) {
		error_dialog->set_text(TTR("ERROR: Invalid animation name!"));
		error_dialog->popup_centered_minsize();
		return;
	}

	if (renaming && animation->get_item_count() > 0 && animation->get_item_text(animation->get_selected()) == new_name) {
		name_dialog->hide();
		return;
	}

	if (player->has_animation(new_name)) {
		error_dialog->set_text(TTR("ERROR: Animation name already exists!"));
		error_dialog->popup_centered_minsize();
		return;
	}

	if (renaming) {
		String current = animation->get_item_text(animation->get_selected());
		Ref<Animation> anim = player->get_animation(current);

		undo_redo->create_action(TTR("Rename Animation"));
		undo_redo->add_do_method(player, "rename_animation", current, new_name);
		undo_redo->add_do_method(anim.ptr(), "set_name", new_name);
		undo_redo->add_undo_method(player, "rename_animation", new_name, current);
		undo_redo->add_undo_method(anim.ptr(), "set_name", current);
		undo_redo->add_do_method(this, "_animation_player_changed", player);
		undo_redo->add_undo_method(this, "_animation_player_changed", player);
		undo_redo->commit_action();

		_select_anim_by_name(new_name);

	} else {

		Ref<Animation> new_anim = Ref<Animation>(memnew(Animation));
		new_anim->set_name(new_name);

		undo_redo->create_action(TTR("Add Animation"));
		undo_redo->add_do_method(player, "add_animation", new_name, new_anim);
		undo_redo->add_undo_method(player, "remove_animation", new_name);
		undo_redo->add_do_method(this, "_animation_player_changed", player);
		undo_redo->add_undo_method(this, "_animation_player_changed", player);
		undo_redo->commit_action();

		_select_anim_by_name(new_name);
	}

	name_dialog->hide();
}

void AnimationPlayerEditor::_blend_editor_next_changed(const int p_idx) {

	if (animation->get_item_count() == 0)
		return;

	String current = animation->get_item_text(animation->get_selected());

	undo_redo->create_action(TTR("Blend Next Changed"));
	undo_redo->add_do_method(player, "animation_set_next", current, blend_editor.next->get_item_text(p_idx));
	undo_redo->add_undo_method(player, "animation_set_next", current, player->animation_get_next(current));
	undo_redo->add_do_method(this, "_animation_player_changed", player);
	undo_redo->add_undo_method(this, "_animation_player_changed", player);
	undo_redo->commit_action();
}

void AnimationPlayerEditor::_animation_blend() {

	if (updating_blends)
		return;

	blend_editor.tree->clear();

	if (animation->get_item_count() == 0)
		return;

	String current = animation->get_item_text(animation->get_selected());

	blend_editor.dialog->popup_centered(Size2(400, 400));

	blend_editor.tree->set_hide_root(true);
	blend_editor.tree->set_column_min_width(0, 10);
	blend_editor.tree->set_column_min_width(1, 3);

	List<StringName> anims;
	player->get_animation_list(&anims);
	TreeItem *root = blend_editor.tree->create_item();
	updating_blends = true;

	int i = 0;
	bool anim_found = false;
	blend_editor.next->clear();
	blend_editor.next->add_item("", i);

	for (List<StringName>::Element *E = anims.front(); E; E = E->next()) {

		String to = E->get();
		TreeItem *blend = blend_editor.tree->create_item(root);
		blend->set_editable(0, false);
		blend->set_editable(1, true);
		blend->set_text(0, to);
		blend->set_cell_mode(1, TreeItem::CELL_MODE_RANGE);
		blend->set_range_config(1, 0, 3600, 0.001);
		blend->set_range(1, player->get_blend_time(current, to));

		i++;
		blend_editor.next->add_item(to, i);
		if (to == player->animation_get_next(current)) {
			blend_editor.next->select(i);
			anim_found = true;
		}
	}

	// make sure we reset it else it becomes out of sync and could contain a deleted animation
	if (!anim_found) {
		blend_editor.next->select(0);
		player->animation_set_next(current, blend_editor.next->get_item_text(0));
	}

	updating_blends = false;
}

void AnimationPlayerEditor::_blend_edited() {

	if (updating_blends)
		return;

	if (animation->get_item_count() == 0)
		return;

	String current = animation->get_item_text(animation->get_selected());

	TreeItem *selected = blend_editor.tree->get_edited();
	if (!selected)
		return;

	updating_blends = true;
	String to = selected->get_text(0);
	float blend_time = selected->get_range(1);
	float prev_blend_time = player->get_blend_time(current, to);

	undo_redo->create_action(TTR("Change Blend Time"));
	undo_redo->add_do_method(player, "set_blend_time", current, to, blend_time);
	undo_redo->add_undo_method(player, "set_blend_time", current, to, prev_blend_time);
	undo_redo->add_do_method(this, "_animation_player_changed", player);
	undo_redo->add_undo_method(this, "_animation_player_changed", player);
	undo_redo->commit_action();
	updating_blends = false;
}

void AnimationPlayerEditor::ensure_visibility() {

	if (player && pin->is_pressed())
		return; // another player is pinned, don't reset

	_animation_edit();
}

Dictionary AnimationPlayerEditor::get_state() const {

	Dictionary d;

	d["visible"] = is_visible_in_tree();
	if (EditorNode::get_singleton()->get_edited_scene() && is_visible_in_tree() && player) {
		d["player"] = EditorNode::get_singleton()->get_edited_scene()->get_path_to(player);
		d["animation"] = player->get_current_animation();
	}

	return d;
}
void AnimationPlayerEditor::set_state(const Dictionary &p_state) {

	if (p_state.has("visible") && p_state["visible"]) {

		if (!EditorNode::get_singleton()->get_edited_scene())
			return;

		Node *n = EditorNode::get_singleton()->get_edited_scene()->get_node(p_state["player"]);
		if (n && n->cast_to<AnimationPlayer>() && EditorNode::get_singleton()->get_editor_selection()->is_selected(n)) {
			player = n->cast_to<AnimationPlayer>();
			_update_player();
			show();
			set_process(true);
			ensure_visibility();
			//EditorNode::get_singleton()->animation_panel_make_visible(true);

			if (p_state.has("animation")) {
				String anim = p_state["animation"];
				_select_anim_by_name(anim);
				_animation_edit();
			}
		}
	}
}

void AnimationPlayerEditor::_animation_resource_edit() {

	if (animation->get_item_count()) {
		String current = animation->get_item_text(animation->get_selected());
		Ref<Animation> anim = player->get_animation(current);
		editor->edit_resource(anim);
	}
}

void AnimationPlayerEditor::_animation_edit() {

	if (animation->get_item_count()) {
		String current = animation->get_item_text(animation->get_selected());
		Ref<Animation> anim = player->get_animation(current);
		key_editor->set_animation(anim);
		Node *root = player->get_node(player->get_root());
		if (root) {
			key_editor->set_root(root);
		}

	} else {

		key_editor->set_animation(Ref<Animation>());
		key_editor->set_root(NULL);
	}
}
void AnimationPlayerEditor::_dialog_action(String p_file) {

	switch (current_option) {
		case RESOURCE_LOAD: {
			ERR_FAIL_COND(!player);

			Ref<Resource> res = ResourceLoader::load(p_file, "Animation");
			ERR_FAIL_COND(res.is_null());
			ERR_FAIL_COND(!res->is_class("Animation"));
			if (p_file.find_last("/") != -1) {

				p_file = p_file.substr(p_file.find_last("/") + 1, p_file.length());
			}
			if (p_file.find_last("\\") != -1) {

				p_file = p_file.substr(p_file.find_last("\\") + 1, p_file.length());
			}

			if (p_file.find(".") != -1)
				p_file = p_file.substr(0, p_file.find("."));

			undo_redo->create_action(TTR("Load Animation"));
			undo_redo->add_do_method(player, "add_animation", p_file, res);
			undo_redo->add_undo_method(player, "remove_animation", p_file);
			if (player->has_animation(p_file)) {
				undo_redo->add_undo_method(player, "add_animation", p_file, player->get_animation(p_file));
			}
			undo_redo->add_do_method(this, "_animation_player_changed", player);
			undo_redo->add_undo_method(this, "_animation_player_changed", player);
			undo_redo->commit_action();
			break;
		}
		case RESOURCE_SAVE: {

			String current = animation->get_item_text(animation->get_selected());
			if (current != "") {
				Ref<Animation> anim = player->get_animation(current);

				ERR_FAIL_COND(!anim->cast_to<Resource>())

				RES current_res = RES(anim->cast_to<Resource>());

				_animation_save_in_path(current_res, p_file);
			}
		}
	}
}

void AnimationPlayerEditor::_scale_changed(const String &p_scale) {

	player->set_speed_scale(p_scale.to_double());
}

void AnimationPlayerEditor::_update_animation() {

	// the purpose of _update_animation is to reflect the current state
	// of the animation player in the current editor..

	updating = true;

	if (player->is_playing()) {

		play->set_pressed(true);
		stop->set_pressed(false);

	} else {

		play->set_pressed(false);
		stop->set_pressed(true);
	}

	scale->set_text(String::num(player->get_speed_scale(), 2));
	String current = player->get_current_animation();

	for (int i = 0; i < animation->get_item_count(); i++) {

		if (animation->get_item_text(i) == current) {
			animation->select(i);
			break;
		}
	}

	updating = false;
}

void AnimationPlayerEditor::_update_player() {

	updating = true;
	List<StringName> animlist;
	if (player)
		player->get_animation_list(&animlist);

	animation->clear();
	if (player)
		nodename->set_text(player->get_name());
	else
		nodename->set_text("<empty>");

	add_anim->set_disabled(player == NULL);
	load_anim->set_disabled(player == NULL);
	stop->set_disabled(animlist.size() == 0);
	play->set_disabled(animlist.size() == 0);
	play_bw->set_disabled(animlist.size() == 0);
	play_bw_from->set_disabled(animlist.size() == 0);
	play_from->set_disabled(animlist.size() == 0);
	autoplay->set_disabled(animlist.size() == 0);
	duplicate_anim->set_disabled(animlist.size() == 0);
	rename_anim->set_disabled(animlist.size() == 0);
	blend_anim->set_disabled(animlist.size() == 0);
	remove_anim->set_disabled(animlist.size() == 0);
	resource_edit_anim->set_disabled(animlist.size() == 0);
	save_anim->set_disabled(animlist.size() == 0);
	tool_anim->set_disabled(player == NULL);

	int active_idx = -1;
	for (List<StringName>::Element *E = animlist.front(); E; E = E->next()) {

		if (player->get_autoplay() == E->get())
			animation->add_icon_item(autoplay_icon, E->get());
		else
			animation->add_item(E->get());

		if (player->get_current_animation() == E->get())
			active_idx = animation->get_item_count() - 1;
	}

	if (!player)
		return;

	updating = false;
	if (active_idx != -1) {
		animation->select(active_idx);
		autoplay->set_pressed(animation->get_item_text(active_idx) == player->get_autoplay());
		_animation_selected(active_idx);

	} else if (animation->get_item_count() > 0) {

		animation->select(0);
		autoplay->set_pressed(animation->get_item_text(0) == player->get_autoplay());
		_animation_selected(0);
	}

	//pause->set_pressed(player->is_paused());

	if (animation->get_item_count()) {
		String current = animation->get_item_text(animation->get_selected());
		Ref<Animation> anim = player->get_animation(current);
		key_editor->set_animation(anim);
		Node *root = player->get_node(player->get_root());
		if (root) {
			key_editor->set_root(root);
		}
	}

	_update_animation();
}

void AnimationPlayerEditor::edit(AnimationPlayer *p_player) {

	if (player && pin->is_pressed())
		return; //ignore, pinned
	player = p_player;

	if (player) {
		_update_player();
		key_editor->show_select_node_warning(false);
	} else {
		key_editor->show_select_node_warning(true);

		//hide();
	}
}

void AnimationPlayerEditor::_animation_duplicate() {

	if (!animation->get_item_count())
		return;

	String current = animation->get_item_text(animation->get_selected());
	Ref<Animation> anim = player->get_animation(current);
	if (!anim.is_valid())
		return;

	Ref<Animation> new_anim = memnew(Animation);
	List<PropertyInfo> plist;
	anim->get_property_list(&plist);
	for (List<PropertyInfo>::Element *E = plist.front(); E; E = E->next()) {

		if (E->get().usage & PROPERTY_USAGE_STORAGE) {

			new_anim->set(E->get().name, anim->get(E->get().name));
		}
	}
	new_anim->set_path("");

	String new_name = current;
	while (player->has_animation(new_name)) {

		new_name = new_name + " (copy)";
	}

	undo_redo->create_action(TTR("Duplicate Animation"));
	undo_redo->add_do_method(player, "add_animation", new_name, new_anim);
	undo_redo->add_undo_method(player, "remove_animation", new_name);
	undo_redo->add_do_method(player, "animation_set_next", new_name, player->animation_get_next(current));
	undo_redo->add_do_method(this, "_animation_player_changed", player);
	undo_redo->add_undo_method(this, "_animation_player_changed", player);
	undo_redo->commit_action();

	for (int i = 0; i < animation->get_item_count(); i++) {

		if (animation->get_item_text(i) == new_name) {

			animation->select(i);
			_animation_selected(i);
			return;
		}
	}
}

void AnimationPlayerEditor::_seek_value_changed(float p_value, bool p_set) {

	if (updating || !player || player->is_playing()) {
		return;
	};

	updating = true;
	String current = player->get_current_animation(); //animation->get_item_text( animation->get_selected() );
	if (current == "" || !player->has_animation(current)) {
		updating = false;
		current = "";
		return;
	};

	Ref<Animation> anim;
	anim = player->get_animation(current);

	float pos = anim->get_length() * (p_value / frame->get_max());
	float step = anim->get_step();
	if (step) {
		pos = Math::stepify(pos, step);
		if (pos < 0)
			pos = 0;
		if (pos >= anim->get_length())
			pos = anim->get_length();
	}

	if (player->is_valid() && !p_set) {
		float cpos = player->get_current_animation_pos();

		player->seek_delta(pos, pos - cpos);
	} else {
		player->seek(pos, true);
	}

	key_editor->set_anim_pos(pos);

	updating = true;
};

void AnimationPlayerEditor::_animation_player_changed(Object *p_pl) {

	if (player == p_pl && is_visible_in_tree()) {

		_update_player();
		if (blend_editor.dialog->is_visible_in_tree())
			_animation_blend(); //update
	}
}

void AnimationPlayerEditor::_list_changed() {

	if (is_visible_in_tree())
		_update_player();
}
#if 0
void AnimationPlayerEditor::_editor_store() {

	if (animation->get_item_count()==0)
		return;
	String current = animation->get_item_text(animation->get_selected());
	Ref<Animation> anim =  player->get_animation(current);

	if (key_editor->get_current_animation()==anim)
		return; //already there


	undo_redo->create_action("Store anim in editor");
	undo_redo->add_do_method(key_editor,"set_animation",anim);
	undo_redo->add_undo_method(key_editor,"remove_animation",anim);
	undo_redo->commit_action();
}

void AnimationPlayerEditor::_editor_load(){

	Ref<Animation> anim = key_editor->get_current_animation();
	if (anim.is_null())
		return;

	String existing = player->find_animation(anim);
	if (existing!="") {
		_select_anim_by_name(existing);
		return; //already has
	}

	int count=1;
	String base=anim->get_name();
	bool noname=false;
	if (base=="") {
		base="New Anim";
		noname=true;
	}

	while(true) {
		String attempt  = base;
		if (count>1)
			attempt+=" ("+itos(count)+")";
		if (player->has_animation(attempt)) {
			count++;
			continue;
		}
		base=attempt;
		break;
	}

	if (noname)
		anim->set_name(base);

	undo_redo->create_action("Add Animation From Editor");
	undo_redo->add_do_method(player,"add_animation",base,anim);
	undo_redo->add_undo_method(player,"remove_animation",base);
	undo_redo->add_do_method(this,"_animation_player_changed",player);
	undo_redo->add_undo_method(this,"_animation_player_changed",player);
	undo_redo->commit_action();

	_select_anim_by_name(base);


}
#endif

void AnimationPlayerEditor::_animation_key_editor_anim_len_changed(float p_len) {

	frame->set_max(p_len);
}

void AnimationPlayerEditor::_animation_key_editor_anim_step_changed(float p_len) {

	if (p_len)
		frame->set_step(p_len);
	else
		frame->set_step(0.00001);
}

void AnimationPlayerEditor::_animation_key_editor_seek(float p_pos, bool p_drag) {

	if (!is_visible_in_tree())
		return;
	if (!player)
		return;

	if (player->is_playing())
		return;

	updating = true;
	frame->set_value(p_pos);
	updating = false;
	_seek_value_changed(p_pos, !p_drag);

	EditorNode::get_singleton()->get_property_editor()->refresh();

	//seekit
}

void AnimationPlayerEditor::_hide_anim_editors() {

	player = NULL;
	hide();
	set_process(false);

	key_editor->set_animation(Ref<Animation>());
	key_editor->set_root(NULL);
	key_editor->show_select_node_warning(true);
	//editor->animation_editor_make_visible(false);
}

void AnimationPlayerEditor::_animation_tool_menu(int p_option) {

	switch (p_option) {

		case TOOL_COPY_ANIM: {

			if (!animation->get_item_count()) {
				error_dialog->set_text(TTR("ERROR: No animation to copy!"));
				error_dialog->popup_centered_minsize();
				return;
			}

			String current = animation->get_item_text(animation->get_selected());
			Ref<Animation> anim = player->get_animation(current);
			//editor->edit_resource(anim);
			EditorSettings::get_singleton()->set_resource_clipboard(anim);

		} break;
		case TOOL_PASTE_ANIM: {

			Ref<Animation> anim = EditorSettings::get_singleton()->get_resource_clipboard();
			if (!anim.is_valid()) {
				error_dialog->set_text(TTR("ERROR: No animation resource on clipboard!"));
				error_dialog->popup_centered_minsize();
				return;
			}

			String name = anim->get_name();
			if (name == "") {
				name = TTR("Pasted Animation");
			}

			int idx = 1;
			String base = name;
			while (player->has_animation(name)) {

				idx++;
				name = base + " " + itos(idx);
			}

			undo_redo->create_action(TTR("Paste Animation"));
			undo_redo->add_do_method(player, "add_animation", name, anim);
			undo_redo->add_undo_method(player, "remove_animation", name);
			undo_redo->add_do_method(this, "_animation_player_changed", player);
			undo_redo->add_undo_method(this, "_animation_player_changed", player);
			undo_redo->commit_action();

			_select_anim_by_name(name);

		} break;
		case TOOL_EDIT_RESOURCE: {

			if (!animation->get_item_count()) {
				error_dialog->set_text(TTR("ERROR: No animation to edit!"));
				error_dialog->popup_centered_minsize();
				return;
			}

			String current = animation->get_item_text(animation->get_selected());
			Ref<Animation> anim = player->get_animation(current);
			editor->edit_resource(anim);

		} break;
	}
}

void AnimationPlayerEditor::_animation_save_menu(int p_option) {

	String current = animation->get_item_text(animation->get_selected());
	if (current != "") {
		Ref<Animation> anim = player->get_animation(current);

		switch (p_option) {
			case ANIM_SAVE:
				_animation_save(anim);
				break;
			case ANIM_SAVE_AS:
				_animation_save_as(anim);
				break;
		}
	}
}

void AnimationPlayerEditor::_unhandled_key_input(const InputEvent &p_ev) {

	if (is_visible_in_tree() && p_ev.type == InputEvent::KEY && p_ev.key.pressed && !p_ev.key.echo && !p_ev.key.mod.alt && !p_ev.key.mod.control && !p_ev.key.mod.meta) {

		switch (p_ev.key.scancode) {

			case KEY_A: {
				if (!p_ev.key.mod.shift)
					_play_bw_from_pressed();
				else
					_play_bw_pressed();
			} break;
			case KEY_S: {
				_stop_pressed();
			} break;
			case KEY_D: {
				if (!p_ev.key.mod.shift)
					_play_from_pressed();
				else
					_play_pressed();
			} break;
		}
	}
}

void AnimationPlayerEditor::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_gui_input"), &AnimationPlayerEditor::_gui_input);
	ClassDB::bind_method(D_METHOD("_node_removed"), &AnimationPlayerEditor::_node_removed);
	ClassDB::bind_method(D_METHOD("_play_pressed"), &AnimationPlayerEditor::_play_pressed);
	ClassDB::bind_method(D_METHOD("_play_from_pressed"), &AnimationPlayerEditor::_play_from_pressed);
	ClassDB::bind_method(D_METHOD("_play_bw_pressed"), &AnimationPlayerEditor::_play_bw_pressed);
	ClassDB::bind_method(D_METHOD("_play_bw_from_pressed"), &AnimationPlayerEditor::_play_bw_from_pressed);
	ClassDB::bind_method(D_METHOD("_stop_pressed"), &AnimationPlayerEditor::_stop_pressed);
	ClassDB::bind_method(D_METHOD("_autoplay_pressed"), &AnimationPlayerEditor::_autoplay_pressed);
	ClassDB::bind_method(D_METHOD("_pause_pressed"), &AnimationPlayerEditor::_pause_pressed);
	ClassDB::bind_method(D_METHOD("_animation_selected"), &AnimationPlayerEditor::_animation_selected);
	ClassDB::bind_method(D_METHOD("_animation_name_edited"), &AnimationPlayerEditor::_animation_name_edited);
	ClassDB::bind_method(D_METHOD("_animation_new"), &AnimationPlayerEditor::_animation_new);
	ClassDB::bind_method(D_METHOD("_animation_rename"), &AnimationPlayerEditor::_animation_rename);
	ClassDB::bind_method(D_METHOD("_animation_load"), &AnimationPlayerEditor::_animation_load);
	ClassDB::bind_method(D_METHOD("_animation_remove"), &AnimationPlayerEditor::_animation_remove);
	ClassDB::bind_method(D_METHOD("_animation_remove_confirmed"), &AnimationPlayerEditor::_animation_remove_confirmed);
	ClassDB::bind_method(D_METHOD("_animation_blend"), &AnimationPlayerEditor::_animation_blend);
	ClassDB::bind_method(D_METHOD("_animation_edit"), &AnimationPlayerEditor::_animation_edit);
	ClassDB::bind_method(D_METHOD("_animation_resource_edit"), &AnimationPlayerEditor::_animation_resource_edit);
	ClassDB::bind_method(D_METHOD("_dialog_action"), &AnimationPlayerEditor::_dialog_action);
	ClassDB::bind_method(D_METHOD("_seek_value_changed"), &AnimationPlayerEditor::_seek_value_changed, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("_animation_player_changed"), &AnimationPlayerEditor::_animation_player_changed);
	ClassDB::bind_method(D_METHOD("_blend_edited"), &AnimationPlayerEditor::_blend_edited);
	//ClassDB::bind_method(D_METHOD("_seek_frame_changed"),&AnimationPlayerEditor::_seek_frame_changed);
	ClassDB::bind_method(D_METHOD("_scale_changed"), &AnimationPlayerEditor::_scale_changed);
	//ClassDB::bind_method(D_METHOD("_editor_store_all"),&AnimationPlayerEditor::_editor_store_all);
	//ClassDB::bind_method(D_METHOD("_editor_load_all"),&AnimationPlayerEditor::_editor_load_all);
	ClassDB::bind_method(D_METHOD("_list_changed"), &AnimationPlayerEditor::_list_changed);
	ClassDB::bind_method(D_METHOD("_animation_key_editor_seek"), &AnimationPlayerEditor::_animation_key_editor_seek);
	ClassDB::bind_method(D_METHOD("_animation_key_editor_anim_len_changed"), &AnimationPlayerEditor::_animation_key_editor_anim_len_changed);
	ClassDB::bind_method(D_METHOD("_animation_key_editor_anim_step_changed"), &AnimationPlayerEditor::_animation_key_editor_anim_step_changed);
	ClassDB::bind_method(D_METHOD("_hide_anim_editors"), &AnimationPlayerEditor::_hide_anim_editors);
	ClassDB::bind_method(D_METHOD("_animation_duplicate"), &AnimationPlayerEditor::_animation_duplicate);
	ClassDB::bind_method(D_METHOD("_blend_editor_next_changed"), &AnimationPlayerEditor::_blend_editor_next_changed);
	ClassDB::bind_method(D_METHOD("_unhandled_key_input"), &AnimationPlayerEditor::_unhandled_key_input);
	ClassDB::bind_method(D_METHOD("_animation_tool_menu"), &AnimationPlayerEditor::_animation_tool_menu);
	ClassDB::bind_method(D_METHOD("_animation_save_menu"), &AnimationPlayerEditor::_animation_save_menu);
}

AnimationPlayerEditor *AnimationPlayerEditor::singleton = NULL;

AnimationPlayer *AnimationPlayerEditor::get_player() const {

	return player;
}
AnimationPlayerEditor::AnimationPlayerEditor(EditorNode *p_editor) {
	editor = p_editor;
	singleton = this;

	updating = false;

	set_focus_mode(FOCUS_ALL);

	player = NULL;
	add_style_override("panel", get_stylebox("panel", "Panel"));

	Label *l;

	/*l= memnew( Label );
	l->set_text("Animation Player:");
	add_child(l);*/

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);

	play_bw_from = memnew(ToolButton);
	play_bw_from->set_tooltip(TTR("Play selected animation backwards from current pos. (A)"));
	hb->add_child(play_bw_from);

	play_bw = memnew(ToolButton);
	play_bw->set_tooltip(TTR("Play selected animation backwards from end. (Shift+A)"));
	hb->add_child(play_bw);

	stop = memnew(ToolButton);
	stop->set_toggle_mode(true);
	hb->add_child(stop);
	stop->set_tooltip(TTR("Stop animation playback. (S)"));

	play = memnew(ToolButton);
	play->set_tooltip(TTR("Play selected animation from start. (Shift+D)"));
	hb->add_child(play);

	play_from = memnew(ToolButton);
	play_from->set_tooltip(TTR("Play selected animation from current pos. (D)"));
	hb->add_child(play_from);

	//pause = memnew( Button );
	//pause->set_toggle_mode(true);
	//hb->add_child(pause);

	frame = memnew(SpinBox);
	hb->add_child(frame);
	frame->set_custom_minimum_size(Size2(60, 0));
	frame->set_stretch_ratio(2);
	frame->set_tooltip(TTR("Animation position (in seconds)."));

	hb->add_child(memnew(VSeparator));

	scale = memnew(LineEdit);
	hb->add_child(scale);
	scale->set_h_size_flags(SIZE_EXPAND_FILL);
	scale->set_stretch_ratio(1);
	scale->set_tooltip(TTR("Scale animation playback globally for the node."));
	scale->hide();

	add_anim = memnew(ToolButton);
	ED_SHORTCUT("animation_player_editor/add_animation", TTR("Create new animation in player."));
	add_anim->set_shortcut(ED_GET_SHORTCUT("animation_player_editor/add_animation"));
	add_anim->set_tooltip(TTR("Create new animation in player."));

	hb->add_child(add_anim);

	load_anim = memnew(ToolButton);
	ED_SHORTCUT("animation_player_editor/load_from_disk", TTR("Load animation from disk."));
	add_anim->set_shortcut(ED_GET_SHORTCUT("animation_player_editor/load_from_disk"));
	load_anim->set_tooltip(TTR("Load an animation from disk."));
	hb->add_child(load_anim);

	save_anim = memnew(MenuButton);
	save_anim->set_tooltip(TTR("Save the current animation"));
	save_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/save", TTR("Save")), ANIM_SAVE);
	save_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/save_as", TTR("Save As")), ANIM_SAVE_AS);
	save_anim->set_focus_mode(Control::FOCUS_NONE);
	hb->add_child(save_anim);

	accept = memnew(AcceptDialog);
	add_child(accept);
	accept->connect("confirmed", this, "_menu_confirm_current");

	delete_dialog = memnew(ConfirmationDialog);
	add_child(delete_dialog);
	delete_dialog->connect("confirmed", this, "_animation_remove_confirmed");

	duplicate_anim = memnew(ToolButton);
	hb->add_child(duplicate_anim);
	ED_SHORTCUT("animation_player_editor/duplicate_animation", TTR("Duplicate Animation"));
	duplicate_anim->set_shortcut(ED_GET_SHORTCUT("animation_player_editor/duplicate_animation"));
	duplicate_anim->set_tooltip(TTR("Duplicate Animation"));

	rename_anim = memnew(ToolButton);
	hb->add_child(rename_anim);
	ED_SHORTCUT("animation_player_editor/rename_animation", TTR("Rename Animation"));
	rename_anim->set_shortcut(ED_GET_SHORTCUT("animation_player_editor/rename_animation"));
	rename_anim->set_tooltip(TTR("Rename Animation"));

	remove_anim = memnew(ToolButton);
	hb->add_child(remove_anim);
	ED_SHORTCUT("animation_player_editor/remove_animation", TTR("Remove Animation"));
	remove_anim->set_shortcut(ED_GET_SHORTCUT("animation_player_editor/remove_animation"));
	remove_anim->set_tooltip(TTR("Remove Animation"));

	animation = memnew(OptionButton);
	hb->add_child(animation);
	animation->set_h_size_flags(SIZE_EXPAND_FILL);
	animation->set_tooltip(TTR("Display list of animations in player."));
	animation->set_clip_text(true);

	autoplay = memnew(ToolButton);
	hb->add_child(autoplay);
	autoplay->set_tooltip(TTR("Autoplay on Load"));

	blend_anim = memnew(ToolButton);
	hb->add_child(blend_anim);
	blend_anim->set_tooltip(TTR("Edit Target Blend Times"));

	tool_anim = memnew(MenuButton);
	//tool_anim->set_flat(false);
	tool_anim->set_tooltip(TTR("Animation Tools"));
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/copy_animation", TTR("Copy Animation")), TOOL_COPY_ANIM);
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/paste_animation", TTR("Paste Animation")), TOOL_PASTE_ANIM);
	//tool_anim->get_popup()->add_separator();
	//tool_anim->get_popup()->add_item("Edit Anim Resource",TOOL_PASTE_ANIM);
	hb->add_child(tool_anim);

	nodename = memnew(Button);
	hb->add_child(nodename);
	pin = memnew(ToolButton);
	pin->set_toggle_mode(true);
	hb->add_child(pin);

	resource_edit_anim = memnew(Button);
	hb->add_child(resource_edit_anim);
	resource_edit_anim->hide();

	file = memnew(EditorFileDialog);
	add_child(file);

	name_dialog = memnew(ConfirmationDialog);
	name_dialog->set_title(TTR("Create New Animation"));
	name_dialog->set_hide_on_ok(false);
	add_child(name_dialog);
	name = memnew(LineEdit);
	name_dialog->add_child(name);
	name->set_pos(Point2(18, 30));
	name->set_anchor_and_margin(MARGIN_RIGHT, ANCHOR_END, 10);
	name_dialog->register_text_enter(name);

	l = memnew(Label);
	l->set_text(TTR("Animation Name:"));
	l->set_pos(Point2(10, 10));

	name_dialog->add_child(l);
	name_title = l;

	error_dialog = memnew(ConfirmationDialog);
	error_dialog->get_ok()->set_text(TTR("Close"));
	//error_dialog->get_cancel()->set_text("Close");
	error_dialog->set_text(TTR("Error!"));
	add_child(error_dialog);

	name_dialog->connect("confirmed", this, "_animation_name_edited");

	blend_editor.dialog = memnew(AcceptDialog);
	add_child(blend_editor.dialog);
	blend_editor.dialog->get_ok()->set_text(TTR("Close"));
	blend_editor.dialog->set_hide_on_ok(true);
	VBoxContainer *blend_vb = memnew(VBoxContainer);
	blend_editor.dialog->add_child(blend_vb);
	//blend_editor.dialog->set_child_rect(blend_vb);
	blend_editor.tree = memnew(Tree);
	blend_editor.tree->set_columns(2);
	blend_vb->add_margin_child(TTR("Blend Times:"), blend_editor.tree, true);
	blend_editor.next = memnew(OptionButton);
	blend_vb->add_margin_child(TTR("Next (Auto Queue):"), blend_editor.next);
	blend_editor.dialog->set_title(TTR("Cross-Animation Blend Times"));
	updating_blends = false;

	blend_editor.tree->connect("item_edited", this, "_blend_edited");

	autoplay->connect("pressed", this, "_autoplay_pressed");
	autoplay->set_toggle_mode(true);
	play->connect("pressed", this, "_play_pressed");
	play_from->connect("pressed", this, "_play_from_pressed");
	play_bw->connect("pressed", this, "_play_bw_pressed");
	play_bw_from->connect("pressed", this, "_play_bw_from_pressed");
	stop->connect("pressed", this, "_stop_pressed");
	//pause->connect("pressed", this,"_pause_pressed");
	add_anim->connect("pressed", this, "_animation_new");
	rename_anim->connect("pressed", this, "_animation_rename");
	load_anim->connect("pressed", this, "_animation_load");
	duplicate_anim->connect("pressed", this, "_animation_duplicate");
	//frame->connect("text_entered", this,"_seek_frame_changed");

	blend_anim->connect("pressed", this, "_animation_blend");
	remove_anim->connect("pressed", this, "_animation_remove");
	animation->connect("item_selected", this, "_animation_selected", Vector<Variant>(), true);
	resource_edit_anim->connect("pressed", this, "_animation_resource_edit");
	file->connect("file_selected", this, "_dialog_action");
	frame->connect("value_changed", this, "_seek_value_changed", Vector<Variant>(), true);
	scale->connect("text_entered", this, "_scale_changed", Vector<Variant>(), true);

	renaming = false;
	last_active = false;

	set_process_unhandled_key_input(true);

	key_editor = memnew(AnimationKeyEditor);
	add_child(key_editor);
	add_constant_override("separation", get_constant("separation", "VBoxContainer"));
	key_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	key_editor->connect("timeline_changed", this, "_animation_key_editor_seek");
	key_editor->connect("animation_len_changed", this, "_animation_key_editor_anim_len_changed");
	key_editor->connect("animation_step_changed", this, "_animation_key_editor_anim_step_changed");

	_update_player();
}

void AnimationPlayerEditorPlugin::edit(Object *p_object) {

	anim_editor->set_undo_redo(&get_undo_redo());
	if (!p_object)
		return;
	anim_editor->edit(p_object->cast_to<AnimationPlayer>());
}

bool AnimationPlayerEditorPlugin::handles(Object *p_object) const {

	return p_object->is_class("AnimationPlayer");
}

void AnimationPlayerEditorPlugin::make_visible(bool p_visible) {

	if (p_visible) {

		editor->make_bottom_panel_item_visible(anim_editor);
		anim_editor->set_process(true);
		anim_editor->ensure_visibility();
		//editor->animation_panel_make_visible(true);
	} else {

		//anim_editor->hide();
		//anim_editor->set_idle_process(false);
	}
}

AnimationPlayerEditorPlugin::AnimationPlayerEditorPlugin(EditorNode *p_node) {

	editor = p_node;
	anim_editor = memnew(AnimationPlayerEditor(editor));
	anim_editor->set_undo_redo(editor->get_undo_redo());

	editor->add_bottom_panel_item(TTR("Animation"), anim_editor);
	/*
	editor->get_viewport()->add_child(anim_editor);
	anim_editor->set_area_as_parent_rect();
	anim_editor->set_anchor( MARGIN_TOP, Control::ANCHOR_END);
	anim_editor->set_margin( MARGIN_TOP, 75 );
	anim_editor->set_anchor( MARGIN_RIGHT, Control::ANCHOR_END);
	anim_editor->set_margin( MARGIN_RIGHT, 0 );*/
	anim_editor->hide();
}

AnimationPlayerEditorPlugin::~AnimationPlayerEditorPlugin() {
}
