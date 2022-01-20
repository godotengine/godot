/*************************************************************************/
/*  animation_player_editor_plugin.cpp                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/config/project_settings.h"
#include "core/input/input.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/keyboard.h"
#include "editor/animation_track_editor.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/plugins/canvas_item_editor_plugin.h" // For onion skinning.
#include "editor/plugins/node_3d_editor_plugin.h" // For onion skinning.
#include "scene/main/window.h"
#include "scene/resources/animation.h"
#include "scene/scene_string_names.h"
#include "servers/rendering_server.h"

void AnimationPlayerEditor::_node_removed(Node *p_node) {
	if (player && player == p_node) {
		player = nullptr;

		set_process(false);

		track_editor->set_animation(Ref<Animation>());
		track_editor->set_root(nullptr);
		track_editor->show_select_node_warning(true);
		_update_player();
	}
}

void AnimationPlayerEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PROCESS: {
			if (!player) {
				return;
			}

			updating = true;

			if (player->is_playing()) {
				{
					String animname = player->get_assigned_animation();

					if (player->has_animation(animname)) {
						Ref<Animation> anim = player->get_animation(animname);
						if (!anim.is_null()) {
							frame->set_max((double)anim->get_length());
						}
					}
				}
				frame->set_value(player->get_current_animation_position());
				track_editor->set_anim_pos(player->get_current_animation_position());

			} else if (!player->is_valid()) {
				// Reset timeline when the player has been stopped externally
				frame->set_value(0);
			} else if (last_active) {
				// Need the last frame after it stopped.
				frame->set_value(player->get_current_animation_position());
			}

			last_active = player->is_playing();
			updating = false;
		} break;
		case NOTIFICATION_ENTER_TREE: {
			tool_anim->get_popup()->connect("id_pressed", callable_mp(this, &AnimationPlayerEditor::_animation_tool_menu));

			onion_skinning->get_popup()->connect("id_pressed", callable_mp(this, &AnimationPlayerEditor::_onion_skinning_menu));

			blend_editor.next->connect("item_selected", callable_mp(this, &AnimationPlayerEditor::_blend_editor_next_changed));

			get_tree()->connect("node_removed", callable_mp(this, &AnimationPlayerEditor::_node_removed));

			add_theme_style_override("panel", editor->get_gui_base()->get_theme_stylebox(SNAME("panel"), SNAME("Panel")));
		} break;
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
			add_theme_style_override("panel", editor->get_gui_base()->get_theme_stylebox(SNAME("panel"), SNAME("Panel")));
		} break;
		case NOTIFICATION_TRANSLATION_CHANGED:
		case NOTIFICATION_LAYOUT_DIRECTION_CHANGED:
		case NOTIFICATION_THEME_CHANGED: {
			autoplay->set_icon(get_theme_icon(SNAME("AutoPlay"), SNAME("EditorIcons")));

			play->set_icon(get_theme_icon(SNAME("PlayStart"), SNAME("EditorIcons")));
			play_from->set_icon(get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
			play_bw->set_icon(get_theme_icon(SNAME("PlayStartBackwards"), SNAME("EditorIcons")));
			play_bw_from->set_icon(get_theme_icon(SNAME("PlayBackwards"), SNAME("EditorIcons")));

			autoplay_icon = get_theme_icon(SNAME("AutoPlay"), SNAME("EditorIcons"));
			reset_icon = get_theme_icon(SNAME("Reload"), SNAME("EditorIcons"));
			{
				Ref<Image> autoplay_img = autoplay_icon->get_image();
				Ref<Image> reset_img = reset_icon->get_image();
				Ref<Image> autoplay_reset_img;
				Size2 icon_size = autoplay_img->get_size();
				autoplay_reset_img.instantiate();
				autoplay_reset_img->create(icon_size.x * 2, icon_size.y, false, autoplay_img->get_format());
				autoplay_reset_img->blit_rect(autoplay_img, Rect2(Point2(), icon_size), Point2());
				autoplay_reset_img->blit_rect(reset_img, Rect2(Point2(), icon_size), Point2(icon_size.x, 0));
				autoplay_reset_icon.instantiate();
				autoplay_reset_icon->create_from_image(autoplay_reset_img);
			}
			stop->set_icon(get_theme_icon(SNAME("Stop"), SNAME("EditorIcons")));

			onion_toggle->set_icon(get_theme_icon(SNAME("Onion"), SNAME("EditorIcons")));
			onion_skinning->set_icon(get_theme_icon(SNAME("GuiTabMenuHl"), SNAME("EditorIcons")));

			pin->set_icon(get_theme_icon(SNAME("Pin"), SNAME("EditorIcons")));

			tool_anim->add_theme_style_override("normal", get_theme_stylebox(SNAME("normal"), SNAME("Button")));
			track_editor->get_edit_menu()->add_theme_style_override("normal", get_theme_stylebox(SNAME("normal"), SNAME("Button")));

#define ITEM_ICON(m_item, m_icon) tool_anim->get_popup()->set_item_icon(tool_anim->get_popup()->get_item_index(m_item), get_theme_icon(SNAME(m_icon), SNAME("EditorIcons")))

			ITEM_ICON(TOOL_NEW_ANIM, "New");
			ITEM_ICON(TOOL_LOAD_ANIM, "Load");
			ITEM_ICON(TOOL_SAVE_ANIM, "Save");
			ITEM_ICON(TOOL_SAVE_AS_ANIM, "Save");
			ITEM_ICON(TOOL_DUPLICATE_ANIM, "Duplicate");
			ITEM_ICON(TOOL_RENAME_ANIM, "Rename");
			ITEM_ICON(TOOL_EDIT_TRANSITIONS, "Blend");
			ITEM_ICON(TOOL_EDIT_RESOURCE, "Edit");
			ITEM_ICON(TOOL_REMOVE_ANIM, "Remove");
		} break;
	}
}

void AnimationPlayerEditor::_autoplay_pressed() {
	if (updating) {
		return;
	}
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

	if (!current.is_empty()) {
		if (current == player->get_assigned_animation()) {
			player->stop(); //so it won't blend with itself
		}
		player->play(current);
	}

	//unstop
	stop->set_pressed(false);
}

void AnimationPlayerEditor::_play_from_pressed() {
	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {
		current = animation->get_item_text(animation->get_selected());
	}

	if (!current.is_empty()) {
		float time = player->get_current_animation_position();

		if (current == player->get_assigned_animation() && player->is_playing()) {
			player->stop(); //so it won't blend with itself
		}

		player->play(current);
		player->seek(time);
	}

	//unstop
	stop->set_pressed(false);
}

void AnimationPlayerEditor::_play_bw_pressed() {
	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {
		current = animation->get_item_text(animation->get_selected());
	}

	if (!current.is_empty()) {
		if (current == player->get_assigned_animation()) {
			player->stop(); //so it won't blend with itself
		}
		player->play(current, -1, -1, true);
	}

	//unstop
	stop->set_pressed(false);
}

void AnimationPlayerEditor::_play_bw_from_pressed() {
	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {
		current = animation->get_item_text(animation->get_selected());
	}

	if (!current.is_empty()) {
		float time = player->get_current_animation_position();
		if (current == player->get_assigned_animation()) {
			player->stop(); //so it won't blend with itself
		}

		player->play(current, -1, -1, true);
		player->seek(time);
	}

	//unstop
	stop->set_pressed(false);
}

void AnimationPlayerEditor::_stop_pressed() {
	if (!player) {
		return;
	}

	player->stop(false);
	play->set_pressed(false);
	stop->set_pressed(true);
}

void AnimationPlayerEditor::_animation_selected(int p_which) {
	if (updating) {
		return;
	}
	// when selecting an animation, the idea is that the only interesting behavior
	// ui-wise is that it should play/blend the next one if currently playing
	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {
		current = animation->get_item_text(animation->get_selected());
	}

	if (!current.is_empty()) {
		player->set_assigned_animation(current);

		Ref<Animation> anim = player->get_animation(current);
		{
			track_editor->set_animation(anim);
			Node *root = player->get_node(player->get_root());
			if (root) {
				track_editor->set_root(root);
			}
		}
		frame->set_max((double)anim->get_length());

	} else {
		track_editor->set_animation(Ref<Animation>());
		track_editor->set_root(nullptr);
	}

	autoplay->set_pressed(current == player->get_autoplay());

	AnimationPlayerEditor::get_singleton()->get_track_editor()->update_keying();
	_animation_key_editor_seek(timeline_position, false);
}

void AnimationPlayerEditor::_animation_new() {
	renaming = false;
	name_title->set_text(TTR("New Animation Name:"));

	int count = 1;
	String base = TTR("New Anim");
	while (true) {
		String attempt = base;
		if (count > 1) {
			attempt += " (" + itos(count) + ")";
		}
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
	if (animation->get_item_count() == 0) {
		return;
	}
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
	file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
	file->clear_filters();
	List<String> extensions;

	ResourceLoader::get_recognized_extensions_for_type("Animation", &extensions);
	for (const String &E : extensions) {
		file->add_filter("*." + E + " ; " + E.to_upper());
	}

	file->popup_file_dialog();
}

void AnimationPlayerEditor::_animation_save_in_path(const Ref<Resource> &p_resource, const String &p_path) {
	int flg = 0;
	if (EditorSettings::get_singleton()->get("filesystem/on_save/compress_binary_resources")) {
		flg |= ResourceSaver::FLAG_COMPRESS;
	}

	String path = ProjectSettings::get_singleton()->localize_path(p_path);
	Error err = ResourceSaver::save(path, p_resource, flg | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS);

	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Error saving resource!"));
		return;
	}

	((Resource *)p_resource.ptr())->set_path(path);
	editor->emit_signal(SNAME("resource_saved"), p_resource);
}

void AnimationPlayerEditor::_animation_save(const Ref<Resource> &p_resource) {
	if (p_resource->get_path().is_resource_file()) {
		_animation_save_in_path(p_resource, p_resource->get_path());
	} else {
		_animation_save_as(p_resource);
	}
}

void AnimationPlayerEditor::_animation_save_as(const Ref<Resource> &p_resource) {
	file->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);

	List<String> extensions;
	ResourceSaver::get_recognized_extensions(p_resource, &extensions);
	file->clear_filters();
	for (int i = 0; i < extensions.size(); i++) {
		file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
	}

	String path;
	//file->set_current_path(current_path);
	if (!p_resource->get_path().is_empty()) {
		path = p_resource->get_path();
		if (extensions.size()) {
			if (extensions.find(p_resource->get_path().get_extension().to_lower()) == nullptr) {
				path = p_resource->get_path().get_base_dir() + p_resource->get_name() + "." + extensions.front()->get();
			}
		}
	} else {
		if (extensions.size()) {
			if (!p_resource->get_name().is_empty()) {
				path = p_resource->get_name() + "." + extensions.front()->get().to_lower();
			} else {
				String resource_name_snake_case = p_resource->get_class().camelcase_to_underscore();
				path = "new_" + resource_name_snake_case + "." + extensions.front()->get().to_lower();
			}
		}
	}
	file->set_current_path(path);
	file->set_title(TTR("Save Resource As..."));
	file->popup_file_dialog();
}

void AnimationPlayerEditor::_animation_remove() {
	if (animation->get_item_count() == 0) {
		return;
	}

	String current = animation->get_item_text(animation->get_selected());

	delete_dialog->set_text(vformat(TTR("Delete Animation '%s'?"), current));
	delete_dialog->popup_centered();
}

void AnimationPlayerEditor::_animation_remove_confirmed() {
	String current = animation->get_item_text(animation->get_selected());
	Ref<Animation> anim = player->get_animation(current);

	undo_redo->create_action(TTR("Remove Animation"));
	if (player->get_autoplay() == current) {
		undo_redo->add_do_method(player, "set_autoplay", "");
		undo_redo->add_undo_method(player, "set_autoplay", current);
		// Avoid having the autoplay icon linger around if there is only one animation in the player.
		undo_redo->add_do_method(this, "_animation_player_changed", player);
	}
	undo_redo->add_do_method(player, "remove_animation", current);
	undo_redo->add_undo_method(player, "add_animation", current, anim);
	undo_redo->add_do_method(this, "_animation_player_changed", player);
	undo_redo->add_undo_method(this, "_animation_player_changed", player);
	if (animation->get_item_count() == 1) {
		undo_redo->add_do_method(this, "_stop_onion_skinning");
		undo_redo->add_undo_method(this, "_start_onion_skinning");
	}
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

double AnimationPlayerEditor::_get_editor_step() const {
	// Returns the effective snapping value depending on snapping modifiers, or 0 if snapping is disabled.
	if (track_editor->is_snap_enabled()) {
		const String current = player->get_assigned_animation();
		const Ref<Animation> anim = player->get_animation(current);
		ERR_FAIL_COND_V(!anim.is_valid(), 0.0);

		// Use more precise snapping when holding Shift
		return Input::get_singleton()->is_key_pressed(Key::SHIFT) ? anim->get_step() * 0.25 : anim->get_step();
	}

	return 0.0;
}

void AnimationPlayerEditor::_animation_name_edited() {
	player->stop();

	String new_name = name->get_text();
	if (new_name.is_empty() || new_name.find(":") != -1 || new_name.find("/") != -1) {
		error_dialog->set_text(TTR("Invalid animation name!"));
		error_dialog->popup_centered();
		return;
	}

	if (renaming && animation->get_item_count() > 0 && animation->get_item_text(animation->get_selected()) == new_name) {
		name_dialog->hide();
		return;
	}

	if (player->has_animation(new_name)) {
		error_dialog->set_text(TTR("Animation name already exists!"));
		error_dialog->popup_centered();
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
		if (animation->get_item_count() == 0) {
			undo_redo->add_do_method(this, "_start_onion_skinning");
			undo_redo->add_undo_method(this, "_stop_onion_skinning");
		}
		undo_redo->commit_action();

		_select_anim_by_name(new_name);
	}

	name_dialog->hide();
}

void AnimationPlayerEditor::_blend_editor_next_changed(const int p_idx) {
	if (animation->get_item_count() == 0) {
		return;
	}

	String current = animation->get_item_text(animation->get_selected());

	undo_redo->create_action(TTR("Blend Next Changed"));
	undo_redo->add_do_method(player, "animation_set_next", current, blend_editor.next->get_item_text(p_idx));
	undo_redo->add_undo_method(player, "animation_set_next", current, player->animation_get_next(current));
	undo_redo->add_do_method(this, "_animation_player_changed", player);
	undo_redo->add_undo_method(this, "_animation_player_changed", player);
	undo_redo->commit_action();
}

void AnimationPlayerEditor::_animation_blend() {
	if (updating_blends) {
		return;
	}

	blend_editor.tree->clear();

	if (animation->get_item_count() == 0) {
		return;
	}

	String current = animation->get_item_text(animation->get_selected());

	blend_editor.dialog->popup_centered(Size2(400, 400) * EDSCALE);

	blend_editor.tree->set_hide_root(true);
	blend_editor.tree->set_column_expand_ratio(0, 10);
	blend_editor.tree->set_column_clip_content(0, true);
	blend_editor.tree->set_column_expand_ratio(1, 3);
	blend_editor.tree->set_column_clip_content(1, true);

	List<StringName> anims;
	player->get_animation_list(&anims);
	TreeItem *root = blend_editor.tree->create_item();
	updating_blends = true;

	int i = 0;
	bool anim_found = false;
	blend_editor.next->clear();
	blend_editor.next->add_item("", i);

	for (const StringName &to : anims) {
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
	if (updating_blends) {
		return;
	}

	if (animation->get_item_count() == 0) {
		return;
	}

	String current = animation->get_item_text(animation->get_selected());

	TreeItem *selected = blend_editor.tree->get_edited();
	if (!selected) {
		return;
	}

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
	if (player && pin->is_pressed()) {
		return; // another player is pinned, don't reset
	}

	_animation_edit();
}

Dictionary AnimationPlayerEditor::get_state() const {
	Dictionary d;

	d["visible"] = is_visible_in_tree();
	if (EditorNode::get_singleton()->get_edited_scene() && is_visible_in_tree() && player) {
		d["player"] = EditorNode::get_singleton()->get_edited_scene()->get_path_to(player);
		d["animation"] = player->get_assigned_animation();
		d["track_editor_state"] = track_editor->get_state();
	}

	return d;
}

void AnimationPlayerEditor::set_state(const Dictionary &p_state) {
	if (!p_state.has("visible") || !p_state["visible"]) {
		return;
	}
	if (!EditorNode::get_singleton()->get_edited_scene()) {
		return;
	}

	if (p_state.has("player")) {
		Node *n = EditorNode::get_singleton()->get_edited_scene()->get_node(p_state["player"]);
		if (Object::cast_to<AnimationPlayer>(n) && EditorNode::get_singleton()->get_editor_selection()->is_selected(n)) {
			player = Object::cast_to<AnimationPlayer>(n);
			_update_player();
			editor->make_bottom_panel_item_visible(this);
			set_process(true);
			ensure_visibility();

			if (p_state.has("animation")) {
				String anim = p_state["animation"];
				if (!anim.is_empty() && player->has_animation(anim)) {
					_select_anim_by_name(anim);
					_animation_edit();
				}
			}
		}
	}

	if (p_state.has("track_editor_state")) {
		track_editor->set_state(p_state["track_editor_state"]);
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
		track_editor->set_animation(anim);

		Node *root = player->get_node(player->get_root());
		if (root) {
			track_editor->set_root(root);
		}
	} else {
		track_editor->set_animation(Ref<Animation>());
		track_editor->set_root(nullptr);
	}
}

void AnimationPlayerEditor::_save_animation(String p_file) {
	String current = animation->get_item_text(animation->get_selected());
	if (!current.is_empty()) {
		Ref<Animation> anim = player->get_animation(current);

		ERR_FAIL_COND(!Object::cast_to<Resource>(*anim));

		RES current_res = RES(Object::cast_to<Resource>(*anim));

		_animation_save_in_path(current_res, p_file);
	}
}

void AnimationPlayerEditor::_load_animations(Vector<String> p_files) {
	ERR_FAIL_COND(!player);

	for (int i = 0; i < p_files.size(); i++) {
		String file = p_files[i];

		Ref<Resource> res = ResourceLoader::load(file, "Animation");
		ERR_FAIL_COND_MSG(res.is_null(), "Cannot load Animation from file '" + file + "'.");
		ERR_FAIL_COND_MSG(!res->is_class("Animation"), "Loaded resource from file '" + file + "' is not Animation.");
		if (file.rfind("/") != -1) {
			file = file.substr(file.rfind("/") + 1, file.length());
		}
		if (file.rfind("\\") != -1) {
			file = file.substr(file.rfind("\\") + 1, file.length());
		}

		if (file.find(".") != -1) {
			file = file.substr(0, file.find("."));
		}

		undo_redo->create_action(TTR("Load Animation"));
		undo_redo->add_do_method(player, "add_animation", file, res);
		undo_redo->add_undo_method(player, "remove_animation", file);
		if (player->has_animation(file)) {
			undo_redo->add_undo_method(player, "add_animation", file, player->get_animation(file));
		}
		undo_redo->add_do_method(this, "_animation_player_changed", player);
		undo_redo->add_undo_method(this, "_animation_player_changed", player);
		undo_redo->commit_action();
	}
}

void AnimationPlayerEditor::_scale_changed(const String &p_scale) {
	player->set_speed_scale(p_scale.to_float());
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
	String current = player->get_assigned_animation();

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
	if (player) {
		player->get_animation_list(&animlist);
	}

	animation->clear();

#define ITEM_DISABLED(m_item, m_disabled) tool_anim->get_popup()->set_item_disabled(tool_anim->get_popup()->get_item_index(m_item), m_disabled)

	ITEM_DISABLED(TOOL_SAVE_ANIM, animlist.size() == 0);
	ITEM_DISABLED(TOOL_SAVE_AS_ANIM, animlist.size() == 0);
	ITEM_DISABLED(TOOL_DUPLICATE_ANIM, animlist.size() == 0);
	ITEM_DISABLED(TOOL_RENAME_ANIM, animlist.size() == 0);
	ITEM_DISABLED(TOOL_EDIT_TRANSITIONS, animlist.size() == 0);
	ITEM_DISABLED(TOOL_COPY_ANIM, animlist.size() == 0);
	ITEM_DISABLED(TOOL_REMOVE_ANIM, animlist.size() == 0);

	stop->set_disabled(animlist.size() == 0);
	play->set_disabled(animlist.size() == 0);
	play_bw->set_disabled(animlist.size() == 0);
	play_bw_from->set_disabled(animlist.size() == 0);
	play_from->set_disabled(animlist.size() == 0);
	frame->set_editable(animlist.size() != 0);
	animation->set_disabled(animlist.size() == 0);
	autoplay->set_disabled(animlist.size() == 0);
	tool_anim->set_disabled(player == nullptr);
	onion_toggle->set_disabled(animlist.size() == 0);
	onion_skinning->set_disabled(animlist.size() == 0);
	pin->set_disabled(player == nullptr);

	if (!player) {
		AnimationPlayerEditor::get_singleton()->get_track_editor()->update_keying();
		return;
	}

	int active_idx = -1;
	for (const StringName &E : animlist) {
		Ref<Texture2D> icon;
		if (E == player->get_autoplay()) {
			if (E == SceneStringNames::get_singleton()->RESET) {
				icon = autoplay_reset_icon;
			} else {
				icon = autoplay_icon;
			}
		} else if (E == SceneStringNames::get_singleton()->RESET) {
			icon = reset_icon;
		}
		animation->add_icon_item(icon, E);

		if (player->get_assigned_animation() == E) {
			active_idx = animation->get_item_count() - 1;
		}
	}

	updating = false;
	if (active_idx != -1) {
		animation->select(active_idx);
		autoplay->set_pressed(animation->get_item_text(active_idx) == player->get_autoplay());
		_animation_selected(active_idx);

	} else if (animation->get_item_count() > 0) {
		animation->select(0);
		autoplay->set_pressed(animation->get_item_text(0) == player->get_autoplay());
		_animation_selected(0);
	} else {
		_animation_selected(0);
	}

	if (animation->get_item_count()) {
		String current = animation->get_item_text(animation->get_selected());
		Ref<Animation> anim = player->get_animation(current);
		track_editor->set_animation(anim);
		Node *root = player->get_node(player->get_root());
		if (root) {
			track_editor->set_root(root);
		}
	}

	_update_animation();
}

void AnimationPlayerEditor::edit(AnimationPlayer *p_player) {
	if (player && pin->is_pressed()) {
		return; // Ignore, pinned.
	}
	player = p_player;

	if (player) {
		_update_player();

		if (onion.enabled) {
			if (animation->get_item_count() > 0) {
				_start_onion_skinning();
			} else {
				_stop_onion_skinning();
			}
		}

		track_editor->show_select_node_warning(false);
	} else {
		if (onion.enabled) {
			_stop_onion_skinning();
		}

		track_editor->show_select_node_warning(true);
	}
}

void AnimationPlayerEditor::forward_force_draw_over_viewport(Control *p_overlay) {
	if (!onion.can_overlay) {
		return;
	}

	// Can happen on viewport resize, at least.
	if (!_are_onion_layers_valid()) {
		return;
	}

	RID ci = p_overlay->get_canvas_item();
	Rect2 src_rect = p_overlay->get_global_rect();
	// Re-flip since captures are already flipped.
	src_rect.position.y = onion.capture_size.y - (src_rect.position.y + src_rect.size.y);
	src_rect.size.y *= -1;

	Rect2 dst_rect = Rect2(Point2(), p_overlay->get_size());

	float alpha_step = 1.0 / (onion.steps + 1);

	int cidx = 0;
	if (onion.past) {
		float alpha = 0;
		do {
			alpha += alpha_step;

			if (onion.captures_valid[cidx]) {
				RS::get_singleton()->canvas_item_add_texture_rect_region(
						ci, dst_rect, RS::get_singleton()->viewport_get_texture(onion.captures[cidx]), src_rect, Color(1, 1, 1, alpha));
			}

			cidx++;
		} while (cidx < onion.steps);
	}
	if (onion.future) {
		float alpha = 1;
		int base_cidx = cidx;
		do {
			alpha -= alpha_step;

			if (onion.captures_valid[cidx]) {
				RS::get_singleton()->canvas_item_add_texture_rect_region(
						ci, dst_rect, RS::get_singleton()->viewport_get_texture(onion.captures[cidx]), src_rect, Color(1, 1, 1, alpha));
			}

			cidx++;
		} while (cidx < base_cidx + onion.steps); // In case there's the present capture at the end, skip it.
	}
}

void AnimationPlayerEditor::_animation_duplicate() {
	if (!animation->get_item_count()) {
		return;
	}

	String current = animation->get_item_text(animation->get_selected());
	Ref<Animation> anim = player->get_animation(current);
	if (!anim.is_valid()) {
		return;
	}

	Ref<Animation> new_anim = _animation_clone(anim);
	String new_name = current;
	while (player->has_animation(new_name)) {
		new_name = new_name + " (copy)";
	}
	new_anim->set_name(new_name);

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

Ref<Animation> AnimationPlayerEditor::_animation_clone(Ref<Animation> p_anim) {
	Ref<Animation> new_anim = memnew(Animation);
	List<PropertyInfo> plist;
	p_anim->get_property_list(&plist);

	for (const PropertyInfo &E : plist) {
		if (E.usage & PROPERTY_USAGE_STORAGE) {
			new_anim->set(E.name, p_anim->get(E.name));
		}
	}
	new_anim->set_path("");

	return new_anim;
}

void AnimationPlayerEditor::_animation_paste(Ref<Animation> p_anim) {
	String name = p_anim->get_name();
	if (name.is_empty()) {
		name = TTR("Pasted Animation");
	}

	int idx = 1;
	String base = name;
	while (player->has_animation(name)) {
		idx++;
		name = base + " " + itos(idx);
	}

	undo_redo->create_action(TTR("Paste Animation"));
	undo_redo->add_do_method(player, "add_animation", name, p_anim);
	undo_redo->add_undo_method(player, "remove_animation", name);
	undo_redo->add_do_method(this, "_animation_player_changed", player);
	undo_redo->add_undo_method(this, "_animation_player_changed", player);
	undo_redo->commit_action();

	_select_anim_by_name(name);
}

void AnimationPlayerEditor::_seek_value_changed(float p_value, bool p_set, bool p_timeline_only) {
	if (updating || !player || player->is_playing()) {
		return;
	};

	updating = true;
	String current = player->get_assigned_animation();
	if (current.is_empty() || !player->has_animation(current)) {
		updating = false;
		current = "";
		return;
	};

	Ref<Animation> anim;
	anim = player->get_animation(current);

	float pos = CLAMP((double)anim->get_length() * (p_value / frame->get_max()), 0, (double)anim->get_length());
	if (track_editor->is_snap_enabled()) {
		pos = Math::snapped(pos, _get_editor_step());
	}

	if (!p_timeline_only) {
		if (player->is_valid() && !p_set) {
			float cpos = player->get_current_animation_position();

			player->seek_delta(pos, pos - cpos);
		} else {
			player->stop(true);
			player->seek(pos, true);
		}
	}

	track_editor->set_anim_pos(pos);
};

void AnimationPlayerEditor::_animation_player_changed(Object *p_pl) {
	if (player == p_pl && is_visible_in_tree()) {
		_update_player();
		if (blend_editor.dialog->is_visible()) {
			_animation_blend(); // Update.
		}
	}
}

void AnimationPlayerEditor::_list_changed() {
	if (is_visible_in_tree()) {
		_update_player();
	}
}

void AnimationPlayerEditor::_animation_key_editor_anim_len_changed(float p_len) {
	frame->set_max(p_len);
}

void AnimationPlayerEditor::_animation_key_editor_seek(float p_pos, bool p_drag, bool p_timeline_only) {
	timeline_position = p_pos;

	if (!is_visible_in_tree()) {
		return;
	}

	if (!player) {
		return;
	}

	if (player->is_playing()) {
		return;
	}

	if (!player->has_animation(player->get_assigned_animation())) {
		return;
	}

	updating = true;
	frame->set_value(Math::snapped(p_pos, _get_editor_step()));
	updating = false;
	_seek_value_changed(p_pos, !p_drag, p_timeline_only);
}

void AnimationPlayerEditor::_animation_tool_menu(int p_option) {
	String current;
	if (animation->get_selected() >= 0 && animation->get_selected() < animation->get_item_count()) {
		current = animation->get_item_text(animation->get_selected());
	}

	Ref<Animation> anim;
	if (!current.is_empty()) {
		anim = player->get_animation(current);
	}

	switch (p_option) {
		case TOOL_NEW_ANIM: {
			_animation_new();
		} break;
		case TOOL_LOAD_ANIM: {
			_animation_load();
		} break;
		case TOOL_SAVE_ANIM: {
			if (anim.is_valid()) {
				_animation_save(anim);
			}
		} break;
		case TOOL_SAVE_AS_ANIM: {
			if (anim.is_valid()) {
				_animation_save_as(anim);
			}
		} break;
		case TOOL_DUPLICATE_ANIM: {
			_animation_duplicate();

			[[fallthrough]]; // Allow immediate rename after animation is duplicated
		}
		case TOOL_RENAME_ANIM: {
			_animation_rename();
		} break;
		case TOOL_EDIT_TRANSITIONS: {
			_animation_blend();
		} break;
		case TOOL_REMOVE_ANIM: {
			_animation_remove();
		} break;
		case TOOL_COPY_ANIM: {
			if (!animation->get_item_count()) {
				error_dialog->set_text(TTR("No animation to copy!"));
				error_dialog->popup_centered();
				return;
			}

			String current2 = animation->get_item_text(animation->get_selected());
			Ref<Animation> anim2 = player->get_animation(current2);
			EditorSettings::get_singleton()->set_resource_clipboard(anim2);
		} break;
		case TOOL_PASTE_ANIM: {
			Ref<Animation> anim2 = EditorSettings::get_singleton()->get_resource_clipboard();
			if (!anim2.is_valid()) {
				error_dialog->set_text(TTR("No animation resource in clipboard!"));
				error_dialog->popup_centered();
				return;
			}
			Ref<Animation> new_anim = _animation_clone(anim2);
			_animation_paste(new_anim);
		} break;
		case TOOL_PASTE_ANIM_REF: {
			Ref<Animation> anim2 = EditorSettings::get_singleton()->get_resource_clipboard();
			if (!anim2.is_valid()) {
				error_dialog->set_text(TTR("No animation resource in clipboard!"));
				error_dialog->popup_centered();
				return;
			}

			_animation_paste(anim2);
		} break;
		case TOOL_EDIT_RESOURCE: {
			if (!animation->get_item_count()) {
				error_dialog->set_text(TTR("No animation to edit!"));
				error_dialog->popup_centered();
				return;
			}

			String current2 = animation->get_item_text(animation->get_selected());
			Ref<Animation> anim2 = player->get_animation(current2);
			editor->edit_resource(anim2);
		} break;
	}
}

void AnimationPlayerEditor::_onion_skinning_menu(int p_option) {
	PopupMenu *menu = onion_skinning->get_popup();
	int idx = menu->get_item_index(p_option);

	switch (p_option) {
		case ONION_SKINNING_ENABLE: {
			onion.enabled = !onion.enabled;

			if (onion.enabled) {
				_start_onion_skinning();
			} else {
				_stop_onion_skinning();
			}

		} break;
		case ONION_SKINNING_PAST: {
			// Ensure at least one of past/future is checked.
			onion.past = onion.future ? !onion.past : true;
			menu->set_item_checked(idx, onion.past);
		} break;
		case ONION_SKINNING_FUTURE: {
			// Ensure at least one of past/future is checked.
			onion.future = onion.past ? !onion.future : true;
			menu->set_item_checked(idx, onion.future);
		} break;
		case ONION_SKINNING_1_STEP: // Fall-through.
		case ONION_SKINNING_2_STEPS:
		case ONION_SKINNING_3_STEPS: {
			onion.steps = (p_option - ONION_SKINNING_1_STEP) + 1;
			int one_frame_idx = menu->get_item_index(ONION_SKINNING_1_STEP);
			for (int i = 0; i <= ONION_SKINNING_LAST_STEPS_OPTION - ONION_SKINNING_1_STEP; i++) {
				menu->set_item_checked(one_frame_idx + i, onion.steps == i + 1);
			}
		} break;
		case ONION_SKINNING_DIFFERENCES_ONLY: {
			onion.differences_only = !onion.differences_only;
			menu->set_item_checked(idx, onion.differences_only);
		} break;
		case ONION_SKINNING_FORCE_WHITE_MODULATE: {
			onion.force_white_modulate = !onion.force_white_modulate;
			menu->set_item_checked(idx, onion.force_white_modulate);
		} break;
		case ONION_SKINNING_INCLUDE_GIZMOS: {
			onion.include_gizmos = !onion.include_gizmos;
			menu->set_item_checked(idx, onion.include_gizmos);
		} break;
	}
}

void AnimationPlayerEditor::unhandled_key_input(const Ref<InputEvent> &p_ev) {
	ERR_FAIL_COND(p_ev.is_null());

	Ref<InputEventKey> k = p_ev;
	if (is_visible_in_tree() && k.is_valid() && k->is_pressed() && !k->is_echo() && !k->is_alt_pressed() && !k->is_ctrl_pressed() && !k->is_meta_pressed()) {
		switch (k->get_keycode()) {
			case Key::A: {
				if (!k->is_shift_pressed()) {
					_play_bw_from_pressed();
				} else {
					_play_bw_pressed();
				}
				accept_event();
			} break;
			case Key::S: {
				_stop_pressed();
				accept_event();
			} break;
			case Key::D: {
				if (!k->is_shift_pressed()) {
					_play_from_pressed();
				} else {
					_play_pressed();
				}
				accept_event();
			} break;
			default:
				break;
		}
	}
}

void AnimationPlayerEditor::_editor_visibility_changed() {
	if (is_visible() && animation->get_item_count() > 0) {
		_start_onion_skinning();
	}
}

bool AnimationPlayerEditor::_are_onion_layers_valid() {
	ERR_FAIL_COND_V(!onion.past && !onion.future, false);

	Point2 capture_size = get_tree()->get_root()->get_size();
	return onion.captures.size() == onion.get_needed_capture_count() && onion.capture_size == capture_size;
}

void AnimationPlayerEditor::_allocate_onion_layers() {
	_free_onion_layers();

	int captures = onion.get_needed_capture_count();
	Point2 capture_size = get_tree()->get_root()->get_size();

	onion.captures.resize(captures);
	onion.captures_valid.resize(captures);

	for (int i = 0; i < captures; i++) {
		bool is_present = onion.differences_only && i == captures - 1;

		// Each capture is a viewport with a canvas item attached that renders a full-size rect with the contents of the main viewport.
		onion.captures.write[i] = RS::get_singleton()->viewport_create();

		RS::get_singleton()->viewport_set_size(onion.captures[i], capture_size.width, capture_size.height);
		RS::get_singleton()->viewport_set_update_mode(onion.captures[i], RS::VIEWPORT_UPDATE_ALWAYS);
		RS::get_singleton()->viewport_set_transparent_background(onion.captures[i], !is_present);
		RS::get_singleton()->viewport_attach_canvas(onion.captures[i], onion.capture.canvas);
	}

	// Reset the capture canvas item to the current root viewport texture (defensive).
	RS::get_singleton()->canvas_item_clear(onion.capture.canvas_item);
	RS::get_singleton()->canvas_item_add_texture_rect(onion.capture.canvas_item, Rect2(Point2(), capture_size), get_tree()->get_root()->get_texture()->get_rid());

	onion.capture_size = capture_size;
}

void AnimationPlayerEditor::_free_onion_layers() {
	for (int i = 0; i < onion.captures.size(); i++) {
		if (onion.captures[i].is_valid()) {
			RS::get_singleton()->free(onion.captures[i]);
		}
	}
	onion.captures.clear();
	onion.captures_valid.clear();
}

void AnimationPlayerEditor::_prepare_onion_layers_1() {
	// This would be called per viewport and we want to act once only.
	int64_t frame = get_tree()->get_frame();
	if (frame == onion.last_frame) {
		return;
	}

	if (!onion.enabled || !is_processing() || !is_visible() || !get_player()) {
		_stop_onion_skinning();
		return;
	}

	onion.last_frame = frame;

	// Refresh viewports with no onion layers overlaid.
	onion.can_overlay = false;
	plugin->update_overlays();

	if (player->is_playing()) {
		return;
	}

	// And go to next step afterwards.
	call_deferred(SNAME("_prepare_onion_layers_2"));
}

void AnimationPlayerEditor::_prepare_onion_layers_1_deferred() {
	call_deferred(SNAME("_prepare_onion_layers_1"));
}

void AnimationPlayerEditor::_prepare_onion_layers_2() {
	Ref<Animation> anim = player->get_animation(player->get_assigned_animation());
	if (!anim.is_valid()) {
		return;
	}

	if (!_are_onion_layers_valid()) {
		_allocate_onion_layers();
	}

	// Hide superfluous elements that would make the overlay unnecessary cluttered.
	Dictionary canvas_edit_state;
	Dictionary spatial_edit_state;
	if (Node3DEditor::get_singleton()->is_visible()) {
		// 3D
		spatial_edit_state = Node3DEditor::get_singleton()->get_state();
		Dictionary new_state = spatial_edit_state.duplicate();
		new_state["show_grid"] = false;
		new_state["show_origin"] = false;
		Array orig_vp = spatial_edit_state["viewports"];
		Array vp;
		vp.resize(4);
		for (int i = 0; i < vp.size(); i++) {
			Dictionary d = ((Dictionary)orig_vp[i]).duplicate();
			d["use_environment"] = false;
			d["doppler"] = false;
			d["gizmos"] = onion.include_gizmos ? d["gizmos"] : Variant(false);
			d["information"] = false;
			vp[i] = d;
		}
		new_state["viewports"] = vp;
		// TODO: Save/restore only affected entries.
		Node3DEditor::get_singleton()->set_state(new_state);
	} else { // CanvasItemEditor
		// 2D
		canvas_edit_state = CanvasItemEditor::get_singleton()->get_state();
		Dictionary new_state = canvas_edit_state.duplicate();
		new_state["show_grid"] = false;
		new_state["show_rulers"] = false;
		new_state["show_guides"] = false;
		new_state["show_helpers"] = false;
		new_state["show_zoom_control"] = false;
		// TODO: Save/restore only affected entries.
		CanvasItemEditor::get_singleton()->set_state(new_state);
	}

	// Tweak the root viewport to ensure it's rendered before our target.
	RID root_vp = get_tree()->get_root()->get_viewport_rid();
	Rect2 root_vp_screen_rect = Rect2(Vector2(), get_tree()->get_root()->get_size());
	RS::get_singleton()->viewport_attach_to_screen(root_vp, Rect2());
	RS::get_singleton()->viewport_set_update_mode(root_vp, RS::VIEWPORT_UPDATE_ALWAYS);

	RID present_rid;
	if (onion.differences_only) {
		// Capture present scene as it is.
		RS::get_singleton()->canvas_item_set_material(onion.capture.canvas_item, RID());
		present_rid = onion.captures[onion.captures.size() - 1];
		RS::get_singleton()->viewport_set_active(present_rid, true);
		RS::get_singleton()->viewport_set_parent_viewport(root_vp, present_rid);
		RS::get_singleton()->draw(false);
		RS::get_singleton()->viewport_set_active(present_rid, false);
	}

	// Backup current animation state.
	Ref<AnimatedValuesBackup> values_backup = player->backup_animated_values();
	float cpos = player->get_current_animation_position();

	// Render every past/future step with the capture shader.

	RS::get_singleton()->canvas_item_set_material(onion.capture.canvas_item, onion.capture.material->get_rid());
	onion.capture.material->set_shader_param("bkg_color", GLOBAL_GET("rendering/environment/defaults/default_clear_color"));
	onion.capture.material->set_shader_param("differences_only", onion.differences_only);
	onion.capture.material->set_shader_param("present", onion.differences_only ? RS::get_singleton()->viewport_get_texture(present_rid) : RID());

	int step_off_a = onion.past ? -onion.steps : 0;
	int step_off_b = onion.future ? onion.steps : 0;
	int cidx = 0;
	onion.capture.material->set_shader_param("dir_color", onion.force_white_modulate ? Color(1, 1, 1) : Color(EDITOR_GET("editors/animation/onion_layers_past_color")));
	for (int step_off = step_off_a; step_off <= step_off_b; step_off++) {
		if (step_off == 0) {
			// Skip present step and switch to the color of future.
			if (!onion.force_white_modulate) {
				onion.capture.material->set_shader_param("dir_color", EDITOR_GET("editors/animation/onion_layers_future_color"));
			}
			continue;
		}

		float pos = cpos + step_off * anim->get_step();

		bool valid = anim->get_loop_mode() != Animation::LoopMode::LOOP_NONE || (pos >= 0 && pos <= anim->get_length());
		onion.captures_valid.write[cidx] = valid;
		if (valid) {
			player->seek(pos, true);
			get_tree()->flush_transform_notifications(); // Needed for transforms of Node3Ds.
			values_backup->update_skeletons(); // Needed for Skeletons (2D & 3D).

			RS::get_singleton()->viewport_set_active(onion.captures[cidx], true);
			RS::get_singleton()->viewport_set_parent_viewport(root_vp, onion.captures[cidx]);
			RS::get_singleton()->draw(false);
			RS::get_singleton()->viewport_set_active(onion.captures[cidx], false);
		}

		cidx++;
	}

	// Restore root viewport.
	RS::get_singleton()->viewport_set_parent_viewport(root_vp, RID());
	RS::get_singleton()->viewport_attach_to_screen(root_vp, root_vp_screen_rect);
	RS::get_singleton()->viewport_set_update_mode(root_vp, RS::VIEWPORT_UPDATE_WHEN_VISIBLE);

	// Restore animation state
	// (Seeking with update=true wouldn't do the trick because the current value of the properties
	// may not match their value for the current point in the animation).
	player->seek(cpos, false);
	values_backup->restore();

	// Restore state of main editors.
	if (Node3DEditor::get_singleton()->is_visible()) {
		// 3D
		Node3DEditor::get_singleton()->set_state(spatial_edit_state);
	} else { // CanvasItemEditor
		// 2D
		CanvasItemEditor::get_singleton()->set_state(canvas_edit_state);
	}

	// Update viewports with skin layers overlaid for the actual engine loop render.
	onion.can_overlay = true;
	plugin->update_overlays();
}

void AnimationPlayerEditor::_start_onion_skinning() {
	// FIXME: Using "process_frame" makes onion layers update one frame behind the current.
	if (!get_tree()->is_connected("process_frame", callable_mp(this, &AnimationPlayerEditor::_prepare_onion_layers_1_deferred))) {
		get_tree()->connect("process_frame", callable_mp(this, &AnimationPlayerEditor::_prepare_onion_layers_1_deferred));
	}
}

void AnimationPlayerEditor::_stop_onion_skinning() {
	if (get_tree()->is_connected("process_frame", callable_mp(this, &AnimationPlayerEditor::_prepare_onion_layers_1_deferred))) {
		get_tree()->disconnect("process_frame", callable_mp(this, &AnimationPlayerEditor::_prepare_onion_layers_1_deferred));

		_free_onion_layers();

		// Clean up the overlay.
		onion.can_overlay = false;
		plugin->update_overlays();
	}
}

void AnimationPlayerEditor::_pin_pressed() {
	SceneTreeDock::get_singleton()->get_tree_editor()->update_tree();
}

void AnimationPlayerEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_animation_new"), &AnimationPlayerEditor::_animation_new);
	ClassDB::bind_method(D_METHOD("_animation_rename"), &AnimationPlayerEditor::_animation_rename);
	ClassDB::bind_method(D_METHOD("_animation_load"), &AnimationPlayerEditor::_animation_load);
	ClassDB::bind_method(D_METHOD("_animation_remove"), &AnimationPlayerEditor::_animation_remove);
	ClassDB::bind_method(D_METHOD("_animation_blend"), &AnimationPlayerEditor::_animation_blend);
	ClassDB::bind_method(D_METHOD("_animation_edit"), &AnimationPlayerEditor::_animation_edit);
	ClassDB::bind_method(D_METHOD("_animation_resource_edit"), &AnimationPlayerEditor::_animation_resource_edit);
	ClassDB::bind_method(D_METHOD("_animation_player_changed"), &AnimationPlayerEditor::_animation_player_changed);
	ClassDB::bind_method(D_METHOD("_list_changed"), &AnimationPlayerEditor::_list_changed);
	ClassDB::bind_method(D_METHOD("_animation_duplicate"), &AnimationPlayerEditor::_animation_duplicate);

	ClassDB::bind_method(D_METHOD("_prepare_onion_layers_1"), &AnimationPlayerEditor::_prepare_onion_layers_1);
	ClassDB::bind_method(D_METHOD("_prepare_onion_layers_2"), &AnimationPlayerEditor::_prepare_onion_layers_2);
	ClassDB::bind_method(D_METHOD("_start_onion_skinning"), &AnimationPlayerEditor::_start_onion_skinning);
	ClassDB::bind_method(D_METHOD("_stop_onion_skinning"), &AnimationPlayerEditor::_stop_onion_skinning);
}

AnimationPlayerEditor *AnimationPlayerEditor::singleton = nullptr;

AnimationPlayer *AnimationPlayerEditor::get_player() const {
	return player;
}

AnimationPlayerEditor::AnimationPlayerEditor(EditorNode *p_editor, AnimationPlayerEditorPlugin *p_plugin) {
	editor = p_editor;
	plugin = p_plugin;
	singleton = this;

	updating = false;

	set_focus_mode(FOCUS_ALL);

	player = nullptr;

	HBoxContainer *hb = memnew(HBoxContainer);
	add_child(hb);

	play_bw_from = memnew(Button);
	play_bw_from->set_flat(true);
	play_bw_from->set_tooltip(TTR("Play selected animation backwards from current pos. (A)"));
	hb->add_child(play_bw_from);

	play_bw = memnew(Button);
	play_bw->set_flat(true);
	play_bw->set_tooltip(TTR("Play selected animation backwards from end. (Shift+A)"));
	hb->add_child(play_bw);

	stop = memnew(Button);
	stop->set_flat(true);
	stop->set_toggle_mode(true);
	hb->add_child(stop);
	stop->set_tooltip(TTR("Stop animation playback. (S)"));

	play = memnew(Button);
	play->set_flat(true);
	play->set_tooltip(TTR("Play selected animation from start. (Shift+D)"));
	hb->add_child(play);

	play_from = memnew(Button);
	play_from->set_flat(true);
	play_from->set_tooltip(TTR("Play selected animation from current pos. (D)"));
	hb->add_child(play_from);

	frame = memnew(SpinBox);
	hb->add_child(frame);
	frame->set_custom_minimum_size(Size2(60, 0));
	frame->set_stretch_ratio(2);
	frame->set_step(0.0001);
	frame->set_tooltip(TTR("Animation position (in seconds)."));

	hb->add_child(memnew(VSeparator));

	scale = memnew(LineEdit);
	hb->add_child(scale);
	scale->set_h_size_flags(SIZE_EXPAND_FILL);
	scale->set_stretch_ratio(1);
	scale->set_tooltip(TTR("Scale animation playback globally for the node."));
	scale->hide();

	delete_dialog = memnew(ConfirmationDialog);
	add_child(delete_dialog);
	delete_dialog->connect("confirmed", callable_mp(this, &AnimationPlayerEditor::_animation_remove_confirmed));

	tool_anim = memnew(MenuButton);
	tool_anim->set_shortcut_context(this);
	tool_anim->set_flat(false);
	tool_anim->set_tooltip(TTR("Animation Tools"));
	tool_anim->set_text(TTR("Animation"));
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/new_animation", TTR("New")), TOOL_NEW_ANIM);
	tool_anim->get_popup()->add_separator();
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/open_animation", TTR("Load")), TOOL_LOAD_ANIM);
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/save_animation", TTR("Save")), TOOL_SAVE_ANIM);
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/save_as_animation", TTR("Save As...")), TOOL_SAVE_AS_ANIM);
	tool_anim->get_popup()->add_separator();
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/copy_animation", TTR("Copy")), TOOL_COPY_ANIM);
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/paste_animation", TTR("Paste")), TOOL_PASTE_ANIM);
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/paste_animation_as_reference", TTR("Paste As Reference")), TOOL_PASTE_ANIM_REF);
	tool_anim->get_popup()->add_separator();
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/duplicate_animation", TTR("Duplicate")), TOOL_DUPLICATE_ANIM);
	tool_anim->get_popup()->add_separator();
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/rename_animation", TTR("Rename...")), TOOL_RENAME_ANIM);
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/edit_transitions", TTR("Edit Transitions...")), TOOL_EDIT_TRANSITIONS);
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/open_animation_in_inspector", TTR("Open in Inspector")), TOOL_EDIT_RESOURCE);
	tool_anim->get_popup()->add_separator();
	tool_anim->get_popup()->add_shortcut(ED_SHORTCUT("animation_player_editor/remove_animation", TTR("Remove")), TOOL_REMOVE_ANIM);
	hb->add_child(tool_anim);

	animation = memnew(OptionButton);
	hb->add_child(animation);
	animation->set_h_size_flags(SIZE_EXPAND_FILL);
	animation->set_tooltip(TTR("Display list of animations in player."));
	animation->set_clip_text(true);

	autoplay = memnew(Button);
	autoplay->set_flat(true);
	hb->add_child(autoplay);
	autoplay->set_tooltip(TTR("Autoplay on Load"));

	hb->add_child(memnew(VSeparator));

	track_editor = memnew(AnimationTrackEditor);

	hb->add_child(track_editor->get_edit_menu());

	hb->add_child(memnew(VSeparator));

	onion_toggle = memnew(Button);
	onion_toggle->set_flat(true);
	onion_toggle->set_toggle_mode(true);
	onion_toggle->set_tooltip(TTR("Enable Onion Skinning"));
	onion_toggle->connect("pressed", callable_mp(this, &AnimationPlayerEditor::_onion_skinning_menu), varray(ONION_SKINNING_ENABLE));
	hb->add_child(onion_toggle);

	onion_skinning = memnew(MenuButton);
	onion_skinning->set_tooltip(TTR("Onion Skinning Options"));
	onion_skinning->get_popup()->add_separator(TTR("Directions"));
	onion_skinning->get_popup()->add_check_item(TTR("Past"), ONION_SKINNING_PAST);
	onion_skinning->get_popup()->set_item_checked(onion_skinning->get_popup()->get_item_count() - 1, true);
	onion_skinning->get_popup()->add_check_item(TTR("Future"), ONION_SKINNING_FUTURE);
	onion_skinning->get_popup()->add_separator(TTR("Depth"));
	onion_skinning->get_popup()->add_radio_check_item(TTR("1 step"), ONION_SKINNING_1_STEP);
	onion_skinning->get_popup()->set_item_checked(onion_skinning->get_popup()->get_item_count() - 1, true);
	onion_skinning->get_popup()->add_radio_check_item(TTR("2 steps"), ONION_SKINNING_2_STEPS);
	onion_skinning->get_popup()->add_radio_check_item(TTR("3 steps"), ONION_SKINNING_3_STEPS);
	onion_skinning->get_popup()->add_separator();
	onion_skinning->get_popup()->add_check_item(TTR("Differences Only"), ONION_SKINNING_DIFFERENCES_ONLY);
	onion_skinning->get_popup()->add_check_item(TTR("Force White Modulate"), ONION_SKINNING_FORCE_WHITE_MODULATE);
	onion_skinning->get_popup()->add_check_item(TTR("Include Gizmos (3D)"), ONION_SKINNING_INCLUDE_GIZMOS);
	hb->add_child(onion_skinning);

	hb->add_child(memnew(VSeparator));

	pin = memnew(Button);
	pin->set_flat(true);
	pin->set_toggle_mode(true);
	pin->set_tooltip(TTR("Pin AnimationPlayer"));
	hb->add_child(pin);
	pin->connect("pressed", callable_mp(this, &AnimationPlayerEditor::_pin_pressed));

	file = memnew(EditorFileDialog);
	add_child(file);

	name_dialog = memnew(ConfirmationDialog);
	name_dialog->set_title(TTR("Create New Animation"));
	name_dialog->set_hide_on_ok(false);
	add_child(name_dialog);
	VBoxContainer *vb = memnew(VBoxContainer);
	name_dialog->add_child(vb);

	name_title = memnew(Label(TTR("Animation Name:")));
	vb->add_child(name_title);

	name = memnew(LineEdit);
	vb->add_child(name);
	name_dialog->register_text_enter(name);

	error_dialog = memnew(ConfirmationDialog);
	error_dialog->get_ok_button()->set_text(TTR("Close"));
	error_dialog->set_title(TTR("Error!"));
	add_child(error_dialog);

	name_dialog->connect("confirmed", callable_mp(this, &AnimationPlayerEditor::_animation_name_edited));

	blend_editor.dialog = memnew(AcceptDialog);
	add_child(blend_editor.dialog);
	blend_editor.dialog->get_ok_button()->set_text(TTR("Close"));
	blend_editor.dialog->set_hide_on_ok(true);
	VBoxContainer *blend_vb = memnew(VBoxContainer);
	blend_editor.dialog->add_child(blend_vb);
	blend_editor.tree = memnew(Tree);
	blend_editor.tree->set_columns(2);
	blend_vb->add_margin_child(TTR("Blend Times:"), blend_editor.tree, true);
	blend_editor.next = memnew(OptionButton);
	blend_vb->add_margin_child(TTR("Next (Auto Queue):"), blend_editor.next);
	blend_editor.dialog->set_title(TTR("Cross-Animation Blend Times"));
	updating_blends = false;

	blend_editor.tree->connect("item_edited", callable_mp(this, &AnimationPlayerEditor::_blend_edited));

	autoplay->connect("pressed", callable_mp(this, &AnimationPlayerEditor::_autoplay_pressed));
	autoplay->set_toggle_mode(true);
	play->connect("pressed", callable_mp(this, &AnimationPlayerEditor::_play_pressed));
	play_from->connect("pressed", callable_mp(this, &AnimationPlayerEditor::_play_from_pressed));
	play_bw->connect("pressed", callable_mp(this, &AnimationPlayerEditor::_play_bw_pressed));
	play_bw_from->connect("pressed", callable_mp(this, &AnimationPlayerEditor::_play_bw_from_pressed));
	stop->connect("pressed", callable_mp(this, &AnimationPlayerEditor::_stop_pressed));

	animation->connect("item_selected", callable_mp(this, &AnimationPlayerEditor::_animation_selected));

	file->connect("file_selected", callable_mp(this, &AnimationPlayerEditor::_save_animation));
	file->connect("files_selected", callable_mp(this, &AnimationPlayerEditor::_load_animations));
	frame->connect("value_changed", callable_mp(this, &AnimationPlayerEditor::_seek_value_changed), make_binds(true, false));
	scale->connect("text_submitted", callable_mp(this, &AnimationPlayerEditor::_scale_changed));

	renaming = false;
	last_active = false;
	timeline_position = 0;

	set_process_unhandled_key_input(true);

	add_child(track_editor);
	track_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	track_editor->connect("timeline_changed", callable_mp(this, &AnimationPlayerEditor::_animation_key_editor_seek));
	track_editor->connect("animation_len_changed", callable_mp(this, &AnimationPlayerEditor::_animation_key_editor_anim_len_changed));

	_update_player();

	// Onion skinning.

	track_editor->connect("visibility_changed", callable_mp(this, &AnimationPlayerEditor::_editor_visibility_changed));

	onion.enabled = false;
	onion.past = true;
	onion.future = false;
	onion.steps = 1;
	onion.differences_only = false;
	onion.force_white_modulate = false;
	onion.include_gizmos = false;

	onion.last_frame = 0;
	onion.can_overlay = false;
	onion.capture_size = Size2();
	onion.capture.canvas = RS::get_singleton()->canvas_create();
	onion.capture.canvas_item = RS::get_singleton()->canvas_item_create();
	RS::get_singleton()->canvas_item_set_parent(onion.capture.canvas_item, onion.capture.canvas);

	onion.capture.material = Ref<ShaderMaterial>(memnew(ShaderMaterial));

	onion.capture.shader = Ref<Shader>(memnew(Shader));
	onion.capture.shader->set_code(R"(
// Animation editor onion skinning shader.

shader_type canvas_item;

uniform vec4 bkg_color;
uniform vec4 dir_color;
uniform bool differences_only;
uniform sampler2D present;

float zero_if_equal(vec4 a, vec4 b) {
	return smoothstep(0.0, 0.005, length(a.rgb - b.rgb) / sqrt(3.0));
}

void fragment() {
	vec4 capture_samp = texture(TEXTURE, UV);
	vec4 present_samp = texture(present, UV);
	float bkg_mask = zero_if_equal(capture_samp, bkg_color);
	float diff_mask = 1.0 - zero_if_equal(present_samp, bkg_color);
	diff_mask = min(1.0, diff_mask + float(!differences_only));
	COLOR = vec4(capture_samp.rgb * dir_color.rgb, bkg_mask * diff_mask);
}
)");
	RS::get_singleton()->material_set_shader(onion.capture.material->get_rid(), onion.capture.shader->get_rid());
}

AnimationPlayerEditor::~AnimationPlayerEditor() {
	_free_onion_layers();
	RS::get_singleton()->free(onion.capture.canvas);
	RS::get_singleton()->free(onion.capture.canvas_item);
}

void AnimationPlayerEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			Node3DEditor::get_singleton()->connect("transform_key_request", callable_mp(this, &AnimationPlayerEditorPlugin::_transform_key_request));
			InspectorDock::get_singleton()->connect("property_keyed", callable_mp(this, &AnimationPlayerEditorPlugin::_property_keyed));
			anim_editor->get_track_editor()->connect("keying_changed", callable_mp(this, &AnimationPlayerEditorPlugin::_update_keying));
			InspectorDock::get_singleton()->connect("edited_object_changed", callable_mp(anim_editor->get_track_editor(), &AnimationTrackEditor::update_keying));
			set_force_draw_over_forwarding_enabled();
		} break;
	}
}

void AnimationPlayerEditorPlugin::_property_keyed(const String &p_keyed, const Variant &p_value, bool p_advance) {
	if (!anim_editor->get_track_editor()->has_keying()) {
		return;
	}
	anim_editor->get_track_editor()->insert_value_key(p_keyed, p_value, p_advance);
}

void AnimationPlayerEditorPlugin::_transform_key_request(Object *sp, const String &p_sub, const Transform3D &p_key) {
	if (!anim_editor->get_track_editor()->has_keying()) {
		return;
	}
	Node3D *s = Object::cast_to<Node3D>(sp);
	if (!s) {
		return;
	}
	anim_editor->get_track_editor()->insert_transform_key(s, p_sub, Animation::TYPE_POSITION_3D, p_key.origin);
	anim_editor->get_track_editor()->insert_transform_key(s, p_sub, Animation::TYPE_ROTATION_3D, p_key.basis.get_rotation_quaternion());
	anim_editor->get_track_editor()->insert_transform_key(s, p_sub, Animation::TYPE_SCALE_3D, p_key.basis.get_scale());
}

void AnimationPlayerEditorPlugin::_update_keying() {
	InspectorDock::get_inspector_singleton()->set_keying(anim_editor->get_track_editor()->has_keying());
}

void AnimationPlayerEditorPlugin::edit(Object *p_object) {
	anim_editor->set_undo_redo(&get_undo_redo());
	if (!p_object) {
		return;
	}
	anim_editor->edit(Object::cast_to<AnimationPlayer>(p_object));
}

bool AnimationPlayerEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("AnimationPlayer");
}

void AnimationPlayerEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		editor->make_bottom_panel_item_visible(anim_editor);
		anim_editor->set_process(true);
		anim_editor->ensure_visibility();
	}
}

AnimationPlayerEditorPlugin::AnimationPlayerEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	anim_editor = memnew(AnimationPlayerEditor(editor, this));
	anim_editor->set_undo_redo(EditorNode::get_undo_redo());
	editor->add_bottom_panel_item(TTR("Animation"), anim_editor);
}

AnimationPlayerEditorPlugin::~AnimationPlayerEditorPlugin() {
}
