/*************************************************************************/
/*  sprite_frames_editor_plugin.cpp                                      */
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

#include "sprite_frames_editor_plugin.h"

#include "core/config/project_settings.h"
#include "core/io/resource_loader.h"
#include "core/os/keyboard.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "scene/3d/sprite_3d.h"
#include "scene/gui/center_container.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/panel_container.h"

void SpriteFramesEditor::_gui_input(Ref<InputEvent> p_event) {
}

void SpriteFramesEditor::_open_sprite_sheet() {
	file_split_sheet->clear_filters();
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Texture2D", &extensions);
	for (int i = 0; i < extensions.size(); i++) {
		file_split_sheet->add_filter("*." + extensions[i]);
	}

	file_split_sheet->popup_file_dialog();
}

void SpriteFramesEditor::_sheet_preview_draw() {
	Size2i size = split_sheet_preview->get_size();
	int h = split_sheet_h->get_value();
	int v = split_sheet_v->get_value();
	int width = size.width / h;
	int height = size.height / v;
	const float a = 0.3;
	for (int i = 1; i < h; i++) {
		int x = i * width;
		split_sheet_preview->draw_line(Point2(x, 0), Point2(x, size.height), Color(1, 1, 1, a));
		split_sheet_preview->draw_line(Point2(x + 1, 0), Point2(x + 1, size.height), Color(0, 0, 0, a));

		for (int j = 1; j < v; j++) {
			int y = j * height;

			split_sheet_preview->draw_line(Point2(0, y), Point2(size.width, y), Color(1, 1, 1, a));
			split_sheet_preview->draw_line(Point2(0, y + 1), Point2(size.width, y + 1), Color(0, 0, 0, a));
		}
	}

	if (frames_selected.size() == 0) {
		split_sheet_dialog->get_ok_button()->set_disabled(true);
		split_sheet_dialog->get_ok_button()->set_text(TTR("No Frames Selected"));
		return;
	}

	Color accent = get_theme_color("accent_color", "Editor");

	for (Set<int>::Element *E = frames_selected.front(); E; E = E->next()) {
		int idx = E->get();
		int xp = idx % h;
		int yp = (idx - xp) / h;
		int x = xp * width;
		int y = yp * height;

		split_sheet_preview->draw_rect(Rect2(x + 5, y + 5, width - 10, height - 10), Color(0, 0, 0, 0.35), true);
		split_sheet_preview->draw_rect(Rect2(x + 0, y + 0, width - 0, height - 0), Color(0, 0, 0, 1), false);
		split_sheet_preview->draw_rect(Rect2(x + 1, y + 1, width - 2, height - 2), Color(0, 0, 0, 1), false);
		split_sheet_preview->draw_rect(Rect2(x + 2, y + 2, width - 4, height - 4), accent, false);
		split_sheet_preview->draw_rect(Rect2(x + 3, y + 3, width - 6, height - 6), accent, false);
		split_sheet_preview->draw_rect(Rect2(x + 4, y + 4, width - 8, height - 8), Color(0, 0, 0, 1), false);
		split_sheet_preview->draw_rect(Rect2(x + 5, y + 5, width - 10, height - 10), Color(0, 0, 0, 1), false);
	}

	split_sheet_dialog->get_ok_button()->set_disabled(false);
	split_sheet_dialog->get_ok_button()->set_text(vformat(TTR("Add %d Frame(s)"), frames_selected.size()));
}

void SpriteFramesEditor::_sheet_preview_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid() && mb->is_pressed() && mb->get_button_index() == MOUSE_BUTTON_LEFT) {
		Size2i size = split_sheet_preview->get_size();
		int h = split_sheet_h->get_value();
		int v = split_sheet_v->get_value();

		int x = CLAMP(int(mb->get_position().x) * h / size.width, 0, h - 1);
		int y = CLAMP(int(mb->get_position().y) * v / size.height, 0, v - 1);

		int idx = h * y + x;

		if (mb->is_shift_pressed() && last_frame_selected >= 0) {
			//select multiple
			int from = idx;
			int to = last_frame_selected;
			if (from > to) {
				SWAP(from, to);
			}

			for (int i = from; i <= to; i++) {
				if (mb->is_ctrl_pressed()) {
					frames_selected.erase(i);
				} else {
					frames_selected.insert(i);
				}
			}
		} else {
			if (frames_selected.has(idx)) {
				frames_selected.erase(idx);
			} else {
				frames_selected.insert(idx);
			}
		}

		last_frame_selected = idx;
		split_sheet_preview->update();
	}
}

void SpriteFramesEditor::_sheet_scroll_input(const Ref<InputEvent> &p_event) {
	const Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		// Zoom in/out using Ctrl + mouse wheel. This is done on the ScrollContainer
		// to allow performing this action anywhere, even if the cursor isn't
		// hovering the texture in the workspace.
		if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_UP && mb->is_pressed() && mb->is_ctrl_pressed()) {
			_sheet_zoom_in();
			// Don't scroll up after zooming in.
			accept_event();
		} else if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_DOWN && mb->is_pressed() && mb->is_ctrl_pressed()) {
			_sheet_zoom_out();
			// Don't scroll down after zooming out.
			accept_event();
		}
	}
}

void SpriteFramesEditor::_sheet_add_frames() {
	Size2i size = split_sheet_preview->get_texture()->get_size();
	int h = split_sheet_h->get_value();
	int v = split_sheet_v->get_value();

	undo_redo->create_action(TTR("Add Frame"));

	int fc = frames->get_frame_count(edited_anim);

	AtlasTexture *atlas_source = Object::cast_to<AtlasTexture>(*split_sheet_preview->get_texture());

	Rect2 region_rect = Rect2();

	if (atlas_source && atlas_source->get_atlas().is_valid()) {
		region_rect = atlas_source->get_region();
	}

	for (Set<int>::Element *E = frames_selected.front(); E; E = E->next()) {
		int idx = E->get();
		int width = size.width / h;
		int height = size.height / v;
		int xp = idx % h;
		int yp = (idx - xp) / h;
		int x = (xp * width) + region_rect.position.x;
		int y = (yp * height) + region_rect.position.y;

		Ref<AtlasTexture> at;
		at.instance();
		at->set_atlas(split_sheet_preview->get_texture());
		at->set_region(Rect2(x, y, width, height));

		undo_redo->add_do_method(frames, "add_frame", edited_anim, at, -1);
		undo_redo->add_undo_method(frames, "remove_frame", edited_anim, fc);
	}

	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void SpriteFramesEditor::_sheet_zoom_in() {
	if (sheet_zoom < max_sheet_zoom) {
		sheet_zoom *= scale_ratio;
		Size2 texture_size = split_sheet_preview->get_texture()->get_size();
		split_sheet_preview->set_custom_minimum_size(texture_size * sheet_zoom);
	}
}

void SpriteFramesEditor::_sheet_zoom_out() {
	if (sheet_zoom > min_sheet_zoom) {
		sheet_zoom /= scale_ratio;
		Size2 texture_size = split_sheet_preview->get_texture()->get_size();
		split_sheet_preview->set_custom_minimum_size(texture_size * sheet_zoom);
	}
}

void SpriteFramesEditor::_sheet_zoom_reset() {
	// Default the zoom to match the editor scale, but don't dezoom on editor scales below 100% to prevent pixel art from looking bad.
	sheet_zoom = MAX(1.0f, EDSCALE);
	Size2 texture_size = split_sheet_preview->get_texture()->get_size();
	split_sheet_preview->set_custom_minimum_size(texture_size * sheet_zoom);
}

void SpriteFramesEditor::_sheet_select_clear_all_frames() {
	bool should_clear = true;
	for (int i = 0; i < split_sheet_h->get_value() * split_sheet_v->get_value(); i++) {
		if (!frames_selected.has(i)) {
			frames_selected.insert(i);
			should_clear = false;
		}
	}
	if (should_clear) {
		frames_selected.clear();
	}

	split_sheet_preview->update();
}

void SpriteFramesEditor::_sheet_spin_changed(double) {
	frames_selected.clear();
	last_frame_selected = -1;
	split_sheet_preview->update();
}

void SpriteFramesEditor::_prepare_sprite_sheet(const String &p_file) {
	Ref<Resource> texture = ResourceLoader::load(p_file);
	if (!texture.is_valid()) {
		EditorNode::get_singleton()->show_warning(TTR("Unable to load images"));
		ERR_FAIL_COND(!texture.is_valid());
	}
	frames_selected.clear();
	last_frame_selected = -1;

	bool new_texture = texture != split_sheet_preview->get_texture();
	split_sheet_preview->set_texture(texture);
	if (new_texture) {
		//different texture, reset to 4x4
		split_sheet_h->set_value(4);
		split_sheet_v->set_value(4);
		//reset zoom
		_sheet_zoom_reset();
	}
	split_sheet_dialog->popup_centered_ratio(0.65);
}

void SpriteFramesEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			load->set_icon(get_theme_icon("Load", "EditorIcons"));
			load_sheet->set_icon(get_theme_icon("SpriteSheet", "EditorIcons"));
			copy->set_icon(get_theme_icon("ActionCopy", "EditorIcons"));
			paste->set_icon(get_theme_icon("ActionPaste", "EditorIcons"));
			empty->set_icon(get_theme_icon("InsertBefore", "EditorIcons"));
			empty2->set_icon(get_theme_icon("InsertAfter", "EditorIcons"));
			move_up->set_icon(get_theme_icon("MoveLeft", "EditorIcons"));
			move_down->set_icon(get_theme_icon("MoveRight", "EditorIcons"));
			_delete->set_icon(get_theme_icon("Remove", "EditorIcons"));
			zoom_out->set_icon(get_theme_icon("ZoomLess", "EditorIcons"));
			zoom_reset->set_icon(get_theme_icon("ZoomReset", "EditorIcons"));
			zoom_in->set_icon(get_theme_icon("ZoomMore", "EditorIcons"));
			new_anim->set_icon(get_theme_icon("New", "EditorIcons"));
			remove_anim->set_icon(get_theme_icon("Remove", "EditorIcons"));
			split_sheet_zoom_out->set_icon(get_theme_icon("ZoomLess", "EditorIcons"));
			split_sheet_zoom_reset->set_icon(get_theme_icon("ZoomReset", "EditorIcons"));
			split_sheet_zoom_in->set_icon(get_theme_icon("ZoomMore", "EditorIcons"));
			[[fallthrough]];
		}
		case NOTIFICATION_THEME_CHANGED: {
			split_sheet_scroll->add_theme_style_override("bg", get_theme_stylebox("bg", "Tree"));
		} break;
		case NOTIFICATION_READY: {
			add_theme_constant_override("autohide", 1); // Fixes the dragger always showing up.
		} break;
	}
}

void SpriteFramesEditor::_file_load_request(const Vector<String> &p_path, int p_at_pos) {
	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	List<Ref<Texture2D>> resources;

	for (int i = 0; i < p_path.size(); i++) {
		Ref<Texture2D> resource;
		resource = ResourceLoader::load(p_path[i]);

		if (resource.is_null()) {
			dialog->set_text(TTR("ERROR: Couldn't load frame resource!"));
			dialog->set_title(TTR("Error!"));

			//dialog->get_cancel()->set_text("Close");
			dialog->get_ok_button()->set_text(TTR("Close"));
			dialog->popup_centered();
			return; ///beh should show an error i guess
		}

		resources.push_back(resource);
	}

	if (resources.is_empty()) {
		return;
	}

	undo_redo->create_action(TTR("Add Frame"));
	int fc = frames->get_frame_count(edited_anim);

	int count = 0;

	for (List<Ref<Texture2D>>::Element *E = resources.front(); E; E = E->next()) {
		undo_redo->add_do_method(frames, "add_frame", edited_anim, E->get(), p_at_pos == -1 ? -1 : p_at_pos + count);
		undo_redo->add_undo_method(frames, "remove_frame", edited_anim, p_at_pos == -1 ? fc : p_at_pos);
		count++;
	}
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");

	undo_redo->commit_action();
}

void SpriteFramesEditor::_load_pressed() {
	ERR_FAIL_COND(!frames->has_animation(edited_anim));
	loading_scene = false;

	file->clear_filters();
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Texture2D", &extensions);
	for (int i = 0; i < extensions.size(); i++) {
		file->add_filter("*." + extensions[i]);
	}

	file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
	file->popup_file_dialog();
}

void SpriteFramesEditor::_paste_pressed() {
	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	Ref<Texture2D> r = EditorSettings::get_singleton()->get_resource_clipboard();
	if (!r.is_valid()) {
		dialog->set_text(TTR("Resource clipboard is empty or not a texture!"));
		dialog->set_title(TTR("Error!"));
		//dialog->get_cancel()->set_text("Close");
		dialog->get_ok_button()->set_text(TTR("Close"));
		dialog->popup_centered();
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

	if (tree->get_current() < 0) {
		return;
	}
	Ref<Texture2D> r = frames->get_frame(edited_anim, tree->get_current());
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

	Ref<Texture2D> r;

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

	Ref<Texture2D> r;

	undo_redo->create_action(TTR("Add Empty"));
	undo_redo->add_do_method(frames, "add_frame", edited_anim, r, from + 1);
	undo_redo->add_undo_method(frames, "remove_frame", edited_anim, from + 1);
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void SpriteFramesEditor::_up_pressed() {
	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	if (tree->get_current() < 0) {
		return;
	}

	int to_move = tree->get_current();
	if (to_move < 1) {
		return;
	}

	sel = to_move;
	sel -= 1;

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

	if (tree->get_current() < 0) {
		return;
	}

	int to_move = tree->get_current();
	if (to_move < 0 || to_move >= frames->get_frame_count(edited_anim) - 1) {
		return;
	}

	sel = to_move;
	sel += 1;

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

	if (tree->get_current() < 0) {
		return;
	}

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
	if (updating) {
		return;
	}

	if (frames->has_animation(edited_anim)) {
		double value = anim_speed->get_line_edit()->get_text().to_float();
		if (!Math::is_equal_approx(value, (double)frames->get_animation_speed(edited_anim))) {
			_animation_fps_changed(value);
		}
	}

	TreeItem *selected = animations->get_selected();
	ERR_FAIL_COND(!selected);
	edited_anim = selected->get_text(0);
	_update_library(true);
}

static void _find_anim_sprites(Node *p_node, List<Node *> *r_nodes, Ref<SpriteFrames> p_sfames) {
	Node *edited = EditorNode::get_singleton()->get_edited_scene();
	if (!edited) {
		return;
	}
	if (p_node != edited && p_node->get_owner() != edited) {
		return;
	}

	{
		AnimatedSprite2D *as = Object::cast_to<AnimatedSprite2D>(p_node);
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
	if (updating) {
		return;
	}

	if (!frames->has_animation(edited_anim)) {
		return;
	}

	TreeItem *edited = animations->get_edited();
	if (!edited) {
		return;
	}

	String new_name = edited->get_text(0);

	if (new_name == String(edited_anim)) {
		return;
	}

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
		undo_redo->add_do_method(E->get(), "set_animation", name);
		undo_redo->add_undo_method(E->get(), "set_animation", current);
	}

	edited_anim = name;

	undo_redo->commit_action();
	animations->grab_focus();
}

void SpriteFramesEditor::_animation_remove() {
	if (updating) {
		return;
	}

	if (!frames->has_animation(edited_anim)) {
		return;
	}

	delete_dialog->set_text(TTR("Delete Animation?"));
	delete_dialog->popup_centered();
}

void SpriteFramesEditor::_animation_remove_confirmed() {
	undo_redo->create_action(TTR("Remove Animation"));
	undo_redo->add_do_method(frames, "remove_animation", edited_anim);
	undo_redo->add_undo_method(frames, "add_animation", edited_anim);
	undo_redo->add_undo_method(frames, "set_animation_speed", edited_anim, frames->get_animation_speed(edited_anim));
	undo_redo->add_undo_method(frames, "set_animation_loop", edited_anim, frames->get_animation_loop(edited_anim));
	int fc = frames->get_frame_count(edited_anim);
	for (int i = 0; i < fc; i++) {
		Ref<Texture2D> frame = frames->get_frame(edited_anim, i);
		undo_redo->add_undo_method(frames, "add_frame", edited_anim, frame);
	}
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");

	edited_anim = StringName();

	undo_redo->commit_action();
}

void SpriteFramesEditor::_animation_loop_changed() {
	if (updating) {
		return;
	}

	undo_redo->create_action(TTR("Change Animation Loop"));
	undo_redo->add_do_method(frames, "set_animation_loop", edited_anim, anim_loop->is_pressed());
	undo_redo->add_undo_method(frames, "set_animation_loop", edited_anim, frames->get_animation_loop(edited_anim));
	undo_redo->add_do_method(this, "_update_library", true);
	undo_redo->add_undo_method(this, "_update_library", true);
	undo_redo->commit_action();
}

void SpriteFramesEditor::_animation_fps_changed(double p_value) {
	if (updating) {
		return;
	}

	undo_redo->create_action(TTR("Change Animation FPS"), UndoRedo::MERGE_ENDS);
	undo_redo->add_do_method(frames, "set_animation_speed", edited_anim, p_value);
	undo_redo->add_undo_method(frames, "set_animation_speed", edited_anim, frames->get_animation_speed(edited_anim));
	undo_redo->add_do_method(this, "_update_library", true);
	undo_redo->add_undo_method(this, "_update_library", true);

	undo_redo->commit_action();
}

void SpriteFramesEditor::_tree_input(const Ref<InputEvent> &p_event) {
	const Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_UP && mb->is_pressed() && mb->is_ctrl_pressed()) {
			_zoom_in();
			// Don't scroll up after zooming in.
			accept_event();
		} else if (mb->get_button_index() == MOUSE_BUTTON_WHEEL_DOWN && mb->is_pressed() && mb->is_ctrl_pressed()) {
			_zoom_out();
			// Don't scroll down after zooming out.
			accept_event();
		}
	}
}

void SpriteFramesEditor::_zoom_in() {
	// Do not zoom in or out with no visible frames
	if (frames->get_frame_count(edited_anim) <= 0) {
		return;
	}
	if (thumbnail_zoom < max_thumbnail_zoom) {
		thumbnail_zoom *= scale_ratio;
		int thumbnail_size = (int)(thumbnail_default_size * thumbnail_zoom);
		tree->set_fixed_column_width(thumbnail_size * 3 / 2);
		tree->set_fixed_icon_size(Size2(thumbnail_size, thumbnail_size));
	}
}

void SpriteFramesEditor::_zoom_out() {
	// Do not zoom in or out with no visible frames
	if (frames->get_frame_count(edited_anim) <= 0) {
		return;
	}
	if (thumbnail_zoom > min_thumbnail_zoom) {
		thumbnail_zoom /= scale_ratio;
		int thumbnail_size = (int)(thumbnail_default_size * thumbnail_zoom);
		tree->set_fixed_column_width(thumbnail_size * 3 / 2);
		tree->set_fixed_icon_size(Size2(thumbnail_size, thumbnail_size));
	}
}

void SpriteFramesEditor::_zoom_reset() {
	thumbnail_zoom = MAX(1.0f, EDSCALE);
	tree->set_fixed_column_width(thumbnail_default_size * 3 / 2);
	tree->set_fixed_icon_size(Size2(thumbnail_default_size, thumbnail_default_size));
}

void SpriteFramesEditor::_update_library(bool p_skip_selector) {
	updating = true;

	if (!p_skip_selector) {
		animations->clear();

		TreeItem *anim_root = animations->create_item();

		List<StringName> anim_names;

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

	if (sel >= frames->get_frame_count(edited_anim)) {
		sel = frames->get_frame_count(edited_anim) - 1;
	} else if (sel < 0 && frames->get_frame_count(edited_anim)) {
		sel = 0;
	}

	for (int i = 0; i < frames->get_frame_count(edited_anim); i++) {
		String name;
		Ref<Texture2D> icon;

		if (frames->get_frame(edited_anim, i).is_null()) {
			name = itos(i) + ": " + TTR("(empty)");

		} else {
			name = itos(i) + ": " + frames->get_frame(edited_anim, i)->get_name();
			icon = frames->get_frame(edited_anim, i);
		}

		tree->add_item(name, icon);
		if (frames->get_frame(edited_anim, i).is_valid()) {
			tree->set_item_tooltip(tree->get_item_count() - 1, frames->get_frame(edited_anim, i)->get_path());
		}
		if (sel == i) {
			tree->select(tree->get_item_count() - 1);
		}
	}

	anim_speed->set_value(frames->get_animation_speed(edited_anim));
	anim_loop->set_pressed(frames->get_animation_loop(edited_anim));

	updating = false;
	//player->add_resource("default",resource);
}

void SpriteFramesEditor::edit(SpriteFrames *p_frames) {
	if (frames == p_frames) {
		return;
	}

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
		// Clear zoom and split sheet texture
		split_sheet_preview->set_texture(Ref<Texture2D>());
		_zoom_reset();
	} else {
		hide();
	}
}

Variant SpriteFramesEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	if (!frames->has_animation(edited_anim)) {
		return false;
	}

	int idx = tree->get_item_at_position(p_point, true);

	if (idx < 0 || idx >= frames->get_frame_count(edited_anim)) {
		return Variant();
	}

	RES frame = frames->get_frame(edited_anim, idx);

	if (frame.is_null()) {
		return Variant();
	}

	Dictionary drag_data = EditorNode::get_singleton()->drag_resource(frame, p_from);
	drag_data["frame"] = idx; // store the frame, in case we want to reorder frames inside 'drop_data_fw'
	return drag_data;
}

bool SpriteFramesEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	Dictionary d = p_data;

	if (!d.has("type")) {
		return false;
	}

	// reordering frames
	if (d.has("from") && (Object *)(d["from"]) == tree) {
		return true;
	}

	if (String(d["type"]) == "resource" && d.has("resource")) {
		RES r = d["resource"];

		Ref<Texture2D> texture = r;

		if (texture.is_valid()) {
			return true;
		}
	}

	if (String(d["type"]) == "files") {
		Vector<String> files = d["files"];

		if (files.size() == 0) {
			return false;
		}

		for (int i = 0; i < files.size(); i++) {
			String file = files[i];
			String ftype = EditorFileSystem::get_singleton()->get_file_type(file);

			if (!ClassDB::is_parent_class(ftype, "Texture2D")) {
				return false;
			}
		}

		return true;
	}
	return false;
}

void SpriteFramesEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	Dictionary d = p_data;

	if (!d.has("type")) {
		return;
	}

	int at_pos = tree->get_item_at_position(p_point, true);

	if (String(d["type"]) == "resource" && d.has("resource")) {
		RES r = d["resource"];

		Ref<Texture2D> texture = r;

		if (texture.is_valid()) {
			bool reorder = false;
			if (d.has("from") && (Object *)(d["from"]) == tree) {
				reorder = true;
			}

			if (reorder) { //drop is from reordering frames
				int from_frame = -1;
				if (d.has("frame")) {
					from_frame = d["frame"];
				}

				undo_redo->create_action(TTR("Move Frame"));
				undo_redo->add_do_method(frames, "remove_frame", edited_anim, from_frame == -1 ? frames->get_frame_count(edited_anim) : from_frame);
				undo_redo->add_do_method(frames, "add_frame", edited_anim, texture, at_pos == -1 ? -1 : at_pos);
				undo_redo->add_undo_method(frames, "remove_frame", edited_anim, at_pos == -1 ? frames->get_frame_count(edited_anim) - 1 : at_pos);
				undo_redo->add_undo_method(frames, "add_frame", edited_anim, texture, from_frame);
				undo_redo->add_do_method(this, "_update_library");
				undo_redo->add_undo_method(this, "_update_library");
				undo_redo->commit_action();
			} else {
				undo_redo->create_action(TTR("Add Frame"));
				undo_redo->add_do_method(frames, "add_frame", edited_anim, texture, at_pos == -1 ? -1 : at_pos);
				undo_redo->add_undo_method(frames, "remove_frame", edited_anim, at_pos == -1 ? frames->get_frame_count(edited_anim) : at_pos);
				undo_redo->add_do_method(this, "_update_library");
				undo_redo->add_undo_method(this, "_update_library");
				undo_redo->commit_action();
			}
		}
	}

	if (String(d["type"]) == "files") {
		Vector<String> files = d["files"];

		if (Input::get_singleton()->is_key_pressed(KEY_CTRL)) {
			_prepare_sprite_sheet(files[0]);
		} else {
			_file_load_request(files, at_pos);
		}
	}
}

void SpriteFramesEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_library", "skipsel"), &SpriteFramesEditor::_update_library, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &SpriteFramesEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &SpriteFramesEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &SpriteFramesEditor::drop_data_fw);
}

SpriteFramesEditor::SpriteFramesEditor() {
	VBoxContainer *vbc_animlist = memnew(VBoxContainer);
	add_child(vbc_animlist);
	vbc_animlist->set_custom_minimum_size(Size2(150, 0) * EDSCALE);

	VBoxContainer *sub_vb = memnew(VBoxContainer);
	vbc_animlist->add_margin_child(TTR("Animations:"), sub_vb, true);
	sub_vb->set_v_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *hbc_animlist = memnew(HBoxContainer);
	sub_vb->add_child(hbc_animlist);

	new_anim = memnew(Button);
	new_anim->set_flat(true);
	new_anim->set_tooltip(TTR("New Animation"));
	hbc_animlist->add_child(new_anim);
	new_anim->connect("pressed", callable_mp(this, &SpriteFramesEditor::_animation_add));

	remove_anim = memnew(Button);
	remove_anim->set_flat(true);
	remove_anim->set_tooltip(TTR("Remove Animation"));
	hbc_animlist->add_child(remove_anim);
	remove_anim->connect("pressed", callable_mp(this, &SpriteFramesEditor::_animation_remove));

	animations = memnew(Tree);
	sub_vb->add_child(animations);
	animations->set_v_size_flags(SIZE_EXPAND_FILL);
	animations->set_hide_root(true);
	animations->connect("cell_selected", callable_mp(this, &SpriteFramesEditor::_animation_select));
	animations->connect("item_edited", callable_mp(this, &SpriteFramesEditor::_animation_name_edited));
	animations->set_allow_reselect(true);

	HBoxContainer *hbc_anim_speed = memnew(HBoxContainer);
	hbc_anim_speed->add_child(memnew(Label(TTR("Speed:"))));
	vbc_animlist->add_child(hbc_anim_speed);
	anim_speed = memnew(SpinBox);
	anim_speed->set_suffix(TTR("FPS"));
	anim_speed->set_min(0);
	anim_speed->set_max(100);
	anim_speed->set_step(0.01);
	anim_speed->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc_anim_speed->add_child(anim_speed);
	anim_speed->connect("value_changed", callable_mp(this, &SpriteFramesEditor::_animation_fps_changed));

	anim_loop = memnew(CheckButton);
	anim_loop->set_text(TTR("Loop"));
	vbc_animlist->add_child(anim_loop);
	anim_loop->connect("pressed", callable_mp(this, &SpriteFramesEditor::_animation_loop_changed));

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);
	vbc->set_h_size_flags(SIZE_EXPAND_FILL);

	sub_vb = memnew(VBoxContainer);
	vbc->add_margin_child(TTR("Animation Frames:"), sub_vb, true);

	HBoxContainer *hbc = memnew(HBoxContainer);
	sub_vb->add_child(hbc);

	load = memnew(Button);
	load->set_flat(true);
	load->set_tooltip(TTR("Add a Texture from File"));
	hbc->add_child(load);

	load_sheet = memnew(Button);
	load_sheet->set_flat(true);
	load_sheet->set_tooltip(TTR("Add Frames from a Sprite Sheet"));
	hbc->add_child(load_sheet);

	hbc->add_child(memnew(VSeparator));

	copy = memnew(Button);
	copy->set_flat(true);
	copy->set_tooltip(TTR("Copy"));
	hbc->add_child(copy);

	paste = memnew(Button);
	paste->set_flat(true);
	paste->set_tooltip(TTR("Paste"));
	hbc->add_child(paste);

	hbc->add_child(memnew(VSeparator));

	empty = memnew(Button);
	empty->set_flat(true);
	empty->set_tooltip(TTR("Insert Empty (Before)"));
	hbc->add_child(empty);

	empty2 = memnew(Button);
	empty2->set_flat(true);
	empty2->set_tooltip(TTR("Insert Empty (After)"));
	hbc->add_child(empty2);

	hbc->add_child(memnew(VSeparator));

	move_up = memnew(Button);
	move_up->set_flat(true);
	move_up->set_tooltip(TTR("Move (Before)"));
	hbc->add_child(move_up);

	move_down = memnew(Button);
	move_down->set_flat(true);
	move_down->set_tooltip(TTR("Move (After)"));
	hbc->add_child(move_down);

	_delete = memnew(Button);
	_delete->set_flat(true);
	_delete->set_tooltip(TTR("Delete"));
	hbc->add_child(_delete);

	hbc->add_spacer();

	zoom_out = memnew(Button);
	zoom_out->connect("pressed", callable_mp(this, &SpriteFramesEditor::_zoom_out));
	zoom_out->set_flat(true);
	zoom_out->set_tooltip(TTR("Zoom Out"));
	hbc->add_child(zoom_out);

	zoom_reset = memnew(Button);
	zoom_reset->connect("pressed", callable_mp(this, &SpriteFramesEditor::_zoom_reset));
	zoom_reset->set_flat(true);
	zoom_reset->set_tooltip(TTR("Zoom Reset"));
	hbc->add_child(zoom_reset);

	zoom_in = memnew(Button);
	zoom_in->connect("pressed", callable_mp(this, &SpriteFramesEditor::_zoom_in));
	zoom_in->set_flat(true);
	zoom_in->set_tooltip(TTR("Zoom In"));
	hbc->add_child(zoom_in);

	file = memnew(EditorFileDialog);
	add_child(file);

	tree = memnew(ItemList);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	tree->set_icon_mode(ItemList::ICON_MODE_TOP);

	tree->set_max_columns(0);
	tree->set_icon_mode(ItemList::ICON_MODE_TOP);
	tree->set_max_text_lines(2);
	tree->set_drag_forwarding(this);
	tree->connect("gui_input", callable_mp(this, &SpriteFramesEditor::_tree_input));

	sub_vb->add_child(tree);

	dialog = memnew(AcceptDialog);
	add_child(dialog);

	load->connect("pressed", callable_mp(this, &SpriteFramesEditor::_load_pressed));
	load_sheet->connect("pressed", callable_mp(this, &SpriteFramesEditor::_open_sprite_sheet));
	_delete->connect("pressed", callable_mp(this, &SpriteFramesEditor::_delete_pressed));
	copy->connect("pressed", callable_mp(this, &SpriteFramesEditor::_copy_pressed));
	paste->connect("pressed", callable_mp(this, &SpriteFramesEditor::_paste_pressed));
	empty->connect("pressed", callable_mp(this, &SpriteFramesEditor::_empty_pressed));
	empty2->connect("pressed", callable_mp(this, &SpriteFramesEditor::_empty2_pressed));
	move_up->connect("pressed", callable_mp(this, &SpriteFramesEditor::_up_pressed));
	move_down->connect("pressed", callable_mp(this, &SpriteFramesEditor::_down_pressed));
	file->connect("files_selected", callable_mp(this, &SpriteFramesEditor::_file_load_request), make_binds(-1));
	loading_scene = false;
	sel = -1;

	updating = false;

	edited_anim = "default";

	delete_dialog = memnew(ConfirmationDialog);
	add_child(delete_dialog);
	delete_dialog->connect("confirmed", callable_mp(this, &SpriteFramesEditor::_animation_remove_confirmed));

	split_sheet_dialog = memnew(ConfirmationDialog);
	add_child(split_sheet_dialog);
	VBoxContainer *split_sheet_vb = memnew(VBoxContainer);
	split_sheet_dialog->add_child(split_sheet_vb);
	split_sheet_dialog->set_title(TTR("Select Frames"));
	split_sheet_dialog->connect("confirmed", callable_mp(this, &SpriteFramesEditor::_sheet_add_frames));

	HBoxContainer *split_sheet_hb = memnew(HBoxContainer);

	Label *ss_label = memnew(Label(TTR("Horizontal:")));
	split_sheet_hb->add_child(ss_label);
	split_sheet_h = memnew(SpinBox);
	split_sheet_h->set_min(1);
	split_sheet_h->set_max(128);
	split_sheet_h->set_step(1);
	split_sheet_hb->add_child(split_sheet_h);
	split_sheet_h->connect("value_changed", callable_mp(this, &SpriteFramesEditor::_sheet_spin_changed));

	ss_label = memnew(Label(TTR("Vertical:")));
	split_sheet_hb->add_child(ss_label);
	split_sheet_v = memnew(SpinBox);
	split_sheet_v->set_min(1);
	split_sheet_v->set_max(128);
	split_sheet_v->set_step(1);
	split_sheet_hb->add_child(split_sheet_v);
	split_sheet_v->connect("value_changed", callable_mp(this, &SpriteFramesEditor::_sheet_spin_changed));

	split_sheet_hb->add_spacer();

	Button *select_clear_all = memnew(Button);
	select_clear_all->set_text(TTR("Select/Clear All Frames"));
	select_clear_all->connect("pressed", callable_mp(this, &SpriteFramesEditor::_sheet_select_clear_all_frames));
	split_sheet_hb->add_child(select_clear_all);

	split_sheet_vb->add_child(split_sheet_hb);

	PanelContainer *split_sheet_panel = memnew(PanelContainer);
	split_sheet_panel->set_h_size_flags(SIZE_EXPAND_FILL);
	split_sheet_panel->set_v_size_flags(SIZE_EXPAND_FILL);
	split_sheet_vb->add_child(split_sheet_panel);

	split_sheet_preview = memnew(TextureRect);
	split_sheet_preview->set_expand(true);
	split_sheet_preview->set_mouse_filter(MOUSE_FILTER_PASS);
	split_sheet_preview->connect("draw", callable_mp(this, &SpriteFramesEditor::_sheet_preview_draw));
	split_sheet_preview->connect("gui_input", callable_mp(this, &SpriteFramesEditor::_sheet_preview_input));

	split_sheet_scroll = memnew(ScrollContainer);
	split_sheet_scroll->set_enable_h_scroll(true);
	split_sheet_scroll->set_enable_v_scroll(true);
	split_sheet_scroll->connect("gui_input", callable_mp(this, &SpriteFramesEditor::_sheet_scroll_input));
	split_sheet_panel->add_child(split_sheet_scroll);
	CenterContainer *cc = memnew(CenterContainer);
	cc->add_child(split_sheet_preview);
	cc->set_h_size_flags(SIZE_EXPAND_FILL);
	cc->set_v_size_flags(SIZE_EXPAND_FILL);
	split_sheet_scroll->add_child(cc);

	MarginContainer *split_sheet_zoom_margin = memnew(MarginContainer);
	split_sheet_panel->add_child(split_sheet_zoom_margin);
	split_sheet_zoom_margin->set_h_size_flags(0);
	split_sheet_zoom_margin->set_v_size_flags(0);
	split_sheet_zoom_margin->add_theme_constant_override("margin_top", 5);
	split_sheet_zoom_margin->add_theme_constant_override("margin_left", 5);
	HBoxContainer *split_sheet_zoom_hb = memnew(HBoxContainer);
	split_sheet_zoom_margin->add_child(split_sheet_zoom_hb);

	split_sheet_zoom_out = memnew(Button);
	split_sheet_zoom_out->set_flat(true);
	split_sheet_zoom_out->set_focus_mode(FOCUS_NONE);
	split_sheet_zoom_out->set_tooltip(TTR("Zoom Out"));
	split_sheet_zoom_out->connect("pressed", callable_mp(this, &SpriteFramesEditor::_sheet_zoom_out));
	split_sheet_zoom_hb->add_child(split_sheet_zoom_out);

	split_sheet_zoom_reset = memnew(Button);
	split_sheet_zoom_reset->set_flat(true);
	split_sheet_zoom_reset->set_focus_mode(FOCUS_NONE);
	split_sheet_zoom_reset->set_tooltip(TTR("Zoom Reset"));
	split_sheet_zoom_reset->connect("pressed", callable_mp(this, &SpriteFramesEditor::_sheet_zoom_reset));
	split_sheet_zoom_hb->add_child(split_sheet_zoom_reset);

	split_sheet_zoom_in = memnew(Button);
	split_sheet_zoom_in->set_flat(true);
	split_sheet_zoom_in->set_focus_mode(FOCUS_NONE);
	split_sheet_zoom_in->set_tooltip(TTR("Zoom In"));
	split_sheet_zoom_in->connect("pressed", callable_mp(this, &SpriteFramesEditor::_sheet_zoom_in));
	split_sheet_zoom_hb->add_child(split_sheet_zoom_in);

	file_split_sheet = memnew(EditorFileDialog);
	file_split_sheet->set_title(TTR("Create Frames from Sprite Sheet"));
	file_split_sheet->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILE);
	add_child(file_split_sheet);
	file_split_sheet->connect("file_selected", callable_mp(this, &SpriteFramesEditor::_prepare_sprite_sheet));

	// Config scale.
	scale_ratio = 1.2f;
	thumbnail_default_size = 96 * MAX(1, EDSCALE);
	thumbnail_zoom = MAX(1.0f, EDSCALE);
	max_thumbnail_zoom = 8.0f * MAX(1.0f, EDSCALE);
	min_thumbnail_zoom = 0.1f * MAX(1.0f, EDSCALE);
	// Default the zoom to match the editor scale, but don't dezoom on editor scales below 100% to prevent pixel art from looking bad.
	sheet_zoom = MAX(1.0f, EDSCALE);
	max_sheet_zoom = 16.0f * MAX(1.0f, EDSCALE);
	min_sheet_zoom = 0.01f * MAX(1.0f, EDSCALE);
	_zoom_reset();
}

void SpriteFramesEditorPlugin::edit(Object *p_object) {
	frames_editor->set_undo_redo(&get_undo_redo());

	SpriteFrames *s;
	AnimatedSprite2D *animated_sprite = Object::cast_to<AnimatedSprite2D>(p_object);
	if (animated_sprite) {
		s = *animated_sprite->get_sprite_frames();
	} else {
		AnimatedSprite3D *animated_sprite_3d = Object::cast_to<AnimatedSprite3D>(p_object);
		if (animated_sprite_3d) {
			s = *animated_sprite_3d->get_sprite_frames();
		} else {
			s = Object::cast_to<SpriteFrames>(p_object);
		}
	}

	frames_editor->edit(s);
}

bool SpriteFramesEditorPlugin::handles(Object *p_object) const {
	AnimatedSprite2D *animated_sprite = Object::cast_to<AnimatedSprite2D>(p_object);
	AnimatedSprite3D *animated_sprite_3d = Object::cast_to<AnimatedSprite3D>(p_object);
	if (animated_sprite && *animated_sprite->get_sprite_frames()) {
		return true;
	} else if (animated_sprite_3d && *animated_sprite_3d->get_sprite_frames()) {
		return true;
	} else {
		return p_object->is_class("SpriteFrames");
	}
}

void SpriteFramesEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button->show();
		editor->make_bottom_panel_item_visible(frames_editor);
	} else {
		button->hide();
		if (frames_editor->is_visible_in_tree()) {
			editor->hide_bottom_panel();
		}
	}
}

SpriteFramesEditorPlugin::SpriteFramesEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	frames_editor = memnew(SpriteFramesEditor);
	frames_editor->set_custom_minimum_size(Size2(0, 300) * EDSCALE);
	button = editor->add_bottom_panel_item(TTR("SpriteFrames"), frames_editor);
	button->hide();
}

SpriteFramesEditorPlugin::~SpriteFramesEditorPlugin() {
}
