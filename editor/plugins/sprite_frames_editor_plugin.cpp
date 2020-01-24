/*************************************************************************/
/*  sprite_frames_editor_plugin.cpp                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/io/resource_loader.h"
#include "core/project_settings.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "scene/3d/sprite_3d.h"
#include "scene/gui/center_container.h"

void SpriteFramesEditor::_gui_input(Ref<InputEvent> p_event) {
}

void SpriteFramesEditor::_open_sprite_sheet() {

	file_split_sheet->clear_filters();
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("Texture", &extensions);
	for (int i = 0; i < extensions.size(); i++) {
		file_split_sheet->add_filter("*." + extensions[i]);
	}

	file_split_sheet->popup_centered_ratio();
}

void SpriteFramesEditor::_sheet_compute_grid() {

	if (split_sheet_preview->get_texture().is_null())
		return;

	Size2i size = split_sheet_preview->get_size();
	int offx = split_offset_x->get_value();
	int offy = split_offset_y->get_value();

	if (split_mode->get_selected() == SPLIT_GRID) {
		int h = split_grid_h->get_value();
		int v = split_grid_v->get_value();
		int width = (size.width - offx) / h;
		int height = (size.height - offy) / v;
		Size2i grid(h * width + offx, v * height + offy);

		frames_grid.clear();
		for (int y = offy; y < grid.height; y += height) {
			for (int x = offx; x < grid.width; x += width) {
				frames_grid.push_back(Rect2(x, y, width, height));
			}
		}
	} else if (split_mode->get_selected() == SPLIT_PIXEL) {
		int width = split_pixel_x->get_value();
		int height = split_pixel_y->get_value();
		int h = (size.width - offx) / width;
		int v = (size.height - offy) / height;
		Size2i grid(h * width + offx, v * height + offy);

		frames_grid.clear();
		for (int y = offy; y < grid.height; y += height) {
			for (int x = offx; x < grid.width; x += width) {
				frames_grid.push_back(Rect2(x, y, width, height));
			}
		}
	} else if (split_mode->get_selected() == SPLIT_AUTO) {
		if (cache_map.has(split_sheet_preview->get_texture()->get_rid())) {
			frames_grid = cache_map[split_sheet_preview->get_texture()->get_rid()];
		} else {
			_update_autoslice();
		}
	}
}

void SpriteFramesEditor::_update_autoslice() {
	frames_grid.clear();

	Ref<Texture> texture = split_sheet_preview->get_texture();

	for (int y = 0; y < texture->get_height(); y++) {
		for (int x = 0; x < texture->get_width(); x++) {
			if (texture->is_pixel_opaque(x, y)) {
				bool found = false;
				for (List<Rect2>::Element *E = frames_grid.front(); E; E = E->next()) {
					Rect2 grown = E->get().grow(1.5);
					if (grown.has_point(Point2(x, y))) {
						E->get().expand_to(Point2(x, y));
						E->get().expand_to(Point2(x + 1, y + 1));
						x = E->get().position.x + E->get().size.x - 1;
						bool merged = true;
						while (merged) {
							merged = false;
							bool queue_erase = false;
							for (List<Rect2>::Element *F = frames_grid.front(); F; F = F->next()) {
								if (queue_erase) {
									frames_grid.erase(F->prev());
									queue_erase = false;
								}
								if (F == E)
									continue;
								if (E->get().grow(1).intersects(F->get())) {
									E->get().expand_to(F->get().position);
									E->get().expand_to(F->get().position + F->get().size);
									if (F->prev()) {
										F = F->prev();
										frames_grid.erase(F->next());
									} else {
										queue_erase = true;
										// Can't delete the first rect in the list.
									}
									merged = true;
								}
							}
						}
						found = true;
						break;
					}
				}
				if (!found) {
					Rect2 new_rect(x, y, 1, 1);
					frames_grid.push_back(new_rect);
				}
			}
		}
	}
	cache_map[texture->get_rid()] = frames_grid;
}

void SpriteFramesEditor::_sheet_preview_draw() {

	Color accent = get_color("accent_color", "Editor");
	_sheet_compute_grid();

	for (List<Rect2>::Element *E = frames_grid.front(); E; E = E->next()) {
		Rect2 r = E->get();
		r.position += Point2(1, 1);
		r.size -= Size2(2, 2);

		split_sheet_preview->draw_rect(E->get(), Color(1, 1, 1, 0.3), false);
		split_sheet_preview->draw_rect(r, Color(0, 0, 0, 0.3), false);
	}

	if (frames_selected.size() == 0) {
		split_sheet_dialog->get_ok()->set_disabled(true);
		split_sheet_dialog->get_ok()->set_text(TTR("No Frames Selected"));
		return;
	}

	for (List<Rect2>::Element *E = frames_selected.front(); E; E = E->next()) {
		Rect2 r = E->get();
		r.position += Point2(1, 1);
		r.size -= Size2(2, 2);

		split_sheet_preview->draw_rect(E->get(), Color(0, 0, 0, 0.35), true);
		split_sheet_preview->draw_rect(r, accent, false);
	}

	split_sheet_dialog->get_ok()->set_disabled(false);
	split_sheet_dialog->get_ok()->set_text(vformat(TTR("Add %d Frame(s)"), frames_selected.size()));
}

void SpriteFramesEditor::_sheet_preview_input(const Ref<InputEvent> &p_event) {

	Ref<InputEventMouseButton> mb = p_event;
	Ref<InputEventMouseMotion> mm = p_event;
	static bool drag_insert = false;
	static bool drag_delete = false;

	if (mb.is_valid()) {
		drag_insert = mb->get_shift();
		drag_delete = drag_insert & mb->get_control();
		if (mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
			Point2 mpos(mb->get_position().x, mb->get_position().y);
			for (List<Rect2>::Element *E = frames_grid.front(); E; E = E->next()) {
				if (E->get().has_point(mpos)) {
					if (frames_selected.find(E->get()) == NULL) {
						if (!drag_delete) {
							frames_selected.push_back(E->get());
						}
					} else {
						if (drag_delete || !drag_insert) {
							frames_selected.erase(E->get());
						}
					}
					break;
				}
			}
			split_sheet_preview->update();
		} else if (drag_insert && !mb->is_pressed() && mb->get_button_index() == BUTTON_LEFT) {
			drag_insert = false;
			drag_delete = false;
		} else {
			// Mouse Wheel Event
			if (mb->get_button_index() == BUTTON_WHEEL_UP && mb->get_control()) {
				_zoom(ZOOM_IN);
			} else if (mb->get_button_index() == BUTTON_WHEEL_DOWN && mb->get_control()) {
				_zoom(ZOOM_OUT);
			}
		}
	}

	if (mm.is_valid() && (drag_delete || drag_insert)) {
		Point2 mpos(mm->get_position().x, mm->get_position().y);
		for (List<Rect2>::Element *E = frames_grid.front(); E; E = E->next()) {
			if (E->get().has_point(mpos)) {
				if (frames_selected.find(E->get()) == NULL) {
					if (!drag_delete) {
						frames_selected.push_back(E->get());
					}
				} else {
					if (drag_delete) {
						frames_selected.erase(E->get());
					}
				}
				break;
			}
		}
		split_sheet_preview->update();
	}
}

void SpriteFramesEditor::_sheet_add_frames() {

	undo_redo->create_action(TTR("Add Frame"));
	int fc = frames->get_frame_count(edited_anim);
	for (List<Rect2>::Element *E = frames_selected.front(); E; E = E->next()) {
		Ref<AtlasTexture> at;
		at.instance();
		at->set_atlas(split_sheet_preview->get_texture());
		at->set_region(E->get());

		undo_redo->add_do_method(frames, "add_frame", edited_anim, at, -1);
		undo_redo->add_undo_method(frames, "remove_frame", edited_anim, fc);
	}

	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void SpriteFramesEditor::_sheet_clear_all_frames() {

	frames_selected.clear();
	split_sheet_preview->update();
}

void SpriteFramesEditor::_sheet_select_all_frames() {

	for (List<Rect2>::Element *E = frames_grid.front(); E; E = E->next()) {
		if (frames_selected.find(E->get()) == NULL) {
			if (skip_empty->is_pressed() && split_mode->get_selected() != SPLIT_AUTO) {
				bool is_empty = true;
				for (int h = E->get().position.x; is_empty && h < E->get().position.x + E->get().size.width; h++) {
					for (int v = E->get().position.y; is_empty && v < E->get().position.y + E->get().size.height; v++) {
						if (split_sheet_preview->get_texture()->is_pixel_opaque(h, v)) {
							is_empty = false;
						}
					}
				}

				if (!is_empty) {
					frames_selected.push_back(E->get());
				}
			} else {
				frames_selected.push_back(E->get());
			}
		}
	}

	split_sheet_preview->update();
}

void SpriteFramesEditor::_zoom(int p_mode) {

	float scale = split_sheet_preview->get_scale().x;
	switch (p_mode) {
		case ZOOM_OUT:
			scale /= 1.2f;
			break;
		case ZOOM_IN:
			scale *= 1.2f;
			break;
		case ZOOM_FIT: {
			Ref<StyleBox> sb = splite_sheet_scroll->get_stylebox("bg");
			Size2 ss = splite_sheet_scroll->get_size() - sb->get_minimum_size();
			Size2 ts = split_sheet_preview->get_texture()->get_size();
			scale = MIN((ss.width / ts.width), (ss.height / ts.height));
			break;
		}
		case ZOOM_RESET:
		default:
			scale = 1;
			break;
	}

	scale = CLAMP(scale, 0.01f, 16.0f);
	zoom_perc->set_text(vformat("%d%%", int(100 * scale)));
	split_sheet_preview->set_scale(Vector2(scale, scale));
	split_panel->set_custom_minimum_size(split_sheet_preview->get_rect().size * scale);
}

void SpriteFramesEditor::_sheet_spin_changed(double) {

	split_sheet_preview->update();
}

void SpriteFramesEditor::_sheet_mode_changed(int p_mode) {

	hv_container->set_visible(p_mode == SPLIT_GRID);
	xy_container->set_visible(p_mode == SPLIT_PIXEL);
	off_container->set_visible(p_mode != SPLIT_AUTO);
	skip_empty->set_disabled(p_mode == SPLIT_AUTO);
	split_sheet_preview->update();
}

void SpriteFramesEditor::_prepare_sprite_sheet(const String &p_file) {

	Ref<Texture> texture = ResourceLoader::load(p_file);
	if (!texture.is_valid()) {
		EditorNode::get_singleton()->show_warning(TTR("Unable to load images"));
		ERR_FAIL_COND(!texture.is_valid());
	}

	if (texture != split_sheet_preview->get_texture()) {
		//different texture, reset to default
		split_grid_h->set_value(4);
		split_grid_v->set_value(4);
		split_pixel_x->set_value(32);
		split_pixel_y->set_value(32);
		split_offset_x->set_value(0);
		split_offset_y->set_value(0);
		split_mode->select(SPLIT_GRID);
		_sheet_mode_changed(SPLIT_GRID);
		cache_map.clear();
	}

	frames_selected.clear();
	split_sheet_preview->set_scale(Vector2(1, 1));
	split_panel->set_custom_minimum_size(texture->get_size());
	split_sheet_preview->set_texture(texture);
	split_sheet_dialog->popup_centered_ratio(0.65);
}

void SpriteFramesEditor::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			load->set_icon(get_icon("Load", "EditorIcons"));
			load_sheet->set_icon(get_icon("SpriteSheet", "EditorIcons"));
			copy->set_icon(get_icon("ActionCopy", "EditorIcons"));
			paste->set_icon(get_icon("ActionPaste", "EditorIcons"));
			empty->set_icon(get_icon("InsertBefore", "EditorIcons"));
			empty2->set_icon(get_icon("InsertAfter", "EditorIcons"));
			move_up->set_icon(get_icon("MoveLeft", "EditorIcons"));
			move_down->set_icon(get_icon("MoveRight", "EditorIcons"));
			_delete->set_icon(get_icon("Remove", "EditorIcons"));
			new_anim->set_icon(get_icon("New", "EditorIcons"));
			remove_anim->set_icon(get_icon("Remove", "EditorIcons"));
			zoom[ZOOM_OUT]->set_icon(get_icon("ZoomLess", "EditorIcons"));
			zoom[ZOOM_RESET]->set_icon(get_icon("ZoomReset", "EditorIcons"));
			zoom[ZOOM_FIT]->set_icon(get_icon("ZoomFit", "EditorIcons"));
			zoom[ZOOM_IN]->set_icon(get_icon("ZoomMore", "EditorIcons"));
			FALLTHROUGH;
		}
		case NOTIFICATION_THEME_CHANGED: {
			splite_sheet_scroll->add_style_override("bg", get_stylebox("bg", "Tree"));
		} break;
		case NOTIFICATION_READY: {
			add_constant_override("autohide", 1); // Fixes the dragger always showing up.
		} break;
	}
}

void SpriteFramesEditor::_file_load_request(const PoolVector<String> &p_path, int p_at_pos) {

	ERR_FAIL_COND(!frames->has_animation(edited_anim));

	List<Ref<Texture> > resources;

	for (int i = 0; i < p_path.size(); i++) {

		Ref<Texture> resource = ResourceLoader::load(p_path[i]);

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

	if (frames->has_animation(edited_anim)) {
		double value = anim_speed->get_line_edit()->get_text().to_double();
		if (!Math::is_equal_approx(value, frames->get_animation_speed(edited_anim)))
			_animation_fps_changed(value);
	}

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

	if (updating)
		return;

	if (!frames->has_animation(edited_anim))
		return;

	delete_dialog->set_text(TTR("Delete Animation?"));
	delete_dialog->popup_centered_minsize();
}

void SpriteFramesEditor::_animation_remove_confirmed() {

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

	edited_anim = StringName();

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

	Dictionary drag_data = EditorNode::get_singleton()->drag_resource(frame, p_from);
	drag_data["frame"] = idx; // store the frame, in case we want to reorder frames inside 'drop_data_fw'
	return drag_data;
}

bool SpriteFramesEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {

	Dictionary d = p_data;

	if (!d.has("type"))
		return false;

	// reordering frames
	if (d.has("from") && (Object *)(d["from"]) == tree)
		return true;

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
			String file = files[i];
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
			bool reorder = false;
			if (d.has("from") && (Object *)(d["from"]) == tree)
				reorder = true;

			if (reorder) { //drop is from reordering frames
				int from_frame = -1;
				if (d.has("frame"))
					from_frame = d["frame"];

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

		PoolVector<String> files = d["files"];

		_file_load_request(files, at_pos);
	}
}

void SpriteFramesEditor::_bind_methods() {

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
	ClassDB::bind_method(D_METHOD("_animation_remove_confirmed"), &SpriteFramesEditor::_animation_remove_confirmed);
	ClassDB::bind_method(D_METHOD("_animation_loop_changed"), &SpriteFramesEditor::_animation_loop_changed);
	ClassDB::bind_method(D_METHOD("_animation_fps_changed"), &SpriteFramesEditor::_animation_fps_changed);
	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &SpriteFramesEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &SpriteFramesEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &SpriteFramesEditor::drop_data_fw);
	ClassDB::bind_method(D_METHOD("_prepare_sprite_sheet"), &SpriteFramesEditor::_prepare_sprite_sheet);
	ClassDB::bind_method(D_METHOD("_open_sprite_sheet"), &SpriteFramesEditor::_open_sprite_sheet);
	ClassDB::bind_method(D_METHOD("_sheet_preview_draw"), &SpriteFramesEditor::_sheet_preview_draw);
	ClassDB::bind_method(D_METHOD("_sheet_preview_input"), &SpriteFramesEditor::_sheet_preview_input);
	ClassDB::bind_method(D_METHOD("_sheet_spin_changed"), &SpriteFramesEditor::_sheet_spin_changed);
	ClassDB::bind_method(D_METHOD("_sheet_add_frames"), &SpriteFramesEditor::_sheet_add_frames);
	ClassDB::bind_method(D_METHOD("_sheet_select_all_frames"), &SpriteFramesEditor::_sheet_select_all_frames);
	ClassDB::bind_method(D_METHOD("_sheet_clear_all_frames"), &SpriteFramesEditor::_sheet_clear_all_frames);
	ClassDB::bind_method(D_METHOD("_sheet_mode_changed"), &SpriteFramesEditor::_sheet_mode_changed);
	ClassDB::bind_method(D_METHOD("_zoom"), &SpriteFramesEditor::_zoom);
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

	new_anim = memnew(ToolButton);
	new_anim->set_tooltip(TTR("New Animation"));
	hbc_animlist->add_child(new_anim);
	new_anim->connect("pressed", this, "_animation_add");

	remove_anim = memnew(ToolButton);
	remove_anim->set_tooltip(TTR("Remove Animation"));
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
	add_child(vbc);
	vbc->set_h_size_flags(SIZE_EXPAND_FILL);

	sub_vb = memnew(VBoxContainer);
	vbc->add_margin_child(TTR("Animation Frames:"), sub_vb, true);

	HBoxContainer *hbc = memnew(HBoxContainer);
	sub_vb->add_child(hbc);

	load = memnew(ToolButton);
	load->set_tooltip(TTR("Add a Texture from File"));
	hbc->add_child(load);

	load_sheet = memnew(ToolButton);
	load_sheet->set_tooltip(TTR("Add Frames from a Sprite Sheet"));
	hbc->add_child(load_sheet);

	hbc->add_child(memnew(VSeparator));

	copy = memnew(ToolButton);
	copy->set_tooltip(TTR("Copy"));
	hbc->add_child(copy);

	paste = memnew(ToolButton);
	paste->set_tooltip(TTR("Paste"));
	hbc->add_child(paste);

	hbc->add_child(memnew(VSeparator));

	empty = memnew(ToolButton);
	empty->set_tooltip(TTR("Insert Empty (Before)"));
	hbc->add_child(empty);

	empty2 = memnew(ToolButton);
	empty2->set_tooltip(TTR("Insert Empty (After)"));
	hbc->add_child(empty2);

	hbc->add_child(memnew(VSeparator));

	move_up = memnew(ToolButton);
	move_up->set_tooltip(TTR("Move (Before)"));
	hbc->add_child(move_up);

	move_down = memnew(ToolButton);
	move_down->set_tooltip(TTR("Move (After)"));
	hbc->add_child(move_down);

	_delete = memnew(ToolButton);
	_delete->set_tooltip(TTR("Delete"));
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
	tree->set_drag_forwarding(this);

	sub_vb->add_child(tree);

	dialog = memnew(AcceptDialog);
	add_child(dialog);

	load->connect("pressed", this, "_load_pressed");
	load_sheet->connect("pressed", this, "_open_sprite_sheet");
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

	delete_dialog = memnew(ConfirmationDialog);
	add_child(delete_dialog);
	delete_dialog->connect("confirmed", this, "_animation_remove_confirmed");

	// Split sheet Dialog

	split_sheet_dialog = memnew(ConfirmationDialog);
	add_child(split_sheet_dialog);
	split_sheet_dialog->set_title(TTR("Select Frames"));
	split_sheet_dialog->connect("confirmed", this, "_sheet_add_frames");

	VBoxContainer *main_vb = memnew(VBoxContainer);
	split_sheet_dialog->add_child(main_vb);

	HBoxContainer *zoom_hb = memnew(HBoxContainer);
	main_vb->add_child(zoom_hb);
	zoom_hb->set_alignment(BoxContainer::ALIGN_END);

	zoom_perc = memnew(Label);
	zoom_perc->set_text("100%");
	zoom_hb->add_child(zoom_perc);

	String z_label[ZOOM_MAX] = {
		TTR("Zoom Out"),
		TTR("Zoom Reset"),
		TTR("Zoom Fit"),
		TTR("Zoom In")
	};

	for (int i = 0; i < (int)ZOOM_MAX; i++) {
		zoom[i] = memnew(ToolButton);
		zoom[i]->set_tooltip(z_label[i]);
		zoom[i]->connect("pressed", this, "_zoom", varray(i));
		zoom_hb->add_child(zoom[i]);
	}

	HBoxContainer *main_hb = memnew(HBoxContainer);
	main_vb->add_child(main_hb);
	main_hb->set_v_size_flags(SIZE_EXPAND_FILL);
	main_hb->set_h_size_flags(SIZE_EXPAND_FILL);

	VBoxContainer *options_vb = memnew(VBoxContainer);
	options_vb->set_v_size_flags(SIZE_EXPAND_FILL);
	options_vb->set_custom_minimum_size(Size2(180, 0));
	main_hb->add_child(options_vb);

	HBoxContainer *hb_container = memnew(HBoxContainer);
	options_vb->add_child(hb_container);

	Label *ss_label = memnew(Label(TTR("Slice Mode:")));
	hb_container->add_child(ss_label);
	ss_label->set_h_size_flags(SIZE_EXPAND_FILL);
	split_mode = memnew(OptionButton);
	split_mode->add_item("Grid");
	split_mode->add_item("Pixel");
	split_mode->add_item("Auto");
	hb_container->add_child(split_mode);
	split_mode->set_h_size_flags(SIZE_EXPAND_FILL);
	split_mode->connect("item_selected", this, "_sheet_mode_changed");

	hv_container = memnew(VBoxContainer);
	options_vb->add_child(hv_container);
	hv_container->set("custom_constants/separation", 0);

	hb_container = memnew(HBoxContainer);
	hv_container->add_child(hb_container);
	ss_label = memnew(Label(TTR("Horizontal")));
	ss_label->set_h_size_flags(SIZE_EXPAND_FILL);
	hb_container->add_child(ss_label);
	split_grid_h = memnew(SpinBox);
	split_grid_h->set_min(1);
	split_grid_h->set_max(128);
	split_grid_h->set_step(1);
	hb_container->add_child(split_grid_h);
	split_grid_h->set_h_size_flags(SIZE_EXPAND_FILL);
	split_grid_h->connect("value_changed", this, "_sheet_spin_changed");

	hb_container = memnew(HBoxContainer);
	hv_container->add_child(hb_container);
	ss_label = memnew(Label(TTR("Vertical")));
	hb_container->add_child(ss_label);
	ss_label->set_h_size_flags(SIZE_EXPAND_FILL);
	split_grid_v = memnew(SpinBox);
	split_grid_v->set_min(1);
	split_grid_v->set_max(128);
	split_grid_v->set_step(1);
	hb_container->add_child(split_grid_v);
	split_grid_v->set_h_size_flags(SIZE_EXPAND_FILL);
	split_grid_v->connect("value_changed", this, "_sheet_spin_changed");

	xy_container = memnew(VBoxContainer);
	options_vb->add_child(xy_container);
	xy_container->set("custom_constants/separation", 0);

	hb_container = memnew(HBoxContainer);
	xy_container->add_child(hb_container);
	ss_label = memnew(Label(TTR("Width")));
	hb_container->add_child(ss_label);
	ss_label->set_h_size_flags(SIZE_EXPAND_FILL);
	split_pixel_x = memnew(SpinBox);
	split_pixel_x->set_min(8);
	split_pixel_x->set_max(512);
	split_pixel_x->set_step(1);
	hb_container->add_child(split_pixel_x);
	split_pixel_x->set_h_size_flags(SIZE_EXPAND_FILL);
	split_pixel_x->connect("value_changed", this, "_sheet_spin_changed");

	hb_container = memnew(HBoxContainer);
	xy_container->add_child(hb_container);
	ss_label = memnew(Label(TTR("Height")));
	hb_container->add_child(ss_label);
	ss_label->set_h_size_flags(SIZE_EXPAND_FILL);
	split_pixel_y = memnew(SpinBox);
	split_pixel_y->set_min(8);
	split_pixel_y->set_max(512);
	split_pixel_y->set_step(1);
	hb_container->add_child(split_pixel_y);
	split_pixel_y->set_h_size_flags(SIZE_EXPAND_FILL);
	split_pixel_y->connect("value_changed", this, "_sheet_spin_changed");

	off_container = memnew(VBoxContainer);
	options_vb->add_child(off_container);
	off_container->set("custom_constants/separation", 0);

	hb_container = memnew(HBoxContainer);
	off_container->add_child(hb_container);
	ss_label = memnew(Label(TTR("Offset x")));
	hb_container->add_child(ss_label);
	ss_label->set_h_size_flags(SIZE_EXPAND_FILL);
	split_offset_x = memnew(SpinBox);
	split_offset_x->set_min(0);
	split_offset_x->set_max(128);
	split_offset_x->set_step(1);
	hb_container->add_child(split_offset_x);
	split_offset_x->set_h_size_flags(SIZE_EXPAND_FILL);
	split_offset_x->connect("value_changed", this, "_sheet_spin_changed");

	hb_container = memnew(HBoxContainer);
	off_container->add_child(hb_container);
	ss_label = memnew(Label(TTR("Offset y")));
	hb_container->add_child(ss_label);
	ss_label->set_h_size_flags(SIZE_EXPAND_FILL);
	split_offset_y = memnew(SpinBox);
	split_offset_y->set_min(0);
	split_offset_y->set_max(128);
	split_offset_y->set_step(1);
	hb_container->add_child(split_offset_y);
	split_offset_y->set_h_size_flags(SIZE_EXPAND_FILL);
	split_offset_y->connect("value_changed", this, "_sheet_spin_changed");

	options_vb->add_spacer();

	VBoxContainer *vb_bottom = memnew(VBoxContainer);
	options_vb->add_child(vb_bottom);
	vb_bottom->set("custom_constants/separation", 0);

	Button *select_all = memnew(Button);
	select_all->set_text(TTR("Select All Frames"));
	select_all->connect("pressed", this, "_sheet_select_all_frames");
	vb_bottom->add_child(select_all);

	HBoxContainer *hb_right = memnew(HBoxContainer);
	vb_bottom->add_child(hb_right);
	hb_right->set_alignment(BoxContainer::ALIGN_END);

	skip_empty = memnew(CheckBox);
	skip_empty->set_text(TTR("Skip Empty Frames"));
	hb_right->add_child(skip_empty);

	Button *clear_all = memnew(Button);
	clear_all->set_text(TTR("Clear All Frames"));
	clear_all->connect("pressed", this, "_sheet_clear_all_frames");
	options_vb->add_child(clear_all);

	split_sheet_preview = memnew(TextureRect);
	split_sheet_preview->set_expand(false);
	split_sheet_preview->set_mouse_filter(MOUSE_FILTER_PASS);
	split_sheet_preview->connect("draw", this, "_sheet_preview_draw");
	split_sheet_preview->connect("gui_input", this, "_sheet_preview_input");

	splite_sheet_scroll = memnew(ScrollContainer);
	splite_sheet_scroll->set_enable_h_scroll(true);
	splite_sheet_scroll->set_enable_v_scroll(true);
	splite_sheet_scroll->set_h_size_flags(SIZE_EXPAND_FILL);
	splite_sheet_scroll->set_v_size_flags(SIZE_EXPAND_FILL);
	split_panel = memnew(Panel);
	split_panel->add_child(split_sheet_preview);
	split_panel->set_h_size_flags(SIZE_EXPAND_FILL);
	split_panel->set_v_size_flags(SIZE_EXPAND_FILL);
	splite_sheet_scroll->add_child(split_panel);

	main_hb->add_child(splite_sheet_scroll);

	file_split_sheet = memnew(EditorFileDialog);
	file_split_sheet->set_title(TTR("Create Frames from Sprite Sheet"));
	file_split_sheet->set_mode(EditorFileDialog::MODE_OPEN_FILE);
	add_child(file_split_sheet);
	file_split_sheet->connect("file_selected", this, "_prepare_sprite_sheet");
}

void SpriteFramesEditorPlugin::edit(Object *p_object) {

	frames_editor->set_undo_redo(&get_undo_redo());

	SpriteFrames *s;
	AnimatedSprite *animated_sprite = Object::cast_to<AnimatedSprite>(p_object);
	if (animated_sprite) {
		s = *animated_sprite->get_sprite_frames();
	} else {
		s = Object::cast_to<SpriteFrames>(p_object);
	}

	frames_editor->edit(s);
}

bool SpriteFramesEditorPlugin::handles(Object *p_object) const {

	AnimatedSprite *animated_sprite = Object::cast_to<AnimatedSprite>(p_object);
	if (animated_sprite && *animated_sprite->get_sprite_frames()) {
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
		if (frames_editor->is_visible_in_tree())
			editor->hide_bottom_panel();
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
