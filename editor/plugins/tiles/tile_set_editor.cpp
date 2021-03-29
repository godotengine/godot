/*************************************************************************/
/*  tile_set_editor.cpp                                                  */
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

#include "tile_set_editor.h"

#include "tile_data_editors.h"
#include "tiles_editor_plugin.h"

#include "editor/editor_scale.h"

#include "scene/gui/box_container.h"
#include "scene/gui/control.h"
#include "scene/gui/tab_container.h"

TileSetEditor *TileSetEditor::singleton = nullptr;

void TileSetEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	ERR_FAIL_COND(!tile_set.is_valid());

	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	if (p_from == sources_list) {
		// Handle dropping a texture in the list of atlas resources.
		int source_id = -1;
		int added = 0;
		Dictionary d = p_data;
		Vector<String> files = d["files"];
		for (int i = 0; i < files.size(); i++) {
			Ref<Texture2D> resource = ResourceLoader::load(files[i]);
			if (resource.is_valid()) {
				// Retrieve the id for the next created source.
				source_id = tile_set->get_next_source_id();

				// Actually create the new source.
				Ref<TileSetAtlasSource> atlas_source = memnew(TileSetAtlasSource);
				atlas_source->set_texture(resource);
				undo_redo->create_action(TTR("Add a new atlas source"));
				undo_redo->add_do_method(*tile_set, "add_source", atlas_source, source_id);
				undo_redo->add_do_method(*atlas_source, "set_texture_region_size", tile_set->get_tile_size());
				undo_redo->add_undo_method(*tile_set, "remove_source", source_id);
				undo_redo->commit_action();
				added += 1;
			}
		}

		if (added == 1) {
			tile_set_atlas_source_editor->init_source();
		}

		// Update the selected source (thus trigerring an update).
		_update_atlas_sources_list(source_id);
	}
}

bool TileSetEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	ERR_FAIL_COND_V(!tile_set.is_valid(), false);

	if (p_from == sources_list) {
		Dictionary d = p_data;

		if (!d.has("type")) {
			return false;
		}

		// Check if we have a Texture2D.
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
	}
	return false;
}

void TileSetEditor::_update_atlas_sources_list(int force_selected_id) {
	ERR_FAIL_COND(!tile_set.is_valid());

	// Get the previously selected id.
	int old_selected = -1;
	if (sources_list->get_current() >= 0) {
		int source_id = sources_list->get_item_metadata(sources_list->get_current());
		if (tile_set->has_source(source_id)) {
			old_selected = source_id;
		}
	}

	int to_select = -1;
	if (force_selected_id >= 0) {
		to_select = force_selected_id;
	} else if (old_selected >= 0) {
		to_select = old_selected;
	}

	// Clear the list.
	sources_list->clear();

	// Update the atlas sources.
	for (int i = 0; i < tile_set->get_source_count(); i++) {
		int source_id = tile_set->get_source_id(i);

		// TODO: handle with virtual functions
		TileSetSource *source = *tile_set->get_source(source_id);
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			Ref<Texture2D> texture = atlas_source->get_texture();
			if (texture.is_valid()) {
				sources_list->add_item(vformat("%s - (id:%d)", texture->get_path().get_file(), source_id), texture);
			} else {
				sources_list->add_item(vformat("No texture atlas source - (id:%d)", source_id), missing_texture_texture);
			}
		} else {
			sources_list->add_item(vformat("Unknown type source - (id:%d)", source_id), missing_texture_texture);
		}
		sources_list->set_item_metadata(sources_list->get_item_count() - 1, source_id);
	}

	// Set again the current selected item if needed.
	if (to_select >= 0) {
		for (int i = 0; i < sources_list->get_item_count(); i++) {
			if ((int)sources_list->get_item_metadata(i) == to_select) {
				sources_list->set_current(i);
				if (old_selected != to_select) {
					sources_list->emit_signal("item_selected", sources_list->get_current());
				}
				break;
			}
		}
	}

	// If nothing is selected, select the first entry.
	if (sources_list->get_current() < 0 && sources_list->get_item_count() > 0) {
		sources_list->set_current(0);
		if (old_selected != int(sources_list->get_item_metadata(0))) {
			sources_list->emit_signal("item_selected", sources_list->get_current());
		}
	}

	// If there is no source left, hide all editors and show the label.
	_source_selected(sources_list->get_current());

	// Synchronize the lists.
	TilesEditor::get_singleton()->set_atlas_sources_lists_current(sources_list->get_current());
}

void TileSetEditor::_source_selected(int p_source_index) {
	ERR_FAIL_COND(!tile_set.is_valid());

	// Update the selected source.
	sources_delete_button->set_disabled(p_source_index < 0);

	if (p_source_index >= 0) {
		int source_id = sources_list->get_item_metadata(p_source_index);
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(*tile_set->get_source(source_id));
		if (atlas_source) {
			tile_set_atlas_source_editor->edit(*tile_set, atlas_source, source_id);
			no_source_selected_label->hide();
			tile_set_atlas_source_editor->show();
		} else {
			no_source_selected_label->show();
			tile_set_atlas_source_editor->hide();
		}
	} else {
		no_source_selected_label->show();
		tile_set_atlas_source_editor->hide();
	}
}

void TileSetEditor::_source_add_pressed() {
	ERR_FAIL_COND(!tile_set.is_valid());

	int source_id = tile_set->get_next_source_id();

	Ref<TileSetAtlasSource> atlas_source = memnew(TileSetAtlasSource);

	// Add a new source.
	undo_redo->create_action(TTR("Add atlas source"));
	undo_redo->add_do_method(*tile_set, "add_source", atlas_source, source_id);
	undo_redo->add_do_method(*atlas_source, "set_texture_region_size", tile_set->get_tile_size());
	undo_redo->add_undo_method(*tile_set, "remove_source", source_id);
	undo_redo->commit_action();

	_update_atlas_sources_list(source_id);
}

void TileSetEditor::_source_delete_pressed() {
	ERR_FAIL_COND(!tile_set.is_valid());

	// Update the selected source.
	int to_delete = sources_list->get_item_metadata(sources_list->get_current());

	Ref<TileSetSource> source = tile_set->get_source(to_delete);

	// Remove the source.
	undo_redo->create_action(TTR("Remove source"));
	undo_redo->add_do_method(*tile_set, "remove_source", to_delete);
	undo_redo->add_undo_method(*tile_set, "add_source", source, to_delete);
	undo_redo->commit_action();

	_update_atlas_sources_list();
}

void TileSetEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
			sources_delete_button->set_icon(get_theme_icon("Remove", "EditorIcons"));
			sources_add_button->set_icon(get_theme_icon("Add", "EditorIcons"));
			missing_texture_texture = get_theme_icon("TileSet", "EditorIcons");
			break;
		case NOTIFICATION_INTERNAL_PROCESS:
			if (tile_set_changed_needs_update) {
				_update_atlas_sources_list();
				tile_set_changed_needs_update = false;
			}
			break;
		default:
			break;
	}
}

void TileSetEditor::_tile_set_changed() {
	tile_set_changed_needs_update = true;
}

void TileSetEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &TileSetEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &TileSetEditor::drop_data_fw);
}

TileDataEditor *TileSetEditor::get_tile_data_editor(String p_property) {
	Vector<String> components = String(p_property).split("/", true);

	if (p_property == "z_index") {
		return tile_data_integer_editor;
	} else if (p_property == "probability") {
		return tile_data_float_editor;
	} else if (p_property == "y_sort_origin") {
		return tile_data_position_editor;
	} else if (p_property == "texture_offset") {
		return tile_data_texture_offset_editor;
	} else if (components.size() >= 1 && components[0].begins_with("occlusion_layer_")) {
		return tile_data_occlusion_shape_editor;
	} else if (components.size() >= 1 && components[0].begins_with("physics_layer_")) {
		return tile_data_collision_shape_editor;
	} else if (p_property == "terrain_mode" || p_property == "terrain" || (components.size() >= 1 && components[0] == "terrains_peering_bit")) {
		return tile_data_terrains_editor;
	} else if (components.size() >= 1 && components[0].begins_with("navigation_layer_")) {
		return tile_data_navigation_polygon_editor;
	}

	return nullptr;
}

void TileSetEditor::edit(Ref<TileSet> p_tile_set) {
	if (p_tile_set == tile_set) {
		return;
	}

	// Remove listener.
	if (tile_set.is_valid()) {
		tile_set->disconnect("changed", callable_mp(this, &TileSetEditor::_tile_set_changed));
	}

	// Change the edited object.
	tile_set = p_tile_set;

	// Add the listener again.
	if (tile_set.is_valid()) {
		tile_set->connect("changed", callable_mp(this, &TileSetEditor::_tile_set_changed));
		_update_atlas_sources_list();
	}

	tile_set_atlas_source_editor->hide();
	no_source_selected_label->show();
}

TileSetEditor::TileSetEditor() {
	singleton = this;

	set_process_internal(true);

	// Split container.
	HSplitContainer *split_container = memnew(HSplitContainer);
	split_container->set_name(TTR("Tiles"));
	split_container->set_h_size_flags(SIZE_EXPAND_FILL);
	split_container->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(split_container);

	// Sources list.
	VBoxContainer *split_container_left_side = memnew(VBoxContainer);
	split_container_left_side->set_h_size_flags(SIZE_EXPAND_FILL);
	split_container_left_side->set_v_size_flags(SIZE_EXPAND_FILL);
	split_container_left_side->set_stretch_ratio(0.25);
	split_container_left_side->set_custom_minimum_size(Size2i(70, 0) * EDSCALE);
	split_container->add_child(split_container_left_side);

	sources_list = memnew(ItemList);
	sources_list->set_fixed_icon_size(Size2i(60, 60) * EDSCALE);
	sources_list->set_h_size_flags(SIZE_EXPAND_FILL);
	sources_list->set_v_size_flags(SIZE_EXPAND_FILL);
	sources_list->connect("item_selected", callable_mp(this, &TileSetEditor::_source_selected));
	sources_list->connect("item_selected", callable_mp(TilesEditor::get_singleton(), &TilesEditor::set_atlas_sources_lists_current));
	sources_list->connect("visibility_changed", callable_mp(TilesEditor::get_singleton(), &TilesEditor::synchronize_atlas_sources_list), varray(sources_list));
	sources_list->set_drag_forwarding(this);
	split_container_left_side->add_child(sources_list);

	HBoxContainer *sources_bottom_actions = memnew(HBoxContainer);
	sources_bottom_actions->set_alignment(HBoxContainer::ALIGN_END);
	split_container_left_side->add_child(sources_bottom_actions);

	sources_delete_button = memnew(Button);
	sources_delete_button->set_flat(true);
	sources_delete_button->set_disabled(true);
	sources_delete_button->connect("pressed", callable_mp(this, &TileSetEditor::_source_delete_pressed));
	sources_bottom_actions->add_child(sources_delete_button);

	sources_add_button = memnew(Button);
	sources_add_button->set_flat(true);
	sources_add_button->connect("pressed", callable_mp(this, &TileSetEditor::_source_add_pressed));
	sources_bottom_actions->add_child(sources_add_button);

	// No source selected.
	no_source_selected_label = memnew(Label);
	no_source_selected_label->set_text(TTR("No TileSet source selected. Select or create a TileSet source."));
	no_source_selected_label->set_h_size_flags(SIZE_EXPAND_FILL);
	no_source_selected_label->set_v_size_flags(SIZE_EXPAND_FILL);
	no_source_selected_label->set_align(Label::ALIGN_CENTER);
	no_source_selected_label->set_valign(Label::VALIGN_CENTER);
	split_container->add_child(no_source_selected_label);

	// Atlases editor.
	tile_set_atlas_source_editor = memnew(TileSetAtlasSourceEditor);
	tile_set_atlas_source_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	tile_set_atlas_source_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	tile_set_atlas_source_editor->connect("source_id_changed", callable_mp(this, &TileSetEditor::_update_atlas_sources_list));
	split_container->add_child(tile_set_atlas_source_editor);
	tile_set_atlas_source_editor->hide();
}

TileSetEditor::~TileSetEditor() {
	if (tile_set.is_valid()) {
		tile_set->disconnect("changed", callable_mp(this, &TileSetEditor::_tile_set_changed));
	}

	// Delete tile data editors.
	memdelete(tile_data_texture_offset_editor);
	memdelete(tile_data_position_editor);
	memdelete(tile_data_integer_editor);
	memdelete(tile_data_float_editor);
	memdelete(tile_data_occlusion_shape_editor);
	memdelete(tile_data_collision_shape_editor);
	memdelete(tile_data_terrains_editor);
	memdelete(tile_data_navigation_polygon_editor);
}
