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

		// Update the selected source (thus triggering an update).
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

		TileSetSource *source = *tile_set->get_source(source_id);

		Ref<Texture2D> texture;
		String item_text;

		// Atlas source.
		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			texture = atlas_source->get_texture();
			if (texture.is_valid()) {
				item_text = vformat("%s (id:%d)", texture->get_path().get_file(), source_id);
			} else {
				item_text = vformat(TTR("No Texture Atlas Source (id:%d)"), source_id);
			}
		}

		// Scene collection source.
		TileSetScenesCollectionSource *scene_collection_source = Object::cast_to<TileSetScenesCollectionSource>(source);
		if (scene_collection_source) {
			texture = get_theme_icon("PackedScene", "EditorIcons");
			item_text = vformat(TTR("Scene Collection Source (id:%d)"), source_id);
		}

		// Use default if not valid.
		if (item_text.is_empty()) {
			item_text = vformat(TTR("Unknown Type Source (id:%d)"), source_id);
		}
		if (!texture.is_valid()) {
			texture = missing_texture_texture;
		}

		sources_list->add_item(item_text, texture);
		sources_list->set_item_metadata(i, source_id);
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
		TileSetScenesCollectionSource *scenes_collection_source = Object::cast_to<TileSetScenesCollectionSource>(*tile_set->get_source(source_id));
		if (atlas_source) {
			no_source_selected_label->hide();
			tile_set_atlas_source_editor->edit(*tile_set, atlas_source, source_id);
			tile_set_atlas_source_editor->show();
			tile_set_scenes_collection_source_editor->hide();
		} else if (scenes_collection_source) {
			no_source_selected_label->hide();
			tile_set_atlas_source_editor->hide();
			tile_set_scenes_collection_source_editor->edit(*tile_set, scenes_collection_source, source_id);
			tile_set_scenes_collection_source_editor->show();
		} else {
			no_source_selected_label->show();
			tile_set_atlas_source_editor->hide();
			tile_set_scenes_collection_source_editor->hide();
		}
	} else {
		no_source_selected_label->show();
		tile_set_atlas_source_editor->hide();
		tile_set_scenes_collection_source_editor->hide();
	}
}

void TileSetEditor::_source_add_id_pressed(int p_id_pressed) {
	ERR_FAIL_COND(!tile_set.is_valid());

	switch (p_id_pressed) {
		case 0: {
			int source_id = tile_set->get_next_source_id();

			Ref<TileSetAtlasSource> atlas_source = memnew(TileSetAtlasSource);

			// Add a new source.
			undo_redo->create_action(TTR("Add atlas source"));
			undo_redo->add_do_method(*tile_set, "add_source", atlas_source, source_id);
			undo_redo->add_do_method(*atlas_source, "set_texture_region_size", tile_set->get_tile_size());
			undo_redo->add_undo_method(*tile_set, "remove_source", source_id);
			undo_redo->commit_action();

			_update_atlas_sources_list(source_id);
		} break;
		case 1: {
			int source_id = tile_set->get_next_source_id();

			Ref<TileSetScenesCollectionSource> scene_collection_source = memnew(TileSetScenesCollectionSource);

			// Add a new source.
			undo_redo->create_action(TTR("Add atlas source"));
			undo_redo->add_do_method(*tile_set, "add_source", scene_collection_source, source_id);
			undo_redo->add_undo_method(*tile_set, "remove_source", source_id);
			undo_redo->commit_action();

			_update_atlas_sources_list(source_id);
		} break;
		default:
			ERR_FAIL();
	}
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
				if (tile_set.is_valid()) {
					tile_set->set_edited(true);
				}
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

void TileSetEditor::_undo_redo_inspector_callback(Object *p_undo_redo, Object *p_edited, String p_property, Variant p_new_value) {
	UndoRedo *undo_redo = Object::cast_to<UndoRedo>(p_undo_redo);
	ERR_FAIL_COND(!undo_redo);

#define ADD_UNDO(obj, property) undo_redo->add_undo_property(obj, property, tile_data->get(property));
	TileSet *tile_set = Object::cast_to<TileSet>(p_edited);
	if (tile_set) {
		Vector<String> components = p_property.split("/", true, 3);
		for (int i = 0; i < tile_set->get_source_count(); i++) {
			int source_id = tile_set->get_source_id(i);

			Ref<TileSetAtlasSource> tas = tile_set->get_source(source_id);
			if (tas.is_valid()) {
				for (int j = 0; j < tas->get_tiles_count(); j++) {
					Vector2i tile_id = tas->get_tile_id(j);
					for (int k = 0; k < tas->get_alternative_tiles_count(tile_id); k++) {
						int alternative_id = tas->get_alternative_tile_id(tile_id, k);
						TileData *tile_data = Object::cast_to<TileData>(tas->get_tile_data(tile_id, alternative_id));
						ERR_FAIL_COND(!tile_data);

						if (p_property == "occlusion_layers_count") {
							int new_layer_count = p_new_value;
							int old_layer_count = tile_set->get_occlusion_layers_count();
							if (new_layer_count < old_layer_count) {
								for (int occclusion_layer_index = new_layer_count - 1; occclusion_layer_index < old_layer_count; occclusion_layer_index++) {
									ADD_UNDO(tile_data, vformat("occlusion_layer_%d/polygon", occclusion_layer_index));
								}
							}
						} else if (p_property == "physics_layers_count") {
							int new_layer_count = p_new_value;
							int old_layer_count = tile_set->get_physics_layers_count();
							if (new_layer_count < old_layer_count) {
								for (int physics_layer_index = new_layer_count - 1; physics_layer_index < old_layer_count; physics_layer_index++) {
									ADD_UNDO(tile_data, vformat("physics_layer_%d/shapes_count", physics_layer_index));
									for (int shape_index = 0; shape_index < tile_data->get_collision_shapes_count(physics_layer_index); shape_index++) {
										ADD_UNDO(tile_data, vformat("physics_layer_%d/shape_%d/shape", physics_layer_index, shape_index));
										ADD_UNDO(tile_data, vformat("physics_layer_%d/shape_%d/one_way", physics_layer_index, shape_index));
										ADD_UNDO(tile_data, vformat("physics_layer_%d/shape_%d/one_way_margin", physics_layer_index, shape_index));
									}
								}
							}
						} else if ((p_property == "terrains_sets_count" && tile_data->get_terrain_set() >= (int)p_new_value) ||
								   (components.size() == 2 && components[0].begins_with("terrain_set_") && components[0].trim_prefix("terrain_set_").is_valid_integer() && components[1] == "mode") ||
								   (components.size() == 2 && components[0].begins_with("terrain_set_") && components[0].trim_prefix("terrain_set_").is_valid_integer() && components[1] == "terrains_count" && tile_data->get_terrain_set() == components[0].trim_prefix("terrain_set_").to_int() && (int)p_new_value < tile_set->get_terrains_count(tile_data->get_terrain_set()))) {
							ADD_UNDO(tile_data, "terrain_set");
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_RIGHT_SIDE)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/right_side");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_RIGHT_CORNER)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/right_corner");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_SIDE)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/bottom_right_side");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_RIGHT_CORNER)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/bottom_right_corner");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_SIDE)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/bottom_side");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_CORNER)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/bottom_corner");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_SIDE)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/bottom_left_side");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_BOTTOM_LEFT_CORNER)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/bottom_left_corner");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_LEFT_SIDE)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/left_side");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_LEFT_CORNER)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/left_corner");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_LEFT_SIDE)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/top_left_side");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_LEFT_CORNER)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/top_left_corner");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_SIDE)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/top_side");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_CORNER)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/top_corner");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_RIGHT_SIDE)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/top_right_side");
							}
							if (tile_data->is_valid_peering_bit_terrain(TileSet::CELL_NEIGHBOR_TOP_RIGHT_CORNER)) {
								ADD_UNDO(tile_data, "terrains_peering_bit/top_right_corner");
							}
						} else if (p_property == "navigation_layers_count") {
							int new_layer_count = p_new_value;
							int old_layer_count = tile_set->get_navigation_layers_count();
							if (new_layer_count < old_layer_count) {
								for (int navigation_layer_index = new_layer_count - 1; navigation_layer_index < old_layer_count; navigation_layer_index++) {
									ADD_UNDO(tile_data, vformat("navigation_layer_%d/polygon", navigation_layer_index));
								}
							}
						} else if (p_property == "custom_data_layers_count") {
							int new_layer_count = p_new_value;
							int old_layer_count = tile_set->get_custom_data_layers_count();
							if (new_layer_count < old_layer_count) {
								for (int custom_data_layer_index = new_layer_count - 1; custom_data_layer_index < old_layer_count; custom_data_layer_index++) {
									ADD_UNDO(tile_data, vformat("custom_data_%d", custom_data_layer_index));
								}
							}
						} else if (components.size() == 2 && components[0].begins_with("custom_data_layer_") && components[0].trim_prefix("custom_data_layer_").is_valid_integer() && components[1] == "type") {
							int custom_data_layer = components[0].trim_prefix("custom_data_layer_").is_valid_integer();
							ADD_UNDO(tile_data, vformat("custom_data_%d", custom_data_layer));
						}
					}
				}
			}
		}
	}
#undef ADD_UNDO
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
		return tile_data_y_sort_editor;
	} else if (p_property == "texture_offset") {
		return tile_data_texture_offset_editor;
	} else if (components.size() >= 1 && components[0].begins_with("occlusion_layer_")) {
		return tile_data_occlusion_shape_editor;
	} else if (components.size() >= 1 && components[0].begins_with("physics_layer_")) {
		return tile_data_collision_shape_editor;
	} else if (p_property == "mode" || p_property == "terrain" || (components.size() >= 1 && components[0] == "terrains_peering_bit")) {
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
	tile_set_scenes_collection_source_editor->hide();
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
	sources_list->set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);
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

	sources_add_button = memnew(MenuButton);
	sources_add_button->set_flat(true);
	sources_add_button->get_popup()->add_item(TTR("Atlas"));
	sources_add_button->get_popup()->add_item(TTR("Scenes Collection"));
	sources_add_button->get_popup()->connect("id_pressed", callable_mp(this, &TileSetEditor::_source_add_id_pressed));
	sources_bottom_actions->add_child(sources_add_button);

	// Right side container.
	VBoxContainer *split_container_right_side = memnew(VBoxContainer);
	split_container_right_side->set_h_size_flags(SIZE_EXPAND_FILL);
	split_container_right_side->set_v_size_flags(SIZE_EXPAND_FILL);
	split_container->add_child(split_container_right_side);

	// No source selected.
	no_source_selected_label = memnew(Label);
	no_source_selected_label->set_text(TTR("No TileSet source selected. Select or create a TileSet source."));
	no_source_selected_label->set_h_size_flags(SIZE_EXPAND_FILL);
	no_source_selected_label->set_v_size_flags(SIZE_EXPAND_FILL);
	no_source_selected_label->set_align(Label::ALIGN_CENTER);
	no_source_selected_label->set_valign(Label::VALIGN_CENTER);
	split_container_right_side->add_child(no_source_selected_label);

	// Atlases editor.
	tile_set_atlas_source_editor = memnew(TileSetAtlasSourceEditor);
	tile_set_atlas_source_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	tile_set_atlas_source_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	tile_set_atlas_source_editor->connect("source_id_changed", callable_mp(this, &TileSetEditor::_update_atlas_sources_list));
	split_container_right_side->add_child(tile_set_atlas_source_editor);
	tile_set_atlas_source_editor->hide();

	// Scenes collection editor.
	tile_set_scenes_collection_source_editor = memnew(TileSetScenesCollectionSourceEditor);
	tile_set_scenes_collection_source_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	tile_set_scenes_collection_source_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	tile_set_scenes_collection_source_editor->connect("source_id_changed", callable_mp(this, &TileSetEditor::_update_atlas_sources_list));
	split_container_right_side->add_child(tile_set_scenes_collection_source_editor);
	tile_set_scenes_collection_source_editor->hide();

	// Registers UndoRedo inspector callback.
	EditorNode::get_singleton()->get_editor_data().add_undo_redo_inspector_hook_callback(callable_mp(this, &TileSetEditor::_undo_redo_inspector_callback));
}

TileSetEditor::~TileSetEditor() {
	if (tile_set.is_valid()) {
		tile_set->disconnect("changed", callable_mp(this, &TileSetEditor::_tile_set_changed));
	}

	// Delete tile data editors.
	memdelete(tile_data_texture_offset_editor);
	memdelete(tile_data_y_sort_editor);
	memdelete(tile_data_integer_editor);
	memdelete(tile_data_float_editor);
	memdelete(tile_data_occlusion_shape_editor);
	memdelete(tile_data_collision_shape_editor);
	memdelete(tile_data_terrains_editor);
	memdelete(tile_data_navigation_polygon_editor);
}
