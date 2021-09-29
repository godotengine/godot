/*************************************************************************/
/*  tiles_editor_plugin.cpp                                              */
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

#include "tiles_editor_plugin.h"

#include "core/os/mutex.h"

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/plugins/canvas_item_editor_plugin.h"

#include "scene/2d/tile_map.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/separator.h"
#include "scene/resources/tile_set.h"

#include "tile_set_editor.h"

TilesEditor *TilesEditor::singleton = nullptr;

void TilesEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			tileset_tilemap_switch_button->set_icon(get_theme_icon(SNAME("TileSet"), SNAME("EditorIcons")));
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (tile_map_changed_needs_update) {
				TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
				if (tile_map) {
					tile_set = tile_map->get_tileset();
				}
				_update_switch_button();
				_update_editors();
			}
		} break;
	}
}

void TilesEditor::_pattern_preview_done(const Variant &p_udata) {
	pattern_preview_done.set();
}

void TilesEditor::_thread_func(void *ud) {
	TilesEditor *te = (TilesEditor *)ud;
	te->_thread();
}

void TilesEditor::_thread() {
	pattern_thread_exited.clear();
	while (!pattern_thread_exit.is_set()) {
		pattern_preview_sem.wait();

		pattern_preview_mutex.lock();
		if (pattern_preview_queue.size()) {
			QueueItem item = pattern_preview_queue.front()->get();
			pattern_preview_queue.pop_front();
			pattern_preview_mutex.unlock();

			int thumbnail_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
			thumbnail_size *= EDSCALE;
			Vector2 thumbnail_size2 = Vector2(thumbnail_size, thumbnail_size);

			if (item.pattern.is_valid() && !item.pattern->is_empty()) {
				// Generate the pattern preview
				SubViewport *viewport = memnew(SubViewport);
				viewport->set_size(thumbnail_size2);
				viewport->set_disable_input(true);
				viewport->set_transparent_background(true);
				viewport->set_update_mode(SubViewport::UPDATE_ONCE);

				TileMap *tile_map = memnew(TileMap);
				tile_map->set_tileset(item.tile_set);
				tile_map->set_pattern(0, Vector2(), item.pattern);
				viewport->add_child(tile_map);

				TypedArray<Vector2i> used_cells = tile_map->get_used_cells(0);

				Rect2 encompassing_rect = Rect2();
				encompassing_rect.set_position(tile_map->map_to_world(used_cells[0]));
				for (int i = 0; i < used_cells.size(); i++) {
					Vector2i cell = used_cells[i];
					Vector2 world_pos = tile_map->map_to_world(cell);
					encompassing_rect.expand_to(world_pos);

					// Texture.
					Ref<TileSetAtlasSource> atlas_source = tile_set->get_source(tile_map->get_cell_source_id(0, cell));
					if (atlas_source.is_valid()) {
						Vector2i coords = tile_map->get_cell_atlas_coords(0, cell);
						int alternative = tile_map->get_cell_alternative_tile(0, cell);

						Vector2 center = world_pos - atlas_source->get_tile_effective_texture_offset(coords, alternative);
						encompassing_rect.expand_to(center - atlas_source->get_tile_texture_region(coords).size / 2);
						encompassing_rect.expand_to(center + atlas_source->get_tile_texture_region(coords).size / 2);
					}
				}

				Vector2 scale = thumbnail_size2 / MAX(encompassing_rect.size.x, encompassing_rect.size.y);
				tile_map->set_scale(scale);
				tile_map->set_position(-(scale * encompassing_rect.get_center()) + thumbnail_size2 / 2);

				// Add the viewport at the lasst moment to avoid rendering too early.
				EditorNode::get_singleton()->add_child(viewport);

				pattern_preview_done.clear();
				RS::get_singleton()->request_frame_drawn_callback(const_cast<TilesEditor *>(this), "_pattern_preview_done", Variant());

				while (!pattern_preview_done.is_set()) {
					OS::get_singleton()->delay_usec(10);
				}

				Ref<Image> image = viewport->get_texture()->get_image();
				Ref<ImageTexture> image_texture;
				image_texture.instantiate();
				image_texture->create_from_image(image);

				// Find the index for the given pattern. TODO: optimize.
				Variant args[] = { item.pattern, image_texture };
				const Variant *args_ptr[] = { &args[0], &args[1] };
				Variant r;
				Callable::CallError error;
				item.callback.call(args_ptr, 2, r, error);

				viewport->queue_delete();
			} else {
				pattern_preview_mutex.unlock();
			}
		}
	}
	pattern_thread_exited.set();
}

void TilesEditor::_tile_map_changed() {
	tile_map_changed_needs_update = true;
}

void TilesEditor::_update_switch_button() {
	// Force the buttons status if needed.
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (tile_map && !tile_set.is_valid()) {
		tileset_tilemap_switch_button->set_pressed(false);
	} else if (!tile_map && tile_set.is_valid()) {
		tileset_tilemap_switch_button->set_pressed(true);
	}
}

void TilesEditor::_update_editors() {
	// Set editors visibility.
	tilemap_toolbar->set_visible(!tileset_tilemap_switch_button->is_pressed());
	tilemap_editor->set_visible(!tileset_tilemap_switch_button->is_pressed());
	tileset_toolbar->set_visible(tileset_tilemap_switch_button->is_pressed());
	tileset_editor->set_visible(tileset_tilemap_switch_button->is_pressed());

	// Enable/disable the switch button.
	if (!tileset_tilemap_switch_button->is_pressed()) {
		if (!tile_set.is_valid()) {
			tileset_tilemap_switch_button->set_disabled(true);
			tileset_tilemap_switch_button->set_tooltip(TTR("This TileMap has no assigned TileSet, assign a TileSet to this TileMap to edit it."));
		} else {
			tileset_tilemap_switch_button->set_disabled(false);
			tileset_tilemap_switch_button->set_tooltip(TTR("Switch between TileSet/TileMap editor."));
		}
	} else {
		TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
		if (!tile_map) {
			tileset_tilemap_switch_button->set_disabled(true);
			tileset_tilemap_switch_button->set_tooltip(TTR("You are editing a TileSet resource. Select a TileMap node to paint."));
		} else {
			tileset_tilemap_switch_button->set_disabled(false);
			tileset_tilemap_switch_button->set_tooltip(TTR("Switch between TileSet/TileMap editor."));
		}
	}

	// If tile_map is not edited, we change the edited only if we are not editing a tile_set.
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (tile_map) {
		tilemap_editor->edit(tile_map);
	} else {
		tilemap_editor->edit(nullptr);
	}
	tileset_editor->edit(tile_set);

	// Update the viewport
	CanvasItemEditor::get_singleton()->update_viewport();
}

void TilesEditor::set_sources_lists_current(int p_current) {
	atlas_sources_lists_current = p_current;
}

void TilesEditor::synchronize_sources_list(Object *p_current) {
	ItemList *item_list = Object::cast_to<ItemList>(p_current);
	ERR_FAIL_COND(!item_list);

	if (item_list->is_visible_in_tree()) {
		if (atlas_sources_lists_current < 0 || atlas_sources_lists_current >= item_list->get_item_count()) {
			item_list->deselect_all();
		} else {
			item_list->set_current(atlas_sources_lists_current);
			item_list->emit_signal(SNAME("item_selected"), atlas_sources_lists_current);
		}
	}
}

void TilesEditor::set_atlas_view_transform(float p_zoom, Vector2 p_scroll) {
	atlas_view_zoom = p_zoom;
	atlas_view_scroll = p_scroll;
}

void TilesEditor::synchronize_atlas_view(Object *p_current) {
	TileAtlasView *tile_atlas_view = Object::cast_to<TileAtlasView>(p_current);
	ERR_FAIL_COND(!tile_atlas_view);

	if (tile_atlas_view->is_visible_in_tree()) {
		tile_atlas_view->set_transform(atlas_view_zoom, atlas_view_scroll);
	}
}

void TilesEditor::queue_pattern_preview(Ref<TileSet> p_tile_set, Ref<TileMapPattern> p_pattern, Callable p_callback) {
	ERR_FAIL_COND(!p_tile_set.is_valid());
	ERR_FAIL_COND(!p_pattern.is_valid());
	{
		MutexLock lock(pattern_preview_mutex);
		pattern_preview_queue.push_back({ p_tile_set, p_pattern, p_callback });
	}
	pattern_preview_sem.post();
}

void TilesEditor::edit(Object *p_object) {
	// Disconnect to changes.
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (tile_map) {
		tile_map->disconnect("changed", callable_mp(this, &TilesEditor::_tile_map_changed));
	}

	// Update edited objects.
	tile_set = Ref<TileSet>();
	if (p_object) {
		if (p_object->is_class("TileMap")) {
			tile_map_id = p_object->get_instance_id();
			tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
			tile_set = tile_map->get_tileset();
		} else if (p_object->is_class("TileSet")) {
			tile_set = Ref<TileSet>(p_object);
			if (tile_map) {
				if (tile_map->get_tileset() != tile_set) {
					tile_map = nullptr;
				}
			}
		}

		// Update pressed status button.
		if (p_object->is_class("TileMap")) {
			tileset_tilemap_switch_button->set_pressed(false);
		} else if (p_object->is_class("TileSet")) {
			tileset_tilemap_switch_button->set_pressed(true);
		}
	}

	// Update the editors.
	_update_switch_button();
	_update_editors();

	// Add change listener.
	if (tile_map) {
		tile_map->connect("changed", callable_mp(this, &TilesEditor::_tile_map_changed));
	}
}

void TilesEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_pattern_preview_done", "pattern"), &TilesEditor::_pattern_preview_done);
}

TilesEditor::TilesEditor(EditorNode *p_editor) {
	set_process_internal(true);

	// Update the singleton.
	singleton = this;

	// Toolbar.
	HBoxContainer *toolbar = memnew(HBoxContainer);
	toolbar->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(toolbar);

	// Switch button.
	tileset_tilemap_switch_button = memnew(Button);
	tileset_tilemap_switch_button->set_flat(true);
	tileset_tilemap_switch_button->set_toggle_mode(true);
	tileset_tilemap_switch_button->connect("toggled", callable_mp(this, &TilesEditor::_update_editors).unbind(1));
	toolbar->add_child(tileset_tilemap_switch_button);

	// Tilemap editor.
	tilemap_editor = memnew(TileMapEditor);
	tilemap_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	tilemap_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	tilemap_editor->hide();
	add_child(tilemap_editor);

	tilemap_toolbar = tilemap_editor->get_toolbar();
	toolbar->add_child(tilemap_toolbar);

	// Tileset editor.
	tileset_editor = memnew(TileSetEditor);
	tileset_editor->set_h_size_flags(SIZE_EXPAND_FILL);
	tileset_editor->set_v_size_flags(SIZE_EXPAND_FILL);
	tileset_editor->hide();
	add_child(tileset_editor);

	tileset_toolbar = tileset_editor->get_toolbar();
	toolbar->add_child(tileset_toolbar);

	// Pattern preview generation thread.
	pattern_preview_thread.start(_thread_func, this);

	// Initialization.
	_update_switch_button();
	_update_editors();
}

TilesEditor::~TilesEditor() {
	if (pattern_preview_thread.is_started()) {
		pattern_thread_exit.set();
		pattern_preview_sem.post();
		while (!pattern_thread_exited.is_set()) {
			OS::get_singleton()->delay_usec(10000);
			RenderingServer::get_singleton()->sync(); //sync pending stuff, as thread may be blocked on visual server
		}
		pattern_preview_thread.wait_to_finish();
	}
}

///////////////////////////////////////////////////////////////

void TilesEditorPlugin::_notification(int p_what) {
}

void TilesEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		tiles_editor_button->show();
		editor_node->make_bottom_panel_item_visible(tiles_editor);
	} else {
		editor_node->hide_bottom_panel();
		tiles_editor_button->hide();
	}
}

void TilesEditorPlugin::edit(Object *p_object) {
	tiles_editor->edit(p_object);
}

bool TilesEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("TileMap") || p_object->is_class("TileSet");
}

TilesEditorPlugin::TilesEditorPlugin(EditorNode *p_node) {
	editor_node = p_node;

	tiles_editor = memnew(TilesEditor(p_node));
	tiles_editor->set_custom_minimum_size(Size2(0, 200) * EDSCALE);
	tiles_editor->hide();

	tiles_editor_button = p_node->add_bottom_panel_item(TTR("Tiles"), tiles_editor);
	tiles_editor_button->hide();
}

TilesEditorPlugin::~TilesEditorPlugin() {
}
