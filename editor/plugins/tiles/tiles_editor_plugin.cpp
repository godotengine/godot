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

#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/plugins/canvas_item_editor_plugin.h"

#include "scene/2d/tile_map.h"
#include "scene/resources/tile_set.h"

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/separator.h"

#include "tile_set_editor.h"

TilesEditor *TilesEditor::singleton = nullptr;

void TilesEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			tileset_tilemap_switch_button->set_icon(get_theme_icon("TileSet", "EditorIcons"));
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

void TilesEditor::set_atlas_sources_lists_current(int p_current) {
	atlas_sources_lists_current = p_current;
}

void TilesEditor::synchronize_atlas_sources_list(Object *p_current) {
	ItemList *item_list = Object::cast_to<ItemList>(p_current);
	ERR_FAIL_COND(!item_list);

	if (item_list->is_visible_in_tree()) {
		if (atlas_sources_lists_current < 0 || atlas_sources_lists_current >= item_list->get_item_count()) {
			item_list->deselect_all();
		} else {
			item_list->set_current(atlas_sources_lists_current);
			item_list->emit_signal("item_selected", atlas_sources_lists_current);
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
		tile_atlas_view->set_transform(atlas_view_zoom, Vector2(atlas_view_scroll.x, atlas_view_scroll.y));
	}
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

	// Initialization.
	_update_switch_button();
	_update_editors();
}

TilesEditor::~TilesEditor() {
}

///////////////////////////////////////////////////////////////

void TilesEditorPlugin::_notification(int p_what) {
}

void TilesEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		tiles_editor_button->show();
		editor_node->make_bottom_panel_item_visible(tiles_editor);
		//get_tree()->connect_compat("idle_frame", tileset_editor, "_on_workspace_process");
	} else {
		editor_node->hide_bottom_panel();
		tiles_editor_button->hide();
		//get_tree()->disconnect_compat("idle_frame", tileset_editor, "_on_workspace_process");
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
