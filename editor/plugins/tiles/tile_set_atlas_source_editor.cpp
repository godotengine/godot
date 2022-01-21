/*************************************************************************/
/*  tile_set_atlas_source_editor.cpp                                     */
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

#include "tile_set_atlas_source_editor.h"

#include "tiles_editor_plugin.h"

#include "editor/editor_inspector.h"
#include "editor/editor_scale.h"
#include "editor/progress_dialog.h"

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/item_list.h"
#include "scene/gui/separator.h"
#include "scene/gui/split_container.h"
#include "scene/gui/tab_container.h"

#include "core/core_string_names.h"
#include "core/math/geometry_2d.h"
#include "core/os/keyboard.h"

void TileSetAtlasSourceEditor::TileSetAtlasSourceProxyObject::set_id(int p_id) {
	ERR_FAIL_COND(p_id < 0);
	if (source_id == p_id) {
		return;
	}
	ERR_FAIL_COND_MSG(tile_set->has_source(p_id), vformat("Cannot change TileSet Atlas Source ID. Another source exists with id %d.", p_id));

	int previous_source = source_id;
	source_id = p_id; // source_id must be updated before, because it's used by the source list update.
	tile_set->set_source_id(previous_source, p_id);
	emit_signal(SNAME("changed"), "id");
}

int TileSetAtlasSourceEditor::TileSetAtlasSourceProxyObject::get_id() {
	return source_id;
}

bool TileSetAtlasSourceEditor::TileSetAtlasSourceProxyObject::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	if (name == "name") {
		// Use the resource_name property to store the source's name.
		name = "resource_name";
	}
	bool valid = false;
	tile_set_atlas_source->set(name, p_value, &valid);
	if (valid) {
		emit_signal(SNAME("changed"), String(name).utf8().get_data());
	}
	return valid;
}

bool TileSetAtlasSourceEditor::TileSetAtlasSourceProxyObject::_get(const StringName &p_name, Variant &r_ret) const {
	if (!tile_set_atlas_source) {
		return false;
	}
	String name = p_name;
	if (name == "name") {
		// Use the resource_name property to store the source's name.
		name = "resource_name";
	}
	bool valid = false;
	r_ret = tile_set_atlas_source->get(name, &valid);
	return valid;
}

void TileSetAtlasSourceEditor::TileSetAtlasSourceProxyObject::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::STRING, "name", PROPERTY_HINT_NONE, ""));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"));
	p_list->push_back(PropertyInfo(Variant::VECTOR2I, "margins", PROPERTY_HINT_NONE, ""));
	p_list->push_back(PropertyInfo(Variant::VECTOR2I, "separation", PROPERTY_HINT_NONE, ""));
	p_list->push_back(PropertyInfo(Variant::VECTOR2I, "texture_region_size", PROPERTY_HINT_NONE, ""));
	p_list->push_back(PropertyInfo(Variant::BOOL, "use_texture_padding", PROPERTY_HINT_NONE, ""));
}

void TileSetAtlasSourceEditor::TileSetAtlasSourceProxyObject::_bind_methods() {
	// -- Shape and layout --
	ClassDB::bind_method(D_METHOD("set_id", "id"), &TileSetAtlasSourceEditor::TileSetAtlasSourceProxyObject::set_id);
	ClassDB::bind_method(D_METHOD("get_id"), &TileSetAtlasSourceEditor::TileSetAtlasSourceProxyObject::get_id);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "id"), "set_id", "get_id");

	ADD_SIGNAL(MethodInfo("changed", PropertyInfo(Variant::STRING, "what")));
}

void TileSetAtlasSourceEditor::TileSetAtlasSourceProxyObject::edit(Ref<TileSet> p_tile_set, TileSetAtlasSource *p_tile_set_atlas_source, int p_source_id) {
	ERR_FAIL_COND(!p_tile_set.is_valid());
	ERR_FAIL_COND(!p_tile_set_atlas_source);
	ERR_FAIL_COND(p_source_id < 0);
	ERR_FAIL_COND(p_tile_set->get_source(p_source_id) != p_tile_set_atlas_source);

	if (p_tile_set == tile_set && p_tile_set_atlas_source == tile_set_atlas_source && p_source_id == source_id) {
		return;
	}

	// Disconnect to changes.
	if (tile_set_atlas_source) {
		tile_set_atlas_source->disconnect(CoreStringNames::get_singleton()->property_list_changed, callable_mp((Object *)this, &Object::notify_property_list_changed));
	}

	tile_set = p_tile_set;
	tile_set_atlas_source = p_tile_set_atlas_source;
	source_id = p_source_id;

	// Connect to changes.
	if (tile_set_atlas_source) {
		if (!tile_set_atlas_source->is_connected(CoreStringNames::get_singleton()->property_list_changed, callable_mp((Object *)this, &Object::notify_property_list_changed))) {
			tile_set_atlas_source->connect(CoreStringNames::get_singleton()->property_list_changed, callable_mp((Object *)this, &Object::notify_property_list_changed));
		}
	}

	notify_property_list_changed();
}

// -- Proxy object used by the tile inspector --
bool TileSetAtlasSourceEditor::AtlasTileProxyObject::_set(const StringName &p_name, const Variant &p_value) {
	if (!tile_set_atlas_source) {
		return false;
	}

	// ID and size related properties.
	if (tiles.size() == 1) {
		const Vector2i &coords = tiles.front()->get().tile;
		const int &alternative = tiles.front()->get().alternative;

		if (alternative == 0) {
			Vector<String> components = String(p_name).split("/", true, 2);
			if (p_name == "atlas_coords") {
				Vector2i as_vector2i = Vector2i(p_value);
				bool has_room_for_tile = tile_set_atlas_source->has_room_for_tile(as_vector2i, tile_set_atlas_source->get_tile_size_in_atlas(coords), tile_set_atlas_source->get_tile_animation_columns(coords), tile_set_atlas_source->get_tile_animation_separation(coords), tile_set_atlas_source->get_tile_animation_frames_count(coords), coords);
				ERR_FAIL_COND_V(!has_room_for_tile, false);

				if (tiles_set_atlas_source_editor->selection.front()->get().tile == coords) {
					tiles_set_atlas_source_editor->selection.clear();
					tiles_set_atlas_source_editor->selection.insert({ as_vector2i, 0 });
					tiles_set_atlas_source_editor->_update_tile_id_label();
				}

				tile_set_atlas_source->move_tile_in_atlas(coords, as_vector2i);
				tiles.clear();
				tiles.insert({ as_vector2i, 0 });
				emit_signal(SNAME("changed"), "atlas_coords");
				return true;
			} else if (p_name == "size_in_atlas") {
				Vector2i as_vector2i = Vector2i(p_value);
				bool has_room_for_tile = tile_set_atlas_source->has_room_for_tile(coords, as_vector2i, tile_set_atlas_source->get_tile_animation_columns(coords), tile_set_atlas_source->get_tile_animation_separation(coords), tile_set_atlas_source->get_tile_animation_frames_count(coords), coords);
				ERR_FAIL_COND_V_EDMSG(!has_room_for_tile, false, "Invalid size or not enough room in the atlas for the tile.");
				tile_set_atlas_source->move_tile_in_atlas(coords, TileSetSource::INVALID_ATLAS_COORDS, as_vector2i);
				emit_signal(SNAME("changed"), "size_in_atlas");
				return true;
			}
		} else if (alternative > 0) {
			if (p_name == "alternative_id") {
				int as_int = int(p_value);
				ERR_FAIL_COND_V(as_int < 0, false);
				ERR_FAIL_COND_V_MSG(tile_set_atlas_source->has_alternative_tile(coords, as_int), false, vformat("Cannot change alternative tile ID. Another alternative exists with id %d for tile at coords %s.", as_int, coords));

				if (tiles_set_atlas_source_editor->selection.front()->get().alternative == alternative) {
					tiles_set_atlas_source_editor->selection.clear();
					tiles_set_atlas_source_editor->selection.insert({ coords, as_int });
				}

				int previous_alternative_tile = alternative;
				tiles.clear();
				tiles.insert({ coords, as_int }); // tiles must be updated before.
				tile_set_atlas_source->set_alternative_tile_id(coords, previous_alternative_tile, as_int);

				emit_signal(SNAME("changed"), "alternative_id");
				return true;
			}
		}
	}

	// Animation.
	// Check if all tiles have an alternative_id of 0.
	bool all_alternatve_id_zero = true;
	for (TileSelection tile : tiles) {
		if (tile.alternative != 0) {
			all_alternatve_id_zero = false;
			break;
		}
	}

	if (all_alternatve_id_zero) {
		Vector<String> components = String(p_name).split("/", true, 2);
		if (p_name == "animation_columns") {
			for (TileSelection tile : tiles) {
				bool has_room_for_tile = tile_set_atlas_source->has_room_for_tile(tile.tile, tile_set_atlas_source->get_tile_size_in_atlas(tile.tile), p_value, tile_set_atlas_source->get_tile_animation_separation(tile.tile), tile_set_atlas_source->get_tile_animation_frames_count(tile.tile), tile.tile);
				if (!has_room_for_tile) {
					ERR_PRINT("No room for tile");
				} else {
					tile_set_atlas_source->set_tile_animation_columns(tile.tile, p_value);
				}
			}
			emit_signal(SNAME("changed"), "animation_columns");
			return true;
		} else if (p_name == "animation_separation") {
			for (TileSelection tile : tiles) {
				bool has_room_for_tile = tile_set_atlas_source->has_room_for_tile(tile.tile, tile_set_atlas_source->get_tile_size_in_atlas(tile.tile), tile_set_atlas_source->get_tile_animation_columns(tile.tile), p_value, tile_set_atlas_source->get_tile_animation_frames_count(tile.tile), tile.tile);
				if (!has_room_for_tile) {
					ERR_PRINT("No room for tile");
				} else {
					tile_set_atlas_source->set_tile_animation_separation(tile.tile, p_value);
				}
			}
			emit_signal(SNAME("changed"), "animation_separation");
			return true;
		} else if (p_name == "animation_speed") {
			for (TileSelection tile : tiles) {
				tile_set_atlas_source->set_tile_animation_speed(tile.tile, p_value);
			}
			emit_signal(SNAME("changed"), "animation_speed");
			return true;
		} else if (p_name == "animation_frames_count") {
			for (TileSelection tile : tiles) {
				bool has_room_for_tile = tile_set_atlas_source->has_room_for_tile(tile.tile, tile_set_atlas_source->get_tile_size_in_atlas(tile.tile), tile_set_atlas_source->get_tile_animation_columns(tile.tile), tile_set_atlas_source->get_tile_animation_separation(tile.tile), p_value, tile.tile);
				if (!has_room_for_tile) {
					ERR_PRINT("No room for tile");
				} else {
					tile_set_atlas_source->set_tile_animation_frames_count(tile.tile, p_value);
				}
			}
			notify_property_list_changed();
			emit_signal(SNAME("changed"), "animation_separation");
			return true;
		} else if (components.size() == 2 && components[0].begins_with("animation_frame_") && components[0].trim_prefix("animation_frame_").is_valid_int()) {
			for (TileSelection tile : tiles) {
				int frame = components[0].trim_prefix("animation_frame_").to_int();
				if (frame < 0 || frame >= tile_set_atlas_source->get_tile_animation_frames_count(tile.tile)) {
					ERR_PRINT(vformat("No tile animation frame with index %d", frame));
				} else {
					if (components[1] == "duration") {
						tile_set_atlas_source->set_tile_animation_frame_duration(tile.tile, frame, p_value);
						return true;
					}
				}
			}
		}
	}

	// Other properties.
	bool any_valid = false;
	for (Set<TileSelection>::Element *E = tiles.front(); E; E = E->next()) {
		const Vector2i &coords = E->get().tile;
		const int &alternative = E->get().alternative;

		bool valid = false;
		TileData *tile_data = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(coords, alternative));
		ERR_FAIL_COND_V(!tile_data, false);
		tile_data->set(p_name, p_value, &valid);

		any_valid |= valid;
	}

	if (any_valid) {
		emit_signal(SNAME("changed"), String(p_name).utf8().get_data());
	}

	return any_valid;
}

bool TileSetAtlasSourceEditor::AtlasTileProxyObject::_get(const StringName &p_name, Variant &r_ret) const {
	if (!tile_set_atlas_source) {
		return false;
	}

	// ID and size related properties.s
	if (tiles.size() == 1) {
		const Vector2i &coords = tiles.front()->get().tile;
		const int &alternative = tiles.front()->get().alternative;

		if (alternative == 0) {
			Vector<String> components = String(p_name).split("/", true, 2);
			if (p_name == "atlas_coords") {
				r_ret = coords;
				return true;
			} else if (p_name == "size_in_atlas") {
				r_ret = tile_set_atlas_source->get_tile_size_in_atlas(coords);
				return true;
			}
		} else if (alternative > 0) {
			if (p_name == "alternative_id") {
				r_ret = alternative;
				return true;
			}
		}
	}

	// Animation.
	// Check if all tiles have an alternative_id of 0.
	bool all_alternatve_id_zero = true;
	for (TileSelection tile : tiles) {
		if (tile.alternative != 0) {
			all_alternatve_id_zero = false;
			break;
		}
	}

	if (all_alternatve_id_zero) {
		const Vector2i &coords = tiles.front()->get().tile;

		Vector<String> components = String(p_name).split("/", true, 2);
		if (p_name == "animation_columns") {
			r_ret = tile_set_atlas_source->get_tile_animation_columns(coords);
			return true;
		} else if (p_name == "animation_separation") {
			r_ret = tile_set_atlas_source->get_tile_animation_separation(coords);
			return true;
		} else if (p_name == "animation_speed") {
			r_ret = tile_set_atlas_source->get_tile_animation_speed(coords);
			return true;
		} else if (p_name == "animation_frames_count") {
			r_ret = tile_set_atlas_source->get_tile_animation_frames_count(coords);
			return true;
		} else if (components.size() == 2 && components[0].begins_with("animation_frame_") && components[0].trim_prefix("animation_frame_").is_valid_int()) {
			int frame = components[0].trim_prefix("animation_frame_").to_int();
			if (components[1] == "duration") {
				if (frame < 0 || frame >= tile_set_atlas_source->get_tile_animation_frames_count(coords)) {
					return false;
				}
				r_ret = tile_set_atlas_source->get_tile_animation_frame_duration(coords, frame);
				return true;
			}
		}
	}

	for (Set<TileSelection>::Element *E = tiles.front(); E; E = E->next()) {
		// Return the first tile with a property matching the name.
		// Note: It's a little bit annoying, but the behavior is the same the one in MultiNodeEdit.
		const Vector2i &coords = E->get().tile;
		const int &alternative = E->get().alternative;

		TileData *tile_data = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(coords, alternative));
		ERR_FAIL_COND_V(!tile_data, false);

		bool valid = false;
		r_ret = tile_data->get(p_name, &valid);
		if (valid) {
			return true;
		}
	}

	return false;
}

void TileSetAtlasSourceEditor::AtlasTileProxyObject::_get_property_list(List<PropertyInfo> *p_list) const {
	if (!tile_set_atlas_source) {
		return;
	}

	// ID and size related properties.
	if (tiles.size() == 1) {
		if (tiles.front()->get().alternative == 0) {
			p_list->push_back(PropertyInfo(Variant::VECTOR2I, "atlas_coords", PROPERTY_HINT_NONE, ""));
			p_list->push_back(PropertyInfo(Variant::VECTOR2I, "size_in_atlas", PROPERTY_HINT_NONE, ""));
		} else {
			p_list->push_back(PropertyInfo(Variant::INT, "alternative_id", PROPERTY_HINT_NONE, ""));
		}
	}

	// Animation.
	// Check if all tiles have an alternative_id of 0.
	bool all_alternatve_id_zero = true;
	for (TileSelection tile : tiles) {
		if (tile.alternative != 0) {
			all_alternatve_id_zero = false;
			break;
		}
	}

	if (all_alternatve_id_zero) {
		p_list->push_back(PropertyInfo(Variant::NIL, "Animation", PROPERTY_HINT_NONE, "animation_", PROPERTY_USAGE_GROUP));
		p_list->push_back(PropertyInfo(Variant::INT, "animation_columns", PROPERTY_HINT_NONE, ""));
		p_list->push_back(PropertyInfo(Variant::VECTOR2I, "animation_separation", PROPERTY_HINT_NONE, ""));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "animation_speed", PROPERTY_HINT_NONE, ""));
		p_list->push_back(PropertyInfo(Variant::INT, "animation_frames_count", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_ARRAY, "Frames,animation_frame_"));
		// Not optimal, but returns value for the first tile. This is similar to what MultiNodeEdit does.
		if (tile_set_atlas_source->get_tile_animation_frames_count(tiles.front()->get().tile) == 1) {
			p_list->push_back(PropertyInfo(Variant::FLOAT, "animation_frame_0/duration", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_READ_ONLY));
		} else {
			for (int i = 0; i < tile_set_atlas_source->get_tile_animation_frames_count(tiles.front()->get().tile); i++) {
				p_list->push_back(PropertyInfo(Variant::FLOAT, vformat("animation_frame_%d/duration", i), PROPERTY_HINT_NONE, ""));
			}
		}
	}

	// Get the list of properties common to all tiles (similar to what's done in MultiNodeEdit).
	struct PropertyId {
		int occurence_id = 0;
		String property;
		bool operator<(const PropertyId &p_other) const {
			return occurence_id == p_other.occurence_id ? property < p_other.property : occurence_id < p_other.occurence_id;
		}
	};
	struct PLData {
		int uses = 0;
		PropertyInfo property_info;
	};
	Map<PropertyId, PLData> usage;

	List<PLData *> data_list;
	for (Set<TileSelection>::Element *E = tiles.front(); E; E = E->next()) {
		const Vector2i &coords = E->get().tile;
		const int &alternative = E->get().alternative;

		TileData *tile_data = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(coords, alternative));
		ERR_FAIL_COND(!tile_data);

		List<PropertyInfo> list;
		tile_data->get_property_list(&list);

		Map<String, int> counts; // Counts the number of time a property appears (useful for groups that may appear more than once)
		for (List<PropertyInfo>::Element *E_property = list.front(); E_property; E_property = E_property->next()) {
			const String &property_string = E_property->get().name;
			if (!tile_data->is_allowing_transform() && (property_string == "flip_h" || property_string == "flip_v" || property_string == "transpose")) {
				continue;
			}

			if (!counts.has(property_string)) {
				counts[property_string] = 1;
			} else {
				counts[property_string] += 1;
			}

			PropertyInfo stored_property_info = E_property->get();
			stored_property_info.usage |= PROPERTY_USAGE_STORAGE; // Ignore the storage flag in comparing properties.

			PropertyId id = { counts[property_string], property_string };
			if (!usage.has(id)) {
				usage[id] = { 1, stored_property_info };
				data_list.push_back(&usage[id]);
			} else if (usage[id].property_info == stored_property_info) {
				usage[id].uses += 1;
			}
		}
	}

	// Add only properties that are common to all tiles.
	for (const PLData *E : data_list) {
		if (E->uses == tiles.size()) {
			p_list->push_back(E->property_info);
		}
	}
}

void TileSetAtlasSourceEditor::AtlasTileProxyObject::edit(TileSetAtlasSource *p_tile_set_atlas_source, Set<TileSelection> p_tiles) {
	ERR_FAIL_COND(!p_tile_set_atlas_source);
	ERR_FAIL_COND(p_tiles.is_empty());
	for (Set<TileSelection>::Element *E = p_tiles.front(); E; E = E->next()) {
		ERR_FAIL_COND(E->get().tile == TileSetSource::INVALID_ATLAS_COORDS);
		ERR_FAIL_COND(E->get().alternative < 0);
	}

	// Disconnect to changes.
	for (Set<TileSelection>::Element *E = tiles.front(); E; E = E->next()) {
		const Vector2i &coords = E->get().tile;
		const int &alternative = E->get().alternative;

		if (tile_set_atlas_source && tile_set_atlas_source->has_tile(coords) && tile_set_atlas_source->has_alternative_tile(coords, alternative)) {
			TileData *tile_data = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(coords, alternative));
			if (tile_data->is_connected(CoreStringNames::get_singleton()->property_list_changed, callable_mp((Object *)this, &Object::notify_property_list_changed))) {
				tile_data->disconnect(CoreStringNames::get_singleton()->property_list_changed, callable_mp((Object *)this, &Object::notify_property_list_changed));
			}
		}
	}

	tile_set_atlas_source = p_tile_set_atlas_source;
	tiles = Set<TileSelection>(p_tiles);

	// Connect to changes.
	for (Set<TileSelection>::Element *E = p_tiles.front(); E; E = E->next()) {
		const Vector2i &coords = E->get().tile;
		const int &alternative = E->get().alternative;

		if (tile_set_atlas_source->has_tile(coords) && tile_set_atlas_source->has_alternative_tile(coords, alternative)) {
			TileData *tile_data = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(coords, alternative));
			if (!tile_data->is_connected(CoreStringNames::get_singleton()->property_list_changed, callable_mp((Object *)this, &Object::notify_property_list_changed))) {
				tile_data->connect(CoreStringNames::get_singleton()->property_list_changed, callable_mp((Object *)this, &Object::notify_property_list_changed));
			}
		}
	}

	notify_property_list_changed();
}

void TileSetAtlasSourceEditor::AtlasTileProxyObject::_bind_methods() {
	ADD_SIGNAL(MethodInfo("changed", PropertyInfo(Variant::STRING, "what")));
}

void TileSetAtlasSourceEditor::_inspector_property_selected(String p_property) {
	selected_property = p_property;
	_update_atlas_view();
	_update_current_tile_data_editor();
}

void TileSetAtlasSourceEditor::_update_tile_id_label() {
	if (selection.size() == 1) {
		TileSelection selected = selection.front()->get();
		tool_tile_id_label->set_text(vformat("%d, %s, %d", tile_set_atlas_source_id, selected.tile, selected.alternative));
		tool_tile_id_label->set_tooltip(vformat(TTR("Selected tile:\nSource: %d\nAtlas coordinates: %s\nAlternative: %d"), tile_set_atlas_source_id, selected.tile, selected.alternative));
		tool_tile_id_label->show();
	} else {
		tool_tile_id_label->hide();
	}
}

void TileSetAtlasSourceEditor::_update_source_inspector() {
	// Update the proxy object.
	atlas_source_proxy_object->edit(tile_set, tile_set_atlas_source, tile_set_atlas_source_id);
}

void TileSetAtlasSourceEditor::_update_fix_selected_and_hovered_tiles() {
	// Fix selected.
	for (Set<TileSelection>::Element *E = selection.front(); E; E = E->next()) {
		TileSelection selected = E->get();
		if (!tile_set_atlas_source->has_tile(selected.tile) || !tile_set_atlas_source->has_alternative_tile(selected.tile, selected.alternative)) {
			selection.erase(E);
		}
	}

	// Fix hovered.
	if (!tile_set_atlas_source->has_tile(hovered_base_tile_coords)) {
		hovered_base_tile_coords = TileSetSource::INVALID_ATLAS_COORDS;
	}
	Vector2i coords = Vector2i(hovered_alternative_tile_coords.x, hovered_alternative_tile_coords.y);
	int alternative = hovered_alternative_tile_coords.z;
	if (!tile_set_atlas_source->has_tile(coords) || !tile_set_atlas_source->has_alternative_tile(coords, alternative)) {
		hovered_alternative_tile_coords = Vector3i(TileSetSource::INVALID_ATLAS_COORDS.x, TileSetSource::INVALID_ATLAS_COORDS.y, TileSetSource::INVALID_TILE_ALTERNATIVE);
	}
}

void TileSetAtlasSourceEditor::_update_atlas_source_inspector() {
	// Update visibility.
	bool visible = tools_button_group->get_pressed_button() == tool_setup_atlas_source_button;
	atlas_source_inspector_label->set_visible(visible);
	atlas_source_inspector->set_visible(visible);
}

void TileSetAtlasSourceEditor::_update_tile_inspector() {
	// Update visibility.
	if (tools_button_group->get_pressed_button() == tool_select_button) {
		if (!selection.is_empty()) {
			tile_proxy_object->edit(tile_set_atlas_source, selection);
		}
		tile_inspector_label->show();
		tile_inspector->set_visible(!selection.is_empty());
		tile_inspector_no_tile_selected_label->set_visible(selection.is_empty());
	} else {
		tile_inspector_label->hide();
		tile_inspector->hide();
		tile_inspector_no_tile_selected_label->hide();
	}
}

void TileSetAtlasSourceEditor::_update_tile_data_editors() {
	String previously_selected;
	if (tile_data_editors_tree && tile_data_editors_tree->get_selected()) {
		previously_selected = tile_data_editors_tree->get_selected()->get_metadata(0);
	}

	tile_data_editors_tree->clear();

	TreeItem *root = tile_data_editors_tree->create_item();

	TreeItem *group;
#define ADD_TILE_DATA_EDITOR_GROUP(text)               \
	group = tile_data_editors_tree->create_item(root); \
	group->set_custom_bg_color(0, group_color);        \
	group->set_selectable(0, false);                   \
	group->set_disable_folding(true);                  \
	group->set_text(0, text);

	TreeItem *item;
#define ADD_TILE_DATA_EDITOR(parent, text, property)    \
	item = tile_data_editors_tree->create_item(parent); \
	item->set_text(0, text);                            \
	item->set_metadata(0, property);                    \
	if (property == previously_selected) {              \
		item->select(0);                                \
	}

	// Theming.
	tile_data_editors_tree->add_theme_constant_override("vseparation", 1);
	tile_data_editors_tree->add_theme_constant_override("hseparation", 3);

	Color group_color = get_theme_color(SNAME("prop_category"), SNAME("Editor"));

	// List of editors.
	// --- Rendering ---
	ADD_TILE_DATA_EDITOR_GROUP("Rendering");

	ADD_TILE_DATA_EDITOR(group, "Texture Offset", "texture_offset");
	if (!tile_data_editors.has("texture_offset")) {
		TileDataTextureOffsetEditor *tile_data_texture_offset_editor = memnew(TileDataTextureOffsetEditor);
		tile_data_texture_offset_editor->hide();
		tile_data_texture_offset_editor->setup_property_editor(Variant::VECTOR2, "texture_offset");
		tile_data_texture_offset_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
		tile_data_texture_offset_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
		tile_data_editors["texture_offset"] = tile_data_texture_offset_editor;
	}

	ADD_TILE_DATA_EDITOR(group, "Modulate", "modulate");
	if (!tile_data_editors.has("modulate")) {
		TileDataDefaultEditor *tile_data_modulate_editor = memnew(TileDataDefaultEditor());
		tile_data_modulate_editor->hide();
		tile_data_modulate_editor->setup_property_editor(Variant::COLOR, "modulate", "", Color(1.0, 1.0, 1.0, 1.0));
		tile_data_modulate_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
		tile_data_modulate_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
		tile_data_editors["modulate"] = tile_data_modulate_editor;
	}

	ADD_TILE_DATA_EDITOR(group, "Z Index", "z_index");
	if (!tile_data_editors.has("z_index")) {
		TileDataDefaultEditor *tile_data_z_index_editor = memnew(TileDataDefaultEditor());
		tile_data_z_index_editor->hide();
		tile_data_z_index_editor->setup_property_editor(Variant::INT, "z_index");
		tile_data_z_index_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
		tile_data_z_index_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
		tile_data_editors["z_index"] = tile_data_z_index_editor;
	}

	ADD_TILE_DATA_EDITOR(group, "Y Sort Origin", "y_sort_origin");
	if (!tile_data_editors.has("y_sort_origin")) {
		TileDataYSortEditor *tile_data_y_sort_editor = memnew(TileDataYSortEditor);
		tile_data_y_sort_editor->hide();
		tile_data_y_sort_editor->setup_property_editor(Variant::INT, "y_sort_origin");
		tile_data_y_sort_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
		tile_data_y_sort_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
		tile_data_editors["y_sort_origin"] = tile_data_y_sort_editor;
	}

	for (int i = 0; i < tile_set->get_occlusion_layers_count(); i++) {
		ADD_TILE_DATA_EDITOR(group, vformat("Occlusion Layer %d", i), vformat("occlusion_layer_%d", i));
		if (!tile_data_editors.has(vformat("occlusion_layer_%d", i))) {
			TileDataOcclusionShapeEditor *tile_data_occlusion_shape_editor = memnew(TileDataOcclusionShapeEditor());
			tile_data_occlusion_shape_editor->hide();
			tile_data_occlusion_shape_editor->set_occlusion_layer(i);
			tile_data_occlusion_shape_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
			tile_data_occlusion_shape_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
			tile_data_editors[vformat("occlusion_layer_%d", i)] = tile_data_occlusion_shape_editor;
		}
	}
	for (int i = tile_set->get_occlusion_layers_count(); tile_data_editors.has(vformat("occlusion_layer_%d", i)); i++) {
		tile_data_editors[vformat("occlusion_layer_%d", i)]->queue_delete();
		tile_data_editors.erase(vformat("occlusion_layer_%d", i));
	}

	// --- Rendering ---
	ADD_TILE_DATA_EDITOR(root, "Terrains", "terrain_set");
	if (!tile_data_editors.has("terrain_set")) {
		TileDataTerrainsEditor *tile_data_terrains_editor = memnew(TileDataTerrainsEditor);
		tile_data_terrains_editor->hide();
		tile_data_terrains_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
		tile_data_terrains_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
		tile_data_editors["terrain_set"] = tile_data_terrains_editor;
	}

	// --- Miscellaneous ---
	ADD_TILE_DATA_EDITOR(root, "Probability", "probability");
	if (!tile_data_editors.has("probability")) {
		TileDataDefaultEditor *tile_data_probability_editor = memnew(TileDataDefaultEditor());
		tile_data_probability_editor->hide();
		tile_data_probability_editor->setup_property_editor(Variant::FLOAT, "probability", "", 1.0);
		tile_data_probability_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
		tile_data_probability_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
		tile_data_editors["probability"] = tile_data_probability_editor;
	}

	// --- Physics ---
	ADD_TILE_DATA_EDITOR_GROUP("Physics");
	for (int i = 0; i < tile_set->get_physics_layers_count(); i++) {
		ADD_TILE_DATA_EDITOR(group, vformat("Physics Layer %d", i), vformat("physics_layer_%d", i));
		if (!tile_data_editors.has(vformat("physics_layer_%d", i))) {
			TileDataCollisionEditor *tile_data_collision_editor = memnew(TileDataCollisionEditor());
			tile_data_collision_editor->hide();
			tile_data_collision_editor->set_physics_layer(i);
			tile_data_collision_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
			tile_data_collision_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
			tile_data_editors[vformat("physics_layer_%d", i)] = tile_data_collision_editor;
		}
	}
	for (int i = tile_set->get_physics_layers_count(); tile_data_editors.has(vformat("physics_layer_%d", i)); i++) {
		tile_data_editors[vformat("physics_layer_%d", i)]->queue_delete();
		tile_data_editors.erase(vformat("physics_layer_%d", i));
	}

	// --- Navigation ---
	ADD_TILE_DATA_EDITOR_GROUP("Navigation");
	for (int i = 0; i < tile_set->get_navigation_layers_count(); i++) {
		ADD_TILE_DATA_EDITOR(group, vformat("Navigation Layer %d", i), vformat("navigation_layer_%d", i));
		if (!tile_data_editors.has(vformat("navigation_layer_%d", i))) {
			TileDataNavigationEditor *tile_data_navigation_editor = memnew(TileDataNavigationEditor());
			tile_data_navigation_editor->hide();
			tile_data_navigation_editor->set_navigation_layer(i);
			tile_data_navigation_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
			tile_data_navigation_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
			tile_data_editors[vformat("navigation_layer_%d", i)] = tile_data_navigation_editor;
		}
	}
	for (int i = tile_set->get_navigation_layers_count(); tile_data_editors.has(vformat("navigation_layer_%d", i)); i++) {
		tile_data_editors[vformat("navigation_layer_%d", i)]->queue_delete();
		tile_data_editors.erase(vformat("navigation_layer_%d", i));
	}

	// --- Custom Data ---
	ADD_TILE_DATA_EDITOR_GROUP("Custom Data");
	for (int i = 0; i < tile_set->get_custom_data_layers_count(); i++) {
		if (tile_set->get_custom_data_name(i).is_empty()) {
			ADD_TILE_DATA_EDITOR(group, vformat("Custom Data %d", i), vformat("custom_data_%d", i));
		} else {
			ADD_TILE_DATA_EDITOR(group, tile_set->get_custom_data_name(i), vformat("custom_data_%d", i));
		}
		if (!tile_data_editors.has(vformat("custom_data_%d", i))) {
			TileDataDefaultEditor *tile_data_custom_data_editor = memnew(TileDataDefaultEditor());
			tile_data_custom_data_editor->hide();
			tile_data_custom_data_editor->setup_property_editor(tile_set->get_custom_data_type(i), vformat("custom_data_%d", i), tile_set->get_custom_data_name(i));
			tile_data_custom_data_editor->connect("needs_redraw", callable_mp((CanvasItem *)tile_atlas_control_unscaled, &Control::update));
			tile_data_custom_data_editor->connect("needs_redraw", callable_mp((CanvasItem *)alternative_tiles_control_unscaled, &Control::update));
			tile_data_editors[vformat("custom_data_%d", i)] = tile_data_custom_data_editor;
		}
	}
	for (int i = tile_set->get_custom_data_layers_count(); tile_data_editors.has(vformat("custom_data_%d", i)); i++) {
		tile_data_editors[vformat("custom_data_%d", i)]->queue_delete();
		tile_data_editors.erase(vformat("custom_data_%d", i));
	}

#undef ADD_TILE_DATA_EDITOR_GROUP
#undef ADD_TILE_DATA_EDITOR

	// Add tile data editors as children.
	for (KeyValue<String, TileDataEditor *> &E : tile_data_editors) {
		// Tile Data Editor.
		TileDataEditor *tile_data_editor = E.value;
		if (!tile_data_editor->is_inside_tree()) {
			tile_data_painting_editor_container->add_child(tile_data_editor);
		}
		tile_data_editor->set_tile_set(tile_set);

		// Toolbar.
		Control *toolbar = tile_data_editor->get_toolbar();
		if (!toolbar->is_inside_tree()) {
			tool_settings_tile_data_toolbar_container->add_child(toolbar);
		}
		toolbar->hide();
	}

	// Update visibility.
	bool is_visible = tools_button_group->get_pressed_button() == tool_paint_button;
	tile_data_editor_dropdown_button->set_visible(is_visible);
	if (tile_data_editors_tree->get_selected()) {
		tile_data_editor_dropdown_button->set_text(tile_data_editors_tree->get_selected()->get_text(0));
	} else {
		tile_data_editor_dropdown_button->set_text(TTR("Select a property editor"));
	}
	tile_data_editors_label->set_visible(is_visible);
}

void TileSetAtlasSourceEditor::_update_current_tile_data_editor() {
	// Find the property to use.
	String property;
	if (tools_button_group->get_pressed_button() == tool_select_button && tile_inspector->is_visible() && !tile_inspector->get_selected_path().is_empty()) {
		Vector<String> components = tile_inspector->get_selected_path().split("/");
		if (components.size() >= 1) {
			property = components[0];

			// Workaround for terrains as they don't have a common first component.
			if (property.begins_with("terrains_")) {
				property = "terrain_set";
			}
		}
	} else if (tools_button_group->get_pressed_button() == tool_paint_button && tile_data_editors_tree->get_selected()) {
		property = tile_data_editors_tree->get_selected()->get_metadata(0);
		tile_data_editor_dropdown_button->set_text(tile_data_editors_tree->get_selected()->get_text(0));
	}

	// Hide all editors but the current one.
	for (const KeyValue<String, TileDataEditor *> &E : tile_data_editors) {
		E.value->hide();
		E.value->get_toolbar()->hide();
	}
	if (tile_data_editors.has(property)) {
		current_tile_data_editor = tile_data_editors[property];
	} else {
		current_tile_data_editor = nullptr;
	}

	// Get the correct editor for the TileData's property.
	if (current_tile_data_editor) {
		current_tile_data_editor_toolbar = current_tile_data_editor->get_toolbar();
		current_property = property;
		current_tile_data_editor->set_visible(tools_button_group->get_pressed_button() == tool_paint_button);
		current_tile_data_editor_toolbar->set_visible(tools_button_group->get_pressed_button() == tool_paint_button);
	}
}

void TileSetAtlasSourceEditor::_tile_data_editor_dropdown_button_draw() {
	if (!has_theme_icon(SNAME("arrow"), SNAME("OptionButton"))) {
		return;
	}

	RID ci = tile_data_editor_dropdown_button->get_canvas_item();
	Ref<Texture2D> arrow = Control::get_theme_icon(SNAME("arrow"), SNAME("OptionButton"));
	Color clr = Color(1, 1, 1);
	if (get_theme_constant(SNAME("modulate_arrow"))) {
		switch (tile_data_editor_dropdown_button->get_draw_mode()) {
			case BaseButton::DRAW_PRESSED:
				clr = get_theme_color(SNAME("font_pressed_color"));
				break;
			case BaseButton::DRAW_HOVER:
				clr = get_theme_color(SNAME("font_hover_color"));
				break;
			case BaseButton::DRAW_DISABLED:
				clr = get_theme_color(SNAME("font_disabled_color"));
				break;
			default:
				if (tile_data_editor_dropdown_button->has_focus()) {
					clr = get_theme_color(SNAME("font_focus_color"));
				} else {
					clr = get_theme_color(SNAME("font_color"));
				}
		}
	}

	Size2 size = tile_data_editor_dropdown_button->get_size();

	Point2 ofs;
	if (is_layout_rtl()) {
		ofs = Point2(get_theme_constant(SNAME("arrow_margin"), SNAME("OptionButton")), int(Math::abs((size.height - arrow->get_height()) / 2)));
	} else {
		ofs = Point2(size.width - arrow->get_width() - get_theme_constant(SNAME("arrow_margin"), SNAME("OptionButton")), int(Math::abs((size.height - arrow->get_height()) / 2)));
	}
	arrow->draw(ci, ofs, clr);
}

void TileSetAtlasSourceEditor::_tile_data_editor_dropdown_button_pressed() {
	Size2 size = tile_data_editor_dropdown_button->get_size();
	tile_data_editors_popup->set_position(tile_data_editor_dropdown_button->get_screen_position() + Size2(0, size.height * get_global_transform().get_scale().y));
	tile_data_editors_popup->set_size(Size2(size.width, 0));
	tile_data_editors_popup->popup();
}

void TileSetAtlasSourceEditor::_tile_data_editors_tree_selected() {
	tile_data_editors_popup->call_deferred(SNAME("hide"));
	_update_current_tile_data_editor();
	tile_atlas_control->update();
	tile_atlas_control_unscaled->update();
	alternative_tiles_control->update();
	alternative_tiles_control_unscaled->update();
}

void TileSetAtlasSourceEditor::_update_atlas_view() {
	// Update the atlas display.
	tile_atlas_view->set_atlas_source(*tile_set, tile_set_atlas_source, tile_set_atlas_source_id);

	// Create a bunch of buttons to add alternative tiles.
	for (int i = 0; i < alternative_tiles_control->get_child_count(); i++) {
		alternative_tiles_control->get_child(i)->queue_delete();
	}

	Vector2i pos;
	Vector2 texture_region_base_size = tile_set_atlas_source->get_texture_region_size();
	int texture_region_base_size_min = MIN(texture_region_base_size.x, texture_region_base_size.y);
	for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
		Vector2i tile_id = tile_set_atlas_source->get_tile_id(i);
		int alternative_count = tile_set_atlas_source->get_alternative_tiles_count(tile_id);
		if (alternative_count > 1) {
			// Compute the right extremity of alternative.
			int y_increment = 0;
			pos.x = 0;
			for (int j = 1; j < alternative_count; j++) {
				int alternative_id = tile_set_atlas_source->get_alternative_tile_id(tile_id, j);
				Rect2i rect = tile_atlas_view->get_alternative_tile_rect(tile_id, alternative_id);
				pos.x = MAX(pos.x, rect.get_end().x);
				y_increment = MAX(y_increment, rect.size.y);
			}

			// Create and position the button.
			Button *button = memnew(Button);
			alternative_tiles_control->add_child(button);
			button->set_flat(true);
			button->set_icon(get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
			button->add_theme_style_override("normal", memnew(StyleBoxEmpty));
			button->add_theme_style_override("hover", memnew(StyleBoxEmpty));
			button->add_theme_style_override("focus", memnew(StyleBoxEmpty));
			button->add_theme_style_override("pressed", memnew(StyleBoxEmpty));
			button->connect("pressed", callable_mp(tile_set_atlas_source, &TileSetAtlasSource::create_alternative_tile), varray(tile_id, TileSetSource::INVALID_TILE_ALTERNATIVE));
			button->set_rect(Rect2(Vector2(pos.x, pos.y + (y_increment - texture_region_base_size.y) / 2.0), Vector2(texture_region_base_size_min, texture_region_base_size_min)));
			button->set_expand_icon(true);

			pos.y += y_increment;
		}
	}
	tile_atlas_view->set_padding(Side::SIDE_RIGHT, texture_region_base_size_min);

	// Redraw everything.
	tile_atlas_control->update();
	tile_atlas_control_unscaled->update();
	alternative_tiles_control->update();
	alternative_tiles_control_unscaled->update();
	tile_atlas_view->update();

	// Synchronize atlas view.
	TilesEditorPlugin::get_singleton()->synchronize_atlas_view(tile_atlas_view);
}

void TileSetAtlasSourceEditor::_update_toolbar() {
	// Show the tools and settings.
	if (tools_button_group->get_pressed_button() == tool_setup_atlas_source_button) {
		if (current_tile_data_editor_toolbar) {
			current_tile_data_editor_toolbar->hide();
		}
		tool_settings_vsep->show();
		tools_settings_erase_button->show();
		tool_advanced_menu_buttom->show();
	} else if (tools_button_group->get_pressed_button() == tool_select_button) {
		if (current_tile_data_editor_toolbar) {
			current_tile_data_editor_toolbar->hide();
		}
		tool_settings_vsep->hide();
		tools_settings_erase_button->hide();
		tool_advanced_menu_buttom->hide();
	} else if (tools_button_group->get_pressed_button() == tool_paint_button) {
		if (current_tile_data_editor_toolbar) {
			current_tile_data_editor_toolbar->show();
		}
		tool_settings_vsep->hide();
		tools_settings_erase_button->hide();
		tool_advanced_menu_buttom->hide();
	}
}

void TileSetAtlasSourceEditor::_tile_atlas_control_mouse_exited() {
	hovered_base_tile_coords = TileSetSource::INVALID_ATLAS_COORDS;
	tile_atlas_control->update();
	tile_atlas_control_unscaled->update();
	tile_atlas_view->update();
}

void TileSetAtlasSourceEditor::_tile_atlas_view_transform_changed() {
	tile_atlas_control->update();
	tile_atlas_control_unscaled->update();
}

void TileSetAtlasSourceEditor::_tile_atlas_control_gui_input(const Ref<InputEvent> &p_event) {
	// Update the hovered coords.
	hovered_base_tile_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());

	// Forward the event to the current tile data editor if we are in the painting mode.
	if (tools_button_group->get_pressed_button() == tool_paint_button) {
		if (current_tile_data_editor) {
			current_tile_data_editor->forward_painting_atlas_gui_input(tile_atlas_view, tile_set_atlas_source, p_event);
		}
		// Update only what's needed.
		tile_set_changed_needs_update = false;

		tile_atlas_control->update();
		tile_atlas_control_unscaled->update();
		alternative_tiles_control->update();
		alternative_tiles_control_unscaled->update();
		tile_atlas_view->update();
		return;
	} else {
		// Handle the event.
		Ref<InputEventMouseMotion> mm = p_event;
		if (mm.is_valid()) {
			Vector2i start_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_start_mouse_pos);
			Vector2i last_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_last_mouse_pos);
			Vector2i new_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());

			Vector2i grid_size = tile_set_atlas_source->get_atlas_grid_size();

			if (drag_type == DRAG_TYPE_NONE) {
				if (selection.size() == 1) {
					// Change the cursor depending on the hovered thing.
					TileSelection selected = selection.front()->get();
					if (selected.tile != TileSetSource::INVALID_ATLAS_COORDS && selected.alternative == 0) {
						Vector2 mouse_local_pos = tile_atlas_control->get_local_mouse_position();
						Vector2i size_in_atlas = tile_set_atlas_source->get_tile_size_in_atlas(selected.tile);
						Rect2 region = tile_set_atlas_source->get_tile_texture_region(selected.tile);
						Size2 zoomed_size = resize_handle->get_size() / tile_atlas_view->get_zoom();
						Rect2 rect = region.grow_individual(zoomed_size.x, zoomed_size.y, 0, 0);
						const Vector2i coords[] = { Vector2i(0, 0), Vector2i(1, 0), Vector2i(1, 1), Vector2i(0, 1) };
						const Vector2i directions[] = { Vector2i(0, -1), Vector2i(1, 0), Vector2i(0, 1), Vector2i(-1, 0) };
						CursorShape cursor_shape = CURSOR_ARROW;
						bool can_grow[4];
						for (int i = 0; i < 4; i++) {
							can_grow[i] = tile_set_atlas_source->has_room_for_tile(selected.tile + directions[i], tile_set_atlas_source->get_tile_size_in_atlas(selected.tile), tile_set_atlas_source->get_tile_animation_columns(selected.tile), tile_set_atlas_source->get_tile_animation_separation(selected.tile), tile_set_atlas_source->get_tile_animation_frames_count(selected.tile), selected.tile);
							can_grow[i] |= (i % 2 == 0) ? size_in_atlas.y > 1 : size_in_atlas.x > 1;
						}
						for (int i = 0; i < 4; i++) {
							Vector2 pos = rect.position + rect.size * coords[i];
							if (can_grow[i] && can_grow[(i + 3) % 4] && Rect2(pos, zoomed_size).has_point(mouse_local_pos)) {
								cursor_shape = (i % 2) ? CURSOR_BDIAGSIZE : CURSOR_FDIAGSIZE;
							}
							Vector2 next_pos = rect.position + rect.size * coords[(i + 1) % 4];
							if (can_grow[i] && Rect2((pos + next_pos) / 2.0, zoomed_size).has_point(mouse_local_pos)) {
								cursor_shape = (i % 2) ? CURSOR_HSIZE : CURSOR_VSIZE;
							}
						}
						tile_atlas_control->set_default_cursor_shape(cursor_shape);
					}
				}
			} else if (drag_type == DRAG_TYPE_CREATE_BIG_TILE) {
				// Create big tile.
				new_base_tiles_coords = new_base_tiles_coords.max(Vector2i(0, 0)).min(grid_size - Vector2i(1, 1));

				Rect2i new_rect = Rect2i(start_base_tiles_coords, new_base_tiles_coords - start_base_tiles_coords).abs();
				new_rect.size += Vector2i(1, 1);
				// Check if the new tile can fit in the new rect.
				if (tile_set_atlas_source->has_room_for_tile(new_rect.position, new_rect.size, tile_set_atlas_source->get_tile_animation_columns(drag_current_tile), tile_set_atlas_source->get_tile_animation_separation(drag_current_tile), tile_set_atlas_source->get_tile_animation_frames_count(drag_current_tile), drag_current_tile)) {
					// Move and resize the tile.
					tile_set_atlas_source->move_tile_in_atlas(drag_current_tile, new_rect.position, new_rect.size);
					drag_current_tile = new_rect.position;
				}
			} else if (drag_type == DRAG_TYPE_CREATE_TILES) {
				// Create tiles.
				last_base_tiles_coords = last_base_tiles_coords.max(Vector2i(0, 0)).min(grid_size - Vector2i(1, 1));
				new_base_tiles_coords = new_base_tiles_coords.max(Vector2i(0, 0)).min(grid_size - Vector2i(1, 1));

				Vector<Point2i> line = Geometry2D::bresenham_line(last_base_tiles_coords, new_base_tiles_coords);
				for (int i = 0; i < line.size(); i++) {
					if (tile_set_atlas_source->get_tile_at_coords(line[i]) == TileSetSource::INVALID_ATLAS_COORDS) {
						tile_set_atlas_source->create_tile(line[i]);
						drag_modified_tiles.insert(line[i]);
					}
				}

				drag_last_mouse_pos = tile_atlas_control->get_local_mouse_position();

			} else if (drag_type == DRAG_TYPE_REMOVE_TILES) {
				// Remove tiles.
				last_base_tiles_coords = last_base_tiles_coords.max(Vector2i(0, 0)).min(grid_size - Vector2i(1, 1));
				new_base_tiles_coords = new_base_tiles_coords.max(Vector2i(0, 0)).min(grid_size - Vector2i(1, 1));

				Vector<Point2i> line = Geometry2D::bresenham_line(last_base_tiles_coords, new_base_tiles_coords);
				for (int i = 0; i < line.size(); i++) {
					Vector2i base_tile_coords = tile_set_atlas_source->get_tile_at_coords(line[i]);
					if (base_tile_coords != TileSetSource::INVALID_ATLAS_COORDS) {
						drag_modified_tiles.insert(base_tile_coords);
					}
				}

				drag_last_mouse_pos = tile_atlas_control->get_local_mouse_position();
			} else if (drag_type == DRAG_TYPE_MOVE_TILE) {
				// Move tile.
				Vector2 mouse_offset = (Vector2(tile_set_atlas_source->get_tile_size_in_atlas(drag_current_tile)) / 2.0 - Vector2(0.5, 0.5)) * tile_set->get_tile_size();
				Vector2i coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position() - mouse_offset);
				coords = coords.max(Vector2i(0, 0)).min(grid_size - Vector2i(1, 1));
				if (drag_current_tile != coords && tile_set_atlas_source->has_room_for_tile(coords, tile_set_atlas_source->get_tile_size_in_atlas(drag_current_tile), tile_set_atlas_source->get_tile_animation_columns(drag_current_tile), tile_set_atlas_source->get_tile_animation_separation(drag_current_tile), tile_set_atlas_source->get_tile_animation_frames_count(drag_current_tile), drag_current_tile)) {
					tile_set_atlas_source->move_tile_in_atlas(drag_current_tile, coords);
					selection.clear();
					selection.insert({ coords, 0 });
					drag_current_tile = coords;

					// Update only what's needed.
					tile_set_changed_needs_update = false;
					_update_tile_inspector();
					_update_atlas_view();
					_update_tile_id_label();
					_update_current_tile_data_editor();
				}
			} else if (drag_type == DRAG_TYPE_MAY_POPUP_MENU) {
				if (Vector2(drag_start_mouse_pos).distance_to(tile_atlas_control->get_local_mouse_position()) > 5.0 * EDSCALE) {
					drag_type = DRAG_TYPE_NONE;
				}
			} else if (drag_type >= DRAG_TYPE_RESIZE_TOP_LEFT && drag_type <= DRAG_TYPE_RESIZE_LEFT) {
				// Resizing a tile.
				new_base_tiles_coords = new_base_tiles_coords.max(Vector2i(-1, -1)).min(grid_size);

				Rect2i old_rect = Rect2i(drag_current_tile, tile_set_atlas_source->get_tile_size_in_atlas(drag_current_tile));
				Rect2i new_rect = old_rect;

				if (drag_type == DRAG_TYPE_RESIZE_LEFT || drag_type == DRAG_TYPE_RESIZE_TOP_LEFT || drag_type == DRAG_TYPE_RESIZE_BOTTOM_LEFT) {
					new_rect.position.x = MIN(new_base_tiles_coords.x + 1, old_rect.get_end().x - 1);
					new_rect.size.x = old_rect.get_end().x - new_rect.position.x;
				}
				if (drag_type == DRAG_TYPE_RESIZE_TOP || drag_type == DRAG_TYPE_RESIZE_TOP_LEFT || drag_type == DRAG_TYPE_RESIZE_TOP_RIGHT) {
					new_rect.position.y = MIN(new_base_tiles_coords.y + 1, old_rect.get_end().y - 1);
					new_rect.size.y = old_rect.get_end().y - new_rect.position.y;
				}

				if (drag_type == DRAG_TYPE_RESIZE_RIGHT || drag_type == DRAG_TYPE_RESIZE_TOP_RIGHT || drag_type == DRAG_TYPE_RESIZE_BOTTOM_RIGHT) {
					new_rect.set_end(Vector2i(MAX(new_base_tiles_coords.x, old_rect.position.x + 1), new_rect.get_end().y));
				}
				if (drag_type == DRAG_TYPE_RESIZE_BOTTOM || drag_type == DRAG_TYPE_RESIZE_BOTTOM_LEFT || drag_type == DRAG_TYPE_RESIZE_BOTTOM_RIGHT) {
					new_rect.set_end(Vector2i(new_rect.get_end().x, MAX(new_base_tiles_coords.y, old_rect.position.y + 1)));
				}

				if (tile_set_atlas_source->has_room_for_tile(new_rect.position, new_rect.size, tile_set_atlas_source->get_tile_animation_columns(drag_current_tile), tile_set_atlas_source->get_tile_animation_separation(drag_current_tile), tile_set_atlas_source->get_tile_animation_frames_count(drag_current_tile), drag_current_tile)) {
					tile_set_atlas_source->move_tile_in_atlas(drag_current_tile, new_rect.position, new_rect.size);
					selection.clear();
					selection.insert({ new_rect.position, 0 });
					drag_current_tile = new_rect.position;

					// Update only what's needed.
					tile_set_changed_needs_update = false;
					_update_tile_inspector();
					_update_atlas_view();
					_update_tile_id_label();
					_update_current_tile_data_editor();
				}
			}

			// Redraw for the hovered tile.
			tile_atlas_control->update();
			tile_atlas_control_unscaled->update();
			alternative_tiles_control->update();
			alternative_tiles_control_unscaled->update();
			tile_atlas_view->update();
			return;
		}

		Ref<InputEventMouseButton> mb = p_event;
		if (mb.is_valid()) {
			Vector2 mouse_local_pos = tile_atlas_control->get_local_mouse_position();
			if (mb->get_button_index() == MouseButton::LEFT) {
				if (mb->is_pressed()) {
					// Left click pressed.
					if (tools_button_group->get_pressed_button() == tool_setup_atlas_source_button) {
						if (tools_settings_erase_button->is_pressed()) {
							// Erasing
							if (mb->is_ctrl_pressed() || mb->is_shift_pressed()) {
								// Remove tiles using rect.

								// Setup the dragging info.
								drag_type = DRAG_TYPE_REMOVE_TILES_USING_RECT;
								drag_start_mouse_pos = mouse_local_pos;
								drag_last_mouse_pos = drag_start_mouse_pos;
							} else {
								// Remove tiles.

								// Setup the dragging info.
								drag_type = DRAG_TYPE_REMOVE_TILES;
								drag_start_mouse_pos = mouse_local_pos;
								drag_last_mouse_pos = drag_start_mouse_pos;

								// Remove a first tile.
								Vector2i coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_start_mouse_pos);
								if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
									coords = tile_set_atlas_source->get_tile_at_coords(coords);
								}
								if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
									drag_modified_tiles.insert(coords);
								}
							}
						} else {
							// Creating
							if (mb->is_shift_pressed()) {
								// Create a big tile.
								Vector2i coords = tile_atlas_view->get_atlas_tile_coords_at_pos(mouse_local_pos);
								if (coords != TileSetSource::INVALID_ATLAS_COORDS && tile_set_atlas_source->get_tile_at_coords(coords) == TileSetSource::INVALID_ATLAS_COORDS) {
									// Setup the dragging info, only if we start on an empty tile.
									drag_type = DRAG_TYPE_CREATE_BIG_TILE;
									drag_start_mouse_pos = mouse_local_pos;
									drag_last_mouse_pos = drag_start_mouse_pos;
									drag_current_tile = coords;

									// Create a tile.
									tile_set_atlas_source->create_tile(coords);
								}
							} else if (mb->is_ctrl_pressed()) {
								// Create tiles using rect.
								drag_type = DRAG_TYPE_CREATE_TILES_USING_RECT;
								drag_start_mouse_pos = mouse_local_pos;
								drag_last_mouse_pos = drag_start_mouse_pos;
							} else {
								// Create tiles.

								// Setup the dragging info.
								drag_type = DRAG_TYPE_CREATE_TILES;
								drag_start_mouse_pos = mouse_local_pos;
								drag_last_mouse_pos = drag_start_mouse_pos;

								// Create a first tile if needed.
								Vector2i coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_start_mouse_pos);
								if (coords != TileSetSource::INVALID_ATLAS_COORDS && tile_set_atlas_source->get_tile_at_coords(coords) == TileSetSource::INVALID_ATLAS_COORDS) {
									tile_set_atlas_source->create_tile(coords);
									drag_modified_tiles.insert(coords);
								}
							}
						}
					} else if (tools_button_group->get_pressed_button() == tool_select_button) {
						// Dragging a handle.
						drag_type = DRAG_TYPE_NONE;
						if (selection.size() == 1) {
							TileSelection selected = selection.front()->get();
							if (selected.tile != TileSetSource::INVALID_ATLAS_COORDS && selected.alternative == 0) {
								Vector2i size_in_atlas = tile_set_atlas_source->get_tile_size_in_atlas(selected.tile);
								Rect2 region = tile_set_atlas_source->get_tile_texture_region(selected.tile);
								Size2 zoomed_size = resize_handle->get_size() / tile_atlas_view->get_zoom();
								Rect2 rect = region.grow_individual(zoomed_size.x, zoomed_size.y, 0, 0);
								const Vector2i coords[] = { Vector2i(0, 0), Vector2i(1, 0), Vector2i(1, 1), Vector2i(0, 1) };
								const Vector2i directions[] = { Vector2i(0, -1), Vector2i(1, 0), Vector2i(0, 1), Vector2i(-1, 0) };
								CursorShape cursor_shape = CURSOR_ARROW;
								bool can_grow[4];
								for (int i = 0; i < 4; i++) {
									can_grow[i] = tile_set_atlas_source->has_room_for_tile(selected.tile + directions[i], tile_set_atlas_source->get_tile_size_in_atlas(selected.tile), tile_set_atlas_source->get_tile_animation_columns(selected.tile), tile_set_atlas_source->get_tile_animation_separation(selected.tile), tile_set_atlas_source->get_tile_animation_frames_count(selected.tile), selected.tile);
									can_grow[i] |= (i % 2 == 0) ? size_in_atlas.y > 1 : size_in_atlas.x > 1;
								}
								for (int i = 0; i < 4; i++) {
									Vector2 pos = rect.position + rect.size * coords[i];
									if (can_grow[i] && can_grow[(i + 3) % 4] && Rect2(pos, zoomed_size).has_point(mouse_local_pos)) {
										drag_type = (DragType)((int)DRAG_TYPE_RESIZE_TOP_LEFT + i * 2);
										drag_start_mouse_pos = mouse_local_pos;
										drag_last_mouse_pos = drag_start_mouse_pos;
										drag_current_tile = selected.tile;
										drag_start_tile_shape = Rect2i(selected.tile, tile_set_atlas_source->get_tile_size_in_atlas(selected.tile));
										cursor_shape = (i % 2) ? CURSOR_BDIAGSIZE : CURSOR_FDIAGSIZE;
									}
									Vector2 next_pos = rect.position + rect.size * coords[(i + 1) % 4];
									if (can_grow[i] && Rect2((pos + next_pos) / 2.0, zoomed_size).has_point(mouse_local_pos)) {
										drag_type = (DragType)((int)DRAG_TYPE_RESIZE_TOP + i * 2);
										drag_start_mouse_pos = mouse_local_pos;
										drag_last_mouse_pos = drag_start_mouse_pos;
										drag_current_tile = selected.tile;
										drag_start_tile_shape = Rect2i(selected.tile, tile_set_atlas_source->get_tile_size_in_atlas(selected.tile));
										cursor_shape = (i % 2) ? CURSOR_HSIZE : CURSOR_VSIZE;
									}
								}
								tile_atlas_control->set_default_cursor_shape(cursor_shape);
							}
						}

						// Selecting then dragging a tile.
						if (drag_type == DRAG_TYPE_NONE) {
							TileSelection selected = { TileSetSource::INVALID_ATLAS_COORDS, TileSetSource::INVALID_TILE_ALTERNATIVE };
							Vector2i coords = tile_atlas_view->get_atlas_tile_coords_at_pos(mouse_local_pos);
							if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
								coords = tile_set_atlas_source->get_tile_at_coords(coords);
								if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
									selected = { coords, 0 };
								}
							}

							bool shift = mb->is_shift_pressed();
							if (!shift && selection.size() == 1 && selected.tile != TileSetSource::INVALID_ATLAS_COORDS && selection.has(selected)) {
								// Start move dragging.
								drag_type = DRAG_TYPE_MOVE_TILE;
								drag_start_mouse_pos = mouse_local_pos;
								drag_last_mouse_pos = drag_start_mouse_pos;
								drag_current_tile = selected.tile;
								drag_start_tile_shape = Rect2i(selected.tile, tile_set_atlas_source->get_tile_size_in_atlas(selected.tile));
								tile_atlas_control->set_default_cursor_shape(CURSOR_MOVE);
							} else {
								// Start selection dragging.
								drag_type = DRAG_TYPE_RECT_SELECT;
								drag_start_mouse_pos = mouse_local_pos;
								drag_last_mouse_pos = drag_start_mouse_pos;
							}
						}
					}
				} else {
					// Left click released.
					_end_dragging();
				}
				tile_atlas_control->update();
				tile_atlas_control_unscaled->update();
				alternative_tiles_control->update();
				alternative_tiles_control_unscaled->update();
				tile_atlas_view->update();
				return;
			} else if (mb->get_button_index() == MouseButton::RIGHT) {
				// Right click pressed.
				if (mb->is_pressed()) {
					drag_type = DRAG_TYPE_MAY_POPUP_MENU;
					drag_start_mouse_pos = tile_atlas_control->get_local_mouse_position();
				} else {
					// Right click released.
					_end_dragging();
				}
				tile_atlas_control->update();
				tile_atlas_control_unscaled->update();
				alternative_tiles_control->update();
				alternative_tiles_control_unscaled->update();
				tile_atlas_view->update();
				return;
			}
		}
	}
}

void TileSetAtlasSourceEditor::_end_dragging() {
	switch (drag_type) {
		case DRAG_TYPE_CREATE_TILES:
			undo_redo->create_action(TTR("Create tiles"));
			for (Set<Vector2i>::Element *E = drag_modified_tiles.front(); E; E = E->next()) {
				undo_redo->add_do_method(tile_set_atlas_source, "create_tile", E->get());
				undo_redo->add_undo_method(tile_set_atlas_source, "remove_tile", E->get());
			}
			undo_redo->commit_action(false);
			break;
		case DRAG_TYPE_CREATE_BIG_TILE:
			undo_redo->create_action(TTR("Create a tile"));
			undo_redo->add_do_method(tile_set_atlas_source, "create_tile", drag_current_tile, tile_set_atlas_source->get_tile_size_in_atlas(drag_current_tile));
			undo_redo->add_undo_method(tile_set_atlas_source, "remove_tile", drag_current_tile);
			undo_redo->commit_action(false);
			break;
		case DRAG_TYPE_REMOVE_TILES: {
			List<PropertyInfo> list;
			tile_set_atlas_source->get_property_list(&list);
			Map<Vector2i, List<const PropertyInfo *>> per_tile = _group_properties_per_tiles(list, tile_set_atlas_source);
			undo_redo->create_action(TTR("Remove tiles"));
			for (Set<Vector2i>::Element *E = drag_modified_tiles.front(); E; E = E->next()) {
				Vector2i coords = E->get();
				undo_redo->add_do_method(tile_set_atlas_source, "remove_tile", coords);
				undo_redo->add_undo_method(tile_set_atlas_source, "create_tile", coords);
				if (per_tile.has(coords)) {
					for (List<const PropertyInfo *>::Element *E_property = per_tile[coords].front(); E_property; E_property = E_property->next()) {
						String property = E_property->get()->name;
						Variant value = tile_set_atlas_source->get(property);
						if (value.get_type() != Variant::NIL) {
							undo_redo->add_undo_method(tile_set_atlas_source, "set", E_property->get()->name, value);
						}
					}
				}
			}
			undo_redo->commit_action();
		} break;
		case DRAG_TYPE_CREATE_TILES_USING_RECT: {
			Vector2i start_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_start_mouse_pos);
			Vector2i new_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
			Rect2i area = Rect2i(start_base_tiles_coords, new_base_tiles_coords - start_base_tiles_coords).abs();
			area.set_end((area.get_end() + Vector2i(1, 1)).min(tile_set_atlas_source->get_atlas_grid_size()));
			undo_redo->create_action(TTR("Create tiles"));
			for (int x = area.get_position().x; x < area.get_end().x; x++) {
				for (int y = area.get_position().y; y < area.get_end().y; y++) {
					Vector2i coords = Vector2i(x, y);
					if (tile_set_atlas_source->get_tile_at_coords(coords) == TileSetSource::INVALID_ATLAS_COORDS) {
						undo_redo->add_do_method(tile_set_atlas_source, "create_tile", coords);
						undo_redo->add_undo_method(tile_set_atlas_source, "remove_tile", coords);
					}
				}
			}
			undo_redo->commit_action();
		} break;
		case DRAG_TYPE_REMOVE_TILES_USING_RECT: {
			Vector2i start_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_start_mouse_pos);
			Vector2i new_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
			Rect2i area = Rect2i(start_base_tiles_coords, new_base_tiles_coords - start_base_tiles_coords).abs();
			area.set_end((area.get_end() + Vector2i(1, 1)).min(tile_set_atlas_source->get_atlas_grid_size()));
			List<PropertyInfo> list;
			tile_set_atlas_source->get_property_list(&list);
			Map<Vector2i, List<const PropertyInfo *>> per_tile = _group_properties_per_tiles(list, tile_set_atlas_source);

			Set<Vector2i> to_delete;
			for (int x = area.get_position().x; x < area.get_end().x; x++) {
				for (int y = area.get_position().y; y < area.get_end().y; y++) {
					Vector2i coords = tile_set_atlas_source->get_tile_at_coords(Vector2i(x, y));
					if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
						to_delete.insert(coords);
					}
				}
			}

			undo_redo->create_action(TTR("Remove tiles"));
			undo_redo->add_do_method(this, "_set_selection_from_array", Array());
			for (Set<Vector2i>::Element *E = to_delete.front(); E; E = E->next()) {
				Vector2i coords = E->get();
				undo_redo->add_do_method(tile_set_atlas_source, "remove_tile", coords);
				undo_redo->add_undo_method(tile_set_atlas_source, "create_tile", coords);
				if (per_tile.has(coords)) {
					for (List<const PropertyInfo *>::Element *E_property = per_tile[coords].front(); E_property; E_property = E_property->next()) {
						String property = E_property->get()->name;
						Variant value = tile_set_atlas_source->get(property);
						if (value.get_type() != Variant::NIL) {
							undo_redo->add_undo_method(tile_set_atlas_source, "set", E_property->get()->name, value);
						}
					}
				}
			}
			undo_redo->add_undo_method(this, "_set_selection_from_array", _get_selection_as_array());
			undo_redo->commit_action();
		} break;
		case DRAG_TYPE_MOVE_TILE:
			if (drag_current_tile != drag_start_tile_shape.position) {
				undo_redo->create_action(TTR("Move a tile"));
				undo_redo->add_do_method(tile_set_atlas_source, "move_tile_in_atlas", drag_start_tile_shape.position, drag_current_tile, tile_set_atlas_source->get_tile_size_in_atlas(drag_current_tile));
				undo_redo->add_do_method(this, "_set_selection_from_array", _get_selection_as_array());
				undo_redo->add_undo_method(tile_set_atlas_source, "move_tile_in_atlas", drag_current_tile, drag_start_tile_shape.position, drag_start_tile_shape.size);
				Array array;
				array.push_back(drag_start_tile_shape.position);
				array.push_back(0);
				undo_redo->add_undo_method(this, "_set_selection_from_array", array);
				undo_redo->commit_action(false);
			}
			break;
		case DRAG_TYPE_RECT_SELECT: {
			Vector2i start_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_start_mouse_pos);
			Vector2i new_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
			ERR_FAIL_COND(start_base_tiles_coords == TileSetSource::INVALID_ATLAS_COORDS);
			ERR_FAIL_COND(new_base_tiles_coords == TileSetSource::INVALID_ATLAS_COORDS);

			Rect2i region = Rect2i(start_base_tiles_coords, new_base_tiles_coords - start_base_tiles_coords).abs();
			region.size += Vector2i(1, 1);

			undo_redo->create_action(TTR("Select tiles"));
			undo_redo->add_undo_method(this, "_set_selection_from_array", _get_selection_as_array());

			// Determine if we clear, then add or remove to the selection.
			bool add_to_selection = true;
			if (Input::get_singleton()->is_key_pressed(Key::SHIFT)) {
				Vector2i coords = tile_set_atlas_source->get_tile_at_coords(start_base_tiles_coords);
				if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
					if (selection.has({ coords, 0 })) {
						add_to_selection = false;
					}
				}
			} else {
				selection.clear();
			}

			// Modify the selection.
			for (int x = region.position.x; x < region.get_end().x; x++) {
				for (int y = region.position.y; y < region.get_end().y; y++) {
					Vector2i coords = Vector2i(x, y);
					coords = tile_set_atlas_source->get_tile_at_coords(coords);
					if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
						if (add_to_selection && !selection.has({ coords, 0 })) {
							selection.insert({ coords, 0 });
						} else if (!add_to_selection && selection.has({ coords, 0 })) {
							selection.erase({ coords, 0 });
						}
					}
				}
			}
			_update_tile_inspector();
			_update_tile_id_label();
			_update_current_tile_data_editor();
			undo_redo->add_do_method(this, "_set_selection_from_array", _get_selection_as_array());
			undo_redo->commit_action(false);
		} break;
		case DRAG_TYPE_MAY_POPUP_MENU: {
			Vector2 mouse_local_pos = tile_atlas_control->get_local_mouse_position();
			TileSelection selected = { tile_atlas_view->get_atlas_tile_coords_at_pos(mouse_local_pos), 0 };
			if (selected.tile != TileSetSource::INVALID_ATLAS_COORDS) {
				selected.tile = tile_set_atlas_source->get_tile_at_coords(selected.tile);
			}

			// Set the selection if needed.
			if (selection.size() <= 1) {
				if (selected.tile != TileSetSource::INVALID_ATLAS_COORDS) {
					undo_redo->create_action(TTR("Select tiles"));
					undo_redo->add_undo_method(this, "_set_selection_from_array", _get_selection_as_array());
					selection.clear();
					selection.insert(selected);
					undo_redo->add_do_method(this, "_set_selection_from_array", _get_selection_as_array());
					undo_redo->commit_action(false);
					_update_tile_inspector();
					_update_tile_id_label();
					_update_current_tile_data_editor();
				}
			}

			// Pops up the correct menu, depending on whether we have a tile or not.
			if (selected.tile != TileSetSource::INVALID_ATLAS_COORDS && selection.has(selected)) {
				// We have a tile.
				menu_option_coords = selected.tile;
				menu_option_alternative = 0;
				base_tile_popup_menu->popup(Rect2i(get_global_mouse_position(), Size2i()));
			} else if (hovered_base_tile_coords != TileSetSource::INVALID_ATLAS_COORDS) {
				// We don't have a tile, but can create one.
				menu_option_coords = hovered_base_tile_coords;
				menu_option_alternative = TileSetSource::INVALID_TILE_ALTERNATIVE;
				empty_base_tile_popup_menu->popup(Rect2i(get_global_mouse_position(), Size2i()));
			}
		} break;
		case DRAG_TYPE_RESIZE_TOP_LEFT:
		case DRAG_TYPE_RESIZE_TOP:
		case DRAG_TYPE_RESIZE_TOP_RIGHT:
		case DRAG_TYPE_RESIZE_RIGHT:
		case DRAG_TYPE_RESIZE_BOTTOM_RIGHT:
		case DRAG_TYPE_RESIZE_BOTTOM:
		case DRAG_TYPE_RESIZE_BOTTOM_LEFT:
		case DRAG_TYPE_RESIZE_LEFT:
			if (drag_start_tile_shape != Rect2i(drag_current_tile, tile_set_atlas_source->get_tile_size_in_atlas(drag_current_tile))) {
				undo_redo->create_action(TTR("Resize a tile"));
				undo_redo->add_do_method(tile_set_atlas_source, "move_tile_in_atlas", drag_start_tile_shape.position, drag_current_tile, tile_set_atlas_source->get_tile_size_in_atlas(drag_current_tile));
				undo_redo->add_do_method(this, "_set_selection_from_array", _get_selection_as_array());
				undo_redo->add_undo_method(tile_set_atlas_source, "move_tile_in_atlas", drag_current_tile, drag_start_tile_shape.position, drag_start_tile_shape.size);
				Array array;
				array.push_back(drag_start_tile_shape.position);
				array.push_back(0);
				undo_redo->add_undo_method(this, "_set_selection_from_array", array);
				undo_redo->commit_action(false);
			}
			break;
		default:
			break;
	}

	drag_modified_tiles.clear();
	drag_type = DRAG_TYPE_NONE;
	tile_atlas_control->set_default_cursor_shape(CURSOR_ARROW);
}

Map<Vector2i, List<const PropertyInfo *>> TileSetAtlasSourceEditor::_group_properties_per_tiles(const List<PropertyInfo> &r_list, const TileSetAtlasSource *p_atlas) {
	// Group properties per tile.
	Map<Vector2i, List<const PropertyInfo *>> per_tile;
	for (const List<PropertyInfo>::Element *E_property = r_list.front(); E_property; E_property = E_property->next()) {
		Vector<String> components = String(E_property->get().name).split("/", true, 1);
		if (components.size() >= 1) {
			Vector<String> coord_arr = components[0].split(":");
			if (coord_arr.size() == 2 && coord_arr[0].is_valid_int() && coord_arr[1].is_valid_int()) {
				Vector2i coords = Vector2i(coord_arr[0].to_int(), coord_arr[1].to_int());
				per_tile[coords].push_back(&(E_property->get()));
			}
		}
	}
	return per_tile;
}

void TileSetAtlasSourceEditor::_menu_option(int p_option) {
	switch (p_option) {
		case TILE_DELETE: {
			List<PropertyInfo> list;
			tile_set_atlas_source->get_property_list(&list);
			Map<Vector2i, List<const PropertyInfo *>> per_tile = _group_properties_per_tiles(list, tile_set_atlas_source);
			undo_redo->create_action(TTR("Remove tile"));

			// Remove tiles
			Set<Vector2i> removed;
			for (Set<TileSelection>::Element *E = selection.front(); E; E = E->next()) {
				TileSelection selected = E->get();
				if (selected.alternative == 0) {
					// Remove a tile.
					undo_redo->add_do_method(tile_set_atlas_source, "remove_tile", selected.tile);
					undo_redo->add_undo_method(tile_set_atlas_source, "create_tile", selected.tile);
					removed.insert(selected.tile);
					if (per_tile.has(selected.tile)) {
						for (List<const PropertyInfo *>::Element *E_property = per_tile[selected.tile].front(); E_property; E_property = E_property->next()) {
							String property = E_property->get()->name;
							Variant value = tile_set_atlas_source->get(property);
							if (value.get_type() != Variant::NIL) {
								undo_redo->add_undo_method(tile_set_atlas_source, "set", E_property->get()->name, value);
							}
						}
					}
				}
			}

			// Remove alternatives
			for (Set<TileSelection>::Element *E = selection.front(); E; E = E->next()) {
				TileSelection selected = E->get();
				if (selected.alternative > 0 && !removed.has(selected.tile)) {
					// Remove an alternative tile.
					undo_redo->add_do_method(tile_set_atlas_source, "remove_alternative_tile", selected.tile, selected.alternative);
					undo_redo->add_undo_method(tile_set_atlas_source, "create_alternative_tile", selected.tile, selected.alternative);
					if (per_tile.has(selected.tile)) {
						for (List<const PropertyInfo *>::Element *E_property = per_tile[selected.tile].front(); E_property; E_property = E_property->next()) {
							Vector<String> components = E_property->get()->name.split("/", true, 2);
							if (components.size() >= 2 && components[1].is_valid_int() && components[1].to_int() == selected.alternative) {
								String property = E_property->get()->name;
								Variant value = tile_set_atlas_source->get(property);
								if (value.get_type() != Variant::NIL) {
									undo_redo->add_undo_method(tile_set_atlas_source, "set", E_property->get()->name, value);
								}
							}
						}
					}
				}
			}
			undo_redo->commit_action();
			_update_fix_selected_and_hovered_tiles();
			_update_tile_id_label();
		} break;
		case TILE_CREATE: {
			undo_redo->create_action(TTR("Create a tile"));
			undo_redo->add_do_method(tile_set_atlas_source, "create_tile", menu_option_coords);
			Array array;
			array.push_back(menu_option_coords);
			array.push_back(0);
			undo_redo->add_do_method(this, "_set_selection_from_array", array);
			undo_redo->add_undo_method(tile_set_atlas_source, "remove_tile", menu_option_coords);
			undo_redo->add_undo_method(this, "_set_selection_from_array", _get_selection_as_array());
			undo_redo->commit_action();
			_update_tile_id_label();
		} break;
		case TILE_CREATE_ALTERNATIVE: {
			undo_redo->create_action(TTR("Create tile alternatives"));
			Array array;
			for (Set<TileSelection>::Element *E = selection.front(); E; E = E->next()) {
				if (E->get().alternative == 0) {
					int next_id = tile_set_atlas_source->get_next_alternative_tile_id(E->get().tile);
					undo_redo->add_do_method(tile_set_atlas_source, "create_alternative_tile", E->get().tile, next_id);
					array.push_back(E->get().tile);
					array.push_back(next_id);
					undo_redo->add_undo_method(tile_set_atlas_source, "remove_alternative_tile", E->get().tile, next_id);
				}
			}
			undo_redo->add_do_method(this, "_set_selection_from_array", array);
			undo_redo->add_undo_method(this, "_set_selection_from_array", _get_selection_as_array());
			undo_redo->commit_action();
			_update_tile_id_label();
		} break;
		case ADVANCED_AUTO_CREATE_TILES: {
			_auto_create_tiles();
		} break;
		case ADVANCED_AUTO_REMOVE_TILES: {
			_auto_remove_tiles();
		} break;
	}
}

void TileSetAtlasSourceEditor::_unhandled_key_input(const Ref<InputEvent> &p_event) {
	// Check for shortcuts.
	if (ED_IS_SHORTCUT("tiles_editor/delete_tile", p_event)) {
		if (tools_button_group->get_pressed_button() == tool_select_button && !selection.is_empty()) {
			_menu_option(TILE_DELETE);
			accept_event();
		}
	}
}

void TileSetAtlasSourceEditor::_set_selection_from_array(Array p_selection) {
	ERR_FAIL_COND((p_selection.size() % 2) != 0);
	selection.clear();
	for (int i = 0; i < p_selection.size() / 2; i++) {
		TileSelection selected = { p_selection[i * 2], p_selection[i * 2 + 1] };
		if (tile_set_atlas_source->has_tile(selected.tile) && tile_set_atlas_source->has_alternative_tile(selected.tile, selected.alternative)) {
			selection.insert(selected);
		}
	}
	_update_tile_inspector();
	_update_tile_id_label();
	_update_atlas_view();
	_update_current_tile_data_editor();
}

Array TileSetAtlasSourceEditor::_get_selection_as_array() {
	Array output;
	for (Set<TileSelection>::Element *E = selection.front(); E; E = E->next()) {
		output.push_back(E->get().tile);
		output.push_back(E->get().alternative);
	}
	return output;
}

void TileSetAtlasSourceEditor::_tile_atlas_control_draw() {
	// Colors.
	Color grid_color = EditorSettings::get_singleton()->get("editors/tiles_editor/grid_color");
	Color selection_color = Color().from_hsv(Math::fposmod(grid_color.get_h() + 0.5, 1.0), grid_color.get_s(), grid_color.get_v(), 1.0);

	// Draw the selected tile.
	if (tools_button_group->get_pressed_button() == tool_select_button) {
		for (Set<TileSelection>::Element *E = selection.front(); E; E = E->next()) {
			TileSelection selected = E->get();
			if (selected.alternative == 0) {
				// Draw the rect.
				for (int frame = 0; frame < tile_set_atlas_source->get_tile_animation_frames_count(selected.tile); frame++) {
					Color color = selection_color;
					if (frame > 0) {
						color.a *= 0.3;
					}
					Rect2 region = tile_set_atlas_source->get_tile_texture_region(selected.tile, frame);
					tile_atlas_control->draw_rect(region, color, false);
				}
			}
		}

		if (selection.size() == 1) {
			// Draw the resize handles (only when it's possible to expand).
			TileSelection selected = selection.front()->get();
			if (selected.alternative == 0) {
				Vector2i size_in_atlas = tile_set_atlas_source->get_tile_size_in_atlas(selected.tile);
				Size2 zoomed_size = resize_handle->get_size() / tile_atlas_view->get_zoom();
				Rect2 region = tile_set_atlas_source->get_tile_texture_region(selected.tile);
				Rect2 rect = region.grow_individual(zoomed_size.x, zoomed_size.y, 0, 0);
				Vector2i coords[] = { Vector2i(0, 0), Vector2i(1, 0), Vector2i(1, 1), Vector2i(0, 1) };
				Vector2i directions[] = { Vector2i(0, -1), Vector2i(1, 0), Vector2i(0, 1), Vector2i(-1, 0) };
				bool can_grow[4];
				for (int i = 0; i < 4; i++) {
					can_grow[i] = tile_set_atlas_source->has_room_for_tile(selected.tile + directions[i], tile_set_atlas_source->get_tile_size_in_atlas(selected.tile), tile_set_atlas_source->get_tile_animation_columns(selected.tile), tile_set_atlas_source->get_tile_animation_separation(selected.tile), tile_set_atlas_source->get_tile_animation_frames_count(selected.tile), selected.tile);
					can_grow[i] |= (i % 2 == 0) ? size_in_atlas.y > 1 : size_in_atlas.x > 1;
				}
				for (int i = 0; i < 4; i++) {
					Vector2 pos = rect.position + rect.size * coords[i];
					if (can_grow[i] && can_grow[(i + 3) % 4]) {
						tile_atlas_control->draw_texture_rect(resize_handle, Rect2(pos, zoomed_size), false);
					} else {
						tile_atlas_control->draw_texture_rect(resize_handle_disabled, Rect2(pos, zoomed_size), false);
					}
					Vector2 next_pos = rect.position + rect.size * coords[(i + 1) % 4];
					if (can_grow[i]) {
						tile_atlas_control->draw_texture_rect(resize_handle, Rect2((pos + next_pos) / 2.0, zoomed_size), false);
					} else {
						tile_atlas_control->draw_texture_rect(resize_handle_disabled, Rect2((pos + next_pos) / 2.0, zoomed_size), false);
					}
				}
			}
		}
	}

	if (drag_type == DRAG_TYPE_REMOVE_TILES) {
		// Draw the tiles to be removed.
		for (Set<Vector2i>::Element *E = drag_modified_tiles.front(); E; E = E->next()) {
			for (int frame = 0; frame < tile_set_atlas_source->get_tile_animation_frames_count(E->get()); frame++) {
				tile_atlas_control->draw_rect(tile_set_atlas_source->get_tile_texture_region(E->get(), frame), Color(0.0, 0.0, 0.0), false);
			}
		}
	} else if (drag_type == DRAG_TYPE_RECT_SELECT || drag_type == DRAG_TYPE_REMOVE_TILES_USING_RECT) {
		// Draw tiles to be removed.
		Vector2i start_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_start_mouse_pos);
		Vector2i new_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
		Rect2i area = Rect2i(start_base_tiles_coords, new_base_tiles_coords - start_base_tiles_coords).abs();
		area.set_end((area.get_end() + Vector2i(1, 1)).min(tile_set_atlas_source->get_atlas_grid_size()));

		Color color = Color(0.0, 0.0, 0.0);
		if (drag_type == DRAG_TYPE_RECT_SELECT) {
			color = selection_color.lightened(0.2);
		}

		Set<Vector2i> to_paint;
		for (int x = area.get_position().x; x < area.get_end().x; x++) {
			for (int y = area.get_position().y; y < area.get_end().y; y++) {
				Vector2i coords = tile_set_atlas_source->get_tile_at_coords(Vector2i(x, y));
				if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
					to_paint.insert(coords);
				}
			}
		}

		for (Set<Vector2i>::Element *E = to_paint.front(); E; E = E->next()) {
			Vector2i coords = E->get();
			tile_atlas_control->draw_rect(tile_set_atlas_source->get_tile_texture_region(coords), color, false);
		}
	} else if (drag_type == DRAG_TYPE_CREATE_TILES_USING_RECT) {
		// Draw tiles to be created.
		Vector2i margins = tile_set_atlas_source->get_margins();
		Vector2i separation = tile_set_atlas_source->get_separation();
		Vector2i tile_size = tile_set_atlas_source->get_texture_region_size();

		Vector2i start_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_start_mouse_pos);
		Vector2i new_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
		Rect2i area = Rect2i(start_base_tiles_coords, new_base_tiles_coords - start_base_tiles_coords).abs();
		area.set_end((area.get_end() + Vector2i(1, 1)).min(tile_set_atlas_source->get_atlas_grid_size()));
		for (int x = area.get_position().x; x < area.get_end().x; x++) {
			for (int y = area.get_position().y; y < area.get_end().y; y++) {
				Vector2i coords = Vector2i(x, y);
				if (tile_set_atlas_source->get_tile_at_coords(coords) == TileSetSource::INVALID_ATLAS_COORDS) {
					Vector2i origin = margins + (coords * (tile_size + separation));
					tile_atlas_control->draw_rect(Rect2i(origin, tile_size), Color(1.0, 1.0, 1.0), false);
				}
			}
		}
	}

	// Draw the hovered tile.
	if (drag_type == DRAG_TYPE_REMOVE_TILES_USING_RECT || drag_type == DRAG_TYPE_CREATE_TILES_USING_RECT) {
		// Draw the rect.
		Vector2i start_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(drag_start_mouse_pos);
		Vector2i new_base_tiles_coords = tile_atlas_view->get_atlas_tile_coords_at_pos(tile_atlas_control->get_local_mouse_position());
		Rect2i area = Rect2i(start_base_tiles_coords, new_base_tiles_coords - start_base_tiles_coords).abs();
		area.set_end((area.get_end() + Vector2i(1, 1)).min(tile_set_atlas_source->get_atlas_grid_size()));
		Vector2i margins = tile_set_atlas_source->get_margins();
		Vector2i separation = tile_set_atlas_source->get_separation();
		Vector2i tile_size = tile_set_atlas_source->get_texture_region_size();
		Vector2i origin = margins + (area.position * (tile_size + separation));
		tile_atlas_control->draw_rect(Rect2i(origin, area.size * tile_size), Color(1.0, 1.0, 1.0), false);
	} else {
		Vector2i grid_size = tile_set_atlas_source->get_atlas_grid_size();
		if (hovered_base_tile_coords.x >= 0 && hovered_base_tile_coords.y >= 0 && hovered_base_tile_coords.x < grid_size.x && hovered_base_tile_coords.y < grid_size.y) {
			Vector2i hovered_tile = tile_set_atlas_source->get_tile_at_coords(hovered_base_tile_coords);
			if (hovered_tile != TileSetSource::INVALID_ATLAS_COORDS) {
				// Draw existing hovered tile.
				for (int frame = 0; frame < tile_set_atlas_source->get_tile_animation_frames_count(hovered_tile); frame++) {
					Color color = Color(1.0, 1.0, 1.0);
					if (frame > 0) {
						color.a *= 0.3;
					}
					tile_atlas_control->draw_rect(tile_set_atlas_source->get_tile_texture_region(hovered_tile, frame), color, false);
				}
			} else {
				// Draw empty tile, only in add/remove tiles mode.
				if (tools_button_group->get_pressed_button() == tool_setup_atlas_source_button) {
					Vector2i margins = tile_set_atlas_source->get_margins();
					Vector2i separation = tile_set_atlas_source->get_separation();
					Vector2i tile_size = tile_set_atlas_source->get_texture_region_size();
					Vector2i origin = margins + (hovered_base_tile_coords * (tile_size + separation));
					tile_atlas_control->draw_rect(Rect2i(origin, tile_size), Color(1.0, 1.0, 1.0), false);
				}
			}
		}
	}
}

void TileSetAtlasSourceEditor::_tile_atlas_control_unscaled_draw() {
	if (current_tile_data_editor) {
		// Draw the preview of the selected property.
		for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
			Vector2i coords = tile_set_atlas_source->get_tile_id(i);
			Rect2i texture_region = tile_set_atlas_source->get_tile_texture_region(coords);
			Vector2i position = texture_region.get_center() + tile_set_atlas_source->get_tile_effective_texture_offset(coords, 0);

			Transform2D xform = tile_atlas_control->get_parent_control()->get_transform();
			xform.translate(position);

			if (tools_button_group->get_pressed_button() == tool_select_button && selection.has({ coords, 0 })) {
				continue;
			}

			TileMapCell cell;
			cell.source_id = tile_set_atlas_source_id;
			cell.set_atlas_coords(coords);
			cell.alternative_tile = 0;
			current_tile_data_editor->draw_over_tile(tile_atlas_control_unscaled, xform, cell);
		}

		// Draw the selection on top of other.
		if (tools_button_group->get_pressed_button() == tool_select_button) {
			for (Set<TileSelection>::Element *E = selection.front(); E; E = E->next()) {
				if (E->get().alternative != 0) {
					continue;
				}
				Rect2i texture_region = tile_set_atlas_source->get_tile_texture_region(E->get().tile);
				Vector2i position = texture_region.get_center() + tile_set_atlas_source->get_tile_effective_texture_offset(E->get().tile, 0);

				Transform2D xform = tile_atlas_control->get_parent_control()->get_transform();
				xform.translate(position);

				TileMapCell cell;
				cell.source_id = tile_set_atlas_source_id;
				cell.set_atlas_coords(E->get().tile);
				cell.alternative_tile = 0;
				current_tile_data_editor->draw_over_tile(tile_atlas_control_unscaled, xform, cell, true);
			}
		}

		// Call the TileData's editor custom draw function.
		if (tools_button_group->get_pressed_button() == tool_paint_button) {
			Transform2D xform = tile_atlas_control->get_parent_control()->get_transform();
			current_tile_data_editor->forward_draw_over_atlas(tile_atlas_view, tile_set_atlas_source, tile_atlas_control_unscaled, xform);
		}
	}
}

void TileSetAtlasSourceEditor::_tile_alternatives_control_gui_input(const Ref<InputEvent> &p_event) {
	// Update the hovered alternative tile.
	hovered_alternative_tile_coords = tile_atlas_view->get_alternative_tile_at_pos(alternative_tiles_control->get_local_mouse_position());

	// Forward the event to the current tile data editor if we are in the painting mode.
	if (tools_button_group->get_pressed_button() == tool_paint_button) {
		if (current_tile_data_editor) {
			current_tile_data_editor->forward_painting_alternatives_gui_input(tile_atlas_view, tile_set_atlas_source, p_event);
		}
		tile_atlas_control->update();
		tile_atlas_control_unscaled->update();
		alternative_tiles_control->update();
		alternative_tiles_control_unscaled->update();
		tile_atlas_view->update();
		return;
	}

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		tile_atlas_control->update();
		tile_atlas_control_unscaled->update();
		alternative_tiles_control->update();
		alternative_tiles_control_unscaled->update();
	}

	Ref<InputEventMouseButton> mb = p_event;
	if (mb.is_valid()) {
		drag_type = DRAG_TYPE_NONE;

		Vector2 mouse_local_pos = alternative_tiles_control->get_local_mouse_position();
		if (mb->get_button_index() == MouseButton::LEFT) {
			if (mb->is_pressed()) {
				// Left click pressed.
				if (tools_button_group->get_pressed_button() == tool_select_button) {
					Vector3 tile = tile_atlas_view->get_alternative_tile_at_pos(mouse_local_pos);

					selection.clear();
					TileSelection selected = { Vector2i(tile.x, tile.y), int(tile.z) };
					if (selected.tile != TileSetSource::INVALID_ATLAS_COORDS) {
						selection.insert(selected);
					}

					_update_tile_inspector();
					_update_tile_id_label();
				}
			}
		} else if (mb->get_button_index() == MouseButton::RIGHT) {
			if (mb->is_pressed()) {
				// Right click pressed
				Vector3 tile = tile_atlas_view->get_alternative_tile_at_pos(mouse_local_pos);

				selection.clear();
				TileSelection selected = { Vector2i(tile.x, tile.y), int(tile.z) };
				if (selected.tile != TileSetSource::INVALID_ATLAS_COORDS) {
					selection.insert(selected);
				}

				_update_tile_inspector();
				_update_tile_id_label();

				if (selection.size() == 1) {
					selected = selection.front()->get();
					menu_option_coords = selected.tile;
					menu_option_alternative = selected.alternative;
					alternative_tile_popup_menu->popup(Rect2i(get_global_mouse_position(), Size2i()));
				}
			}
		}
		tile_atlas_control->update();
		tile_atlas_control_unscaled->update();
		alternative_tiles_control->update();
		alternative_tiles_control_unscaled->update();
	}
}

void TileSetAtlasSourceEditor::_tile_alternatives_control_mouse_exited() {
	hovered_alternative_tile_coords = Vector3i(TileSetSource::INVALID_ATLAS_COORDS.x, TileSetSource::INVALID_ATLAS_COORDS.y, TileSetSource::INVALID_TILE_ALTERNATIVE);
	tile_atlas_control->update();
	tile_atlas_control_unscaled->update();
	alternative_tiles_control->update();
	alternative_tiles_control_unscaled->update();
}

void TileSetAtlasSourceEditor::_tile_alternatives_control_draw() {
	Color grid_color = EditorSettings::get_singleton()->get("editors/tiles_editor/grid_color");
	Color selection_color = Color().from_hsv(Math::fposmod(grid_color.get_h() + 0.5, 1.0), grid_color.get_s(), grid_color.get_v(), 1.0);

	// Update the hovered alternative tile.
	if (tools_button_group->get_pressed_button() == tool_select_button) {
		// Draw hovered tile.
		Vector2i coords = Vector2(hovered_alternative_tile_coords.x, hovered_alternative_tile_coords.y);
		if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
			Rect2i rect = tile_atlas_view->get_alternative_tile_rect(coords, hovered_alternative_tile_coords.z);
			if (rect != Rect2i()) {
				alternative_tiles_control->draw_rect(rect, Color(1.0, 1.0, 1.0), false);
			}
		}

		// Draw selected tile.
		for (Set<TileSelection>::Element *E = selection.front(); E; E = E->next()) {
			TileSelection selected = E->get();
			if (selected.alternative >= 1) {
				Rect2i rect = tile_atlas_view->get_alternative_tile_rect(selected.tile, selected.alternative);
				if (rect != Rect2i()) {
					alternative_tiles_control->draw_rect(rect, selection_color, false);
				}
			}
		}
	}
}

void TileSetAtlasSourceEditor::_tile_alternatives_control_unscaled_draw() {
	// Draw the preview of the selected property.
	if (current_tile_data_editor) {
		// Draw the preview of the currently selected property.
		for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
			Vector2i coords = tile_set_atlas_source->get_tile_id(i);
			for (int j = 0; j < tile_set_atlas_source->get_alternative_tiles_count(coords); j++) {
				int alternative_tile = tile_set_atlas_source->get_alternative_tile_id(coords, j);
				if (alternative_tile == 0) {
					continue;
				}
				Rect2i rect = tile_atlas_view->get_alternative_tile_rect(coords, alternative_tile);
				Vector2 position = rect.get_center();

				Transform2D xform = alternative_tiles_control->get_parent_control()->get_transform();
				xform.translate(position);

				if (tools_button_group->get_pressed_button() == tool_select_button && selection.has({ coords, alternative_tile })) {
					continue;
				}

				TileMapCell cell;
				cell.source_id = tile_set_atlas_source_id;
				cell.set_atlas_coords(coords);
				cell.alternative_tile = alternative_tile;
				current_tile_data_editor->draw_over_tile(alternative_tiles_control_unscaled, xform, cell);
			}
		}

		// Draw the selection on top of other.
		if (tools_button_group->get_pressed_button() == tool_select_button) {
			for (Set<TileSelection>::Element *E = selection.front(); E; E = E->next()) {
				if (E->get().alternative == 0) {
					continue;
				}
				Rect2i rect = tile_atlas_view->get_alternative_tile_rect(E->get().tile, E->get().alternative);
				Vector2 position = rect.get_center();

				Transform2D xform = alternative_tiles_control->get_parent_control()->get_transform();
				xform.translate(position);

				TileMapCell cell;
				cell.source_id = tile_set_atlas_source_id;
				cell.set_atlas_coords(E->get().tile);
				cell.alternative_tile = E->get().alternative;
				current_tile_data_editor->draw_over_tile(alternative_tiles_control_unscaled, xform, cell, true);
			}
		}

		// Call the TileData's editor custom draw function.
		if (tools_button_group->get_pressed_button() == tool_paint_button) {
			Transform2D xform = tile_atlas_control->get_parent_control()->get_transform();
			current_tile_data_editor->forward_draw_over_alternatives(tile_atlas_view, tile_set_atlas_source, alternative_tiles_control_unscaled, xform);
		}
	}
}

void TileSetAtlasSourceEditor::_tile_set_changed() {
	tile_set_changed_needs_update = true;
}

void TileSetAtlasSourceEditor::_tile_proxy_object_changed(String p_what) {
	tile_set_changed_needs_update = false; // Avoid updating too many things.
	_update_atlas_view();
}

void TileSetAtlasSourceEditor::_atlas_source_proxy_object_changed(String p_what) {
	if (p_what == "texture" && !atlas_source_proxy_object->get("texture").is_null()) {
		confirm_auto_create_tiles->popup_centered();
	} else if (p_what == "id") {
		emit_signal(SNAME("source_id_changed"), atlas_source_proxy_object->get_id());
	}
}

void TileSetAtlasSourceEditor::_undo_redo_inspector_callback(Object *p_undo_redo, Object *p_edited, String p_property, Variant p_new_value) {
	UndoRedo *undo_redo = Object::cast_to<UndoRedo>(p_undo_redo);
	ERR_FAIL_COND(!undo_redo);

#define ADD_UNDO(obj, property) undo_redo->add_undo_property(obj, property, obj->get(property));

	undo_redo->start_force_keep_in_merge_ends();
	AtlasTileProxyObject *tile_data_proxy = Object::cast_to<AtlasTileProxyObject>(p_edited);
	if (tile_data_proxy) {
		Vector<String> components = String(p_property).split("/", true, 2);
		if (components.size() == 2 && components[1] == "polygons_count") {
			int layer_index = components[0].trim_prefix("physics_layer_").to_int();
			int new_polygons_count = p_new_value;
			int old_polygons_count = tile_data_proxy->get(vformat("physics_layer_%d/polygons_count", layer_index));
			if (new_polygons_count < old_polygons_count) {
				for (int i = new_polygons_count; i < old_polygons_count; i++) {
					ADD_UNDO(tile_data_proxy, vformat("physics_layer_%d/polygon_%d/points", layer_index, i));
					ADD_UNDO(tile_data_proxy, vformat("physics_layer_%d/polygon_%d/one_way", layer_index, i));
					ADD_UNDO(tile_data_proxy, vformat("physics_layer_%d/polygon_%d/one_way_margin", layer_index, i));
				}
			}
		} else if (p_property == "terrain_set") {
			int current_terrain_set = tile_data_proxy->get("terrain_set");
			for (int i = 0; i < TileSet::CELL_NEIGHBOR_MAX; i++) {
				TileSet::CellNeighbor bit = TileSet::CellNeighbor(i);
				if (tile_set->is_valid_peering_bit_terrain(current_terrain_set, bit)) {
					ADD_UNDO(tile_data_proxy, "terrains_peering_bit/" + String(TileSet::CELL_NEIGHBOR_ENUM_TO_TEXT[i]));
				}
			}
		}
	}

	TileSetAtlasSourceProxyObject *atlas_source_proxy = Object::cast_to<TileSetAtlasSourceProxyObject>(p_edited);
	if (atlas_source_proxy) {
		TileSetAtlasSource *atlas_source = atlas_source_proxy->get_edited();
		ERR_FAIL_COND(!atlas_source);

		PackedVector2Array arr;
		if (p_property == "texture") {
			arr = atlas_source->get_tiles_to_be_removed_on_change(p_new_value, atlas_source->get_margins(), atlas_source->get_separation(), atlas_source->get_texture_region_size());
		} else if (p_property == "margins") {
			arr = atlas_source->get_tiles_to_be_removed_on_change(atlas_source->get_texture(), p_new_value, atlas_source->get_separation(), atlas_source->get_texture_region_size());
		} else if (p_property == "separation") {
			arr = atlas_source->get_tiles_to_be_removed_on_change(atlas_source->get_texture(), atlas_source->get_margins(), p_new_value, atlas_source->get_texture_region_size());
		} else if (p_property == "texture_region_size") {
			arr = atlas_source->get_tiles_to_be_removed_on_change(atlas_source->get_texture(), atlas_source->get_margins(), atlas_source->get_separation(), p_new_value);
		}

		if (!arr.is_empty()) {
			// Get all properties assigned to a tile.
			List<PropertyInfo> properties;
			atlas_source->get_property_list(&properties);

			for (int i = 0; i < arr.size(); i++) {
				Vector2i coords = arr[i];
				String prefix = vformat("%d:%d/", coords.x, coords.y);
				for (PropertyInfo pi : properties) {
					if (pi.name.begins_with(prefix)) {
						ADD_UNDO(atlas_source, pi.name);
					}
				}
			}
		}
	}
	undo_redo->end_force_keep_in_merge_ends();

#undef ADD_UNDO
}

void TileSetAtlasSourceEditor::edit(Ref<TileSet> p_tile_set, TileSetAtlasSource *p_tile_set_atlas_source, int p_source_id) {
	ERR_FAIL_COND(!p_tile_set.is_valid());
	ERR_FAIL_COND(!p_tile_set_atlas_source);
	ERR_FAIL_COND(p_source_id < 0);
	ERR_FAIL_COND(p_tile_set->get_source(p_source_id) != p_tile_set_atlas_source);

	if (p_tile_set == tile_set && p_tile_set_atlas_source == tile_set_atlas_source && p_source_id == tile_set_atlas_source_id) {
		return;
	}

	// Remove listener for old objects.
	if (tile_set.is_valid()) {
		tile_set->disconnect("changed", callable_mp(this, &TileSetAtlasSourceEditor::_tile_set_changed));
	}

	// Clear the selection.
	selection.clear();

	// Change the edited object.
	tile_set = p_tile_set;
	tile_set_atlas_source = p_tile_set_atlas_source;
	tile_set_atlas_source_id = p_source_id;

	// Add the listener again.
	if (tile_set.is_valid()) {
		tile_set->connect("changed", callable_mp(this, &TileSetAtlasSourceEditor::_tile_set_changed));
	}

	// Update everything.
	_update_source_inspector();

	// Update the selected tile.
	_update_fix_selected_and_hovered_tiles();
	_update_tile_id_label();
	_update_atlas_view();
	_update_atlas_source_inspector();
	_update_tile_inspector();
	_update_tile_data_editors();
	_update_current_tile_data_editor();
}

void TileSetAtlasSourceEditor::init_source() {
	confirm_auto_create_tiles->popup_centered();
}

void TileSetAtlasSourceEditor::_auto_create_tiles() {
	if (!tile_set_atlas_source) {
		return;
	}

	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();
	if (texture.is_valid()) {
		Vector2i margins = tile_set_atlas_source->get_margins();
		Vector2i separation = tile_set_atlas_source->get_separation();
		Vector2i texture_region_size = tile_set_atlas_source->get_texture_region_size();
		Size2i grid_size = tile_set_atlas_source->get_atlas_grid_size();
		undo_redo->create_action(TTR("Create tiles in non-transparent texture regions"));
		for (int y = 0; y < grid_size.y; y++) {
			for (int x = 0; x < grid_size.x; x++) {
				// Check if we have a tile at the coord
				Vector2i coords = Vector2i(x, y);
				if (tile_set_atlas_source->get_tile_at_coords(coords) == TileSetSource::INVALID_ATLAS_COORDS) {
					// Check if the texture is empty at the given coords.
					Rect2i region = Rect2i(margins + (coords * (texture_region_size + separation)), texture_region_size);
					bool is_opaque = false;
					for (int region_x = region.get_position().x; region_x < region.get_end().x; region_x++) {
						for (int region_y = region.get_position().y; region_y < region.get_end().y; region_y++) {
							if (texture->is_pixel_opaque(region_x, region_y)) {
								is_opaque = true;
								break;
							}
						}
						if (is_opaque) {
							break;
						}
					}

					// If we do have opaque pixels, create a tile.
					if (is_opaque) {
						undo_redo->add_do_method(tile_set_atlas_source, "create_tile", coords);
						undo_redo->add_undo_method(tile_set_atlas_source, "remove_tile", coords);
					}
				}
			}
		}
		undo_redo->commit_action();
	}
}

void TileSetAtlasSourceEditor::_auto_remove_tiles() {
	if (!tile_set_atlas_source) {
		return;
	}

	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();
	if (texture.is_valid()) {
		Vector2i margins = tile_set_atlas_source->get_margins();
		Vector2i separation = tile_set_atlas_source->get_separation();
		Vector2i texture_region_size = tile_set_atlas_source->get_texture_region_size();
		Vector2i grid_size = tile_set_atlas_source->get_atlas_grid_size();

		undo_redo->create_action(TTR("Remove tiles in fully transparent texture regions"));

		List<PropertyInfo> list;
		tile_set_atlas_source->get_property_list(&list);
		Map<Vector2i, List<const PropertyInfo *>> per_tile = _group_properties_per_tiles(list, tile_set_atlas_source);

		for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
			Vector2i coords = tile_set_atlas_source->get_tile_id(i);
			Vector2i size_in_atlas = tile_set_atlas_source->get_tile_size_in_atlas(coords);

			// Skip tiles outside texture.
			if ((coords.x + size_in_atlas.x) > grid_size.x || (coords.y + size_in_atlas.y) > grid_size.y) {
				continue;
			}

			// Check if the texture is empty at the given coords.
			Rect2i region = Rect2i(margins + (coords * (texture_region_size + separation)), texture_region_size * size_in_atlas);
			bool is_opaque = false;
			for (int region_x = region.get_position().x; region_x < region.get_end().x; region_x++) {
				for (int region_y = region.get_position().y; region_y < region.get_end().y; region_y++) {
					if (texture->is_pixel_opaque(region_x, region_y)) {
						is_opaque = true;
						break;
					}
				}
				if (is_opaque) {
					break;
				}
			}

			// If we do have opaque pixels, create a tile.
			if (!is_opaque) {
				undo_redo->add_do_method(tile_set_atlas_source, "remove_tile", coords);
				undo_redo->add_undo_method(tile_set_atlas_source, "create_tile", coords);
				if (per_tile.has(coords)) {
					for (List<const PropertyInfo *>::Element *E_property = per_tile[coords].front(); E_property; E_property = E_property->next()) {
						String property = E_property->get()->name;
						Variant value = tile_set_atlas_source->get(property);
						if (value.get_type() != Variant::NIL) {
							undo_redo->add_undo_method(tile_set_atlas_source, "set", E_property->get()->name, value);
						}
					}
				}
			}
		}
		undo_redo->commit_action();
	}
}

void TileSetAtlasSourceEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED:
			tool_setup_atlas_source_button->set_icon(get_theme_icon(SNAME("Tools"), SNAME("EditorIcons")));
			tool_select_button->set_icon(get_theme_icon(SNAME("ToolSelect"), SNAME("EditorIcons")));
			tool_paint_button->set_icon(get_theme_icon(SNAME("CanvasItem"), SNAME("EditorIcons")));

			tools_settings_erase_button->set_icon(get_theme_icon(SNAME("Eraser"), SNAME("EditorIcons")));

			tool_advanced_menu_buttom->set_icon(get_theme_icon(SNAME("GuiTabMenuHl"), SNAME("EditorIcons")));

			resize_handle = get_theme_icon(SNAME("EditorHandle"), SNAME("EditorIcons"));
			resize_handle_disabled = get_theme_icon(SNAME("EditorHandleDisabled"), SNAME("EditorIcons"));
			break;
		case NOTIFICATION_INTERNAL_PROCESS:
			if (tile_set_changed_needs_update) {
				// Update everything.
				_update_source_inspector();

				// Update the selected tile.
				_update_fix_selected_and_hovered_tiles();
				_update_tile_id_label();
				_update_atlas_view();
				_update_atlas_source_inspector();
				_update_tile_inspector();
				_update_tile_data_editors();
				_update_current_tile_data_editor();

				tile_set_changed_needs_update = false;
			}
			break;
		default:
			break;
	}
}

void TileSetAtlasSourceEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_unhandled_key_input"), &TileSetAtlasSourceEditor::_unhandled_key_input);
	ClassDB::bind_method(D_METHOD("_set_selection_from_array"), &TileSetAtlasSourceEditor::_set_selection_from_array);

	ADD_SIGNAL(MethodInfo("source_id_changed", PropertyInfo(Variant::INT, "source_id")));
}

TileSetAtlasSourceEditor::TileSetAtlasSourceEditor() {
	set_process_unhandled_key_input(true);
	set_process_internal(true);

	// -- Right side --
	HSplitContainer *split_container_right_side = memnew(HSplitContainer);
	split_container_right_side->set_h_size_flags(SIZE_EXPAND_FILL);
	add_child(split_container_right_side);

	// Middle panel.
	ScrollContainer *middle_panel = memnew(ScrollContainer);
	middle_panel->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	middle_panel->set_custom_minimum_size(Size2i(200, 0) * EDSCALE);
	split_container_right_side->add_child(middle_panel);

	VBoxContainer *middle_vbox_container = memnew(VBoxContainer);
	middle_vbox_container->set_h_size_flags(SIZE_EXPAND_FILL);
	middle_panel->add_child(middle_vbox_container);

	// Tile inspector.
	tile_inspector_label = memnew(Label);
	tile_inspector_label->set_text(TTR("Tile Properties:"));
	middle_vbox_container->add_child(tile_inspector_label);

	tile_proxy_object = memnew(AtlasTileProxyObject(this));
	tile_proxy_object->connect("changed", callable_mp(this, &TileSetAtlasSourceEditor::_tile_proxy_object_changed));

	tile_inspector = memnew(EditorInspector);
	tile_inspector->set_undo_redo(undo_redo);
	tile_inspector->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	tile_inspector->edit(tile_proxy_object);
	tile_inspector->set_use_folding(true);
	tile_inspector->connect("property_selected", callable_mp(this, &TileSetAtlasSourceEditor::_inspector_property_selected));
	middle_vbox_container->add_child(tile_inspector);

	tile_inspector_no_tile_selected_label = memnew(Label);
	tile_inspector_no_tile_selected_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	tile_inspector_no_tile_selected_label->set_text(TTR("No tile selected."));
	middle_vbox_container->add_child(tile_inspector_no_tile_selected_label);

	// Property values palette.
	tile_data_editors_popup = memnew(Popup);

	tile_data_editors_label = memnew(Label);
	tile_data_editors_label->set_text(TTR("Paint Properties:"));
	middle_vbox_container->add_child(tile_data_editors_label);

	tile_data_editor_dropdown_button = memnew(Button);
	tile_data_editor_dropdown_button->connect("draw", callable_mp(this, &TileSetAtlasSourceEditor::_tile_data_editor_dropdown_button_draw));
	tile_data_editor_dropdown_button->connect("pressed", callable_mp(this, &TileSetAtlasSourceEditor::_tile_data_editor_dropdown_button_pressed));
	middle_vbox_container->add_child(tile_data_editor_dropdown_button);
	tile_data_editor_dropdown_button->add_child(tile_data_editors_popup);

	tile_data_editors_tree = memnew(Tree);
	tile_data_editors_tree->set_hide_root(true);
	tile_data_editors_tree->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	tile_data_editors_tree->set_h_scroll_enabled(false);
	tile_data_editors_tree->set_v_scroll_enabled(false);
	tile_data_editors_tree->connect("item_selected", callable_mp(this, &TileSetAtlasSourceEditor::_tile_data_editors_tree_selected));
	tile_data_editors_popup->add_child(tile_data_editors_tree);

	tile_data_painting_editor_container = memnew(VBoxContainer);
	tile_data_painting_editor_container->set_h_size_flags(SIZE_EXPAND_FILL);
	middle_vbox_container->add_child(tile_data_painting_editor_container);

	// Atlas source inspector.
	atlas_source_inspector_label = memnew(Label);
	atlas_source_inspector_label->set_text(TTR("Atlas Properties:"));
	middle_vbox_container->add_child(atlas_source_inspector_label);

	atlas_source_proxy_object = memnew(TileSetAtlasSourceProxyObject());
	atlas_source_proxy_object->connect("changed", callable_mp(this, &TileSetAtlasSourceEditor::_atlas_source_proxy_object_changed));

	atlas_source_inspector = memnew(EditorInspector);
	atlas_source_inspector->set_undo_redo(undo_redo);
	atlas_source_inspector->set_vertical_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
	atlas_source_inspector->edit(atlas_source_proxy_object);
	middle_vbox_container->add_child(atlas_source_inspector);

	// Right panel.
	VBoxContainer *right_panel = memnew(VBoxContainer);
	right_panel->set_h_size_flags(SIZE_EXPAND_FILL);
	right_panel->set_v_size_flags(SIZE_EXPAND_FILL);
	split_container_right_side->add_child(right_panel);

	// -- Dialogs --
	confirm_auto_create_tiles = memnew(AcceptDialog);
	confirm_auto_create_tiles->set_title(TTR("Auto Create Tiles in Non-Transparent Texture Regions?"));
	confirm_auto_create_tiles->set_text(TTR("The atlas's texture was modified.\nWould you like to automatically create tiles in the atlas?"));
	confirm_auto_create_tiles->get_ok_button()->set_text(TTR("Yes"));
	confirm_auto_create_tiles->add_cancel_button()->set_text(TTR("No"));
	confirm_auto_create_tiles->connect("confirmed", callable_mp(this, &TileSetAtlasSourceEditor::_auto_create_tiles));
	add_child(confirm_auto_create_tiles);

	// -- Toolbox --
	tools_button_group.instantiate();
	tools_button_group->connect("pressed", callable_mp(this, &TileSetAtlasSourceEditor::_update_fix_selected_and_hovered_tiles).unbind(1));
	tools_button_group->connect("pressed", callable_mp(this, &TileSetAtlasSourceEditor::_update_tile_id_label).unbind(1));
	tools_button_group->connect("pressed", callable_mp(this, &TileSetAtlasSourceEditor::_update_atlas_source_inspector).unbind(1));
	tools_button_group->connect("pressed", callable_mp(this, &TileSetAtlasSourceEditor::_update_tile_inspector).unbind(1));
	tools_button_group->connect("pressed", callable_mp(this, &TileSetAtlasSourceEditor::_update_tile_data_editors).unbind(1));
	tools_button_group->connect("pressed", callable_mp(this, &TileSetAtlasSourceEditor::_update_current_tile_data_editor).unbind(1));
	tools_button_group->connect("pressed", callable_mp(this, &TileSetAtlasSourceEditor::_update_atlas_view).unbind(1));
	tools_button_group->connect("pressed", callable_mp(this, &TileSetAtlasSourceEditor::_update_toolbar).unbind(1));

	toolbox = memnew(HBoxContainer);
	right_panel->add_child(toolbox);

	tool_setup_atlas_source_button = memnew(Button);
	tool_setup_atlas_source_button->set_flat(true);
	tool_setup_atlas_source_button->set_toggle_mode(true);
	tool_setup_atlas_source_button->set_pressed(true);
	tool_setup_atlas_source_button->set_button_group(tools_button_group);
	tool_setup_atlas_source_button->set_tooltip(TTR("Atlas setup. Add/Remove tiles tool (use the shift key to create big tiles, control for rectangle editing)."));
	toolbox->add_child(tool_setup_atlas_source_button);

	tool_select_button = memnew(Button);
	tool_select_button->set_flat(true);
	tool_select_button->set_toggle_mode(true);
	tool_select_button->set_pressed(false);
	tool_select_button->set_button_group(tools_button_group);
	tool_select_button->set_tooltip(TTR("Select tiles."));
	toolbox->add_child(tool_select_button);

	tool_paint_button = memnew(Button);
	tool_paint_button->set_flat(true);
	tool_paint_button->set_toggle_mode(true);
	tool_paint_button->set_button_group(tools_button_group);
	tool_paint_button->set_tooltip(TTR("Paint properties."));
	toolbox->add_child(tool_paint_button);

	// Tool settings.
	tool_settings = memnew(HBoxContainer);
	toolbox->add_child(tool_settings);

	tool_settings_vsep = memnew(VSeparator);
	tool_settings->add_child(tool_settings_vsep);

	tool_settings_tile_data_toolbar_container = memnew(HBoxContainer);
	tool_settings->add_child(tool_settings_tile_data_toolbar_container);

	tools_settings_erase_button = memnew(Button);
	tools_settings_erase_button->set_flat(true);
	tools_settings_erase_button->set_toggle_mode(true);
	tools_settings_erase_button->set_shortcut(ED_SHORTCUT("tiles_editor/eraser", "Eraser", Key::E));
	tools_settings_erase_button->set_shortcut_context(this);
	tool_settings->add_child(tools_settings_erase_button);

	tool_advanced_menu_buttom = memnew(MenuButton);
	tool_advanced_menu_buttom->set_flat(true);
	tool_advanced_menu_buttom->get_popup()->add_item(TTR("Create Tiles in Non-Transparent Texture Regions"), ADVANCED_AUTO_CREATE_TILES);
	tool_advanced_menu_buttom->get_popup()->add_item(TTR("Remove Tiles in Fully Transparent Texture Regions"), ADVANCED_AUTO_REMOVE_TILES);
	tool_advanced_menu_buttom->get_popup()->connect("id_pressed", callable_mp(this, &TileSetAtlasSourceEditor::_menu_option));
	toolbox->add_child(tool_advanced_menu_buttom);

	_update_toolbar();

	// Right side of toolbar.
	Control *middle_space = memnew(Control);
	middle_space->set_h_size_flags(SIZE_EXPAND_FILL);
	toolbox->add_child(middle_space);

	tool_tile_id_label = memnew(Label);
	tool_tile_id_label->set_mouse_filter(Control::MOUSE_FILTER_STOP);
	toolbox->add_child(tool_tile_id_label);
	_update_tile_id_label();

	// Tile atlas view.
	tile_atlas_view = memnew(TileAtlasView);
	tile_atlas_view->set_h_size_flags(SIZE_EXPAND_FILL);
	tile_atlas_view->set_v_size_flags(SIZE_EXPAND_FILL);
	tile_atlas_view->connect("transform_changed", callable_mp(TilesEditorPlugin::get_singleton(), &TilesEditorPlugin::set_atlas_view_transform));
	tile_atlas_view->connect("transform_changed", callable_mp(this, &TileSetAtlasSourceEditor::_tile_atlas_view_transform_changed).unbind(2));
	right_panel->add_child(tile_atlas_view);

	base_tile_popup_menu = memnew(PopupMenu);
	base_tile_popup_menu->add_shortcut(ED_SHORTCUT("tiles_editor/delete", TTR("Delete"), Key::KEY_DELETE), TILE_DELETE);
	base_tile_popup_menu->add_item(TTR("Create an Alternative Tile"), TILE_CREATE_ALTERNATIVE);
	base_tile_popup_menu->connect("id_pressed", callable_mp(this, &TileSetAtlasSourceEditor::_menu_option));
	tile_atlas_view->add_child(base_tile_popup_menu);

	empty_base_tile_popup_menu = memnew(PopupMenu);
	empty_base_tile_popup_menu->add_item(TTR("Create a Tile"), TILE_CREATE);
	empty_base_tile_popup_menu->connect("id_pressed", callable_mp(this, &TileSetAtlasSourceEditor::_menu_option));
	tile_atlas_view->add_child(empty_base_tile_popup_menu);

	tile_atlas_control = memnew(Control);
	tile_atlas_control->connect("draw", callable_mp(this, &TileSetAtlasSourceEditor::_tile_atlas_control_draw));
	tile_atlas_control->connect("mouse_exited", callable_mp(this, &TileSetAtlasSourceEditor::_tile_atlas_control_mouse_exited));
	tile_atlas_control->connect("gui_input", callable_mp(this, &TileSetAtlasSourceEditor::_tile_atlas_control_gui_input));
	tile_atlas_view->add_control_over_atlas_tiles(tile_atlas_control);

	tile_atlas_control_unscaled = memnew(Control);
	tile_atlas_control_unscaled->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	tile_atlas_control_unscaled->connect("draw", callable_mp(this, &TileSetAtlasSourceEditor::_tile_atlas_control_unscaled_draw));
	tile_atlas_view->add_control_over_atlas_tiles(tile_atlas_control_unscaled, false);
	tile_atlas_control_unscaled->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);

	alternative_tile_popup_menu = memnew(PopupMenu);
	alternative_tile_popup_menu->add_shortcut(ED_SHORTCUT("tiles_editor/delete_tile", TTR("Delete"), Key::KEY_DELETE), TILE_DELETE);
	alternative_tile_popup_menu->connect("id_pressed", callable_mp(this, &TileSetAtlasSourceEditor::_menu_option));
	tile_atlas_view->add_child(alternative_tile_popup_menu);

	alternative_tiles_control = memnew(Control);
	alternative_tiles_control->connect("draw", callable_mp(this, &TileSetAtlasSourceEditor::_tile_alternatives_control_draw));
	alternative_tiles_control->connect("mouse_exited", callable_mp(this, &TileSetAtlasSourceEditor::_tile_alternatives_control_mouse_exited));
	alternative_tiles_control->connect("gui_input", callable_mp(this, &TileSetAtlasSourceEditor::_tile_alternatives_control_gui_input));
	tile_atlas_view->add_control_over_alternative_tiles(alternative_tiles_control);

	alternative_tiles_control_unscaled = memnew(Control);
	alternative_tiles_control_unscaled->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	alternative_tiles_control_unscaled->connect("draw", callable_mp(this, &TileSetAtlasSourceEditor::_tile_alternatives_control_unscaled_draw));
	tile_atlas_view->add_control_over_alternative_tiles(alternative_tiles_control_unscaled, false);
	alternative_tiles_control_unscaled->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);

	tile_atlas_view_missing_source_label = memnew(Label);
	tile_atlas_view_missing_source_label->set_text(TTR("Add or select an atlas texture to the left panel."));
	tile_atlas_view_missing_source_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	tile_atlas_view_missing_source_label->set_vertical_alignment(VERTICAL_ALIGNMENT_CENTER);
	tile_atlas_view_missing_source_label->set_h_size_flags(SIZE_EXPAND_FILL);
	tile_atlas_view_missing_source_label->set_v_size_flags(SIZE_EXPAND_FILL);
	tile_atlas_view_missing_source_label->hide();
	right_panel->add_child(tile_atlas_view_missing_source_label);

	EditorNode::get_singleton()->get_editor_data().add_undo_redo_inspector_hook_callback(callable_mp(this, &TileSetAtlasSourceEditor::_undo_redo_inspector_callback));

	// Inspector plugin.
	Ref<EditorInspectorPluginTileData> tile_data_inspector_plugin;
	tile_data_inspector_plugin.instantiate();
	EditorInspector::add_inspector_plugin(tile_data_inspector_plugin);
}

TileSetAtlasSourceEditor::~TileSetAtlasSourceEditor() {
	memdelete(tile_proxy_object);
	memdelete(atlas_source_proxy_object);
}

////// EditorPropertyTilePolygon //////

void EditorPropertyTilePolygon::_add_focusable_children(Node *p_node) {
	Control *control = Object::cast_to<Control>(p_node);
	if (control && control->get_focus_mode() != Control::FOCUS_NONE) {
		add_focusable(control);
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {
		_add_focusable_children(p_node->get_child(i));
	}
}

void EditorPropertyTilePolygon::_polygons_changed() {
	if (String(count_property).is_empty()) {
		if (base_type == "OccluderPolygon2D") {
			// Single OccluderPolygon2D.
			Ref<OccluderPolygon2D> occluder;
			if (generic_tile_polygon_editor->get_polygon_count() >= 1) {
				occluder.instantiate();
				occluder->set_polygon(generic_tile_polygon_editor->get_polygon(0));
			}
			emit_changed(get_edited_property(), occluder);
		} else if (base_type == "NavigationPolygon") {
			Ref<NavigationPolygon> navigation_polygon;
			if (generic_tile_polygon_editor->get_polygon_count() >= 1) {
				navigation_polygon.instantiate();
				for (int i = 0; i < generic_tile_polygon_editor->get_polygon_count(); i++) {
					Vector<Vector2> polygon = generic_tile_polygon_editor->get_polygon(i);
					navigation_polygon->add_outline(polygon);
				}
				navigation_polygon->make_polygons_from_outlines();
			}
			emit_changed(get_edited_property(), navigation_polygon);
		}
	} else {
		if (base_type.is_empty()) {
			// Multiple array of vertices.
			Vector<String> changed_properties;
			Array values;
			int count = generic_tile_polygon_editor->get_polygon_count();
			changed_properties.push_back(count_property);
			values.push_back(count);
			for (int i = 0; i < count; i++) {
				changed_properties.push_back(vformat(element_pattern, i));
				values.push_back(generic_tile_polygon_editor->get_polygon(i));
			}
			emit_signal("multiple_properties_changed", changed_properties, values, false);
		}
	}
}

void EditorPropertyTilePolygon::update_property() {
	TileSetAtlasSourceEditor::AtlasTileProxyObject *atlas_tile_proxy_object = Object::cast_to<TileSetAtlasSourceEditor::AtlasTileProxyObject>(get_edited_object());
	ERR_FAIL_COND(!atlas_tile_proxy_object);
	ERR_FAIL_COND(atlas_tile_proxy_object->get_edited_tiles().is_empty());

	TileSetAtlasSource *tile_set_atlas_source = atlas_tile_proxy_object->get_edited_tile_set_atlas_source();
	generic_tile_polygon_editor->set_tile_set(Ref<TileSet>(tile_set_atlas_source->get_tile_set()));

	// Set the background
	Vector2i coords = atlas_tile_proxy_object->get_edited_tiles().front()->get().tile;
	int alternative = atlas_tile_proxy_object->get_edited_tiles().front()->get().alternative;
	TileData *tile_data = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(coords, alternative));
	generic_tile_polygon_editor->set_background(tile_set_atlas_source->get_texture(), tile_set_atlas_source->get_tile_texture_region(coords), tile_set_atlas_source->get_tile_effective_texture_offset(coords, alternative), tile_data->get_flip_h(), tile_data->get_flip_v(), tile_data->get_transpose(), tile_data->get_modulate());

	// Reset the polygons.
	generic_tile_polygon_editor->clear_polygons();

	if (String(count_property).is_empty()) {
		if (base_type == "OccluderPolygon2D") {
			// Single OccluderPolygon2D.
			Ref<OccluderPolygon2D> occluder = get_edited_object()->get(get_edited_property());
			generic_tile_polygon_editor->clear_polygons();
			if (occluder.is_valid()) {
				generic_tile_polygon_editor->add_polygon(occluder->get_polygon());
			}
		} else if (base_type == "NavigationPolygon") {
			// Single OccluderPolygon2D.
			Ref<NavigationPolygon> navigation_polygon = get_edited_object()->get(get_edited_property());
			generic_tile_polygon_editor->clear_polygons();
			if (navigation_polygon.is_valid()) {
				for (int i = 0; i < navigation_polygon->get_outline_count(); i++) {
					generic_tile_polygon_editor->add_polygon(navigation_polygon->get_outline(i));
				}
			}
		}
	} else {
		int count = get_edited_object()->get(count_property);
		if (base_type.is_empty()) {
			// Multiple array of vertices.
			generic_tile_polygon_editor->clear_polygons();
			for (int i = 0; i < count; i++) {
				generic_tile_polygon_editor->add_polygon(get_edited_object()->get(vformat(element_pattern, i)));
			}
		}
	}
}

void EditorPropertyTilePolygon::setup_single_mode(const StringName &p_property, const String &p_base_type) {
	set_object_and_property(nullptr, p_property);
	base_type = p_base_type;

	generic_tile_polygon_editor->set_multiple_polygon_mode(false);
}

void EditorPropertyTilePolygon::setup_multiple_mode(const StringName &p_property, const StringName &p_count_property, const String &p_element_pattern, const String &p_base_type) {
	set_object_and_property(nullptr, p_property);
	count_property = p_count_property;
	element_pattern = p_element_pattern;
	base_type = p_base_type;

	generic_tile_polygon_editor->set_multiple_polygon_mode(true);
}

EditorPropertyTilePolygon::EditorPropertyTilePolygon() {
	// Setup the polygon editor.
	generic_tile_polygon_editor = memnew(GenericTilePolygonEditor);
	generic_tile_polygon_editor->set_use_undo_redo(false);
	generic_tile_polygon_editor->clear_polygons();
	add_child(generic_tile_polygon_editor);
	generic_tile_polygon_editor->connect("polygons_changed", callable_mp(this, &EditorPropertyTilePolygon::_polygons_changed));

	// Add all focussable children of generic_tile_polygon_editor as focussable.
	_add_focusable_children(generic_tile_polygon_editor);
}

////// EditorInspectorPluginTileData //////

bool EditorInspectorPluginTileData::can_handle(Object *p_object) {
	return Object::cast_to<TileSetAtlasSourceEditor::AtlasTileProxyObject>(p_object) != nullptr;
}

bool EditorInspectorPluginTileData::parse_property(Object *p_object, const Variant::Type p_type, const String &p_path, const PropertyHint p_hint, const String &p_hint_text, const uint32_t p_usage, const bool p_wide) {
	Vector<String> components = String(p_path).split("/", true, 2);
	if (components.size() == 2 && components[0].begins_with("occlusion_layer_") && components[0].trim_prefix("occlusion_layer_").is_valid_int()) {
		// Occlusion layers.
		int layer_index = components[0].trim_prefix("occlusion_layer_").to_int();
		ERR_FAIL_COND_V(layer_index < 0, false);
		if (components[1] == "polygon") {
			EditorPropertyTilePolygon *ep = memnew(EditorPropertyTilePolygon);
			ep->setup_single_mode(p_path, "OccluderPolygon2D");
			add_property_editor(p_path, ep);
			return true;
		}
	} else if (components.size() >= 2 && components[0].begins_with("physics_layer_") && components[0].trim_prefix("physics_layer_").is_valid_int()) {
		// Physics layers.
		int layer_index = components[0].trim_prefix("physics_layer_").to_int();
		ERR_FAIL_COND_V(layer_index < 0, false);
		if (components[1] == "polygons_count") {
			EditorPropertyTilePolygon *ep = memnew(EditorPropertyTilePolygon);
			ep->setup_multiple_mode(vformat("physics_layer_%d/polygons", layer_index), vformat("physics_layer_%d/polygons_count", layer_index), vformat("physics_layer_%d/polygon_%%d/points", layer_index), "");
			Vector<String> properties;
			properties.push_back(p_path);
			int count = p_object->get(vformat("physics_layer_%d/polygons_count", layer_index));
			for (int i = 0; i < count; i++) {
				properties.push_back(vformat(vformat("physics_layer_%d/polygon_%d/points", layer_index, i)));
			}
			add_property_editor_for_multiple_properties("Polygons", properties, ep);
			return true;
		} else if (components.size() == 3 && components[1].begins_with("polygon_") && components[1].trim_prefix("polygon_").is_valid_int()) {
			int polygon_index = components[1].trim_prefix("polygon_").to_int();
			ERR_FAIL_COND_V(polygon_index < 0, false);
			if (components[2] == "points") {
				return true;
			}
		}
	} else if (components.size() == 2 && components[0].begins_with("navigation_layer_") && components[0].trim_prefix("navigation_layer_").is_valid_int()) {
		// Navigation layers.
		int layer_index = components[0].trim_prefix("navigation_layer_").to_int();
		ERR_FAIL_COND_V(layer_index < 0, false);
		if (components[1] == "polygon") {
			EditorPropertyTilePolygon *ep = memnew(EditorPropertyTilePolygon);
			ep->setup_single_mode(p_path, "NavigationPolygon");
			add_property_editor(p_path, ep);
			return true;
		}
	}
	return false;
}
