/*************************************************************************/
/*  tile_atlas_view.cpp                                                  */
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

#include "tile_atlas_view.h"

#include "core/input/input.h"
#include "core/os/keyboard.h"
#include "scene/2d/tile_map.h"
#include "scene/gui/box_container.h"
#include "scene/gui/label.h"
#include "scene/gui/panel.h"
#include "scene/gui/texture_rect.h"
#include "scene/gui/view_panner.h"

#include "editor/editor_scale.h"
#include "editor/editor_settings.h"

void TileAtlasView::gui_input(const Ref<InputEvent> &p_event) {
	if (panner->gui_input(p_event)) {
		accept_event();
	}
}

void TileAtlasView::_scroll_callback(Vector2 p_scroll_vec, bool p_alt) {
	_pan_callback(-p_scroll_vec * 32);
}

void TileAtlasView::_pan_callback(Vector2 p_scroll_vec) {
	panning += p_scroll_vec;
	emit_signal(SNAME("transform_changed"), zoom_widget->get_zoom(), panning);
	_update_zoom_and_panning(true);
}

void TileAtlasView::_zoom_callback(Vector2 p_scroll_vec, Vector2 p_origin, bool p_alt) {
	zoom_widget->set_zoom_by_increments(-p_scroll_vec.y * 2);
	emit_signal(SNAME("transform_changed"), zoom_widget->get_zoom(), panning);
	_update_zoom_and_panning(true);
}

Size2i TileAtlasView::_compute_base_tiles_control_size() {
	// Update the texture.
	Vector2i size;
	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();
	if (texture.is_valid()) {
		size = texture->get_size();
	}
	return size;
}

Size2i TileAtlasView::_compute_alternative_tiles_control_size() {
	Vector2i size;
	for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
		Vector2i tile_id = tile_set_atlas_source->get_tile_id(i);
		int alternatives_count = tile_set_atlas_source->get_alternative_tiles_count(tile_id);
		Vector2i line_size;
		Size2i texture_region_size = tile_set_atlas_source->get_tile_texture_region(tile_id).size;
		for (int j = 1; j < alternatives_count; j++) {
			int alternative_id = tile_set_atlas_source->get_alternative_tile_id(tile_id, j);
			bool transposed = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(tile_id, alternative_id))->get_transpose();
			line_size.x += transposed ? texture_region_size.y : texture_region_size.x;
			line_size.y = MAX(line_size.y, transposed ? texture_region_size.x : texture_region_size.y);
		}
		size.x = MAX(size.x, line_size.x);
		size.y += line_size.y;
	}

	return size;
}

void TileAtlasView::_update_zoom_and_panning(bool p_zoom_on_mouse_pos) {
	float zoom = zoom_widget->get_zoom();

	// Compute the minimum sizes.
	Size2i base_tiles_control_size = _compute_base_tiles_control_size();
	base_tiles_root_control->set_custom_minimum_size(Vector2(base_tiles_control_size) * zoom);

	Size2i alternative_tiles_control_size = _compute_alternative_tiles_control_size();
	alternative_tiles_root_control->set_custom_minimum_size(Vector2(alternative_tiles_control_size) * zoom);

	// Set the texture for the base tiles.
	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();

	// Set the scales.
	if (base_tiles_control_size.x > 0 && base_tiles_control_size.y > 0) {
		base_tiles_drawing_root->set_scale(Vector2(zoom, zoom));
	} else {
		base_tiles_drawing_root->set_scale(Vector2(1, 1));
	}
	if (alternative_tiles_control_size.x > 0 && alternative_tiles_control_size.y > 0) {
		alternative_tiles_drawing_root->set_scale(Vector2(zoom, zoom));
	} else {
		alternative_tiles_drawing_root->set_scale(Vector2(1, 1));
	}

	// Update the margin container's margins.
	const char *constants[] = { "margin_left", "margin_top", "margin_right", "margin_bottom" };
	for (int i = 0; i < 4; i++) {
		margin_container->add_theme_constant_override(constants[i], margin_container_paddings[i] * zoom);
	}

	// Update the backgrounds.
	background_left->update();
	background_right->update();

	// Zoom on the position.
	if (p_zoom_on_mouse_pos) {
		// Offset the panning relative to the center of panel.
		Vector2 relative_mpos = get_local_mouse_position() - get_size() / 2;
		panning = (panning - relative_mpos) * zoom / previous_zoom + relative_mpos;
	} else {
		// Center of panel.
		panning = panning * zoom / previous_zoom;
	}
	button_center_view->set_disabled(panning.is_equal_approx(Vector2()));

	previous_zoom = zoom;

	center_container->set_begin(panning - center_container->get_minimum_size() / 2);
	center_container->set_size(center_container->get_minimum_size());
}

void TileAtlasView::_zoom_widget_changed() {
	_update_zoom_and_panning();
	emit_signal(SNAME("transform_changed"), zoom_widget->get_zoom(), panning);
}

void TileAtlasView::_center_view() {
	panning = Vector2();
	button_center_view->set_disabled(true);
	_update_zoom_and_panning();
	emit_signal(SNAME("transform_changed"), zoom_widget->get_zoom(), panning);
}

void TileAtlasView::_base_tiles_root_control_gui_input(const Ref<InputEvent> &p_event) {
	base_tiles_root_control->set_tooltip("");

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		Transform2D xform = base_tiles_drawing_root->get_transform().affine_inverse();
		Vector2i coords = get_atlas_tile_coords_at_pos(xform.xform(mm->get_position()));
		if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
			coords = tile_set_atlas_source->get_tile_at_coords(coords);
			if (coords != TileSetSource::INVALID_ATLAS_COORDS) {
				base_tiles_root_control->set_tooltip(vformat(TTR("Source: %d\nAtlas coordinates: %s\nAlternative: 0"), source_id, coords));
			}
		}
	}
}

void TileAtlasView::_draw_base_tiles() {
	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();
	if (texture.is_valid()) {
		Vector2i margins = tile_set_atlas_source->get_margins();
		Vector2i separation = tile_set_atlas_source->get_separation();
		Vector2i texture_region_size = tile_set_atlas_source->get_texture_region_size();
		Size2i grid_size = tile_set_atlas_source->get_atlas_grid_size();

		// Draw the texture where there is no tile.
		for (int x = 0; x < grid_size.x; x++) {
			for (int y = 0; y < grid_size.y; y++) {
				Vector2i coords = Vector2i(x, y);
				if (tile_set_atlas_source->get_tile_at_coords(coords) == TileSetSource::INVALID_ATLAS_COORDS) {
					Rect2i rect = Rect2i((texture_region_size + separation) * coords + margins, texture_region_size + separation);
					rect = rect.intersection(Rect2i(Vector2(), texture->get_size()));
					if (rect.size.x > 0 && rect.size.y > 0) {
						base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
						base_tiles_draw->draw_rect(rect, Color(0.0, 0.0, 0.0, 0.5));
					}
				}
			}
		}

		// Draw the texture around the grid.
		Rect2i rect;

		// Top.
		rect.position = Vector2i();
		rect.set_end(Vector2i(texture->get_size().x, margins.y));
		base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
		base_tiles_draw->draw_rect(rect, Color(0.0, 0.0, 0.0, 0.5));

		// Bottom
		int bottom_border = margins.y + (grid_size.y * (texture_region_size.y + separation.y));
		if (bottom_border < texture->get_size().y) {
			rect.position = Vector2i(0, bottom_border);
			rect.set_end(texture->get_size());
			base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
			base_tiles_draw->draw_rect(rect, Color(0.0, 0.0, 0.0, 0.5));
		}

		// Left
		rect.position = Vector2i(0, margins.y);
		rect.set_end(Vector2i(margins.x, margins.y + (grid_size.y * (texture_region_size.y + separation.y))));
		base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
		base_tiles_draw->draw_rect(rect, Color(0.0, 0.0, 0.0, 0.5));

		// Right.
		int right_border = margins.x + (grid_size.x * (texture_region_size.x + separation.x));
		if (right_border < texture->get_size().x) {
			rect.position = Vector2i(right_border, margins.y);
			rect.set_end(Vector2i(texture->get_size().x, margins.y + (grid_size.y * (texture_region_size.y + separation.y))));
			base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
			base_tiles_draw->draw_rect(rect, Color(0.0, 0.0, 0.0, 0.5));
		}

		// Draw actual tiles, using their properties (modulation, etc...)
		for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
			Vector2i atlas_coords = tile_set_atlas_source->get_tile_id(i);

			for (int frame = 0; frame < tile_set_atlas_source->get_tile_animation_frames_count(atlas_coords); frame++) {
				// Update the y to max value.
				Rect2i base_frame_rect = tile_set_atlas_source->get_tile_texture_region(atlas_coords, frame);
				Vector2i offset_pos = base_frame_rect.get_center() + tile_set_atlas_source->get_tile_effective_texture_offset(atlas_coords, 0);

				// Draw the tile.
				TileMap::draw_tile(base_tiles_draw->get_canvas_item(), offset_pos, tile_set, source_id, atlas_coords, 0, frame);

				// Draw, the texture in the separation areas
				if (separation.x > 0) {
					Rect2i right_sep_rect = Rect2i(base_frame_rect.get_position() + Vector2i(base_frame_rect.size.x, 0), Vector2i(separation.x, base_frame_rect.size.y));
					right_sep_rect = right_sep_rect.intersection(Rect2i(Vector2(), texture->get_size()));
					if (right_sep_rect.size.x > 0 && right_sep_rect.size.y > 0) {
						base_tiles_draw->draw_texture_rect_region(texture, right_sep_rect, right_sep_rect);
						base_tiles_draw->draw_rect(right_sep_rect, Color(0.0, 0.0, 0.0, 0.5));
					}
				}

				if (separation.y > 0) {
					Rect2i bottom_sep_rect = Rect2i(base_frame_rect.get_position() + Vector2i(0, base_frame_rect.size.y), Vector2i(base_frame_rect.size.x + separation.x, separation.y));
					bottom_sep_rect = bottom_sep_rect.intersection(Rect2i(Vector2(), texture->get_size()));
					if (bottom_sep_rect.size.x > 0 && bottom_sep_rect.size.y > 0) {
						base_tiles_draw->draw_texture_rect_region(texture, bottom_sep_rect, bottom_sep_rect);
						base_tiles_draw->draw_rect(bottom_sep_rect, Color(0.0, 0.0, 0.0, 0.5));
					}
				}
			}
		}
	}
}

void TileAtlasView::_draw_base_tiles_texture_grid() {
	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();
	if (texture.is_valid()) {
		Vector2i margins = tile_set_atlas_source->get_margins();
		Vector2i separation = tile_set_atlas_source->get_separation();
		Vector2i texture_region_size = tile_set_atlas_source->get_texture_region_size();

		Size2i grid_size = tile_set_atlas_source->get_atlas_grid_size();

		// Draw each tile texture region.
		for (int x = 0; x < grid_size.x; x++) {
			for (int y = 0; y < grid_size.y; y++) {
				Vector2i origin = margins + (Vector2i(x, y) * (texture_region_size + separation));
				Vector2i base_tile_coords = tile_set_atlas_source->get_tile_at_coords(Vector2i(x, y));
				if (base_tile_coords != TileSetSource::INVALID_ATLAS_COORDS) {
					if (base_tile_coords == Vector2i(x, y)) {
						// Draw existing tile.
						Vector2i size_in_atlas = tile_set_atlas_source->get_tile_size_in_atlas(base_tile_coords);
						Vector2 region_size = texture_region_size * size_in_atlas + separation * (size_in_atlas - Vector2i(1, 1));
						base_tiles_texture_grid->draw_rect(Rect2i(origin, region_size), Color(1.0, 1.0, 1.0, 0.8), false);
					}
				} else {
					// Draw the grid.
					base_tiles_texture_grid->draw_rect(Rect2i(origin, texture_region_size), Color(0.7, 0.7, 0.7, 0.1), false);
				}
			}
		}
	}
}

void TileAtlasView::_draw_base_tiles_shape_grid() {
	// Draw the shapes.
	Color grid_color = EditorSettings::get_singleton()->get("editors/tiles_editor/grid_color");
	Vector2i tile_shape_size = tile_set->get_tile_size();
	for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
		Vector2i tile_id = tile_set_atlas_source->get_tile_id(i);
		Vector2 in_tile_base_offset = tile_set_atlas_source->get_tile_effective_texture_offset(tile_id, 0);

		for (int frame = 0; frame < tile_set_atlas_source->get_tile_animation_frames_count(tile_id); frame++) {
			Color color = grid_color;
			if (frame > 0) {
				color.a *= 0.3;
			}
			Rect2i texture_region = tile_set_atlas_source->get_tile_texture_region(tile_id);
			Transform2D tile_xform;
			tile_xform.set_origin(texture_region.get_center() + in_tile_base_offset);
			tile_xform.set_scale(tile_shape_size);
			tile_set->draw_tile_shape(base_tiles_shape_grid, tile_xform, color);
		}
	}
}

void TileAtlasView::_alternative_tiles_root_control_gui_input(const Ref<InputEvent> &p_event) {
	alternative_tiles_root_control->set_tooltip("");

	Ref<InputEventMouseMotion> mm = p_event;
	if (mm.is_valid()) {
		Transform2D xform = alternative_tiles_drawing_root->get_transform().affine_inverse();
		Vector3i coords3 = get_alternative_tile_at_pos(xform.xform(mm->get_position()));
		Vector2i coords = Vector2i(coords3.x, coords3.y);
		int alternative_id = coords3.z;
		if (coords != TileSetSource::INVALID_ATLAS_COORDS && alternative_id != TileSetSource::INVALID_TILE_ALTERNATIVE) {
			alternative_tiles_root_control->set_tooltip(vformat(TTR("Source: %d\nAtlas coordinates: %s\nAlternative: %d"), source_id, coords, alternative_id));
		}
	}
}

void TileAtlasView::_draw_alternatives() {
	// Draw the alternative tiles.
	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();
	if (texture.is_valid()) {
		Vector2 current_pos;
		for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
			Vector2i atlas_coords = tile_set_atlas_source->get_tile_id(i);
			current_pos.x = 0;
			int y_increment = 0;
			Size2i texture_region_size = tile_set_atlas_source->get_tile_texture_region(atlas_coords).size;
			int alternatives_count = tile_set_atlas_source->get_alternative_tiles_count(atlas_coords);
			for (int j = 1; j < alternatives_count; j++) {
				int alternative_id = tile_set_atlas_source->get_alternative_tile_id(atlas_coords, j);
				TileData *tile_data = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(atlas_coords, alternative_id));
				bool transposed = tile_data->get_transpose();

				// Update the y to max value.
				Vector2i offset_pos = current_pos;
				if (transposed) {
					offset_pos = (current_pos + Vector2(texture_region_size.y, texture_region_size.x) / 2 + tile_set_atlas_source->get_tile_effective_texture_offset(atlas_coords, alternative_id));
					y_increment = MAX(y_increment, texture_region_size.x);
				} else {
					offset_pos = (current_pos + texture_region_size / 2 + tile_set_atlas_source->get_tile_effective_texture_offset(atlas_coords, alternative_id));
					y_increment = MAX(y_increment, texture_region_size.y);
				}

				// Draw the tile.
				TileMap::draw_tile(alternatives_draw->get_canvas_item(), offset_pos, tile_set, source_id, atlas_coords, alternative_id);

				// Increment the x position.
				current_pos.x += transposed ? texture_region_size.y : texture_region_size.x;
			}
			if (alternatives_count > 1) {
				current_pos.y += y_increment;
			}
		}
	}
}

void TileAtlasView::_draw_background_left() {
	Ref<Texture2D> texture = get_theme_icon(SNAME("Checkerboard"), SNAME("EditorIcons"));
	background_left->set_size(base_tiles_root_control->get_custom_minimum_size());
	background_left->draw_texture_rect(texture, Rect2(Vector2(), background_left->get_size()), true);
}

void TileAtlasView::_draw_background_right() {
	Ref<Texture2D> texture = get_theme_icon(SNAME("Checkerboard"), SNAME("EditorIcons"));
	background_right->set_size(alternative_tiles_root_control->get_custom_minimum_size());
	background_right->draw_texture_rect(texture, Rect2(Vector2(), background_right->get_size()), true);
}

void TileAtlasView::set_atlas_source(TileSet *p_tile_set, TileSetAtlasSource *p_tile_set_atlas_source, int p_source_id) {
	ERR_FAIL_COND(!p_tile_set);
	ERR_FAIL_COND(!p_tile_set_atlas_source);
	ERR_FAIL_COND(p_source_id < 0);
	ERR_FAIL_COND(p_tile_set->get_source(p_source_id) != p_tile_set_atlas_source);

	tile_set = p_tile_set;
	tile_set_atlas_source = p_tile_set_atlas_source;
	source_id = p_source_id;

	// Show or hide the view.
	bool valid = tile_set_atlas_source->get_texture().is_valid();
	hbox->set_visible(valid);
	missing_source_label->set_visible(!valid);

	// Update the rect cache.
	_update_alternative_tiles_rect_cache();

	// Update everything.
	_update_zoom_and_panning();

	// Change children control size.
	Size2i base_tiles_control_size = _compute_base_tiles_control_size();
	for (int i = 0; i < base_tiles_drawing_root->get_child_count(); i++) {
		Control *control = Object::cast_to<Control>(base_tiles_drawing_root->get_child(i));
		if (control) {
			control->set_size(base_tiles_control_size);
		}
	}

	Size2i alternative_control_size = _compute_alternative_tiles_control_size();
	for (int i = 0; i < alternative_tiles_drawing_root->get_child_count(); i++) {
		Control *control = Object::cast_to<Control>(alternative_tiles_drawing_root->get_child(i));
		if (control) {
			control->set_size(alternative_control_size);
		}
	}

	// Update.
	base_tiles_draw->update();
	base_tiles_texture_grid->update();
	base_tiles_shape_grid->update();
	alternatives_draw->update();
	background_left->update();
	background_right->update();
}

float TileAtlasView::get_zoom() const {
	return zoom_widget->get_zoom();
};

void TileAtlasView::set_transform(float p_zoom, Vector2i p_panning) {
	zoom_widget->set_zoom(p_zoom);
	panning = p_panning;
	_update_zoom_and_panning();
};

void TileAtlasView::set_padding(Side p_side, int p_padding) {
	ERR_FAIL_COND(p_padding < 0);
	margin_container_paddings[p_side] = p_padding;
}

Vector2i TileAtlasView::get_atlas_tile_coords_at_pos(const Vector2 p_pos) const {
	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();
	if (texture.is_valid()) {
		Vector2i margins = tile_set_atlas_source->get_margins();
		Vector2i separation = tile_set_atlas_source->get_separation();
		Vector2i texture_region_size = tile_set_atlas_source->get_texture_region_size();

		// Compute index in atlas
		Vector2 pos = p_pos - margins;
		Vector2i ret = (pos / (texture_region_size + separation)).floor();

		return ret;
	}

	return TileSetSource::INVALID_ATLAS_COORDS;
}

void TileAtlasView::_update_alternative_tiles_rect_cache() {
	alternative_tiles_rect_cache.clear();

	Rect2i current;
	for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
		Vector2i tile_id = tile_set_atlas_source->get_tile_id(i);
		int alternatives_count = tile_set_atlas_source->get_alternative_tiles_count(tile_id);
		Size2i texture_region_size = tile_set_atlas_source->get_tile_texture_region(tile_id).size;
		int line_height = 0;
		for (int j = 1; j < alternatives_count; j++) {
			int alternative_id = tile_set_atlas_source->get_alternative_tile_id(tile_id, j);
			TileData *tile_data = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(tile_id, alternative_id));
			bool transposed = tile_data->get_transpose();
			current.size = transposed ? Vector2i(texture_region_size.y, texture_region_size.x) : texture_region_size;

			// Update the rect.
			if (!alternative_tiles_rect_cache.has(tile_id)) {
				alternative_tiles_rect_cache[tile_id] = Map<int, Rect2i>();
			}
			alternative_tiles_rect_cache[tile_id][alternative_id] = current;

			current.position.x += transposed ? texture_region_size.y : texture_region_size.x;
			line_height = MAX(line_height, transposed ? texture_region_size.x : texture_region_size.y);
		}

		current.position.x = 0;
		current.position.y += line_height;
	}
}

Vector3i TileAtlasView::get_alternative_tile_at_pos(const Vector2 p_pos) const {
	for (const KeyValue<Vector2, Map<int, Rect2i>> &E_coords : alternative_tiles_rect_cache) {
		for (const KeyValue<int, Rect2i> &E_alternative : E_coords.value) {
			if (E_alternative.value.has_point(p_pos)) {
				return Vector3i(E_coords.key.x, E_coords.key.y, E_alternative.key);
			}
		}
	}

	return Vector3i(TileSetSource::INVALID_ATLAS_COORDS.x, TileSetSource::INVALID_ATLAS_COORDS.y, TileSetSource::INVALID_TILE_ALTERNATIVE);
}

Rect2i TileAtlasView::get_alternative_tile_rect(const Vector2i p_coords, int p_alternative_tile) {
	ERR_FAIL_COND_V_MSG(!alternative_tiles_rect_cache.has(p_coords), Rect2i(), vformat("No cached rect for tile coords:%s", p_coords));
	ERR_FAIL_COND_V_MSG(!alternative_tiles_rect_cache[p_coords].has(p_alternative_tile), Rect2i(), vformat("No cached rect for tile coords:%s alternative_id:%d", p_coords, p_alternative_tile));

	return alternative_tiles_rect_cache[p_coords][p_alternative_tile];
}

void TileAtlasView::update() {
	base_tiles_draw->update();
	base_tiles_texture_grid->update();
	base_tiles_shape_grid->update();
	alternatives_draw->update();
	background_left->update();
	background_right->update();
}

void TileAtlasView::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED:
			panner->setup((ViewPanner::ControlScheme)EDITOR_GET("editors/panning/sub_editors_panning_scheme").operator int(), ED_GET_SHORTCUT("canvas_item_editor/pan_view"), bool(EditorSettings::get_singleton()->get("editors/panning/simple_panning")));
			break;

		case NOTIFICATION_READY:
			button_center_view->set_icon(get_theme_icon(SNAME("CenterView"), SNAME("EditorIcons")));
			break;
	}
}

void TileAtlasView::_bind_methods() {
	ADD_SIGNAL(MethodInfo("transform_changed", PropertyInfo(Variant::FLOAT, "zoom"), PropertyInfo(Variant::VECTOR2, "scroll")));
}

TileAtlasView::TileAtlasView() {
	set_texture_filter(CanvasItem::TEXTURE_FILTER_NEAREST);

	Panel *panel = memnew(Panel);
	panel->set_clip_contents(true);
	panel->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	panel->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	panel->set_h_size_flags(SIZE_EXPAND_FILL);
	panel->set_v_size_flags(SIZE_EXPAND_FILL);
	add_child(panel);

	// Scrollingsc
	zoom_widget = memnew(EditorZoomWidget);
	add_child(zoom_widget);
	zoom_widget->set_anchors_and_offsets_preset(Control::PRESET_TOP_LEFT, Control::PRESET_MODE_MINSIZE, 2 * EDSCALE);
	zoom_widget->connect("zoom_changed", callable_mp(this, &TileAtlasView::_zoom_widget_changed).unbind(1));

	button_center_view = memnew(Button);
	button_center_view->set_icon(get_theme_icon(SNAME("CenterView"), SNAME("EditorIcons")));
	button_center_view->set_anchors_and_offsets_preset(Control::PRESET_TOP_RIGHT, Control::PRESET_MODE_MINSIZE, 5);
	button_center_view->connect("pressed", callable_mp(this, &TileAtlasView::_center_view));
	button_center_view->set_flat(true);
	button_center_view->set_disabled(true);
	button_center_view->set_tooltip(TTR("Center View"));
	add_child(button_center_view);

	panner.instantiate();
	panner->set_callbacks(callable_mp(this, &TileAtlasView::_scroll_callback), callable_mp(this, &TileAtlasView::_pan_callback), callable_mp(this, &TileAtlasView::_zoom_callback));
	panner->set_enable_rmb(true);

	center_container = memnew(CenterContainer);
	center_container->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	center_container->set_anchors_preset(Control::PRESET_CENTER);
	center_container->connect("gui_input", callable_mp(this, &TileAtlasView::gui_input));
	center_container->connect("focus_exited", callable_mp(panner.ptr(), &ViewPanner::release_pan_key));
	center_container->set_focus_mode(FOCUS_CLICK);
	panel->add_child(center_container);

	missing_source_label = memnew(Label);
	missing_source_label->set_text(TTR("No atlas source with a valid texture selected."));
	center_container->add_child(missing_source_label);

	margin_container = memnew(MarginContainer);
	margin_container->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	center_container->add_child(margin_container);

	hbox = memnew(HBoxContainer);
	hbox->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	hbox->add_theme_constant_override("separation", 10);
	hbox->hide();
	margin_container->add_child(hbox);

	VBoxContainer *left_vbox = memnew(VBoxContainer);
	left_vbox->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	hbox->add_child(left_vbox);

	VBoxContainer *right_vbox = memnew(VBoxContainer);
	right_vbox->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	hbox->add_child(right_vbox);

	// Base tiles.
	Label *base_tile_label = memnew(Label);
	base_tile_label->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	base_tile_label->set_text(TTR("Base Tiles"));
	base_tile_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	left_vbox->add_child(base_tile_label);

	base_tiles_root_control = memnew(Control);
	base_tiles_root_control->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	base_tiles_root_control->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	base_tiles_root_control->connect("gui_input", callable_mp(this, &TileAtlasView::_base_tiles_root_control_gui_input));
	left_vbox->add_child(base_tiles_root_control);

	background_left = memnew(Control);
	background_left->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	background_left->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	background_left->set_texture_repeat(TextureRepeat::TEXTURE_REPEAT_ENABLED);
	background_left->connect("draw", callable_mp(this, &TileAtlasView::_draw_background_left));
	base_tiles_root_control->add_child(background_left);

	base_tiles_drawing_root = memnew(Control);
	base_tiles_drawing_root->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	base_tiles_drawing_root->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	base_tiles_drawing_root->set_texture_filter(TEXTURE_FILTER_NEAREST);
	base_tiles_root_control->add_child(base_tiles_drawing_root);

	base_tiles_draw = memnew(Control);
	base_tiles_draw->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	base_tiles_draw->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	base_tiles_draw->connect("draw", callable_mp(this, &TileAtlasView::_draw_base_tiles));
	base_tiles_drawing_root->add_child(base_tiles_draw);

	base_tiles_texture_grid = memnew(Control);
	base_tiles_texture_grid->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	base_tiles_texture_grid->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	base_tiles_texture_grid->connect("draw", callable_mp(this, &TileAtlasView::_draw_base_tiles_texture_grid));
	base_tiles_drawing_root->add_child(base_tiles_texture_grid);

	base_tiles_shape_grid = memnew(Control);
	base_tiles_shape_grid->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	base_tiles_shape_grid->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	base_tiles_shape_grid->connect("draw", callable_mp(this, &TileAtlasView::_draw_base_tiles_shape_grid));
	base_tiles_drawing_root->add_child(base_tiles_shape_grid);

	// Alternative tiles.
	Label *alternative_tiles_label = memnew(Label);
	alternative_tiles_label->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	alternative_tiles_label->set_text(TTR("Alternative Tiles"));
	alternative_tiles_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
	right_vbox->add_child(alternative_tiles_label);

	alternative_tiles_root_control = memnew(Control);
	alternative_tiles_root_control->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	alternative_tiles_root_control->connect("gui_input", callable_mp(this, &TileAtlasView::_alternative_tiles_root_control_gui_input));
	right_vbox->add_child(alternative_tiles_root_control);

	background_right = memnew(Control);
	background_right->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	background_right->set_texture_repeat(TextureRepeat::TEXTURE_REPEAT_ENABLED);
	background_right->connect("draw", callable_mp(this, &TileAtlasView::_draw_background_right));

	alternative_tiles_root_control->add_child(background_right);

	alternative_tiles_drawing_root = memnew(Control);
	alternative_tiles_drawing_root->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	alternative_tiles_drawing_root->set_texture_filter(TEXTURE_FILTER_NEAREST);
	alternative_tiles_root_control->add_child(alternative_tiles_drawing_root);

	alternatives_draw = memnew(Control);
	alternatives_draw->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	alternatives_draw->connect("draw", callable_mp(this, &TileAtlasView::_draw_alternatives));
	alternative_tiles_drawing_root->add_child(alternatives_draw);
}
