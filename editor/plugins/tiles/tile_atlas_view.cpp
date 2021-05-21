/*************************************************************************/
/*  tile_atlas_view.cpp                                                  */
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

#include "tile_atlas_view.h"

#include "core/input/input.h"
#include "core/os/keyboard.h"
#include "scene/gui/box_container.h"
#include "scene/gui/center_container.h"
#include "scene/gui/label.h"
#include "scene/gui/panel.h"
#include "scene/gui/texture_rect.h"

#include "editor/editor_scale.h"
#include "editor/editor_settings.h"

void TileAtlasView::_gui_input(const Ref<InputEvent> &p_event) {
	bool ctrl = Input::get_singleton()->is_key_pressed(KEY_CTRL);

	Ref<InputEventMouseButton> b = p_event;
	if (b.is_valid()) {
		if (ctrl && b->is_pressed() && b->get_button_index() == MOUSE_BUTTON_WHEEL_DOWN) {
			// Zoom out
			zoom_widget->set_zoom_by_increments(-2);
			emit_signal("transform_changed", zoom_widget->get_zoom(), Vector2(scroll_container->get_h_scroll(), scroll_container->get_v_scroll()));
			_update_zoom(zoom_widget->get_zoom(), true);
			accept_event();
		}

		if (ctrl && b->is_pressed() && b->get_button_index() == MOUSE_BUTTON_WHEEL_UP) {
			// Zoom in
			zoom_widget->set_zoom_by_increments(2);
			emit_signal("transform_changed", zoom_widget->get_zoom(), Vector2(scroll_container->get_h_scroll(), scroll_container->get_v_scroll()));
			_update_zoom(zoom_widget->get_zoom(), true);
			accept_event();
		}
	}
}

Size2i TileAtlasView::_compute_base_tiles_control_size() {
	// Update the texture.
	Vector2i size;
	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();
	if (texture.is_valid()) {
		size = texture->get_size();
	}

	// Extend the size to all existing tiles.
	Size2i grid_size = tile_set_atlas_source->get_atlas_grid_size();
	for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
		Vector2i tile_id = tile_set_atlas_source->get_tile_id(i);
		grid_size = grid_size.max(tile_id + Vector2i(1, 1));
	}
	size = size.max(grid_size * (tile_set_atlas_source->get_texture_region_size() + tile_set_atlas_source->get_separation()) + tile_set_atlas_source->get_margins());

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

void TileAtlasView::_update_zoom(float p_zoom, bool p_zoom_on_mouse_pos, Vector2i p_scroll) {
	// Compute the minimum sizes.
	Size2i base_tiles_control_size = _compute_base_tiles_control_size();
	base_tiles_root_control->set_custom_minimum_size(Vector2(base_tiles_control_size) * p_zoom);

	Size2i alternative_tiles_control_size = _compute_alternative_tiles_control_size();
	alternative_tiles_root_control->set_custom_minimum_size(Vector2(alternative_tiles_control_size) * p_zoom);

	// Set the texture for the base tiles.
	Ref<Texture2D> texture = tile_set_atlas_source->get_texture();

	// Set the scales.
	if (base_tiles_control_size.x > 0 && base_tiles_control_size.y > 0) {
		base_tiles_drawing_root->set_scale(Vector2(p_zoom, p_zoom));
	} else {
		base_tiles_drawing_root->set_scale(Vector2(1, 1));
	}
	if (alternative_tiles_control_size.x > 0 && alternative_tiles_control_size.y > 0) {
		alternative_tiles_drawing_root->set_scale(Vector2(p_zoom, p_zoom));
	} else {
		alternative_tiles_drawing_root->set_scale(Vector2(1, 1));
	}

	// Update the margin container's margins.
	const char *constants[] = { "margin_left", "margin_top", "margin_right", "margin_bottom" };
	for (int i = 0; i < 4; i++) {
		margin_container->add_theme_constant_override(constants[i], margin_container_paddings[i] * p_zoom);
	}

	// Update the backgrounds.
	background_left->update();
	background_right->update();

	if (p_scroll != Vector2i(-1, -1)) {
		scroll_container->set_h_scroll(p_scroll.x);
		scroll_container->set_v_scroll(p_scroll.y);
	}

	// Zoom on the position.
	if (previous_zoom != p_zoom) {
		// TODO: solve this.
		// There is however an issue with scrollcainter preventing this, as it seems
		// that the scrollbars are not updated right aways after its children update.

		// Compute point on previous area.
		/*Vector2 max = Vector2(scroll_container->get_h_scrollbar()->get_max(), scroll_container->get_v_scrollbar()->get_max());
		Vector2 min = Vector2(scroll_container->get_h_scrollbar()->get_min(), scroll_container->get_v_scrollbar()->get_min());
		Vector2 value = Vector2(scroll_container->get_h_scrollbar()->get_value(), scroll_container->get_v_scrollbar()->get_value());

		Vector2 old_max = max * previous_zoom / p_zoom;

		Vector2 max_pixel_change = max - old_max;
		Vector2 ratio = ((value + scroll_container->get_local_mouse_position()) / old_max).max(Vector2()).min(Vector2(1,1));
		Vector2 offset = max_pixel_change * ratio;

		print_line("--- ZOOMED ---");
		print_line(vformat("max: %s", max));
		print_line(vformat("min: %s", min));
		print_line(vformat("value: %s", value));
		print_line(vformat("size: %s", scroll_container->get_size()));
		print_line(vformat("mouse_pos: %s", scroll_container->get_local_mouse_position()));

		print_line(vformat("ratio: %s", ratio));
		print_line(vformat("max_pixel_change: %s", max_pixel_change));
		print_line(vformat("offset: %s", offset));


		print_line(vformat("value before: %s", Vector2(scroll_container->get_h_scroll(), scroll_container->get_v_scroll())));
		scroll_container->set_h_scroll(10000);//scroll_container->get_h_scroll()+offset.x);
		scroll_container->set_v_scroll(10000);//scroll_container->get_v_scroll()+offset.y);
		print_line(vformat("value after: %s", Vector2(scroll_container->get_h_scroll(), scroll_container->get_v_scroll())));
		*/

		previous_zoom = p_zoom;
	}
}

void TileAtlasView::_scroll_changed() {
	emit_signal("transform_changed", zoom_widget->get_zoom(), Vector2(scroll_container->get_h_scroll(), scroll_container->get_v_scroll()));
}

void TileAtlasView::_zoom_widget_changed() {
	_update_zoom(zoom_widget->get_zoom());
	emit_signal("transform_changed", zoom_widget->get_zoom(), Vector2(scroll_container->get_h_scroll(), scroll_container->get_v_scroll()));
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
		Vector2i texture_region_size = tile_set_atlas_source->get_texture_region_size();

		// Draw the texture, square by square.
		Size2i grid_size = tile_set_atlas_source->get_atlas_grid_size();
		for (int x = 0; x < grid_size.x; x++) {
			for (int y = 0; y < grid_size.y; y++) {
				Vector2i coords = Vector2i(x, y);
				if (tile_set_atlas_source->get_tile_at_coords(coords) == TileSetSource::INVALID_ATLAS_COORDS) {
					Rect2i rect = Rect2i(texture_region_size * coords + margins, texture_region_size);
					base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
				}
			}
		}

		// Draw the texture around the grid.
		Rect2i rect;
		// Top.
		rect.position = Vector2i();
		rect.set_end(Vector2i(texture->get_size().x, margins.y));
		base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
		// Bottom
		int bottom_border = margins.y + (grid_size.y * texture_region_size.y);
		if (bottom_border < texture->get_size().y) {
			rect.position = Vector2i(0, bottom_border);
			rect.set_end(texture->get_size());
			base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
		}
		// Left
		rect.position = Vector2i(0, margins.y);
		rect.set_end(Vector2i(margins.x, margins.y + (grid_size.y * texture_region_size.y)));
		base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
		// Right.
		int right_border = margins.x + (grid_size.x * texture_region_size.x);
		if (right_border < texture->get_size().x) {
			rect.position = Vector2i(right_border, margins.y);
			rect.set_end(Vector2i(texture->get_size().x, margins.y + (grid_size.y * texture_region_size.y)));
			base_tiles_draw->draw_texture_rect_region(texture, rect, rect);
		}

		// Draw actual tiles, using their properties (modulation, etc...)
		for (int i = 0; i < tile_set_atlas_source->get_tiles_count(); i++) {
			Vector2i atlas_coords = tile_set_atlas_source->get_tile_id(i);

			// Update the y to max value.
			Vector2i offset_pos = (margins + (atlas_coords * texture_region_size) + tile_set_atlas_source->get_tile_texture_region(atlas_coords).size / 2 + tile_set_atlas_source->get_tile_effective_texture_offset(atlas_coords, 0));

			// Draw the tile.
			TileSetPluginAtlasRendering::draw_tile(base_tiles_draw->get_canvas_item(), offset_pos, tile_set, source_id, atlas_coords, 0);
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

void TileAtlasView::_draw_base_tiles_dark() {
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

				if (base_tile_coords == TileSetSource::INVALID_ATLAS_COORDS) {
					// Draw the grid.
					base_tiles_dark->draw_rect(Rect2i(origin, texture_region_size), Color(0.0, 0.0, 0.0, 0.5), true);
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
		Rect2i texture_region = tile_set_atlas_source->get_tile_texture_region(tile_id);
		Vector2 origin = texture_region.position + (texture_region.size - tile_shape_size) / 2 + in_tile_base_offset;

		// Draw only if the tile shape fits in the texture region
		tile_set->draw_tile_shape(base_tiles_shape_grid, Rect2(origin, tile_shape_size), grid_color);
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
			Rect2i texture_region = tile_set_atlas_source->get_tile_texture_region(atlas_coords);
			int alternatives_count = tile_set_atlas_source->get_alternative_tiles_count(atlas_coords);
			for (int j = 1; j < alternatives_count; j++) {
				int alternative_id = tile_set_atlas_source->get_alternative_tile_id(atlas_coords, j);
				TileData *tile_data = Object::cast_to<TileData>(tile_set_atlas_source->get_tile_data(atlas_coords, alternative_id));
				bool transposed = tile_data->get_transpose();

				// Update the y to max value.
				Vector2i offset_pos = current_pos;
				if (transposed) {
					offset_pos = (current_pos + Vector2(texture_region.size.y, texture_region.size.x) / 2 + tile_set_atlas_source->get_tile_effective_texture_offset(atlas_coords, alternative_id));
					y_increment = MAX(y_increment, texture_region.size.x);
				} else {
					offset_pos = (current_pos + texture_region.size / 2 + tile_set_atlas_source->get_tile_effective_texture_offset(atlas_coords, alternative_id));
					y_increment = MAX(y_increment, texture_region.size.y);
				}

				// Draw the tile.
				TileSetPluginAtlasRendering::draw_tile(alternatives_draw->get_canvas_item(), offset_pos, tile_set, source_id, atlas_coords, alternative_id);

				// Increment the x position.
				current_pos.x += transposed ? texture_region.size.y : texture_region.size.x;
			}
			if (alternatives_count > 1) {
				current_pos.y += y_increment;
			}
		}
	}
}

void TileAtlasView::_draw_background_left() {
	Ref<Texture2D> texture = get_theme_icon("Checkerboard", "EditorIcons");
	background_left->set_size(base_tiles_root_control->get_custom_minimum_size());
	background_left->draw_texture_rect(texture, Rect2(Vector2(), background_left->get_size()), true);
}

void TileAtlasView::_draw_background_right() {
	Ref<Texture2D> texture = get_theme_icon("Checkerboard", "EditorIcons");
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
	_update_zoom(zoom_widget->get_zoom());

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
	base_tiles_dark->update();
	alternatives_draw->update();
	background_left->update();
	background_right->update();
}

float TileAtlasView::get_zoom() const {
	return zoom_widget->get_zoom();
};

void TileAtlasView::set_transform(float p_zoom, Vector2i p_scroll) {
	zoom_widget->set_zoom(p_zoom);
	_update_zoom(zoom_widget->get_zoom(), false, p_scroll);
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
	for (Map<Vector2, Map<int, Rect2i>>::Element *E_coords = alternative_tiles_rect_cache.front(); E_coords; E_coords = E_coords->next()) {
		for (Map<int, Rect2i>::Element *E_alternative = E_coords->value().front(); E_alternative; E_alternative = E_alternative->next()) {
			if (E_alternative->value().has_point(p_pos)) {
				return Vector3i(E_coords->key().x, E_coords->key().y, E_alternative->key());
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
	scroll_container->update();
	base_tiles_texture_grid->update();
	base_tiles_shape_grid->update();
	base_tiles_dark->update();
	alternatives_draw->update();
	background_left->update();
	background_right->update();
}

void TileAtlasView::_bind_methods() {
	ADD_SIGNAL(MethodInfo("transform_changed", PropertyInfo(Variant::FLOAT, "zoom"), PropertyInfo(Variant::VECTOR2, "scroll")));
}

TileAtlasView::TileAtlasView() {
	Panel *panel_container = memnew(Panel);
	panel_container->set_h_size_flags(SIZE_EXPAND_FILL);
	panel_container->set_v_size_flags(SIZE_EXPAND_FILL);
	panel_container->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	add_child(panel_container);

	//Scrolling
	scroll_container = memnew(ScrollContainer);
	scroll_container->get_h_scrollbar()->connect("value_changed", callable_mp(this, &TileAtlasView::_scroll_changed).unbind(1));
	scroll_container->get_v_scrollbar()->connect("value_changed", callable_mp(this, &TileAtlasView::_scroll_changed).unbind(1));
	panel_container->add_child(scroll_container);
	scroll_container->set_anchors_and_offsets_preset(Control::PRESET_WIDE);

	zoom_widget = memnew(EditorZoomWidget);
	add_child(zoom_widget);
	zoom_widget->set_anchors_and_offsets_preset(Control::PRESET_TOP_LEFT, Control::PRESET_MODE_MINSIZE, 2 * EDSCALE);
	zoom_widget->connect("zoom_changed", callable_mp(this, &TileAtlasView::_zoom_widget_changed).unbind(1));

	CenterContainer *center_container = memnew(CenterContainer);
	center_container->set_h_size_flags(SIZE_EXPAND_FILL);
	center_container->set_v_size_flags(SIZE_EXPAND_FILL);
	center_container->connect("gui_input", callable_mp(this, &TileAtlasView::_gui_input));
	scroll_container->add_child(center_container);

	missing_source_label = memnew(Label);
	missing_source_label->set_text(TTR("No atlas source with a valid texture selected."));
	center_container->add_child(missing_source_label);

	margin_container = memnew(MarginContainer);
	center_container->add_child(margin_container);

	hbox = memnew(HBoxContainer);
	hbox->add_theme_constant_override("separation", 10);
	hbox->hide();
	margin_container->add_child(hbox);

	VBoxContainer *left_vbox = memnew(VBoxContainer);
	hbox->add_child(left_vbox);

	VBoxContainer *right_vbox = memnew(VBoxContainer);
	hbox->add_child(right_vbox);

	// Base tiles.
	Label *base_tile_label = memnew(Label);
	base_tile_label->set_text(TTR("Base Tiles"));
	base_tile_label->set_align(Label::ALIGN_CENTER);
	left_vbox->add_child(base_tile_label);

	base_tiles_root_control = memnew(Control);
	base_tiles_root_control->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	base_tiles_root_control->connect("gui_input", callable_mp(this, &TileAtlasView::_base_tiles_root_control_gui_input));
	left_vbox->add_child(base_tiles_root_control);

	background_left = memnew(Control);
	background_left->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	background_left->set_texture_repeat(TextureRepeat::TEXTURE_REPEAT_ENABLED);
	background_left->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	background_left->connect("draw", callable_mp(this, &TileAtlasView::_draw_background_left));
	base_tiles_root_control->add_child(background_left);

	base_tiles_drawing_root = memnew(Control);
	base_tiles_drawing_root->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	base_tiles_drawing_root->set_texture_filter(TEXTURE_FILTER_NEAREST);
	base_tiles_drawing_root->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	base_tiles_root_control->add_child(base_tiles_drawing_root);

	base_tiles_draw = memnew(Control);
	base_tiles_draw->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	base_tiles_draw->connect("draw", callable_mp(this, &TileAtlasView::_draw_base_tiles));
	base_tiles_draw->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	base_tiles_drawing_root->add_child(base_tiles_draw);

	base_tiles_texture_grid = memnew(Control);
	base_tiles_texture_grid->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	base_tiles_texture_grid->connect("draw", callable_mp(this, &TileAtlasView::_draw_base_tiles_texture_grid));
	base_tiles_texture_grid->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	base_tiles_drawing_root->add_child(base_tiles_texture_grid);

	base_tiles_shape_grid = memnew(Control);
	base_tiles_shape_grid->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	base_tiles_shape_grid->connect("draw", callable_mp(this, &TileAtlasView::_draw_base_tiles_shape_grid));
	base_tiles_shape_grid->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	base_tiles_drawing_root->add_child(base_tiles_shape_grid);

	base_tiles_dark = memnew(Control);
	base_tiles_dark->set_anchors_and_offsets_preset(Control::PRESET_WIDE);
	base_tiles_dark->connect("draw", callable_mp(this, &TileAtlasView::_draw_base_tiles_dark));
	base_tiles_dark->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	base_tiles_drawing_root->add_child(base_tiles_dark);

	// Alternative tiles.
	Label *alternative_tiles_label = memnew(Label);
	alternative_tiles_label->set_text(TTR("Alternative Tiles"));
	alternative_tiles_label->set_align(Label::ALIGN_CENTER);
	right_vbox->add_child(alternative_tiles_label);

	alternative_tiles_root_control = memnew(Control);
	alternative_tiles_root_control->connect("gui_input", callable_mp(this, &TileAtlasView::_alternative_tiles_root_control_gui_input));
	right_vbox->add_child(alternative_tiles_root_control);

	background_right = memnew(Control);
	background_right->set_texture_repeat(TextureRepeat::TEXTURE_REPEAT_ENABLED);
	background_right->connect("draw", callable_mp(this, &TileAtlasView::_draw_background_right));
	background_right->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	alternative_tiles_root_control->add_child(background_right);

	alternative_tiles_drawing_root = memnew(Control);
	alternative_tiles_drawing_root->set_texture_filter(TEXTURE_FILTER_NEAREST);
	alternative_tiles_drawing_root->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	alternative_tiles_root_control->add_child(alternative_tiles_drawing_root);

	alternatives_draw = memnew(Control);
	alternatives_draw->connect("draw", callable_mp(this, &TileAtlasView::_draw_alternatives));
	alternatives_draw->set_mouse_filter(Control::MOUSE_FILTER_IGNORE);
	alternative_tiles_drawing_root->add_child(alternatives_draw);
}
