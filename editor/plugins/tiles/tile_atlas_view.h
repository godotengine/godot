/*************************************************************************/
/*  tile_atlas_view.h                                                    */
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

#ifndef TILE_ATLAS_VIEW_H
#define TILE_ATLAS_VIEW_H

#include "editor/editor_zoom_widget.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/center_container.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/texture_rect.h"
#include "scene/resources/tile_set.h"

class ViewPanner;

class TileAtlasView : public Control {
	GDCLASS(TileAtlasView, Control);

private:
	TileSet *tile_set;
	TileSetAtlasSource *tile_set_atlas_source;
	int source_id = TileSet::INVALID_SOURCE;

	enum DragType {
		DRAG_TYPE_NONE,
		DRAG_TYPE_PAN,
	};
	DragType drag_type = DRAG_TYPE_NONE;
	float previous_zoom = 1.0;
	EditorZoomWidget *zoom_widget;
	Button *button_center_view;
	CenterContainer *center_container;
	Vector2 panning;
	void _update_zoom_and_panning(bool p_zoom_on_mouse_pos = false);
	void _zoom_widget_changed();
	void _center_view();
	virtual void gui_input(const Ref<InputEvent> &p_event) override;

	Ref<ViewPanner> panner;
	void _scroll_callback(Vector2 p_scroll_vec);
	void _pan_callback(Vector2 p_scroll_vec);
	void _zoom_callback(Vector2 p_scroll_vec, Vector2 p_origin);

	Map<Vector2, Map<int, Rect2i>> alternative_tiles_rect_cache;
	void _update_alternative_tiles_rect_cache();

	MarginContainer *margin_container;
	int margin_container_paddings[4] = { 0, 0, 0, 0 };
	HBoxContainer *hbox;
	Label *missing_source_label;

	// Background
	Control *background_left;
	void _draw_background_left();
	Control *background_right;
	void _draw_background_right();

	// Left side.
	Control *base_tiles_root_control;
	void _base_tiles_root_control_gui_input(const Ref<InputEvent> &p_event);

	Control *base_tiles_drawing_root;

	Control *base_tiles_draw;
	void _draw_base_tiles();

	Control *base_tiles_texture_grid;
	void _draw_base_tiles_texture_grid();

	Control *base_tiles_shape_grid;
	void _draw_base_tiles_shape_grid();

	Size2i _compute_base_tiles_control_size();

	// Right side.
	Control *alternative_tiles_root_control;
	void _alternative_tiles_root_control_gui_input(const Ref<InputEvent> &p_event);

	Control *alternative_tiles_drawing_root;

	Control *alternatives_draw;
	void _draw_alternatives();

	Size2i _compute_alternative_tiles_control_size();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	// Global.
	void set_atlas_source(TileSet *p_tile_set, TileSetAtlasSource *p_tile_set_atlas_source, int p_source_id);

	float get_zoom() const;
	void set_transform(float p_zoom, Vector2i p_panning);

	void set_padding(Side p_side, int p_padding);

	// Left side.
	void set_texture_grid_visible(bool p_visible) { base_tiles_texture_grid->set_visible(p_visible); };
	void set_tile_shape_grid_visible(bool p_visible) { base_tiles_shape_grid->set_visible(p_visible); };

	Vector2i get_atlas_tile_coords_at_pos(const Vector2 p_pos) const;

	void add_control_over_atlas_tiles(Control *p_control, bool scaled = true) {
		if (scaled) {
			base_tiles_drawing_root->add_child(p_control);
		} else {
			base_tiles_root_control->add_child(p_control);
		}
		p_control->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	};

	// Right side.
	Vector3i get_alternative_tile_at_pos(const Vector2 p_pos) const;
	Rect2i get_alternative_tile_rect(const Vector2i p_coords, int p_alternative_tile);

	void add_control_over_alternative_tiles(Control *p_control, bool scaled = true) {
		if (scaled) {
			alternative_tiles_drawing_root->add_child(p_control);
		} else {
			alternative_tiles_root_control->add_child(p_control);
		}
		p_control->set_mouse_filter(Control::MOUSE_FILTER_PASS);
	};

	// Update everything.
	void update();

	TileAtlasView();
};

#endif // TILE_ATLAS_VIEW
