/**************************************************************************/
/*  tiles_editor_plugin.cpp                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "tiles_editor_plugin.h"

#include "tile_set_editor.h"

#include "core/os/mutex.h"

#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/plugins/canvas_item_editor_plugin.h"

#include "scene/2d/tile_map.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/control.h"
#include "scene/gui/separator.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/tile_set.h"

TilesEditorUtils *TilesEditorUtils::singleton = nullptr;
TileMapEditorPlugin *tile_map_plugin_singleton = nullptr;
TileSetEditorPlugin *tile_set_plugin_singleton = nullptr;

void TilesEditorUtils::_preview_frame_started() {
	RS::get_singleton()->request_frame_drawn_callback(callable_mp(const_cast<TilesEditorUtils *>(this), &TilesEditorUtils::_pattern_preview_done));
}

void TilesEditorUtils::_pattern_preview_done() {
	pattern_preview_done.post();
}

void TilesEditorUtils::_thread_func(void *ud) {
	TilesEditorUtils *te = static_cast<TilesEditorUtils *>(ud);
	set_current_thread_safe_for_nodes(true);
	te->_thread();
}

void TilesEditorUtils::_thread() {
	pattern_thread_exited.clear();
	while (!pattern_thread_exit.is_set()) {
		pattern_preview_sem.wait();

		pattern_preview_mutex.lock();
		if (pattern_preview_queue.size() == 0) {
			pattern_preview_mutex.unlock();
		} else {
			QueueItem item = pattern_preview_queue.front()->get();
			pattern_preview_queue.pop_front();
			pattern_preview_mutex.unlock();

			int thumbnail_size = EDITOR_GET("filesystem/file_dialog/thumbnail_size");
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

				Rect2 encompassing_rect;
				encompassing_rect.set_position(tile_map->map_to_local(used_cells[0]));
				for (int i = 0; i < used_cells.size(); i++) {
					Vector2i cell = used_cells[i];
					Vector2 world_pos = tile_map->map_to_local(cell);
					encompassing_rect.expand_to(world_pos);

					// Texture.
					Ref<TileSetAtlasSource> atlas_source = item.tile_set->get_source(tile_map->get_cell_source_id(0, cell));
					if (atlas_source.is_valid()) {
						Vector2i coords = tile_map->get_cell_atlas_coords(0, cell);
						int alternative = tile_map->get_cell_alternative_tile(0, cell);

						if (atlas_source->has_tile(coords) && atlas_source->has_alternative_tile(coords, alternative)) {
							Vector2 center = world_pos - atlas_source->get_tile_data(coords, alternative)->get_texture_origin();
							encompassing_rect.expand_to(center - atlas_source->get_tile_texture_region(coords).size / 2);
							encompassing_rect.expand_to(center + atlas_source->get_tile_texture_region(coords).size / 2);
						}
					}
				}

				Vector2 scale = thumbnail_size2 / MAX(encompassing_rect.size.x, encompassing_rect.size.y);
				tile_map->set_scale(scale);
				tile_map->set_position(-(scale * encompassing_rect.get_center()) + thumbnail_size2 / 2);

				// Add the viewport at the last moment to avoid rendering too early.
				EditorNode::get_singleton()->call_deferred("add_child", viewport);

				RS::get_singleton()->connect(SNAME("frame_pre_draw"), callable_mp(const_cast<TilesEditorUtils *>(this), &TilesEditorUtils::_preview_frame_started), Object::CONNECT_ONE_SHOT);

				pattern_preview_done.wait();

				Ref<Image> image = viewport->get_texture()->get_image();

				// Find the index for the given pattern. TODO: optimize.
				item.callback.call(item.pattern, ImageTexture::create_from_image(image));

				viewport->queue_free();
			}
		}
	}
	pattern_thread_exited.set();
}

void TilesEditorUtils::queue_pattern_preview(Ref<TileSet> p_tile_set, Ref<TileMapPattern> p_pattern, Callable p_callback) {
	ERR_FAIL_COND(!p_tile_set.is_valid());
	ERR_FAIL_COND(!p_pattern.is_valid());
	{
		MutexLock lock(pattern_preview_mutex);
		pattern_preview_queue.push_back({ p_tile_set, p_pattern, p_callback });
	}
	pattern_preview_sem.post();
}

void TilesEditorUtils::set_sources_lists_current(int p_current) {
	atlas_sources_lists_current = p_current;
}

void TilesEditorUtils::synchronize_sources_list(Object *p_current_list, Object *p_current_sort_button) {
	ItemList *item_list = Object::cast_to<ItemList>(p_current_list);
	MenuButton *sorting_button = Object::cast_to<MenuButton>(p_current_sort_button);
	ERR_FAIL_NULL(item_list);
	ERR_FAIL_NULL(sorting_button);

	if (sorting_button->is_visible_in_tree()) {
		for (int i = 0; i != SOURCE_SORT_MAX; i++) {
			sorting_button->get_popup()->set_item_checked(i, (i == (int)source_sort));
		}
	}

	if (item_list->is_visible_in_tree()) {
		// Make sure the selection is not overwritten after sorting.
		int atlas_sources_lists_current_mem = atlas_sources_lists_current;
		item_list->emit_signal(SNAME("sort_request"));
		atlas_sources_lists_current = atlas_sources_lists_current_mem;

		if (atlas_sources_lists_current < 0 || atlas_sources_lists_current >= item_list->get_item_count()) {
			item_list->deselect_all();
		} else {
			item_list->set_current(atlas_sources_lists_current);
			item_list->ensure_current_is_visible();
			item_list->emit_signal(SNAME("item_selected"), atlas_sources_lists_current);
		}
	}
}

void TilesEditorUtils::set_atlas_view_transform(float p_zoom, Vector2 p_scroll) {
	atlas_view_zoom = p_zoom;
	atlas_view_scroll = p_scroll;
}

void TilesEditorUtils::synchronize_atlas_view(Object *p_current) {
	TileAtlasView *tile_atlas_view = Object::cast_to<TileAtlasView>(p_current);
	ERR_FAIL_NULL(tile_atlas_view);

	if (tile_atlas_view->is_visible_in_tree()) {
		tile_atlas_view->set_transform(atlas_view_zoom, atlas_view_scroll);
	}
}

void TilesEditorUtils::set_sorting_option(int p_option) {
	source_sort = p_option;
}

List<int> TilesEditorUtils::get_sorted_sources(const Ref<TileSet> p_tile_set) const {
	SourceNameComparator::tile_set = p_tile_set;
	List<int> source_ids;

	for (int i = 0; i < p_tile_set->get_source_count(); i++) {
		source_ids.push_back(p_tile_set->get_source_id(i));
	}

	switch (source_sort) {
		case SOURCE_SORT_ID_REVERSE:
			// Already sorted.
			source_ids.reverse();
			break;
		case SOURCE_SORT_NAME:
			source_ids.sort_custom<SourceNameComparator>();
			break;
		case SOURCE_SORT_NAME_REVERSE:
			source_ids.sort_custom<SourceNameComparator>();
			source_ids.reverse();
			break;
		default: // SOURCE_SORT_ID
			break;
	}

	SourceNameComparator::tile_set.unref();
	return source_ids;
}

Ref<TileSet> TilesEditorUtils::SourceNameComparator::tile_set;

bool TilesEditorUtils::SourceNameComparator::operator()(const int &p_a, const int &p_b) const {
	String name_a;
	String name_b;

	{
		TileSetSource *source = *tile_set->get_source(p_a);

		if (!source->get_name().is_empty()) {
			name_a = source->get_name();
		}

		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			Ref<Texture2D> texture = atlas_source->get_texture();
			if (name_a.is_empty() && texture.is_valid()) {
				name_a = texture->get_path().get_file();
			}
		}

		if (name_a.is_empty()) {
			name_a = itos(p_a);
		}
	}

	{
		TileSetSource *source = *tile_set->get_source(p_b);

		if (!source->get_name().is_empty()) {
			name_b = source->get_name();
		}

		TileSetAtlasSource *atlas_source = Object::cast_to<TileSetAtlasSource>(source);
		if (atlas_source) {
			Ref<Texture2D> texture = atlas_source->get_texture();
			if (name_b.is_empty() && texture.is_valid()) {
				name_b = texture->get_path().get_file();
			}
		}

		if (name_b.is_empty()) {
			name_b = itos(p_b);
		}
	}

	return NaturalNoCaseComparator()(name_a, name_b);
}

void TilesEditorUtils::display_tile_set_editor_panel() {
	tile_map_plugin_singleton->hide_editor();
	tile_set_plugin_singleton->make_visible(true);
}

void TilesEditorUtils::draw_selection_rect(CanvasItem *p_ci, const Rect2 &p_rect, const Color &p_color) {
	Ref<Texture2D> selection_texture = EditorNode::get_singleton()->get_editor_theme()->get_icon(SNAME("TileSelection"), EditorStringName(EditorIcons));

	real_t scale = p_ci->get_global_transform().get_scale().x * 0.5;
	p_ci->draw_set_transform(p_rect.position, 0, Vector2(1, 1) / scale);
	RS::get_singleton()->canvas_item_add_nine_patch(
			p_ci->get_canvas_item(), Rect2(Vector2(), p_rect.size * scale), Rect2(), selection_texture->get_rid(),
			Vector2(2, 2), Vector2(2, 2), RS::NINE_PATCH_STRETCH, RS::NINE_PATCH_STRETCH, false, p_color);
	p_ci->draw_set_transform_matrix(Transform2D());
}

TilesEditorUtils::TilesEditorUtils() {
	singleton = this;
	// Pattern preview generation thread.
	pattern_preview_thread.start(_thread_func, this);
}

TilesEditorUtils::~TilesEditorUtils() {
	if (pattern_preview_thread.is_started()) {
		pattern_thread_exit.set();
		pattern_preview_sem.post();
		while (!pattern_thread_exited.is_set()) {
			OS::get_singleton()->delay_usec(10000);
			RenderingServer::get_singleton()->sync(); //sync pending stuff, as thread may be blocked on visual server
		}
		pattern_preview_thread.wait_to_finish();
	}
	singleton = nullptr;
}

void TileMapEditorPlugin::_tile_map_changed() {
	if (tile_map_changed_needs_update) {
		return;
	}
	tile_map_changed_needs_update = true;
	callable_mp(this, &TileMapEditorPlugin::_update_tile_map).call_deferred();
}

void TileMapEditorPlugin::_update_tile_map() {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (tile_map) {
		Ref<TileSet> tile_set = tile_map->get_tileset();
		if (tile_set.is_valid() && edited_tileset != tile_set->get_instance_id()) {
			tile_set_plugin_singleton->edit(tile_map->get_tileset().ptr());
			tile_set_plugin_singleton->make_visible(true);
			edited_tileset = tile_set->get_instance_id();
		} else if (tile_set.is_null()) {
			tile_set_plugin_singleton->edit(nullptr);
			tile_set_plugin_singleton->make_visible(false);
			edited_tileset = ObjectID();
		}
	}
	tile_map_changed_needs_update = false;
}

void TileMapEditorPlugin::_notification(int p_notification) {
	if (p_notification == NOTIFICATION_EXIT_TREE) {
		get_tree()->queue_delete(TilesEditorUtils::get_singleton());
	}
}

void TileMapEditorPlugin::edit(Object *p_object) {
	TileMap *tile_map = Object::cast_to<TileMap>(ObjectDB::get_instance(tile_map_id));
	if (tile_map) {
		tile_map->disconnect("changed", callable_mp(this, &TileMapEditorPlugin::_tile_map_changed));
	}

	tile_map = Object::cast_to<TileMap>(p_object);
	if (tile_map) {
		tile_map_id = tile_map->get_instance_id();
	} else {
		tile_map_id = ObjectID();
	}

	editor->edit(tile_map);
	if (tile_map) {
		tile_map->connect("changed", callable_mp(this, &TileMapEditorPlugin::_tile_map_changed));

		if (tile_map->get_tileset().is_valid()) {
			tile_set_plugin_singleton->edit(tile_map->get_tileset().ptr());
			tile_set_plugin_singleton->make_visible(true);
			edited_tileset = tile_map->get_tileset()->get_instance_id();
		}
	} else if (edited_tileset.is_valid()) {
		// Hide the TileSet editor, unless another TileSet is being edited.
		if (tile_set_plugin_singleton->get_edited_tileset() == edited_tileset) {
			tile_set_plugin_singleton->edit(nullptr);
			tile_set_plugin_singleton->make_visible(false);
		}
		edited_tileset = ObjectID();
	}
}

bool TileMapEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<TileMap>(p_object) != nullptr;
}

void TileMapEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button->show();
		EditorNode::get_singleton()->make_bottom_panel_item_visible(editor);
	} else {
		button->hide();
		if (editor->is_visible_in_tree()) {
			EditorNode::get_singleton()->hide_bottom_panel();
		}
	}
}

bool TileMapEditorPlugin::forward_canvas_gui_input(const Ref<InputEvent> &p_event) {
	return editor->forward_canvas_gui_input(p_event);
}

void TileMapEditorPlugin::forward_canvas_draw_over_viewport(Control *p_overlay) {
	editor->forward_canvas_draw_over_viewport(p_overlay);
}

void TileMapEditorPlugin::hide_editor() {
	if (editor->is_visible_in_tree()) {
		EditorNode::get_singleton()->hide_bottom_panel();
	}
}

bool TileMapEditorPlugin::is_editor_visible() const {
	return editor->is_visible_in_tree();
}

TileMapEditorPlugin::TileMapEditorPlugin() {
	if (!TilesEditorUtils::get_singleton()) {
		memnew(TilesEditorUtils);
	}
	tile_map_plugin_singleton = this;

	editor = memnew(TileMapEditor);
	editor->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor->set_custom_minimum_size(Size2(0, 200) * EDSCALE);
	editor->hide();

	button = EditorNode::get_singleton()->add_bottom_panel_item(TTR("TileMap"), editor);
	button->hide();
}

TileMapEditorPlugin::~TileMapEditorPlugin() {
	tile_map_plugin_singleton = nullptr;
}

void TileSetEditorPlugin::edit(Object *p_object) {
	editor->edit(Ref<TileSet>(p_object));
	if (p_object) {
		edited_tileset = p_object->get_instance_id();
	} else {
		edited_tileset = ObjectID();
	}
}

bool TileSetEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<TileSet>(p_object) != nullptr;
}

void TileSetEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		button->show();
		if (!tile_map_plugin_singleton->is_editor_visible()) {
			EditorNode::get_singleton()->make_bottom_panel_item_visible(editor);
		}
	} else {
		button->hide();
		if (editor->is_visible_in_tree()) {
			EditorNode::get_singleton()->hide_bottom_panel();
		}
	}
}

ObjectID TileSetEditorPlugin::get_edited_tileset() const {
	return edited_tileset;
}

TileSetEditorPlugin::TileSetEditorPlugin() {
	if (!TilesEditorUtils::get_singleton()) {
		memnew(TilesEditorUtils);
	}
	tile_set_plugin_singleton = this;

	editor = memnew(TileSetEditor);
	editor->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	editor->set_custom_minimum_size(Size2(0, 200) * EDSCALE);
	editor->hide();

	button = EditorNode::get_singleton()->add_bottom_panel_item(TTR("TileSet"), editor);
	button->hide();
}

TileSetEditorPlugin::~TileSetEditorPlugin() {
	tile_set_plugin_singleton = nullptr;
}
