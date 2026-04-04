/**************************************************************************/
/*  tile_set_scenes_collection_source.hpp                                 */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/tile_set_source.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PackedScene;

class TileSetScenesCollectionSource : public TileSetSource {
	GDEXTENSION_CLASS(TileSetScenesCollectionSource, TileSetSource)

public:
	int32_t get_scene_tiles_count();
	int32_t get_scene_tile_id(int32_t p_index);
	bool has_scene_tile_id(int32_t p_id);
	int32_t create_scene_tile(const Ref<PackedScene> &p_packed_scene, int32_t p_id_override = -1);
	void set_scene_tile_id(int32_t p_id, int32_t p_new_id);
	void set_scene_tile_scene(int32_t p_id, const Ref<PackedScene> &p_packed_scene);
	Ref<PackedScene> get_scene_tile_scene(int32_t p_id) const;
	void set_scene_tile_display_placeholder(int32_t p_id, bool p_display_placeholder);
	bool get_scene_tile_display_placeholder(int32_t p_id) const;
	void remove_scene_tile(int32_t p_id);
	int32_t get_next_scene_tile_id() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		TileSetSource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

