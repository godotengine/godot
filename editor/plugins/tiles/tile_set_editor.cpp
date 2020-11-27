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

#include "scene/gui/box_container.h"
#include "scene/gui/control.h"
#include "scene/gui/tab_container.h"

void TileSetEditor::edit(TileSet *p_tile_set) {
	if (p_tile_set == tile_set) {
		return;
	}
	tile_set = p_tile_set;
	tileset_tab_sources->edit(p_tile_set);
}

TileSetEditor::TileSetEditor() {
	TabContainer *tileset_tabs_container = memnew(TabContainer);
	tileset_tabs_container->set_v_size_flags(SIZE_EXPAND_FILL);
	tileset_tabs_container->set_tab_align(TabContainer::ALIGN_LEFT);
	add_child(tileset_tabs_container);

	// - Atlas tab -
	tileset_tab_sources = memnew(TileSetEditorSourcesTab);
	tileset_tab_sources->set_name("Atlases");
	tileset_tab_sources->set_h_size_flags(SIZE_EXPAND_FILL);
	tileset_tab_sources->set_v_size_flags(SIZE_EXPAND_FILL);
	tileset_tabs_container->add_child(tileset_tab_sources);

	// - Layers tab -
	Control *tileset_tab_tileset_layers = memnew(Control);
	tileset_tab_tileset_layers->set_name("Tileset layers");
	tileset_tabs_container->add_child(tileset_tab_tileset_layers);

	// - Properties tab -
	Control *tileset_tab_tile_properties = memnew(Control);
	tileset_tab_tile_properties->set_name("Tile properties");
	tileset_tabs_container->add_child(tileset_tab_tile_properties);

	// - Scenes tab -
	Control *tileset_tab_scenes = memnew(Control);
	tileset_tab_scenes->set_name("Scenes");
	tileset_tabs_container->add_child(tileset_tab_scenes);

	// Disable unused tabs.
	tileset_tabs_container->set_tab_disabled(1, true);
	tileset_tabs_container->set_tab_disabled(2, true);
	tileset_tabs_container->set_tab_disabled(3, true);
}
