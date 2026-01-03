/**************************************************************************/
/*  tile_set.compat.inc                                                   */
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

#ifndef DISABLE_DEPRECATED

#include "tile_set.h"

#ifndef NAVIGATION_2D_DISABLED
Ref<NavigationPolygon> TileData::_get_navigation_polygon_bind_compat_84660(int p_layer_id) const {
	return get_navigation_polygon(p_layer_id, false, false, false);
}
#endif // NAVIGATION_2D_DISABLED

Ref<OccluderPolygon2D> TileData::_get_occluder_bind_compat_84660(int p_layer_id) const {
	return get_occluder_polygon(p_layer_id, 0, false, false, false);
}

void TileData::_bind_compatibility_methods() {
#ifndef NAVIGATION_2D_DISABLED
	ClassDB::bind_compatibility_method(D_METHOD("get_navigation_polygon"), &TileData::_get_navigation_polygon_bind_compat_84660);
#endif // NAVIGATION_2D_DISABLED
	ClassDB::bind_compatibility_method(D_METHOD("get_occluder"), &TileData::_get_occluder_bind_compat_84660);
}

#endif
