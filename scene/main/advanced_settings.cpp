/*************************************************************************/
/*  advanced_settings.cpp                                                */
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

#include "advanced_settings.h"
#include "core/engine.h"
#include "core/snappers.h"

VARIANT_ENUM_CAST(AdvancedSettings::Snap2DType);
VARIANT_ENUM_CAST(AdvancedSettings::RoundMode);

void AdvancedSettings::set_transform_snap_2d(Snap2DType p_type, RoundMode p_mode_x, RoundMode p_mode_y) {
	Engine::get_singleton()->get_snappers().set_transform_snap_2d(p_type, p_mode_x, p_mode_y);
}

void AdvancedSettings::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_transform_snap_2d", "snap_type", "mode_x", "mode_y"), &AdvancedSettings::set_transform_snap_2d);

	BIND_ENUM_CONSTANT(SNAP2D_TYPE_ITEM_PRE);
	BIND_ENUM_CONSTANT(SNAP2D_TYPE_ITEM_POST);
	BIND_ENUM_CONSTANT(SNAP2D_TYPE_ITEM_READ);
	BIND_ENUM_CONSTANT(SNAP2D_TYPE_VIEWPORT_PRE);
	BIND_ENUM_CONSTANT(SNAP2D_TYPE_VIEWPORT_POST);
	BIND_ENUM_CONSTANT(SNAP2D_TYPE_VIEWPORT_PARENT_PRE);

	BIND_ENUM_CONSTANT(ROUND_MODE_DISABLED);
	BIND_ENUM_CONSTANT(ROUND_MODE_FLOOR);
	BIND_ENUM_CONSTANT(ROUND_MODE_CEILING);
	BIND_ENUM_CONSTANT(ROUND_MODE_ROUND);
}
