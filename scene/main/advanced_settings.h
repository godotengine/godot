/*************************************************************************/
/*  advanced_settings.h                                                  */
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

#ifndef ADVANCED_SETTINGS_H
#define ADVANCED_SETTINGS_H

#include "scene/main/node.h"

class AdvancedSettings : public Reference {
	GDCLASS(AdvancedSettings, Reference);

public:
	enum Snap2DType {
		SNAP2D_TYPE_ITEM_PRE,
		SNAP2D_TYPE_ITEM_POST,
		SNAP2D_TYPE_ITEM_READ,
		SNAP2D_TYPE_VIEWPORT_PRE,
		SNAP2D_TYPE_VIEWPORT_POST,
		SNAP2D_TYPE_VIEWPORT_PARENT_PRE,
	};

	enum RoundMode {
		ROUND_MODE_DISABLED,
		ROUND_MODE_FLOOR,
		ROUND_MODE_CEILING,
		ROUND_MODE_ROUND,
	};

	void set_transform_snap_2d(Snap2DType p_type, RoundMode p_mode_x, RoundMode p_mode_y);

protected:
	static void _bind_methods();
};

#endif // ADVANCED_SETTINGS_H
