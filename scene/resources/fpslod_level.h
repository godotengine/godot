/**************************************************************************/
/*  fpslod_level.h                                                        */
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

#pragma once

#include "core/io/resource.h"

class FPSLODLevel : public Resource {
	GDCLASS(FPSLODLevel, Resource);

	double distance = 0.0;
	int skip_frames = 0;

protected:
	static void _bind_methods() {
		ClassDB::bind_method(D_METHOD("set_distance", "value"), &FPSLODLevel::set_distance);
		ClassDB::bind_method(D_METHOD("get_distance"), &FPSLODLevel::get_distance);
		ClassDB::bind_method(D_METHOD("set_skip_frames", "value"), &FPSLODLevel::set_skip_frames);
		ClassDB::bind_method(D_METHOD("get_skip_frames"), &FPSLODLevel::get_skip_frames);

		ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "distance", PROPERTY_HINT_RANGE, "0,10000,0.1"), "set_distance", "get_distance");
		ADD_PROPERTY(PropertyInfo(Variant::INT, "skip_frames", PROPERTY_HINT_RANGE, "0,60,1"), "set_skip_frames", "get_skip_frames");
	}

public:
	void set_distance(double p_value) { distance = p_value; }
	double get_distance() const { return distance; }

	void set_skip_frames(int p_value) { skip_frames = p_value; }
	int get_skip_frames() const { return skip_frames; }
};
