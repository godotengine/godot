/**************************************************************************/
/*  animation.compat.inc                                                  */
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

Vector3 Animation::_position_track_interpolate_bind_compat_86629(int p_track, double p_time) const {
	return position_track_interpolate(p_track, p_time, false);
}

Quaternion Animation::_rotation_track_interpolate_bind_compat_86629(int p_track, double p_time) const {
	return rotation_track_interpolate(p_track, p_time, false);
}

Vector3 Animation::_scale_track_interpolate_bind_compat_86629(int p_track, double p_time) const {
	return scale_track_interpolate(p_track, p_time, false);
}

float Animation::_blend_shape_track_interpolate_bind_compat_86629(int p_track, double p_time) const {
	return blend_shape_track_interpolate(p_track, p_time, false);
}

Variant Animation::_value_track_interpolate_bind_compat_86629(int p_track, double p_time) const {
	return value_track_interpolate(p_track, p_time, false);
}

int Animation::_track_find_key_bind_compat_92861(int p_track, double p_time, FindMode p_find_mode) const {
	return track_find_key(p_track, p_time, p_find_mode, false, false);
}

void Animation::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("position_track_interpolate", "track_idx", "time_sec"), &Animation::_position_track_interpolate_bind_compat_86629);
	ClassDB::bind_compatibility_method(D_METHOD("rotation_track_interpolate", "track_idx", "time_sec"), &Animation::_rotation_track_interpolate_bind_compat_86629);
	ClassDB::bind_compatibility_method(D_METHOD("scale_track_interpolate", "track_idx", "time_sec"), &Animation::_scale_track_interpolate_bind_compat_86629);
	ClassDB::bind_compatibility_method(D_METHOD("blend_shape_track_interpolate", "track_idx", "time_sec"), &Animation::_blend_shape_track_interpolate_bind_compat_86629);
	ClassDB::bind_compatibility_method(D_METHOD("value_track_interpolate", "track_idx", "time_sec"), &Animation::_value_track_interpolate_bind_compat_86629);
	ClassDB::bind_compatibility_method(D_METHOD("track_find_key", "track_idx", "time", "find_mode"), &Animation::_track_find_key_bind_compat_92861, DEFVAL(FIND_MODE_NEAREST));
}

#endif // DISABLE_DEPRECATED
