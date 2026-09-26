/**************************************************************************/
/*  area_2d.compat.inc                                                    */
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

#include "area_2d.h"

#include "core/object/class_db.h"

// Old enum properties for compatibility.

void Area2D::_set_gravity_space_override_mode_bind_compat_121102(SpaceOverride p_mode) {
	set_gravity_space_override_mode((PS2DE::AreaSpaceOverrideMode)p_mode);
}

Area2D::SpaceOverride Area2D::_get_gravity_space_override_mode_bind_compat_121102() const {
	return (Area2D::SpaceOverride)get_gravity_space_override_mode();
}

void Area2D::_set_linear_damp_space_override_mode_bind_compat_121102(SpaceOverride p_mode) {
	set_linear_damp_space_override_mode((PS2DE::AreaSpaceOverrideMode)p_mode);
}

Area2D::SpaceOverride Area2D::_get_linear_damp_space_override_mode_bind_compat_121102() const {
	return (Area2D::SpaceOverride)get_linear_damp_space_override_mode();
}

void Area2D::_set_angular_damp_space_override_mode_bind_compat_121102(SpaceOverride p_mode) {
	set_angular_damp_space_override_mode((PS2DE::AreaSpaceOverrideMode)p_mode);
}

Area2D::SpaceOverride Area2D::_get_angular_damp_space_override_mode_bind_compat_121102() const {
	return (Area2D::SpaceOverride)get_angular_damp_space_override_mode();
}

void Area2D::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("set_gravity_space_override_mode", "mode"), &Area2D::_set_gravity_space_override_mode_bind_compat_121102);
	ClassDB::bind_compatibility_method(D_METHOD("get_gravity_space_override_mode"), &Area2D::_get_gravity_space_override_mode_bind_compat_121102);
	ClassDB::bind_compatibility_method(D_METHOD("set_linear_damp_space_override_mode", "mode"), &Area2D::_set_linear_damp_space_override_mode_bind_compat_121102);
	ClassDB::bind_compatibility_method(D_METHOD("get_linear_damp_space_override_mode"), &Area2D::_get_linear_damp_space_override_mode_bind_compat_121102);
	ClassDB::bind_compatibility_method(D_METHOD("set_angular_damp_space_override_mode", "mode"), &Area2D::_set_angular_damp_space_override_mode_bind_compat_121102);
	ClassDB::bind_compatibility_method(D_METHOD("get_angular_damp_space_override_mode"), &Area2D::_get_angular_damp_space_override_mode_bind_compat_121102);
}

#endif // DISABLE_DEPRECATED
