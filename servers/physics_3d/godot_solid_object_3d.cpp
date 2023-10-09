/**************************************************************************/
/*  godot_solid_object_3d.cpp                                             */
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

#include "godot_solid_object_3d.h"

#include "godot_space_3d.h"

Vector3 GodotSolidObject3D::compute_gravity() {
	Vector3 gravity;
	// Add gravity from areas in order of priority.
	int area_count = areas.size();
	if (area_count) {
		areas.sort();
		const Area3DCMP *aa = &areas[0];
		for (int i = area_count - 1; i >= 0; i--) {
			PhysicsServer3D::AreaSpaceOverrideMode area_gravity_mode = (PhysicsServer3D::AreaSpaceOverrideMode)(int)aa[i].area->get_param(PhysicsServer3D::AREA_PARAM_GRAVITY_OVERRIDE_MODE);
			if (area_gravity_mode != PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED) {
				Vector3 area_gravity;
				aa[i].area->compute_gravity(get_transform().get_origin(), area_gravity);
				switch (area_gravity_mode) {
					case PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE:
					case PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE_REPLACE: {
						gravity += area_gravity;
						if (area_gravity_mode == PhysicsServer3D::AREA_SPACE_OVERRIDE_COMBINE_REPLACE) {
							return gravity;
						}
					} break;
					case PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE:
					case PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE_COMBINE: {
						gravity = area_gravity;
						if (area_gravity_mode == PhysicsServer3D::AREA_SPACE_OVERRIDE_REPLACE) {
							return gravity;
						}
					} break;
					case PhysicsServer3D::AREA_SPACE_OVERRIDE_DISABLED: {
					}
				}
			}
		}
	}
	// Add global world gravity from the space's default area.
	GodotArea3D *default_area = get_space()->get_default_area();
	ERR_FAIL_NULL_V(default_area, gravity);
	Vector3 global_default_gravity;
	default_area->compute_gravity(get_transform().get_origin(), global_default_gravity);
	gravity += global_default_gravity;
	return gravity;
}
