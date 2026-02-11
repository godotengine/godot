/**************************************************************************/
/*  openxr_spatial_container_state.cpp                                    */
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

#include "openxr_spatial_container_state.h"

#include "core/object/class_db.h"

OpenXRSpatialContainerState::OpenXRSpatialContainerState(const XrSpatialContainerStateEXT &p_state) {
	bounds_mode = _to_bounds_mode(p_state.boundsMode);
	interactable = p_state.interactable;
	visible = p_state.visible;
}

void OpenXRSpatialContainerState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_bounds_mode"), &OpenXRSpatialContainerState::get_bounds_mode);
	ClassDB::bind_method(D_METHOD("is_interactable"), &OpenXRSpatialContainerState::is_interactable);
	ClassDB::bind_method(D_METHOD("is_visible"), &OpenXRSpatialContainerState::is_visible);

	BIND_ENUM_CONSTANT(BOUNDS_MODE_BOUNDED);
	BIND_ENUM_CONSTANT(BOUNDS_MODE_IMMERSIVE);
}

OpenXRSpatialContainerState::BoundsMode OpenXRSpatialContainerState::get_bounds_mode() const {
	return bounds_mode;
}

bool OpenXRSpatialContainerState::is_interactable() const {
	return interactable;
}

bool OpenXRSpatialContainerState::is_visible() const {
	return visible;
}

XrSpatialContainerBoundsModeEXT OpenXRSpatialContainerState::_from_bounds_mode(OpenXRSpatialContainerState::BoundsMode p_mode) {
	switch (p_mode) {
		case BOUNDS_MODE_BOUNDED:
		default: {
			return XR_SPATIAL_CONTAINER_BOUNDS_MODE_BOUNDED_EXT;
		} break;
		case BOUNDS_MODE_IMMERSIVE: {
			return XR_SPATIAL_CONTAINER_BOUNDS_MODE_IMMERSIVE_EXT;
		} break;
	}
}

OpenXRSpatialContainerState::BoundsMode OpenXRSpatialContainerState::_to_bounds_mode(XrSpatialContainerBoundsModeEXT p_mode) {
	switch (p_mode) {
		case XR_SPATIAL_CONTAINER_BOUNDS_MODE_BOUNDED_EXT:
		default: {
			return BOUNDS_MODE_BOUNDED;
		} break;
		case XR_SPATIAL_CONTAINER_BOUNDS_MODE_IMMERSIVE_EXT: {
			return BOUNDS_MODE_IMMERSIVE;
		} break;
	}
}
