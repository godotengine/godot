/**************************************************************************/
/*  node_3d_editor_constants.h                                            */
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

#include "core/math/math_defs.h"

namespace Node3DEditorConstants {

constexpr real_t GIZMO_ARROW_SIZE = 0.35;
constexpr real_t GIZMO_RING_HALF_WIDTH = 0.1;
constexpr real_t GIZMO_PLANE_SIZE = 0.2;
constexpr real_t GIZMO_PLANE_DST = 0.3;
constexpr real_t GIZMO_CIRCLE_SIZE = 1.1;

constexpr real_t GIZMO_SCALE_OFFSET = GIZMO_CIRCLE_SIZE + 0.3;
constexpr real_t GIZMO_ARROW_OFFSET = GIZMO_CIRCLE_SIZE + 0.3;

constexpr real_t TRACKBALL_SENSITIVITY = 0.005;
constexpr int TRACKBALL_SPHERE_RINGS = 16;
constexpr int TRACKBALL_SPHERE_SECTORS = 32;
constexpr real_t TRACKBALL_HIGHLIGHT_ALPHA = 0.01;
constexpr int GIZMO_HIGHLIGHT_AXIS_VIEW_ROTATION = 15;

constexpr float VERTEX_SNAP_THRESHOLD = 30.0f;
constexpr int GIZMO_HIGHLIGHT_AXIS_TRACKBALL = 16;

constexpr real_t ZOOM_FREELOOK_INDICATOR_DELAY_S = 1.5;

constexpr real_t MIN_Z = 0.01;
constexpr real_t MAX_Z = 1000000.0;

constexpr real_t MIN_FOV = 0.01;
constexpr real_t MAX_FOV = 179;

} // namespace Node3DEditorConstants
