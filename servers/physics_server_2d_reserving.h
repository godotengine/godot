/*************************************************************************/
/*  physics_server_2d_reserving.h                                        */
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

#ifndef PHYSICS_2D_SERVER_RESERVING_H
#define PHYSICS_2D_SERVER_RESERVING_H

#include "servers/physics_server_2d.h"

#define DECLARE_RESERVING_API(m_type)                             \
	virtual RID m_type##_create() = 0;                            \
	virtual RID m_type##_create_reserved(RID p_reserved_rid) = 0; \
	virtual RID m_type##_reserve_rid() = 0;

class PhysicsServer2DReserving : public PhysicsServer2D {
public:
	DECLARE_RESERVING_API(line_shape)
	DECLARE_RESERVING_API(ray_shape)
	DECLARE_RESERVING_API(segment_shape)
	DECLARE_RESERVING_API(circle_shape)
	DECLARE_RESERVING_API(rectangle_shape)
	DECLARE_RESERVING_API(capsule_shape)
	DECLARE_RESERVING_API(convex_polygon_shape)
	DECLARE_RESERVING_API(concave_polygon_shape)
	DECLARE_RESERVING_API(space)
	DECLARE_RESERVING_API(area)
	DECLARE_RESERVING_API(body)
};

#undef DECLARE_RESERVING_API

#endif // PHYSICS_2D_SERVER_RESERVING_H
