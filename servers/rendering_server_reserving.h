/*************************************************************************/
/*  rendering_server_reserving.h                                         */
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

#ifndef RENDERING_SERVER_RESERVING_H
#define RENDERING_SERVER_RESERVING_H

#include "servers/rendering_server.h"

#define DECLARE_RESERVING_API(m_type)                             \
	virtual RID m_type##_create() = 0;                            \
	virtual RID m_type##_create_reserved(RID p_reserved_rid) = 0; \
	virtual RID m_type##_reserve_rid() = 0;

class RenderingServerReserving : public RenderingServer {
public:
	DECLARE_RESERVING_API(shader)
	DECLARE_RESERVING_API(material)
	DECLARE_RESERVING_API(mesh)
	DECLARE_RESERVING_API(multimesh)
	DECLARE_RESERVING_API(immediate)
	DECLARE_RESERVING_API(skeleton)
	DECLARE_RESERVING_API(directional_light)
	DECLARE_RESERVING_API(omni_light)
	DECLARE_RESERVING_API(spot_light)
	DECLARE_RESERVING_API(reflection_probe)
	DECLARE_RESERVING_API(decal)
	DECLARE_RESERVING_API(gi_probe)
	DECLARE_RESERVING_API(lightmap)
	DECLARE_RESERVING_API(particles)
	DECLARE_RESERVING_API(particles_collision)
	DECLARE_RESERVING_API(camera)
	DECLARE_RESERVING_API(viewport)
	DECLARE_RESERVING_API(sky)
	DECLARE_RESERVING_API(environment)
	DECLARE_RESERVING_API(camera_effects)
	DECLARE_RESERVING_API(scenario)
	DECLARE_RESERVING_API(instance)
	DECLARE_RESERVING_API(canvas)
	DECLARE_RESERVING_API(canvas_texture)
	DECLARE_RESERVING_API(canvas_item)
	DECLARE_RESERVING_API(canvas_light)
	DECLARE_RESERVING_API(canvas_light_occluder)
	DECLARE_RESERVING_API(canvas_occluder_polygon)
};

#undef DECLARE_RESERVING_API

#endif // RENDERING_SERVER_RESERVING_H
