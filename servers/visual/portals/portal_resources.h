/**************************************************************************/
/*  portal_resources.h                                                    */
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

#ifndef PORTAL_RESOURCES_H
#define PORTAL_RESOURCES_H

#include "core/math/geometry.h"
#include "portal_types.h"

// Although the portal renderer is owned by a scenario,
// resources are not associated with a scenario and can be shared
// potentially across multiple scenarios. They must therefore be held in
// some form of global.

class PortalResources {
	friend class PortalRenderer;

public:
	OccluderResourceHandle occluder_resource_create();
	void occluder_resource_prepare(OccluderResourceHandle p_handle, VSOccluder_Instance::Type p_type);
	void occluder_resource_update_spheres(OccluderResourceHandle p_handle, const Vector<Plane> &p_spheres);
	void occluder_resource_update_mesh(OccluderResourceHandle p_handle, const Geometry::OccluderMeshData &p_mesh_data);
	void occluder_resource_destroy(OccluderResourceHandle p_handle);

	const VSOccluder_Resource &get_pool_occluder_resource(uint32_t p_pool_id) const { return _occluder_resource_pool[p_pool_id]; }
	VSOccluder_Resource &get_pool_occluder_resource(uint32_t p_pool_id) { return _occluder_resource_pool[p_pool_id]; }

	// Local space is shared resources
	const VSOccluder_Sphere &get_pool_occluder_local_sphere(uint32_t p_pool_id) const { return _occluder_local_sphere_pool[p_pool_id]; }
	const VSOccluder_Poly &get_pool_occluder_local_poly(uint32_t p_pool_id) const { return _occluder_local_poly_pool[p_pool_id]; }
	const VSOccluder_Hole &get_pool_occluder_local_hole(uint32_t p_pool_id) const { return _occluder_local_hole_pool[p_pool_id]; }
	VSOccluder_Hole &get_pool_occluder_local_hole(uint32_t p_pool_id) { return _occluder_local_hole_pool[p_pool_id]; }

private:
	TrackedPooledList<VSOccluder_Resource> _occluder_resource_pool;
	TrackedPooledList<VSOccluder_Sphere, uint32_t, true> _occluder_local_sphere_pool;
	TrackedPooledList<VSOccluder_Poly, uint32_t, true> _occluder_local_poly_pool;
	TrackedPooledList<VSOccluder_Hole, uint32_t, true> _occluder_local_hole_pool;
};

#endif // PORTAL_RESOURCES_H
