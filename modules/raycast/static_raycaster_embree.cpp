/**************************************************************************/
/*  static_raycaster_embree.cpp                                           */
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

#include "static_raycaster_embree.h"

#ifdef TOOLS_ENABLED

#ifdef __SSE2__
#include <pmmintrin.h>
#endif

RTCDevice StaticRaycasterEmbree::embree_device;

StaticRaycaster *StaticRaycasterEmbree::create_embree_raycaster() {
	return memnew(StaticRaycasterEmbree);
}

void StaticRaycasterEmbree::make_default_raycaster() {
	create_function = create_embree_raycaster;
}

void StaticRaycasterEmbree::free() {
	if (embree_device) {
		rtcReleaseDevice(embree_device);
	}
}

bool StaticRaycasterEmbree::intersect(Ray &r_ray) {
	RTCRayQueryContext context;
	rtcInitRayQueryContext(&context);
	RTCIntersectArguments args;
	rtcInitIntersectArguments(&args);
	args.context = &context;
	rtcIntersect1(embree_scene, (RTCRayHit *)&r_ray, &args);
	return r_ray.geomID != RTC_INVALID_GEOMETRY_ID;
}

void StaticRaycasterEmbree::intersect(Vector<Ray> &r_rays) {
	Ray *rays = r_rays.ptrw();
	for (int i = 0; i < r_rays.size(); ++i) {
		intersect(rays[i]);
	}
}

void StaticRaycasterEmbree::add_mesh(const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices, unsigned int p_id) {
	RTCGeometry embree_mesh = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);

	int vertex_count = p_vertices.size();

	Vector3 *embree_vertices = (Vector3 *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vector3), vertex_count);
	memcpy(embree_vertices, p_vertices.ptr(), sizeof(Vector3) * vertex_count);

	if (p_indices.is_empty()) {
		ERR_FAIL_COND(vertex_count % 3 != 0);
		uint32_t *embree_triangles = (uint32_t *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(uint32_t) * 3, vertex_count / 3);
		for (int i = 0; i < vertex_count; i++) {
			embree_triangles[i] = i;
		}
	} else {
		uint32_t *embree_triangles = (uint32_t *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(uint32_t) * 3, p_indices.size() / 3);
		memcpy(embree_triangles, p_indices.ptr(), sizeof(uint32_t) * p_indices.size());
	}

	rtcCommitGeometry(embree_mesh);
	rtcAttachGeometryByID(embree_scene, embree_mesh, p_id);
	rtcReleaseGeometry(embree_mesh);
}

void StaticRaycasterEmbree::commit() {
	rtcCommitScene(embree_scene);
}

void StaticRaycasterEmbree::set_mesh_filter(const HashSet<int> &p_mesh_ids) {
	for (const int &E : p_mesh_ids) {
		rtcDisableGeometry(rtcGetGeometry(embree_scene, E));
	}
	rtcCommitScene(embree_scene);
	filter_meshes = p_mesh_ids;
}

void StaticRaycasterEmbree::clear_mesh_filter() {
	for (const int &E : filter_meshes) {
		rtcEnableGeometry(rtcGetGeometry(embree_scene, E));
	}
	rtcCommitScene(embree_scene);
	filter_meshes.clear();
}

void embree_error_handler(void *p_user_data, RTCError p_code, const char *p_str) {
	print_error("Embree error: " + String(p_str));
}

StaticRaycasterEmbree::StaticRaycasterEmbree() {
#ifdef __SSE2__
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	if (!embree_device) {
		embree_device = rtcNewDevice(nullptr);
		rtcSetDeviceErrorFunction(embree_device, &embree_error_handler, nullptr);
	}

	embree_scene = rtcNewScene(embree_device);
}

StaticRaycasterEmbree::~StaticRaycasterEmbree() {
	if (embree_scene != nullptr) {
		rtcReleaseScene(embree_scene);
	}
}

#endif // TOOLS_ENABLED
