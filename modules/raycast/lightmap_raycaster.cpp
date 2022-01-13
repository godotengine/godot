/*************************************************************************/
/*  lightmap_raycaster.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "lightmap_raycaster.h"

#ifdef __SSE2__
#include <pmmintrin.h>
#endif

LightmapRaycaster *LightmapRaycasterEmbree::create_embree_raycaster() {
	return memnew(LightmapRaycasterEmbree);
}

void LightmapRaycasterEmbree::make_default_raycaster() {
	create_function = create_embree_raycaster;
}

void LightmapRaycasterEmbree::filter_function(const struct RTCFilterFunctionNArguments *p_args) {
	RTCHit *hit = (RTCHit *)p_args->hit;

	unsigned int geomID = hit->geomID;
	float u = hit->u;
	float v = hit->v;

	LightmapRaycasterEmbree *scene = (LightmapRaycasterEmbree *)p_args->geometryUserPtr;
	RTCGeometry geom = rtcGetGeometry(scene->embree_scene, geomID);

	rtcInterpolate0(geom, hit->primID, hit->u, hit->v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, &hit->u, 2);

	if (scene->alpha_textures.has(geomID)) {
		const AlphaTextureData &alpha_texture = scene->alpha_textures[geomID];

		if (alpha_texture.sample(hit->u, hit->v) < 128) {
			p_args->valid[0] = 0;
			return;
		}
	}

	rtcInterpolate0(geom, hit->primID, u, v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1, &hit->Ng_x, 3);
}

bool LightmapRaycasterEmbree::intersect(Ray &r_ray) {
	RTCIntersectContext context;

	rtcInitIntersectContext(&context);

	rtcIntersect1(embree_scene, &context, (RTCRayHit *)&r_ray);
	return r_ray.geomID != RTC_INVALID_GEOMETRY_ID;
}

void LightmapRaycasterEmbree::intersect(Vector<Ray> &r_rays) {
	Ray *rays = r_rays.ptrw();
	for (int i = 0; i < r_rays.size(); ++i) {
		intersect(rays[i]);
	}
}

void LightmapRaycasterEmbree::set_mesh_alpha_texture(Ref<Image> p_alpha_texture, unsigned int p_id) {
	if (p_alpha_texture.is_valid() && p_alpha_texture->get_size() != Vector2i()) {
		AlphaTextureData tex;
		tex.size = p_alpha_texture->get_size();
		tex.data.resize(tex.size.x * tex.size.y);

		{
			PoolVector<uint8_t>::Read r = p_alpha_texture->get_data().read();
			uint8_t *ptrw = tex.data.ptrw();
			for (int i = 0; i < tex.size.x * tex.size.y; ++i) {
				ptrw[i] = r[i];
			}
		}

		alpha_textures.insert(p_id, tex);
	}
}

float blerp(float c00, float c10, float c01, float c11, float tx, float ty) {
	return Math::lerp(Math::lerp(c00, c10, tx), Math::lerp(c01, c11, tx), ty);
}

uint8_t LightmapRaycasterEmbree::AlphaTextureData::sample(float u, float v) const {
	float x = u * size.x;
	float y = v * size.y;
	int xi = (int)x;
	int yi = (int)y;

	uint8_t texels[4];

	for (int i = 0; i < 4; ++i) {
		int sample_x = CLAMP(xi + i % 2, 0, size.x - 1);
		int sample_y = CLAMP(yi + i / 2, 0, size.y - 1);
		texels[i] = data[sample_y * size.x + sample_x];
	}

	return Math::round(blerp(texels[0], texels[1], texels[2], texels[3], x - xi, y - yi));
}

void LightmapRaycasterEmbree::add_mesh(const Vector<Vector3> &p_vertices, const Vector<Vector3> &p_normals, const Vector<Vector2> &p_uv2s, unsigned int p_id) {
	RTCGeometry embree_mesh = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);

	rtcSetGeometryVertexAttributeCount(embree_mesh, 2);

	int vertex_count = p_vertices.size();

	ERR_FAIL_COND(vertex_count % 3 != 0);
	ERR_FAIL_COND(vertex_count != p_uv2s.size());
	ERR_FAIL_COND(!p_normals.empty() && vertex_count != p_normals.size());

	Vector3 *embree_vertices = (Vector3 *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vector3), vertex_count);
	memcpy(embree_vertices, p_vertices.ptr(), sizeof(Vector3) * vertex_count);

	Vector2 *embree_light_uvs = (Vector2 *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT2, sizeof(Vector2), vertex_count);
	memcpy(embree_light_uvs, p_uv2s.ptr(), sizeof(Vector2) * vertex_count);

	uint32_t *embree_triangles = (uint32_t *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(uint32_t) * 3, vertex_count / 3);
	for (int i = 0; i < vertex_count; i++) {
		embree_triangles[i] = i;
	}

	if (!p_normals.empty()) {
		Vector3 *embree_normals = (Vector3 *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1, RTC_FORMAT_FLOAT3, sizeof(Vector3), vertex_count);
		memcpy(embree_normals, p_normals.ptr(), sizeof(Vector3) * vertex_count);
	}

	rtcCommitGeometry(embree_mesh);
	rtcSetGeometryIntersectFilterFunction(embree_mesh, filter_function);
	rtcSetGeometryUserData(embree_mesh, this);
	rtcAttachGeometryByID(embree_scene, embree_mesh, p_id);
	rtcReleaseGeometry(embree_mesh);
}

void LightmapRaycasterEmbree::commit() {
	rtcCommitScene(embree_scene);
}

void LightmapRaycasterEmbree::set_mesh_filter(const Set<int> &p_mesh_ids) {
	for (Set<int>::Element *E = p_mesh_ids.front(); E; E = E->next()) {
		rtcDisableGeometry(rtcGetGeometry(embree_scene, E->get()));
	}
	rtcCommitScene(embree_scene);
	filter_meshes = p_mesh_ids;
}

void LightmapRaycasterEmbree::clear_mesh_filter() {
	for (Set<int>::Element *E = filter_meshes.front(); E; E = E->next()) {
		rtcEnableGeometry(rtcGetGeometry(embree_scene, E->get()));
	}
	rtcCommitScene(embree_scene);
	filter_meshes.clear();
}

void embree_error_handler(void *p_user_data, RTCError p_code, const char *p_str) {
	print_error("Embree error: " + String(p_str));
}

LightmapRaycasterEmbree::LightmapRaycasterEmbree() {
#ifdef __SSE2__
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

	embree_device = rtcNewDevice(nullptr);
	rtcSetDeviceErrorFunction(embree_device, &embree_error_handler, nullptr);
	embree_scene = rtcNewScene(embree_device);
}

LightmapRaycasterEmbree::~LightmapRaycasterEmbree() {
#ifdef __SSE2__
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
#endif

	if (embree_scene != nullptr) {
		rtcReleaseScene(embree_scene);
	}

	if (embree_device != nullptr) {
		rtcReleaseDevice(embree_device);
	}
}
