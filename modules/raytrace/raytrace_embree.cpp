#include "raytrace_embree.h"
#include "math/vec2.h"
#include "math/vec3.h"

using namespace embree;

RaytraceEngine *RaytraceEngine::singleton = NULL;

RaytraceEngine::RaytraceEngine() {
	singleton = this;
}

RaytraceEngine *RaytraceEngine::get_singleton() {
	return singleton;
}

RTCScene EmbreeRaytraceEngine::embree_scene;
RTCDevice EmbreeRaytraceEngine::embree_device;
Map<unsigned int, EmbreeRaytraceEngine::AlphaTextureData> EmbreeRaytraceEngine::alpha_textures;
Set<int> EmbreeRaytraceEngine::filter_meshes;

void EmbreeRaytraceEngine::filter_function(const struct RTCFilterFunctionNArguments *p_args) {

	RTCHit *hit = (RTCHit *)p_args->hit;

	if (filter_meshes.has(hit->geomID)) {
		p_args->valid[0] = 0;
		return;
	}

	unsigned int geomID = hit->geomID;
	rtcInterpolate0(rtcGetGeometry(embree_scene, geomID), hit->primID, hit->u, hit->v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, &hit->u, 2);

	if (alpha_textures.has(geomID)) {
		const AlphaTextureData &alpha_texture = alpha_textures[geomID];

		if (alpha_texture.sample(hit->u, hit->v) < 128) {
			p_args->valid[0] = 0;
			return;
		}
	}
}

bool EmbreeRaytraceEngine::intersect(RaytraceEngine::Ray &r_ray) {
	RTCIntersectContext context;

	rtcInitIntersectContext(&context);
	context.filter = filter_function;

	rtcIntersect1(embree_scene, &context, (RTCRayHit *)&r_ray);
	return r_ray.geomID != RTC_INVALID_GEOMETRY_ID;
}

void EmbreeRaytraceEngine::intersect(Vector<RaytraceEngine::Ray> &r_rays) {
	Ray *rays = r_rays.ptrw();
	for (int i = 0; i < r_rays.size(); ++i) {
		intersect(rays[i]);
	}
}

void embree_error_handler(void *p_user_data, RTCError p_code, const char *p_str) {
	print_error("Embree error: " + String(p_str));
}

void EmbreeRaytraceEngine::init_scene() {
	if (embree_device == NULL) {
		embree_device = rtcNewDevice(NULL);
	}

	rtcSetDeviceErrorFunction(embree_device, &embree_error_handler, NULL);

	if (embree_scene != NULL) {
		rtcReleaseScene(embree_scene);
	}

	embree_scene = rtcNewScene(embree_device);

	clear_mesh_filter();
	alpha_textures.clear();
}

void EmbreeRaytraceEngine::_add_vertex(Vector<float> &r_vertex_array, const Vector3 &p_vec3) {
	r_vertex_array.push_back(p_vec3.x);
	r_vertex_array.push_back(p_vec3.y);
	r_vertex_array.push_back(p_vec3.z);
}

float blerp(float c00, float c10, float c01, float c11, float tx, float ty) {
	return Math::lerp(Math::lerp(c00, c10, tx), Math::lerp(c01, c11, tx), ty);
}

uint8_t EmbreeRaytraceEngine::AlphaTextureData::sample(float u, float v) const {
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

void EmbreeRaytraceEngine::add_mesh(const Ref<Mesh> p_mesh, const Transform &p_xform, unsigned int p_id) {

	Vector<float> vertices;
	Vector<float> light_uvs;
	Vector<unsigned> triangles;

	int current_vertex_count = 0;

	for (int i = 0; i < p_mesh->get_surface_count(); i++) {

		if (p_mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES)
			continue;

		current_vertex_count = vertices.size() / 3;

		int index_count = 0;
		if (p_mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {
			index_count = p_mesh->surface_get_array_index_len(i);
		} else {
			index_count = p_mesh->surface_get_array_len(i);
		}

		ERR_CONTINUE((index_count == 0 || (index_count % 3) != 0));

		int face_count = index_count / 3;

		Array a = p_mesh->surface_get_arrays(i);

		PoolVector<Vector3> mesh_vertices = a[Mesh::ARRAY_VERTEX];
		PoolVector<Vector3>::Read vr = mesh_vertices.read();

		PoolVector<Vector2> mesh_lightmap_uvs = a[Mesh::ARRAY_TEX_UV2];
		PoolVector<Vector2>::Read uv2r = mesh_lightmap_uvs.read();

		if (p_mesh->surface_get_format(i) & Mesh::ARRAY_FORMAT_INDEX) {

			PoolVector<int> mesh_indices = a[Mesh::ARRAY_INDEX];
			PoolVector<int>::Read ir = mesh_indices.read();

			for (int j = 0; j < mesh_vertices.size(); j++) {
				_add_vertex(vertices, p_xform.xform(vr[j]));
			}

			if (mesh_lightmap_uvs.size() > 0) {
				for (int j = 0; j < mesh_vertices.size(); j++) {
					light_uvs.push_back(uv2r[j].x);
					light_uvs.push_back(uv2r[j].y);
				}
			} else {
				for (int j = 0; j < mesh_vertices.size(); j++) {
					light_uvs.push_back(-1.0);
					light_uvs.push_back(-1.0);
				}
			}

			for (int j = 0; j < face_count; j++) {
				// CCW
				triangles.push_back(current_vertex_count + (ir[j * 3 + 0]));
				triangles.push_back(current_vertex_count + (ir[j * 3 + 2]));
				triangles.push_back(current_vertex_count + (ir[j * 3 + 1]));
			}
		} else {
			face_count = mesh_vertices.size() / 3;
			for (int j = 0; j < face_count; j++) {
				_add_vertex(vertices, p_xform.xform(vr[j * 3 + 0]));
				_add_vertex(vertices, p_xform.xform(vr[j * 3 + 2]));
				_add_vertex(vertices, p_xform.xform(vr[j * 3 + 1]));

				if (mesh_lightmap_uvs.size() > 0) {
					light_uvs.push_back(uv2r[j * 3 + 0].x);
					light_uvs.push_back(uv2r[j * 3 + 0].y);
					light_uvs.push_back(uv2r[j * 3 + 2].x);
					light_uvs.push_back(uv2r[j * 3 + 2].y);
					light_uvs.push_back(uv2r[j * 3 + 1].x);
					light_uvs.push_back(uv2r[j * 3 + 1].y);
				} else {
					for (int k = 0; k < 6; ++k)
						light_uvs.push_back(-1.0);
				}

				triangles.push_back(current_vertex_count + (j * 3 + 0));
				triangles.push_back(current_vertex_count + (j * 3 + 1));
				triangles.push_back(current_vertex_count + (j * 3 + 2));
			}
		}
	}

	RTCGeometry embree_mesh = rtcNewGeometry(embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);

	rtcSetGeometryVertexAttributeCount(embree_mesh, 1);

	Vec3fa *embree_vertices = (Vec3fa *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vec3fa), vertices.size() / 3);
	Vec2fa *embree_light_uvs = (Vec2fa *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT2, sizeof(Vec2fa), light_uvs.size() / 2);
	Triangle *embree_triangles = (Triangle *)rtcSetNewGeometryBuffer(embree_mesh, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle), triangles.size() / 3);

	for (int i = 0; i < vertices.size(); i += 3) {
		embree_vertices[i / 3] = Vec3fa(vertices[i + 0], vertices[i + 1], vertices[i + 2]);
	}

	for (int i = 0; i < light_uvs.size(); i += 2) {
		embree_light_uvs[i / 2] = Vec2fa(light_uvs[i + 0], light_uvs[i + 1]);
	}

	for (int i = 0; i < triangles.size(); i += 3) {
		embree_triangles[i / 3] = Triangle(triangles[i + 0], triangles[i + 1], triangles[i + 2]);
	}

	rtcCommitGeometry(embree_mesh);
	rtcAttachGeometryByID(embree_scene, embree_mesh, p_id);
	rtcReleaseGeometry(embree_mesh);
}

void EmbreeRaytraceEngine::commit_scene() {
	rtcSetSceneFlags(embree_scene, RTC_SCENE_FLAG_CONTEXT_FILTER_FUNCTION);
	rtcCommitScene(embree_scene);
}

void EmbreeRaytraceEngine::set_mesh_filter(const Set<int> &p_mesh_ids) {
	filter_meshes = p_mesh_ids;
}

void EmbreeRaytraceEngine::clear_mesh_filter() {
	filter_meshes.clear();
}

EmbreeRaytraceEngine::EmbreeRaytraceEngine() {
	embree_scene = NULL;
	embree_device = NULL;
}

EmbreeRaytraceEngine::~EmbreeRaytraceEngine() {
	if (embree_scene != NULL)
		rtcReleaseScene(embree_scene);
	if (embree_device != NULL)
		rtcReleaseDevice(embree_device);
}

void EmbreeRaytraceEngine::set_mesh_alpha_texture(Ref<Image> p_alpha_texture, unsigned int p_id) {
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
