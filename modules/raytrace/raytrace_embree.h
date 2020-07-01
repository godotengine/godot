#ifndef RAYTRACE_EMBREE_H
#define RAYTRACE_EMBREE_H

#include "raytrace.h"
#include "rtcore.h"

class EmbreeRaytraceEngine : public RaytraceEngine {
private:
	struct AlphaTextureData {
		Vector<uint8_t> data;
		Vector2i size;

		uint8_t sample(float u, float v) const;
	};

	static RTCDevice embree_device;
	static RTCScene embree_scene;

	static void _add_vertex(Vector<float> &r_vertex_array, const Vector3 &p_vec3);
	static void filter_function(const struct RTCFilterFunctionNArguments *p_args);

	static Map<unsigned int, AlphaTextureData> alpha_textures;
	static Set<int> filter_meshes;

public:
	virtual bool intersect(RaytraceEngine::Ray &p_ray);
	virtual void intersect(Vector<Ray> &r_rays);

	virtual void init_scene();
	virtual void add_mesh(const Ref<Mesh> p_mesh, const Transform &p_xform, unsigned int p_id);
	virtual void set_mesh_alpha_texture(Ref<Image> p_alpha_texture, unsigned int p_id);
	virtual void commit_scene();

	virtual void set_mesh_filter(const Set<int> &p_mesh_ids);
	virtual void clear_mesh_filter();

	EmbreeRaytraceEngine();
	~EmbreeRaytraceEngine();
};

#endif //RAYTRACE_EMBREE_H
