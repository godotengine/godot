/*************************************************************************/
/*  raytrace_embree.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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
