/*************************************************************************/
/*  gi_probe.h                                                           */
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

#ifndef GIPROBE_H
#define GIPROBE_H

#include "multimesh_instance.h"
#include "scene/3d/visual_instance.h"

class GIProbeData : public Resource {
	GDCLASS(GIProbeData, Resource);

	RID probe;

protected:
	static void _bind_methods();

public:
	void set_bounds(const AABB &p_bounds);
	AABB get_bounds() const;

	void set_cell_size(float p_size);
	float get_cell_size() const;

	void set_to_cell_xform(const Transform &p_xform);
	Transform get_to_cell_xform() const;

	void set_dynamic_data(const PoolVector<int> &p_data);
	PoolVector<int> get_dynamic_data() const;

	void set_dynamic_range(int p_range);
	int get_dynamic_range() const;

	void set_propagation(float p_range);
	float get_propagation() const;

	void set_energy(float p_range);
	float get_energy() const;

	void set_bias(float p_range);
	float get_bias() const;

	void set_normal_bias(float p_range);
	float get_normal_bias() const;

	void set_interior(bool p_enable);
	bool is_interior() const;

	void set_compress(bool p_enable);
	bool is_compressed() const;

	virtual RID get_rid() const;

	GIProbeData();
	~GIProbeData();
};

class GIProbe : public VisualInstance {
	GDCLASS(GIProbe, VisualInstance);

public:
	enum Subdiv {
		SUBDIV_64,
		SUBDIV_128,
		SUBDIV_256,
		SUBDIV_512,
		SUBDIV_MAX

	};

	typedef void (*BakeBeginFunc)(int);
	typedef void (*BakeStepFunc)(int, const String &);
	typedef void (*BakeEndFunc)();

private:
	Ref<GIProbeData> probe_data;

	RID gi_probe;

	Subdiv subdiv;
	Vector3 extents;
	int dynamic_range;
	float energy;
	float bias;
	float normal_bias;
	float propagation;
	bool interior;
	bool compress;

	struct PlotMesh {
		Ref<Material> override_material;
		Vector<Ref<Material>> instance_materials;
		Ref<Mesh> mesh;
		Transform local_xform;
	};

	void _find_meshes(Node *p_at_node, List<PlotMesh> &plot_meshes);
	void _debug_bake();

protected:
	static void _bind_methods();

public:
	static BakeBeginFunc bake_begin_function;
	static BakeStepFunc bake_step_function;
	static BakeEndFunc bake_end_function;

	void set_probe_data(const Ref<GIProbeData> &p_data);
	Ref<GIProbeData> get_probe_data() const;

	void set_subdiv(Subdiv p_subdiv);
	Subdiv get_subdiv() const;

	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const;

	void set_dynamic_range(int p_dynamic_range);
	int get_dynamic_range() const;

	void set_energy(float p_energy);
	float get_energy() const;

	void set_bias(float p_bias);
	float get_bias() const;

	void set_normal_bias(float p_normal_bias);
	float get_normal_bias() const;

	void set_propagation(float p_propagation);
	float get_propagation() const;

	void set_interior(bool p_enable);
	bool is_interior() const;

	void set_compress(bool p_enable);
	bool is_compressed() const;

	void bake(Node *p_from_node = nullptr, bool p_create_visual_debug = false);

	virtual AABB get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	virtual String get_configuration_warning() const;

	GIProbe();
	~GIProbe();
};

VARIANT_ENUM_CAST(GIProbe::Subdiv)

#endif // GIPROBE_H
