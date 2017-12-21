#ifndef BAKED_INDIRECT_LIGHT_H
#define BAKED_INDIRECT_LIGHT_H

#include "multimesh_instance.h"
#include "scene/3d/light.h"
#include "scene/3d/visual_instance.h"

class BakedLightmapData : public Resource {
	GDCLASS(BakedLightmapData, Resource);

	RID baked_light;
	AABB bounds;
	float energy;
	int cell_subdiv;
	Transform cell_space_xform;

	struct User {

		NodePath path;
		Ref<Texture> lightmap;
		int instance_index;
	};

	Vector<User> users;

	void _set_user_data(const Array &p_data);
	Array _get_user_data() const;

protected:
	static void _bind_methods();

public:
	void set_bounds(const AABB &p_bounds);
	AABB get_bounds() const;

	void set_octree(const PoolVector<uint8_t> &p_octree);
	PoolVector<uint8_t> get_octree() const;

	void set_cell_space_transform(const Transform &p_xform);
	Transform get_cell_space_transform() const;

	void set_cell_subdiv(int p_cell_subdiv);
	int get_cell_subdiv() const;

	void set_energy(float p_energy);
	float get_energy() const;

	void add_user(const NodePath &p_path, const Ref<Texture> &p_lightmap, int p_instance = -1);
	int get_user_count() const;
	NodePath get_user_path(int p_user) const;
	Ref<Texture> get_user_lightmap(int p_user) const;
	int get_user_instance(int p_user) const;
	void clear_users();

	virtual RID get_rid() const;
	BakedLightmapData();
	~BakedLightmapData();
};

class BakedLightmap : public VisualInstance {
	GDCLASS(BakedLightmap, VisualInstance);

public:
	enum BakeQuality {
		BAKE_QUALITY_LOW,
		BAKE_QUALITY_MEDIUM,
		BAKE_QUALITY_HIGH
	};

	enum BakeMode {
		BAKE_MODE_CONE_TRACE,
		BAKE_MODE_RAY_TRACE,
	};

	enum BakeError {
		BAKE_ERROR_OK,
		BAKE_ERROR_NO_SAVE_PATH,
		BAKE_ERROR_NO_MESHES,
		BAKE_ERROR_CANT_CREATE_IMAGE,
		BAKE_ERROR_USER_ABORTED

	};

	typedef void (*BakeBeginFunc)(int);
	typedef bool (*BakeStepFunc)(int, const String &);
	typedef void (*BakeEndFunc)();

private:
	float bake_cell_size;
	float capture_cell_size;
	Vector3 extents;
	float propagation;
	float energy;
	BakeQuality bake_quality;
	BakeMode bake_mode;
	bool hdr;
	String image_path;

	Ref<BakedLightmapData> light_data;

	struct PlotMesh {
		Ref<Material> override_material;
		Vector<Ref<Material> > instance_materials;
		Ref<Mesh> mesh;
		Transform local_xform;
		NodePath path;
		int instance_idx;
	};

	struct PlotLight {
		Light *light;
		Transform local_xform;
	};

	void _find_meshes_and_lights(Node *p_at_node, List<PlotMesh> &plot_meshes, List<PlotLight> &plot_lights);

	void _debug_bake();

	void _assign_lightmaps();
	void _clear_lightmaps();

	static bool _bake_time(void *ud, float p_secs, float p_progress);

	struct BakeTimeData {
		String text;
		int pass;
		uint64_t last_step;
	};

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	static BakeBeginFunc bake_begin_function;
	static BakeStepFunc bake_step_function;
	static BakeEndFunc bake_end_function;

	void set_light_data(const Ref<BakedLightmapData> &p_data);
	Ref<BakedLightmapData> get_light_data() const;

	void set_bake_cell_size(float p_cell_size);
	float get_bake_cell_size() const;

	void set_capture_cell_size(float p_cell_size);
	float get_capture_cell_size() const;

	void set_extents(const Vector3 &p_extents);
	Vector3 get_extents() const;

	void set_propagation(float p_propagation);
	float get_propagation() const;

	void set_energy(float p_energy);
	float get_energy() const;

	void set_bake_quality(BakeQuality p_quality);
	BakeQuality get_bake_quality() const;

	void set_bake_mode(BakeMode p_mode);
	BakeMode get_bake_mode() const;

	void set_hdr(bool p_enable);
	bool is_hdr() const;

	void set_image_path(const String &p_path);
	String get_image_path() const;

	AABB get_aabb() const;
	PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;

	BakeError bake(Node *p_from_node, bool p_create_visual_debug = false);
	BakedLightmap();
};

VARIANT_ENUM_CAST(BakedLightmap::BakeQuality);
VARIANT_ENUM_CAST(BakedLightmap::BakeMode);
VARIANT_ENUM_CAST(BakedLightmap::BakeError);

#endif // BAKED_INDIRECT_LIGHT_H
