#ifndef REFLECTIONPROBE_H
#define REFLECTIONPROBE_H

#include "scene/3d/visual_instance.h"
#include "scene/resources/texture.h"
#include "scene/resources/sky_box.h"
#include "servers/visual_server.h"

class ReflectionProbe : public VisualInstance {
	GDCLASS(ReflectionProbe,VisualInstance);

public:

	enum UpdateMode {
		UPDATE_ONCE,
		UPDATE_ALWAYS,
	};


private:

	RID probe;
	float intensity;
	float max_distance;
	Vector3 extents;
	Vector3 origin_offset;
	bool box_projection;
	bool enable_shadows;
	bool interior;
	Color interior_ambient;
	float interior_ambient_energy;
	float interior_ambient_probe_contribution;

	uint32_t cull_mask;
	UpdateMode update_mode;

protected:

	static void _bind_methods();
	void _validate_property(PropertyInfo& property) const;

public:

	void set_intensity(float p_intensity);
	float get_intensity() const;

	void set_interior_ambient(Color p_ambient);
	Color get_interior_ambient() const;

	void set_interior_ambient_energy(float p_energy);
	float get_interior_ambient_energy() const;

	void set_interior_ambient_probe_contribution(float p_contribution);
	float get_interior_ambient_probe_contribution() const;

	void set_max_distance(float p_distance);
	float get_max_distance() const;

	void set_extents(const Vector3& p_extents);
	Vector3 get_extents() const;

	void set_origin_offset(const Vector3& p_extents);
	Vector3 get_origin_offset() const;

	void set_as_interior(bool p_enable);
	bool is_set_as_interior() const;

	void set_enable_box_projection(bool p_enable);
	bool is_box_projection_enabled() const;

	void set_enable_shadows(bool p_enable);
	bool are_shadows_enabled() const;

	void set_cull_mask(uint32_t p_layers);
	uint32_t get_cull_mask() const;

	void set_update_mode(UpdateMode p_mode);
	UpdateMode get_update_mode() const;

	virtual Rect3 get_aabb() const;
	virtual PoolVector<Face3> get_faces(uint32_t p_usage_flags) const;



	ReflectionProbe();
	~ReflectionProbe();
};


VARIANT_ENUM_CAST( ReflectionProbe::UpdateMode );

#endif // REFLECTIONPROBE_H
