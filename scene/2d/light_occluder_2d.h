#ifndef LIGHTOCCLUDER2D_H
#define LIGHTOCCLUDER2D_H

#include "scene/2d/node_2d.h"

class OccluderPolygon2D : public Resource {

	OBJ_TYPE(OccluderPolygon2D,Resource);
public:

	enum CullMode {
		CULL_DISABLED,
		CULL_CLOCKWISE,
		CULL_COUNTER_CLOCKWISE
	};
private:


	RID occ_polygon;
	DVector<Vector2> polygon;
	bool closed;
	CullMode cull;

protected:

	static void _bind_methods();
public:

	void set_polygon(const DVector<Vector2>& p_polygon);
	DVector<Vector2> get_polygon() const;

	void set_closed(bool p_closed);
	bool is_closed() const;

	void set_cull_mode(CullMode p_mode);
	CullMode get_cull_mode() const;

	virtual RID get_rid() const;
	OccluderPolygon2D();
	~OccluderPolygon2D();

};

VARIANT_ENUM_CAST(OccluderPolygon2D::CullMode);

class LightOccluder2D : public Node2D {
	OBJ_TYPE(LightOccluder2D,Node2D);

	RID occluder;
	bool enabled;
	int mask;
	Ref<OccluderPolygon2D> occluder_polygon;

#ifdef DEBUG_ENABLED
	void _poly_changed();
#endif

protected:
	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_occluder_polygon(const Ref<OccluderPolygon2D>& p_polygon);
	Ref<OccluderPolygon2D> get_occluder_polygon() const;

	void set_occluder_light_mask(int p_mask);
	int get_occluder_light_mask() const;

	String get_configuration_warning() const;

	LightOccluder2D();
	~LightOccluder2D();
};

#endif // LIGHTOCCLUDER2D_H
