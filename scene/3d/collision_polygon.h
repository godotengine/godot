#ifndef COLLISION_POLYGON_H
#define COLLISION_POLYGON_H

#include "scene/3d/spatial.h"
#include "scene/resources/shape.h"



class CollisionPolygon : public Spatial {

	OBJ_TYPE(CollisionPolygon,Spatial);
public:

	enum BuildMode {
		BUILD_SOLIDS,
		BUILD_TRIANGLES,
	};

protected:


	float depth;
	AABB aabb;
	BuildMode build_mode;
	Vector<Point2> polygon;


	void _add_to_collision_object(Object *p_obj);
	void _update_parent();

	bool can_update_body;
	int shape_from;
	int shape_to;

	void _set_shape_range(const Vector2& p_range);
	Vector2 _get_shape_range() const;

protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_build_mode(BuildMode p_mode);
	BuildMode get_build_mode() const;

	void set_depth(float p_depth);
	float get_depth() const;

	void set_polygon(const Vector<Point2>& p_polygon);
	Vector<Point2> get_polygon() const;

	virtual AABB get_item_rect() const;

	int get_collision_object_first_shape() const { return shape_from; }
	int get_collision_object_last_shape() const { return shape_to; }

	CollisionPolygon();
};

VARIANT_ENUM_CAST( CollisionPolygon::BuildMode );
#endif // COLLISION_POLYGON_H
