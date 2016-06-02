#ifndef NAVIGATION_POLYGON_H
#define NAVIGATION_POLYGON_H

#include "scene/2d/node_2d.h"


class NavigationPolygon : public Resource  {

	OBJ_TYPE( NavigationPolygon, Resource );

	DVector<Vector2> vertices;
	struct Polygon {
		Vector<int> indices;
	};
	Vector<Polygon> polygons;
	Vector< DVector<Vector2> > outlines;

protected:

	static void _bind_methods();

	void _set_polygons(const Array& p_array);
	Array _get_polygons() const;

	void _set_outlines(const Array& p_array);
	Array _get_outlines() const;

public:



	void set_vertices(const DVector<Vector2>& p_vertices);
	DVector<Vector2> get_vertices() const;

	void add_polygon(const Vector<int>& p_polygon);
	int get_polygon_count() const;

	void add_outline(const DVector<Vector2>& p_outline);
	void add_outline_at_index(const DVector<Vector2>& p_outline,int p_index);
	void set_outline(int p_idx,const DVector<Vector2>& p_outline);
	DVector<Vector2> get_outline(int p_idx) const;
	void remove_outline(int p_idx);
	int get_outline_count() const;

	void clear_outlines();
	void make_polygons_from_outlines();

	Vector<int> get_polygon(int p_idx);
	void clear_polygons();

	NavigationPolygon();
};


class Navigation2D;

class NavigationPolygonInstance : public Node2D {

	OBJ_TYPE(NavigationPolygonInstance,Node2D);

	bool enabled;
	int nav_id;
	Navigation2D *navigation;
	Ref<NavigationPolygon> navpoly;

	void _navpoly_changed();

protected:

	void _notification(int p_what);
	static void _bind_methods();
public:

	void set_enabled(bool p_enabled);
	bool is_enabled() const;

	void set_navigation_polygon(const Ref<NavigationPolygon>& p_navpoly);
	Ref<NavigationPolygon> get_navigation_polygon() const;

	String get_configuration_warning() const;

	NavigationPolygonInstance();
};


#endif // NAVIGATIONPOLYGON_H
