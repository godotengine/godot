#ifndef IMMEDIATE_GEOMETRY_H
#define IMMEDIATE_GEOMETRY_H

#include "scene/3d/visual_instance.h"
#include "scene/resources/mesh.h"

class ImmediateGeometry : public GeometryInstance {

	OBJ_TYPE(ImmediateGeometry,GeometryInstance);


	RID im;
	List<Ref<Texture> > cached_textures;
	bool empty;
	AABB aabb;
protected:

	static void _bind_methods();
public:


	void begin(Mesh::PrimitiveType p_primitive,const Ref<Texture>& p_texture);
	void set_normal(const Vector3& p_normal);
	void set_tangent(const Plane& p_tangent);
	void set_color(const Color& p_color);
	void set_uv(const Vector2& tex_uv);
	void set_uv2(const Vector2& tex_uv);

	void add_vertex(const Vector3& p_vertex);

	void end();
	void clear();


	void add_sphere(int p_lats,int p_lons,float p_radius);



	virtual AABB get_aabb() const;
	virtual DVector<Face3> get_faces(uint32_t p_usage_flags) const;

	ImmediateGeometry();
	~ImmediateGeometry();
};

#endif // IMMEDIATE_GEOMETRY_H
