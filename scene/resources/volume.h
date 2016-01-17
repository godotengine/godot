/*************************************************************************/
/*  volume.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef VOLUME_H
#define VOLUME_H

#include "resource.h"

#if 0
#include "servers/physics_server.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Volume : public Resource {

	OBJ_TYPE( Volume, Resource );	
	RID volume;
	
protected:
	
	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;
	
	static void _bind_methods();	
public:

	enum ShapeType {
		SHAPE_PLANE = PhysicsServer::SHAPE_PLANE, ///< plane:"plane"
		SHAPE_SPHERE = PhysicsServer::SHAPE_SPHERE, ///< float:"radius"
		SHAPE_BOX = PhysicsServer::SHAPE_BOX, ///< vec3:"extents"
		SHAPE_CYLINDER = PhysicsServer::SHAPE_CYLINDER, ///< dict(float:"radius", float:"height"):cylinder
		SHAPE_CAPSULE = PhysicsServer::SHAPE_CAPSULE, ///< dict(float:"radius", float:"height"):capsule
		SHAPE_CONVEX_POLYGON = PhysicsServer::SHAPE_CONVEX_POLYGON, ///< array of planes:"planes"
		SHAPE_CONCAVE_POLYGON = PhysicsServer::SHAPE_CONCAVE_POLYGON, ///< vector3 array:"triangles"
	};
	
	void add_shape(ShapeType p_shape_type, const Variant& p_data, const Transform& p_transform=Transform ());
	
	void add_plane_shape(const Plane& p_plane,const Transform& p_transform);
	void add_sphere_shape(float p_radius,const Transform& p_transform);
	void add_box_shape(const Vector3& p_half_extents,const Transform& p_transform);
	void add_cylinder_shape(float p_radius, float p_height,const Transform& p_transform);
	void add_capsule_shape(float p_radius, float p_height,const Transform& p_transform);
	
	int get_shape_count() const;
	ShapeType get_shape_type(int p_shape) const;
	Transform get_shape_transform(int p_shape) const;
	Variant get_shape(int p_shape) const;

	virtual RID get_rid();
	
	Volume();
	~Volume();

};

VARIANT_ENUM_CAST( Volume::ShapeType );

#endif
#endif
