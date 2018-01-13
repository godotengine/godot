/*! \file btGImpactShape.h
\author Francisco Leon Najera
*/
/*
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2007 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com


This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#ifndef GIMPACT_TRIANGLE_SHAPE_EX_H
#define GIMPACT_TRIANGLE_SHAPE_EX_H

#include "BulletCollision/CollisionShapes/btCollisionShape.h"
#include "BulletCollision/CollisionShapes/btTriangleShape.h"
#include "btBoxCollision.h"
#include "btClipPolygon.h"
#include "btGeometryOperations.h"


#define MAX_TRI_CLIPPING 16

//! Structure for collision
struct GIM_TRIANGLE_CONTACT
{
    btScalar m_penetration_depth;
    int m_point_count;
    btVector4 m_separating_normal;
    btVector3 m_points[MAX_TRI_CLIPPING];

	SIMD_FORCE_INLINE void copy_from(const GIM_TRIANGLE_CONTACT& other)
	{
		m_penetration_depth = other.m_penetration_depth;
		m_separating_normal = other.m_separating_normal;
		m_point_count = other.m_point_count;
		int i = m_point_count;
		while(i--)
		{
			m_points[i] = other.m_points[i];
		}
	}

	GIM_TRIANGLE_CONTACT()
	{
	}

	GIM_TRIANGLE_CONTACT(const GIM_TRIANGLE_CONTACT& other)
	{
		copy_from(other);
	}

    //! classify points that are closer
    void merge_points(const btVector4 & plane,
    				btScalar margin, const btVector3 * points, int point_count);

};



class btPrimitiveTriangle
{
public:
	btVector3 m_vertices[3];
	btVector4 m_plane;
	btScalar m_margin;
	btScalar m_dummy;
	btPrimitiveTriangle():m_margin(0.01f)
	{

	}


	SIMD_FORCE_INLINE void buildTriPlane()
	{
		btVector3 normal = (m_vertices[1]-m_vertices[0]).cross(m_vertices[2]-m_vertices[0]);
		normal.normalize();
		m_plane.setValue(normal[0],normal[1],normal[2],m_vertices[0].dot(normal));
	}

	//! Test if triangles could collide
	bool overlap_test_conservative(const btPrimitiveTriangle& other);

	//! Calcs the plane which is paralele to the edge and perpendicular to the triangle plane
	/*!
	\pre this triangle must have its plane calculated.
	*/
	SIMD_FORCE_INLINE void get_edge_plane(int edge_index, btVector4 &plane)  const
    {
		const btVector3 & e0 = m_vertices[edge_index];
		const btVector3 & e1 = m_vertices[(edge_index+1)%3];
		bt_edge_plane(e0,e1,m_plane,plane);
    }

    void applyTransform(const btTransform& t)
	{
		m_vertices[0] = t(m_vertices[0]);
		m_vertices[1] = t(m_vertices[1]);
		m_vertices[2] = t(m_vertices[2]);
	}

	//! Clips the triangle against this
	/*!
	\pre clipped_points must have MAX_TRI_CLIPPING size, and this triangle must have its plane calculated.
	\return the number of clipped points
	*/
    int clip_triangle(btPrimitiveTriangle & other, btVector3 * clipped_points );

	//! Find collision using the clipping method
	/*!
	\pre this triangle and other must have their triangles calculated
	*/
    bool find_triangle_collision_clip_method(btPrimitiveTriangle & other, GIM_TRIANGLE_CONTACT & contacts);
};



//! Helper class for colliding Bullet Triangle Shapes
/*!
This class implements a better getAabb method than the previous btTriangleShape class
*/
class btTriangleShapeEx: public btTriangleShape
{
public:

	btTriangleShapeEx():btTriangleShape(btVector3(0,0,0),btVector3(0,0,0),btVector3(0,0,0))
	{
	}

	btTriangleShapeEx(const btVector3& p0,const btVector3& p1,const btVector3& p2):	btTriangleShape(p0,p1,p2)
	{
	}

	btTriangleShapeEx(const btTriangleShapeEx & other):	btTriangleShape(other.m_vertices1[0],other.m_vertices1[1],other.m_vertices1[2])
	{
	}

	virtual void getAabb(const btTransform& t,btVector3& aabbMin,btVector3& aabbMax)const
	{
		btVector3 tv0 = t(m_vertices1[0]);
		btVector3 tv1 = t(m_vertices1[1]);
		btVector3 tv2 = t(m_vertices1[2]);

		btAABB trianglebox(tv0,tv1,tv2,m_collisionMargin);
		aabbMin = trianglebox.m_min;
		aabbMax = trianglebox.m_max;
	}

	void applyTransform(const btTransform& t)
	{
		m_vertices1[0] = t(m_vertices1[0]);
		m_vertices1[1] = t(m_vertices1[1]);
		m_vertices1[2] = t(m_vertices1[2]);
	}

	SIMD_FORCE_INLINE void buildTriPlane(btVector4 & plane) const
	{
		btVector3 normal = (m_vertices1[1]-m_vertices1[0]).cross(m_vertices1[2]-m_vertices1[0]);
		normal.normalize();
		plane.setValue(normal[0],normal[1],normal[2],m_vertices1[0].dot(normal));
	}

	bool overlap_test_conservative(const btTriangleShapeEx& other);
};


#endif //GIMPACT_TRIANGLE_MESH_SHAPE_H
