#ifndef BT_CONTACT_H_STRUCTS_INCLUDED
#define BT_CONTACT_H_STRUCTS_INCLUDED

/*! \file gim_contact.h
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

#include "LinearMath/btTransform.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "btTriangleShapeEx.h"


/**
Configuration var for applying interpolation of  contact normals
*/
#define NORMAL_CONTACT_AVERAGE 1

#define CONTACT_DIFF_EPSILON 0.00001f

///The GIM_CONTACT is an internal GIMPACT structure, similar to btManifoldPoint.
///@todo: remove and replace GIM_CONTACT by btManifoldPoint.
class GIM_CONTACT
{
public:
    btVector3 m_point;
    btVector3 m_normal;
    btScalar m_depth;//Positive value indicates interpenetration
    btScalar m_distance;//Padding not for use
    int m_feature1;//Face number
    int m_feature2;//Face number
public:
    GIM_CONTACT()
    {
    }

    GIM_CONTACT(const GIM_CONTACT & contact):
				m_point(contact.m_point),
				m_normal(contact.m_normal),
				m_depth(contact.m_depth),
				m_feature1(contact.m_feature1),
				m_feature2(contact.m_feature2)
    {
    }

    GIM_CONTACT(const btVector3 &point,const btVector3 & normal,
    	 			btScalar depth, int feature1, int feature2):
				m_point(point),
				m_normal(normal),
				m_depth(depth),
				m_feature1(feature1),
				m_feature2(feature2)
    {
    }

	//! Calcs key for coord classification
    SIMD_FORCE_INLINE unsigned int calc_key_contact() const
    {
    	int _coords[] = {
    		(int)(m_point[0]*1000.0f+1.0f),
    		(int)(m_point[1]*1333.0f),
    		(int)(m_point[2]*2133.0f+3.0f)};
		unsigned int _hash=0;
		unsigned int *_uitmp = (unsigned int *)(&_coords[0]);
		_hash = *_uitmp;
		_uitmp++;
		_hash += (*_uitmp)<<4;
		_uitmp++;
		_hash += (*_uitmp)<<8;
		return _hash;
    }

    SIMD_FORCE_INLINE void interpolate_normals( btVector3 * normals,int normal_count)
    {
    	btVector3 vec_sum(m_normal);
		for(int i=0;i<normal_count;i++)
		{
			vec_sum += normals[i];
		}

		btScalar vec_sum_len = vec_sum.length2();
		if(vec_sum_len <CONTACT_DIFF_EPSILON) return;

		//GIM_INV_SQRT(vec_sum_len,vec_sum_len); // 1/sqrt(vec_sum_len)

		m_normal = vec_sum/btSqrt(vec_sum_len);
    }

};

#endif // BT_CONTACT_H_STRUCTS_INCLUDED
