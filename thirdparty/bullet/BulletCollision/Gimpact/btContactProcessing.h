#ifndef BT_CONTACT_H_INCLUDED
#define BT_CONTACT_H_INCLUDED

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
#include "btContactProcessingStructs.h"

class btContactArray : public btAlignedObjectArray<GIM_CONTACT>
{
public:
	btContactArray()
	{
		reserve(64);
	}

	SIMD_FORCE_INLINE void push_contact(
		const btVector3 &point, const btVector3 &normal,
		btScalar depth, int feature1, int feature2)
	{
		push_back(GIM_CONTACT(point, normal, depth, feature1, feature2));
	}

	SIMD_FORCE_INLINE void push_triangle_contacts(
		const GIM_TRIANGLE_CONTACT &tricontact,
		int feature1, int feature2)
	{
		for (int i = 0; i < tricontact.m_point_count; i++)
		{
			push_contact(
				tricontact.m_points[i],
				tricontact.m_separating_normal,
				tricontact.m_penetration_depth, feature1, feature2);
		}
	}

	void merge_contacts(const btContactArray &contacts, bool normal_contact_average = true);

	void merge_contacts_unique(const btContactArray &contacts);
};

#endif  // GIM_CONTACT_H_INCLUDED
