/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_FEATHERSTONE_LINK_COLLIDER_H
#define BT_FEATHERSTONE_LINK_COLLIDER_H

#include "BulletCollision/CollisionDispatch/btCollisionObject.h"

#include "btMultiBody.h"
#include "LinearMath/btSerializer.h"

#ifdef BT_USE_DOUBLE_PRECISION
#define btMultiBodyLinkColliderData btMultiBodyLinkColliderDoubleData
#define btMultiBodyLinkColliderDataName "btMultiBodyLinkColliderDoubleData"
#else
#define btMultiBodyLinkColliderData btMultiBodyLinkColliderFloatData
#define btMultiBodyLinkColliderDataName "btMultiBodyLinkColliderFloatData"
#endif

class btMultiBodyLinkCollider : public btCollisionObject
{
	//protected:
public:
	btMultiBody* m_multiBody;
	int m_link;

	virtual ~btMultiBodyLinkCollider()
	{

	}
	btMultiBodyLinkCollider(btMultiBody* multiBody, int link)
		: m_multiBody(multiBody),
		  m_link(link)
	{
		m_checkCollideWith = true;
		//we need to remove the 'CF_STATIC_OBJECT' flag, otherwise links/base doesn't merge islands
		//this means that some constraints might point to bodies that are not in the islands, causing crashes
		//if (link>=0 || (multiBody && !multiBody->hasFixedBase()))
		{
			m_collisionFlags &= (~btCollisionObject::CF_STATIC_OBJECT);
		}
		// else
		//{
		//	m_collisionFlags |= (btCollisionObject::CF_STATIC_OBJECT);
		//}

		m_internalType = CO_FEATHERSTONE_LINK;
	}
	static btMultiBodyLinkCollider* upcast(btCollisionObject* colObj)
	{
		if (colObj->getInternalType() & btCollisionObject::CO_FEATHERSTONE_LINK)
			return (btMultiBodyLinkCollider*)colObj;
		return 0;
	}
	static const btMultiBodyLinkCollider* upcast(const btCollisionObject* colObj)
	{
		if (colObj->getInternalType() & btCollisionObject::CO_FEATHERSTONE_LINK)
			return (btMultiBodyLinkCollider*)colObj;
		return 0;
	}

	virtual bool checkCollideWithOverride(const btCollisionObject* co) const
	{
		const btMultiBodyLinkCollider* other = btMultiBodyLinkCollider::upcast(co);
		if (!other)
			return true;
		if (other->m_multiBody != this->m_multiBody)
			return true;
		if (!m_multiBody->hasSelfCollision())
			return false;

		//check if 'link' has collision disabled
		if (m_link >= 0)
		{
			const btMultibodyLink& link = m_multiBody->getLink(this->m_link);
			if (link.m_flags & BT_MULTIBODYLINKFLAGS_DISABLE_ALL_PARENT_COLLISION)
			{
				int parent_of_this = m_link;
				while (1)
				{
					if (parent_of_this == -1)
						break;
					parent_of_this = m_multiBody->getLink(parent_of_this).m_parent;
					if (parent_of_this == other->m_link)
					{
						return false;
					}
				}
			}
			else if (link.m_flags & BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION)
			{
				if (link.m_parent == other->m_link)
					return false;
			}
		}

		if (other->m_link >= 0)
		{
			const btMultibodyLink& otherLink = other->m_multiBody->getLink(other->m_link);
			if (otherLink.m_flags & BT_MULTIBODYLINKFLAGS_DISABLE_ALL_PARENT_COLLISION)
			{
				int parent_of_other = other->m_link;
				while (1)
				{
					if (parent_of_other == -1)
						break;
					parent_of_other = m_multiBody->getLink(parent_of_other).m_parent;
					if (parent_of_other == this->m_link)
						return false;
				}
			}
			else if (otherLink.m_flags & BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION)
			{
				if (otherLink.m_parent == this->m_link)
					return false;
			}
		}
		return true;
	}

	virtual int calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, class btSerializer* serializer) const;
};

// clang-format off

struct	btMultiBodyLinkColliderFloatData
{
	btCollisionObjectFloatData m_colObjData;
	btMultiBodyFloatData	*m_multiBody;
	int			m_link;
	char		m_padding[4];
};

struct	btMultiBodyLinkColliderDoubleData
{
	btCollisionObjectDoubleData m_colObjData;
	btMultiBodyDoubleData		*m_multiBody;
	int			m_link;
	char		m_padding[4];
};

// clang-format on

SIMD_FORCE_INLINE int btMultiBodyLinkCollider::calculateSerializeBufferSize() const
{
	return sizeof(btMultiBodyLinkColliderData);
}

SIMD_FORCE_INLINE const char* btMultiBodyLinkCollider::serialize(void* dataBuffer, class btSerializer* serializer) const
{
	btMultiBodyLinkColliderData* dataOut = (btMultiBodyLinkColliderData*)dataBuffer;
	btCollisionObject::serialize(&dataOut->m_colObjData, serializer);

	dataOut->m_link = this->m_link;
	dataOut->m_multiBody = (btMultiBodyData*)serializer->getUniquePointer(m_multiBody);

	// Fill padding with zeros to appease msan.
	memset(dataOut->m_padding, 0, sizeof(dataOut->m_padding));

	return btMultiBodyLinkColliderDataName;
}

#endif  //BT_FEATHERSTONE_LINK_COLLIDER_H
