#ifndef B3_CONTACT4DATA_H
#define B3_CONTACT4DATA_H

#include "Bullet3Common/shared/b3Float4.h"

typedef  struct b3Contact4Data b3Contact4Data_t;

struct b3Contact4Data
{
	b3Float4	m_worldPosB[4];
//	b3Float4	m_localPosA[4];
//	b3Float4	m_localPosB[4];
	b3Float4	m_worldNormalOnB;	//	w: m_nPoints
	unsigned short  m_restituitionCoeffCmp;
	unsigned short  m_frictionCoeffCmp;
	int m_batchIdx;
	int m_bodyAPtrAndSignBit;//x:m_bodyAPtr, y:m_bodyBPtr
	int m_bodyBPtrAndSignBit;

	int	m_childIndexA;
	int	m_childIndexB;
	int m_unused1;
	int m_unused2;


};

inline int b3Contact4Data_getNumPoints(const struct b3Contact4Data* contact)
{
	return (int)contact->m_worldNormalOnB.w;
};

inline void b3Contact4Data_setNumPoints(struct b3Contact4Data* contact, int numPoints)
{
	contact->m_worldNormalOnB.w = (float)numPoints;
};



#endif //B3_CONTACT4DATA_H