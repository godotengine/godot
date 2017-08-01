
#ifndef B3_CONSTRAINT4_h
#define B3_CONSTRAINT4_h
#include "Bullet3Common/b3Vector3.h"

#include "Bullet3Dynamics/shared/b3ContactConstraint4.h"


B3_ATTRIBUTE_ALIGNED16(struct) b3GpuConstraint4 : public b3ContactConstraint4
{
	B3_DECLARE_ALIGNED_ALLOCATOR();

	inline	void setFrictionCoeff(float value) { m_linear[3] = value; }
	inline	float getFrictionCoeff() const { return m_linear[3]; }
};

#endif //B3_CONSTRAINT4_h

