#ifndef B3_SAP_AABB_H
#define B3_SAP_AABB_H

#include "Bullet3Common/b3Scalar.h"
#include "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h"

///just make sure that the b3Aabb is 16-byte aligned
B3_ATTRIBUTE_ALIGNED16(struct) b3SapAabb : public b3Aabb
{

};


#endif //B3_SAP_AABB_H
