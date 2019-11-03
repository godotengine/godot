/*
Copyright (c) 2013 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans


#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include "Bullet3Dynamics/shared/b3IntegrateTransforms.h"



__kernel void 
  integrateTransformsKernel( __global b3RigidBodyData_t* bodies,const int numNodes, float timeStep, float angularDamping, float4 gravityAcceleration)
{
	int nodeID = get_global_id(0);
	
	if( nodeID < numNodes)
	{
		integrateSingleTransform(bodies,nodeID, timeStep, angularDamping,gravityAcceleration);
	}
}
