/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_COLLISION_MARGIN_H
#define BT_COLLISION_MARGIN_H

///The CONVEX_DISTANCE_MARGIN is a default collision margin for convex collision shapes derived from btConvexInternalShape.
///This collision margin is used by Gjk and some other algorithms
///Note that when creating small objects, you need to make sure to set a smaller collision margin, using the 'setMargin' API
#define CONVEX_DISTANCE_MARGIN btScalar(0.04)  // btScalar(0.1)//;//btScalar(0.01)

#endif  //BT_COLLISION_MARGIN_H
