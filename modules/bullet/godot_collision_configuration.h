/*************************************************************************/
/*  godot_collision_configuration.h                                      */
/*  Author: AndreaCatania                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifndef GODOT_COLLISION_CONFIGURATION_H
#define GODOT_COLLISION_CONFIGURATION_H

#include "BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h"
#include "thirdparty/Bullet/src/BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.h"
#include "thirdparty\Bullet/src/BulletCollision/CollisionDispatch/btConvexConcaveCollisionAlgorithm.h"

class GodotRayConvexAlgorithm : public btConvexConvexAlgorithm {

	bool m_isSwapped;

public:
	GodotRayConvexAlgorithm(btPersistentManifold *mf, const btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, btConvexPenetrationDepthSolver *pdSolver, int numPerturbationIterations, int minimumPointsPerturbationThreshold, bool isSwapped);

	virtual void processCollision(const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut);

	struct CreateFunc : public btConvexConvexAlgorithm::CreateFunc {

		CreateFunc(btConvexPenetrationDepthSolver *pdSolver);

		virtual btCollisionAlgorithm *CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap) {
			void *mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btConvexConvexAlgorithm));
			return new (mem) GodotRayConvexAlgorithm(ci.m_manifold, ci, body0Wrap, body1Wrap, m_pdSolver, m_numPerturbationIterations, m_minimumPointsPerturbationThreshold, false);
		}
	};

	struct SwappedCreateFunc : public btConvexConvexAlgorithm::CreateFunc {

		SwappedCreateFunc(btConvexPenetrationDepthSolver *pdSolver);

		virtual btCollisionAlgorithm *CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap) {
			void *mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btConvexConvexAlgorithm));
			return new (mem) GodotRayConvexAlgorithm(ci.m_manifold, ci, body0Wrap, body1Wrap, m_pdSolver, m_numPerturbationIterations, m_minimumPointsPerturbationThreshold, true);
		}
	};
};

class GodotRayConcaveAlgorithm : public btConvexConcaveCollisionAlgorithm {

	bool m_isSwapped;

public:
	GodotRayConcaveAlgorithm(const btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, bool isSwapped);

	virtual void processCollision(const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut);

	struct CreateFunc : public btConvexConcaveCollisionAlgorithm::CreateFunc {

		virtual btCollisionAlgorithm *CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap) {
			void *mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btConvexConvexAlgorithm));
			return new (mem) GodotRayConcaveAlgorithm(ci, body0Wrap, body1Wrap, false);
		}
	};

	struct SwappedCreateFunc : public btConvexConcaveCollisionAlgorithm::CreateFunc {

		virtual btCollisionAlgorithm *CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap) {
			void *mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(btConvexConvexAlgorithm));
			return new (mem) GodotRayConcaveAlgorithm(ci, body0Wrap, body1Wrap, true);
		}
	};
};

class GodotCollisionConfiguration : public btDefaultCollisionConfiguration {
	btCollisionAlgorithmCreateFunc *m_rayConvexCF;
	btCollisionAlgorithmCreateFunc *m_swappedRayConvexCF;
	btCollisionAlgorithmCreateFunc *m_rayConcaveCF;
	btCollisionAlgorithmCreateFunc *m_swappedRayConcaveCF;

public:
	GodotCollisionConfiguration(const btDefaultCollisionConstructionInfo &constructionInfo = btDefaultCollisionConstructionInfo());
	virtual ~GodotCollisionConfiguration();

	virtual btCollisionAlgorithmCreateFunc *getCollisionAlgorithmCreateFunc(int proxyType0, int proxyType1);
	virtual btCollisionAlgorithmCreateFunc *getClosestPointsAlgorithmCreateFunc(int proxyType0, int proxyType1);
};
#endif
