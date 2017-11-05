/*************************************************************************/
/*  godot_ray_world_algorithm.h                                          */
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

#ifndef GODOT_RAY_WORLD_ALGORITHM_H
#define GODOT_RAY_WORLD_ALGORITHM_H

#include "BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.h"
#include "BulletCollision/CollisionDispatch/btCollisionCreateFunc.h"
#include "BulletCollision/CollisionDispatch/btCollisionDispatcher.h"

class btDiscreteDynamicsWorld;

class GodotRayWorldAlgorithm : public btActivatingCollisionAlgorithm {

	const btDiscreteDynamicsWorld *m_world;
	btPersistentManifold *m_manifoldPtr;
	bool m_ownManifold;
	bool m_isSwapped;

public:
	GodotRayWorldAlgorithm(const btDiscreteDynamicsWorld *m_world, btPersistentManifold *mf, const btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, bool isSwapped);
	virtual ~GodotRayWorldAlgorithm();

	virtual void processCollision(const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut);
	virtual btScalar calculateTimeOfImpact(btCollisionObject *body0, btCollisionObject *body1, const btDispatcherInfo &dispatchInfo, btManifoldResult *resultOut);

	virtual void getAllContactManifolds(btManifoldArray &manifoldArray) {
		///should we use m_ownManifold to avoid adding duplicates?
		if (m_manifoldPtr && m_ownManifold)
			manifoldArray.push_back(m_manifoldPtr);
	}
	struct CreateFunc : public btCollisionAlgorithmCreateFunc {

		const btDiscreteDynamicsWorld *m_world;
		CreateFunc(const btDiscreteDynamicsWorld *world);

		virtual btCollisionAlgorithm *CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap) {
			void *mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(GodotRayWorldAlgorithm));
			return new (mem) GodotRayWorldAlgorithm(m_world, ci.m_manifold, ci, body0Wrap, body1Wrap, false);
		}
	};

	struct SwappedCreateFunc : public btCollisionAlgorithmCreateFunc {

		const btDiscreteDynamicsWorld *m_world;
		SwappedCreateFunc(const btDiscreteDynamicsWorld *world);

		virtual btCollisionAlgorithm *CreateCollisionAlgorithm(btCollisionAlgorithmConstructionInfo &ci, const btCollisionObjectWrapper *body0Wrap, const btCollisionObjectWrapper *body1Wrap) {
			void *mem = ci.m_dispatcher1->allocateCollisionAlgorithm(sizeof(GodotRayWorldAlgorithm));
			return new (mem) GodotRayWorldAlgorithm(m_world, ci.m_manifold, ci, body0Wrap, body1Wrap, true);
		}
	};
};

#endif // GODOT_RAY_WORLD_ALGORITHM_H
