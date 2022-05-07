/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "BulletCollision/CollisionShapes/btTriangleIndexVertexArray.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionShapes/btCollisionShape.h"

#include "btDefaultSoftBodySolver.h"
#include "BulletCollision/CollisionShapes/btCapsuleShape.h"
#include "BulletSoftBody/btSoftBody.h"

btDefaultSoftBodySolver::btDefaultSoftBodySolver()
{
	// Initial we will clearly need to update solver constants
	// For now this is global for the cloths linked with this solver - we should probably make this body specific
	// for performance in future once we understand more clearly when constants need to be updated
	m_updateSolverConstants = true;
}

btDefaultSoftBodySolver::~btDefaultSoftBodySolver()
{
}

// In this case the data is already in the soft bodies so there is no need for us to do anything
void btDefaultSoftBodySolver::copyBackToSoftBodies(bool bMove)
{
}

void btDefaultSoftBodySolver::optimize(btAlignedObjectArray<btSoftBody *> &softBodies, bool forceUpdate)
{
	m_softBodySet.copyFromArray(softBodies);
}

void btDefaultSoftBodySolver::updateSoftBodies()
{
	for (int i = 0; i < m_softBodySet.size(); i++)
	{
		btSoftBody *psb = (btSoftBody *)m_softBodySet[i];
		if (psb->isActive())
		{
			psb->integrateMotion();
		}
	}
}  // updateSoftBodies

bool btDefaultSoftBodySolver::checkInitialized()
{
	return true;
}

void btDefaultSoftBodySolver::solveConstraints(btScalar solverdt)
{
	// Solve constraints for non-solver softbodies
	for (int i = 0; i < m_softBodySet.size(); ++i)
	{
		btSoftBody *psb = static_cast<btSoftBody *>(m_softBodySet[i]);
		if (psb->isActive())
		{
			psb->solveConstraints();
		}
	}
}  // btDefaultSoftBodySolver::solveConstraints

void btDefaultSoftBodySolver::copySoftBodyToVertexBuffer(const btSoftBody *const softBody, btVertexBufferDescriptor *vertexBuffer)
{
	// Currently only support CPU output buffers
	// TODO: check for DX11 buffers. Take all offsets into the same DX11 buffer
	// and use them together on a single kernel call if possible by setting up a
	// per-cloth target buffer array for the copy kernel.

	if (vertexBuffer->getBufferType() == btVertexBufferDescriptor::CPU_BUFFER)
	{
		const btAlignedObjectArray<btSoftBody::Node> &clothVertices(softBody->m_nodes);
		int numVertices = clothVertices.size();

		const btCPUVertexBufferDescriptor *cpuVertexBuffer = static_cast<btCPUVertexBufferDescriptor *>(vertexBuffer);
		float *basePointer = cpuVertexBuffer->getBasePointer();

		if (vertexBuffer->hasVertexPositions())
		{
			const int vertexOffset = cpuVertexBuffer->getVertexOffset();
			const int vertexStride = cpuVertexBuffer->getVertexStride();
			float *vertexPointer = basePointer + vertexOffset;

			for (int vertexIndex = 0; vertexIndex < numVertices; ++vertexIndex)
			{
				btVector3 position = clothVertices[vertexIndex].m_x;
				*(vertexPointer + 0) = (float)position.getX();
				*(vertexPointer + 1) = (float)position.getY();
				*(vertexPointer + 2) = (float)position.getZ();
				vertexPointer += vertexStride;
			}
		}
		if (vertexBuffer->hasNormals())
		{
			const int normalOffset = cpuVertexBuffer->getNormalOffset();
			const int normalStride = cpuVertexBuffer->getNormalStride();
			float *normalPointer = basePointer + normalOffset;

			for (int vertexIndex = 0; vertexIndex < numVertices; ++vertexIndex)
			{
				btVector3 normal = clothVertices[vertexIndex].m_n;
				*(normalPointer + 0) = (float)normal.getX();
				*(normalPointer + 1) = (float)normal.getY();
				*(normalPointer + 2) = (float)normal.getZ();
				normalPointer += normalStride;
			}
		}
	}
}  // btDefaultSoftBodySolver::copySoftBodyToVertexBuffer

void btDefaultSoftBodySolver::processCollision(btSoftBody *softBody, btSoftBody *otherSoftBody)
{
	softBody->defaultCollisionHandler(otherSoftBody);
}

// For the default solver just leave the soft body to do its collision processing
void btDefaultSoftBodySolver::processCollision(btSoftBody *softBody, const btCollisionObjectWrapper *collisionObjectWrap)
{
	softBody->defaultCollisionHandler(collisionObjectWrap);
}  // btDefaultSoftBodySolver::processCollision

void btDefaultSoftBodySolver::predictMotion(btScalar timeStep)
{
	for (int i = 0; i < m_softBodySet.size(); ++i)
	{
		btSoftBody *psb = m_softBodySet[i];

		if (psb->isActive())
		{
			psb->predictMotion(timeStep);
		}
	}
}
