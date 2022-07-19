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

#include "b3StridingMeshInterface.h"

b3StridingMeshInterface::~b3StridingMeshInterface()
{
}

void b3StridingMeshInterface::InternalProcessAllTriangles(b3InternalTriangleIndexCallback* callback, const b3Vector3& aabbMin, const b3Vector3& aabbMax) const
{
	(void)aabbMin;
	(void)aabbMax;
	int numtotalphysicsverts = 0;
	int part, graphicssubparts = getNumSubParts();
	const unsigned char* vertexbase;
	const unsigned char* indexbase;
	int indexstride;
	PHY_ScalarType type;
	PHY_ScalarType gfxindextype;
	int stride, numverts, numtriangles;
	int gfxindex;
	b3Vector3 triangle[3];

	b3Vector3 meshScaling = getScaling();

	///if the number of parts is big, the performance might drop due to the innerloop switch on indextype
	for (part = 0; part < graphicssubparts; part++)
	{
		getLockedReadOnlyVertexIndexBase(&vertexbase, numverts, type, stride, &indexbase, indexstride, numtriangles, gfxindextype, part);
		numtotalphysicsverts += numtriangles * 3;  //upper bound

		///unlike that developers want to pass in double-precision meshes in single-precision Bullet build
		///so disable this feature by default
		///see patch http://code.google.com/p/bullet/issues/detail?id=213

		switch (type)
		{
			case PHY_FLOAT:
			{
				float* graphicsbase;

				switch (gfxindextype)
				{
					case PHY_INTEGER:
					{
						for (gfxindex = 0; gfxindex < numtriangles; gfxindex++)
						{
							unsigned int* tri_indices = (unsigned int*)(indexbase + gfxindex * indexstride);
							graphicsbase = (float*)(vertexbase + tri_indices[0] * stride);
							triangle[0].setValue(graphicsbase[0] * meshScaling.getX(), graphicsbase[1] * meshScaling.getY(), graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (float*)(vertexbase + tri_indices[1] * stride);
							triangle[1].setValue(graphicsbase[0] * meshScaling.getX(), graphicsbase[1] * meshScaling.getY(), graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (float*)(vertexbase + tri_indices[2] * stride);
							triangle[2].setValue(graphicsbase[0] * meshScaling.getX(), graphicsbase[1] * meshScaling.getY(), graphicsbase[2] * meshScaling.getZ());
							callback->internalProcessTriangleIndex(triangle, part, gfxindex);
						}
						break;
					}
					case PHY_SHORT:
					{
						for (gfxindex = 0; gfxindex < numtriangles; gfxindex++)
						{
							unsigned short int* tri_indices = (unsigned short int*)(indexbase + gfxindex * indexstride);
							graphicsbase = (float*)(vertexbase + tri_indices[0] * stride);
							triangle[0].setValue(graphicsbase[0] * meshScaling.getX(), graphicsbase[1] * meshScaling.getY(), graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (float*)(vertexbase + tri_indices[1] * stride);
							triangle[1].setValue(graphicsbase[0] * meshScaling.getX(), graphicsbase[1] * meshScaling.getY(), graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (float*)(vertexbase + tri_indices[2] * stride);
							triangle[2].setValue(graphicsbase[0] * meshScaling.getX(), graphicsbase[1] * meshScaling.getY(), graphicsbase[2] * meshScaling.getZ());
							callback->internalProcessTriangleIndex(triangle, part, gfxindex);
						}
						break;
					}
					case PHY_UCHAR:
					{
						for (gfxindex = 0; gfxindex < numtriangles; gfxindex++)
						{
							unsigned char* tri_indices = (unsigned char*)(indexbase + gfxindex * indexstride);
							graphicsbase = (float*)(vertexbase + tri_indices[0] * stride);
							triangle[0].setValue(graphicsbase[0] * meshScaling.getX(), graphicsbase[1] * meshScaling.getY(), graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (float*)(vertexbase + tri_indices[1] * stride);
							triangle[1].setValue(graphicsbase[0] * meshScaling.getX(), graphicsbase[1] * meshScaling.getY(), graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (float*)(vertexbase + tri_indices[2] * stride);
							triangle[2].setValue(graphicsbase[0] * meshScaling.getX(), graphicsbase[1] * meshScaling.getY(), graphicsbase[2] * meshScaling.getZ());
							callback->internalProcessTriangleIndex(triangle, part, gfxindex);
						}
						break;
					}
					default:
						b3Assert((gfxindextype == PHY_INTEGER) || (gfxindextype == PHY_SHORT));
				}
				break;
			}

			case PHY_DOUBLE:
			{
				double* graphicsbase;

				switch (gfxindextype)
				{
					case PHY_INTEGER:
					{
						for (gfxindex = 0; gfxindex < numtriangles; gfxindex++)
						{
							unsigned int* tri_indices = (unsigned int*)(indexbase + gfxindex * indexstride);
							graphicsbase = (double*)(vertexbase + tri_indices[0] * stride);
							triangle[0].setValue((b3Scalar)graphicsbase[0] * meshScaling.getX(), (b3Scalar)graphicsbase[1] * meshScaling.getY(), (b3Scalar)graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (double*)(vertexbase + tri_indices[1] * stride);
							triangle[1].setValue((b3Scalar)graphicsbase[0] * meshScaling.getX(), (b3Scalar)graphicsbase[1] * meshScaling.getY(), (b3Scalar)graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (double*)(vertexbase + tri_indices[2] * stride);
							triangle[2].setValue((b3Scalar)graphicsbase[0] * meshScaling.getX(), (b3Scalar)graphicsbase[1] * meshScaling.getY(), (b3Scalar)graphicsbase[2] * meshScaling.getZ());
							callback->internalProcessTriangleIndex(triangle, part, gfxindex);
						}
						break;
					}
					case PHY_SHORT:
					{
						for (gfxindex = 0; gfxindex < numtriangles; gfxindex++)
						{
							unsigned short int* tri_indices = (unsigned short int*)(indexbase + gfxindex * indexstride);
							graphicsbase = (double*)(vertexbase + tri_indices[0] * stride);
							triangle[0].setValue((b3Scalar)graphicsbase[0] * meshScaling.getX(), (b3Scalar)graphicsbase[1] * meshScaling.getY(), (b3Scalar)graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (double*)(vertexbase + tri_indices[1] * stride);
							triangle[1].setValue((b3Scalar)graphicsbase[0] * meshScaling.getX(), (b3Scalar)graphicsbase[1] * meshScaling.getY(), (b3Scalar)graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (double*)(vertexbase + tri_indices[2] * stride);
							triangle[2].setValue((b3Scalar)graphicsbase[0] * meshScaling.getX(), (b3Scalar)graphicsbase[1] * meshScaling.getY(), (b3Scalar)graphicsbase[2] * meshScaling.getZ());
							callback->internalProcessTriangleIndex(triangle, part, gfxindex);
						}
						break;
					}
					case PHY_UCHAR:
					{
						for (gfxindex = 0; gfxindex < numtriangles; gfxindex++)
						{
							unsigned char* tri_indices = (unsigned char*)(indexbase + gfxindex * indexstride);
							graphicsbase = (double*)(vertexbase + tri_indices[0] * stride);
							triangle[0].setValue((b3Scalar)graphicsbase[0] * meshScaling.getX(), (b3Scalar)graphicsbase[1] * meshScaling.getY(), (b3Scalar)graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (double*)(vertexbase + tri_indices[1] * stride);
							triangle[1].setValue((b3Scalar)graphicsbase[0] * meshScaling.getX(), (b3Scalar)graphicsbase[1] * meshScaling.getY(), (b3Scalar)graphicsbase[2] * meshScaling.getZ());
							graphicsbase = (double*)(vertexbase + tri_indices[2] * stride);
							triangle[2].setValue((b3Scalar)graphicsbase[0] * meshScaling.getX(), (b3Scalar)graphicsbase[1] * meshScaling.getY(), (b3Scalar)graphicsbase[2] * meshScaling.getZ());
							callback->internalProcessTriangleIndex(triangle, part, gfxindex);
						}
						break;
					}
					default:
						b3Assert((gfxindextype == PHY_INTEGER) || (gfxindextype == PHY_SHORT));
				}
				break;
			}
			default:
				b3Assert((type == PHY_FLOAT) || (type == PHY_DOUBLE));
		}

		unLockReadOnlyVertexBase(part);
	}
}

void b3StridingMeshInterface::calculateAabbBruteForce(b3Vector3& aabbMin, b3Vector3& aabbMax)
{
	struct AabbCalculationCallback : public b3InternalTriangleIndexCallback
	{
		b3Vector3 m_aabbMin;
		b3Vector3 m_aabbMax;

		AabbCalculationCallback()
		{
			m_aabbMin.setValue(b3Scalar(B3_LARGE_FLOAT), b3Scalar(B3_LARGE_FLOAT), b3Scalar(B3_LARGE_FLOAT));
			m_aabbMax.setValue(b3Scalar(-B3_LARGE_FLOAT), b3Scalar(-B3_LARGE_FLOAT), b3Scalar(-B3_LARGE_FLOAT));
		}

		virtual void internalProcessTriangleIndex(b3Vector3* triangle, int partId, int triangleIndex)
		{
			(void)partId;
			(void)triangleIndex;

			m_aabbMin.setMin(triangle[0]);
			m_aabbMax.setMax(triangle[0]);
			m_aabbMin.setMin(triangle[1]);
			m_aabbMax.setMax(triangle[1]);
			m_aabbMin.setMin(triangle[2]);
			m_aabbMax.setMax(triangle[2]);
		}
	};

	//first calculate the total aabb for all triangles
	AabbCalculationCallback aabbCallback;
	aabbMin.setValue(b3Scalar(-B3_LARGE_FLOAT), b3Scalar(-B3_LARGE_FLOAT), b3Scalar(-B3_LARGE_FLOAT));
	aabbMax.setValue(b3Scalar(B3_LARGE_FLOAT), b3Scalar(B3_LARGE_FLOAT), b3Scalar(B3_LARGE_FLOAT));
	InternalProcessAllTriangles(&aabbCallback, aabbMin, aabbMax);

	aabbMin = aabbCallback.m_aabbMin;
	aabbMax = aabbCallback.m_aabbMax;
}
