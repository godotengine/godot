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

#ifndef BT_IDEBUG_DRAW__H
#define BT_IDEBUG_DRAW__H

#include "btVector3.h"
#include "btTransform.h"

///The btIDebugDraw interface class allows hooking up a debug renderer to visually debug simulations.
///Typical use case: create a debug drawer object, and assign it to a btCollisionWorld or btDynamicsWorld using setDebugDrawer and call debugDrawWorld.
///A class that implements the btIDebugDraw interface has to implement the drawLine method at a minimum.
///For color arguments the X,Y,Z components refer to Red, Green and Blue each in the range [0..1]
class btIDebugDraw
{
public:
	ATTRIBUTE_ALIGNED16(struct)
	DefaultColors
	{
		btVector3 m_activeObject;
		btVector3 m_deactivatedObject;
		btVector3 m_wantsDeactivationObject;
		btVector3 m_disabledDeactivationObject;
		btVector3 m_disabledSimulationObject;
		btVector3 m_aabb;
		btVector3 m_contactPoint;

		DefaultColors()
			: m_activeObject(1, 1, 1),
			  m_deactivatedObject(0, 1, 0),
			  m_wantsDeactivationObject(0, 1, 1),
			  m_disabledDeactivationObject(1, 0, 0),
			  m_disabledSimulationObject(1, 1, 0),
			  m_aabb(1, 0, 0),
			  m_contactPoint(1, 1, 0)
		{
		}
	};

	enum DebugDrawModes
	{
		DBG_NoDebug = 0,
		DBG_DrawWireframe = 1,
		DBG_DrawAabb = 2,
		DBG_DrawFeaturesText = 4,
		DBG_DrawContactPoints = 8,
		DBG_NoDeactivation = 16,
		DBG_NoHelpText = 32,
		DBG_DrawText = 64,
		DBG_ProfileTimings = 128,
		DBG_EnableSatComparison = 256,
		DBG_DisableBulletLCP = 512,
		DBG_EnableCCD = 1024,
		DBG_DrawConstraints = (1 << 11),
		DBG_DrawConstraintLimits = (1 << 12),
		DBG_FastWireframe = (1 << 13),
		DBG_DrawNormals = (1 << 14),
		DBG_DrawFrames = (1 << 15),
		DBG_MAX_DEBUG_DRAW_MODE
	};

	virtual ~btIDebugDraw(){};

	virtual DefaultColors getDefaultColors() const
	{
		DefaultColors colors;
		return colors;
	}
	///the default implementation for setDefaultColors has no effect. A derived class can implement it and store the colors.
	virtual void setDefaultColors(const DefaultColors& /*colors*/) {}

	virtual void drawLine(const btVector3& from, const btVector3& to, const btVector3& color) = 0;

	virtual void drawLine(const btVector3& from, const btVector3& to, const btVector3& fromColor, const btVector3& toColor)
	{
		(void)toColor;
		drawLine(from, to, fromColor);
	}

	virtual void drawSphere(btScalar radius, const btTransform& transform, const btVector3& color)
	{
		btVector3 center = transform.getOrigin();
		btVector3 up = transform.getBasis().getColumn(1);
		btVector3 axis = transform.getBasis().getColumn(0);
		btScalar minTh = -SIMD_HALF_PI;
		btScalar maxTh = SIMD_HALF_PI;
		btScalar minPs = -SIMD_HALF_PI;
		btScalar maxPs = SIMD_HALF_PI;
		btScalar stepDegrees = 30.f;
		drawSpherePatch(center, up, axis, radius, minTh, maxTh, minPs, maxPs, color, stepDegrees, false);
		drawSpherePatch(center, up, -axis, radius, minTh, maxTh, minPs, maxPs, color, stepDegrees, false);
	}

	virtual void drawSphere(const btVector3& p, btScalar radius, const btVector3& color)
	{
		btTransform tr;
		tr.setIdentity();
		tr.setOrigin(p);
		drawSphere(radius, tr, color);
	}

	virtual void drawTriangle(const btVector3& v0, const btVector3& v1, const btVector3& v2, const btVector3& /*n0*/, const btVector3& /*n1*/, const btVector3& /*n2*/, const btVector3& color, btScalar alpha)
	{
		drawTriangle(v0, v1, v2, color, alpha);
	}
	virtual void drawTriangle(const btVector3& v0, const btVector3& v1, const btVector3& v2, const btVector3& color, btScalar /*alpha*/)
	{
		drawLine(v0, v1, color);
		drawLine(v1, v2, color);
		drawLine(v2, v0, color);
	}

	virtual void drawContactPoint(const btVector3& PointOnB, const btVector3& normalOnB, btScalar distance, int lifeTime, const btVector3& color) = 0;

	virtual void reportErrorWarning(const char* warningString) = 0;

	virtual void draw3dText(const btVector3& location, const char* textString) = 0;

	virtual void setDebugMode(int debugMode) = 0;

	virtual int getDebugMode() const = 0;

	virtual void drawAabb(const btVector3& from, const btVector3& to, const btVector3& color)
	{
		btVector3 halfExtents = (to - from) * 0.5f;
		btVector3 center = (to + from) * 0.5f;
		int i, j;

		btVector3 edgecoord(1.f, 1.f, 1.f), pa, pb;
		for (i = 0; i < 4; i++)
		{
			for (j = 0; j < 3; j++)
			{
				pa = btVector3(edgecoord[0] * halfExtents[0], edgecoord[1] * halfExtents[1],
							   edgecoord[2] * halfExtents[2]);
				pa += center;

				int othercoord = j % 3;
				edgecoord[othercoord] *= -1.f;
				pb = btVector3(edgecoord[0] * halfExtents[0], edgecoord[1] * halfExtents[1],
							   edgecoord[2] * halfExtents[2]);
				pb += center;

				drawLine(pa, pb, color);
			}
			edgecoord = btVector3(-1.f, -1.f, -1.f);
			if (i < 3)
				edgecoord[i] *= -1.f;
		}
	}
	virtual void drawTransform(const btTransform& transform, btScalar orthoLen)
	{
		btVector3 start = transform.getOrigin();
		drawLine(start, start + transform.getBasis() * btVector3(orthoLen, 0, 0), btVector3(btScalar(1.), btScalar(0.3), btScalar(0.3)));
		drawLine(start, start + transform.getBasis() * btVector3(0, orthoLen, 0), btVector3(btScalar(0.3), btScalar(1.), btScalar(0.3)));
		drawLine(start, start + transform.getBasis() * btVector3(0, 0, orthoLen), btVector3(btScalar(0.3), btScalar(0.3), btScalar(1.)));
	}

	virtual void drawArc(const btVector3& center, const btVector3& normal, const btVector3& axis, btScalar radiusA, btScalar radiusB, btScalar minAngle, btScalar maxAngle,
						 const btVector3& color, bool drawSect, btScalar stepDegrees = btScalar(10.f))
	{
		const btVector3& vx = axis;
		btVector3 vy = normal.cross(axis);
		btScalar step = stepDegrees * SIMD_RADS_PER_DEG;
		int nSteps = (int)btFabs((maxAngle - minAngle) / step);
		if (!nSteps) nSteps = 1;
		btVector3 prev = center + radiusA * vx * btCos(minAngle) + radiusB * vy * btSin(minAngle);
		if (drawSect)
		{
			drawLine(center, prev, color);
		}
		for (int i = 1; i <= nSteps; i++)
		{
			btScalar angle = minAngle + (maxAngle - minAngle) * btScalar(i) / btScalar(nSteps);
			btVector3 next = center + radiusA * vx * btCos(angle) + radiusB * vy * btSin(angle);
			drawLine(prev, next, color);
			prev = next;
		}
		if (drawSect)
		{
			drawLine(center, prev, color);
		}
	}
	virtual void drawSpherePatch(const btVector3& center, const btVector3& up, const btVector3& axis, btScalar radius,
								 btScalar minTh, btScalar maxTh, btScalar minPs, btScalar maxPs, const btVector3& color, btScalar stepDegrees = btScalar(10.f), bool drawCenter = true)
	{
		btVector3 vA[74];
		btVector3 vB[74];
		btVector3 *pvA = vA, *pvB = vB, *pT;
		btVector3 npole = center + up * radius;
		btVector3 spole = center - up * radius;
		btVector3 arcStart;
		btScalar step = stepDegrees * SIMD_RADS_PER_DEG;
		const btVector3& kv = up;
		const btVector3& iv = axis;
		btVector3 jv = kv.cross(iv);
		bool drawN = false;
		bool drawS = false;
		if (minTh <= -SIMD_HALF_PI)
		{
			minTh = -SIMD_HALF_PI + step;
			drawN = true;
		}
		if (maxTh >= SIMD_HALF_PI)
		{
			maxTh = SIMD_HALF_PI - step;
			drawS = true;
		}
		if (minTh > maxTh)
		{
			minTh = -SIMD_HALF_PI + step;
			maxTh = SIMD_HALF_PI - step;
			drawN = drawS = true;
		}
		int n_hor = (int)((maxTh - minTh) / step) + 1;
		if (n_hor < 2) n_hor = 2;
		btScalar step_h = (maxTh - minTh) / btScalar(n_hor - 1);
		bool isClosed = false;
		if (minPs > maxPs)
		{
			minPs = -SIMD_PI + step;
			maxPs = SIMD_PI;
			isClosed = true;
		}
		else if ((maxPs - minPs) >= SIMD_PI * btScalar(2.f))
		{
			isClosed = true;
		}
		else
		{
			isClosed = false;
		}
		int n_vert = (int)((maxPs - minPs) / step) + 1;
		if (n_vert < 2) n_vert = 2;
		btScalar step_v = (maxPs - minPs) / btScalar(n_vert - 1);
		for (int i = 0; i < n_hor; i++)
		{
			btScalar th = minTh + btScalar(i) * step_h;
			btScalar sth = radius * btSin(th);
			btScalar cth = radius * btCos(th);
			for (int j = 0; j < n_vert; j++)
			{
				btScalar psi = minPs + btScalar(j) * step_v;
				btScalar sps = btSin(psi);
				btScalar cps = btCos(psi);
				pvB[j] = center + cth * cps * iv + cth * sps * jv + sth * kv;
				if (i)
				{
					drawLine(pvA[j], pvB[j], color);
				}
				else if (drawS)
				{
					drawLine(spole, pvB[j], color);
				}
				if (j)
				{
					drawLine(pvB[j - 1], pvB[j], color);
				}
				else
				{
					arcStart = pvB[j];
				}
				if ((i == (n_hor - 1)) && drawN)
				{
					drawLine(npole, pvB[j], color);
				}

				if (drawCenter)
				{
					if (isClosed)
					{
						if (j == (n_vert - 1))
						{
							drawLine(arcStart, pvB[j], color);
						}
					}
					else
					{
						if (((!i) || (i == (n_hor - 1))) && ((!j) || (j == (n_vert - 1))))
						{
							drawLine(center, pvB[j], color);
						}
					}
				}
			}
			pT = pvA;
			pvA = pvB;
			pvB = pT;
		}
	}

	virtual void drawBox(const btVector3& bbMin, const btVector3& bbMax, const btVector3& color)
	{
		drawLine(btVector3(bbMin[0], bbMin[1], bbMin[2]), btVector3(bbMax[0], bbMin[1], bbMin[2]), color);
		drawLine(btVector3(bbMax[0], bbMin[1], bbMin[2]), btVector3(bbMax[0], bbMax[1], bbMin[2]), color);
		drawLine(btVector3(bbMax[0], bbMax[1], bbMin[2]), btVector3(bbMin[0], bbMax[1], bbMin[2]), color);
		drawLine(btVector3(bbMin[0], bbMax[1], bbMin[2]), btVector3(bbMin[0], bbMin[1], bbMin[2]), color);
		drawLine(btVector3(bbMin[0], bbMin[1], bbMin[2]), btVector3(bbMin[0], bbMin[1], bbMax[2]), color);
		drawLine(btVector3(bbMax[0], bbMin[1], bbMin[2]), btVector3(bbMax[0], bbMin[1], bbMax[2]), color);
		drawLine(btVector3(bbMax[0], bbMax[1], bbMin[2]), btVector3(bbMax[0], bbMax[1], bbMax[2]), color);
		drawLine(btVector3(bbMin[0], bbMax[1], bbMin[2]), btVector3(bbMin[0], bbMax[1], bbMax[2]), color);
		drawLine(btVector3(bbMin[0], bbMin[1], bbMax[2]), btVector3(bbMax[0], bbMin[1], bbMax[2]), color);
		drawLine(btVector3(bbMax[0], bbMin[1], bbMax[2]), btVector3(bbMax[0], bbMax[1], bbMax[2]), color);
		drawLine(btVector3(bbMax[0], bbMax[1], bbMax[2]), btVector3(bbMin[0], bbMax[1], bbMax[2]), color);
		drawLine(btVector3(bbMin[0], bbMax[1], bbMax[2]), btVector3(bbMin[0], bbMin[1], bbMax[2]), color);
	}
	virtual void drawBox(const btVector3& bbMin, const btVector3& bbMax, const btTransform& trans, const btVector3& color)
	{
		drawLine(trans * btVector3(bbMin[0], bbMin[1], bbMin[2]), trans * btVector3(bbMax[0], bbMin[1], bbMin[2]), color);
		drawLine(trans * btVector3(bbMax[0], bbMin[1], bbMin[2]), trans * btVector3(bbMax[0], bbMax[1], bbMin[2]), color);
		drawLine(trans * btVector3(bbMax[0], bbMax[1], bbMin[2]), trans * btVector3(bbMin[0], bbMax[1], bbMin[2]), color);
		drawLine(trans * btVector3(bbMin[0], bbMax[1], bbMin[2]), trans * btVector3(bbMin[0], bbMin[1], bbMin[2]), color);
		drawLine(trans * btVector3(bbMin[0], bbMin[1], bbMin[2]), trans * btVector3(bbMin[0], bbMin[1], bbMax[2]), color);
		drawLine(trans * btVector3(bbMax[0], bbMin[1], bbMin[2]), trans * btVector3(bbMax[0], bbMin[1], bbMax[2]), color);
		drawLine(trans * btVector3(bbMax[0], bbMax[1], bbMin[2]), trans * btVector3(bbMax[0], bbMax[1], bbMax[2]), color);
		drawLine(trans * btVector3(bbMin[0], bbMax[1], bbMin[2]), trans * btVector3(bbMin[0], bbMax[1], bbMax[2]), color);
		drawLine(trans * btVector3(bbMin[0], bbMin[1], bbMax[2]), trans * btVector3(bbMax[0], bbMin[1], bbMax[2]), color);
		drawLine(trans * btVector3(bbMax[0], bbMin[1], bbMax[2]), trans * btVector3(bbMax[0], bbMax[1], bbMax[2]), color);
		drawLine(trans * btVector3(bbMax[0], bbMax[1], bbMax[2]), trans * btVector3(bbMin[0], bbMax[1], bbMax[2]), color);
		drawLine(trans * btVector3(bbMin[0], bbMax[1], bbMax[2]), trans * btVector3(bbMin[0], bbMin[1], bbMax[2]), color);
	}

	virtual void drawCapsule(btScalar radius, btScalar halfHeight, int upAxis, const btTransform& transform, const btVector3& color)
	{
		int stepDegrees = 30;

		btVector3 capStart(0.f, 0.f, 0.f);
		capStart[upAxis] = -halfHeight;

		btVector3 capEnd(0.f, 0.f, 0.f);
		capEnd[upAxis] = halfHeight;

		// Draw the ends
		{
			btTransform childTransform = transform;
			childTransform.getOrigin() = transform * capStart;
			{
				btVector3 center = childTransform.getOrigin();
				btVector3 up = childTransform.getBasis().getColumn((upAxis + 1) % 3);
				btVector3 axis = -childTransform.getBasis().getColumn(upAxis);
				btScalar minTh = -SIMD_HALF_PI;
				btScalar maxTh = SIMD_HALF_PI;
				btScalar minPs = -SIMD_HALF_PI;
				btScalar maxPs = SIMD_HALF_PI;

				drawSpherePatch(center, up, axis, radius, minTh, maxTh, minPs, maxPs, color, btScalar(stepDegrees), false);
			}
		}

		{
			btTransform childTransform = transform;
			childTransform.getOrigin() = transform * capEnd;
			{
				btVector3 center = childTransform.getOrigin();
				btVector3 up = childTransform.getBasis().getColumn((upAxis + 1) % 3);
				btVector3 axis = childTransform.getBasis().getColumn(upAxis);
				btScalar minTh = -SIMD_HALF_PI;
				btScalar maxTh = SIMD_HALF_PI;
				btScalar minPs = -SIMD_HALF_PI;
				btScalar maxPs = SIMD_HALF_PI;
				drawSpherePatch(center, up, axis, radius, minTh, maxTh, minPs, maxPs, color, btScalar(stepDegrees), false);
			}
		}

		// Draw some additional lines
		btVector3 start = transform.getOrigin();

		for (int i = 0; i < 360; i += stepDegrees)
		{
			capEnd[(upAxis + 1) % 3] = capStart[(upAxis + 1) % 3] = btSin(btScalar(i) * SIMD_RADS_PER_DEG) * radius;
			capEnd[(upAxis + 2) % 3] = capStart[(upAxis + 2) % 3] = btCos(btScalar(i) * SIMD_RADS_PER_DEG) * radius;
			drawLine(start + transform.getBasis() * capStart, start + transform.getBasis() * capEnd, color);
		}
	}

	virtual void drawCylinder(btScalar radius, btScalar halfHeight, int upAxis, const btTransform& transform, const btVector3& color)
	{
		btVector3 start = transform.getOrigin();
		btVector3 offsetHeight(0, 0, 0);
		offsetHeight[upAxis] = halfHeight;
		int stepDegrees = 30;
		btVector3 capStart(0.f, 0.f, 0.f);
		capStart[upAxis] = -halfHeight;
		btVector3 capEnd(0.f, 0.f, 0.f);
		capEnd[upAxis] = halfHeight;

		for (int i = 0; i < 360; i += stepDegrees)
		{
			capEnd[(upAxis + 1) % 3] = capStart[(upAxis + 1) % 3] = btSin(btScalar(i) * SIMD_RADS_PER_DEG) * radius;
			capEnd[(upAxis + 2) % 3] = capStart[(upAxis + 2) % 3] = btCos(btScalar(i) * SIMD_RADS_PER_DEG) * radius;
			drawLine(start + transform.getBasis() * capStart, start + transform.getBasis() * capEnd, color);
		}
		// Drawing top and bottom caps of the cylinder
		btVector3 yaxis(0, 0, 0);
		yaxis[upAxis] = btScalar(1.0);
		btVector3 xaxis(0, 0, 0);
		xaxis[(upAxis + 1) % 3] = btScalar(1.0);
		drawArc(start - transform.getBasis() * (offsetHeight), transform.getBasis() * yaxis, transform.getBasis() * xaxis, radius, radius, 0, SIMD_2_PI, color, false, btScalar(10.0));
		drawArc(start + transform.getBasis() * (offsetHeight), transform.getBasis() * yaxis, transform.getBasis() * xaxis, radius, radius, 0, SIMD_2_PI, color, false, btScalar(10.0));
	}

	virtual void drawCone(btScalar radius, btScalar height, int upAxis, const btTransform& transform, const btVector3& color)
	{
		int stepDegrees = 30;
		btVector3 start = transform.getOrigin();

		btVector3 offsetHeight(0, 0, 0);
		btScalar halfHeight = height * btScalar(0.5);
		offsetHeight[upAxis] = halfHeight;
		btVector3 offsetRadius(0, 0, 0);
		offsetRadius[(upAxis + 1) % 3] = radius;
		btVector3 offset2Radius(0, 0, 0);
		offset2Radius[(upAxis + 2) % 3] = radius;

		btVector3 capEnd(0.f, 0.f, 0.f);
		capEnd[upAxis] = -halfHeight;

		for (int i = 0; i < 360; i += stepDegrees)
		{
			capEnd[(upAxis + 1) % 3] = btSin(btScalar(i) * SIMD_RADS_PER_DEG) * radius;
			capEnd[(upAxis + 2) % 3] = btCos(btScalar(i) * SIMD_RADS_PER_DEG) * radius;
			drawLine(start + transform.getBasis() * (offsetHeight), start + transform.getBasis() * capEnd, color);
		}

		drawLine(start + transform.getBasis() * (offsetHeight), start + transform.getBasis() * (-offsetHeight + offsetRadius), color);
		drawLine(start + transform.getBasis() * (offsetHeight), start + transform.getBasis() * (-offsetHeight - offsetRadius), color);
		drawLine(start + transform.getBasis() * (offsetHeight), start + transform.getBasis() * (-offsetHeight + offset2Radius), color);
		drawLine(start + transform.getBasis() * (offsetHeight), start + transform.getBasis() * (-offsetHeight - offset2Radius), color);

		// Drawing the base of the cone
		btVector3 yaxis(0, 0, 0);
		yaxis[upAxis] = btScalar(1.0);
		btVector3 xaxis(0, 0, 0);
		xaxis[(upAxis + 1) % 3] = btScalar(1.0);
		drawArc(start - transform.getBasis() * (offsetHeight), transform.getBasis() * yaxis, transform.getBasis() * xaxis, radius, radius, 0, SIMD_2_PI, color, false, 10.0);
	}

	virtual void drawPlane(const btVector3& planeNormal, btScalar planeConst, const btTransform& transform, const btVector3& color)
	{
		btVector3 planeOrigin = planeNormal * planeConst;
		btVector3 vec0, vec1;
		btPlaneSpace1(planeNormal, vec0, vec1);
		btScalar vecLen = 100.f;
		btVector3 pt0 = planeOrigin + vec0 * vecLen;
		btVector3 pt1 = planeOrigin - vec0 * vecLen;
		btVector3 pt2 = planeOrigin + vec1 * vecLen;
		btVector3 pt3 = planeOrigin - vec1 * vecLen;
		drawLine(transform * pt0, transform * pt1, color);
		drawLine(transform * pt2, transform * pt3, color);
	}

	virtual void clearLines()
	{
	}

	virtual void flushLines()
	{
	}
};

#endif  //BT_IDEBUG_DRAW__H
