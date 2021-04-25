/*
 Written by Xuchen Han <xuchenhan2015@u.northwestern.edu>
 
 Bullet Continuous Collision Detection and Physics Library
 Copyright (c) 2019 Google Inc. http://bulletphysics.org
 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it freely,
 subject to the following restrictions:
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef BT_KRYLOV_SOLVER_H
#define BT_KRYLOV_SOLVER_H
#include <iostream>
#include <cmath>
#include <limits>
#include <LinearMath/btAlignedObjectArray.h>
#include <LinearMath/btVector3.h>
#include <LinearMath/btScalar.h>
#include "LinearMath/btQuickprof.h"

template <class MatrixX>
class btKrylovSolver
{
	typedef btAlignedObjectArray<btVector3> TVStack;

public:
	int m_maxIterations;
	btScalar m_tolerance;
	btKrylovSolver(int maxIterations, btScalar tolerance)
		: m_maxIterations(maxIterations), m_tolerance(tolerance)
	{
	}

	virtual ~btKrylovSolver() {}

	virtual int solve(MatrixX& A, TVStack& x, const TVStack& b, bool verbose = false) = 0;

	virtual void reinitialize(const TVStack& b) = 0;

	virtual SIMD_FORCE_INLINE TVStack sub(const TVStack& a, const TVStack& b)
	{
		// c = a-b
		btAssert(a.size() == b.size());
		TVStack c;
		c.resize(a.size());
		for (int i = 0; i < a.size(); ++i)
		{
			c[i] = a[i] - b[i];
		}
		return c;
	}

	virtual SIMD_FORCE_INLINE btScalar squaredNorm(const TVStack& a)
	{
		return dot(a, a);
	}

	virtual SIMD_FORCE_INLINE btScalar norm(const TVStack& a)
	{
		btScalar ret = 0;
		for (int i = 0; i < a.size(); ++i)
		{
			for (int d = 0; d < 3; ++d)
			{
				ret = btMax(ret, btFabs(a[i][d]));
			}
		}
		return ret;
	}

	virtual SIMD_FORCE_INLINE btScalar dot(const TVStack& a, const TVStack& b)
	{
		btScalar ans(0);
		for (int i = 0; i < a.size(); ++i)
			ans += a[i].dot(b[i]);
		return ans;
	}

	virtual SIMD_FORCE_INLINE void multAndAddTo(btScalar s, const TVStack& a, TVStack& result)
	{
		//        result += s*a
		btAssert(a.size() == result.size());
		for (int i = 0; i < a.size(); ++i)
			result[i] += s * a[i];
	}

	virtual SIMD_FORCE_INLINE TVStack multAndAdd(btScalar s, const TVStack& a, const TVStack& b)
	{
		// result = a*s + b
		TVStack result;
		result.resize(a.size());
		for (int i = 0; i < a.size(); ++i)
			result[i] = s * a[i] + b[i];
		return result;
	}

	virtual SIMD_FORCE_INLINE void setTolerance(btScalar tolerance)
	{
		m_tolerance = tolerance;
	}
};
#endif /* BT_KRYLOV_SOLVER_H */
