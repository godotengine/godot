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

#ifndef BT_CONJUGATE_GRADIENT_H
#define BT_CONJUGATE_GRADIENT_H
#include "btKrylovSolver.h"
template <class MatrixX>
class btConjugateGradient : public btKrylovSolver<MatrixX>
{
	typedef btAlignedObjectArray<btVector3> TVStack;
	typedef btKrylovSolver<MatrixX> Base;
	TVStack r, p, z, temp;

public:
	btConjugateGradient(const int max_it_in)
		: btKrylovSolver<MatrixX>(max_it_in, SIMD_EPSILON)
	{
	}

	virtual ~btConjugateGradient() {}

	// return the number of iterations taken
	int solve(MatrixX& A, TVStack& x, const TVStack& b, bool verbose = false)
	{
		BT_PROFILE("CGSolve");
		btAssert(x.size() == b.size());
		reinitialize(b);
		temp = b;
		A.project(temp);
		p = temp;
		A.precondition(p, z);
		btScalar d0 = this->dot(z, temp);
		d0 = btMin(btScalar(1), d0);
		// r = b - A * x --with assigned dof zeroed out
		A.multiply(x, temp);
		r = this->sub(b, temp);
		A.project(r);
		// z = M^(-1) * r
		A.precondition(r, z);
		A.project(z);
		btScalar r_dot_z = this->dot(z, r);
		if (r_dot_z <= Base::m_tolerance * d0)
		{
			if (verbose)
			{
				std::cout << "Iteration = 0" << std::endl;
				std::cout << "Two norm of the residual = " << r_dot_z << std::endl;
			}
			return 0;
		}
		p = z;
		btScalar r_dot_z_new = r_dot_z;
		for (int k = 1; k <= Base::m_maxIterations; k++)
		{
			// temp = A*p
			A.multiply(p, temp);
			A.project(temp);
			if (this->dot(p, temp) < 0)
			{
				if (verbose)
					std::cout << "Encountered negative direction in CG!" << std::endl;
				if (k == 1)
				{
					x = b;
				}
				return k;
			}
			// alpha = r^T * z / (p^T * A * p)
			btScalar alpha = r_dot_z_new / this->dot(p, temp);
			//  x += alpha * p;
			this->multAndAddTo(alpha, p, x);
			//  r -= alpha * temp;
			this->multAndAddTo(-alpha, temp, r);
			// z = M^(-1) * r
			A.precondition(r, z);
			r_dot_z = r_dot_z_new;
			r_dot_z_new = this->dot(r, z);
			if (r_dot_z_new < Base::m_tolerance * d0)
			{
				if (verbose)
				{
					std::cout << "ConjugateGradient iterations " << k << " residual = " << r_dot_z_new << std::endl;
				}
				return k;
			}

			btScalar beta = r_dot_z_new / r_dot_z;
			p = this->multAndAdd(beta, p, z);
		}
		if (verbose)
		{
			std::cout << "ConjugateGradient max iterations reached " << Base::m_maxIterations << " error = " << r_dot_z_new << std::endl;
		}
		return Base::m_maxIterations;
	}

	void reinitialize(const TVStack& b)
	{
		r.resize(b.size());
		p.resize(b.size());
		z.resize(b.size());
		temp.resize(b.size());
	}
};
#endif /* btConjugateGradient_h */
