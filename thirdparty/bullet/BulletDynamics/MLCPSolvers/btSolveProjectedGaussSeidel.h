/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
///original version written by Erwin Coumans, October 2013

#ifndef BT_SOLVE_PROJECTED_GAUSS_SEIDEL_H
#define BT_SOLVE_PROJECTED_GAUSS_SEIDEL_H


#include "btMLCPSolverInterface.h"

///This solver is mainly for debug/learning purposes: it is functionally equivalent to the btSequentialImpulseConstraintSolver solver, but much slower (it builds the full LCP matrix)
class btSolveProjectedGaussSeidel : public btMLCPSolverInterface
{

public:

	btScalar m_leastSquaresResidualThreshold;
	btScalar m_leastSquaresResidual;

	btSolveProjectedGaussSeidel()
		:m_leastSquaresResidualThreshold(0),
		m_leastSquaresResidual(0)
	{
	}

	virtual bool solveMLCP(const btMatrixXu & A, const btVectorXu & b, btVectorXu& x, const btVectorXu & lo,const btVectorXu & hi,const btAlignedObjectArray<int>& limitDependency, int numIterations, bool useSparsity = true)
	{
		if (!A.rows())
			return true;
		//the A matrix is sparse, so compute the non-zero elements
		A.rowComputeNonZeroElements();

		//A is a m-n matrix, m rows, n columns
		btAssert(A.rows() == b.rows());

		int i, j, numRows = A.rows();
	
		btScalar delta;

		for (int k = 0; k <numIterations; k++)
		{
			m_leastSquaresResidual = 0.f;
			for (i = 0; i <numRows; i++)
			{
				delta = 0.0f;
				if (useSparsity)
				{
					for (int h=0;h<A.m_rowNonZeroElements1[i].size();h++)
					{
						int j = A.m_rowNonZeroElements1[i][h];
						if (j != i)//skip main diagonal
						{
							delta += A(i,j) * x[j];
						}
					}
				} else
				{
					for (j = 0; j <i; j++) 
						delta += A(i,j) * x[j];
					for (j = i+1; j<numRows; j++) 
						delta += A(i,j) * x[j];
				}

				btScalar aDiag = A(i,i);
				btScalar xOld = x[i];
				x [i] = (b [i] - delta) / aDiag;
				btScalar s = 1.f;

				if (limitDependency[i]>=0)
				{
					s = x[limitDependency[i]];
					if (s<0)
						s=1;
				}
			
				if (x[i]<lo[i]*s)
					x[i]=lo[i]*s;
				if (x[i]>hi[i]*s)
					x[i]=hi[i]*s;
				btScalar diff = x[i] - xOld;
				m_leastSquaresResidual += diff*diff;
			}

			btScalar eps  = m_leastSquaresResidualThreshold;
			if ((m_leastSquaresResidual < eps) || (k >=(numIterations-1)))
			{
#ifdef VERBOSE_PRINTF_RESIDUAL
				printf("totalLenSqr = %f at iteration #%d\n", m_leastSquaresResidual,k);
#endif
				break;
			}
		}
		return true;
	}

};

#endif //BT_SOLVE_PROJECTED_GAUSS_SEIDEL_H
