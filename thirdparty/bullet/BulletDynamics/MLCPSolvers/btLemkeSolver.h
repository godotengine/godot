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

#ifndef BT_LEMKE_SOLVER_H
#define BT_LEMKE_SOLVER_H

#include "btMLCPSolverInterface.h"
#include "btLemkeAlgorithm.h"

///The btLemkeSolver is based on "Fast Implementation of Lemke’s Algorithm for Rigid Body Contact Simulation (John E. Lloyd) "
///It is a slower but more accurate solver. Increase the m_maxLoops for better convergence, at the cost of more CPU time.
///The original implementation of the btLemkeAlgorithm was done by Kilian Grundl from the MBSim team
class btLemkeSolver : public btMLCPSolverInterface
{
protected:
public:
	btScalar m_maxValue;
	int m_debugLevel;
	int m_maxLoops;
	bool m_useLoHighBounds;

	btLemkeSolver()
		: m_maxValue(100000),
		  m_debugLevel(0),
		  m_maxLoops(1000),
		  m_useLoHighBounds(true)
	{
	}
	virtual bool solveMLCP(const btMatrixXu& A, const btVectorXu& b, btVectorXu& x, const btVectorXu& lo, const btVectorXu& hi, const btAlignedObjectArray<int>& limitDependency, int numIterations, bool useSparsity = true)
	{
		if (m_useLoHighBounds)
		{
			BT_PROFILE("btLemkeSolver::solveMLCP");
			int n = A.rows();
			if (0 == n)
				return true;

			bool fail = false;

			btVectorXu solution(n);
			btVectorXu q1;
			q1.resize(n);
			for (int row = 0; row < n; row++)
			{
				q1[row] = -b[row];
			}

			//		cout << "A" << endl;
			//		cout << A << endl;

			/////////////////////////////////////

			//slow matrix inversion, replace with LU decomposition
			btMatrixXu A1;
			btMatrixXu B(n, n);
			{
				//BT_PROFILE("inverse(slow)");
				A1.resize(A.rows(), A.cols());
				for (int row = 0; row < A.rows(); row++)
				{
					for (int col = 0; col < A.cols(); col++)
					{
						A1.setElem(row, col, A(row, col));
					}
				}

				btMatrixXu matrix;
				matrix.resize(n, 2 * n);
				for (int row = 0; row < n; row++)
				{
					for (int col = 0; col < n; col++)
					{
						matrix.setElem(row, col, A1(row, col));
					}
				}

				btScalar ratio, a;
				int i, j, k;
				for (i = 0; i < n; i++)
				{
					for (j = n; j < 2 * n; j++)
					{
						if (i == (j - n))
							matrix.setElem(i, j, 1.0);
						else
							matrix.setElem(i, j, 0.0);
					}
				}
				for (i = 0; i < n; i++)
				{
					for (j = 0; j < n; j++)
					{
						if (i != j)
						{
							btScalar v = matrix(i, i);
							if (btFuzzyZero(v))
							{
								a = 0.000001f;
							}
							ratio = matrix(j, i) / matrix(i, i);
							for (k = 0; k < 2 * n; k++)
							{
								matrix.addElem(j, k, -ratio * matrix(i, k));
							}
						}
					}
				}
				for (i = 0; i < n; i++)
				{
					a = matrix(i, i);
					if (btFuzzyZero(a))
					{
						a = 0.000001f;
					}
					btScalar invA = 1.f / a;
					for (j = 0; j < 2 * n; j++)
					{
						matrix.mulElem(i, j, invA);
					}
				}

				for (int row = 0; row < n; row++)
				{
					for (int col = 0; col < n; col++)
					{
						B.setElem(row, col, matrix(row, n + col));
					}
				}
			}

			btMatrixXu b1(n, 1);

			btMatrixXu M(n * 2, n * 2);
			for (int row = 0; row < n; row++)
			{
				b1.setElem(row, 0, -b[row]);
				for (int col = 0; col < n; col++)
				{
					btScalar v = B(row, col);
					M.setElem(row, col, v);
					M.setElem(n + row, n + col, v);
					M.setElem(n + row, col, -v);
					M.setElem(row, n + col, -v);
				}
			}

			btMatrixXu Bb1 = B * b1;
			//		q = [ (-B*b1 - lo)'   (hi + B*b1)' ]'

			btVectorXu qq;
			qq.resize(n * 2);
			for (int row = 0; row < n; row++)
			{
				qq[row] = -Bb1(row, 0) - lo[row];
				qq[n + row] = Bb1(row, 0) + hi[row];
			}

			btVectorXu z1;

			btMatrixXu y1;
			y1.resize(n, 1);
			btLemkeAlgorithm lemke(M, qq, m_debugLevel);
			{
				//BT_PROFILE("lemke.solve");
				lemke.setSystem(M, qq);
				z1 = lemke.solve(m_maxLoops);
			}
			for (int row = 0; row < n; row++)
			{
				y1.setElem(row, 0, z1[2 * n + row] - z1[3 * n + row]);
			}
			btMatrixXu y1_b1(n, 1);
			for (int i = 0; i < n; i++)
			{
				y1_b1.setElem(i, 0, y1(i, 0) - b1(i, 0));
			}

			btMatrixXu x1;

			x1 = B * (y1_b1);

			for (int row = 0; row < n; row++)
			{
				solution[row] = x1(row, 0);  //n];
			}

			int errorIndexMax = -1;
			int errorIndexMin = -1;
			float errorValueMax = -1e30;
			float errorValueMin = 1e30;

			for (int i = 0; i < n; i++)
			{
				x[i] = solution[i];
				volatile btScalar check = x[i];
				if (x[i] != check)
				{
					//printf("Lemke result is #NAN\n");
					x.setZero();
					return false;
				}

				//this is some hack/safety mechanism, to discard invalid solutions from the Lemke solver
				//we need to figure out why it happens, and fix it, or detect it properly)
				if (x[i] > m_maxValue)
				{
					if (x[i] > errorValueMax)
					{
						fail = true;
						errorIndexMax = i;
						errorValueMax = x[i];
					}
					////printf("x[i] = %f,",x[i]);
				}
				if (x[i] < -m_maxValue)
				{
					if (x[i] < errorValueMin)
					{
						errorIndexMin = i;
						errorValueMin = x[i];
						fail = true;
						//printf("x[i] = %f,",x[i]);
					}
				}
			}
			if (fail)
			{
				int m_errorCountTimes = 0;
				if (errorIndexMin < 0)
					errorValueMin = 0.f;
				if (errorIndexMax < 0)
					errorValueMax = 0.f;
				m_errorCountTimes++;
				//	printf("Error (x[%d] = %f, x[%d] = %f), resetting %d times\n", errorIndexMin,errorValueMin, errorIndexMax, errorValueMax, errorCountTimes++);
				for (int i = 0; i < n; i++)
				{
					x[i] = 0.f;
				}
			}
			return !fail;
		}
		else

		{
			int dimension = A.rows();
			if (0 == dimension)
				return true;

			//		printf("================ solving using Lemke/Newton/Fixpoint\n");

			btVectorXu q;
			q.resize(dimension);
			for (int row = 0; row < dimension; row++)
			{
				q[row] = -b[row];
			}

			btLemkeAlgorithm lemke(A, q, m_debugLevel);

			lemke.setSystem(A, q);

			btVectorXu solution = lemke.solve(m_maxLoops);

			//check solution

			bool fail = false;
			int errorIndexMax = -1;
			int errorIndexMin = -1;
			float errorValueMax = -1e30;
			float errorValueMin = 1e30;

			for (int i = 0; i < dimension; i++)
			{
				x[i] = solution[i + dimension];
				volatile btScalar check = x[i];
				if (x[i] != check)
				{
					x.setZero();
					return false;
				}

				//this is some hack/safety mechanism, to discard invalid solutions from the Lemke solver
				//we need to figure out why it happens, and fix it, or detect it properly)
				if (x[i] > m_maxValue)
				{
					if (x[i] > errorValueMax)
					{
						fail = true;
						errorIndexMax = i;
						errorValueMax = x[i];
					}
					////printf("x[i] = %f,",x[i]);
				}
				if (x[i] < -m_maxValue)
				{
					if (x[i] < errorValueMin)
					{
						errorIndexMin = i;
						errorValueMin = x[i];
						fail = true;
						//printf("x[i] = %f,",x[i]);
					}
				}
			}
			if (fail)
			{
				static int errorCountTimes = 0;
				if (errorIndexMin < 0)
					errorValueMin = 0.f;
				if (errorIndexMax < 0)
					errorValueMax = 0.f;
				printf("Error (x[%d] = %f, x[%d] = %f), resetting %d times\n", errorIndexMin, errorValueMin, errorIndexMax, errorValueMax, errorCountTimes++);
				for (int i = 0; i < dimension; i++)
				{
					x[i] = 0.f;
				}
			}

			return !fail;
		}
		return true;
	}
};

#endif  //BT_LEMKE_SOLVER_H
