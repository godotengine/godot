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


#ifndef BT_PATH_SOLVER_H
#define BT_PATH_SOLVER_H

//#define BT_USE_PATH
#ifdef BT_USE_PATH

extern "C" {
#include "PATH/SimpleLCP.h"
#include "PATH/License.h"
#include "PATH/Error_Interface.h"
};
  void __stdcall MyError(Void *data, Char *msg)
{
	printf("Path Error: %s\n",msg);
}
  void __stdcall MyWarning(Void *data, Char *msg)
{
	printf("Path Warning: %s\n",msg);
}

Error_Interface e;



#include "btMLCPSolverInterface.h"
#include "Dantzig/lcp.h"

class btPathSolver : public btMLCPSolverInterface
{
public:

	btPathSolver()
	{
		License_SetString("2069810742&Courtesy_License&&&USR&2013&14_12_2011&1000&PATH&GEN&31_12_2013&0_0_0&0&0_0");
		e.error_data = 0;
		e.warning = MyWarning;
		e.error = MyError;
		Error_SetInterface(&e);
	}


	virtual bool solveMLCP(const btMatrixXu & A, const btVectorXu & b, btVectorXu& x, const btVectorXu & lo,const btVectorXu & hi,const btAlignedObjectArray<int>& limitDependency, int numIterations, bool useSparsity = true)
	{
		MCP_Termination status;
		

		int numVariables = b.rows();
		if (0==numVariables)
			return true;

			/*	 - variables - the number of variables in the problem
			- m_nnz - the number of nonzeros in the M matrix
			- m_i - a vector of size m_nnz containing the row indices for M
			- m_j - a vector of size m_nnz containing the column indices for M
			- m_ij - a vector of size m_nnz containing the data for M
			- q - a vector of size variables
			- lb - a vector of size variables containing the lower bounds on x
			- ub - a vector of size variables containing the upper bounds on x
			*/
		btAlignedObjectArray<double> values;
		btAlignedObjectArray<int> rowIndices;
		btAlignedObjectArray<int> colIndices;

		for (int i=0;i<A.rows();i++)
		{
			for (int j=0;j<A.cols();j++)
			{
				if (A(i,j)!=0.f)
				{
					//add 1, because Path starts at 1, instead of 0
					rowIndices.push_back(i+1);
					colIndices.push_back(j+1);
					values.push_back(A(i,j));
				}
			}
		}
		int numNonZero = rowIndices.size();
		btAlignedObjectArray<double> zResult;
		zResult.resize(numVariables);
		btAlignedObjectArray<double> rhs;
		btAlignedObjectArray<double> upperBounds;
		btAlignedObjectArray<double> lowerBounds;
		for (int i=0;i<numVariables;i++)
		{
			upperBounds.push_back(hi[i]);
			lowerBounds.push_back(lo[i]);
			rhs.push_back(-b[i]);
		}


		SimpleLCP(numVariables,numNonZero,&rowIndices[0],&colIndices[0],&values[0],&rhs[0],&lowerBounds[0],&upperBounds[0], &status, &zResult[0]);

		if (status != MCP_Solved)
		{
			static const char* gReturnMsgs[] = {
				"Invalid return",
				"MCP_Solved: The problem was solved",
				"MCP_NoProgress: A stationary point was found",
				"MCP_MajorIterationLimit: Major iteration limit met",
				"MCP_MinorIterationLimit: Cumulative minor iteration limit met",
				"MCP_TimeLimit: Ran out of time",
				"MCP_UserInterrupt: Control-C, typically",
				"MCP_BoundError: Problem has a bound error",
				"MCP_DomainError: Could not find starting point",
				"MCP_Infeasible: Problem has no solution",
				"MCP_Error: An error occurred within the code",
				"MCP_LicenseError: License could not be found",
				"MCP_OK"
			};

			printf("ERROR: The PATH MCP solver failed: %s\n", gReturnMsgs[(unsigned int)status]);// << std::endl;
			printf("using Projected Gauss Seidel fallback\n");
			
			return false;
		} else
		{
			for (int i=0;i<numVariables;i++)
			{
				x[i] = zResult[i];
				//check for #NAN
				if (x[i] != zResult[i])
					return false;
			}
			return true;

		}

	}
};

#endif //BT_USE_PATH


#endif //BT_PATH_SOLVER_H
