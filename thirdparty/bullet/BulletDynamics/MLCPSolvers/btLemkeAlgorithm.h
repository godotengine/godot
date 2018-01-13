/* Copyright (C) 2004-2013 MBSim Development Team

Code was converted for the Bullet Continuous Collision Detection and Physics Library

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

//The original version is here
//https://code.google.com/p/mbsim-env/source/browse/trunk/kernel/mbsim/numerics/linear_complementarity_problem/lemke_algorithm.cc
//This file is re-distributed under the ZLib license, with permission of the original author (Kilian Grundl)
//Math library was replaced from fmatvec to a the file src/LinearMath/btMatrixX.h
//STL/std::vector replaced by btAlignedObjectArray



#ifndef BT_NUMERICS_LEMKE_ALGORITHM_H_
#define BT_NUMERICS_LEMKE_ALGORITHM_H_

#include "LinearMath/btMatrixX.h"


#include <vector> //todo: replace by btAlignedObjectArray

class btLemkeAlgorithm
{
public:
 

  btLemkeAlgorithm(const btMatrixXu& M_, const btVectorXu& q_, const int & DEBUGLEVEL_ = 0) :
	  DEBUGLEVEL(DEBUGLEVEL_)
  {
	setSystem(M_, q_);
  }

  /* GETTER / SETTER */
  /**
   * \brief return info of solution process
   */
  int getInfo() {
	return info;
  }

  /**
   * \brief get the number of steps until the solution was found
   */
  int getSteps(void) {
	return steps;
  }



  /**
   * \brief set system with Matrix M and vector q
   */
  void setSystem(const btMatrixXu & M_, const btVectorXu & q_)
	{
		m_M = M_;
		m_q = q_;
  }
  /***************************************************/

  /**
   * \brief solve algorithm adapted from : Fast Implementation of Lemke’s Algorithm for Rigid Body Contact Simulation (John E. Lloyd)
   */
  btVectorXu solve(unsigned int maxloops = 0);

  virtual ~btLemkeAlgorithm() {
  }

protected:
  int findLexicographicMinimum(const btMatrixXu &A, const int & pivotColIndex);
  bool LexicographicPositive(const btVectorXu & v);
  void GaussJordanEliminationStep(btMatrixXu &A, int pivotRowIndex, int pivotColumnIndex, const btAlignedObjectArray<int>& basis);
  bool greaterZero(const btVectorXu & vector);
  bool validBasis(const btAlignedObjectArray<int>& basis);

  btMatrixXu m_M;
  btVectorXu m_q;

  /**
   * \brief number of steps until the Lemke algorithm found a solution
   */
  unsigned int steps;

  /**
   * \brief define level of debug output
   */
  int DEBUGLEVEL;

  /**
   * \brief did the algorithm find a solution
   *
   * -1 : not successful
   *  0 : successful
   */
  int info;
};


#endif /* BT_NUMERICS_LEMKE_ALGORITHM_H_ */
