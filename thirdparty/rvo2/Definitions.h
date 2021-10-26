/*
 * Definitions.h
 * RVO2-3D Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

/**
 * \file   Definitions.h
 * \brief  Contains functions and constants used in multiple classes.
 */

#ifndef RVO_DEFINITIONS_H_
#define RVO_DEFINITIONS_H_

#include "API.h"

namespace RVO {
	/**
	 * \brief   Computes the square of a float.
	 * \param   scalar  The float to be squared.
	 * \return  The square of the float.
	 */
	inline float sqr(float scalar)
	{
		return scalar * scalar;
	}
}

#endif /* RVO_DEFINITIONS_H_ */
