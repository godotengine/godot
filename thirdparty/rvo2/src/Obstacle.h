/*
 * Obstacle.h
 * RVO2 Library
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

#ifndef RVO_OBSTACLE_H_
#define RVO_OBSTACLE_H_

/**
 * \file       Obstacle.h
 * \brief      Contains the Obstacle class.
 */

#include "Definitions.h"

namespace RVO {
	/**
	 * \brief      Defines static obstacles in the simulation.
	 */
	class Obstacle {
	public:
		/**
		 * \brief      Constructs a static obstacle instance.
		 */
		Obstacle();

		bool isConvex_;
		Obstacle *nextObstacle_;
		Vector2 point_;
		Obstacle *prevObstacle_;
		Vector2 unitDir_;

		size_t id_;
	};
}

#endif /* RVO_OBSTACLE_H_ */
