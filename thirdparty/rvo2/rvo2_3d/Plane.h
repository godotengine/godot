/*
 * Plane.h
 * RVO2-3D Library
 *
 * SPDX-FileCopyrightText: 2008 University of North Carolina at Chapel Hill
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
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
 * <https://gamma.cs.unc.edu/RVO2/>
 */

#ifndef RVO3D_PLANE_H_
#define RVO3D_PLANE_H_

/**
 * @file  Plane.h
 * @brief Contains the Plane class.
 */

#include "Vector3.h"

namespace RVO3D {
/**
 * @brief Defines a plane.
 */
class Plane {
 public:
  /**
   * @brief Constructs a plane.
   */
  Plane();

  /**
   * @brief A point on the plane.
   */
  Vector3 point;

  /**
   * @brief The normal to the plane.
   */
  Vector3 normal;
};
} /* namespace RVO3D */

#endif /* RVO3D_PLANE_H_ */
