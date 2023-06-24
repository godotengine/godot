/*
 * Vector2.cpp
 * RVO2 Library
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

/**
 * @file  Vector2.cc
 * @brief Defines the Vector2 class.
 */

#include "Vector2.h"

#include <cmath>
#include <ostream>

namespace RVO2D {
const float RVO2D_EPSILON = 0.00001F;

Vector2::Vector2() : x_(0.0F), y_(0.0F) {}

Vector2::Vector2(float x, float y) : x_(x), y_(y) {}

Vector2 Vector2::operator-() const { return Vector2(-x_, -y_); }

float Vector2::operator*(const Vector2 &vector) const {
  return x_ * vector.x_ + y_ * vector.y_;
}

Vector2 Vector2::operator*(float scalar) const {
  return Vector2(x_ * scalar, y_ * scalar);
}

Vector2 Vector2::operator/(float scalar) const {
  const float invScalar = 1.0F / scalar;

  return Vector2(x_ * invScalar, y_ * invScalar);
}

Vector2 Vector2::operator+(const Vector2 &vector) const {
  return Vector2(x_ + vector.x_, y_ + vector.y_);
}

Vector2 Vector2::operator-(const Vector2 &vector) const {
  return Vector2(x_ - vector.x_, y_ - vector.y_);
}

bool Vector2::operator==(const Vector2 &vector) const {
  return x_ == vector.x_ && y_ == vector.y_;
}

bool Vector2::operator!=(const Vector2 &vector) const {
  return x_ != vector.x_ || y_ != vector.y_;
}

Vector2 &Vector2::operator*=(float scalar) {
  x_ *= scalar;
  y_ *= scalar;

  return *this;
}

Vector2 &Vector2::operator/=(float scalar) {
  const float invScalar = 1.0F / scalar;
  x_ *= invScalar;
  y_ *= invScalar;

  return *this;
}

Vector2 &Vector2::operator+=(const Vector2 &vector) {
  x_ += vector.x_;
  y_ += vector.y_;

  return *this;
}

Vector2 &Vector2::operator-=(const Vector2 &vector) {
  x_ -= vector.x_;
  y_ -= vector.y_;

  return *this;
}

Vector2 operator*(float scalar, const Vector2 &vector) {
  return Vector2(scalar * vector.x(), scalar * vector.y());
}

std::ostream &operator<<(std::ostream &stream, const Vector2 &vector) {
  stream << "(" << vector.x() << "," << vector.y() << ")";

  return stream;
}

float abs(const Vector2 &vector) { return std::sqrt(vector * vector); }

float absSq(const Vector2 &vector) { return vector * vector; }

float det(const Vector2 &vector1, const Vector2 &vector2) {
  return vector1.x() * vector2.y() - vector1.y() * vector2.x();
}

float leftOf(const Vector2 &vector1, const Vector2 &vector2,
             const Vector2 &vector3) {
  return det(vector1 - vector3, vector2 - vector1);
}

Vector2 normalize(const Vector2 &vector) { return vector / abs(vector); }
} /* namespace RVO */
