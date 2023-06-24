/*
 * Vector3.cc
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

#include "Vector3.h"

#include <cmath>
#include <ostream>

namespace RVO3D {
Vector3::Vector3() : val_() {
  val_[0] = 0.0F;
  val_[1] = 0.0F;
  val_[2] = 0.0F;
}

Vector3::Vector3(const Vector3 &vector) : val_() {
  val_[0] = vector[0];
  val_[1] = vector[1];
  val_[2] = vector[2];
}

Vector3::Vector3(const float val[3]) : val_() {
  val_[0] = val[0];
  val_[1] = val[1];
  val_[2] = val[2];
}

Vector3::Vector3(float x, float y, float z) : val_() {
  val_[0] = x;
  val_[1] = y;
  val_[2] = z;
}

Vector3::~Vector3() {}

Vector3 &Vector3::operator=(const Vector3 &vector) {
  if (this != &vector) {
    val_[0] = vector[0];
    val_[1] = vector[1];
    val_[2] = vector[2];
  }

  return *this;
}

float Vector3::operator[](std::size_t i) const { return val_[i]; }

float &Vector3::operator[](std::size_t i) { return val_[i]; }

Vector3 Vector3::operator-() const {
  return Vector3(-val_[0], -val_[1], -val_[2]);
}

float Vector3::operator*(const Vector3 &vector) const {
  return val_[0] * vector[0] + val_[1] * vector[1] + val_[2] * vector[2];
}

Vector3 Vector3::operator*(float scalar) const {
  return Vector3(val_[0] * scalar, val_[1] * scalar, val_[2] * scalar);
}

Vector3 Vector3::operator/(float scalar) const {
  const float invScalar = 1.0F / scalar;

  return Vector3(val_[0] * invScalar, val_[1] * invScalar, val_[2] * invScalar);
}

Vector3 Vector3::operator+(const Vector3 &vector) const {
  return Vector3(val_[0] + vector[0], val_[1] + vector[1], val_[2] + vector[2]);
}

Vector3 Vector3::operator-(const Vector3 &vector) const {
  return Vector3(val_[0] - vector[0], val_[1] - vector[1], val_[2] - vector[2]);
}

bool Vector3::operator==(const Vector3 &vector) const {
  return val_[0] == vector[0] && val_[1] == vector[1] && val_[2] == vector[2];
}

bool Vector3::operator!=(const Vector3 &vector) const {
  return val_[0] != vector[0] || val_[1] != vector[1] || val_[2] != vector[2];
}

Vector3 &Vector3::operator*=(float scalar) {
  val_[0] *= scalar;
  val_[1] *= scalar;
  val_[2] *= scalar;

  return *this;
}

Vector3 &Vector3::operator/=(float scalar) {
  const float invScalar = 1.0F / scalar;

  val_[0] *= invScalar;
  val_[1] *= invScalar;
  val_[2] *= invScalar;

  return *this;
}

Vector3 &Vector3::operator+=(const Vector3 &vector) {
  val_[0] += vector[0];
  val_[1] += vector[1];
  val_[2] += vector[2];

  return *this;
}

Vector3 &Vector3::operator-=(const Vector3 &vector) {
  val_[0] -= vector[0];
  val_[1] -= vector[1];
  val_[2] -= vector[2];

  return *this;
}

Vector3 operator*(float scalar, const Vector3 &vector) {
  return Vector3(scalar * vector[0], scalar * vector[1], scalar * vector[2]);
}

std::ostream &operator<<(std::ostream &stream, const Vector3 &vector) {
  stream << "(" << vector[0] << "," << vector[1] << "," << vector[2] << ")";

  return stream;
}

float abs(const Vector3 &vector) { return std::sqrt(vector * vector); }

float absSq(const Vector3 &vector) { return vector * vector; }

Vector3 cross(const Vector3 &vector1, const Vector3 &vector2) {
  return Vector3(vector1[1] * vector2[2] - vector1[2] * vector2[1],
                 vector1[2] * vector2[0] - vector1[0] * vector2[2],
                 vector1[0] * vector2[1] - vector1[1] * vector2[0]);
}

Vector3 normalize(const Vector3 &vector) { return vector / abs(vector); }

} /* namespace RVO3D */
