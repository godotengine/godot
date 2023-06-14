/*
 * Vector2.h
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

#ifndef RVO_VECTOR2_H_
#define RVO_VECTOR2_H_

/**
 * @file  Vector2.h
 * @brief Declares and defines the Vector2 class.
 */

#include <iosfwd>

namespace RVO2D {
/**
 * @brief A sufficiently small positive number.
 */
extern const float RVO2D_EPSILON;

/**
 * @brief Defines a two-dimensional vector.
 */
class Vector2 {
 public:
  /**
   * @brief Constructs and initializes a two-dimensional vector instance to
   *        (0.0, 0.0).
   */
  Vector2();

  /**
   * @brief     Constructs and initializes a two-dimensional vector from the
   *            specified xy-coordinates.
   * @param[in] x The x-coordinate of the two-dimensional vector.
   * @param[in] y The y-coordinate of the two-dimensional vector.
   */
  Vector2(float x, float y);

  /**
   * @brief  Returns the x-coordinate of this two-dimensional vector.
   * @return The x-coordinate of the two-dimensional vector.
   */
  float x() const { return x_; }

  /**
   * @brief  Returns the y-coordinate of this two-dimensional vector.
   * @return The y-coordinate of the two-dimensional vector.
   */
  float y() const { return y_; }

  /**
   * @brief  Computes the negation of this two-dimensional vector.
   * @return The negation of this two-dimensional vector.
   */
  Vector2 operator-() const;

  /**
   * @brief     Computes the dot product of this two-dimensional vector with the
   *            specified two-dimensional vector.
   * @param[in] vector The two-dimensional vector with which the dot product
   *                   should be computed.
   * @return    The dot product of this two-dimensional vector with a specified
   *            two-dimensional vector.
   */
  float operator*(const Vector2 &vector) const;

  /**
   * @brief     Computes the scalar multiplication of this two-dimensional
   *            vector with the specified scalar value.
   * @param[in] scalar The scalar value with which the scalar multiplication
   *                   should be computed.
   * @return    The scalar multiplication of this two-dimensional vector with a
   *            specified scalar value.
   */
  Vector2 operator*(float scalar) const;

  /**
   * @brief     Computes the scalar division of this two-dimensional vector with
   *            the specified scalar value.
   * @param[in] scalar The scalar value with which the scalar division should be
   *                   computed.
   * @return    The scalar division of this two-dimensional vector with a
   *            specified scalar value.
   */
  Vector2 operator/(float scalar) const;

  /**
   * @brief     Computes the vector sum of this two-dimensional vector with the
   *            specified two-dimensional vector.
   * @param[in] vector The two-dimensional vector with which the vector sum
   *                   should be computed.
   * @return    The vector sum of this two-dimensional vector with a specified
   *            two-dimensional vector.
   */
  Vector2 operator+(const Vector2 &vector) const;

  /**
   * @brief     Computes the vector difference of this two-dimensional vector
   *            with the specified two-dimensional vector.
   * @param[in] vector The two-dimensional vector with which the vector
   *                   difference should be computed.
   * @return    The vector difference of this two-dimensional vector with a
   *            specified two-dimensional vector.
   */
  Vector2 operator-(const Vector2 &vector) const;

  /**
   * @brief     Tests this two-dimensional vector for equality with the
   *            specified two-dimensional vector.
   * @param[in] vector The two-dimensional vector with which to test for
   *                   equality.
   * @return    True if the two-dimensional vectors are equal.
   */
  bool operator==(const Vector2 &vector) const;

  /**
   * @brief     Tests this two-dimensional vector for inequality with the
   *            specified two-dimensional vector.
   * @param[in] vector The two-dimensional vector with which to test for
   *                   inequality.
   * @return    True if the two-dimensional vectors are not equal.
   */
  bool operator!=(const Vector2 &vector) const;

  /**
   * @brief     Sets the value of this two-dimensional vector to the scalar
   *            multiplication of itself with the specified scalar value.
   * @param[in] scalar The scalar value with which the scalar multiplication
   *                   should be computed.
   * @return    A reference to this two-dimensional vector.
   */
  Vector2 &operator*=(float scalar);

  /**
   * @brief     Sets the value of this two-dimensional vector to the scalar
   *            division of itself with the specified scalar value.
   * @param[in] scalar The scalar value with which the scalar division should be
   *                   computed.
   * @return    A reference to this two-dimensional vector.
   */
  Vector2 &operator/=(float scalar);

  /**
   * @brief     Sets the value of this two-dimensional vector to the vector sum
   *            of itself with the specified two-dimensional vector.
   * @param[in] vector The two-dimensional vector with which the vector sum
   *                   should be computed.
   * @return    A reference to this two-dimensional vector.
   */
  Vector2 &operator+=(const Vector2 &vector);

  /**
   * @brief     Sets the value of this two-dimensional vector to the vector
   *            difference of itself with the specified two-dimensional vector.
   * @param[in] vector The two-dimensional vector with which the vector
   *                   difference should be computed.
   * @return    A reference to this two-dimensional vector.
   */
  Vector2 &operator-=(const Vector2 &vector);

 private:
  float x_;
  float y_;
};

/**
 * @relates   Vector2
 * @brief     Computes the scalar multiplication of the specified
 *            two-dimensional vector with the specified scalar value.
 * @param[in] scalar The scalar value with which the scalar multiplication
 *                   should be computed.
 * @param[in] vector The two-dimensional vector with which the scalar
 *                   multiplication should be computed.
 * @return    The scalar multiplication of the two-dimensional vector with the
 *            scalar value.
 */
 Vector2 operator*(float scalar, const Vector2 &vector);

/**
 * @relates        Vector2
 * @brief          Inserts the specified two-dimensional vector into the
 *                 specified output stream.
 * @param[in, out] stream The output stream into which the two-dimensional
 *                        vector should be inserted.
 * @param[in]      vector The two-dimensional vector which to insert into the
 *                        output stream.
 * @return         A reference to the output stream.
 */
 std::ostream &operator<<(std::ostream &stream,
                                    const Vector2 &vector);

/**
 * @relates   Vector2
 * @brief     Computes the length of a specified two-dimensional vector.
 * @param[in] vector The two-dimensional vector whose length is to be computed.
 * @return    The length of the two-dimensional vector.
 */
 float abs(const Vector2 &vector);

/**
 * @relates   Vector2
 * @brief     Computes the squared length of a specified two-dimensional vector.
 * @param[in] vector The two-dimensional vector whose squared length is to be
 *                   computed.
 * @return    The squared length of the two-dimensional vector.
 */
 float absSq(const Vector2 &vector);

/**
 * @relates   Vector2
 * @brief     Computes the determinant of a two-dimensional square matrix with
 *            rows consisting of the specified two-dimensional vectors.
 * @param[in] vector1 The top row of the two-dimensional square matrix.
 * @param[in] vector2 The bottom row of the two-dimensional square matrix.
 * @return    The determinant of the two-dimensional square matrix.
 */
 float det(const Vector2 &vector1, const Vector2 &vector2);

/**
 * @brief     Computes the signed distance from a line connecting th specified
 *            points to a specified point.
 * @param[in] vector1 The first point on the line.
 * @param[in] vector2 The second point on the line.
 * @param[in] vector3 The point to which the signed distance is to be
 *                    calculated.
 * @return    Positive when the point vector3 lies to the left of the line
 *            vector1-vector2.
 */
 float leftOf(const Vector2 &vector1, const Vector2 &vector2,
                        const Vector2 &vector3);

/**
 * @relates   Vector2
 * @brief     Computes the normalization of the specified two-dimensional
 *            vector.
 * @param[in] vector The two-dimensional vector whose normalization is to be
 *                   computed.
 * @return    The normalization of the two-dimensional vector.
 */
 Vector2 normalize(const Vector2 &vector);
} /* namespace RVO2D */

#endif /* RVO_VECTOR2_H_ */
