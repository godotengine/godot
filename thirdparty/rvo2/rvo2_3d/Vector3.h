/*
 * Vector3.h
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

#ifndef RVO3D_VECTOR3_H_
#define RVO3D_VECTOR3_H_

/**
 * @file  Vector3.h
 * @brief Contains the Vector3 class.
 */

#include <cstddef>
#include <iosfwd>

namespace RVO3D {
/**
 * @brief Defines a three-dimensional vector.
 */
class Vector3 {
 public:
  /**
   * @brief Constructs and initializes a three-dimensional vector instance to
   *        zero.
   */
  Vector3();

  /**
   * @brief     Constructs and initializes a three-dimensional vector from the
   *            specified three-dimensional vector.
   * @param[in] vector The three-dimensional vector containing the
   *                   xyz-coordinates.
   */
  Vector3(const Vector3 &vector);

  /**
   * @brief     Constructs and initializes a three-dimensional vector from the
   *            specified three-element array.
   * @param[in] val The three-element array containing the xyz-coordinates.
   */
  explicit Vector3(const float val[3]);

  /**
   * @brief     Constructs and initializes a three-dimensional vector from the
   *            specified xyz-coordinates.
   * @param[in] x The x-coordinate of the three-dimensional vector.
   * @param[in] y The y-coordinate of the three-dimensional vector.
   * @param[in] z The z-coordinate of the three-dimensional vector.
   */
  Vector3(float x, float y, float z);

  /**
   * @brief Destroys this three-dimensional vector instance.
   */
  ~Vector3();

  /**
   * @brief  Returns the x-coordinate of this three-dimensional vector.
   * @return The x-coordinate of the three-dimensional vector.
   */
  float x() const { return val_[0]; }

  /**
   * @brief  Returns the y-coordinate of this three-dimensional vector.
   * @return The y-coordinate of the three-dimensional vector.
   */
  float y() const { return val_[1]; }

  /**
   * @brief  Returns the z-coordinate of this three-dimensional vector.
   * @return The z-coordinate of the three-dimensional vector.
   */
  float z() const { return val_[2]; }

  /**
   * @brief     Assigns a copy of the specified three-dimensional vector to
   *            this three-dimensional vector instance.
   * @param[in] vector The three-dimensional vector containing the
   *                   xyz-coordinates.
   * @return    A reference to this three-dimensional vector instance.
   */
  Vector3 &operator=(const Vector3 &vector);

  /**
   * @brief     Returns the specified coordinate of this three-dimensional
   *            vector.
   * @param[in] i The coordinate that should be returned (0 <= i < 3).
   * @return    The specified coordinate of the three-dimensional vector.
   */
  float operator[](std::size_t i) const;

  /**
   * @brief     Returns a reference to the specified coordinate of this
   *            three-dimensional vector.
   * @param[in] i The coordinate to which a reference should be returned
   *              (0 <= i < 3).
   * @return    A reference to the specified coordinate of the three-dimensional
   *            vector.
   */
  float &operator[](std::size_t i);

  /**
   * @brief  Computes the negation of this three-dimensional vector.
   * @return The negation of this three-dimensional vector.
   */
  Vector3 operator-() const;

  /**
   * @brief     Computes the dot product of this three-dimensional vector with
   *            the specified three-dimensional vector.
   * @param[in] vector The three-dimensional vector with which the dot product
   *                   should be computed.
   * @return    The dot product of this three-dimensional vector with a
   *            specified three-dimensional vector.
   */
  float operator*(const Vector3 &vector) const;

  /**
   * @brief     Computes the scalar multiplication of this three-dimensional
   *            vector with the specified scalar value.
   * @param[in] scalar The scalar value with which the scalar multiplication
   *                   should be computed.
   * @return    The scalar multiplication of this three-dimensional vector with
   *            a specified scalar value.
   */
  Vector3 operator*(float scalar) const;

  /**
   * @brief     Computes the scalar division of this three-dimensional vector
   *            with the specified scalar value.
   * @param[in] scalar The scalar value with which the scalar division should be
   *                   computed.
   * @return    The scalar division of this three-dimensional vector with a
   *            specified scalar value.
   */
  Vector3 operator/(float scalar) const;

  /**
   * @brief     Computes the vector sum of this three-dimensional vector with
   *            the specified three-dimensional vector.
   * @param[in] vector The three-dimensional vector with which the vector sum
   *                   should be computed.
   * @return    The vector sum of this three-dimensional vector with a specified
   *            three-dimensional vector.
   */
  Vector3 operator+(const Vector3 &vector) const;

  /**
   * @brief      Computes the vector difference of this three-dimensional vector
   *             with the specified three-dimensional vector.
   * @param[in]  vector The three-dimensional vector with which the vector
   *                    difference should be computed.
   * @return     The vector difference of this three-dimensional vector with a
   *             specified three-dimensional vector.
   */
  Vector3 operator-(const Vector3 &vector) const;

  /**
   * @brief     Tests this three-dimensional vector for equality with the
   *            specified three-dimensional vector.
   * @param[in] vector The three-dimensional vector with which to test for
   *                   equality.
   * @return    True if the three-dimensional vectors are equal.
   */
  bool operator==(const Vector3 &vector) const;

  /**
   * @brief     Tests this three-dimensional vector for inequality with the
   *            specified three-dimensional vector
   * @param[in] vector The three-dimensional vector with which to test for
   *                   inequality.
   * @return    True if the three-dimensional vectors are not equal.
   */
  bool operator!=(const Vector3 &vector) const;

  /**
   * @brief     Sets the value of this three-dimensional vector to the scalar
   *            multiplication of itself with the specified scalar value.
   * @param[in] scalar The scalar value with which the scalar multiplication
   *                   should be computed.
   * @return    A reference to this three-dimensional vector.
   */
  Vector3 &operator*=(float scalar);

  /**
   * @brief     Sets the value of this three-dimensional vector to the scalar
   *            division of itself with the specified scalar value.
   * @param[in] scalar The scalar value with which the scalar division should be
   *                   computed.
   * @return    A reference to this three-dimensional vector.
   */
  Vector3 &operator/=(float scalar);

  /**
   * @brief     Sets the value of this three-dimensional vector to the vector
   *            sum of itself with the specified three-dimensional vector.
   * @param[in] vector The three-dimensional vector with which the vector sum
   *                   should be computed.
   * @return    A reference to this three-dimensional vector.
   */
  Vector3 &operator+=(const Vector3 &vector);

  /**
   * @brief     Sets the value of this three-dimensional vector to the vector
   *            difference of itself with the specified three-dimensional
   *            vector.
   * @param[in] vector The three-dimensional vector with which the vector
   *                   difference should be computed.
   * @return    A reference to this three-dimensional vector.
   */
  Vector3 &operator-=(const Vector3 &vector);

 private:
  float val_[3];
};

/**
 * @relates   Vector3
 * @brief     Computes the scalar multiplication of the specified
 *            three-dimensional vector with the specified scalar value.
 * @param[in] scalar The scalar value with which the scalar multiplication
 *                   should be computed.
 * @param[in] vector The three-dimensional vector with which the scalar
 *                   multiplication should be computed.
 * @return    The scalar multiplication of the three-dimensional vector with the
 *            scalar value.
 */
Vector3 operator*(float scalar, const Vector3 &vector);

/**
 * @relates       Vector3
 * @brief         Inserts the specified three-dimensional vector into the
 *                specified output stream.
 * @param[in,out] os     The output stream into which the three-dimensional
 *                       vector should be inserted.
 * @param[in]     vector The three-dimensional vector which to insert into the
 *                       output stream.
 * @return        A reference to the output stream.
 */
std::ostream &operator<<(std::ostream &stream,
                                      const Vector3 &vector);

/**
 * @relates   Vector3
 * @brief     Computes the length of a specified three-dimensional vector.
 * @param[in] vector The three-dimensional vector whose length is to be
 *            computed.
 * @return    The length of the three-dimensional vector.
 */
float abs(const Vector3 &vector);

/**
 * @relates   Vector3
 * @brief     Computes the squared length of a specified three-dimensional
 *            vector.
 * @param[in] vector The three-dimensional vector whose squared length is to be
 *                   computed.
 * @return    The squared length of the three-dimensional vector.
 */
float absSq(const Vector3 &vector);

/**
 * @relates   Vector3
 * @brief     Computes the cross product of the specified three-dimensional
 *            vectors.
 * @param[in] vector1 The first vector with which the cross product should be
 *                    computed.
 * @param[in] vector2 The second vector with which the cross product should be
 *                    computed.
 * @return    The cross product of the two specified vectors.
 */
Vector3 cross(const Vector3 &vector1, const Vector3 &vector2);

/**
 * @relates   Vector3
 * @brief     Computes the normalization of the specified three-dimensiona
 *            vector.
 * @param[in] vector The three-dimensional vector whose normalization is to be
 *                   computed.
 * @return    The normalization of the three-dimensional vector.
 */
Vector3 normalize(const Vector3 &vector);
} /* namespace RVO3D */

#endif /* RVO3D_VECTOR3_H_ */
