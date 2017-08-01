/// @file Math utility functions used in inverse dynamics library.
///	   Defined here as they may not be provided by the math library.

#ifndef IDMATH_HPP_
#define IDMATH_HPP_
#include "IDConfig.hpp"

namespace btInverseDynamics {
/// set all elements to zero
void setZero(vec3& v);
/// set all elements to zero
void setZero(vecx& v);
/// set all elements to zero
void setZero(mat33& m);

/// return maximum absolute value
idScalar maxAbs(const vecx& v);
#ifndef ID_LINEAR_MATH_USE_EIGEN
/// return maximum absolute value
idScalar maxAbs(const vec3& v);
#endif  //ID_LINEAR_MATH_USE_EIGEN

#if (defined BT_ID_HAVE_MAT3X)
idScalar maxAbsMat3x(const mat3x& m);
void setZero(mat3x&m);
// define math functions on mat3x here to avoid allocations in operators.
void mul(const mat33&a, const mat3x&b, mat3x* result);
void add(const mat3x&a, const mat3x&b, mat3x* result);
void sub(const mat3x&a, const mat3x&b, mat3x* result);
#endif

/// get offset vector & transform matrix from DH parameters
/// TODO: add documentation
void getVecMatFromDH(idScalar theta, idScalar d, idScalar a, idScalar alpha, vec3* r, mat33* T);

/// Check if a 3x3 matrix is positive definite
/// @param m a 3x3 matrix
/// @return true if m>0, false otherwise
bool isPositiveDefinite(const mat33& m);

/// Check if a 3x3 matrix is positive semi definite
/// @param m a 3x3 matrix
/// @return true if m>=0, false otherwise
bool isPositiveSemiDefinite(const mat33& m);
/// Check if a 3x3 matrix is positive semi definite within numeric limits
/// @param m a 3x3 matrix
/// @return true if m>=-eps, false otherwise
bool isPositiveSemiDefiniteFuzzy(const mat33& m);

/// Determinant of 3x3 matrix
/// NOTE: implemented here for portability, as determinant operation
///	   will be implemented differently for various matrix/vector libraries
/// @param m a 3x3 matrix
/// @return det(m)
idScalar determinant(const mat33& m);

/// Test if a 3x3 matrix satisfies some properties of inertia matrices
/// @param I a 3x3 matrix
/// @param index body index (for error messages)
/// @param has_fixed_joint: if true, positive semi-definite matrices are accepted
/// @return true if I satisfies inertia matrix properties, false otherwise.
bool isValidInertiaMatrix(const mat33& I, int index, bool has_fixed_joint);

/// Check if a 3x3 matrix is a valid transform (rotation) matrix
/// @param m a 3x3 matrix
/// @return true if m is a rotation matrix, false otherwise
bool isValidTransformMatrix(const mat33& m);
/// Transform matrix from parent to child frame,
/// when the child frame is rotated about @param axis by @angle
/// (mathematically positive)
/// @param axis the axis of rotation
/// @param angle rotation angle
/// @param T pointer to transform matrix
void bodyTParentFromAxisAngle(const vec3& axis, const idScalar& angle, mat33* T);

/// Check if this is a unit vector
/// @param vector
/// @return true if |vector|=1 within numeric limits
bool isUnitVector(const vec3& vector);

/// @input a vector in R^3
/// @returns corresponding spin tensor
mat33 tildeOperator(const vec3& v);
/// @param alpha angle in radians
/// @returns transform matrix for ratation with @param alpha about x-axis
mat33 transformX(const idScalar& alpha);
/// @param beta angle in radians
/// @returns transform matrix for ratation with @param beta about y-axis
mat33 transformY(const idScalar& beta);
/// @param gamma angle in radians
/// @returns transform matrix for ratation with @param gamma about z-axis
mat33 transformZ(const idScalar& gamma);
///calculate rpy angles (x-y-z Euler angles) from a given rotation matrix
/// @param rot rotation matrix
/// @returns x-y-z Euler angles
vec3 rpyFromMatrix(const mat33&rot);
}
#endif  // IDMATH_HPP_
