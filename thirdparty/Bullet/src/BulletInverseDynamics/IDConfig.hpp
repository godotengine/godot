///@file Configuration for Inverse Dynamics Library,
///	  such as choice of linear algebra library and underlying scalar type
#ifndef IDCONFIG_HPP_
#define IDCONFIG_HPP_

// If true, enable jacobian calculations.
// This adds a 3xN matrix to every body, + 2 3-Vectors.
// so it is not advised for large systems if it is not absolutely necessary.
// Also, this is not required for standard inverse dynamics calculations.
// Will only work with vector math libraries that support 3xN matrices.
#define BT_ID_WITH_JACOBIANS

// If we have a custom configuration, compile without using other parts of bullet.
#ifdef BT_CUSTOM_INVERSE_DYNAMICS_CONFIG_H
#include <cmath>
#define BT_ID_WO_BULLET
#define BT_ID_SQRT(x) std::sqrt(x)
#define BT_ID_FABS(x) std::fabs(x)
#define BT_ID_COS(x) std::cos(x)
#define BT_ID_SIN(x) std::sin(x)
#define BT_ID_ATAN2(x, y) std::atan2(x, y)
#define BT_ID_POW(x, y) std::pow(x, y)
#define BT_ID_SNPRINTF snprintf
#define BT_ID_PI M_PI
#define BT_ID_USE_DOUBLE_PRECISION
#else
#define BT_ID_SQRT(x) btSqrt(x)
#define BT_ID_FABS(x) btFabs(x)
#define BT_ID_COS(x) btCos(x)
#define BT_ID_SIN(x) btSin(x)
#define BT_ID_ATAN2(x, y) btAtan2(x, y)
#define BT_ID_POW(x, y) btPow(x, y)
#define BT_ID_PI SIMD_PI
#ifdef _WIN32
	#define BT_ID_SNPRINTF _snprintf
#else
	#define BT_ID_SNPRINTF snprintf
#endif //
#endif
// error messages
#include "IDErrorMessages.hpp"

#ifdef BT_CUSTOM_INVERSE_DYNAMICS_CONFIG_H
/*
#include "IDConfigEigen.hpp"
#include "IDConfigBuiltin.hpp"
*/
#define INVDYN_INCLUDE_HELPER_2(x) #x
#define INVDYN_INCLUDE_HELPER(x) INVDYN_INCLUDE_HELPER_2(x)
#include INVDYN_INCLUDE_HELPER(BT_CUSTOM_INVERSE_DYNAMICS_CONFIG_H)
#ifndef btInverseDynamics
#error "custom inverse dynamics config, but no custom namespace defined"
#endif

#define BT_ID_MAX(a,b) std::max(a,b)
#define BT_ID_MIN(a,b) std::min(a,b)

#else
#define btInverseDynamics btInverseDynamicsBullet3
// Use default configuration with bullet's types
// Use the same scalar type as rest of bullet library
#include "LinearMath/btScalar.h"
typedef btScalar idScalar;
#include "LinearMath/btMinMax.h"
#define BT_ID_MAX(a,b) btMax(a,b)
#define BT_ID_MIN(a,b) btMin(a,b)

#ifdef BT_USE_DOUBLE_PRECISION
#define BT_ID_USE_DOUBLE_PRECISION
#endif

#ifndef BT_USE_INVERSE_DYNAMICS_WITH_BULLET2


// use bullet types for arrays and array indices
#include "Bullet3Common/b3AlignedObjectArray.h"
// this is to make it work with C++2003, otherwise we could do this:
// template <typename T>
// using idArray = b3AlignedObjectArray<T>;
template <typename T>
struct idArray {
	typedef b3AlignedObjectArray<T> type;
};
typedef int idArrayIdx;
#define ID_DECLARE_ALIGNED_ALLOCATOR() B3_DECLARE_ALIGNED_ALLOCATOR()

#else // BT_USE_INVERSE_DYNAMICS_WITH_BULLET2

#include "LinearMath/btAlignedObjectArray.h"
template <typename T>
struct idArray {
	typedef btAlignedObjectArray<T> type;
};
typedef int idArrayIdx;
#define ID_DECLARE_ALIGNED_ALLOCATOR() BT_DECLARE_ALIGNED_ALLOCATOR()

#endif // BT_USE_INVERSE_DYNAMICS_WITH_BULLET2


// use bullet's allocator functions
#define idMalloc btAllocFunc
#define idFree btFreeFunc

#define ID_LINEAR_MATH_USE_BULLET
#include "details/IDLinearMathInterface.hpp"
#endif
#endif
