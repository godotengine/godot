/// @ref core
/// @file glm/glm.cpp

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/gtx/dual_quaternion.hpp>
#include <glm/gtc/vec1.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/ext/scalar_int_sized.hpp>
#include <glm/ext/scalar_uint_sized.hpp>
#include <glm/glm.hpp>

namespace glm
{
// tvec1 type explicit instantiation
template struct vec<1, uint8, lowp>;
template struct vec<1, uint16, lowp>;
template struct vec<1, uint32, lowp>;
template struct vec<1, uint64, lowp>;
template struct vec<1, int8, lowp>;
template struct vec<1, int16, lowp>;
template struct vec<1, int32, lowp>;
template struct vec<1, int64, lowp>;
template struct vec<1, float32, lowp>;
template struct vec<1, float64, lowp>;

template struct vec<1, uint8, mediump>;
template struct vec<1, uint16, mediump>;
template struct vec<1, uint32, mediump>;
template struct vec<1, uint64, mediump>;
template struct vec<1, int8, mediump>;
template struct vec<1, int16, mediump>;
template struct vec<1, int32, mediump>;
template struct vec<1, int64, mediump>;
template struct vec<1, float32, mediump>;
template struct vec<1, float64, mediump>;

template struct vec<1, uint8, highp>;
template struct vec<1, uint16, highp>;
template struct vec<1, uint32, highp>;
template struct vec<1, uint64, highp>;
template struct vec<1, int8, highp>;
template struct vec<1, int16, highp>;
template struct vec<1, int32, highp>;
template struct vec<1, int64, highp>;
template struct vec<1, float32, highp>;
template struct vec<1, float64, highp>;

// tvec2 type explicit instantiation
template struct vec<2, uint8, lowp>;
template struct vec<2, uint16, lowp>;
template struct vec<2, uint32, lowp>;
template struct vec<2, uint64, lowp>;
template struct vec<2, int8, lowp>;
template struct vec<2, int16, lowp>;
template struct vec<2, int32, lowp>;
template struct vec<2, int64, lowp>;
template struct vec<2, float32, lowp>;
template struct vec<2, float64, lowp>;

template struct vec<2, uint8, mediump>;
template struct vec<2, uint16, mediump>;
template struct vec<2, uint32, mediump>;
template struct vec<2, uint64, mediump>;
template struct vec<2, int8, mediump>;
template struct vec<2, int16, mediump>;
template struct vec<2, int32, mediump>;
template struct vec<2, int64, mediump>;
template struct vec<2, float32, mediump>;
template struct vec<2, float64, mediump>;

template struct vec<2, uint8, highp>;
template struct vec<2, uint16, highp>;
template struct vec<2, uint32, highp>;
template struct vec<2, uint64, highp>;
template struct vec<2, int8, highp>;
template struct vec<2, int16, highp>;
template struct vec<2, int32, highp>;
template struct vec<2, int64, highp>;
template struct vec<2, float32, highp>;
template struct vec<2, float64, highp>;

// tvec3 type explicit instantiation
template struct vec<3, uint8, lowp>;
template struct vec<3, uint16, lowp>;
template struct vec<3, uint32, lowp>;
template struct vec<3, uint64, lowp>;
template struct vec<3, int8, lowp>;
template struct vec<3, int16, lowp>;
template struct vec<3, int32, lowp>;
template struct vec<3, int64, lowp>;
template struct vec<3, float32, lowp>;
template struct vec<3, float64, lowp>;

template struct vec<3, uint8, mediump>;
template struct vec<3, uint16, mediump>;
template struct vec<3, uint32, mediump>;
template struct vec<3, uint64, mediump>;
template struct vec<3, int8, mediump>;
template struct vec<3, int16, mediump>;
template struct vec<3, int32, mediump>;
template struct vec<3, int64, mediump>;
template struct vec<3, float32, mediump>;
template struct vec<3, float64, mediump>;

template struct vec<3, uint8, highp>;
template struct vec<3, uint16, highp>;
template struct vec<3, uint32, highp>;
template struct vec<3, uint64, highp>;
template struct vec<3, int8, highp>;
template struct vec<3, int16, highp>;
template struct vec<3, int32, highp>;
template struct vec<3, int64, highp>;
template struct vec<3, float32, highp>;
template struct vec<3, float64, highp>;

// tvec4 type explicit instantiation
template struct vec<4, uint8, lowp>;
template struct vec<4, uint16, lowp>;
template struct vec<4, uint32, lowp>;
template struct vec<4, uint64, lowp>;
template struct vec<4, int8, lowp>;
template struct vec<4, int16, lowp>;
template struct vec<4, int32, lowp>;
template struct vec<4, int64, lowp>;
template struct vec<4, float32, lowp>;
template struct vec<4, float64, lowp>;

template struct vec<4, uint8, mediump>;
template struct vec<4, uint16, mediump>;
template struct vec<4, uint32, mediump>;
template struct vec<4, uint64, mediump>;
template struct vec<4, int8, mediump>;
template struct vec<4, int16, mediump>;
template struct vec<4, int32, mediump>;
template struct vec<4, int64, mediump>;
template struct vec<4, float32, mediump>;
template struct vec<4, float64, mediump>;

template struct vec<4, uint8, highp>;
template struct vec<4, uint16, highp>;
template struct vec<4, uint32, highp>;
template struct vec<4, uint64, highp>;
template struct vec<4, int8, highp>;
template struct vec<4, int16, highp>;
template struct vec<4, int32, highp>;
template struct vec<4, int64, highp>;
template struct vec<4, float32, highp>;
template struct vec<4, float64, highp>;

// tmat2x2 type explicit instantiation
template struct mat<2, 2, float32, lowp>;
template struct mat<2, 2, float64, lowp>;

template struct mat<2, 2, float32, mediump>;
template struct mat<2, 2, float64, mediump>;

template struct mat<2, 2, float32, highp>;
template struct mat<2, 2, float64, highp>;

// tmat2x3 type explicit instantiation
template struct mat<2, 3, float32, lowp>;
template struct mat<2, 3, float64, lowp>;

template struct mat<2, 3, float32, mediump>;
template struct mat<2, 3, float64, mediump>;

template struct mat<2, 3, float32, highp>;
template struct mat<2, 3, float64, highp>;

// tmat2x4 type explicit instantiation
template struct mat<2, 4, float32, lowp>;
template struct mat<2, 4, float64, lowp>;

template struct mat<2, 4, float32, mediump>;
template struct mat<2, 4, float64, mediump>;

template struct mat<2, 4, float32, highp>;
template struct mat<2, 4, float64, highp>;

// tmat3x2 type explicit instantiation
template struct mat<3, 2, float32, lowp>;
template struct mat<3, 2, float64, lowp>;

template struct mat<3, 2, float32, mediump>;
template struct mat<3, 2, float64, mediump>;

template struct mat<3, 2, float32, highp>;
template struct mat<3, 2, float64, highp>;

// tmat3x3 type explicit instantiation
template struct mat<3, 3, float32, lowp>;
template struct mat<3, 3, float64, lowp>;

template struct mat<3, 3, float32, mediump>;
template struct mat<3, 3, float64, mediump>;

template struct mat<3, 3, float32, highp>;
template struct mat<3, 3, float64, highp>;

// tmat3x4 type explicit instantiation
template struct mat<3, 4, float32, lowp>;
template struct mat<3, 4, float64, lowp>;

template struct mat<3, 4, float32, mediump>;
template struct mat<3, 4, float64, mediump>;

template struct mat<3, 4, float32, highp>;
template struct mat<3, 4, float64, highp>;

// tmat4x2 type explicit instantiation
template struct mat<4, 2, float32, lowp>;
template struct mat<4, 2, float64, lowp>;

template struct mat<4, 2, float32, mediump>;
template struct mat<4, 2, float64, mediump>;

template struct mat<4, 2, float32, highp>;
template struct mat<4, 2, float64, highp>;

// tmat4x3 type explicit instantiation
template struct mat<4, 3, float32, lowp>;
template struct mat<4, 3, float64, lowp>;

template struct mat<4, 3, float32, mediump>;
template struct mat<4, 3, float64, mediump>;

template struct mat<4, 3, float32, highp>;
template struct mat<4, 3, float64, highp>;

// tmat4x4 type explicit instantiation
template struct mat<4, 4, float32, lowp>;
template struct mat<4, 4, float64, lowp>;

template struct mat<4, 4, float32, mediump>;
template struct mat<4, 4, float64, mediump>;

template struct mat<4, 4, float32, highp>;
template struct mat<4, 4, float64, highp>;

// tquat type explicit instantiation
template struct qua<float32, lowp>;
template struct qua<float64, lowp>;

template struct qua<float32, mediump>;
template struct qua<float64, mediump>;

template struct qua<float32, highp>;
template struct qua<float64, highp>;

//tdualquat type explicit instantiation
template struct tdualquat<float32, lowp>;
template struct tdualquat<float64, lowp>;

template struct tdualquat<float32, mediump>;
template struct tdualquat<float64, mediump>;

template struct tdualquat<float32, highp>;
template struct tdualquat<float64, highp>;

}//namespace glm

