/// @ref gtx_compatibility
/// @file glm/gtx/compatibility.hpp
///
/// @see core (dependence)
///
/// @defgroup gtx_compatibility GLM_GTX_compatibility
/// @ingroup gtx
///
/// Include <glm/gtx/compatibility.hpp> to use the features of this extension.
///
/// Provide functions to increase the compatibility with Cg and HLSL languages

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtc/quaternion.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_compatibility is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_compatibility extension included")
#	endif
#endif

#if GLM_COMPILER & GLM_COMPILER_VC
#	include <cfloat>
#elif GLM_COMPILER & GLM_COMPILER_GCC
#	include <cmath>
#	if(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
#		undef isfinite
#	endif
#endif//GLM_COMPILER

namespace glm
{
	/// @addtogroup gtx_compatibility
	/// @{

	template<typename T> GLM_FUNC_QUALIFIER T lerp(T x, T y, T a){return mix(x, y, a);}																					//!< \brief Returns x * (1.0 - a) + y * a, i.e., the linear blend of x and y using the floating-point value a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<2, T, Q> lerp(const vec<2, T, Q>& x, const vec<2, T, Q>& y, T a){return mix(x, y, a);}							//!< \brief Returns x * (1.0 - a) + y * a, i.e., the linear blend of x and y using the floating-point value a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)

	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<3, T, Q> lerp(const vec<3, T, Q>& x, const vec<3, T, Q>& y, T a){return mix(x, y, a);}							//!< \brief Returns x * (1.0 - a) + y * a, i.e., the linear blend of x and y using the floating-point value a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<4, T, Q> lerp(const vec<4, T, Q>& x, const vec<4, T, Q>& y, T a){return mix(x, y, a);}							//!< \brief Returns x * (1.0 - a) + y * a, i.e., the linear blend of x and y using the floating-point value a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<2, T, Q> lerp(const vec<2, T, Q>& x, const vec<2, T, Q>& y, const vec<2, T, Q>& a){return mix(x, y, a);}	//!< \brief Returns the component-wise result of x * (1.0 - a) + y * a, i.e., the linear blend of x and y using vector a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<3, T, Q> lerp(const vec<3, T, Q>& x, const vec<3, T, Q>& y, const vec<3, T, Q>& a){return mix(x, y, a);}	//!< \brief Returns the component-wise result of x * (1.0 - a) + y * a, i.e., the linear blend of x and y using vector a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<4, T, Q> lerp(const vec<4, T, Q>& x, const vec<4, T, Q>& y, const vec<4, T, Q>& a){return mix(x, y, a);}	//!< \brief Returns the component-wise result of x * (1.0 - a) + y * a, i.e., the linear blend of x and y using vector a. The value for a is not restricted to the range [0, 1]. (From GLM_GTX_compatibility)

	template<typename T> GLM_FUNC_QUALIFIER T saturate(T x){return clamp(x, T(0), T(1));}														//!< \brief Returns clamp(x, 0, 1) for each component in x. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<2, T, Q> saturate(const vec<2, T, Q>& x){return clamp(x, T(0), T(1));}					//!< \brief Returns clamp(x, 0, 1) for each component in x. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<3, T, Q> saturate(const vec<3, T, Q>& x){return clamp(x, T(0), T(1));}					//!< \brief Returns clamp(x, 0, 1) for each component in x. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<4, T, Q> saturate(const vec<4, T, Q>& x){return clamp(x, T(0), T(1));}					//!< \brief Returns clamp(x, 0, 1) for each component in x. (From GLM_GTX_compatibility)

	template<typename T> GLM_FUNC_QUALIFIER T atan2(T y, T x){return atan(y, x);}																//!< \brief Arc tangent. Returns an angle whose tangent is y/x. The signs of x and y are used to determine what quadrant the angle is in. The range of values returned by this function is [-PI, PI]. Results are undefined if x and y are both 0. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<2, T, Q> atan2(const vec<2, T, Q>& y, const vec<2, T, Q>& x){return atan(y, x);}	//!< \brief Arc tangent. Returns an angle whose tangent is y/x. The signs of x and y are used to determine what quadrant the angle is in. The range of values returned by this function is [-PI, PI]. Results are undefined if x and y are both 0. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<3, T, Q> atan2(const vec<3, T, Q>& y, const vec<3, T, Q>& x){return atan(y, x);}	//!< \brief Arc tangent. Returns an angle whose tangent is y/x. The signs of x and y are used to determine what quadrant the angle is in. The range of values returned by this function is [-PI, PI]. Results are undefined if x and y are both 0. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_QUALIFIER vec<4, T, Q> atan2(const vec<4, T, Q>& y, const vec<4, T, Q>& x){return atan(y, x);}	//!< \brief Arc tangent. Returns an angle whose tangent is y/x. The signs of x and y are used to determine what quadrant the angle is in. The range of values returned by this function is [-PI, PI]. Results are undefined if x and y are both 0. (From GLM_GTX_compatibility)

	template<typename genType> GLM_FUNC_DECL bool isfinite(genType const& x);											//!< \brief Test whether or not a scalar or each vector component is a finite value. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_DECL vec<1, bool, Q> isfinite(const vec<1, T, Q>& x);				//!< \brief Test whether or not a scalar or each vector component is a finite value. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_DECL vec<2, bool, Q> isfinite(const vec<2, T, Q>& x);				//!< \brief Test whether or not a scalar or each vector component is a finite value. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_DECL vec<3, bool, Q> isfinite(const vec<3, T, Q>& x);				//!< \brief Test whether or not a scalar or each vector component is a finite value. (From GLM_GTX_compatibility)
	template<typename T, qualifier Q> GLM_FUNC_DECL vec<4, bool, Q> isfinite(const vec<4, T, Q>& x);				//!< \brief Test whether or not a scalar or each vector component is a finite value. (From GLM_GTX_compatibility)

	typedef bool						bool1;			//!< \brief boolean type with 1 component. (From GLM_GTX_compatibility extension)
	typedef vec<2, bool, highp>			bool2;			//!< \brief boolean type with 2 components. (From GLM_GTX_compatibility extension)
	typedef vec<3, bool, highp>			bool3;			//!< \brief boolean type with 3 components. (From GLM_GTX_compatibility extension)
	typedef vec<4, bool, highp>			bool4;			//!< \brief boolean type with 4 components. (From GLM_GTX_compatibility extension)

	typedef bool						bool1x1;		//!< \brief boolean matrix with 1 x 1 component. (From GLM_GTX_compatibility extension)
	typedef mat<2, 2, bool, highp>		bool2x2;		//!< \brief boolean matrix with 2 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<2, 3, bool, highp>		bool2x3;		//!< \brief boolean matrix with 2 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<2, 4, bool, highp>		bool2x4;		//!< \brief boolean matrix with 2 x 4 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 2, bool, highp>		bool3x2;		//!< \brief boolean matrix with 3 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 3, bool, highp>		bool3x3;		//!< \brief boolean matrix with 3 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 4, bool, highp>		bool3x4;		//!< \brief boolean matrix with 3 x 4 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 2, bool, highp>		bool4x2;		//!< \brief boolean matrix with 4 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 3, bool, highp>		bool4x3;		//!< \brief boolean matrix with 4 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 4, bool, highp>		bool4x4;		//!< \brief boolean matrix with 4 x 4 components. (From GLM_GTX_compatibility extension)

	typedef int							int1;			//!< \brief integer vector with 1 component. (From GLM_GTX_compatibility extension)
	typedef vec<2, int, highp>			int2;			//!< \brief integer vector with 2 components. (From GLM_GTX_compatibility extension)
	typedef vec<3, int, highp>			int3;			//!< \brief integer vector with 3 components. (From GLM_GTX_compatibility extension)
	typedef vec<4, int, highp>			int4;			//!< \brief integer vector with 4 components. (From GLM_GTX_compatibility extension)

	typedef int							int1x1;			//!< \brief integer matrix with 1 component. (From GLM_GTX_compatibility extension)
	typedef mat<2, 2, int, highp>		int2x2;			//!< \brief integer matrix with 2 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<2, 3, int, highp>		int2x3;			//!< \brief integer matrix with 2 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<2, 4, int, highp>		int2x4;			//!< \brief integer matrix with 2 x 4 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 2, int, highp>		int3x2;			//!< \brief integer matrix with 3 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 3, int, highp>		int3x3;			//!< \brief integer matrix with 3 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 4, int, highp>		int3x4;			//!< \brief integer matrix with 3 x 4 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 2, int, highp>		int4x2;			//!< \brief integer matrix with 4 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 3, int, highp>		int4x3;			//!< \brief integer matrix with 4 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 4, int, highp>		int4x4;			//!< \brief integer matrix with 4 x 4 components. (From GLM_GTX_compatibility extension)

	typedef float						float1;			//!< \brief single-qualifier floating-point vector with 1 component. (From GLM_GTX_compatibility extension)
	typedef vec<2, float, highp>		float2;			//!< \brief single-qualifier floating-point vector with 2 components. (From GLM_GTX_compatibility extension)
	typedef vec<3, float, highp>		float3;			//!< \brief single-qualifier floating-point vector with 3 components. (From GLM_GTX_compatibility extension)
	typedef vec<4, float, highp>		float4;			//!< \brief single-qualifier floating-point vector with 4 components. (From GLM_GTX_compatibility extension)

	typedef float						float1x1;		//!< \brief single-qualifier floating-point matrix with 1 component. (From GLM_GTX_compatibility extension)
	typedef mat<2, 2, float, highp>		float2x2;		//!< \brief single-qualifier floating-point matrix with 2 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<2, 3, float, highp>		float2x3;		//!< \brief single-qualifier floating-point matrix with 2 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<2, 4, float, highp>		float2x4;		//!< \brief single-qualifier floating-point matrix with 2 x 4 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 2, float, highp>		float3x2;		//!< \brief single-qualifier floating-point matrix with 3 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 3, float, highp>		float3x3;		//!< \brief single-qualifier floating-point matrix with 3 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 4, float, highp>		float3x4;		//!< \brief single-qualifier floating-point matrix with 3 x 4 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 2, float, highp>		float4x2;		//!< \brief single-qualifier floating-point matrix with 4 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 3, float, highp>		float4x3;		//!< \brief single-qualifier floating-point matrix with 4 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 4, float, highp>		float4x4;		//!< \brief single-qualifier floating-point matrix with 4 x 4 components. (From GLM_GTX_compatibility extension)

	typedef double						double1;		//!< \brief double-qualifier floating-point vector with 1 component. (From GLM_GTX_compatibility extension)
	typedef vec<2, double, highp>		double2;		//!< \brief double-qualifier floating-point vector with 2 components. (From GLM_GTX_compatibility extension)
	typedef vec<3, double, highp>		double3;		//!< \brief double-qualifier floating-point vector with 3 components. (From GLM_GTX_compatibility extension)
	typedef vec<4, double, highp>		double4;		//!< \brief double-qualifier floating-point vector with 4 components. (From GLM_GTX_compatibility extension)

	typedef double						double1x1;		//!< \brief double-qualifier floating-point matrix with 1 component. (From GLM_GTX_compatibility extension)
	typedef mat<2, 2, double, highp>		double2x2;		//!< \brief double-qualifier floating-point matrix with 2 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<2, 3, double, highp>		double2x3;		//!< \brief double-qualifier floating-point matrix with 2 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<2, 4, double, highp>		double2x4;		//!< \brief double-qualifier floating-point matrix with 2 x 4 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 2, double, highp>		double3x2;		//!< \brief double-qualifier floating-point matrix with 3 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 3, double, highp>		double3x3;		//!< \brief double-qualifier floating-point matrix with 3 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<3, 4, double, highp>		double3x4;		//!< \brief double-qualifier floating-point matrix with 3 x 4 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 2, double, highp>		double4x2;		//!< \brief double-qualifier floating-point matrix with 4 x 2 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 3, double, highp>		double4x3;		//!< \brief double-qualifier floating-point matrix with 4 x 3 components. (From GLM_GTX_compatibility extension)
	typedef mat<4, 4, double, highp>		double4x4;		//!< \brief double-qualifier floating-point matrix with 4 x 4 components. (From GLM_GTX_compatibility extension)

	/// @}
}//namespace glm

#include "compatibility.inl"
