/// @ref gtx_easing
/// @file glm/gtx/easing.hpp
/// @author Robert Chisholm
///
/// @see core (dependence)
///
/// @defgroup gtx_easing GLM_GTX_easing
/// @ingroup gtx
///
/// Include <glm/gtx/easing.hpp> to use the features of this extension.
///
/// Easing functions for animations and transitions
/// All functions take a parameter x in the range [0.0,1.0]
///
/// Based on the AHEasing project of Warren Moore (https://github.com/warrenm/AHEasing)

#pragma once

// Dependency:
#include "../glm.hpp"
#include "../gtc/constants.hpp"
#include "../detail/qualifier.hpp"

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	ifndef GLM_ENABLE_EXPERIMENTAL
#		pragma message("GLM: GLM_GTX_easing is an experimental extension and may change in the future. Use #define GLM_ENABLE_EXPERIMENTAL before including it, if you really want to use it.")
#	else
#		pragma message("GLM: GLM_GTX_easing extension included")
#	endif
#endif

namespace glm{
	/// @addtogroup gtx_easing
	/// @{

	/// Modelled after the line y = x
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType linearInterpolation(genType const & a);

	/// Modelled after the parabola y = x^2
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType quadraticEaseIn(genType const & a);

	/// Modelled after the parabola y = -x^2 + 2x
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType quadraticEaseOut(genType const & a);

	/// Modelled after the piecewise quadratic
	/// y = (1/2)((2x)^2)				; [0, 0.5)
	/// y = -(1/2)((2x-1)*(2x-3) - 1)	; [0.5, 1]
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType quadraticEaseInOut(genType const & a);

	/// Modelled after the cubic y = x^3
	template <typename genType>
	GLM_FUNC_DECL genType cubicEaseIn(genType const & a);

	/// Modelled after the cubic y = (x - 1)^3 + 1
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType cubicEaseOut(genType const & a);

	/// Modelled after the piecewise cubic
	/// y = (1/2)((2x)^3)		; [0, 0.5)
	/// y = (1/2)((2x-2)^3 + 2)	; [0.5, 1]
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType cubicEaseInOut(genType const & a);

	/// Modelled after the quartic x^4
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType quarticEaseIn(genType const & a);

	/// Modelled after the quartic y = 1 - (x - 1)^4
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType quarticEaseOut(genType const & a);

	/// Modelled after the piecewise quartic
	/// y = (1/2)((2x)^4)			; [0, 0.5)
	/// y = -(1/2)((2x-2)^4 - 2)	; [0.5, 1]
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType quarticEaseInOut(genType const & a);

	/// Modelled after the quintic y = x^5
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType quinticEaseIn(genType const & a);

	/// Modelled after the quintic y = (x - 1)^5 + 1
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType quinticEaseOut(genType const & a);

	/// Modelled after the piecewise quintic
	/// y = (1/2)((2x)^5)		; [0, 0.5)
	/// y = (1/2)((2x-2)^5 + 2) ; [0.5, 1]
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType quinticEaseInOut(genType const & a);

	/// Modelled after quarter-cycle of sine wave
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType sineEaseIn(genType const & a);

	/// Modelled after quarter-cycle of sine wave (different phase)
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType sineEaseOut(genType const & a);

	/// Modelled after half sine wave
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType sineEaseInOut(genType const & a);

	/// Modelled after shifted quadrant IV of unit circle
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType circularEaseIn(genType const & a);

	/// Modelled after shifted quadrant II of unit circle
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType circularEaseOut(genType const & a);

	/// Modelled after the piecewise circular function
	/// y = (1/2)(1 - sqrt(1 - 4x^2))			; [0, 0.5)
	/// y = (1/2)(sqrt(-(2x - 3)*(2x - 1)) + 1) ; [0.5, 1]
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType circularEaseInOut(genType const & a);

	/// Modelled after the exponential function y = 2^(10(x - 1))
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType exponentialEaseIn(genType const & a);

	/// Modelled after the exponential function y = -2^(-10x) + 1
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType exponentialEaseOut(genType const & a);

	/// Modelled after the piecewise exponential
	/// y = (1/2)2^(10(2x - 1))			; [0,0.5)
	/// y = -(1/2)*2^(-10(2x - 1))) + 1 ; [0.5,1]
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType exponentialEaseInOut(genType const & a);

	/// Modelled after the damped sine wave y = sin(13pi/2*x)*pow(2, 10 * (x - 1))
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType elasticEaseIn(genType const & a);

	/// Modelled after the damped sine wave y = sin(-13pi/2*(x + 1))*pow(2, -10x) + 1
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType elasticEaseOut(genType const & a);

	/// Modelled after the piecewise exponentially-damped sine wave:
	/// y = (1/2)*sin(13pi/2*(2*x))*pow(2, 10 * ((2*x) - 1))		; [0,0.5)
	/// y = (1/2)*(sin(-13pi/2*((2x-1)+1))*pow(2,-10(2*x-1)) + 2)	; [0.5, 1]
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType elasticEaseInOut(genType const & a);

	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType backEaseIn(genType const& a);

	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType backEaseOut(genType const& a);

	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType backEaseInOut(genType const& a);

	/// @param a parameter
	/// @param o Optional overshoot modifier
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType backEaseIn(genType const& a, genType const& o);

	/// @param a parameter
	/// @param o Optional overshoot modifier
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType backEaseOut(genType const& a, genType const& o);

	/// @param a parameter
	/// @param o Optional overshoot modifier
	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType backEaseInOut(genType const& a, genType const& o);

	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType bounceEaseIn(genType const& a);

	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType bounceEaseOut(genType const& a);

	/// @see gtx_easing
	template <typename genType>
	GLM_FUNC_DECL genType bounceEaseInOut(genType const& a);

	/// @}
}//namespace glm

#include "easing.inl"
