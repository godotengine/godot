/// @ref core
/// @file glm/detail/type_mat2x4.hpp

#pragma once

#include "type_vec2.hpp"
#include "type_vec4.hpp"
#include <limits>
#include <cstddef>

namespace glm
{
	template<typename T, qualifier Q>
	struct mat<2, 4, T, Q>
	{
		typedef vec<4, T, Q> col_type;
		typedef vec<2, T, Q> row_type;
		typedef mat<2, 4, T, Q> type;
		typedef mat<4, 2, T, Q> transpose_type;
		typedef T value_type;

	private:
		col_type value[2];

	public:
		// -- Accesses --

		typedef length_t length_type;
		GLM_FUNC_DECL static GLM_CONSTEXPR length_type length() { return 2; }

		GLM_FUNC_DECL GLM_CONSTEXPR col_type & operator[](length_type i) GLM_NOEXCEPT;
		GLM_FUNC_DECL GLM_CONSTEXPR col_type const& operator[](length_type i) const GLM_NOEXCEPT;

		// -- Constructors --

		GLM_DEFAULTED_DEFAULT_CTOR_DECL GLM_CONSTEXPR mat() GLM_DEFAULT_CTOR;
		template<qualifier P>
		GLM_FUNC_DECL GLM_CONSTEXPR mat(mat<2, 4, T, P> const& m);

		GLM_FUNC_DECL explicit GLM_CONSTEXPR mat(T scalar);
		GLM_FUNC_DECL GLM_CONSTEXPR mat(
			T x0, T y0, T z0, T w0,
			T x1, T y1, T z1, T w1);
		GLM_FUNC_DECL GLM_CONSTEXPR mat(
			col_type const& v0,
			col_type const& v1);

		// -- Conversions --

		template<
			typename X1, typename Y1, typename Z1, typename W1,
			typename X2, typename Y2, typename Z2, typename W2>
		GLM_FUNC_DECL GLM_CONSTEXPR mat(
			X1 x1, Y1 y1, Z1 z1, W1 w1,
			X2 x2, Y2 y2, Z2 z2, W2 w2);

		template<typename U, typename V>
		GLM_FUNC_DECL GLM_CONSTEXPR mat(
			vec<4, U, Q> const& v1,
			vec<4, V, Q> const& v2);

		// -- Matrix conversions --

		template<typename U, qualifier P>
		GLM_FUNC_DECL GLM_EXPLICIT GLM_CONSTEXPR mat(mat<2, 4, U, P> const& m);

		GLM_FUNC_DECL GLM_EXPLICIT GLM_CONSTEXPR mat(mat<2, 2, T, Q> const& x);
		GLM_FUNC_DECL GLM_EXPLICIT GLM_CONSTEXPR mat(mat<3, 3, T, Q> const& x);
		GLM_FUNC_DECL GLM_EXPLICIT GLM_CONSTEXPR mat(mat<4, 4, T, Q> const& x);
		GLM_FUNC_DECL GLM_EXPLICIT GLM_CONSTEXPR mat(mat<2, 3, T, Q> const& x);
		GLM_FUNC_DECL GLM_EXPLICIT GLM_CONSTEXPR mat(mat<3, 2, T, Q> const& x);
		GLM_FUNC_DECL GLM_EXPLICIT GLM_CONSTEXPR mat(mat<3, 4, T, Q> const& x);
		GLM_FUNC_DECL GLM_EXPLICIT GLM_CONSTEXPR mat(mat<4, 2, T, Q> const& x);
		GLM_FUNC_DECL GLM_EXPLICIT GLM_CONSTEXPR mat(mat<4, 3, T, Q> const& x);

		// -- Unary arithmetic operators --

		template<typename U>
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> & operator=(mat<2, 4, U, Q> const& m);
		template<typename U>
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> & operator+=(U s);
		template<typename U>
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> & operator+=(mat<2, 4, U, Q> const& m);
		template<typename U>
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> & operator-=(U s);
		template<typename U>
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> & operator-=(mat<2, 4, U, Q> const& m);
		template<typename U>
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> & operator*=(U s);
		template<typename U>
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> & operator/=(U s);

		// -- Increment and decrement operators --

		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> & operator++ ();
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> & operator-- ();
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator++(int);
		GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator--(int);
	};

	// -- Unary operators --

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator+(mat<2, 4, T, Q> const& m);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator-(mat<2, 4, T, Q> const& m);

	// -- Binary operators --

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator+(mat<2, 4, T, Q> const& m, T scalar);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator+(mat<2, 4, T, Q> const& m1, mat<2, 4, T, Q> const& m2);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator-(mat<2, 4, T, Q> const& m, T scalar);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator-(mat<2, 4, T, Q> const& m1, mat<2, 4, T, Q> const& m2);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator*(mat<2, 4, T, Q> const& m, T scalar);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator*(T scalar, mat<2, 4, T, Q> const& m);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR typename mat<2, 4, T, Q>::col_type operator*(mat<2, 4, T, Q> const& m, typename mat<2, 4, T, Q>::row_type const& v);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR typename mat<2, 4, T, Q>::row_type operator*(typename mat<2, 4, T, Q>::col_type const& v, mat<2, 4, T, Q> const& m);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<4, 4, T, Q> operator*(mat<2, 4, T, Q> const& m1, mat<4, 2, T, Q> const& m2);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator*(mat<2, 4, T, Q> const& m1, mat<2, 2, T, Q> const& m2);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<3, 4, T, Q> operator*(mat<2, 4, T, Q> const& m1, mat<3, 2, T, Q> const& m2);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator/(mat<2, 4, T, Q> const& m, T scalar);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR mat<2, 4, T, Q> operator/(T scalar, mat<2, 4, T, Q> const& m);

	// -- Boolean operators --

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR bool operator==(mat<2, 4, T, Q> const& m1, mat<2, 4, T, Q> const& m2);

	template<typename T, qualifier Q>
	GLM_FUNC_DECL GLM_CONSTEXPR bool operator!=(mat<2, 4, T, Q> const& m1, mat<2, 4, T, Q> const& m2);
}//namespace glm

#ifndef GLM_EXTERNAL_TEMPLATE
#include "type_mat2x4.inl"
#endif
