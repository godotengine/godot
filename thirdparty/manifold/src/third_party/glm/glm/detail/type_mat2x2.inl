#include "../matrix.hpp"

namespace glm
{
	// -- Constructors --

#	if GLM_CONFIG_DEFAULTED_DEFAULT_CTOR == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat()
#			if GLM_CONFIG_CTOR_INIT == GLM_CTOR_INITIALIZER_LIST
				: value{col_type(1, 0), col_type(0, 1)}
#			endif
		{
#			if GLM_CONFIG_CTOR_INIT == GLM_CTOR_INITIALISATION
				this->value[0] = col_type(1, 0);
				this->value[1] = col_type(0, 1);
#			endif
		}
#	endif

	template<typename T, qualifier Q>
	template<qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<2, 2, T, P> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = m[0];
			this->value[1] = m[1];
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(T scalar)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(scalar, 0), col_type(0, scalar)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(scalar, 0);
			this->value[1] = col_type(0, scalar);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat
	(
		T const& x0, T const& y0,
		T const& x1, T const& y1
	)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(x0, y0), col_type(x1, y1)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(x0, y0);
			this->value[1] = col_type(x1, y1);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(col_type const& v0, col_type const& v1)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{v0, v1}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = v0;
			this->value[1] = v1;
#		endif
	}

	// -- Conversion constructors --

	template<typename T, qualifier Q>
	template<typename X1, typename Y1, typename X2, typename Y2>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat
	(
		X1 const& x1, Y1 const& y1,
		X2 const& x2, Y2 const& y2
	)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(static_cast<T>(x1), value_type(y1)), col_type(static_cast<T>(x2), value_type(y2)) }
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(static_cast<T>(x1), value_type(y1));
			this->value[1] = col_type(static_cast<T>(x2), value_type(y2));
#		endif
	}

	template<typename T, qualifier Q>
	template<typename V1, typename V2>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(vec<2, V1, Q> const& v1, vec<2, V2, Q> const& v2)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(v1), col_type(v2)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(v1);
			this->value[1] = col_type(v2);
#		endif
	}

	// -- mat2x2 matrix conversions --

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<2, 2, U, P> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<3, 3, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<4, 4, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<2, 3, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<3, 2, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<2, 4, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<4, 2, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<3, 4, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<2, 2, T, Q>::mat(mat<4, 3, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
#		endif
	}

	// -- Accesses --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename mat<2, 2, T, Q>::col_type& mat<2, 2, T, Q>::operator[](typename mat<2, 2, T, Q>::length_type i)
	{
		assert(i < this->length());
		return this->value[i];
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename mat<2, 2, T, Q>::col_type const& mat<2, 2, T, Q>::operator[](typename mat<2, 2, T, Q>::length_type i) const
	{
		assert(i < this->length());
		return this->value[i];
	}

	// -- Unary updatable operators --

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator=(mat<2, 2, U, Q> const& m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator+=(U scalar)
	{
		this->value[0] += scalar;
		this->value[1] += scalar;
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator+=(mat<2, 2, U, Q> const& m)
	{
		this->value[0] += m[0];
		this->value[1] += m[1];
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator-=(U scalar)
	{
		this->value[0] -= scalar;
		this->value[1] -= scalar;
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator-=(mat<2, 2, U, Q> const& m)
	{
		this->value[0] -= m[0];
		this->value[1] -= m[1];
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator*=(U scalar)
	{
		this->value[0] *= scalar;
		this->value[1] *= scalar;
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator*=(mat<2, 2, U, Q> const& m)
	{
		return (*this = *this * m);
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator/=(U scalar)
	{
		this->value[0] /= scalar;
		this->value[1] /= scalar;
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator/=(mat<2, 2, U, Q> const& m)
	{
		return *this *= inverse(m);
	}

	// -- Increment and decrement operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator++()
	{
		++this->value[0];
		++this->value[1];
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q>& mat<2, 2, T, Q>::operator--()
	{
		--this->value[0];
		--this->value[1];
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> mat<2, 2, T, Q>::operator++(int)
	{
		mat<2, 2, T, Q> Result(*this);
		++*this;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> mat<2, 2, T, Q>::operator--(int)
	{
		mat<2, 2, T, Q> Result(*this);
		--*this;
		return Result;
	}

	// -- Unary arithmetic operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator+(mat<2, 2, T, Q> const& m)
	{
		return m;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator-(mat<2, 2, T, Q> const& m)
	{
		return mat<2, 2, T, Q>(
			-m[0],
			-m[1]);
	}

	// -- Binary arithmetic operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator+(mat<2, 2, T, Q> const& m, T scalar)
	{
		return mat<2, 2, T, Q>(
			m[0] + scalar,
			m[1] + scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator+(T scalar, mat<2, 2, T, Q> const& m)
	{
		return mat<2, 2, T, Q>(
			m[0] + scalar,
			m[1] + scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator+(mat<2, 2, T, Q> const& m1, mat<2, 2, T, Q> const& m2)
	{
		return mat<2, 2, T, Q>(
			m1[0] + m2[0],
			m1[1] + m2[1]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator-(mat<2, 2, T, Q> const& m, T scalar)
	{
		return mat<2, 2, T, Q>(
			m[0] - scalar,
			m[1] - scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator-(T scalar, mat<2, 2, T, Q> const& m)
	{
		return mat<2, 2, T, Q>(
			scalar - m[0],
			scalar - m[1]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator-(mat<2, 2, T, Q> const& m1, mat<2, 2, T, Q> const& m2)
	{
		return mat<2, 2, T, Q>(
			m1[0] - m2[0],
			m1[1] - m2[1]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator*(mat<2, 2, T, Q> const& m, T scalar)
	{
		return mat<2, 2, T, Q>(
			m[0] * scalar,
			m[1] * scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator*(T scalar, mat<2, 2, T, Q> const& m)
	{
		return mat<2, 2, T, Q>(
			m[0] * scalar,
			m[1] * scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename mat<2, 2, T, Q>::col_type operator*
	(
		mat<2, 2, T, Q> const& m,
		typename mat<2, 2, T, Q>::row_type const& v
	)
	{
		return vec<2, T, Q>(
			m[0][0] * v.x + m[1][0] * v.y,
			m[0][1] * v.x + m[1][1] * v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename mat<2, 2, T, Q>::row_type operator*
	(
		typename mat<2, 2, T, Q>::col_type const& v,
		mat<2, 2, T, Q> const& m
	)
	{
		return vec<2, T, Q>(
			v.x * m[0][0] + v.y * m[0][1],
			v.x * m[1][0] + v.y * m[1][1]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator*(mat<2, 2, T, Q> const& m1, mat<2, 2, T, Q> const& m2)
	{
		return mat<2, 2, T, Q>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 2, T, Q> operator*(mat<2, 2, T, Q> const& m1, mat<3, 2, T, Q> const& m2)
	{
		return mat<3, 2, T, Q>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1],
			m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1],
			m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 2, T, Q> operator*(mat<2, 2, T, Q> const& m1, mat<4, 2, T, Q> const& m2)
	{
		return mat<4, 2, T, Q>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1],
			m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1],
			m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1],
			m1[0][0] * m2[3][0] + m1[1][0] * m2[3][1],
			m1[0][1] * m2[3][0] + m1[1][1] * m2[3][1]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator/(mat<2, 2, T, Q> const& m, T scalar)
	{
		return mat<2, 2, T, Q>(
			m[0] / scalar,
			m[1] / scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator/(T scalar, mat<2, 2, T, Q> const& m)
	{
		return mat<2, 2, T, Q>(
			scalar / m[0],
			scalar / m[1]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename mat<2, 2, T, Q>::col_type operator/(mat<2, 2, T, Q> const& m, typename mat<2, 2, T, Q>::row_type const& v)
	{
		return inverse(m) * v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename mat<2, 2, T, Q>::row_type operator/(typename mat<2, 2, T, Q>::col_type const& v, mat<2, 2, T, Q> const& m)
	{
		return v *  inverse(m);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 2, T, Q> operator/(mat<2, 2, T, Q> const& m1, mat<2, 2, T, Q> const& m2)
	{
		mat<2, 2, T, Q> m1_copy(m1);
		return m1_copy /= m2;
	}

	// -- Boolean operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool operator==(mat<2, 2, T, Q> const& m1, mat<2, 2, T, Q> const& m2)
	{
		return (m1[0] == m2[0]) && (m1[1] == m2[1]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool operator!=(mat<2, 2, T, Q> const& m1, mat<2, 2, T, Q> const& m2)
	{
		return (m1[0] != m2[0]) || (m1[1] != m2[1]);
	}
} //namespace glm
