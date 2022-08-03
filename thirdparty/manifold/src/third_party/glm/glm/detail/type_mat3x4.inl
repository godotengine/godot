namespace glm
{
	// -- Constructors --

#	if GLM_CONFIG_DEFAULTED_DEFAULT_CTOR == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat()
#			if GLM_CONFIG_CTOR_INIT == GLM_CTOR_INITIALIZER_LIST
				: value{col_type(1, 0, 0, 0), col_type(0, 1, 0, 0), col_type(0, 0, 1, 0)}
#			endif
		{
#			if GLM_CONFIG_CTOR_INIT == GLM_CTOR_INITIALISATION
				this->value[0] = col_type(1, 0, 0, 0);
				this->value[1] = col_type(0, 1, 0, 0);
				this->value[2] = col_type(0, 0, 1, 0);
#			endif
		}
#	endif

	template<typename T, qualifier Q>
	template<qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<3, 4, T, P> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1]), col_type(m[2])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = m[0];
			this->value[1] = m[1];
			this->value[2] = m[2];
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(T s)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(s, 0, 0, 0), col_type(0, s, 0, 0), col_type(0, 0, s, 0)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(s, 0, 0, 0);
			this->value[1] = col_type(0, s, 0, 0);
			this->value[2] = col_type(0, 0, s, 0);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat
	(
		T x0, T y0, T z0, T w0,
		T x1, T y1, T z1, T w1,
		T x2, T y2, T z2, T w2
	)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{
				col_type(x0, y0, z0, w0),
				col_type(x1, y1, z1, w1),
				col_type(x2, y2, z2, w2)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(x0, y0, z0, w0);
			this->value[1] = col_type(x1, y1, z1, w1);
			this->value[2] = col_type(x2, y2, z2, w2);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(col_type const& v0, col_type const& v1, col_type const& v2)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(v0), col_type(v1), col_type(v2)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = v0;
			this->value[1] = v1;
			this->value[2] = v2;
#		endif
	}

	// -- Conversion constructors --

	template<typename T, qualifier Q>
	template<
		typename X0, typename Y0, typename Z0, typename W0,
		typename X1, typename Y1, typename Z1, typename W1,
		typename X2, typename Y2, typename Z2, typename W2>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat
	(
		X0 x0, Y0 y0, Z0 z0, W0 w0,
		X1 x1, Y1 y1, Z1 z1, W1 w1,
		X2 x2, Y2 y2, Z2 z2, W2 w2
	)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{
				col_type(x0, y0, z0, w0),
				col_type(x1, y1, z1, w1),
				col_type(x2, y2, z2, w2)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(x0, y0, z0, w0);
			this->value[1] = col_type(x1, y1, z1, w1);
			this->value[2] = col_type(x2, y2, z2, w2);
#		endif
	}

	template<typename T, qualifier Q>
	template<typename V1, typename V2, typename V3>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(vec<4, V1, Q> const& v0, vec<4, V2, Q> const& v1, vec<4, V3, Q> const& v2)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(v0), col_type(v1), col_type(v2)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(v0);
			this->value[1] = col_type(v1);
			this->value[2] = col_type(v2);
#		endif
	}

	// -- Matrix conversions --

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<3, 4, U, P> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1]), col_type(m[2])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
			this->value[2] = col_type(m[2]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<2, 2, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0], 0, 0), col_type(m[1], 0, 0), col_type(0, 0, 1, 0)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0], 0, 0);
			this->value[1] = col_type(m[1], 0, 0);
			this->value[2] = col_type(0, 0, 1, 0);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<3, 3, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0], 0), col_type(m[1], 0), col_type(m[2], 0)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0], 0);
			this->value[1] = col_type(m[1], 0);
			this->value[2] = col_type(m[2], 0);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<4, 4, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1]), col_type(m[2])}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
			this->value[2] = col_type(m[2]);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<2, 3, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0], 0), col_type(m[1], 0), col_type(0, 0, 1, 0)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0], 0);
			this->value[1] = col_type(m[1], 0);
			this->value[2] = col_type(0, 0, 1, 0);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<3, 2, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0], 0, 0), col_type(m[1], 0, 0), col_type(m[2], 1, 0)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0], 0, 0);
			this->value[1] = col_type(m[1], 0, 0);
			this->value[2] = col_type(m[2], 1, 0);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<2, 4, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0]), col_type(m[1]), col_type(0, 0, 1, 0)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0]);
			this->value[1] = col_type(m[1]);
			this->value[2] = col_type(0, 0, 1, 0);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<4, 2, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0], 0, 0), col_type(m[1], 0, 0), col_type(m[2], 1, 0)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0], 0, 0);
			this->value[1] = col_type(m[1], 0, 0);
			this->value[2] = col_type(m[2], 1, 0);
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR mat<3, 4, T, Q>::mat(mat<4, 3, T, Q> const& m)
#		if GLM_HAS_INITIALIZER_LISTS
			: value{col_type(m[0], 0), col_type(m[1], 0), col_type(m[2], 0)}
#		endif
	{
#		if !GLM_HAS_INITIALIZER_LISTS
			this->value[0] = col_type(m[0], 0);
			this->value[1] = col_type(m[1], 0);
			this->value[2] = col_type(m[2], 0);
#		endif
	}

	// -- Accesses --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename mat<3, 4, T, Q>::col_type & mat<3, 4, T, Q>::operator[](typename mat<3, 4, T, Q>::length_type i)
	{
		assert(i < this->length());
		return this->value[i];
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR typename mat<3, 4, T, Q>::col_type const& mat<3, 4, T, Q>::operator[](typename mat<3, 4, T, Q>::length_type i) const
	{
		assert(i < this->length());
		return this->value[i];
	}

	// -- Unary updatable operators --

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q>& mat<3, 4, T, Q>::operator=(mat<3, 4, U, Q> const& m)
	{
		this->value[0] = m[0];
		this->value[1] = m[1];
		this->value[2] = m[2];
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q>& mat<3, 4, T, Q>::operator+=(U s)
	{
		this->value[0] += s;
		this->value[1] += s;
		this->value[2] += s;
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q>& mat<3, 4, T, Q>::operator+=(mat<3, 4, U, Q> const& m)
	{
		this->value[0] += m[0];
		this->value[1] += m[1];
		this->value[2] += m[2];
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q>& mat<3, 4, T, Q>::operator-=(U s)
	{
		this->value[0] -= s;
		this->value[1] -= s;
		this->value[2] -= s;
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q>& mat<3, 4, T, Q>::operator-=(mat<3, 4, U, Q> const& m)
	{
		this->value[0] -= m[0];
		this->value[1] -= m[1];
		this->value[2] -= m[2];
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q>& mat<3, 4, T, Q>::operator*=(U s)
	{
		this->value[0] *= s;
		this->value[1] *= s;
		this->value[2] *= s;
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> & mat<3, 4, T, Q>::operator/=(U s)
	{
		this->value[0] /= s;
		this->value[1] /= s;
		this->value[2] /= s;
		return *this;
	}

	// -- Increment and decrement operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q>& mat<3, 4, T, Q>::operator++()
	{
		++this->value[0];
		++this->value[1];
		++this->value[2];
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q>& mat<3, 4, T, Q>::operator--()
	{
		--this->value[0];
		--this->value[1];
		--this->value[2];
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> mat<3, 4, T, Q>::operator++(int)
	{
		mat<3, 4, T, Q> Result(*this);
		++*this;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> mat<3, 4, T, Q>::operator--(int)
	{
		mat<3, 4, T, Q> Result(*this);
		--*this;
		return Result;
	}

	// -- Unary arithmetic operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator+(mat<3, 4, T, Q> const& m)
	{
		return m;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator-(mat<3, 4, T, Q> const& m)
	{
		return mat<3, 4, T, Q>(
			-m[0],
			-m[1],
			-m[2]);
	}

	// -- Binary arithmetic operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator+(mat<3, 4, T, Q> const& m, T scalar)
	{
		return mat<3, 4, T, Q>(
			m[0] + scalar,
			m[1] + scalar,
			m[2] + scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator+(mat<3, 4, T, Q> const& m1, mat<3, 4, T, Q> const& m2)
	{
		return mat<3, 4, T, Q>(
			m1[0] + m2[0],
			m1[1] + m2[1],
			m1[2] + m2[2]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator-(mat<3, 4, T, Q> const& m,	T scalar)
	{
		return mat<3, 4, T, Q>(
			m[0] - scalar,
			m[1] - scalar,
			m[2] - scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator-(mat<3, 4, T, Q> const& m1, mat<3, 4, T, Q> const& m2)
	{
		return mat<3, 4, T, Q>(
			m1[0] - m2[0],
			m1[1] - m2[1],
			m1[2] - m2[2]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator*(mat<3, 4, T, Q> const& m, T scalar)
	{
		return mat<3, 4, T, Q>(
			m[0] * scalar,
			m[1] * scalar,
			m[2] * scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator*(T scalar, mat<3, 4, T, Q> const& m)
	{
		return mat<3, 4, T, Q>(
			m[0] * scalar,
			m[1] * scalar,
			m[2] * scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename mat<3, 4, T, Q>::col_type operator*
	(
		mat<3, 4, T, Q> const& m,
		typename mat<3, 4, T, Q>::row_type const& v
	)
	{
		return typename mat<3, 4, T, Q>::col_type(
			m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z,
			m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z,
			m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z,
			m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename mat<3, 4, T, Q>::row_type operator*
	(
		typename mat<3, 4, T, Q>::col_type const& v,
		mat<3, 4, T, Q> const& m
	)
	{
		return typename mat<3, 4, T, Q>::row_type(
			v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2] + v.w * m[0][3],
			v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2] + v.w * m[1][3],
			v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2] + v.w * m[2][3]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> operator*(mat<3, 4, T, Q> const& m1, mat<4, 3, T, Q> const& m2)
	{
		const T SrcA00 = m1[0][0];
		const T SrcA01 = m1[0][1];
		const T SrcA02 = m1[0][2];
		const T SrcA03 = m1[0][3];
		const T SrcA10 = m1[1][0];
		const T SrcA11 = m1[1][1];
		const T SrcA12 = m1[1][2];
		const T SrcA13 = m1[1][3];
		const T SrcA20 = m1[2][0];
		const T SrcA21 = m1[2][1];
		const T SrcA22 = m1[2][2];
		const T SrcA23 = m1[2][3];

		const T SrcB00 = m2[0][0];
		const T SrcB01 = m2[0][1];
		const T SrcB02 = m2[0][2];
		const T SrcB10 = m2[1][0];
		const T SrcB11 = m2[1][1];
		const T SrcB12 = m2[1][2];
		const T SrcB20 = m2[2][0];
		const T SrcB21 = m2[2][1];
		const T SrcB22 = m2[2][2];
		const T SrcB30 = m2[3][0];
		const T SrcB31 = m2[3][1];
		const T SrcB32 = m2[3][2];

		mat<4, 4, T, Q> Result;
		Result[0][0] = SrcA00 * SrcB00 + SrcA10 * SrcB01 + SrcA20 * SrcB02;
		Result[0][1] = SrcA01 * SrcB00 + SrcA11 * SrcB01 + SrcA21 * SrcB02;
		Result[0][2] = SrcA02 * SrcB00 + SrcA12 * SrcB01 + SrcA22 * SrcB02;
		Result[0][3] = SrcA03 * SrcB00 + SrcA13 * SrcB01 + SrcA23 * SrcB02;
		Result[1][0] = SrcA00 * SrcB10 + SrcA10 * SrcB11 + SrcA20 * SrcB12;
		Result[1][1] = SrcA01 * SrcB10 + SrcA11 * SrcB11 + SrcA21 * SrcB12;
		Result[1][2] = SrcA02 * SrcB10 + SrcA12 * SrcB11 + SrcA22 * SrcB12;
		Result[1][3] = SrcA03 * SrcB10 + SrcA13 * SrcB11 + SrcA23 * SrcB12;
		Result[2][0] = SrcA00 * SrcB20 + SrcA10 * SrcB21 + SrcA20 * SrcB22;
		Result[2][1] = SrcA01 * SrcB20 + SrcA11 * SrcB21 + SrcA21 * SrcB22;
		Result[2][2] = SrcA02 * SrcB20 + SrcA12 * SrcB21 + SrcA22 * SrcB22;
		Result[2][3] = SrcA03 * SrcB20 + SrcA13 * SrcB21 + SrcA23 * SrcB22;
		Result[3][0] = SrcA00 * SrcB30 + SrcA10 * SrcB31 + SrcA20 * SrcB32;
		Result[3][1] = SrcA01 * SrcB30 + SrcA11 * SrcB31 + SrcA21 * SrcB32;
		Result[3][2] = SrcA02 * SrcB30 + SrcA12 * SrcB31 + SrcA22 * SrcB32;
		Result[3][3] = SrcA03 * SrcB30 + SrcA13 * SrcB31 + SrcA23 * SrcB32;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 4, T, Q> operator*(mat<3, 4, T, Q> const& m1, mat<2, 3, T, Q> const& m2)
	{
		return mat<2, 4, T, Q>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2],
			m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2],
			m1[0][3] * m2[0][0] + m1[1][3] * m2[0][1] + m1[2][3] * m2[0][2],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2],
			m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2],
			m1[0][3] * m2[1][0] + m1[1][3] * m2[1][1] + m1[2][3] * m2[1][2]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator*(mat<3, 4, T, Q> const& m1, mat<3, 3, T, Q> const& m2)
	{
		return mat<3, 4, T, Q>(
			m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2],
			m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2],
			m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2],
			m1[0][3] * m2[0][0] + m1[1][3] * m2[0][1] + m1[2][3] * m2[0][2],
			m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2],
			m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2],
			m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2],
			m1[0][3] * m2[1][0] + m1[1][3] * m2[1][1] + m1[2][3] * m2[1][2],
			m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1] + m1[2][0] * m2[2][2],
			m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] + m1[2][1] * m2[2][2],
			m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1] + m1[2][2] * m2[2][2],
			m1[0][3] * m2[2][0] + m1[1][3] * m2[2][1] + m1[2][3] * m2[2][2]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator/(mat<3, 4, T, Q> const& m,	T scalar)
	{
		return mat<3, 4, T, Q>(
			m[0] / scalar,
			m[1] / scalar,
			m[2] / scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> operator/(T scalar, mat<3, 4, T, Q> const& m)
	{
		return mat<3, 4, T, Q>(
			scalar / m[0],
			scalar / m[1],
			scalar / m[2]);
	}

	// -- Boolean operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool operator==(mat<3, 4, T, Q> const& m1, mat<3, 4, T, Q> const& m2)
	{
		return (m1[0] == m2[0]) && (m1[1] == m2[1]) && (m1[2] == m2[2]);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool operator!=(mat<3, 4, T, Q> const& m1, mat<3, 4, T, Q> const& m2)
	{
		return (m1[0] != m2[0]) || (m1[1] != m2[1]) || (m1[2] != m2[2]);
	}
} //namespace glm
