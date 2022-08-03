/// @ref core

#include "./compute_vector_relational.hpp"

namespace glm
{
	// -- Implicit basic constructors --

#	if GLM_CONFIG_DEFAULTED_DEFAULT_CTOR == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec()
#			if GLM_CONFIG_CTOR_INIT != GLM_CTOR_INIT_DISABLE
				: x(0), y(0)
#			endif
		{}
#	endif

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(vec<2, T, Q> const& v)
			: x(v.x), y(v.y)
		{}
#	endif

	template<typename T, qualifier Q>
	template<qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(vec<2, T, P> const& v)
		: x(v.x), y(v.y)
	{}

	// -- Explicit basic constructors --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(T scalar)
		: x(scalar), y(scalar)
	{}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(T _x, T _y)
		: x(_x), y(_y)
	{}

	// -- Conversion scalar constructors --

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(vec<1, U, P> const& v)
		: x(static_cast<T>(v.x))
		, y(static_cast<T>(v.x))
	{}

	template<typename T, qualifier Q>
	template<typename A, typename B>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(A _x, B _y)
		: x(static_cast<T>(_x))
		, y(static_cast<T>(_y))
	{}

	template<typename T, qualifier Q>
	template<typename A, typename B>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(vec<1, A, Q> const& _x, B _y)
		: x(static_cast<T>(_x.x))
		, y(static_cast<T>(_y))
	{}

	template<typename T, qualifier Q>
	template<typename A, typename B>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(A _x, vec<1, B, Q> const& _y)
		: x(static_cast<T>(_x))
		, y(static_cast<T>(_y.x))
	{}

	template<typename T, qualifier Q>
	template<typename A, typename B>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(vec<1, A, Q> const& _x, vec<1, B, Q> const& _y)
		: x(static_cast<T>(_x.x))
		, y(static_cast<T>(_y.x))
	{}

	// -- Conversion vector constructors --

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(vec<2, U, P> const& v)
		: x(static_cast<T>(v.x))
		, y(static_cast<T>(v.y))
	{}

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(vec<3, U, P> const& v)
		: x(static_cast<T>(v.x))
		, y(static_cast<T>(v.y))
	{}

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q>::vec(vec<4, U, P> const& v)
		: x(static_cast<T>(v.x))
		, y(static_cast<T>(v.y))
	{}

	// -- Component accesses --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T & vec<2, T, Q>::operator[](typename vec<2, T, Q>::length_type i)
	{
		assert(i >= 0 && i < this->length());
		switch(i)
		{
		default:
		case 0:
			return x;
		case 1:
			return y;
		}
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T const& vec<2, T, Q>::operator[](typename vec<2, T, Q>::length_type i) const
	{
		assert(i >= 0 && i < this->length());
		switch(i)
		{
		default:
		case 0:
			return x;
		case 1:
			return y;
		}
	}

	// -- Unary arithmetic operators --

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator=(vec<2, T, Q> const& v)
		{
			this->x = v.x;
			this->y = v.y;
			return *this;
		}
#	endif

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator=(vec<2, U, Q> const& v)
	{
		this->x = static_cast<T>(v.x);
		this->y = static_cast<T>(v.y);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator+=(U scalar)
	{
		this->x += static_cast<T>(scalar);
		this->y += static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator+=(vec<1, U, Q> const& v)
	{
		this->x += static_cast<T>(v.x);
		this->y += static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator+=(vec<2, U, Q> const& v)
	{
		this->x += static_cast<T>(v.x);
		this->y += static_cast<T>(v.y);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator-=(U scalar)
	{
		this->x -= static_cast<T>(scalar);
		this->y -= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator-=(vec<1, U, Q> const& v)
	{
		this->x -= static_cast<T>(v.x);
		this->y -= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator-=(vec<2, U, Q> const& v)
	{
		this->x -= static_cast<T>(v.x);
		this->y -= static_cast<T>(v.y);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator*=(U scalar)
	{
		this->x *= static_cast<T>(scalar);
		this->y *= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator*=(vec<1, U, Q> const& v)
	{
		this->x *= static_cast<T>(v.x);
		this->y *= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator*=(vec<2, U, Q> const& v)
	{
		this->x *= static_cast<T>(v.x);
		this->y *= static_cast<T>(v.y);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator/=(U scalar)
	{
		this->x /= static_cast<T>(scalar);
		this->y /= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator/=(vec<1, U, Q> const& v)
	{
		this->x /= static_cast<T>(v.x);
		this->y /= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator/=(vec<2, U, Q> const& v)
	{
		this->x /= static_cast<T>(v.x);
		this->y /= static_cast<T>(v.y);
		return *this;
	}

	// -- Increment and decrement operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator++()
	{
		++this->x;
		++this->y;
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator--()
	{
		--this->x;
		--this->y;
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> vec<2, T, Q>::operator++(int)
	{
		vec<2, T, Q> Result(*this);
		++*this;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> vec<2, T, Q>::operator--(int)
	{
		vec<2, T, Q> Result(*this);
		--*this;
		return Result;
	}

	// -- Unary bit operators --

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator%=(U scalar)
	{
		this->x %= static_cast<T>(scalar);
		this->y %= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator%=(vec<1, U, Q> const& v)
	{
		this->x %= static_cast<T>(v.x);
		this->y %= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator%=(vec<2, U, Q> const& v)
	{
		this->x %= static_cast<T>(v.x);
		this->y %= static_cast<T>(v.y);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator&=(U scalar)
	{
		this->x &= static_cast<T>(scalar);
		this->y &= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator&=(vec<1, U, Q> const& v)
	{
		this->x &= static_cast<T>(v.x);
		this->y &= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator&=(vec<2, U, Q> const& v)
	{
		this->x &= static_cast<T>(v.x);
		this->y &= static_cast<T>(v.y);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator|=(U scalar)
	{
		this->x |= static_cast<T>(scalar);
		this->y |= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator|=(vec<1, U, Q> const& v)
	{
		this->x |= static_cast<T>(v.x);
		this->y |= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator|=(vec<2, U, Q> const& v)
	{
		this->x |= static_cast<T>(v.x);
		this->y |= static_cast<T>(v.y);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator^=(U scalar)
	{
		this->x ^= static_cast<T>(scalar);
		this->y ^= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator^=(vec<1, U, Q> const& v)
	{
		this->x ^= static_cast<T>(v.x);
		this->y ^= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator^=(vec<2, U, Q> const& v)
	{
		this->x ^= static_cast<T>(v.x);
		this->y ^= static_cast<T>(v.y);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator<<=(U scalar)
	{
		this->x <<= static_cast<T>(scalar);
		this->y <<= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator<<=(vec<1, U, Q> const& v)
	{
		this->x <<= static_cast<T>(v.x);
		this->y <<= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator<<=(vec<2, U, Q> const& v)
	{
		this->x <<= static_cast<T>(v.x);
		this->y <<= static_cast<T>(v.y);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator>>=(U scalar)
	{
		this->x >>= static_cast<T>(scalar);
		this->y >>= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator>>=(vec<1, U, Q> const& v)
	{
		this->x >>= static_cast<T>(v.x);
		this->y >>= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> & vec<2, T, Q>::operator>>=(vec<2, U, Q> const& v)
	{
		this->x >>= static_cast<T>(v.x);
		this->y >>= static_cast<T>(v.y);
		return *this;
	}

	// -- Unary arithmetic operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator+(vec<2, T, Q> const& v)
	{
		return v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator-(vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			-v.x,
			-v.y);
	}

	// -- Binary arithmetic operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator+(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x + scalar,
			v.y + scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator+(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x + v2.x,
			v1.y + v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator+(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar + v.x,
			scalar + v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator+(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x + v2.x,
			v1.x + v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator+(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x + v2.x,
			v1.y + v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator-(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x - scalar,
			v.y - scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator-(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x - v2.x,
			v1.y - v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator-(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar - v.x,
			scalar - v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator-(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x - v2.x,
			v1.x - v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator-(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x - v2.x,
			v1.y - v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator*(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x * scalar,
			v.y * scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator*(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x * v2.x,
			v1.y * v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator*(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar * v.x,
			scalar * v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator*(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x * v2.x,
			v1.x * v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator*(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x * v2.x,
			v1.y * v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator/(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x / scalar,
			v.y / scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator/(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x / v2.x,
			v1.y / v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator/(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar / v.x,
			scalar / v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator/(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x / v2.x,
			v1.x / v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator/(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x / v2.x,
			v1.y / v2.y);
	}

	// -- Binary bit operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator%(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x % scalar,
			v.y % scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator%(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x % v2.x,
			v1.y % v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator%(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar % v.x,
			scalar % v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator%(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x % v2.x,
			v1.x % v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator%(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x % v2.x,
			v1.y % v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator&(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x & scalar,
			v.y & scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator&(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x & v2.x,
			v1.y & v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator&(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar & v.x,
			scalar & v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator&(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x & v2.x,
			v1.x & v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator&(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x & v2.x,
			v1.y & v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator|(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x | scalar,
			v.y | scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator|(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x | v2.x,
			v1.y | v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator|(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar | v.x,
			scalar | v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator|(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x | v2.x,
			v1.x | v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator|(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x | v2.x,
			v1.y | v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator^(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x ^ scalar,
			v.y ^ scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator^(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x ^ v2.x,
			v1.y ^ v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator^(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar ^ v.x,
			scalar ^ v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator^(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x ^ v2.x,
			v1.x ^ v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator^(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x ^ v2.x,
			v1.y ^ v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator<<(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x << scalar,
			v.y << scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator<<(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x << v2.x,
			v1.y << v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator<<(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar << v.x,
			scalar << v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator<<(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x << v2.x,
			v1.x << v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator<<(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x << v2.x,
			v1.y << v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator>>(vec<2, T, Q> const& v, T scalar)
	{
		return vec<2, T, Q>(
			v.x >> scalar,
			v.y >> scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator>>(vec<2, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x >> v2.x,
			v1.y >> v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator>>(T scalar, vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			scalar >> v.x,
			scalar >> v.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator>>(vec<1, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x >> v2.x,
			v1.x >> v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator>>(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return vec<2, T, Q>(
			v1.x >> v2.x,
			v1.y >> v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, T, Q> operator~(vec<2, T, Q> const& v)
	{
		return vec<2, T, Q>(
			~v.x,
			~v.y);
	}

	// -- Boolean operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR bool operator==(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return
			detail::compute_equal<T, std::numeric_limits<T>::is_iec559>::call(v1.x, v2.x) &&
			detail::compute_equal<T, std::numeric_limits<T>::is_iec559>::call(v1.y, v2.y);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR bool operator!=(vec<2, T, Q> const& v1, vec<2, T, Q> const& v2)
	{
		return !(v1 == v2);
	}

	template<qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, bool, Q> operator&&(vec<2, bool, Q> const& v1, vec<2, bool, Q> const& v2)
	{
		return vec<2, bool, Q>(v1.x && v2.x, v1.y && v2.y);
	}

	template<qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<2, bool, Q> operator||(vec<2, bool, Q> const& v1, vec<2, bool, Q> const& v2)
	{
		return vec<2, bool, Q>(v1.x || v2.x, v1.y || v2.y);
	}
}//namespace glm
