/// @ref core

#include "./compute_vector_relational.hpp"

namespace glm
{
	// -- Implicit basic constructors --

#	if GLM_CONFIG_DEFAULTED_DEFAULT_CTOR == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q>::vec()
#			if GLM_CONFIG_CTOR_INIT != GLM_CTOR_INIT_DISABLE
				: x(0)
#			endif
		{}
#	endif

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q>::vec(vec<1, T, Q> const& v)
			: x(v.x)
		{}
#	endif

	template<typename T, qualifier Q>
	template<qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q>::vec(vec<1, T, P> const& v)
		: x(v.x)
	{}

	// -- Explicit basic constructors --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q>::vec(T scalar)
		: x(scalar)
	{}

	// -- Conversion vector constructors --

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q>::vec(vec<1, U, P> const& v)
		: x(static_cast<T>(v.x))
	{}

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q>::vec(vec<2, U, P> const& v)
		: x(static_cast<T>(v.x))
	{}

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q>::vec(vec<3, U, P> const& v)
		: x(static_cast<T>(v.x))
	{}

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q>::vec(vec<4, U, P> const& v)
		: x(static_cast<T>(v.x))
	{}

	// -- Component accesses --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T & vec<1, T, Q>::operator[](typename vec<1, T, Q>::length_type)
	{
		return x;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T const& vec<1, T, Q>::operator[](typename vec<1, T, Q>::length_type) const
	{
		return x;
	}

	// -- Unary arithmetic operators --

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator=(vec<1, T, Q> const& v)
		{
			this->x = v.x;
			return *this;
		}
#	endif

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator=(vec<1, U, Q> const& v)
	{
		this->x = static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator+=(U scalar)
	{
		this->x += static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator+=(vec<1, U, Q> const& v)
	{
		this->x += static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator-=(U scalar)
	{
		this->x -= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator-=(vec<1, U, Q> const& v)
	{
		this->x -= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator*=(U scalar)
	{
		this->x *= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator*=(vec<1, U, Q> const& v)
	{
		this->x *= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator/=(U scalar)
	{
		this->x /= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator/=(vec<1, U, Q> const& v)
	{
		this->x /= static_cast<T>(v.x);
		return *this;
	}

	// -- Increment and decrement operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator++()
	{
		++this->x;
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator--()
	{
		--this->x;
		return *this;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> vec<1, T, Q>::operator++(int)
	{
		vec<1, T, Q> Result(*this);
		++*this;
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> vec<1, T, Q>::operator--(int)
	{
		vec<1, T, Q> Result(*this);
		--*this;
		return Result;
	}

	// -- Unary bit operators --

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator%=(U scalar)
	{
		this->x %= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator%=(vec<1, U, Q> const& v)
	{
		this->x %= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator&=(U scalar)
	{
		this->x &= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator&=(vec<1, U, Q> const& v)
	{
		this->x &= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator|=(U scalar)
	{
		this->x |= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator|=(vec<1, U, Q> const& v)
	{
		this->x |= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator^=(U scalar)
	{
		this->x ^= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator^=(vec<1, U, Q> const& v)
	{
		this->x ^= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator<<=(U scalar)
	{
		this->x <<= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator<<=(vec<1, U, Q> const& v)
	{
		this->x <<= static_cast<T>(v.x);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator>>=(U scalar)
	{
		this->x >>= static_cast<T>(scalar);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> & vec<1, T, Q>::operator>>=(vec<1, U, Q> const& v)
	{
		this->x >>= static_cast<T>(v.x);
		return *this;
	}

	// -- Unary constant operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator+(vec<1, T, Q> const& v)
	{
		return v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator-(vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			-v.x);
	}

	// -- Binary arithmetic operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator+(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			v.x + scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator+(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			scalar + v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator+(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			v1.x + v2.x);
	}

	//operator-
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator-(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			v.x - scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator-(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			scalar - v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator-(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			v1.x - v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator*(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			v.x * scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator*(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			scalar * v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator*(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			v1.x * v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator/(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			v.x / scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator/(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			scalar / v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator/(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			v1.x / v2.x);
	}

	// -- Binary bit operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator%(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			v.x % scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator%(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			scalar % v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator%(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			v1.x % v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator&(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			v.x & scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator&(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			scalar & v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator&(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			v1.x & v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator|(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			v.x | scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator|(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			scalar | v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator|(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			v1.x | v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator^(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			v.x ^ scalar);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator^(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			scalar ^ v.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator^(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			v1.x ^ v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator<<(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			static_cast<T>(v.x << scalar));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator<<(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			static_cast<T>(scalar << v.x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator<<(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			static_cast<T>(v1.x << v2.x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator>>(vec<1, T, Q> const& v, T scalar)
	{
		return vec<1, T, Q>(
			static_cast<T>(v.x >> scalar));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator>>(T scalar, vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			static_cast<T>(scalar >> v.x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator>>(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return vec<1, T, Q>(
			static_cast<T>(v1.x >> v2.x));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, T, Q> operator~(vec<1, T, Q> const& v)
	{
		return vec<1, T, Q>(
			~v.x);
	}

	// -- Boolean operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR bool operator==(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return detail::compute_equal<T, std::numeric_limits<T>::is_iec559>::call(v1.x, v2.x);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR bool operator!=(vec<1, T, Q> const& v1, vec<1, T, Q> const& v2)
	{
		return !(v1 == v2);
	}

	template<qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, bool, Q> operator&&(vec<1, bool, Q> const& v1, vec<1, bool, Q> const& v2)
	{
		return vec<1, bool, Q>(v1.x && v2.x);
	}

	template<qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<1, bool, Q> operator||(vec<1, bool, Q> const& v1, vec<1, bool, Q> const& v2)
	{
		return vec<1, bool, Q>(v1.x || v2.x);
	}
}//namespace glm
