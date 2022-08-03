#include "../trigonometric.hpp"
#include "../exponential.hpp"
#include "../ext/quaternion_common.hpp"
#include "../ext/quaternion_geometric.hpp"
#include <limits>

namespace glm{
namespace detail
{
	template <typename T>
	struct genTypeTrait<qua<T> >
	{
		static const genTypeEnum GENTYPE = GENTYPE_QUAT;
	};

	template<typename T, qualifier Q, bool Aligned>
	struct compute_dot<qua<T, Q>, T, Aligned>
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static T call(qua<T, Q> const& a, qua<T, Q> const& b)
		{
			vec<4, T, Q> tmp(a.w * b.w, a.x * b.x, a.y * b.y, a.z * b.z);
			return (tmp.x + tmp.y) + (tmp.z + tmp.w);
		}
	};

	template<typename T, qualifier Q, bool Aligned>
	struct compute_quat_add
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static qua<T, Q> call(qua<T, Q> const& q, qua<T, Q> const& p)
		{
			return qua<T, Q>(q.w + p.w, q.x + p.x, q.y + p.y, q.z + p.z);
		}
	};

	template<typename T, qualifier Q, bool Aligned>
	struct compute_quat_sub
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static qua<T, Q> call(qua<T, Q> const& q, qua<T, Q> const& p)
		{
			return qua<T, Q>(q.w - p.w, q.x - p.x, q.y - p.y, q.z - p.z);
		}
	};

	template<typename T, qualifier Q, bool Aligned>
	struct compute_quat_mul_scalar
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static qua<T, Q> call(qua<T, Q> const& q, T s)
		{
			return qua<T, Q>(q.w * s, q.x * s, q.y * s, q.z * s);
		}
	};

	template<typename T, qualifier Q, bool Aligned>
	struct compute_quat_div_scalar
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static qua<T, Q> call(qua<T, Q> const& q, T s)
		{
			return qua<T, Q>(q.w / s, q.x / s, q.y / s, q.z / s);
		}
	};

	template<typename T, qualifier Q, bool Aligned>
	struct compute_quat_mul_vec4
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static vec<4, T, Q> call(qua<T, Q> const& q, vec<4, T, Q> const& v)
		{
			return vec<4, T, Q>(q * vec<3, T, Q>(v), v.w);
		}
	};
}//namespace detail

	// -- Component accesses --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T & qua<T, Q>::operator[](typename qua<T, Q>::length_type i)
	{
		assert(i >= 0 && i < this->length());
#		ifdef GLM_FORCE_QUAT_DATA_XYZW
			return (&x)[i];
#		else
			return (&w)[i];
#		endif
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR T const& qua<T, Q>::operator[](typename qua<T, Q>::length_type i) const
	{
		assert(i >= 0 && i < this->length());
#		ifdef GLM_FORCE_QUAT_DATA_XYZW
			return (&x)[i];
#		else
			return (&w)[i];
#		endif
	}

	// -- Implicit basic constructors --

#	if GLM_CONFIG_DEFAULTED_DEFAULT_CTOR == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua()
#			if GLM_CONFIG_CTOR_INIT != GLM_CTOR_INIT_DISABLE
#				ifdef GLM_FORCE_QUAT_DATA_XYZW
					: x(0), y(0), z(0), w(1)
#				else
					: w(1), x(0), y(0), z(0)
#				endif
#			endif
		{}
#	endif

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua(qua<T, Q> const& q)
#			ifdef GLM_FORCE_QUAT_DATA_XYZW
				: x(q.x), y(q.y), z(q.z), w(q.w)
#			else
				: w(q.w), x(q.x), y(q.y), z(q.z)
#			endif
		{}
#	endif

	template<typename T, qualifier Q>
	template<qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua(qua<T, P> const& q)
#		ifdef GLM_FORCE_QUAT_DATA_XYZW
			: x(q.x), y(q.y), z(q.z), w(q.w)
#		else
			: w(q.w), x(q.x), y(q.y), z(q.z)
#		endif
	{}

	// -- Explicit basic constructors --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua(T s, vec<3, T, Q> const& v)
#		ifdef GLM_FORCE_QUAT_DATA_XYZW
			: x(v.x), y(v.y), z(v.z), w(s)
#		else
			: w(s), x(v.x), y(v.y), z(v.z)
#		endif
	{}

	template <typename T, qualifier Q>
#	ifdef GLM_FORCE_QUAT_DATA_XYZW
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua(T _x, T _y, T _z, T _w)
			: x(_x), y(_y), z(_z), w(_w)
#	else
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua(T _w, T _x, T _y, T _z)
			: w(_w), x(_x), y(_y), z(_z)
#	endif
	{}

	// -- Conversion constructors --

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua(qua<U, P> const& q)
#		ifdef GLM_FORCE_QUAT_DATA_XYZW
			: x(static_cast<T>(q.x)), y(static_cast<T>(q.y)), z(static_cast<T>(q.z)), w(static_cast<T>(q.w))
#		else
			: w(static_cast<T>(q.w)), x(static_cast<T>(q.x)), y(static_cast<T>(q.y)), z(static_cast<T>(q.z))
#		endif
	{}

	//template<typename valType>
	//GLM_FUNC_QUALIFIER qua<valType>::qua
	//(
	//	valType const& pitch,
	//	valType const& yaw,
	//	valType const& roll
	//)
	//{
	//	vec<3, valType> eulerAngle(pitch * valType(0.5), yaw * valType(0.5), roll * valType(0.5));
	//	vec<3, valType> c = glm::cos(eulerAngle * valType(0.5));
	//	vec<3, valType> s = glm::sin(eulerAngle * valType(0.5));
	//
	//	this->w = c.x * c.y * c.z + s.x * s.y * s.z;
	//	this->x = s.x * c.y * c.z - c.x * s.y * s.z;
	//	this->y = c.x * s.y * c.z + s.x * c.y * s.z;
	//	this->z = c.x * c.y * s.z - s.x * s.y * c.z;
	//}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q>::qua(vec<3, T, Q> const& u, vec<3, T, Q> const& v)
	{
		T norm_u_norm_v = sqrt(dot(u, u) * dot(v, v));
		T real_part = norm_u_norm_v + dot(u, v);
		vec<3, T, Q> t;

		if(real_part < static_cast<T>(1.e-6f) * norm_u_norm_v)
		{
			// If u and v are exactly opposite, rotate 180 degrees
			// around an arbitrary orthogonal axis. Axis normalisation
			// can happen later, when we normalise the quaternion.
			real_part = static_cast<T>(0);
			t = abs(u.x) > abs(u.z) ? vec<3, T, Q>(-u.y, u.x, static_cast<T>(0)) : vec<3, T, Q>(static_cast<T>(0), -u.z, u.y);
		}
		else
		{
			// Otherwise, build quaternion the standard way.
			t = cross(u, v);
		}

		*this = normalize(qua<T, Q>(real_part, t.x, t.y, t.z));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q>::qua(vec<3, T, Q> const& eulerAngle)
	{
		vec<3, T, Q> c = glm::cos(eulerAngle * T(0.5));
		vec<3, T, Q> s = glm::sin(eulerAngle * T(0.5));

		this->w = c.x * c.y * c.z + s.x * s.y * s.z;
		this->x = s.x * c.y * c.z - c.x * s.y * s.z;
		this->y = c.x * s.y * c.z + s.x * c.y * s.z;
		this->z = c.x * c.y * s.z - s.x * s.y * c.z;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q>::qua(mat<3, 3, T, Q> const& m)
	{
		*this = quat_cast(m);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q>::qua(mat<4, 4, T, Q> const& m)
	{
		*this = quat_cast(m);
	}

#	if GLM_HAS_EXPLICIT_CONVERSION_OPERATORS
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q>::operator mat<3, 3, T, Q>() const
	{
		return mat3_cast(*this);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER qua<T, Q>::operator mat<4, 4, T, Q>() const
	{
		return mat4_cast(*this);
	}
#	endif//GLM_HAS_EXPLICIT_CONVERSION_OPERATORS

	// -- Unary arithmetic operators --

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> & qua<T, Q>::operator=(qua<T, Q> const& q)
		{
			this->w = q.w;
			this->x = q.x;
			this->y = q.y;
			this->z = q.z;
			return *this;
		}
#	endif

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> & qua<T, Q>::operator=(qua<U, Q> const& q)
	{
		this->w = static_cast<T>(q.w);
		this->x = static_cast<T>(q.x);
		this->y = static_cast<T>(q.y);
		this->z = static_cast<T>(q.z);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> & qua<T, Q>::operator+=(qua<U, Q> const& q)
	{
		return (*this = detail::compute_quat_add<T, Q, detail::is_aligned<Q>::value>::call(*this, qua<T, Q>(q)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> & qua<T, Q>::operator-=(qua<U, Q> const& q)
	{
		return (*this = detail::compute_quat_sub<T, Q, detail::is_aligned<Q>::value>::call(*this, qua<T, Q>(q)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> & qua<T, Q>::operator*=(qua<U, Q> const& r)
	{
		qua<T, Q> const p(*this);
		qua<T, Q> const q(r);

		this->w = p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z;
		this->x = p.w * q.x + p.x * q.w + p.y * q.z - p.z * q.y;
		this->y = p.w * q.y + p.y * q.w + p.z * q.x - p.x * q.z;
		this->z = p.w * q.z + p.z * q.w + p.x * q.y - p.y * q.x;
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> & qua<T, Q>::operator*=(U s)
	{
		return (*this = detail::compute_quat_mul_scalar<T, Q, detail::is_aligned<Q>::value>::call(*this, static_cast<U>(s)));
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> & qua<T, Q>::operator/=(U s)
	{
		return (*this = detail::compute_quat_div_scalar<T, Q, detail::is_aligned<Q>::value>::call(*this, static_cast<U>(s)));
	}

	// -- Unary bit operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> operator+(qua<T, Q> const& q)
	{
		return q;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> operator-(qua<T, Q> const& q)
	{
		return qua<T, Q>(-q.w, -q.x, -q.y, -q.z);
	}

	// -- Binary operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> operator+(qua<T, Q> const& q, qua<T, Q> const& p)
	{
		return qua<T, Q>(q) += p;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> operator-(qua<T, Q> const& q, qua<T, Q> const& p)
	{
		return qua<T, Q>(q) -= p;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> operator*(qua<T, Q> const& q, qua<T, Q> const& p)
	{
		return qua<T, Q>(q) *= p;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator*(qua<T, Q> const& q, vec<3, T, Q> const& v)
	{
		vec<3, T, Q> const QuatVector(q.x, q.y, q.z);
		vec<3, T, Q> const uv(glm::cross(QuatVector, v));
		vec<3, T, Q> const uuv(glm::cross(QuatVector, uv));

		return v + ((uv * q.w) + uuv) * static_cast<T>(2);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<3, T, Q> operator*(vec<3, T, Q> const& v, qua<T, Q> const& q)
	{
		return glm::inverse(q) * v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, T, Q> operator*(qua<T, Q> const& q, vec<4, T, Q> const& v)
	{
		return detail::compute_quat_mul_vec4<T, Q, detail::is_aligned<Q>::value>::call(q, v);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, T, Q> operator*(vec<4, T, Q> const& v, qua<T, Q> const& q)
	{
		return glm::inverse(q) * v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> operator*(qua<T, Q> const& q, T const& s)
	{
		return qua<T, Q>(
			q.w * s, q.x * s, q.y * s, q.z * s);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> operator*(T const& s, qua<T, Q> const& q)
	{
		return q * s;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR qua<T, Q> operator/(qua<T, Q> const& q, T const& s)
	{
		return qua<T, Q>(
			q.w / s, q.x / s, q.y / s, q.z / s);
	}

	// -- Boolean operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR bool operator==(qua<T, Q> const& q1, qua<T, Q> const& q2)
	{
		return q1.x == q2.x && q1.y == q2.y && q1.z == q2.z && q1.w == q2.w;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR bool operator!=(qua<T, Q> const& q1, qua<T, Q> const& q2)
	{
		return q1.x != q2.x || q1.y != q2.y || q1.z != q2.z || q1.w != q2.w;
	}
}//namespace glm

#if GLM_CONFIG_SIMD == GLM_ENABLE
#	include "type_quat_simd.inl"
#endif

