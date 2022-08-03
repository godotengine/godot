/// @ref gtx_dual_quaternion

#include "../geometric.hpp"
#include <limits>

namespace glm
{
	// -- Component accesses --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename tdualquat<T, Q>::part_type & tdualquat<T, Q>::operator[](typename tdualquat<T, Q>::length_type i)
	{
		assert(i >= 0 && i < this->length());
		return (&real)[i];
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER typename tdualquat<T, Q>::part_type const& tdualquat<T, Q>::operator[](typename tdualquat<T, Q>::length_type i) const
	{
		assert(i >= 0 && i < this->length());
		return (&real)[i];
	}

	// -- Implicit basic constructors --

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR tdualquat<T, Q>::tdualquat()
#			if GLM_CONFIG_DEFAULTED_FUNCTIONS != GLM_DISABLE
			: real(qua<T, Q>())
			, dual(qua<T, Q>(0, 0, 0, 0))
#			endif
		{}

		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR tdualquat<T, Q>::tdualquat(tdualquat<T, Q> const& d)
			: real(d.real)
			, dual(d.dual)
		{}
#	endif

	template<typename T, qualifier Q>
	template<qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR tdualquat<T, Q>::tdualquat(tdualquat<T, P> const& d)
		: real(d.real)
		, dual(d.dual)
	{}

	// -- Explicit basic constructors --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR tdualquat<T, Q>::tdualquat(qua<T, Q> const& r)
		: real(r), dual(qua<T, Q>(0, 0, 0, 0))
	{}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR tdualquat<T, Q>::tdualquat(qua<T, Q> const& q, vec<3, T, Q> const& p)
		: real(q), dual(
			T(-0.5) * ( p.x*q.x + p.y*q.y + p.z*q.z),
			T(+0.5) * ( p.x*q.w + p.y*q.z - p.z*q.y),
			T(+0.5) * (-p.x*q.z + p.y*q.w + p.z*q.x),
			T(+0.5) * ( p.x*q.y - p.y*q.x + p.z*q.w))
	{}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR tdualquat<T, Q>::tdualquat(qua<T, Q> const& r, qua<T, Q> const& d)
		: real(r), dual(d)
	{}

	// -- Conversion constructors --

	template<typename T, qualifier Q>
	template<typename U, qualifier P>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR tdualquat<T, Q>::tdualquat(tdualquat<U, P> const& q)
		: real(q.real)
		, dual(q.dual)
	{}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR tdualquat<T, Q>::tdualquat(mat<2, 4, T, Q> const& m)
	{
		*this = dualquat_cast(m);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR tdualquat<T, Q>::tdualquat(mat<3, 4, T, Q> const& m)
	{
		*this = dualquat_cast(m);
	}

	// -- Unary arithmetic operators --

#	if GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_DISABLE
		template<typename T, qualifier Q>
		GLM_FUNC_QUALIFIER tdualquat<T, Q> & tdualquat<T, Q>::operator=(tdualquat<T, Q> const& q)
		{
			this->real = q.real;
			this->dual = q.dual;
			return *this;
		}
#	endif

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> & tdualquat<T, Q>::operator=(tdualquat<U, Q> const& q)
	{
		this->real = q.real;
		this->dual = q.dual;
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> & tdualquat<T, Q>::operator*=(U s)
	{
		this->real *= static_cast<T>(s);
		this->dual *= static_cast<T>(s);
		return *this;
	}

	template<typename T, qualifier Q>
	template<typename U>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> & tdualquat<T, Q>::operator/=(U s)
	{
		this->real /= static_cast<T>(s);
		this->dual /= static_cast<T>(s);
		return *this;
	}

	// -- Unary bit operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> operator+(tdualquat<T, Q> const& q)
	{
		return q;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> operator-(tdualquat<T, Q> const& q)
	{
		return tdualquat<T, Q>(-q.real, -q.dual);
	}

	// -- Binary operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> operator+(tdualquat<T, Q> const& q, tdualquat<T, Q> const& p)
	{
		return tdualquat<T, Q>(q.real + p.real,q.dual + p.dual);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> operator*(tdualquat<T, Q> const& p, tdualquat<T, Q> const& o)
	{
		return tdualquat<T, Q>(p.real * o.real,p.real * o.dual + p.dual * o.real);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> operator*(tdualquat<T, Q> const& q, vec<3, T, Q> const& v)
	{
		vec<3, T, Q> const real_v3(q.real.x,q.real.y,q.real.z);
		vec<3, T, Q> const dual_v3(q.dual.x,q.dual.y,q.dual.z);
		return (cross(real_v3, cross(real_v3,v) + v * q.real.w + dual_v3) + dual_v3 * q.real.w - real_v3 * q.dual.w) * T(2) + v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> operator*(vec<3, T, Q> const& v,	tdualquat<T, Q> const& q)
	{
		return glm::inverse(q) * v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, T, Q> operator*(tdualquat<T, Q> const& q, vec<4, T, Q> const& v)
	{
		return vec<4, T, Q>(q * vec<3, T, Q>(v), v.w);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, T, Q> operator*(vec<4, T, Q> const& v,	tdualquat<T, Q> const& q)
	{
		return glm::inverse(q) * v;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> operator*(tdualquat<T, Q> const& q, T const& s)
	{
		return tdualquat<T, Q>(q.real * s, q.dual * s);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> operator*(T const& s, tdualquat<T, Q> const& q)
	{
		return q * s;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> operator/(tdualquat<T, Q> const& q,	T const& s)
	{
		return tdualquat<T, Q>(q.real / s, q.dual / s);
	}

	// -- Boolean operators --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool operator==(tdualquat<T, Q> const& q1, tdualquat<T, Q> const& q2)
	{
		return (q1.real == q2.real) && (q1.dual == q2.dual);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER bool operator!=(tdualquat<T, Q> const& q1, tdualquat<T, Q> const& q2)
	{
		return (q1.real != q2.real) || (q1.dual != q2.dual);
	}

	// -- Operations --

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> dual_quat_identity()
	{
		return tdualquat<T, Q>(
			qua<T, Q>(static_cast<T>(1), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)),
			qua<T, Q>(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> normalize(tdualquat<T, Q> const& q)
	{
		return q / length(q.real);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> lerp(tdualquat<T, Q> const& x, tdualquat<T, Q> const& y, T const& a)
	{
		// Dual Quaternion Linear blend aka DLB:
		// Lerp is only defined in [0, 1]
		assert(a >= static_cast<T>(0));
		assert(a <= static_cast<T>(1));
		T const k = dot(x.real,y.real) < static_cast<T>(0) ? -a : a;
		T const one(1);
		return tdualquat<T, Q>(x * (one - a) + y * k);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> inverse(tdualquat<T, Q> const& q)
	{
		const glm::qua<T, Q> real = conjugate(q.real);
		const glm::qua<T, Q> dual = conjugate(q.dual);
		return tdualquat<T, Q>(real, dual + (real * (-2.0f * dot(real,dual))));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<2, 4, T, Q> mat2x4_cast(tdualquat<T, Q> const& x)
	{
		return mat<2, 4, T, Q>( x[0].x, x[0].y, x[0].z, x[0].w, x[1].x, x[1].y, x[1].z, x[1].w );
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<3, 4, T, Q> mat3x4_cast(tdualquat<T, Q> const& x)
	{
		qua<T, Q> r = x.real / length2(x.real);

		qua<T, Q> const rr(r.w * x.real.w, r.x * x.real.x, r.y * x.real.y, r.z * x.real.z);
		r *= static_cast<T>(2);

		T const xy = r.x * x.real.y;
		T const xz = r.x * x.real.z;
		T const yz = r.y * x.real.z;
		T const wx = r.w * x.real.x;
		T const wy = r.w * x.real.y;
		T const wz = r.w * x.real.z;

		vec<4, T, Q> const a(
			rr.w + rr.x - rr.y - rr.z,
			xy - wz,
			xz + wy,
			-(x.dual.w * r.x - x.dual.x * r.w + x.dual.y * r.z - x.dual.z * r.y));

		vec<4, T, Q> const b(
			xy + wz,
			rr.w + rr.y - rr.x - rr.z,
			yz - wx,
			-(x.dual.w * r.y - x.dual.x * r.z - x.dual.y * r.w + x.dual.z * r.x));

		vec<4, T, Q> const c(
			xz - wy,
			yz + wx,
			rr.w + rr.z - rr.x - rr.y,
			-(x.dual.w * r.z + x.dual.x * r.y - x.dual.y * r.x - x.dual.z * r.w));

		return mat<3, 4, T, Q>(a, b, c);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> dualquat_cast(mat<2, 4, T, Q> const& x)
	{
		return tdualquat<T, Q>(
			qua<T, Q>( x[0].w, x[0].x, x[0].y, x[0].z ),
			qua<T, Q>( x[1].w, x[1].x, x[1].y, x[1].z ));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER tdualquat<T, Q> dualquat_cast(mat<3, 4, T, Q> const& x)
	{
		qua<T, Q> real;

		T const trace = x[0].x + x[1].y + x[2].z;
		if(trace > static_cast<T>(0))
		{
			T const r = sqrt(T(1) + trace);
			T const invr = static_cast<T>(0.5) / r;
			real.w = static_cast<T>(0.5) * r;
			real.x = (x[2].y - x[1].z) * invr;
			real.y = (x[0].z - x[2].x) * invr;
			real.z = (x[1].x - x[0].y) * invr;
		}
		else if(x[0].x > x[1].y && x[0].x > x[2].z)
		{
			T const r = sqrt(T(1) + x[0].x - x[1].y - x[2].z);
			T const invr = static_cast<T>(0.5) / r;
			real.x = static_cast<T>(0.5)*r;
			real.y = (x[1].x + x[0].y) * invr;
			real.z = (x[0].z + x[2].x) * invr;
			real.w = (x[2].y - x[1].z) * invr;
		}
		else if(x[1].y > x[2].z)
		{
			T const r = sqrt(T(1) + x[1].y - x[0].x - x[2].z);
			T const invr = static_cast<T>(0.5) / r;
			real.x = (x[1].x + x[0].y) * invr;
			real.y = static_cast<T>(0.5) * r;
			real.z = (x[2].y + x[1].z) * invr;
			real.w = (x[0].z - x[2].x) * invr;
		}
		else
		{
			T const r = sqrt(T(1) + x[2].z - x[0].x - x[1].y);
			T const invr = static_cast<T>(0.5) / r;
			real.x = (x[0].z + x[2].x) * invr;
			real.y = (x[2].y + x[1].z) * invr;
			real.z = static_cast<T>(0.5) * r;
			real.w = (x[1].x - x[0].y) * invr;
		}

		qua<T, Q> dual;
		dual.x =  static_cast<T>(0.5) * ( x[0].w * real.w + x[1].w * real.z - x[2].w * real.y);
		dual.y =  static_cast<T>(0.5) * (-x[0].w * real.z + x[1].w * real.w + x[2].w * real.x);
		dual.z =  static_cast<T>(0.5) * ( x[0].w * real.y - x[1].w * real.x + x[2].w * real.w);
		dual.w = -static_cast<T>(0.5) * ( x[0].w * real.x + x[1].w * real.y + x[2].w * real.z);
		return tdualquat<T, Q>(real, dual);
	}
}//namespace glm
