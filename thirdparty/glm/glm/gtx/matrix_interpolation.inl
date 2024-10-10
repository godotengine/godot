/// @ref gtx_matrix_interpolation

#include "../ext/scalar_constants.hpp"

#include <limits>

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER void axisAngle(mat<4, 4, T, Q> const& m, vec<3, T, Q>& axis, T& angle)
	{
		T const epsilon =
		    std::numeric_limits<T>::epsilon() * static_cast<T>(1e2);

        bool const nearSymmetrical =
            abs(m[1][0] - m[0][1]) < epsilon &&
            abs(m[2][0] - m[0][2]) < epsilon &&
            abs(m[2][1] - m[1][2]) < epsilon;

		if(nearSymmetrical)
		{
            bool const nearIdentity =
                abs(m[1][0] + m[0][1]) < epsilon &&
                abs(m[2][0] + m[0][2]) < epsilon &&
                abs(m[2][1] + m[1][2]) < epsilon &&
                abs(m[0][0] + m[1][1] + m[2][2] - T(3.0)) < epsilon;
			if (nearIdentity)
			{
				angle = static_cast<T>(0.0);
				axis = vec<3, T, Q>(
				    static_cast<T>(1.0), static_cast<T>(0.0), static_cast<T>(0.0));
				return;
			}
			angle = pi<T>();
			T xx = (m[0][0] + static_cast<T>(1.0)) * static_cast<T>(0.5);
			T yy = (m[1][1] + static_cast<T>(1.0)) * static_cast<T>(0.5);
			T zz = (m[2][2] + static_cast<T>(1.0)) * static_cast<T>(0.5);
			T xy = (m[1][0] + m[0][1]) * static_cast<T>(0.25);
			T xz = (m[2][0] + m[0][2]) * static_cast<T>(0.25);
			T yz = (m[2][1] + m[1][2]) * static_cast<T>(0.25);
			if((xx > yy) && (xx > zz))
			{
				if(xx < epsilon)
				{
					axis.x = static_cast<T>(0.0);
					axis.y = static_cast<T>(0.7071);
					axis.z = static_cast<T>(0.7071);
				}
				else
				{
					axis.x = sqrt(xx);
					axis.y = xy / axis.x;
					axis.z = xz / axis.x;
				}
			}
			else if (yy > zz)
			{
				if(yy < epsilon)
				{
					axis.x = static_cast<T>(0.7071);
					axis.y = static_cast<T>(0.0);
					axis.z = static_cast<T>(0.7071);
				}
				else
				{
					axis.y = sqrt(yy);
					axis.x = xy / axis.y;
					axis.z = yz / axis.y;
				}
			}
			else
			{
				if (zz < epsilon)
				{
					axis.x = static_cast<T>(0.7071);
					axis.y = static_cast<T>(0.7071);
					axis.z = static_cast<T>(0.0);
				}
				else
				{
					axis.z = sqrt(zz);
					axis.x = xz / axis.z;
					axis.y = yz / axis.z;
				}
			}
			return;
		}

		T const angleCos = (m[0][0] + m[1][1] + m[2][2] - static_cast<T>(1)) * static_cast<T>(0.5);
		if(angleCos >= static_cast<T>(1.0))
		{
			angle = static_cast<T>(0.0);
		}
		else if (angleCos <= static_cast<T>(-1.0))
		{
			angle = pi<T>();
		}
		else
		{
			angle = acos(angleCos);
		}

        axis = glm::normalize(glm::vec<3, T, Q>(
            m[1][2] - m[2][1], m[2][0] - m[0][2], m[0][1] - m[1][0]));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> axisAngleMatrix(vec<3, T, Q> const& axis, T const angle)
	{
		T c = cos(angle);
		T s = sin(angle);
		T t = static_cast<T>(1) - c;
		vec<3, T, Q> n = normalize(axis);

		return mat<4, 4, T, Q>(
			t * n.x * n.x + c,          t * n.x * n.y + n.z * s,    t * n.x * n.z - n.y * s,    static_cast<T>(0.0),
			t * n.x * n.y - n.z * s,    t * n.y * n.y + c,          t * n.y * n.z + n.x * s,    static_cast<T>(0.0),
			t * n.x * n.z + n.y * s,    t * n.y * n.z - n.x * s,    t * n.z * n.z + c,          static_cast<T>(0.0),
			static_cast<T>(0.0),        static_cast<T>(0.0),        static_cast<T>(0.0),        static_cast<T>(1.0));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> extractMatrixRotation(mat<4, 4, T, Q> const& m)
	{
		return mat<4, 4, T, Q>(
			m[0][0], m[0][1], m[0][2], static_cast<T>(0.0),
			m[1][0], m[1][1], m[1][2], static_cast<T>(0.0),
			m[2][0], m[2][1], m[2][2], static_cast<T>(0.0),
			static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(0.0), static_cast<T>(1.0));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> interpolate(mat<4, 4, T, Q> const& m1, mat<4, 4, T, Q> const& m2, T const delta)
	{
		mat<4, 4, T, Q> m1rot = extractMatrixRotation(m1);
		mat<4, 4, T, Q> dltRotation = m2 * transpose(m1rot);
		vec<3, T, Q> dltAxis;
		T dltAngle;
		axisAngle(dltRotation, dltAxis, dltAngle);
		mat<4, 4, T, Q> out = axisAngleMatrix(dltAxis, dltAngle * delta) * m1rot;
		out[3][0] = m1[3][0] + delta * (m2[3][0] - m1[3][0]);
		out[3][1] = m1[3][1] + delta * (m2[3][1] - m1[3][1]);
		out[3][2] = m1[3][2] + delta * (m2[3][2] - m1[3][2]);
		return out;
	}
}//namespace glm
