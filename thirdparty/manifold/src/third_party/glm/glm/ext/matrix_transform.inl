namespace glm
{
	template<typename genType>
	GLM_FUNC_QUALIFIER GLM_CONSTEXPR genType identity()
	{
		return detail::init_gentype<genType, detail::genTypeTrait<genType>::GENTYPE>::identity();
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> translate(mat<4, 4, T, Q> const& m, vec<3, T, Q> const& v)
	{
		mat<4, 4, T, Q> Result(m);
		Result[3] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2] + m[3];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> rotate(mat<4, 4, T, Q> const& m, T angle, vec<3, T, Q> const& v)
	{
		T const a = angle;
		T const c = cos(a);
		T const s = sin(a);

		vec<3, T, Q> axis(normalize(v));
		vec<3, T, Q> temp((T(1) - c) * axis);

		mat<4, 4, T, Q> Rotate;
		Rotate[0][0] = c + temp[0] * axis[0];
		Rotate[0][1] = temp[0] * axis[1] + s * axis[2];
		Rotate[0][2] = temp[0] * axis[2] - s * axis[1];

		Rotate[1][0] = temp[1] * axis[0] - s * axis[2];
		Rotate[1][1] = c + temp[1] * axis[1];
		Rotate[1][2] = temp[1] * axis[2] + s * axis[0];

		Rotate[2][0] = temp[2] * axis[0] + s * axis[1];
		Rotate[2][1] = temp[2] * axis[1] - s * axis[0];
		Rotate[2][2] = c + temp[2] * axis[2];

		mat<4, 4, T, Q> Result;
		Result[0] = m[0] * Rotate[0][0] + m[1] * Rotate[0][1] + m[2] * Rotate[0][2];
		Result[1] = m[0] * Rotate[1][0] + m[1] * Rotate[1][1] + m[2] * Rotate[1][2];
		Result[2] = m[0] * Rotate[2][0] + m[1] * Rotate[2][1] + m[2] * Rotate[2][2];
		Result[3] = m[3];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> rotate_slow(mat<4, 4, T, Q> const& m, T angle, vec<3, T, Q> const& v)
	{
		T const a = angle;
		T const c = cos(a);
		T const s = sin(a);
		mat<4, 4, T, Q> Result;

		vec<3, T, Q> axis = normalize(v);

		Result[0][0] = c + (static_cast<T>(1) - c)      * axis.x     * axis.x;
		Result[0][1] = (static_cast<T>(1) - c) * axis.x * axis.y + s * axis.z;
		Result[0][2] = (static_cast<T>(1) - c) * axis.x * axis.z - s * axis.y;
		Result[0][3] = static_cast<T>(0);

		Result[1][0] = (static_cast<T>(1) - c) * axis.y * axis.x - s * axis.z;
		Result[1][1] = c + (static_cast<T>(1) - c) * axis.y * axis.y;
		Result[1][2] = (static_cast<T>(1) - c) * axis.y * axis.z + s * axis.x;
		Result[1][3] = static_cast<T>(0);

		Result[2][0] = (static_cast<T>(1) - c) * axis.z * axis.x + s * axis.y;
		Result[2][1] = (static_cast<T>(1) - c) * axis.z * axis.y - s * axis.x;
		Result[2][2] = c + (static_cast<T>(1) - c) * axis.z * axis.z;
		Result[2][3] = static_cast<T>(0);

		Result[3] = vec<4, T, Q>(0, 0, 0, 1);
		return m * Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> scale(mat<4, 4, T, Q> const& m, vec<3, T, Q> const& v)
	{
		mat<4, 4, T, Q> Result;
		Result[0] = m[0] * v[0];
		Result[1] = m[1] * v[1];
		Result[2] = m[2] * v[2];
		Result[3] = m[3];
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> scale_slow(mat<4, 4, T, Q> const& m, vec<3, T, Q> const& v)
	{
		mat<4, 4, T, Q> Result(T(1));
		Result[0][0] = v.x;
		Result[1][1] = v.y;
		Result[2][2] = v.z;
		return m * Result;
	}

    template <typename T, qualifier Q>
    GLM_FUNC_QUALIFIER mat<4, 4, T, Q> shear(mat<4, 4, T, Q> const &m, vec<3, T, Q> const& p, vec<2, T, Q> const &l_x, vec<2, T, Q> const &l_y, vec<2, T, Q> const &l_z)
    {
        T const lambda_xy = l_x[0];
        T const lambda_xz = l_x[1];
        T const lambda_yx = l_y[0];
        T const lambda_yz = l_y[1];
        T const lambda_zx = l_z[0];
        T const lambda_zy = l_z[1];

        vec<3, T, Q> point_lambda = vec<3, T, Q>(
            (lambda_xy + lambda_xz), (lambda_yx + lambda_yz), (lambda_zx + lambda_zy)
        );

        mat<4, 4, T, Q> Shear = mat<4, 4, T, Q>(
            1                      , lambda_yx              , lambda_zx              , 0,
            lambda_xy              , 1                      , lambda_zy              , 0,
            lambda_xz              , lambda_yz              , 1                      , 0,
            -point_lambda[0] * p[0], -point_lambda[1] * p[1], -point_lambda[2] * p[2], 1
        );

        mat<4, 4, T, Q> Result;
        Result[0] = Shear[0] * m[0][0] + Shear[1] * m[0][1] + Shear[2] * m[0][2] + Shear[3] * m[0][3];
        Result[1] = Shear[0] * m[1][0] + Shear[1] * m[1][1] + Shear[2] * m[1][2] + Shear[3] * m[1][3];
        Result[2] = Shear[0] * m[2][0] + Shear[1] * m[2][1] + Shear[2] * m[2][2] + Shear[3] * m[2][3];
        Result[3] = Shear[0] * m[3][0] + Shear[1] * m[3][1] + Shear[2] * m[3][2] + Shear[3] * m[3][3];
        return Result;
    }

    template <typename T, qualifier Q>
    GLM_FUNC_QUALIFIER mat<4, 4, T, Q> shear_slow(mat<4, 4, T, Q> const &m, vec<3, T, Q> const& p, vec<2, T, Q> const &l_x, vec<2, T, Q> const &l_y, vec<2, T, Q> const &l_z)
    {
        T const lambda_xy = static_cast<T>(l_x[0]);
        T const lambda_xz = static_cast<T>(l_x[1]);
        T const lambda_yx = static_cast<T>(l_y[0]);
        T const lambda_yz = static_cast<T>(l_y[1]);
        T const lambda_zx = static_cast<T>(l_z[0]);
        T const lambda_zy = static_cast<T>(l_z[1]);

        vec<3, T, Q> point_lambda = vec<3, T, Q>(
            static_cast<T>(lambda_xy + lambda_xz),
            static_cast<T>(lambda_yx + lambda_yz),
            static_cast<T>(lambda_zx + lambda_zy)
        );

        mat<4, 4, T, Q> Shear = mat<4, 4, T, Q>(
            1                      , lambda_yx              , lambda_zx              , 0,
            lambda_xy              , 1                      , lambda_zy              , 0,
            lambda_xz              , lambda_yz              , 1                      , 0,
            -point_lambda[0] * p[0], -point_lambda[1] * p[1], -point_lambda[2] * p[2], 1
        );
        return m * Shear;
    }

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> lookAtRH(vec<3, T, Q> const& eye, vec<3, T, Q> const& center, vec<3, T, Q> const& up)
	{
		vec<3, T, Q> const f(normalize(center - eye));
		vec<3, T, Q> const s(normalize(cross(f, up)));
		vec<3, T, Q> const u(cross(s, f));

		mat<4, 4, T, Q> Result(1);
		Result[0][0] = s.x;
		Result[1][0] = s.y;
		Result[2][0] = s.z;
		Result[0][1] = u.x;
		Result[1][1] = u.y;
		Result[2][1] = u.z;
		Result[0][2] =-f.x;
		Result[1][2] =-f.y;
		Result[2][2] =-f.z;
		Result[3][0] =-dot(s, eye);
		Result[3][1] =-dot(u, eye);
		Result[3][2] = dot(f, eye);
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> lookAtLH(vec<3, T, Q> const& eye, vec<3, T, Q> const& center, vec<3, T, Q> const& up)
	{
		vec<3, T, Q> const f(normalize(center - eye));
		vec<3, T, Q> const s(normalize(cross(up, f)));
		vec<3, T, Q> const u(cross(f, s));

		mat<4, 4, T, Q> Result(1);
		Result[0][0] = s.x;
		Result[1][0] = s.y;
		Result[2][0] = s.z;
		Result[0][1] = u.x;
		Result[1][1] = u.y;
		Result[2][1] = u.z;
		Result[0][2] = f.x;
		Result[1][2] = f.y;
		Result[2][2] = f.z;
		Result[3][0] = -dot(s, eye);
		Result[3][1] = -dot(u, eye);
		Result[3][2] = -dot(f, eye);
		return Result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<4, 4, T, Q> lookAt(vec<3, T, Q> const& eye, vec<3, T, Q> const& center, vec<3, T, Q> const& up)
	{
#       if (GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT)
            return lookAtLH(eye, center, up);
#       else
            return lookAtRH(eye, center, up);
#       endif
	}
}//namespace glm
