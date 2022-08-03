/// @ref gtx_color_space_YCoCg

namespace glm
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> rgb2YCoCg
	(
		vec<3, T, Q> const& rgbColor
	)
	{
		vec<3, T, Q> result;
		result.x/*Y */ =   rgbColor.r / T(4) + rgbColor.g / T(2) + rgbColor.b / T(4);
		result.y/*Co*/ =   rgbColor.r / T(2) + rgbColor.g * T(0) - rgbColor.b / T(2);
		result.z/*Cg*/ = - rgbColor.r / T(4) + rgbColor.g / T(2) - rgbColor.b / T(4);
		return result;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> YCoCg2rgb
	(
		vec<3, T, Q> const& YCoCgColor
	)
	{
		vec<3, T, Q> result;
		result.r = YCoCgColor.x + YCoCgColor.y - YCoCgColor.z;
		result.g = YCoCgColor.x				   + YCoCgColor.z;
		result.b = YCoCgColor.x - YCoCgColor.y - YCoCgColor.z;
		return result;
	}

	template<typename T, qualifier Q, bool isInteger>
	class compute_YCoCgR {
	public:
		static GLM_FUNC_QUALIFIER vec<3, T, Q> rgb2YCoCgR
		(
			vec<3, T, Q> const& rgbColor
		)
		{
			vec<3, T, Q> result;
			result.x/*Y */ = rgbColor.g * static_cast<T>(0.5) + (rgbColor.r + rgbColor.b) * static_cast<T>(0.25);
			result.y/*Co*/ = rgbColor.r - rgbColor.b;
			result.z/*Cg*/ = rgbColor.g - (rgbColor.r + rgbColor.b) * static_cast<T>(0.5);
			return result;
		}

		static GLM_FUNC_QUALIFIER vec<3, T, Q> YCoCgR2rgb
		(
			vec<3, T, Q> const& YCoCgRColor
		)
		{
			vec<3, T, Q> result;
			T tmp = YCoCgRColor.x - (YCoCgRColor.z * static_cast<T>(0.5));
			result.g = YCoCgRColor.z + tmp;
			result.b = tmp - (YCoCgRColor.y * static_cast<T>(0.5));
			result.r = result.b + YCoCgRColor.y;
			return result;
		}
	};

	template<typename T, qualifier Q>
	class compute_YCoCgR<T, Q, true> {
	public:
		static GLM_FUNC_QUALIFIER vec<3, T, Q> rgb2YCoCgR
		(
			vec<3, T, Q> const& rgbColor
		)
		{
			vec<3, T, Q> result;
			result.y/*Co*/ = rgbColor.r - rgbColor.b;
			T tmp = rgbColor.b + (result.y >> 1);
			result.z/*Cg*/ = rgbColor.g - tmp;
			result.x/*Y */ = tmp + (result.z >> 1);
			return result;
		}

		static GLM_FUNC_QUALIFIER vec<3, T, Q> YCoCgR2rgb
		(
			vec<3, T, Q> const& YCoCgRColor
		)
		{
			vec<3, T, Q> result;
			T tmp = YCoCgRColor.x - (YCoCgRColor.z >> 1);
			result.g = YCoCgRColor.z + tmp;
			result.b = tmp - (YCoCgRColor.y >> 1);
			result.r = result.b + YCoCgRColor.y;
			return result;
		}
	};

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> rgb2YCoCgR
	(
		vec<3, T, Q> const& rgbColor
	)
	{
		return compute_YCoCgR<T, Q, std::numeric_limits<T>::is_integer>::rgb2YCoCgR(rgbColor);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<3, T, Q> YCoCgR2rgb
	(
		vec<3, T, Q> const& YCoCgRColor
	)
	{
		return compute_YCoCgR<T, Q, std::numeric_limits<T>::is_integer>::YCoCgR2rgb(YCoCgRColor);
	}
}//namespace glm
