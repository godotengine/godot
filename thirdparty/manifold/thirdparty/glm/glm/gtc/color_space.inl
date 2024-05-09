/// @ref gtc_color_space

namespace glm{
namespace detail
{
	template<length_t L, typename T, qualifier Q>
	struct compute_rgbToSrgb
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& ColorRGB, T GammaCorrection)
		{
			vec<L, T, Q> const ClampedColor(clamp(ColorRGB, static_cast<T>(0), static_cast<T>(1)));

			return mix(
				pow(ClampedColor, vec<L, T, Q>(GammaCorrection)) * static_cast<T>(1.055) - static_cast<T>(0.055),
				ClampedColor * static_cast<T>(12.92),
				lessThan(ClampedColor, vec<L, T, Q>(static_cast<T>(0.0031308))));
		}
	};

	template<typename T, qualifier Q>
	struct compute_rgbToSrgb<4, T, Q>
	{
		GLM_FUNC_QUALIFIER static vec<4, T, Q> call(vec<4, T, Q> const& ColorRGB, T GammaCorrection)
		{
			return vec<4, T, Q>(compute_rgbToSrgb<3, T, Q>::call(vec<3, T, Q>(ColorRGB), GammaCorrection), ColorRGB.w);
		}
	};

	template<length_t L, typename T, qualifier Q>
	struct compute_srgbToRgb
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& ColorSRGB, T Gamma)
		{
			return mix(
				pow((ColorSRGB + static_cast<T>(0.055)) * static_cast<T>(0.94786729857819905213270142180095), vec<L, T, Q>(Gamma)),
				ColorSRGB * static_cast<T>(0.07739938080495356037151702786378),
				lessThanEqual(ColorSRGB, vec<L, T, Q>(static_cast<T>(0.04045))));
		}
	};

	template<typename T, qualifier Q>
	struct compute_srgbToRgb<4, T, Q>
	{
		GLM_FUNC_QUALIFIER static vec<4, T, Q> call(vec<4, T, Q> const& ColorSRGB, T Gamma)
		{
			return vec<4, T, Q>(compute_srgbToRgb<3, T, Q>::call(vec<3, T, Q>(ColorSRGB), Gamma), ColorSRGB.w);
		}
	};
}//namespace detail

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> convertLinearToSRGB(vec<L, T, Q> const& ColorLinear)
	{
		return detail::compute_rgbToSrgb<L, T, Q>::call(ColorLinear, static_cast<T>(0.41666));
	}

	// Based on Ian Taylor http://chilliant.blogspot.fr/2012/08/srgb-approximations-for-hlsl.html
	template<>
	GLM_FUNC_QUALIFIER vec<3, float, lowp> convertLinearToSRGB(vec<3, float, lowp> const& ColorLinear)
	{
		vec<3, float, lowp> S1 = sqrt(ColorLinear);
		vec<3, float, lowp> S2 = sqrt(S1);
		vec<3, float, lowp> S3 = sqrt(S2);
		return 0.662002687f * S1 + 0.684122060f * S2 - 0.323583601f * S3 - 0.0225411470f * ColorLinear;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> convertLinearToSRGB(vec<L, T, Q> const& ColorLinear, T Gamma)
	{
		return detail::compute_rgbToSrgb<L, T, Q>::call(ColorLinear, static_cast<T>(1) / Gamma);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> convertSRGBToLinear(vec<L, T, Q> const& ColorSRGB)
	{
		return detail::compute_srgbToRgb<L, T, Q>::call(ColorSRGB, static_cast<T>(2.4));
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> convertSRGBToLinear(vec<L, T, Q> const& ColorSRGB, T Gamma)
	{
		return detail::compute_srgbToRgb<L, T, Q>::call(ColorSRGB, Gamma);
	}
}//namespace glm
