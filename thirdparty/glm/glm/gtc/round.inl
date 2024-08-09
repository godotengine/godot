/// @ref gtc_round

#include "../integer.hpp"
#include "../ext/vector_integer.hpp"

namespace glm{
namespace detail
{
	template<bool is_float, bool is_signed>
	struct compute_roundMultiple {};

	template<>
	struct compute_roundMultiple<true, true>
	{
		template<typename genType>
		GLM_FUNC_QUALIFIER static genType call(genType Source, genType Multiple)
		{
			if (Source >= genType(0))
				return Source - std::fmod(Source, Multiple);
			else
			{
				genType Tmp = Source + genType(1);
				return Tmp - std::fmod(Tmp, Multiple) - Multiple;
			}
		}
	};

	template<>
	struct compute_roundMultiple<false, false>
	{
		template<typename genType>
		GLM_FUNC_QUALIFIER static genType call(genType Source, genType Multiple)
		{
			if (Source >= genType(0))
				return Source - Source % Multiple;
			else
			{
				genType Tmp = Source + genType(1);
				return Tmp - Tmp % Multiple - Multiple;
			}
		}
	};

	template<>
	struct compute_roundMultiple<false, true>
	{
		template<typename genType>
		GLM_FUNC_QUALIFIER static genType call(genType Source, genType Multiple)
		{
			if (Source >= genType(0))
				return Source - Source % Multiple;
			else
			{
				genType Tmp = Source + genType(1);
				return Tmp - Tmp % Multiple - Multiple;
			}
		}
	};
}//namespace detail

	//////////////////
	// ceilPowerOfTwo

	template<typename genType>
	GLM_FUNC_QUALIFIER genType ceilPowerOfTwo(genType value)
	{
		return detail::compute_ceilPowerOfTwo<1, genType, defaultp, std::numeric_limits<genType>::is_signed>::call(vec<1, genType, defaultp>(value)).x;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> ceilPowerOfTwo(vec<L, T, Q> const& v)
	{
		return detail::compute_ceilPowerOfTwo<L, T, Q, std::numeric_limits<T>::is_signed>::call(v);
	}

	///////////////////
	// floorPowerOfTwo

	template<typename genType>
	GLM_FUNC_QUALIFIER genType floorPowerOfTwo(genType value)
	{
		return isPowerOfTwo(value) ? value : static_cast<genType>(1) << findMSB(value);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> floorPowerOfTwo(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(floorPowerOfTwo, v);
	}

	///////////////////
	// roundPowerOfTwo

	template<typename genIUType>
	GLM_FUNC_QUALIFIER genIUType roundPowerOfTwo(genIUType value)
	{
		if(isPowerOfTwo(value))
			return value;

		genIUType const prev = static_cast<genIUType>(1) << findMSB(value);
		genIUType const next = prev << static_cast<genIUType>(1);
		return (next - value) < (value - prev) ? next : prev;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> roundPowerOfTwo(vec<L, T, Q> const& v)
	{
		return detail::functor1<vec, L, T, T, Q>::call(roundPowerOfTwo, v);
	}

	//////////////////////
	// ceilMultiple

	template<typename genType>
	GLM_FUNC_QUALIFIER genType ceilMultiple(genType Source, genType Multiple)
	{
		return detail::compute_ceilMultiple<std::numeric_limits<genType>::is_iec559, std::numeric_limits<genType>::is_signed>::call(Source, Multiple);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> ceilMultiple(vec<L, T, Q> const& Source, vec<L, T, Q> const& Multiple)
	{
		return detail::functor2<vec, L, T, Q>::call(ceilMultiple, Source, Multiple);
	}

	//////////////////////
	// floorMultiple

	template<typename genType>
	GLM_FUNC_QUALIFIER genType floorMultiple(genType Source, genType Multiple)
	{
		return detail::compute_floorMultiple<std::numeric_limits<genType>::is_iec559, std::numeric_limits<genType>::is_signed>::call(Source, Multiple);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> floorMultiple(vec<L, T, Q> const& Source, vec<L, T, Q> const& Multiple)
	{
		return detail::functor2<vec, L, T, Q>::call(floorMultiple, Source, Multiple);
	}

	//////////////////////
	// roundMultiple

	template<typename genType>
	GLM_FUNC_QUALIFIER genType roundMultiple(genType Source, genType Multiple)
	{
		return detail::compute_roundMultiple<std::numeric_limits<genType>::is_iec559, std::numeric_limits<genType>::is_signed>::call(Source, Multiple);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> roundMultiple(vec<L, T, Q> const& Source, vec<L, T, Q> const& Multiple)
	{
		return detail::functor2<vec, L, T, Q>::call(roundMultiple, Source, Multiple);
	}
}//namespace glm
