#include "../geometric.hpp"
#include "../exponential.hpp"
#include "../trigonometric.hpp"
#include "../detail/type_vec1.hpp"
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cmath>

namespace glm{
namespace detail
{
	template <length_t L, typename T, qualifier Q>
	struct compute_rand
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call();
	};

	template <qualifier P>
	struct compute_rand<1, uint8, P>
	{
		GLM_FUNC_QUALIFIER static vec<1, uint8, P> call()
		{
			return vec<1, uint8, P>(
				static_cast<uint8>(std::rand() % std::numeric_limits<uint8>::max()));
		}
	};

	template <qualifier P>
	struct compute_rand<2, uint8, P>
	{
		GLM_FUNC_QUALIFIER static vec<2, uint8, P> call()
		{
			return vec<2, uint8, P>(
				std::rand() % std::numeric_limits<uint8>::max(),
				std::rand() % std::numeric_limits<uint8>::max());
		}
	};

	template <qualifier P>
	struct compute_rand<3, uint8, P>
	{
		GLM_FUNC_QUALIFIER static vec<3, uint8, P> call()
		{
			return vec<3, uint8, P>(
				std::rand() % std::numeric_limits<uint8>::max(),
				std::rand() % std::numeric_limits<uint8>::max(),
				std::rand() % std::numeric_limits<uint8>::max());
		}
	};

	template <qualifier P>
	struct compute_rand<4, uint8, P>
	{
		GLM_FUNC_QUALIFIER static vec<4, uint8, P> call()
		{
			return vec<4, uint8, P>(
				std::rand() % std::numeric_limits<uint8>::max(),
				std::rand() % std::numeric_limits<uint8>::max(),
				std::rand() % std::numeric_limits<uint8>::max(),
				std::rand() % std::numeric_limits<uint8>::max());
		}
	};

	template <length_t L, qualifier Q>
	struct compute_rand<L, uint16, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, uint16, Q> call()
		{
			return
				(vec<L, uint16, Q>(compute_rand<L, uint8, Q>::call()) << static_cast<uint16>(8)) |
				(vec<L, uint16, Q>(compute_rand<L, uint8, Q>::call()) << static_cast<uint16>(0));
		}
	};

	template <length_t L, qualifier Q>
	struct compute_rand<L, uint32, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, uint32, Q> call()
		{
			return
				(vec<L, uint32, Q>(compute_rand<L, uint16, Q>::call()) << static_cast<uint32>(16)) |
				(vec<L, uint32, Q>(compute_rand<L, uint16, Q>::call()) << static_cast<uint32>(0));
		}
	};

	template <length_t L, qualifier Q>
	struct compute_rand<L, uint64, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, uint64, Q> call()
		{
			return
				(vec<L, uint64, Q>(compute_rand<L, uint32, Q>::call()) << static_cast<uint64>(32)) |
				(vec<L, uint64, Q>(compute_rand<L, uint32, Q>::call()) << static_cast<uint64>(0));
		}
	};

	template <length_t L, typename T, qualifier Q>
	struct compute_linearRand
	{
		GLM_FUNC_QUALIFIER static vec<L, T, Q> call(vec<L, T, Q> const& Min, vec<L, T, Q> const& Max);
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, int8, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, int8, Q> call(vec<L, int8, Q> const& Min, vec<L, int8, Q> const& Max)
		{
			return (vec<L, int8, Q>(compute_rand<L, uint8, Q>::call() % vec<L, uint8, Q>(Max + static_cast<int8>(1) - Min))) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, uint8, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, uint8, Q> call(vec<L, uint8, Q> const& Min, vec<L, uint8, Q> const& Max)
		{
			return (compute_rand<L, uint8, Q>::call() % (Max + static_cast<uint8>(1) - Min)) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, int16, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, int16, Q> call(vec<L, int16, Q> const& Min, vec<L, int16, Q> const& Max)
		{
			return (vec<L, int16, Q>(compute_rand<L, uint16, Q>::call() % vec<L, uint16, Q>(Max + static_cast<int16>(1) - Min))) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, uint16, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, uint16, Q> call(vec<L, uint16, Q> const& Min, vec<L, uint16, Q> const& Max)
		{
			return (compute_rand<L, uint16, Q>::call() % (Max + static_cast<uint16>(1) - Min)) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, int32, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, int32, Q> call(vec<L, int32, Q> const& Min, vec<L, int32, Q> const& Max)
		{
			return (vec<L, int32, Q>(compute_rand<L, uint32, Q>::call() % vec<L, uint32, Q>(Max + static_cast<int32>(1) - Min))) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, uint32, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, uint32, Q> call(vec<L, uint32, Q> const& Min, vec<L, uint32, Q> const& Max)
		{
			return (compute_rand<L, uint32, Q>::call() % (Max + static_cast<uint32>(1) - Min)) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, int64, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, int64, Q> call(vec<L, int64, Q> const& Min, vec<L, int64, Q> const& Max)
		{
			return (vec<L, int64, Q>(compute_rand<L, uint64, Q>::call() % vec<L, uint64, Q>(Max + static_cast<int64>(1) - Min))) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, uint64, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, uint64, Q> call(vec<L, uint64, Q> const& Min, vec<L, uint64, Q> const& Max)
		{
			return (compute_rand<L, uint64, Q>::call() % (Max + static_cast<uint64>(1) - Min)) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, float, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, float, Q> call(vec<L, float, Q> const& Min, vec<L, float, Q> const& Max)
		{
			return vec<L, float, Q>(compute_rand<L, uint32, Q>::call()) / static_cast<float>(std::numeric_limits<uint32>::max()) * (Max - Min) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, double, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, double, Q> call(vec<L, double, Q> const& Min, vec<L, double, Q> const& Max)
		{
			return vec<L, double, Q>(compute_rand<L, uint64, Q>::call()) / static_cast<double>(std::numeric_limits<uint64>::max()) * (Max - Min) + Min;
		}
	};

	template<length_t L, qualifier Q>
	struct compute_linearRand<L, long double, Q>
	{
		GLM_FUNC_QUALIFIER static vec<L, long double, Q> call(vec<L, long double, Q> const& Min, vec<L, long double, Q> const& Max)
		{
			return vec<L, long double, Q>(compute_rand<L, uint64, Q>::call()) / static_cast<long double>(std::numeric_limits<uint64>::max()) * (Max - Min) + Min;
		}
	};
}//namespace detail

	template<typename genType>
	GLM_FUNC_QUALIFIER genType linearRand(genType Min, genType Max)
	{
		return detail::compute_linearRand<1, genType, highp>::call(
			vec<1, genType, highp>(Min),
			vec<1, genType, highp>(Max)).x;
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> linearRand(vec<L, T, Q> const& Min, vec<L, T, Q> const& Max)
	{
		return detail::compute_linearRand<L, T, Q>::call(Min, Max);
	}

	template<typename genType>
	GLM_FUNC_QUALIFIER genType gaussRand(genType Mean, genType Deviation)
	{
		genType w, x1, x2;

		do
		{
			x1 = linearRand(genType(-1), genType(1));
			x2 = linearRand(genType(-1), genType(1));

			w = x1 * x1 + x2 * x2;
		} while(w > genType(1));

		return static_cast<genType>(x2 * Deviation * Deviation * sqrt((genType(-2) * log(w)) / w) + Mean);
	}

	template<length_t L, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<L, T, Q> gaussRand(vec<L, T, Q> const& Mean, vec<L, T, Q> const& Deviation)
	{
		return detail::functor2<vec, L, T, Q>::call(gaussRand, Mean, Deviation);
	}

	template<typename T>
	GLM_FUNC_QUALIFIER vec<2, T, defaultp> diskRand(T Radius)
	{
		assert(Radius > static_cast<T>(0));

		vec<2, T, defaultp> Result(T(0));
		T LenRadius(T(0));

		do
		{
			Result = linearRand(
				vec<2, T, defaultp>(-Radius),
				vec<2, T, defaultp>(Radius));
			LenRadius = length(Result);
		}
		while(LenRadius > Radius);

		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER vec<3, T, defaultp> ballRand(T Radius)
	{
		assert(Radius > static_cast<T>(0));

		vec<3, T, defaultp> Result(T(0));
		T LenRadius(T(0));

		do
		{
			Result = linearRand(
				vec<3, T, defaultp>(-Radius),
				vec<3, T, defaultp>(Radius));
			LenRadius = length(Result);
		}
		while(LenRadius > Radius);

		return Result;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER vec<2, T, defaultp> circularRand(T Radius)
	{
		assert(Radius > static_cast<T>(0));

		T a = linearRand(T(0), static_cast<T>(6.283185307179586476925286766559));
		return vec<2, T, defaultp>(glm::cos(a), glm::sin(a)) * Radius;
	}

	template<typename T>
	GLM_FUNC_QUALIFIER vec<3, T, defaultp> sphericalRand(T Radius)
	{
		assert(Radius > static_cast<T>(0));

		T theta = linearRand(T(0), T(6.283185307179586476925286766559f));
		T phi = std::acos(linearRand(T(-1.0f), T(1.0f)));

		T x = std::sin(phi) * std::cos(theta);
		T y = std::sin(phi) * std::sin(theta);
		T z = std::cos(phi);

		return vec<3, T, defaultp>(x, y, z) * Radius;
	}
}//namespace glm
