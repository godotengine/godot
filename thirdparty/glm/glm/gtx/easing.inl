/// @ref gtx_easing

#include <cmath>

namespace glm{

	template <typename genType>
	GLM_FUNC_QUALIFIER genType linearInterpolation(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return a;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType quadraticEaseIn(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return a * a;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType quadraticEaseOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return -(a * (a - static_cast<genType>(2)));
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType quadraticEaseInOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a < static_cast<genType>(0.5))
		{
			return static_cast<genType>(2) * a * a;
		}
		else
		{
			return (-static_cast<genType>(2) * a * a) + (4 * a) - one<genType>();
		}
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType cubicEaseIn(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return a * a * a;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType cubicEaseOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		genType const f = a - one<genType>();
		return f * f * f + one<genType>();
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType cubicEaseInOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if (a < static_cast<genType>(0.5))
		{
			return static_cast<genType>(4) * a * a * a;
		}
		else
		{
			genType const f = ((static_cast<genType>(2) * a) - static_cast<genType>(2));
			return static_cast<genType>(0.5) * f * f * f + one<genType>();
		}
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType quarticEaseIn(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return a * a * a * a;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType quarticEaseOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		genType const f = (a - one<genType>());
		return f * f * f * (one<genType>() - a) + one<genType>();
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType quarticEaseInOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a < static_cast<genType>(0.5))
		{
			return static_cast<genType>(8) * a * a * a * a;
		}
		else
		{
			genType const f = (a - one<genType>());
			return -static_cast<genType>(8) * f * f * f * f + one<genType>();
		}
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType quinticEaseIn(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return a * a * a * a * a;
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType quinticEaseOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		genType const f = (a - one<genType>());
		return f * f * f * f * f + one<genType>();
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType quinticEaseInOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a < static_cast<genType>(0.5))
		{
			return static_cast<genType>(16) * a * a * a * a * a;
		}
		else
		{
			genType const f = ((static_cast<genType>(2) * a) - static_cast<genType>(2));
			return static_cast<genType>(0.5) * f * f * f * f * f + one<genType>();
		}
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType sineEaseIn(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return sin((a - one<genType>()) * half_pi<genType>()) + one<genType>();
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType sineEaseOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return sin(a * half_pi<genType>());
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType sineEaseInOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return static_cast<genType>(0.5) * (one<genType>() - cos(a * pi<genType>()));
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType circularEaseIn(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return one<genType>() - sqrt(one<genType>() - (a * a));
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType circularEaseOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return sqrt((static_cast<genType>(2) - a) * a);
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType circularEaseInOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a < static_cast<genType>(0.5))
		{
			return static_cast<genType>(0.5) * (one<genType>() - std::sqrt(one<genType>() - static_cast<genType>(4) * (a * a)));
		}
		else
		{
			return static_cast<genType>(0.5) * (std::sqrt(-((static_cast<genType>(2) * a) - static_cast<genType>(3)) * ((static_cast<genType>(2) * a) - one<genType>())) + one<genType>());
		}
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType exponentialEaseIn(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a <= zero<genType>())
			return a;
		else
		{
			genType const Complementary = a - one<genType>();
			genType const Two = static_cast<genType>(2);
			
			return glm::pow(Two, Complementary * static_cast<genType>(10));
		}
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType exponentialEaseOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a >= one<genType>())
			return a;
		else
		{
			return one<genType>() - glm::pow(static_cast<genType>(2), -static_cast<genType>(10) * a);
		}
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType exponentialEaseInOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a < static_cast<genType>(0.5))
			return static_cast<genType>(0.5) * glm::pow(static_cast<genType>(2), (static_cast<genType>(20) * a) - static_cast<genType>(10));
		else
			return -static_cast<genType>(0.5) * glm::pow(static_cast<genType>(2), (-static_cast<genType>(20) * a) + static_cast<genType>(10)) + one<genType>();
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType elasticEaseIn(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return std::sin(static_cast<genType>(13) * half_pi<genType>() * a) * glm::pow(static_cast<genType>(2), static_cast<genType>(10) * (a - one<genType>()));
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType elasticEaseOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return std::sin(-static_cast<genType>(13) * half_pi<genType>() * (a + one<genType>())) * glm::pow(static_cast<genType>(2), -static_cast<genType>(10) * a) + one<genType>();
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType elasticEaseInOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a < static_cast<genType>(0.5))
			return static_cast<genType>(0.5) * std::sin(static_cast<genType>(13) * half_pi<genType>() * (static_cast<genType>(2) * a)) * glm::pow(static_cast<genType>(2), static_cast<genType>(10) * ((static_cast<genType>(2) * a) - one<genType>()));
		else
			return static_cast<genType>(0.5) * (std::sin(-static_cast<genType>(13) * half_pi<genType>() * ((static_cast<genType>(2) * a - one<genType>()) + one<genType>())) * glm::pow(static_cast<genType>(2), -static_cast<genType>(10) * (static_cast<genType>(2) * a - one<genType>())) + static_cast<genType>(2));
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType backEaseIn(genType const& a, genType const& o)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		genType z = ((o + one<genType>()) * a) - o;
		return (a * a * z);
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType backEaseOut(genType const& a, genType const& o)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		genType n = a - one<genType>();
		genType z = ((o + one<genType>()) * n) + o;
		return (n * n * z) + one<genType>();
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType backEaseInOut(genType const& a, genType const& o)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		genType s = o * static_cast<genType>(1.525);
		genType x = static_cast<genType>(0.5);
		genType n = a / static_cast<genType>(0.5);

		if (n < static_cast<genType>(1))
		{
			genType z = ((s + static_cast<genType>(1)) * n) - s;
			genType m = n * n * z;
			return x * m;
		}
		else 
		{
			n -= static_cast<genType>(2);
			genType z = ((s + static_cast<genType>(1)) * n) + s;
			genType m = (n*n*z) + static_cast<genType>(2);
			return x * m;
		}
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType backEaseIn(genType const& a)
	{
		return backEaseIn(a, static_cast<genType>(1.70158));
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType backEaseOut(genType const& a)
	{
		return backEaseOut(a, static_cast<genType>(1.70158));
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType backEaseInOut(genType const& a)
	{
		return backEaseInOut(a, static_cast<genType>(1.70158));
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType bounceEaseOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a < static_cast<genType>(4.0 / 11.0))
		{
			return (static_cast<genType>(121) * a * a) / static_cast<genType>(16);
		}
		else if(a < static_cast<genType>(8.0 / 11.0))
		{
			return (static_cast<genType>(363.0 / 40.0) * a * a) - (static_cast<genType>(99.0 / 10.0) * a) + static_cast<genType>(17.0 / 5.0);
		}
		else if(a < static_cast<genType>(9.0 / 10.0))
		{
			return (static_cast<genType>(4356.0 / 361.0) * a * a) - (static_cast<genType>(35442.0 / 1805.0) * a) + static_cast<genType>(16061.0 / 1805.0);
		}
		else
		{
			return (static_cast<genType>(54.0 / 5.0) * a * a) - (static_cast<genType>(513.0 / 25.0) * a) + static_cast<genType>(268.0 / 25.0);
		}
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType bounceEaseIn(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		return one<genType>() - bounceEaseOut(one<genType>() - a);
	}

	template <typename genType>
	GLM_FUNC_QUALIFIER genType bounceEaseInOut(genType const& a)
	{
		// Only defined in [0, 1]
		assert(a >= zero<genType>());
		assert(a <= one<genType>());

		if(a < static_cast<genType>(0.5))
		{
			return static_cast<genType>(0.5) * (one<genType>() - bounceEaseOut(one<genType>() - a * static_cast<genType>(2)));
		}
		else
		{
			return static_cast<genType>(0.5) * bounceEaseOut(a * static_cast<genType>(2) - one<genType>()) + static_cast<genType>(0.5);
		}
	}

}//namespace glm
