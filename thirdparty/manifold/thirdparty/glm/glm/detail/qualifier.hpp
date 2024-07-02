#pragma once

#include "setup.hpp"

namespace glm
{
	/// Qualify GLM types in term of alignment (packed, aligned) and precision in term of ULPs (lowp, mediump, highp)
	enum qualifier
	{
		packed_highp, ///< Typed data is tightly packed in memory and operations are executed with high precision in term of ULPs
		packed_mediump, ///< Typed data is tightly packed in memory  and operations are executed with medium precision in term of ULPs for higher performance
		packed_lowp, ///< Typed data is tightly packed in memory  and operations are executed with low precision in term of ULPs to maximize performance

#		if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
			aligned_highp, ///< Typed data is aligned in memory allowing SIMD optimizations and operations are executed with high precision in term of ULPs
			aligned_mediump, ///< Typed data is aligned in memory allowing SIMD optimizations and operations are executed with high precision in term of ULPs for higher performance
			aligned_lowp, // ///< Typed data is aligned in memory allowing SIMD optimizations and operations are executed with high precision in term of ULPs to maximize performance
			aligned = aligned_highp, ///< By default aligned qualifier is also high precision
#		endif

		highp = packed_highp, ///< By default highp qualifier is also packed
		mediump = packed_mediump, ///< By default mediump qualifier is also packed
		lowp = packed_lowp, ///< By default lowp qualifier is also packed
		packed = packed_highp, ///< By default packed qualifier is also high precision

#		if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE && defined(GLM_FORCE_DEFAULT_ALIGNED_GENTYPES)
			defaultp = aligned_highp
#		else
			defaultp = highp
#		endif
	};

	typedef qualifier precision;

	template<length_t L, typename T, qualifier Q = defaultp> struct vec;
	template<length_t C, length_t R, typename T, qualifier Q = defaultp> struct mat;
	template<typename T, qualifier Q = defaultp> struct qua;

#	if GLM_HAS_TEMPLATE_ALIASES
		template <typename T, qualifier Q = defaultp> using tvec1 = vec<1, T, Q>;
		template <typename T, qualifier Q = defaultp> using tvec2 = vec<2, T, Q>;
		template <typename T, qualifier Q = defaultp> using tvec3 = vec<3, T, Q>;
		template <typename T, qualifier Q = defaultp> using tvec4 = vec<4, T, Q>;
		template <typename T, qualifier Q = defaultp> using tmat2x2 = mat<2, 2, T, Q>;
		template <typename T, qualifier Q = defaultp> using tmat2x3 = mat<2, 3, T, Q>;
		template <typename T, qualifier Q = defaultp> using tmat2x4 = mat<2, 4, T, Q>;
		template <typename T, qualifier Q = defaultp> using tmat3x2 = mat<3, 2, T, Q>;
		template <typename T, qualifier Q = defaultp> using tmat3x3 = mat<3, 3, T, Q>;
		template <typename T, qualifier Q = defaultp> using tmat3x4 = mat<3, 4, T, Q>;
		template <typename T, qualifier Q = defaultp> using tmat4x2 = mat<4, 2, T, Q>;
		template <typename T, qualifier Q = defaultp> using tmat4x3 = mat<4, 3, T, Q>;
		template <typename T, qualifier Q = defaultp> using tmat4x4 = mat<4, 4, T, Q>;
		template <typename T, qualifier Q = defaultp> using tquat = qua<T, Q>;
#	endif

namespace detail
{
	template<glm::qualifier P>
	struct is_aligned
	{
		static const bool value = false;
	};

#	if GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
		template<>
		struct is_aligned<glm::aligned_lowp>
		{
			static const bool value = true;
		};

		template<>
		struct is_aligned<glm::aligned_mediump>
		{
			static const bool value = true;
		};

		template<>
		struct is_aligned<glm::aligned_highp>
		{
			static const bool value = true;
		};
#	endif

	template<length_t L, typename T, bool is_aligned>
	struct storage
	{
		typedef struct type {
			T data[L];
		} type;
	};

#	if GLM_HAS_ALIGNOF
		template<length_t L, typename T>
		struct storage<L, T, true>
		{
			typedef struct alignas(L * sizeof(T)) type {
				T data[L];
			} type;
		};

		template<typename T>
		struct storage<3, T, true>
		{
			typedef struct alignas(4 * sizeof(T)) type {
				T data[4];
			} type;
		};
#	endif

#	if GLM_ARCH & GLM_ARCH_SSE2_BIT
	template<>
	struct storage<4, float, true>
	{
		typedef glm_f32vec4 type;
	};

	template<>
	struct storage<4, int, true>
	{
		typedef glm_i32vec4 type;
	};

	template<>
	struct storage<4, unsigned int, true>
	{
		typedef glm_u32vec4 type;
	};

	template<>
	struct storage<2, double, true>
	{
		typedef glm_f64vec2 type;
	};

	template<>
	struct storage<2, detail::int64, true>
	{
		typedef glm_i64vec2 type;
	};

	template<>
	struct storage<2, detail::uint64, true>
	{
		typedef glm_u64vec2 type;
	};
#	endif
#	if (GLM_ARCH & GLM_ARCH_AVX_BIT)
	template<>
	struct storage<4, double, true>
	{
		typedef glm_f64vec4 type;
	};
#	endif

#	if (GLM_ARCH & GLM_ARCH_AVX2_BIT)
	template<>
	struct storage<4, detail::int64, true>
	{
		typedef glm_i64vec4 type;
	};

	template<>
	struct storage<4, detail::uint64, true>
	{
		typedef glm_u64vec4 type;
	};
#	endif

#	if GLM_ARCH & GLM_ARCH_NEON_BIT
	template<>
	struct storage<4, float, true>
	{
		typedef glm_f32vec4 type;
	};

	template<>
	struct storage<4, int, true>
	{
		typedef glm_i32vec4 type;
	};

	template<>
	struct storage<4, unsigned int, true>
	{
		typedef glm_u32vec4 type;
	};
#	endif

	enum genTypeEnum
	{
		GENTYPE_VEC,
		GENTYPE_MAT,
		GENTYPE_QUAT
	};

	template <typename genType>
	struct genTypeTrait
	{};

	template <length_t C, length_t R, typename T>
	struct genTypeTrait<mat<C, R, T> >
	{
		static const genTypeEnum GENTYPE = GENTYPE_MAT;
	};

	template<typename genType, genTypeEnum type>
	struct init_gentype
	{
	};

	template<typename genType>
	struct init_gentype<genType, GENTYPE_QUAT>
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static genType identity()
		{
			return genType(1, 0, 0, 0);
		}
	};

	template<typename genType>
	struct init_gentype<genType, GENTYPE_MAT>
	{
		GLM_FUNC_QUALIFIER GLM_CONSTEXPR static genType identity()
		{
			return genType(1);
		}
	};
}//namespace detail
}//namespace glm
