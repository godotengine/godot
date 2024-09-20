#pragma once

#include "vec-scalar.h"

namespace math
{
namespace meta
{
    struct v4i
    {
        typedef __m128i packed;
        typedef int     type;

#   if defined(_MSC_VER)
        // patch for Visual C++
        typedef const __m128i&  __m128i_arg;
#   else
        typedef __m128i         __m128i_arg;
#   endif

        enum
        {
            NOP = 0,
            MOV,
            MOV_R,
            UNPCKL,
            UNPCKL_R,
            UNPCKH,
            UNPCKH_R,
            SHUF,
            SHUF_R,
#       if defined(__SSE2__) && !defined(__SSE4_1__)
            SLLI1,
            SLLI2,
            SLLI3,
            SRLI1,
            SRLI2,
            SRLI3,
#       endif
            BLEND,
            BLEND_R,
            CLEAR,
#       if defined(__SSSE3__)
            PALIGNR1,
            PALIGNR1_R,
            PALIGNR3,
            PALIGNR3_R,
#       endif
#       if defined(__SSE4_1__)
            EXTRACT,
            INSERT,
            INSERT_R,
#       endif
            DEFAULT
        };

        enum
        {
#       if defined(__SSE4_1__)
            EXTRACT_COST = 2,
            INSERT_COST = 3,
            BLEND_COST = 3,
#       endif
#       if defined(__SSSE3__)
            PALIGNR_COST = 2,
            ABS_COST = 1,
#       endif
#       if defined(__AVX2__)
            PERM_COST = 1,
#       elif defined(__SSE2__)
            PERM_COST = 2,
#       else
            PERM_COST = 3,
#       endif
            SHUF_COST = 3,
#       if defined(__SSE2__)
            SLLI_COST = 2,
            SRLI_COST = 2,
            SRAI_COST = 2,
#       endif
            UNPCKL_COST = 2,
            UNPCKH_COST = 2,
#       if defined(__SSE2__)
            ADD_COST = 2,
            SUB_COST = 2,
            CMP_COST = 2,
            AND_COST = 2,
            ANDN_COST = 2,
            XOR_COST = 2,
            OR_COST = 2,
#       else
            ADD_COST = 8,
            SUB_COST = 8,
            CMP_COST = 8,
            AND_COST = 3,
            ANDN_COST = 3,
            XOR_COST = 3,
            OR_COST = 3,
#       endif
#       if defined(__SSE4_1__)
            MUL_COST = 4,
            MIN_COST = 4,
            MAX_COST = 4,
#       else
            MUL_COST = 16,
#       endif
            DIV_COST = 32
        };

        static MATH_FORCEINLINE __m128i ZERO()
        {
            return _mm_setzero_si128();
        }

#   if defined(__clang__)

        static MATH_FORCEINLINE __m128i CTOR(int x)
        {
            return ((const int __attribute__ ((ext_vector_type(4)))) { x, x, x, x });
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y)
        {
            return ((const int __attribute__ ((ext_vector_type(4)))) { x, y, x, y });
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y, int z)
        {
            return ((const int __attribute__ ((ext_vector_type(4)))) { x, y, z, 0 });
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y, int z, int w)
        {
            return ((const int __attribute__ ((ext_vector_type(4)))) { x, y, z, w });
        }

#   elif defined(__GNUC__)

        static MATH_FORCEINLINE __m128i CTOR(int x)
        {
            return (__m128i)((const int __attribute__ ((vector_size(16)))) { x, x, x, x });
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y)
        {
            return (__m128i)((const int __attribute__ ((vector_size(16)))) { x, y, x, y });
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y, int z)
        {
            return (__m128i)((const int __attribute__ ((vector_size(16)))) { x, y, z, 0 });
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y, int z, int w)
        {
            return (__m128i)((const int __attribute__ ((vector_size(16)))) { x, y, z, w });
        }

#   elif defined(__SSE2__)

        static MATH_FORCEINLINE __m128i CTOR(int x)
        {
            return _mm_set1_epi32(x);
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y)
        {
            return _mm_set_epi32(y, x, y, x);
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y, int z)
        {
            return _mm_set_epi32(0, z, y, x);
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y, int z, int w)
        {
            return _mm_set_epi32(w, z, y, x);
        }

#   else

        static MATH_FORCEINLINE __m128i CTOR(int x)
        {
            union { __m128i p; int i[4]; } u; u.i[0] = u.i[1] = u.i[2] = u.i[3] = x;
            return u.p;
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y)
        {
            union { __m128i p; int i[4]; } u; u.i[0] = u.i[1] = x; u.i[2] = u.i[3] = y;
            return u.p;
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y, int z)
        {
            union { __m128i p; int i[4]; } u; u.i[0] = x; u.i[1] = y; u.i[2] = z; u.i[3] = 0;
            return u.p;
        }

        static MATH_FORCEINLINE __m128i CTOR(int x, int y, int z, int w)
        {
            union { __m128i p; int i[4]; } u; u.i[0] = x; u.i[1] = y; u.i[2] = z; u.i[3] = w;
            return u.p;
        }

#   endif

        template<unsigned SWZ> struct SWIZ
        {
            enum
            {
                SCL = ISDUP(SWZ),

                IGN = IGNORED(SWZ),
                MSK = MASK(SWZ),
                USE = USED(SWZ),

                OP =
                    SCL ? MOV :
                    MATCH_SWIZ(0, SWZ, SWZ_XYZW, MSK, IGN) ? MOV :
#           ifdef META_PEEPHOLE
#               if defined(__SSE2__) && !defined(__SSE4_1__)
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_0), MSK, IGN) ? SRLI1 :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_Z, COMP_W, COMP_0, COMP_0), MSK, IGN) ? SRLI2 :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_W, COMP_0, COMP_0, COMP_0), MSK, IGN) ? SRLI3 :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_0, COMP_X, COMP_Y, COMP_Z), MSK, IGN) ? SLLI1 :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_0, COMP_0, COMP_X, COMP_Y), MSK, IGN) ? SLLI2 :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_0, COMP_0, COMP_0, COMP_X), MSK, IGN) ? SLLI3 :
#               endif
#               if !defined(__SSE2__)
                    MATCH_SWIZ(0, SWZ, SWZ_XXYY, MSK, IGN) ? UNPCKL :
                    MATCH_SWIZ(0, SWZ, SWZ_ZZWW, MSK, IGN) ? UNPCKH :
#               endif
#           endif
                    DEFAULT
            };

            template_decl(struct load, int op)
            {
                enum
                {
                    DEF = COMP(SWZ, FFS(SWZ)),
                    EXT = SWZ | (SWZ(DEF, DEF, DEF, DEF) & ~MSK),
                    X = COMP(EXT, 0) - COMP_X,
                    Y = COMP(EXT, 1) - COMP_X,
                    Z = COMP(EXT, 2) - COMP_X,
                    W = COMP(EXT, 3) - COMP_X,
                    IMM = _MM_SHUFFLE(W, Z, Y, X)
                };

                META_COST({ COST = PERM_COST })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
#               if defined(__SSE2__)
                    return _mm_shuffle_epi32(p, IMM);
#               else
                    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(p), _mm_castsi128_ps(p), IMM));
#               endif
                }
            };

            template_spec(struct load, MOV)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
                    return p;
                }
            };

#   ifdef META_PEEPHOLE

#       if defined(__SSE2__) && !defined(__SSE4_1__)

            template_spec(struct load, SRLI1)
            {
                META_COST({ COST = SRLI_COST })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
                    return _mm_srli_si128(p, 4);
                }
            };

            template_spec(struct load, SRLI2)
            {
                META_COST({ COST = SRLI_COST })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
                    return _mm_srli_si128(p, 8);
                }
            };

            template_spec(struct load, SRLI3)
            {
                META_COST({ COST = SRLI_COST })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
                    return _mm_srli_si128(p, 12);
                }
            };

            template_spec(struct load, SLLI1)
            {
                META_COST({ COST = SLLI_COST })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
                    return _mm_slli_si128(p, 4);
                }
            };

            template_spec(struct load, SLLI2)
            {
                META_COST({ COST = SLLI_COST })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
                    return _mm_slli_si128(p, 8);
                }
            };

            template_spec(struct load, SLLI3)
            {
                META_COST({ COST = SLLI_COST })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
                    return _mm_slli_si128(p, 12);
                }
            };

#       endif

#       if !defined(__SSE2__)

            template_spec(struct load, UNPCKL)
            {
                META_COST({ COST = UNPCKL_COST })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
                    return _mm_unpacklo_epi32(p, p);
                }
            };

            template_spec(struct load, UNPCKH)
            {
                META_COST({ COST = UNPCKH_COST })

                static MATH_FORCEINLINE __m128i f(__m128i p)
                {
                    return _mm_unpackhi_epi32(p, p);
                }
            };

#       endif

#   endif

            META_COST({ COST = template_inst(load, OP) ::COST })

            static MATH_FORCEINLINE __m128i f(__m128i p)
            {
                return template_inst(load, OP) ::f(p);
            }
        };

        template<int LHS, int RHS, int SEL> struct MASK
        {
            enum
            {
                L_SCL = ISDUP(LHS),
                R_SCL = ISDUP(RHS),

                L_MSK = ~SEL & MASK(LHS),
                L_SWZ = (L_SCL ? SWZ_XYZW : LHS) & L_MSK,
                R_MSK = ~L_MSK & MASK(RHS),
                R_SWZ = (R_SCL ? SWZ_XYZW : RHS) & R_MSK,

                IGN = ~(L_MSK | R_MSK) & MSK_XYZW,

                L_POP = POP(L_SWZ),
                R_POP = POP(R_SWZ)
            };

            enum
            {
                SIMPLE =
                    (MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_Y, COMP_Z, COMP_W), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_N, COMP_N, COMP_N), R_MSK, IGN)) ||
                    (MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_Y, COMP_Z, COMP_W), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_N, COMP_N, COMP_N), L_MSK, IGN)),

                MERGE =
                    (L_SWZ == 0 && R_SWZ == 0) ? CLEAR :
                    L_SWZ == 0 ? MOV_R :
                    R_SWZ == 0 ? MOV :
#               ifdef META_PEEPHOLE
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_N, COMP_Y, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Y), R_MSK, IGN) ? UNPCKL :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_N, COMP_Y, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Y), L_MSK, IGN) ? UNPCKL_R :
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Z, COMP_N, COMP_W, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_Z, COMP_N, COMP_W), R_MSK, IGN) ? UNPCKH :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Z, COMP_N, COMP_W, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_Z, COMP_N, COMP_W), L_MSK, IGN) ? UNPCKH_R :
#               endif
                    (R_MSK & MSK_XY) == 0 && (L_MSK & MSK_ZW) == 0 ? SHUF :
                    (R_MSK & MSK_ZW) == 0 && (L_MSK & MSK_XY) == 0 ? SHUF_R :
#           ifdef META_PEEPHOLE
#               if defined(__SSE4_1__)
                    (R_POP == 1 && !SIMPLE) ? INSERT :
                    (L_POP == 1 && !SIMPLE) ? INSERT_R :
#               endif
#               if defined(__SSSE3__)
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_W, COMP_N, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_X, COMP_Y, COMP_Z), R_MSK, IGN) ? PALIGNR1 :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_W, COMP_N, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_X, COMP_Y, COMP_Z), L_MSK, IGN) ? PALIGNR1_R :
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_N, COMP_X), R_MSK, IGN) ? PALIGNR3 :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_N, COMP_X), L_MSK, IGN) ? PALIGNR3_R :
#               endif
#           endif
                    L_MSK < R_MSK ? BLEND : BLEND_R
            };

            enum
            {
                LLOAD =
                    L_SWZ == 0 ? MOV :
                    L_SCL ? MOV :
                    R_SWZ == 0 ?
                    L_SWZ == (SWZ_XYZW & L_MSK) ? MOV :
                    DEFAULT :
                    MERGE != int(BLEND) && MERGE != int(BLEND_R) ? MOV :
                    (L_SCL || L_SWZ == (SWZ_XYZW & L_MSK)) ? MOV :
                    DEFAULT
            };

            enum
            {
                RLOAD =
                    R_SWZ == 0 ? MOV :
                    R_SCL ? MOV :
                    L_SWZ == 0 ?
                    R_SWZ == (SWZ_XYZW & R_MSK) ? MOV :
                    DEFAULT :
                    MERGE != int(BLEND) && MERGE != int(BLEND_R) ? MOV :
                    (R_SCL || R_SWZ == (SWZ_XYZW & R_MSK)) ? MOV :
                    DEFAULT
            };

            template_decl(struct lload, unsigned op)
            {
                META_COST({ COST = SWIZ<L_SWZ>::COST })

                static MATH_FORCEINLINE __m128i f(__m128i a)
                {
                    return SWIZ<L_SWZ>::f(a);
                }
            };

            template_spec(struct lload, MOV)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128i f(__m128i a)
                {
                    return a;
                }
            };

            template_decl(struct rload, unsigned op)
            {
                META_COST({ COST = SWIZ<R_SWZ>::COST })

                static MATH_FORCEINLINE __m128i f(__m128i a)
                {
                    return SWIZ<R_SWZ>::f(a);
                }
            };

            template_spec(struct rload, MOV)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128i f(__m128i a)
                {
                    return a;
                }
            };

            template_decl(struct merge, unsigned op)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i)
                {
                    return a;
                }
            };

            template_spec(struct merge, CLEAR)
            {
                META_COST({ COST = XOR_COST })

                static MATH_FORCEINLINE __m128i f(__m128i, __m128i)
                {
                    return _mm_setzero_ps();
                }
            };

            template_spec(struct merge, MOV)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i)
                {
                    return a;
                }
            };

            template_spec(struct merge, MOV_R)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128i f(__m128i, __m128i b)
                {
                    return b;
                }
            };

#   ifdef META_PEEPHOLE

#       ifdef __SSE4_1__

            template_spec(struct merge, INSERT)
            {
                META_COST({ COST = INSERT_COST })

                enum
                {
                    COMP = COMP(R_SWZ, FFS(R_SWZ)) - COMP_X,
                    COUNT_S = COMP,
                    COUNT_D = COMP(R_MSK, 0) ? 0 : (COMP(R_MSK, 1) ? 1 : (COMP(R_MSK, 2) ? 2 : 3)),
                    ZMASK = 0,
                    NDX = ZMASK | (COUNT_D << 4) | (COUNT_S << 6)
                };

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_insert_epi32(a, _mm_extract_epi32(b, COUNT_S), COUNT_D);
                }
            };

            template_spec(struct merge, INSERT_R)
            {
                META_COST({ COST = INSERT_COST })

                enum
                {
                    COMP = COMP(L_SWZ, FFS(L_SWZ)) - COMP_X,
                    COUNT_S = COMP,
                    COUNT_D = COMP(L_MSK, 0) ? 0 : (COMP(L_MSK, 1) ? 1 : (COMP(L_MSK, 2) ? 2 : 3)),
                    ZMASK = 0,
                    NDX = ZMASK | (COUNT_D << 4) | (COUNT_S << 6)
                };

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_insert_epi32(b, _mm_extract_epi32(a, COUNT_S), COUNT_D);
                }
            };

#       endif

#   endif

            template_spec(struct merge, UNPCKL)
            {
                META_COST({ COST = UNPCKL_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_unpacklo_epi32(a, b);
                }
            };

            template_spec(struct merge, UNPCKL_R)
            {
                META_COST({ COST = UNPCKL_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_unpacklo_epi32(b, a);
                }
            };

            template_spec(struct merge, UNPCKH)
            {
                META_COST({ COST = UNPCKH_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_unpackhi_epi32(a, b);
                }
            };

            template_spec(struct merge, UNPCKH_R)
            {
                META_COST({ COST = UNPCKH_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_unpackhi_epi32(b, a);
                }
            };

            template_spec(struct merge, SHUF)
            {
                enum
                {
                    L_DEF = COMP_X + FFS(L_MSK),
                    L_EXT = (L_MSK & L_SWZ) | (SWZ(L_DEF, L_DEF, L_DEF, L_DEF) & ~L_MSK),
                    R_DEF = COMP_X + FFS(R_MSK),
                    R_EXT = (R_MSK & R_SWZ) | (SWZ(R_DEF, R_DEF, R_DEF, R_DEF) & ~R_MSK),
                    X = COMP(L_EXT, 0) == COMP_A ? 0 : (COMP(L_EXT, 0) - COMP_X),
                    Y = COMP(L_EXT, 1) == COMP_A ? 1 : (COMP(L_EXT, 1) - COMP_X),
                    Z = COMP(R_EXT, 2) == COMP_A ? 2 : (COMP(R_EXT, 2) - COMP_X),
                    W = COMP(R_EXT, 3) == COMP_A ? 3 : (COMP(R_EXT, 3) - COMP_X),
                    IMM = _MM_SHUFFLE(W, Z, Y, X)
                };

                META_COST({ COST = SHUF_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), IMM));
                }
            };

            template_spec(struct merge, SHUF_R)
            {
                enum
                {
                    L_DEF = COMP_X + FFS(L_MSK),
                    L_EXT = (L_MSK & L_SWZ) | (SWZ(L_DEF, L_DEF, L_DEF, L_DEF) & ~L_MSK),
                    R_DEF = COMP_X + FFS(R_MSK),
                    R_EXT = (R_MSK & R_SWZ) | (SWZ(R_DEF, R_DEF, R_DEF, R_DEF) & ~R_MSK),
                    X = COMP(R_EXT, 0) == COMP_A ? 0 : (COMP(R_EXT, 0) - COMP_X),
                    Y = COMP(R_EXT, 1) == COMP_A ? 1 : (COMP(R_EXT, 1) - COMP_X),
                    Z = COMP(L_EXT, 2) == COMP_A ? 2 : (COMP(L_EXT, 2) - COMP_X),
                    W = COMP(L_EXT, 3) == COMP_A ? 3 : (COMP(L_EXT, 3) - COMP_X),
                    IMM = _MM_SHUFFLE(W, Z, Y, X)
                };

                META_COST({ COST = SHUF_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(b), _mm_castsi128_ps(a), IMM));
                }
            };

#       if defined(__SSSE3__)

            template_spec(struct merge, PALIGNR1)
            {
                META_COST({ COST = PALIGNR_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_alignr_epi8(a, b, 4);
                }
            };

            template_spec(struct merge, PALIGNR1_R)
            {
                META_COST({ COST = PALIGNR_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_alignr_epi8(b, a, 4);
                }
            };

            template_spec(struct merge, PALIGNR3)
            {
                META_COST({ COST = PALIGNR_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_alignr_epi8(a, b, 12);
                }
            };

            template_spec(struct merge, PALIGNR3_R)
            {
                META_COST({ COST = PALIGNR_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_alignr_epi8(b, a, 12);
                }
            };

#       endif

            template_spec(struct merge, BLEND)
            {
#           ifdef __SSE4_1__

                enum
                {
                    MASK = (COMP(R_MSK, 0) ? 1 : 0) | (COMP(R_MSK, 1) ? 2 : 0) | (COMP(R_MSK, 2) ? 4 : 0) | (COMP(R_MSK, 3) ? 8 : 0)
                };

                META_COST({ COST = BLEND_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_castps_si128(_mm_blend_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), MASK));
                }

#           else

                enum
                {
                    X = COMP(R_MSK, 0) != 0,
                    Y = COMP(R_MSK, 1) != 0,
                    Z = COMP(R_MSK, 2) != 0,
                    W = COMP(R_MSK, 3) != 0
                };

                META_COST({ COST = AND_COST + ANDN_COST + OR_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    const __m128i c = _mm_setr_epi32(-X, -Y, -Z, -W);
                    return _mm_or_si128(_mm_and_si128(b, c), _mm_andnot_si128(c, a));
                }

#           endif
            };

            template_spec(struct merge, BLEND_R)
            {
#           ifdef __SSE4_1__

                enum
                {
                    MASK = (COMP(L_MSK, 0) ? 1 : 0) | (COMP(L_MSK, 1) ? 2 : 0) | (COMP(L_MSK, 2) ? 4 : 0) | (COMP(L_MSK, 3) ? 8 : 0)
                };

                META_COST({ COST = BLEND_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    return _mm_castps_si128(_mm_blend_ps(_mm_castsi128_ps(b), _mm_castsi128_ps(a), MASK));
                }

#           else

                enum
                {
                    X = COMP(L_MSK, 0) != 0,
                    Y = COMP(L_MSK, 1) != 0,
                    Z = COMP(L_MSK, 2) != 0,
                    W = COMP(L_MSK, 3) != 0
                };

                META_COST({ COST = AND_COST + ANDN_COST + OR_COST })

                static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
                {
                    const __m128i c = _mm_setr_epi32(-X, -Y, -Z, -W);
                    return _mm_or_si128(_mm_and_si128(a, c), _mm_andnot_si128(c, b));
                }

#           endif
            };

            META_COST({ COST = template_inst(lload, LLOAD) ::COST + template_inst(rload, RLOAD) ::COST + template_inst(merge, MERGE) ::COST })

            static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b)
            {
                __m128i l = template_inst(lload, LLOAD) ::f(a);
                __m128i r = template_inst(rload, RLOAD) ::f(b);
                __m128i d = template_inst(merge, MERGE) ::f(l, r);
                return d;
            }
        };

#   if defined(__SSE4_1__)

        template<int c> struct GET
        {
            static MATH_FORCEINLINE int f(__m128i p)
            {
                return _mm_extract_epi32(p, FFS(USED(c)));
            }
        };

        template<int c> struct SET
        {
            static MATH_FORCEINLINE __m128i f(__m128i p, int i)
            {
                return _mm_insert_epi32(p, i, FFS(USED(c)));
            }
        };

#   else

        template<int c> struct GET
        {
            template_decl(struct impl, int u)
            {
                static MATH_FORCEINLINE int g(__m128i p)
                {
                    return _mm_cvtsi128_si32(SWIZ<c>::f(p));
                }
            };

            template_spec(struct impl, 0)
            {
                static MATH_FORCEINLINE int g(__m128i p)
                {
                    return _mm_cvtsi128_si32(p);
                }
            };

            static MATH_FORCEINLINE int f(__m128i p)
            {
                return template_inst(impl, FFS(USED(c))) ::g(p);
            }
        };

        template<int c> struct SET
        {
            static MATH_FORCEINLINE __m128i f(__m128i p, int i)
            {
                return MASK<SWZ_XYZW, Inverse<COMP_X, c>::SWZ, USED(c)>::f(p, _mm_cvtsi32_si128(i));
            }
        };

#   endif

        template<int X, int Y, int Z, int W> struct GATHER
        {
            enum
            {
                IGN = (ISDUP(X) ? MSK_X : 0) | (ISDUP(Y) ? MSK_Y : 0) | (ISDUP(Z) ? MSK_Z : 0) | (ISDUP(W) ? MSK_W : 0),
                SWZ = SWZ(COMP(X, 0), COMP(Y, 0), COMP(Z, 0), COMP(W, 0)),
                MSK = MASK(SWZ),
                SEL =
                    (MSK & MSK_ZW) == 0 ? SWZ_XY :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_X, COMP_Y, COMP_Z, COMP_W), MSK, IGN) ? SWZ_XYZW :
#           ifdef META_PEEPHOLE
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_X, COMP_X, COMP_X, COMP_X), MSK, IGN) ? SWZ_XXXX :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_Y), MSK, IGN) ? SWZ_YYYY :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_Z), MSK, IGN) ? SWZ_ZZZZ :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_W, COMP_W, COMP_W, COMP_W), MSK, IGN) ? SWZ_WWWW :
#           endif
                    0
            };

            template_decl(struct impl, int swz)
            {
                static MATH_FORCEINLINE __m128i g(__m128i a, __m128i b, __m128i c, __m128i_arg d)
                {
                    return MASK<SWZ_XYZW, SWZ_ZWXY, MSK_ZW>::f(MASK<X & MSK_X, (Y << 4) & MSK_Y, MSK_Y>::f(a, b), MASK<Z & MSK_X, (W << 4) & MSK_Y, MSK_Y>::f(c, d));
                }
            };

            template_spec(struct impl, SWZ_XY)
            {
                static MATH_FORCEINLINE __m128i g(__m128i a, __m128i b, __m128i, __m128i_arg)
                {
                    return MASK<X & MSK_X, (Y << 4) & MSK_Y, MSK_Y>::f(a, b);
                }
            };

            template_spec(struct impl, SWZ_XYZW)
            {
                static MATH_FORCEINLINE __m128i g(__m128i a, __m128i b, __m128i c, __m128i_arg d)
                {
                    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(_mm_unpacklo_epi32(a, b)), _mm_castsi128_ps(_mm_unpackhi_epi32(c, d)), _MM_SHUFFLE(3, 0, 3, 0)));
                }
            };

#       ifdef META_PEEPHOLE

            template_spec(struct impl, SWZ_XXXX)
            {
                static MATH_FORCEINLINE __m128i g(__m128i a, __m128i b, __m128i c, __m128i_arg d)
                {
                    return _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_unpacklo_epi32(a, b)), _mm_castsi128_ps(_mm_unpacklo_epi32(c, d))));
                }
            };

            template_spec(struct impl, SWZ_YYYY)
            {
                static MATH_FORCEINLINE __m128i g(__m128i a, __m128i b, __m128i c, __m128i_arg d)
                {
                    return _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm_unpacklo_epi32(c, d)), _mm_castsi128_ps(_mm_unpacklo_epi32(a, b))));
                }
            };

            template_spec(struct impl, SWZ_ZZZZ)
            {
                static MATH_FORCEINLINE __m128i g(__m128i a, __m128i b, __m128i c, __m128i_arg d)
                {
                    return _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(_mm_unpackhi_epi32(a, b)), _mm_castsi128_ps(_mm_unpackhi_epi32(c, d))));
                }
            };

            template_spec(struct impl, SWZ_WWWW)
            {
                static MATH_FORCEINLINE __m128i g(__m128i a, __m128i b, __m128i c, __m128i_arg d)
                {
                    return _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(_mm_unpackhi_epi32(c, d)), _mm_castsi128_ps(_mm_unpackhi_epi32(a, b))));
                }
            };

#       endif

            static MATH_FORCEINLINE __m128i f(__m128i a, __m128i b, __m128i c, __m128i_arg d)
            {
                return template_inst(impl, SEL) ::g(a, b, c, d);
            }
        };

        template<int SWZ> struct ANY
        {
            enum
            {
                USE = USED(SWZ),
                MSK = (COMP(USE, 0) ? 1 : 0) | (COMP(USE, 1) ? 2 : 0) | (COMP(USE, 2) ? 4 : 0) | (COMP(USE, 3) ? 8 : 0)
            };

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE int g(__m128i p)
                {
                    return (_mm_movemask_ps(_mm_castsi128_ps(p)) & MSK) != 0;
                }
            };

            template_spec(struct impl, MSK_XYZW)
            {
                static MATH_FORCEINLINE int g(__m128i p)
                {
                    return _mm_movemask_ps(_mm_castsi128_ps(p)) != 0;
                }
            };

            static MATH_FORCEINLINE int f(__m128i p)
            {
                return template_inst(impl, USE) ::g(p);
            }
        };

        template<int SWZ> struct ALL
        {
            enum
            {
                USE = USED(SWZ),
                MSK = (COMP(USE, 0) ? 1 : 0) | (COMP(USE, 1) ? 2 : 0) | (COMP(USE, 2) ? 4 : 0) | (COMP(USE, 3) ? 8 : 0)
            };

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE int g(__m128i p)
                {
                    return (_mm_movemask_ps(_mm_castsi128_ps(p)) & MSK) == MSK;
                }
            };

            template_spec(struct impl, MSK_XYZW)
            {
                static MATH_FORCEINLINE int g(__m128i p)
                {
                    return _mm_movemask_ps(_mm_castsi128_ps(p)) == MSK;
                }
            };

            static MATH_FORCEINLINE int f(__m128i p)
            {
                return template_inst(impl, USE) ::g(p);
            }
        };

#   if defined(__SSE2__)

        static MATH_FORCEINLINE __m128i BOOL(__m128i a)
        {
            return _mm_srli_epi32(a, 31);
        }

        template<int RHS = SWZ_XYZW> struct NEG
        {
            enum
            {
                R = UnOp<v4i, RHS>::R,
                S = UnOp<v4i, RHS>::S
            };

            META_COST({ COST = (UnOp<v4i, RHS>::COST) + SUB_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i rhs)
            {
                return _mm_sub_epi32(_mm_setzero_si128(), SWIZ<R>::f(rhs));
            }
        };

        template<int RHS = SWZ_XYZW> struct ABS
        {
            enum
            {
                R = UnOp<v4i, RHS>::R,
                S = UnOp<v4i, RHS>::S
            };
            typedef v4i     type;

#       if defined(__SSSE3__)
            META_COST({ COST = (UnOp<v4i, RHS>::COST) + ABS_COST })
#       else
            META_COST({ COST = (UnOp<v4i, RHS>::COST) + SRAI_COST + XOR_COST + SUB_COST })
#       endif

            static MATH_FORCEINLINE __m128i f(__m128i rhs)
            {
#       if defined(__SSSE3__)
                return _mm_abs_epi32(SWIZ<R>::f(rhs));
#       else
                __m128i r = SWIZ<R>::f(rhs), c = _mm_srai_epi32(r, 31);
                return _mm_sub_epi32(_mm_xor_si128(r, c), c);
#       endif
            }
        };

        template<int RHS = SWZ_XYZW> struct NOT
        {
            enum
            {
                R = UnOp<v4i, RHS>::R,
                S = UnOp<v4i, RHS>::S
            };

            META_COST({ COST = (UnOp<v4i, RHS>::COST) + 1 })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i rhs)
            {
#           if  defined(__SSE2__)
                return _mm_xor_si128(SWIZ<R>::f(rhs), _mm_setr_epi32(~0, ~0, ~0, ~0));
#           else
                return _mm_castps_si128(_mm_xor_ps(_mm_castsi128_ps(SWIZ<R>::f(rhs)), _mm_castsi128_ps(_mm_setr_epi32(~0, ~0, ~0, ~0))));
#           endif
            }
        };

        template<int LHS = SWZ_XYZW> struct SLLI
        {
            enum
            {
                L = UnOp<v4i, LHS>::R,
                S = UnOp<v4i, LHS>::S
            };

            META_COST({ COST = (UnOp<v4i, LHS>::COST) + SLLI_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, int i)
            {
                return _mm_slli_epi32(SWIZ<L>::f(lhs), i);
            }
        };

        template<int LHS = SWZ_XYZW> struct SRLI
        {
            enum
            {
                L = UnOp<v4i, LHS>::R,
                S = UnOp<v4i, LHS>::S
            };

            META_COST({ COST = (UnOp<v4i, LHS>::COST) + SRLI_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, int i)
            {
                return _mm_srli_epi32(SWIZ<L>::f(lhs), i);
            }
        };

        template<int LHS = SWZ_XYZW> struct SRAI
        {
            enum
            {
                L = UnOp<v4i, LHS>::R,
                S = UnOp<v4i, LHS>::S
            };

            META_COST({ COST = (UnOp<v4i, LHS>::COST) + SRAI_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, int i)
            {
                return _mm_srai_epi32(SWIZ<L>::f(lhs), i);
            }
        };

#   else

        static MATH_FORCEINLINE __m128i BOOL(__m128i a)
        {
            return _mm_castps_si128(_mm_and_ps(a, _mm_castsi128_ps(_mm_setr_epi32(1, 1, 1, 1))));
        }

        template<int RHS = SWZ_XYZW> struct NEG
        {
            enum
            {
                R = UnOp<v4i, RHS>::R,
                S = UnOp<v4i, RHS>::S
                    N = Swizzle<R>::N,
                R0 = Swizzle<R>::C0,
                R1 = Swizzle<R>::C1,
                R2 = Swizzle<R>::C2,
                R3 = Swizzle<R>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i rhs)
                {
                    return type::CTOR(
                        -((const int*)&rhs)[R0],
                        -((const int*)&rhs)[R1],
                        -((const int*)&rhs)[R2],
                        -((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i rhs)
                {
                    return type::CTOR(
                        -((const int*)&rhs)[R0],
                        -((const int*)&rhs)[R1],
                        -((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i rhs)
                {
                    return type::CTOR(
                        -((const int*)&rhs)[R0],
                        -((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i rhs)
                {
                    return type::CTOR(
                        -((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i rhs)
            {
                return template_inst(impl, SEL) ::g(rhs);
            }
        };

        template<int RHS = SWZ_XYZW> struct ABS
        {
            enum
            {
                R = UnOp<v4i, RHS>::R,
                S = UnOp<v4i, RHS>::S
                    N = Swizzle<R>::N,
                R0 = Swizzle<R>::C0,
                R1 = Swizzle<R>::C1,
                R2 = Swizzle<R>::C2,
                R3 = Swizzle<R>::C3,
                SEL = N
            };

            META_COST({ COST = 4 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i rhs)
                {
                    return type::CTOR(
                        abs(((const int*)&rhs)[R0]),
                        abs(((const int*)&rhs)[R1]),
                        abs(((const int*)&rhs)[R2]),
                        abs(((const int*)&rhs)[R3])
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i rhs)
                {
                    return type::CTOR(
                        abs(((const int*)&rhs)[R0]),
                        abs(((const int*)&rhs)[R1]),
                        abs(((const int*)&rhs)[R2])
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i rhs)
                {
                    return type::CTOR(
                        abs(((const int*)&rhs)[R0]),
                        abs(((const int*)&rhs)[R1])
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i rhs)
                {
                    return type::CTOR(
                        abs(((const int*)&rhs)[R0])
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i rhs)
            {
                return template_inst(impl, SEL) ::g(rhs);
            }
        };

        template<int LHS = SWZ_XYZW> struct SLLI
        {
            enum
            {
                R = UnOp<v4i, RHS>::R,
                S = UnOp<v4i, RHS>::S
                    N = Swizzle<R>::N,
                R0 = Swizzle<R>::C0,
                R1 = Swizzle<R>::C1,
                R2 = Swizzle<R>::C2,
                R3 = Swizzle<R>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] << i,
                        ((const int*)&lhs)[L1] << i,
                        ((const int*)&lhs)[L2] << i,
                        ((const int*)&lhs)[L3] << i
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] << i,
                        ((const int*)&lhs)[L1] << i,
                        ((const int*)&lhs)[L2] << i
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] << i,
                        ((const int*)&lhs)[L1] << i
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] << i
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, unsigned i)
            {
                return template_inst(impl, SEL) ::g(lhs, i);
            }
        };

        template<int LHS = SWZ_XYZW> struct SRLI
        {
            enum
            {
                R = UnOp<v4i, RHS>::R,
                S = UnOp<v4i, RHS>::S
                    N = Swizzle<R>::N,
                R0 = Swizzle<R>::C0,
                R1 = Swizzle<R>::C1,
                R2 = Swizzle<R>::C2,
                R3 = Swizzle<R>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        (int)(((const unsigned int*)&lhs)[L0] >> i),
                        (int)(((const unsigned int*)&lhs)[L1] >> i),
                        (int)(((const unsigned int*)&lhs)[L2] >> i),
                        (int)(((const unsigned int*)&lhs)[L3] >> i)
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        (int)(((const unsigned int*)&lhs)[L0] >> i),
                        (int)(((const unsigned int*)&lhs)[L1] >> i),
                        (int)(((const unsigned int*)&lhs)[L2] >> i)
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        (int)(((const unsigned int*)&lhs)[L0] >> i),
                        (int)(((const unsigned int*)&lhs)[L1] >> i)
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        (int)(((const int*)&lhs)[L0] >> i)
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, unsigned i)
            {
                return template_inst(impl, SEL) ::g(lhs, i);
            }
        };

        template<int LHS = SWZ_XYZW> struct SRAI
        {
            enum
            {
                R = UnOp<v4i, RHS>::R,
                S = UnOp<v4i, RHS>::S
                    N = Swizzle<R>::N,
                R0 = Swizzle<R>::C0,
                R1 = Swizzle<R>::C1,
                R2 = Swizzle<R>::C2,
                R3 = Swizzle<R>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >> i,
                        ((const int*)&lhs)[L1] >> i,
                        ((const int*)&lhs)[L2] >> i,
                        ((const int*)&lhs)[L3] >> i
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >> i,
                        ((const int*)&lhs)[L1] >> i,
                        ((const int*)&lhs)[L2] >> i
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >> i,
                        ((const int*)&lhs)[L1] >> i
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, unsigned i)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >> i
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, unsigned i)
            {
                return template_inst(impl, SEL) ::g(lhs, i);
            }
        };

#   endif

#   if defined(__SSE2__)

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct ADD
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + ADD_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_add_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SUB
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + SUB_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_sub_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

#   else

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct ADD
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] + ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] + ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] + ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] + ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] + ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] + ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] + ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] + ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] + ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] + ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SUB
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] - ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] - ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] - ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] - ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] - ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] - ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] - ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] - ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] - ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] - ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

#   endif

#   if defined(__SSE4_1__)

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MUL
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + MUL_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
            {
                return _mm_mullo_epi32(lhs, rhs);
            }

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

#   else

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MUL
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 4 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] * ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] * ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] * ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] * ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] * ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] * ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] * ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] * ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] * ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] * ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

#   endif

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct DIV
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 8 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] / ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] / ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] / ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] / ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] / ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] / ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] / ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] / ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] / ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] / ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct AND
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + AND_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
#           if defined(__SSE2__)
                return _mm_and_si128(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
#           else
                return _mm_castps_si128(_mm_and_ps(_mm_castsi128_ps(SWIZ<L>::f(lhs)), _mm_castsi128_ps(SWIZ<R>::f(rhs))));
#           endif
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct ANDN
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + ANDN_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
#           if defined(__SSE2__)
                return _mm_andnot_si128(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
#           else
                return _mm_castps_si128(_mm_andnot_ps(_mm_castsi128_ps(SWIZ<L>::f(lhs)), _mm_castsi128_ps(SWIZ<R>::f(rhs))));
#           endif
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct OR
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + OR_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
#           if defined(__SSE2__)
                return _mm_or_si128(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
#           else
                return _mm_castps_si128(_mm_or_ps(_mm_castsi128_ps(SWIZ<L>::f(lhs)), _mm_castsi128_ps(SWIZ<R>::f(rhs))));
#           endif
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct XOR
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + XOR_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
#           if defined(__SSE2__)
                return _mm_xor_si128(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
#           else
                return _mm_castps_si128(_mm_xor_ps(_mm_castsi128_ps(SWIZ<L>::f(lhs)), _mm_castsi128_ps(SWIZ<R>::f(rhs))));
#           endif
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SLL
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] << ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] << ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] << ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] << ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] << ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] << ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] << ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] << ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] << ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] << ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SLR
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        int(((const unsigned int*)&lhs)[L0] >> ((const int*)&rhs)[R0]),
                        int(((const unsigned int*)&lhs)[L1] >> ((const int*)&rhs)[R1]),
                        int(((const unsigned int*)&lhs)[L2] >> ((const int*)&rhs)[R2]),
                        int(((const unsigned int*)&lhs)[L3] >> ((const int*)&rhs)[R3])
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        int(((const unsigned int*)&lhs)[L0] >> ((const int*)&rhs)[R0]),
                        int(((const unsigned int*)&lhs)[L1] >> ((const int*)&rhs)[R1]),
                        int(((const unsigned int*)&lhs)[L2] >> ((const int*)&rhs)[R2])
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        int(((const unsigned int*)&lhs)[L0] >> ((const int*)&rhs)[R0]),
                        int(((const unsigned int*)&lhs)[L1] >> ((const int*)&rhs)[R1])
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        int(((const unsigned int*)&lhs)[L0] >> ((const int*)&rhs)[R0])
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SAR
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >> ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] >> ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] >> ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] >> ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >> ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] >> ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] >> ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >> ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] >> ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >> ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

#   if defined(__SSE2__)

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPEQ
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + CMP_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_cmpeq_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPNE
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + CMP_COST + XOR_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_xor_si128(_mm_cmpeq_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)), (__m128i)_mm_setr_epi32(~0, ~0, ~0, ~0));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPGT
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + CMP_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_cmpgt_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPGE
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + CMP_COST + XOR_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_xor_si128(_mm_cmplt_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)), (__m128i)_mm_setr_epi32(~0, ~0, ~0, ~0));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPLT
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + CMP_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_cmplt_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPLE
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + CMP_COST + XOR_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_xor_si128(_mm_cmpgt_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)), (__m128i)_mm_setr_epi32(~0, ~0, ~0, ~0));
            }
        };

#   else

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPEQ
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] == ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] == ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] == ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] == ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] == ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] == ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] == ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] == ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] == ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] == ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPNE
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] != ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] != ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] != ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] != ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] != ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] != ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] != ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] != ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] != ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] != ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPGT
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] > ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] > ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] > ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] > ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] > ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] > ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] > ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] > ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] > ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] > ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPGE
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >= ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] >= ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] >= ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] >= ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >= ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] >= ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] >= ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >= ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] >= ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] >= ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPLT
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] < ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] < ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] < ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] < ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] < ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] < ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] < ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] < ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] < ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] < ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPLE
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 2 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] <= ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] <= ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] <= ((const int*)&rhs)[R2],
                        ((const int*)&lhs)[L3] <= ((const int*)&rhs)[R3]
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] <= ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] <= ((const int*)&rhs)[R1],
                        ((const int*)&lhs)[L2] <= ((const int*)&rhs)[R2]
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] <= ((const int*)&rhs)[R0],
                        ((const int*)&lhs)[L1] <= ((const int*)&rhs)[R1]
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        ((const int*)&lhs)[L0] <= ((const int*)&rhs)[R0]
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

#   endif

#   if defined(__SSE4_1__)

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MIN
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + MIN_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_min_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MAX
        {
            enum
            {
                L = BinOp<v4i, LHS, RHS>::L,
                R = BinOp<v4i, LHS, RHS>::R,
                S = BinOp<v4i, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + MAX_COST })

            typedef v4i     type;

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return _mm_max_epi32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

#   else

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MIN
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 4 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        min(((const int*)&lhs)[L0], ((const int*)&rhs)[R0]),
                        min(((const int*)&lhs)[L1], ((const int*)&rhs)[R1]),
                        min(((const int*)&lhs)[L2], ((const int*)&rhs)[R2]),
                        min(((const int*)&lhs)[L3], ((const int*)&rhs)[R3])
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        min(((const int*)&lhs)[L0], ((const int*)&rhs)[R0]),
                        min(((const int*)&lhs)[L1], ((const int*)&rhs)[R1]),
                        min(((const int*)&lhs)[L2], ((const int*)&rhs)[R2])
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        min(((const int*)&lhs)[L0], ((const int*)&rhs)[R0]),
                        min(((const int*)&lhs)[L1], ((const int*)&rhs)[R1])
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        min(((const int*)&lhs)[L0], ((const int*)&rhs)[R0])
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MAX
        {
            enum
            {
                N = Swizzle<LHS>::N,
                L = LHS,
                R = RHS,
                S = N == 4 ? SWZ_XYZW : N == 3 ? SWZ_XYZ : N == 2 ? SWZ_XY : SWZ_X,
                L0 = Swizzle<LHS>::C0,
                L1 = Swizzle<LHS>::C1,
                L2 = Swizzle<LHS>::C2,
                L3 = Swizzle<LHS>::C3,
                R0 = Swizzle<RHS>::C0,
                R1 = Swizzle<RHS>::C1,
                R2 = Swizzle<RHS>::C2,
                R3 = Swizzle<RHS>::C3,
                SEL = N
            };

            META_COST({ COST = 4 * N })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        max(((const int*)&lhs)[L0], ((const int*)&rhs)[R0]),
                        max(((const int*)&lhs)[L1], ((const int*)&rhs)[R1]),
                        max(((const int*)&lhs)[L2], ((const int*)&rhs)[R2]),
                        max(((const int*)&lhs)[L3], ((const int*)&rhs)[R3])
                    );
                }
            };

            template_spec(struct impl, 3)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        max(((const int*)&lhs)[L0], ((const int*)&rhs)[R0]),
                        max(((const int*)&lhs)[L1], ((const int*)&rhs)[R1]),
                        max(((const int*)&lhs)[L2], ((const int*)&rhs)[R2])
                    );
                }
            };

            template_spec(struct impl, 2)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        max(((const int*)&lhs)[L0], ((const int*)&rhs)[R0]),
                        max(((const int*)&lhs)[L1], ((const int*)&rhs)[R1])
                    );
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128i lhs, __m128i rhs)
                {
                    return type::CTOR(
                        max(((const int*)&lhs)[L0], ((const int*)&rhs)[R0])
                    );
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128i lhs, __m128i rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, rhs);
            }
        };

#   endif
    };

    struct v4f
    {
        typedef __m128  packed;
        typedef float   type;

#   if defined(_MSC_VER)
        // patch for Visual C++
        typedef const __m128&   __m128_arg;
#   else
        typedef __m128          __m128_arg;
#   endif

        enum
        {
            NOP = 0,
            MOV,
            MOV_R,
            MOVS,
            MOVS_R,
            MOVHL,
            MOVHL_R,
            MOVLH,
            MOVLH_R,
            UNPCKL,
            UNPCKL_R,
            UNPCKH,
            UNPCKH_R,
            SHUF,
            SHUF_R,
            BLEND,
            BLEND_R,
            CLEAR,
#       if defined(__SSSE3__)
            PALIGNR1,
            PALIGNR1_R,
            PALIGNR3,
            PALIGNR3_R,
#       endif
#       if defined(__SSE3__)
            MOVLDUP,
            MOVLDUP_R,
            MOVHDUP,
            MOVHDUP_R,
#       endif
#       if defined(__SSE4_1__)
            EXTRACT,
            INSERT,
            INSERT_R,
#       endif
#       if defined(__AVX2__)
            BROADCAST,
#       endif
            DEFAULT
        };

        enum
        {
#       if defined(__AVX2__)
            BROADCAST_COST = 1,
#       endif
#       if defined(__SSE4_1__)
            EXTRACT_COST = 2,
            INSERT_COST = 3,
            BLEND_COST = 3,
            RND_COST = 1,
#       endif
#       if defined(__SSE3__)
            MOVLDUP_COST = 1,
            MOVHDUP_COST = 1,
#       endif
#       if defined(__AVX__)
            PERM_COST = 1,
#       elif defined(__SSE2__)
            PERM_COST = 2,
#       else
            PERM_COST = 3,
#       endif
#       if defined(__SSSE3__)
            PALIGNR_COST = 2,
#       endif
            SHUF_COST = 3,
            MOVS_COST = 1,
            MOVHL_COST = 2,
            MOVLH_COST = 2,
            UNPCKL_COST = 2,
            UNPCKH_COST = 2,
            ADD_COST = 2,
            SUB_COST = 2,
            MUL_COST = 2,
            MIN_COST = 2,
            MAX_COST = 2,
            CMP_COST = 2,
            XOR_COST = 2,
            OR_COST = 2,
            AND_COST = 2,
            ANDN_COST = 2,
            CVT_COST = 3,
#       if defined(__FMA__)
            MADD_COST = 3,
#       endif
            DIV_COST = 10
        };

        static MATH_FORCEINLINE __m128 ZERO()
        {
            return _mm_setzero_ps();
        }

#   if defined(__GNUC__)

        static MATH_FORCEINLINE __m128 CTOR(float x)
        {
            return (__m128) {x, x, x, x };
        }

        static MATH_FORCEINLINE __m128 CTOR(float x, float y)
        {
            return (__m128) {x, y, x, y };
        }

        static MATH_FORCEINLINE __m128 CTOR(float x, float y, float z)
        {
            return (__m128) {x, y, z, 0.f };
        }

        static MATH_FORCEINLINE __m128 CTOR(float x, float y, float z, float w)
        {
            return (__m128) {x, y, z, w };
        }

#   else

        static MATH_FORCEINLINE __m128 CTOR(float x)
        {
            return _mm_set1_ps(x);
        }

        static MATH_FORCEINLINE __m128 CTOR(float x, float y)
        {
            return _mm_set_ps(y, x, y, x);
        }

        static MATH_FORCEINLINE __m128 CTOR(float x, float y, float z)
        {
            return _mm_set_ps(0.f, z, y, x);
        }

        static MATH_FORCEINLINE __m128 CTOR(float x, float y, float z, float w)
        {
            return _mm_set_ps(w, z, y, x);
        }

#   endif

        template<unsigned SWZ> struct SWIZ
        {
            enum
            {
                IGN = IGNORED(SWZ),
                MSK = MASK(SWZ & ~IGN),

                OP =
                    ISDUP(SWZ) ? MOV :
                    MATCH_SWIZ(0, SWZ, SWZ_XYZW, MSK, IGN) ? MOV :
#           ifdef META_PEEPHOLE
#               ifdef __AVX2__
                    MATCH_SWIZ(0, SWZ, SWZ_XXXX, MSK, IGN) ? BROADCAST :
#               endif
#               ifdef __SSE3__
                    MATCH_SWIZ(0, SWZ, SWZ_XXZZ, MSK, IGN) ? MOVLDUP :
                    MATCH_SWIZ(0, SWZ, SWZ_YYWW, MSK, IGN) ? MOVHDUP :
#               endif
                    MATCH_SWIZ(0, SWZ, SWZ_XYXY, MSK, IGN) ? MOVLH :
                    MATCH_SWIZ(0, SWZ, SWZ_ZWZW, MSK, IGN) ? MOVHL :

                    MATCH_SWIZ(0, SWZ, SWZ_XXYY, MSK, IGN) ? UNPCKL :
                    MATCH_SWIZ(0, SWZ, SWZ_ZZWW, MSK, IGN) ? UNPCKH :
#           endif
                    DEFAULT
            };

            template_decl(struct load, int op)
            {
                enum
                {
                    DEF = COMP(SWZ, FFS(SWZ)),
                    EXT = SWZ | (SWZ(DEF, DEF, DEF, DEF) & ~MSK),
                    X = COMP(EXT, 0) - COMP_X,
                    Y = COMP(EXT, 1) - COMP_X,
                    Z = COMP(EXT, 2) - COMP_X,
                    W = COMP(EXT, 3) - COMP_X,
                    IMM = _MM_SHUFFLE(W, Z, Y, X)
                };

                META_COST({ COST = PERM_COST })

                static MATH_FORCEINLINE __m128 f(__m128 p)
                {
#           if defined(__AVX__)
                    return _mm_permute_ps(p, IMM);
#           elif defined(__SSE2__)
                    return _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(p), IMM));
#           else
                    return _mm_shuffle_ps(p, p, IMM);
#           endif
                }
            };

            template_spec(struct load, MOV)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128 f(__m128 p)
                {
                    return p;
                }
            };

#   ifdef META_PEEPHOLE

            template_spec(struct load, MOVLH)
            {
                META_COST({ COST = MOVLH_COST })

                static MATH_FORCEINLINE __m128 f(__m128 p)
                {
                    return _mm_movelh_ps(p, p);
                }
            };

            template_spec(struct load, MOVHL)
            {
                META_COST({ COST = MOVHL_COST })

                static MATH_FORCEINLINE __m128 f(__m128 p)
                {
                    return _mm_movehl_ps(p, p);
                }
            };

            template_spec(struct load, UNPCKL)
            {
                META_COST({ COST = UNPCKL_COST })

                static MATH_FORCEINLINE __m128 f(__m128 p)
                {
                    return _mm_unpacklo_ps(p, p);
                }
            };

            template_spec(struct load, UNPCKH)
            {
                META_COST({ COST = UNPCKH_COST })

                static MATH_FORCEINLINE __m128 f(__m128 p)
                {
                    return _mm_unpackhi_ps(p, p);
                }
            };

#       ifdef __SSE3__

            template_spec(struct load, MOVLDUP)
            {
                META_COST({ COST = MOVLDUP_COST })

                static MATH_FORCEINLINE __m128 f(__m128 p)
                {
                    return _mm_moveldup_ps(p);
                }
            };

            template_spec(struct load, MOVHDUP)
            {
                META_COST({ COST = MOVHDUP_COST })

                static MATH_FORCEINLINE __m128 f(__m128 p)
                {
                    return _mm_movehdup_ps(p);
                }
            };

#       endif

#       if defined(__AVX2__)

            template_spec(struct load, BROADCAST)
            {
                META_COST({ COST = BROADCAST_COST })

                static MATH_FORCEINLINE __m128 f(__m128 p)
                {
                    return _mm_broadcastss_ps(p);
                }
            };

#       endif

#   endif

            META_COST({ COST = template_inst(load, OP) ::COST })

            static MATH_FORCEINLINE __m128 f(__m128 p)
            {
                return template_inst(load, OP) ::f(p);
            }
        };

        template<int LHS, int RHS, int SEL> struct MASK
        {
            enum
            {
                L_SCL = ISDUP(LHS),
                R_SCL = ISDUP(RHS),

                L_MSK = ~SEL & MASK(LHS),
                L_SWZ = (L_SCL ? SWZ_XYZW : LHS) & L_MSK,
                R_MSK = ~L_MSK & MASK(RHS),
                R_SWZ = (R_SCL ? SWZ_XYZW : RHS) & R_MSK,

                IGN = ~(L_MSK | R_MSK) & MSK_XYZW,

                L_POP = POP(L_SWZ),
                R_POP = POP(R_SWZ)
            };

            enum
            {
                SIMPLE =
                    (MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_Y, COMP_Z, COMP_W), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_N, COMP_N, COMP_N), R_MSK, IGN)) ||
                    (MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_Y, COMP_Z, COMP_W), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_N, COMP_N, COMP_N), L_MSK, IGN)),

                MERGE =
                    (L_SWZ == 0 && R_SWZ == 0) ? DEFAULT :
                    L_SWZ == 0 ? MOV_R :
                    R_SWZ == 0 ? MOV :
#           ifdef META_PEEPHOLE
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_N, COMP_Y, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Y), R_MSK, IGN) ? UNPCKL :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_N, COMP_Y, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Y), L_MSK, IGN) ? UNPCKL_R :
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Z, COMP_N, COMP_W, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_Z, COMP_N, COMP_W), R_MSK, IGN) ? UNPCKH :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Z, COMP_N, COMP_W, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_Z, COMP_N, COMP_W), L_MSK, IGN) ? UNPCKH_R :
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_Z, COMP_W), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Z, COMP_W, COMP_N, COMP_N), R_MSK, IGN) ? MOVHL :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_Z, COMP_W), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Z, COMP_W, COMP_N, COMP_N), L_MSK, IGN) ? MOVHL_R :
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_Y, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_X, COMP_Y), R_MSK, IGN) ? MOVLH :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_Y, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_X, COMP_Y), L_MSK, IGN) ? MOVLH_R :
#           endif
                    (R_MSK & MSK_XY) == 0 && (L_MSK & MSK_ZW) == 0 ? SHUF :
                    (R_MSK & MSK_ZW) == 0 && (L_MSK & MSK_XY) == 0 ? SHUF_R :
#           ifdef META_PEEPHOLE
#               ifdef __SSE4_1__
                    (R_POP == 1 && !SIMPLE && MATCH_SWIZ(L_SCL, L_SWZ, SWZ_XYZW, L_MSK, IGN)) ? INSERT :
                    (L_POP == 1 && !SIMPLE && MATCH_SWIZ(R_SCL, R_SWZ, SWZ_XYZW, R_MSK, IGN)) ? INSERT_R :
#               endif
#               if defined(__SSSE3__)
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_W, COMP_N, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_X, COMP_Y, COMP_Z), R_MSK, IGN) ? PALIGNR1 :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_W, COMP_N, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_X, COMP_Y, COMP_Z), L_MSK, IGN) ? PALIGNR1_R :
                    MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_N, COMP_X), R_MSK, IGN) ? PALIGNR3 :
                    MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_N, COMP_X), L_MSK, IGN) ? PALIGNR3_R :
#               endif
#           endif
                    L_MSK == (int)MSK_X ? MOVS_R :
                    R_MSK == (int)MSK_X ? MOVS :
                    L_MSK < (int)R_MSK ? BLEND : BLEND_R
            };

            enum
            {
                LLOAD =
                    L_SWZ == 0 ? MOV :
                    L_SCL ? MOV :
                    R_SWZ == 0 ?
                    L_SWZ == (SWZ_XYZW & L_MSK) ? MOV :
                    DEFAULT :
                    MERGE != int(MOVS) && MERGE != int(MOVS_R) && MERGE != int(BLEND) && MERGE != int(BLEND_R) ? MOV :
                    (L_SCL || L_SWZ == (SWZ_XYZW & L_MSK)) ? MOV :
                    DEFAULT
            };

            enum
            {
                RLOAD =
                    R_SWZ == 0 ? MOV :
                    R_SCL ? MOV :
                    L_SWZ == 0 ?
                    R_SWZ == (SWZ_XYZW & R_MSK) ? MOV :
                    DEFAULT :
                    MERGE != int(MOVS) && MERGE != int(MOVS_R) && MERGE != int(BLEND) && MERGE != int(BLEND_R) ? MOV :
                    (R_SCL || R_SWZ == (SWZ_XYZW & R_MSK)) ? MOV :
                    DEFAULT
            };

            template_decl(struct lload, unsigned op)
            {
                META_COST({ COST = SWIZ<L_SWZ>::COST })

                static MATH_FORCEINLINE __m128 f(__m128 a)
                {
                    return SWIZ<L_SWZ>::f(a);
                }
            };

            template_spec(struct lload, MOV)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128 f(__m128 a)
                {
                    return a;
                }
            };

            template_decl(struct rload, unsigned op)
            {
                META_COST({ COST = SWIZ<R_SWZ>::COST })

                static MATH_FORCEINLINE __m128 f(__m128 a)
                {
                    return SWIZ<R_SWZ>::f(a);
                }
            };

            template_spec(struct rload, MOV)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128 f(__m128 a)
                {
                    return a;
                }
            };

            template_decl(struct merge, unsigned op)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128)
                {
                    return a;
                }
            };

            template_spec(struct merge, CLEAR)
            {
                META_COST({ COST = XOR_COST })

                static MATH_FORCEINLINE __m128 f(__m128, __m128)
                {
                    return _mm_setzero_ps();
                }
            };

            template_spec(struct merge, MOV)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128)
                {
                    return a;
                }
            };

            template_spec(struct merge, MOV_R)
            {
                META_COST({ COST = 0 })

                static MATH_FORCEINLINE __m128 f(__m128, __m128 b)
                {
                    return b;
                }
            };

            template_spec(struct merge, MOVS)
            {
                META_COST({ COST = MOVS_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_move_ss(a, b);
                }
            };

            template_spec(struct merge, MOVS_R)
            {
                META_COST({ COST = MOVS_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_move_ss(b, a);
                }
            };

#   ifdef META_PEEPHOLE

#       ifdef __SSE4_1__

            template_spec(struct merge, INSERT)
            {
                enum
                {
                    COMP = COMP(R_SWZ, FFS(R_SWZ)) - COMP_X,
                    COUNT_S = COMP,
                    COUNT_D = COMP(R_MSK, 0) ? 0 : (COMP(R_MSK, 1) ? 1 : (COMP(R_MSK, 2) ? 2 : 3)),
                    ZMASK = 0,
                    NDX = ZMASK | (COUNT_D << 4) | (COUNT_S << 6)
                };

                META_COST({ COST = INSERT_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_insert_ps(a, b, NDX);
                }
            };

            template_spec(struct merge, INSERT_R)
            {
                enum
                {
                    COMP = COMP(L_SWZ, FFS(L_SWZ)) - COMP_X,
                    COUNT_S = COMP,
                    COUNT_D = COMP(L_MSK, 0) ? 0 : (COMP(L_MSK, 1) ? 1 : (COMP(L_MSK, 2) ? 2 : 3)),
                    ZMASK = 0,
                    NDX = ZMASK | (COUNT_D << 4) | (COUNT_S << 6)
                };

                META_COST({ COST = INSERT_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_insert_ps(b, a, NDX);
                }
            };

#       endif

#   endif

            template_spec(struct merge, UNPCKL)
            {
                META_COST({ COST = UNPCKL_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_unpacklo_ps(a, b);
                }
            };

            template_spec(struct merge, UNPCKL_R)
            {
                META_COST({ COST = UNPCKL_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_unpacklo_ps(b, a);
                }
            };

            template_spec(struct merge, UNPCKH)
            {
                META_COST({ COST = UNPCKH_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_unpackhi_ps(a, b);
                }
            };

            template_spec(struct merge, UNPCKH_R)
            {
                META_COST({ COST = UNPCKH_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_unpackhi_ps(b, a);
                }
            };

            template_spec(struct merge, MOVLH)
            {
                META_COST({ COST = MOVLH_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_movelh_ps(a, b);
                }
            };

            template_spec(struct merge, MOVLH_R)
            {
                META_COST({ COST = MOVLH_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_movelh_ps(b, a);
                }
            };

            template_spec(struct merge, MOVHL)
            {
                META_COST({ COST = MOVHL_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_movehl_ps(a, b);
                }
            };

            template_spec(struct merge, MOVHL_R)
            {
                META_COST({ COST = MOVHL_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_movehl_ps(b, a);
                }
            };

            template_spec(struct merge, SHUF)
            {
                enum
                {
                    L_DEF = COMP_X + FFS(L_MSK),
                    L_EXT = (L_MSK & LHS) | (SWZ(L_DEF, L_DEF, L_DEF, L_DEF) & ~L_MSK),
                    R_DEF = COMP_X + FFS(R_MSK),
                    R_EXT = (R_MSK & RHS) | (SWZ(R_DEF, R_DEF, R_DEF, R_DEF) & ~R_MSK),
                    X = COMP(L_EXT, 0) == COMP_A ? 0 : (COMP(L_EXT, 0) - COMP_X),
                    Y = COMP(L_EXT, 1) == COMP_A ? 1 : (COMP(L_EXT, 1) - COMP_X),
                    Z = COMP(R_EXT, 2) == COMP_A ? 2 : (COMP(R_EXT, 2) - COMP_X),
                    W = COMP(R_EXT, 3) == COMP_A ? 3 : (COMP(R_EXT, 3) - COMP_X),
                    IMM = _MM_SHUFFLE(W, Z, Y, X)
                };

                META_COST({ COST = SHUF_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_shuffle_ps(a, b, IMM);
                }
            };

            template_spec(struct merge, SHUF_R)
            {
                enum
                {
                    L_DEF = COMP_X + FFS(L_MSK),
                    L_EXT = (L_MSK & LHS) | (SWZ(L_DEF, L_DEF, L_DEF, L_DEF) & ~L_MSK),
                    R_DEF = COMP_X + FFS(R_MSK),
                    R_EXT = (R_MSK & RHS) | (SWZ(R_DEF, R_DEF, R_DEF, R_DEF) & ~R_MSK),
                    X = COMP(R_EXT, 0) == COMP_A ? 0 : (COMP(R_EXT, 0) - COMP_X),
                    Y = COMP(R_EXT, 1) == COMP_A ? 1 : (COMP(R_EXT, 1) - COMP_X),
                    Z = COMP(L_EXT, 2) == COMP_A ? 2 : (COMP(L_EXT, 2) - COMP_X),
                    W = COMP(L_EXT, 3) == COMP_A ? 3 : (COMP(L_EXT, 3) - COMP_X),
                    IMM = _MM_SHUFFLE(W, Z, Y, X)
                };

                META_COST({ COST = SHUF_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_shuffle_ps(b, a, IMM);
                }
            };

#       if defined(__SSSE3__)

            template_spec(struct merge, PALIGNR1)
            {
                META_COST({ COST = PALIGNR_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(a), _mm_castps_si128(b), 4));
                }
            };

            template_spec(struct merge, PALIGNR1_R)
            {
                META_COST({ COST = PALIGNR_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(b), _mm_castps_si128(a), 4));
                }
            };

            template_spec(struct merge, PALIGNR3)
            {
                META_COST({ COST = PALIGNR_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(a), _mm_castps_si128(b), 12));
                }
            };

            template_spec(struct merge, PALIGNR3_R)
            {
                META_COST({ COST = PALIGNR_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(b), _mm_castps_si128(a), 12));
                }
            };

#       endif

            template_spec(struct merge, BLEND)
            {
#           ifdef __SSE4_1__

                enum
                {
                    MASK = (COMP(R_MSK, 0) ? 1 : 0) | (COMP(R_MSK, 1) ? 2 : 0) | (COMP(R_MSK, 2) ? 4 : 0) | (COMP(R_MSK, 3) ? 8 : 0)
                };

                META_COST({ COST = BLEND_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_blend_ps(a, b, MASK);
                }

#           else

                enum
                {
                    X = COMP(R_MSK, 0) != 0,
                    Y = COMP(R_MSK, 1) != 0,
                    Z = COMP(R_MSK, 2) != 0,
                    W = COMP(R_MSK, 3) != 0
                };

                META_COST({ COST = AND_COST + ANDN_COST + OR_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    const __m128 c = _mm_castsi128_ps((__m128i)_mm_setr_epi32(-X, -Y, -Z, -W));
                    return _mm_or_ps(_mm_and_ps(b, c), _mm_andnot_ps(c, a));
                }

#           endif
            };

            template_spec(struct merge, BLEND_R)
            {
#           ifdef __SSE4_1__

                enum
                {
                    MASK = (COMP(L_MSK, 0) ? 1 : 0) | (COMP(L_MSK, 1) ? 2 : 0) | (COMP(L_MSK, 2) ? 4 : 0) | (COMP(L_MSK, 3) ? 8 : 0)
                };

                META_COST({ COST = BLEND_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    return _mm_blend_ps(b, a, MASK);
                }

#           else

                enum
                {
                    X = COMP(L_MSK, 0) != 0,
                    Y = COMP(L_MSK, 1) != 0,
                    Z = COMP(L_MSK, 2) != 0,
                    W = COMP(L_MSK, 3) != 0
                };

                META_COST({ COST = AND_COST + ANDN_COST + OR_COST })

                static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
                {
                    const __m128 c = _mm_castsi128_ps((__m128i)_mm_setr_epi32(-X, -Y, -Z, -W));
                    return _mm_or_ps(_mm_and_ps(a, c), _mm_andnot_ps(c, b));
                }

#           endif
            };

            META_COST({ COST = template_inst(lload, LLOAD) ::COST + template_inst(rload, RLOAD) ::COST + template_inst(merge, MERGE) ::COST })

            static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b)
            {
                __m128 l = template_inst(lload, LLOAD) ::f(a);
                __m128 r = template_inst(rload, RLOAD) ::f(b);
                __m128 d = template_inst(merge, MERGE) ::f(l, r);
                return d;
            }
        };

        template<int c> struct GET
        {
            template_decl(struct impl, int u)
            {
                static MATH_FORCEINLINE float g(__m128 p)
                {
                    return _mm_cvtss_f32(SWIZ<c>::f(p));
                }
            };

            template_spec(struct impl, 0)
            {
                static MATH_FORCEINLINE float g(__m128 p)
                {
                    return _mm_cvtss_f32(p);
                }
            };

            static MATH_FORCEINLINE float f(__m128 p)
            {
                return template_inst(impl, FFS(USED(c))) ::g(p);
            }
        };

        template<int c> struct SET
        {
            static MATH_FORCEINLINE __m128 f(__m128 p, float x)
            {
                return MASK<SWZ_XYZW, Inverse<COMP_X, c>::SWZ, USED(c)>::f(p, _mm_cvtf32_ss(x));
            }
        };

        template<int X, int Y, int Z, int W> struct GATHER
        {
            enum
            {
                IGN = (ISDUP(X) ? MSK_X : 0) | (ISDUP(Y) ? MSK_Y : 0) | (ISDUP(Z) ? MSK_Z : 0) | (ISDUP(W) ? MSK_W : 0),
                SWZ = SWZ(COMP(X, 0), COMP(Y, 0), COMP(Z, 0), COMP(W, 0)),
                MSK = MASK(SWZ),
                SEL =
                    (MSK & MSK_ZW) == 0 ? SWZ_XY :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_X, COMP_Y, COMP_Z, COMP_W), MSK, IGN) ? SWZ_XYZW :
#           ifdef META_PEEPHOLE
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_X, COMP_X, COMP_X, COMP_X), MSK, IGN) ? SWZ_XXXX :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_Y), MSK, IGN) ? SWZ_YYYY :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_Z), MSK, IGN) ? SWZ_ZZZZ :
                    MATCH_SWIZ(0, SWZ, SWZ(COMP_W, COMP_W, COMP_W, COMP_W), MSK, IGN) ? SWZ_WWWW :
#           endif
                    0
            };

            template_decl(struct impl, int swz)
            {
                static MATH_FORCEINLINE __m128 g(__m128 a, __m128 b, __m128 c, __m128_arg d)
                {
                    return MASK<SWZ_XYZW, SWZ_ZWXY, MSK_ZW>::f(MASK<X & MSK_X, (Y << 4) & MSK_Y, MSK_Y>::f(a, b), MASK<Z & MSK_X, (W << 4) & MSK_Y, MSK_Y>::f(c, d));
                }
            };

            template_spec(struct impl, SWZ_XY)
            {
                static MATH_FORCEINLINE __m128 g(__m128 a, __m128 b, __m128, __m128_arg)
                {
                    return MASK<X & MSK_X, (Y << 4) & MSK_Y, MSK_Y>::f(a, b);
                }
            };

            template_spec(struct impl, SWZ_XYZW)
            {
                static MATH_FORCEINLINE __m128 g(__m128 a, __m128 b, __m128 c, __m128_arg d)
                {
                    return _mm_shuffle_ps(_mm_unpacklo_ps(a, b), _mm_unpackhi_ps(c, d), _MM_SHUFFLE(3, 0, 3, 0));
                }
            };

#       ifdef META_PEEPHOLE

            template_spec(struct impl, SWZ_XXXX)
            {
                static MATH_FORCEINLINE __m128 g(__m128 a, __m128 b, __m128 c, __m128_arg d)
                {
                    return _mm_movelh_ps(_mm_unpacklo_ps(a, b), _mm_unpacklo_ps(c, d));
                }
            };

            template_spec(struct impl, SWZ_YYYY)
            {
                static MATH_FORCEINLINE __m128 g(__m128 a, __m128 b, __m128 c, __m128_arg d)
                {
                    return _mm_movehl_ps(_mm_unpacklo_ps(c, d), _mm_unpacklo_ps(a, b));
                }
            };

            template_spec(struct impl, SWZ_ZZZZ)
            {
                static MATH_FORCEINLINE __m128 g(__m128 a, __m128 b, __m128 c, __m128_arg d)
                {
                    return _mm_movelh_ps(_mm_unpackhi_ps(a, b), _mm_unpackhi_ps(c, d));
                }
            };

            template_spec(struct impl, SWZ_WWWW)
            {
                static MATH_FORCEINLINE __m128 g(__m128 a, __m128 b, __m128 c, __m128_arg d)
                {
                    return _mm_movehl_ps(_mm_unpackhi_ps(c, d), _mm_unpackhi_ps(a, b));
                }
            };

#endif

            static MATH_FORCEINLINE __m128 f(__m128 a, __m128 b, __m128 c, __m128_arg d)
            {
                return template_inst(impl, SEL) ::g(a, b, c, d);
            }
        };

        template<int SWZ> struct ANY
        {
            enum
            {
                USE = USED(SWZ),
                MSK = (COMP(USE, 0) ? 1 : 0) | (COMP(USE, 1) ? 2 : 0) | (COMP(USE, 2) ? 4 : 0) | (COMP(USE, 3) ? 8 : 0)
            };

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE int g(__m128 p)
                {
                    return (_mm_movemask_ps(p) & MSK) != 0;
                }
            };

            template_spec(struct impl, MSK_XYZW)
            {
                static MATH_FORCEINLINE int g(__m128 p)
                {
                    return _mm_movemask_ps(p) != 0;
                }
            };

            static MATH_FORCEINLINE int f(__m128 p)
            {
                return template_inst(impl, USE) ::g(p);
            }
        };

        template<int SWZ> struct ALL
        {
            enum
            {
                USE = USED(SWZ),
                MSK = (COMP(USE, 0) ? 1 : 0) | (COMP(USE, 1) ? 2 : 0) | (COMP(USE, 2) ? 4 : 0) | (COMP(USE, 3) ? 8 : 0)
            };

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE int g(__m128 p)
                {
                    return (_mm_movemask_ps(p) & MSK) == MSK;
                }
            };

            template_spec(struct impl, MSK_XYZW)
            {
                static MATH_FORCEINLINE int g(__m128 p)
                {
                    return _mm_movemask_ps(p) == MSK;
                }
            };

            static MATH_FORCEINLINE int f(__m128 p)
            {
                return template_inst(impl, USE) ::g(p);
            }
        };

        template<int RHS = SWZ_XYZW> struct NEG
        {
            enum
            {
                R = UnOp<v4f, RHS>::R,
                S = UnOp<v4f, RHS>::S
            };

            META_COST({ COST = (UnOp<v4f, RHS>::COST) + XOR_COST })

            typedef v4f     type;

            static MATH_FORCEINLINE __m128 f(__m128 rhs)
            {
                return _mm_xor_ps(SWIZ<R>::f(rhs), cv4f(-0.f, -0.f, -0.f, -0.f));
            }
        };

        template<int RHS = SWZ_XYZW> struct ABS
        {
            enum
            {
                R = UnOp<v4f, RHS>::R,
                S = UnOp<v4f, RHS>::S
            };

            META_COST({ COST = (UnOp<v4f, RHS>::COST) + AND_COST })

            typedef v4f     type;

            static MATH_FORCEINLINE __m128 f(__m128 rhs)
            {
                return _mm_and_ps(SWIZ<R>::f(rhs), _mm_castsi128_ps((__m128i)_mm_setr_epi32(0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff)));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct ADD
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + ADD_COST })

            typedef v4f     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 rhs)
                {
                    return _mm_add_ps(lhs, rhs);
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 rhs)
                {
                    return _mm_add_ss(lhs, rhs);
                }
            };

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SUB
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + SUB_COST })

            typedef v4f     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 rhs)
                {
                    return _mm_sub_ps(lhs, rhs);
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 rhs)
                {
                    return _mm_sub_ss(lhs, rhs);
                }
            };

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MUL
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + MUL_COST })

            typedef v4f     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 rhs)
                {
                    return _mm_mul_ps(lhs, rhs);
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 rhs)
                {
                    return _mm_mul_ss(lhs, rhs);
                }
            };

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct DIV
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + DIV_COST })

            typedef v4f     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 rhs)
                {
                    return _mm_div_ps(lhs, rhs);
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 rhs)
                {
                    return _mm_div_ss(lhs, rhs);
                }
            };

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPEQ
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + CMP_COST })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmpeq_ps(lhs, rhs));
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmpeq_ss(lhs, rhs));
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPNE
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + CMP_COST })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmpneq_ps(lhs, rhs));
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmpneq_ss(lhs, rhs));
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPLE
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + CMP_COST })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmple_ps(lhs, rhs));
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmple_ss(lhs, rhs));
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPLT
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + CMP_COST })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmplt_ps(lhs, rhs));
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmplt_ss(lhs, rhs));
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPGE
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + CMP_COST })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmpge_ps(lhs, rhs));
                }
            };
            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmpge_ss(lhs, rhs));
                }
            };
            static MATH_FORCEINLINE __m128i f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPGT
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS, MSK_X>::L,
                R = BinOp<v4f, LHS, RHS, MSK_X>::R,
                S = BinOp<v4f, LHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS, MSK_X>::COST) + CMP_COST })

            typedef v4i     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmpgt_ps(lhs, rhs));
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128i g(__m128 lhs, __m128 rhs)
                {
                    return _mm_castps_si128(_mm_cmpgt_ss(lhs, rhs));
                }
            };

            static MATH_FORCEINLINE __m128i f(__m128 lhs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int CHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MADD
        {
#       if defined(__FMA__)

            enum
            {
                L = TernOp<v4f, LHS, CHS, RHS, MSK_X>::L,
                C = TernOp<v4f, LHS, CHS, RHS, MSK_X>::C,
                R = TernOp<v4f, LHS, CHS, RHS, MSK_X>::R,
                S = TernOp<v4f, LHS, CHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = SWIZ<L>::COST + SWIZ<C>::COST + SWIZ<R>::COST + MADD_COST })

            typedef v4f     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 chs, __m128 rhs)
                {
                    return _mm_fmadd_ps(SWIZ<L>::f(lhs), SWIZ<C>::f(chs), SWIZ<R>::f(rhs));
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 chs, __m128 rhs)
                {
                    return _mm_fmadd_ss(SWIZ<L>::f(lhs), SWIZ<C>::f(chs), SWIZ<R>::f(rhs));
                }
            };

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 chs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, chs, rhs);
            }

#       else

            enum
            {
                L0 = TernOp<v4f, LHS, CHS, RHS, MSK_X>::L0,
                C0 = TernOp<v4f, LHS, CHS, RHS, MSK_X>::C0,
                S0 = TernOp<v4f, LHS, CHS, RHS, MSK_X>::S0,
                LC = TernOp<v4f, LHS, CHS, RHS, MSK_X>::LC,
                R = TernOp<v4f, LHS, CHS, RHS, MSK_X>::R,
                S = TernOp<v4f, LHS, CHS, RHS, MSK_X>::S,
                SEL = USED(S0) == MSK_X && USED(S) == MSK_X
            };

            META_COST({ COST = (TernOp<v4f, LHS, CHS, RHS, MSK_X>::COST) + MUL_COST + ADD_COST })

            typedef v4f     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 chs, __m128 rhs)
                {
                    return _mm_add_ps(SWIZ<LC>::f(_mm_mul_ps(SWIZ<L0>::f(lhs), SWIZ<C0>::f(chs))), SWIZ<R>::f(rhs));
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 chs, __m128 rhs)
                {
                    return _mm_add_ss(SWIZ<LC>::f(_mm_mul_ss(SWIZ<L0>::f(lhs), SWIZ<C0>::f(chs))), SWIZ<R>::f(rhs));
                }
            };

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 chs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, chs, rhs);
            }

#       endif
        };

        template<int LHS = SWZ_XYZW, int CHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MSUB
        {
#       if defined(__FMA__)

            enum
            {
                L = TernOp<v4f, LHS, CHS, RHS, MSK_X>::L,
                C = TernOp<v4f, LHS, CHS, RHS, MSK_X>::C,
                R = TernOp<v4f, LHS, CHS, RHS, MSK_X>::R,
                S = TernOp<v4f, LHS, CHS, RHS, MSK_X>::S,
                SEL = USED(S) == MSK_X
            };

            META_COST({ COST = SWIZ<L>::COST + SWIZ<C>::COST + SWIZ<R>::COST + MADD_COST })

            typedef v4f     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 chs, __m128 rhs)
                {
                    return _mm_fmsub_ps(SWIZ<L>::f(lhs), SWIZ<C>::f(chs), SWIZ<R>::f(rhs));
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 chs, __m128 rhs)
                {
                    return _mm_fmsub_ss(SWIZ<L>::f(lhs), SWIZ<C>::f(chs), SWIZ<R>::f(rhs));
                }
            };

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 chs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, chs, rhs);
            }

#       else

            enum
            {
                L0 = TernOp<v4f, LHS, CHS, RHS, MSK_X>::L0,
                C0 = TernOp<v4f, LHS, CHS, RHS, MSK_X>::C0,
                S0 = TernOp<v4f, LHS, CHS, RHS, MSK_X>::S0,
                LC = TernOp<v4f, LHS, CHS, RHS, MSK_X>::LC,
                R = TernOp<v4f, LHS, CHS, RHS, MSK_X>::R,
                S = TernOp<v4f, LHS, CHS, RHS, MSK_X>::S,
                SEL = USED(S0) == MSK_X && USED(S) == MSK_X
            };

            META_COST({ COST = (TernOp<v4f, LHS, CHS, RHS, MSK_X>::COST) + MUL_COST + SUB_COST })

            typedef v4f     type;

            template_decl(struct impl, int sel)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 chs, __m128 rhs)
                {
                    return _mm_sub_ps(SWIZ<LC>::f(_mm_mul_ps(SWIZ<L0>::f(lhs), SWIZ<C0>::f(chs))), SWIZ<R>::f(rhs));
                }
            };

            template_spec(struct impl, 1)
            {
                static MATH_FORCEINLINE __m128 g(__m128 lhs, __m128 chs, __m128 rhs)
                {
                    return _mm_sub_ss(SWIZ<LC>::f(_mm_mul_ss(SWIZ<L0>::f(lhs), SWIZ<C0>::f(chs))), SWIZ<R>::f(rhs));
                }
            };

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 chs, __m128 rhs)
            {
                return template_inst(impl, SEL) ::g(lhs, chs, rhs);
            }

#       endif
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MIN
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS>::L,
                R = BinOp<v4f, LHS, RHS>::R,
                S = BinOp<v4f, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + MIN_COST })

            typedef v4f     type;

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 rhs)
            {
                return _mm_min_ps(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };

        template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MAX
        {
            enum
            {
                L = BinOp<v4f, LHS, RHS>::L,
                R = BinOp<v4f, LHS, RHS>::R,
                S = BinOp<v4f, LHS, RHS>::S
            };

            META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + MAX_COST })

            typedef v4f     type;

            static MATH_FORCEINLINE __m128 f(__m128 lhs, __m128 rhs)
            {
                return _mm_max_ps(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
            }
        };
    };
}
}
