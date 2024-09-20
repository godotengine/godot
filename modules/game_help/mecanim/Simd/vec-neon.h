#pragma once

namespace math
{

namespace meta
{

	struct v4i;

	struct v4f
	{
		typedef float32x4_t	packed;
		typedef float		type;
		
		enum
		{
			// COST: 0
			CASE_XYZW,
			// COST: 1.1
			CASE_XYXY,
			// COST: 1.1
			CASE_ZWZW,
			// COST: 1.1
			CASE_XXXX,
			// COST: 1.1
			CASE_YYYY,
			// COST: 1.1
			CASE_ZZZZ,
			// COST: 1.1
			CASE_WWWW,
			// COST: 1.1
			CASE_YXWZ,
			// COST: 1.1
			CASE_YZWX,
			// COST: 1.1
			CASE_ZWXY,
			// COST: 1.1
			CASE_WXYZ,
			
#		if defined(__AARCH64_SIMD__) || defined(META_PEEPHOLE)

			// COST: 1.1
			CASE_XYXX,
			// COST: 1.1
			CASE_XYYX,
			// COST: 1.1
			CASE_XYWX,
			// COST: 1.1
			CASE_XYZZ,
			// COST: 1.1
			CASE_XYWZ,
			// COST: 1.1
			CASE_XXZW,
			// COST: 1.1
			CASE_YXZW,
			// COST: 1.1
			CASE_WXZW,
			// COST: 1.1
			CASE_YYZW,
			// COST: 1.1
			CASE_WWZW,
			// COST: 1.1
			CASE_XYWW,

			// COST: 1.1+
			CASE_ZWXX,
			// COST: 1.1+
			CASE_ZWYX,
			// COST: 1.1+
			CASE_YYXY,
			// COST: 1.1+
			CASE_YZXY,
			// COST: 1.1+
			CASE_ZZXY,
			// COST: 1.1+
			CASE_WZXY,
			// COST: 1.1+
			CASE_WWXY,
			// COST: 1.1+
			CASE_ZWYY,
			// COST: 1.1+
			CASE_ZWYZ,
			// COST: 1.1+
			CASE_ZWZZ,

			// COST: 2.1
			CASE_WXXX,
			// COST: 2.1
			CASE_YZXX,
			// COST: 2.1
			CASE_ZZXX,
			// COST: 2.1
			CASE_WZXX,
			// COST: 2.1
			CASE_WWXX,
			// COST: 2.1
			CASE_WXYX,
			// COST: 2.1
			CASE_YYYX,
			// COST: 2.1
			CASE_YZYX,
			// COST: 2.1
			CASE_ZZYX,
			// COST: 2.1
			CASE_WZYX,
			// COST: 2.1
			CASE_WWYX,
			// COST: 2.1
			CASE_XXWX,
			// COST: 2.1
			CASE_YXWX,
			// COST: 2.1
			CASE_YYWX,
			// COST: 2.1
			CASE_ZZWX,
			// COST: 2.1
			CASE_WZWX,
			// COST: 2.1
			CASE_WWWX,
			// COST: 2.1
			CASE_WXYY,
			// COST: 2.1
			CASE_YZYY,
			// COST: 2.1
			CASE_ZZYY,
			// COST: 2.1
			CASE_WZYY,
			// COST: 2.1
			CASE_WWYY,
			// COST: 2.1
			CASE_XXYZ,
			// COST: 2.1
			CASE_YXYZ,
			// COST: 2.1
			CASE_YYYZ,
			// COST: 2.1
			CASE_ZZYZ,
			// COST: 2.1
			CASE_WZYZ,
			// COST: 2.1
			CASE_WWYZ,
			// COST: 2.1
			CASE_YXZZ,
			// COST: 2.1
			CASE_WXZZ,
			// COST: 2.1
			CASE_YZZZ,
			// COST: 2.1
			CASE_WZZZ,
			// COST: 2.1
			CASE_XWZZ,
			// COST: 2.1
			CASE_XXWZ,
			// COST: 2.1
			CASE_WXWZ,
			// COST: 2.1
			CASE_YYWZ,
			// COST: 2.1
			CASE_YZWZ,
			// COST: 2.1
			CASE_WWWZ,
			// COST: 2.1
			CASE_XXWW,
			// COST: 2.1
			CASE_YXWW,
			// COST: 2.1
			CASE_WXWW,
			// COST: 2.1
			CASE_YZWW,

			// COST: 2.2
			CASE_XXYY,
			// COST: 2.2
			CASE_ZZWW,
			// COST: 2.2
			CASE_ZXWY,
			// COST: 2.2
			CASE_XZXZ,
			// COST: 2.2
			CASE_YWYW,
			// COST: 2.2
			CASE_XZXX,
			// COST: 2.2
			CASE_YWXX,
			// COST: 2.2
			CASE_XXYX,
			// COST: 2.2
			CASE_XYYY,
			// COST: 2.2
			CASE_ZXWX,
			// COST: 2.2
			CASE_XZYY,
			// COST: 2.2
			CASE_YWYY,
			// COST: 2.2
			CASE_XZYW,
			// COST: 2.2
			CASE_YWXZ,
			// COST: 2.2
			CASE_XYYZ,
			// COST: 2.2
			CASE_XWYX,
			// COST: 2.2
			CASE_ZYWZ,
			// COST: 2.2
			CASE_XYZX,
			// COST: 2.2
			CASE_ZWWX,
			// COST: 2.2
			CASE_WYZW,
			// COST: 2.2
			CASE_ZXYW,
			// COST: 2.2
			CASE_YWZX,
			// COST: 2.2
			CASE_XZWY,
			// COST: 2.2
			CASE_XXXY,
			// COST: 2.2
			CASE_WYYY,
			// COST: 2.2
			CASE_XWZX,
			// COST: 2.2
			CASE_YZWY,
			// COST: 2.2
			CASE_WYXW,
			// COST: 2.2
			CASE_XWZY,
			// COST: 2.2
			CASE_ZYXW,
			// COST: 2.2
			CASE_XXZZ,
			// COST: 2.2
			CASE_YYWW,
			// COST: 2.2
			CASE_XXZX,
			// COST: 2.2
			CASE_YYWY,
			// COST: 2.2
			CASE_YXXX,
			// COST: 2.2
			CASE_YXYX,
			// COST: 2.2
			CASE_XWWX,
			// COST: 2.2
			CASE_WXZY,
			// COST: 2.2
			CASE_XYZY,
			// COST: 2.2
			CASE_YZZY,
			// COST: 2.2
			CASE_ZYWY,
			// COST: 2.2
			CASE_ZYYZ,
			// COST: 2.2
			CASE_YZYZ,
			// COST: 2.2
			CASE_WZWZ,
			// COST: 2.2
			CASE_XYXW,
			// COST: 2.2
			CASE_XYYW,
			// COST: 2.2
			CASE_WXWX,
			// COST: 2.2
			CASE_YZYW,
			// COST: 2.2
			CASE_ZYZW,
			// COST: 2.2
			CASE_XZZW,
			// COST: 2.2
			CASE_XWZW,

			// COST: 2.2+
			CASE_XZWW,
			// COST: 2.2+
			CASE_XWYW,
			// COST: 2.2+
			CASE_XZYZ,
			// COST: 2.2+
			CASE_ZZWZ,
			// COST: 2.2+
			CASE_XZZZ,
			// COST: 2.2+
			CASE_ZZZX,
			// COST: 2.2+
			CASE_ZWWW,
			// COST: 2.2+
			CASE_ZYXY,
			// COST: 2.2+
			CASE_XZXY,
			// COST: 2.2+
			CASE_XWXY,
			// COST: 2.2+
			CASE_ZWZY,
			// COST: 2.2+
			CASE_ZWXW,
			// COST: 2.2+
			CASE_ZWYW,
#		endif

			// COST: 2.2+
			CASE_YZXW,

#		if defined(__AARCH64_SIMD__) || defined(META_PEEPHOLE)

			// COST: 3.2
			CASE_ZYXX,
			// COST: 3.2
			CASE_XWXX,
			// COST: 3.2
			CASE_ZYYX,
			// COST: 3.2
			CASE_XZYX,
			// COST: 3.2
			CASE_YWYX,
			// COST: 3.2
			CASE_ZYWX,
			// COST: 3.2
			CASE_ZYYY,
			// COST: 3.2
			CASE_XWYY,
			// COST: 3.2
			CASE_XXZY,
			// COST: 3.2
			CASE_YYZY,
			// COST: 3.2
			CASE_ZZZY,
			// COST: 3.2
			CASE_WZZY,
			// COST: 3.2
			CASE_WWZY,
			// COST: 3.2
			CASE_XXXZ,
			// COST: 3.2
			CASE_YXXZ,
			// COST: 3.2
			CASE_YYXZ,
			// COST: 3.2
			CASE_ZZXZ,
			// COST: 3.2
			CASE_WZXZ,
			// COST: 3.2
			CASE_WWXZ,
			// COST: 3.2
			CASE_ZYZZ,
			// COST: 3.2
			CASE_YWZZ,
			// COST: 3.2
			CASE_XZWZ,
			// COST: 3.2
			CASE_XWWZ,
			// COST: 3.2
			CASE_YWWZ,
			// COST: 3.2
			CASE_XXXW,
			// COST: 3.2
			CASE_YXXW,
			// COST: 3.2
			CASE_YYXW,
			// COST: 3.2
			CASE_ZZXW,
			// COST: 3.2
			CASE_WWXW,
			// COST: 3.2
			CASE_XXYW,
			// COST: 3.2
			CASE_YXYW,
			// COST: 3.2
			CASE_YYYW,
			// COST: 3.2
			CASE_ZZYW,
			// COST: 3.2
			CASE_WZYW,
			// COST: 3.2
			CASE_WWYW,
			// COST: 3.2
			CASE_ZYWW,
			// COST: 3.2
			CASE_XWWW,
			// COST: 3.2
			CASE_YWWW,

			// COST: 3.3
			CASE_YYXX,
			// COST: 3.3
			CASE_YYZX,
			// COST: 3.3
			CASE_WYZX,
			// COST: 3.3
			CASE_XZZX,
			// COST: 3.3
			CASE_YXXY,
			// COST: 3.3
			CASE_WXXY,
			// COST: 3.3
			CASE_YXYY,
			// COST: 3.3
			CASE_YXZY,
			// COST: 3.3
			CASE_ZYZY,
			// COST: 3.3
			CASE_XXWY,
			// COST: 3.3
			CASE_XYWY,
			// COST: 3.3
			CASE_ZZWY,
			// COST: 3.3
			CASE_WYYZ,
			// COST: 3.3
			CASE_XWWY,
			// COST: 3.3
			CASE_ZXXW,
			// COST: 3.3
			CASE_WZWY,
			// COST: 3.3
			CASE_XZXW,
			// COST: 3.3
			CASE_YWWY,
			// COST: 3.3
			CASE_ZXXZ,
			// COST: 3.3
			CASE_XYXZ,
			// COST: 3.3
			CASE_WYXZ,
			// COST: 3.3
			CASE_XWXZ,
			// COST: 3.3
			CASE_ZXYZ,
			// COST: 3.3
			CASE_XWYZ,
			// COST: 3.3
			CASE_YYZZ,
			// COST: 3.3
			CASE_WYZZ,
			// COST: 3.3
			CASE_WWZZ,
			// COST: 3.3
			CASE_ZWWZ,
			// COST: 3.3
			CASE_WXXW,
			// COST: 3.3
			CASE_WZXW,
			// COST: 3.3
			CASE_XWXW,
			// COST: 3.3
			CASE_WYYW,
			// COST: 3.3
			CASE_ZXZW,
			// COST: 3.3
			CASE_YZZW,
			// COST: 3.3
			CASE_ZZZW,
			// COST: 3.3
			CASE_WZZW,
			// COST: 3.3
			CASE_YWZW,
			// COST: 3.3
			CASE_ZXWW,
			// COST: 3.3
			CASE_WZWW,

			// COST: 3.3+
			CASE_ZWZX,
			// COST: 3.3+
			CASE_WYXY,
			// COST: 3.3+
			CASE_YWXY,
			// COST: 3.3+
			CASE_ZXXY,
			// COST: 3.3+
			CASE_ZWWY,
			// COST: 3.3+
			CASE_ZWXZ,

			// COST: 4.2
			CASE_ZYXZ,

			// COST: 4.3
			CASE_ZXXX,
			// COST: 4.3
			CASE_WYXX,
			// COST: 4.3
			CASE_ZXYX,
			// COST: 4.3
			CASE_WYYX,
			// COST: 4.3
			CASE_YXZX,
			// COST: 4.3
			CASE_WZZX,
			// COST: 4.3
			CASE_WWZX,
			// COST: 4.3
			CASE_XZWX,
			// COST: 4.3
			CASE_YWWX,
			// COST: 4.3
			CASE_ZXYY,
			// COST: 4.3
			CASE_XZZY,
			// COST: 4.3
			CASE_YXWY,
			// COST: 4.3
			CASE_WWWY,
			// COST: 4.3
			CASE_WXXZ,
			// COST: 4.3
			CASE_YWYZ,
			// COST: 4.3
			CASE_ZXZZ,
			// COST: 4.3
			CASE_ZXWZ,
			// COST: 4.3
			CASE_WYWZ,
			// COST: 4.3
			CASE_WXYW,
			// COST: 4.3
			CASE_WYWW,

			// COST: 4.4
			CASE_ZXZX,
			// COST: 4.4
			CASE_WXZX,
			// COST: 4.4
			CASE_ZYZX,
			// COST: 4.4
			CASE_WYWX,
			// COST: 4.4
			CASE_WXWY,
			// COST: 4.4
			CASE_WYWY,
			// COST: 4.4
			CASE_YWXW,
			// COST: 4.4
			CASE_YZXZ,

			// COST: 5.3
			CASE_YWZY,
			// COST: 5.3
			CASE_ZYYW,

			// COST: 5.4
			CASE_YZZX,

			// COST: 5.5
			CASE_ZXZY,

			// COST: 6.4
			CASE_WYZY,
			
#		endif

			// COST: 8
			CASE_DEFAULT
		};

		static MATH_FORCEINLINE float32x4_t ZERO()
		{
			return vdupq_n_f32(0.f);
		}

		static MATH_FORCEINLINE float32x4_t CTOR(float x)
		{
			return vdupq_n_f32(x);
		}
		
		static MATH_FORCEINLINE float32x4_t CTOR(float x, float y)
		{
#if defined(_MSC_VER)
			const float values[] = { x, y, x, y };
			return vld1q_f32(values);
#else
			return (float32x4_t) { x, y, x, y };
#endif
		}
		
		static MATH_FORCEINLINE float32x4_t CTOR(float x, float y, float z)
		{
#if defined(_MSC_VER)
			const float values[] = { x, y, z, 0.f };
			return vld1q_f32(values);
#else
			return (float32x4_t) { x, y, z, 0.f };
#endif
		}
		
		static MATH_FORCEINLINE float32x4_t CTOR(float x, float y, float z, float w)
		{
#if defined(_MSC_VER)
			const float values[] = { x, y, z, w };
			return vld1q_f32(values);
#else
			return (float32x4_t) { x, y, z, w };
#endif
		}

#		define CASE_MATCH(X, Y, Z, W)	(SWZ & MSK)==(SWZ(COMP_##X, COMP_##Y, COMP_##Z, COMP_##W) & MSK) ? CASE_##X##Y##Z##W
		
		template<unsigned SWZ> struct SWIZ
		{
			enum
			{
				SCL = ISDUP(SWZ),

				MSK = MASK(SWZ),
				USE = USED(SWZ),
				
				CASE =

				// COST: 0
				SCL ? CASE_XYZW :
				(SWZ & MSK)==(SWZ_XYZW & MSK) ? CASE_XYZW :

				// COST: 1.1
				CASE_MATCH(X, Y, X, Y) :
				// COST: 1.1
				CASE_MATCH(Z, W, Z, W) :
				// COST: 1.1
				CASE_MATCH(X, X, X, X) :
				// COST: 1.1
				CASE_MATCH(Y, Y, Y, Y) :
				// COST: 1.1
				CASE_MATCH(Z, Z, Z, Z) :
				// COST: 1.1
				CASE_MATCH(W, W, W, W) :
				// COST: 1.1
				CASE_MATCH(Y, X, W, Z) :
				// COST: 1.1
				CASE_MATCH(Y, Z, W, X) :
				// COST: 1.1
				CASE_MATCH(Z, W, X, Y) :
				// COST: 1.1
				CASE_MATCH(W, X, Y, Z) :
				
#			if defined(__AARCH64_SIMD__) || defined(META_PEEPHOLE)

				// COST: 1.1
				CASE_MATCH(X, Y, X, X) :
				// COST: 1.1
				CASE_MATCH(X, Y, Y, X) :
				// COST: 1.1
				CASE_MATCH(X, Y, W, X) :
				// COST: 1.1
				CASE_MATCH(X, Y, Z, Z) :
				// COST: 1.1
				CASE_MATCH(X, Y, W, Z) :
				// COST: 1.1
				CASE_MATCH(X, X, Z, W) :
				// COST: 1.1
				CASE_MATCH(Y, X, Z, W) :
				// COST: 1.1
				CASE_MATCH(W, X, Z, W) :
				// COST: 1.1
				CASE_MATCH(Y, Y, Z, W) :
				// COST: 1.1
				CASE_MATCH(W, W, Z, W) :
				// COST: 1.1
				CASE_MATCH(X, Y, W, W) :

				// COST: 1.1+
				CASE_MATCH(Z, W, X, X) :
				// COST: 1.1+
				CASE_MATCH(Z, W, Y, X) :
				// COST: 1.1+
				CASE_MATCH(Y, Y, X, Y) :
				// COST: 1.1+
				CASE_MATCH(Y, Z, X, Y) :
				// COST: 1.1+
				CASE_MATCH(Z, Z, X, Y) :
				// COST: 1.1+
				CASE_MATCH(W, Z, X, Y) :
				// COST: 1.1+
				CASE_MATCH(W, W, X, Y) :
				// COST: 1.1+
				CASE_MATCH(Z, W, Y, Y) :
				// COST: 1.1+
				CASE_MATCH(Z, W, Y, Z) :
				// COST: 1.1+
				CASE_MATCH(Z, W, Z, Z) :

				// COST: 2.1
				CASE_MATCH(W, X, X, X) :
				// COST: 2.1
				CASE_MATCH(Y, Z, X, X) :
				// COST: 2.1
				CASE_MATCH(Z, Z, X, X) :
				// COST: 2.1
				CASE_MATCH(W, Z, X, X) :
				// COST: 2.1
				CASE_MATCH(W, W, X, X) :
				// COST: 2.1
				CASE_MATCH(W, X, Y, X) :
				// COST: 2.1
				CASE_MATCH(Y, Y, Y, X) :
				// COST: 2.1
				CASE_MATCH(Y, Z, Y, X) :
				// COST: 2.1
				CASE_MATCH(Z, Z, Y, X) :
				// COST: 2.1
				CASE_MATCH(W, Z, Y, X) :
				// COST: 2.1
				CASE_MATCH(W, W, Y, X) :
				// COST: 2.1
				CASE_MATCH(X, X, W, X) :
				// COST: 2.1
				CASE_MATCH(Y, X, W, X) :
				// COST: 2.1
				CASE_MATCH(Y, Y, W, X) :
				// COST: 2.1
				CASE_MATCH(Z, Z, W, X) :
				// COST: 2.1
				CASE_MATCH(W, Z, W, X) :
				// COST: 2.1
				CASE_MATCH(W, W, W, X) :
				// COST: 2.1
				CASE_MATCH(W, X, Y, Y) :
				// COST: 2.1
				CASE_MATCH(Y, Z, Y, Y) :
				// COST: 2.1
				CASE_MATCH(Z, Z, Y, Y) :
				// COST: 2.1
				CASE_MATCH(W, Z, Y, Y) :
				// COST: 2.1
				CASE_MATCH(W, W, Y, Y) :
				// COST: 2.1
				CASE_MATCH(X, X, Y, Z) :
				// COST: 2.1
				CASE_MATCH(Y, X, Y, Z) :
				// COST: 2.1
				CASE_MATCH(Y, Y, Y, Z) :
				// COST: 2.1
				CASE_MATCH(Z, Z, Y, Z) :
				// COST: 2.1
				CASE_MATCH(W, Z, Y, Z) :
				// COST: 2.1
				CASE_MATCH(W, W, Y, Z) :
				// COST: 2.1
				CASE_MATCH(Y, X, Z, Z) :
				// COST: 2.1
				CASE_MATCH(W, X, Z, Z) :
				// COST: 2.1
				CASE_MATCH(Y, Z, Z, Z) :
				// COST: 2.1
				CASE_MATCH(W, Z, Z, Z) :
				// COST: 2.1
				CASE_MATCH(X, W, Z, Z) :
				// COST: 2.1
				CASE_MATCH(X, X, W, Z) :
				// COST: 2.1
				CASE_MATCH(W, X, W, Z) :
				// COST: 2.1
				CASE_MATCH(Y, Y, W, Z) :
				// COST: 2.1
				CASE_MATCH(Y, Z, W, Z) :
				// COST: 2.1
				CASE_MATCH(W, W, W, Z) :
				// COST: 2.1
				CASE_MATCH(X, X, W, W) :
				// COST: 2.1
				CASE_MATCH(Y, X, W, W) :
				// COST: 2.1
				CASE_MATCH(W, X, W, W) :
				// COST: 2.1
				CASE_MATCH(Y, Z, W, W) :

				// COST: 2.2
				CASE_MATCH(X, X, Y, Y) :
				// COST: 2.2
				CASE_MATCH(Z, Z, W, W) :
				// COST: 2.2
				CASE_MATCH(Z, X, W, Y) :
				// COST: 2.2
				CASE_MATCH(X, Z, X, Z) :
				// COST: 2.2
				CASE_MATCH(Y, W, Y, W) :
				// COST: 2.2
				CASE_MATCH(X, Z, X, X) :
				// COST: 2.2
				CASE_MATCH(Y, W, X, X) :
				// COST: 2.2
				CASE_MATCH(X, X, Y, X) :
				// COST: 2.2
				CASE_MATCH(X, Y, Y, Y) :
				// COST: 2.2
				CASE_MATCH(Z, X, W, X) :
				// COST: 2.2
				CASE_MATCH(X, Z, Y, Y) :
				// COST: 2.2
				CASE_MATCH(Y, W, Y, Y) :
				// COST: 2.2
				CASE_MATCH(X, Z, Y, W) :
				// COST: 2.2
				CASE_MATCH(Y, W, X, Z) :
				// COST: 2.2
				CASE_MATCH(X, Y, Y, Z) :
				// COST: 2.2
				CASE_MATCH(X, W, Y, X) :
				// COST: 2.2
				CASE_MATCH(Z, Y, W, Z) :
				// COST: 2.2
				CASE_MATCH(X, Y, Z, X) :
				// COST: 2.2
				CASE_MATCH(Z, W, W, X) :
				// COST: 2.2
				CASE_MATCH(W, Y, Z, W) :
				// COST: 2.2
				CASE_MATCH(Z, X, Y, W) :
				// COST: 2.2
				CASE_MATCH(Y, W, Z, X) :
				// COST: 2.2
				CASE_MATCH(X, Z, W, Y) :
				// COST: 2.2
				CASE_MATCH(X, X, X, Y) :
				// COST: 2.2
				CASE_MATCH(W, Y, Y, Y) :
				// COST: 2.2
				CASE_MATCH(X, W, Z, X) :
				// COST: 2.2
				CASE_MATCH(Y, Z, W, Y) :
				// COST: 2.2
				CASE_MATCH(W, Y, X, W) :
				// COST: 2.2
				CASE_MATCH(X, W, Z, Y) :
				// COST: 2.2
				CASE_MATCH(Z, Y, X, W) :
				// COST: 2.2
				CASE_MATCH(X, X, Z, Z) :
				// COST: 2.2
				CASE_MATCH(Y, Y, W, W) :
				// COST: 2.2
				CASE_MATCH(X, X, Z, X) :
				// COST: 2.2
				CASE_MATCH(Y, Y, W, Y) :
				// COST: 2.2
				CASE_MATCH(Y, X, X, X) :
				// COST: 2.2
				CASE_MATCH(Y, X, Y, X) :
				// COST: 2.2
				CASE_MATCH(X, W, W, X) :
				// COST: 2.2
				CASE_MATCH(W, X, Z, Y) :
				// COST: 2.2
				CASE_MATCH(X, Y, Z, Y) :
				// COST: 2.2
				CASE_MATCH(Y, Z, Z, Y) :
				// COST: 2.2
				CASE_MATCH(Z, Y, W, Y) :
				// COST: 2.2
				CASE_MATCH(Z, Y, Y, Z) :
				// COST: 2.2
				CASE_MATCH(Y, Z, Y, Z) :
				// COST: 2.2
				CASE_MATCH(W, Z, W, Z) :
				// COST: 2.2
				CASE_MATCH(X, Y, X, W) :
				// COST: 2.2
				CASE_MATCH(X, Y, Y, W) :
				// COST: 2.2
				CASE_MATCH(W, X, W, X) :
				// COST: 2.2
				CASE_MATCH(Y, Z, Y, W) :
				// COST: 2.2
				CASE_MATCH(Z, Y, Z, W) :
				// COST: 2.2
				CASE_MATCH(X, Z, Z, W) :
				// COST: 2.2
				CASE_MATCH(X, W, Z, W) :

				// COST: 2.2+
				CASE_MATCH(X, Z, W, W) :
				// COST: 2.2+
				CASE_MATCH(X, W, Y, W) :
				// COST: 2.2+
				CASE_MATCH(X, Z, Y, Z) :
				// COST: 2.2+
				CASE_MATCH(Z, Z, W, Z) :
				// COST: 2.2+
				CASE_MATCH(X, Z, Z, Z) :
				// COST: 2.2+
				CASE_MATCH(Z, Z, Z, X) :
				// COST: 2.2+
				CASE_MATCH(Z, W, W, W) :
				// COST: 2.2+
				CASE_MATCH(Z, Y, X, Y) :
				// COST: 2.2+
				CASE_MATCH(X, Z, X, Y) :
				// COST: 2.2+
				CASE_MATCH(X, W, X, Y) :
				// COST: 2.2+
				CASE_MATCH(Z, W, Z, Y) :
				// COST: 2.2+
				CASE_MATCH(Z, W, X, W) :
				// COST: 2.2+
				CASE_MATCH(Z, W, Y, W) :
				
#			endif

				// COST: 2.2+
				CASE_MATCH(Y, Z, X, W) :

#			if defined(__AARCH64_SIMD__) || defined(META_PEEPHOLE)

				// COST: 3.2
				CASE_MATCH(Z, Y, X, X) :
				// COST: 3.2
				CASE_MATCH(X, W, X, X) :
				// COST: 3.2
				CASE_MATCH(Z, Y, Y, X) :
				// COST: 3.2
				CASE_MATCH(X, Z, Y, X) :
				// COST: 3.2
				CASE_MATCH(Y, W, Y, X) :
				// COST: 3.2
				CASE_MATCH(Z, Y, W, X) :
				// COST: 3.2
				CASE_MATCH(Z, Y, Y, Y) :
				// COST: 3.2
				CASE_MATCH(X, W, Y, Y) :
				// COST: 3.2
				CASE_MATCH(X, X, Z, Y) :
				// COST: 3.2
				CASE_MATCH(Y, Y, Z, Y) :
				// COST: 3.2
				CASE_MATCH(Z, Z, Z, Y) :
				// COST: 3.2
				CASE_MATCH(W, Z, Z, Y) :
				// COST: 3.2
				CASE_MATCH(W, W, Z, Y) :
				// COST: 3.2
				CASE_MATCH(X, X, X, Z) :
				// COST: 3.2
				CASE_MATCH(Y, X, X, Z) :
				// COST: 3.2
				CASE_MATCH(Y, Y, X, Z) :
				// COST: 3.2
				CASE_MATCH(Z, Z, X, Z) :
				// COST: 3.2
				CASE_MATCH(W, Z, X, Z) :
				// COST: 3.2
				CASE_MATCH(W, W, X, Z) :
				// COST: 3.2
				CASE_MATCH(Z, Y, Z, Z) :
				// COST: 3.2
				CASE_MATCH(Y, W, Z, Z) :
				// COST: 3.2
				CASE_MATCH(X, Z, W, Z) :
				// COST: 3.2
				CASE_MATCH(X, W, W, Z) :
				// COST: 3.2
				CASE_MATCH(Y, W, W, Z) :
				// COST: 3.2
				CASE_MATCH(X, X, X, W) :
				// COST: 3.2
				CASE_MATCH(Y, X, X, W) :
				// COST: 3.2
				CASE_MATCH(Y, Y, X, W) :
				// COST: 3.2
				CASE_MATCH(Z, Z, X, W) :
				// COST: 3.2
				CASE_MATCH(W, W, X, W) :
				// COST: 3.2
				CASE_MATCH(X, X, Y, W) :
				// COST: 3.2
				CASE_MATCH(Y, X, Y, W) :
				// COST: 3.2
				CASE_MATCH(Y, Y, Y, W) :
				// COST: 3.2
				CASE_MATCH(Z, Z, Y, W) :
				// COST: 3.2
				CASE_MATCH(W, Z, Y, W) :
				// COST: 3.2
				CASE_MATCH(W, W, Y, W) :
				// COST: 3.2
				CASE_MATCH(Z, Y, W, W) :
				// COST: 3.2
				CASE_MATCH(X, W, W, W) :
				// COST: 3.2
				CASE_MATCH(Y, W, W, W) :

				// COST: 3.3
				CASE_MATCH(Y, Y, X, X) :
				// COST: 3.3
				CASE_MATCH(Y, Y, Z, X) :
				// COST: 3.3
				CASE_MATCH(W, Y, Z, X) :
				// COST: 3.3
				CASE_MATCH(X, Z, Z, X) :
				// COST: 3.3
				CASE_MATCH(Y, X, X, Y) :
				// COST: 3.3
				CASE_MATCH(W, X, X, Y) :
				// COST: 3.3
				CASE_MATCH(Y, X, Y, Y) :
				// COST: 3.3
				CASE_MATCH(Y, X, Z, Y) :
				// COST: 3.3
				CASE_MATCH(Z, Y, Z, Y) :
				// COST: 3.3
				CASE_MATCH(X, X, W, Y) :
				// COST: 3.3
				CASE_MATCH(X, Y, W, Y) :
				// COST: 3.3
				CASE_MATCH(Z, Z, W, Y) :
				// COST: 3.3
				CASE_MATCH(W, Y, Y, Z) :
				// COST: 3.3
				CASE_MATCH(X, W, W, Y) :
				// COST: 3.3
				CASE_MATCH(Z, X, X, W) :
				// COST: 3.3
				CASE_MATCH(W, Z, W, Y) :
				// COST: 3.3
				CASE_MATCH(X, Z, X, W) :
				// COST: 3.3
				CASE_MATCH(Y, W, W, Y) :
				// COST: 3.3
				CASE_MATCH(Z, X, X, Z) :
				// COST: 3.3
				CASE_MATCH(X, Y, X, Z) :
				// COST: 3.3
				CASE_MATCH(W, Y, X, Z) :
				// COST: 3.3
				CASE_MATCH(X, W, X, Z) :
				// COST: 3.3
				CASE_MATCH(Z, X, Y, Z) :
				// COST: 3.3
				CASE_MATCH(X, W, Y, Z) :
				// COST: 3.3
				CASE_MATCH(Y, Y, Z, Z) :
				// COST: 3.3
				CASE_MATCH(W, Y, Z, Z) :
				// COST: 3.3
				CASE_MATCH(W, W, Z, Z) :
				// COST: 3.3
				CASE_MATCH(Z, W, W, Z) :
				// COST: 3.3
				CASE_MATCH(W, X, X, W) :
				// COST: 3.3
				CASE_MATCH(W, Z, X, W) :
				// COST: 3.3
				CASE_MATCH(X, W, X, W) :
				// COST: 3.3
				CASE_MATCH(W, Y, Y, W) :
				// COST: 3.3
				CASE_MATCH(Z, X, Z, W) :
				// COST: 3.3
				CASE_MATCH(Y, Z, Z, W) :
				// COST: 3.3
				CASE_MATCH(Z, Z, Z, W) :
				// COST: 3.3
				CASE_MATCH(W, Z, Z, W) :
				// COST: 3.3
				CASE_MATCH(Y, W, Z, W) :
				// COST: 3.3
				CASE_MATCH(Z, X, W, W) :
				// COST: 3.3
				CASE_MATCH(W, Z, W, W) :

				// COST: 3.3+
				CASE_MATCH(Z, W, Z, X) :
				// COST: 3.3+
				CASE_MATCH(W, Y, X, Y) :
				// COST: 3.3+
				CASE_MATCH(Y, W, X, Y) :
				// COST: 3.3+
				CASE_MATCH(Z, X, X, Y) :
				// COST: 3.3+
				CASE_MATCH(Z, W, W, Y) :
				// COST: 3.3+
				CASE_MATCH(Z, W, X, Z) :

				// COST: 4.2
				CASE_MATCH(Z, Y, X, Z) :

				// COST: 4.3
				CASE_MATCH(Z, X, X, X) :
				// COST: 4.3
				CASE_MATCH(W, Y, X, X) :
				// COST: 4.3
				CASE_MATCH(Z, X, Y, X) :
				// COST: 4.3
				CASE_MATCH(W, Y, Y, X) :
				// COST: 4.3
				CASE_MATCH(Y, X, Z, X) :
				// COST: 4.3
				CASE_MATCH(W, Z, Z, X) :
				// COST: 4.3
				CASE_MATCH(W, W, Z, X) :
				// COST: 4.3
				CASE_MATCH(X, Z, W, X) :
				// COST: 4.3
				CASE_MATCH(Y, W, W, X) :
				// COST: 4.3
				CASE_MATCH(Z, X, Y, Y) :
				// COST: 4.3
				CASE_MATCH(X, Z, Z, Y) :
				// COST: 4.3
				CASE_MATCH(Y, X, W, Y) :
				// COST: 4.3
				CASE_MATCH(W, W, W, Y) :
				// COST: 4.3
				CASE_MATCH(W, X, X, Z) :
				// COST: 4.3
				CASE_MATCH(Y, W, Y, Z) :
				// COST: 4.3
				CASE_MATCH(Z, X, Z, Z) :
				// COST: 4.3
				CASE_MATCH(Z, X, W, Z) :
				// COST: 4.3
				CASE_MATCH(W, Y, W, Z) :
				// COST: 4.3
				CASE_MATCH(W, X, Y, W) :
				// COST: 4.3
				CASE_MATCH(W, Y, W, W) :

				// COST: 4.4
				CASE_MATCH(Z, X, Z, X) :
				// COST: 4.4
				CASE_MATCH(W, X, Z, X) :
				// COST: 4.4
				CASE_MATCH(Z, Y, Z, X) :
				// COST: 4.4
				CASE_MATCH(W, Y, W, X) :
				// COST: 4.4
				CASE_MATCH(W, X, W, Y) :
				// COST: 4.4
				CASE_MATCH(W, Y, W, Y) :
				// COST: 4.4
				CASE_MATCH(Y, W, X, W) :
				// COST: 4.4
				CASE_MATCH(Y, Z, X, Z) :

				// COST: 5.3
				CASE_MATCH(Y, W, Z, Y) :
				// COST: 5.3
				CASE_MATCH(Z, Y, Y, W) :

				// COST: 5.4
				CASE_MATCH(Y, Z, Z, X) :

				// COST: 5.5
				CASE_MATCH(Z, X, Z, Y) :

				// COST: 6.4
				CASE_MATCH(W, Y, Z, Y) :
				
#			endif

				CASE_DEFAULT
			};

			META_COST({ COST = ((int)CASE >= (int)CASE_XYXY) + ((int)CASE >= (int)CASE_WXXX) + ((int)CASE >= (int)CASE_ZYXX) + ((int)CASE >= (int)CASE_ZYXZ) + ((int)CASE >= (int)CASE_YWZY) + ((int)CASE >= (int)CASE_WYZY) + 2*((int)CASE >= (int)CASE_DEFAULT) })
			
			template_decl(struct load, int op)
#		if !defined(__AARCH64_SIMD__)
			{
				enum
				{
					DEF = COMP(SWZ, FFS(SWZ)),
					EXT = SWZ | (SWZ(DEF, DEF, DEF, DEF) & ~MSK),
					X = COMP(EXT, 0) - COMP_X,
					Y = COMP(EXT, 1) - COMP_X,
					Z = COMP(EXT, 2) - COMP_X,
					W = COMP(EXT, 3) - COMP_X
				};
				
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					uint8x8x2_t q;
					q.val[0] = vreinterpret_u8_f32(vget_low_f32(p));
					q.val[1] = vreinterpret_u8_f32(vget_high_f32(p));
					return vreinterpretq_f32_u8(vcombine_u8(
						vtbl2_u8(q, (const uint8x8_t) { 4*X, 4*X+1, 4*X+2, 4*X+3, 4*Y, 4*Y+1, 4*Y+2, 4*Y+3 }),
						vtbl2_u8(q, (const uint8x8_t) { 4*Z, 4*Z+1, 4*Z+2, 4*Z+3, 4*W, 4*W+1, 4*W+2, 4*W+3 })
					));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					uint8x8x2_t q;
					q.val[0] = vreinterpret_u8_s32(vget_low_s32(p));
					q.val[1] = vreinterpret_u8_s32(vget_high_s32(p));
					return vreinterpretq_s32_u8(vcombine_u8(
						vtbl2_u8(q, (const uint8x8_t) { 4*X, 4*X+1, 4*X+2, 4*X+3, 4*Y, 4*Y+1, 4*Y+2, 4*Y+3 }),
						vtbl2_u8(q, (const uint8x8_t) { 4*Z, 4*Z+1, 4*Z+2, 4*Z+3, 4*W, 4*W+1, 4*W+2, 4*W+3 })
					));
				}
			}
#		endif
			;
			
			// COST: 0
			template_spec(struct load, CASE_XYZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return p;
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return p;
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_XYXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vget_low_s32(p));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_ZWZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vget_high_s32(p));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_XXXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vdupq_lane_f32(vget_low_f32(p), 0);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vdupq_lane_s32(vget_low_s32(p), 0);
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_YYYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vdupq_lane_f32(vget_low_f32(p), 1);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vdupq_lane_s32(vget_low_s32(p), 1);
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_ZZZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vdupq_lane_f32(vget_high_f32(p), 0);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vdupq_lane_s32(vget_high_s32(p), 0);
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_WWWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vdupq_lane_f32(vget_high_f32(p), 1);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vdupq_lane_s32(vget_high_s32(p), 1);
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_YXWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vrev64q_f32(p);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vrev64q_s32(p);
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_YZWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, p, 1);
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_ZWXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, p, 2);
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_WXYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, p, 3);
				}
			};
			
#		if defined(__AARCH64_SIMD__) || defined(META_PEEPHOLE)

			// COST: 2.2
			template_spec(struct load, CASE_XXYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, p).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, p).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZZWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, p).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, p).val[1];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZXWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_high_f32(p), vget_low_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_high_s32(p), vget_low_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XZXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, p).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, p).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YWYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, p).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, p).val[1];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XZXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vdupq_lane_f32(vget_low_f32(p), 0)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vdupq_lane_s32(vget_low_s32(p), 0)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YWXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vdupq_lane_f32(vget_low_f32(p), 0)).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vdupq_lane_s32(vget_low_s32(p), 0)).val[1];
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_XZWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vdupq_lane_f32(vget_high_f32(p), 1)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vdupq_lane_s32(vget_high_s32(p), 1)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XXYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vdupq_lane_f32(vget_low_f32(p), 0)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vdupq_lane_s32(vget_low_s32(p), 0)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XYYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vdupq_lane_f32(vget_low_f32(p), 1)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vdupq_lane_s32(vget_low_s32(p), 1)).val[0];
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_XWYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vdupq_lane_f32(vget_high_f32(p), 1)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vdupq_lane_s32(vget_high_s32(p), 1)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZXWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vdupq_lane_f32(vget_low_f32(p), 0)).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vdupq_lane_s32(vget_low_s32(p), 0)).val[1];
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_XZYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vdupq_lane_f32(vget_high_f32(p), 0)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vdupq_lane_s32(vget_high_s32(p), 0)).val[0];
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_ZZWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vdupq_lane_f32(vget_high_f32(p), 0)).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vdupq_lane_s32(vget_high_s32(p), 0)).val[1];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XZYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vdupq_lane_f32(vget_low_f32(p), 1)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vdupq_lane_s32(vget_low_s32(p), 1)).val[0];
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_XZZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vdupq_lane_f32(vget_high_f32(p), 0)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vdupq_lane_s32(vget_high_s32(p), 0)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YWYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vdupq_lane_f32(vget_low_f32(p), 1)).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vdupq_lane_s32(vget_low_s32(p), 1)).val[1];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XZYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vrev64q_f32(p)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vrev64q_s32(p)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YWXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vrev64q_f32(p)).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vrev64q_s32(p)).val[1];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XYYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1))).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vextq_s32(p, p, 1)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XWYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3))).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vextq_s32(p, p, 3)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZYWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3))).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vextq_s32(p, p, 3)).val[1];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XYZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3), vreinterpretq_s32_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(vextq_s32(p, p, 3), p, 1);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZWWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3), 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, vextq_s32(p, p, 3), 2);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_WYZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1), 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, vextq_s32(p, p, 1), 3);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZXYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1)), p).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(vextq_s32(p, p, 1), p).val[1];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YWZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1))).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vextq_s32(p, p, 1)).val[1];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XZWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vuzpq_f32(p, vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3))).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vuzpq_s32(p, vextq_s32(p, p, 3)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XXXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(vdupq_lane_f32(vget_low_f32(p), 0)), vreinterpretq_s32_f32(p), 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(vdupq_lane_s32(vget_low_s32(p), 0), p, 2);
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_ZZZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(vdupq_lane_f32(vget_high_f32(p), 0)), vreinterpretq_s32_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(vdupq_lane_s32(vget_high_s32(p), 0), p, 1);
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_ZWWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(vdupq_lane_f32(vget_high_f32(p), 1)), 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, vdupq_lane_s32(vget_high_s32(p), 1), 2);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_WYYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(vdupq_lane_f32(vget_low_f32(p), 1)), 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, vdupq_lane_s32(vget_low_s32(p), 1), 3);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XWZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(vrev64q_f32(p)), vreinterpretq_s32_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(vrev64q_s32(p), p, 1);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YZWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(vrev64q_f32(p)), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, vrev64q_s32(p), 1);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_WYXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(vrev64q_f32(p)), 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, vrev64q_s32(p), 3);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XWZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vreinterpretq_s32_f32(vrev64q_f32(p));
					return vreinterpretq_f32_s32(vextq_s32(q, q, 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vrev64q_s32(p);
					return vextq_s32(q, q, 1);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZYXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vreinterpretq_s32_f32(vrev64q_f32(p));
					return vreinterpretq_f32_s32(vextq_s32(q, q, 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vrev64q_s32(p);
					return vextq_s32(q, q, 3);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XXZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vtrnq_f32(p, p).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vtrnq_s32(p, p).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YYWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vtrnq_f32(p, p).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vtrnq_s32(p, p).val[1];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XXZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vtrnq_f32(p, vdupq_lane_f32(vget_low_f32(p), 0)).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vtrnq_s32(p, vdupq_lane_s32(vget_low_s32(p), 0)).val[0];
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YYWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vtrnq_f32(p, vdupq_lane_f32(vget_low_f32(p), 1)).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vtrnq_s32(p, vdupq_lane_s32(vget_low_s32(p), 1)).val[1];
				}
			};
			
#		endif

			// COST: 2.2+
			template_spec(struct load, CASE_YZXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vrev64_f32(vget_low_f32(p)), vget_high_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vrev64_s32(vget_low_s32(p)), vget_high_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

#		if defined(__AARCH64_SIMD__) || defined(META_PEEPHOLE)

			// COST: 2.2
			template_spec(struct load, CASE_YXXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_ZXXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WXXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_XYXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_YYXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(p), vget_low_f32(p));
					int32x4_t i = vreinterpretq_s32_f32(vcombine_f32(q.val[0], q.val[1]));
					return vreinterpretq_f32_s32(vextq_s32(i, i, 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(p), vget_low_s32(p));
					int32x4_t i = vcombine_s32(q.val[0], q.val[1]);
					return vextq_s32(i, i, 2);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_ZYXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_WYXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YZXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_ZZXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WZXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XWXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_ZWXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WWXX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vdup_lane_f32(vget_low_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vdup_lane_s32(vget_low_s32(p), 0));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YXYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vrev64_f32(vget_low_f32(p));
					return vcombine_f32(q, q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vrev64_s32(vget_low_s32(p));
					return vcombine_s32(q, q);
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_ZXYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WXYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_XYYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YYYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_ZYYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_WYYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XZYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0], vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0], vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YZYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_ZZYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WZYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					// vswpq_f32(vrev64q_f32(p))
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YWYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1], vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1], vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_ZWYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WWYX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vrev64_f32(vget_low_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vrev64_s32(vget_low_s32(p)));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_YXZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]));
				}
			};

			// COST: 4.4
			template_spec(struct load, CASE_ZXZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
					return vcombine_f32(q, q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
					return vcombine_s32(q, q);
				}
			};

			// COST: 4.4
			template_spec(struct load, CASE_WXZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(vuzpq_f32(p, p).val[0]), 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, vuzpq_s32(p, p).val[0], 3);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_YYZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vzipq_s32(vreinterpretq_s32_f32(p), vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1)).val[0];
					return vreinterpretq_f32_s32(vextq_s32(q, q, 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vzipq_s32(p, vextq_s32(p, p, 1)).val[0];
					return vextq_s32(q, q, 1);
				}
			};

			// COST: 4.4
			template_spec(struct load, CASE_ZYZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vreinterpretq_s32_f32(vzipq_f32(p, vdupq_lane_f32(vget_high_f32(p), 0)).val[0]);
					return vreinterpretq_f32_s32(vextq_s32(q, vreinterpretq_s32_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vzipq_s32(p, vdupq_lane_s32(vget_high_s32(p), 0)).val[0];
					return vextq_s32(q, p, 1);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WYZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vzipq_s32(vreinterpretq_s32_f32(p), vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 2)).val[1];
					return vreinterpretq_f32_s32(vextq_s32(q, q, 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vzipq_s32(p, vextq_s32(p, p, 2)).val[1];
					return vextq_s32(q, q, 2);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_XZZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(p), vget_high_f32(p));
					return vcombine_f32(q.val[0], vrev64_f32(q.val[0]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(p), vget_high_s32(p));
					return vcombine_s32(q.val[0], vrev64_s32(q.val[0]));
				}
			};

			// COST: 5.4
			template_spec(struct load, CASE_YZZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)), vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1), vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_WZZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]));
				}
			};

			// COST: 3.3+
			template_spec(struct load, CASE_ZWZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_WWZX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_XXWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 0), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 0), vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YXWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_WXWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1));
					return vcombine_f32(q, q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vext_s32(vget_high_s32(p), vget_low_s32(p), 1);
					return vcombine_s32(q, q);
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_XYWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YYWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_ZYWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 4.4
			template_spec(struct load, CASE_WYWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x4_t q = vuzpq_f32(p, p).val[1];
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(q), vreinterpretq_s32_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vuzpq_s32(p, p).val[1];
					return vextq_s32(q, p, 1);
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_XZWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0], vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0], vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_ZZWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WZWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XWWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1));
					return vcombine_f32(vrev64_f32(q), q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vext_s32(vget_high_s32(p), vget_low_s32(p), 1);
					return vcombine_s32(vrev64_s32(q), q);
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_YWWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1], vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1], vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WWWX)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_YXXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(p), vget_low_f32(p));
					int32x4_t i = vcombine_s32(vreinterpret_s32_f32(q.val[0]), vreinterpret_s32_f32(q.val[1]));
					return vreinterpretq_f32_s32(vextq_s32(i, i, 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(p), vget_low_s32(p));
					int32x4_t i = vcombine_s32(q.val[0], q.val[1]);
					return vextq_s32(i, i, 3);
				}
			};

			// COST: 3.3+
			template_spec(struct load, CASE_ZXXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]), vget_low_s32(p));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WXXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3))), vget_low_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(vextq_s32(p, p, 3)), vget_low_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_YYXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vget_low_s32(p));
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_ZYXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vget_low_s32(p));
				}
			};

			// COST: 3.3+
			template_spec(struct load, CASE_WYXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]), vget_low_s32(p));
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_XZXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0], vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0], vget_low_s32(p));
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_YZXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1), vget_low_s32(p));
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_ZZXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vget_low_s32(p));
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_WZXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vget_low_s32(p));
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_XWXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)), vget_low_s32(p));
				}
			};

			// COST: 3.3+
			template_spec(struct load, CASE_YWXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1], vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1], vget_low_s32(p));
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_WWXY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vget_low_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vget_low_s32(p));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_YXYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vdup_lane_f32(vget_low_f32(p), 1), vget_low_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vdup_lane_s32(vget_low_s32(p), 1), vget_low_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_ZXYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]), vdup_lane_f32(vget_low_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]), vdup_lane_s32(vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WXYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)), vdup_lane_f32(vget_low_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1), vdup_lane_s32(vget_low_s32(p), 1));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_ZYYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vdup_lane_f32(vget_low_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vdup_lane_s32(vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YZYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)), vdup_lane_f32(vget_low_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1), vdup_lane_s32(vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_ZZYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vdup_lane_f32(vget_low_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vdup_lane_s32(vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WZYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vdup_lane_f32(vget_low_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vdup_lane_s32(vget_low_s32(p), 1));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XWYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))), vdup_lane_f32(vget_low_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)), vdup_lane_s32(vget_low_s32(p), 1));
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_ZWYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vdup_lane_f32(vget_low_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vdup_lane_s32(vget_low_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WWYY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vdup_lane_f32(vget_low_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vdup_lane_s32(vget_low_s32(p), 1));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XXZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 0), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 0), vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_YXZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1))), vget_low_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(vextq_s32(p, p, 1)), vget_low_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 5.5
			template_spec(struct load, CASE_ZXZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vdup_lane_f32(vget_high_f32(p), 0), vget_low_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vdup_lane_s32(vget_high_s32(p), 0), vget_low_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_WXZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vrev64_f32(vget_high_f32(p)), vget_low_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vrev64_s32(vget_high_s32(p)), vget_low_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XYZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YYZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_ZYZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)));
					return vcombine_f32(q, q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1));
					return vcombine_s32(q, q);
				}
			};

			// COST: 6.4
			template_spec(struct load, CASE_WYZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]), vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_XZZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0], vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0], vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YZZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1));
					return vcombine_f32(q, vrev64_f32(q));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vext_s32(vget_low_s32(p), vget_high_s32(p), 1);
					return vcombine_s32(q, vrev64_s32(q));
				}
			};
			
			// COST: 3.2
			template_spec(struct load, CASE_ZZZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_WZZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 5.3
			template_spec(struct load, CASE_YWZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1], vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1], vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_ZWZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_WWZY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_XXWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vzipq_s32(vreinterpretq_s32_f32(p), vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3)).val[0];
					return vreinterpretq_f32_s32(vextq_s32(q, q, 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vzipq_s32(p, vextq_s32(p, p, 3)).val[0];
					return vextq_s32(q, q, 3);
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_YXWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]));
				}
			};

			// COST: 4.4
			template_spec(struct load, CASE_WXWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vzipq_s32(vreinterpretq_s32_f32(p), vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3)).val[0];
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), q, 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vzipq_s32(p, vextq_s32(p, p, 3)).val[0];
					return vextq_s32(p, q, 3);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_XYWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZYWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(p, vdupq_lane_f32(vget_low_f32(p), 1)).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(p, vdupq_lane_s32(vget_low_s32(p), 1)).val[1];
				}
			};

			// COST: 4.4
			template_spec(struct load, CASE_WYWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
					return vcombine_f32(q, q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
					return vcombine_s32(q, q);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_ZZWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vreinterpretq_s32_f32(vrev64q_f32(p)), r = vextq_s32(vreinterpretq_s32_f32(p), q, 2);
					return vreinterpretq_f32_s32(vextq_s32(q, r, 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vrev64q_s32(p), r = vextq_s32(p, q, 2);
					return vextq_s32(q, r, 3);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WYYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(vrev64q_f32(p)), 1);
					return vreinterpretq_f32_s32(vextq_s32(q, q, 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vextq_s32(p, vrev64q_s32(p), 1);
					return vextq_s32(q, q, 2);
				}
			};

			// COST: 4.4
			template_spec(struct load, CASE_YWXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vreinterpretq_s32_f32(vzipq_f32(p, vdupq_lane_f32(vget_high_f32(p), 1)).val[0]);
					return vreinterpretq_f32_s32(vextq_s32(q, q, 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vzipq_s32(p, vdupq_lane_s32(vget_high_s32(p), 1)).val[0];
					return vextq_s32(q, q, 2);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_XWWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(vrev64q_f32(p)), 3);
					return vreinterpretq_f32_s32(vextq_s32(q, q, 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vextq_s32(p, vrev64q_s32(p), 3);
					return vextq_s32(q, q, 2);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_ZXXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vextq_s32(vreinterpretq_s32_f32(vrev64q_f32(p)), vreinterpretq_s32_f32(p), 1);
					return vreinterpretq_f32_s32(vextq_s32(q, q, 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vextq_s32(vrev64q_s32(p), p, 1);
					return vextq_s32(q, q, 2);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WZWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(vrev64q_f32(p)), 2), 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(p, vextq_s32(p, vrev64q_s32(p), 2), 3);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_XZXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vreinterpretq_s32_f32(vzipq_f32(p, vdupq_lane_f32(vget_low_f32(p), 0)).val[1]);
					return vreinterpretq_f32_s32(vextq_s32(q, q, 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vzipq_s32(p, vdupq_lane_s32(vget_low_s32(p), 0)).val[1];
					return vextq_s32(q, q, 3);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_YWWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1];
					return vcombine_f32(q, vrev64_f32(q));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1];
					return vcombine_s32(q, vrev64_s32(q));
				}
			};

			// COST: 3.3+
			template_spec(struct load, CASE_ZWWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_WWWY)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XXXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 0), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 0), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YXXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_ZXXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0];
					return vcombine_f32(vrev64_f32(q), q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0];
					return vcombine_s32(vrev64_s32(q), q);
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_WXXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_XYXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YYXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 4.2
			template_spec(struct load, CASE_ZYXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WYXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(p), vget_high_f32(p));
					return vcombine_f32(vrev64_f32(q.val[1]), q.val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(p), vget_high_s32(p));
					return vcombine_s32(vrev64_s32(q.val[1]), q.val[0]);
				}
			};

			// COST: 4.4
			template_spec(struct load, CASE_YZXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vreinterpretq_s32_f32(vzipq_f32(p, vdupq_lane_f32(vget_high_f32(p), 0)).val[0]);
					return vreinterpretq_f32_s32(vextq_s32(q, q, 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vzipq_s32(p, vdupq_lane_s32(vget_high_s32(p), 0)).val[0];
					return vextq_s32(q, q, 2);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_ZZXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_WZXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_XWXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vreinterpretq_s32_f32(vzipq_f32(p, vdupq_lane_f32(vget_low_f32(p), 0)).val[1]);
					return vreinterpretq_f32_s32(vextq_s32(q, q, 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vzipq_s32(p, vdupq_lane_s32(vget_low_s32(p), 0)).val[1];
					return vextq_s32(q, q, 1);
				}
			};

			// COST: 3.3+
			template_spec(struct load, CASE_ZWXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_WWXZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]);
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_XXYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 0), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 0), vext_s32(vget_low_s32(p), vget_high_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YXYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vext_s32(vget_low_s32(p), vget_high_s32(p), 1));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_ZXYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(vdupq_lane_f32(vget_high_f32(p), 0)), vreinterpretq_s32_f32(p), 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vextq_s32(vdupq_lane_s32(vget_high_s32(p), 0), p, 3);
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YYYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vext_s32(vget_low_s32(p), vget_high_s32(p), 1));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZYYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1));
					return vcombine_f32(vrev64_f32(q), q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vext_s32(vget_low_s32(p), vget_high_s32(p), 1);
					return vcombine_s32(vrev64_s32(q), q);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YZYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1));
					return vcombine_f32(q, q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vext_s32(vget_low_s32(p), vget_high_s32(p), 1);
					return vcombine_s32(q, q);
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_ZZYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vext_s32(vget_low_s32(p), vget_high_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WZYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vext_s32(vget_low_s32(p), vget_high_s32(p), 1));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_XWYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(p), vrev64_f32(vget_high_f32(p)));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(p), vrev64_s32(vget_high_s32(p)));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_YWYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1], vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1], vext_s32(vget_low_s32(p), vget_high_s32(p), 1));
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_ZWYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vext_s32(vget_low_s32(p), vget_high_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WWYZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vext_s32(vget_low_s32(p), vget_high_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YXZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_ZXZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]), vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]), vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WXZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)), vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1), vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_XYZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_YYZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t t = vget_low_f32(vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1)));
					float32x2x2_t q = vtrn_f32(t, t);
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t t = vget_low_s32(vextq_s32(p, p, 1));
					int32x2x2_t q = vtrn_s32(t, t);
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_ZYZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WYZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vextq_s32(vreinterpretq_s32_f32(p), vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1), 2);
					return vreinterpretq_f32_s32(vextq_s32(q, q, 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vextq_s32(p, vextq_s32(p, p, 1), 2);
					return vextq_s32(q, q, 1);
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YZZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)), vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1), vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WZZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_XWZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))), vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)), vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YWZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1], vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1], vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 1.1+
			template_spec(struct load, CASE_ZWZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vdup_lane_f32(vget_high_f32(p), 0));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vdup_lane_s32(vget_high_s32(p), 0));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WWZZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_high_f32(p), vget_high_f32(p));
					int32x4_t t = vreinterpretq_s32_f32(vcombine_f32(q.val[0], q.val[1]));
					return vreinterpretq_f32_s32(vextq_s32(t, t, 2));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_high_s32(p), vget_high_s32(p));
					int32x4_t t = vcombine_s32(q.val[0], q.val[1]);
					return vextq_s32(t, t, 2);
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_XXWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 0), vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 0), vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_ZXWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]), vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]), vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WXWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)), vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1), vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_XYWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YYWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_WYWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]), vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]), vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XZWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0], vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0], vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YZWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)), vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1), vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_WZWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vrev64_f32(vget_high_f32(p));
					return vcombine_f32(q, q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vrev64_s32(vget_high_s32(p));
					return vcombine_s32(q, q);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XWWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))), vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)), vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YWWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1], vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1], vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_ZWWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_high_f32(p), vget_high_f32(p));
					int32x4_t t = vreinterpretq_s32_f32(vcombine_f32(q.val[0], q.val[1]));
					return vreinterpretq_f32_s32(vextq_s32(t, t, 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_high_s32(p), vget_high_s32(p));
					int32x4_t t = vcombine_s32(q.val[0], q.val[1]);
					return vextq_s32(t, t, 1);
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WWWZ)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vrev64_f32(vget_high_f32(p)));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vrev64_s32(vget_high_s32(p)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XXXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 0), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 0), vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YXXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WXXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t t = vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1));
					return vcombine_f32(t, vrev64_f32(t));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t t = vext_s32(vget_high_s32(p), vget_low_s32(p), 1);
					return vcombine_s32(t, vrev64_s32(t));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XYXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YYXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_ZZXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WZXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_high_f32(vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 1))), vget_high_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_high_s32(vextq_s32(p, p, 1)), vget_high_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_XWXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)));
					return vcombine_f32(q, q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1));
					return vcombine_s32(q, q);
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_ZWXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_WWXW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XXYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 0), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 0), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YXYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_WXYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XYYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YYYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 5.3
			template_spec(struct load, CASE_ZYYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WYYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2_t q = vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1];
					return vcombine_f32(vrev64_f32(q), q);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2_t q = vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1];
					return vcombine_s32(vrev64_s32(q), q);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_YZYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vzipq_f32(vdupq_lane_f32(vget_low_f32(p), 1), p).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vzipq_s32(vdupq_lane_s32(vget_low_s32(p), 1), p).val[1];
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_ZZYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 0), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 0), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_WZYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_high_f32(p)), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_high_s32(p)), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 2.2+
			template_spec(struct load, CASE_ZWYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_high_f32(p), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_high_s32(p), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_WWYW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]);
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_XXZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 0), vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 0), vget_high_s32(p));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_YXZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vget_high_s32(p));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_ZXZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0]), vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0]), vget_high_s32(p));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_WXZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)), vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1), vget_high_s32(p));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_YYZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 1), vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 1), vget_high_s32(p));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_ZYZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vget_high_s32(p));
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XZZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[0], vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[0], vget_high_s32(p));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_YZZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_high_f32(vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(p), 3))), vget_high_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_high_s32(vextq_s32(p, p, 3)), vget_high_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_ZZZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vdup_lane_f32(vget_high_f32(p), 0), vget_high_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vdup_lane_s32(vget_high_s32(p), 0), vget_high_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WZZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vget_high_f32(p), vget_high_f32(p));
					int32x4_t t = vreinterpretq_s32_f32(vcombine_f32(q.val[0], q.val[1]));
					return vreinterpretq_f32_s32(vextq_s32(t, t, 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vget_high_s32(p), vget_high_s32(p));
					int32x4_t t = vcombine_s32(q.val[0], q.val[1]);
					return vextq_s32(t, t, 3);
				}
			};

			// COST: 2.2
			template_spec(struct load, CASE_XWZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))), vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)), vget_high_s32(p));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_YWZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1], vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1], vget_high_s32(p));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_WWZW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_high_f32(p), 1), vget_high_f32(p));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_high_s32(p), 1), vget_high_s32(p));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_XXWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vdup_lane_f32(vget_low_f32(p), 0), vdup_lane_f32(vget_high_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vdup_lane_s32(vget_low_s32(p), 0), vdup_lane_s32(vget_high_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YXWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vget_low_f32(p)), vdup_lane_f32(vget_high_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vget_low_s32(p)), vdup_lane_s32(vget_high_s32(p), 1));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_ZXWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					int32x4_t q = vextq_s32(vreinterpretq_s32_f32(p), vreinterpretq_s32_f32(vrev64q_f32(p)), 3);
					return vzipq_f32(p, vreinterpretq_f32_s32(q)).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x4_t q = vextq_s32(p, vrev64q_s32(p), 3);
					return vzipq_s32(p, q).val[1];
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_WXWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1)), vdup_lane_f32(vget_high_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1), vdup_lane_s32(vget_high_s32(p), 1));
				}
			};

			// COST: 1.1
			template_spec(struct load, CASE_XYWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vget_low_f32(p), vdup_lane_f32(vget_high_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vget_low_s32(p), vdup_lane_s32(vget_high_s32(p), 1));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_ZYWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1))), vdup_lane_f32(vget_high_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1)), vdup_lane_s32(vget_high_s32(p), 1));
				}
			};

			// COST: 4.3
			template_spec(struct load, CASE_WYWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1]), vdup_lane_f32(vget_high_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1]), vdup_lane_s32(vget_high_s32(p), 1));
				}
			};

			// COST: 2.1
			template_spec(struct load, CASE_YZWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_low_f32(p)), vreinterpret_s32_f32(vget_high_f32(p)), 1)), vdup_lane_f32(vget_high_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vext_s32(vget_low_s32(p), vget_high_s32(p), 1), vdup_lane_s32(vget_high_s32(p), 1));
				}
			};

			// COST: 3.3
			template_spec(struct load, CASE_WZWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					float32x2x2_t q = vtrn_f32(vdup_lane_f32(vget_high_f32(p), 1), vget_high_f32(p));
					return vcombine_f32(q.val[0], q.val[1]);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					int32x2x2_t q = vtrn_s32(vdup_lane_s32(vget_high_s32(p), 1), vget_high_s32(p));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_XWWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vrev64_f32(vreinterpret_f32_s32(vext_s32(vreinterpret_s32_f32(vget_high_f32(p)), vreinterpret_s32_f32(vget_low_f32(p)), 1))), vdup_lane_f32(vget_high_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vrev64_s32(vext_s32(vget_high_s32(p), vget_low_s32(p), 1)), vdup_lane_s32(vget_high_s32(p), 1));
				}
			};

			// COST: 3.2
			template_spec(struct load, CASE_YWWW)
			{
				static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
				{
					return vcombine_f32(vtrn_f32(vget_low_f32(p), vget_high_f32(p)).val[1], vdup_lane_f32(vget_high_f32(p), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
				{
					return vcombine_s32(vtrn_s32(vget_low_s32(p), vget_high_s32(p)).val[1], vdup_lane_s32(vget_high_s32(p), 1));
				}
			};
			
#		endif

			static MATH_FORCEINLINE float32x4_t f(float32x4_t p)
			{
				return template_inst(load, CASE)::f(p);
			}
			
			static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
			{
				return template_inst(load, CASE)::f(p);
			}
		};

		enum
		{
			CLR,
			MOV,
			MOV_R,
			CMB_LL,
			CMB_HH,
			CMB_LH,
			CMB_LL_R,
			CMB_HH_R,
			CMB_LH_R,
			CMB_HL,
			CMB_HL_R,
			
			ZIP0,
			ZIP1,
			ZIP0_R,
			ZIP1_R,
			UZP0,
			UZP1,
			UZP0_R,
			UZP1_R,
			TRN0,
			TRN1,
			TRN0_R,
			TRN1_R,

			TRN_LH,
			TRN_HL,
			TRN_LL_R,
			TRN_HH_R,
			TRN_LH_R,
			TRN_HL_R,
			EXT1,
			EXT1_R,
			EXT3,
			EXT3_R,
			BSL,
			BSL_R,
			TBL,
			DEFAULT
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

				IGN = ~(L_MSK | R_MSK) & MSK_XYZW
			};
			
			enum
			{
				MERGE =
					(L_SWZ==0 && R_SWZ==0) ? CLR :
					L_SWZ==0 ? MOV_R :
					R_SWZ==0 ? MOV :
#			ifdef META_PEEPHOLE
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_Y, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_X, COMP_Y), R_MSK, IGN) ? CMB_LL :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_Y, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_Z, COMP_W), R_MSK, IGN) ? CMB_LH :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Z, COMP_W, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_X, COMP_Y), R_MSK, IGN) ? CMB_HL :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Z, COMP_W, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_Z, COMP_W), R_MSK, IGN) ? CMB_HH :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_Y, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_X, COMP_Y), L_MSK, IGN) ? CMB_LL_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_Y, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_Z, COMP_W), L_MSK, IGN) ? CMB_LH_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Z, COMP_W, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_X, COMP_Y), L_MSK, IGN) ? CMB_HL_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Z, COMP_W, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_Z, COMP_W), L_MSK, IGN) ? CMB_HH_R :

					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_N, COMP_Y, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Y), R_MSK, IGN) ? ZIP0 :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Z, COMP_N, COMP_W, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_Z, COMP_N, COMP_W), R_MSK, IGN) ? ZIP1 :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_Z, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_X, COMP_Z), R_MSK, IGN) ? UZP0 :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Y, COMP_W, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_Y, COMP_W), R_MSK, IGN) ? UZP1 :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_N, COMP_Z, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Z), R_MSK, IGN) ? TRN0 :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Y, COMP_N, COMP_W, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_Y, COMP_N, COMP_W), R_MSK, IGN) ? TRN1 :

					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_N, COMP_Y, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Y), L_MSK, IGN) ? ZIP0_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Z, COMP_N, COMP_W, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_Z, COMP_N, COMP_W), L_MSK, IGN) ? ZIP1_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_Z, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_X, COMP_Z), L_MSK, IGN) ? UZP0_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Y, COMP_W, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_Y, COMP_W), L_MSK, IGN) ? UZP1_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_N, COMP_Z, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Z), L_MSK, IGN) ? TRN0_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Y, COMP_N, COMP_W, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_Y, COMP_N, COMP_W), L_MSK, IGN) ? TRN1_R :

					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_X, COMP_N, COMP_Y, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_Z, COMP_N, COMP_W), R_MSK, IGN) ? TRN_LH :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Z, COMP_N, COMP_W, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Y), R_MSK, IGN) ? TRN_HL :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_N, COMP_Y, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Y), L_MSK, IGN) ? TRN_LL_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_X, COMP_N, COMP_Y, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_Z, COMP_N, COMP_W), L_MSK, IGN) ? TRN_LH_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Z, COMP_N, COMP_W, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_X, COMP_N, COMP_Y), L_MSK, IGN) ? TRN_HL_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Z, COMP_N, COMP_W, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_Z, COMP_N, COMP_W), L_MSK, IGN) ? TRN_HH_R :

					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_N, COMP_N, COMP_X), R_MSK, IGN) ? EXT1 :
					MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_W, COMP_N, COMP_N, COMP_N), L_MSK, IGN) && MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_N, COMP_X, COMP_Y, COMP_Z), R_MSK, IGN) ? EXT3 :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_Y, COMP_Z, COMP_W, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_N, COMP_N, COMP_X), L_MSK, IGN) ? EXT1_R :
					MATCH_SWIZ(R_SCL, R_SWZ, SWZ(COMP_W, COMP_N, COMP_N, COMP_N), R_MSK, IGN) && MATCH_SWIZ(L_SCL, L_SWZ, SWZ(COMP_N, COMP_X, COMP_Y, COMP_Z), L_MSK, IGN) ? EXT3_R :
#			endif
					(int)SWIZ<L_SWZ>::CASE != (int)CASE_DEFAULT && (int)SWIZ<R_SWZ>::CASE != (int)CASE_DEFAULT ? L_MSK < R_MSK ? BSL : BSL_R :
					TBL
			};
			
			enum
			{
				LLOAD =
					L_SWZ==0 ? MOV :
					L_SCL ? MOV :
					R_SWZ==0 ?
						L_SWZ==(SWZ_XYZW & L_MSK) ? MOV :
						DEFAULT :
					MERGE==int(TBL) ? MOV :
					(L_SCL || L_SWZ==(SWZ_XYZW & L_MSK)) ? MOV :
					DEFAULT
			};
			
			enum
			{
				RLOAD =
					R_SWZ==0 ? MOV :
					R_SCL ? MOV :
					L_SWZ==0 ?
						R_SWZ==(SWZ_XYZW & R_MSK) ? MOV :
						DEFAULT :
					MERGE==int(TBL) ? MOV :
					(R_SCL || R_SWZ==(SWZ_XYZW & R_MSK)) ? MOV :
					DEFAULT
			};

			template_decl(struct lload, unsigned op)
			{
				META_COST({ COST = SWIZ<L_SWZ>::COST })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a)
				{
					return SWIZ<L_SWZ>::f(a);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a)
				{
					return SWIZ<L_SWZ>::f(a);
				}
			};
			template_spec(struct lload, MOV)
			{
				META_COST({ COST = 0 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a)
				{
					return a;
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a)
				{
					return a;
				}
			};

			template_decl(struct rload, unsigned op)
			{
				META_COST({ COST = SWIZ<R_SWZ>::COST })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a)
				{
					return SWIZ<R_SWZ>::f(a);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a)
				{
					return SWIZ<R_SWZ>::f(a);
				}
			};
			template_spec(struct rload, MOV)
			{
				META_COST({ COST = 0 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a)
				{
					return a;
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a)
				{
					return a;
				}
			};

			template_decl(struct merge, int op)
			{
				META_COST({ COST = 0 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t)
				{
					return a;
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t)
				{
					return a;
				}
			};
			template_spec(struct merge, CLR)
			{
				META_COST({ COST = 1 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t, float32x4_t)
				{
					return vdupq_n_f32(0.f);
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t, int32x4_t)
				{
					return vdupq_n_s32(0.f);
				}
			};
			template_spec(struct merge, MOV)
			{
				META_COST({ COST = 0 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t)
				{
					return a;
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t)
				{
					return a;
				}
			};
			template_spec(struct merge, MOV_R)
			{
				META_COST({ COST = 0 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t, float32x4_t b)
				{
					return b;
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t, int32x4_t b)
				{
					return b;
				}
			};
			
			template_spec(struct merge, CMB_LL)
			{
				META_COST({ COST = 1 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vcombine_f32(vget_low_f32(a), vget_low_f32(b));
				}
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vcombine_s32(vget_low_s32(a), vget_low_s32(b));
				}
			};
			template_spec(struct merge, CMB_HH)
			{
				META_COST({ COST = 1 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vcombine_f32(vget_high_f32(a), vget_high_f32(b));
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vcombine_s32(vget_high_s32(a), vget_high_s32(b));
				}
			};
			template_spec(struct merge, CMB_LH)
			{
				META_COST({ COST = 1 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vcombine_f32(vget_low_f32(a), vget_high_f32(b));
				}
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vcombine_s32(vget_low_s32(a), vget_high_s32(b));
				}
			};
			template_spec(struct merge, CMB_HL)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vcombine_f32(vget_high_f32(a), vget_low_f32(b));
				}
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vcombine_s32(vget_high_s32(a), vget_low_s32(b));
				}
			};
			template_spec(struct merge, CMB_LL_R)
			{
				META_COST({ COST = 1 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vcombine_f32(vget_low_f32(b), vget_low_f32(a));
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vcombine_s32(vget_low_s32(b), vget_low_s32(a));
				}
			};
			template_spec(struct merge, CMB_HH_R)
			{
				META_COST({ COST = 1 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vcombine_f32(vget_high_f32(b), vget_high_f32(a));
				}
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vcombine_s32(vget_high_s32(b), vget_high_s32(a));
				}
			};
			template_spec(struct merge, CMB_LH_R)
			{
				META_COST({ COST = 1 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vcombine_f32(vget_low_f32(b), vget_high_f32(a));
				}
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vcombine_s32(vget_low_s32(b), vget_high_s32(a));
				}
			};
			template_spec(struct merge, CMB_HL_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vcombine_f32(vget_high_f32(b), vget_low_f32(a));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vcombine_s32(vget_high_s32(b), vget_low_s32(a));
				}
			};

			template_spec(struct merge, ZIP0)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vzipq_f32(a, b).val[0];
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vzipq_s32(a, b).val[0];
				}
			};
			template_spec(struct merge, ZIP1)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vzipq_f32(a, b).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vzipq_s32(a, b).val[1];
				}
			};

			template_spec(struct merge, ZIP0_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vzipq_f32(b, a).val[0];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vzipq_s32(b, a).val[0];
				}
			};
			template_spec(struct merge, ZIP1_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vzipq_f32(b, a).val[1];
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vzipq_s32(b, a).val[1];
				}
			};

			template_spec(struct merge, UZP0)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vuzpq_f32(a, b).val[0];
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vuzpq_s32(a, b).val[0];
				}
			};
			template_spec(struct merge, UZP1)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vuzpq_f32(a, b).val[1];
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vuzpq_s32(a, b).val[1];
				}
			};

			template_spec(struct merge, UZP0_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vuzpq_f32(b, a).val[0];
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vuzpq_s32(b, a).val[0];
				}
			};
			template_spec(struct merge, UZP1_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vuzpq_f32(b, a).val[1];
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vuzpq_s32(b, a).val[1];
				}
			};

			template_spec(struct merge, TRN0)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vtrnq_f32(a, b).val[0];
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vtrnq_s32(a, b).val[0];
				}
			};
			template_spec(struct merge, TRN1)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vtrnq_f32(a, b).val[1];
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vtrnq_s32(a, b).val[1];
				}
			};

			template_spec(struct merge, TRN0_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vtrnq_f32(b, a).val[0];
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vtrnq_s32(b, a).val[0];
				}
			};
			template_spec(struct merge, TRN1_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vtrnq_f32(b, a).val[1];
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vtrnq_s32(b, a).val[1];
				}
			};

			template_spec(struct merge, TRN_LH)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(a), vget_high_f32(b));
					return vcombine_f32(q.val[0], q.val[1]);
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(a), vget_high_s32(b));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};
			template_spec(struct merge, TRN_HL)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					float32x2x2_t q = vtrn_f32(vget_high_f32(a), vget_low_f32(b));
					return vcombine_f32(q.val[0], q.val[1]);
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					int32x2x2_t q = vtrn_s32(vget_high_s32(a), vget_low_s32(b));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			template_spec(struct merge, TRN_LL_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(b), vget_low_f32(a));
					return vcombine_f32(q.val[0], q.val[1]);
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(b), vget_low_s32(a));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};
			template_spec(struct merge, TRN_HH_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					float32x2x2_t q = vtrn_f32(vget_high_f32(b), vget_high_f32(a));
					return vcombine_f32(q.val[0], q.val[1]);
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					int32x2x2_t q = vtrn_s32(vget_high_s32(b), vget_high_s32(a));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};
			template_spec(struct merge, TRN_LH_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					float32x2x2_t q = vtrn_f32(vget_low_f32(b), vget_high_f32(a));
					return vcombine_f32(q.val[0], q.val[1]);
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					int32x2x2_t q = vtrn_s32(vget_low_s32(b), vget_high_s32(a));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};
			template_spec(struct merge, TRN_HL_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					float32x2x2_t q = vtrn_f32(vget_high_f32(b), vget_low_f32(a));
					return vcombine_f32(q.val[0], q.val[1]);
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					int32x2x2_t q = vtrn_s32(vget_high_s32(b), vget_low_s32(a));
					return vcombine_s32(q.val[0], q.val[1]);
				}
			};

			template_spec(struct merge, EXT1)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b), 1));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vextq_s32(a, b, 1);
				}
			};
			template_spec(struct merge, EXT3)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(a), vreinterpretq_s32_f32(b), 3));
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vextq_s32(a, b, 3);
				}
			};
			template_spec(struct merge, EXT1_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(b), vreinterpretq_s32_f32(a), 1));
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vextq_s32(b, a, 1);
				}
			};
			template_spec(struct merge, EXT3_R)
			{
				META_COST({ COST = 2 })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vreinterpretq_f32_s32(vextq_s32(vreinterpretq_s32_f32(b), vreinterpretq_s32_f32(a), 3));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vextq_s32(b, a, 3);
				}
			};

			template_spec(struct merge, BSL)
			{
				META_COST({ COST = 2 })

				enum
				{
					M0 = COMP(R_MSK, 0) != 0,
					M1 = COMP(R_MSK, 1) != 0,
					M2 = COMP(R_MSK, 2) != 0,
					M3 = COMP(R_MSK, 3) != 0
				};

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vbslq_f32((uint32x4_t) cv4i(-M0, -M1, -M2, -M3), b, a);
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vbslq_s32((uint32x4_t) cv4i(-M0, -M1, -M2, -M3), b, a);
				}
			};
			template_spec(struct merge, BSL_R)
			{
				META_COST({ COST = 2 })

				enum
				{
					M0 = COMP(L_MSK, 0) != 0,
					M1 = COMP(L_MSK, 1) != 0,
					M2 = COMP(L_MSK, 2) != 0,
					M3 = COMP(L_MSK, 3) != 0
				};

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vbslq_f32((uint32x4_t) cv4i(-M0, -M1, -M2, -M3), a, b);
				}

				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vbslq_s32((uint32x4_t) cv4i(-M0, -M1, -M2, -M3), a, b);
				}
			};

			template_spec(struct merge, TBL)
			{
#			if defined(__AARCH64_SIMD__)

				enum
				{
					L = L_SCL ? SWZ_ANY : L_SWZ,
					R = R_SCL ? SWZ_ANY : R_SWZ,
					M0 = COMP(R_MSK, 0) != 0,
					M1 = COMP(R_MSK, 1) != 0,
					M2 = COMP(R_MSK, 2) != 0,
					M3 = COMP(R_MSK, 3) != 0
				};

				META_COST({ COST = 2 + SWIZ<L>::COST + SWIZ<R>::COST })

				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					return vbslq_f32((uint32x4_t) cv4i(-M0, -M1, -M2, -M3), SWIZ<R>::f(b), SWIZ<L>::f(a));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					return vbslq_s32((uint32x4_t) cv4i(-M0, -M1, -M2, -M3), SWIZ<R>::f(b), SWIZ<L>::f(a));
				}
				
#			else

				META_COST({ COST = 8 })
				
				enum
				{
					DEF = COMP(R_SWZ, FFS(R_SWZ)),
					EXT = L_SWZ | R_SWZ | (SWZ(DEF, DEF, DEF, DEF) & ~(L_MSK|R_MSK)),
					X = COMP(EXT, 0) - COMP_X + (COMP(SEL, 0) ? 4 : 0),
					Y = COMP(EXT, 1) - COMP_X + (COMP(SEL, 0) ? 4 : 0),
					Z = COMP(EXT, 2) - COMP_X + (COMP(SEL, 0) ? 4 : 0),
					W = COMP(EXT, 3) - COMP_X + (COMP(SEL, 0) ? 4 : 0)
				};
				
				static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
				{
					union { uint8x8x4_t b8x4; float32x4x2_t f4x2; } u;
					u.f4x2.val[0] = a;
					u.f4x2.val[1] = b;
					return vreinterpretq_f32_u8(vcombine_u8(
						vtbl4_u8(u.b8x4, (const uint8x8_t) { 4*X, 4*X+1, 4*X+2, 4*X+3, 4*Y, 4*Y+1, 4*Y+2, 4*Y+3 }),
						vtbl4_u8(u.b8x4, (const uint8x8_t) { 4*Z, 4*Z+1, 4*Z+2, 4*Z+3, 4*W, 4*W+1, 4*W+2, 4*W+3 })
					));
				}
				
				static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
				{
					union { uint8x8x4_t b8x4; int32x4x2_t f4x2; } u;
					u.i4x2.val[0] = a;
					u.i4x2.val[1] = b;
					return vreinterpretq_s32_u8(vcombine_u8(
						vtbl4_u8(u.b8x4, (const uint8x8_t) { 4*X, 4*X+1, 4*X+2, 4*X+3, 4*Y, 4*Y+1, 4*Y+2, 4*Y+3 }),
						vtbl4_u8(u.b8x4, (const uint8x8_t) { 4*Z, 4*Z+1, 4*Z+2, 4*Z+3, 4*W, 4*W+1, 4*W+2, 4*W+3 })
					));
				}
				
#			endif
			};

			static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b)
			{
				float32x4_t l = template_inst(lload, LLOAD)::f(a);
				float32x4_t r = template_inst(rload, RLOAD)::f(b);
				float32x4_t d = template_inst(merge, MERGE)::f(l, r);
				return d;
			}

			static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
			{
				int32x4_t l = template_inst(lload, LLOAD)::f(a);
				int32x4_t r = template_inst(rload, RLOAD)::f(b);
				int32x4_t d = template_inst(merge, MERGE)::f(l, r);
				return d;
			}
		};

		template<int c> struct GET
		{
			static MATH_FORCEINLINE float f(float32x4_t p)
			{
				return vgetq_lane_f32(p, FFS(USED(c)));
			}
		};

		template<int c> struct SET
		{
			static MATH_FORCEINLINE float32x4_t f(float32x4_t p, float x)
			{
				return vsetq_lane_f32(x, p, FFS(USED(c)));
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
					(MSK & MSK_ZW)==0 ? SWZ_XY :
					MATCH_SWIZ(0, SWZ, SWZ(COMP_X, COMP_Y, COMP_Z, COMP_W), MSK, IGN) ? SWZ_XYZW :
#			ifdef META_PEEPHOLE
					MATCH_SWIZ(0, SWZ, SWZ(COMP_X, COMP_X, COMP_X, COMP_X), MSK, IGN) ? SWZ_XXXX :
					MATCH_SWIZ(0, SWZ, SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_Y), MSK, IGN) ? SWZ_YYYY :
					MATCH_SWIZ(0, SWZ, SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_Z), MSK, IGN) ? SWZ_ZZZZ :
					MATCH_SWIZ(0, SWZ, SWZ(COMP_W, COMP_W, COMP_W, COMP_W), MSK, IGN) ? SWZ_WWWW :
#			endif
					0
			};
			
			template_decl(struct impl, int swz)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d)
				{
					return MASK<SWZ_XYZW, SWZ_ZWXY, MSK_ZW>::f(MASK<X & MSK_X, (Y << 4) & MSK_Y, MSK_Y>::f(a, b), MASK<Z & MSK_X, (W << 4) & MSK_Y, MSK_Y>::f(c, d));
				}
			};
			
			template_spec(struct impl, SWZ_XY)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t a, float32x4_t b, float32x4_t, float32x4_t)
				{
					return MASK<X & MSK_X, (Y << 4) & MSK_Y, MSK_Y>::f(a, b);
				}
			};
			
			template_spec(struct impl, SWZ_XYZW)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d)
				{
					return vtrnq_f32(vcombine_f32(vget_low_f32(a), vget_high_f32(c)), vrev64q_f32(vcombine_f32(vget_low_f32(b), vget_high_f32(d)))).val[0];
				}
			};
			
			template_spec(struct impl, SWZ_XXXX)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d)
				{
					return vcombine_f32(vzip_f32(vget_low_f32(a), vget_low_f32(b)).val[0], vzip_f32(vget_low_f32(c), vget_low_f32(d)).val[0]);
				}
			};
			
			template_spec(struct impl, SWZ_YYYY)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d)
				{
					return vcombine_f32(vzip_f32(vget_low_f32(a), vget_low_f32(b)).val[1], vzip_f32(vget_low_f32(c), vget_low_f32(d)).val[1]);
				}
			};
			
			template_spec(struct impl, SWZ_ZZZZ)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d)
				{
					return vcombine_f32(vzip_f32(vget_high_f32(a), vget_high_f32(b)).val[0], vzip_f32(vget_high_f32(c), vget_high_f32(d)).val[0]);
				}
			};
			
			template_spec(struct impl, SWZ_WWWW)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d)
				{
					return vcombine_f32(vzip_f32(vget_high_f32(a), vget_high_f32(b)).val[1], vzip_f32(vget_high_f32(c), vget_high_f32(d)).val[1]);
				}
			};
			
			static MATH_FORCEINLINE float32x4_t f(float32x4_t a, float32x4_t b, float32x4_t c, float32x4_t d)
			{
				return template_inst(impl, SEL)::g(a, b, c, d);
			}
		};

		template<int RHS = SWZ_XYZW> struct NEG
		{
			enum
			{
				R = UnOp<v4f, RHS>::R,
				S = UnOp<v4f, RHS>::S
			};
			
			META_COST({ COST = (UnOp<v4f, RHS>::COST) + 1 })

			typedef v4f		type;

			static MATH_FORCEINLINE float32x4_t f(float32x4_t rhs)
			{
				return vnegq_f32(SWIZ<R>::f(rhs));
			}
		};

		template<int RHS = SWZ_XYZW> struct ABS
		{
			enum
			{
				R = UnOp<v4f, RHS>::R,
				S = UnOp<v4f, RHS>::S
			};

			META_COST({ COST = (UnOp<v4f, RHS>::COST) + 1 })

			typedef v4f		type;

			static MATH_FORCEINLINE float32x4_t f(float32x4_t rhs)
			{
				return vabsq_f32(SWIZ<R>::f(rhs));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct ADD
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4f		type;

			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vaddq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
			}
		};
		
		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SUB
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};
			
			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4f		type;

			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vsubq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MUL
		{

			enum
			{
/* Disable META_PEEPHOLE for MUL
#			ifdef META_PEEPHOLE
				LU = USED(LHS),
				RU = USED(RHS),
				SEL = POP(LU)==1 ? (1 + (LU>=MSK_Z)) : POP(RU)==1 ? (3 + (RU>=MSK_Z)) : 0,
				L = SEL==0 ? BinOp<v4f, LHS, RHS>::L : (LU & SWZ_XYZW),
				R = SEL==0 ? BinOp<v4f, LHS, RHS>::R : (RU & SWZ_XYZW),
				S = SEL==0 ? BinOp<v4f, LHS, RHS>::S : SEL>=3 ? L : R
#			else
*/
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S,
				SEL = 0
//#			endif
			};

			META_COST({ COST = SEL==0 ? (BinOp<v4f, LHS, RHS>::COST) + 2 : 2 })

			typedef v4f		type;

			template_decl(struct impl, int sel)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t rhs)
				{
					return vmulq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
				}
			};
/*			
#		ifdef META_PEEPHOLE

			template_spec(struct impl, 1)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t rhs)
				{
					return vmulq_lane_f32(rhs, vget_low_f32(lhs), (int)LU >= (int)MSK_Y);
				}
			};
			
			template_spec(struct impl, 2)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t rhs)
				{
					return vmulq_lane_f32(rhs, vget_high_f32(lhs), (int)LU >= (int)MSK_W);
				}
			};
			
			template_spec(struct impl, 3)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t rhs)
				{
					return vmulq_lane_f32(lhs, vget_low_f32(rhs), (int)RU >= (int)MSK_Y);
				}
			};
			
			template_spec(struct impl, 4)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t rhs)
				{
					return vmulq_lane_f32(lhs, vget_high_f32(rhs), (int)RU >= (int)MSK_W);
				}
			};
			
#		endif
*/
			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return template_inst(impl, SEL)::g(lhs, rhs);
			}
		};
	
		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct DIV
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 9 })

			typedef v4f		type;

#		if defined(__AARCH64_SIMD__)
			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vdivq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
			}
#		else
			static MATH_FORCEINLINE float32x4_t g(float32x4_t rhs)
			{
				float32x4_t e0 = vrecpeq_f32(rhs);
				float32x4_t r = vrecpsq_f32(rhs, e0), e = vmulq_f32(e0, r);
				r = vrecpsq_f32(rhs, e); e = vmulq_f32(e, r);
				return vbslq_f32(vceqq_f32(rhs, cv4f(0.f, 0.f, 0.f, 0.f)), e0, e);
			}
			
			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vmulq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(g(rhs)));
			}
#		endif
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPEQ
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};
			
			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vceqq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPNE
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return veorq_s32(vreinterpretq_s32_u32(vceqq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs))), cv4i(~0, ~0, ~0, ~0));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPGT
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vcgtq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPGE
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vcgeq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPLT
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};
			
			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vcltq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPLE
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vcleq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
			}
		};

		template<int LHS = SWZ_XYZW, int CHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MADD
		{
		
#	if defined(MATH_HAS_FAST_MADD)

			enum
			{
#			ifdef META_PEEPHOLE
				LU = USED(LHS),
				CU = USED(CHS),
				SEL = POP(LU)==1 ? (1 + (LU>=MSK_Z)) : POP(CU)==1 ? (3 + (CU>=MSK_Z)) : 0,
				L = SEL==0 ? TernOp<v4f, LHS, CHS, RHS>::L : SEL>=3 ? BinOp<v4f, LHS, RHS>::L : (LU & SWZ_XYZW),
				C = SEL==0 ? TernOp<v4f, LHS, CHS, RHS>::C : SEL>=3 ? (CU & SWZ_XYZW) : BinOp<v4f, CHS, RHS>::L,
				R = SEL==0 ? TernOp<v4f, LHS, CHS, RHS>::R : SEL>=3 ? BinOp<v4f, LHS, RHS>::R : BinOp<v4f, CHS, RHS>::R,
				S = SEL==0 ? TernOp<v4f, LHS, CHS, RHS>::S : SEL>=3 ? BinOp<v4f, LHS, RHS>::S : BinOp<v4f, CHS, RHS>::S
#			else
				L = TernOp<v4f, LHS, CHS, RHS>::L,
				C = TernOp<v4f, LHS, CHS, RHS>::C,
				R = TernOp<v4f, LHS, CHS, RHS>::R,
				S = TernOp<v4f, LHS, CHS, RHS>::S,
				SEL = 0
#			endif
			};

			META_COST({ COST = (SEL==0 ? (SWIZ<L>::COST + SWIZ<C>::COST + SWIZ<R>::COST) : SEL>=3 ? (BinOp<v4f, LHS, RHS>::COST) : (BinOp<v4f, CHS, RHS>::COST)) + 3 })

			typedef v4f		type;

			template_decl(struct impl, int sel)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlaq_f32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), SWIZ<C>::f(chs));
				}
			};
			
#		ifdef META_PEEPHOLE

			template_spec(struct impl, 1)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlaq_lane_f32(SWIZ<R>::f(rhs), SWIZ<C>::f(chs), vget_low_f32(lhs), LU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 2)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlaq_lane_f32(SWIZ<R>::f(rhs), SWIZ<C>::f(chs), vget_high_f32(lhs), LU >= MSK_W);
				}
			};
			
			template_spec(struct impl, 3)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlaq_lane_f32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), vget_low_f32(chs), CU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 4)
			{
				static MATH_FORCEINLINE float32x4_t gfloat32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlaq_lane_f32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), vget_high_f32(chs), CU >= MSK_W);
				}
			};
			
#		endif

			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
			{
				return template_inst(impl, SEL)::g(lhs, chs, rhs);
			}
			
#	else

			enum
			{
				L0 = TernOp<v4f, LHS, CHS, RHS>::L0,
				C0 = TernOp<v4f, LHS, CHS, RHS>::C0,
				S0 = TernOp<v4f, LHS, CHS, RHS>::S0,
				LC = TernOp<v4f, LHS, CHS, RHS>::LC,
				R = TernOp<v4f, LHS, CHS, RHS>::R,
				S = TernOp<v4f, LHS, CHS, RHS>::S
			};
			
			META_COST({ COST = (TernOp<v4f, LHS, CHS, RHS>::COST) + 2 + 2 })

			typedef v4f		type;

			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
			{
				return vaddq_f32(SWIZ<LC>::f(vmulq_f32(SWIZ<L0>::f(lhs), SWIZ<C0>::f(chs))), SWIZ<R>::f(rhs));
			}

#	endif

		};

		template<int LHS = SWZ_XYZW, int CHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MSUB
		{
		
#	if defined(MATH_HAS_FAST_MADD)
		
			enum
			{
#			ifdef META_PEEPHOLE
				LU = USED(LHS),
				CU = USED(CHS),
				SEL = POP(LU)==1 ? (1 + (LU>=MSK_Z)) : POP(CU)==1 ? (3 + (CU>=MSK_Z)) : 0,
				L = SEL==0 ? TernOp<v4f, LHS, CHS, RHS>::L : SEL>=3 ? BinOp<v4f, LHS, RHS>::L : (LU & SWZ_XYZW),
				C = SEL==0 ? TernOp<v4f, LHS, CHS, RHS>::C : SEL>=3 ? (CU & SWZ_XYZW) : BinOp<v4f, CHS, RHS>::L,
				R = SEL==0 ? TernOp<v4f, LHS, CHS, RHS>::R : SEL>=3 ? BinOp<v4f, LHS, RHS>::R : BinOp<v4f, CHS, RHS>::R,
				S = SEL==0 ? TernOp<v4f, LHS, CHS, RHS>::S : SEL>=3 ? BinOp<v4f, LHS, RHS>::S : BinOp<v4f, CHS, RHS>::S
#			else
				L = TernOp<v4f, LHS, CHS, RHS>::L,
				C = TernOp<v4f, LHS, CHS, RHS>::C,
				R = TernOp<v4f, LHS, CHS, RHS>::R,
				S = TernOp<v4f, LHS, CHS, RHS>::S,
				SEL = 0
#			endif
			};

			META_COST({ COST = (SEL==0 ? (SWIZ<L>::COST + SWIZ<C>::COST + SWIZ<R>::COST) : SEL>=3 ? (BinOp<v4f, LHS, RHS>::COST) : (BinOp<v4f, CHS, RHS>::COST)) + 3 })

			typedef v4f		type;

			template_decl(struct impl, int sel)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlsq_f32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), SWIZ<C>::f(chs));
				}
			};
			
#		ifdef META_PEEPHOLE

			template_spec(struct impl, 1)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlsq_lane_f32(SWIZ<R>::f(rhs), SWIZ<C>::f(chs), vget_low_f32(lhs), LU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 2)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlsq_lane_f32(SWIZ<R>::f(rhs), SWIZ<C>::f(chs), vget_high_f32(lhs), LU >= MSK_W);
				}
			};
			
			template_spec(struct impl, 3)
			{
				static MATH_FORCEINLINE float32x4_t g(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlsq_lane_f32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), vget_low_f32(chs), CU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 4)
			{
				static MATH_FORCEINLINE float32x4_t gfloat32x4_t lhs, float32x4_t chs, float32x4_t rhs)
				{
					return vmlsq_lane_f32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), vget_high_f32(chs), CU >= MSK_W);
				}
			};

#		endif

			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
			{
				return template_inst(impl, SEL)::g(lhs, chs, rhs);
			}

#	else

			enum
			{
				L0 = TernOp<v4f, LHS, CHS, RHS>::L0,
				C0 = TernOp<v4f, LHS, CHS, RHS>::C0,
				S0 = TernOp<v4f, LHS, CHS, RHS>::S0,
				LC = TernOp<v4f, LHS, CHS, RHS>::LC,
				R = TernOp<v4f, LHS, CHS, RHS>::R,
				S = TernOp<v4f, LHS, CHS, RHS>::S
			};
			
			META_COST({ COST = (TernOp<v4f, LHS, CHS, RHS>::COST) + 2 + 2 })

			typedef v4f		type;

			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t chs, float32x4_t rhs)
			{
				return vsubq_f32(SWIZ<LC>::f(vmulq_f32(SWIZ<L0>::f(lhs), SWIZ<C0>::f(chs))), SWIZ<R>::f(rhs));
			}

#	endif

		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MIN
		{
			enum
			{
				L = BinOp<v4f, LHS, RHS>::L,
				R = BinOp<v4f, LHS, RHS>::R,
				S = BinOp<v4f, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4f		type;

			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vminq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
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

			META_COST({ COST = (BinOp<v4f, LHS, RHS>::COST) + 2 })

			typedef v4f		type;

			static MATH_FORCEINLINE float32x4_t f(float32x4_t lhs, float32x4_t rhs)
			{
				return vmaxq_f32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
			}
		};
	};

	struct v4i
	{
	
		typedef int32x4_t	packed;
		typedef int			type;
		
		static MATH_FORCEINLINE int32x4_t ZERO()
		{
			return vdupq_n_s32(0);
		}

		static MATH_FORCEINLINE int32x4_t CTOR(int x)
		{
			return vdupq_n_s32(x);
		}
		
		static MATH_FORCEINLINE int32x4_t CTOR(int x, int y)
		{
#if defined(_MSC_VER)
			const int values[] = { x, y, x, y };
			return vld1q_s32(values);
#else
			return (int32x4_t) { x, y, x, y };
#endif
		}
		
		static MATH_FORCEINLINE int32x4_t CTOR(int x, int y, int z)
		{
#if defined(_MSC_VER)
			const int values[] = { x, y, z, 0 };
			return vld1q_s32(values);
#else
			return (int32x4_t) { x, y, z, 0 };
#endif
		}
		
		static MATH_FORCEINLINE int32x4_t CTOR(int x, int y, int z, int w)
		{
#if defined(_MSC_VER)
			const int values[] = { x, y, z, w };
			return vld1q_s32(values);
#else
			return (int32x4_t) { x, y, z, w };
#endif
		}

#		define CASE_MATCH(X, Y, Z, W)	(SWZ & MSK)==(SWZ(COMP_##X, COMP_##Y, COMP_##Z, COMP_##W) & MSK) ? CASE_##X##Y##Z##W
		
		template<unsigned SWZ> struct SWIZ
		{
			META_COST({ COST = v4f::SWIZ<SWZ>::COST })

			static MATH_FORCEINLINE int32x4_t f(int32x4_t p)
			{
				return v4f::SWIZ<SWZ>::f(p);
			}
		};

		template<int LHS, int RHS, int SEL> struct MASK
		{
			static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b)
			{
				return v4f::MASK<LHS, RHS, SEL>::f(a, b);
			}
		};

		template<int c> struct GET
		{
			static MATH_FORCEINLINE int f(int32x4_t p)
			{
				return vgetq_lane_s32(p, FFS(USED(c)));
			}
		};

		template<int c> struct SET
		{
			static MATH_FORCEINLINE int32x4_t f(int32x4_t p, int i)
			{
				return vsetq_lane_s32(i, p, FFS(USED(c)));
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
					(MSK & MSK_ZW)==0 ? SWZ_XY :
					MATCH_SWIZ(0, SWZ, SWZ(COMP_X, COMP_Y, COMP_Z, COMP_W), MSK, IGN) ? SWZ_XYZW :
#			ifdef META_PEEPHOLE
					MATCH_SWIZ(0, SWZ, SWZ(COMP_X, COMP_X, COMP_X, COMP_X), MSK, IGN) ? SWZ_XXXX :
					MATCH_SWIZ(0, SWZ, SWZ(COMP_Y, COMP_Y, COMP_Y, COMP_Y), MSK, IGN) ? SWZ_YYYY :
					MATCH_SWIZ(0, SWZ, SWZ(COMP_Z, COMP_Z, COMP_Z, COMP_Z), MSK, IGN) ? SWZ_ZZZZ :
					MATCH_SWIZ(0, SWZ, SWZ(COMP_W, COMP_W, COMP_W, COMP_W), MSK, IGN) ? SWZ_WWWW :
#			endif
					0
			};
			
			template_decl(struct impl, int swz)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d)
				{
					return MASK<SWZ_XYZW, SWZ_ZWXY, MSK_ZW>::f(MASK<X & MSK_X, (Y << 4) & MSK_Y, MSK_Y>::f(a, b), MASK<Z & MSK_X, (W << 4) & MSK_Y, MSK_Y>::f(c, d));
				}
			};
			
			template_spec(struct impl, SWZ_XY)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t a, int32x4_t b, int32x4_t, int32x4_t)
				{
					return MASK<X & MSK_X, (Y << 4) & MSK_Y, MSK_Y>::f(a, b);
				}
			};
			
			template_spec(struct impl, SWZ_XYZW)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d)
				{
					return vtrnq_s32(vcombine_s32(vget_low_s32(a), vget_high_s32(c)), vrev64q_s32(vcombine_s32(vget_low_s32(b), vget_high_s32(d)))).val[0];
				}
			};
			
#		ifdef META_PEEPHOLE

			template_spec(struct impl, SWZ_XXXX)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d)
				{
					return vcombine_s32(vzip_s32(vget_low_s32(a), vget_low_s32(b)).val[0], vzip_s32(vget_low_s32(c), vget_low_s32(d)).val[0]);
				}
			};
			
			template_spec(struct impl, SWZ_YYYY)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d)
				{
					return vcombine_s32(vzip_s32(vget_low_s32(a), vget_low_s32(b)).val[1], vzip_s32(vget_low_s32(c), vget_low_s32(d)).val[1]);
				}
			};
			
			template_spec(struct impl, SWZ_ZZZZ)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d)
				{
					return vcombine_s32(vzip_s32(vget_high_s32(a), vget_high_s32(b)).val[0], vzip_s32(vget_high_s32(c), vget_high_s32(d)).val[0]);
				}
			};
			
			template_spec(struct impl, SWZ_WWWW)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d)
				{
					return vcombine_s32(vzip_s32(vget_high_s32(a), vget_high_s32(b)).val[1], vzip_s32(vget_high_s32(c), vget_high_s32(d)).val[1]);
				}
			};
			
#		endif

			static MATH_FORCEINLINE int32x4_t f(int32x4_t a, int32x4_t b, int32x4_t c, int32x4_t d)
			{
				return template_inst(impl, SEL)::g(a, b, c, d);
			}
		};

		template<int SWZ> struct ANY
		{
			enum
			{
				USE = USED(SWZ)
			};
			
			template_decl(struct impl, int sel)
			{
				static MATH_FORCEINLINE int g(int32x4_t)
				{
					return 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0, 0, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vgetq_lane_s32(p, 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0xf, 0, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vgetq_lane_s32(p, 1) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0xf, 0, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vpmin_s32(vget_low_s32(p), vget_low_s32(p)), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0, 0xf, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vgetq_lane_s32(p, 2) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0, 0xf, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vmin_s32(vget_low_s32(p), vget_high_s32(p)), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0xf, 0xf, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vmin_s32(vrev64_s32(vget_low_s32(p)), vget_high_s32(p)), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0xf, 0xf, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmin_s32(vget_low_s32(p), vget_low_s32(p));
					return vget_lane_s32(vmin_s32(vget_high_s32(p), r), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0, 0, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vgetq_lane_s32(p, 3) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0, 0, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vmin_s32(vget_low_s32(p), vrev64_s32(vget_high_s32(p))), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0xf, 0, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vmin_s32(vget_low_s32(p), vget_high_s32(p)), 1) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0xf, 0, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmin_s32(vget_low_s32(p), vget_low_s32(p));
					return vget_lane_s32(vmin_s32(vget_high_s32(p), r), 1) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0, 0xf, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vpmin_s32(vget_high_s32(p), vget_high_s32(p)), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0, 0xf, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmin_s32(vget_high_s32(p), vget_high_s32(p));
					return vget_lane_s32(vmin_s32(vget_low_s32(p), r), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0xf, 0xf, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmin_s32(vget_high_s32(p), vget_high_s32(p));
					return vget_lane_s32(vmin_s32(vget_low_s32(p), r), 1) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0xf, 0xf, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmin_s32(vget_low_s32(p), vget_high_s32(p));
					return vget_lane_s32(vpmin_s32(r, r), 0) < 0;
				}
			};
			
			static MATH_FORCEINLINE int f(int32x4_t p)
			{
				return template_inst(impl, USE)::g(p);
			}
		};
		
		template<int SWZ> struct ALL
		{
			enum
			{
				USE = USED(SWZ)
			};
			
			template_decl(struct impl, int sel)
			{
				static MATH_FORCEINLINE int g(int32x4_t)
				{
					return 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0, 0, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vgetq_lane_s32(p, 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0xf, 0, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vgetq_lane_s32(p, 1) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0xf, 0, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vpmax_s32(vget_low_s32(p), vget_low_s32(p)), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0, 0xf, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vgetq_lane_s32(p, 2) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0, 0xf, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vmax_s32(vget_low_s32(p), vget_high_s32(p)), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0xf, 0xf, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vmax_s32(vrev64_s32(vget_low_s32(p)), vget_high_s32(p)), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0xf, 0xf, 0))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmax_s32(vget_low_s32(p), vget_low_s32(p));
					return vget_lane_s32(vmax_s32(vget_high_s32(p), r), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0, 0, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vgetq_lane_s32(p, 3) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0, 0, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vmax_s32(vget_low_s32(p), vrev64_s32(vget_high_s32(p))), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0xf, 0, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vmax_s32(vget_low_s32(p), vget_high_s32(p)), 1) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0xf, 0, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmax_s32(vget_low_s32(p), vget_low_s32(p));
					return vget_lane_s32(vmax_s32(vget_high_s32(p), r), 1) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0, 0xf, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					return vget_lane_s32(vpmax_s32(vget_high_s32(p), vget_high_s32(p)), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0, 0xf, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmax_s32(vget_high_s32(p), vget_high_s32(p));
					return vget_lane_s32(vmax_s32(vget_low_s32(p), r), 0) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0, 0xf, 0xf, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmax_s32(vget_high_s32(p), vget_high_s32(p));
					return vget_lane_s32(vmax_s32(vget_low_s32(p), r), 1) < 0;
				}
			};
			
			template_spec(struct impl, MSK(0xf, 0xf, 0xf, 0xf))
			{
				static MATH_FORCEINLINE int g(int32x4_t p)
				{
					int32x2_t r = vpmax_s32(vget_low_s32(p), vget_high_s32(p));
					return vget_lane_s32(vpmax_s32(r, r), 0) < 0;
				}
			};
			
			static MATH_FORCEINLINE int f(int32x4_t p)
			{
				return template_inst(impl, USE)::g(p);
			}
		};

		static MATH_FORCEINLINE int32x4_t BOOL(int32x4_t a)
		{
			return vnegq_s32(a);
		}

		template<int RHS = SWZ_XYZW> struct NEG
		{
			enum
			{
				R = UnOp<v4i, RHS>::R,
				S = UnOp<v4i, RHS>::S
			};

			META_COST({ COST = (UnOp<v4i, RHS>::COST) + 1 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t rhs)
			{
				return vnegq_s32(SWIZ<R>::f(rhs));
			}
		};

		template<int RHS = SWZ_XYZW> struct ABS
		{
			enum
			{
				R = UnOp<v4i, RHS>::R,
				S = UnOp<v4i, RHS>::S
			};

			META_COST({ COST = (UnOp<v4i, RHS>::COST) + 1 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t rhs)
			{
				return vabsq_s32(SWIZ<R>::f(rhs));
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

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t rhs)
			{
				return veorq_s32(SWIZ<R>::f(rhs), cv4i(~0, ~0, ~0, ~0));
			}
		};

		template<int LHS = SWZ_XYZW> struct SLLI
		{
			enum
			{
				L = UnOp<v4i, LHS>::R,
				S = UnOp<v4i, LHS>::S
			};

			META_COST({ COST = (UnOp<v4i, LHS>::COST) + 1 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, unsigned i)
			// TODO: find a way to use vshlq_n_s32 if <i> is known at compile-time
			// { return vshlq_n_s32(lhs, i); }
			// { int32x4_t r; __asm__("vshl.s32 %0, %1, %2" : "=w" (r) : "w" ((int32x4_t) lhs), "I" (i)); return r; }
			{
				return vshlq_s32(SWIZ<L>::f(lhs), CTOR(i));
			}
		};

		template<int LHS = SWZ_XYZW> struct SRLI
		{
			enum
			{
				L = UnOp<v4i, LHS>::R,
				S = UnOp<v4i, LHS>::S
			};

			META_COST({ COST = (UnOp<v4i, LHS>::COST) + 1 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, unsigned i)
			// TODO: find a way to use vshrq_n_u32 if <i> is known at compile-time
			// { return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(lhs), i)); }
			// { int32x4_t r; __asm__("vshr.u32 %0, %1, %2" : "=w" (r) : "w" ((int32x4_t) lhs), "I" (i)); return r; }
			{
				return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(SWIZ<L>::f(lhs)), CTOR(-i)));
			}
		};

		template<int LHS = SWZ_XYZW> struct SRAI
		{
			enum
			{
				L = UnOp<v4i, LHS>::R,
				S = UnOp<v4i, LHS>::S
			};

			META_COST({ COST = (UnOp<v4i, LHS>::COST) + 1 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, unsigned i)
			// TODO: find a way to use vshrq_n_s32 if <i> is known at compile-time
			// { return vshrq_n_s32(lhs, i); }
			// { int32x4_t r; __asm__("vshr.s32 %0, %1, %2" : "=w" (r) : "w" ((int32x4_t) lhs), "I" (i)); return r; }
			{
				return vshlq_s32(SWIZ<L>::f(lhs), CTOR(-static_cast<int>(i)));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct ADD
		{
			enum
			{
				L = BinOp<v4i, LHS, RHS>::L,
				R = BinOp<v4i, LHS, RHS>::R,
				S = BinOp<v4i, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vaddq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vsubq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MUL
		{
			enum
			{
			// Disable META_PEEPHOLE for MUL
#			if defined(META_PEEPHOLE) && 0
				LU = USED(LHS),
				RU = USED(RHS),
				SEL = POP(LU)==1 ? (1 + (LU>=MSK_Z)) : POP(RU)==1 ? (3 + (RU>=MSK_Z)) : 0,
				L = SEL==0 ? BinOp<v4i, LHS, RHS>::L : (LU & SWZ_XYZW),
				R = SEL==0 ? BinOp<v4i, LHS, RHS>::R : (RU & SWZ_XYZW),
				S = SEL==0 ? BinOp<v4i, LHS, RHS>::S : SEL>=3 ? L : R
#			else

				LU = USED(LHS),
				RU = USED(RHS),
				L = BinOp<v4i, LHS, RHS>::L,
				R = BinOp<v4i, LHS, RHS>::R,
				S = BinOp<v4i, LHS, RHS>::S,
				SEL = 0
#			endif
			};

			META_COST({ COST = SEL==0 ? (BinOp<v4i, LHS, RHS>::COST) + 2 : 2 })

			typedef v4i		type;

			template_decl(struct impl, int sel)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t rhs)
				{
					return vmulq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
				}
			};
			
			template_spec(struct impl, 1)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t rhs)
				{
					return vmulq_lane_s32(rhs, vget_low_s32(lhs), LU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 2)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t rhs)
				{
					return vmulq_lane_s32(rhs, vget_high_s32(lhs), LU >= MSK_W);
				}
			};
			
			template_spec(struct impl, 3)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t rhs)
				{
					return vmulq_lane_s32(lhs, vget_low_s32(rhs), RU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 4)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t rhs)
				{
					return vmulq_lane_s32(lhs, vget_high_s32(rhs), RU >= MSK_W);
				}
			};
			
			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return template_inst(impl, SEL)::g(lhs, rhs);
			}
		};
	
		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct DIV
		{
			enum
			{
				N = Swizzle<LHS>::N,
				L = LHS,
				R = RHS,
				S = N==4 ? SWZ_XYZW : N==3 ? SWZ_XYZ : N==2 ? SWZ_XY : SWZ_X,
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

			META_COST({ COST = 16*N })

			typedef v4i		type;

			template_decl(struct impl, int sel)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t rhs)
				{
					return type::CTOR(
						((const int *) &lhs)[L0]/((const int *) &rhs)[R0],
						((const int *) &lhs)[L1]/((const int *) &rhs)[R1],
						((const int *) &lhs)[L2]/((const int *) &rhs)[R2],
						((const int *) &lhs)[L3]/((const int *) &rhs)[R3]
					);
				}
			};
			
			template_spec(struct impl, 3)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t rhs)
				{
					return type::CTOR(
						((const int *) &lhs)[L0]/((const int *) &rhs)[R0],
						((const int *) &lhs)[L1]/((const int *) &rhs)[R1],
						((const int *) &lhs)[L2]/((const int *) &rhs)[R2]
					);
				}
			};
			
			template_spec(struct impl, 2)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t rhs)
				{
					return type::CTOR(
						((const int *) &lhs)[L0]/((const int *) &rhs)[R0],
						((const int *) &lhs)[L1]/((const int *) &rhs)[R1]
					);
				}
			};
			
			template_spec(struct impl, 1)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t rhs)
				{
					return type::CTOR(
						((const int *) &lhs)[L0]/((const int *) &rhs)[R0]
					);
				}
			};
			
			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return template_inst(impl, SEL)::g(lhs, rhs);
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SLL
		{
			enum
			{
				L = BinOp<v4i, LHS, RHS>::L,
				R = BinOp<v4i, LHS, RHS>::R,
				S = BinOp<v4i, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vshlq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SLR
		{
			enum
			{
				L = BinOp<v4i, LHS, RHS>::L,
				R = BinOp<v4i, LHS, RHS>::R,
				S = BinOp<v4i, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vshrq_u32(vreinterpretq_u32_s32(SWIZ<L>::f(lhs)), SWIZ<R>::f(rhs)));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct SRA
		{
			enum
			{
				L = BinOp<v4i, LHS, RHS>::L,
				R = BinOp<v4i, LHS, RHS>::R,
				S = BinOp<v4i, LHS, RHS>::S
			};
			typedef v4i		type;

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vshlq_s32(SWIZ<L>::f(lhs), vnegq_s32(SWIZ<R>::f(rhs)));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vandq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vorrq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return veorq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
			}
		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct CMPEQ
		{
			enum
			{
				L = BinOp<v4i, LHS, RHS>::L,
				R = BinOp<v4i, LHS, RHS>::R,
				S = BinOp<v4i, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vceqq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return veorq_s32(vreinterpretq_s32_u32(vceqq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs))), cv4i(~0, ~0, ~0, ~0));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vcgtq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vcgeq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vcltq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vreinterpretq_s32_u32(vcleq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs)));
			}
		};

		template<int LHS = SWZ_XYZW, int CHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MADD
		{

#		if defined(MATH_HAS_FAST_MADD)

			enum
			{
#			ifdef META_PEEPHOLE
				LU = USED(LHS),
				CU = USED(CHS),
				SEL = POP(LU)==1 ? (1 + (LU>=MSK_Z)) : POP(CU)==1 ? (3 + (CU>=MSK_Z)) : 0,
				L = SEL==0 ? TernOp<v4i, LHS, CHS, RHS>::L : SEL>=3 ? BinOp<v4i, LHS, RHS>::L : (LU & SWZ_XYZW),
				C = SEL==0 ? TernOp<v4i, LHS, CHS, RHS>::C : SEL>=3 ? (CU & SWZ_XYZW) : BinOp<v4i, CHS, RHS>::L,
				R = SEL==0 ? TernOp<v4i, LHS, CHS, RHS>::R : SEL>=3 ? BinOp<v4i, LHS, RHS>::R : BinOp<v4i, CHS, RHS>::R,
				S = SEL==0 ? TernOp<v4i, LHS, CHS, RHS>::S : SEL>=3 ? BinOp<v4i, LHS, RHS>::S : BinOp<v4i, CHS, RHS>::S
#			else
				L = TernOp<v4i, LHS, CHS, RHS>::L,
				C = TernOp<v4i, LHS, CHS, RHS>::C,
				R = TernOp<v4i, LHS, CHS, RHS>::R,
				S = TernOp<v4i, LHS, CHS, RHS>::S,
				SEL = 0
#			endif
			};

			META_COST({ COST = (SEL==0 ? (SWIZ<L>::COST + SWIZ<C>::COST + SWIZ<R>::COST) : SEL>=3 ? (BinOp<v4i, LHS, RHS>::COST) : (BinOp<v4i, CHS, RHS>::COST)) + 3 })

			typedef v4i		type;

			template_decl(struct impl, int sel)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlaq_s32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), SWIZ<C>::f(chs));
				}
			};
			
#		ifdef META_PEEPHOLE

			template_spec(struct impl, 1)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlaq_lane_s32(SWIZ<R>::f(rhs), SWIZ<C>::f(chs), vget_low_s32(lhs), LU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 2)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlaq_lane_s32(SWIZ<R>::f(rhs), SWIZ<C>::f(chs), vget_high_s32(lhs), LU >= MSK_W);
				}
			};
			
			template_spec(struct impl, 3)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlaq_lane_s32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), vget_low_s32(chs), CU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 4)
			{
				static MATH_FORCEINLINE int32x4_t gint32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlaq_lane_s32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), vget_high_s32(chs), CU >= MSK_W);
				}
			};
			
#		endif

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
			{
				return template_inst(impl, SEL)::g(lhs, chs, rhs);
			}

#	else

			enum
			{
				L0 = TernOp<v4i, LHS, CHS, RHS>::L0,
				C0 = TernOp<v4i, LHS, CHS, RHS>::C0,
				S0 = TernOp<v4i, LHS, CHS, RHS>::S0,
				LC = TernOp<v4i, LHS, CHS, RHS>::LC,
				R = TernOp<v4i, LHS, CHS, RHS>::R,
				S = TernOp<v4i, LHS, CHS, RHS>::S
			};

			META_COST({ COST = (TernOp<v4i, LHS, CHS, RHS>::COST) + 2 + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
			{
				return vaddq_s32(SWIZ<LC>::f(vmulq_s32(SWIZ<L0>::f(lhs), SWIZ<C0>::f(chs))), SWIZ<R>::f(rhs));
			}

#	endif

		};

		template<int LHS = SWZ_XYZW, int CHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MSUB
		{

#		if defined(MATH_HAS_FAST_MADD)

			enum
			{
#			ifdef META_PEEPHOLE
				LU = USED(LHS),
				CU = USED(CHS),
				SEL = POP(LU)==1 ? (1 + (LU>=MSK_Z)) : POP(CU)==1 ? (3 + (CU>=MSK_Z)) : 0,
				L = SEL==0 ? TernOp<v4i, LHS, CHS, RHS>::L : SEL>=3 ? BinOp<v4i, LHS, RHS>::L : (LU & SWZ_XYZW),
				C = SEL==0 ? TernOp<v4i, LHS, CHS, RHS>::C : SEL>=3 ? (CU & SWZ_XYZW) : BinOp<v4i, CHS, RHS>::L,
				R = SEL==0 ? TernOp<v4i, LHS, CHS, RHS>::R : SEL>=3 ? BinOp<v4i, LHS, RHS>::R : BinOp<v4i, CHS, RHS>::R,
				S = SEL==0 ? TernOp<v4i, LHS, CHS, RHS>::S : SEL>=3 ? BinOp<v4i, LHS, RHS>::S : BinOp<v4i, CHS, RHS>::S,
#			else
				L = TernOp<v4i, LHS, CHS, RHS>::L,
				C = TernOp<v4i, LHS, CHS, RHS>::C,
				R = TernOp<v4i, LHS, CHS, RHS>::R,
				S = TernOp<v4i, LHS, CHS, RHS>::S,
				SEL = 0
#			endif
			};

			META_COST({ COST = (SEL==0 ? (SWIZ<L>::COST + SWIZ<C>::COST + SWIZ<R>::COST) : SEL>=3 ? (BinOp<v4i, LHS, RHS>::COST) : (BinOp<v4i, CHS, RHS>::COST)) + 3 })

			typedef v4i		type;

			template_decl(struct impl, int sel)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlsq_s32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), SWIZ<C>::f(chs));
				}
			};
			
#		ifdef META_PEEPHOLE

			template_spec(struct impl, 1)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlsq_lane_s32(SWIZ<R>::f(rhs), SWIZ<C>::f(chs), vget_low_s32(lhs), LU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 2)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlsq_lane_s32(SWIZ<R>::f(rhs), SWIZ<C>::f(chs), vget_high_s32(lhs), LU >= MSK_W);
				}
			};
			
			template_spec(struct impl, 3)
			{
				static MATH_FORCEINLINE int32x4_t g(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlsq_lane_s32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), vget_low_s32(chs), CU >= MSK_Y);
				}
			};
			
			template_spec(struct impl, 4)
			{
				static MATH_FORCEINLINE int32x4_t gint32x4_t lhs, int32x4_t chs, int32x4_t rhs)
				{
					return vmlsq_lane_s32(SWIZ<R>::f(rhs), SWIZ<L>::f(lhs), vget_high_s32(chs), CU >= MSK_W);
				}
			};
			
#		endif

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
			{
				return template_inst(impl, SEL)::g(lhs, chs, rhs);
			}

#	else

			enum
			{
				L0 = TernOp<v4i, LHS, CHS, RHS>::L0,
				C0 = TernOp<v4i, LHS, CHS, RHS>::C0,
				S0 = TernOp<v4i, LHS, CHS, RHS>::S0,
				LC = TernOp<v4i, LHS, CHS, RHS>::LC,
				R = TernOp<v4i, LHS, CHS, RHS>::R,
				S = TernOp<v4i, LHS, CHS, RHS>::S
			};

			META_COST({ COST = (TernOp<v4i, LHS, CHS, RHS>::COST) + 2 + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t chs, int32x4_t rhs)
			{
				return vsubq_s32(SWIZ<LC>::f(vmulq_s32(SWIZ<L0>::f(lhs), SWIZ<C0>::f(chs))), SWIZ<R>::f(rhs));
			}

#	endif

		};

		template<int LHS = SWZ_XYZW, int RHS = SWZ_XYZW> struct MIN
		{
			enum
			{
				L = BinOp<v4i, LHS, RHS>::L,
				R = BinOp<v4i, LHS, RHS>::R,
				S = BinOp<v4i, LHS, RHS>::S
			};

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vminq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
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

			META_COST({ COST = (BinOp<v4i, LHS, RHS>::COST) + 2 })

			typedef v4i		type;

			static MATH_FORCEINLINE int32x4_t f(int32x4_t lhs, int32x4_t rhs)
			{
				return vmaxq_s32(SWIZ<L>::f(lhs), SWIZ<R>::f(rhs));
			}
		};
	};

}

}
