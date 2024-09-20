//
//  sinpi: Returns the sine of an angle of 'a', expressed in multiple of PI.
//      by example if you want to compute the sin of 90 degree you pass 1/2 (PI/2 = 90)
//  note: This function is more precise and faster than sin
//
static MATH_FORCEINLINE float4 sinpi(const float4 &a);
static MATH_FORCEINLINE float3 sinpi(const float3 &a);
static MATH_FORCEINLINE float2 sinpi(const float2 &a);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 sinpi(const float1 &a);
#endif

//
//  cospi: Returns the cosine of an angle of 'a', expressed in multiple of PI.
//      by example if you want to compute the cos of 90 degree you pass 1/2 (PI/2 = 90)
//  note: This function is more precise and faster than cos
//
static MATH_FORCEINLINE float4 cospi(const float4 &a);
static MATH_FORCEINLINE float3 cospi(const float3 &a);
static MATH_FORCEINLINE float2 cospi(const float2 &a);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 cospi(const float1 &a);
#endif

//
//  sin: Returns the sine of an angle of 'a', expressed in radians.
//
static MATH_FORCEINLINE float4 sin(const float4 &a);
static MATH_FORCEINLINE float3 sin(const float3 &a);
static MATH_FORCEINLINE float2 sin(const float2 &a);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 sin(const float1 &a);
#endif
static MATH_FORCEINLINE float sin(float x);

//
//  cos: Returns the cosine of an angle of 'a', expressed in radians.
//
static MATH_FORCEINLINE float4 cos(const float4 &a);
static MATH_FORCEINLINE float3 cos(const float3 &a);
static MATH_FORCEINLINE float2 cos(const float2 &a);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 cos(const float1 &a);
#endif
static MATH_FORCEINLINE float cos(float x);

//
//  sincos: Returns the sine and cosine of an angle of 'v', expressed in radians.
//
static MATH_FORCEINLINE void sincos(const float4 &v, float4& s, float4& c);
static MATH_FORCEINLINE void sincos(const float3 &v, float3& s, float3& c);
static MATH_FORCEINLINE void sincos(const float2 &v, float2& s, float2& c);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE void sincos(const float1 &v, float1& s, float1& c);
#endif
static MATH_FORCEINLINE void sincos(float v, float& s, float& c);


//
//  sincos_sssc: Returns the sine and cosine of an angle of 'v', expressed in radians.
//  return value is float4(s,s,s,c)
//
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float4 sincos_sssc(const float1 &a);
#endif
static MATH_FORCEINLINE float4 sincos_sssc(float a);

//
//  tan: Returns the tangent of an angle of 'v', expressed in radians.
//
static MATH_FORCEINLINE float4 tan(const float4 &v);
static MATH_FORCEINLINE float3 tan(const float3 &v);
static MATH_FORCEINLINE float2 tan(const float2 &v);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 tan(const float1 &v);
#endif
static MATH_FORCEINLINE float tan(float v);

//
//  atan: Returns the arc tangent of an angle of 'a', expressed in radians.
//  note: Notice that because of the sign ambiguity, the function cannot determine
//        with certainty in which quadrant the angle falls only by its tangent value.
//        See atan2 for an alternative that takes a fractional argument instead.
//
static MATH_FORCEINLINE float4 atan(const float4 &a);
static MATH_FORCEINLINE float3 atan(const float3 &a);
static MATH_FORCEINLINE float2 atan(const float2 &a);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 atan(const float1 &a);
#endif
static MATH_FORCEINLINE float atan(float a);

//
// atan2: Returns the arc tangent of a/b, expressed in radians.
// note: To compute the value, the function takes into account the sign of both arguments in order to determine the quadrant.
//
static MATH_FORCEINLINE float4 atan2(const float4 &a, const float4 &b);
static MATH_FORCEINLINE float3 atan2(const float3 &a, const float3& b);
static MATH_FORCEINLINE float2 atan2(const float2 &a, const float2& b);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 atan2(const float1 &a, const float1 &b);
#endif
static MATH_FORCEINLINE float atan2(float a, float b);

//
// asin: Returns the arcsine of a, expressed in radians, in the range [-pi/2,+pi/2], expecting a to be in the range [-1,+1].
//
static MATH_FORCEINLINE float4 asin(const float4 &x);
static MATH_FORCEINLINE float3 asin(const float3 &a);
static MATH_FORCEINLINE float2 asin(const float2 &a);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 asin(const float1 &a);
#endif
static MATH_FORCEINLINE float asin(float a);

//
// acos: Returns the arc cosine of a, expressed in radians, in the range [0,pi], expecting a to be in the range [-1,+1].
//
static MATH_FORCEINLINE float4 acos(const float4 &a);
static MATH_FORCEINLINE float3 acos(const float3 &a);
static MATH_FORCEINLINE float2 acos(const float2 &a);
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 acos(const float1 &a);
#endif
static MATH_FORCEINLINE float acos(float a);


/////////////////////////////////////////////////////////////////////
// Implementation

//
//  sinpi: Returns the sine of an angle of 'a', expressed in multiple of PI.
//      by example if you want to compute the sin of 90 degree you pass 1/2 (PI/2 = 90)
//  note: This function is more precise and faster than sin
//
static MATH_FORCEINLINE float4 sinpi(const float4 &a)
{
    float4 x = mad(.5f, a, -.25f); x = .25f - abs(x - round(x));
    return VEC_MATH_SIN_COS_POLY(x);
}

static MATH_FORCEINLINE float3 sinpi(const float3 &a)
{
    return sinpi(float4(as_native(a))).xyz;
}

static MATH_FORCEINLINE float2 sinpi(const float2 &a)
{
    return sinpi(float4(as_native(a))).xy;
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 sinpi(const float1 &a)
{
    return float1((float1::packed)sinpi(float4((float1::packed)a)));
}

#endif

//
//  cospi: Returns the cosine of an angle of 'a', expressed in multiple of PI.
//      by example if you want to compute the cos of 90 degree you pass 1/2 (PI/2 = 90)
//  note: This function is more precise and faster than cos
//
static MATH_FORCEINLINE float4 cospi(const float4 &a)
{
    float4 x = .5f * a; x = .25f - abs(x - round(x));
    return VEC_MATH_SIN_COS_POLY(x);
}

static MATH_FORCEINLINE float3 cospi(const float3 &a)
{
    return cospi(float4(as_native(a))).xyz;
}

static MATH_FORCEINLINE float2 cospi(const float2 &a)
{
    return cospi(float4(as_native(a))).xy;
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 cospi(const float1 &a)
{
    return float1((float1::packed)cospi(float4((float1::packed)a)));
}

#endif

//
//  sin: Returns the sine of an angle of 'a', expressed in radians.
//
static MATH_FORCEINLINE float4 sin(const float4 &a)
{
    float4 x = mad(one_over_two_pi(), a, -.25f); x = .25f - abs(x - round(x));
    return VEC_MATH_SIN_COS_POLY(x);
}

static MATH_FORCEINLINE float3 sin(const float3 &a)
{
    return sin(float4(as_native(a))).xyz;
}

static MATH_FORCEINLINE float2 sin(const float2 &a)
{
    return sin(float4(as_native(a))).xy;
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 sin(const float1 &a)
{
    return float1((float1::packed)sin(float4((float1::packed)a)));
}

#endif
static MATH_FORCEINLINE float sin(float x)
{
    return ::sinf(x);
}

//
//  cos: Returns the cosine of an angle of 'a', expressed in radians.
//
static MATH_FORCEINLINE float4 cos(const float4 &a)
{
    float4 x = one_over_two_pi() * a; x = .25f - abs(x - round(x));
    return VEC_MATH_SIN_COS_POLY(x);
}

static MATH_FORCEINLINE float3 cos(const float3 &a)
{
    return cos(float4(as_native(a))).xyz;
}

static MATH_FORCEINLINE float2 cos(const float2 &a)
{
    return cos(float4(as_native(a))).xy;
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 cos(const float1 &a)
{
    return float1((float1::packed)cos(float4((float1::packed)a)));
}

#endif
static MATH_FORCEINLINE float cos(float x)
{
    return ::cosf(x);
}

//
//  sincos: Returns the sine and cosine of an angle of 'v', expressed in radians.
//
static MATH_FORCEINLINE void sincos(const float4 &v, float4& s, float4& c)
{
    c = cos(v);
    s = sin(v);
}

static MATH_FORCEINLINE void sincos(const float3 &v, float3& s, float3& c)
{
    c = cos(v);
    s = sin(v);
}

static MATH_FORCEINLINE void sincos(const float2 &v, float2& s, float2& c)
{
    c = cos(v);
    s = sin(v);
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE void sincos(const float1 &v, float1& s, float1& c)
{
    float4 r = sincos_sssc(v);
    c = r.w;
    s = r.x;
}

#endif
static MATH_FORCEINLINE void sincos(float v, float& s, float& c)
{
    c = cos(v);
    s = sin(v);
}

//
//  sincos_sssc: Returns the sine and cosine of an angle of 'v', expressed in radians.
//  return value is float4(s,s,s,c)
//
#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float4 sincos_sssc(const float1 &a)
{
    float4 x = mad(float4(one_over_two_pi()), float4(a), float4(-.25f, -.25f, -.25f, 0.f)); x = .25f - abs(x - round(x));
    return VEC_MATH_SIN_COS_POLY(x);
}

#endif
static MATH_FORCEINLINE float4 sincos_sssc(float a)
{
    float4 x = mad(float4(one_over_two_pi()), float4(a), float4(-.25f, -.25f, -.25f, 0.f)); x = .25f - abs(x - round(x));
    return VEC_MATH_SIN_COS_POLY(x);
}

//
//  tan: Returns the tangent of an angle of 'v', expressed in radians.
//
static MATH_FORCEINLINE float4 tan(const float4 &v)
{
    float4 c, s;
    sincos(v, s, c);
    return s / c;
}

static MATH_FORCEINLINE float3 tan(const float3 &v)
{
    float3 c, s;
    sincos(v, s, c);
    return s / c;
}

static MATH_FORCEINLINE float2 tan(const float2 &v)
{
    float2 c, s;
    sincos(v, s, c);
    return s / c;
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 tan(const float1 &v)
{
    float4 r = sincos_sssc(v);
    return r.x / r.w;
}

#endif
static MATH_FORCEINLINE float tan(float v)
{
    float4 r = sincos_sssc(v);
    return r.x / r.w;
}

//
//  atan: Returns the arc tangent of an angle of 'a', expressed in radians.
//  note: Notice that because of the sign ambiguity, the function cannot determine
//        with certainty in which quadrant the angle falls only by its tangent value.
//        See atan2 for an alternative that takes a fractional argument instead.
//
static MATH_FORCEINLINE float4 atan(const float4 &x)
{
    int4 c = abs(x) > float4(1.f);
    float4 z = chgsign(math::pi_over_two(), x);
    float4 y = select(x, rcp(x), c);
    y = VEC_MATH_ATAN_POLY(y);
    return select(y, z - y, c);
}

static MATH_FORCEINLINE float3 atan(const float3 &a)
{
    return atan(float4(as_native(a))).xyz;
}

static MATH_FORCEINLINE float2 atan(const float2 &a)
{
    return atan(float4(as_native(a))).xy;
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 atan(const float1 &a)
{
    return float1((float1::packed)atan(float4((float1::packed)a)));
}

#endif
static MATH_FORCEINLINE float atan(float a)
{
    return ::atanf(a);
}

//
// atan2: Returns the arc tangent of y/x, expressed in radians.
// note: To compute the value, the function takes into account the sign of both arguments in order to determine the quadrant.
//
static MATH_FORCEINLINE float4 atan2(const float4 &y, const float4 &x)
{
    float4 z = atan(abs(y / x));
    return chgsign(select(z, math::pi() - z, as_int4(x)), y);
}

static MATH_FORCEINLINE float3 atan2(const float3 &y, const float3& x)
{
    return atan2(float4(as_native(y)), float4(as_native(x))).xyz;
}

static MATH_FORCEINLINE float2 atan2(const float2 &y, const float2& x)
{
    return atan2(float4(as_native(y)), float4(as_native(x))).xy;
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 atan2(const float1 &y, const float1 &x)
{
    return float1((float1::packed)atan2(float4((float1::packed)y), float4((float1::packed)x)));
}

#endif
static MATH_FORCEINLINE float atan2(float y, float x)
{
    return ::atan2f(y, x);
}

//
// asin: Returns the arcsine of a, expressed in radians, in the range [-pi/2,+pi/2], expecting a to be in the range [-1,+1].
//
static MATH_FORCEINLINE float4 asin(const float4 &y)
{
    float4 rx = rsqrt(1.f - y * y);
    float4 z = atan(abs(y * rx));
    return chgsign(z, y);
}

static MATH_FORCEINLINE float3 asin(const float3 &y)
{
    return asin(float4(as_native(y))).xyz;
}

static MATH_FORCEINLINE float2 asin(const float2 &y)
{
    return asin(float4(as_native(y))).xy;
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 asin(const float1 &y)
{
    return float1((float1::packed)asin(float4((float1::packed)y)));
}

#endif
static MATH_FORCEINLINE float asin(float y)
{
    return ::asinf(y);
}

//
// acos: Returns the arc cosine of a, expressed in radians, in the range [0,pi], expecting a to be in the range [-1,+1].
//
static MATH_FORCEINLINE float4 acos(const float4 &x)
{
    float4 y = sqrt(1.f - x * x);
    float4 z = atan(abs(y / x));
    return select(z, math::pi() - z, as_int4(x));
}

static MATH_FORCEINLINE float3 acos(const float3 &x)
{
    return acos(float4(as_native(x))).xyz;
}

static MATH_FORCEINLINE float2 acos(const float2 &x)
{
    return acos(float4(as_native(x))).xy;
}

#if defined(MATH_HAS_SIMD_FLOAT)
static MATH_FORCEINLINE float1 acos(const float1 &x)
{
    return float1((float1::packed)acos(float4((float1::packed)x)));
}

#endif
static MATH_FORCEINLINE float acos(float x)
{
    return ::acosf(x);
}
