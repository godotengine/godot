/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#include "SDL_sysaudio.h"

#include "SDL_audioresample.h"

// SDL's resampler uses a "bandlimited interpolation" algorithm:
//     https://ccrma.stanford.edu/~jos/resample/

// TODO: Support changing this at runtime?
#if defined(SDL_SSE_INTRINSICS) || defined(SDL_NEON_INTRINSICS)
// In <current year>, SSE is basically mandatory anyway
// We want RESAMPLER_SAMPLES_PER_FRAME to be a multiple of 4, to make SIMD easier
#define RESAMPLER_ZERO_CROSSINGS 6
#else
#define RESAMPLER_ZERO_CROSSINGS 5
#endif

#define RESAMPLER_SAMPLES_PER_FRAME (RESAMPLER_ZERO_CROSSINGS * 2)

// For a given srcpos, `srcpos + frame` are sampled, where `-RESAMPLER_ZERO_CROSSINGS < frame <= RESAMPLER_ZERO_CROSSINGS`.
// Note, when upsampling, it is also possible to start sampling from `srcpos = -1`.
#define RESAMPLER_MAX_PADDING_FRAMES (RESAMPLER_ZERO_CROSSINGS + 1)

// More bits gives more precision, at the cost of a larger table.
#define RESAMPLER_BITS_PER_ZERO_CROSSING    3
#define RESAMPLER_SAMPLES_PER_ZERO_CROSSING (1 << RESAMPLER_BITS_PER_ZERO_CROSSING)
#define RESAMPLER_FILTER_INTERP_BITS        (32 - RESAMPLER_BITS_PER_ZERO_CROSSING)
#define RESAMPLER_FILTER_INTERP_RANGE       (1 << RESAMPLER_FILTER_INTERP_BITS)

// ResampleFrame is just a vector/matrix/matrix multiplication.
// It performs cubic interpolation of the filter, then multiplies that with the input.
// dst = [1, frac, frac^2, frac^3] * filter * src

// Cubic Polynomial
typedef union Cubic
{
    float v[4];

#ifdef SDL_SSE_INTRINSICS
    // Aligned loads can be used directly as memory operands for mul/add
    __m128 v128;
#endif

#ifdef SDL_NEON_INTRINSICS
    float32x4_t v128;
#endif

} Cubic;

static void ResampleFrame_Generic(const float *src, float *dst, const Cubic *filter, float frac, int chans)
{
    const float frac2 = frac * frac;
    const float frac3 = frac * frac2;

    int i, chan;
    float scales[RESAMPLER_SAMPLES_PER_FRAME];

    for (i = 0; i < RESAMPLER_SAMPLES_PER_FRAME; ++i, ++filter) {
        scales[i] = filter->v[0] + (filter->v[1] * frac) + (filter->v[2] * frac2) + (filter->v[3] * frac3);
    }

    for (chan = 0; chan < chans; ++chan) {
        float out = 0.0f;

        for (i = 0; i < RESAMPLER_SAMPLES_PER_FRAME; ++i) {
            out += src[i * chans + chan] * scales[i];
        }

        dst[chan] = out;
    }
}

static void ResampleFrame_Mono(const float *src, float *dst, const Cubic *filter, float frac, int chans)
{
    const float frac2 = frac * frac;
    const float frac3 = frac * frac2;

    int i;
    float out = 0.0f;

    for (i = 0; i < RESAMPLER_SAMPLES_PER_FRAME; ++i, ++filter) {
        // Interpolate between the nearest two filters
        const float scale = filter->v[0] + (filter->v[1] * frac) + (filter->v[2] * frac2) + (filter->v[3] * frac3);

        out += src[i] * scale;
    }

    dst[0] = out;
}

static void ResampleFrame_Stereo(const float *src, float *dst, const Cubic *filter, float frac, int chans)
{
    const float frac2 = frac * frac;
    const float frac3 = frac * frac2;

    int i;
    float out0 = 0.0f;
    float out1 = 0.0f;

    for (i = 0; i < RESAMPLER_SAMPLES_PER_FRAME; ++i, ++filter) {
        // Interpolate between the nearest two filters
        const float scale = filter->v[0] + (filter->v[1] * frac) + (filter->v[2] * frac2) + (filter->v[3] * frac3);

        out0 += src[i * 2 + 0] * scale;
        out1 += src[i * 2 + 1] * scale;
    }

    dst[0] = out0;
    dst[1] = out1;
}

#ifdef SDL_SSE_INTRINSICS
#define sdl_madd_ps(a, b, c) _mm_add_ps(a, _mm_mul_ps(b, c)) // Not-so-fused multiply-add

static void SDL_TARGETING("sse") ResampleFrame_Generic_SSE(const float *src, float *dst, const Cubic *filter, float frac, int chans)
{
#if RESAMPLER_SAMPLES_PER_FRAME != 12
#error Invalid samples per frame
#endif

    __m128 f0, f1, f2;

    {
        const __m128 frac1 = _mm_set1_ps(frac);
        const __m128 frac2 = _mm_mul_ps(frac1, frac1);
        const __m128 frac3 = _mm_mul_ps(frac1, frac2);

// Transposed in SetupAudioResampler
// Explicitly use _mm_load_ps to workaround ICE in GCC 4.9.4 accessing Cubic.v128
#define X(out)                                               \
    out = _mm_load_ps(filter[0].v);                          \
    out = sdl_madd_ps(out, frac1, _mm_load_ps(filter[1].v)); \
    out = sdl_madd_ps(out, frac2, _mm_load_ps(filter[2].v)); \
    out = sdl_madd_ps(out, frac3, _mm_load_ps(filter[3].v)); \
    filter += 4

        X(f0);
        X(f1);
        X(f2);

#undef X
    }

    if (chans == 2) {
        // Duplicate each of the filter elements and multiply by the input
        // Use two accumulators to improve throughput
        __m128 out0 = _mm_mul_ps(_mm_loadu_ps(src + 0), _mm_unpacklo_ps(f0, f0));
        __m128 out1 = _mm_mul_ps(_mm_loadu_ps(src + 4), _mm_unpackhi_ps(f0, f0));
        out0 = sdl_madd_ps(out0, _mm_loadu_ps(src + 8), _mm_unpacklo_ps(f1, f1));
        out1 = sdl_madd_ps(out1, _mm_loadu_ps(src + 12), _mm_unpackhi_ps(f1, f1));
        out0 = sdl_madd_ps(out0, _mm_loadu_ps(src + 16), _mm_unpacklo_ps(f2, f2));
        out1 = sdl_madd_ps(out1, _mm_loadu_ps(src + 20), _mm_unpackhi_ps(f2, f2));

        // Add the accumulators together
        __m128 out = _mm_add_ps(out0, out1);

        // Add the lower and upper pairs together
        out = _mm_add_ps(out, _mm_movehl_ps(out, out));

        // Store the result
        _mm_storel_pi((__m64 *)dst, out);
        return;
    }

    if (chans == 1) {
        // Multiply the filter by the input
        __m128 out = _mm_mul_ps(f0, _mm_loadu_ps(src + 0));
        out = sdl_madd_ps(out, f1, _mm_loadu_ps(src + 4));
        out = sdl_madd_ps(out, f2, _mm_loadu_ps(src + 8));

        // Horizontal sum
        __m128 shuf = _mm_shuffle_ps(out, out, _MM_SHUFFLE(2, 3, 0, 1));
        out = _mm_add_ps(out, shuf);
        out = _mm_add_ss(out, _mm_movehl_ps(shuf, out));

        _mm_store_ss(dst, out);
        return;
    }

    int chan = 0;

    // Process 4 channels at once
    for (; chan + 4 <= chans; chan += 4) {
        const float *in = &src[chan];
        __m128 out0 = _mm_setzero_ps();
        __m128 out1 = _mm_setzero_ps();

#define X(a, b, out)                                                                         \
    out = sdl_madd_ps(out, _mm_loadu_ps(in), _mm_shuffle_ps(a, a, _MM_SHUFFLE(b, b, b, b))); \
    in += chans

#define Y(a)       \
    X(a, 0, out0); \
    X(a, 1, out1); \
    X(a, 2, out0); \
    X(a, 3, out1)

        Y(f0);
        Y(f1);
        Y(f2);

#undef X
#undef Y

        // Add the accumulators together
        __m128 out = _mm_add_ps(out0, out1);

        _mm_storeu_ps(&dst[chan], out);
    }

    // Process the remaining channels one at a time.
    // Channel counts 1,2,4,8 are already handled above, leaving 3,5,6,7 to deal with (looping 3,1,2,3 times).
    // Without vgatherdps (AVX2), this gets quite messy.
    for (; chan < chans; ++chan) {
        const float *in = &src[chan];
        __m128 v0, v1, v2;

#define X(x)                                                                         \
    x = _mm_unpacklo_ps(_mm_load_ss(in), _mm_load_ss(in + chans));                   \
    in += chans + chans;                                                             \
    x = _mm_movelh_ps(x, _mm_unpacklo_ps(_mm_load_ss(in), _mm_load_ss(in + chans))); \
    in += chans + chans

        X(v0);
        X(v1);
        X(v2);

#undef X

        __m128 out = _mm_mul_ps(f0, v0);
        out = sdl_madd_ps(out, f1, v1);
        out = sdl_madd_ps(out, f2, v2);

        // Horizontal sum
        __m128 shuf = _mm_shuffle_ps(out, out, _MM_SHUFFLE(2, 3, 0, 1));
        out = _mm_add_ps(out, shuf);
        out = _mm_add_ss(out, _mm_movehl_ps(shuf, out));

        _mm_store_ss(&dst[chan], out);
    }
}

#undef sdl_madd_ps
#endif

#ifdef SDL_NEON_INTRINSICS
static void ResampleFrame_Generic_NEON(const float *src, float *dst, const Cubic *filter, float frac, int chans)
{
#if RESAMPLER_SAMPLES_PER_FRAME != 12
#error Invalid samples per frame
#endif

    float32x4_t f0, f1, f2;

    {
        const float32x4_t frac1 = vdupq_n_f32(frac);
        const float32x4_t frac2 = vmulq_f32(frac1, frac1);
        const float32x4_t frac3 = vmulq_f32(frac1, frac2);

// Transposed in SetupAudioResampler
#define X(out)                                                                                                                  \
    out = vmlaq_f32(vmlaq_f32(vmlaq_f32(filter[0].v128, filter[1].v128, frac1), filter[2].v128, frac2), filter[3].v128, frac3); \
    filter += 4

        X(f0);
        X(f1);
        X(f2);

#undef X
    }

    if (chans == 2) {
        float32x4x2_t g0 = vzipq_f32(f0, f0);
        float32x4x2_t g1 = vzipq_f32(f1, f1);
        float32x4x2_t g2 = vzipq_f32(f2, f2);

        // Duplicate each of the filter elements and multiply by the input
        // Use two accumulators to improve throughput
        float32x4_t out0 = vmulq_f32(vld1q_f32(src + 0), g0.val[0]);
        float32x4_t out1 = vmulq_f32(vld1q_f32(src + 4), g0.val[1]);
        out0 = vmlaq_f32(out0, vld1q_f32(src + 8), g1.val[0]);
        out1 = vmlaq_f32(out1, vld1q_f32(src + 12), g1.val[1]);
        out0 = vmlaq_f32(out0, vld1q_f32(src + 16), g2.val[0]);
        out1 = vmlaq_f32(out1, vld1q_f32(src + 20), g2.val[1]);

        // Add the accumulators together
        out0 = vaddq_f32(out0, out1);

        // Add the lower and upper pairs together
        float32x2_t out = vadd_f32(vget_low_f32(out0), vget_high_f32(out0));

        // Store the result
        vst1_f32(dst, out);
        return;
    }

    if (chans == 1) {
        // Multiply the filter by the input
        float32x4_t out = vmulq_f32(f0, vld1q_f32(src + 0));
        out = vmlaq_f32(out, f1, vld1q_f32(src + 4));
        out = vmlaq_f32(out, f2, vld1q_f32(src + 8));

        // Horizontal sum
        float32x2_t sum = vadd_f32(vget_low_f32(out), vget_high_f32(out));
        sum = vpadd_f32(sum, sum);

        vst1_lane_f32(dst, sum, 0);
        return;
    }

    int chan = 0;

    // Process 4 channels at once
    for (; chan + 4 <= chans; chan += 4) {
        const float *in = &src[chan];
        float32x4_t out0 = vdupq_n_f32(0);
        float32x4_t out1 = vdupq_n_f32(0);

#define X(a, b, out)                                           \
    out = vmlaq_f32(out, vld1q_f32(in), vdupq_lane_f32(a, b)); \
    in += chans

#define Y(a)                      \
    X(vget_low_f32(a), 0, out0);  \
    X(vget_low_f32(a), 1, out1);  \
    X(vget_high_f32(a), 0, out0); \
    X(vget_high_f32(a), 1, out1)

        Y(f0);
        Y(f1);
        Y(f2);

#undef X
#undef Y

        // Add the accumulators together
        float32x4_t out = vaddq_f32(out0, out1);

        vst1q_f32(&dst[chan], out);
    }

    // Process the remaining channels one at a time.
    // Channel counts 1,2,4,8 are already handled above, leaving 3,5,6,7 to deal with (looping 3,1,2,3 times).
    for (; chan < chans; ++chan) {
        const float *in = &src[chan];
        float32x4_t v0, v1, v2;

#define X(x)                      \
    x = vld1q_dup_f32(in);        \
    in += chans;                  \
    x = vld1q_lane_f32(in, x, 1); \
    in += chans;                  \
    x = vld1q_lane_f32(in, x, 2); \
    in += chans;                  \
    x = vld1q_lane_f32(in, x, 3); \
    in += chans

        X(v0);
        X(v1);
        X(v2);

#undef X

        float32x4_t out = vmulq_f32(f0, v0);
        out = vmlaq_f32(out, f1, v1);
        out = vmlaq_f32(out, f2, v2);

        // Horizontal sum
        float32x2_t sum = vadd_f32(vget_low_f32(out), vget_high_f32(out));
        sum = vpadd_f32(sum, sum);

        vst1_lane_f32(&dst[chan], sum, 0);
    }
}
#endif

// Calculate the cubic equation which passes through all four points.
// https://en.wikipedia.org/wiki/Ordinary_least_squares
// https://en.wikipedia.org/wiki/Polynomial_regression
static void CubicLeastSquares(Cubic *coeffs, float y0, float y1, float y2, float y3)
{
    // Least squares matrix for xs = [0, 1/3, 2/3, 1]
    // [  1.0   0.0   0.0  0.0 ]
    // [ -5.5   9.0  -4.5  1.0 ]
    // [  9.0 -22.5  18.0 -4.5 ]
    // [ -4.5  13.5 -13.5  4.5 ]

    coeffs->v[0] = y0;
    coeffs->v[1] = -5.5f * y0 + 9.0f * y1 - 4.5f * y2 + y3;
    coeffs->v[2] = 9.0f * y0 - 22.5f * y1 + 18.0f * y2 - 4.5f * y3;
    coeffs->v[3] = -4.5f * y0 + 13.5f * y1 - 13.5f * y2 + 4.5f * y3;
}

// Zeroth-order modified Bessel function of the first kind
// https://mathworld.wolfram.com/ModifiedBesselFunctionoftheFirstKind.html
static float BesselI0(float x)
{
    float sum = 0.0f;
    float i = 1.0f;
    float t = 1.0f;
    x *= x * 0.25f;

    while (t >= sum * SDL_FLT_EPSILON) {
        sum += t;
        t *= x / (i * i);
        ++i;
    }

    return sum;
}

// Pre-calculate 180 degrees of sin(pi * x) / pi
// The speedup from this isn't huge, but it also avoids precision issues.
// If sinf isn't available, SDL_sinf just calls SDL_sin.
// Know what SDL_sin(SDL_PI_F) equals? Not quite zero.
static void SincTable(float *table, int len)
{
    int i;

    for (i = 0; i < len; ++i) {
        table[i] = SDL_sinf(i * (SDL_PI_F / len)) / SDL_PI_F;
    }
}

// Calculate Sinc(x/y), using a lookup table
static float Sinc(const float *table, int x, int y)
{
    float s = table[x % y];
    s = ((x / y) & 1) ? -s : s;
    return (s * y) / x;
}

static Cubic ResamplerFilter[RESAMPLER_SAMPLES_PER_ZERO_CROSSING][RESAMPLER_SAMPLES_PER_FRAME];

static void GenerateResamplerFilter(void)
{
    enum
    {
        // Generate samples at 3x the target resolution, so that we have samples at [0, 1/3, 2/3, 1] of each position
        TABLE_SAMPLES_PER_ZERO_CROSSING = RESAMPLER_SAMPLES_PER_ZERO_CROSSING * 3,
        TABLE_SIZE = RESAMPLER_ZERO_CROSSINGS * TABLE_SAMPLES_PER_ZERO_CROSSING,
    };

    // if dB > 50, beta=(0.1102 * (dB - 8.7)), according to Matlab.
    const float dB = 80.0f;
    const float beta = 0.1102f * (dB - 8.7f);
    const float bessel_beta = BesselI0(beta);
    const float lensqr = TABLE_SIZE * TABLE_SIZE;

    int i, j;

    float sinc[TABLE_SAMPLES_PER_ZERO_CROSSING];
    SincTable(sinc, TABLE_SAMPLES_PER_ZERO_CROSSING);

    // Generate one wing of the filter
    // https://en.wikipedia.org/wiki/Kaiser_window
    // https://en.wikipedia.org/wiki/Whittaker%E2%80%93Shannon_interpolation_formula
    float filter[TABLE_SIZE + 1];
    filter[0] = 1.0f;

    for (i = 1; i <= TABLE_SIZE; ++i) {
        float b = BesselI0(beta * SDL_sqrtf((lensqr - (i * i)) / lensqr)) / bessel_beta;
        float s = Sinc(sinc, i, TABLE_SAMPLES_PER_ZERO_CROSSING);
        filter[i] = b * s;
    }

    // Generate the coefficients for each point
    // When interpolating, the fraction represents how far we are between input samples,
    // so we need to align the filter by "moving" it to the right.
    //
    // For the left wing, this means interpolating "forwards" (away from the center)
    // For the right wing, this means interpolating "backwards" (towards the center)
    //
    // The center of the filter is at the end of the left wing (RESAMPLER_ZERO_CROSSINGS - 1)
    // The left wing is the filter, but reversed
    // The right wing is the filter, but offset by 1
    //
    // Since the right wing is offset by 1, this just means we interpolate backwards
    // between the same points, instead of forwards
    // interp(p[n], p[n+1], t) = interp(p[n+1], p[n+1-1], 1 - t) = interp(p[n+1], p[n], 1 - t)
    for (i = 0; i < RESAMPLER_SAMPLES_PER_ZERO_CROSSING; ++i) {
        for (j = 0; j < RESAMPLER_ZERO_CROSSINGS; ++j) {
            const float *ys = &filter[((j * RESAMPLER_SAMPLES_PER_ZERO_CROSSING) + i) * 3];

            Cubic *fwd = &ResamplerFilter[i][RESAMPLER_ZERO_CROSSINGS - j - 1];
            Cubic *rev = &ResamplerFilter[RESAMPLER_SAMPLES_PER_ZERO_CROSSING - i - 1][RESAMPLER_ZERO_CROSSINGS + j];

            // Calculate the cubic equation of the 4 points
            CubicLeastSquares(fwd, ys[0], ys[1], ys[2], ys[3]);
            CubicLeastSquares(rev, ys[3], ys[2], ys[1], ys[0]);
        }
    }
}

typedef void (*ResampleFrameFunc)(const float *src, float *dst, const Cubic *filter, float frac, int chans);
static ResampleFrameFunc ResampleFrame[8];

// Transpose 4x4 floats
static void Transpose4x4(Cubic *data)
{
    int i, j;

    Cubic temp[4] = { data[0], data[1], data[2], data[3] };

    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
            data[i].v[j] = temp[j].v[i];
        }
    }
}

static void SetupAudioResampler(void)
{
    int i, j;
    bool transpose = false;

    GenerateResamplerFilter();

#ifdef SDL_SSE_INTRINSICS
    if (SDL_HasSSE()) {
        for (i = 0; i < 8; ++i) {
            ResampleFrame[i] = ResampleFrame_Generic_SSE;
        }
        transpose = true;
    } else
#endif
#ifdef SDL_NEON_INTRINSICS
    if (SDL_HasNEON()) {
        for (i = 0; i < 8; ++i) {
            ResampleFrame[i] = ResampleFrame_Generic_NEON;
        }
        transpose = true;
    } else
#endif
    {
        for (i = 0; i < 8; ++i) {
            ResampleFrame[i] = ResampleFrame_Generic;
        }

        ResampleFrame[0] = ResampleFrame_Mono;
        ResampleFrame[1] = ResampleFrame_Stereo;
    }

    if (transpose) {
        // Transpose each set of 4 coefficients, to reduce work when resampling
        for (i = 0; i < RESAMPLER_SAMPLES_PER_ZERO_CROSSING; ++i) {
            for (j = 0; j + 4 <= RESAMPLER_SAMPLES_PER_FRAME; j += 4) {
                Transpose4x4(&ResamplerFilter[i][j]);
            }
        }
    }
}

void SDL_SetupAudioResampler(void)
{
    static SDL_InitState init;

    if (SDL_ShouldInit(&init)) {
        SetupAudioResampler();
        SDL_SetInitialized(&init, true);
    }
}

Sint64 SDL_GetResampleRate(int src_rate, int dst_rate)
{
    SDL_assert(src_rate > 0);
    SDL_assert(dst_rate > 0);

    Sint64 numerator = (Sint64)src_rate << 32;
    Sint64 denominator = (Sint64)dst_rate;

    // Generally it's expected that `dst_frames = (src_frames * dst_rate) / src_rate`
    // To match this as closely as possible without infinite precision, always round up the resample rate.
    // For example, without rounding up, a sample ratio of 2:3 would have `sample_rate = 0xAAAAAAAA`
    // After 3 frames, the position would be 0x1.FFFFFFFE, meaning we haven't fully consumed the second input frame.
    // By rounding up to 0xAAAAAAAB, we would instead reach 0x2.00000001, fulling consuming the second frame.
    // Technically you could say this is kicking the can 0x100000000 steps down the road, but I'm fine with that :)
    // sample_rate = div_ceil(numerator, denominator)
    Sint64 sample_rate = ((numerator - 1) / denominator) + 1;

    SDL_assert(sample_rate > 0);

    return sample_rate;
}

int SDL_GetResamplerHistoryFrames(void)
{
    // Even if we aren't currently resampling, make sure to keep enough history in case we need to later.

    return RESAMPLER_MAX_PADDING_FRAMES;
}

int SDL_GetResamplerPaddingFrames(Sint64 resample_rate)
{
    // This must always be <= SDL_GetResamplerHistoryFrames()

    return resample_rate ? RESAMPLER_MAX_PADDING_FRAMES : 0;
}

// These are not general purpose. They do not check for all possible underflow/overflow
SDL_FORCE_INLINE bool ResamplerAdd(Sint64 a, Sint64 b, Sint64 *ret)
{
    if ((b > 0) && (a > SDL_MAX_SINT64 - b)) {
        return false;
    }

    *ret = a + b;
    return true;
}

SDL_FORCE_INLINE bool ResamplerMul(Sint64 a, Sint64 b, Sint64 *ret)
{
    if ((b > 0) && (a > SDL_MAX_SINT64 / b)) {
        return false;
    }

    *ret = a * b;
    return true;
}

Sint64 SDL_GetResamplerInputFrames(Sint64 output_frames, Sint64 resample_rate, Sint64 resample_offset)
{
    // Calculate the index of the last input frame, then add 1.
    // ((((output_frames - 1) * resample_rate) + resample_offset) >> 32) + 1

    Sint64 output_offset;
    if (!ResamplerMul(output_frames, resample_rate, &output_offset) ||
        !ResamplerAdd(output_offset, -resample_rate + resample_offset + 0x100000000, &output_offset)) {
        output_offset = SDL_MAX_SINT64;
    }

    Sint64 input_frames = (Sint64)(Sint32)(output_offset >> 32);
    input_frames = SDL_max(input_frames, 0);

    return input_frames;
}

Sint64 SDL_GetResamplerOutputFrames(Sint64 input_frames, Sint64 resample_rate, Sint64 *inout_resample_offset)
{
    Sint64 resample_offset = *inout_resample_offset;

    // input_offset = (input_frames << 32) - resample_offset;
    Sint64 input_offset;
    if (!ResamplerMul(input_frames, 0x100000000, &input_offset) ||
        !ResamplerAdd(input_offset, -resample_offset, &input_offset)) {
        input_offset = SDL_MAX_SINT64;
    }

    // output_frames = div_ceil(input_offset, resample_rate)
    Sint64 output_frames = (input_offset > 0) ? ((input_offset - 1) / resample_rate) + 1 : 0;

    *inout_resample_offset = (output_frames * resample_rate) - input_offset;

    return output_frames;
}

void SDL_ResampleAudio(int chans, const float *src, int inframes, float *dst, int outframes,
                       Sint64 resample_rate, Sint64 *inout_resample_offset)
{
    int i;
    Sint64 srcpos = *inout_resample_offset;
    ResampleFrameFunc resample_frame = ResampleFrame[chans - 1];

    SDL_assert(resample_rate > 0);

    src -= (RESAMPLER_ZERO_CROSSINGS - 1) * chans;

    for (i = 0; i < outframes; ++i) {
        int srcindex = (int)(Sint32)(srcpos >> 32);
        Uint32 srcfraction = (Uint32)(srcpos & 0xFFFFFFFF);
        srcpos += resample_rate;

        SDL_assert(srcindex >= -1 && srcindex < inframes);

        const Cubic *filter = ResamplerFilter[srcfraction >> RESAMPLER_FILTER_INTERP_BITS];
        const float frac = (float)(srcfraction & (RESAMPLER_FILTER_INTERP_RANGE - 1)) * (1.0f / RESAMPLER_FILTER_INTERP_RANGE);

        const float *frame = &src[srcindex * chans];
        resample_frame(frame, dst, filter, frac, chans);

        dst += chans;
    }

    *inout_resample_offset = srcpos - ((Sint64)inframes << 32);
}
