// Copyright(c) 2021 Björn Ottosson
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files(the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and /or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions :
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef OK_COLOR_H
#define OK_COLOR_H

#include <cmath>
#include <cfloat>

class ok_color
{
public:

struct Lab { float L; float a; float b; };
struct RGB { float r; float g; float b; };
struct HSV { float h; float s; float v; };
struct HSL { float h; float s; float l; };
struct LC { float L; float C; };

// Alternative representation of (L_cusp, C_cusp)
// Encoded so S = C_cusp/L_cusp and T = C_cusp/(1-L_cusp)
// The maximum value for C in the triangle is then found as fmin(S*L, T*(1-L)), for a given L
struct ST { float S; float T; };

static constexpr float pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062f;

static float clamp(float x, float min, float max)
{
	if (x < min)
		return min;
	if (x > max)
		return max;

	return x;
}

static float sgn(float x)
{
	return (float)(0.f < x) - (float)(x < 0.f);
}

static float srgb_transfer_function(float a)
{
	return .0031308f >= a ? 12.92f * a : 1.055f * powf(a, .4166666666666667f) - .055f;
}

static float srgb_transfer_function_inv(float a)
{
	return .04045f < a ? powf((a + .055f) / 1.055f, 2.4f) : a / 12.92f;
}

static Lab linear_srgb_to_oklab(RGB c)
{
	float l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b;
	float m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b;
	float s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b;

	float l_ = cbrtf(l);
	float m_ = cbrtf(m);
	float s_ = cbrtf(s);

	return {
		0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
		1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
		0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_,
	};
}

static RGB oklab_to_linear_srgb(Lab c)
{
	float l_ = c.L + 0.3963377774f * c.a + 0.2158037573f * c.b;
	float m_ = c.L - 0.1055613458f * c.a - 0.0638541728f * c.b;
	float s_ = c.L - 0.0894841775f * c.a - 1.2914855480f * c.b;

	float l = l_ * l_ * l_;
	float m = m_ * m_ * m_;
	float s = s_ * s_ * s_;

	return {
		+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
		-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
		-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s,
	};
}

// Finds the maximum saturation possible for a given hue that fits in sRGB
// Saturation here is defined as S = C/L
// a and b must be normalized so a^2 + b^2 == 1
static float compute_max_saturation(float a, float b)
{
	// Max saturation will be when one of r, g or b goes below zero.

	// Select different coefficients depending on which component goes below zero first
	float k0, k1, k2, k3, k4, wl, wm, ws;

	if (-1.88170328f * a - 0.80936493f * b > 1)
	{
		// Red component
		k0 = +1.19086277f; k1 = +1.76576728f; k2 = +0.59662641f; k3 = +0.75515197f; k4 = +0.56771245f;
		wl = +4.0767416621f; wm = -3.3077115913f; ws = +0.2309699292f;
	}
	else if (1.81444104f * a - 1.19445276f * b > 1)
	{
		// Green component
		k0 = +0.73956515f; k1 = -0.45954404f; k2 = +0.08285427f; k3 = +0.12541070f; k4 = +0.14503204f;
		wl = -1.2684380046f; wm = +2.6097574011f; ws = -0.3413193965f;
	}
	else
	{
		// Blue component
		k0 = +1.35733652f; k1 = -0.00915799f; k2 = -1.15130210f; k3 = -0.50559606f; k4 = +0.00692167f;
		wl = -0.0041960863f; wm = -0.7034186147f; ws = +1.7076147010f;
	}

	// Approximate max saturation using a polynomial:
	float S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

	// Do one step Halley's method to get closer
	// this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
	// this should be sufficient for most applications, otherwise do two/three steps

	float k_l = +0.3963377774f * a + 0.2158037573f * b;
	float k_m = -0.1055613458f * a - 0.0638541728f * b;
	float k_s = -0.0894841775f * a - 1.2914855480f * b;

	{
		float l_ = 1.f + S * k_l;
		float m_ = 1.f + S * k_m;
		float s_ = 1.f + S * k_s;

		float l = l_ * l_ * l_;
		float m = m_ * m_ * m_;
		float s = s_ * s_ * s_;

		float l_dS = 3.f * k_l * l_ * l_;
		float m_dS = 3.f * k_m * m_ * m_;
		float s_dS = 3.f * k_s * s_ * s_;

		float l_dS2 = 6.f * k_l * k_l * l_;
		float m_dS2 = 6.f * k_m * k_m * m_;
		float s_dS2 = 6.f * k_s * k_s * s_;

		float f = wl * l + wm * m + ws * s;
		float f1 = wl * l_dS + wm * m_dS + ws * s_dS;
		float f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2;

		S = S - f * f1 / (f1 * f1 - 0.5f * f * f2);
	}

	return S;
}

// finds L_cusp and C_cusp for a given hue
// a and b must be normalized so a^2 + b^2 == 1
static LC find_cusp(float a, float b)
{
	// First, find the maximum saturation (saturation S = C/L)
	float S_cusp = compute_max_saturation(a, b);

	// Convert to linear sRGB to find the first point where at least one of r,g or b >= 1:
	RGB rgb_at_max = oklab_to_linear_srgb({ 1, S_cusp * a, S_cusp * b });
	float L_cusp = cbrtf(1.f / fmax(fmax(rgb_at_max.r, rgb_at_max.g), rgb_at_max.b));
	float C_cusp = L_cusp * S_cusp;

	return { L_cusp , C_cusp };
}

// Finds intersection of the line defined by
// L = L0 * (1 - t) + t * L1;
// C = t * C1;
// a and b must be normalized so a^2 + b^2 == 1
static float find_gamut_intersection(float a, float b, float L1, float C1, float L0, LC cusp)
{
	// Find the intersection for upper and lower half seprately
	float t;
	if (((L1 - L0) * cusp.C - (cusp.L - L0) * C1) <= 0.f)
	{
		// Lower half

		t = cusp.C * L0 / (C1 * cusp.L + cusp.C * (L0 - L1));
	}
	else
	{
		// Upper half

		// First intersect with triangle
		t = cusp.C * (L0 - 1.f) / (C1 * (cusp.L - 1.f) + cusp.C * (L0 - L1));

		// Then one step Halley's method
		{
			float dL = L1 - L0;
			float dC = C1;

			float k_l = +0.3963377774f * a + 0.2158037573f * b;
			float k_m = -0.1055613458f * a - 0.0638541728f * b;
			float k_s = -0.0894841775f * a - 1.2914855480f * b;

			float l_dt = dL + dC * k_l;
			float m_dt = dL + dC * k_m;
			float s_dt = dL + dC * k_s;


			// If higher accuracy is required, 2 or 3 iterations of the following block can be used:
			{
				float L = L0 * (1.f - t) + t * L1;
				float C = t * C1;

				float l_ = L + C * k_l;
				float m_ = L + C * k_m;
				float s_ = L + C * k_s;

				float l = l_ * l_ * l_;
				float m = m_ * m_ * m_;
				float s = s_ * s_ * s_;

				float ldt = 3 * l_dt * l_ * l_;
				float mdt = 3 * m_dt * m_ * m_;
				float sdt = 3 * s_dt * s_ * s_;

				float ldt2 = 6 * l_dt * l_dt * l_;
				float mdt2 = 6 * m_dt * m_dt * m_;
				float sdt2 = 6 * s_dt * s_dt * s_;

				float r = 4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s - 1;
				float r1 = 4.0767416621f * ldt - 3.3077115913f * mdt + 0.2309699292f * sdt;
				float r2 = 4.0767416621f * ldt2 - 3.3077115913f * mdt2 + 0.2309699292f * sdt2;

				float u_r = r1 / (r1 * r1 - 0.5f * r * r2);
				float t_r = -r * u_r;

				float g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s - 1;
				float g1 = -1.2684380046f * ldt + 2.6097574011f * mdt - 0.3413193965f * sdt;
				float g2 = -1.2684380046f * ldt2 + 2.6097574011f * mdt2 - 0.3413193965f * sdt2;

				float u_g = g1 / (g1 * g1 - 0.5f * g * g2);
				float t_g = -g * u_g;

				b = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s - 1;
				float b1 = -0.0041960863f * ldt - 0.7034186147f * mdt + 1.7076147010f * sdt;
				float b2 = -0.0041960863f * ldt2 - 0.7034186147f * mdt2 + 1.7076147010f * sdt2;

				float u_b = b1 / (b1 * b1 - 0.5f * b * b2);
				float t_b = -b * u_b;

				t_r = u_r >= 0.f ? t_r : FLT_MAX;
				t_g = u_g >= 0.f ? t_g : FLT_MAX;
				t_b = u_b >= 0.f ? t_b : FLT_MAX;

				t += fmin(t_r, fmin(t_g, t_b));
			}
		}
	}

	return t;
}

static float find_gamut_intersection(float a, float b, float L1, float C1, float L0)
{
	// Find the cusp of the gamut triangle
	LC cusp = find_cusp(a, b);

	return find_gamut_intersection(a, b, L1, C1, L0, cusp);
}

static RGB gamut_clip_preserve_chroma(RGB rgb)
{
	if (rgb.r < 1 && rgb.g < 1 && rgb.b < 1 && rgb.r > 0 && rgb.g > 0 && rgb.b > 0)
		return rgb;

	Lab lab = linear_srgb_to_oklab(rgb);

	float L = lab.L;
	float eps = 0.00001f;
	float C = fmax(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
	float a_ = lab.a / C;
	float b_ = lab.b / C;

	float L0 = clamp(L, 0, 1);

	float t = find_gamut_intersection(a_, b_, L, C, L0);
	float L_clipped = L0 * (1 - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb({ L_clipped, C_clipped * a_, C_clipped * b_ });
}

static RGB gamut_clip_project_to_0_5(RGB rgb)
{
	if (rgb.r < 1 && rgb.g < 1 && rgb.b < 1 && rgb.r > 0 && rgb.g > 0 && rgb.b > 0)
		return rgb;

	Lab lab = linear_srgb_to_oklab(rgb);

	float L = lab.L;
	float eps = 0.00001f;
	float C = fmax(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
	float a_ = lab.a / C;
	float b_ = lab.b / C;

	float L0 = 0.5;

	float t = find_gamut_intersection(a_, b_, L, C, L0);
	float L_clipped = L0 * (1 - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb({ L_clipped, C_clipped * a_, C_clipped * b_ });
}

static RGB gamut_clip_project_to_L_cusp(RGB rgb)
{
	if (rgb.r < 1 && rgb.g < 1 && rgb.b < 1 && rgb.r > 0 && rgb.g > 0 && rgb.b > 0)
		return rgb;

	Lab lab = linear_srgb_to_oklab(rgb);

	float L = lab.L;
	float eps = 0.00001f;
	float C = fmax(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
	float a_ = lab.a / C;
	float b_ = lab.b / C;

	// The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
	LC cusp = find_cusp(a_, b_);

	float L0 = cusp.L;

	float t = find_gamut_intersection(a_, b_, L, C, L0);

	float L_clipped = L0 * (1 - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb({ L_clipped, C_clipped * a_, C_clipped * b_ });
}

static RGB gamut_clip_adaptive_L0_0_5(RGB rgb, float alpha = 0.05f)
{
	if (rgb.r < 1 && rgb.g < 1 && rgb.b < 1 && rgb.r > 0 && rgb.g > 0 && rgb.b > 0)
		return rgb;

	Lab lab = linear_srgb_to_oklab(rgb);

	float L = lab.L;
	float eps = 0.00001f;
	float C = fmax(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
	float a_ = lab.a / C;
	float b_ = lab.b / C;

	float Ld = L - 0.5f;
	float e1 = 0.5f + fabs(Ld) + alpha * C;
	float L0 = 0.5f * (1.f + sgn(Ld) * (e1 - sqrtf(e1 * e1 - 2.f * fabs(Ld))));

	float t = find_gamut_intersection(a_, b_, L, C, L0);
	float L_clipped = L0 * (1.f - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb({ L_clipped, C_clipped * a_, C_clipped * b_ });
}

static RGB gamut_clip_adaptive_L0_L_cusp(RGB rgb, float alpha = 0.05f)
{
	if (rgb.r < 1 && rgb.g < 1 && rgb.b < 1 && rgb.r > 0 && rgb.g > 0 && rgb.b > 0)
		return rgb;

	Lab lab = linear_srgb_to_oklab(rgb);

	float L = lab.L;
	float eps = 0.00001f;
	float C = fmax(eps, sqrtf(lab.a * lab.a + lab.b * lab.b));
	float a_ = lab.a / C;
	float b_ = lab.b / C;

	// The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
	LC cusp = find_cusp(a_, b_);

	float Ld = L - cusp.L;
	float k = 2.f * (Ld > 0 ? 1.f - cusp.L : cusp.L);

	float e1 = 0.5f * k + fabs(Ld) + alpha * C / k;
	float L0 = cusp.L + 0.5f * (sgn(Ld) * (e1 - sqrtf(e1 * e1 - 2.f * k * fabs(Ld))));

	float t = find_gamut_intersection(a_, b_, L, C, L0);
	float L_clipped = L0 * (1.f - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb({ L_clipped, C_clipped * a_, C_clipped * b_ });
}

static float toe(float x)
{
	constexpr float k_1 = 0.206f;
	constexpr float k_2 = 0.03f;
	constexpr float k_3 = (1.f + k_1) / (1.f + k_2);
	return 0.5f * (k_3 * x - k_1 + sqrtf((k_3 * x - k_1) * (k_3 * x - k_1) + 4 * k_2 * k_3 * x));
}

static float toe_inv(float x)
{
	constexpr float k_1 = 0.206f;
	constexpr float k_2 = 0.03f;
	constexpr float k_3 = (1.f + k_1) / (1.f + k_2);
	return (x * x + k_1 * x) / (k_3 * (x + k_2));
}

static ST to_ST(LC cusp)
{
	float L = cusp.L;
	float C = cusp.C;
	return { C / L, C / (1 - L) };
}

// Returns a smooth approximation of the location of the cusp
// This polynomial was created by an optimization process
// It has been designed so that S_mid < S_max and T_mid < T_max
static ST get_ST_mid(float a_, float b_)
{
	float S = 0.11516993f + 1.f / (
		+7.44778970f + 4.15901240f * b_
		+ a_ * (-2.19557347f + 1.75198401f * b_
			+ a_ * (-2.13704948f - 10.02301043f * b_
				+ a_ * (-4.24894561f + 5.38770819f * b_ + 4.69891013f * a_
					)))
		);

	float T = 0.11239642f + 1.f / (
		+1.61320320f - 0.68124379f * b_
		+ a_ * (+0.40370612f + 0.90148123f * b_
			+ a_ * (-0.27087943f + 0.61223990f * b_
				+ a_ * (+0.00299215f - 0.45399568f * b_ - 0.14661872f * a_
					)))
		);

	return { S, T };
}

struct Cs { float C_0; float C_mid; float C_max; };
static Cs get_Cs(float L, float a_, float b_)
{
	LC cusp = find_cusp(a_, b_);

	float C_max = find_gamut_intersection(a_, b_, L, 1, L, cusp);
	ST ST_max = to_ST(cusp);

	// Scale factor to compensate for the curved part of gamut shape:
	float k = C_max / fmin((L * ST_max.S), (1 - L) * ST_max.T);

	float C_mid;
	{
		ST ST_mid = get_ST_mid(a_, b_);

		// Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
		float C_a = L * ST_mid.S;
		float C_b = (1.f - L) * ST_mid.T;
		C_mid = 0.9f * k * sqrtf(sqrtf(1.f / (1.f / (C_a * C_a * C_a * C_a) + 1.f / (C_b * C_b * C_b * C_b))));
	}

	float C_0;
	{
		// for C_0, the shape is independent of hue, so ST are constant. Values picked to roughly be the average values of ST.
		float C_a = L * 0.4f;
		float C_b = (1.f - L) * 0.8f;

		// Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
		C_0 = sqrtf(1.f / (1.f / (C_a * C_a) + 1.f / (C_b * C_b)));
	}

	return { C_0, C_mid, C_max };
}

static RGB okhsl_to_srgb(HSL hsl)
{
	float h = hsl.h;
	float s = hsl.s;
	float l = hsl.l;

	if (l == 1.0f)
	{
		return { 1.f, 1.f, 1.f };
	}

	else if (l == 0.f)
	{
		return { 0.f, 0.f, 0.f };
	}

	float a_ = cosf(2.f * pi * h);
	float b_ = sinf(2.f * pi * h);
	float L = toe_inv(l);

	Cs cs = get_Cs(L, a_, b_);
	float C_0 = cs.C_0;
	float C_mid = cs.C_mid;
	float C_max = cs.C_max;

	float mid = 0.8f;
	float mid_inv = 1.25f;

	float C, t, k_0, k_1, k_2;

	if (s < mid)
	{
		t = mid_inv * s;

		k_1 = mid * C_0;
		k_2 = (1.f - k_1 / C_mid);

		C = t * k_1 / (1.f - k_2 * t);
	}
	else
	{
		t = (s - mid)/ (1 - mid);

		k_0 = C_mid;
		k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		k_2 = (1.f - (k_1) / (C_max - C_mid));

		C = k_0 + t * k_1 / (1.f - k_2 * t);
	}

	RGB rgb = oklab_to_linear_srgb({ L, C * a_, C * b_ });
	return {
		srgb_transfer_function(rgb.r),
		srgb_transfer_function(rgb.g),
		srgb_transfer_function(rgb.b),
	};
}

static HSL srgb_to_okhsl(RGB rgb)
{
	Lab lab = linear_srgb_to_oklab({
		srgb_transfer_function_inv(rgb.r),
		srgb_transfer_function_inv(rgb.g),
		srgb_transfer_function_inv(rgb.b)
		});

	float C = sqrtf(lab.a * lab.a + lab.b * lab.b);
	float a_ = lab.a / C;
	float b_ = lab.b / C;

	float L = lab.L;
	float h = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) / pi;

	Cs cs = get_Cs(L, a_, b_);
	float C_0 = cs.C_0;
	float C_mid = cs.C_mid;
	float C_max = cs.C_max;

	// Inverse of the interpolation in okhsl_to_srgb:

	float mid = 0.8f;
	float mid_inv = 1.25f;

	float s;
	if (C < C_mid)
	{
		float k_1 = mid * C_0;
		float k_2 = (1.f - k_1 / C_mid);

		float t = C / (k_1 + k_2 * C);
		s = t * mid;
	}
	else
	{
		float k_0 = C_mid;
		float k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		float k_2 = (1.f - (k_1) / (C_max - C_mid));

		float t = (C - k_0) / (k_1 + k_2 * (C - k_0));
		s = mid + (1.f - mid) * t;
	}

	float l = toe(L);
	return { h, s, l };
}


static RGB okhsv_to_srgb(HSV hsv)
{
	float h = hsv.h;
	float s = hsv.s;
	float v = hsv.v;

	float a_ = cosf(2.f * pi * h);
	float b_ = sinf(2.f * pi * h);

	LC cusp = find_cusp(a_, b_);
	ST ST_max = to_ST(cusp);
	float S_max = ST_max.S;
	float T_max = ST_max.T;
	float S_0 = 0.5f;
	float k = 1 - S_0 / S_max;

	// first we compute L and V as if the gamut is a perfect triangle:

	// L, C when v==1:
	float L_v = 1     - s * S_0 / (S_0 + T_max - T_max * k * s);
	float C_v = s * T_max * S_0 / (S_0 + T_max - T_max * k * s);

	float L = v * L_v;
	float C = v * C_v;

	// then we compensate for both toe and the curved top part of the triangle:
	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt / L_v;

	float L_new = toe_inv(L);
	C = C * L_new / L;
	L = L_new;

	RGB rgb_scale = oklab_to_linear_srgb({ L_vt, a_ * C_vt, b_ * C_vt });
	float scale_L = cbrtf(1.f / fmax(fmax(rgb_scale.r, rgb_scale.g), fmax(rgb_scale.b, 0.f)));

	L = L * scale_L;
	C = C * scale_L;

	RGB rgb = oklab_to_linear_srgb({ L, C * a_, C * b_ });
	return {
		srgb_transfer_function(rgb.r),
		srgb_transfer_function(rgb.g),
		srgb_transfer_function(rgb.b),
	};
}

static HSV srgb_to_okhsv(RGB rgb)
{
	Lab lab = linear_srgb_to_oklab({
		srgb_transfer_function_inv(rgb.r),
		srgb_transfer_function_inv(rgb.g),
		srgb_transfer_function_inv(rgb.b)
		});

	float C = sqrtf(lab.a * lab.a + lab.b * lab.b);
	float a_ = lab.a / C;
	float b_ = lab.b / C;

	float L = lab.L;
	float h = 0.5f + 0.5f * atan2f(-lab.b, -lab.a) / pi;

	LC cusp = find_cusp(a_, b_);
	ST ST_max = to_ST(cusp);
	float S_max = ST_max.S;
	float T_max = ST_max.T;
	float S_0 = 0.5f;
	float k = 1 - S_0 / S_max;

	// first we find L_v, C_v, L_vt and C_vt

	float t = T_max / (C + L * T_max);
	float L_v = t * L;
	float C_v = t * C;

	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt / L_v;

	// we can then use these to invert the step that compensates for the toe and the curved top part of the triangle:
	RGB rgb_scale = oklab_to_linear_srgb({ L_vt, a_ * C_vt, b_ * C_vt });
	float scale_L = cbrtf(1.f / fmax(fmax(rgb_scale.r, rgb_scale.g), fmax(rgb_scale.b, 0.f)));

	L = L / scale_L;
	C = C / scale_L;

	C = C * toe(L) / L;
	L = toe(L);

	// we can now compute v and s:

	float v = L / L_v;
	float s = (S_0 + T_max) * C_v / ((T_max * S_0) + T_max * k * C_v);

	return { h, s, v };
}

};
#endif // OK_COLOR_H
