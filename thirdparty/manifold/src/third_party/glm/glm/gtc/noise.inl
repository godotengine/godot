/// @ref gtc_noise
///
// Based on the work of Stefan Gustavson and Ashima Arts on "webgl-noise":
// https://github.com/ashima/webgl-noise
// Following Stefan Gustavson's paper "Simplex noise demystified":
// http://www.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf

namespace glm{
namespace gtc
{
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER vec<4, T, Q> grad4(T const& j, vec<4, T, Q> const& ip)
	{
		vec<3, T, Q> pXYZ = floor(fract(vec<3, T, Q>(j) * vec<3, T, Q>(ip)) * T(7)) * ip[2] - T(1);
		T pW = static_cast<T>(1.5) - dot(abs(pXYZ), vec<3, T, Q>(1));
		vec<4, T, Q> s = vec<4, T, Q>(lessThan(vec<4, T, Q>(pXYZ, pW), vec<4, T, Q>(0.0)));
		pXYZ = pXYZ + (vec<3, T, Q>(s) * T(2) - T(1)) * s.w;
		return vec<4, T, Q>(pXYZ, pW);
	}
}//namespace gtc

	// Classic Perlin noise
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T perlin(vec<2, T, Q> const& Position)
	{
		vec<4, T, Q> Pi = glm::floor(vec<4, T, Q>(Position.x, Position.y, Position.x, Position.y)) + vec<4, T, Q>(0.0, 0.0, 1.0, 1.0);
		vec<4, T, Q> Pf = glm::fract(vec<4, T, Q>(Position.x, Position.y, Position.x, Position.y)) - vec<4, T, Q>(0.0, 0.0, 1.0, 1.0);
		Pi = mod(Pi, vec<4, T, Q>(289)); // To avoid truncation effects in permutation
		vec<4, T, Q> ix(Pi.x, Pi.z, Pi.x, Pi.z);
		vec<4, T, Q> iy(Pi.y, Pi.y, Pi.w, Pi.w);
		vec<4, T, Q> fx(Pf.x, Pf.z, Pf.x, Pf.z);
		vec<4, T, Q> fy(Pf.y, Pf.y, Pf.w, Pf.w);

		vec<4, T, Q> i = detail::permute(detail::permute(ix) + iy);

		vec<4, T, Q> gx = static_cast<T>(2) * glm::fract(i / T(41)) - T(1);
		vec<4, T, Q> gy = glm::abs(gx) - T(0.5);
		vec<4, T, Q> tx = glm::floor(gx + T(0.5));
		gx = gx - tx;

		vec<2, T, Q> g00(gx.x, gy.x);
		vec<2, T, Q> g10(gx.y, gy.y);
		vec<2, T, Q> g01(gx.z, gy.z);
		vec<2, T, Q> g11(gx.w, gy.w);

		vec<4, T, Q> norm = detail::taylorInvSqrt(vec<4, T, Q>(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
		g00 *= norm.x;
		g01 *= norm.y;
		g10 *= norm.z;
		g11 *= norm.w;

		T n00 = dot(g00, vec<2, T, Q>(fx.x, fy.x));
		T n10 = dot(g10, vec<2, T, Q>(fx.y, fy.y));
		T n01 = dot(g01, vec<2, T, Q>(fx.z, fy.z));
		T n11 = dot(g11, vec<2, T, Q>(fx.w, fy.w));

		vec<2, T, Q> fade_xy = detail::fade(vec<2, T, Q>(Pf.x, Pf.y));
		vec<2, T, Q> n_x = mix(vec<2, T, Q>(n00, n01), vec<2, T, Q>(n10, n11), fade_xy.x);
		T n_xy = mix(n_x.x, n_x.y, fade_xy.y);
		return T(2.3) * n_xy;
	}

	// Classic Perlin noise
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T perlin(vec<3, T, Q> const& Position)
	{
		vec<3, T, Q> Pi0 = floor(Position); // Integer part for indexing
		vec<3, T, Q> Pi1 = Pi0 + T(1); // Integer part + 1
		Pi0 = detail::mod289(Pi0);
		Pi1 = detail::mod289(Pi1);
		vec<3, T, Q> Pf0 = fract(Position); // Fractional part for interpolation
		vec<3, T, Q> Pf1 = Pf0 - T(1); // Fractional part - 1.0
		vec<4, T, Q> ix(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
		vec<4, T, Q> iy = vec<4, T, Q>(vec<2, T, Q>(Pi0.y), vec<2, T, Q>(Pi1.y));
		vec<4, T, Q> iz0(Pi0.z);
		vec<4, T, Q> iz1(Pi1.z);

		vec<4, T, Q> ixy = detail::permute(detail::permute(ix) + iy);
		vec<4, T, Q> ixy0 = detail::permute(ixy + iz0);
		vec<4, T, Q> ixy1 = detail::permute(ixy + iz1);

		vec<4, T, Q> gx0 = ixy0 * T(1.0 / 7.0);
		vec<4, T, Q> gy0 = fract(floor(gx0) * T(1.0 / 7.0)) - T(0.5);
		gx0 = fract(gx0);
		vec<4, T, Q> gz0 = vec<4, T, Q>(0.5) - abs(gx0) - abs(gy0);
		vec<4, T, Q> sz0 = step(gz0, vec<4, T, Q>(0.0));
		gx0 -= sz0 * (step(T(0), gx0) - T(0.5));
		gy0 -= sz0 * (step(T(0), gy0) - T(0.5));

		vec<4, T, Q> gx1 = ixy1 * T(1.0 / 7.0);
		vec<4, T, Q> gy1 = fract(floor(gx1) * T(1.0 / 7.0)) - T(0.5);
		gx1 = fract(gx1);
		vec<4, T, Q> gz1 = vec<4, T, Q>(0.5) - abs(gx1) - abs(gy1);
		vec<4, T, Q> sz1 = step(gz1, vec<4, T, Q>(0.0));
		gx1 -= sz1 * (step(T(0), gx1) - T(0.5));
		gy1 -= sz1 * (step(T(0), gy1) - T(0.5));

		vec<3, T, Q> g000(gx0.x, gy0.x, gz0.x);
		vec<3, T, Q> g100(gx0.y, gy0.y, gz0.y);
		vec<3, T, Q> g010(gx0.z, gy0.z, gz0.z);
		vec<3, T, Q> g110(gx0.w, gy0.w, gz0.w);
		vec<3, T, Q> g001(gx1.x, gy1.x, gz1.x);
		vec<3, T, Q> g101(gx1.y, gy1.y, gz1.y);
		vec<3, T, Q> g011(gx1.z, gy1.z, gz1.z);
		vec<3, T, Q> g111(gx1.w, gy1.w, gz1.w);

		vec<4, T, Q> norm0 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
		g000 *= norm0.x;
		g010 *= norm0.y;
		g100 *= norm0.z;
		g110 *= norm0.w;
		vec<4, T, Q> norm1 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
		g001 *= norm1.x;
		g011 *= norm1.y;
		g101 *= norm1.z;
		g111 *= norm1.w;

		T n000 = dot(g000, Pf0);
		T n100 = dot(g100, vec<3, T, Q>(Pf1.x, Pf0.y, Pf0.z));
		T n010 = dot(g010, vec<3, T, Q>(Pf0.x, Pf1.y, Pf0.z));
		T n110 = dot(g110, vec<3, T, Q>(Pf1.x, Pf1.y, Pf0.z));
		T n001 = dot(g001, vec<3, T, Q>(Pf0.x, Pf0.y, Pf1.z));
		T n101 = dot(g101, vec<3, T, Q>(Pf1.x, Pf0.y, Pf1.z));
		T n011 = dot(g011, vec<3, T, Q>(Pf0.x, Pf1.y, Pf1.z));
		T n111 = dot(g111, Pf1);

		vec<3, T, Q> fade_xyz = detail::fade(Pf0);
		vec<4, T, Q> n_z = mix(vec<4, T, Q>(n000, n100, n010, n110), vec<4, T, Q>(n001, n101, n011, n111), fade_xyz.z);
		vec<2, T, Q> n_yz = mix(vec<2, T, Q>(n_z.x, n_z.y), vec<2, T, Q>(n_z.z, n_z.w), fade_xyz.y);
		T n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
		return T(2.2) * n_xyz;
	}
	/*
	// Classic Perlin noise
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T perlin(vec<3, T, Q> const& P)
	{
		vec<3, T, Q> Pi0 = floor(P); // Integer part for indexing
		vec<3, T, Q> Pi1 = Pi0 + T(1); // Integer part + 1
		Pi0 = mod(Pi0, T(289));
		Pi1 = mod(Pi1, T(289));
		vec<3, T, Q> Pf0 = fract(P); // Fractional part for interpolation
		vec<3, T, Q> Pf1 = Pf0 - T(1); // Fractional part - 1.0
		vec<4, T, Q> ix(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
		vec<4, T, Q> iy(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
		vec<4, T, Q> iz0(Pi0.z);
		vec<4, T, Q> iz1(Pi1.z);

		vec<4, T, Q> ixy = permute(permute(ix) + iy);
		vec<4, T, Q> ixy0 = permute(ixy + iz0);
		vec<4, T, Q> ixy1 = permute(ixy + iz1);

		vec<4, T, Q> gx0 = ixy0 / T(7);
		vec<4, T, Q> gy0 = fract(floor(gx0) / T(7)) - T(0.5);
		gx0 = fract(gx0);
		vec<4, T, Q> gz0 = vec<4, T, Q>(0.5) - abs(gx0) - abs(gy0);
		vec<4, T, Q> sz0 = step(gz0, vec<4, T, Q>(0.0));
		gx0 -= sz0 * (step(0.0, gx0) - T(0.5));
		gy0 -= sz0 * (step(0.0, gy0) - T(0.5));

		vec<4, T, Q> gx1 = ixy1 / T(7);
		vec<4, T, Q> gy1 = fract(floor(gx1) / T(7)) - T(0.5);
		gx1 = fract(gx1);
		vec<4, T, Q> gz1 = vec<4, T, Q>(0.5) - abs(gx1) - abs(gy1);
		vec<4, T, Q> sz1 = step(gz1, vec<4, T, Q>(0.0));
		gx1 -= sz1 * (step(T(0), gx1) - T(0.5));
		gy1 -= sz1 * (step(T(0), gy1) - T(0.5));

		vec<3, T, Q> g000(gx0.x, gy0.x, gz0.x);
		vec<3, T, Q> g100(gx0.y, gy0.y, gz0.y);
		vec<3, T, Q> g010(gx0.z, gy0.z, gz0.z);
		vec<3, T, Q> g110(gx0.w, gy0.w, gz0.w);
		vec<3, T, Q> g001(gx1.x, gy1.x, gz1.x);
		vec<3, T, Q> g101(gx1.y, gy1.y, gz1.y);
		vec<3, T, Q> g011(gx1.z, gy1.z, gz1.z);
		vec<3, T, Q> g111(gx1.w, gy1.w, gz1.w);

		vec<4, T, Q> norm0 = taylorInvSqrt(vec<4, T, Q>(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
		g000 *= norm0.x;
		g010 *= norm0.y;
		g100 *= norm0.z;
		g110 *= norm0.w;
		vec<4, T, Q> norm1 = taylorInvSqrt(vec<4, T, Q>(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
		g001 *= norm1.x;
		g011 *= norm1.y;
		g101 *= norm1.z;
		g111 *= norm1.w;

		T n000 = dot(g000, Pf0);
		T n100 = dot(g100, vec<3, T, Q>(Pf1.x, Pf0.y, Pf0.z));
		T n010 = dot(g010, vec<3, T, Q>(Pf0.x, Pf1.y, Pf0.z));
		T n110 = dot(g110, vec<3, T, Q>(Pf1.x, Pf1.y, Pf0.z));
		T n001 = dot(g001, vec<3, T, Q>(Pf0.x, Pf0.y, Pf1.z));
		T n101 = dot(g101, vec<3, T, Q>(Pf1.x, Pf0.y, Pf1.z));
		T n011 = dot(g011, vec<3, T, Q>(Pf0.x, Pf1.y, Pf1.z));
		T n111 = dot(g111, Pf1);

		vec<3, T, Q> fade_xyz = fade(Pf0);
		vec<4, T, Q> n_z = mix(vec<4, T, Q>(n000, n100, n010, n110), vec<4, T, Q>(n001, n101, n011, n111), fade_xyz.z);
		vec<2, T, Q> n_yz = mix(
			vec<2, T, Q>(n_z.x, n_z.y),
			vec<2, T, Q>(n_z.z, n_z.w), fade_xyz.y);
		T n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
		return T(2.2) * n_xyz;
	}
	*/
	// Classic Perlin noise
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T perlin(vec<4, T, Q> const& Position)
	{
		vec<4, T, Q> Pi0 = floor(Position);	// Integer part for indexing
		vec<4, T, Q> Pi1 = Pi0 + T(1);		// Integer part + 1
		Pi0 = mod(Pi0, vec<4, T, Q>(289));
		Pi1 = mod(Pi1, vec<4, T, Q>(289));
		vec<4, T, Q> Pf0 = fract(Position);	// Fractional part for interpolation
		vec<4, T, Q> Pf1 = Pf0 - T(1);		// Fractional part - 1.0
		vec<4, T, Q> ix(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
		vec<4, T, Q> iy(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
		vec<4, T, Q> iz0(Pi0.z);
		vec<4, T, Q> iz1(Pi1.z);
		vec<4, T, Q> iw0(Pi0.w);
		vec<4, T, Q> iw1(Pi1.w);

		vec<4, T, Q> ixy = detail::permute(detail::permute(ix) + iy);
		vec<4, T, Q> ixy0 = detail::permute(ixy + iz0);
		vec<4, T, Q> ixy1 = detail::permute(ixy + iz1);
		vec<4, T, Q> ixy00 = detail::permute(ixy0 + iw0);
		vec<4, T, Q> ixy01 = detail::permute(ixy0 + iw1);
		vec<4, T, Q> ixy10 = detail::permute(ixy1 + iw0);
		vec<4, T, Q> ixy11 = detail::permute(ixy1 + iw1);

		vec<4, T, Q> gx00 = ixy00 / T(7);
		vec<4, T, Q> gy00 = floor(gx00) / T(7);
		vec<4, T, Q> gz00 = floor(gy00) / T(6);
		gx00 = fract(gx00) - T(0.5);
		gy00 = fract(gy00) - T(0.5);
		gz00 = fract(gz00) - T(0.5);
		vec<4, T, Q> gw00 = vec<4, T, Q>(0.75) - abs(gx00) - abs(gy00) - abs(gz00);
		vec<4, T, Q> sw00 = step(gw00, vec<4, T, Q>(0.0));
		gx00 -= sw00 * (step(T(0), gx00) - T(0.5));
		gy00 -= sw00 * (step(T(0), gy00) - T(0.5));

		vec<4, T, Q> gx01 = ixy01 / T(7);
		vec<4, T, Q> gy01 = floor(gx01) / T(7);
		vec<4, T, Q> gz01 = floor(gy01) / T(6);
		gx01 = fract(gx01) - T(0.5);
		gy01 = fract(gy01) - T(0.5);
		gz01 = fract(gz01) - T(0.5);
		vec<4, T, Q> gw01 = vec<4, T, Q>(0.75) - abs(gx01) - abs(gy01) - abs(gz01);
		vec<4, T, Q> sw01 = step(gw01, vec<4, T, Q>(0.0));
		gx01 -= sw01 * (step(T(0), gx01) - T(0.5));
		gy01 -= sw01 * (step(T(0), gy01) - T(0.5));

		vec<4, T, Q> gx10 = ixy10 / T(7);
		vec<4, T, Q> gy10 = floor(gx10) / T(7);
		vec<4, T, Q> gz10 = floor(gy10) / T(6);
		gx10 = fract(gx10) - T(0.5);
		gy10 = fract(gy10) - T(0.5);
		gz10 = fract(gz10) - T(0.5);
		vec<4, T, Q> gw10 = vec<4, T, Q>(0.75) - abs(gx10) - abs(gy10) - abs(gz10);
		vec<4, T, Q> sw10 = step(gw10, vec<4, T, Q>(0));
		gx10 -= sw10 * (step(T(0), gx10) - T(0.5));
		gy10 -= sw10 * (step(T(0), gy10) - T(0.5));

		vec<4, T, Q> gx11 = ixy11 / T(7);
		vec<4, T, Q> gy11 = floor(gx11) / T(7);
		vec<4, T, Q> gz11 = floor(gy11) / T(6);
		gx11 = fract(gx11) - T(0.5);
		gy11 = fract(gy11) - T(0.5);
		gz11 = fract(gz11) - T(0.5);
		vec<4, T, Q> gw11 = vec<4, T, Q>(0.75) - abs(gx11) - abs(gy11) - abs(gz11);
		vec<4, T, Q> sw11 = step(gw11, vec<4, T, Q>(0.0));
		gx11 -= sw11 * (step(T(0), gx11) - T(0.5));
		gy11 -= sw11 * (step(T(0), gy11) - T(0.5));

		vec<4, T, Q> g0000(gx00.x, gy00.x, gz00.x, gw00.x);
		vec<4, T, Q> g1000(gx00.y, gy00.y, gz00.y, gw00.y);
		vec<4, T, Q> g0100(gx00.z, gy00.z, gz00.z, gw00.z);
		vec<4, T, Q> g1100(gx00.w, gy00.w, gz00.w, gw00.w);
		vec<4, T, Q> g0010(gx10.x, gy10.x, gz10.x, gw10.x);
		vec<4, T, Q> g1010(gx10.y, gy10.y, gz10.y, gw10.y);
		vec<4, T, Q> g0110(gx10.z, gy10.z, gz10.z, gw10.z);
		vec<4, T, Q> g1110(gx10.w, gy10.w, gz10.w, gw10.w);
		vec<4, T, Q> g0001(gx01.x, gy01.x, gz01.x, gw01.x);
		vec<4, T, Q> g1001(gx01.y, gy01.y, gz01.y, gw01.y);
		vec<4, T, Q> g0101(gx01.z, gy01.z, gz01.z, gw01.z);
		vec<4, T, Q> g1101(gx01.w, gy01.w, gz01.w, gw01.w);
		vec<4, T, Q> g0011(gx11.x, gy11.x, gz11.x, gw11.x);
		vec<4, T, Q> g1011(gx11.y, gy11.y, gz11.y, gw11.y);
		vec<4, T, Q> g0111(gx11.z, gy11.z, gz11.z, gw11.z);
		vec<4, T, Q> g1111(gx11.w, gy11.w, gz11.w, gw11.w);

		vec<4, T, Q> norm00 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
		g0000 *= norm00.x;
		g0100 *= norm00.y;
		g1000 *= norm00.z;
		g1100 *= norm00.w;

		vec<4, T, Q> norm01 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
		g0001 *= norm01.x;
		g0101 *= norm01.y;
		g1001 *= norm01.z;
		g1101 *= norm01.w;

		vec<4, T, Q> norm10 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
		g0010 *= norm10.x;
		g0110 *= norm10.y;
		g1010 *= norm10.z;
		g1110 *= norm10.w;

		vec<4, T, Q> norm11 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
		g0011 *= norm11.x;
		g0111 *= norm11.y;
		g1011 *= norm11.z;
		g1111 *= norm11.w;

		T n0000 = dot(g0000, Pf0);
		T n1000 = dot(g1000, vec<4, T, Q>(Pf1.x, Pf0.y, Pf0.z, Pf0.w));
		T n0100 = dot(g0100, vec<4, T, Q>(Pf0.x, Pf1.y, Pf0.z, Pf0.w));
		T n1100 = dot(g1100, vec<4, T, Q>(Pf1.x, Pf1.y, Pf0.z, Pf0.w));
		T n0010 = dot(g0010, vec<4, T, Q>(Pf0.x, Pf0.y, Pf1.z, Pf0.w));
		T n1010 = dot(g1010, vec<4, T, Q>(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
		T n0110 = dot(g0110, vec<4, T, Q>(Pf0.x, Pf1.y, Pf1.z, Pf0.w));
		T n1110 = dot(g1110, vec<4, T, Q>(Pf1.x, Pf1.y, Pf1.z, Pf0.w));
		T n0001 = dot(g0001, vec<4, T, Q>(Pf0.x, Pf0.y, Pf0.z, Pf1.w));
		T n1001 = dot(g1001, vec<4, T, Q>(Pf1.x, Pf0.y, Pf0.z, Pf1.w));
		T n0101 = dot(g0101, vec<4, T, Q>(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
		T n1101 = dot(g1101, vec<4, T, Q>(Pf1.x, Pf1.y, Pf0.z, Pf1.w));
		T n0011 = dot(g0011, vec<4, T, Q>(Pf0.x, Pf0.y, Pf1.z, Pf1.w));
		T n1011 = dot(g1011, vec<4, T, Q>(Pf1.x, Pf0.y, Pf1.z, Pf1.w));
		T n0111 = dot(g0111, vec<4, T, Q>(Pf0.x, Pf1.y, Pf1.z, Pf1.w));
		T n1111 = dot(g1111, Pf1);

		vec<4, T, Q> fade_xyzw = detail::fade(Pf0);
		vec<4, T, Q> n_0w = mix(vec<4, T, Q>(n0000, n1000, n0100, n1100), vec<4, T, Q>(n0001, n1001, n0101, n1101), fade_xyzw.w);
		vec<4, T, Q> n_1w = mix(vec<4, T, Q>(n0010, n1010, n0110, n1110), vec<4, T, Q>(n0011, n1011, n0111, n1111), fade_xyzw.w);
		vec<4, T, Q> n_zw = mix(n_0w, n_1w, fade_xyzw.z);
		vec<2, T, Q> n_yzw = mix(vec<2, T, Q>(n_zw.x, n_zw.y), vec<2, T, Q>(n_zw.z, n_zw.w), fade_xyzw.y);
		T n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
		return T(2.2) * n_xyzw;
	}

	// Classic Perlin noise, periodic variant
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T perlin(vec<2, T, Q> const& Position, vec<2, T, Q> const& rep)
	{
		vec<4, T, Q> Pi = floor(vec<4, T, Q>(Position.x, Position.y, Position.x, Position.y)) + vec<4, T, Q>(0.0, 0.0, 1.0, 1.0);
		vec<4, T, Q> Pf = fract(vec<4, T, Q>(Position.x, Position.y, Position.x, Position.y)) - vec<4, T, Q>(0.0, 0.0, 1.0, 1.0);
		Pi = mod(Pi, vec<4, T, Q>(rep.x, rep.y, rep.x, rep.y)); // To create noise with explicit period
		Pi = mod(Pi, vec<4, T, Q>(289)); // To avoid truncation effects in permutation
		vec<4, T, Q> ix(Pi.x, Pi.z, Pi.x, Pi.z);
		vec<4, T, Q> iy(Pi.y, Pi.y, Pi.w, Pi.w);
		vec<4, T, Q> fx(Pf.x, Pf.z, Pf.x, Pf.z);
		vec<4, T, Q> fy(Pf.y, Pf.y, Pf.w, Pf.w);

		vec<4, T, Q> i = detail::permute(detail::permute(ix) + iy);

		vec<4, T, Q> gx = static_cast<T>(2) * fract(i / T(41)) - T(1);
		vec<4, T, Q> gy = abs(gx) - T(0.5);
		vec<4, T, Q> tx = floor(gx + T(0.5));
		gx = gx - tx;

		vec<2, T, Q> g00(gx.x, gy.x);
		vec<2, T, Q> g10(gx.y, gy.y);
		vec<2, T, Q> g01(gx.z, gy.z);
		vec<2, T, Q> g11(gx.w, gy.w);

		vec<4, T, Q> norm = detail::taylorInvSqrt(vec<4, T, Q>(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
		g00 *= norm.x;
		g01 *= norm.y;
		g10 *= norm.z;
		g11 *= norm.w;

		T n00 = dot(g00, vec<2, T, Q>(fx.x, fy.x));
		T n10 = dot(g10, vec<2, T, Q>(fx.y, fy.y));
		T n01 = dot(g01, vec<2, T, Q>(fx.z, fy.z));
		T n11 = dot(g11, vec<2, T, Q>(fx.w, fy.w));

		vec<2, T, Q> fade_xy = detail::fade(vec<2, T, Q>(Pf.x, Pf.y));
		vec<2, T, Q> n_x = mix(vec<2, T, Q>(n00, n01), vec<2, T, Q>(n10, n11), fade_xy.x);
		T n_xy = mix(n_x.x, n_x.y, fade_xy.y);
		return T(2.3) * n_xy;
	}

	// Classic Perlin noise, periodic variant
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T perlin(vec<3, T, Q> const& Position, vec<3, T, Q> const& rep)
	{
		vec<3, T, Q> Pi0 = mod(floor(Position), rep); // Integer part, modulo period
		vec<3, T, Q> Pi1 = mod(Pi0 + vec<3, T, Q>(T(1)), rep); // Integer part + 1, mod period
		Pi0 = mod(Pi0, vec<3, T, Q>(289));
		Pi1 = mod(Pi1, vec<3, T, Q>(289));
		vec<3, T, Q> Pf0 = fract(Position); // Fractional part for interpolation
		vec<3, T, Q> Pf1 = Pf0 - vec<3, T, Q>(T(1)); // Fractional part - 1.0
		vec<4, T, Q> ix = vec<4, T, Q>(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
		vec<4, T, Q> iy = vec<4, T, Q>(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
		vec<4, T, Q> iz0(Pi0.z);
		vec<4, T, Q> iz1(Pi1.z);

		vec<4, T, Q> ixy = detail::permute(detail::permute(ix) + iy);
		vec<4, T, Q> ixy0 = detail::permute(ixy + iz0);
		vec<4, T, Q> ixy1 = detail::permute(ixy + iz1);

		vec<4, T, Q> gx0 = ixy0 / T(7);
		vec<4, T, Q> gy0 = fract(floor(gx0) / T(7)) - T(0.5);
		gx0 = fract(gx0);
		vec<4, T, Q> gz0 = vec<4, T, Q>(0.5) - abs(gx0) - abs(gy0);
		vec<4, T, Q> sz0 = step(gz0, vec<4, T, Q>(0));
		gx0 -= sz0 * (step(T(0), gx0) - T(0.5));
		gy0 -= sz0 * (step(T(0), gy0) - T(0.5));

		vec<4, T, Q> gx1 = ixy1 / T(7);
		vec<4, T, Q> gy1 = fract(floor(gx1) / T(7)) - T(0.5);
		gx1 = fract(gx1);
		vec<4, T, Q> gz1 = vec<4, T, Q>(0.5) - abs(gx1) - abs(gy1);
		vec<4, T, Q> sz1 = step(gz1, vec<4, T, Q>(T(0)));
		gx1 -= sz1 * (step(T(0), gx1) - T(0.5));
		gy1 -= sz1 * (step(T(0), gy1) - T(0.5));

		vec<3, T, Q> g000 = vec<3, T, Q>(gx0.x, gy0.x, gz0.x);
		vec<3, T, Q> g100 = vec<3, T, Q>(gx0.y, gy0.y, gz0.y);
		vec<3, T, Q> g010 = vec<3, T, Q>(gx0.z, gy0.z, gz0.z);
		vec<3, T, Q> g110 = vec<3, T, Q>(gx0.w, gy0.w, gz0.w);
		vec<3, T, Q> g001 = vec<3, T, Q>(gx1.x, gy1.x, gz1.x);
		vec<3, T, Q> g101 = vec<3, T, Q>(gx1.y, gy1.y, gz1.y);
		vec<3, T, Q> g011 = vec<3, T, Q>(gx1.z, gy1.z, gz1.z);
		vec<3, T, Q> g111 = vec<3, T, Q>(gx1.w, gy1.w, gz1.w);

		vec<4, T, Q> norm0 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
		g000 *= norm0.x;
		g010 *= norm0.y;
		g100 *= norm0.z;
		g110 *= norm0.w;
		vec<4, T, Q> norm1 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
		g001 *= norm1.x;
		g011 *= norm1.y;
		g101 *= norm1.z;
		g111 *= norm1.w;

		T n000 = dot(g000, Pf0);
		T n100 = dot(g100, vec<3, T, Q>(Pf1.x, Pf0.y, Pf0.z));
		T n010 = dot(g010, vec<3, T, Q>(Pf0.x, Pf1.y, Pf0.z));
		T n110 = dot(g110, vec<3, T, Q>(Pf1.x, Pf1.y, Pf0.z));
		T n001 = dot(g001, vec<3, T, Q>(Pf0.x, Pf0.y, Pf1.z));
		T n101 = dot(g101, vec<3, T, Q>(Pf1.x, Pf0.y, Pf1.z));
		T n011 = dot(g011, vec<3, T, Q>(Pf0.x, Pf1.y, Pf1.z));
		T n111 = dot(g111, Pf1);

		vec<3, T, Q> fade_xyz = detail::fade(Pf0);
		vec<4, T, Q> n_z = mix(vec<4, T, Q>(n000, n100, n010, n110), vec<4, T, Q>(n001, n101, n011, n111), fade_xyz.z);
		vec<2, T, Q> n_yz = mix(vec<2, T, Q>(n_z.x, n_z.y), vec<2, T, Q>(n_z.z, n_z.w), fade_xyz.y);
		T n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
		return T(2.2) * n_xyz;
	}

	// Classic Perlin noise, periodic version
	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T perlin(vec<4, T, Q> const& Position, vec<4, T, Q> const& rep)
	{
		vec<4, T, Q> Pi0 = mod(floor(Position), rep); // Integer part modulo rep
		vec<4, T, Q> Pi1 = mod(Pi0 + T(1), rep); // Integer part + 1 mod rep
		vec<4, T, Q> Pf0 = fract(Position); // Fractional part for interpolation
		vec<4, T, Q> Pf1 = Pf0 - T(1); // Fractional part - 1.0
		vec<4, T, Q> ix = vec<4, T, Q>(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
		vec<4, T, Q> iy = vec<4, T, Q>(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
		vec<4, T, Q> iz0(Pi0.z);
		vec<4, T, Q> iz1(Pi1.z);
		vec<4, T, Q> iw0(Pi0.w);
		vec<4, T, Q> iw1(Pi1.w);

		vec<4, T, Q> ixy = detail::permute(detail::permute(ix) + iy);
		vec<4, T, Q> ixy0 = detail::permute(ixy + iz0);
		vec<4, T, Q> ixy1 = detail::permute(ixy + iz1);
		vec<4, T, Q> ixy00 = detail::permute(ixy0 + iw0);
		vec<4, T, Q> ixy01 = detail::permute(ixy0 + iw1);
		vec<4, T, Q> ixy10 = detail::permute(ixy1 + iw0);
		vec<4, T, Q> ixy11 = detail::permute(ixy1 + iw1);

		vec<4, T, Q> gx00 = ixy00 / T(7);
		vec<4, T, Q> gy00 = floor(gx00) / T(7);
		vec<4, T, Q> gz00 = floor(gy00) / T(6);
		gx00 = fract(gx00) - T(0.5);
		gy00 = fract(gy00) - T(0.5);
		gz00 = fract(gz00) - T(0.5);
		vec<4, T, Q> gw00 = vec<4, T, Q>(0.75) - abs(gx00) - abs(gy00) - abs(gz00);
		vec<4, T, Q> sw00 = step(gw00, vec<4, T, Q>(0));
		gx00 -= sw00 * (step(T(0), gx00) - T(0.5));
		gy00 -= sw00 * (step(T(0), gy00) - T(0.5));

		vec<4, T, Q> gx01 = ixy01 / T(7);
		vec<4, T, Q> gy01 = floor(gx01) / T(7);
		vec<4, T, Q> gz01 = floor(gy01) / T(6);
		gx01 = fract(gx01) - T(0.5);
		gy01 = fract(gy01) - T(0.5);
		gz01 = fract(gz01) - T(0.5);
		vec<4, T, Q> gw01 = vec<4, T, Q>(0.75) - abs(gx01) - abs(gy01) - abs(gz01);
		vec<4, T, Q> sw01 = step(gw01, vec<4, T, Q>(0.0));
		gx01 -= sw01 * (step(T(0), gx01) - T(0.5));
		gy01 -= sw01 * (step(T(0), gy01) - T(0.5));

		vec<4, T, Q> gx10 = ixy10 / T(7);
		vec<4, T, Q> gy10 = floor(gx10) / T(7);
		vec<4, T, Q> gz10 = floor(gy10) / T(6);
		gx10 = fract(gx10) - T(0.5);
		gy10 = fract(gy10) - T(0.5);
		gz10 = fract(gz10) - T(0.5);
		vec<4, T, Q> gw10 = vec<4, T, Q>(0.75) - abs(gx10) - abs(gy10) - abs(gz10);
		vec<4, T, Q> sw10 = step(gw10, vec<4, T, Q>(0.0));
		gx10 -= sw10 * (step(T(0), gx10) - T(0.5));
		gy10 -= sw10 * (step(T(0), gy10) - T(0.5));

		vec<4, T, Q> gx11 = ixy11 / T(7);
		vec<4, T, Q> gy11 = floor(gx11) / T(7);
		vec<4, T, Q> gz11 = floor(gy11) / T(6);
		gx11 = fract(gx11) - T(0.5);
		gy11 = fract(gy11) - T(0.5);
		gz11 = fract(gz11) - T(0.5);
		vec<4, T, Q> gw11 = vec<4, T, Q>(0.75) - abs(gx11) - abs(gy11) - abs(gz11);
		vec<4, T, Q> sw11 = step(gw11, vec<4, T, Q>(T(0)));
		gx11 -= sw11 * (step(T(0), gx11) - T(0.5));
		gy11 -= sw11 * (step(T(0), gy11) - T(0.5));

		vec<4, T, Q> g0000(gx00.x, gy00.x, gz00.x, gw00.x);
		vec<4, T, Q> g1000(gx00.y, gy00.y, gz00.y, gw00.y);
		vec<4, T, Q> g0100(gx00.z, gy00.z, gz00.z, gw00.z);
		vec<4, T, Q> g1100(gx00.w, gy00.w, gz00.w, gw00.w);
		vec<4, T, Q> g0010(gx10.x, gy10.x, gz10.x, gw10.x);
		vec<4, T, Q> g1010(gx10.y, gy10.y, gz10.y, gw10.y);
		vec<4, T, Q> g0110(gx10.z, gy10.z, gz10.z, gw10.z);
		vec<4, T, Q> g1110(gx10.w, gy10.w, gz10.w, gw10.w);
		vec<4, T, Q> g0001(gx01.x, gy01.x, gz01.x, gw01.x);
		vec<4, T, Q> g1001(gx01.y, gy01.y, gz01.y, gw01.y);
		vec<4, T, Q> g0101(gx01.z, gy01.z, gz01.z, gw01.z);
		vec<4, T, Q> g1101(gx01.w, gy01.w, gz01.w, gw01.w);
		vec<4, T, Q> g0011(gx11.x, gy11.x, gz11.x, gw11.x);
		vec<4, T, Q> g1011(gx11.y, gy11.y, gz11.y, gw11.y);
		vec<4, T, Q> g0111(gx11.z, gy11.z, gz11.z, gw11.z);
		vec<4, T, Q> g1111(gx11.w, gy11.w, gz11.w, gw11.w);

		vec<4, T, Q> norm00 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g0000, g0000), dot(g0100, g0100), dot(g1000, g1000), dot(g1100, g1100)));
		g0000 *= norm00.x;
		g0100 *= norm00.y;
		g1000 *= norm00.z;
		g1100 *= norm00.w;

		vec<4, T, Q> norm01 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g0001, g0001), dot(g0101, g0101), dot(g1001, g1001), dot(g1101, g1101)));
		g0001 *= norm01.x;
		g0101 *= norm01.y;
		g1001 *= norm01.z;
		g1101 *= norm01.w;

		vec<4, T, Q> norm10 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g0010, g0010), dot(g0110, g0110), dot(g1010, g1010), dot(g1110, g1110)));
		g0010 *= norm10.x;
		g0110 *= norm10.y;
		g1010 *= norm10.z;
		g1110 *= norm10.w;

		vec<4, T, Q> norm11 = detail::taylorInvSqrt(vec<4, T, Q>(dot(g0011, g0011), dot(g0111, g0111), dot(g1011, g1011), dot(g1111, g1111)));
		g0011 *= norm11.x;
		g0111 *= norm11.y;
		g1011 *= norm11.z;
		g1111 *= norm11.w;

		T n0000 = dot(g0000, Pf0);
		T n1000 = dot(g1000, vec<4, T, Q>(Pf1.x, Pf0.y, Pf0.z, Pf0.w));
		T n0100 = dot(g0100, vec<4, T, Q>(Pf0.x, Pf1.y, Pf0.z, Pf0.w));
		T n1100 = dot(g1100, vec<4, T, Q>(Pf1.x, Pf1.y, Pf0.z, Pf0.w));
		T n0010 = dot(g0010, vec<4, T, Q>(Pf0.x, Pf0.y, Pf1.z, Pf0.w));
		T n1010 = dot(g1010, vec<4, T, Q>(Pf1.x, Pf0.y, Pf1.z, Pf0.w));
		T n0110 = dot(g0110, vec<4, T, Q>(Pf0.x, Pf1.y, Pf1.z, Pf0.w));
		T n1110 = dot(g1110, vec<4, T, Q>(Pf1.x, Pf1.y, Pf1.z, Pf0.w));
		T n0001 = dot(g0001, vec<4, T, Q>(Pf0.x, Pf0.y, Pf0.z, Pf1.w));
		T n1001 = dot(g1001, vec<4, T, Q>(Pf1.x, Pf0.y, Pf0.z, Pf1.w));
		T n0101 = dot(g0101, vec<4, T, Q>(Pf0.x, Pf1.y, Pf0.z, Pf1.w));
		T n1101 = dot(g1101, vec<4, T, Q>(Pf1.x, Pf1.y, Pf0.z, Pf1.w));
		T n0011 = dot(g0011, vec<4, T, Q>(Pf0.x, Pf0.y, Pf1.z, Pf1.w));
		T n1011 = dot(g1011, vec<4, T, Q>(Pf1.x, Pf0.y, Pf1.z, Pf1.w));
		T n0111 = dot(g0111, vec<4, T, Q>(Pf0.x, Pf1.y, Pf1.z, Pf1.w));
		T n1111 = dot(g1111, Pf1);

		vec<4, T, Q> fade_xyzw = detail::fade(Pf0);
		vec<4, T, Q> n_0w = mix(vec<4, T, Q>(n0000, n1000, n0100, n1100), vec<4, T, Q>(n0001, n1001, n0101, n1101), fade_xyzw.w);
		vec<4, T, Q> n_1w = mix(vec<4, T, Q>(n0010, n1010, n0110, n1110), vec<4, T, Q>(n0011, n1011, n0111, n1111), fade_xyzw.w);
		vec<4, T, Q> n_zw = mix(n_0w, n_1w, fade_xyzw.z);
		vec<2, T, Q> n_yzw = mix(vec<2, T, Q>(n_zw.x, n_zw.y), vec<2, T, Q>(n_zw.z, n_zw.w), fade_xyzw.y);
		T n_xyzw = mix(n_yzw.x, n_yzw.y, fade_xyzw.x);
		return T(2.2) * n_xyzw;
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T simplex(glm::vec<2, T, Q> const& v)
	{
		vec<4, T, Q> const C = vec<4, T, Q>(
			T( 0.211324865405187),  // (3.0 -  sqrt(3.0)) / 6.0
			T( 0.366025403784439),  //  0.5 * (sqrt(3.0)  - 1.0)
			T(-0.577350269189626),	// -1.0 + 2.0 * C.x
			T( 0.024390243902439)); //  1.0 / 41.0

		// First corner
		vec<2, T, Q> i  = floor(v + dot(v, vec<2, T, Q>(C[1])));
		vec<2, T, Q> x0 = v -   i + dot(i, vec<2, T, Q>(C[0]));

		// Other corners
		//i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
		//i1.y = 1.0 - i1.x;
		vec<2, T, Q> i1 = (x0.x > x0.y) ? vec<2, T, Q>(1, 0) : vec<2, T, Q>(0, 1);
		// x0 = x0 - 0.0 + 0.0 * C.xx ;
		// x1 = x0 - i1 + 1.0 * C.xx ;
		// x2 = x0 - 1.0 + 2.0 * C.xx ;
		vec<4, T, Q> x12 = vec<4, T, Q>(x0.x, x0.y, x0.x, x0.y) + vec<4, T, Q>(C.x, C.x, C.z, C.z);
		x12 = vec<4, T, Q>(vec<2, T, Q>(x12) - i1, x12.z, x12.w);

		// Permutations
		i = mod(i, vec<2, T, Q>(289)); // Avoid truncation effects in permutation
		vec<3, T, Q> p = detail::permute(
			detail::permute(i.y + vec<3, T, Q>(T(0), i1.y, T(1)))
			+ i.x + vec<3, T, Q>(T(0), i1.x, T(1)));

		vec<3, T, Q> m = max(vec<3, T, Q>(0.5) - vec<3, T, Q>(
			dot(x0, x0),
			dot(vec<2, T, Q>(x12.x, x12.y), vec<2, T, Q>(x12.x, x12.y)),
			dot(vec<2, T, Q>(x12.z, x12.w), vec<2, T, Q>(x12.z, x12.w))), vec<3, T, Q>(0));
		m = m * m ;
		m = m * m ;

		// Gradients: 41 points uniformly over a line, mapped onto a diamond.
		// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

		vec<3, T, Q> x = static_cast<T>(2) * fract(p * C.w) - T(1);
		vec<3, T, Q> h = abs(x) - T(0.5);
		vec<3, T, Q> ox = floor(x + T(0.5));
		vec<3, T, Q> a0 = x - ox;

		// Normalise gradients implicitly by scaling m
		// Inlined for speed: m *= taylorInvSqrt( a0*a0 + h*h );
		m *= static_cast<T>(1.79284291400159) - T(0.85373472095314) * (a0 * a0 + h * h);

		// Compute final noise value at P
		vec<3, T, Q> g;
		g.x  = a0.x  * x0.x  + h.x  * x0.y;
		//g.yz = a0.yz * x12.xz + h.yz * x12.yw;
		g.y = a0.y * x12.x + h.y * x12.y;
		g.z = a0.z * x12.z + h.z * x12.w;
		return T(130) * dot(m, g);
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T simplex(vec<3, T, Q> const& v)
	{
		vec<2, T, Q> const C(1.0 / 6.0, 1.0 / 3.0);
		vec<4, T, Q> const D(0.0, 0.5, 1.0, 2.0);

		// First corner
		vec<3, T, Q> i(floor(v + dot(v, vec<3, T, Q>(C.y))));
		vec<3, T, Q> x0(v - i + dot(i, vec<3, T, Q>(C.x)));

		// Other corners
		vec<3, T, Q> g(step(vec<3, T, Q>(x0.y, x0.z, x0.x), x0));
		vec<3, T, Q> l(T(1) - g);
		vec<3, T, Q> i1(min(g, vec<3, T, Q>(l.z, l.x, l.y)));
		vec<3, T, Q> i2(max(g, vec<3, T, Q>(l.z, l.x, l.y)));

		//   x0 = x0 - 0.0 + 0.0 * C.xxx;
		//   x1 = x0 - i1  + 1.0 * C.xxx;
		//   x2 = x0 - i2  + 2.0 * C.xxx;
		//   x3 = x0 - 1.0 + 3.0 * C.xxx;
		vec<3, T, Q> x1(x0 - i1 + C.x);
		vec<3, T, Q> x2(x0 - i2 + C.y); // 2.0*C.x = 1/3 = C.y
		vec<3, T, Q> x3(x0 - D.y);      // -1.0+3.0*C.x = -0.5 = -D.y

		// Permutations
		i = detail::mod289(i);
		vec<4, T, Q> p(detail::permute(detail::permute(detail::permute(
			i.z + vec<4, T, Q>(T(0), i1.z, i2.z, T(1))) +
			i.y + vec<4, T, Q>(T(0), i1.y, i2.y, T(1))) +
			i.x + vec<4, T, Q>(T(0), i1.x, i2.x, T(1))));

		// Gradients: 7x7 points over a square, mapped onto an octahedron.
		// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
		T n_ = static_cast<T>(0.142857142857); // 1.0/7.0
		vec<3, T, Q> ns(n_ * vec<3, T, Q>(D.w, D.y, D.z) - vec<3, T, Q>(D.x, D.z, D.x));

		vec<4, T, Q> j(p - T(49) * floor(p * ns.z * ns.z));  //  mod(p,7*7)

		vec<4, T, Q> x_(floor(j * ns.z));
		vec<4, T, Q> y_(floor(j - T(7) * x_));    // mod(j,N)

		vec<4, T, Q> x(x_ * ns.x + ns.y);
		vec<4, T, Q> y(y_ * ns.x + ns.y);
		vec<4, T, Q> h(T(1) - abs(x) - abs(y));

		vec<4, T, Q> b0(x.x, x.y, y.x, y.y);
		vec<4, T, Q> b1(x.z, x.w, y.z, y.w);

		// vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
		// vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
		vec<4, T, Q> s0(floor(b0) * T(2) + T(1));
		vec<4, T, Q> s1(floor(b1) * T(2) + T(1));
		vec<4, T, Q> sh(-step(h, vec<4, T, Q>(0.0)));

		vec<4, T, Q> a0 = vec<4, T, Q>(b0.x, b0.z, b0.y, b0.w) + vec<4, T, Q>(s0.x, s0.z, s0.y, s0.w) * vec<4, T, Q>(sh.x, sh.x, sh.y, sh.y);
		vec<4, T, Q> a1 = vec<4, T, Q>(b1.x, b1.z, b1.y, b1.w) + vec<4, T, Q>(s1.x, s1.z, s1.y, s1.w) * vec<4, T, Q>(sh.z, sh.z, sh.w, sh.w);

		vec<3, T, Q> p0(a0.x, a0.y, h.x);
		vec<3, T, Q> p1(a0.z, a0.w, h.y);
		vec<3, T, Q> p2(a1.x, a1.y, h.z);
		vec<3, T, Q> p3(a1.z, a1.w, h.w);

		// Normalise gradients
		vec<4, T, Q> norm = detail::taylorInvSqrt(vec<4, T, Q>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
		p0 *= norm.x;
		p1 *= norm.y;
		p2 *= norm.z;
		p3 *= norm.w;

		// Mix final noise value
		vec<4, T, Q> m = max(T(0.6) - vec<4, T, Q>(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec<4, T, Q>(0));
		m = m * m;
		return T(42) * dot(m * m, vec<4, T, Q>(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
	}

	template<typename T, qualifier Q>
	GLM_FUNC_QUALIFIER T simplex(vec<4, T, Q> const& v)
	{
		vec<4, T, Q> const C(
			0.138196601125011,  // (5 - sqrt(5))/20  G4
			0.276393202250021,  // 2 * G4
			0.414589803375032,  // 3 * G4
			-0.447213595499958); // -1 + 4 * G4

		// (sqrt(5) - 1)/4 = F4, used once below
		T const F4 = static_cast<T>(0.309016994374947451);

		// First corner
		vec<4, T, Q> i  = floor(v + dot(v, vec<4, T, Q>(F4)));
		vec<4, T, Q> x0 = v -   i + dot(i, vec<4, T, Q>(C.x));

		// Other corners

		// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
		vec<4, T, Q> i0;
		vec<3, T, Q> isX = step(vec<3, T, Q>(x0.y, x0.z, x0.w), vec<3, T, Q>(x0.x));
		vec<3, T, Q> isYZ = step(vec<3, T, Q>(x0.z, x0.w, x0.w), vec<3, T, Q>(x0.y, x0.y, x0.z));
		//  i0.x = dot(isX, vec3(1.0));
		//i0.x = isX.x + isX.y + isX.z;
		//i0.yzw = static_cast<T>(1) - isX;
		i0 = vec<4, T, Q>(isX.x + isX.y + isX.z, T(1) - isX);
		//  i0.y += dot(isYZ.xy, vec2(1.0));
		i0.y += isYZ.x + isYZ.y;
		//i0.zw += 1.0 - vec<2, T, Q>(isYZ.x, isYZ.y);
		i0.z += static_cast<T>(1) - isYZ.x;
		i0.w += static_cast<T>(1) - isYZ.y;
		i0.z += isYZ.z;
		i0.w += static_cast<T>(1) - isYZ.z;

		// i0 now contains the unique values 0,1,2,3 in each channel
		vec<4, T, Q> i3 = clamp(i0, T(0), T(1));
		vec<4, T, Q> i2 = clamp(i0 - T(1), T(0), T(1));
		vec<4, T, Q> i1 = clamp(i0 - T(2), T(0), T(1));

		//  x0 = x0 - 0.0 + 0.0 * C.xxxx
		//  x1 = x0 - i1  + 0.0 * C.xxxx
		//  x2 = x0 - i2  + 0.0 * C.xxxx
		//  x3 = x0 - i3  + 0.0 * C.xxxx
		//  x4 = x0 - 1.0 + 4.0 * C.xxxx
		vec<4, T, Q> x1 = x0 - i1 + C.x;
		vec<4, T, Q> x2 = x0 - i2 + C.y;
		vec<4, T, Q> x3 = x0 - i3 + C.z;
		vec<4, T, Q> x4 = x0 + C.w;

		// Permutations
		i = mod(i, vec<4, T, Q>(289));
		T j0 = detail::permute(detail::permute(detail::permute(detail::permute(i.w) + i.z) + i.y) + i.x);
		vec<4, T, Q> j1 = detail::permute(detail::permute(detail::permute(detail::permute(
			i.w + vec<4, T, Q>(i1.w, i2.w, i3.w, T(1))) +
			i.z + vec<4, T, Q>(i1.z, i2.z, i3.z, T(1))) +
			i.y + vec<4, T, Q>(i1.y, i2.y, i3.y, T(1))) +
			i.x + vec<4, T, Q>(i1.x, i2.x, i3.x, T(1)));

		// Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
		// 7*7*6 = 294, which is close to the ring size 17*17 = 289.
		vec<4, T, Q> ip = vec<4, T, Q>(T(1) / T(294), T(1) / T(49), T(1) / T(7), T(0));

		vec<4, T, Q> p0 = gtc::grad4(j0,   ip);
		vec<4, T, Q> p1 = gtc::grad4(j1.x, ip);
		vec<4, T, Q> p2 = gtc::grad4(j1.y, ip);
		vec<4, T, Q> p3 = gtc::grad4(j1.z, ip);
		vec<4, T, Q> p4 = gtc::grad4(j1.w, ip);

		// Normalise gradients
		vec<4, T, Q> norm = detail::taylorInvSqrt(vec<4, T, Q>(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
		p0 *= norm.x;
		p1 *= norm.y;
		p2 *= norm.z;
		p3 *= norm.w;
		p4 *= detail::taylorInvSqrt(dot(p4, p4));

		// Mix contributions from the five corners
		vec<3, T, Q> m0 = max(T(0.6) - vec<3, T, Q>(dot(x0, x0), dot(x1, x1), dot(x2, x2)), vec<3, T, Q>(0));
		vec<2, T, Q> m1 = max(T(0.6) - vec<2, T, Q>(dot(x3, x3), dot(x4, x4)             ), vec<2, T, Q>(0));
		m0 = m0 * m0;
		m1 = m1 * m1;
		return T(49) *
			(dot(m0 * m0, vec<3, T, Q>(dot(p0, x0), dot(p1, x1), dot(p2, x2))) +
			dot(m1 * m1, vec<2, T, Q>(dot(p3, x3), dot(p4, x4))));
	}
}//namespace glm
