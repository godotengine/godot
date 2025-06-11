#ifdef REFERENCE_SH_IMPL
// This is standard SH evaluation at L1 order. Not used.
hvec3 evaluate_sh_l1(hvec3 dir, vec4 sh_coeffs[7]) {
	return hvec3(sh_coeffs[0][0], sh_coeffs[1][0], sh_coeffs[2][0]) +
			hvec3(sh_coeffs[0][1], sh_coeffs[1][1], sh_coeffs[2][1]) * dir.y +
			hvec3(sh_coeffs[0][2], sh_coeffs[1][2], sh_coeffs[2][2]) * dir.z +
			hvec3(sh_coeffs[0][3], sh_coeffs[1][3], sh_coeffs[2][3]) * dir.x;
}

// This is a modified SH evaluation at L1 order that fights the math to avoid "ringing" artifacts
// caused by evaluate_sh_l1() returning negative values for certain inputs while conserving energy
// i.e. physically correct. This function only handles one channel.
//
// 1. lenR1 can be precomputed. But it's not needed (see lenR1 / R0).
// 2. R1[i] = R1[i] / lenR1[i] can be precomputed into R1[i].
// 3. lenR1 / R0 can be precomputed.
// 4. p and a can be precomputed.
mediump half shEvaluateDiffuseL1Geomerics(hvec3 n, vec4 sh_coeffs[7], int channel) {
	// https://web.archive.org/web/20160313132301/http://www.geomerics.com/wp-content/uploads/2015/08/CEDEC_Geomerics_ReconstructingDiffuseLighting1.pdf
	// http://www.geomerics.com/wp-content/uploads/2015/08/CEDEC_Geomerics_ReconstructingDiffuseLighting1.pdf
	hvec4 sh = hvec4(sh_coeffs[channel][0], sh_coeffs[channel][1], sh_coeffs[channel][2], sh_coeffs[channel][3]);

	half R0 = sh[0];

	hvec3 R1 = half(0.5) * hvec3(sh[3], sh[1], sh[2]);
	half lenR1 = length(R1);
	half q = half(0.5) * (half(1.0) + dot(R1 / lenR1, n));
	half p = half(1.0) + 2.0f * lenR1 / R0;
	half a = (half(1.0) - lenR1 / R0) / (half(1.0) + lenR1 / R0);

	return R0 * (a + (half(1.0) - a) * (p + half(1.0)) * pow(q, p));
}

// Performs Geomerics' modified formula on all channels.
hvec3 evaluate_sh_l1_geomerics(hvec3 n, vec4 sh_coeffs[7]) {
	return hvec3(shEvaluateDiffuseL1Geomerics(n, sh_coeffs, 0),
			shEvaluateDiffuseL1Geomerics(n, sh_coeffs, 1),
			shEvaluateDiffuseL1Geomerics(n, sh_coeffs, 2));
}

// This is the same as evaluate_sh_l1_geomerics_inlined, but inlined so that RGB is evaluated
// in the same body. It also contains optimization notes.
hvec3 evaluate_sh_l1_geomerics_inlined(hvec3 dir, vec4 sh_coeffs[7]) {
	const hvec3 R0 = hvec3(sh_coeffs[0][0], sh_coeffs[1][0], sh_coeffs[2][0]);
	hvec3 R1[3];
	// [Optimization] We can bake the 0.5 into the coeffs in the Compute Shader that calculates sh_coeffs.
	R1[0] = half(0.5) * hvec3(sh_coeffs[0][3], sh_coeffs[0][1], sh_coeffs[0][2]);
	R1[1] = half(0.5) * hvec3(sh_coeffs[1][3], sh_coeffs[1][1], sh_coeffs[1][2]);
	R1[2] = half(0.5) * hvec3(sh_coeffs[2][3], sh_coeffs[2][1], sh_coeffs[2][2]);
	// [Optimization] lenR1 can be precomputed. But it's not needed (see lenR1 / R0).
	// [Optimization] R1[i] / lenR1[i] can be precomputed into R1[i].
	// [Optimization] lenR1 / R0 can be precomputed.But it's not needed (see p & a).
	// [Optimization] p and a can be precomputed.
	const hvec3 lenR1 = hvec3(length(R1[0]), length(R1[1]), length(R1[2]));
	const hvec3 q = half(0.5) * (half(1.0) + hvec3(dot(R1[0] / lenR1[0], dir), // r
													 dot(R1[1] / lenR1[1], dir), // g
													 dot(R1[2] / lenR1[2], dir))); // b
	const hvec3 p = half(1.0) + half(2.0) * lenR1 / R0;
	const hvec3 a = (half(1.0) - lenR1 / R0) / (half(1.0) + lenR1 / R0);

	return R0 * (a + (half(1.0) - a) * (p + half(1.0)) * hvec3(pow(q.x, p.x), pow(q.y, p.y), pow(q.z, p.z)));
}
#else

// See REFERENCE_SH_IMPL version for details.
// sh_coeffs no longer contains the SH coefficients, but rather prebaked versions to avoid
// uniform-on-uniform math. See "[Optimization]" in evaluate_sh_l1_geomerics_inlined.
hvec3 evaluate_sh_l1_geomerics(hvec3 dir, vec4 sh_coeffs[7]) {
	// Load from uniforms.
	const hvec3 R0 = hvec3(sh_coeffs[0].xyz);
	hvec3 R1_div_lenR1[3];
	R1_div_lenR1[0] = hvec3(sh_coeffs[0].w, sh_coeffs[1].xy);
	R1_div_lenR1[1] = hvec3(sh_coeffs[1].zw, sh_coeffs[2].x);
	R1_div_lenR1[2] = hvec3(sh_coeffs[2].yzw);
	const hvec3 p = hvec3(sh_coeffs[3].xyz);
	const hvec3 a = hvec3(sh_coeffs[3].w, sh_coeffs[4].xy);

	// Actual math.
	hvec3 q = half(0.5) * (half(1.0) + hvec3(dot(R1_div_lenR1[0], dir), // r
											   dot(R1_div_lenR1[1], dir), // g
											   dot(R1_div_lenR1[2], dir))); // b
	// q should be in range [0; 1] but due to floating point woes, it may land out of
	// range as a small dot at the poles. Negative values will produce NaNs.
	q = clamp(q, hvec3(0.0), hvec3(1.0));

	return R0 * (a + (half(1.0) - a) * (p + half(1.0)) * hvec3(pow(q.x, p.x), pow(q.y, p.y), pow(q.z, p.z)));
}
#endif

hvec3 evaluate_sh_l2(hvec3 dir, vec4 sh_coeffs[7]) {
	const hvec3 resl0 = hvec3(sh_coeffs[0].xyz);
	const hvec3 resl1n = hvec3(sh_coeffs[1].xyz);
	const hvec3 resl10 = hvec3(sh_coeffs[2].xyz);
	const hvec3 resl1p = hvec3(sh_coeffs[0].w, sh_coeffs[1].w, sh_coeffs[2].w);
	const hvec3 resl2n2 = hvec3(sh_coeffs[3].xyz);
	const hvec3 resl2n1 = hvec3(sh_coeffs[4].xyz);
	const hvec3 resl200 = hvec3(sh_coeffs[5].xyz);
	const hvec3 resl2p1 = hvec3(sh_coeffs[3].w, sh_coeffs[4].w, sh_coeffs[5].w);
	const hvec3 resl2p2 = hvec3(sh_coeffs[6].xyz);

	return resl0 //
			+ resl1n * dir.y //
			+ resl10 * dir.z //
			+ resl1p * dir.x //
			+ resl2n2 * dir.x * dir.y //
			+ resl2n1 * dir.y * dir.z //
			+ resl200 * (half(3.0) * dir.z * dir.z - half(1.0)) //
			+ resl2p1 * dir.x * dir.z //
			+ resl2p2 * (dir.x * dir.x - dir.y * dir.y);
}
