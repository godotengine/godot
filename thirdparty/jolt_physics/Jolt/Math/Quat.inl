// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

JPH_NAMESPACE_BEGIN

Quat Quat::operator * (QuatArg inRHS) const
{
#ifdef JPH_USE_SSE
	__m128 abcd = mValue.mValue;
	__m128 xyzw = inRHS.mValue.mValue;

	// Names based on logical order, opposite of shuffle order.
	__m128 abca = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(0, 2, 1, 0));
	__m128 bcab = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(1, 0, 2, 1));
	__m128 cabc = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(2, 1, 0, 2));
	__m128 dddd = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(3, 3, 3, 3));

	__m128 wwwx = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(0, 3, 3, 3));
	__m128 zxyy = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(1, 1, 0, 2));
	__m128 yzxz = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(2, 0, 2, 1));

	__m128 m2 = _mm_mul_ps(bcab, zxyy);
#ifdef JPH_USE_FMADD
	__m128 m3 = _mm_fmadd_ps(abca, wwwx, m2);
#else
	__m128 m1 = _mm_mul_ps(abca, wwwx);
	__m128 m3 = _mm_add_ps(m1, m2);
#endif

	// Negate last (logical) component.
	m3 = _mm_xor_ps(_mm_set_ps(-0.0f, 0.0f, 0.0f, 0.0f), m3);

#ifdef JPH_USE_FMADD
	__m128 m5 = _mm_fnmadd_ps(cabc, yzxz, m3);
	__m128 m7 = _mm_fmadd_ps(dddd, xyzw, m5);
#else
	__m128 m4 = _mm_mul_ps(dddd, xyzw);
	__m128 m5 = _mm_mul_ps(cabc, yzxz);
	__m128 m6 = _mm_sub_ps(m4, m5);
	__m128 m7 = _mm_add_ps(m3, m6);
#endif

	// [(aw+bz)+(dx-cy),(bw+cx)+(dy-az),(cw+ay)+(dz-bx),-(ax+by)+(dw-cz)]
	return Quat(Vec4(m7));
#elif defined(JPH_USE_NEON)
	float32x4_t abcd = mValue.mValue;
	float32x4_t xyzw = inRHS.mValue.mValue;

	float32x4_t abca = vcopyq_laneq_f32(abcd, 3, abcd, 0);
	float32x4_t bcab = JPH_NEON_SHUFFLE_F32x4(abcd, abcd, 1, 2, 0, 1);
	float32x4_t cabc = JPH_NEON_SHUFFLE_F32x4(abcd, abcd, 2, 0, 1, 2);
	float32x4_t dddd = vdupq_laneq_f32(abcd, 3);

	float32x4_t wwwx = vcopyq_laneq_f32(vdupq_laneq_f32(xyzw, 3), 3, xyzw, 0);
	float32x4_t zxyy = JPH_NEON_SHUFFLE_F32x4(xyzw, xyzw, 2, 0, 1, 1);
	float32x4_t yzxz = JPH_NEON_SHUFFLE_F32x4(xyzw, xyzw, 1, 2, 0, 2);

	float32x4_t m1 = vmulq_f32(abca, wwwx);
	float32x4_t m2 = vmulq_f32(bcab, zxyy);
	float32x4_t m3 = vaddq_f32(m1, m2);

	uint32x4_t w_neg_mask = JPH_NEON_UINT32x4(0, 0, 0, 0x80000000u);
	m3 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(m3), w_neg_mask));

	float32x4_t m4 = vmulq_f32(dddd, xyzw);
	float32x4_t m5 = vmulq_f32(cabc, yzxz);
	float32x4_t m6 = vsubq_f32(m4, m5);
	float32x4_t m7 = vaddq_f32(m3, m6);

	return Quat(Vec4(m7));
#else
	float a = mValue.GetX();
	float b = mValue.GetY();
	float c = mValue.GetZ();
	float d = mValue.GetW();

	float x = inRHS.mValue.GetX();
	float y = inRHS.mValue.GetY();
	float z = inRHS.mValue.GetZ();
	float w = inRHS.mValue.GetW();

	return Quat((a * w + b * z) + (d * x - c * y),
				(b * w + c * x) + (d * y - a * z),
				(c * w + a * y) + (d * z - b * x),
				-(a * x + b * y) + (d * w - c * z));
#endif
}

Quat Quat::sMultiplyImaginary(Vec3Arg inLHS, QuatArg inRHS)
{
#ifdef JPH_USE_SSE
	__m128 abc0 = inLHS.mValue;
	__m128 xyzw = inRHS.mValue.mValue;

	// Names based on logical order, opposite of shuffle order.
	__m128 abca = _mm_shuffle_ps(abc0, abc0, _MM_SHUFFLE(0, 2, 1, 0));
	__m128 bcab = _mm_shuffle_ps(abc0, abc0, _MM_SHUFFLE(1, 0, 2, 1));
	__m128 cabc = _mm_shuffle_ps(abc0, abc0, _MM_SHUFFLE(2, 1, 0, 2));

	__m128 wwwx = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(0, 3, 3, 3));
	__m128 zxyy = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(1, 1, 0, 2));
	__m128 yzxz = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(2, 0, 2, 1));

	__m128 m2 = _mm_mul_ps(bcab, zxyy);
#ifdef JPH_USE_FMADD
	__m128 m3 = _mm_fmadd_ps(abca, wwwx, m2);
#else
	__m128 m1 = _mm_mul_ps(abca, wwwx);
	__m128 m3 = _mm_add_ps(m1, m2);
#endif

	// Negate last (logical) component.
	m3 = _mm_xor_ps(_mm_set_ps(-0.0f, 0.0f, 0.0f, 0.0f), m3);

	__m128 m4 = _mm_mul_ps(cabc, yzxz);

	// [(aw+bz)-cy,(bw+cx)-az,(cw+ay)-bx,-(ax+by)-cz]
	return Quat(Vec4(_mm_sub_ps(m3, m4)));
#elif defined(JPH_USE_NEON)
	float32x4_t abc0 = inLHS.mValue;
	float32x4_t xyzw = inRHS.mValue.mValue;

	float32x4_t abca = vcopyq_laneq_f32(abc0, 3, abc0, 0);
	float32x4_t bcab = JPH_NEON_SHUFFLE_F32x4(abc0, abc0, 1, 2, 0, 1);
	float32x4_t cabc = JPH_NEON_SHUFFLE_F32x4(abc0, abc0, 2, 0, 1, 2);

	float32x4_t wwwx = vcopyq_laneq_f32(vdupq_laneq_f32(xyzw, 3), 3, xyzw, 0);
	float32x4_t zxyy = JPH_NEON_SHUFFLE_F32x4(xyzw, xyzw, 2, 0, 1, 1);
	float32x4_t yzxz = JPH_NEON_SHUFFLE_F32x4(xyzw, xyzw, 1, 2, 0, 2);

	float32x4_t m1 = vmulq_f32(abca, wwwx);
	float32x4_t m2 = vmulq_f32(bcab, zxyy);
	float32x4_t m3 = vaddq_f32(m1, m2);

	uint32x4_t w_neg_mask = JPH_NEON_UINT32x4(0, 0, 0, 0x80000000u);
	m3 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(m3), w_neg_mask));

	float32x4_t m4 = vmulq_f32(cabc, yzxz);
	float32x4_t m7 = vsubq_f32(m3, m4);

	return Quat(Vec4(m7));
#else
	float a = inLHS.GetX();
	float b = inLHS.GetY();
	float c = inLHS.GetZ();

	float x = inRHS.mValue.GetX();
	float y = inRHS.mValue.GetY();
	float z = inRHS.mValue.GetZ();
	float w = inRHS.mValue.GetW();

	return Quat((a * w + b * z) - c * y,
				(b * w + c * x) - a * z,
				(c * w + a * y) - b * x,
				-(a * x + b * y) - c * z);
#endif
}

Quat Quat::sRotation(Vec3Arg inAxis, float inAngle)
{
	// returns [inAxis * sin(0.5f * inAngle), cos(0.5f * inAngle)]
	JPH_ASSERT(inAxis.IsNormalized());
	Vec4 s, c;
	Vec4::sReplicate(0.5f * inAngle).SinCos(s, c);
	return Quat(Vec4::sSelect(Vec4(inAxis) * s, c, UVec4(0, 0, 0, 0xffffffffU)));
}

void Quat::GetAxisAngle(Vec3 &outAxis, float &outAngle) const
{
	JPH_ASSERT(IsNormalized());
	Quat w_pos = EnsureWPositive();
	float abs_w = w_pos.GetW();
	if (abs_w >= 1.0f)
	{
		outAxis = Vec3::sZero();
		outAngle = 0.0f;
	}
	else
	{
		outAngle = 2.0f * ACos(abs_w);
		outAxis = w_pos.GetXYZ().NormalizedOr(Vec3::sZero());
	}
}

Vec3 Quat::GetAngularVelocity(float inDeltaTime) const
{
	JPH_ASSERT(IsNormalized());

	// w = cos(angle / 2), ensure it is positive so that we get an angle in the range [0, PI]
	Quat w_pos = EnsureWPositive();

	// The imaginary part of the quaternion is axis * sin(angle / 2),
	// if the length is small use the approximation sin(x) = x to calculate angular velocity
	Vec3 xyz = w_pos.GetXYZ();
	float xyz_len_sq = xyz.LengthSq();
	if (xyz_len_sq < 4.0e-4f) // Max error introduced is sin(0.02) - 0.02 = 7e-5 (when w is near 1 the angle becomes more inaccurate in the code below, so don't make this number too small)
		return (2.0f / inDeltaTime) * xyz;

	// Otherwise calculate the angle from w = cos(angle / 2) and determine the axis by normalizing the imaginary part
	// Note that it is also possible to calculate the angle through angle = 2 * atan2(|xyz|, w). This is more accurate but also 2x as expensive.
	float angle = 2.0f * ACos(w_pos.GetW());
	return (xyz / (Sqrt(xyz_len_sq) * inDeltaTime)) * angle;
}

Quat Quat::sFromTo(Vec3Arg inFrom, Vec3Arg inTo)
{
	/*
		Uses (inFrom = v1, inTo = v2):

		angle = arcos(v1 . v2 / |v1||v2|)
		axis = normalize(v1 x v2)

		Quaternion is then:

		s = sin(angle / 2)
		x = axis.x * s
		y = axis.y * s
		z = axis.z * s
		w = cos(angle / 2)

		Using identities:

		sin(2 * a) = 2 * sin(a) * cos(a)
		cos(2 * a) = cos(a)^2 - sin(a)^2
		sin(a)^2 + cos(a)^2 = 1

		This reduces to:

		x = (v1 x v2).x
		y = (v1 x v2).y
		z = (v1 x v2).z
		w = |v1||v2| + v1 . v2

		which then needs to be normalized because the whole equation was multiplied by 2 cos(angle / 2)
	*/

	float len_v1_v2 = Sqrt(inFrom.LengthSq() * inTo.LengthSq());
	float w = len_v1_v2 + inFrom.Dot(inTo);

	if (w == 0.0f)
	{
		if (len_v1_v2 == 0.0f)
		{
			// If either of the vectors has zero length, there is no rotation and we return identity
			return Quat::sIdentity();
		}
		else
		{
			// If vectors are perpendicular, take one of the many 180 degree rotations that exist
			return Quat(Vec4(inFrom.GetNormalizedPerpendicular(), 0));
		}
	}

	Vec3 v = inFrom.Cross(inTo);
	return Quat(Vec4(v, w)).Normalized();
}

template <class Random>
Quat Quat::sRandom(Random &inRandom)
{
	// Using Uniform Random Rotations - Graphics Gems III - Ken Shoemake
	float x0 = float(inRandom() - inRandom.min()) / float(inRandom.max() - inRandom.min());
	float r1 = Sqrt(1.0f - x0), r2 = Sqrt(x0);
	float theta1 = 2.0f * JPH_PI * float(inRandom() - inRandom.min()) / float(inRandom.max() - inRandom.min());
	float theta2 = 2.0f * JPH_PI * float(inRandom() - inRandom.min()) / float(inRandom.max() - inRandom.min());
	Vec4 s, c;
	Vec4(theta1, theta2, 0, 0).SinCos(s, c);
	return Quat(s.GetX() * r1, c.GetX() * r1, s.GetY() * r2, c.GetY() * r2);
}

Quat Quat::sEulerAngles(Vec3Arg inAngles)
{
	Vec4 half(0.5f * inAngles);
	Vec4 s, c;
	half.SinCos(s, c);

	float cx = c.GetX();
	float sx = s.GetX();
	float cy = c.GetY();
	float sy = s.GetY();
	float cz = c.GetZ();
	float sz = s.GetZ();

	return Quat(
		cz * sx * cy - sz * cx * sy,
		cz * cx * sy + sz * sx * cy,
		sz * cx * cy - cz * sx * sy,
		cz * cx * cy + sz * sx * sy);
}

Vec3 Quat::GetEulerAngles() const
{
	float x = GetX(), y = GetY(), z = GetZ(), w = GetW();
	float y_sq = y * y;

	// X
	float t0 = 2.0f * (w * x + y * z);
	float t1 = 1.0f - 2.0f * (x * x + y_sq);

	// Y
	float t2 = 2.0f * (w * y - z * x);
	t2 = t2 > 1.0f? 1.0f : t2;
	t2 = t2 < -1.0f? -1.0f : t2;

	// Z
	float t3 = 2.0f * (w * z + x * y);
	float t4 = 1.0f - 2.0f * (y_sq + z * z);

	return Vec3(ATan2(t0, t1), ASin(t2), ATan2(t3, t4));
}

Quat Quat::GetTwist(Vec3Arg inAxis) const
{
	Quat twist(Vec4(GetXYZ().Dot(inAxis) * inAxis, GetW()));
	float twist_len = twist.LengthSq();
	if (twist_len != 0.0f)
		return twist / Sqrt(twist_len);
	else
		return Quat::sIdentity();
}

void Quat::GetSwingTwist(Quat &outSwing, Quat &outTwist) const
{
	float x = GetX(), y = GetY(), z = GetZ(), w = GetW();
	float s = Sqrt(Square(w) + Square(x));
	if (s != 0.0f)
	{
		outTwist = Quat(x / s, 0, 0, w / s);
		outSwing = Quat(0, (w * y - x * z) / s, (w * z + x * y) / s, s);
	}
	else
	{
		// If both x and w are zero, this must be a 180 degree rotation around either y or z
		outTwist = Quat::sIdentity();
		outSwing = *this;
	}
}

Quat Quat::LERP(QuatArg inDestination, float inFraction) const
{
	float scale0 = 1.0f - inFraction;
	return Quat(scale0 * mValue + inFraction * inDestination.mValue);
}

Quat Quat::SLERP(QuatArg inDestination, float inFraction) const
{
	// Difference at which to LERP instead of SLERP
	const float delta = 0.0001f;

	// Calc cosine
	float sign_scale1 = 1.0f;
	float cos_omega = Dot(inDestination);

	// Adjust signs (if necessary)
	if (cos_omega < 0.0f)
	{
		cos_omega = -cos_omega;
		sign_scale1 = -1.0f;
	}

	// Calculate coefficients
	float scale0, scale1;
	if (1.0f - cos_omega > delta)
	{
		// Standard case (slerp)
		float omega = ACos(cos_omega);
		float sin_omega = Sin(omega);
		scale0 = Sin((1.0f - inFraction) * omega) / sin_omega;
		scale1 = sign_scale1 * Sin(inFraction * omega) / sin_omega;
	}
	else
	{
		// Quaternions are very close so we can do a linear interpolation
		scale0 = 1.0f - inFraction;
		scale1 = sign_scale1 * inFraction;
	}

	// Interpolate between the two quaternions
	return Quat(scale0 * mValue + scale1 * inDestination.mValue).Normalized();
}

Vec3 Quat::operator * (Vec3Arg inValue) const
{
	// Rotating a vector by a quaternion is done by: p' = q * (p, 0) * q^-1 (q^-1 = conjugated(q) for a unit quaternion)
	// Using Rodrigues formula: https://en.m.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula
	// This is equivalent to: p' = p + 2 * (q.w * q.xyz x p + q.xyz x (q.xyz x p))
	//
	// This is:
	//
	// Vec3 xyz = GetXYZ();
	// Vec3 q_cross_p = xyz.Cross(inValue);
	// Vec3 q_cross_q_cross_p = xyz.Cross(q_cross_p);
	// Vec3 v = mValue.SplatW3() * q_cross_p + q_cross_q_cross_p;
	// return inValue + (v + v);
	//
	// But we can write out the cross products in a more efficient way:
	JPH_ASSERT(IsNormalized());
	Vec3 xyz = GetXYZ();
	Vec3 yzx = xyz.Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>();
	Vec3 q_cross_p = (inValue.Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>() * xyz - yzx * inValue).Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>();
	Vec3 q_cross_q_cross_p = (q_cross_p.Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>() * xyz - yzx * q_cross_p).Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>();
	Vec3 v = mValue.SplatW3() * q_cross_p + q_cross_q_cross_p;
	return inValue + (v + v);
}

Vec3 Quat::InverseRotate(Vec3Arg inValue) const
{
	JPH_ASSERT(IsNormalized());
	Vec3 xyz = GetXYZ(); // Needs to be negated, but we do this in the equations below
	Vec3 yzx = xyz.Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>();
	Vec3 q_cross_p = (yzx * inValue - inValue.Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>() * xyz).Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>();
	Vec3 q_cross_q_cross_p = (yzx * q_cross_p - q_cross_p.Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>() * xyz).Swizzle<SWIZZLE_Y, SWIZZLE_Z, SWIZZLE_X>();
	Vec3 v = mValue.SplatW3() * q_cross_p + q_cross_q_cross_p;
	return inValue + (v + v);
}

Vec3 Quat::RotateAxisX() const
{
	// This is *this * Vec3::sAxisX() written out:
	JPH_ASSERT(IsNormalized());
	Vec4 t = mValue + mValue;
	return Vec3(t.SplatX() * mValue + (t.SplatW() * mValue.Swizzle<SWIZZLE_W, SWIZZLE_Z, SWIZZLE_Y, SWIZZLE_X>()).FlipSign<1, 1, -1, 1>() - Vec4(1, 0, 0, 0));
}

Vec3 Quat::RotateAxisY() const
{
	// This is *this * Vec3::sAxisY() written out:
	JPH_ASSERT(IsNormalized());
	Vec4 t = mValue + mValue;
	return Vec3(t.SplatY() * mValue + (t.SplatW() * mValue.Swizzle<SWIZZLE_Z, SWIZZLE_W, SWIZZLE_X, SWIZZLE_Y>()).FlipSign<-1, 1, 1, 1>() - Vec4(0, 1, 0, 0));
}

Vec3 Quat::RotateAxisZ() const
{
	// This is *this * Vec3::sAxisZ() written out:
	JPH_ASSERT(IsNormalized());
	Vec4 t = mValue + mValue;
	return Vec3(t.SplatZ() * mValue + (t.SplatW() * mValue.Swizzle<SWIZZLE_Y, SWIZZLE_X, SWIZZLE_W, SWIZZLE_Z>()).FlipSign<1, -1, 1, 1>() - Vec4(0, 0, 1, 0));
}

void Quat::StoreFloat3(Float3 *outV) const
{
	JPH_ASSERT(IsNormalized());
	EnsureWPositive().GetXYZ().StoreFloat3(outV);
}

void Quat::StoreFloat4(Float4 *outV) const
{
	mValue.StoreFloat4(outV);
}

Quat Quat::sLoadFloat3Unsafe(const Float3 &inV)
{
	Vec3 v = Vec3::sLoadFloat3Unsafe(inV);
	float w = Sqrt(max(1.0f - v.LengthSq(), 0.0f)); // It is possible that the length of v is a fraction above 1, and we don't want to introduce NaN's in that case so we clamp to 0
	return Quat(Vec4(v, w));
}

JPH_NAMESPACE_END
