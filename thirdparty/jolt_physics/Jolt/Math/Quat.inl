// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

JPH_NAMESPACE_BEGIN

Quat Quat::operator * (QuatArg inRHS) const
{
#if defined(JPH_USE_SSE4_1)
	// Taken from: http://momchil-velikov.blogspot.nl/2013/10/fast-sse-quternion-multiplication.html
	__m128 abcd = mValue.mValue;
	__m128 xyzw = inRHS.mValue.mValue;

	__m128 t0 = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(3, 3, 3, 3));
	__m128 t1 = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(2, 3, 0, 1));

	__m128 t3 = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 t4 = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(1, 0, 3, 2));

	__m128 t5 = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 t6 = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(2, 0, 3, 1));

	// [d,d,d,d] * [z,w,x,y] = [dz,dw,dx,dy]
	__m128 m0 = _mm_mul_ps(t0, t1);

	// [a,a,a,a] * [y,x,w,z] = [ay,ax,aw,az]
	__m128 m1 = _mm_mul_ps(t3, t4);

	// [b,b,b,b] * [z,x,w,y] = [bz,bx,bw,by]
	__m128 m2 = _mm_mul_ps(t5, t6);

	// [c,c,c,c] * [w,z,x,y] = [cw,cz,cx,cy]
	__m128 t7 = _mm_shuffle_ps(abcd, abcd, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 t8 = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(3, 2, 0, 1));
	__m128 m3 = _mm_mul_ps(t7, t8);

	// [dz,dw,dx,dy] + -[ay,ax,aw,az] = [dz+ay,dw-ax,dx+aw,dy-az]
	__m128 e = _mm_addsub_ps(m0, m1);

	// [dx+aw,dz+ay,dy-az,dw-ax]
	e = _mm_shuffle_ps(e, e, _MM_SHUFFLE(1, 3, 0, 2));

	// [dx+aw,dz+ay,dy-az,dw-ax] + -[bz,bx,bw,by] = [dx+aw+bz,dz+ay-bx,dy-az+bw,dw-ax-by]
	e = _mm_addsub_ps(e, m2);

	// [dz+ay-bx,dw-ax-by,dy-az+bw,dx+aw+bz]
	e = _mm_shuffle_ps(e, e, _MM_SHUFFLE(2, 0, 1, 3));

	// [dz+ay-bx,dw-ax-by,dy-az+bw,dx+aw+bz] + -[cw,cz,cx,cy] = [dz+ay-bx+cw,dw-ax-by-cz,dy-az+bw+cx,dx+aw+bz-cy]
	e = _mm_addsub_ps(e, m3);

	// [dw-ax-by-cz,dz+ay-bx+cw,dy-az+bw+cx,dx+aw+bz-cy]
	return Quat(Vec4(_mm_shuffle_ps(e, e, _MM_SHUFFLE(2, 3, 1, 0))));
#else
	float lx = mValue.GetX();
	float ly = mValue.GetY();
	float lz = mValue.GetZ();
	float lw = mValue.GetW();

	float rx = inRHS.mValue.GetX();
	float ry = inRHS.mValue.GetY();
	float rz = inRHS.mValue.GetZ();
	float rw = inRHS.mValue.GetW();

	float x = lw * rx + lx * rw + ly * rz - lz * ry;
	float y = lw * ry - lx * rz + ly * rw + lz * rx;
	float z = lw * rz + lx * ry - ly * rx + lz * rw;
	float w = lw * rw - lx * rx - ly * ry - lz * rz;

	return Quat(x, y, z, w);
#endif
}

Quat Quat::sMultiplyImaginary(Vec3Arg inLHS, QuatArg inRHS)
{
#if defined(JPH_USE_SSE4_1)
	__m128 abc0 = inLHS.mValue;
	__m128 xyzw = inRHS.mValue.mValue;

	// [a,a,a,a] * [w,y,z,x] = [aw,ay,az,ax]
	__m128 aaaa = _mm_shuffle_ps(abc0, abc0, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 xzyw = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(3, 1, 2, 0));
	__m128 axazayaw = _mm_mul_ps(aaaa, xzyw);

	// [b,b,b,b] * [z,x,w,y] = [bz,bx,bw,by]
	__m128 bbbb = _mm_shuffle_ps(abc0, abc0, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 ywxz = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(2, 0, 3, 1));
	__m128 bybwbxbz = _mm_mul_ps(bbbb, ywxz);

	// [c,c,c,c] * [w,z,x,y] = [cw,cz,cx,cy]
	__m128 cccc = _mm_shuffle_ps(abc0, abc0, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 yxzw = _mm_shuffle_ps(xyzw, xyzw, _MM_SHUFFLE(3, 2, 0, 1));
	__m128 cycxczcw = _mm_mul_ps(cccc, yxzw);

	// [+aw,+ay,-az,-ax]
	__m128 e = _mm_xor_ps(axazayaw, _mm_set_ps(0.0f, 0.0f, -0.0f, -0.0f));

	// [+aw,+ay,-az,-ax] + -[bz,bx,bw,by] = [+aw+bz,+ay-bx,-az+bw,-ax-by]
	e = _mm_addsub_ps(e, bybwbxbz);

	// [+ay-bx,-ax-by,-az+bw,+aw+bz]
	e = _mm_shuffle_ps(e, e, _MM_SHUFFLE(2, 0, 1, 3));

	// [+ay-bx,-ax-by,-az+bw,+aw+bz] + -[cw,cz,cx,cy] = [+ay-bx+cw,-ax-by-cz,-az+bw+cx,+aw+bz-cy]
	e = _mm_addsub_ps(e, cycxczcw);

	// [-ax-by-cz,+ay-bx+cw,-az+bw+cx,+aw+bz-cy]
	return Quat(Vec4(_mm_shuffle_ps(e, e, _MM_SHUFFLE(2, 3, 1, 0))));
#else
	float lx = inLHS.GetX();
	float ly = inLHS.GetY();
	float lz = inLHS.GetZ();

	float rx = inRHS.mValue.GetX();
	float ry = inRHS.mValue.GetY();
	float rz = inRHS.mValue.GetZ();
	float rw = inRHS.mValue.GetW();

	float x =  (lx * rw) + ly * rz - lz * ry;
	float y = -(lx * rz) + ly * rw + lz * rx;
	float z =  (lx * ry) - ly * rx + lz * rw;
	float w = -(lx * rx) - ly * ry - lz * rz;

	return Quat(x, y, z, w);
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

	float len_v1_v2 = sqrt(inFrom.LengthSq() * inTo.LengthSq());
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
	std::uniform_real_distribution<float> zero_to_one(0.0f, 1.0f);
	float x0 = zero_to_one(inRandom);
	float r1 = sqrt(1.0f - x0), r2 = sqrt(x0);
	std::uniform_real_distribution<float> zero_to_two_pi(0.0f, 2.0f * JPH_PI);
	Vec4 s, c;
	Vec4(zero_to_two_pi(inRandom), zero_to_two_pi(inRandom), 0, 0).SinCos(s, c);
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
	float y_sq = GetY() * GetY();

	// X
	float t0 = 2.0f * (GetW() * GetX() + GetY() * GetZ());
	float t1 = 1.0f - 2.0f * (GetX() * GetX() + y_sq);

	// Y
	float t2 = 2.0f * (GetW() * GetY() - GetZ() * GetX());
	t2 = t2 > 1.0f? 1.0f : t2;
	t2 = t2 < -1.0f? -1.0f : t2;

	// Z
	float t3 = 2.0f * (GetW() * GetZ() + GetX() * GetY());
	float t4 = 1.0f - 2.0f * (y_sq + GetZ() * GetZ());

	return Vec3(ATan2(t0, t1), ASin(t2), ATan2(t3, t4));
}

Quat Quat::GetTwist(Vec3Arg inAxis) const
{
	Quat twist(Vec4(GetXYZ().Dot(inAxis) * inAxis, GetW()));
	float twist_len = twist.LengthSq();
	if (twist_len != 0.0f)
		return twist / sqrt(twist_len);
	else
		return Quat::sIdentity();
}

void Quat::GetSwingTwist(Quat &outSwing, Quat &outTwist) const
{
	float x = GetX(), y = GetY(), z = GetZ(), w = GetW();
	float s = sqrt(Square(w) + Square(x));
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
	return Quat(Vec4::sReplicate(scale0) * mValue + Vec4::sReplicate(inFraction) * inDestination.mValue);
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
	return Quat(Vec4::sReplicate(scale0) * mValue + Vec4::sReplicate(scale1) * inDestination.mValue).Normalized();
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
	float w = sqrt(max(1.0f - v.LengthSq(), 0.0f)); // It is possible that the length of v is a fraction above 1, and we don't want to introduce NaN's in that case so we clamp to 0
	return Quat(Vec4(v, w));
}

JPH_NAMESPACE_END
