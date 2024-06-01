// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Math/Vec8.h>
#include <Jolt/Geometry/RayAABox.h>

JPH_NAMESPACE_BEGIN

/// Intersect 8 AABBs with ray, returns minimal distance along ray or FLT_MAX if no hit
/// Note: Can return negative value if ray starts in box
JPH_INLINE Vec8 RayAABox8(Vec3Arg inOrigin, const RayInvDirection &inInvDirection, Vec8Arg inBoundsMinX, Vec8Arg inBoundsMinY, Vec8Arg inBoundsMinZ, Vec8Arg inBoundsMaxX, Vec8Arg inBoundsMaxY, Vec8Arg inBoundsMaxZ)
{
	// Constants
	Vec8 flt_min = Vec8::sReplicate(-FLT_MAX);
	Vec8 flt_max = Vec8::sReplicate(FLT_MAX);

	// Origin
	Vec8 originx = Vec8::sSplatX(Vec4(inOrigin));
	Vec8 originy = Vec8::sSplatY(Vec4(inOrigin));
	Vec8 originz = Vec8::sSplatZ(Vec4(inOrigin));

	// Parallel
	UVec8 parallelx = UVec8::sSplatX(inInvDirection.mIsParallel);
	UVec8 parallely = UVec8::sSplatY(inInvDirection.mIsParallel);
	UVec8 parallelz = UVec8::sSplatZ(inInvDirection.mIsParallel);

	// Inverse direction
	Vec8 invdirx = Vec8::sSplatX(Vec4(inInvDirection.mInvDirection));
	Vec8 invdiry = Vec8::sSplatY(Vec4(inInvDirection.mInvDirection));
	Vec8 invdirz = Vec8::sSplatZ(Vec4(inInvDirection.mInvDirection));

	// Test against all three axii simultaneously.
	Vec8 t1x = (inBoundsMinX - originx) * invdirx;
	Vec8 t1y = (inBoundsMinY - originy) * invdiry;
	Vec8 t1z = (inBoundsMinZ - originz) * invdirz;
	Vec8 t2x = (inBoundsMaxX - originx) * invdirx;
	Vec8 t2y = (inBoundsMaxY - originy) * invdiry;
	Vec8 t2z = (inBoundsMaxZ - originz) * invdirz;

	// Compute the max of min(t1,t2) and the min of max(t1,t2) ensuring we don't
	// use the results from any directions parallel to the slab.
	Vec8 t_minx = Vec8::sSelect(Vec8::sMin(t1x, t2x), flt_min, parallelx);
	Vec8 t_miny = Vec8::sSelect(Vec8::sMin(t1y, t2y), flt_min, parallely);
	Vec8 t_minz = Vec8::sSelect(Vec8::sMin(t1z, t2z), flt_min, parallelz);
	Vec8 t_maxx = Vec8::sSelect(Vec8::sMax(t1x, t2x), flt_max, parallelx);
	Vec8 t_maxy = Vec8::sSelect(Vec8::sMax(t1y, t2y), flt_max, parallely);
	Vec8 t_maxz = Vec8::sSelect(Vec8::sMax(t1z, t2z), flt_max, parallelz);

	// t_min.xyz = maximum(t_min.x, t_min.y, t_min.z);
	Vec8 t_min = Vec8::sMax(Vec8::sMax(t_minx, t_miny), t_minz);

	// t_max.xyz = minimum(t_max.x, t_max.y, t_max.z);
	Vec8 t_max = Vec8::sMin(Vec8::sMin(t_maxx, t_maxy), t_maxz);

	// if (t_min > t_max) return FLT_MAX;
	UVec8 no_intersection = Vec8::sGreater(t_min, t_max);

	// if (t_max < 0.0f) return FLT_MAX;
	no_intersection = UVec8::sOr(no_intersection, Vec8::sLess(t_max, Vec8::sZero()));

	// if bounds are invalid return FLOAT_MAX;
	UVec8 bounds_invalid = UVec8::sOr(UVec8::sOr(Vec8::sGreater(inBoundsMinX, inBoundsMaxX), Vec8::sGreater(inBoundsMinY, inBoundsMaxY)), Vec8::sGreater(inBoundsMinZ, inBoundsMaxZ));
	no_intersection = UVec8::sOr(no_intersection, bounds_invalid);

	// if (inInvDirection.mIsParallel && !(Min <= inOrigin && inOrigin <= Max)) return FLT_MAX; else return t_min;
	UVec8 no_parallel_overlapx = UVec8::sAnd(parallelx, UVec8::sOr(Vec8::sLess(originx, inBoundsMinX), Vec8::sGreater(originx, inBoundsMaxX)));
	UVec8 no_parallel_overlapy = UVec8::sAnd(parallely, UVec8::sOr(Vec8::sLess(originy, inBoundsMinY), Vec8::sGreater(originy, inBoundsMaxY)));
	UVec8 no_parallel_overlapz = UVec8::sAnd(parallelz, UVec8::sOr(Vec8::sLess(originz, inBoundsMinZ), Vec8::sGreater(originz, inBoundsMaxZ)));
	no_intersection = UVec8::sOr(no_intersection, UVec8::sOr(UVec8::sOr(no_parallel_overlapx, no_parallel_overlapy), no_parallel_overlapz));
	return Vec8::sSelect(t_min, flt_max, no_intersection);
}

JPH_NAMESPACE_END
