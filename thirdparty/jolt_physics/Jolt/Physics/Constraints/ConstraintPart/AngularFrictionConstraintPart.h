// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Constraints/ConstraintPart/ContactConstraintPart.h>

JPH_NAMESPACE_BEGIN

/// Decide which members this constraint part needs based on motion type
template <EMotionType Type1>
class AngularFrictionConstraintPart1 : public ContactConstraintPart1<EMotionType::Static>
{
};

template <>
class AngularFrictionConstraintPart1<EMotionType::Dynamic> : public AngularFrictionConstraintPart1<EMotionType::Static>
{
protected:
	// Note: Constructor will not be called
	Float3						mInvI1_Axis;
};

template <EMotionType Type2>
class AngularFrictionConstraintPart2
{
};

template <>
class AngularFrictionConstraintPart2<EMotionType::Dynamic> : public AngularFrictionConstraintPart2<EMotionType::Static>
{
protected:
	// Note: Constructor will not be called
	Float3						mInvI2_Axis;
};

/// This is a copy of AngleConstraintPart, specialized to handle contact constraints. See the documentation of AngleConstraintPart for more documentation behind the math.
template <EMotionType Type1, EMotionType Type2>
class AngularFrictionConstraintPart : public AngularFrictionConstraintPart1<Type1>, public AngularFrictionConstraintPart2<Type2>
{
	/// Internal helper function to update velocities of bodies after Lagrange multiplier is calculated
	JPH_INLINE bool				ApplyVelocityStep(Vec3 &ioAngularVelocity1, Vec3 &ioAngularVelocity2, float inLambda) const
	{
		// Apply impulse if delta is not zero
		if (inLambda != 0.0f)
		{
			if constexpr (Type1 == EMotionType::Dynamic)
				ioAngularVelocity1 -= inLambda * Vec3::sLoadFloat3Unsafe(this->mInvI1_Axis);
			if constexpr (Type2 == EMotionType::Dynamic)
				ioAngularVelocity2 += inLambda * Vec3::sLoadFloat3Unsafe(this->mInvI2_Axis);
			return true;
		}

		return false;
	}

public:
	/// See: AngleConstraintPart::CalculateConstraintProperties
	inline void					CalculateConstraintProperties(Mat44Arg inInvI1, Mat44Arg inInvI2, Vec3Arg inWorldSpaceAxis, float inBias = 0.0f)
	{
		JPH_ASSERT(inWorldSpaceAxis.IsNormalized(1.0e-4f));

		// Store bias
		mBias = inBias;

		Vec3 invi1_axis, invi2_axis;
		if constexpr (Type1 == EMotionType::Dynamic)
		{
			invi1_axis = inInvI1.Multiply3x3(inWorldSpaceAxis);
			invi1_axis.StoreFloat3(&this->mInvI1_Axis);
		}
		if constexpr (Type2 == EMotionType::Dynamic)
		{
			invi2_axis = inInvI2.Multiply3x3(inWorldSpaceAxis);
			invi2_axis.StoreFloat3(&this->mInvI2_Axis);
		}

		float inv_effective_mass = 0.0f;
		if constexpr (Type1 == EMotionType::Dynamic && Type2 == EMotionType::Dynamic)
			inv_effective_mass = inWorldSpaceAxis.Dot(invi1_axis + invi2_axis);
		else if constexpr (Type1 == EMotionType::Dynamic)
			inv_effective_mass = inWorldSpaceAxis.Dot(invi1_axis);
		else if constexpr (Type2 == EMotionType::Dynamic)
			inv_effective_mass = inWorldSpaceAxis.Dot(invi2_axis);
		else
			JPH_ASSERT(false); // Static vs static is nonsensical!

		if (inv_effective_mass == 0.0f)
			this->Deactivate();
		else
			this->mEffectiveMass = 1.0f / inv_effective_mass;
	}

	/// See: AngleConstraintPart::WarmStart
	inline bool					WarmStart(Vec3 &ioAngularVelocity1, Vec3 &ioAngularVelocity2, float inWarmStartImpulseRatio)
	{
		this->mTotalLambda *= inWarmStartImpulseRatio;
		return ApplyVelocityStep(ioAngularVelocity1, ioAngularVelocity2, this->mTotalLambda);
	}

	/// See: AngleConstraintPart::SolveVelocityConstraint
	inline bool					SolveVelocityConstraint(Vec3 &ioAngularVelocity1, Vec3 &ioAngularVelocity2, Vec3Arg inWorldSpaceAxis, float inMinLambda, float inMaxLambda)
	{
		float jv;
		if constexpr (Type1 != EMotionType::Static && Type2 != EMotionType::Static)
			jv = inWorldSpaceAxis.Dot(ioAngularVelocity1 - ioAngularVelocity2);
		else if constexpr (Type1 != EMotionType::Static)
			jv = inWorldSpaceAxis.Dot(ioAngularVelocity1);
		else if constexpr (Type2 != EMotionType::Static)
			jv = -inWorldSpaceAxis.Dot(ioAngularVelocity2);
		else
			JPH_ASSERT(false); // Static vs static is nonsensical!

		float lambda = this->mEffectiveMass * (jv - mBias);
		float new_lambda = Clamp(this->mTotalLambda + lambda, inMinLambda, inMaxLambda); // Clamp impulse
		lambda = new_lambda - this->mTotalLambda; // Lambda potentially got clamped, calculate the new impulse to apply
		this->mTotalLambda = new_lambda; // Store accumulated impulse

		return ApplyVelocityStep(ioAngularVelocity1, ioAngularVelocity2, lambda);
	}

private:
	// Note: Constructor will not be called. This serves as 1 extra float so we can read the previous member using Vec3::sLoadFloat3Unsafe
	float						mBias;
};

static_assert(sizeof(AngularFrictionConstraintPart<EMotionType::Dynamic, EMotionType::Dynamic>) == 3 * sizeof(float) + 2 * sizeof(Float3));
static_assert(sizeof(AngularFrictionConstraintPart<EMotionType::Dynamic, EMotionType::Kinematic>) == 3 * sizeof(float) + sizeof(Float3));
static_assert(sizeof(AngularFrictionConstraintPart<EMotionType::Dynamic, EMotionType::Static>) == 3 * sizeof(float) + sizeof(Float3));
static_assert(sizeof(AngularFrictionConstraintPart<EMotionType::Kinematic, EMotionType::Dynamic>) == 3 * sizeof(float) + sizeof(Float3));
static_assert(sizeof(AngularFrictionConstraintPart<EMotionType::Kinematic, EMotionType::Kinematic>) == 3 * sizeof(float));
static_assert(sizeof(AngularFrictionConstraintPart<EMotionType::Kinematic, EMotionType::Static>) == 3 * sizeof(float));
static_assert(sizeof(AngularFrictionConstraintPart<EMotionType::Static, EMotionType::Dynamic>) == 3 * sizeof(float) + sizeof(Float3));
static_assert(sizeof(AngularFrictionConstraintPart<EMotionType::Static, EMotionType::Kinematic>) == 3 * sizeof(float));
static_assert(sizeof(AngularFrictionConstraintPart<EMotionType::Static, EMotionType::Static>) == 3 * sizeof(float));

JPH_NAMESPACE_END
