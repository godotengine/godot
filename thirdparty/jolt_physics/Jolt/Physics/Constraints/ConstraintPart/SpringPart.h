// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Constraints/SpringSettings.h>

JPH_NAMESPACE_BEGIN
#ifndef JPH_PLATFORM_DOXYGEN // Somehow Doxygen gets confused and thinks the parameters to CalculateSpringProperties belong to this macro
JPH_MSVC_SUPPRESS_WARNING(4723) // potential divide by 0 - caused by line: outEffectiveMass = 1.0f / inInvEffectiveMass, note that JPH_NAMESPACE_BEGIN already pushes the warning state
#endif // !JPH_PLATFORM_DOXYGEN

/// Class used in other constraint parts to calculate the required bias factor in the lagrange multiplier for creating springs
class SpringPart
{
public:
	/// Turn off the spring and set a bias only
	///
	/// @param inBias Bias term (b) for the constraint impulse: lambda = J v + b
	JPH_INLINE void				CalculateSpringPropertiesWithBias(float inBias)
	{
		mSoftness = 0.0f;
		mBias = inBias;
	}

	/// Calculate spring properties with spring Stiffness (k) and damping (c), this is based on the spring equation: F = -k * x - c * v
	///
	/// @param inDeltaTime Time step
	/// @param inInvEffectiveMass Inverse effective mass K
	/// @param inBias Bias term (b) for the constraint impulse: lambda = J v + b
	///	@param inC Value of the constraint equation (C).
	///	@param inStiffness Spring stiffness k.
	///	@param inDamping Spring damping coefficient c.
	/// @param outEffectiveMass On return, this contains the new effective mass K^-1
	JPH_INLINE void				CalculateSpringPropertiesWithStiffnessAndDamping(float inDeltaTime, float inInvEffectiveMass, float inBias, float inC, float inStiffness, float inDamping, float &outEffectiveMass)
	{
		JPH_ASSERT(inStiffness > 0.0f || inDamping > 0.0f);

		// Soft constraints as per: Soft Constraints: Reinventing The Spring - Erin Catto - GDC 2011

		// Note that the calculation of beta and gamma below are based on the solution of an implicit Euler integration scheme
		// This scheme is unconditionally stable but has built in damping, so even when you set the damping ratio to 0 there will still
		// be damping. See page 16 and 32.

		// Calculate softness (gamma in the slides)
		// See page 34 and note that the gamma needs to be divided by delta time since we're working with impulses rather than forces:
		// softness = 1 / (dt * (c + dt * k))
		// Note that the spring stiffness is k and the spring damping is c
		mSoftness = 1.0f / (inDeltaTime * (inDamping + inDeltaTime * inStiffness));

		// Calculate bias factor (baumgarte stabilization):
		// beta = dt * k / (c + dt * k) = dt * k^2 * softness
		// b = beta / dt * C = dt * k * softness * C
		mBias = inBias + inDeltaTime * inStiffness * mSoftness * inC;

		// Update the effective mass, see post by Erin Catto: http://www.bulletphysics.org/Bullet/phpBB3/viewtopic.php?f=4&t=1354
		//
		// Newton's Law:
		// M * (v2 - v1) = J^T * lambda
		//
		// Velocity constraint with softness and Baumgarte:
		// J * v2 + softness * lambda + b = 0
		//
		// where b = beta * C / dt
		//
		// We know everything except v2 and lambda.
		//
		// First solve Newton's law for v2 in terms of lambda:
		//
		// v2 = v1 + M^-1 * J^T * lambda
		//
		// Substitute this expression into the velocity constraint:
		//
		// J * (v1 + M^-1 * J^T * lambda) + softness * lambda + b = 0
		//
		// Now collect coefficients of lambda:
		//
		// (J * M^-1 * J^T + softness) * lambda = - J * v1 - b
		//
		// Now we define:
		//
		// K = J * M^-1 * J^T + softness
		//
		// So our new effective mass is K^-1
		outEffectiveMass = 1.0f / (inInvEffectiveMass + mSoftness);
	}

	/// Calculate spring properties based on frequency and damping ratio
	///
	/// @param inDeltaTime Time step
	/// @param inInvEffectiveMass Inverse effective mass K
	/// @param inBias Bias term (b) for the constraint impulse: lambda = J v + b
	///	@param inC Value of the constraint equation (C).
	///	@param inFrequency Oscillation frequency (Hz).
	///	@param inDamping Damping factor (0 = no damping, 1 = critical damping).
	/// @param outEffectiveMass On return, this contains the new effective mass K^-1
	JPH_INLINE void				CalculateSpringPropertiesWithFrequencyAndDamping(float inDeltaTime, float inInvEffectiveMass, float inBias, float inC, float inFrequency, float inDamping, float &outEffectiveMass)
	{
		JPH_ASSERT(inFrequency > 0.0f);

		outEffectiveMass = 1.0f / inInvEffectiveMass;

		// Calculate angular frequency
		float omega = 2.0f * JPH_PI * inFrequency;

		// Calculate spring stiffness k and damping constant c (page 45)
		float k = outEffectiveMass * Square(omega);
		float c = 2.0f * outEffectiveMass * inDamping * omega;

		CalculateSpringPropertiesWithStiffnessAndDamping(inDeltaTime, inInvEffectiveMass, inBias, inC, k, c, outEffectiveMass);
	}

	/// Calculate spring properties with spring stiffness (k) and damping (c) in acceleration mode, this is based on the spring equation: F = m_eff * (-k * x - c * v) where m_eff is the effective mass of the constraint
	///
	/// @param inDeltaTime Time step
	/// @param inInvEffectiveMass Inverse effective mass K
	/// @param inBias Bias term (b) for the constraint impulse: lambda = J v + b
	///	@param inC Value of the constraint equation (C).
	///	@param inStiffness Spring stiffness k.
	///	@param inDamping Spring damping coefficient c.
	/// @param outEffectiveMass On return, this contains the new effective mass K^-1
	JPH_INLINE void				CalculateSpringPropertiesWithMassNormalizedStiffnessAndDamping(float inDeltaTime, float inInvEffectiveMass, float inBias, float inC, float inStiffness, float inDamping, float &outEffectiveMass)
	{
		CalculateSpringPropertiesWithStiffnessAndDamping(inDeltaTime, inInvEffectiveMass, inBias, inC, inStiffness / inInvEffectiveMass, inDamping / inInvEffectiveMass, outEffectiveMass);
	}

	/// Calculate spring properties based on SpringSettings object.
	/// Assumes the spring has either stiffness or damping.
	JPH_INLINE void				CalculateSpringPropertiesWithSettings(float inDeltaTime, float inInvEffectiveMass, float inBias, float inC, const SpringSettings &inSpringSettings, float &outEffectiveMass)
	{
		switch (inSpringSettings.mMode)
		{
		case ESpringMode::FrequencyAndDamping:
			CalculateSpringPropertiesWithFrequencyAndDamping(inDeltaTime, inInvEffectiveMass, inBias, inC, inSpringSettings.mFrequency, inSpringSettings.mDamping, outEffectiveMass);
			break;
		case ESpringMode::StiffnessAndDamping:
			CalculateSpringPropertiesWithStiffnessAndDamping(inDeltaTime, inInvEffectiveMass, inBias, inC, inSpringSettings.mStiffness, inSpringSettings.mDamping, outEffectiveMass);
			break;
		case ESpringMode::MassNormalizedStiffnessAndDamping:
			CalculateSpringPropertiesWithMassNormalizedStiffnessAndDamping(inDeltaTime, inInvEffectiveMass, inBias, inC, inSpringSettings.mStiffness, inSpringSettings.mDamping, outEffectiveMass);
			break;
		}
	}

	/// Returns if this spring is active
	JPH_INLINE bool				IsActive() const
	{
		return mSoftness != 0.0f;
	}

	/// Get total bias b, including supplied bias and bias for spring: lambda = J v + b
	JPH_INLINE float			GetBias(float inTotalLambda) const
	{
		// Remainder of post by Erin Catto: http://www.bulletphysics.org/Bullet/phpBB3/viewtopic.php?f=4&t=1354
		//
		// Each iteration we are not computing the whole impulse, we are computing an increment to the impulse and we are updating the velocity.
		// Also, as we solve each constraint we get a perfect v2, but then some other constraint will come along and mess it up.
		// So we want to patch up the constraint while acknowledging the accumulated impulse and the damaged velocity.
		// To help with that we use P for the accumulated impulse and lambda as the update. Mathematically we have:
		//
		// M * (v2new - v2damaged) = J^T * lambda
		// J * v2new + softness * (total_lambda + lambda) + b = 0
		//
		// If we solve this we get:
		//
		// v2new = v2damaged + M^-1 * J^T * lambda
		// J * (v2damaged + M^-1 * J^T * lambda) + softness * total_lambda + softness * lambda + b = 0
		//
		// (J * M^-1 * J^T + softness) * lambda = -(J * v2damaged + softness * total_lambda + b)
		//
		// So our lagrange multiplier becomes:
		//
		// lambda = -K^-1 (J v + softness * total_lambda + b)
		//
		// So we return the bias: softness * total_lambda + b
		return mSoftness * inTotalLambda + mBias;
	}

private:
	float						mBias  = 0.0f;
	float						mSoftness  = 0.0f;
};

JPH_NAMESPACE_END
