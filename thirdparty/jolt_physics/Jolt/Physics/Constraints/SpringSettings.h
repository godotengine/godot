// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2023 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/ObjectStream/SerializableObject.h>

JPH_NAMESPACE_BEGIN

class StreamIn;
class StreamOut;

/// Enum used by constraints to specify how the spring is defined
enum class ESpringMode : uint8
{
	FrequencyAndDamping,					///< Frequency and damping are specified.
	StiffnessAndDamping,					///< Stiffness and damping are specified.
	MassNormalizedStiffnessAndDamping,		///< Stiffness and damping divided by mass / inertia are specified (also known as acceleration mode). This makes it easier to tune the spring and makes it mass independent.
};

/// Settings for a linear or angular spring
class JPH_EXPORT SpringSettings
{
	JPH_DECLARE_SERIALIZABLE_NON_VIRTUAL(JPH_EXPORT, SpringSettings)

public:
	/// Constructor
								SpringSettings() = default;
								SpringSettings(const SpringSettings &) = default;
	SpringSettings &			operator = (const SpringSettings &) = default;
								SpringSettings(ESpringMode inMode, float inFrequencyOrStiffness, float inDamping) : mMode(inMode), mFrequency(inFrequencyOrStiffness), mDamping(inDamping) { }

	/// Saves the contents of the spring settings in binary form to inStream.
	void						SaveBinaryState(StreamOut &inStream) const;

	/// Restores contents from the binary stream inStream.
	void						RestoreBinaryState(StreamIn &inStream);

	/// Check if the spring has a valid frequency / stiffness, if not the spring will be hard
	inline bool					HasStiffness() const							{ return mFrequency > 0.0f; }

	/// Check if this spring has stiffness or damping (making it active), if not the constraint will be hard
	inline bool					HasStiffnessOrDamping() const					{ return mFrequency > 0.0f || (mMode != ESpringMode::FrequencyAndDamping && mDamping > 0.0f); }

	/// Selects the way in which the spring is defined. See the descriptions of the mFrequency, mStiffness and mDamping properties.
	ESpringMode					mMode = ESpringMode::FrequencyAndDamping;

	union
	{
		/// Valid when mMode = ESpringMode::FrequencyAndDamping.
		/// If > 0 the constraint will be soft and this specifies the oscillation frequency in Hz.
		/// If <= 0, mDamping is ignored and the constraint will have hard limits (as hard as the time step / the number of velocity / position solver steps allows).
		float					mFrequency = 0.0f;

		/// When mMode = ESpringMode::StiffnessAndDamping:
		/// Specifies the stiffness (k) in the spring equation F = -k * x - c * v for a linear or T = -k * theta - c * w for an angular spring.
		/// Units are N / m for a linear spring and N m / rad for an angular spring.
		///
		/// Note that stiffness values are large numbers. To calculate a ballpark value for the needed stiffness you can use:
		/// force = stiffness * delta_spring_length = mass * gravity <=> stiffness = mass * gravity / delta_spring_length.
		/// So if your object weighs 1500 kg and the spring compresses by 2 meters, you need a stiffness in the order of 1500 * 9.81 / 2 ~ 7500 N/m.
		///
		/// When mMode = ESpringMode::MassNormalizedStiffnessAndDamping:
		/// Specifies the stiffness (k) in the spring equation F = m_eff * (-k * x - c * v) for a linear or T = i_eff * (-k * theta - c * w) for an angular spring.
		/// m_eff / i_eff is the effective mass / inertia of the constraint.
		/// Units are 1 / s^2 for a linear spring and 1 / rad s^2 for an angular spring.
		///
		/// Since the stiffness is multiplied by the effective mass / inertia of the constraint, you can use much smaller stiffness values and they will be mass independent.
		float					mStiffness;
	};

	/// When mMode = ESpringMode::FrequencyAndDamping this is the damping ratio (0 = no damping, 1 = critical damping).
	///
	/// When mMode = ESpringMode::StiffnessAndDamping this is the damping (c) in the spring equation F = -k * x - c * v for a linear or T = -k * theta - c * w for an angular spring.
	/// Units are N s / m for a linear spring and N s m / rad for an angular spring.
	///
	/// When mMode = ESpringMode::MassNormalizedStiffnessAndDamping this is the damping (c) in the spring equation F = m_eff * (-k * x - c * v) for a linear or T = i_eff * (-k * theta - c * w) for an angular spring.
	/// m_eff / i_eff is the effective mass / inertia of the constraint.
	/// Units are 1 / s for a linear spring and 1 / rad s for an angular spring.
	///
	/// Note that if you set this to 0, you will not get an infinite oscillation. Because we integrate physics using an explicit Euler scheme, there is always energy loss.
	/// This is done to keep the simulation from exploding, because with a damping of 0 and even the slightest rounding error, the oscillation could become bigger and bigger until the simulation explodes.
	float						mDamping = 0.0f;
};

JPH_NAMESPACE_END
