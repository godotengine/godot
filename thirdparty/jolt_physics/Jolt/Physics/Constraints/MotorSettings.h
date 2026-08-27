// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Core/Reference.h>
#include <Jolt/ObjectStream/SerializableObject.h>
#include <Jolt/Physics/Constraints/SpringSettings.h>

JPH_NAMESPACE_BEGIN

class StreamIn;
class StreamOut;

enum class EMotorState
{
	Off,																///< Motor is off
	Velocity,															///< Motor will drive to target velocity limited only by max force/torque that the motor can apply
	Position,															///< Motor will drive to target position using: force = stiffness * (target_position - current_position) - damping * current_velocity
	PositionAndVelocity,												///< Motor will drive both to target position and velocity using: force = stiffness * (target_position - current_position) + damping * (target_velocity - current_velocity)
};

// Ability to check if position/velocity is enabled for a motor state
constexpr bool				IsVelocityMotor(EMotorState inMotorState)	{ return (int(inMotorState) & int(EMotorState::Velocity)) != 0; }
constexpr bool				IsPositionMotor(EMotorState inMotorState)	{ return (int(inMotorState) & int(EMotorState::Position)) != 0; }

static_assert(!IsVelocityMotor(EMotorState::Off));
static_assert(IsVelocityMotor(EMotorState::Velocity));
static_assert(!IsVelocityMotor(EMotorState::Position));
static_assert(IsPositionMotor(EMotorState::PositionAndVelocity));
static_assert(!IsPositionMotor(EMotorState::Off));
static_assert(!IsPositionMotor(EMotorState::Velocity));
static_assert(IsPositionMotor(EMotorState::Position));
static_assert(IsPositionMotor(EMotorState::PositionAndVelocity));

/// Class that contains the settings for a constraint motor.
/// See the main page of the API documentation for more information on how to configure a motor.
class JPH_EXPORT MotorSettings
{
	JPH_DECLARE_SERIALIZABLE_NON_VIRTUAL(JPH_EXPORT, MotorSettings)

public:
	/// Constructor
							MotorSettings() = default;
							MotorSettings(const MotorSettings &) = default;
	MotorSettings &			operator = (const MotorSettings &) = default;
							MotorSettings(ESpringMode inMode, float inFrequency, float inDamping) : mSpringSettings(inMode, inFrequency, inDamping) { JPH_ASSERT(IsValid()); }
							MotorSettings(float inFrequency, float inDamping) : mSpringSettings(ESpringMode::FrequencyAndDamping, inFrequency, inDamping) { JPH_ASSERT(IsValid()); }
							MotorSettings(ESpringMode inMode, float inFrequency, float inDamping, float inForceLimit, float inTorqueLimit) : mSpringSettings(inMode, inFrequency, inDamping), mMinForceLimit(-inForceLimit), mMaxForceLimit(inForceLimit), mMinTorqueLimit(-inTorqueLimit), mMaxTorqueLimit(inTorqueLimit) { JPH_ASSERT(IsValid()); }
							MotorSettings(float inFrequency, float inDamping, float inForceLimit, float inTorqueLimit) : mSpringSettings(ESpringMode::FrequencyAndDamping, inFrequency, inDamping), mMinForceLimit(-inForceLimit), mMaxForceLimit(inForceLimit), mMinTorqueLimit(-inTorqueLimit), mMaxTorqueLimit(inTorqueLimit) { JPH_ASSERT(IsValid()); }

	/// Set asymmetric force limits
	void					SetForceLimits(float inMin, float inMax)	{ JPH_ASSERT(inMin <= inMax); mMinForceLimit = inMin; mMaxForceLimit = inMax; }

	/// Set asymmetric torque limits
	void					SetTorqueLimits(float inMin, float inMax)	{ JPH_ASSERT(inMin <= inMax); mMinTorqueLimit = inMin; mMaxTorqueLimit = inMax; }

	/// Set symmetric force limits
	void					SetForceLimit(float inLimit)				{ mMinForceLimit = -inLimit; mMaxForceLimit = inLimit; }

	/// Set symmetric torque limits
	void					SetTorqueLimit(float inLimit)				{ mMinTorqueLimit = -inLimit; mMaxTorqueLimit = inLimit; }

	/// Check if settings are valid
	bool					IsValid() const								{ return mSpringSettings.mFrequency >= 0.0f && mSpringSettings.mDamping >= 0.0f && mMinForceLimit <= mMaxForceLimit && mMinTorqueLimit <= mMaxTorqueLimit; }

	/// Saves the contents of the motor settings in binary form to inStream.
	void					SaveBinaryState(StreamOut &inStream) const;

	/// Restores contents from the binary stream inStream.
	void					RestoreBinaryState(StreamIn &inStream);

	// Settings
	SpringSettings			mSpringSettings { ESpringMode::FrequencyAndDamping, 2.0f, 1.0f }; ///< Settings for the spring that is used to drive to the position target (not used when motor is a velocity motor).
	float					mMinForceLimit = -FLT_MAX;					///< Minimum force to apply in case of a linear constraint (N). Usually this is -mMaxForceLimit unless you want a motor that can e.g. push but not pull. Not used when motor is an angular motor.
	float					mMaxForceLimit = FLT_MAX;					///< Maximum force to apply in case of a linear constraint (N). Not used when motor is an angular motor.
	float					mMinTorqueLimit = -FLT_MAX;					///< Minimum torque to apply in case of a angular constraint (N m). Usually this is -mMaxTorqueLimit unless you want a motor that can e.g. push but not pull. Not used when motor is a linear motor.
	float					mMaxTorqueLimit = FLT_MAX;					///< Maximum torque to apply in case of a angular constraint (N m). Not used when motor is a linear motor.
};

JPH_NAMESPACE_END
