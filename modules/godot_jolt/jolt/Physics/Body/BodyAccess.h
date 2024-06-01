// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#ifdef JPH_ENABLE_ASSERTS

JPH_NAMESPACE_BEGIN

class BodyAccess
{
public:
	/// Access rules, used to detect race conditions during simulation
	enum class EAccess : uint8
	{
		None		= 0,
		Read		= 1,
		ReadWrite	= 3,
	};

	/// Grant a scope specific access rights on the current thread
	class Grant
	{
	public:
		inline							Grant(EAccess inVelocity, EAccess inPosition)
		{
			JPH_ASSERT(sVelocityAccess == EAccess::ReadWrite);
			JPH_ASSERT(sPositionAccess == EAccess::ReadWrite);

			sVelocityAccess = inVelocity;
			sPositionAccess = inPosition;
		}

		inline							~Grant()
		{
			sVelocityAccess = EAccess::ReadWrite;
			sPositionAccess = EAccess::ReadWrite;
		}
	};

	/// Check if we have permission
	static bool							sCheckRights(EAccess inRights, EAccess inDesiredRights)
	{
		return (uint8(inRights) & uint8(inDesiredRights)) == uint8(inDesiredRights);
	}

	// Various permissions that can be granted
	static thread_local EAccess			sVelocityAccess;
	static thread_local EAccess			sPositionAccess;
};

JPH_NAMESPACE_END

#endif // JPH_ENABLE_ASSERTS
