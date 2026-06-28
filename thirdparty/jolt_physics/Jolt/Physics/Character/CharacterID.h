// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2025 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Core/HashCombine.h>

JPH_NAMESPACE_BEGIN

/// ID of a character. Used primarily to identify deleted characters and to sort deterministically.
class JPH_EXPORT CharacterID
{
public:
	JPH_OVERRIDE_NEW_DELETE

	static constexpr uint32	cInvalidCharacterID = 0xffffffff;	///< The value for an invalid character ID

	/// Construct invalid character ID
							CharacterID() :
		mID(cInvalidCharacterID)
	{
	}

	/// Construct with specific value, make sure you don't use the same value twice!
	explicit				CharacterID(uint32 inID) :
		mID(inID)
	{
	}

	/// Get the numeric value of the ID
	inline uint32			GetValue() const
	{
		return mID;
	}

	/// Check if the ID is valid
	inline bool				IsInvalid() const
	{
		return mID == cInvalidCharacterID;
	}

	/// Equals check
	inline bool				operator == (const CharacterID &inRHS) const
	{
		return mID == inRHS.mID;
	}

	/// Not equals check
	inline bool				operator != (const CharacterID &inRHS) const
	{
		return mID != inRHS.mID;
	}

	/// Smaller than operator, can be used for sorting characters
	inline bool				operator < (const CharacterID &inRHS) const
	{
		return mID < inRHS.mID;
	}

	/// Greater than operator, can be used for sorting characters
	inline bool				operator > (const CharacterID &inRHS) const
	{
		return mID > inRHS.mID;
	}

	/// Get the hash for this character ID
	inline uint64			GetHash() const
	{
		return Hash<uint32>{} (mID);
	}

	/// Generate the next available character ID
	static CharacterID		sNextCharacterID()
	{
		for (;;)
		{
			uint32 next = sNextID.fetch_add(1, std::memory_order_relaxed);
			if (next != cInvalidCharacterID)
				return CharacterID(next);
		}
	}

	/// Set the next available character ID, can be used after destroying all character to prepare for a second deterministic run
	static void				sSetNextCharacterID(uint32 inNextValue = 1)
	{
		sNextID.store(inNextValue, std::memory_order_relaxed);
	}

private:
	/// Next character ID to be assigned
	inline static atomic<uint32> sNextID = 1;

	/// ID value
	uint32					mID;
};

JPH_NAMESPACE_END
