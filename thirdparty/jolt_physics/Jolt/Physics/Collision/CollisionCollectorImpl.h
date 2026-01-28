// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Collision/CollisionCollector.h>
#include <Jolt/Core/QuickSort.h>

JPH_NAMESPACE_BEGIN

/// Simple implementation that collects all hits and optionally sorts them on distance
template <class CollectorType>
class AllHitCollisionCollector : public CollectorType
{
public:
	/// Redeclare ResultType
	using ResultType = typename CollectorType::ResultType;

	// See: CollectorType::Reset
	virtual void		Reset() override
	{
		CollectorType::Reset();

		mHits.clear();
	}

	// See: CollectorType::AddHit
	virtual void		AddHit(const ResultType &inResult) override
	{
		mHits.push_back(inResult);
	}

	/// Order hits on closest first
	void				Sort()
	{
		QuickSort(mHits.begin(), mHits.end(), [](const ResultType &inLHS, const ResultType &inRHS) { return inLHS.GetEarlyOutFraction() < inRHS.GetEarlyOutFraction(); });
	}

	/// Check if any hits were collected
	inline bool			HadHit() const
	{
		return !mHits.empty();
	}

	Array<ResultType>	mHits;
};

/// Simple implementation that collects the closest / deepest hit
template <class CollectorType>
class ClosestHitCollisionCollector : public CollectorType
{
public:
	/// Redeclare ResultType
	using ResultType = typename CollectorType::ResultType;

	// See: CollectorType::Reset
	virtual void		Reset() override
	{
		CollectorType::Reset();

		mHadHit = false;
	}

	// See: CollectorType::AddHit
	virtual void		AddHit(const ResultType &inResult) override
	{
		float early_out = inResult.GetEarlyOutFraction();
		if (!mHadHit || early_out < mHit.GetEarlyOutFraction())
		{
			// Update early out fraction
			CollectorType::UpdateEarlyOutFraction(early_out);

			// Store hit
			mHit = inResult;
			mHadHit = true;
		}
	}

	/// Check if this collector has had a hit
	inline bool			HadHit() const
	{
		return mHadHit;
	}

	ResultType			mHit;

private:
	bool				mHadHit = false;
};

/// Implementation that collects the closest / deepest hit for each body and optionally sorts them on distance
template <class CollectorType>
class ClosestHitPerBodyCollisionCollector : public CollectorType
{
public:
	/// Redeclare ResultType
	using ResultType = typename CollectorType::ResultType;

	// See: CollectorType::Reset
	virtual void		Reset() override
	{
		CollectorType::Reset();

		mHits.clear();
		mHadHit = false;
	}

	// See: CollectorType::OnBody
	virtual void		OnBody(const Body &inBody) override
	{
		// Store the early out fraction so we can restore it after we've collected all hits for this body
		mPreviousEarlyOutFraction = CollectorType::GetEarlyOutFraction();
	}

	// See: CollectorType::AddHit
	virtual void		AddHit(const ResultType &inResult) override
	{
		float early_out = inResult.GetEarlyOutFraction();
		if (!mHadHit || early_out < CollectorType::GetEarlyOutFraction())
		{
			// Update early out fraction to avoid spending work on collecting further hits for this body
			CollectorType::UpdateEarlyOutFraction(early_out);

			if (!mHadHit)
			{
				// First time we have a hit we append it to the array
				mHits.push_back(inResult);
				mHadHit = true;
			}
			else
			{
				// Closer hits will override the previous one
				mHits.back() = inResult;
			}
		}
	}

	// See: CollectorType::OnBodyEnd
	virtual void		OnBodyEnd() override
	{
		if (mHadHit)
		{
			// Reset the early out fraction to the configured value so that we will continue
			// to collect hits at any distance for other bodies
			JPH_ASSERT(mPreviousEarlyOutFraction != -FLT_MAX); // Check that we got a call to OnBody
			CollectorType::ResetEarlyOutFraction(mPreviousEarlyOutFraction);
			mHadHit = false;
		}

		// For asserting purposes we reset the stored early out fraction so we can detect that OnBody was called
		JPH_IF_ENABLE_ASSERTS(mPreviousEarlyOutFraction = -FLT_MAX;)
	}

	/// Order hits on closest first
	void				Sort()
	{
		QuickSort(mHits.begin(), mHits.end(), [](const ResultType &inLHS, const ResultType &inRHS) { return inLHS.GetEarlyOutFraction() < inRHS.GetEarlyOutFraction(); });
	}

	/// Check if any hits were collected
	inline bool			HadHit() const
	{
		return !mHits.empty();
	}

	Array<ResultType>	mHits;

private:
	// Store early out fraction that was initially configured for the collector
	float				mPreviousEarlyOutFraction = -FLT_MAX;

	// Flag to indicate if we have a hit for the current body
	bool				mHadHit = false;
};

/// Simple implementation that collects any hit
template <class CollectorType>
class AnyHitCollisionCollector : public CollectorType
{
public:
	/// Redeclare ResultType
	using ResultType = typename CollectorType::ResultType;

	// See: CollectorType::Reset
	virtual void		Reset() override
	{
		CollectorType::Reset();

		mHadHit = false;
	}

	// See: CollectorType::AddHit
	virtual void		AddHit(const ResultType &inResult) override
	{
		// Test that the collector is not collecting more hits after forcing an early out
		JPH_ASSERT(!mHadHit);

		// Abort any further testing
		CollectorType::ForceEarlyOut();

		// Store hit
		mHit = inResult;
		mHadHit = true;
	}

	/// Check if this collector has had a hit
	inline bool			HadHit() const
	{
		return mHadHit;
	}

	ResultType			mHit;

private:
	bool				mHadHit = false;
};

JPH_NAMESPACE_END
