// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2024 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Collision/SimShapeFilter.h>
#include <Jolt/Physics/Body/Body.h>

JPH_NAMESPACE_BEGIN

/// Helper class to forward ShapeFilter calls to a SimShapeFilter
/// INTERNAL CLASS DO NOT USE!
class SimShapeFilterWrapper : private ShapeFilter
{
public:
	/// Constructor
							SimShapeFilterWrapper(const SimShapeFilter *inFilter, const Body *inBody1) :
		mFilter(inFilter),
		mBody1(inBody1)
	{
		// Fall back to an empty filter if no simulation shape filter is set, this reduces the virtual call to 'return true'
		mFinalFilter = inFilter != nullptr? this : &mDefault;
	}

	/// Forward to the simulation shape filter
	virtual bool			ShouldCollide(const Shape *inShape1, const SubShapeID &inSubShapeIDOfShape1, const Shape *inShape2, const SubShapeID &inSubShapeIDOfShape2) const override
	{
		return mFilter->ShouldCollide(*mBody1, inShape1, inSubShapeIDOfShape1, *mBody2, inShape2, inSubShapeIDOfShape2);
	}

	/// Forward to the simulation shape filter
	virtual bool			ShouldCollide(const Shape *inShape2, const SubShapeID &inSubShapeIDOfShape2) const override
	{
		return mFilter->ShouldCollide(*mBody1, mBody1->GetShape(), SubShapeID(), *mBody2, inShape2, inSubShapeIDOfShape2);
	}

	/// Set the body we're colliding against
	void					SetBody2(const Body *inBody2)
	{
		mBody2 = inBody2;
	}

	/// Returns the actual filter to use for collision detection
	const ShapeFilter &		GetFilter() const
	{
		return *mFinalFilter;
	}

private:
	const ShapeFilter *		mFinalFilter;
	const SimShapeFilter *	mFilter;
	const Body *			mBody1;
	const Body *			mBody2 = nullptr;
	const ShapeFilter		mDefault;
};

JPH_NAMESPACE_END
