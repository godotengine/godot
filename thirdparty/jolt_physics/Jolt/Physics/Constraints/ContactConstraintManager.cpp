// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Constraints/ContactConstraintManager.h>
#include <Jolt/Physics/Constraints/CalculateSolverSteps.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/PhysicsUpdateContext.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/IslandBuilder.h>
#include <Jolt/Physics/DeterminismLog.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/QuickSort.h>
#include <Jolt/Core/Prefetch.h>
#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

using namespace literals;

#ifdef JPH_DEBUG_RENDERER
bool ContactConstraintManager::sDrawContactPoint = false;
bool ContactConstraintManager::sDrawSupportingFaces = false;
bool ContactConstraintManager::sDrawContactPointReduction = false;
bool ContactConstraintManager::sDrawContactManifolds = false;
#endif // JPH_DEBUG_RENDERER

//#define JPH_MANIFOLD_CACHE_DEBUG

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::WorldContactPoint
////////////////////////////////////////////////////////////////////////////////////////////////////////

template <EMotionType Type1, EMotionType Type2>
JPH_INLINE void ContactConstraintManager::WorldContactPoint<Type1, Type2>::CalculateNonPenetrationConstraintProperties(float inDeltaTime, Vec3Arg inGravity, const Body &inBody1, const Body &inBody2, float inInvM1, float inInvM2, Mat44Arg inInvI1, Mat44Arg inInvI2, RVec3Arg inWorldSpacePosition1, RVec3Arg inWorldSpacePosition2, Vec3Arg inWorldSpaceNormal, const ContactSettings &inSettings, float inMinVelocityForRestitution)
{
	JPH_DET_LOG("CalculateNonPenetrationConstraintProperties: p1: " << inWorldSpacePosition1 << " p2: " << inWorldSpacePosition2
		<< " normal: " << inWorldSpaceNormal << " restitution: " << inSettings.mCombinedRestitution << " minv: " << inMinVelocityForRestitution);

	// Calculate collision points relative to body
	RVec3 p = 0.5_r * (inWorldSpacePosition1 + inWorldSpacePosition2);
	Vec3 r1 = Vec3(p - inBody1.GetCenterOfMassPosition());
	Vec3 r2 = Vec3(p - inBody2.GetCenterOfMassPosition());

	const MotionProperties *mp1 = inBody1.GetMotionPropertiesUnchecked();
	const MotionProperties *mp2 = inBody2.GetMotionPropertiesUnchecked();

	// Calculate velocity of collision points
	Vec3 relative_velocity;
	if constexpr (Type1 != EMotionType::Static && Type2 != EMotionType::Static)
		relative_velocity = mp2->GetPointVelocityCOM(r2) - mp1->GetPointVelocityCOM(r1);
	else if constexpr (Type1 != EMotionType::Static)
		relative_velocity = -mp1->GetPointVelocityCOM(r1);
	else if constexpr (Type2 != EMotionType::Static)
		relative_velocity = mp2->GetPointVelocityCOM(r2);
	else
	{
		JPH_ASSERT(false, "Static vs static makes no sense");
		relative_velocity = Vec3::sZero();
	}
	float normal_velocity = relative_velocity.Dot(inWorldSpaceNormal);

	// How much the shapes are penetrating (> 0 if penetrating, < 0 if separated)
	float penetration = Vec3(inWorldSpacePosition1 - inWorldSpacePosition2).Dot(inWorldSpaceNormal);

	// If there is no penetration, this is a speculative contact and we will apply a bias to the contact constraint
	// so that the constraint becomes relative_velocity . contact normal > -penetration / delta_time
	// instead of relative_velocity . contact normal > 0
	// See: GDC 2013: "Physics for Game Programmers; Continuous Collision" - Erin Catto
	float speculative_contact_velocity_bias = max(0.0f, -penetration / inDeltaTime);

	// Determine if the velocity is big enough for restitution
	float normal_velocity_bias;
	if (inSettings.mCombinedRestitution > 0.0f && normal_velocity < -inMinVelocityForRestitution)
	{
		// We have a velocity that is big enough for restitution. This is where speculative contacts don't work
		// great as we have to decide now if we're going to apply the restitution or not. If the relative
		// velocity is big enough for a hit, we apply the restitution (in the end, due to other constraints,
		// the objects may actually not collide and we will have applied restitution incorrectly). Another
		// artifact that occurs because of this approximation is that the object will bounce from its current
		// position rather than from a position where it is touching the other object. This causes the object
		// to appear to move faster for 1 frame (the opposite of time stealing).
		if (normal_velocity < -speculative_contact_velocity_bias)
		{
			// The gravity / constant forces are applied in the beginning of the time step.
			// If we get here, there was a collision at the beginning of the time step, so we've applied too much force.
			// This means that our calculated restitution can be too high resulting in an increase in energy.
			// So, when we apply restitution, we cancel the added velocity due to these forces.
			Vec3 relative_acceleration;

			// Calculate effect of gravity
			if constexpr (Type1 != EMotionType::Static && Type2 != EMotionType::Static)
				relative_acceleration = inGravity * (mp2->GetGravityFactor() - mp1->GetGravityFactor());
			else if constexpr (Type1 != EMotionType::Static)
				relative_acceleration = -inGravity * mp1->GetGravityFactor();
			else if constexpr (Type2 != EMotionType::Static)
				relative_acceleration = inGravity * mp2->GetGravityFactor();
			else
			{
				JPH_ASSERT(false, "Static vs static makes no sense");
				relative_acceleration = Vec3::sZero();
			}

			// Calculate effect of accumulated forces
			if constexpr (Type1 == EMotionType::Dynamic)
				relative_acceleration -= mp1->GetAccumulatedForce() * mp1->GetInverseMass();
			if constexpr (Type2 == EMotionType::Dynamic)
				relative_acceleration += mp2->GetAccumulatedForce() * mp2->GetInverseMass();

			// We only compensate forces towards the contact normal.
			float force_delta_velocity = min(0.0f, relative_acceleration.Dot(inWorldSpaceNormal) * inDeltaTime);

			normal_velocity_bias = inSettings.mCombinedRestitution * (normal_velocity - force_delta_velocity);
		}
		else
		{
			// In this case we have predicted that we don't hit the other object, but if we do (due to other constraints changing velocities)
			// the speculative contact will prevent penetration but will not apply restitution leading to another artifact.
			normal_velocity_bias = speculative_contact_velocity_bias;
		}
	}
	else
	{
		// No restitution. We can safely apply our contact velocity bias.
		normal_velocity_bias = speculative_contact_velocity_bias;
	}

	mNonPenetrationConstraint.CalculateConstraintProperties(inInvM1, inInvI1, r1, inInvM2, inInvI2, r2, inWorldSpaceNormal, normal_velocity_bias);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::ContactConstraint
////////////////////////////////////////////////////////////////////////////////////////////////////////

template <EMotionType Type1, EMotionType Type2>
void ContactConstraintManager::ContactConstraint<Type1, Type2>::CalculateFrictionConstraintProperties(const Body &inBody1, const Body &inBody2, float inInvM1, float inInvM2, Mat44Arg inInvI1, Mat44Arg inInvI2, const RVec3 *inWorldSpaceContacts, Vec3Arg inWorldSpaceNormal, Vec3Arg inWorldSpaceTangent1, Vec3Arg inWorldSpaceTangent2, const ContactSettings &inSettings)
{
	// Calculate friction part
	if (inSettings.mCombinedFriction > 0.0f)
	{
		// Calculate point where the friction applies by averaging the contact points
		RVec3 friction_point = RVec3::sZero();
		for (uint32 i = 0; i < mNumContactPoints; ++i)
			friction_point += inWorldSpaceContacts[i];
		friction_point /= Real(mNumContactPoints);

		JPH_DET_LOG("CalculateFrictionConstraintProperties: point: " << friction_point
			<< " friction: " << inSettings.mCombinedFriction
			<< " surface_vel: " << inSettings.mRelativeLinearSurfaceVelocity << " surface_ang: " << inSettings.mRelativeAngularSurfaceVelocity);

		// Calculate distance of contact points to friction center in the normal plane
		for (uint32 i = 0; i < mNumContactPoints; ++i)
		{
			Vec3 delta = Vec3(inWorldSpaceContacts[i] - friction_point);
			mContactPoints[i].mDistanceToFrictionCenter = (delta - delta.Dot(inWorldSpaceNormal) * inWorldSpaceNormal).Length();
		}

		// Calculate relative friction points
		Vec3 r1 = Vec3(friction_point - inBody1.GetCenterOfMassPosition());
		Vec3 r2 = Vec3(friction_point - inBody2.GetCenterOfMassPosition());

		// Get surface velocity relative to tangents
		Vec3 ws_surface_velocity = inSettings.mRelativeLinearSurfaceVelocity + inSettings.mRelativeAngularSurfaceVelocity.Cross(r1);
		float surface_velocity1 = inWorldSpaceTangent1.Dot(ws_surface_velocity);
		float surface_velocity2 = inWorldSpaceTangent2.Dot(ws_surface_velocity);

		// Implement friction as 2 ContactConstraintParts
		mFrictionConstraint1.CalculateConstraintProperties(inInvM1, inInvI1, r1, inInvM2, inInvI2, r2, inWorldSpaceTangent1, surface_velocity1);
		mFrictionConstraint2.CalculateConstraintProperties(inInvM1, inInvI1, r1, inInvM2, inInvI2, r2, inWorldSpaceTangent2, surface_velocity2);

		// Only apply angular friction if we have more than 1 contact point
		if (mNumContactPoints > 1)
			mAngularFrictionConstraint.CalculateConstraintProperties(inInvI1, inInvI2, inWorldSpaceNormal, inSettings.mRelativeAngularSurfaceVelocity.Dot(inWorldSpaceNormal));
		else
			mAngularFrictionConstraint.Deactivate();
	}
	else
	{
		// Turn off friction constraint
		mFrictionConstraint1.Deactivate();
		mFrictionConstraint2.Deactivate();
		mAngularFrictionConstraint.Deactivate();
	}
}

#ifdef JPH_DEBUG_RENDERER
template <EMotionType Type1, EMotionType Type2>
void ContactConstraintManager::ContactConstraint<Type1, Type2>::Draw(DebugRenderer *inRenderer, const ManifoldCache &inManifoldCache, ColorArg inManifoldColor) const
{
	if (mNumContactPoints == 0)
		return;

	const CachedManifold &cached_manifold = inManifoldCache.FromHandle(mCachedManifoldHandle)->GetValue();

	// Get body transforms
	RMat44 transform_body1 = mBody1->GetCenterOfMassTransform();
	RMat44 transform_body2 = mBody2->GetCenterOfMassTransform();

	RVec3 prev_point = transform_body1 * Vec3::sLoadFloat3Unsafe(cached_manifold.mContactPoints[mNumContactPoints - 1].mPosition1);
	for (uint32 i = 0; i < mNumContactPoints; ++i)
	{
		const WorldContactPoint<Type1, Type2> &wcp = mContactPoints[i];
		const CachedContactPoint &ccp = cached_manifold.mContactPoints[i];

		// Test if any lambda from the previous frame was transferred
		float radius = wcp.mNonPenetrationConstraint.GetTotalLambda() == 0.0f
					&& mFrictionConstraint1.GetTotalLambda() == 0.0f
					&& mFrictionConstraint2.GetTotalLambda() == 0.0f
					&& mAngularFrictionConstraint.GetTotalLambda() == 0.0f? 0.1f :  0.2f;

		RVec3 next_point = transform_body1 * Vec3::sLoadFloat3Unsafe(ccp.mPosition1);
		inRenderer->DrawMarker(next_point, Color::sCyan, radius);
		inRenderer->DrawMarker(transform_body2 * Vec3::sLoadFloat3Unsafe(ccp.mPosition2), Color::sPurple, radius);

		// Draw edge
		inRenderer->DrawArrow(prev_point, next_point, inManifoldColor, 0.05f);
		prev_point = next_point;
	}

	// Draw normal
	RVec3 wp = transform_body1 * Vec3::sLoadFloat3Unsafe(cached_manifold.mContactPoints[0].mPosition1);
	inRenderer->DrawArrow(wp, wp + GetWorldSpaceNormal(), Color::sRed, 0.05f);

	// Get tangents
	Vec3 t1, t2;
	GetTangents(t1, t2);

	// Draw tangents
	inRenderer->DrawLine(wp, wp + t1, Color::sGreen);
	inRenderer->DrawLine(wp, wp + t2, Color::sBlue);
}
#endif // JPH_DEBUG_RENDERER

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::CachedContactPoint
////////////////////////////////////////////////////////////////////////////////////////////////////////

void ContactConstraintManager::CachedContactPoint::SaveState(StateRecorder &inStream) const
{
	inStream.Write(mPosition1);
	inStream.Write(mPosition2);
	inStream.Write(mNonPenetrationLambda);
}

void ContactConstraintManager::CachedContactPoint::RestoreState(StateRecorder &inStream)
{
	inStream.Read(mPosition1);
	inStream.Read(mPosition2);
	inStream.Read(mNonPenetrationLambda);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::CachedManifold
////////////////////////////////////////////////////////////////////////////////////////////////////////

void ContactConstraintManager::CachedManifold::SaveState(StateRecorder &inStream) const
{
	inStream.Write(mContactNormal);
	inStream.Write(mFrictionLambda);
	inStream.Write(mAngularFrictionLambda);
}

void ContactConstraintManager::CachedManifold::RestoreState(StateRecorder &inStream)
{
	inStream.Read(mContactNormal);
	inStream.Read(mFrictionLambda);
	inStream.Read(mAngularFrictionLambda);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::CachedBodyPair
////////////////////////////////////////////////////////////////////////////////////////////////////////

void ContactConstraintManager::CachedBodyPair::SaveState(StateRecorder &inStream) const
{
	inStream.Write(mDeltaPosition);
	inStream.Write(mDeltaRotation);
}

void ContactConstraintManager::CachedBodyPair::RestoreState(StateRecorder &inStream)
{
	inStream.Read(mDeltaPosition);
	inStream.Read(mDeltaRotation);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager::ManifoldCache
////////////////////////////////////////////////////////////////////////////////////////////////////////

void ContactConstraintManager::ManifoldCache::Init(uint inMaxBodyPairs, uint inMaxContactConstraints, uint inCachedManifoldsSize)
{
	JPH_ASSERT(inMaxContactConstraints <= cMaxContactConstraintsLimit); // Should have been enforced by caller

	uint max_body_pairs = min(inMaxBodyPairs, cMaxBodyPairsLimit);
	JPH_ASSERT(max_body_pairs == inMaxBodyPairs, "Cannot support this many body pairs!");
	max_body_pairs = max(max_body_pairs, 4u); // Because our hash map requires at least 4 buckets, we need to have a minimum number of body pairs

	mAllocator.Init(uint(min(uint64(max_body_pairs) * sizeof(BPKeyValue) + inCachedManifoldsSize, uint64(~uint(0)))));

	mCachedManifolds.Init(GetNextPowerOf2(inMaxContactConstraints));
	mCachedBodyPairs.Init(GetNextPowerOf2(max_body_pairs));
}

void ContactConstraintManager::ManifoldCache::Clear()
{
	JPH_PROFILE_FUNCTION();

	mCachedManifolds.Clear();
	mCachedBodyPairs.Clear();
	mAllocator.Clear();

#ifdef JPH_ENABLE_ASSERTS
	// Mark as incomplete
	mIsFinalized = false;
#endif
}

void ContactConstraintManager::ManifoldCache::Prepare(uint inExpectedNumBodyPairs, uint inExpectedNumManifolds)
{
	// Minimum amount of buckets to use in the hash map
	constexpr uint32 cMinBuckets = 1024;

	// Use the next higher power of 2 of amount of objects in the cache from last frame to determine the amount of buckets in this frame
	mCachedManifolds.SetNumBuckets(min(max(cMinBuckets, GetNextPowerOf2(inExpectedNumManifolds)), mCachedManifolds.GetMaxBuckets()));
	mCachedBodyPairs.SetNumBuckets(min(max(cMinBuckets, GetNextPowerOf2(inExpectedNumBodyPairs)), mCachedBodyPairs.GetMaxBuckets()));
}

const ContactConstraintManager::MKeyValue *ContactConstraintManager::ManifoldCache::Find(const SubShapeIDPair &inKey, uint64 inKeyHash) const
{
	JPH_ASSERT(mIsFinalized);
	return mCachedManifolds.Find(inKey, inKeyHash);
}

ContactConstraintManager::MKeyValue *ContactConstraintManager::ManifoldCache::Create(ContactAllocator &ioContactAllocator, const SubShapeIDPair &inKey, uint64 inKeyHash, int inNumContactPoints)
{
	JPH_ASSERT(!mIsFinalized);
	MKeyValue *kv = mCachedManifolds.Create(ioContactAllocator, inKey, inKeyHash, CachedManifold::sGetRequiredExtraSize(inNumContactPoints));
	if (kv == nullptr)
	{
		ioContactAllocator.mErrors |= EPhysicsUpdateError::ManifoldCacheFull;
		return nullptr;
	}
	kv->GetValue().mNumContactPoints = uint16(inNumContactPoints);
	++ioContactAllocator.mNumManifolds;
	return kv;
}

ContactConstraintManager::MKVAndCreated ContactConstraintManager::ManifoldCache::FindOrCreate(ContactAllocator &ioContactAllocator, const SubShapeIDPair &inKey, uint64 inKeyHash, int inNumContactPoints)
{
	MKeyValue *kv = const_cast<MKeyValue *>(mCachedManifolds.Find(inKey, inKeyHash));
	if (kv != nullptr)
		return { kv, false };

	return { Create(ioContactAllocator, inKey, inKeyHash, inNumContactPoints), true };
}

uint32 ContactConstraintManager::ManifoldCache::ToHandle(const MKeyValue *inKeyValue) const
{
	JPH_ASSERT(!mIsFinalized);
	return mCachedManifolds.ToHandle(inKeyValue);
}

const ContactConstraintManager::MKeyValue *ContactConstraintManager::ManifoldCache::FromHandle(uint32 inHandle) const
{
	return mCachedManifolds.FromHandle(inHandle);
}

ContactConstraintManager::MKeyValue *ContactConstraintManager::ManifoldCache::FromHandle(uint32 inHandle)
{
	return mCachedManifolds.FromHandle(inHandle);
}

const ContactConstraintManager::BPKeyValue *ContactConstraintManager::ManifoldCache::Find(const BodyPair &inKey, uint64 inKeyHash) const
{
	JPH_ASSERT(mIsFinalized);
	return mCachedBodyPairs.Find(inKey, inKeyHash);
}

ContactConstraintManager::BPKeyValue *ContactConstraintManager::ManifoldCache::Create(ContactAllocator &ioContactAllocator, const BodyPair &inKey, uint64 inKeyHash)
{
	JPH_ASSERT(!mIsFinalized);
	BPKeyValue *kv = mCachedBodyPairs.Create(ioContactAllocator, inKey, inKeyHash, 0);
	if (kv == nullptr)
	{
		ioContactAllocator.mErrors |= EPhysicsUpdateError::BodyPairCacheFull;
		return nullptr;
	}
	++ioContactAllocator.mNumBodyPairs;
	return kv;
}

void ContactConstraintManager::ManifoldCache::GetAllBodyPairsSorted(Array<const BPKeyValue *> &outAll) const
{
	JPH_ASSERT(mIsFinalized);
	mCachedBodyPairs.GetAllKeyValues(outAll);

	// Sort by key
	QuickSort(outAll.begin(), outAll.end(), [](const BPKeyValue *inLHS, const BPKeyValue *inRHS) {
		return inLHS->GetKey() < inRHS->GetKey();
	});
}

void ContactConstraintManager::ManifoldCache::GetAllManifoldsSorted(const CachedBodyPair &inBodyPair, Array<const MKeyValue *> &outAll) const
{
	JPH_ASSERT(mIsFinalized);

	// Iterate through the attached manifolds
	for (uint32 handle = inBodyPair.mFirstCachedManifold; handle != ManifoldMap::cInvalidHandle; handle = FromHandle(handle)->GetValue().mNextWithSameBodyPair)
	{
		const MKeyValue *kv = mCachedManifolds.FromHandle(handle);
		outAll.push_back(kv);
	}

	// Sort by key
	QuickSort(outAll.begin(), outAll.end(), [](const MKeyValue *inLHS, const MKeyValue *inRHS) {
		return inLHS->GetKey() < inRHS->GetKey();
	});
}

void ContactConstraintManager::ManifoldCache::GetAllCCDManifoldsSorted(Array<const MKeyValue *> &outAll) const
{
	mCachedManifolds.GetAllKeyValues(outAll);

	for (int i = (int)outAll.size() - 1; i >= 0; --i)
		if ((outAll[i]->GetValue().mFlags & (uint16)CachedManifold::EFlags::CCDContact) == 0)
		{
			outAll[i] = outAll.back();
			outAll.pop_back();
		}

	// Sort by key
	QuickSort(outAll.begin(), outAll.end(), [](const MKeyValue *inLHS, const MKeyValue *inRHS) {
		return inLHS->GetKey() < inRHS->GetKey();
	});
}

void ContactConstraintManager::ManifoldCache::ContactPointRemovedCallbacks(ContactListener *inListener)
{
	JPH_PROFILE_FUNCTION();

	for (MKeyValue &kv : mCachedManifolds)
		if ((kv.GetValue().mFlags & uint16(CachedManifold::EFlags::ContactPersisted)) == 0)
			inListener->OnContactRemoved(kv.GetKey());
}

#ifdef JPH_ENABLE_ASSERTS

void ContactConstraintManager::ManifoldCache::Finalize()
{
	mIsFinalized = true;

#ifdef JPH_MANIFOLD_CACHE_DEBUG
	Trace("ManifoldMap:");
	mCachedManifolds.TraceStats();
	Trace("BodyPairMap:");
	mCachedBodyPairs.TraceStats();
#endif // JPH_MANIFOLD_CACHE_DEBUG
}

#endif

void ContactConstraintManager::ManifoldCache::SaveState(StateRecorder &inStream, const StateRecorderFilter *inFilter) const
{
	JPH_ASSERT(mIsFinalized);

	// Get contents of cache
	Array<const BPKeyValue *> all_bp;
	GetAllBodyPairsSorted(all_bp);

	// Determine which ones to save
	Array<const BPKeyValue *> selected_bp;
	if (inFilter == nullptr)
		selected_bp = std::move(all_bp);
	else
	{
		selected_bp.reserve(all_bp.size());
		for (const BPKeyValue *bp_kv : all_bp)
			if (inFilter->ShouldSaveContact(bp_kv->GetKey().mBodyA, bp_kv->GetKey().mBodyB))
				selected_bp.push_back(bp_kv);
	}

	// Write body pairs
	uint32 num_body_pairs = uint32(selected_bp.size());
	inStream.Write(num_body_pairs);
	for (const BPKeyValue *bp_kv : selected_bp)
	{
		// Write body pair key
		inStream.Write(bp_kv->GetKey());

		// Write body pair
		const CachedBodyPair &bp = bp_kv->GetValue();
		bp.SaveState(inStream);

		// Get attached manifolds
		Array<const MKeyValue *> all_m;
		GetAllManifoldsSorted(bp, all_m);

		// Write num manifolds
		uint32 num_manifolds = uint32(all_m.size());
		inStream.Write(num_manifolds);

		// Write all manifolds
		for (const MKeyValue *m_kv : all_m)
		{
			// Write key
			inStream.Write(m_kv->GetKey());
			const CachedManifold &cm = m_kv->GetValue();
			JPH_ASSERT((cm.mFlags & (uint16)CachedManifold::EFlags::CCDContact) == 0);

			// Write amount of contacts
			inStream.Write(cm.mNumContactPoints);

			// Write manifold
			cm.SaveState(inStream);

			// Write contact points
			for (uint32 i = 0; i < cm.mNumContactPoints; ++i)
				cm.mContactPoints[i].SaveState(inStream);
		}
	}

	// Get CCD manifolds
	Array<const MKeyValue *> all_m;
	GetAllCCDManifoldsSorted(all_m);

	// Determine which ones to save
	Array<const MKeyValue *> selected_m;
	if (inFilter == nullptr)
		selected_m = std::move(all_m);
	else
	{
		selected_m.reserve(all_m.size());
		for (const MKeyValue *m_kv : all_m)
			if (inFilter->ShouldSaveContact(m_kv->GetKey().GetBody1ID(), m_kv->GetKey().GetBody2ID()))
				selected_m.push_back(m_kv);
	}

	// Write all CCD manifold keys
	uint32 num_manifolds = uint32(selected_m.size());
	inStream.Write(num_manifolds);
	for (const MKeyValue *m_kv : selected_m)
		inStream.Write(m_kv->GetKey());
}

bool ContactConstraintManager::ManifoldCache::RestoreState(const ManifoldCache &inReadCache, StateRecorder &inStream, const StateRecorderFilter *inFilter)
{
	JPH_ASSERT(!mIsFinalized);

	bool success = true;

	// Create a contact allocator for restoring the contact cache
	ContactAllocator contact_allocator(GetContactAllocator());

	// When validating, get all existing body pairs
	Array<const BPKeyValue *> all_bp;
	if (inStream.IsValidating())
		inReadCache.GetAllBodyPairsSorted(all_bp);

	// Read amount of body pairs
	uint32 num_body_pairs;
	if (inStream.IsValidating())
		num_body_pairs = uint32(all_bp.size());
	inStream.Read(num_body_pairs);

	// Read entire cache
	for (uint32 i = 0; i < num_body_pairs; ++i)
	{
		// Read key
		BodyPair body_pair_key;
		if (inStream.IsValidating() && i < all_bp.size())
			body_pair_key = all_bp[i]->GetKey();
		inStream.Read(body_pair_key);

		// Check if we want to restore this contact
		if (inFilter == nullptr || inFilter->ShouldRestoreContact(body_pair_key.mBodyA, body_pair_key.mBodyB))
		{
			// Create new entry for this body pair
			uint64 body_pair_hash = body_pair_key.GetHash();
			BPKeyValue *bp_kv = Create(contact_allocator, body_pair_key, body_pair_hash);
			if (bp_kv == nullptr)
			{
				// Out of cache space
				success = false;
				break;
			}
			CachedBodyPair &bp = bp_kv->GetValue();

			// Read body pair
			if (inStream.IsValidating() && i < all_bp.size())
				memcpy(&bp, &all_bp[i]->GetValue(), sizeof(CachedBodyPair));
			bp.RestoreState(inStream);

			// When validating, get all existing manifolds
			Array<const MKeyValue *> all_m;
			if (inStream.IsValidating() && i < all_bp.size())
				inReadCache.GetAllManifoldsSorted(all_bp[i]->GetValue(), all_m);

			// Read amount of manifolds
			uint32 num_manifolds = 0;
			if (inStream.IsValidating())
				num_manifolds = uint32(all_m.size());
			inStream.Read(num_manifolds);

			uint32 handle = ManifoldMap::cInvalidHandle;
			for (uint32 j = 0; j < num_manifolds; ++j)
			{
				// Read key
				SubShapeIDPair sub_shape_key;
				if (inStream.IsValidating() && j < all_m.size())
					sub_shape_key = all_m[j]->GetKey();
				inStream.Read(sub_shape_key);
				uint64 sub_shape_key_hash = sub_shape_key.GetHash();

				// Read amount of contact points
				uint16 num_contact_points = 0;
				if (inStream.IsValidating() && j < all_m.size())
					num_contact_points = all_m[j]->GetValue().mNumContactPoints;
				inStream.Read(num_contact_points);

				// Read manifold
				MKeyValue *m_kv = Create(contact_allocator, sub_shape_key, sub_shape_key_hash, num_contact_points);
				if (m_kv == nullptr)
				{
					// Out of cache space
					success = false;
					break;
				}
				CachedManifold &cm = m_kv->GetValue();
				if (inStream.IsValidating() && j < all_m.size())
				{
					memcpy(&cm, &all_m[j]->GetValue(), CachedManifold::sGetRequiredTotalSize(num_contact_points));
					cm.mNumContactPoints = uint16(num_contact_points); // Restore num contact points
				}
				cm.RestoreState(inStream);
				cm.mNextWithSameBodyPair = handle;
				handle = ToHandle(m_kv);

				// Read contact points
				for (uint32 k = 0; k < num_contact_points; ++k)
					cm.mContactPoints[k].RestoreState(inStream);
			}
			bp.mFirstCachedManifold = handle;
		}
		else
		{
			// Skip the contact
			CachedBodyPair bp;
			bp.RestoreState(inStream);
			uint32 num_manifolds = 0;
			inStream.Read(num_manifolds);
			for (uint32 j = 0; j < num_manifolds; ++j)
			{
				SubShapeIDPair sub_shape_key;
				inStream.Read(sub_shape_key);
				uint16 num_contact_points;
				inStream.Read(num_contact_points);
				CachedManifold cm;
				cm.RestoreState(inStream);
				for (uint32 k = 0; k < num_contact_points; ++k)
					cm.mContactPoints[0].RestoreState(inStream);
			}
		}
	}

	// When validating, get all existing CCD manifolds
	Array<const MKeyValue *> all_m;
	if (inStream.IsValidating())
		inReadCache.GetAllCCDManifoldsSorted(all_m);

	// Read amount of CCD manifolds
	uint32 num_manifolds;
	if (inStream.IsValidating())
		num_manifolds = uint32(all_m.size());
	inStream.Read(num_manifolds);

	for (uint32 j = 0; j < num_manifolds; ++j)
	{
		// Read key
		SubShapeIDPair sub_shape_key;
		if (inStream.IsValidating() && j < all_m.size())
			sub_shape_key = all_m[j]->GetKey();
		inStream.Read(sub_shape_key);

		// Check if we want to restore this contact
		if (inFilter == nullptr || inFilter->ShouldRestoreContact(sub_shape_key.GetBody1ID(), sub_shape_key.GetBody2ID()))
		{
			// Create CCD manifold
			uint64 sub_shape_key_hash = sub_shape_key.GetHash();
			MKeyValue *m_kv = Create(contact_allocator, sub_shape_key, sub_shape_key_hash, 0);
			if (m_kv == nullptr)
			{
				// Out of cache space
				success = false;
				break;
			}
			CachedManifold &cm = m_kv->GetValue();
			cm.mFlags |= (uint16)CachedManifold::EFlags::CCDContact;
		}
	}

#ifdef JPH_ENABLE_ASSERTS
	// We don't finalize until the last part is restored
	if (inStream.IsLastPart())
		mIsFinalized = true;
#endif

	return success;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// ContactConstraintManager
////////////////////////////////////////////////////////////////////////////////////////////////////////

ContactConstraintManager::ContactConstraintManager(const PhysicsSettings &inPhysicsSettings) :
	mPhysicsSettings(inPhysicsSettings)
{
#ifdef JPH_ENABLE_ASSERTS
	// For the first frame mark this empty buffer as finalized
	mCache[mCacheWriteIdx ^ 1].Finalize();
#endif
}

ContactConstraintManager::~ContactConstraintManager()
{
	JPH_ASSERT(mConstraints == nullptr);
	JPH_ASSERT(mConstraintIdxToOffset == nullptr);
}

void ContactConstraintManager::Init(uint inMaxBodyPairs, uint inMaxContactConstraints)
{
	// Limit the number of constraints so that the allocation size fits in an unsigned integer
	mMaxConstraints = min(inMaxContactConstraints, cMaxContactConstraintsLimit);
	JPH_ASSERT(mMaxConstraints == inMaxContactConstraints, "Cannot support this many contact constraints!");
	mMaxConstraints = max(mMaxConstraints, 4u); // Because our hash map requires at least 4 buckets, we need to have a minimum number of constraints

	// Calculate worst case cache usage
	constexpr uint cMaxManifoldSizePerConstraint = sizeof(MKeyValue) + CachedManifold::sGetRequiredExtraSize(MaxContactPoints);
	static_assert(cMaxManifoldSizePerConstraint < cMaxConstraintSize); // If not true, then the next line can overflow
	uint cached_manifolds_size = mMaxConstraints * cMaxManifoldSizePerConstraint;

	// Init the caches
	mCache[0].Init(inMaxBodyPairs, mMaxConstraints, cached_manifolds_size);
	mCache[1].Init(inMaxBodyPairs, mMaxConstraints, cached_manifolds_size);
}

void ContactConstraintManager::PrepareConstraintBuffer(PhysicsUpdateContext *inContext)
{
	// Store context
	mUpdateContext = inContext;

	// Store read / write cache
	mReadCache = &mCache[mCacheWriteIdx ^ 1];
	mWriteCache = &mCache[mCacheWriteIdx];

	// Allocate temporary constraint buffer
	JPH_ASSERT(mConstraints == nullptr);
	mConstraints = (uint8 *)inContext->mTempAllocator->Allocate(mMaxConstraints * cMaxConstraintSize);
	JPH_ASSERT(mConstraintIdxToOffset == nullptr);
	mConstraintIdxToOffset = (uint32 *)inContext->mTempAllocator->Allocate(mMaxConstraints * sizeof(uint32));
}

template <EMotionType Type1, EMotionType Type2>
JPH_INLINE ContactConstraintManager::ContactConstraint<Type1, Type2> *ContactConstraintManager::CreateConstraint(bool &ioActivateAndLinkBodies, Body &inBody1, Body &inBody2, uint64 inSortKey, uint32 inCachedManifoldHandle, Vec3Arg inWorldSpaceNormal, const ContactSettings &inSettings, uint32 inNumContactPoints)
{
	// Calculate the size of this constraint
	uint32 constraint_size = (uint32)AlignUp(sizeof(ContactConstraint<Type1, Type2>) + (inNumContactPoints - 1) * sizeof(WorldContactPoint<Type1, Type2>), alignof(ContactConstraint<Type1, Type2>));
	JPH_ASSERT(constraint_size <= cMaxConstraintSize);

	// Reserve space for constraint
	uint64 constraint_idx_and_constraint_offset = mNumConstraintsAndNextConstraintOffset.fetch_add((uint64(constraint_size) << 32) + 1, memory_order_relaxed);
	uint32 constraint_idx = uint32(constraint_idx_and_constraint_offset);
	if (constraint_idx >= mMaxConstraints)
		return nullptr;

	bool body1_dynamic = inBody1.IsDynamic();
	bool body2_dynamic = inBody2.IsDynamic();

	if (ioActivateAndLinkBodies)
	{
		// Do this only once
		ioActivateAndLinkBodies = false;

		// Wake up sleeping bodies
		BodyID body_ids[2];
		int num_bodies = 0;
		if (body1_dynamic && !inBody1.IsActive())
			body_ids[num_bodies++] = inBody1.GetID();
		if (body2_dynamic && !inBody2.IsActive())
			body_ids[num_bodies++] = inBody2.GetID();
		if (num_bodies > 0)
			mUpdateContext->mBodyManager->ActivateBodies(body_ids, num_bodies);

		// Link the two bodies only if both are dynamic. If one of them is static or kinematic they don't need to go into
		// the same simulation island as a constraint cannot affect the velocity of a kinematic body.
		if (body1_dynamic && body2_dynamic)
			mUpdateContext->mIslandBuilder->LinkBodies(inBody1.GetIndexInActiveBodiesInternal(), inBody2.GetIndexInActiveBodiesInternal());
	}

	// Link the contact to the first dynamic body
	if (body1_dynamic)
		mUpdateContext->mIslandBuilder->LinkContact(constraint_idx, inBody1.GetIndexInActiveBodiesInternal());
	else
	{
		JPH_ASSERT(body2_dynamic);
		mUpdateContext->mIslandBuilder->LinkContact(constraint_idx, inBody2.GetIndexInActiveBodiesInternal());
	}

	// Store offset for constraint
	uint32 constraint_offset = uint32(constraint_idx_and_constraint_offset >> 32);
	JPH_ASSERT(constraint_offset + constraint_size <= mMaxConstraints * cMaxConstraintSize);
	mConstraintIdxToOffset[constraint_idx] = constraint_offset;

	// Construct constraint
	ContactConstraint<Type1, Type2> *constraint = reinterpret_cast<ContactConstraint<Type1, Type2> *>(mConstraints + constraint_offset);
	JPH_ASSERT(IsAligned(constraint, alignof(ContactConstraint<Type1, Type2>)));
	new (constraint) ContactConstraint<Type1, Type2>();
	constraint->mBody1 = &inBody1;
	constraint->mBody2 = &inBody2;
	constraint->mSortKey = inSortKey;
	inWorldSpaceNormal.StoreFloat3(&constraint->mWorldSpaceNormal);
	constraint->mCombinedFriction = inSettings.mCombinedFriction;
	constraint->mInvInertiaScale1 = inSettings.mInvInertiaScale1;
	constraint->mInvInertiaScale2 = inSettings.mInvInertiaScale2;
	constraint->mCachedManifoldHandle = inCachedManifoldHandle;
	constraint->mNumContactPoints = inNumContactPoints;

#ifdef JPH_TRACK_SIMULATION_STATS
	// Track new contact constraints
	if constexpr (Type1 != EMotionType::Static)
		inBody1.GetMotionPropertiesUnchecked()->GetSimulationStats().mNumContactConstraints.fetch_add(1, memory_order_relaxed);
	if constexpr (Type2 != EMotionType::Static)
		inBody2.GetMotionPropertiesUnchecked()->GetSimulationStats().mNumContactConstraints.fetch_add(1, memory_order_relaxed);
#endif

	return constraint;
}

template <EMotionType Type1, EMotionType Type2>
void ContactConstraintManager::TemplatedGetContactsFromCache(ContactAllocator &ioContactAllocator, Body &inBody1, Body &inBody2, const CachedBodyPair &inCachedBodyPair, CachedBodyPair &outCachedBodyPair)
{
	// Get body transforms
	RMat44 transform_body1 = inBody1.GetCenterOfMassTransform();
	RMat44 transform_body2 = inBody2.GetCenterOfMassTransform();

	// Get time step and gravity
	float delta_time = mUpdateContext->mStepDeltaTime;
	Vec3 gravity = mUpdateContext->mPhysicsSystem->GetGravity();

	// Copy manifolds
	uint32 output_handle = ManifoldMap::cInvalidHandle;
	uint32 input_handle = inCachedBodyPair.mFirstCachedManifold;
	bool link_bodies = true;
	do
	{
		JPH_PROFILE("Add Constraint From Cached Manifold");

		// Find the existing manifold
		const MKeyValue *input_kv = mReadCache->FromHandle(input_handle);
		const SubShapeIDPair &input_key = input_kv->GetKey();
		const CachedManifold &input_cm = input_kv->GetValue();
		JPH_ASSERT(input_cm.mNumContactPoints > 0); // There should be contact points in this manifold!

		// Create room for manifold in write buffer and copy data
		uint64 input_hash = input_key.GetHash();
		MKeyValue *output_kv = mWriteCache->Create(ioContactAllocator, input_key, input_hash, input_cm.mNumContactPoints);
		if (output_kv == nullptr)
			break; // Out of cache space
		CachedManifold *output_cm = &output_kv->GetValue();
		memcpy(output_cm, &input_cm, CachedManifold::sGetRequiredTotalSize(input_cm.mNumContactPoints));

		// Link the object under the body pairs
		output_cm->mNextWithSameBodyPair = output_handle;
		output_handle = mWriteCache->ToHandle(output_kv);

		// Calculate default contact settings
		ContactSettings settings;
		settings.mCombinedFriction = mCombineFriction(inBody1, input_key.GetSubShapeID1(), inBody2, input_key.GetSubShapeID2());
		settings.mCombinedRestitution = mCombineRestitution(inBody1, input_key.GetSubShapeID1(), inBody2, input_key.GetSubShapeID2());
		settings.mIsSensor = inBody1.IsSensor() || inBody2.IsSensor();

		// Calculate world space contact normal
		Vec3 world_space_normal = transform_body2.Multiply3x3(Vec3::sLoadFloat3Unsafe(output_cm->mContactNormal)).Normalized();

		// Call contact listener to update settings
		if (mContactListener != nullptr)
		{
			// Convert constraint to manifold structure for callback
			ContactManifold manifold;
			manifold.mWorldSpaceNormal = world_space_normal;
			manifold.mSubShapeID1 = input_key.GetSubShapeID1();
			manifold.mSubShapeID2 = input_key.GetSubShapeID2();
			manifold.mBaseOffset = transform_body1.GetTranslation();
			manifold.mRelativeContactPointsOn1.resize(output_cm->mNumContactPoints);
			manifold.mRelativeContactPointsOn2.resize(output_cm->mNumContactPoints);
			Mat44 local_transform_body2 = transform_body2.PostTranslated(-manifold.mBaseOffset).ToMat44();
			float penetration_depth = -FLT_MAX;
			for (uint32 i = 0; i < output_cm->mNumContactPoints; ++i)
			{
				const CachedContactPoint &ccp = output_cm->mContactPoints[i];
				manifold.mRelativeContactPointsOn1[i] = transform_body1.Multiply3x3(Vec3::sLoadFloat3Unsafe(ccp.mPosition1));
				manifold.mRelativeContactPointsOn2[i] = local_transform_body2 * Vec3::sLoadFloat3Unsafe(ccp.mPosition2);
				penetration_depth = max(penetration_depth, (manifold.mRelativeContactPointsOn1[i] - manifold.mRelativeContactPointsOn2[i]).Dot(world_space_normal));
			}
			manifold.mPenetrationDepth = penetration_depth; // We don't have the penetration depth anymore, estimate it

			// Notify callback
			mContactListener->OnContactPersisted(inBody1, inBody2, manifold, settings);
		}

		JPH_ASSERT(settings.mIsSensor || !(inBody1.IsSensor() || inBody2.IsSensor()), "Sensors cannot be converted into regular bodies by a contact callback!");
		if (!settings.mIsSensor // If one of the bodies is a sensor, don't actually create the constraint
			&& ((Type1 == EMotionType::Dynamic && settings.mInvMassScale1 != 0.0f) // One of the bodies must have mass to be able to create a contact constraint
				|| (Type2 == EMotionType::Dynamic && settings.mInvMassScale2 != 0.0f)))
		{
			// Create a new constraint
			ContactConstraint<Type1, Type2> *constraint = CreateConstraint<Type1, Type2>(link_bodies, inBody1, inBody2, input_hash, output_handle, world_space_normal, settings, output_cm->mNumContactPoints);
			if (constraint == nullptr)
			{
				ioContactAllocator.mErrors |= EPhysicsUpdateError::ContactConstraintsFull;
				break;
			}

			JPH_DET_LOG("GetContactsFromCache: id1: " << inBody1.GetID() << " id2: " << inBody2.GetID() << " key: " << constraint->mSortKey);

			// Calculate scaled mass and inertia
			Mat44 inv_i1;
			if constexpr (Type1 == EMotionType::Dynamic)
			{
				const MotionProperties *mp1 = inBody1.GetMotionPropertiesUnchecked();
				constraint->mInvMass1 = settings.mInvMassScale1 * mp1->GetInverseMass();
				inv_i1 = settings.mInvInertiaScale1 * mp1->GetInverseInertiaForRotation(transform_body1.GetRotation());
			}
			else
			{
				constraint->mInvMass1 = 0.0f;
				inv_i1 = Mat44::sZero();
			}

			Mat44 inv_i2;
			if constexpr (Type2 == EMotionType::Dynamic)
			{
				const MotionProperties *mp2 = inBody2.GetMotionPropertiesUnchecked();
				constraint->mInvMass2 = settings.mInvMassScale2 * mp2->GetInverseMass();
				inv_i2 = settings.mInvInertiaScale2 * mp2->GetInverseInertiaForRotation(transform_body2.GetRotation());
			}
			else
			{
				constraint->mInvMass2 = 0.0f;
				inv_i2 = Mat44::sZero();
			}

			// Setup non-penetration constraints
			RVec3 ws_contacts[MaxContactPoints];
			for (uint32 i = 0; i < constraint->mNumContactPoints; ++i)
			{
				const CachedContactPoint &ccp = output_cm->mContactPoints[i];
				WorldContactPoint<Type1, Type2> &wcp = constraint->mContactPoints[i];

				RVec3 p1_ws = transform_body1 * Vec3::sLoadFloat3Unsafe(ccp.mPosition1);
				RVec3 p2_ws = transform_body2 * Vec3::sLoadFloat3Unsafe(ccp.mPosition2);

				// Remember where to apply friction
				ws_contacts[i] = 0.5_r * (p1_ws + p2_ws);

				wcp.mNonPenetrationConstraint.SetTotalLambda(ccp.mNonPenetrationLambda);
				wcp.CalculateNonPenetrationConstraintProperties(delta_time, gravity, inBody1, inBody2, constraint->mInvMass1, constraint->mInvMass2, inv_i1, inv_i2, p1_ws, p2_ws, world_space_normal, settings, mPhysicsSettings.mMinVelocityForRestitution);
			}

			// Calculate tangents
			Vec3 t1, t2;
			constraint->GetTangents(t1, t2);

			// Setup friction constraints
			constraint->mFrictionConstraint1.SetTotalLambda(output_cm->mFrictionLambda[0]);
			constraint->mFrictionConstraint2.SetTotalLambda(output_cm->mFrictionLambda[1]);
			constraint->mAngularFrictionConstraint.SetTotalLambda(output_cm->mAngularFrictionLambda);
			constraint->CalculateFrictionConstraintProperties(inBody1, inBody2, constraint->mInvMass1, constraint->mInvMass2, inv_i1, inv_i2, ws_contacts, world_space_normal, t1, t2, settings);

		#ifdef JPH_DEBUG_RENDERER
			// Draw the manifold
			if (sDrawContactManifolds)
				constraint->Draw(DebugRenderer::sInstance, *mWriteCache, Color::sYellow);
		#endif // JPH_DEBUG_RENDERER
		}

		// Mark contact as persisted so that we won't fire OnContactRemoved callbacks
		input_cm.mFlags |= (uint16)CachedManifold::EFlags::ContactPersisted;

		// Fetch the next manifold
		input_handle = input_cm.mNextWithSameBodyPair;
	}
	while (input_handle != ManifoldMap::cInvalidHandle);
	outCachedBodyPair.mFirstCachedManifold = output_handle;
}

void ContactConstraintManager::GetContactsFromCache(ContactAllocator &ioContactAllocator, Body &inBody1, Body &inBody2, bool &outPairHandled)
{
	// Start with not handled
	outPairHandled = false;

	// Swap bodies so that body 1 id < body 2 id
	Body *body1, *body2;
	if (inBody1.GetID() < inBody2.GetID())
	{
		body1 = &inBody1;
		body2 = &inBody2;
	}
	else
	{
		body1 = &inBody2;
		body2 = &inBody1;
	}

	// Find the cached body pair
	BodyPair body_pair_key(body1->GetID(), body2->GetID());
	uint64 body_pair_hash = body_pair_key.GetHash();
	const BPKeyValue *kv = mReadCache->Find(body_pair_key, body_pair_hash);
	if (kv == nullptr)
		return;
	const CachedBodyPair &input_cbp = kv->GetValue();

	// Get relative translation
	Quat inv_r1 = body1->GetRotation().Conjugated();
	Vec3 delta_position = inv_r1 * Vec3(body2->GetCenterOfMassPosition() - body1->GetCenterOfMassPosition());

	// Get old position delta
	Vec3 old_delta_position = Vec3::sLoadFloat3Unsafe(input_cbp.mDeltaPosition);

	// Check if bodies are still roughly in the same relative position
	if ((delta_position - old_delta_position).LengthSq() > mPhysicsSettings.mBodyPairCacheMaxDeltaPositionSq)
		return;

	// Determine relative orientation
	Quat delta_rotation = inv_r1 * body2->GetRotation();

	// Reconstruct old quaternion delta
	Quat old_delta_rotation = Quat::sLoadFloat3Unsafe(input_cbp.mDeltaRotation);

	// Check if bodies are still roughly in the same relative orientation
	// The delta between 2 quaternions p and q is: p q^* = [rotation_axis * sin(angle / 2), cos(angle / 2)]
	// From the W component we can extract the angle: cos(angle / 2) = px * qx + py * qy + pz * qz + pw * qw = p . q
	// Since we want to abort if the rotation is smaller than -angle or bigger than angle, we can write the comparison as |p . q| < cos(angle / 2)
	if (abs(delta_rotation.Dot(old_delta_rotation)) < mPhysicsSettings.mBodyPairCacheCosMaxDeltaRotationDiv2)
		return;

	// The cache is valid, return that we've handled this body pair
	outPairHandled = true;

	// Copy the cached body pair to this frame
	BPKeyValue *output_bp_kv = mWriteCache->Create(ioContactAllocator, body_pair_key, body_pair_hash);
	if (output_bp_kv == nullptr)
		return; // Out of cache space
	CachedBodyPair *output_cbp = &output_bp_kv->GetValue();
	memcpy(output_cbp, &input_cbp, sizeof(CachedBodyPair));

	// If there were no contacts, we have handled the contact
	if (input_cbp.mFirstCachedManifold == ManifoldMap::cInvalidHandle)
		return;

	// Build dispatch table
	// Note: Non-dynamic vs non-dynamic can happen in this case due to one body being a sensor, so we need to have an extended table here
	using DispatchFunc = void (ContactConstraintManager::*)(ContactAllocator &, Body &, Body &, const CachedBodyPair &, CachedBodyPair &);
	static const DispatchFunc table[3][3] = {
		{
			nullptr, // Static vs static doesn't exist
			&ContactConstraintManager::TemplatedGetContactsFromCache<EMotionType::Static, EMotionType::Kinematic>,
			&ContactConstraintManager::TemplatedGetContactsFromCache<EMotionType::Static, EMotionType::Dynamic>
		},
		{
			&ContactConstraintManager::TemplatedGetContactsFromCache<EMotionType::Kinematic, EMotionType::Static>,
			&ContactConstraintManager::TemplatedGetContactsFromCache<EMotionType::Kinematic, EMotionType::Kinematic>,
			&ContactConstraintManager::TemplatedGetContactsFromCache<EMotionType::Kinematic, EMotionType::Dynamic>
		},
		{
			&ContactConstraintManager::TemplatedGetContactsFromCache<EMotionType::Dynamic, EMotionType::Static>,
			&ContactConstraintManager::TemplatedGetContactsFromCache<EMotionType::Dynamic, EMotionType::Kinematic>,
			&ContactConstraintManager::TemplatedGetContactsFromCache<EMotionType::Dynamic, EMotionType::Dynamic>
		}
	};

	// Dispatch to the correct templated form
	(this->*table[(int)body1->GetMotionType()][(int)body2->GetMotionType()])(ioContactAllocator, *body1, *body2, input_cbp, *output_cbp);
}

ContactConstraintManager::BodyPairHandle ContactConstraintManager::AddBodyPair(ContactAllocator &ioContactAllocator, const Body &inBody1, const Body &inBody2)
{
	// Swap bodies so that body 1 id < body 2 id
	const Body *body1, *body2;
	if (inBody1.GetID() < inBody2.GetID())
	{
		body1 = &inBody1;
		body2 = &inBody2;
	}
	else
	{
		body1 = &inBody2;
		body2 = &inBody1;
	}

	// Add an entry
	BodyPair body_pair_key(body1->GetID(), body2->GetID());
	uint64 body_pair_hash = body_pair_key.GetHash();
	BPKeyValue *body_pair_kv = mWriteCache->Create(ioContactAllocator, body_pair_key, body_pair_hash);
	if (body_pair_kv == nullptr)
		return nullptr; // Out of cache space
	CachedBodyPair *cbp = &body_pair_kv->GetValue();
	cbp->mFirstCachedManifold = ManifoldMap::cInvalidHandle;

	// Get relative translation
	Quat inv_r1 = body1->GetRotation().Conjugated();
	Vec3 delta_position = inv_r1 * Vec3(body2->GetCenterOfMassPosition() - body1->GetCenterOfMassPosition());

	// Store it
	delta_position.StoreFloat3(&cbp->mDeltaPosition);

	// Determine relative orientation
	Quat delta_rotation = inv_r1 * body2->GetRotation();

	// Store it
	delta_rotation.StoreFloat3(&cbp->mDeltaRotation);

	return cbp;
}

template <EMotionType Type1, EMotionType Type2>
void ContactConstraintManager::TemplatedAddContactConstraint(ContactAllocator &ioContactAllocator, bool &ioActivateAndLinkBodies, BodyPairHandle inBodyPairHandle, Body &inBody1, Body &inBody2, const ContactManifold &inManifold)
{
	// Calculate hash
	SubShapeIDPair key { inBody1.GetID(), inManifold.mSubShapeID1, inBody2.GetID(), inManifold.mSubShapeID2 };
	uint64 key_hash = key.GetHash();

	// Determine number of contact points
	int num_contact_points = (int)inManifold.mRelativeContactPointsOn1.size();
	JPH_ASSERT(num_contact_points <= MaxContactPoints);
	JPH_ASSERT(num_contact_points == (int)inManifold.mRelativeContactPointsOn2.size());

	// Reserve space for new contact cache entry
	// Note that for dynamic vs dynamic we always require the first body to have a lower body id to get a consistent key
	// under which to look up the contact
	MKeyValue *new_manifold_kv = mWriteCache->Create(ioContactAllocator, key, key_hash, num_contact_points);
	if (new_manifold_kv == nullptr)
		return; // Out of cache space
	CachedManifold *new_manifold = &new_manifold_kv->GetValue();
	uint32 new_manifold_handle = mWriteCache->ToHandle(new_manifold_kv);

	// Transform the world space normal to the space of body 2 (this is usually the static body)
	RMat44 inverse_transform_body2 = inBody2.GetInverseCenterOfMassTransform();
	inverse_transform_body2.Multiply3x3(inManifold.mWorldSpaceNormal).Normalized().StoreFloat3(&new_manifold->mContactNormal);

	// Settings object that gets passed to the callback
	ContactSettings settings;
	settings.mCombinedFriction = mCombineFriction(inBody1, inManifold.mSubShapeID1, inBody2, inManifold.mSubShapeID2);
	settings.mCombinedRestitution = mCombineRestitution(inBody1, inManifold.mSubShapeID1, inBody2, inManifold.mSubShapeID2);
	settings.mIsSensor = inBody1.IsSensor() || inBody2.IsSensor();

	// Get the contact points for the old cache entry
	const MKeyValue *old_manifold_kv = mReadCache->Find(key, key_hash);
	const CachedContactPoint *ccp_start;
	const CachedContactPoint *ccp_end;
	if (old_manifold_kv != nullptr)
	{
		// Call point persisted listener
		if (mContactListener != nullptr)
			mContactListener->OnContactPersisted(inBody1, inBody2, inManifold, settings);

		// Fetch the contact points from the old manifold
		const CachedManifold *old_manifold = &old_manifold_kv->GetValue();
		ccp_start = old_manifold->mContactPoints;
		ccp_end = ccp_start + old_manifold->mNumContactPoints;

		// Mark contact as persisted so that we won't fire OnContactRemoved callbacks
		old_manifold->mFlags |= (uint16)CachedManifold::EFlags::ContactPersisted;
	}
	else
	{
		// Call point added listener
		if (mContactListener != nullptr)
			mContactListener->OnContactAdded(inBody1, inBody2, inManifold, settings);

		// No contact points available from old manifold
		ccp_start = nullptr;
		ccp_end = nullptr;
	}

	// Get inverse transform for body 1
	RMat44 inverse_transform_body1 = inBody1.GetInverseCenterOfMassTransform();

	// If one of the bodies is a sensor, don't actually create the constraint
	JPH_ASSERT(settings.mIsSensor || !(inBody1.IsSensor() || inBody2.IsSensor()), "Sensors cannot be converted into regular bodies by a contact callback!");
	if (!settings.mIsSensor
		&& ((Type1 == EMotionType::Dynamic && settings.mInvMassScale1 != 0.0f) // One of the bodies must have mass to be able to create a contact constraint
			|| (Type2 == EMotionType::Dynamic && settings.mInvMassScale2 != 0.0f)))
	{
		// Create a new constraint
		ContactConstraint<Type1, Type2> *constraint = CreateConstraint<Type1, Type2>(ioActivateAndLinkBodies, inBody1, inBody2, key_hash, new_manifold_handle, inManifold.mWorldSpaceNormal, settings, num_contact_points);
		if (constraint == nullptr)
		{
			ioContactAllocator.mErrors |= EPhysicsUpdateError::ContactConstraintsFull;

			// Manifold has been created already, we're not filling it in, so we need to reset the contact number of points.
			// Note that we don't hook it up to the body pair cache so that it won't be used as a cache during the next simulation.
			new_manifold->mNumContactPoints = 0;
			return;
		}

		JPH_DET_LOG("AddContactConstraint: id1: " << constraint->mBody1->GetID() << " id2: " << constraint->mBody2->GetID() << " key: " << constraint->mSortKey);

		// Get time step and gravity
		float delta_time = mUpdateContext->mStepDeltaTime;
		Vec3 gravity = mUpdateContext->mPhysicsSystem->GetGravity();

		// Calculate scaled mass and inertia
		Mat44 inv_i1;
		if constexpr (Type1 == EMotionType::Dynamic)
		{
			const MotionProperties *mp1 = inBody1.GetMotionPropertiesUnchecked();
			constraint->mInvMass1 = settings.mInvMassScale1 * mp1->GetInverseMass();
			inv_i1 = settings.mInvInertiaScale1 * mp1->GetInverseInertiaForRotation(inverse_transform_body1.Transposed3x3());
		}
		else
		{
			constraint->mInvMass1 = 0.0f;
			inv_i1 = Mat44::sZero();
		}

		Mat44 inv_i2;
		if constexpr (Type2 == EMotionType::Dynamic)
		{
			const MotionProperties *mp2 = inBody2.GetMotionPropertiesUnchecked();
			constraint->mInvMass2 = settings.mInvMassScale2 * mp2->GetInverseMass();
			inv_i2 = settings.mInvInertiaScale2 * mp2->GetInverseInertiaForRotation(inverse_transform_body2.Transposed3x3());
		}
		else
		{
			constraint->mInvMass2 = 0.0f;
			inv_i2 = Mat44::sZero();
		}

		RVec3 ws_contacts[MaxContactPoints];
		for (int i = 0; i < num_contact_points; ++i)
		{
			// Convert to world space and set positions
			WorldContactPoint<Type1, Type2> &wcp = constraint->mContactPoints[i];
			RVec3 p1_ws = inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn1[i];
			RVec3 p2_ws = inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn2[i];

			// Remember where to apply friction
			ws_contacts[i] = 0.5_r * (p1_ws + p2_ws);

			// Convert to local space to the body
			Vec3 p1_ls = Vec3(inverse_transform_body1 * p1_ws);
			Vec3 p2_ls = Vec3(inverse_transform_body2 * p2_ws);

			// Store contact points
			CachedContactPoint &cp = new_manifold->mContactPoints[i];
			p1_ls.StoreFloat3(&cp.mPosition1);
			p2_ls.StoreFloat3(&cp.mPosition2);

			// Check if we have a close contact point from last update
			wcp.mNonPenetrationConstraint.SetTotalLambda(0.0f);
			for (const CachedContactPoint *ccp = ccp_start; ccp < ccp_end; ccp++)
				if (Vec3::sLoadFloat3Unsafe(ccp->mPosition1).IsClose(p1_ls, mPhysicsSettings.mContactPointPreserveLambdaMaxDistSq)
					&& Vec3::sLoadFloat3Unsafe(ccp->mPosition2).IsClose(p2_ls, mPhysicsSettings.mContactPointPreserveLambdaMaxDistSq))
				{
					// Get lambdas from previous frame
					wcp.mNonPenetrationConstraint.SetTotalLambda(ccp->mNonPenetrationLambda);
					break;
				}

			// Setup velocity constraint
			wcp.CalculateNonPenetrationConstraintProperties(delta_time, gravity, inBody1, inBody2, constraint->mInvMass1, constraint->mInvMass2, inv_i1, inv_i2, p1_ws, p2_ws, inManifold.mWorldSpaceNormal, settings, mPhysicsSettings.mMinVelocityForRestitution);
		}

		// Calculate tangents
		Vec3 t1, t2;
		constraint->GetTangents(t1, t2);

		// Setup friction constraint
		if (old_manifold_kv != nullptr)
		{
			const CachedManifold *old_manifold = &old_manifold_kv->GetValue();
			constraint->mFrictionConstraint1.SetTotalLambda(old_manifold->mFrictionLambda[0]);
			constraint->mFrictionConstraint2.SetTotalLambda(old_manifold->mFrictionLambda[1]);
			constraint->mAngularFrictionConstraint.SetTotalLambda(old_manifold->mAngularFrictionLambda);
		}
		else
		{
			constraint->mFrictionConstraint1.SetTotalLambda(0.0f);
			constraint->mFrictionConstraint2.SetTotalLambda(0.0f);
			constraint->mAngularFrictionConstraint.SetTotalLambda(0.0f);
		}
		constraint->CalculateFrictionConstraintProperties(inBody1, inBody2, constraint->mInvMass1, constraint->mInvMass2, inv_i1, inv_i2, ws_contacts, inManifold.mWorldSpaceNormal, t1, t2, settings);

#ifdef JPH_DEBUG_RENDERER
		// Draw the manifold
		if (sDrawContactManifolds)
			constraint->Draw(DebugRenderer::sInstance, *mWriteCache, Color::sOrange);
	#endif // JPH_DEBUG_RENDERER
	}
	else
	{
		// Store the contact manifold in the cache
		for (int i = 0; i < num_contact_points; ++i)
		{
			// Convert to local space to the body
			Vec3 p1 = Vec3(inverse_transform_body1 * (inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn1[i]));
			Vec3 p2 = Vec3(inverse_transform_body2 * (inManifold.mBaseOffset + inManifold.mRelativeContactPointsOn2[i]));

			// Create new contact point
			CachedContactPoint &cp = new_manifold->mContactPoints[i];
			p1.StoreFloat3(&cp.mPosition1);
			p2.StoreFloat3(&cp.mPosition2);

			// Reset contact impulses, we haven't applied any
			cp.mNonPenetrationLambda = 0.0f;
		}

		new_manifold->mFrictionLambda[0] = 0.0f;
		new_manifold->mFrictionLambda[1] = 0.0f;
		new_manifold->mAngularFrictionLambda = 0.0f;
	}

	// Store cached contact point in body pair cache
	CachedBodyPair *cbp = reinterpret_cast<CachedBodyPair *>(inBodyPairHandle);
	new_manifold->mNextWithSameBodyPair = cbp->mFirstCachedManifold;
	cbp->mFirstCachedManifold = new_manifold_handle;
}

void ContactConstraintManager::AddContactConstraint(ContactAllocator &ioContactAllocator, bool &ioActivateAndLinkBodies, BodyPairHandle inBodyPairHandle, Body &inBody1, Body &inBody2, const ContactManifold &inManifold)
{
	JPH_PROFILE_FUNCTION();

	JPH_DET_LOG("AddContactConstraint: id1: " << inBody1.GetID() << " id2: " << inBody2.GetID()
		<< " subshape1: " << inManifold.mSubShapeID1 << " subshape2: " << inManifold.mSubShapeID2
		<< " normal: " << inManifold.mWorldSpaceNormal << " pendepth: " << inManifold.mPenetrationDepth);

	JPH_ASSERT(inManifold.mWorldSpaceNormal.IsNormalized());

	// Swap bodies so that body 1 id < body 2 id
	const ContactManifold *manifold;
	Body *body1, *body2;
	ContactManifold temp;
	if (inBody2.GetID() < inBody1.GetID())
	{
		body1 = &inBody2;
		body2 = &inBody1;
		temp = inManifold.SwapShapes();
		manifold = &temp;
	}
	else
	{
		body1 = &inBody1;
		body2 = &inBody2;
		manifold = &inManifold;
	}

	// Build dispatch table
	// Note: Non-dynamic vs non-dynamic can happen in this case due to one body being a sensor, so we need to have an extended table here
	using DispatchFunc = void (ContactConstraintManager::*)(ContactAllocator &, bool &, BodyPairHandle, Body &, Body &, const ContactManifold &);
	static const DispatchFunc table[3][3] = {
		{
			nullptr, // Static vs static doesn't exist
			&ContactConstraintManager::TemplatedAddContactConstraint<EMotionType::Static, EMotionType::Kinematic>,
			&ContactConstraintManager::TemplatedAddContactConstraint<EMotionType::Static, EMotionType::Dynamic>
		},
		{
			&ContactConstraintManager::TemplatedAddContactConstraint<EMotionType::Kinematic, EMotionType::Static>,
			&ContactConstraintManager::TemplatedAddContactConstraint<EMotionType::Kinematic, EMotionType::Kinematic>,
			&ContactConstraintManager::TemplatedAddContactConstraint<EMotionType::Kinematic, EMotionType::Dynamic>
		},
		{
			&ContactConstraintManager::TemplatedAddContactConstraint<EMotionType::Dynamic, EMotionType::Static>,
			&ContactConstraintManager::TemplatedAddContactConstraint<EMotionType::Dynamic, EMotionType::Kinematic>,
			&ContactConstraintManager::TemplatedAddContactConstraint<EMotionType::Dynamic, EMotionType::Dynamic>
		}
	};

	// Dispatch to the correct templated form
	return (this->*table[(int)body1->GetMotionType()][(int)body2->GetMotionType()])(ioContactAllocator, ioActivateAndLinkBodies, inBodyPairHandle, *body1, *body2, *manifold);
}

void ContactConstraintManager::OnCCDContactAdded(ContactAllocator &ioContactAllocator, const Body &inBody1, const Body &inBody2, const ContactManifold &inManifold, ContactSettings &outSettings)
{
	JPH_ASSERT(inManifold.mWorldSpaceNormal.IsNormalized());

	// Calculate contact settings
	outSettings.mCombinedFriction = mCombineFriction(inBody1, inManifold.mSubShapeID1, inBody2, inManifold.mSubShapeID2);
	outSettings.mCombinedRestitution = mCombineRestitution(inBody1, inManifold.mSubShapeID1, inBody2, inManifold.mSubShapeID2);
	outSettings.mIsSensor = false; // For now, no sensors are supported during CCD

	// The remainder of this function only deals with calling contact callbacks, if there's no contact callback we also don't need to do this work
	if (mContactListener != nullptr)
	{
		// Swap bodies so that body 1 id < body 2 id
		const ContactManifold *manifold;
		const Body *body1, *body2;
		ContactManifold temp;
		if (inBody2.GetID() < inBody1.GetID())
		{
			body1 = &inBody2;
			body2 = &inBody1;
			temp = inManifold.SwapShapes();
			manifold = &temp;
		}
		else
		{
			body1 = &inBody1;
			body2 = &inBody2;
			manifold = &inManifold;
		}

		// Calculate hash
		SubShapeIDPair key { body1->GetID(), manifold->mSubShapeID1, body2->GetID(), manifold->mSubShapeID2 };
		uint64 key_hash = key.GetHash();

		// Check if we already created this contact this physics update
		MKVAndCreated new_manifold_kv = mWriteCache->FindOrCreate(ioContactAllocator, key, key_hash, 0);
		if (new_manifold_kv.second)
		{
			// This contact is new for this physics update, check if previous update we already had this contact.
			const MKeyValue *old_manifold_kv = mReadCache->Find(key, key_hash);
			if (old_manifold_kv == nullptr)
			{
				// New contact
				mContactListener->OnContactAdded(*body1, *body2, *manifold, outSettings);
			}
			else
			{
				// Existing contact
				mContactListener->OnContactPersisted(*body1, *body2, *manifold, outSettings);

				// Mark contact as persisted so that we won't fire OnContactRemoved callbacks
				old_manifold_kv->GetValue().mFlags |= (uint16)CachedManifold::EFlags::ContactPersisted;
			}

			// Check if the cache is full
			if (new_manifold_kv.first != nullptr)
			{
				// We don't store any contact points in this manifold as it is not for caching impulses, we only need to know that the contact was created
				CachedManifold &new_manifold = new_manifold_kv.first->GetValue();
				new_manifold.mContactNormal = { 0, 0, 0 };
				new_manifold.mFlags |= (uint16)CachedManifold::EFlags::CCDContact;
			}
		}
		else
		{
			// Already found this contact this physics update.
			// Note that we can trigger OnContactPersisted multiple times per physics update, but otherwise we have no way of obtaining the settings
			mContactListener->OnContactPersisted(*body1, *body2, *manifold, outSettings);
		}

		// If we swapped body1 and body2 we need to swap the mass scales back
		if (manifold == &temp)
		{
			std::swap(outSettings.mInvMassScale1, outSettings.mInvMassScale2);
			std::swap(outSettings.mInvInertiaScale1, outSettings.mInvInertiaScale2);
			// Note we do not need to negate the relative surface velocity as it is not applied by the CCD collision constraint
		}
	}

	JPH_ASSERT(outSettings.mIsSensor || !(inBody1.IsSensor() || inBody2.IsSensor()), "Sensors cannot be converted into regular bodies by a contact callback!");
}

void ContactConstraintManager::ConstraintIdxToConstraintOffset(uint32 *ioConstraintIdxBegin, const uint32 *inConstraintIdxEnd) const
{
	for (uint32 *i = ioConstraintIdxBegin; i < inConstraintIdxEnd; ++i)
		*i = mConstraintIdxToOffset[*i];
}

void ContactConstraintManager::SortContacts(uint32 *ioConstraintOffsetBegin, uint32 *inConstraintOffsetEnd) const
{
	JPH_PROFILE_FUNCTION();

	QuickSort(ioConstraintOffsetBegin, inConstraintOffsetEnd, [this](uint32 inLHS, uint32 inRHS) {
		const ContactConstraintBase &lhs = *reinterpret_cast<const ContactConstraintBase *>(mConstraints + inLHS);
		const ContactConstraintBase &rhs = *reinterpret_cast<const ContactConstraintBase *>(mConstraints + inRHS);

		// Most of the time the sort key will be different so we sort on that
		if (lhs.mSortKey != rhs.mSortKey)
			return lhs.mSortKey < rhs.mSortKey;

		// If they're equal we use the IDs of body 1 to order
		if (lhs.mBody1 != rhs.mBody1)
			return lhs.mBody1->GetID() < rhs.mBody1->GetID();

		// If they're still equal we use the IDs of body 2 to order
		if (lhs.mBody2 != rhs.mBody2)
			return lhs.mBody2->GetID() < rhs.mBody2->GetID();

		JPH_ASSERT(inLHS == inRHS, "Hash collision, ordering will be inconsistent");
		return false;
	});
}

void ContactConstraintManager::FinalizeContactCacheAndCallContactPointRemovedCallbacks(uint inExpectedNumBodyPairs, uint inExpectedNumManifolds)
{
	JPH_PROFILE_FUNCTION();

#ifdef JPH_ENABLE_ASSERTS
	// Mark cache as finalized
	ManifoldCache &old_write_cache = mCache[mCacheWriteIdx];
	old_write_cache.Finalize();

	// Check that the count of body pairs and manifolds that we tracked outside of the cache (to avoid contention on an atomic) is correct
	JPH_ASSERT(old_write_cache.GetNumBodyPairs() == inExpectedNumBodyPairs);
	JPH_ASSERT(old_write_cache.GetNumManifolds() == inExpectedNumManifolds);
#endif

	// Buffers are now complete, make write buffer the read buffer
	mCacheWriteIdx ^= 1;

	// Get the old read cache / new write cache
	ManifoldCache &old_read_cache = mCache[mCacheWriteIdx];

	// Call the contact point removal callbacks
	if (mContactListener != nullptr)
		old_read_cache.ContactPointRemovedCallbacks(mContactListener);

	// We're done with the old read cache now
	old_read_cache.Clear();

	// Use the amount of contacts from the last iteration to determine the amount of buckets to use in the hash map for the next iteration
	old_read_cache.Prepare(inExpectedNumBodyPairs, inExpectedNumManifolds);
}

bool ContactConstraintManager::WereBodiesInContact(const BodyID &inBody1ID, const BodyID &inBody2ID) const
{
	// The body pair needs to be in the cache and it needs to have a manifold (otherwise it's just a record indicating that there are no collisions)
	const ManifoldCache &read_cache = mCache[mCacheWriteIdx ^ 1];
	BodyPair key;
	if (inBody1ID < inBody2ID)
		key = BodyPair(inBody1ID, inBody2ID);
	else
		key = BodyPair(inBody2ID, inBody1ID);
	uint64 key_hash = key.GetHash();
	const BPKeyValue *kv = read_cache.Find(key, key_hash);
	return kv != nullptr && kv->GetValue().mFirstCachedManifold != ManifoldMap::cInvalidHandle;
}

template <EMotionType Type1, EMotionType Type2>
void ContactConstraintManager::sGetVelocities(const MotionProperties *inMotionProperties1, const MotionProperties *inMotionProperties2, Vec3 &outLinearVelocity1, Vec3 &outAngularVelocity1, Vec3 &outLinearVelocity2, Vec3 &outAngularVelocity2)
{
	if constexpr (Type1 != EMotionType::Static)
	{
		outLinearVelocity1 = inMotionProperties1->GetLinearVelocity();
		outAngularVelocity1 = inMotionProperties1->GetAngularVelocity();
	}
	else
	{
		JPH_IF_DEBUG(outLinearVelocity1 = Vec3::sNaN();)
		JPH_IF_DEBUG(outAngularVelocity1 = Vec3::sNaN();)
	}

	if constexpr (Type2 != EMotionType::Static)
	{
		outLinearVelocity2 = inMotionProperties2->GetLinearVelocity();
		outAngularVelocity2 = inMotionProperties2->GetAngularVelocity();
	}
	else
	{
		JPH_IF_DEBUG(outLinearVelocity2 = Vec3::sNaN();)
		JPH_IF_DEBUG(outAngularVelocity2 = Vec3::sNaN();)
	}
}

template <EMotionType Type1, EMotionType Type2>
void ContactConstraintManager::sSetVelocities(MotionProperties *ioMotionProperties1, MotionProperties *ioMotionProperties2, Vec3Arg inLinearVelocity1, Vec3Arg inAngularVelocity1, Vec3Arg inLinearVelocity2, Vec3Arg inAngularVelocity2)
{
	if constexpr (Type1 == EMotionType::Dynamic)
	{
		ioMotionProperties1->ApplyLinearVelocityStep(inLinearVelocity1);
		ioMotionProperties1->ApplyAngularVelocityStep(inAngularVelocity1);
	}

	if constexpr (Type2 == EMotionType::Dynamic)
	{
		ioMotionProperties2->ApplyLinearVelocityStep(inLinearVelocity2);
		ioMotionProperties2->ApplyAngularVelocityStep(inAngularVelocity2);
	}
}

template <EMotionType Type1, EMotionType Type2>
void ContactConstraintManager::sWarmStartConstraint(ContactConstraintBase &ioConstraint, MotionProperties *ioMotionProperties1, MotionProperties *ioMotionProperties2, float inWarmStartImpulseRatio)
{
	ContactConstraint<Type1, Type2> &constraint = static_cast<ContactConstraint<Type1, Type2> &>(ioConstraint);

	bool any_impulse_applied = false;

	// Calculate tangents
	Vec3 t1, t2;
	constraint.GetTangents(t1, t2);

	// Get velocities
	Vec3 linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2;
	sGetVelocities<Type1, Type2>(ioMotionProperties1, ioMotionProperties2, linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2);

	Vec3 ws_normal = constraint.GetWorldSpaceNormal();

	// Warm starting: Apply impulse from last frame
	if (constraint.mFrictionConstraint1.IsActive() && constraint.mFrictionConstraint1.WarmStart(linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2, constraint.mInvMass1, constraint.mInvMass2, t1, inWarmStartImpulseRatio))
		any_impulse_applied = true;
	if (constraint.mFrictionConstraint2.IsActive() && constraint.mFrictionConstraint2.WarmStart(linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2, constraint.mInvMass1, constraint.mInvMass2, t2, inWarmStartImpulseRatio))
		any_impulse_applied = true;
	if (constraint.mAngularFrictionConstraint.IsActive() && constraint.mAngularFrictionConstraint.WarmStart(angular_velocity1, angular_velocity2, inWarmStartImpulseRatio))
		any_impulse_applied = true;

	for (uint32 i = 0; i < constraint.mNumContactPoints; ++i)
	{
		WorldContactPoint<Type1, Type2> &wcp = constraint.mContactPoints[i];
		if (wcp.mNonPenetrationConstraint.WarmStart(linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2, constraint.mInvMass1, constraint.mInvMass2, ws_normal, inWarmStartImpulseRatio))
			any_impulse_applied = true;
	}

	// Apply changed velocities
	if (any_impulse_applied)
		sSetVelocities<Type1, Type2>(ioMotionProperties1, ioMotionProperties2, linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2);
}

template <class MotionPropertiesCallback>
void ContactConstraintManager::WarmStartVelocityConstraints(const uint32 *inConstraintOffsetBegin, const uint32 *inConstraintOffsetEnd, float inWarmStartImpulseRatio, MotionPropertiesCallback &ioCallback)
{
	JPH_PROFILE_FUNCTION();

	// Build dispatch table
	using DispatchFunc = void (*)(ContactConstraintBase &, MotionProperties *, MotionProperties *, float);
	static const DispatchFunc table[3][3] = {
		{
			nullptr, // Static vs static doesn't exist
			nullptr, // Static vs kinematic doesn't exist
			sWarmStartConstraint<EMotionType::Static, EMotionType::Dynamic>
		},
		{
			nullptr, // Kinematic vs static doesn't exist
			nullptr, // Kinematic vs kinematic doesn't exist
			sWarmStartConstraint<EMotionType::Kinematic, EMotionType::Dynamic>
		},
		{
			sWarmStartConstraint<EMotionType::Dynamic, EMotionType::Static>,
			sWarmStartConstraint<EMotionType::Dynamic, EMotionType::Kinematic>,
			sWarmStartConstraint<EMotionType::Dynamic, EMotionType::Dynamic>
		}
	};

	if (inConstraintOffsetBegin >= inConstraintOffsetEnd)
		return;

	ContactConstraintBase *next_constraint = reinterpret_cast<ContactConstraintBase *>(mConstraints + *inConstraintOffsetBegin);
	for (const uint32 *next_constraint_offset = inConstraintOffsetBegin + 1; next_constraint != nullptr; ++next_constraint_offset)
	{
		ContactConstraintBase &constraint = *next_constraint;
		if (next_constraint_offset < inConstraintOffsetEnd)
		{
			next_constraint = reinterpret_cast<ContactConstraintBase *>(mConstraints + *next_constraint_offset);
			PrefetchL1(next_constraint);
		}
		else
			next_constraint = nullptr;

		// Dispatch to the correct templated form
		Body &body1 = *constraint.mBody1;
		Body &body2 = *constraint.mBody2;
		MotionProperties *motion_properties1 = body1.GetMotionPropertiesUnchecked();
		MotionProperties *motion_properties2 = body2.GetMotionPropertiesUnchecked();
		table[(int)body1.GetMotionType()][(int)body2.GetMotionType()](constraint, motion_properties1, motion_properties2, inWarmStartImpulseRatio);

		// Call callbacks
		if (body1.IsDynamic())
			ioCallback(motion_properties1);
		if (body2.IsDynamic())
			ioCallback(motion_properties2);
	}
}

// Specialize for the two body callback types
template void ContactConstraintManager::WarmStartVelocityConstraints<CalculateSolverSteps>(const uint32 *inConstraintOffsetBegin, const uint32 *inConstraintOffsetEnd, float inWarmStartImpulseRatio, CalculateSolverSteps &ioCallback);
template void ContactConstraintManager::WarmStartVelocityConstraints<DummyCalculateSolverSteps>(const uint32 *inConstraintOffsetBegin, const uint32 *inConstraintOffsetEnd, float inWarmStartImpulseRatio, DummyCalculateSolverSteps &ioCallback);

template <EMotionType Type1, EMotionType Type2>
bool ContactConstraintManager::sSolveVelocityConstraint(ContactConstraintBase &ioConstraint, MotionProperties *ioMotionProperties1, MotionProperties *ioMotionProperties2)
{
	ContactConstraint<Type1, Type2> &constraint = static_cast<ContactConstraint<Type1, Type2> &>(ioConstraint);

	bool any_impulse_applied = false;

	// Calculate tangents
	Vec3 t1, t2;
	constraint.GetTangents(t1, t2);

	// Get velocities
	Vec3 linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2;
	sGetVelocities<Type1, Type2>(ioMotionProperties1, ioMotionProperties2, linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2);

	bool linear_friction_active = constraint.mFrictionConstraint1.IsActive() || constraint.mFrictionConstraint2.IsActive();
	bool angular_friction_active = constraint.mAngularFrictionConstraint.IsActive();

	// Calculate max impulse that can be applied. Note that we're using the non-penetration impulse from the previous iteration here.
	// We do this because non-penetration is more important so is solved last (the last things that are solved in an iterative solver
	// contribute the most).
	float max_linear_lambda = 0.0f, max_angular_lambda = 0.0f;
	if (linear_friction_active || angular_friction_active)
	{
		for (uint32 i = 0; i < constraint.mNumContactPoints; ++i)
		{
			WorldContactPoint<Type1, Type2> &wcp = constraint.mContactPoints[i];
			float lambda = wcp.mNonPenetrationConstraint.GetTotalLambda();
			max_linear_lambda += lambda;
			max_angular_lambda += wcp.mDistanceToFrictionCenter * lambda;
		}
		max_linear_lambda *= constraint.mCombinedFriction;
		max_angular_lambda *= constraint.mCombinedFriction;
	}

	// First apply friction constraint (non-penetration is more important than friction)
	if (linear_friction_active)
	{
		// Calculate impulse to stop motion in tangential direction
		float lambda1 = constraint.mFrictionConstraint1.SolveVelocityConstraintGetTotalLambda(linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2, t1);
		float lambda2 = constraint.mFrictionConstraint2.SolveVelocityConstraintGetTotalLambda(linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2, t2);

		// If the total lambda that we will apply is too large, scale it back
		float total_lambda_sq = Square(lambda1) + Square(lambda2);
		if (total_lambda_sq > Square(max_linear_lambda))
		{
			float scale = max_linear_lambda / Sqrt(total_lambda_sq);
			lambda1 *= scale;
			lambda2 *= scale;
		}

		// Apply the friction impulse
		if (constraint.mFrictionConstraint1.SolveVelocityConstraintApplyLambda(linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2, constraint.mInvMass1, constraint.mInvMass2, t1, lambda1))
			any_impulse_applied = true;
		if (constraint.mFrictionConstraint2.SolveVelocityConstraintApplyLambda(linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2, constraint.mInvMass1, constraint.mInvMass2, t2, lambda2))
			any_impulse_applied = true;
	}

	// Apply angular friction
	Vec3 ws_normal = constraint.GetWorldSpaceNormal();
	if (angular_friction_active && constraint.mAngularFrictionConstraint.SolveVelocityConstraint(angular_velocity1, angular_velocity2, ws_normal, -max_angular_lambda, max_angular_lambda))
		any_impulse_applied = true;

	// Then apply all non-penetration constraints
	for (uint32 i = 0; i < constraint.mNumContactPoints; ++i)
	{
		WorldContactPoint<Type1, Type2> &wcp = constraint.mContactPoints[i];

		// Calculate impulse
		float total_lambda = wcp.mNonPenetrationConstraint.SolveVelocityConstraintGetTotalLambda(linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2, ws_normal);

		// Contact constraints can only push and not pull
		total_lambda = max(total_lambda, 0.0f);

		// Apply impulse
		if (wcp.mNonPenetrationConstraint.SolveVelocityConstraintApplyLambda(linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2, constraint.mInvMass1, constraint.mInvMass2, ws_normal, total_lambda))
			any_impulse_applied = true;
	}

	if (!any_impulse_applied)
		return false;

	sSetVelocities<Type1, Type2>(ioMotionProperties1, ioMotionProperties2, linear_velocity1, angular_velocity1, linear_velocity2, angular_velocity2);
	return true;
}

bool ContactConstraintManager::SolveVelocityConstraints(const uint32 *inConstraintOffsetBegin, const uint32 *inConstraintOffsetEnd)
{
	JPH_PROFILE_FUNCTION();

	// Build dispatch table
	using DispatchFunc = bool (*)(ContactConstraintBase &, MotionProperties *, MotionProperties *);
	static const DispatchFunc table[3][3] = {
		{
			nullptr, // Static vs static doesn't exist
			nullptr, // Static vs kinematic doesn't exist
			sSolveVelocityConstraint<EMotionType::Static, EMotionType::Dynamic>
		},
		{
			nullptr, // Kinematic vs static doesn't exist
			nullptr, // Kinematic vs kinematic doesn't exist
			sSolveVelocityConstraint<EMotionType::Kinematic, EMotionType::Dynamic>
		},
		{
			sSolveVelocityConstraint<EMotionType::Dynamic, EMotionType::Static>,
			sSolveVelocityConstraint<EMotionType::Dynamic, EMotionType::Kinematic>,
			sSolveVelocityConstraint<EMotionType::Dynamic, EMotionType::Dynamic>
		}
	};

	if (inConstraintOffsetBegin >= inConstraintOffsetEnd)
		return false;

	bool any_impulse_applied = false;

	ContactConstraintBase *next_constraint = reinterpret_cast<ContactConstraintBase *>(mConstraints + *inConstraintOffsetBegin);
	for (const uint32 *next_constraint_offset = inConstraintOffsetBegin + 1; next_constraint != nullptr; ++next_constraint_offset)
	{
		ContactConstraintBase &constraint = *next_constraint;
		if (next_constraint_offset < inConstraintOffsetEnd)
		{
			next_constraint = reinterpret_cast<ContactConstraintBase *>(mConstraints + *next_constraint_offset);
			PrefetchL1(next_constraint);
		}
		else
			next_constraint = nullptr;

		// Dispatch to the correct templated form
		Body &body1 = *constraint.mBody1;
		Body &body2 = *constraint.mBody2;
		any_impulse_applied |= table[(int)body1.GetMotionType()][(int)body2.GetMotionType()](constraint, body1.GetMotionPropertiesUnchecked(), body2.GetMotionPropertiesUnchecked());
	}

	return any_impulse_applied;
}

template <EMotionType Type1, EMotionType Type2>
void ContactConstraintManager::sStoreAppliedImpulses(ContactConstraintBase &ioConstraint, ManifoldCache &inManifoldCache)
{
	ContactConstraint<Type1, Type2> &constraint = static_cast<ContactConstraint<Type1, Type2> &>(ioConstraint);
	CachedManifold &cached_manifold = inManifoldCache.FromHandle(constraint.mCachedManifoldHandle)->GetValue();

	for (uint32 i = 0; i < constraint.mNumContactPoints; ++i)
	{
		const WorldContactPoint<Type1, Type2> &wcp = constraint.mContactPoints[i];
		CachedContactPoint &ccp = cached_manifold.mContactPoints[i];
		ccp.mNonPenetrationLambda = wcp.mNonPenetrationConstraint.GetTotalLambda();
	}

	cached_manifold.mFrictionLambda[0] = constraint.mFrictionConstraint1.GetTotalLambda();
	cached_manifold.mFrictionLambda[1] = constraint.mFrictionConstraint2.GetTotalLambda();
	cached_manifold.mAngularFrictionLambda = constraint.mAngularFrictionConstraint.GetTotalLambda();
}

void ContactConstraintManager::StoreAppliedImpulses(const uint32 *inConstraintOffsetBegin, const uint32 *inConstraintOffsetEnd) const
{
	// Build dispatch table
	using DispatchFunc = void (*)(ContactConstraintBase &, ManifoldCache &);
	static const DispatchFunc table[3][3] = {
		{
			nullptr, // Static vs static doesn't exist
			nullptr, // Static vs kinematic doesn't exist
			sStoreAppliedImpulses<EMotionType::Static, EMotionType::Dynamic>
		},
		{
			nullptr, // Kinematic vs static doesn't exist
			nullptr, // Kinematic vs kinematic doesn't exist
			sStoreAppliedImpulses<EMotionType::Kinematic, EMotionType::Dynamic>
		},
		{
			sStoreAppliedImpulses<EMotionType::Dynamic, EMotionType::Static>,
			sStoreAppliedImpulses<EMotionType::Dynamic, EMotionType::Kinematic>,
			sStoreAppliedImpulses<EMotionType::Dynamic, EMotionType::Dynamic>
		}
	};

	if (inConstraintOffsetBegin >= inConstraintOffsetEnd)
		return;

	// Copy back total applied impulse to cache for the next frame
	ContactConstraintBase *next_constraint = reinterpret_cast<ContactConstraintBase *>(mConstraints + *inConstraintOffsetBegin);
	for (const uint32 *next_constraint_offset = inConstraintOffsetBegin + 1; next_constraint != nullptr; ++next_constraint_offset)
	{
		ContactConstraintBase &constraint = *next_constraint;
		if (next_constraint_offset < inConstraintOffsetEnd)
		{
			next_constraint = reinterpret_cast<ContactConstraintBase *>(mConstraints + *next_constraint_offset);
			PrefetchL1(next_constraint);
		}
		else
			next_constraint = nullptr;

		// Dispatch to the correct templated form
		table[(int)constraint.mBody1->GetMotionType()][(int)constraint.mBody2->GetMotionType()](constraint, *mWriteCache);
	}
}

template <EMotionType Type1, EMotionType Type2>
bool ContactConstraintManager::sSolvePositionConstraint(ContactConstraintBase &ioConstraint, Body &ioBody1, Body &ioBody2, const PhysicsSettings &inSettings, const ManifoldCache &inManifoldCache)
{
	ContactConstraint<Type1, Type2> &constraint = static_cast<ContactConstraint<Type1, Type2> &>(ioConstraint);
	const CachedManifold &cached_manifold = inManifoldCache.FromHandle(constraint.mCachedManifoldHandle)->GetValue();

	// Get transforms
	RMat44 transform1 = ioBody1.GetCenterOfMassTransform();
	RMat44 transform2 = ioBody2.GetCenterOfMassTransform();

	Vec3 ws_normal = constraint.GetWorldSpaceNormal();

	bool any_impulse_applied = false;

	for (uint32 i = 0; i < constraint.mNumContactPoints; ++i)
	{
		WorldContactPoint<Type1, Type2> &wcp = constraint.mContactPoints[i];
		const CachedContactPoint &ccp = cached_manifold.mContactPoints[i];

		// Calculate new contact point positions in world space (the bodies may have moved)
		RVec3 p1 = transform1 * Vec3::sLoadFloat3Unsafe(ccp.mPosition1);
		RVec3 p2 = transform2 * Vec3::sLoadFloat3Unsafe(ccp.mPosition2);

		// Calculate separation along the normal (negative if interpenetrating)
		// Allow a little penetration by default (PhysicsSettings::mPenetrationSlop) to avoid jittering between contact/no-contact which wipes out the contact cache and warm start impulses
		// Clamp penetration to a max PhysicsSettings::mMaxPenetrationDistance so that we don't apply a huge impulse if we're penetrating a lot
		float separation = max(Vec3(p2 - p1).Dot(ws_normal) + inSettings.mPenetrationSlop, -inSettings.mMaxPenetrationDistance);

		// Only enforce constraint when separation < 0 (otherwise we're apart)
		if (separation < 0.0f)
		{
			// Calculate scaled inertia
			Mat44 inv_i1;
			if constexpr (Type1 == EMotionType::Dynamic)
				inv_i1 = constraint.mInvInertiaScale1 * ioBody1.GetInverseInertia();
			else
				inv_i1 = Mat44::sZero();

			Mat44 inv_i2;
			if constexpr (Type2 == EMotionType::Dynamic)
				inv_i2 = constraint.mInvInertiaScale2 * ioBody2.GetInverseInertia();
			else
				inv_i2 = Mat44::sZero();

			// Calculate collision points relative to body
			RVec3 p = 0.5_r * (p1 + p2);
			Vec3 r1 = Vec3(p - ioBody1.GetCenterOfMassPosition());
			Vec3 r2 = Vec3(p - ioBody2.GetCenterOfMassPosition());

			// Update constraint properties (bodies may have moved)
			wcp.mNonPenetrationConstraint.CalculateConstraintProperties(constraint.mInvMass1, inv_i1, r1, constraint.mInvMass2, inv_i2, r2, ws_normal);

			// Solve position errors
			if (wcp.mNonPenetrationConstraint.SolvePositionConstraint(ioBody1, constraint.mInvMass1, ioBody2, constraint.mInvMass2, ws_normal, separation, inSettings.mBaumgarte))
				any_impulse_applied = true;
		}
	}

	return any_impulse_applied;
}

bool ContactConstraintManager::SolvePositionConstraints(const uint32 *inConstraintOffsetBegin, const uint32 *inConstraintOffsetEnd)
{
	JPH_PROFILE_FUNCTION();

	// Build dispatch table
	using DispatchFunc = bool (*)(ContactConstraintBase &, Body &, Body &, const PhysicsSettings &, const ManifoldCache &);
	static const DispatchFunc table[3][3] = {
		{
			nullptr, // Static vs static doesn't exist
			nullptr, // Static vs kinematic doesn't exist
			sSolvePositionConstraint<EMotionType::Static, EMotionType::Dynamic>
		},
		{
			nullptr, // Kinematic vs static doesn't exist
			nullptr, // Kinematic vs kinematic doesn't exist
			sSolvePositionConstraint<EMotionType::Kinematic, EMotionType::Dynamic>
		},
		{
			sSolvePositionConstraint<EMotionType::Dynamic, EMotionType::Static>,
			sSolvePositionConstraint<EMotionType::Dynamic, EMotionType::Kinematic>,
			sSolvePositionConstraint<EMotionType::Dynamic, EMotionType::Dynamic>
		}
	};

	if (inConstraintOffsetBegin >= inConstraintOffsetEnd)
		return false;

	bool any_impulse_applied = false;

	ContactConstraintBase *next_constraint = reinterpret_cast<ContactConstraintBase *>(mConstraints + *inConstraintOffsetBegin);
	for (const uint32 *next_constraint_offset = inConstraintOffsetBegin + 1; next_constraint != nullptr; ++next_constraint_offset)
	{
		ContactConstraintBase &constraint = *next_constraint;
		if (next_constraint_offset < inConstraintOffsetEnd)
		{
			next_constraint = reinterpret_cast<ContactConstraintBase *>(mConstraints + *next_constraint_offset);
			PrefetchL1(next_constraint);
		}
		else
			next_constraint = nullptr;

		// Fetch bodies
		Body &body1 = *constraint.mBody1;
		Body &body2 = *constraint.mBody2;

		// Dispatch to the correct templated form
		any_impulse_applied |= table[(int)body1.GetMotionType()][(int)body2.GetMotionType()](constraint, body1, body2, mPhysicsSettings, *mWriteCache);
	}

	return any_impulse_applied;
}

void ContactConstraintManager::RecycleConstraintBuffer()
{
	// Reset constraint array
	mNumConstraintsAndNextConstraintOffset = 0;

	// Store read / write cache
	mReadCache = &mCache[mCacheWriteIdx ^ 1];
	mWriteCache = &mCache[mCacheWriteIdx];
}

void ContactConstraintManager::FinishConstraintBuffer()
{
	// Free constraints buffer
	mUpdateContext->mTempAllocator->Free(mConstraintIdxToOffset, mMaxConstraints * sizeof(uint32));
	mConstraintIdxToOffset = nullptr;
	mUpdateContext->mTempAllocator->Free(mConstraints, mMaxConstraints * cMaxConstraintSize);
	mConstraints = nullptr;
	mNumConstraintsAndNextConstraintOffset = 0;

	// Reset update context
	mUpdateContext = nullptr;
	mReadCache = nullptr;
	mWriteCache = nullptr;
}

void ContactConstraintManager::SaveState(StateRecorder &inStream, const StateRecorderFilter *inFilter) const
{
	mCache[mCacheWriteIdx ^ 1].SaveState(inStream, inFilter);
}

bool ContactConstraintManager::RestoreState(StateRecorder &inStream, const StateRecorderFilter *inFilter)
{
	bool success = mCache[mCacheWriteIdx].RestoreState(mCache[mCacheWriteIdx ^ 1], inStream, inFilter);

	// If this is the last part, the cache is finalized
	if (inStream.IsLastPart())
	{
		mCacheWriteIdx ^= 1;
		mCache[mCacheWriteIdx].Clear();
	}

	return success;
}

JPH_NAMESPACE_END
