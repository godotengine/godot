// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2023 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodyUpdateContext.h>
#include <Jolt/ObjectStream/TypeDeclarations.h>
#include <Jolt/Core/StreamIn.h>
#include <Jolt/Core/StreamOut.h>
#include <Jolt/Core/QuickSort.h>
#include <Jolt/Core/UnorderedMap.h>
#include <Jolt/Core/UnorderedSet.h>
#include <Jolt/Core/BinaryHeap.h>

JPH_NAMESPACE_BEGIN

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::Vertex)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Vertex, mPosition)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Vertex, mVelocity)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Vertex, mInvMass)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::Face)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Face, mVertex)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Face, mMaterialIndex)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::Edge)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Edge, mVertex)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Edge, mRestLength)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Edge, mCompliance)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::RodStretchShear)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::RodStretchShear, mVertex)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::RodStretchShear, mLength)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::RodStretchShear, mInvMass)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::RodStretchShear, mCompliance)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::RodStretchShear, mBishop)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::RodBendTwist)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::RodBendTwist, mRod)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::RodBendTwist, mCompliance)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::RodBendTwist, mOmega0)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::DihedralBend)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::DihedralBend, mVertex)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::DihedralBend, mCompliance)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::DihedralBend, mInitialAngle)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::Volume)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Volume, mVertex)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Volume, mSixRestVolume)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Volume, mCompliance)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::InvBind)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::InvBind, mJointIndex)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::InvBind, mInvBind)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::SkinWeight)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::SkinWeight, mInvBindIndex)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::SkinWeight, mWeight)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::Skinned)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Skinned, mVertex)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Skinned, mWeights)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Skinned, mMaxDistance)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Skinned, mBackStopDistance)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::Skinned, mBackStopRadius)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings::LRA)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::LRA, mVertex)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings::LRA, mMaxDistance)
}

JPH_IMPLEMENT_SERIALIZABLE_NON_VIRTUAL(SoftBodySharedSettings)
{
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mVertices)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mFaces)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mEdgeConstraints)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mDihedralBendConstraints)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mVolumeConstraints)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mSkinnedConstraints)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mInvBindMatrices)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mLRAConstraints)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mRodStretchShearConstraints)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mRodBendTwistConstraints)
	JPH_ADD_ATTRIBUTE(SoftBodySharedSettings, mMaterials)
}

void SoftBodySharedSettings::CalculateClosestKinematic()
{
	// Check if we already calculated this
	if (!mClosestKinematic.empty())
		return;

	// Reserve output size
	mClosestKinematic.resize(mVertices.size());

	// Create a list of connected vertices
	Array<Array<uint32>> connectivity;
	connectivity.resize(mVertices.size());
	for (const Edge &e : mEdgeConstraints)
	{
		connectivity[e.mVertex[0]].push_back(e.mVertex[1]);
		connectivity[e.mVertex[1]].push_back(e.mVertex[0]);
	}
	for (const RodStretchShear &r : mRodStretchShearConstraints)
	{
		connectivity[r.mVertex[0]].push_back(r.mVertex[1]);
		connectivity[r.mVertex[1]].push_back(r.mVertex[0]);
	}

	// Use Dijkstra's algorithm to find the closest kinematic vertex for each vertex
	// See: https://en.wikipedia.org/wiki/Dijkstra's_algorithm
	//
	// An element in the open list
	struct Open
	{
		// Order so that we get the shortest distance first
		bool	operator < (const Open &inRHS) const
		{
			return mDistance > inRHS.mDistance;
		}

		uint32	mVertex;
		float	mDistance;
	};

	// Start with all kinematic elements
	Array<Open> to_visit;
	for (uint32 v = 0; v < mVertices.size(); ++v)
		if (mVertices[v].mInvMass == 0.0f)
		{
			mClosestKinematic[v].mVertex = v;
			mClosestKinematic[v].mHops = 0;
			mClosestKinematic[v].mDistance = 0.0f;
			to_visit.push_back({ v, 0.0f });
			BinaryHeapPush(to_visit.begin(), to_visit.end(), std::less<Open> { });
		}

	// Visit all vertices remembering the closest kinematic vertex and its distance
	JPH_IF_ENABLE_ASSERTS(float last_closest = 0.0f;)
	while (!to_visit.empty())
	{
		// Pop element from the open list
		BinaryHeapPop(to_visit.begin(), to_visit.end(), std::less<Open> { });
		Open current = to_visit.back();
		to_visit.pop_back();
		JPH_ASSERT(current.mDistance >= last_closest);
		JPH_IF_ENABLE_ASSERTS(last_closest = current.mDistance;)

		// Loop through all of its connected vertices
		for (uint32 v : connectivity[current.mVertex])
		{
			// Calculate distance from the current vertex to this target vertex and check if it is smaller
			float new_distance = current.mDistance + (Vec3(mVertices[v].mPosition) - Vec3(mVertices[current.mVertex].mPosition)).Length();
			if (new_distance < mClosestKinematic[v].mDistance)
			{
				// Remember new closest vertex
				mClosestKinematic[v].mVertex = mClosestKinematic[current.mVertex].mVertex;
				mClosestKinematic[v].mHops = mClosestKinematic[current.mVertex].mHops + 1;
				mClosestKinematic[v].mDistance = new_distance;
				to_visit.push_back({ v, new_distance });
				BinaryHeapPush(to_visit.begin(), to_visit.end(), std::less<Open> { });
			}
		}
	}
}

void SoftBodySharedSettings::CreateConstraints(const VertexAttributes *inVertexAttributes, uint inVertexAttributesLength, EBendType inBendType, float inAngleTolerance)
{
	struct EdgeHelper
	{
		uint32	mVertex[2];
		uint32	mEdgeIdx;
	};

	// Create list of all edges
	Array<EdgeHelper> edges;
	edges.reserve(mFaces.size() * 3);
	for (const Face &f : mFaces)
		for (int i = 0; i < 3; ++i)
		{
			uint32 v0 = f.mVertex[i];
			uint32 v1 = f.mVertex[(i + 1) % 3];

			EdgeHelper e;
			e.mVertex[0] = min(v0, v1);
			e.mVertex[1] = max(v0, v1);
			e.mEdgeIdx = uint32(&f - mFaces.data()) * 3 + i;
			edges.push_back(e);
		}

	// Sort the edges
	QuickSort(edges.begin(), edges.end(), [](const EdgeHelper &inLHS, const EdgeHelper &inRHS) { return inLHS.mVertex[0] < inRHS.mVertex[0] || (inLHS.mVertex[0] == inRHS.mVertex[0] && inLHS.mVertex[1] < inRHS.mVertex[1]); });

	// Only add edges if one of the vertices is movable
	auto add_edge = [this](uint32 inVtx1, uint32 inVtx2, float inCompliance1, float inCompliance2) {
		if ((mVertices[inVtx1].mInvMass > 0.0f || mVertices[inVtx2].mInvMass > 0.0f)
			&& inCompliance1 < FLT_MAX && inCompliance2 < FLT_MAX)
		{
			Edge temp_edge;
			temp_edge.mVertex[0] = inVtx1;
			temp_edge.mVertex[1] = inVtx2;
			temp_edge.mCompliance = 0.5f * (inCompliance1 + inCompliance2);
			temp_edge.mRestLength = (Vec3(mVertices[inVtx2].mPosition) - Vec3(mVertices[inVtx1].mPosition)).Length();
			JPH_ASSERT(temp_edge.mRestLength > 0.0f);
			mEdgeConstraints.push_back(temp_edge);
		}
	};

	// Helper function to get the attributes of a vertex
	auto attr = [inVertexAttributes, inVertexAttributesLength](uint32 inVertex) {
		return inVertexAttributes[min(inVertex, inVertexAttributesLength - 1)];
	};

	// Create the constraints
	float sq_sin_tolerance = Square(Sin(inAngleTolerance));
	float sq_cos_tolerance = Square(Cos(inAngleTolerance));
	mEdgeConstraints.clear();
	mEdgeConstraints.reserve(edges.size());
	for (Array<EdgeHelper>::size_type i = 0; i < edges.size(); ++i)
	{
		const EdgeHelper &e0 = edges[i];

		// Get attributes for the vertices of the edge
		const VertexAttributes &a0 = attr(e0.mVertex[0]);
		const VertexAttributes &a1 = attr(e0.mVertex[1]);

		// Flag that indicates if this edge is a shear edge (if 2 triangles form a quad-like shape and this edge is on the diagonal)
		bool is_shear = false;

		// Test if there are any shared edges
		for (Array<EdgeHelper>::size_type j = i + 1; j < edges.size(); ++j)
		{
			const EdgeHelper &e1 = edges[j];
			if (e0.mVertex[0] == e1.mVertex[0] && e0.mVertex[1] == e1.mVertex[1])
			{
				// Get opposing vertices
				const Face &f0 = mFaces[e0.mEdgeIdx / 3];
				const Face &f1 = mFaces[e1.mEdgeIdx / 3];
				uint32 vopposite0 = f0.mVertex[(e0.mEdgeIdx + 2) % 3];
				uint32 vopposite1 = f1.mVertex[(e1.mEdgeIdx + 2) % 3];
				const VertexAttributes &a_opposite0 = attr(vopposite0);
				const VertexAttributes &a_opposite1 = attr(vopposite1);

				// If the opposite vertices happen to be the same vertex then we have 2 triangles back to back and we skip creating shear / bend constraints
				if (vopposite0 == vopposite1)
					continue;

				// Faces should be roughly in a plane
				Vec3 n0 = (Vec3(mVertices[f0.mVertex[2]].mPosition) - Vec3(mVertices[f0.mVertex[0]].mPosition)).Cross(Vec3(mVertices[f0.mVertex[1]].mPosition) - Vec3(mVertices[f0.mVertex[0]].mPosition));
				Vec3 n1 = (Vec3(mVertices[f1.mVertex[2]].mPosition) - Vec3(mVertices[f1.mVertex[0]].mPosition)).Cross(Vec3(mVertices[f1.mVertex[1]].mPosition) - Vec3(mVertices[f1.mVertex[0]].mPosition));
				float n0_dot_n1 = n0.Dot(n1);
				if (n0_dot_n1 > 0.0f
					&& Square(n0_dot_n1) > sq_cos_tolerance * n0.LengthSq() * n1.LengthSq())
				{
					// Faces should approximately form a quad
					Vec3 e0_dir = Vec3(mVertices[vopposite0].mPosition) - Vec3(mVertices[e0.mVertex[0]].mPosition);
					Vec3 e1_dir = Vec3(mVertices[vopposite1].mPosition) - Vec3(mVertices[e0.mVertex[0]].mPosition);
					if (Square(e0_dir.Dot(e1_dir)) < sq_sin_tolerance * e0_dir.LengthSq() * e1_dir.LengthSq())
					{
						// Shear constraint
						add_edge(vopposite0, vopposite1, a_opposite0.mShearCompliance, a_opposite1.mShearCompliance);
						is_shear = true;
					}
				}

				// Bend constraint
				switch (inBendType)
				{
				case EBendType::None:
					// Do nothing
					break;

				case EBendType::Distance:
					// Create an edge constraint to represent the bend constraint
					// Use the bend compliance of the shared edge
					if (!is_shear)
						add_edge(vopposite0, vopposite1, a0.mBendCompliance, a1.mBendCompliance);
					break;

				case EBendType::Dihedral:
					// Test if both opposite vertices are free to move
					if ((mVertices[vopposite0].mInvMass > 0.0f || mVertices[vopposite1].mInvMass > 0.0f)
						&& a0.mBendCompliance < FLT_MAX && a1.mBendCompliance < FLT_MAX)
					{
						// Create a bend constraint
						// Use the bend compliance of the shared edge
						mDihedralBendConstraints.emplace_back(e0.mVertex[0], e0.mVertex[1], vopposite0, vopposite1, 0.5f * (a0.mBendCompliance + a1.mBendCompliance));
					}
					break;
				}
			}
			else
			{
				// Start iterating from the first non-shared edge
				i = j - 1;
				break;
			}
		}

		// Create a edge constraint for the current edge
		add_edge(e0.mVertex[0], e0.mVertex[1], is_shear? a0.mShearCompliance : a0.mCompliance, is_shear? a1.mShearCompliance : a1.mCompliance);
	}
	mEdgeConstraints.shrink_to_fit();

	// Calculate the initial angle for all bend constraints
	CalculateBendConstraintConstants();

	// Check if any vertices have LRA constraints
	bool has_lra_constraints = false;
	for (const VertexAttributes *va = inVertexAttributes; va < inVertexAttributes + inVertexAttributesLength; ++va)
		if (va->mLRAType != ELRAType::None)
		{
			has_lra_constraints = true;
			break;
		}
	if (has_lra_constraints)
	{
		// Ensure we have calculated the closest kinematic vertex for each vertex
		CalculateClosestKinematic();

		// Find non-kinematic vertices
		for (uint32 v = 0; v < (uint32)mVertices.size(); ++v)
			if (mVertices[v].mInvMass > 0.0f)
			{
				// Check if a closest vertex was found
				uint32 closest = mClosestKinematic[v].mVertex;
				if (closest != 0xffffffff)
				{
					// Check which LRA constraint to create
					const VertexAttributes &va = attr(v);
					switch (va.mLRAType)
					{
					case ELRAType::None:
						break;

					case ELRAType::EuclideanDistance:
						mLRAConstraints.emplace_back(closest, v, va.mLRAMaxDistanceMultiplier * (Vec3(mVertices[closest].mPosition) - Vec3(mVertices[v].mPosition)).Length());
						break;

					case ELRAType::GeodesicDistance:
						mLRAConstraints.emplace_back(closest, v, va.mLRAMaxDistanceMultiplier * mClosestKinematic[v].mDistance);
						break;
					}
				}
			}
	}
}

void SoftBodySharedSettings::CalculateEdgeLengths()
{
	for (Edge &e : mEdgeConstraints)
	{
		JPH_ASSERT(e.mVertex[0] != e.mVertex[1], "Edges need to connect 2 different vertices");
		e.mRestLength = (Vec3(mVertices[e.mVertex[1]].mPosition) - Vec3(mVertices[e.mVertex[0]].mPosition)).Length();
		JPH_ASSERT(e.mRestLength > 0.0f);
	}
}

void SoftBodySharedSettings::CalculateRodProperties()
{
	// Mark connections through bend twist constraints
	Array<Array<uint32>> connections;
	connections.resize(mRodStretchShearConstraints.size());
	for (const RodBendTwist &c : mRodBendTwistConstraints)
	{
		JPH_ASSERT(c.mRod[0] != c.mRod[1], "A bend twist constraint needs to be attached to different rods");
		connections[c.mRod[1]].push_back(c.mRod[0]);
		connections[c.mRod[0]].push_back(c.mRod[1]);
	}

	// Now calculate the Bishop frames for all rods
	struct Entry
	{
		uint32	mFrom;	// Rod we're coming from
		uint32	mTo;	// Rod we're going to
	};
	Array<Entry> stack;
	stack.reserve(mRodStretchShearConstraints.size());
	for (uint32 r0_idx = 0; r0_idx < mRodStretchShearConstraints.size(); ++r0_idx)
	{
		RodStretchShear &r0 = mRodStretchShearConstraints[r0_idx];

		// Do not calculate a 2nd time
		if (r0.mBishop == Quat::sZero())
		{
			// Calculate the frame for this rod
			{
				Vec3 tangent = Vec3(mVertices[r0.mVertex[1]].mPosition) - Vec3(mVertices[r0.mVertex[0]].mPosition);
				r0.mLength = tangent.Length();
				JPH_ASSERT(r0.mLength > 0.0f, "Rods of zero length are not supported!");
				tangent /= r0.mLength;
				Vec3 normal = tangent.GetNormalizedPerpendicular();
				Vec3 binormal = tangent.Cross(normal);
				r0.mBishop = Mat44(Vec4(normal, 0), Vec4(binormal, 0), Vec4(tangent, 0), Vec4(0, 0, 0, 1)).GetQuaternion().Normalized();
			}

			// Add connected rods to the stack if they haven't been calculated yet
			for (uint32 r1_idx : connections[r0_idx])
				if (mRodStretchShearConstraints[r1_idx].mBishop == Quat::sZero())
					stack.push_back({ r0_idx, r1_idx });

			// Now connect the bishop frame for all connected rods on the stack
			// This follows the procedure outlined in "Discrete Elastic Rods" - M. Bergou et al.
			// See: https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf
			while (!stack.empty())
			{
				uint32 r1_idx = stack.back().mFrom;
				uint32 r2_idx = stack.back().mTo;
				stack.pop_back();

				const RodStretchShear &r1 = mRodStretchShearConstraints[r1_idx];
				RodStretchShear &r2 = mRodStretchShearConstraints[r2_idx];

				// Get the normal and tangent of the first rod's Bishop frame (that was already calculated)
				Mat44 r1_frame = Mat44::sRotation(r1.mBishop);
				Vec3 tangent1 = r1_frame.GetAxisZ();
				Vec3 normal1 = r1_frame.GetAxisX();

				// Calculate the Bishop frame for the 2nd rod
				Vec3 tangent2 = Vec3(mVertices[r2.mVertex[1]].mPosition) - Vec3(mVertices[r2.mVertex[0]].mPosition);
				if (tangent1.Dot(tangent2) < 0.0f)
				{
					// Edge is oriented in the opposite direction of the previous edge, flip it
					std::swap(r2.mVertex[0], r2.mVertex[1]);
					tangent2 = -tangent2;
				}
				r2.mLength = tangent2.Length();
				JPH_ASSERT(r2.mLength > 0.0f, "Rods of zero length are not supported!");
				tangent2 /= r2.mLength;
				Vec3 t1_cross_t2 = tangent1.Cross(tangent2);
				float sin_angle = t1_cross_t2.Length();
				Vec3 normal2 = normal1;
				if (sin_angle > 1.0e-6f)
				{
					t1_cross_t2 /= sin_angle;
					normal2 = Quat::sRotation(t1_cross_t2, ASin(sin_angle)) * normal2;
				}
				Vec3 binormal2 = tangent2.Cross(normal2);
				r2.mBishop = Mat44(Vec4(normal2, 0), Vec4(binormal2, 0), Vec4(tangent2, 0), Vec4(0, 0, 0, 1)).GetQuaternion().Normalized();

				// Add connected rods to the stack if they haven't been calculated yet
				for (uint32 r3_idx : connections[r2_idx])
					if (mRodStretchShearConstraints[r3_idx].mBishop == Quat::sZero())
						stack.push_back({ r2_idx, r3_idx });
			}
		}
	}

	// Calculate inverse mass for all rods by taking the minimum inverse mass (aka the heaviest vertex) of both vertices
	for (RodStretchShear &r : mRodStretchShearConstraints)
	{
		JPH_ASSERT(r.mVertex[0] != r.mVertex[1], "A rod stretch shear constraint requires two different vertices");
		r.mInvMass = min(mVertices[r.mVertex[0]].mInvMass, mVertices[r.mVertex[1]].mInvMass);
	}

	// Calculate the initial rotation between the rods
	for (RodBendTwist &r : mRodBendTwistConstraints)
		r.mOmega0 = (mRodStretchShearConstraints[r.mRod[0]].mBishop.Conjugated() * mRodStretchShearConstraints[r.mRod[1]].mBishop).Normalized();
}

void SoftBodySharedSettings::CalculateLRALengths(float inMaxDistanceMultiplier)
{
	for (LRA &l : mLRAConstraints)
	{
		JPH_ASSERT(l.mVertex[0] != l.mVertex[1], "LRA constraints need to connect 2 different vertices");
		l.mMaxDistance = inMaxDistanceMultiplier * (Vec3(mVertices[l.mVertex[1]].mPosition) - Vec3(mVertices[l.mVertex[0]].mPosition)).Length();
		JPH_ASSERT(l.mMaxDistance > 0.0f);
	}
}

void SoftBodySharedSettings::CalculateBendConstraintConstants()
{
	for (DihedralBend &b : mDihedralBendConstraints)
	{
		JPH_ASSERT(b.mVertex[0] != b.mVertex[1] && b.mVertex[0] != b.mVertex[2] && b.mVertex[0] != b.mVertex[3]
			&& b.mVertex[1] != b.mVertex[2] && b.mVertex[1] != b.mVertex[3]
			&& b.mVertex[2] != b.mVertex[3], "Bend constraints need 4 different vertices");

		// Get positions
		Vec3 x0 = Vec3(mVertices[b.mVertex[0]].mPosition);
		Vec3 x1 = Vec3(mVertices[b.mVertex[1]].mPosition);
		Vec3 x2 = Vec3(mVertices[b.mVertex[2]].mPosition);
		Vec3 x3 = Vec3(mVertices[b.mVertex[3]].mPosition);

		/*
		   x2
		e1/  \e3
		 /    \
		x0----x1
		 \ e0 /
		e2\  /e4
		   x3
		*/

		// Calculate edges
		Vec3 e0 = x1 - x0;
		Vec3 e1 = x2 - x0;
		Vec3 e2 = x3 - x0;

		// Normals of both triangles
		Vec3 n1 = e0.Cross(e1);
		Vec3 n2 = e2.Cross(e0);
		float denom = sqrt(n1.LengthSq() * n2.LengthSq());
		if (denom < 1.0e-12f)
			b.mInitialAngle = 0.0f;
		else
		{
			float sign = Sign(n2.Cross(n1).Dot(e0));
			b.mInitialAngle = sign * ACosApproximate(n1.Dot(n2) / denom); // Runtime uses the approximation too
		}
	}
}

void SoftBodySharedSettings::CalculateVolumeConstraintVolumes()
{
	for (Volume &v : mVolumeConstraints)
	{
		JPH_ASSERT(v.mVertex[0] != v.mVertex[1] && v.mVertex[0] != v.mVertex[2] && v.mVertex[0] != v.mVertex[3]
			&& v.mVertex[1] != v.mVertex[2] && v.mVertex[1] != v.mVertex[3]
			&& v.mVertex[2] != v.mVertex[3], "Volume constraints need 4 different vertices");

		Vec3 x1(mVertices[v.mVertex[0]].mPosition);
		Vec3 x2(mVertices[v.mVertex[1]].mPosition);
		Vec3 x3(mVertices[v.mVertex[2]].mPosition);
		Vec3 x4(mVertices[v.mVertex[3]].mPosition);

		Vec3 x1x2 = x2 - x1;
		Vec3 x1x3 = x3 - x1;
		Vec3 x1x4 = x4 - x1;

		v.mSixRestVolume = abs(x1x2.Cross(x1x3).Dot(x1x4));
	}
}

void SoftBodySharedSettings::CalculateSkinnedConstraintNormals()
{
	// Clear any previous results
	mSkinnedConstraintNormals.clear();

	// If there are no skinned constraints, we're done
	if (mSkinnedConstraints.empty())
		return;

	// First collect all vertices that are skinned
	using VertexIndexSet = UnorderedSet<uint32>;
	VertexIndexSet skinned_vertices;
	skinned_vertices.reserve(VertexIndexSet::size_type(mSkinnedConstraints.size()));
	for (const Skinned &s : mSkinnedConstraints)
		skinned_vertices.insert(s.mVertex);

	// Now collect all faces that connect only to skinned vertices
	using ConnectedFacesMap = UnorderedMap<uint32, VertexIndexSet>;
	ConnectedFacesMap connected_faces;
	connected_faces.reserve(ConnectedFacesMap::size_type(mVertices.size()));
	for (const Face &f : mFaces)
	{
		// Must connect to only skinned vertices
		bool valid = true;
		for (uint32 v : f.mVertex)
			valid &= skinned_vertices.find(v) != skinned_vertices.end();
		if (!valid)
			continue;

		// Store faces that connect to vertices
		for (uint32 v : f.mVertex)
			connected_faces[v].insert(uint32(&f - mFaces.data()));
	}

	// Populate the list of connecting faces per skinned vertex
	mSkinnedConstraintNormals.reserve(mFaces.size());
	for (Skinned &s : mSkinnedConstraints)
	{
		uint32 start = uint32(mSkinnedConstraintNormals.size());
		JPH_ASSERT((start >> 24) == 0);
		ConnectedFacesMap::const_iterator connected_faces_it = connected_faces.find(s.mVertex);
		if (connected_faces_it != connected_faces.cend())
		{
			const VertexIndexSet &faces = connected_faces_it->second;
			uint32 num = uint32(faces.size());
			JPH_ASSERT(num < 256);
			mSkinnedConstraintNormals.insert(mSkinnedConstraintNormals.end(), faces.begin(), faces.end());
			QuickSort(mSkinnedConstraintNormals.begin() + start, mSkinnedConstraintNormals.begin() + start + num);
			s.mNormalInfo = start + (num << 24);
		}
		else
			s.mNormalInfo = 0;
	}
	mSkinnedConstraintNormals.shrink_to_fit();
}

void SoftBodySharedSettings::Optimize(OptimizationResults &outResults)
{
	// Clear any previous results
	mUpdateGroups.clear();

	// Create a list of connected vertices
	struct Connection
	{
		uint32	mVertex;
		uint32	mCount;
	};
	Array<Array<Connection>> connectivity;
	connectivity.resize(mVertices.size());
	auto add_connection = [&connectivity](uint inV1, uint inV2) {
			for (int i = 0; i < 2; ++i)
			{
				bool found = false;
				for (Connection &c : connectivity[inV1])
					if (c.mVertex == inV2)
					{
						c.mCount++;
						found = true;
						break;
					}
				if (!found)
					connectivity[inV1].push_back({ inV2, 1 });

				std::swap(inV1, inV2);
			}
		};
	for (const Edge &c : mEdgeConstraints)
		add_connection(c.mVertex[0], c.mVertex[1]);
	for (const LRA &c : mLRAConstraints)
		add_connection(c.mVertex[0], c.mVertex[1]);
	for (const RodStretchShear &c : mRodStretchShearConstraints)
		add_connection(c.mVertex[0], c.mVertex[1]);
	for (const RodBendTwist &c : mRodBendTwistConstraints)
	{
		add_connection(mRodStretchShearConstraints[c.mRod[0]].mVertex[0], mRodStretchShearConstraints[c.mRod[1]].mVertex[0]);
		add_connection(mRodStretchShearConstraints[c.mRod[0]].mVertex[1], mRodStretchShearConstraints[c.mRod[1]].mVertex[0]);
		add_connection(mRodStretchShearConstraints[c.mRod[0]].mVertex[0], mRodStretchShearConstraints[c.mRod[1]].mVertex[1]);
		add_connection(mRodStretchShearConstraints[c.mRod[0]].mVertex[1], mRodStretchShearConstraints[c.mRod[1]].mVertex[1]);
	}
	for (const DihedralBend &c : mDihedralBendConstraints)
	{
		add_connection(c.mVertex[0], c.mVertex[1]);
		add_connection(c.mVertex[0], c.mVertex[2]);
		add_connection(c.mVertex[0], c.mVertex[3]);
		add_connection(c.mVertex[1], c.mVertex[2]);
		add_connection(c.mVertex[1], c.mVertex[3]);
		add_connection(c.mVertex[2], c.mVertex[3]);
	}
	for (const Volume &c : mVolumeConstraints)
	{
		add_connection(c.mVertex[0], c.mVertex[1]);
		add_connection(c.mVertex[0], c.mVertex[2]);
		add_connection(c.mVertex[0], c.mVertex[3]);
		add_connection(c.mVertex[1], c.mVertex[2]);
		add_connection(c.mVertex[1], c.mVertex[3]);
		add_connection(c.mVertex[2], c.mVertex[3]);
	}
	// Skinned constraints only update 1 vertex, so we don't need special logic here

	// Maps each of the vertices to a group index
	Array<int> group_idx;
	group_idx.resize(mVertices.size(), -1);

	// Which group we are currently filling and its vertices
	int current_group_idx = 0;
	Array<uint> current_group;

	// Start greedy algorithm to group vertices
	for (;;)
	{
		// Find the bounding box of the ungrouped vertices
		AABox bounds;
		for (uint i = 0; i < (uint)mVertices.size(); ++i)
			if (group_idx[i] == -1)
				bounds.Encapsulate(Vec3(mVertices[i].mPosition));

		// If the bounds are invalid, it means that there were no ungrouped vertices
		if (!bounds.IsValid())
			break;

		// Determine longest and shortest axis
		Vec3 bounds_size = bounds.GetSize();
		uint max_axis = bounds_size.GetHighestComponentIndex();
		uint min_axis = bounds_size.GetLowestComponentIndex();
		if (min_axis == max_axis)
			min_axis = (min_axis + 1) % 3;
		uint mid_axis = 3 - min_axis - max_axis;

		// Find the vertex that has the lowest value on the axis with the largest extent
		uint current_vertex = UINT_MAX;
		Float3 current_vertex_position { FLT_MAX, FLT_MAX, FLT_MAX };
		for (uint i = 0; i < (uint)mVertices.size(); ++i)
			if (group_idx[i] == -1)
			{
				const Float3 &vertex_position = mVertices[i].mPosition;
				float max_axis_value = vertex_position[max_axis];
				float mid_axis_value = vertex_position[mid_axis];
				float min_axis_value = vertex_position[min_axis];

				if (max_axis_value < current_vertex_position[max_axis]
					|| (max_axis_value == current_vertex_position[max_axis]
						&& (mid_axis_value < current_vertex_position[mid_axis]
							|| (mid_axis_value == current_vertex_position[mid_axis]
								&& min_axis_value < current_vertex_position[min_axis]))))
				{
					current_vertex_position = mVertices[i].mPosition;
					current_vertex = i;
				}
			}
		if (current_vertex == UINT_MAX)
			break;

		// Initialize the current group with 1 vertex
		current_group.push_back(current_vertex);
		group_idx[current_vertex] = current_group_idx;

		// Fill up the group
		for (;;)
		{
			// Find the vertex that is most connected to the current group
			uint best_vertex = UINT_MAX;
			uint best_num_connections = 0;
			float best_dist_sq = FLT_MAX;
			for (uint i = 0; i < (uint)current_group.size(); ++i) // For all vertices in the current group
				for (const Connection &c : connectivity[current_group[i]]) // For all connections to other vertices
				{
					uint v = c.mVertex;
					if (group_idx[v] == -1) // Ungrouped vertices only
					{
						// Count the number of connections to this group
						uint num_connections = 0;
						for (const Connection &v2 : connectivity[v])
							if (group_idx[v2.mVertex] == current_group_idx)
								num_connections += v2.mCount;

						// Calculate distance to group centroid
						float dist_sq = (Vec3(mVertices[v].mPosition) - Vec3(mVertices[current_group.front()].mPosition)).LengthSq();

						if (best_vertex == UINT_MAX
							|| num_connections > best_num_connections
							|| (num_connections == best_num_connections && dist_sq < best_dist_sq))
						{
							best_vertex = v;
							best_num_connections = num_connections;
							best_dist_sq = dist_sq;
						}
					}
				}

			// Add the best vertex to the current group
			if (best_vertex != UINT_MAX)
			{
				current_group.push_back(best_vertex);
				group_idx[best_vertex] = current_group_idx;
			}

			// Create a new group?
			if (current_group.size() >= SoftBodyUpdateContext::cVertexConstraintBatch // If full, yes
				|| (current_group.size() > SoftBodyUpdateContext::cVertexConstraintBatch / 2 && best_vertex == UINT_MAX)) // If half full and we found no connected vertex, yes
			{
				current_group.clear();
				current_group_idx++;
				break;
			}

			// If we didn't find a connected vertex, we need to find a new starting vertex
			if (best_vertex == UINT_MAX)
				break;
		}
	}

	// If the last group is more than half full, we'll keep it as a separate group, otherwise we merge it with the 'non parallel' group
	if (current_group.size() > SoftBodyUpdateContext::cVertexConstraintBatch / 2)
		++current_group_idx;

	// We no longer need the current group array, free the memory
	current_group.clear();
	current_group.shrink_to_fit();

	// We're done with the connectivity list, free the memory
	connectivity.clear();
	connectivity.shrink_to_fit();

	// Assign the constraints to their groups
	struct Group
	{
		uint			GetSize() const
		{
			return (uint)mEdgeConstraints.size() + (uint)mLRAConstraints.size() + (uint)mRodStretchShearConstraints.size() + (uint)mRodBendTwistConstraints.size() + (uint)mDihedralBendConstraints.size() + (uint)mVolumeConstraints.size() + (uint)mSkinnedConstraints.size();
		}

		Array<uint>		mEdgeConstraints;
		Array<uint>		mLRAConstraints;
		Array<uint>		mRodStretchShearConstraints;
		Array<uint>		mRodBendTwistConstraints;
		Array<uint>		mDihedralBendConstraints;
		Array<uint>		mVolumeConstraints;
		Array<uint>		mSkinnedConstraints;
	};
	Array<Group> groups;
	groups.resize(current_group_idx + 1); // + non parallel group
	for (const Edge &e : mEdgeConstraints)
	{
		int g1 = group_idx[e.mVertex[0]];
		int g2 = group_idx[e.mVertex[1]];
		JPH_ASSERT(g1 >= 0 && g2 >= 0);
		if (g1 == g2) // In the same group
			groups[g1].mEdgeConstraints.push_back(uint(&e - mEdgeConstraints.data()));
		else // In different groups -> parallel group
			groups.back().mEdgeConstraints.push_back(uint(&e - mEdgeConstraints.data()));
	}
	for (const LRA &l : mLRAConstraints)
	{
		int g1 = group_idx[l.mVertex[0]];
		int g2 = group_idx[l.mVertex[1]];
		JPH_ASSERT(g1 >= 0 && g2 >= 0);
		if (g1 == g2) // In the same group
			groups[g1].mLRAConstraints.push_back(uint(&l - mLRAConstraints.data()));
		else // In different groups -> parallel group
			groups.back().mLRAConstraints.push_back(uint(&l - mLRAConstraints.data()));
	}
	for (const RodStretchShear &r : mRodStretchShearConstraints)
	{
		int g1 = group_idx[r.mVertex[0]];
		int g2 = group_idx[r.mVertex[1]];
		JPH_ASSERT(g1 >= 0 && g2 >= 0);
		if (g1 == g2) // In the same group
			groups[g1].mRodStretchShearConstraints.push_back(uint(&r - mRodStretchShearConstraints.data()));
		else // In different groups -> parallel group
			groups.back().mRodStretchShearConstraints.push_back(uint(&r - mRodStretchShearConstraints.data()));
	}
	for (const RodBendTwist &r : mRodBendTwistConstraints)
	{
		int g1 = group_idx[mRodStretchShearConstraints[r.mRod[0]].mVertex[0]];
		int g2 = group_idx[mRodStretchShearConstraints[r.mRod[0]].mVertex[1]];
		int g3 = group_idx[mRodStretchShearConstraints[r.mRod[1]].mVertex[0]];
		int g4 = group_idx[mRodStretchShearConstraints[r.mRod[1]].mVertex[1]];
		JPH_ASSERT(g1 >= 0 && g2 >= 0 && g3 >= 0 && g4 >= 0);
		if (g1 == g2 && g1 == g3 && g1 == g4) // In the same group
			groups[g1].mRodBendTwistConstraints.push_back(uint(&r - mRodBendTwistConstraints.data()));
		else // In different groups -> parallel group
			groups.back().mRodBendTwistConstraints.push_back(uint(&r - mRodBendTwistConstraints.data()));
	}
	for (const DihedralBend &d : mDihedralBendConstraints)
	{
		int g1 = group_idx[d.mVertex[0]];
		int g2 = group_idx[d.mVertex[1]];
		int g3 = group_idx[d.mVertex[2]];
		int g4 = group_idx[d.mVertex[3]];
		JPH_ASSERT(g1 >= 0 && g2 >= 0 && g3 >= 0 && g4 >= 0);
		if (g1 == g2 && g1 == g3 && g1 == g4) // In the same group
			groups[g1].mDihedralBendConstraints.push_back(uint(&d - mDihedralBendConstraints.data()));
		else // In different groups -> parallel group
			groups.back().mDihedralBendConstraints.push_back(uint(&d - mDihedralBendConstraints.data()));
	}
	for (const Volume &v : mVolumeConstraints)
	{
		int g1 = group_idx[v.mVertex[0]];
		int g2 = group_idx[v.mVertex[1]];
		int g3 = group_idx[v.mVertex[2]];
		int g4 = group_idx[v.mVertex[3]];
		JPH_ASSERT(g1 >= 0 && g2 >= 0 && g3 >= 0 && g4 >= 0);
		if (g1 == g2 && g1 == g3 && g1 == g4) // In the same group
			groups[g1].mVolumeConstraints.push_back(uint(&v - mVolumeConstraints.data()));
		else // In different groups -> parallel group
			groups.back().mVolumeConstraints.push_back(uint(&v - mVolumeConstraints.data()));
	}
	for (const Skinned &s : mSkinnedConstraints)
	{
		int g1 = group_idx[s.mVertex];
		JPH_ASSERT(g1 >= 0);
		groups[g1].mSkinnedConstraints.push_back(uint(&s - mSkinnedConstraints.data()));
	}

	// Sort the parallel groups from big to small (this means the big groups will be scheduled first and have more time to complete)
	QuickSort(groups.begin(), groups.end() - 1, [](const Group &inLHS, const Group &inRHS) { return inLHS.GetSize() > inRHS.GetSize(); });

	// Make sure we know the closest kinematic vertex so we can sort
	CalculateClosestKinematic();

	// Sort within each group
	for (Group &group : groups)
	{
		// Sort the edge constraints
		QuickSort(group.mEdgeConstraints.begin(), group.mEdgeConstraints.end(), [this](uint inLHS, uint inRHS)
			{
				const Edge &e1 = mEdgeConstraints[inLHS];
				const Edge &e2 = mEdgeConstraints[inRHS];

				// First sort so that the edge with the smallest distance to a kinematic vertex comes first
				float d1 = min(mClosestKinematic[e1.mVertex[0]].mDistance, mClosestKinematic[e1.mVertex[1]].mDistance);
				float d2 = min(mClosestKinematic[e2.mVertex[0]].mDistance, mClosestKinematic[e2.mVertex[1]].mDistance);
				if (d1 != d2)
					return d1 < d2;

				// Order the edges so that the ones with the smallest index go first (hoping to get better cache locality when we process the edges).
				// Note we could also re-order the vertices but that would be much more of a burden to the end user
				uint32 m1 = e1.GetMinVertexIndex();
				uint32 m2 = e2.GetMinVertexIndex();
				if (m1 != m2)
					return m1 < m2;

				return inLHS < inRHS;
			});

		// Sort the LRA constraints
		QuickSort(group.mLRAConstraints.begin(), group.mLRAConstraints.end(), [this](uint inLHS, uint inRHS)
			{
				const LRA &l1 = mLRAConstraints[inLHS];
				const LRA &l2 = mLRAConstraints[inRHS];

				// First sort so that the longest constraint comes first (meaning the shortest constraint has the most influence on the end result)
				// Most of the time there will be a single LRA constraint per vertex and since the LRA constraint only modifies a single vertex,
				// updating one constraint will not violate another constraint.
				if (l1.mMaxDistance != l2.mMaxDistance)
					return l1.mMaxDistance > l2.mMaxDistance;

				// Order constraints so that the ones with the smallest index go first
				uint32 m1 = l1.GetMinVertexIndex();
				uint32 m2 = l2.GetMinVertexIndex();
				if (m1 != m2)
					return m1 < m2;

				return inLHS < inRHS;
			});

		// Sort the rod stretch shear constraints
		QuickSort(group.mRodStretchShearConstraints.begin(), group.mRodStretchShearConstraints.end(), [this](uint inLHS, uint inRHS)
			{
				const RodStretchShear &r1 = mRodStretchShearConstraints[inLHS];
				const RodStretchShear &r2 = mRodStretchShearConstraints[inRHS];

				// First sort so that the rod with the smallest distance to a kinematic vertex comes first
				float d1 = min(mClosestKinematic[r1.mVertex[0]].mDistance, mClosestKinematic[r1.mVertex[1]].mDistance);
				float d2 = min(mClosestKinematic[r2.mVertex[0]].mDistance, mClosestKinematic[r2.mVertex[1]].mDistance);
				if (d1 != d2)
					return d1 < d2;

				// Then sort on the rod that connects to the smallest kinematic vertex
				uint32 m1 = min(mClosestKinematic[r1.mVertex[0]].mVertex, mClosestKinematic[r1.mVertex[1]].mVertex);
				uint32 m2 = min(mClosestKinematic[r2.mVertex[0]].mVertex, mClosestKinematic[r2.mVertex[1]].mVertex);
				if (m1 != m2)
					return m1 < m2;

				// Order the rods so that the ones with the smallest index go first (hoping to get better cache locality when we process the rods).
				m1 = r1.GetMinVertexIndex();
				m2 = r2.GetMinVertexIndex();
				if (m1 != m2)
					return m1 < m2;

				return inLHS < inRHS;
			});

		// Sort the rod bend twist constraints
		QuickSort(group.mRodBendTwistConstraints.begin(), group.mRodBendTwistConstraints.end(), [this](uint inLHS, uint inRHS)
			{
				const RodBendTwist &b1 = mRodBendTwistConstraints[inLHS];
				const RodStretchShear &b1_r1 = mRodStretchShearConstraints[b1.mRod[0]];
				const RodStretchShear &b1_r2 = mRodStretchShearConstraints[b1.mRod[1]];

				const RodBendTwist &b2 = mRodBendTwistConstraints[inRHS];
				const RodStretchShear &b2_r1 = mRodStretchShearConstraints[b2.mRod[0]];
				const RodStretchShear &b2_r2 = mRodStretchShearConstraints[b2.mRod[1]];

				// First sort so that the rod with the smallest number of hops to a kinematic vertex comes first.
				// Note that we don't use distance because of the bilateral interleaving below.
				uint32 m1 = min(
							min(mClosestKinematic[b1_r1.mVertex[0]].mHops, mClosestKinematic[b1_r1.mVertex[1]].mHops),
							min(mClosestKinematic[b1_r2.mVertex[0]].mHops, mClosestKinematic[b1_r2.mVertex[1]].mHops));
				uint32 m2 = min(
							min(mClosestKinematic[b2_r1.mVertex[0]].mHops, mClosestKinematic[b2_r1.mVertex[1]].mHops),
							min(mClosestKinematic[b2_r2.mVertex[0]].mHops, mClosestKinematic[b2_r2.mVertex[1]].mHops));
				if (m1 != m2)
					return m1 < m2;

				// Then sort on the rod that connects to the kinematic vertex with lowest index.
				// This ensures that we consistently order the rods that are attached to other kinematic constraints.
				// Again, this helps bilateral interleaving below.
				m1 = min(
							min(mClosestKinematic[b1_r1.mVertex[0]].mVertex, mClosestKinematic[b1_r1.mVertex[1]].mVertex),
							min(mClosestKinematic[b1_r2.mVertex[0]].mVertex, mClosestKinematic[b1_r2.mVertex[1]].mVertex));
				m2 = min(
							min(mClosestKinematic[b2_r1.mVertex[0]].mVertex, mClosestKinematic[b2_r1.mVertex[1]].mVertex),
							min(mClosestKinematic[b2_r2.mVertex[0]].mVertex, mClosestKinematic[b2_r2.mVertex[1]].mVertex));
				if (m1 != m2)
					return m1 < m2;

				// Finally order so that the smallest vertex index goes first
				m1 = min(b1_r1.GetMinVertexIndex(), b1_r2.GetMinVertexIndex());
				m2 = min(b2_r1.GetMinVertexIndex(), b2_r2.GetMinVertexIndex());
				if (m1 != m2)
					return m1 < m2;

				return inLHS < inRHS;
			});

		// Bilateral interleaving, see figure 4 of "Position and Orientation Based Cosserat Rods" - Kugelstadt and Schoemer - SIGGRAPH 2016
		// Keeping the twist constraints sorted often results in an unstable simulation
		for (Array<uint>::size_type i = 1, s = group.mRodBendTwistConstraints.size(), s2 = s >> 1; i < s2; i += 2)
			std::swap(group.mRodBendTwistConstraints[i], group.mRodBendTwistConstraints[s - i]);

		// Sort the dihedral bend constraints
		QuickSort(group.mDihedralBendConstraints.begin(), group.mDihedralBendConstraints.end(), [this](uint inLHS, uint inRHS)
		{
			const DihedralBend &b1 = mDihedralBendConstraints[inLHS];
			const DihedralBend &b2 = mDihedralBendConstraints[inRHS];

			// First sort so that the constraint with the smallest distance to a kinematic vertex comes first
			float d1 = min(
						min(mClosestKinematic[b1.mVertex[0]].mDistance, mClosestKinematic[b1.mVertex[1]].mDistance),
						min(mClosestKinematic[b1.mVertex[2]].mDistance, mClosestKinematic[b1.mVertex[3]].mDistance));
			float d2 = min(
						min(mClosestKinematic[b2.mVertex[0]].mDistance, mClosestKinematic[b2.mVertex[1]].mDistance),
						min(mClosestKinematic[b2.mVertex[2]].mDistance, mClosestKinematic[b2.mVertex[3]].mDistance));
			if (d1 != d2)
				return d1 < d2;

			// Finally order so that the smallest vertex index goes first
			uint32 m1 = b1.GetMinVertexIndex();
			uint32 m2 = b2.GetMinVertexIndex();
			if (m1 != m2)
				return m1 < m2;

			return inLHS < inRHS;
		});

		// Sort the volume constraints
		QuickSort(group.mVolumeConstraints.begin(), group.mVolumeConstraints.end(), [this](uint inLHS, uint inRHS)
		{
			const Volume &v1 = mVolumeConstraints[inLHS];
			const Volume &v2 = mVolumeConstraints[inRHS];

			// First sort so that the constraint with the smallest distance to a kinematic vertex comes first
			float d1 = min(
						min(mClosestKinematic[v1.mVertex[0]].mDistance, mClosestKinematic[v1.mVertex[1]].mDistance),
						min(mClosestKinematic[v1.mVertex[2]].mDistance, mClosestKinematic[v1.mVertex[3]].mDistance));
			float d2 = min(
						min(mClosestKinematic[v2.mVertex[0]].mDistance, mClosestKinematic[v2.mVertex[1]].mDistance),
						min(mClosestKinematic[v2.mVertex[2]].mDistance, mClosestKinematic[v2.mVertex[3]].mDistance));
			if (d1 != d2)
				return d1 < d2;

			// Order constraints so that the ones with the smallest index go first
			uint32 m1 = v1.GetMinVertexIndex();
			uint32 m2 = v2.GetMinVertexIndex();
			if (m1 != m2)
				return m1 < m2;

			return inLHS < inRHS;
		});

		// Sort the skinned constraints
		QuickSort(group.mSkinnedConstraints.begin(), group.mSkinnedConstraints.end(), [this](uint inLHS, uint inRHS)
			{
				const Skinned &s1 = mSkinnedConstraints[inLHS];
				const Skinned &s2 = mSkinnedConstraints[inRHS];

				// Order the skinned constraints so that the ones with the smallest index go first (hoping to get better cache locality when we process the edges).
				if (s1.mVertex != s2.mVertex)
					return s1.mVertex < s2.mVertex;

				return inLHS < inRHS;
			});
	}

	// Temporary store constraints as we reorder them
	Array<Edge> temp_edges;
	temp_edges.swap(mEdgeConstraints);
	mEdgeConstraints.reserve(temp_edges.size());
	outResults.mEdgeRemap.resize(temp_edges.size(), ~uint(0));

	Array<LRA> temp_lra;
	temp_lra.swap(mLRAConstraints);
	mLRAConstraints.reserve(temp_lra.size());
	outResults.mLRARemap.resize(temp_lra.size(), ~uint(0));

	Array<RodStretchShear> temp_rod_stretch_shear;
	temp_rod_stretch_shear.swap(mRodStretchShearConstraints);
	mRodStretchShearConstraints.reserve(temp_rod_stretch_shear.size());
	outResults.mRodStretchShearConstraintRemap.resize(temp_rod_stretch_shear.size(), ~uint(0));

	Array<RodBendTwist> temp_rod_bend_twist;
	temp_rod_bend_twist.swap(mRodBendTwistConstraints);
	mRodBendTwistConstraints.reserve(temp_rod_bend_twist.size());
	outResults.mRodBendTwistConstraintRemap.resize(temp_rod_bend_twist.size(), ~uint(0));

	Array<DihedralBend> temp_dihedral_bend;
	temp_dihedral_bend.swap(mDihedralBendConstraints);
	mDihedralBendConstraints.reserve(temp_dihedral_bend.size());
	outResults.mDihedralBendRemap.resize(temp_dihedral_bend.size(), ~uint(0));

	Array<Volume> temp_volume;
	temp_volume.swap(mVolumeConstraints);
	mVolumeConstraints.reserve(temp_volume.size());
	outResults.mVolumeRemap.resize(temp_volume.size(), ~uint(0));

	Array<Skinned> temp_skinned;
	temp_skinned.swap(mSkinnedConstraints);
	mSkinnedConstraints.reserve(temp_skinned.size());
	outResults.mSkinnedRemap.resize(temp_skinned.size(), ~uint(0));

	// Finalize update groups
	for (const Group &group : groups)
	{
		// Reorder edge constraints for this group
		for (uint idx : group.mEdgeConstraints)
		{
			outResults.mEdgeRemap[idx] = (uint)mEdgeConstraints.size();
			mEdgeConstraints.push_back(temp_edges[idx]);
		}

		// Reorder LRA constraints for this group
		for (uint idx : group.mLRAConstraints)
		{
			outResults.mLRARemap[idx] = (uint)mLRAConstraints.size();
			mLRAConstraints.push_back(temp_lra[idx]);
		}

		// Reorder rod stretch shear constraints for this group
		for (uint idx : group.mRodStretchShearConstraints)
		{
			outResults.mRodStretchShearConstraintRemap[idx] = (uint)mRodStretchShearConstraints.size();
			mRodStretchShearConstraints.push_back(temp_rod_stretch_shear[idx]);
		}

		// Reorder rod bend twist constraints for this group
		for (uint idx : group.mRodBendTwistConstraints)
		{
			outResults.mRodBendTwistConstraintRemap[idx] = (uint)mRodBendTwistConstraints.size();
			mRodBendTwistConstraints.push_back(temp_rod_bend_twist[idx]);
		}

		// Reorder dihedral bend constraints for this group
		for (uint idx : group.mDihedralBendConstraints)
		{
			outResults.mDihedralBendRemap[idx] = (uint)mDihedralBendConstraints.size();
			mDihedralBendConstraints.push_back(temp_dihedral_bend[idx]);
		}

		// Reorder volume constraints for this group
		for (uint idx : group.mVolumeConstraints)
		{
			outResults.mVolumeRemap[idx] = (uint)mVolumeConstraints.size();
			mVolumeConstraints.push_back(temp_volume[idx]);
		}

		// Reorder skinned constraints for this group
		for (uint idx : group.mSkinnedConstraints)
		{
			outResults.mSkinnedRemap[idx] = (uint)mSkinnedConstraints.size();
			mSkinnedConstraints.push_back(temp_skinned[idx]);
		}

		// Store end indices
		mUpdateGroups.push_back({ (uint)mEdgeConstraints.size(), (uint)mLRAConstraints.size(), (uint)mRodStretchShearConstraints.size(), (uint)mRodBendTwistConstraints.size(), (uint)mDihedralBendConstraints.size(), (uint)mVolumeConstraints.size(), (uint)mSkinnedConstraints.size() });
	}

	// Remap bend twist indices because mRodStretchShearConstraints has been reordered
	for (RodBendTwist &r : mRodBendTwistConstraints)
		for (int i = 0; i < 2; ++i)
			r.mRod[i] = outResults.mRodStretchShearConstraintRemap[r.mRod[i]];

	// Free closest kinematic buffer
	mClosestKinematic.clear();
	mClosestKinematic.shrink_to_fit();
}

Ref<SoftBodySharedSettings> SoftBodySharedSettings::Clone() const
{
	Ref<SoftBodySharedSettings> clone = new SoftBodySharedSettings;
	clone->mVertices = mVertices;
	clone->mFaces = mFaces;
	clone->mEdgeConstraints = mEdgeConstraints;
	clone->mDihedralBendConstraints = mDihedralBendConstraints;
	clone->mVolumeConstraints = mVolumeConstraints;
	clone->mSkinnedConstraints = mSkinnedConstraints;
	clone->mSkinnedConstraintNormals = mSkinnedConstraintNormals;
	clone->mInvBindMatrices = mInvBindMatrices;
	clone->mLRAConstraints = mLRAConstraints;
	clone->mRodStretchShearConstraints = mRodStretchShearConstraints;
	clone->mRodBendTwistConstraints = mRodBendTwistConstraints;
	clone->mMaterials = mMaterials;
	clone->mUpdateGroups = mUpdateGroups;
	return clone;
}

void SoftBodySharedSettings::SaveBinaryState(StreamOut &inStream) const
{
	inStream.Write(mVertices);
	inStream.Write(mFaces);
	inStream.Write(mEdgeConstraints);
	inStream.Write(mDihedralBendConstraints);
	inStream.Write(mVolumeConstraints);
	inStream.Write(mSkinnedConstraints);
	inStream.Write(mSkinnedConstraintNormals);
	inStream.Write(mLRAConstraints);
	inStream.Write(mUpdateGroups);

	// Can't write mRodStretchShearConstraints directly because the class contains padding
	inStream.Write(mRodStretchShearConstraints, [](const RodStretchShear &inElement, StreamOut &inS) {
		inS.Write(inElement.mVertex);
		inS.Write(inElement.mLength);
		inS.Write(inElement.mInvMass);
		inS.Write(inElement.mCompliance);
		inS.Write(inElement.mBishop);
	});

	// Can't write mRodBendTwistConstraints directly because the class contains padding
	inStream.Write(mRodBendTwistConstraints, [](const RodBendTwist &inElement, StreamOut &inS) {
		inS.Write(inElement.mRod);
		inS.Write(inElement.mCompliance);
		inS.Write(inElement.mOmega0);
	});

	// Can't write mInvBindMatrices directly because the class contains padding
	inStream.Write(mInvBindMatrices, [](const InvBind &inElement, StreamOut &inS) {
		inS.Write(inElement.mJointIndex);
		inS.Write(inElement.mInvBind);
	});
}

void SoftBodySharedSettings::RestoreBinaryState(StreamIn &inStream)
{
	inStream.Read(mVertices);
	inStream.Read(mFaces);
	inStream.Read(mEdgeConstraints);
	inStream.Read(mDihedralBendConstraints);
	inStream.Read(mVolumeConstraints);
	inStream.Read(mSkinnedConstraints);
	inStream.Read(mSkinnedConstraintNormals);
	inStream.Read(mLRAConstraints);
	inStream.Read(mUpdateGroups);

	inStream.Read(mRodStretchShearConstraints, [](StreamIn &inS, RodStretchShear &outElement) {
		inS.Read(outElement.mVertex);
		inS.Read(outElement.mLength);
		inS.Read(outElement.mInvMass);
		inS.Read(outElement.mCompliance);
		inS.Read(outElement.mBishop);
	});

	inStream.Read(mRodBendTwistConstraints, [](StreamIn &inS, RodBendTwist &outElement) {
		inS.Read(outElement.mRod);
		inS.Read(outElement.mCompliance);
		inS.Read(outElement.mOmega0);
	});

	inStream.Read(mInvBindMatrices, [](StreamIn &inS, InvBind &outElement) {
		inS.Read(outElement.mJointIndex);
		inS.Read(outElement.mInvBind);
	});
}

void SoftBodySharedSettings::SaveWithMaterials(StreamOut &inStream, SharedSettingsToIDMap &ioSettingsMap, MaterialToIDMap &ioMaterialMap) const
{
	SharedSettingsToIDMap::const_iterator settings_iter = ioSettingsMap.find(this);
	if (settings_iter == ioSettingsMap.end())
	{
		// Write settings ID
		uint32 settings_id = ioSettingsMap.size();
		ioSettingsMap[this] = settings_id;
		inStream.Write(settings_id);

		// Write the settings
		SaveBinaryState(inStream);

		// Write materials
		StreamUtils::SaveObjectArray(inStream, mMaterials, &ioMaterialMap);
	}
	else
	{
		// Known settings, just write the ID
		inStream.Write(settings_iter->second);
	}
}

SoftBodySharedSettings::SettingsResult SoftBodySharedSettings::sRestoreWithMaterials(StreamIn &inStream, IDToSharedSettingsMap &ioSettingsMap, IDToMaterialMap &ioMaterialMap)
{
	SettingsResult result;

	// Read settings id
	uint32 settings_id;
	inStream.Read(settings_id);
	if (inStream.IsEOF() || inStream.IsFailed())
	{
		result.SetError("Failed to read settings id");
		return result;
	}

	// Check nullptr settings
	if (settings_id == ~uint32(0))
	{
		result.Set(nullptr);
		return result;
	}

	// Check if we already read this settings
	if (settings_id < ioSettingsMap.size())
	{
		result.Set(ioSettingsMap[settings_id]);
		return result;
	}

	// Create new object
	Ref<SoftBodySharedSettings> settings = new SoftBodySharedSettings;

	// Read state
	settings->RestoreBinaryState(inStream);

	// Read materials
	Result mlresult = StreamUtils::RestoreObjectArray<PhysicsMaterialList>(inStream, ioMaterialMap);
	if (mlresult.HasError())
	{
		result.SetError(mlresult.GetError());
		return result;
	}
	settings->mMaterials = mlresult.Get();

	// Add the settings to the map
	ioSettingsMap.push_back(settings);

	result.Set(settings);
	return result;
}

Ref<SoftBodySharedSettings> SoftBodySharedSettings::sCreateCube(uint inGridSize, float inGridSpacing)
{
	const Vec3 cOffset = Vec3::sReplicate(-0.5f * inGridSpacing * (inGridSize - 1));

	// Create settings
	SoftBodySharedSettings *settings = new SoftBodySharedSettings;
	for (uint z = 0; z < inGridSize; ++z)
		for (uint y = 0; y < inGridSize; ++y)
			for (uint x = 0; x < inGridSize; ++x)
			{
				SoftBodySharedSettings::Vertex v;
				(cOffset + Vec3::sReplicate(inGridSpacing) * Vec3(float(x), float(y), float(z))).StoreFloat3(&v.mPosition);
				settings->mVertices.push_back(v);
			}

	// Function to get the vertex index of a point on the cube
	auto vertex_index = [inGridSize](uint inX, uint inY, uint inZ)
	{
		return inX + inY * inGridSize + inZ * inGridSize * inGridSize;
	};

	// Create edges
	for (uint z = 0; z < inGridSize; ++z)
		for (uint y = 0; y < inGridSize; ++y)
			for (uint x = 0; x < inGridSize; ++x)
			{
				SoftBodySharedSettings::Edge e;
				e.mVertex[0] = vertex_index(x, y, z);
				if (x < inGridSize - 1)
				{
					e.mVertex[1] = vertex_index(x + 1, y, z);
					settings->mEdgeConstraints.push_back(e);
				}
				if (y < inGridSize - 1)
				{
					e.mVertex[1] = vertex_index(x, y + 1, z);
					settings->mEdgeConstraints.push_back(e);
				}
				if (z < inGridSize - 1)
				{
					e.mVertex[1] = vertex_index(x, y, z + 1);
					settings->mEdgeConstraints.push_back(e);
				}
			}
	settings->CalculateEdgeLengths();

	// Tetrahedrons to fill a cube
	const int tetra_indices[6][4][3] = {
		{ {0, 0, 0}, {0, 1, 1}, {0, 0, 1}, {1, 1, 1} },
		{ {0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {1, 1, 1} },
		{ {0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1} },
		{ {0, 0, 0}, {1, 0, 1}, {1, 0, 0}, {1, 1, 1} },
		{ {0, 0, 0}, {1, 1, 0}, {0, 1, 0}, {1, 1, 1} },
		{ {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {1, 1, 1} }
	};

	// Create volume constraints
	for (uint z = 0; z < inGridSize - 1; ++z)
		for (uint y = 0; y < inGridSize - 1; ++y)
			for (uint x = 0; x < inGridSize - 1; ++x)
				for (uint t = 0; t < 6; ++t)
				{
					SoftBodySharedSettings::Volume v;
					for (uint i = 0; i < 4; ++i)
						v.mVertex[i] = vertex_index(x + tetra_indices[t][i][0], y + tetra_indices[t][i][1], z + tetra_indices[t][i][2]);
					settings->mVolumeConstraints.push_back(v);
				}

	settings->CalculateVolumeConstraintVolumes();

	// Create faces
	for (uint y = 0; y < inGridSize - 1; ++y)
		for (uint x = 0; x < inGridSize - 1; ++x)
		{
			SoftBodySharedSettings::Face f;

			// Face 1
			f.mVertex[0] = vertex_index(x, y, 0);
			f.mVertex[1] = vertex_index(x, y + 1, 0);
			f.mVertex[2] = vertex_index(x + 1, y + 1, 0);
			settings->AddFace(f);

			f.mVertex[1] = vertex_index(x + 1, y + 1, 0);
			f.mVertex[2] = vertex_index(x + 1, y, 0);
			settings->AddFace(f);

			// Face 2
			f.mVertex[0] = vertex_index(x, y, inGridSize - 1);
			f.mVertex[1] = vertex_index(x + 1, y + 1, inGridSize - 1);
			f.mVertex[2] = vertex_index(x, y + 1, inGridSize - 1);
			settings->AddFace(f);

			f.mVertex[1] = vertex_index(x + 1, y, inGridSize - 1);
			f.mVertex[2] = vertex_index(x + 1, y + 1, inGridSize - 1);
			settings->AddFace(f);

			// Face 3
			f.mVertex[0] = vertex_index(x, 0, y);
			f.mVertex[1] = vertex_index(x + 1, 0, y + 1);
			f.mVertex[2] = vertex_index(x, 0, y + 1);
			settings->AddFace(f);

			f.mVertex[1] = vertex_index(x + 1, 0, y);
			f.mVertex[2] = vertex_index(x + 1, 0, y + 1);
			settings->AddFace(f);

			// Face 4
			f.mVertex[0] = vertex_index(x, inGridSize - 1, y);
			f.mVertex[1] = vertex_index(x, inGridSize - 1, y + 1);
			f.mVertex[2] = vertex_index(x + 1, inGridSize - 1, y + 1);
			settings->AddFace(f);

			f.mVertex[1] = vertex_index(x + 1, inGridSize - 1, y + 1);
			f.mVertex[2] = vertex_index(x + 1, inGridSize - 1, y);
			settings->AddFace(f);

			// Face 5
			f.mVertex[0] = vertex_index(0, x, y);
			f.mVertex[1] = vertex_index(0, x, y + 1);
			f.mVertex[2] = vertex_index(0, x + 1, y + 1);
			settings->AddFace(f);

			f.mVertex[1] = vertex_index(0, x + 1, y + 1);
			f.mVertex[2] = vertex_index(0, x + 1, y);
			settings->AddFace(f);

			// Face 6
			f.mVertex[0] = vertex_index(inGridSize - 1, x, y);
			f.mVertex[1] = vertex_index(inGridSize - 1, x + 1, y + 1);
			f.mVertex[2] = vertex_index(inGridSize - 1, x, y + 1);
			settings->AddFace(f);

			f.mVertex[1] = vertex_index(inGridSize - 1, x + 1, y);
			f.mVertex[2] = vertex_index(inGridSize - 1, x + 1, y + 1);
			settings->AddFace(f);
		}

	// Optimize the settings
	settings->Optimize();

	return settings;
}

JPH_NAMESPACE_END
