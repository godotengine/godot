// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/AABBTree/AABBTreeBuilder.h>

JPH_NAMESPACE_BEGIN

AABBTreeBuilder::Node::Node()
{
	mChild[0] = nullptr;
	mChild[1] = nullptr;
}

AABBTreeBuilder::Node::~Node()
{
	delete mChild[0];
	delete mChild[1];
}

uint AABBTreeBuilder::Node::GetMinDepth() const
{
	if (HasChildren())
	{
		uint left = mChild[0]->GetMinDepth();
		uint right = mChild[1]->GetMinDepth();
		return min(left, right) + 1;
	}
	else
		return 1;
}

uint AABBTreeBuilder::Node::GetMaxDepth() const
{
	if (HasChildren())
	{
		uint left = mChild[0]->GetMaxDepth();
		uint right = mChild[1]->GetMaxDepth();
		return max(left, right) + 1;
	}
	else
		return 1;
}

uint AABBTreeBuilder::Node::GetNodeCount() const
{
	if (HasChildren())
		return mChild[0]->GetNodeCount() + mChild[1]->GetNodeCount() + 1;
	else
		return 1;
}

uint AABBTreeBuilder::Node::GetLeafNodeCount() const
{
	if (HasChildren())
		return mChild[0]->GetLeafNodeCount() + mChild[1]->GetLeafNodeCount();
	else
		return 1;
}

uint AABBTreeBuilder::Node::GetTriangleCountInTree() const
{
	if (HasChildren())
		return mChild[0]->GetTriangleCountInTree() + mChild[1]->GetTriangleCountInTree();
	else
		return GetTriangleCount();
}

void AABBTreeBuilder::Node::GetTriangleCountPerNode(float &outAverage, uint &outMin, uint &outMax) const
{
	outMin = INT_MAX;
	outMax = 0;
	outAverage = 0;
	uint avg_divisor = 0;
	GetTriangleCountPerNodeInternal(outAverage, avg_divisor, outMin, outMax);
	if (avg_divisor > 0)
		outAverage /= avg_divisor;
}

float AABBTreeBuilder::Node::CalculateSAHCost(float inCostTraversal, float inCostLeaf) const
{
	float surface_area = mBounds.GetSurfaceArea();
	return surface_area > 0.0f? CalculateSAHCostInternal(inCostTraversal / surface_area, inCostLeaf / surface_area) : 0.0f;
}

void AABBTreeBuilder::Node::GetNChildren(uint inN, Array<const Node *> &outChildren) const
{
	JPH_ASSERT(outChildren.empty());

	// Check if there is anything to expand
	if (!HasChildren())
		return;

	// Start with the children of this node
	outChildren.push_back(mChild[0]);
	outChildren.push_back(mChild[1]);

	size_t next = 0;
	bool all_triangles = true;
	while (outChildren.size() < inN)
	{
		// If we have looped over all nodes, start over with the first node again
		if (next >= outChildren.size())
		{
			// If there only triangle nodes left, we have to terminate
			if (all_triangles)
				return;
			next = 0;
			all_triangles = true;
		}

		// Try to expand this node into its two children
		const Node *to_expand = outChildren[next];
		if (to_expand->HasChildren())
		{
			outChildren.erase(outChildren.begin() + next);
			outChildren.push_back(to_expand->mChild[0]);
			outChildren.push_back(to_expand->mChild[1]);
			all_triangles = false;
		}
		else
		{
			++next;
		}
	}
}

float AABBTreeBuilder::Node::CalculateSAHCostInternal(float inCostTraversalDivSurfaceArea, float inCostLeafDivSurfaceArea) const
{
	if (HasChildren())
		return inCostTraversalDivSurfaceArea * mBounds.GetSurfaceArea()
			+ mChild[0]->CalculateSAHCostInternal(inCostTraversalDivSurfaceArea, inCostLeafDivSurfaceArea)
			+ mChild[1]->CalculateSAHCostInternal(inCostTraversalDivSurfaceArea, inCostLeafDivSurfaceArea);
	else
		return inCostLeafDivSurfaceArea * mBounds.GetSurfaceArea() * GetTriangleCount();
}

void AABBTreeBuilder::Node::GetTriangleCountPerNodeInternal(float &outAverage, uint &outAverageDivisor, uint &outMin, uint &outMax) const
{
	if (HasChildren())
	{
		mChild[0]->GetTriangleCountPerNodeInternal(outAverage, outAverageDivisor, outMin, outMax);
		mChild[1]->GetTriangleCountPerNodeInternal(outAverage, outAverageDivisor, outMin, outMax);
	}
	else
	{
		outAverage += GetTriangleCount();
		outAverageDivisor++;
		outMin = min(outMin, GetTriangleCount());
		outMax = max(outMax, GetTriangleCount());
	}
}

AABBTreeBuilder::AABBTreeBuilder(TriangleSplitter &inSplitter, uint inMaxTrianglesPerLeaf) :
	mTriangleSplitter(inSplitter),
	mMaxTrianglesPerLeaf(inMaxTrianglesPerLeaf)
{
}

AABBTreeBuilder::Node *AABBTreeBuilder::Build(AABBTreeBuilderStats &outStats)
{
	TriangleSplitter::Range initial = mTriangleSplitter.GetInitialRange();
	Node *root = BuildInternal(initial);

	float avg_triangles_per_leaf;
	uint min_triangles_per_leaf, max_triangles_per_leaf;
	root->GetTriangleCountPerNode(avg_triangles_per_leaf, min_triangles_per_leaf, max_triangles_per_leaf);

	mTriangleSplitter.GetStats(outStats.mSplitterStats);

	outStats.mSAHCost = root->CalculateSAHCost(1.0f, 1.0f);
	outStats.mMinDepth = root->GetMinDepth();
	outStats.mMaxDepth = root->GetMaxDepth();
	outStats.mNodeCount = root->GetNodeCount();
	outStats.mLeafNodeCount = root->GetLeafNodeCount();
	outStats.mMaxTrianglesPerLeaf = mMaxTrianglesPerLeaf;
	outStats.mTreeMinTrianglesPerLeaf = min_triangles_per_leaf;
	outStats.mTreeMaxTrianglesPerLeaf = max_triangles_per_leaf;
	outStats.mTreeAvgTrianglesPerLeaf = avg_triangles_per_leaf;

	return root;
}

AABBTreeBuilder::Node *AABBTreeBuilder::BuildInternal(const TriangleSplitter::Range &inTriangles)
{
	// Check if there are too many triangles left
	if (inTriangles.Count() > mMaxTrianglesPerLeaf)
	{
		// Split triangles in two batches
		TriangleSplitter::Range left, right;
		if (!mTriangleSplitter.Split(inTriangles, left, right))
		{
			// When the trace below triggers:
			//
			// This code builds a tree structure to accelerate collision detection.
			// At top level it will start with all triangles in a mesh and then divides the triangles into two batches.
			// This process repeats until until the batch size is smaller than mMaxTrianglePerLeaf.
			//
			// It uses a TriangleSplitter to find a good split. When this warning triggers, the splitter was not able
			// to create a reasonable split for the triangles. This usually happens when the triangles in a batch are
			// intersecting. They could also be overlapping when projected on the 3 coordinate axis.
			//
			// To solve this issue, you could try to pass your mesh through a mesh cleaning / optimization algorithm.
			// You could also inspect the triangles that cause this issue and see if that part of the mesh can be fixed manually.
			//
			// When you do not fix this warning, the tree will be less efficient for collision detection, but it will still work.
			JPH_IF_DEBUG(Trace("AABBTreeBuilder: Doing random split for %d triangles (max per node: %u)!", (int)inTriangles.Count(), mMaxTrianglesPerLeaf);)
			int half = inTriangles.Count() / 2;
			JPH_ASSERT(half > 0);
			left = TriangleSplitter::Range(inTriangles.mBegin, inTriangles.mBegin + half);
			right = TriangleSplitter::Range(inTriangles.mBegin + half, inTriangles.mEnd);
		}

		// Recursively build
		Node *node = new Node();
		node->mChild[0] = BuildInternal(left);
		node->mChild[1] = BuildInternal(right);
		node->mBounds = node->mChild[0]->mBounds;
		node->mBounds.Encapsulate(node->mChild[1]->mBounds);
		return node;
	}

	// Create leaf node
	Node *node = new Node();
	node->mTriangles.reserve(inTriangles.Count());
	for (uint i = inTriangles.mBegin; i < inTriangles.mEnd; ++i)
	{
		const IndexedTriangle &t = mTriangleSplitter.GetTriangle(i);
		const VertexList &v = mTriangleSplitter.GetVertices();
		node->mTriangles.push_back(t);
		node->mBounds.Encapsulate(v, t);
	}

	return node;
}

JPH_NAMESPACE_END
