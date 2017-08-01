



#ifndef B3_GJK_PAIR_DETECTOR_H
#define B3_GJK_PAIR_DETECTOR_H


#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Common/b3AlignedObjectArray.h"

class b3Transform;
struct b3GjkEpaSolver2;
class b3VoronoiSimplexSolver;
struct b3ConvexPolyhedronData;

B3_ATTRIBUTE_ALIGNED16(struct) b3GjkPairDetector
{
	

	b3Vector3	m_cachedSeparatingAxis;
	b3GjkEpaSolver2*	m_penetrationDepthSolver;
	b3VoronoiSimplexSolver* m_simplexSolver;


	bool		m_ignoreMargin;
	b3Scalar	m_cachedSeparatingDistance;
	

public:

	//some debugging to fix degeneracy problems
	int			m_lastUsedMethod;
	int			m_curIter;
	int			m_degenerateSimplex;
	int			m_catchDegeneracies;
	int			m_fixContactNormalDirection;

	b3GjkPairDetector(b3VoronoiSimplexSolver* simplexSolver,b3GjkEpaSolver2*	penetrationDepthSolver);
	
	virtual ~b3GjkPairDetector() {};

	
	//void	getClosestPoints(,Result& output);
	
	void setCachedSeperatingAxis(const b3Vector3& seperatingAxis)
	{
		m_cachedSeparatingAxis = seperatingAxis;
	}

	const b3Vector3& getCachedSeparatingAxis() const
	{
		return m_cachedSeparatingAxis;
	}
	b3Scalar	getCachedSeparatingDistance() const
	{
		return m_cachedSeparatingDistance;
	}

	void	setPenetrationDepthSolver(b3GjkEpaSolver2*	penetrationDepthSolver)
	{
		m_penetrationDepthSolver = penetrationDepthSolver;
	}

	///don't use setIgnoreMargin, it's for Bullet's internal use
	void	setIgnoreMargin(bool ignoreMargin)
	{
		m_ignoreMargin = ignoreMargin;
	}


};


bool getClosestPoints(b3GjkPairDetector* gjkDetector, const b3Transform&	transA, const b3Transform&	transB,
	const b3ConvexPolyhedronData& hullA, const b3ConvexPolyhedronData& hullB, 
	const b3AlignedObjectArray<b3Vector3>& verticesA,
	const b3AlignedObjectArray<b3Vector3>& verticesB,
	b3Scalar maximumDistanceSquared,
	b3Vector3& resultSepNormal,
	float& resultSepDistance,
	b3Vector3& resultPointOnB);

#endif //B3_GJK_PAIR_DETECTOR_H
