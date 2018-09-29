//
// Copyright (c) 2009-2010 Mikko Mononen memon@inside.org
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any damages
// arising from the use of this software.
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would be
//    appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and must not be
//    misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source distribution.
//

#ifndef DETOURLOCALBOUNDARY_H
#define DETOURLOCALBOUNDARY_H

#include "DetourNavMeshQuery.h"


class dtLocalBoundary
{
	static const int MAX_LOCAL_SEGS = 8;
	static const int MAX_LOCAL_POLYS = 16;
	
	struct Segment
	{
		float s[6];	///< Segment start/end
		float d;	///< Distance for pruning.
	};
	
	float m_center[3];
	Segment m_segs[MAX_LOCAL_SEGS];
	int m_nsegs;
	
	dtPolyRef m_polys[MAX_LOCAL_POLYS];
	int m_npolys;

	void addSegment(const float dist, const float* s);
	
public:
	dtLocalBoundary();
	~dtLocalBoundary();
	
	void reset();
	
	void update(dtPolyRef ref, const float* pos, const float collisionQueryRange,
				dtNavMeshQuery* navquery, const dtQueryFilter* filter);
	
	bool isValid(dtNavMeshQuery* navquery, const dtQueryFilter* filter);
	
	inline const float* getCenter() const { return m_center; }
	inline int getSegmentCount() const { return m_nsegs; }
	inline const float* getSegment(int i) const { return m_segs[i].s; }

private:
	// Explicitly disabled copy constructor and copy assignment operator.
	dtLocalBoundary(const dtLocalBoundary&);
	dtLocalBoundary& operator=(const dtLocalBoundary&);
};

#endif // DETOURLOCALBOUNDARY_H
