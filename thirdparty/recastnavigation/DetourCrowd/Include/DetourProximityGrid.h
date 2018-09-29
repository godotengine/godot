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

#ifndef DETOURPROXIMITYGRID_H
#define DETOURPROXIMITYGRID_H

class dtProximityGrid
{
	float m_cellSize;
	float m_invCellSize;
	
	struct Item
	{
		unsigned short id;
		short x,y;
		unsigned short next;
	};
	Item* m_pool;
	int m_poolHead;
	int m_poolSize;
	
	unsigned short* m_buckets;
	int m_bucketsSize;
	
	int m_bounds[4];
	
public:
	dtProximityGrid();
	~dtProximityGrid();
	
	bool init(const int poolSize, const float cellSize);
	
	void clear();
	
	void addItem(const unsigned short id,
				 const float minx, const float miny,
				 const float maxx, const float maxy);
	
	int queryItems(const float minx, const float miny,
				   const float maxx, const float maxy,
				   unsigned short* ids, const int maxIds) const;
	
	int getItemCountAt(const int x, const int y) const;
	
	inline const int* getBounds() const { return m_bounds; }
	inline float getCellSize() const { return m_cellSize; }

private:
	// Explicitly disabled copy constructor and copy assignment operator.
	dtProximityGrid(const dtProximityGrid&);
	dtProximityGrid& operator=(const dtProximityGrid&);
};

dtProximityGrid* dtAllocProximityGrid();
void dtFreeProximityGrid(dtProximityGrid* ptr);


#endif // DETOURPROXIMITYGRID_H

