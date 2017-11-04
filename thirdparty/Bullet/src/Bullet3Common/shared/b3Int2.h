/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B3_INT2_H
#define B3_INT2_H

#ifdef __cplusplus

struct b3UnsignedInt2
{
	union
	{
		struct
		{
			unsigned int x,y;
		};
		struct
		{
			unsigned int s[2];
		};
	};
};

struct b3Int2
{
	union
	{
		struct
		{
			int x,y;
		};
		struct
		{
			int s[2];
		};
	};
};

inline b3Int2 b3MakeInt2(int x, int y)
{
	b3Int2 v;
	v.s[0] = x; v.s[1] = y;
	return v;
}
#else

#define b3UnsignedInt2 uint2
#define b3Int2 int2
#define b3MakeInt2 (int2)

#endif //__cplusplus
#endif