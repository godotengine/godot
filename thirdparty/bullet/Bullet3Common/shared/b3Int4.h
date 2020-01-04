#ifndef B3_INT4_H
#define B3_INT4_H

#ifdef __cplusplus

#include "Bullet3Common/b3Scalar.h"

B3_ATTRIBUTE_ALIGNED16(struct)
b3UnsignedInt4
{
	B3_DECLARE_ALIGNED_ALLOCATOR();

	union {
		struct
		{
			unsigned int x, y, z, w;
		};
		struct
		{
			unsigned int s[4];
		};
	};
};

B3_ATTRIBUTE_ALIGNED16(struct)
b3Int4
{
	B3_DECLARE_ALIGNED_ALLOCATOR();

	union {
		struct
		{
			int x, y, z, w;
		};
		struct
		{
			int s[4];
		};
	};
};

B3_FORCE_INLINE b3Int4 b3MakeInt4(int x, int y, int z, int w = 0)
{
	b3Int4 v;
	v.s[0] = x;
	v.s[1] = y;
	v.s[2] = z;
	v.s[3] = w;
	return v;
}

B3_FORCE_INLINE b3UnsignedInt4 b3MakeUnsignedInt4(unsigned int x, unsigned int y, unsigned int z, unsigned int w = 0)
{
	b3UnsignedInt4 v;
	v.s[0] = x;
	v.s[1] = y;
	v.s[2] = z;
	v.s[3] = w;
	return v;
}

#else

#define b3UnsignedInt4 uint4
#define b3Int4 int4
#define b3MakeInt4 (int4)
#define b3MakeUnsignedInt4 (uint4)

#endif  //__cplusplus

#endif  //B3_INT4_H
