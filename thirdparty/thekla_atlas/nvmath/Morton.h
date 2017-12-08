
// Code from ryg:
// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/


// "Insert" a 0 bit after each of the 16 low bits of x
inline uint32 part1By1(uint32 x)
{
	x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
inline uint32 part1By2(uint32 x)
{
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

inline uint32 encodeMorton2(uint32 x, uint32 y)
{
	return (part1By1(y) << 1) + part1By1(x);
}

inline uint32 encodeMorton3(uint32 x, uint32 y, uint32 z)
{
	return (part1By2(z) << 2) + (part1By2(y) << 1) + part1By2(x);
}

// Inverse of part1By1 - "delete" all odd-indexed bits
inline uint32 compact1By1(uint32 x)
{
	x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x >>  1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x >>  2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x >>  4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x >>  8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
	return x;
}

// Inverse of part1By2 - "delete" all bits not at positions divisible by 3
inline uint32 compact1By2(uint32 x)
{
	x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}

inline uint32 decodeMorton2X(uint32 code)
{
	return compact1By1(code >> 0);
}

inline uint32 decodeMorton2Y(uint32 code)
{
	return compact1By1(code >> 1);
}

inline uint32 decodeMorton3X(uint32 code)
{
	return compact1By2(code >> 0);
}

inline uint32 decodeMorton3Y(uint32 code)
{
	return compact1By2(code >> 1);
}

inline uint32 decodeMorton3Z(uint32 code)
{
	return compact1By2(code >> 2);
}