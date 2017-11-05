#ifndef B3_PLATFORM_DEFINITIONS_H
#define B3_PLATFORM_DEFINITIONS_H

struct MyTest
{
	int bla;
};

#ifdef __cplusplus
//#define b3ConstArray(a) const b3AlignedObjectArray<a>&
#define b3ConstArray(a) const a*
#define b3AtomicInc(a) ((*a)++)

inline int b3AtomicAdd (volatile int *p, int val)
{
	int oldValue = *p;
	int newValue = oldValue+val;
	*p = newValue;
	return oldValue;
}

#define __global 

#define B3_STATIC static
#else
//keep B3_LARGE_FLOAT*B3_LARGE_FLOAT < FLT_MAX
#define B3_LARGE_FLOAT 1e18f
#define B3_INFINITY 1e18f
#define b3Assert(a)
#define b3ConstArray(a) __global const a*
#define b3AtomicInc atomic_inc
#define b3AtomicAdd atomic_add
#define b3Fabs fabs
#define b3Sqrt native_sqrt
#define b3Sin native_sin
#define b3Cos native_cos

#define B3_STATIC
#endif

#endif
