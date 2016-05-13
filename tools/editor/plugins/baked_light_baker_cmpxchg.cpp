
#include "typedefs.h"


#if (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 40100

void baked_light_baker_add_64f(double *dst,double value) {


	union {
		int64_t i;
		double f;
	} swapy;


	while(true) {
		swapy.f=*dst;
		int64_t from = swapy.i;
		swapy.f+=value;
		int64_t to=swapy.i;
		if (__sync_bool_compare_and_swap((int64_t*)dst,from,to))
			break;
	}
}

void baked_light_baker_add_64i(int64_t *dst,int64_t value) {

	while(!__sync_bool_compare_and_swap(dst,*dst,(*dst)+value)) {}

}

#elif defined(WINDOWS_ENABLED)

#include "windows.h"

void baked_light_baker_add_64f(double *dst,double value) {

	union {
		int64_t i;
		double f;
	} swapy;


	while(true) {
		swapy.f=*dst;
		int64_t from = swapy.i;
		swapy.f+=value;
		int64_t to=swapy.i;
		int64_t result = InterlockedCompareExchange64((int64_t*)dst,to,from);
		if (result==from)
			break;
	}

}

void baked_light_baker_add_64i(int64_t *dst,int64_t value) {

	while(true) {
		int64_t from = *dst;
		int64_t to = from+value;
		int64_t result = InterlockedCompareExchange64(dst,to,from);
		if (result==from)
			break;
	}
}


#else

//in goder (the god of programmers) we trust
#warning seems this platform or compiler does not support safe cmpxchg, your baked lighting may be funny

void baked_light_baker_add_64f(double *dst,double value) {

	*dst+=value;

}

void baked_light_baker_add_64i(int64_t *dst,int64_t value) {

	*dst+=value;

}

#endif
