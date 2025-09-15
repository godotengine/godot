#ifndef SPX_UTILS_H
#define SPX_UTILS_H

#include "gdextension_spx_ext.h"

#define SPX_FLOAT_TO_INT_FACTOR 10000 
#define SPX_INT_TO_FLOAT_FACTOR 0.0001

inline GdInt spx_float_to_int(GdFloat value){
	return (GdInt)(value * SPX_FLOAT_TO_INT_FACTOR);
}

inline GdFloat spx_int_to_float(GdInt value){
	return (GdFloat)value * SPX_INT_TO_FLOAT_FACTOR;
}

#endif // SPX_UTILS_H
