#ifndef SPX_UTIL_H
#define SPX_UTIL_H
#include "core/string/ustring.h"
typedef void (*RegisterSpxInterfaceFunc)();

class SpxUtil {
public:
	static RegisterSpxInterfaceFunc register_func;
	static bool debug_mode;
};

#endif // SPX_UTIL_H