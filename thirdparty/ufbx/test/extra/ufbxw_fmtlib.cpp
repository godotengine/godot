#include "../../ufbx_write.h"

#include "fmt/format.h"
#if defined(_MSC_VER) && _MSVC_LANG >= 201703L
	#include "fmt/compile.h"
#elif __cplusplus >= 201703L
	#include "fmt/compile.h"
#endif

#define UFBXW_FMTLIB_IMPLEMENTATION
#include "../../extra/ufbxw_fmtlib.h"

