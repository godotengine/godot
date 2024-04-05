#ifndef _C4_CONFIG_HPP_
#define _C4_CONFIG_HPP_

/** @defgroup basic_headers Basic headers
 * @brief Headers providing basic macros, platform+cpu+compiler information,
 * C++ facilities and basic typedefs. */

/** @file config.hpp Contains configuration defines and includes the basic_headers.
 * @ingroup basic_headers */

//#define C4_DEBUG

#define C4_ERROR_SHOWS_FILELINE
//#define C4_ERROR_SHOWS_FUNC
//#define C4_ERROR_THROWS_EXCEPTION
//#define C4_NO_ALLOC_DEFAULTS
//#define C4_REDEFINE_CPPNEW

#ifndef C4_SIZE_TYPE
#   define C4_SIZE_TYPE size_t
#endif

#ifndef C4_STR_SIZE_TYPE
#   define C4_STR_SIZE_TYPE C4_SIZE_TYPE
#endif

#ifndef C4_TIME_TYPE
#   define C4_TIME_TYPE double
#endif

#include "export.hpp"
#include "preprocessor.hpp"
#include "platform.hpp"
#include "cpu.hpp"
#include "compiler.hpp"
#include "language.hpp"
#include "types.hpp"

#endif // _C4_CONFIG_HPP_
