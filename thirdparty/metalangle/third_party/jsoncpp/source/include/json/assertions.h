// Copyright 2007-2010 Baptiste Lepilleur and The JsonCpp Authors
// Distributed under MIT license, or public domain if desired and
// recognized in your jurisdiction.
// See file LICENSE for detail or copy at http://jsoncpp.sourceforge.net/LICENSE

#ifndef CPPTL_JSON_ASSERTIONS_H_INCLUDED
#define CPPTL_JSON_ASSERTIONS_H_INCLUDED

#include <cstdlib>
#include <sstream>

#if !defined(JSON_IS_AMALGAMATION)
#include "config.h"
#endif // if !defined(JSON_IS_AMALGAMATION)

/** It should not be possible for a maliciously designed file to
 *  cause an abort() or seg-fault, so these macros are used only
 *  for pre-condition violations and internal logic errors.
 */
#if JSON_USE_EXCEPTION

// @todo <= add detail about condition in exception
#define JSON_ASSERT(condition)                                                 \
  {                                                                            \
    if (!(condition)) {                                                        \
      Json::throwLogicError("assert json failed");                             \
    }                                                                          \
  }

#define JSON_FAIL_MESSAGE(message)                                             \
  {                                                                            \
    OStringStream oss;                                                         \
    oss << message;                                                            \
    Json::throwLogicError(oss.str());                                          \
    abort();                                                                   \
  }

#else // JSON_USE_EXCEPTION

#define JSON_ASSERT(condition) assert(condition)

// The call to assert() will show the failure message in debug builds. In
// release builds we abort, for a core-dump or debugger.
#define JSON_FAIL_MESSAGE(message)                                             \
  {                                                                            \
    OStringStream oss;                                                         \
    oss << message;                                                            \
    assert(false && oss.str().c_str());                                        \
    abort();                                                                   \
  }

#endif

#define JSON_ASSERT_MESSAGE(condition, message)                                \
  if (!(condition)) {                                                          \
    JSON_FAIL_MESSAGE(message);                                                \
  }

#endif // CPPTL_JSON_ASSERTIONS_H_INCLUDED
