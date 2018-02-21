/*
 * libopenmpt_version.h
 * --------------------
 * Purpose: libopenmpt public interface version
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#ifndef LIBOPENMPT_VERSION_H
#define LIBOPENMPT_VERSION_H

/*! \addtogroup libopenmpt
  @{
*/

/*! \brief libopenmpt major version number */
#define OPENMPT_API_VERSION_MAJOR 0
/*! \brief libopenmpt minor version number */
#define OPENMPT_API_VERSION_MINOR 3
/*! \brief libopenmpt patch version number */
#define OPENMPT_API_VERSION_PATCH 6
/*! \brief libopenmpt pre-release tag */
#define OPENMPT_API_VERSION_PREREL ""
/*! \brief libopenmpt pre-release flag */
#define OPENMPT_API_VERSION_IS_PREREL 0

/*! \brief libopenmpt version number as a single integer value
 *  \since 0.3
 *  \remarks Use the following shim if you need to support earlier libopenmpt versions:
 *           \code
 *           #include <libopenmpt/libopenmpt_version.h>
 *           #if !defined(OPENMPT_API_VERSION_MAKE)
 *           #define OPENMPT_API_VERSION_MAKE(major, minor, patch) (((major)<<24)|((minor)<<16)|((patch)<<0))
 *           #endif
 *           \endcode
 */
#define OPENMPT_API_VERSION_MAKE(major, minor, patch) (((major)<<24)|((minor)<<16)|((patch)<<0))

/*! \brief libopenmpt API version number */
#define OPENMPT_API_VERSION OPENMPT_API_VERSION_MAKE(OPENMPT_API_VERSION_MAJOR, OPENMPT_API_VERSION_MINOR, OPENMPT_API_VERSION_PATCH)

/*! \brief Check whether the libopenmpt API is at least the provided version
 *  \since 0.3
 *  \remarks Use the following shim if you need to support earlier libopenmpt versions:
 *           \code
 *           #include <libopenmpt/libopenmpt_version.h>
 *           #if !defined(OPENMPT_API_VERSION_AT_LEAST)
 *           #define OPENMPT_API_VERSION_AT_LEAST(major, minor, patch) (OPENMPT_API_VERSION >= OPENMPT_API_VERSION_MAKE((major), (minor), (patch)))
 *           #endif
 *           \endcode
 */
#define OPENMPT_API_VERSION_AT_LEAST(major, minor, patch) (OPENMPT_API_VERSION >= OPENMPT_API_VERSION_MAKE((major), (minor), (patch)))

/*! \brief Check whether the libopenmpt API is before the provided version
 *  \since 0.3
 *  \remarks Use the following shim if you need to support earlier libopenmpt versions:
 *           \code
 *           #include <libopenmpt/libopenmpt_version.h>
 *           #if !defined(OPENMPT_API_VERSION_BEFORE)
 *           #define OPENMPT_API_VERSION_BEFORE(major, minor, patch) (OPENMPT_API_VERSION < OPENMPT_API_VERSION_MAKE((major), (minor), (patch)))
 *           #endif
 *           \endcode
 */
#define OPENMPT_API_VERSION_BEFORE(major, minor, patch) (OPENMPT_API_VERSION < OPENMPT_API_VERSION_MAKE((major), (minor), (patch)))

#define OPENMPT_API_VERSION_HELPER_STRINGIZE(x) #x
#define OPENMPT_API_VERSION_STRINGIZE(x) OPENMPT_API_VERSION_HELPER_STRINGIZE(x)
#define OPENMPT_API_VERSION_STRING OPENMPT_API_VERSION_STRINGIZE(OPENMPT_API_VERSION_MAJOR) "." OPENMPT_API_VERSION_STRINGIZE(OPENMPT_API_VERSION_MINOR) "." OPENMPT_API_VERSION_STRINGIZE(OPENMPT_API_VERSION_PATCH) OPENMPT_API_VERSION_PREREL

/*!
  @}
*/

#endif /* LIBOPENMPT_VERSION_H */
