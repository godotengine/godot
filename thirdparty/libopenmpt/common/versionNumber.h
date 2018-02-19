/*
 * versionNumber.h
 * ---------------
 * Purpose: OpenMPT version handling.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

OPENMPT_NAMESPACE_BEGIN

#define VER_HELPER_STRINGIZE(x) #x
#define VER_STRINGIZE(x)        VER_HELPER_STRINGIZE(x)

//Version definitions. The only thing that needs to be changed when changing version number.
#define VER_MAJORMAJOR          1
#define VER_MAJOR               27
#define VER_MINOR               04
#define VER_MINORMINOR          02

//Version string. For example "1.17.02.28"
#define MPT_VERSION_STR         VER_STRINGIZE(VER_MAJORMAJOR) "." VER_STRINGIZE(VER_MAJOR) "." VER_STRINGIZE(VER_MINOR) "." VER_STRINGIZE(VER_MINORMINOR)

//Numerical value of the version.
#define MPT_VERSION_NUMERIC     MAKE_VERSION_NUMERIC(VER_MAJORMAJOR,VER_MAJOR,VER_MINOR,VER_MINORMINOR)

OPENMPT_NAMESPACE_END
