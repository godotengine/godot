// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_MESH_H
#define NV_MESH_H

#include "nvcore/nvcore.h"

// Function linkage
#if NVMESH_SHARED
#ifdef NVMESH_EXPORTS
#define NVMESH_API DLL_EXPORT
#define NVMESH_CLASS DLL_EXPORT_CLASS
#else
#define NVMESH_API DLL_IMPORT
#define NVMESH_CLASS DLL_IMPORT
#endif
#else
#define NVMESH_API
#define NVMESH_CLASS
#endif

#if 1 //USE_PRECOMPILED_HEADERS // If using precompiled headers:
//#include <string.h> // strlen, strcmp, etc.
//#include "nvcore/StrLib.h"
//#include "nvcore/StdStream.h"
//#include "nvcore/Memory.h"
//#include "nvcore/Debug.h"
//#include "nvmath/Vector.h"
//#include "nvcore/Array.h"
//#include "nvcore/HashMap.h"
#endif

#endif // NV_MESH_H
