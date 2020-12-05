/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2011 Advanced Micro Devices, Inc.  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B3_OPENCL_INCLUDE_H
#define B3_OPENCL_INCLUDE_H

#ifdef B3_USE_CLEW
#include "clew/clew.h"
#else

#ifdef __APPLE__
#ifdef USE_MINICL
#include <MiniCL/cl.h>
#else
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>  //clLogMessagesToStderrAPPLE
#endif
#else
#ifdef USE_MINICL
#include <MiniCL/cl.h>
#else
#include <CL/cl.h>
#ifdef _WIN32
#include "CL/cl_gl.h"
#endif  //_WIN32
#endif
#endif  //__APPLE__
#endif  //B3_USE_CLEW

#include <assert.h>
#include <stdio.h>
#define oclCHECKERROR(a, b)              \
	if ((a) != (b))                      \
	{                                    \
		printf("OCL Error : %d\n", (a)); \
		assert((a) == (b));              \
	}

#endif  //B3_OPENCL_INCLUDE_H
