/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Takahiro Harada

#ifndef B3_BOUNDSEARCH_H
#define B3_BOUNDSEARCH_H

#pragma once

/*#include <Adl/Adl.h>
#include <AdlPrimitives/Math/Math.h>
#include <AdlPrimitives/Sort/SortData.h>
#include <AdlPrimitives/Fill/Fill.h>
*/

#include "b3OpenCLArray.h"
#include "b3FillCL.h"
#include "b3RadixSort32CL.h"  //for b3SortData (perhaps move it?)
class b3BoundSearchCL
{
public:
	enum Option
	{
		BOUND_LOWER,
		BOUND_UPPER,
		COUNT,
	};

	cl_context m_context;
	cl_device_id m_device;
	cl_command_queue m_queue;

	cl_kernel m_lowerSortDataKernel;
	cl_kernel m_upperSortDataKernel;
	cl_kernel m_subtractKernel;

	b3OpenCLArray<b3Int4>* m_constbtOpenCLArray;
	b3OpenCLArray<unsigned int>* m_lower;
	b3OpenCLArray<unsigned int>* m_upper;

	b3FillCL* m_filler;

	b3BoundSearchCL(cl_context context, cl_device_id device, cl_command_queue queue, int size);

	virtual ~b3BoundSearchCL();

	//	src has to be src[i].m_key <= src[i+1].m_key
	void execute(b3OpenCLArray<b3SortData>& src, int nSrc, b3OpenCLArray<unsigned int>& dst, int nDst, Option option = BOUND_LOWER);

	void executeHost(b3AlignedObjectArray<b3SortData>& src, int nSrc, b3AlignedObjectArray<unsigned int>& dst, int nDst, Option option = BOUND_LOWER);
};

#endif  //B3_BOUNDSEARCH_H
