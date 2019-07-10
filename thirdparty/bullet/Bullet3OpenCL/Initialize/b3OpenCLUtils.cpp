/*
Bullet Continuous Collision Detection and Physics Library, http://bulletphysics.org
Copyright (C) 2006 - 2011 Sony Computer Entertainment Inc.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

//Original author: Roman Ponomarev
//Mostly Reimplemented by Erwin Coumans

bool gDebugForceLoadingFromSource = false;
bool gDebugSkipLoadingBinary = false;

#include "Bullet3Common/b3Logging.h"

#include <string.h>

#ifdef _WIN32
#pragma warning(disable : 4996)
#endif
#include "b3OpenCLUtils.h"
//#include "b3OpenCLInclude.h"

#include <stdio.h>
#include <stdlib.h>

#define B3_MAX_CL_DEVICES 16  //who needs 16 devices?

#ifdef _WIN32
#include <windows.h>
#endif

#include <assert.h>
#define b3Assert assert
#ifndef _WIN32
#include <sys/stat.h>

#endif

static const char* sCachedBinaryPath = "cache";

//Set the preferred platform vendor using the OpenCL SDK
static const char* spPlatformVendor =
#if defined(CL_PLATFORM_MINI_CL)
	"MiniCL, SCEA";
#elif defined(CL_PLATFORM_AMD)
	"Advanced Micro Devices, Inc.";
#elif defined(CL_PLATFORM_NVIDIA)
	"NVIDIA Corporation";
#elif defined(CL_PLATFORM_INTEL)
	"Intel(R) Corporation";
#elif defined(B3_USE_CLEW)
	"clew (OpenCL Extension Wrangler library)";
#else
	"Unknown Vendor";
#endif

#ifndef CL_PLATFORM_MINI_CL
#ifdef _WIN32
#ifndef B3_USE_CLEW
#include "CL/cl_gl.h"
#endif  //B3_USE_CLEW
#endif  //_WIN32
#endif

void MyFatalBreakAPPLE(const char* errstr,
					   const void* private_info,
					   size_t cb,
					   void* user_data)
{
	const char* patloc = strstr(errstr, "Warning");
	//find out if it is a warning or error, exit if error

	if (patloc)
	{
		b3Warning("Warning: %s\n", errstr);
	}
	else
	{
		b3Error("Error: %s\n", errstr);
		b3Assert(0);
	}
}

#ifdef B3_USE_CLEW

int b3OpenCLUtils_clewInit()
{
	int result = -1;

#ifdef _WIN32
	const char* cl = "OpenCL.dll";
#elif defined __APPLE__
	const char* cl = "/System/Library/Frameworks/OpenCL.framework/Versions/Current/OpenCL";
#else  //presumable Linux? \
	   //linux (tested on Ubuntu 12.10 with Catalyst 13.4 beta drivers, not that there is no symbolic link from libOpenCL.so
	const char* cl = "libOpenCL.so.1";
	result = clewInit(cl);
	if (result != CLEW_SUCCESS)
	{
		cl = "libOpenCL.so";
	}
	else
	{
		clewExit();
	}
#endif
	result = clewInit(cl);
	if (result != CLEW_SUCCESS)
	{
		b3Error("clewInit failed with error code %d\n", result);
	}
	else
	{
		b3Printf("clewInit succesfull using %s\n", cl);
	}
	return result;
}
#endif

int b3OpenCLUtils_getNumPlatforms(cl_int* pErrNum)
{
#ifdef B3_USE_CLEW
	b3OpenCLUtils_clewInit();
#endif

	cl_platform_id pPlatforms[10] = {0};

	cl_uint numPlatforms = 0;
	cl_int ciErrNum = clGetPlatformIDs(10, pPlatforms, &numPlatforms);
	//cl_int ciErrNum = clGetPlatformIDs(0, NULL, &numPlatforms);

	if (ciErrNum != CL_SUCCESS)
	{
		if (pErrNum != NULL)
			*pErrNum = ciErrNum;
	}
	return numPlatforms;
}

const char* b3OpenCLUtils_getSdkVendorName()
{
	return spPlatformVendor;
}

void b3OpenCLUtils_setCachePath(const char* path)
{
	sCachedBinaryPath = path;
}

cl_platform_id b3OpenCLUtils_getPlatform(int platformIndex0, cl_int* pErrNum)
{
#ifdef B3_USE_CLEW
	b3OpenCLUtils_clewInit();
#endif

	cl_platform_id platform = 0;
	unsigned int platformIndex = (unsigned int)platformIndex0;
	cl_uint numPlatforms;
	cl_int ciErrNum = clGetPlatformIDs(0, NULL, &numPlatforms);

	if (platformIndex < numPlatforms)
	{
		cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
		ciErrNum = clGetPlatformIDs(numPlatforms, platforms, NULL);
		if (ciErrNum != CL_SUCCESS)
		{
			if (pErrNum != NULL)
				*pErrNum = ciErrNum;
			return platform;
		}

		platform = platforms[platformIndex];

		free(platforms);
	}

	return platform;
}

void b3OpenCLUtils::getPlatformInfo(cl_platform_id platform, b3OpenCLPlatformInfo* platformInfo)
{
	b3Assert(platform);
	cl_int ciErrNum;
	ciErrNum = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, B3_MAX_STRING_LENGTH, platformInfo->m_platformVendor, NULL);
	oclCHECKERROR(ciErrNum, CL_SUCCESS);
	ciErrNum = clGetPlatformInfo(platform, CL_PLATFORM_NAME, B3_MAX_STRING_LENGTH, platformInfo->m_platformName, NULL);
	oclCHECKERROR(ciErrNum, CL_SUCCESS);
	ciErrNum = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, B3_MAX_STRING_LENGTH, platformInfo->m_platformVersion, NULL);
	oclCHECKERROR(ciErrNum, CL_SUCCESS);
}

void b3OpenCLUtils_printPlatformInfo(cl_platform_id platform)
{
	b3OpenCLPlatformInfo platformInfo;
	b3OpenCLUtils::getPlatformInfo(platform, &platformInfo);
	b3Printf("Platform info:\n");
	b3Printf("  CL_PLATFORM_VENDOR: \t\t\t%s\n", platformInfo.m_platformVendor);
	b3Printf("  CL_PLATFORM_NAME: \t\t\t%s\n", platformInfo.m_platformName);
	b3Printf("  CL_PLATFORM_VERSION: \t\t\t%s\n", platformInfo.m_platformVersion);
}

cl_context b3OpenCLUtils_createContextFromPlatform(cl_platform_id platform, cl_device_type deviceType, cl_int* pErrNum, void* pGLContext, void* pGLDC, int preferredDeviceIndex, int preferredPlatformIndex)
{
	cl_context retContext = 0;
	cl_int ciErrNum = 0;
	cl_uint num_entries;
	cl_device_id devices[B3_MAX_CL_DEVICES];
	cl_uint num_devices;
	cl_context_properties* cprops;

	/*
	* If we could find our platform, use it. Otherwise pass a NULL and get whatever the
	* implementation thinks we should be using.
	*/
	cl_context_properties cps[7] = {0, 0, 0, 0, 0, 0, 0};
	cps[0] = CL_CONTEXT_PLATFORM;
	cps[1] = (cl_context_properties)platform;
#ifdef _WIN32
#ifndef B3_USE_CLEW
	if (pGLContext && pGLDC)
	{
		cps[2] = CL_GL_CONTEXT_KHR;
		cps[3] = (cl_context_properties)pGLContext;
		cps[4] = CL_WGL_HDC_KHR;
		cps[5] = (cl_context_properties)pGLDC;
	}
#endif  //B3_USE_CLEW
#endif  //_WIN32
	num_entries = B3_MAX_CL_DEVICES;

	num_devices = -1;

	ciErrNum = clGetDeviceIDs(
		platform,
		deviceType,
		num_entries,
		devices,
		&num_devices);

	if (ciErrNum < 0)
	{
		b3Printf("clGetDeviceIDs returned %d\n", ciErrNum);
		return 0;
	}
	cprops = (NULL == platform) ? NULL : cps;

	if (!num_devices)
		return 0;

	if (pGLContext)
	{
		//search for the GPU that relates to the OpenCL context
		unsigned int i;
		for (i = 0; i < num_devices; i++)
		{
			retContext = clCreateContext(cprops, 1, &devices[i], NULL, NULL, &ciErrNum);
			if (ciErrNum == CL_SUCCESS)
				break;
		}
	}
	else
	{
		if (preferredDeviceIndex >= 0 && (unsigned int)preferredDeviceIndex < num_devices)
		{
			//create a context of the preferred device index
			retContext = clCreateContext(cprops, 1, &devices[preferredDeviceIndex], NULL, NULL, &ciErrNum);
		}
		else
		{
			//create a context of all devices
#if defined(__APPLE__)
			retContext = clCreateContext(cprops, num_devices, devices, MyFatalBreakAPPLE, NULL, &ciErrNum);
#else
			b3Printf("numDevices=%d\n", num_devices);

			retContext = clCreateContext(cprops, num_devices, devices, NULL, NULL, &ciErrNum);
#endif
		}
	}
	if (pErrNum != NULL)
	{
		*pErrNum = ciErrNum;
	};

	return retContext;
}

cl_context b3OpenCLUtils_createContextFromType(cl_device_type deviceType, cl_int* pErrNum, void* pGLContext, void* pGLDC, int preferredDeviceIndex, int preferredPlatformIndex, cl_platform_id* retPlatformId)
{
#ifdef B3_USE_CLEW
	b3OpenCLUtils_clewInit();
#endif

	cl_uint numPlatforms;
	cl_context retContext = 0;
	unsigned int i;

	cl_int ciErrNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (ciErrNum != CL_SUCCESS)
	{
		if (pErrNum != NULL) *pErrNum = ciErrNum;
		return NULL;
	}
	if (numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
		ciErrNum = clGetPlatformIDs(numPlatforms, platforms, NULL);
		if (ciErrNum != CL_SUCCESS)
		{
			if (pErrNum != NULL)
				*pErrNum = ciErrNum;
			free(platforms);
			return NULL;
		}

		for (i = 0; i < numPlatforms; ++i)
		{
			char pbuf[128];
			ciErrNum = clGetPlatformInfo(platforms[i],
										 CL_PLATFORM_VENDOR,
										 sizeof(pbuf),
										 pbuf,
										 NULL);
			if (ciErrNum != CL_SUCCESS)
			{
				if (pErrNum != NULL) *pErrNum = ciErrNum;
				return NULL;
			}

			if (preferredPlatformIndex >= 0 && i == preferredPlatformIndex)
			{
				cl_platform_id tmpPlatform = platforms[0];
				platforms[0] = platforms[i];
				platforms[i] = tmpPlatform;
				break;
			}
			else
			{
				if (!strcmp(pbuf, spPlatformVendor))
				{
					cl_platform_id tmpPlatform = platforms[0];
					platforms[0] = platforms[i];
					platforms[i] = tmpPlatform;
				}
			}
		}

		for (i = 0; i < numPlatforms; ++i)
		{
			cl_platform_id platform = platforms[i];
			assert(platform);

			retContext = b3OpenCLUtils_createContextFromPlatform(platform, deviceType, pErrNum, pGLContext, pGLDC, preferredDeviceIndex, preferredPlatformIndex);

			if (retContext)
			{
				//				printf("OpenCL platform details:\n");
				b3OpenCLPlatformInfo platformInfo;

				b3OpenCLUtils::getPlatformInfo(platform, &platformInfo);

				if (retPlatformId)
					*retPlatformId = platform;

				break;
			}
		}

		free(platforms);
	}
	return retContext;
}

//////////////////////////////////////////////////////////////////////////////
//! Gets the id of the nth device from the context
//!
//! @return the id or -1 when out of range
//! @param cxMainContext         OpenCL context
//! @param device_idx            index of the device of interest
//////////////////////////////////////////////////////////////////////////////
cl_device_id b3OpenCLUtils_getDevice(cl_context cxMainContext, int deviceIndex)
{
	assert(cxMainContext);

	size_t szParmDataBytes;
	cl_device_id* cdDevices;
	cl_device_id device;

	// get the list of devices associated with context
	clGetContextInfo(cxMainContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);

	if (szParmDataBytes / sizeof(cl_device_id) < (unsigned int)deviceIndex)
	{
		return (cl_device_id)-1;
	}

	cdDevices = (cl_device_id*)malloc(szParmDataBytes);

	clGetContextInfo(cxMainContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

	device = cdDevices[deviceIndex];
	free(cdDevices);

	return device;
}

int b3OpenCLUtils_getNumDevices(cl_context cxMainContext)
{
	size_t szParamDataBytes;
	int device_count;
	clGetContextInfo(cxMainContext, CL_CONTEXT_DEVICES, 0, NULL, &szParamDataBytes);
	device_count = (int)szParamDataBytes / sizeof(cl_device_id);
	return device_count;
}

void b3OpenCLUtils::getDeviceInfo(cl_device_id device, b3OpenCLDeviceInfo* info)
{
	// CL_DEVICE_NAME
	clGetDeviceInfo(device, CL_DEVICE_NAME, B3_MAX_STRING_LENGTH, &info->m_deviceName, NULL);

	// CL_DEVICE_VENDOR
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, B3_MAX_STRING_LENGTH, &info->m_deviceVendor, NULL);

	// CL_DRIVER_VERSION
	clGetDeviceInfo(device, CL_DRIVER_VERSION, B3_MAX_STRING_LENGTH, &info->m_driverVersion, NULL);

	// CL_DEVICE_INFO
	clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &info->m_deviceType, NULL);

	// CL_DEVICE_MAX_COMPUTE_UNITS
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info->m_computeUnits), &info->m_computeUnits, NULL);

	// CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(info->m_workitemDims), &info->m_workitemDims, NULL);

	// CL_DEVICE_MAX_WORK_ITEM_SIZES
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(info->m_workItemSize), &info->m_workItemSize, NULL);

	// CL_DEVICE_MAX_WORK_GROUP_SIZE
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info->m_workgroupSize), &info->m_workgroupSize, NULL);

	// CL_DEVICE_MAX_CLOCK_FREQUENCY
	clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(info->m_clockFrequency), &info->m_clockFrequency, NULL);

	// CL_DEVICE_ADDRESS_BITS
	clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS, sizeof(info->m_addressBits), &info->m_addressBits, NULL);

	// CL_DEVICE_MAX_MEM_ALLOC_SIZE
	clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(info->m_maxMemAllocSize), &info->m_maxMemAllocSize, NULL);

	// CL_DEVICE_GLOBAL_MEM_SIZE
	clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info->m_globalMemSize), &info->m_globalMemSize, NULL);

	// CL_DEVICE_ERROR_CORRECTION_SUPPORT
	clGetDeviceInfo(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(info->m_errorCorrectionSupport), &info->m_errorCorrectionSupport, NULL);

	// CL_DEVICE_LOCAL_MEM_TYPE
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(info->m_localMemType), &info->m_localMemType, NULL);

	// CL_DEVICE_LOCAL_MEM_SIZE
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(info->m_localMemSize), &info->m_localMemSize, NULL);

	// CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
	clGetDeviceInfo(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(info->m_constantBufferSize), &info->m_constantBufferSize, NULL);

	// CL_DEVICE_QUEUE_PROPERTIES
	clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(info->m_queueProperties), &info->m_queueProperties, NULL);

	// CL_DEVICE_IMAGE_SUPPORT
	clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(info->m_imageSupport), &info->m_imageSupport, NULL);

	// CL_DEVICE_MAX_READ_IMAGE_ARGS
	clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(info->m_maxReadImageArgs), &info->m_maxReadImageArgs, NULL);

	// CL_DEVICE_MAX_WRITE_IMAGE_ARGS
	clGetDeviceInfo(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(info->m_maxWriteImageArgs), &info->m_maxWriteImageArgs, NULL);

	// CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT, CL_DEVICE_IMAGE3D_MAX_DEPTH
	clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &info->m_image2dMaxWidth, NULL);
	clGetDeviceInfo(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &info->m_image2dMaxHeight, NULL);
	clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), &info->m_image3dMaxWidth, NULL);
	clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), &info->m_image3dMaxHeight, NULL);
	clGetDeviceInfo(device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), &info->m_image3dMaxDepth, NULL);

	// CL_DEVICE_EXTENSIONS: get device extensions, and if any then parse & log the string onto separate lines
	clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, B3_MAX_STRING_LENGTH, &info->m_deviceExtensions, NULL);

	// CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), &info->m_vecWidthChar, NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), &info->m_vecWidthShort, NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), &info->m_vecWidthInt, NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), &info->m_vecWidthLong, NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), &info->m_vecWidthFloat, NULL);
	clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &info->m_vecWidthDouble, NULL);
}

void b3OpenCLUtils_printDeviceInfo(cl_device_id device)
{
	b3OpenCLDeviceInfo info;
	b3OpenCLUtils::getDeviceInfo(device, &info);
	b3Printf("Device Info:\n");
	b3Printf("  CL_DEVICE_NAME: \t\t\t%s\n", info.m_deviceName);
	b3Printf("  CL_DEVICE_VENDOR: \t\t\t%s\n", info.m_deviceVendor);
	b3Printf("  CL_DRIVER_VERSION: \t\t\t%s\n", info.m_driverVersion);

	if (info.m_deviceType & CL_DEVICE_TYPE_CPU)
		b3Printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_CPU");
	if (info.m_deviceType & CL_DEVICE_TYPE_GPU)
		b3Printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
	if (info.m_deviceType & CL_DEVICE_TYPE_ACCELERATOR)
		b3Printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_ACCELERATOR");
	if (info.m_deviceType & CL_DEVICE_TYPE_DEFAULT)
		b3Printf("  CL_DEVICE_TYPE:\t\t\t%s\n", "CL_DEVICE_TYPE_DEFAULT");

	b3Printf("  CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%u\n", info.m_computeUnits);
	b3Printf("  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%u\n", info.m_workitemDims);
	b3Printf("  CL_DEVICE_MAX_WORK_ITEM_SIZES:\t%u / %u / %u \n", info.m_workItemSize[0], info.m_workItemSize[1], info.m_workItemSize[2]);
	b3Printf("  CL_DEVICE_MAX_WORK_GROUP_SIZE:\t%u\n", info.m_workgroupSize);
	b3Printf("  CL_DEVICE_MAX_CLOCK_FREQUENCY:\t%u MHz\n", info.m_clockFrequency);
	b3Printf("  CL_DEVICE_ADDRESS_BITS:\t\t%u\n", info.m_addressBits);
	b3Printf("  CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%u MByte\n", (unsigned int)(info.m_maxMemAllocSize / (1024 * 1024)));
	b3Printf("  CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%u MByte\n", (unsigned int)(info.m_globalMemSize / (1024 * 1024)));
	b3Printf("  CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", info.m_errorCorrectionSupport == CL_TRUE ? "yes" : "no");
	b3Printf("  CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", info.m_localMemType == 1 ? "local" : "global");
	b3Printf("  CL_DEVICE_LOCAL_MEM_SIZE:\t\t%u KByte\n", (unsigned int)(info.m_localMemSize / 1024));
	b3Printf("  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%u KByte\n", (unsigned int)(info.m_constantBufferSize / 1024));
	if (info.m_queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
		b3Printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
	if (info.m_queueProperties & CL_QUEUE_PROFILING_ENABLE)
		b3Printf("  CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_PROFILING_ENABLE");

	b3Printf("  CL_DEVICE_IMAGE_SUPPORT:\t\t%u\n", info.m_imageSupport);

	b3Printf("  CL_DEVICE_MAX_READ_IMAGE_ARGS:\t%u\n", info.m_maxReadImageArgs);
	b3Printf("  CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t%u\n", info.m_maxWriteImageArgs);
	b3Printf("\n  CL_DEVICE_IMAGE <dim>");
	b3Printf("\t\t\t2D_MAX_WIDTH\t %u\n", info.m_image2dMaxWidth);
	b3Printf("\t\t\t\t\t2D_MAX_HEIGHT\t %u\n", info.m_image2dMaxHeight);
	b3Printf("\t\t\t\t\t3D_MAX_WIDTH\t %u\n", info.m_image3dMaxWidth);
	b3Printf("\t\t\t\t\t3D_MAX_HEIGHT\t %u\n", info.m_image3dMaxHeight);
	b3Printf("\t\t\t\t\t3D_MAX_DEPTH\t %u\n", info.m_image3dMaxDepth);
	if (*info.m_deviceExtensions != 0)
	{
		b3Printf("\n  CL_DEVICE_EXTENSIONS:%s\n", info.m_deviceExtensions);
	}
	else
	{
		b3Printf("  CL_DEVICE_EXTENSIONS: None\n");
	}
	b3Printf("  CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t");
	b3Printf("CHAR %u, SHORT %u, INT %u,LONG %u, FLOAT %u, DOUBLE %u\n\n\n",
			 info.m_vecWidthChar, info.m_vecWidthShort, info.m_vecWidthInt, info.m_vecWidthLong, info.m_vecWidthFloat, info.m_vecWidthDouble);
}

static const char* strip2(const char* name, const char* pattern)
{
	size_t const patlen = strlen(pattern);
	size_t patcnt = 0;
	const char* oriptr;
	const char* patloc;
	// find how many times the pattern occurs in the original string
	for (oriptr = name; (patloc = strstr(oriptr, pattern)); oriptr = patloc + patlen)
	{
		patcnt++;
	}
	return oriptr;
}

cl_program b3OpenCLUtils_compileCLProgramFromString(cl_context clContext, cl_device_id device, const char* kernelSourceOrg, cl_int* pErrNum, const char* additionalMacrosArg, const char* clFileNameForCaching, bool disableBinaryCaching)
{
	const char* additionalMacros = additionalMacrosArg ? additionalMacrosArg : "";

	if (disableBinaryCaching)
	{
		//kernelSourceOrg = 0;
	}

	cl_program m_cpProgram = 0;
	cl_int status;

	char binaryFileName[B3_MAX_STRING_LENGTH];

	char deviceName[256];
	char driverVersion[256];
	const char* strippedName;
	int fileUpToDate = 0;
#ifdef _WIN32
	int binaryFileValid = 0;
#endif
	if (!disableBinaryCaching && clFileNameForCaching)
	{
		clGetDeviceInfo(device, CL_DEVICE_NAME, 256, &deviceName, NULL);
		clGetDeviceInfo(device, CL_DRIVER_VERSION, 256, &driverVersion, NULL);

		strippedName = strip2(clFileNameForCaching, "\\");
		strippedName = strip2(strippedName, "/");

#ifdef _MSC_VER
		sprintf_s(binaryFileName, B3_MAX_STRING_LENGTH, "%s/%s.%s.%s.bin", sCachedBinaryPath, strippedName, deviceName, driverVersion);
#else
		sprintf(binaryFileName, "%s/%s.%s.%s.bin", sCachedBinaryPath, strippedName, deviceName, driverVersion);
#endif
	}
	if (clFileNameForCaching && !(disableBinaryCaching || gDebugSkipLoadingBinary || gDebugForceLoadingFromSource))
	{
#ifdef _WIN32
		char* bla = 0;

		//printf("searching for %s\n", binaryFileName);

		FILETIME modtimeBinary;
		CreateDirectoryA(sCachedBinaryPath, 0);
		{
			HANDLE binaryFileHandle = CreateFileA(binaryFileName, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
			if (binaryFileHandle == INVALID_HANDLE_VALUE)
			{
				DWORD errorCode;
				errorCode = GetLastError();
				switch (errorCode)
				{
					case ERROR_FILE_NOT_FOUND:
					{
						b3Warning("\nCached file not found %s\n", binaryFileName);
						break;
					}
					case ERROR_PATH_NOT_FOUND:
					{
						b3Warning("\nCached file path not found %s\n", binaryFileName);
						break;
					}
					default:
					{
						b3Warning("\nFailed reading cached file with errorCode = %d\n", errorCode);
					}
				}
			}
			else
			{
				if (GetFileTime(binaryFileHandle, NULL, NULL, &modtimeBinary) == 0)
				{
					DWORD errorCode;
					errorCode = GetLastError();
					b3Warning("\nGetFileTime errorCode = %d\n", errorCode);
				}
				else
				{
					binaryFileValid = 1;
				}
				CloseHandle(binaryFileHandle);
			}

			if (binaryFileValid)
			{
				HANDLE srcFileHandle = CreateFileA(clFileNameForCaching, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);

				if (srcFileHandle == INVALID_HANDLE_VALUE)
				{
					const char* prefix[] = {"./", "../", "../../", "../../../", "../../../../"};
					for (int i = 0; (srcFileHandle == INVALID_HANDLE_VALUE) && i < 5; i++)
					{
						char relativeFileName[1024];
						sprintf(relativeFileName, "%s%s", prefix[i], clFileNameForCaching);
						srcFileHandle = CreateFileA(relativeFileName, GENERIC_READ, 0, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
					}
				}

				if (srcFileHandle != INVALID_HANDLE_VALUE)
				{
					FILETIME modtimeSrc;
					if (GetFileTime(srcFileHandle, NULL, NULL, &modtimeSrc) == 0)
					{
						DWORD errorCode;
						errorCode = GetLastError();
						b3Warning("\nGetFileTime errorCode = %d\n", errorCode);
					}
					if ((modtimeSrc.dwHighDateTime < modtimeBinary.dwHighDateTime) || ((modtimeSrc.dwHighDateTime == modtimeBinary.dwHighDateTime) && (modtimeSrc.dwLowDateTime <= modtimeBinary.dwLowDateTime)))
					{
						fileUpToDate = 1;
					}
					else
					{
						b3Warning("\nCached binary file out-of-date (%s)\n", binaryFileName);
					}
					CloseHandle(srcFileHandle);
				}
				else
				{
#ifdef _DEBUG
					DWORD errorCode;
					errorCode = GetLastError();
					switch (errorCode)
					{
						case ERROR_FILE_NOT_FOUND:
						{
							b3Warning("\nSrc file not found %s\n", clFileNameForCaching);
							break;
						}
						case ERROR_PATH_NOT_FOUND:
						{
							b3Warning("\nSrc path not found %s\n", clFileNameForCaching);
							break;
						}
						default:
						{
							b3Warning("\nnSrc file reading errorCode = %d\n", errorCode);
						}
					}

					//we should make sure the src file exists so we can verify the timestamp with binary
					//					assert(0);
					b3Warning("Warning: cannot find OpenCL kernel %s to verify timestamp of binary cached kernel %s\n", clFileNameForCaching, binaryFileName);
					fileUpToDate = true;
#else
					//if we cannot find the source, assume it is OK in release builds
					fileUpToDate = true;
#endif
				}
			}
		}

#else
		fileUpToDate = true;
		if (mkdir(sCachedBinaryPath, 0777) == -1)
		{
		}
		else
		{
			b3Printf("Succesfully created cache directory: %s\n", sCachedBinaryPath);
		}
#endif  //_WIN32
	}

	if (fileUpToDate)
	{
#ifdef _MSC_VER
		FILE* file;
		if (fopen_s(&file, binaryFileName, "rb") != 0)
			file = 0;
#else
		FILE* file = fopen(binaryFileName, "rb");
#endif

		if (file)
		{
			size_t binarySize = 0;
			char* binary = 0;

			fseek(file, 0L, SEEK_END);
			binarySize = ftell(file);
			rewind(file);
			binary = (char*)malloc(sizeof(char) * binarySize);
			int bytesRead;
			bytesRead = fread(binary, sizeof(char), binarySize, file);
			fclose(file);

			m_cpProgram = clCreateProgramWithBinary(clContext, 1, &device, &binarySize, (const unsigned char**)&binary, 0, &status);
			b3Assert(status == CL_SUCCESS);
			status = clBuildProgram(m_cpProgram, 1, &device, additionalMacros, 0, 0);
			b3Assert(status == CL_SUCCESS);

			if (status != CL_SUCCESS)
			{
				char* build_log;
				size_t ret_val_size;
				clGetProgramBuildInfo(m_cpProgram, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
				build_log = (char*)malloc(sizeof(char) * (ret_val_size + 1));
				clGetProgramBuildInfo(m_cpProgram, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
				build_log[ret_val_size] = '\0';
				b3Error("%s\n", build_log);
				free(build_log);
				b3Assert(0);
				m_cpProgram = 0;

				b3Warning("clBuildProgram reported failure on cached binary: %s\n", binaryFileName);
			}
			else
			{
				b3Printf("clBuildProgram successfully compiled cached binary: %s\n", binaryFileName);
			}
			free(binary);
		}
		else
		{
			b3Warning("Cannot open cached binary: %s\n", binaryFileName);
		}
	}

	if (!m_cpProgram)
	{
		cl_int localErrNum;
		char* compileFlags;
		int flagsize;

		const char* kernelSource = kernelSourceOrg;

		if (!kernelSourceOrg || gDebugForceLoadingFromSource)
		{
			if (clFileNameForCaching)
			{
				FILE* file = fopen(clFileNameForCaching, "rb");
				//in many cases the relative path is a few levels up the directory hierarchy, so try it
				if (!file)
				{
					const char* prefix[] = {"../", "../../", "../../../", "../../../../"};
					for (int i = 0; !file && i < 3; i++)
					{
						char relativeFileName[1024];
						sprintf(relativeFileName, "%s%s", prefix[i], clFileNameForCaching);
						file = fopen(relativeFileName, "rb");
					}
				}

				if (file)
				{
					char* kernelSrc = 0;
					fseek(file, 0L, SEEK_END);
					int kernelSize = ftell(file);
					rewind(file);
					kernelSrc = (char*)malloc(kernelSize + 1);
					int readBytes;
					readBytes = fread((void*)kernelSrc, 1, kernelSize, file);
					kernelSrc[kernelSize] = 0;
					fclose(file);
					kernelSource = kernelSrc;
				}
			}
		}

		size_t program_length = kernelSource ? strlen(kernelSource) : 0;
#ifdef MAC  //or __APPLE__?
		char* flags = "-cl-mad-enable -DMAC ";
#else
		const char* flags = "";
#endif

		m_cpProgram = clCreateProgramWithSource(clContext, 1, (const char**)&kernelSource, &program_length, &localErrNum);
		if (localErrNum != CL_SUCCESS)
		{
			if (pErrNum)
				*pErrNum = localErrNum;
			return 0;
		}

		// Build the program with 'mad' Optimization option

		flagsize = sizeof(char) * (strlen(additionalMacros) + strlen(flags) + 5);
		compileFlags = (char*)malloc(flagsize);
#ifdef _MSC_VER
		sprintf_s(compileFlags, flagsize, "%s %s", flags, additionalMacros);
#else
		sprintf(compileFlags, "%s %s", flags, additionalMacros);
#endif
		localErrNum = clBuildProgram(m_cpProgram, 1, &device, compileFlags, NULL, NULL);
		if (localErrNum != CL_SUCCESS)
		{
			char* build_log;
			size_t ret_val_size;
			clGetProgramBuildInfo(m_cpProgram, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
			build_log = (char*)malloc(sizeof(char) * (ret_val_size + 1));
			clGetProgramBuildInfo(m_cpProgram, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

			// to be carefully, terminate with \0
			// there's no information in the reference whether the string is 0 terminated or not
			build_log[ret_val_size] = '\0';

			b3Error("Error in clBuildProgram, Line %u in file %s, Log: \n%s\n !!!\n\n", __LINE__, __FILE__, build_log);
			free(build_log);
			if (pErrNum)
				*pErrNum = localErrNum;
			return 0;
		}

		if (!disableBinaryCaching && clFileNameForCaching)
		{  //	write to binary

			cl_uint numAssociatedDevices;
			status = clGetProgramInfo(m_cpProgram, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numAssociatedDevices, 0);
			b3Assert(status == CL_SUCCESS);
			if (numAssociatedDevices == 1)
			{
				size_t binarySize;
				char* binary;

				status = clGetProgramInfo(m_cpProgram, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binarySize, 0);
				b3Assert(status == CL_SUCCESS);

				binary = (char*)malloc(sizeof(char) * binarySize);

				status = clGetProgramInfo(m_cpProgram, CL_PROGRAM_BINARIES, sizeof(char*), &binary, 0);
				b3Assert(status == CL_SUCCESS);

				{
					FILE* file = 0;
#ifdef _MSC_VER
					if (fopen_s(&file, binaryFileName, "wb") != 0)
						file = 0;
#else
					file = fopen(binaryFileName, "wb");
#endif
					if (file)
					{
						fwrite(binary, sizeof(char), binarySize, file);
						fclose(file);
					}
					else
					{
						b3Warning("cannot write file %s\n", binaryFileName);
					}
				}

				free(binary);
			}
		}

		free(compileFlags);
	}
	return m_cpProgram;
}

cl_kernel b3OpenCLUtils_compileCLKernelFromString(cl_context clContext, cl_device_id device, const char* kernelSource, const char* kernelName, cl_int* pErrNum, cl_program prog, const char* additionalMacros)
{
	cl_kernel kernel;
	cl_int localErrNum;

	cl_program m_cpProgram = prog;

	b3Printf("compiling kernel %s ", kernelName);

	if (!m_cpProgram)
	{
		m_cpProgram = b3OpenCLUtils_compileCLProgramFromString(clContext, device, kernelSource, pErrNum, additionalMacros, 0, false);
	}

	// Create the kernel
	kernel = clCreateKernel(m_cpProgram, kernelName, &localErrNum);
	if (localErrNum != CL_SUCCESS)
	{
		b3Error("Error in clCreateKernel, Line %u in file %s, cannot find kernel function %s !!!\n\n", __LINE__, __FILE__, kernelName);
		assert(0);
		if (pErrNum)
			*pErrNum = localErrNum;
		return 0;
	}

	if (!prog && m_cpProgram)
	{
		clReleaseProgram(m_cpProgram);
	}
	b3Printf("ready. \n");

	if (pErrNum)
		*pErrNum = CL_SUCCESS;
	return kernel;
}
