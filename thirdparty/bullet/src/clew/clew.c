//////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2009 Organic Vectory B.V.
//  Written by George van Venrooij
//
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file license.txt)
//////////////////////////////////////////////////////////////////////////

#include "clew.h"

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #define VC_EXTRALEAN
    #include <windows.h>

    typedef HMODULE             CLEW_DYNLIB_HANDLE;

    #define CLEW_DYNLIB_OPEN    LoadLibrary
    #define CLEW_DYNLIB_CLOSE   FreeLibrary
    #define CLEW_DYNLIB_IMPORT  GetProcAddress
#else
    #include <dlfcn.h>
    
    typedef void*                   CLEW_DYNLIB_HANDLE;

    #define CLEW_DYNLIB_OPEN(path)  dlopen(path, RTLD_NOW | RTLD_GLOBAL)
    #define CLEW_DYNLIB_CLOSE       dlclose
    #define CLEW_DYNLIB_IMPORT      dlsym
#endif

#include <stdlib.h>

//! \brief module handle
static CLEW_DYNLIB_HANDLE module = NULL;

//  Variables holding function entry points
PFNCLGETPLATFORMIDS                 __clewGetPlatformIDs                = NULL;
PFNCLGETPLATFORMINFO                __clewGetPlatformInfo               = NULL;
PFNCLGETDEVICEIDS                   __clewGetDeviceIDs                  = NULL;
PFNCLGETDEVICEINFO                  __clewGetDeviceInfo                 = NULL;
PFNCLCREATECONTEXT                  __clewCreateContext                 = NULL;
PFNCLCREATECONTEXTFROMTYPE          __clewCreateContextFromType         = NULL;
PFNCLRETAINCONTEXT                  __clewRetainContext                 = NULL;
PFNCLRELEASECONTEXT                 __clewReleaseContext                = NULL;
PFNCLGETCONTEXTINFO                 __clewGetContextInfo                = NULL;
PFNCLCREATECOMMANDQUEUE             __clewCreateCommandQueue            = NULL;
PFNCLRETAINCOMMANDQUEUE             __clewRetainCommandQueue            = NULL;
PFNCLRELEASECOMMANDQUEUE            __clewReleaseCommandQueue           = NULL;
PFNCLGETCOMMANDQUEUEINFO            __clewGetCommandQueueInfo           = NULL;
#ifdef CL_USE_DEPRECATED_OPENCL_1_0_APIS
PFNCLSETCOMMANDQUEUEPROPERTY        __clewSetCommandQueueProperty       = NULL;
#endif
PFNCLCREATEBUFFER                   __clewCreateBuffer                  = NULL;
PFNCLCREATESUBBUFFER                __clewCreateSubBuffer               = NULL;
PFNCLCREATEIMAGE2D                  __clewCreateImage2D                 = NULL;
PFNCLCREATEIMAGE3D                  __clewCreateImage3D                 = NULL;
PFNCLRETAINMEMOBJECT                __clewRetainMemObject               = NULL;
PFNCLRELEASEMEMOBJECT               __clewReleaseMemObject              = NULL;
PFNCLGETSUPPORTEDIMAGEFORMATS       __clewGetSupportedImageFormats      = NULL;
PFNCLGETMEMOBJECTINFO               __clewGetMemObjectInfo              = NULL;
PFNCLGETIMAGEINFO                   __clewGetImageInfo                  = NULL;
PFNCLSETMEMOBJECTDESTRUCTORCALLBACK __clewSetMemObjectDestructorCallback = NULL;
PFNCLCREATESAMPLER                  __clewCreateSampler                 = NULL;
PFNCLRETAINSAMPLER                  __clewRetainSampler                 = NULL;
PFNCLRELEASESAMPLER                 __clewReleaseSampler                = NULL;
PFNCLGETSAMPLERINFO                 __clewGetSamplerInfo                = NULL;
PFNCLCREATEPROGRAMWITHSOURCE        __clewCreateProgramWithSource       = NULL;
PFNCLCREATEPROGRAMWITHBINARY        __clewCreateProgramWithBinary       = NULL;
PFNCLRETAINPROGRAM                  __clewRetainProgram                 = NULL;
PFNCLRELEASEPROGRAM                 __clewReleaseProgram                = NULL;
PFNCLBUILDPROGRAM                   __clewBuildProgram                  = NULL;
PFNCLUNLOADCOMPILER                 __clewUnloadCompiler                = NULL;
PFNCLGETPROGRAMINFO                 __clewGetProgramInfo                = NULL;
PFNCLGETPROGRAMBUILDINFO            __clewGetProgramBuildInfo           = NULL;
PFNCLCREATEKERNEL                   __clewCreateKernel                  = NULL;
PFNCLCREATEKERNELSINPROGRAM         __clewCreateKernelsInProgram        = NULL;
PFNCLRETAINKERNEL                   __clewRetainKernel                  = NULL;
PFNCLRELEASEKERNEL                  __clewReleaseKernel                 = NULL;
PFNCLSETKERNELARG                   __clewSetKernelArg                  = NULL;
PFNCLGETKERNELINFO                  __clewGetKernelInfo                 = NULL;
PFNCLGETKERNELWORKGROUPINFO         __clewGetKernelWorkGroupInfo        = NULL;
PFNCLWAITFOREVENTS                  __clewWaitForEvents                 = NULL;
PFNCLGETEVENTINFO                   __clewGetEventInfo                  = NULL;
PFNCLCREATEUSEREVENT                __clewCreateUserEvent               = NULL;
PFNCLRETAINEVENT                    __clewRetainEvent                   = NULL;
PFNCLRELEASEEVENT                   __clewReleaseEvent                  = NULL;
PFNCLSETUSEREVENTSTATUS             __clewSetUserEventStatus            = NULL;
PFNCLSETEVENTCALLBACK               __clewSetEventCallback              = NULL;
PFNCLGETEVENTPROFILINGINFO          __clewGetEventProfilingInfo         = NULL;
PFNCLFLUSH                          __clewFlush                         = NULL;
PFNCLFINISH                         __clewFinish                        = NULL;
PFNCLENQUEUEREADBUFFER              __clewEnqueueReadBuffer             = NULL;
PFNCLENQUEUEREADBUFFERRECT          __clewEnqueueReadBufferRect         = NULL;
PFNCLENQUEUEWRITEBUFFER             __clewEnqueueWriteBuffer            = NULL;
PFNCLENQUEUEWRITEBUFFERRECT         __clewEnqueueWriteBufferRect        = NULL;
PFNCLENQUEUECOPYBUFFER              __clewEnqueueCopyBuffer             = NULL;
PFNCLENQUEUEREADIMAGE               __clewEnqueueReadImage              = NULL;
PFNCLENQUEUEWRITEIMAGE              __clewEnqueueWriteImage             = NULL;
PFNCLENQUEUECOPYIMAGE               __clewEnqueueCopyImage              = NULL;
PFNCLENQUEUECOPYBUFFERRECT          __clewEnqueueCopyBufferRect         = NULL;
PFNCLENQUEUECOPYIMAGETOBUFFER       __clewEnqueueCopyImageToBuffer      = NULL;
PFNCLENQUEUECOPYBUFFERTOIMAGE       __clewEnqueueCopyBufferToImage      = NULL;
PFNCLENQUEUEMAPBUFFER               __clewEnqueueMapBuffer              = NULL;
PFNCLENQUEUEMAPIMAGE                __clewEnqueueMapImage               = NULL;
PFNCLENQUEUEUNMAPMEMOBJECT          __clewEnqueueUnmapMemObject         = NULL;
PFNCLENQUEUENDRANGEKERNEL           __clewEnqueueNDRangeKernel          = NULL;
PFNCLENQUEUETASK                    __clewEnqueueTask                   = NULL;
PFNCLENQUEUENATIVEKERNEL            __clewEnqueueNativeKernel           = NULL;
PFNCLENQUEUEMARKER                  __clewEnqueueMarker                 = NULL;
PFNCLENQUEUEWAITFOREVENTS           __clewEnqueueWaitForEvents          = NULL;
PFNCLENQUEUEBARRIER                 __clewEnqueueBarrier                = NULL;
PFNCLGETEXTENSIONFUNCTIONADDRESS    __clewGetExtensionFunctionAddress   = NULL;


void clewExit(void)
{
    if (module != NULL)
    {
        //  Ignore errors
        CLEW_DYNLIB_CLOSE(module);
        module = NULL;
    }
}

int clewInit(const char* path)
{
    int error = 0;

    //  Check if already initialized
    if (module != NULL)
    {
        return CLEW_SUCCESS;
    }

    //  Load library
    module = CLEW_DYNLIB_OPEN(path);

    //  Check for errors
    if (module == NULL)
    {
        return CLEW_ERROR_OPEN_FAILED;
    }

    //  Set unloading
    error = atexit(clewExit);

    if (error)
    {
        //  Failure queuing atexit, shutdown with error
        CLEW_DYNLIB_CLOSE(module);
        module = NULL;

        return CLEW_ERROR_ATEXIT_FAILED;
    }

    //  Determine function entry-points
    __clewGetPlatformIDs                = (PFNCLGETPLATFORMIDS              )CLEW_DYNLIB_IMPORT(module, "clGetPlatformIDs");
    __clewGetPlatformInfo               = (PFNCLGETPLATFORMINFO             )CLEW_DYNLIB_IMPORT(module, "clGetPlatformInfo");
    __clewGetDeviceIDs                  = (PFNCLGETDEVICEIDS                )CLEW_DYNLIB_IMPORT(module, "clGetDeviceIDs");
    __clewGetDeviceInfo                 = (PFNCLGETDEVICEINFO               )CLEW_DYNLIB_IMPORT(module, "clGetDeviceInfo");
    __clewCreateContext                 = (PFNCLCREATECONTEXT               )CLEW_DYNLIB_IMPORT(module, "clCreateContext");
    __clewCreateContextFromType         = (PFNCLCREATECONTEXTFROMTYPE       )CLEW_DYNLIB_IMPORT(module, "clCreateContextFromType");
    __clewRetainContext                 = (PFNCLRETAINCONTEXT               )CLEW_DYNLIB_IMPORT(module, "clRetainContext");
    __clewReleaseContext                = (PFNCLRELEASECONTEXT              )CLEW_DYNLIB_IMPORT(module, "clReleaseContext");
    __clewGetContextInfo                = (PFNCLGETCONTEXTINFO              )CLEW_DYNLIB_IMPORT(module, "clGetContextInfo");
    __clewCreateCommandQueue            = (PFNCLCREATECOMMANDQUEUE          )CLEW_DYNLIB_IMPORT(module, "clCreateCommandQueue");
    __clewRetainCommandQueue            = (PFNCLRETAINCOMMANDQUEUE          )CLEW_DYNLIB_IMPORT(module, "clRetainCommandQueue");
    __clewReleaseCommandQueue           = (PFNCLRELEASECOMMANDQUEUE         )CLEW_DYNLIB_IMPORT(module, "clReleaseCommandQueue");
    __clewGetCommandQueueInfo           = (PFNCLGETCOMMANDQUEUEINFO         )CLEW_DYNLIB_IMPORT(module, "clGetCommandQueueInfo");
#ifdef CL_USE_DEPRECATED_OPENCL_1_0_APIS
    __clewSetCommandQueueProperty       = (PFNCLSETCOMMANDQUEUEPROPERTY     )CLEW_DYNLIB_IMPORT(module, "clSetCommandQueueProperty");
#endif
    __clewCreateBuffer                  = (PFNCLCREATEBUFFER                )CLEW_DYNLIB_IMPORT(module, "clCreateBuffer");
    __clewCreateSubBuffer               = (PFNCLCREATESUBBUFFER                )CLEW_DYNLIB_IMPORT(module, "clCreateBuffer");
    __clewCreateImage2D                 = (PFNCLCREATEIMAGE2D               )CLEW_DYNLIB_IMPORT(module, "clCreateImage2D");
    __clewCreateImage3D                 = (PFNCLCREATEIMAGE3D               )CLEW_DYNLIB_IMPORT(module, "clCreateImage3D");
    __clewRetainMemObject               = (PFNCLRETAINMEMOBJECT             )CLEW_DYNLIB_IMPORT(module, "clRetainMemObject");
    __clewReleaseMemObject              = (PFNCLRELEASEMEMOBJECT            )CLEW_DYNLIB_IMPORT(module, "clReleaseMemObject");
    __clewGetSupportedImageFormats      = (PFNCLGETSUPPORTEDIMAGEFORMATS    )CLEW_DYNLIB_IMPORT(module, "clGetSupportedImageFormats");
    __clewGetMemObjectInfo              = (PFNCLGETMEMOBJECTINFO            )CLEW_DYNLIB_IMPORT(module, "clGetMemObjectInfo");
    __clewGetImageInfo                  = (PFNCLGETIMAGEINFO                )CLEW_DYNLIB_IMPORT(module, "clGetImageInfo");
    __clewSetMemObjectDestructorCallback = (PFNCLSETMEMOBJECTDESTRUCTORCALLBACK)CLEW_DYNLIB_IMPORT(module, "clSetMemObjectDestructorCallback");
    __clewCreateSampler                 = (PFNCLCREATESAMPLER               )CLEW_DYNLIB_IMPORT(module, "clCreateSampler");
    __clewRetainSampler                 = (PFNCLRETAINSAMPLER               )CLEW_DYNLIB_IMPORT(module, "clRetainSampler");
    __clewReleaseSampler                = (PFNCLRELEASESAMPLER              )CLEW_DYNLIB_IMPORT(module, "clReleaseSampler");
    __clewGetSamplerInfo                = (PFNCLGETSAMPLERINFO              )CLEW_DYNLIB_IMPORT(module, "clGetSamplerInfo");
    __clewCreateProgramWithSource       = (PFNCLCREATEPROGRAMWITHSOURCE     )CLEW_DYNLIB_IMPORT(module, "clCreateProgramWithSource");
    __clewCreateProgramWithBinary       = (PFNCLCREATEPROGRAMWITHBINARY     )CLEW_DYNLIB_IMPORT(module, "clCreateProgramWithBinary");
    __clewRetainProgram                 = (PFNCLRETAINPROGRAM               )CLEW_DYNLIB_IMPORT(module, "clRetainProgram");
    __clewReleaseProgram                = (PFNCLRELEASEPROGRAM              )CLEW_DYNLIB_IMPORT(module, "clReleaseProgram");
    __clewBuildProgram                  = (PFNCLBUILDPROGRAM                )CLEW_DYNLIB_IMPORT(module, "clBuildProgram");
    __clewUnloadCompiler                = (PFNCLUNLOADCOMPILER              )CLEW_DYNLIB_IMPORT(module, "clUnloadCompiler");
    __clewGetProgramInfo                = (PFNCLGETPROGRAMINFO              )CLEW_DYNLIB_IMPORT(module, "clGetProgramInfo");
    __clewGetProgramBuildInfo           = (PFNCLGETPROGRAMBUILDINFO         )CLEW_DYNLIB_IMPORT(module, "clGetProgramBuildInfo");
    __clewCreateKernel                  = (PFNCLCREATEKERNEL                )CLEW_DYNLIB_IMPORT(module, "clCreateKernel");
    __clewCreateKernelsInProgram        = (PFNCLCREATEKERNELSINPROGRAM      )CLEW_DYNLIB_IMPORT(module, "clCreateKernelsInProgram");
    __clewRetainKernel                  = (PFNCLRETAINKERNEL                )CLEW_DYNLIB_IMPORT(module, "clRetainKernel");
    __clewReleaseKernel                 = (PFNCLRELEASEKERNEL               )CLEW_DYNLIB_IMPORT(module, "clReleaseKernel");
    __clewSetKernelArg                  = (PFNCLSETKERNELARG                )CLEW_DYNLIB_IMPORT(module, "clSetKernelArg");
    __clewGetKernelInfo                 = (PFNCLGETKERNELINFO               )CLEW_DYNLIB_IMPORT(module, "clGetKernelInfo");
    __clewGetKernelWorkGroupInfo        = (PFNCLGETKERNELWORKGROUPINFO      )CLEW_DYNLIB_IMPORT(module, "clGetKernelWorkGroupInfo");
    __clewWaitForEvents                 = (PFNCLWAITFOREVENTS               )CLEW_DYNLIB_IMPORT(module, "clWaitForEvents");
    __clewGetEventInfo                  = (PFNCLGETEVENTINFO                )CLEW_DYNLIB_IMPORT(module, "clGetEventInfo");
    __clewCreateUserEvent               = (PFNCLCREATEUSEREVENT             )CLEW_DYNLIB_IMPORT(module, "clCreateUserEvent");
    __clewRetainEvent                   = (PFNCLRETAINEVENT                 )CLEW_DYNLIB_IMPORT(module, "clRetainEvent");
    __clewReleaseEvent                  = (PFNCLRELEASEEVENT                )CLEW_DYNLIB_IMPORT(module, "clReleaseEvent");
    __clewSetUserEventStatus            = (PFNCLSETUSEREVENTSTATUS          )CLEW_DYNLIB_IMPORT(module, "clSetUserEventStatus");
    __clewSetEventCallback              = (PFNCLSETEVENTCALLBACK            )CLEW_DYNLIB_IMPORT(module, "clSetEventCallback");
    __clewGetEventProfilingInfo         = (PFNCLGETEVENTPROFILINGINFO       )CLEW_DYNLIB_IMPORT(module, "clGetEventProfilingInfo");
    __clewFlush                         = (PFNCLFLUSH                       )CLEW_DYNLIB_IMPORT(module, "clFlush");
    __clewFinish                        = (PFNCLFINISH                      )CLEW_DYNLIB_IMPORT(module, "clFinish");
    __clewEnqueueReadBuffer             = (PFNCLENQUEUEREADBUFFER           )CLEW_DYNLIB_IMPORT(module, "clEnqueueReadBuffer");
    __clewEnqueueReadBufferRect         = (PFNCLENQUEUEREADBUFFERRECT       )CLEW_DYNLIB_IMPORT(module, "clEnqueueReadBufferRect");
    __clewEnqueueWriteBuffer            = (PFNCLENQUEUEWRITEBUFFER          )CLEW_DYNLIB_IMPORT(module, "clEnqueueWriteBuffer");
    __clewEnqueueWriteBufferRect        = (PFNCLENQUEUEWRITEBUFFERRECT      )CLEW_DYNLIB_IMPORT(module, "clEnqueueWriteBufferRect");
    __clewEnqueueCopyBuffer             = (PFNCLENQUEUECOPYBUFFER           )CLEW_DYNLIB_IMPORT(module, "clEnqueueCopyBuffer");
    __clewEnqueueCopyBufferRect         = (PFNCLENQUEUECOPYBUFFERRECT       )CLEW_DYNLIB_IMPORT(module, "clEnqueueCopyBufferRect");
    __clewEnqueueReadImage              = (PFNCLENQUEUEREADIMAGE            )CLEW_DYNLIB_IMPORT(module, "clEnqueueReadImage");
    __clewEnqueueWriteImage             = (PFNCLENQUEUEWRITEIMAGE           )CLEW_DYNLIB_IMPORT(module, "clEnqueueWriteImage");
    __clewEnqueueCopyImage              = (PFNCLENQUEUECOPYIMAGE            )CLEW_DYNLIB_IMPORT(module, "clEnqueueCopyImage");
    __clewEnqueueCopyImageToBuffer      = (PFNCLENQUEUECOPYIMAGETOBUFFER    )CLEW_DYNLIB_IMPORT(module, "clEnqueueCopyImageToBuffer");
    __clewEnqueueCopyBufferToImage      = (PFNCLENQUEUECOPYBUFFERTOIMAGE    )CLEW_DYNLIB_IMPORT(module, "clEnqueueCopyBufferToImage");
    __clewEnqueueMapBuffer              = (PFNCLENQUEUEMAPBUFFER            )CLEW_DYNLIB_IMPORT(module, "clEnqueueMapBuffer");
    __clewEnqueueMapImage               = (PFNCLENQUEUEMAPIMAGE             )CLEW_DYNLIB_IMPORT(module, "clEnqueueMapImage");
    __clewEnqueueUnmapMemObject         = (PFNCLENQUEUEUNMAPMEMOBJECT       )CLEW_DYNLIB_IMPORT(module, "clEnqueueUnmapMemObject");
    __clewEnqueueNDRangeKernel          = (PFNCLENQUEUENDRANGEKERNEL        )CLEW_DYNLIB_IMPORT(module, "clEnqueueNDRangeKernel");
    __clewEnqueueTask                   = (PFNCLENQUEUETASK                 )CLEW_DYNLIB_IMPORT(module, "clEnqueueTask");
    __clewEnqueueNativeKernel           = (PFNCLENQUEUENATIVEKERNEL         )CLEW_DYNLIB_IMPORT(module, "clEnqueueNativeKernel");
    __clewEnqueueMarker                 = (PFNCLENQUEUEMARKER               )CLEW_DYNLIB_IMPORT(module, "clEnqueueMarker");
    __clewEnqueueWaitForEvents          = (PFNCLENQUEUEWAITFOREVENTS        )CLEW_DYNLIB_IMPORT(module, "clEnqueueWaitForEvents");
    __clewEnqueueBarrier                = (PFNCLENQUEUEBARRIER              )CLEW_DYNLIB_IMPORT(module, "clEnqueueBarrier");
    __clewGetExtensionFunctionAddress   = (PFNCLGETEXTENSIONFUNCTIONADDRESS )CLEW_DYNLIB_IMPORT(module, "clGetExtensionFunctionAddress");

    return CLEW_SUCCESS;
}

const char* clewErrorString(cl_int error)
{
    static const char* strings[] =
    {
        // Error Codes
          "CL_SUCCESS"                                  //   0
        , "CL_DEVICE_NOT_FOUND"                         //  -1
        , "CL_DEVICE_NOT_AVAILABLE"                     //  -2
        , "CL_COMPILER_NOT_AVAILABLE"                   //  -3
        , "CL_MEM_OBJECT_ALLOCATION_FAILURE"            //  -4
        , "CL_OUT_OF_RESOURCES"                         //  -5
        , "CL_OUT_OF_HOST_MEMORY"                       //  -6
        , "CL_PROFILING_INFO_NOT_AVAILABLE"             //  -7
        , "CL_MEM_COPY_OVERLAP"                         //  -8
        , "CL_IMAGE_FORMAT_MISMATCH"                    //  -9
        , "CL_IMAGE_FORMAT_NOT_SUPPORTED"               //  -10
        , "CL_BUILD_PROGRAM_FAILURE"                    //  -11
        , "CL_MAP_FAILURE"                              //  -12

        , ""    //  -13
        , ""    //  -14
        , ""    //  -15
        , ""    //  -16
        , ""    //  -17
        , ""    //  -18
        , ""    //  -19

        , ""    //  -20
        , ""    //  -21
        , ""    //  -22
        , ""    //  -23
        , ""    //  -24
        , ""    //  -25
        , ""    //  -26
        , ""    //  -27
        , ""    //  -28
        , ""    //  -29

        , "CL_INVALID_VALUE"                            //  -30
        , "CL_INVALID_DEVICE_TYPE"                      //  -31
        , "CL_INVALID_PLATFORM"                         //  -32
        , "CL_INVALID_DEVICE"                           //  -33
        , "CL_INVALID_CONTEXT"                          //  -34
        , "CL_INVALID_QUEUE_PROPERTIES"                 //  -35
        , "CL_INVALID_COMMAND_QUEUE"                    //  -36
        , "CL_INVALID_HOST_PTR"                         //  -37
        , "CL_INVALID_MEM_OBJECT"                       //  -38
        , "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"          //  -39
        , "CL_INVALID_IMAGE_SIZE"                       //  -40
        , "CL_INVALID_SAMPLER"                          //  -41
        , "CL_INVALID_BINARY"                           //  -42
        , "CL_INVALID_BUILD_OPTIONS"                    //  -43
        , "CL_INVALID_PROGRAM"                          //  -44
        , "CL_INVALID_PROGRAM_EXECUTABLE"               //  -45
        , "CL_INVALID_KERNEL_NAME"                      //  -46
        , "CL_INVALID_KERNEL_DEFINITION"                //  -47
        , "CL_INVALID_KERNEL"                           //  -48
        , "CL_INVALID_ARG_INDEX"                        //  -49
        , "CL_INVALID_ARG_VALUE"                        //  -50
        , "CL_INVALID_ARG_SIZE"                         //  -51
        , "CL_INVALID_KERNEL_ARGS"                      //  -52
        , "CL_INVALID_WORK_DIMENSION"                   //  -53
        , "CL_INVALID_WORK_GROUP_SIZE"                  //  -54
        , "CL_INVALID_WORK_ITEM_SIZE"                   //  -55
        , "CL_INVALID_GLOBAL_OFFSET"                    //  -56
        , "CL_INVALID_EVENT_WAIT_LIST"                  //  -57
        , "CL_INVALID_EVENT"                            //  -58
        , "CL_INVALID_OPERATION"                        //  -59
        , "CL_INVALID_GL_OBJECT"                        //  -60
        , "CL_INVALID_BUFFER_SIZE"                      //  -61
        , "CL_INVALID_MIP_LEVEL"                        //  -62
        , "CL_INVALID_GLOBAL_WORK_SIZE"                 //  -63
    };

    return strings[-error];
}
