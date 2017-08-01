/*
Copyright (c) 2013 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans

#include "b3Logging.h"

#include <stdio.h>
#include <stdarg.h>

#ifdef _WIN32
#include <windows.h>
#endif //_WIN32


void b3PrintfFuncDefault(const char* msg)
{
#ifdef _WIN32
	OutputDebugStringA(msg);
#endif
	printf("%s",msg);
    //is this portable?
    fflush(stdout);
}

void b3WarningMessageFuncDefault(const char* msg)
{
#ifdef _WIN32
	OutputDebugStringA(msg);
#endif
	printf("%s",msg);
    //is this portable?
    fflush(stdout);

}


void b3ErrorMessageFuncDefault(const char* msg)
{
#ifdef _WIN32
	OutputDebugStringA(msg);
#endif
	printf("%s",msg);

    //is this portable?
    fflush(stdout);
    
}



static b3PrintfFunc* b3s_printfFunc = b3PrintfFuncDefault;
static b3WarningMessageFunc* b3s_warningMessageFunc = b3WarningMessageFuncDefault;
static b3ErrorMessageFunc* b3s_errorMessageFunc = b3ErrorMessageFuncDefault;


///The developer can route b3Printf output using their own implementation
void b3SetCustomPrintfFunc(b3PrintfFunc* printfFunc)
{
	b3s_printfFunc = printfFunc;
}
void b3SetCustomWarningMessageFunc(b3PrintfFunc* warningMessageFunc)
{
	b3s_warningMessageFunc = warningMessageFunc;
}
void b3SetCustomErrorMessageFunc(b3PrintfFunc* errorMessageFunc)
{
	b3s_errorMessageFunc = errorMessageFunc;
}

//#define B3_MAX_DEBUG_STRING_LENGTH 2048
#define B3_MAX_DEBUG_STRING_LENGTH 32768


void b3OutputPrintfVarArgsInternal(const char *str, ...)
{
    char strDebug[B3_MAX_DEBUG_STRING_LENGTH]={0};
    va_list argList;
    va_start(argList, str);
#ifdef _MSC_VER
    vsprintf_s(strDebug,B3_MAX_DEBUG_STRING_LENGTH,str,argList);
#else
    vsnprintf(strDebug,B3_MAX_DEBUG_STRING_LENGTH,str,argList);
#endif
        (b3s_printfFunc)(strDebug);
    va_end(argList);    
}
void b3OutputWarningMessageVarArgsInternal(const char *str, ...)
{
    char strDebug[B3_MAX_DEBUG_STRING_LENGTH]={0};
    va_list argList;
    va_start(argList, str);
#ifdef _MSC_VER
    vsprintf_s(strDebug,B3_MAX_DEBUG_STRING_LENGTH,str,argList);
#else
    vsnprintf(strDebug,B3_MAX_DEBUG_STRING_LENGTH,str,argList);
#endif
        (b3s_warningMessageFunc)(strDebug);
    va_end(argList);    
}
void b3OutputErrorMessageVarArgsInternal(const char *str, ...)
{
	
    char strDebug[B3_MAX_DEBUG_STRING_LENGTH]={0};
    va_list argList;
    va_start(argList, str);
#ifdef _MSC_VER
    vsprintf_s(strDebug,B3_MAX_DEBUG_STRING_LENGTH,str,argList);
#else
    vsnprintf(strDebug,B3_MAX_DEBUG_STRING_LENGTH,str,argList);
#endif
        (b3s_errorMessageFunc)(strDebug);
    va_end(argList);    

}


void	b3EnterProfileZoneDefault(const char* name)
{
}
void	b3LeaveProfileZoneDefault()
{
}
static b3EnterProfileZoneFunc* b3s_enterFunc = b3EnterProfileZoneDefault;
static b3LeaveProfileZoneFunc* b3s_leaveFunc = b3LeaveProfileZoneDefault;
void b3EnterProfileZone(const char* name)
{
	(b3s_enterFunc)(name);
}
void b3LeaveProfileZone()
{
	(b3s_leaveFunc)();
}

void b3SetCustomEnterProfileZoneFunc(b3EnterProfileZoneFunc* enterFunc)
{
	b3s_enterFunc = enterFunc;
}
void b3SetCustomLeaveProfileZoneFunc(b3LeaveProfileZoneFunc* leaveFunc)
{
	b3s_leaveFunc = leaveFunc;
}




#ifndef _MSC_VER
#undef vsprintf_s
#endif

