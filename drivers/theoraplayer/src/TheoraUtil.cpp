/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <map>
#ifndef _WIN32
#include <unistd.h>
#include <pthread.h>
#endif

#include "TheoraUtil.h"
#include "TheoraException.h"

#ifdef _WIN32
#include <windows.h>
#pragma warning( disable: 4996 ) // MSVC++
#endif

std::string str(int i)
{
    char s[32];
    sprintf(s, "%d", i);
    return std::string(s);
}

std::string strf(float i)
{
    char s[32];
    sprintf(s, "%.3f", i);
    return std::string(s);
}

void _psleep(int miliseconds)
{
#ifdef _WIN32
#ifndef _WINRT
	Sleep(miliseconds);
#else
	WaitForSingleObjectEx(GetCurrentThread(), miliseconds, 0);
#endif
#else
	usleep(miliseconds * 1000);
#endif
}


int _nextPow2(int x)
{
	int y;
	for (y = 1; y < x; y *= 2);
	return y;
}
