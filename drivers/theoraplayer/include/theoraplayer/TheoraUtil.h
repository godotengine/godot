/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#ifndef _TheoraUtil_h
#define _TheoraUtil_h

#include <string>
#include <vector>

#ifndef THEORAUTIL_NOMACROS

#define foreach(type,lst) for (std::vector<type>::iterator it=lst.begin();it != lst.end(); ++it)
#define foreach_l(type,lst) for (std::list<type>::iterator it=lst.begin();it != lst.end(); ++it)
#define foreach_r(type,lst) for (std::vector<type>::reverse_iterator it=lst.rbegin();it != lst.rend(); ++it)
#define foreach_in_map(type,lst) for (std::map<std::string,type>::iterator it=lst.begin();it != lst.end(); ++it)

#endif

#define th_writelog(x) TheoraVideoManager::getSingleton().logMessage(x)


std::string str(int i);
std::string strf(float i);
void _psleep(int milliseconds);
int _nextPow2(int x);

#endif