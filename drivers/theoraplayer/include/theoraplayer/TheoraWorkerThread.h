/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#ifndef _TheoraWorkerThread_h
#define _TheoraWorkerThread_h

#include "TheoraAsync.h"

class TheoraVideoClip;

/**
	This is the worker thread, requests work from TheoraVideoManager
	and decodes assigned TheoraVideoClip objects
*/
class TheoraWorkerThread : public TheoraThread
{
	TheoraVideoClip* mClip;
public:
	TheoraWorkerThread();
	~TheoraWorkerThread();

	TheoraVideoClip* getAssignedClip() { return mClip; }

    //! Main Thread Body - do not call directly!
	void execute();
};
#endif
