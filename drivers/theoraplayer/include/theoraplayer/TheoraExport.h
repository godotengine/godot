/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#ifndef _theoraVideoExport_h
#define _theoraVideoExport_h

	#ifdef _LIB
		#define TheoraPlayerExport
		#define TheoraPlayerFnExport
	#else
		#ifdef _WIN32
			#ifdef THEORAVIDEO_EXPORTS
				#define TheoraPlayerExport __declspec(dllexport)
				#define TheoraPlayerFnExport __declspec(dllexport)
			#else
				#define TheoraPlayerExport __declspec(dllimport)
				#define TheoraPlayerFnExport __declspec(dllimport)
			#endif
		#else
			#define TheoraPlayerExport __attribute__ ((visibility("default")))
			#define TheoraPlayerFnExport __attribute__ ((visibility("default")))
		#endif
	#endif
	#ifndef DEPRECATED_ATTRIBUTE
		#ifdef _MSC_VER
			#define DEPRECATED_ATTRIBUTE __declspec(deprecated("function is deprecated"))
		#else
			#define DEPRECATED_ATTRIBUTE __attribute__((deprecated))
		#endif
	#endif

#endif

