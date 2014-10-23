/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#ifndef EXCEPTION_H
#define EXCEPTION_H

#include <string>
#include "TheoraExport.h"

class TheoraPlayerExport _TheoraGenericException
{
public:
    std::string mErrText,mFile,mType;
	int mLineNumber;

	_TheoraGenericException(const std::string& errorText, std::string type = "",std::string file = "", int line = 0);
    virtual ~_TheoraGenericException() {}
    
	virtual std::string repr();
    
	void writeOutput();
	
	virtual const std::string& getErrorText() { return mErrText; }
    
	const std::string getType(){ return mType; }
};

#define TheoraGenericException(msg) _TheoraGenericException(msg, "TheoraGenericException", __FILE__, __LINE__)


#define exception_cls(name) class name : public _TheoraGenericException \
{ \
public: \
	name(const std::string& errorText,std::string type = "",std::string file = "",int line = 0) : \
	  _TheoraGenericException(errorText, type, file, line){} \
}

exception_cls(_KeyException);


#endif
