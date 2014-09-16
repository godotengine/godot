/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#include <stdio.h>
#include <memory.h>
#include "TheoraDataSource.h"
#include "TheoraException.h"
#include "TheoraVideoManager.h"
#include "TheoraUtil.h"

TheoraDataSource::~TheoraDataSource()
{

}

TheoraFileDataSource::TheoraFileDataSource(std::string filename)
{
	mFilename = filename;
	mFilePtr = NULL;
}

TheoraFileDataSource::~TheoraFileDataSource()
{
	if (mFilePtr)
	{
		fclose(mFilePtr);
		mFilePtr = NULL;
	}
}

void TheoraFileDataSource::openFile()
{
	if (mFilePtr == NULL)
	{
		mFilePtr=fopen(mFilename.c_str(), "rb");
		if (!mFilePtr)
        {
            std::string msg = "Can't open video file: " + mFilename;
            th_writelog(msg);
            throw TheoraGenericException(msg);
        }
		fseek(mFilePtr, 0, SEEK_END);
		mSize = ftell(mFilePtr);
		fseek(mFilePtr, 0, SEEK_SET);
	}
}

int TheoraFileDataSource::read(void* output, int nBytes)
{
	if (mFilePtr == NULL) openFile();
	size_t n = fread(output, 1, nBytes, mFilePtr);
	return (int) n;
}

void TheoraFileDataSource::seek(unsigned long byte_index)
{
	if (mFilePtr == NULL) openFile();
	fseek(mFilePtr, byte_index, SEEK_SET);
}

unsigned long TheoraFileDataSource::size()
{
	if (mFilePtr == NULL) openFile();
	return mSize;
}

unsigned long TheoraFileDataSource::tell()
{
	if (mFilePtr == NULL) return 0;
	return ftell(mFilePtr);
}

TheoraMemoryFileDataSource::TheoraMemoryFileDataSource(std::string filename) :
	mReadPointer(0),
	mData(0)
{
	mFilename=filename;
	FILE* f=fopen(filename.c_str(),"rb");
	if (!f) throw TheoraGenericException("Can't open video file: "+filename);
	fseek(f,0,SEEK_END);
	mSize=ftell(f);
	fseek(f,0,SEEK_SET);
	mData=new unsigned char[mSize];
	fread(mData,1,mSize,f);
	fclose(f);
}

TheoraMemoryFileDataSource::TheoraMemoryFileDataSource(unsigned char* data, long size, const std::string& filename)
{
	mFilename = filename;
	mData = data;
	mSize = size;
	mReadPointer = 0;
}

TheoraMemoryFileDataSource::~TheoraMemoryFileDataSource()
{
	if (mData) delete [] mData;
}

int TheoraMemoryFileDataSource::read(void* output, int nBytes)
{
	int n = (int) ((mReadPointer+nBytes <= mSize) ? nBytes : mSize - mReadPointer);
	if (!n) return 0;
	memcpy(output, mData + mReadPointer, n);
	mReadPointer += n;
	return n;
}

void TheoraMemoryFileDataSource::seek(unsigned long byte_index)
{
	mReadPointer=byte_index;
}

unsigned long TheoraMemoryFileDataSource::size()
{
	return mSize;
}

unsigned long TheoraMemoryFileDataSource::tell()
{
	return mReadPointer;
}
