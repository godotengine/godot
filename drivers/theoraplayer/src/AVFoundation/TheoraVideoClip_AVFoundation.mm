/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/
#ifdef __AVFOUNDATION
#define AVFOUNDATION_CLASSES_DEFINED
#import <AVFoundation/AVFoundation.h>
#include "TheoraAudioInterface.h"
#include "TheoraDataSource.h"
#include "TheoraException.h"
#include "TheoraTimer.h"
#include "TheoraUtil.h"
#include "TheoraFrameQueue.h"
#include "TheoraVideoFrame.h"
#include "TheoraVideoManager.h"
#include "TheoraVideoClip_AVFoundation.h"
#include "TheoraPixelTransform.h"

#ifdef _AVFOUNDATION_BGRX
// a fast function developed to use kernel byte swapping calls to optimize alpha decoding.
// In AVFoundation, BGRX mode conversion is prefered to YUV conversion because apple's YUV
// conversion on iOS seems to run faster than libtheoraplayer's implementation
// This may change in the future with more optimizations to libtheoraplayers's YUV conversion
// code, making this function obsolete.
static void bgrx2rgba(unsigned char* dest, int w, int h, struct TheoraPixelTransform* t)
{
	unsigned register int a;
	unsigned int *dst = (unsigned int*) dest, *dstEnd;
	unsigned char* src = t->raw;
	int y, x, ax;
	
	for (y = 0; y < h; ++y, src += t->rawStride)
	{
		for (x = 0, ax = w * 4, dstEnd = dst + w; dst != dstEnd; x += 4, ax += 4, ++dst)
		{
            // use the full alpha range here because the Y channel has already been converted
            // to RGB and that's in [0, 255] range.
			a = src[ax];
            *dst = (OSReadSwapInt32(src, x) >> 8) | (a << 24);
		}
	}
}
#endif

static CVPlanarPixelBufferInfo_YCbCrPlanar getYUVStruct(void* src)
{
	CVPlanarPixelBufferInfo_YCbCrPlanar* bigEndianYuv = (CVPlanarPixelBufferInfo_YCbCrPlanar*) src;
	CVPlanarPixelBufferInfo_YCbCrPlanar yuv;
	yuv.componentInfoY.offset = OSSwapInt32(bigEndianYuv->componentInfoY.offset);
	yuv.componentInfoY.rowBytes = OSSwapInt32(bigEndianYuv->componentInfoY.rowBytes);
	yuv.componentInfoCb.offset = OSSwapInt32(bigEndianYuv->componentInfoCb.offset);
	yuv.componentInfoCb.rowBytes = OSSwapInt32(bigEndianYuv->componentInfoCb.rowBytes);
	yuv.componentInfoCr.offset = OSSwapInt32(bigEndianYuv->componentInfoCr.offset);
	yuv.componentInfoCr.rowBytes = OSSwapInt32(bigEndianYuv->componentInfoCr.rowBytes);
	return yuv;
}

TheoraVideoClip_AVFoundation::TheoraVideoClip_AVFoundation(TheoraDataSource* data_source,
											   TheoraOutputMode output_mode,
											   int nPrecachedFrames,
											   bool usePower2Stride):
	TheoraVideoClip(data_source, output_mode, nPrecachedFrames, usePower2Stride),
	TheoraAudioPacketQueue()
{
	mLoaded = 0;
	mReader = NULL;
	mOutput = mAudioOutput = NULL;
	mReadAudioSamples = mAudioFrequency = mNumAudioChannels = 0;
}

TheoraVideoClip_AVFoundation::~TheoraVideoClip_AVFoundation()
{
	unload();
}

void TheoraVideoClip_AVFoundation::unload()
{
	if (mOutput != NULL || mAudioOutput != NULL || mReader != NULL)
	{
		NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];

		if (mOutput != NULL)
		{
			[mOutput release];
			mOutput = NULL;
		}
		
		if (mAudioOutput)
		{
			[mAudioOutput release];
			mAudioOutput = NULL;
		}
		
		if (mReader != NULL)
		{
			[mReader release];
			mReader = NULL;
		}
		
		[pool release];
	}
}

bool TheoraVideoClip_AVFoundation::_readData()
{
	return 1;
}

bool TheoraVideoClip_AVFoundation::decodeNextFrame()
{
	if (mReader == NULL || mEndOfFile) return 0;
	AVAssetReaderStatus status = [mReader status];
	if (status == AVAssetReaderStatusFailed)
	{
		// This can happen on iOS when you suspend the app... Only happens on the device, iOS simulator seems to work fine.
		th_writelog("AVAssetReader reading failed, restarting...");

		mSeekFrame = mTimer->getTime() * mFPS;
		// just in case
		if (mSeekFrame < 0) mSeekFrame = 0;
		if (mSeekFrame > mDuration * mFPS - 1) mSeekFrame = mDuration * mFPS - 1;
		_restart();
		status = [mReader status];
		if (status == AVAssetReaderStatusFailed)
		{
			th_writelog("AVAssetReader restart failed!");
			return 0;
		}
		th_writelog("AVAssetReader restart succeeded!");
	}

	TheoraVideoFrame* frame = mFrameQueue->requestEmptyFrame();
	if (!frame) return 0;

	CMSampleBufferRef sampleBuffer = NULL;
	NSAutoreleasePool* pool = NULL;
	CMTime presentationTime;
	
	if (mAudioInterface) decodeAudio();
	
	if (status == AVAssetReaderStatusReading)
	{
		pool = [[NSAutoreleasePool alloc] init];
		
		while ((sampleBuffer = [mOutput copyNextSampleBuffer]))
		{
			presentationTime = CMSampleBufferGetOutputPresentationTimeStamp(sampleBuffer);
			frame->mTimeToDisplay = (float) CMTimeGetSeconds(presentationTime);
			frame->mIteration = mIteration;
			frame->_setFrameNumber(mFrameNumber);
			++mFrameNumber;
			if (frame->mTimeToDisplay < mTimer->getTime() && !mRestarted && mFrameNumber % 16 != 0)
			{
				// %16 operation is here to prevent a playback halt during video playback if the decoder can't keep up with demand.
#ifdef _DEBUG
				th_writelog(mName + ": pre-dropped frame " + str(mFrameNumber - 1));
#endif
				++mNumDisplayedFrames;
				++mNumDroppedFrames;
				CMSampleBufferInvalidate(sampleBuffer);
				CFRelease(sampleBuffer);
				sampleBuffer = NULL;
				continue; // drop frame
			}

			CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
			CVPixelBufferLockBaseAddress(imageBuffer, 0);
			void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
			
			mStride = CVPixelBufferGetBytesPerRow(imageBuffer);
			size_t width = CVPixelBufferGetWidth(imageBuffer);
			size_t height = CVPixelBufferGetHeight(imageBuffer);

			TheoraPixelTransform t;
			memset(&t, 0, sizeof(TheoraPixelTransform));
#ifdef _AVFOUNDATION_BGRX
			if (mOutputMode == TH_BGRX || mOutputMode == TH_RGBA)
			{
				t.raw = (unsigned char*) baseAddress;
				t.rawStride = mStride;
			}
			else
#endif
			{
				CVPlanarPixelBufferInfo_YCbCrPlanar yuv = getYUVStruct(baseAddress);
				
				t.y = (unsigned char*) baseAddress + yuv.componentInfoY.offset;  t.yStride = yuv.componentInfoY.rowBytes;
				t.u = (unsigned char*) baseAddress + yuv.componentInfoCb.offset; t.uStride = yuv.componentInfoCb.rowBytes;
				t.v = (unsigned char*) baseAddress + yuv.componentInfoCr.offset; t.vStride = yuv.componentInfoCr.rowBytes;
			}
#ifdef _AVFOUNDATION_BGRX
			if (mOutputMode == TH_RGBA)
			{
				for (int i = 0; i < 1000; ++i)
					bgrx2rgba(frame->getBuffer(), mWidth / 2, mHeight, &t);
				frame->mReady = true;
			}
			else
#endif
			frame->decode(&t);

			CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
			CMSampleBufferInvalidate(sampleBuffer);
			CFRelease(sampleBuffer);

			break; // TODO - should this really be a while loop instead of an if block?
		}
	}
	if (pool) [pool release];

	if (!frame->mReady) // in case the frame wasn't used
	{
		frame->mInUse = 0;
	}

	if (sampleBuffer == NULL && mReader.status == AVAssetReaderStatusCompleted) // other cases could be app suspended
	{
		if (mAutoRestart)
        {
            ++mIteration;
			_restart();
        }
		else
		{
			unload();
			mEndOfFile = true;
		}
		return 0;
	}
	
	
	return 1;
}

void TheoraVideoClip_AVFoundation::_restart()
{
	mEndOfFile = false;
	unload();
	load(mStream);
	mRestarted = true;
}

void TheoraVideoClip_AVFoundation::load(TheoraDataSource* source)
{
	mStream = source;
	mFrameNumber = 0;
	mEndOfFile = false;
	TheoraFileDataSource* fileDataSource = dynamic_cast<TheoraFileDataSource*>(source);
	std::string filename;
	if (fileDataSource != NULL) filename = fileDataSource->getFilename();
	else
	{
		TheoraMemoryFileDataSource* memoryDataSource = dynamic_cast<TheoraMemoryFileDataSource*>(source);
		if (memoryDataSource != NULL) filename = memoryDataSource->getFilename();
		else throw TheoraGenericException("Unable to load MP4 file");
	}
	
	NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
	NSString* path = [NSString stringWithUTF8String:filename.c_str()];
	NSError* err;
	NSURL *url = [NSURL fileURLWithPath:path];
	AVAsset* asset = [[AVURLAsset alloc] initWithURL:url options:nil];
	mReader = [[AVAssetReader alloc] initWithAsset:asset error:&err];
	NSArray* tracks = [asset tracksWithMediaType:AVMediaTypeVideo];
	if ([tracks count] == 0)
		throw TheoraGenericException("Unable to open video file: " + filename);
	AVAssetTrack *videoTrack = [tracks objectAtIndex:0];

	NSArray* audioTracks = [asset tracksWithMediaType:AVMediaTypeAudio];
	if (audio_track >= audioTracks.count)
		audio_track = 0;
	AVAssetTrack *audioTrack = audioTracks.count > 0 ? [audioTracks objectAtIndex:audio_track] : NULL;
	printf("*********** using audio track %i\n", audio_track);
	
#ifdef _AVFOUNDATION_BGRX
	bool yuv_output = (mOutputMode != TH_BGRX && mOutputMode != TH_RGBA);
#else
	bool yuv_output = true;
#endif
	
	NSDictionary *videoOptions = [NSDictionary dictionaryWithObjectsAndKeys:[NSNumber numberWithInt:(yuv_output) ? kCVPixelFormatType_420YpCbCr8Planar : kCVPixelFormatType_32BGRA], kCVPixelBufferPixelFormatTypeKey, nil];

	mOutput = [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack outputSettings:videoOptions];
	[mReader addOutput:mOutput];
	if ([mOutput respondsToSelector:@selector(setAlwaysCopiesSampleData:)]) // Not supported on iOS versions older than 5.0
		mOutput.alwaysCopiesSampleData = NO;

	mFPS = videoTrack.nominalFrameRate;
	mWidth = mSubFrameWidth = mStride = videoTrack.naturalSize.width;
	mHeight = mSubFrameHeight = videoTrack.naturalSize.height;
	mFrameDuration = 1.0f / mFPS;
	mDuration = (float) CMTimeGetSeconds(asset.duration);
	if (mFrameQueue == NULL)
	{
		mFrameQueue = new TheoraFrameQueue(this);
		mFrameQueue->setSize(mNumPrecachedFrames);
	}

	if (mSeekFrame != -1)
	{
		mFrameNumber = mSeekFrame;
		[mReader setTimeRange: CMTimeRangeMake(CMTimeMakeWithSeconds(mSeekFrame / mFPS, 1), kCMTimePositiveInfinity)];
	}
	if (audioTrack)
	{
		TheoraAudioInterfaceFactory* audio_factory = TheoraVideoManager::getSingleton().getAudioInterfaceFactory();
		if (audio_factory)
		{
			NSDictionary *audioOptions = [NSDictionary dictionaryWithObjectsAndKeys:
										  [NSNumber numberWithInt:kAudioFormatLinearPCM], AVFormatIDKey,
										  [NSNumber numberWithBool:NO], AVLinearPCMIsNonInterleaved,
										  [NSNumber numberWithBool:NO], AVLinearPCMIsBigEndianKey,
										  [NSNumber numberWithBool:YES], AVLinearPCMIsFloatKey,
										  [NSNumber numberWithInt:32], AVLinearPCMBitDepthKey,
										  nil];

			mAudioOutput = [[AVAssetReaderTrackOutput alloc] initWithTrack:audioTrack outputSettings:audioOptions];
			[mReader addOutput:mAudioOutput];
			if ([mAudioOutput respondsToSelector:@selector(setAlwaysCopiesSampleData:)]) // Not supported on iOS versions older than 5.0
				mAudioOutput.alwaysCopiesSampleData = NO;
			
			NSArray* desclst = audioTrack.formatDescriptions;
			CMAudioFormatDescriptionRef desc = (CMAudioFormatDescriptionRef) [desclst objectAtIndex:0];
			const AudioStreamBasicDescription* audioDesc = CMAudioFormatDescriptionGetStreamBasicDescription(desc);
			mAudioFrequency = (unsigned int) audioDesc->mSampleRate;
			mNumAudioChannels = audioDesc->mChannelsPerFrame;
			
			if (mSeekFrame != -1)
			{
				mReadAudioSamples = mFrameNumber * (mAudioFrequency * mNumAudioChannels) / mFPS;
			}
			else mReadAudioSamples = 0;

			if (mAudioInterface == NULL)
				setAudioInterface(audio_factory->createInstance(this, mNumAudioChannels, mAudioFrequency));
		}
	}
	
#ifdef _DEBUG
	else if (!mLoaded)
	{
		th_writelog("-----\nwidth: " + str(mWidth) + ", height: " + str(mHeight) + ", fps: " + str((int) getFPS()));
		th_writelog("duration: " + strf(mDuration) + " seconds\n-----");
	}
#endif
	[mReader startReading];
	[pool release];
	mLoaded = true;
}
 
void TheoraVideoClip_AVFoundation::decodedAudioCheck()
{
	if (!mAudioInterface || mTimer->isPaused()) return;
	
	mAudioMutex->lock();
	flushAudioPackets(mAudioInterface);
	mAudioMutex->unlock();
}

float TheoraVideoClip_AVFoundation::decodeAudio()
{
	if (mRestarted) return -1;

	if (mReader == NULL || mEndOfFile) return 0;
	AVAssetReaderStatus status = [mReader status];

	if (mAudioOutput)
	{
		CMSampleBufferRef sampleBuffer = NULL;
		NSAutoreleasePool* pool = NULL;
		bool mutexLocked = 0;

		float factor = 1.0f / (mAudioFrequency * mNumAudioChannels);
		float videoTime = (float) mFrameNumber / mFPS;
		float min = mFrameQueue->getSize() / mFPS + 1.0f;
		
		if (status == AVAssetReaderStatusReading)
		{
			pool = [[NSAutoreleasePool alloc] init];

			// always buffer up of audio ahead of the frames
			while (mReadAudioSamples * factor - videoTime < min)
			{
				if ((sampleBuffer = [mAudioOutput copyNextSampleBuffer]))
				{
					AudioBufferList audioBufferList;
					
					CMBlockBufferRef blockBuffer = NULL;
					CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(sampleBuffer, NULL, &audioBufferList, sizeof(audioBufferList), NULL, NULL, 0, &blockBuffer);
					
					for (int y = 0; y < audioBufferList.mNumberBuffers; ++y)
					{
						AudioBuffer audioBuffer = audioBufferList.mBuffers[y];
						float *frame = (float*) audioBuffer.mData;

						if (!mutexLocked)
						{
							mAudioMutex->lock();
							mutexLocked = 1;
						}
						addAudioPacket(frame, audioBuffer.mDataByteSize / (mNumAudioChannels * sizeof(float)), mAudioGain);
						
						mReadAudioSamples += audioBuffer.mDataByteSize / (sizeof(float));
					}

					CFRelease(blockBuffer);
					CMSampleBufferInvalidate(sampleBuffer);
					CFRelease(sampleBuffer);
				}
				else
				{
					[mAudioOutput release];
					mAudioOutput = nil;
					break;
				}
			}
			[pool release];
		}
		if (mutexLocked) mAudioMutex->unlock();
	}
	
	return -1;
}

void TheoraVideoClip_AVFoundation::doSeek()
{
#if _DEBUG
	th_writelog(mName + " [seek]: seeking to frame " + str(mSeekFrame));
#endif
	int frame;
	float time = mSeekFrame / getFPS();
	mTimer->seek(time);
	bool paused = mTimer->isPaused();
	if (!paused) mTimer->pause(); // pause until seeking is done
	
	mEndOfFile = false;
	mRestarted = false;
	
    resetFrameQueue();
	unload();
	load(mStream);

	if (mAudioInterface)
	{
		mAudioMutex->lock();
		destroyAllAudioPackets();
		mAudioMutex->unlock();
	}

	if (!paused) mTimer->play();
	mSeekFrame = -1;
}
#endif
