/************************************************************************************
This source file is part of the Theora Video Playback Library
For latest info, see http://libtheoraplayer.googlecode.com
*************************************************************************************
Copyright (c) 2008-2014 Kresimir Spes (kspes@cateia.com)
This program is free software; you can redistribute it and/or modify it under
the terms of the BSD license: http://opensource.org/licenses/BSD-3-Clause
*************************************************************************************/

#ifndef _TheoraVideoClip_h
#define _TheoraVideoClip_h

#include <string>
#include "TheoraExport.h"

// forward class declarations
class TheoraMutex;
class TheoraFrameQueue;
class TheoraTimer;
class TheoraAudioInterface;
class TheoraWorkerThread;
class TheoraDataSource;
class TheoraVideoFrame;

/**
    format of the TheoraVideoFrame pixels. Affects decoding time
 */
enum TheoraOutputMode
{
	// A = full alpha (255), order of letters represents the byte order for a pixel
	// A means the image is treated as if it contains an alpha channel, while X formats
	// just mean that RGB frame is transformed to a 4 byte format
	TH_UNDEFINED = 0,
	TH_RGB    =  1,
	TH_RGBA   =  2,
	TH_RGBX   =  3,
	TH_ARGB   =  4,
	TH_XRGB   =  5,
	TH_BGR    =  6,
	TH_BGRA   =  7,
	TH_BGRX   =  8,
	TH_ABGR   =  9,
	TH_XBGR   = 10,
	TH_GREY   = 11,
	TH_GREY3  = 12,
	TH_GREY3A = 13,
	TH_GREY3X = 14,
	TH_AGREY3 = 15,
	TH_XGREY3 = 16,
	TH_YUV    = 17,
	TH_YUVA   = 18,
	TH_YUVX   = 19,
	TH_AYUV   = 20,
	TH_XYUV   = 21
};

/**
	This object contains all data related to video playback, eg. the open source file,
	the frame queue etc.
*/
class TheoraPlayerExport TheoraVideoClip
{
	friend class TheoraWorkerThread;
	friend class TheoraVideoFrame;
	friend class TheoraVideoManager;
protected:
	TheoraFrameQueue* mFrameQueue;
	TheoraAudioInterface* mAudioInterface;
	TheoraDataSource* mStream;

	TheoraTimer *mTimer, *mDefaultTimer;

	TheoraWorkerThread* mAssignedWorkerThread;
	
	bool mUseAlpha;
	
	bool mWaitingForCache;
	
	// benchmark vars
	int mNumDroppedFrames, mNumDisplayedFrames, mNumPrecachedFrames;

	int mThreadAccessCount; //! counter used by TheoraVideoManager to schedule workload
	
	int mSeekFrame; //! stores desired seek position as a frame number. next worker thread will do the seeking and reset this var to -1
	float mDuration, mFrameDuration, mFPS;
	float mPriority; //! User assigned priority. Default value is 1
    std::string mName;
	int mWidth, mHeight, mStride;
	int mNumFrames;
	int audio_track;

	int mSubFrameWidth, mSubFrameHeight, mSubFrameOffsetX, mSubFrameOffsetY;
	float mAudioGain; //! multiplier for audio samples. between 0 and 1

	TheoraOutputMode mOutputMode, mRequestedOutputMode;
	bool mFirstFrameDisplayed;
	bool mAutoRestart;
	bool mEndOfFile, mRestarted;
	int mIteration, mPlaybackIteration; //! used to ensure smooth playback of looping videos

	TheoraMutex* mAudioMutex; //! syncs audio decoding and extraction
	TheoraMutex* mThreadAccessMutex;
	
	/**
	 * Get the priority of a video clip. based on a forumula that includes user
	 * priority factor, whether the video is paused or not, how many precached
	 * frames it has etc.
	 * This function is used in TheoraVideoManager to efficiently distribute job
	 * assignments among worker threads
	 * @return priority number of this video clip
	 */
	int calculatePriority();
	void readTheoraVorbisHeaders();
	virtual void doSeek() = 0; //! called by WorkerThread to seek to mSeekFrame
	virtual bool _readData() = 0;
	bool isBusy();

	/**
	 * decodes audio from the vorbis stream and stores it in audio packets
	 * This is an internal function of TheoraVideoClip, called regularly if playing an
	 * audio enabled video clip.
	 * @return last decoded timestamp (if found in decoded packet's granule position)
	 */
	virtual float decodeAudio() = 0;
    
    int _getNumReadyFrames();
    void resetFrameQueue();
    int discardOutdatedFrames(float absTime);
    float getAbsPlaybackTime();
	virtual void load(TheoraDataSource* source) = 0;

	virtual void _restart() = 0; // resets the decoder and stream but leaves the frame queue intact
public:
	TheoraVideoClip(TheoraDataSource* data_source,
		            TheoraOutputMode output_mode,
					int nPrecachedFrames,
					bool usePower2Stride);
	virtual ~TheoraVideoClip();

	std::string getName();
	//! Returns the string name of the decoder backend (eg. Theora, AVFoundation)
	virtual std::string getDecoderName() = 0;

	//! benchmark function
	int getNumDisplayedFrames() { return mNumDisplayedFrames; }
	//! benchmark function
	int getNumDroppedFrames() { return mNumDroppedFrames; }

	//! return width in pixels of the video clip
	int getWidth();
	//! return height in pixels of the video clip
	int getHeight();
    
    //! Width of the actual picture inside a video frame (depending on implementation, this may be equal to mWidth or differ within a codec block size (usually 16))
    int getSubFrameWidth();
    //! Height of the actual picture inside a video frame (depending on implementation, this may be equal to mHeight or differ within a codec block size (usually 16))
	int getSubFrameHeight();
    //! X Offset of the actual picture inside a video frame (depending on implementation, this may be 0 or within a codec block size (usually 16))
	int getSubFrameOffsetX();
    //! Y Offset of the actual picture inside a video frame (depending on implementation, this may be 0 or differ within a codec block size (usually 16))
	int getSubFrameOffsetY();
    /**
	    \brief return stride in pixels

		If you've specified usePower2Stride when creating the TheoraVideoClip object
		then this value will be the next power of two size compared to width,
		eg: w=376, stride=512.

		Otherwise, stride will be equal to width
	 */
	int getStride() { return mStride; }

	//! retur the timer objet associated with this object
	TheoraTimer* getTimer();
	//! replace the timer object with a new one
	void setTimer(TheoraTimer* timer);

	//! used by TheoraWorkerThread, do not call directly
	virtual bool decodeNextFrame() = 0;

	//! advance time. TheoraVideoManager calls this
	void update(float timeDelta);
	/**
	    \brief update timer to the display time of the next frame

		useful if you want to grab frames instead of regular display
		\return time advanced. 0 if no frames are ready
	*/
	float updateToNextFrame();

	
	TheoraFrameQueue* getFrameQueue();
	
	/**
	    \brief pop the frame from the front of the FrameQueue

		see TheoraFrameQueue::pop() for more details
	 */
	void popFrame();

	/**
	    \brief Returns the first available frame in the queue or NULL if no frames are available.

		see TheoraFrameQueue::getFirstAvailableFrame() for more details
	*/
	TheoraVideoFrame* getNextFrame();
	/**
	    check if there is enough audio data decoded to submit to the audio interface

		TheoraWorkerThread calls this
	 */
	virtual void decodedAudioCheck() = 0;

	void setAudioInterface(TheoraAudioInterface* iface);
	TheoraAudioInterface* getAudioInterface();

	/**
	    \brief resize the frame queues

		Warning: this call discards ready frames in the frame queue
	 */
	void setNumPrecachedFrames(int n);
	//! returns the size of the frame queue
	int getNumPrecachedFrames();
	//! returns the number of ready frames in the frame queue
	int getNumReadyFrames();

	//! if you want to adjust the audio gain. range [0,1]
	void setAudioGain(float gain);
	float getAudioGain();

	//! if you want the video to automatically and smoothly restart when the last frame is reached
	void setAutoRestart(bool value);
	bool getAutoRestart() { return mAutoRestart; }


	void set_audio_track(int p_track) { audio_track=p_track; }

	/**
	    TODO: user priority. Useful only when more than one video is being decoded
	 */
	void setPriority(float priority);
	float getPriority();

	//! Used by TheoraVideoManager to schedule work
	float getPriorityIndex();

	//! get the current time index from the timer object
	float getTimePosition();
	//! get the duration of the movie in seconds
	float getDuration();
	//! return the clips' frame rate, warning, fps can be a non integer number!
	float getFPS();
	//! get the number of frames in this movie
	int getNumFrames() { return mNumFrames; }

	//! return the current output mode for this video object
	TheoraOutputMode getOutputMode();
	/**
	    set a new output mode

		Warning: this discards the frame queue. ready frames will be lost.
	 */
	void setOutputMode(TheoraOutputMode mode);

    bool isDone();
	void play();
	void pause();
	void restart();
	bool isPaused();
	void stop();
    void setPlaybackSpeed(float speed);
    float getPlaybackSpeed();
	//! seek to a given time position
	void seek(float time);
	//! seek to a given frame number
	void seekToFrame(int frame);
	//! wait max_time for the clip to cache a given percentage of frames, factor in range [0,1]
	void waitForCache(float desired_cache_factor = 0.5f, float max_wait_time = 1.0f);
};

#endif
