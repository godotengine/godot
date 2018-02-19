/*
 * MixerSettings.h
 * ---------------
 * Purpose: A struct containing settings for the mixer of soundlib.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once


OPENMPT_NAMESPACE_BEGIN


struct MixerSettings
{

	int32 m_nStereoSeparation;
	static const int32 StereoSeparationScale = 128;
	
	uint32 m_nMaxMixChannels;
	uint32 DSPMask;
	uint32 MixerFlags;
	uint32 gdwMixingFreq;
	uint32 gnChannels;
	uint32 m_nPreAmp;

	int32 VolumeRampUpMicroseconds;
	int32 VolumeRampDownMicroseconds;
	int32 GetVolumeRampUpMicroseconds() const { return VolumeRampUpMicroseconds; }
	int32 GetVolumeRampDownMicroseconds() const { return VolumeRampDownMicroseconds; }
	void SetVolumeRampUpMicroseconds(int32 rampUpMicroseconds) { VolumeRampUpMicroseconds = rampUpMicroseconds; }
	void SetVolumeRampDownMicroseconds(int32 rampDownMicroseconds) { VolumeRampDownMicroseconds = rampDownMicroseconds; }
	
	int32 GetVolumeRampUpSamples() const;
	int32 GetVolumeRampDownSamples() const;

	void SetVolumeRampUpSamples(int32 rampUpSamples);
	void SetVolumeRampDownSamples(int32 rampDownSamples);
	
	bool IsValid() const
	{
		return (gdwMixingFreq > 0) && (gnChannels == 1 || gnChannels == 2 || gnChannels == 4);
	}
	
	MixerSettings();

};


OPENMPT_NAMESPACE_END
