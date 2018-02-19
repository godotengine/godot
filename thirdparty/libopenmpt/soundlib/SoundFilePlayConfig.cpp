/*
 * SoundFilePlayConfig.cpp
 * -----------------------
 * Purpose: Configuration of sound levels, pan laws, etc... for various mix configurations.
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Mixer.h"
#include "SoundFilePlayConfig.h"

OPENMPT_NAMESPACE_BEGIN

CSoundFilePlayConfig::CSoundFilePlayConfig()
{
	setVSTiVolume(1.0f);
}

CSoundFilePlayConfig::~CSoundFilePlayConfig()
{
}

void CSoundFilePlayConfig::SetMixLevels(MixLevels mixLevelType)
{
	switch (mixLevelType)
	{

		// Olivier's version gives us floats in [-0.5; 0.5] and slightly saturates VSTis.
		case mixLevelsOriginal:
			setVSTiAttenuation(NO_ATTENUATION);
			setIntToFloat(1.0f/static_cast<float>(1<<28));
			setFloatToInt(static_cast<float>(1<<28));
			setGlobalVolumeAppliesToMaster(false);
			setUseGlobalPreAmp(true);
			setForcePanningMode(dontForcePanningMode);
			setDisplayDBValues(false);
			setNormalSamplePreAmp(256.0);
			setNormalVSTiVol(100.0);
			setNormalGlobalVol(128.0);
			setExtraSampleAttenuation(MIXING_ATTENUATION);
			break;

		// Ericus' version gives us floats in [-0.06;0.06] and requires attenuation to
		// avoid massive VSTi saturation.
		case mixLevels1_17RC1:
			setVSTiAttenuation(32.0f);
			setIntToFloat(1.0f/static_cast<float>(0x07FFFFFFF));
			setFloatToInt(static_cast<float>(0x07FFFFFFF));
			setGlobalVolumeAppliesToMaster(false);
			setUseGlobalPreAmp(true);
			setForcePanningMode(dontForcePanningMode);
			setDisplayDBValues(false);
			setNormalSamplePreAmp(256.0);
			setNormalVSTiVol(100.0);
			setNormalGlobalVol(128.0);
			setExtraSampleAttenuation(MIXING_ATTENUATION);
			break;

		// 117RC2 gives us floats in [-1.0; 1.0] and hopefully plays VSTis at
		// the right volume... but we attenuate by 2x to approx. match sample volume.
	
		case mixLevels1_17RC2:
			setVSTiAttenuation(2.0f);
			setIntToFloat(1.0f/MIXING_SCALEF);
			setFloatToInt(MIXING_SCALEF);
			setGlobalVolumeAppliesToMaster(true);
			setUseGlobalPreAmp(true);
			setForcePanningMode(dontForcePanningMode);
			setDisplayDBValues(false);
			setNormalSamplePreAmp(256.0);
			setNormalVSTiVol(100.0);
			setNormalGlobalVol(128.0);
			setExtraSampleAttenuation(MIXING_ATTENUATION);
			break;

		// 117RC3 ignores the horrible global, system-specific pre-amp,
		// treats panning as balance to avoid saturation on loud sample (and because I think it's better :),
		// and allows display of attenuation in decibels.
		default:
		case mixLevels1_17RC3:
			setVSTiAttenuation(1.0f);
			setIntToFloat(1.0f/MIXING_SCALEF);
			setFloatToInt(MIXING_SCALEF);
			setGlobalVolumeAppliesToMaster(true);
			setUseGlobalPreAmp(false);
			setForcePanningMode(forceSoftPanning);
			setDisplayDBValues(true);
			setNormalSamplePreAmp(128.0);
			setNormalVSTiVol(128.0);
			setNormalGlobalVol(256.0);
			setExtraSampleAttenuation(0);
			break;

		// A mixmode that is intended to be compatible to legacy trackers (IT/FT2/etc).
		// This is basically derived from mixmode 1.17 RC3, with panning mode and volume levels changed.
		// Sample attenuation is the same as in Schism Tracker (more attenuation than with RC3, thus VSTi attenuation is also higher).
		case mixLevelsCompatible:
		case mixLevelsCompatibleFT2:
			setVSTiAttenuation(0.75f);
			setIntToFloat(1.0f/MIXING_SCALEF);
			setFloatToInt(MIXING_SCALEF);
			setGlobalVolumeAppliesToMaster(true);
			setUseGlobalPreAmp(false);
			setForcePanningMode(mixLevelType == mixLevelsCompatible ? forceNoSoftPanning : forceFT2Panning);
			setDisplayDBValues(true);
			setNormalSamplePreAmp(mixLevelType == mixLevelsCompatible ? 256.0 : 192.0);
			setNormalVSTiVol(mixLevelType == mixLevelsCompatible ? 256.0 : 192.0);
			setNormalGlobalVol(256.0);
			setExtraSampleAttenuation(1);
			break;

	}

	return;
}


OPENMPT_NAMESPACE_END
