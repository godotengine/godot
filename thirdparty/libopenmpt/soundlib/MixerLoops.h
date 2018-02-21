/*
 * MixerLoops.h
 * ------------
 * Purpose: Utility inner loops for mixer-related functionality.
 * Notes  : none.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


#include "Mixer.h"


OPENMPT_NAMESPACE_BEGIN


struct ModChannel;


void StereoMixToFloat(const int32 *pSrc, float *pOut1, float *pOut2, uint32 nCount, const float _i2fc);
void FloatToStereoMix(const float *pIn1, const float *pIn2, int32 *pOut, uint32 uint32, const float _f2ic);
void MonoMixToFloat(const int32 *pSrc, float *pOut, uint32 uint32, const float _i2fc);
void FloatToMonoMix(const float *pIn, int32 *pOut, uint32 uint32, const float _f2ic);

#ifndef MODPLUG_TRACKER
void ApplyGain(int32 *soundBuffer, std::size_t channels, std::size_t countChunk, int32 gainFactor16_16);
void ApplyGain(float *outputBuffer, float * const *outputBuffers, std::size_t offset, std::size_t channels, std::size_t countChunk, float gainFactor);
#endif // !MODPLUG_TRACKER

void InitMixBuffer(mixsample_t *pBuffer, uint32 nSamples);
void InterleaveFrontRear(mixsample_t *pFrontBuf, mixsample_t *pRearBuf, uint32 nFrames);
void MonoFromStereo(mixsample_t *pMixBuf, uint32 nSamples);

void InterleaveStereo(const mixsample_t *inputL, const mixsample_t *inputR, mixsample_t *output, size_t numSamples);
void DeinterleaveStereo(const mixsample_t *input, mixsample_t *outputL, mixsample_t *outputR, size_t numSamples);

void EndChannelOfs(ModChannel &chn, mixsample_t *pBuffer, uint32 nSamples);
void StereoFill(mixsample_t *pBuffer, uint32 nSamples, mixsample_t &rofs, mixsample_t &lofs);


OPENMPT_NAMESPACE_END
