/*
 * Fastmix.cpp
 * -----------
 * Purpose: Mixer core for rendering samples, mixing plugins, etc...
 * Notes  : If this is Fastmix.cpp, where is Slowmix.cpp? :)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


// FIXME:
// - Playing samples backwards should reverse interpolation LUTs for interpolation modes
//   with more than two taps since they're not symmetric. We might need separate LUTs
//   because otherwise we will add tons of branches.
// - Loop wraparound works pretty well in general, but not at the start of bidi samples.
// - The loop lookahead stuff might still fail for samples with backward loops.

#include "stdafx.h"
#include "Sndfile.h"
#include "MixerLoops.h"
#include "MixFuncTable.h"
#include <cfloat>	// For FLT_EPSILON
#include "plugins/PlugInterface.h"
#include <algorithm>


OPENMPT_NAMESPACE_BEGIN


/////////////////////////////////////////////////////////////////////////

struct MixLoopState
{
	const int8 * samplePointer;
	const int8 * lookaheadPointer;
	SmpLength lookaheadStart;
	uint32 maxSamples;

	MixLoopState(const ModChannel &chn)
	{
		UpdateLookaheadPointers(chn);

		// For platforms that have no fast 64-bit division, precompute this constant
		// as it won't change during the invocation of CreateStereoMix.
		SamplePosition increment = chn.increment;
		if(increment.IsNegative())
			increment.Negate();
		maxSamples = 16384u / (increment.GetUInt() + 1u);
		if(maxSamples < 2) maxSamples = 2;
	}

	// Calculate offset of loop wrap-around buffer for this sample.
	void UpdateLookaheadPointers(const ModChannel &chn)
	{
		samplePointer = static_cast<const int8 *>(chn.pCurrentSample);
		lookaheadPointer = nullptr;
		if(chn.nLoopEnd < InterpolationMaxLookahead)
			lookaheadStart = chn.nLoopStart;
		else
			lookaheadStart = std::max(chn.nLoopStart, chn.nLoopEnd - InterpolationMaxLookahead);
		// We only need to apply the loop wrap-around logic if the sample is actually looping and if interpolation is applied.
		// If there is no interpolation happening, there is no lookahead happening the sample read-out is exact.
		if(chn.dwFlags[CHN_LOOP] && chn.resamplingMode != SRCMODE_NEAREST)
		{
			const bool inSustainLoop = chn.InSustainLoop();

			// Do not enable wraparound magic if we're previewing a custom loop!
			if(inSustainLoop || chn.nLoopEnd == chn.pModSample->nLoopEnd)
			{
				SmpLength lookaheadOffset = 3 * InterpolationMaxLookahead + chn.pModSample->nLength - chn.nLoopEnd;
				if(inSustainLoop)
				{
					lookaheadOffset += 4 * InterpolationMaxLookahead;
				}
				lookaheadPointer = samplePointer + lookaheadOffset * chn.pModSample->GetBytesPerSample();
			}
		}
	}

	// Returns the buffer length required to render a certain amount of samples, based on the channel's playback speed.
	static MPT_FORCEINLINE uint32 DistanceToBufferLength(SamplePosition from, SamplePosition to, SamplePosition inc)
	{
		return static_cast<uint32>((to - from - SamplePosition(1)) / inc) + 1;
	}

	// Check how many samples can be rendered without encountering loop or sample end, and also update loop position / direction
	MPT_FORCEINLINE uint32 GetSampleCount(ModChannel &chn, uint32 nSamples, bool ITPingPongMode) const
	{
		int32 nLoopStart = chn.dwFlags[CHN_LOOP] ? chn.nLoopStart : 0;
		SamplePosition nInc = chn.increment;

		if ((nSamples <= 0) || nInc.IsZero() || (!chn.nLength)) return 0;

		// Part 1: Making sure the play position is valid, and if necessary, invert the play direction in case we reached a loop boundary of a ping-pong loop.
		chn.pCurrentSample = samplePointer;

		// Under zero ?
		if (chn.position.GetInt() < nLoopStart)
		{
			if (nInc.IsNegative())
			{
				// Invert loop for bidi loops
				chn.position = SamplePosition(nLoopStart + nLoopStart, 0) - chn.position;
				if ((chn.position.GetInt() < nLoopStart) || (chn.position.GetUInt() >= (nLoopStart + chn.nLength) / 2))
				{
					chn.position.Set(nLoopStart, 0);
				}
				nInc.Negate();
				chn.increment = nInc;
				if(chn.dwFlags[CHN_PINGPONGLOOP])
				{
					chn.dwFlags.reset(CHN_PINGPONGFLAG); // go forward
				} else
				{
					chn.dwFlags.set(CHN_PINGPONGFLAG);
					chn.position.SetInt(chn.nLength - 1);
					chn.increment.Negate();
				}
				if(!chn.dwFlags[CHN_LOOP] || chn.position.GetUInt() >= chn.nLength)
				{
					chn.position.Set(chn.nLength);
					return 0;
				}
			} else
			{
				// We probably didn't hit the loop end yet (first loop), so we do nothing
				if (chn.position.GetInt() < 0) chn.position.SetInt(0);
			}
		} else if (chn.position.GetUInt() >= chn.nLength)
		{
			// Past the end
			if(!chn.dwFlags[CHN_LOOP]) return 0; // not looping -> stop this channel
			if(chn.dwFlags[CHN_PINGPONGLOOP])
			{
				// Invert loop
				if (nInc.IsPositive())
				{
					nInc.Negate();
					chn.increment = nInc;
				}
				chn.dwFlags.set(CHN_PINGPONGFLAG);
				// adjust loop position

				SamplePosition invFract = chn.position.GetInvertedFract();
				chn.position = SamplePosition(chn.nLength - (chn.position.GetInt() - chn.nLength) - invFract.GetInt(), invFract.GetFract());
				if ((chn.position.GetUInt() <= chn.nLoopStart) || (chn.position.GetUInt() >= chn.nLength))
				{
					// Impulse Tracker's software mixer would put a -2 (instead of -1) in the following line (doesn't happen on a GUS)
					chn.position.SetInt(chn.nLength - std::min<SmpLength>(chn.nLength, ITPingPongMode ? 2 : 1));
				}
			} else
			{
				if (nInc.IsNegative()) // This is a bug
				{
					nInc.Negate();
					chn.increment = nInc;
				}
				// Restart at loop start
				chn.position += SamplePosition(nLoopStart - chn.nLength, 0);
				MPT_ASSERT(chn.position.GetInt() >= nLoopStart);
				// Interpolate correctly after wrapping around
				chn.dwFlags.set(CHN_WRAPPED_LOOP);
			}
		}

		// Part 2: Compute how many samples we can render until we reach the end of sample / loop boundary / etc.

		SamplePosition nPos = chn.position;
		// too big increment, and/or too small loop length
		if (nPos.GetInt() < nLoopStart)
		{
			if (nPos.IsNegative() || nInc.IsNegative()) return 0;
		}
		if (nPos.IsNegative() || nPos.GetUInt() >= chn.nLength) return 0;
		uint32 nSmpCount = nSamples;
		SamplePosition nInv = nInc;
		if (nInc.IsNegative())
		{
			nInv.Negate();
		}
		LimitMax(nSamples, maxSamples);
		SamplePosition incSamples = nInc * (nSamples - 1);
		int32 nPosDest = (nPos + incSamples).GetInt();

		const SmpLength nPosInt = nPos.GetUInt();
		const bool isAtLoopStart = (nPosInt >= chn.nLoopStart && nPosInt < chn.nLoopStart + InterpolationMaxLookahead);
		if(!isAtLoopStart)
		{
			chn.dwFlags.reset(CHN_WRAPPED_LOOP);
		}

		// Loop wrap-around magic.
		bool checkDest = true;
		if(lookaheadPointer != nullptr)
		{
			if(nPos.GetUInt() >= lookaheadStart)
			{
#if 0
				const uint32 oldCount = nSmpCount;

				// When going backwards - we can only go back up to lookaheadStart.
				// When going forwards - read through the whole pre-computed wrap-around buffer if possible.
				// TODO: ProTracker sample swapping needs hard cut at sample end.
				int32 samplesToRead = nInc.IsNegative()
					? (nPosInt - lookaheadStart)
					//: 2 * InterpolationMaxLookahead - (nPosInt - mixLoopState.lookaheadStart);
					: (chn.nLoopEnd - nPosInt);
				//LimitMax(samplesToRead, chn.nLoopEnd - chn.nLoopStart);
				nSmpCount = SamplesToBufferLength(samplesToRead, chn);
				Limit(nSmpCount, 1u, oldCount);
#else
				if (nInc.IsNegative())
				{
					nSmpCount = DistanceToBufferLength(SamplePosition(lookaheadStart, 0), nPos, nInv);
				} else
				{
					nSmpCount = DistanceToBufferLength(nPos, SamplePosition(chn.nLoopEnd, 0), nInv);
				}
#endif
				chn.pCurrentSample = lookaheadPointer;
				checkDest = false;
			} else if(chn.dwFlags[CHN_WRAPPED_LOOP] && isAtLoopStart)
			{
				// We just restarted the loop, so interpolate correctly after wrapping around
				nSmpCount = DistanceToBufferLength(nPos, SamplePosition(nLoopStart + InterpolationMaxLookahead, 0), nInv);
				chn.pCurrentSample = lookaheadPointer + (chn.nLoopEnd - nLoopStart) * chn.pModSample->GetBytesPerSample();
				checkDest = false;
			} else if(nInc.IsPositive() && static_cast<SmpLength>(nPosDest) >= lookaheadStart && nSmpCount > 1)
			{
				// We shouldn't read that far if we're not using the pre-computed wrap-around buffer.
				nSmpCount = DistanceToBufferLength(nPos, SamplePosition(lookaheadStart, 0), nInv);
				checkDest = false;
			}
		}

		if(checkDest)
		{
			// Fix up sample count if target position is invalid
			if (nInc.IsNegative())
			{
				if (nPosDest < nLoopStart)
				{
					nSmpCount = DistanceToBufferLength(SamplePosition(chn.nLoopStart, 0), nPos, nInv);
				}
			} else
			{
				if (nPosDest >= (int32)chn.nLength)
				{
					nSmpCount = DistanceToBufferLength(nPos, SamplePosition(chn.nLength, 0), nInv);
				}
			}
		}

		Limit(nSmpCount, 1u, nSamples);

#ifdef _DEBUG
		{
			SmpLength posDest = (nPos + nInc * (nSmpCount - 1)).GetUInt();
			if (posDest < 0 || posDest > chn.nLength)
			{
				// We computed an invalid delta!
				MPT_ASSERT_NOTREACHED();
				return 0;
			}
		}
#endif

		return nSmpCount;
	}
};


// Render count * number of channels samples
void CSoundFile::CreateStereoMix(int count)
{
	mixsample_t *pOfsL, *pOfsR;

	if (!count) return;

	// Resetting sound buffer
	StereoFill(MixSoundBuffer, count, gnDryROfsVol, gnDryLOfsVol);
	if(m_MixerSettings.gnChannels > 2) InitMixBuffer(MixRearBuffer, count*2);

	CHANNELINDEX nchmixed = 0;

	const bool ITPingPongMode = m_playBehaviour[kITPingPongMode];

	for(uint32 nChn = 0; nChn < m_nMixChannels; nChn++)
	{
		ModChannel &chn = m_PlayState.Chn[m_PlayState.ChnMix[nChn]];

		if(!chn.pCurrentSample) continue;
		pOfsR = &gnDryROfsVol;
		pOfsL = &gnDryLOfsVol;

		uint32 functionNdx = MixFuncTable::ResamplingModeToMixFlags(static_cast<ResamplingMode>(chn.resamplingMode));
		if(chn.dwFlags[CHN_16BIT]) functionNdx |= MixFuncTable::ndx16Bit;
		if(chn.dwFlags[CHN_STEREO]) functionNdx |= MixFuncTable::ndxStereo;
#ifndef NO_FILTER
		if(chn.dwFlags[CHN_FILTER]) functionNdx |= MixFuncTable::ndxFilter;
#endif

		mixsample_t *pbuffer = MixSoundBuffer;
#ifndef NO_REVERB
		if(((m_MixerSettings.DSPMask & SNDDSP_REVERB) && !chn.dwFlags[CHN_NOREVERB]) || chn.dwFlags[CHN_REVERB])
		{
			pbuffer = m_Reverb.GetReverbSendBuffer(count);
			pOfsR = &m_Reverb.gnRvbROfsVol;
			pOfsL = &m_Reverb.gnRvbLOfsVol;
		}
#endif
		if(chn.dwFlags[CHN_SURROUND] && m_MixerSettings.gnChannels > 2)
			pbuffer = MixRearBuffer;

		//Look for plugins associated with this implicit tracker channel.
#ifndef NO_PLUGINS
		PLUGINDEX nMixPlugin = GetBestPlugin(m_PlayState.ChnMix[nChn], PrioritiseInstrument, RespectMutes);

		if ((nMixPlugin > 0) && (nMixPlugin <= MAX_MIXPLUGINS) && m_MixPlugins[nMixPlugin - 1].pMixPlugin != nullptr)
		{
			// Render into plugin buffer instead of global buffer
			SNDMIXPLUGINSTATE &mixState = m_MixPlugins[nMixPlugin - 1].pMixPlugin->m_MixState;
			if (mixState.pMixBuffer)
			{
				pbuffer = mixState.pMixBuffer;
				pOfsR = &mixState.nVolDecayR;
				pOfsL = &mixState.nVolDecayL;
				if (!(mixState.dwFlags & SNDMIXPLUGINSTATE::psfMixReady))
				{
					StereoFill(pbuffer, count, *pOfsR, *pOfsL);
					mixState.dwFlags |= SNDMIXPLUGINSTATE::psfMixReady;
				}
			}
		}
#endif // NO_PLUGINS

		MixLoopState mixLoopState(chn);

		////////////////////////////////////////////////////
		CHANNELINDEX naddmix = 0;
		int nsamples = count;
		// Keep mixing this sample until the buffer is filled.
		do
		{
			uint32 nrampsamples = nsamples;
			int32 nSmpCount;
			if(chn.nRampLength > 0)
			{
				if (nrampsamples > chn.nRampLength) nrampsamples = chn.nRampLength;
			}

			if((nSmpCount = mixLoopState.GetSampleCount(chn, nrampsamples, ITPingPongMode)) <= 0)
			{
				// Stopping the channel
				chn.pCurrentSample = nullptr;
				chn.nLength = 0;
				chn.position.Set(0);
				chn.nRampLength = 0;
				EndChannelOfs(chn, pbuffer, nsamples);
				*pOfsR += chn.nROfs;
				*pOfsL += chn.nLOfs;
				chn.nROfs = chn.nLOfs = 0;
				chn.dwFlags.reset(CHN_PINGPONGFLAG);
				break;
			}

			// Should we mix this channel ?
			if((nchmixed >= m_MixerSettings.m_nMaxMixChannels)				// Too many channels
				|| (!chn.nRampLength && !(chn.leftVol | chn.rightVol)))		// Channel is completely silent
			{
				chn.position += chn.increment * nSmpCount;
				chn.nROfs = chn.nLOfs = 0;
				pbuffer += nSmpCount * 2;
				naddmix = 0;
			} else
			{
				// Do mixing
				mixsample_t *pbufmax = pbuffer + (nSmpCount * 2);
				chn.nROfs = - *(pbufmax-2);
				chn.nLOfs = - *(pbufmax-1);

#ifdef _DEBUG
				SamplePosition targetpos = chn.position + chn.increment * nSmpCount;
#endif
				MixFuncTable::Functions[functionNdx | (chn.nRampLength ? MixFuncTable::ndxRamp : 0)](chn, m_Resampler, pbuffer, nSmpCount);
#ifdef _DEBUG
				MPT_ASSERT(chn.position.GetUInt() == targetpos.GetUInt());
#endif

				chn.nROfs += *(pbufmax-2);
				chn.nLOfs += *(pbufmax-1);
				pbuffer = pbufmax;
				naddmix = 1;
			}
			nsamples -= nSmpCount;
			if (chn.nRampLength)
			{
				if (chn.nRampLength <= static_cast<uint32>(nSmpCount))
				{
					// Ramping is done
					chn.nRampLength = 0;
					chn.leftVol = chn.newLeftVol;
					chn.rightVol = chn.newRightVol;
					chn.rightRamp = chn.leftRamp = 0;
					if(chn.dwFlags[CHN_NOTEFADE] && !chn.nFadeOutVol)
					{
						chn.nLength = 0;
						chn.pCurrentSample = nullptr;
					}
				} else
				{
					chn.nRampLength -= nSmpCount;
				}
			}

			if(chn.position.GetUInt() >= chn.nLoopEnd && chn.dwFlags[CHN_LOOP])
			{
				if(m_playBehaviour[kMODSampleSwap] && chn.nNewIns && chn.nNewIns <= GetNumSamples() && chn.pModSample != &Samples[chn.nNewIns])
				{
					// ProTracker compatibility: Instrument changes without a note do not happen instantly, but rather when the sample loop has finished playing.
					// Test case: PTInstrSwap.mod
					const ModSample &smp = Samples[chn.nNewIns];
					chn.pModSample = &smp;
					chn.pCurrentSample = smp.pSample;
					chn.dwFlags = (chn.dwFlags & CHN_CHANNELFLAGS) | smp.uFlags;
					chn.nLength = smp.uFlags[CHN_LOOP] ? smp.nLoopEnd : smp.nLength;
					chn.nLoopStart = smp.nLoopStart;
					chn.nLoopEnd = smp.nLoopEnd;
					chn.position.SetInt(chn.nLoopStart);
					mixLoopState.UpdateLookaheadPointers(chn);
					if(!chn.pCurrentSample)
					{
						break;
					}
				} else if(m_playBehaviour[kMODOneShotLoops] && chn.nLoopStart == 0)
				{
					// ProTracker "oneshot" loops (if loop start is 0, play the whole sample once and then repeat until loop end)
					chn.position.SetInt(0);
					chn.nLoopEnd = chn.nLength = chn.pModSample->nLoopEnd;
				}
			}
		} while(nsamples > 0);

		// Restore sample pointer in case it got changed through loop wrap-around
		chn.pCurrentSample = mixLoopState.samplePointer;
		nchmixed += naddmix;
	
#ifndef NO_PLUGINS
		if(naddmix && nMixPlugin > 0 && nMixPlugin <= MAX_MIXPLUGINS && m_MixPlugins[nMixPlugin - 1].pMixPlugin)
		{
			m_MixPlugins[nMixPlugin - 1].pMixPlugin->ResetSilence();
		}
#endif // NO_PLUGINS
	}
	m_nMixStat = std::max<CHANNELINDEX>(m_nMixStat, nchmixed);
}


void CSoundFile::ProcessPlugins(uint32 nCount)
{
#ifndef NO_PLUGINS
	// If any sample channels are active or any plugin has some input, possibly suspended master plugins need to be woken up.
	bool masterHasInput = (m_nMixStat > 0);

#ifdef MPT_INTMIXER
	const float IntToFloat = m_PlayConfig.getIntToFloat();
	const float FloatToInt = m_PlayConfig.getFloatToInt();
#endif // MPT_INTMIXER

	// Setup float inputs from samples
	for(PLUGINDEX plug = 0; plug < MAX_MIXPLUGINS; plug++)
	{
		SNDMIXPLUGIN &plugin = m_MixPlugins[plug];
		if(plugin.pMixPlugin != nullptr
			&& plugin.pMixPlugin->m_MixState.pMixBuffer != nullptr
			&& plugin.pMixPlugin->m_mixBuffer.Ok())
		{
			IMixPlugin *mixPlug = plugin.pMixPlugin;
			SNDMIXPLUGINSTATE &state = mixPlug->m_MixState;

			//We should only ever reach this point if the song is playing.
			if (!mixPlug->IsSongPlaying())
			{
				//Plugin doesn't know it is in a song that is playing;
				//we must have added it during playback. Initialise it!
				mixPlug->NotifySongPlaying(true);
				mixPlug->Resume();
			}


			// Setup float input
			float *plugInputL = mixPlug->m_mixBuffer.GetInputBuffer(0);
			float *plugInputR = mixPlug->m_mixBuffer.GetInputBuffer(1);
			if (state.dwFlags & SNDMIXPLUGINSTATE::psfMixReady)
			{
#ifdef MPT_INTMIXER
				StereoMixToFloat(state.pMixBuffer, plugInputL, plugInputR, nCount, IntToFloat);
#else
				DeinterleaveStereo(pState->pMixBuffer, plugInputL, plugInputR, nCount);
#endif // MPT_INTMIXER
			} else if (state.nVolDecayR || state.nVolDecayL)
			{
				StereoFill(state.pMixBuffer, nCount, state.nVolDecayR, state.nVolDecayL);
#ifdef MPT_INTMIXER
				StereoMixToFloat(state.pMixBuffer, plugInputL, plugInputR, nCount, IntToFloat);
#else
				DeinterleaveStereo(pState->pMixBuffer, plugInputL, plugInputR, nCount);
#endif // MPT_INTMIXER
			} else
			{
				memset(plugInputL, 0, nCount * sizeof(plugInputL[0]));
				memset(plugInputR, 0, nCount * sizeof(plugInputR[0]));
			}
			state.dwFlags &= ~SNDMIXPLUGINSTATE::psfMixReady;
			
			if(!plugin.IsMasterEffect() && !(state.dwFlags & SNDMIXPLUGINSTATE::psfSilenceBypass))
			{
				masterHasInput = true;
			}
		}
	}
	// Convert mix buffer
#ifdef MPT_INTMIXER
	StereoMixToFloat(MixSoundBuffer, MixFloatBuffer[0], MixFloatBuffer[1], nCount, IntToFloat);
#else
	DeinterleaveStereo(MixSoundBuffer, MixFloatBuffer[0], MixFloatBuffer[1], nCount);
#endif // MPT_INTMIXER
	float *pMixL = MixFloatBuffer[0];
	float *pMixR = MixFloatBuffer[1];

	const bool positionChanged = HasPositionChanged();

	// Process Plugins
	for(PLUGINDEX plug = 0; plug < MAX_MIXPLUGINS; plug++)
	{
		SNDMIXPLUGIN &plugin = m_MixPlugins[plug];
		if (plugin.pMixPlugin != nullptr
			&& plugin.pMixPlugin->m_MixState.pMixBuffer != nullptr
			&& plugin.pMixPlugin->m_mixBuffer.Ok())
		{
			IMixPlugin *pObject = plugin.pMixPlugin;
			if(!plugin.IsMasterEffect() && !plugin.pMixPlugin->ShouldProcessSilence() && !(plugin.pMixPlugin->m_MixState.dwFlags & SNDMIXPLUGINSTATE::psfHasInput))
			{
				// If plugin has no inputs and isn't a master plugin, we shouldn't let it process silence if possible.
				// I have yet to encounter a plugin which actually sets this flag.
				bool hasInput = false;
				for(PLUGINDEX inPlug = 0; inPlug < plug; inPlug++)
				{
					if(m_MixPlugins[inPlug].GetOutputPlugin() == plug)
					{
						hasInput = true;
						break;
					}
				}
				if(!hasInput)
				{
					continue;
				}
			}

			bool isMasterMix = false;
			float *plugInputL = pObject->m_mixBuffer.GetInputBuffer(0);
			float *plugInputR = pObject->m_mixBuffer.GetInputBuffer(1);

			if (pMixL == plugInputL)
			{
				isMasterMix = true;
				pMixL = MixFloatBuffer[0];
				pMixR = MixFloatBuffer[1];
			}
			SNDMIXPLUGINSTATE &state = plugin.pMixPlugin->m_MixState;
			float *pOutL = pMixL;
			float *pOutR = pMixR;

			if (!plugin.IsOutputToMaster())
			{
				PLUGINDEX nOutput = plugin.GetOutputPlugin();
				if(nOutput > plug && nOutput != PLUGINDEX_INVALID
					&& m_MixPlugins[nOutput].pMixPlugin != nullptr)
				{
					IMixPlugin *outPlugin = m_MixPlugins[nOutput].pMixPlugin;
					if(!(state.dwFlags & SNDMIXPLUGINSTATE::psfSilenceBypass)) outPlugin->ResetSilence();

					if(outPlugin->m_mixBuffer.Ok())
					{
						pOutL = outPlugin->m_mixBuffer.GetInputBuffer(0);
						pOutR = outPlugin->m_mixBuffer.GetInputBuffer(1);
					}
				}
			}

			/*
			if (plugin.multiRouting) {
				int nOutput=0;
				for (int nOutput=0; nOutput < plugin.nOutputs / 2; nOutput++) {
					destinationPlug = plugin.multiRoutingDestinations[nOutput];
					pOutState = m_MixPlugins[destinationPlug].pMixState;
					pOutputs[2 * nOutput] = plugInputL;
					pOutputs[2 * (nOutput + 1)] = plugInputR;
				}

			}*/

			if (plugin.IsMasterEffect())
			{
				if (!isMasterMix)
				{
					float *pInL = plugInputL;
					float *pInR = plugInputR;
					for (uint32 i=0; i<nCount; i++)
					{
						pInL[i] += pMixL[i];
						pInR[i] += pMixR[i];
						pMixL[i] = 0;
						pMixR[i] = 0;
					}
				}
				pMixL = pOutL;
				pMixR = pOutR;

				if(masterHasInput)
				{
					// Samples or plugins are being rendered, so turn off auto-bypass for this master effect.
					if(plugin.pMixPlugin != nullptr) plugin.pMixPlugin->ResetSilence();
					SNDMIXPLUGIN *chain = &plugin;
					PLUGINDEX out = chain->GetOutputPlugin(), prevOut = plug;
					while(out > prevOut && out < MAX_MIXPLUGINS)
					{
						chain = &m_MixPlugins[out];
						prevOut = out;
						out = chain->GetOutputPlugin();
						if(chain->pMixPlugin)
						{
							chain->pMixPlugin->ResetSilence();
						}
					}
				}
			}

			if(plugin.IsBypassed() || (plugin.IsAutoSuspendable() && (state.dwFlags & SNDMIXPLUGINSTATE::psfSilenceBypass)))
			{
				const float * const pInL = plugInputL;
				const float * const pInR = plugInputR;
				for (uint32 i=0; i<nCount; i++)
				{
					pOutL[i] += pInL[i];
					pOutR[i] += pInR[i];
				}
			} else
			{
				if(positionChanged)
					pObject->PositionChanged();
				pObject->Process(pOutL, pOutR, nCount);

				state.inputSilenceCount += nCount;
				if(plugin.IsAutoSuspendable() && pObject->GetNumOutputChannels() > 0 && state.inputSilenceCount >= m_MixerSettings.gdwMixingFreq * 4)
				{
					bool isSilent = true;
					for(uint32 i = 0; i < nCount; i++)
					{
						if(pOutL[i] >= FLT_EPSILON || pOutL[i] <= -FLT_EPSILON
							|| pOutR[i] >= FLT_EPSILON || pOutR[i] <= -FLT_EPSILON)
						{
							isSilent = false;
							break;
						}
					}
					if(isSilent)
					{
						state.dwFlags |= SNDMIXPLUGINSTATE::psfSilenceBypass;
					} else
					{
						state.inputSilenceCount = 0;
					}
				}
			}
			state.dwFlags &= ~SNDMIXPLUGINSTATE::psfHasInput;
		}
	}
#ifdef MPT_INTMIXER
	FloatToStereoMix(pMixL, pMixR, MixSoundBuffer, nCount, FloatToInt);
#else
	InterleaveStereo(pMixL, pMixR, MixSoundBuffer, nCount);
#endif // MPT_INTMIXER

#else
	MPT_UNREFERENCED_PARAMETER(nCount);
#endif // NO_PLUGINS
}


OPENMPT_NAMESPACE_END
