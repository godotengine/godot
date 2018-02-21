/*
 * I3DL2Reverb.cpp
 * ---------------
 * Purpose: Implementation of the DMO I3DL2Reverb DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_PLUGINS
#include "../../Sndfile.h"
#include "I3DL2Reverb.h"
#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_PLUGINS

namespace DMO
{

void I3DL2Reverb::DelayLine::Init(int32 ms, int32 padding, uint32 sampleRate, int32 delayTap)
{
	m_length = Util::muldiv(sampleRate, ms, 1000) + padding;
	m_position = 0;
	SetDelayTap(delayTap);
	assign(m_length, 0.0f);
}


void I3DL2Reverb::DelayLine::SetDelayTap(int32 delayTap)
{
	if(m_length > 0)
		m_delayPosition = (delayTap + m_position + m_length) % m_length;
}


void I3DL2Reverb::DelayLine::Advance()
{
	if(--m_position < 0)
		m_position += m_length;
	if(--m_delayPosition < 0)
		m_delayPosition += m_length;
}


MPT_FORCEINLINE void I3DL2Reverb::DelayLine::Set(float value)
{
	at(m_position) = value;
}


float I3DL2Reverb::DelayLine::Get(int32 offset) const
{
	offset = (offset + m_position) % m_length;
	if(offset < 0)
		offset += m_length;
	return at(offset);
}


MPT_FORCEINLINE float I3DL2Reverb::DelayLine::Get() const
{
	return at(m_delayPosition);
}


IMixPlugin* I3DL2Reverb::Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
{
	return new (std::nothrow) I3DL2Reverb(factory, sndFile, mixStruct);
}


I3DL2Reverb::I3DL2Reverb(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
	: IMixPlugin(factory, sndFile, mixStruct)
	, m_recalcParams(true)
{
	m_param[kI3DL2ReverbRoom] = 0.9f;
	m_param[kI3DL2ReverbRoomHF] = 0.99f;
	m_param[kI3DL2ReverbRoomRolloffFactor] = 0.0f;
	m_param[kI3DL2ReverbDecayTime] = 0.07f;
	m_param[kI3DL2ReverbDecayHFRatio] = 0.3842105f;
	m_param[kI3DL2ReverbReflections] = 0.672545433f;
	m_param[kI3DL2ReverbReflectionsDelay] = 0.233333333f;
	m_param[kI3DL2ReverbReverb] = 0.85f;
	m_param[kI3DL2ReverbReverbDelay] = 0.11f;
	m_param[kI3DL2ReverbDiffusion] = 1.0f;
	m_param[kI3DL2ReverbDensity] = 1.0f;
	m_param[kI3DL2ReverbHFReference] = (5000.0f - 20.0f) / 19980.0f;
	m_param[kI3DL2ReverbQuality] = 2.0f / 3.0f;

	m_mixBuffer.Initialize(2, 2);
	InsertIntoFactoryList();
}


void I3DL2Reverb::Process(float *pOutL, float *pOutR, uint32 numFrames)
{
	if(m_recalcParams)
	{
		auto sampleRate = m_effectiveSampleRate;
		RecalculateI3DL2ReverbParams();
		// Resize and clear delay lines if quality has changed
		if(sampleRate != m_effectiveSampleRate)
			PositionChanged();
	}

	if(!m_ok || !m_mixBuffer.Ok())
		return;

	const float *in[2] = { m_mixBuffer.GetInputBuffer(0), m_mixBuffer.GetInputBuffer(1) };
	float *out[2] = { m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1) };

	uint32 frames = numFrames;
	if(!(m_quality & kFullSampleRate) && m_remain && frames > 0)
	{
		// Remaining frame from previous render call
		frames--;
		*(out[0]++) = m_prevL;
		*(out[1]++) = m_prevR;
		in[0]++;
		in[1]++;
		m_remain = false;
	}
	
	while(frames > 0)
	{
		// Apply room filter and insert into early reflection delay lines
		const float inL = *(in[0]++);
		const float inRoomL = (m_filterHist[12] - inL) * m_roomFilter + inL;
		m_filterHist[12] = inRoomL;
		m_delayLines[15].Set(inRoomL);

		const float inR = *(in[1]++);
		const float inRoomR = (m_filterHist[13] - inR) * m_roomFilter + inR;
		m_filterHist[13] = inRoomR;
		m_delayLines[16].Set(inRoomR);

		// Early reflections (left)
		float earlyL = m_delayLines[15].Get(m_earlyTaps[0][1]) * 0.68f
			- m_delayLines[15].Get(m_earlyTaps[0][2]) * 0.5f
			- m_delayLines[15].Get(m_earlyTaps[0][3]) * 0.62f
			- m_delayLines[15].Get(m_earlyTaps[0][4]) * 0.5f
			- m_delayLines[15].Get(m_earlyTaps[0][5]) * 0.62f;
		if(m_quality & kMoreDelayLines)
		{
			float earlyL2 = earlyL;
			earlyL = m_delayLines[13].Get() + earlyL * 0.618034f;
			m_delayLines[13].Set(earlyL2 - earlyL * 0.618034f);
		}
		const float earlyRefOutL = earlyL * m_ERLevel;
		m_filterHist[15] = m_delayLines[15].Get(m_earlyTaps[0][0]) + m_filterHist[15];
		m_filterHist[16] = m_delayLines[16].Get(m_earlyTaps[1][0]) + m_filterHist[16];

		// Lots of slightly different copy-pasta ahead
		float reverbL1, reverbL2, reverbL3, reverbR1, reverbR2, reverbR3;

		reverbL1 = -m_filterHist[15] * 0.707f;
		reverbL2 = m_filterHist[16] * 0.707f + reverbL1;
		reverbR2 = reverbL1 - m_filterHist[16] * 0.707f;

		m_filterHist[5] = (m_filterHist[5] - m_delayLines[5].Get()) * m_delayCoeffs[5][1] + m_delayLines[5].Get();
		reverbL1 = m_filterHist[5] * m_delayCoeffs[5][0] + reverbL2 * m_diffusion;
		m_delayLines[5].Set(reverbL2 - reverbL1 * m_diffusion);
		reverbL2 = reverbL1;
		reverbL3 = -0.15f * reverbL1;

		m_filterHist[4] = (m_filterHist[4] - m_delayLines[4].Get()) * m_delayCoeffs[4][1] + m_delayLines[4].Get();
		reverbL1 = m_filterHist[4] * m_delayCoeffs[4][0] + reverbL2 * m_diffusion;
		m_delayLines[4].Set(reverbL2 - reverbL1 * m_diffusion);
		reverbL2 = reverbL1;
		reverbL3 -= reverbL1 * 0.2f;

		if(m_quality & kMoreDelayLines)
		{
			m_filterHist[3] = (m_filterHist[3] - m_delayLines[3].Get()) * m_delayCoeffs[3][1] + m_delayLines[3].Get();
			reverbL1 = m_filterHist[3] * m_delayCoeffs[3][0] + reverbL2 * m_diffusion;
			m_delayLines[3].Set(reverbL2 - reverbL1 * m_diffusion);
			reverbL2 = reverbL1;
			reverbL3 += 0.35f * reverbL1;

			m_filterHist[2] = (m_filterHist[2] - m_delayLines[2].Get()) * m_delayCoeffs[2][1] + m_delayLines[2].Get();
			reverbL1 = m_filterHist[2] * m_delayCoeffs[2][0] + reverbL2 * m_diffusion;
			m_delayLines[2].Set(reverbL2 - reverbL1 * m_diffusion);
			reverbL2 = reverbL1;
			reverbL3 -= reverbL1 * 0.38f;
		}
		m_delayLines[17].Set(reverbL2);

		reverbL1 = m_delayLines[17].Get() * m_delayCoeffs[12][0];
		m_filterHist[17] = (m_filterHist[17] - reverbL1) * m_delayCoeffs[12][1] + reverbL1;

		m_filterHist[1] = (m_filterHist[1] - m_delayLines[1].Get()) * m_delayCoeffs[1][1] + m_delayLines[1].Get();
		reverbL1 = m_filterHist[17] * m_diffusion + m_filterHist[1] * m_delayCoeffs[1][0];
		m_delayLines[1].Set(m_filterHist[17] - reverbL1 * m_diffusion);
		reverbL2 = reverbL1;
		float reverbL4 = reverbL1 * 0.38f;

		m_filterHist[0] = (m_filterHist[0] - m_delayLines[0].Get()) * m_delayCoeffs[0][1] + m_delayLines[0].Get();
		reverbL1 = m_filterHist[0] * m_delayCoeffs[0][0] + reverbL2 * m_diffusion;
		m_delayLines[0].Set(reverbL2 - reverbL1 * m_diffusion);
		reverbL3 -= reverbL1 * 0.38f;
		m_filterHist[15] = reverbL1;
		
		// Early reflections (right)
		float earlyR = m_delayLines[16].Get(m_earlyTaps[1][1]) * 0.707f
			- m_delayLines[16].Get(m_earlyTaps[1][2]) * 0.6f
			- m_delayLines[16].Get(m_earlyTaps[1][3]) * 0.5f
			- m_delayLines[16].Get(m_earlyTaps[1][4]) * 0.6f
			- m_delayLines[16].Get(m_earlyTaps[1][5]) * 0.5f;
		if(m_quality & kMoreDelayLines)
		{
			float earlyR2 = earlyR;
			earlyR = m_delayLines[14].Get() + earlyR * 0.618034f;
			m_delayLines[14].Set(earlyR2 - earlyR * 0.618034f);
		}
		const float earlyRefOutR = earlyR * m_ERLevel;

		m_filterHist[11] = (m_filterHist[11] - m_delayLines[11].Get()) * m_delayCoeffs[11][1] + m_delayLines[11].Get();
		reverbR1 = m_filterHist[11] * m_delayCoeffs[11][0] + reverbR2 * m_diffusion;
		m_delayLines[11].Set(reverbR2 - reverbR1 * m_diffusion);
		reverbR2 = reverbR1;

		m_filterHist[10] = (m_filterHist[10] - m_delayLines[10].Get()) * m_delayCoeffs[10][1] + m_delayLines[10].Get();
		reverbR1 = m_filterHist[10] * m_delayCoeffs[10][0] + reverbR2 * m_diffusion;
		m_delayLines[10].Set(reverbR2 - reverbR1 * m_diffusion);
		reverbR3 = reverbL4 - reverbR2 * 0.15f - reverbR1 * 0.2f;
		reverbR2 = reverbR1;

		if(m_quality & kMoreDelayLines)
		{
			m_filterHist[9] = (m_filterHist[9] - m_delayLines[9].Get()) * m_delayCoeffs[9][1] + m_delayLines[9].Get();
			reverbR1 = m_filterHist[9] * m_delayCoeffs[9][0] + reverbR2 * m_diffusion;
			m_delayLines[9].Set(reverbR2 - reverbR1 * m_diffusion);
			reverbR2 = reverbR1;
			reverbR3 += reverbR1 * 0.35f;

			m_filterHist[8] = (m_filterHist[8] - m_delayLines[8].Get()) * m_delayCoeffs[8][1] + m_delayLines[8].Get();
			reverbR1 = m_filterHist[8] * m_delayCoeffs[8][0] + reverbR2 * m_diffusion;
			m_delayLines[8].Set(reverbR2 - reverbR1 * m_diffusion);
			reverbR2 = reverbR1;
			reverbR3 -= reverbR1 * 0.38f;
		}
		m_delayLines[18].Set(reverbR2);

		reverbR1 = m_delayLines[18].Get() * m_delayCoeffs[12][0];
		m_filterHist[18] = (m_filterHist[18] - reverbR1) * m_delayCoeffs[12][1] + reverbR1;
			
		m_filterHist[7] = (m_filterHist[7] - m_delayLines[7].Get()) * m_delayCoeffs[7][1] + m_delayLines[7].Get();
		reverbR1 = m_filterHist[18] * m_diffusion + m_filterHist[7] * m_delayCoeffs[7][0];
		m_delayLines[7].Set(m_filterHist[18] - reverbR1 * m_diffusion);
		reverbR2 = reverbR1;

		float lateRevOutL = (reverbL3 + reverbR1 * 0.38f) * m_ReverbLevelL;

		m_filterHist[6] = (m_filterHist[6] - m_delayLines[6].Get()) * m_delayCoeffs[6][1] + m_delayLines[6].Get();
		reverbR1 = m_filterHist[6] * m_delayCoeffs[6][0] + reverbR2 * m_diffusion;
		m_delayLines[6].Set(reverbR2 - reverbR1 * m_diffusion);
		m_filterHist[16] = reverbR1;

		float lateRevOutR = (reverbR3 - reverbR1 * 0.38f) * m_ReverbLevelR;

		float outL = earlyRefOutL + lateRevOutL;
		float outR = earlyRefOutR + lateRevOutR;

		for(std::size_t d = 0; d < mpt::size(m_delayLines); d++)
			m_delayLines[d].Advance();

		if(!(m_quality & kFullSampleRate))
		{
			*(out[0]++) = (outL + m_prevL) * 0.5f;
			*(out[1]++) = (outR + m_prevR) * 0.5f;
			m_prevL = outL;
			m_prevR = outR;
			in[0]++;
			in[1]++;
			if(frames-- == 1)
			{
				m_remain = true;
				break;
			}
		}
		*(out[0]++) = outL;
		*(out[1]++) = outR;
		frames--;
	}

	ProcessMixOps(pOutL, pOutR, m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1), numFrames);
}


PlugParamValue I3DL2Reverb::GetParameter(PlugParamIndex index)
{
	if(index < kI3DL2ReverbNumParameters)
	{
		return m_param[index];
	}
	return 0;
}


void I3DL2Reverb::SetParameter(PlugParamIndex index, PlugParamValue value)
{
	if(index < kI3DL2ReverbNumParameters)
	{
		Limit(value, 0.0f, 1.0f);
		if(index == kI3DL2ReverbQuality)
			value = Util::Round(value * 3.0f) / 3.0f;
		m_param[index] = value;
		m_recalcParams = true;
	}
}


void I3DL2Reverb::Resume()
{
	RecalculateI3DL2ReverbParams();
	PositionChanged();
	m_isResumed = true;
}


void I3DL2Reverb::PositionChanged()
{
	MemsetZero(m_filterHist);
	m_prevL = 0;
	m_prevR = 0;
	m_remain = false;

	try
	{
		uint32 sampleRate = static_cast<uint32>(m_effectiveSampleRate);
		m_delayLines[0].Init(67, 5, sampleRate, m_delayTaps[0]);
		m_delayLines[1].Init(62, 5, sampleRate, m_delayTaps[1]);
		m_delayLines[2].Init(53, 5, sampleRate, m_delayTaps[2]);
		m_delayLines[3].Init(43, 5, sampleRate, m_delayTaps[3]);
		m_delayLines[4].Init(32, 5, sampleRate, m_delayTaps[4]);
		m_delayLines[5].Init(22, 5, sampleRate, m_delayTaps[5]);
		m_delayLines[6].Init(75, 5, sampleRate, m_delayTaps[6]);
		m_delayLines[7].Init(69, 5, sampleRate, m_delayTaps[7]);
		m_delayLines[8].Init(60, 5, sampleRate, m_delayTaps[8]);
		m_delayLines[9].Init(48, 5, sampleRate, m_delayTaps[9]);
		m_delayLines[10].Init(36, 5, sampleRate, m_delayTaps[10]);
		m_delayLines[11].Init(25, 5, sampleRate, m_delayTaps[11]);
		m_delayLines[12].Init(0, 0, 0);	// Dummy for array index consistency with both tap and coefficient arrays
		m_delayLines[13].Init(3, 0, sampleRate, m_delayTaps[13]);
		m_delayLines[14].Init(3, 0, sampleRate, m_delayTaps[14]);
		m_delayLines[15].Init(407, 1, sampleRate);
		m_delayLines[16].Init(400, 1, sampleRate);
		m_delayLines[17].Init(10, 0, sampleRate, -1);
		m_delayLines[18].Init(10, 0, sampleRate, -1);
		m_ok = true;
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		m_ok = false;
		MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
	}
}


#ifdef MODPLUG_TRACKER

CString I3DL2Reverb::GetParamName(PlugParamIndex param)
{
	switch(param)
	{
	case kI3DL2ReverbRoom: return _T("Room");
	case kI3DL2ReverbRoomHF: return _T("RoomHF");
	case kI3DL2ReverbRoomRolloffFactor: return _T("RoomRolloffFactor");
	case kI3DL2ReverbDecayTime: return _T("DecayTime");
	case kI3DL2ReverbDecayHFRatio: return _T("DecayHFRatio");
	case kI3DL2ReverbReflections: return _T("Reflections");
	case kI3DL2ReverbReflectionsDelay: return _T("ReflectionsDelay");
	case kI3DL2ReverbReverb: return _T("Reverb");
	case kI3DL2ReverbReverbDelay: return _T("ReverbDelay");
	case kI3DL2ReverbDiffusion: return _T("Diffusion");
	case kI3DL2ReverbDensity: return _T("Density");
	case kI3DL2ReverbHFReference: return _T("HFRefrence");
	case kI3DL2ReverbQuality: return _T("Quality");
	}
	return CString();
}


CString I3DL2Reverb::GetParamLabel(PlugParamIndex param)
{
	switch(param)
	{
	case kI3DL2ReverbRoom:
	case kI3DL2ReverbRoomHF:
	case kI3DL2ReverbReflections:
	case kI3DL2ReverbReverb:
		return _T("dB");
	case kI3DL2ReverbDecayTime:
	case kI3DL2ReverbReflectionsDelay:
	case kI3DL2ReverbReverbDelay:
		return _T("s");
	case kI3DL2ReverbDiffusion:
	case kI3DL2ReverbDensity:
		return _T("%");
	case kI3DL2ReverbHFReference:
		return _T("Hz");
	}
	return CString();
}


CString I3DL2Reverb::GetParamDisplay(PlugParamIndex param)
{
	static const TCHAR *modes[] = { _T("LQ"), _T("LQ+"), _T("HQ"), _T("HQ+") };
	float value = m_param[param];
	switch(param)
	{
	case kI3DL2ReverbRoom: value = Room() * 0.01f; break;
	case kI3DL2ReverbRoomHF: value = RoomHF() * 0.01f; break;
	case kI3DL2ReverbRoomRolloffFactor: value = RoomRolloffFactor(); break;
	case kI3DL2ReverbDecayTime: value = DecayTime(); break;
	case kI3DL2ReverbDecayHFRatio: value = DecayHFRatio(); break;
	case kI3DL2ReverbReflections: value = Reflections() * 0.01f; break;
	case kI3DL2ReverbReflectionsDelay: value = ReflectionsDelay(); break;
	case kI3DL2ReverbReverb: value = Reverb() * 0.01f; break;
	case kI3DL2ReverbReverbDelay: value = ReverbDelay(); break;
	case kI3DL2ReverbDiffusion: value = Diffusion(); break;
	case kI3DL2ReverbDensity: value = Density(); break;
	case kI3DL2ReverbHFReference: value = HFReference(); break;
	case kI3DL2ReverbQuality: return modes[Quality() % 4u];
	}
	CString s;
	s.Format(_T("%.2f"), value);
	return s;
}

#endif // MODPLUG_TRACKER


void I3DL2Reverb::RecalculateI3DL2ReverbParams()
{
	m_quality = Quality();
	m_effectiveSampleRate = static_cast<float>(m_SndFile.GetSampleRate() / ((m_quality & kFullSampleRate) ? 1u : 2u));

	// Diffusion
	m_diffusion = Diffusion() * (0.618034f / 100.0f);
	// Early Reflection Level
	m_ERLevel = std::min(std::pow(10.0f, (Room() + Reflections()) / (100.0f * 20.0f)), 1.0f) * 0.761f;

	// Room Filter
	float roomHF = std::pow(10.0f, RoomHF() / 100.0f / 10.0f);
	if(roomHF == 1.0)
	{
		m_roomFilter = 0.0f;
	} else
	{
		float freq = std::cos(HFReference() * static_cast<float>(2.0 * M_PI) / m_effectiveSampleRate);
		float roomFilter = (freq * (roomHF + roomHF) - 2.0f + std::sqrt(freq * (roomHF * roomHF * freq * 4.0f) + roomHF * 8.0f - roomHF * roomHF * 4.0f - roomHF * freq * 8.0f)) / (roomHF + roomHF - 2.0f);
		m_roomFilter = Clamp(roomFilter, 0.0f, 1.0f);
	}

	SetDelayTaps();
	SetDecayCoeffs();

	m_recalcParams = false;
}


void I3DL2Reverb::SetDelayTaps()
{
	// Early reflections
	static const float delays[] =
	{
		1.0000f, 1.0000f, 0.0000f, 0.1078f, 0.1768f, 0.2727f,
		0.3953f, 0.5386f, 0.6899f, 0.8306f, 0.9400f, 0.9800f,
	};

	const float sampleRate = m_effectiveSampleRate;
	const float reflectionsDelay = ReflectionsDelay();
	const float reverbDelay = std::max(ReverbDelay(), 5.0f / 1000.0f);
	m_earlyTaps[0][0] = static_cast<int32>((reverbDelay + reflectionsDelay + 7.0f / 1000.0f) * sampleRate);
	for(uint32 i = 1; i < 12; i++)
	{
		m_earlyTaps[i % 2u][i / 2u] = static_cast<int32>((reverbDelay * delays[i] + reflectionsDelay) * sampleRate);
	}

	// Late reflections
	float density = std::min((Density() / 100.0f + 0.1f) * 0.9091f, 1.0f);
	float delayL = density * 67.0f / 1000.0f * sampleRate;
	float delayR = density * 75.0f / 1000.0f * sampleRate;
	for(int i = 0, power = 0; i < 6; i++)
	{
		power += i;
		float factor = std::pow(0.93f, power);
		m_delayTaps[i + 0] = static_cast<int32>(delayL * factor);
		m_delayTaps[i + 6] = static_cast<int32>(delayR * factor);
	}
	m_delayTaps[12] = static_cast<int32>(10.0f / 1000.0f * sampleRate);
	// Early reflections (extra delay lines)
	m_delayTaps[13] = static_cast<int32>(3.25f / 1000.0f * sampleRate);
	m_delayTaps[14] = static_cast<int32>(3.53f / 1000.0f * sampleRate);

	for(std::size_t d = 0; d < mpt::size(m_delayTaps); d++)
		m_delayLines[d].SetDelayTap(m_delayTaps[d]);
}


void I3DL2Reverb::SetDecayCoeffs()
{
	float levelLtmp = 1.0f, levelRtmp = 1.0f;
	float levelL = 0.0f, levelR = 0.0f;

	levelLtmp *= CalcDecayCoeffs(5);
	levelRtmp *= CalcDecayCoeffs(11);
	levelL += levelLtmp * 0.0225f;
	levelR += levelRtmp * 0.0225f;

	levelLtmp *= CalcDecayCoeffs(4);
	levelRtmp *= CalcDecayCoeffs(10);
	levelL += levelLtmp * 0.04f;
	levelR += levelRtmp * 0.04f;
	
	if(m_quality & kMoreDelayLines)
	{
		levelLtmp *= CalcDecayCoeffs(3);
		levelRtmp *= CalcDecayCoeffs(9);
		levelL += levelLtmp * 0.1225f;
		levelR += levelRtmp * 0.1225f;

		levelLtmp *= CalcDecayCoeffs(2);
		levelRtmp *= CalcDecayCoeffs(8);
		levelL += levelLtmp * 0.1444f;
		levelR += levelRtmp * 0.1444f;
	}
	CalcDecayCoeffs(12);
	levelLtmp *= m_delayCoeffs[12][0] * m_delayCoeffs[12][0];
	levelRtmp *= m_delayCoeffs[12][0] * m_delayCoeffs[12][0];

	levelLtmp *= CalcDecayCoeffs(1);
	levelRtmp *= CalcDecayCoeffs(7);
	levelL += levelRtmp * 0.1444f;
	levelR += levelLtmp * 0.1444f;

	levelLtmp *= CalcDecayCoeffs(0);
	levelRtmp *= CalcDecayCoeffs(6);
	levelL += levelLtmp * 0.1444f;
	levelR += levelRtmp * 0.1444f;

	// Final Reverb Level
	float level = std::min(std::pow(10.0f, (Room() + Reverb()) / (100.0f * 20.0f)), 1.0f);
	float monoInv = 1.0f - ((levelLtmp + levelRtmp) * 0.5f);
	m_ReverbLevelL = level * std::sqrt(monoInv / levelL);
	m_ReverbLevelR = level * std::sqrt(monoInv / levelR);
}


float I3DL2Reverb::CalcDecayCoeffs(int32 index)
{
	float hfRef = static_cast<float>(2.0 * M_PI) / m_effectiveSampleRate * HFReference();
	float decayHFRatio = DecayHFRatio();
	if(decayHFRatio > 1.0f)
		hfRef = static_cast<float>(M_PI);

	float c1 = std::pow(10.0f, ((m_delayTaps[index] / m_effectiveSampleRate) * -60.0f / DecayTime()) / 20.0f);
	float c2 = 0.0f;

	float c21 = (std::pow(c1, 2.0f - 2.0f / decayHFRatio) - 1.0f) / (1.0f - std::cos(hfRef));
	if(c21 != 0)
	{
		float c22 = -2.0f * c21 - 2.0f;
		float c23 = std::sqrt(c22 * c22 - c21 * c21 * 4.0f);
		c2 = (c23 - c22) / (c21 + c21);
		if(mpt::abs(c2) > 1.0)
			c2 = (-c22 - c23) / (c21 + c21);
	}
	m_delayCoeffs[index][0] = c1;
	m_delayCoeffs[index][1] = c2;

	c1 *= c1;
	float diff2 = m_diffusion * m_diffusion;
	return diff2 + c1 / (1.0f - diff2 * c1) * (1.0f - diff2) * (1.0f - diff2);
}

} // namespace DMO

#else
MPT_MSVC_WORKAROUND_LNK4221(I3DL2Reverb)

#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_END
