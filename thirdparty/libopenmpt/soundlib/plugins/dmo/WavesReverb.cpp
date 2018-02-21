/*
 * WavesReverb.cpp
 * ---------------
 * Purpose: Implementation of the DMO WavesReverb DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_PLUGINS
#include "../../Sndfile.h"
#include "WavesReverb.h"
#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_PLUGINS

namespace DMO
{

IMixPlugin* WavesReverb::Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
{
	return new (std::nothrow) WavesReverb(factory, sndFile, mixStruct);
}


WavesReverb::WavesReverb(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
	: IMixPlugin(factory, sndFile, mixStruct)
{
	m_param[kRvbInGain] = 1.0f;
	m_param[kRvbReverbMix] = 1.0f;
	m_param[kRvbReverbTime] = 1.0f / 3.0f;
	m_param[kRvbHighFreqRTRatio] = 0.0f;

	m_mixBuffer.Initialize(2, 2);
	InsertIntoFactoryList();
}


void WavesReverb::Process(float *pOutL, float *pOutR, uint32 numFrames)
{
	if(!m_mixBuffer.Ok())
		return;

	const float *in[2] = { m_mixBuffer.GetInputBuffer(0), m_mixBuffer.GetInputBuffer(1) };
	float *out[2] = { m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1) };

	uint32 combPos = m_state.combPos, allpassPos = m_state.allpassPos;
	uint32 delay0 = (m_delay[0] + combPos + 1) & 0xFFF;
	uint32 delay1 = (m_delay[1] + combPos + 1) & 0xFFF;
	uint32 delay2 = (m_delay[2] + combPos + 1) & 0xFFF;
	uint32 delay3 = (m_delay[3] + combPos + 1) & 0xFFF;
	uint32 delay4 = (m_delay[4] + allpassPos) & 0x3FF;
	uint32 delay5 = (m_delay[5] + allpassPos) & 0x3FF;
	float delay0old = m_state.comb[delay0][0];
	float delay1old = m_state.comb[delay1][1];
	float delay2old = m_state.comb[delay2][2];
	float delay3old = m_state.comb[delay3][3];

	for(uint32 i = numFrames; i != 0; i--)
	{
		const float leftIn  = *(in[0])++ + 1e-30f;	// Prevent denormals
		const float rightIn = *(in[1])++ + 1e-30f;	// Prevent denormals

		// Advance buffer index for the four comb filters
		delay0 = (delay0 - 1) & 0xFFF;
		delay1 = (delay1 - 1) & 0xFFF;
		delay2 = (delay2 - 1) & 0xFFF;
		delay3 = (delay3 - 1) & 0xFFF;
		float &delay0new = m_state.comb[delay0][0];
		float &delay1new = m_state.comb[delay1][1];
		float &delay2new = m_state.comb[delay2][2];
		float &delay3new = m_state.comb[delay3][3];

		float r1, r2;
		
		r1 = delay1new * 0.61803401f + m_state.allpass1[delay4][0] * m_coeffs[0];
		r2 = m_state.allpass1[delay4][1] * m_coeffs[0] - delay0new * 0.61803401f;
		m_state.allpass1[allpassPos][0] = r2 * 0.61803401f + delay0new;
		m_state.allpass1[allpassPos][1] = delay1new - r1 * 0.61803401f;
		delay0new = r1;
		delay1new = r2;

		r1 = delay3new * 0.61803401f + m_state.allpass2[delay5][0] * m_coeffs[1];
		r2 = m_state.allpass2[delay5][1] * m_coeffs[1] - delay2new * 0.61803401f;
		m_state.allpass2[allpassPos][0] = r2 * 0.61803401f + delay2new;
		m_state.allpass2[allpassPos][1] = delay3new - r1 * 0.61803401f;
		delay2new = r1;
		delay3new = r2;

		*(out[0])++ = (leftIn  * m_dryFactor) + delay0new + delay2new;
		*(out[1])++ = (rightIn * m_dryFactor) + delay1new + delay3new;

		const float leftWet  = leftIn  * m_wetFactor;
		const float rightWet = rightIn * m_wetFactor;
		m_state.comb[combPos][0] = (delay0new * m_coeffs[2]) + (delay0old * m_coeffs[3]) + leftWet;
		m_state.comb[combPos][1] = (delay1new * m_coeffs[4]) + (delay1old * m_coeffs[5]) + rightWet;
		m_state.comb[combPos][2] = (delay2new * m_coeffs[6]) + (delay2old * m_coeffs[7]) - rightWet;
		m_state.comb[combPos][3] = (delay3new * m_coeffs[8]) + (delay3old * m_coeffs[9]) + leftWet;

		delay0old = delay0new;
		delay1old = delay1new;
		delay2old = delay2new;
		delay3old = delay3new;

		// Advance buffer index
		combPos = (combPos - 1) & 0xFFF;
		allpassPos = (allpassPos - 1) & 0x3FF;
		delay4 = (delay4 - 1) & 0x3FF;
		delay5 = (delay5 - 1) & 0x3FF;
	}
	m_state.combPos = combPos;
	m_state.allpassPos = allpassPos;

	ProcessMixOps(pOutL, pOutR, m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1), numFrames);
}


PlugParamValue WavesReverb::GetParameter(PlugParamIndex index)
{
	if(index < kDistNumParameters)
	{
		return m_param[index];
	}
	return 0;
}


void WavesReverb::SetParameter(PlugParamIndex index, PlugParamValue value)
{
	if(index < kDistNumParameters)
	{
		Limit(value, 0.0f, 1.0f);
		m_param[index] = value;
		RecalculateWavesReverbParams();
	}
}


void WavesReverb::Resume()
{
	m_isResumed = true;
	// Recalculate delays
	uint32 delay0 = Util::Round<uint32>(m_SndFile.GetSampleRate() * 0.045f);
	uint32 delay1 = Util::Round<uint32>(delay0 * 1.18920707f);	// 2^0.25
	uint32 delay2 = Util::Round<uint32>(delay1 * 1.18920707f);
	uint32 delay3 = Util::Round<uint32>(delay2 * 1.18920707f);
	uint32 delay4 = Util::Round<uint32>((delay0 + delay2) * 0.11546667f);
	uint32 delay5 = Util::Round<uint32>((delay1 + delay3) * 0.11546667f);
	// Comb delays
	m_delay[0] = delay0 - delay4;
	m_delay[1] = delay2 - delay4;
	m_delay[2] = delay1 - delay5;
	m_delay[3] = delay3 - delay5;
	// Allpass delays
	m_delay[4] = delay4;
	m_delay[5] = delay5;

	RecalculateWavesReverbParams();
	PositionChanged();
}


void WavesReverb::PositionChanged()
{
	MemsetZero(m_state);
}


#ifdef MODPLUG_TRACKER

CString WavesReverb::GetParamName(PlugParamIndex param)
{
	switch(param)
	{
	case kRvbInGain: return _T("InGain");
	case kRvbReverbMix: return _T("ReverbMix");
	case kRvbReverbTime: return _T("ReverbTime");
	case kRvbHighFreqRTRatio: return _T("HighFreqRTRatio");
	}
	return CString();
}


CString WavesReverb::GetParamLabel(PlugParamIndex param)
{
	switch(param)
	{
	case kRvbInGain:
	case kRvbReverbMix:
		return _T("dB");
	case kRvbReverbTime:
		return _T("ms");
	}
	return CString();
}


CString WavesReverb::GetParamDisplay(PlugParamIndex param)
{
	float value = m_param[param];
	switch(param)
	{
	case kRvbInGain:
	case kRvbReverbMix:
		value = GainInDecibel(value);
		break;
	case kRvbReverbTime:
		value = ReverbTime();
		break;
	case kRvbHighFreqRTRatio:
		value = HighFreqRTRatio();
		break;
	}
	CString s;
	s.Format(_T("%.2f"), value);
	return s;
}

#endif // MODPLUG_TRACKER


void WavesReverb::RecalculateWavesReverbParams()
{
	// Recalculate filters
	const double ReverbTimeSmp = -3000.0 / (m_SndFile.GetSampleRate() * ReverbTime());
	const double ReverbTimeSmpHF = ReverbTimeSmp * (1.0 / HighFreqRTRatio() - 1.0);

	m_coeffs[0] = static_cast<float>(std::pow(10.0, m_delay[4] * ReverbTimeSmp));
	m_coeffs[1] = static_cast<float>(std::pow(10.0, m_delay[5] * ReverbTimeSmp));

	double sum = 0.0;
	for(uint32 pair = 0; pair < 4; pair++)
	{
		double gain1 = std::pow(10.0, m_delay[pair] * ReverbTimeSmp);
		double gain2 = (1.0 - std::pow(10.0, (m_delay[pair] + m_delay[4 + pair / 2]) * ReverbTimeSmpHF)) * 0.5;
		double gain3 = gain1 * m_coeffs[pair / 2];
		double gain4 = gain3 * (((gain3 + 1.0) * gain3 + 1.0) * gain3 + 1.0) + 1.0;
		m_coeffs[2 + pair * 2] = static_cast<float>(gain1 * (1.0 - gain2));
		m_coeffs[3 + pair * 2] = static_cast<float>(gain1 * gain2);
		sum += gain4 * gain4;
	}

	double inGain = std::pow(10.0, GainInDecibel(m_param[kRvbInGain]) * 0.05);
	double reverbMix = std::pow(10.0, GainInDecibel(m_param[kRvbReverbMix]) * 0.1);
	m_dryFactor = static_cast<float>(std::sqrt(1.0 - reverbMix) * inGain);
	m_wetFactor = static_cast<float>(std::sqrt(reverbMix) * (4.0 / std::sqrt(sum) * inGain));
}

} // namespace DMO

#else
MPT_MSVC_WORKAROUND_LNK4221(WavesReverb)

#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_END
