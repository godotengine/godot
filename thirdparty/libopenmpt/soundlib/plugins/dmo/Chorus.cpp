/*
 * Chorus.cpp
 * ----------
 * Purpose: Implementation of the DMO Chorus DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_PLUGINS
#include "../../Sndfile.h"
#include "Chorus.h"
#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_PLUGINS

namespace DMO
{

IMixPlugin* Chorus::Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
{
	return new (std::nothrow) Chorus(factory, sndFile, mixStruct);
}


Chorus::Chorus(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
	: IMixPlugin(factory, sndFile, mixStruct)
{
	m_param[kChorusWetDryMix] = 0.5f;
	m_param[kChorusDepth] = 0.1f;
	m_param[kChorusFrequency] = 0.11f;
	m_param[kChorusWaveShape] = 1.0f;
	m_param[kChorusPhase] = 0.75f;
	m_param[kChorusFeedback] = (25.0f + 99.0f) / 198.0f;
	m_param[kChorusDelay] = 0.8f;

	m_mixBuffer.Initialize(2, 2);
	InsertIntoFactoryList();
}


// Integer part of buffer position
int32 Chorus::GetBufferIntOffset(int32 fpOffset) const
{
	if(fpOffset < 0)
		fpOffset += m_bufSize * 4096;
	MPT_ASSERT(fpOffset >= 0);
	return (fpOffset / 4096) % m_bufSize;
}


void Chorus::Process(float *pOutL, float *pOutR, uint32 numFrames)
{
	if(!m_bufSize || !m_mixBuffer.Ok())
		return;

	const float *in[2] = { m_mixBuffer.GetInputBuffer(0), m_mixBuffer.GetInputBuffer(1) };
	float *out[2] = { m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1) };

	const bool isTriangle = IsTriangle();

	const float feedback = Feedback() / 100.0f;
	const float wetDryMix = WetDryMix();
	const uint32 phase = Phase();
	for(uint32 i = numFrames; i != 0; i--)
	{
		const float leftIn = *(in[0])++;
		const float rightIn = *(in[1])++;

		int32 readOffset = GetBufferIntOffset(m_bufPos + m_delayOffset);
		int32 writeOffset = GetBufferIntOffset(m_bufPos);
		m_buffer[writeOffset] = (m_buffer[readOffset] * feedback) + (rightIn + leftIn) * 0.5f;

		float waveMin;
		float waveMax;
		if(isTriangle)
		{
			m_waveShapeMin += m_waveShapeVal;
			m_waveShapeMax += m_waveShapeVal;
			if(m_waveShapeMin > 1)
				m_waveShapeMin -= 2;
			if(m_waveShapeMax > 1)
				m_waveShapeMax -= 2;
			waveMin = mpt::abs(m_waveShapeMin) * 2 - 1;
			waveMax = mpt::abs(m_waveShapeMax) * 2 - 1;
		} else
		{
			m_waveShapeMin = m_waveShapeMax * m_waveShapeVal + m_waveShapeMin;
			m_waveShapeMax = m_waveShapeMax - m_waveShapeMin * m_waveShapeVal;
			waveMin = m_waveShapeMin;
			waveMax = m_waveShapeMax;
		}

		float left1 = m_buffer[GetBufferIntOffset(m_bufPos + m_delayL1)];
		float left2 = m_buffer[GetBufferIntOffset(m_bufPos + m_delayL2)];
		float fracPos = (m_delayL1 & 0xFFF) * (1.0f / 4096.0f);
		float leftOut = (left2 - left1) * fracPos + left1;
		*(out[0])++ = leftIn + (leftOut - leftIn) * wetDryMix;

		float right1 = m_buffer[GetBufferIntOffset(m_bufPos + m_delayR1)];
		float right2 = m_buffer[GetBufferIntOffset(m_bufPos + m_delayR2)];
		fracPos = (m_delayR1 & 0xFFF) * (1.0f / 4096.0f);
		float rightOut = (right2 - right1) * fracPos + right1;
		*(out[1])++ = rightIn + (rightOut - rightIn) * wetDryMix;

		// Increment delay positions
		m_delayL1 = m_delayOffset + (phase < 4 ? 1 : -1) * static_cast<int32>(waveMin * m_depthDelay);
		m_delayL2 = m_delayL1 + 4096;

		m_delayR1 = m_delayOffset + (phase < 2 ? -1 : 1) * static_cast<int32>(((phase % 2u) ? waveMax : waveMin) * m_depthDelay);
		m_delayR2 = m_delayR1 + 4096;

		if(m_bufPos <= 0)
			m_bufPos += m_bufSize * 4096;
		m_bufPos -= 4096;
	}

	ProcessMixOps(pOutL, pOutR, m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1), numFrames);
}


PlugParamValue Chorus::GetParameter(PlugParamIndex index)
{
	if(index < kChorusNumParameters)
	{
		return m_param[index];
	}
	return 0;
}


void Chorus::SetParameter(PlugParamIndex index, PlugParamValue value)
{
	if(index < kChorusNumParameters)
	{
		Limit(value, 0.0f, 1.0f);
		if(index == kChorusWaveShape && value < 1.0f)
			value = 0.0f;
		else if(index == kChorusPhase)
			value = Util::Round(value * 4.0f) / 4.0f;
		m_param[index] = value;
		RecalculateChorusParams();
	}
}


void Chorus::Resume()
{
	PositionChanged();
	RecalculateChorusParams();

	m_isResumed = true;
	m_waveShapeMin = 0.0f;
	m_waveShapeMax = IsTriangle() ? 0.5f : 1.0f;
	m_delayL1 = m_delayL2 = m_delayR1 = m_delayR2 = m_delayOffset;
	m_bufPos = 0;
}


void Chorus::PositionChanged()
{
	m_bufSize = Util::muldiv(m_SndFile.GetSampleRate(), 3840, 1000);
	try
	{
		m_buffer.assign(m_bufSize, 0.0f);
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
		m_bufSize = 0;
	}
}


#ifdef MODPLUG_TRACKER

CString Chorus::GetParamName(PlugParamIndex param)
{
	switch(param)
	{
	case kChorusWetDryMix: return _T("WetDryMix");
	case kChorusDepth: return _T("Depth");
	case kChorusFrequency: return _T("Frequency");
	case kChorusWaveShape: return _T("WaveShape");
	case kChorusPhase: return _T("Phase");
	case kChorusFeedback: return _T("Feedback");
	case kChorusDelay: return _T("Delay");
	}
	return CString();
}


CString Chorus::GetParamLabel(PlugParamIndex param)
{
	switch(param)
	{
	case kChorusWetDryMix:
	case kChorusDepth:
	case kChorusFeedback:
		return _T("%");
	case kChorusFrequency:
		return _T("Hz");
	case kChorusPhase:
		return _T("°");
	case kChorusDelay:
		return _T("ms");
	}
	return CString();
}


CString Chorus::GetParamDisplay(PlugParamIndex param)
{
	CString s;
	float value = m_param[param];
	switch(param)
	{
	case kChorusWetDryMix:
	case kChorusDepth:
		value *= 100.0f;
		break;
	case kChorusFrequency:
		value = FrequencyInHertz();
		break;
	case kChorusWaveShape:
		return (value < 0.5f) ? _T("Triangle") : _T("Sine");
		break;
	case kChorusPhase:
		switch(Phase())
		{
		case 0: return _T("-180");
		case 1: return _T("-90");
		case 2: return _T("0");
		case 3: return _T("90");
		case 4: return _T("180");
		}
		break;
	case kChorusFeedback:
		value = Feedback();
		break;
	case kChorusDelay:
		value = Delay();
	}
	s.Format(_T("%.2f"), value);
	return s;
}

#endif // MODPLUG_TRACKER


void Chorus::RecalculateChorusParams()
{
	const float sampleRate = static_cast<float>(m_SndFile.GetSampleRate());

	float delaySamples = Delay() * sampleRate / 1000.0f;
	m_depthDelay = Depth() * delaySamples * 2048.0f;
	m_delayOffset = Util::Round<int32>(4096.0f * (delaySamples + 2.0f));
	m_frequency = FrequencyInHertz();
	const float frequencySamples = m_frequency / sampleRate;
	if(IsTriangle())
		m_waveShapeVal = frequencySamples * 2.0f;
	else
		m_waveShapeVal = std::sin(frequencySamples * float(M_PI)) * 2.0f;
}

} // namespace DMO

#else
MPT_MSVC_WORKAROUND_LNK4221(Chorus)

#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_END
