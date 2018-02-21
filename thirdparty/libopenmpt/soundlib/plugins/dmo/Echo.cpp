/*
 * Echo.cpp
 * --------
 * Purpose: Implementation of the DMO Echo DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_PLUGINS
#include "../../Sndfile.h"
#include "Echo.h"
#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_PLUGINS

namespace DMO
{

IMixPlugin* Echo::Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
{
	return new (std::nothrow) Echo(factory, sndFile, mixStruct);
}


Echo::Echo(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
	: IMixPlugin(factory, sndFile, mixStruct)
	, m_bufferSize(0)
	, m_writePos(0)
	, m_sampleRate(sndFile.GetSampleRate())
	, m_initialFeedback(0.0f)
{
	m_param[kEchoWetDry] = 0.5f;
	m_param[kEchoFeedback] = 0.5f;
	m_param[kEchoLeftDelay] = 0.25f;
	m_param[kEchoRightDelay] = 0.25f;
	m_param[kEchoPanDelay] = 0.0f;

	m_mixBuffer.Initialize(2, 2);
	InsertIntoFactoryList();
}


void Echo::Process(float *pOutL, float *pOutR, uint32 numFrames)
{
	if(!m_bufferSize || !m_mixBuffer.Ok())
		return;
	const float wetMix = m_param[kEchoWetDry], dryMix = 1 - wetMix;
	const float *in[2] = { m_mixBuffer.GetInputBuffer(0), m_mixBuffer.GetInputBuffer(1) };
	float *out[2] = { m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1) };

	for(uint32 i = numFrames; i != 0; i--)
	{
		for(uint8 channel = 0; channel < 2; channel++)
		{
			const uint8 readChannel = (m_crossEcho ? (1 - channel) : channel);
			int readPos = m_writePos - m_delayTime[readChannel];
			if(readPos < 0)
				readPos += m_bufferSize;

			float chnInput = *(in[channel])++;
			float chnDelay = m_delayLine[readPos * 2 + readChannel];

			// Calculate the delay
			float chnOutput = chnInput * m_initialFeedback;
			chnOutput += chnDelay * m_param[kEchoFeedback];

			// Prevent denormals
			if(mpt::abs(chnOutput) < 1e-24f)
				chnOutput = 0.0f;

			m_delayLine[m_writePos * 2 + channel] = chnOutput;
			// Output samples now
			*(out[channel])++ = (chnInput * dryMix + chnDelay * wetMix);
		}
		m_writePos++;
		if(m_writePos == m_bufferSize)
			m_writePos = 0;
	}

	ProcessMixOps(pOutL, pOutR, m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1), numFrames);
}


PlugParamValue Echo::GetParameter(PlugParamIndex index)
{
	if(index < kEchoNumParameters)
	{
		return m_param[index];
	}
	return 0;
}


void Echo::SetParameter(PlugParamIndex index, PlugParamValue value)
{
	if(index < kEchoNumParameters)
	{
		Limit(value, 0.0f, 1.0f);
		if(index == kEchoPanDelay)
			value = Util::Round(value);
		m_param[index] = value;
		RecalculateEchoParams();
	}
}


void Echo::Resume()
{
	m_isResumed = true;
	m_sampleRate = m_SndFile.GetSampleRate();
	RecalculateEchoParams();
	PositionChanged();
}


void Echo::PositionChanged()
{
	m_bufferSize = m_sampleRate * 2u;
	try
	{
		m_delayLine.assign(m_bufferSize * 2, 0);
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
		m_bufferSize = 0;
	}
	m_writePos = 0;
}


#ifdef MODPLUG_TRACKER

CString Echo::GetParamName(PlugParamIndex param)
{
	switch(param)
	{
	case kEchoWetDry: return _T("WetDryMix");
	case kEchoFeedback: return _T("Feedback");
	case kEchoLeftDelay: return _T("LeftDelay");
	case kEchoRightDelay: return _T("RightDelay");
	case kEchoPanDelay: return _T("PanDelay");
	}
	return CString();
}


CString Echo::GetParamLabel(PlugParamIndex param)
{
	if(param == kEchoLeftDelay || param == kEchoRightDelay)
		return _T("ms");
	return CString();
}


CString Echo::GetParamDisplay(PlugParamIndex param)
{
	CString s;
	switch(param)
	{
	case kEchoWetDry:
	case kEchoFeedback:
		s.Format(_T("%.2f"), m_param[param] * 100.0f);
		break;
	case kEchoLeftDelay:
	case kEchoRightDelay:
		s.Format(_T("%.2f"), m_param[param] * 2000.0f);
		break;
	case kEchoPanDelay:
		s = (m_param[param] <= 0.5) ? _T("No") : _T("Yes");
	}
	return s;
}

#endif // MODPLUG_TRACKER


void Echo::RecalculateEchoParams()
{
	m_initialFeedback = std::sqrt(1.0f - (m_param[kEchoFeedback] * m_param[kEchoFeedback]));
	m_delayTime[0] = static_cast<uint32>(m_param[kEchoLeftDelay] * (2 * m_sampleRate));
	m_delayTime[1] = static_cast<uint32>(m_param[kEchoRightDelay] * (2 * m_sampleRate));
	m_crossEcho = (m_param[kEchoPanDelay]) > 0.5f;
}

} // namespace DMO

#else
MPT_MSVC_WORKAROUND_LNK4221(Echo)

#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_END
