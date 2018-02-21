/*
 * Gargle.cpp
 * ----------
 * Purpose: Implementation of the DMO Gargle DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_PLUGINS
#include "../../Sndfile.h"
#include "Gargle.h"
#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_PLUGINS

namespace DMO
{

IMixPlugin* Gargle::Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
{
	return new (std::nothrow) Gargle(factory, sndFile, mixStruct);
}


Gargle::Gargle(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
	: IMixPlugin(factory, sndFile, mixStruct)
{
	m_param[kGargleRate] = 0.02f;
	m_param[kGargleWaveShape] = 0.0f;

	m_mixBuffer.Initialize(2, 2);
	InsertIntoFactoryList();
}


void Gargle::Process(float *pOutL, float *pOutR, uint32 numFrames)
{
	if(!m_mixBuffer.Ok())
		return;

	const float *inL = m_mixBuffer.GetInputBuffer(0), *inR = m_mixBuffer.GetInputBuffer(1);
	float *outL = m_mixBuffer.GetOutputBuffer(0), *outR = m_mixBuffer.GetOutputBuffer(1);
	const bool triangle = m_param[kGargleWaveShape] < 1.0f;

	for(uint32 frame = numFrames; frame != 0;)
	{
		if(m_counter < m_periodHalf)
		{
			// First half of gargle period
			const uint32 remain = std::min(frame, m_periodHalf - m_counter);
			if(triangle)
			{
				const uint32 stop = m_counter + remain;
				const float factor = 1.0f / m_periodHalf;
				for(uint32 i = m_counter; i < stop; i++)
				{
					*outL++ = *inL++ * i * factor;
					*outR++ = *inR++ * i * factor;
				}
			} else
			{
				for(uint32 i = 0; i < remain; i++)
				{
					*outL++ = *inL++;
					*outR++ = *inR++;
				}
			}
			frame -= remain;
			m_counter += remain;
		} else
		{
			// Second half of gargle period
			const uint32 remain = std::min(frame, m_period - m_counter);
			if(triangle)
			{
				const uint32 stop = m_period - m_counter - remain;
				const float factor = 1.0f / m_periodHalf;
				for(uint32 i = m_period - m_counter; i > stop; i--)
				{
					*outL++ = *inL++ * i * factor;
					*outR++ = *inR++ * i * factor;
				}
			} else
			{
				for(uint32 i = 0; i < remain; i++)
				{
					*outL++ = 0;
					*outR++ = 0;
				}
				inL += remain;
				inR += remain;

			}
			frame -= remain;
			m_counter += remain;
			if(m_counter >= m_period) m_counter = 0;
		}
	}

	ProcessMixOps(pOutL, pOutR, m_mixBuffer.GetOutputBuffer(0), m_mixBuffer.GetOutputBuffer(1), numFrames);
}


PlugParamValue Gargle::GetParameter(PlugParamIndex index)
{
	if(index < kEqNumParameters)
	{
		return m_param[index];
	}
	return 0;
}


void Gargle::SetParameter(PlugParamIndex index, PlugParamValue value)
{
	if(index < kEqNumParameters)
	{
		Limit(value, 0.0f, 1.0f);
		if(index == kGargleWaveShape)
			value = Util::Round(value);
		m_param[index] = value;
		RecalculateGargleParams();
	}
}


void Gargle::Resume()
{
	RecalculateGargleParams();
	m_counter = 0;
	m_isResumed = true;
}


#ifdef MODPLUG_TRACKER

CString Gargle::GetParamName(PlugParamIndex param)
{
	switch(param)
	{
	case kGargleRate: return _T("Rate");
	case kGargleWaveShape: return _T("WaveShape");
	}
	return CString();
}


CString Gargle::GetParamLabel(PlugParamIndex param)
{
	switch(param)
	{
	case kGargleRate: return _T("Hz");
	}
	return CString();
}


CString Gargle::GetParamDisplay(PlugParamIndex param)
{
	CString s;
	switch(param)
	{
	case kGargleRate:
		s.Format(_T("%d"), RateInHertz());
		break;
	case kGargleWaveShape:
		return (m_param[param] < 0.5) ? _T("Triangle") : _T("Square");
	}
	return s;
}

#endif // MODPLUG_TRACKER


uint32 Gargle::RateInHertz() const
{
	return Util::Round<uint32>(m_param[kGargleRate] * 999.0f) + 1;
}


void Gargle::RecalculateGargleParams()
{
	m_period = m_SndFile.GetSampleRate() / RateInHertz();
	if(m_period < 2) m_period = 2;
	m_periodHalf = m_period / 2;
	LimitMax(m_counter, m_period);
}

} // namespace DMO

#else
MPT_MSVC_WORKAROUND_LNK4221(Gargle)

#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_END
