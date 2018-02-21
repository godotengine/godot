/*
 * Flanger.cpp
 * -----------
 * Purpose: Implementation of the DMO Flanger DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#ifndef NO_PLUGINS
#include "../../Sndfile.h"
#include "Flanger.h"
#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_PLUGINS

namespace DMO
{

IMixPlugin* Flanger::Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
{
	return new (std::nothrow) Flanger(factory, sndFile, mixStruct);
}


Flanger::Flanger(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
	: Chorus(factory, sndFile, mixStruct)
{
	m_param[kFlangerWetDryMix] = 0.5f;
	m_param[kFlangerWaveShape] = 1.0f;
	m_param[kFlangerFrequency] = 0.025f;
	m_param[kFlangerDepth] = 1.0f;
	m_param[kFlangerPhase] = 0.5f;
	m_param[kFlangerFeedback] = (-50.0f + 99.0f) / 198.0f;
	m_param[kFlangerDelay] = 0.5f;

	// Already done in Chorus constructor
	//m_mixBuffer.Initialize(2, 2);
	//InsertIntoFactoryList();
}


void Flanger::SetParameter(PlugParamIndex index, PlugParamValue value)
{
	if(index < kFlangerNumParameters)
	{
		Limit(value, 0.0f, 1.0f);
		if(index == kFlangerWaveShape && value < 1.0f)
			value = 0.0f;
		else if(index == kFlangerPhase)
			value = Util::Round(value * 4.0f) / 4.0f;
		m_param[index] = value;
		RecalculateChorusParams();
	}
}


#ifdef MODPLUG_TRACKER

CString Flanger::GetParamName(PlugParamIndex param)
{
	switch(param)
	{
	case kFlangerWetDryMix: return _T("WetDryMix");
	case kFlangerWaveShape: return _T("WaveShape");
	case kFlangerFrequency: return _T("Frequency");
	case kFlangerDepth: return _T("Depth");
	case kFlangerPhase: return _T("Phase");
	case kFlangerFeedback: return _T("Feedback");
	case kFlangerDelay: return _T("Delay");
	}
	return CString();
}


CString Flanger::GetParamLabel(PlugParamIndex param)
{
	switch(param)
	{
	case kFlangerWetDryMix:
	case kFlangerDepth:
	case kFlangerFeedback:
		return _T("%");
	case kFlangerFrequency:
		return _T("Hz");
	case kFlangerPhase:
		return _T("°");
	case kFlangerDelay:
		return _T("ms");
	}
	return CString();
}


CString Flanger::GetParamDisplay(PlugParamIndex param)
{
	CString s;
	float value = m_param[param];
	switch(param)
	{
	case kFlangerWetDryMix:
	case kFlangerDepth:
		value *= 100.0f;
		break;
	case kFlangerFrequency:
		value = FrequencyInHertz();
		break;
	case kFlangerWaveShape:
		return (value < 1) ? _T("Triangle") : _T("Sine");
		break;
	case kFlangerPhase:
		switch(Phase())
		{
		case 0: return _T("-180");
		case 1: return _T("-90");
		case 2: return _T("0");
		case 3: return _T("90");
		case 4: return _T("180");
		}
		break;
	case kFlangerFeedback:
		value = Feedback();
		break;
	case kFlangerDelay:
		value = Delay();
	}
	s.Format(_T("%.2f"), value);
	return s;
}

#endif // MODPLUG_TRACKER

} // namespace DMO

#else
MPT_MSVC_WORKAROUND_LNK4221(Flanger)

#endif // !NO_PLUGINS

OPENMPT_NAMESPACE_END
