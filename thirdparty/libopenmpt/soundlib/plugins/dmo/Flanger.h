/*
 * Flanger.h
 * ---------
 * Purpose: Implementation of the DMO Flanger DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#ifndef NO_PLUGINS

#include "Chorus.h"

OPENMPT_NAMESPACE_BEGIN

namespace DMO
{

class Flanger : public Chorus
{
protected:
	enum Parameters
	{
		kFlangerWetDryMix = 0,
		kFlangerWaveShape,
		kFlangerFrequency,
		kFlangerDepth,
		kFlangerPhase,
		kFlangerFeedback,
		kFlangerDelay,
		kFlangerNumParameters
	};

public:
	static IMixPlugin* Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);
	Flanger(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);

	void Release() override { delete this; }
	int32 GetUID() const override { return 0xEFCA3D92; }

	PlugParamIndex GetNumParameters() const override { return kFlangerNumParameters; }
	void SetParameter(PlugParamIndex index, PlugParamValue value) override;

#ifdef MODPLUG_TRACKER
	CString GetDefaultEffectName() override { return _T("Flanger"); }

	CString GetParamName(PlugParamIndex param) override;
	CString GetParamLabel(PlugParamIndex) override;
	CString GetParamDisplay(PlugParamIndex param) override;
#endif

protected:
	float WetDryMix() const override { return m_param[kFlangerWetDryMix]; }
	bool IsTriangle() const override { return m_param[kFlangerWaveShape] < 1; }
	float Depth() const override { return m_param[kFlangerDepth]; }
	float Feedback() const override { return -99.0f + m_param[kFlangerFeedback] * 198.0f; }
	float Delay() const override { return m_param[kFlangerDelay] * 4.0f; }
	float FrequencyInHertz() const override { return m_param[kFlangerFrequency] * 10.0f; }
	int Phase() const override { return Util::Round<uint32>(m_param[kFlangerPhase] * 4.0f); }
};

} // namespace DMO

OPENMPT_NAMESPACE_END

#endif // !NO_PLUGINS
