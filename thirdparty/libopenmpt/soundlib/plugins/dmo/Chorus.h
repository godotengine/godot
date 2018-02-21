/*
 * Chorus.h
 * --------
 * Purpose: Implementation of the DMO Chorus DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#ifndef NO_PLUGINS

#include "../PlugInterface.h"

OPENMPT_NAMESPACE_BEGIN

namespace DMO
{

class Chorus : public IMixPlugin
{
protected:
	enum Parameters
	{
		kChorusWetDryMix = 0,
		kChorusDepth,
		kChorusFrequency,
		kChorusWaveShape,
		kChorusPhase,
		kChorusFeedback,
		kChorusDelay,
		kChorusNumParameters
	};

	float m_param[kChorusNumParameters];

	// Calculated parameters
	float m_waveShapeMin, m_waveShapeMax, m_waveShapeVal;
	float m_depthDelay;
	float m_frequency;
	int32 m_delayOffset;

	// State
	std::vector<float> m_buffer;
	int32 m_bufPos, m_bufSize;

	int32 m_delayL1, m_delayL2, m_delayR1, m_delayR2;

public:
	static IMixPlugin* Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);
	Chorus(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);

	void Release() override { delete this; }
	int32 GetUID() const override { return 0xEFE6629C; }
	int32 GetVersion() const override { return 0; }
	void Idle() override { }
	uint32 GetLatency() const override { return 0; }

	void Process(float *pOutL, float *pOutR, uint32 numFrames) override;

	float RenderSilence(uint32) override { return 0.0f; }

	int32 GetNumPrograms() const override { return 0; }
	int32 GetCurrentProgram() override { return 0; }
	void SetCurrentProgram(int32) override { }

	PlugParamIndex GetNumParameters() const override { return kChorusNumParameters; }
	PlugParamValue GetParameter(PlugParamIndex index) override;
	void SetParameter(PlugParamIndex index, PlugParamValue value) override;

	void Resume() override;
	void Suspend() override { m_isResumed = false; }
	void PositionChanged() override;
	bool IsInstrument() const override { return false; }
	bool CanRecieveMidiEvents() override { return false; }
	bool ShouldProcessSilence() override { return true; }

#ifdef MODPLUG_TRACKER
	CString GetDefaultEffectName() override { return _T("Chorus"); }

	CString GetParamName(PlugParamIndex param) override;
	CString GetParamLabel(PlugParamIndex) override;
	CString GetParamDisplay(PlugParamIndex param) override;

	CString GetCurrentProgramName() override { return CString(); }
	void SetCurrentProgramName(const CString &) override { }
	CString GetProgramName(int32) override { return CString(); }

	bool HasEditor() const override { return false; }
#endif

	void BeginSetProgram(int32) override { }
	void EndSetProgram() override { }

	int GetNumInputChannels() const override { return 2; }
	int GetNumOutputChannels() const override { return 2; }

protected:
	int32 GetBufferIntOffset(int32 fpOffset) const;

	virtual float WetDryMix() const { return m_param[kChorusWetDryMix]; }
	virtual bool IsTriangle() const { return m_param[kChorusWaveShape] < 1; }
	virtual float Depth() const { return m_param[kChorusDepth]; }
	virtual float Feedback() const { return -99.0f + m_param[kChorusFeedback] * 198.0f; }
	virtual float Delay() const { return m_param[kChorusDelay] * 20.0f; }
	virtual float FrequencyInHertz() const { return m_param[kChorusFrequency] * 10.0f; }
	virtual int Phase() const { return Util::Round<uint32>(m_param[kChorusPhase] * 4.0f); }
	void RecalculateChorusParams();
};

} // namespace DMO

OPENMPT_NAMESPACE_END

#endif // !NO_PLUGINS
