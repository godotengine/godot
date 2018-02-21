/*
 * Distortion.h
 * ------------
 * Purpose: Implementation of the DMO Distortion DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#ifndef NO_PLUGINS

#include "../PlugInterface.h"

OPENMPT_NAMESPACE_BEGIN

namespace DMO
{

class Distortion : public IMixPlugin
{
protected:
	enum Parameters
	{
		kDistGain = 0,
		kDistEdge,
		kDistPreLowpassCutoff,
		kDistPostEQCenterFrequency,
		kDistPostEQBandwidth,
		kDistNumParameters
	};

	float m_param[kDistNumParameters];

	// Pre-EQ coefficients
	float m_preEQz1[2], m_preEQb1, m_preEQa0;
	// Post-EQ coefficients
	float m_postEQz1[2], m_postEQz2[2], m_postEQa0, m_postEQb0, m_postEQb1;
	uint8 m_edge, m_shift;

public:
	static IMixPlugin* Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);
	Distortion(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);

	void Release() override { delete this; }
	int32 GetUID() const override { return 0xEF114C90; }
	int32 GetVersion() const override { return 0; }
	void Idle() override { }
	uint32 GetLatency() const override { return 0; }

	void Process(float *pOutL, float *pOutR, uint32 numFrames) override;

	float RenderSilence(uint32) override { return 0.0f; }

	int32 GetNumPrograms() const override { return 0; }
	int32 GetCurrentProgram() override { return 0; }
	void SetCurrentProgram(int32) override { }

	PlugParamIndex GetNumParameters() const override { return kDistNumParameters; }
	PlugParamValue GetParameter(PlugParamIndex index) override;
	void SetParameter(PlugParamIndex index, PlugParamValue value) override;

	void Resume() override;
	void Suspend() override { m_isResumed = false; }
	void PositionChanged() override;
	bool IsInstrument() const override { return false; }
	bool CanRecieveMidiEvents() override { return false; }
	bool ShouldProcessSilence() override { return true; }

#ifdef MODPLUG_TRACKER
	CString GetDefaultEffectName() override { return _T("Distortion"); }

	CString GetParamName(PlugParamIndex param) override;
	CString GetParamLabel(PlugParamIndex) override;
	CString GetParamDisplay(PlugParamIndex param) override;

	CString GetCurrentProgramName() override { return CString(); }
	void SetCurrentProgramName(const CString &) override { }
	CString GetProgramName(int32) override { return CString(); }

	bool HasEditor() const override { return false; }
#endif

	int GetNumInputChannels() const override { return 2; }
	int GetNumOutputChannels() const override { return 2; }

protected:
	static float FreqInHertz(float param) { return 100.0f + param * 7900.0f; }
	float GainInDecibel() const { return -60.0f + m_param[kDistGain] * 60.0f; }
	void RecalculateDistortionParams();
};

} // namespace DMO

OPENMPT_NAMESPACE_END

#endif // !NO_PLUGINS
