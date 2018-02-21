/*
 * WavesReverb.h
 * -------------
 * Purpose: Implementation of the DMO WavesReverb DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#ifndef NO_PLUGINS

#include "../PlugInterface.h"

OPENMPT_NAMESPACE_BEGIN

namespace DMO
{

class WavesReverb : public IMixPlugin
{
protected:
	enum Parameters
	{
		kRvbInGain = 0,
		kRvbReverbMix,
		kRvbReverbTime,
		kRvbHighFreqRTRatio,
		kDistNumParameters
	};

	float m_param[kDistNumParameters];

	// Parameters and coefficients
	float m_dryFactor;
	float m_wetFactor;
	float m_coeffs[10];
	uint32 m_delay[6];

	// State
	struct ReverbState
	{
		uint32 combPos, allpassPos;
		float comb[4096][4];
		float allpass1[1024][2];
		float allpass2[1024][2];
	} m_state;

public:
	static IMixPlugin* Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);
	WavesReverb(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);

	void Release() override { delete this; }
	int32 GetUID() const override { return 0x87FC0268; }
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
	CString GetDefaultEffectName() override { return _T("WavesReverb"); }

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
	static float GainInDecibel(float param) { return -96.0f + param * 96.0f; }
	float ReverbTime() const { return 0.001f + m_param[kRvbReverbTime] * 2999.999f; }
	float HighFreqRTRatio() const { return 0.001f + m_param[kRvbHighFreqRTRatio] * 0.998f; }
	void RecalculateWavesReverbParams();
};

} // namespace DMO

OPENMPT_NAMESPACE_END

#endif // !NO_PLUGINS
