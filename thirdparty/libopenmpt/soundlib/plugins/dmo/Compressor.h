/*
 * Compressor.h
 * -------------
 * Purpose: Implementation of the DMO Compressor DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#ifndef NO_PLUGINS

#include "../PlugInterface.h"

OPENMPT_NAMESPACE_BEGIN

namespace DMO
{

class Compressor : public IMixPlugin
{
protected:
	enum Parameters
	{
		kCompGain = 0,
		kCompAttack,
		kCompRelease,
		kCompThreshold,
		kCompRatio,
		kCompPredelay,
		kCompNumParameters
	};

	float m_param[kCompNumParameters];

	// Calculated parameters and coefficients
	float m_gain;
	float m_attack;
	float m_release;
	float m_threshold;
	float m_ratio;
	int32 m_predelay;

	// State
	std::vector<float> m_buffer;
	int32 m_bufPos, m_bufSize;
	float m_peak;

public:
	static IMixPlugin* Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);
	Compressor(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);

	void Release() override { delete this; }
	int32 GetUID() const override { return 0xEF011F79; }
	int32 GetVersion() const override { return 0; }
	void Idle() override { }
	uint32 GetLatency() const override { return 0; }

	void Process(float *pOutL, float *pOutR, uint32 numFrames) override;

	float RenderSilence(uint32) override { return 0.0f; }

	int32 GetNumPrograms() const override { return 0; }
	int32 GetCurrentProgram() override { return 0; }
	void SetCurrentProgram(int32) override { }

	PlugParamIndex GetNumParameters() const override { return kCompNumParameters; }
	PlugParamValue GetParameter(PlugParamIndex index) override;
	void SetParameter(PlugParamIndex index, PlugParamValue value) override;

	void Resume() override;
	void Suspend() override { m_isResumed = false; }
	void PositionChanged() override;
	bool IsInstrument() const override { return false; }
	bool CanRecieveMidiEvents() override { return false; }
	bool ShouldProcessSilence() override { return true; }

#ifdef MODPLUG_TRACKER
	CString GetDefaultEffectName() override { return _T("Compressor"); }

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
	float GainInDecibel() const { return -60.0f + m_param[kCompGain] * 120.0f; }
	float AttackTime() const { return 0.01f + m_param[kCompAttack] * 499.99f; }
	float ReleaseTime() const { return 50.0f + m_param[kCompRelease] * 2950.0f; }
	float ThresholdInDecibel() const { return -60.0f + m_param[kCompThreshold] * 60.0f; }
	float CompressorRatio() const { return 1.0f + m_param[kCompRatio] * 99.0f; }
	float PreDelay() const { return m_param[kCompPredelay] * 4.0f; }
	void RecalculateCompressorParams();
};

} // namespace DMO

OPENMPT_NAMESPACE_END

#endif // !NO_PLUGINS && NO_DMO
