/*
 * Echo.h
 * ------
 * Purpose: Implementation of the DMO Echo DSP (for non-Windows platforms)
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#ifndef NO_PLUGINS

#include "../PlugInterface.h"

OPENMPT_NAMESPACE_BEGIN

namespace DMO
{

class Echo : public IMixPlugin
{
protected:
	enum Parameters
	{
		kEchoWetDry = 0,
		kEchoFeedback,
		kEchoLeftDelay,
		kEchoRightDelay,
		kEchoPanDelay,
		kEchoNumParameters
	};

	std::vector<float> m_delayLine;	// Echo delay line
	float m_param[kEchoNumParameters];
	uint32 m_bufferSize;			// Delay line length in frames
	uint32 m_writePos;				// Current write position in the delay line
	uint32 m_delayTime[2];			// In frames
	uint32 m_sampleRate;

	// Echo calculation coefficients
	float m_initialFeedback;
	bool m_crossEcho;

public:
	static IMixPlugin* Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);
	Echo(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);

	void Release() override { delete this; }
	int32 GetUID() const override { return 0xEF3E932C; }
	int32 GetVersion() const override { return 0; }
	void Idle() override { }
	uint32 GetLatency() const override { return 0; }

	void Process(float *pOutL, float *pOutR, uint32 numFrames)override;

	float RenderSilence(uint32) override { return 0.0f; }

	int32 GetNumPrograms() const override { return 0; }
	int32 GetCurrentProgram() override { return 0; }
	void SetCurrentProgram(int32) override { }

	PlugParamIndex GetNumParameters() const override { return kEchoNumParameters; }
	PlugParamValue GetParameter(PlugParamIndex index) override;
	void SetParameter(PlugParamIndex index, PlugParamValue value) override;

	void Resume() override;
	void Suspend() override { m_isResumed = false; }
	void PositionChanged() override;

	bool IsInstrument() const override { return false; }
	bool CanRecieveMidiEvents() override { return false; }
	bool ShouldProcessSilence() override { return true; }

#ifdef MODPLUG_TRACKER
	CString GetDefaultEffectName() override { return _T("Echo"); }

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
	void RecalculateEchoParams();
};

} // namespace DMO

OPENMPT_NAMESPACE_END

#endif // !NO_PLUGINS
