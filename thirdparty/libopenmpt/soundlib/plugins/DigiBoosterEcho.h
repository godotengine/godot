/*
 * DigiBoosterEcho.h
 * -----------------
 * Purpose: Implementation of the DigiBooster Pro Echo DSP
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#ifndef NO_PLUGINS

#include "PlugInterface.h"

OPENMPT_NAMESPACE_BEGIN

class DigiBoosterEcho : public IMixPlugin
{
public:
	enum Parameters
	{
		kEchoDelay = 0,
		kEchoFeedback,
		kEchoMix,
		kEchoCross,
		kEchoNumParameters
	};

	// Our settings chunk for file I/O, as it will be written to files
	struct PluginChunk
	{
		char  id[4];
		uint8 param[kEchoNumParameters];

		PluginChunk(uint8 delay = 80, uint8 feedback = 150, uint8 mix = 80, uint8 cross = 255)
		{
			memcpy(id, "Echo", 4);
			param[kEchoDelay] = delay;
			param[kEchoFeedback] = feedback;
			param[kEchoMix] = mix;
			param[kEchoCross] = cross;

			STATIC_ASSERT(sizeof(PluginChunk) == 8);
		}
	};

protected:
	std::vector<float> m_delayLine;	// Echo delay line
	uint32 m_bufferSize;			// Delay line length in frames
	uint32 m_writePos;				// Current write position in the delay line
	uint32 m_delayTime;				// In frames
	uint32 m_sampleRate;

	// Echo calculation coefficients
	float m_PMix, m_NMix;
	float m_PCrossPBack, m_PCrossNBack;
	float m_NCrossPBack, m_NCrossNBack;

	// Settings chunk for file I/O
	PluginChunk m_chunk;

public:
	static IMixPlugin* Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);
	DigiBoosterEcho(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);

	void Release() override { delete this; }
	void SaveAllParameters() override;
	void RestoreAllParameters(int32 program) override;
	int32 GetUID() const override { int32le id; memcpy(&id, "Echo", 4); return id; }
	int32 GetVersion() const override { return 0; }
	void Idle() override { }
	uint32 GetLatency() const override { return 0; }

	void Process(float *pOutL, float *pOutR, uint32 numFrames) override;

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

	bool ProgramsAreChunks() const override { return true; }
	ChunkData GetChunk(bool) override;
	void SetChunk(const ChunkData &chunk, bool) override;

protected:
	void RecalculateEchoParams();
};

OPENMPT_NAMESPACE_END

#endif // NO_PLUGINS
