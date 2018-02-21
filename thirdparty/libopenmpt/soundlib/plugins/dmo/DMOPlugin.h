/*
 * DMOPlugin.h
 * -----------
 * Purpose: DirectX Media Object plugin handling / processing.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#ifndef NO_DMO

#include "../PlugInterface.h"
#include <dmoreg.h>
#include <strmif.h>

typedef interface IMediaObject IMediaObject;
typedef interface IMediaObjectInPlace IMediaObjectInPlace;
typedef interface IMediaParamInfo IMediaParamInfo;
typedef interface IMediaParams IMediaParams;

OPENMPT_NAMESPACE_BEGIN

class DMOPlugin : public IMixPlugin
{
protected:
	IMediaObject *m_pMediaObject;
	IMediaObjectInPlace *m_pMediaProcess;
	IMediaParamInfo *m_pParamInfo;
	IMediaParams *m_pMediaParams;

	uint32 m_nSamplesPerSec;
	uint32 m_uid;
	union
	{
		int16 *i16;
		float *f32;
	} m_alignedBuffer;
	union
	{
		int16 i16[MIXBUFFERSIZE * 2 + 16];		// 16-bit PCM Stereo interleaved
		float f32[MIXBUFFERSIZE * 2 + 16];		// 32-bit Float Stereo interleaved
	} m_interleavedBuffer;
	bool m_useFloat;

public:
	static IMixPlugin* Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct);

protected:
	DMOPlugin(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct, IMediaObject *pMO, IMediaObjectInPlace *pMOIP, uint32 uid);
	~DMOPlugin();

public:
	void Release() override { delete this; }
	int32 GetUID() const override { return m_uid; }
	int32 GetVersion() const override { return 2; }
	void Idle() override { }
	uint32 GetLatency() const override;

	void Process(float *pOutL, float *pOutR, uint32 numFrames) override;

	int32 GetNumPrograms() const override { return 0; }
	int32 GetCurrentProgram() override { return 0; }
	void SetCurrentProgram(int32 /*nIndex*/) override { }

	PlugParamIndex GetNumParameters() const override;
	PlugParamValue GetParameter(PlugParamIndex index) override;
	void SetParameter(PlugParamIndex index, PlugParamValue value) override;

	void Resume() override;
	void Suspend() override;
	void PositionChanged() override;

	bool IsInstrument() const  override { return false; }
	bool CanRecieveMidiEvents()  override { return false; }
	bool ShouldProcessSilence()  override { return true; }

#ifdef MODPLUG_TRACKER
	CString GetDefaultEffectName() override { return CString(); }

	CString GetParamName(PlugParamIndex param) override;
	CString GetParamLabel(PlugParamIndex param) override;
	CString GetParamDisplay(PlugParamIndex param) override;

	// TODO we could simply add our own preset mechanism. But is it really useful for these plugins?
	CString GetCurrentProgramName() override { return CString(); }
	void SetCurrentProgramName(const CString &) override { }
	CString GetProgramName(int32) override { return CString(); }

	bool HasEditor() const override { return false; }
#endif

	int GetNumInputChannels() const override { return 2; }
	int GetNumOutputChannels() const override { return 2; }
};

OPENMPT_NAMESPACE_END

#endif // NO_DMO

