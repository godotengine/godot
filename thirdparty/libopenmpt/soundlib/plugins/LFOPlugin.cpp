/*
* LFOPlugin.cpp
* -------------
* Purpose: Plugin for automating other plugins' parameters
* Notes  : (currently none)
* Authors: OpenMPT Devs
* The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
*/


#include "stdafx.h"

#ifndef NO_PLUGINS
#include "LFOPlugin.h"
#include "../Sndfile.h"
#include "../../common/FileReader.h"
#ifdef MODPLUG_TRACKER
#include "../../mptrack/plugins/LFOPluginEditor.h"
#endif // MODPLUG_TRACKER

OPENMPT_NAMESPACE_BEGIN

IMixPlugin* LFOPlugin::Create(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
{
	return new (std::nothrow) LFOPlugin(factory, sndFile, mixStruct);
}


LFOPlugin::LFOPlugin(VSTPluginLib &factory, CSoundFile &sndFile, SNDMIXPLUGIN *mixStruct)
	: IMixPlugin(factory, sndFile, mixStruct)
	, m_nextRandom(0)
	, m_tempo(0)
	, m_PRNG(mpt::make_prng<mpt::fast_prng>(mpt::global_prng()))
{
	m_amplitude = 0.5f;
	m_offset = 0.5f;
	m_frequency = 0.290241f;
	m_tempoSync = false;
	m_waveForm = kSine;
	m_polarity = false;
	m_bypassed = false;
	m_outputToCC = false;
	m_outputParam = int32_max;
	m_oneshot = false;
	RecalculateFrequency();
	RecalculateIncrement();

	m_mixBuffer.Initialize(2, 2);
	InsertIntoFactoryList();
}


// Processing (we do not process audio, just send out parameters)
void LFOPlugin::Process(float *pOutL, float *pOutR, uint32 numFrames)
{
	if(!m_bypassed)
	{
		ResetSilence();
		if(m_tempoSync)
		{
			double tempo = m_SndFile.GetCurrentBPM();
			if(tempo != m_tempo)
			{
				m_tempo = tempo;
				RecalculateIncrement();
			}
		}

		if(m_oneshot)
		{
			LimitMax(m_phase, 1.0);
		} else
		{
			int intPhase = static_cast<int>(m_phase);
			if(intPhase > 0 && (m_waveForm == kSHNoise || m_waveForm == kSmoothNoise))
			{
				// Phase wrap-around happened
				NextRandom();
			}
			m_phase -= intPhase;
		}

		double value = 0;
		switch(m_waveForm)
		{
		case kSine:
			value = std::sin(m_phase * 2.0 * M_PI);
			break;
		case kTriangle:
			value = 1.0 - 4.0 * mpt::abs(m_phase - 0.5);
			break;
		case kSaw:
			value = 2.0 * m_phase - 1.0;
			break;
		case kSquare:
			value = m_phase < 0.5 ? -1.0 : 1.0;
			break;
		case kSHNoise:
			value = m_random;
			break;
		case kSmoothNoise:
			value = m_phase * m_phase * m_phase * (m_phase * (m_phase * 6 - 15) + 10);	// Smootherstep
			value = m_nextRandom * value + m_random * (1.0 - value);
			break;
		default:
			break;
		}
		if(m_polarity)
			value = -value;
		// Transform value from -1...+1 to 0...1 range and apply offset/amplitude
		value = value * m_amplitude + m_offset;
		Limit(value, 0.0, 1.0);

		IMixPlugin *plugin = GetOutputPlugin();
		if(plugin != nullptr)
		{
			if(m_outputToCC)
			{
				plugin->MidiSend(MIDIEvents::CC(static_cast<MIDIEvents::MidiCC>(m_outputParam & 0x7F), static_cast<uint8>((m_outputParam >> 8) & 0x0F), Util::Round<uint8>(value * 127.0f)));
			} else
			{
				plugin->SetParameter(m_outputParam, static_cast<PlugParamValue>(value));
			}
		}

		m_phase += m_increment * numFrames;
	}

	ProcessMixOps(pOutL, pOutR, m_mixBuffer.GetInputBuffer(0), m_mixBuffer.GetInputBuffer(1), numFrames);
}


PlugParamValue LFOPlugin::GetParameter(PlugParamIndex index)
{
	switch(index)
	{
	case kAmplitude: return m_amplitude;
	case kOffset: return m_offset;
	case kFrequency: return m_frequency;
	case kTempoSync: return m_tempoSync ? 1.0f : 0.0f;
	case kWaveform: return WaveformToParam(m_waveForm);
	case kPolarity: return m_polarity ? 1.0f : 0.0f;
	case kBypassed: return m_bypassed ? 1.0f : 0.0f;
	case kLoopMode: return m_oneshot ? 1.0f : 0.0f;
	default: return 0;
	}
}


void LFOPlugin::SetParameter(PlugParamIndex index, PlugParamValue value)
{
	ResetSilence();
	Limit(value, 0.0f, 1.0f);
	switch(index)
	{
	case kAmplitude: m_amplitude = value; break;
	case kOffset: m_offset = value; break;
	case kFrequency:
		m_frequency = value;
		RecalculateFrequency();
		break;
	case kTempoSync:
		m_tempoSync = (value >= 0.5f);
		RecalculateFrequency();
		break;
	case kWaveform:
		m_waveForm = ParamToWaveform(value);
		if(m_waveForm >= kNumWaveforms)
			m_waveForm = static_cast<LFOWaveform>(kNumWaveforms - 1);
		break;
	case kPolarity: m_polarity = (value >= 0.5f); break;
	case kBypassed: m_bypassed = (value >= 0.5f); break;
	case kLoopMode: m_oneshot = (value >= 0.5f); break;
	case kCurrentPhase:
		if(value == 0)
		{
			// Enforce next random value for random LFOs
			NextRandom();
		}
		m_phase = value;
		return;

	default: return;
	}

#ifdef MODPLUG_TRACKER
	if(GetEditor() != nullptr)
	{
		GetEditor()->PostMessage(WM_PARAM_UDPATE, GetSlot(), index);
	}
#endif
}


void LFOPlugin::Resume()
{
	m_isResumed = true;
	RecalculateIncrement();
	NextRandom();
	PositionChanged();
}


void LFOPlugin::PositionChanged()
{
	// TODO Changing tempo (with tempo sync enabled), parameter automation over time and setting the LFO phase manually is not considered here.
	m_phase = m_increment * m_SndFile.GetTotalSampleCount();
	m_phase -= static_cast<int64>(m_phase);
}


bool LFOPlugin::MidiSend(uint32 midiCode)
{
	if(IMixPlugin *plugin = GetOutputPlugin())
		return plugin->MidiSend(midiCode);
	else
		return true;
}


bool LFOPlugin::MidiSysexSend(const void *message, uint32 length)
{
	if(IMixPlugin *plugin = GetOutputPlugin())
		return plugin->MidiSysexSend(message, length);
	else
		return true;
}


void LFOPlugin::MidiCC(uint8 nMidiCh, MIDIEvents::MidiCC nController, uint8 nParam, CHANNELINDEX trackChannel)
{
	if(IMixPlugin *plugin = GetOutputPlugin())
	{
		plugin->MidiCC(nMidiCh, nController, nParam, trackChannel);
	}
}


void LFOPlugin::MidiPitchBend(uint8 nMidiCh, int32 increment, int8 pwd)
{
	if(IMixPlugin *plugin = GetOutputPlugin())
	{
		plugin->MidiPitchBend(nMidiCh, increment, pwd);
	}
}


void LFOPlugin::MidiVibrato(uint8 nMidiCh, int32 depth, int8 pwd)
{
	if(IMixPlugin *plugin = GetOutputPlugin())
	{
		plugin->MidiVibrato(nMidiCh, depth, pwd);
	}
}


void LFOPlugin::MidiCommand(uint8 nMidiCh, uint8 nMidiProg, uint16 wMidiBank, uint16 note, uint16 vol, CHANNELINDEX trackChannel)
{
	if(ModCommand::IsNote(static_cast<ModCommand::NOTE>(note)) && vol > 0)
	{
		SetParameter(kCurrentPhase, 0);
	}
	if(IMixPlugin *plugin = GetOutputPlugin())
	{
		plugin->MidiCommand(nMidiCh, nMidiProg, wMidiBank, note, vol, trackChannel);
	}
}


void LFOPlugin::HardAllNotesOff()
{
	if(IMixPlugin *plugin = GetOutputPlugin())
	{
		plugin->HardAllNotesOff();
	}
}


bool LFOPlugin::IsNotePlaying(uint32 note, uint32 midiChn, uint32 trackerChn)
{
	if(IMixPlugin *plugin = GetOutputPlugin())
		return plugin->IsNotePlaying(note, midiChn, trackerChn);
	else
		return false;
}


void LFOPlugin::SaveAllParameters()
{
	auto chunk = GetChunk(false);
	if(chunk.empty())
		return;

	m_pMixStruct->defaultProgram = -1;
	m_pMixStruct->pluginData.assign(chunk.cbegin(), chunk.cend());
}


void LFOPlugin::RestoreAllParameters(int32 /*program*/)
{
	SetChunk(mpt::as_span(m_pMixStruct->pluginData), false);
}


struct PluginData
{
	char     magic[4];
	uint32le version;
	uint32le amplitude;	// float
	uint32le offset;	// float
	uint32le frequency;	// float
	uint32le waveForm;
	uint32le outputParam;
	uint8le  tempoSync;
	uint8le  polarity;
	uint8le  bypassed;
	uint8le  outputToCC;
	uint8le  loopMode;
};

MPT_BINARY_STRUCT(PluginData, 33)


IMixPlugin::ChunkData LFOPlugin::GetChunk(bool)
{
	PluginData chunk;
	memcpy(chunk.magic, "LFO ", 4);
	chunk.version = 0;
	chunk.amplitude = IEEE754binary32LE(m_amplitude).GetInt32();
	chunk.offset = IEEE754binary32LE(m_offset).GetInt32();
	chunk.frequency = IEEE754binary32LE(m_frequency).GetInt32();
	chunk.waveForm = m_waveForm;
	chunk.outputParam = m_outputParam;
	chunk.tempoSync = m_tempoSync ? 1 : 0;
	chunk.polarity = m_polarity ? 1 : 0;
	chunk.bypassed = m_bypassed ? 1 : 0;
	chunk.outputToCC = m_outputToCC ? 1 : 0;
	chunk.loopMode = m_oneshot ? 1 : 0;

	m_chunkData.resize(sizeof(chunk));
	memcpy(m_chunkData.data(), &chunk, sizeof(chunk));
	return mpt::as_span(m_chunkData);
}


void LFOPlugin::SetChunk(const ChunkData &chunk, bool)
{
	FileReader file(chunk);
	PluginData data;
	if(file.ReadStructPartial(data, file.BytesLeft())
		&& !memcmp(data.magic, "LFO ", 4)
		&& data.version == 0)
	{
		m_amplitude = Clamp<float>(IEEE754binary32LE().SetInt32(data.amplitude), 0.0f, 1.0f);
		m_offset = Clamp<float>(IEEE754binary32LE().SetInt32(data.offset), 0.0f, 1.0f);
		m_frequency = Clamp<float>(IEEE754binary32LE().SetInt32(data.frequency), 0.0f, 1.0f);
		if(data.waveForm < kNumWaveforms)
			m_waveForm = static_cast<LFOWaveform>(data.waveForm.get());
		m_outputParam = data.outputParam;
		m_tempoSync = data.tempoSync != 0;
		m_polarity = data.polarity != 0;
		m_bypassed = data.bypassed != 0;
		m_outputToCC = data.outputToCC != 0;
		m_oneshot = data.loopMode != 0;
	}
}


#ifdef MODPLUG_TRACKER

CString LFOPlugin::GetParamName(PlugParamIndex param)
{
	switch(param)
	{
	case kAmplitude: return _T("Amplitude");
	case kOffset: return _T("Offset");
	case kFrequency: return _T("Frequency");
	case kTempoSync: return _T("Tempo Sync");
	case kWaveform: return _T("Waveform");
	case kPolarity: return _T("Polarity");
	case kBypassed: return _T("Bypassed");
	case kLoopMode: return _T("Loop Mode");
	case kCurrentPhase: return _T("Set LFO Phase");
	}
	return CString();
}


CString LFOPlugin::GetParamLabel(PlugParamIndex param)
{
	if(param == kFrequency)
	{
		if(m_tempoSync && m_computedFrequency > 0.0 && m_computedFrequency < 1.0)
			return _T("Beats Per Cycle");
		else if(m_tempoSync)
			return _T("Cycles Per Beat");
		else
			return _T("Hz");
	}
	return CString();
}


CString LFOPlugin::GetParamDisplay(PlugParamIndex param)
{
	CString s;
	if(param == kPolarity)
	{
		return m_polarity ? _T("Inverted") : _T("Normal");
	} else if(param == kTempoSync)
	{
		return m_tempoSync ? _T("Yes") : _T("No");
	} else if(param == kBypassed)
	{
		return m_bypassed ? _T("Yes") : _T("No");
	} else if(param == kWaveform)
	{
		static const TCHAR *waveforms[] = { _T("Sine"), _T("Triangle"), _T("Saw"), _T("Square"), _T("Noise"), _T("Smoothed Noise") };
		if(m_waveForm < MPT_ARRAY_COUNT(waveforms))
			return waveforms[m_waveForm];
	} else if(param == kLoopMode)
	{
		return m_oneshot ? _T("One-Shot") : _T("Looped");
	} else if(param == kCurrentPhase)
	{
		return _T("Write-Only");
	} else if(param < kLFONumParameters)
	{
		auto val = GetParameter(param);
		if(param == kOffset)
			val = 2.0f * val - 1.0f;
		if(param == kFrequency)
		{
			val = static_cast<PlugParamValue>(m_computedFrequency);
			if(m_tempoSync && val > 0.0f && val < 1.0f)
				val = static_cast<PlugParamValue>(1.0 / m_computedFrequency);
		}
		s.Format(_T("%.3f"), val);
	}
	return s;
}


CAbstractVstEditor *LFOPlugin::OpenEditor()
{
	try
	{
		return new LFOPluginEditor(*this);
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
		return nullptr;
	}
}

#endif // MODPLUG_TRACKER


void LFOPlugin::NextRandom()
{
	m_random = m_nextRandom;
	m_nextRandom = mpt::random<int32>(m_PRNG) / static_cast<float>(int32_min);
}


void LFOPlugin::RecalculateFrequency()
{
	m_computedFrequency = 0.25 * std::pow(2.0, m_frequency * 8.0) - 0.25;
	if(m_tempoSync)
	{
		if(m_computedFrequency > 0.00045)
		{
			double freqLog = std::log(m_computedFrequency) / M_LN2;
			double freqFrac = freqLog - std::floor(freqLog);
			freqLog -= freqFrac;

			// Lock to powers of two and 1.5 times or 1.333333... times the powers of two
			if(freqFrac < 0.20751874963942190927313052802609)
				freqFrac = 0.0;
			else if(freqFrac < 0.5)
				freqFrac = 0.41503749927884381854626105605218;
			else if(freqFrac < 0.79248125036057809072686947197391)
				freqFrac = 0.58496250072115618145373894394782;
			else
				freqFrac = 1.0;

			m_computedFrequency = std::pow(2.0, freqLog + freqFrac) * 0.5;
		} else
		{
			m_computedFrequency = 0;
		}
	}
	RecalculateIncrement();
}


void LFOPlugin::RecalculateIncrement()
{
	m_increment = m_computedFrequency / m_SndFile.GetSampleRate();
	if(m_tempoSync)
	{
		m_increment *= m_tempo / 60.0;
	}
}


IMixPlugin *LFOPlugin::GetOutputPlugin() const
{
	PLUGINDEX outPlug = m_pMixStruct->GetOutputPlugin();
	if(outPlug > m_nSlot && outPlug < MAX_MIXPLUGINS)
		return m_SndFile.m_MixPlugins[outPlug].pMixPlugin;
	else
		return nullptr;
}


OPENMPT_NAMESPACE_END

#else
MPT_MSVC_WORKAROUND_LNK4221(LFOPlugin)

#endif // !NO_PLUGINS
