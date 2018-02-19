/*
 * Snd_Defs.h
 * ----------
 * Purpose: Basic definitions of data types, enums, etc. for the playback engine core.
 * Notes  : (currently none)
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "../common/typedefs.h"
#include "../common/FlagSet.h"


OPENMPT_NAMESPACE_BEGIN


typedef uint32 ROWINDEX;
	const ROWINDEX ROWINDEX_INVALID = uint32_max;
typedef uint16 CHANNELINDEX;
	const CHANNELINDEX CHANNELINDEX_INVALID = uint16_max;
typedef uint16 ORDERINDEX;
	const ORDERINDEX ORDERINDEX_INVALID = uint16_max;
	const ORDERINDEX ORDERINDEX_MAX = uint16_max - 1;
typedef uint16 PATTERNINDEX;
	const PATTERNINDEX PATTERNINDEX_INVALID = uint16_max;
typedef uint8  PLUGINDEX;
	const PLUGINDEX PLUGINDEX_INVALID = uint8_max;
typedef uint16 SAMPLEINDEX;
	const SAMPLEINDEX SAMPLEINDEX_INVALID = uint16_max;
typedef uint16 INSTRUMENTINDEX;
	const INSTRUMENTINDEX INSTRUMENTINDEX_INVALID = uint16_max;
typedef uint8 SEQUENCEINDEX;
	const SEQUENCEINDEX SEQUENCEINDEX_INVALID = uint8_max;

typedef uint32 SmpLength;


const SmpLength MAX_SAMPLE_LENGTH	= 0x10000000;	// Sample length in *frames*
													// Note: Sample size in bytes can be more than this (= 256 MB).

const ROWINDEX MAX_PATTERN_ROWS			= 1024;
const ORDERINDEX MAX_ORDERS				= ORDERINDEX_MAX + 1;
const PATTERNINDEX MAX_PATTERNS			= 4000;
const SAMPLEINDEX MAX_SAMPLES			= 4000;
const INSTRUMENTINDEX MAX_INSTRUMENTS	= 256;
const PLUGINDEX MAX_MIXPLUGINS			= 250;

const SEQUENCEINDEX MAX_SEQUENCES		= 50;

const CHANNELINDEX MAX_BASECHANNELS		= 127;	// Maximum pattern channels.
const CHANNELINDEX MAX_CHANNELS			= 256;	// Maximum number of mixing channels.

#define FREQ_FRACBITS		4		// Number of fractional bits in return value of CSoundFile::GetFreqFromPeriod()

// String lengths (including trailing null char)
#define MAX_SAMPLENAME			32
#define MAX_SAMPLEFILENAME		22
#define MAX_INSTRUMENTNAME		32
#define MAX_INSTRUMENTFILENAME	32
#define MAX_PATTERNNAME			32
#define MAX_CHANNELNAME			20

enum MODTYPE
{
	MOD_TYPE_NONE	= 0x00,
	MOD_TYPE_MOD	= 0x01,
	MOD_TYPE_S3M	= 0x02,
	MOD_TYPE_XM		= 0x04,
	MOD_TYPE_MED	= 0x08,
	MOD_TYPE_MTM	= 0x10,
	MOD_TYPE_IT		= 0x20,
	MOD_TYPE_669	= 0x40,
	MOD_TYPE_ULT	= 0x80,
	MOD_TYPE_STM	= 0x100,
	MOD_TYPE_FAR	= 0x200,
	MOD_TYPE_DTM	= 0x400,
	MOD_TYPE_AMF	= 0x800,
	MOD_TYPE_AMS	= 0x1000,
	MOD_TYPE_DSM	= 0x2000,
	MOD_TYPE_MDL	= 0x4000,
	MOD_TYPE_OKT	= 0x8000,
	MOD_TYPE_MID	= 0x10000,
	MOD_TYPE_DMF	= 0x20000,
	MOD_TYPE_PTM	= 0x40000,
	MOD_TYPE_DBM	= 0x80000,
	MOD_TYPE_MT2	= 0x100000,
	MOD_TYPE_AMF0	= 0x200000,
	MOD_TYPE_PSM	= 0x400000,
	MOD_TYPE_J2B	= 0x800000,
	MOD_TYPE_MPT	= 0x1000000,
	MOD_TYPE_IMF	= 0x2000000,
	MOD_TYPE_AMS2	= 0x4000000,
	MOD_TYPE_DIGI	= 0x8000000,
	MOD_TYPE_STP	= 0x10000000,
	MOD_TYPE_PLM	= 0x20000000,
	MOD_TYPE_SFX	= 0x40000000,
};
DECLARE_FLAGSET(MODTYPE)


enum MODCONTAINERTYPE
{
	MOD_CONTAINERTYPE_NONE = 0x0,
	MOD_CONTAINERTYPE_MO3  = 0x1,
	MOD_CONTAINERTYPE_GDM  = 0x2,
	MOD_CONTAINERTYPE_UMX  = 0x3,
	MOD_CONTAINERTYPE_XPK  = 0x4,
	MOD_CONTAINERTYPE_PP20 = 0x5,
	MOD_CONTAINERTYPE_MMCMP= 0x6,
	MOD_CONTAINERTYPE_WAV  = 0x7, // WAV as module
	MOD_CONTAINERTYPE_UAX  = 0x8, // Unreal sample set as module
};


// Module channel / sample flags
enum ChannelFlags
{
	// Sample Flags
	CHN_16BIT			= 0x01,			// 16-bit sample
	CHN_LOOP			= 0x02,			// looped sample
	CHN_PINGPONGLOOP	= 0x04,			// bidi-looped sample
	CHN_SUSTAINLOOP		= 0x08,			// sample with sustain loop
	CHN_PINGPONGSUSTAIN	= 0x10,			// sample with bidi sustain loop
	CHN_PANNING			= 0x20,			// sample with forced panning
	CHN_STEREO			= 0x40,			// stereo sample
	CHN_REVERSE			= 0x80,			// start sample playback from sample / loop end (Velvet Studio feature) - this is intentionally the same flag as CHN_PINGPONGFLAG.
	// Channel Flags
	CHN_PINGPONGFLAG	= 0x80,			// when flag is on, sample is processed backwards
	CHN_MUTE			= 0x100,		// muted channel
	CHN_KEYOFF			= 0x200,		// exit sustain
	CHN_NOTEFADE		= 0x400,		// fade note (instrument mode)
	CHN_SURROUND		= 0x800,		// use surround channel
	CHN_WRAPPED_LOOP	= 0x1000,		// loop just wrapped around to loop start (required for correct interpolation around loop points)
	CHN_AMIGAFILTER		= 0x2000,		// Apply Amiga low-pass filter
	CHN_FILTER			= 0x4000,		// Apply resonant filter on sample
	CHN_VOLUMERAMP		= 0x8000,		// Apply volume ramping
	CHN_VIBRATO			= 0x10000,		// Apply vibrato
	CHN_TREMOLO			= 0x20000,		// Apply tremolo
	//CHN_PANBRELLO		= 0x40000,		// Apply panbrello
	CHN_PORTAMENTO		= 0x80000,		// Apply portamento
	CHN_GLISSANDO		= 0x100000,		// Glissando (force portamento to semitones) mode
	CHN_FASTVOLRAMP		= 0x200000,		// Force usage of global ramping settings instead of ramping over the complete render buffer length
	CHN_EXTRALOUD		= 0x400000,		// Force sample to play at 0dB
	CHN_REVERB			= 0x800000,		// Apply reverb on this channel
	CHN_NOREVERB		= 0x1000000,	// Disable reverb on this channel
	CHN_SOLO			= 0x2000000,	// solo channel -> CODE#0012 -> DESC="midi keyboard split" -! NEW_FEATURE#0012
	CHN_NOFX			= 0x4000000,	// dry channel -> CODE#0015 -> DESC="channels management dlg" -! NEW_FEATURE#0015
	CHN_SYNCMUTE		= 0x8000000,	// keep sample sync on mute

	// Sample flags (only present in ModSample::uFlags, may overlap with CHN_CHANNELFLAGS)
	SMP_MODIFIED		= 0x1000,	// Sample data has been edited in the tracker
	SMP_KEEPONDISK		= 0x2000,	// Sample is not saved to file, data is restored from original sample file
	SMP_NODEFAULTVOLUME	= 0x4000,	// Ignore default volume setting
};
DECLARE_FLAGSET(ChannelFlags)

#define CHN_SAMPLEFLAGS (CHN_16BIT | CHN_LOOP | CHN_PINGPONGLOOP | CHN_SUSTAINLOOP | CHN_PINGPONGSUSTAIN | CHN_PANNING | CHN_STEREO | CHN_PINGPONGFLAG | CHN_REVERSE)
#define CHN_CHANNELFLAGS (~CHN_SAMPLEFLAGS)

// Sample flags fit into the first 16 bits, and with the current memory layout, storing them as a 16-bit integer packs struct ModSample nicely.
typedef FlagSet<ChannelFlags, uint16> SampleFlags;


// Instrument envelope-specific flags
enum EnvelopeFlags
{
	ENV_ENABLED		= 0x01,	// env is enabled
	ENV_LOOP		= 0x02,	// env loop
	ENV_SUSTAIN		= 0x04,	// env sustain
	ENV_CARRY		= 0x08,	// env carry
	ENV_FILTER		= 0x10,	// filter env enabled (this has to be combined with ENV_ENABLED in the pitch envelope's flags)
};
DECLARE_FLAGSET(EnvelopeFlags)


// Envelope value boundaries
#define ENVELOPE_MIN		0		// Vertical min value of a point
#define ENVELOPE_MID		32		// Vertical middle line
#define ENVELOPE_MAX		64		// Vertical max value of a point
#define MAX_ENVPOINTS		240		// Maximum length of each instrument envelope


// Instrument-specific flags
enum InstrumentFlags
{
	INS_SETPANNING	= 0x01,	// Panning enabled
	INS_MUTE		= 0x02,	// Instrument is muted
};
DECLARE_FLAGSET(InstrumentFlags)


// envelope types in instrument editor
enum EnvelopeType
{
	ENV_VOLUME = 0,
	ENV_PANNING,
	ENV_PITCH,

	ENV_MAXTYPES
};

// Filter Modes
#define FLTMODE_UNCHANGED		0xFF
#define FLTMODE_LOWPASS			0
#define FLTMODE_HIGHPASS		1


// NNA types (New Note Action)
#define NNA_NOTECUT		0
#define NNA_CONTINUE	1
#define NNA_NOTEOFF		2
#define NNA_NOTEFADE	3

// DCT types (Duplicate Check Types)
#define DCT_NONE		0
#define DCT_NOTE		1
#define DCT_SAMPLE		2
#define DCT_INSTRUMENT	3
#define DCT_PLUGIN		4

// DNA types (Duplicate Note Action)
#define DNA_NOTECUT		0
#define DNA_NOTEOFF		1
#define DNA_NOTEFADE	2


// Module flags - contains both song configuration and playback state... Use SONG_FILE_FLAGS and SONG_PLAY_FLAGS distinguish between the two.
enum SongFlags
{
	//SONG_EMBEDMIDICFG	= 0x0001,		// Embed macros in file
	SONG_FASTVOLSLIDES	= 0x0002,		// Old Scream Tracker 3.0 volume slides
	SONG_ITOLDEFFECTS	= 0x0004,		// Old Impulse Tracker effect implementations
	SONG_ITCOMPATGXX	= 0x0008,		// IT "Compatible Gxx" (IT's flag to behave more like other trackers w/r/t portamento effects)
	SONG_LINEARSLIDES	= 0x0010,		// Linear slides vs. Amiga slides
	SONG_PATTERNLOOP	= 0x0020,		// Loop current pattern (pattern editor)
	SONG_STEP			= 0x0040,		// Song is in "step" mode (pattern editor)
	SONG_PAUSED			= 0x0080,		// Song is paused (no tick processing, just rendering audio)
	SONG_FADINGSONG		= 0x0100,		// Song is fading out
	SONG_ENDREACHED		= 0x0200,		// Song is finished
	//SONG_GLOBALFADE	= 0x0400,		// Song is fading out
	//SONG_CPUVERYHIGH	= 0x0800,		// High CPU usage
	SONG_FIRSTTICK		= 0x1000,		// Is set when the current tick is the first tick of the row
	SONG_MPTFILTERMODE	= 0x2000,		// Local filter mode (reset filter on each note)
	SONG_SURROUNDPAN	= 0x4000,		// Pan in the rear channels
	SONG_EXFILTERRANGE	= 0x8000,		// Cutoff Filter has double frequency range (up to ~10Khz)
	SONG_AMIGALIMITS	= 0x10000,		// Enforce amiga frequency limits
	SONG_S3MOLDVIBRATO	= 0x20000,		// ScreamTracker 2 vibrato in S3M files
	//SONG_ITPEMBEDIH	= 0x40000,		// Embed instrument headers in project file
	SONG_BREAKTOROW		= 0x80000,		// Break to row command encountered (internal flag, do not touch)
	SONG_POSJUMP		= 0x100000,		// Position jump encountered (internal flag, do not touch)
	SONG_PT_MODE		= 0x200000,		// ProTracker 1/2 playback mode
	SONG_PLAYALLSONGS	= 0x400000,		// Play all subsongs consecutively (libopenmpt)
	SONG_ISAMIGA		= 0x800000,		// Is an Amiga module and thus qualifies to be played using the Paula BLEP resampler
};
DECLARE_FLAGSET(SongFlags)

#define SONG_FILE_FLAGS (SONG_FASTVOLSLIDES|SONG_ITOLDEFFECTS|SONG_ITCOMPATGXX|SONG_LINEARSLIDES|SONG_EXFILTERRANGE|SONG_AMIGALIMITS|SONG_S3MOLDVIBRATO|SONG_PT_MODE|SONG_ISAMIGA)
#define SONG_PLAY_FLAGS (~SONG_FILE_FLAGS)

// Global Options (Renderer)
#ifndef NO_AGC
#define SNDDSP_AGC            0x40	// automatic gain control
#endif // ~NO_AGC
#ifndef NO_DSP
#define SNDDSP_MEGABASS       0x02	// bass expansion
#define SNDDSP_SURROUND       0x08	// surround mix
#endif // NO_DSP
#ifndef NO_REVERB
#define SNDDSP_REVERB         0x20	// apply reverb
#endif // NO_REVERB
#ifndef NO_EQ
#define SNDDSP_EQ             0x80	// apply EQ
#endif // NO_EQ

#define SNDMIX_SOFTPANNING    0x10	// soft panning mode (this is forced with mixmode RC3 and later)

// Misc Flags (can safely be turned on or off)
#define SNDMIX_MAXDEFAULTPAN	0x80000		// Used by the MOD loader (currently unused)
#define SNDMIX_MUTECHNMODE		0x100000	// Notes are not played on muted channels


#define MAX_GLOBAL_VOLUME 256u

// Resampling modes
enum ResamplingMode
{
	// ATTENTION: Do not change ANY of these values, as they get written out to files in per instrument interpolation settings
	// and old files have these exact values in them which should not change meaning.
	SRCMODE_NEAREST   = 0,
	SRCMODE_LINEAR    = 1,
	SRCMODE_SPLINE    = 2,
	SRCMODE_POLYPHASE = 3,
	SRCMODE_FIRFILTER = 4,
	SRCMODE_DEFAULT   = 5,

	SRCMODE_AMIGA = 0xFF,	// Not explicitely user-selectable
};

static inline bool IsKnownResamplingMode(int mode)
{
	return (mode >= 0) && (mode < SRCMODE_DEFAULT);
}


// Release node defines
#define ENV_RELEASE_NODE_UNSET	0xFF
#define NOT_YET_RELEASED		(-1)
STATIC_ASSERT(ENV_RELEASE_NODE_UNSET > MAX_ENVPOINTS);


enum PluginPriority
{
	ChannelOnly,
	InstrumentOnly,
	PrioritiseInstrument,
	PrioritiseChannel,
};

enum PluginMutePriority
{
	EvenIfMuted,
	RespectMutes,
};

//Plugin velocity handling options
enum PLUGVELOCITYHANDLING
{
	PLUGIN_VELOCITYHANDLING_CHANNEL = 0,
	PLUGIN_VELOCITYHANDLING_VOLUME
};

//Plugin volumecommand handling options
enum PLUGVOLUMEHANDLING
{
	PLUGIN_VOLUMEHANDLING_MIDI = 0,
	PLUGIN_VOLUMEHANDLING_DRYWET,
	PLUGIN_VOLUMEHANDLING_IGNORE,
	PLUGIN_VOLUMEHANDLING_CUSTOM,
	PLUGIN_VOLUMEHANDLING_MAX,
};

enum MidiChannel
{
	MidiNoChannel		= 0,
	MidiFirstChannel	= 1,
	MidiLastChannel		= 16,
	MidiMappedChannel	= 17,
};


// Vibrato Types
enum VibratoType
{
	VIB_SINE = 0,
	VIB_SQUARE,
	VIB_RAMP_UP,
	VIB_RAMP_DOWN,
	VIB_RANDOM
};


// Tracker-specific playback behaviour
// Note: The index of every flag has to be fixed, so do not remove flags. Always add new flags at the end!
enum PlayBehaviour
{
	MSF_COMPATIBLE_PLAY,			// No-op - only used during loading (Old general compatibility flag for IT/MPT/XM)
	kMPTOldSwingBehaviour,			// MPT 1.16 swing behaviour (IT/MPT, deprecated)
	kMIDICCBugEmulation,			// Emulate broken volume MIDI CC behaviour (IT/MPT/XM, deprecated)
	kOldMIDIPitchBends,				// Old VST MIDI pitch bend behaviour (IT/MPT/XM, deprecated)
	kFT2VolumeRamping,				// Smooth volume ramping like in FT2 (XM)
	kMODVBlankTiming,				// F21 and above set speed instead of tempo
	kSlidesAtSpeed1,				// Execute normal slides at speed 1 as if they were fine slides
	kHertzInLinearMode,				// Compute note frequency in hertz rather than periods
	kTempoClamp,					// Clamp tempo to 32-255 range.
	kPerChannelGlobalVolSlide,		// Global volume slide memory is per-channel
	kPanOverride,					// Panning commands override surround and random pan variation

	kITInstrWithoutNote,			// Avoid instrument handling if there is no note
	kITVolColFinePortamento,		// Volume column portamento never does fine portamento
	kITArpeggio,					// IT arpeggio algorithm
	kITOutOfRangeDelay,				// Out-of-range delay command behaviour in IT
	kITPortaMemoryShare,			// Gxx shares memory with Exx and Fxx
	kITPatternLoopTargetReset,		// After finishing a pattern loop, set the pattern loop target to the next row
	kITFT2PatternLoop,				// Nested pattern loop behaviour
	kITPingPongNoReset,				// Don't reset ping pong direction with instrument numbers
	kITEnvelopeReset,				// IT envelope reset behaviour
	kITClearOldNoteAfterCut,		// Forget the previous note after cutting it
	kITVibratoTremoloPanbrello,		// More IT-like Hxx / hx, Rxx, Yxx and autovibrato handling, including more precise LUTs
	kITTremor,						// Ixx behaves like in IT
	kITRetrigger,					// Qxx behaves like in IT
	kITMultiSampleBehaviour,		// Properly update C-5 frequency when changing in multisampled instrument
	kITPortaTargetReached,			// Clear portamento target after it has been reached
	kITPatternLoopBreak,			// Don't reset loop count on pattern break.
	kITOffset,						// IT-style Oxx edge case handling
	kITSwingBehaviour,				// IT's swing behaviour
	kITNNAReset,					// NNA is reset on every note change, not every instrument change
	kITSCxStopsSample,				// SCx really stops the sample and does not just mute it
	kITEnvelopePositionHandling,	// IT-style envelope position advance + enable/disable behaviour
	kITPortamentoInstrument,		// No sample changes during portamento with Compatible Gxx enabled, instrument envelope reset with portamento
	kITPingPongMode,				// Don't repeat last sample point in ping pong loop, like IT's software mixer
	kITRealNoteMapping,				// Use triggered note rather than translated note for PPS and other effects
	kITHighOffsetNoRetrig,			// SAx should not apply an offset effect to a note next to it
	kITFilterBehaviour,				// User IT's filter coefficients (unless extended filter range is used)
	kITNoSurroundPan,				// Panning and surround are mutually exclusive
	kITShortSampleRetrig,			// Don't retrigger already stopped channels
	kITPortaNoNote,					// Don't apply any portamento if no previous note is playing
	kITDontResetNoteOffOnPorta,		// Only reset note-off status on portamento in IT Compatible Gxx mode
	kITVolColMemory,				// IT volume column effects share their memory with the effect column
	kITPortamentoSwapResetsPos,		// Portamento with sample swap plays the new sample from the beginning
	kITEmptyNoteMapSlot,			// IT ignores instrument note map entries with no note completely
	kITFirstTickHandling,			// IT-style first tick handling
	kITSampleAndHoldPanbrello,		// IT-style sample&hold panbrello waveform
	kITClearPortaTarget,			// New notes reset portamento target in IT
	kITPanbrelloHold,				// Don't reset panbrello effect until next note or panning effect
	kITPanningReset,				// Sample and instrument panning is only applied on note change, not instrument change
	kITPatternLoopWithJumps,		// Bxx on the same row as SBx terminates the loop in IT
	kITInstrWithNoteOff,			// Instrument number with note-off recalls default volume

	kFT2Arpeggio,					// FT2 arpeggio algorithm
	kFT2Retrigger,					// Rxx behaves like in FT2
	kFT2VolColVibrato,				// Vibrato depth in volume column does not actually execute the vibrato effect
	kFT2PortaNoNote,				// Don't play portamento-ed note if no previous note is playing
	kFT2KeyOff,						// FT2-style Kxx handling
	kFT2PanSlide,					// Volume-column pan slides should be handled like fine slides
	kFT2OffsetOutOfRange,			// FT2-style 9xx edge case handling
	kFT2RestrictXCommand,			// Don't allow MPT extensions to Xxx command in XM
	kFT2RetrigWithNoteDelay,		// Retrigger envelopes if there is a note delay with no note
	kFT2SetPanEnvPos,				// Lxx only sets the pan env position if the volume envelope's sustain flag is set
	kFT2PortaIgnoreInstr,			// Portamento plus instrument number applies the volume settings of the new sample, but not the new sample itself.
	kFT2VolColMemory,				// No volume column memory in FT2
	kFT2LoopE60Restart,				// Next pattern starts on the same row as the last E60 command
	kFT2ProcessSilentChannels,		// Keep processing silent channels for later 3xx pickup
	kFT2ReloadSampleSettings,		// Reload sample settings even if a note-off is placed next to an instrument number
	kFT2PortaDelay,					// Portamento with note delay next to it is ignored in FT2
	kFT2Transpose,					// Out-of-range transposed notes in FT2
	kFT2PatternLoopWithJumps,		// Bxx or Dxx on the same row as E6x terminates the loop in FT2
	kFT2PortaTargetNoReset,			// Portamento target is not reset with new notes in FT2
	kFT2EnvelopeEscape,				// FT2 sustain point at end of envelope
	kFT2Tremor,						// Txx behaves like in FT2
	kFT2OutOfRangeDelay,			// Out-of-range delay command behaviour in FT2
	kFT2Periods,					// Use FT2's broken period handling
	kFT2PanWithDelayedNoteOff,		// Pan command with delayed note-off
	kFT2VolColDelay,				// FT2-style volume column handling if there is a note delay
	kFT2FinetunePrecision,			// Only take the upper 4 bits of sample finetune.

	kST3NoMutedChannels,			// Don't process any effects on muted S3M channels
	kST3EffectMemory,				// Most effects share the same memory in ST3
	kST3PortaSampleChange,			// Portamento plus instrument number applies the volume settings of the new sample, but not the new sample itself.
	kST3VibratoMemory,				// Do not remember vibrato type in effect memory
	kST3LimitPeriod,				// Cut note instead of limiting  final period (ModPlug Tracker style)
	KST3PortaAfterArpeggio,			// Portamento after arpeggio continues at the note where the arpeggio left off

	kMODOneShotLoops,				// Allow ProTracker-like oneshot loops
	kMODIgnorePanning,				// Do not process any panning commands
	kMODSampleSwap,					// On-the-fly sample swapping

	kFT2NoteOffFlags,				// Set and reset the correct fade/key-off flags with note-off and instrument number after note-off
	kITMultiSampleInstrumentNumber,	// After portamento to different sample within multi-sampled instrument, lone instrument numbers in patterns always recall the new sample's default settings
	kRowDelayWithNoteDelay,			// Retrigger note delays on every reptition of a row
	kFT2TremoloRampWaveform,		// FT2-compatible tremolo ramp down / triangle waveform
	kFT2PortaUpDownMemory,			// Portamento up and down have separate memory

	kMODOutOfRangeNoteDelay,		// ProTracker behaviour for out-of-range note delays
	kMODTempoOnSecondTick,			// ProTracker sets tempo after the first tick

	// Add new play behaviours here.

	kMaxPlayBehaviours,
};


// Tempo swing determines how much every row in modern tempo mode contributes to a beat.
class TempoSwing : public std::vector<uint32>
{
public:
	enum { Unity = 1u << 24 };
	// Normalize the tempo swing coefficients so that they add up to exactly the specified tempo again
	void Normalize();
	void resize(size_type newSize, value_type val = Unity) { std::vector<uint32>::resize(newSize, val); Normalize(); }

	static void Serialize(std::ostream &oStrm, const TempoSwing &swing);
	static void Deserialize(std::istream &iStrm, TempoSwing &swing, const size_t);
};


// Sample position and sample position increment value
struct SamplePosition
{
	typedef int64 value_t;
	typedef uint64 unsigned_value_t;

protected:
	value_t v;

public:
	static const uint32 fractMax = 0xFFFFFFFFu;
	static MPT_FORCEINLINE uint32 GetFractMax() { return fractMax; }

	SamplePosition() : v(0) { }
	explicit SamplePosition(value_t pos) : v(pos) { }
	SamplePosition(int32 intPart, uint32 fractPart) : v((static_cast<value_t>(intPart) * (1ll<<32)) | fractPart) { }
	static SamplePosition Ratio(uint32 dividend, uint32 divisor) { return SamplePosition((static_cast<int64>(dividend) << 32) / divisor); }
	static SamplePosition FromDouble(double pos) { return SamplePosition(static_cast<value_t>(pos * 4294967296.0)); }

	// Set integer and fractional part
	MPT_FORCEINLINE SamplePosition &Set(int32 intPart, uint32 fractPart = 0) { v = (static_cast<int64>(intPart) << 32) | fractPart; return *this; }
	// Set integer part, keep fractional part
	MPT_FORCEINLINE SamplePosition &SetInt(int32 intPart) { v = (static_cast<value_t>(intPart) << 32) | GetFract(); return *this; }
	// Get integer part (as sample length / position)
	MPT_FORCEINLINE SmpLength GetUInt() const { return static_cast<SmpLength>(static_cast<unsigned_value_t>(v) >> 32); }
	// Get integer part
	MPT_FORCEINLINE int32 GetInt() const { return static_cast<int32>(static_cast<unsigned_value_t>(v) >> 32); }
	// Get fractional part
	MPT_FORCEINLINE uint32 GetFract() const { return static_cast<uint32>(v); }
	// Get the inverted fractional part
	MPT_FORCEINLINE SamplePosition GetInvertedFract() const { return SamplePosition(0x100000000ll - GetFract()); }
	// Get the raw fixed-point value
	MPT_FORCEINLINE int64 GetRaw() const { return v; }
	// Negate the current value
	MPT_FORCEINLINE SamplePosition &Negate() { v = -v; return *this; }
	// Multiply and divide by given integer scalars
	MPT_FORCEINLINE SamplePosition &MulDiv(uint32 mul, uint32 div) { v = (v * mul) / div; return *this; }
	// Removes the integer part, only keeping fractions
	MPT_FORCEINLINE SamplePosition &RemoveInt() { v &= fractMax; return *this; }
	// Check if value is 1.0
	MPT_FORCEINLINE bool IsUnity() const { return v == 0x100000000ll; }
	// Check if value is 0
	MPT_FORCEINLINE bool IsZero() const { return v == 0; }
	// Check if value is > 0
	MPT_FORCEINLINE bool IsPositive() const { return v > 0; }
	// Check if value is < 0
	MPT_FORCEINLINE bool IsNegative() const { return v < 0; }

	// Addition / subtraction of another fixed-point number
	SamplePosition operator+ (const SamplePosition &other) const { return SamplePosition(v + other.v); }
	SamplePosition operator- (const SamplePosition &other) const { return SamplePosition(v - other.v); }
	void operator+= (const SamplePosition &other) { v += other.v; }
	void operator-= (const SamplePosition &other) { v -= other.v; }

	// Multiplication with integer scalar
	template<typename T>
	SamplePosition operator* (T other) const { return SamplePosition(static_cast<value_t>(v * other)); }
	template<typename T>
	void operator*= (T other) { v = static_cast<value_t>(v *other); }

	// Division by other fractional point number; returns scalar
	value_t operator/ (SamplePosition other) const { return v / other.v; }
	// Division by scalar; returns fractional point number
	SamplePosition operator/ (int div) const { return SamplePosition(v / div); }

	bool operator== (const SamplePosition &other) const { return v == other.v; }
	bool operator!= (const SamplePosition &other) const { return v != other.v; }
	bool operator<= (const SamplePosition &other) const { return v <= other.v; }
	bool operator>= (const SamplePosition &other) const { return v >= other.v; }
	bool operator< (const SamplePosition &other) const { return v < other.v; }
	bool operator> (const SamplePosition &other) const { return v > other.v; }
};


// Aaaand another fixed-point type, e.g. used for fractional tempos
// Note that this doesn't use classical bit shifting for the fixed point part.
// This is mostly for the clarity of stored values and to be able to represent any value .0000 to .9999 properly.
// For easier debugging, use the Debugger Visualizers available in build/vs/debug/
// to easily display the stored values.
template<size_t FFact, typename T>
struct FPInt
{
protected:
	T v;
	MPT_CONSTEXPR11_FUN FPInt(T rawValue) : v(rawValue) { }

public:
	static const size_t fractFact = FFact;
	typedef T store_t;

	MPT_CONSTEXPR11_FUN FPInt() : v(0) { }
	MPT_CONSTEXPR11_FUN FPInt(const FPInt<fractFact, T> &other) : v(other.v) { }
	MPT_CONSTEXPR11_FUN FPInt(T intPart, T fractPart) : v((intPart * fractFact) + (fractPart % fractFact)) { }
	explicit MPT_CONSTEXPR11_FUN FPInt(float f) : v(static_cast<T>(f * float(fractFact))) { }
	explicit MPT_CONSTEXPR11_FUN FPInt(double f) : v(static_cast<T>(f * double(fractFact))) { }

	// Set integer and fractional part
	MPT_CONSTEXPR14_FUN FPInt<fractFact, T> &Set(T intPart, T fractPart = 0) { v = (intPart * fractFact) + (fractPart % fractFact); return *this; }
	// Set raw internal representation directly
	MPT_CONSTEXPR14_FUN FPInt<fractFact, T> &SetRaw(T value) { v = value; return *this; }
	// Retrieve the integer part of the stored value
	MPT_CONSTEXPR11_FUN T GetInt() const { return v / fractFact; }
	// Retrieve the fractional part of the stored value
	MPT_CONSTEXPR11_FUN T GetFract() const { return v % fractFact; }
	// Retrieve the raw internal representation of the stored value
	MPT_CONSTEXPR11_FUN T GetRaw() const { return v; }
	// Formats the stored value as a floating-point value
	MPT_CONSTEXPR11_FUN double ToDouble() const { return v / double(fractFact); }

	MPT_CONSTEXPR11_FUN FPInt<fractFact, T> operator+ (const FPInt<fractFact, T> &other) const { return FPInt<fractFact, T>(v + other.v); }
	MPT_CONSTEXPR11_FUN FPInt<fractFact, T> operator- (const FPInt<fractFact, T> &other) const { return FPInt<fractFact, T>(v - other.v); }
	MPT_CONSTEXPR14_FUN FPInt<fractFact, T> operator+= (const FPInt<fractFact, T> &other) { v += other.v; return *this; }
	MPT_CONSTEXPR14_FUN FPInt<fractFact, T> operator-= (const FPInt<fractFact, T> &other) { v -= other.v; return *this; }

	MPT_CONSTEXPR11_FUN bool operator== (const FPInt<fractFact, T> &other) const { return v == other.v; }
	MPT_CONSTEXPR11_FUN bool operator!= (const FPInt<fractFact, T> &other) const { return v != other.v; }
	MPT_CONSTEXPR11_FUN bool operator<= (const FPInt<fractFact, T> &other) const { return v <= other.v; }
	MPT_CONSTEXPR11_FUN bool operator>= (const FPInt<fractFact, T> &other) const { return v >= other.v; }
	MPT_CONSTEXPR11_FUN bool operator< (const FPInt<fractFact, T> &other) const { return v < other.v; }
	MPT_CONSTEXPR11_FUN bool operator> (const FPInt<fractFact, T> &other) const { return v > other.v; }
};

typedef FPInt<10000, uint32> TEMPO;

OPENMPT_NAMESPACE_END
