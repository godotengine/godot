/*
 * MIDIEvents.h
 * ------------
 * Purpose: MIDI event handling, event lists, ...
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


OPENMPT_NAMESPACE_BEGIN


// MIDI related enums and helper functions
namespace MIDIEvents
{

	// MIDI Event Types
	enum EventType
	{
		evNoteOff			= 0x8,	// Note Off event
		evNoteOn			= 0x9,	// Note On event
		evPolyAftertouch	= 0xA,	// Poly Aftertouch / Poly Pressure event
		evControllerChange	= 0xB,	// Controller Change (see MidiCC enum)
		evProgramChange		= 0xC,	// Program Change
		evChannelAftertouch	= 0xD,	// Channel Aftertouch
		evPitchBend			= 0xE,	// Pitchbend event (see PitchBend enum)
		evSystem			= 0xF,	// System event (see SystemEvent enum)
	};

	// System Events (Fx ...)
	enum SystemEvent
	{
		sysExStart			= 0x0,	// Begin of System Exclusive message
		sysQuarterFrame		= 0x1,	// Quarter Frame Message
		sysPositionPointer	= 0x2,	// Song Position Pointer
		sysSongSelect		= 0x3,	// Song Select
		sysTuneRequest		= 0x6,	// Tune Request
		sysExEnd			= 0x7,	// End of System Exclusive message
		sysMIDIClock		= 0x8,	// MIDI Clock event
		sysMIDITick			= 0x9,	// MIDI Tick event
		sysStart			= 0xA,	// Start Song
		sysContinue			= 0xB,	// Continue Song
		sysStop				= 0xC,	// Stop Song
		sysActiveSense		= 0xE,	// Active Sense Message
		sysReset			= 0xF,	// Reset Device
	};

	// MIDI Pitchbend Constants
	enum PitchBend
	{
		pitchBendMin     = 0x00,
		pitchBendCentre  = 0x2000,
		pitchBendMax     = 0x3FFF
	};

	// MIDI Continuous Controller Codes
	// http://home.roadrunner.com/~jgglatt/tech/midispec/ctllist.htm
	enum MidiCC
	{
		MIDICC_start = 0,
		MIDICC_BankSelect_Coarse = MIDICC_start,
		MIDICC_ModulationWheel_Coarse = 1,
		MIDICC_Breathcontroller_Coarse = 2,
		MIDICC_FootPedal_Coarse = 4,
		MIDICC_PortamentoTime_Coarse = 5,
		MIDICC_DataEntry_Coarse = 6,
		MIDICC_Volume_Coarse = 7,
		MIDICC_Balance_Coarse = 8,
		MIDICC_Panposition_Coarse = 10,
		MIDICC_Expression_Coarse = 11,
		MIDICC_EffectControl1_Coarse = 12,
		MIDICC_EffectControl2_Coarse = 13,
		MIDICC_GeneralPurposeSlider1 = 16,
		MIDICC_GeneralPurposeSlider2 = 17,
		MIDICC_GeneralPurposeSlider3 = 18,
		MIDICC_GeneralPurposeSlider4 = 19,
		MIDICC_BankSelect_Fine = 32,
		MIDICC_ModulationWheel_Fine = 33,
		MIDICC_Breathcontroller_Fine = 34,
		MIDICC_FootPedal_Fine = 36,
		MIDICC_PortamentoTime_Fine = 37,
		MIDICC_DataEntry_Fine = 38,
		MIDICC_Volume_Fine = 39,
		MIDICC_Balance_Fine = 40,
		MIDICC_Panposition_Fine = 42,
		MIDICC_Expression_Fine = 43,
		MIDICC_EffectControl1_Fine = 44,
		MIDICC_EffectControl2_Fine = 45,
		MIDICC_HoldPedal_OnOff = 64,
		MIDICC_Portamento_OnOff = 65,
		MIDICC_SustenutoPedal_OnOff = 66,
		MIDICC_SoftPedal_OnOff = 67,
		MIDICC_LegatoPedal_OnOff = 68,
		MIDICC_Hold2Pedal_OnOff = 69,
		MIDICC_SoundVariation = 70,
		MIDICC_SoundTimbre = 71,
		MIDICC_SoundReleaseTime = 72,
		MIDICC_SoundAttackTime = 73,
		MIDICC_SoundBrightness = 74,
		MIDICC_SoundControl6 = 75,
		MIDICC_SoundControl7 = 76,
		MIDICC_SoundControl8 = 77,
		MIDICC_SoundControl9 = 78,
		MIDICC_SoundControl10 = 79,
		MIDICC_GeneralPurposeButton1_OnOff = 80,
		MIDICC_GeneralPurposeButton2_OnOff = 81,
		MIDICC_GeneralPurposeButton3_OnOff = 82,
		MIDICC_GeneralPurposeButton4_OnOff = 83,
		MIDICC_EffectsLevel = 91,
		MIDICC_TremoloLevel = 92,
		MIDICC_ChorusLevel = 93,
		MIDICC_CelesteLevel = 94,
		MIDICC_PhaserLevel = 95,
		MIDICC_DataButtonincrement = 96,
		MIDICC_DataButtondecrement = 97,
		MIDICC_NonRegisteredParameter_Fine = 98,
		MIDICC_NonRegisteredParameter_Coarse = 99,
		MIDICC_RegisteredParameter_Fine = 100,
		MIDICC_RegisteredParameter_Coarse = 101,
		MIDICC_AllSoundOff = 120,
		MIDICC_AllControllersOff = 121,
		MIDICC_LocalKeyboard_OnOff = 122,
		MIDICC_AllNotesOff = 123,
		MIDICC_OmniModeOff = 124,
		MIDICC_OmniModeOn = 125,
		MIDICC_MonoOperation = 126,
		MIDICC_PolyOperation = 127,
		MIDICC_end = MIDICC_PolyOperation,
	};

	// MIDI CC Names
	extern const char* const MidiCCNames[MIDICC_end + 1];

	// Build a generic MIDI event
	uint32 Event(EventType eventType, uint8 midiChannel, uint8 dataByte1, uint8 dataByte2);
	// Build a MIDI CC event
	uint32 CC(MidiCC midiCC, uint8 midiChannel, uint8 param);
	// Build a MIDI Pitchbend event
	uint32 PitchBend(uint8 midiChannel, uint16 bendAmount);
	// Build a MIDI Program Change event
	uint32 ProgramChange(uint8 midiChannel, uint8 program);
	// Build a MIDI Note Off event
	uint32 NoteOff(uint8 midiChannel, uint8 note, uint8 velocity);
	// Build a MIDI Note On event
	uint32 NoteOn(uint8 midiChannel, uint8 note, uint8 velocity);
	// Build a MIDI System Event
	uint8 System(SystemEvent eventType);

	// Get MIDI channel from a MIDI event
	uint8 GetChannelFromEvent(uint32 midiMsg);
	// Get MIDI Event type from a MIDI event
	EventType GetTypeFromEvent(uint32 midiMsg);
	// Get first data byte from a MIDI event
	uint8 GetDataByte1FromEvent(uint32 midiMsg);
	// Get second data byte from a MIDI event
	uint8 GetDataByte2FromEvent(uint32 midiMsg);

	// Get the length of a MIDI event in bytes
	uint8 GetEventLength(uint8 firstByte);

}


OPENMPT_NAMESPACE_END
