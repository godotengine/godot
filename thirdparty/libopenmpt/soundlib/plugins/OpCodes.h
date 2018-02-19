/*
 * OpCodes.h
 * ---------
 * Purpose: A human-readable list of VST opcodes, for error reporting purposes.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

OPENMPT_NAMESPACE_BEGIN

#ifndef NO_VST
static const char *VstOpCodes[] =
{
	"effOpen",
	"effClose",
	"effSetProgram",
	"effGetProgram",
	"effSetProgramName",
	"effGetProgramName",
	"effGetParamLabel",
	"effGetParamDisplay",
	"effGetParamName",
	"effGetVu",
	"effSetSampleRate",
	"effSetBlockSize",
	"effMainsChanged",
	"effEditGetRect",
	"effEditOpen",
	"effEditClose",
	"effEditDraw",
	"effEditMouse",
	"effEditKey",
	"effEditIdle",
	"effEditTop",
	"effEditSleep",
	"effIdentify",
	"effGetChunk",
	"effSetChunk",
	"effProcessEvents",
	"effCanBeAutomated",
	"effString2Parameter",
	"effGetNumProgramCategories",
	"effGetProgramNameIndexed",
	"effCopyProgram",
	"effConnectInput",
	"effConnectOutput",
	"effGetInputProperties",
	"effGetOutputProperties",
	"effGetPlugCategory",
	"effGetCurrentPosition",
	"effGetDestinationBuffer",
	"effOfflineNotify",
	"effOfflinePrepare",
	"effOfflineRun",
	"effProcessVarIo",
	"effSetSpeakerArrangement",
	"effSetBlockSizeAndSampleRate",
	"effSetBypass",
	"effGetEffectName",
	"effGetErrorText",
	"effGetVendorString",
	"effGetProductString",
	"effGetVendorVersion",
	"effVendorSpecific",
	"effCanDo",
	"effGetTailSize",
	"effIdle",
	"effGetIcon",
	"effSetViewPosition",
	"effGetParameterProperties",
	"effKeysRequired",
	"effGetVstVersion",
	"effEditKeyDown",
	"effEditKeyUp",
	"effSetEditKnobMode",
	"effGetMidiProgramName",
	"effGetCurrentMidiProgram",
	"effGetMidiProgramCategory",
	"effHasMidiProgramsChanged",
	"effGetMidiKeyName",
	"effBeginSetProgram",
	"effEndSetProgram",
	"effGetSpeakerArrangement",
	"effShellGetNextPlugin",
	"effStartProcess",
	"effStopProcess",
	"effSetTotalSampleToProcess",
	"effSetPanLaw",
	"effBeginLoadBank",
	"effBeginLoadProgram",
	"effSetProcessPrecision",
	"effGetNumMidiInputChannels",
	"effGetNumMidiOutputChannels"
};
#endif

OPENMPT_NAMESPACE_END
