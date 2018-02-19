/*
 * ModCommand.cpp
 * --------------
 * Purpose: Various functions for writing effects to patterns, converting ModCommands, etc.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#include "mod_specifications.h"
#include "Tables.h"


OPENMPT_NAMESPACE_BEGIN


const EffectType effectTypes[] =
{
	EFFECT_TYPE_NORMAL, EFFECT_TYPE_NORMAL,  EFFECT_TYPE_PITCH,  EFFECT_TYPE_PITCH,
	EFFECT_TYPE_PITCH,  EFFECT_TYPE_PITCH,   EFFECT_TYPE_VOLUME, EFFECT_TYPE_VOLUME,
	EFFECT_TYPE_VOLUME, EFFECT_TYPE_PANNING, EFFECT_TYPE_NORMAL, EFFECT_TYPE_VOLUME,
	EFFECT_TYPE_GLOBAL, EFFECT_TYPE_VOLUME,  EFFECT_TYPE_GLOBAL, EFFECT_TYPE_NORMAL,
	EFFECT_TYPE_GLOBAL, EFFECT_TYPE_GLOBAL,  EFFECT_TYPE_NORMAL, EFFECT_TYPE_NORMAL,
	EFFECT_TYPE_NORMAL, EFFECT_TYPE_VOLUME,  EFFECT_TYPE_VOLUME, EFFECT_TYPE_GLOBAL,
	EFFECT_TYPE_GLOBAL, EFFECT_TYPE_NORMAL,  EFFECT_TYPE_PITCH,  EFFECT_TYPE_PANNING,
	EFFECT_TYPE_PITCH,  EFFECT_TYPE_PANNING, EFFECT_TYPE_NORMAL, EFFECT_TYPE_NORMAL,
	EFFECT_TYPE_NORMAL, EFFECT_TYPE_NORMAL,  EFFECT_TYPE_NORMAL, EFFECT_TYPE_PITCH,
	EFFECT_TYPE_PITCH,  EFFECT_TYPE_PITCH,   EFFECT_TYPE_PITCH,  EFFECT_TYPE_NORMAL,
	EFFECT_TYPE_NORMAL, EFFECT_TYPE_NORMAL,
};

STATIC_ASSERT(CountOf(effectTypes) == MAX_EFFECTS);


const EffectType volumeEffectTypes[] =
{
	EFFECT_TYPE_NORMAL, EFFECT_TYPE_VOLUME,  EFFECT_TYPE_PANNING, EFFECT_TYPE_VOLUME,
	EFFECT_TYPE_VOLUME, EFFECT_TYPE_VOLUME,  EFFECT_TYPE_VOLUME,  EFFECT_TYPE_PITCH,
	EFFECT_TYPE_PITCH,  EFFECT_TYPE_PANNING, EFFECT_TYPE_PANNING, EFFECT_TYPE_PITCH,
	EFFECT_TYPE_PITCH,  EFFECT_TYPE_PITCH,   EFFECT_TYPE_NORMAL,  EFFECT_TYPE_NORMAL,
};

STATIC_ASSERT(CountOf(volumeEffectTypes) == MAX_VOLCMDS);


EffectType ModCommand::GetEffectType(COMMAND cmd)
{
	if(cmd < CountOf(effectTypes))
	{
		return effectTypes[cmd];
	} else
	{
		return EFFECT_TYPE_NORMAL;
	}
}


EffectType ModCommand::GetVolumeEffectType(VOLCMD volcmd)
{
	if(volcmd < CountOf(volumeEffectTypes))
	{
		return volumeEffectTypes[volcmd];
	} else
	{
		return EFFECT_TYPE_NORMAL;
	}
}


// Convert an Exx command (MOD) to Sxx command (S3M)
void ModCommand::ExtendedMODtoS3MEffect()
{
	if(command != CMD_MODCMDEX)
		return;

	command = CMD_S3MCMDEX;
	switch(param & 0xF0)
	{
	case 0x10: command = CMD_PORTAMENTOUP; param |= 0xF0; break;
	case 0x20: command = CMD_PORTAMENTODOWN; param |= 0xF0; break;
	case 0x30: param = (param & 0x0F) | 0x10; break;
	case 0x40: param = (param & 0x03) | 0x30; break;
	case 0x50: param = (param & 0x0F) | 0x20; break;
	case 0x60: param = (param & 0x0F) | 0xB0; break;
	case 0x70: param = (param & 0x03) | 0x40; break;
	case 0x90: command = CMD_RETRIG; param = (param & 0x0F); break;
	case 0xA0: if(param & 0x0F) { command = CMD_VOLUMESLIDE; param = (param << 4) | 0x0F; } else command = CMD_NONE; break;
	case 0xB0: if(param & 0x0F) { command = CMD_VOLUMESLIDE; param |= 0xF0; } else command = CMD_NONE; break;
	case 0xC0: if(param == 0xC0) { command = CMD_NONE; note = NOTE_NOTECUT; } break;  // this does different things in IT and ST3
	case 0xD0: if(param == 0xD0) { command = CMD_NONE; } break;  // ditto
	// rest are the same
	}
}


// Convert an Sxx command (S3M) to Exx command (MOD)
void ModCommand::ExtendedS3MtoMODEffect()
{
	if(command != CMD_S3MCMDEX)
		return;

	command = CMD_MODCMDEX;
	switch(param & 0xF0)
	{
	case 0x10: param = (param & 0x0F) | 0x30; break;
	case 0x20: param = (param & 0x0F) | 0x50; break;
	case 0x30: param = (param & 0x0F) | 0x40; break;
	case 0x40: param = (param & 0x0F) | 0x70; break;
	case 0x50: command = CMD_XFINEPORTAUPDOWN; break;  // map to unused X5x
	case 0x60: command = CMD_XFINEPORTAUPDOWN; break;  // map to unused X6x
	case 0x90: command = CMD_XFINEPORTAUPDOWN; break;  // map to unused X9x
	case 0xA0: command = CMD_XFINEPORTAUPDOWN; break;  // map to unused XAx
	case 0xB0: param = (param & 0x0F) | 0x60; break;
	case 0x70: command = CMD_NONE; break;  // No NNA / envelope control in MOD/XM format
	// rest are the same
	}
}


// Convert a mod command from one format to another.
void ModCommand::Convert(MODTYPE fromType, MODTYPE toType, const CSoundFile &sndFile)
{
	if(fromType == toType)
	{
		return;
	}

	if(fromType == MOD_TYPE_MTM)
	{
		// Special MTM fixups.
		// Retrigger with param 0
		if(command == CMD_MODCMDEX && param == 0x90)
		{
			command = CMD_NONE;
		} else if(command == CMD_VIBRATO)
		{
			// Vibrato is approximately half as deep compared to MOD/S3M.
			uint8 speed = (param & 0xF0);
			uint8 depth = (param & 0x0F) >> 1;
			param = speed | depth;
		}
		// Apart from these special fixups, do a regular conversion from MOD.
		fromType = MOD_TYPE_MOD;
	}

	// helper variables
	const bool oldTypeIsMOD = (fromType == MOD_TYPE_MOD), oldTypeIsXM = (fromType == MOD_TYPE_XM),
		oldTypeIsS3M = (fromType == MOD_TYPE_S3M), oldTypeIsIT = (fromType == MOD_TYPE_IT),
		oldTypeIsMPT = (fromType == MOD_TYPE_MPT), oldTypeIsMOD_XM = (oldTypeIsMOD || oldTypeIsXM),
		oldTypeIsS3M_IT_MPT = (oldTypeIsS3M || oldTypeIsIT || oldTypeIsMPT),
		oldTypeIsIT_MPT = (oldTypeIsIT || oldTypeIsMPT);

	const bool newTypeIsMOD = (toType == MOD_TYPE_MOD), newTypeIsXM =  (toType == MOD_TYPE_XM),
		newTypeIsS3M = (toType == MOD_TYPE_S3M), newTypeIsIT = (toType == MOD_TYPE_IT),
		newTypeIsMPT = (toType == MOD_TYPE_MPT), newTypeIsMOD_XM = (newTypeIsMOD || newTypeIsXM),
		newTypeIsS3M_IT_MPT = (newTypeIsS3M || newTypeIsIT || newTypeIsMPT),
		newTypeIsIT_MPT = (newTypeIsIT || newTypeIsMPT);

	const CModSpecifications &newSpecs = CSoundFile::GetModSpecifications(toType);

	//////////////////////////
	// Convert 8-bit Panning
	if(command == CMD_PANNING8)
	{
		if(newTypeIsS3M)
		{
			param = (param + 1) >> 1;
		} else if(oldTypeIsS3M)
		{
			if(param == 0xA4)
			{
				// surround remap
				command = static_cast<COMMAND>((toType & (MOD_TYPE_IT | MOD_TYPE_MPT)) ? CMD_S3MCMDEX : CMD_XFINEPORTAUPDOWN);
				param = 0x91;
			} else
			{
				param = std::min<PARAM>(param << 1, 0xFF);
			}
		}
	} // End if(command == CMD_PANNING8)

	// Re-map \xx to Zxx if the new format only knows the latter command.
	if(command == CMD_SMOOTHMIDI && !newSpecs.HasCommand(CMD_SMOOTHMIDI) && newSpecs.HasCommand(CMD_MIDI))
	{
		command = CMD_MIDI;
	}

	///////////////////////////////////////////////////////////////////////////////////////
	// MPTM to anything: Convert param control, extended envelope control, note delay+cut
	if(oldTypeIsMPT)
	{
		if(IsPcNote())
		{
			COMMAND newCmd = static_cast<COMMAND>(note == NOTE_PC ? CMD_MIDI : CMD_SMOOTHMIDI);
			if(!newSpecs.HasCommand(newCmd))
			{
				newCmd = CMD_MIDI;	// assuming that this was CMD_SMOOTHMIDI
				if(!newSpecs.HasCommand(newCmd))
				{
					newCmd = CMD_NONE;
				}
			}

			param = (PARAM)(std::min<uint16>(maxColumnValue, GetValueEffectCol()) * 0x7F / maxColumnValue);
			command = newCmd; // might be removed later
			volcmd = VOLCMD_NONE;
			note = NOTE_NONE;
			instr = 0;
		}

		// adjust extended envelope control commands
		if((command == CMD_S3MCMDEX) && ((param & 0xF0) == 0x70) && ((param & 0x0F) > 0x0C))
		{
			param = 0x7C;
		}

		if(command == CMD_DELAYCUT)
		{
			command = CMD_S3MCMDEX;			// When converting to MOD/XM, this will be converted to CMD_MODCMDEX later
			param = 0xD0 | (param >> 4);	// Preserve delay nibble.
		}
	} // End if(oldTypeIsMPT)

	/////////////////////////////////////////
	// Convert MOD / XM to S3M / IT / MPTM
	if(oldTypeIsMOD_XM && newTypeIsS3M_IT_MPT)
	{
		switch(command)
		{
		case CMD_ARPEGGIO:
			if(!param) command = CMD_NONE;	// 000 does nothing in MOD/XM
			break;

		case CMD_MODCMDEX:
			ExtendedMODtoS3MEffect();
			break;

		case CMD_VOLUME:
			// Effect column volume command overrides the volume column in XM.
			if(volcmd == VOLCMD_NONE || volcmd == VOLCMD_VOLUME)
			{
				volcmd = VOLCMD_VOLUME;
				vol = param;
				if(vol > 64) vol = 64;
				command = CMD_NONE;
				param = 0;
			} else if(volcmd == VOLCMD_PANNING)
			{
				std::swap(vol, param);
				volcmd = VOLCMD_VOLUME;
				if(vol > 64) vol = 64;
				command = CMD_S3MCMDEX;
				param = 0x80 | (param / 4);	// XM volcol panning is actually 4-Bit, so we can use 4-Bit panning here.
			}
			break;

		case CMD_PORTAMENTOUP:
			if(param > 0xDF) param = 0xDF;
			break;

		case CMD_PORTAMENTODOWN:
			if(param > 0xDF) param = 0xDF;
			break;

		case CMD_XFINEPORTAUPDOWN:
			switch(param & 0xF0)
			{
			case 0x10:	command = CMD_PORTAMENTOUP; param = (param & 0x0F) | 0xE0; break;
			case 0x20:	command = CMD_PORTAMENTODOWN; param = (param & 0x0F) | 0xE0; break;
			case 0x50:
			case 0x60:
			case 0x70:
			case 0x90:
			case 0xA0:
				command = CMD_S3MCMDEX;
				// Surround remap (this is the "official" command)
				if(toType & MOD_TYPE_S3M && param == 0x91)
				{
					command = CMD_PANNING8;
					param = 0xA4;
				}
				break;
			}
			break;

		case CMD_KEYOFF:
			if(note == NOTE_NONE)
			{
				note = newTypeIsS3M ? NOTE_NOTECUT : NOTE_KEYOFF;
				command = CMD_S3MCMDEX;
				if(param == 0)
					instr = 0;
				param = 0xD0 | (param & 0x0F);
			}
			break;

		case CMD_PANNINGSLIDE:
			// swap L/R, convert to fine slide
			if(param & 0xF0)
			{
				param = 0xF0 | std::min<PARAM>(0x0E, (param >> 4));
			} else
			{
				param = 0x0F | (std::min<PARAM>(0x0E, param & 0x0F) << 4);
			}

		default:
			break;
		}
	} // End if(oldTypeIsMOD_XM && newTypeIsS3M_IT_MPT)


	/////////////////////////////////////////
	// Convert S3M / IT / MPTM to MOD / XM
	else if(oldTypeIsS3M_IT_MPT && newTypeIsMOD_XM)
	{
		if(note == NOTE_NOTECUT)
		{
			// convert note cut to EC0 if possible or volume command otherwise (MOD/XM has no real way of cutting notes that cannot be "undone" by volume commands)
			note = NOTE_NONE;
			if(command == CMD_NONE || !newTypeIsXM)
			{
				command = CMD_MODCMDEX;
				param = 0xC0;
			} else
			{
				volcmd = VOLCMD_VOLUME;
				vol = 0;
			}
		} else if(note == NOTE_FADE)
		{
			// convert note fade to note off
			note = NOTE_KEYOFF;
		}

		switch(command)
		{
		case CMD_S3MCMDEX:
			ExtendedS3MtoMODEffect();
			break;

		case CMD_TONEPORTAVOL:	// Can't do fine slides and portamento/vibrato at the same time :(
		case CMD_VIBRATOVOL:	// ditto
			if(volcmd == VOLCMD_NONE && (((param & 0xF0) && ((param & 0x0F) == 0x0F)) || ((param & 0x0F) && ((param & 0xF0) == 0xF0))))
			{
				// Try to salvage portamento/vibrato
				if(command == CMD_TONEPORTAVOL)
					volcmd = VOLCMD_TONEPORTAMENTO;
				else if(command == CMD_VIBRATOVOL)
					volcmd = VOLCMD_VIBRATODEPTH;
				vol = 0;
			}

		MPT_FALLTHROUGH;
		case CMD_VOLUMESLIDE:
			if((param & 0xF0) && ((param & 0x0F) == 0x0F))
			{
				command = CMD_MODCMDEX;
				param = (param >> 4) | 0xA0;
			} else if((param & 0x0F) && ((param & 0xF0) == 0xF0))
			{
				command = CMD_MODCMDEX;
				param = (param & 0x0F) | 0xB0;
			}
			break;

		case CMD_PORTAMENTOUP:
			if(param >= 0xF0)
			{
				command = CMD_MODCMDEX;
				param = (param & 0x0F) | 0x10;
			} else if(param >= 0xE0)
			{
				if(newTypeIsXM)
				{
					command = CMD_XFINEPORTAUPDOWN;
					param = 0x10 | (param & 0x0F);
				} else
				{
					command = CMD_MODCMDEX;
					param = (((param & 0x0F) + 3) >> 2) | 0x10;
				}
			} else
			{
				command = CMD_PORTAMENTOUP;
			}
			break;

		case CMD_PORTAMENTODOWN:
			if(param >= 0xF0)
			{
				command = CMD_MODCMDEX;
				param = (param & 0x0F) | 0x20;
			} else if(param >= 0xE0)
			{
				if(newTypeIsXM)
				{
					command = CMD_XFINEPORTAUPDOWN;
					param = 0x20 | (param & 0x0F);
				} else
				{
					command = CMD_MODCMDEX;
					param = (((param & 0x0F) + 3) >> 2) | 0x20;
				}
			} else
			{
				command = CMD_PORTAMENTODOWN;
			}
			break;

		case CMD_TEMPO:
			if(param < 0x20) command = CMD_NONE; // no tempo slides
			break;

		case CMD_PANNINGSLIDE:
			// swap L/R, convert fine slides to normal slides
			if((param & 0x0F) == 0x0F && (param & 0xF0))
			{
				param = (param >> 4);
			} else if((param & 0xF0) == 0xF0 && (param & 0x0F))
			{
				param = (param & 0x0F) << 4;
			} else if(param & 0x0F)
			{
				param = 0xF0;
			} else if(param & 0xF0)
			{
				param = 0x0F;
			} else
			{
				param = 0;
			}
			break;

		case CMD_RETRIG:
			// Retrig: Q0y doesn't change volume in IT/S3M, but R0y in XM takes the last x parameter
			if(param != 0 && (param & 0xF0) == 0)
			{
				param |= 0x80;
			}
			break;

		default:
			break;
		}
	} // End if(oldTypeIsS3M_IT_MPT && newTypeIsMOD_XM)


	///////////////////////
	// Convert IT to S3M
	else if(oldTypeIsIT_MPT && newTypeIsS3M)
	{
		if(note == NOTE_KEYOFF || note == NOTE_FADE)
			note = NOTE_NOTECUT;

		switch(command)
		{
		case CMD_S3MCMDEX:
			switch(param & 0xF0)
			{
			case 0x70: command = CMD_NONE; break;	// No NNA / envelope control in S3M format
			case 0x90:
				if(param == 0x91)
				{
					// surround remap (this is the "official" command)
					command = CMD_PANNING8;
					param = 0xA4;
				} else if(param == 0x90)
				{
					command = CMD_PANNING8;
					param = 0x40;
				}
				break;
			}
			break;

		case CMD_GLOBALVOLUME:
			param = (std::min<PARAM>(0x80, param) + 1) / 2u;
			break;

		default:
			break;
		}
	} // End if(oldTypeIsIT_MPT && newTypeIsS3M)

	//////////////////////
	// Convert IT to XM
	if(oldTypeIsIT_MPT && newTypeIsXM)
	{
		switch(command)
		{
		case CMD_VIBRATO:
			// With linear slides, strength is roughly doubled.
			param = (param & 0xF0) | (((param & 0x0F) + 1) / 2u);
			break;
		case CMD_GLOBALVOLUME:
			param = (std::min<PARAM>(0x80, param) + 1) / 2u;
			break;
		}
	} // End if(oldTypeIsIT_MPT && newTypeIsXM)

	//////////////////////
	// Convert XM to IT
	if(oldTypeIsXM && newTypeIsIT_MPT)
	{
		switch(command)
		{
		case CMD_VIBRATO:
			// With linear slides, strength is roughly halved.
			param = (param & 0xF0) | std::min<PARAM>((param & 0x0F) * 2u, 15);
			break;
		case CMD_GLOBALVOLUME:
			param = std::min<PARAM>(0x80, param * 2u);
			break;
		}
	} // End if(oldTypeIsIT_MPT && newTypeIsXM)

	///////////////////////////////////
	// MOD / XM Speed/Tempo limits
	if(newTypeIsMOD_XM)
	{
		switch(command)
		{
		case CMD_SPEED:
			param = std::min<PARAM>(param, 0x1F);
			break;
			break;
		case CMD_TEMPO:
			param = std::max<PARAM>(param, 0x20);
			break;
		}
	}

	///////////////////////////////////////////////////////////////////////
	// Convert MOD to anything - adjust effect memory, remove Invert Loop
	if(oldTypeIsMOD)
	{
		switch(command)
		{
		case CMD_TONEPORTAVOL: // lacks memory -> 500 is the same as 300
			if(param == 0x00) command = CMD_TONEPORTAMENTO;
			break;

		case CMD_VIBRATOVOL: // lacks memory -> 600 is the same as 400
			if(param == 0x00) command = CMD_VIBRATO;
			break;

		case CMD_MODCMDEX: // This would turn into "Set Active Macro", so let's better remove it
		case CMD_S3MCMDEX:
			if((param & 0xF0) == 0xF0) command = CMD_NONE;
			break;
		}
	} // End if(oldTypeIsMOD && newTypeIsXM)

	/////////////////////////////////////////////////////////////////////
	// Convert anything to MOD - remove volume column, remove Set Macro
	if(newTypeIsMOD)
	{
		// convert note off events
		if(IsSpecialNote())
		{
			note = NOTE_NONE;
			// no effect present, so just convert note off to volume 0
			if(command == CMD_NONE)
			{
				command = CMD_VOLUME;
				param = 0;
				// EDx effect present, so convert it to ECx
			} else if((command == CMD_MODCMDEX) && ((param & 0xF0) == 0xD0))
			{
				param = 0xC0 | (param & 0x0F);
			}
		}

		if(command != CMD_NONE) switch(command)
		{
			case CMD_RETRIG: // MOD only has E9x
				command = CMD_MODCMDEX;
				param = 0x90 | (param & 0x0F);
				break;

			case CMD_MODCMDEX: // This would turn into "Invert Loop", so let's better remove it
				if((param & 0xF0) == 0xF0) command = CMD_NONE;
				break;
		}

		if(command == CMD_NONE) switch(volcmd)
		{
			case VOLCMD_VOLUME:
				command = CMD_VOLUME;
				param = vol;
				break;

			case VOLCMD_PANNING:
				command = CMD_PANNING8;
				param = vol < 64 ? vol << 2 : 255;
				break;

			case VOLCMD_VOLSLIDEDOWN:
				command = CMD_VOLUMESLIDE;
				param = vol;
				break;

			case VOLCMD_VOLSLIDEUP:
				command = CMD_VOLUMESLIDE;
				param = vol << 4;
				break;

			case VOLCMD_FINEVOLDOWN:
				command = CMD_MODCMDEX;
				param = 0xB0 | vol;
				break;

			case VOLCMD_FINEVOLUP:
				command = CMD_MODCMDEX;
				param = 0xA0 | vol;
				break;

			case VOLCMD_PORTADOWN:
				command = CMD_PORTAMENTODOWN;
				param = vol << 2;
				break;

			case VOLCMD_PORTAUP:
				command = CMD_PORTAMENTOUP;
				param = vol << 2;
				break;

			case VOLCMD_TONEPORTAMENTO:
				command = CMD_TONEPORTAMENTO;
				param = vol << 2;
				break;

			case VOLCMD_VIBRATODEPTH:
				command = CMD_VIBRATO;
				param = vol;
				break;

			case VOLCMD_VIBRATOSPEED:
				command = CMD_VIBRATO;
				param = vol << 4;
				break;
		}
		volcmd = VOLCMD_NONE;
	} // End if(newTypeIsMOD)

	///////////////////////////////////////////////////
	// Convert anything to S3M - adjust volume column
	if(newTypeIsS3M)
	{
		if(command == CMD_NONE) switch(volcmd)
		{
			case VOLCMD_VOLSLIDEDOWN:
				command = CMD_VOLUMESLIDE;
				param = vol;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_VOLSLIDEUP:
				command = CMD_VOLUMESLIDE;
				param = vol << 4;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_FINEVOLDOWN:
				command = CMD_VOLUMESLIDE;
				param = 0xF0 | vol;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_FINEVOLUP:
				command = CMD_VOLUMESLIDE;
				param = (vol << 4) | 0x0F;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_PORTADOWN:
				command = CMD_PORTAMENTODOWN;
				param = vol << 2;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_PORTAUP:
				command = CMD_PORTAMENTOUP;
				param = vol << 2;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_TONEPORTAMENTO:
				command = CMD_TONEPORTAMENTO;
				param = vol << 2;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_VIBRATODEPTH:
				command = CMD_VIBRATO;
				param = vol;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_VIBRATOSPEED:
				command = CMD_VIBRATO;
				param = vol << 4;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_PANSLIDELEFT:
				command = CMD_PANNINGSLIDE;
				param = vol << 4;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_PANSLIDERIGHT:
				command = CMD_PANNINGSLIDE;
				param = vol;
				volcmd = VOLCMD_NONE;
				break;
		}
	} // End if(newTypeIsS3M)

	////////////////////////////////////////////////////////////////////////
	// Convert anything to XM - adjust volume column, breaking EDx command
	if(newTypeIsXM)
	{
		// remove EDx if no note is next to it, or it will retrigger the note in FT2 mode
		if(command == CMD_MODCMDEX && (param & 0xF0) == 0xD0 && note == NOTE_NONE)
		{
			command = CMD_NONE;
			param = 0;
		}

		if(IsSpecialNote())
		{
			// Instrument numbers next to Note Off reset instrument settings
			instr = 0;

			if(command == CMD_MODCMDEX && (param & 0xF0) == 0xD0)
			{
				// Note Off + Note Delay does nothing when using envelopes.
				note = NOTE_NONE;
				command = CMD_KEYOFF;
				param &= 0x0F;
			}
		}

		// Convert some commands which behave differently or don't exist
		if(command == CMD_NONE) switch(volcmd)
		{
			case VOLCMD_PORTADOWN:
				command = CMD_PORTAMENTODOWN;
				param = vol << 2;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_PORTAUP:
				command = CMD_PORTAMENTOUP;
				param = vol << 2;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_TONEPORTAMENTO:
				command = CMD_TONEPORTAMENTO;
				param = ImpulseTrackerPortaVolCmd[vol & 0x0F];
				volcmd = VOLCMD_NONE;
				break;
		}
	} // End if(newTypeIsXM)

	///////////////////////////////////////////////////
	// Convert anything to IT - adjust volume column
	if(newTypeIsIT_MPT)
	{
		// Convert some commands which behave differently or don't exist
		if(command == CMD_NONE) switch(volcmd)
		{
			case VOLCMD_PANSLIDELEFT:
				command = CMD_PANNINGSLIDE;
				param = vol << 4;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_PANSLIDERIGHT:
				command = CMD_PANNINGSLIDE;
				param = vol;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_VIBRATOSPEED:
				command = CMD_VIBRATO;
				param = vol << 4;
				volcmd = VOLCMD_NONE;
				break;

			case VOLCMD_TONEPORTAMENTO:
				command = CMD_TONEPORTAMENTO;
				param = vol << 4;
				volcmd = VOLCMD_NONE;
				break;
		}

		switch(volcmd)
		{
		case VOLCMD_VOLSLIDEDOWN:
		case VOLCMD_VOLSLIDEUP:
		case VOLCMD_FINEVOLDOWN:
		case VOLCMD_FINEVOLUP:
		case VOLCMD_PORTADOWN:
		case VOLCMD_PORTAUP:
		case VOLCMD_TONEPORTAMENTO:
		case VOLCMD_VIBRATODEPTH:
			// OpenMPT-specific commands
		case VOLCMD_OFFSET:
			vol = std::min<PARAM>(vol, 9);
			break;
		}
	} // End if(newTypeIsIT_MPT)

	// Fix volume column offset for formats that don't have it.
	if(volcmd == VOLCMD_OFFSET && !newSpecs.HasVolCommand(VOLCMD_OFFSET) && (command == CMD_NONE || !newSpecs.HasCommand(command)))
	{
		command = CMD_OFFSET;
		volcmd = VOLCMD_NONE;
		SAMPLEINDEX smp = instr;
		if(smp > 0 && smp <= sndFile.GetNumInstruments() && IsNote() && sndFile.Instruments[smp] != nullptr)
			smp = sndFile.Instruments[smp]->Keyboard[note - NOTE_MIN];

		if(smp > 0 && smp <= sndFile.GetNumSamples() && vol > 0 && vol <= CountOf(sndFile.GetSample(smp).cues))
			param = mpt::saturate_cast<ModCommand::PARAM>((sndFile.GetSample(smp).cues[vol - 1] + 128) >> 8);
		else
			param = vol << 3;
	}

	if((command == CMD_REVERSEOFFSET || command == CMD_OFFSETPERCENTAGE) && !newSpecs.HasCommand(command))
	{
		command = CMD_OFFSET;
	}

	if(!newSpecs.HasNote(note))
		note = NOTE_NONE;

	// ensure the commands really exist in this format
	if(!newSpecs.HasCommand(command))
		command = CMD_NONE;
	if(!newSpecs.HasVolCommand(volcmd))
		volcmd = VOLCMD_NONE;

}


bool ModCommand::IsGlobalCommand() const
{
	switch(command)
	{
	case CMD_POSITIONJUMP:
	case CMD_PATTERNBREAK:
	case CMD_SPEED:
	case CMD_TEMPO:
	case CMD_GLOBALVOLUME:
	case CMD_GLOBALVOLSLIDE:
	case CMD_MIDI:
	case CMD_SMOOTHMIDI:
	case CMD_DBMECHO:
		return true;
	case CMD_MODCMDEX:
		switch(param & 0xF0)
		{
		case 0x00:	// LED Filter
		case 0x60:	// Pattern Loop
		case 0xE0:	// Row Delay
			return true;
		default:
			return false;
		}
	case CMD_XFINEPORTAUPDOWN:
	case CMD_S3MCMDEX:
		switch(param & 0xF0)
		{
		case 0x60:	// Tick Delay
		case 0x90:	// Sound Control
		case 0xB0:	// Pattern Loop
		case 0xE0:	// Row Delay
			return true;
		default:
			return false;
		}

	default:
		return false;
	}
}

// "Importance" of every FX command. Table is used for importing from formats with multiple effect colums
// and is approximately the same as in SchismTracker.
size_t ModCommand::GetEffectWeight(COMMAND cmd)
{
	// Effect weights, sorted from lowest to highest weight.
	static const COMMAND weights[] =
	{
		CMD_NONE,
		CMD_XPARAM,
		CMD_SETENVPOSITION,
		CMD_KEYOFF,
		CMD_TREMOLO,
		CMD_FINEVIBRATO,
		CMD_VIBRATO,
		CMD_XFINEPORTAUPDOWN,
		CMD_PANBRELLO,
		CMD_S3MCMDEX,
		CMD_MODCMDEX,
		CMD_DELAYCUT,
		CMD_MIDI,
		CMD_SMOOTHMIDI,
		CMD_PANNINGSLIDE,
		CMD_PANNING8,
		CMD_NOTESLIDEUPRETRIG,
		CMD_NOTESLIDEUP,
		CMD_NOTESLIDEDOWNRETRIG,
		CMD_NOTESLIDEDOWN,
		CMD_PORTAMENTOUP,
		CMD_PORTAMENTODOWN,
		CMD_VOLUMESLIDE,
		CMD_VIBRATOVOL,
		CMD_VOLUME,
		CMD_REVERSEOFFSET,
		CMD_OFFSETPERCENTAGE,
		CMD_OFFSET,
		CMD_TREMOR,
		CMD_RETRIG,
		CMD_ARPEGGIO,
		CMD_TONEPORTAMENTO,
		CMD_TONEPORTAVOL,
		CMD_DBMECHO,
		CMD_GLOBALVOLSLIDE,
		CMD_CHANNELVOLUME,
		CMD_GLOBALVOLSLIDE,
		CMD_GLOBALVOLUME,
		CMD_TEMPO,
		CMD_SPEED,
		CMD_POSITIONJUMP,
		CMD_PATTERNBREAK,
	};
	STATIC_ASSERT(CountOf(weights) == MAX_EFFECTS);

	for(size_t i = 0; i < CountOf(weights); i++)
	{
		if(weights[i] == cmd)
		{
			return i;
		}
	}
	// Invalid / unknown command.
	return 0;
}


// Try to convert a fx column command (&effect) into a volume column command.
// Returns true if successful.
// Some commands can only be converted by losing some precision.
// If moving the command into the volume column is more important than accuracy, use force = true.
// (Code translated from SchismTracker and mainly supposed to be used with loaders ported from this tracker)
bool ModCommand::ConvertVolEffect(uint8 &effect, uint8 &param, bool force)
{
	switch(effect)
	{
	case CMD_NONE:
		return true;
	case CMD_VOLUME:
		effect = VOLCMD_VOLUME;
		param = std::min<PARAM>(param, 64);
		break;
	case CMD_PORTAMENTOUP:
		// if not force, reject when dividing causes loss of data in LSB, or if the final value is too
		// large to fit. (volume column Ex/Fx are four times stronger than effect column)
		if(!force && ((param & 3) || param >= 0xE0))
			return false;
		param /= 4;
		effect = VOLCMD_PORTAUP;
		break;
	case CMD_PORTAMENTODOWN:
		if(!force && ((param & 3) || param >= 0xE0))
			return false;
		param /= 4;
		effect = VOLCMD_PORTADOWN;
		break;
	case CMD_TONEPORTAMENTO:
		if(param >= 0xF0)
		{
			// hack for people who can't type F twice :)
			effect = VOLCMD_TONEPORTAMENTO;
			param = 9;
			return true;
		}
		for(uint8 n = 0; n < 10; n++)
		{
			if(force
				? (param <= ImpulseTrackerPortaVolCmd[n])
				: (param == ImpulseTrackerPortaVolCmd[n]))
			{
				effect = VOLCMD_TONEPORTAMENTO;
				param = n;
				return true;
			}
		}
		return false;
	case CMD_VIBRATO:
		if(force)
			param = std::min<PARAM>(param & 0x0F, 9);
		else if((param & 0x0F) > 9 || (param & 0xF0) != 0)
			return false;
		param &= 0x0F;
		effect = VOLCMD_VIBRATODEPTH;
		break;
	case CMD_FINEVIBRATO:
		if(force)
			param = 0;
		else if(param)
			return false;
		effect = VOLCMD_VIBRATODEPTH;
		break;
	case CMD_PANNING8:
		if(param == 255)
			param = 64;
		else
			param /= 4;
		effect = VOLCMD_PANNING;
		break;
	case CMD_VOLUMESLIDE:
		if(param == 0)
			return false;
		if((param & 0xF) == 0)	// Dx0 / Cx
		{
			param >>= 4;
			effect = VOLCMD_VOLSLIDEUP;
		} else if((param & 0xF0) == 0)	// D0x / Dx
		{
			effect = VOLCMD_VOLSLIDEDOWN;
		} else if((param & 0xF) == 0xF)	// DxF / Ax
		{
			param >>= 4;
			effect = VOLCMD_FINEVOLUP;
		} else if((param & 0xF0) == 0xF0)	// DFx / Bx
		{
			param &= 0xF;
			effect = VOLCMD_FINEVOLDOWN;
		} else // ???
		{
			return false;
		}
		break;
	case CMD_S3MCMDEX:
		switch (param >> 4)
		{
		case 8:
			effect = VOLCMD_PANNING;
			param = ((param & 0xF) << 2) + 2;
			return true;
		case 0: case 1: case 2: case 0xF:
			if(force)
			{
				effect = param = 0;
				return true;
			}
			break;
		default:
			break;
		}
		return false;
	default:
		return false;
	}
	return true;
}

// Try to combine two commands into one. Returns true on success and the combined command is placed in eff1 / param1.
bool ModCommand::CombineEffects(uint8 &eff1, uint8 &param1, uint8 &eff2, uint8 &param2)
{
	if(eff1 == CMD_VOLUMESLIDE && (eff2 == CMD_VIBRATO || eff2 == CMD_TONEPORTAVOL) && param2 == 0)
	{
		// Merge commands
		if(eff2 == CMD_VIBRATO)
		{
			eff1 = CMD_VIBRATOVOL;
		} else
		{
			eff1 = CMD_TONEPORTAVOL;
		}
		eff2 = CMD_NONE;
		return true;
	} else if(eff2 == CMD_VOLUMESLIDE && (eff1 == CMD_VIBRATO || eff1 == CMD_TONEPORTAVOL) && param1 == 0)
	{
		// Merge commands
		if(eff1 == CMD_VIBRATO)
		{
			eff1 = CMD_VIBRATOVOL;
		} else
		{
			eff1 = CMD_TONEPORTAVOL;
		}
		param1 = param2;
		eff2 = CMD_NONE;
		return true;
	} else if(eff1 == CMD_OFFSET && eff2 == CMD_S3MCMDEX && param2 == 0x9F)
	{
		// Reverse offset
		eff1 = CMD_REVERSEOFFSET;
		eff2 = CMD_NONE;
		return true;
	} else if(eff1 == CMD_S3MCMDEX && param1 == 0x9F && eff2 == CMD_OFFSET)
	{
		// Reverse offset
		eff1 = CMD_REVERSEOFFSET;
		param1 = param2;
		eff2 = CMD_NONE;
		return true;
	} else
	{
		return false;
	}
}


bool ModCommand::TwoRegularCommandsToMPT(uint8 &effect1, uint8 &param1, uint8 &effect2, uint8 &param2)
{
	for(uint8 n = 0; n < 4; n++)
	{
		if(ModCommand::ConvertVolEffect(effect1, param1, (n > 1)))
		{
			return true;
		}
		std::swap(effect1, effect2);
		std::swap(param1, param2);
	}

	// Can only keep one command :(
	if(GetEffectWeight(static_cast<COMMAND>(effect1)) > GetEffectWeight(static_cast<COMMAND>(effect2)))
	{
		std::swap(effect1, effect2);
		std::swap(param1, param2);
	}
	effect1 = VOLCMD_NONE;
	param1 = 0;
	return false;
}


OPENMPT_NAMESPACE_END
