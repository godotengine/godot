/*
 * Pattern.cpp
 * -----------
 * Purpose: Module Pattern header class
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "pattern.h"
#include "patternContainer.h"
#include "../common/serialization_utils.h"
#include "../common/version.h"
#include "ITTools.h"
#include "Sndfile.h"
#include "mod_specifications.h"


OPENMPT_NAMESPACE_BEGIN


CSoundFile& CPattern::GetSoundFile() { return m_rPatternContainer.GetSoundFile(); }
const CSoundFile& CPattern::GetSoundFile() const { return m_rPatternContainer.GetSoundFile(); }


CHANNELINDEX CPattern::GetNumChannels() const
{
	return GetSoundFile().GetNumChannels();
}


// Check if there is any note data on a given row.
bool CPattern::IsEmptyRow(ROWINDEX row) const
{
	if(m_ModCommands.empty() || !IsValidRow(row))
	{
		return true;
	}

	PatternRow data = GetRow(row);
	for(CHANNELINDEX chn = 0; chn < GetNumChannels(); chn++, data++)
	{
		if(!data->IsEmpty())
		{
			return false;
		}
	}
	return true;
}


bool CPattern::SetSignature(const ROWINDEX rowsPerBeat, const ROWINDEX rowsPerMeasure)
{
	if(rowsPerBeat < 1
		|| rowsPerBeat > GetSoundFile().GetModSpecifications().patternRowsMax
		|| rowsPerMeasure < rowsPerBeat
		|| rowsPerMeasure > GetSoundFile().GetModSpecifications().patternRowsMax)
	{
		return false;
	}
	m_RowsPerBeat = rowsPerBeat;
	m_RowsPerMeasure = rowsPerMeasure;
	return true;
}


// Add or remove rows from the pattern.
bool CPattern::Resize(const ROWINDEX newRowCount, bool enforceFormatLimits, bool resizeAtEnd)
{
	CSoundFile &sndFile = GetSoundFile();

	if(newRowCount == m_Rows || newRowCount < 1 || newRowCount > MAX_PATTERN_ROWS)
	{
		return false;
	}
	if(enforceFormatLimits)
	{
		auto &specs = sndFile.GetModSpecifications();
		if(newRowCount > specs.patternRowsMax || newRowCount < specs.patternRowsMin) return false;
	}

	try
	{
		size_t count = ((newRowCount > m_Rows) ? (newRowCount - m_Rows) : (m_Rows - newRowCount)) * GetNumChannels();

		if(newRowCount > m_Rows)
			m_ModCommands.insert(resizeAtEnd ? m_ModCommands.end() : m_ModCommands.begin(), count, ModCommand::Empty());
		else if(resizeAtEnd)
			m_ModCommands.erase(m_ModCommands.end() - count, m_ModCommands.end());
		else
			m_ModCommands.erase(m_ModCommands.begin(), m_ModCommands.begin() + count);
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
		return false;
	}

	m_Rows = newRowCount;
	return true;
}


void CPattern::ClearCommands()
{
	std::fill(m_ModCommands.begin(), m_ModCommands.end(), ModCommand::Empty());
}


bool CPattern::AllocatePattern(ROWINDEX rows)
{
	size_t newSize = GetNumChannels() * rows;
	if(rows == 0)
	{
		return false;
	} else if(rows == GetNumRows() && m_ModCommands.size() == newSize)
	{
		// Re-use allocated memory
		ClearCommands();
		return true;
	} else
	{
		// Do this in two steps in order to keep the old pattern data in case of OOM
		decltype(m_ModCommands) newPattern(newSize, ModCommand::Empty());
		m_ModCommands = std::move(newPattern);
	}
	m_Rows = rows;
	return true;
}


void CPattern::Deallocate()
{
	m_Rows = m_RowsPerBeat = m_RowsPerMeasure = 0;
	m_ModCommands.clear();
	m_PatternName.clear();
}


bool CPattern::operator== (const CPattern &other) const
{
	return GetNumRows() == other.GetNumRows()
		&& GetNumChannels() == other.GetNumChannels()
		&& GetOverrideSignature() == other.GetOverrideSignature()
		&& GetRowsPerBeat() == other.GetRowsPerBeat()
		&& GetRowsPerMeasure() == other.GetRowsPerMeasure()
		&& GetTempoSwing() == other.GetTempoSwing()
		&& m_ModCommands == other.m_ModCommands;
}


#ifdef MODPLUG_TRACKER

bool CPattern::Expand()
{
	const ROWINDEX newRows = m_Rows * 2;
	const CHANNELINDEX nChns = GetNumChannels();

	if(m_ModCommands.empty()
		|| newRows > GetSoundFile().GetModSpecifications().patternRowsMax)
	{
		return false;
	}

	decltype(m_ModCommands) newPattern;
	try
	{
		newPattern.assign(m_ModCommands.size() * 2, ModCommand::Empty());
	} MPT_EXCEPTION_CATCH_OUT_OF_MEMORY(e)
	{
		MPT_EXCEPTION_DELETE_OUT_OF_MEMORY(e);
		return false;
	}

	for(auto mSrc = m_ModCommands.begin(), mDst = newPattern.begin(); mSrc != m_ModCommands.end(); mSrc += nChns, mDst += 2 * nChns)
	{
		std::copy(mSrc, mSrc + nChns, mDst);
	}

	m_ModCommands = std::move(newPattern);
	m_Rows = newRows;

	return true;
}


bool CPattern::Shrink()
{
	if (m_ModCommands.empty()
		|| m_Rows < GetSoundFile().GetModSpecifications().patternRowsMin * 2)
	{
		return false;
	}

	m_Rows /= 2;
	const CHANNELINDEX nChns = GetNumChannels();

	for(ROWINDEX y = 0; y < m_Rows; y++)
	{
		const PatternRow srcRow = GetRow(y * 2);
		const PatternRow nextSrcRow = GetRow(y * 2 + 1);
		PatternRow destRow = GetRow(y);

		for(CHANNELINDEX x = 0; x < nChns; x++)
		{
			const ModCommand &src = srcRow[x];
			const ModCommand &srcNext = nextSrcRow[x];
			ModCommand &dest = destRow[x];
			dest = src;

			if(dest.note == NOTE_NONE && !dest.instr)
			{
				// Fill in data from next row if field is empty
				dest.note = srcNext.note;
				dest.instr = srcNext.instr;
				if(srcNext.volcmd != VOLCMD_NONE)
				{
					dest.volcmd = srcNext.volcmd;
					dest.vol = srcNext.vol;
				}
				if(dest.command == CMD_NONE)
				{
					dest.command = srcNext.command;
					dest.param = srcNext.param;
				}
			}
		}
	}
	m_ModCommands.resize(m_ModCommands.size() / 2);

	return true;
}


#endif // MODPLUG_TRACKER


bool CPattern::SetName(const std::string &newName)
{
	m_PatternName = newName;
	return true;
}


bool CPattern::SetName(const char *newName, size_t maxChars)
{
	if(newName == nullptr || maxChars == 0)
	{
		return false;
	}
	m_PatternName.assign(newName, strnlen(newName, maxChars));
	return true;
}


// Write some kind of effect data to the pattern. Exact data to be written and write behaviour can be found in the EffectWriter object.
bool CPattern::WriteEffect(EffectWriter &settings)
{
	// First, reject invalid parameters.
	if(m_ModCommands.empty()
		|| settings.m_row >= GetNumRows()
		|| (settings.m_channel >= GetNumChannels() && settings.m_channel != CHANNELINDEX_INVALID))
	{
		return false;
	}

	CHANNELINDEX scanChnMin = settings.m_channel, scanChnMax = settings.m_channel;

	// Scan all channels
	if(settings.m_channel == CHANNELINDEX_INVALID)
	{
		scanChnMin = 0;
		scanChnMax = GetNumChannels() - 1;
	}

	ModCommand * const baseCommand = GetpModCommand(settings.m_row, scanChnMin);
	ModCommand *m;

	// Scan channel(s) for same effect type - if an effect of the same type is already present, exit.
	if(!settings.m_allowMultiple)
	{
		m = baseCommand;
		for(CHANNELINDEX i = scanChnMin; i <= scanChnMax; i++, m++)
		{
			if(!settings.m_isVolEffect && m->command == settings.m_command)
				return true;
			if(settings.m_isVolEffect && m->volcmd == settings.m_volcmd)
				return true;
		}
	}

	// Easy case: check if there's some space left to put the effect somewhere
	m = baseCommand;
	for(CHANNELINDEX i = scanChnMin; i <= scanChnMax; i++, m++)
	{
		if(!settings.m_isVolEffect && m->command == CMD_NONE)
		{
			m->command = settings.m_command;
			m->param = settings.m_param;
			return true;
		}
		if(settings.m_isVolEffect && m->volcmd == VOLCMD_NONE)
		{
			m->volcmd = settings.m_volcmd;
			m->vol = settings.m_vol;
			return true;
		}
	}

	// Ok, apparently there's no space. If we haven't tried already, try to map it to the volume column or effect column instead.
	if(settings.m_retry)
	{
		const bool isS3M = (GetSoundFile().GetType() & MOD_TYPE_S3M);

		// Move some effects that also work in the volume column, so there's place for our new effect.
		if(!settings.m_isVolEffect)
		{
			m = baseCommand;
			for(CHANNELINDEX i = scanChnMin; i <= scanChnMax; i++, m++)
			{
				switch(m->command)
				{
				case CMD_VOLUME:
					if(!GetSoundFile().GetModSpecifications().HasVolCommand(VOLCMD_VOLUME))
					{
						break;
					}
					m->volcmd = VOLCMD_VOLUME;
					m->vol = m->param;
					m->command = settings.m_command;
					m->param = settings.m_param;
					return true;

				case CMD_PANNING8:
					if(isS3M && m->param > 0x80)
					{
						break;
					}

					m->volcmd = VOLCMD_PANNING;
					m->command = settings.m_command;

					if(isS3M)
						m->vol = (m->param + 1u) / 2u;
					else
						m->vol = (m->param + 2u) / 4u;

					m->param = settings.m_param;
					return true;

				default:
					break;
				}
			}
		}

		// Let's try it again by writing into the "other" effect column.
		if(settings.m_isVolEffect)
		{
			// Convert volume effect to normal effect
			ModCommand::COMMAND newCommand = CMD_NONE;
			ModCommand::PARAM newParam = settings.m_vol;
			switch(settings.m_volcmd)
			{
			case VOLCMD_PANNING:
				newCommand = CMD_PANNING8;
				newParam = mpt::saturate_cast<ModCommand::PARAM>(settings.m_vol * (isS3M ? 2u : 4u));
				break;
			case VOLCMD_VOLUME:
				newCommand = CMD_VOLUME;
				break;
			default:
				break;
			}

			if(newCommand != CMD_NONE)
			{
				settings.m_command = static_cast<EffectCommand>(newCommand);
				settings.m_param = newParam;
				settings.m_retry = false;
			}
		} else
		{
			// Convert normal effect to volume effect
			ModCommand::VOLCMD newVolCmd = VOLCMD_NONE;
			ModCommand::VOL newVol = settings.m_param;
			if(settings.m_command == CMD_PANNING8 && isS3M)
			{
				// This needs some manual fixing.
				if(settings.m_param <= 0x80)
				{
					// Can't have surround in volume column, only normal panning
					newVolCmd = VOLCMD_PANNING;
					newVol /= 2u;
				}
			} else
			{
				newVolCmd = settings.m_command;
				if(!ModCommand::ConvertVolEffect(newVolCmd, newVol, true))
				{
					// No Success :(
					newVolCmd = VOLCMD_NONE;
				}
			}

			if(newVolCmd != CMD_NONE)
			{
				settings.m_volcmd = static_cast<VolumeCommand>(newVolCmd);
				settings.m_vol = newVol;
				settings.m_retry = false;
			}
		}

		if(!settings.m_retry)
		{
			settings.m_isVolEffect = !settings.m_isVolEffect;
			if(WriteEffect(settings))
			{
				return true;
			}
		}
	}

	// Try in the next row if possible (this may also happen if we already retried)
	if(settings.m_retryMode == EffectWriter::rmTryNextRow && settings.m_row + 1 < GetNumRows())
	{
		settings.m_row++;
		settings.m_retry = true;
		return WriteEffect(settings);
	} else if(settings.m_retryMode == EffectWriter::rmTryPreviousRow && settings.m_row > 0)
	{
		settings.m_row--;
		settings.m_retry = true;
		return WriteEffect(settings);
	}

	return false;
}


////////////////////////////////////////////////////////////////////////
//
//	Pattern serialization functions
//
////////////////////////////////////////////////////////////////////////


enum maskbits
{
	noteBit			= (1 << 0),
	instrBit		= (1 << 1),
	volcmdBit		= (1 << 2),
	volBit			= (1 << 3),
	commandBit		= (1 << 4),
	effectParamBit	= (1 << 5),
	extraData		= (1 << 6)
};

void WriteData(std::ostream& oStrm, const CPattern& pat);
void ReadData(std::istream& iStrm, CPattern& pat, const size_t nSize = 0);

void WriteModPattern(std::ostream& oStrm, const CPattern& pat)
{
	srlztn::SsbWrite ssb(oStrm);
	ssb.BeginWrite(FileIdPattern, MptVersion::num);
	ssb.WriteItem(pat, "data", &WriteData);
	// pattern time signature
	if(pat.GetOverrideSignature())
	{
		ssb.WriteItem<uint32>(pat.GetRowsPerBeat(), "RPB.");
		ssb.WriteItem<uint32>(pat.GetRowsPerMeasure(), "RPM.");
	}
	if(pat.HasTempoSwing())
	{
		ssb.WriteItem<TempoSwing>(pat.GetTempoSwing(), "SWNG", TempoSwing::Serialize);
	}
	ssb.FinishWrite();
}


void ReadModPattern(std::istream& iStrm, CPattern& pat, const size_t)
{
	srlztn::SsbRead ssb(iStrm);
	ssb.BeginRead(FileIdPattern, MptVersion::num);
	if ((ssb.GetStatus() & srlztn::SNT_FAILURE) != 0)
		return;
	ssb.ReadItem(pat, "data", &ReadData);
	// pattern time signature
	uint32 nRPB = 0, nRPM = 0;
	ssb.ReadItem<uint32>(nRPB, "RPB.");
	ssb.ReadItem<uint32>(nRPM, "RPM.");
	pat.SetSignature(nRPB, nRPM);
	TempoSwing swing;
	ssb.ReadItem<TempoSwing>(swing, "SWNG", TempoSwing::Deserialize);
	if(!swing.empty()) swing.resize(nRPB);
	pat.SetTempoSwing(swing);
}


static uint8 CreateDiffMask(const ModCommand &chnMC, const ModCommand &newMC)
{
	uint8 mask = 0;
	if(chnMC.note != newMC.note)
		mask |= noteBit;
	if(chnMC.instr != newMC.instr)
		mask |= instrBit;
	if(chnMC.volcmd != newMC.volcmd)
		mask |= volcmdBit;
	if(chnMC.vol != newMC.vol)
		mask |= volBit;
	if(chnMC.command != newMC.command)
		mask |= commandBit;
	if(chnMC.param != newMC.param)
		mask |= effectParamBit;
	return mask;
}


// Writes pattern data. Adapted from SaveIT.
void WriteData(std::ostream& oStrm, const CPattern& pat)
{
	if(!pat.IsValid())
		return;

	const ROWINDEX rows = pat.GetNumRows();
	const CHANNELINDEX chns = pat.GetNumChannels();
	std::vector<ModCommand> lastChnMC(chns);

	for(ROWINDEX r = 0; r<rows; r++)
	{
		for(CHANNELINDEX c = 0; c<chns; c++)
		{
			const ModCommand m = *pat.GetpModCommand(r, c);
			// Writing only commands not written in IT-pattern writing:
			// For now this means only NOTE_PC and NOTE_PCS.
			if(!m.IsPcNote())
				continue;
			uint8 diffmask = CreateDiffMask(lastChnMC[c], m);
			uint8 chval = static_cast<uint8>(c+1);
			if(diffmask != 0)
				chval |= IT_bitmask_patternChanEnabled_c;

			mpt::IO::WriteIntLE<uint8>(oStrm, chval);

			if(diffmask)
			{
				lastChnMC[c] = m;
				mpt::IO::WriteIntLE<uint8>(oStrm, diffmask);
				if(diffmask & noteBit) mpt::IO::WriteIntLE<uint8>(oStrm, m.note);
				if(diffmask & instrBit) mpt::IO::WriteIntLE<uint8>(oStrm, m.instr);
				if(diffmask & volcmdBit) mpt::IO::WriteIntLE<uint8>(oStrm, m.volcmd);
				if(diffmask & volBit) mpt::IO::WriteIntLE<uint8>(oStrm, m.vol);
				if(diffmask & commandBit) mpt::IO::WriteIntLE<uint8>(oStrm, m.command);
				if(diffmask & effectParamBit) mpt::IO::WriteIntLE<uint8>(oStrm, m.param);
			}
		}
		mpt::IO::WriteIntLE<uint8>(oStrm, 0); // Write end of row marker.
	}
}


#define READITEM(itembit,id)		\
if(diffmask & itembit)				\
{									\
	mpt::IO::ReadIntLE<uint8>(iStrm, temp);	\
	if(ch < chns)					\
		lastChnMC[ch].id = temp;	\
}									\
if(ch < chns)						\
	m.id = lastChnMC[ch].id;


void ReadData(std::istream& iStrm, CPattern& pat, const size_t)
{
	if (!pat.IsValid()) // Expecting patterns to be allocated and resized properly.
		return;

	const CHANNELINDEX chns = pat.GetNumChannels();
	const ROWINDEX rows = pat.GetNumRows();

	std::vector<ModCommand> lastChnMC(chns);

	ROWINDEX row = 0;
	while(row < rows && iStrm.good())
	{
		uint8 t = 0;
		mpt::IO::ReadIntLE<uint8>(iStrm, t);
		if(t == 0)
		{
			row++;
			continue;
		}

		CHANNELINDEX ch = (t & IT_bitmask_patternChanField_c);
		if(ch > 0)
			ch--;

		uint8 diffmask = 0;
		if((t & IT_bitmask_patternChanEnabled_c) != 0)
			mpt::IO::ReadIntLE<uint8>(iStrm, diffmask);
		uint8 temp = 0;

		ModCommand dummy = ModCommand::Empty();
		ModCommand& m = (ch < chns) ? *pat.GetpModCommand(row, ch) : dummy;

		READITEM(noteBit, note);
		READITEM(instrBit, instr);
		READITEM(volcmdBit, volcmd);
		READITEM(volBit, vol);
		READITEM(commandBit, command);
		READITEM(effectParamBit, param);
		if(diffmask & extraData)
		{
			//Ignore additional data.
			uint8 size;
			mpt::IO::ReadIntLE<uint8>(iStrm, size);
			iStrm.ignore(size);
		}
	}
}

#undef READITEM


OPENMPT_NAMESPACE_END
