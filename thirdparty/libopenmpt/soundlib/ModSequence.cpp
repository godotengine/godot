/*
 * ModSequence.cpp
 * ---------------
 * Purpose: Order and sequence handling.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "ModSequence.h"
#include "Sndfile.h"
#include "mod_specifications.h"
#ifdef MODPLUG_TRACKER
#include "../mptrack/Reporting.h"
#endif // MODPLUG_TRACKER
#include "../common/version.h"
#include "../common/serialization_utils.h"

OPENMPT_NAMESPACE_BEGIN

#define str_SequenceTruncationNote ("Module has sequence of length %1; it will be truncated to maximum supported length, %2.")


ModSequence::ModSequence(CSoundFile &sndFile)
	: m_sndFile(sndFile)
	, m_restartPos(0)
{
}


ModSequence& ModSequence::operator=(const ModSequence &other)
{
	if(&other == this)
		return *this;
	std::vector<PATTERNINDEX>::assign(other.begin(), other.end());
	m_name = other.m_name;
	m_restartPos = other.m_restartPos;
	return *this;
}


bool ModSequence::operator== (const ModSequence &other) const
{
	return static_cast<const std::vector<PATTERNINDEX> &>(*this) == other
		&& m_name == other.m_name
		&& m_restartPos == other.m_restartPos;
}


bool ModSequence::NeedsExtraDatafield() const
{
	return (m_sndFile.GetType() == MOD_TYPE_MPT && m_sndFile.Patterns.GetNumPatterns() > 0xFD);
}


void ModSequence::AdjustToNewModType(const MODTYPE oldtype)
{
	auto &specs = m_sndFile.GetModSpecifications();

	if(oldtype != MOD_TYPE_NONE)
	{
		// If not supported, remove "+++" separator order items.
		if(!specs.hasIgnoreIndex)
		{
			RemovePattern(GetIgnoreIndex());
		}
		// If not supported, remove "---" items between patterns.
		if(!specs.hasStopIndex)
		{
			RemovePattern(GetInvalidPatIndex());
		}
	}

	//Resize orderlist if needed.
	if(specs.ordersMax < size())
	{
		// Order list too long? Remove "unnecessary" order items first.
		if(oldtype != MOD_TYPE_NONE && specs.ordersMax < GetLengthTailTrimmed())
		{
			erase(std::remove_if(begin(), end(), [&] (PATTERNINDEX pat) { return !m_sndFile.Patterns.IsValidPat(pat); }), end());
			if(GetLengthTailTrimmed() > specs.ordersMax)
			{
				m_sndFile.AddToLog("WARNING: Order list has been trimmed!");
			}
		}
		resize(specs.ordersMax);
	}
}


ORDERINDEX ModSequence::GetLengthTailTrimmed() const
{
	if(empty())
		return 0;
	auto last = std::find_if(rbegin(), rend(), [] (PATTERNINDEX pat) { return pat != GetInvalidPatIndex(); });
	return static_cast<ORDERINDEX>(std::distance(begin(), last.base()));
}


ORDERINDEX ModSequence::GetLengthFirstEmpty() const
{
	return static_cast<ORDERINDEX>(std::distance(begin(), std::find(begin(), end(), GetInvalidPatIndex())));
}


ORDERINDEX ModSequence::GetNextOrderIgnoringSkips(const ORDERINDEX start) const
{
	if(empty())
		return 0;
	auto length = GetLength();
	ORDERINDEX next = std::min(ORDERINDEX(length - 1), ORDERINDEX(start + 1));
	while(next + 1 < length && at(next) == GetIgnoreIndex()) next++;
	return next;
}


ORDERINDEX ModSequence::GetPreviousOrderIgnoringSkips(const ORDERINDEX start) const
{
	const ORDERINDEX last = GetLastIndex();
	if(start == 0 || last == 0) return 0;
	ORDERINDEX prev = std::min(ORDERINDEX(start - 1), last);
	while(prev > 0 && at(prev) == GetIgnoreIndex()) prev--;
	return prev;
}


void ModSequence::Remove(ORDERINDEX posBegin, ORDERINDEX posEnd)
{
	if(posEnd < posBegin || posEnd >= size())
		return;
	erase(begin() + posBegin, begin() + posEnd + 1);
}


// Remove all references to a given pattern index from the order list. Jump commands are updated accordingly.
void ModSequence::RemovePattern(PATTERNINDEX pat)
{
	// First, calculate the offset that needs to be applied to jump commands
	const ORDERINDEX orderLength = GetLengthTailTrimmed();
	std::vector<ORDERINDEX> jumpOffset(orderLength, 0);
	ORDERINDEX maxJump = 0;
	for(ORDERINDEX i = 0; i < orderLength; i++)
	{
		jumpOffset[i] = i - maxJump;
		if(at(i) == pat)
		{
			maxJump++;
		}
	}
	if(!maxJump)
	{
		return;
	}

	erase(std::remove_if(begin(), end(), [pat](PATTERNINDEX p) { return p == pat; }), end());

	// Only apply to patterns actually found in this sequence
	for(auto p : *this) if(m_sndFile.Patterns.IsValidPat(p))
	{
		for(auto &m : m_sndFile.Patterns[p])
		{
			if(m.command == CMD_POSITIONJUMP && m.param < jumpOffset.size())
			{
				m.param = static_cast<ModCommand::PARAM>(jumpOffset[m.param]);
			}
		}
	}
	if(m_restartPos < jumpOffset.size())
	{
		m_restartPos = jumpOffset[m_restartPos];
	}
}


void ModSequence::assign(ORDERINDEX newSize, PATTERNINDEX pat)
{
	LimitMax(newSize, m_sndFile.GetModSpecifications().ordersMax);
	std::vector<PATTERNINDEX>::assign(newSize, pat);
}


ORDERINDEX ModSequence::insert(ORDERINDEX pos, ORDERINDEX count, PATTERNINDEX fill)
{
	if (pos >= m_sndFile.GetModSpecifications().ordersMax || count == 0)
		return 0;
	// Limit number of orders to be inserted so that we don't exceed the format limit.
	LimitMax(count, ORDERINDEX(m_sndFile.GetModSpecifications().ordersMax - pos));
	reserve(pos + count);
	// Inserting past the end of the container?
	if(pos > size())
		resize(pos);
	std::vector<PATTERNINDEX>::insert(begin() + pos, count, fill);
	// Did we overgrow? Remove patterns at end.
	if(size() > m_sndFile.GetModSpecifications().ordersMax)
		resize(m_sndFile.GetModSpecifications().ordersMax);
	return count;
}


bool ModSequence::IsValidPat(ORDERINDEX ord) const
{
	if(ord < size())
		return m_sndFile.Patterns.IsValidPat(at(ord));
	return false;
}


ORDERINDEX ModSequence::FindOrder(PATTERNINDEX pat, ORDERINDEX startSearchAt, bool searchForward) const
{
	const ORDERINDEX length = GetLength();
	ORDERINDEX ord = startSearchAt;
	for(ORDERINDEX p = 0; p < length; p++)
	{
		if(at(ord) == pat)
		{
			return ord;
		}
		if(searchForward)
		{
			if(++ord >= length)
				ord = 0;
		} else
		{
			if(ord-- == 0)
				ord = length - 1;
		}
	}
	return ORDERINDEX_INVALID;
}


PATTERNINDEX ModSequence::EnsureUnique(ORDERINDEX ord)
{
	PATTERNINDEX pat = at(ord);
	for(const auto &sequence : m_sndFile.Order)
	{
		ORDERINDEX ords = sequence.GetLength();
		for(ORDERINDEX o = 0; o < ords; o++)
		{
			if(sequence[o] == pat && (o != ord || &sequence != this))
			{
				// Found duplicate usage.
				PATTERNINDEX newPat = m_sndFile.Patterns.Duplicate(pat);
				if(newPat != PATTERNINDEX_INVALID)
				{
					at(ord) = newPat;
					return newPat;
				}
			}
		}
	}
	return pat;
}


/////////////////////////////////////
// ModSequenceSet
/////////////////////////////////////


ModSequenceSet::ModSequenceSet(CSoundFile &sndFile)
	: m_sndFile(sndFile)
{
	Initialize();
}


void ModSequenceSet::Initialize()
{
	m_currentSeq = 0;
	m_Sequences.assign(1, ModSequence(m_sndFile));
}


void ModSequenceSet::SetSequence(SEQUENCEINDEX n)
{
	if(n < m_Sequences.size())
		m_currentSeq = n;
}


SEQUENCEINDEX ModSequenceSet::AddSequence(bool duplicate)
{
	if(GetNumSequences() == MAX_SEQUENCES)
		return SEQUENCEINDEX_INVALID;
	if(duplicate)
	{
		m_Sequences.push_back(m_Sequences[m_currentSeq]);
		m_Sequences.back().m_name.clear();	// Don't copy sequence name.
	} else
	{
		m_Sequences.push_back(ModSequence(m_sndFile));
	}
	SetSequence(GetNumSequences() - 1);
	return GetNumSequences() - 1;
}


void ModSequenceSet::RemoveSequence(SEQUENCEINDEX i)
{
	// Do nothing if index is invalid or if there's only one sequence left.
	if(i >= m_Sequences.size() || m_Sequences.size() <= 1)
		return;
	m_Sequences.erase(m_Sequences.begin() + i);
	if(i < m_currentSeq || m_currentSeq >= GetNumSequences())
		m_currentSeq--;
}


#ifdef MODPLUG_TRACKER

void ModSequenceSet::OnModTypeChanged(MODTYPE oldType)
{
	for(auto &seq : m_Sequences)
	{
		seq.AdjustToNewModType(oldType);
	}
	// Multisequences not suppported by other formats
	if(oldType != MOD_TYPE_NONE && m_sndFile.GetModSpecifications().sequencesMax <= 1)
		MergeSequences();

	// Convert sequence with separator patterns into multiple sequences?
	if(oldType != MOD_TYPE_NONE && m_sndFile.GetModSpecifications().sequencesMax > 1 && GetNumSequences() == 1)
		ConvertSubsongsToMultipleSequences();
}


bool ModSequenceSet::ConvertSubsongsToMultipleSequences()
{
	// Allow conversion only if there's only one sequence.
	if(GetNumSequences() != 1 || m_sndFile.GetModSpecifications().sequencesMax <= 1)
		return false;

	m_Sequences[0].Shrink();
	bool hasSepPatterns = std::find_if(m_Sequences[0].begin(), m_Sequences[0].end(),
		[&] (PATTERNINDEX pat) { return pat != GetIgnoreIndex() && !m_sndFile.Patterns.IsValidPat(pat); }) != m_Sequences[0].end();
	bool modified = false;

	if(hasSepPatterns &&
		Reporting::Confirm("The order list contains separator items.\nThe new format supports multiple sequences, do you want to convert those separate tracks into multiple song sequences?",
		"Order list conversion", false, true) == cnfYes)
	{
		ORDERINDEX length = m_Sequences[0].GetLengthTailTrimmed();
		for(ORDERINDEX ord = 0; ord < length; ord++)
		{
			// End of subsong?
			if(!m_Sequences[0].IsValidPat(ord) && m_Sequences[0][ord] != GetIgnoreIndex())
			{
				// remove all separator patterns between current and next subsong first
				while(ord < length && !m_sndFile.Patterns.IsValidIndex(m_Sequences[0][ord]))
				{
					m_Sequences[0][ord] = GetInvalidPatIndex();
					ord++;
					modified = true;
				}
				if(ord >= length) break;
				ORDERINDEX startOrd = ord;
				modified = true;

				SEQUENCEINDEX newSeq = AddSequence(false);
				m_Sequences[newSeq].reserve(length - startOrd);

				// now, move all following orders to the new sequence
				while(ord < length && m_Sequences[0][ord] != GetInvalidPatIndex())
				{
					PATTERNINDEX copyPat = m_Sequences[0][ord];
					m_Sequences[newSeq].push_back(copyPat);
					m_Sequences[0][ord] = GetInvalidPatIndex();
					ord++;

					// is this a valid pattern? adjust pattern jump commands, if necessary.
					if(m_sndFile.Patterns.IsValidPat(copyPat))
					{
						for(auto &m : m_sndFile.Patterns[copyPat])
						{
							if(m.command == CMD_POSITIONJUMP && m.param >= startOrd)
							{
								m.param = static_cast<ModCommand::PARAM>(m.param - startOrd);
							}
						}
					}
				}
			}
		}
		SetSequence(0);
	}
	return modified;
}


// Convert the sequence's restart position information to a pattern command.
bool ModSequenceSet::RestartPosToPattern(SEQUENCEINDEX seq)
{
	bool result = false;
	auto length = m_sndFile.GetLength(eNoAdjust, GetLengthTarget(true).StartPos(seq, 0, 0));
	ModSequence &order = m_Sequences[seq];
	for(const auto &subSong : length)
	{
		if(subSong.endOrder != ORDERINDEX_INVALID && subSong.endRow != ROWINDEX_INVALID)
		{
			if(Util::TypeCanHoldValue<ModCommand::PARAM>(order.GetRestartPos()))
			{
				PATTERNINDEX writePat = order.EnsureUnique(subSong.endOrder);
				result = m_sndFile.Patterns[writePat].WriteEffect(
					EffectWriter(CMD_POSITIONJUMP, static_cast<ModCommand::PARAM>(order.GetRestartPos())).Row(subSong.endRow).RetryNextRow());
			} else
			{
				result = false;
			}
		}
	}
	order.SetRestartPos(0);
	return result;
}


bool ModSequenceSet::MergeSequences()
{
	if(GetNumSequences() <= 1)
		return false;

	ModSequence &firstSeq = m_Sequences[0];
	firstSeq.resize(firstSeq.GetLengthTailTrimmed());
	std::vector<SEQUENCEINDEX> patternsFixed(m_sndFile.Patterns.Size(), SEQUENCEINDEX_INVALID); // pattern fixed by other sequence already?
	// Mark patterns handled in first sequence
	for(auto pat : firstSeq)
	{
		if(m_sndFile.Patterns.IsValidPat(pat))
			patternsFixed[pat] = 0;
	}

	for(SEQUENCEINDEX seqNum = 1; seqNum < GetNumSequences(); seqNum++)
	{
		ModSequence &seq = m_Sequences[seqNum];
		const ORDERINDEX firstOrder = firstSeq.GetLength() + 1; // +1 for separator item
		const ORDERINDEX lengthTrimmed = seq.GetLengthTailTrimmed();
		if(firstOrder + lengthTrimmed > m_sndFile.GetModSpecifications().ordersMax)
		{
			m_sndFile.AddToLog(mpt::format("WARNING: Cannot merge Sequence %1 (too long!)")(seqNum));
			continue;
		}
		firstSeq.reserve(firstOrder + lengthTrimmed);
		firstSeq.push_back(); // Separator item
		RestartPosToPattern(seqNum);
		for(ORDERINDEX ord = 0; ord < lengthTrimmed; ord++)
		{
			PATTERNINDEX pat = seq[ord];
			firstSeq.push_back(pat);

			// Try to fix pattern jump commands
			if(!m_sndFile.Patterns.IsValidPat(pat)) continue;

			auto m = m_sndFile.Patterns[pat].begin();
			for(size_t len = 0; len < m_sndFile.Patterns[pat].GetNumRows() * m_sndFile.m_nChannels; m++, len++)
			{
				if(m->command == CMD_POSITIONJUMP)
				{
					if(patternsFixed[pat] != SEQUENCEINDEX_INVALID && patternsFixed[pat] != seqNum)
					{
						// Oops, some other sequence uses this pattern already.
						const PATTERNINDEX newPat = m_sndFile.Patterns.Duplicate(pat, true);
						if(newPat != PATTERNINDEX_INVALID)
						{
							// Could create new pattern - copy data over and continue from here.
							firstSeq[firstOrder + ord] = newPat;
							m = m_sndFile.Patterns[newPat].begin() + len;
							if(newPat >= patternsFixed.size())
								patternsFixed.resize(newPat + 1, SEQUENCEINDEX_INVALID);
							pat = newPat;
						} else
						{
							// Cannot create new pattern: notify the user
							m_sndFile.AddToLog(mpt::format("CONFLICT: Pattern break commands in Pattern %1 might be broken since it has been used in several sequences!")(pat));
						}
					}
					m->param = static_cast<ModCommand::PARAM>(m->param + firstOrder);
					patternsFixed[pat] = seqNum;
				}
			}
		}
	}
	m_Sequences.erase(m_Sequences.begin() + 1, m_Sequences.end());
	return true;
}


// Check if a playback position is currently locked (inaccessible)
bool ModSequence::IsPositionLocked(ORDERINDEX position) const
{
	return(m_sndFile.m_lockOrderStart != ORDERINDEX_INVALID
		&& (position < m_sndFile.m_lockOrderStart || position > m_sndFile.m_lockOrderEnd));
}
#endif // MODPLUG_TRACKER


/////////////////////////////////////
// Read/Write
/////////////////////////////////////


size_t ModSequence::WriteAsByte(FILE *f, const ORDERINDEX count, uint8 stopIndex, uint8 ignoreIndex) const
{
	const size_t limit = std::min(count, GetLength());

	for(size_t i = 0; i < limit; i++)
	{
		const PATTERNINDEX pat = at(i);
		uint8 temp = static_cast<uint8>(pat);

		if(pat == GetInvalidPatIndex()) temp = stopIndex;
		else if(pat == GetIgnoreIndex() || pat > 0xFF) temp = ignoreIndex;
		fwrite(&temp, 1, 1, f);
	}
	// Fill non-existing order items with stop indices
	for(size_t i = limit; i < count; i++)
	{
		fwrite(&stopIndex, 1, 1, f);
	}
	return count; //Returns the number of bytes written.
}


void ReadModSequenceOld(std::istream& iStrm, ModSequenceSet& seq, const size_t)
{
	uint16 size;
	mpt::IO::ReadIntLE<uint16>(iStrm, size);
	if(size > ModSpecs::mptm.ordersMax)
	{
		seq.m_sndFile.AddToLog(mpt::format(str_SequenceTruncationNote)(size, ModSpecs::mptm.ordersMax));
		size = ModSpecs::mptm.ordersMax;
	}
	seq(0).resize(size);
	for(auto &pat : seq(0))
	{
		uint16 temp;
		mpt::IO::ReadIntLE<uint16>(iStrm, temp);
		pat = temp;
	}
}


void WriteModSequenceOld(std::ostream& oStrm, const ModSequenceSet& seq)
{
	const uint16 size = seq().GetLength();
	mpt::IO::WriteIntLE<uint16>(oStrm, size);
	for(auto pat : seq())
	{
		mpt::IO::WriteIntLE<uint16>(oStrm, static_cast<uint16>(pat));
	}
}


void WriteModSequence(std::ostream& oStrm, const ModSequence& seq)
{
	srlztn::SsbWrite ssb(oStrm);
	ssb.BeginWrite(FileIdSequence, MptVersion::num);
	ssb.WriteItem(seq.GetName(), "n");
	const uint16 length = seq.GetLengthTailTrimmed();
	ssb.WriteItem<uint16>(length, "l");
	ssb.WriteItem(seq, "a", srlztn::VectorWriter<uint16>(length));
	if(seq.GetRestartPos() > 0)
		ssb.WriteItem<uint16>(seq.GetRestartPos(), "r");
	ssb.FinishWrite();
}


void ReadModSequence(std::istream& iStrm, ModSequence& seq, const size_t)
{
	srlztn::SsbRead ssb(iStrm);
	ssb.BeginRead(FileIdSequence, MptVersion::num);
	if ((ssb.GetStatus() & srlztn::SNT_FAILURE) != 0)
		return;
	std::string str;
	ssb.ReadItem(str, "n");
	seq.SetName(str);
	ORDERINDEX nSize = 0;
	ssb.ReadItem(nSize, "l");
	LimitMax(nSize, ModSpecs::mptm.ordersMax);
	ssb.ReadItem(seq, "a", srlztn::VectorReader<uint16>(nSize));

	ORDERINDEX restartPos = ORDERINDEX_INVALID;
	if(ssb.ReadItem(restartPos, "r") != srlztn::SsbRead::EntryNotFound && restartPos < nSize)
		seq.SetRestartPos(restartPos);
}


void WriteModSequences(std::ostream& oStrm, const ModSequenceSet& seq)
{
	srlztn::SsbWrite ssb(oStrm);
	ssb.BeginWrite(FileIdSequences, MptVersion::num);
	const uint8 nSeqs = seq.GetNumSequences();
	const uint8 nCurrent = seq.GetCurrentSequenceIndex();
	ssb.WriteItem(nSeqs, "n");
	ssb.WriteItem(nCurrent, "c");
	for(uint8 i = 0; i < nSeqs; i++)
	{
		ssb.WriteItem(seq(i), srlztn::ID::FromInt<uint8>(i), &WriteModSequence);
	}
	ssb.FinishWrite();
}


void ReadModSequences(std::istream& iStrm, ModSequenceSet& seq, const size_t)
{
	srlztn::SsbRead ssb(iStrm);
	ssb.BeginRead(FileIdSequences, MptVersion::num);
	if ((ssb.GetStatus() & srlztn::SNT_FAILURE) != 0)
		return;
	SEQUENCEINDEX seqs = 0;
	uint8 currentSeq = 0;
	ssb.ReadItem(seqs, "n");
	if (seqs == 0)
		return;
	LimitMax(seqs, MAX_SEQUENCES);
	ssb.ReadItem(currentSeq, "c");
	if (seq.GetNumSequences() < seqs)
		seq.m_Sequences.resize(seqs, ModSequence(seq.m_sndFile));

	// There used to be only one restart position for all sequences
	ORDERINDEX legacyRestartPos = seq(0).GetRestartPos();

	for(SEQUENCEINDEX i = 0; i < seqs; i++)
	{
		seq(i).SetRestartPos(legacyRestartPos);
		ssb.ReadItem(seq(i), srlztn::ID::FromInt<uint8>(i), &ReadModSequence);
	}
	seq.m_currentSeq = (currentSeq < seq.GetNumSequences()) ? currentSeq : 0;
}


OPENMPT_NAMESPACE_END
