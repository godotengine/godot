/*
 * ModSequence.h
 * -------------
 * Purpose: Order and sequence handling.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include <algorithm>
#include <vector>
#include "Snd_defs.h"

OPENMPT_NAMESPACE_BEGIN

class CSoundFile;
class ModSequenceSet;

class ModSequence: public std::vector<PATTERNINDEX>
{
	friend class ModSequenceSet;

protected:
	std::string m_name;			// Sequence name.
	CSoundFile &m_sndFile;		// Associated CSoundFile.
	ORDERINDEX m_restartPos;	// Restart position when playback of this order ended

public:
	ModSequence(CSoundFile &sndFile);
	ModSequence(ModSequence &&) noexcept = default;
	ModSequence(const ModSequence &) = default;
	ModSequence& operator=(const ModSequence &other);

	bool operator==(const ModSequence &other) const;
	bool operator!=(const ModSequence &other) const { return !(*this == other); }

	ORDERINDEX GetLength() const { return mpt::saturate_cast<ORDERINDEX>(size()); }
	// Returns last accessible index, i.e. GetLength() - 1, or 0 if the order list is empty.
	ORDERINDEX GetLastIndex() const { return std::max(ORDERINDEX(1), GetLength()) - 1u; }
	// Returns length of sequence without counting trailing '---' items.
	ORDERINDEX GetLengthTailTrimmed() const;
	// Returns length of sequence stopping counting on first '---' (or at the end of sequence).
	ORDERINDEX GetLengthFirstEmpty() const;

	// Replaces order list with 'newSize' copies of 'pat'.
	void assign(ORDERINDEX newSize, PATTERNINDEX pat);

	// Inserts 'count' orders starting from 'pos' using 'fill' as the pattern index for all inserted orders.
	// Sequence will automatically grow if needed and if it can't grow enough, some tail orders will be discarded.
	// Return: Number of orders inserted (up to 'count' many).
	ORDERINDEX insert(ORDERINDEX pos, ORDERINDEX count) { return insert(pos, count, GetInvalidPatIndex()); }
	ORDERINDEX insert(ORDERINDEX pos, ORDERINDEX count, PATTERNINDEX fill);

	void push_back() { push_back(GetInvalidPatIndex()); }
	void push_back(PATTERNINDEX pat) { if(GetLength() < MAX_ORDERS) std::vector<PATTERNINDEX>::push_back(pat); }

	void resize(ORDERINDEX newSize) { resize(newSize, GetInvalidPatIndex()); }
	void resize(ORDERINDEX newSize, PATTERNINDEX pat) { std::vector<PATTERNINDEX>::resize(std::min(MAX_ORDERS, newSize), pat); }

	// Removes orders from range [posBegin, posEnd].
	void Remove(ORDERINDEX posBegin, ORDERINDEX posEnd);

	// Remove all references to a given pattern index from the order list. Jump commands are updated accordingly.
	void RemovePattern(PATTERNINDEX pat);

	// Replaces all occurences of oldPat with newPat.
	void Replace(PATTERNINDEX oldPat, PATTERNINDEX newPat) { if(oldPat != newPat) std::replace(begin(), end(), oldPat, newPat); }

	// Removes any "---" patterns at the end of the list.
	void Shrink() { resize(GetLengthTailTrimmed()); }

	// Check if pattern at sequence position ord is valid.
	bool IsValidPat(ORDERINDEX ord) const;

	void AdjustToNewModType(const MODTYPE oldtype);

	// Returns the internal representation of a stop '---' index
	static PATTERNINDEX GetInvalidPatIndex() { return uint16_max; }
	// Returns the internal representation of an ignore '+++' index
	static PATTERNINDEX GetIgnoreIndex() { return uint16_max - 1; }

	// Returns the previous/next order ignoring skip indices (+++).
	// If no previous/next order exists, return first/last order, and zero
	// when orderlist is empty.
	ORDERINDEX GetPreviousOrderIgnoringSkips(const ORDERINDEX start) const;
	ORDERINDEX GetNextOrderIgnoringSkips(const ORDERINDEX start) const;

	// Find an order item that contains a given pattern number.
	ORDERINDEX FindOrder(PATTERNINDEX pat, ORDERINDEX startSearchAt = 0, bool searchForward = true) const;

	// Ensures that the pattern at the specified order position is used only once (across all sequences).
	// If another usage is found, the pattern is replaced by a copy and the new index is returned.
	PATTERNINDEX EnsureUnique(ORDERINDEX ord);

	// Write order items as bytes. '---' is written as stopIndex, '+++' is written as ignoreIndex
	size_t WriteAsByte(FILE *f, const ORDERINDEX count, uint8 stopIndex = 0xFF, uint8 ignoreIndex = 0xFE) const;

	// Returns true if the IT orderlist datafield is not sufficient to store orderlist information.
	bool NeedsExtraDatafield() const;

#ifdef MODPLUG_TRACKER
	// Check if a playback position is currently locked (inaccessible)
	bool IsPositionLocked(ORDERINDEX position) const;
#endif // MODPLUG_TRACKER

	// Sequence name setter / getter
	inline void SetName(const std::string &newName) { m_name = newName;}
	inline std::string GetName() const { return m_name; }

	// Restart position setter / getter
	inline void SetRestartPos(ORDERINDEX restartPos) { m_restartPos = restartPos; }
	inline ORDERINDEX GetRestartPos() const { return m_restartPos; }
};


class ModSequenceSet
{
	friend void ReadModSequenceOld(std::istream& iStrm, ModSequenceSet& seq, const size_t);
	friend void ReadModSequences(std::istream& iStrm, ModSequenceSet& seq, const size_t);

protected:
	std::vector<ModSequence> m_Sequences;	// Array of sequences.
	CSoundFile &m_sndFile;
	SEQUENCEINDEX m_currentSeq;				// Index of current sequence.

public:
	ModSequenceSet(CSoundFile &sndFile);
	ModSequenceSet(ModSequenceSet &&) noexcept = default;

	// Remove all sequences and initialize default sequence
	void Initialize();

	// Get the working sequence
	ModSequence& operator() () { return m_Sequences[m_currentSeq]; }
	const ModSequence& operator() () const { return m_Sequences[m_currentSeq]; }
	// Get an arbitrary sequence
	ModSequence& operator() (SEQUENCEINDEX seq) { return m_Sequences[seq]; }
	const ModSequence& operator() (SEQUENCEINDEX seq) const { return m_Sequences[seq]; }

	SEQUENCEINDEX GetNumSequences() const { return static_cast<SEQUENCEINDEX>(m_Sequences.size()); }
	SEQUENCEINDEX GetCurrentSequenceIndex() const { return m_currentSeq; }

	// Sets working sequence.
	void SetSequence(SEQUENCEINDEX);
	// Add new sequence.
	// If duplicate is true, new sequence is a duplicate of the current sequence.
	// Returns the ID of the new sequence, or SEQUENCEINDEX_INVALID on failure.
	SEQUENCEINDEX AddSequence(bool duplicate = true);
	// Removes given sequence.
	void RemoveSequence(SEQUENCEINDEX);

	// Returns the internal representation of a stop '---' index
	static PATTERNINDEX GetInvalidPatIndex() { return ModSequence::GetInvalidPatIndex(); }
	// Returns the internal representation of an ignore '+++' index
	static PATTERNINDEX GetIgnoreIndex() { return ModSequence::GetIgnoreIndex(); }

#ifdef MODPLUG_TRACKER
	// Adjust sequence when converting between module formats
	void OnModTypeChanged(MODTYPE oldType);
	// If there are subsongs (separated by "---" patterns) in the module,
	// asks user whether to convert these into multiple sequences (given that the 
	// modformat supports multiple sequences).
	// Returns true if sequences were modified, false otherwise.
	bool ConvertSubsongsToMultipleSequences();

	// Convert the sequence's restart position information to a pattern command.
	bool RestartPosToPattern(SEQUENCEINDEX seq);
	// Merges multiple sequences into one and destroys all other sequences.
	// Returns false if there were no sequences to merge, true otherwise.
	bool MergeSequences();
#endif // MODPLUG_TRACKER

	std::vector<ModSequence>::iterator begin() { return m_Sequences.begin(); }
	std::vector<ModSequence>::const_iterator begin() const { return m_Sequences.begin(); }
	std::vector<ModSequence>::const_iterator cbegin() const { return m_Sequences.cbegin(); }
	std::vector<ModSequence>::iterator end() { return m_Sequences.end(); }
	std::vector<ModSequence>::const_iterator end() const { return m_Sequences.end(); }
	std::vector<ModSequence>::const_iterator cend() const { return m_Sequences.cend(); }
};


const char FileIdSequences[] = "mptSeqC";
const char FileIdSequence[] = "mptSeq";

void WriteModSequences(std::ostream& oStrm, const ModSequenceSet& seq);
void ReadModSequences(std::istream& iStrm, ModSequenceSet& seq, const size_t nSize = 0);

void WriteModSequence(std::ostream& oStrm, const ModSequence& seq);
void ReadModSequence(std::istream& iStrm, ModSequence& seq, const size_t);

void WriteModSequenceOld(std::ostream& oStrm, const ModSequenceSet& seq);
void ReadModSequenceOld(std::istream& iStrm, ModSequenceSet& seq, const size_t);


OPENMPT_NAMESPACE_END
