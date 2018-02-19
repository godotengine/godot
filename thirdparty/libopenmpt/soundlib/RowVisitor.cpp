/*
 * RowVisitor.cpp
 * --------------
 * Purpose: Class for managing which rows of a song has already been visited. Useful for detecting backwards jumps, loops, etc.
 * Notes  : The class keeps track of rows that have been visited by the player before.
 *          This way, we can tell when the module starts to loop, i.e. we can determine the song length,
 *          or find out that a given point of the module can never be reached.
 *
 *          Specific implementations:
 *
 *          Length detection code:
 *          As the ModPlug engine already deals with pattern loops sufficiently (though not always correctly),
 *          there's no problem with (infinite) pattern loops in this code.
 *
 *          Normal player code:
 *          Bear in mind that rows inside pattern loops should only be evaluated once, or else the algorithm will cancel too early!
 *          So in that case, the pattern loop rows have to be reset when looping back.
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Sndfile.h"
#include "RowVisitor.h"

OPENMPT_NAMESPACE_BEGIN

RowVisitor::RowVisitor(const CSoundFile &sf, SEQUENCEINDEX sequence)
	: m_sndFile(sf)
	, m_currentOrder(0)
	, m_sequence(sequence)
{
	Initialize(true);
}


const ModSequence &RowVisitor::Order() const
{
	if(m_sequence >= m_sndFile.Order.GetNumSequences())
		return m_sndFile.Order();
	else
		return m_sndFile.Order(m_sequence);
}


// Resize / Clear the row vector.
// If reset is true, the vector is not only resized to the required dimensions, but also completely cleared (i.e. all visited rows are unset).
void RowVisitor::Initialize(bool reset)
{
	auto &order = Order();
	const ORDERINDEX endOrder = order.GetLengthTailTrimmed();
	m_visitedRows.resize(endOrder);
	if(reset)
	{
		m_visitOrder.clear();
		// Pre-allocate maximum amount of memory most likely needed for keeping track of visited rows in a pattern
		if(m_visitOrder.capacity() < MAX_PATTERN_ROWS)
		{
			ROWINDEX maxRows = 0;
			for(PATTERNINDEX pat = 0; pat < m_sndFile.Patterns.Size(); pat++)
			{
				maxRows = std::max(maxRows, m_sndFile.Patterns[pat].GetNumRows());
			}
			m_visitOrder.reserve(maxRows);
		}
	}

	for(ORDERINDEX ord = 0; ord < endOrder; ord++)
	{
		auto &row = m_visitedRows[ord];
		const size_t size = GetVisitedRowsVectorSize(order[ord]);
		if(reset)
		{
			// If we want to reset the vectors completely, we overwrite existing items with false.
			row.assign(size, false);
		} else
		{
			row.resize(size, false);
		}
	}
}


// (Un)sets a given row as visited.
// order, row - which row should be (un)set
// If visited is true, the row will be set as visited.
void RowVisitor::SetVisited(ORDERINDEX ord, ROWINDEX row, bool visited)
{
	auto &order = Order();
	if(ord >= order.size() || row >= GetVisitedRowsVectorSize(order[ord]))
	{
		return;
	}

	// The module might have been edited in the meantime - so we have to extend this a bit.
	if(ord >= m_visitedRows.size() || row >= m_visitedRows[ord].size())
	{
		Initialize(false);
		// If it's still past the end of the vector, this means that ord >= order.GetLengthTailTrimmed(), i.e. we are trying to play an empty order.
		if(ord >= m_visitedRows.size())
		{
			return;
		}
	}

	m_visitedRows[ord][row] = visited;
	if(visited)
	{
		AddVisitedRow(ord, row);
	}
}


// Returns whether a given row has been visited yet.
// If autoSet is true, the queried row will automatically be marked as visited.
// Use this parameter instead of consecutive IsRowVisited / SetRowVisited calls.
bool RowVisitor::IsVisited(ORDERINDEX ord, ROWINDEX row, bool autoSet)
{
	if(ord >= Order().size())
	{
		return false;
	}

	// The row slot for this row has not been assigned yet - Just return false, as this means that the program has not played the row yet.
	if(ord >= m_visitedRows.size() || row >= m_visitedRows[ord].size())
	{
		if(autoSet)
		{
			SetVisited(ord, row, true);
		}
		return false;
	}

	if(m_visitedRows[ord][row])
	{
		// We visited this row already - this module must be looping.
		return true;
	} else if(autoSet)
	{
		m_visitedRows[ord][row] = true;
		AddVisitedRow(ord, row);
	}

	return false;
}


// Get the needed vector size for pattern nPat.
size_t RowVisitor::GetVisitedRowsVectorSize(PATTERNINDEX pattern) const
{
	if(m_sndFile.Patterns.IsValidPat(pattern))
	{
		return static_cast<size_t>(m_sndFile.Patterns[pattern].GetNumRows());
	} else
	{
		// Invalid patterns consist of a "fake" row.
		return 1;
	}
}


// Find the first row that has not been played yet.
// The order and row is stored in the order and row variables on success, on failure they contain invalid values.
// If fastSearch is true (default), only the first row of each pattern is looked at, otherwise every row is examined.
// Function returns true on success.
bool RowVisitor::GetFirstUnvisitedRow(ORDERINDEX &ord, ROWINDEX &row, bool fastSearch) const
{
	auto &order = Order();
	const ORDERINDEX endOrder = order.GetLengthTailTrimmed();
	for(ord = 0; ord < endOrder; ord++)
	{
		const PATTERNINDEX pattern = order[ord];
		if(!m_sndFile.Patterns.IsValidPat(pattern))
		{
			continue;
		}

		if(ord >= m_visitedRows.size())
		{
			// Not yet initialized => unvisited
			return true;
		}

		const ROWINDEX endRow = (fastSearch ? 1 : m_sndFile.Patterns[pattern].GetNumRows());
		for(row = 0; row < endRow; row++)
		{
			if(row >= m_visitedRows[ord].size() || m_visitedRows[ord][row] == false)
			{
				// Not yet initialized, or unvisited
				return true;
			}
		}
	}

	// Didn't find anything :(
	ord = ORDERINDEX_INVALID;
	row = ROWINDEX_INVALID;
	return false;
}


// Set all rows of a previous pattern loop as unvisited.
void RowVisitor::ResetPatternLoop(ORDERINDEX ord, ROWINDEX startRow)
{
	MPT_ASSERT(ord == m_currentOrder);	// Shouldn't trigger, unless we're jumping around in the GUI during a pattern loop.

	// Unvisit all rows that are in the visited row buffer, until we hit the start row for this pattern loop.
	ROWINDEX row = ROWINDEX_INVALID;
	for(auto iter = m_visitOrder.crbegin(); iter != m_visitOrder.crend() && row != startRow; iter++)
	{
		row = *iter;
		Unvisit(ord, row);
	}
}


// Add a row to the visited row memory for this pattern.
void RowVisitor::AddVisitedRow(ORDERINDEX ord, ROWINDEX row)
{
	if(ord != m_currentOrder)
	{
		// We're in a new pattern! Forget about which rows we previously visited...
		m_visitOrder.clear();
		m_currentOrder = ord;
	}
	if(m_visitOrder.empty())
	{
		m_visitOrder.reserve(GetVisitedRowsVectorSize(Order()[ord]));
	}
	// And now add the played row to our memory.
	m_visitOrder.push_back(row);
}


OPENMPT_NAMESPACE_END
