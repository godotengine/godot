/*
 * tuning.cpp
 * ----------
 * Purpose: Alternative sample tuning.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"

#include "tuning.h"
#include "../common/mptIO.h"
#include "../common/serialization_utils.h"
#include "../common/misc_util.h"
#include <string>
#include <cmath>


OPENMPT_NAMESPACE_BEGIN


namespace Tuning {


namespace CTuningS11n
{
	void ReadStr(std::istream& iStrm, std::string& str, const size_t);
	void ReadNoteMap(std::istream& iStrm, std::map<NOTEINDEXTYPE, std::string>& m, const size_t);
	void ReadRatioTable(std::istream& iStrm, std::vector<RATIOTYPE>& v, const size_t);

	void WriteNoteMap(std::ostream& oStrm, const std::map<NOTEINDEXTYPE, std::string>& m);
	void WriteStr(std::ostream& oStrm, const std::string& str);

	struct RatioWriter
	{
		RatioWriter(uint16 nWriteCount = s_nDefaultWriteCount) : m_nWriteCount(nWriteCount) {}

		void operator()(std::ostream& oStrm, const std::vector<float>& v);
		uint16 m_nWriteCount;
		static const uint16 s_nDefaultWriteCount = (uint16_max >> 2);
	};
}

using namespace CTuningS11n;


/*
Version changes:
	3->4: Finetune related internal structure and serialization revamp.
	2->3: The type for the size_type in the serialisation changed
		  from default(size_t, uint32) to unsigned STEPTYPE. (March 2007)
*/


CTuningRTI::CTuningRTI()
	: m_TuningType(TT_GENERAL)
	, m_FineStepCount(0)
{
	{
		m_RatioTable.clear();
		m_StepMin = s_StepMinDefault;
		m_RatioTable.resize(s_RatioTableSizeDefault, 1);
		m_GroupSize = 0;
		m_GroupRatio = 0;
		m_RatioTableFine.clear();
	}
}


bool CTuningRTI::ProCreateGroupGeometric(const std::vector<RATIOTYPE>& v, const RATIOTYPE& r, const VRPAIR& vr, const NOTEINDEXTYPE& ratiostartpos)
{
	if(v.size() == 0
		|| r <= 0
		|| vr.second < vr.first
		|| ratiostartpos < vr.first)
	{
		return true;
	}

	m_StepMin = vr.first;
	m_GroupSize = mpt::saturate_cast<NOTEINDEXTYPE>(v.size());
	m_GroupRatio = std::fabs(r);

	m_RatioTable.resize(vr.second-vr.first+1);
	std::copy(v.begin(), v.end(), m_RatioTable.begin() + (ratiostartpos - vr.first));

	for(int32 i = ratiostartpos-1; i>=m_StepMin && ratiostartpos > NOTEINDEXTYPE_MIN; i--)
	{
		m_RatioTable[i-m_StepMin] = m_RatioTable[i - m_StepMin + m_GroupSize] / m_GroupRatio;
	}
	for(int32 i = ratiostartpos+m_GroupSize; i<=vr.second && ratiostartpos <= (NOTEINDEXTYPE_MAX - m_GroupSize); i++)
	{
		m_RatioTable[i-m_StepMin] = m_GroupRatio * m_RatioTable[i - m_StepMin - m_GroupSize];
	}

	return false;
}


bool CTuningRTI::ProCreateGeometric(const UNOTEINDEXTYPE& s, const RATIOTYPE& r, const VRPAIR& vr)
{
	if(vr.second - vr.first + 1 > NOTEINDEXTYPE_MAX) return true;
	//Note: Setting finestep is handled by base class when CreateGeometric is called.
	{
		m_RatioTable.clear();
		m_StepMin = s_StepMinDefault;
		m_RatioTable.resize(s_RatioTableSizeDefault, static_cast<RATIOTYPE>(1.0));
		m_GroupSize = 0;
		m_GroupRatio = 0;
		m_RatioTableFine.clear();
	}
	m_StepMin = vr.first;
	
	m_GroupSize = mpt::saturate_cast<NOTEINDEXTYPE>(s);
	m_GroupRatio = std::fabs(r);
	const RATIOTYPE stepRatio = std::pow(m_GroupRatio, static_cast<RATIOTYPE>(1.0)/ static_cast<RATIOTYPE>(m_GroupSize));

	m_RatioTable.resize(vr.second - vr.first + 1);
	for(int32 i = vr.first; i<=vr.second; i++)
	{
		m_RatioTable[i-m_StepMin] = std::pow(stepRatio, static_cast<RATIOTYPE>(i));
	}
	return false;
}

std::string CTuningRTI::GetNoteName(const NOTEINDEXTYPE& x, bool addOctave) const
{
	if(!IsValidNote(x))
	{
		return std::string();
	}
	if(GetGroupSize() < 1)
	{
		const auto i = m_NoteNameMap.find(x);
		if(i != m_NoteNameMap.end())
			return i->second;
		else
			return mpt::fmt::val(x);
	}
	else
	{
		const NOTEINDEXTYPE pos = static_cast<NOTEINDEXTYPE>(mpt::wrapping_modulo(x, m_GroupSize));
		const NOTEINDEXTYPE middlePeriodNumber = 5;
		std::string rValue;
		const auto nmi = m_NoteNameMap.find(pos);
		if(nmi != m_NoteNameMap.end())
		{
			rValue = nmi->second;
			if(addOctave)
			{
				rValue += mpt::fmt::val(middlePeriodNumber + mpt::wrapping_divide(x, m_GroupSize));
			}
		}
		else
		{
			//By default, using notation nnP for notes; nn <-> note character starting
			//from 'A' with char ':' as fill char, and P is period integer. For example:
			//C:5, D:3, R:7
			if(m_GroupSize <= 26)
			{
				rValue = std::string(1, static_cast<char>(pos + 'A'));
				rValue += ":";
			} else
			{
				rValue = mpt::fmt::HEX0<1>(pos % 16) + mpt::fmt::HEX0<1>((pos / 16) % 16);
				if(pos > 0xff)
				{
					rValue = mpt::ToLowerCaseAscii(rValue);
				}
			}
			if(addOctave)
			{
				rValue += mpt::fmt::val(middlePeriodNumber + mpt::wrapping_divide(x, m_GroupSize));
			}
		}
		return rValue;
	}
}


const RATIOTYPE CTuningRTI::s_DefaultFallbackRatio = 1.0f;


//Without finetune
RATIOTYPE CTuningRTI::GetRatio(const NOTEINDEXTYPE& stepsFromCentre) const
{
	if(stepsFromCentre < m_StepMin) return s_DefaultFallbackRatio;
	if(stepsFromCentre >= m_StepMin + static_cast<NOTEINDEXTYPE>(m_RatioTable.size())) return s_DefaultFallbackRatio;
	return m_RatioTable[stepsFromCentre - m_StepMin];
}


//With finetune
RATIOTYPE CTuningRTI::GetRatio(const NOTEINDEXTYPE& baseNote, const STEPINDEXTYPE& baseStepDiff) const
{
	const STEPINDEXTYPE fsCount = static_cast<STEPINDEXTYPE>(GetFineStepCount());
	if(fsCount == 0 || baseStepDiff == 0)
	{
		return GetRatio(static_cast<NOTEINDEXTYPE>(baseNote + baseStepDiff));
	}

	//If baseStepDiff is more than the number of finesteps between notes,
	//note is increased. So first figuring out what step and fineStep values to
	//actually use. Interpreting finestep -1 on note x so that it is the same as
	//finestep GetFineStepCount() on note x-1.
	//Note: If finestepcount is n, n+1 steps are needed to get to
	//next note.
	NOTEINDEXTYPE note;
	STEPINDEXTYPE fineStep;
	note = static_cast<NOTEINDEXTYPE>(baseNote + mpt::wrapping_divide(baseStepDiff, (fsCount+1)));
	fineStep = mpt::wrapping_modulo(baseStepDiff, (fsCount+1));

	if(note < m_StepMin) return s_DefaultFallbackRatio;
	if(note >= m_StepMin + static_cast<NOTEINDEXTYPE>(m_RatioTable.size())) return s_DefaultFallbackRatio;

	if(fineStep) return m_RatioTable[note - m_StepMin] * GetRatioFine(note, fineStep);
	else return m_RatioTable[note - m_StepMin];
}


RATIOTYPE CTuningRTI::GetRatioFine(const NOTEINDEXTYPE& note, USTEPINDEXTYPE sd) const
{
	if(GetFineStepCount() <= 0)
		return 1;

	//Neither of these should happen.
	if(sd <= 0) sd = 1;
	if(sd > GetFineStepCount()) sd = GetFineStepCount();

	if(GetType() != TT_GENERAL && m_RatioTableFine.size() > 0) //Taking fineratio from table
	{
		if(GetType() == TT_GEOMETRIC)
		{
			return m_RatioTableFine[sd-1];
		}
		if(GetType() == TT_GROUPGEOMETRIC)
			return m_RatioTableFine[GetRefNote(note) * GetFineStepCount() + sd - 1];

		MPT_ASSERT_NOTREACHED();
		return m_RatioTableFine[0]; //Shouldn't happen.
	}
	else //Calculating ratio 'on the fly'.
	{
		//'Geometric finestepping'.
		return pow(GetRatio(note+1) / GetRatio(note), static_cast<RATIOTYPE>(sd)/(GetFineStepCount()+1));

	}

}


bool CTuningRTI::SetRatio(const NOTEINDEXTYPE& s, const RATIOTYPE& r)
{
	if(GetType() != TT_GROUPGEOMETRIC && GetType() != TT_GENERAL)
	{
		return false;
	}
	//Creating ratio table if doesn't exist.
	if(m_RatioTable.empty())
	{
		m_RatioTable.assign(s_RatioTableSizeDefault, 1);
		m_StepMin = s_StepMinDefault;
	}
	if(!IsNoteInTable(s))
	{
		return false;
	}
	m_RatioTable[s - m_StepMin] = std::fabs(r);
	if(GetType() == TT_GROUPGEOMETRIC)
	{ // update other groups
		for(NOTEINDEXTYPE n = m_StepMin; n < m_StepMin + static_cast<NOTEINDEXTYPE>(m_RatioTable.size()); ++n)
		{
			if(n == s)
			{
				// nothing
			} else if(mpt::abs(n - s) % m_GroupSize == 0)
			{
				m_RatioTable[n - m_StepMin] = std::pow(m_GroupRatio, static_cast<RATIOTYPE>(n - s) / static_cast<RATIOTYPE>(m_GroupSize)) * m_RatioTable[s - m_StepMin];
			}
		}
		UpdateFineStepTable();
	}
	return true;
}


void CTuningRTI::SetFineStepCount(const USTEPINDEXTYPE& fs)
{
	m_FineStepCount = mpt::clamp(mpt::saturate_cast<STEPINDEXTYPE>(fs), 0, FINESTEPCOUNT_MAX);
	UpdateFineStepTable();
}


void CTuningRTI::UpdateFineStepTable()
{
	if(m_FineStepCount <= 0)
	{
		m_RatioTableFine.clear();
		return;
	}
	if(GetType() == TT_GEOMETRIC)
	{
		if(m_FineStepCount > s_RatioTableFineSizeMaxDefault)
		{
			m_RatioTableFine.clear();
			return;
		}
		m_RatioTableFine.resize(m_FineStepCount);
		const RATIOTYPE q = GetRatio(GetValidityRange().first + 1) / GetRatio(GetValidityRange().first);
		const RATIOTYPE rFineStep = pow(q, static_cast<RATIOTYPE>(1)/(m_FineStepCount+1));
		for(USTEPINDEXTYPE i = 1; i<=m_FineStepCount; i++)
			m_RatioTableFine[i-1] = std::pow(rFineStep, static_cast<RATIOTYPE>(i));
		return;
	}
	if(GetType() == TT_GROUPGEOMETRIC)
	{
		const UNOTEINDEXTYPE p = GetGroupSize();
		if(p > s_RatioTableFineSizeMaxDefault / m_FineStepCount)
		{
			//In case fineratiotable would become too large, not using
			//table for it.
			m_RatioTableFine.clear();
			return;
		}
		else
		{
			//Creating 'geometric' finestepping between notes.
			m_RatioTableFine.resize(p * m_FineStepCount);
			const NOTEINDEXTYPE startnote = GetRefNote(GetValidityRange().first);
			for(UNOTEINDEXTYPE i = 0; i<p; i++)
			{
				const NOTEINDEXTYPE refnote = GetRefNote(startnote+i);
				const RATIOTYPE rFineStep = pow(GetRatio(refnote+1) / GetRatio(refnote), static_cast<RATIOTYPE>(1)/(m_FineStepCount+1));
				for(UNOTEINDEXTYPE j = 1; j<=m_FineStepCount; j++)
				{
					m_RatioTableFine[m_FineStepCount * refnote + (j-1)] = pow(rFineStep, static_cast<RATIOTYPE>(j));
				}
			}
			return;
		}

	}
	if(GetType() == TT_GENERAL)
	{
		//Not using table with tuning of type general.
		m_RatioTableFine.clear();
		return;
	}

	//Should not reach here.
	m_RatioTableFine.clear();
	m_FineStepCount = 0;
}


NOTEINDEXTYPE CTuningRTI::GetRefNote(const NOTEINDEXTYPE note) const
{
	if((GetType() != TT_GROUPGEOMETRIC) && (GetType() != TT_GEOMETRIC)) return 0;
	return static_cast<NOTEINDEXTYPE>(mpt::wrapping_modulo(note, GetGroupSize()));
}


SerializationResult CTuningRTI::InitDeserialize(std::istream& iStrm)
{
	// Note: OpenMPT since at least r323 writes version number (4<<24)+4 while it
	// reads version number (5<<24)+4 or earlier.
	// We keep this behaviour.

	if(iStrm.fail())
		return SerializationResult::Failure;

	srlztn::SsbRead ssb(iStrm);
	ssb.BeginRead("CTB244RTI", (5 << 24) + 4); // version
	ssb.ReadItem(m_TuningName, "0", ReadStr);
	uint16 dummyEditMask = 0xffff;
	ssb.ReadItem(dummyEditMask, "1");
	ssb.ReadItem(m_TuningType, "2");
	ssb.ReadItem(m_NoteNameMap, "3", ReadNoteMap);
	ssb.ReadItem(m_FineStepCount, "4");

	// RTI entries.
	ssb.ReadItem(m_RatioTable, "RTI0", ReadRatioTable);
	ssb.ReadItem(m_StepMin, "RTI1");
	ssb.ReadItem(m_GroupSize, "RTI2");
	ssb.ReadItem(m_GroupRatio, "RTI3");
	UNOTEINDEXTYPE ratiotableSize = 0;
	ssb.ReadItem(ratiotableSize, "RTI4");

	// If reader status is ok and m_StepMin is somewhat reasonable, process data.
	if(!((ssb.GetStatus() & srlztn::SNT_FAILURE) == 0 && m_StepMin >= -300 && m_StepMin <= 300))
	{
		return SerializationResult::Failure;
	}

	// reject unknown types
	if(m_TuningType != TT_GENERAL && m_TuningType != TT_GROUPGEOMETRIC && m_TuningType != TT_GEOMETRIC)
	{
		return SerializationResult::Failure;
	}
	if(m_GroupSize < 0)
	{
		return SerializationResult::Failure;
	}
	if(m_RatioTable.size() > static_cast<size_t>(NOTEINDEXTYPE_MAX))
	{
		return SerializationResult::Failure;
	}
	if((GetType() == TT_GROUPGEOMETRIC) || (GetType() == TT_GEOMETRIC))
	{
		if(ratiotableSize < 1 || ratiotableSize > NOTEINDEXTYPE_MAX)
		{
			return SerializationResult::Failure;
		}
		if(GetType() == TT_GEOMETRIC)
		{
			if(CreateGeometric(GetGroupSize(), GetGroupRatio(), VRPAIR(m_StepMin, static_cast<NOTEINDEXTYPE>(m_StepMin + ratiotableSize - 1))) != false)
			{
				return SerializationResult::Failure;
			}
		} else
		{
			if(CreateGroupGeometric(m_RatioTable, GetGroupRatio(), VRPAIR(m_StepMin, static_cast<NOTEINDEXTYPE>(m_StepMin+ratiotableSize-1)), m_StepMin) != false)
			{
				return SerializationResult::Failure;
			}
		}
	} else
	{
		UpdateFineStepTable();
	}
	return SerializationResult::Success;
}


template<class T, class SIZETYPE, class Tdst>
static bool VectorFromBinaryStream(std::istream& inStrm, std::vector<Tdst>& v, const SIZETYPE maxSize = (std::numeric_limits<SIZETYPE>::max)())
{
	if(!inStrm.good()) return true;

	SIZETYPE size = 0;
	mpt::IO::ReadIntLE<SIZETYPE>(inStrm, size);

	if(size > maxSize)
		return true;

	v.resize(size);
	for(std::size_t i = 0; i<size; i++)
	{
		T tmp = T();
		mpt::IO::Read(inStrm, tmp);
		v[i] = tmp;
	}
	if(inStrm.good())
		return false;
	else
		return true;
}


SerializationResult CTuningRTI::InitDeserializeOLD(std::istream& inStrm)
{
	if(!inStrm.good())
		return SerializationResult::Failure;

	const std::streamoff startPos = inStrm.tellg();

	//First checking is there expected begin sequence.
	char begin[8];
	MemsetZero(begin);
	inStrm.read(begin, sizeof(begin));
	if(std::memcmp(begin, "CTRTI_B.", 8))
	{
		//Returning stream position if beginmarker was not found.
		inStrm.seekg(startPos);
		return SerializationResult::Failure;
	}

	//Version
	int16 version = 0;
	mpt::IO::ReadIntLE<int16>(inStrm, version);
	if(version != 2 && version != 3)
		return SerializationResult::Failure;

	char begin2[8];
	MemsetZero(begin2);
	inStrm.read(begin2, sizeof(begin2));
	if(std::memcmp(begin2, "CT<sfs>B", 8))
	{
		return SerializationResult::Failure;
	}

	int16 version2 = 0;
	mpt::IO::ReadIntLE<int16>(inStrm, version2);
	if(version2 != 3 && version2 != 4)
	{
		return SerializationResult::Failure;
	}

	//Tuning name
	if(version2 <= 3)
	{
		if(!mpt::IO::ReadSizedStringLE<uint32>(inStrm, m_TuningName, 0xffff))
		{
			return SerializationResult::Failure;
		}
	} else
	{
		if(!mpt::IO::ReadSizedStringLE<uint8>(inStrm, m_TuningName))
		{
			return SerializationResult::Failure;
		}
	}

	//Const mask
	int16 em = 0;
	mpt::IO::ReadIntLE<int16>(inStrm, em);

	//Tuning type
	int16 tt = 0;
	mpt::IO::ReadIntLE<int16>(inStrm, tt);
	m_TuningType = tt;

	//Notemap
	uint16 size = 0;
	if(version2 <= 3)
	{
		uint32 tempsize = 0;
		mpt::IO::ReadIntLE<uint32>(inStrm, tempsize);
		if(tempsize > 0xffff)
		{
			return SerializationResult::Failure;
		}
		size = mpt::saturate_cast<uint16>(tempsize);
	} else
	{
		mpt::IO::ReadIntLE<uint16>(inStrm, size);
	}
	for(UNOTEINDEXTYPE i = 0; i<size; i++)
	{
		std::string str;
		int16 n = 0;
		mpt::IO::ReadIntLE<int16>(inStrm, n);
		if(version2 <= 3)
		{
			if(!mpt::IO::ReadSizedStringLE<uint32>(inStrm, str, 0xffff))
			{
				return SerializationResult::Failure;
			}
		} else
		{
			if(!mpt::IO::ReadSizedStringLE<uint8>(inStrm, str))
			{
				return SerializationResult::Failure;
			}
		}
		m_NoteNameMap[n] = str;
	}

	//End marker
	char end2[8];
	MemsetZero(end2);
	inStrm.read(end2, sizeof(end2));
	if(std::memcmp(end2, "CT<sfs>E", 8))
	{
		return SerializationResult::Failure;
	}

	// reject unknown types
	if(m_TuningType != TT_GENERAL && m_TuningType != TT_GROUPGEOMETRIC && m_TuningType != TT_GEOMETRIC)
	{
		return SerializationResult::Failure;
	}

	//Ratiotable
	if(version <= 2)
	{
		if(VectorFromBinaryStream<IEEE754binary32LE, uint32>(inStrm, m_RatioTable, 0xffff))
		{
			return SerializationResult::Failure;
		}
	} else
	{
		if(VectorFromBinaryStream<IEEE754binary32LE, uint16>(inStrm, m_RatioTable))
		{
			return SerializationResult::Failure;
		}
	}

	//Fineratios
	if(version <= 2)
	{
		if(VectorFromBinaryStream<IEEE754binary32LE, uint32>(inStrm, m_RatioTableFine, 0xffff))
		{
			return SerializationResult::Failure;
		}
	} else
	{
		if(VectorFromBinaryStream<IEEE754binary32LE, uint16>(inStrm, m_RatioTableFine))
		{
			return SerializationResult::Failure;
		}
	}
	m_FineStepCount = mpt::saturate_cast<USTEPINDEXTYPE>(m_RatioTableFine.size());

	//m_StepMin
	int16 stepmin = 0;
	mpt::IO::ReadIntLE<int16>(inStrm, stepmin);
	m_StepMin = stepmin;
	if(m_StepMin < -200 || m_StepMin > 200)
	{
		return SerializationResult::Failure;
	}

	//m_GroupSize
	int16 groupsize = 0;
	mpt::IO::ReadIntLE<int16>(inStrm, groupsize);
	m_GroupSize = groupsize;
	if(m_GroupSize < 0)
	{
		return SerializationResult::Failure;
	}

	//m_GroupRatio
	IEEE754binary32LE groupratio = IEEE754binary32LE(0.0f);
	mpt::IO::Read(inStrm, groupratio);
	m_GroupRatio = groupratio;
	if(m_GroupRatio < 0)
	{
		return SerializationResult::Failure;
	}

	char end[8];
	MemsetZero(end);
	inStrm.read(reinterpret_cast<char*>(&end), sizeof(end));
	if(std::memcmp(end, "CTRTI_E.", 8))
	{
		return SerializationResult::Failure;
	}

	// reject corrupt tunings
	if(m_RatioTable.size() > static_cast<std::size_t>(NOTEINDEXTYPE_MAX))
	{
		return SerializationResult::Failure;
	}
	if((m_GroupSize <= 0 || m_GroupRatio <= 0) && m_TuningType != TT_GENERAL)
	{
		return SerializationResult::Failure;
	}
	if(m_TuningType == TT_GROUPGEOMETRIC || m_TuningType == TT_GEOMETRIC)
	{
		if(m_RatioTable.size() < static_cast<std::size_t>(m_GroupSize))
		{
			return SerializationResult::Failure;
		}
	}

	// convert old finestepcount
	if(m_FineStepCount > 0)
	{
		m_FineStepCount -= 1;
	}
	UpdateFineStepTable();

	if(m_TuningType == TT_GEOMETRIC)
	{
		// Convert old geometric to new groupgeometric because old geometric tunings
		// can have ratio(0) != 1.0, which would get lost when saving nowadays.
		if(mpt::saturate_cast<NOTEINDEXTYPE>(m_RatioTable.size()) >= m_GroupSize - m_StepMin)
		{
			std::vector<RATIOTYPE> ratios;
			for(NOTEINDEXTYPE n = 0; n < m_GroupSize; ++n)
			{
				ratios.push_back(m_RatioTable[n - m_StepMin]);
			}
			CreateGroupGeometric(ratios, m_GroupRatio, GetValidityRange(), 0);
		}
	}

	return SerializationResult::Success;
}


Tuning::SerializationResult CTuningRTI::Serialize(std::ostream& outStrm) const
{
	// Note: OpenMPT since at least r323 writes version number (4<<24)+4 while it
	// reads version number (5<<24)+4.
	// We keep this behaviour.
	srlztn::SsbWrite ssb(outStrm);
	ssb.BeginWrite("CTB244RTI", (4 << 24) + 4); // version
	if (m_TuningName.length() > 0)
		ssb.WriteItem(m_TuningName, "0", WriteStr);
	uint16 dummyEditMask = 0xffff;
	ssb.WriteItem(dummyEditMask, "1");
	ssb.WriteItem(m_TuningType, "2");
	if (m_NoteNameMap.size() > 0)
		ssb.WriteItem(m_NoteNameMap, "3", WriteNoteMap);
	if (GetFineStepCount() > 0)
		ssb.WriteItem(m_FineStepCount, "4");

	const TUNINGTYPE tt = GetType();
	if (GetGroupRatio() > 0)
		ssb.WriteItem(m_GroupRatio, "RTI3");
	if (tt == TT_GROUPGEOMETRIC)
		ssb.WriteItem(m_RatioTable, "RTI0", RatioWriter(GetGroupSize()));
	if (tt == TT_GENERAL)
		ssb.WriteItem(m_RatioTable, "RTI0", RatioWriter());
	if (tt == TT_GEOMETRIC)
		ssb.WriteItem(m_GroupSize, "RTI2");

	if(tt == TT_GEOMETRIC || tt == TT_GROUPGEOMETRIC)
	{	//For Groupgeometric this data is the number of ratios in ratiotable.
		UNOTEINDEXTYPE ratiotableSize = static_cast<UNOTEINDEXTYPE>(m_RatioTable.size());
		ssb.WriteItem(ratiotableSize, "RTI4");
	}

	//m_StepMin
	ssb.WriteItem(m_StepMin, "RTI1");

	ssb.FinishWrite();

	return ((ssb.GetStatus() & srlztn::SNT_FAILURE) != 0) ? Tuning::SerializationResult::Failure : Tuning::SerializationResult::Success;
}


#ifdef MODPLUG_TRACKER

bool CTuningRTI::WriteSCL(std::ostream &f, const mpt::PathString &filename) const
{
	mpt::IO::WriteTextCRLF(f, mpt::format("! %1")(mpt::ToCharset(mpt::CharsetISO8859_1, (filename.GetFileName() + filename.GetFileExt()).ToUnicode())));
	mpt::IO::WriteTextCRLF(f, "!");
	std::string name = mpt::ToCharset(mpt::CharsetISO8859_1, mpt::CharsetLocale, GetName());
	for(auto & c : name) { if(static_cast<uint8>(c) < 32) c = ' '; } // remove control characters
	if(name.length() >= 1 && name[0] == '!') name[0] = '?'; // do not confuse description with comment
	mpt::IO::WriteTextCRLF(f, name);
	if(GetType() == TT_GEOMETRIC)
	{
		mpt::IO::WriteTextCRLF(f, mpt::format(" %1")(m_GroupSize));
		mpt::IO::WriteTextCRLF(f, "!");
		for(NOTEINDEXTYPE n = 0; n < m_GroupSize; ++n)
		{
			double ratio = std::pow(static_cast<double>(m_GroupRatio), static_cast<double>(n + 1) / static_cast<double>(m_GroupSize));
			double cents = std::log2(ratio) * 1200.0;
			mpt::IO::WriteTextCRLF(f, mpt::format(" %1 ! %2")(
				mpt::fmt::fix(cents),
				mpt::ToCharset(mpt::CharsetISO8859_1, mpt::CharsetLocale, GetNoteName((n + 1) % m_GroupSize, false))
				));
		}
	} else if(GetType() == TT_GROUPGEOMETRIC)
	{
		mpt::IO::WriteTextCRLF(f, mpt::format(" %1")(m_GroupSize));
		mpt::IO::WriteTextCRLF(f, "!");
		for(NOTEINDEXTYPE n = 0; n < m_GroupSize; ++n)
		{
			bool last = (n == (m_GroupSize - 1));
			double baseratio = static_cast<double>(GetRatio(0));
			double ratio = static_cast<double>(last ? m_GroupRatio : GetRatio(n + 1)) / baseratio;
			double cents = std::log2(ratio) * 1200.0;
			mpt::IO::WriteTextCRLF(f, mpt::format(" %1 ! %2")(
				mpt::fmt::fix(cents),
				mpt::ToCharset(mpt::CharsetISO8859_1, mpt::CharsetLocale, GetNoteName((n + 1) % m_GroupSize, false))
				));
		}
	} else if(GetType() == TT_GENERAL)
	{
		mpt::IO::WriteTextCRLF(f, mpt::format(" %1")(m_RatioTable.size() + 1));
		mpt::IO::WriteTextCRLF(f, "!");
		double baseratio = 1.0;
		for(NOTEINDEXTYPE n = 0; n < mpt::saturate_cast<NOTEINDEXTYPE>(m_RatioTable.size()); ++n)
		{
			baseratio = std::min(baseratio, static_cast<double>(m_RatioTable[n]));
		}
		for(NOTEINDEXTYPE n = 0; n < mpt::saturate_cast<NOTEINDEXTYPE>(m_RatioTable.size()); ++n)
		{
			double ratio = static_cast<double>(m_RatioTable[n]) / baseratio;
			double cents = std::log2(ratio) * 1200.0;
			mpt::IO::WriteTextCRLF(f, mpt::format(" %1 ! %2")(
				mpt::fmt::fix(cents),
				mpt::ToCharset(mpt::CharsetISO8859_1, mpt::CharsetLocale, GetNoteName(n + m_StepMin, false))
				));
		}
		mpt::IO::WriteTextCRLF(f, mpt::format(" %1 ! %2")(
			mpt::fmt::val(1),
			std::string()
			));
	} else
	{
		return false;
	}
	return true;
}

#endif


namespace CTuningS11n
{

void RatioWriter::operator()(std::ostream& oStrm, const std::vector<float>& v)
{
	const size_t nWriteCount = MIN(v.size(), m_nWriteCount);
	mpt::IO::WriteAdaptiveInt64LE(oStrm, nWriteCount);
	for(size_t i = 0; i < nWriteCount; i++)
		mpt::IO::Write(oStrm, IEEE754binary32LE(v[i]));
}


void ReadNoteMap(std::istream& iStrm, std::map<NOTEINDEXTYPE,std::string>& m, const size_t)
{
	uint64 val;
	mpt::IO::ReadAdaptiveInt64LE(iStrm, val);
	LimitMax(val, 256u); // Read 256 at max.
	for(size_t i = 0; i < val; i++)
	{
		int16 key;
		mpt::IO::ReadIntLE<int16>(iStrm, key);
		std::string str;
		mpt::IO::ReadSizedStringLE<uint8>(iStrm, str);
		m[key] = str;
	}
}


void ReadRatioTable(std::istream& iStrm, std::vector<RATIOTYPE>& v, const size_t)
{
	uint64 val;
	mpt::IO::ReadAdaptiveInt64LE(iStrm, val);
	v.resize( static_cast<size_t>(MIN(val, 256u))); // Read 256 vals at max.
	for(size_t i = 0; i < v.size(); i++)
	{
		IEEE754binary32LE tmp(0.0f);
		mpt::IO::Read(iStrm, tmp);
		v[i] = tmp;
	}
}


void ReadStr(std::istream& iStrm, std::string& str, const size_t)
{
	uint64 val;
	mpt::IO::ReadAdaptiveInt64LE(iStrm, val);
	size_t nSize = (val > 255) ? 255 : static_cast<size_t>(val); // Read 255 characters at max.
	str.clear();
	str.resize(nSize);
	for(size_t i = 0; i < nSize; i++)
		mpt::IO::ReadIntLE(iStrm, str[i]);
	if(str.find_first_of('\0') != std::string::npos)
	{ // trim \0 at the end
		str.resize(str.find_first_of('\0'));
	}
}


void WriteNoteMap(std::ostream& oStrm, const std::map<NOTEINDEXTYPE, std::string>& m)
{
	mpt::IO::WriteAdaptiveInt64LE(oStrm, m.size());
	for(auto &mi : m)
	{
		mpt::IO::WriteIntLE<int16>(oStrm, mi.first);
		mpt::IO::WriteSizedStringLE<uint8>(oStrm, mi.second);
	}
}


void WriteStr(std::ostream& oStrm, const std::string& str)
{
	mpt::IO::WriteAdaptiveInt64LE(oStrm, str.size());
	oStrm.write(str.c_str(), str.size());
}

} // namespace CTuningS11n.


} // namespace Tuning


OPENMPT_NAMESPACE_END
