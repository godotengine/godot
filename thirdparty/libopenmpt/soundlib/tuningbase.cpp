/*
 * tuningbase.cpp
 * --------------
 * Purpose: Alternative sample tuning.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "tuningbase.h"
#include "tuning.h"
#include "../common/mptIO.h"
#include "../common/serialization_utils.h"

#include <cmath>

#include <istream>
#include <ostream>


OPENMPT_NAMESPACE_BEGIN


namespace Tuning {


/*
Version history:
	4->5: Lots of changes, finestep interpretation revamp, fileformat revamp.
	3->4: Changed sizetypes in serialisation from size_t(uint32) to
			smaller types (uint8, USTEPTYPE) (March 2007)
*/



const char CTuningRTI::s_FileExtension[5] = ".tun";



void CTuningRTI::SetNoteName(const NOTEINDEXTYPE& n, const std::string& str)
{
	if(!str.empty())
	{
		m_NoteNameMap[n] = str;
	} else
	{
		const auto iter = m_NoteNameMap.find(n);
		if(iter != m_NoteNameMap.end())
		{
			m_NoteNameMap.erase(iter);
		}
	}
}



bool CTuningRTI::Multiply(const RATIOTYPE& r)
{
	if(r <= 0)
		return true;

	//Note: Multiplying ratios by constant doesn't
	//change, e.g. 'geometricness' status.
	for(auto & ratio : m_RatioTable)
	{
		ratio *= r;
	}
	return false;
}


bool CTuningRTI::CreateGroupGeometric(const NOTEINDEXTYPE& s, const RATIOTYPE& r, const NOTEINDEXTYPE& startindex)
{
	if(s < 1 || r <= 0 || startindex < GetValidityRange().first)
		return true;

	std::vector<RATIOTYPE> v;
	v.reserve(s);
	for(NOTEINDEXTYPE i = startindex; i<startindex+s; i++)
		v.push_back(GetRatio(i));
	return CreateGroupGeometric(v, r, GetValidityRange(), startindex);
}


bool CTuningRTI::CreateGroupGeometric(const std::vector<RATIOTYPE>& v, const RATIOTYPE& r, const VRPAIR vr, const NOTEINDEXTYPE ratiostartpos)
{
	{
		if(vr.first > vr.second || v.size() == 0) return true;
		if(ratiostartpos < vr.first || vr.second < ratiostartpos || static_cast<UNOTEINDEXTYPE>(vr.second - ratiostartpos) < static_cast<UNOTEINDEXTYPE>(v.size() - 1)) return true;
		if(GetFineStepCount() > FINESTEPCOUNT_MAX) return true;
		for(size_t i = 0; i<v.size(); i++) {if(v[i] < 0) return true;}
		if(ProCreateGroupGeometric(v,r, vr, ratiostartpos))
			return true;
		else
		{
			m_TuningType = TT_GROUPGEOMETRIC;
			UpdateFineStepTable();
			return false;
		}
	}
}



bool CTuningRTI::CreateGeometric(const UNOTEINDEXTYPE& s, const RATIOTYPE& r, const VRPAIR vr)
{
	{
		if(vr.first > vr.second) return true;
		if(s < 1 || r <= 0) return true;
		if(ProCreateGeometric(s,r,vr))
			return true;
		else
		{
			m_TuningType = TT_GEOMETRIC;
			UpdateFineStepTable();
			return false;
		}
	}
}




bool CTuningRTI::ChangeGroupsize(const NOTEINDEXTYPE& s)
{
	if(s < 1)
		return true;

	if(m_TuningType == TT_GROUPGEOMETRIC)
		return CreateGroupGeometric(s, GetGroupRatio(), 0);

	if(m_TuningType == TT_GEOMETRIC)
		return CreateGeometric(s, GetGroupRatio());

	return true;
}



bool CTuningRTI::ChangeGroupRatio(const RATIOTYPE& r)
{
	if(r <= 0)
		return true;

	if(m_TuningType == TT_GROUPGEOMETRIC)
		return CreateGroupGeometric(GetGroupSize(), r, 0);

	if(m_TuningType == TT_GEOMETRIC)
		return CreateGeometric(GetGroupSize(), r);

	return true;
}


} // namespace Tuning


OPENMPT_NAMESPACE_END
