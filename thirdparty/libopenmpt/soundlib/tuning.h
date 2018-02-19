/*
 * tuning.h
 * --------
 * Purpose: Alternative sample tuning.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include <map>

#include "tuningbase.h"


OPENMPT_NAMESPACE_BEGIN


namespace Tuning {


class CTuningRTI
{

public:

	static const char s_FileExtension[5];

	enum
	{
		TT_GENERAL        = 0,
		TT_GROUPGEOMETRIC = 1,
		TT_GEOMETRIC      = 3,
	};

	static const RATIOTYPE s_DefaultFallbackRatio;
	static const NOTEINDEXTYPE s_StepMinDefault = -64;
	static const UNOTEINDEXTYPE s_RatioTableSizeDefault = 128;
	static const USTEPINDEXTYPE s_RatioTableFineSizeMaxDefault = 1000;

public:

	//To return ratio of certain note.
	RATIOTYPE GetRatio(const NOTEINDEXTYPE& stepsFromCentre) const;

	//To return ratio from a 'step'(noteindex + stepindex)
	RATIOTYPE GetRatio(const NOTEINDEXTYPE& stepsFromCentre, const STEPINDEXTYPE& fineSteps) const;

	UNOTEINDEXTYPE GetRatioTableSize() const {return static_cast<UNOTEINDEXTYPE>(m_RatioTable.size());}

	NOTEINDEXTYPE GetRatioTableBeginNote() const {return m_StepMin;}

	//Tuning might not be valid for arbitrarily large range,
	//so this can be used to ask where it is valid. Tells the lowest and highest
	//note that are valid.
	VRPAIR GetValidityRange() const {return VRPAIR(m_StepMin, static_cast<NOTEINDEXTYPE>(m_StepMin + static_cast<NOTEINDEXTYPE>(m_RatioTable.size()) - 1));}

	//Return true if note is within validity range - false otherwise.
	bool IsValidNote(const NOTEINDEXTYPE n) const {return (n >= GetValidityRange().first && n <= GetValidityRange().second);}

	UNOTEINDEXTYPE GetGroupSize() const {return m_GroupSize;}

	RATIOTYPE GetGroupRatio() const {return m_GroupRatio;}

	//To return (fine)stepcount between two consecutive mainsteps.
	USTEPINDEXTYPE GetFineStepCount() const {return m_FineStepCount;}

	//To return 'directed distance' between given notes.
	STEPINDEXTYPE GetStepDistance(const NOTEINDEXTYPE& from, const NOTEINDEXTYPE& to) const
		{return (to - from)*(static_cast<NOTEINDEXTYPE>(GetFineStepCount())+1);}

	//To return 'directed distance' between given steps.
	STEPINDEXTYPE GetStepDistance(const NOTEINDEXTYPE& noteFrom, const STEPINDEXTYPE& stepDistFrom, const NOTEINDEXTYPE& noteTo, const STEPINDEXTYPE& stepDistTo) const
		{return GetStepDistance(noteFrom, noteTo) + stepDistTo - stepDistFrom;}

	//To set finestepcount between two consecutive mainsteps.
	//Finestep count == 0 means that
	//stepdistances become the same as note distances.
	void SetFineStepCount(const USTEPINDEXTYPE& fs);

	//Multiply all ratios by given number.
	bool Multiply(const RATIOTYPE&);

	bool SetRatio(const NOTEINDEXTYPE& s, const RATIOTYPE& r);

	TUNINGTYPE GetType() const {return m_TuningType;}

	std::string GetNoteName(const NOTEINDEXTYPE& x, bool addOctave = true) const;

	void SetNoteName(const NOTEINDEXTYPE&, const std::string&);

	static CTuningRTI* CreateDeserialize(std::istream & f)
	{
		CTuningRTI *pT = new CTuningRTI();
		if(pT->InitDeserialize(f) != SerializationResult::Success)
		{
			delete pT;
			return nullptr;
		}
		return pT;
	}

	//Try to read old version (v.3) and return pointer to new instance if succesfull, else nullptr.
	static CTuningRTI* CreateDeserializeOLD(std::istream & f)
	{
		CTuningRTI *pT = new CTuningRTI();
		if(pT->InitDeserializeOLD(f) != SerializationResult::Success)
		{
			delete pT;
			return nullptr;
		}
		return pT;
	}

	static CTuningRTI* CreateGeneral(const std::string &name)
	{
		CTuningRTI *pT = new CTuningRTI();
		pT->SetName(name);
		return pT;
	}

	static CTuningRTI* CreateGroupGeometric(const std::string &name, UNOTEINDEXTYPE groupsize, RATIOTYPE groupratio, USTEPINDEXTYPE finestepcount)
	{
		CTuningRTI *pT = new CTuningRTI();
		pT->SetName(name);
		if(pT->CreateGroupGeometric(groupsize, groupratio, 0) != false)
		{
			delete pT;
			return nullptr;
		}
		pT->SetFineStepCount(finestepcount);
		return pT;
	}

	static CTuningRTI* CreateGroupGeometric(const std::string &name, const std::vector<RATIOTYPE> &ratios, RATIOTYPE groupratio, USTEPINDEXTYPE finestepcount)
	{
		CTuningRTI *pT = new CTuningRTI();
		pT->SetName(name);
		VRPAIR range = std::make_pair(s_StepMinDefault, static_cast<NOTEINDEXTYPE>(s_StepMinDefault + s_RatioTableSizeDefault - 1));
		range.second = std::max(range.second, mpt::saturate_cast<NOTEINDEXTYPE>(ratios.size() - 1));
		range.first = 0 - range.second - 1;
		if(pT->CreateGroupGeometric(ratios, groupratio, range, 0) != false)
		{
			delete pT;
			return nullptr;
		}
		pT->SetFineStepCount(finestepcount);
		return pT;
	}

	static CTuningRTI* CreateGeometric(const std::string &name, UNOTEINDEXTYPE groupsize, RATIOTYPE groupratio, USTEPINDEXTYPE finestepcount)
	{
		CTuningRTI *pT = new CTuningRTI();
		pT->SetName(name);
		if(pT->CreateGeometric(groupsize, groupratio) != false)
		{
			delete pT;
			return nullptr;
		}
		pT->SetFineStepCount(finestepcount);
		return pT;
	}

	Tuning::SerializationResult Serialize(std::ostream& out) const;

#ifdef MODPLUG_TRACKER
	bool WriteSCL(std::ostream &f, const mpt::PathString &filename) const;
#endif

	bool ChangeGroupsize(const NOTEINDEXTYPE&);
	bool ChangeGroupRatio(const RATIOTYPE&);

	void SetName(const std::string& s) { m_TuningName = s; }
	std::string GetName() const {return m_TuningName;}

private:

	CTuningRTI();

	SerializationResult InitDeserialize(std::istream& inStrm);

	//Try to read old version (v.3) and return pointer to new instance if succesfull, else nullptr.
	SerializationResult InitDeserializeOLD(std::istream&);

	//Create GroupGeometric tuning of *this using virtual ProCreateGroupGeometric.
	bool CreateGroupGeometric(const std::vector<RATIOTYPE>&, const RATIOTYPE&, const VRPAIR vr, const NOTEINDEXTYPE ratiostartpos);

	//Create GroupGeometric of *this using ratios from 'itself' and ratios starting from
	//position given as third argument.
	bool CreateGroupGeometric(const NOTEINDEXTYPE&, const RATIOTYPE&, const NOTEINDEXTYPE&);

	//Create geometric tuning of *this using ratio(0) = 1.
	bool CreateGeometric(const UNOTEINDEXTYPE& p, const RATIOTYPE& r) {return CreateGeometric(p,r,GetValidityRange());}
	bool CreateGeometric(const UNOTEINDEXTYPE&, const RATIOTYPE&, const VRPAIR vr);

	//The two methods below return false if action was done, true otherwise.
	bool ProCreateGroupGeometric(const std::vector<RATIOTYPE>&, const RATIOTYPE&, const VRPAIR&, const NOTEINDEXTYPE& ratiostartpos);
	bool ProCreateGeometric(const UNOTEINDEXTYPE&, const RATIOTYPE&, const VRPAIR&);

	void UpdateFineStepTable();

	//Note: Stepdiff should be in range [1, finestepcount]
	RATIOTYPE GetRatioFine(const NOTEINDEXTYPE& note, USTEPINDEXTYPE stepDiff) const;

	//GroupPeriodic-specific.
	//Get the corresponding note in [0, period-1].
	//For example GetRefNote(-1) is to return note :'groupsize-1'.
	NOTEINDEXTYPE GetRefNote(NOTEINDEXTYPE note) const;

	bool IsNoteInTable(const NOTEINDEXTYPE& s) const
	{
		if(s < m_StepMin || s >= m_StepMin + static_cast<NOTEINDEXTYPE>(m_RatioTable.size()))
			return false;
		else
			return true;
	}

private:

	TUNINGTYPE m_TuningType;

	//Noteratios
	std::vector<RATIOTYPE> m_RatioTable;

	//'Fineratios'
	std::vector<RATIOTYPE> m_RatioTableFine;

	//The lowest index of note in the table
	NOTEINDEXTYPE m_StepMin; // this should REALLY be called 'm_NoteMin' renaming was missed in r192

	//For groupgeometric tunings, tells the 'group size' and 'group ratio'
	//m_GroupSize should always be >= 0.
	NOTEINDEXTYPE m_GroupSize;
	RATIOTYPE m_GroupRatio;

	USTEPINDEXTYPE m_FineStepCount;

	std::string m_TuningName;

	std::map<NOTEINDEXTYPE, std::string> m_NoteNameMap;

}; // class CTuningRTI


typedef CTuningRTI CTuning;


} // namespace Tuning


OPENMPT_NAMESPACE_END
