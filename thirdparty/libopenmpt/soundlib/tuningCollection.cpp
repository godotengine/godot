/*
 * tuningCollection.cpp
 * --------------------
 * Purpose: Alternative sample tuning collection class.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "tuningcollection.h"
#include "../common/mptIO.h"
#include "../common/serialization_utils.h"
#include <algorithm>
#include "../common/mptFileIO.h"
#include "Loaders.h"


OPENMPT_NAMESPACE_BEGIN


namespace Tuning {


/*
Version history:
	2->3: Serialization revamp(August 2007)
	1->2: Sizetypes of string serialisation from size_t(uint32)
		  to uint8. (March 2007)
*/


const char CTuningCollection::s_FileExtension[4] = ".tc";


namespace CTuningS11n
{
	void ReadStr(std::istream& iStrm, std::string& str, const size_t);
	void WriteStr(std::ostream& oStrm, const std::string& str);
} // namespace CTuningS11n

using namespace CTuningS11n;


static void ReadTuning(std::istream& iStrm, CTuningCollection& Tc, const size_t)
{
	Tc.AddTuning(iStrm);
}

static void WriteTuning(std::ostream& oStrm, const CTuning& t)
{
	t.Serialize(oStrm);
}


CTuning* CTuningCollection::GetTuning(const std::string& name)
{
	for(std::size_t i = 0; i<m_Tunings.size(); i++)
	{
		if(m_Tunings[i]->GetName() == name)
		{
			return m_Tunings[i].get();
		}
	}
	return nullptr;
}

const CTuning* CTuningCollection::GetTuning(const std::string& name) const
{
	for(std::size_t i = 0; i<m_Tunings.size(); i++)
	{
		if(m_Tunings[i]->GetName() == name)
		{
			return m_Tunings[i].get();
		}
	}
	return nullptr;
}


Tuning::SerializationResult CTuningCollection::Serialize(std::ostream& oStrm, const std::string &name) const
{
	srlztn::SsbWrite ssb(oStrm);
	ssb.BeginWrite("TC", 3); // version
	ssb.WriteItem(name, "0", &WriteStr);
	uint16 dummyEditMask = 0xffff;
	ssb.WriteItem(dummyEditMask, "1");

	const size_t tcount = m_Tunings.size();
	for(size_t i = 0; i<tcount; i++)
		ssb.WriteItem(*m_Tunings[i], "2", &WriteTuning);
	ssb.FinishWrite();
		
	if(ssb.GetStatus() & srlztn::SNT_FAILURE)
		return Tuning::SerializationResult::Failure;
	else
		return Tuning::SerializationResult::Success;
}


Tuning::SerializationResult CTuningCollection::Deserialize(std::istream& iStrm, std::string &name)
{
	std::istream::pos_type startpos = iStrm.tellg();
	
	const Tuning::SerializationResult oldLoadingResult = DeserializeOLD(iStrm, name);

	if(oldLoadingResult == Tuning::SerializationResult::NoMagic)
	{	// An old version was not recognised - trying new version.
		iStrm.clear();
		iStrm.seekg(startpos);
		srlztn::SsbRead ssb(iStrm);
		ssb.BeginRead("TC", 3); // version

		const srlztn::SsbRead::ReadIterator iterBeg = ssb.GetReadBegin();
		const srlztn::SsbRead::ReadIterator iterEnd = ssb.GetReadEnd();
		for(srlztn::SsbRead::ReadIterator iter = iterBeg; iter != iterEnd; iter++)
		{
			uint16 dummyEditMask = 0xffff;
			if (ssb.CompareId(iter, "0") == srlztn::SsbRead::IdMatch)
				ssb.ReadIterItem(iter, name, &ReadStr);
			else if (ssb.CompareId(iter, "1") == srlztn::SsbRead::IdMatch)
				ssb.ReadIterItem(iter, dummyEditMask);
			else if (ssb.CompareId(iter, "2") == srlztn::SsbRead::IdMatch)
				ssb.ReadIterItem(iter, *this, &ReadTuning);
		}

		if(ssb.GetStatus() & srlztn::SNT_FAILURE)
			return Tuning::SerializationResult::Failure;
		else
			return Tuning::SerializationResult::Success;
	}
	else
	{
		return oldLoadingResult;
	}
}


Tuning::SerializationResult CTuningCollection::DeserializeOLD(std::istream& inStrm, std::string &name)
{

	//1. begin marker:
	int32 beginMarker = 0;
	mpt::IO::ReadIntLE<int32>(inStrm, beginMarker);
	if(beginMarker != MAGIC4BE('T','C','S','H'))
		return Tuning::SerializationResult::NoMagic;

	//2. version
	int32 version = 0;
	mpt::IO::ReadIntLE<int32>(inStrm, version);
	if(version > 2 || version < 1)
		return Tuning::SerializationResult::Failure;

	//3. Name
	if(version < 2)
	{
		if(!mpt::IO::ReadSizedStringLE<uint32>(inStrm, name, 256))
			return Tuning::SerializationResult::Failure;
	}
	else
	{
		if(!mpt::IO::ReadSizedStringLE<uint8>(inStrm, name))
			return Tuning::SerializationResult::Failure;
	}

	//4. Editmask
	int16 em = 0;
	mpt::IO::ReadIntLE<int16>(inStrm, em);
	//Not assigning the value yet, for if it sets some property const,
	//further loading might fail.

	//5. Tunings
	{
		uint32 s = 0;
		mpt::IO::ReadIntLE<uint32>(inStrm, s);
		if(s > 50)
			return Tuning::SerializationResult::Failure;
		for(size_t i = 0; i<s; i++)
		{
			if(AddTuning(inStrm))
				return Tuning::SerializationResult::Failure;
		}
	}

	//6. End marker
	int32 endMarker = 0;
	mpt::IO::ReadIntLE<int32>(inStrm, endMarker);
	if(endMarker != MAGIC4BE('T','C','S','F'))
		return Tuning::SerializationResult::Failure;
	
	return Tuning::SerializationResult::Success;
}



bool CTuningCollection::Remove(const CTuning *pT)
{
	const auto it = std::find_if(m_Tunings.begin(), m_Tunings.end(),
		[&] (const std::unique_ptr<CTuning> & upT) -> bool
		{
			return upT.get() == pT;
		}
		);
	if(it == m_Tunings.end())
	{
		return false;
	}
	m_Tunings.erase(it);
	return true;
}


bool CTuningCollection::Remove(const std::size_t i)
{
	if(i >= m_Tunings.size())
	{
		return false;
	}
	m_Tunings.erase(m_Tunings.begin() + i);
	return true;
}


bool CTuningCollection::AddTuning(CTuning *pT)
{
	if(m_Tunings.size() >= s_nMaxTuningCount)
		return true;

	if(pT == NULL)
		return true;

	m_Tunings.push_back(std::unique_ptr<CTuning>(pT));

	return false;
}


bool CTuningCollection::AddTuning(std::istream& inStrm)
{
	if(m_Tunings.size() >= s_nMaxTuningCount)
		return true;

	if(!inStrm.good()) return true;

	CTuning* pT = CTuning::CreateDeserializeOLD(inStrm);
	if(pT == 0) pT = CTuning::CreateDeserialize(inStrm);

	if(pT == 0)
		return true;
	else
	{
		m_Tunings.push_back(std::unique_ptr<CTuning>(pT));
		return false;
	}
}


#ifdef MODPLUG_TRACKER


bool UnpackTuningCollection(const CTuningCollection &tc, const mpt::PathString &prefix)
{
	bool error = false;
	auto numberFmt = mpt::FormatSpec().Dec().FillNul().Width(1 + static_cast<int>(std::log10(tc.GetNumTunings())));
	for(std::size_t i = 0; i < tc.GetNumTunings(); ++i)
	{
		const CTuning & tuning = tc.GetTuning(i);
		mpt::PathString fn;
		fn += prefix;
		mpt::ustring tuningName = mpt::ToUnicode(mpt::CharsetLocale, tuning.GetName());
		if(tuningName.empty())
		{
			tuningName = MPT_USTRING("untitled");
		}
		SanitizeFilename(tuningName);
		fn += mpt::PathString::FromUnicode(mpt::format(MPT_USTRING("%1 - %2"))(numberFmt.ToWString(i + 1), tuningName));
		fn += mpt::PathString::FromUTF8(CTuning::s_FileExtension);
		if(fn.FileOrDirectoryExists())
		{
			error = true;
		} else
		{
			mpt::ofstream fout(fn, std::ios::binary);
			if(tuning.Serialize(fout) != Tuning::SerializationResult::Success)
			{
				error = true;
			}
			fout.close();
		}
	}
	return !error;
}


#endif


} // namespace Tuning


OPENMPT_NAMESPACE_END
