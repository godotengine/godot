/*
 * tuningCollection.h
 * ------------------
 * Purpose: Alternative sample tuning collection class.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "tuning.h"
#include <vector>
#include <string>


OPENMPT_NAMESPACE_BEGIN


namespace Tuning {


class CTuningCollection
{

public:

	static const char s_FileExtension[4];

	// OpenMPT <= 1.26 had to following limits:
	//  *  255 built-in tunings (only 2 were ever actually provided)
	//  *  255 local tunings
	//  *  255 tune-specific tunings
	// As 1.27 copies all used tunings into the module, the limit of 255 is no
	// longer sufficient. In the worst case scenario, the module contains 255
	// unused tunings and uses 255 local ones. In addition to that, allow the
	// user to additionally import both built-in tunings.
	// Older OpenMPT versions will silently skip loading tunings beyond index
	// 255.
	static const size_t s_nMaxTuningCount = 255 + 255 + 2;

public:

	//Note: Given pointer is deleted by CTuningCollection
	//at some point.
	bool AddTuning(CTuning *pT);
	bool AddTuning(std::istream& inStrm);
	
	bool Remove(const std::size_t i);
	bool Remove(const CTuning *pT);

	CTuning& GetTuning(size_t i) {return *m_Tunings.at(i).get();}
	const CTuning& GetTuning(size_t i) const {return *m_Tunings.at(i).get();}
	CTuning* GetTuning(const std::string& name);
	const CTuning* GetTuning(const std::string& name) const;

	size_t GetNumTunings() const {return m_Tunings.size();}

	Tuning::SerializationResult Serialize(std::ostream&, const std::string &name) const;
	Tuning::SerializationResult Deserialize(std::istream&, std::string &name);

private:

	std::vector<std::unique_ptr<CTuning> > m_Tunings;

private:

	Tuning::SerializationResult DeserializeOLD(std::istream&, std::string &name);

};


#ifdef MODPLUG_TRACKER
bool UnpackTuningCollection(const CTuningCollection &tc, const mpt::PathString &prefix);
#endif


} // namespace Tuning


typedef Tuning::CTuningCollection CTuningCollection;


OPENMPT_NAMESPACE_END
