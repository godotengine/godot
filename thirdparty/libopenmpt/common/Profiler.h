/*
 * Profiler.h
 * ----------
 * Purpose: Performance measuring
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


#include "../common/mptMutex.h"
#include <string>
#include <vector>


OPENMPT_NAMESPACE_BEGIN


#if defined(MODPLUG_TRACKER)

//#define USE_PROFILER

#endif

#ifdef USE_PROFILER

class Profiler
{
public:
	enum Category
	{
		GUI,
		Audio,
		Notify,
		CategoriesCount
	};
	static std::vector<std::string> GetCategoryNames()
	{
		std::vector<std::string> ret;
		ret.push_back("GUI");
		ret.push_back("Audio");
		ret.push_back("Notify");
		return ret;
	}
public:
	static void Update();
	static std::string DumpProfiles();
	static std::vector<double> DumpCategories();
};


class Profile
{
private:
	mutable mpt::mutex datamutex;
public:
	struct Data
	{
		uint64 Calls;
		uint64 Sum;
		int64  Overhead;
		uint64 Start;
	};
public:
	Data data;
	uint64 EnterTime;
	Profiler::Category Category;
	const char * const Name;
	uint64 GetTime() const;
	uint64 GetFrequency() const;
public:
	Profile(Profiler::Category category, const char *name);
	~Profile();
	void Reset();
	void Enter();
	void Leave();
	class Scope
	{
	private:
		Profile &profile;
	public:
		Scope(Profile &p) : profile(p) { profile.Enter(); }
		~Scope() { profile.Leave(); }
	};
public:
	Data GetAndResetData();
};


#define OPENMPT_PROFILE_SCOPE(cat, name) \
	static Profile OPENMPT_PROFILE_VAR(cat, name);\
	Profile::Scope OPENMPT_PROFILE_SCOPE_VAR(OPENMPT_PROFILE_VAR); \
/**/


#define OPENMPT_PROFILE_FUNCTION(cat) OPENMPT_PROFILE_SCOPE(cat, __FUNCTION__)


#else // !USE_PROFILER


class Profiler
{
public:
	enum Category
	{
		CategoriesCount
	};
	static std::vector<std::string> GetCategoryNames() { return std::vector<std::string>(); } 
public:
	static void Update() { }
	static std::string DumpProfiles() { return std::string(); }
	static std::vector<double> DumpCategories() { return std::vector<double>(); }
};
#define OPENMPT_PROFILE_SCOPE(cat, name) MPT_DO { } MPT_WHILE_0
#define OPENMPT_PROFILE_FUNCTION(cat) MPT_DO { } MPT_WHILE_0


#endif // USE_PROFILER


OPENMPT_NAMESPACE_END
