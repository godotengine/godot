/*
 * Profiler.cpp
 * ------------
 * Purpose: Performance measuring
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "Profiler.h"


OPENMPT_NAMESPACE_BEGIN


#ifdef USE_PROFILER


class Statistics
{
public:
	Profile &profile;
	Profile::Data data;
	double usage;
	Statistics(Profile &p) : profile(p)
	{
		usage = 0.0;
		Update();
	}
	void Update()
	{
		data = profile.GetAndResetData();
		uint64 now = profile.GetTime();
		uint64 timewindow = now - data.Start;
		if(data.Calls > 0 && timewindow > 0)
		{
			usage = (double)data.Sum / (double)timewindow;
		} else
		{
			usage = 0.0;
		}
	}
};


struct ProfileBlock
{
	class Profile * profile;
	const char * name;
	class Statistics * stats;
};

static const std::size_t MAX_PROFILES = 1024;

static ProfileBlock Profiles[ MAX_PROFILES ];

static std::size_t NextProfile = 0;


static void RegisterProfile(Profile *newprofile)
{
	if(NextProfile < MAX_PROFILES)
	{
		Profiles[NextProfile].profile = newprofile;
		Profiles[NextProfile].stats = 0;
		NextProfile++;
	}
}


static void UnregisterProfile(Profile *oldprofile)
{
	for(std::size_t i=0; i<NextProfile; i++) {
		if(Profiles[i].profile == oldprofile) {
			Profiles[i].profile = 0;
			delete Profiles[i].stats;
			Profiles[i].stats = 0;
		}
	}
}


void Profiler::Update()
{
	for(std::size_t i=0; i<NextProfile; i++)
	{
		if(!Profiles[i].stats)
		{
			Profiles[i].stats = new Statistics(*Profiles[i].profile);
		} else
		{
			Profiles[i].stats->Update();
		}
	}
}


std::string Profiler::DumpProfiles()
{
	std::string ret;
	for(std::size_t i=0; i<NextProfile; i++)
	{
		if(Profiles[i].stats)
		{
			Statistics &stats = *Profiles[i].stats;
			std::string cat;
			switch(stats.profile.Category)
			{
			case Profiler::GUI: cat = "GUI"; break;
			case Profiler::Audio: cat = "Audio"; break;
			case Profiler::Notify: cat = "Notify"; break;
			}
			ret += cat + " " + std::string(stats.profile.Name) + ": " + mpt::fmt::f("%6.3f", stats.usage * 100.0) + "%\r\n";
		}
	}
	ret += "\r\n";
	return ret;
}


std::vector<double> Profiler::DumpCategories()
{
	std::vector<double> ret;
	ret.resize(Profiler::CategoriesCount);
	for(std::size_t i=0; i<NextProfile; i++)
	{
		if(Profiles[i].stats)
		{
			ret[Profiles[i].profile->Category] += Profiles[i].stats->usage;
		}
	}
	return ret;
}


uint64 Profile::GetTime() const
{
	LARGE_INTEGER ret;
	ret.QuadPart = 0;
	QueryPerformanceCounter(&ret);
	return ret.QuadPart;
}


uint64 Profile::GetFrequency() const
{
	LARGE_INTEGER ret;
	ret.QuadPart = 0;
	QueryPerformanceFrequency(&ret);
	return ret.QuadPart;
}


Profile::Profile(Profiler::Category category, const char *name) : Category(category), Name(name)
{
	data.Calls = 0;
	data.Sum = 0;
	data.Overhead = 0;
	data.Start = GetTime();
	EnterTime = 0;
	RegisterProfile(this);
}


Profile::~Profile()
{
	UnregisterProfile(this);
}


Profile::Data Profile::GetAndResetData()
{
	Profile::Data ret;
	datamutex.lock();
	ret = data;
	data.Calls = 0;
	data.Sum = 0;
	data.Overhead = 0;
	data.Start = GetTime();
	datamutex.unlock();
	return ret;
}


void Profile::Reset()
{
	datamutex.lock();
	data.Calls = 0;
	data.Sum = 0;
	data.Overhead = 0;
	data.Start = GetTime();
	datamutex.unlock();
}


void Profile::Enter()
{
	EnterTime = GetTime();
}


void Profile::Leave()
{
	uint64 LeaveTime = GetTime();
	datamutex.lock();
	data.Calls += 1;
	data.Sum += LeaveTime - EnterTime;
	datamutex.unlock();
}


#else // !USE_PROFILER

MPT_MSVC_WORKAROUND_LNK4221(Profiler)

#endif // USE_PROFILER


OPENMPT_NAMESPACE_END
