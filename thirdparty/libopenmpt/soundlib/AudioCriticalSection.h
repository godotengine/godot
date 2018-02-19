/*
 * AudioCriticalSection.h
 * ---------
 * Purpose: Implementation of OpenMPT's critical section for access to CSoundFile.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

OPENMPT_NAMESPACE_BEGIN

#if defined(MODPLUG_TRACKER)

namespace mpt {
class recursive_mutex_with_lock_count;
} // namespace mpt

namespace Tracker { // implemented in mptrack/Mptrack.cpp
mpt::recursive_mutex_with_lock_count & GetGlobalMutexRef();
} // namespace Tracker

// Critical section handling done in (safe) RAII style.
// Create a CriticalSection object whenever you need exclusive access to CSoundFile.
// One object = one lock / critical section.
// The critical section is automatically left when the object is destroyed, but
// Enter() and Leave() can also be called manually if needed.
class CriticalSection
{
private:
	mpt::recursive_mutex_with_lock_count & m_refGlobalMutex;
protected:
	bool inSection;
public:
	enum InitialState
	{
		InitialLocked = 0,
		InitialUnlocked = 1
	};
public:
	CriticalSection();
	CriticalSection(CriticalSection &&other);
	explicit CriticalSection(InitialState state);
	void Enter();
	void Leave();
	~CriticalSection();
};

#else // !MODPLUG_TRACKER

class CriticalSection
{
public:
	enum InitialState
	{
		InitialLocked = 0,
		InitialUnlocked = 1
	};
public:
	CriticalSection() {}
	explicit CriticalSection(InitialState state) { MPT_UNREFERENCED_PARAMETER(state); }
	void Enter() {}
	void Leave() {}
	~CriticalSection() {}
};

#endif // MODPLUG_TRACKER

OPENMPT_NAMESPACE_END
