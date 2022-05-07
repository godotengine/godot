/*

***************************************************************************************************
**
** profile.cpp
**
** Real-Time Hierarchical Profiling for Game Programming Gems 3
**
** by Greg Hjelstrom & Byon Garrabrant
**
***************************************************************************************************/

// Credits: The Clock class was inspired by the Timer classes in
// Ogre (www.ogre3d.org).

#include "btQuickprof.h"
#include "btThreads.h"

#ifdef __CELLOS_LV2__
#include <sys/sys_time.h>
#include <sys/time_util.h>
#include <stdio.h>
#endif

#if defined(SUNOS) || defined(__SUNOS__)
#include <stdio.h>
#endif
#ifdef __APPLE__
#include <mach/mach_time.h>
#include <TargetConditionals.h>
#endif

#if defined(WIN32) || defined(_WIN32)

#define BT_USE_WINDOWS_TIMERS
#define WIN32_LEAN_AND_MEAN
#define NOWINRES
#define NOMCX
#define NOIME

#ifdef _XBOX
#include <Xtl.h>
#else  //_XBOX
#include <windows.h>

#if WINVER < 0x0602
#define GetTickCount64 GetTickCount
#endif

#endif  //_XBOX

#include <time.h>

#else  //_WIN32
#include <sys/time.h>

#ifdef BT_LINUX_REALTIME
//required linking against rt (librt)
#include <time.h>
#endif  //BT_LINUX_REALTIME

#endif  //_WIN32

#define mymin(a, b) (a > b ? a : b)

struct btClockData
{
#ifdef BT_USE_WINDOWS_TIMERS
	LARGE_INTEGER mClockFrequency;
	LONGLONG mStartTick;
	LARGE_INTEGER mStartTime;
#else
#ifdef __CELLOS_LV2__
	uint64_t mStartTime;
#else
#ifdef __APPLE__
	uint64_t mStartTimeNano;
#endif
	struct timeval mStartTime;
#endif
#endif  //__CELLOS_LV2__
};

///The btClock is a portable basic clock that measures accurate time in seconds, use for profiling.
btClock::btClock()
{
	m_data = new btClockData;
#ifdef BT_USE_WINDOWS_TIMERS
	QueryPerformanceFrequency(&m_data->mClockFrequency);
#endif
	reset();
}

btClock::~btClock()
{
	delete m_data;
}

btClock::btClock(const btClock& other)
{
	m_data = new btClockData;
	*m_data = *other.m_data;
}

btClock& btClock::operator=(const btClock& other)
{
	*m_data = *other.m_data;
	return *this;
}

/// Resets the initial reference time.
void btClock::reset()
{
#ifdef BT_USE_WINDOWS_TIMERS
	QueryPerformanceCounter(&m_data->mStartTime);
	m_data->mStartTick = GetTickCount64();
#else
#ifdef __CELLOS_LV2__

	typedef uint64_t ClockSize;
	ClockSize newTime;
	//__asm __volatile__( "mftb %0" : "=r" (newTime) : : "memory");
	SYS_TIMEBASE_GET(newTime);
	m_data->mStartTime = newTime;
#else
#ifdef __APPLE__
	m_data->mStartTimeNano = mach_absolute_time();
#endif
	gettimeofday(&m_data->mStartTime, 0);
#endif
#endif
}

/// Returns the time in ms since the last call to reset or since
/// the btClock was created.
unsigned long long int btClock::getTimeMilliseconds()
{
#ifdef BT_USE_WINDOWS_TIMERS
	LARGE_INTEGER currentTime;
	QueryPerformanceCounter(&currentTime);
	LONGLONG elapsedTime = currentTime.QuadPart -
						   m_data->mStartTime.QuadPart;
	// Compute the number of millisecond ticks elapsed.
	unsigned long msecTicks = (unsigned long)(1000 * elapsedTime /
											  m_data->mClockFrequency.QuadPart);

	return msecTicks;
#else

#ifdef __CELLOS_LV2__
	uint64_t freq = sys_time_get_timebase_frequency();
	double dFreq = ((double)freq) / 1000.0;
	typedef uint64_t ClockSize;
	ClockSize newTime;
	SYS_TIMEBASE_GET(newTime);
	//__asm __volatile__( "mftb %0" : "=r" (newTime) : : "memory");

	return (unsigned long int)((double(newTime - m_data->mStartTime)) / dFreq);
#else

	struct timeval currentTime;
	gettimeofday(&currentTime, 0);
	return (currentTime.tv_sec - m_data->mStartTime.tv_sec) * 1000 +
		   (currentTime.tv_usec - m_data->mStartTime.tv_usec) / 1000;
#endif  //__CELLOS_LV2__
#endif
}

/// Returns the time in us since the last call to reset or since
/// the Clock was created.
unsigned long long int btClock::getTimeMicroseconds()
{
#ifdef BT_USE_WINDOWS_TIMERS
	//see https://msdn.microsoft.com/en-us/library/windows/desktop/dn553408(v=vs.85).aspx
	LARGE_INTEGER currentTime, elapsedTime;

	QueryPerformanceCounter(&currentTime);
	elapsedTime.QuadPart = currentTime.QuadPart -
						   m_data->mStartTime.QuadPart;
	elapsedTime.QuadPart *= 1000000;
	elapsedTime.QuadPart /= m_data->mClockFrequency.QuadPart;

	return (unsigned long long)elapsedTime.QuadPart;
#else

#ifdef __CELLOS_LV2__
	uint64_t freq = sys_time_get_timebase_frequency();
	double dFreq = ((double)freq) / 1000000.0;
	typedef uint64_t ClockSize;
	ClockSize newTime;
	//__asm __volatile__( "mftb %0" : "=r" (newTime) : : "memory");
	SYS_TIMEBASE_GET(newTime);

	return (unsigned long int)((double(newTime - m_data->mStartTime)) / dFreq);
#else

	struct timeval currentTime;
	gettimeofday(&currentTime, 0);
	return (currentTime.tv_sec - m_data->mStartTime.tv_sec) * 1000000 +
		   (currentTime.tv_usec - m_data->mStartTime.tv_usec);
#endif  //__CELLOS_LV2__
#endif
}

unsigned long long int btClock::getTimeNanoseconds()
{
#ifdef BT_USE_WINDOWS_TIMERS
	//see https://msdn.microsoft.com/en-us/library/windows/desktop/dn553408(v=vs.85).aspx
	LARGE_INTEGER currentTime, elapsedTime;

	QueryPerformanceCounter(&currentTime);
	elapsedTime.QuadPart = currentTime.QuadPart -
						   m_data->mStartTime.QuadPart;
	elapsedTime.QuadPart *= 1000000000;
	elapsedTime.QuadPart /= m_data->mClockFrequency.QuadPart;

	return (unsigned long long)elapsedTime.QuadPart;
#else

#ifdef __CELLOS_LV2__
	uint64_t freq = sys_time_get_timebase_frequency();
	double dFreq = ((double)freq) / 1e9;
	typedef uint64_t ClockSize;
	ClockSize newTime;
	//__asm __volatile__( "mftb %0" : "=r" (newTime) : : "memory");
	SYS_TIMEBASE_GET(newTime);

	return (unsigned long int)((double(newTime - m_data->mStartTime)) / dFreq);
#else
#ifdef __APPLE__
	uint64_t ticks = mach_absolute_time() - m_data->mStartTimeNano;
	static long double conversion = 0.0L;
	if (0.0L == conversion)
	{
		// attempt to get conversion to nanoseconds
		mach_timebase_info_data_t info;
		int err = mach_timebase_info(&info);
		if (err)
		{
			btAssert(0);
			conversion = 1.;
		}
		conversion = info.numer / info.denom;
	}
	return (ticks * conversion);

#else  //__APPLE__

#ifdef BT_LINUX_REALTIME
	timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	return 1000000000 * ts.tv_sec + ts.tv_nsec;
#else
	struct timeval currentTime;
	gettimeofday(&currentTime, 0);
	return (currentTime.tv_sec - m_data->mStartTime.tv_sec) * 1e9 +
		   (currentTime.tv_usec - m_data->mStartTime.tv_usec) * 1000;
#endif  //BT_LINUX_REALTIME

#endif  //__APPLE__
#endif  //__CELLOS_LV2__
#endif
}

/// Returns the time in s since the last call to reset or since
/// the Clock was created.
btScalar btClock::getTimeSeconds()
{
	static const btScalar microseconds_to_seconds = btScalar(0.000001);
	return btScalar(getTimeMicroseconds()) * microseconds_to_seconds;
}

#ifndef BT_NO_PROFILE

static btClock gProfileClock;

inline void Profile_Get_Ticks(unsigned long int* ticks)
{
	*ticks = (unsigned long int)gProfileClock.getTimeMicroseconds();
}

inline float Profile_Get_Tick_Rate(void)
{
	//	return 1000000.f;
	return 1000.f;
}

/***************************************************************************************************
**
** CProfileNode
**
***************************************************************************************************/

/***********************************************************************************************
 * INPUT:                                                                                      *
 * name - pointer to a static string which is the name of this profile node                    *
 * parent - parent pointer                                                                     *
 *                                                                                             *
 * WARNINGS:                                                                                   *
 * The name is assumed to be a static pointer, only the pointer is stored and compared for     *
 * efficiency reasons.                                                                         *
 *=============================================================================================*/
CProfileNode::CProfileNode(const char* name, CProfileNode* parent) : Name(name),
																	 TotalCalls(0),
																	 TotalTime(0),
																	 StartTime(0),
																	 RecursionCounter(0),
																	 Parent(parent),
																	 Child(NULL),
																	 Sibling(NULL),
																	 m_userPtr(0)
{
	Reset();
}

void CProfileNode::CleanupMemory()
{
	delete (Child);
	Child = NULL;
	delete (Sibling);
	Sibling = NULL;
}

CProfileNode::~CProfileNode(void)
{
	CleanupMemory();
}

/***********************************************************************************************
 * INPUT:                                                                                      *
 * name - static string pointer to the name of the node we are searching for                   *
 *                                                                                             *
 * WARNINGS:                                                                                   *
 * All profile names are assumed to be static strings so this function uses pointer compares   *
 * to find the named node.                                                                     *
 *=============================================================================================*/
CProfileNode* CProfileNode::Get_Sub_Node(const char* name)
{
	// Try to find this sub node
	CProfileNode* child = Child;
	while (child)
	{
		if (child->Name == name)
		{
			return child;
		}
		child = child->Sibling;
	}

	// We didn't find it, so add it

	CProfileNode* node = new CProfileNode(name, this);
	node->Sibling = Child;
	Child = node;
	return node;
}

void CProfileNode::Reset(void)
{
	TotalCalls = 0;
	TotalTime = 0.0f;

	if (Child)
	{
		Child->Reset();
	}
	if (Sibling)
	{
		Sibling->Reset();
	}
}

void CProfileNode::Call(void)
{
	TotalCalls++;
	if (RecursionCounter++ == 0)
	{
		Profile_Get_Ticks(&StartTime);
	}
}

bool CProfileNode::Return(void)
{
	if (--RecursionCounter == 0 && TotalCalls != 0)
	{
		unsigned long int time;
		Profile_Get_Ticks(&time);

		time -= StartTime;
		TotalTime += (float)time / Profile_Get_Tick_Rate();
	}
	return (RecursionCounter == 0);
}

/***************************************************************************************************
**
** CProfileIterator
**
***************************************************************************************************/
CProfileIterator::CProfileIterator(CProfileNode* start)
{
	CurrentParent = start;
	CurrentChild = CurrentParent->Get_Child();
}

void CProfileIterator::First(void)
{
	CurrentChild = CurrentParent->Get_Child();
}

void CProfileIterator::Next(void)
{
	CurrentChild = CurrentChild->Get_Sibling();
}

bool CProfileIterator::Is_Done(void)
{
	return CurrentChild == NULL;
}

void CProfileIterator::Enter_Child(int index)
{
	CurrentChild = CurrentParent->Get_Child();
	while ((CurrentChild != NULL) && (index != 0))
	{
		index--;
		CurrentChild = CurrentChild->Get_Sibling();
	}

	if (CurrentChild != NULL)
	{
		CurrentParent = CurrentChild;
		CurrentChild = CurrentParent->Get_Child();
	}
}

void CProfileIterator::Enter_Parent(void)
{
	if (CurrentParent->Get_Parent() != NULL)
	{
		CurrentParent = CurrentParent->Get_Parent();
	}
	CurrentChild = CurrentParent->Get_Child();
}

/***************************************************************************************************
**
** CProfileManager
**
***************************************************************************************************/

CProfileNode gRoots[BT_QUICKPROF_MAX_THREAD_COUNT] = {
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL),
	CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL), CProfileNode("Root", NULL)};

CProfileNode* gCurrentNodes[BT_QUICKPROF_MAX_THREAD_COUNT] =
	{
		&gRoots[0],
		&gRoots[1],
		&gRoots[2],
		&gRoots[3],
		&gRoots[4],
		&gRoots[5],
		&gRoots[6],
		&gRoots[7],
		&gRoots[8],
		&gRoots[9],
		&gRoots[10],
		&gRoots[11],
		&gRoots[12],
		&gRoots[13],
		&gRoots[14],
		&gRoots[15],
		&gRoots[16],
		&gRoots[17],
		&gRoots[18],
		&gRoots[19],
		&gRoots[20],
		&gRoots[21],
		&gRoots[22],
		&gRoots[23],
		&gRoots[24],
		&gRoots[25],
		&gRoots[26],
		&gRoots[27],
		&gRoots[28],
		&gRoots[29],
		&gRoots[30],
		&gRoots[31],
		&gRoots[32],
		&gRoots[33],
		&gRoots[34],
		&gRoots[35],
		&gRoots[36],
		&gRoots[37],
		&gRoots[38],
		&gRoots[39],
		&gRoots[40],
		&gRoots[41],
		&gRoots[42],
		&gRoots[43],
		&gRoots[44],
		&gRoots[45],
		&gRoots[46],
		&gRoots[47],
		&gRoots[48],
		&gRoots[49],
		&gRoots[50],
		&gRoots[51],
		&gRoots[52],
		&gRoots[53],
		&gRoots[54],
		&gRoots[55],
		&gRoots[56],
		&gRoots[57],
		&gRoots[58],
		&gRoots[59],
		&gRoots[60],
		&gRoots[61],
		&gRoots[62],
		&gRoots[63],
};

int CProfileManager::FrameCounter = 0;
unsigned long int CProfileManager::ResetTime = 0;

CProfileIterator* CProfileManager::Get_Iterator(void)
{
	int threadIndex = btQuickprofGetCurrentThreadIndex2();
	if ((threadIndex < 0) || threadIndex >= BT_QUICKPROF_MAX_THREAD_COUNT)
		return 0;

	return new CProfileIterator(&gRoots[threadIndex]);
}

void CProfileManager::CleanupMemory(void)
{
	for (int i = 0; i < BT_QUICKPROF_MAX_THREAD_COUNT; i++)
	{
		gRoots[i].CleanupMemory();
	}
}

/***********************************************************************************************
 * CProfileManager::Start_Profile -- Begin a named profile                                    *
 *                                                                                             *
 * Steps one level deeper into the tree, if a child already exists with the specified name     *
 * then it accumulates the profiling; otherwise a new child node is added to the profile tree. *
 *                                                                                             *
 * INPUT:                                                                                      *
 * name - name of this profiling record                                                        *
 *                                                                                             *
 * WARNINGS:                                                                                   *
 * The string used is assumed to be a static string; pointer compares are used throughout      *
 * the profiling code for efficiency.                                                          *
 *=============================================================================================*/
void CProfileManager::Start_Profile(const char* name)
{
	int threadIndex = btQuickprofGetCurrentThreadIndex2();
	if ((threadIndex < 0) || threadIndex >= BT_QUICKPROF_MAX_THREAD_COUNT)
		return;

	if (name != gCurrentNodes[threadIndex]->Get_Name())
	{
		gCurrentNodes[threadIndex] = gCurrentNodes[threadIndex]->Get_Sub_Node(name);
	}

	gCurrentNodes[threadIndex]->Call();
}

/***********************************************************************************************
 * CProfileManager::Stop_Profile -- Stop timing and record the results.                       *
 *=============================================================================================*/
void CProfileManager::Stop_Profile(void)
{
	int threadIndex = btQuickprofGetCurrentThreadIndex2();
	if ((threadIndex < 0) || threadIndex >= BT_QUICKPROF_MAX_THREAD_COUNT)
		return;

	// Return will indicate whether we should back up to our parent (we may
	// be profiling a recursive function)
	if (gCurrentNodes[threadIndex]->Return())
	{
		gCurrentNodes[threadIndex] = gCurrentNodes[threadIndex]->Get_Parent();
	}
}

/***********************************************************************************************
 * CProfileManager::Reset -- Reset the contents of the profiling system                       *
 *                                                                                             *
 *    This resets everything except for the tree structure.  All of the timing data is reset.  *
 *=============================================================================================*/
void CProfileManager::Reset(void)
{
	gProfileClock.reset();
	int threadIndex = btQuickprofGetCurrentThreadIndex2();
	if ((threadIndex < 0) || threadIndex >= BT_QUICKPROF_MAX_THREAD_COUNT)
		return;
	gRoots[threadIndex].Reset();
	gRoots[threadIndex].Call();
	FrameCounter = 0;
	Profile_Get_Ticks(&ResetTime);
}

/***********************************************************************************************
 * CProfileManager::Increment_Frame_Counter -- Increment the frame counter                    *
 *=============================================================================================*/
void CProfileManager::Increment_Frame_Counter(void)
{
	FrameCounter++;
}

/***********************************************************************************************
 * CProfileManager::Get_Time_Since_Reset -- returns the elapsed time since last reset         *
 *=============================================================================================*/
float CProfileManager::Get_Time_Since_Reset(void)
{
	unsigned long int time;
	Profile_Get_Ticks(&time);
	time -= ResetTime;
	return (float)time / Profile_Get_Tick_Rate();
}

#include <stdio.h>

void CProfileManager::dumpRecursive(CProfileIterator* profileIterator, int spacing)
{
	profileIterator->First();
	if (profileIterator->Is_Done())
		return;

	float accumulated_time = 0, parent_time = profileIterator->Is_Root() ? CProfileManager::Get_Time_Since_Reset() : profileIterator->Get_Current_Parent_Total_Time();
	int i;
	int frames_since_reset = CProfileManager::Get_Frame_Count_Since_Reset();
	for (i = 0; i < spacing; i++) printf(".");
	printf("----------------------------------\n");
	for (i = 0; i < spacing; i++) printf(".");
	printf("Profiling: %s (total running time: %.3f ms) ---\n", profileIterator->Get_Current_Parent_Name(), parent_time);
	float totalTime = 0.f;

	int numChildren = 0;

	for (i = 0; !profileIterator->Is_Done(); i++, profileIterator->Next())
	{
		numChildren++;
		float current_total_time = profileIterator->Get_Current_Total_Time();
		accumulated_time += current_total_time;
		float fraction = parent_time > SIMD_EPSILON ? (current_total_time / parent_time) * 100 : 0.f;
		{
			int i;
			for (i = 0; i < spacing; i++) printf(".");
		}
		printf("%d -- %s (%.2f %%) :: %.3f ms / frame (%d calls)\n", i, profileIterator->Get_Current_Name(), fraction, (current_total_time / (double)frames_since_reset), profileIterator->Get_Current_Total_Calls());
		totalTime += current_total_time;
		//recurse into children
	}

	if (parent_time < accumulated_time)
	{
		//printf("what's wrong\n");
	}
	for (i = 0; i < spacing; i++) printf(".");
	printf("%s (%.3f %%) :: %.3f ms\n", "Unaccounted:", parent_time > SIMD_EPSILON ? ((parent_time - accumulated_time) / parent_time) * 100 : 0.f, parent_time - accumulated_time);

	for (i = 0; i < numChildren; i++)
	{
		profileIterator->Enter_Child(i);
		dumpRecursive(profileIterator, spacing + 3);
		profileIterator->Enter_Parent();
	}
}

void CProfileManager::dumpAll()
{
	CProfileIterator* profileIterator = 0;
	profileIterator = CProfileManager::Get_Iterator();

	dumpRecursive(profileIterator, 0);

	CProfileManager::Release_Iterator(profileIterator);
}


void btEnterProfileZoneDefault(const char* name)
{
}
void btLeaveProfileZoneDefault()
{
}

#else
void btEnterProfileZoneDefault(const char* name)
{
}
void btLeaveProfileZoneDefault()
{
}
#endif  //BT_NO_PROFILE


// clang-format off
#if defined(_WIN32) && (defined(__MINGW32__) || defined(__MINGW64__))
  #define BT_HAVE_TLS 1
#elif __APPLE__ && !TARGET_OS_IPHONE
  // TODO: Modern versions of iOS support TLS now with updated version checking.
  #define BT_HAVE_TLS 1
#elif __linux__
  #define BT_HAVE_TLS 1
#elif defined(__FreeBSD__) || defined(__NetBSD__)
  // TODO: At the moment disabling purposely OpenBSD, albeit tls support exists but not fully functioning
  #define BT_HAVE_TLS 1
#endif

// __thread is broken on Andorid clang until r12b. See
// https://github.com/android-ndk/ndk/issues/8
#if defined(__ANDROID__) && defined(__clang__)
  #if __has_include(<android/ndk-version.h>)
    #include <android/ndk-version.h>
  #endif  // __has_include(<android/ndk-version.h>)
  #if defined(__NDK_MAJOR__) && \
    ((__NDK_MAJOR__ < 12) || ((__NDK_MAJOR__ == 12) && (__NDK_MINOR__ < 1)))
    #undef BT_HAVE_TLS
  #endif
#endif  // defined(__ANDROID__) && defined(__clang__)
// clang-format on

unsigned int btQuickprofGetCurrentThreadIndex2()
{
	const unsigned int kNullIndex = ~0U;

#if BT_THREADSAFE
	return btGetCurrentThreadIndex();
#else
#if defined(BT_HAVE_TLS)
	static __thread unsigned int sThreadIndex = kNullIndex;
#elif defined(_WIN32)
	__declspec(thread) static unsigned int sThreadIndex = kNullIndex;
#else
	unsigned int sThreadIndex = 0;
	return -1;
#endif

	static int gThreadCounter = 0;

	if (sThreadIndex == kNullIndex)
	{
		sThreadIndex = gThreadCounter++;
	}
	return sThreadIndex;
#endif  //BT_THREADSAFE
}

static btEnterProfileZoneFunc* bts_enterFunc = btEnterProfileZoneDefault;
static btLeaveProfileZoneFunc* bts_leaveFunc = btLeaveProfileZoneDefault;

void btEnterProfileZone(const char* name)
{
	(bts_enterFunc)(name);
}
void btLeaveProfileZone()
{
	(bts_leaveFunc)();
}

btEnterProfileZoneFunc* btGetCurrentEnterProfileZoneFunc()
{
	return bts_enterFunc;
}
btLeaveProfileZoneFunc* btGetCurrentLeaveProfileZoneFunc()
{
	return bts_leaveFunc;
}

void btSetCustomEnterProfileZoneFunc(btEnterProfileZoneFunc* enterFunc)
{
	bts_enterFunc = enterFunc;
}
void btSetCustomLeaveProfileZoneFunc(btLeaveProfileZoneFunc* leaveFunc)
{
	bts_leaveFunc = leaveFunc;
}

CProfileSample::CProfileSample(const char* name)
{
	btEnterProfileZone(name);
}

CProfileSample::~CProfileSample(void)
{
	btLeaveProfileZone();
}
