/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2018 Erwin Coumans  http://bulletphysics.com

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#if defined( _WIN32 ) &&  BT_THREADSAFE

#include "LinearMath/btScalar.h"
#include "LinearMath/btMinMax.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btThreads.h"
#include "btThreadSupportInterface.h"
#include <windows.h>
#include <stdio.h>


struct btProcessorInfo
{
    int numLogicalProcessors;
    int numCores;
    int numNumaNodes;
    int numL1Cache;
    int numL2Cache;
    int numL3Cache;
    int numPhysicalPackages;
    static const int maxNumTeamMasks = 32;
    int numTeamMasks;
    UINT64 processorTeamMasks[ maxNumTeamMasks ];
};

UINT64 getProcessorTeamMask( const btProcessorInfo& procInfo, int procId )
{
    UINT64 procMask = UINT64( 1 ) << procId;
    for ( int i = 0; i < procInfo.numTeamMasks; ++i )
    {
        if ( procMask & procInfo.processorTeamMasks[ i ] )
        {
            return procInfo.processorTeamMasks[ i ];
        }
    }
    return 0;
}

int getProcessorTeamIndex( const btProcessorInfo& procInfo, int procId )
{
    UINT64 procMask = UINT64( 1 ) << procId;
    for ( int i = 0; i < procInfo.numTeamMasks; ++i )
    {
        if ( procMask & procInfo.processorTeamMasks[ i ] )
        {
            return i;
        }
    }
    return -1;
}

int countSetBits( ULONG64 bits )
{
    int count = 0;
    while ( bits )
    {
        if ( bits & 1 )
        {
            count++;
        }
        bits >>= 1;
    }
    return count;
}


typedef BOOL( WINAPI *Pfn_GetLogicalProcessorInformation )( PSYSTEM_LOGICAL_PROCESSOR_INFORMATION, PDWORD );


void getProcessorInformation( btProcessorInfo* procInfo )
{
    memset( procInfo, 0, sizeof( *procInfo ) );
    Pfn_GetLogicalProcessorInformation getLogicalProcInfo =
        (Pfn_GetLogicalProcessorInformation) GetProcAddress( GetModuleHandle( TEXT( "kernel32" ) ), "GetLogicalProcessorInformation" );
    if ( getLogicalProcInfo == NULL )
    {
        // no info
        return;
    }
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buf = NULL;
    DWORD bufSize = 0;
    while ( true )
    {
        if ( getLogicalProcInfo( buf, &bufSize ) )
        {
            break;
        }
        else
        {
            if ( GetLastError() == ERROR_INSUFFICIENT_BUFFER )
            {
                if ( buf )
                {
                    free( buf );
                }
                buf = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION) malloc( bufSize );
            }
        }
    }

    int len = bufSize / sizeof( *buf );
    for ( int i = 0; i < len; ++i )
    {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION info = buf + i;
        switch ( info->Relationship )
        {
        case RelationNumaNode:
            procInfo->numNumaNodes++;
            break;

        case RelationProcessorCore:
            procInfo->numCores++;
            procInfo->numLogicalProcessors += countSetBits( info->ProcessorMask );
            break;

        case RelationCache:
            if ( info->Cache.Level == 1 )
            {
                procInfo->numL1Cache++;
            }
            else if ( info->Cache.Level == 2 )
            {
                procInfo->numL2Cache++;
            }
            else if ( info->Cache.Level == 3 )
            {
                procInfo->numL3Cache++;
                // processors that share L3 cache are considered to be on the same team
                // because they can more easily work together on the same data.
                // Large performance penalties will occur if 2 or more threads from different
                // teams attempt to frequently read and modify the same cache lines.
                //
                // On the AMD Ryzen 7 CPU for example, the 8 cores on the CPU are split into
                // 2 CCX units of 4 cores each. Each CCX has a separate L3 cache, so if both
                // CCXs are operating on the same data, many cycles will be spent keeping the
                // two caches coherent.
                if ( procInfo->numTeamMasks < btProcessorInfo::maxNumTeamMasks )
                {
                    procInfo->processorTeamMasks[ procInfo->numTeamMasks ] = info->ProcessorMask;
                    procInfo->numTeamMasks++;
                }
            }
            break;

        case RelationProcessorPackage:
            procInfo->numPhysicalPackages++;
            break;
        }
    }
    free( buf );
}



///btThreadSupportWin32 helps to initialize/shutdown libspe2, start/stop SPU tasks and communication
class btThreadSupportWin32 : public btThreadSupportInterface
{
public:
    struct btThreadStatus
    {
        int m_taskId;
        int m_commandId;
        int m_status;

        ThreadFunc m_userThreadFunc;
        void* m_userPtr; //for taskDesc etc

        void* m_threadHandle; //this one is calling 'Win32ThreadFunc'

        void* m_eventStartHandle;
        char m_eventStartHandleName[ 32 ];

        void* m_eventCompleteHandle;
        char m_eventCompleteHandleName[ 32 ];
    };

private:
    btAlignedObjectArray<btThreadStatus> m_activeThreadStatus;
    btAlignedObjectArray<void*> m_completeHandles;
    int m_numThreads;
    DWORD_PTR m_startedThreadMask;
    btProcessorInfo m_processorInfo;

    void startThreads( const ConstructionInfo& threadInfo );
    void stopThreads();
    int waitForResponse();

public:

    btThreadSupportWin32( const ConstructionInfo& threadConstructionInfo );
    virtual ~btThreadSupportWin32();

    virtual int getNumWorkerThreads() const BT_OVERRIDE { return m_numThreads; }
    virtual int getCacheFriendlyNumThreads() const BT_OVERRIDE { return countSetBits(m_processorInfo.processorTeamMasks[0]); }
    virtual int getLogicalToPhysicalCoreRatio() const BT_OVERRIDE { return m_processorInfo.numLogicalProcessors / m_processorInfo.numCores; }

    virtual void runTask( int threadIndex, void* userData ) BT_OVERRIDE;
    virtual void waitForAllTasks() BT_OVERRIDE;

    virtual btCriticalSection* createCriticalSection() BT_OVERRIDE;
    virtual void deleteCriticalSection( btCriticalSection* criticalSection ) BT_OVERRIDE;
};


btThreadSupportWin32::btThreadSupportWin32( const ConstructionInfo & threadConstructionInfo )
{
    startThreads( threadConstructionInfo );
}


btThreadSupportWin32::~btThreadSupportWin32()
{
    stopThreads();
}


DWORD WINAPI win32threadStartFunc( LPVOID lpParam )
{
    btThreadSupportWin32::btThreadStatus* status = ( btThreadSupportWin32::btThreadStatus* )lpParam;

    while ( 1 )
    {
        WaitForSingleObject( status->m_eventStartHandle, INFINITE );
        void* userPtr = status->m_userPtr;

        if ( userPtr )
        {
            btAssert( status->m_status );
            status->m_userThreadFunc( userPtr );
            status->m_status = 2;
            SetEvent( status->m_eventCompleteHandle );
        }
        else
        {
            //exit Thread
            status->m_status = 3;
            printf( "Thread with taskId %i with handle %p exiting\n", status->m_taskId, status->m_threadHandle );
            SetEvent( status->m_eventCompleteHandle );
            break;
        }
    }
    printf( "Thread TERMINATED\n" );
    return 0;
}


void btThreadSupportWin32::runTask( int threadIndex, void* userData )
{
    btThreadStatus& threadStatus = m_activeThreadStatus[ threadIndex ];
    btAssert( threadIndex >= 0 );
    btAssert( int( threadIndex ) < m_activeThreadStatus.size() );

    threadStatus.m_commandId = 1;
    threadStatus.m_status = 1;
    threadStatus.m_userPtr = userData;
    m_startedThreadMask |= DWORD_PTR( 1 ) << threadIndex;

    ///fire event to start new task
    SetEvent( threadStatus.m_eventStartHandle );
}


int btThreadSupportWin32::waitForResponse()
{
    btAssert( m_activeThreadStatus.size() );

    int last = -1;
    DWORD res = WaitForMultipleObjects( m_completeHandles.size(), &m_completeHandles[ 0 ], FALSE, INFINITE );
    btAssert( res != WAIT_FAILED );
    last = res - WAIT_OBJECT_0;

    btThreadStatus& threadStatus = m_activeThreadStatus[ last ];
    btAssert( threadStatus.m_threadHandle );
    btAssert( threadStatus.m_eventCompleteHandle );

    //WaitForSingleObject(threadStatus.m_eventCompleteHandle, INFINITE);
    btAssert( threadStatus.m_status > 1 );
    threadStatus.m_status = 0;

    ///need to find an active spu
    btAssert( last >= 0 );
    m_startedThreadMask &= ~( DWORD_PTR( 1 ) << last );

    return last;
}


void btThreadSupportWin32::waitForAllTasks()
{
    while ( m_startedThreadMask )
    {
        waitForResponse();
    }
}


void btThreadSupportWin32::startThreads( const ConstructionInfo& threadConstructionInfo )
{
    static int uniqueId = 0;
    uniqueId++;
    btProcessorInfo& procInfo = m_processorInfo;
    getProcessorInformation( &procInfo );
    DWORD_PTR dwProcessAffinityMask = 0;
    DWORD_PTR dwSystemAffinityMask = 0;
    if ( !GetProcessAffinityMask( GetCurrentProcess(), &dwProcessAffinityMask, &dwSystemAffinityMask ) )
    {
        dwProcessAffinityMask = 0;
    }
    ///The number of threads should be equal to the number of available cores - 1
    m_numThreads = btMin(procInfo.numLogicalProcessors, int(BT_MAX_THREAD_COUNT)) - 1; // cap to max thread count (-1 because main thread already exists)

    m_activeThreadStatus.resize( m_numThreads );
    m_completeHandles.resize( m_numThreads );
    m_startedThreadMask = 0;

    // set main thread affinity
    if ( DWORD_PTR mask = dwProcessAffinityMask & getProcessorTeamMask( procInfo, 0 ))
    {
        SetThreadAffinityMask( GetCurrentThread(), mask );
        SetThreadIdealProcessor( GetCurrentThread(), 0 );
    }

    for ( int i = 0; i < m_numThreads; i++ )
    {
        printf( "starting thread %d\n", i );

        btThreadStatus& threadStatus = m_activeThreadStatus[ i ];

        LPSECURITY_ATTRIBUTES lpThreadAttributes = NULL;
        SIZE_T dwStackSize = threadConstructionInfo.m_threadStackSize;
        LPTHREAD_START_ROUTINE lpStartAddress = &win32threadStartFunc;
        LPVOID lpParameter = &threadStatus;
        DWORD dwCreationFlags = 0;
        LPDWORD lpThreadId = 0;

        threadStatus.m_userPtr = 0;

        sprintf( threadStatus.m_eventStartHandleName, "es%.8s%d%d", threadConstructionInfo.m_uniqueName, uniqueId, i );
        threadStatus.m_eventStartHandle = CreateEventA( 0, false, false, threadStatus.m_eventStartHandleName );

        sprintf( threadStatus.m_eventCompleteHandleName, "ec%.8s%d%d", threadConstructionInfo.m_uniqueName, uniqueId, i );
        threadStatus.m_eventCompleteHandle = CreateEventA( 0, false, false, threadStatus.m_eventCompleteHandleName );

        m_completeHandles[ i ] = threadStatus.m_eventCompleteHandle;

        HANDLE handle = CreateThread( lpThreadAttributes, dwStackSize, lpStartAddress, lpParameter, dwCreationFlags, lpThreadId );
        //SetThreadPriority( handle, THREAD_PRIORITY_HIGHEST );
        // highest priority -- can cause erratic performance when numThreads > numCores
        //                     we don't want worker threads to be higher priority than the main thread or the main thread could get
        //                     totally shut out and unable to tell the workers to stop
        //SetThreadPriority( handle, THREAD_PRIORITY_BELOW_NORMAL );

        {
            int processorId = i + 1;  // leave processor 0 for main thread
            DWORD_PTR teamMask = getProcessorTeamMask( procInfo, processorId );
            if ( teamMask )
            {
                // bind each thread to only execute on processors of it's assigned team
                //  - for single-socket Intel x86 CPUs this has no effect (only a single, shared L3 cache so there is only 1 team)
                //  - for multi-socket Intel this will keep threads from migrating from one socket to another
                //  - for AMD Ryzen this will keep threads from migrating from one CCX to another
                DWORD_PTR mask = teamMask & dwProcessAffinityMask;
                if ( mask )
                {
                    SetThreadAffinityMask( handle, mask );
                }
            }
            SetThreadIdealProcessor( handle, processorId );
        }

        threadStatus.m_taskId = i;
        threadStatus.m_commandId = 0;
        threadStatus.m_status = 0;
        threadStatus.m_threadHandle = handle;
        threadStatus.m_userThreadFunc = threadConstructionInfo.m_userThreadFunc;

        printf( "started %s thread %d with threadHandle %p\n", threadConstructionInfo.m_uniqueName, i, handle );
    }
}

///tell the task scheduler we are done with the SPU tasks
void btThreadSupportWin32::stopThreads()
{
    for ( int i = 0; i < m_activeThreadStatus.size(); i++ )
    {
        btThreadStatus& threadStatus = m_activeThreadStatus[ i ];
        if ( threadStatus.m_status > 0 )
        {
            WaitForSingleObject( threadStatus.m_eventCompleteHandle, INFINITE );
        }

        threadStatus.m_userPtr = NULL;
        SetEvent( threadStatus.m_eventStartHandle );
        WaitForSingleObject( threadStatus.m_eventCompleteHandle, INFINITE );

        CloseHandle( threadStatus.m_eventCompleteHandle );
        CloseHandle( threadStatus.m_eventStartHandle );
        CloseHandle( threadStatus.m_threadHandle );

    }

    m_activeThreadStatus.clear();
    m_completeHandles.clear();
}


class btWin32CriticalSection : public btCriticalSection
{
private:
    CRITICAL_SECTION mCriticalSection;

public:
    btWin32CriticalSection()
    {
        InitializeCriticalSection( &mCriticalSection );
    }

    ~btWin32CriticalSection()
    {
        DeleteCriticalSection( &mCriticalSection );
    }

    void lock()
    {
        EnterCriticalSection( &mCriticalSection );
    }

    void unlock()
    {
        LeaveCriticalSection( &mCriticalSection );
    }
};


btCriticalSection* btThreadSupportWin32::createCriticalSection()
{
    unsigned char* mem = (unsigned char*) btAlignedAlloc( sizeof( btWin32CriticalSection ), 16 );
    btWin32CriticalSection* cs = new( mem ) btWin32CriticalSection();
    return cs;
}

void btThreadSupportWin32::deleteCriticalSection( btCriticalSection* criticalSection )
{
    criticalSection->~btCriticalSection();
    btAlignedFree( criticalSection );
}


btThreadSupportInterface* btThreadSupportInterface::create( const ConstructionInfo& info )
{
    return new btThreadSupportWin32( info );
}



#endif //defined(_WIN32) && BT_THREADSAFE

