/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Platform Specific Functions, Win32 Implementation */

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN

#include <windows.h>

#include "platform.h"
#include "openhmdi.h"

double ohmd_get_tick()
{
	double high, low;
	FILETIME filetime;

	GetSystemTimeAsFileTime(&filetime);

	high = filetime.dwHighDateTime;
	low = filetime.dwLowDateTime;

	return (high * 4294967296.0 + low) / 10000000;
}

static const uint64_t NUM_10_000_000 = 10000000;

void ohmd_monotonic_init(ohmd_context* ctx)
{
	ctx->monotonic_ticks_per_sec = NUM_10_000_000;
}

uint64_t ohmd_monotonic_get(ohmd_context* ctx)
{
	FILETIME filetime;
	GetSystemTimeAsFileTime(&filetime);

	return ((uint64_t)filetime.dwHighDateTime << 32) | filetime.dwLowDateTime;
}

// TODO higher resolution
void ohmd_sleep(double seconds)
{
	Sleep((DWORD)(seconds * 1000));
}

// threads

struct ohmd_thread {
	HANDLE handle;
	void* arg;
	unsigned int (*routine)(void* arg);
};

struct ohmd_mutex {
	HANDLE handle;
};

DWORD __stdcall ohmd_thread_wrapper(void* t)
{
	ohmd_thread* thread = (ohmd_thread*)t;
	return thread->routine(thread->arg);
}

ohmd_thread* ohmd_create_thread(ohmd_context* ctx, unsigned int (*routine)(void* arg), void* arg)
{
	ohmd_thread* thread = ohmd_alloc(ctx, sizeof(ohmd_thread));
	if(!thread)
		return NULL;

	thread->routine = routine;
	thread->arg = arg;

	thread->handle = CreateThread(NULL, 0, ohmd_thread_wrapper, thread, 0, NULL);

	return thread;
}

void ohmd_destroy_thread(ohmd_thread* thread)
{
	ohmd_sleep(3);
	WaitForSingleObject(thread->handle, INFINITE);
	CloseHandle(thread->handle);
	free(thread);
}

ohmd_mutex* ohmd_create_mutex(ohmd_context* ctx)
{
	ohmd_mutex* mutex = ohmd_alloc(ctx, sizeof(ohmd_mutex));
	if(!mutex)
		return NULL;
	
	mutex->handle = CreateMutex(NULL, FALSE, NULL);

	return mutex;
}

void ohmd_destroy_mutex(ohmd_mutex* mutex)
{
	CloseHandle(mutex->handle);
	free(mutex);
}

void ohmd_lock_mutex(ohmd_mutex* mutex)
{
	if(mutex)
		WaitForSingleObject(mutex->handle, INFINITE);
}

void ohmd_unlock_mutex(ohmd_mutex* mutex)
{
	if(mutex)
		ReleaseMutex(mutex->handle);
}

int findEndPoint(char* path, int endpoint)
{
	char comp[8];
	sprintf(comp,"mi_0%d",endpoint);

	if (strstr(path, comp) != NULL) {
		return 1;
	}

	return 0;
}

/// Handling ovr service
static int _enable_ovr_service = 0;

void ohmd_toggle_ovr_service(int state) //State is 0 for Disable, 1 for Enable
{
	SC_HANDLE serviceDbHandle = OpenSCManager(NULL,NULL,SC_MANAGER_ALL_ACCESS);
	SC_HANDLE serviceHandle = OpenService(serviceDbHandle, "OVRService", SC_MANAGER_ALL_ACCESS);

	SERVICE_STATUS_PROCESS status;
	DWORD bytesNeeded;
	QueryServiceStatusEx(serviceHandle, SC_STATUS_PROCESS_INFO,(LPBYTE) &status,sizeof(SERVICE_STATUS_PROCESS), &bytesNeeded);

	if (state == 0 || status.dwCurrentState == SERVICE_RUNNING)
	{
		// Stop it
		BOOL b = ControlService(serviceHandle, SERVICE_CONTROL_STOP, (LPSERVICE_STATUS) &status);
		if (b)
		{
			printf("OVRService stopped\n");
			_enable_ovr_service = 1;
		}
		else 
			printf("Error: OVRService failed to stop, please try running with Administrator rights\n");
	}
	else if (state == 1 && _enable_ovr_service)
	{
		// Start it 
		BOOL b = StartService(serviceHandle, NULL, NULL); 
		if (b) 
			printf("OVRService started\n");
		else 
			printf("Error: OVRService failed to start, please try running with Administrator rights\n");
	} 
	CloseServiceHandle(serviceHandle); 
	CloseServiceHandle(serviceDbHandle); 
}
#endif
