/* Snippets extracted from https://github.com/Microsoft/openssl/blob/ec7e430e06e4e3ac87c183dee33cb216814cf980/ms/winrt.cpp
 * Adapted for Godot definitions
 */
/* uwp.cpp
 * Copyright 2014 Microsoft Corporation
 * C++/CX Entropy/shims for Windows Phone/Windows Store platform
 * written by Alejandro Jimenez Martinez
 * (aljim@microsoft.com) for the OpenSSL project 2014.
 */

#include <windows.h>
#if defined(WINAPI_FAMILY)
extern "C"
{
	unsigned entropyRT(BYTE *buffer, unsigned len);
	void RAND_add(const void *buf,int num,double entropy);
	int RAND_poll(void);
}
#endif

unsigned entropyRT(BYTE *buffer, unsigned len)
	{
	using namespace Platform;
	using namespace Windows::Foundation;
	using namespace Windows::Foundation::Collections;
	using namespace Windows::Security::Cryptography;
	using namespace Windows::Storage::Streams;
	IBuffer ^buf = CryptographicBuffer::GenerateRandom(len);
	Array<unsigned char> ^arr;
	CryptographicBuffer::CopyToByteArray(buf, &arr);
	unsigned arrayLen = arr->Length;

	// Make sure not to overflow the copy
	arrayLen = (arrayLen > len) ? len : arrayLen;
	memcpy(buffer, arr->Data, arrayLen);
	return arrayLen;
	}

int RAND_poll(void)
	{
	BYTE buf[60];
	unsigned collected = entropyRT(buf , sizeof(buf));
	RAND_add(buf, collected, collected);
	return 1;
	}

#if defined(UWP_ENABLED)
extern "C"
{
#include<stdio.h>
#include<string.h>
#include<stdlib.h>

	void* GetModuleHandle(
						 _In_opt_  LPCTSTR lpModuleName
						 )
		{
		return NULL;
		}
	//no log for phone
	int RegisterEventSource(
						   _In_  LPCTSTR lpUNCServerName,
						   _In_  LPCTSTR lpSourceName
						   )
		{
		return NULL;
		}

	int ReportEvent(
				   _In_  HANDLE hEventLog,
				   _In_  WORD wType,
				   _In_  WORD wCategory,
				   _In_  DWORD dwEventID,
				   _In_  PSID lpUserSid,
				   _In_  WORD wNumStrings,
				   _In_  DWORD dwDataSize,
				   _In_  LPCTSTR *lpStrings,
				   _In_  LPVOID lpRawData
				   )
		{
		return 0;
		}
	int MessageBox(
				  _In_opt_  HWND hWnd,
				  _In_opt_  LPCTSTR lpText,
				  _In_opt_  LPCTSTR lpCaption,
				  _In_      UINT uType
				  )
		{
		return 0;
		}
	int __cdecl GetProcessWindowStation(void)
		{
		return NULL;
		}
	BOOL __cdecl GetUserObjectInformationW(
										 _In_       HANDLE hObj,
										 _In_       int nIndex,
										 _Out_opt_  PVOID pvInfo,
										 _In_       DWORD nLength,
										 _Out_opt_  LPDWORD lpnLengthNeeded
										 )
		{
		return 0;
		}
#ifndef STD_ERROR_HANDLE
	int __cdecl GetStdHandle(
						   _In_  DWORD nStdHandle
						   )
		{
		return 0;
		}
#endif
	BOOL DeregisterEventSource(
							  _Inout_  HANDLE hEventLog
							  )
		{
		return 0;
		}
	char *getenv(
					  const char *varname
					  )
		{
		//hardcoded environmental variables used for the appx testing application for store/phone
		if (!strcmp(varname, "OPENSSL_CONF"))
			{
			return "./openssl.cnf";
			}
		return 0;
		}
	int setenv(const char *envname, const char *envval, int overwrite)
		{
		return -1;
		}
	int _getch(void)
		{
		return 0;
		}
	int _kbhit()
		{
		return 0;
		}
	BOOL __cdecl FlushConsoleInputBuffer(
									   _In_  HANDLE hConsoleInput
									   )
		{
		return 0;
		}
	int uwp_GetTickCount(void)
		{
		LARGE_INTEGER t;
		return(int) (QueryPerformanceCounter(&t) ? t.QuadPart : 0);
		}
	void *OPENSSL_UplinkTable [26]= {0};
} //extern C

#endif /*defined(UWP_ENABLED)*/
