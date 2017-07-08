/*
 * windows backend for libusb 1.0
 * Copyright © 2009-2012 Pete Batard <pete@akeo.ie>
 * With contributions from Michael Plante, Orin Eman et al.
 * Parts of this code adapted from libusb-win32-v1 by Stephan Meyer
 * HID Reports IOCTLs inspired from HIDAPI by Alan Ott, Signal 11 Software
 * Hash table functions adapted from glibc, by Ulrich Drepper et al.
 * Major code testing contribution by Xiaofan Chen
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <config.h>

#include <inttypes.h>
#include <process.h>
#include <stdio.h>

#include "libusbi.h"
#include "windows_common.h"
#include "windows_nt_common.h"

// Global variables for clock_gettime mechanism
static uint64_t hires_ticks_to_ps;
static uint64_t hires_frequency;

#define TIMER_REQUEST_RETRY_MS	100
#define WM_TIMER_REQUEST	(WM_USER + 1)
#define WM_TIMER_EXIT		(WM_USER + 2)

// used for monotonic clock_gettime()
struct timer_request {
	struct timespec *tp;
	HANDLE event;
};

// Timer thread
static HANDLE timer_thread = NULL;
static DWORD timer_thread_id = 0;

/* User32 dependencies */
DLL_DECLARE_HANDLE(User32);
DLL_DECLARE_FUNC_PREFIXED(WINAPI, BOOL, p, GetMessageA, (LPMSG, HWND, UINT, UINT));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, BOOL, p, PeekMessageA, (LPMSG, HWND, UINT, UINT, UINT));
DLL_DECLARE_FUNC_PREFIXED(WINAPI, BOOL, p, PostThreadMessageA, (DWORD, UINT, WPARAM, LPARAM));

static unsigned __stdcall windows_clock_gettime_threaded(void *param);

/*
* Converts a windows error to human readable string
* uses retval as errorcode, or, if 0, use GetLastError()
*/
#if defined(ENABLE_LOGGING)
const char *windows_error_str(DWORD error_code)
{
	static char err_string[ERR_BUFFER_SIZE];

	DWORD size;
	int len;

	if (error_code == 0)
		error_code = GetLastError();

	len = sprintf(err_string, "[%u] ", (unsigned int)error_code);

	// Translate codes returned by SetupAPI. The ones we are dealing with are either
	// in 0x0000xxxx or 0xE000xxxx and can be distinguished from standard error codes.
	// See http://msdn.microsoft.com/en-us/library/windows/hardware/ff545011.aspx
	switch (error_code & 0xE0000000) {
	case 0:
		error_code = HRESULT_FROM_WIN32(error_code); // Still leaves ERROR_SUCCESS unmodified
		break;
	case 0xE0000000:
		error_code = 0x80000000 | (FACILITY_SETUPAPI << 16) | (error_code & 0x0000FFFF);
		break;
	default:
		break;
	}

	size = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			&err_string[len], ERR_BUFFER_SIZE - len, NULL);
	if (size == 0) {
		DWORD format_error = GetLastError();
		if (format_error)
			snprintf(err_string, ERR_BUFFER_SIZE,
				"Windows error code %u (FormatMessage error code %u)",
				(unsigned int)error_code, (unsigned int)format_error);
		else
			snprintf(err_string, ERR_BUFFER_SIZE, "Unknown error code %u", (unsigned int)error_code);
	} else {
		// Remove CRLF from end of message, if present
		size_t pos = len + size - 2;
		if (err_string[pos] == '\r')
			err_string[pos] = '\0';
	}

	return err_string;
}
#endif

/* Hash table functions - modified From glibc 2.3.2:
   [Aho,Sethi,Ullman] Compilers: Principles, Techniques and Tools, 1986
   [Knuth]            The Art of Computer Programming, part 3 (6.4)  */

#define HTAB_SIZE 1021UL	// *MUST* be a prime number!!

typedef struct htab_entry {
	unsigned long used;
	char *str;
} htab_entry;

static htab_entry *htab_table = NULL;
static usbi_mutex_t htab_mutex = NULL;
static unsigned long htab_filled;

/* Before using the hash table we must allocate memory for it.
   We allocate one element more as the found prime number says.
   This is done for more effective indexing as explained in the
   comment for the hash function.  */
static bool htab_create(struct libusb_context *ctx)
{
	if (htab_table != NULL) {
		usbi_err(ctx, "hash table already allocated");
		return true;
	}

	// Create a mutex
	usbi_mutex_init(&htab_mutex);

	usbi_dbg("using %lu entries hash table", HTAB_SIZE);
	htab_filled = 0;

	// allocate memory and zero out.
	htab_table = calloc(HTAB_SIZE + 1, sizeof(htab_entry));
	if (htab_table == NULL) {
		usbi_err(ctx, "could not allocate space for hash table");
		return false;
	}

	return true;
}

/* After using the hash table it has to be destroyed.  */
static void htab_destroy(void)
{
	unsigned long i;

	if (htab_table == NULL)
		return;

	for (i = 0; i < HTAB_SIZE; i++)
		free(htab_table[i].str);

	safe_free(htab_table);

	usbi_mutex_destroy(&htab_mutex);
}

/* This is the search function. It uses double hashing with open addressing.
   We use a trick to speed up the lookup. The table is created with one
   more element available. This enables us to use the index zero special.
   This index will never be used because we store the first hash index in
   the field used where zero means not used. Every other value means used.
   The used field can be used as a first fast comparison for equality of
   the stored and the parameter value. This helps to prevent unnecessary
   expensive calls of strcmp.  */
unsigned long htab_hash(const char *str)
{
	unsigned long hval, hval2;
	unsigned long idx;
	unsigned long r = 5381;
	int c;
	const char *sz = str;

	if (str == NULL)
		return 0;

	// Compute main hash value (algorithm suggested by Nokia)
	while ((c = *sz++) != 0)
		r = ((r << 5) + r) + c;
	if (r == 0)
		++r;

	// compute table hash: simply take the modulus
	hval = r % HTAB_SIZE;
	if (hval == 0)
		++hval;

	// Try the first index
	idx = hval;

	// Mutually exclusive access (R/W lock would be better)
	usbi_mutex_lock(&htab_mutex);

	if (htab_table[idx].used) {
		if ((htab_table[idx].used == hval) && (strcmp(str, htab_table[idx].str) == 0))
			goto out_unlock; // existing hash

		usbi_dbg("hash collision ('%s' vs '%s')", str, htab_table[idx].str);

		// Second hash function, as suggested in [Knuth]
		hval2 = 1 + hval % (HTAB_SIZE - 2);

		do {
			// Because size is prime this guarantees to step through all available indexes
			if (idx <= hval2)
				idx = HTAB_SIZE + idx - hval2;
			else
				idx -= hval2;

			// If we visited all entries leave the loop unsuccessfully
			if (idx == hval)
				break;

			// If entry is found use it.
			if ((htab_table[idx].used == hval) && (strcmp(str, htab_table[idx].str) == 0))
				goto out_unlock;
		} while (htab_table[idx].used);
	}

	// Not found => New entry

	// If the table is full return an error
	if (htab_filled >= HTAB_SIZE) {
		usbi_err(NULL, "hash table is full (%lu entries)", HTAB_SIZE);
		idx = 0;
		goto out_unlock;
	}

	htab_table[idx].str = _strdup(str);
	if (htab_table[idx].str == NULL) {
		usbi_err(NULL, "could not duplicate string for hash table");
		idx = 0;
		goto out_unlock;
	}

	htab_table[idx].used = hval;
	++htab_filled;

out_unlock:
	usbi_mutex_unlock(&htab_mutex);

	return idx;
}

static int windows_init_dlls(void)
{
	DLL_GET_HANDLE(User32);
	DLL_LOAD_FUNC_PREFIXED(User32, p, GetMessageA, TRUE);
	DLL_LOAD_FUNC_PREFIXED(User32, p, PeekMessageA, TRUE);
	DLL_LOAD_FUNC_PREFIXED(User32, p, PostThreadMessageA, TRUE);

	return LIBUSB_SUCCESS;
}

static void windows_exit_dlls(void)
{
	DLL_FREE_HANDLE(User32);
}

static bool windows_init_clock(struct libusb_context *ctx)
{
	DWORD_PTR affinity, dummy;
	HANDLE event = NULL;
	LARGE_INTEGER li_frequency;
	int i;

	if (QueryPerformanceFrequency(&li_frequency)) {
		// Load DLL imports
		if (windows_init_dlls() != LIBUSB_SUCCESS) {
			usbi_err(ctx, "could not resolve DLL functions");
			return false;
		}

		// The hires frequency can go as high as 4 GHz, so we'll use a conversion
		// to picoseconds to compute the tv_nsecs part in clock_gettime
		hires_frequency = li_frequency.QuadPart;
		hires_ticks_to_ps = UINT64_C(1000000000000) / hires_frequency;
		usbi_dbg("hires timer available (Frequency: %"PRIu64" Hz)", hires_frequency);

		// Because QueryPerformanceCounter might report different values when
		// running on different cores, we create a separate thread for the timer
		// calls, which we glue to the first available core always to prevent timing discrepancies.
		if (!GetProcessAffinityMask(GetCurrentProcess(), &affinity, &dummy) || (affinity == 0)) {
			usbi_err(ctx, "could not get process affinity: %s", windows_error_str(0));
			return false;
		}

		// The process affinity mask is a bitmask where each set bit represents a core on
		// which this process is allowed to run, so we find the first set bit
		for (i = 0; !(affinity & (DWORD_PTR)(1 << i)); i++);
		affinity = (DWORD_PTR)(1 << i);

		usbi_dbg("timer thread will run on core #%d", i);

		event = CreateEvent(NULL, FALSE, FALSE, NULL);
		if (event == NULL) {
			usbi_err(ctx, "could not create event: %s", windows_error_str(0));
			return false;
		}

		timer_thread = (HANDLE)_beginthreadex(NULL, 0, windows_clock_gettime_threaded, (void *)event,
				0, (unsigned int *)&timer_thread_id);
		if (timer_thread == NULL) {
			usbi_err(ctx, "unable to create timer thread - aborting");
			CloseHandle(event);
			return false;
		}

		if (!SetThreadAffinityMask(timer_thread, affinity))
			usbi_warn(ctx, "unable to set timer thread affinity, timer discrepancies may arise");

		// Wait for timer thread to init before continuing.
		if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0) {
			usbi_err(ctx, "failed to wait for timer thread to become ready - aborting");
			CloseHandle(event);
			return false;
		}

		CloseHandle(event);
	} else {
		usbi_dbg("no hires timer available on this platform");
		hires_frequency = 0;
		hires_ticks_to_ps = UINT64_C(0);
	}

	return true;
}

void windows_destroy_clock(void)
{
	if (timer_thread) {
		// actually the signal to quit the thread.
		if (!pPostThreadMessageA(timer_thread_id, WM_TIMER_EXIT, 0, 0)
				|| (WaitForSingleObject(timer_thread, INFINITE) != WAIT_OBJECT_0)) {
			usbi_dbg("could not wait for timer thread to quit");
			TerminateThread(timer_thread, 1);
			// shouldn't happen, but we're destroying
			// all objects it might have held anyway.
		}
		CloseHandle(timer_thread);
		timer_thread = NULL;
		timer_thread_id = 0;
	}
}

/*
* Monotonic and real time functions
*/
static unsigned __stdcall windows_clock_gettime_threaded(void *param)
{
	struct timer_request *request;
	LARGE_INTEGER hires_counter;
	MSG msg;

	// The following call will create this thread's message queue
	// See https://msdn.microsoft.com/en-us/library/windows/desktop/ms644946.aspx
	pPeekMessageA(&msg, NULL, WM_USER, WM_USER, PM_NOREMOVE);

	// Signal windows_init_clock() that we're ready to service requests
	if (!SetEvent((HANDLE)param))
		usbi_dbg("SetEvent failed for timer init event: %s", windows_error_str(0));
	param = NULL;

	// Main loop - wait for requests
	while (1) {
		if (pGetMessageA(&msg, NULL, WM_TIMER_REQUEST, WM_TIMER_EXIT) == -1) {
			usbi_err(NULL, "GetMessage failed for timer thread: %s", windows_error_str(0));
			return 1;
		}

		switch (msg.message) {
		case WM_TIMER_REQUEST:
			// Requests to this thread are for hires always
			// Microsoft says that this function always succeeds on XP and later
			// See https://msdn.microsoft.com/en-us/library/windows/desktop/ms644904.aspx
			request = (struct timer_request *)msg.lParam;
			QueryPerformanceCounter(&hires_counter);
			request->tp->tv_sec = (long)(hires_counter.QuadPart / hires_frequency);
			request->tp->tv_nsec = (long)(((hires_counter.QuadPart % hires_frequency) / 1000) * hires_ticks_to_ps);
			if (!SetEvent(request->event))
				usbi_err(NULL, "SetEvent failed for timer request: %s", windows_error_str(0));
			break;
		case WM_TIMER_EXIT:
			usbi_dbg("timer thread quitting");
			return 0;
		}
	}
}

int windows_clock_gettime(int clk_id, struct timespec *tp)
{
	struct timer_request request;
#if !defined(_MSC_VER) || (_MSC_VER < 1900)
	FILETIME filetime;
	ULARGE_INTEGER rtime;
#endif
	DWORD r;

	switch (clk_id) {
	case USBI_CLOCK_MONOTONIC:
		if (timer_thread) {
			request.tp = tp;
			request.event = CreateEvent(NULL, FALSE, FALSE, NULL);
			if (request.event == NULL)
				return LIBUSB_ERROR_NO_MEM;

			if (!pPostThreadMessageA(timer_thread_id, WM_TIMER_REQUEST, 0, (LPARAM)&request)) {
				usbi_err(NULL, "PostThreadMessage failed for timer thread: %s", windows_error_str(0));
				CloseHandle(request.event);
				return LIBUSB_ERROR_OTHER;
			}

			do {
				r = WaitForSingleObject(request.event, TIMER_REQUEST_RETRY_MS);
				if (r == WAIT_TIMEOUT)
					usbi_dbg("could not obtain a timer value within reasonable timeframe - too much load?");
				else if (r == WAIT_FAILED)
					usbi_err(NULL, "WaitForSingleObject failed: %s", windows_error_str(0));
			} while (r == WAIT_TIMEOUT);
			CloseHandle(request.event);

			if (r == WAIT_OBJECT_0)
				return LIBUSB_SUCCESS;
			else
				return LIBUSB_ERROR_OTHER;
		}
		// Fall through and return real-time if monotonic was not detected @ timer init
	case USBI_CLOCK_REALTIME:
#if defined(_MSC_VER) && (_MSC_VER >= 1900)
		timespec_get(tp, TIME_UTC);
#else
		// We follow http://msdn.microsoft.com/en-us/library/ms724928%28VS.85%29.aspx
		// with a predef epoch time to have an epoch that starts at 1970.01.01 00:00
		// Note however that our resolution is bounded by the Windows system time
		// functions and is at best of the order of 1 ms (or, usually, worse)
		GetSystemTimeAsFileTime(&filetime);
		rtime.LowPart = filetime.dwLowDateTime;
		rtime.HighPart = filetime.dwHighDateTime;
		rtime.QuadPart -= EPOCH_TIME;
		tp->tv_sec = (long)(rtime.QuadPart / 10000000);
		tp->tv_nsec = (long)((rtime.QuadPart % 10000000) * 100);
#endif
		return LIBUSB_SUCCESS;
	default:
		return LIBUSB_ERROR_INVALID_PARAM;
	}
}

static void windows_transfer_callback(struct usbi_transfer *itransfer, uint32_t io_result, uint32_t io_size)
{
	int status, istatus;

	usbi_dbg("handling I/O completion with errcode %u, size %u", io_result, io_size);

	switch (io_result) {
	case NO_ERROR:
		status = windows_copy_transfer_data(itransfer, io_size);
		break;
	case ERROR_GEN_FAILURE:
		usbi_dbg("detected endpoint stall");
		status = LIBUSB_TRANSFER_STALL;
		break;
	case ERROR_SEM_TIMEOUT:
		usbi_dbg("detected semaphore timeout");
		status = LIBUSB_TRANSFER_TIMED_OUT;
		break;
	case ERROR_OPERATION_ABORTED:
		istatus = windows_copy_transfer_data(itransfer, io_size);
		if (istatus != LIBUSB_TRANSFER_COMPLETED)
			usbi_dbg("Failed to copy partial data in aborted operation: %d", istatus);

		usbi_dbg("detected operation aborted");
		status = LIBUSB_TRANSFER_CANCELLED;
		break;
	default:
		usbi_err(ITRANSFER_CTX(itransfer), "detected I/O error %u: %s", io_result, windows_error_str(io_result));
		status = LIBUSB_TRANSFER_ERROR;
		break;
	}
	windows_clear_transfer_priv(itransfer);	// Cancel polling
	if (status == LIBUSB_TRANSFER_CANCELLED)
		usbi_handle_transfer_cancellation(itransfer);
	else
		usbi_handle_transfer_completion(itransfer, (enum libusb_transfer_status)status);
}

void windows_handle_callback(struct usbi_transfer *itransfer, uint32_t io_result, uint32_t io_size)
{
	struct libusb_transfer *transfer = USBI_TRANSFER_TO_LIBUSB_TRANSFER(itransfer);

	switch (transfer->type) {
	case LIBUSB_TRANSFER_TYPE_CONTROL:
	case LIBUSB_TRANSFER_TYPE_BULK:
	case LIBUSB_TRANSFER_TYPE_INTERRUPT:
	case LIBUSB_TRANSFER_TYPE_ISOCHRONOUS:
		windows_transfer_callback(itransfer, io_result, io_size);
		break;
	case LIBUSB_TRANSFER_TYPE_BULK_STREAM:
		usbi_warn(ITRANSFER_CTX(itransfer), "bulk stream transfers are not yet supported on this platform");
		break;
	default:
		usbi_err(ITRANSFER_CTX(itransfer), "unknown endpoint type %d", transfer->type);
	}
}

int windows_handle_events(struct libusb_context *ctx, struct pollfd *fds, POLL_NFDS_TYPE nfds, int num_ready)
{
	POLL_NFDS_TYPE i;
	bool found = false;
	struct usbi_transfer *transfer;
	struct winfd *pollable_fd = NULL;
	DWORD io_size, io_result;
	int r = LIBUSB_SUCCESS;

	usbi_mutex_lock(&ctx->open_devs_lock);
	for (i = 0; i < nfds && num_ready > 0; i++) {

		usbi_dbg("checking fd %d with revents = %04x", fds[i].fd, fds[i].revents);

		if (!fds[i].revents)
			continue;

		num_ready--;

		// Because a Windows OVERLAPPED is used for poll emulation,
		// a pollable fd is created and stored with each transfer
		usbi_mutex_lock(&ctx->flying_transfers_lock);
		found = false;
		list_for_each_entry(transfer, &ctx->flying_transfers, list, struct usbi_transfer) {
			pollable_fd = windows_get_fd(transfer);
			if (pollable_fd->fd == fds[i].fd) {
				found = true;
				break;
			}
		}
		usbi_mutex_unlock(&ctx->flying_transfers_lock);

		if (found) {
			windows_get_overlapped_result(transfer, pollable_fd, &io_result, &io_size);

			usbi_remove_pollfd(ctx, pollable_fd->fd);
			// let handle_callback free the event using the transfer wfd
			// If you don't use the transfer wfd, you run a risk of trying to free a
			// newly allocated wfd that took the place of the one from the transfer.
			windows_handle_callback(transfer, io_result, io_size);
		} else {
			usbi_err(ctx, "could not find a matching transfer for fd %d", fds[i]);
			r = LIBUSB_ERROR_NOT_FOUND;
			break;
		}
	}
	usbi_mutex_unlock(&ctx->open_devs_lock);

	return r;
}

int windows_common_init(struct libusb_context *ctx)
{
	if (!windows_init_clock(ctx))
		goto error_roll_back;

	if (!htab_create(ctx))
		goto error_roll_back;

	return LIBUSB_SUCCESS;

error_roll_back:
	windows_common_exit();
	return LIBUSB_ERROR_NO_MEM;
}

void windows_common_exit(void)
{
	htab_destroy();
	windows_destroy_clock();
	windows_exit_dlls();
}
