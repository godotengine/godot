/**************************************************************************/
/*  audio_driver_miniaudio.cpp                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "audio_driver_miniaudio.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/profiling/profiling.h"
#include "servers/audio/audio_server.h"

#include <thirdparty/miniaudio/miniaudio.h>

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

/*
 Integration of miniaudio for audio endpoint device handling
 -----------------------------------------------------------
 - Initializing playback device and capture if input is enabled in config.
 - Most, if not all, the miniaudio device interactions encapsulated in MADeviceHandler.
 - More endpoints can be initialzied by simply configuring more MADeviceHandlers (if AudioServer API would support it).
 - MADeviceHandler wraps common device logic, initial config determines whether it's input or output (capture or playback).
 - If a device is lost, or rerouted - recovery is attempted at fixed intervals with fallback to last known valid device.
 - Capture and playback devices run on their own threads.
 - Data is passed from capture -> playback via audio thread safe ring buffer.
 - Plaback device callback sends the data from the capture ring buffer to AudioServer via staging buffer to avoid locking.
 - Synchronization between threads is lock-free.
 - The only lock is around `audio_server_process()` call, as this is currently required by AudioServer.
 - Logging can allocate (lock-free buffer/queue synchronization can be added to defer printing to main thread).
 - Device initialization and reinitialization initiated and synchronously coordinated from the main thread, audio thread can be safely teared down.
 - All the AudioDriver API calls from on the main thread are synchronized by miniaudio API if necessary.
 - Currently the playback device is initialized to backend-selected buffer size/count, avoiding exta data conversion.
 - Capture device initialized to the same sample rate as playback and fixed buffer size, to accommodate data transfer via fixed buffer capacity.
*/

// These two flags mainly needed for our manual WASAPI reinit trigger
static SafeFlag g_default_output_device_changed{ false };
static SafeFlag g_default_input_device_changed{ false };

// Name to use when user requests system default device,
// as opposed to specific named device.
constexpr const char *DEFAULT_DEVICE_NAME = "Default";

// If device is lost, or unavailable, this is the timeer
// at which we try to reinitialize the same, or fallback device.
constexpr uint64_t DEVICE_REINIT_PERIOD_MS = 1000;

/// We have to do a bit of WASAPI specific device handling.
///
/// miniaudio as of v0.11.25 handles stream rerouting from within IMMNotificationClient::OnDefaultDeviceChanged callback,
/// which can cause deadlock if trying to connect to a device that has an exclusive owner.
/// dev-0.12 branch has new deferred handling, but still some issues with rerouting.
/// Therefore on WASAPI we'll handle default device rerouting ourselves.
///
/// Opened issue (2026-08-18): https://github.com/mackron/miniaudio/issues/1149
/// When the issue is resolved, all the Windows specific code can be simply removed from here,
/// the system is going to automatically use the common path with `ma_device_notification_type_rerouted`.
#define GDT_WASAPI_WORKAROUND 1

#define GDT_MINIAUDIO_V12 1
#if GDT_MINIAUDIO_V12
using ma_pcm_rb = ma_audio_ring_buffer;
#endif

#if defined(WINDOWS_ENABLED) and GDT_WASAPI_WORKAROUND
#include <mmdeviceapi.h>
#include <wrl/client.h>

GODOT_GCC_WARNING_PUSH_AND_IGNORE("-Wnon-virtual-dtor") // Silence warning due to a COM API weirdness (GH-35194).
namespace GodotMAWASAPI {

bool g_backend_is_wasapi = false;

using Microsoft::WRL::ComPtr;

class CMMNotificationClient : public IMMNotificationClient {
	LONG _cRef = 1;

public:
	ComPtr<IMMDeviceEnumerator> enumerator = nullptr;

	CMMNotificationClient() {}
	virtual ~CMMNotificationClient() {}

	ULONG STDMETHODCALLTYPE AddRef() {
		return InterlockedIncrement(&_cRef);
	}

	ULONG STDMETHODCALLTYPE Release() {
		ULONG ulRef = InterlockedDecrement(&_cRef);
		if (0 == ulRef) {
			delete this;
		}
		return ulRef;
	}

	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, VOID **ppvInterface) {
		if (IID_IUnknown == riid) {
			AddRef();
			*ppvInterface = (IUnknown *)this;
		} else if (__uuidof(IMMNotificationClient) == riid) {
			AddRef();
			*ppvInterface = (IMMNotificationClient *)this;
		} else {
			*ppvInterface = nullptr;
			return E_NOINTERFACE;
		}
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDeviceAdded(LPCWSTR pwstrDeviceId) {
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDeviceRemoved(LPCWSTR pwstrDeviceId) {
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDeviceStateChanged(LPCWSTR pwstrDeviceId, DWORD dwNewState) {
		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnDefaultDeviceChanged(EDataFlow flow, ERole role, LPCWSTR pwstrDeviceId) {
		if (role == eConsole) {
			if (flow == eRender) {
				g_default_output_device_changed.set();
			} else if (flow == eCapture) {
				g_default_input_device_changed.set();
			}
		}

		return S_OK;
	}

	HRESULT STDMETHODCALLTYPE OnPropertyValueChanged(LPCWSTR pwstrDeviceId, const PROPERTYKEY key) {
		return S_OK;
	}

	bool IsRegistered() const { return enumerator != nullptr; }
};

GODOT_GCC_WARNING_POP

static CMMNotificationClient notif_client;

static void release_notification_client() {
	if (notif_client.enumerator != nullptr) {
		notif_client.enumerator->UnregisterEndpointNotificationCallback(&notif_client);
		notif_client.enumerator = nullptr;
	}
}

static Error reinit_notification_client() {
	release_notification_client();

	ComPtr<IMMDeviceEnumerator> enumerator = nullptr;
	HRESULT hr = CoCreateInstance(CLSID_MMDeviceEnumerator, nullptr, CLSCTX_ALL, IID_IMMDeviceEnumerator, (void **)&enumerator);
	ERR_FAIL_COND_V_MSG(hr != S_OK, ERR_CANT_OPEN,
			"WASAPI: Failed to CoCreateInstance of IMMDeviceEnumerator when registering notification client.");

	hr = enumerator->RegisterEndpointNotificationCallback(&notif_client);
	if (hr == S_OK) {
		notif_client.enumerator = enumerator;
	} else {
		ERR_PRINT("WASAPI: RegisterEndpointNotificationCallback error. HRESULT: 0x" + String::num_uint64(hr, 16) + ".");
		return Error::FAILED;
	}

	return Error::OK;
}

} // namespace GodotMAWASAPI
#endif // WINDOWS_ENABLED

//==============================================================================
/// Realtime-safe pointer wrapper.
/// Non-realtime mutable. Must be created and destroyed on non-realtime thread.
/// This is useful when you just need to give realtime therad a new object,
/// without concern about previous state.
template <class T>
class RTSafePtr {
public:
	using Type = T;

public:
	explicit RTSafePtr(Type *p_obj) : _storage(p_obj), _ptr(p_obj), _current_obj(nullptr) {
	}

	RTSafePtr() : RTSafePtr(nullptr) {
	}

	~RTSafePtr() {
		// Wait for realtime-thread to release the object
		non_rt_replace(nullptr);
	}

	[[nodiscard]] Type *rt_acquire() noexcept {
		_current_obj = _ptr.exchange(nullptr);
		return _current_obj;
	}

	void rt_release() noexcept {
		_ptr.store(_current_obj);
	}

	/// Replace the pointer with new pointer, taking ownership.
	/// The previous object is returned to the caller,
	/// it is responsibility of the caller to delete it.
	[[nodiscard]] Type *non_rt_exchage(Type *p_new_ptr) {
		Type *ptr;
		do { // Block until realtime thread is done using the object
			ptr = _storage;
		} while (not _ptr.compare_exchange_weak(ptr, p_new_ptr));

		_storage = p_new_ptr;
		return ptr;
	}

	/// Replace the pointer with new pointer, taking ownership.
	/// The previous object is deleted.
	void non_rt_replace(Type *p_new_ptr) {
		Type *ptr = non_rt_exchage(p_new_ptr);
		memdelete(ptr);
	}

private:
	Type *_storage;
	std::atomic<Type *> _ptr;
	Type *_current_obj; // Only accessed by realtime thread and only when "locked"
};

//==============================================================================
/// Simple utility to set a flag in X ms in the future and test repeatedly.
class RetryFlag {
public:
	explicit RetryFlag(uint64_t cooldown_ms = 100) : _cooldown_usec(cooldown_ms * 1000) {}

	_FORCE_INLINE_ void set(uint64_t cooldown_ms) {
		_cooldown_usec = cooldown_ms * 1000;
		_retry_at = OS::get_singleton()->get_ticks_usec() + _cooldown_usec;
	}

	_FORCE_INLINE_ void clear() { _retry_at = std::numeric_limits<uint64_t>::max(); }
	_FORCE_INLINE_ void retry() { _retry_at = OS::get_singleton()->get_ticks_usec() + _cooldown_usec; }

	[[nodiscard]] _FORCE_INLINE_ bool is_set() const { return _retry_at != std::numeric_limits<uint64_t>::max(); }
	[[nodiscard]] _FORCE_INLINE_ bool is_set_and_ready() const { return OS::get_singleton()->get_ticks_usec() >= _retry_at; }

private:
	uint64_t _cooldown_usec{ 0 };
	uint64_t _retry_at{ std::numeric_limits<uint64_t>::max() };
};

namespace MABridge {
static void *ma_memalloc(size_t size, void *user_data) {
	return memalloc(size);
}

static void *ma_memrealloc(void *ptr, size_t size, void *user_data) {
	return memrealloc(ptr, size);
}

static void ma_memfree(void *ptr, void *user_data) {
	memfree(ptr);
}

const ma_allocation_callbacks allocation_callbacks = {
	nullptr, // user data
	ma_memalloc,
	ma_memrealloc,
	ma_memfree
};

#if GDT_MINIAUDIO_V12 // device init was moved to audio thread
std::atomic<bool> g_quiet_reinit_logging{ false };
#else
thread_local std::atomic<bool> g_quiet_reinit_logging{ false };
#endif

/// Note: this can be called from audio thread, it might be a good idea to eventually
/// send this to main thread to print, without any allocations in this log callback.
static void ma_log_callback(void * /* pUserData */, ma_uint32 p_level, const char *p_message) {
	// Godot will allocate String internally in its print functoins anyway,
	// we might as well use String to remove extra new line character it prints on top of miniaudio's.
	String str(p_message);
	if (str.ends_with("\n")) {
		str.resize_uninitialized(str.size() - 1);
	}

	switch (static_cast<ma_log_level>(p_level)) {
		default:
		case MA_LOG_LEVEL_INFO: {
			print_line(str);
		} break;
		case MA_LOG_LEVEL_WARNING: {
			WARN_PRINT(str);
		} break;
		case MA_LOG_LEVEL_ERROR: {
			if (g_quiet_reinit_logging.load(std::memory_order_acquire)) {
				print_verbose(str);
			} else {
				ERR_PRINT(str);
			}
		} break;
		case MA_LOG_LEVEL_DEBUG: {
			print_verbose(str);
		} break;
	}
}

template <class Function>
static bool enumerate_devices(ma_context *p_context, Function &&p_callback) {
	using Fn = std::remove_reference_t<Function>;

	auto enumerate_cb = [](ma_device_type p_type, const ma_device_info *p_info, void *p_data) {
		auto &callback = *static_cast<Fn *>(p_data);
		return std::invoke(callback, p_type, p_info);
	};

	const ma_result result = ma_context_enumerate_devices(p_context, enumerate_cb, std::addressof(p_callback));
	ERR_FAIL_COND_V_MSG(result != MA_SUCCESS, false,
			vformat("AudioDriverMiniaudio: Failed to enumerate devices. %s",
					ma_result_description(result)));
	return true;
}

static bool find_device_info(ma_context *p_context, ma_device_type p_type, const String &p_device_name, ma_device_info &r_info) {
	bool found = false;
	return enumerate_devices(p_context, [&](ma_device_type p_ma_type, const ma_device_info *p_info) {
		if (p_ma_type == p_type and String(p_info->name) == p_device_name) {
			r_info = *p_info;
			found = true;
#if GDT_MINIAUDIO_V12
			return MA_DEVICE_ENUMERATION_ABORT; // stop enumerating
#else
			return false;
#endif
		}
#if GDT_MINIAUDIO_V12
		return MA_DEVICE_ENUMERATION_CONTINUE;
#else
		return true;
#endif
	}) and found;
}

static bool get_device_list(ma_context *p_context, ma_device_type p_type, PackedStringArray &r_list) {
	return MABridge::enumerate_devices(p_context, [&](ma_device_type p_ma_type, const ma_device_info *p_info) {
		if (p_ma_type == p_type) {
			r_list.push_back(String(p_info->name));
		}
#if GDT_MINIAUDIO_V12
		return MA_DEVICE_ENUMERATION_CONTINUE; // keep enumerating
#else
		return true;
#endif
	});
}

//==========================================================================
/// Deleter customization to use with smart pointers using Godot's memdelete()
struct DefaultDeleter {
	constexpr DefaultDeleter() noexcept = default;

	template <class T>
	void operator()(T *ptr) const noexcept {
		if (ptr) {
			memdelete(ptr);
		}
	}
};

//==========================================================================
/// Deleter customization to use with smart pointers, miniaudio's
/// ma_uninit_XXX() on destruction and Godot's memdelete()
template <class MAType, auto UninitFunc>
struct Deleter {
	constexpr Deleter() noexcept = default;

	void operator()(MAType *ptr) const noexcept {
		if (ptr) {
			std::invoke(UninitFunc, ptr);
			memdelete(ptr);
		}
	}
};

using LogDeleter = Deleter<ma_log, ma_log_uninit>;
using DeviceDeleter = Deleter<ma_device, ma_device_uninit>;
using ContextDeleter = Deleter<ma_context, ma_context_uninit>;
} // namespace MABridge

//==============================================================================
/// Handles ma_device and persistent data for reinitialization.
/// MADeviceHandler can be initialized for a number of devices providing generic
/// high level interface.
class MADeviceHandler {
public:
	MADeviceHandler() = default;
	~MADeviceHandler() { reset(); }

	using ReinitCallback = void (*)(MADeviceHandler &handler, void *p_user_data);

	// In some cases there's a need to do something when device reinitialized,
	// but not started yet. E.g. clear buffer while device data thread is not running.
	// Such callback can be registered via this function.
	void register_reinit_cb(ReinitCallback p_callback, void *p_user_data) {
		_reinit_callback_user_data = p_user_data;
		_reinit_callback = p_callback;
	}

	// Initialize device with given context and config.
	ma_result init(ma_context *p_context, const ma_device_config &p_config) {
		ERR_FAIL_COND_V((p_config.deviceType != ma_device_type_playback) and (p_config.deviceType != ma_device_type_capture), MA_INVALID_ARGS);

		if (not p_context) {
			return MA_INVALID_ARGS;
		}

		context = p_context;
		_config = p_config;

		device.reset(memnew(ma_device));

#if defined(WINDOWS_ENABLED) and GDT_WASAPI_WORKAROUND
		if (GodotMAWASAPI::notif_client.IsRegistered()) {
#if GDT_MINIAUDIO_V12
			_config.wasapi.noAutoStreamRouting = GodotMAWASAPI::g_backend_is_wasapi ? MA_TRUE : MA_FALSE;
#else
			const bool manual_wasapi_routing = p_context->backend == ma_backend_wasapi;
			_config.wasapi.noAutoStreamRouting = manual_wasapi_routing ? MA_TRUE : MA_FALSE;
#endif
		}
#endif

		const ma_result result = ma_device_init(p_context, &_config, device.get());
		if (result == MA_SUCCESS) {
			// Store sample rate the device was actually initialized to
			_sample_rate = static_cast<int>(device->sampleRate);
			// We retain the same sample rate throughout the lifetime of applicatoin
			_config.sampleRate = static_cast<ma_uint32>(_sample_rate);
			// Get the updated channel count from the new endpoint
			_num_channels = get_device_channels(*device);

			if (not is_supported_channel_count(_num_channels, _config.deviceType)) {
				reset();
				return MA_INVALID_DEVICE_CONFIG;
			}
			_aprox_latency = get_device_cb_latency(*device);

			// Save device name
			if (get_device_id(p_config) == nullptr) {
				device_name = DEFAULT_DEVICE_NAME;
			} else {
				query_device_name(*device, device_name);
			}

		} else {
			// currently ma_device may not be safe to uninit() after failed to init(),
			// so we clear the struct before reset()
			(*device) = {};
			reset();
		}

		return result;
	}

	static bool is_supported_channel_count(uint32_t p_channel_count, ma_device_type p_device_type) {
		if (p_device_type == ma_device_type_playback) {
			// Currently Godot support this specific set of output channels,
			// mapping them to specific speaker layouts.
			for (uint32_t supported_cc : { 2, 4, 6, 8 }) {
				if (p_channel_count == supported_cc) {
					return true;
				}
			}
			return false;
		} else {
			DEV_ASSERT(p_device_type == ma_device_type_capture);
			// We fold any input device channel count into Godot's hardcoded stereo.
			return p_channel_count > 0;
		}
	}

	ma_result reinit(const ma_device_id *p_id, bool p_should_be_started) {
		ma_result init_result = MA_FAILED_TO_INIT_BACKEND;

		// Uninitialize current device
		reset();

		if (p_id != nullptr) {
			// On reinit we use device ID we store or get when this is called,
			// device ID pointer inside the config may already be invalid.
			set_device_id(_config, p_id);

			init_result = init(context, _config);
			if (init_result == MA_SUCCESS) {
				// Must be called before device is started
				if (_reinit_callback) {
					_reinit_callback(*this, _reinit_callback_user_data);
				}

				if (p_should_be_started) {
					const ma_result start_result = start();
					if (start_result != MA_SUCCESS) {
						// If failed to start non-default device, uninitialize
						reset();
						ERR_PRINT(vformat("AudioDriverMiniaudio: Reinitializing device. "
										  "Requested device failed to start. %s. Falling back to last valid device.",
								ma_result_description(start_result)));
					}
				}
			} else {
				WARN_PRINT(vformat("AudioDriverMiniaudio: Reinitializing device. "
								   "Failed to initialize requested device. %s. Falling back to last valid device.",
						ma_result_description(init_result)));
			}
		}

		// This can be either fallback from failed explicit device init,
		// or user has requested default device (p_id = nullptr)
		//
		// Note: the fallback behavior may or may not be desirable
		if (not is_valid()) {
			if (p_id != nullptr) {
				// Failed to init explicit device, fallback to last valid
				p_id = last_valid_id.get();
			}

			set_device_id(_config, p_id);

			init_result = init(context, _config);
			if (init_result == MA_SUCCESS) {
				// Must be called before device is started
				if (_reinit_callback) {
					_reinit_callback(*this, _reinit_callback_user_data);
				}

				if (p_should_be_started) {
					const ma_result start_result = start();
					if (start_result != MA_SUCCESS) {
						reset();
						ERR_PRINT(vformat("AudioDriverMiniaudio: Reinitializing device. "
										  "Last valid device failed to start. %s",
								ma_result_description(start_result)));
					}
				}
			}
		}

		// We want to set last_valid_id after successfully initializing and starting here.
		// instead of inside init(), because here failed to start(), means the ID is not valid for retry.
		// Note: this assumes that public init() called from AudioDriverMiniauio doesn't take explicit device,
		// so we only need to save it on reinit(). Alternatively, we could expose this as "cache_id_if_valid()",
		// and let AudioDriverMiniaudio call it after successful initialization;
		if (is_valid()) {
			const ma_device_id *device_id = get_device_id(_config);
			last_valid_id.reset(device_id ? memnew(ma_device_id(*device_id)) : nullptr);
			is_explicit = device_id != nullptr;
		}

		return init_result;
	}

	struct UpdateContext {
		bool default_device_changed = false;
		bool should_be_started = true;
	};

	// @returns true if device was reinitialized
	bool handle_update(UpdateContext update_context) {
		if (not is_configured()) {
			return false;
		}

		bool was_reinitialized = false;

		// This is reroute notification for non-WASAPI backends
		bool should_reinitialize = was_rerouted.clear_if_set();
		//bool should_reinitialize = false;//? TMP. deisabled for debugging was_rerouted.clear_if_set();

		// This is our own reroute notification for WASAPI backend
#if defined(WINDOWS_ENABLED) and GDT_WASAPI_WORKAROUND
		if (update_context.default_device_changed and not is_explicit) {
			should_reinitialize = true;
		}
#endif

		if (should_reinitialize) {
			// Start off periodic retry procedure
			retry_reinit_flag.set(DEVICE_REINIT_PERIOD_MS);
			_print_first_error = true;

		} else if (not retry_reinit_flag.is_set()) {
			// If we don't have output device, while we should,
			// start probing if we can initialize last known or default device.

#if GDT_MINIAUDIO_V12
			const ma_device_status state = get_state();

			if (state == ma_device_status_uninitialized or
					state == ma_device_status_errored or
					(state == ma_device_status_stopped and update_context.should_be_started)) {
#else
			const ma_device_state state = get_state();

			if (state == ma_device_state_uninitialized or
					(state == ma_device_state_stopped and update_context.should_be_started)) {
#endif
				// Start off periodic retry procedure
				retry_reinit_flag.set(DEVICE_REINIT_PERIOD_MS);
				_print_first_error = true;
			}
		}

		if (retry_reinit_flag.is_set_and_ready()) {
			MABridge::g_quiet_reinit_logging.store(not std::exchange(_print_first_error, false), std::memory_order_release);

			// If reinit() fails for whatever reason, we'll keep retrying
			reinit(last_valid_id.get(), update_context.should_be_started);

			if (update_context.should_be_started) {
				was_reinitialized = is_started();
			} else {
				was_reinitialized = is_initialized();
			}

			if (was_reinitialized) {
				print_line("AudioDriverMiniaudio: Successfully reinitialized audio device.");
				retry_reinit_flag.clear();
			} else {
				// Device object is reset and we keep retrying...
				retry_reinit_flag.retry();
			}

			MABridge::g_quiet_reinit_logging.store(false, std::memory_order_release);
		}

		return was_reinitialized;
	}

#if GDT_MINIAUDIO_V12
	ma_device_status get_state() const { return ma_device_get_status(device.get()); }
#else
	ma_device_state get_state() const { return ma_device_get_status(device.get()); }
#endif

	bool is_valid() const { return device.get(); }
	bool is_configured() const { return context != nullptr; }
#if GDT_MINIAUDIO_V12
	bool is_started() const { return _num_channels > 0 and _sample_rate > 0 and get_state() == ma_device_status_started; }
	bool is_initialized() const { return _num_channels > 0 and _sample_rate > 0 and get_state() != ma_device_status_uninitialized; }
#else
	bool is_started() const { return _num_channels > 0 and _sample_rate > 0 and get_state() == ma_device_state_started; }
	bool is_initialized() const { return _num_channels > 0 and _sample_rate > 0 and get_state() != ma_device_state_uninitialized; }
#endif

	ma_result start() { return ma_device_start(device.get()); }
	ma_result stop() { return ma_device_stop(device.get()); }

	// Reset internal device and temporary data related to the device.
	// Keep reinitialziation info (context, config, etc.)
	void reset() {
		device.reset();
		was_rerouted.clear();
		_num_channels = 0;
		_sample_rate = 0;
		_aprox_latency = 0.0f;
		device_name.clear();
	}

	bool get_device_name(String &r_name) const {
		if (not device) {
			return false;
		}

		// If device was explicit, name must match default device name constant
		DEV_ASSERT(is_explicit or (last_valid_id == nullptr and device_name == DEFAULT_DEVICE_NAME));

		r_name = device_name;
		return not device_name.is_empty();
	}

	int get_num_channels() const { return _num_channels; }
	int get_sample_rate() const { return _sample_rate; }
	float get_device_latency_s() const { return _aprox_latency; }

private:
	static const ma_device_id *get_device_id(const ma_device_config &p_config) {
		if (p_config.deviceType == ma_device_type_playback) {
			return p_config.playback.pDeviceID;
		} else {
			ERR_FAIL_COND_V(p_config.deviceType != ma_device_type_capture, nullptr);
			return p_config.capture.pDeviceID;
		}
	}

	static void set_device_id(ma_device_config &p_config, const ma_device_id *id) {
		if (p_config.deviceType == ma_device_type_playback) {
			p_config.playback.pDeviceID = id;
		} else {
			ERR_FAIL_COND(p_config.deviceType != ma_device_type_capture);
			p_config.capture.pDeviceID = id;
		}
	}

	static uint32_t get_device_channels(ma_device &p_device) {
		if (p_device.type == ma_device_type_playback) {
			return p_device.playback.channels;
		} else {
			ERR_FAIL_COND_V(p_device.type != ma_device_type_capture, 0);
			return p_device.capture.channels;
		}
	}

	// Miniaudio does not expose backend-reported stream latency through its public cross-platform API.
	// We approximate it using the negotiated buffer duration. This excludes additional OS, hardware, and conversion latency.
	static float get_device_cb_latency(ma_device &p_device) {
		if (p_device.type == ma_device_type_playback) {
			const ma_uint32 sample_rate = p_device.playback.internalSampleRate;
			const ma_uint32 frames = p_device.playback.internalPeriodSizeInFrames * p_device.playback.internalPeriods;
			return sample_rate > 0 ? frames / static_cast<float>(sample_rate) : 0.0f;
		} else {
			ERR_FAIL_COND_V(p_device.type != ma_device_type_capture, 0.0f);
			const ma_uint32 sample_rate = p_device.capture.internalSampleRate;
			const ma_uint32 frames = p_device.capture.internalPeriodSizeInFrames * p_device.capture.internalPeriods;
			return sample_rate > 0 ? frames / static_cast<float>(sample_rate) : 0.0f;
		}
	}

	static bool query_device_name(ma_device &p_device, String &r_name) {
		char buffer[MA_MAX_DEVICE_NAME_LENGTH]{};
		size_t actual_len = 0; // not including null-terminator
		const ma_result result = ma_device_get_name(&p_device, p_device.type, buffer, MA_MAX_DEVICE_NAME_LENGTH, &actual_len);
		ERR_FAIL_COND_V(result != MA_SUCCESS, false);

		r_name.clear();
		r_name.append_latin1(Span(buffer, actual_len));
		return true;
	}

public:
	ma_context *context = nullptr;
	std::unique_ptr<ma_device, MABridge::DeviceDeleter> device = nullptr;

	// We have to store device name because user might be querying it every frame,
	// and querying backend on every frame is no good.
	String device_name;

	SafeFlag was_rerouted;
	RetryFlag retry_reinit_flag;

	// Whetere user has explicitly set specific device.
	// This tells us to ignore default device rerouting.
	bool is_explicit = false;

	std::unique_ptr<ma_device_id, MABridge::DefaultDeleter> last_valid_id;

private:
	// Store a copy of the config for device reinitialization
	ma_device_config _config = {};
	int _num_channels = 0;
	int _sample_rate = 0;
	float _aprox_latency = 0.0f;

	// This a hack, but we need to print first error
	// when trying to reinit device and route the rest into
	// verbose log.
	bool _print_first_error = true;

	ReinitCallback _reinit_callback = nullptr;
	void *_reinit_callback_user_data = nullptr;
};

//==============================================================================
/// miniaudio data and handlers stored in a private struct to avoid leaking
/// miniaudio outside of this translation unit.
/// Also serves as a convenient separation for implementation details from
/// high level AudioDriver logic.
struct AudioDriverMiniaudio::MAData {
	std::unique_ptr<ma_log, MABridge::LogDeleter> log = nullptr;
	std::unique_ptr<ma_context, MABridge::ContextDeleter> context = nullptr;

	MADeviceHandler playback; // output
	MADeviceHandler capture; // input

	ma_pcm_rb *current_input_ring_buffer = nullptr;
	RTSafePtr<ma_pcm_rb> input_ring_buffer;
};

//==============================================================================
template <>
void AudioDriverMiniaudio::output_device_notification_cb(const ma_device_notification *p_notification) {
	// Note: DirectSound doesn't have a way to notify about stream rerouting,
	// but that backend is pretty much obsolete
	if (p_notification->type == ma_device_notification_type_rerouted) {
		AudioDriverMiniaudio *ad = static_cast<AudioDriverMiniaudio *>(p_notification->pDevice->pUserData);
		ad->_ma_data->playback.was_rerouted.set();
	}
}

template <>
void AudioDriverMiniaudio::input_device_notification_cb(const ma_device_notification *p_notification) {
	if (p_notification->type == ma_device_notification_type_rerouted) {
		AudioDriverMiniaudio *ad = static_cast<AudioDriverMiniaudio *>(p_notification->pDevice->pUserData);
		ad->_ma_data->capture.was_rerouted.set();
	}
}

//==============================================================================
void AudioDriverMiniaudio::output_device_cb(ma_device *p_device, void *p_output, const void * /*p_input*/, uint32_t p_frame_count) {
#ifdef GODOT_USE_TRACY
	static thread_local bool tracy_thread_named = false;
	if (!tracy_thread_named) {
		tracy::SetThreadName("AudioDriverMiniaudio");
		tracy_thread_named = true;
	}
#endif

	GodotProfileZone("Miniaudio output callback");

	AudioDriverMiniaudio *ad = static_cast<AudioDriverMiniaudio *>(p_device->pUserData);

	const bool clear_input_buffers = ad->_device_was_reinitialized.clear_if_set();

	// No need to lock this
	if (clear_input_buffers) {
		ad->_reset_input_buffers();
	} else {
		ad->_drain_input_buffer();
	}

	{
		// Current implementation of AudioServer relies
		// on audio driver locking `audio_server_process()` section
		std::scoped_lock mix_lock(ad->_mix_mutex);

		if (clear_input_buffers) {
			// Reset AudioDriver's ring buffer serving public API
			ad->input_position = 0;
			ad->input_size = 0;
		} else {
			ad->_commit_input_buffer(); // this has to be under lock, we're touching AudioDriver::input_buffer
		}
		ad->start_counting_ticks();

		{
			GodotProfileZone("Miniaudio AudioServer process");

			ad->audio_server_process(p_frame_count, static_cast<int32_t *>(p_output));
		}

		ad->stop_counting_ticks();
	}
}

void AudioDriverMiniaudio::input_device_cb(ma_device *p_device, void * /*p_output*/, const void *p_input, uint32_t p_frame_count) {
	AudioDriverMiniaudio *ad = static_cast<AudioDriverMiniaudio *>(p_device->pUserData);

	// This is safe because input device thread won't be running
	// when input ring buffer is replaced
	ma_pcm_rb *rb = ad->_ma_data->current_input_ring_buffer;
	if (rb == nullptr) {
		return;
	}

	ma_uint32 frames_written = 0;

#if not GDT_MINIAUDIO_V12
	while (frames_written < p_frame_count) {
		void *mapped_buffer;
		ma_uint32 frames = p_frame_count - frames_written;

		if (ma_pcm_rb_acquire_write(rb, &frames, &mapped_buffer) != MA_SUCCESS) {
			break;
		}

		if (frames == 0) {
			if (ma_pcm_rb_pointer_distance(rb) == static_cast<ma_int32>(ma_pcm_rb_get_subbuffer_size(rb))) {
#if GDT_MA_AUDIO_DRIVER_DEBUG
				ad->_input_rb_overrun.increment();
#endif
				break; // Overrun. Not enough room in the ring buffer for input frame. Excess frames are dropped.
			}
		} else {
			ma_copy_pcm_frames(
					mapped_buffer,
					ma_offset_pcm_frames_const_ptr(p_input, frames_written, p_device->capture.format, p_device->capture.channels),
					frames,
					p_device->capture.format,
					p_device->capture.channels);
		}

		if (ma_pcm_rb_commit_write(rb, frames) != MA_SUCCESS) {
			break;
		}

		frames_written += frames;
	}
#else

	ma_audio_ring_buffer_write_pcm_frames(rb, p_input, p_frame_count, &frames_written);
#if GDT_MA_AUDIO_DRIVER_DEBUG
	if (frames_written < p_frame_count) {
		ad->_input_rb_overrun.increment();
	}
#endif
#endif
}

void AudioDriverMiniaudio::_drain_input_buffer() {
	_input_stage_buffer.clear();

	ma_pcm_rb *rb = _ma_data->input_ring_buffer.rt_acquire();

	if (rb == nullptr) {
		return;
	}
#if not GDT_MINIAUDIO_V12
	while (true) {
		ma_uint32 frames = ma_pcm_rb_available_read(rb);
		if (frames == 0) {
			break;
		}

		void *mapped_buffer = nullptr;
		const ma_result result = ma_pcm_rb_acquire_read(rb, &frames, &mapped_buffer);

		if (result == MA_SUCCESS) {
			if (frames > 0) {
				const auto *samples = static_cast<const int32_t *>(mapped_buffer);

				const ma_uint32 space_left = _input_stage_buffer.get_capacity() - _input_stage_buffer.size();
				const ma_uint32 to_drain = MIN(space_left, frames * NUM_IN_CHANNELS);

				for (ma_uint32 i = 0; i < to_drain; i++) {
					_input_stage_buffer.push_back(samples[i]);
				}

				const ma_uint32 drained_frames = to_drain / NUM_IN_CHANNELS;

#if GDT_MA_AUDIO_DRIVER_DEBUG
				if (drained_frames < frames) {
					// Overrun
					_input_sb_overrun.increment();
				}
#endif

				// Make sure we commit what fit into scratch buffer
				frames = drained_frames;

			} else {
				if (ma_pcm_rb_pointer_distance(rb) == 0) {
					break; // Underrun. This shoulsn't happen since we read only what's available.
				}
			}
		}

		if (ma_pcm_rb_commit_read(rb, frames) != MA_SUCCESS) {
			break;
		}
		// ...continue until the buffer is drained
	}
#else

	// Try to read as much, as we have space in staging buffer
	uint32_t input_frames = (_input_stage_buffer.get_capacity() - _input_stage_buffer.size()) / NUM_IN_CHANNELS;

	void *mapped_buffer = nullptr;
	input_frames = ma_audio_ring_buffer_map_consume(rb, input_frames, &mapped_buffer);

	if (input_frames > 0) {
		const uint32_t num_new_samples = input_frames * NUM_IN_CHANNELS;
		const uint32_t write_pos = _input_stage_buffer.size();
		_input_stage_buffer.resize_uninitialized(_input_stage_buffer.size() + num_new_samples);
		memcpy(&_input_stage_buffer[write_pos], mapped_buffer, ma_get_bytes_per_sample(rb->format) * num_new_samples);
	}
	ma_audio_ring_buffer_unmap_consume(rb, input_frames);
#endif

	_ma_data->input_ring_buffer.rt_release();
}

void AudioDriverMiniaudio::_commit_input_buffer() {
	for (int32_t sample : _input_stage_buffer) {
		input_buffer_write(sample);
	}
	_input_stage_buffer.clear();
}

void AudioDriverMiniaudio::_reset_input_buffers() {
	ma_pcm_rb *rb = _ma_data->input_ring_buffer.rt_acquire();

	if (rb != nullptr) {
#if GDT_MINIAUDIO_V12
		ma_uint32 length = 0;
		const ma_result result = ma_audio_ring_buffer_get_length_in_pcm_frames(rb, &length);
		if (result == MA_SUCCESS) {
			void *buffer;
			ma_uint32 mapped = ma_audio_ring_buffer_map_consume(rb, length, &buffer);
			ma_audio_ring_buffer_unmap_consume(rb, mapped);
		}
#else
		const ma_uint32 queued_frames = ma_pcm_rb_available_read(rb);
		ma_pcm_rb_seek_read(rb, queued_frames);
#endif
		_ma_data->input_ring_buffer.rt_release();
	}
	_input_stage_buffer.clear();
}

//==============================================================================
AudioDriverMiniaudio::AudioDriverMiniaudio() : _owner_therad_id(Thread::UNASSIGNED_ID), _update_cb_registered(false), _out_should_be_started(false), _in_should_be_started(false), _sample_rate(0), _num_channels(0), _ma_data(memnew(MAData)) {
}

AudioDriverMiniaudio::~AudioDriverMiniaudio() {
	memdelete(_ma_data);
}

Error AudioDriverMiniaudio::init() {
	_owner_therad_id = Thread::get_caller_id();

#if 1
	// Use device native sample rate
	_sample_rate = 0;

	//! Note: for now, for the draft, we ignore user requested latency,
	//! as it means different things for different backens.
	// There's really no deterministic way to set latency to requested milliseconds value.
	// The best we can do is to initialize device's native format,
	// which will give us fast path without extra intermediary buffers and conversion,
	// and use our own softwware queue with configurable number of buffers and buffer size.
	// At the moment AudioServer's hardcoded 512 mix buffer size is sort of our latency/fixed size callback layer.
	const uint32_t num_periods = 0; // miniaudio will initialize to sensible 2
	const bool fixed_size_out_cb = false; // out output callback is going to have device native (possibly variable) frame size

#else
	_sample_rate = MAX(0, _get_configured_mix_rate());

	const bool fixed_size_out_cb = true;

	// We keep period size fixed to PERIOD_FRAME_COUNT,
	// which is what AudioDriver uses for single mix callback.
	// The requested latency in ms control the number of periods.
	// Note: eventually it's better to giver user the more relevant
	// config settings "buffer size" + "number of beffers", corresponding
	// to `period size` and `number of periods`.
	const uint32_t target_latency_ms = static_cast<uint32_t>(MAX(0, Engine::get_singleton()->get_audio_output_latency()));
	const uint32_t tartget_latency_frames = _sample_rate * (target_latency_ms * 0.001f);
	const uint32_t num_periods = MAX(2u, (tartget_latency_frames + PERIOD_FRAME_COUNT - 1) / PERIOD_FRAME_COUNT);
	// Note: there's no reliable way to set "global latency", the best we can do,
	// is to set minimal reliable period size and number of periods (2),
	// and later, if more refined latency adjustment is required, we can implement
	// internal software queue, while keeping the device initialziation at low-latency configuration.
#endif
	// Initialize log
	{
		ma_result result;

		_ma_data->log.reset(memnew(ma_log));

		result = ma_log_init(
				&MABridge::allocation_callbacks,
				_ma_data->log.get());
		if (result != MA_SUCCESS) {
			memdelete(_ma_data->log.release());

			ERR_FAIL_V_MSG(Error::FAILED,
					vformat("AudioDriverMiniaudio: Failed to initialize logging. %s",
							ma_result_description(result)));
		}
		result = ma_log_register_callback(
				_ma_data->log.get(),
				ma_log_callback_init(MABridge::ma_log_callback, /* user data*/ nullptr));
		if (result != MA_SUCCESS) {
			_ma_data->log.reset();

			ERR_FAIL_V_MSG(Error::FAILED,
					vformat("AudioDriverMiniaudio: Failed to register log callback. %s",
							ma_result_description(result)));
		}
	}

	// Initialize context
	{
		ma_result result;

		_ma_data->context.reset(memnew(ma_context));

		ma_context_config context_config = ma_context_config_init();
		context_config.allocationCallbacks = MABridge::allocation_callbacks;
		context_config.pLog = _ma_data->log.get();

		result = ma_context_init(
				nullptr, // backends
				0, // backend count
				&context_config,
				_ma_data->context.get());
		if (result != MA_SUCCESS) {
			_ma_data->log.reset();
			memdelete(_ma_data->context.release());

			ERR_FAIL_V_MSG(Error::FAILED,
					vformat("AudioDriverMiniaudio: Failed to initialize context. %s",
							ma_result_description(result)));
		}
	}

#if defined(WINDOWS_ENABLED) and GDT_WASAPI_WORKAROUND
#if GDT_MINIAUDIO_V12
	GodotMAWASAPI::g_backend_is_wasapi = _ma_data->context->pVTable == ma_wasapi_get_vtable();
	if (GodotMAWASAPI::g_backend_is_wasapi) {
		if (GodotMAWASAPI::reinit_notification_client() != Error::OK) {
			ERR_PRINT("WASAPI: Falling back to internal 'miniaudio' device reroute handling, "
					  "which may be unsafe if rerouting into a device which has an exclusive owner.");
		}
	}
#else
	if (_ma_data->context->backend == ma_backend_wasapi) {
		if (GodotMAWASAPI::reinit_notification_client() != Error::OK) {
			ERR_PRINT("WASAPI: Falling back to internal 'miniaudio' device reroute handling, "
					  "which may be unsafe if rerouting into a device which has an exclusive owner.");
		}
	}
#endif
#endif

	// Initialize

	// TODO: replace with whatever Godot is using, currently it's s32,
	// but in the future it might switch to f32
	const ma_format sample_format = ma_format_s32;

	// Playback
	{
		ma_device_config device_config = ma_device_config_init(ma_device_type_playback);
		device_config.playback.pDeviceID = nullptr; // Initialize default playback device
		device_config.playback.format = sample_format;
		device_config.playback.channels = 0; // Use device's native channel count
		device_config.dataCallback = output_device_cb;
		device_config.sampleRate = _sample_rate;

		device_config.periodSizeInFrames = 0; //PERIOD_FRAME_COUNT;
		device_config.periods = num_periods;
		device_config.noFixedSizedCallback = not fixed_size_out_cb;

		device_config.pUserData = this;
		device_config.notificationCallback = AudioDriverMiniaudio::output_device_notification_cb;

		const ma_result playback_init_result = _ma_data->playback.init(_ma_data->context.get(), device_config);
		if (playback_init_result != MA_SUCCESS) {
			ERR_PRINT(vformat("AudioDriverMiniaudio: Failed to initialize playback device. %s.",
					ma_result_description(playback_init_result)));
			// For now we don't start device reinit attempts if failed in AudioDriver::init,
			// instead we let Godot handle fallback, since this was prior behavior.
			finish();
			return Error::FAILED;
		} else {
			_sample_rate = _ma_data->playback.get_sample_rate();
			_num_channels = _ma_data->playback.get_num_channels();

			_ma_data->playback.register_reinit_cb([](MADeviceHandler &p_handler, void *p_this) {
				AudioDriverMiniaudio *ad = static_cast<AudioDriverMiniaudio *>(p_this);

				const int new_channel_count = p_handler.get_num_channels();

				// Update cached num channels value to let Godot update its buses
				if (ad->_num_channels != new_channel_count) {
					ad->_num_channels = p_handler.get_num_channels();

					// AudioServer::get_singleton()->init_channels_and_buffers();
					// Unfortunitely we cannot call this private function,
					// or let AudioServer know the channel count has changed in any other direct way,
					// therefore AudioServer is going to have to reallocate it's buffers
					// from the running audio device thread where it queries the updated channel count.
				}

				ad->_device_was_reinitialized.set();
			},
					this);
		}
	}

	// Capture
	if (GLOBAL_GET("audio/driver/enable_input")) {
		ma_device_config device_config;
		device_config = ma_device_config_init(ma_device_type_capture);
		device_config.capture.pDeviceID = nullptr; // Initialize default capture device
		device_config.capture.format = sample_format;
		device_config.capture.channels = NUM_IN_CHANNELS; // Set explicit channel count, so that on device reinit it stays the same
		device_config.sampleRate = _sample_rate;
		device_config.dataCallback = input_device_cb;

		// We still want fixed size input callback to be able to preallocate appropriate size transfer buffers
		device_config.periodSizeInFrames = PERIOD_FRAME_COUNT;
		device_config.periods = num_periods;

		device_config.pUserData = this;
		device_config.notificationCallback = AudioDriverMiniaudio::input_device_notification_cb;

		const ma_result input_init_result = _ma_data->capture.init(_ma_data->context.get(), device_config);

		if (input_init_result != MA_SUCCESS) {
			ERR_PRINT(vformat("AudioDriverMiniaudio: Failed to initialize capture device. %s",
					ma_result_description(input_init_result)));
		} else {
			// Capture is not started yet, initialize buffers

			// Out input->output staging buffer
			_input_stage_buffer.reserve(PERIOD_FRAME_COUNT * 4 * NUM_IN_CHANNELS);

			// Godot's ring buffer
			input_buffer_init(PERIOD_FRAME_COUNT);

			// Out input->output thread-safe ring buffer
			if (not _init_input_ring_buffer()) {
				_ma_data->capture.reset();
			}

			// When capture device is reinitialized, reinitialize the ring buffer
			_ma_data->capture.register_reinit_cb([](MADeviceHandler &p_handler, void *p_this) {
				AudioDriverMiniaudio *ad = static_cast<AudioDriverMiniaudio *>(p_this);
				if (not ad->_init_input_ring_buffer()) {
					p_handler.reset();
				}
				ad->_device_was_reinitialized.set();
			},
					this);
		}

		// Note: at the moment we don't need to reinitialzie input device when output device changes,
		// because we keep the original sample rate and format of the output and channel count difference here doesn't matter.
		// If sample rate changes that would be a reason to reinit input device, since we'd need sample rate conversion.
	}

	return Error::OK;
}

void AudioDriverMiniaudio::start() {
	ERR_FAIL_COND(not _is_owner_thread());

	// If device failed to start, use may select another device,
	// we want to make sure it will be started immediately,
	// since AudioDirver::start may not be called ever again after init.
	// Note: for now we'll assume this behavior, as it's the way it was in Godot,
	// but later we may require user to call 'start()' when manually selecting new device.
	_out_should_be_started = true;

	if (not _update_cb_registered) {
		if (AudioServer *audio_server = AudioServer::get_singleton()) {
			audio_server->add_update_callback(_audio_server_update, this);
			_update_cb_registered = true;
		}
	}

	ERR_FAIL_COND(_ma_data->playback.start() != MA_SUCCESS);
}

int AudioDriverMiniaudio::get_mix_rate() const {
	return _sample_rate;
}

AudioDriver::SpeakerMode AudioDriverMiniaudio::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(_num_channels);
}

float AudioDriverMiniaudio::get_latency() {
	ERR_FAIL_COND_V(not _is_owner_thread(), 0.0f);
	return _ma_data->playback.get_device_latency_s();
}

void AudioDriverMiniaudio::lock() {
	_mix_mutex.lock();
}

void AudioDriverMiniaudio::unlock() {
	_mix_mutex.unlock();
}

void AudioDriverMiniaudio::finish() {
	if (not _is_owner_thread()) {
		WARN_PRINT("AudioDriverMiniaudio::finish() called from different thread than init(). "
				   "Ensure no other thread is touching AudioDriverMiniaudio at this point.");
	}

	_out_should_be_started = false;
	_in_should_be_started = false;

#if defined(WINDOWS_ENABLED) and GDT_WASAPI_WORKAROUND
	GodotMAWASAPI::release_notification_client();
#endif

	if (_update_cb_registered) {
		if (AudioServer *audio_server = AudioServer::get_singleton()) {
			audio_server->remove_update_callback(_audio_server_update, this);
		}
		_update_cb_registered = false;
	}

	_ma_data->capture.reset();
	_ma_data->playback.reset();

	ma_pcm_rb *rb = _ma_data->input_ring_buffer.non_rt_exchage(nullptr);
	ma_pcm_rb *crb = std::exchange(_ma_data->current_input_ring_buffer, nullptr);

	if (rb or crb) {
		if (crb != nullptr and crb != rb) {
			// This should never happen
			ERR_PRINT("AudioDriverMiniaudio: capture device ring buffer synchronization error.");

			// We still don't want to leak it, even if there's a mismatch
#if GDT_MINIAUDIO_V12
			ma_audio_ring_buffer_uninit(crb);
#else
			ma_pcm_rb_uninit(crb);
#endif
			memdelete(crb);
		}

		if (rb != nullptr) {
#if GDT_MINIAUDIO_V12
			ma_audio_ring_buffer_uninit(rb);
#else
			ma_pcm_rb_uninit(rb);
#endif
			memdelete(rb);
		}
	}

	_ma_data->context.reset();
	_ma_data->log.reset();
}

PackedStringArray AudioDriverMiniaudio::get_output_device_list() {
	PackedStringArray list;
	ERR_FAIL_COND_V(not _is_owner_thread(), list);

	list.push_back(DEFAULT_DEVICE_NAME);

	MABridge::get_device_list(_ma_data->context.get(), ma_device_type_playback, list);
	return list;
}

String AudioDriverMiniaudio::get_output_device() {
	String name;
	ERR_FAIL_COND_V(not _is_owner_thread(), name);
	_ma_data->playback.get_device_name(name);
	return name;
}

void AudioDriverMiniaudio::set_output_device(const String &p_name) {
	ERR_FAIL_COND(not _is_owner_thread());

	if (p_name == get_output_device()) {
		return;
	}

	if (p_name == DEFAULT_DEVICE_NAME) {
		_ma_data->playback.reinit(/* device id */ nullptr, /* should be started */ true); // output is always started

	} else {
		ma_device_info device_info = {};
		ERR_FAIL_COND_MSG(not MABridge::find_device_info(_ma_data->context.get(), ma_device_type_playback, p_name, device_info),
				vformat("AudioDriverMiniaudio: Failed to find device %s to switch to.", p_name));

		// If this fails and device is nullptr, main thread callback should start retrying to reinit
		_ma_data->playback.reinit(&device_info.id, /* should be started */ true); // output is always started
	}
}

Error AudioDriverMiniaudio::input_start() {
	ERR_FAIL_COND_V(not _is_owner_thread(), Error::FAILED);

	ERR_FAIL_COND_V(_ma_data->capture.start() != MA_SUCCESS, Error::FAILED);
	_in_should_be_started = true;
	return Error::OK;
}

Error AudioDriverMiniaudio::input_stop() {
	ERR_FAIL_COND_V(not _is_owner_thread(), Error::FAILED);

	_in_should_be_started = false;
	// input_stop() can be called after finish() where we uninit input device,
	// if input device already uninitialized, stopping it is not a valid operation
#if GDT_MINIAUDIO_V12
	if (_ma_data->capture.is_initialized()) {
#else
	if (_ma_data->capture.is_initialized()) {
#endif
		ERR_FAIL_COND_V(_ma_data->capture.stop() != MA_SUCCESS, Error::FAILED);
	}
	return Error::OK;
}

PackedStringArray AudioDriverMiniaudio::get_input_device_list() {
	PackedStringArray list;
	ERR_FAIL_COND_V(not _is_owner_thread(), list);

	list.push_back(DEFAULT_DEVICE_NAME);

	MABridge::get_device_list(_ma_data->context.get(), ma_device_type_capture, list);
	return list;
}

String AudioDriverMiniaudio::get_input_device() {
	String name;
	ERR_FAIL_COND_V(not _is_owner_thread(), name);
	_ma_data->capture.get_device_name(name);
	return name;
}

void AudioDriverMiniaudio::set_input_device(const String &p_name) {
	ERR_FAIL_COND(not _is_owner_thread());

	if (p_name == get_input_device()) {
		return;
	}

	if (p_name == DEFAULT_DEVICE_NAME) {
		_ma_data->capture.reinit(/* device id */ nullptr, _in_should_be_started);
	} else {
		ma_device_info device_info = {};
		ERR_FAIL_COND_MSG(not MABridge::find_device_info(_ma_data->context.get(), ma_device_type_capture, p_name, device_info),
				vformat("AudioDriverMiniaudio: Failed to find device %s to switch to.", p_name));

		_ma_data->capture.reinit(&device_info.id, _in_should_be_started);
	}
}

bool AudioDriverMiniaudio::_init_input_ring_buffer() {
	if (not _ma_data->capture.is_valid()) {
		return false;
	}

	ma_result result = MA_SUCCESS;
	{
		auto *new_rb = memnew(ma_pcm_rb);

		// Initialize the ring buffer.
#if GDT_MINIAUDIO_V12
		ma_audio_ring_buffer_config config =
				ma_audio_ring_buffer_config_init(
						_ma_data->capture.device->capture.format,
						_ma_data->capture.device->capture.channels,
						_ma_data->capture.get_sample_rate(),
						PERIOD_FRAME_COUNT * 4 // buffer size
				);
		config.pAllocationCallbacks = &_ma_data->context->allocationCallbacks;
		result = ma_audio_ring_buffer_init(&config, new_rb);
#else
		result = ma_pcm_rb_init(
				_ma_data->capture.device->capture.format,
				_ma_data->capture.device->capture.channels,
				PERIOD_FRAME_COUNT * 4, // buffer size
				nullptr, // optional preallocated buffer
				&_ma_data->context->allocationCallbacks,
				new_rb);
#endif

		if (result != MA_SUCCESS) {
			memdelete(new_rb);
			new_rb = nullptr;
		}

		ma_pcm_rb *old_rb = _ma_data->input_ring_buffer.non_rt_exchage(new_rb);
		DEV_ASSERT(old_rb == _ma_data->current_input_ring_buffer);
		_ma_data->current_input_ring_buffer = new_rb;

		if (old_rb) {
#if GDT_MINIAUDIO_V12
			ma_audio_ring_buffer_uninit(old_rb);
#else
			ma_pcm_rb_uninit(old_rb);
#endif
			memdelete(old_rb);
		}
	}

	if (result != MA_SUCCESS) {
		ERR_FAIL_V_MSG(false,
				vformat("AudioDriverMiniaudio: Failed to initialize ring buffer for capture device. %s",
						ma_result_description(result)));
	}

	return true;
}

void AudioDriverMiniaudio::_audio_server_update(void *p_userdata) {
	auto *driver = static_cast<AudioDriverMiniaudio *>(p_userdata);
	driver->_process_main_thread_update();
}

void AudioDriverMiniaudio::_process_main_thread_update() {
	ERR_FAIL_COND(not _is_owner_thread());

#if GDT_MA_AUDIO_DRIVER_DEBUG
	static uint64_t last_input_sb_overrun = 0;
	static uint64_t last_input_rb_overrun = 0;

	const uint64_t new_input_sb_overrun = _input_sb_overrun.get();
	if (new_input_sb_overrun != last_input_sb_overrun) {
		ERR_PRINT(vformat("AudioDriverMiniaudio: Input stage buffer overrun detected. SB overrun count: %d", new_input_sb_overrun));
		last_input_sb_overrun = new_input_sb_overrun;
	}

	const uint64_t new_input_rb_overrun = _input_rb_overrun.get();
	if (new_input_rb_overrun != last_input_rb_overrun) {
		ERR_PRINT(vformat("AudioDriverMiniaudio: Input ring buffer overrun detected. RB overrun count: %d", new_input_rb_overrun));
		last_input_rb_overrun = new_input_rb_overrun;
	}
#endif

	if (!_out_should_be_started) {
		_ma_data->playback.was_rerouted.clear();
		_ma_data->playback.retry_reinit_flag.clear();
		g_default_output_device_changed.clear();

		_ma_data->capture.was_rerouted.clear();
		_ma_data->capture.retry_reinit_flag.clear();
		g_default_input_device_changed.clear();
		return;
	}

	using Context = typename MADeviceHandler::UpdateContext;

	Context out_upd_context;
	out_upd_context.default_device_changed = g_default_output_device_changed.clear_if_set();
	out_upd_context.should_be_started = true;
	_ma_data->playback.handle_update(out_upd_context);

	if (_in_should_be_started) {
		Context in_upd_context;
		in_upd_context.default_device_changed = g_default_input_device_changed.clear_if_set();
		in_upd_context.should_be_started = _in_should_be_started;
		_ma_data->capture.handle_update(in_upd_context);
	}
}
