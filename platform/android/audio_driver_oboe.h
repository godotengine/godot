/**************************************************************************/
/*  audio_driver_oboe.h                                                   */
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

#ifndef AUDIO_DRIVER_OBOE_H
#define AUDIO_DRIVER_OBOE_H

#include "core/os/mutex.h"
#include "servers/audio_server.h"

#include "oboe/Oboe.h"

#include <jni.h>

class AudioDriverOboe : public AudioDriver {
	class AudioStreamOboe {
	protected:
		// https://developer.android.com/reference/android/media/AudioManager#getDevices(int)
		enum DeviceFlag {
			GET_DEVICES_NONE = 0,
			GET_DEVICES_INPUTS = 1 << 0,
			GET_DEVICES_OUTPUTS = 1 << 1,
		};

		// https://developer.android.com/reference/android/media/AudioManager#ACTION_SCO_AUDIO_STATE_UPDATED
		enum SCOState {
			SCO_AUDIO_STATE_DISCONNECTED,
			SCO_AUDIO_STATE_CONNECTED,
			SCO_AUDIO_STATE_CONNECTING,
		};

		// https://developer.android.com/reference/android/media/AudioDeviceInfo#getType()
		// TYPE_ECHO_REFERENCE: https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/media/java/android/media/AudioDeviceInfo.java
		enum DeviceType {
			TYPE_UNKNOWN,
			TYPE_BUILTIN_EARPIECE,
			TYPE_BUILTIN_SPEAKER,
			TYPE_WIRED_HEADSET,
			TYPE_WIRED_HEADPHONES,
			TYPE_LINE_ANALOG,
			TYPE_LINE_DIGITAL,
			TYPE_BLUETOOTH_SCO,
			TYPE_BLUETOOTH_A2DP,
			TYPE_HDMI,
			TYPE_HDMI_ARC,
			TYPE_USB_DEVICE,
			TYPE_USB_ACCESSORY,
			TYPE_DOCK,
			TYPE_FM,
			TYPE_BUILTIN_MIC,
			TYPE_FM_TUNER,
			TYPE_TV_TUNER,
			TYPE_TELEPHONY,
			TYPE_AUX_LINE,
			TYPE_IP,
			TYPE_BUS,
			TYPE_USB_HEADSET,
			TYPE_HEARING_AID,
			TYPE_BUILTIN_SPEAKER_SAFE,
			TYPE_REMOTE_SUBMIX,
			TYPE_BLE_HEADSET,
			TYPE_BLE_SPEAKER,
			TYPE_ECHO_REFERENCE,
			TYPE_HDMI_EARC,
			TYPE_BLE_BROADCAST,
			TYPE_DOCK_ANALOG,
			TYPE_MAX,
		};

		enum class SCOResult {
			CONTINUE,
			// Wait for set_sco_permission or sco_state_updated.
			WAIT,
			FAIL,
		};

		enum class PostOpenResult {
			CONTINUE,
			REOPEN,
			FAIL,
		};

		static String type_to_string[TYPE_MAX];
		static oboe::Usage type_to_usage[TYPE_MAX];

		static int sco_streams_count;

		static bool is_sco_started;
		static bool is_communication_device_set;

		static SCOState sco_state;

		bool playing = false;

		oboe::Direction direction = oboe::Direction::Output;
		DeviceFlag device_flag = GET_DEVICES_NONE;

		int request_channel_count = 0;
		int request_mix_rate = 0;

		int32_t device_id = 0;
		DeviceType device_type = TYPE_UNKNOWN;
		String device_name = "Default";
		jobject device_info = nullptr;

		bool stream_using_sco = false;

		std::shared_ptr<oboe::AudioStreamDataCallback> data_callback;
		std::shared_ptr<oboe::AudioStreamErrorCallback> error_callback;

		std::shared_ptr<oboe::AudioStream> stream;

		static bool _is_device_unsupported(DeviceType p_type, JNIEnv *p_env);
		static String _get_device_name(jobject p_device, JNIEnv *p_env, DeviceType *r_type = nullptr);

		jobject _find_device_info(int p_id, JNIEnv *p_env);
		void _set_device(int p_id, JNIEnv *p_env);
		void _reset_request_values();

		bool _open_stream(JNIEnv *p_env);
		SCOResult _close_stream(JNIEnv *p_env, bool p_modify_sco = true);

		SCOResult _init_sco(JNIEnv *p_env);
		SCOResult _deinit_sco(JNIEnv *p_env);

		SCOResult _set_communication_device(JNIEnv *p_env);
		void _clear_communication_device(JNIEnv *p_env);

		SCOResult _start_sco(JNIEnv *p_env);
		void _stop_sco(JNIEnv *p_env);

		bool _is_paused(oboe::StreamState *r_state = nullptr) const;

		virtual int _get_prefered_mix_rate() const = 0;
		virtual int _get_default_mix_rate() const = 0;

		virtual jobject _get_communication_device(JNIEnv *p_env) = 0;

		virtual PostOpenResult _post_open_stream() = 0;

		AudioStreamOboe(oboe::Direction p_direction, DeviceFlag p_device_flag, oboe::AudioStreamDataCallback *p_data_callback, oboe::AudioStreamErrorCallback *p_error_callback);
		virtual ~AudioStreamOboe() = default;

	public:
		virtual Error start_stream();
		void pause_stream();
		void close_stream();

		PackedStringArray get_device_list();
		void set_device(const String &p_name);
		String get_device();
		void reset_device();

		bool is_playing() const;
		float get_latency() const;
		int get_mix_rate() const;

		static void set_sco_state(int p_sco_state);

		void set_sco_permission(bool p_result);
		void sco_state_updated();
	};

	class OutputAudioStreamOboe : public AudioStreamOboe {
	private:
		class ErrorCallback : public oboe::AudioStreamErrorCallback {
		public:
			virtual bool onError(oboe::AudioStream *p_audio_stream, oboe::Result p_error) override;
		};

		class DataCallback : public oboe::AudioStreamDataCallback {
		public:
			virtual oboe::DataCallbackResult onAudioReady(oboe::AudioStream *p_audio_stream, void *p_audio_data, int32_t p_frames) override;
		};

		oboe::LatencyTuner *latency_tuner = nullptr;

		virtual jobject _get_communication_device(JNIEnv *p_env) override;

		virtual int _get_prefered_mix_rate() const override;
		virtual int _get_default_mix_rate() const override;

		virtual PostOpenResult _post_open_stream() override;

	public:
		OutputAudioStreamOboe();
		~OutputAudioStreamOboe();
	};

	class InputAudioStreamOboe : public AudioStreamOboe {
	private:
		class ErrorCallback : public oboe::AudioStreamErrorCallback {
		public:
			virtual bool onError(oboe::AudioStream *p_audio_stream, oboe::Result p_error) override;
		};

		class DataCallback : public oboe::AudioStreamDataCallback {
		public:
			virtual oboe::DataCallbackResult onAudioReady(oboe::AudioStream *p_audio_stream, void *p_audio_data, int32_t p_frames) override;
		};

		virtual jobject _get_communication_device(JNIEnv *p_env) override;

		virtual int _get_prefered_mix_rate() const override;
		virtual int _get_default_mix_rate() const override;

		virtual PostOpenResult _post_open_stream() override;

	public:
		virtual Error start_stream() override;

		InputAudioStreamOboe();
	};

	static AudioDriverOboe *singleton;

	jobject manager = nullptr;

	jmethodID _manager_get_devices = nullptr;
	jmethodID _manager_is_bluetooth_sco_available_off_call = nullptr;

	jmethodID _manager_get_available_communication_devices = nullptr;
	jmethodID _manager_get_communication_device = nullptr;
	jmethodID _manager_set_communication_device = nullptr;
	jmethodID _manager_clear_communication_device = nullptr;

	jmethodID _manager_start_bluetooth_sco = nullptr;
	jmethodID _manager_stop_bluetooth_sco = nullptr;

	jmethodID _device_info_equals = nullptr;
	jmethodID _device_info_get_product_name = nullptr;
	jmethodID _device_info_get_id = nullptr;
	jmethodID _device_info_get_type = nullptr;
	jmethodID _device_info_get_address = nullptr;

	bool pause = false;

	Mutex mutex;

	Vector<int32_t> buffer;

	OutputAudioStreamOboe audio_output;
	InputAudioStreamOboe audio_input;

	int request_mix_rate = 0;
	int channel_count = 0;

	static int32_t _read_sample(oboe::AudioFormat p_foramt, void *p_audio_data, int i);
	static void _write_sample(oboe::AudioFormat p_foramt, void *p_audio_data, int i, int32_t p_sample);

public:
	virtual const char *get_name() const override {
		return "Android";
	}

	virtual Error init() override;
	virtual void start() override;
	virtual int get_mix_rate() const override;
	virtual SpeakerMode get_speaker_mode() const override;
	virtual float get_latency() override;

	virtual void lock() override;
	virtual void unlock() override;
	virtual void finish() override;

	virtual PackedStringArray get_output_device_list() override;
	virtual String get_output_device() override;
	virtual void set_output_device(const String &p_name) override;

	virtual Error input_start() override;
	virtual Error input_stop() override;

	virtual PackedStringArray get_input_device_list() override;
	virtual String get_input_device() override;
	virtual void set_input_device(const String &p_name) override;

	bool try_lock();

	void set_pause(bool p_pause);

	void _java_sco_callback(int p_sco_state);
	void request_permission_result(const String &p_permission, bool p_result);
	void setup(jobject p_audio_manager);

	_FORCE_INLINE_ static AudioDriverOboe *get_singleton() { return singleton; }

	AudioDriverOboe() { singleton = this; }
};

#endif // AUDIO_DRIVER_OBOE_H
