/**************************************************************************/
/*  audio_driver_oboe.cpp                                                 */
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

#include "audio_driver_oboe.h"

#include "java_godot_wrapper.h"

// https://developer.android.com/reference/android/Manifest.permission
#define PERMISSION_RECORD_AUDIO "android.permission.RECORD_AUDIO"
#define PERMISSION_MODIFY_AUDIO_SETTINGS "android.permission.MODIFY_AUDIO_SETTINGS"

// https://developer.android.com/reference/android/media/AudioManager#getProperty(java.lang.String)
#define PROPERTY_OUTPUT_SAMPLE_RATE "android.media.property.OUTPUT_SAMPLE_RATE"
#define PROPERTY_OUTPUT_FRAMES_PER_BUFFER "android.media.property.OUTPUT_FRAMES_PER_BUFFER"

String AudioDriverOboe::AudioStreamOboe::type_to_string[TYPE_MAX] = {
	"unknown", // TYPE_UNKNOWN
	"built-in_earpiece", // TYPE_BUILTIN_EARPIECE
	"built-in_speaker", // TYPE_BUILTIN_SPEAKER
	"wired_headset", // TYPE_WIRED_HEADSET
	"wired_headphones", // TYPE_WIRED_HEADPHONES
	"line_analog", // TYPE_LINE_ANALOG
	"line_digital", // TYPE_LINE_DIGITAL
	"bluetooth_sco", // TYPE_BLUETOOTH_SCO
	"bluetooth_a2dp", // TYPE_BLUETOOTH_A2DP
	"hdmi", // TYPE_HDMI
	"hdmi_audio_return_channel", // TYPE_HDMI_ARC
	"usb_device", // TYPE_USB_DEVICE
	"usb_accessory", // TYPE_USB_ACCESSORY
	"dock", // TYPE_DOCK
	"fm", // TYPE_FM
	"built-in_microphone", // TYPE_BUILTIN_MIC
	"fm_tuner", // TYPE_FM_TUNER
	"tv_tuner", // TYPE_TV_TUNER
	"telephony", // TYPE_TELEPHONY
	"auxiliary_line", // TYPE_AUX_LINE
	"ip", // TYPE_IP
	"bus", // TYPE_BUS
	"usb_headset", // TYPE_USB_HEADSET
	"hearing_aid", // TYPE_HEARING_AID
	"built-in_speaker_safe", // TYPE_BUILTIN_SPEAKER_SAFE
	"remote_submix", // TYPE_REMOTE_SUBMIX
	"bluetooth_low_energy_headset", // TYPE_BLE_HEADSET
	"bluetooth_low_energy_speaker", // TYPE_BLE_SPEAKER
	"echo_reference", // TYPE_ECHO_REFERENCE
	"hdmi_enhanced_audio_return_channel", // TYPE_HDMI_EARC
	"bluetooth_low_energy_broadcast", // TYPE_BLE_BROADCAST
	"dock_analog", // TYPE_DOCK_ANALOG
};

oboe::Usage AudioDriverOboe::AudioStreamOboe::type_to_usage[TYPE_MAX] = {
	oboe::Usage::Game, // TYPE_UNKNOWN
	oboe::Usage::VoiceCommunication, // TYPE_BUILTIN_EARPIECE
	oboe::Usage::Game, // TYPE_BUILTIN_SPEAKER
	oboe::Usage::Game, // TYPE_WIRED_HEADSET
	oboe::Usage::Game, // TYPE_WIRED_HEADPHONES
	oboe::Usage::Game, // TYPE_LINE_ANALOG
	oboe::Usage::Game, // TYPE_LINE_DIGITAL
	oboe::Usage::Game, // TYPE_BLUETOOTH_SCO
	oboe::Usage::Game, // TYPE_BLUETOOTH_A2DP
	oboe::Usage::Game, // TYPE_HDMI
	oboe::Usage::Game, // TYPE_HDMI_ARC
	oboe::Usage::Game, // TYPE_USB_DEVICE
	oboe::Usage::Game, // TYPE_USB_ACCESSORY
	oboe::Usage::Game, // TYPE_DOCK
	oboe::Usage::Game, // TYPE_FM
	oboe::Usage::Game, // TYPE_BUILTIN_MIC
	oboe::Usage::Game, // TYPE_FM_TUNER
	oboe::Usage::Game, // TYPE_TV_TUNER
	oboe::Usage::VoiceCommunication, // TYPE_TELEPHONY
	oboe::Usage::Game, // TYPE_AUX_LINE
	oboe::Usage::VoiceCommunication, // TYPE_IP
	oboe::Usage::Game, // TYPE_BUS
	oboe::Usage::Game, // TYPE_USB_HEADSET
	oboe::Usage::Game, // TYPE_HEARING_AID
	oboe::Usage::Notification, // TYPE_BUILTIN_SPEAKER_SAFE
	oboe::Usage::Game, // TYPE_REMOTE_SUBMIX
	oboe::Usage::Game, // TYPE_BLE_HEADSET
	oboe::Usage::Game, // TYPE_BLE_SPEAKER
	oboe::Usage::Game, // TYPE_ECHO_REFERENCE
	oboe::Usage::Game, // TYPE_HDMI_EARC
	oboe::Usage::Game, // TYPE_BLE_BROADCAST
	oboe::Usage::Game, // TYPE_DOCK_ANALOG
};

int AudioDriverOboe::AudioStreamOboe::sco_streams_count = 0;
bool AudioDriverOboe::AudioStreamOboe::is_sco_started = false;
bool AudioDriverOboe::AudioStreamOboe::is_communication_device_set = false;

AudioDriverOboe::AudioStreamOboe::SCOState AudioDriverOboe::AudioStreamOboe::sco_state = SCO_AUDIO_STATE_DISCONNECTED;

AudioDriverOboe *AudioDriverOboe::singleton = nullptr;

bool AudioDriverOboe::AudioStreamOboe::_open_stream(JNIEnv *p_env) {
	oboe::Result res;

	oboe::Usage usage = device_type < TYPE_MAX ? type_to_usage[device_type] : type_to_usage[TYPE_UNKNOWN];

	oboe::AudioStreamBuilder builder;
	builder.setDirection(direction);
	builder.setDeviceId(device_id);
	builder.setChannelCount(request_channel_count);
	builder.setSampleRate(request_mix_rate);
	builder.setUsage(usage);
	builder.setDataCallback(data_callback);
	builder.setErrorCallback(error_callback);

	// Low latency.
	builder.setPerformanceMode(oboe::PerformanceMode::LowLatency);
	builder.setSharingMode(oboe::SharingMode::Exclusive);

	res = builder.openStream(stream);
	if (res != oboe::Result::OK) {
		if (request_mix_rate == _get_default_mix_rate()) {
			ERR_FAIL_COND_V_MSG(device_id == 0, false, vformat("Oboe: Can't open an %s stream.", direction == oboe::Direction::Output ? "output" : "input"));

			reset_device();
			SCOResult res = _deinit_sco(p_env);
			return res == SCOResult::WAIT ? true : _open_stream(p_env);
		}

		request_mix_rate = _get_default_mix_rate();
		return _open_stream(p_env);
	}

	stream->setPerformanceHintEnabled(true);

	switch (_post_open_stream()) {
		case PostOpenResult::REOPEN: {
			// Modify SCO only if device was reset.
			SCOResult res = _close_stream(p_env, device_id == 0);
			return res == SCOResult::WAIT ? true : _open_stream(p_env);
		}

		case PostOpenResult::FAIL:
			_close_stream(p_env);
			return false;

		default:
			break;
	}

	res = stream->requestStart();
	if (res != oboe::Result::OK) {
		SCOResult res = _close_stream(p_env);

		ERR_FAIL_COND_V_MSG(device_id == 0, false, vformat("Oboe: Can't start an %s stream.", direction == oboe::Direction::Output ? "output" : "input"));

		reset_device();
		return res == SCOResult::WAIT ? true : _open_stream(p_env);
	}
	return true;
}

Error AudioDriverOboe::AudioStreamOboe::start_stream() {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);

	playing = true;

	if (ad->pause)
		return OK;

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, ERR_CANT_OPEN);

	oboe::StreamState state = oboe::StreamState::Uninitialized;
	if (_is_paused(&state)) {
		// Locked before ErrorCallback::onError.
		if (state == oboe::StreamState::Disconnected) {
			reset_device();

			SCOResult res = _close_stream(env);
			if (res == SCOResult::WAIT)
				return OK;
		}

		switch (_init_sco(env)) {
			case SCOResult::WAIT:
				return OK;

			case SCOResult::FAIL:
				reset_device();
				break;

			default:
				break;
		}

		// Device might have changed.
		_reset_request_values();

		if (!_open_stream(env))
			return ERR_CANT_OPEN;

		if (device_id != 0 && device_id != stream->getDeviceId())
			_set_device(stream->getDeviceId(), env);
	}
	return OK;
}

AudioDriverOboe::AudioStreamOboe::SCOResult AudioDriverOboe::AudioStreamOboe::_init_sco(JNIEnv *p_env) {
	if (device_type == TYPE_BLUETOOTH_SCO) {
		SCOResult res = _set_communication_device(p_env);
		switch (res) {
			case SCOResult::FAIL:
				res = _start_sco(p_env);
				break;

			case SCOResult::CONTINUE:
				if (sco_state != SCO_AUDIO_STATE_CONNECTED)
					res = SCOResult::WAIT;
				break;

			default:
				break;
		}

		if (res != SCOResult::FAIL && !stream_using_sco) {
			sco_streams_count++;
			stream_using_sco = true;
		}
		return res;
	}
	if (device_type == TYPE_BLUETOOTH_A2DP && sco_state != SCO_AUDIO_STATE_DISCONNECTED)
		return SCOResult::FAIL;
	return SCOResult::CONTINUE;
}

AudioDriverOboe::AudioStreamOboe::SCOResult AudioDriverOboe::AudioStreamOboe::_deinit_sco(JNIEnv *p_env) {
	if (stream_using_sco) {
		sco_streams_count--;
		stream_using_sco = false;
	}

	if (sco_streams_count == 0) {
		_clear_communication_device(p_env);
		_stop_sco(p_env);

		return sco_state == SCO_AUDIO_STATE_DISCONNECTED ? SCOResult::CONTINUE : SCOResult::WAIT;
	}

	return SCOResult::CONTINUE;
}

AudioDriverOboe::AudioStreamOboe::SCOResult AudioDriverOboe::AudioStreamOboe::_set_communication_device(JNIEnv *p_env) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (!ad->_manager_set_communication_device)
		return SCOResult::FAIL;

	jobject communication_device = _get_communication_device(p_env);
	if (communication_device) {
		jobject current_communication_device = p_env->CallObjectMethod(ad->manager, ad->_manager_get_communication_device);
		if (current_communication_device && p_env->CallBooleanMethod(current_communication_device, ad->_device_info_equals, communication_device))
			return SCOResult::CONTINUE;

		bool ret = p_env->CallBooleanMethod(ad->manager, ad->_manager_set_communication_device, communication_device);
		if (!p_env->ExceptionCheck()) {
			if (ret) {
				is_communication_device_set = true;
				return SCOResult::CONTINUE;
			}
		} else {
			p_env->ExceptionClear();
		}
	}

	_clear_communication_device(p_env);
	return SCOResult::FAIL;
}

void AudioDriverOboe::AudioStreamOboe::_clear_communication_device(JNIEnv *p_env) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (!is_communication_device_set)
		return;

	p_env->CallVoidMethod(ad->manager, ad->_manager_clear_communication_device);
	is_communication_device_set = false;
}

AudioDriverOboe::AudioStreamOboe::SCOResult AudioDriverOboe::AudioStreamOboe::_start_sco(JNIEnv *p_env) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (!ad->_manager_start_bluetooth_sco)
		return SCOResult::FAIL;

	switch (sco_state) {
		case SCO_AUDIO_STATE_CONNECTED:
			return SCOResult::CONTINUE;

		case SCO_AUDIO_STATE_CONNECTING:
			return SCOResult::WAIT;

		default:
			if (OS::get_singleton()->request_permission(PERMISSION_MODIFY_AUDIO_SETTINGS)) {
				p_env->CallVoidMethod(ad->manager, ad->_manager_start_bluetooth_sco);
				is_sco_started = true;
			} else {
				WARN_PRINT("Oboe: Unable to start Bluetooth SCO - No MODIFY_AUDIO_SETTINGS permission");
			}
			return SCOResult::WAIT;
	}
}

void AudioDriverOboe::AudioStreamOboe::_stop_sco(JNIEnv *p_env) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (!is_sco_started)
		return;

	p_env->CallVoidMethod(ad->manager, ad->_manager_stop_bluetooth_sco);
	is_sco_started = false;
}

AudioDriverOboe::AudioStreamOboe::SCOResult AudioDriverOboe::AudioStreamOboe::_close_stream(JNIEnv *p_env, bool p_modify_sco) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);

	if (stream)
		stream->close();

	if (p_modify_sco)
		return _deinit_sco(p_env);
	return SCOResult::CONTINUE;
}

void AudioDriverOboe::AudioStreamOboe::pause_stream() {
	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);

	_close_stream(env);
}

void AudioDriverOboe::AudioStreamOboe::close_stream() {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);

	playing = false;
	_close_stream(env);
}

void AudioDriverOboe::AudioStreamOboe::_reset_request_values() {
	request_channel_count = 0;
	request_mix_rate = _get_prefered_mix_rate();
}

bool AudioDriverOboe::AudioStreamOboe::_is_device_unsupported(DeviceType p_type, JNIEnv *p_env) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	return p_type == TYPE_ECHO_REFERENCE || p_type == TYPE_TELEPHONY || p_type == TYPE_REMOTE_SUBMIX ||
			(p_type == TYPE_BLUETOOTH_SCO && !p_env->CallBooleanMethod(ad->manager, ad->_manager_is_bluetooth_sco_available_off_call));
}

String AudioDriverOboe::AudioStreamOboe::_get_device_name(jobject p_device, JNIEnv *p_env, DeviceType *r_type) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	DeviceType type = (DeviceType)p_env->CallIntMethod(p_device, ad->_device_info_get_type);
	if (_is_device_unsupported(type, p_env))
		return String();

	String type_string = type < TYPE_MAX ? type_to_string[type] : type_to_string[TYPE_UNKNOWN];

	jobject product_name = p_env->CallObjectMethod(p_device, ad->_device_info_get_product_name);
	String product_name_string = char_sequence_to_string(product_name, p_env);

	String address_string;
	if (ad->_device_info_get_address) {
		jstring address = (jstring)p_env->CallObjectMethod(p_device, ad->_device_info_get_address);
		address_string = jstring_to_string(address, p_env);

		p_env->DeleteLocalRef(address);
	}

	p_env->DeleteLocalRef(product_name);

	if (r_type)
		*r_type = type;

	if (address_string.is_empty())
		return vformat("%s.%s", product_name_string, type_string);
	return vformat("%s.%s.%s", product_name_string, type_string, address_string);
}

jobject AudioDriverOboe::AudioStreamOboe::_find_device_info(int p_id, JNIEnv *p_env) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (!oboe::AudioStreamBuilder::isAAudioRecommended())
		return nullptr;

	jobjectArray device_info_array = (jobjectArray)p_env->CallObjectMethod(ad->manager, ad->_manager_get_devices, device_flag);

	jobject ret = nullptr;
	int size = p_env->GetArrayLength(device_info_array);
	for (int i = 0; i < size; i++) {
		jobject device_info = p_env->GetObjectArrayElement(device_info_array, i);

		int device_id = p_env->CallIntMethod(device_info, ad->_device_info_get_id);

		if (device_id == p_id) {
			ret = device_info;
			break;
		}

		p_env->DeleteLocalRef(device_info);
	}

	p_env->DeleteLocalRef(device_info_array);
	return ret;
}

void AudioDriverOboe::AudioStreamOboe::_set_device(int p_id, JNIEnv *p_env) {
	jobject found_device_info = _find_device_info(p_id, p_env);
	if (found_device_info == nullptr)
		return;

	device_id = p_id;
	device_name = _get_device_name(found_device_info, p_env, &device_type);

	if (device_type == TYPE_BLUETOOTH_SCO && !stream_using_sco) {
		sco_streams_count++;
		stream_using_sco = true;
	}

	if (device_info)
		p_env->DeleteGlobalRef(device_info);
	device_info = p_env->NewGlobalRef(found_device_info);

	p_env->DeleteLocalRef(found_device_info);
}

PackedStringArray AudioDriverOboe::AudioStreamOboe::get_device_list() {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (!oboe::AudioStreamBuilder::isAAudioRecommended())
		return { "Default" };

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, { "Default" });

	PackedStringArray device_list = { String() };

	jobjectArray device_info_array = (jobjectArray)env->CallObjectMethod(ad->manager, ad->_manager_get_devices, device_flag);

	int device_size = env->GetArrayLength(device_info_array);
	device_list.resize(device_size + 1);

	int j = 1;
	for (int i = 0; i < device_size; i++) {
		jobject device_info = env->GetObjectArrayElement(device_info_array, i);

		String device_name = _get_device_name(device_info, env);
		if (!device_name.is_empty()) {
			device_list.write[j] = device_name;
			j++;
		}

		env->DeleteLocalRef(device_info);
	}

	env->DeleteLocalRef(device_info_array);

	device_list.resize(j);
	device_list.sort();

	// The first element will always be String() that was added at the beginning.
	device_list.write[0] = "Default";
	return device_list;
}

void AudioDriverOboe::AudioStreamOboe::set_device(const String &p_name) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (!oboe::AudioStreamBuilder::isAAudioRecommended() || device_name == p_name)
		return;

	int32_t new_device_id = 0;
	DeviceType new_device_type = TYPE_UNKNOWN;
	String new_device_name = "Default";
	jobject new_device = nullptr;

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);

	if (!p_name.is_empty() && p_name != new_device_name) {
		jobjectArray device_info_array = (jobjectArray)env->CallObjectMethod(ad->manager, ad->_manager_get_devices, device_flag);

		int device_size = env->GetArrayLength(device_info_array);
		for (int i = 0; i < device_size; i++) {
			jobject device_info = env->GetObjectArrayElement(device_info_array, i);

			DeviceType device_info_type = TYPE_UNKNOWN;
			String device_info_name = _get_device_name(device_info, env, &device_info_type);

			if (p_name == device_info_name) {
				new_device_id = env->CallIntMethod(device_info, ad->_device_info_get_id);
				new_device_type = device_info_type;
				new_device_name = p_name;
				new_device = device_info;
				break;
			}

			env->DeleteLocalRef(device_info);
		}

		env->DeleteLocalRef(device_info_array);
	}

	if (new_device_id != device_id) {
		MutexLock lock(ad->mutex);

		device_id = new_device_id;
		device_type = new_device_type;
		device_name = new_device_name;

		if (device_info)
			env->DeleteGlobalRef(device_info);
		device_info = new_device ? env->NewGlobalRef(new_device) : nullptr;

		if (_is_paused() || new_device_id != stream->getDeviceId()) {
			_reset_request_values();

			SCOResult res = _close_stream(env, new_device_type != TYPE_BLUETOOTH_SCO);
			if ((res != SCOResult::WAIT || (device_id != 0 && device_type != TYPE_BLUETOOTH_A2DP)) && playing)
				start_stream();
		} else {
			if (new_device_type == TYPE_BLUETOOTH_SCO && !stream_using_sco) {
				sco_streams_count++;
				stream_using_sco = true;
			}
			if (new_device_type != TYPE_BLUETOOTH_SCO && stream_using_sco) {
				sco_streams_count--;
				stream_using_sco = false;
			}
		}
	}

	if (new_device)
		env->DeleteLocalRef(new_device);
	return;
}

String AudioDriverOboe::AudioStreamOboe::get_device() {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);
	return device_name;
}

void AudioDriverOboe::AudioStreamOboe::reset_device() {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (device_id == 0)
		return;

	MutexLock lock(ad->mutex);

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);

	device_id = 0;
	device_type = TYPE_UNKNOWN;
	device_name = "Default";

	if (device_info) {
		env->DeleteGlobalRef(device_info);
		device_info = nullptr;
	}

	_reset_request_values();
}

bool AudioDriverOboe::AudioStreamOboe::is_playing() const {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);
	return playing;
}

bool AudioDriverOboe::AudioStreamOboe::_is_paused(oboe::StreamState *r_state) const {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);
	oboe::StreamState state = stream ? stream->getState() : oboe::StreamState::Uninitialized;
	if (r_state)
		*r_state = state;
	return state == oboe::StreamState::Uninitialized || state == oboe::StreamState::Unknown ||
			state == oboe::StreamState::Disconnected ||
			state == oboe::StreamState::Stopping || state == oboe::StreamState::Stopped ||
			state == oboe::StreamState::Closing || state == oboe::StreamState::Closed;
}

float AudioDriverOboe::AudioStreamOboe::get_latency() const {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);
	if (stream) {
		oboe::ResultWithValue<double> res = stream->calculateLatencyMillis();
		if (res)
			return res.value() > 0 ? res.value() / 1000.0 : 0.0;
	}
	return 0;
}

int AudioDriverOboe::AudioStreamOboe::get_mix_rate() const {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);
	return stream ? stream->getSampleRate() : _get_prefered_mix_rate();
}

void AudioDriverOboe::AudioStreamOboe::set_sco_permission(bool p_result) {
	if (device_type != TYPE_BLUETOOTH_SCO)
		return;

	if (!p_result)
		reset_device();

	if (playing)
		start_stream();
}

void AudioDriverOboe::AudioStreamOboe::set_sco_state(int p_sco_state) {
	sco_state = (SCOState)p_sco_state;
}

void AudioDriverOboe::AudioStreamOboe::sco_state_updated() {
	if (!oboe::AudioStreamBuilder::isAAudioRecommended())
		return;

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);

	switch (sco_state) {
		case SCO_AUDIO_STATE_CONNECTED:
			if (playing)
				start_stream();
			break;

		case SCO_AUDIO_STATE_DISCONNECTED:
			if (playing)
				start_stream();
			break;

		default:
			break;
	}
}

AudioDriverOboe::AudioStreamOboe::AudioStreamOboe(oboe::Direction p_direction, DeviceFlag p_device_flag, oboe::AudioStreamDataCallback *p_data_callback, oboe::AudioStreamErrorCallback *p_error_callback) :
		direction(p_direction), device_flag(p_device_flag), data_callback(p_data_callback), error_callback(p_error_callback) {
}

jobject AudioDriverOboe::OutputAudioStreamOboe::_get_communication_device(JNIEnv *p_env) {
	return device_info;
}

int AudioDriverOboe::OutputAudioStreamOboe::_get_prefered_mix_rate() const {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	return ad->request_mix_rate;
}

int AudioDriverOboe::OutputAudioStreamOboe::_get_default_mix_rate() const {
	return 0;
}

AudioDriverOboe::AudioStreamOboe::PostOpenResult AudioDriverOboe::OutputAudioStreamOboe::_post_open_stream() {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	switch (stream->getChannelCount()) {
		case 1: // Mono
		case 3: // Surround 2.1
		case 5: // Surround 5.0
		case 7: // Surround 7.0
			ad->channel_count = stream->getChannelCount() + 1;
			break;

		case 2: // Stereo
		case 4: // Surround 3.1
		case 6: // Surround 5.1
		case 8: // Surround 7.1
			ad->channel_count = stream->getChannelCount();
			break;

		default:
			if (request_channel_count == 2) {
				if (device_id == 0) {
					WARN_PRINT(vformat("Oboe: Unsupported number of channels: %d", stream->getChannelCount()));
					ad->channel_count = 2;
					break;
				}

				reset_device();
				return PostOpenResult::REOPEN;
			}

			request_channel_count = 2;
			return PostOpenResult::REOPEN;
	}

	memdelete_notnull(latency_tuner);
	latency_tuner = memnew(oboe::LatencyTuner(*stream));

	// See AudioDriverOboe::OutputAudioStreamOboe::DataCallback::onAudioReady.
	if (ad->channel_count != stream->getChannelCount() || stream->getFormat() != oboe::AudioFormat::I32)
		ad->buffer.resize(stream->getFramesPerBurst() * ad->channel_count);

	ad->audio_server_init_channels_and_buffers();
	return PostOpenResult::CONTINUE;
}

bool AudioDriverOboe::OutputAudioStreamOboe::ErrorCallback::onError(oboe::AudioStream *p_audio_stream, oboe::Result p_error) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);

	// start_stream got the lock before.
	if (p_audio_stream != ad->audio_output.stream.get())
		return true;

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, false);

	if (p_error == oboe::Result::ErrorDisconnected)
		ad->audio_output.reset_device();

	// Modify SCO only if device is reset.
	SCOResult res = ad->audio_output._close_stream(env, p_error == oboe::Result::ErrorDisconnected);
	if (res != SCOResult::WAIT)
		ad->audio_output.start_stream();
	return true;
}

oboe::DataCallbackResult AudioDriverOboe::OutputAudioStreamOboe::DataCallback::onAudioReady(oboe::AudioStream *p_audio_stream, void *p_audio_data, int32_t p_frames) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (ad->pause || !ad->try_lock()) {
		for (int32_t i = 0; i < p_frames * p_audio_stream->getChannelCount(); i++) {
			AudioDriverOboe::_write_sample(p_audio_stream->getFormat(), p_audio_data, i, 0);
		}
		return oboe::DataCallbackResult::Continue;
	}

	ad->start_counting_ticks();

	if (ad->channel_count == p_audio_stream->getChannelCount()) {
		if (p_audio_stream->getFormat() == oboe::AudioFormat::I32) {
			ad->audio_server_process(p_frames, (int32_t *)p_audio_data);
		} else {
			ad->audio_server_process(p_frames, ad->buffer.ptrw());

			for (int32_t i = 0; i < p_frames * p_audio_stream->getChannelCount(); i++) {
				AudioDriverOboe::_write_sample(p_audio_stream->getFormat(), p_audio_data, i, ad->buffer[i]);
			}
		}
	} else if (ad->channel_count == p_audio_stream->getChannelCount() + 1) {
		ad->audio_server_process(p_frames, ad->buffer.ptrw());

		// Pass all channels except the last two as-is, and then mix the last two
		// together as one channel. E.g. stereo -> mono, or 3.1 -> 2.1.
		int32_t last_chan = p_audio_stream->getChannelCount() - 1;
		for (int32_t i = 0; i < p_frames; i++) {
			for (int32_t j = 0; j < last_chan; j++) {
				AudioDriverOboe::_write_sample(p_audio_stream->getFormat(), p_audio_data, i * p_audio_stream->getChannelCount() + j, ad->buffer[i * ad->channel_count + j]);
			}
			int32_t l = ad->buffer[i * ad->channel_count + last_chan];
			int32_t r = ad->buffer[i * ad->channel_count + last_chan + 1];
			int32_t c = (int32_t)(((int64_t)l + (int64_t)r) / 2);
			AudioDriverOboe::_write_sample(p_audio_stream->getFormat(), p_audio_data, i * p_audio_stream->getChannelCount() + last_chan, c);
		}
	} else {
		for (int32_t i = 0; i < p_frames; i++) {
			for (int32_t j = 0; j < ad->channel_count; j++) {
				AudioDriverOboe::_write_sample(p_audio_stream->getFormat(), p_audio_data, i * p_audio_stream->getChannelCount() + j, ad->buffer[i * ad->channel_count + j]);
			}
			for (int32_t j = ad->channel_count; j < p_audio_stream->getChannelCount(); j++) {
				AudioDriverOboe::_write_sample(p_audio_stream->getFormat(), p_audio_data, i * p_audio_stream->getChannelCount() + j, 0);
			}
		}
	}

	ad->stop_counting_ticks();

	ad->audio_output.latency_tuner->tune();
	ad->unlock();
	return oboe::DataCallbackResult::Continue;
}

AudioDriverOboe::OutputAudioStreamOboe::OutputAudioStreamOboe() :
		AudioStreamOboe(oboe::Direction::Output, GET_DEVICES_OUTPUTS, memnew(DataCallback), memnew(ErrorCallback)) {
}

AudioDriverOboe::OutputAudioStreamOboe::~OutputAudioStreamOboe() {
	memdelete_notnull(latency_tuner);
}

jobject AudioDriverOboe::InputAudioStreamOboe::_get_communication_device(JNIEnv *p_env) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	jobject communication_devices = p_env->CallObjectMethod(ad->manager, ad->_manager_get_available_communication_devices);

	jclass list_class = p_env->GetObjectClass(communication_devices);

	jmethodID _list_size = p_env->GetMethodID(list_class, "size", "()I");
	jmethodID _list_get = p_env->GetMethodID(list_class, "get", "(I)Ljava/lang/Object;");

	jobject ret = nullptr;
	int size = p_env->CallIntMethod(communication_devices, _list_size);
	for (int i = 0; i < size; i++) {
		jobject device_info = p_env->CallObjectMethod(communication_devices, _list_get, i);

		if (device_name == _get_device_name(device_info, p_env)) {
			ret = device_info;
			break;
		}

		p_env->DeleteLocalRef(device_info);
	}

	p_env->DeleteLocalRef(communication_devices);
	p_env->DeleteLocalRef(list_class);
	return ret;
}

int AudioDriverOboe::InputAudioStreamOboe::_get_prefered_mix_rate() const {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	return ad->audio_output.get_mix_rate();
}

int AudioDriverOboe::InputAudioStreamOboe::_get_default_mix_rate() const {
	return _get_prefered_mix_rate();
}

AudioDriverOboe::AudioStreamOboe::PostOpenResult AudioDriverOboe::InputAudioStreamOboe::_post_open_stream() {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (get_mix_rate() != _get_prefered_mix_rate()) {
		ERR_FAIL_COND_V_MSG(device_id == 0, PostOpenResult::FAIL, vformat("Oboe: Requested mix rate (%d) is not supported by the audio input device.", _get_prefered_mix_rate()));

		reset_device();
		return PostOpenResult::REOPEN;
	}

	ad->input_buffer_init(stream->getFramesPerBurst());
	return PostOpenResult::CONTINUE;
}

Error AudioDriverOboe::InputAudioStreamOboe::start_stream() {
	if (OS::get_singleton()->request_permission(PERMISSION_RECORD_AUDIO)) {
		return AudioStreamOboe::start_stream();
	}

	WARN_PRINT("Oboe: Unable to start audio capture - No RECORD_AUDIO permission");
	return ERR_UNAUTHORIZED;
}

bool AudioDriverOboe::InputAudioStreamOboe::ErrorCallback::onError(oboe::AudioStream *p_audio_stream, oboe::Result p_error) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();
	MutexLock lock(ad->mutex);

	// start_stream got the lock before.
	if (p_audio_stream != ad->audio_input.stream.get())
		return true;

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL_V(env, false);

	if (p_error == oboe::Result::ErrorDisconnected)
		ad->audio_input.reset_device();

	// Modify SCO only if device is reset.
	SCOResult res = ad->audio_input._close_stream(env, p_error == oboe::Result::ErrorDisconnected);
	if (res != SCOResult::WAIT)
		ad->audio_input.start_stream();
	return true;
}

oboe::DataCallbackResult AudioDriverOboe::InputAudioStreamOboe::DataCallback::onAudioReady(oboe::AudioStream *p_audio_stream, void *p_audio_data, int32_t p_frames) {
	AudioDriverOboe *ad = AudioDriverOboe::get_singleton();

	if (ad->pause || !ad->try_lock())
		return oboe::DataCallbackResult::Continue;

	ad->start_counting_ticks();

	for (int i = 0; i < p_frames; i++) {
		int32_t l, r;

		if (p_audio_stream->getChannelCount() == 1) {
			l = r = _read_sample(p_audio_stream->getFormat(), p_audio_data, i);
		} else {
			// TODO: Is it always the best way to read the first two samples?
			l = _read_sample(p_audio_stream->getFormat(), p_audio_data, i * p_audio_stream->getChannelCount());
			r = _read_sample(p_audio_stream->getFormat(), p_audio_data, i * p_audio_stream->getChannelCount() + 1);
		}

		ad->input_buffer_write(l);
		ad->input_buffer_write(r);
	}

	ad->stop_counting_ticks();
	ad->unlock();
	return oboe::DataCallbackResult::Continue;
}

AudioDriverOboe::InputAudioStreamOboe::InputAudioStreamOboe() :
		AudioStreamOboe(oboe::Direction::Input, GET_DEVICES_INPUTS, memnew(DataCallback), memnew(ErrorCallback)) {
}

void AudioDriverOboe::_write_sample(oboe::AudioFormat p_foramt, void *p_audio_data, int i, int32_t p_sample) {
	switch (p_foramt) {
		case oboe::AudioFormat::IEC61937:
		case oboe::AudioFormat::I16:
			((int16_t *)p_audio_data)[i] = p_sample >> 16;
			break;

		case oboe::AudioFormat::I24:
			((int8_t *)p_audio_data)[i * 3 + 2] = p_sample >> 24;
			((int8_t *)p_audio_data)[i * 3 + 1] = p_sample >> 16;
			((int8_t *)p_audio_data)[i * 3 + 0] = p_sample >> 8;
			break;

		case oboe::AudioFormat::I32:
			((int32_t *)p_audio_data)[i] = p_sample;
			break;

		case oboe::AudioFormat::Float:
			((float *)p_audio_data)[i] = p_sample / float(1ull << 31ull);
			break;

		default:
			ERR_PRINT("Oboe: Unknown format");
			break;
	}
}

int32_t AudioDriverOboe::_read_sample(oboe::AudioFormat p_foramt, void *p_audio_data, int i) {
	int32_t sample = 0;

	switch (p_foramt) {
		case oboe::AudioFormat::IEC61937:
		case oboe::AudioFormat::I16:
			sample = int32_t(((int16_t *)p_audio_data)[i]) << 16;
			break;

		case oboe::AudioFormat::I24:
			sample |= int32_t(((int8_t *)p_audio_data)[i * 3 + 2]) << 24;
			sample |= int32_t(((int8_t *)p_audio_data)[i * 3 + 1]) << 16;
			sample |= int32_t(((int8_t *)p_audio_data)[i * 3 + 0]) << 8;
			break;

		case oboe::AudioFormat::I32:
			sample = ((int32_t *)p_audio_data)[i];
			break;

		case oboe::AudioFormat::Float:
			sample = int32_t(((float *)p_audio_data)[i] * float(1ull << 31ull));
			break;

		default:
			ERR_PRINT("Oboe: Unknown format");
			break;
	}

	return sample;
}

Error AudioDriverOboe::init() {
	request_mix_rate = _get_configured_mix_rate();
	return OK;
}

void AudioDriverOboe::start() {
	Error error = audio_output.start_stream();
	ERR_FAIL_COND(error != OK);
}

PackedStringArray AudioDriverOboe::get_output_device_list() {
	return audio_output.get_device_list();
}

String AudioDriverOboe::get_output_device() {
	return audio_output.get_device();
}

void AudioDriverOboe::set_output_device(const String &p_name) {
	audio_output.set_device(p_name);
}

Error AudioDriverOboe::input_start() {
	return audio_input.start_stream();
}

Error AudioDriverOboe::input_stop() {
	audio_input.close_stream();
	return OK;
}

PackedStringArray AudioDriverOboe::get_input_device_list() {
	return audio_input.get_device_list();
}

String AudioDriverOboe::get_input_device() {
	return audio_input.get_device();
}

void AudioDriverOboe::set_input_device(const String &p_name) {
	audio_input.set_device(p_name);
}

int AudioDriverOboe::get_mix_rate() const {
	return audio_output.get_mix_rate();
}

AudioDriver::SpeakerMode AudioDriverOboe::get_speaker_mode() const {
	return get_speaker_mode_by_total_channels(channel_count);
}

float AudioDriverOboe::get_latency() {
	return audio_output.get_latency();
}

void AudioDriverOboe::lock() {
	mutex.lock();
}

void AudioDriverOboe::unlock() {
	mutex.unlock();
}

bool AudioDriverOboe::try_lock() {
	return mutex.try_lock();
}

void AudioDriverOboe::finish() {
	audio_output.close_stream();
	audio_input.close_stream();

	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);
	env->DeleteGlobalRef(manager);
}

void AudioDriverOboe::set_pause(bool p_pause) {
	pause = p_pause;

	if (p_pause) {
		audio_output.pause_stream();
		audio_input.pause_stream();
	} else {
		audio_output.start_stream();
		if (audio_input.is_playing())
			audio_input.start_stream();
	}
}

void AudioDriverOboe::_java_sco_callback(int p_sco_state) {
	MutexLock lock(mutex);

	AudioStreamOboe::set_sco_state(p_sco_state);

	audio_output.sco_state_updated();
	audio_input.sco_state_updated();
}

void AudioDriverOboe::request_permission_result(const String &p_permission, bool p_result) {
	if (p_permission == PERMISSION_RECORD_AUDIO && p_result) {
		input_start();
	} else if (p_permission == PERMISSION_MODIFY_AUDIO_SETTINGS && oboe::AudioStreamBuilder::isAAudioRecommended()) {
		MutexLock lock(mutex);

		audio_output.set_sco_permission(p_result);
		audio_input.set_sco_permission(p_result);
	}
}

void AudioDriverOboe::setup(jobject p_audio_manager) {
	JNIEnv *env = get_jni_env();
	ERR_FAIL_NULL(env);

	manager = env->NewGlobalRef(p_audio_manager);
	jclass manager_class = env->GetObjectClass(manager);

	jmethodID _manager_get_property = env->GetMethodID(manager_class, "getProperty", "(Ljava/lang/String;)Ljava/lang/String;");
	if (!env->ExceptionCheck()) {
		jstring property_output_sample_rate = env->NewStringUTF(PROPERTY_OUTPUT_SAMPLE_RATE);
		jstring property_output_frames_per_buffer = env->NewStringUTF(PROPERTY_OUTPUT_FRAMES_PER_BUFFER);

		jstring output_sample_rate = (jstring)env->CallObjectMethod(manager, _manager_get_property, property_output_sample_rate);
		jstring frames_per_buffer = (jstring)env->CallObjectMethod(manager, _manager_get_property, property_output_frames_per_buffer);

		oboe::DefaultStreamValues::SampleRate = jstring_to_string(output_sample_rate, env).to_int();
		oboe::DefaultStreamValues::FramesPerBurst = jstring_to_string(frames_per_buffer, env).to_int();
	} else {
		env->ExceptionClear();
	}

	if (!oboe::AudioStreamBuilder::isAAudioRecommended())
		return;

	_manager_get_devices = env->GetMethodID(manager_class, "getDevices", "(I)[Landroid/media/AudioDeviceInfo;");

	_manager_is_bluetooth_sco_available_off_call = env->GetMethodID(manager_class, "isBluetoothScoAvailableOffCall", "()Z");

	_manager_get_available_communication_devices = env->GetMethodID(manager_class, "getAvailableCommunicationDevices", "()Ljava/util/List;");
	if (!env->ExceptionCheck()) {
		_manager_get_communication_device = env->GetMethodID(manager_class, "getCommunicationDevice", "()Landroid/media/AudioDeviceInfo;");
		_manager_set_communication_device = env->GetMethodID(manager_class, "setCommunicationDevice", "(Landroid/media/AudioDeviceInfo;)Z");
		_manager_clear_communication_device = env->GetMethodID(manager_class, "clearCommunicationDevice", "()V");
	} else {
		env->ExceptionClear();
	}

	_manager_start_bluetooth_sco = env->GetMethodID(manager_class, "startBluetoothSco", "()V");
	if (!env->ExceptionCheck()) {
		_manager_stop_bluetooth_sco = env->GetMethodID(manager_class, "stopBluetoothSco", "()V");
	} else {
		env->ExceptionClear();
	}

	jclass device_info_class = env->FindClass("android/media/AudioDeviceInfo");

	_device_info_equals = env->GetMethodID(device_info_class, "equals", "(Ljava/lang/Object;)Z");
	_device_info_get_product_name = env->GetMethodID(device_info_class, "getProductName", "()Ljava/lang/CharSequence;");
	_device_info_get_id = env->GetMethodID(device_info_class, "getId", "()I");
	_device_info_get_type = env->GetMethodID(device_info_class, "getType", "()I");

	_device_info_get_address = env->GetMethodID(device_info_class, "getAddress", "()Ljava/lang/String;");
	if (env->ExceptionCheck())
		env->ExceptionClear();
}
