/**************************************************************************/
/*  audio_driver_hybrid.cpp                                              */
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

#include "audio_driver_hybrid.h"
#include "servers/movie_writer/independent_audio_recorder.h"
#include "core/os/os.h"

HybridAudioDriver::HybridAudioDriver() {
    recording_driver = memnew(AudioDriverDummy);
}

HybridAudioDriver::~HybridAudioDriver() {
    finish();
    if (recording_driver) {
        memdelete(recording_driver);
        recording_driver = nullptr;
    }
}

Error HybridAudioDriver::init(int p_mix_rate, AudioDriver::SpeakerMode p_speaker_mode) {
    mix_rate = p_mix_rate;
    speaker_mode = p_speaker_mode;
    
    // Calculate the actual number of audio channels (consistent with AudioServer)
    // AudioServer data format is interleaved stereo, so always 2 channels
    channels = 2;
    
    // Initialize recording driver
    recording_driver->set_mix_rate(mix_rate);
    recording_driver->set_speaker_mode(speaker_mode);
    recording_driver->set_use_threads(false); // Key: Do not use thread mode
    
    Error err = recording_driver->init();
    if (err != OK) {
        ERR_PRINT("HybridAudioDriver: Failed to initialize recording driver");
        return err;
    }
    
    // Initialize double buffer (buffer size based on sample rate and buffer duration)
    int buffer_frame_count = int(mix_rate * buffer_length_seconds);
    
    // RingBuffer constructor requires power of 2, not direct size
    // Calculate the smallest power that can accommodate buffer_frame_count
    int power = 0;
    while ((1 << power) < buffer_frame_count) {
        power++;
    }
    int actual_buffer_size = 1 << power;  // Actual buffer size
    
    original_buffer_size = actual_buffer_size;  // Save actual size
    write_buffer = RingBuffer<AudioFrame>(power);
    read_buffer = RingBuffer<AudioFrame>(power);
    buffer_initialized.set();
    swap_pending.clear();
    
    initialized = true;
    if (OS::get_singleton()->is_stdout_verbose()) {
        print_line(vformat("HybridAudioDriver initialized: %d Hz, %d channels, %.1f sec buffer (%d frames)", 
                   mix_rate, channels, buffer_length_seconds, actual_buffer_size));
    }
    
    return OK;
}

void HybridAudioDriver::start() {
    if (!initialized) {
        return;
    }
    
    // Start recording driver
    recording_driver->start();
}

void HybridAudioDriver::finish() {
    if (!initialized) {
        return;
    }
    
    recording_enabled = false;
    buffer_initialized.clear();
    
    // Stop recording driver
    if (recording_driver) {
        recording_driver->finish();
    }
    
    initialized = false;
}

void HybridAudioDriver::enable_recording(bool p_enable) {
    recording_enabled = p_enable;
    
    if (p_enable) {
        // Clear buffer, start new recording
        data_mutex.lock();
        if (buffer_initialized.is_set()) {
            // Calculate correct power
            int power = 0;
            while ((1 << power) < original_buffer_size) {
                power++;
            }
            write_buffer = RingBuffer<AudioFrame>(power);
            read_buffer = RingBuffer<AudioFrame>(power);
            swap_pending.clear();
        }
        data_mutex.unlock();
        if (OS::get_singleton()->is_stdout_verbose()) {
            print_line("HybridAudioDriver: Recording enabled");
        }
    } else {
        if (OS::get_singleton()->is_stdout_verbose()) {
            print_line("HybridAudioDriver: Recording disabled");
        }
    }
}

void HybridAudioDriver::capture_audio_data(const int32_t *p_buffer, int p_frames, int p_channels) {
    if (!recording_enabled || !initialized || !buffer_initialized.is_set()) {
        return;
    }
    
    // Record capture timestamp
    last_capture_time = OS::get_singleton()->get_ticks_usec();
    capture_count++;
    
    // p_channels in AudioServer is "channel pairs", actual audio channels are p_channels * 2
    // For stereo: p_channels=1, actual channels=2 (left, right)
    int actual_channels = p_channels * 2;
    
    // Convert int32_t audio data to AudioFrame and write to write_buffer
    for (int i = 0; i < p_frames; i++) {
        AudioFrame frame;
        
        if (actual_channels >= 2) {
            // Stereo or multi-channel - take the first two channels as AudioFrame
            frame.left = float(p_buffer[i * actual_channels]) / float(1 << 31);
            frame.right = float(p_buffer[i * actual_channels + 1]) / float(1 << 31);
        } else {
            // Single channel case
            float sample = float(p_buffer[i]) / float(1 << 31);
            frame.left = sample;
            frame.right = sample;
        }
        
        // If buffer is full, discard oldest data
        if (write_buffer.space_left() == 0) {
            AudioFrame dummy;
            write_buffer.read(&dummy, 1);
        }
        
        write_buffer.write(&frame, 1);
    }
    
    // Check if buffer swap should be triggered
    int current_data = write_buffer.data_left();
    int threshold = write_buffer.size() / 4;  // Trigger swap when 25% full
    
    if (current_data >= threshold && !swap_pending.is_set()) {
        swap_pending.set();
    }
    
    // Distribute audio data to all registered recorders
    {
        MutexLock lock(recorders_mutex);
        for (IndependentAudioRecorder* recorder : registered_recorders) {
            if (recorder && recorder->is_recording()) {
                recorder->on_audio_output(p_buffer, p_frames);
            }
        }
    }
}

int HybridAudioDriver::get_captured_audio_data(int32_t *p_output_buffer, int p_requested_frames) {
    if (!recording_enabled || !initialized || !buffer_initialized.is_set()) {
        // If no data, fill with silence (stereo format)
        int total_samples = p_requested_frames * 2;  // 2 channels: left + right
        for (int i = 0; i < total_samples; i++) {
            p_output_buffer[i] = 0;
        }
        return p_requested_frames;
    }
    
    // Check if buffer needs to be swapped (fast operation, minimal lock time)
    if (swap_pending.is_set()) {
        data_mutex.lock();
        
        // Swap buffers: write_buffer becomes read_buffer
        RingBuffer<AudioFrame> temp = read_buffer;
        read_buffer = write_buffer;
        write_buffer = temp;
        
        // Clear new write_buffer for continued writing
        int power = 0;
        while ((1 << power) < original_buffer_size) {
            power++;
        }
        write_buffer = RingBuffer<AudioFrame>(power);
        
        swap_pending.clear();
        data_mutex.unlock();
    }
    
    int available_frames = read_buffer.data_left();
    int frames_to_read = MIN(p_requested_frames, available_frames);
    
    if (frames_to_read > 0) {
        // Read available audio data
        Vector<AudioFrame> temp_buffer;
        temp_buffer.resize(frames_to_read);
        read_buffer.read(temp_buffer.ptrw(), frames_to_read);
        
        // Convert AudioFrame to int32_t format (interleaved stereo)
        for (int i = 0; i < frames_to_read; i++) {
            AudioFrame frame = temp_buffer[i];
            
            // Output interleaved stereo format: [left, right, left, right...]
            p_output_buffer[i * 2] = int32_t(CLAMP(frame.left, -1.0, 1.0) * float(1 << 31));
            p_output_buffer[i * 2 + 1] = int32_t(CLAMP(frame.right, -1.0, 1.0) * float(1 << 31));
        }
    }
    
    // If requested frames exceed available data, fill remaining with silence
    if (p_requested_frames > frames_to_read) {
        int remaining_samples = (p_requested_frames - frames_to_read) * 2;  // 2 channels: left + right
        int start_index = frames_to_read * 2;
        for (int i = 0; i < remaining_samples; i++) {
            p_output_buffer[start_index + i] = 0;
        }
    }
    
    return p_requested_frames;
}

int HybridAudioDriver::get_available_frames() const {
    if (!buffer_initialized.is_set()) {
        return 0;
    }
    return read_buffer.data_left();
}

bool HybridAudioDriver::has_audio_data() const {
    return buffer_initialized.is_set() && read_buffer.data_left() > 0;
}

void HybridAudioDriver::register_audio_recorder(IndependentAudioRecorder* recorder) {
    if (!recorder) {
        return;
    }
    
    MutexLock lock(recorders_mutex);
    
    // Check if already registered
    for (const IndependentAudioRecorder* existing : registered_recorders) {
        if (existing == recorder) {
            return; // Already registered
        }
    }
    
    registered_recorders.push_back(recorder);
    if (OS::get_singleton()->is_stdout_verbose()) {
        print_line(vformat("HybridAudioDriver: register audio recorder, current total: %d", registered_recorders.size()));
    }
}

void HybridAudioDriver::unregister_audio_recorder(IndependentAudioRecorder* recorder) {
    if (!recorder) {
        return;
    }
    
    // Try to acquire the mutex with a timeout to avoid blocking indefinitely
    bool mutex_acquired = false;
    const int max_attempts = 10;
    
    for (int attempt = 0; attempt < max_attempts; attempt++) {
        if (recorders_mutex.try_lock()) {
            mutex_acquired = true;
            break;
        } else {
            OS::get_singleton()->delay_usec(10000); // Wait 10ms
        }
    }
    
    if (!mutex_acquired) {
        // Mark the recorder as inactive immediately to prevent new audio data from being sent
        recorder->mark_inactive();
        return;
    }
    
    for (int i = 0; i < registered_recorders.size(); i++) {
        if (registered_recorders[i] == recorder) {
            registered_recorders.remove_at(i);
            if (OS::get_singleton()->is_stdout_verbose()) {
                print_line(vformat("HybridAudioDriver: unregister audio recorder completed, current total: %d", registered_recorders.size()));
            }
            recorders_mutex.unlock();
            return;
        }
    }
    
    recorders_mutex.unlock();
}

int HybridAudioDriver::get_registered_recorder_count() const {
    MutexLock lock(recorders_mutex);
    return registered_recorders.size();
} 

