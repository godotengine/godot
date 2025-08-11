/**************************************************************************/
/*  post_merge_processor.h                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#ifndef POST_MERGE_PROCESSOR_H
#define POST_MERGE_PROCESSOR_H

#include "core/string/ustring.h"
#include "core/error/error_macros.h"
#include "core/io/file_access.h"

/**
 * Post-processing video/audio file merger
 * Implements simplified approach: independent audio/video sampling + post-merge
 * Similar to: ffmpeg -i video.avi -i audio.avi -c copy output.avi -y
 */
class PostMergeProcessor {
public:
    // Merge methods supported
    enum MergeMethod {
        METHOD_FFMPEG_SYSTEM,   // Use system FFmpeg command
        METHOD_CUSTOM_AVI,      // Custom AVI container merge (Phase 2)
        METHOD_NONE             // No merge, keep separate files
    };
    
    // Merge configuration
    struct MergeConfig {
        MergeMethod method = METHOD_FFMPEG_SYSTEM;
        bool keep_intermediate_files = false;    // Keep original video/audio files after merge
        bool enable_debug_output = false;        // Enable debug logging
        String ffmpeg_path = "ffmpeg";          // FFmpeg executable path
        
        // Interleaving strategy
        enum InterleaveStrategy {
            INTERLEAVE_SIMPLE_ALTERNATE,    // Simple alternating (legacy)
            INTERLEAVE_TIMESTAMP_BASED,     // Sort by timestamp
            INTERLEAVE_BUFFERED_TIMESTAMP   // Timestamp with I/O buffering
        };
        InterleaveStrategy interleave_strategy = INTERLEAVE_TIMESTAMP_BASED;
        uint32_t buffer_duration_ms = 500;      // Buffer window for buffered strategy
        bool validate_sync = true;              // Enable A/V sync validation
        uint64_t max_av_drift_us = 40000;     // Max allowed A/V drift (40ms)
        
        MergeConfig() {}
    };
    
    // Merge result information
    struct MergeResult {
        Error error_code = OK;
        String output_file_path;
        String error_message;
        float merge_duration_seconds = 0.0f;
        bool intermediate_files_cleaned = false;
        
        MergeResult() {}
    };

private:
    MergeConfig config;
    
    // FFmpeg system command implementation
    Error ffmpeg_system_merge(const String &video_path, const String &audio_path, const String &output_path, MergeResult &result);
    
    // Custom AVI merge implementation (Phase 2)
    Error custom_avi_merge(const String &video_path, const String &audio_path, const String &output_path, MergeResult &result);
    
    // Timestamped chunk for interleaving
    struct TimestampedChunk {
        enum ChunkType {
            VIDEO_CHUNK,
            AUDIO_CHUNK
        };
        
        ChunkType type;
        uint64_t timestamp_us;      // Timestamp in microseconds
        uint64_t file_offset;       // Offset in source file
        uint32_t chunk_size;        // Size of chunk data
        uint32_t index;             // Original index for debugging
        
        // For sorting by timestamp
        bool operator<(const TimestampedChunk &other) const {
            return timestamp_us < other.timestamp_us;
        }
    };
    
    // AVI file parsing structures
    struct AviFileInfo {
        // Main AVI header
        uint32_t microsec_per_frame;
        uint32_t max_bytes_per_sec;
        uint32_t total_frames;
        uint32_t streams;
        uint32_t width;
        uint32_t height;
        
        // Stream information
        struct StreamInfo {
            char fourcc_type[4];    // 'vids' or 'auds'
            char fourcc_handler[4]; // 'MJPG' or 'PCM '
            uint32_t scale;
            uint32_t rate;          // fps = rate/scale
            uint32_t length;        // frames/samples count
            uint32_t sample_size;
        };
        
        Vector<StreamInfo> streams_info;
        
        // File offsets
        uint64_t movi_list_offset;
        uint64_t movi_data_offset;
        uint64_t movi_data_size;
        
        // Index information
        struct IndexEntry {
            char fourcc[4];
            uint32_t flags;
            uint32_t chunk_offset;  // Relative to movi data start
            uint32_t chunk_size;
        };
        Vector<IndexEntry> index_entries;
        
        AviFileInfo() {
            microsec_per_frame = 0;
            max_bytes_per_sec = 0;
            total_frames = 0;
            streams = 0;
            width = 0;
            height = 0;
            movi_list_offset = 0;
            movi_data_offset = 0;
            movi_data_size = 0;
        }
    };
    
    // AVI parsing and merging helper methods
    Error parse_avi_file(const String &file_path, AviFileInfo &avi_info);
    Error parse_hdrl_chunk(Ref<FileAccess> file, AviFileInfo &avi_info, uint32_t chunk_size);
    Error parse_stream_header(Ref<FileAccess> file, AviFileInfo::StreamInfo &stream_info, uint32_t chunk_size);
    Error parse_idx1_chunk(Ref<FileAccess> file, AviFileInfo &avi_info, uint32_t chunk_size);
    Error write_merged_avi_header(Ref<FileAccess> output_file, const AviFileInfo &video_info, const AviFileInfo &audio_info);
    Error write_video_stream_header(Ref<FileAccess> output_file, const AviFileInfo &video_info);
    Error write_audio_stream_header(Ref<FileAccess> output_file, const AviFileInfo &audio_info);
    Error interleave_avi_data(const String &video_path, const String &audio_path, Ref<FileAccess> output_file, const AviFileInfo &video_info, const AviFileInfo &audio_info);
    Error interleave_avi_data_timestamped(const String &video_path, const String &audio_path, Ref<FileAccess> output_file, const AviFileInfo &video_info, const AviFileInfo &audio_info);
    Error write_merged_avi_index(Ref<FileAccess> output_file, const Vector<AviFileInfo::IndexEntry> &merged_index);
    
    // Helper methods for timestamp-based interleaving
    void calculate_chunk_timestamps(const Vector<AviFileInfo::IndexEntry> &chunks, const AviFileInfo &info, bool is_video, Vector<TimestampedChunk> &timestamped_chunks);
    uint32_t calculate_optimal_audio_chunk_size(uint32_t video_fps, uint32_t sample_rate);
    bool validate_av_sync(uint64_t video_ts, uint64_t audio_ts);
    Error write_timestamped_chunk(Ref<FileAccess> input_file, Ref<FileAccess> output_file, const TimestampedChunk &chunk, bool is_video, uint32_t &chunk_offset, Vector<AviFileInfo::IndexEntry> &merged_index);
    
    // Utility methods
    bool check_ffmpeg_availability();
    bool file_exists(const String &path);
    Error cleanup_intermediate_files(const String &video_path, const String &audio_path);
    uint64_t get_file_size(const String &path);
    void scan_movi_chunks(Ref<FileAccess> file, uint64_t movi_offset, uint32_t movi_size, Vector<AviFileInfo::IndexEntry> &chunks);

public:
    PostMergeProcessor();
    ~PostMergeProcessor();
    
    /**
     * Set merge configuration
     */
    void set_config(const MergeConfig &p_config);
    const MergeConfig &get_config() const { return config; }
    
    /**
     * Main merge interface
     * Merges separate video and audio files into a single output file
     * 
     * @param video_path Path to video file (e.g., "movie_video.avi")
     * @param audio_path Path to audio file (e.g., "movie_audio.avi") 
     * @param output_path Path for merged output file (e.g., "movie_merged.avi")
     * @return MergeResult with operation result and details
     */
    MergeResult merge_files(const String &video_path, const String &audio_path, const String &output_path);
    
    /**
     * Check if merge method is available on current platform
     */
    bool is_method_available(MergeMethod method);
    
    /**
     * Get recommended merge method for current platform
     */
    MergeMethod get_recommended_method();
    
    /**
     * Get human-readable method name
     */
    String get_method_name(MergeMethod method);
    
    /**
     * Validate file paths and configuration before merge
     */
    Error validate_merge_request(const String &video_path, const String &audio_path, const String &output_path);
};

#endif // POST_MERGE_PROCESSOR_H