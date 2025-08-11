/**************************************************************************/
/*  post_merge_processor.cpp                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "post_merge_processor.h"
#include "core/os/os.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "core/os/time.h"
#include "movie_utils.h"

PostMergeProcessor::PostMergeProcessor() {
    // Initialize with default configuration
}

PostMergeProcessor::~PostMergeProcessor() {
}

void PostMergeProcessor::set_config(const MergeConfig &p_config) {
    config = p_config;
}

PostMergeProcessor::MergeResult PostMergeProcessor::merge_files(const String &video_path, const String &audio_path, const String &output_path) {
    MergeResult result;
    uint64_t start_time = OS::get_singleton()->get_ticks_usec();
    
    if (config.enable_debug_output && MovieDebugUtils::is_stdout_verbose()) {
        print_line("PostMergeProcessor: Starting merge operation");
        print_line("  Video file: " + video_path);
        print_line("  Audio file: " + audio_path);
        print_line("  Output file: " + output_path);
        print_line("  Method: " + get_method_name(config.method));
    }
    
    // Validate merge request
    Error validation_error = validate_merge_request(video_path, audio_path, output_path);
    if (validation_error != OK) {
        result.error_code = validation_error;
        result.error_message = "Validation failed";
        return result;
    }
    
    // Execute merge based on selected method
    switch (config.method) {
        case METHOD_FFMPEG_SYSTEM:
            result.error_code = ffmpeg_system_merge(video_path, audio_path, output_path, result);
            break;
            
        case METHOD_CUSTOM_AVI:
            result.error_code = custom_avi_merge(video_path, audio_path, output_path, result);
            break;
            
        case METHOD_NONE:
            result.error_code = OK;
            result.output_file_path = video_path; // Return video file as primary output
            break;
            
        default:
            result.error_code = ERR_PARAMETER_RANGE_ERROR;
            result.error_message = "Unknown merge method";
            break;
    }
    
    // Calculate merge duration
    uint64_t end_time = OS::get_singleton()->get_ticks_usec();
    result.merge_duration_seconds = (end_time - start_time) / 1000000.0f;
    
    // Clean up intermediate files if requested and merge was successful
    if (result.error_code == OK && config.keep_intermediate_files == false && config.method != METHOD_NONE) {
        Error cleanup_error = cleanup_intermediate_files(video_path, audio_path);
        result.intermediate_files_cleaned = (cleanup_error == OK);
        
        if (config.enable_debug_output && MovieDebugUtils::is_stdout_verbose()) {
            if (result.intermediate_files_cleaned) {
                print_line("PostMergeProcessor: Intermediate files cleaned up successfully");
            } else {
                print_line("PostMergeProcessor: Warning - Failed to clean up intermediate files");
            }
        }
    }
    
    // Store output file path
    if (result.error_code == OK && config.method != METHOD_NONE) {
        result.output_file_path = output_path;
    }
    
    if (config.enable_debug_output) {
        if (MovieDebugUtils::is_stdout_verbose()) {
            print_line(String("PostMergeProcessor: Merge completed in ") + String::num(result.merge_duration_seconds, 2) + " seconds");
        }
        if (result.error_code != OK) {
            print_line("PostMergeProcessor: Merge failed - " + result.error_message);
        }
    }
    
    return result;
}

Error PostMergeProcessor::ffmpeg_system_merge(const String &video_path, const String &audio_path, const String &output_path, MergeResult &result) {
#ifdef WEB_ENABLED
    // Web platform doesn't support system command execution
    result.error_message = "FFmpeg system merge not supported on Web platform";
    return ERR_UNAVAILABLE;
#else
    if (!check_ffmpeg_availability()) {
        result.error_message = "FFmpeg not found in system PATH";
        return ERR_FILE_NOT_FOUND;
    }
    
    // Build FFmpeg command arguments
    List<String> args;
    args.push_back("-i");
    args.push_back(video_path);
    args.push_back("-i");
    args.push_back(audio_path);
    args.push_back("-c");
    args.push_back("copy");  // Stream copy, no re-encoding
    args.push_back(output_path);
    args.push_back("-y");    // Overwrite output file if exists
    
    
    // Execute FFmpeg command
    String stdout_output;
    int exit_code = OS::get_singleton()->execute(config.ffmpeg_path, args, &stdout_output);
    
    
    if (exit_code != 0) {
        result.error_message = String("FFmpeg failed with exit code ") + String::num_int64(exit_code);
        if (!stdout_output.is_empty()) {
            result.error_message += " - " + stdout_output;
        }
        return ERR_FILE_CANT_WRITE;
    }
    
    // Verify output file was created
    if (!file_exists(output_path)) {
        result.error_message = "Output file was not created by FFmpeg";
        return ERR_FILE_CANT_WRITE;
    }
    
    return OK;
#endif
}

Error PostMergeProcessor::custom_avi_merge(const String &video_path, const String &audio_path, const String &output_path, MergeResult &result) {
    
    uint64_t merge_start_time = OS::get_singleton()->get_ticks_usec();
    
    // Parse input AVI files
    AviFileInfo video_info, audio_info;
    
    Error video_parse_error = parse_avi_file(video_path, video_info);
    if (video_parse_error != OK) {
        result.error_message = "Failed to parse video AVI file: " + video_path;
        return video_parse_error;
    }
    
    Error audio_parse_error = parse_avi_file(audio_path, audio_info);
    if (audio_parse_error != OK) {
        result.error_message = "Failed to parse audio AVI file: " + audio_path;
        return audio_parse_error;
    }
    
    // Create output file
    Ref<FileAccess> output_file = FileAccess::open(output_path, FileAccess::WRITE);
    if (output_file.is_null()) {
        result.error_message = "Cannot create output file: " + output_path;
        return ERR_FILE_CANT_WRITE;
    }
    
    // Write merged AVI header
    Error header_error = write_merged_avi_header(output_file, video_info, audio_info);
    if (header_error != OK) {
        result.error_message = "Failed to write merged AVI header";
        output_file->close();
        return header_error;
    }
    
    // Interleave video and audio data
    
    Error interleave_error;
    if (config.interleave_strategy == MergeConfig::INTERLEAVE_SIMPLE_ALTERNATE) {
        interleave_error = interleave_avi_data(video_path, audio_path, output_file, video_info, audio_info);
    } else {
        interleave_error = interleave_avi_data_timestamped(video_path, audio_path, output_file, video_info, audio_info);
    }
    
    if (interleave_error != OK) {
        result.error_message = "Failed to interleave video and audio data";
        output_file->close();
        return interleave_error;
    }
    
    // Update file size in RIFF header
    uint64_t final_size = output_file->get_position();
    output_file->seek(4);
    output_file->store_32((uint32_t)(final_size - 8));
    
    output_file->close();
    
    // Calculate merge duration
    uint64_t merge_end_time = OS::get_singleton()->get_ticks_usec();
    result.merge_duration_seconds = (merge_end_time - merge_start_time) / 1000000.0f;
    result.output_file_path = output_path;
    
    if (config.enable_debug_output && MovieDebugUtils::is_stdout_verbose()) {
        print_line("PostMergeProcessor: Custom AVI merge completed successfully");
        print_line("  Output size: " + String::num_int64(get_file_size(output_path)) + " bytes");
        print_line("  Merge time: " + String::num_real(result.merge_duration_seconds) + " seconds");
    }
    
    return OK;
}

bool PostMergeProcessor::check_ffmpeg_availability() {
#ifdef WEB_ENABLED
    return false;
#else
    // Try to execute ffmpeg -version to check availability
    List<String> args;
    args.push_back("-version");
    
    String output;
    int exit_code = OS::get_singleton()->execute(config.ffmpeg_path, args, &output);
    
    return (exit_code == 0 && output.contains("ffmpeg"));
#endif
}

bool PostMergeProcessor::file_exists(const String &path) {
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    return file.is_valid();
}

Error PostMergeProcessor::cleanup_intermediate_files(const String &video_path, const String &audio_path) {
    Error video_error = OK;
    Error audio_error = OK;
    
    if (file_exists(video_path)) {
        String video_dir = video_path.get_base_dir();
        Ref<DirAccess> dir = DirAccess::open(video_dir);
        if (dir.is_valid()) {
            video_error = dir->remove(video_path.get_file());
        } else {
            video_error = ERR_FILE_CANT_OPEN;
        }
    }
    
    if (file_exists(audio_path)) {
        String audio_dir = audio_path.get_base_dir();
        Ref<DirAccess> dir = DirAccess::open(audio_dir);
        if (dir.is_valid()) {
            audio_error = dir->remove(audio_path.get_file());
        } else {
            audio_error = ERR_FILE_CANT_OPEN;
        }
    }
    
    // Return error if either cleanup failed
    if (video_error != OK) return video_error;
    if (audio_error != OK) return audio_error;
    
    return OK;
}

uint64_t PostMergeProcessor::get_file_size(const String &path) {
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    if (file.is_valid()) {
        return file->get_length();
    }
    return 0;
}

// AVI file parsing implementation
Error PostMergeProcessor::parse_avi_file(const String &file_path, AviFileInfo &avi_info) {
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
    if (file.is_null()) {
        return ERR_FILE_CANT_OPEN;
    }
    
    
    // Read RIFF header
    char riff_header[4];
    file->get_buffer((uint8_t*)riff_header, 4);
    if (memcmp(riff_header, "RIFF", 4) != 0) {
        return ERR_FILE_CORRUPT;
    }
    
    file->get_32(); // Skip file size
    
    char avi_header[4];
    file->get_buffer((uint8_t*)avi_header, 4);
    if (memcmp(avi_header, "AVI ", 4) != 0) {
        return ERR_FILE_CORRUPT;
    }
    
    // Parse AVI structure based on SimpleAudioWriter/SimpleVideoWriter format
    bool found_hdrl = false;
    bool found_movi = false;
    
    uint64_t pos = 12; // Start after RIFF+AVI header
    
    // Look for LIST hdrl chunk first
    if (pos < file->get_length() - 8) {
        file->seek(pos);
        char chunk_fourcc[4];
        file->get_buffer((uint8_t*)chunk_fourcc, 4);
        uint32_t chunk_size = file->get_32();
        
        if (memcmp(chunk_fourcc, "LIST", 4) == 0) {
            char list_type[4];
            file->get_buffer((uint8_t*)list_type, 4);
            
            if (memcmp(list_type, "hdrl", 4) == 0) {
                Error hdrl_error = parse_hdrl_chunk(file, avi_info, chunk_size - 4);
                if (hdrl_error == OK) {
                    found_hdrl = true;
                }
                // SimpleVideoWriter includes odml LIST inside hdrl, so don't skip yet
                // The hdrl parser will handle the internal structure
            } else {
                // Not hdrl, skip this LIST
                pos += 8 + chunk_size;
                if (chunk_size % 2 != 0) pos++;
            }
        } else {
            // Not a LIST chunk
            pos += 8 + chunk_size;
            if (chunk_size % 2 != 0) pos++;
        }
    }
    
    // Now look for movi LIST - it should be right after hdrl
    // Update position based on actual file position after hdrl parsing
    pos = file->get_position();
    
    if (pos < file->get_length() - 8) {
        file->seek(pos);
        char chunk_fourcc[4];
        file->get_buffer((uint8_t*)chunk_fourcc, 4);
        uint32_t chunk_size = file->get_32();
        
        if (memcmp(chunk_fourcc, "LIST", 4) == 0) {
            char list_type[4];
            file->get_buffer((uint8_t*)list_type, 4);
            
            if (memcmp(list_type, "movi", 4) == 0) {
                avi_info.movi_list_offset = pos;
                avi_info.movi_data_offset = pos + 12; // LIST+size+movi = 12
                avi_info.movi_data_size = chunk_size - 4;
                found_movi = true;
                pos += 8 + chunk_size; // Move past movi LIST
                if (chunk_size % 2 != 0) pos++; // Align to even boundary
            }
        }
    }
    
    // Look for idx1 chunk
    if (found_movi && pos < file->get_length() - 8) {
        file->seek(pos);
        char chunk_fourcc[4];
        file->get_buffer((uint8_t*)chunk_fourcc, 4);
        uint32_t chunk_size = file->get_32();
        
        if (memcmp(chunk_fourcc, "idx1", 4) == 0) {
            Error idx_error = parse_idx1_chunk(file, avi_info, chunk_size);
            if (idx_error != OK) {
            }
        }
    }
    
    if (!found_hdrl || !found_movi) {
        return ERR_FILE_CORRUPT;
    }
    
    
    return OK;
}

Error PostMergeProcessor::parse_hdrl_chunk(Ref<FileAccess> file, AviFileInfo &avi_info, uint32_t chunk_size) {
    uint64_t hdrl_start = file->get_position();
    uint64_t chunk_end = hdrl_start + chunk_size;
    
    
    // First, look for avih chunk
    char chunk_fourcc[4];
    file->get_buffer((uint8_t*)chunk_fourcc, 4);
    uint32_t sub_chunk_size = file->get_32();
    
    if (memcmp(chunk_fourcc, "avih", 4) == 0) {
        // Parse main AVI header
        avi_info.microsec_per_frame = file->get_32();
        avi_info.max_bytes_per_sec = file->get_32();
        file->get_32(); // padding_granularity
        file->get_32(); // flags
        avi_info.total_frames = file->get_32();
        file->get_32(); // initial_frames
        avi_info.streams = file->get_32();
        file->get_32(); // suggested_buffer_size
        avi_info.width = file->get_32();
        avi_info.height = file->get_32();
        // Skip reserved fields (4 * 4 = 16 bytes)
        file->seek(file->get_position() + 16);
    } else {
        return ERR_FILE_CORRUPT;
    }
    
    // Look for LIST strl chunk(s)
    while (file->get_position() < chunk_end) {
        // Check if we have enough bytes for another chunk
        if (file->get_position() + 8 > chunk_end) {
            break;
        }
        
        uint64_t chunk_start = file->get_position();
        file->get_buffer((uint8_t*)chunk_fourcc, 4);
        sub_chunk_size = file->get_32();
        
        if (memcmp(chunk_fourcc, "LIST", 4) == 0) {
            char list_type[4];
            file->get_buffer((uint8_t*)list_type, 4);
            
            if (memcmp(list_type, "strl", 4) == 0) {
                // Parse stream header
                AviFileInfo::StreamInfo stream_info;
                Error stream_error = parse_stream_header(file, stream_info, sub_chunk_size - 4);
                if (stream_error == OK) {
                    avi_info.streams_info.push_back(stream_info);
                }
            } else if (memcmp(list_type, "odml", 4) == 0) {
                // Skip odml content
                file->seek(chunk_start + 8 + sub_chunk_size);
            } else {
                // Skip unknown LIST type
                file->seek(chunk_start + 8 + sub_chunk_size);
            }
        } else {
            // Skip unknown chunk
            file->seek(chunk_start + 8 + sub_chunk_size);
        }
        
        // Align to even boundary
        if (file->get_position() % 2 != 0) {
            file->seek(file->get_position() + 1);
        }
    }
    
    return OK;
}

Error PostMergeProcessor::parse_stream_header(Ref<FileAccess> file, AviFileInfo::StreamInfo &stream_info, uint32_t chunk_size) {
    uint64_t strl_start = file->get_position();
    uint64_t chunk_end = strl_start + chunk_size;
    
    
    // Look for strh chunk first
    char chunk_fourcc[4];
    file->get_buffer((uint8_t*)chunk_fourcc, 4);
    uint32_t sub_chunk_size = file->get_32();
    
    if (memcmp(chunk_fourcc, "strh", 4) == 0) {
        // Parse stream header
        file->get_buffer((uint8_t*)stream_info.fourcc_type, 4);
        file->get_buffer((uint8_t*)stream_info.fourcc_handler, 4);
        file->get_32(); // flags
        file->get_32(); // priority + language
        file->get_32(); // initial_frames
        stream_info.scale = file->get_32();
        stream_info.rate = file->get_32();
        file->get_32(); // start
        stream_info.length = file->get_32();
        file->get_32(); // suggested_buffer_size
        file->get_32(); // quality
        stream_info.sample_size = file->get_32();
        // Skip remaining strh fields
        file->seek(file->get_position() + sub_chunk_size - 48);
    } else {
        if (config.enable_debug_output) {
            print_line("PostMergeProcessor: Expected strh chunk but found something else");
        }
        return ERR_FILE_CORRUPT;
    }
    
    // Look for strf chunk
    if (file->get_position() < chunk_end - 8) {
        file->get_buffer((uint8_t*)chunk_fourcc, 4);
        sub_chunk_size = file->get_32();
        
        if (memcmp(chunk_fourcc, "strf", 4) == 0) {
            if (config.enable_debug_output) {
                print_line("PostMergeProcessor: Found strf chunk, size=" + String::num_int64(sub_chunk_size));
            }
            // Skip strf data - we don't need it for merging
            file->seek(file->get_position() + sub_chunk_size);
        }
    }
    
    return OK;
}

Error PostMergeProcessor::parse_idx1_chunk(Ref<FileAccess> file, AviFileInfo &avi_info, uint32_t chunk_size) {
    uint32_t entry_count = chunk_size / 16; // Each index entry is 16 bytes
    
    if (config.enable_debug_output) {
        print_line("PostMergeProcessor: Parsing " + String::num_int64(entry_count) + " index entries");
    }
    
    for (uint32_t i = 0; i < entry_count; i++) {
        AviFileInfo::IndexEntry entry;
        file->get_buffer((uint8_t*)entry.fourcc, 4);
        entry.flags = file->get_32();
        entry.chunk_offset = file->get_32();
        entry.chunk_size = file->get_32();
        
        avi_info.index_entries.push_back(entry);
    }
    
    return OK;
}

bool PostMergeProcessor::is_method_available(MergeMethod method) {
    switch (method) {
        case METHOD_FFMPEG_SYSTEM:
#ifdef WEB_ENABLED
            return false;
#else
            return check_ffmpeg_availability();
#endif
            
        case METHOD_CUSTOM_AVI:
            // Always available (Phase 2)
            return true; // Custom AVI merge implementation
            
        case METHOD_NONE:
            return true;
            
        default:
            return false;
    }
}

PostMergeProcessor::MergeMethod PostMergeProcessor::get_recommended_method() {
#ifdef WEB_ENABLED
    return METHOD_NONE; // Web platform keeps separate files
#else
    // For testing Phase 2: prioritize custom AVI merge over FFmpeg
    if (is_method_available(METHOD_CUSTOM_AVI)) {
        return METHOD_CUSTOM_AVI;
    } else if (is_method_available(METHOD_FFMPEG_SYSTEM)) {
        return METHOD_FFMPEG_SYSTEM;
    } else {
        return METHOD_NONE;
    }
#endif
}

String PostMergeProcessor::get_method_name(MergeMethod method) {
    switch (method) {
        case METHOD_FFMPEG_SYSTEM:
            return "FFmpeg System Command";
        case METHOD_CUSTOM_AVI:
            return "Custom AVI Merge";
        case METHOD_NONE:
            return "No Merge";
        default:
            return "Unknown";
    }
}

Error PostMergeProcessor::validate_merge_request(const String &video_path, const String &audio_path, const String &output_path) {
    // Check if method is available
    if (!is_method_available(config.method)) {
        return ERR_UNAVAILABLE;
    }
    
    // For non-merge method, skip file validation
    if (config.method == METHOD_NONE) {
        return OK;
    }
    
    // Check input files exist
    if (!file_exists(video_path)) {
        return ERR_FILE_NOT_FOUND;
    }
    
    if (!file_exists(audio_path)) {
        return ERR_FILE_NOT_FOUND;
    }
    
    // Check input files are not empty
    if (get_file_size(video_path) == 0) {
        return ERR_FILE_CORRUPT;
    }
    
    if (get_file_size(audio_path) == 0) {
        return ERR_FILE_CORRUPT;
    }
    
    // Check output path is valid (directory exists)
    String output_dir = output_path.get_base_dir();
    if (!output_dir.is_empty()) {
        Ref<DirAccess> dir = DirAccess::open(output_dir);
        if (!dir.is_valid()) {
            return ERR_FILE_BAD_PATH;
        }
    }
    
    return OK;
}

// AVI header merging implementation
Error PostMergeProcessor::write_merged_avi_header(Ref<FileAccess> output_file, const AviFileInfo &video_info, const AviFileInfo &audio_info) {
    
    // Write RIFF header
    output_file->store_buffer((const uint8_t*)"RIFF", 4);
    output_file->store_32(0); // Placeholder for file size
    output_file->store_buffer((const uint8_t*)"AVI ", 4);
    
    // Write header list
    output_file->store_buffer((const uint8_t*)"LIST", 4);
    uint64_t hdrl_size_pos = output_file->get_position();
    output_file->store_32(0); // Placeholder for hdrl size
    output_file->store_buffer((const uint8_t*)"hdrl", 4);
    uint64_t hdrl_start = output_file->get_position();
    
    // Write main AVI header (avih)
    output_file->store_buffer((const uint8_t*)"avih", 4);
    output_file->store_32(56); // avih chunk size
    
    // Use video timing as primary
    output_file->store_32(video_info.microsec_per_frame); // μs per frame
    output_file->store_32(video_info.max_bytes_per_sec + 44100 * 4 * 2); // Est. bytes/sec
    output_file->store_32(0); // padding_granularity
    output_file->store_32(0x10); // flags (HASINDEX)
    output_file->store_32(video_info.total_frames); // total frames
    output_file->store_32(0); // initial_frames
    output_file->store_32(2); // total streams (video + audio)
    output_file->store_32(0); // suggested_buffer_size
    output_file->store_32(video_info.width);
    output_file->store_32(video_info.height);
    // Reserved fields
    for (int i = 0; i < 4; i++) {
        output_file->store_32(0);
    }
    
    // Write video stream header
    write_video_stream_header(output_file, video_info);
    
    // Write audio stream header - convert 32-bit to 16-bit format
    write_audio_stream_header(output_file, audio_info);
    
    // Update hdrl size
    uint64_t hdrl_end = output_file->get_position();
    uint32_t hdrl_size = hdrl_end - hdrl_start;
    output_file->seek(hdrl_size_pos);
    output_file->store_32(hdrl_size + 4); // +4 for 'hdrl' fourcc
    output_file->seek(hdrl_end);
    
    
    return OK;
}

Error PostMergeProcessor::write_video_stream_header(Ref<FileAccess> output_file, const AviFileInfo &video_info) {
    // VIDEO STREAM (Stream 0)
    output_file->store_buffer((const uint8_t*)"LIST", 4);
    output_file->store_32(132); // strl size: 4 + 4 + 4 + 56 + 4 + 4 + 40 + 4 + 4 + 16
    output_file->store_buffer((const uint8_t*)"strl", 4);
    
    // Video stream header (strh)
    output_file->store_buffer((const uint8_t*)"strh", 4);
    output_file->store_32(56); // strh size
    
    output_file->store_buffer((const uint8_t*)"vids", 4); // Stream type
    output_file->store_buffer((const uint8_t*)"MJPG", 4); // Handler
    output_file->store_32(0); // flags
    output_file->store_16(0); // priority
    output_file->store_16(0); // language
    output_file->store_32(0); // initial_frames
    output_file->store_32(1); // scale
    output_file->store_32(1000000 / video_info.microsec_per_frame); // rate (FPS)
    output_file->store_32(0); // start
    output_file->store_32(video_info.total_frames); // length
    output_file->store_32(0); // suggested_buffer_size
    output_file->store_32(0); // quality
    output_file->store_32(0); // sample_size (variable for video)
    output_file->store_16(0); // left
    output_file->store_16(0); // top
    output_file->store_16(video_info.width); // right
    output_file->store_16(video_info.height); // bottom
    
    // Video stream format (strf) - BITMAPINFOHEADER
    output_file->store_buffer((const uint8_t*)"strf", 4);
    output_file->store_32(40); // Size
    
    output_file->store_32(40); // biSize
    output_file->store_32(video_info.width); // biWidth
    output_file->store_32(video_info.height); // biHeight
    output_file->store_16(1); // biPlanes
    output_file->store_16(24); // biBitCount
    output_file->store_buffer((const uint8_t*)"MJPG", 4); // biCompression
    output_file->store_32(((video_info.width * 24 / 8 + 3) & 0xFFFFFFFC) * video_info.height); // biSizeImage
    output_file->store_32(0); // biXPelsPerMeter
    output_file->store_32(0); // biYPelsPerMeter
    output_file->store_32(0); // biClrUsed
    output_file->store_32(0); // biClrImportant
    
    // OpenDML header for large file support
    output_file->store_buffer((const uint8_t*)"LIST", 4);
    output_file->store_32(16); // Size
    output_file->store_buffer((const uint8_t*)"odml", 4);
    output_file->store_buffer((const uint8_t*)"dmlh", 4);
    output_file->store_32(4); // Size
    output_file->store_32(video_info.total_frames); // Total frames
    
    return OK;
}

Error PostMergeProcessor::write_audio_stream_header(Ref<FileAccess> output_file, const AviFileInfo &audio_info) {
    // Get audio sample rate and channels from stream info
    uint32_t sample_rate = 44100; // Default
    uint32_t channels = 2; // Default stereo
    
    if (audio_info.streams_info.size() > 0) {
        const AviFileInfo::StreamInfo &audio_stream = audio_info.streams_info[0];
        if (audio_stream.rate > 0 && audio_stream.scale > 0) {
            // For SimpleAudioWriter: scale = block_align, rate = sample_rate * block_align
            sample_rate = audio_stream.rate / audio_stream.scale;
            channels = audio_stream.sample_size / 4; // 32-bit per channel
        }
    }
    
    // AUDIO STREAM (Stream 1) - Convert to 16-bit
    output_file->store_buffer((const uint8_t*)"LIST", 4);
    output_file->store_32(84); // strl size: 4 + 4 + 4 + 56 + 4 + 4 + 16
    output_file->store_buffer((const uint8_t*)"strl", 4);
    
    // Audio stream header (strh)
    output_file->store_buffer((const uint8_t*)"strh", 4);
    output_file->store_32(56); // strh size
    
    output_file->store_buffer((const uint8_t*)"auds", 4); // Stream type
    output_file->store_32(0); // Handler (none for PCM)
    output_file->store_32(0); // flags
    output_file->store_16(0); // priority
    output_file->store_16(0); // language
    output_file->store_32(0); // initial_frames
    output_file->store_32(channels * 2); // scale (16-bit per channel)
    output_file->store_32(sample_rate * channels * 2); // rate (sample_rate * block_align)
    output_file->store_32(0); // start
    output_file->store_32(audio_info.streams_info.size() > 0 ? audio_info.streams_info[0].length : 0); // length (in samples)
    output_file->store_32(12288); // suggested_buffer_size
    output_file->store_32(0xFFFFFFFF); // quality (max for audio)
    output_file->store_32(channels * 2); // sample_size (16-bit per channel)
    output_file->store_32(0); // reserved
    output_file->store_32(0); // reserved
    output_file->store_32(0); // reserved
    output_file->store_32(0); // reserved
    
    // Audio stream format (strf) - WAVEFORMATEX
    output_file->store_buffer((const uint8_t*)"strf", 4);
    output_file->store_32(16); // Size (standard PCM format)
    
    output_file->store_16(1); // wFormatTag (PCM)
    output_file->store_16(channels); // nChannels
    output_file->store_32(sample_rate); // nSamplesPerSec
    output_file->store_32(sample_rate * channels * 2); // nAvgBytesPerSec (16-bit)
    output_file->store_16(channels * 2); // nBlockAlign (16-bit per channel)
    output_file->store_16(16); // wBitsPerSample
    
    return OK;
}

Error PostMergeProcessor::interleave_avi_data(const String &video_path, const String &audio_path, Ref<FileAccess> output_file, const AviFileInfo &video_info, const AviFileInfo &audio_info) {
    if (config.enable_debug_output) {
        print_line("PostMergeProcessor: Starting data interleaving");
        print_line("  Video movi offset: " + String::num_int64(video_info.movi_data_offset) + ", size: " + String::num_int64(video_info.movi_data_size));
        print_line("  Audio movi offset: " + String::num_int64(audio_info.movi_data_offset) + ", size: " + String::num_int64(audio_info.movi_data_size));
    }
    
    // Open input files
    Ref<FileAccess> video_file = FileAccess::open(video_path, FileAccess::READ);
    Ref<FileAccess> audio_file = FileAccess::open(audio_path, FileAccess::READ);
    
    if (video_file.is_null() || audio_file.is_null()) {
        return ERR_FILE_CANT_OPEN;
    }
    
    // If no index entries, scan movi data directly
    Vector<AviFileInfo::IndexEntry> video_chunks;
    Vector<AviFileInfo::IndexEntry> audio_chunks;
    
    if (video_info.index_entries.size() == 0 && video_info.movi_data_offset > 0) {
        // Scan video movi data
        scan_movi_chunks(video_file, video_info.movi_data_offset, video_info.movi_data_size, video_chunks);
        if (config.enable_debug_output) {
            print_line("PostMergeProcessor: Found " + String::num_int64(video_chunks.size()) + " video chunks by scanning");
        }
    } else {
        video_chunks = video_info.index_entries;
    }
    
    if (audio_info.index_entries.size() == 0 && audio_info.movi_data_offset > 0) {
        // Scan audio movi data
        scan_movi_chunks(audio_file, audio_info.movi_data_offset, audio_info.movi_data_size, audio_chunks);
        if (config.enable_debug_output) {
            print_line("PostMergeProcessor: Found " + String::num_int64(audio_chunks.size()) + " audio chunks by scanning");
        }
    } else {
        audio_chunks = audio_info.index_entries;
    }
    
    // Write movi list header
    output_file->store_buffer((const uint8_t*)"LIST", 4);
    uint64_t movi_size_pos = output_file->get_position();
    output_file->store_32(0); // Placeholder
    output_file->store_buffer((const uint8_t*)"movi", 4);
    uint64_t movi_start = output_file->get_position();
    
    Vector<AviFileInfo::IndexEntry> merged_index;
    
    // Simple interleaving: alternate between video and audio chunks
    int video_idx = 0, audio_idx = 0;
    uint32_t chunk_offset = 4; // Relative to movi data start
    
    while (video_idx < video_chunks.size() || audio_idx < audio_chunks.size()) {
        // Write video chunk if available
        if (video_idx < video_chunks.size()) {
            const AviFileInfo::IndexEntry &video_entry = video_chunks[video_idx];
            
            // Read and copy video chunk (keep original format)
            video_file->seek(video_info.movi_data_offset + video_entry.chunk_offset);
            
            // Read chunk header
            char fourcc[4];
            video_file->get_buffer((uint8_t*)fourcc, 4);
            uint32_t chunk_size = video_file->get_32();
            
            // Write video chunk with stream 0 ID
            output_file->store_buffer((const uint8_t*)"00db", 4); // Stream 0, video
            output_file->store_32(chunk_size);
            
            // Copy data
            Vector<uint8_t> chunk_data;
            chunk_data.resize(chunk_size);
            video_file->get_buffer(chunk_data.ptrw(), chunk_size);
            output_file->store_buffer(chunk_data.ptr(), chunk_size);
            
            // Pad to even boundary
            if (chunk_size % 2 != 0) {
                output_file->store_8(0);
                chunk_size++;
            }
            
            // Add to merged index
            AviFileInfo::IndexEntry merged_entry;
            memcpy(merged_entry.fourcc, "00db", 4);
            merged_entry.flags = 0x10; // AVIIF_KEYFRAME
            merged_entry.chunk_offset = chunk_offset;
            merged_entry.chunk_size = video_entry.chunk_size;
            merged_index.push_back(merged_entry);
            
            chunk_offset += 8 + chunk_size; // 4cc + size + data + padding
            video_idx++;
        }
        
        // Write audio chunk if available (convert 32-bit to 16-bit)
        if (audio_idx < audio_chunks.size()) {
            const AviFileInfo::IndexEntry &audio_entry = audio_chunks[audio_idx];
            
            // Read audio chunk
            audio_file->seek(audio_info.movi_data_offset + audio_entry.chunk_offset);
            
            // Read chunk header
            char fourcc[4];
            audio_file->get_buffer((uint8_t*)fourcc, 4);
            uint32_t chunk_size = audio_file->get_32();
            
            // Read 32-bit audio data
            Vector<int32_t> audio_data_32;
            audio_data_32.resize(chunk_size / 4);
            audio_file->get_buffer((uint8_t*)audio_data_32.ptrw(), chunk_size);
            
            // Convert to 16-bit
            Vector<int16_t> audio_data_16;
            audio_data_16.resize(audio_data_32.size());
            
            for (int i = 0; i < audio_data_32.size(); i++) {
                // Convert 32-bit to 16-bit with proper scaling
                int32_t sample_32 = audio_data_32[i];
                int16_t sample_16 = (int16_t)CLAMP(sample_32 >> 16, -32768, 32767);
                audio_data_16.write[i] = sample_16;
            }
            
            uint32_t converted_size = audio_data_16.size() * 2; // 16-bit
            
            // Write audio chunk with stream 1 ID
            output_file->store_buffer((const uint8_t*)"01wb", 4); // Stream 1, audio
            output_file->store_32(converted_size);
            output_file->store_buffer((const uint8_t*)audio_data_16.ptr(), converted_size);
            
            // Pad to even boundary
            if (converted_size % 2 != 0) {
                output_file->store_8(0);
                converted_size++;
            }
            
            // Add to merged index
            AviFileInfo::IndexEntry merged_entry;
            memcpy(merged_entry.fourcc, "01wb", 4);
            merged_entry.flags = 0; // No special flags for audio
            merged_entry.chunk_offset = chunk_offset;
            merged_entry.chunk_size = converted_size;
            merged_index.push_back(merged_entry);
            
            chunk_offset += 8 + converted_size; // 4cc + size + data + padding
            audio_idx++;
        }
    }
    
    // Update movi size
    uint64_t movi_end = output_file->get_position();
    uint32_t movi_size = movi_end - movi_start;
    output_file->seek(movi_size_pos);
    output_file->store_32(movi_size + 4); // +4 for 'movi' fourcc
    output_file->seek(movi_end);
    
    // Write index
    Error index_error = write_merged_avi_index(output_file, merged_index);
    if (index_error != OK) {
        return index_error;
    }
    
        print_line("PostMergeProcessor: Data interleaving completed");
        print_line("  Merged chunks: " + String::num_int64(merged_index.size()));
        print_line("  Movi size: " + String::num_int64(movi_size) + " bytes");
    return OK;
}

void PostMergeProcessor::scan_movi_chunks(Ref<FileAccess> file, uint64_t movi_offset, uint32_t movi_size, Vector<AviFileInfo::IndexEntry> &chunks) {
    uint64_t pos = movi_offset;
    uint64_t movi_end = movi_offset + movi_size;
    
    while (pos < movi_end - 8) {
        file->seek(pos);
        
        // Read chunk header
        char fourcc[4];
        file->get_buffer((uint8_t*)fourcc, 4);
        uint32_t chunk_size = file->get_32();
        
        // Validate chunk
        if (chunk_size > movi_size || chunk_size > 0x7FFFFFFF) {
            break;
        }
        
        // Check if it's a valid data chunk
        if ((memcmp(fourcc, "00db", 4) == 0) || // Video
            (memcmp(fourcc, "00dc", 4) == 0) || // Compressed video
            (memcmp(fourcc, "00wb", 4) == 0) || // Audio  
            (memcmp(fourcc, "01wb", 4) == 0)) { // Audio stream 1
            
            AviFileInfo::IndexEntry entry;
            memcpy(entry.fourcc, fourcc, 4);
            entry.flags = (fourcc[2] == 'd') ? 0x10 : 0; // KEYFRAME for video
            entry.chunk_offset = pos - movi_offset; // Relative to movi start
            entry.chunk_size = chunk_size;
            chunks.push_back(entry);
        }
        
        // Move to next chunk
        pos += 8 + chunk_size;
        if (chunk_size % 2 != 0) {
            pos++; // Align to even boundary
        }
    }
}

Error PostMergeProcessor::write_merged_avi_index(Ref<FileAccess> output_file, const Vector<AviFileInfo::IndexEntry> &merged_index) {
    
    // Write idx1 chunk
    output_file->store_buffer((const uint8_t*)"idx1", 4);
    output_file->store_32(merged_index.size() * 16); // Each entry is 16 bytes
    
    for (const AviFileInfo::IndexEntry &entry : merged_index) {
        output_file->store_buffer((const uint8_t*)entry.fourcc, 4);
        output_file->store_32(entry.flags);
        output_file->store_32(entry.chunk_offset);
        output_file->store_32(entry.chunk_size);
    }
    
    return OK;
}

// Timestamp-based interleaving implementation
Error PostMergeProcessor::interleave_avi_data_timestamped(const String &video_path, const String &audio_path, Ref<FileAccess> output_file, const AviFileInfo &video_info, const AviFileInfo &audio_info) {
    
    // Open input files
    Ref<FileAccess> video_file = FileAccess::open(video_path, FileAccess::READ);
    Ref<FileAccess> audio_file = FileAccess::open(audio_path, FileAccess::READ);
    
    if (video_file.is_null() || audio_file.is_null()) {
        return ERR_FILE_CANT_OPEN;
    }
    
    // Get chunks with fallback to scanning if no index
    Vector<AviFileInfo::IndexEntry> video_chunks;
    Vector<AviFileInfo::IndexEntry> audio_chunks;
    
    if (video_info.index_entries.size() == 0 && video_info.movi_data_offset > 0) {
        scan_movi_chunks(video_file, video_info.movi_data_offset, video_info.movi_data_size, video_chunks);
    } else {
        video_chunks = video_info.index_entries;
    }
    
    if (audio_info.index_entries.size() == 0 && audio_info.movi_data_offset > 0) {
        scan_movi_chunks(audio_file, audio_info.movi_data_offset, audio_info.movi_data_size, audio_chunks);
    } else {
        audio_chunks = audio_info.index_entries;
    }
    
    // Calculate timestamps for all chunks
    Vector<TimestampedChunk> all_chunks;
    calculate_chunk_timestamps(video_chunks, video_info, true, all_chunks);
    calculate_chunk_timestamps(audio_chunks, audio_info, false, all_chunks);
    
    
    // Sort by timestamp
    all_chunks.sort();
    
    // Write movi list header
    output_file->store_buffer((const uint8_t*)"LIST", 4);
    uint64_t movi_size_pos = output_file->get_position();
    output_file->store_32(0); // Placeholder
    output_file->store_buffer((const uint8_t*)"movi", 4);
    uint64_t movi_start = output_file->get_position();
    
    Vector<AviFileInfo::IndexEntry> merged_index;
    uint32_t chunk_offset = 4; // Relative to movi data start
    
    // Process chunks in timestamp order
    for (int i = 0; i < all_chunks.size(); i++) {
        const TimestampedChunk &chunk = all_chunks[i];
        
        
        Error write_error = write_timestamped_chunk(
            chunk.type == TimestampedChunk::VIDEO_CHUNK ? video_file : audio_file,
            output_file,
            chunk,
            chunk.type == TimestampedChunk::VIDEO_CHUNK,
            chunk_offset,
            merged_index
        );
        
        if (write_error != OK) {
            return write_error;
        }
        
        // Validate A/V sync periodically
        if (config.validate_sync && i > 0 && i % 30 == 0) {
            // Find nearest video and audio chunks
            uint64_t nearest_video_ts = 0;
            uint64_t nearest_audio_ts = 0;
            bool found_video = false;
            bool found_audio = false;
            
            for (int j = i - 30; j <= i; j++) {
                if (all_chunks[j].type == TimestampedChunk::VIDEO_CHUNK && !found_video) {
                    nearest_video_ts = all_chunks[j].timestamp_us;
                    found_video = true;
                }
                if (all_chunks[j].type == TimestampedChunk::AUDIO_CHUNK && !found_audio) {
                    nearest_audio_ts = all_chunks[j].timestamp_us;
                    found_audio = true;
                }
                if (found_video && found_audio) break;
            }
            
            if (found_video && found_audio) {
                if (!validate_av_sync(nearest_video_ts, nearest_audio_ts)) {
                }
            }
        }
    }
    
    // Update movi size
    uint64_t movi_end = output_file->get_position();
    uint32_t movi_size = movi_end - movi_start;
    output_file->seek(movi_size_pos);
    output_file->store_32(movi_size + 4); // +4 for 'movi' fourcc
    output_file->seek(movi_end);
    
    // Write index
    Error index_error = write_merged_avi_index(output_file, merged_index);
    if (index_error != OK) {
        return index_error;
    }
    
    
    return OK;
}

void PostMergeProcessor::calculate_chunk_timestamps(const Vector<AviFileInfo::IndexEntry> &chunks, const AviFileInfo &info, bool is_video, Vector<TimestampedChunk> &timestamped_chunks) {
    if (is_video) {
        // Video timestamp calculation
        uint64_t frame_duration_us = info.microsec_per_frame;
        
        for (int i = 0; i < chunks.size(); i++) {
            TimestampedChunk tc;
            tc.type = TimestampedChunk::VIDEO_CHUNK;
            tc.timestamp_us = i * frame_duration_us;
            tc.file_offset = info.movi_data_offset + chunks[i].chunk_offset;
            tc.chunk_size = chunks[i].chunk_size;
            tc.index = i;
            timestamped_chunks.push_back(tc);
        }
        
    } else {
        // Audio timestamp calculation
        uint32_t sample_rate = 44100; // Default
        uint32_t channels = 2; // Default stereo
        uint32_t bytes_per_sample = 4; // Original 32-bit
        
        // Get actual values from stream info if available
        if (info.streams_info.size() > 0) {
            const AviFileInfo::StreamInfo &audio_stream = info.streams_info[0];
            if (audio_stream.rate > 0 && audio_stream.scale > 0) {
                sample_rate = audio_stream.rate / audio_stream.scale;
                channels = audio_stream.sample_size / 4; // 32-bit per channel
            }
        }
        
        // Calculate timestamp for each audio chunk
        uint64_t cumulative_samples = 0;
        
        for (int i = 0; i < chunks.size(); i++) {
            TimestampedChunk tc;
            tc.type = TimestampedChunk::AUDIO_CHUNK;
            
            // Calculate timestamp based on cumulative samples
            tc.timestamp_us = (cumulative_samples * 1000000) / sample_rate;
            tc.file_offset = info.movi_data_offset + chunks[i].chunk_offset;
            tc.chunk_size = chunks[i].chunk_size;
            tc.index = i;
            timestamped_chunks.push_back(tc);
            
            // Update cumulative samples
            uint32_t samples_in_chunk = chunks[i].chunk_size / (channels * bytes_per_sample);
            cumulative_samples += samples_in_chunk;
        }
        
    }
}

uint32_t PostMergeProcessor::calculate_optimal_audio_chunk_size(uint32_t video_fps, uint32_t sample_rate) {
    // Calculate optimal audio chunk size to match video frame duration
    double frame_duration_s = 1.0 / video_fps;
    uint32_t samples_per_frame = sample_rate * frame_duration_s;
    
    // Align to 256 sample boundary for efficiency
    return ((samples_per_frame + 255) / 256) * 256;
}

bool PostMergeProcessor::validate_av_sync(uint64_t video_ts, uint64_t audio_ts) {
    uint64_t drift = video_ts > audio_ts ? video_ts - audio_ts : audio_ts - video_ts;
    
    if (drift > config.max_av_drift_us) {
        return false;
    }
    
    return true;
}

Error PostMergeProcessor::write_timestamped_chunk(Ref<FileAccess> input_file, Ref<FileAccess> output_file, 
                                                 const TimestampedChunk &chunk, bool is_video, 
                                                 uint32_t &chunk_offset, Vector<AviFileInfo::IndexEntry> &merged_index) {
    // Seek to chunk location
    input_file->seek(chunk.file_offset);
    
    // Read chunk header
    char fourcc[4];
    input_file->get_buffer((uint8_t*)fourcc, 4);
    uint32_t original_size = input_file->get_32();
    
    if (is_video) {
        // Write video chunk (keep original format)
        output_file->store_buffer((const uint8_t*)"00db", 4); // Stream 0, video
        output_file->store_32(original_size);
        
        // Copy data
        Vector<uint8_t> chunk_data;
        chunk_data.resize(original_size);
        input_file->get_buffer(chunk_data.ptrw(), original_size);
        output_file->store_buffer(chunk_data.ptr(), original_size);
        
        // Pad to even boundary
        if (original_size % 2 != 0) {
            output_file->store_8(0);
            original_size++;
        }
        
        // Add to index
        AviFileInfo::IndexEntry entry;
        memcpy(entry.fourcc, "00db", 4);
        entry.flags = 0x10; // AVIIF_KEYFRAME
        entry.chunk_offset = chunk_offset;
        entry.chunk_size = original_size;
        merged_index.push_back(entry);
        
        chunk_offset += 8 + original_size;
    } else {
        // Audio chunk - convert 32-bit to 16-bit
        Vector<int32_t> audio_data_32;
        audio_data_32.resize(original_size / 4);
        input_file->get_buffer((uint8_t*)audio_data_32.ptrw(), original_size);
        
        // Convert to 16-bit
        Vector<int16_t> audio_data_16;
        audio_data_16.resize(audio_data_32.size());
        
        for (int i = 0; i < audio_data_32.size(); i++) {
            int32_t sample_32 = audio_data_32[i];
            int16_t sample_16 = (int16_t)CLAMP(sample_32 >> 16, -32768, 32767);
            audio_data_16.write[i] = sample_16;
        }
        
        uint32_t converted_size = audio_data_16.size() * 2;
        
        // Write audio chunk
        output_file->store_buffer((const uint8_t*)"01wb", 4); // Stream 1, audio
        output_file->store_32(converted_size);
        output_file->store_buffer((const uint8_t*)audio_data_16.ptr(), converted_size);
        
        // Pad to even boundary
        if (converted_size % 2 != 0) {
            output_file->store_8(0);
            converted_size++;
        }
        
        // Add to index
        AviFileInfo::IndexEntry entry;
        memcpy(entry.fourcc, "01wb", 4);
        entry.flags = 0;
        entry.chunk_offset = chunk_offset;
        entry.chunk_size = converted_size;
        merged_index.push_back(entry);
        
        chunk_offset += 8 + converted_size;
    }
    
    return OK;
}