/**************************************************************************/
/*  simple_video_writer.h                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#ifndef SIMPLE_VIDEO_WRITER_H
#define SIMPLE_VIDEO_WRITER_H

#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/object/ref_counted.h"

class SimpleVideoWriter : public RefCounted {
    GDCLASS(SimpleVideoWriter, RefCounted);

private:
    Ref<FileAccess> f;
    String base_path;
    
    uint32_t fps;
    uint32_t frame_count = 0;
    float quality = 0.75f;
    
    uint64_t total_frames_ofs;
    uint64_t total_frames_ofs2; 
    uint64_t total_frames_ofs3;
    uint64_t movi_data_ofs;
    
    Vector<uint32_t> jpg_frame_sizes;

public:
    SimpleVideoWriter();
    ~SimpleVideoWriter();
    
    Error open(const String &p_path, const Size2i &p_movie_size, uint32_t p_fps, float p_quality = 0.75f);
    Error write_frame(const Ref<Image> &p_image);
    void close();
    
    void set_quality(float p_quality) { quality = p_quality; }
    uint32_t get_frame_count() const { return frame_count; }
};

#endif // SIMPLE_VIDEO_WRITER_H 