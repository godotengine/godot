#ifndef MINIMP3_EXT_H
#define MINIMP3_EXT_H
/*
    https://github.com/lieff/minimp3
    To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide.
    This software is distributed without any warranty.
    See <http://creativecommons.org/publicdomain/zero/1.0/>.
*/
#include <stddef.h>
#include "minimp3.h"

/* flags for mp3dec_ex_open_* functions */
#define MP3D_SEEK_TO_BYTE   0      /* mp3dec_ex_seek seeks to byte in stream */
#define MP3D_SEEK_TO_SAMPLE 1      /* mp3dec_ex_seek precisely seeks to sample using index (created during duration calculation scan or when mp3dec_ex_seek called) */
#define MP3D_DO_NOT_SCAN    2      /* do not scan whole stream for duration if vbrtag not found, mp3dec_ex_t::samples will be filled only if mp3dec_ex_t::vbr_tag_found == 1 */
#ifdef MINIMP3_ALLOW_MONO_STEREO_TRANSITION
#define MP3D_ALLOW_MONO_STEREO_TRANSITION  4
#define MP3D_FLAGS_MASK 7
#else
#define MP3D_FLAGS_MASK 3
#endif

/* compile-time config */
#define MINIMP3_PREDECODE_FRAMES 2 /* frames to pre-decode and skip after seek (to fill internal structures) */
/*#define MINIMP3_SEEK_IDX_LINEAR_SEARCH*/ /* define to use linear index search instead of binary search on seek */
#define MINIMP3_IO_SIZE (128*1024) /* io buffer size for streaming functions, must be greater than MINIMP3_BUF_SIZE */
#define MINIMP3_BUF_SIZE (16*1024) /* buffer which can hold minimum 10 consecutive mp3 frames (~16KB) worst case */
/*#define MINIMP3_SCAN_LIMIT (256*1024)*/ /* how many bytes will be scanned to search first valid mp3 frame, to prevent stall on large non-mp3 files */
#define MINIMP3_ENABLE_RING 0      /* WIP enable hardware magic ring buffer if available, to make less input buffer memmove(s) in callback IO mode */

/* return error codes */
#define MP3D_E_PARAM   -1
#define MP3D_E_MEMORY  -2
#define MP3D_E_IOERROR -3
#define MP3D_E_USER    -4  /* can be used to stop processing from callbacks without indicating specific error */
#define MP3D_E_DECODE  -5  /* decode error which can't be safely skipped, such as sample rate, layer and channels change */

typedef struct
{
    mp3d_sample_t *buffer;
    size_t samples; /* channels included, byte size = samples*sizeof(mp3d_sample_t) */
    int channels, hz, layer, avg_bitrate_kbps;
} mp3dec_file_info_t;

typedef struct
{
    const uint8_t *buffer;
    size_t size;
} mp3dec_map_info_t;

typedef struct
{
    uint64_t sample;
    uint64_t offset;
} mp3dec_frame_t;

typedef struct
{
    mp3dec_frame_t *frames;
    size_t num_frames, capacity;
} mp3dec_index_t;

typedef size_t (*MP3D_READ_CB)(void *buf, size_t size, void *user_data);
typedef int (*MP3D_SEEK_CB)(uint64_t position, void *user_data);

typedef struct
{
    MP3D_READ_CB read;
    void *read_data;
    MP3D_SEEK_CB seek;
    void *seek_data;
} mp3dec_io_t;

typedef struct
{
    mp3dec_t mp3d;
    mp3dec_map_info_t file;
    mp3dec_io_t *io;
    mp3dec_index_t index;
    uint64_t offset, samples, detected_samples, cur_sample, start_offset, end_offset;
    mp3dec_frame_info_t info;
    mp3d_sample_t buffer[MINIMP3_MAX_SAMPLES_PER_FRAME];
    size_t input_consumed, input_filled;
    int is_file, flags, vbr_tag_found, indexes_built;
    int free_format_bytes;
    int buffer_samples, buffer_consumed, to_skip, start_delay;
    int last_error;
} mp3dec_ex_t;

typedef int (*MP3D_ITERATE_CB)(void *user_data, const uint8_t *frame, int frame_size, int free_format_bytes, size_t buf_size, uint64_t offset, mp3dec_frame_info_t *info);
typedef int (*MP3D_PROGRESS_CB)(void *user_data, size_t file_size, uint64_t offset, mp3dec_frame_info_t *info);

#ifdef __cplusplus
extern "C" {
#endif

/* detect mp3/mpa format */
int mp3dec_detect_buf(const uint8_t *buf, size_t buf_size);
int mp3dec_detect_cb(mp3dec_io_t *io, uint8_t *buf, size_t buf_size);
/* decode whole buffer block */
int mp3dec_load_buf(mp3dec_t *dec, const uint8_t *buf, size_t buf_size, mp3dec_file_info_t *info, MP3D_PROGRESS_CB progress_cb, void *user_data);
int mp3dec_load_cb(mp3dec_t *dec, mp3dec_io_t *io, uint8_t *buf, size_t buf_size, mp3dec_file_info_t *info, MP3D_PROGRESS_CB progress_cb, void *user_data);
/* iterate through frames */
int mp3dec_iterate_buf(const uint8_t *buf, size_t buf_size, MP3D_ITERATE_CB callback, void *user_data);
int mp3dec_iterate_cb(mp3dec_io_t *io, uint8_t *buf, size_t buf_size, MP3D_ITERATE_CB callback, void *user_data);
/* streaming decoder with seeking capability */
int mp3dec_ex_open_buf(mp3dec_ex_t *dec, const uint8_t *buf, size_t buf_size, int flags);
int mp3dec_ex_open_cb(mp3dec_ex_t *dec, mp3dec_io_t *io, int flags);
void mp3dec_ex_close(mp3dec_ex_t *dec);
int mp3dec_ex_seek(mp3dec_ex_t *dec, uint64_t position);
size_t mp3dec_ex_read_frame(mp3dec_ex_t *dec, mp3d_sample_t **buf, mp3dec_frame_info_t *frame_info, size_t max_samples);
size_t mp3dec_ex_read(mp3dec_ex_t *dec, mp3d_sample_t *buf, size_t samples);
#ifndef MINIMP3_NO_STDIO
/* stdio versions of file detect, load, iterate and stream */
int mp3dec_detect(const char *file_name);
int mp3dec_load(mp3dec_t *dec, const char *file_name, mp3dec_file_info_t *info, MP3D_PROGRESS_CB progress_cb, void *user_data);
int mp3dec_iterate(const char *file_name, MP3D_ITERATE_CB callback, void *user_data);
int mp3dec_ex_open(mp3dec_ex_t *dec, const char *file_name, int flags);
#ifdef _WIN32
int mp3dec_detect_w(const wchar_t *file_name);
int mp3dec_load_w(mp3dec_t *dec, const wchar_t *file_name, mp3dec_file_info_t *info, MP3D_PROGRESS_CB progress_cb, void *user_data);
int mp3dec_iterate_w(const wchar_t *file_name, MP3D_ITERATE_CB callback, void *user_data);
int mp3dec_ex_open_w(mp3dec_ex_t *dec, const wchar_t *file_name, int flags);
#endif
#endif

#ifdef __cplusplus
}
#endif
#endif /*MINIMP3_EXT_H*/

#if defined(MINIMP3_IMPLEMENTATION) && !defined(_MINIMP3_EX_IMPLEMENTATION_GUARD)
#define _MINIMP3_EX_IMPLEMENTATION_GUARD
#include <limits.h>
#include "minimp3.h"

static void mp3dec_skip_id3v1(const uint8_t *buf, size_t *pbuf_size)
{
    size_t buf_size = *pbuf_size;
#ifndef MINIMP3_NOSKIP_ID3V1
    if (buf_size >= 128 && !memcmp(buf + buf_size - 128, "TAG", 3))
    {
        buf_size -= 128;
        if (buf_size >= 227 && !memcmp(buf + buf_size - 227, "TAG+", 4))
            buf_size -= 227;
    }
#endif
#ifndef MINIMP3_NOSKIP_APEV2
    if (buf_size > 32 && !memcmp(buf + buf_size - 32, "APETAGEX", 8))
    {
        buf_size -= 32;
        const uint8_t *tag = buf + buf_size + 8 + 4;
        uint32_t tag_size = (uint32_t)(tag[3] << 24) | (tag[2] << 16) | (tag[1] << 8) | tag[0];
        if (buf_size >= tag_size)
            buf_size -= tag_size;
    }
#endif
    *pbuf_size = buf_size;
}

static size_t mp3dec_skip_id3v2(const uint8_t *buf, size_t buf_size)
{
#define MINIMP3_ID3_DETECT_SIZE 10
#ifndef MINIMP3_NOSKIP_ID3V2
    if (buf_size >= MINIMP3_ID3_DETECT_SIZE && !memcmp(buf, "ID3", 3) && !((buf[5] & 15) || (buf[6] & 0x80) || (buf[7] & 0x80) || (buf[8] & 0x80) || (buf[9] & 0x80)))
    {
        size_t id3v2size = (((buf[6] & 0x7f) << 21) | ((buf[7] & 0x7f) << 14) | ((buf[8] & 0x7f) << 7) | (buf[9] & 0x7f)) + 10;
        if ((buf[5] & 16))
            id3v2size += 10; /* footer */
        return id3v2size;
    }
#endif
    return 0;
}

static void mp3dec_skip_id3(const uint8_t **pbuf, size_t *pbuf_size)
{
    uint8_t *buf = (uint8_t *)(*pbuf);
    size_t buf_size = *pbuf_size;
    size_t id3v2size = mp3dec_skip_id3v2(buf, buf_size);
    if (id3v2size)
    {
        if (id3v2size >= buf_size)
            id3v2size = buf_size;
        buf      += id3v2size;
        buf_size -= id3v2size;
    }
    mp3dec_skip_id3v1(buf, &buf_size);
    *pbuf = (const uint8_t *)buf;
    *pbuf_size = buf_size;
}

static int mp3dec_check_vbrtag(const uint8_t *frame, int frame_size, uint32_t *frames, int *delay, int *padding)
{
    static const char g_xing_tag[4] = { 'X', 'i', 'n', 'g' };
    static const char g_info_tag[4] = { 'I', 'n', 'f', 'o' };
#define FRAMES_FLAG     1
#define BYTES_FLAG      2
#define TOC_FLAG        4
#define VBR_SCALE_FLAG  8
    /* Side info offsets after header:
    /                Mono  Stereo
    /  MPEG1          17     32
    /  MPEG2 & 2.5     9     17*/
    bs_t bs[1];
    L3_gr_info_t gr_info[4];
    bs_init(bs, frame + HDR_SIZE, frame_size - HDR_SIZE);
    if (HDR_IS_CRC(frame))
        get_bits(bs, 16);
    if (L3_read_side_info(bs, gr_info, frame) < 0)
        return 0; /* side info corrupted */

    const uint8_t *tag = frame + HDR_SIZE + bs->pos/8;
    if (memcmp(g_xing_tag, tag, 4) && memcmp(g_info_tag, tag, 4))
        return 0;
    int flags = tag[7];
    if (!((flags & FRAMES_FLAG)))
        return -1;
    tag += 8;
    *frames = (uint32_t)(tag[0] << 24) | (tag[1] << 16) | (tag[2] << 8) | tag[3];
    tag += 4;
    if (flags & BYTES_FLAG)
        tag += 4;
    if (flags & TOC_FLAG)
        tag += 100;
    if (flags & VBR_SCALE_FLAG)
        tag += 4;
    *delay = *padding = 0;
    if (*tag)
    {   /* extension, LAME, Lavc, etc. Should be the same structure. */
        tag += 21;
        if (tag - frame + 14 >= frame_size)
            return 0;
        *delay   = ((tag[0] << 4) | (tag[1] >> 4)) + (528 + 1);
        *padding = (((tag[1] & 0xF) << 8) | tag[2]) - (528 + 1);
    }
    return 1;
}

int mp3dec_detect_buf(const uint8_t *buf, size_t buf_size)
{
    return mp3dec_detect_cb(0, (uint8_t *)buf, buf_size);
}

int mp3dec_detect_cb(mp3dec_io_t *io, uint8_t *buf, size_t buf_size)
{
    if (!buf || (size_t)-1 == buf_size || (io && buf_size < MINIMP3_BUF_SIZE))
        return MP3D_E_PARAM;
    size_t filled = buf_size;
    if (io)
    {
        if (io->seek(0, io->seek_data))
            return MP3D_E_IOERROR;
        filled = io->read(buf, MINIMP3_ID3_DETECT_SIZE, io->read_data);
        if (filled > MINIMP3_ID3_DETECT_SIZE)
            return MP3D_E_IOERROR;
    }
    if (filled < MINIMP3_ID3_DETECT_SIZE)
        return MP3D_E_USER; /* too small, can't be mp3/mpa */
    if (mp3dec_skip_id3v2(buf, filled))
        return 0; /* id3v2 tag is enough evidence */
    if (io)
    {
        size_t readed = io->read(buf + MINIMP3_ID3_DETECT_SIZE, buf_size - MINIMP3_ID3_DETECT_SIZE, io->read_data);
        if (readed > (buf_size - MINIMP3_ID3_DETECT_SIZE))
            return MP3D_E_IOERROR;
        filled += readed;
        if (filled < MINIMP3_BUF_SIZE)
            mp3dec_skip_id3v1(buf, &filled);
    } else
    {
        mp3dec_skip_id3v1(buf, &filled);
        if (filled > MINIMP3_BUF_SIZE)
            filled = MINIMP3_BUF_SIZE;
    }
    int free_format_bytes, frame_size;
    mp3d_find_frame(buf, filled, &free_format_bytes, &frame_size);
    if (frame_size)
        return 0; /* MAX_FRAME_SYNC_MATCHES consecutive frames found */
    return MP3D_E_USER;
}

int mp3dec_load_buf(mp3dec_t *dec, const uint8_t *buf, size_t buf_size, mp3dec_file_info_t *info, MP3D_PROGRESS_CB progress_cb, void *user_data)
{
    return mp3dec_load_cb(dec, 0, (uint8_t *)buf, buf_size, info, progress_cb, user_data);
}

int mp3dec_load_cb(mp3dec_t *dec, mp3dec_io_t *io, uint8_t *buf, size_t buf_size, mp3dec_file_info_t *info, MP3D_PROGRESS_CB progress_cb, void *user_data)
{
    if (!dec || !buf || !info || (size_t)-1 == buf_size || (io && buf_size < MINIMP3_BUF_SIZE))
        return MP3D_E_PARAM;
    uint64_t detected_samples = 0;
    size_t orig_buf_size = buf_size;
    int to_skip = 0;
    mp3dec_frame_info_t frame_info;
    memset(info, 0, sizeof(*info));
    memset(&frame_info, 0, sizeof(frame_info));

    /* skip id3 */
    size_t filled = 0, consumed = 0;
    int eof = 0, ret = 0;
    if (io)
    {
        if (io->seek(0, io->seek_data))
            return MP3D_E_IOERROR;
        filled = io->read(buf, MINIMP3_ID3_DETECT_SIZE, io->read_data);
        if (filled > MINIMP3_ID3_DETECT_SIZE)
            return MP3D_E_IOERROR;
        if (MINIMP3_ID3_DETECT_SIZE != filled)
            return 0;
        size_t id3v2size = mp3dec_skip_id3v2(buf, filled);
        if (id3v2size)
        {
            if (io->seek(id3v2size, io->seek_data))
                return MP3D_E_IOERROR;
            filled = io->read(buf, buf_size, io->read_data);
            if (filled > buf_size)
                return MP3D_E_IOERROR;
        } else
        {
            size_t readed = io->read(buf + MINIMP3_ID3_DETECT_SIZE, buf_size - MINIMP3_ID3_DETECT_SIZE, io->read_data);
            if (readed > (buf_size - MINIMP3_ID3_DETECT_SIZE))
                return MP3D_E_IOERROR;
            filled += readed;
        }
        if (filled < MINIMP3_BUF_SIZE)
            mp3dec_skip_id3v1(buf, &filled);
    } else
    {
        mp3dec_skip_id3((const uint8_t **)&buf, &buf_size);
        if (!buf_size)
            return 0;
    }
    /* try to make allocation size assumption by first frame or vbr tag */
    mp3dec_init(dec);
    int samples;
    do
    {
        uint32_t frames;
        int i, delay, padding, free_format_bytes = 0, frame_size = 0;
        const uint8_t *hdr;
        if (io)
        {
            if (!eof && filled - consumed < MINIMP3_BUF_SIZE)
            {   /* keep minimum 10 consecutive mp3 frames (~16KB) worst case */
                memmove(buf, buf + consumed, filled - consumed);
                filled -= consumed;
                consumed = 0;
                size_t readed = io->read(buf + filled, buf_size - filled, io->read_data);
                if (readed > (buf_size - filled))
                    return MP3D_E_IOERROR;
                if (readed != (buf_size - filled))
                    eof = 1;
                filled += readed;
                if (eof)
                    mp3dec_skip_id3v1(buf, &filled);
            }
            i = mp3d_find_frame(buf + consumed, filled - consumed, &free_format_bytes, &frame_size);
            consumed += i;
            hdr = buf + consumed;
        } else
        {
            i = mp3d_find_frame(buf, buf_size, &free_format_bytes, &frame_size);
            buf      += i;
            buf_size -= i;
            hdr = buf;
        }
        if (i && !frame_size)
            continue;
        if (!frame_size)
            return 0;
        frame_info.channels = HDR_IS_MONO(hdr) ? 1 : 2;
        frame_info.hz = hdr_sample_rate_hz(hdr);
        frame_info.layer = 4 - HDR_GET_LAYER(hdr);
        frame_info.bitrate_kbps = hdr_bitrate_kbps(hdr);
        frame_info.frame_bytes = frame_size;
        samples = hdr_frame_samples(hdr)*frame_info.channels;
        if (3 != frame_info.layer)
            break;
        ret = mp3dec_check_vbrtag(hdr, frame_size, &frames, &delay, &padding);
        if (ret > 0)
        {
            padding *= frame_info.channels;
            to_skip = delay*frame_info.channels;
            detected_samples = samples*(uint64_t)frames;
            if (detected_samples >= (uint64_t)to_skip)
                detected_samples -= to_skip;
            if (padding > 0 && detected_samples >= (uint64_t)padding)
                detected_samples -= padding;
            if (!detected_samples)
                return 0;
        }
        if (ret)
        {
            if (io)
            {
                consumed += frame_size;
            } else
            {
                buf      += frame_size;
                buf_size -= frame_size;
            }
        }
        break;
    } while(1);
    size_t allocated = MINIMP3_MAX_SAMPLES_PER_FRAME*sizeof(mp3d_sample_t);
    if (detected_samples)
        allocated += detected_samples*sizeof(mp3d_sample_t);
    else
        allocated += (buf_size/frame_info.frame_bytes)*samples*sizeof(mp3d_sample_t);
    info->buffer = (mp3d_sample_t*)malloc(allocated);
    if (!info->buffer)
        return MP3D_E_MEMORY;
    /* save info */
    info->channels = frame_info.channels;
    info->hz       = frame_info.hz;
    info->layer    = frame_info.layer;
    /* decode all frames */
    size_t avg_bitrate_kbps = 0, frames = 0;
    do
    {
        if ((allocated - info->samples*sizeof(mp3d_sample_t)) < MINIMP3_MAX_SAMPLES_PER_FRAME*sizeof(mp3d_sample_t))
        {
            allocated *= 2;
            mp3d_sample_t *alloc_buf = (mp3d_sample_t*)realloc(info->buffer, allocated);
            if (!alloc_buf)
                return MP3D_E_MEMORY;
            info->buffer = alloc_buf;
        }
        if (io)
        {
            if (!eof && filled - consumed < MINIMP3_BUF_SIZE)
            {   /* keep minimum 10 consecutive mp3 frames (~16KB) worst case */
                memmove(buf, buf + consumed, filled - consumed);
                filled -= consumed;
                consumed = 0;
                size_t readed = io->read(buf + filled, buf_size - filled, io->read_data);
                if (readed != (buf_size - filled))
                    eof = 1;
                filled += readed;
                if (eof)
                    mp3dec_skip_id3v1(buf, &filled);
            }
            samples = mp3dec_decode_frame(dec, buf + consumed, filled - consumed, info->buffer + info->samples, &frame_info);
            consumed += frame_info.frame_bytes;
        } else
        {
            samples = mp3dec_decode_frame(dec, buf, MINIMP3_MIN(buf_size, (size_t)INT_MAX), info->buffer + info->samples, &frame_info);
            buf      += frame_info.frame_bytes;
            buf_size -= frame_info.frame_bytes;
        }
        if (samples)
        {
            if (info->hz != frame_info.hz || info->layer != frame_info.layer)
            {
                ret = MP3D_E_DECODE;
                break;
            }
            if (info->channels && info->channels != frame_info.channels)
            {
#ifdef MINIMP3_ALLOW_MONO_STEREO_TRANSITION
                info->channels = 0; /* mark file with mono-stereo transition */
#else
                ret = MP3D_E_DECODE;
                break;
#endif
            }
            samples *= frame_info.channels;
            if (to_skip)
            {
                size_t skip = MINIMP3_MIN(samples, to_skip);
                to_skip -= skip;
                samples -= skip;
                memmove(info->buffer, info->buffer + skip, samples*sizeof(mp3d_sample_t));
            }
            info->samples += samples;
            avg_bitrate_kbps += frame_info.bitrate_kbps;
            frames++;
            if (progress_cb)
            {
                ret = progress_cb(user_data, orig_buf_size, orig_buf_size - buf_size, &frame_info);
                if (ret)
                    break;
            }
        }
    } while (frame_info.frame_bytes);
    if (detected_samples && info->samples > detected_samples)
        info->samples = detected_samples; /* cut padding */
    /* reallocate to normal buffer size */
    if (allocated != info->samples*sizeof(mp3d_sample_t))
    {
        mp3d_sample_t *alloc_buf = (mp3d_sample_t*)realloc(info->buffer, info->samples*sizeof(mp3d_sample_t));
        if (!alloc_buf && info->samples)
            return MP3D_E_MEMORY;
        info->buffer = alloc_buf;
    }
    if (frames)
        info->avg_bitrate_kbps = avg_bitrate_kbps/frames;
    return ret;
}

int mp3dec_iterate_buf(const uint8_t *buf, size_t buf_size, MP3D_ITERATE_CB callback, void *user_data)
{
    const uint8_t *orig_buf = buf;
    if (!buf || (size_t)-1 == buf_size || !callback)
        return MP3D_E_PARAM;
    /* skip id3 */
    mp3dec_skip_id3(&buf, &buf_size);
    if (!buf_size)
        return 0;
    mp3dec_frame_info_t frame_info;
    memset(&frame_info, 0, sizeof(frame_info));
    do
    {
        int free_format_bytes = 0, frame_size = 0, ret;
        int i = mp3d_find_frame(buf, buf_size, &free_format_bytes, &frame_size);
        buf      += i;
        buf_size -= i;
        if (i && !frame_size)
            continue;
        if (!frame_size)
            break;
        const uint8_t *hdr = buf;
        frame_info.channels = HDR_IS_MONO(hdr) ? 1 : 2;
        frame_info.hz = hdr_sample_rate_hz(hdr);
        frame_info.layer = 4 - HDR_GET_LAYER(hdr);
        frame_info.bitrate_kbps = hdr_bitrate_kbps(hdr);
        frame_info.frame_bytes = frame_size;

        if (callback)
        {
            ret = callback(user_data, hdr, frame_size, free_format_bytes, buf_size, hdr - orig_buf, &frame_info);
            if (ret != 0)
                return ret;
        }
        buf      += frame_size;
        buf_size -= frame_size;
    } while (1);
    return 0;
}

int mp3dec_iterate_cb(mp3dec_io_t *io, uint8_t *buf, size_t buf_size, MP3D_ITERATE_CB callback, void *user_data)
{
    if (!io || !buf || (size_t)-1 == buf_size || buf_size < MINIMP3_BUF_SIZE || !callback)
        return MP3D_E_PARAM;
    size_t filled = io->read(buf, MINIMP3_ID3_DETECT_SIZE, io->read_data), consumed = 0;
    uint64_t readed = 0;
    mp3dec_frame_info_t frame_info;
    int eof = 0;
    memset(&frame_info, 0, sizeof(frame_info));
    if (filled > MINIMP3_ID3_DETECT_SIZE)
        return MP3D_E_IOERROR;
    if (MINIMP3_ID3_DETECT_SIZE != filled)
        return 0;
    size_t id3v2size = mp3dec_skip_id3v2(buf, filled);
    if (id3v2size)
    {
        if (io->seek(id3v2size, io->seek_data))
            return MP3D_E_IOERROR;
        filled = io->read(buf, buf_size, io->read_data);
        if (filled > buf_size)
            return MP3D_E_IOERROR;
        readed += id3v2size;
    } else
    {
        readed = io->read(buf + MINIMP3_ID3_DETECT_SIZE, buf_size - MINIMP3_ID3_DETECT_SIZE, io->read_data);
        if (readed > (buf_size - MINIMP3_ID3_DETECT_SIZE))
            return MP3D_E_IOERROR;
        filled += readed;
    }
    if (filled < MINIMP3_BUF_SIZE)
        mp3dec_skip_id3v1(buf, &filled);
    do
    {
        int free_format_bytes = 0, frame_size = 0, ret;
        int i = mp3d_find_frame(buf + consumed, filled - consumed, &free_format_bytes, &frame_size);
        if (i && !frame_size)
        {
            consumed += i;
            continue;
        }
        if (!frame_size)
            break;
        const uint8_t *hdr = buf + consumed + i;
        frame_info.channels = HDR_IS_MONO(hdr) ? 1 : 2;
        frame_info.hz = hdr_sample_rate_hz(hdr);
        frame_info.layer = 4 - HDR_GET_LAYER(hdr);
        frame_info.bitrate_kbps = hdr_bitrate_kbps(hdr);
        frame_info.frame_bytes = frame_size;

        readed += i;
        if (callback)
        {
            ret = callback(user_data, hdr, frame_size, free_format_bytes, filled - consumed, readed, &frame_info);
            if (ret != 0)
                return ret;
        }
        readed += frame_size;
        consumed += i + frame_size;
        if (!eof && filled - consumed < MINIMP3_BUF_SIZE)
        {   /* keep minimum 10 consecutive mp3 frames (~16KB) worst case */
            memmove(buf, buf + consumed, filled - consumed);
            filled -= consumed;
            consumed = 0;
            readed = io->read(buf + filled, buf_size - filled, io->read_data);
            if (readed > (buf_size - filled))
                return MP3D_E_IOERROR;
            if (readed != (buf_size - filled))
                eof = 1;
            filled += readed;
            if (eof)
                mp3dec_skip_id3v1(buf, &filled);
        }
    } while (1);
    return 0;
}

static int mp3dec_load_index(void *user_data, const uint8_t *frame, int frame_size, int free_format_bytes, size_t buf_size, uint64_t offset, mp3dec_frame_info_t *info)
{
    mp3dec_frame_t *idx_frame;
    mp3dec_ex_t *dec = (mp3dec_ex_t *)user_data;
    if (!dec->index.frames && !dec->start_offset)
    {   /* detect VBR tag and try to avoid full scan */
        uint32_t frames;
        int delay, padding;
        dec->info = *info;
        dec->start_offset = dec->offset = offset;
        dec->end_offset   = offset + buf_size;
        dec->free_format_bytes = free_format_bytes; /* should not change */
        if (3 == dec->info.layer)
        {
            int ret = mp3dec_check_vbrtag(frame, frame_size, &frames, &delay, &padding);
            if (ret)
                dec->start_offset = dec->offset = offset + frame_size;
            if (ret > 0)
            {
                padding *= info->channels;
                dec->start_delay = dec->to_skip = delay*info->channels;
                dec->samples = hdr_frame_samples(frame)*info->channels*(uint64_t)frames;
                if (dec->samples >= (uint64_t)dec->start_delay)
                    dec->samples -= dec->start_delay;
                if (padding > 0 && dec->samples >= (uint64_t)padding)
                    dec->samples -= padding;
                dec->detected_samples = dec->samples;
                dec->vbr_tag_found = 1;
                return MP3D_E_USER;
            } else if (ret < 0)
                return 0;
        }
    }
    if (dec->flags & MP3D_DO_NOT_SCAN)
        return MP3D_E_USER;
    if (dec->index.num_frames + 1 > dec->index.capacity)
    {
        if (!dec->index.capacity)
            dec->index.capacity = 4096;
        else
            dec->index.capacity *= 2;
        mp3dec_frame_t *alloc_buf = (mp3dec_frame_t *)realloc((void*)dec->index.frames, sizeof(mp3dec_frame_t)*dec->index.capacity);
        if (!alloc_buf)
            return MP3D_E_MEMORY;
        dec->index.frames = alloc_buf;
    }
    idx_frame = &dec->index.frames[dec->index.num_frames++];
    idx_frame->offset = offset;
    idx_frame->sample = dec->samples;
    if (!dec->buffer_samples && dec->index.num_frames < 256)
    {   /* for some cutted mp3 frames, bit-reservoir not filled and decoding can't be started from first frames */
        /* try to decode up to 255 first frames till samples starts to decode */
        dec->buffer_samples = mp3dec_decode_frame(&dec->mp3d, frame, MINIMP3_MIN(buf_size, (size_t)INT_MAX), dec->buffer, info);
        dec->samples += dec->buffer_samples*info->channels;
    } else
        dec->samples += hdr_frame_samples(frame)*info->channels;
    return 0;
}

int mp3dec_ex_open_buf(mp3dec_ex_t *dec, const uint8_t *buf, size_t buf_size, int flags)
{
    if (!dec || !buf || (size_t)-1 == buf_size || (flags & (~MP3D_FLAGS_MASK)))
        return MP3D_E_PARAM;
    memset(dec, 0, sizeof(*dec));
    dec->file.buffer = buf;
    dec->file.size   = buf_size;
    dec->flags       = flags;
    mp3dec_init(&dec->mp3d);
    int ret = mp3dec_iterate_buf(dec->file.buffer, dec->file.size, mp3dec_load_index, dec);
    if (ret && MP3D_E_USER != ret)
        return ret;
    mp3dec_init(&dec->mp3d);
    dec->buffer_samples = 0;
    dec->indexes_built = !(dec->vbr_tag_found || (flags & MP3D_DO_NOT_SCAN));
    dec->flags &= (~MP3D_DO_NOT_SCAN);
    return 0;
}

#ifndef MINIMP3_SEEK_IDX_LINEAR_SEARCH
static size_t mp3dec_idx_binary_search(mp3dec_index_t *idx, uint64_t position)
{
    size_t end = idx->num_frames, start = 0, index = 0;
    while (start <= end)
    {
        size_t mid = (start + end) / 2;
        if (idx->frames[mid].sample >= position)
        {   /* move left side. */
            if (idx->frames[mid].sample == position)
                return mid;
            end = mid - 1;
        }  else
        {   /* move to right side */
            index = mid;
            start = mid + 1;
            if (start == idx->num_frames)
                break;
        }
    }
    return index;
}
#endif

int mp3dec_ex_seek(mp3dec_ex_t *dec, uint64_t position)
{
    size_t i;
    if (!dec)
        return MP3D_E_PARAM;
    if (!(dec->flags & MP3D_SEEK_TO_SAMPLE))
    {
        if (dec->io)
        {
            dec->offset = position;
        } else
        {
            dec->offset = MINIMP3_MIN(position, dec->file.size);
        }
        dec->cur_sample = 0;
        goto do_exit;
    }
    dec->cur_sample = position;
    position += dec->start_delay;
    if (0 == position)
    {   /* optimize seek to zero, no index needed */
seek_zero:
        dec->offset  = dec->start_offset;
        dec->to_skip = 0;
        goto do_exit;
    }
    if (!dec->indexes_built)
    {   /* no index created yet (vbr tag used to calculate track length or MP3D_DO_NOT_SCAN open flag used) */
        dec->indexes_built = 1;
        dec->samples = 0;
        dec->buffer_samples = 0;
        if (dec->io)
        {
            if (dec->io->seek(dec->start_offset, dec->io->seek_data))
                return MP3D_E_IOERROR;
            int ret = mp3dec_iterate_cb(dec->io, (uint8_t *)dec->file.buffer, dec->file.size, mp3dec_load_index, dec);
            if (ret && MP3D_E_USER != ret)
                return ret;
        } else
        {
            int ret = mp3dec_iterate_buf(dec->file.buffer + dec->start_offset, dec->file.size - dec->start_offset, mp3dec_load_index, dec);
            if (ret && MP3D_E_USER != ret)
                return ret;
        }
        for (i = 0; i < dec->index.num_frames; i++)
            dec->index.frames[i].offset += dec->start_offset;
        dec->samples = dec->detected_samples;
    }
    if (!dec->index.frames)
        goto seek_zero; /* no frames in file - seek to zero */
#ifdef MINIMP3_SEEK_IDX_LINEAR_SEARCH
    for (i = 0; i < dec->index.num_frames; i++)
    {
        if (dec->index.frames[i].sample >= position)
            break;
    }
#else
    i = mp3dec_idx_binary_search(&dec->index, position);
#endif
    if (i)
    {
        int to_fill_bytes = 511;
        int skip_frames = MINIMP3_PREDECODE_FRAMES
#ifdef MINIMP3_SEEK_IDX_LINEAR_SEARCH
         + ((dec->index.frames[i].sample == position) ? 0 : 1)
#endif
        ;
        i -= MINIMP3_MIN(i, (size_t)skip_frames);
        if (3 == dec->info.layer)
        {
            while (i && to_fill_bytes)
            {   /* make sure bit-reservoir is filled when we start decoding */
                bs_t bs[1];
                L3_gr_info_t gr_info[4];
                int frame_bytes, frame_size;
                const uint8_t *hdr;
                if (dec->io)
                {
                    hdr = dec->file.buffer;
                    if (dec->io->seek(dec->index.frames[i - 1].offset, dec->io->seek_data))
                        return MP3D_E_IOERROR;
                    size_t readed = dec->io->read((uint8_t *)hdr, HDR_SIZE, dec->io->read_data);
                    if (readed != HDR_SIZE)
                        return MP3D_E_IOERROR;
                    frame_size = hdr_frame_bytes(hdr, dec->free_format_bytes) + hdr_padding(hdr);
                    readed = dec->io->read((uint8_t *)hdr + HDR_SIZE, frame_size - HDR_SIZE, dec->io->read_data);
                    if (readed != (size_t)(frame_size - HDR_SIZE))
                        return MP3D_E_IOERROR;
                    bs_init(bs, hdr + HDR_SIZE, frame_size - HDR_SIZE);
                } else
                {
                    hdr = dec->file.buffer + dec->index.frames[i - 1].offset;
                    frame_size = hdr_frame_bytes(hdr, dec->free_format_bytes) + hdr_padding(hdr);
                    bs_init(bs, hdr + HDR_SIZE, frame_size - HDR_SIZE);
                }
                if (HDR_IS_CRC(hdr))
                    get_bits(bs, 16);
                i--;
                if (L3_read_side_info(bs, gr_info, hdr) < 0)
                    break; /* frame not decodable, we can start from here */
                frame_bytes = (bs->limit - bs->pos)/8;
                to_fill_bytes -= MINIMP3_MIN(to_fill_bytes, frame_bytes);
            }
        }
    }
    dec->offset = dec->index.frames[i].offset;
    dec->to_skip = position - dec->index.frames[i].sample;
    while ((i + 1) < dec->index.num_frames && !dec->index.frames[i].sample && !dec->index.frames[i + 1].sample)
    {   /* skip not decodable first frames */
        const uint8_t *hdr;
        if (dec->io)
        {
            hdr = dec->file.buffer;
            if (dec->io->seek(dec->index.frames[i].offset, dec->io->seek_data))
                return MP3D_E_IOERROR;
            size_t readed = dec->io->read((uint8_t *)hdr, HDR_SIZE, dec->io->read_data);
            if (readed != HDR_SIZE)
                return MP3D_E_IOERROR;
        } else
            hdr = dec->file.buffer + dec->index.frames[i].offset;
        dec->to_skip += hdr_frame_samples(hdr)*dec->info.channels;
        i++;
    }
do_exit:
    if (dec->io)
    {
        if (dec->io->seek(dec->offset, dec->io->seek_data))
            return MP3D_E_IOERROR;
    }
    dec->buffer_samples  = 0;
    dec->buffer_consumed = 0;
    dec->input_consumed  = 0;
    dec->input_filled    = 0;
    dec->last_error      = 0;
    mp3dec_init(&dec->mp3d);
    return 0;
}

size_t mp3dec_ex_read_frame(mp3dec_ex_t *dec, mp3d_sample_t **buf, mp3dec_frame_info_t *frame_info, size_t max_samples)
{
    if (!dec || !buf || !frame_info)
    {
        if (dec)
            dec->last_error = MP3D_E_PARAM;
        return 0;
    }
    if (dec->detected_samples && dec->cur_sample >= dec->detected_samples)
        return 0; /* at end of stream */
    if (dec->last_error)
        return 0; /* error eof state, seek can reset it */
    *buf = NULL;
    uint64_t end_offset = dec->end_offset ? dec->end_offset : dec->file.size;
    int eof = 0;
    while (dec->buffer_consumed == dec->buffer_samples)
    {
        const uint8_t *dec_buf;
        if (dec->io)
        {
            if (!eof && (dec->input_filled - dec->input_consumed) < MINIMP3_BUF_SIZE)
            {   /* keep minimum 10 consecutive mp3 frames (~16KB) worst case */
                memmove((uint8_t*)dec->file.buffer, (uint8_t*)dec->file.buffer + dec->input_consumed, dec->input_filled - dec->input_consumed);
                dec->input_filled -= dec->input_consumed;
                dec->input_consumed = 0;
                size_t readed = dec->io->read((uint8_t*)dec->file.buffer + dec->input_filled, dec->file.size - dec->input_filled, dec->io->read_data);
                if (readed > (dec->file.size - dec->input_filled))
                {
                    dec->last_error = MP3D_E_IOERROR;
                    readed = 0;
                }
                if (readed != (dec->file.size - dec->input_filled))
                    eof = 1;
                dec->input_filled += readed;
                if (eof)
                    mp3dec_skip_id3v1((uint8_t*)dec->file.buffer, &dec->input_filled);
            }
            dec_buf = dec->file.buffer + dec->input_consumed;
            if (!(dec->input_filled - dec->input_consumed))
                return 0;
            dec->buffer_samples = mp3dec_decode_frame(&dec->mp3d, dec_buf, dec->input_filled - dec->input_consumed, dec->buffer, frame_info);
            dec->input_consumed += frame_info->frame_bytes;
        } else
        {
            dec_buf = dec->file.buffer + dec->offset;
            uint64_t buf_size = end_offset - dec->offset;
            if (!buf_size)
                return 0;
            dec->buffer_samples = mp3dec_decode_frame(&dec->mp3d, dec_buf, MINIMP3_MIN(buf_size, (uint64_t)INT_MAX), dec->buffer, frame_info);
        }
        dec->buffer_consumed = 0;
        if (dec->info.hz != frame_info->hz || dec->info.layer != frame_info->layer)
        {
return_e_decode:
            dec->last_error = MP3D_E_DECODE;
            return 0;
        }
        if (dec->buffer_samples)
        {
            dec->buffer_samples *= frame_info->channels;
            if (dec->to_skip)
            {
                size_t skip = MINIMP3_MIN(dec->buffer_samples, dec->to_skip);
                dec->buffer_consumed += skip;
                dec->to_skip -= skip;
            }
            if (
#ifdef MINIMP3_ALLOW_MONO_STEREO_TRANSITION
                !(dec->flags & MP3D_ALLOW_MONO_STEREO_TRANSITION) &&
#endif
                dec->buffer_consumed != dec->buffer_samples && dec->info.channels != frame_info->channels)
            {
                goto return_e_decode;
            }
        } else if (dec->to_skip)
        {   /* In mp3 decoding not always can start decode from any frame because of bit reservoir,
               count skip samples for such frames */
            int frame_samples = hdr_frame_samples(dec_buf)*frame_info->channels;
            dec->to_skip -= MINIMP3_MIN(frame_samples, dec->to_skip);
        }
        dec->offset += frame_info->frame_bytes;
    }
    size_t out_samples = MINIMP3_MIN((size_t)(dec->buffer_samples - dec->buffer_consumed), max_samples);
    if (dec->detected_samples)
    {   /* count decoded samples to properly cut padding */
        if (dec->cur_sample + out_samples >= dec->detected_samples)
            out_samples = dec->detected_samples - dec->cur_sample;
    }
    dec->cur_sample += out_samples;
    *buf = dec->buffer + dec->buffer_consumed;
    dec->buffer_consumed += out_samples;
    return out_samples;
}

size_t mp3dec_ex_read(mp3dec_ex_t *dec, mp3d_sample_t *buf, size_t samples)
{
    if (!dec || !buf)
    {
        if (dec)
            dec->last_error = MP3D_E_PARAM;
        return 0;
    }
    mp3dec_frame_info_t frame_info;
    memset(&frame_info, 0, sizeof(frame_info));
    size_t samples_requested = samples;
    while (samples)
    {
        mp3d_sample_t *buf_frame = NULL;
        size_t read_samples = mp3dec_ex_read_frame(dec, &buf_frame, &frame_info, samples);
        if (!read_samples)
        {
            break;
        }
        memcpy(buf, buf_frame, read_samples * sizeof(mp3d_sample_t));
        buf += read_samples;
        samples -= read_samples;
    }
    return samples_requested - samples;
}

int mp3dec_ex_open_cb(mp3dec_ex_t *dec, mp3dec_io_t *io, int flags)
{
    if (!dec || !io || (flags & (~MP3D_FLAGS_MASK)))
        return MP3D_E_PARAM;
    memset(dec, 0, sizeof(*dec));
#ifdef MINIMP3_HAVE_RING
    int ret;
    if (ret = mp3dec_open_ring(&dec->file, MINIMP3_IO_SIZE))
        return ret;
#else
    dec->file.size = MINIMP3_IO_SIZE;
    dec->file.buffer = (const uint8_t*)malloc(dec->file.size);
    if (!dec->file.buffer)
        return MP3D_E_MEMORY;
#endif
    dec->flags = flags;
    dec->io = io;
    mp3dec_init(&dec->mp3d);
    if (io->seek(0, io->seek_data))
        return MP3D_E_IOERROR;
    int ret = mp3dec_iterate_cb(io, (uint8_t *)dec->file.buffer, dec->file.size, mp3dec_load_index, dec);
    if (ret && MP3D_E_USER != ret)
        return ret;
    if (dec->io->seek(dec->start_offset, dec->io->seek_data))
        return MP3D_E_IOERROR;
    mp3dec_init(&dec->mp3d);
    dec->buffer_samples = 0;
    dec->indexes_built = !(dec->vbr_tag_found || (flags & MP3D_DO_NOT_SCAN));
    dec->flags &= (~MP3D_DO_NOT_SCAN);
    return 0;
}


#ifndef MINIMP3_NO_STDIO

#if defined(__linux__) || defined(__FreeBSD__)
#include <errno.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#if !defined(_GNU_SOURCE)
#include <sys/ipc.h>
#include <sys/shm.h>
#endif
#if !defined(MAP_POPULATE) && defined(__linux__)
#define MAP_POPULATE 0x08000
#elif !defined(MAP_POPULATE)
#define MAP_POPULATE 0
#endif

static void mp3dec_close_file(mp3dec_map_info_t *map_info)
{
    if (map_info->buffer && MAP_FAILED != map_info->buffer)
        munmap((void *)map_info->buffer, map_info->size);
    map_info->buffer = 0;
    map_info->size   = 0;
}

static int mp3dec_open_file(const char *file_name, mp3dec_map_info_t *map_info)
{
    if (!file_name)
        return MP3D_E_PARAM;
    int file;
    struct stat st;
    memset(map_info, 0, sizeof(*map_info));
retry_open:
    file = open(file_name, O_RDONLY);
    if (file < 0 && (errno == EAGAIN || errno == EINTR))
        goto retry_open;
    if (file < 0 || fstat(file, &st) < 0)
    {
        close(file);
        return MP3D_E_IOERROR;
    }

    map_info->size = st.st_size;
retry_mmap:
    map_info->buffer = (const uint8_t *)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, file, 0);
    if (MAP_FAILED == map_info->buffer && (errno == EAGAIN || errno == EINTR))
        goto retry_mmap;
    close(file);
    if (MAP_FAILED == map_info->buffer)
        return MP3D_E_IOERROR;
    return 0;
}

#if MINIMP3_ENABLE_RING && defined(__linux__) && defined(_GNU_SOURCE)
#define MINIMP3_HAVE_RING
static void mp3dec_close_ring(mp3dec_map_info_t *map_info)
{
#if defined(__linux__) && defined(_GNU_SOURCE)
    if (map_info->buffer && MAP_FAILED != map_info->buffer)
        munmap((void *)map_info->buffer, map_info->size*2);
#else
    if (map_info->buffer)
    {
        shmdt(map_info->buffer);
        shmdt(map_info->buffer + map_info->size);
    }
#endif
    map_info->buffer = 0;
    map_info->size   = 0;
}

static int mp3dec_open_ring(mp3dec_map_info_t *map_info, size_t size)
{
    int memfd, page_size;
#if defined(__linux__) && defined(_GNU_SOURCE)
    void *buffer;
    int res;
#endif
    memset(map_info, 0, sizeof(*map_info));

#ifdef _SC_PAGESIZE
    page_size = sysconf(_SC_PAGESIZE);
#else
    page_size = getpagesize();
#endif
    map_info->size = (size + page_size - 1)/page_size*page_size;

#if defined(__linux__) && defined(_GNU_SOURCE)
    memfd = memfd_create("mp3_ring", 0);
    if (memfd < 0)
        return MP3D_E_MEMORY;

retry_ftruncate:
    res = ftruncate(memfd, map_info->size);
    if (res && (errno == EAGAIN || errno == EINTR))
        goto retry_ftruncate;
    if (res)
        goto error;

retry_mmap:
    map_info->buffer = (const uint8_t *)mmap(NULL, map_info->size*2, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == map_info->buffer && (errno == EAGAIN || errno == EINTR))
        goto retry_mmap;
    if (MAP_FAILED == map_info->buffer || !map_info->buffer)
        goto error;
retry_mmap2:
    buffer = mmap((void *)map_info->buffer, map_info->size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, memfd, 0);
    if (MAP_FAILED == map_info->buffer && (errno == EAGAIN || errno == EINTR))
        goto retry_mmap2;
    if (MAP_FAILED == map_info->buffer || buffer != (void *)map_info->buffer)
        goto error;
retry_mmap3:
    buffer = mmap((void *)map_info->buffer + map_info->size, map_info->size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, memfd, 0);
    if (MAP_FAILED == map_info->buffer && (errno == EAGAIN || errno == EINTR))
        goto retry_mmap3;
    if (MAP_FAILED == map_info->buffer || buffer != (void *)(map_info->buffer + map_info->size))
        goto error;

    close(memfd);
    return 0;
error:
    close(memfd);
    mp3dec_close_ring(map_info);
    return MP3D_E_MEMORY;
#else
    memfd = shmget(IPC_PRIVATE, map_info->size, IPC_CREAT | 0700);
    if (memfd < 0)
        return MP3D_E_MEMORY;
retry_mmap:
    map_info->buffer = (const uint8_t *)mmap(NULL, map_info->size*2, PROT_NONE, MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == map_info->buffer && (errno == EAGAIN || errno == EINTR))
        goto retry_mmap;
    if (MAP_FAILED == map_info->buffer)
        goto error;
    if (map_info->buffer != shmat(memfd, map_info->buffer, 0))
        goto error;
    if ((map_info->buffer + map_info->size) != shmat(memfd, map_info->buffer + map_info->size, 0))
        goto error;
    if (shmctl(memfd, IPC_RMID, NULL) < 0)
        return MP3D_E_MEMORY;
    return 0;
error:
    shmctl(memfd, IPC_RMID, NULL);
    mp3dec_close_ring(map_info);
    return MP3D_E_MEMORY;
#endif
}
#endif /*MINIMP3_ENABLE_RING*/
#elif defined(_WIN32)
#include <windows.h>

static void mp3dec_close_file(mp3dec_map_info_t *map_info)
{
    if (map_info->buffer)
        UnmapViewOfFile(map_info->buffer);
    map_info->buffer = 0;
    map_info->size   = 0;
}

static int mp3dec_open_file_h(HANDLE file, mp3dec_map_info_t *map_info)
{
    memset(map_info, 0, sizeof(*map_info));

    HANDLE mapping = NULL;
    LARGE_INTEGER s;
    s.LowPart = GetFileSize(file, (DWORD*)&s.HighPart);
    if (s.LowPart == INVALID_FILE_SIZE && GetLastError() != NO_ERROR)
        goto error;
    map_info->size = s.QuadPart;

    mapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping)
        goto error;
    map_info->buffer = (const uint8_t*)MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, s.QuadPart);
    CloseHandle(mapping);
    if (!map_info->buffer)
        goto error;

    CloseHandle(file);
    return 0;
error:
    mp3dec_close_file(map_info);
    CloseHandle(file);
    return MP3D_E_IOERROR;
}

static int mp3dec_open_file(const char *file_name, mp3dec_map_info_t *map_info)
{
    if (!file_name)
        return MP3D_E_PARAM;
    HANDLE file = CreateFileA(file_name, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    if (INVALID_HANDLE_VALUE == file)
        return MP3D_E_IOERROR;
    return mp3dec_open_file_h(file, map_info);
}

static int mp3dec_open_file_w(const wchar_t *file_name, mp3dec_map_info_t *map_info)
{
    if (!file_name)
        return MP3D_E_PARAM;
    HANDLE file = CreateFileW(file_name, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, 0, 0);
    if (INVALID_HANDLE_VALUE == file)
        return MP3D_E_IOERROR;
    return mp3dec_open_file_h(file, map_info);
}
#else
#include <stdio.h>

static void mp3dec_close_file(mp3dec_map_info_t *map_info)
{
    if (map_info->buffer)
        free((void *)map_info->buffer);
    map_info->buffer = 0;
    map_info->size = 0;
}

static int mp3dec_open_file(const char *file_name, mp3dec_map_info_t *map_info)
{
    if (!file_name)
        return MP3D_E_PARAM;
    memset(map_info, 0, sizeof(*map_info));
    FILE *file = fopen(file_name, "rb");
    if (!file)
        return MP3D_E_IOERROR;
    int res = MP3D_E_IOERROR;
    long size = -1;
    if (fseek(file, 0, SEEK_END))
        goto error;
    size = ftell(file);
    if (size < 0)
        goto error;
    map_info->size = (size_t)size;
    if (fseek(file, 0, SEEK_SET))
        goto error;
    map_info->buffer = (uint8_t *)malloc(map_info->size);
    if (!map_info->buffer)
    {
        res = MP3D_E_MEMORY;
        goto error;
    }
    if (fread((void *)map_info->buffer, 1, map_info->size, file) != map_info->size)
        goto error;
    fclose(file);
    return 0;
error:
    mp3dec_close_file(map_info);
    fclose(file);
    return res;
}
#endif

static int mp3dec_detect_mapinfo(mp3dec_map_info_t *map_info)
{
    int ret = mp3dec_detect_buf(map_info->buffer, map_info->size);
    mp3dec_close_file(map_info);
    return ret;
}

static int mp3dec_load_mapinfo(mp3dec_t *dec, mp3dec_map_info_t *map_info, mp3dec_file_info_t *info, MP3D_PROGRESS_CB progress_cb, void *user_data)
{
    int ret = mp3dec_load_buf(dec, map_info->buffer, map_info->size, info, progress_cb, user_data);
    mp3dec_close_file(map_info);
    return ret;
}

static int mp3dec_iterate_mapinfo(mp3dec_map_info_t *map_info, MP3D_ITERATE_CB callback, void *user_data)
{
    int ret = mp3dec_iterate_buf(map_info->buffer, map_info->size, callback, user_data);
    mp3dec_close_file(map_info);
    return ret;
}

static int mp3dec_ex_open_mapinfo(mp3dec_ex_t *dec, int flags)
{
    int ret = mp3dec_ex_open_buf(dec, dec->file.buffer, dec->file.size, flags);
    dec->is_file = 1;
    if (ret)
        mp3dec_ex_close(dec);
    return ret;
}

int mp3dec_detect(const char *file_name)
{
    int ret;
    mp3dec_map_info_t map_info;
    if ((ret = mp3dec_open_file(file_name, &map_info)))
        return ret;
    return mp3dec_detect_mapinfo(&map_info);
}

int mp3dec_load(mp3dec_t *dec, const char *file_name, mp3dec_file_info_t *info, MP3D_PROGRESS_CB progress_cb, void *user_data)
{
    int ret;
    mp3dec_map_info_t map_info;
    if ((ret = mp3dec_open_file(file_name, &map_info)))
        return ret;
    return mp3dec_load_mapinfo(dec, &map_info, info, progress_cb, user_data);
}

int mp3dec_iterate(const char *file_name, MP3D_ITERATE_CB callback, void *user_data)
{
    int ret;
    mp3dec_map_info_t map_info;
    if ((ret = mp3dec_open_file(file_name, &map_info)))
        return ret;
    return mp3dec_iterate_mapinfo(&map_info, callback, user_data);
}

int mp3dec_ex_open(mp3dec_ex_t *dec, const char *file_name, int flags)
{
    int ret;
    if (!dec)
        return MP3D_E_PARAM;
    if ((ret = mp3dec_open_file(file_name, &dec->file)))
        return ret;
    return mp3dec_ex_open_mapinfo(dec, flags);
}

void mp3dec_ex_close(mp3dec_ex_t *dec)
{
#ifdef MINIMP3_HAVE_RING
    if (dec->io)
        mp3dec_close_ring(&dec->file);
#else
    if (dec->io && dec->file.buffer)
        free((void*)dec->file.buffer);
#endif
    if (dec->is_file)
        mp3dec_close_file(&dec->file);
    if (dec->index.frames)
        free(dec->index.frames);
    memset(dec, 0, sizeof(*dec));
}

#ifdef _WIN32
int mp3dec_detect_w(const wchar_t *file_name)
{
    int ret;
    mp3dec_map_info_t map_info;
    if ((ret = mp3dec_open_file_w(file_name, &map_info)))
        return ret;
    return mp3dec_detect_mapinfo(&map_info);
}

int mp3dec_load_w(mp3dec_t *dec, const wchar_t *file_name, mp3dec_file_info_t *info, MP3D_PROGRESS_CB progress_cb, void *user_data)
{
    int ret;
    mp3dec_map_info_t map_info;
    if ((ret = mp3dec_open_file_w(file_name, &map_info)))
        return ret;
    return mp3dec_load_mapinfo(dec, &map_info, info, progress_cb, user_data);
}

int mp3dec_iterate_w(const wchar_t *file_name, MP3D_ITERATE_CB callback, void *user_data)
{
    int ret;
    mp3dec_map_info_t map_info;
    if ((ret = mp3dec_open_file_w(file_name, &map_info)))
        return ret;
    return mp3dec_iterate_mapinfo(&map_info, callback, user_data);
}

int mp3dec_ex_open_w(mp3dec_ex_t *dec, const wchar_t *file_name, int flags)
{
    int ret;
    if ((ret = mp3dec_open_file_w(file_name, &dec->file)))
        return ret;
    return mp3dec_ex_open_mapinfo(dec, flags);
}
#endif
#else /* MINIMP3_NO_STDIO */
void mp3dec_ex_close(mp3dec_ex_t *dec)
{
#ifdef MINIMP3_HAVE_RING
    if (dec->io)
        mp3dec_close_ring(&dec->file);
#else
    if (dec->io && dec->file.buffer)
        free((void*)dec->file.buffer);
#endif
    if (dec->index.frames)
        free(dec->index.frames);
    memset(dec, 0, sizeof(*dec));
}
#endif

#endif /* MINIMP3_IMPLEMENTATION && !_MINIMP3_EX_IMPLEMENTATION_GUARD */
