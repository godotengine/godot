/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#include "SDL_audioqueue.h"
#include "SDL_sysaudio.h"

typedef struct SDL_MemoryPool SDL_MemoryPool;

struct SDL_MemoryPool
{
    void *free_blocks;
    size_t block_size;
    size_t num_free;
    size_t max_free;
};

struct SDL_AudioTrack
{
    SDL_AudioSpec spec;
    int *chmap;
    bool flushed;
    SDL_AudioTrack *next;

    void *userdata;
    SDL_ReleaseAudioBufferCallback callback;

    Uint8 *data;
    size_t head;
    size_t tail;
    size_t capacity;

    int chmap_storage[SDL_MAX_CHANNELMAP_CHANNELS];  // !!! FIXME: this needs to grow if SDL ever supports more channels. But if it grows, we should probably be more clever about allocations.
};

struct SDL_AudioQueue
{
    SDL_AudioTrack *head;
    SDL_AudioTrack *tail;

    Uint8 *history_buffer;
    size_t history_length;
    size_t history_capacity;

    SDL_MemoryPool track_pool;
    SDL_MemoryPool chunk_pool;
};

// Allocate a new block, avoiding checking for ones already in the pool
static void *AllocNewMemoryPoolBlock(const SDL_MemoryPool *pool)
{
    return SDL_malloc(pool->block_size);
}

// Allocate a new block, first checking if there are any in the pool
static void *AllocMemoryPoolBlock(SDL_MemoryPool *pool)
{
    if (pool->num_free == 0) {
        return AllocNewMemoryPoolBlock(pool);
    }

    void *block = pool->free_blocks;
    pool->free_blocks = *(void **)block;
    --pool->num_free;
    return block;
}

// Free a block, or add it to the pool if there's room
static void FreeMemoryPoolBlock(SDL_MemoryPool *pool, void *block)
{
    if (pool->num_free < pool->max_free) {
        *(void **)block = pool->free_blocks;
        pool->free_blocks = block;
        ++pool->num_free;
    } else {
        SDL_free(block);
    }
}

// Destroy a pool and all of its blocks
static void DestroyMemoryPool(SDL_MemoryPool *pool)
{
    void *block = pool->free_blocks;
    pool->free_blocks = NULL;
    pool->num_free = 0;

    while (block) {
        void *next = *(void **)block;
        SDL_free(block);
        block = next;
    }
}

// Keeping a list of free chunks reduces memory allocations,
// But also increases the amount of work to perform when freeing the track.
static void InitMemoryPool(SDL_MemoryPool *pool, size_t block_size, size_t max_free)
{
    SDL_zerop(pool);

    SDL_assert(block_size >= sizeof(void *));
    pool->block_size = block_size;
    pool->max_free = max_free;
}

// Allocates a number of blocks and adds them to the pool
static bool ReserveMemoryPoolBlocks(SDL_MemoryPool *pool, size_t num_blocks)
{
    for (; num_blocks; --num_blocks) {
        void *block = AllocNewMemoryPoolBlock(pool);

        if (block == NULL) {
            return false;
        }

        *(void **)block = pool->free_blocks;
        pool->free_blocks = block;
        ++pool->num_free;
    }

    return true;
}

void SDL_DestroyAudioQueue(SDL_AudioQueue *queue)
{
    SDL_ClearAudioQueue(queue);

    DestroyMemoryPool(&queue->track_pool);
    DestroyMemoryPool(&queue->chunk_pool);
    SDL_aligned_free(queue->history_buffer);

    SDL_free(queue);
}

SDL_AudioQueue *SDL_CreateAudioQueue(size_t chunk_size)
{
    SDL_AudioQueue *queue = (SDL_AudioQueue *)SDL_calloc(1, sizeof(*queue));

    if (!queue) {
        return NULL;
    }

    InitMemoryPool(&queue->track_pool, sizeof(SDL_AudioTrack), 8);
    InitMemoryPool(&queue->chunk_pool, chunk_size, 4);

    if (!ReserveMemoryPoolBlocks(&queue->track_pool, 2)) {
        SDL_DestroyAudioQueue(queue);
        return NULL;
    }

    return queue;
}

static void DestroyAudioTrack(SDL_AudioQueue *queue, SDL_AudioTrack *track)
{
    track->callback(track->userdata, track->data, (int)track->capacity);

    FreeMemoryPoolBlock(&queue->track_pool, track);
}

void SDL_ClearAudioQueue(SDL_AudioQueue *queue)
{
    SDL_AudioTrack *track = queue->head;

    queue->head = NULL;
    queue->tail = NULL;
    queue->history_length = 0;

    while (track) {
        SDL_AudioTrack *next = track->next;
        DestroyAudioTrack(queue, track);
        track = next;
    }
}

static void FlushAudioTrack(SDL_AudioTrack *track)
{
    track->flushed = true;
}

void SDL_FlushAudioQueue(SDL_AudioQueue *queue)
{
    SDL_AudioTrack *track = queue->tail;

    if (track) {
        FlushAudioTrack(track);
    }
}

void SDL_PopAudioQueueHead(SDL_AudioQueue *queue)
{
    SDL_AudioTrack *track = queue->head;

    for (;;) {
        bool flushed = track->flushed;

        SDL_AudioTrack *next = track->next;
        DestroyAudioTrack(queue, track);
        track = next;

        if (flushed) {
            break;
        }
    }

    queue->head = track;
    queue->history_length = 0;

    if (!track) {
        queue->tail = NULL;
    }
}

SDL_AudioTrack *SDL_CreateAudioTrack(
    SDL_AudioQueue *queue, const SDL_AudioSpec *spec, const int *chmap,
    Uint8 *data, size_t len, size_t capacity,
    SDL_ReleaseAudioBufferCallback callback, void *userdata)
{
    SDL_AudioTrack *track = (SDL_AudioTrack *)AllocMemoryPoolBlock(&queue->track_pool);

    if (!track) {
        return NULL;
    }

    SDL_zerop(track);

    if (chmap) {
        SDL_assert(SDL_arraysize(track->chmap_storage) >= spec->channels);
        SDL_memcpy(track->chmap_storage, chmap, sizeof (*chmap) * spec->channels);
        track->chmap = track->chmap_storage;
    }

    SDL_copyp(&track->spec, spec);

    track->userdata = userdata;
    track->callback = callback;
    track->data = data;
    track->head = 0;
    track->tail = len;
    track->capacity = capacity;

    return track;
}

static void SDLCALL FreeChunkedAudioBuffer(void *userdata, const void *buf, int len)
{
    SDL_AudioQueue *queue = (SDL_AudioQueue *)userdata;

    FreeMemoryPoolBlock(&queue->chunk_pool, (void *)buf);
}

static SDL_AudioTrack *CreateChunkedAudioTrack(SDL_AudioQueue *queue, const SDL_AudioSpec *spec, const int *chmap)
{
    Uint8 *chunk = (Uint8 *)AllocMemoryPoolBlock(&queue->chunk_pool);

    if (!chunk) {
        return NULL;
    }

    size_t capacity = queue->chunk_pool.block_size;
    capacity -= capacity % SDL_AUDIO_FRAMESIZE(*spec);

    SDL_AudioTrack *track = SDL_CreateAudioTrack(queue, spec, chmap, chunk, 0, capacity, FreeChunkedAudioBuffer, queue);

    if (!track) {
        FreeMemoryPoolBlock(&queue->chunk_pool, chunk);
        return NULL;
    }

    return track;
}

void SDL_AddTrackToAudioQueue(SDL_AudioQueue *queue, SDL_AudioTrack *track)
{
    SDL_AudioTrack *tail = queue->tail;

    if (tail) {
        // If the spec has changed, make sure to flush the previous track
        if (!SDL_AudioSpecsEqual(&tail->spec, &track->spec, tail->chmap, track->chmap)) {
            FlushAudioTrack(tail);
        }

        tail->next = track;
    } else {
        queue->head = track;
    }

    queue->tail = track;
}

static size_t WriteToAudioTrack(SDL_AudioTrack *track, const Uint8 *data, size_t len)
{
    if (track->flushed || track->tail >= track->capacity) {
        return 0;
    }

    len = SDL_min(len, track->capacity - track->tail);
    SDL_memcpy(&track->data[track->tail], data, len);
    track->tail += len;

    return len;
}

bool SDL_WriteToAudioQueue(SDL_AudioQueue *queue, const SDL_AudioSpec *spec, const int *chmap, const Uint8 *data, size_t len)
{
    if (len == 0) {
        return true;
    }

    SDL_AudioTrack *track = queue->tail;

    if (track) {
        if (!SDL_AudioSpecsEqual(&track->spec, spec, track->chmap, chmap)) {
            FlushAudioTrack(track);
        }
    } else {
        SDL_assert(!queue->head);
        track = CreateChunkedAudioTrack(queue, spec, chmap);

        if (!track) {
            return false;
        }

        queue->head = track;
        queue->tail = track;
    }

    for (;;) {
        const size_t written = WriteToAudioTrack(track, data, len);
        data += written;
        len -= written;

        if (len == 0) {
            break;
        }

        SDL_AudioTrack *new_track = CreateChunkedAudioTrack(queue, spec, chmap);

        if (!new_track) {
            return false;
        }

        track->next = new_track;
        queue->tail = new_track;
        track = new_track;
    }

    return true;
}

void *SDL_BeginAudioQueueIter(SDL_AudioQueue *queue)
{
    return queue->head;
}

size_t SDL_NextAudioQueueIter(SDL_AudioQueue *queue, void **inout_iter, SDL_AudioSpec *out_spec, int **out_chmap, bool *out_flushed)
{
    SDL_AudioTrack *iter = (SDL_AudioTrack *)(*inout_iter);
    SDL_assert(iter != NULL);

    SDL_copyp(out_spec, &iter->spec);
    *out_chmap = iter->chmap;

    bool flushed = false;
    size_t queued_bytes = 0;

    while (iter) {
        SDL_AudioTrack *track = iter;
        iter = iter->next;

        size_t avail = track->tail - track->head;

        if (avail >= SDL_SIZE_MAX - queued_bytes) {
            queued_bytes = SDL_SIZE_MAX;
            flushed = false;
            break;
        }

        queued_bytes += avail;
        flushed = track->flushed;

        if (flushed) {
            break;
        }
    }

    *inout_iter = iter;
    *out_flushed = flushed;

    return queued_bytes;
}

static const Uint8 *PeekIntoAudioQueuePast(SDL_AudioQueue *queue, Uint8 *data, size_t len)
{
    SDL_AudioTrack *track = queue->head;

    if (track->head >= len) {
        return &track->data[track->head - len];
    }

    size_t past = len - track->head;

    if (past > queue->history_length) {
        return NULL;
    }

    SDL_memcpy(data, &queue->history_buffer[queue->history_length - past], past);
    SDL_memcpy(&data[past], track->data, track->head);

    return data;
}

static void UpdateAudioQueueHistory(SDL_AudioQueue *queue,
                                    const Uint8 *data, size_t len)
{
    Uint8 *history_buffer = queue->history_buffer;
    size_t history_bytes = queue->history_length;

    if (len >= history_bytes) {
        SDL_memcpy(history_buffer, &data[len - history_bytes], history_bytes);
    } else {
        size_t preserve = history_bytes - len;
        SDL_memmove(history_buffer, &history_buffer[len], preserve);
        SDL_memcpy(&history_buffer[preserve], data, len);
    }
}

static const Uint8 *ReadFromAudioQueue(SDL_AudioQueue *queue, Uint8 *data, size_t len)
{
    SDL_AudioTrack *track = queue->head;

    if (track->tail - track->head >= len) {
        const Uint8 *ptr = &track->data[track->head];
        track->head += len;
        return ptr;
    }

    size_t total = 0;

    for (;;) {
        size_t avail = SDL_min(len - total, track->tail - track->head);
        SDL_memcpy(&data[total], &track->data[track->head], avail);
        track->head += avail;
        total += avail;

        if (total == len) {
            break;
        }

        if (track->flushed) {
            SDL_SetError("Reading past end of flushed track");
            return NULL;
        }

        SDL_AudioTrack *next = track->next;

        if (!next) {
            SDL_SetError("Reading past end of incomplete track");
            return NULL;
        }

        UpdateAudioQueueHistory(queue, track->data, track->tail);

        queue->head = next;
        DestroyAudioTrack(queue, track);
        track = next;
    }

    return data;
}

static const Uint8 *PeekIntoAudioQueueFuture(SDL_AudioQueue *queue, Uint8 *data, size_t len)
{
    SDL_AudioTrack *track = queue->head;

    if (track->tail - track->head >= len) {
        return &track->data[track->head];
    }

    size_t total = 0;

    for (;;) {
        size_t avail = SDL_min(len - total, track->tail - track->head);
        SDL_memcpy(&data[total], &track->data[track->head], avail);
        total += avail;

        if (total == len) {
            break;
        }

        if (track->flushed) {
            // If we have run out of data, fill the rest with silence.
            SDL_memset(&data[total], SDL_GetSilenceValueForFormat(track->spec.format), len - total);
            break;
        }

        track = track->next;

        if (!track) {
            SDL_SetError("Peeking past end of incomplete track");
            return NULL;
        }
    }

    return data;
}

const Uint8 *SDL_ReadFromAudioQueue(SDL_AudioQueue *queue,
                                    Uint8 *dst, SDL_AudioFormat dst_format, int dst_channels, const int *dst_map,
                                    int past_frames, int present_frames, int future_frames,
                                    Uint8 *scratch, float gain)
{
    SDL_AudioTrack *track = queue->head;

    if (!track) {
        return NULL;
    }

    SDL_AudioFormat src_format = track->spec.format;
    int src_channels = track->spec.channels;
    const int *src_map = track->chmap;

    size_t src_frame_size = SDL_AUDIO_BYTESIZE(src_format) * src_channels;
    size_t dst_frame_size = SDL_AUDIO_BYTESIZE(dst_format) * dst_channels;

    size_t src_past_bytes = past_frames * src_frame_size;
    size_t src_present_bytes = present_frames * src_frame_size;
    size_t src_future_bytes = future_frames * src_frame_size;

    size_t dst_past_bytes = past_frames * dst_frame_size;
    size_t dst_present_bytes = present_frames * dst_frame_size;
    size_t dst_future_bytes = future_frames * dst_frame_size;

    const bool convert = (src_format != dst_format) || (src_channels != dst_channels) || (gain != 1.0f);

    if (convert && !dst) {
        // The user didn't ask for the data to be copied, but we need to convert it, so store it in the scratch buffer
        dst = scratch;
    }

    // Can we get all of the data straight from this track?
    if ((track->head >= src_past_bytes) && ((track->tail - track->head) >= (src_present_bytes + src_future_bytes))) {
        const Uint8 *ptr = &track->data[track->head - src_past_bytes];
        track->head += src_present_bytes;

        // Do we still need to copy/convert the data?
        if (dst) {
            ConvertAudio(past_frames + present_frames + future_frames, ptr,
                         src_format, src_channels, src_map, dst, dst_format, dst_channels, dst_map, scratch, gain);
            ptr = dst;
        }

        return ptr;
    }

    if (!dst) {
        // The user didn't ask for the data to be copied, but we need to, so store it in the scratch buffer
        dst = scratch;
    } else if (!convert) {
        // We are only copying, not converting, so copy straight into the dst buffer
        scratch = dst;
    }

    Uint8 *ptr = dst;

    if (src_past_bytes) {
        ConvertAudio(past_frames, PeekIntoAudioQueuePast(queue, scratch, src_past_bytes), src_format, src_channels, src_map, dst, dst_format, dst_channels, dst_map, scratch, gain);
        dst += dst_past_bytes;
        scratch += dst_past_bytes;
    }

    if (src_present_bytes) {
        ConvertAudio(present_frames, ReadFromAudioQueue(queue, scratch, src_present_bytes), src_format, src_channels, src_map, dst, dst_format, dst_channels, dst_map, scratch, gain);
        dst += dst_present_bytes;
        scratch += dst_present_bytes;
    }

    if (src_future_bytes) {
        ConvertAudio(future_frames, PeekIntoAudioQueueFuture(queue, scratch, src_future_bytes), src_format, src_channels, src_map, dst, dst_format, dst_channels, dst_map, scratch, gain);
        dst += dst_future_bytes;
        scratch += dst_future_bytes;
    }

    return ptr;
}

size_t SDL_GetAudioQueueQueued(SDL_AudioQueue *queue)
{
    size_t total = 0;
    void *iter = SDL_BeginAudioQueueIter(queue);

    while (iter) {
        SDL_AudioSpec src_spec;
        int *src_chmap;
        bool flushed;

        size_t avail = SDL_NextAudioQueueIter(queue, &iter, &src_spec, &src_chmap, &flushed);

        if (avail >= SDL_SIZE_MAX - total) {
            total = SDL_SIZE_MAX;
            break;
        }

        total += avail;
    }

    return total;
}

bool SDL_ResetAudioQueueHistory(SDL_AudioQueue *queue, int num_frames)
{
    SDL_AudioTrack *track = queue->head;

    if (!track) {
        return false;
    }

    size_t length = num_frames * SDL_AUDIO_FRAMESIZE(track->spec);
    Uint8 *history_buffer = queue->history_buffer;

    if (queue->history_capacity < length) {
        history_buffer = (Uint8 *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), length);
        if (!history_buffer) {
            return false;
        }
        SDL_aligned_free(queue->history_buffer);
        queue->history_buffer = history_buffer;
        queue->history_capacity = length;
    }

    queue->history_length = length;
    SDL_memset(history_buffer, SDL_GetSilenceValueForFormat(track->spec.format), length);

    return true;
}
