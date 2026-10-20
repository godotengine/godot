// (C) 2005-2006 Oded Shimon
// This file is available under the MIT/X license, see COPYING

#ifndef LIBNUT_PRIV_H
#define LIBNUT_PRIV_H

//#define NDEBUG // disables asserts
//#define DEBUG
//#define TRACE

#ifdef DEBUG
#define debug_msg(...) fprintf(stderr, __VA_ARGS__)
#else
#define debug_msg(...)
#endif

#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define ID_STRING "nut/multimedia container"

#define      MAIN_STARTCODE (0x7A561F5F04ADULL + (((uint64_t)('N'<<8) + 'M')<<48))
#define    STREAM_STARTCODE (0x11405BF2F9DBULL + (((uint64_t)('N'<<8) + 'S')<<48))
#define SYNCPOINT_STARTCODE (0xE4ADEECA4569ULL + (((uint64_t)('N'<<8) + 'K')<<48))
#define     INDEX_STARTCODE (0xDD672F23E64EULL + (((uint64_t)('N'<<8) + 'X')<<48))
#define      INFO_STARTCODE (0xAB68B596BA78ULL + (((uint64_t)('N'<<8) + 'I')<<48))

#define NUT_API_FLAGS    3

#define FLAG_CODED_PTS   8
#define FLAG_STREAM_ID  16
#define FLAG_SIZE_MSB   32
#define FLAG_CHECKSUM   64
#define FLAG_RESERVED  128
#define FLAG_CODED    4096
#define FLAG_INVALID  8192

#define PREALLOC_SIZE 4096

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define ABS(a) ((a) > 0 ? (a) : -(a))

typedef struct {
	nut_input_stream_tt isc;
	int is_mem;
	uint8_t * buf;
	uint8_t * buf_ptr;
	int write_len; // allocated memory
	int read_len;  // data in memory
	off_t file_pos;
	off_t filesize;
	nut_alloc_tt * alloc;
} input_buffer_tt;

typedef struct {
	nut_output_stream_tt osc;
        int is_mem;
	uint8_t * buf;
	uint8_t * buf_ptr;
	int write_len; // allocated memory
	off_t file_pos;
	nut_alloc_tt * alloc;
} output_buffer_tt;

typedef struct {
	uint16_t flags;
	uint16_t mul;
	uint16_t lsb;
	int16_t pts_delta;
	uint8_t reserved;
	uint8_t stream;
} frame_table_tt;

typedef struct {
	off_t pos;
	uint64_t pts; // coded in '% timebase_count'
	int back_ptr:30;
	unsigned int seen_next:1;
	unsigned int pts_valid:1;
} syncpoint_tt;

typedef struct syncpoint_linked_s syncpoint_linked_tt;
struct syncpoint_linked_s {
	syncpoint_linked_tt * prev;
	syncpoint_tt s;
	uint64_t pts_eor[1];
};

typedef struct {
	int len;
	int alloc_len;
	syncpoint_tt * s;
	uint64_t * pts; // each elem is stream_count items, +1 to real pts, 0 means there is no key
	uint64_t * eor; // same as pts, is the pts of last eor in syncpoint region _IF_ eor is set by syncpoint.
	int cached_pos;
	syncpoint_linked_tt * linked; // entries are entered in reverse order for speed, points to END of list
} syncpoint_list_tt;

typedef struct {
	nut_packet_tt p;
	uint8_t * buf;
	int64_t dts;
} reorder_packet_tt;

typedef struct {
	int active;
	uint64_t pts; // requested pts;
	uint64_t old_last_pts;
	off_t good_key;
	int pts_higher; // for active streams
} seek_state_tt;

typedef struct {
	uint64_t last_key; // muxer.c, reset to 0 on every keyframe
	uint64_t last_pts;
	int64_t last_dts;
	int msb_pts_shift;
	int max_pts_distance;
	int timebase_id;
	nut_stream_header_tt sh;
	int64_t * pts_cache;
	int64_t eor;
	seek_state_tt state;
	// reorder.c
	int64_t next_pts;
	reorder_packet_tt * packets;
	int num_packets;
	int64_t * reorder_pts_cache;
	// debug stuff
	int overhead;
	int tot_size;
	int total_frames;
} stream_context_tt;

struct nut_context_s {
	nut_muxer_opts_tt mopts;
	nut_demuxer_opts_tt dopts;
	nut_alloc_tt * alloc;
	input_buffer_tt * i;
	output_buffer_tt * o;
	output_buffer_tt * tmp_buffer;
	output_buffer_tt * tmp_buffer2;

	int timebase_count;
	nut_timebase_tt * tb;

	int stream_count;
	stream_context_tt * sc;

	int info_count;
	nut_info_packet_tt * info;

	int max_distance;
	frame_table_tt ft[256];

	off_t last_syncpoint; // for checking corruption and putting syncpoints, also for back_ptr
	off_t last_headers; // for header repetition and state for demuxer
	int headers_written; // for muxer header repetition

	off_t before_seek; // position before any seek mess
	off_t seek_status;
	off_t binary_guess;
	double seek_time_pos;

	syncpoint_list_tt syncpoints;
	struct find_syncpoint_state_s {
		int i, begin, seeked;
		off_t pos;
	} find_syncpoint_state;

	// debug
	int sync_overhead;
};

static inline uint32_t crc32(uint8_t * buf, int len){
	static const uint32_t table[16] = {
		0x00000000, 0x04C11DB7, 0x09823B6E, 0x0D4326D9,
		0x130476DC, 0x17C56B6B, 0x1A864DB2, 0x1E475005,
		0x2608EDB8, 0x22C9F00F, 0x2F8AD6D6, 0x2B4BCB61,
		0x350C9B64, 0x31CD86D3, 0x3C8EA00A, 0x384FBDBD,
	};
	uint32_t crc = 0;
	while (len--) {
		crc ^= *buf++ << 24;
		crc = (crc<<4) ^ table[crc>>28];
		crc = (crc<<4) ^ table[crc>>28];
	}
	return crc;
}

static inline uint64_t convert_ts(uint64_t sn, nut_timebase_tt from, nut_timebase_tt to) {
	uint64_t ln, d1, d2;
	ln = (uint64_t)from.num * to.den;
	d1 = from.den;
	d2 = to.num;
	return (ln / d1 * sn + (ln%d1) * sn / d1) / d2;
}

static inline int compare_ts(uint64_t a, nut_timebase_tt at, uint64_t b, nut_timebase_tt bt) {
	if (convert_ts(a, at, bt) < b) return -1;
	if (convert_ts(b, bt, at) < a) return  1;
	return 0;
}

static inline int64_t get_dts(int d, int64_t * pts_cache, int64_t pts) {
	while (d--) {
		int64_t t = pts_cache[d];
		if (t < pts) {
			pts_cache[d] = pts;
			pts = t;
		}
	}
	return pts;
}

static inline int64_t peek_dts(int d, int64_t * pts_cache, int64_t pts) {
	while (d--) if (pts_cache[d] < pts) pts = pts_cache[d];
	return pts;
}

static inline int gcd(int a, int b) {
	while (b != 0) {
		int t = b;
		b = a % b;
		a = t;
	}
	return a;
}

#define bctello(bc) ((bc)->file_pos + ((bc)->buf_ptr - (bc)->buf))

#define TO_PTS(prefix, pts) \
	int prefix##_tb = (pts) % nut->timebase_count; \
	uint64_t prefix##_p = (pts) / nut->timebase_count;

#define TO_DOUBLE(t, pts) ((double)(pts) / nut->tb[t].den * nut->tb[t].num)

#define TO_DOUBLE_PTS(pts) TO_DOUBLE((pts) % nut->timebase_count, (pts) / nut->timebase_count)

#define TO_TB(i) nut->tb[nut->sc[i].timebase_id]

#endif // LIBNUT_PRIV_H
