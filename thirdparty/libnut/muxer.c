// (C) 2005-2006 Oded Shimon
// This file is available under the MIT/X license, see COPYING

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include "libnut.h"
#include "priv.h"

static int stream_write(void * priv, size_t len, const uint8_t * buf) {
	return fwrite(buf, 1, len, priv);
}

static void flush_buf(output_buffer_tt * bc) {
	assert(bc->osc.write);
	bc->file_pos += bc->osc.write(bc->osc.priv, bc->buf_ptr - bc->buf, bc->buf);
	bc->buf_ptr = bc->buf;
}

static void ready_write_buf(output_buffer_tt * bc, int amount) {
	if (bc->write_len - (bc->buf_ptr - bc->buf) > amount) return;

        if (bc->is_mem) {
		int tmp = bc->buf_ptr - bc->buf;
		bc->write_len = tmp + amount + PREALLOC_SIZE;
		bc->buf = bc->alloc->realloc(bc->buf, bc->write_len);
		bc->buf_ptr = bc->buf + tmp;
	} else {
		flush_buf(bc);
		if (bc->write_len < amount) {
			bc->alloc->free(bc->buf);
			bc->write_len = amount + PREALLOC_SIZE;
			bc->buf_ptr = bc->buf = bc->alloc->malloc(bc->write_len);
		}
	}
}

static output_buffer_tt * new_mem_buffer(nut_alloc_tt * alloc) {
	output_buffer_tt * bc = alloc->malloc(sizeof(output_buffer_tt));
	bc->alloc = alloc;
	bc->write_len = PREALLOC_SIZE;
	bc->is_mem = 1;
	bc->file_pos = 0;
	bc->buf_ptr = bc->buf = alloc->malloc(bc->write_len);
	bc->osc.write = NULL;
	return bc;
}

static output_buffer_tt * new_output_buffer(nut_alloc_tt * alloc, nut_output_stream_tt osc) {
	output_buffer_tt * bc = new_mem_buffer(alloc);
	bc->is_mem = 0;
	bc->osc = osc;
	if (!bc->osc.write) bc->osc.write = stream_write;
	return bc;
}

static void free_buffer(output_buffer_tt * bc) {
	if (!bc) return;
	if (!bc->is_mem) flush_buf(bc);
	bc->alloc->free(bc->buf);
	bc->alloc->free(bc);
}

static output_buffer_tt * clear_buffer(output_buffer_tt * bc) {
	assert(bc->is_mem);
	bc->buf_ptr = bc->buf;
	return bc;
}

static void put_bytes(output_buffer_tt * bc, int count, uint64_t val) {
	ready_write_buf(bc, count);
	for(count--; count >= 0; count--){
		*(bc->buf_ptr++) = val >> (8 * count);
	}
}

static int v_len(uint64_t val) {
	int i;
	val &= 0x7FFFFFFFFFFFFFFFULL;
	for(i=1; val>>(i*7); i++);
	return i;
}

static void put_v(output_buffer_tt * bc, uint64_t val) {
	int i = v_len(val);
	ready_write_buf(bc, i);
	for(i = (i-1)*7; i; i-=7) {
		*(bc->buf_ptr++) = 0x80 | (val>>i);
	}
	*(bc->buf_ptr++)= val & 0x7F;
}

static void put_s(output_buffer_tt * bc, int64_t val) {
	if (val<=0) put_v(bc, -2*val  );
	else        put_v(bc,  2*val-1);
}

static void put_data(output_buffer_tt * bc, int len, const void * data) {
	if (!len) return;
	assert(data);
	if (bc->write_len - (bc->buf_ptr - bc->buf) > len || bc->is_mem) {
		ready_write_buf(bc, len);
		memcpy(bc->buf_ptr, data, len);
		bc->buf_ptr += len;
	} else {
		flush_buf(bc);
		bc->file_pos += bc->osc.write(bc->osc.priv, len, data);
	}
}

static void put_vb(output_buffer_tt * bc, int len, const void * data) {
	put_v(bc, len);
	put_data(bc, len, data);
}

static void put_header(output_buffer_tt * bc, output_buffer_tt * in, output_buffer_tt * tmp, uint64_t startcode, int index_ptr) {
	int forward_ptr;
	assert(in->is_mem);
	assert(tmp->is_mem);
	clear_buffer(tmp);

	forward_ptr = bctello(in);
	forward_ptr += 4; // checksum
	if (index_ptr) forward_ptr += 8;

	// packet_header
	put_bytes(tmp, 8, startcode);
	put_v(tmp, forward_ptr);
	if (forward_ptr > 4096) put_bytes(tmp, 4, crc32(tmp->buf, bctello(tmp)));
	put_data(bc, bctello(tmp), tmp->buf);

	// packet_footer
	if (index_ptr) put_bytes(in, 8, bctello(tmp) + bctello(in) + 8 + 4);
	put_bytes(in, 4, crc32(in->buf, bctello(in)));

	put_data(bc, bctello(in), in->buf);
	if (startcode != SYNCPOINT_STARTCODE) debug_msg("header/index size: %d\n", (int)(bctello(tmp) + bctello(in)));
}

static void put_main_header(nut_context_tt * nut) {
	output_buffer_tt * tmp = clear_buffer(nut->tmp_buffer);
	int i;
	int flag, fields, timestamp = 0, mul = 1, stream = 0, size, count;

	put_v(tmp, NUT_VERSION);
	put_v(tmp, nut->stream_count);
	put_v(tmp, nut->max_distance);
	put_v(tmp, nut->timebase_count);
	for (i = 0; i < nut->timebase_count; i++) {
		put_v(tmp, nut->tb[i].num);
		put_v(tmp, nut->tb[i].den);
	}
	for (i = 0; i < 256; ) {
		fields = 0;
		flag = nut->ft[i].flags;
		if (nut->ft[i].pts_delta != timestamp) fields = 1;
		timestamp = nut->ft[i].pts_delta;
		if (nut->ft[i].mul != mul) fields = 2;
		mul = nut->ft[i].mul;
		if (nut->ft[i].stream != stream) fields = 3;
		stream = nut->ft[i].stream;
		if (nut->ft[i].lsb != 0) fields = 4;
		size = nut->ft[i].lsb;

		for (count = 0; i < 256; count++, i++) {
			if (i == 'N') { count--; continue; }
			if (nut->ft[i].flags != flag) break;
			if (nut->ft[i].stream != stream) break;
			if (nut->ft[i].mul != mul) break;
			if (nut->ft[i].lsb != size + count) break;
			if (nut->ft[i].pts_delta != timestamp) break;
		}
		if (count != mul - size) fields = 6;

		put_v(tmp, flag);
		put_v(tmp, fields);
		if (fields > 0) put_s(tmp, timestamp);
		if (fields > 1) put_v(tmp, mul);
		if (fields > 2) put_v(tmp, stream);
		if (fields > 3) put_v(tmp, size);
		if (fields > 4) put_v(tmp, 0); // reserved
		if (fields > 5) put_v(tmp, count);
	}
	put_v(tmp, 0); // header_count_minus1
	put_v(tmp, 0); // main_flags

	put_header(nut->o, tmp, nut->tmp_buffer2, MAIN_STARTCODE, 0);
}

static void put_stream_header(nut_context_tt * nut, int id) {
	output_buffer_tt * tmp = clear_buffer(nut->tmp_buffer);
	stream_context_tt * sc = &nut->sc[id];

	put_v(tmp, id);
	put_v(tmp, sc->sh.type);
	put_vb(tmp, sc->sh.fourcc_len, sc->sh.fourcc);
	put_v(tmp, sc->timebase_id);
	put_v(tmp, sc->msb_pts_shift);
	put_v(tmp, sc->max_pts_distance);
	put_v(tmp, sc->sh.decode_delay);
	put_v(tmp, sc->sh.fixed_fps ? 1 : 0);
	put_vb(tmp, sc->sh.codec_specific_len, sc->sh.codec_specific);

	switch (sc->sh.type) {
		case NUT_VIDEO_CLASS:
			put_v(tmp, sc->sh.width);
			put_v(tmp, sc->sh.height);
			put_v(tmp, sc->sh.sample_width);
			put_v(tmp, sc->sh.sample_height);
			put_v(tmp, sc->sh.colorspace_type);
			break;
		case NUT_AUDIO_CLASS:
			put_v(tmp, sc->sh.samplerate_num);
			put_v(tmp, sc->sh.samplerate_denom);
			put_v(tmp, sc->sh.channel_count);
			break;
	}

	put_header(nut->o, tmp, nut->tmp_buffer2, STREAM_STARTCODE, 0);
}

static void put_info(nut_context_tt * nut, const nut_info_packet_tt * info) {
	output_buffer_tt * tmp = clear_buffer(nut->tmp_buffer);
	int i;

	put_v(tmp, info->stream_id_plus1);
	put_s(tmp, info->chapter_id);
	for (i = 0; i < nut->timebase_count; i++) if (compare_ts(1, info->chapter_tb, 1, nut->tb[i]) == 0) break;
	put_v(tmp, info->chapter_start * nut->timebase_count + i);
	put_v(tmp, info->chapter_len);
	put_v(tmp, info->count);

	for(i = 0; i < info->count; i++){
		nut_info_field_tt * field = &info->fields[i];
		put_vb(tmp, strlen(field->name), field->name);
		if (!strcmp(field->type, "v")) {
			put_s(tmp, field->val);
		} else if (!strcmp(field->type, "s")) {
			put_s(tmp, -3);
			put_s(tmp, field->val);
		} else if (!strcmp(field->type, "t")) {
			int j;
			for (j = 0; j < nut->timebase_count; j++) if (compare_ts(1, field->tb, 1, nut->tb[j]) == 0) break;
			put_s(tmp, -4);
			put_v(tmp, field->val * nut->timebase_count + j);
		} else if (!strcmp(field->type, "r")) {
			put_s(tmp, -(field->den + 4));
			put_s(tmp, field->val);
		} else {
			if (strcmp(field->type, "UTF-8")) {
				put_s(tmp, -1);
			} else {
				put_s(tmp, -2);
				put_vb(tmp, strlen(field->type), field->type);
			}
			put_vb(tmp, field->val, field->data);
		}
	}

	put_header(nut->o, tmp, nut->tmp_buffer2, INFO_STARTCODE, 0);
}

static void put_headers(nut_context_tt * nut) {
	int i;
	nut->last_headers = bctello(nut->o);
	nut->headers_written++;

	put_main_header(nut);
	for (i = 0; i < nut->stream_count; i++) put_stream_header(nut, i);
	for (i = 0; i < nut->info_count; i++) put_info(nut, &nut->info[i]);
}

static void put_syncpoint(nut_context_tt * nut) {
	output_buffer_tt * tmp = clear_buffer(nut->tmp_buffer);
	int i;
	uint64_t pts = 0;
	int timebase = 0;
	int back_ptr = 0;
	int keys[NUT_MAX_STREAMS];
	syncpoint_list_tt * s = &nut->syncpoints;

	assert(nut->stream_count <= NUT_MAX_STREAMS);
	nut->last_syncpoint = bctello(nut->o);

	for (i = 0; i < nut->stream_count; i++) {
		if (nut->sc[i].last_dts > 0 && compare_ts(nut->sc[i].last_dts, TO_TB(i), pts, nut->tb[timebase]) > 0) {
			pts = nut->sc[i].last_dts;
			timebase = nut->sc[i].timebase_id;
		}
	}

	if (s->alloc_len <= s->len) {
		s->alloc_len += PREALLOC_SIZE;
		s->s = nut->alloc->realloc(s->s, s->alloc_len * sizeof(syncpoint_tt));
		s->pts = nut->alloc->realloc(s->pts, s->alloc_len * nut->stream_count * sizeof(uint64_t));
		s->eor = nut->alloc->realloc(s->eor, s->alloc_len * nut->stream_count * sizeof(uint64_t));
	}

	for (i = 0; i < nut->stream_count; i++) {
		s->pts[s->len * nut->stream_count + i] = nut->sc[i].last_key;
		s->eor[s->len * nut->stream_count + i] = nut->sc[i].eor > 0 ? nut->sc[i].eor : 0;
	}
	s->s[s->len].pos = nut->last_syncpoint;
	s->len++;

	for (i = 0; i < nut->stream_count; i++) keys[i] = !!nut->sc[i].eor;
	for (i = s->len; --i; ) {
		int j;
		int n = 1;
		for (j = 0; j < nut->stream_count; j++) {
			if (keys[j]) continue;
			if (!s->pts[i * nut->stream_count + j]) continue;
			if (compare_ts(s->pts[i * nut->stream_count + j] - 1, TO_TB(j), pts, nut->tb[timebase]) <= 0) keys[j] = 1;
		}
		for (j = 0; j < nut->stream_count; j++) n &= keys[j];
		if (n) { i--; break; }
	}
	back_ptr = (nut->last_syncpoint - s->s[i].pos) / 16;
	if (!nut->mopts.write_index) { // clear some syncpoint cache if possible
		s->len -= i;
		memmove(s->s, s->s + i, s->len * sizeof(syncpoint_tt));
		memmove(s->pts, s->pts + i * nut->stream_count, s->len * nut->stream_count * sizeof(uint64_t));
		memmove(s->eor, s->eor + i * nut->stream_count, s->len * nut->stream_count * sizeof(uint64_t));
	}

	for (i = 0; i < nut->stream_count; i++) {
		nut->sc[i].last_pts = convert_ts(pts, nut->tb[timebase], TO_TB(i));
		nut->sc[i].last_key = 0;
		if (nut->sc[i].eor) nut->sc[i].eor = -1; // so we know to ignore this stream in future syncpoints
	}

	put_v(tmp, pts * nut->timebase_count + timebase);
	put_v(tmp, back_ptr);

	put_header(nut->o, tmp, nut->tmp_buffer2, SYNCPOINT_STARTCODE, 0);

	nut->sync_overhead += bctello(tmp) + bctello(nut->tmp_buffer2);
}

static void put_index(nut_context_tt * nut) {
	output_buffer_tt * tmp = clear_buffer(nut->tmp_buffer);
	syncpoint_list_tt * s = &nut->syncpoints;
	int i;
	uint64_t max_pts = 0;
	int timebase = 0;

	for (i = 0; i < nut->stream_count; i++) {
		if (compare_ts(nut->sc[i].sh.max_pts, TO_TB(i), max_pts, nut->tb[timebase]) > 0) {
			max_pts = nut->sc[i].sh.max_pts;
			timebase = nut->sc[i].timebase_id;
		}
	}
	put_v(tmp, max_pts * nut->timebase_count + timebase);

	put_v(tmp, s->len);
	for (i = 0; i < s->len; i++) {
		off_t pos = s->s[i].pos / 16;
		off_t last_pos = i ? s->s[i-1].pos / 16 : 0;
		put_v(tmp, pos - last_pos);
	}

	for (i = 0; i < nut->stream_count; i++) {
		uint64_t a, last = 0; // all of pts[] array is off by one. using 0 for last pts is equivalent to -1 in spec.
		int j;
		for (j = 0; j < s->len; ) {
			int k;
			a = 0;
			for (k = 0; k < 5 && j+k < s->len; k++) a |= !!s->pts[(j + k) * nut->stream_count + i] << k;
			if (a == 0 || a == ((1 << k) - 1)) {
				int flag = a & 2;
				for (k = 0; j+k < s->len; k++) if (!s->pts[(j+k) * nut->stream_count + i] != !flag) break;
				put_v(tmp, k << 2 | flag | 1);
				if (j+k < s->len) k++;
			} else {
				for (; k+7 < 62 && j+k < s->len; ) {
					uint64_t b = 0;
					int tmp2;
					for (tmp2 = 0; tmp2 < 7 && j+k+tmp2 < s->len; tmp2++) {
						b |= !!s->pts[(j + k + tmp2) * nut->stream_count + i] << tmp2;
					}
					if (b == 0 || b == ((1 << tmp2) - 1)) break;
					a |= b << k;
					k += tmp2;
				}
				put_v(tmp, ((1LL << k) | a) << 1);
			}
			assert(k > 4 || j+k == s->len);
			j += k;
			for (k = j - k; k < j; k++) {
				if (!s->pts[k * nut->stream_count + i]) continue;
				if (s->eor[k * nut->stream_count + i]) {
					put_v(tmp, 0);
					put_v(tmp, s->pts[k * nut->stream_count + i] - last);
					put_v(tmp, s->eor[k * nut->stream_count + i] - s->pts[k * nut->stream_count + i]);
					last = s->eor[k * nut->stream_count + i];
				} else {
					put_v(tmp, s->pts[k * nut->stream_count + i] - last);
					last = s->pts[k * nut->stream_count + i];
				}
			}
		}
	}

	put_header(nut->o, tmp, nut->tmp_buffer2, INDEX_STARTCODE, 1);
}

static int frame_header(nut_context_tt * nut, output_buffer_tt * tmp, const nut_packet_tt * fd) {
	stream_context_tt * sc = &nut->sc[fd->stream];
	int i, ftnum = -1, size = 0, coded_pts, coded_flags = 0, msb_pts = (1 << sc->msb_pts_shift);
	int checksum = 0, pts_delta = (int64_t)fd->pts - (int64_t)sc->last_pts;

	if (ABS(pts_delta) < (msb_pts/2) - 1) coded_pts = fd->pts & (msb_pts - 1);
	else coded_pts = fd->pts + msb_pts;

	if (fd->len > 2*nut->max_distance) checksum = 1;
	if (ABS(pts_delta) > sc->max_pts_distance) {
		debug_msg("%d > %d || %d - %d > %d   \n", fd->len, 2*nut->max_distance, (int)fd->pts, (int)sc->last_pts, sc->max_pts_distance);
		checksum = 1;
	}

	for (i = 0; i < 256; i++) {
		int len = 1; // frame code
		int flags = nut->ft[i].flags;
		if (flags & FLAG_INVALID) continue;
		if (flags & FLAG_CODED) {
			flags = fd->flags & NUT_API_FLAGS;
			if (nut->ft[i].stream != fd->stream) flags |= FLAG_STREAM_ID;
			if (nut->ft[i].pts_delta != pts_delta) flags |= FLAG_CODED_PTS;
			if (nut->ft[i].lsb != fd->len) flags |= FLAG_SIZE_MSB;
			if (checksum) flags |= FLAG_CHECKSUM;
			flags |= FLAG_CODED;
		}
		if ((flags ^ fd->flags) & NUT_API_FLAGS) continue;
		if (!(flags & FLAG_STREAM_ID) && nut->ft[i].stream != fd->stream) continue;
		if (!(flags & FLAG_CODED_PTS) && nut->ft[i].pts_delta != pts_delta) continue;
		if (flags & FLAG_SIZE_MSB) { if ((fd->len - nut->ft[i].lsb) % nut->ft[i].mul) continue; }
		else { if (nut->ft[i].lsb != fd->len) continue; }
		if (!(flags & FLAG_CHECKSUM) && checksum) continue;

		len += !(flags & FLAG_CODED)    ? 0 : v_len(flags ^ nut->ft[i].flags);
		len += !(flags & FLAG_STREAM_ID)? 0 : v_len(fd->stream);
		len += !(flags & FLAG_CODED_PTS)? 0 : v_len(coded_pts);
		len += !(flags & FLAG_SIZE_MSB) ? 0 : v_len((fd->len - nut->ft[i].lsb) / nut->ft[i].mul);
		len += !(flags & FLAG_CHECKSUM) ? 0 : 4;
		if (!size || len < size) { ftnum = i; coded_flags = flags; size = len; }
	}
	assert(ftnum != -1);
	if (tmp) {
		put_bytes(tmp, 1, ftnum); // frame_code
		if (coded_flags & FLAG_CODED)     put_v(tmp, coded_flags ^ nut->ft[ftnum].flags);
		if (coded_flags & FLAG_STREAM_ID) put_v(tmp, fd->stream);
		if (coded_flags & FLAG_CODED_PTS) put_v(tmp, coded_pts);
		if (coded_flags & FLAG_SIZE_MSB)  put_v(tmp, (fd->len - nut->ft[ftnum].lsb) / nut->ft[ftnum].mul);
		if (coded_flags & FLAG_CHECKSUM)  put_bytes(tmp, 4, crc32(tmp->buf, bctello(tmp)));
	}
	return size;
}

static int add_timebase(nut_context_tt * nut, nut_timebase_tt tb) {
	int i;
	for (i = 0; i < nut->timebase_count; i++) if (compare_ts(1, nut->tb[i], 1, tb) == 0) break;
	if (i == nut->timebase_count) {
		nut->tb = nut->alloc->realloc(nut->tb, sizeof(nut_timebase_tt) * ++nut->timebase_count);
		nut->tb[i] = tb;
	}
	return i;
}

static void check_header_repetition(nut_context_tt * nut) {
	if (nut->mopts.realtime_stream) return;
	if (bctello(nut->o) >= (1 << 23)) {
		int i; // ### magic value for header repetition
		for (i = 24; bctello(nut->o) >= (1 << i); i++);
		i--;
		if (nut->last_headers < (1 << i)) {
			put_headers(nut);
		}
	}
}

void nut_write_frame(nut_context_tt * nut, const nut_packet_tt * fd, const uint8_t * buf) {
	stream_context_tt * sc = &nut->sc[fd->stream];
	output_buffer_tt * tmp;
	int i;

	check_header_repetition(nut);
	// distance syncpoints
	if (nut->last_syncpoint < nut->last_headers ||
		bctello(nut->o) - nut->last_syncpoint + fd->len + frame_header(nut, NULL, fd) > nut->max_distance) put_syncpoint(nut);

	tmp = clear_buffer(nut->tmp_buffer);
	sc->overhead += frame_header(nut, tmp, fd);
	sc->total_frames++;
	sc->tot_size += fd->len;

	put_data(nut->o, bctello(tmp), tmp->buf);
	put_data(nut->o, fd->len, buf);

        for (i = 0; i < nut->stream_count; i++) {
		if (nut->sc[i].last_dts == -1) continue;
		if (compare_ts(fd->pts, TO_TB(fd->stream), nut->sc[i].last_dts, TO_TB(i)) < 0)
			debug_msg("%lld %d (%f) %lld %d (%f) \n",
				fd->pts, fd->stream, TO_DOUBLE(fd->stream, fd->pts),
				nut->sc[i].last_dts, i, TO_DOUBLE(i, nut->sc[i].last_dts));
		assert(compare_ts(fd->pts, TO_TB(fd->stream), nut->sc[i].last_dts, TO_TB(i)) >= 0);
	}

	sc->last_pts = fd->pts;
	sc->last_dts = get_dts(sc->sh.decode_delay, sc->pts_cache, fd->pts);
	sc->sh.max_pts = MAX(sc->sh.max_pts, fd->pts);

	if ((fd->flags & NUT_FLAG_KEY) && !sc->last_key) sc->last_key = fd->pts + 1;
	if (fd->flags & NUT_FLAG_EOR) sc->eor = fd->pts + 1;
	else sc->eor = 0;

	if (nut->mopts.realtime_stream) flush_buf(nut->o);
}

void nut_write_info(nut_context_tt * nut, const nut_info_packet_tt * info) {
	if (!nut->mopts.realtime_stream) return;

	nut->last_headers = bctello(nut->o); // to force syncpoint writing after the info header
	put_info(nut, info);
	if (nut->mopts.realtime_stream) flush_buf(nut->o);
}

nut_context_tt * nut_muxer_init(const nut_muxer_opts_tt * mopts, const nut_stream_header_tt s[], const nut_info_packet_tt info[]) {
	nut_context_tt * nut;
	nut_frame_table_input_tt * fti = mopts->fti, mfti[256];
	int i, n;
	// TODO check that all input is valid

	if (mopts->alloc.malloc) nut = mopts->alloc.malloc(sizeof(nut_context_tt));
	else nut = malloc(sizeof(nut_context_tt));

	nut->mopts = *mopts;
	if (nut->mopts.realtime_stream) nut->mopts.write_index = 0;

	nut->alloc = &nut->mopts.alloc;

	if (!nut->alloc->malloc) {
		nut->alloc->malloc = malloc;
		nut->alloc->realloc = realloc;
		nut->alloc->free = free;
	}

	nut->o = new_output_buffer(nut->alloc, mopts->output);
	nut->tmp_buffer = new_mem_buffer(nut->alloc); // general purpose buffer
	nut->tmp_buffer2 = new_mem_buffer(nut->alloc); //  for packet_headers
	nut->max_distance = mopts->max_distance;

	if (nut->mopts.realtime_stream) nut->o->is_mem = 1;

	if (nut->max_distance > 65536) nut->max_distance = 65536;

	if (!fti) {
		nut_framecode_generate(s, mfti);
		fti = mfti;
	}
	debug_msg("/""/ { %4s, %3s, %6s, %3s, %4s, %5s },\n", "flag", "pts", "stream", "mul", "size", "count");
	for (n=i=0; i < 256; n++) {
		int j;
		assert(fti[n].flag != -1);

		debug_msg("   { %4d, %3d, %6d, %3d, %4d, %5d },\n", fti[n].flag, fti[n].pts, fti[n].stream, fti[n].mul, fti[n].size, fti[n].count);
		for(j = 0; j < fti[n].count && i < 256; j++, i++) {
			if (i == 'N') {
				nut->ft[i].flags = FLAG_INVALID;
				j--;
				continue;
			}
			nut->ft[i].flags     = fti[n].flag;
			nut->ft[i].pts_delta = fti[n].pts;
			nut->ft[i].mul       = fti[n].mul;
			nut->ft[i].stream    = fti[n].stream;
			nut->ft[i].lsb       = fti[n].size + j;
		}
	}
	debug_msg("   { %4d, %3d, %6d, %3d, %4d, %5d },\n", fti[n].flag, fti[n].pts, fti[n].stream, fti[n].mul, fti[n].size, fti[n].count);
	assert(fti[n].flag == -1);

	nut->sync_overhead = 0;

	nut->syncpoints.len = 0;
	nut->syncpoints.alloc_len = 0;
	nut->syncpoints.s = NULL;
	nut->syncpoints.pts = NULL;
	nut->syncpoints.eor = NULL;
	nut->last_syncpoint = 0;
	nut->headers_written = 0;

	for (nut->stream_count = 0; s[nut->stream_count].type >= 0; nut->stream_count++);

	nut->sc = nut->alloc->malloc(sizeof(stream_context_tt) * nut->stream_count);
	nut->tb = NULL;
	nut->timebase_count = 0;

	for (i = 0; i < nut->stream_count; i++) {
		int j;
		nut->sc[i].last_key = 0;
		nut->sc[i].last_pts = 0;
		nut->sc[i].last_dts = -1;
		nut->sc[i].msb_pts_shift = 7; // TODO
		nut->sc[i].max_pts_distance = (s[i].time_base.den + s[i].time_base.num - 1) / s[i].time_base.num; // TODO
		nut->sc[i].eor = 0;
		nut->sc[i].sh = s[i];
		nut->sc[i].sh.max_pts = 0;

		nut->sc[i].sh.fourcc = nut->alloc->malloc(s[i].fourcc_len);
		memcpy(nut->sc[i].sh.fourcc, s[i].fourcc, s[i].fourcc_len);

		nut->sc[i].sh.codec_specific = nut->alloc->malloc(s[i].codec_specific_len);
		memcpy(nut->sc[i].sh.codec_specific, s[i].codec_specific, s[i].codec_specific_len);

		nut->sc[i].pts_cache = nut->alloc->malloc(nut->sc[i].sh.decode_delay * sizeof(int64_t));

		nut->sc[i].timebase_id = add_timebase(nut, s[i].time_base);

		// reorder.c
		nut->sc[i].reorder_pts_cache = nut->alloc->malloc(nut->sc[i].sh.decode_delay * sizeof(int64_t));
		for (j = 0; j < nut->sc[i].sh.decode_delay; j++) nut->sc[i].reorder_pts_cache[j] = nut->sc[i].pts_cache[j] = -1;
		nut->sc[i].next_pts = 0;
		nut->sc[i].packets = NULL;
		nut->sc[i].num_packets = 0;

		// debug
		nut->sc[i].total_frames = 0;
		nut->sc[i].overhead = 0;
		nut->sc[i].tot_size = 0;
	}

	if (info) {
		for (nut->info_count = 0; info[nut->info_count].count >= 0; nut->info_count++);

		nut->info = nut->alloc->malloc(sizeof(nut_info_packet_tt) * nut->info_count);

		for (i = 0; i < nut->info_count; i++) {
			int j;
			nut->info[i] = info[i];
			nut->info[i].fields = nut->alloc->malloc(sizeof(nut_info_field_tt) * info[i].count);
			add_timebase(nut, nut->info[i].chapter_tb);
			for (j = 0; j < info[i].count; j++) {
				nut->info[i].fields[j] = info[i].fields[j];
				if (info[i].fields[j].data) {
					nut->info[i].fields[j].data = nut->alloc->malloc(info[i].fields[j].val);
					memcpy(nut->info[i].fields[j].data, info[i].fields[j].data, info[i].fields[j].val);
				}
				if (!strcmp(nut->info[i].fields[j].type, "t")) add_timebase(nut, nut->info[i].fields[j].tb);
			}
		}
	} else {
		nut->info_count = 0;
		nut->info = NULL;
	}

	for (i = 0; i < nut->timebase_count; i++) {
		int t = gcd(nut->tb[i].num, nut->tb[i].den);
		nut->tb[i].num /= t;
		nut->tb[i].den /= t;
	}

	put_data(nut->o, strlen(ID_STRING) + 1, ID_STRING);

	put_headers(nut);

	if (nut->mopts.realtime_stream) flush_buf(nut->o);

	return nut;
}

void nut_muxer_uninit(nut_context_tt * nut) {
	int i;
	int total = 0;
	if (!nut) return;

	if (!nut->mopts.realtime_stream) {
		while (nut->headers_written < 2) put_headers(nut); // force 3rd copy of main headers
		put_headers(nut);
	}
	if (nut->mopts.write_index) put_index(nut);

	for (i = 0; i < nut->stream_count; i++) {
		total += nut->sc[i].tot_size;
		debug_msg("Stream %d:\n", i);
		debug_msg("   frames: %d\n", nut->sc[i].total_frames);
		debug_msg("   TOT: ");
		debug_msg("packet size: %d ", nut->sc[i].tot_size);
		debug_msg("packet overhead: %d ", nut->sc[i].overhead);
		debug_msg("(%.2lf%%)\n", (double)nut->sc[i].overhead / nut->sc[i].tot_size * 100);
		debug_msg("   AVG: ");
		debug_msg("packet size: %.2lf ", (double)nut->sc[i].tot_size / nut->sc[i].total_frames);
		debug_msg("packet overhead: %.2lf\n", (double)nut->sc[i].overhead / nut->sc[i].total_frames);

		nut->alloc->free(nut->sc[i].sh.fourcc);
		nut->alloc->free(nut->sc[i].sh.codec_specific);
		nut->alloc->free(nut->sc[i].pts_cache);
		nut->alloc->free(nut->sc[i].reorder_pts_cache);
	}
	nut->alloc->free(nut->sc);
	nut->alloc->free(nut->tb);

	for (i = 0; i < nut->info_count; i++) {
		int j;
		for (j = 0; j < nut->info[i].count; j++) nut->alloc->free(nut->info[i].fields[j].data);
		nut->alloc->free(nut->info[i].fields);
	}
	nut->alloc->free(nut->info);

	debug_msg("Syncpoints: %d size: %d\n", nut->syncpoints.len, nut->sync_overhead);

	nut->alloc->free(nut->syncpoints.s);
	nut->alloc->free(nut->syncpoints.pts);
	nut->alloc->free(nut->syncpoints.eor);

	free_buffer(nut->tmp_buffer);
	free_buffer(nut->tmp_buffer2);
	debug_msg("TOTAL: %d bytes data, %d bytes overhead, %.2lf%% overhead\n", total,
		(int)bctello(nut->o) - total, (double)(bctello(nut->o) - total) / total*100);
	free_buffer(nut->o); // flushes file
	nut->alloc->free(nut);
}
