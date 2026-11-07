// (C) 2005-2006 Oded Shimon
// This file is available under the MIT/X license, see COPYING

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include "libnut.h"
#include "priv.h"

typedef nut_frame_table_input_tt fti_tt; // just a shortcut

static int count_streams(const nut_stream_header_tt * s) {
	int i;
	for (i = 0; s[i].type != -1; i++);
	return i;
}

void nut_framecode_generate(const nut_stream_header_tt s[], nut_frame_table_input_tt fti[256]) {
	int stream_count = count_streams(s);
	int i, n = 0, m = 0, tot_con = 0;
	enum {
		e_consume_none = 0,
		e_consume_mpeg4,
		e_consume_h264,
		e_consume_video,
		e_consume_vorbis,
	} consume[NUT_MAX_STREAMS];

	assert(stream_count <= NUT_MAX_STREAMS);
	for (i = 0; i < stream_count; i++) consume[i] = e_consume_none;

	// the basic framecodes.                                                flag,  pts, stream, mul, size, count
	fti[n++] = (fti_tt){                            /*invalid 0x00*/ FLAG_INVALID,    0,      0,   1,    0,     1 };
	fti[n++] = (fti_tt){ NUT_FLAG_KEY|FLAG_CODED_PTS|FLAG_STREAM_ID|FLAG_SIZE_MSB,    0,      0,   1,    0,     1 };
	fti[n++] = (fti_tt){              FLAG_CODED_PTS|FLAG_STREAM_ID|FLAG_SIZE_MSB,    0,      0,   1,    0,     1 };
	fti[n++] = (fti_tt){                          /*extreme fallback*/ FLAG_CODED,    1,      0,   1,    0,     1 };

	for (i = 0; i < stream_count; i++) {
		if (n + m > 230) break; // that's enough! don't overflow
		switch (s[i].type) {
		case NUT_VIDEO_CLASS:
			fti[n++] = (fti_tt){ NUT_FLAG_KEY|              FLAG_SIZE_MSB,    1,      i,   1,    0,     1 };
			fti[n++] = (fti_tt){ NUT_FLAG_KEY|FLAG_CHECKSUM|FLAG_SIZE_MSB,    1,      i,   1,    0,     1 };
			fti[n++] = (fti_tt){             FLAG_CODED_PTS|FLAG_SIZE_MSB,    0,      i,   1,    0,     1 };
			if (s[i].fourcc_len == 4 && !strncmp((char*)s[i].fourcc, "mp4v", 4)) {
				fti[n++] = (fti_tt){                                0,    1,      i,   7,    6,     1 };
				fti[n++] = (fti_tt){                                0,    2,      i,   7,    6,     1 };
				consume[i] = e_consume_mpeg4;
			} else if (s[i].fourcc_len == 4 && !strncmp((char*)s[i].fourcc, "h264", 4)) {
				consume[i] = e_consume_h264;
			} else {
				consume[i] = e_consume_video;
			}
			break;
		case NUT_AUDIO_CLASS:
			fti[n++] = (fti_tt){NUT_FLAG_KEY|               FLAG_SIZE_MSB,    1,      i,   1,    0,     1 };
			fti[n++] = (fti_tt){NUT_FLAG_KEY|FLAG_CODED_PTS|FLAG_SIZE_MSB,    0,      i,   1,    0,     1 };
			if (s[i].fourcc_len == 4 && !strncmp((char*)s[i].fourcc, "mp3 ", 4)) {
				int j, a[] = {288,336,384,480,576,672,768,960};
				for (j = 0; j < sizeof a/sizeof*a; j++)
					fti[n++] = (fti_tt){             NUT_FLAG_KEY,    1,   i, a[j]+1, a[j],     1 };
				fti[n++] = (fti_tt){       NUT_FLAG_KEY|FLAG_SIZE_MSB,    1,      i,   4,    0,     1 };
				fti[n++] = (fti_tt){       NUT_FLAG_KEY|FLAG_SIZE_MSB,    1,      i,   4,    2,     1 };
			} else if (s[i].fourcc_len == 4 && !strncmp((char*)s[i].fourcc, "vrbs", 4)) {
				fti[n++] = (fti_tt){       NUT_FLAG_KEY|FLAG_SIZE_MSB,    2,      i,   1,    0,     1 };
				fti[n++] = (fti_tt){       NUT_FLAG_KEY|FLAG_SIZE_MSB,    9,      i,   1,    0,     1 };
				fti[n++] = (fti_tt){       NUT_FLAG_KEY|FLAG_SIZE_MSB,   23,      i,   1,    0,     1 };
				fti[n++] = (fti_tt){       NUT_FLAG_KEY|FLAG_SIZE_MSB,   16,      i,   6,    0,     6 }; m+=5;
				consume[i] = e_consume_vorbis;
			}
			break;
		case NUT_SUBTITLE_CLASS:
			fti[n++] = (fti_tt){NUT_FLAG_KEY|FLAG_SIZE_MSB|FLAG_CODED_PTS,    0,      i,   5,    0,     5 }; m+=4;
			fti[n++] = (fti_tt){NUT_FLAG_KEY|NUT_FLAG_EOR| FLAG_CODED_PTS,    0,      i,   1,    0,     1 };
			break;
		case NUT_USERDATA_CLASS:
			fti[n++] = (fti_tt){NUT_FLAG_KEY|FLAG_SIZE_MSB|FLAG_CODED_PTS,    0,      i,   1,    0,     1 };
			fti[n++] = (fti_tt){             FLAG_SIZE_MSB|FLAG_CODED_PTS,    0,      i,   1,    0,     1 };
			break;
		default:
			assert(0);
		}
	}

	for (i = 0; i < stream_count; i++) if (consume[i]) tot_con++;

	if (tot_con) tot_con = (254 - (n+m))/tot_con; // 256 - 'N' - 0xFF invalid
	if (tot_con) for (i = 0; i < stream_count; i++) {
		int al = tot_con;
		switch (consume[i]) {
		case e_consume_none:
			break;
		case e_consume_mpeg4: {
			int al1 = al*35/100;
			int al2 = al*45/100;
			int al3 = al-al1-al2;
			fti[n++] = (fti_tt){                            FLAG_SIZE_MSB,    1,      i, al1,    0,   al1 }; m+=al1-1;
			fti[n++] = (fti_tt){                            FLAG_SIZE_MSB,    2,      i, al2,    0,   al2 }; m+=al2-1;
			fti[n++] = (fti_tt){                            FLAG_SIZE_MSB,   -1,      i, al3,    0,   al3 }; m+=al3-1;
			break;
		}
		case e_consume_h264: {
			int al1 = al*35/100;
			int al2 = al*35/100;
			int al3 = al*20/100;
			int al4 = al-al1-al2-al3;
			fti[n++] = (fti_tt){                            FLAG_SIZE_MSB,    1,      i, al1,    0,   al1 }; m+=al1-1;
			fti[n++] = (fti_tt){                            FLAG_SIZE_MSB,    2,      i, al2,    0,   al2 }; m+=al2-1;
			fti[n++] = (fti_tt){                            FLAG_SIZE_MSB,   -1,      i, al3,    0,   al3 }; m+=al3-1;
			fti[n++] = (fti_tt){             FLAG_CODED_PTS|FLAG_SIZE_MSB,    0,      i, al4,    0,   al4 }; m+=al4-1;
			break;
		}
		case e_consume_video:
			fti[n++] = (fti_tt){                            FLAG_SIZE_MSB,    1,      i,  al,    0,    al }; m+=al-1;
			break;
		case e_consume_vorbis: {
			int al1 = al*70/100;
			int al2 = al-al1;
			al1 /= 2; al2 /= 2;
			fti[n++] = (fti_tt){                             NUT_FLAG_KEY,   16, i,240+al1,240-al1, al1*2 }; m+=al1*2-1;
			fti[n++] = (fti_tt){                             NUT_FLAG_KEY,    2, i, 65+al2, 65-al2, al2*2 }; m+=al2*2-1;
			break;
		}
		}
	}
	i = 255-n-m;
	fti[n++] = (fti_tt){                            /*invalid 0xFF*/ FLAG_INVALID,    0,      0,   i,    0,     i };
	// the final framecode.                                                 flag,  pts, stream, mul, size, count
	fti[n++] = (fti_tt){ -1 };
}
