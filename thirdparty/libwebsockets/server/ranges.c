/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * RFC7233 ranges parser
 *
 * Copyright (C) 2016 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 */

#include "private-libwebsockets.h"

/*
 * RFC7233 examples
 *
 * o  The first 500 bytes (byte offsets 0-499, inclusive):
 *
 *      bytes=0-499
 *
 * o  The second 500 bytes (byte offsets 500-999, inclusive):
 *
 *      bytes=500-999
 *
 * o  The final 500 bytes (byte offsets 9500-9999, inclusive):
 *
 *      bytes=-500
 *
 * Or:
 *
 *      bytes=9500-
 *
 * o  The first and last bytes only (bytes 0 and 9999):
 *
 *      bytes=0-0,-1
 *
 * o  Other valid (but not canonical) specifications of the second 500
 *    bytes (byte offsets 500-999, inclusive):
 *
 *      bytes=500-600,601-999
 *      bytes=500-700,601-999
 */

/*
 * returns 1 if the range struct represents a usable range
 *   if no ranges header, you get one of these for the whole
 *   file.  Otherwise you get one for each valid range in the
 *   header.
 *
 * returns 0 if no further valid range forthcoming; rp->state
 *   may be LWSRS_SYNTAX or LWSRS_COMPLETED
 */

int
lws_ranges_next(struct lws_range_parsing *rp)
{
	static const char * const beq = "bytes=";
	char c;

	while (1) {

		c = rp->buf[rp->pos];

		switch (rp->state) {
		case LWSRS_SYNTAX:
		case LWSRS_COMPLETED:
			return 0;

		case LWSRS_NO_ACTIVE_RANGE:
			rp->state = LWSRS_COMPLETED;
			return 0;

		case LWSRS_BYTES_EQ: // looking for "bytes="
			if (c != beq[rp->pos]) {
				rp->state = LWSRS_SYNTAX;
				return -1;
			}
			if (rp->pos == 5)
				rp->state = LWSRS_FIRST;
			break;

		case LWSRS_FIRST:
			rp->start = 0;
			rp->end = 0;
			rp->start_valid = 0;
			rp->end_valid = 0;

			rp->state = LWSRS_STARTING;

			// fallthru

		case LWSRS_STARTING:
			if (c == '-') {
				rp->state = LWSRS_ENDING;
				break;
			}

			if (!(c >= '0' && c <= '9')) {
				rp->state = LWSRS_SYNTAX;
				return 0;
			}
			rp->start = (rp->start * 10) + (c - '0');
			rp->start_valid = 1;
			break;

		case LWSRS_ENDING:
			if (c == ',' || c == '\0') {
				rp->state = LWSRS_FIRST;
				if (c == ',')
					rp->pos++;

				/*
				 * By the end of this, start and end are
				 * always valid if the range still is
				 */

				if (!rp->start_valid) { /* eg, -500 */
					if (rp->end > rp->extent)
						rp->end = rp->extent;

					rp->start = rp->extent - rp->end;
					rp->end = rp->extent - 1;
				} else
					if (!rp->end_valid)
						rp->end = rp->extent - 1;

				rp->did_try = 1;

				/* end must be >= start or ignore it */
				if (rp->end < rp->start) {
					if (c == ',')
						break;
					rp->state = LWSRS_COMPLETED;
					return 0;
				}

				return 1; /* issue range */
			}

			if (!(c >= '0' && c <= '9')) {
				rp->state = LWSRS_SYNTAX;
				return 0;
			}
			rp->end = (rp->end * 10) + (c - '0');
			rp->end_valid = 1;
			break;
		}

		rp->pos++;
	}
}

void
lws_ranges_reset(struct lws_range_parsing *rp)
{
	rp->pos = 0;
	rp->ctr = 0;
	rp->start = 0;
	rp->end = 0;
	rp->start_valid = 0;
	rp->end_valid = 0;
	rp->state = LWSRS_BYTES_EQ;
}

/*
 * returns count of valid ranges
 */
int
lws_ranges_init(struct lws *wsi, struct lws_range_parsing *rp,
		unsigned long long extent)
{
	rp->agg = 0;
	rp->send_ctr = 0;
	rp->inside = 0;
	rp->count_ranges = 0;
	rp->did_try = 0;
	lws_ranges_reset(rp);
	rp->state = LWSRS_COMPLETED;

	rp->extent = extent;

	if (lws_hdr_copy(wsi, (char *)rp->buf, sizeof(rp->buf),
			 WSI_TOKEN_HTTP_RANGE) <= 0)
		return 0;

	rp->state = LWSRS_BYTES_EQ;

	while (lws_ranges_next(rp)) {
		rp->count_ranges++;
		rp->agg += rp->end - rp->start + 1;
	}

	lwsl_debug("%s: count %d\n", __func__, rp->count_ranges);
	lws_ranges_reset(rp);

	if (rp->did_try && !rp->count_ranges)
		return -1; /* "not satisfiable */

	lws_ranges_next(rp);

	return rp->count_ranges;
}
