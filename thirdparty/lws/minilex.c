/*
 * minilex.c
 *
 * High efficiency lexical state parser
 *
 * Copyright (C)2011-2014 Andy Green <andy@warmcat.com>
 *
 * Licensed under LGPL2
 *
 * Usage: gcc minilex.c -o minilex && ./minilex > lextable.h
 *
 * Run it twice to test parsing on the generated table on stderr
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lextable-strings.h"

/*
 * b7 = 0 = 1-byte seq
 *	    0x08 = fail
 *	    2-byte seq
 *	    0x00 - 0x07, then terminal as given in 2nd byte
	    3-byte seq
 *	    no match: go fwd 3 byte, match: jump fwd by amt in +1/+2 bytes
 *    = 1 = 1-byte seq
 *	    no match: die, match go fwd 1 byte
 */

unsigned char lextable[] = {
	#include "lextable.h"
};

#define PARALLEL 30

struct state {
	char c[PARALLEL];
	int state[PARALLEL];
	int count;
	int bytepos;

	int real_pos;
};

struct state state[1000];
int next = 1;

#define FAIL_CHAR 0x08

int lextable_decode(int pos, char c)
{
	while (1) {
		if (lextable[pos] & (1 << 7)) { /* 1-byte, fail on mismatch */
			if ((lextable[pos] & 0x7f) != c)
				return -1;
			/* fall thru */
			pos++;
			if (lextable[pos] == FAIL_CHAR)
				return -1;
			return pos;
		} else { /* b7 = 0, end or 3-byte */
			if (lextable[pos] < FAIL_CHAR) /* terminal marker */
				return pos;

			if (lextable[pos] == c) /* goto */
				return pos + (lextable[pos + 1]) +
						(lextable[pos + 2] << 8);
			/* fall thru goto */
			pos += 3;
			/* continue */
		}
	}
}

int main(void)
{
	int n = 0;
	int m = 0;
	int prev;
	char c;
	int walk;
	int saw;
	int y;
	int j;
	int pos = 0;

	while (n < sizeof(set) / sizeof(set[0])) {

		m = 0;
		walk = 0;
		prev = 0;

		if (set[n][0] == '\0') {
			n++;
			continue;
		}

		while (set[n][m]) {

			saw = 0;
			for (y = 0; y < state[walk].count; y++)
				if (state[walk].c[y] == set[n][m]) {
					/* exists -- go forward */
					walk = state[walk].state[y];
					saw = 1;
					break;
				}

			if (saw)
				goto again;

			/* something we didn't see before */

			state[walk].c[state[walk].count] = set[n][m];

			state[walk].state[state[walk].count] = next;
			state[walk].count++;
			walk = next++;
again:
			m++;
		}

		state[walk].c[0] = n++;
		state[walk].state[0] = 0; /* terminal marker */
		state[walk].count = 1;
	}

	walk = 0;
	for (n = 0; n < next; n++) {
		state[n].bytepos = walk;
		walk += (2 * state[n].count);
	}

	/* compute everyone's position first */

	pos = 0;
	walk = 0;
	for (n = 0; n < next; n++) {

		state[n].real_pos = pos;

		for (m = 0; m < state[n].count; m++) {

			if (state[n].state[m] == 0)
				pos += 2; /* terminal marker */
			else { /* c is a character */
				if ((state[state[n].state[m]].bytepos -
								walk) == 2)
					pos++;
				else {
					pos += 3;
					if (m == state[n].count - 1)
						pos++; /* fail */
				}
			}
			walk += 2;
		}
	}

	walk = 0;
	pos = 0;
	for (n = 0; n < next; n++) {
		for (m = 0; m < state[n].count; m++) {

			if (!m)
				fprintf(stdout, "/* pos %04x: %3d */ ",
							  state[n].real_pos, n);
			else
				fprintf(stdout, "                    ");

			y = state[n].c[m];
			saw = state[n].state[m];

			if (saw == 0) { // c is a terminal then

				if (y > 0x7ff) {
					fprintf(stderr, "terminal too big\n");
					return 2;
				}

				fprintf(stdout, "   0x%02X, 0x%02X           "
					"       "
					"/* - terminal marker %2d - */,\n",
						    y >> 8, y & 0xff, y & 0x7f);
				pos += 2;
				walk += 2;
				continue;
			}

			/* c is a character */

			prev = y &0x7f;
			if (prev < 32 || prev > 126)
				prev = '.';


			if ((state[saw].bytepos - walk) == 2) {
				fprintf(stdout, "   0x%02X /* '%c' -> */,\n",
						y | 0x80, prev);
				pos++;
				walk += 2;
				continue;
			}

			j = state[saw].real_pos - pos;

			if (j > 0xffff) {
				fprintf(stderr,
				  "Jump > 64K bytes ahead (%d to %d)\n",
					state[n].real_pos, state[saw].real_pos);
				return 1;
			}
			fprintf(stdout, "   0x%02X /* '%c' */, 0x%02X, 0x%02X  "
				"/* (to 0x%04X state %3d) */,\n",
				y, prev,
				j & 0xff, j >> 8,
				state[saw].real_pos, saw);
			pos += 3;

			if (m == state[n].count - 1) {
				fprintf(stdout,
				  "                       0x%02X, /* fail */\n",
								FAIL_CHAR);
				pos++; /* fail */
			}

			walk += 2;
		}
	}

	fprintf(stdout, "/* total size %d bytes */\n", pos);

	/*
	 * Try to parse every legal input string
	 */

	for (n = 0; n < sizeof(set) / sizeof(set[0]); n++) {
		walk = 0;
		m = 0;
		y = -1;

		if (set[n][0] == '\0')
			continue;

		fprintf(stderr, "  trying '%s'\n", set[n]);

		while (set[n][m]) {
			walk = lextable_decode(walk, set[n][m]);
			if (walk < 0) {
				fprintf(stderr, "failed\n");
				return 3;
			}

			if (lextable[walk] < FAIL_CHAR) {
				y = (lextable[walk] << 8) + lextable[walk + 1];
				break;
			}
			m++;
		}

		if (y != n) {
			fprintf(stderr, "decode failed %d\n", y);
			return 4;
		}
	}

	fprintf(stderr, "All decode OK\n");

	return 0;
}
