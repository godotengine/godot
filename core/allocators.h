/*************************************************************************/
/*  allocators.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef ALLOCATORS_H
#define ALLOCATORS_H

#include "core/os/memory.h"

template <int PREALLOC_COUNT = 64, int MAX_HANDS = 8>
class BalloonAllocator {

	enum {
		USED_FLAG = (1 << 30),
		USED_MASK = USED_FLAG - 1
	};

	struct Balloon {
		Balloon *prev;
		Balloon *next;
		uint32_t hand;
	};

	struct Hand {
		int used;
		int allocated;
		Balloon *first;
		Balloon *last;
	};

	Hand hands[MAX_HANDS];

public:
	void *alloc(size_t p_size) {
		size_t max = (1 << MAX_HANDS);
		ERR_FAIL_COND_V(p_size > max, NULL);

		uint32_t hand = 0;
		while (p_size > (size_t)(1 << hand))
			++hand;

		Hand &h = hands[hand];
		Balloon *b = h.last;

		if (h.used == h.allocated) {
			for (int i = 0; i < PREALLOC_COUNT; ++i) { // Create more unused balloons
				b = (Balloon *)memalloc(sizeof(Balloon) + (1 << hand));
				b->hand = hand;

				if (h.last) {
					b->prev = h.last;
					h.last->next = b;

				} else { // Balloon is first
					b->prev = NULL;
					h.first = b;
				}

				h.last = b;
			}

			h.last->next = NULL;
			h.allocated += PREALLOC_COUNT;
		}

		ERR_FAIL_COND_V((b->hand & USED_FLAG), NULL);

		// Remove last unused balloon, which b is currently pointing at
		h.last = b->prev;
		h.last->next = NULL;

		// Place balloon first and set used parameters
		b->prev = NULL;
		b->next = h.first;
		h.first->prev = b;
		h.first = b;
		++(h.used);
		b->hand |= USED_FLAG;

		return ++b;
	}

	void free(void *p_ptr) {
		Balloon *b = (Balloon *)p_ptr - 1;

		ERR_FAIL_COND(!(b->hand & USED_FLAG));

		b->hand = b->hand & USED_MASK; // not used
		int hand = b->hand;
		Hand &h = hands[hand];

		// Remove balloon
		if (b->prev)
			b->prev->next = b->next;
		else
			h.first = b->next;

		if (b->next)
			b->next->prev = b->prev;

		if (h.last != b) { // Place balloon last, so it can be reused
			b->prev = h.last;
			b->next = NULL;
			h.last->next = b;
			h.last = b;
		}

		--(h.used);
		if (h.used <= (h.allocated - (PREALLOC_COUNT * 2))) { // Used to free up memory

			for (int i = 0; i < PREALLOC_COUNT; ++i) {
				ERR_CONTINUE(h.last->hand & USED_FLAG);

				b = h.last->prev;
				if (b)
					b->next = NULL;
				memfree(h.last);
				h.last = b;
			}

			h.allocated -= PREALLOC_COUNT;
		}
	}

	BalloonAllocator() {
		for (int i = 0; i < MAX_HANDS; ++i) {
			hands[i].allocated = 0;
			hands[i].used = 0;
			hands[i].first = NULL;
			hands[i].last = NULL;
		}
	}

	void clear() {
		for (int i = 0; i < MAX_HANDS; ++i) {
			while (hands[i].first) {
				Balloon *b = hands[i].first;
				hands[i].first = b->next;
				memfree(b);
			}

			hands[i].allocated = 0;
			hands[i].used = 0;
			hands[i].first = NULL;
			hands[i].last = NULL;
		}
	}

	~BalloonAllocator() {
		clear();
	}
};

#endif // ALLOCATORS_H
