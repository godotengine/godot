/*-
 * Copyright 2003-2005 Colin Percival
 * All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted providing that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#if 0
__FBSDID("$FreeBSD: src/usr.bin/bsdiff/bsdiff/bsdiff.c,v 1.1 2005/08/06 01:59:05 cperciva Exp $");
#endif

#include "bsdiff/bsdiff.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>

#include "bsdiff/diff_encoder.h"
#include "bsdiff/suffix_array_index.h"

namespace bsdiff {

int bsdiff(const uint8_t* old_buf, size_t oldsize, const uint8_t* new_buf,
           size_t newsize, PatchWriterInterface* patch,
           SuffixArrayIndexInterface** sai_cache) {
	return bsdiff(old_buf, oldsize, new_buf, newsize, 0, patch,
	              sai_cache);
}

int bsdiff(const uint8_t* old_buf, size_t oldsize, const uint8_t* new_buf,
           size_t newsize, size_t min_length, PatchWriterInterface* patch,
           SuffixArrayIndexInterface** sai_cache) {
	size_t scsc, scan;
	uint64_t pos=0;
	size_t len;
	size_t lastscan,lastpos,lastoffset;
	uint64_t oldscore;
	int64_t s,Sf,lenf,Sb,lenb;
	int64_t overlap,Ss,lens;
	int64_t i;

	std::unique_ptr<SuffixArrayIndexInterface> local_sai;
	SuffixArrayIndexInterface* sai;

	if (sai_cache && *sai_cache) {
		sai = *sai_cache;
	} else {
		local_sai = CreateSuffixArrayIndex(old_buf, oldsize);
		if (!local_sai)
			return 1;
		sai = local_sai.get();

		// Transfer ownership to the caller.
		if (sai_cache)
			*sai_cache = local_sai.release();
	}

	/* Initialize the patch file encoder */
	DiffEncoder diff_encoder(patch, old_buf, oldsize, new_buf, newsize);
	if (!diff_encoder.Init())
		return 1;

	/* Compute the differences, writing ctrl as we go */
	scan=0;len=0;
	lastscan=0;lastpos=0;lastoffset=0;
	while(scan<newsize) {
		oldscore=0;

		/* If we come across a large block of data that only differs
		 * by less than 8 bytes, this loop will take a long time to
		 * go past that block of data. We need to track the number of
		 * times we're stuck in the block and break out of it. */
		int num_less_than_eight = 0;
		size_t prev_len;
		uint64_t prev_pos, prev_oldscore;
		for(scsc=scan+=len;scan<newsize;scan++) {
			prev_len=len;
			prev_oldscore=oldscore;
			prev_pos=pos;

			sai->SearchPrefix(new_buf + scan, newsize - scan, &len, &pos);

			for(;scsc<scan+len && scsc+lastoffset<oldsize;scsc++)
				if(old_buf[scsc+lastoffset] == new_buf[scsc])
					oldscore++;

			if(((len==oldscore) && (len!=0)) ||
				(len>=oldscore+8 && len>=min_length)) break;

			if((scan+lastoffset<oldsize) &&
				(old_buf[scan+lastoffset] == new_buf[scan]))
				oldscore--;

			const size_t fuzz = 8;
			if (prev_len-fuzz<=len && len<=prev_len &&
			    prev_oldscore-fuzz<=oldscore &&
			    oldscore<=prev_oldscore &&
			    prev_pos<=pos && pos <=prev_pos+fuzz &&
			    oldscore<=len && len<=oldscore+fuzz)
				++num_less_than_eight;
			else
				num_less_than_eight=0;
			if (num_less_than_eight > 100) break;
		};

		if((len!=oldscore) || (scan==newsize)) {
			s=0;Sf=0;lenf=0;
			for(i=0;(lastscan+i<scan)&&(lastpos+i<oldsize);) {
				if(old_buf[lastpos+i]==new_buf[lastscan+i]) s++;
				i++;
				if(s*2-i>Sf*2-lenf) { Sf=s; lenf=i; };
			};

			lenb=0;
			if(scan<newsize) {
				s=0;Sb=0;
				for(i=1;(scan>=lastscan+i)&&(pos>=static_cast<uint64_t>(i));i++) {
					if(old_buf[pos-i]==new_buf[scan-i]) s++;
					if(s*2-i>Sb*2-lenb) { Sb=s; lenb=i; };
				};
			};

			if(lastscan+lenf>scan-lenb) {
				overlap=(lastscan+lenf)-(scan-lenb);
				s=0;Ss=0;lens=0;
				for(i=0;i<overlap;i++) {
					if(new_buf[lastscan+lenf-overlap+i]==
					   old_buf[lastpos+lenf-overlap+i]) s++;
					if(new_buf[scan-lenb+i]==
					   old_buf[pos-lenb+i]) s--;
					if(s>Ss) { Ss=s; lens=i+1; };
				};

				lenf+=lens-overlap;
				lenb-=lens;
			};

			if (!diff_encoder.AddControlEntry(
			        ControlEntry(lenf,
			                     (scan - lenb) - (lastscan + lenf),
			                     (pos - lenb) - (lastpos + lenf))))
				return 1;

			lastscan=scan-lenb;
			lastpos=pos-lenb;
			lastoffset=pos-scan;
		};
	};
	if (!diff_encoder.Close())
		return 1;

	return 0;
}

}  // namespace bsdiff
