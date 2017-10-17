/*
 * Copyright 2015 The Etc2Comp Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
EtcSortedBlockList.cpp

SortedBlockList is a list of 4x4 blocks that can be used by the "effort" system to prioritize
the encoding of the 4x4 blocks.

The sorting is done with buckets, where each bucket is an indication of how much error each 4x4 block has

*/

#include "EtcConfig.h"
#include "EtcSortedBlockList.h"

#include "EtcBlock4x4.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

namespace Etc
{

	// ----------------------------------------------------------------------------------------------------
	// construct an empty list
	//
	// allocate enough memory to add all of the image's 4x4 blocks later
	// allocate enough buckets to sort the blocks
	//
	SortedBlockList::SortedBlockList(unsigned int a_uiImageBlocks, unsigned int a_uiBuckets)
	{
		m_uiImageBlocks = a_uiImageBlocks;
		m_iBuckets = (int)a_uiBuckets;

		m_uiAddedBlocks = 0;
		m_uiSortedBlocks = 0;
		m_palinkPool = new Link[m_uiImageBlocks];
		m_pabucket = new Bucket[m_iBuckets];
		m_fMaxError = 0.0f;

		InitBuckets();

	}

	// ----------------------------------------------------------------------------------------------------
	//
	SortedBlockList::~SortedBlockList(void)
	{
		delete[] m_palinkPool;
		delete[] m_pabucket;
	}

	// ----------------------------------------------------------------------------------------------------
    // add a 4x4 block to the list
	// the 4x4 block will be sorted later
	//
    void SortedBlockList::AddBlock(Block4x4 *a_pblock)
    {
        assert(m_uiAddedBlocks < m_uiImageBlocks);
        Link *plink = &m_palinkPool[m_uiAddedBlocks++];
		plink->Init(a_pblock);
    }

	// ----------------------------------------------------------------------------------------------------
	// sort all of the 4x4 blocks that have been added to the list
	//
	// first, determine the maximum error, then assign an error range to each bucket
	// next, determine which bucket each 4x4 block belongs to based on the 4x4 block's error
	// add the 4x4 block to the appropriate bucket
	// lastly, walk thru the buckets and add each bucket to a sorted linked list
	//
	// the resultant sorting is an approximate sorting from most to least error
	//
    void SortedBlockList::Sort(void)
    {
		assert(m_uiAddedBlocks == m_uiImageBlocks);
        InitBuckets();

        // find max block error
        m_fMaxError = -1.0f;

        for (unsigned int uiLink = 0; uiLink < m_uiAddedBlocks; uiLink++)
        {
            Link *plinkBlock = &m_palinkPool[uiLink];

            float fBlockError = plinkBlock->GetBlock()->GetError();
            if (fBlockError > m_fMaxError)
            {
                m_fMaxError = fBlockError;
            }
        }
        // prevent divide by zero or divide by negative
        if (m_fMaxError <= 0.0f)
        {
            m_fMaxError = 1.0f;
        }
		//used for debugging
		//int numDone = 0;
        // put all of the blocks with unfinished encodings into the appropriate bucket
		m_uiSortedBlocks = 0;
        for (unsigned int uiLink = 0; uiLink < m_uiAddedBlocks; uiLink++)
        {
            Link *plinkBlock = &m_palinkPool[uiLink];

			// if the encoding is done, don't add it to the list
			if (plinkBlock->GetBlock()->GetEncoding()->IsDone())
			{
				//numDone++;
				continue;
			}

            // calculate the appropriate sort bucket
            float fBlockError = plinkBlock->GetBlock()->GetError();
            int iBucket = (int) floorf(m_iBuckets * fBlockError / m_fMaxError);
            // clamp to bucket index
            iBucket = iBucket < 0 ? 0 : iBucket >= m_iBuckets ? m_iBuckets - 1 : iBucket;

            // add block to bucket
			{
				Bucket *pbucket = &m_pabucket[iBucket];
				if (pbucket->plinkLast)
				{
					pbucket->plinkLast->SetNext(plinkBlock);
					pbucket->plinkLast = plinkBlock;
				}
				else
				{
					pbucket->plinkFirst = pbucket->plinkLast = plinkBlock;
				}
				plinkBlock->SetNext(nullptr);
			}

			m_uiSortedBlocks++;

            if (0)
            {
                printf("%u: e=%.3f\n", uiLink, fBlockError);
                Print();
                printf("\n\n\n");
            }
        }
		//printf("num blocks already done: %d\n",numDone);
		//link the blocks together across buckets
		m_plinkFirst = nullptr;
		m_plinkLast = nullptr;
		for (int iBucket = m_iBuckets - 1; iBucket >= 0; iBucket--)
		{
			Bucket *pbucket = &m_pabucket[iBucket];

			if (pbucket->plinkFirst)
			{
				if (m_plinkFirst == nullptr)
				{
					m_plinkFirst = pbucket->plinkFirst;
				}
				else
				{
					assert(pbucket->plinkLast->GetNext() == nullptr);
					m_plinkLast->SetNext(pbucket->plinkFirst);
				}

				m_plinkLast = pbucket->plinkLast;
			}
		}


	}

	// ----------------------------------------------------------------------------------------------------
	// clear all of the buckets.  normally done in preparation for a sort
	//
	void SortedBlockList::InitBuckets(void)
    {
        for (int iBucket = 0; iBucket < m_iBuckets; iBucket++)
        {
            Bucket *pbucket = &m_pabucket[iBucket];

            pbucket->plinkFirst = 0;
            pbucket->plinkLast = 0;
        }
    }

    // ----------------------------------------------------------------------------------------------------
    // print out the list of sorted 4x4 blocks
	// normally used for debugging
	//
    void SortedBlockList::Print(void)
    {
        for (int iBucket = m_iBuckets-1; iBucket >= 0; iBucket--)
        {
            Bucket *pbucket = &m_pabucket[iBucket];

            unsigned int uiBlocks = 0;
            for (Link *plink = pbucket->plinkFirst; plink != nullptr; plink = plink->GetNext() )
            {
                uiBlocks++;

				if (plink == pbucket->plinkLast)
				{
					break;
				}
            }

            float fBucketError = m_fMaxError * iBucket / m_iBuckets;
            float fBucketRMS = sqrtf(fBucketError / (4.0f*16.0f) );
            printf("%3d: e=%.3f rms=%.6f %u\n", iBucket, fBucketError, fBucketRMS, uiBlocks);
        }
    }

    // ----------------------------------------------------------------------------------------------------
    //

}   // namespace Etc
