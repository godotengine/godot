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

#pragma once

namespace Etc
{
	class Block4x4;

    class SortedBlockList
    {
    public:

		class Link
		{
		public:

			inline void Init(Block4x4 *a_pblock)
			{
				m_pblock = a_pblock;
				m_plinkNext = nullptr;
			}

			inline Block4x4 * GetBlock(void)
			{
				return m_pblock;
			}

			inline void SetNext(Link *a_plinkNext)
			{
				m_plinkNext = a_plinkNext;
			}

			inline Link * GetNext(void)
			{
				return m_plinkNext;
			}

			inline Link * Advance(unsigned int a_uiSteps = 1)
			{
				Link *plink = this;

				for (unsigned int uiStep = 0; uiStep < a_uiSteps; uiStep++)
				{
					if (plink == nullptr)
					{
						break;
					}

					plink = plink->m_plinkNext;
				}

				return plink;
			}

		private:

			Block4x4 *m_pblock;
			Link *m_plinkNext;
		};

		SortedBlockList(unsigned int a_uiImageBlocks, unsigned int a_uiBuckets);
		~SortedBlockList(void);

        void AddBlock(Block4x4 *a_pblock);

        void Sort(void);

		inline Link * GetLinkToFirstBlock(void)
		{
			return m_plinkFirst;
		}

		inline unsigned int GetNumberOfAddedBlocks(void)
		{
			return m_uiAddedBlocks;
		}

		inline unsigned int GetNumberOfSortedBlocks(void)
		{
			return m_uiSortedBlocks;
		}

		void Print(void);

	private:

        void InitBuckets(void);

        class Bucket
        {
        public:
            Link *plinkFirst;
            Link *plinkLast;
        };

        unsigned int m_uiImageBlocks;
        int m_iBuckets;

		unsigned int m_uiAddedBlocks;
		unsigned int m_uiSortedBlocks;
		Link *m_palinkPool;
        Bucket *m_pabucket;
        float m_fMaxError;

		Link *m_plinkFirst;
		Link *m_plinkLast;

    };

} // namespace Etc
