#ifndef FOLIAGE_MEMORY_POOL_H
#define FOLIAGE_MEMORY_POOL_H
#include "core/templates/hash_map.h"

namespace Foliage
{
    // 内存分配器
    class MemoryPool
    {
        public:
        // 内存块结构体
        class Block
        {
            public:
            uint32_t offset = 0;
            uint32_t size = 0;      // 内存块大小
            bool available = false; // 内存块是否可用（true：可用，false：不可用）
            Block* next = nullptr;   // 指向链表中下一个内存块的指针
            static Block* s_freeBlock;
            int Start()
            {
                return offset;
            }
            int End()
            {
                return offset + size;
                
            }
            static Block* Allocal()
            {
                if (s_freeBlock != nullptr)
                {
                    Block* r = s_freeBlock;
                    s_freeBlock = s_freeBlock->next;
                    r->next = nullptr;
                    return r;
                }
                return memnew(Block);
            }
            static void Free(Block* _block)
            {
                _block->next = s_freeBlock;
                s_freeBlock = _block;
            }
        };
        // 构造函数，初始化内存池
        MemoryPool(int total_size)
        {
            m_totalMemorySize = 0;
            AddFreeBlock(total_size);
        }

        // 析构函数，释放内存池和Block信息
        void Release()
        {
            for (auto& entry : m_block_map)
            {
                Block::Free(entry.value);
            }
            m_block_map.clear();
        }
        Block* AddFreeBlock(int count = 256)
        {
            Block* first_block = Block::Allocal();
            first_block->available = true;
            first_block->offset = m_totalMemorySize;
            first_block->size = count;
            m_block_map.insert(first_block->offset, first_block);
            m_totalMemorySize += count;

            return first_block;

        }

        // Best Fit 分配算法
        // 注意如果_auto_addcount 不为0，返回的区间有可能自动分配，超出当前的区间
        Block* Allocate(uint32_t requested_size, uint32_t _auto_addcount = 0)
        {
            if (requested_size <= 0)
            {
                //Debug.LogError($"无法分配{requested_size}内存，数量非法！");
                return nullptr;
            }
            //Block previous = null;
            Block* best_fit_block = nullptr;
            Block* current_block = nullptr;

            uint32_t min_gap = 2147483647;

            // 遍历已释放内存块列表
            uint32_t current_gap;
            uint32_t remaining_size = 0;
            for (auto& entry : m_block_map)
            {
                current_block = entry.value;
                if (current_block->available && current_block->size >= requested_size)
                {
                    current_gap = current_block->size - requested_size;
                    if (current_gap < 0)
                        continue;
                    if (current_gap < min_gap)
                    {
                        min_gap = current_gap;
                        best_fit_block = current_block;
                        remaining_size = current_gap;
                    }
                    // 找到了最佳大小
                    if (current_gap == 0)
                    {
                        break;
                    }
                }
            }

            // 若找到合适的内存块
            if (best_fit_block != nullptr)
            {
                // 若分配后剩余大小足够容纳一个新的内存块
                if (remaining_size > 0)
                {
                    // 更新当前最佳匹配内存块的信息
                    best_fit_block->size = requested_size;
                    best_fit_block->available = false;

                    // 在空闲链表中添加一个新的内存块
                    Block* remaining_block = Block::Allocal();
                    remaining_block->offset = best_fit_block->offset + requested_size;
                    // 设置大小为剩余大小
                    remaining_block->size = remaining_size;
                    remaining_block->available = true;
                    remaining_block->next = nullptr;
                    if (remaining_block->offset < 0)
                    {
                        //Debug.LogError("纳尼！有毛病！");
                    }
                    m_block_map.insert(remaining_block->offset, remaining_block);

                }
                else
                {
                    // 标记已分配内存块为不可用
                    best_fit_block->available = false;

                }
                return best_fit_block;
            }
            if (_auto_addcount >= 0)
            {
                // 还没有找到，并且允许自动分配就新增一个内存块
                _auto_addcount = MAX(requested_size, _auto_addcount);
                AddFreeBlock(_auto_addcount);
                return Allocate(requested_size);
            }
            // 如果没有找到合适的内存块，返回nullptr
            return nullptr;
        }
        // 减少内存
        Block* ReductionMemory(Block* block, uint32_t reduction_count)
        {
            if (reduction_count == 0)
            {
                return block;
            }
            if (reduction_count > block->size)
            {
                //Debug.LogError("你这是要掘地三尺的减少呀！");
                return block;
            }
            if (!m_block_map.has(block->Start()))
            {
                //Debug.LogError("发现外来物种入侵，请弄死他！");
                return block;
            }

            Block* free = Block::Allocal();
            block->size -= reduction_count;
            free->offset = block->End();
            free->size = reduction_count;
            free->available = false;
            m_block_map.insert(free->Start(), free);
            FreeMemory(free);
            return block;
        }
        void FreeMemory(int block_offset)
        {
            auto it = m_block_map.find(block_offset);
            if (it != m_block_map.end())
            {
                FreeMemory(it->value);

            }
            else
            {
                //Debug.LogError("非法索引！block_offset 不存在！");
            }

        }

        // 释放内存块
        void FreeMemory(Block* freed_block)
        {
            if (freed_block->available == true)
            {
                return;
            }
            freed_block->available = true;


            //Block current_block = freed_block;
            // 查找是否存在可以链接的下一个空闲区域
            int current_memory = freed_block.offset;
            //int next_memory = freed_block.End;
            Block next_block_entry;
            int nextKey = freed_block.End;
            auto it = m_block_map.find(nextKey);
            if (it != m_block_map.end())
            {
                auto next_block_entry = it->value;
                Block* next_block = next_block_entry;
                if (next_block->available)
                {
                    // 合并相邻的内存块
                    freed_block->size += next_block->size;


                    // 从映射表和堆内存中删除被合并的相邻内存块
                    m_block_map.erase(nextKey);
                    Block::Free(next_block);
                }
            }

            Block* current_block = nullptr;
            // 合并上一个空白区域
            // 如果可用就自动合并
            for (auto& entry : m_block_map)
            {
                current_block = entry.value;
                // 判断是否是当前块的下一个节点是否为当前需要释放的节点
                if (current_block->End == freed_block.Start)
                {
                    if (!current_block->available)
                    {
                        // 不可用直接返回
                        break;
                    }

                    current_block->size += freed_block->size;
                    Block::Free(freed_block);
                    m_block_map.erase(freed_block->offset);
                    break;
                }
            }

        }
        private:
        int m_totalMemorySize;
        HashMap<int, Block*> m_block_map; // 保存内存地址和Block信息的映射表
    };

}
#endif