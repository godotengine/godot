// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2025 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

/**
 * @brief Functions and data declarations for the outer context.
 *
 * The outer context includes thread-pool management, which is slower to
 * compile due to increased use of C++ stdlib. The inner context used in the
 * majority of the codec library does not include this.
 */

#ifndef ASTCENC_INTERNAL_ENTRY_INCLUDED
#define ASTCENC_INTERNAL_ENTRY_INCLUDED

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>

#include "astcenc_internal.h"

/* ============================================================================
  Parallel execution control
============================================================================ */

/**
 * @brief A simple counter-based manager for parallel task execution.
 *
 * The task processing execution consists of:
 *
 *     * A single-threaded init stage.
 *     * A multi-threaded processing stage.
 *     * A condition variable so threads can wait for processing completion.
 *
 * The init stage will be executed by the first thread to arrive in the critical section, there is
 * no main thread in the thread pool.
 *
 * The processing stage uses dynamic dispatch to assign task tickets to threads on an on-demand
 * basis. Threads may each therefore executed different numbers of tasks, depending on their
 * processing complexity. The task queue and the task tickets are just counters; the caller must map
 * these integers to an actual processing partition in a specific problem domain.
 *
 * The exit wait condition is needed to ensure processing has finished before a worker thread can
 * progress to the next stage of the pipeline. Specifically a worker may exit the processing stage
 * because there are no new tasks to assign to it while other worker threads are still processing.
 * Calling @c wait() will ensure that all other worker have finished before the thread can proceed.
 *
 * The basic usage model:
 *
 *     // --------- From single-threaded code ---------
 *
 *     // Reset the tracker state
 *     manager->reset()
 *
 *     // --------- From multi-threaded code ---------
 *
 *     // Run the stage init; only first thread actually runs the lambda
 *     manager->init(<lambda>)
 *
 *     do
 *     {
 *         // Request a task assignment
 *         uint task_count;
 *         uint base_index = manager->get_tasks(<granule>, task_count);
 *
 *         // Process any tasks we were given (task_count <= granule size)
 *         if (task_count)
 *         {
 *             // Run the user task processing code for N tasks here
 *             ...
 *
 *             // Flag these tasks as complete
 *             manager->complete_tasks(task_count);
 *         }
 *     } while (task_count);
 *
 *     // Wait for all threads to complete tasks before progressing
 *     manager->wait()
 *
  *     // Run the stage term; only first thread actually runs the lambda
 *     manager->term(<lambda>)
 */
class ParallelManager
{
private:
	/** @brief Lock used for critical section and condition synchronization. */
	std::mutex m_lock;

	/** @brief True if the current operation is cancelled. */
	std::atomic<bool> m_is_cancelled;

	/** @brief True if the stage init() step has been executed. */
	bool m_init_done;

	/** @brief True if the stage term() step has been executed. */
	bool m_term_done;

	/** @brief Condition variable for tracking stage processing completion. */
	std::condition_variable m_complete;

	/** @brief Number of tasks started, but not necessarily finished. */
	std::atomic<unsigned int> m_start_count;

	/** @brief Number of tasks finished. */
	unsigned int m_done_count;

	/** @brief Number of tasks that need to be processed. */
	unsigned int m_task_count;

	/** @brief Progress callback (optional). */
	astcenc_progress_callback m_callback;

	/** @brief Lock used for callback synchronization. */
	std::mutex m_callback_lock;

	/** @brief Minimum progress before making a callback. */
	float m_callback_min_diff;

	/** @brief Last progress callback value. */
	float m_callback_last_value;

public:
	/** @brief Create a new ParallelManager. */
	ParallelManager()
	{
		reset();
	}

	/**
	 * @brief Reset the tracker for a new processing batch.
	 *
	 * This must be called from single-threaded code before starting the multi-threaded processing
	 * operations.
	 */
	void reset()
	{
		m_init_done = false;
		m_term_done = false;
		m_is_cancelled = false;
		m_start_count = 0;
		m_done_count = 0;
		m_task_count = 0;
		m_callback = nullptr;
		m_callback_last_value = 0.0f;
		m_callback_min_diff = 1.0f;
	}

	/**
	 * @brief Clear the tracker and stop new tasks being assigned.
	 *
	 * Note, all in-flight tasks in a worker will still complete normally.
	 */
	void cancel()
	{
		m_is_cancelled = true;
	}

	/**
	 * @brief Trigger the pipeline stage init step.
	 *
	 * This can be called from multi-threaded code. The first thread to hit this will process the
	 * initialization. Other threads will block and wait for it to complete.
	 *
	 * @param init_func   Callable which executes the stage initialization. It must return the
	 *                    total number of tasks in the stage.
	 */
	void init(std::function<unsigned int(void)> init_func)
	{
		std::lock_guard<std::mutex> lck(m_lock);
		if (!m_init_done)
		{
			m_task_count = init_func();
			m_init_done = true;
		}
	}

	/**
	 * @brief Trigger the pipeline stage init step.
	 *
	 * This can be called from multi-threaded code. The first thread to hit this will process the
	 * initialization. Other threads will block and wait for it to complete.
	 *
	 * @param task_count   Total number of tasks needing processing.
	 * @param callback     Function pointer for progress status callbacks.
	 */
	void init(unsigned int task_count, astcenc_progress_callback callback)
	{
		std::lock_guard<std::mutex> lck(m_lock);
		if (!m_init_done)
		{
			m_callback = callback;
			m_task_count = task_count;
			m_init_done = true;

			// Report every 1% or 4096 blocks, whichever is larger, to avoid callback overhead
			float min_diff = (4096.0f / static_cast<float>(task_count)) * 100.0f;
			m_callback_min_diff = astc::max(min_diff, 1.0f);
		}
	}

	/**
	 * @brief Request a task assignment.
	 *
	 * Assign up to @c granule tasks to the caller for processing.
	 *
	 * @param      granule   Maximum number of tasks that can be assigned.
	 * @param[out] count     Actual number of tasks assigned, or zero if no tasks were assigned.
	 *
	 * @return Task index of the first assigned task; assigned tasks increment from this.
	 */
	unsigned int get_task_assignment(unsigned int granule, unsigned int& count)
	{
		unsigned int base = m_start_count.fetch_add(granule, std::memory_order_relaxed);
		if (m_is_cancelled || base >= m_task_count)
		{
			count = 0;
			return 0;
		}

		count = astc::min(m_task_count - base, granule);
		return base;
	}

	/**
	 * @brief Complete a task assignment.
	 *
	 * Mark @c count tasks as complete. This will notify all threads blocked on @c wait() if this
	 * completes the processing of the stage.
	 *
	 * @param count   The number of completed tasks.
	 */
	void complete_task_assignment(unsigned int count)
	{
		// Note: m_done_count cannot use an atomic without the mutex; this has a race between the
		// update here and the wait() for other threads
		unsigned int local_count;
		float local_last_value;
		{
			std::unique_lock<std::mutex> lck(m_lock);
			m_done_count += count;
			local_count = m_done_count;
			local_last_value = m_callback_last_value;

			// Ensure the progress bar hits 100%
			if (m_callback && m_done_count == m_task_count)
			{
				std::unique_lock<std::mutex> cblck(m_callback_lock);
				m_callback(100.0f);
				m_callback_last_value = 100.0f;
			}

			// Notify if nothing left to do
			if (m_is_cancelled || m_done_count == m_task_count)
			{
				lck.unlock();
				m_complete.notify_all();
			}
		}

		// Process progress callback if we have one
		if (m_callback)
		{
			// Initial lockless test - have we progressed enough to emit?
			float num = static_cast<float>(local_count);
			float den = static_cast<float>(m_task_count);
			float this_value =  (num / den) * 100.0f;
			bool report_test = (this_value - local_last_value) > m_callback_min_diff;

			// Recheck under lock, because another thread might report first
			if (report_test)
			{
				std::unique_lock<std::mutex> cblck(m_callback_lock);
				bool report_retest = (this_value - m_callback_last_value) > m_callback_min_diff;
				if (report_retest)
				{
					m_callback(this_value);
					m_callback_last_value = this_value;
				}
			}
		}
	}

	/**
	 * @brief Wait for stage processing to complete.
	 */
	void wait()
	{
		std::unique_lock<std::mutex> lck(m_lock);
		m_complete.wait(lck, [this]{ return m_is_cancelled || m_done_count == m_task_count; });
	}

	/**
	 * @brief Trigger the pipeline stage term step.
	 *
	 * This can be called from multi-threaded code. The first thread to hit this will process the
	 * work pool termination. Caller must have called @c wait() prior to calling this function to
	 * ensure that processing is complete.
	 *
	 * @param term_func   Callable which executes the stage termination.
	 */
	void term(std::function<void(void)> term_func)
	{
		std::lock_guard<std::mutex> lck(m_lock);
		if (!m_term_done)
		{
			term_func();
			m_term_done = true;
		}
	}
};

/**
 * @brief The astcenc compression context.
 */
struct astcenc_context
{
	/** @brief The context internal state. */
	astcenc_contexti context;

#if !defined(ASTCENC_DECOMPRESS_ONLY)
	/** @brief The parallel manager for averages computation. */
	ParallelManager manage_avg;

	/** @brief The parallel manager for compression. */
	ParallelManager manage_compress;
#endif

	/** @brief The parallel manager for decompression. */
	ParallelManager manage_decompress;
};

#endif
