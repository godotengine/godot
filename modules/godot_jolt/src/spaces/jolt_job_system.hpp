#pragma once
#include "../common.h"
#include "containers/free_list.hpp"
class JoltJobSystem final : public JPH::JobSystemWithBarrier {
public:
	JoltJobSystem();

	void pre_step();

	void post_step();

#ifdef GDJ_CONFIG_EDITOR
	void flush_timings();
#endif // GDJ_CONFIG_EDITOR

private:
	class Job : public JPH::JobSystem::Job {
	public:
		Job(const char* p_name,
			JPH::ColorArg p_color,
			JPH::JobSystem* p_job_system,
			const JPH::JobSystem::JobFunction& p_job_function,
			JPH::uint32 p_dependency_count);

		Job(const Job& p_other) = delete;

		Job(Job&& p_other) = delete;

		~Job();

		static void push_completed(Job* p_job);

		static Job* pop_completed();

		void queue();

		Job& operator=(const Job& p_other) = delete;

		Job& operator=(Job&& p_other) = delete;

	private:
		static void _execute(void* p_user_data);

		inline static std::atomic<Job*> completed_head = nullptr;

#ifdef GDJ_CONFIG_EDITOR
		const char* name = nullptr;
#endif // GDJ_CONFIG_EDITOR

		int64_t task_id = -1;

		std::atomic<Job*> completed_next = nullptr;
	};

	int GetMaxConcurrency() const override;

	JPH::JobHandle CreateJob(
		const char* p_name,
		JPH::ColorArg p_color,
		const JPH::JobSystem::JobFunction& p_job_function,
		JPH::uint32 p_dependency_count = 0
	) override;

	void QueueJob(JPH::JobSystem::Job* p_job) override;

	void QueueJobs(JPH::JobSystem::Job** p_jobs, JPH::uint p_job_count) override;

	void FreeJob(JPH::JobSystem::Job* p_job) override;

	void _reclaim_jobs();

#ifdef GDJ_CONFIG_EDITOR
	// HACK(mihe): We use `const void*` here to avoid the cost of hashing the actual string, since
	// the job names are always literals and as such will point to the same address every time.
	inline static HashMap<const void*, uint64_t> timings_by_job;

	inline static SpinLock timings_lock;
#endif // GDJ_CONFIG_EDITOR

	FreeList<Job> jobs;

	int32_t thread_count = 0;
};
