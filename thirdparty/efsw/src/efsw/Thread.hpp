#ifndef EFSW_THREAD_HPP
#define EFSW_THREAD_HPP

#include <efsw/base.hpp>
#include <functional>
#include <memory>
#include <thread>

namespace efsw {

/** @brief Thread manager class */
class Thread {
  public:

	Thread(std::function<void()> fun)
  : mFun{std::move(fun)}
  {
 	}

	~Thread()
	{
		wait();
	}

	/** Launch the thread */
	void launch()
	{
		if (!mThread)
			mThread.reset(new std::thread{std::move(mFun)});
	}

	/** Wait the thread until end */
	void wait()
	{
		if (mThread)
		{
			mThread->join();
			mThread.reset();
		}
	}
private:

	std::unique_ptr<std::thread> mThread;
	std::function<void()> mFun;
};

} // namespace efsw

#endif
