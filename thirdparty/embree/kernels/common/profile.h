// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "default.h"

namespace embree
{
  /*! helper structure for the implementation of the profile functions below */
  struct ProfileTimer
  {
    static const size_t N = 20;
    
    ProfileTimer () {}

    ProfileTimer (const size_t numSkip) : i(0), j(0), maxJ(0), numSkip(numSkip), t0(0)
    {
      for (size_t i=0; i<N; i++) names[i] = nullptr;
      for (size_t i=0; i<N; i++) dt_fst[i] = 0.0;
      for (size_t i=0; i<N; i++) dt_min[i] = pos_inf;
      for (size_t i=0; i<N; i++) dt_avg[i] = 0.0;
      for (size_t i=0; i<N; i++) dt_max[i] = neg_inf;
    }
    
    __forceinline void begin() 
    {
      j=0;
      t0 = tj = getSeconds();
    }

    __forceinline void end() {
      absolute("total");
      i++;
    }

    __forceinline void operator() (const char* name) {
      relative(name);
    }

    __forceinline void absolute (const char* name) 
    {
      const double t1 = getSeconds();
      const double dt = t1-t0;
      assert(names[j] == nullptr || names[j] == name);
      names[j] = name;
      if (i == 0) dt_fst[j] = dt;
      if (i>=numSkip) {
        dt_min[j] = min(dt_min[j],dt);
        dt_avg[j] = dt_avg[j] + dt;
        dt_max[j] = max(dt_max[j],dt);
      }
      j++;
      maxJ = max(maxJ,j);
    }

    __forceinline void relative (const char* name) 
    {
      const double t1 = getSeconds();
      const double dt = t1-tj;
      tj = t1;
      assert(names[j] == nullptr || names[j] == name);
      names[j] = name;
      if (i == 0) dt_fst[j] = dt;
      if (i>=numSkip) {
        dt_min[j] = min(dt_min[j],dt);
        dt_avg[j] = dt_avg[j] + dt;
        dt_max[j] = max(dt_max[j],dt);
      }
      j++;
      maxJ = max(maxJ,j);
    }

    void print(size_t numElements) 
    {
      for (size_t k=0; k<N; k++) 
        dt_avg[k] /= double(i-numSkip);

      printf("  profile [M/s]:\n");
      for (size_t j=0; j<maxJ; j++)
        printf("%20s:  fst = %7.2f M/s, min = %7.2f M/s, avg = %7.2f M/s, max = %7.2f M/s\n",
               names[j],numElements/dt_fst[j]*1E-6,numElements/dt_max[j]*1E-6,numElements/dt_avg[j]*1E-6,numElements/dt_min[j]*1E-6);

      printf("  profile [ms]:\n");
      for (size_t j=0; j<maxJ; j++) 
        printf("%20s:  fst = %7.2f ms, min = %7.2f ms, avg = %7.2f ms, max = %7.2fms\n",
               names[j],1000.0*dt_fst[j],1000.0*dt_min[j],1000.0*dt_avg[j],1000.0*dt_max[j]);
    }

    void print() 
    {
      printf("  profile:\n");

      for (size_t k=0; k<N; k++) 
        dt_avg[k] /= double(i-numSkip);

      for (size_t j=0; j<maxJ; j++) {
        printf("%20s:  fst = %7.2f ms, min = %7.2f ms, avg = %7.2f ms, max = %7.2fms\n",
               names[j],1000.0*dt_fst[j],1000.0*dt_min[j],1000.0*dt_avg[j],1000.0*dt_max[j]);
      }
    }

    double avg() {
      return dt_avg[maxJ-1]/double(i-numSkip);
    }
    
  private:
    size_t i;
    size_t j;
    size_t maxJ;
    size_t numSkip;
    double t0;
    double tj;
    const char* names[N];
    double dt_fst[N];
    double dt_min[N];
    double dt_avg[N];
    double dt_max[N];
  };

  /*! This function executes some code block multiple times and measured sections of it. 
      Use the following way:

      profile(1,10,1000,[&](ProfileTimer& timer) {
        // code
        timer("A");
        // code 
        timer("B");
      });
  */
  template<typename Closure>
    void profile(const size_t numSkip, const size_t numIter, const size_t numElements, const Closure& closure) 
    {
      ProfileTimer timer(numSkip);
      
      for (size_t i=0; i<numSkip+numIter; i++) 
      {
        timer.begin();
	closure(timer);
        timer.end();
      }
      timer.print(numElements);
    }

  /*! similar as the function above, but the timer object comes externally */
  template<typename Closure>
    void profile(ProfileTimer& timer, const size_t numSkip, const size_t numIter, const size_t numElements, const Closure& closure) 
    {
      timer = ProfileTimer(numSkip);
      
      for (size_t i=0; i<numSkip+numIter; i++) 
      {
        timer.begin();
	closure(timer);
        timer.end();
      }
      timer.print(numElements);
    }
}
