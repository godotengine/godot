#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <random>
#endif

#include <algorithm>
#include <numeric>

#include <map>
#include <string>
#include <exception>

#include <iostream>

#include <cassert>
#include <cstdlib>    // For `atoi`.
#include <climits>    // For CHAR_BIT.
#include <cmath>      // For `sqrt` and `abs`.

#include <stdint.h>   // For `intN_t`.

#include "random.h"
#include "timer.h"

#if defined(HAVE_TBB)
  #include "tbb_algos.h"
#endif

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  #include <thrust/system_error.h>      // For `thrust::system_error`
  #include <thrust/system/cuda/error.h> // For `thrust::cuda_category`
#endif

// We don't use THRUST_PP_STRINGIZE and THRUST_PP_CAT because they are new, and
// we want this benchmark to be backwards-compatible to older versions of Thrust.
#define PP_STRINGIZE_(expr) #expr
#define PP_STRINGIZE(expr)  PP_STRINGIZE_(expr)

#define PP_CAT(a, b) a ## b

// We don't use THRUST_NOEXCEPT because it's new, and we want this benchmark to
// be backwards-compatible to older versions of Thrust.
#if THRUST_CPP_DIALECT >= 2011
  #define NOEXCEPT noexcept
#else
  #define NOEXCEPT throw()
#endif

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct squared_difference
{
private:
  T const average;

public:
  __host__ __device__
  squared_difference(squared_difference const& rhs) : average(rhs.average) {}

  __host__ __device__
  squared_difference(T average_) : average(average_) {}

  __host__ __device__
  T operator()(T x) const
  {
    return (x - average) * (x - average);
  }
};

template <typename T>
struct value_and_count
{
  T           value;
  uint64_t count;

  __host__ __device__
  value_and_count(value_and_count const& other)
    : value(other.value), count(other.count) {}

  __host__ __device__
  value_and_count(T const& value_)
    : value(value_), count(1) {}

  __host__ __device__
  value_and_count(T const& value_, uint64_t count_)
    : value(value_), count(count_) {}

  __host__ __device__
  value_and_count& operator=(value_and_count const& other)
  {
    value = other.value;
    count = other.count;
    return *this;
  }

  __host__ __device__
  value_and_count& operator=(T const& value_)
  {
    value = value_;
    count = 1;
    return *this;
  }
};

template <typename T, typename ReduceOp>
struct counting_op
{
private:
  ReduceOp reduce;

public:
  __host__ __device__
  counting_op() : reduce() {}

  __host__ __device__
  counting_op(counting_op const& other) : reduce(other.reduce) {}

  __host__ __device__
  counting_op(ReduceOp const& reduce_) : reduce(reduce_) {}

  __host__ __device__
  value_and_count<T> operator()(
      value_and_count<T> const& x
    , T const&                  y
    ) const
  {
    return value_and_count<T>(reduce(x.value, y), x.count + 1);
  }

  __host__ __device__
  value_and_count<T> operator()(
      value_and_count<T> const& x
    , value_and_count<T> const& y
    ) const
  {
    return value_and_count<T>(reduce(x.value, y.value), x.count + y.count);
  }
};

template <typename InputIt, typename T>
T arithmetic_mean(InputIt first, InputIt last, T init)
{
  value_and_count<T> init_vc(init, 0);

  counting_op<T, thrust::plus<T> > reduce_vc;

  value_and_count<T> vc
    = thrust::reduce(first, last, init_vc, reduce_vc);

  return vc.value / vc.count;
}

template <typename InputIt>
typename thrust::iterator_traits<InputIt>::value_type
arithmetic_mean(InputIt first, InputIt last)
{
  typedef typename thrust::iterator_traits<InputIt>::value_type T;
  return arithmetic_mean(first, last, T());
}

template <typename InputIt, typename T>
T sample_standard_deviation(InputIt first, InputIt last, T average)
{
  value_and_count<T> init_vc(T(), 0);

  counting_op<T, thrust::plus<T> > reduce_vc;

  squared_difference<T> transform(average);

  value_and_count<T> vc
    = thrust::transform_reduce(first, last, transform, init_vc, reduce_vc);

  return std::sqrt(vc.value / T(vc.count - 1));
}

///////////////////////////////////////////////////////////////////////////////

// Formulas for propagation of uncertainty from:
//
//   https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulas
//
// Even though it's Wikipedia, I trust it as I helped write that table.
//
// XXX Replace with a proper reference.

// Compute the propagated uncertainty from the multiplication of two uncertain
// values, `A +/- A_unc` and `B +/- B_unc`. Given `f = AB` or `f = A/B`, where
// `A != 0` and `B != 0`, the uncertainty in `f` is approximately:
//
//   f_unc = abs(f) * sqrt((A_unc / A) ^ 2 + (B_unc / B) ^ 2)
//
template <typename T>
__host__ __device__
T uncertainty_multiplicative(
    T const& f
  , T const& A, T const& A_unc
  , T const& B, T const& B_unc
    )
{
  return std::abs(f)
       * std::sqrt((A_unc / A) * (A_unc / A) + (B_unc / B) * (B_unc / B));
}

// Compute the propagated uncertainty from addition of two uncertain values,
// `A +/- A_unc` and `B +/- B_unc`. Given `f = cA + dB` (where `c` and `d` are
// certain constants), the uncertainty in `f` is approximately:
//
//   f_unc = sqrt(c ^ 2 * A_unc ^ 2 + d ^ 2 * B_unc ^ 2)
//
template <typename T>
__host__ __device__
T uncertainty_additive(
    T const& c, T const& A_unc
  , T const& d, T const& B_unc
    )
{
  return std::sqrt((c * c * A_unc * A_unc) + (d * d * B_unc * B_unc));
}

///////////////////////////////////////////////////////////////////////////////

// Return the significant digit of `x`. The result is the number of digits
// after the decimal place to round to (negative numbers indicate rounding
// before the decimal place)
template <typename T>
int find_significant_digit(T x)
{
  if (x == T(0)) return T(0);
  return -int(std::floor(std::log10(std::abs(x))));
}

// Round `x` to `ndigits` after the decimal place (Python-style).
template <typename T, typename N>
T round_to_precision(T x, N ndigits)
{
  double m = (x < 0.0) ? -1.0 : 1.0;
  double pwr = std::pow(T(10.0), ndigits);
  return (std::floor(x * m * pwr + 0.5) / pwr) * m;
}

///////////////////////////////////////////////////////////////////////////////

void print_experiment_header()
{ // {{{
  std::cout << "Thrust Version"
    << ","  << "Algorithm"
    << ","  << "Element Type"
    << ","  << "Element Size"
    << ","  << "Elements per Trial"
    << ","  << "Total Input Size"
    << ","  << "STL Trials"
    << ","  << "STL Average Walltime"
    << ","  << "STL Walltime Uncertainty"
    << ","  << "STL Average Throughput"
    << ","  << "STL Throughput Uncertainty"
    << ","  << "Thrust Trials"
    << ","  << "Thrust Average Walltime"
    << ","  << "Thrust Walltime Uncertainty"
    << ","  << "Thrust Average Throughput"
    << ","  << "Thrust Throughput Uncertainty"
    #if defined(HAVE_TBB)
    << ","  << "TBB Trials"
    << ","  << "TBB Average Walltime"
    << ","  << "TBB Walltime Uncertainty"
    << ","  << "TBB Average Throughput"
    << ","  << "TBB Throughput Uncertainty"
    #endif
    << std::endl;

  std::cout << ""                // Thrust Version.
    << ","  << ""                // Algorithm.
    << ","  << ""                // Element Type.
    << ","  << "bits/element"    // Element Size.
    << ","  << "elements"        // Elements per Trial.
    << ","  << "MiBs"            // Total Input Size.
    << ","  << "trials"          // STL Trials.
    << ","  << "secs"            // STL Average Walltime.
    << ","  << "secs"            // STL Walltime Uncertainty.
    << ","  << "elements/sec"    // STL Average Throughput.
    << ","  << "elements/sec"    // STL Throughput Uncertainty.
    << ","  << "trials"          // Thrust Trials.
    << ","  << "secs"            // Thrust Average Walltime.
    << ","  << "secs"            // Thrust Walltime Uncertainty.
    << ","  << "elements/sec"    // Thrust Average Throughput.
    << ","  << "elements/sec"    // Thrust Throughput Uncertainty.
    #if defined(HAVE_TBB)
    << ","  << "trials"          // TBB Trials.
    << ","  << "secs"            // TBB Average Walltime.
    << ","  << "secs"            // TBB Walltime Uncertainty.
    << ","  << "elements/sec"    // TBB Average Throughput.
    << ","  << "elements/sec"    // TBB Throughput Uncertainty.
    #endif
    << std::endl;
} // }}}

///////////////////////////////////////////////////////////////////////////////

struct experiment_results
{
  double const average_time; // Arithmetic mean of trial times in seconds.
  double const stdev_time;   // Sample standard deviation of trial times.

  experiment_results(double average_time_, double stdev_time_)
    : average_time(average_time_), stdev_time(stdev_time_) {}
};

///////////////////////////////////////////////////////////////////////////////

template <
    template <typename> class Test
  , typename                  ElementMetaType // Has an embedded typedef `type,
                                              // and a static method `name` that
                                              // returns a char const*.
  , uint64_t                  Elements
  , uint64_t                  BaselineTrials
  , uint64_t                  RegularTrials
>
struct experiment_driver
{
  typedef typename ElementMetaType::type element_type;

  static char const* const test_name;
  static char const* const element_type_name; // Element type name as a string.

  static uint64_t const elements;             // # of elements per trial.
  static uint64_t const element_size;         // Size of each element in bits.
  static double   const input_size;           // `elements` * `element_size` in MiB.
  static uint64_t const baseline_trials;      // # of baseline trials per experiment.
  static uint64_t const regular_trials;       // # of regular trials per experiment.

  static void run_experiment()
  { // {{{
    experiment_results stl    = std_experiment();
    experiment_results thrust = thrust_experiment();
    #if defined(HAVE_TBB)
    experiment_results tbb    = tbb_experiment();
    #endif

    double stl_average_walltime    = stl.average_time;
    double thrust_average_walltime = thrust.average_time;
    #if defined(HAVE_TBB)
    double tbb_average_walltime    = tbb.average_time;
    #endif

    double stl_average_throughput    = elements / stl.average_time;
    double thrust_average_throughput = elements / thrust.average_time;
    #if defined(HAVE_TBB)
    double tbb_average_throughput    = elements / tbb.average_time;
    #endif

    double stl_walltime_uncertainty    = stl.stdev_time;
    double thrust_walltime_uncertainty = thrust.stdev_time;
    #if defined(HAVE_TBB)
    double tbb_walltime_uncertainty    = tbb.stdev_time;
    #endif

    double stl_throughput_uncertainty    = uncertainty_multiplicative(
        stl_average_throughput
      , double(elements), 0.0
      , stl_average_walltime, stl_walltime_uncertainty
    );
    double thrust_throughput_uncertainty = uncertainty_multiplicative(
        thrust_average_throughput
      , double(elements), 0.0
      , thrust_average_walltime, thrust_walltime_uncertainty
    );

    #if defined(HAVE_TBB)
    double tbb_throughput_uncertainty    = uncertainty_multiplicative(
        tbb_average_throughput
      , double(elements), 0.0
      , tbb_average_walltime, tbb_walltime_uncertainty
    );
    #endif

    // Round the average walltime and walltime uncertainty to the
    // significant figure of the walltime uncertainty.
    int stl_walltime_precision = std::max(
        find_significant_digit(stl.average_time)
      , find_significant_digit(stl.stdev_time)
    );
    int thrust_walltime_precision = std::max(
        find_significant_digit(thrust.average_time)
      , find_significant_digit(thrust.stdev_time)
    );
    #if defined(HAVE_TBB)
    int tbb_walltime_precision = std::max(
        find_significant_digit(tbb.average_time)
      , find_significant_digit(tbb.stdev_time)
    );
    #endif

    stl_average_walltime = round_to_precision(
        stl_average_walltime, stl_walltime_precision
    );
    thrust_average_walltime = round_to_precision(
        thrust_average_walltime, thrust_walltime_precision
    );
    #if defined(HAVE_TBB)
    tbb_average_walltime = round_to_precision(
        tbb_average_walltime, tbb_walltime_precision
    );
    #endif

    stl_walltime_uncertainty = round_to_precision(
        stl_walltime_uncertainty, stl_walltime_precision
    );
    thrust_walltime_uncertainty = round_to_precision(
        thrust_walltime_uncertainty, thrust_walltime_precision
    );
    #if defined(HAVE_TBB)
    tbb_walltime_uncertainty = round_to_precision(
        tbb_walltime_uncertainty, tbb_walltime_precision
    );
    #endif

    // Round the average throughput and throughput uncertainty to the
    // significant figure of the throughput uncertainty.
    int stl_throughput_precision = std::max(
        find_significant_digit(stl_average_throughput)
      , find_significant_digit(stl_throughput_uncertainty)
    );
    int thrust_throughput_precision = std::max(
        find_significant_digit(thrust_average_throughput)
      , find_significant_digit(thrust_throughput_uncertainty)
    );
    #if defined(HAVE_TBB)
    int tbb_throughput_precision = std::max(
        find_significant_digit(tbb_average_throughput)
      , find_significant_digit(tbb_throughput_uncertainty)
    );
    #endif

    stl_average_throughput = round_to_precision(
        stl_average_throughput, stl_throughput_precision
    );
    thrust_average_throughput = round_to_precision(
        thrust_average_throughput, thrust_throughput_precision
    );
    #if defined(HAVE_TBB)
    tbb_average_throughput = round_to_precision(
        tbb_average_throughput, tbb_throughput_precision
    );
    #endif

    stl_throughput_uncertainty = round_to_precision(
        stl_throughput_uncertainty, stl_throughput_precision
    );
    thrust_throughput_uncertainty = round_to_precision(
        thrust_throughput_uncertainty, thrust_throughput_precision
    );
    #if defined(HAVE_TBB)
    tbb_throughput_uncertainty = round_to_precision(
        tbb_throughput_uncertainty, tbb_throughput_precision
    );
    #endif

    std::cout << THRUST_VERSION                // Thrust Version.
      << ","  << test_name                     // Algorithm.
      << ","  << element_type_name             // Element Type.
      << ","  << element_size                  // Element Size.
      << ","  << elements                      // Elements per Trial.
      << ","  << input_size                    // Total Input Size.
      << ","  << baseline_trials               // STL Trials.
      << ","  << stl_average_walltime          // STL Average Walltime.
      << ","  << stl_walltime_uncertainty      // STL Walltime Uncertainty.
      << ","  << stl_average_throughput        // STL Average Throughput.
      << ","  << stl_throughput_uncertainty    // STL Throughput Uncertainty.
      << ","  << regular_trials                // Thrust Trials.
      << ","  << thrust_average_walltime       // Thrust Average Walltime.
      << ","  << thrust_walltime_uncertainty   // Thrust Walltime Uncertainty.
      << ","  << thrust_average_throughput     // Thrust Average Throughput.
      << ","  << thrust_throughput_uncertainty // Thrust Throughput Uncertainty.
      #if defined(HAVE_TBB)
      << ","  << regular_trials                // TBB Trials.
      << ","  << tbb_average_walltime          // TBB Average Walltime.
      << ","  << tbb_walltime_uncertainty      // TBB Walltime Uncertainty.
      << ","  << tbb_average_throughput        // TBB Average Throughput.
      << ","  << tbb_throughput_uncertainty    // TBB Throughput Uncertainty.
      #endif
      << std::endl;
  } // }}}

private:
  static experiment_results std_experiment()
  {
    return experiment<typename Test<element_type>::std_trial>();
  }

  static experiment_results thrust_experiment()
  {
    return experiment<typename Test<element_type>::thrust_trial>();
  }

  #if defined(HAVE_TBB)
  static experiment_results tbb_experiment()
  {
    return experiment<typename Test<element_type>::tbb_trial>();
  }
  #endif

  template <typename Trial>
  static experiment_results experiment()
  { // {{{
    Trial trial;

    // Allocate storage and generate random input for the warmup trial.
    trial.setup(elements);

    // Warmup trial.
    trial();

    uint64_t const trials
      = trial.is_baseline() ? baseline_trials : regular_trials;

    std::vector<double> times;
    times.reserve(trials);

    for (uint64_t t = 0; t < trials; ++t)
    {
      // Generate random input for next trial.
      trial.setup(elements);

      steady_timer e;

      // Benchmark.
      e.start();
      trial();
      e.stop();

      times.push_back(e.seconds_elapsed());
    }

    double average_time
      = arithmetic_mean(times.begin(), times.end());

    double stdev_time
      = sample_standard_deviation(times.begin(), times.end(), average_time);

    return experiment_results(average_time, stdev_time);
  } // }}}
};

template <
    template <typename> class Test
  , typename                  ElementMetaType
  , uint64_t                  Elements
  , uint64_t                  BaselineTrials
  , uint64_t                  RegularTrials
>
char const* const
experiment_driver<
  Test, ElementMetaType, Elements, BaselineTrials, RegularTrials
>::test_name
  = Test<typename ElementMetaType::type>::test_name();

template <
    template <typename> class Test
  , typename                  ElementMetaType
  , uint64_t                  Elements
  , uint64_t                  BaselineTrials
  , uint64_t                  RegularTrials
>
char const* const
experiment_driver<
  Test, ElementMetaType, Elements, BaselineTrials, RegularTrials
>::element_type_name
  = ElementMetaType::name();

template <
    template <typename> class Test
  , typename                  ElementMetaType
  , uint64_t                  Elements
  , uint64_t                  BaselineTrials
  , uint64_t                  RegularTrials
>
uint64_t const
experiment_driver<
  Test, ElementMetaType, Elements, BaselineTrials, RegularTrials
>::element_size
  = CHAR_BIT * sizeof(typename ElementMetaType::type);

template <
    template <typename> class Test
  , typename                  ElementMetaType
  , uint64_t                  Elements
  , uint64_t                  BaselineTrials
  , uint64_t                  RegularTrials
>
uint64_t const
experiment_driver<
  Test, ElementMetaType, Elements, BaselineTrials, RegularTrials
>::elements
  = Elements;

template <
    template <typename> class Test
  , typename                  ElementMetaType
  , uint64_t                  Elements
  , uint64_t                  BaselineTrials
  , uint64_t                  RegularTrials
>
double const
experiment_driver<
  Test, ElementMetaType, Elements, BaselineTrials, RegularTrials
>::input_size
  = double( Elements /* [elements] */
          * sizeof(typename ElementMetaType::type) /* [bytes/element] */
          )
  / double(1024 * 1024 /* [bytes/MiB] */);

template <
    template <typename> class Test
  , typename                  ElementMetaType
  , uint64_t                  Elements
  , uint64_t                  BaselineTrials
  , uint64_t                  RegularTrials
>
uint64_t const
experiment_driver<
  Test, ElementMetaType, Elements, BaselineTrials, RegularTrials
>::baseline_trials
  = BaselineTrials;

template <
    template <typename> class Test
  , typename                  ElementMetaType
  , uint64_t                  Elements
  , uint64_t                  BaselineTrials
  , uint64_t                  RegularTrials
>
uint64_t const
experiment_driver<
  Test, ElementMetaType, Elements, BaselineTrials, RegularTrials
>::regular_trials
  = RegularTrials;

///////////////////////////////////////////////////////////////////////////////

// Never create variables, pointers or references of any of the `*_trial_base`
// classes. They are purely mixin base classes and do not have vtables and
// virtual destructors. Using them for polymorphism instead of composition will
// probably cause slicing.

struct baseline_trial {};
struct regular_trial {};

template <typename TrialKind = regular_trial>
struct trial_base;

template <>
struct trial_base<baseline_trial>
{
  static bool is_baseline() { return true; }
};

template <>
struct trial_base<regular_trial>
{
  static bool is_baseline() { return false; }
};

template <typename Container, typename TrialKind = regular_trial>
struct inplace_trial_base : trial_base<TrialKind>
{
  Container input;

  void setup(uint64_t elements)
  {
    input.resize(elements);

    randomize(input);
  }
};

template <typename Container, typename TrialKind = regular_trial>
struct copy_trial_base : trial_base<TrialKind>
{
  Container input;
  Container output;

  void setup(uint64_t elements)
  {
    input.resize(elements);
    output.resize(elements);

    randomize(input);
  }
};

#if THRUST_CPP_DIALECT >= 2011
template <typename Container, typename TrialKind = regular_trial>
struct shuffle_trial_base : trial_base<TrialKind>
{
  Container input;

  void setup(uint64_t elements)
  {
    input.resize(elements);

    randomize(input);
  }
};
#endif

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct reduce_tester
{
  static char const* test_name() { return "reduce"; }

  struct std_trial : inplace_trial_base<std::vector<T>, baseline_trial>
  {
    void operator()()
    {
      if (std::accumulate(this->input.begin(), this->input.end(), T(0)) == 0)
        // Prevent optimizer from removing body.
        std::cout << "xyz";
    }
  };

  struct thrust_trial : inplace_trial_base<thrust::device_vector<T> >
  {
    void operator()()
    {
      thrust::reduce(this->input.begin(), this->input.end());
    }
  };

  #if defined(HAVE_TBB)
  struct tbb_trial : inplace_trial_base<std::vector<T> >
  {
    void operator()()
    {
      tbb_reduce(this->input);
    }
  };
  #endif
};

template <typename T>
struct sort_tester
{
  static char const* test_name() { return "sort"; }

  struct std_trial : inplace_trial_base<std::vector<T>, baseline_trial>
  {
    void operator()()
    {
      std::sort(this->input.begin(), this->input.end());
    }
  };

  struct thrust_trial : inplace_trial_base<thrust::device_vector<T> >
  {
    void operator()()
    {
      thrust::sort(this->input.begin(), this->input.end());
      #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
          throw thrust::error_code(err, thrust::cuda_category());
      #endif
    }
  };

  #if defined(HAVE_TBB)
  struct tbb_trial : inplace_trial_base<std::vector<T> >
  {
    void operator()()
    {
      tbb_sort(this->input);
    }
  }
  #endif
};


template <typename T>
struct transform_inplace_tester
{
  static char const* test_name() { return "transform_inplace"; }

  struct std_trial : inplace_trial_base<std::vector<T>, baseline_trial>
  {
    void operator()()
    {
      std::transform(
          this->input.begin(), this->input.end(), this->input.begin()
        , thrust::negate<T>()
      );
    }
  };

  struct thrust_trial : inplace_trial_base<thrust::device_vector<T> >
  {
    void operator()()
    {
      thrust::transform(
          this->input.begin(), this->input.end(), this->input.begin()
        , thrust::negate<T>()
      );
      #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
          throw thrust::error_code(err, thrust::cuda_category());
      #endif
    }
  };

  #if defined(HAVE_TBB)
  struct tbb_trial : inplace_trial_base<std::vector<T> >
  {
    void operator()()
    {
      tbb_transform(this->input);
    }
  };
  #endif
};

template <typename T>
struct inclusive_scan_inplace_tester
{
  static char const* test_name() { return "inclusive_scan_inplace"; }

  struct std_trial : inplace_trial_base<std::vector<T>, baseline_trial>
  {
    void operator()()
    {
      std::partial_sum(
          this->input.begin(), this->input.end(), this->input.begin()
      );
    }
  };

  struct thrust_trial : inplace_trial_base<thrust::device_vector<T> >
  {
    void operator()()
    {
      thrust::inclusive_scan(
          this->input.begin(), this->input.end(), this->input.begin()
      );
      #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
          throw thrust::error_code(err, thrust::cuda_category());
      #endif
    }
  };

  #if defined(HAVE_TBB)
  struct tbb_trial : inplace_trial_base<std::vector<T> >
  {
    void operator()()
    {
      tbb_scan(this->input);
    }
  };
  #endif
};

template <typename T>
struct copy_tester
{
  static char const* test_name() { return "copy"; }

  struct std_trial : copy_trial_base<std::vector<T> >
  {
    void operator()()
    {
      std::copy(this->input.begin(), this->input.end(), this->output.begin());
    }
  };

  struct thrust_trial : copy_trial_base<thrust::device_vector<T> >
  {
    void operator()()
    {
      thrust::copy(this->input.begin(), this->input.end(), this->input.begin());
      #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
          throw thrust::error_code(err, thrust::cuda_category());
      #endif
    }
  };

  #if defined(HAVE_TBB)
  struct tbb_trial : copy_trial_base<std::vector<T> >
  {
    void operator()()
    {
      tbb_copy(this->input, this->output);
    }
  };
  #endif
};

#if THRUST_CPP_DIALECT >= 2011
template <typename T>
struct shuffle_tester
{
  static char const* test_name() { return "shuffle"; }

  struct std_trial : shuffle_trial_base<std::vector<T>, baseline_trial>
  {
    std::default_random_engine g;
    void operator()()
    {
      std::shuffle(this->input.begin(), this->input.end(), this->g);
    }
  };

  struct thrust_trial : shuffle_trial_base<thrust::device_vector<T> >
  {
    thrust::default_random_engine g;
    void operator()()
    {
      thrust::shuffle(this->input.begin(), this->input.end(), this->g);
      #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
          throw thrust::error_code(err, thrust::cuda_category());
      #endif
    }
  };
};
#endif

///////////////////////////////////////////////////////////////////////////////

template <
    typename ElementMetaType
  , uint64_t Elements
  , uint64_t BaselineTrials
  , uint64_t RegularTrials
>
void run_core_primitives_experiments_for_type()
{
  experiment_driver<
      reduce_tester
    , ElementMetaType
    , Elements / sizeof(typename ElementMetaType::type)
    , BaselineTrials
    , RegularTrials
  >::run_experiment();

  experiment_driver<
    transform_inplace_tester
    , ElementMetaType
    , Elements / sizeof(typename ElementMetaType::type)
    , BaselineTrials
    , RegularTrials
  >::run_experiment();

  experiment_driver<
      inclusive_scan_inplace_tester
    , ElementMetaType
    , Elements / sizeof(typename ElementMetaType::type)
    , BaselineTrials
    , RegularTrials
  >::run_experiment();

  experiment_driver<
      sort_tester
    , ElementMetaType
//    , Elements / sizeof(typename ElementMetaType::type)
    , (Elements >> 6) // Sorting is more sensitive to element count than
                      // memory footprint.
    , BaselineTrials
    , RegularTrials
  >::run_experiment();

  experiment_driver<
      copy_tester
    , ElementMetaType
    , Elements / sizeof(typename ElementMetaType::type)
    , BaselineTrials
    , RegularTrials
  >::run_experiment();

  experiment_driver<
      shuffle_tester
    , ElementMetaType
    , Elements / sizeof(typename ElementMetaType::type)
    , BaselineTrials
    , RegularTrials
  >::run_experiment();
}

///////////////////////////////////////////////////////////////////////////////

#define DEFINE_ELEMENT_META_TYPE(T)                       \
  struct PP_CAT(T, _meta)                                 \
  {                                                       \
    typedef T type;                                       \
                                                          \
    static char const* name() { return PP_STRINGIZE(T); } \
  };                                                      \
  /**/

DEFINE_ELEMENT_META_TYPE(char);
DEFINE_ELEMENT_META_TYPE(int);
DEFINE_ELEMENT_META_TYPE(int8_t);
DEFINE_ELEMENT_META_TYPE(int16_t);
DEFINE_ELEMENT_META_TYPE(int32_t);
DEFINE_ELEMENT_META_TYPE(int64_t);
DEFINE_ELEMENT_META_TYPE(float);
DEFINE_ELEMENT_META_TYPE(double);

///////////////////////////////////////////////////////////////////////////////

template <
    uint64_t Elements
  , uint64_t BaselineTrials
  , uint64_t RegularTrials
>
void run_core_primitives_experiments()
{
  run_core_primitives_experiments_for_type<
    char_meta,    Elements, BaselineTrials, RegularTrials
  >();
  run_core_primitives_experiments_for_type<
    int_meta,     Elements, BaselineTrials, RegularTrials
  >();
  run_core_primitives_experiments_for_type<
    int8_t_meta,  Elements, BaselineTrials, RegularTrials
  >();
  run_core_primitives_experiments_for_type<
    int16_t_meta, Elements, BaselineTrials, RegularTrials
  >();
  run_core_primitives_experiments_for_type<
    int32_t_meta, Elements, BaselineTrials, RegularTrials
  >();
  run_core_primitives_experiments_for_type<
    int64_t_meta, Elements, BaselineTrials, RegularTrials
  >();
  run_core_primitives_experiments_for_type<
    float_meta,   Elements, BaselineTrials, RegularTrials
  >();
  run_core_primitives_experiments_for_type<
    double_meta,  Elements, BaselineTrials, RegularTrials
  >();
}

///////////////////////////////////////////////////////////////////////////////

// XXX Use `std::string_view` when possible.
std::vector<std::string> split(std::string const& str, std::string const& delim)
{
  std::vector<std::string> tokens;
  std::string::size_type prev = 0, pos = 0;
  do
  {
    pos = str.find(delim, prev);
    if (pos == std::string::npos) pos = str.length();
    std::string token = str.substr(prev, pos - prev);
    if (!token.empty()) tokens.push_back(token);
    prev = pos + delim.length();
  }
  while (pos < str.length() && prev < str.length());
  return tokens;
}

///////////////////////////////////////////////////////////////////////////////

struct command_line_option_error : std::exception
{
  virtual ~command_line_option_error() NOEXCEPT {}
  virtual const char* what() const NOEXCEPT = 0;
};

struct only_one_option_allowed : command_line_option_error
{
  // Construct a new `only_one_option_allowed` exception. `key` is the
  // option name and `[first, last)` is a sequence of
  // `std::pair<std::string const, std::string>`s (the values).
  template <typename InputIt>
  only_one_option_allowed(std::string const& key, InputIt first, InputIt last)
    : message()
  {
    message  = "Only one `--";
    message += key;
    message += "` option is allowed, but multiple were received: ";

    for (; first != last; ++first)
    {
      message += "`";
      message += (*first).second;
      message += "` ";
    }

    // Remove the trailing space added by the last iteration of the above loop.
    message.erase(message.size() - 1, 1);

    message += ".";
  }

  virtual ~only_one_option_allowed() NOEXCEPT {}

  virtual const char* what() const NOEXCEPT
  {
    return message.c_str();
  }

private:
  std::string message;
};

struct required_option_missing : command_line_option_error
{
  // Construct a new `requirement_option_missing` exception. `key` is the
  // option name.
  required_option_missing(std::string const& key)
    : message()
  {
    message  = "`--";
    message += key;
    message += "` option is required.";
  }

  virtual ~required_option_missing() NOEXCEPT {}

  virtual const char* what() const NOEXCEPT
  {
    return message.c_str();
  }

private:
  std::string message;
};

struct command_line_processor
{
  typedef std::vector<std::string> positional_options_type;

  typedef std::multimap<std::string, std::string> keyword_options_type;

  typedef std::pair<
    keyword_options_type::const_iterator
  , keyword_options_type::const_iterator
  > keyword_option_values;

  command_line_processor(int argc, char** argv)
    : pos_args(), kw_args()
  { // {{{
    for (int i = 1; i < argc; ++i)
    {
      std::string arg(argv[i]);

      // Look for --key or --key=value options.
      if (arg.substr(0, 2) == "--")
      {
        std::string::size_type n = arg.find('=', 2);

        keyword_options_type::value_type key_value;

        if (n == std::string::npos) // --key
          kw_args.insert(keyword_options_type::value_type(
            arg.substr(2), ""
          ));
        else                        // --key=value
          kw_args.insert(keyword_options_type::value_type(
            arg.substr(2, n - 2), arg.substr(n + 1)
          ));

        kw_args.insert(key_value);
      }
      else // Assume it's positional.
        pos_args.push_back(arg);
    }
  } // }}}

  // Return the value for option `key`.
  //
  // Throws:
  // * `only_one_option_allowed` if there is more than one value for `key`.
  // * `required_option_missing` if there is no value for `key`.
  std::string operator()(std::string const& key) const
  {
    keyword_option_values v = kw_args.equal_range(key);

    keyword_options_type::difference_type d = std::distance(v.first, v.second);

    if      (1 < d)  // Too many options.
      throw only_one_option_allowed(key, v.first, v.second);
    else if (0 == d) // No option.
      throw required_option_missing(key);

    return (*v.first).second;
  }

  // Return the value for option `key`, or `dflt` if `key` has no value.
  //
  // Throws: `only_one_option_allowed` if there is more than one value for `key`.
  std::string operator()(std::string const& key, std::string const& dflt) const
  {
    keyword_option_values v = kw_args.equal_range(key);

    keyword_options_type::difference_type d = std::distance(v.first, v.second);

    if (1 < d)  // Too many options.
      throw only_one_option_allowed(key, v.first, v.second);

    if (0 == d) // No option.
      return dflt;
    else        // 1 option.
      return (*v.first).second;
  }

  // Returns `true` if the option `key` was specified at least once.
  bool has(std::string const& key) const
  {
    return kw_args.count(key) > 0;
  }

private:
  positional_options_type pos_args;
  keyword_options_type    kw_args;
};

///////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  command_line_processor clp(argc, argv);

  #if defined(HAVE_TBB)
  tbb::task_scheduler_init init;

  test_tbb();
  #endif

  #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    // Set the CUDA device to use for the benchmark - `0` by default.

    int device = std::atoi(clp("device", "0").c_str());
    // `std::atoi` returns 0 if the conversion fails.

    cudaSetDevice(device);
  #endif

  if (!clp.has("no-header"))
    print_experiment_header();

                                          /* Elements |       Trials       */
                                          /*          | Baseline | Regular */
//run_core_primitives_experiments< 1LLU << 21LLU      , 4        , 16      >();
//run_core_primitives_experiments< 1LLU << 22LLU      , 4        , 16      >();
//run_core_primitives_experiments< 1LLU << 23LLU      , 4        , 16      >();
//run_core_primitives_experiments< 1LLU << 24LLU      , 4        , 16      >();
//run_core_primitives_experiments< 1LLU << 25LLU      , 4        , 16      >();
  run_core_primitives_experiments< 1LLU << 26LLU      , 4        , 16      >();
  run_core_primitives_experiments< 1LLU << 27LLU      , 4        , 16      >();
//run_core_primitives_experiments< 1LLU << 28LLU      , 4        , 16      >();
//run_core_primitives_experiments< 1LLU << 29LLU      , 4        , 16      >();

  return 0;
}

// TODO: Add different input sizes and half precision
