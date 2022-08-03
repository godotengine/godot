#include "unittest/testframework.h"
#include "unittest/exceptions.h"
#include <thrust/memory.h>

// #include backends' testframework.h, if they exist and are required for the build
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#include <unittest/cuda/testframework.h>
#endif

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <string>
#include <limits>
#include <ctime>
#include <limits>


const size_t standard_test_sizes[] =
{
  0, 1, 2, 3, 4, 5, 8, 10, 13, 16, 17, 19, 27, 30, 31, 32,
  33, 35, 42, 53, 58, 63, 64, 65, 72, 97, 100, 127, 128, 129, 142, 183, 192, 201, 240, 255, 256,
  257, 302, 511, 512, 513, 687, 900, 1023, 1024, 1025, 1565, 1786, 1973, 2047, 2048, 2049, 3050, 4095, 4096,
  4097, 5030, 7791, 10000, 10027, 12345, 16384, 17354, 26255, 32768, 43718, 65533, 65536,
  65539, 123456, 131072, 731588, 1048575, 1048576,
  3398570, 9760840, (1 << 24) - 1, (1 << 24),
  (1 << 24) + 1, (1 << 25) - 1, (1 << 25), (1 << 25) + 1, (1 << 26) - 1, 1 << 26,
  (1 << 26) + 1, (1 << 27) - 1, (1 << 27)
};


const size_t tiny_threshold    = 1 <<  5;  //   32
const size_t small_threshold   = 1 <<  8;  //  256
const size_t medium_threshold  = 1 << 12;  //   4K
const size_t default_threshold = 1 << 16;  //  64K
const size_t large_threshold   = 1 << 20;  //   1M
const size_t huge_threshold    = 1 << 24;  //  16M
const size_t epic_threshold    = 1 << 26;  //  64M
const size_t max_threshold     = (std::numeric_limits<size_t>::max)();


std::vector<size_t> test_sizes;


std::vector<size_t> get_test_sizes(void)
{
  return test_sizes;
}


void set_test_sizes(const std::string& val)
{
  size_t threshold = 0;

  if(val == "tiny")
    threshold = tiny_threshold;
  else if(val == "small")
    threshold = small_threshold;
  else if(val == "medium")
    threshold = medium_threshold;
  else if(val == "default")
    threshold = default_threshold;
  else if(val == "large")
    threshold = large_threshold;
  else if(val == "huge")
    threshold = huge_threshold;
  else if(val == "epic")
    threshold = epic_threshold;
  else if(val == "max")
    threshold = max_threshold;
  else
  {
    std::cerr << "invalid test size \"" << val << "\"" << std::endl;
    exit(1);
  }

  for(size_t i = 0; i < sizeof(standard_test_sizes) / sizeof(*standard_test_sizes); i++)
  {
    if(standard_test_sizes[i] <= threshold)
      test_sizes.push_back(standard_test_sizes[i]);
  }
}


void UnitTestDriver::register_test(UnitTest * test)
{
  if(UnitTestDriver::s_driver().test_map.count(test->name) )
  {
    std::cout << "[WARNING] Test name \"" << test->name << " already encountered " << std::endl;
  }

  UnitTestDriver::s_driver().test_map[test->name] = test;
}


UnitTest::UnitTest(const char * _name) : name(_name)
{
  UnitTestDriver::s_driver().register_test(this);
}


void process_args(int argc, char ** argv,
                  ArgumentSet& args,
                  ArgumentMap& kwargs)

{
  for(int i = 1; i < argc; i++)
  {
    std::string arg(argv[i]);

    // look for --key or --key=value arguments
    if(arg.substr(0,2) == "--")
    {
      std::string::size_type n = arg.find('=',2);

      if(n == std::string::npos)
      {
        kwargs[arg.substr(2)] = std::string();              // (key,"")
      }
      else
      {
        kwargs[arg.substr(2, n - 2)] = arg.substr(n + 1);   // (key,value)
      }
    }
    else
    {
      args.insert(arg);
    }
  }
}


void usage(int /*argc*/, char** argv)
{
  std::string indent = "  ";

  std::cout << "Example Usage:\n";
  std::cout << indent << argv[0] << "\n";
  std::cout << indent << argv[0] << " TestName1 [TestName2 ...] \n";
  std::cout << indent << argv[0] << " PartialTestName1* [PartialTestName2* ...] \n";
  std::cout << indent << argv[0] << " --device=1\n";
  std::cout << indent << argv[0] << " --sizes={tiny,small,medium,default,large,huge,epic,max}\n";
  std::cout << indent << argv[0] << " --verbose or --concise\n";
  std::cout << indent << argv[0] << " --list\n";
  std::cout << indent << argv[0] << " --help\n";
  std::cout << "\n";
  std::cout << "Options:\n";
  std::cout << indent << "The sizes option determines which input sizes are tested.\n";
  std::cout << indent << indent << "--sizes=tiny    tests sizes up to " << tiny_threshold    << "\n";
  std::cout << indent << indent << "--sizes=small   tests sizes up to " << small_threshold   << "\n";
  std::cout << indent << indent << "--sizes=medium  tests sizes up to " << medium_threshold  << "\n";
  std::cout << indent << indent << "--sizes=default tests sizes up to " << default_threshold << "\n";
  std::cout << indent << indent << "--sizes=large   tests sizes up to " << large_threshold   << " (0.25 GB memory)\n";
  std::cout << indent << indent << "--sizes=huge    tests sizes up to " << huge_threshold    << " (1.50 GB memory)\n";
  std::cout << indent << indent << "--sizes=epic    tests sizes up to " << epic_threshold    << " (3.00 GB memory)\n";
  std::cout << indent << indent << "--sizes=max     tests all available sizes\n";
}


struct TestResult
{
  TestStatus  status;
  std::string name;
  std::string message;

  // XXX use a c++11 timer result when available
  std::clock_t elapsed;

  TestResult(const TestStatus status, std::clock_t elapsed, const UnitTest& u, const std::string& message = "")
      : status(status), name(u.name), message(message), elapsed(elapsed)
  {}

  bool operator<(const TestResult& tr) const
  {
    if(status < tr.status)
    {
      return true;
    }
    else if(tr.status < status)
    {
      return false;
    }
    else
    {
      return name < tr.name;
    }
  }
};


void record_result(const TestResult& test_result, std::vector< TestResult >& test_results)
{
  test_results.push_back(test_result);
}


void report_results(std::vector< TestResult >& test_results, double elapsed_minutes)
{
  std::cout << std::endl;

  std::string hline = "================================================================";

  std::sort(test_results.begin(), test_results.end());

  size_t num_passes = 0;
  size_t num_failures = 0;
  size_t num_known_failures = 0;
  size_t num_errors = 0;

  for(size_t i = 0; i < test_results.size(); i++)
  {
    const TestResult& tr = test_results[i];

    if(tr.status == Pass)
    {
      num_passes++;
    }
    else
    {
      std::cout << hline << std::endl;

      switch(tr.status)
      {
        case Failure:
          std::cout << "FAILURE";       num_failures++;       break;
        case KnownFailure:
          std::cout << "KNOWN FAILURE"; num_known_failures++; break;
        case Error:
          std::cout << "ERROR";         num_errors++;         break;
        default:
          break;
      }

      std::cout << ": " << tr.name << std::endl << tr.message << std::endl;
    }
  }

  std::cout << hline << std::endl;

  std::cout << "Totals: ";
  std::cout << num_failures << " failures, ";
  std::cout << num_known_failures << " known failures, ";
  std::cout << num_errors << " errors, and ";
  std::cout << num_passes << " passes." << std::endl;
  std::cout << "Time:  " << elapsed_minutes << " minutes" << std::endl;
}


void UnitTestDriver::list_tests(void)
{
  for(TestMap::iterator iter = test_map.begin(); iter != test_map.end(); iter++)
  {
    std::cout << iter->second->name << std::endl;
  }
}


bool UnitTestDriver::post_test_smoke_check(const UnitTest &/*test*/, bool /*concise*/)
{
  return true;
}


bool UnitTestDriver::run_tests(std::vector<UnitTest *>& tests_to_run, const ArgumentMap& kwargs)
{
  std::time_t start_time = std::time(0);

  THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_BEGIN
  bool verbose = kwargs.count("verbose");
  bool concise = kwargs.count("concise");
  THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_END

  std::vector< TestResult > test_results;

  if(verbose && concise)
  {
    std::cout << "--verbose and --concise cannot be used together" << std::endl;
    exit(EXIT_FAILURE);
  }

  if(!concise)
  {
    std::cout << "Running " << tests_to_run.size() << " unit tests." << std::endl;
  }

  for(size_t i = 0; i < tests_to_run.size(); i++)
  {
     UnitTest& test = *tests_to_run[i];

     if(verbose)
     {
       std::cout << "Running " << test.name << "..." << std::flush;
     }

     try
     {
       // time the test
       std::clock_t start = std::clock();

       // run the test
       test.run();

       // test passed
       record_result(TestResult(Pass, std::clock() - start, test), test_results);
     }
     catch(unittest::UnitTestFailure& f)
     {
       record_result(TestResult(Failure, (std::numeric_limits<std::clock_t>::max)(), test, f.message), test_results);
     }
     catch(unittest::UnitTestKnownFailure& f)
     {
       record_result(TestResult(KnownFailure, (std::numeric_limits<std::clock_t>::max)(), test, f.message), test_results);
     }
     catch(std::bad_alloc& e)
     {
       record_result(TestResult(Error, (std::numeric_limits<std::clock_t>::max)(), test, e.what()), test_results);
     }
     catch(unittest::UnitTestError& e)
     {
       record_result(TestResult(Error, (std::numeric_limits<std::clock_t>::max)(), test, e.message), test_results);
     }

     // immediate report
     if(!concise)
     {
       if(verbose)
       {
         switch(test_results.back().status)
         {
           case Pass:
             std::cout << "\r[PASS] ";
             std::cout << std::setw(10) << 1000.f * float(test_results.back().elapsed) / CLOCKS_PER_SEC << " ms";
             break;
           case Failure:
             std::cout << "\r[FAILURE]           "; break;
           case KnownFailure:
             std::cout << "\r[KNOWN FAILURE]     "; break;
           case Error:
             std::cout << "\r[ERROR]             "; break;
           default:
             break;
         }

         std::cout << " " << test.name << std::endl;
       }
       else
       {
         switch(test_results.back().status)
         {
           case Pass:
             std::cout << "."; break;
           case Failure:
             std::cout << "F"; break;
           case KnownFailure:
             std::cout << "K"; break;
           case Error:
             std::cout << "E"; break;
           default:
             break;
         }
       }
     }

     if(!post_test_smoke_check(test, concise))
     {
       return false;
     }

     std::cout.flush();
  }

  double elapsed_minutes = double(std::time(0) - start_time) / 60;

  // summary report
  if(!concise)
  {
    report_results(test_results, elapsed_minutes);
  }


  // if any failures or errors return false
  for(size_t i = 0; i < test_results.size(); i++)
  {
    if(test_results[i].status != Pass && test_results[i].status != KnownFailure)
    {
      return false;
    }
  }

  // all tests pass or are known failures
  return true;
}


bool UnitTestDriver::run_tests(const ArgumentSet& args, const ArgumentMap& kwargs)
{
  if(args.empty())
  {
    // run all tests
    std::vector<UnitTest *> tests_to_run;

    for(TestMap::iterator iter = test_map.begin(); iter != test_map.end(); iter++)
    {
      tests_to_run.push_back(iter->second);
    }

    return run_tests(tests_to_run, kwargs);
  }
  else
  {
    // all non-keyword arguments are assumed to be test names or partial test names

    typedef TestMap::iterator               TestMapIterator;

    // vector to accumulate tests
    std::vector<UnitTest *> tests_to_run;

    for(ArgumentSet::const_iterator iter = args.begin(); iter != args.end(); iter++)
    {
      const std::string& arg = *iter;

      size_t len = arg.size();
      size_t matches = 0;

      if(arg[len-1] == '*')
      {
        // wildcard search
        std::string search = arg.substr(0,len-1);

        TestMapIterator lb = test_map.lower_bound(search);
        while(lb != test_map.end())
        {
          if(search != lb->first.substr(0,len-1))
          {
            break;
          }

          tests_to_run.push_back(lb->second);
          lb++;
          matches++;
        }
      }
      else
      {
        // non-wildcard search
        TestMapIterator lb = test_map.find(arg);

        if(lb != test_map.end())
        {
          tests_to_run.push_back(lb->second);
          matches++;
        }
      }

      if(matches == 0)
      {
        std::cout << "[ERROR] found no test names matching the pattern: " << arg << std::endl;
        return false;
      }
    }

    return run_tests(tests_to_run, kwargs);
  }
}


// driver_instance maps a DeviceSystem to a singleton UnitTestDriver
template<typename DeviceSystem>
UnitTestDriver &driver_instance(DeviceSystem)
{
  static UnitTestDriver s_instance;
  return s_instance;
}


// if we need a special kind of UnitTestDriver, overload
// driver_instance in that function
UnitTestDriver &UnitTestDriver::s_driver()
{
  return driver_instance(thrust::device_system_tag());
}


int main(int argc, char **argv)
{
  ArgumentSet args;
  ArgumentMap kwargs;

  process_args(argc, argv, args, kwargs);

  if(kwargs.count("help"))
  {
    usage(argc, argv);
    return 0;
  }

  if(kwargs.count("list"))
  {
    UnitTestDriver::s_driver().list_tests();
    return 0;
  }

  if(kwargs.count("sizes"))
  {
    set_test_sizes(kwargs["sizes"]);
  }
  else
  {
    set_test_sizes("default");
  }

  bool passed = UnitTestDriver::s_driver().run_tests(args, kwargs);

  if(kwargs.count("concise"))
  {
    std::cout << ((passed) ? "PASSED" : "FAILED") << std::endl;
  }

  return (passed) ? EXIT_SUCCESS : EXIT_FAILURE;
}

