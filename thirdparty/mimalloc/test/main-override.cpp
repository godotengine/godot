#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>

#include <mimalloc.h>
#include <new>
#include <vector>
#include <future>
#include <iostream>

#include <thread>
#include <mimalloc.h>
#include <assert.h>

#ifdef _WIN32
#include <mimalloc-new-delete.h>
#endif

#ifdef _WIN32
#include <Windows.h>
static void msleep(unsigned long msecs) { Sleep(msecs); }
#else
#include <unistd.h>
static void msleep(unsigned long msecs) { usleep(msecs * 1000UL); }
#endif

static void heap_thread_free_large(); // issue #221
static void heap_no_delete();         // issue #202
static void heap_late_free();         // issue #204
static void padding_shrink();         // issue #209
static void various_tests();
static void test_mt_shutdown();
static void large_alloc(void);        // issue #363
static void fail_aslr();              // issue #372
static void tsan_numa_test();         // issue #414
static void strdup_test();            // issue #445 
static void bench_alloc_large(void);  // issue #xxx
//static void test_large_migrate(void); // issue #691
static void heap_thread_free_huge();
static void test_std_string();        // issue #697

static void test_stl_allocators();


int main() {
  // mi_stats_reset();  // ignore earlier allocations
  
  // test_std_string();
  // heap_thread_free_huge();
  /*
   heap_thread_free_huge();
   heap_thread_free_large();
   heap_no_delete();
   heap_late_free();
   padding_shrink();
   various_tests();
   large_alloc();
   tsan_numa_test();
   strdup_test();
  */
  // test_stl_allocators();
  // test_mt_shutdown();
  // test_large_migrate();
  
  //fail_aslr();
  // bench_alloc_large();
  // mi_stats_print(NULL);
  return 0;
}

static void* p = malloc(8);

void free_p() {
  free(p);
  return;
}

class Test {
private:
  int i;
public:
  Test(int x) { i = x; }
  ~Test() { }
};


static void various_tests() {
  atexit(free_p);
  void* p1 = malloc(78);
  void* p2 = mi_malloc_aligned(24, 16);
  free(p1);
  p1 = malloc(8);
  char* s = mi_strdup("hello\n");

  mi_free(p2);
  p2 = malloc(16);
  p1 = realloc(p1, 32);
  free(p1);
  free(p2);
  mi_free(s);

  Test* t = new Test(42);
  delete t;
  t = new (std::nothrow) Test(42);
  delete t;
  auto tbuf = new unsigned char[sizeof(Test)];
  t = new (tbuf) Test(42);
  t->~Test();
  delete[] tbuf;
}

class Static {
private:
  void* p;
public:
  Static() {
    p = malloc(64);
    return;
  }
  ~Static() {
    free(p);
    return;
  }
};

static Static s = Static();


static bool test_stl_allocator1() {
  std::vector<int, mi_stl_allocator<int> > vec;
  vec.push_back(1);
  vec.pop_back();
  return vec.size() == 0;
}

struct some_struct { int i; int j; double z; };

static bool test_stl_allocator2() {
  std::vector<some_struct, mi_stl_allocator<some_struct> > vec;
  vec.push_back(some_struct());
  vec.pop_back();
  return vec.size() == 0;
}

#if MI_HAS_HEAP_STL_ALLOCATOR
static bool test_stl_allocator3() {
  std::vector<int, mi_heap_stl_allocator<int> > vec;
  vec.push_back(1);
  vec.pop_back();
  return vec.size() == 0;
}

static bool test_stl_allocator4() {
  std::vector<some_struct, mi_heap_stl_allocator<some_struct> > vec;
  vec.push_back(some_struct());
  vec.pop_back();
  return vec.size() == 0;
}

static bool test_stl_allocator5() {
  std::vector<int, mi_heap_destroy_stl_allocator<int> > vec;
  vec.push_back(1);
  vec.pop_back();
  return vec.size() == 0;
}

static bool test_stl_allocator6() {
  std::vector<some_struct, mi_heap_destroy_stl_allocator<some_struct> > vec;
  vec.push_back(some_struct());
  vec.pop_back();
  return vec.size() == 0;
}
#endif

static void test_stl_allocators() {
  test_stl_allocator1();
  test_stl_allocator2();
#if MI_HAS_HEAP_STL_ALLOCATOR
  test_stl_allocator3();
  test_stl_allocator4();
  test_stl_allocator5();
  test_stl_allocator6();
#endif
}

#if 0
// issue #691
static char* cptr;

static void* thread1_allocate()
{
  cptr = mi_calloc_tp(char,22085632);
  return NULL;
}

static void* thread2_free()
{
  assert(cptr);
  mi_free(cptr);
  cptr = NULL;
  return NULL;
}

static void test_large_migrate(void) {
  auto t1 = std::thread(thread1_allocate);
  t1.join();
  auto t2 = std::thread(thread2_free);
  t2.join();
  /*
  pthread_t thread1, thread2;

  pthread_create(&thread1, NULL, &thread1_allocate, NULL);
  pthread_join(thread1, NULL);

  pthread_create(&thread2, NULL, &thread2_free, NULL);
  pthread_join(thread2, NULL);
  */
  return;
}
#endif

// issue 445
static void strdup_test() {
#ifdef _MSC_VER
  char* s = _strdup("hello\n");
  char* buf = NULL;
  size_t len;
  _dupenv_s(&buf, &len, "MIMALLOC_VERBOSE");
  mi_free(buf);
  mi_free(s);
#endif
}

// Issue #202
static void heap_no_delete_worker() {
  mi_heap_t* heap = mi_heap_new();
  void* q = mi_heap_malloc(heap, 1024); (void)(q);
  // mi_heap_delete(heap); // uncomment to prevent assertion
}

static void heap_no_delete() {
  auto t1 = std::thread(heap_no_delete_worker);
  t1.join();
}


// Issue #697
static void test_std_string() {
  std::string path = "/Users/xxxx/Library/Developer/Xcode/DerivedData/xxxxxxxxxx/Build/Intermediates.noindex/xxxxxxxxxxx/arm64/XX_lto.o/0.arm64.lto.o";
  std::string path1 = "/Users/xxxx/Library/Developer/Xcode/DerivedData/xxxxxxxxxx/Build/Intermediates.noindex/xxxxxxxxxxx/arm64/XX_lto.o/1.arm64.lto.o";
  std::cout << path + "\n>>>            " + path1 + "\n>>>            " << std::endl;
}

// Issue #204
static volatile void* global_p;

static void t1main() {
  mi_heap_t* heap = mi_heap_new();
  global_p = mi_heap_malloc(heap, 1024);
  mi_heap_delete(heap);
}

static void heap_late_free() {
  auto t1 = std::thread(t1main);

  msleep(2000);
  assert(global_p);
  mi_free((void*)global_p);

  t1.join();
}

// issue  #209
static void* shared_p;
static void alloc0(/* void* arg */)
{
  shared_p = mi_malloc(8);
}

static void padding_shrink(void)
{
  auto t1 = std::thread(alloc0);
  t1.join();
  mi_free(shared_p);
}


// Issue #221
static void heap_thread_free_large_worker() {
  mi_free(shared_p);
}

static void heap_thread_free_large() {
  for (int i = 0; i < 100; i++) {
    shared_p = mi_malloc_aligned(2 * 1024 * 1024 + 1, 8);
    auto t1 = std::thread(heap_thread_free_large_worker);
    t1.join();
  }
}

static void heap_thread_free_huge_worker() {
  mi_free(shared_p);
}

static void heap_thread_free_huge() {
  for (int i = 0; i < 100; i++) {
    shared_p = mi_malloc(1024 * 1024 * 1024);
    auto t1 = std::thread(heap_thread_free_huge_worker);
    t1.join();
  }
}

static void test_mt_shutdown()
{
  const int threads = 5;
  std::vector< std::future< std::vector< char* > > > ts;

  auto fn = [&]()
  {
    std::vector< char* > ps;
    ps.reserve(1000);
    for (int i = 0; i < 1000; i++)
      ps.emplace_back(new char[1]);
    return ps;
  };

  for (int i = 0; i < threads; i++)
    ts.emplace_back(std::async(std::launch::async, fn));

  for (auto& f : ts)
    for (auto& p : f.get())
      delete[] p;

  std::cout << "done" << std::endl;
}

// issue #363
using namespace std;

void large_alloc(void)
{
  char* a = new char[1ull << 25];
  thread th([&] {
    delete[] a;
    });
  th.join();
}

// issue #372
static void fail_aslr() {
  size_t sz = (4ULL << 40); // 4TiB
  void* p = malloc(sz);
  printf("pointer p: %p: area up to %p\n", p, (uint8_t*)p + sz);
  *(int*)0x5FFFFFFF000 = 0;  // should segfault
}

// issues #414
static void dummy_worker() {
  void* p = mi_malloc(0);
  mi_free(p);
}

static void tsan_numa_test() {
  auto t1 = std::thread(dummy_worker);
  dummy_worker();
  t1.join();
}

// issue #?
#include <chrono>
#include <random>
#include <iostream>

static void bench_alloc_large(void) {
  static constexpr int kNumBuffers = 20;
  static constexpr size_t kMinBufferSize = 5 * 1024 * 1024;
  static constexpr size_t kMaxBufferSize = 25 * 1024 * 1024;
  std::unique_ptr<char[]> buffers[kNumBuffers];

  std::random_device rd;  (void)rd;
  std::mt19937 gen(42); //rd());
  std::uniform_int_distribution<> size_distribution(kMinBufferSize, kMaxBufferSize);
  std::uniform_int_distribution<> buf_number_distribution(0, kNumBuffers - 1);

  static constexpr int kNumIterations = 2000;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kNumIterations; ++i) {
    int buffer_idx = buf_number_distribution(gen);
    size_t new_size = size_distribution(gen);
    buffers[buffer_idx] = std::make_unique<char[]>(new_size);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto num_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  const auto us_per_allocation = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / kNumIterations;
  std::cout << kNumIterations << " allocations Done in " << num_ms << "ms." << std::endl;
  std::cout << "Avg " << us_per_allocation << " us per allocation" << std::endl;
}

