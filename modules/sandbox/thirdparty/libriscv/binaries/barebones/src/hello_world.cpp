#include <include/libc.hpp>
#include <cassert>
#include <memory>
#include <string>

#include <microthread.hpp>

int testval = 0;

extern "C"
__attribute__((constructor))
void test_constructor() {
	static const char hello[] = "Hello, Global Constructor!\n";
	sys_write(hello, sizeof(hello)-1);
	testval = 22;
}

struct testdata
{
	int depth     = 0;
	const int max_depth = 20;
	std::vector<microthread::Thread_ptr> threads;
};
static testdata tdata;

static long recursive_function(testdata* data)
{
	data->depth++;
	printf("%d: Thread depth %d / %d\n",
			microthread::gettid(), data->depth, data->max_depth);

	if (data->depth < data->max_depth)
	{
		auto thread = microthread::create(recursive_function, data);
		if (thread == nullptr) {
			printf("Failed to create thread!\n");
			return -1;
		}
		data->threads.push_back(std::move(thread));
	}
	printf("%d: Thread yielding %d / %d\n",
			microthread::gettid(), data->depth, data->max_depth);
	microthread::yield();

	printf("%d: Thread exiting %d / %d\n",
			microthread::gettid(), data->depth, data->max_depth);
	data->depth--;
	return 0;
}

int main(int argc, char** argv)
{
	printf("Arguments: %d\n", argc);
	for (int i = 0; i < argc; i++) {
		printf("Arg %d: %s\n", i, argv[i]);
	}
	printf("Note: If you see only garbage here, activate the native-heap "
			"system calls in the emulator.\n");
	static const char* hello = "Hello %s World v%d.%d!\n";
	assert(testval == 22);
	// Heap test
	{
		auto b = std::make_unique<char[]> (64);
		assert(b != nullptr);
		// copy into string
		strcpy(b.get(), hello);
		// va_list & stdarg test
		int len = printf(b.get(), "RISC-V", 1, 0);
		assert(len > 0); (void) len;
	}

	printf("Main thread tid=%d\n", microthread::gettid());

	auto thread = microthread::create(
		[] (int a, int b, int c) -> long {
			printf("Hello from microthread tid=%d!\n"
					"a = %d, b = %d, c = %d\n",
					microthread::gettid(), a, b, c);
			auto t2 = microthread::create([] () -> long {
				printf("Second thread tid=%d, yielding directly to tid=1!\n",
						microthread::gettid());
				int ret = microthread::yield_to(1);
				printf("Second thread back from yielding, returned %d\n", ret);
				microthread::exit(222);
			});
			printf("I'm back in the first microthread now! Yielding back.\n");
			int ret = microthread::yield_to(2);
			// get return value from second thread
			long rv = microthread::join(t2);

			printf("yield=%d, Starting recursive nightmare!\n", ret);
			recursive_function(&tdata);
			for (auto& thread : tdata.threads) {
				microthread::join(thread);
			}
			return rv;
		}, 111, 222, 333);
	printf("back in thread 0, joining microthread\n");
	long retval = microthread::join(thread);
	printf("microthread returned %ld\n", retval);
	microthread::yield_to(0);
	printf("Main thread tid=%d\n", microthread::gettid());

	return 666;
}
