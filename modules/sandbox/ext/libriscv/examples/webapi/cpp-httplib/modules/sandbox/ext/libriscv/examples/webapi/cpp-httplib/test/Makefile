CXX = clang++
CXXFLAGS = -g -std=c++11 -I. -Wall -Wextra -Wtype-limits -Wconversion -Wshadow $(EXTRA_CXXFLAGS) -DCPPHTTPLIB_USE_NON_BLOCKING_GETADDRINFO # -fno-exceptions -DCPPHTTPLIB_NO_EXCEPTIONS -fsanitize=address

PREFIX ?= $(shell brew --prefix)

OPENSSL_DIR = $(PREFIX)/opt/openssl@3
OPENSSL_SUPPORT = -DCPPHTTPLIB_OPENSSL_SUPPORT -I$(OPENSSL_DIR)/include -L$(OPENSSL_DIR)/lib -lssl -lcrypto

ifneq ($(OS), Windows_NT)
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S), Darwin)
		OPENSSL_SUPPORT += -DCPPHTTPLIB_USE_CERTS_FROM_MACOSX_KEYCHAIN -framework Security
	endif
endif

ZLIB_SUPPORT = -DCPPHTTPLIB_ZLIB_SUPPORT -lz

BROTLI_DIR = $(PREFIX)/opt/brotli
BROTLI_SUPPORT = -DCPPHTTPLIB_BROTLI_SUPPORT -I$(BROTLI_DIR)/include -L$(BROTLI_DIR)/lib -lbrotlicommon -lbrotlienc -lbrotlidec

ZSTD_DIR = $(PREFIX)/opt/zstd
ZSTD_SUPPORT = -DCPPHTTPLIB_ZSTD_SUPPORT -I$(ZSTD_DIR)/include -L$(ZSTD_DIR)/lib -lzstd

LIBS = -lpthread -lcurl
ifneq ($(OS), Windows_NT)
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S), Darwin)
		LIBS += -framework CoreFoundation -framework CFNetwork
	endif
	ifneq ($(UNAME_S), Darwin)
		LIBS += -lanl
	endif
endif

TEST_ARGS = gtest/src/gtest-all.cc gtest/src/gtest_main.cc -Igtest -Igtest/include $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT) $(ZSTD_SUPPORT) $(LIBS)

# By default, use standalone_fuzz_target_runner.
# This runner does no fuzzing, but simply executes the inputs
# provided via parameters.
# Run e.g. "make all LIB_FUZZING_ENGINE=/path/to/libFuzzer.a"
# to link the fuzzer(s) against a real fuzzing engine.
# OSS-Fuzz will define its own value for LIB_FUZZING_ENGINE.
LIB_FUZZING_ENGINE ?= standalone_fuzz_target_runner.o

CLANG_FORMAT = clang-format
REALPATH = $(shell which grealpath 2>/dev/null || which realpath 2>/dev/null)
STYLE_CHECK_FILES = $(filter-out httplib.h httplib.cc, \
	$(wildcard example/*.h example/*.cc fuzzing/*.h fuzzing/*.cc *.h *.cc ../httplib.h))

all : test test_split
	./test

proxy : test_proxy
	@echo "Starting proxy server..."
	cd proxy && \
	docker compose up -d
	@echo "Waiting for proxy to be ready..."
	@until nc -z localhost 3128 && nc -z localhost 3129; do sleep 1; done
	@echo "Proxy servers are ready, waiting additional 5 seconds for full startup..."
	@sleep 5
	@echo "Checking proxy server status..."
	@cd proxy && docker compose ps
	@echo "Checking proxy server logs..."
	@cd proxy && docker compose logs --tail=20
	@echo "Running proxy tests..."
	./test_proxy; \
	exit_code=$$?; \
	echo "Stopping proxy server..."; \
	cd proxy && docker compose down; \
	exit $$exit_code

test : test.cc include_httplib.cc ../httplib.h Makefile cert.pem
	$(CXX) -o $@ -I.. $(CXXFLAGS) test.cc include_httplib.cc $(TEST_ARGS)
	@file $@

# Note: The intention of test_split is to verify that it works to compile and
# link the split httplib.h, so there is normally no need to execute it.
test_split : test.cc ../httplib.h httplib.cc Makefile cert.pem
	$(CXX) -o $@ $(CXXFLAGS) test.cc httplib.cc $(TEST_ARGS)

check_abi:
	@./check-shared-library-abi-compatibility.sh

.PHONY: style_check
style_check: $(STYLE_CHECK_FILES)
	@for file in $(STYLE_CHECK_FILES); do \
		$(CLANG_FORMAT) $$file > $$file.formatted; \
		if ! diff -u $$file $$file.formatted; then \
			file2=$$($(REALPATH) --relative-to=.. $$file); \
			printf "\n%*s\n" 80 | tr ' ' '#'; \
			printf "##%*s##\n" 76; \
			printf "##   %-70s   ##\n" "$$file2 not properly formatted. Please run clang-format."; \
			printf "##%*s##\n" 76; \
			printf "%*s\n\n" 80 | tr ' ' '#'; \
			failed=1; \
		fi; \
		rm -f $$file.formatted; \
	done; \
	if [ -n "$$failed" ]; then \
		echo "Style check failed for one or more files. See above for details."; \
		false; \
	else \
		echo "All files are properly formatted."; \
	fi

test_proxy : test_proxy.cc ../httplib.h Makefile cert.pem
	$(CXX) -o $@ -I.. $(CXXFLAGS) test_proxy.cc $(TEST_ARGS)

# Runs server_fuzzer.cc based on value of $(LIB_FUZZING_ENGINE).
# Usage: make fuzz_test LIB_FUZZING_ENGINE=/path/to/libFuzzer
fuzz_test: server_fuzzer
	./server_fuzzer fuzzing/corpus/*

# Fuzz target, so that you can choose which $(LIB_FUZZING_ENGINE) to use.
server_fuzzer : fuzzing/server_fuzzer.cc ../httplib.h standalone_fuzz_target_runner.o
	$(CXX) -o $@ -I.. $(CXXFLAGS) $< $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT) $(LIB_FUZZING_ENGINE) $(ZSTD_SUPPORT) $(LIBS)
	@file $@

# Standalone fuzz runner, which just reads inputs from fuzzing/corpus/ dir and
# feeds it to server_fuzzer.
standalone_fuzz_target_runner.o : fuzzing/standalone_fuzz_target_runner.cpp
	$(CXX) -o $@ -I.. $(CXXFLAGS) -c $<

httplib.cc : ../httplib.h
	python3 ../split.py -o .

cert.pem:
	./gen-certs.sh

clean:
	rm -rf test test_split test_proxy server_fuzzer *.pem *.0 *.o *.1 *.srl httplib.h httplib.cc _build* *.dSYM

