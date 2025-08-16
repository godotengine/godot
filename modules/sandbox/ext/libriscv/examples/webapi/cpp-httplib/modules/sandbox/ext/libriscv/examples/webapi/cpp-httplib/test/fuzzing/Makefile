
#CXX = clang++
# Do not add default sanitizer flags here as OSS-fuzz adds its own sanitizer flags.
CXXFLAGS += -ggdb -O0 -std=c++11 -DGTEST_USE_OWN_TR1_TUPLE -I../.. -I. -Wall -Wextra -Wtype-limits -Wconversion

OPENSSL_DIR = /usr/local/opt/openssl@1.1

# Using full path to libssl and libcrypto to avoid accidentally picking openssl libs brought in by msan.
OPENSSL_SUPPORT = -DCPPHTTPLIB_OPENSSL_SUPPORT -I$(OPENSSL_DIR)/include -I$(OPENSSL_DIR)/lib  /usr/local/lib/libssl.a /usr/local/lib/libcrypto.a

ZLIB_SUPPORT = -DCPPHTTPLIB_ZLIB_SUPPORT -lz

BROTLI_DIR = /usr/local/opt/brotli
# BROTLI_SUPPORT = -DCPPHTTPLIB_BROTLI_SUPPORT -I$(BROTLI_DIR)/include -L$(BROTLI_DIR)/lib -lbrotlicommon -lbrotlienc -lbrotlidec

# Runs all the tests and also fuzz tests against seed corpus.
all : server_fuzzer
	./server_fuzzer corpus/*

# Fuzz target, so that you can choose which $(LIB_FUZZING_ENGINE) to use.
server_fuzzer : server_fuzzer.cc ../../httplib.h
# 	$(CXX) $(CXXFLAGS) -o $@  $<  -Wl,-Bstatic $(OPENSSL_SUPPORT)  -Wl,-Bdynamic -ldl  $(ZLIB_SUPPORT)  $(LIB_FUZZING_ENGINE) -pthread
	$(CXX) $(CXXFLAGS) -o $@  $<  $(ZLIB_SUPPORT)  $(LIB_FUZZING_ENGINE) -pthread -lanl
	zip -q -r server_fuzzer_seed_corpus.zip corpus

clean:
	rm -f server_fuzzer pem *.0 *.o *.1 *.srl *.zip
