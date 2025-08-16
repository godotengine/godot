#CXX = clang++
CXXFLAGS = -O2 -std=c++11 -I.. -Wall -Wextra -pthread

PREFIX ?= $(shell brew --prefix)

OPENSSL_DIR = $(PREFIX)/opt/openssl@3
OPENSSL_SUPPORT = -DCPPHTTPLIB_OPENSSL_SUPPORT -I$(OPENSSL_DIR)/include -L$(OPENSSL_DIR)/lib -lssl -lcrypto

ifneq ($(OS), Windows_NT)
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S), Darwin)
		OPENSSL_SUPPORT += -framework CoreFoundation -framework Security
	endif
endif

ZLIB_SUPPORT = -DCPPHTTPLIB_ZLIB_SUPPORT -lz

BROTLI_DIR = $(PREFIX)/opt/brotli
BROTLI_SUPPORT = -DCPPHTTPLIB_BROTLI_SUPPORT -I$(BROTLI_DIR)/include -L$(BROTLI_DIR)/lib -lbrotlicommon -lbrotlienc -lbrotlidec

all: server client hello simplecli simplesvr upload redirect ssesvr ssecli benchmark one_time_request server_and_client accept_header

server : server.cc ../httplib.h Makefile
	$(CXX) -o server $(CXXFLAGS) server.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

client : client.cc ../httplib.h Makefile
	$(CXX) -o client $(CXXFLAGS) client.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

hello : hello.cc ../httplib.h Makefile
	$(CXX) -o hello $(CXXFLAGS) hello.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

simplecli : simplecli.cc ../httplib.h Makefile
	$(CXX) -o simplecli $(CXXFLAGS) simplecli.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

simplesvr : simplesvr.cc ../httplib.h Makefile
	$(CXX) -o simplesvr $(CXXFLAGS) simplesvr.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

upload : upload.cc ../httplib.h Makefile
	$(CXX) -o upload $(CXXFLAGS) upload.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

redirect : redirect.cc ../httplib.h Makefile
	$(CXX) -o redirect $(CXXFLAGS) redirect.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

ssesvr : ssesvr.cc ../httplib.h Makefile
	$(CXX) -o ssesvr $(CXXFLAGS) ssesvr.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

ssecli : ssecli.cc ../httplib.h Makefile
	$(CXX) -o ssecli $(CXXFLAGS) ssecli.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

benchmark : benchmark.cc ../httplib.h Makefile
	$(CXX) -o benchmark $(CXXFLAGS) benchmark.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

one_time_request : one_time_request.cc ../httplib.h Makefile
	$(CXX) -o one_time_request $(CXXFLAGS) one_time_request.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

server_and_client : server_and_client.cc ../httplib.h Makefile
	$(CXX) -o server_and_client $(CXXFLAGS) server_and_client.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

accept_header : accept_header.cc ../httplib.h Makefile
	$(CXX) -o accept_header $(CXXFLAGS) accept_header.cc $(OPENSSL_SUPPORT) $(ZLIB_SUPPORT) $(BROTLI_SUPPORT)

pem:
	openssl genrsa 2048 > key.pem
	openssl req -new -key key.pem | openssl x509 -days 3650 -req -signkey key.pem > cert.pem

clean:
	rm server client hello simplecli simplesvr upload redirect ssesvr ssecli benchmark one_time_request server_and_client accept_header *.pem
