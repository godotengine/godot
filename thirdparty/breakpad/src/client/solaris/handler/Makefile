# Copyright (c) 2007, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Author: Alfred Peng

CC=cc
CXX=CC

CPPFLAGS=-g -I../../.. -DNDEBUG -features=extensions -D_REENTRANT
LDFLAGS=-lpthread -lssl -lgnutls-openssl -lelf

OBJ_DIR=.
BIN_DIR=.

THREAD_SRC=solaris_lwp.cc
SHARE_SRC=../../minidump_file_writer.cc\
	  ../../../common/convert_UTF.cc\
	  ../../../common/md5.cc\
	  ../../../common/string_conversion.cc\
	  ../../../common/solaris/file_id.cc\
	  minidump_generator.cc
HANDLER_SRC=exception_handler.cc\
	  ../../../common/solaris/guid_creator.cc

MINIDUMP_TEST_SRC=minidump_test.cc
EXCEPTION_TEST_SRC=exception_handler_test.cc

THREAD_OBJ=$(patsubst %.cc,$(OBJ_DIR)/%.o,$(THREAD_SRC))
SHARE_OBJ=$(patsubst %.cc,$(OBJ_DIR)/%.o,$(SHARE_SRC))
HANDLER_OBJ=$(patsubst %.cc,$(OBJ_DIR)/%.o,$(HANDLER_SRC))
MINIDUMP_TEST_OBJ=$(patsubst %.cc,$(OBJ_DIR)/%.o, $(MINIDUMP_TEST_SRC))\
		  $(THREAD_OBJ) $(SHARE_OBJ) $(HANDLER_OBJ)
EXCEPTION_TEST_OBJ=$(patsubst %.cc,$(OBJ_DIR)/%.o, $(EXCEPTION_TEST_SRC))\
          $(THREAD_OBJ) $(SHARE_OBJ) $(HANDLER_OBJ)

BIN=$(BIN_DIR)/minidump_test\
    $(BIN_DIR)/exception_handler_test

.PHONY:all clean

all:$(BIN)

$(BIN_DIR)/minidump_test:$(MINIDUMP_TEST_OBJ)
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $^ -o $@

$(BIN_DIR)/exception_handler_test:$(EXCEPTION_TEST_OBJ)
	$(CXX) $(CPPFLAGS) $(LDFLAGS) $^ -o $@

clean:
	rm -f $(BIN) *.o *.out *.dmp core ../../minidump_file_writer.o\
		../../../common/*.o ../../../common/solaris/*.o
