###########################################
# Simple Makefile for HIDAPI test program
#
# Alan Ott
# Signal 11 Software
# 2010-06-01
###########################################

all: testgui

CC=gcc
CXX=g++
COBJS=../libusb/hid.o
CPPOBJS=test.o
OBJS=$(COBJS) $(CPPOBJS)
CFLAGS=-I../hidapi -Wall -g -c `fox-config --cflags` `pkg-config libusb-1.0 --cflags`
LIBS=-ludev -lrt -lpthread `fox-config --libs` `pkg-config libusb-1.0 --libs`


testgui: $(OBJS)
	g++ -Wall -g $^ $(LIBS) -o testgui

$(COBJS): %.o: %.c
	$(CC) $(CFLAGS) $< -o $@

$(CPPOBJS): %.o: %.cpp
	$(CXX) $(CFLAGS) $< -o $@

clean:
	rm *.o testgui

.PHONY: clean
