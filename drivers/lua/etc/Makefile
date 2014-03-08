# makefile for Lua etc

TOP= ..
LIB= $(TOP)/src
INC= $(TOP)/src
BIN= $(TOP)/src
SRC= $(TOP)/src
TST= $(TOP)/test

CC= gcc
CFLAGS= -O2 -Wall -I$(INC) $(MYCFLAGS)
MYCFLAGS= 
MYLDFLAGS= -Wl,-E
MYLIBS= -lm
#MYLIBS= -lm -Wl,-E -ldl -lreadline -lhistory -lncurses
RM= rm -f

default:
	@echo 'Please choose a target: min noparser one strict clean'

min:	min.c
	$(CC) $(CFLAGS) $@.c -L$(LIB) -llua $(MYLIBS)
	echo 'print"Hello there!"' | ./a.out

noparser: noparser.o
	$(CC) noparser.o $(SRC)/lua.o -L$(LIB) -llua $(MYLIBS)
	$(BIN)/luac $(TST)/hello.lua
	-./a.out luac.out
	-./a.out -e'a=1'

one:
	$(CC) $(CFLAGS) all.c $(MYLIBS)
	./a.out $(TST)/hello.lua

strict:
	-$(BIN)/lua -e 'print(a);b=2'
	-$(BIN)/lua -lstrict -e 'print(a)'
	-$(BIN)/lua -e 'function f() b=2 end f()'
	-$(BIN)/lua -lstrict -e 'function f() b=2 end f()'

clean:
	$(RM) a.out core core.* *.o luac.out

.PHONY:	default min noparser one strict clean
