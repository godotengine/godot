/*************************************************************************/
/*  godot_win.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "os_windows.h"
#include "main/main.h"
#include <stdio.h>
#include <locale.h>

PCHAR*
    CommandLineToArgvA(
        PCHAR CmdLine,
        int* _argc
        )
    {
        PCHAR* argv;
        PCHAR  _argv;
        ULONG   len;
        ULONG   argc;
        CHAR   a;
        ULONG   i, j;

        BOOLEAN  in_QM;
        BOOLEAN  in_TEXT;
        BOOLEAN  in_SPACE;

        len = strlen(CmdLine);
        i = ((len+2)/2)*sizeof(PVOID) + sizeof(PVOID);

        argv = (PCHAR*)GlobalAlloc(GMEM_FIXED,
            i + (len+2)*sizeof(CHAR));

        _argv = (PCHAR)(((PUCHAR)argv)+i);

        argc = 0;
        argv[argc] = _argv;
        in_QM = FALSE;
        in_TEXT = FALSE;
        in_SPACE = TRUE;
        i = 0;
        j = 0;

        while( (a = CmdLine[i]) ) {
            if(in_QM) {
                if(a == '\"') {
                    in_QM = FALSE;
                } else {
                    _argv[j] = a;
                    j++;
                }
            } else {
                switch(a) {
                case '\"':
                    in_QM = TRUE;
                    in_TEXT = TRUE;
                    if(in_SPACE) {
                        argv[argc] = _argv+j;
                        argc++;
                    }
                    in_SPACE = FALSE;
                    break;
                case ' ':
                case '\t':
                case '\n':
                case '\r':
                    if(in_TEXT) {
                        _argv[j] = '\0';
                        j++;
                    }
                    in_TEXT = FALSE;
                    in_SPACE = TRUE;
                    break;
                default:
                    in_TEXT = TRUE;
                    if(in_SPACE) {
                        argv[argc] = _argv+j;
                        argc++;
                    }
                    _argv[j] = a;
                    j++;
                    in_SPACE = FALSE;
                    break;
                }
            }
            i++;
        }
        _argv[j] = '\0';
        argv[argc] = NULL;

        (*_argc) = argc;
        return argv;
    }

char* mb_to_utf8(const char* mbs) {

	int wlen = MultiByteToWideChar(CP_ACP,0,mbs,-1,NULL,0); // returns 0 if failed
	wchar_t *wbuf = new wchar_t[wlen + 1];
	MultiByteToWideChar(CP_ACP,0,mbs,-1,wbuf,wlen);
	wbuf[wlen]=0;

	int ulen = WideCharToMultiByte(CP_UTF8,0,wbuf,-1,NULL,0,NULL,NULL);
	char * ubuf = new char[ulen + 1];
	WideCharToMultiByte(CP_UTF8,0,wbuf,-1,ubuf,ulen,NULL,NULL);
	ubuf[ulen] = 0;
	return ubuf;
}

int main(int argc, char** argv) {

	OS_Windows os(NULL);

	setlocale(LC_CTYPE, "");

	char ** argv_utf8 = new char*[argc];
	for(int i=0; i<argc; ++i) {
		argv_utf8[i] = mb_to_utf8(argv[i]);
	}

	Main::setup(argv_utf8[0], argc - 1, &argv_utf8[1]);
	if (Main::start())
		os.run();
	Main::cleanup();


	for (int i=0; i<argc; ++i) {
		delete[] argv_utf8[i];
	}
	delete[] argv_utf8;

	return os.get_exit_code();
};

HINSTANCE godot_hinstance = NULL;

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)	{

    int    argc;
    char** argv;

    char*  arg;
    int    index;
    int    result;

    // count the arguments

    argc = 1;
    arg  = lpCmdLine;

    while (arg[0] != 0) {

        while (arg[0] != 0 && arg[0] == ' ') {
            arg++;
        }

        if (arg[0] != 0) {

            argc++;

            while (arg[0] != 0 && arg[0] != ' ') {
                arg++;
            }

        }

    }

    // tokenize the arguments

    argv = (char**)malloc(argc * sizeof(char*));

    arg = lpCmdLine;
    index = 1;

    while (arg[0] != 0) {

        while (arg[0] != 0 && arg[0] == ' ') {
            arg++;
        }

        if (arg[0] != 0) {

            argv[index] = arg;
            index++;

            while (arg[0] != 0 && arg[0] != ' ') {
                arg++;
            }

            if (arg[0] != 0) {
                arg[0] = 0;
                arg++;
            }

        }

    }

    // put the program name into argv[0]

    char filename[_MAX_PATH];

    GetModuleFileName(NULL, filename, _MAX_PATH);
    argv[0] = filename;

    // call the user specified main function

    result = main(argc, argv);

    free(argv);
    return result;
}
