
#include "memory.h"
#include "speex_bind.h"
#include
#ifdef __cplusplus
extern "C" {
#endif

void *speex_alloc (int size) {

	uint8_t * mem = (uint8_t*)memalloc(size);
	for(int i=0;i<size;i++)
		mem[i]=0;
	return mem;
}

void *speex_alloc_scratch (int size) {

	return memalloc(size);
}

void *speex_realloc (void *ptr, int size) {

	return memrealloc(ptr,size);
}

void speex_free (void *ptr) {

	memfree(ptr);
}

void speex_free_scratch (void *ptr) {

	memfree(ptr);
}

void _speex_fatal(const char *str, const char *file, int line) {

	_err_print_error("SPEEX ERROR",p_file,p_line,str);
}

void speex_warning(const char *str) {

	_err_print_error("SPEEX WARNING","",0,str);
}

void speex_warning_int(const char *str, int val) {

	_err_print_error("SPEEX WARNING INT","Value",val,str);
}

void speex_notify(const char *str) {

	print_line(str);
}

void _speex_putc(int ch, void *file) {

	// will not putc, no.
}

#ifdef __cplusplus
}
#endif
