
/* pngerror.c - stub functions for i/o and memory allocation
 *
 * Last changed in libpng 1.5.19 [August 21, 2014]
 * Copyright (c) 1998-2002,2004,2006-2014 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 *
 * This code is released under the libpng license.
 * For conditions of distribution and use, see the disclaimer
 * and license in png.h
 *
 * This file provides a location for all error handling.  Users who
 * need special error handling are expected to write replacement functions
 * and use png_set_error_fn() to use those functions.  See the instructions
 * at each function.
 */

#include "pngpriv.h"

#if defined(PNG_READ_SUPPORTED) || defined(PNG_WRITE_SUPPORTED)

static PNG_FUNCTION(void, png_default_error,PNGARG((png_structp png_ptr,
    png_const_charp error_message)),PNG_NORETURN);

#ifdef PNG_WARNINGS_SUPPORTED
static void /* PRIVATE */
png_default_warning PNGARG((png_structp png_ptr,
   png_const_charp warning_message));
#endif /* PNG_WARNINGS_SUPPORTED */

/* This function is called whenever there is a fatal error.  This function
 * should not be changed.  If there is a need to handle errors differently,
 * you should supply a replacement error function and use png_set_error_fn()
 * to replace the error function at run-time.
 */
#ifdef PNG_ERROR_TEXT_SUPPORTED
PNG_FUNCTION(void,PNGAPI
png_error,(png_structp png_ptr, png_const_charp error_message),PNG_NORETURN)
{
#ifdef PNG_ERROR_NUMBERS_SUPPORTED
   char msg[16];
   if (png_ptr != NULL)
   {
      if (png_ptr->flags&
         (PNG_FLAG_STRIP_ERROR_NUMBERS|PNG_FLAG_STRIP_ERROR_TEXT))
      {
         if (*error_message == PNG_LITERAL_SHARP)
         {
            /* Strip "#nnnn " from beginning of error message. */
            int offset;
            for (offset = 1; offset<15; offset++)
               if (error_message[offset] == ' ')
                  break;

            if (png_ptr->flags&PNG_FLAG_STRIP_ERROR_TEXT)
            {
               int i;
               for (i = 0; i < offset - 1; i++)
                  msg[i] = error_message[i + 1];
               msg[i - 1] = '\0';
               error_message = msg;
            }

            else
               error_message += offset;
      }

      else
      {
         if (png_ptr->flags&PNG_FLAG_STRIP_ERROR_TEXT)
         {
            msg[0] = '0';
            msg[1] = '\0';
            error_message = msg;
         }
       }
     }
   }
#endif
   if (png_ptr != NULL && png_ptr->error_fn != NULL)
      (*(png_ptr->error_fn))(png_ptr, error_message);

   /* If the custom handler doesn't exist, or if it returns,
      use the default handler, which will not return. */
   png_default_error(png_ptr, error_message);
}
#else
PNG_FUNCTION(void,PNGAPI
png_err,(png_structp png_ptr),PNG_NORETURN)
{
   /* Prior to 1.5.2 the error_fn received a NULL pointer, expressed
    * erroneously as '\0', instead of the empty string "".  This was
    * apparently an error, introduced in libpng-1.2.20, and png_default_error
    * will crash in this case.
    */
   if (png_ptr != NULL && png_ptr->error_fn != NULL)
      (*(png_ptr->error_fn))(png_ptr, "");

   /* If the custom handler doesn't exist, or if it returns,
      use the default handler, which will not return. */
   png_default_error(png_ptr, "");
}
#endif /* PNG_ERROR_TEXT_SUPPORTED */

/* Utility to safely appends strings to a buffer.  This never errors out so
 * error checking is not required in the caller.
 */
size_t
png_safecat(png_charp buffer, size_t bufsize, size_t pos,
   png_const_charp string)
{
   if (buffer != NULL && pos < bufsize)
   {
      if (string != NULL)
         while (*string != '\0' && pos < bufsize-1)
           buffer[pos++] = *string++;

      buffer[pos] = '\0';
   }

   return pos;
}

#if defined(PNG_WARNINGS_SUPPORTED) || defined(PNG_TIME_RFC1123_SUPPORTED)
/* Utility to dump an unsigned value into a buffer, given a start pointer and
 * and end pointer (which should point just *beyond* the end of the buffer!)
 * Returns the pointer to the start of the formatted string.
 */
png_charp
png_format_number(png_const_charp start, png_charp end, int format,
   png_alloc_size_t number)
{
   int count = 0;    /* number of digits output */
   int mincount = 1; /* minimum number required */
   int output = 0;   /* digit output (for the fixed point format) */

   *--end = '\0';

   /* This is written so that the loop always runs at least once, even with
    * number zero.
    */
   while (end > start && (number != 0 || count < mincount))
   {

      static const char digits[] = "0123456789ABCDEF";

      switch (format)
      {
         case PNG_NUMBER_FORMAT_fixed:
            /* Needs five digits (the fraction) */
            mincount = 5;
            if (output || number % 10 != 0)
            {
               *--end = digits[number % 10];
               output = 1;
            }
            number /= 10;
            break;

         case PNG_NUMBER_FORMAT_02u:
            /* Expects at least 2 digits. */
            mincount = 2;
            /* FALL THROUGH */

         case PNG_NUMBER_FORMAT_u:
            *--end = digits[number % 10];
            number /= 10;
            break;

         case PNG_NUMBER_FORMAT_02x:
            /* This format expects at least two digits */
            mincount = 2;
            /* FALL THROUGH */

         case PNG_NUMBER_FORMAT_x:
            *--end = digits[number & 0xf];
            number >>= 4;
            break;

         default: /* an error */
            number = 0;
            break;
      }

      /* Keep track of the number of digits added */
      ++count;

      /* Float a fixed number here: */
      if (format == PNG_NUMBER_FORMAT_fixed) if (count == 5) if (end > start)
      {
         /* End of the fraction, but maybe nothing was output?  In that case
          * drop the decimal point.  If the number is a true zero handle that
          * here.
          */
         if (output != 0)
            *--end = '.';
         else if (number == 0) /* and !output */
            *--end = '0';
      }
   }

   return end;
}
#endif

#ifdef PNG_WARNINGS_SUPPORTED
/* This function is called whenever there is a non-fatal error.  This function
 * should not be changed.  If there is a need to handle warnings differently,
 * you should supply a replacement warning function and use
 * png_set_error_fn() to replace the warning function at run-time.
 */
void PNGAPI
png_warning(png_structp png_ptr, png_const_charp warning_message)
{
   int offset = 0;
   if (png_ptr != NULL)
   {
#ifdef PNG_ERROR_NUMBERS_SUPPORTED
   if (png_ptr->flags&
       (PNG_FLAG_STRIP_ERROR_NUMBERS|PNG_FLAG_STRIP_ERROR_TEXT))
#endif
      {
         if (*warning_message == PNG_LITERAL_SHARP)
         {
            for (offset = 1; offset < 15; offset++)
               if (warning_message[offset] == ' ')
                  break;
         }
      }
   }
   if (png_ptr != NULL && png_ptr->warning_fn != NULL)
      (*(png_ptr->warning_fn))(png_ptr, warning_message + offset);
   else
      png_default_warning(png_ptr, warning_message + offset);
}

/* These functions support 'formatted' warning messages with up to
 * PNG_WARNING_PARAMETER_COUNT parameters.  In the format string the parameter
 * is introduced by @<number>, where 'number' starts at 1.  This follows the
 * standard established by X/Open for internationalizable error messages.
 */
void
png_warning_parameter(png_warning_parameters p, int number,
   png_const_charp string)
{
   if (number > 0 && number <= PNG_WARNING_PARAMETER_COUNT)
      (void)png_safecat(p[number-1], (sizeof p[number-1]), 0, string);
}

void
png_warning_parameter_unsigned(png_warning_parameters p, int number, int format,
   png_alloc_size_t value)
{
   char buffer[PNG_NUMBER_BUFFER_SIZE];
   png_warning_parameter(p, number, PNG_FORMAT_NUMBER(buffer, format, value));
}

void
png_warning_parameter_signed(png_warning_parameters p, int number, int format,
   png_int_32 value)
{
   png_alloc_size_t u;
   png_charp str;
   char buffer[PNG_NUMBER_BUFFER_SIZE];

   /* Avoid overflow by doing the negate in a png_alloc_size_t: */
   u = (png_alloc_size_t)value;
   if (value < 0)
      u = ~u + 1;

   str = PNG_FORMAT_NUMBER(buffer, format, u);

   if (value < 0 && str > buffer)
      *--str = '-';

   png_warning_parameter(p, number, str);
}

void
png_formatted_warning(png_structp png_ptr, png_warning_parameters p,
   png_const_charp message)
{
   /* The internal buffer is just 192 bytes - enough for all our messages,
    * overflow doesn't happen because this code checks!  If someone figures
    * out how to send us a message longer than 192 bytes, all that will
    * happen is that the message will be truncated appropriately.
    */
   size_t i = 0; /* Index in the msg[] buffer: */
   char msg[192];

   /* Each iteration through the following loop writes at most one character
    * to msg[i++] then returns here to validate that there is still space for
    * the trailing '\0'.  It may (in the case of a parameter) read more than
    * one character from message[]; it must check for '\0' and continue to the
    * test if it finds the end of string.
    */
   while (i<(sizeof msg)-1 && *message != '\0')
   {
      /* '@' at end of string is now just printed (previously it was skipped);
       * it is an error in the calling code to terminate the string with @.
       */
      if (p != NULL && *message == '@' && message[1] != '\0')
      {
         int parameter_char = *++message; /* Consume the '@' */
         static const char valid_parameters[] = "123456789";
         int parameter = 0;

         /* Search for the parameter digit, the index in the string is the
          * parameter to use.
          */
         while (valid_parameters[parameter] != parameter_char &&
            valid_parameters[parameter] != '\0')
            ++parameter;

         /* If the parameter digit is out of range it will just get printed. */
         if (parameter < PNG_WARNING_PARAMETER_COUNT)
         {
            /* Append this parameter */
            png_const_charp parm = p[parameter];
            png_const_charp pend = p[parameter] + (sizeof p[parameter]);

            /* No need to copy the trailing '\0' here, but there is no guarantee
             * that parm[] has been initialized, so there is no guarantee of a
             * trailing '\0':
             */
            while (i<(sizeof msg)-1 && *parm != '\0' && parm < pend)
               msg[i++] = *parm++;

            /* Consume the parameter digit too: */
            ++message;
            continue;
         }

         /* else not a parameter and there is a character after the @ sign; just
          * copy that.  This is known not to be '\0' because of the test above.
          */
      }

      /* At this point *message can't be '\0', even in the bad parameter case
       * above where there is a lone '@' at the end of the message string.
       */
      msg[i++] = *message++;
   }

   /* i is always less than (sizeof msg), so: */
   msg[i] = '\0';

   /* And this is the formatted message, it may be larger than
    * PNG_MAX_ERROR_TEXT, but that is only used for 'chunk' errors and these are
    * not (currently) formatted.
    */
   png_warning(png_ptr, msg);
}
#endif /* PNG_WARNINGS_SUPPORTED */

#ifdef PNG_BENIGN_ERRORS_SUPPORTED
void PNGAPI
png_benign_error(png_structp png_ptr, png_const_charp error_message)
{
  if (png_ptr->flags & PNG_FLAG_BENIGN_ERRORS_WARN)
     png_warning(png_ptr, error_message);
  else
     png_error(png_ptr, error_message);
}
#endif

/* These utilities are used internally to build an error message that relates
 * to the current chunk.  The chunk name comes from png_ptr->chunk_name,
 * this is used to prefix the message.  The message is limited in length
 * to 63 bytes, the name characters are output as hex digits wrapped in []
 * if the character is invalid.
 */
#define isnonalpha(c) ((c) < 65 || (c) > 122 || ((c) > 90 && (c) < 97))
static PNG_CONST char png_digit[16] = {
   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
   'A', 'B', 'C', 'D', 'E', 'F'
};

#define PNG_MAX_ERROR_TEXT 64
#if defined(PNG_WARNINGS_SUPPORTED) || defined(PNG_ERROR_TEXT_SUPPORTED)
static void /* PRIVATE */
png_format_buffer(png_structp png_ptr, png_charp buffer, png_const_charp
    error_message)
{
   png_uint_32 chunk_name = png_ptr->chunk_name;
   int iout = 0, ishift = 24;

   while (ishift >= 0)
   {
      int c = (int)(chunk_name >> ishift) & 0xff;

      ishift -= 8;
      if (isnonalpha(c))
      {
         buffer[iout++] = PNG_LITERAL_LEFT_SQUARE_BRACKET;
         buffer[iout++] = png_digit[(c & 0xf0) >> 4];
         buffer[iout++] = png_digit[c & 0x0f];
         buffer[iout++] = PNG_LITERAL_RIGHT_SQUARE_BRACKET;
      }

      else
      {
         buffer[iout++] = (char)c;
      }
   }

   if (error_message == NULL)
      buffer[iout] = '\0';

   else
   {
      int iin = 0;

      buffer[iout++] = ':';
      buffer[iout++] = ' ';

      while (iin < PNG_MAX_ERROR_TEXT-1 && error_message[iin] != '\0')
         buffer[iout++] = error_message[iin++];

      /* iin < PNG_MAX_ERROR_TEXT, so the following is safe: */
      buffer[iout] = '\0';
   }
}
#endif /* PNG_WARNINGS_SUPPORTED || PNG_ERROR_TEXT_SUPPORTED */

#if defined(PNG_READ_SUPPORTED) && defined(PNG_ERROR_TEXT_SUPPORTED)
PNG_FUNCTION(void,PNGAPI
png_chunk_error,(png_structp png_ptr, png_const_charp error_message),
   PNG_NORETURN)
{
   char msg[18+PNG_MAX_ERROR_TEXT];
   if (png_ptr == NULL)
      png_error(png_ptr, error_message);

   else
   {
      png_format_buffer(png_ptr, msg, error_message);
      png_error(png_ptr, msg);
   }
}
#endif /* PNG_READ_SUPPORTED && PNG_ERROR_TEXT_SUPPORTED */

#ifdef PNG_WARNINGS_SUPPORTED
void PNGAPI
png_chunk_warning(png_structp png_ptr, png_const_charp warning_message)
{
   char msg[18+PNG_MAX_ERROR_TEXT];
   if (png_ptr == NULL)
      png_warning(png_ptr, warning_message);

   else
   {
      png_format_buffer(png_ptr, msg, warning_message);
      png_warning(png_ptr, msg);
   }
}
#endif /* PNG_WARNINGS_SUPPORTED */

#ifdef PNG_READ_SUPPORTED
#ifdef PNG_BENIGN_ERRORS_SUPPORTED
void PNGAPI
png_chunk_benign_error(png_structp png_ptr, png_const_charp error_message)
{
   if (png_ptr->flags & PNG_FLAG_BENIGN_ERRORS_WARN)
      png_chunk_warning(png_ptr, error_message);

   else
      png_chunk_error(png_ptr, error_message);
}
#endif
#endif /* PNG_READ_SUPPORTED */

#ifdef PNG_ERROR_TEXT_SUPPORTED
#ifdef PNG_FLOATING_POINT_SUPPORTED
PNG_FUNCTION(void,
png_fixed_error,(png_structp png_ptr, png_const_charp name),PNG_NORETURN)
{
#  define fixed_message "fixed point overflow in "
#  define fixed_message_ln ((sizeof fixed_message)-1)
   int  iin;
   char msg[fixed_message_ln+PNG_MAX_ERROR_TEXT];
   png_memcpy(msg, fixed_message, fixed_message_ln);
   iin = 0;
   if (name != NULL) while (iin < (PNG_MAX_ERROR_TEXT-1) && name[iin] != 0)
   {
      msg[fixed_message_ln + iin] = name[iin];
      ++iin;
   }
   msg[fixed_message_ln + iin] = 0;
   png_error(png_ptr, msg);
}
#endif
#endif

#ifdef PNG_SETJMP_SUPPORTED
/* This API only exists if ANSI-C style error handling is used,
 * otherwise it is necessary for png_default_error to be overridden.
 */
jmp_buf* PNGAPI
png_set_longjmp_fn(png_structp png_ptr, png_longjmp_ptr longjmp_fn,
    size_t jmp_buf_size)
{
   if (png_ptr == NULL || jmp_buf_size != png_sizeof(jmp_buf))
      return NULL;

   png_ptr->longjmp_fn = longjmp_fn;
   return &png_ptr->longjmp_buffer;
}
#endif

/* This is the default error handling function.  Note that replacements for
 * this function MUST NOT RETURN, or the program will likely crash.  This
 * function is used by default, or if the program supplies NULL for the
 * error function pointer in png_set_error_fn().
 */
static PNG_FUNCTION(void /* PRIVATE */,
png_default_error,(png_structp png_ptr, png_const_charp error_message),
   PNG_NORETURN)
{
#ifdef PNG_CONSOLE_IO_SUPPORTED
#ifdef PNG_ERROR_NUMBERS_SUPPORTED
   /* Check on NULL only added in 1.5.4 */
   if (error_message != NULL && *error_message == PNG_LITERAL_SHARP)
   {
      /* Strip "#nnnn " from beginning of error message. */
      int offset;
      char error_number[16];
      for (offset = 0; offset<15; offset++)
      {
         error_number[offset] = error_message[offset + 1];
         if (error_message[offset] == ' ')
            break;
      }

      if ((offset > 1) && (offset < 15))
      {
         error_number[offset - 1] = '\0';
         fprintf(stderr, "libpng error no. %s: %s",
             error_number, error_message + offset + 1);
         fprintf(stderr, PNG_STRING_NEWLINE);
      }

      else
      {
         fprintf(stderr, "libpng error: %s, offset=%d",
             error_message, offset);
         fprintf(stderr, PNG_STRING_NEWLINE);
      }
   }
   else
#endif
   {
      fprintf(stderr, "libpng error: %s", error_message ? error_message :
         "undefined");
      fprintf(stderr, PNG_STRING_NEWLINE);
   }
#else
   PNG_UNUSED(error_message) /* Make compiler happy */
#endif
   png_longjmp(png_ptr, 1);
}

PNG_FUNCTION(void,PNGAPI
png_longjmp,(png_structp png_ptr, int val),PNG_NORETURN)
{
#ifdef PNG_SETJMP_SUPPORTED
   if (png_ptr && png_ptr->longjmp_fn)
   {
#  ifdef USE_FAR_KEYWORD
      {
         jmp_buf tmp_jmpbuf;
         png_memcpy(tmp_jmpbuf, png_ptr->longjmp_buffer, png_sizeof(jmp_buf));
         png_ptr->longjmp_fn(tmp_jmpbuf, val);
      }

#  else
   png_ptr->longjmp_fn(png_ptr->longjmp_buffer, val);
#  endif
   }
#else
   PNG_UNUSED(png_ptr);
   PNG_UNUSED(val);
#endif
   /* Here if not setjmp support or if png_ptr is null. */
   PNG_ABORT();
}

#ifdef PNG_WARNINGS_SUPPORTED
/* This function is called when there is a warning, but the library thinks
 * it can continue anyway.  Replacement functions don't have to do anything
 * here if you don't want them to.  In the default configuration, png_ptr is
 * not used, but it is passed in case it may be useful.
 */
static void /* PRIVATE */
png_default_warning(png_structp png_ptr, png_const_charp warning_message)
{
#ifdef PNG_CONSOLE_IO_SUPPORTED
#  ifdef PNG_ERROR_NUMBERS_SUPPORTED
   if (*warning_message == PNG_LITERAL_SHARP)
   {
      int offset;
      char warning_number[16];
      for (offset = 0; offset < 15; offset++)
      {
         warning_number[offset] = warning_message[offset + 1];
         if (warning_message[offset] == ' ')
            break;
      }

      if ((offset > 1) && (offset < 15))
      {
         warning_number[offset + 1] = '\0';
         fprintf(stderr, "libpng warning no. %s: %s",
             warning_number, warning_message + offset);
         fprintf(stderr, PNG_STRING_NEWLINE);
      }

      else
      {
         fprintf(stderr, "libpng warning: %s",
             warning_message);
         fprintf(stderr, PNG_STRING_NEWLINE);
      }
   }
   else
#  endif

   {
      fprintf(stderr, "libpng warning: %s", warning_message);
      fprintf(stderr, PNG_STRING_NEWLINE);
   }
#else
   PNG_UNUSED(warning_message) /* Make compiler happy */
#endif
   PNG_UNUSED(png_ptr) /* Make compiler happy */
}
#endif /* PNG_WARNINGS_SUPPORTED */

/* This function is called when the application wants to use another method
 * of handling errors and warnings.  Note that the error function MUST NOT
 * return to the calling routine or serious problems will occur.  The return
 * method used in the default routine calls longjmp(png_ptr->longjmp_buffer, 1)
 */
void PNGAPI
png_set_error_fn(png_structp png_ptr, png_voidp error_ptr,
    png_error_ptr error_fn, png_error_ptr warning_fn)
{
   if (png_ptr == NULL)
      return;

   png_ptr->error_ptr = error_ptr;
   png_ptr->error_fn = error_fn;
#ifdef PNG_WARNINGS_SUPPORTED
   png_ptr->warning_fn = warning_fn;
#else
   PNG_UNUSED(warning_fn)
#endif
}


/* This function returns a pointer to the error_ptr associated with the user
 * functions.  The application should free any memory associated with this
 * pointer before png_write_destroy and png_read_destroy are called.
 */
png_voidp PNGAPI
png_get_error_ptr(png_const_structp png_ptr)
{
   if (png_ptr == NULL)
      return NULL;

   return ((png_voidp)png_ptr->error_ptr);
}


#ifdef PNG_ERROR_NUMBERS_SUPPORTED
void PNGAPI
png_set_strip_error_numbers(png_structp png_ptr, png_uint_32 strip_mode)
{
   if (png_ptr != NULL)
   {
      png_ptr->flags &=
         ((~(PNG_FLAG_STRIP_ERROR_NUMBERS |
         PNG_FLAG_STRIP_ERROR_TEXT))&strip_mode);
   }
}
#endif
#endif /* PNG_READ_SUPPORTED || PNG_WRITE_SUPPORTED */
