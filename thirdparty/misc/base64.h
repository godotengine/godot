/*
 * File: base64.h
 * Description: Simple BASE64 conversion methods
 * Author: Ari Edelkind
 * License: Public Domain
 * Website: http://episec.com/people/edelkind/c.html
 */

#ifndef BASE64_H
#define BASE64_H

extern "C" {

long base64_encode(char *to, char *from, unsigned int len);
long base64_decode(char *to, char *from, unsigned int len);
};

#endif /* BASE64_H */
