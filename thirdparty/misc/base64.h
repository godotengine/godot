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

uint32_t base64_encode(char *to, char *from, uint32_t len);
uint32_t base64_decode(char *to, char *from, uint32_t len);
};

#endif /* BASE64_H */
