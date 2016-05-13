#ifndef BASE64_H
#define BASE64_H

extern "C" {

uint32_t base64_encode (char* to, char* from, uint32_t len);
uint32_t base64_decode (char* to, char* from, uint32_t len);

};

#endif /* BASE64_H */
