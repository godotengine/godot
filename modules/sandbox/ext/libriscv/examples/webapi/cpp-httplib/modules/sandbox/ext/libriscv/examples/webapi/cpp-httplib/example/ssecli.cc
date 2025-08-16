//
//  ssecli.cc
//
//  Copyright (c) 2019 Yuji Hirose. All rights reserved.
//  MIT License
//

#include <httplib.h>
#include <iostream>

using namespace std;

int main(void) {
  httplib::Client("http://localhost:1234")
      .Get("/event1", [&](const char *data, size_t data_length) {
        std::cout << string(data, data_length);
        return true;
      });

  return 0;
}
