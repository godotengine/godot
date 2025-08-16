//
//  redirect.cc
//
//  Copyright (c) 2019 Yuji Hirose. All rights reserved.
//  MIT License
//

#include <httplib.h>

#define SERVER_CERT_FILE "./cert.pem"
#define SERVER_PRIVATE_KEY_FILE "./key.pem"

using namespace httplib;

int main(void) {
  // HTTP server
  Server http;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  SSLServer https(SERVER_CERT_FILE, SERVER_PRIVATE_KEY_FILE);
#endif

  http.Get("/test", [](const Request & /*req*/, Response &res) {
    res.set_content("Test\n", "text/plain");
  });

  http.set_error_handler([](const Request & /*req*/, Response &res) {
    res.set_redirect("https://localhost:8081/");
  });

  // HTTPS server
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  https.Get("/", [=](const Request & /*req*/, Response &res) {
    res.set_redirect("/hi");
  });

  https.Get("/hi", [](const Request & /*req*/, Response &res) {
    res.set_content("Hello World!\n", "text/plain");
  });

  https.Get("/stop", [&](const Request & /*req*/, Response & /*res*/) {
    https.stop();
    http.stop();
  });
#endif

  // Run servers
  auto httpThread = std::thread([&]() { http.listen("localhost", 8080); });

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  auto httpsThread = std::thread([&]() { https.listen("localhost", 8081); });
#endif

  httpThread.join();

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  httpsThread.join();
#endif

  return 0;
}
