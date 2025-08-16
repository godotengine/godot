#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <httplib.h>

using namespace std;
using namespace httplib;

std::string normalizeJson(const std::string &json) {
  std::string result;
  for (char c : json) {
    if (c != ' ' && c != '\t' && c != '\n' && c != '\r') { result += c; }
  }
  return result;
}

template <typename T> void ProxyTest(T &cli, bool basic) {
  cli.set_proxy("localhost", basic ? 3128 : 3129);
  auto res = cli.Get("/httpbin/get");
  ASSERT_TRUE(res != nullptr);
  EXPECT_EQ(StatusCode::ProxyAuthenticationRequired_407, res->status);
}

TEST(ProxyTest, NoSSLBasic) {
  Client cli("nghttp2.org");
  ProxyTest(cli, true);
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
TEST(ProxyTest, SSLBasic) {
  SSLClient cli("nghttp2.org");
  ProxyTest(cli, true);
}

TEST(ProxyTest, NoSSLDigest) {
  Client cli("nghttp2.org");
  ProxyTest(cli, false);
}

TEST(ProxyTest, SSLDigest) {
  SSLClient cli("nghttp2.org");
  ProxyTest(cli, false);
}
#endif

// ----------------------------------------------------------------------------

template <typename T>
void RedirectProxyText(T &cli, const char *path, bool basic) {
  cli.set_proxy("localhost", basic ? 3128 : 3129);
  if (basic) {
    cli.set_proxy_basic_auth("hello", "world");
  } else {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    cli.set_proxy_digest_auth("hello", "world");
#endif
  }
  cli.set_follow_location(true);

  auto res = cli.Get(path);
  ASSERT_TRUE(res != nullptr);
  EXPECT_EQ(StatusCode::OK_200, res->status);
}

TEST(RedirectTest, HTTPBinNoSSLBasic) {
  Client cli("nghttp2.org");
  RedirectProxyText(cli, "/httpbin/redirect/2", true);
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
TEST(RedirectTest, HTTPBinNoSSLDigest) {
  Client cli("nghttp2.org");
  RedirectProxyText(cli, "/httpbin/redirect/2", false);
}

TEST(RedirectTest, HTTPBinSSLBasic) {
  SSLClient cli("nghttp2.org");
  RedirectProxyText(cli, "/httpbin/redirect/2", true);
}

TEST(RedirectTest, HTTPBinSSLDigest) {
  SSLClient cli("nghttp2.org");
  RedirectProxyText(cli, "/httpbin/redirect/2", false);
}
#endif

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
TEST(RedirectTest, YouTubeNoSSLBasic) {
  Client cli("youtube.com");
  RedirectProxyText(cli, "/", true);
}

TEST(RedirectTest, YouTubeNoSSLDigest) {
  Client cli("youtube.com");
  RedirectProxyText(cli, "/", false);
}

TEST(RedirectTest, YouTubeSSLBasic) {
  SSLClient cli("youtube.com");
  RedirectProxyText(cli, "/", true);
}

TEST(RedirectTest, YouTubeSSLDigest) {
  std::this_thread::sleep_for(std::chrono::seconds(3));
  SSLClient cli("youtube.com");
  RedirectProxyText(cli, "/", false);
}
#endif

// ----------------------------------------------------------------------------

template <typename T> void BaseAuthTestFromHTTPWatch(T &cli) {
  cli.set_proxy("localhost", 3128);
  cli.set_proxy_basic_auth("hello", "world");

  {
    auto res = cli.Get("/basic-auth/hello/world");
    ASSERT_TRUE(res != nullptr);
    EXPECT_EQ(StatusCode::Unauthorized_401, res->status);
  }

  {
    auto res = cli.Get("/basic-auth/hello/world",
                       {make_basic_authentication_header("hello", "world")});
    ASSERT_TRUE(res != nullptr);
    EXPECT_EQ(normalizeJson("{\"authenticated\":true,\"user\":\"hello\"}\n"),
              normalizeJson(res->body));
    EXPECT_EQ(StatusCode::OK_200, res->status);
  }

  {
    cli.set_basic_auth("hello", "world");
    auto res = cli.Get("/basic-auth/hello/world");
    ASSERT_TRUE(res != nullptr);
    EXPECT_EQ(normalizeJson("{\"authenticated\":true,\"user\":\"hello\"}\n"),
              normalizeJson(res->body));
    EXPECT_EQ(StatusCode::OK_200, res->status);
  }

  {
    cli.set_basic_auth("hello", "bad");
    auto res = cli.Get("/basic-auth/hello/world");
    ASSERT_TRUE(res != nullptr);
    EXPECT_EQ(StatusCode::Unauthorized_401, res->status);
  }

  {
    cli.set_basic_auth("bad", "world");
    auto res = cli.Get("/basic-auth/hello/world");
    ASSERT_TRUE(res != nullptr);
    EXPECT_EQ(StatusCode::Unauthorized_401, res->status);
  }
}

TEST(BaseAuthTest, NoSSL) {
  Client cli("httpbin.org");
  BaseAuthTestFromHTTPWatch(cli);
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
TEST(BaseAuthTest, SSL) {
  SSLClient cli("httpbin.org");
  BaseAuthTestFromHTTPWatch(cli);
}
#endif

// ----------------------------------------------------------------------------

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
template <typename T> void DigestAuthTestFromHTTPWatch(T &cli) {
  cli.set_proxy("localhost", 3129);
  cli.set_proxy_digest_auth("hello", "world");

  {
    auto res = cli.Get("/digest-auth/auth/hello/world");
    ASSERT_TRUE(res != nullptr);
    EXPECT_EQ(StatusCode::Unauthorized_401, res->status);
  }

  {
    std::vector<std::string> paths = {
        "/digest-auth/auth/hello/world/MD5",
        "/digest-auth/auth/hello/world/SHA-256",
        "/digest-auth/auth/hello/world/SHA-512",
        "/digest-auth/auth-int/hello/world/MD5",
    };

    cli.set_digest_auth("hello", "world");
    for (auto path : paths) {
      auto res = cli.Get(path.c_str());
      ASSERT_TRUE(res != nullptr);
      EXPECT_EQ(normalizeJson("{\"authenticated\":true,\"user\":\"hello\"}\n"),
                normalizeJson(res->body));
      EXPECT_EQ(StatusCode::OK_200, res->status);
    }

    cli.set_digest_auth("hello", "bad");
    for (auto path : paths) {
      auto res = cli.Get(path.c_str());
      ASSERT_TRUE(res != nullptr);
      EXPECT_EQ(StatusCode::Unauthorized_401, res->status);
    }

    // NOTE: Until httpbin.org fixes issue #46, the following test is commented
    // out. Please see https://httpbin.org/digest-auth/auth/hello/world
    // cli.set_digest_auth("bad", "world");
    // for (auto path : paths) {
    //   auto res = cli.Get(path.c_str());
    //   ASSERT_TRUE(res != nullptr);
    //   EXPECT_EQ(StatusCode::Unauthorized_401, res->status);
    // }
  }
}

TEST(DigestAuthTest, SSL) {
  SSLClient cli("httpbin.org");
  DigestAuthTestFromHTTPWatch(cli);
}

TEST(DigestAuthTest, NoSSL) {
  Client cli("httpbin.org");
  DigestAuthTestFromHTTPWatch(cli);
}
#endif

// ----------------------------------------------------------------------------

template <typename T> void KeepAliveTest(T &cli, bool basic) {
  cli.set_proxy("localhost", basic ? 3128 : 3129);
  if (basic) {
    cli.set_proxy_basic_auth("hello", "world");
  } else {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    cli.set_proxy_digest_auth("hello", "world");
#endif
  }

  cli.set_follow_location(true);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
  cli.set_digest_auth("hello", "world");
#endif

  {
    auto res = cli.Get("/httpbin/get");
    EXPECT_EQ(StatusCode::OK_200, res->status);
  }
  {
    auto res = cli.Get("/httpbin/redirect/2");
    EXPECT_EQ(StatusCode::OK_200, res->status);
  }

  {
    std::vector<std::string> paths = {
        "/httpbin/digest-auth/auth/hello/world/MD5",
        "/httpbin/digest-auth/auth/hello/world/SHA-256",
        "/httpbin/digest-auth/auth/hello/world/SHA-512",
        "/httpbin/digest-auth/auth-int/hello/world/MD5",
    };

    for (auto path : paths) {
      auto res = cli.Get(path.c_str());
      EXPECT_EQ(normalizeJson("{\"authenticated\":true,\"user\":\"hello\"}\n"),
                normalizeJson(res->body));
      EXPECT_EQ(StatusCode::OK_200, res->status);
    }
  }

  {
    int count = 10;
    while (count--) {
      auto res = cli.Get("/httpbin/get");
      EXPECT_EQ(StatusCode::OK_200, res->status);
    }
  }
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
TEST(KeepAliveTest, NoSSLWithBasic) {
  Client cli("nghttp2.org");
  KeepAliveTest(cli, true);
}

TEST(KeepAliveTest, SSLWithBasic) {
  SSLClient cli("nghttp2.org");
  KeepAliveTest(cli, true);
}

TEST(KeepAliveTest, NoSSLWithDigest) {
  Client cli("nghttp2.org");
  KeepAliveTest(cli, false);
}

TEST(KeepAliveTest, SSLWithDigest) {
  SSLClient cli("nghttp2.org");
  KeepAliveTest(cli, false);
}
#endif
