#pragma once

#if !defined(TINYUSDZ_PRODUCTION_BUILD) && !defined(TINYUSDZ_FUZZER_BUILD)
#if defined(TINYUSDZ_DEBUG_PRINT)
#define TINYUSDZ_LOCAL_DEBUG_PRINT
#endif
#endif

#if defined(TINYUSDZ_PRODUCTION_BUILD)
// Do not include full filepath for privacy.

#define PUSH_ERROR_AND_RETURN(s)                         \
  do {                                                   \
    std::ostringstream ss_e;                             \
    ss_e << "[error]"                                    \
         << ":" << __func__ << "():" << __LINE__ << " "; \
    ss_e << s << "\n";                                   \
    PushError(ss_e.str());                               \
    return false;                                        \
  } while (0)

#define PUSH_ERROR_AND_RETURN_TAG(tag, s)                                    \
  do {                                                                       \
    std::ostringstream ss_e;                                                 \
    ss_e << "[error]" << tag << ":" << __func__ << "():" << __LINE__ << " "; \
    ss_e << s << "\n";                                                       \
    PushError(ss_e.str());                                                   \
    return false;                                                            \
  } while (0)

#define PUSH_ERROR(s)                                    \
  do {                                                   \
    std::ostringstream ss_e;                             \
    ss_e << "[error]"                                    \
         << ":" << __func__ << "():" << __LINE__ << " "; \
    ss_e << s << "\n";                                   \
    PushError(ss_e.str());                               \
  } while (0)

#define PUSH_WARN(s)                                     \
  do {                                                   \
    std::ostringstream ss_w;                             \
    ss_w << "[warn]"                                     \
         << ":" << __func__ << "():" << __LINE__ << " "; \
    ss_w << s << "\n";                                   \
    PushWarn(ss_w.str());                                \
  } while (0)

#else  // TINYUSDZ_PRODUCTION_BUILD

#define PUSH_ERROR_AND_RETURN(s)                                          \
  do {                                                                    \
    std::ostringstream ss_e;                                              \
    ss_e << "[error]" << __FILE__ << ":" << __func__ << "():" << __LINE__ \
         << " ";                                                          \
    ss_e << s << "\n";                                                    \
    PushError(ss_e.str());                                                \
    return false;                                                         \
  } while (0)

#define PUSH_ERROR_AND_RETURN_TAG(tag, s)                   \
  do {                                                      \
    std::ostringstream ss_e;                                \
    ss_e << "[error]" << __FILE__ << tag << ":" << __func__ \
         << "():" << __LINE__ << " ";                       \
    ss_e << s << "\n";                                      \
    PushError(ss_e.str());                                  \
    return false;                                           \
  } while (0)

#define PUSH_ERROR(s)                                                     \
  do {                                                                    \
    std::ostringstream ss_e;                                              \
    ss_e << "[error]" << __FILE__ << ":" << __func__ << "():" << __LINE__ \
         << " ";                                                          \
    ss_e << s << "\n";                                                    \
    PushError(ss_e.str());                                                \
  } while (0)

#define PUSH_WARN(s)                                                     \
  do {                                                                   \
    std::ostringstream ss_w;                                             \
    ss_w << "[warn]" << __FILE__ << ":" << __func__ << "():" << __LINE__ \
         << " ";                                                         \
    ss_w << s << "\n";                                                   \
    PushWarn(ss_w.str());                                                \
  } while (0)

#endif  // TINYUSDZ_PRODUCTION_BUILD

#if defined(TINYUSDZ_LOCAL_DEBUG_PRINT)
#define DCOUT(x)                                               \
  do {                                                         \
    std::cout << __FILE__ << ":" << __func__ << ":"            \
              << std::to_string(__LINE__) << " " << x << "\n"; \
  } while (false)
#else
#define DCOUT(x)
#endif

// Simple auto-free class
// Use this class when saving stack size is required(e.g. recursive function call).
// T must have default constructor
template<typename T>
class AutoFree {
 private:
  T *_v{nullptr}; 

 public:

  AutoFree() : _v(new T()) { 
  }

  ~AutoFree() {
    delete _v;
  }

  T &value() { return *_v; }
};
