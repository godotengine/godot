#ifndef C4_EXPORT_HPP_
#define C4_EXPORT_HPP_

#ifdef _WIN32
    #ifdef C4CORE_SHARED
        #ifdef C4CORE_EXPORTS
            #define C4CORE_EXPORT __declspec(dllexport)
        #else
            #define C4CORE_EXPORT __declspec(dllimport)
        #endif
    #else
        #define C4CORE_EXPORT
    #endif
#else
    #define C4CORE_EXPORT
#endif

#endif /* C4CORE_EXPORT_HPP_ */
