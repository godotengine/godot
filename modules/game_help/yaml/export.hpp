#ifndef C4_YML_EXPORT_HPP_
#define C4_YML_EXPORT_HPP_

#ifdef _WIN32
    #ifdef RYML_SHARED
        #ifdef RYML_EXPORTS
            #define RYML_EXPORT __declspec(dllexport)
        #else
            #define RYML_EXPORT __declspec(dllimport)
        #endif
    #else
        #define RYML_EXPORT
    #endif
#else
    #define RYML_EXPORT
#endif

#endif /* C4_YML_EXPORT_HPP_ */
