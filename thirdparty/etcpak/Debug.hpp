#ifndef __DARKRL__DEBUG_HPP__
#define __DARKRL__DEBUG_HPP__

#ifdef DEBUG
#  include <sstream>
#  define DBGPRINT(msg) { std::stringstream __buf; __buf << msg; DebugLog::Message( __buf.str().c_str() ); }
#else
#  define DBGPRINT(msg) ((void)0)
#endif

class DebugLog
{
public:
    struct Callback
    {
        virtual void OnDebugMessage( const char* msg ) = 0;
    };

    static void Message( const char* msg );
    static void AddCallback( Callback* c );
    static void RemoveCallback( Callback* c );

private:
    DebugLog() {}
};

#endif
