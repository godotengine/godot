
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

// 210-Evt-EventListeners.cpp

// Contents:
// 1. Printing of listener data
// 2. My listener and registration
// 3. Test cases

#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <iostream>

// -----------------------------------------------------------------------
// 1. Printing of listener data:
//


namespace {
std::string ws(int const level) {
    return std::string( 2 * level, ' ' );
}

std::ostream& operator<<(std::ostream& out, Catch::Tag t) {
    return out << "original: " << t.original;
}

template< typename T >
std::ostream& operator<<( std::ostream& os, std::vector<T> const& v ) {
    os << "{ ";
    for ( const auto& x : v )
        os << x << ", ";
    return os << "}";
}
// struct SourceLineInfo {
//     char const* file;
//     std::size_t line;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::SourceLineInfo const& info ) {
    os << ws(level  ) << title << ":\n"
       << ws(level+1) << "- file: " << info.file << "\n"
       << ws(level+1) << "- line: " << info.line << "\n";
}

//struct MessageInfo {
//    std::string macroName;
//    std::string message;
//    SourceLineInfo lineInfo;
//    ResultWas::OfType type;
//    unsigned int sequence;
//};

void print( std::ostream& os, int const level, Catch::MessageInfo const& info ) {
    os << ws(level+1) << "- macroName: '" << info.macroName << "'\n"
       << ws(level+1) << "- message '"    << info.message   << "'\n";
    print( os,level+1  , "- lineInfo", info.lineInfo );
    os << ws(level+1) << "- sequence "    << info.sequence  << "\n";
}

void print( std::ostream& os, int const level, std::string const& title, std::vector<Catch::MessageInfo> const& v ) {
    os << ws(level  ) << title << ":\n";
    for ( const auto& x : v )
    {
        os << ws(level+1) << "{\n";
        print( os, level+2, x );
        os << ws(level+1) << "}\n";
    }
//    os << ws(level+1) << "\n";
}

// struct TestRunInfo {
//     std::string name;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::TestRunInfo const& info ) {
    os << ws(level  ) << title << ":\n"
       << ws(level+1) << "- name: " << info.name << "\n";
}

// struct Counts {
//     std::size_t total() const;
//     bool allPassed() const;
//     bool allOk() const;
//
//     std::size_t passed = 0;
//     std::size_t failed = 0;
//     std::size_t failedButOk = 0;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::Counts const& info ) {
    os << ws(level  ) << title << ":\n"
       << ws(level+1) << "- total(): "     << info.total()     << "\n"
       << ws(level+1) << "- allPassed(): " << info.allPassed() << "\n"
       << ws(level+1) << "- allOk(): "     << info.allOk()     << "\n"
       << ws(level+1) << "- passed: "      << info.passed      << "\n"
       << ws(level+1) << "- failed: "      << info.failed      << "\n"
       << ws(level+1) << "- failedButOk: " << info.failedButOk << "\n";
}

// struct Totals {
//     Counts assertions;
//     Counts testCases;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::Totals const& info ) {
    os << ws(level) << title << ":\n";
    print( os, level+1, "- assertions", info.assertions );
    print( os, level+1, "- testCases" , info.testCases  );
}

// struct TestRunStats {
//     TestRunInfo runInfo;
//     Totals totals;
//     bool aborting;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::TestRunStats const& info ) {
    os << ws(level) << title << ":\n";
    print( os, level+1 , "- runInfo", info.runInfo );
    print( os, level+1 , "- totals" , info.totals  );
    os << ws(level+1) << "- aborting: " << info.aborting << "\n";
}

//    struct Tag {
//        StringRef original, lowerCased;
//    };
//
//
//    enum class TestCaseProperties : uint8_t {
//        None = 0,
//        IsHidden = 1 << 1,
//        ShouldFail = 1 << 2,
//        MayFail = 1 << 3,
//        Throws = 1 << 4,
//        NonPortable = 1 << 5,
//        Benchmark = 1 << 6
//    };
//
//
//    struct TestCaseInfo : NonCopyable {
//
//        bool isHidden() const;
//        bool throws() const;
//        bool okToFail() const;
//        bool expectedToFail() const;
//
//
//        std::string name;
//        std::string className;
//        std::vector<Tag> tags;
//        SourceLineInfo lineInfo;
//        TestCaseProperties properties = TestCaseProperties::None;
//    };

void print( std::ostream& os, int const level, std::string const& title, Catch::TestCaseInfo const& info ) {
    os << ws(level  ) << title << ":\n"
       << ws(level+1) << "- isHidden(): "       << info.isHidden() << "\n"
       << ws(level+1) << "- throws(): "         << info.throws() << "\n"
       << ws(level+1) << "- okToFail(): "       << info.okToFail() << "\n"
       << ws(level+1) << "- expectedToFail(): " << info.expectedToFail() << "\n"
       << ws(level+1) << "- tagsAsString(): '"  << info.tagsAsString() << "'\n"
       << ws(level+1) << "- name: '"            << info.name << "'\n"
       << ws(level+1) << "- className: '"       << info.className << "'\n"
       << ws(level+1) << "- tags: "             << info.tags << "\n";
    print( os, level+1 , "- lineInfo", info.lineInfo );
    os << ws(level+1) << "- properties (flags): 0x" << std::hex << static_cast<uint32_t>(info.properties) << std::dec << "\n";
}

// struct TestCaseStats {
//     TestCaseInfo testInfo;
//     Totals totals;
//     std::string stdOut;
//     std::string stdErr;
//     bool aborting;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::TestCaseStats const& info ) {
    os << ws(level  ) << title << ":\n";
    print( os, level+1 , "- testInfo", *info.testInfo );
    print( os, level+1 , "- totals"  , info.totals   );
    os << ws(level+1) << "- stdOut: "   << info.stdOut << "\n"
       << ws(level+1) << "- stdErr: "   << info.stdErr << "\n"
       << ws(level+1) << "- aborting: " << info.aborting << "\n";
}

// struct SectionInfo {
//     std::string name;
//     std::string description;
//     SourceLineInfo lineInfo;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::SectionInfo const& info ) {
    os << ws(level  ) << title << ":\n"
       << ws(level+1) << "- name: "         << info.name << "\n";
    print( os, level+1 , "- lineInfo", info.lineInfo );
}

// struct SectionStats {
//     SectionInfo sectionInfo;
//     Counts assertions;
//     double durationInSeconds;
//     bool missingAssertions;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::SectionStats const& info ) {
    os << ws(level  ) << title << ":\n";
    print( os, level+1 , "- sectionInfo", info.sectionInfo );
    print( os, level+1 , "- assertions" , info.assertions );
    os << ws(level+1) << "- durationInSeconds: " << info.durationInSeconds << "\n"
       << ws(level+1) << "- missingAssertions: " << info.missingAssertions << "\n";
}

// struct AssertionInfo
// {
//     StringRef macroName;
//     SourceLineInfo lineInfo;
//     StringRef capturedExpression;
//     ResultDisposition::Flags resultDisposition;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::AssertionInfo const& info ) {
    os << ws(level  ) << title << ":\n"
       << ws(level+1) << "- macroName: '"  << info.macroName << "'\n";
    print( os, level+1 , "- lineInfo" , info.lineInfo );
    os << ws(level+1) << "- capturedExpression: '" << info.capturedExpression << "'\n"
       << ws(level+1) << "- resultDisposition (flags): 0x" << std::hex << info.resultDisposition  << std::dec << "\n";
}

//struct AssertionResultData
//{
//    std::string reconstructExpression() const;
//
//    std::string message;
//    mutable std::string reconstructedExpression;
//    LazyExpression lazyExpression;
//    ResultWas::OfType resultType;
//};

void print( std::ostream& os, int const level, std::string const& title, Catch::AssertionResultData const& info ) {
    os << ws(level  ) << title << ":\n"
       << ws(level+1) << "- reconstructExpression(): '" <<   info.reconstructExpression() << "'\n"
       << ws(level+1) << "- message: '"                 <<   info.message << "'\n"
       << ws(level+1) << "- lazyExpression: '"          << "(info.lazyExpression)" << "'\n"
       << ws(level+1) << "- resultType: '"              <<   info.resultType << "'\n";
}

//class AssertionResult {
//    bool isOk() const;
//    bool succeeded() const;
//    ResultWas::OfType getResultType() const;
//    bool hasExpression() const;
//    bool hasMessage() const;
//    std::string getExpression() const;
//    std::string getExpressionInMacro() const;
//    bool hasExpandedExpression() const;
//    std::string getExpandedExpression() const;
//    std::string getMessage() const;
//    SourceLineInfo getSourceInfo() const;
//    std::string getTestMacroName() const;
//
//    AssertionInfo m_info;
//    AssertionResultData m_resultData;
//};

void print( std::ostream& os, int const level, std::string const& title, Catch::AssertionResult const& info ) {
    os << ws(level  ) << title << ":\n"
       << ws(level+1) << "- isOk(): "  << info.isOk() << "\n"
       << ws(level+1) << "- succeeded(): "  << info.succeeded() << "\n"
       << ws(level+1) << "- getResultType(): "  << info.getResultType() << "\n"
       << ws(level+1) << "- hasExpression(): "  << info.hasExpression() << "\n"
       << ws(level+1) << "- hasMessage(): "  << info.hasMessage() << "\n"
       << ws(level+1) << "- getExpression(): '"  << info.getExpression() << "'\n"
       << ws(level+1) << "- getExpressionInMacro(): '"  << info.getExpressionInMacro()  << "'\n"
       << ws(level+1) << "- hasExpandedExpression(): "  << info.hasExpandedExpression() << "\n"
       << ws(level+1) << "- getExpandedExpression(): "  << info.getExpandedExpression() << "'\n"
       << ws(level+1) << "- getMessage(): '"  << info.getMessage() << "'\n";
    print( os, level+1 , "- getSourceInfo(): ", info.getSourceInfo() );
    os << ws(level+1) << "- getTestMacroName(): '"  << info.getTestMacroName() << "'\n";

    print( os, level+1 , "- *** m_info (AssertionInfo)", info.m_info );
    print( os, level+1 , "- *** m_resultData (AssertionResultData)", info.m_resultData );
}

// struct AssertionStats {
//     AssertionResult assertionResult;
//     std::vector<MessageInfo> infoMessages;
//     Totals totals;
// };

void print( std::ostream& os, int const level, std::string const& title, Catch::AssertionStats const& info ) {
    os << ws(level  ) << title << ":\n";
    print( os, level+1 , "- assertionResult", info.assertionResult );
    print( os, level+1 , "- infoMessages", info.infoMessages );
    print( os, level+1 , "- totals", info.totals );
}

// -----------------------------------------------------------------------
// 2. My listener and registration:
//

char const * const dashed_line =
    "--------------------------------------------------------------------------";


struct MyListener : Catch::EventListenerBase {

    using EventListenerBase::EventListenerBase; // inherit constructor

    // Get rid of Wweak-tables
    ~MyListener() override;

    // The whole test run starting
    void testRunStarting( Catch::TestRunInfo const& testRunInfo ) override {
        std::cout
            << std::boolalpha
            << "\nEvent: testRunStarting:\n";
        print( std::cout, 1, "- testRunInfo", testRunInfo );
    }

    // The whole test run ending
    void testRunEnded( Catch::TestRunStats const& testRunStats ) override {
        std::cout
            << dashed_line
            << "\nEvent: testRunEnded:\n";
        print( std::cout, 1, "- testRunStats", testRunStats );
    }

    // A test is being skipped (because it is "hidden")
    void skipTest( Catch::TestCaseInfo const& testInfo ) override {
        std::cout
            << dashed_line
            << "\nEvent: skipTest:\n";
        print( std::cout, 1, "- testInfo", testInfo );
    }

    // Test cases starting
    void testCaseStarting( Catch::TestCaseInfo const& testInfo ) override {
        std::cout
            << dashed_line
            << "\nEvent: testCaseStarting:\n";
        print( std::cout, 1, "- testInfo", testInfo );
    }

    // Test cases ending
    void testCaseEnded( Catch::TestCaseStats const& testCaseStats ) override {
        std::cout << "\nEvent: testCaseEnded:\n";
        print( std::cout, 1, "testCaseStats", testCaseStats );
    }

    // Sections starting
    void sectionStarting( Catch::SectionInfo const& sectionInfo ) override {
        std::cout << "\nEvent: sectionStarting:\n";
        print( std::cout, 1, "- sectionInfo", sectionInfo );
    }

    // Sections ending
    void sectionEnded( Catch::SectionStats const& sectionStats ) override {
        std::cout << "\nEvent: sectionEnded:\n";
        print( std::cout, 1, "- sectionStats", sectionStats );
    }

    // Assertions before/ after
    void assertionStarting( Catch::AssertionInfo const& assertionInfo ) override {
        std::cout << "\nEvent: assertionStarting:\n";
        print( std::cout, 1, "- assertionInfo", assertionInfo );
    }

    void assertionEnded( Catch::AssertionStats const& assertionStats ) override {
        std::cout << "\nEvent: assertionEnded:\n";
        print( std::cout, 1, "- assertionStats", assertionStats );
    }
};

} // end anonymous namespace

CATCH_REGISTER_LISTENER( MyListener )

// Get rid of Wweak-tables
MyListener::~MyListener() = default;

// -----------------------------------------------------------------------
// 3. Test cases:
//

TEST_CASE( "1: Hidden testcase", "[.hidden]" ) {
}

TEST_CASE( "2: Testcase with sections", "[tag-A][tag-B]" ) {

    int i = 42;

    REQUIRE( i == 42 );

    SECTION("Section 1") {
        INFO("Section 1");
        i = 7;
        SECTION("Section 1.1") {
            INFO("Section 1.1");
            REQUIRE( i == 42 );
        }
    }

    SECTION("Section 2") {
        INFO("Section 2");
        REQUIRE( i == 42 );
    }
    WARN("At end of test case");
}

struct Fixture {
    int fortytwo() const {
        return 42;
    }
};

TEST_CASE_METHOD( Fixture, "3: Testcase with class-based fixture", "[tag-C][tag-D]" ) {
    REQUIRE( fortytwo() == 42 );
}

// Compile & run:
// - g++ -std=c++14 -Wall -I$(CATCH_SINGLE_INCLUDE) -o 210-Evt-EventListeners 210-Evt-EventListeners.cpp && 210-Evt-EventListeners --success
// - cl -EHsc -I%CATCH_SINGLE_INCLUDE% 210-Evt-EventListeners.cpp && 210-Evt-EventListeners --success

// Expected compact output (all assertions):
//
// prompt> 210-Evt-EventListeners --reporter compact --success
// result omitted for brevity.
