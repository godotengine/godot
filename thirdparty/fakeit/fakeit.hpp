#pragma once
/*
 *  FakeIt - A Simplified C++ Mocking Framework
 *  Copyright (c) Eran Pe'er 2013
 *  Generated: 2023-04-17 21:28:50.758101
 *  Distributed under the MIT License. Please refer to the LICENSE file at:
 *  https://github.com/eranpeer/FakeIt
 */

#ifndef fakeit_h__
#define fakeit_h__



#include <functional>
#include <memory>
#include <set>
#include <vector>
#include <stdexcept>
#if defined (__GNUG__) || _MSC_VER >= 1900
#   define FAKEIT_THROWS noexcept(false)
#   define FAKEIT_NO_THROWS noexcept(true)
#elif defined (_MSC_VER)
#   define FAKEIT_THROWS throw(...)
#   define FAKEIT_NO_THROWS
#endif

#ifdef _MSVC_LANG
#   define FAKEIT_CPLUSPLUS _MSVC_LANG
#else
#   define FAKEIT_CPLUSPLUS __cplusplus
#endif

#ifdef __GNUG__
#   define FAKEIT_DISARM_UBSAN __attribute__((no_sanitize("undefined")))
#else
#   define FAKEIT_DISARM_UBSAN
#endif
#include <typeinfo>
#include <unordered_set>
#include <tuple>
#include <string>
#include <iosfwd>
#include <atomic>
#include <tuple>


namespace fakeit {

    template<class...>
    using fk_void_t = void;

    template <bool...> struct bool_pack;

    template <bool... v>
    using all_true = std::is_same<bool_pack<true, v...>, bool_pack<v..., true>>;

    template<class C>
    struct naked_type {
        typedef typename std::remove_cv<typename std::remove_reference<C>::type>::type type;
    };

    template< class T > struct tuple_arg         { typedef T  type; };
    template< class T > struct tuple_arg < T& >  { typedef T& type; };
    template< class T > struct tuple_arg < T&& > { typedef T&&  type; };




    template<typename... arglist>
    using ArgumentsTuple = std::tuple < arglist... > ;

    template< class T > struct test_arg         { typedef T& type; };
    template< class T > struct test_arg< T& >   { typedef T& type; };
    template< class T > struct test_arg< T&& >  { typedef T& type; };

    template< class T > struct production_arg         { typedef T& type; };
    template< class T > struct production_arg< T& >   { typedef T& type; };
    template< class T > struct production_arg< T&& >  { typedef T&&  type; };

    template <typename T>
    class is_ostreamable {
        struct no {};
#if defined(_MSC_VER) && _MSC_VER < 1900
        template <typename Type1>
        static decltype(operator<<(std::declval<std::ostream&>(), std::declval<const Type1>())) test(std::ostream &s, const Type1 &t);
#else
        template <typename Type1>
        static auto test(std::ostream &s, const Type1 &t) -> decltype(s << t);
#endif
        static no test(...);
    public:
        is_ostreamable() {};

        static const bool value =
            std::is_arithmetic<T>::value ||
            std::is_pointer<T>::value ||
            std::is_same<decltype(test(*(std::ostream *)nullptr,
                std::declval<T>())), std::ostream &>::value;
    };


    template <>
    class is_ostreamable<std::ios_base& (*)(std::ios_base&)> {
    public:
        static const bool value = true;
    };

    template <typename CharT, typename Traits>
    class is_ostreamable<std::basic_ios<CharT,Traits>& (*)(std::basic_ios<CharT,Traits>&)> {
    public:
        static const bool value = true;
    };

    template <typename CharT, typename Traits>
    class is_ostreamable<std::basic_ostream<CharT,Traits>& (*)(std::basic_ostream<CharT,Traits>&)> {
    public:
        static const bool value = true;
    };

    template<typename R, typename... arglist>
    struct VTableMethodType {
#if defined (__GNUG__)
        typedef R(*type)(void *, arglist...);
#elif defined (_MSC_VER)
        typedef R(__thiscall *type)(void *, arglist...);
#endif
    };

    template<template<typename>class test, typename T>
    struct smart_test : test<T> {};

    template<template<typename>class test, typename T, typename A>
    struct smart_test <test, std::vector<T, A>> : smart_test < test, T> {};

    template<typename T>
    using smart_is_copy_constructible = smart_test < std::is_copy_constructible, T >;
}
#include <typeinfo>
#include <tuple>
#include <string>
#include <iosfwd>
#include <sstream>
#include <string>

namespace fakeit {

    struct FakeitContext;

    template<typename C>
    struct MockObject {
        virtual ~MockObject() FAKEIT_THROWS { };

        virtual C &get() = 0;

        virtual FakeitContext &getFakeIt() = 0;
    };

    struct MethodInfo {

        static unsigned int nextMethodOrdinal() {
            static std::atomic_uint ordinal{0};
            return ++ordinal;
        }

        MethodInfo(unsigned int anId, std::string aName) :
                _id(anId), _name(aName) { }

        unsigned int id() const {
            return _id;
        }

        std::string name() const {
            return _name;
        }

        void setName(const std::string &value) {
            _name = value;
        }

    private:
        unsigned int _id;
        std::string _name;
    };

    struct UnknownMethod {

        static MethodInfo &instance() {
            static MethodInfo instance(MethodInfo::nextMethodOrdinal(), "unknown");
            return instance;
        }

    };

}
namespace fakeit {
    class Destructible {
    public:
        virtual ~Destructible() {}
    };
}

namespace fakeit {

    struct Invocation : Destructible {

        static unsigned int nextInvocationOrdinal() {
            static std::atomic_uint invocationOrdinal{0};
            return ++invocationOrdinal;
        }

        struct Matcher {

            virtual ~Matcher() FAKEIT_THROWS {
            }

            virtual bool matches(Invocation &invocation) = 0;

            virtual std::string format() const = 0;
        };

        Invocation(unsigned int ordinal, MethodInfo &method) :
                _ordinal(ordinal), _method(method), _isVerified(false) {
        }

        virtual ~Invocation() override = default;

        unsigned int getOrdinal() const {
            return _ordinal;
        }

        MethodInfo &getMethod() const {
            return _method;
        }

        void markAsVerified() {
            _isVerified = true;
        }

        bool isVerified() const {
            return _isVerified;
        }

        virtual std::string format() const = 0;

    private:
        const unsigned int _ordinal;
        MethodInfo &_method;
        bool _isVerified;
    };

}
#include <iosfwd>
#include <tuple>
#include <string>
#include <sstream>
#include <ostream>

namespace fakeit {

	template<typename T, class Enable = void>
	struct Formatter;

	template <>
	struct Formatter<bool>
	{
		static std::string format(bool const &val)
		{
			return val ? "true" : "false";
		}
	};

	template <>
	struct Formatter<char>
	{
		static std::string format(char const &val)
		{
			std::string s;
			s += "'";
			s += val;
			s += "'";
			return s;
		}
	};

	template <>
	struct Formatter<char const*>
	{
		static std::string format(char const* const &val)
		{
			std::string s;
			if(val != nullptr)
			{
				s += '"';
				s += val;
				s += '"';
			}
			else
			{
				s = "[nullptr]";
			}
			return s;
		}
	};

	template <>
	struct Formatter<char*>
	{
		static std::string format(char* const &val)
		{
			return Formatter<char const*>::format( val );
		}
	};

	template<class C>
	struct Formatter<C, typename std::enable_if<!is_ostreamable<C>::value>::type> {
		static std::string format(C const &)
		{
			return "?";
		}
	};

	template<class C>
	struct Formatter<C, typename std::enable_if<is_ostreamable<C>::value>::type> {
		static std::string format(C const &val)
		{
			std::ostringstream os;
			os << val;
			return os.str();
		}
	};


	template <typename T>
	using TypeFormatter = Formatter<typename fakeit::naked_type<T>::type>;
}

namespace fakeit {


    template<class Tuple, std::size_t N>
    struct TuplePrinter {
        static void print(std::ostream &strm, const Tuple &t) {
            TuplePrinter<Tuple, N - 1>::print(strm, t);
            strm << ", " << fakeit::TypeFormatter<decltype(std::get<N - 1>(t))>::format(std::get<N - 1>(t));
        }
    };

    template<class Tuple>
    struct TuplePrinter<Tuple, 1> {
        static void print(std::ostream &strm, const Tuple &t) {
            strm << fakeit::TypeFormatter<decltype(std::get<0>(t))>::format(std::get<0>(t));
        }
    };

    template<class Tuple>
    struct TuplePrinter<Tuple, 0> {
        static void print(std::ostream &, const Tuple &) {
        }
    };

    template<class ... Args>
    void print(std::ostream &strm, const std::tuple<Args...> &t) {
        strm << "(";
        TuplePrinter<decltype(t), sizeof...(Args)>::print(strm, t);
        strm << ")";
    }

    template<class ... Args>
    std::ostream &operator<<(std::ostream &strm, const std::tuple<Args...> &t) {
        print(strm, t);
        return strm;
    }

}


namespace fakeit {

    template<typename ... arglist>
    struct ActualInvocation : public Invocation {

        struct Matcher : public virtual Destructible {
            virtual bool matches(ActualInvocation<arglist...> &actualInvocation) = 0;

            virtual std::string format() const = 0;
        };

        ActualInvocation(unsigned int ordinal, MethodInfo &method, const typename fakeit::production_arg<arglist>::type... args) :
            Invocation(ordinal, method), _matcher{ nullptr }
            , actualArguments{ std::forward<arglist>(args)... }
        {
        }

        ArgumentsTuple<arglist...> & getActualArguments() {
            return actualArguments;
        }


        void setActualMatcher(Matcher *matcher) {
            this->_matcher = matcher;
        }

        Matcher *getActualMatcher() {
            return _matcher;
        }

        virtual std::string format() const override {
            std::ostringstream out;
            out << getMethod().name();
            print(out, actualArguments);
            return out.str();
        }

    private:

        Matcher *_matcher;
        ArgumentsTuple<arglist...> actualArguments;
    };

    template<typename ... arglist>
    std::ostream &operator<<(std::ostream &strm, const ActualInvocation<arglist...> &ai) {
        strm << ai.format();
        return strm;
    }

}





#include <unordered_set>

namespace fakeit {

	struct ActualInvocationsContainer {
		virtual void clear() = 0;

		virtual ~ActualInvocationsContainer() FAKEIT_NO_THROWS { }
	};

    struct ActualInvocationsSource {
        virtual void getActualInvocations(std::unordered_set<fakeit::Invocation *> &into) const = 0;

        virtual ~ActualInvocationsSource() FAKEIT_NO_THROWS { }
    };

    struct InvocationsSourceProxy : public ActualInvocationsSource {

        InvocationsSourceProxy(ActualInvocationsSource *inner) :
                _inner(inner) {
        }

        void getActualInvocations(std::unordered_set<fakeit::Invocation *> &into) const override {
            _inner->getActualInvocations(into);
        }

    private:
        std::shared_ptr<ActualInvocationsSource> _inner;
    };

    struct UnverifiedInvocationsSource : public ActualInvocationsSource {

        UnverifiedInvocationsSource(InvocationsSourceProxy decorated) : _decorated(decorated) {
        }

        void getActualInvocations(std::unordered_set<fakeit::Invocation *> &into) const override {
            std::unordered_set<fakeit::Invocation *> all;
            _decorated.getActualInvocations(all);
            for (fakeit::Invocation *i : all) {
                if (!i->isVerified()) {
                    into.insert(i);
                }
            }
        }

    private:
        InvocationsSourceProxy _decorated;
    };

    struct AggregateInvocationsSource : public ActualInvocationsSource {

        AggregateInvocationsSource(std::vector<ActualInvocationsSource *> &sources) : _sources(sources) {
        }

        void getActualInvocations(std::unordered_set<fakeit::Invocation *> &into) const override {
            std::unordered_set<fakeit::Invocation *> tmp;
            for (ActualInvocationsSource *source : _sources) {
                source->getActualInvocations(tmp);
            }
            filter(tmp, into);
        }

    protected:
        bool shouldInclude(fakeit::Invocation *) const {
            return true;
        }

    private:
        std::vector<ActualInvocationsSource *> _sources;

        void filter(std::unordered_set<Invocation *> &source, std::unordered_set<Invocation *> &target) const {
            for (Invocation *i:source) {
                if (shouldInclude(i)) {
                    target.insert(i);
                }
            }
        }
    };
}

namespace fakeit {

    class Sequence {
    private:

    protected:

        Sequence() {
        }

        virtual ~Sequence() FAKEIT_THROWS {
        }

    public:


        virtual void getExpectedSequence(std::vector<Invocation::Matcher *> &into) const = 0;


        virtual void getInvolvedMocks(std::vector<ActualInvocationsSource *> &into) const = 0;

        virtual unsigned int size() const = 0;

        friend class VerifyFunctor;
    };

    class ConcatenatedSequence : public virtual Sequence {
    private:
        const Sequence &s1;
        const Sequence &s2;

    protected:
        ConcatenatedSequence(const Sequence &seq1, const Sequence &seq2) :
                s1(seq1), s2(seq2) {
        }

    public:

        virtual ~ConcatenatedSequence() {
        }

        unsigned int size() const override {
            return s1.size() + s2.size();
        }

        const Sequence &getLeft() const {
            return s1;
        }

        const Sequence &getRight() const {
            return s2;
        }

        void getExpectedSequence(std::vector<Invocation::Matcher *> &into) const override {
            s1.getExpectedSequence(into);
            s2.getExpectedSequence(into);
        }

        virtual void getInvolvedMocks(std::vector<ActualInvocationsSource *> &into) const override {
            s1.getInvolvedMocks(into);
            s2.getInvolvedMocks(into);
        }

        friend inline ConcatenatedSequence operator+(const Sequence &s1, const Sequence &s2);
    };

    class RepeatedSequence : public virtual Sequence {
    private:
        const Sequence &_s;
        const int times;

    protected:
        RepeatedSequence(const Sequence &s, const int t) :
                _s(s), times(t) {
        }

    public:

        ~RepeatedSequence() {
        }

        unsigned int size() const override {
            return _s.size() * times;
        }

        friend inline RepeatedSequence operator*(const Sequence &s, int times);

        friend inline RepeatedSequence operator*(int times, const Sequence &s);

        void getInvolvedMocks(std::vector<ActualInvocationsSource *> &into) const override {
            _s.getInvolvedMocks(into);
        }

        void getExpectedSequence(std::vector<Invocation::Matcher *> &into) const override {
            for (int i = 0; i < times; i++)
                _s.getExpectedSequence(into);
        }

        int getTimes() const {
            return times;
        }

        const Sequence &getSequence() const {
            return _s;
        }
    };

    inline ConcatenatedSequence operator+(const Sequence &s1, const Sequence &s2) {
        return ConcatenatedSequence(s1, s2);
    }

    inline RepeatedSequence operator*(const Sequence &s, int times) {
        if (times <= 0)
            FAIL("times");
        return RepeatedSequence(s, times);
    }

    inline RepeatedSequence operator*(int times, const Sequence &s) {
        if (times <= 0)
            FAIL("times");
        return RepeatedSequence(s, times);
    }

}

namespace fakeit {

    enum class VerificationType {
        Exact, AtLeast, NoMoreInvocations
    };

    enum class UnexpectedType {
        Unmocked, Unmatched
    };

    struct VerificationEvent {

        VerificationEvent(VerificationType aVerificationType) :
                _verificationType(aVerificationType), _line(0) {
        }

        virtual ~VerificationEvent() = default;

        VerificationType verificationType() const {
            return _verificationType;
        }

        void setFileInfo(const char * aFile, int aLine, const char * aCallingMethod) {
            _file = aFile;
            _callingMethod = aCallingMethod;
            _line = aLine;
        }

        const char * file() const {
            return _file;
        }

        int line() const {
            return _line;
        }

        const char * callingMethod() const {
            return _callingMethod;
        }

    private:
        VerificationType _verificationType;
		const char * _file;
        int _line;
        const char * _callingMethod;
    };

    struct NoMoreInvocationsVerificationEvent : public VerificationEvent {

        ~NoMoreInvocationsVerificationEvent() = default;

        NoMoreInvocationsVerificationEvent(
                std::vector<Invocation *> &allTheIvocations,
                std::vector<Invocation *> &anUnverifedIvocations) :
                VerificationEvent(VerificationType::NoMoreInvocations),
                _allIvocations(allTheIvocations),
                _unverifedIvocations(anUnverifedIvocations) {
        }

        const std::vector<Invocation *> &allIvocations() const {
            return _allIvocations;
        }

        const std::vector<Invocation *> &unverifedIvocations() const {
            return _unverifedIvocations;
        }

    private:
        const std::vector<Invocation *> _allIvocations;
        const std::vector<Invocation *> _unverifedIvocations;
    };

    struct SequenceVerificationEvent : public VerificationEvent {

        ~SequenceVerificationEvent() = default;

        SequenceVerificationEvent(VerificationType aVerificationType,
                                  std::vector<Sequence *> &anExpectedPattern,
                                  std::vector<Invocation *> &anActualSequence,
                                  int anExpectedCount,
                                  int anActualCount) :
                VerificationEvent(aVerificationType),
                _expectedPattern(anExpectedPattern),
                _actualSequence(anActualSequence),
                _expectedCount(anExpectedCount),
                _actualCount(anActualCount)
        {
        }

        const std::vector<Sequence *> &expectedPattern() const {
            return _expectedPattern;
        }

        const std::vector<Invocation *> &actualSequence() const {
            return _actualSequence;
        }

        int expectedCount() const {
            return _expectedCount;
        }

        int actualCount() const {
            return _actualCount;
        }

    private:
        const std::vector<Sequence *> _expectedPattern;
        const std::vector<Invocation *> _actualSequence;
        const int _expectedCount;
        const int _actualCount;
    };

    struct UnexpectedMethodCallEvent {
        UnexpectedMethodCallEvent(UnexpectedType unexpectedType, const Invocation &invocation) :
                _unexpectedType(unexpectedType), _invocation(invocation) {
        }

        const Invocation &getInvocation() const {
            return _invocation;
        }

        UnexpectedType getUnexpectedType() const {
            return _unexpectedType;
        }

        const UnexpectedType _unexpectedType;
        const Invocation &_invocation;
    };

}

namespace fakeit {

    struct VerificationEventHandler {
        virtual void handle(const SequenceVerificationEvent &e) = 0;

        virtual void handle(const NoMoreInvocationsVerificationEvent &e) = 0;

        virtual ~VerificationEventHandler() { }
    };

    struct EventHandler : public VerificationEventHandler {
        using VerificationEventHandler::handle;

        virtual void handle(const UnexpectedMethodCallEvent &e) = 0;
    };

}
#include <vector>
#include <string>

namespace fakeit {

    struct UnexpectedMethodCallEvent;
    struct SequenceVerificationEvent;
    struct NoMoreInvocationsVerificationEvent;

    struct EventFormatter {

        virtual std::string format(const fakeit::UnexpectedMethodCallEvent &e) = 0;

        virtual std::string format(const fakeit::SequenceVerificationEvent &e) = 0;

        virtual std::string format(const fakeit::NoMoreInvocationsVerificationEvent &e) = 0;

        virtual ~EventFormatter() { }
    };

}
#ifdef FAKEIT_ASSERT_ON_UNEXPECTED_METHOD_INVOCATION
#include <cassert>
#endif

namespace fakeit {

    struct FakeitContext : public EventHandler, protected EventFormatter {

        virtual ~FakeitContext() = default;

        void handle(const UnexpectedMethodCallEvent &e) override {
            fireEvent(e);
            auto &eh = getTestingFrameworkAdapter();
            #ifdef FAKEIT_ASSERT_ON_UNEXPECTED_METHOD_INVOCATION
            assert(!"Unexpected method invocation");
            #endif
            eh.handle(e);
        }

        void handle(const SequenceVerificationEvent &e) override {
            fireEvent(e);
            auto &eh = getTestingFrameworkAdapter();
            return eh.handle(e);
        }

        void handle(const NoMoreInvocationsVerificationEvent &e) override {
            fireEvent(e);
            auto &eh = getTestingFrameworkAdapter();
            return eh.handle(e);
        }

        std::string format(const UnexpectedMethodCallEvent &e) override {
            auto &eventFormatter = getEventFormatter();
            return eventFormatter.format(e);
        }

        std::string format(const SequenceVerificationEvent &e) override {
            auto &eventFormatter = getEventFormatter();
            return eventFormatter.format(e);
        }

        std::string format(const NoMoreInvocationsVerificationEvent &e) override {
            auto &eventFormatter = getEventFormatter();
            return eventFormatter.format(e);
        }

        void addEventHandler(EventHandler &eventListener) {
            _eventListeners.push_back(&eventListener);
        }

        void clearEventHandlers() {
            _eventListeners.clear();
        }

    protected:
        virtual EventHandler &getTestingFrameworkAdapter() = 0;

        virtual EventFormatter &getEventFormatter() = 0;

    private:
        std::vector<EventHandler *> _eventListeners;

        void fireEvent(const NoMoreInvocationsVerificationEvent &evt) {
            for (auto listener : _eventListeners)
                listener->handle(evt);
        }

        void fireEvent(const UnexpectedMethodCallEvent &evt) {
            for (auto listener : _eventListeners)
                listener->handle(evt);
        }

        void fireEvent(const SequenceVerificationEvent &evt) {
            for (auto listener : _eventListeners)
                listener->handle(evt);
        }

    };

}
#include <iostream>
#include <iosfwd>

namespace fakeit {

    struct DefaultEventFormatter : public EventFormatter {

        virtual std::string format(const UnexpectedMethodCallEvent &e) override {
            std::ostringstream out;
            out << "Unexpected method invocation: ";
            out << e.getInvocation().format() << std::endl;
            if (UnexpectedType::Unmatched == e.getUnexpectedType()) {
                out << "  Could not find any recorded behavior to support this method call.";
            } else {
                out << "  An unmocked method was invoked. All used virtual methods must be stubbed!";
            }
            return out.str();
        }


        virtual std::string format(const SequenceVerificationEvent &e) override {
            std::ostringstream out;
            out << "Verification error" << std::endl;

            out << "Expected pattern: ";
            const std::vector<fakeit::Sequence *> expectedPattern = e.expectedPattern();
            out << formatExpectedPattern(expectedPattern) << std::endl;

            out << "Expected matches: ";
            formatExpectedCount(out, e.verificationType(), e.expectedCount());
            out << std::endl;

            out << "Actual matches  : " << e.actualCount() << std::endl;

            auto actualSequence = e.actualSequence();
            out << "Actual sequence : total of " << actualSequence.size() << " actual invocations";
            if (actualSequence.size() == 0) {
                out << ".";
            } else {
                out << ":" << std::endl;
            }
            formatInvocationList(out, actualSequence);

            return out.str();
        }

        virtual std::string format(const NoMoreInvocationsVerificationEvent &e) override {
            std::ostringstream out;
            out << "Verification error" << std::endl;
            out << "Expected no more invocations!! but the following unverified invocations were found:" << std::endl;
            formatInvocationList(out, e.unverifedIvocations());
            return out.str();
        }

        static std::string formatExpectedPattern(const std::vector<fakeit::Sequence *> &expectedPattern) {
            std::string expectedPatternStr;
            for (unsigned int i = 0; i < expectedPattern.size(); i++) {
                Sequence *s = expectedPattern[i];
                expectedPatternStr += formatSequence(*s);
                if (i < expectedPattern.size() - 1)
                    expectedPatternStr += " ... ";
            }
            return expectedPatternStr;
        }

    private:

        static std::string formatSequence(const Sequence &val) {
            const ConcatenatedSequence *cs = dynamic_cast<const ConcatenatedSequence *>(&val);
            if (cs) {
                return format(*cs);
            }
            const RepeatedSequence *rs = dynamic_cast<const RepeatedSequence *>(&val);
            if (rs) {
                return format(*rs);
            }


            std::vector<Invocation::Matcher *> vec;
            val.getExpectedSequence(vec);
            return vec[0]->format();
        }

        static void formatExpectedCount(std::ostream &out, fakeit::VerificationType verificationType,
                                        int expectedCount) {
            if (verificationType == fakeit::VerificationType::Exact)
                out << "exactly ";

            if (verificationType == fakeit::VerificationType::AtLeast)
                out << "at least ";

            out << expectedCount;
        }

        static void formatInvocationList(std::ostream &out, const std::vector<fakeit::Invocation *> &actualSequence) {
            size_t max_size = actualSequence.size();
            if (max_size > 50)
                max_size = 50;

            for (unsigned int i = 0; i < max_size; i++) {
                out << "  ";
                auto invocation = actualSequence[i];
                out << invocation->format();
                if (i < max_size - 1)
                    out << std::endl;
            }

            if (actualSequence.size() > max_size)
                out << std::endl << "  ...";
        }

        static std::string format(const ConcatenatedSequence &val) {
            std::ostringstream out;
            out << formatSequence(val.getLeft()) << " + " << formatSequence(val.getRight());
            return out.str();
        }

        static std::string format(const RepeatedSequence &val) {
            std::ostringstream out;
            const ConcatenatedSequence *cs = dynamic_cast<const ConcatenatedSequence *>(&val.getSequence());
            const RepeatedSequence *rs = dynamic_cast<const RepeatedSequence *>(&val.getSequence());
            if (rs || cs)
                out << '(';
            out << formatSequence(val.getSequence());
            if (rs || cs)
                out << ')';

            out << " * " << val.getTimes();
            return out.str();
        }
    };
}
#include <exception>



namespace fakeit {
#if FAKEIT_CPLUSPLUS >= 201703L || defined(__cpp_lib_uncaught_exceptions)
    inline bool UncaughtException () {
        return std::uncaught_exceptions() >= 1;
    }
#else
    inline bool UncaughtException () {
      return std::uncaught_exception();
    }
#endif

    struct FakeitException {
        std::exception err;

        virtual ~FakeitException() = default;

        virtual std::string what() const = 0;

        friend std::ostream &operator<<(std::ostream &os, const FakeitException &val) {
            os << val.what();
            return os;
        }
    };

/* clang-format off */
    struct UnexpectedMethodCallException : public FakeitException {

    UnexpectedMethodCallException(std::string format) :
            _format(format) {
    }

    virtual std::string what() const override {
        return _format;
    }

    private:
        std::string _format;
    };
/* clang-format on */
}

namespace fakeit {

    struct DefaultEventLogger : public fakeit::EventHandler {

        DefaultEventLogger(EventFormatter &formatter) : _formatter(formatter), _out(std::cout) { }

        virtual void handle(const UnexpectedMethodCallEvent &e) override {
            _out << _formatter.format(e) << std::endl;
        }

        virtual void handle(const SequenceVerificationEvent &e) override {
            _out << _formatter.format(e) << std::endl;
        }

        virtual void handle(const NoMoreInvocationsVerificationEvent &e) override {
            _out << _formatter.format(e) << std::endl;
        }

    private:
        EventFormatter &_formatter;
        std::ostream &_out;
    };

}

namespace fakeit {

    class AbstractFakeit : public FakeitContext {
    public:
        virtual ~AbstractFakeit() = default;

    protected:

        virtual fakeit::EventHandler &accessTestingFrameworkAdapter() = 0;

        virtual EventFormatter &accessEventFormatter() = 0;
    };

    class DefaultFakeit : public AbstractFakeit {
        DefaultEventFormatter _formatter;
        fakeit::EventFormatter *_customFormatter;
        fakeit::EventHandler *_testingFrameworkAdapter;

    public:

        DefaultFakeit() : _formatter(),
                          _customFormatter(nullptr),
                          _testingFrameworkAdapter(nullptr) {
        }

        virtual ~DefaultFakeit() = default;

        void setCustomEventFormatter(fakeit::EventFormatter &customEventFormatter) {
            _customFormatter = &customEventFormatter;
        }

        void resetCustomEventFormatter() {
            _customFormatter = nullptr;
        }

        void setTestingFrameworkAdapter(fakeit::EventHandler &testingFrameforkAdapter) {
            _testingFrameworkAdapter = &testingFrameforkAdapter;
        }

        void resetTestingFrameworkAdapter() {
            _testingFrameworkAdapter = nullptr;
        }

    protected:

        fakeit::EventHandler &getTestingFrameworkAdapter() override {
            if (_testingFrameworkAdapter)
                return *_testingFrameworkAdapter;
            return accessTestingFrameworkAdapter();
        }

        EventFormatter &getEventFormatter() override {
            if (_customFormatter)
                return *_customFormatter;
            return accessEventFormatter();
        }

        EventFormatter &accessEventFormatter() override {
            return _formatter;
        }

    };
}
#include <string>
#include <sstream>
#include <iomanip>

namespace fakeit {

    template<typename T>
    static std::string to_string(const T &n) {
        std::ostringstream stm;
        stm << n;
        return stm.str();
    }

}
#include "thirdparty/doctest/doctest.h"

namespace fakeit
{
    class DoctestAdapter : public EventHandler
    {
        EventFormatter &_formatter;

    public:
        virtual ~DoctestAdapter() = default;

        DoctestAdapter(EventFormatter &formatter)
                : _formatter(formatter) {}

        void fail(const char* fileName,
                  int lineNumber,
                  std::string fomattedMessage,
                  bool fatalFailure)
        {
            if (fatalFailure)
            {
                DOCTEST_ADD_FAIL_AT(fileName, lineNumber, fomattedMessage);
            }
            else
            {
                DOCTEST_ADD_FAIL_CHECK_AT(fileName, lineNumber, fomattedMessage);
            }
        }

        void handle(const UnexpectedMethodCallEvent &evt) override
        {
            fail("Unknown file", 0, _formatter.format(evt), true);
        }

        void handle(const SequenceVerificationEvent &evt) override
        {
            fail(evt.file(), evt.line(), _formatter.format(evt), false);
        }

        void handle(const NoMoreInvocationsVerificationEvent &evt) override
        {
            fail(evt.file(), evt.line(), _formatter.format(evt), false);
        }
    };

    class DoctestFakeit : public DefaultFakeit
    {
    public:
        virtual ~DoctestFakeit() = default;

        DoctestFakeit() : _doctestAdapter(*this) {}

        static DoctestFakeit &getInstance()
        {
            static DoctestFakeit instance;
            return instance;
        }

    protected:
        fakeit::EventHandler &accessTestingFrameworkAdapter() override
        {
            return _doctestAdapter;
        }

    private:
        DoctestAdapter _doctestAdapter;
    };
}

static fakeit::DefaultFakeit& Fakeit = fakeit::DoctestFakeit::getInstance();


#include <type_traits>
#include <unordered_set>

#include <memory>
#undef max
#include <functional>
#include <type_traits>
#include <vector>
#include <array>
#include <new>
#include <limits>

#include <functional>
#include <type_traits>
namespace fakeit {

    struct VirtualOffsetSelector {

        unsigned int offset;

        virtual unsigned int offset0(int) {
            return offset = 0;
        }

        virtual unsigned int offset1(int) {
            return offset = 1;
        }

        virtual unsigned int offset2(int) {
            return offset = 2;
        }

        virtual unsigned int offset3(int) {
            return offset = 3;
        }

        virtual unsigned int offset4(int) {
            return offset = 4;
        }

        virtual unsigned int offset5(int) {
            return offset = 5;
        }

        virtual unsigned int offset6(int) {
            return offset = 6;
        }

        virtual unsigned int offset7(int) {
            return offset = 7;
        }

        virtual unsigned int offset8(int) {
            return offset = 8;
        }

        virtual unsigned int offset9(int) {
            return offset = 9;
        }

        virtual unsigned int offset10(int) {
            return offset = 10;
        }

        virtual unsigned int offset11(int) {
            return offset = 11;
        }

        virtual unsigned int offset12(int) {
            return offset = 12;
        }

        virtual unsigned int offset13(int) {
            return offset = 13;
        }

        virtual unsigned int offset14(int) {
            return offset = 14;
        }

        virtual unsigned int offset15(int) {
            return offset = 15;
        }

        virtual unsigned int offset16(int) {
            return offset = 16;
        }

        virtual unsigned int offset17(int) {
            return offset = 17;
        }

        virtual unsigned int offset18(int) {
            return offset = 18;
        }

        virtual unsigned int offset19(int) {
            return offset = 19;
        }

        virtual unsigned int offset20(int) {
            return offset = 20;
        }

        virtual unsigned int offset21(int) {
            return offset = 21;
        }

        virtual unsigned int offset22(int) {
            return offset = 22;
        }

        virtual unsigned int offset23(int) {
            return offset = 23;
        }

        virtual unsigned int offset24(int) {
            return offset = 24;
        }

        virtual unsigned int offset25(int) {
            return offset = 25;
        }

        virtual unsigned int offset26(int) {
            return offset = 26;
        }

        virtual unsigned int offset27(int) {
            return offset = 27;
        }

        virtual unsigned int offset28(int) {
            return offset = 28;
        }

        virtual unsigned int offset29(int) {
            return offset = 29;
        }

        virtual unsigned int offset30(int) {
            return offset = 30;
        }

        virtual unsigned int offset31(int) {
            return offset = 31;
        }

        virtual unsigned int offset32(int) {
            return offset = 32;
        }

        virtual unsigned int offset33(int) {
            return offset = 33;
        }

        virtual unsigned int offset34(int) {
            return offset = 34;
        }

        virtual unsigned int offset35(int) {
            return offset = 35;
        }

        virtual unsigned int offset36(int) {
            return offset = 36;
        }

        virtual unsigned int offset37(int) {
            return offset = 37;
        }

        virtual unsigned int offset38(int) {
            return offset = 38;
        }

        virtual unsigned int offset39(int) {
            return offset = 39;
        }

        virtual unsigned int offset40(int) {
            return offset = 40;
        }

        virtual unsigned int offset41(int) {
            return offset = 41;
        }

        virtual unsigned int offset42(int) {
            return offset = 42;
        }

        virtual unsigned int offset43(int) {
            return offset = 43;
        }

        virtual unsigned int offset44(int) {
            return offset = 44;
        }

        virtual unsigned int offset45(int) {
            return offset = 45;
        }

        virtual unsigned int offset46(int) {
            return offset = 46;
        }

        virtual unsigned int offset47(int) {
            return offset = 47;
        }

        virtual unsigned int offset48(int) {
            return offset = 48;
        }

        virtual unsigned int offset49(int) {
            return offset = 49;
        }

        virtual unsigned int offset50(int) {
            return offset = 50;
        }

        virtual unsigned int offset51(int) {
            return offset = 51;
        }

        virtual unsigned int offset52(int) {
            return offset = 52;
        }

        virtual unsigned int offset53(int) {
            return offset = 53;
        }

        virtual unsigned int offset54(int) {
            return offset = 54;
        }

        virtual unsigned int offset55(int) {
            return offset = 55;
        }

        virtual unsigned int offset56(int) {
            return offset = 56;
        }

        virtual unsigned int offset57(int) {
            return offset = 57;
        }

        virtual unsigned int offset58(int) {
            return offset = 58;
        }

        virtual unsigned int offset59(int) {
            return offset = 59;
        }

        virtual unsigned int offset60(int) {
            return offset = 60;
        }

        virtual unsigned int offset61(int) {
            return offset = 61;
        }

        virtual unsigned int offset62(int) {
            return offset = 62;
        }

        virtual unsigned int offset63(int) {
            return offset = 63;
        }

        virtual unsigned int offset64(int) {
            return offset = 64;
        }

        virtual unsigned int offset65(int) {
            return offset = 65;
        }

        virtual unsigned int offset66(int) {
            return offset = 66;
        }

        virtual unsigned int offset67(int) {
            return offset = 67;
        }

        virtual unsigned int offset68(int) {
            return offset = 68;
        }

        virtual unsigned int offset69(int) {
            return offset = 69;
        }

        virtual unsigned int offset70(int) {
            return offset = 70;
        }

        virtual unsigned int offset71(int) {
            return offset = 71;
        }

        virtual unsigned int offset72(int) {
            return offset = 72;
        }

        virtual unsigned int offset73(int) {
            return offset = 73;
        }

        virtual unsigned int offset74(int) {
            return offset = 74;
        }

        virtual unsigned int offset75(int) {
            return offset = 75;
        }

        virtual unsigned int offset76(int) {
            return offset = 76;
        }

        virtual unsigned int offset77(int) {
            return offset = 77;
        }

        virtual unsigned int offset78(int) {
            return offset = 78;
        }

        virtual unsigned int offset79(int) {
            return offset = 79;
        }

        virtual unsigned int offset80(int) {
            return offset = 80;
        }

        virtual unsigned int offset81(int) {
            return offset = 81;
        }

        virtual unsigned int offset82(int) {
            return offset = 82;
        }

        virtual unsigned int offset83(int) {
            return offset = 83;
        }

        virtual unsigned int offset84(int) {
            return offset = 84;
        }

        virtual unsigned int offset85(int) {
            return offset = 85;
        }

        virtual unsigned int offset86(int) {
            return offset = 86;
        }

        virtual unsigned int offset87(int) {
            return offset = 87;
        }

        virtual unsigned int offset88(int) {
            return offset = 88;
        }

        virtual unsigned int offset89(int) {
            return offset = 89;
        }

        virtual unsigned int offset90(int) {
            return offset = 90;
        }

        virtual unsigned int offset91(int) {
            return offset = 91;
        }

        virtual unsigned int offset92(int) {
            return offset = 92;
        }

        virtual unsigned int offset93(int) {
            return offset = 93;
        }

        virtual unsigned int offset94(int) {
            return offset = 94;
        }

        virtual unsigned int offset95(int) {
            return offset = 95;
        }

        virtual unsigned int offset96(int) {
            return offset = 96;
        }

        virtual unsigned int offset97(int) {
            return offset = 97;
        }

        virtual unsigned int offset98(int) {
            return offset = 98;
        }

        virtual unsigned int offset99(int) {
            return offset = 99;
        }

        virtual unsigned int offset100(int) {
            return offset = 100;
        }

        virtual unsigned int offset101(int) {
            return offset = 101;
        }

        virtual unsigned int offset102(int) {
            return offset = 102;
        }

        virtual unsigned int offset103(int) {
            return offset = 103;
        }

        virtual unsigned int offset104(int) {
            return offset = 104;
        }

        virtual unsigned int offset105(int) {
            return offset = 105;
        }

        virtual unsigned int offset106(int) {
            return offset = 106;
        }

        virtual unsigned int offset107(int) {
            return offset = 107;
        }

        virtual unsigned int offset108(int) {
            return offset = 108;
        }

        virtual unsigned int offset109(int) {
            return offset = 109;
        }

        virtual unsigned int offset110(int) {
            return offset = 110;
        }

        virtual unsigned int offset111(int) {
            return offset = 111;
        }

        virtual unsigned int offset112(int) {
            return offset = 112;
        }

        virtual unsigned int offset113(int) {
            return offset = 113;
        }

        virtual unsigned int offset114(int) {
            return offset = 114;
        }

        virtual unsigned int offset115(int) {
            return offset = 115;
        }

        virtual unsigned int offset116(int) {
            return offset = 116;
        }

        virtual unsigned int offset117(int) {
            return offset = 117;
        }

        virtual unsigned int offset118(int) {
            return offset = 118;
        }

        virtual unsigned int offset119(int) {
            return offset = 119;
        }

        virtual unsigned int offset120(int) {
            return offset = 120;
        }

        virtual unsigned int offset121(int) {
            return offset = 121;
        }

        virtual unsigned int offset122(int) {
            return offset = 122;
        }

        virtual unsigned int offset123(int) {
            return offset = 123;
        }

        virtual unsigned int offset124(int) {
            return offset = 124;
        }

        virtual unsigned int offset125(int) {
            return offset = 125;
        }

        virtual unsigned int offset126(int) {
            return offset = 126;
        }

        virtual unsigned int offset127(int) {
            return offset = 127;
        }

        virtual unsigned int offset128(int) {
            return offset = 128;
        }

        virtual unsigned int offset129(int) {
            return offset = 129;
        }

        virtual unsigned int offset130(int) {
            return offset = 130;
        }

        virtual unsigned int offset131(int) {
            return offset = 131;
        }

        virtual unsigned int offset132(int) {
            return offset = 132;
        }

        virtual unsigned int offset133(int) {
            return offset = 133;
        }

        virtual unsigned int offset134(int) {
            return offset = 134;
        }

        virtual unsigned int offset135(int) {
            return offset = 135;
        }

        virtual unsigned int offset136(int) {
            return offset = 136;
        }

        virtual unsigned int offset137(int) {
            return offset = 137;
        }

        virtual unsigned int offset138(int) {
            return offset = 138;
        }

        virtual unsigned int offset139(int) {
            return offset = 139;
        }

        virtual unsigned int offset140(int) {
            return offset = 140;
        }

        virtual unsigned int offset141(int) {
            return offset = 141;
        }

        virtual unsigned int offset142(int) {
            return offset = 142;
        }

        virtual unsigned int offset143(int) {
            return offset = 143;
        }

        virtual unsigned int offset144(int) {
            return offset = 144;
        }

        virtual unsigned int offset145(int) {
            return offset = 145;
        }

        virtual unsigned int offset146(int) {
            return offset = 146;
        }

        virtual unsigned int offset147(int) {
            return offset = 147;
        }

        virtual unsigned int offset148(int) {
            return offset = 148;
        }

        virtual unsigned int offset149(int) {
            return offset = 149;
        }

        virtual unsigned int offset150(int) {
            return offset = 150;
        }

        virtual unsigned int offset151(int) {
            return offset = 151;
        }

        virtual unsigned int offset152(int) {
            return offset = 152;
        }

        virtual unsigned int offset153(int) {
            return offset = 153;
        }

        virtual unsigned int offset154(int) {
            return offset = 154;
        }

        virtual unsigned int offset155(int) {
            return offset = 155;
        }

        virtual unsigned int offset156(int) {
            return offset = 156;
        }

        virtual unsigned int offset157(int) {
            return offset = 157;
        }

        virtual unsigned int offset158(int) {
            return offset = 158;
        }

        virtual unsigned int offset159(int) {
            return offset = 159;
        }

        virtual unsigned int offset160(int) {
            return offset = 160;
        }

        virtual unsigned int offset161(int) {
            return offset = 161;
        }

        virtual unsigned int offset162(int) {
            return offset = 162;
        }

        virtual unsigned int offset163(int) {
            return offset = 163;
        }

        virtual unsigned int offset164(int) {
            return offset = 164;
        }

        virtual unsigned int offset165(int) {
            return offset = 165;
        }

        virtual unsigned int offset166(int) {
            return offset = 166;
        }

        virtual unsigned int offset167(int) {
            return offset = 167;
        }

        virtual unsigned int offset168(int) {
            return offset = 168;
        }

        virtual unsigned int offset169(int) {
            return offset = 169;
        }

        virtual unsigned int offset170(int) {
            return offset = 170;
        }

        virtual unsigned int offset171(int) {
            return offset = 171;
        }

        virtual unsigned int offset172(int) {
            return offset = 172;
        }

        virtual unsigned int offset173(int) {
            return offset = 173;
        }

        virtual unsigned int offset174(int) {
            return offset = 174;
        }

        virtual unsigned int offset175(int) {
            return offset = 175;
        }

        virtual unsigned int offset176(int) {
            return offset = 176;
        }

        virtual unsigned int offset177(int) {
            return offset = 177;
        }

        virtual unsigned int offset178(int) {
            return offset = 178;
        }

        virtual unsigned int offset179(int) {
            return offset = 179;
        }

        virtual unsigned int offset180(int) {
            return offset = 180;
        }

        virtual unsigned int offset181(int) {
            return offset = 181;
        }

        virtual unsigned int offset182(int) {
            return offset = 182;
        }

        virtual unsigned int offset183(int) {
            return offset = 183;
        }

        virtual unsigned int offset184(int) {
            return offset = 184;
        }

        virtual unsigned int offset185(int) {
            return offset = 185;
        }

        virtual unsigned int offset186(int) {
            return offset = 186;
        }

        virtual unsigned int offset187(int) {
            return offset = 187;
        }

        virtual unsigned int offset188(int) {
            return offset = 188;
        }

        virtual unsigned int offset189(int) {
            return offset = 189;
        }

        virtual unsigned int offset190(int) {
            return offset = 190;
        }

        virtual unsigned int offset191(int) {
            return offset = 191;
        }

        virtual unsigned int offset192(int) {
            return offset = 192;
        }

        virtual unsigned int offset193(int) {
            return offset = 193;
        }

        virtual unsigned int offset194(int) {
            return offset = 194;
        }

        virtual unsigned int offset195(int) {
            return offset = 195;
        }

        virtual unsigned int offset196(int) {
            return offset = 196;
        }

        virtual unsigned int offset197(int) {
            return offset = 197;
        }

        virtual unsigned int offset198(int) {
            return offset = 198;
        }

        virtual unsigned int offset199(int) {
            return offset = 199;
        }


        virtual unsigned int offset200(int) {
            return offset = 200;
        }

        virtual unsigned int offset201(int) {
            return offset = 201;
        }

        virtual unsigned int offset202(int) {
            return offset = 202;
        }

        virtual unsigned int offset203(int) {
            return offset = 203;
        }

        virtual unsigned int offset204(int) {
            return offset = 204;
        }

        virtual unsigned int offset205(int) {
            return offset = 205;
        }

        virtual unsigned int offset206(int) {
            return offset = 206;
        }

        virtual unsigned int offset207(int) {
            return offset = 207;
        }

        virtual unsigned int offset208(int) {
            return offset = 208;
        }

        virtual unsigned int offset209(int) {
            return offset = 209;
        }

        virtual unsigned int offset210(int) {
            return offset = 210;
        }

        virtual unsigned int offset211(int) {
            return offset = 211;
        }

        virtual unsigned int offset212(int) {
            return offset = 212;
        }

        virtual unsigned int offset213(int) {
            return offset = 213;
        }

        virtual unsigned int offset214(int) {
            return offset = 214;
        }

        virtual unsigned int offset215(int) {
            return offset = 215;
        }

        virtual unsigned int offset216(int) {
            return offset = 216;
        }

        virtual unsigned int offset217(int) {
            return offset = 217;
        }

        virtual unsigned int offset218(int) {
            return offset = 218;
        }

        virtual unsigned int offset219(int) {
            return offset = 219;
        }

        virtual unsigned int offset220(int) {
            return offset = 220;
        }

        virtual unsigned int offset221(int) {
            return offset = 221;
        }

        virtual unsigned int offset222(int) {
            return offset = 222;
        }

        virtual unsigned int offset223(int) {
            return offset = 223;
        }

        virtual unsigned int offset224(int) {
            return offset = 224;
        }

        virtual unsigned int offset225(int) {
            return offset = 225;
        }

        virtual unsigned int offset226(int) {
            return offset = 226;
        }

        virtual unsigned int offset227(int) {
            return offset = 227;
        }

        virtual unsigned int offset228(int) {
            return offset = 228;
        }

        virtual unsigned int offset229(int) {
            return offset = 229;
        }

        virtual unsigned int offset230(int) {
            return offset = 230;
        }

        virtual unsigned int offset231(int) {
            return offset = 231;
        }

        virtual unsigned int offset232(int) {
            return offset = 232;
        }

        virtual unsigned int offset233(int) {
            return offset = 233;
        }

        virtual unsigned int offset234(int) {
            return offset = 234;
        }

        virtual unsigned int offset235(int) {
            return offset = 235;
        }

        virtual unsigned int offset236(int) {
            return offset = 236;
        }

        virtual unsigned int offset237(int) {
            return offset = 237;
        }

        virtual unsigned int offset238(int) {
            return offset = 238;
        }

        virtual unsigned int offset239(int) {
            return offset = 239;
        }

        virtual unsigned int offset240(int) {
            return offset = 240;
        }

        virtual unsigned int offset241(int) {
            return offset = 241;
        }

        virtual unsigned int offset242(int) {
            return offset = 242;
        }

        virtual unsigned int offset243(int) {
            return offset = 243;
        }

        virtual unsigned int offset244(int) {
            return offset = 244;
        }

        virtual unsigned int offset245(int) {
            return offset = 245;
        }

        virtual unsigned int offset246(int) {
            return offset = 246;
        }

        virtual unsigned int offset247(int) {
            return offset = 247;
        }

        virtual unsigned int offset248(int) {
            return offset = 248;
        }

        virtual unsigned int offset249(int) {
            return offset = 249;
        }

        virtual unsigned int offset250(int) {
            return offset = 250;
        }

        virtual unsigned int offset251(int) {
            return offset = 251;
        }

        virtual unsigned int offset252(int) {
            return offset = 252;
        }

        virtual unsigned int offset253(int) {
            return offset = 253;
        }

        virtual unsigned int offset254(int) {
            return offset = 254;
        }

        virtual unsigned int offset255(int) {
            return offset = 255;
        }

        virtual unsigned int offset256(int) {
            return offset = 256;
        }

        virtual unsigned int offset257(int) {
            return offset = 257;
        }

        virtual unsigned int offset258(int) {
            return offset = 258;
        }

        virtual unsigned int offset259(int) {
            return offset = 259;
        }

        virtual unsigned int offset260(int) {
            return offset = 260;
        }

        virtual unsigned int offset261(int) {
            return offset = 261;
        }

        virtual unsigned int offset262(int) {
            return offset = 262;
        }

        virtual unsigned int offset263(int) {
            return offset = 263;
        }

        virtual unsigned int offset264(int) {
            return offset = 264;
        }

        virtual unsigned int offset265(int) {
            return offset = 265;
        }

        virtual unsigned int offset266(int) {
            return offset = 266;
        }

        virtual unsigned int offset267(int) {
            return offset = 267;
        }

        virtual unsigned int offset268(int) {
            return offset = 268;
        }

        virtual unsigned int offset269(int) {
            return offset = 269;
        }

        virtual unsigned int offset270(int) {
            return offset = 270;
        }

        virtual unsigned int offset271(int) {
            return offset = 271;
        }

        virtual unsigned int offset272(int) {
            return offset = 272;
        }

        virtual unsigned int offset273(int) {
            return offset = 273;
        }

        virtual unsigned int offset274(int) {
            return offset = 274;
        }

        virtual unsigned int offset275(int) {
            return offset = 275;
        }

        virtual unsigned int offset276(int) {
            return offset = 276;
        }

        virtual unsigned int offset277(int) {
            return offset = 277;
        }

        virtual unsigned int offset278(int) {
            return offset = 278;
        }

        virtual unsigned int offset279(int) {
            return offset = 279;
        }

        virtual unsigned int offset280(int) {
            return offset = 280;
        }

        virtual unsigned int offset281(int) {
            return offset = 281;
        }

        virtual unsigned int offset282(int) {
            return offset = 282;
        }

        virtual unsigned int offset283(int) {
            return offset = 283;
        }

        virtual unsigned int offset284(int) {
            return offset = 284;
        }

        virtual unsigned int offset285(int) {
            return offset = 285;
        }

        virtual unsigned int offset286(int) {
            return offset = 286;
        }

        virtual unsigned int offset287(int) {
            return offset = 287;
        }

        virtual unsigned int offset288(int) {
            return offset = 288;
        }

        virtual unsigned int offset289(int) {
            return offset = 289;
        }

        virtual unsigned int offset290(int) {
            return offset = 290;
        }

        virtual unsigned int offset291(int) {
            return offset = 291;
        }

        virtual unsigned int offset292(int) {
            return offset = 292;
        }

        virtual unsigned int offset293(int) {
            return offset = 293;
        }

        virtual unsigned int offset294(int) {
            return offset = 294;
        }

        virtual unsigned int offset295(int) {
            return offset = 295;
        }

        virtual unsigned int offset296(int) {
            return offset = 296;
        }

        virtual unsigned int offset297(int) {
            return offset = 297;
        }

        virtual unsigned int offset298(int) {
            return offset = 298;
        }

        virtual unsigned int offset299(int) {
            return offset = 299;
        }


        virtual unsigned int offset300(int) {
            return offset = 300;
        }

        virtual unsigned int offset301(int) {
            return offset = 301;
        }

        virtual unsigned int offset302(int) {
            return offset = 302;
        }

        virtual unsigned int offset303(int) {
            return offset = 303;
        }

        virtual unsigned int offset304(int) {
            return offset = 304;
        }

        virtual unsigned int offset305(int) {
            return offset = 305;
        }

        virtual unsigned int offset306(int) {
            return offset = 306;
        }

        virtual unsigned int offset307(int) {
            return offset = 307;
        }

        virtual unsigned int offset308(int) {
            return offset = 308;
        }

        virtual unsigned int offset309(int) {
            return offset = 309;
        }

        virtual unsigned int offset310(int) {
            return offset = 310;
        }

        virtual unsigned int offset311(int) {
            return offset = 311;
        }

        virtual unsigned int offset312(int) {
            return offset = 312;
        }

        virtual unsigned int offset313(int) {
            return offset = 313;
        }

        virtual unsigned int offset314(int) {
            return offset = 314;
        }

        virtual unsigned int offset315(int) {
            return offset = 315;
        }

        virtual unsigned int offset316(int) {
            return offset = 316;
        }

        virtual unsigned int offset317(int) {
            return offset = 317;
        }

        virtual unsigned int offset318(int) {
            return offset = 318;
        }

        virtual unsigned int offset319(int) {
            return offset = 319;
        }

        virtual unsigned int offset320(int) {
            return offset = 320;
        }

        virtual unsigned int offset321(int) {
            return offset = 321;
        }

        virtual unsigned int offset322(int) {
            return offset = 322;
        }

        virtual unsigned int offset323(int) {
            return offset = 323;
        }

        virtual unsigned int offset324(int) {
            return offset = 324;
        }

        virtual unsigned int offset325(int) {
            return offset = 325;
        }

        virtual unsigned int offset326(int) {
            return offset = 326;
        }

        virtual unsigned int offset327(int) {
            return offset = 327;
        }

        virtual unsigned int offset328(int) {
            return offset = 328;
        }

        virtual unsigned int offset329(int) {
            return offset = 329;
        }

        virtual unsigned int offset330(int) {
            return offset = 330;
        }

        virtual unsigned int offset331(int) {
            return offset = 331;
        }

        virtual unsigned int offset332(int) {
            return offset = 332;
        }

        virtual unsigned int offset333(int) {
            return offset = 333;
        }

        virtual unsigned int offset334(int) {
            return offset = 334;
        }

        virtual unsigned int offset335(int) {
            return offset = 335;
        }

        virtual unsigned int offset336(int) {
            return offset = 336;
        }

        virtual unsigned int offset337(int) {
            return offset = 337;
        }

        virtual unsigned int offset338(int) {
            return offset = 338;
        }

        virtual unsigned int offset339(int) {
            return offset = 339;
        }

        virtual unsigned int offset340(int) {
            return offset = 340;
        }

        virtual unsigned int offset341(int) {
            return offset = 341;
        }

        virtual unsigned int offset342(int) {
            return offset = 342;
        }

        virtual unsigned int offset343(int) {
            return offset = 343;
        }

        virtual unsigned int offset344(int) {
            return offset = 344;
        }

        virtual unsigned int offset345(int) {
            return offset = 345;
        }

        virtual unsigned int offset346(int) {
            return offset = 346;
        }

        virtual unsigned int offset347(int) {
            return offset = 347;
        }

        virtual unsigned int offset348(int) {
            return offset = 348;
        }

        virtual unsigned int offset349(int) {
            return offset = 349;
        }

        virtual unsigned int offset350(int) {
            return offset = 350;
        }

        virtual unsigned int offset351(int) {
            return offset = 351;
        }

        virtual unsigned int offset352(int) {
            return offset = 352;
        }

        virtual unsigned int offset353(int) {
            return offset = 353;
        }

        virtual unsigned int offset354(int) {
            return offset = 354;
        }

        virtual unsigned int offset355(int) {
            return offset = 355;
        }

        virtual unsigned int offset356(int) {
            return offset = 356;
        }

        virtual unsigned int offset357(int) {
            return offset = 357;
        }

        virtual unsigned int offset358(int) {
            return offset = 358;
        }

        virtual unsigned int offset359(int) {
            return offset = 359;
        }

        virtual unsigned int offset360(int) {
            return offset = 360;
        }

        virtual unsigned int offset361(int) {
            return offset = 361;
        }

        virtual unsigned int offset362(int) {
            return offset = 362;
        }

        virtual unsigned int offset363(int) {
            return offset = 363;
        }

        virtual unsigned int offset364(int) {
            return offset = 364;
        }

        virtual unsigned int offset365(int) {
            return offset = 365;
        }

        virtual unsigned int offset366(int) {
            return offset = 366;
        }

        virtual unsigned int offset367(int) {
            return offset = 367;
        }

        virtual unsigned int offset368(int) {
            return offset = 368;
        }

        virtual unsigned int offset369(int) {
            return offset = 369;
        }

        virtual unsigned int offset370(int) {
            return offset = 370;
        }

        virtual unsigned int offset371(int) {
            return offset = 371;
        }

        virtual unsigned int offset372(int) {
            return offset = 372;
        }

        virtual unsigned int offset373(int) {
            return offset = 373;
        }

        virtual unsigned int offset374(int) {
            return offset = 374;
        }

        virtual unsigned int offset375(int) {
            return offset = 375;
        }

        virtual unsigned int offset376(int) {
            return offset = 376;
        }

        virtual unsigned int offset377(int) {
            return offset = 377;
        }

        virtual unsigned int offset378(int) {
            return offset = 378;
        }

        virtual unsigned int offset379(int) {
            return offset = 379;
        }

        virtual unsigned int offset380(int) {
            return offset = 380;
        }

        virtual unsigned int offset381(int) {
            return offset = 381;
        }

        virtual unsigned int offset382(int) {
            return offset = 382;
        }

        virtual unsigned int offset383(int) {
            return offset = 383;
        }

        virtual unsigned int offset384(int) {
            return offset = 384;
        }

        virtual unsigned int offset385(int) {
            return offset = 385;
        }

        virtual unsigned int offset386(int) {
            return offset = 386;
        }

        virtual unsigned int offset387(int) {
            return offset = 387;
        }

        virtual unsigned int offset388(int) {
            return offset = 388;
        }

        virtual unsigned int offset389(int) {
            return offset = 389;
        }

        virtual unsigned int offset390(int) {
            return offset = 390;
        }

        virtual unsigned int offset391(int) {
            return offset = 391;
        }

        virtual unsigned int offset392(int) {
            return offset = 392;
        }

        virtual unsigned int offset393(int) {
            return offset = 393;
        }

        virtual unsigned int offset394(int) {
            return offset = 394;
        }

        virtual unsigned int offset395(int) {
            return offset = 395;
        }

        virtual unsigned int offset396(int) {
            return offset = 396;
        }

        virtual unsigned int offset397(int) {
            return offset = 397;
        }

        virtual unsigned int offset398(int) {
            return offset = 398;
        }

        virtual unsigned int offset399(int) {
            return offset = 399;
        }


        virtual unsigned int offset400(int) {
            return offset = 400;
        }

        virtual unsigned int offset401(int) {
            return offset = 401;
        }

        virtual unsigned int offset402(int) {
            return offset = 402;
        }

        virtual unsigned int offset403(int) {
            return offset = 403;
        }

        virtual unsigned int offset404(int) {
            return offset = 404;
        }

        virtual unsigned int offset405(int) {
            return offset = 405;
        }

        virtual unsigned int offset406(int) {
            return offset = 406;
        }

        virtual unsigned int offset407(int) {
            return offset = 407;
        }

        virtual unsigned int offset408(int) {
            return offset = 408;
        }

        virtual unsigned int offset409(int) {
            return offset = 409;
        }

        virtual unsigned int offset410(int) {
            return offset = 410;
        }

        virtual unsigned int offset411(int) {
            return offset = 411;
        }

        virtual unsigned int offset412(int) {
            return offset = 412;
        }

        virtual unsigned int offset413(int) {
            return offset = 413;
        }

        virtual unsigned int offset414(int) {
            return offset = 414;
        }

        virtual unsigned int offset415(int) {
            return offset = 415;
        }

        virtual unsigned int offset416(int) {
            return offset = 416;
        }

        virtual unsigned int offset417(int) {
            return offset = 417;
        }

        virtual unsigned int offset418(int) {
            return offset = 418;
        }

        virtual unsigned int offset419(int) {
            return offset = 419;
        }

        virtual unsigned int offset420(int) {
            return offset = 420;
        }

        virtual unsigned int offset421(int) {
            return offset = 421;
        }

        virtual unsigned int offset422(int) {
            return offset = 422;
        }

        virtual unsigned int offset423(int) {
            return offset = 423;
        }

        virtual unsigned int offset424(int) {
            return offset = 424;
        }

        virtual unsigned int offset425(int) {
            return offset = 425;
        }

        virtual unsigned int offset426(int) {
            return offset = 426;
        }

        virtual unsigned int offset427(int) {
            return offset = 427;
        }

        virtual unsigned int offset428(int) {
            return offset = 428;
        }

        virtual unsigned int offset429(int) {
            return offset = 429;
        }

        virtual unsigned int offset430(int) {
            return offset = 430;
        }

        virtual unsigned int offset431(int) {
            return offset = 431;
        }

        virtual unsigned int offset432(int) {
            return offset = 432;
        }

        virtual unsigned int offset433(int) {
            return offset = 433;
        }

        virtual unsigned int offset434(int) {
            return offset = 434;
        }

        virtual unsigned int offset435(int) {
            return offset = 435;
        }

        virtual unsigned int offset436(int) {
            return offset = 436;
        }

        virtual unsigned int offset437(int) {
            return offset = 437;
        }

        virtual unsigned int offset438(int) {
            return offset = 438;
        }

        virtual unsigned int offset439(int) {
            return offset = 439;
        }

        virtual unsigned int offset440(int) {
            return offset = 440;
        }

        virtual unsigned int offset441(int) {
            return offset = 441;
        }

        virtual unsigned int offset442(int) {
            return offset = 442;
        }

        virtual unsigned int offset443(int) {
            return offset = 443;
        }

        virtual unsigned int offset444(int) {
            return offset = 444;
        }

        virtual unsigned int offset445(int) {
            return offset = 445;
        }

        virtual unsigned int offset446(int) {
            return offset = 446;
        }

        virtual unsigned int offset447(int) {
            return offset = 447;
        }

        virtual unsigned int offset448(int) {
            return offset = 448;
        }

        virtual unsigned int offset449(int) {
            return offset = 449;
        }

        virtual unsigned int offset450(int) {
            return offset = 450;
        }

        virtual unsigned int offset451(int) {
            return offset = 451;
        }

        virtual unsigned int offset452(int) {
            return offset = 452;
        }

        virtual unsigned int offset453(int) {
            return offset = 453;
        }

        virtual unsigned int offset454(int) {
            return offset = 454;
        }

        virtual unsigned int offset455(int) {
            return offset = 455;
        }

        virtual unsigned int offset456(int) {
            return offset = 456;
        }

        virtual unsigned int offset457(int) {
            return offset = 457;
        }

        virtual unsigned int offset458(int) {
            return offset = 458;
        }

        virtual unsigned int offset459(int) {
            return offset = 459;
        }

        virtual unsigned int offset460(int) {
            return offset = 460;
        }

        virtual unsigned int offset461(int) {
            return offset = 461;
        }

        virtual unsigned int offset462(int) {
            return offset = 462;
        }

        virtual unsigned int offset463(int) {
            return offset = 463;
        }

        virtual unsigned int offset464(int) {
            return offset = 464;
        }

        virtual unsigned int offset465(int) {
            return offset = 465;
        }

        virtual unsigned int offset466(int) {
            return offset = 466;
        }

        virtual unsigned int offset467(int) {
            return offset = 467;
        }

        virtual unsigned int offset468(int) {
            return offset = 468;
        }

        virtual unsigned int offset469(int) {
            return offset = 469;
        }

        virtual unsigned int offset470(int) {
            return offset = 470;
        }

        virtual unsigned int offset471(int) {
            return offset = 471;
        }

        virtual unsigned int offset472(int) {
            return offset = 472;
        }

        virtual unsigned int offset473(int) {
            return offset = 473;
        }

        virtual unsigned int offset474(int) {
            return offset = 474;
        }

        virtual unsigned int offset475(int) {
            return offset = 475;
        }

        virtual unsigned int offset476(int) {
            return offset = 476;
        }

        virtual unsigned int offset477(int) {
            return offset = 477;
        }

        virtual unsigned int offset478(int) {
            return offset = 478;
        }

        virtual unsigned int offset479(int) {
            return offset = 479;
        }

        virtual unsigned int offset480(int) {
            return offset = 480;
        }

        virtual unsigned int offset481(int) {
            return offset = 481;
        }

        virtual unsigned int offset482(int) {
            return offset = 482;
        }

        virtual unsigned int offset483(int) {
            return offset = 483;
        }

        virtual unsigned int offset484(int) {
            return offset = 484;
        }

        virtual unsigned int offset485(int) {
            return offset = 485;
        }

        virtual unsigned int offset486(int) {
            return offset = 486;
        }

        virtual unsigned int offset487(int) {
            return offset = 487;
        }

        virtual unsigned int offset488(int) {
            return offset = 488;
        }

        virtual unsigned int offset489(int) {
            return offset = 489;
        }

        virtual unsigned int offset490(int) {
            return offset = 490;
        }

        virtual unsigned int offset491(int) {
            return offset = 491;
        }

        virtual unsigned int offset492(int) {
            return offset = 492;
        }

        virtual unsigned int offset493(int) {
            return offset = 493;
        }

        virtual unsigned int offset494(int) {
            return offset = 494;
        }

        virtual unsigned int offset495(int) {
            return offset = 495;
        }

        virtual unsigned int offset496(int) {
            return offset = 496;
        }

        virtual unsigned int offset497(int) {
            return offset = 497;
        }

        virtual unsigned int offset498(int) {
            return offset = 498;
        }

        virtual unsigned int offset499(int) {
            return offset = 499;
        }


        virtual unsigned int offset500(int) {
            return offset = 500;
        }

        virtual unsigned int offset501(int) {
            return offset = 501;
        }

        virtual unsigned int offset502(int) {
            return offset = 502;
        }

        virtual unsigned int offset503(int) {
            return offset = 503;
        }

        virtual unsigned int offset504(int) {
            return offset = 504;
        }

        virtual unsigned int offset505(int) {
            return offset = 505;
        }

        virtual unsigned int offset506(int) {
            return offset = 506;
        }

        virtual unsigned int offset507(int) {
            return offset = 507;
        }

        virtual unsigned int offset508(int) {
            return offset = 508;
        }

        virtual unsigned int offset509(int) {
            return offset = 509;
        }

        virtual unsigned int offset510(int) {
            return offset = 510;
        }

        virtual unsigned int offset511(int) {
            return offset = 511;
        }

        virtual unsigned int offset512(int) {
            return offset = 512;
        }

        virtual unsigned int offset513(int) {
            return offset = 513;
        }

        virtual unsigned int offset514(int) {
            return offset = 514;
        }

        virtual unsigned int offset515(int) {
            return offset = 515;
        }

        virtual unsigned int offset516(int) {
            return offset = 516;
        }

        virtual unsigned int offset517(int) {
            return offset = 517;
        }

        virtual unsigned int offset518(int) {
            return offset = 518;
        }

        virtual unsigned int offset519(int) {
            return offset = 519;
        }

        virtual unsigned int offset520(int) {
            return offset = 520;
        }

        virtual unsigned int offset521(int) {
            return offset = 521;
        }

        virtual unsigned int offset522(int) {
            return offset = 522;
        }

        virtual unsigned int offset523(int) {
            return offset = 523;
        }

        virtual unsigned int offset524(int) {
            return offset = 524;
        }

        virtual unsigned int offset525(int) {
            return offset = 525;
        }

        virtual unsigned int offset526(int) {
            return offset = 526;
        }

        virtual unsigned int offset527(int) {
            return offset = 527;
        }

        virtual unsigned int offset528(int) {
            return offset = 528;
        }

        virtual unsigned int offset529(int) {
            return offset = 529;
        }

        virtual unsigned int offset530(int) {
            return offset = 530;
        }

        virtual unsigned int offset531(int) {
            return offset = 531;
        }

        virtual unsigned int offset532(int) {
            return offset = 532;
        }

        virtual unsigned int offset533(int) {
            return offset = 533;
        }

        virtual unsigned int offset534(int) {
            return offset = 534;
        }

        virtual unsigned int offset535(int) {
            return offset = 535;
        }

        virtual unsigned int offset536(int) {
            return offset = 536;
        }

        virtual unsigned int offset537(int) {
            return offset = 537;
        }

        virtual unsigned int offset538(int) {
            return offset = 538;
        }

        virtual unsigned int offset539(int) {
            return offset = 539;
        }

        virtual unsigned int offset540(int) {
            return offset = 540;
        }

        virtual unsigned int offset541(int) {
            return offset = 541;
        }

        virtual unsigned int offset542(int) {
            return offset = 542;
        }

        virtual unsigned int offset543(int) {
            return offset = 543;
        }

        virtual unsigned int offset544(int) {
            return offset = 544;
        }

        virtual unsigned int offset545(int) {
            return offset = 545;
        }

        virtual unsigned int offset546(int) {
            return offset = 546;
        }

        virtual unsigned int offset547(int) {
            return offset = 547;
        }

        virtual unsigned int offset548(int) {
            return offset = 548;
        }

        virtual unsigned int offset549(int) {
            return offset = 549;
        }

        virtual unsigned int offset550(int) {
            return offset = 550;
        }

        virtual unsigned int offset551(int) {
            return offset = 551;
        }

        virtual unsigned int offset552(int) {
            return offset = 552;
        }

        virtual unsigned int offset553(int) {
            return offset = 553;
        }

        virtual unsigned int offset554(int) {
            return offset = 554;
        }

        virtual unsigned int offset555(int) {
            return offset = 555;
        }

        virtual unsigned int offset556(int) {
            return offset = 556;
        }

        virtual unsigned int offset557(int) {
            return offset = 557;
        }

        virtual unsigned int offset558(int) {
            return offset = 558;
        }

        virtual unsigned int offset559(int) {
            return offset = 559;
        }

        virtual unsigned int offset560(int) {
            return offset = 560;
        }

        virtual unsigned int offset561(int) {
            return offset = 561;
        }

        virtual unsigned int offset562(int) {
            return offset = 562;
        }

        virtual unsigned int offset563(int) {
            return offset = 563;
        }

        virtual unsigned int offset564(int) {
            return offset = 564;
        }

        virtual unsigned int offset565(int) {
            return offset = 565;
        }

        virtual unsigned int offset566(int) {
            return offset = 566;
        }

        virtual unsigned int offset567(int) {
            return offset = 567;
        }

        virtual unsigned int offset568(int) {
            return offset = 568;
        }

        virtual unsigned int offset569(int) {
            return offset = 569;
        }

        virtual unsigned int offset570(int) {
            return offset = 570;
        }

        virtual unsigned int offset571(int) {
            return offset = 571;
        }

        virtual unsigned int offset572(int) {
            return offset = 572;
        }

        virtual unsigned int offset573(int) {
            return offset = 573;
        }

        virtual unsigned int offset574(int) {
            return offset = 574;
        }

        virtual unsigned int offset575(int) {
            return offset = 575;
        }

        virtual unsigned int offset576(int) {
            return offset = 576;
        }

        virtual unsigned int offset577(int) {
            return offset = 577;
        }

        virtual unsigned int offset578(int) {
            return offset = 578;
        }

        virtual unsigned int offset579(int) {
            return offset = 579;
        }

        virtual unsigned int offset580(int) {
            return offset = 580;
        }

        virtual unsigned int offset581(int) {
            return offset = 581;
        }

        virtual unsigned int offset582(int) {
            return offset = 582;
        }

        virtual unsigned int offset583(int) {
            return offset = 583;
        }

        virtual unsigned int offset584(int) {
            return offset = 584;
        }

        virtual unsigned int offset585(int) {
            return offset = 585;
        }

        virtual unsigned int offset586(int) {
            return offset = 586;
        }

        virtual unsigned int offset587(int) {
            return offset = 587;
        }

        virtual unsigned int offset588(int) {
            return offset = 588;
        }

        virtual unsigned int offset589(int) {
            return offset = 589;
        }

        virtual unsigned int offset590(int) {
            return offset = 590;
        }

        virtual unsigned int offset591(int) {
            return offset = 591;
        }

        virtual unsigned int offset592(int) {
            return offset = 592;
        }

        virtual unsigned int offset593(int) {
            return offset = 593;
        }

        virtual unsigned int offset594(int) {
            return offset = 594;
        }

        virtual unsigned int offset595(int) {
            return offset = 595;
        }

        virtual unsigned int offset596(int) {
            return offset = 596;
        }

        virtual unsigned int offset597(int) {
            return offset = 597;
        }

        virtual unsigned int offset598(int) {
            return offset = 598;
        }

        virtual unsigned int offset599(int) {
            return offset = 599;
        }


        virtual unsigned int offset600(int) {
            return offset = 600;
        }

        virtual unsigned int offset601(int) {
            return offset = 601;
        }

        virtual unsigned int offset602(int) {
            return offset = 602;
        }

        virtual unsigned int offset603(int) {
            return offset = 603;
        }

        virtual unsigned int offset604(int) {
            return offset = 604;
        }

        virtual unsigned int offset605(int) {
            return offset = 605;
        }

        virtual unsigned int offset606(int) {
            return offset = 606;
        }

        virtual unsigned int offset607(int) {
            return offset = 607;
        }

        virtual unsigned int offset608(int) {
            return offset = 608;
        }

        virtual unsigned int offset609(int) {
            return offset = 609;
        }

        virtual unsigned int offset610(int) {
            return offset = 610;
        }

        virtual unsigned int offset611(int) {
            return offset = 611;
        }

        virtual unsigned int offset612(int) {
            return offset = 612;
        }

        virtual unsigned int offset613(int) {
            return offset = 613;
        }

        virtual unsigned int offset614(int) {
            return offset = 614;
        }

        virtual unsigned int offset615(int) {
            return offset = 615;
        }

        virtual unsigned int offset616(int) {
            return offset = 616;
        }

        virtual unsigned int offset617(int) {
            return offset = 617;
        }

        virtual unsigned int offset618(int) {
            return offset = 618;
        }

        virtual unsigned int offset619(int) {
            return offset = 619;
        }

        virtual unsigned int offset620(int) {
            return offset = 620;
        }

        virtual unsigned int offset621(int) {
            return offset = 621;
        }

        virtual unsigned int offset622(int) {
            return offset = 622;
        }

        virtual unsigned int offset623(int) {
            return offset = 623;
        }

        virtual unsigned int offset624(int) {
            return offset = 624;
        }

        virtual unsigned int offset625(int) {
            return offset = 625;
        }

        virtual unsigned int offset626(int) {
            return offset = 626;
        }

        virtual unsigned int offset627(int) {
            return offset = 627;
        }

        virtual unsigned int offset628(int) {
            return offset = 628;
        }

        virtual unsigned int offset629(int) {
            return offset = 629;
        }

        virtual unsigned int offset630(int) {
            return offset = 630;
        }

        virtual unsigned int offset631(int) {
            return offset = 631;
        }

        virtual unsigned int offset632(int) {
            return offset = 632;
        }

        virtual unsigned int offset633(int) {
            return offset = 633;
        }

        virtual unsigned int offset634(int) {
            return offset = 634;
        }

        virtual unsigned int offset635(int) {
            return offset = 635;
        }

        virtual unsigned int offset636(int) {
            return offset = 636;
        }

        virtual unsigned int offset637(int) {
            return offset = 637;
        }

        virtual unsigned int offset638(int) {
            return offset = 638;
        }

        virtual unsigned int offset639(int) {
            return offset = 639;
        }

        virtual unsigned int offset640(int) {
            return offset = 640;
        }

        virtual unsigned int offset641(int) {
            return offset = 641;
        }

        virtual unsigned int offset642(int) {
            return offset = 642;
        }

        virtual unsigned int offset643(int) {
            return offset = 643;
        }

        virtual unsigned int offset644(int) {
            return offset = 644;
        }

        virtual unsigned int offset645(int) {
            return offset = 645;
        }

        virtual unsigned int offset646(int) {
            return offset = 646;
        }

        virtual unsigned int offset647(int) {
            return offset = 647;
        }

        virtual unsigned int offset648(int) {
            return offset = 648;
        }

        virtual unsigned int offset649(int) {
            return offset = 649;
        }

        virtual unsigned int offset650(int) {
            return offset = 650;
        }

        virtual unsigned int offset651(int) {
            return offset = 651;
        }

        virtual unsigned int offset652(int) {
            return offset = 652;
        }

        virtual unsigned int offset653(int) {
            return offset = 653;
        }

        virtual unsigned int offset654(int) {
            return offset = 654;
        }

        virtual unsigned int offset655(int) {
            return offset = 655;
        }

        virtual unsigned int offset656(int) {
            return offset = 656;
        }

        virtual unsigned int offset657(int) {
            return offset = 657;
        }

        virtual unsigned int offset658(int) {
            return offset = 658;
        }

        virtual unsigned int offset659(int) {
            return offset = 659;
        }

        virtual unsigned int offset660(int) {
            return offset = 660;
        }

        virtual unsigned int offset661(int) {
            return offset = 661;
        }

        virtual unsigned int offset662(int) {
            return offset = 662;
        }

        virtual unsigned int offset663(int) {
            return offset = 663;
        }

        virtual unsigned int offset664(int) {
            return offset = 664;
        }

        virtual unsigned int offset665(int) {
            return offset = 665;
        }

        virtual unsigned int offset666(int) {
            return offset = 666;
        }

        virtual unsigned int offset667(int) {
            return offset = 667;
        }

        virtual unsigned int offset668(int) {
            return offset = 668;
        }

        virtual unsigned int offset669(int) {
            return offset = 669;
        }

        virtual unsigned int offset670(int) {
            return offset = 670;
        }

        virtual unsigned int offset671(int) {
            return offset = 671;
        }

        virtual unsigned int offset672(int) {
            return offset = 672;
        }

        virtual unsigned int offset673(int) {
            return offset = 673;
        }

        virtual unsigned int offset674(int) {
            return offset = 674;
        }

        virtual unsigned int offset675(int) {
            return offset = 675;
        }

        virtual unsigned int offset676(int) {
            return offset = 676;
        }

        virtual unsigned int offset677(int) {
            return offset = 677;
        }

        virtual unsigned int offset678(int) {
            return offset = 678;
        }

        virtual unsigned int offset679(int) {
            return offset = 679;
        }

        virtual unsigned int offset680(int) {
            return offset = 680;
        }

        virtual unsigned int offset681(int) {
            return offset = 681;
        }

        virtual unsigned int offset682(int) {
            return offset = 682;
        }

        virtual unsigned int offset683(int) {
            return offset = 683;
        }

        virtual unsigned int offset684(int) {
            return offset = 684;
        }

        virtual unsigned int offset685(int) {
            return offset = 685;
        }

        virtual unsigned int offset686(int) {
            return offset = 686;
        }

        virtual unsigned int offset687(int) {
            return offset = 687;
        }

        virtual unsigned int offset688(int) {
            return offset = 688;
        }

        virtual unsigned int offset689(int) {
            return offset = 689;
        }

        virtual unsigned int offset690(int) {
            return offset = 690;
        }

        virtual unsigned int offset691(int) {
            return offset = 691;
        }

        virtual unsigned int offset692(int) {
            return offset = 692;
        }

        virtual unsigned int offset693(int) {
            return offset = 693;
        }

        virtual unsigned int offset694(int) {
            return offset = 694;
        }

        virtual unsigned int offset695(int) {
            return offset = 695;
        }

        virtual unsigned int offset696(int) {
            return offset = 696;
        }

        virtual unsigned int offset697(int) {
            return offset = 697;
        }

        virtual unsigned int offset698(int) {
            return offset = 698;
        }

        virtual unsigned int offset699(int) {
            return offset = 699;
        }


        virtual unsigned int offset700(int) {
            return offset = 700;
        }

        virtual unsigned int offset701(int) {
            return offset = 701;
        }

        virtual unsigned int offset702(int) {
            return offset = 702;
        }

        virtual unsigned int offset703(int) {
            return offset = 703;
        }

        virtual unsigned int offset704(int) {
            return offset = 704;
        }

        virtual unsigned int offset705(int) {
            return offset = 705;
        }

        virtual unsigned int offset706(int) {
            return offset = 706;
        }

        virtual unsigned int offset707(int) {
            return offset = 707;
        }

        virtual unsigned int offset708(int) {
            return offset = 708;
        }

        virtual unsigned int offset709(int) {
            return offset = 709;
        }

        virtual unsigned int offset710(int) {
            return offset = 710;
        }

        virtual unsigned int offset711(int) {
            return offset = 711;
        }

        virtual unsigned int offset712(int) {
            return offset = 712;
        }

        virtual unsigned int offset713(int) {
            return offset = 713;
        }

        virtual unsigned int offset714(int) {
            return offset = 714;
        }

        virtual unsigned int offset715(int) {
            return offset = 715;
        }

        virtual unsigned int offset716(int) {
            return offset = 716;
        }

        virtual unsigned int offset717(int) {
            return offset = 717;
        }

        virtual unsigned int offset718(int) {
            return offset = 718;
        }

        virtual unsigned int offset719(int) {
            return offset = 719;
        }

        virtual unsigned int offset720(int) {
            return offset = 720;
        }

        virtual unsigned int offset721(int) {
            return offset = 721;
        }

        virtual unsigned int offset722(int) {
            return offset = 722;
        }

        virtual unsigned int offset723(int) {
            return offset = 723;
        }

        virtual unsigned int offset724(int) {
            return offset = 724;
        }

        virtual unsigned int offset725(int) {
            return offset = 725;
        }

        virtual unsigned int offset726(int) {
            return offset = 726;
        }

        virtual unsigned int offset727(int) {
            return offset = 727;
        }

        virtual unsigned int offset728(int) {
            return offset = 728;
        }

        virtual unsigned int offset729(int) {
            return offset = 729;
        }

        virtual unsigned int offset730(int) {
            return offset = 730;
        }

        virtual unsigned int offset731(int) {
            return offset = 731;
        }

        virtual unsigned int offset732(int) {
            return offset = 732;
        }

        virtual unsigned int offset733(int) {
            return offset = 733;
        }

        virtual unsigned int offset734(int) {
            return offset = 734;
        }

        virtual unsigned int offset735(int) {
            return offset = 735;
        }

        virtual unsigned int offset736(int) {
            return offset = 736;
        }

        virtual unsigned int offset737(int) {
            return offset = 737;
        }

        virtual unsigned int offset738(int) {
            return offset = 738;
        }

        virtual unsigned int offset739(int) {
            return offset = 739;
        }

        virtual unsigned int offset740(int) {
            return offset = 740;
        }

        virtual unsigned int offset741(int) {
            return offset = 741;
        }

        virtual unsigned int offset742(int) {
            return offset = 742;
        }

        virtual unsigned int offset743(int) {
            return offset = 743;
        }

        virtual unsigned int offset744(int) {
            return offset = 744;
        }

        virtual unsigned int offset745(int) {
            return offset = 745;
        }

        virtual unsigned int offset746(int) {
            return offset = 746;
        }

        virtual unsigned int offset747(int) {
            return offset = 747;
        }

        virtual unsigned int offset748(int) {
            return offset = 748;
        }

        virtual unsigned int offset749(int) {
            return offset = 749;
        }

        virtual unsigned int offset750(int) {
            return offset = 750;
        }

        virtual unsigned int offset751(int) {
            return offset = 751;
        }

        virtual unsigned int offset752(int) {
            return offset = 752;
        }

        virtual unsigned int offset753(int) {
            return offset = 753;
        }

        virtual unsigned int offset754(int) {
            return offset = 754;
        }

        virtual unsigned int offset755(int) {
            return offset = 755;
        }

        virtual unsigned int offset756(int) {
            return offset = 756;
        }

        virtual unsigned int offset757(int) {
            return offset = 757;
        }

        virtual unsigned int offset758(int) {
            return offset = 758;
        }

        virtual unsigned int offset759(int) {
            return offset = 759;
        }

        virtual unsigned int offset760(int) {
            return offset = 760;
        }

        virtual unsigned int offset761(int) {
            return offset = 761;
        }

        virtual unsigned int offset762(int) {
            return offset = 762;
        }

        virtual unsigned int offset763(int) {
            return offset = 763;
        }

        virtual unsigned int offset764(int) {
            return offset = 764;
        }

        virtual unsigned int offset765(int) {
            return offset = 765;
        }

        virtual unsigned int offset766(int) {
            return offset = 766;
        }

        virtual unsigned int offset767(int) {
            return offset = 767;
        }

        virtual unsigned int offset768(int) {
            return offset = 768;
        }

        virtual unsigned int offset769(int) {
            return offset = 769;
        }

        virtual unsigned int offset770(int) {
            return offset = 770;
        }

        virtual unsigned int offset771(int) {
            return offset = 771;
        }

        virtual unsigned int offset772(int) {
            return offset = 772;
        }

        virtual unsigned int offset773(int) {
            return offset = 773;
        }

        virtual unsigned int offset774(int) {
            return offset = 774;
        }

        virtual unsigned int offset775(int) {
            return offset = 775;
        }

        virtual unsigned int offset776(int) {
            return offset = 776;
        }

        virtual unsigned int offset777(int) {
            return offset = 777;
        }

        virtual unsigned int offset778(int) {
            return offset = 778;
        }

        virtual unsigned int offset779(int) {
            return offset = 779;
        }

        virtual unsigned int offset780(int) {
            return offset = 780;
        }

        virtual unsigned int offset781(int) {
            return offset = 781;
        }

        virtual unsigned int offset782(int) {
            return offset = 782;
        }

        virtual unsigned int offset783(int) {
            return offset = 783;
        }

        virtual unsigned int offset784(int) {
            return offset = 784;
        }

        virtual unsigned int offset785(int) {
            return offset = 785;
        }

        virtual unsigned int offset786(int) {
            return offset = 786;
        }

        virtual unsigned int offset787(int) {
            return offset = 787;
        }

        virtual unsigned int offset788(int) {
            return offset = 788;
        }

        virtual unsigned int offset789(int) {
            return offset = 789;
        }

        virtual unsigned int offset790(int) {
            return offset = 790;
        }

        virtual unsigned int offset791(int) {
            return offset = 791;
        }

        virtual unsigned int offset792(int) {
            return offset = 792;
        }

        virtual unsigned int offset793(int) {
            return offset = 793;
        }

        virtual unsigned int offset794(int) {
            return offset = 794;
        }

        virtual unsigned int offset795(int) {
            return offset = 795;
        }

        virtual unsigned int offset796(int) {
            return offset = 796;
        }

        virtual unsigned int offset797(int) {
            return offset = 797;
        }

        virtual unsigned int offset798(int) {
            return offset = 798;
        }

        virtual unsigned int offset799(int) {
            return offset = 799;
        }


        virtual unsigned int offset800(int) {
            return offset = 800;
        }

        virtual unsigned int offset801(int) {
            return offset = 801;
        }

        virtual unsigned int offset802(int) {
            return offset = 802;
        }

        virtual unsigned int offset803(int) {
            return offset = 803;
        }

        virtual unsigned int offset804(int) {
            return offset = 804;
        }

        virtual unsigned int offset805(int) {
            return offset = 805;
        }

        virtual unsigned int offset806(int) {
            return offset = 806;
        }

        virtual unsigned int offset807(int) {
            return offset = 807;
        }

        virtual unsigned int offset808(int) {
            return offset = 808;
        }

        virtual unsigned int offset809(int) {
            return offset = 809;
        }

        virtual unsigned int offset810(int) {
            return offset = 810;
        }

        virtual unsigned int offset811(int) {
            return offset = 811;
        }

        virtual unsigned int offset812(int) {
            return offset = 812;
        }

        virtual unsigned int offset813(int) {
            return offset = 813;
        }

        virtual unsigned int offset814(int) {
            return offset = 814;
        }

        virtual unsigned int offset815(int) {
            return offset = 815;
        }

        virtual unsigned int offset816(int) {
            return offset = 816;
        }

        virtual unsigned int offset817(int) {
            return offset = 817;
        }

        virtual unsigned int offset818(int) {
            return offset = 818;
        }

        virtual unsigned int offset819(int) {
            return offset = 819;
        }

        virtual unsigned int offset820(int) {
            return offset = 820;
        }

        virtual unsigned int offset821(int) {
            return offset = 821;
        }

        virtual unsigned int offset822(int) {
            return offset = 822;
        }

        virtual unsigned int offset823(int) {
            return offset = 823;
        }

        virtual unsigned int offset824(int) {
            return offset = 824;
        }

        virtual unsigned int offset825(int) {
            return offset = 825;
        }

        virtual unsigned int offset826(int) {
            return offset = 826;
        }

        virtual unsigned int offset827(int) {
            return offset = 827;
        }

        virtual unsigned int offset828(int) {
            return offset = 828;
        }

        virtual unsigned int offset829(int) {
            return offset = 829;
        }

        virtual unsigned int offset830(int) {
            return offset = 830;
        }

        virtual unsigned int offset831(int) {
            return offset = 831;
        }

        virtual unsigned int offset832(int) {
            return offset = 832;
        }

        virtual unsigned int offset833(int) {
            return offset = 833;
        }

        virtual unsigned int offset834(int) {
            return offset = 834;
        }

        virtual unsigned int offset835(int) {
            return offset = 835;
        }

        virtual unsigned int offset836(int) {
            return offset = 836;
        }

        virtual unsigned int offset837(int) {
            return offset = 837;
        }

        virtual unsigned int offset838(int) {
            return offset = 838;
        }

        virtual unsigned int offset839(int) {
            return offset = 839;
        }

        virtual unsigned int offset840(int) {
            return offset = 840;
        }

        virtual unsigned int offset841(int) {
            return offset = 841;
        }

        virtual unsigned int offset842(int) {
            return offset = 842;
        }

        virtual unsigned int offset843(int) {
            return offset = 843;
        }

        virtual unsigned int offset844(int) {
            return offset = 844;
        }

        virtual unsigned int offset845(int) {
            return offset = 845;
        }

        virtual unsigned int offset846(int) {
            return offset = 846;
        }

        virtual unsigned int offset847(int) {
            return offset = 847;
        }

        virtual unsigned int offset848(int) {
            return offset = 848;
        }

        virtual unsigned int offset849(int) {
            return offset = 849;
        }

        virtual unsigned int offset850(int) {
            return offset = 850;
        }

        virtual unsigned int offset851(int) {
            return offset = 851;
        }

        virtual unsigned int offset852(int) {
            return offset = 852;
        }

        virtual unsigned int offset853(int) {
            return offset = 853;
        }

        virtual unsigned int offset854(int) {
            return offset = 854;
        }

        virtual unsigned int offset855(int) {
            return offset = 855;
        }

        virtual unsigned int offset856(int) {
            return offset = 856;
        }

        virtual unsigned int offset857(int) {
            return offset = 857;
        }

        virtual unsigned int offset858(int) {
            return offset = 858;
        }

        virtual unsigned int offset859(int) {
            return offset = 859;
        }

        virtual unsigned int offset860(int) {
            return offset = 860;
        }

        virtual unsigned int offset861(int) {
            return offset = 861;
        }

        virtual unsigned int offset862(int) {
            return offset = 862;
        }

        virtual unsigned int offset863(int) {
            return offset = 863;
        }

        virtual unsigned int offset864(int) {
            return offset = 864;
        }

        virtual unsigned int offset865(int) {
            return offset = 865;
        }

        virtual unsigned int offset866(int) {
            return offset = 866;
        }

        virtual unsigned int offset867(int) {
            return offset = 867;
        }

        virtual unsigned int offset868(int) {
            return offset = 868;
        }

        virtual unsigned int offset869(int) {
            return offset = 869;
        }

        virtual unsigned int offset870(int) {
            return offset = 870;
        }

        virtual unsigned int offset871(int) {
            return offset = 871;
        }

        virtual unsigned int offset872(int) {
            return offset = 872;
        }

        virtual unsigned int offset873(int) {
            return offset = 873;
        }

        virtual unsigned int offset874(int) {
            return offset = 874;
        }

        virtual unsigned int offset875(int) {
            return offset = 875;
        }

        virtual unsigned int offset876(int) {
            return offset = 876;
        }

        virtual unsigned int offset877(int) {
            return offset = 877;
        }

        virtual unsigned int offset878(int) {
            return offset = 878;
        }

        virtual unsigned int offset879(int) {
            return offset = 879;
        }

        virtual unsigned int offset880(int) {
            return offset = 880;
        }

        virtual unsigned int offset881(int) {
            return offset = 881;
        }

        virtual unsigned int offset882(int) {
            return offset = 882;
        }

        virtual unsigned int offset883(int) {
            return offset = 883;
        }

        virtual unsigned int offset884(int) {
            return offset = 884;
        }

        virtual unsigned int offset885(int) {
            return offset = 885;
        }

        virtual unsigned int offset886(int) {
            return offset = 886;
        }

        virtual unsigned int offset887(int) {
            return offset = 887;
        }

        virtual unsigned int offset888(int) {
            return offset = 888;
        }

        virtual unsigned int offset889(int) {
            return offset = 889;
        }

        virtual unsigned int offset890(int) {
            return offset = 890;
        }

        virtual unsigned int offset891(int) {
            return offset = 891;
        }

        virtual unsigned int offset892(int) {
            return offset = 892;
        }

        virtual unsigned int offset893(int) {
            return offset = 893;
        }

        virtual unsigned int offset894(int) {
            return offset = 894;
        }

        virtual unsigned int offset895(int) {
            return offset = 895;
        }

        virtual unsigned int offset896(int) {
            return offset = 896;
        }

        virtual unsigned int offset897(int) {
            return offset = 897;
        }

        virtual unsigned int offset898(int) {
            return offset = 898;
        }

        virtual unsigned int offset899(int) {
            return offset = 899;
        }


        virtual unsigned int offset900(int) {
            return offset = 900;
        }

        virtual unsigned int offset901(int) {
            return offset = 901;
        }

        virtual unsigned int offset902(int) {
            return offset = 902;
        }

        virtual unsigned int offset903(int) {
            return offset = 903;
        }

        virtual unsigned int offset904(int) {
            return offset = 904;
        }

        virtual unsigned int offset905(int) {
            return offset = 905;
        }

        virtual unsigned int offset906(int) {
            return offset = 906;
        }

        virtual unsigned int offset907(int) {
            return offset = 907;
        }

        virtual unsigned int offset908(int) {
            return offset = 908;
        }

        virtual unsigned int offset909(int) {
            return offset = 909;
        }

        virtual unsigned int offset910(int) {
            return offset = 910;
        }

        virtual unsigned int offset911(int) {
            return offset = 911;
        }

        virtual unsigned int offset912(int) {
            return offset = 912;
        }

        virtual unsigned int offset913(int) {
            return offset = 913;
        }

        virtual unsigned int offset914(int) {
            return offset = 914;
        }

        virtual unsigned int offset915(int) {
            return offset = 915;
        }

        virtual unsigned int offset916(int) {
            return offset = 916;
        }

        virtual unsigned int offset917(int) {
            return offset = 917;
        }

        virtual unsigned int offset918(int) {
            return offset = 918;
        }

        virtual unsigned int offset919(int) {
            return offset = 919;
        }

        virtual unsigned int offset920(int) {
            return offset = 920;
        }

        virtual unsigned int offset921(int) {
            return offset = 921;
        }

        virtual unsigned int offset922(int) {
            return offset = 922;
        }

        virtual unsigned int offset923(int) {
            return offset = 923;
        }

        virtual unsigned int offset924(int) {
            return offset = 924;
        }

        virtual unsigned int offset925(int) {
            return offset = 925;
        }

        virtual unsigned int offset926(int) {
            return offset = 926;
        }

        virtual unsigned int offset927(int) {
            return offset = 927;
        }

        virtual unsigned int offset928(int) {
            return offset = 928;
        }

        virtual unsigned int offset929(int) {
            return offset = 929;
        }

        virtual unsigned int offset930(int) {
            return offset = 930;
        }

        virtual unsigned int offset931(int) {
            return offset = 931;
        }

        virtual unsigned int offset932(int) {
            return offset = 932;
        }

        virtual unsigned int offset933(int) {
            return offset = 933;
        }

        virtual unsigned int offset934(int) {
            return offset = 934;
        }

        virtual unsigned int offset935(int) {
            return offset = 935;
        }

        virtual unsigned int offset936(int) {
            return offset = 936;
        }

        virtual unsigned int offset937(int) {
            return offset = 937;
        }

        virtual unsigned int offset938(int) {
            return offset = 938;
        }

        virtual unsigned int offset939(int) {
            return offset = 939;
        }

        virtual unsigned int offset940(int) {
            return offset = 940;
        }

        virtual unsigned int offset941(int) {
            return offset = 941;
        }

        virtual unsigned int offset942(int) {
            return offset = 942;
        }

        virtual unsigned int offset943(int) {
            return offset = 943;
        }

        virtual unsigned int offset944(int) {
            return offset = 944;
        }

        virtual unsigned int offset945(int) {
            return offset = 945;
        }

        virtual unsigned int offset946(int) {
            return offset = 946;
        }

        virtual unsigned int offset947(int) {
            return offset = 947;
        }

        virtual unsigned int offset948(int) {
            return offset = 948;
        }

        virtual unsigned int offset949(int) {
            return offset = 949;
        }

        virtual unsigned int offset950(int) {
            return offset = 950;
        }

        virtual unsigned int offset951(int) {
            return offset = 951;
        }

        virtual unsigned int offset952(int) {
            return offset = 952;
        }

        virtual unsigned int offset953(int) {
            return offset = 953;
        }

        virtual unsigned int offset954(int) {
            return offset = 954;
        }

        virtual unsigned int offset955(int) {
            return offset = 955;
        }

        virtual unsigned int offset956(int) {
            return offset = 956;
        }

        virtual unsigned int offset957(int) {
            return offset = 957;
        }

        virtual unsigned int offset958(int) {
            return offset = 958;
        }

        virtual unsigned int offset959(int) {
            return offset = 959;
        }

        virtual unsigned int offset960(int) {
            return offset = 960;
        }

        virtual unsigned int offset961(int) {
            return offset = 961;
        }

        virtual unsigned int offset962(int) {
            return offset = 962;
        }

        virtual unsigned int offset963(int) {
            return offset = 963;
        }

        virtual unsigned int offset964(int) {
            return offset = 964;
        }

        virtual unsigned int offset965(int) {
            return offset = 965;
        }

        virtual unsigned int offset966(int) {
            return offset = 966;
        }

        virtual unsigned int offset967(int) {
            return offset = 967;
        }

        virtual unsigned int offset968(int) {
            return offset = 968;
        }

        virtual unsigned int offset969(int) {
            return offset = 969;
        }

        virtual unsigned int offset970(int) {
            return offset = 970;
        }

        virtual unsigned int offset971(int) {
            return offset = 971;
        }

        virtual unsigned int offset972(int) {
            return offset = 972;
        }

        virtual unsigned int offset973(int) {
            return offset = 973;
        }

        virtual unsigned int offset974(int) {
            return offset = 974;
        }

        virtual unsigned int offset975(int) {
            return offset = 975;
        }

        virtual unsigned int offset976(int) {
            return offset = 976;
        }

        virtual unsigned int offset977(int) {
            return offset = 977;
        }

        virtual unsigned int offset978(int) {
            return offset = 978;
        }

        virtual unsigned int offset979(int) {
            return offset = 979;
        }

        virtual unsigned int offset980(int) {
            return offset = 980;
        }

        virtual unsigned int offset981(int) {
            return offset = 981;
        }

        virtual unsigned int offset982(int) {
            return offset = 982;
        }

        virtual unsigned int offset983(int) {
            return offset = 983;
        }

        virtual unsigned int offset984(int) {
            return offset = 984;
        }

        virtual unsigned int offset985(int) {
            return offset = 985;
        }

        virtual unsigned int offset986(int) {
            return offset = 986;
        }

        virtual unsigned int offset987(int) {
            return offset = 987;
        }

        virtual unsigned int offset988(int) {
            return offset = 988;
        }

        virtual unsigned int offset989(int) {
            return offset = 989;
        }

        virtual unsigned int offset990(int) {
            return offset = 990;
        }

        virtual unsigned int offset991(int) {
            return offset = 991;
        }

        virtual unsigned int offset992(int) {
            return offset = 992;
        }

        virtual unsigned int offset993(int) {
            return offset = 993;
        }

        virtual unsigned int offset994(int) {
            return offset = 994;
        }

        virtual unsigned int offset995(int) {
            return offset = 995;
        }

        virtual unsigned int offset996(int) {
            return offset = 996;
        }

        virtual unsigned int offset997(int) {
            return offset = 997;
        }

        virtual unsigned int offset998(int) {
            return offset = 998;
        }

        virtual unsigned int offset999(int) {
            return offset = 999;
        }

        virtual unsigned int offset1000(int) {
            return offset = 1000;
        }

        virtual ~VirtualOffsetSelector() { }
    };
}
#if defined(__GNUG__) && !defined(__clang__)
#define FAKEIT_NO_DEVIRTUALIZE_ATTR [[gnu::optimize("no-devirtualize")]]
#else
#define FAKEIT_NO_DEVIRTUALIZE_ATTR
#endif

namespace fakeit {

    template<typename TargetType, typename SourceType>
    FAKEIT_NO_DEVIRTUALIZE_ATTR
    TargetType union_cast(SourceType source) {

        union {
            SourceType source;
            TargetType target;
        } u;
        u.source = source;
        return u.target;
    }

}

namespace fakeit {
    class VTUtils {
    public:

#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
        template<typename C, typename R, typename ... arglist>
        static unsigned int getOffset(R (C::*vMethod)(arglist...)) {
            auto sMethod = reinterpret_cast<unsigned int (VirtualOffsetSelector::*)(int)>(vMethod);
            VirtualOffsetSelector offsetSelctor;
            return (offsetSelctor.*sMethod)(0);
        }
#if defined(__GNUG__) && !defined(__clang__) && __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif

        template<typename C>
        FAKEIT_DISARM_UBSAN
        static typename std::enable_if<std::has_virtual_destructor<C>::value, unsigned int>::type
        getDestructorOffset() {
            VirtualOffsetSelector offsetSelctor;
            union_cast<C *>(&offsetSelctor)->~C();
            return offsetSelctor.offset;
        }

        template<typename C>
        static typename std::enable_if<!std::has_virtual_destructor<C>::value, unsigned int>::type
        getDestructorOffset() {
            FAIL("Can't mock the destructor. No virtual destructor was found");
        }

		template<typename C>
		static typename std::enable_if<std::has_virtual_destructor<C>::value, bool>::type
			hasVirtualDestructor() {
			return true;
		}

		template<typename C>
		static typename std::enable_if<!std::has_virtual_destructor<C>::value, bool>::type
			hasVirtualDestructor() {
			return false;
		}

        template<typename C>
        static unsigned int getVTSize() {
            struct Derrived : public C {
                virtual void endOfVt() {
                }
            };

            unsigned int vtSize = getOffset(&Derrived::endOfVt);
            return vtSize;
        }
    };


}
#ifdef _MSC_VER
namespace fakeit {

    typedef unsigned long dword_;

    struct TypeDescriptor {
        TypeDescriptor() :
                ptrToVTable(0), spare(0) {

            int **tiVFTPtr = (int **) (&typeid(void));
            int *i = (int *) tiVFTPtr[0];
			char *type_info_vft_ptr = (char *) i;
            ptrToVTable = type_info_vft_ptr;
        }

		char *ptrToVTable;
        dword_ spare;
        char name[8];
    };

    struct PmdInfo {



        int mdisp;

        int pdisp;
        int vdisp;

        PmdInfo() :
                mdisp(0), pdisp(-1), vdisp(0) {
        }
    };

    struct RTTIBaseClassDescriptor {
        RTTIBaseClassDescriptor() :
                pTypeDescriptor(nullptr), numContainedBases(0), attributes(0) {
        }

        const std::type_info *pTypeDescriptor;
        dword_ numContainedBases;
        struct PmdInfo where;
        dword_ attributes;
    };

    template<typename C, typename... baseclasses>
    struct RTTIClassHierarchyDescriptor {
        RTTIClassHierarchyDescriptor() :
                signature(0),
                attributes(0),
                numBaseClasses(0),
                pBaseClassArray(nullptr) {
            pBaseClassArray = new RTTIBaseClassDescriptor *[1 + sizeof...(baseclasses)];
            addBaseClass < C, baseclasses...>();
        }

        ~RTTIClassHierarchyDescriptor() {
            for (int i = 0; i < 1 + sizeof...(baseclasses); i++) {
                RTTIBaseClassDescriptor *desc = pBaseClassArray[i];
                delete desc;
            }
            delete[] pBaseClassArray;
        }

        dword_ signature;
        dword_ attributes;
        dword_ numBaseClasses;
        RTTIBaseClassDescriptor **pBaseClassArray;

        template<typename BaseType>
        void addBaseClass() {
            static_assert(std::is_base_of<BaseType, C>::value, "C must be a derived class of BaseType");
            RTTIBaseClassDescriptor *desc = new RTTIBaseClassDescriptor();
            desc->pTypeDescriptor = &typeid(BaseType);
            pBaseClassArray[numBaseClasses] = desc;
            for (unsigned int i = 0; i < numBaseClasses; i++) {
                pBaseClassArray[i]->numContainedBases++;
            }
            numBaseClasses++;
        }

        template<typename head, typename Base1, typename... tail>
        void addBaseClass() {
            static_assert(std::is_base_of<Base1, head>::value, "invalid inheritance list");
            addBaseClass<head>();
            addBaseClass<Base1, tail...>();
        }

    };

	template<typename C, typename... baseclasses>
	struct RTTICompleteObjectLocator {
#ifdef _WIN64
		RTTICompleteObjectLocator(const std::type_info &unused) :
			signature(0), offset(0), cdOffset(0),
			typeDescriptorOffset(0), classDescriptorOffset(0)
		{
                    (void)unused;
		}

		dword_ signature;
		dword_ offset;
		dword_ cdOffset;
		dword_ typeDescriptorOffset;
		dword_ classDescriptorOffset;
#else
		RTTICompleteObjectLocator(const std::type_info &info) :
			signature(0), offset(0), cdOffset(0),
			pTypeDescriptor(&info),
			pClassDescriptor(new RTTIClassHierarchyDescriptor<C, baseclasses...>()) {
		}

		~RTTICompleteObjectLocator() {
			delete pClassDescriptor;
		}

		dword_ signature;
		dword_ offset;
		dword_ cdOffset;
		const std::type_info *pTypeDescriptor;
		struct RTTIClassHierarchyDescriptor<C, baseclasses...> *pClassDescriptor;
#endif
	};


    struct VirtualTableBase {

        static VirtualTableBase &getVTable(void *instance) {
            fakeit::VirtualTableBase *vt = (fakeit::VirtualTableBase *) (instance);
            return *vt;
        }

        VirtualTableBase(void **firstMethod) : _firstMethod(firstMethod) { }

        void *getCookie(int index) {
            return _firstMethod[-2 - index];
        }

        void setCookie(int index, void *value) {
            _firstMethod[-2 - index] = value;
        }

        void *getMethod(unsigned int index) const {
            return _firstMethod[index];
        }

        void setMethod(unsigned int index, void *method) {
            _firstMethod[index] = method;
        }

    protected:
        void **_firstMethod;
    };

    template<class C, class... baseclasses>
    struct VirtualTable : public VirtualTableBase {

        class Handle {

            friend struct VirtualTable<C, baseclasses...>;

            void **firstMethod;

            Handle(void **method) : firstMethod(method) { }

        public:

            VirtualTable<C, baseclasses...> &restore() {
                VirtualTable<C, baseclasses...> *vt = (VirtualTable<C, baseclasses...> *) this;
                return *vt;
            }
        };

        static VirtualTable<C, baseclasses...> &getVTable(C &instance) {
            fakeit::VirtualTable<C, baseclasses...> *vt = (fakeit::VirtualTable<C, baseclasses...> *) (&instance);
            return *vt;
        }

        void copyFrom(VirtualTable<C, baseclasses...> &from) {
            unsigned int size = VTUtils::getVTSize<C>();
            for (unsigned int i = 0; i < size; i++) {
                _firstMethod[i] = from.getMethod(i);
            }
            if (VTUtils::hasVirtualDestructor<C>())
                setCookie(dtorCookieIndex, from.getCookie(dtorCookieIndex));
        }

        VirtualTable() : VirtualTable(buildVTArray()) {
        }

        ~VirtualTable() {

        }

        void dispose() {
            _firstMethod--;
            RTTICompleteObjectLocator<C, baseclasses...> *locator = (RTTICompleteObjectLocator<C, baseclasses...> *) _firstMethod[0];
            delete locator;
            _firstMethod -= numOfCookies;
            delete[] _firstMethod;
        }


        unsigned int dtor(int) {
            C *c = (C *) this;
            C &cRef = *c;
            auto vt = VirtualTable<C, baseclasses...>::getVTable(cRef);
            void *dtorPtr = vt.getCookie(dtorCookieIndex);
            void(*method)(C *) = reinterpret_cast<void (*)(C *)>(dtorPtr);
            method(c);
            return 0;
        }

        void setDtor(void *method) {





            void *dtorPtr = union_cast<void *>(&VirtualTable<C, baseclasses...>::dtor);
            unsigned int index = VTUtils::getDestructorOffset<C>();
            _firstMethod[index] = dtorPtr;
            setCookie(dtorCookieIndex, method);
        }

        unsigned int getSize() {
            return VTUtils::getVTSize<C>();
        }

        void initAll(void *value) {
            auto size = getSize();
            for (unsigned int i = 0; i < size; i++) {
                setMethod(i, value);
            }
        }

        Handle createHandle() {
            Handle h(_firstMethod);
            return h;
        }

    private:

        class SimpleType {
        };

        static_assert(sizeof(unsigned int (SimpleType::*)()) == sizeof(unsigned int (C::*)()),
            "Can't mock a type with multiple inheritance or with non-polymorphic base class");
        static const unsigned int numOfCookies = 3;
        static const unsigned int dtorCookieIndex = numOfCookies - 1;

        static void **buildVTArray() {
            int vtSize = VTUtils::getVTSize<C>();
            auto array = new void *[vtSize + numOfCookies + 1]{};
            RTTICompleteObjectLocator<C, baseclasses...> *objectLocator = new RTTICompleteObjectLocator<C, baseclasses...>(
                    typeid(C));
            array += numOfCookies;
            array[0] = objectLocator;
            array++;
            return array;
        }

        VirtualTable(void **firstMethod) : VirtualTableBase(firstMethod) {
        }
    };
}
#else
#ifndef __clang__
#include <type_traits>
#include <tr2/type_traits>

namespace fakeit {
    template<typename ... Type1>
    class has_one_base {
    };

    template<typename Type1, typename Type2, typename ... types>
    class has_one_base<std::tr2::__reflection_typelist<Type1, Type2, types...>> : public std::false_type {
    };

    template<typename Type1>
    class has_one_base<std::tr2::__reflection_typelist<Type1>>
            : public has_one_base<typename std::tr2::direct_bases<Type1>::type> {
    };

    template<>
    class has_one_base<std::tr2::__reflection_typelist<>> : public std::true_type {
    };

    template<typename T>
    class is_simple_inheritance_layout : public has_one_base<typename std::tr2::direct_bases<T>::type> {
    };
}

#endif

namespace fakeit {

    struct VirtualTableBase {

        static VirtualTableBase &getVTable(void *instance) {
            fakeit::VirtualTableBase *vt = (fakeit::VirtualTableBase *) (instance);
            return *vt;
        }

        VirtualTableBase(void **firstMethod) : _firstMethod(firstMethod) { }

        void *getCookie(int index) {
            return _firstMethod[-3 - index];
        }

        void setCookie(int index, void *value) {
            _firstMethod[-3 - index] = value;
        }

        void *getMethod(unsigned int index) const {
            return _firstMethod[index];
        }

        void setMethod(unsigned int index, void *method) {
            _firstMethod[index] = method;
        }

    protected:
        void **_firstMethod;
    };

    template<class C, class ... baseclasses>
    struct VirtualTable : public VirtualTableBase {

#ifndef __clang__
        static_assert(is_simple_inheritance_layout<C>::value, "Can't mock a type with multiple inheritance");
#endif

        class Handle {

            friend struct VirtualTable<C, baseclasses...>;
            void **firstMethod;

            Handle(void **method) :
                    firstMethod(method) {
            }

        public:

            VirtualTable<C, baseclasses...> &restore() {
                VirtualTable<C, baseclasses...> *vt = (VirtualTable<C, baseclasses...> *) this;
                return *vt;
            }
        };

        static VirtualTable<C, baseclasses...> &getVTable(C &instance) {
            fakeit::VirtualTable<C, baseclasses...> *vt = (fakeit::VirtualTable<C, baseclasses...> *) (&instance);
            return *vt;
        }

        void copyFrom(VirtualTable<C, baseclasses...> &from) {
            unsigned int size = VTUtils::getVTSize<C>();

            for (size_t i = 0; i < size; ++i) {
                _firstMethod[i] = from.getMethod(i);
            }
        }

        VirtualTable() :
                VirtualTable(buildVTArray()) {
        }

        void dispose() {
            _firstMethod--;
            _firstMethod--;
            _firstMethod -= numOfCookies;
            delete[] _firstMethod;
        }

        unsigned int dtor(int) {
            C *c = (C *) this;
            C &cRef = *c;
            auto vt = VirtualTable<C, baseclasses...>::getVTable(cRef);
            unsigned int index = VTUtils::getDestructorOffset<C>();
            void *dtorPtr = vt.getMethod(index);
            void(*method)(C *) = union_cast<void (*)(C *)>(dtorPtr);
            method(c);
            return 0;
        }


        void setDtor(void *method) {
            unsigned int index = VTUtils::getDestructorOffset<C>();
            void *dtorPtr = union_cast<void *>(&VirtualTable<C, baseclasses...>::dtor);


            _firstMethod[index] = method;

            _firstMethod[index + 1] = dtorPtr;
        }


        unsigned int getSize() {
            return VTUtils::getVTSize<C>();
        }

        void initAll(void *value) {
            unsigned int size = getSize();
            for (unsigned int i = 0; i < size; i++) {
                setMethod(i, value);
            }
        }

        const std::type_info *getTypeId() {
            return (const std::type_info *) (_firstMethod[-1]);
        }

        Handle createHandle() {
            Handle h(_firstMethod);
            return h;
        }

    private:
        static const unsigned int numOfCookies = 2;

        static void **buildVTArray() {
            int size = VTUtils::getVTSize<C>();
            auto array = new void *[size + 2 + numOfCookies]{};
            array += numOfCookies;
            array++;
            array[0] = const_cast<std::type_info *>(&typeid(C));
            array++;
            return array;
        }

        VirtualTable(void **firstMethod) : VirtualTableBase(firstMethod) {
        }

    };
}
#endif
namespace fakeit {

    template<typename R, typename ... arglist>
    struct MethodInvocationHandler : Destructible {
        virtual R handleMethodInvocation(const typename fakeit::production_arg<arglist>::type... args) = 0;
    };

}
#include <new>


namespace fakeit
{
    namespace details
    {
        template <int instanceAreaSize, typename C, typename... BaseClasses>
        class FakeObjectImpl
        {
        public:
            void initializeDataMembersArea()
            {
                for (size_t i = 0; i < instanceAreaSize; ++i)
                {
                    instanceArea[i] = (char) 0;
                }
            }

        protected:
            VirtualTable<C, BaseClasses...> vtable;
            char instanceArea[instanceAreaSize];
        };

        template <typename C, typename... BaseClasses>
        class FakeObjectImpl<0, C, BaseClasses...>
        {
        public:
            void initializeDataMembersArea()
            {}

        protected:
            VirtualTable<C, BaseClasses...> vtable;
        };
    }

    template <typename C, typename... BaseClasses>
    class FakeObject
        : public details::FakeObjectImpl<sizeof(C) - sizeof(VirtualTable<C, BaseClasses...>), C, BaseClasses...>
    {
        FakeObject(FakeObject const&) = delete;
        FakeObject& operator=(FakeObject const&) = delete;

    public:
        FakeObject()
        {
            this->initializeDataMembersArea();
        }

        ~FakeObject()
        {
            this->vtable.dispose();
        }

        void setMethod(unsigned int index, void* method)
        {
            this->vtable.setMethod(index, method);
        }

        VirtualTable<C, BaseClasses...>& getVirtualTable()
        {
            return this->vtable;
        }

        void setVirtualTable(VirtualTable<C, BaseClasses...>& t)
        {
            this->vtable = t;
        }

        void setDtor(void* dtor)
        {
            this->vtable.setDtor(dtor);
        }
    };
}
#include <functional>

namespace fakeit {

    class Finally {
    private:
        std::function<void()> _finallyClause;

        Finally(const Finally &) = delete;

        Finally &operator=(const Finally &) = delete;

    public:
        explicit Finally(std::function<void()> f) :
                _finallyClause(f) {
        }

        Finally(Finally&& other) {
             _finallyClause.swap(other._finallyClause);
        }

        ~Finally() {
            _finallyClause();
        }
    };
}
namespace fakeit {

    struct MethodProxy {

        MethodProxy(unsigned int id, unsigned int offset, void *vMethod) :
                _id(id),
                _offset(offset),
                _vMethod(vMethod) {
        }

        unsigned int getOffset() const {
            return _offset;
        }

        unsigned int getId() const {
            return _id;
        }

        void *getProxy() const {
            return union_cast<void *>(_vMethod);
        }

    private:
        unsigned int _id;
        unsigned int _offset;
        void *_vMethod;
    };
}
#include <utility>


namespace fakeit {

    struct InvocationHandlerCollection {
        static const unsigned int VtCookieIndex = 0;

        virtual Destructible *getInvocatoinHandlerPtrById(unsigned int index) = 0;

        static InvocationHandlerCollection *getInvocationHandlerCollection(void *instance) {
            VirtualTableBase &vt = VirtualTableBase::getVTable(instance);
            InvocationHandlerCollection *invocationHandlerCollection = (InvocationHandlerCollection *) vt.getCookie(
                    InvocationHandlerCollection::VtCookieIndex);
            return invocationHandlerCollection;
        }

        virtual ~InvocationHandlerCollection() { }
    };


    template<typename R, typename ... arglist>
    class MethodProxyCreator {



    public:

        template<unsigned int id>
        MethodProxy createMethodProxy(unsigned int offset) {
            return MethodProxy(id, offset, union_cast<void *>(&MethodProxyCreator::methodProxyX < id > ));
        }

        template<unsigned int id>
        MethodProxy createMethodProxyStatic(unsigned int offset) {
            return MethodProxy(id, offset, union_cast<void *>(&MethodProxyCreator::methodProxyXStatic < id > ));
        }

    protected:

        R methodProxy(unsigned int id, const typename fakeit::production_arg<arglist>::type... args) {
            InvocationHandlerCollection *invocationHandlerCollection = InvocationHandlerCollection::getInvocationHandlerCollection(
                    this);
            MethodInvocationHandler<R, arglist...> *invocationHandler =
                    (MethodInvocationHandler<R, arglist...> *) invocationHandlerCollection->getInvocatoinHandlerPtrById(
                            id);
            return invocationHandler->handleMethodInvocation(std::forward<const typename fakeit::production_arg<arglist>::type>(args)...);
        }

        template<int id>
        R methodProxyX(arglist ... args) {
            return methodProxy(id, std::forward<const typename fakeit::production_arg<arglist>::type>(args)...);
        }

        static R methodProxyStatic(void* instance, unsigned int id, const typename fakeit::production_arg<arglist>::type... args) {
            InvocationHandlerCollection *invocationHandlerCollection = InvocationHandlerCollection::getInvocationHandlerCollection(
                instance);
            MethodInvocationHandler<R, arglist...> *invocationHandler =
                (MethodInvocationHandler<R, arglist...> *) invocationHandlerCollection->getInvocatoinHandlerPtrById(
                    id);
            return invocationHandler->handleMethodInvocation(std::forward<const typename fakeit::production_arg<arglist>::type>(args)...);
        }

        template<int id>
        static R methodProxyXStatic(void* instance, arglist ... args) {
            return methodProxyStatic(instance, id, std::forward<const typename fakeit::production_arg<arglist>::type>(args)...);
        }
    };
}

namespace fakeit {

    class InvocationHandlers : public InvocationHandlerCollection {
        std::vector<std::shared_ptr<Destructible>> &_methodMocks;
        std::vector<unsigned int> &_offsets;

        unsigned int getOffset(unsigned int id) const
        {
            unsigned int offset = 0;
            for (; offset < _offsets.size(); offset++) {
                if (_offsets[offset] == id) {
                    break;
                }
            }
            return offset;
        }

    public:
        InvocationHandlers(
                std::vector<std::shared_ptr<Destructible>> &methodMocks,
                std::vector<unsigned int> &offsets) :
                _methodMocks(methodMocks), _offsets(offsets) {
			for (std::vector<unsigned int>::iterator it = _offsets.begin(); it != _offsets.end(); ++it)
			{
				*it = std::numeric_limits<int>::max();
			}
        }

        Destructible *getInvocatoinHandlerPtrById(unsigned int id) override {
            unsigned int offset = getOffset(id);
            std::shared_ptr<Destructible> ptr = _methodMocks[offset];
            return ptr.get();
        }

    };

    template<typename C, typename ... baseclasses>
    struct DynamicProxy {

        static_assert(std::is_polymorphic<C>::value, "DynamicProxy requires a polymorphic type");

        DynamicProxy(C &inst) :
                instance(inst),
                originalVtHandle(VirtualTable<C, baseclasses...>::getVTable(instance).createHandle()),
                _methodMocks(VTUtils::getVTSize<C>()),
                _offsets(VTUtils::getVTSize<C>()),
                _invocationHandlers(_methodMocks, _offsets) {
            _cloneVt.copyFrom(originalVtHandle.restore());
            _cloneVt.setCookie(InvocationHandlerCollection::VtCookieIndex, &_invocationHandlers);
            getFake().setVirtualTable(_cloneVt);
        }

        void detach() {
            getFake().setVirtualTable(originalVtHandle.restore());
        }

        ~DynamicProxy() {
            _cloneVt.dispose();
        }

        C &get() {
            return instance;
        }

        void Reset() {
			_methodMocks = {};
            _methodMocks.resize(VTUtils::getVTSize<C>());
            _members = {};
			_offsets = {};
            _offsets.resize(VTUtils::getVTSize<C>());
            _cloneVt.copyFrom(originalVtHandle.restore());
        }

		void Clear()
        {
        }

        template<int id, typename R, typename ... arglist>
        void stubMethod(R(C::*vMethod)(arglist...), MethodInvocationHandler<R, arglist...> *methodInvocationHandler) {
            auto offset = VTUtils::getOffset(vMethod);
            MethodProxyCreator<R, arglist...> creator;
            bind(creator.template createMethodProxy<id + 1>(offset), methodInvocationHandler);
        }

        void stubDtor(MethodInvocationHandler<void> *methodInvocationHandler) {
            auto offset = VTUtils::getDestructorOffset<C>();
            MethodProxyCreator<void> creator;






#ifdef _MSC_VER
            bindDtor(creator.createMethodProxyStatic<0>(offset), methodInvocationHandler);
#else
            bindDtor(creator.createMethodProxy<0>(offset), methodInvocationHandler);
#endif
        }

        template<typename R, typename ... arglist>
        bool isMethodStubbed(R(C::*vMethod)(arglist...)) {
            unsigned int offset = VTUtils::getOffset(vMethod);
            return isBinded(offset);
        }

        bool isDtorStubbed() {
            unsigned int offset = VTUtils::getDestructorOffset<C>();
            return isBinded(offset);
        }

        template<typename R, typename ... arglist>
        Destructible *getMethodMock(R(C::*vMethod)(arglist...)) {
            auto offset = VTUtils::getOffset(vMethod);
            std::shared_ptr<Destructible> ptr = _methodMocks[offset];
            return ptr.get();
        }

        Destructible *getDtorMock() {
            auto offset = VTUtils::getDestructorOffset<C>();
            std::shared_ptr<Destructible> ptr = _methodMocks[offset];
            return ptr.get();
        }

        template<typename DataType, typename ... arglist>
        void stubDataMember(DataType C::*member, const arglist &... initargs) {
            DataType C::*theMember = (DataType C::*) member;
            C &mock = get();
            DataType *memberPtr = &(mock.*theMember);
            _members.push_back(
                    std::shared_ptr<DataMemeberWrapper < DataType, arglist...> >
                    {new DataMemeberWrapper < DataType, arglist...>(memberPtr,
                    initargs...)});
        }

        template<typename DataType>
        void getMethodMocks(std::vector<DataType> &into) const {
            for (std::shared_ptr<Destructible> ptr : _methodMocks) {
                DataType p = dynamic_cast<DataType>(ptr.get());
                if (p) {
                    into.push_back(p);
                }
            }
        }

        VirtualTable<C, baseclasses...> &getOriginalVT() {
            VirtualTable<C, baseclasses...> &vt = originalVtHandle.restore();
            return vt;
        }

        template<typename R, typename ... arglist>
        Finally createRaiiMethodSwapper(R(C::*vMethod)(arglist...)) {
            auto offset = VTUtils::getOffset(vMethod);
            auto fakeMethod = getFake().getVirtualTable().getMethod(offset);
            auto originalMethod = getOriginalVT().getMethod(offset);

            getFake().setMethod(offset, originalMethod);
            return Finally{[&, offset, fakeMethod](){
                getFake().setMethod(offset, fakeMethod);
            }};
        }

    private:

        template<typename DataType, typename ... arglist>
        class DataMemeberWrapper : public Destructible {
        private:
            DataType *dataMember;
        public:
            DataMemeberWrapper(DataType *dataMem, const arglist &... initargs) :
                    dataMember(dataMem) {
                new(dataMember) DataType{initargs ...};
            }

            ~DataMemeberWrapper() override
            {
                dataMember->~DataType();
            }
        };

        static_assert(sizeof(C) == sizeof(FakeObject<C, baseclasses...>), "This is a problem");

        C &instance;
        typename VirtualTable<C, baseclasses...>::Handle originalVtHandle;
        VirtualTable<C, baseclasses...> _cloneVt;

        std::vector<std::shared_ptr<Destructible>> _methodMocks;
        std::vector<std::shared_ptr<Destructible>> _members;
        std::vector<unsigned int> _offsets;
        InvocationHandlers _invocationHandlers;

        FakeObject<C, baseclasses...> &getFake() {
            return reinterpret_cast<FakeObject<C, baseclasses...> &>(instance);
        }

        void bind(const MethodProxy &methodProxy, Destructible *invocationHandler) {
            getFake().setMethod(methodProxy.getOffset(), methodProxy.getProxy());
            _methodMocks[methodProxy.getOffset()].reset(invocationHandler);
            _offsets[methodProxy.getOffset()] = methodProxy.getId();
        }

        void bindDtor(const MethodProxy &methodProxy, Destructible *invocationHandler) {
            getFake().setDtor(methodProxy.getProxy());
            _methodMocks[methodProxy.getOffset()].reset(invocationHandler);
            _offsets[methodProxy.getOffset()] = methodProxy.getId();
        }

        template<typename DataType>
        DataType getMethodMock(unsigned int offset) {
            std::shared_ptr<Destructible> ptr = _methodMocks[offset];
            return dynamic_cast<DataType>(ptr.get());
        }

        template<typename BaseClass>
        void checkMultipleInheritance() {
            C *ptr = (C *) (unsigned int) 1;
            BaseClass *basePtr = ptr;
            int delta = (unsigned long) basePtr - (unsigned long) ptr;
            if (delta > 0) {
                FAIL("multiple inheritance is not supported");
            }
        }

        bool isBinded(unsigned int offset) {
            std::shared_ptr<Destructible> ptr = _methodMocks[offset];
            return ptr.get() != nullptr;
        }

    };
}
#include <functional>
#include <type_traits>
#include <memory>
#include <iosfwd>
#include <vector>
#include <functional>
#include <tuple>
#include <tuple>

namespace fakeit {

    template<int N>
    struct apply_func {
        template<typename R, typename ... ArgsF, typename ... ArgsT, typename ... Args, typename FunctionType>
        static R applyTuple(FunctionType&& f, std::tuple<ArgsT...> &t, Args &... args) {
            return apply_func<N - 1>::template applyTuple<R>(std::forward<FunctionType>(f), t, std::get<N - 1>(t), args...);
        }
    };

    template<>
    struct apply_func < 0 > {
        template<typename R, typename ... ArgsF, typename ... ArgsT, typename ... Args, typename FunctionType>
        static R applyTuple(FunctionType&& f, std::tuple<ArgsT...> & , Args &... args) {
            return std::forward<FunctionType>(f)(args...);
        }
    };

    struct TupleDispatcher {

        template<typename R, typename ... ArgsF, typename ... ArgsT, typename FunctionType>
        static R applyTuple(FunctionType&& f, std::tuple<ArgsT...> &t) {
            return apply_func<sizeof...(ArgsT)>::template applyTuple<R>(std::forward<FunctionType>(f), t);
        }

        template<typename R, typename ...arglist, typename FunctionType>
        static R invoke(FunctionType&& func, const std::tuple<arglist...> &arguments) {
            std::tuple<arglist...> &args = const_cast<std::tuple<arglist...> &>(arguments);
            return applyTuple<R>(std::forward<FunctionType>(func), args);
        }

        template<typename TupleType, typename FunctionType>
        static void for_each(TupleType &&, FunctionType &,
            std::integral_constant<size_t, std::tuple_size<typename std::remove_reference<TupleType>::type>::value>) {
        }

        template<std::size_t I, typename TupleType, typename FunctionType, typename = typename std::enable_if<
            I != std::tuple_size<typename std::remove_reference<TupleType>::type>::value>::type>
            static void for_each(TupleType &&t, FunctionType &f, std::integral_constant<size_t, I>) {
            f(I, std::get < I >(t));
            for_each(std::forward < TupleType >(t), f, std::integral_constant<size_t, I + 1>());
        }

        template<typename TupleType, typename FunctionType>
        static void for_each(TupleType &&t, FunctionType &f) {
            for_each(std::forward < TupleType >(t), f, std::integral_constant<size_t, 0>());
        }

        template<typename TupleType1, typename TupleType2, typename FunctionType>
        static void for_each(TupleType1 &&, TupleType2 &&, FunctionType &,
            std::integral_constant<size_t, std::tuple_size<typename std::remove_reference<TupleType1>::type>::value>) {
        }

        template<std::size_t I, typename TupleType1, typename TupleType2, typename FunctionType, typename = typename std::enable_if<
            I != std::tuple_size<typename std::remove_reference<TupleType1>::type>::value>::type>
            static void for_each(TupleType1 &&t, TupleType2 &&t2, FunctionType &f, std::integral_constant<size_t, I>) {
            f(I, std::get < I >(t), std::get < I >(t2));
            for_each(std::forward < TupleType1 >(t), std::forward < TupleType2 >(t2), f, std::integral_constant<size_t, I + 1>());
        }

        template<typename TupleType1, typename TupleType2, typename FunctionType>
        static void for_each(TupleType1 &&t, TupleType2 &&t2, FunctionType &f) {
            for_each(std::forward < TupleType1 >(t), std::forward < TupleType2 >(t2), f, std::integral_constant<size_t, 0>());
        }
    };
}
namespace fakeit {

    template<typename R, typename ... arglist>
    struct ActualInvocationHandler : Destructible {
        virtual R handleMethodInvocation(ArgumentsTuple<arglist...> & args) = 0;
    };

}
#include <functional>
#include <tuple>
#include <string>
#include <iosfwd>
#include <type_traits>
#include <typeinfo>

namespace fakeit {

    struct DefaultValueInstatiationException {
        virtual ~DefaultValueInstatiationException() = default;

        virtual std::string what() const = 0;
    };


    template<class C>
    struct is_constructible_type {
        static const bool value =
                std::is_default_constructible<typename naked_type<C>::type>::value
                && !std::is_abstract<typename naked_type<C>::type>::value;
    };

    template<class C, class Enable = void>
    struct DefaultValue;

    template<class C>
    struct DefaultValue<C, typename std::enable_if<!is_constructible_type<C>::value>::type> {
        static C &value() {
            if (std::is_reference<C>::value) {
                typename naked_type<C>::type *ptr = nullptr;
                return *ptr;
            }

            FAIL((std::string("Type ") + std::string(typeid(C).name())
                            + std::string(
                            " is not default constructible. Could not instantiate a default return value")));
        }
    };

    template<class C>
    struct DefaultValue<C, typename std::enable_if<is_constructible_type<C>::value>::type> {
        static C &value() {
            static typename naked_type<C>::type val{};
            return val;
        }
    };


    template<>
    struct DefaultValue<void> {
        static void value() {
            return;
        }
    };

    template<>
    struct DefaultValue<bool> {
        static bool &value() {
            static bool value{false};
            return value;
        }
    };

    template<>
    struct DefaultValue<char> {
        static char &value() {
            static char value{0};
            return value;
        }
    };

    template<>
    struct DefaultValue<char16_t> {
        static char16_t &value() {
            static char16_t value{0};
            return value;
        }
    };

    template<>
    struct DefaultValue<char32_t> {
        static char32_t &value() {
            static char32_t value{0};
            return value;
        }
    };

    template<>
    struct DefaultValue<wchar_t> {
        static wchar_t &value() {
            static wchar_t value{0};
            return value;
        }
    };

    template<>
    struct DefaultValue<short> {
        static short &value() {
            static short value{0};
            return value;
        }
    };

    template<>
    struct DefaultValue<int> {
        static int &value() {
            static int value{0};
            return value;
        }
    };

    template<>
    struct DefaultValue<long> {
        static long &value() {
            static long value{0};
            return value;
        }
    };

    template<>
    struct DefaultValue<long long> {
        static long long &value() {
            static long long value{0};
            return value;
        }
    };

    template<>
    struct DefaultValue<std::string> {
        static std::string &value() {
            static std::string value{};
            return value;
        }
    };

}
#include <cmath>
#include <cstring>


namespace fakeit {

    struct IMatcher : Destructible {
        ~IMatcher() = default;
        virtual std::string format() const = 0;
    };

    template<typename ActualT>
    struct TypedMatcher : IMatcher {
        virtual bool matches(const ActualT &actual) const = 0;
    };

    template<typename ExpectedTRef>
    struct ComparisonMatcherCreatorBase {
        using ExpectedT = typename naked_type<ExpectedTRef>::type;

        ExpectedTRef _expectedRef;

        template <typename T>
        ComparisonMatcherCreatorBase(T &&expectedRef)
                : _expectedRef(std::forward<T>(expectedRef)) {
        }

        template <typename ActualT, typename = ExpectedT, typename = void>
        struct MatcherBase : public TypedMatcher<ActualT> {
            const ExpectedT _expected;

            MatcherBase(ExpectedTRef expected)
                    : _expected{std::forward<ExpectedTRef>(expected)} {
            }
        };

        template <typename ActualT, typename U>
        struct MatcherBase<ActualT, U, typename std::enable_if<std::is_same<U, ExpectedT>::value && std::is_array<U>::value>::type> : public TypedMatcher<ActualT> {
            ExpectedT _expected;

            MatcherBase(ExpectedTRef expected) {
                std::memcpy(_expected, expected, sizeof(_expected));
            }
        };
    };

    namespace internal {
        struct AnyMatcherCreator{
            template <typename ActualT>
            struct IsTypeCompatible : std::true_type {};

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public TypedMatcher<ActualT> {
                    bool matches(const ActualT &) const override {
                        return true;
                    }

                    std::string format() const override {
                        return "Any";
                    }
                };

                return new Matcher();
            }
        };

        template<typename ExpectedTRef>
        struct EqMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(std::declval<ActualT>() == std::declval<ExpectedT>())>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return actual == this->_expected;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template<typename ExpectedTRef>
        struct GtMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(std::declval<ActualT>() > std::declval<ExpectedT>())>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string(">") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return actual > this->_expected;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template<typename ExpectedTRef>
        struct GeMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(std::declval<ActualT>() >= std::declval<ExpectedT>())>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string(">=") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return actual >= this->_expected;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template<typename ExpectedTRef>
        struct LtMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(std::declval<ActualT>() < std::declval<ExpectedT>())>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string("<") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return actual < this->_expected;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template<typename ExpectedTRef>
        struct LeMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(std::declval<ActualT>() <= std::declval<ExpectedT>())>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string("<=") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return actual <= this->_expected;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template<typename ExpectedTRef>
        struct NeMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(std::declval<ActualT>() != std::declval<ExpectedT>())>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string("!=") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return actual != this->_expected;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template <typename ExpectedTRef>
        struct StrEqMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(strcmp(std::declval<ActualT>(), std::declval<const char*>()))>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return std::strcmp(actual, this->_expected.c_str()) == 0;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template <typename ExpectedTRef>
        struct StrGtMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(strcmp(std::declval<ActualT>(), std::declval<const char*>()))>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string(">") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return std::strcmp(actual, this->_expected.c_str()) > 0;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template <typename ExpectedTRef>
        struct StrGeMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(strcmp(std::declval<ActualT>(), std::declval<const char*>()))>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string(">=") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return std::strcmp(actual, this->_expected.c_str()) >= 0;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template <typename ExpectedTRef>
        struct StrLtMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(strcmp(std::declval<ActualT>(), std::declval<const char*>()))>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string("<") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return std::strcmp(actual, this->_expected.c_str()) < 0;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template <typename ExpectedTRef>
        struct StrLeMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(strcmp(std::declval<ActualT>(), std::declval<const char*>()))>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string("<=") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return std::strcmp(actual, this->_expected.c_str()) <= 0;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template <typename ExpectedTRef>
        struct StrNeMatcherCreator : public ComparisonMatcherCreatorBase<ExpectedTRef> {
            using ExpectedT = typename ComparisonMatcherCreatorBase<ExpectedTRef>::ExpectedT;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(strcmp(std::declval<ActualT>(), std::declval<const char*>()))>> : std::true_type {};

            using ComparisonMatcherCreatorBase<ExpectedTRef>::ComparisonMatcherCreatorBase;

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT> {
                    using ComparisonMatcherCreatorBase<ExpectedTRef>::template MatcherBase<ActualT>::MatcherBase;

                    virtual std::string format() const override {
                        return std::string("!=") + TypeFormatter<ExpectedT>::format(this->_expected);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return std::strcmp(actual, this->_expected.c_str()) != 0;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef));
            }
        };

        template<typename ExpectedTRef, typename ExpectedMarginTRef>
        struct ApproxEqCreator {
            using ExpectedT = typename naked_type<ExpectedTRef>::type;
            using ExpectedMarginT = typename naked_type<ExpectedMarginTRef>::type;

            template <typename ActualT, typename = void>
            struct IsTypeCompatible : std::false_type {};

            template <typename ActualT>
            struct IsTypeCompatible<ActualT, fk_void_t<decltype(std::abs(std::declval<ActualT>() - std::declval<ExpectedT>()) <= std::declval<ExpectedMarginT>())>> : std::true_type {};

            ExpectedTRef _expectedRef;
            ExpectedMarginTRef _expectedMarginRef;

            template <typename T, typename U>
            ApproxEqCreator(T &&expectedRef, U &&expectedMarginRef)
                    : _expectedRef(std::forward<T>(expectedRef))
                    , _expectedMarginRef(std::forward<U>(expectedMarginRef)) {
            }

            template<typename ActualT>
            TypedMatcher<ActualT> *createMatcher() const {
                struct Matcher : public TypedMatcher<ActualT> {
                    const ExpectedT _expected;
                    const ExpectedMarginT _expectedMargin;

                    Matcher(ExpectedTRef expected, ExpectedMarginTRef expectedMargin)
                            : _expected{std::forward<ExpectedTRef>(expected)}
                            , _expectedMargin{std::forward<ExpectedMarginTRef>(expectedMargin)} {
                    }

                    virtual std::string format() const override {
                        return TypeFormatter<ExpectedT>::format(this->_expected) + std::string("+/-") + TypeFormatter<ExpectedMarginT>::format(this->_expectedMargin);
                    }

                    virtual bool matches(const ActualT &actual) const override {
                        return std::abs(actual - this->_expected) <= this->_expectedMargin;
                    }
                };

                return new Matcher(std::forward<ExpectedTRef>(this->_expectedRef), std::forward<ExpectedMarginTRef>(this->_expectedMarginRef));
            }
        };
    }

/* clang-format off */
    struct AnyMatcher {
    } static _;
/* clang-format on */

    template <typename T>
    internal::AnyMatcherCreator Any() {
        static_assert(sizeof(T) >= 0, "To maintain backward compatibility, this function takes an useless template argument.");
        internal::AnyMatcherCreator mc;
        return mc;
    }

    inline internal::AnyMatcherCreator Any() {
        internal::AnyMatcherCreator mc;
        return mc;
    }

    template<typename T>
    internal::EqMatcherCreator<T&&> Eq(T &&arg) {
        internal::EqMatcherCreator<T&&> mc(std::forward<T>(arg));
        return mc;
    }

    template<typename T>
    internal::GtMatcherCreator<T&&> Gt(T &&arg) {
        internal::GtMatcherCreator<T&&> mc(std::forward<T>(arg));
        return mc;
    }

    template<typename T>
    internal::GeMatcherCreator<T&&> Ge(T &&arg) {
        internal::GeMatcherCreator<T&&> mc(std::forward<T>(arg));
        return mc;
    }

    template<typename T>
    internal::LtMatcherCreator<T&&> Lt(T &&arg) {
        internal::LtMatcherCreator<T&&> mc(std::forward<T>(arg));
        return mc;
    }

    template<typename T>
    internal::LeMatcherCreator<T&&> Le(T &&arg) {
        internal::LeMatcherCreator<T&&> mc(std::forward<T>(arg));
        return mc;
    }

    template<typename T>
    internal::NeMatcherCreator<T&&> Ne(T &&arg) {
        internal::NeMatcherCreator<T&&> mc(std::forward<T>(arg));
        return mc;
    }

    inline internal::StrEqMatcherCreator<std::string&&> StrEq(std::string&& arg) {
        internal::StrEqMatcherCreator<std::string&&> mc(std::move(arg));
        return mc;
    }

    inline internal::StrEqMatcherCreator<const std::string&> StrEq(const std::string& arg) {
        internal::StrEqMatcherCreator<const std::string&> mc(arg);
        return mc;
    }

    inline internal::StrGtMatcherCreator<std::string&&> StrGt(std::string&& arg) {
        internal::StrGtMatcherCreator<std::string&&> mc(std::move(arg));
        return mc;
    }

    inline internal::StrGtMatcherCreator<const std::string&> StrGt(const std::string& arg) {
        internal::StrGtMatcherCreator<const std::string&> mc(arg);
        return mc;
    }

    inline internal::StrGeMatcherCreator<std::string&&> StrGe(std::string&& arg) {
        internal::StrGeMatcherCreator<std::string&&> mc(std::move(arg));
        return mc;
    }

    inline internal::StrGeMatcherCreator<const std::string&> StrGe(const std::string& arg) {
        internal::StrGeMatcherCreator<const std::string&> mc(arg);
        return mc;
    }

    inline internal::StrLtMatcherCreator<std::string&&> StrLt(std::string&& arg) {
        internal::StrLtMatcherCreator<std::string&&> mc(std::move(arg));
        return mc;
    }

    inline internal::StrLtMatcherCreator<const std::string&> StrLt(const std::string& arg) {
        internal::StrLtMatcherCreator<const std::string&> mc(arg);
        return mc;
    }

    inline internal::StrLeMatcherCreator<std::string&&> StrLe(std::string&& arg) {
        internal::StrLeMatcherCreator<std::string&&> mc(std::move(arg));
        return mc;
    }

    inline internal::StrLeMatcherCreator<const std::string&> StrLe(const std::string& arg) {
        internal::StrLeMatcherCreator<const std::string&> mc(arg);
        return mc;
    }

    inline internal::StrNeMatcherCreator<std::string&&> StrNe(std::string&& arg) {
        internal::StrNeMatcherCreator<std::string&&> mc(std::move(arg));
        return mc;
    }

    inline internal::StrNeMatcherCreator<const std::string&> StrNe(const std::string& arg) {
        internal::StrNeMatcherCreator<const std::string&> mc(arg);
        return mc;
    }

    template<typename T, typename U,
        typename std::enable_if<std::is_arithmetic<typename naked_type<T>::type>::value, int>::type = 0,
        typename std::enable_if<std::is_arithmetic<typename naked_type<U>::type>::value, int>::type = 0>
    internal::ApproxEqCreator<T&&, U&&> ApproxEq(T &&expected, U &&margin) {
        internal::ApproxEqCreator<T&&, U&&> mc(std::forward<T>(expected), std::forward<U>(margin));
        return mc;
    }

}

namespace fakeit {

    template<typename ... arglist>
    struct ArgumentsMatcherInvocationMatcher : public ActualInvocation<arglist...>::Matcher {

        virtual ~ArgumentsMatcherInvocationMatcher() {
            for (unsigned int i = 0; i < _matchers.size(); i++)
                delete _matchers[i];
        }

        ArgumentsMatcherInvocationMatcher(const std::vector<Destructible *> &args)
                : _matchers(args) {
        }

        virtual bool matches(ActualInvocation<arglist...> &invocation) override {
            if (invocation.getActualMatcher() == this)
                return true;
            return matches(invocation.getActualArguments());
        }

        virtual std::string format() const override {
            std::ostringstream out;
            out << "(";
            for (unsigned int i = 0; i < _matchers.size(); i++) {
                if (i > 0) out << ", ";
                IMatcher *m = dynamic_cast<IMatcher *>(_matchers[i]);
                out << m->format();
            }
            out << ")";
            return out.str();
        }

    private:

        struct MatchingLambda {
            MatchingLambda(const std::vector<Destructible *> &matchers)
                    : _matchers(matchers) {
            }

            template<typename A>
            void operator()(int index, A &actualArg) {
                TypedMatcher<typename naked_type<A>::type> *matcher =
                        dynamic_cast<TypedMatcher<typename naked_type<A>::type> *>(_matchers[index]);
                if (_matching)
                    _matching = matcher->matches(actualArg);
            }

            bool isMatching() {
                return _matching;
            }

        private:
            bool _matching = true;
            const std::vector<Destructible *> &_matchers;
        };

        virtual bool matches(ArgumentsTuple<arglist...>& actualArguments) {
            MatchingLambda l(_matchers);
            fakeit::TupleDispatcher::for_each(actualArguments, l);
            return l.isMatching();
        }

        const std::vector<Destructible *> _matchers;
    };
































    template<typename ... arglist>
    struct UserDefinedInvocationMatcher : ActualInvocation<arglist...>::Matcher {
        virtual ~UserDefinedInvocationMatcher() = default;

        UserDefinedInvocationMatcher(const std::function<bool(arglist &...)>& match)
                : matcher{match} {
        }

        virtual bool matches(ActualInvocation<arglist...> &invocation) override {
            if (invocation.getActualMatcher() == this)
                return true;
            return matches(invocation.getActualArguments());
        }

        virtual std::string format() const override {
            return {"( user defined matcher )"};
        }

    private:
        virtual bool matches(ArgumentsTuple<arglist...>& actualArguments) {
            return TupleDispatcher::invoke<bool, typename tuple_arg<arglist>::type...>(matcher, actualArguments);
        }

        const std::function<bool(arglist &...)> matcher;
    };

    template<typename ... arglist>
    struct DefaultInvocationMatcher : public ActualInvocation<arglist...>::Matcher {

        virtual ~DefaultInvocationMatcher() = default;

        DefaultInvocationMatcher() {
        }

        virtual bool matches(ActualInvocation<arglist...> &invocation) override {
            return matches(invocation.getActualArguments());
        }

        virtual std::string format() const override {
            return {"( Any arguments )"};
        }

    private:

        virtual bool matches(const ArgumentsTuple<arglist...>&) {
            return true;
        }
    };

}

namespace fakeit {


    template<typename R, typename ... arglist>
    class RecordedMethodBody : public MethodInvocationHandler<R, arglist...>, public ActualInvocationsSource, public ActualInvocationsContainer {

        struct MatchedInvocationHandler : ActualInvocationHandler<R, arglist...> {

            virtual ~MatchedInvocationHandler() = default;

            MatchedInvocationHandler(typename ActualInvocation<arglist...>::Matcher *matcher,
                ActualInvocationHandler<R, arglist...> *invocationHandler) :
                    _matcher{matcher}, _invocationHandler{invocationHandler} {
            }

            virtual R handleMethodInvocation(ArgumentsTuple<arglist...> & args) override
            {
                Destructible &destructable = *_invocationHandler;
                ActualInvocationHandler<R, arglist...> &invocationHandler = dynamic_cast<ActualInvocationHandler<R, arglist...> &>(destructable);
                return invocationHandler.handleMethodInvocation(args);
            }

            typename ActualInvocation<arglist...>::Matcher &getMatcher() const {
                Destructible &destructable = *_matcher;
                typename ActualInvocation<arglist...>::Matcher &matcher = dynamic_cast<typename ActualInvocation<arglist...>::Matcher &>(destructable);
                return matcher;
            }

        private:
            std::shared_ptr<Destructible> _matcher;
            std::shared_ptr<Destructible> _invocationHandler;
        };


        FakeitContext &_fakeit;
        MethodInfo _method;

        std::vector<std::shared_ptr<Destructible>> _invocationHandlers;
        std::vector<std::shared_ptr<Destructible>> _actualInvocations;

        MatchedInvocationHandler *buildMatchedInvocationHandler(
                typename ActualInvocation<arglist...>::Matcher *invocationMatcher,
                ActualInvocationHandler<R, arglist...> *invocationHandler) {
            return new MatchedInvocationHandler(invocationMatcher, invocationHandler);
        }

        MatchedInvocationHandler *getInvocationHandlerForActualArgs(ActualInvocation<arglist...> &invocation) {
            for (auto i = _invocationHandlers.rbegin(); i != _invocationHandlers.rend(); ++i) {
                std::shared_ptr<Destructible> curr = *i;
                Destructible &destructable = *curr;
                MatchedInvocationHandler &im = asMatchedInvocationHandler(destructable);
                if (im.getMatcher().matches(invocation)) {
                    return &im;
                }
            }
            return nullptr;
        }

        MatchedInvocationHandler &asMatchedInvocationHandler(Destructible &destructable) {
            MatchedInvocationHandler &im = dynamic_cast<MatchedInvocationHandler &>(destructable);
            return im;
        }

        ActualInvocation<arglist...> &asActualInvocation(Destructible &destructable) const {
            ActualInvocation<arglist...> &invocation = dynamic_cast<ActualInvocation<arglist...> &>(destructable);
            return invocation;
        }

    public:

        RecordedMethodBody(FakeitContext &fakeit, std::string name) :
                _fakeit(fakeit), _method{MethodInfo::nextMethodOrdinal(), name} { }

        virtual ~RecordedMethodBody() FAKEIT_NO_THROWS {
        }

        MethodInfo &getMethod() {
            return _method;
        }

        bool isOfMethod(MethodInfo &method) {

            return method.id() == _method.id();
        }

        void addMethodInvocationHandler(typename ActualInvocation<arglist...>::Matcher *matcher,
            ActualInvocationHandler<R, arglist...> *invocationHandler) {
            ActualInvocationHandler<R, arglist...> *mock = buildMatchedInvocationHandler(matcher, invocationHandler);
            std::shared_ptr<Destructible> destructable{mock};
            _invocationHandlers.push_back(destructable);
        }

        void reset() {
            _invocationHandlers.clear();
            _actualInvocations.clear();
        }

		void clear() override {
			_actualInvocations.clear();
		}

        R handleMethodInvocation(const typename fakeit::production_arg<arglist>::type... args) override {
            unsigned int ordinal = Invocation::nextInvocationOrdinal();
            MethodInfo &method = this->getMethod();
            auto actualInvocation = new ActualInvocation<arglist...>(ordinal, method, std::forward<const typename fakeit::production_arg<arglist>::type>(args)...);


            std::shared_ptr<Destructible> actualInvocationDtor{actualInvocation};

            auto invocationHandler = getInvocationHandlerForActualArgs(*actualInvocation);
            if (invocationHandler) {
                auto &matcher = invocationHandler->getMatcher();
                actualInvocation->setActualMatcher(&matcher);
                _actualInvocations.push_back(actualInvocationDtor);
                return invocationHandler->handleMethodInvocation(actualInvocation->getActualArguments());
            }

            UnexpectedMethodCallEvent event(UnexpectedType::Unmatched, *actualInvocation);
            _fakeit.handle(event);
            std::string format{_fakeit.format(event)};
            FAIL(format);
            return R();
        }

        void scanActualInvocations(const std::function<void(ActualInvocation<arglist...> &)> &scanner) {
            for (auto destructablePtr : _actualInvocations) {
                ActualInvocation<arglist...> &invocation = asActualInvocation(*destructablePtr);
                scanner(invocation);
            }
        }

        void getActualInvocations(std::unordered_set<Invocation *> &into) const override {
            for (auto destructablePtr : _actualInvocations) {
                Invocation &invocation = asActualInvocation(*destructablePtr);
                into.insert(&invocation);
            }
        }

        void setMethodDetails(const std::string &mockName, const std::string &methodName) {
            const std::string fullName{mockName + "." + methodName};
            _method.setName(fullName);
        }

    };

}
#include <functional>
#include <type_traits>
#include <stdexcept>
#include <utility>
#include <functional>
#include <type_traits>

namespace fakeit {

    struct Quantity {
        Quantity(const int q) :
                quantity(q) {
        }

        const int quantity;
    } static Once(1);

    template<typename R>
    struct Quantifier : public Quantity {
        Quantifier(const int q, const R &val) :
                Quantity(q), value(val) {
        }

        const R &value;
    };

    template<>
    struct Quantifier<void> : public Quantity {
        explicit Quantifier(const int q) :
                Quantity(q) {
        }
    };

    struct QuantifierFunctor : public Quantifier<void> {
        QuantifierFunctor(const int q) :
                Quantifier<void>(q) {
        }

        template<typename R>
        Quantifier<R> operator()(const R &value) {
            return Quantifier<R>(quantity, value);
        }
    };

    template<int q>
    struct Times : public Quantity {

        Times() : Quantity(q) { }

        template<typename R>
        static Quantifier<R> of(const R &value) {
            return Quantifier<R>(q, value);
        }

        static Quantifier<void> Void() {
            return Quantifier<void>(q);
        }
    };

#if defined (__GNUG__) || (_MSC_VER >= 1900)

    inline QuantifierFunctor operator
    ""

    _Times(unsigned long long n) {
        return QuantifierFunctor((int) n);
    }

    inline QuantifierFunctor operator
    ""

    _Time(unsigned long long n) {
        if (n != 1)
            FAIL("Only 1_Time is supported. Use X_Times (with s) if X is bigger than 1");
        return QuantifierFunctor((int) n);
    }

#endif

}
#include <functional>
#include <atomic>
#include <tuple>
#include <type_traits>


namespace fakeit {

    template<typename R, typename ... arglist>
    struct Action : Destructible {
        virtual R invoke(const ArgumentsTuple<arglist...> &) = 0;

        virtual bool isDone() = 0;
    };

    template<typename R, typename ... arglist>
    struct Repeat : Action<R, arglist...> {
        virtual ~Repeat() = default;

        Repeat(std::function<R(typename fakeit::test_arg<arglist>::type...)> func) :
                f(func), times(1) {
        }

        Repeat(std::function<R(typename fakeit::test_arg<arglist>::type...)> func, long t) :
                f(func), times(t) {
        }

        virtual R invoke(const ArgumentsTuple<arglist...> & args) override {
            times--;
            return TupleDispatcher::invoke<R, arglist...>(f, args);
        }

        virtual bool isDone() override {
            return times == 0;
        }

    private:
        std::function<R(typename fakeit::test_arg<arglist>::type...)> f;
        long times;
    };

    template<typename R, typename ... arglist>
    struct RepeatForever : public Action<R, arglist...> {

        virtual ~RepeatForever() = default;

        RepeatForever(std::function<R(typename fakeit::test_arg<arglist>::type...)> func) :
                f(func) {
        }

        virtual R invoke(const ArgumentsTuple<arglist...> & args) override {
            return TupleDispatcher::invoke<R, arglist...>(f, args);
        }

        virtual bool isDone() override {
            return false;
        }

    private:
        std::function<R(typename fakeit::test_arg<arglist>::type...)> f;
    };

    template<typename R, typename ... arglist>
    struct ReturnDefaultValue : public Action<R, arglist...> {
        virtual ~ReturnDefaultValue() = default;

        virtual R invoke(const ArgumentsTuple<arglist...> &) override {
            return DefaultValue<R>::value();
        }

        virtual bool isDone() override {
            return false;
        }
    };

    template<typename R, typename ... arglist>
    struct ReturnDelegateValue : public Action<R, arglist...> {

        ReturnDelegateValue(std::function<R(const typename fakeit::test_arg<arglist>::type...)> delegate) : _delegate(delegate) { }

        virtual ~ReturnDelegateValue() = default;

        virtual R invoke(const ArgumentsTuple<arglist...> & args) override {
            return TupleDispatcher::invoke<R, arglist...>(_delegate, args);
        }

        virtual bool isDone() override {
            return false;
        }

    private:
        std::function<R(const typename fakeit::test_arg<arglist>::type...)> _delegate;
    };

}

namespace fakeit {

    namespace helper
    {
        template <typename T, int N>
        struct ArgValue;

        template <int max_index, int tuple_index>
        struct ArgValidator;

        template<int arg_index, typename current_arg, typename ...T, int ...N, typename ... arglist>
        static void
        Assign(std::tuple<ArgValue<T, N>...> arg_vals, current_arg &&p, arglist &&... args);

        template<int N>
        struct ParamWalker;

    }


    template<typename R, typename ... arglist>
    struct MethodStubbingProgress {

        virtual ~MethodStubbingProgress() FAKEIT_THROWS {
        }

        template<typename U = R>
        typename std::enable_if<!std::is_reference<U>::value, MethodStubbingProgress<R, arglist...> &>::type
        Return(const R &r) {
            return Do([r](const typename fakeit::test_arg<arglist>::type...) -> R { return r; });
        }

        template<typename U = R>
        typename std::enable_if<std::is_reference<U>::value, MethodStubbingProgress<R, arglist...> &>::type
        Return(const R &r) {
            return Do([&r](const typename fakeit::test_arg<arglist>::type...) -> R { return r; });
        }

        template<typename U = R>
        typename std::enable_if<!std::is_copy_constructible<U>::value, MethodStubbingProgress<R, arglist...>&>::type
            Return(R&& r) {
            auto store = std::make_shared<R>(std::move(r));
            return Do([store](const typename fakeit::test_arg<arglist>::type...) mutable -> R {
                return std::move(*store);
            });
        }

        MethodStubbingProgress<R, arglist...> &
        Return(const Quantifier<R> &q) {
            const R &value = q.value;
            auto method = [value](const arglist &...) -> R { return value; };
            return DoImpl(new Repeat<R, arglist...>(method, q.quantity));
        }

        template<typename first, typename second, typename ... tail>
        MethodStubbingProgress<R, arglist...> &
        Return(const first &f, const second &s, const tail &... t) {
            Return(f);
            return Return(s, t...);
        }


        template<typename U = R>
        typename std::enable_if<!std::is_reference<U>::value, void>::type
        AlwaysReturn(const R &r) {
            return AlwaysDo([r](const typename fakeit::test_arg<arglist>::type...) -> R { return r; });
        }

        template<typename U = R>
        typename std::enable_if<std::is_reference<U>::value, void>::type
        AlwaysReturn(const R &r) {
            return AlwaysDo([&r](const typename fakeit::test_arg<arglist>::type...) -> R { return r; });
        }

        MethodStubbingProgress<R, arglist...> &
        Return() {
            return Do([](const typename fakeit::test_arg<arglist>::type...) -> R { return DefaultValue<R>::value(); });
        }

        void AlwaysReturn() {
            return AlwaysDo([](const typename fakeit::test_arg<arglist>::type...) -> R { return DefaultValue<R>::value(); });
        }

        template<typename ... valuelist>
        MethodStubbingProgress<R, arglist...> &
        ReturnAndSet(R &&r, valuelist &&... arg_vals) {
            return Do(GetAssigner(std::forward<R>(r),
                    std::forward<valuelist>(arg_vals)...));
        }

        template<typename ... valuelist>
        void AlwaysReturnAndSet(R &&r, valuelist &&... arg_vals) {
            AlwaysDo(GetAssigner(std::forward<R>(r),
                std::forward<valuelist>(arg_vals)...));
        }

        virtual MethodStubbingProgress<R, arglist...> &
            Do(std::function<R(const typename fakeit::test_arg<arglist>::type...)> method) {
            return DoImpl(new Repeat<R, arglist...>(method));
        }

        template<typename F>
        MethodStubbingProgress<R, arglist...> &
        Do(const Quantifier<F> &q) {
            return DoImpl(new Repeat<R, arglist...>(q.value, q.quantity));
        }

        template<typename first, typename second, typename ... tail>
        MethodStubbingProgress<R, arglist...> &
        Do(const first &f, const second &s, const tail &... t) {
            Do(f);
            return Do(s, t...);
        }

        virtual void AlwaysDo(std::function<R(const typename fakeit::test_arg<arglist>::type...)> method) {
            DoImpl(new RepeatForever<R, arglist...>(method));
        }

    protected:

        virtual MethodStubbingProgress<R, arglist...> &DoImpl(Action<R, arglist...> *action) = 0;

    private:
        MethodStubbingProgress &operator=(const MethodStubbingProgress &other) = delete;

        template<typename ... valuelist>
#if FAKEIT_CPLUSPLUS >= 201402L
        auto
#else
        std::function<R (typename fakeit::test_arg<arglist>::type...)>
#endif
        GetAssigner(R &&r, valuelist &&... arg_vals) {
            class Lambda {
            public:
                Lambda(R &&r, valuelist &&... arg_vals)
                    : vals_tuple{std::forward<R>(r), std::forward<valuelist>(arg_vals)...} {}

                R operator()(typename fakeit::test_arg<arglist>::type... args) {
                    helper::ParamWalker<sizeof...(valuelist)>::Assign(vals_tuple,
                        std::forward<arglist>(args)...);
                    return std::get<0>(vals_tuple);
                }

            private:
                ArgumentsTuple<R, valuelist...> vals_tuple;
            };

            return Lambda(std::forward<R>(r), std::forward<valuelist>(arg_vals)...);
        }

        template<typename ...T, int ...N>
#if FAKEIT_CPLUSPLUS >= 201402L
        auto
#else
        std::function<R (typename fakeit::test_arg<arglist>::type...)>
#endif
        GetAssigner(R &&r, helper::ArgValue<T, N>... arg_vals) {
            class Lambda {
            public:
                Lambda(R &&r, helper::ArgValue<T, N>... arg_vals)
                    : ret{std::forward<R>(r)}
                    , vals_tuple{std::forward<helper::ArgValue<T, N>>(arg_vals)...} {}

                R operator()(typename fakeit::test_arg<arglist>::type... args) {
                    helper::ArgValidator<sizeof...(arglist), sizeof...(T) - 1>::CheckPositions(vals_tuple);
                    helper::Assign<1>(vals_tuple, std::forward<arglist>(args)...);
                    return std::get<0>(ret);
                }

            private:
                std::tuple<R> ret;
                ArgumentsTuple<helper::ArgValue<T, N>...> vals_tuple;
            };

            return Lambda(std::forward<R>(r), std::forward<helper::ArgValue<T, N>>(arg_vals)...);
        }

    };


    template<typename ... arglist>
    struct MethodStubbingProgress<void, arglist...> {

        virtual ~MethodStubbingProgress() FAKEIT_THROWS {
        }

        MethodStubbingProgress<void, arglist...> &Return() {
            auto lambda = [](const typename fakeit::test_arg<arglist>::type...) -> void {
                return DefaultValue<void>::value();
            };
            return Do(lambda);
        }

        virtual MethodStubbingProgress<void, arglist...> &Do(
            std::function<void(const typename fakeit::test_arg<arglist>::type...)> method) {
            return DoImpl(new Repeat<void, arglist...>(method));
        }


        void AlwaysReturn() {
            return AlwaysDo([](const typename fakeit::test_arg<arglist>::type...) -> void { return DefaultValue<void>::value(); });
        }

        MethodStubbingProgress<void, arglist...> &
        Return(const Quantifier<void> &q) {
            auto method = [](const arglist &...) -> void { return DefaultValue<void>::value(); };
            return DoImpl(new Repeat<void, arglist...>(method, q.quantity));
        }

        template<typename ... valuelist>
        MethodStubbingProgress<void, arglist...> &
        ReturnAndSet(valuelist &&... arg_vals) {
            return Do(GetAssigner(std::forward<valuelist>(arg_vals)...));
        }

        template<typename ... valuelist>
        void AlwaysReturnAndSet(valuelist &&... arg_vals) {
            AlwaysDo(GetAssigner(std::forward<valuelist>(arg_vals)...));
        }

        template<typename F>
        MethodStubbingProgress<void, arglist...> &
        Do(const Quantifier<F> &q) {
            return DoImpl(new Repeat<void, arglist...>(q.value, q.quantity));
        }

        template<typename first, typename second, typename ... tail>
        MethodStubbingProgress<void, arglist...> &
        Do(const first &f, const second &s, const tail &... t) {
            Do(f);
            return Do(s, t...);
        }

        virtual void AlwaysDo(std::function<void(const typename fakeit::test_arg<arglist>::type...)> method) {
            DoImpl(new RepeatForever<void, arglist...>(method));
        }

    protected:

        virtual MethodStubbingProgress<void, arglist...> &DoImpl(Action<void, arglist...> *action) = 0;

    private:
        MethodStubbingProgress &operator=(const MethodStubbingProgress &other) = delete;

        template<typename ... valuelist>
#if FAKEIT_CPLUSPLUS >= 201402L
        auto
#else
        std::function<void (typename fakeit::test_arg<arglist>::type...)>
#endif
        GetAssigner(valuelist &&... arg_vals) {
            class Lambda {
            public:
                Lambda(valuelist &&... arg_vals)
                    : vals_tuple{std::forward<valuelist>(arg_vals)...} {}

                void operator()(typename fakeit::test_arg<arglist>::type... args) {
                    helper::ParamWalker<sizeof...(valuelist)>::Assign(vals_tuple,
                        std::forward<arglist>(args)...);
                }

            private:
                ArgumentsTuple<valuelist...> vals_tuple;
            };

            return Lambda(std::forward<valuelist>(arg_vals)...);
        }

        template<typename ...T, int ...N>
#if FAKEIT_CPLUSPLUS >= 201402L
        auto
#else
        std::function<void (typename fakeit::test_arg<arglist>::type...)>
#endif
        GetAssigner(helper::ArgValue<T, N>... arg_vals) {
            class Lambda {
            public:
                Lambda(helper::ArgValue<T, N>... arg_vals)
                    : vals_tuple{std::forward<helper::ArgValue<T, N>>(arg_vals)...} {}

                void operator()(typename fakeit::test_arg<arglist>::type... args) {
                    helper::ArgValidator<sizeof...(arglist), sizeof...(T) - 1>::CheckPositions(vals_tuple);
                    helper::Assign<1>(vals_tuple, std::forward<arglist>(args)...);
                }

            private:
                ArgumentsTuple<helper::ArgValue<T, N>...> vals_tuple;
            };

            return Lambda(std::forward<helper::ArgValue<T, N>>(arg_vals)...);
        }

    };


    namespace helper
    {
        template <typename T, int N>
        struct ArgValue
        {
            ArgValue(T &&v): value ( std::forward<T>(v) ) {}
            constexpr static int pos = N;
            T value;
        };

        template <int max_index, int tuple_index>
        struct ArgValidator
        {
            template <typename ...T, int ...N>
            static void CheckPositions(const std::tuple<ArgValue<T, N>...> arg_vals)
            {
#if FAKEIT_CPLUSPLUS >= 201402L && !defined(_WIN32)
                static_assert(std::get<tuple_index>(arg_vals).pos <= max_index,
                    "Argument index out of range");
                ArgValidator<max_index, tuple_index - 1>::CheckPositions(arg_vals);
#else
                (void)arg_vals;
#endif
            }
        };

        template <int max_index>
        struct ArgValidator<max_index, -1>
        {
            template <typename T>
            static void CheckPositions(T) {}
        };

        template <typename current_arg>
        typename std::enable_if<std::is_pointer<current_arg>::value,
            typename std::remove_pointer<current_arg>::type &>::type
        GetArg(current_arg &&t)
        {
            return *t;
        }

        template <typename current_arg>
        typename std::enable_if<!std::is_pointer<current_arg>::value, current_arg>::type
        GetArg(current_arg &&t)
        {
            return std::forward<current_arg>(t);
        }

        template<int N>
        struct ParamWalker {
            template<typename current_arg, typename ... valuelist, typename ... arglist>
            static void
            Assign(ArgumentsTuple<valuelist...> arg_vals, current_arg &&p, arglist&&... args) {
                ParamWalker<N - 1>::template Assign(arg_vals, std::forward<arglist>(args)...);
                GetArg(std::forward<current_arg>(p)) = std::get<sizeof...(valuelist) - N>(arg_vals);
            }
        };

        template<>
        struct ParamWalker<0> {
            template<typename ... valuelist, typename ... arglist>
            static void Assign(ArgumentsTuple<valuelist...>, arglist... ) {}
        };

        template<int arg_index, int check_index>
        struct ArgLocator {
            template<typename current_arg, typename ...T, int ...N>
            static void AssignArg(current_arg &&p, std::tuple<ArgValue<T, N>...> arg_vals) {
#if FAKEIT_CPLUSPLUS >= 201703L && !defined (_WIN32)
                if constexpr (std::get<check_index>(arg_vals).pos == arg_index)
                    GetArg(std::forward<current_arg>(p)) = std::get<check_index>(arg_vals).value;
#else
                if (std::get<check_index>(arg_vals).pos == arg_index)
                    Set(std::forward<current_arg>(p), std::get<check_index>(arg_vals).value);
#endif
                else if (check_index > 0)
                    ArgLocator<arg_index, check_index - 1>::AssignArg(std::forward<current_arg>(p), arg_vals);
            }

#if FAKEIT_CPLUSPLUS < 201703L || defined (_WIN32)
        private:
            template<typename T, typename U>
            static
            typename std::enable_if<std::is_assignable<decltype(GetArg(std::declval<T>())), U>::value, void>::type
            Set(T &&p, U &&v)
            {
                GetArg(std::forward<T>(p)) = v;
            }

            template<typename T, typename U>
            static
            typename std::enable_if<!std::is_assignable<decltype(GetArg(std::declval<T>())), U>::value, void>::type
            Set(T &&, U &&)
            {
                throw std::logic_error("ReturnAndSet(): Invalid value type");
            }
#endif

        };

        template<int arg_index>
        struct ArgLocator<arg_index, -1> {
            template<typename current_arg, typename T>
            static void AssignArg(current_arg, T) {
            }
        };

        template<int arg_index, typename current_arg, typename ...T, int ...N, typename ... arglist>
        static void
        Assign(std::tuple<ArgValue<T, N>...> arg_vals, current_arg &&p, arglist &&... args) {
            ArgLocator<arg_index, sizeof...(N) - 1>::AssignArg(std::forward<current_arg>(p), arg_vals);
            Assign<arg_index + 1>(arg_vals, std::forward<arglist>(args)...);
        }

        template<int arg_index,  typename ... valuelist>
        static void Assign(std::tuple<valuelist...>) {}

    }


    namespace placeholders
    {
        using namespace std::placeholders;

        template <typename PlaceHolder, typename ArgType,
            typename std::enable_if<static_cast<bool>(std::is_placeholder<PlaceHolder>::value), bool>::type = true>
        helper::ArgValue<ArgType, std::is_placeholder<PlaceHolder>::value>
        operator<=(PlaceHolder, ArgType &&arg)
        {
            return { std::forward<ArgType>(arg) };
        }

    }

    using placeholders::operator <=;
}
#include <vector>



namespace fakeit {


    template<typename R, typename ... arglist>
    struct ActionSequence : ActualInvocationHandler<R,arglist...> {

        ActionSequence() {
            clear();
        }

        void AppendDo(Action<R, arglist...> *action) {
            append(action);
        }

        virtual R handleMethodInvocation(ArgumentsTuple<arglist...> & args) override
        {
            std::shared_ptr<Destructible> destructablePtr = _recordedActions.front();
            Destructible &destructable = *destructablePtr;
            Action<R, arglist...> &action = dynamic_cast<Action<R, arglist...> &>(destructable);
            std::function<void()> finallyClause = [&]() -> void {
                if (action.isDone())
                {
                    _recordedActions.erase(_recordedActions.begin());
                    _usedActions.push_back(destructablePtr);
                }
            };
            Finally onExit(finallyClause);
            return action.invoke(args);
        }

    private:

        struct NoMoreRecordedAction : Action<R, arglist...> {







            virtual R invoke(const ArgumentsTuple<arglist...> &) override {
                FAIL("no more recorded actions");
                return R();
            }

            virtual bool isDone() override {
                return false;
            }
        };

        void append(Action<R, arglist...> *action) {
            std::shared_ptr<Destructible> destructable{action};
            _recordedActions.insert(_recordedActions.end() - 1, destructable);
        }

        void clear() {
            _recordedActions.clear();
            _usedActions.clear();
            auto actionPtr = std::shared_ptr<Destructible> {new NoMoreRecordedAction()};
            _recordedActions.push_back(actionPtr);
        }

        std::vector<std::shared_ptr<Destructible>> _recordedActions;
        std::vector<std::shared_ptr<Destructible>> _usedActions;
    };

}

namespace fakeit {

    template<typename C, typename DataType>
    class DataMemberStubbingRoot {
    private:

    public:
        DataMemberStubbingRoot(const DataMemberStubbingRoot &) = default;

        DataMemberStubbingRoot() = default;

        void operator=(const DataType&) {
        }
    };

}
#include <functional>
#include <utility>
#include <type_traits>
#include <tuple>
#include <memory>
#include <vector>
#include <unordered_set>
#include <set>
#include <iosfwd>

namespace fakeit {

    struct Xaction {
        virtual void commit() = 0;

        virtual ~Xaction() { }
    };
}

namespace fakeit {


    template<typename R, typename ... arglist>
    struct SpyingContext : Xaction {
        virtual void appendAction(Action<R, arglist...> *action) = 0;

        virtual std::function<R(arglist&...)> getOriginalMethodCopyArgs() = 0;
        virtual std::function<R(arglist&...)> getOriginalMethodForwardArgs() = 0;
    };
}
namespace fakeit {


    template<typename R, typename ... arglist>
    struct StubbingContext : public Xaction {
        virtual void appendAction(Action<R, arglist...> *action) = 0;
    };
}
#include <functional>
#include <type_traits>
#include <tuple>
#include <memory>
#include <vector>
#include <unordered_set>


namespace fakeit {

    template<unsigned int index, typename ... arglist>
    class MatchersCollector {

        std::vector<Destructible *> &_matchers;

    public:


        template<std::size_t N>
        using ArgType = typename std::tuple_element<N, std::tuple<arglist...>>::type;

        template<std::size_t N>
        using NakedArgType = typename naked_type<ArgType<index>>::type;

        template <typename MatcherCreatorT, typename = void>
        struct IsMatcherCreatorTypeCompatible : std::false_type {};

        template <typename MatcherCreatorT>
        struct IsMatcherCreatorTypeCompatible<MatcherCreatorT, typename std::enable_if<MatcherCreatorT::template IsTypeCompatible<NakedArgType<index>>::value, void>::type> : std::true_type {};

        MatchersCollector(std::vector<Destructible *> &matchers)
                : _matchers(matchers) {
        }

        void CollectMatchers() {
        }

        template<typename Head>
        typename std::enable_if<
                !std::is_same<AnyMatcher, typename naked_type<Head>::type>::value &&
                !IsMatcherCreatorTypeCompatible<typename naked_type<Head>::type>::value &&
                std::is_constructible<NakedArgType<index>, Head&&>::value, void>
        ::type CollectMatchers(Head &&value) {

            TypedMatcher<NakedArgType<index>> *d = Eq(std::forward<Head>(value)).template createMatcher<NakedArgType<index>>();
            _matchers.push_back(d);
        }

        template<typename Head>
        typename std::enable_if<
                IsMatcherCreatorTypeCompatible<typename naked_type<Head>::type>::value, void>
        ::type CollectMatchers(Head &&creator) {
            TypedMatcher<NakedArgType<index>> *d = creator.template createMatcher<NakedArgType<index>>();
            _matchers.push_back(d);
        }

        template<typename Head>
        typename std::enable_if<
                std::is_same<AnyMatcher, typename naked_type<Head>::type>::value, void>
        ::type CollectMatchers(Head &&) {
            TypedMatcher<NakedArgType<index>> *d = Any().template createMatcher<NakedArgType<index>>();
            _matchers.push_back(d);
        }

        template<typename Head, typename ...Tail>
        void CollectMatchers(Head &&head, Tail &&... tail) {
            CollectMatchers(std::forward<Head>(head));
            MatchersCollector<index + 1, arglist...> c(_matchers);
            c.CollectMatchers(std::forward<Tail>(tail)...);
        }

    };

}

namespace fakeit {

    template<typename R, typename ... arglist>
    class MethodMockingContext :
            public Sequence,
            public ActualInvocationsSource,
            public virtual StubbingContext<R, arglist...>,
            public virtual SpyingContext<R, arglist...>,
            private Invocation::Matcher {
    public:

        struct Context : Destructible {


            virtual typename std::function<R(arglist&...)> getOriginalMethodCopyArgs() = 0;
            virtual typename std::function<R(arglist&...)> getOriginalMethodForwardArgs() = 0;

            virtual std::string getMethodName() = 0;

            virtual void addMethodInvocationHandler(typename ActualInvocation<arglist...>::Matcher *matcher,
                ActualInvocationHandler<R, arglist...> *invocationHandler) = 0;

            virtual void scanActualInvocations(const std::function<void(ActualInvocation<arglist...> &)> &scanner) = 0;

            virtual void setMethodDetails(std::string mockName, std::string methodName) = 0;

            virtual bool isOfMethod(MethodInfo &method) = 0;

            virtual ActualInvocationsSource &getInvolvedMock() = 0;
        };

    private:
        class Implementation {

            Context *_stubbingContext;
            ActionSequence<R, arglist...> *_recordedActionSequence;
            typename ActualInvocation<arglist...>::Matcher *_invocationMatcher;
            bool _commited;

            Context &getStubbingContext() const {
                return *_stubbingContext;
            }

        public:

            Implementation(Context *stubbingContext)
                    : _stubbingContext(stubbingContext),
                      _recordedActionSequence(new ActionSequence<R, arglist...>()),
                      _invocationMatcher
                              {
                                      new DefaultInvocationMatcher<arglist...>()}, _commited(false) {
            }

            ~Implementation() {
                delete _stubbingContext;
                if (!_commited) {

                    delete _recordedActionSequence;
                    delete _invocationMatcher;
                }
            }

            ActionSequence<R, arglist...> &getRecordedActionSequence() {
                return *_recordedActionSequence;
            }

            std::string format() const {
                std::string s = getStubbingContext().getMethodName();
                s += _invocationMatcher->format();
                return s;
            }

            void getActualInvocations(std::unordered_set<Invocation *> &into) const {
                auto scanner = [&](ActualInvocation<arglist...> &a) {
                    if (_invocationMatcher->matches(a)) {
                        into.insert(&a);
                    }
                };
                getStubbingContext().scanActualInvocations(scanner);
            }


            bool matches(Invocation &invocation) {
                MethodInfo &actualMethod = invocation.getMethod();
                if (!getStubbingContext().isOfMethod(actualMethod)) {
                    return false;
                }

                ActualInvocation<arglist...> &actualInvocation = dynamic_cast<ActualInvocation<arglist...> &>(invocation);
                return _invocationMatcher->matches(actualInvocation);
            }

            void commit() {
                getStubbingContext().addMethodInvocationHandler(_invocationMatcher, _recordedActionSequence);
                _commited = true;
            }

            void appendAction(Action<R, arglist...> *action) {
                getRecordedActionSequence().AppendDo(action);
            }

            void setMethodBodyByAssignment(std::function<R(const typename fakeit::test_arg<arglist>::type...)> method) {
                appendAction(new RepeatForever<R, arglist...>(method));
                commit();
            }

            void setMethodDetails(std::string mockName, std::string methodName) {
                getStubbingContext().setMethodDetails(mockName, methodName);
            }

            void getInvolvedMocks(std::vector<ActualInvocationsSource *> &into) const {
                into.push_back(&getStubbingContext().getInvolvedMock());
            }

            typename std::function<R(arglist &...)> getOriginalMethodCopyArgs() {
                return getStubbingContext().getOriginalMethodCopyArgs();
            }

            typename std::function<R(arglist &...)> getOriginalMethodForwardArgs() {
                return getStubbingContext().getOriginalMethodForwardArgs();
            }

            void setInvocationMatcher(typename ActualInvocation<arglist...>::Matcher *matcher) {
                delete _invocationMatcher;
                _invocationMatcher = matcher;
            }
        };

    protected:

        MethodMockingContext(Context *stubbingContext)
                : _impl{new Implementation(stubbingContext)} {
        }

        MethodMockingContext(const MethodMockingContext &) = default;



        MethodMockingContext(MethodMockingContext &&other)
                : _impl(std::move(other._impl)) {
        }

        virtual ~MethodMockingContext() FAKEIT_NO_THROWS { }

        std::string format() const override {
            return _impl->format();
        }

        unsigned int size() const override {
            return 1;
        }


        void getInvolvedMocks(std::vector<ActualInvocationsSource *> &into) const override {
            _impl->getInvolvedMocks(into);
        }

        void getExpectedSequence(std::vector<Invocation::Matcher *> &into) const override {
            const Invocation::Matcher *b = this;
            Invocation::Matcher *c = const_cast<Invocation::Matcher *>(b);
            into.push_back(c);
        }


        void getActualInvocations(std::unordered_set<Invocation *> &into) const override {
            _impl->getActualInvocations(into);
        }


        bool matches(Invocation &invocation) override {
            return _impl->matches(invocation);
        }

        void commit() override {
            _impl->commit();
        }

        void setMethodDetails(std::string mockName, std::string methodName) {
            _impl->setMethodDetails(mockName, methodName);
        }

        void setMatchingCriteria(const std::function<bool(arglist &...)>& predicate) {
            typename ActualInvocation<arglist...>::Matcher *matcher{
                    new UserDefinedInvocationMatcher<arglist...>(predicate)};
            _impl->setInvocationMatcher(matcher);
        }

        void setMatchingCriteria(std::vector<Destructible *> &matchers) {
            typename ActualInvocation<arglist...>::Matcher *matcher{
                    new ArgumentsMatcherInvocationMatcher<arglist...>(matchers)};
            _impl->setInvocationMatcher(matcher);
        }


        void appendAction(Action<R, arglist...> *action) override {
            _impl->appendAction(action);
        }

        void setMethodBodyByAssignment(std::function<R(const typename fakeit::test_arg<arglist>::type...)> method) {
            _impl->setMethodBodyByAssignment(method);
        }

        template<class ...matcherCreators>
        typename std::enable_if<
                sizeof...(matcherCreators) == sizeof...(arglist), void>
        ::type setMatchingCriteria(matcherCreators &&... matcherCreator) {
            std::vector<Destructible *> matchers;

            MatchersCollector<0, arglist...> c(matchers);
            c.CollectMatchers(std::forward<matcherCreators>(matcherCreator)...);

            MethodMockingContext<R, arglist...>::setMatchingCriteria(matchers);
        }

    private:

        typename std::function<R(arglist&...)> getOriginalMethodCopyArgs() override {
            return _impl->getOriginalMethodCopyArgs();
        }

        typename std::function<R(arglist&...)> getOriginalMethodForwardArgs() override {
            return _impl->getOriginalMethodForwardArgs();
        }

        std::shared_ptr<Implementation> _impl;
    };

    template<typename R, typename ... arglist>
    class MockingContext :
            public MethodMockingContext<R, arglist...> {
        MockingContext &operator=(const MockingContext &) = delete;

    public:

        MockingContext(typename MethodMockingContext<R, arglist...>::Context *stubbingContext)
                : MethodMockingContext<R, arglist...>(stubbingContext) {
        }

        MockingContext(const MockingContext &) = default;

        MockingContext(MockingContext &&other)
                : MethodMockingContext<R, arglist...>(std::move(other)) {
        }

        MockingContext<R, arglist...> &setMethodDetails(std::string mockName, std::string methodName) {
            MethodMockingContext<R, arglist...>::setMethodDetails(mockName, methodName);
            return *this;
        }

        template<class ...arg_matcher>
        MockingContext<R, arglist...> &Using(arg_matcher &&... arg_matchers) {
            MethodMockingContext<R, arglist...>::setMatchingCriteria(std::forward<arg_matcher>(arg_matchers)...);
            return *this;
        }

        MockingContext<R, arglist...> &Matching(const std::function<bool(arglist &...)>& matcher) {
            MethodMockingContext<R, arglist...>::setMatchingCriteria(matcher);
            return *this;
        }

        MockingContext<R, arglist...> &operator()(const arglist &... args) {
            MethodMockingContext<R, arglist...>::setMatchingCriteria(args...);
            return *this;
        }

        MockingContext<R, arglist...> &operator()(const std::function<bool(arglist &...)>& matcher) {
            MethodMockingContext<R, arglist...>::setMatchingCriteria(matcher);
            return *this;
        }

        void operator=(std::function<R(arglist &...)> method) {
            MethodMockingContext<R, arglist...>::setMethodBodyByAssignment(method);
        }

        template<typename U = R>
        typename std::enable_if<!std::is_reference<U>::value, void>::type operator=(const R &r) {
            auto method = [r](const typename fakeit::test_arg<arglist>::type...) -> R { return r; };
            MethodMockingContext<R, arglist...>::setMethodBodyByAssignment(method);
        }

        template<typename U = R>
        typename std::enable_if<std::is_reference<U>::value, void>::type operator=(const R &r) {
            auto method = [&r](const typename fakeit::test_arg<arglist>::type...) -> R { return r; };
            MethodMockingContext<R, arglist...>::setMethodBodyByAssignment(method);
        }
    };

    template<typename ... arglist>
    class MockingContext<void, arglist...> :
            public MethodMockingContext<void, arglist...> {
        MockingContext &operator=(const MockingContext &) = delete;

    public:

        MockingContext(typename MethodMockingContext<void, arglist...>::Context *stubbingContext)
                : MethodMockingContext<void, arglist...>(stubbingContext) {
        }

        MockingContext(const MockingContext &) = default;

        MockingContext(MockingContext &&other)
                : MethodMockingContext<void, arglist...>(std::move(other)) {
        }

        MockingContext<void, arglist...> &setMethodDetails(std::string mockName, std::string methodName) {
            MethodMockingContext<void, arglist...>::setMethodDetails(mockName, methodName);
            return *this;
        }

        template<class ...arg_matcher>
        MockingContext<void, arglist...> &Using(arg_matcher &&... arg_matchers) {
            MethodMockingContext<void, arglist...>::setMatchingCriteria(std::forward<arg_matcher>(arg_matchers)...);
            return *this;
        }

        MockingContext<void, arglist...> &Matching(const std::function<bool(arglist &...)>& matcher) {
            MethodMockingContext<void, arglist...>::setMatchingCriteria(matcher);
            return *this;
        }

        MockingContext<void, arglist...> &operator()(const arglist &... args) {
            MethodMockingContext<void, arglist...>::setMatchingCriteria(args...);
            return *this;
        }

        MockingContext<void, arglist...> &operator()(const std::function<bool(arglist &...)>& matcher) {
            MethodMockingContext<void, arglist...>::setMatchingCriteria(matcher);
            return *this;
        }

        void operator=(std::function<void(arglist &...)> method) {
            MethodMockingContext<void, arglist...>::setMethodBodyByAssignment(method);
        }

    };

    class DtorMockingContext : public MethodMockingContext<void> {
    public:

        DtorMockingContext(MethodMockingContext<void>::Context *stubbingContext)
                : MethodMockingContext<void>(stubbingContext) {
        }

        DtorMockingContext(const DtorMockingContext &other) : MethodMockingContext<void>(other) {
        }

        DtorMockingContext(DtorMockingContext &&other) : MethodMockingContext<void>(std::move(other)) {
        }

        void operator=(std::function<void()> method) {
            MethodMockingContext<void>::setMethodBodyByAssignment(method);
        }

        DtorMockingContext &setMethodDetails(std::string mockName, std::string methodName) {
            MethodMockingContext<void>::setMethodDetails(mockName, methodName);
            return *this;
        }
    };

}

namespace fakeit {


    template<typename C, typename ... baseclasses>
    class MockImpl : private MockObject<C>, public virtual ActualInvocationsSource {
    public:

        MockImpl(FakeitContext &fakeit, C &obj)
                : MockImpl<C, baseclasses...>(fakeit, obj, true) {
        }

        MockImpl(FakeitContext &fakeit)
                : MockImpl<C, baseclasses...>(fakeit, *(createFakeInstance()), false){
            FakeObject<C, baseclasses...> *fake = asFakeObject(_instanceOwner.get());
            fake->getVirtualTable().setCookie(1, this);
        }

        virtual ~MockImpl() FAKEIT_NO_THROWS {
            _proxy.detach();
        }


        void getActualInvocations(std::unordered_set<Invocation *> &into) const override {
            std::vector<ActualInvocationsSource *> vec;
            _proxy.getMethodMocks(vec);
            for (ActualInvocationsSource *s : vec) {
                s->getActualInvocations(into);
            }
        }

	    void initDataMembersIfOwner()
	    {
		    if (isOwner()) {
			    FakeObject<C, baseclasses...> *fake = asFakeObject(_instanceOwner.get());
			    fake->initializeDataMembersArea();
		    }
	    }

	    void reset() {
            _proxy.Reset();
		    initDataMembersIfOwner();
	    }

		void clear()
        {
			std::vector<ActualInvocationsContainer *> vec;
			_proxy.getMethodMocks(vec);
			for (ActualInvocationsContainer *s : vec) {
				s->clear();
			}
			initDataMembersIfOwner();
        }

        virtual C &get() override {
            return _proxy.get();
        }

        virtual FakeitContext &getFakeIt() override {
            return _fakeit;
        }

        template<class DataType, typename T, typename ... arglist, class = typename std::enable_if<std::is_base_of<T, C>::value>::type>
        DataMemberStubbingRoot<C, DataType> stubDataMember(DataType T::*member, const arglist &... ctorargs) {
            _proxy.stubDataMember(member, ctorargs...);
            return DataMemberStubbingRoot<T, DataType>();
        }

        template<int id, typename R, typename T, typename ... arglist, class = typename std::enable_if<std::is_base_of<T, C>::value>::type>
        MockingContext<R, arglist...> stubMethod(R(T::*vMethod)(arglist...)) {
            return MockingContext<R, arglist...>(new UniqueMethodMockingContextImpl < id, R, arglist... >
                   (*this, vMethod));
        }

        DtorMockingContext stubDtor() {
            return DtorMockingContext(new DtorMockingContextImpl(*this));
        }







    private:









		std::shared_ptr<FakeObject<C, baseclasses...>> _instanceOwner;
		DynamicProxy<C, baseclasses...> _proxy;
        FakeitContext &_fakeit;

        MockImpl(FakeitContext &fakeit, C &obj, bool isSpy)
                : _instanceOwner(isSpy ? nullptr : asFakeObject(&obj))
				, _proxy{obj}
				, _fakeit(fakeit) {}

        static FakeObject<C, baseclasses...>* asFakeObject(void* instance){
            return reinterpret_cast<FakeObject<C, baseclasses...> *>(instance);
        }

        template<typename R, typename ... arglist>
        class MethodMockingContextBase : public MethodMockingContext<R, arglist...>::Context {
        protected:
            MockImpl<C, baseclasses...> &_mock;

            virtual RecordedMethodBody<R, arglist...> &getRecordedMethodBody() = 0;

        public:
            MethodMockingContextBase(MockImpl<C, baseclasses...> &mock) : _mock(mock) { }

            virtual ~MethodMockingContextBase() = default;

            void addMethodInvocationHandler(typename ActualInvocation<arglist...>::Matcher *matcher,
                ActualInvocationHandler<R, arglist...> *invocationHandler) {
                getRecordedMethodBody().addMethodInvocationHandler(matcher, invocationHandler);
            }

            void scanActualInvocations(const std::function<void(ActualInvocation<arglist...> &)> &scanner) {
                getRecordedMethodBody().scanActualInvocations(scanner);
            }

            void setMethodDetails(std::string mockName, std::string methodName) {
                getRecordedMethodBody().setMethodDetails(mockName, methodName);
            }

            bool isOfMethod(MethodInfo &method) {
                return getRecordedMethodBody().isOfMethod(method);
            }

            ActualInvocationsSource &getInvolvedMock() {
                return _mock;
            }

            std::string getMethodName() {
                return getRecordedMethodBody().getMethod().name();
            }

        };

        template<typename R, typename ... arglist>
        class MethodMockingContextImpl : public MethodMockingContextBase<R, arglist...> {
        protected:

            R (C::*_vMethod)(arglist...);

        public:
            virtual ~MethodMockingContextImpl() = default;

            MethodMockingContextImpl(MockImpl<C, baseclasses...> &mock, R (C::*vMethod)(arglist...))
                    : MethodMockingContextBase<R, arglist...>(mock), _vMethod(vMethod) {
            }

            template<typename ... T, typename std::enable_if<all_true<smart_is_copy_constructible<T>::value...>::value, int>::type = 0>
            std::function<R(arglist&...)> getOriginalMethodCopyArgsInternal(int) {
                auto mPtr = _vMethod;
                auto& mock = MethodMockingContextBase<R, arglist...>::_mock;
                C * instance = &(MethodMockingContextBase<R, arglist...>::_mock.get());
                return [=, &mock](arglist&... args) -> R {
                    auto methodSwapper = mock.createRaiiMethodSwapper(mPtr);
                    return (instance->*mPtr)(args...);
                };
            }


            template<typename ... T>
            [[noreturn]] std::function<R(arglist&...)> getOriginalMethodCopyArgsInternal(long) {
                std::abort();
            }


            std::function<R(arglist&...)> getOriginalMethodCopyArgs() override {
                return getOriginalMethodCopyArgsInternal<arglist...>(0);
            }

            std::function<R(arglist&...)> getOriginalMethodForwardArgs() override {
                auto mPtr = _vMethod;
                auto& mock = MethodMockingContextBase<R, arglist...>::_mock;
                C * instance = &(MethodMockingContextBase<R, arglist...>::_mock.get());
                return [=, &mock](arglist&... args) -> R {
                    auto methodSwapper = mock.createRaiiMethodSwapper(mPtr);
                    return (instance->*mPtr)(std::forward<arglist>(args)...);
                };
            }
        };


        template<int id, typename R, typename ... arglist>
        class UniqueMethodMockingContextImpl : public MethodMockingContextImpl<R, arglist...> {
        protected:

            virtual RecordedMethodBody<R, arglist...> &getRecordedMethodBody() override {
                return MethodMockingContextBase<R, arglist...>::_mock.template stubMethodIfNotStubbed<id>(
                        MethodMockingContextBase<R, arglist...>::_mock._proxy,
                        MethodMockingContextImpl<R, arglist...>::_vMethod);
            }

        public:

            UniqueMethodMockingContextImpl(MockImpl<C, baseclasses...> &mock, R (C::*vMethod)(arglist...))
                    : MethodMockingContextImpl<R, arglist...>(mock, vMethod) {
            }
        };

        class DtorMockingContextImpl : public MethodMockingContextBase<void> {

        protected:

            virtual RecordedMethodBody<void> &getRecordedMethodBody() override {
                return MethodMockingContextBase<void>::_mock.stubDtorIfNotStubbed(
                        MethodMockingContextBase<void>::_mock._proxy);
            }

        public:
            virtual ~DtorMockingContextImpl() = default;

            DtorMockingContextImpl(MockImpl<C, baseclasses...> &mock)
                    : MethodMockingContextBase<void>(mock) {
            }

            std::function<void()> getOriginalMethodCopyArgs() override {
                return [=]() -> void {
                };
            }

            std::function<void()> getOriginalMethodForwardArgs() override {
                return [=]() -> void {
                };
            }

        };

        static MockImpl<C, baseclasses...> *getMockImpl(void *instance) {
            FakeObject<C, baseclasses...> *fake = asFakeObject(instance);
            MockImpl<C, baseclasses...> *mock = reinterpret_cast<MockImpl<C, baseclasses...> *>(fake->getVirtualTable().getCookie(
                    1));
            return mock;
        }

        bool isOwner(){ return _instanceOwner != nullptr;}

		void unmockedDtor() {}

        void unmocked() {
            ActualInvocation<> invocation(Invocation::nextInvocationOrdinal(), UnknownMethod::instance());
            UnexpectedMethodCallEvent event(UnexpectedType::Unmocked, invocation);
            auto &fakeit = getMockImpl(this)->_fakeit;
            fakeit.handle(event);

            std::string format = fakeit.format(event);
            FAIL(format);
        }

        static C *createFakeInstance() {
            FakeObject<C, baseclasses...> *fake = new FakeObject<C, baseclasses...>();
            void *unmockedMethodStubPtr = union_cast<void *>(&MockImpl<C, baseclasses...>::unmocked);
			void *unmockedDtorStubPtr = union_cast<void *>(&MockImpl<C, baseclasses...>::unmockedDtor);
			fake->getVirtualTable().initAll(unmockedMethodStubPtr);
			if (VTUtils::hasVirtualDestructor<C>())
				fake->setDtor(unmockedDtorStubPtr);
			return reinterpret_cast<C *>(fake);
        }

        template<typename R, typename ... arglist>
        Finally createRaiiMethodSwapper(R(C::*vMethod)(arglist...)) {
            return _proxy.createRaiiMethodSwapper(vMethod);
        }

        template<typename R, typename ... arglist>
        void *getOriginalMethod(R (C::*vMethod)(arglist...)) {
            auto vt = _proxy.getOriginalVT();
            auto offset = VTUtils::getOffset(vMethod);
            void *origMethodPtr = vt.getMethod(offset);
            return origMethodPtr;
        }

        void *getOriginalDtor() {
            auto vt = _proxy.getOriginalVT();
            auto offset = VTUtils::getDestructorOffset<C>();
            void *origMethodPtr = vt.getMethod(offset);
            return origMethodPtr;
        }

        template<unsigned int id, typename R, typename ... arglist>
        RecordedMethodBody<R, arglist...> &stubMethodIfNotStubbed(DynamicProxy<C, baseclasses...> &proxy,
                                                                  R (C::*vMethod)(arglist...)) {
            if (!proxy.isMethodStubbed(vMethod)) {
                proxy.template stubMethod<id>(vMethod, createRecordedMethodBody < R, arglist... > (*this, vMethod));
            }
            Destructible *d = proxy.getMethodMock(vMethod);
            RecordedMethodBody<R, arglist...> *methodMock = dynamic_cast<RecordedMethodBody<R, arglist...> *>(d);
            return *methodMock;
        }

        RecordedMethodBody<void> &stubDtorIfNotStubbed(DynamicProxy<C, baseclasses...> &proxy) {
            if (!proxy.isDtorStubbed()) {
                proxy.stubDtor(createRecordedDtorBody(*this));
            }
            Destructible *d = proxy.getDtorMock();
            RecordedMethodBody<void> *dtorMock = dynamic_cast<RecordedMethodBody<void> *>(d);
            return *dtorMock;
        }

        template<typename R, typename ... arglist>
        static RecordedMethodBody<R, arglist...> *createRecordedMethodBody(MockObject<C> &mock,
                                                                           R(C::*vMethod)(arglist...)) {
            return new RecordedMethodBody<R, arglist...>(mock.getFakeIt(), typeid(vMethod).name());
        }

        static RecordedMethodBody<void> *createRecordedDtorBody(MockObject<C> &mock) {
            return new RecordedMethodBody<void>(mock.getFakeIt(), "dtor");
        }
    };
}
namespace fakeit {

    template<typename R, typename... Args>
    struct Prototype;

    template<typename R, typename... Args>
    struct Prototype<R(Args...)> {

        template<class C>
        struct MemberType {

            using Type = R (C::*)(Args...);
            using ConstType = R (C::*)(Args...) const;
            using RefType = R (C::*)(Args...) &;
            using ConstRefType = R (C::*)(Args...) const&;
            using RValRefType = R (C::*)(Args...) &&;
            using ConstRValRefType = R (C::*)(Args...) const&&;

            static Type get(Type t) {
                return t;
            }

            static ConstType getConst(ConstType t) {
                return t;
            }

            static RefType getRef(RefType t) {
                return t;
            }

            static ConstRefType getConstRef(ConstRefType t) {
                return t;
            }

            static RValRefType getRValRef(RValRefType t) {
                return t;
            }

            static ConstRValRefType getConstRValRef(ConstRValRefType t) {
                return t;
            }

        };

    };

    template<int X, typename R, typename C, typename... arglist>
    struct UniqueMethod {
        R (C::*method)(arglist...);

        UniqueMethod(R (C::*vMethod)(arglist...)) : method(vMethod) { }

        int uniqueId() {
            return X;
        }




    };

}

namespace fakeit {
    namespace internal {
        template<typename T, typename = void>
        struct WithCommonVoid {
            using type = T;
        };





        template<typename T>
        struct WithCommonVoid<T, typename std::enable_if<std::is_void<T>::value, void>::type> {
            using type = void;
        };

        template<typename T>
        using WithCommonVoid_t = typename WithCommonVoid<T>::type;
    }

    template<typename C, typename ... baseclasses>
    class Mock : public ActualInvocationsSource {
        MockImpl<C, baseclasses...> impl;
    public:
        virtual ~Mock() = default;

        static_assert(std::is_polymorphic<C>::value, "Can only mock a polymorphic type");

        Mock() : impl(Fakeit) {
        }

        explicit Mock(C &obj) : impl(Fakeit, obj) {
        }

        virtual C &get() {
            return impl.get();
        }





		C &operator()() {
            return get();
        }

        void Reset() {
            impl.reset();
        }

		void ClearInvocationHistory() {
			impl.clear();
		}

        template<class DataType, typename ... arglist,
                class = typename std::enable_if<std::is_member_object_pointer<DataType C::*>::value>::type>
        DataMemberStubbingRoot<C, DataType> Stub(DataType C::* member, const arglist &... ctorargs) {
            return impl.stubDataMember(member, ctorargs...);
        }


        template<int id, typename R, typename T, typename ... arglist, class = typename std::enable_if<
                std::is_base_of<T, C>::value>::type>
        MockingContext<internal::WithCommonVoid_t<R>, arglist...> stub(R (T::*vMethod)(arglist...) const) {
            auto methodWithoutConstVolatile = reinterpret_cast<internal::WithCommonVoid_t<R> (T::*)(arglist...)>(vMethod);
            return impl.template stubMethod<id>(methodWithoutConstVolatile);
        }


        template<int id, typename R, typename T, typename... arglist, class = typename std::enable_if<
                std::is_base_of<T, C>::value>::type>
        MockingContext<internal::WithCommonVoid_t<R>, arglist...> stub(R(T::*vMethod)(arglist...) volatile) {
            auto methodWithoutConstVolatile = reinterpret_cast<internal::WithCommonVoid_t<R>(T::*)(arglist...)>(vMethod);
            return impl.template stubMethod<id>(methodWithoutConstVolatile);
        }


        template<int id, typename R, typename T, typename... arglist, class = typename std::enable_if<
                std::is_base_of<T, C>::value>::type>
        MockingContext<internal::WithCommonVoid_t<R>, arglist...> stub(R(T::*vMethod)(arglist...) const volatile) {
            auto methodWithoutConstVolatile = reinterpret_cast<internal::WithCommonVoid_t<R>(T::*)(arglist...)>(vMethod);
            return impl.template stubMethod<id>(methodWithoutConstVolatile);
        }


        template<int id, typename R, typename T, typename... arglist, class = typename std::enable_if<
                std::is_base_of<T, C>::value>::type>
        MockingContext<internal::WithCommonVoid_t<R>, arglist...> stub(R(T::*vMethod)(arglist...)) {
            auto methodWithoutConstVolatile = reinterpret_cast<internal::WithCommonVoid_t<R>(T::*)(arglist...)>(vMethod);
            return impl.template stubMethod<id>(methodWithoutConstVolatile);
        }


        template<int id, typename R, typename T, typename... arglist, class = typename std::enable_if<
                std::is_base_of<T, C>::value>::type>
        MockingContext<internal::WithCommonVoid_t<R>, arglist...> stub(R(T::*vMethod)(arglist...) &) {
            auto methodWithoutConstVolatile = reinterpret_cast<internal::WithCommonVoid_t<R>(T::*)(arglist...)>(vMethod);
            return impl.template stubMethod<id>(methodWithoutConstVolatile);
        }


        template<int id, typename R, typename T, typename... arglist, class = typename std::enable_if<
                std::is_base_of<T, C>::value>::type>
        MockingContext<internal::WithCommonVoid_t<R>, arglist...> stub(R(T::*vMethod)(arglist...) const&) {
            auto methodWithoutConstVolatile = reinterpret_cast<internal::WithCommonVoid_t<R>(T::*)(arglist...)>(vMethod);
            return impl.template stubMethod<id>(methodWithoutConstVolatile);
        }


        template<int id, typename R, typename T, typename... arglist, class = typename std::enable_if<
                std::is_base_of<T, C>::value>::type>
        MockingContext<internal::WithCommonVoid_t<R>, arglist...> stub(R(T::*vMethod)(arglist...) &&) {
            auto methodWithoutConstVolatile = reinterpret_cast<internal::WithCommonVoid_t<R>(T::*)(arglist...)>(vMethod);
            return impl.template stubMethod<id>(methodWithoutConstVolatile);
        }


        template<int id, typename R, typename T, typename... arglist, class = typename std::enable_if<
                std::is_base_of<T, C>::value>::type>
        MockingContext<internal::WithCommonVoid_t<R>, arglist...> stub(R(T::*vMethod)(arglist...) const&&) {
            auto methodWithoutConstVolatile = reinterpret_cast<internal::WithCommonVoid_t<R>(T::*)(arglist...)>(vMethod);
            return impl.template stubMethod<id>(methodWithoutConstVolatile);
        }

        DtorMockingContext dtor() {
            return impl.stubDtor();
        }

        void getActualInvocations(std::unordered_set<Invocation *> &into) const override {
            impl.getActualInvocations(into);
        }

    };

}

#include <exception>

namespace fakeit {

    class RefCount {
    private:
        int count;

    public:
        void AddRef() {
            count++;
        }

        int Release() {
            return --count;
        }
    };

    template<typename T>
    class smart_ptr {
    private:
        T *pData;
        RefCount *reference;

    public:
        smart_ptr() : pData(0), reference(0) {
            reference = new RefCount();
            reference->AddRef();
        }

        smart_ptr(T *pValue) : pData(pValue), reference(0) {
            reference = new RefCount();
            reference->AddRef();
        }

        smart_ptr(const smart_ptr<T> &sp) : pData(sp.pData), reference(sp.reference) {
            reference->AddRef();
        }

        ~smart_ptr() FAKEIT_THROWS {
            if (reference->Release() == 0) {
                delete reference;
                delete pData;
            }
        }

        T &operator*() {
            return *pData;
        }

        T *operator->() {
            return pData;
        }

        smart_ptr<T> &operator=(const smart_ptr<T> &sp) {
            if (this != &sp) {


                if (reference->Release() == 0) {
                    delete reference;
                    delete pData;
                }



                pData = sp.pData;
                reference = sp.reference;
                reference->AddRef();
            }
            return *this;
        }
    };

}

namespace fakeit {

    class WhenFunctor {

        struct StubbingChange {

            friend class WhenFunctor;

            virtual ~StubbingChange() FAKEIT_THROWS {

                if (UncaughtException()) {
                    return;
                }

                _xaction.commit();
            }

            StubbingChange(const StubbingChange &other) :
                    _xaction(other._xaction) {
            }

        private:

            StubbingChange(Xaction &xaction)
                    : _xaction(xaction) {
            }

            Xaction &_xaction;
        };

    public:

        template<typename R, typename ... arglist>
        struct MethodProgress : MethodStubbingProgress<R, arglist...> {

            friend class WhenFunctor;

            virtual ~MethodProgress() override = default;

            MethodProgress(const MethodProgress &other) :
                    _progress(other._progress), _context(other._context) {
            }

            MethodProgress(StubbingContext<R, arglist...> &xaction) :
                    _progress(new StubbingChange(xaction)), _context(xaction) {
            }

        protected:

            virtual MethodStubbingProgress<R, arglist...> &DoImpl(Action<R, arglist...> *action) override {
                _context.appendAction(action);
                return *this;
            }

        private:
            smart_ptr<StubbingChange> _progress;
            StubbingContext<R, arglist...> &_context;
        };


        WhenFunctor() {
        }

        template<typename R, typename ... arglist>
        MethodProgress<R, arglist...> operator()(const StubbingContext<R, arglist...> &stubbingContext) {
            StubbingContext<R, arglist...> &rootWithoutConst = const_cast<StubbingContext<R, arglist...> &>(stubbingContext);
            MethodProgress<R, arglist...> progress(rootWithoutConst);
            return progress;
        }

    };

}
namespace fakeit {

    class FakeFunctor {
    private:
        template<typename R, typename ... arglist>
        void fake(const StubbingContext<R, arglist...> &root) {
            StubbingContext<R, arglist...> &rootWithoutConst = const_cast<StubbingContext<R, arglist...> &>(root);
            rootWithoutConst.appendAction(new ReturnDefaultValue<R, arglist...>());
            rootWithoutConst.commit();
        }

        void operator()() {
        }

    public:

        template<typename H, typename ... M>
        void operator()(const H &head, const M &... tail) {
            fake(head);
            this->operator()(tail...);
        }

    };

}
#include <set>
#include <set>


namespace fakeit {

    struct InvocationUtils {

        static void sortByInvocationOrder(std::unordered_set<Invocation *> &ivocations,
                                          std::vector<Invocation *> &result) {
            auto comparator = [](Invocation *a, Invocation *b) -> bool {
                return a->getOrdinal() < b->getOrdinal();
            };
            std::set<Invocation *, bool (*)(Invocation *a, Invocation *b)> sortedIvocations(comparator);
            for (auto i : ivocations)
                sortedIvocations.insert(i);

            for (auto i : sortedIvocations)
                result.push_back(i);
        }

        static void collectActualInvocations(std::unordered_set<Invocation *> &actualInvocations,
                                             std::vector<ActualInvocationsSource *> &invocationSources) {
            for (auto source : invocationSources) {
                source->getActualInvocations(actualInvocations);
            }
        }

        static void selectNonVerifiedInvocations(std::unordered_set<Invocation *> &actualInvocations,
                                                 std::unordered_set<Invocation *> &into) {
            for (auto invocation : actualInvocations) {
                if (!invocation->isVerified()) {
                    into.insert(invocation);
                }
            }
        }

        static void collectInvocationSources(std::vector<ActualInvocationsSource *> &) {
        }

        template<typename ... list>
        static void collectInvocationSources(std::vector<ActualInvocationsSource *> &into,
                                             const ActualInvocationsSource &mock,
                                             const list &... tail) {
            into.push_back(const_cast<ActualInvocationsSource *>(&mock));
            collectInvocationSources(into, tail...);
        }

        static void collectSequences(std::vector<Sequence *> &) {
        }

        template<typename ... list>
        static void collectSequences(std::vector<Sequence *> &vec, const Sequence &sequence, const list &... tail) {
            vec.push_back(&const_cast<Sequence &>(sequence));
            collectSequences(vec, tail...);
        }

        static void collectInvolvedMocks(std::vector<Sequence *> &allSequences,
                                         std::vector<ActualInvocationsSource *> &involvedMocks) {
            for (auto sequence : allSequences) {
                sequence->getInvolvedMocks(involvedMocks);
            }
        }

        template<class T>
        static T &remove_const(const T &s) {
            return const_cast<T &>(s);
        }

    };

}

#include <memory>

#include <vector>
#include <unordered_set>

namespace fakeit {
    struct MatchAnalysis {
        std::vector<Invocation *> actualSequence;
        std::vector<Invocation *> matchedInvocations;
        int count;

        void run(InvocationsSourceProxy &involvedInvocationSources, std::vector<Sequence *> &expectedPattern) {
            getActualInvocationSequence(involvedInvocationSources, actualSequence);
            count = countMatches(expectedPattern, actualSequence, matchedInvocations);
        }

    private:
        static void getActualInvocationSequence(InvocationsSourceProxy &involvedMocks,
                                                std::vector<Invocation *> &actualSequence) {
            std::unordered_set<Invocation *> actualInvocations;
            collectActualInvocations(involvedMocks, actualInvocations);
            InvocationUtils::sortByInvocationOrder(actualInvocations, actualSequence);
        }

        static int countMatches(std::vector<Sequence *> &pattern, std::vector<Invocation *> &actualSequence,
                                std::vector<Invocation *> &matchedInvocations) {
            int end = -1;
            int count = 0;
            int startSearchIndex = 0;
            while (findNextMatch(pattern, actualSequence, startSearchIndex, end, matchedInvocations)) {
                count++;
                startSearchIndex = end;
            }
            return count;
        }

        static void collectActualInvocations(InvocationsSourceProxy &involvedMocks,
                                             std::unordered_set<Invocation *> &actualInvocations) {
            involvedMocks.getActualInvocations(actualInvocations);
        }

        static bool findNextMatch(std::vector<Sequence *> &pattern, std::vector<Invocation *> &actualSequence,
                                  int startSearchIndex, int &end,
                                  std::vector<Invocation *> &matchedInvocations) {
            for (auto sequence : pattern) {
                int index = findNextMatch(sequence, actualSequence, startSearchIndex);
                if (index == -1) {
                    return false;
                }
                collectMatchedInvocations(actualSequence, matchedInvocations, index, sequence->size());
                startSearchIndex = index + sequence->size();
            }
            end = startSearchIndex;
            return true;
        }


        static void collectMatchedInvocations(std::vector<Invocation *> &actualSequence,
                                              std::vector<Invocation *> &matchedInvocations, int start,
                                              int length) {
            int indexAfterMatchedPattern = start + length;
            for (; start < indexAfterMatchedPattern; start++) {
                matchedInvocations.push_back(actualSequence[start]);
            }
        }


        static bool isMatch(std::vector<Invocation *> &actualSequence,
                            std::vector<Invocation::Matcher *> &expectedSequence, int start) {
            bool found = true;
            for (unsigned int j = 0; found && j < expectedSequence.size(); j++) {
                Invocation *actual = actualSequence[start + j];
                Invocation::Matcher *expected = expectedSequence[j];
                found = found && expected->matches(*actual);
            }
            return found;
        }

        static int findNextMatch(Sequence *&pattern, std::vector<Invocation *> &actualSequence, int startSearchIndex) {
            std::vector<Invocation::Matcher *> expectedSequence;
            pattern->getExpectedSequence(expectedSequence);
            for (int i = startSearchIndex; i < ((int) actualSequence.size() - (int) expectedSequence.size() + 1); i++) {
                if (isMatch(actualSequence, expectedSequence, i)) {
                    return i;
                }
            }
            return -1;
        }

    };
}

namespace fakeit {

    struct SequenceVerificationExpectation {

        friend class SequenceVerificationProgress;

        ~SequenceVerificationExpectation() FAKEIT_THROWS {
            if (UncaughtException()) {
                return;
            }
            VerifyExpectation(_fakeit);
        }

        void setExpectedPattern(std::vector<Sequence *> expectedPattern) {
            _expectedPattern = expectedPattern;
        }

        void setExpectedCount(const int count) {
            _expectedCount = count;
        }

        void expectAnything() {
            _expectAnything = true;
        }

        void setFileInfo(const char * file, int line, const char * callingMethod) {
            _file = file;
            _line = line;
            _testMethod = callingMethod;
        }

    private:

        VerificationEventHandler &_fakeit;
        InvocationsSourceProxy _involvedInvocationSources;
        std::vector<Sequence *> _expectedPattern;
        int _expectedCount;
        bool _expectAnything;

        const char * _file;
        int _line;
		const char * _testMethod;
        bool _isVerified;

        SequenceVerificationExpectation(
                VerificationEventHandler &fakeit,
                InvocationsSourceProxy mocks,
                std::vector<Sequence *> &expectedPattern) :
                _fakeit(fakeit),
                _involvedInvocationSources(mocks),
                _expectedPattern(expectedPattern),
                _expectedCount(-1),
                _expectAnything(false),
                _line(0),
                _isVerified(false) {
        }


        void VerifyExpectation(VerificationEventHandler &verificationErrorHandler) {
            if (_isVerified)
                return;
            _isVerified = true;

            MatchAnalysis ma;
            ma.run(_involvedInvocationSources, _expectedPattern);

            if (isNotAnythingVerification()) {
                if (isAtLeastVerification() && atLeastLimitNotReached(ma.count)) {
                    return handleAtLeastVerificationEvent(verificationErrorHandler, ma.actualSequence, ma.count);
                }

                if (isExactVerification() && exactLimitNotMatched(ma.count)) {
                    return handleExactVerificationEvent(verificationErrorHandler, ma.actualSequence, ma.count);
                }
            }

            markAsVerified(ma.matchedInvocations);
        }

        std::vector<Sequence *> &collectSequences(std::vector<Sequence *> &vec) {
            return vec;
        }

        template<typename ... list>
        std::vector<Sequence *> &collectSequences(std::vector<Sequence *> &vec, const Sequence &sequence,
                                                  const list &... tail) {
            vec.push_back(&const_cast<Sequence &>(sequence));
            return collectSequences(vec, tail...);
        }


        static void markAsVerified(std::vector<Invocation *> &matchedInvocations) {
            for (auto i : matchedInvocations) {
                i->markAsVerified();
            }
        }

        bool isNotAnythingVerification() {
            return !_expectAnything;
        }

        bool isAtLeastVerification() {

            return _expectedCount < 0;
        }

        bool isExactVerification() {
            return !isAtLeastVerification();
        }

        bool atLeastLimitNotReached(int actualCount) {
            return actualCount < -_expectedCount;
        }

        bool exactLimitNotMatched(int actualCount) {
            return actualCount != _expectedCount;
        }

        void handleExactVerificationEvent(VerificationEventHandler &verificationErrorHandler,
                                          std::vector<Invocation *> actualSequence, int count) {
            SequenceVerificationEvent evt(VerificationType::Exact, _expectedPattern, actualSequence, _expectedCount,
                                          count);
            evt.setFileInfo(_file, _line, _testMethod);
            return verificationErrorHandler.handle(evt);
        }

        void handleAtLeastVerificationEvent(VerificationEventHandler &verificationErrorHandler,
                                            std::vector<Invocation *> actualSequence, int count) {
            SequenceVerificationEvent evt(VerificationType::AtLeast, _expectedPattern, actualSequence, -_expectedCount,
                                          count);
            evt.setFileInfo(_file, _line, _testMethod);
            return verificationErrorHandler.handle(evt);
        }

    };

}
namespace fakeit {
    class FalseEventHandler : public VerificationEventHandler {

        void handle(const SequenceVerificationEvent &) override {
            handle_result = false;
        }

        void handle(const NoMoreInvocationsVerificationEvent &) override {
            handle_result = false;
        }

    public:
        bool handle_result = true;
    };
}


namespace fakeit {

    struct FakeitContext;

    class SequenceVerificationProgress {

        friend class UsingFunctor;

        friend class VerifyFunctor;

        friend class UsingProgress;

        smart_ptr<SequenceVerificationExpectation> _expectationPtr;

        SequenceVerificationProgress(SequenceVerificationExpectation *ptr) : _expectationPtr(ptr) {
        }

        SequenceVerificationProgress(
                FakeitContext &fakeit,
                InvocationsSourceProxy sources,
                std::vector<Sequence *> &allSequences) :
                SequenceVerificationProgress(new SequenceVerificationExpectation(fakeit, sources, allSequences)) {
        }

        virtual void verifyInvocations(const int times) {
            _expectationPtr->setExpectedCount(times);
        }

        class Terminator {
            smart_ptr<SequenceVerificationExpectation> _expectationPtr;

            bool toBool() {
                FalseEventHandler eh;
                _expectationPtr->VerifyExpectation(eh);
                return eh.handle_result;
            }

        public:
            Terminator(smart_ptr<SequenceVerificationExpectation> expectationPtr) : _expectationPtr(expectationPtr) { };

            operator bool() {
                return toBool();
            }

            bool operator!() const { return !const_cast<Terminator *>(this)->toBool(); }
        };

    public:

        virtual ~SequenceVerificationProgress() FAKEIT_THROWS { };

        operator bool() const {
            return Terminator(_expectationPtr);
        }

        bool operator!() const { return !Terminator(_expectationPtr); }

        Terminator Any() {
            _expectationPtr->expectAnything();
            return Terminator(_expectationPtr);
        }

        Terminator Never() {
            Exactly(0);
            return Terminator(_expectationPtr);
        }

        Terminator Once() {
            Exactly(1);
            return Terminator(_expectationPtr);
        }

        Terminator Twice() {
            Exactly(2);
            return Terminator(_expectationPtr);
        }

        Terminator AtLeastOnce() {
            verifyInvocations(-1);
            return Terminator(_expectationPtr);
        }

        Terminator Exactly(const int times) {
            if (times < 0) {
                FAIL(std::string("bad argument times:").append(fakeit::to_string(times)));
            }
            verifyInvocations(times);
            return Terminator(_expectationPtr);
        }

        Terminator Exactly(const Quantity &q) {
            Exactly(q.quantity);
            return Terminator(_expectationPtr);
        }

        Terminator AtLeast(const int times) {
            if (times < 0) {
                FAIL(std::string("bad argument times:").append(fakeit::to_string(times)));
            }
            verifyInvocations(-times);
            return Terminator(_expectationPtr);
        }

        Terminator AtLeast(const Quantity &q) {
            AtLeast(q.quantity);
            return Terminator(_expectationPtr);
        }

        SequenceVerificationProgress setFileInfo(const char * file, int line, const char * callingMethod) {
            _expectationPtr->setFileInfo(file, line, callingMethod);
            return *this;
        }
    };
}

namespace fakeit {

    class UsingProgress {
        fakeit::FakeitContext &_fakeit;
        InvocationsSourceProxy _sources;

        void collectSequences(std::vector<fakeit::Sequence *> &) {
        }

        template<typename ... list>
        void collectSequences(std::vector<fakeit::Sequence *> &vec, const fakeit::Sequence &sequence,
                              const list &... tail) {
            vec.push_back(&const_cast<fakeit::Sequence &>(sequence));
            collectSequences(vec, tail...);
        }

    public:

        UsingProgress(fakeit::FakeitContext &fakeit, InvocationsSourceProxy source) :
                _fakeit(fakeit),
                _sources(source) {
        }

        template<typename ... list>
        SequenceVerificationProgress Verify(const fakeit::Sequence &sequence, const list &... tail) {
            std::vector<fakeit::Sequence *> allSequences;
            collectSequences(allSequences, sequence, tail...);
            SequenceVerificationProgress progress(_fakeit, _sources, allSequences);
            return progress;
        }

    };
}

namespace fakeit {

    class UsingFunctor {

        friend class VerifyFunctor;

        FakeitContext &_fakeit;

    public:

        UsingFunctor(FakeitContext &fakeit) : _fakeit(fakeit) {
        }

        template<typename ... list>
        UsingProgress operator()(const ActualInvocationsSource &head, const list &... tail) {
            std::vector<ActualInvocationsSource *> allMocks{&InvocationUtils::remove_const(head),
                                                            &InvocationUtils::remove_const(tail)...};
            InvocationsSourceProxy aggregateInvocationsSource{new AggregateInvocationsSource(allMocks)};
            UsingProgress progress(_fakeit, aggregateInvocationsSource);
            return progress;
        }

    };
}
#include <set>

namespace fakeit {

    class VerifyFunctor {

        FakeitContext &_fakeit;


    public:

        VerifyFunctor(FakeitContext &fakeit) : _fakeit(fakeit) {
        }

        template<typename ... list>
        SequenceVerificationProgress operator()(const Sequence &sequence, const list &... tail) {
            std::vector<Sequence *> allSequences{&InvocationUtils::remove_const(sequence),
                                                 &InvocationUtils::remove_const(tail)...};

            std::vector<ActualInvocationsSource *> involvedSources;
            InvocationUtils::collectInvolvedMocks(allSequences, involvedSources);
            InvocationsSourceProxy aggregateInvocationsSource{new AggregateInvocationsSource(involvedSources)};

            UsingProgress usingProgress(_fakeit, aggregateInvocationsSource);
            return usingProgress.Verify(sequence, tail...);
        }

    };

}
#include <set>
#include <memory>
namespace fakeit {

    class VerifyNoOtherInvocationsVerificationProgress {

        friend class VerifyNoOtherInvocationsFunctor;

        struct VerifyNoOtherInvocationsExpectation {

            friend class VerifyNoOtherInvocationsVerificationProgress;

            ~VerifyNoOtherInvocationsExpectation() FAKEIT_THROWS {
                if (UncaughtException()) {
                    return;
                }

                VerifyExpectation(_fakeit);
            }

            void setFileInfo(const char * file, int line, const char * callingMethod) {
                _file = file;
                _line = line;
                _callingMethod = callingMethod;
            }

        private:

            VerificationEventHandler &_fakeit;
            std::vector<ActualInvocationsSource *> _mocks;

			const char * _file;
            int _line;
			const char * _callingMethod;
            bool _isVerified;

            VerifyNoOtherInvocationsExpectation(VerificationEventHandler &fakeit,
                                                std::vector<ActualInvocationsSource *> mocks) :
                    _fakeit(fakeit),
                    _mocks(mocks),
                    _line(0),
                    _isVerified(false) {
            }

            VerifyNoOtherInvocationsExpectation(const VerifyNoOtherInvocationsExpectation &other) = default;

            void VerifyExpectation(VerificationEventHandler &verificationErrorHandler) {
                if (_isVerified)
                    return;
                _isVerified = true;

                std::unordered_set<Invocation *> actualInvocations;
                InvocationUtils::collectActualInvocations(actualInvocations, _mocks);

                std::unordered_set<Invocation *> nonVerifiedInvocations;
                InvocationUtils::selectNonVerifiedInvocations(actualInvocations, nonVerifiedInvocations);

                if (nonVerifiedInvocations.size() > 0) {
                    std::vector<Invocation *> sortedNonVerifiedInvocations;
                    InvocationUtils::sortByInvocationOrder(nonVerifiedInvocations, sortedNonVerifiedInvocations);

                    std::vector<Invocation *> sortedActualInvocations;
                    InvocationUtils::sortByInvocationOrder(actualInvocations, sortedActualInvocations);

                    NoMoreInvocationsVerificationEvent evt(sortedActualInvocations, sortedNonVerifiedInvocations);
                    evt.setFileInfo(_file, _line, _callingMethod);
                    return verificationErrorHandler.handle(evt);
                }
            }

        };

        fakeit::smart_ptr<VerifyNoOtherInvocationsExpectation> _ptr;

        VerifyNoOtherInvocationsVerificationProgress(VerifyNoOtherInvocationsExpectation *ptr) :
                _ptr(ptr) {
        }

        VerifyNoOtherInvocationsVerificationProgress(FakeitContext &fakeit,
                                                     std::vector<ActualInvocationsSource *> &invocationSources)
                : VerifyNoOtherInvocationsVerificationProgress(
                new VerifyNoOtherInvocationsExpectation(fakeit, invocationSources)
        ) {
        }

        bool toBool() {
            FalseEventHandler ev;
            _ptr->VerifyExpectation(ev);
            return ev.handle_result;
        }

    public:


        ~VerifyNoOtherInvocationsVerificationProgress() FAKEIT_THROWS {
        };

        VerifyNoOtherInvocationsVerificationProgress setFileInfo(const char * file, int line,
			const char * callingMethod) {
            _ptr->setFileInfo(file, line, callingMethod);
            return *this;
        }

        operator bool() const {
            return const_cast<VerifyNoOtherInvocationsVerificationProgress *>(this)->toBool();
        }

        bool operator!() const { return !const_cast<VerifyNoOtherInvocationsVerificationProgress *>(this)->toBool(); }

    };

}


namespace fakeit {
    class VerifyNoOtherInvocationsFunctor {

        FakeitContext &_fakeit;

    public:

        VerifyNoOtherInvocationsFunctor(FakeitContext &fakeit) : _fakeit(fakeit) {
        }

        void operator()() {
        }

        template<typename ... list>
        VerifyNoOtherInvocationsVerificationProgress operator()(const ActualInvocationsSource &head,
                                                                const list &... tail) {
            std::vector<ActualInvocationsSource *> invocationSources{&InvocationUtils::remove_const(head),
                                                                     &InvocationUtils::remove_const(tail)...};
            VerifyNoOtherInvocationsVerificationProgress progress{_fakeit, invocationSources};
            return progress;
        }
    };

}
#include <type_traits>


namespace fakeit {

    class SpyFunctor {
    private:

        template<typename R, typename ... arglist, typename std::enable_if<all_true<smart_is_copy_constructible<arglist>::value...>::value, int>::type = 0>
        void spy(const SpyingContext<R, arglist...> &root, int) {
            SpyingContext<R, arglist...> &rootWithoutConst = const_cast<SpyingContext<R, arglist...> &>(root);
            auto methodFromOriginalVT = rootWithoutConst.getOriginalMethodCopyArgs();
            rootWithoutConst.appendAction(new ReturnDelegateValue<R, arglist...>(methodFromOriginalVT));
            rootWithoutConst.commit();
        }

        template<typename R, typename ... arglist>
        void spy(const SpyingContext<R, arglist...> &, long) {
            static_assert(!std::is_same<R, R>::value, "Spy() cannot accept move-only args, use SpyWithoutVerify() instead which is able to forward these args but then they won't be available for Verify().");
        }

        void operator()() {
        }

    public:

        template<typename H, typename ... M>
        void operator()(const H &head, const M &... tail) {
            spy(head, 0);
            this->operator()(tail...);
        }

    };

}

namespace fakeit {

    class SpyWithoutVerifyFunctor {
    private:

        template<typename R, typename ... arglist>
        void spy(const SpyingContext<R, arglist...> &root) {
            SpyingContext<R, arglist...> &rootWithoutConst = const_cast<SpyingContext<R, arglist...> &>(root);
            auto methodFromOriginalVT = rootWithoutConst.getOriginalMethodForwardArgs();
            rootWithoutConst.appendAction(new ReturnDelegateValue<R, arglist...>(methodFromOriginalVT));
            rootWithoutConst.commit();
        }

        void operator()() {
        }

    public:

        template<typename H, typename ... M>
        void operator()(const H &head, const M &... tail) {
            spy(head);
            this->operator()(tail...);
        }

    };

}
#include <vector>
#include <set>

namespace fakeit {
    class VerifyUnverifiedFunctor {

        FakeitContext &_fakeit;

    public:

        VerifyUnverifiedFunctor(FakeitContext &fakeit) : _fakeit(fakeit) {
        }

        template<typename ... list>
        SequenceVerificationProgress operator()(const Sequence &sequence, const list &... tail) {
            std::vector<Sequence *> allSequences{&InvocationUtils::remove_const(sequence),
                                                 &InvocationUtils::remove_const(tail)...};

            std::vector<ActualInvocationsSource *> involvedSources;
            InvocationUtils::collectInvolvedMocks(allSequences, involvedSources);

            InvocationsSourceProxy aggregateInvocationsSource{new AggregateInvocationsSource(involvedSources)};
            InvocationsSourceProxy unverifiedInvocationsSource{
                    new UnverifiedInvocationsSource(aggregateInvocationsSource)};

            UsingProgress usingProgress(_fakeit, unverifiedInvocationsSource);
            return usingProgress.Verify(sequence, tail...);
        }

    };

    class UnverifiedFunctor {
    public:
        UnverifiedFunctor(FakeitContext &fakeit) : Verify(fakeit) {
        }

        VerifyUnverifiedFunctor Verify;

        template<typename ... list>
        UnverifiedInvocationsSource operator()(const ActualInvocationsSource &head, const list &... tail) {
            std::vector<ActualInvocationsSource *> allMocks{&InvocationUtils::remove_const(head),
                                                            &InvocationUtils::remove_const(tail)...};
            InvocationsSourceProxy aggregateInvocationsSource{new AggregateInvocationsSource(allMocks)};
            UnverifiedInvocationsSource unverifiedInvocationsSource{aggregateInvocationsSource};
            return unverifiedInvocationsSource;
        }













    };
}
/* clang-format off */
namespace fakeit {

    static UsingFunctor Using(Fakeit);
    static VerifyFunctor Verify(Fakeit);
    static VerifyNoOtherInvocationsFunctor VerifyNoOtherInvocations(Fakeit);
    static UnverifiedFunctor Unverified(Fakeit);
    static SpyFunctor Spy;
    static SpyWithoutVerifyFunctor SpyWithoutVerify;
    static FakeFunctor Fake;
    static WhenFunctor When;

    template<class T>
    class SilenceUnusedVariableWarnings {

        void use(void *) {
        }

        SilenceUnusedVariableWarnings() {
            use(&Fake);
            use(&When);
            use(&Spy);
            use(&SpyWithoutVerify);
            use(&Using);
            use(&Verify);
            use(&VerifyNoOtherInvocations);
            use(&_);
        }
    };
}
/* clang-format on */
#ifdef _MSC_VER
#define __func__ __FUNCTION__
#endif

#define MOCK_TYPE(mock) \
    std::remove_reference<decltype((mock).get())>::type

#define OVERLOADED_METHOD_PTR(mock, method, prototype) \
    fakeit::Prototype<prototype>::template MemberType<typename MOCK_TYPE(mock)>::get(&MOCK_TYPE(mock)::method)

#define CONST_OVERLOADED_METHOD_PTR(mock, method, prototype) \
    fakeit::Prototype<prototype>::template MemberType<typename MOCK_TYPE(mock)>::getConst(&MOCK_TYPE(mock)::method)

#define REF_OVERLOADED_METHOD_PTR(mock, method, prototype) \
    fakeit::Prototype<prototype>::MemberType<typename MOCK_TYPE(mock)>::getRef(&MOCK_TYPE(mock)::method)

#define CONST_REF_OVERLOADED_METHOD_PTR(mock, method, prototype) \
    fakeit::Prototype<prototype>::MemberType<typename MOCK_TYPE(mock)>::getConstRef(&MOCK_TYPE(mock)::method)

#define R_VAL_REF_OVERLOADED_METHOD_PTR(mock, method, prototype) \
    fakeit::Prototype<prototype>::MemberType<typename MOCK_TYPE(mock)>::getRValRef(&MOCK_TYPE(mock)::method)

#define CONST_R_VAL_REF_OVERLOADED_METHOD_PTR(mock, method, prototype) \
    fakeit::Prototype<prototype>::MemberType<typename MOCK_TYPE(mock)>::getConstRValRef(&MOCK_TYPE(mock)::method)

#define Dtor(mock) \
    (mock).dtor().setMethodDetails(#mock,"destructor")

#define Method(mock, method) \
    (mock).template stub<__COUNTER__>(&MOCK_TYPE(mock)::method).setMethodDetails(#mock,#method)

#define OverloadedMethod(mock, method, prototype) \
    (mock).template stub<__COUNTER__>(OVERLOADED_METHOD_PTR( mock , method, prototype )).setMethodDetails(#mock,#method)

#define ConstOverloadedMethod(mock, method, prototype) \
    (mock).template stub<__COUNTER__>(CONST_OVERLOADED_METHOD_PTR( mock , method, prototype )).setMethodDetails(#mock,#method)

#define RefOverloadedMethod(mock, method, prototype) \
    (mock).template stub<__COUNTER__>(REF_OVERLOADED_METHOD_PTR( mock , method, prototype )).setMethodDetails(#mock,#method)

#define ConstRefOverloadedMethod(mock, method, prototype) \
    (mock).template stub<__COUNTER__>(CONST_REF_OVERLOADED_METHOD_PTR( mock , method, prototype )).setMethodDetails(#mock,#method)

#define RValRefOverloadedMethod(mock, method, prototype) \
    (mock).template stub<__COUNTER__>(R_VAL_REF_OVERLOADED_METHOD_PTR( mock , method, prototype )).setMethodDetails(#mock,#method)

#define ConstRValRefOverloadedMethod(mock, method, prototype) \
    (mock).template stub<__COUNTER__>(CONST_R_VAL_REF_OVERLOADED_METHOD_PTR( mock , method, prototype )).setMethodDetails(#mock,#method)

#define Verify(...) \
        Verify( __VA_ARGS__ ).setFileInfo(__FILE__, __LINE__, __func__)

#define Using(...) \
        Using( __VA_ARGS__ )

#define VerifyNoOtherInvocations(...) \
    VerifyNoOtherInvocations( __VA_ARGS__ ).setFileInfo(__FILE__, __LINE__, __func__)

#define Fake(...) \
    Fake( __VA_ARGS__ )

#define When(call) \
    When(call)

#endif
