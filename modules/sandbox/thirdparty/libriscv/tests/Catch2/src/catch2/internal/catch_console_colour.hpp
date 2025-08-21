
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_CONSOLE_COLOUR_HPP_INCLUDED
#define CATCH_CONSOLE_COLOUR_HPP_INCLUDED

#include <catch2/internal/catch_unique_ptr.hpp>

#include <iosfwd>
#include <cstdint>

namespace Catch {

    enum class ColourMode : std::uint8_t;
    class IStream;

    struct Colour {
        enum Code {
            None = 0,

            White,
            Red,
            Green,
            Blue,
            Cyan,
            Yellow,
            Grey,

            Bright = 0x10,

            BrightRed = Bright | Red,
            BrightGreen = Bright | Green,
            LightGrey = Bright | Grey,
            BrightWhite = Bright | White,
            BrightYellow = Bright | Yellow,

            // By intention
            FileName = LightGrey,
            Warning = BrightYellow,
            ResultError = BrightRed,
            ResultSuccess = BrightGreen,
            ResultExpectedFailure = Warning,

            Error = BrightRed,
            Success = Green,
            Skip = LightGrey,

            OriginalExpression = Cyan,
            ReconstructedExpression = BrightYellow,

            SecondaryText = LightGrey,
            Headers = White
        };
    };

    class ColourImpl {
    protected:
        //! The associated stream of this ColourImpl instance
        IStream* m_stream;
    public:
        ColourImpl( IStream* stream ): m_stream( stream ) {}

        //! RAII wrapper around writing specific colour of text using specific
        //! colour impl into a stream.
        class ColourGuard {
            ColourImpl const* m_colourImpl;
            Colour::Code m_code;
            bool m_engaged = false;

        public:
            //! Does **not** engage the guard/start the colour
            ColourGuard( Colour::Code code,
                         ColourImpl const* colour );

            ColourGuard( ColourGuard const& rhs ) = delete;
            ColourGuard& operator=( ColourGuard const& rhs ) = delete;

            ColourGuard( ColourGuard&& rhs ) noexcept;
            ColourGuard& operator=( ColourGuard&& rhs ) noexcept;

            //! Removes colour _if_ the guard was engaged
            ~ColourGuard();

            /**
             * Explicitly engages colour for given stream.
             *
             * The API based on operator<< should be preferred.
             */
            ColourGuard& engage( std::ostream& stream ) &;
            /**
             * Explicitly engages colour for given stream.
             *
             * The API based on operator<< should be preferred.
             */
            ColourGuard&& engage( std::ostream& stream ) &&;

        private:
            //! Engages the guard and starts using colour
            friend std::ostream& operator<<( std::ostream& lhs,
                                             ColourGuard& guard ) {
                guard.engageImpl( lhs );
                return lhs;
            }
            //! Engages the guard and starts using colour
            friend std::ostream& operator<<( std::ostream& lhs,
                                            ColourGuard&& guard) {
                guard.engageImpl( lhs );
                return lhs;
            }

            void engageImpl( std::ostream& stream );

        };

        virtual ~ColourImpl(); // = default
        /**
         * Creates a guard object for given colour and this colour impl
         *
         * **Important:**
         * the guard starts disengaged, and has to be engaged explicitly.
         */
        ColourGuard guardColour( Colour::Code colourCode );

    private:
        virtual void use( Colour::Code colourCode ) const = 0;
    };

    //! Provides ColourImpl based on global config and target compilation platform
    Detail::unique_ptr<ColourImpl> makeColourImpl( ColourMode colourSelection,
                                                   IStream* stream );

    //! Checks if specific colour impl has been compiled into the binary
    bool isColourImplAvailable( ColourMode colourSelection );

} // end namespace Catch

#endif // CATCH_CONSOLE_COLOUR_HPP_INCLUDED
