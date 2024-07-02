/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


/*! \file system/system_error.h
 *  \brief An exception object used to report error conditions that have an
 *         associated error code
 */

#pragma once

#include <thrust/detail/config.h>
#include <stdexcept>
#include <string>

#include <thrust/system/error_code.h>

THRUST_NAMESPACE_BEGIN

namespace system
{

// [19.5.5] Class system_error

// [19.5.5.1] Class system_error overview

/*! \addtogroup system_diagnostics System Diagnostics
 *  \ingroup system
 *  \{
 */

/*! \brief The class \p system_error describes an exception object used to report error
 *  conditions that have an associated \p error_code. Such error conditions typically
 *  originate from the operating system or other low-level application program interfaces.
 *
 *  Thrust uses \p system_error to report the error codes returned from device backends
 *  such as the CUDA runtime.
 *
 *  The following code listing demonstrates how to catch a \p system_error to recover
 *  from an error.
 *
 *  \code
 *
 *  #include <thrust/device_vector.h>
 *  #include <thrust/system.h>
 *  #include <thrust/sort.h>
 *
 *  void terminate_gracefully(void)
 *  {
 *    // application-specific termination code here
 *    ...
 *  }
 *
 *  int main(void)
 *  {
 *    try
 *    {
 *      thrust::device_vector<float> vec;
 *      thrust::sort(vec.begin(), vec.end());
 *    }
 *    catch(thrust::system_error e)
 *    {
 *      std::cerr << "Error inside sort: " << e.what() << std::endl;
 *      terminate_gracefully();
 *    }
 *
 *    return 0;
 *  }
 *
 *  \endcode
 *
 *  \note If an error represents an out-of-memory condition, implementations are encouraged
 *  to throw an exception object of type \p std::bad_alloc rather than \p system_error.
 */
class system_error
  : public std::runtime_error
{
  public:
    // [19.5.5.2] Class system_error members
    
    /*! Constructs an object of class \p system_error.
     *  \param ec The value returned by \p code().
     *  \param what_arg A string to include in the result returned by \p what().
     *  \post <tt>code() == ec</tt>.
     *  \post <tt>std::string(what()).find(what_arg) != string::npos</tt>.
     */
    inline system_error(error_code ec, const std::string &what_arg);

    /*! Constructs an object of class \p system_error.
     *  \param ec The value returned by \p code().
     *  \param what_arg A string to include in the result returned by \p what().
     *  \post <tt>code() == ec</tt>.
     *  \post <tt>std::string(what()).find(what_arg) != string::npos</tt>.
     */
    inline system_error(error_code ec, const char *what_arg);

    /*! Constructs an object of class \p system_error.
     *  \param ec The value returned by \p code().
     *  \post <tt>code() == ec</tt>.
     */
    inline system_error(error_code ec);

    /*! Constructs an object of class \p system_error.
     *  \param ev The error value used to create an \p error_code.
     *  \param ecat The \p error_category used to create an \p error_code.
     *  \param what_arg A string to include in the result returned by \p what().
     *  \post <tt>code() == error_code(ev, ecat)</tt>.
     *  \post <tt>std::string(what()).find(what_arg) != string::npos</tt>.
     */
    inline system_error(int ev, const error_category &ecat, const std::string &what_arg);

    /*! Constructs an object of class \p system_error.
     *  \param ev The error value used to create an \p error_code.
     *  \param ecat The \p error_category used to create an \p error_code.
     *  \param what_arg A string to include in the result returned by \p what().
     *  \post <tt>code() == error_code(ev, ecat)</tt>.
     *  \post <tt>std::string(what()).find(what_arg) != string::npos</tt>.
     */
    inline system_error(int ev, const error_category &ecat, const char *what_arg);

    /*! Constructs an object of class \p system_error.
     *  \param ev The error value used to create an \p error_code.
     *  \param ecat The \p error_category used to create an \p error_code.
     *  \post <tt>code() == error_code(ev, ecat)</tt>.
     */
    inline system_error(int ev, const error_category &ecat);

    /*! Destructor does not throw.
     */
    inline virtual ~system_error(void) noexcept {};
    
    /*! Returns an object encoding the error.
     *  \return <tt>ec</tt> or <tt>error_code(ev, ecat)</tt>, from the
     *          constructor, as appropriate.
     */
    inline const error_code &code(void) const noexcept;

    /*! Returns a human-readable string indicating the nature of the error.
     *  \return a string incorporating <tt>code().message()</tt> and the
     *          arguments supplied in the constructor.
     */
    inline const char *what(void) const noexcept;

    /*! \cond
     */
  private:
    error_code          m_error_code;
    mutable std::string m_what;

    /*! \endcond
     */
}; // end system_error

} // end system

/*! \} // end system_diagnostics
 */

// import names into thrust::
using system::system_error;

THRUST_NAMESPACE_END

#include <thrust/system/detail/system_error.inl>

