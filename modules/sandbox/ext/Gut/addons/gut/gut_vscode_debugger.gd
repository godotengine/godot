# ------------------------------------------------------------------------------
# Entry point for using the debugger through VSCode.  The gut-extension for
# VSCode launches this instead of gut_cmdln.gd when running tests through the
# debugger.
#
# This could become more complex overtime, but right now all we have to do is
# to make sure the console printer is enabled or you do not get any output.
# ------------------------------------------------------------------------------
extends 'res://addons/gut/gut_cmdln.gd'

func run_tests(runner):
	runner.get_gut().get_logger().disable_printer('console', false)
	runner.run_tests()


# ##############################################################################
#(G)odot (U)nit (T)est class
#
# ##############################################################################
# The MIT License (MIT)
# =====================
#
# Copyright (c) 2025 Tom "Butch" Wesley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ##############################################################################
