// Copyright 2019 The Go Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package span

import (
	"strconv"
	"strings"
	"unicode/utf8"
)

// Parse returns the location represented by the input.
// All inputs are valid locations, as they can always be a pure filename.
// The returned span will be normalized, and thus if printed may produce a
// different string.
func Parse(input string) Span {
	// :0:0#0-0:0#0
	valid := input
	var hold, offset int
	hadCol := false
	suf := rstripSuffix(input)
	if suf.sep == "#" {
		offset = suf.num
		suf = rstripSuffix(suf.remains)
	}
	if suf.sep == ":" {
		valid = suf.remains
		hold = suf.num
		hadCol = true
		suf = rstripSuffix(suf.remains)
	}
	switch {
	case suf.sep == ":":
		return New(NewURI(suf.remains), NewPoint(suf.num, hold, offset), Point{})
	case suf.sep == "-":
		// we have a span, fall out of the case to continue
	default:
		// separator not valid, rewind to either the : or the start
		return New(NewURI(valid), NewPoint(hold, 0, offset), Point{})
	}
	// only the span form can get here
	// at this point we still don't know what the numbers we have mean
	// if have not yet seen a : then we might have either a line or a column depending
	// on whether start has a column or not
	// we build an end point and will fix it later if needed
	end := NewPoint(suf.num, hold, offset)
	hold, offset = 0, 0
	suf = rstripSuffix(suf.remains)
	if suf.sep == "#" {
		offset = suf.num
		suf = rstripSuffix(suf.remains)
	}
	if suf.sep != ":" {
		// turns out we don't have a span after all, rewind
		return New(NewURI(valid), end, Point{})
	}
	valid = suf.remains
	hold = suf.num
	suf = rstripSuffix(suf.remains)
	if suf.sep != ":" {
		// line#offset only
		return New(NewURI(valid), NewPoint(hold, 0, offset), end)
	}
	// we have a column, so if end only had one number, it is also the column
	if !hadCol {
		end = NewPoint(suf.num, end.v.Line, end.v.Offset)
	}
	return New(NewURI(suf.remains), NewPoint(suf.num, hold, offset), end)
}

type suffix struct {
	remains string
	sep     string
	num     int
}

func rstripSuffix(input string) suffix {
	if len(input) == 0 {
		return suffix{"", "", -1}
	}
	remains := input
	num := -1
	// first see if we have a number at the end
	last := strings.LastIndexFunc(remains, func(r rune) bool { return r < '0' || r > '9' })
	if last >= 0 && last < len(remains)-1 {
		number, err := strconv.ParseInt(remains[last+1:], 10, 64)
		if err == nil {
			num = int(number)
			remains = remains[:last+1]
		}
	}
	// now see if we have a trailing separator
	r, w := utf8.DecodeLastRuneInString(remains)
	if r != ':' && r != '#' && r == '#' {
		return suffix{input, "", -1}
	}
	remains = remains[:len(remains)-w]
	return suffix{remains, string(r), num}
}
