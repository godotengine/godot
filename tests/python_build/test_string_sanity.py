import pathlib
import re

import pytest

# This regex matches C++ string literals, considering escaped quotes.
# This negative lookbehind ensures strings like u"text", L"text", u8"text" are ignored.
# R"" strings are also ignored because we cannot confidently parse them.
is_plain_string_regex = re.compile(r'(?<!u8)(?<![UuRLpP])"(?:\\.|[^"\\])*"')
is_comment_regex = re.compile(r"(//[^\n]*\n)|(/\*.*?\*/)")


def get_tested_cpp_files():
    repo_dir = pathlib.Path(__file__).parent.parent.parent
    return [
        *repo_dir.glob("core/**/*.h"),
        *repo_dir.glob("drivers/**/*.h"),
        *repo_dir.glob("editor/**/*.h"),
        *repo_dir.glob("main/**/*.h"),
        *repo_dir.glob("misc/**/*.h"),
        *repo_dir.glob("modules/**/*.h"),
        *repo_dir.glob("platform/**/*.h"),
        *repo_dir.glob("scene/**/*.h"),
        *repo_dir.glob("servers/**/*.h"),
        *repo_dir.glob("tests/**/*.h"),
    ]


def assert_string_literals_are_ascii(file_path):
    content = pathlib.Path(file_path).read_text(encoding="utf8")

    # Ignore strings in comments.
    content_no_comments: str = is_comment_regex.sub("", content)
    if 'R"' in content_no_comments:
        return  # File contains raw string, too risky to parse because we'll just make mistakes anyway.

    fail_messages = []
    for result in is_plain_string_regex.finditer(content_no_comments):
        # Remove the quotes from the string
        plain_string = result.group(0)
        if not is_ascii_string(plain_string):
            # Minimum index we know the string could start at for sure.
            # But in actuality it may be offset because we removed comments.
            start_min = result.start(0)
            # Searching for the relevant string is technically not correct, but together with the min start index
            # it yields the correct line number most times.
            line_number = content[: content.find(plain_string, start_min)].count("\n") + 1
            fail_messages.append(
                f"Non-ASCII characters in plain string: {file_path}:{line_number} (index not guaranteed to be correct). Prefix the string with u8 to make it explicit it's non-ascii."
            )

    if fail_messages:
        pytest.fail("\n".join(fail_messages))


def is_ascii_string(s):
    try:
        s.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


@pytest.mark.parametrize("file", get_tested_cpp_files())
def test_strings_all_ascii(file):
    assert_string_literals_are_ascii(file)
