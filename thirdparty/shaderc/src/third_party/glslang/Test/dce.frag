#version 400

const bool flag = false;

int c = 0;

void bar()
{
    if (flag)
        ++c;  // should still show up in AST
    else
        ++c;

    flag ? ++c : ++c;  // both should still show up in AST

    switch (c) {
    case 1:
        ++c;
        break;
        ++c;  // should still show up in AST
    case 2:
        break;
        ++c;  // should still show up in AST
    default:
        break;
    }

    for (int i = 0; i < 0; ++i)
        ++c;  // should still show up in AST

    for (int i = 0; i < 10; ++i) {
        if (c < 3) {
            break; 
            ++c;    // should still show up in AST
        } else {
            continue;
            ++c;    // should still show up in AST
        }
    }

    return;

    ++c;      // should still show up in AST
}

int foo()     // not called, but should still show up in AST
{
    if (c > 4) {
        return 4;
        ++c;   // should still show up in AST
    }

    return 5;

    ++c;       // should still show up in AST
}
