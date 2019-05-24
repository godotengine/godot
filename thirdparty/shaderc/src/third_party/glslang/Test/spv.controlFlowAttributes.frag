#version 450

#extension GL_EXT_control_flow_attributes : enable

bool cond;

void main()
{
        [[unroll]]                 for (int i = 0; i < 8; ++i) { }
        [[loop]]                   for (;;) { }
        [[dont_unroll]]            while(true) {  }
        [[dependency_infinite]]    do {  } while(true);
        [[dependency_length(1+3)]] for (int i = 0; i < 8; ++i) { }
        [[flatten]]                if (cond) { } else { }
        [[branch]]                 if (cond) cond = false;
        [[dont_flatten]]           switch(3) {  }                      // dropped
        [[dont_flatten]]           switch(3) { case 3: break; }

        // warnings on all these
        [[unroll(2)]]              for (int i = 0; i < 8; ++i) { }
        [[dont_unroll(-2)]]        while(true) {  }
        [[dependency_infinite(3)]] do {  } while(true);
        [[dependency_length]]      for (int i = 0; i < 8; ++i) { }
        [[flatten(3)]]             if (cond) { } else { }
        [[branch(5.2)]]            if (cond) cond = false;
        [[dont_flatten(3 + 7)]]    switch(3) { case 3: break; }

        // other valid uses
        [[ unroll, dont_unroll, dependency_length(2) ]]  while(cond) {  }
        [ [ dont_flatten , branch ] ]                    switch(3) { case 3: break; }
        [
            // attribute
            [
                // here
                flatten
            ]
        ]                       if (cond) { } else { }
        [[ dependency_length(2), dependency_infinite ]]  while(cond) {  }
}
