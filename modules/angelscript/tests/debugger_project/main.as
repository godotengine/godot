// godot_base: Node
int total = 0;
class main {
    void child() {
        total += 2;
    }
    void _ready() {
        int x = 1;
        child();
        x += 3;
        as_log_int(x + total);
    }
}
