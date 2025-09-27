#include <iostream>

void swap_vals(int &a, int &b) {
    // Swap the values of a and b
    int temp = a;
    a = b;
    b = temp;
}

int main() {
    int x = 42;
    int y = 2025;
    std::cout << "Values before swap: x = " << x << ", y = " << y << std::endl;

    swap_vals(x, y);
    std::cout << "Values after swap: x = " << x << ", y = " << y << std::endl;

    return 0;
}