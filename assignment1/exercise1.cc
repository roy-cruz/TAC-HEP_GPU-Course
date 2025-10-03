#include <iostream>

void swap_vals(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

void print_array(const int* arr, int length) {
    for (int i = 0; i < length; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    const int length = 10;
    int A[length] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int B[length] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    std::cout << "Array A before swapping: ";
    print_array(A, length);
    std::cout << "Array B before swapping: ";
    print_array(B, length);
    std::cout << std::endl;

    for (int i = 0; i < length; i++) {
        swap_vals(A[i], B[i]);
    }

    std::cout << "Array A after swapping: ";
    print_array(A, length);
    std::cout << "Array B after swapping: ";
    print_array(B, length);
    std::cout << std::endl;

    return 0;
}