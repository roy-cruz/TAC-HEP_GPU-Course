#include <iostream>

int input_rps();
int play_rps(int plyr1, int plyr2);

int main() {
    std::cout << "Player 1..." << std::endl;
    int input1 = input_rps();
    std::cout << "Player 2..." << std::endl;
    int input2 = input_rps();

    play_rps(input1, input2);

    return 0;
}

int input_rps() {
    int choice;
    std::cout << "Enter your choice (0: Scissors, 1: Paper, 2: Rock): ";
    std::cin >> choice;
    return choice;
};

// 0 + 0 -> 0 (Draw)
// 1 + 1 -> 2 (Draw)
// 2 + 2 -> 1 (Draw)

// 1 + 2 -> 0 (P1 wins)
// 0 + 1 -> 1 (P1 wins)
// 2 + 0 -> 2 (P1 wins)

// 2 + 1 -> 0 (P1 loses)
// 1 + 0 -> 1 (P1 loses)
// 0 + 2 -> 2 (P1 loses)



int play_rps(int plyr1, int plyr2) {
    plyr1 = plyr1 % 3;
    plyr2 = plyr2 % 3;

    if ((plyr1 - plyr2) == 0){
        std::cout << "Draw!" << std::endl;
        return 0;
    }
    if (((plyr1 + 1) % 3) == plyr2){
        std::cout << "Player 1 wins!" << std::endl;
        return 0;
    }
    std::cout << "Player 2 wins!" << std::endl;
    return 0;
}