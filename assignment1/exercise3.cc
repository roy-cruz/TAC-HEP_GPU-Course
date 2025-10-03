#include <iostream>

int input_rps() {
    int choice;
    std::cout << "Enter your choice (0: Scissors, 1: Paper, 2: Rock): ";
    std::cin >> choice;
    if (choice < 0 || choice > 2) {
        std::cout << "Invalid choice. Please enter 0, 1, or 2." << std::endl;
        return input_rps();
    }
    return choice;
};

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

int main() {
    std::cout << "Player 1..." << std::endl;
    int input1 = input_rps();
    std::cout << "Player 2..." << std::endl;
    int input2 = input_rps();

    std::cout << "---------------------" << std::endl;
    std::cout << "Result:" << std::endl;
    play_rps(input1, input2);

    return 0;
}