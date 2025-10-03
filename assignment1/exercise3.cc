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
    // Should return 0 if draw, 1 if player 1 wins, 2 if player 2 wins
    plyr1 = plyr1 % 3;
    plyr2 = plyr2 % 3;

    if ((plyr1 - plyr2) == 0){
        // Draw
        return 0;
    }
    if (((plyr1 + 1) % 3) == plyr2){
        // Player 1 wins
        return 1;
    }
    // Player 2 wins
    return 2;
}

int main() {
    std::cout << "Player 1..." << std::endl;
    int input1 = input_rps();
    std::cout << "Player 2..." << std::endl;
    int input2 = input_rps();

    std::cout << "---------------------" << std::endl;
    std::cout << "Result:" << std::endl;
    int rslt = play_rps(input1, input2);

    // Print the result
    if (rslt == 0) {
        std::cout << "It's a draw!" << std::endl;
    } else if (rslt == 1) {
        std::cout << "Player 1 wins!" << std::endl; 
    } else {
        std::cout << "Player 2 wins!" << std::endl;
    }

    return 0;
}