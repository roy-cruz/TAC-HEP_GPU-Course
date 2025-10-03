#include <iostream>

struct Student {
    std::string name;
    std::string email;
    std::string user;
    std::string experiment;
};

void print_student(const Student& student){
    std::cout << "Name:       " << student.name << std::endl;
    std::cout << "Email:      " << student.email << std::endl;
    std::cout << "User:       " << student.user << std::endl;
    std::cout << "Experiment: " << student.experiment << std::endl;
    std::cout << std::endl;
}

int main() {
    Student student1 = {
        "Roy F. Cruz", 
        "rfcruz@wisc.edu",
        "rfcruz",
        "CMS"
    };

    Student student3 = {
        "Jesse Pinkman", 
        "jpinkman@wisc.edu",
        "jpinkman",
        "The Krystal Ship"
    };

    Student student2 = {
        "Michael Scott", 
        "mscott@wisc.edu",
        "mscott",
        "Dunder Mifflin"
    };

    std::cout << "Student Information:" << std::endl;
    std::cout << "---------------------" << std::endl;
    print_student(student1);
    print_student(student2);
    print_student(student3);
    return 0;
}