#include <iostream>

struct Student {
    std::string name;
    std::string email;
    std::string user;
    std::string experiment;
};

void print_student(const Student& student);

int main() {
    Student student1 = {
        "Roy", 
        "rfcruz@wisc.edu",
        "rfcruz",
        "CMS"
    };

    Student student2 = {
        "Jane Doe", 
        "jdoe@wisc.edu",
        "jdoe",
        "ATLAS"
    };

    Student student3 = {
        "John Smith", 
        "jsmith@wisc.edu",
        "jsmith",
        "IceCube"
    };

    std::cout << "Student Information:" << std::endl;
    print_student(student1);
    std::cout << std::endl;
    print_student(student2);
    std::cout << std::endl;
    print_student(student3);
    return 0;
}

void print_student(const Student& student){
    std::cout << "Name: " << student.name << std::endl;
    std::cout << "Email: " << student.email << std::endl;
    std::cout << "User: " << student.user << std::endl;
    std::cout << "Experiment: " << student.experiment << std::endl;
}