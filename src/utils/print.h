#pragma once
#include <string>
#include <iostream>
#include <vector>

template<typename T>
void print_vector(const std::string& name, const std::vector<T>& v) {
    std::cout << name << ": [";
    for (int i = 0; i < v.size(); i++) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << (int) v[i];
    }
    std::cout << "]" << std::endl;
}