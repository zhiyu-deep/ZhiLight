#pragma once
#include <stdexcept>

class ZhiLightException : public std::runtime_error {
public:
    explicit ZhiLightException(const std::string& msg) : std::runtime_error(msg) { }
};
