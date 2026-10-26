#ifndef ADAPTQ_LOGGER_HPP
#define ADAPTQ_LOGGER_HPP
#include <iostream>
namespace adaptq { namespace log { inline void info(const char* msg) { std::cout << msg << std::endl; } } }
#endif
