project(mylight)
add_definitions(-std=c++11)
add_executable(mylight 
    input.cpp
    fun.cpp
    Zernike.cpp
    optical_field.cpp
    main.cpp)