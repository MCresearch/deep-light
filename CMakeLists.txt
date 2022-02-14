project(mymd)
add_definitions(-std=c++11)
add_executable(mymd 
    input.cpp
    fun.cpp
    Zernike.cpp
    optical_field.cpp
    main.cpp)