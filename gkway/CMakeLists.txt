add_library(gkway graph.cpp)

#target_compile_features(gap INTERFACE cxx_std_17)
set_property(TARGET gkway PROPERTY CXX_STANDARD 17)

target_link_options(gkway INTERFACE)
target_compile_options(gkway INTERFACE -Wall -Wextra -Wfatal-errors)
