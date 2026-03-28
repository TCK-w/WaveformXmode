# This is a Windows-specific flag that enables exception handling in host code
if(WIN32)
    set(WIN_FLAG "/EHsc")
endif()


set(COMPILE_FLAGS "-fsycl -Wall ${WIN_FLAG}")
set(LINK_FLAGS "-fsycl")


add_library(WaveformXmode SHARED WaveformXmode.cpp)
set_target_properties(WaveformXmode PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(WaveformXmode PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
