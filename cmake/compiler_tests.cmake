

function(get_expected_result lang src_dir var)

  set(version_full ${CMAKE_${lang}_COMPILER_VERSION})
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" version_major ${version_full})
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" version_minor ${version_full})
  set(version "${version_major}.${version_minor}")

  #see if an explicit override for this version exists
  set(override_path "${src_dir}/result.${version}")
  set(result_path "${src_dir}/result")
  if(EXISTS override_path)
    set(result_path ${override_path})
  endif()

  #load up the result
  file(READ ${result_path} result)
  set(${var} ${result} PARENT_SCOPE)

endfunction()

function(add_compile_test lang dir)

  set(name ${dir})
  set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${name}/")
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}/")

  #need to make sure the build_dir exists otherwise the configure
  #stage fails for some reason, even though -B supports making non-existent
  #directories
  file(MAKE_DIRECTORY ${build_dir})

  add_test(
    NAME ${name}_configure
    COMMAND ${CMAKE_COMMAND} -B${build_dir} -H${src_dir}
    WORKING_DIRECTORY ${build_dir}
    )

  add_test(
    NAME ${name}
    COMMAND "${CMAKE_COMMAND}" --build ${build_dir}
    WORKING_DIRECTORY ${build_dir}
    )

  set_tests_properties(${name}_configure PROPERTIES FIXTURES_SETUP ${name})
  set_tests_properties(${name} PROPERTIES FIXTURES_REQUIRED ${name})

  #setup up the rules on if we expect the test to pass or fail
  set(result 1)
  get_expected_result(${lang} ${src_dir} result)
  message(STATUS "result for ${name} is: ${result}")
  set_tests_properties(${name} PROPERTIES WILL_FAIL ${result})

endfunction()
