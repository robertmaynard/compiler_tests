

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

  #convert the result string which might have a newline character into a pure
  #number by grabbing the first character
  string(SUBSTRING "${result}" 0 1 result)
  set(${var} ${result} PARENT_SCOPE)
endfunction()


function(build_and_test_directory lang dir)

  set(options RUNABLE)
  set(oneValueArgs )
  set(multiValueArgs )
  cmake_parse_arguments(CT "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(name ${dir})
  set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${name}/")
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${name}/")

  #determine if the test is expected to compile or fail to build. We use
  #this information to built the test name to make it clear to the user
  #what a 'passing' test means
  set(retcode 1)
  get_expected_result(${lang} ${src_dir} retcode)

  set(build_name ${name})
  if(retcode EQUAL 0)
    set(build_name "${build_name}_builds")
    set(test_label "BUILDS")
  else()
    set(build_name "${build_name}_fails")
    set(test_label "FAILS")
  endif()

  if(CT_RUNABLE)
  add_test(NAME ${name}_compile
           COMMAND ${CMAKE_CTEST_COMMAND}
           --build-and-test ${src_dir} ${build_dir}
           --build-generator ${CMAKE_GENERATOR}
           )
  add_test(NAME ${build_name}
           COMMAND ${build_dir}/${name}
           )
  set_tests_properties(${name}_compile PROPERTIES FIXTURES_SETUP ${name})
  set_tests_properties(${build_name} PROPERTIES FIXTURES_REQUIRED ${name})

  else()
  add_test(NAME ${build_name}
           COMMAND ${CMAKE_CTEST_COMMAND}
           --build-and-test ${src_dir} ${build_dir}
           --build-generator ${CMAKE_GENERATOR}
           )
  endif()

  set_tests_properties(${build_name} PROPERTIES WILL_FAIL ${retcode})
  set_tests_properties(${build_name} PROPERTIES LABELS ${test_label} )
endfunction()

function(add_compile_test lang dir)
  build_and_test_directory(${lang} ${dir})
endfunction()

function(add_compile_run_test lang dir)
  build_and_test_directory(${lang} ${dir} RUNABLE)
endfunction()
