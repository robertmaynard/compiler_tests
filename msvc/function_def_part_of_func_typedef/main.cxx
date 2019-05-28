#include <iostream>

//Bug 1: Can't parse function definitions passed as part of a function typedef
struct F {};
typedef void example(F(F,F), F(F,F,F));

int main()
{

  return 0;
}

