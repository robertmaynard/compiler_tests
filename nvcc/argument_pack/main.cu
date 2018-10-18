#include <type_traits>


template <typename... Ls>
struct apply_impl
{
  using type = std::integral_constant<bool,true>;
};  

template <typename... Args>
using apply = typename apply_impl<Args...>::type;

template<typename T>
using type_impl = std::integral_constant<bool,T::value>;

template<typename P, typename T>
struct nope
{
using that = apply<P, T>;  
using type = std::integral_constant<bool,that::value>;
// using type = type_impl<that>;
};


int main(int, char*[])

{
	return 0;
}
