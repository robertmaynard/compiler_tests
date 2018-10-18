#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

template <class... T> struct list {};
template<class T> struct type_ { using type = T; };

namespace detail
{
  template<class, class>
  struct dup_append_list;
  template<template<class...> class List, class... Ts, class... Us>
  struct dup_append_list<List<Ts...>, List<Us...>>
  {
    using type = List<Ts..., Ts..., Us...>;
  };
  template<class T, template<class...> class List, std::size_t N>
  struct filled_list_impl
  : dup_append_list<
    typename filled_list_impl<T, List, N/2>::type,
    typename filled_list_impl<T, List, N - N/2*2>::type
  >
  {};
  template<class T, template<class...> class List>
  struct filled_list_impl<T, List, 1>
  {
    using type = List<T>;
  };
  template<class T, template<class...> class List>
  struct filled_list_impl<T, List, 0>
  {
    using type = List<>;
  };
}

template<class T, std::size_t N, template<class...> class List = list>
using filled_list = typename detail::filled_list_impl<T, List, N>::type;

namespace detail
{
  template<class T> struct element_at;
  
  template<class... Ts>
  struct element_at<list<Ts...>>
  {
    //previous version that worked
    template<class T> type_<T> static at(Ts..., type_<T>*, ...);
  };
    
  template<class T> T extract_type(type_<T>*);
  
  template<std::size_t N, class Seq> struct at_impl;
  
  //version that worked in CUDA 8
  template<std::size_t N, template<typename...> class L, class... Ts>
  struct at_impl<N,L<Ts...>>
     : decltype(element_at<filled_list<void const *, N>>::at(static_cast<type_<Ts>*>(nullptr)...))
  {
  };

}

template <class L, std::size_t Index>
using at_c = typename detail::at_impl<Index, L>::type;


int main(int, char*[])
{
	using vec_float = std::vector<float>;
	using l = list<float, vec_float, vec_float, vec_float>;
	using result_type = at_c<l, 1UL>;

	//should print out std::vector<float> type
	result_type temp {};
	std::cout << typeid(temp).name() << std::endl;
	return 0;
}
