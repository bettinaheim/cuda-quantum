/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <set>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "dynamics/evaluation.h"
#include "dynamics/operator_leafs.h"
#include "dynamics/templates.h"
#include "utils/cudaq_utils.h"
#include "utils/tensor.h"
#include "common/FmtCore.h"

namespace cudaq {

// fixme: here backward compatibility only
enum class pauli { I, X, Y, Z };
class spin_operator;

/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
template <typename HandlerTy>
class operator_sum {
  template <typename T>
  friend class operator_sum;
  template <typename T>
  friend class product_operator;

private:
  // inserts a new term combining it with an existing one if possible
  void insert(product_operator<HandlerTy> &&other);
  void insert(const product_operator<HandlerTy> &other);

  void aggregate_terms();

  template <typename... Args>
  void aggregate_terms(product_operator<HandlerTy> &&head, Args &&...args);

  template <typename EvalTy>
  EvalTy evaluate(operator_arithmetics<EvalTy> arithmetics) const;

protected:
  std::unordered_map<std::string, int>
      term_map; // quick access to term index given its id (used for aggregating
                // terms)
  std::vector<std::vector<HandlerTy>> terms;
  std::vector<scalar_operator> coefficients;

  template <typename... Args,
            std::enable_if_t<std::conjunction<std::is_same<
                                 product_operator<HandlerTy>, Args>...>::value,
                             bool> = true>
  operator_sum(Args &&...args);

public:
  // read-only properties

  /// @brief The degrees of freedom that the operator acts on.
  /// By default, degrees reflect the ordering convention (endianness) used in
  /// CUDA-Q, and the ordering of the matrix returned by default by `to_matrix`.
  std::vector<int> degrees(bool application_order = true) const;

  /// @brief Return the number of operator terms that make up this operator sum.
  std::size_t num_terms() const;

  /// FIXME: GET RID OF THIS (MAKE ITERABLE INSTEAD)
  std::vector<product_operator<HandlerTy>> get_terms() const;

  // constructors and destructors

  operator_sum(const product_operator<HandlerTy> &other);

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  operator_sum(const operator_sum<T> &other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, matrix_operator>::value &&
                                 !std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  operator_sum(const operator_sum<T> &other,
               const matrix_operator::commutation_behavior &behavior);

  // copy constructor
  operator_sum(const operator_sum<HandlerTy> &other, int size = 0);

  // move constructor
  operator_sum(operator_sum<HandlerTy> &&other, int size = 0);

  ~operator_sum() = default;

  // assignments

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  operator_sum<HandlerTy> &operator=(const product_operator<T> &other);

  operator_sum<HandlerTy> &operator=(const product_operator<HandlerTy> &other);

  operator_sum<HandlerTy> &operator=(product_operator<HandlerTy> &&other);

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  operator_sum<HandlerTy> &operator=(const operator_sum<T> &other);

  // assignment operator
  operator_sum<HandlerTy> &operator=(const operator_sum<HandlerTy> &other);

  // move assignment operator
  operator_sum<HandlerTy> &operator=(operator_sum<HandlerTy> &&other);

  // evaluations

  /// @brief Return the operator_sum<HandlerTy> as a string.
  std::string to_string() const;

  /// @brief Return the matrix representation of the operator.
  /// By default, the matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by default by `degrees`.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  matrix_2 to_matrix(std::unordered_map<int, int> dimensions = {},
                     const std::unordered_map<std::string, std::complex<double>>
                         &parameters = {},
                     bool application_order = true) const;

  // unary operators

  operator_sum<HandlerTy> operator-() const &;
  operator_sum<HandlerTy> operator-() &&;
  operator_sum<HandlerTy> operator+() const &;
  operator_sum<HandlerTy> operator+() &&;

  // right-hand arithmetics

  operator_sum<HandlerTy> operator*(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator*(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator/(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator/(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator+(scalar_operator &&other) const &;
  operator_sum<HandlerTy> operator+(scalar_operator &&other) &&;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator-(scalar_operator &&other) const &;
  operator_sum<HandlerTy> operator-(scalar_operator &&other) &&;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) &&;
  operator_sum<HandlerTy>
  operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy>
  operator+(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy>
  operator-(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator+(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator-(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) &&;

  operator_sum<HandlerTy> &operator*=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator/=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator+=(scalar_operator &&other);
  operator_sum<HandlerTy> &operator+=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator-=(scalar_operator &&other);
  operator_sum<HandlerTy> &operator-=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator*=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(product_operator<HandlerTy> &&other);
  operator_sum<HandlerTy> &operator-=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> &operator-=(product_operator<HandlerTy> &&other);
  operator_sum<HandlerTy> &operator*=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(operator_sum<HandlerTy> &&other);
  operator_sum<HandlerTy> &operator-=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> &operator-=(operator_sum<HandlerTy> &&other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template
  // instantiation is a nightmare.
  template <typename T>
  friend operator_sum<T> operator*(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator*(const scalar_operator &other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   operator_sum<T> &&self);

  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   product_operator<T> &&self);

  // common operators

  template <typename T>
  friend operator_sum<T> operator_handler::empty();

  // utility functions for backward compatibility

#define SPIN_OPS_BACKWARD_COMPATIBILITY                                                   \
  template <typename T = HandlerTy, std::enable_if_t<                                     \
                                      std::is_same<HandlerTy, spin_operator>::value &&    \
                                      std::is_same<HandlerTy, T>::value, bool> = true>

  SPIN_OPS_BACKWARD_COMPATIBILITY
  size_t num_qubits() const {
    return this->degrees().size();
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  void for_each_pauli(std::function<void(pauli, std::size_t)> &&functor) const {
    if (this->terms.size() != 1)
      throw std::runtime_error(
        "spin_op::for_each_pauli on valid for spin_op with n_terms == 1.");
    // FIXME: check order
    for (std::size_t i = 0; i < this->terms[0].size(); ++i) {
      auto str = this->terms[0][i].to_string(false);
      if (str == "Y")
        functor(pauli::Y, i);
      else if (str == "X")
        functor(pauli::X, i);
      else if (str == "Z")
        functor(pauli::Z, i);
      else {
        assert(str == "I");
        functor(pauli::I, i);
      }
    }
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  void for_each_term(std::function<void(operator_sum<HandlerTy> &)> &&functor) const {
    auto prods = this->get_terms();
    for (operator_sum<HandlerTy> term : prods)
      functor(term); // FIXME: functor could modify the term??
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  bool is_identity() const {
    // ignores the coefficients (according to the old behavior)
    for (const auto &term : this->terms) {
      for (const auto &op : term)
        if (op.to_string(false) != "I") return false;
    }
    return true;
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::complex<double> get_coefficient() const {
    if (this->terms.size() != 1)
      throw std::runtime_error(
        "spin_op::get_coefficient called on spin_op with > 1 terms.");
    return this->coefficients[0].evaluate(); // fails if we have parameters
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::string to_string(bool printCoeffs) const {
    std::unordered_map<int, int> dims;
    auto terms = std::move(
      this->evaluate(
              operator_arithmetics<operator_handler::canonical_evaluation>(
                  dims, {})) // fails if operator is parameterized
          .terms);
    std::stringstream ss;
    if (!printCoeffs) {
      std::vector<std::string> printOut;
      printOut.reserve(terms.size());
      for (auto &[coeff, term_str] : terms)
        printOut.emplace_back(term_str);
      std::sort(printOut.begin(), printOut.end());
      ss << fmt::format("{}", fmt::join(printOut, "")); // fixme: why is there no separator between terms??
    } else {
      for (auto &[coeff, term_str] : terms) {
        ss << fmt::format("[{}{}{}j]", coeff.real(),
                          coeff.imag() < 0.0 ? "-" : "+", std::fabs(coeff.imag()))
           << " ";
        ss << term_str;
        ss << "\n";
      }
    }
    return ss.str();
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  static operator_sum<HandlerTy> from_word(const std::string &word);

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::vector<operator_sum<HandlerTy>> distribute_terms(std::size_t numChunks) const {
    // Calculate how many terms we can equally divide amongst the chunks
    auto nTermsPerChunk = num_terms() / numChunks;
    auto leftover = num_terms() % numChunks;

    // Slice the given spin_op into subsets for each chunk
    std::vector<operator_sum<HandlerTy>> chunks;
    for (auto it = this->term_map.cbegin(); it != this->term_map.cend();) {
      operator_sum<HandlerTy> chunk;
      // Evenly distribute any leftovers across the early chunks
      for (auto count = nTermsPerChunk + (chunks.size() < leftover ? 1 : 0); count > 0; --count, ++it)
        chunk += product_operator<HandlerTy>(this->coefficients[it->second], this->terms[it->second]);
      chunks.push_back(chunk);
    }
    return std::move(chunks);
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  static operator_sum<HandlerTy> random(std::size_t nQubits, std::size_t nTerms, unsigned int seed);

  SPIN_OPS_BACKWARD_COMPATIBILITY
  operator_sum(const std::vector<double> &input_vec, std::size_t nQubits);

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::pair<std::vector<std::vector<bool>>, std::vector<std::complex<double>>> get_raw_data() const {
    // fixme: I think we want to start from 0 here, even if the operator 
    // does not contain consecutive degrees starting from 0....
    // fixme: the above can be taken care of by writing a dedicated transformation into bsf directly
    std::unordered_map<int, int> dims;
    auto degrees = this->degrees(false); // degrees in canonical order
    auto term_size = operator_handler::canonical_order(1, 0) ? degrees[0] + 1 : degrees.back() + 1;
    auto evaluated =
      this->evaluate(operator_arithmetics<operator_handler::canonical_evaluation>(
          dims, {})); // fails if we have parameters

    std::vector<std::vector<bool>> bsf_terms;
    std::vector<std::complex<double>> coeffs;
    bsf_terms.reserve(evaluated.terms.size());
    coeffs.reserve(evaluated.terms.size());

    for (const auto &term : evaluated.terms) {
      std::vector<bool> bsf(term_size << 1, 0);
      for (std::size_t i = 0; i < degrees.size(); ++i) {
        if (term.second[i] == 'X')
          bsf[degrees[i]] = 1;
        else if (term.second[i] == 'Z')
          bsf[degrees[i] + term_size] = 1;
        else if (term.second[i] == 'Y') {
          bsf[degrees[i]] = 1;
          bsf[degrees[i] + term_size] = 1;
        }
      }
      bsf_terms.push_back(std::move(bsf));
      coeffs.push_back(term.first);
    }

    return std::pair<std::vector<std::vector<bool>>, std::vector<std::complex<double>>>(
      std::move(bsf_terms), std::move(coeffs));
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::vector<double> getDataRepresentation() const {
    // FIXME: this is an imperfect representation because it does not capture targets accurately
    std::vector<double> dataVec;
    for(std::size_t i = 0; i < this->terms.size(); ++i) {
      for(std::size_t j = 0; j < this->terms[i].size(); ++j) {
        auto op_str = this->terms[i][j].to_string(false);
        // FIXME: align numbering with op codes
        if (op_str == "X")
          dataVec.push_back(1.);
        else if (op_str == "Z")
          dataVec.push_back(2.);
        else if (op_str == "Y")
          dataVec.push_back(3.);
        else
          dataVec.push_back(0.);
      }
      auto coeff = this->coefficients[i].evaluate(); // fails if we have params
      dataVec.push_back(coeff.real());
      dataVec.push_back(coeff.imag());
    }
    dataVec.push_back(this->terms.size());
    return dataVec;
  }
};

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
template <typename HandlerTy>
class product_operator {
  template <typename T>
  friend class product_operator;
  template <typename T>
  friend class operator_sum;

private:
  // template defined as long as T implements an in-place multiplication -
  // won't work if the in-place multiplication was inherited from a base class
  template <typename T>
  static decltype(std::declval<T>().inplace_mult(std::declval<T>()))
  handler_mult(int);
  template <typename T>
  static std::false_type handler_mult(
      ...); // ellipsis ensures the template above is picked if it exists
  static constexpr bool supports_inplace_mult =
      !std::is_same<decltype(handler_mult<HandlerTy>(0)),
                    std::false_type>::value;

#if !defined(NDEBUG)
  bool is_canonicalized() const;
#endif

  typename std::vector<HandlerTy>::const_iterator
  find_insert_at(const HandlerTy &other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, T>::value &&
                                 !product_operator<T>::supports_inplace_mult,
                             std::false_type> = std::false_type()>
  void insert(T &&other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, T>::value &&
                                 product_operator<T>::supports_inplace_mult,
                             std::true_type> = std::true_type()>
  void insert(T &&other);

  std::string get_term_id() const;

  void aggregate_terms();

  template <typename... Args>
  void aggregate_terms(HandlerTy &&head, Args &&...args);

  template <typename EvalTy>
  EvalTy evaluate(operator_arithmetics<EvalTy> arithmetics) const;

protected:
  std::vector<HandlerTy> operators;
  scalar_operator coefficient;

  template <typename... Args,
            std::enable_if_t<
                std::conjunction<std::is_same<HandlerTy, Args>...>::value,
                bool> = true>
  product_operator(scalar_operator coefficient, Args &&...args);

  // keep this constructor protected (otherwise it needs to ensure canonical
  // order)
  product_operator(scalar_operator coefficient,
                   const std::vector<HandlerTy> &atomic_operators,
                   int size = 0);

  // keep this constructor protected (otherwise it needs to ensure canonical
  // order)
  product_operator(scalar_operator coefficient,
                   std::vector<HandlerTy> &&atomic_operators, int size = 0);

public:
  // read-only properties

  /// @brief The degrees of freedom that the operator acts on.
  /// By default, degrees reflect the ordering convention (endianness) used in
  /// CUDA-Q, and the ordering of the matrix returned by default by `to_matrix`.
  std::vector<int> degrees(bool application_order = true) const;

  /// @brief Return the number of operator terms that make up this product
  /// operator.
  std::size_t num_terms() const;

  /// FIXME: GET RID OF THIS (MAKE ITERABLE INSTEAD)
  const std::vector<HandlerTy> &get_terms() const;

  scalar_operator get_coefficient() const;

  // constructors and destructors

  product_operator(double coefficient);

  product_operator(std::complex<double> coefficient);

  product_operator(HandlerTy &&atomic);

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_operator(const product_operator<T> &other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, matrix_operator>::value &&
                                 !std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_operator(const product_operator<T> &other,
                   const matrix_operator::commutation_behavior &behavior);

  // copy constructor
  product_operator(const product_operator<HandlerTy> &other, int size = 0);

  // move constructor
  product_operator(product_operator<HandlerTy> &&other, int size = 0);

  ~product_operator() = default;

  // assignments

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_operator<HandlerTy> &operator=(const product_operator<T> &other);

  // assignment operator
  product_operator<HandlerTy> &
  operator=(const product_operator<HandlerTy> &other);

  // move assignment operator
  product_operator<HandlerTy> &operator=(product_operator<HandlerTy> &&other);

  // evaluations

  /// @brief Return the `product_operator<HandlerTy>` as a string.
  std::string to_string() const;

  /// @brief Return the matrix representation of the operator.
  /// By default, the matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by default by `degrees`.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  matrix_2 to_matrix(std::unordered_map<int, int> dimensions = {},
                     const std::unordered_map<std::string, std::complex<double>>
                         &parameters = {},
                     bool application_order = true) const;

  // comparisons

  /// @brief True, if the other value is an operator_sum<HandlerTy> with
  /// equivalent terms,
  ///  and False otherwise. The equality takes into account that operator
  ///  addition is commutative, as is the product of two operators if they
  ///  act on different degrees of freedom.
  ///  The equality comparison does *not* take commutation relations into
  ///  account, and does not try to reorder terms `blockwise`; it may hence
  ///  evaluate to False, even if two operators in reality are the same.
  ///  If the equality evaluates to True, on the other hand, the operators
  ///  are guaranteed to represent the same transformation for all arguments.
  bool operator==(const product_operator<HandlerTy> &other) const;

  // unary operators

  product_operator<HandlerTy> operator-() const &;
  product_operator<HandlerTy> operator-() &&;
  product_operator<HandlerTy> operator+() const &;
  product_operator<HandlerTy> operator+() &&;

  // right-hand arithmetics

  product_operator<HandlerTy> operator*(scalar_operator &&other) const &;
  product_operator<HandlerTy> operator*(scalar_operator &&other) &&;
  product_operator<HandlerTy> operator*(const scalar_operator &other) const &;
  product_operator<HandlerTy> operator*(const scalar_operator &other) &&;
  product_operator<HandlerTy> operator/(scalar_operator &&other) const &;
  product_operator<HandlerTy> operator/(scalar_operator &&other) &&;
  product_operator<HandlerTy> operator/(const scalar_operator &other) const &;
  product_operator<HandlerTy> operator/(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator+(scalar_operator &&other) const &;
  operator_sum<HandlerTy> operator+(scalar_operator &&other) &&;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator-(scalar_operator &&other) const &;
  operator_sum<HandlerTy> operator-(scalar_operator &&other) &&;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) &&;
  product_operator<HandlerTy>
  operator*(const product_operator<HandlerTy> &other) const &;
  product_operator<HandlerTy>
  operator*(const product_operator<HandlerTy> &other) &&;
  product_operator<HandlerTy>
  operator*(product_operator<HandlerTy> &&other) const &;
  product_operator<HandlerTy> operator*(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy>
  operator+(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy>
  operator-(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator+(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator-(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) &&;

  product_operator<HandlerTy> &operator*=(const scalar_operator &other);
  product_operator<HandlerTy> &operator/=(const scalar_operator &other);
  product_operator<HandlerTy> &
  operator*=(const product_operator<HandlerTy> &other);
  product_operator<HandlerTy> &operator*=(product_operator<HandlerTy> &&other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template
  // instantiation is a nightmare.
  template <typename T>
  friend product_operator<T> operator*(scalar_operator &&other,
                                       const product_operator<T> &self);
  template <typename T>
  friend product_operator<T> operator*(scalar_operator &&other,
                                       product_operator<T> &&self);
  template <typename T>
  friend product_operator<T> operator*(const scalar_operator &other,
                                       const product_operator<T> &self);
  template <typename T>
  friend product_operator<T> operator*(const scalar_operator &other,
                                       product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   product_operator<T> &&self);

  template <typename T>
  friend operator_sum<T> operator*(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator*(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator*(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator*(const scalar_operator &other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   operator_sum<T> &&self);

  // common operators

  template <typename T>
  friend product_operator<T> operator_handler::identity();
  template <typename T>
  friend product_operator<T> operator_handler::identity(int target);

  // utility functions for backward compatibility
  
  SPIN_OPS_BACKWARD_COMPATIBILITY
  bool is_identity() const {
    // ignores the coefficients (according to the old behavior)
    for (const auto &op : this->operators)
      if (op.to_string(false) != "I") return false;
    return true;
  }

};

// type aliases for convenience
typedef std::unordered_map<std::string, std::complex<double>> parameter_map;
typedef std::unordered_map<int, int> dimension_map;
typedef operator_sum<spin_operator> spin_op;

#ifndef CUDAQ_INSTANTIATE_TEMPLATES
extern template class product_operator<matrix_operator>;
extern template class product_operator<spin_operator>;
extern template class product_operator<boson_operator>;
extern template class product_operator<fermion_operator>;

extern template class operator_sum<matrix_operator>;
extern template class operator_sum<spin_operator>;
extern template class operator_sum<boson_operator>;
extern template class operator_sum<fermion_operator>;
#endif

} // namespace cudaq
