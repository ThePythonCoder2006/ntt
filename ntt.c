#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#define __TIMER_IMPLEMENTATION__
#include "timer.h"

uint32_t powmod(uint64_t a, uint64_t b, uint32_t m);
uint32_t find_generator(const uint32_t N, const uint32_t phi_N, const uint32_t *const facts_of_phi_N, const uint32_t num_facts);
uint32_t find_prime_modulus(uint32_t *c, uint32_t max_val, uint32_t n);
uint8_t is_prime(uint32_t candidate);
uint32_t factorise(uint32_t *facts, size_t facts_size, uint32_t a);
uint32_t mod_inv(uint64_t a, uint32_t m);

void ntt_naive(uint32_t *X, size_t n, uint32_t N, uint64_t wn);
void intt_naive(uint32_t *X, uint32_t n, uint32_t n_inv, uint32_t N, uint64_t wn_inv);
void ntt_recursive(uint32_t *X, size_t n, uint32_t N, uint32_t wn);
void intt_recursive(uint32_t *X, uint32_t n, uint32_t n_inv, uint32_t N, uint32_t wn_inv);
void intt_recursive_unscaled(uint32_t *X, uint32_t n, uint32_t N, uint32_t wn_inv);
void mult_numbers(uint32_t *C, const size_t c_size, uint32_t *A, const size_t a_size, uint32_t *B, const size_t b_size, uint32_t base);

void reverse_list(uint32_t *L, size_t n);

int main(int argc, char **argv)
{
  (void)argc, (void)argv;

  // 2^256 = 1238 926361 552897 * 93 461639 715357 977769 163558 199606 896584 051237 541638 188580 280321

  uint32_t A[] = {1, 238, 926, 361, 552, 897};
#define a_size (sizeof(A) / sizeof(A[0]))
  uint32_t B[] = {93, 461, 639, 715, 357, 977, 769, 163, 558, 199, 606, 896, 584, 51, 237, 541, 638, 188, 580, 280, 321};
#define b_size (sizeof(B) / sizeof(B[0]))
#define c_size 33
  uint32_t C[c_size] = {0};

  mult_numbers(C, c_size, A, a_size, B, b_size, 1000);

  printf("  ");
  for (uint32_t i = 0; i < c_size; ++i)
    if (i >= c_size - a_size)
      printf("%03u ", A[i - (c_size - a_size)]);
    else
      printf("    ");

  printf("\n* ");
  for (uint32_t i = 0; i < c_size; ++i)
    if (i >= c_size - b_size)
      printf("%03u ", B[i - (c_size - b_size)]);
    else
      printf("    ");

  printf("\n__");
  for (uint32_t i = 0; i < c_size; ++i)
    printf("____");

  printf("\n  ");

  uint32_t start = 0;
  while (C[start] == 0 && start < c_size)
    ++start, printf("    ");

  for (uint32_t i = start; i < c_size; ++i)
    printf("%03u ", C[i]);

#undef a_size
#undef b_size
#undef c_size

  return EXIT_SUCCESS;
}

#define ADDMOD(a, b, m) (((uint64_t)(a) + (uint64_t)(b)) % m)
#define SUBMOD(a, b, m) (((uint64_t)(a) - (uint64_t)(b)) % m)
#define MULMOD(a, b, m) (((uint64_t)(a) * (uint64_t)(b)) % m)
#define FMAMOD(a, b, c, m) (ADDMOD(a, MULMOD(b, c, m), m))
#define FMSMOD(a, b, c, m) (SUBMOD(a, MULMOD(b, c, m), m))

void ntt_naive(uint32_t *X, size_t n, uint32_t N, uint64_t wn)
{
  assert(powmod(wn, n, N) == 1 && "w must be a root of unity mod N");

  typeof(X[0]) *Y = calloc(n, sizeof(Y[0]));

  uint32_t w_k = 1;

  for (uint32_t k = 0; k < n; ++k)
  {
    Y[k] = 0;

    uint32_t w_kj = 1;

    for (uint32_t j = 0; j < n; ++j)
    {
      Y[k] = FMAMOD(Y[k], X[j], w_kj, N);
      w_kj = MULMOD(w_kj, w_k, N);
    }

    w_k = MULMOD(w_k, wn, N);
  }

  memcpy(X, Y, n * sizeof(X[0]));

  free(Y);

  return;
}

void intt_naive(uint32_t *X, uint32_t n, uint32_t n_inv, uint32_t N, uint64_t w_inv)
{
  typeof(X[0]) *Y = calloc(n, sizeof(Y[0]));

  uint64_t w_k = 1;

  for (uint32_t k = 0; k < n; ++k)
  {
    Y[k] = 0;

    uint64_t w_kj = 1;

    for (uint32_t j = 0; j < n; ++j)
    {
      Y[k] = FMAMOD(Y[k], X[j], w_kj, N);
      w_kj = MULMOD(w_kj, w_k, N);
    }

    Y[k] = MULMOD(Y[k], n_inv, N);

    w_k = MULMOD(w_k, w_inv, N);
  }

  memcpy(X, Y, n * sizeof(X[0]));

  free(Y);

  return;
}

/*
 * n must be a power of two
 * wn must be an nth root of unity mod N
 */
void ntt_recursive(uint32_t *X, size_t n, uint32_t N, uint32_t wn)
{
  // base case
  if (n == 1)
  {
    X[0] %= N;
    return;
  }

  // split the polinomial into odd and even coefficients
  typeof(X) X0 = calloc(n / 2, sizeof(X[0])),
            X1 = calloc(n / 2, sizeof(X[0]));
  for (uint32_t i = 0; 2 * i < n; ++i)
  {
    X0[i] = X[2 * i];
    X1[i] = X[2 * i + 1];
  }

  // compute recursive ntt
  uint32_t wn2 = MULMOD(wn, wn, N);
  ntt_recursive(X0, n / 2, N, wn2);
  ntt_recursive(X1, n / 2, N, wn2);

  // compute the output with butterfly
  uint32_t w = 1;
  for (uint32_t i = 0; 2 * i < n; ++i)
  {
    X[i] = FMAMOD(X0[i], w, X1[i], N);
    X[i + n / 2] = FMSMOD(X0[i] + N, w, X1[i], N); // + N to ensure result stays positive
    w = MULMOD(w, wn, N);
  }

  free(X0);
  free(X1);

  return;
}

void intt_recursive(uint32_t *X, uint32_t n, uint32_t n_inv, uint32_t N, uint32_t wn_inv)
{
  intt_recursive_unscaled(X, n, N, wn_inv);

  for (uint32_t i = 0; i < n; ++i)
    X[i] = MULMOD(X[i], n_inv, N);

  return;
}

// n must be a power of two
void intt_recursive_unscaled(uint32_t *X, uint32_t n, uint32_t N, uint32_t wn_inv)
{
  // base case
  if (n == 1)
  {
    X[0] %= N;
    return;
  }

  // split the polinomial into odd and even coefficients
  typeof(X) X0 = calloc(n / 2, sizeof(X[0])),
            X1 = calloc(n / 2, sizeof(X[0]));
  for (uint32_t i = 0; 2 * i < n; ++i)
  {
    X0[i] = X[2 * i];
    X1[i] = X[2 * i + 1];
  }

  // compute recursive intt
  uint32_t wn_inv2 = MULMOD(wn_inv, wn_inv, N);
  intt_recursive_unscaled(X0, n / 2, N, wn_inv2);
  intt_recursive_unscaled(X1, n / 2, N, wn_inv2);

  // compute the output with butterfly
  uint32_t w = 1;
  for (uint32_t i = 0; 2 * i < n; ++i)
  {
    X[i] = FMAMOD(X0[i], w, X1[i], N);
    X[i + n / 2] = FMSMOD((X0[i] + N), w, X1[i], N); // the + N is to make sure the result is non-negative
    w = MULMOD(w, wn_inv, N);
  }

  free(X0);
  free(X1);

  return;
}

/*
 * computes C = A*B in base `base`
 * all "digits" of A and B must be strictly less than base (and positive)
 */
void mult_numbers(uint32_t *C, const size_t c_size, uint32_t *A, const size_t a_size, uint32_t *B, const size_t b_size, uint32_t base)
{
  assert(base >= 2 && "base MUST be greater or equal to two, or else the notion of base does not make sens !!!");

  uint32_t n = 1;
  while (n < a_size + b_size)
    n <<= 1;

  assert(c_size > n && "the size of the output array must be larger !!"); // inequality must be strict in case of carry out

  uint32_t c;
  const uint64_t M = (base - 1ULL) * (base - 1ULL) * (n + 1ULL); // we need that all product of single digit number be not reduced
  assert(M <= UINT32_MAX && "Try with smaller base, the minimal working modulus is too big for current implementation");
  const uint32_t N = find_prime_modulus(&c, M, n);
  printf("N = %u\n c = %u\n", N, c);

#define FACTS_OF_PHI_N_SIZE 256ULL
  uint32_t facts_of_phi_N[FACTS_OF_PHI_N_SIZE] = {2}; // 2 because n = 2^k
  size_t num_facts_of_phi_N = factorise(facts_of_phi_N + 1, FACTS_OF_PHI_N_SIZE, c) + 1;
  // plus one for the 2 factor introduced by n = 2^k
  // facts_of_phi_N may contain the factor 2 two distinct times but that doesn't matter
  const uint32_t g = find_generator(N, N - 1, facts_of_phi_N, num_facts_of_phi_N); // because N is prime, phi(N) == N - 1
  printf("g = %u\n", g);
  assert(g != 0 && "there exist no primitive root of unity mod N");
  assert(powmod(g, N - 1, N) == 1 && "g should be a generator over N");

  /*
   * as g is a generator in F_N => g^(N - 1) === 1 (mod N)
   * => g^(N - 1)/n is an n-th primitive root of unity mod N
   * but (N - 1)/n = (c*n + 1 - 1)/n = (c*n)/n = c
   * in consequence, g^c is an n-th primitive root of unity mod N
   */
  const uint32_t wn = powmod(g, c, N);
  printf("w_n = %u\n", wn);
  assert(powmod(wn, n, N) == 1 && "w_n should be an n-th primitive root of unity");

  // resize the arrays to be a power of 2
  typeof(A) A_resized = calloc(n, sizeof(A[0])),
            B_resized = calloc(n, sizeof(B[0]));
  memcpy(A_resized + (n - a_size), A, a_size * sizeof(A[0]));
  memcpy(B_resized + (n - b_size), B, b_size * sizeof(B[0]));

  reverse_list(A_resized, n);
  reverse_list(B_resized, n);

  ntt_recursive(A_resized, n, N, wn);
  ntt_recursive(B_resized, n, N, wn);

  for (uint32_t i = 0; i < n; ++i)
    A_resized[i] = MULMOD(A_resized[i], B_resized[i], N); //(A_resized[i] * B_resized[i]) % N;

  intt_recursive(A_resized, n, mod_inv(n, N), N, mod_inv(wn, N));

  memcpy(C, A_resized, n * sizeof(A[0]));

  uint32_t carry = 0;
  for (size_t i = 0; i < n; i++)
  {
    C[i] += carry;
    carry = C[i] / base;
    C[i] %= base;
  }

  C[n] += carry;

  reverse_list(C, c_size);

  return;
}

uint32_t powmod(uint64_t a, uint64_t b, uint32_t m)
{
  if (a == 0)
    return 0;

  a %= m;
  uint64_t res = 1; // uint64_t to not overflow

  while (b > 0)
  {
    if (b & 1)
      res = MULMOD(res, a, m); // res * a % m;
    a = MULMOD(a, a, m);       // a * a % m;
    b >>= 1;
  }

  return res;
}

// returns 0 if no generator mod N exists
uint32_t find_generator(const uint32_t N, const uint32_t phi_N, const uint32_t *const facts_of_phi_N, const uint32_t num_facts)
{
  // uint32_t g = 0;
  for (uint32_t i = 2; i < phi_N; ++i)
  {
    if (powmod(i, phi_N, N) != 1)
      continue;
    /*
     * check if i is a generator mod N, if it is NOT it will have a period such that :
     * i^(phi(N)/p) === 1 (mod N)
     * with p being a factor of phi(N)
     */
    uint8_t fail = 0;
    for (uint32_t j = 0; j < num_facts; ++j)
    {
      // if (facts_of_phi_N[j] == 1)
      //   continue;

      if (powmod(i, phi_N / facts_of_phi_N[j], N) == 1)
      {
        fail = 1;
        break;
      }
    }

    if (!fail)
      return i;
  }

  return 0;
}

/*
 * finds N = c*n + 1 such that:
 *  N is prime
 *  N > max_val
 * returns c by pointer
 */
uint32_t find_prime_modulus(uint32_t *c, uint32_t max_val, uint32_t n)
{

  uint32_t mod;
  // mod = c *n + 1 ~ = max_val but mod < max_val making sure all needed values are checked
  for (*c = max_val / n - 1; *c < (UINT32_MAX / n); ++(*c))
  {
    mod = (*c) * n + 1;

    // we need to ensure mod > max_val such that for all vals <= max_val : val % mod == val
    if (mod >= max_val)
      if (is_prime(mod))
        break;
  }

  uint32_t N = (*c) * n + 1;
  return N;
}

uint8_t is_prime(uint32_t candidate)
{
  if (candidate <= 1)
    return 0;

  if (candidate == 2 || candidate == 3 || candidate == 5)
    return 1;

  if (candidate % 2 == 0 || candidate % 3 == 0 || candidate % 5 == 0)
    return 0;

  const uint32_t offsets[] = {0, 4, 6, 10, 12, 16, 22, 24};
  const size_t offsets_count = sizeof(offsets) / sizeof(offsets[0]);

  for (uint32_t i = 7; i * i < candidate; i += 30)
    for (uint32_t j = 0; j < offsets_count; ++j)
      if (candidate % (i + offsets[j]) == 0)
        return 0;

  return 1;
}

/*
 returns the number of factors of a found
 facts countains all factors of a that are strictly greater than 1
*/
uint32_t factorise(uint32_t *facts, size_t facts_size, uint32_t a)
{
  assert(facts_size >= 1 && "facts array must be biger");

  // only case where facts contains factors less than 1
  if (a == 0)
  {
    facts[0] = 0;
    return 1;
  }
  if (a == 1)
    return 0;

  size_t n = 0; // the number of spots already used in facts

  // checks if a is a multiple of 2, 3 and 5, the further 2,3 and 5 factors of a do not matter as the loop does not try number that are multiples of 2, 3 and 5
  const uint32_t spokes[3] = {2, 3, 5};
  for (uint32_t i = 0; i < 3; ++i)
  {
    if (a % spokes[i] == 0)
    {
      facts[n++] = spokes[i];
      a /= spokes[i];
      while (a % spokes[i] == 0)
        a /= spokes[i];
    }
  }

  // all of the spokes of the wheel
  const uint32_t offsets[] = {4, 2, 4, 2, 4, 6, 2, 6};
  const size_t offsets_count = sizeof(offsets) / sizeof(offsets[0]);

  uint32_t i = 0;
  for (uint32_t d = 7; d * d < a; d += offsets[i++])
  {
    if (a % d == 0)
    {
      assert(facts_size >= n + 1 && "the facts array must be bigger");
      facts[n++] = d;
      a /= d;

      while (a % d == 0)
        a /= d;
    }

    if (i == offsets_count)
      i = 0;
  }

  if (a > 1)
    facts[n++] = a;

  return n;
}
/*
 source : https://cp-algorithms.com/algebra/module-inverse.html
 * m MUST be prime or else the algorithm will provide wrong results !!!
 * and a MUST be s.t. 0 < a < m
 */
uint32_t mod_inv(uint64_t a, uint32_t m)
{
  return a <= 1 ? a : m - (uint64_t)(m / a) * mod_inv(m % a, m) % m;
}

void reverse_list(uint32_t *L, size_t n)
{
  uint32_t t;
  for (size_t i = 0; i < n / 2; ++i)
  {
    t = L[i];
    L[i] = L[n - i - 1];
    L[n - i - 1] = t;
  }

  return;
}