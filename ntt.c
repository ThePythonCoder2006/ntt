#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <complex.h>

#define INP 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, \
            1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4

#define LEN (sizeof((char[]){INP}))

uint32_t
powmod(uint32_t base, uint32_t exp, uint32_t m);
uint32_t find_primitive_root(const uint32_t m, const uint32_t phi_m, const uint32_t *const facts_of_phi_n, const uint32_t num_facts);
uint32_t find_prime_modulus(uint32_t max_val, uint32_t n);
uint8_t is_prime(uint32_t candidate);

void ntt_naive(uint32_t *X, size_t n, uint32_t N, uint32_t w);
void intt_naive(uint32_t *X, size_t n, size_t n_inv, uint32_t N, uint32_t wn_inv);
void ntt_recursive(uint32_t *X, size_t n, uint32_t N, uint32_t w);
void intt_recursive(uint32_t *X, size_t n, uint32_t N, uint32_t wn_inv);

int main(int argc, char **argv)
{
  (void)argc, (void)argv;

  uint32_t X_naive[LEN] = {INP};
  uint32_t X_recursive[LEN] = {INP};

  printf("n = %u\n", (uint32_t)LEN);

  const uint32_t M = 5;

  const uint32_t N = find_prime_modulus(5, LEN);
  printf("N = %u\n", N);

  const uint32_t facts_of_phi_N[] = {2, FACTS_OF_K};
  assert(N >= LEN && "N must be bigger than n");
  assert(N >= M && "N must be bigger or equal to M");

  uint32_t g = find_primitive_root(N, N - 1, facts_of_phi_N, num_of_facts_of_k + 1);
  assert(g != 0 && "there exist no primitive root of unity mod N");

  const uint32_t w = powmod(g, k, N);
  assert(powmod(w, LEN, N) == 1 && "w must an n-th root of unity modulo N");
  printf("g = %u\n"
         "w = %u\n",
         g, w);

  printf("naive start\n");
  fflush(stdout);
  ntt_naive(X_naive, LEN, N, w);
  printf("naive end\n");
  printf("recursive start\n");
  fflush(stdout);
  ntt_recursive(X_recursive, LEN, N, w);
  printf("recursive end\n");

  // printf("idx|inp \tniv \trec\n");
  // for (uint8_t i = 0; i < LEN; ++i)
  // printf("%2u | %2u \t %2u \t %2u\n", i, X[i], X_naive[i], X_recursive[i]);
  // if (X_naive[i] != X_recursive[i])
  // printf("[WARNING] the two results were not the same : X_naive[%u] = %u != %u = X_recursive[%u]\n", i, X_naive[i], i, X_recursive[i]);
  // putchar('\n');

  return EXIT_SUCCESS;
}

void ntt_naive(uint32_t *X, size_t n, uint32_t N, uint32_t w)
{
  assert(powmod(w, n, N) == 1 && "w must be a root of unity mod N");

  typeof(X[0]) *Y = calloc(n, sizeof(Y[0]));

  for (uint32_t k = 0; k < n; ++k)
  {
    Y[k] = 0;

#if __DEBUG__
    printf("%2u| ", k);
#endif

    for (uint32_t j = 0; j < n; ++j)
    {
      uint32_t w_kj = powmod(w, k * j, N);
      Y[k] = (Y[k] + (X[j] * w_kj)) % N;

#if __DEBUG__
      printf("%2u*%-2u ", w_kj, X[j]);
#endif // __DEBUG__
    }

#if __DEBUG__
    putchar('\n');
#endif // __DEBUG__
  }

  memcpy(X, Y, n * sizeof(X[0]));

  free(Y);

  return;
}

void intt_naive(uint32_t *X, size_t n, size_t n_inv, uint32_t N, uint32_t w_inv)
{
  for (uint32_t k = 0; k < n; ++k)
  {
    uint32_t acc = 0;
    for (uint32_t i = 0; i < n; ++i)
      acc = (acc + (X[i] * powmod(w_inv, i * k, N))) % N;
    acc = (acc * n_inv) % N;
  }

  return;
}

// n must be a power of two
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
  ntt_recursive(X0, n / 2, N, wn * wn % N);
  ntt_recursive(X1, n / 2, N, wn * wn % N);

  // compute the output with butterfly
  uint32_t w = 1;
  for (uint32_t i = 0; 2 * i < n; ++i)
  {
    X[i] = (X0[i] + w * X1[i]) % N;
    X[i + n / 2] = ((X0[i] + N) - ((w * X1[i]) % N)) % N;
    w = (w * wn) % N;
  }

  free(X0);
  free(X1);

  return;
}

// n must be a power of two
void intt_recursive(uint32_t *X, size_t n, uint32_t N, uint32_t wn_inv)
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
  ntt_recursive(X0, n / 2, N, wn_inv * wn_inv % N);
  ntt_recursive(X1, n / 2, N, wn_inv * wn_inv % N);

  // compute the output with butterfly
  uint32_t w = 1;
  for (uint32_t i = 0; 2 * i < n; ++i)
  {
    X[i] = (X0[i] + w * X1[i]) % N;
    X[i + n / 2] = ((X0[i] + N) - ((w * X1[i]) % N)) % N;
    w = (w * wn_inv) % N;

    // divide by two n times to divide by n in total
    X[i] /= 2;
    X[i + n / 2] /= 2;
  }

  free(X0);
  free(X1);

  return;
}

#if 0
void mult_numbers(uint32_t *C, const size_t c_size, uint32_t *A, const size_t a_size, uint32_t *B, const size_t b_size)
{
  uint32_t n = 1;
  while (n < a_size + b_size)
    n <<= 1;

  assert(n >= c_size && "the size of the output array must be larger !!");

  // resize the arrays to be a power of 2
  typeof(A) A_resized = calloc(n, sizeof(A[0])),
            B_resized = calloc(n, sizeof(B[0]));
  memcpy(A_resized, A, a_size * sizeof(A[0]));
  memcpy(B_resized, B, b_size * sizeof(B[0]));

  ntt_recursive(A_resized, n, );
  ntt_recursive(B_resized, n, );

  for (uint32_t i = 0; i < n; ++i)
    A_resized[i] *= B_resized[i];

  intt_recursive(A_resized, n, );

  uint32_t carry = 0;
  for (int i = 0; i < n; i++)
  {
    C[i] += carry;
    carry = C[i] / 10;
    C[i] %= 10;
  }

  return;
}
#endif

uint32_t powmod(uint32_t base, uint32_t exp, uint32_t m)
{
  if (exp == 0)
    return 1;
  uint64_t res = 1; // 64 bits to not worry about mult overflowing

  for (uint32_t i = 0; i < exp; ++i)
    res = (res * base) % m;

  return res;
}

// returns 0 if no root exists
uint32_t find_primitive_root(const uint32_t m, const uint32_t phi_m, const uint32_t *const facts_of_phi_n, const uint32_t num_facts)
{
  uint32_t g = 0;
  for (uint32_t i = 2; i < m; ++i)
  {
    /*
     * check if i is a primitive root unity, if it is NOT it will have a period such that :
     * i^(phi(N)/p) === 1 (mod N)
     * with p being a factor of phi(N)
     */
    uint8_t fail = 0;
    for (uint32_t j = 0; j < num_facts; ++j)
    {
      if (facts_of_phi_n[j] == 1)
        continue;

      if (powmod(i, phi_m / facts_of_phi_n[j], m) == 1)
      {
        fail = 1;
        break;
      }
    }

    if (!fail)
      g = i;
  }

  return g;
}

uint32_t find_prime_modulus(uint32_t max_val, uint32_t n)
{
  uint32_t k = 0;

  uint32_t mod = 1; // 0 * n + 1
  do
  {
    ++k;
    mod = k * n + 1;
    if (mod <= max_val)
      continue;
    // printf("N = %u\n", mod);
  } while (!is_prime(mod));

  uint32_t N = k * n + 1;
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
      {
        printf("%u %% %u == 0\n", candidate, i + offsets[j]);
        return 0;
      }

  return 1;
}

void factorise(uint32_t *facts, size_t facts_size, uint32_t a)
{
  assert(facts_size >= 1 && "facts array must be biger");

  if (is_prime(a))
  {
    facts[0] = a;
    return;
  }

  if (a == 0 || a == 1)
  {
    facts[0] = a;
    return;
  }

  if (a == 2 || a == 3 || a == 5)
    return;

  if (a % 2 == 0 || a % 3 == 0 || a % 5 == 0)
    return;

  const uint32_t offsets[] = {0, 4, 6, 10, 12, 16, 22, 24};
  const size_t offsets_count = sizeof(offsets) / sizeof(offsets[0]);

  size_t n = 0; // the number of spots already used in facts

  for (uint32_t i = 7; i * i < a; i += 30)
    for (uint32_t j = 0; j < offsets_count; ++j)
      if (a % (i + offsets[j]) == 0)
      {
        assert(facts_size >= n + 1 && "the facts array must be bigger");
        facts[n + 1] = i + offsets[j];
      }

  return;
}