#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

void print_mults(uint32_t base, uint32_t *primes_before, uint32_t primes_before_count);

int main(int argc, char **argv)
{
  (void)argc, (void)argv;
  print_mults(7, (uint32_t[]){2, 3, 5}, 3);
  return 0;
}

void print_mults(uint32_t base, uint32_t *primes_before, uint32_t primes_before_count)
{
  for (uint32_t i = 0; 7 * i < 1000; ++i)
  {
    printf("%4u\t", i * base);
    for (uint32_t j = 0; j < primes_before_count; ++j)
    {
      uint32_t prime = primes_before[primes_before_count - j - 1];
      if ((i * base) % prime == 0)
      {
        for (uint32_t k = 0; k < prime; ++k)
          putchar(' ');
        printf("%-3u", prime);
        break;
      }
    }
    putchar('\n');
  }
  return;
}