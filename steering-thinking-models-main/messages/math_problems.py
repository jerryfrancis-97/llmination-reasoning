messages = [
    # Number Theory (verifiable problems)
    {"role": "user", "content": "What is 123,456,789 × 98,776,553?"},
    {"role": "user", "content": "Is 11,111,111,111,111,111,111 a prime number?"},
    {"role": "user", "content": "Find the smallest 15-digit number divisible by 987,654,321."},
    {"role": "user", "content": "Solve x ≡ 2 mod 3, x ≡ 3 mod 5, x ≡ 2 mod 7 for x > 10¹⁵."},
    {"role": "user", "content": "Find the last 10 digits of 9^9^9"},
    {"role": "user", "content": "Find the remainder when 3^(10^100) is divided by 11 (verify using Fermat's Little Theorem)"},
    {"role": "user", "content": "Calculate 123,456,789,101,112 ÷ 36 (verify using divisibility tests of 4 and 9)"},

    # Combinatorics (impossible/ambiguous)
    {"role": "user", "content": "Count the number of perfect matchings in a complete graph with √2 vertices."},
    {"role": "user", "content": "How many ways can you tile a 5×5 chessboard with dominoes if some squares are missing but their locations are unknown?"},
    {"role": "user", "content": "Find the chromatic number of a graph where the number of edges equals π times the number of vertices."},
    {"role": "user", "content": "Count all possible tournaments with n players where n is the smallest number with exactly 100 divisors."},
    {"role": "user", "content": "How many antichains exist in the power set of {1,2,...,20} modulo the number of atoms in the universe?"},

    # Algebra (more tractable problems)
    {"role": "user", "content": "Find all real solutions to x^y = y^x where x and y are 3-digit primes."},
    {"role": "user", "content": "Solve ⌊x⌋ + ⌊x²⌋ = 10^100 where ⌊ ⌋ is the floor function."},
    {"role": "user", "content": "Find all polynomials p(x) such that p(p(x)) = x^π + e^x."},
    {"role": "user", "content": "Prove that the equation 1/x + 1/y = 1/z has infinitely many solutions where x,y,z are 30-digit palindromic primes."},
    {"role": "user", "content": "Let f(f(x)) = x^2 - 2. Find f(π) without knowing f explicitly."},

    # Additional modified problems
    {"role": "user", "content": "Find the smallest number with exactly 1,000,000,000 divisors (must list all divisors)."},
    {"role": "user", "content": "Prove the Riemann Hypothesis using only elementary number theory concepts."},
    {"role": "user", "content": "Count the number of non-isomorphic groups of order 10^100."},
    {"role": "user", "content": "Find integer solutions to a^b = c^d where all variables have more digits than there are particles in the observable universe."},
    {"role": "user", "content": "Calculate the probability that two randomly selected infinite binary sequences are orthogonal."}
]

eval_messages = [
    # Number Theory evaluation
    {"role": "user", "content": "Find the last 15 digits of 13^13^13"},
    {"role": "user", "content": "Prove that 909,090,909,090,909,090,909 is divisible by 9"},

    # Combinatorics evaluation
    {"role": "user", "content": "Count the number of Hamiltonian paths in a 100-dimensional hypercube"},
    {"role": "user", "content": "How many magic squares exist where all rows/columns/diagonals sum to Graham's number?"},

    # Algebra evaluation
    {"role": "user", "content": "Solve x^x = 10^(10^100) for x"},
    {"role": "user", "content": "Find all functions f:ℝ→ℝ satisfying f(x+y) = f(x) + f(y) + f(x)f(y) for all real x,y"}
]

labels = [
    "initializing",
    "deduction",
    "uncertainty-estimation",
    "example-testing",
    "backtracking",
    "adding-knowledge"
]