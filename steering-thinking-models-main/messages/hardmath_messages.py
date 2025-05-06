messages = [
    # Mathematical Logic
    {"role": "user", "content": "Mia saw a 240-foot shark with 4 36-inch remoras attached to it. What percentage of the shark's body length is the combined length of the remoras?"},
    {"role": "user", "content": "Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor of $97_b.$"},
    {"role": "user", "content": "Let $n \geqslant 3$ be a positive integer. Furthermore, let $x_1, x_2,\ldots, x_n \in [0, 2]$ be real numbers subject to $x_1 + x_2 +\cdots + x_n = 5$. Prove the inequality$$x_1^2 + x_2^2 + \cdots + x_n^2 \leqslant 9.$$When does equality hold?"},
]

labels = [
    "initializing",
    "deduction",
    "uncertainty-estimation",
    "example-testing",
    "backtracking",
    "adding-knowledge",
    "checking",
    "separate"
]