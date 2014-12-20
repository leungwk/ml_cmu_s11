from ..chain import gen_test_data, calc_xn_x1
import numpy as np

"""
Under "01" for epsilon = 0.01

P(X_3=1|X_1=1)
&= \frac{1}{P(X_1=1)} P(X_3=1,X_1=1)\\
&= \frac{1}{P(X_1=1)} \sum_{x_2} P(X_2=x_2,X_3=1,X_1=1)\\
&= \frac{1}{P(X_1=1)} \sum_{x_2} P(X_1=1)P(X_2=x_2|X_1=1)P(X_3=1|X_2=x_2)\\
&= \sum_{x_2 \in \set{0,1}} P(X_2=x_2|X_1=1)P(X_3=1|X_2=x_2)\\
&= 0.9802

P(X_4=1|X_1=1)
&= \frac{1}{P(X_1=1)} P(X_4=1,X_1=1)\\
&= \frac{1}{P(X_1=1)} \sum_{x_2,x_3} P(X_2=x_2,X_3=x_3,X_4=1,X_1=1)\\
&= \frac{1}{P(X_1=1)} \sum_{x_2,x_3} P(X_1=1)P(X_2=x_2|X_1=1)P(X_3=x_3|X_2=x_2)P(X_4=1|X_3=x_3)\\
&= \sum_{x_2,x_3 \in \set{0,1}} P(X_2=x_2|X_1=1)P(X_3=x_3|X_2=x_2)P(X_4=1|X_3=x_3)\\
&= \sum_{x_2} P(X_2=x_2|X_1=1) \sum_{x_3} P(X_3=x_3|X_2=x_2)P(X_4=1|X_3=x_3)\\
&= \sum_{x_2} P(X_2=x_2|X_1=1) [P(X_3=1|X_2=x_2)P(X_4=1|X_3=1) +P(X_3=0|X_2=x_2)P(X_4=1|X_3=0)]\\
&= P(X_2=1|X_1=1) [P(X_3=1|X_2=1)P(X_4=1|X_3=1) +P(X_3=0|X_2=1)P(X_4=1|X_3=0)]\\
&+ P(X_2=0|X_1=1) [P(X_3=1|X_2=0)P(X_4=1|X_3=1) +P(X_3=0|X_2=0)P(X_4=1|X_3=0)]\\
&= 0.99 [0.99*0.99 +0.01*0.01]\\
&+ (1-0.99) [0.01*0.99 +(1-0.01)*0.01]
&= 0.970596
"""

kind = '01'
seed = 0
epsilon = 0.01

for n_var, res in [[2, 0.99], [3, 0.9802], [4, 0.970596]]:
    df_px = gen_test_data(n_var=n_var, kind=kind, seed=seed, epsilon=epsilon)
    f, df_stats = calc_xn_x1(df_px)
    np.testing.assert_almost_equal(res, f(1))

df_px = gen_test_data(n_var=2, kind=kind, seed=seed, epsilon=epsilon)
f, df_stats = calc_xn_x1(df_px)
# a chain of length two should involve only a table lookup, and no calculations
np.testing.assert_almost_equal(df_px.loc[2,1], f(1))
assert 0.9802 != f(1) # a value if there are one too many iterations 
