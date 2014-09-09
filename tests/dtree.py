import pandas as pd
from ..dtree import build_tree, parse_input, classification_accuracy, tree_stats, prune_top_down, prune_bottom_up

df_train = parse_input('data/hw1/noisy10_train.ssv')
df_test = parse_input('data/hw1/noisy10_test.ssv')
df_valid = parse_input('data/hw1/noisy10_valid.ssv')

# df_tab32 = pd.read_csv('data/hw1/mitchell_tab3.2.csv')
# info_gain(df_tab32, 'playtennis', 'humidity') # 0.151



acc = [
    (0,'a','m'),
    (0,'a','m'),
    (0,'a','n'),
    (0,'a','n'),
    (1,'b','m'),
    (1,'b','m'),
    (1,'b','n'),
    (1,'b','n'),
    ]
df_samp = pd.DataFrame(acc, columns=['y','x1','x2'])
tree1 = build_tree(df_samp, ['x1','x2'], 'y')


acc = [
    (0,'a','m'),
    (0,'a','m'),
    (0,'a','m'),
    (0,'a','m'),
    (1,'a','n'),
    (1,'a','n'),
    (1,'a','n'),
    (1,'a','n'),
    (1,'b','m'),
    (1,'b','m'),
    (1,'b','m'),
    (1,'b','m'),
    ]
df_samp2 = pd.DataFrame(acc, columns=['y','x1','x2'])
tree2 = build_tree(df_samp2, ['x1','x2'], 'y')




# degenerate
acc = [
    (0,'a','m'),
    (0,'b','n'),
    (0,'c','o'),
    (1,'d','p'),
    (1,'e','q'),
    (1,'f','r'),
]
df_samp3 = pd.DataFrame(acc, columns=['y','x1','x2'])
tree3 = build_tree(df_samp3, ['x1','x2'], 'y')

acc = [
    (1,'a','m'),
    ]
df_samp4 = pd.DataFrame(acc, columns=['y','x1','x2'])
tree4 = build_tree(df_samp4, ['x1','x2'], 'y')
# TODO: output an attribute?

_, _, _, _, _ = classification_accuracy(tree4, df_samp4, df_samp4, attr_target='y')
n_node, depth = tree_stats(tree4)
assert depth == 0
assert n_node == 1



# acc = []
# df_samp5 = pd.DataFrame()
# tree5 = build_tree(df_samp5, [], '') # doesn't handle this

acc = [
    (0,'a','m','x'),
    (0,'a','m','x'),
    (1,'b','m','y'),
    (1,'b','m','y'),
    (1,'b','n','x'),
    (1,'b','n','x'),
    (0,'b','n','y'),
    (0,'b','n','y'),
    (0,'b','n','y'),
    ]
df_samp6 = pd.DataFrame(acc, columns=['y','x1','x2','x3'])
tree6 = build_tree(df_samp6, ['x1','x2','x3'], 'y')
# print_tree(tree6, attr_target='y')

acc_test = [
    (0,'a','m','x'),
    (0,'a','m','x'),
    (0,'b','m','y'), # only this changed (1 \to 0)
    (1,'b','m','y'),
    (1,'b','n','x'),
    (1,'b','n','x'),
    (0,'b','n','y'),
    (0,'b','n','y'),
    (0,'b','n','y'),
    ]
df_samp6_test = pd.DataFrame(acc_test, columns=['y','x1','x2','x3'])
# tree6_test = build_tree(df_samp6_test, ['x1','x2','x3'], 'y')
# print_tree(tree6, attr_target='y')

_, _, _, _, _ = classification_accuracy(tree6, df_samp6, df_samp6_test, attr_target='y')
n_node, depth = tree_stats(tree6)
assert depth == 3
assert n_node == 7

tree6_prune_top, df_stats_6_top = prune_top_down(tree6, df_samp6, df_samp6_test, attr_target='y', epsilon=0.005)
tree6_prune_bottom, df_stats_6_bottom = prune_bottom_up(tree6, df_samp6, df_samp6_test, attr_target='y', epsilon=0.005)

_, _, _, _, _ = classification_accuracy(tree6_prune_top, df_samp6, df_samp6_test, attr_target='y')
_, _, _, _, _ = classification_accuracy(tree6_prune_bottom, df_samp6, df_samp6_test, attr_target='y')
    # (0,'b','m','y'), # only this changed (1 \to 0)
    # (1,'b','m','y'),
# TODO: there shouldn't be 100% test accuracy...
# classification_accuracy() ignores where there is no value? (should it be doing this?, or should it classify irregardless ...)
# because in the bottom prune, (b,m,*) is missing. It should go per entry in data, and if there are no further children, use the current node. This will handle the case where the tree is not complete over the input space (ie. there is no node for the entire x_1,\dots,x_n sequence), and will allow for only partial sequences.



import copy
root = copy.deepcopy(tree6)
acc_stats = []
epsilon = 0.005
attr_target='y'
from dtree import __prune
# %lprun -f __prune __prune(tree6, 0, 0, root, df_samp6, df_samp6_test, attr_target, epsilon, acc_stats)
