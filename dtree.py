import pandas as pd
from scipy.stats import entropy
import copy
import numpy as np
import time as pytime

SEP_SSV=' '

class Node(object):
    """Nodes used to construct the tree. Stores the attribute and value splitted upon (from the parent), and the predicted label"""
    def __init__(self, children=None, parent=None, data=None, attribute=None, attribute_value=None, label=None):
        self.children = children if children is not None else []
        self.parent = parent # for going up
        self.data = data
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.label = label


def print_tree(root, attr_target):
    def _vals(df, attr_target):
        d = df[attr_target].value_counts().to_dict()
        return d.get('0', d.get(0, None)), d.get('1', d.get(1, None))

    return [root.attribute, root.label, root.attribute_value, _vals(root.data, attr_target), [print_tree(child, attr_target) for child in root.children]]


def tree_stats(root):
    """Starting at root, return the number of nodes in the tree and its depth"""
    def _ts(tree, depth):
        if not tree.children: # is a leaf
            return 1, depth

        n_node = 1
        max_depth = depth
        for child in tree.children:
            nn, ret_d = _ts(child, depth+1)
            max_depth = max(max_depth, ret_d)
            n_node += nn
        return n_node, max_depth
    return _ts(root, 0)


def parse_input(filepath):
    """Read their ssv format (see data/hw1/readme.dt)"""
    acc = []
    with open(filepath, 'r') as pile:
        n_attr, _ = pile.readline().strip().split(SEP_SSV)
        header = pile.readline().strip().split(SEP_SSV)
        pile.readline()
        for line in pile:
            acc.append( line.strip().split(SEP_SSV) )
    df = pd.DataFrame(acc, columns=header)
    return df


def info_gain(df_root, attr_target, attr_sel):
    root_counts = df_root[attr_target].value_counts()
    ent_root = entropy(root_counts, base=2) # H(S) # handles binary, whether '0'/'1' or 'no'/'yes'
    size_s = root_counts.sum()

    ## count positive/negative examples of attr_target by values of attr_sel
    sel = [attr_sel,attr_target]
    df_g = df_root[sel].groupby(sel).aggregate(len)

    ## calculate entropy(root|attr_sel) as \sum_v \frac{|S_v|}{|S|} H(S_v) where v is the value of the attribute (attr_sel)
    ent_cond = 0 # for H(S|S_v)
    for val in set(df_g.index.get_level_values(attr_sel)):
        value_labels = df_g.xs(val, level=attr_sel)
        size_s_v = value_labels.sum()
        ent_s_given_v = (1.*size_s_v/size_s) * entropy(value_labels, base=2)
        ent_cond += ent_s_given_v
    info_gain = ent_root -ent_cond
    return info_gain


def arg_and_max(arg_vals):
    if not arg_vals:
        return (None, None)
    return sorted(arg_vals, key=lambda r: r[1], reverse=True)[0]


def build_tree(df_train, attr_tests, attr_target):
    """ID3"""
    label_counts = df_train[attr_target].value_counts()
    best_label = label_counts.argmax()
    label_counts = label_counts.to_dict() # .to_dict() because of a bug with Index in pandas 0.14.1 (compared to Int64Index)
    root = Node(data=df_train, label=best_label) # create a root for this tree
    ## check if all of one label
    if not label_counts.get(0, label_counts.get(str(0))): # 0 "0"s
        root.label = 1
        return root
    elif not label_counts.get(1, label_counts.get(str(1))): # 0 "1"s
        root.label = 0
        return root

    if not attr_tests:
        root.label = best_label
        return root

    best_attr, _ = arg_and_max([
        (attr_sel,info_gain(df_train, attr_target, attr_sel))
        for attr_sel in attr_tests])
    root.attribute = best_attr
    value_counts = df_train[best_attr].value_counts()
    for val in value_counts.index:
        df_attr_val_sel = df_train[df_train[best_attr] == val]
        if len(df_attr_val_sel) == 0:
            leaf = Node(parent=root, attribute_value=val)
            root.children.append(leaf)
        else:
            child = build_tree(
                df_attr_val_sel,
                [x for x in df_attr_val_sel.columns if x not in [attr_target,best_attr]], # select the remaining attributes (ie. use each attribute at most once)
                 attr_target)
            child.attribute_value = val
            root.children.append(child)
    return root


def classification_accuracy(root, df_train_in, df_test_in, attr_target):
    """calculate classification accuracy for train and test data at the same time so to traverse only once"""
    def _bin_count(counts):
        return counts.get('0', counts.get(0, 0)), counts.get('1', counts.get(1, 0))        

    def _ca(tree, df_train, df_test, depth):
        if not tree.children: # is a leaf
            tmp_ser_train_vals = df_train[attr_target].value_counts()
            tmp_ser_test_vals = df_test[attr_target].value_counts()
            label_train, label_test = None, None
            val_0_train, val_1_train = 0,0
            val_0_test, val_1_test = 0,0
            if not tmp_ser_train_vals.empty:
                label_train = tmp_ser_train_vals.argmax()
                val_0_train, val_1_train = _bin_count(tmp_ser_train_vals.to_dict()) # count the exact number
            if not tmp_ser_test_vals.empty:
                label_test = tmp_ser_test_vals.argmax()
                val_0_test, val_1_test = _bin_count(tmp_ser_test_vals.to_dict())

            return [[tree.label, label_train, label_test, np.array([val_0_train, val_1_train]), np.array([val_0_test, val_1_test])]], [depth, 1]

        ## not a leaf
        acc = []
        max_depth = 0
        n_node = 1 # for the inner node
        for child in tree.children:
            attr = tree.attribute
            attr_val = child.attribute_value
            res, (res_depth, cnt_child) = _ca(child, df_train[df_train[attr] == attr_val], df_test[df_test[attr] == attr_val], depth+1)
            max_depth = max(max_depth, res_depth)
            n_node += cnt_child
            acc.extend(res) # to flatten the tree output
        return acc, [max_depth, n_node]

    res, (res_depth, n_node) = _ca(root, df_train_in, df_test_in, 0)
    mtx_lab_train = pd.DataFrame(np.zeros((2,2)), columns=['1','0'], index=['1','0'])
    mtx_lab_train.index.name = 'out' # output value, not the base label
    mtx_lab_test = mtx_lab_train.copy()

    mtx_cnt_train, mtx_cnt_test = mtx_lab_train.copy(), mtx_lab_train.copy()
    def _sel(label_tree, label):
        return (str(label) if str(label_tree) == str(label) else '0'), str(label_tree)
    for label_tree, label_train, label_test, cnt_train, cnt_test in res:
        ## count the predicted label
        mtx_lab_train.loc[_sel(label_tree, label_train)] += 1
        mtx_lab_test.loc[_sel(label_tree, label_test)] += 1

        ## count per example
        mtx_cnt_train.loc[str(label_tree), '0'] += cnt_train[0]
        mtx_cnt_train.loc[str(label_tree), '1'] += cnt_train[1]

        mtx_cnt_test.loc[str(label_tree), '0'] += cnt_test[0]
        mtx_cnt_test.loc[str(label_tree), '1'] += cnt_test[1]

    return res, mtx_lab_train, mtx_lab_test, mtx_cnt_train, mtx_cnt_test


def _ca(df_mtx):
    denom = long(df_mtx.sum().sum())
    ac = 1.*(df_mtx.loc['1','1'] +df_mtx.loc['0','0'])/denom if denom > 0 else None
    return ac


def prune_top_down(tree_in, df_train_in, df_valid_in, attr_target, epsilon=0.005):
    """Prune by comparing the pre- and post- prune validation accuracy"""
    root = copy.deepcopy(tree_in) # classification_accuracy() expects the root, even if nodes below have been pruned; use deepcopy to ease destructive updates
    acc_stats = []
    def _prune(tree, branch_num, depth):
        if not tree.children: # is a leaf
            ## don't check and return
            return

        ## depth-first traversal (check before recur); destructive
        ## before pruning
        _, _, _, _, df_valid_old = classification_accuracy(root, df_train_in, df_valid_in, attr_target) # use "root" because classification_accuracy() starts from the top and might visit every branch; that df_train_in is expected should not add too much unneeded overhead even though its results are not used
        ## after pruning
        tmp_els = tree.children
        tree.children = []
        _, _, _, _, df_valid = classification_accuracy(root, df_train_in, df_valid_in, attr_target)
        
        ac_old, ac = _ca(df_valid_old), _ca(df_valid)
        acc_stats.append( (ac_old, ac, branch_num, depth) )
        if ac -ac_old >= epsilon:
            pass # ie. prune
        else:
            tree.children = tmp_els # add back values
            for idx, child in enumerate(tree.children): # recur
                _prune(child, idx, depth+1)
    _prune(root, 0, 0)
    return root, pd.DataFrame(acc_stats, columns=['ac_b','ac_a','cidx','depth'])


def prune_bottom_up(tree_in, df_train_in, df_valid_in, attr_target, epsilon=0.005):
    """Prune by comparing the pre- and post- prune validation accuracy"""
    if not tree_in.children: # is a stump
        return tree_in, pd.DataFrame([])

    root = copy.deepcopy(tree_in) # see prune_top_down
    acc_stats = []
    def _prune(tree, branch_num, depth):
        idx = 0
        while True:
            try:
                child = tree.children[idx]
            except IndexError:
                break # end of list

            if not child.children: # child is a leaf, so check as if one recurred
                ## both classification_accuracy() s take up all the time
                ## before
                _, _, _, _, df_valid_old = classification_accuracy(root, df_train_in, df_valid_in, attr_target)
                ## after
                del tree.children[idx]
                _, _, _, _, df_valid = classification_accuracy(root, df_train_in, df_valid_in, attr_target)
                
                ac_old, ac = _ca(df_valid_old), _ca(df_valid)
                acc_stats.append( (ac_old, ac, branch_num, depth) )
                if ac -ac_old >= epsilon:
                    pass # ie. prune (don't add child back)
                    ## idx now at the next item after `child'
                else: # add child back
                    tree.children.insert(idx, child)
                    idx += 1 # move to the next item
            else: # recur
                _prune(child, idx, depth+1) # could potentially remove all children of this child
                idx += 1
    _prune(root, 0, 0)
    return root, pd.DataFrame(acc_stats, columns=['ac_b','ac_a','cidx','depth'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', help='input file')
    parser.add_argument('--test', help='input file')
    parser.add_argument('--valid', help='input file')
    args = parser.parse_args()

    df_train = parse_input(args.train)
    df_test = parse_input(args.test)
    df_valid = parse_input(args.valid)

    attr_target = 'poisonous'
    attr_tests = [x for x in df_train.columns if x != attr_target]

    tree = build_tree(df_train, attr_tests, attr_target)
    _, _, _, mtx_cnt_train, mtx_cnt_test = classification_accuracy(tree, df_train, df_test, attr_target=attr_target)

    ## training and testing
    ## accuracy without pruning
    _ca(mtx_cnt_train), _ca(mtx_cnt_test)
    tree_stats(tree)


    ## experiment with different \epsilons
    acc_stats_top = []
    acc_stats_bottom = []
    epsilons = [0.001,0.002,0.003,0.004,0.006,0.008,0.009,0.010,0.012,0.014,0.015,0.016,0.020,0.024,0.028,0.032,0.064]
    for epsilon in epsilons:

        start_time = pytime.time()
        tree_prune_bottom, df_stats_bottom_prune = prune_bottom_up(tree, df_train, df_valid, attr_target, epsilon=epsilon)
        _, _, _, mtx_cnt_train_bottom_prune, mtx_cnt_test_bottom_prune = classification_accuracy(tree_prune_bottom, df_train, df_test, attr_target=attr_target)
        ca_bottom_prune_train, ca_bottom_prune_test = _ca(mtx_cnt_train_bottom_prune), _ca(mtx_cnt_test_bottom_prune)
        n_node, depth = tree_stats(tree_prune_bottom)
        end_time = pytime.time()

        row = (epsilon, ca_bottom_prune_train, ca_bottom_prune_test, n_node, depth, end_time -start_time)
        acc_stats_bottom.append( row )
        print ('bottom', row)


        start_time = pytime.time()
        tree_prune_top, df_stats_top_prune = prune_top_down(tree, df_train, df_valid, attr_target, epsilon=epsilon)
        _, _, _, mtx_cnt_train_top_prune, mtx_cnt_test_top_prune = classification_accuracy(tree_prune_top, df_train, df_test, attr_target=attr_target)
        # training and testing
        # accuracy with top-down pruning
        ca_prune_train, ca_prune_test = _ca(mtx_cnt_train_top_prune), _ca(mtx_cnt_test_top_prune)
        n_node, depth = tree_stats(tree_prune_top)
        end_time = pytime.time()

        row = (epsilon, ca_prune_train, ca_prune_test, n_node, depth, end_time -start_time)
        acc_stats_top.append( row )
        print ('top', row)

    df_prune_top_stats_tot = pd.DataFrame(acc_stats_top, columns=['e','cap_train','cap_test','n','d','t'])
    df_prune_bottom_stats_tot = pd.DataFrame(acc_stats_bottom, columns=['e','cap_train','cap_test','n','d','t'])

    print df_prune_top_stats_tot
    print df_prune_bottom_stats_tot
