import pandas as pd
from scipy.stats import entropy
import numpy as np

SEP_SSV=' '

class Node(object):
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
            child = build_tree(df_attr_val_sel, [x for x in df_attr_val_sel.columns if x not in [attr_target,best_attr]], attr_target)
            child.attribute_value = val
            root.children.append(child)
    return root


def classification_accuracy(tree, df_train_in, df_test_in, attr_target):
    def _ca(tree, df_train, df_test):
        if not tree.children: # is a leaf
            label_train = df_train[attr_target].value_counts().argmax()
            label_test = df_test[attr_target].value_counts().argmax()
            return [[tree.label, label_train, label_test]]

        ## not a leaf
        acc = []
        for child in tree.children:
            attr = tree.attribute
            attr_val = child.attribute_value
            res = _ca(child, df_train[df_train[attr] == attr_val], df_test[df_test[attr] == attr_val])
            acc.extend(res) # to flatten the tree output
        return acc

    res = _ca(tree, df_train_in, df_test_in)
    mtx_train = pd.DataFrame(np.zeros((2,2)), columns=['1','0'], index=['1','0'])
    mtx_train.index.name = 'out'
    mtx_test = mtx_train.copy()
    def _sel(label_tree, label):
        return (str(label_train) if label_tree == label else '0'), ('1' if label_tree in [1,'1'] else '0')
    for label_tree, label_train, label_test in res:
        mtx_train.loc[_sel(label_tree, label_train)] += 1
        mtx_test.loc[_sel(label_tree, label_test)] += 1

    return res, mtx_train, mtx_test


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
