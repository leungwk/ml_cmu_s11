import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def train_naive_bayes(df_train_data, df_train_labels, df_vocab, df_ng_labels, alpha):
    """"""

    ## P(Y) under MLE is the sample mean
    ## not over docs but labels
    # p_y = 1.*df_train_data.groupby(level='doc').aggregate(sum)/df_train_data['cnt'].sum()
    # p_y.columns = ['p']
    p_y = (1.*df_train_labels['id_lab'].value_counts()/len(df_train_labels)).sort_index().to_frame()
    p_y.columns = ['p']
    p_y['log_p'] = np.log10(p_y) # faster than .map(np.log10)

    ## P(X|Y) under MAP (ie. MLE w/ smoothing), and assuming word prob are positionally independent

    ## setup the sample space first, otherwise some potentials (with priors) will not be assigned a value. The prior is for the sample space, not event space, because a rv is defined on \Omega, not E, where \Omega is the vocabulary across all documents (that we do have).
    lab_word_space = [(id_lab,id_word) for id_lab in df_ng_labels.index for id_word in df_vocab.index]
    df_train_lab_word_cnt = pd.DataFrame(np.zeros(len(lab_word_space)))
    df_train_lab_word_cnt.index = pd.MultiIndex.from_tuples(lab_word_space)
    df_train_lab_word_cnt.index.names = ['id_lab','id_word']
    df_train_lab_word_cnt.columns = ['cnt']

    tmp_df = df_train_data.join(df_train_labels).reset_index()[['id_word','cnt','id_lab']].groupby(['id_lab','id_word']).aggregate(sum) # ie. each label, more abstract than a particular document, has a "true" distribution estimatable from existing, labelled documents. Thus one isn't calculating the distribution of words for a particular document, a summary stat of not much use for classifying new documents.
    df_train_lab_word_cnt['cnt'] = tmp_df['cnt'] # NaN for labels without a count
    df_train_lab_word_cnt.columns = ['numer']
    
    df_train_lab_word_tot_cnt = df_train_lab_word_cnt.groupby(level='id_lab').aggregate(sum)
    df_train_lab_word_tot_cnt.columns = ['denom']

    ## add in the prior
    df_train_lab_word_cnt['numer'] = df_train_lab_word_cnt['numer'].fillna(0) +alpha
    df_train_lab_word_tot_cnt['denom'] += df_train_lab_word_tot_cnt['denom']*alpha

    ## calculate probabilities
    df_train_ps = df_train_lab_word_cnt.join(df_train_lab_word_tot_cnt)
    df_train_ps['p'] = 1.*df_train_ps['numer']/df_train_ps['denom']
    df_train_ps['log_p'] = np.log10(df_train_ps['p'])
    p_x_given_y = df_train_ps
    p_x_given_y_tab = p_x_given_y.reset_index()[['id_lab','id_word','log_p']].pivot_table(columns='id_lab',index='id_word',values='log_p')
    
    return p_y, p_x_given_y_tab


def classify(df_test_data, df_test_labels, p_x_given_y_tab, p_y):
    acc_base_pred = []
    for doc in sorted(set(df_test_data.index.get_level_values('doc'))):
        label_base = int(df_test_labels.loc[doc])
        df_test_data_doc = df_test_data.xs(doc,level='doc')

        # p_y_given_x = p_x_given_y_tab.loc[df_test_data_doc.index] * df_test_data_doc['cnt'] +p_y['log_p'] # wrong (and does not fail)
        p_y_given_x = p_x_given_y_tab.loc[df_test_data_doc.index].apply(lambda col: col*df_test_data_doc['cnt']).sum() +p_y['log_p']
        label_pred = p_y_given_x.argmax()

        acc_base_pred.append( (label_base, label_pred) )

    df_base_pred = pd.DataFrame(acc_base_pred, columns=['base','pred'])

    return df_base_pred


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train-data', help='input file', default='data/hw2/train.data')
    parser.add_argument('--train-label', help='input file', default='data/hw2/train.label')
    parser.add_argument('--test-data', help='input file', default='data/hw2/test.data')
    parser.add_argument('--test-label', help='input file', default='data/hw2/test.label')
    parser.add_argument('--vocabulary', help='input file', default='data/hw2/vocabulary.txt')
    parser.add_argument('--newsgrouplabels', help='input file', default='data/hw2/newsgrouplabels.txt')
    args = parser.parse_args()

    ## read input
    def _index_one(df, name='id'):
        df.index = range(1, len(df) +1)
        df.index.name = name

    df_vocab = pd.read_csv(args.vocabulary, names=['word'])
    _index_one(df_vocab, 'id_word')

    df_ng_labels = pd.read_csv(args.newsgrouplabels, names=['label'])
    _index_one(df_ng_labels, 'id_lab')

    ## normalized tables of (doc,word) with counts
    ## df_ng_labels: docid --> name
    ## df_vocab: wordid --> name
    names_data = ['doc','id_word','cnt']
    idx = ['doc','id_word']
    df_train_data = pd.read_csv(args.train_data, names=names_data, sep=' ').set_index(idx)
    df_test_data = pd.read_csv(args.test_data, names=names_data, sep=' ').set_index(idx)

    df_train_labels = pd.read_csv(args.train_label, names=['id_lab'])
    _index_one(df_train_labels, 'doc')
    df_test_labels = pd.read_csv(args.test_label, names=['id_lab'])
    _index_one(df_test_labels, 'doc')

    size_vocab = len(df_vocab)
    alpha = 1./size_vocab
    p_y, p_x_given_y_tab = train_naive_bayes(df_train_data, df_train_labels, df_vocab, df_ng_labels, alpha)

    df_base_pred = classify(df_test_data, df_test_labels, p_x_given_y_tab, p_y)

    ## stats
    accuracy = 1.*len(df_base_pred[df_base_pred['base'] == df_base_pred['pred']])/len(df_base_pred)
    ## confusion matrix
    df_confuse = pd.DataFrame(confusion_matrix(df_base_pred['base'], df_base_pred['pred']))
    df_confuse.columns = df_ng_labels.index.copy()
    df_confuse.columns.name = 'pred'
    df_confuse.index = df_ng_labels.index.copy()
    df_confuse.index.name = 'base'

    ## number of times base was confused for something else
    tmp_df = df_test_labels['id_lab'].value_counts().to_frame()
    tmp_df.columns = ['cnt_base']
    tmp_df_2 = (df_confuse -np.diag(np.diag(df_confuse))).sum(axis=1).to_frame()
    tmp_df_2.columns = ['cnt_confuse']
    df_cnt_confuse = tmp_df_2.join(df_ng_labels).join(tmp_df)
    df_cnt_confuse['p_confuse'] = 1.*df_cnt_confuse['cnt_confuse']/df_cnt_confuse['cnt_base']

    ## various alphas
    size_vocab = len(df_vocab)
    alpha = 1./size_vocab
    alphas = [alpha,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1]
    accuracy_list = []
    for alpha in alphas:
        p_y, p_x_given_y_tab = train_naive_bayes(df_train_data, df_train_labels, df_vocab, df_ng_labels, alpha)
        df_base_pred = classify(df_test_data, df_test_labels, p_x_given_y_tab, p_y)

        accuracy = 1.*len(df_base_pred[df_base_pred['base'] == df_base_pred['pred']])/len(df_base_pred)
        accuracy_list.append(accuracy)
    print alphas
    print accuracy_list
    df_alphas = pd.DataFrame(zip(alphas, accuracy_list), columns=['alpha','acc'])

    plt.close()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(alphas, accuracy_list, '-', alphas, accuracy_list, 'o', color='k') # line with data points
    ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('test accuracy (%)')
    ax.set_title('Effect of different Dirichlet priors on test accuracy')
    ## format for less visual cluster
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.savefig('img/hw2_-_alpha,test_accuracy.png', format='png')
    # plt.show()
