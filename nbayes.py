import pandas as pd
import numpy as np

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

    # TODO: replace file_path with argparse vars
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

    ## P(Y) under MLE is the sample mean
    ## not over docs but labels
    # p_y = 1.*df_train_data.groupby(level='doc').aggregate(sum)/df_train_data['cnt'].sum()
    # p_y.columns = ['p']
    p_y = (1.*df_train_labels['id_lab'].value_counts()/len(df_train_labels)).sort_index().to_frame()
    p_y.columns = ['p']
    p_y['log_p'] = np.log10(p_y) # faster than .map(np.log10)

    ## P(X|Y) under MAP (ie. MLE w/ smoothing), and assuming word prob are positionally independent
    size_vocab = len(df_vocab)
    alpha = 1./size_vocab

    ## setup the possible space first, otherwise some potentials (with priors) will not have a value
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

    ## classify
    acc_base_pred = []
    for doc in sorted(set(df_test_data.index.get_level_values('doc'))):
        label_base = int(df_test_labels.loc[doc])
        df_test_data_doc = df_test_data.xs(doc,level='doc')

        # p_y_given_x = p_x_given_y_tab.loc[df_test_data_doc.index] * df_test_data_doc['cnt'] +p_y['log_p'] # wrong (and does not fail)
        p_y_given_x = p_x_given_y_tab.loc[df_test_data_doc.index].apply(lambda col: col*df_test_data_doc['cnt']).sum() +p_y['log_p']
        ## todo: fill na with \alpha?
        label_pred = p_y_given_x.argmax()

        acc_base_pred.append( (label_base, label_pred) )

    df_base_pred = pd.DataFrame(acc_base_pred, columns=['base','pred'])
    accuracy = 1.*len(df_base_pred[df_base_pred['base'] == df_base_pred['pred']])/len(df_base_pred)
    print 'naive bayes testing accuracy: {}'.format(accuracy)
