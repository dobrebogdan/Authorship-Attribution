import numpy as np
from math import sqrt, fabs
from scipy.stats import kendalltau
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import nltk


def pearson_coefficient(X, Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    x_dev = np.std(X)
    y_dev = np.std(Y)
    r = 0.0
    for i in range(0, len(X)):
        r = r + ((X[i] - x_mean) / x_dev) * ((Y[i] - y_mean) / y_dev)
    r = r / (len(X))
    return fabs(r)


def euclidean_distance(X, Y):
    r = pearson_coefficient(X, Y)
    return sqrt(2*(1-r))


def spearman_rank_order(X, Y):
    sqsum = 0.0
    n = 1.0 * len(X)
    for i in range(0, len(X)):
        sqsum += (X[i] - Y[i]) * (X[i] - Y[i])
    r = 1.0 - (6.0 / (n * (n * n - 1.0))) * sqsum
    return fabs(r)


def spearman_footrule(X, Y):
    sqsum = 0.0
    n = 1.0 * len(X)
    for i in range(0, len(X)):
        sqsum += fabs(X[i] - Y[i])
    r = 1.0 - 3.0 / (n * n - 1.0) * sqsum
    return r


def get_PQ(X, Y):
    P = 0.0
    Q = 0.0
    for i in range(0, len(X)):
        for j in range(i + 1, len(X)):
            prod = (X[i] - X[j]) * (Y[i] - Y[j])
            if prod > 0.0:
                P += 1
            elif prod < 0.0:
                Q += 1
    return (P, Q)


def gamma(X, Y):
    (P, Q) = get_PQ(X, Y)
    return fabs((P - Q) / (P + Q))


# need to subtract result from 1 for clustering
def tau(X, Y):
    return (1.0 - kendalltau(X,Y)[0])


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def create_dendrogram(X, labels_list):
    # metric cam be replaced with any other of the ones defined above
    dists = pairwise_distances(X, X, metric=tau)

    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='complete')

    model = model.fit(dists)
    # print("Distances are: ")
    # print(dists)

    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode='level', p=12, labels=labels_list, orientation='left')
    plt.xlabel("")
    plt.show()


def get_words_from_files():
    titles_list = ["sense_and_sensibility", "pride_and_prejudice", "1984", "homage_to_catalonia", "dubliners",
                   "artist_portrait", "felix_holt", "middlemarch", "jane_eyre", "shirley", "wuthering_heights",
                   "agnes_gray", "tenant", "harry_potter_1", "harry_potter_2", "harry_potter_3", "harry_potter_4",
                   "the_two_towers", "fellowship_of_the_rings", "the_return_of_the_king"]
    words = {}
    for filename in titles_list:
        with open(filename, "r") as f:
            text = f.read()
            tokens = nltk.word_tokenize(text)

            for token in tokens:
                if token in words.keys():
                    words[token] = words[token] + 1
                else:
                    if len(token) > 1 and token[0].islower():
                        words[token] = 1
    return words

def get_words_from_file(filename):
    with open(filename, "r") as f:
        text = f.read()
        tokens = nltk.word_tokenize(text)
        words = {}
        for token in tokens:
            if token in words.keys():
                words[token] = words[token] + 1
            else:
                if len(token) > 1 and token[0].islower():
                    words[token] = 1
        return words


def get_function_word_rankings():
    function_words = []
    function_words_ranking = {}
    words = get_words_from_files()

    words_list = []
    for (word, cnt) in words.items():
        words_list.append((cnt, word))
    words_count = sorted(words_list, reverse=True)
    # 13
    top_words_counts = words_count[0:13]
    rank = 1
    for (_, word) in top_words_counts:
        function_words.append(word)
        function_words_ranking[word] = rank
        rank += 1
    return function_words, function_words_ranking


(function_words, function_words_ranking) = get_function_word_rankings()

X = []
# 2 works of each of the following: Jane Austen, George Orwell, James Joycew, George Eliot,
# Charlotte Bronte and 1 work for Emily Bronte and 2 for Anne Bronte, 4 for JK Rowling and 3 for John R. R. Tolkie
titles_list = ["sense_and_sensibility", "pride_and_prejudice", "1984", "homage_to_catalonia", "dubliners",
               "artist_portrait", "felix_holt", "middlemarch", "jane_eyre", "shirley", "wuthering_heights",
               "agnes_gray", "tenant", "harry_potter_1", "harry_potter_2", "harry_potter_3", "harry_potter_4",
               "the_two_towers", "fellowship_of_the_rings", "the_return_of_the_king"]

labels_list = ["Jane Austen 1", "Jane Austen 2", "George Orwell 1", "George Orwell 2", "James Joycew 1",
               "James Joycew 2", "George Eliot 1", "George Eliot 2", "Charlotte Bronte 1", "Charlotte Bronte 2",
               "Emily Bronte", "Anne Bronte 1", "Anne Bronte 2", "JK Rowling 1", "JK Rowling 2", "JK Rowling 3",
               "JK Rowling 4", "John R. R. Tolkie 1", "John R. R. Tolkie 2", "John R. R. Tolkie 3"]

for filename in titles_list:
    words = get_words_from_file(filename)
    # print(filename)
    # print(len(words))
    words_list = []
    for (word, cnt) in words.items():
        words_list.append((cnt, word))
    words_count = sorted(words_list, reverse=True)

    # print(words_count)
    # compute the function words if not already existing

    curr_ranking = [0] * len(function_words)
    curr_word_rank = 0.0
    for (_, word) in words_count:
        if word in function_words:
            curr_word_rank += 1
            poz = int(function_words_ranking[word] - 1)
            curr_ranking[poz] = curr_word_rank
    # print(filename)
    # print(curr_ranking)
    X.append(curr_ranking)

create_dendrogram(np.array(X), labels_list)
