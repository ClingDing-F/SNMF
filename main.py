import snmf as snmf
from utils import generate_2_class_data
from sklearn.neighbors import KNeighborsClassifier



if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test = generate_2_class_data(data_num=135, dim=512, bias=0.2)

    snmf_net = snmf.SNMF(rank=2, max_iters=2000, mu=1e-14, eps=1e-6, lamda=1, cstab=1e-9, alpha=0.8, output=True)
    snmf_net.fit(Data_matrix=X_train, label=Y_train)
    V_test = snmf_net.transform(Data_matrix=X_test)
    knn3 = KNeighborsClassifier(n_neighbors=10, weights="distance", metric='euclidean')
    knn3.fit(snmf_net.X_trained_feature, Y_train)
    snmf_acc = knn3.score(V_test, Y_test)
    print(snmf_acc)
