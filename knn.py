from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
import numpy as np
import json

mpl.rcParams.update({'figure.dpi':150})
data = np.array([[5, 12, 1],
                     [6, 21, 0],
                     [14, 5, 0],
                     [16, 10, 0],
                     [13, 19, 0],
                     [13, 32, 1],
                     [17, 27, 1],
                     [18, 24, 1],
                     [20, 20, 0],
                     [23, 14, 1],
                     [23, 25, 1],
                     [23, 31, 1],
                     [26, 8, 0],
                     [30, 17, 1],
                     [30, 26, 1],
                     [34, 8, 0],
                     [34, 19, 1],
                     [37, 28, 1]])
#x,y,label
class KNN_SPACE_SPLIT:
    def __init__(self,data):
        self.data = data
        self.dim = data.shape[1]-1
    def solve_and_visible_show(self,models_num=2):
        X_train = data[:,0:self.dim]
        y_train = data[:,self.dim]
        models = (KNeighborsClassifier(n_neighbors=1,n_jobs=-1),
                  KNeighborsClassifier(n_neighbors=2,n_jobs=-1))
        models = (clf.fit(X_train,y_train) for clf in models)
        titles = ('K Neighbors with k=1',
                  'K Neighbors with k=2')
        # 设置图形的大小和图间距
        fig = plt.figure(figsize=(15, 5))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        X0, X1 = X_train[:, 0], X_train[:, 1]
        # 得到坐标轴的最小值和最大值
        x_min, x_max = X0.min() - 1, X0.max() + 1
        y_min, y_max = X1.min() - 1, X1.max() + 1
        # 构造网格点坐标矩阵
        # 设置0.2的目的是生成更多的网格点，数值越小，划分空间之间的分隔线越清晰
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                             np.arange(y_min, y_max, 0.2))

        #print(np.c_[xx.ravel(),yy.ravel()])
        for clf,title,ax in zip(models,titles,fig.subplots(1,2).flatten()):
            Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
            Z = Z.reshape(xx.shape)
            # 设置颜色列表
            colors = ["r", 'green', 'lightgreen', 'gray', 'cyan']
            cmap = ListedColormap(colors[:len(np.unique(Z))])
            # 绘制分隔线，contourf函数用于绘制等高线，alpha表示颜色的透明度，一般设置成0.5
            ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)
            # 绘制样本点
            ax.scatter(X0, X1, c=y_train, s=50, edgecolors='k', cmap=cmap, alpha=0.5)
            # （3）根据模型得到的预测结果，计算预测准确率，并设置图形标题
            # 计算预测准确率
            acc = clf.score(X_train, y_train)
            # 设置标题
            ax.set_title(title + ' (Accuracy: %d%%)' % (acc * 100))
        plt.show()
train_data = np.array([[2, 3],
                       [5, 4],
                       [9, 6],
                       [4, 7],
                       [8, 1],
                       [7, 2]])
class KDTree_search:
    def __init__(self,train_data,target,k=1):
        self.data = train_data
        self.target = target
        self.k = k
    def solve(self):
        tree = KDTree(train_data,leaf_size=2)
        dist,ind = tree.query(np.array([self.target]),k=self.k)
        node_index = ind[0]
        x1 = train_data[node_index][0][0]
        x2 = train_data[node_index][0][1]
        print("x点(3,4.5)的最近邻点是({0}, {1})".format(x1, x2))
#KDTree_search(train_data,[3,4.5]).solve()

class KNN_search:
    def __init__(object,X_train):
        object.X_train = X_train

    class Node:
        def __init__(self,value,index,left=None,right=None):
            self.value = value.tolist()
            self.index = index
            self.left = left
            self.right = right
        def __repr__(self):
            return json.dumps(self, indent=3, default=lambda obj: obj.__dict__, ensure_ascii=False, allow_nan=False)
    class KDTree_k:
        def __init__(self,data):
            self.data = np.asarray(data)
            self.kd_tree = None
            self._create_kd_tree(data)
        def _split_sub_tree(self,data,depth=0):
            if len(data) == 0:
                return None
            l = depth % data.shape[1]
            # 对数据进行排序,依据列
            data = data[data[:,l].argsort()]
            # 算法3.2第1步：将所有实例坐标的中位数作为切分点
            median_index = data.shape[0]//2
            # 获取结点在数据集中的位置
            node_index = [i for i,v in enumerate(self.data) if list(v) == list(data[median_index])]
            return KNN_search.Node(value=data[median_index],
                        index = node_index[0],
                        left = self._split_sub_tree(data[:median_index],depth+1),
                        right=self._split_sub_tree(data[median_index+1:],depth+1))

        def _create_kd_tree(self,X):
            self.kd_tree = self._split_sub_tree(X)
        def query(self,data,k=1):
            data = np.asarray(data)
            hits = self._search(data,self.kd_tree,k=k,k_neighbor_sets=list())
            dd = np.array([hit[0] for hit in hits])
            ii = np.array([hit[1] for hit in hits])
            return dd,ii
        def __repr__(self):
            return str(self.kd_tree)
        @staticmethod
        def _cal_node_distance(node1,node2):
            return np.sqrt(np.sum(np.square(node1-node2)))
        def _search(self,point,tree=None,k=1,k_neighbor_sets=None,depth=0):
            n = point.shape[1]
            if k_neighbor_sets is None:
                k_neighbor_sets=[]
            if tree is None:
                return k_neighbor_sets
            # (1)找到包含目标点x的叶结点
            if tree.left is None and tree.right is None:
                # 更新当前k近邻点集
                return self._update_k_neighbor_sets(k_neighbor_sets, k, tree, point)
            if point[0][depth%n] < tree.value[depth%n]:
                direct = 'left'
                next_branch = tree.left
            else:
                direct = 'right'
                next_branch = tree.right
            if next_branch is not None:
                # (3)(b)检查另一子结点对应的区域是否相交
                k_neighbor_sets = self._search(point,tree=next_branch,k=k,depth=depth+1,k_neighbor_sets=k_neighbor_sets)
                # 计算目标点与切分点形成的分割超平面的距离
                temp_dist = abs(tree.value[depth % n] - point[0][depth % n])
                if direct == 'left':
                    if not (k_neighbor_sets[0][0] < temp_dist and len(k_neighbor_sets)==k):
                # 如果相交，递归地进行近邻搜索
                # (3)(a) 判断当前结点，并更新当前k近邻点集
                         k_neighbor_sets = self._update_k_neighbor_sets(k_neighbor_sets, k, tree, point)
                         return self._search(point, tree=tree.right, k=k, depth=depth + 1,
                                        k_neighbor_sets=k_neighbor_sets)
                else:
                    # 判断超球体是否与超平面相交
                    if not (k_neighbor_sets[0][0] < temp_dist and len(k_neighbor_sets) == k):
                        # 如果相交，递归地进行近邻搜索
                        # (3)(a) 判断当前结点，并更新当前k近邻点集
                        k_neighbor_sets = self._update_k_neighbor_sets(k_neighbor_sets, k, tree, point)
                        return self._search(point, tree=tree.left, k=k, depth=depth + 1,
                                            k_neighbor_sets=k_neighbor_sets)
            else:
                return self._update_k_neighbor_sets(k_neighbor_sets,k=k,tree=tree,point=point)
            return k_neighbor_sets


        def _update_k_neighbor_sets(self, best, k, tree, point):
            node_distance = self._cal_node_distance(point,tree.value)
            if len(best) == 0:
                best.append((node_distance,tree.index,node_distance))
            elif len(best) < k:
                self._insert_k_neighbor_sets(best,tree,node_distance)
            else:
                if best[0][0] > node_distance:
                    best = best[1:]
                    self._insert_k_neighbor_sets(best,tree,node_distance)
            return best
        @staticmethod
        def _insert_k_neighbor_sets(best,tree,node_distance):
            n = len(best)
            for i,item in enumerate(best):
                if item[0] < node_distance:
                    best.insert(i,(node_distance,tree.index,tree))
                    break
            if len(best) == n:
                best.append((node_distance,tree.index,tree.value))


        # 打印信息
    @staticmethod
    def print_k_neighbor_sets(k, ii, dd,X_train):
        if k == 1:
            text = "x点的最近邻点是"
        else:
            text = "x点的%d个近邻点是" % k

        for i, index in enumerate(ii):
            res =X_train[index]
            if i == 0:
                text += str(tuple(res))
            else:
                text += ", " + str(tuple(res))

        if k == 1:
            text += "，距离是"
        else:
            text += "，距离分别是"
        for i, dist in enumerate(dd):
            if i == 0:
                text += "%.4f" % dist
            else:
                text += ", %.4f" % dist

        print(text)
X_train = np.array([[2,3,4],
                    [5,4,4],
                    [9, 6, 4],
                    [4, 7, 4],
                    [8, 1, 4],
                    [7, 2, 4]
                    ])
mask = KNN_search(X_train).KDTree_k(X_train)
k = 3
dists,indices = mask.query(np.array([[3,4.5,4]]),k=k)
KNN_search.print_k_neighbor_sets(k,indices,dists,X_train)













