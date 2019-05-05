import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, [])
train_data = np.delete(iris.data, [], axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print test_target
print clf.predict(test_data)

# viz code
from sklearn.externals.six import StringIO
import pydot

dotfile = StringIO()

tree.export_graphviz(clf, out_file=dotfile, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=False, impurity=False)

(graph,)=pydot.graph_from_dot_data(dotfile.getvalue())
graph.write_pdf("iris.pdf")
