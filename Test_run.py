import argparse
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from pandas.core.frame import DataFrame
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

parser = argparse.ArgumentParser()
parser.add_argument("-I","--input_sample", required = True, help="input sample data full path including file name")
parser.add_argument("-O","--output_path",required=True, help="output file full path including  file name")
parser.add_argument("-M","--method", required=True, help="Machine learning method")
args = parser.parse_args()

input_file = args.input_sample
train_df = pd.read_csv(input_file,index_col = 0)
#split train and test data
test_sample = "unknown"
train_rows = [m for m in train_df._stat_axis if m not in [test_sample]]
train_df_1 = train_df.loc[train_rows,]
test_df_1 = train_df.loc[test_sample,]
target = "type"
x_columns = [x for x in train_df_1.columns if x not in [target]]
train_selected_data = train_df_1[x_columns]
train_selected_y = train_df_1[target]
test_selected_data = test_df_1[x_columns]

X = train_selected_data
y = train_selected_y.values.ravel()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, stratify= y, shuffle=True, random_state = 1)

model_smote = SMOTE()
x_smote,y_smote = model_smote.fit_resample(X_train,y_train)
X_train = x_smote
y_train = y_smote
x_valid_smote,y_valid_smote = model_smote.fit_resample(X_valid,y_valid)
X_valid = x_valid_smote
y_valid = y_valid_smote

if args.method == "svm":
    def hyperopt_train_test(params):
        clf = SVC(**params)
        return cross_val_score(clf, X_train, y_train).mean()

    search_space = hp.choice("classifier_type", [
        {"C": hp.uniform("C", 0, 10.0), "kernel": hp.choice("kernel", ["linear", "rbf"]),
         "gamma": hp.uniform("gamma", 0, 20.0)}
    ])

    count = 0
    best = 0
    def func(params):
        global best, count
        count += 1
        acc = hyperopt_train_test(params.copy())
        if acc > best:
            print("new best: %f, using: %s" % (acc, params))
            print("-" * 100)
            best = acc
        if count % 10 == 0:
            print("iters: %d, acc: %f, using: %s" % (count, acc, params))
        return ({"loss": -acc, "status": STATUS_OK})

    trials = Trials()
    best = fmin(func, search_space, algo=tpe.suggest, max_evals=50, trials=trials)
    print("=" * 80)
    print("best: ", best)

    kernel = ["linear", "rbf"]
    model = SVC(C=best["C"], kernel=kernel[best["kernel"]], gamma=best["gamma"])
    model.fit(X_train, y_train)
    model_clf = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    model_clf.fit(X_valid, y_valid)

    test_y_pred = model_clf.predict(test_selected_data.values.reshape(1, -1))
    test_y_pred_prob = model_clf.predict_proba(test_selected_data.values.reshape(1, -1))
    result_df = DataFrame([test_y_pred, test_y_pred_prob.ravel()])
    result_df.index = ["predicted_cancer_type", "predicted_probability"]
    result_df.columns = ["breast", "cervical", "esophagus", "head", "intestinal", "liver", "lung", "ovarian",
                         "stomach", "thyroid"]
    result_df.to_csv(args.output_path)

elif args.method == "lgr":
    def hyperopt_train_test(params):
        clf = clf = LogisticRegression(**params)
        return cross_val_score(clf, X_train, y_train).mean()

    search_space = hp.choice("classifier_type", [
        {"penalty": hp.choice("lgr_penalty", ["l1", "l2", "elasticnet", "none"]), "C": hp.uniform("lgr_C", 0, 10.0),
         "solver": hp.choice("lgr_solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])}
    ])

    count = 0
    best = 0
    def func(params):
        global best, count
        count += 1
        acc = hyperopt_train_test(params.copy())
        if acc > best:
            print("new best: %f, using: %s" % (acc, params))
            print("-" * 100)
            best = acc
        if count % 10 == 0:
            print("iters: %d, acc: %f, using: %s" % (count, acc, params))
        return ({"loss": -acc, "status": STATUS_OK})

    trials = Trials()
    best = fmin(func, search_space, algo=tpe.suggest, max_evals=50, trials=trials)
    print("=" * 80)
    print("best: ", best)

    lgr_penalty_type = ["l1", "l2", "elasticnet", "none"]
    lgr_solver_type = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    model = LogisticRegression(penalty=lgr_penalty_type[best["lgr_penalty"]], C=best["lgr_C"],
                               solver=lgr_solver_type[best["lgr_solver"]])
    model.fit(X_train, y_train)
    model_clf = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    model_clf.fit(X_valid, y_valid)

    test_y_pred = model_clf.predict(test_selected_data.values.reshape(1, -1))
    test_y_pred_prob = model_clf.predict_proba(test_selected_data.values.reshape(1, -1))
    result_df = DataFrame([test_y_pred, test_y_pred_prob.ravel()])
    result_df.index = ["predicted_cancer_type", "predicted_probability"]
    result_df.columns = ["breast", "cervical", "esophagus", "head", "intestinal", "liver", "lung", "ovarian",
                         "stomach", "thyroid"]
    result_df.to_csv(args.output_path)

elif args.method == "linearsvc":
    def hyperopt_train_test(params):
        clf = LinearSVC(**params)
        return cross_val_score(clf, X_train, y_train).mean()

    search_space = hp.choice("classifier_type", [
        {"penalty": hp.choice("penalty", ["l1","l2"]),
         "loss": hp.choice("loss", ["hinge", "squared_hinge"]),
         "tol": hp.uniform("tol", 0.00001, 0.001),
         "C": hp.uniform("C", 0.001, 1.0),
         "multi_class": hp.choice("multi_class", ['ovr', 'crammer_singer']),
         "max_iter": hp.randint("max_iter", 10000)}
    ])

    count = 0
    best = 0
    def func(params):
        global best, count
        count += 1
        acc = hyperopt_train_test(params.copy())
        if acc > best:
            print("new best: %f, using: %s" % (acc, params))
            print("-" * 100)
            best = acc
        if count % 10 == 0:
            print("iters: %d, acc: %f, using: %s" % (count, acc, params))
        return ({"loss": -acc, "status": STATUS_OK})

    trials = Trials()
    best = fmin(func, search_space, algo=tpe.suggest, max_evals=50, trials=trials)
    print("=" * 80)
    print("best: ", best)

    penalty = ["l1","l2"]
    loss = ["hinge", "squared_hinge"]
    multi_class = ['ovr', 'crammer_singer']

    model = LinearSVC(penalty=penalty[best["penalty"]], loss=loss[best["loss"]], tol=best["tol"],
                      C=best["C"], multi_class=multi_class[best["multi_class"]], max_iter=best["max_iter"])

    model.fit(X_train, y_train)
    model_clf = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    model_clf.fit(X_valid, y_valid)

    test_y_pred = model_clf.predict(test_selected_data.values.reshape(1, -1))
    test_y_pred_prob = model_clf.predict_proba(test_selected_data.values.reshape(1, -1))
    result_df = DataFrame([test_y_pred, test_y_pred_prob.ravel()])
    result_df.index = ["predicted_cancer_type", "predicted_probability"]
    result_df.columns = ["breast", "cervical", "esophagus", "head", "intestinal", "liver", "lung", "ovarian",
                         "stomach", "thyroid"]
    result_df.to_csv(args.output_path)

elif args.method == "randomforest":
    def hyperopt_train_test(params):
        clf = RandomForestClassifier(**params)
        return cross_val_score(clf, X_train, y_train).mean()

    search_space = hp.choice("classifier_type", [
        {"max_depth": hp.choice("max_depth", range(1, 20)),
         "max_features": hp.choice("max_features", range(1, 5)),
         "n_estimators": hp.choice("n_estimators", range(1, 1000)),
         "criterion": hp.choice("criterion", ["gini", "entropy"])}
    ])

    count = 0
    best = 0
    def func(params):
        global best, count
        count += 1
        acc = hyperopt_train_test(params.copy())
        if acc > best:
            print("new best: %f, using: %s" % (acc, params))
            print("-" * 100)
            best = acc
        if count % 10 == 0:
            print("iters: %d, acc: %f, using: %s" % (count, acc, params))
        return ({"loss": -acc, "status": STATUS_OK})

    trials = Trials()
    best = fmin(func, search_space, algo=tpe.suggest, max_evals=50, trials=trials)
    print("=" * 80)
    print("best: ", best)

    criterion_type = ["gini", "entropy"]
    model = RandomForestClassifier(max_depth=best["max_depth"], max_features=best["max_features"],
                                   n_estimators=best["n_estimators"], criterion=criterion_type[best["criterion"]])
    model.fit(X_train, y_train)
    model_clf = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    model_clf.fit(X_valid, y_valid)

    test_y_pred = model_clf.predict(test_selected_data.values.reshape(1, -1))
    test_y_pred_prob = model_clf.predict_proba(test_selected_data.values.reshape(1, -1))
    result_df = DataFrame([test_y_pred, test_y_pred_prob.ravel()])
    result_df.index = ["predicted_cancer_type", "predicted_probability"]
    result_df.columns = ["breast", "cervical", "esophagus", "head", "intestinal", "liver", "lung", "ovarian",
                         "stomach", "thyroid"]
    result_df.to_csv(args.output_path)

elif args.method == "knn":
    def hyperopt_train_test(params):
        clf = KNeighborsClassifier(**params)
        return cross_val_score(clf, X_train, y_train).mean()

    search_space = hp.choice("classifier_type", [
        {"n_neighbors":hp.choice("n_neighbors",range(2,50)),
        "algorithm": hp.choice("algorithm", ["auto", "ball_tree", "kd_tree", "brute"]),
        "weights":hp.choice("weights",["uniform", "distance"])
        }
    ])

    count = 0
    best = 0
    def func(params):
        global best, count
        count += 1
        acc = hyperopt_train_test(params.copy())
        if acc > best:
            print("new best: %f, using: %s" % (acc, params))
            print("-" * 100)
            best = acc
        if count % 10 == 0:
            print("iters: %d, acc: %f, using: %s" % (count, acc, params))
        return ({"loss": -acc, "status": STATUS_OK})

    trials = Trials()
    best = fmin(func, search_space, algo=tpe.suggest, max_evals=50, trials=trials)
    print("=" * 80)
    print("best: ", best)

    algorithm_type = ["auto", "ball_tree", "kd_tree", "brute"]
    weight_type = ["uniform", "distance"]

    model = KNeighborsClassifier(n_neighbors=best["n_neighbors"],algorithm = algorithm_type[best["algorithm"]],weights = weight_type[best["weights"]])
    model.fit(X_train, y_train)
    model_clf = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    model_clf.fit(X_valid, y_valid)

    test_y_pred = model_clf.predict(test_selected_data.values.reshape(1, -1))
    test_y_pred_prob = model_clf.predict_proba(test_selected_data.values.reshape(1, -1))
    result_df = DataFrame([test_y_pred, test_y_pred_prob.ravel()])
    result_df.index = ["predicted_cancer_type", "predicted_probability"]
    result_df.columns = ["breast", "cervical", "esophagus", "head", "intestinal", "liver", "lung", "ovarian",
                         "stomach", "thyroid"]
    result_df.to_csv(args.output_path)

elif args.method == "adaboost":
    def hyperopt_train_test(params):
        clf = AdaBoostClassifier(**params)
        return cross_val_score(clf, X_train, y_train).mean()


    search_space = hp.choice("classifier_type", [
        {"n_estimators": hp.choice("n_estimators", range(1, 100)),
         "learning_rate": hp.uniform("learning_rate", 0.000001, 1),
         "algorithm": hp.choice("algorithm", ["SAMME", "SAMME.R"])}
    ])

    count = 0
    best = 0


    def func(params):
        global best, count
        count += 1
        acc = hyperopt_train_test(params.copy())
        if acc > best:
            print("new best: %f, using: %s" % (acc, params))
            print("-" * 100)
            best = acc
        if count % 10 == 0:
            print("iters: %d, acc: %f, using: %s" % (count, acc, params))
        return ({"loss": -acc, "status": STATUS_OK})


    trials = Trials()
    best = fmin(func, search_space, algo=tpe.suggest, max_evals=50, trials=trials)
    print("=" * 80)
    print("best: ", best)

    algorithm_type = ["SAMME", "SAMME.R"]
    model = AdaBoostClassifier(n_estimators=best["n_estimators"], learning_rate=best["learning_rate"],
                                algorithm=algorithm_type[best["algorithm"]])
    model.fit(X_train, y_train)
    model_clf = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    model_clf.fit(X_valid, y_valid)

    test_y_pred = model_clf.predict(test_selected_data.values.reshape(1, -1))
    test_y_pred_prob = model_clf.predict_proba(test_selected_data.values.reshape(1, -1))
    result_df = DataFrame([test_y_pred, test_y_pred_prob.ravel()])
    result_df.index = ["predicted_cancer_type", "predicted_probability"]
    result_df.columns = ["breast", "cervical", "esophagus", "head", "intestinal", "liver", "lung", "ovarian",
                         "stomach", "thyroid"]
    result_df.to_csv(args.output_path)

elif args.method == "nb":
    def hyperopt_train_test(params):
        clf = BernoulliNB(**params)
        return cross_val_score(clf, X_train, y_train).mean()


    search_space = hp.choice("classifier_type", [
        {"alpha":hp.uniform("alpha",0.0,2.0)}
    ])

    count = 0
    best = 0


    def func(params):
        global best, count
        count += 1
        acc = hyperopt_train_test(params.copy())
        if acc > best:
            print("new best: %f, using: %s" % (acc, params))
            print("-" * 100)
            best = acc
        if count % 10 == 0:
            print("iters: %d, acc: %f, using: %s" % (count, acc, params))
        return ({"loss": -acc, "status": STATUS_OK})


    trials = Trials()
    best = fmin(func, search_space, algo=tpe.suggest, max_evals=50, trials=trials)
    print("=" * 80)
    print("best: ", best)

    model = BernoulliNB(alpha = best["alpha"])
    model.fit(X_train, y_train)
    model_clf = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    model_clf.fit(X_valid, y_valid)

    test_y_pred = model_clf.predict(test_selected_data.values.reshape(1, -1))
    test_y_pred_prob = model_clf.predict_proba(test_selected_data.values.reshape(1, -1))
    result_df = DataFrame([test_y_pred, test_y_pred_prob.ravel()])
    result_df.index = ["predicted_cancer_type", "predicted_probability"]
    result_df.columns = ["breast", "cervical", "esophagus", "head", "intestinal", "liver", "lung", "ovarian",
                         "stomach", "thyroid"]
    result_df.to_csv(args.output_path)


