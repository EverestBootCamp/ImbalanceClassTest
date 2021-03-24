class Test():

    def __init__(self, **hyperparams, scaled_X_train, scaled_X_test, y_train):
        clf = svm.SVC(self.hyperparams, class_weight='balanced', \
                      random_state=0)
        clf.fit(scaled_X_train, y_train)
        pred = clf.predict(scaled_X_test)

    def eval(self):
        # Compute confusion matrix to evaluate the accuracy of a classification
        print("confusion_matrix:")
        print(confusion_matrix(y_test, pred))

        # Compute Area Under the Receiver Operating Characteristic Curve
        # (ROC AUC) from prediction scores.
        print("roc_auc_score:")
        print(roc_auc_score(y_test, pred))

        # Build a text report showing the main classification metrics
        # (which contains the F1 Score).
        print("classification_report:")
        print(classification_report(y_test, pred))

        # Computes the Cohen Kappa Score
        print("Cohen Kappa Score:")
        print(cohen_kappa_score(y_test, pred))

        # Average Precision Score
        average_precision = average_precision_score(y_test, y_score)
        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        # Precision-Recall Curve
        disp = plot_precision_recall_curve(clf, scaled_X_test, y_test)
        disp.ax_.set_title('2-class Precision-Recall curve: '
                           'AP={0:0.2f}'.format(average_precision))
        # ROC Curve
        svc_disp = plot_roc_curve(clf, scaled_X_test, y_test)
        svc_disp.ax_.set_title('ROC curve')
        plt.show()


