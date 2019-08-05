def calculate_metrics(y_label, y_pred):
    #for binary classification
    TP,FP,TN,FN = 0,0,0,0
    for i in range(len(y_label)):
        if y_label[i] == y_pred[i]:
            if y_label[i] == 1:
                TP = TP + 1
            else:
                TN = TN + 1
        else:
            if y_pred[i] == 1:
                FP = FP + 1
            else:
                FN = FN + 1
    
    accuracy = (TP + TN) / float(TP + TN + FP + FN)
    sensitivity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)
    precision = TP / float(TP + FP)

    return [accuracy, sensitivity, specificity, precision]
