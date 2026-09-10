import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,ConfusionMatrixDisplay,RocCurveDisplay,classification_report

RANDOM_STATE=42

def main():
    data=load_breast_cancer()
    X=pd.DataFrame(data.data,columns=data.feature_names)
    y=pd.Series(data.target,name="diagnosis")

    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=RANDOM_STATE,stratify=y)

    model=Pipeline([("scale",StandardScaler()),
                    ("model",LogisticRegression(max_iter=3000,random_state=RANDOM_STATE))])
    model.fit(Xtr,ytr)
    pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]

    print("\nDISEASE PREDICTION MODEL")
    print("-"*32)
    print("Samples :",len(X)," Features:",X.shape[1])
    for name,val in [("Accuracy",accuracy_score(yte,pred)),("Precision",precision_score(yte,pred)),
                     ("Recall",recall_score(yte,pred)),("F1 Score",f1_score(yte,pred)),
                     ("ROC-AUC",roc_auc_score(yte,prob))]:
        print(f"{name:10}: {val:.3f}")
    print("\n",classification_report(yte,pred,target_names=data.target_names))

    export=X.copy()
    export["diagnosis"]=[data.target_names[v] for v in y]
    export.to_csv("../breast_cancer_dataset.csv",index=False)

    ConfusionMatrixDisplay(confusion_matrix(yte,pred),display_labels=data.target_names).plot()
    plt.title("Disease Prediction - Confusion Matrix"); plt.tight_layout()
    plt.savefig("../outputs/confusion_matrix.png",dpi=160); plt.close()

    RocCurveDisplay.from_predictions(yte,prob)
    plt.title("Disease Prediction - ROC Curve"); plt.tight_layout()
    plt.savefig("../outputs/roc_curve.png",dpi=160); plt.close()
    print("\nDataset and evaluation graphs saved.")

if __name__=="__main__":
    main()
