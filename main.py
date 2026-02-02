from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from model import lr, rf, scaler, X_test, y_test, feature_names
import numpy as np
import pandas as pd
import time

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    show_graph = False
    cm_file = roc_file = fi_file = dist_file = None

    selected_model = request.form.get("model", "rf")
    model = rf if selected_model == "rf" else lr

    if request.method == "POST":
        show_graph = True
        ts = int(time.time())

        input_data = [
            float(request.form["Pendapatan"]),
            int(request.form["Jumlah_Tanggungan"]),
            int(request.form["Kondisi_Rumah"]),
            int(request.form["Status_Pekerjaan"]),
            int(request.form["Kepemilikan_Aset"])
        ]

        input_scaled = scaler.transform([input_data])
        pred = model.predict(input_scaled)[0]
        result = "LAYAK MENERIMA BANTUAN" if pred == 1 else "TIDAK LAYAK"

        # Confusion Matrix
        cm_file = f"cm_{ts}.png"
        plt.figure(figsize=(4,4))
        sns.heatmap(
            confusion_matrix(y_test, model.predict(X_test)),
            annot=True, fmt="d",
            cmap="Blues" if selected_model=="lr" else "Greens"
        )
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"static/{cm_file}")
        plt.close()

        # ROC Curve
        roc_file = f"roc_{ts}.png"
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = auc(fpr, tpr)

        plt.figure(figsize=(4,4))
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0,1],[0,1],'--', alpha=0.6)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (AUC = {auc_score:.2f})")
        plt.tight_layout()
        plt.savefig(f"static/{roc_file}")
        plt.close()

        # Feature Importance (RF only)
        if selected_model == "rf":
            fi_file = f"fi_{ts}.png"
            fi = pd.Series(
                rf.feature_importances_,
                index=feature_names
            ).sort_values()

            plt.figure(figsize=(5,4))
            fi.plot(kind="barh", color="#4fd1c5")
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.savefig(f"static/{fi_file}")
            plt.close()

        # Distribution Plot
        dist_file = f"dist_{ts}.png"
        plt.figure(figsize=(4,4))
        sns.countplot(x=y_test, palette="muted")
        plt.title("Distribusi Kelas Target")
        plt.tight_layout()
        plt.savefig(f"static/{dist_file}")
        plt.close()

    return render_template(
        "index.html",
        result=result,
        show_graph=show_graph,
        cm_file=cm_file,
        roc_file=roc_file,
        fi_file=fi_file,
        dist_file=dist_file,
        selected_model=selected_model
    )

if __name__ == "__main__":
    app.run(debug=True)
